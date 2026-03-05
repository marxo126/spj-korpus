"""MediaPipe pose extraction pipeline (Tasks API, mediapipe >= 0.10.18).

Captures per frame:
  - 33 body pose landmarks   (PoseLandmarker)
  - 21 left-hand landmarks   (HandLandmarker — left)
  - 21 right-hand landmarks  (HandLandmarker — right)
  - 468 face mesh landmarks  (FaceLandmarker)

Total: 543 landmarks × 4 channels (x, y, z, visibility/confidence).

Parallelism:
  - extract_pose()       — single video, one GPU pipeline
  - extract_pose_batch() — N videos in parallel, each with its own GPU pipeline
                           worker count auto-scales to free RAM / GPU headroom

Model files (~30–150 MB each) are downloaded automatically to models/
on first run.
"""
from __future__ import annotations

import json
import logging
import queue
import subprocess
import threading
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Recycle GPU detectors every N frames to flush Metal memory leak
# (GitHub: mediapipe #5652, #6223). ~1s overhead per recycle.
_GPU_RECYCLE_INTERVAL = 2000

# ---------------------------------------------------------------------------
# Model download URLs (Google MediaPipe model garden)
# ---------------------------------------------------------------------------
_MODEL_DIR = Path(__file__).parent.parent.parent / "models"

_MODELS = {
    "pose": (
        "pose_landmarker_full.task",
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    ),
    "hand": (
        "hand_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    ),
    "face": (
        "face_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    ),
}


def ensure_models(model_dir: Path | None = None) -> dict[str, Path]:
    """Download model .task files if not already present."""
    d = Path(model_dir) if model_dir else _MODEL_DIR
    d.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, (filename, url) in _MODELS.items():
        dest = d / filename
        if not dest.exists():
            logger.info(f"Downloading {filename} …")
            urllib.request.urlretrieve(url, dest)
            logger.info(f"Saved {filename} ({dest.stat().st_size // 1_048_576} MB)")
        paths[key] = dest
    return paths


# ---------------------------------------------------------------------------
# Resource auto-scaling
# ---------------------------------------------------------------------------

def recommend_workers(n_videos: int | None = None) -> int:
    """Return the recommended number of parallel workers for this machine.

    Uses available RAM to estimate how many simultaneous GPU pipelines
    we can sustain without memory pressure.

    Each worker needs roughly:
      - ~500 MB for three MediaPipe model graphs on GPU
      - Frame buffer: queue_size × frame_bytes (computed separately)
      - ~200 MB overhead (numpy, Python, result arrays)

    Args:
        n_videos: Cap workers to this if supplied (no point spinning up
                  more workers than there are videos).

    Returns:
        Recommended worker count (at least 1).
    """
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / 1_073_741_824
    except ImportError:
        available_gb = 8.0  # conservative fallback

    # Empirical: ~1.5 GB per worker comfortably fits models + buffers
    workers = max(1, int(available_gb / 1.5))
    # Hard cap — beyond 6 the Metal scheduler overhead outweighs the gain
    workers = min(workers, 6)
    if n_videos is not None:
        workers = min(workers, n_videos)

    logger.info(f"recommend_workers → {workers}  (free RAM: {available_gb:.1f} GB)")
    return workers


def _compute_queue_size(frame_w: int, frame_h: int, n_workers: int) -> int:
    """Compute per-worker prefetch queue size from available RAM.

    Allocates up to 25 % of free RAM across all workers' frame queues.
    """
    try:
        import psutil
        available_mb = psutil.virtual_memory().available / 1_048_576
    except ImportError:
        available_mb = 4_000.0

    frame_mb = (frame_w * frame_h * 4) / 1_048_576  # RGBA bytes → MB
    budget_per_worker_mb = (available_mb * 0.25) / max(n_workers, 1)
    size = int(budget_per_worker_mb / frame_mb) if frame_mb > 0 else 64
    size = max(8, min(size, 512))  # floor=8, ceiling=512
    logger.info(
        f"Queue: {size} frames × {frame_mb:.2f} MB = {size * frame_mb:.0f} MB/worker "
        f"(free: {available_mb:.0f} MB, workers: {n_workers})"
    )
    return size


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _make_progress_writer(
    progress_file_str: str, throttle: int = 10,
) -> Callable[[float], None]:
    """Return a progress callback that writes a fraction to a temp file.

    Writes only every `throttle` calls to reduce disk I/O (default: every 10th).
    Always writes when frac >= 1.0 (completion).
    """
    _count = [0]  # mutable counter (closure-safe)

    def _write(frac: float):
        if not progress_file_str:
            return
        _count[0] += 1
        if frac < 1.0 and _count[0] % throttle != 0:
            return
        try:
            Path(progress_file_str).write_text(str(round(frac, 4)))
        except Exception:
            pass
    return _write


def _recycle_detectors(
    pose_det, hand_det, face_det, model_paths: dict[str, Path],
):
    """Close and recreate GPU detectors to flush Metal memory leak."""
    pose_det.close()
    hand_det.close()
    face_det.close()
    return _build_detectors(model_paths, use_gpu=True)


def extract_pose(
    video_path: Path,
    output_path: Path,
    progress_callback: Optional[Callable[[float], None]] = None,
    model_dir: Path | None = None,
    n_concurrent: int = 1,
    use_gpu: bool | None = None,
    crop: tuple[int, int, int, int] | None = None,
) -> dict:
    """Extract pose landmarks from a single video.

    Args:
        video_path: Input video file.
        output_path: Destination .pose file.
        progress_callback: Called with float in [0, 1] as frames are processed.
        model_dir: Override model storage directory.
        n_concurrent: How many parallel extract_pose calls are running right now
                      (used to size the per-worker frame buffer from free RAM).
        crop: Optional (x, y, w, h) rectangle to crop each frame before
              processing. Useful for dual-view videos where each half contains
              a separate signer.

    Returns:
        Dict: frames, duration_sec, output_size_bytes, device.
    """
    import cv2

    video_path = Path(video_path)
    output_path = Path(output_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_paths = ensure_models(model_dir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080

    _, total_frames = _probe_video_info(video_path, fps)
    # Use cropped dimensions for queue sizing if crop is set
    effective_w = crop[2] if crop else frame_w
    effective_h = crop[3] if crop else frame_h
    queue_size = _compute_queue_size(effective_w, effective_h, n_concurrent)

    # GPU safe only when one pipeline at a time — concurrent Metal CVPixelBuffer
    # allocation fails with -6662 (pool exhausted) when n_concurrent > 1
    if use_gpu is None:
        use_gpu = (n_concurrent == 1)
    pose_det, hand_det, face_det, device = _build_detectors(model_paths, use_gpu=use_gpu)
    logger.info(f"[{video_path.name}] device={device} frames≈{total_frames} queue={queue_size}"
                + (f" crop={crop}" if crop else ""))

    frame_q: queue.Queue = queue.Queue(maxsize=queue_size)

    def _reader():
        idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if crop:
                    cx, cy, cw, ch = crop
                    frame = frame[cy:cy + ch, cx:cx + cw]
                rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                frame_q.put((rgba, int(idx * 1000 / fps)))
                idx += 1
        finally:
            frame_q.put(None)

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    pose_frames: list[np.ndarray] = []
    processed = 0
    frames_since_recycle = 0

    try:
        while True:
            item = frame_q.get()
            if item is None:
                break
            rgba, timestamp_ms = item
            lm = _process_frame(rgba, timestamp_ms, pose_det, hand_det, face_det)
            pose_frames.append(lm)
            processed += 1
            frames_since_recycle += 1

            if progress_callback is not None and total_frames > 0:
                progress_callback(min(processed / total_frames, 0.99))

            if use_gpu and frames_since_recycle >= _GPU_RECYCLE_INTERVAL:
                pose_det, hand_det, face_det, device = _recycle_detectors(
                    pose_det, hand_det, face_det, model_paths)
                frames_since_recycle = 0
    finally:
        reader_thread.join(timeout=5)
        cap.release()
        pose_det.close()
        hand_det.close()
        face_det.close()

    duration_sec = processed / fps if fps > 0 else 0.0
    _save_pose(pose_frames, fps, output_path)

    if progress_callback is not None:
        progress_callback(1.0)

    output_size = output_path.stat().st_size if output_path.exists() else 0

    return {
        "frames": processed,
        "duration_sec": round(duration_sec, 2),
        "output_size_bytes": output_size,
        "device": device,
        "path": str(video_path),
    }


def _batch_worker(args: tuple) -> dict:
    """Top-level picklable worker for ProcessPoolExecutor (CPU path).

    Args (packed as tuple for pickling):
        video_path_str, output_path_str, model_dir_str, progress_file_str
        progress_file_str may be empty string (no progress reporting).
        Optional 5th element: crop tuple (x, y, w, h) or None.
    """
    video_path_str, output_path_str, model_dir_str, progress_file_str = args[:4]
    crop = args[4] if len(args) > 4 else None
    video_path = Path(video_path_str)
    output_path = Path(output_path_str)
    model_dir = Path(model_dir_str) if model_dir_str else None

    return extract_pose(
        video_path,
        output_path,
        progress_callback=_make_progress_writer(progress_file_str),
        model_dir=model_dir,
        n_concurrent=1,
        use_gpu=False,   # CPU/XNNPACK — stable for multi-process
        crop=crop,
    )


def _metal_gpu_worker(args: tuple) -> dict:
    """Thread worker for Metal GPU batch extraction.

    Each thread creates its own MediaPipe detectors with Metal GPU delegate.
    Threads share the process Metal context — no CVPixelBuffer pool conflicts
    (-6662) that occur with separate processes.

    Uses a prefetch reader thread so frame decode (CPU) overlaps with GPU
    inference — without this, GPU sits idle ~75% of the time waiting for
    cv2.read()/cvtColor().
    """
    import cv2

    video_path_str, output_path_str, model_dir_str, progress_file_str = args[:4]
    crop = args[4] if len(args) > 4 else None
    video_path = Path(video_path_str)
    output_path = Path(output_path_str)
    model_dir = Path(model_dir_str) if model_dir_str else None
    model_paths = ensure_models(model_dir)
    _write_progress = _make_progress_writer(progress_file_str)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": f"Cannot open video: {video_path}", "path": str(video_path)}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    _, total_frames = _probe_video_info(video_path, fps)

    try:
        pose_det, hand_det, face_det, device = _build_detectors(model_paths, use_gpu=True)
    except Exception as exc:
        cap.release()
        return {"error": f"Metal GPU init failed: {exc}", "path": str(video_path)}

    # Prefetch queue: reader thread decodes frames while GPU processes.
    # Keep queue SMALL (2-3 frames) — each frame becomes a Metal CVPixelBuffer
    # when processed. Large queues waste memory for no gain since GPU inference
    # is the bottleneck.
    q_size = 3
    frame_q: queue.Queue = queue.Queue(maxsize=q_size)

    def _reader():
        ridx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if crop:
                    cx, cy, cw, ch = crop
                    frame = frame[cy:cy + ch, cx:cx + cw]
                rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                frame_q.put((rgba, int(ridx * 1000 / fps)))
                ridx += 1
        finally:
            frame_q.put(None)

    reader_t = threading.Thread(target=_reader, daemon=True)
    reader_t.start()

    frames: list[np.ndarray] = []
    idx = 0
    frames_since_recycle = 0
    try:
        while True:
            item = frame_q.get()
            if item is None:
                break
            rgba, timestamp_ms = item
            lm = _process_frame(rgba, timestamp_ms, pose_det, hand_det, face_det)
            frames.append(lm)
            idx += 1
            frames_since_recycle += 1
            if total_frames > 0:
                _write_progress(min(idx / total_frames, 0.99))

            if frames_since_recycle >= _GPU_RECYCLE_INTERVAL:
                pose_det, hand_det, face_det, device = _recycle_detectors(
                    pose_det, hand_det, face_det, model_paths)
                frames_since_recycle = 0
    finally:
        reader_t.join(timeout=5)
        cap.release()
        pose_det.close()
        hand_det.close()
        face_det.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_pose(frames, fps, output_path)
    _write_progress(1.0)

    duration_sec = len(frames) / fps if fps > 0 else 0.0
    output_size = output_path.stat().st_size if output_path.exists() else 0

    return {
        "frames": len(frames),
        "duration_sec": round(duration_sec, 2),
        "output_size_bytes": output_size,
        "device": device,
        "path": str(video_path),
    }


def extract_pose_batch(
    video_paths: list[Path],
    output_dir: Path,
    progress_dir: Path | None = None,
    model_dir: Path | None = None,
    n_workers: int | None = None,
    use_gpu: bool = False,
) -> list[dict]:
    """Process multiple videos in parallel.

    Two modes:
      - use_gpu=False (default): ProcessPoolExecutor with CPU/XNNPACK.
        Each process is independent. Stable, no memory leaks.
      - use_gpu=True: ThreadPoolExecutor with Metal GPU delegate.
        Threads share one Metal context — uses all 40 GPU cores on Apple Silicon.
        ~60% faster than Apple Vision ANE on M4 Max.
        Known concern: MediaPipe Metal has memory leaks on macOS (issues #5652, #6223).

    Per-video progress is written to .progress temp files in progress_dir
    (float 0.0–1.0) so the caller can poll them for live UI updates.

    Args:
        video_paths: Videos to process.
        output_dir: Directory to write .pose files.
        progress_dir: Optional directory for per-video .progress temp files.
        model_dir: Override model directory.
        n_workers: Parallel workers. Auto-scaled if None.
        use_gpu: Use Metal GPU via ThreadPoolExecutor (default: False = CPU).

    Returns:
        List of result dicts in completion order.
        Failed videos included as {'error': str, 'path': str}.
    """
    if not video_paths:
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-download models in the main process before spawning workers
    ensure_models(model_dir)

    # Build args tuples (shared across backends)
    args_list = []
    for vpath in video_paths:
        vpath = Path(vpath)
        out = output_dir / f"{vpath.stem}.pose"
        pfile = str(Path(progress_dir) / f"{vpath.stem}.progress") \
                if progress_dir else ""
        args_list.append((str(vpath), str(out),
                          str(model_dir) if model_dir else "", pfile))

    # Select worker function and executor type based on backend
    if use_gpu:
        if n_workers is None:
            n_workers = min(8, len(video_paths))  # 8 threads optimal on M4 Max
        worker_fn = _metal_gpu_worker
        ExecutorCls = ThreadPoolExecutor
    else:
        if n_workers is None:
            n_workers = recommend_workers(len(video_paths))
        worker_fn = _batch_worker
        ExecutorCls = ProcessPoolExecutor

    results: list[dict] = []
    with ExecutorCls(max_workers=n_workers) as pool:
        futures = {pool.submit(worker_fn, args): Path(args[0])
                   for args in args_list}
        for future in as_completed(futures):
            vpath = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                logger.error(f"Failed {vpath.name}: {exc}")
                results.append({"error": str(exc), "path": str(vpath)})
    return results


def extract_pose_dual_view(
    video_path: Path,
    output_dir: Path,
    crop_left: tuple[int, int, int, int] = (0, 0, 632, 720),
    crop_right: tuple[int, int, int, int] = (648, 0, 632, 720),
    suffix_left: str = "_60",
    suffix_right: str = "_90",
    progress_callback: Optional[Callable[[float], None]] = None,
    model_dir: Path | None = None,
    use_gpu: bool | None = None,
) -> list[dict]:
    """Extract poses from a dual-view (split-screen) video.

    category-source videos have two camera angles side by side (1280x720):
    left half = ~60 deg frontal view, right half = ~90 deg profile view,
    with a dark divider at cols 632-647.

    This function calls extract_pose() twice (once per crop) and saves two
    .pose files: ``{stem}{suffix_left}.pose`` and ``{stem}{suffix_right}.pose``.

    Crop bounds are auto-adjusted to the actual frame height when it differs
    from the default 720 (e.g. 1280x556 videos).

    Args:
        video_path: Input dual-view video.
        output_dir: Directory for output .pose files.
        crop_left: (x, y, w, h) for the left half.
        crop_right: (x, y, w, h) for the right half.
        suffix_left: Suffix appended to stem for left output.
        suffix_right: Suffix appended to stem for right output.
        progress_callback: Called with float in [0, 1] (0-0.5 for left, 0.5-1.0 for right).
        model_dir: Override model storage directory.
        use_gpu: Use Metal GPU (None = auto-detect).

    Returns:
        List of two result dicts (left, right).
    """
    import cv2

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    # Probe actual frame height and adjust crops if needed
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    cap.release()

    def _adjust_crop(crop: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x, y, w, h = crop
        if h != actual_h:
            h = actual_h
        return (x, y, w, h)

    adj_left = _adjust_crop(crop_left)
    adj_right = _adjust_crop(crop_right)

    results: list[dict] = []

    # Left half (e.g. 60 deg frontal)
    out_left = output_dir / f"{stem}{suffix_left}.pose"

    def _prog_left(frac: float):
        if progress_callback:
            progress_callback(frac * 0.5)

    res_left = extract_pose(
        video_path, out_left,
        progress_callback=_prog_left,
        model_dir=model_dir,
        use_gpu=use_gpu,
        crop=adj_left,
    )
    res_left["view"] = "left"
    res_left["suffix"] = suffix_left
    results.append(res_left)

    # Right half (e.g. 90 deg profile)
    out_right = output_dir / f"{stem}{suffix_right}.pose"

    def _prog_right(frac: float):
        if progress_callback:
            progress_callback(0.5 + frac * 0.5)

    res_right = extract_pose(
        video_path, out_right,
        progress_callback=_prog_right,
        model_dir=model_dir,
        use_gpu=use_gpu,
        crop=adj_right,
    )
    res_right["view"] = "right"
    res_right["suffix"] = suffix_right
    results.append(res_right)

    return results


def _dual_view_gpu_worker(args: tuple) -> dict:
    """Thread worker for dual-view Metal GPU batch extraction.

    Processes one video producing two .pose files (left + right crop).
    Calls _metal_gpu_worker twice sequentially with different crops.
    """
    (video_path_str, output_dir_str, model_dir_str, progress_file_str,
     crop_left, crop_right, suffix_left, suffix_right) = args

    stem = Path(video_path_str).stem
    output_dir = Path(output_dir_str)

    results = []
    for crop, suffix, label in [
        (crop_left, suffix_left, "left"),
        (crop_right, suffix_right, "right"),
    ]:
        out_path = str(output_dir / f"{stem}{suffix}.pose")
        sub_args = (video_path_str, out_path, model_dir_str, "", crop)
        res = _metal_gpu_worker(sub_args)
        res["view"] = label
        res["suffix"] = suffix
        results.append(res)

    # Write progress=1.0 for the video as a whole
    if progress_file_str:
        try:
            Path(progress_file_str).write_text("1.0")
        except Exception:
            pass

    return results


def _dual_view_cpu_worker(args: tuple) -> dict:
    """Process worker for dual-view CPU batch extraction.

    Processes one video producing two .pose files (left + right crop).
    """
    (video_path_str, output_dir_str, model_dir_str, progress_file_str,
     crop_left, crop_right, suffix_left, suffix_right) = args

    stem = Path(video_path_str).stem
    output_dir = Path(output_dir_str)

    results = []
    for crop, suffix, label in [
        (crop_left, suffix_left, "left"),
        (crop_right, suffix_right, "right"),
    ]:
        out_path = str(output_dir / f"{stem}{suffix}.pose")
        sub_args = (video_path_str, out_path, model_dir_str, "", crop)
        res = _batch_worker(sub_args)
        res["view"] = label
        res["suffix"] = suffix
        results.append(res)

    if progress_file_str:
        try:
            Path(progress_file_str).write_text("1.0")
        except Exception:
            pass

    return results


def extract_pose_dual_view_batch(
    video_paths: list[Path],
    output_dir: Path,
    progress_dir: Path | None = None,
    model_dir: Path | None = None,
    n_workers: int | None = None,
    use_gpu: bool = False,
    crop_left: tuple[int, int, int, int] = (0, 0, 632, 720),
    crop_right: tuple[int, int, int, int] = (648, 0, 632, 720),
    suffix_left: str = "_60",
    suffix_right: str = "_90",
) -> list[dict]:
    """Process multiple dual-view videos in parallel.

    Each video produces two .pose files (left and right crops).
    Uses the same executor strategy as extract_pose_batch():
    ThreadPoolExecutor for GPU, ProcessPoolExecutor for CPU.

    Crop heights are auto-adjusted per video based on actual frame height
    (handles 1280x720 and 1280x556 category-source videos).

    Args:
        video_paths: Dual-view videos to process.
        output_dir: Directory to write .pose files.
        progress_dir: Optional directory for per-video .progress temp files.
        model_dir: Override model directory.
        n_workers: Parallel workers. Auto-scaled if None.
        use_gpu: Use Metal GPU via ThreadPoolExecutor (default: False = CPU).
        crop_left: (x, y, w, h) for the left half.
        crop_right: (x, y, w, h) for the right half.
        suffix_left: Suffix for left-view output files.
        suffix_right: Suffix for right-view output files.

    Returns:
        List of result lists (each is [left_result, right_result]).
        Failed videos included as [{'error': str, 'path': str}].
    """
    import cv2

    if not video_paths:
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-download models
    ensure_models(model_dir)

    # Build args — adjust crop height per video
    args_list = []
    for vpath in video_paths:
        vpath = Path(vpath)
        pfile = str(Path(progress_dir) / f"{vpath.stem}.progress") \
                if progress_dir else ""

        # Probe actual height to adjust crop bounds
        cap = cv2.VideoCapture(str(vpath))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        cap.release()

        def _adj(crop: tuple[int, int, int, int], h: int) -> tuple[int, int, int, int]:
            return (crop[0], crop[1], crop[2], h)

        adj_l = _adj(crop_left, actual_h)
        adj_r = _adj(crop_right, actual_h)

        args_list.append((
            str(vpath), str(output_dir),
            str(model_dir) if model_dir else "",
            pfile, adj_l, adj_r, suffix_left, suffix_right,
        ))

    # Select worker and executor
    if use_gpu:
        if n_workers is None:
            # Cropped frames are 632px wide = "small" → 8 threads OK
            n_workers = min(8, len(video_paths))
        worker_fn = _dual_view_gpu_worker
        ExecutorCls = ThreadPoolExecutor
    else:
        if n_workers is None:
            n_workers = recommend_workers(len(video_paths))
        worker_fn = _dual_view_cpu_worker
        ExecutorCls = ProcessPoolExecutor

    results: list[dict] = []
    with ExecutorCls(max_workers=n_workers) as pool:
        futures = {pool.submit(worker_fn, args): Path(args[0])
                   for args in args_list}
        for future in as_completed(futures):
            vpath = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                logger.error(f"Dual-view failed {vpath.name}: {exc}")
                results.append([{"error": str(exc), "path": str(vpath)}])

    return results


def find_corrupted_poses(
    pose_dir: Path,
    min_bytes: int = 10_000,
) -> list[dict]:
    """Scan pose directory for files smaller than threshold (likely corrupted).

    Args:
        pose_dir: Directory containing .pose files.
        min_bytes: Files smaller than this are flagged (default 10KB).

    Returns:
        List of dicts: [{filename, path, size_bytes}, ...]
    """
    pose_dir = Path(pose_dir)
    if not pose_dir.exists():
        return []

    corrupted = []
    for f in sorted(pose_dir.glob("*.pose")):
        try:
            size = f.stat().st_size
            if size < min_bytes:
                corrupted.append({
                    "filename": f.name,
                    "path": str(f),
                    "size_bytes": size,
                })
        except OSError:
            corrupted.append({
                "filename": f.name,
                "path": str(f),
                "size_bytes": -1,
            })
    return corrupted


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _probe_video_info(video_path: Path, cv2_fps: float = 25.0) -> tuple[float, int]:
    """Return (fps, frame_count) from a single ffprobe call.

    cv2.CAP_PROP_FRAME_COUNT is unreliable for MKV/WebM from yt-dlp.
    Falls back to cv2 if ffprobe fails.
    """
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", "-show_format", str(video_path)],
            capture_output=True, text=True, timeout=15,
        )
        data = json.loads(probe.stdout)
        # Extract FPS from video stream
        fps = cv2_fps
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                r_fps = stream.get("r_frame_rate", "")
                if "/" in r_fps:
                    num, den = r_fps.split("/")
                    if float(den) > 0:
                        fps = float(num) / float(den)
                break
        # Frame count from duration × fps
        duration = float(data.get("format", {}).get("duration", 0))
        if duration > 0:
            return fps, max(1, int(duration * fps))
    except Exception:
        pass
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return cv2_fps, max(n, 1)


def _build_detectors(model_paths: dict[str, Path], use_gpu: bool = True):
    """Create the three landmarkers in VIDEO mode.

    Args:
        use_gpu: If True, try Metal GPU first then fall back to CPU.
                 If False, go straight to CPU (XNNPACK).
                 Must be False when running multiple parallel workers —
                 macOS CoreVideo pixel buffer pool exhausts at -6662 with
                 concurrent Metal contexts.
    """
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks import python as mp_python

    Delegate = mp_python.BaseOptions.Delegate
    RunningMode = vision.RunningMode

    candidates = [(Delegate.GPU, "GPU"), (Delegate.CPU, "CPU")] if use_gpu \
                 else [(Delegate.CPU, "CPU")]

    for delegate, label in candidates:
        try:
            def base(key, _d=delegate):
                return mp_python.BaseOptions(
                    model_asset_path=str(model_paths[key]),
                    delegate=_d,
                )

            pose_opts = vision.PoseLandmarkerOptions(
                base_options=base("pose"),
                running_mode=RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            hand_opts = vision.HandLandmarkerOptions(
                base_options=base("hand"),
                running_mode=RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            face_opts = vision.FaceLandmarkerOptions(
                base_options=base("face"),
                running_mode=RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )

            return (
                vision.PoseLandmarker.create_from_options(pose_opts),
                vision.HandLandmarker.create_from_options(hand_opts),
                vision.FaceLandmarker.create_from_options(face_opts),
                label,
            )
        except Exception as exc:
            if label == "GPU":
                logger.warning(f"GPU delegate failed ({exc}), falling back to CPU")
            else:
                raise


def _process_frame(
    rgba_frame: np.ndarray,
    timestamp_ms: int,
    pose_det,
    hand_det,
    face_det,
) -> np.ndarray:
    """Run all three detectors; return (543, 4) float32 array."""
    import mediapipe as mp

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=rgba_frame)

    pose_result = pose_det.detect_for_video(mp_image, timestamp_ms)
    hand_result = hand_det.detect_for_video(mp_image, timestamp_ms)
    face_result = face_det.detect_for_video(mp_image, timestamp_ms)

    body = np.zeros((33, 4), dtype=np.float32)
    if pose_result.pose_landmarks:
        for i, lm in enumerate(pose_result.pose_landmarks[0]):
            body[i] = [lm.x, lm.y, lm.z, lm.visibility]

    left_hand = np.zeros((21, 4), dtype=np.float32)
    right_hand = np.zeros((21, 4), dtype=np.float32)
    if hand_result.hand_landmarks:
        for idx, landmarks in enumerate(hand_result.hand_landmarks):
            handedness = hand_result.handedness[idx][0].category_name
            arr = np.array([
                [lm.x, lm.y, lm.z,
                 lm.visibility if lm.visibility is not None and lm.visibility > 0 else 1.0]
                for lm in landmarks
            ], dtype=np.float32)
            if handedness == "Left":
                left_hand = arr
            else:
                right_hand = arr

    face = np.zeros((468, 4), dtype=np.float32)
    if face_result.face_landmarks:
        for i, lm in enumerate(face_result.face_landmarks[0]):
            if i >= 468:
                break
            face[i] = [lm.x, lm.y, lm.z,
                       lm.visibility if lm.visibility is not None and lm.visibility > 0 else 1.0]

    return np.concatenate([body, left_hand, right_hand, face], axis=0)


def _save_pose(frames: list[np.ndarray], fps: float, output_path: Path) -> None:
    """Save frames to .pose; falls back to .npz if pose_format unavailable."""
    try:
        from pose_format import Pose
        from pose_format.pose_header import (
            PoseHeader, PoseHeaderComponent, PoseHeaderDimensions,
        )
        from pose_format.numpy.pose_body import NumPyPoseBody

        dimensions = PoseHeaderDimensions(width=1, height=1, depth=1)
        # points must be List[str] — one name per landmark
        point_names = (
            [f"BODY_{i}"       for i in range(33)]   # 0–32
          + [f"LEFT_HAND_{i}"  for i in range(21)]   # 33–53
          + [f"RIGHT_HAND_{i}" for i in range(21)]   # 54–74
          + [f"FACE_{i}"       for i in range(468)]  # 75–542
        )
        components = [
            PoseHeaderComponent(
                name="POSE_LANDMARKS",
                points=point_names,
                limbs=[],
                colors=[],
                point_format="XYZC",
            )
        ]
        header = PoseHeader(version=0.1, dimensions=dimensions, components=components)
        stacked = np.stack(frames, axis=0)[:, np.newaxis, :, :]  # (T, 1, 543, 4)
        data = stacked[:, :, :, :3].astype(np.float32)           # (T, 1, 543, 3) XYZ
        conf = stacked[:, :, :, 3:4].astype(np.float32)          # (T, 1, 543, 1) visibility
        body = NumPyPoseBody(fps=int(fps), data=data, confidence=conf)
        pose = Pose(header=header, body=body)
        with open(output_path, "wb") as f:
            pose.write(f)

    except ImportError:
        logger.warning("pose_format unavailable — saving as .npz fallback")
        arr = np.stack(frames, axis=0) if frames else np.zeros((0, 543, 4))
        npz_path = output_path.with_suffix(".npz")
        np.savez_compressed(str(npz_path), pose=arr, fps=np.array(fps))
        output_path.touch()


# ---------------------------------------------------------------------------
# Apple Vision backend (Swift CLI, ANE+GPU on Apple Silicon)
# ---------------------------------------------------------------------------

_APPLE_VISION_BINARY = Path(__file__).parent.parent.parent / "tools" / "apple-vision-pose" / ".build" / "release" / "apple-vision-pose"


def apple_vision_available() -> bool:
    """Return True if the Apple Vision Swift CLI binary exists and is executable."""
    return _APPLE_VISION_BINARY.exists() and _APPLE_VISION_BINARY.is_file()


def _parse_jsonl(jsonl_path: Path, video_fps: float) -> list[np.ndarray]:
    """Parse JSONL output from Apple Vision CLI into list of (543, 4) frames.

    Each JSONL line has: {"frame": int, "timestamp": float, "landmarks": [[x,y,conf], ...]}
    Converts to (543, 4) format: x, y, z=0, confidence.
    """
    frames: list[np.ndarray] = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            landmarks = obj.get("landmarks", [])
            frame = np.zeros((543, 4), dtype=np.float32)
            for i, lm in enumerate(landmarks[:543]):
                if len(lm) >= 3:
                    frame[i] = [lm[0], lm[1], 0.0, lm[2]]  # x, y, z=0, confidence
            frames.append(frame)
    return frames


def extract_pose_apple(
    video_path: Path,
    output_path: Path,
    progress_callback: Optional[Callable[[float], None]] = None,
    target_fps: int = 0,
    concurrent_frames: int = 1,
) -> dict:
    """Extract pose landmarks using Apple Vision (ANE+GPU).

    Calls the Swift CLI tool, reads its JSONL output, converts to .pose format
    via _save_pose().

    Args:
        video_path: Input video file.
        output_path: Destination .pose file.
        progress_callback: Called with float in [0, 1] as frames are processed.
        target_fps: Target FPS for sampling (0 = every frame).
        concurrent_frames: Concurrent frame workers within the Swift CLI
            (1 = sequential, >1 = GCD concurrent Vision inference).

    Returns:
        Dict: frames, duration_sec, output_size_bytes, device.
    """
    import tempfile

    video_path = Path(video_path)
    output_path = Path(output_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not apple_vision_available():
        raise RuntimeError(
            f"Apple Vision binary not found at {_APPLE_VISION_BINARY}. "
            "Build with: cd tools/apple-vision-pose && swift build -c release"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Run Swift CLI → JSONL temp file
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        jsonl_path = Path(tmp.name)

    try:
        cmd = [
            str(_APPLE_VISION_BINARY), "extract",
            "--input", str(video_path),
            "--output", str(jsonl_path),
        ]
        if target_fps > 0:
            cmd.extend(["--fps", str(target_fps)])
        if concurrent_frames > 1:
            cmd.extend(["--concurrent", str(concurrent_frames)])

        # Run with progress monitoring via stderr
        proc = subprocess.Popen(
            cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True,
        )

        # Parse stderr progress lines
        total_frames = 0
        for line in proc.stderr:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                if "total" in msg and "frame" in msg:
                    total_frames = msg["total"]
                    if progress_callback and total_frames > 0:
                        progress_callback(min(msg["frame"] / total_frames, 0.99))
            except json.JSONDecodeError:
                pass

        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"Apple Vision CLI exited with code {proc.returncode}")

        # Parse JSONL and get video FPS
        video_fps, _ = _probe_video_info(video_path)
        frames = _parse_jsonl(jsonl_path, video_fps)

        if not frames:
            raise RuntimeError("Apple Vision produced 0 frames")

        # Save as .pose format
        _save_pose(frames, video_fps, output_path)

        if progress_callback:
            progress_callback(1.0)

        duration_sec = len(frames) / video_fps if video_fps > 0 else 0.0
        output_size = output_path.stat().st_size if output_path.exists() else 0

        return {
            "frames": len(frames),
            "duration_sec": round(duration_sec, 2),
            "output_size_bytes": output_size,
            "device": "Apple Vision (ANE+GPU)",
            "path": str(video_path),
        }
    finally:
        jsonl_path.unlink(missing_ok=True)


def _apple_batch_worker(args: tuple) -> dict:
    """Worker for Apple Vision batch extraction via ThreadPoolExecutor."""
    video_path_str, output_path_str, target_fps, progress_file_str, concurrent_frames = args
    video_path = Path(video_path_str)
    output_path = Path(output_path_str)

    return extract_pose_apple(
        video_path, output_path,
        progress_callback=_make_progress_writer(progress_file_str),
        target_fps=target_fps,
        concurrent_frames=concurrent_frames,
    )


def extract_pose_apple_batch(
    video_paths: list[Path],
    output_dir: Path,
    progress_dir: Path | None = None,
    target_fps: int = 0,
    n_workers: int | None = None,
    concurrent_frames: int = 4,
) -> list[dict]:
    """Process multiple videos in parallel using Apple Vision.

    Uses ThreadPoolExecutor (not Process) — Apple Vision runs on ANE/GPU
    per-process, no Metal context conflicts with threads.

    Args:
        video_paths: Videos to process.
        output_dir: Directory to write .pose files.
        progress_dir: Optional directory for per-video .progress temp files.
        target_fps: Target FPS for sampling (0 = every frame).
        n_workers: Parallel workers. Default 4 (ANE handles scheduling).
        concurrent_frames: Concurrent frame workers per video within the
            Swift CLI (1 = sequential, >1 = GCD concurrent Vision inference).

    Returns:
        List of result dicts in completion order.
    """
    if not video_paths:
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if n_workers is None:
        n_workers = min(4, len(video_paths))

    args_list = []
    for vpath in video_paths:
        vpath = Path(vpath)
        out = output_dir / f"{vpath.stem}.pose"
        pfile = str(Path(progress_dir) / f"{vpath.stem}.progress") \
                if progress_dir else ""
        args_list.append((str(vpath), str(out), target_fps, pfile, concurrent_frames))

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_apple_batch_worker, args): Path(args[0])
                   for args in args_list}
        for future in as_completed(futures):
            vpath = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                logger.error(f"Apple Vision failed {vpath.name}: {exc}")
                results.append({"error": str(exc), "path": str(vpath)})

    return results
