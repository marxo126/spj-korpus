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

def pose_exists(video_path: Path, pose_dir: Path) -> bool:
    """Return True if a .pose output already exists for this video."""
    stem = Path(video_path).stem
    return (Path(pose_dir) / f"{stem}.pose").exists()


def extract_pose(
    video_path: Path,
    output_path: Path,
    progress_callback: Optional[Callable[[float], None]] = None,
    model_dir: Path | None = None,
    n_concurrent: int = 1,
    use_gpu: bool | None = None,
) -> dict:
    """Extract pose landmarks from a single video.

    Args:
        video_path: Input video file.
        output_path: Destination .pose file.
        progress_callback: Called with float in [0, 1] as frames are processed.
        model_dir: Override model storage directory.
        n_concurrent: How many parallel extract_pose calls are running right now
                      (used to size the per-worker frame buffer from free RAM).

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

    total_frames = _probe_frame_count(video_path, fps)
    queue_size = _compute_queue_size(frame_w, frame_h, n_concurrent)

    # GPU safe only when one pipeline at a time — concurrent Metal CVPixelBuffer
    # allocation fails with -6662 (pool exhausted) when n_concurrent > 1
    if use_gpu is None:
        use_gpu = (n_concurrent == 1)
    pose_det, hand_det, face_det, device = _build_detectors(model_paths, use_gpu=use_gpu)
    logger.info(f"[{video_path.name}] device={device} frames≈{total_frames} queue={queue_size}")

    frame_q: queue.Queue = queue.Queue(maxsize=queue_size)

    def _reader():
        idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                frame_q.put((rgba, int(idx * 1000 / fps)))
                idx += 1
        finally:
            frame_q.put(None)

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    pose_frames: list[np.ndarray] = []
    processed = 0

    try:
        while True:
            item = frame_q.get()
            if item is None:
                break
            rgba, timestamp_ms = item
            lm = _process_frame(rgba, timestamp_ms, pose_det, hand_det, face_det)
            pose_frames.append(lm)
            processed += 1

            if progress_callback is not None and total_frames > 0:
                progress_callback(min(processed / total_frames, 0.99))
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
    """Top-level picklable worker for ProcessPoolExecutor.

    Uses CPU (XNNPACK) — the only officially supported path for macOS.
    MediaPipe documents GPU support for Ubuntu only; macOS GPU has known
    unbounded memory leaks (issues #5652, #6223) across all running modes.

    Each process is independent (spawn start method on macOS), so N workers
    fully utilise N × XNNPACK threads across the M-series CPU cores.

    Args (packed as tuple for pickling):
        video_path_str, output_path_str, model_dir_str, progress_file_str
        progress_file_str may be empty string (no progress reporting).
    """
    video_path_str, output_path_str, model_dir_str, progress_file_str = args
    video_path = Path(video_path_str)
    output_path = Path(output_path_str)
    model_dir = Path(model_dir_str) if model_dir_str else None

    def _write_progress(frac: float):
        if progress_file_str:
            try:
                Path(progress_file_str).write_text(str(round(frac, 4)))
            except Exception:
                pass

    return extract_pose(
        video_path,
        output_path,
        progress_callback=_write_progress,
        model_dir=model_dir,
        n_concurrent=1,
        use_gpu=False,   # CPU/XNNPACK — stable, no memory leaks, officially supported
    )


def extract_pose_batch(
    video_paths: list[Path],
    output_dir: Path,
    progress_dir: Path | None = None,
    model_dir: Path | None = None,
    n_workers: int | None = None,
) -> list[dict]:
    """Process multiple videos in parallel using separate OS processes.

    Each worker process has its own Metal GPU context and CVPixelBuffer pool,
    so all 40 GPU cores can be used concurrently without -6662 pool conflicts.

    Per-video progress is written to .progress temp files in progress_dir
    (float 0.0–1.0) so the caller can poll them for live UI updates.

    Args:
        video_paths: Videos to process.
        output_dir: Directory to write .pose files.
        progress_dir: Optional directory for per-video .progress temp files.
        model_dir: Override model directory.
        n_workers: Parallel workers. Auto-scaled to RAM if None.

    Returns:
        List of result dicts in completion order.
        Failed videos included as {'error': str, 'path': str}.
    """
    if not video_paths:
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if n_workers is None:
        n_workers = recommend_workers(len(video_paths))

    # Pre-download models in the main process before spawning workers
    ensure_models(model_dir)

    # Build args tuples (must be picklable — no lambdas, no closures)
    args_list = []
    for vpath in video_paths:
        vpath = Path(vpath)
        out = output_dir / f"{vpath.stem}.pose"
        pfile = str(Path(progress_dir) / f"{vpath.stem}.progress") \
                if progress_dir else ""
        args_list.append((str(vpath), str(out),
                          str(model_dir) if model_dir else "", pfile))

    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_batch_worker, args): Path(args[0])
                   for args in args_list}
        for future in as_completed(futures):
            vpath = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                logger.error(f"Failed {vpath.name}: {exc}")
                results.append({"error": str(exc), "path": str(vpath)})

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _probe_frame_count(video_path: Path, fps: float) -> int:
    """Accurate frame count via ffprobe duration × fps.

    cv2.CAP_PROP_FRAME_COUNT is unreliable for MKV/WebM from yt-dlp.
    """
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format",
             str(video_path)],
            capture_output=True, text=True, timeout=15,
        )
        data = json.loads(probe.stdout)
        duration = float(data.get("format", {}).get("duration", 0))
        if duration > 0:
            return max(1, int(duration * fps))
    except Exception:
        pass
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(n, 1)


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
            arr = np.array([[lm.x, lm.y, lm.z, 1.0] for lm in landmarks], dtype=np.float32)
            if handedness == "Left":
                left_hand = arr
            else:
                right_hand = arr

    face = np.zeros((468, 4), dtype=np.float32)
    if face_result.face_landmarks:
        for i, lm in enumerate(face_result.face_landmarks[0]):
            if i >= 468:
                break
            face[i] = [lm.x, lm.y, lm.z, 1.0]

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
