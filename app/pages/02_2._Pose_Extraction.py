"""Pose Extraction page — parallel extraction using MediaPipe or Apple Vision."""
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from spj.inventory import pose_exists
from spj.pose import (
    _batch_worker, _metal_gpu_worker, ensure_models, recommend_workers,
    apple_vision_available, _apple_batch_worker,
)

# Backend constants
_BACKEND_METAL = "MediaPipe Metal GPU"
_BACKEND_CPU = "MediaPipe (CPU)"
_BACKEND_AV = "Apple Vision (ANE)"

DATA_DIR = Path(__file__).parent.parent.parent / "data"
INVENTORY_CSV = DATA_DIR / "inventory.csv"
POSE_DIR = DATA_DIR / "pose"

st.header("2. 🏃 Pose Extraction")
st.caption("Page 2/10 · Extract 543 body+hand+face keypoints from each video.")
with st.expander("ℹ️ How to use this page", expanded=False):
    st.markdown("""
**Needs:**
- Inventory scanned (page 1 — Inventory)

**Steps:**
1. Set **Parallel workers** (each uses its own CPU thread; 5 is fast on M4 Max).
2. Tick **Run in background** if you want to navigate while it runs.
3. Click **▶ Extract All**.
4. Return here to check the counter. When done, Pending drops to 0.

**Creates:** `data/pose/<videoname>.pose` (one per video)

**Note:** First run downloads 3 MediaPipe models (~300 MB total) automatically.
""")

# ------------------------------------------------------------------ #
# Load inventory
# ------------------------------------------------------------------ #
inv: pd.DataFrame | None = st.session_state.get("inventory")
if inv is None:
    if INVENTORY_CSV.exists():
        inv = pd.read_csv(INVENTORY_CSV)
        st.session_state["inventory"] = inv
    else:
        st.warning("No inventory found. Go to the **Inventory** page first and scan your videos.")
        st.stop()

if inv.empty:
    st.warning("Inventory is empty.")
    st.stop()

inv["pose_extracted"] = inv["path"].apply(lambda p: pose_exists(Path(p), POSE_DIR))
st.session_state["inventory"] = inv

pending = inv[~inv["pose_extracted"]].copy()
done    = inv[inv["pose_extracted"]].copy()

col1, col2, col3 = st.columns(3)
col1.metric("Total videos", len(inv))
col2.metric("Pose done", len(done))
col3.metric("Pending", len(pending))

# Warn about 0-byte pose files that will be re-extracted
def _is_zero_byte(p: Path) -> bool:
    try:
        return p.stat().st_size == 0
    except FileNotFoundError:
        return False

_zero_byte = [
    Path(r["path"]).name for _, r in inv.iterrows()
    if _is_zero_byte(POSE_DIR / f"{Path(r['path']).stem}.pose")
]
if _zero_byte:
    st.warning(
        f"**{len(_zero_byte)} pose file(s) are 0 bytes** (failed extraction). "
        f"They are counted as pending and will be re-extracted: "
        + ", ".join(f"`{n}`" for n in _zero_byte[:5])
        + ("…" if len(_zero_byte) > 5 else "")
    )

# Check for corrupted (too-small) pose files — cached to avoid re-scanning on every rerun
from spj.pose import find_corrupted_poses
if "corrupted_poses" not in st.session_state:
    st.session_state["corrupted_poses"] = find_corrupted_poses(POSE_DIR, min_bytes=10_000)
_corrupted = st.session_state["corrupted_poses"]
# Exclude 0-byte files (already shown above)
_corrupted_nonzero = [c for c in _corrupted if c["size_bytes"] > 0]
if _corrupted_nonzero:
    with st.expander(f"⚠️ {len(_corrupted_nonzero)} potentially corrupted pose file(s) (< 10 KB)", expanded=False):
        st.caption(
            "These files are very small and may contain failed or partial extractions. "
            "Consider re-extracting them."
        )
        st.dataframe(
            pd.DataFrame(_corrupted_nonzero)[["filename", "size_bytes"]],
            hide_index=True, use_container_width=True,
        )

# Completion message stored by the extraction loop (shown once after rerun)
if "pose_completion_msg" in st.session_state:
    msg    = st.session_state.pop("pose_completion_msg")
    errors = st.session_state.pop("pose_completion_errors", [])
    if errors:
        with st.expander(f"Errors ({len(errors)})"):
            for e in errors:
                st.markdown(e)
    st.success(msg)

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
_POSE_SS_KEYS = (
    "pose_state", "pose_executor", "pose_futures_map", "pose_pending_futs",
    "pose_completed_count", "pose_log_lines", "pose_total",
    "pose_progress_files", "pose_active_slots", "pose_tmp_dir", "pose_bg_mode",
)


def _read_progress(pfile) -> float:
    try:
        p = Path(str(pfile))
        return float(p.read_text()) if p.exists() else 0.0
    except Exception:
        return 0.0


def _cancel_extraction() -> None:
    """Cancel pending futures, shut down executor, delete temp files."""
    executor     = st.session_state.get("pose_executor")
    pending_futs = st.session_state.get("pose_pending_futs", set())
    prog_files   = st.session_state.get("pose_progress_files", {})
    tmp_dir      = st.session_state.get("pose_tmp_dir")

    for fut in pending_futs:
        fut.cancel()
    if executor:
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
    for pf in prog_files.values():
        try:
            Path(str(pf)).unlink(missing_ok=True)
        except Exception:
            pass
    if tmp_dir:
        try:
            Path(str(tmp_dir)).rmdir()
        except Exception:
            pass
    for key in _POSE_SS_KEYS:
        st.session_state.pop(key, None)


# ------------------------------------------------------------------ #
# Active extraction — re-entered on every page visit while running
# ------------------------------------------------------------------ #
_state = st.session_state.get("pose_state", "idle")

if _state == "running":
    executor        = st.session_state["pose_executor"]
    futures_map     = st.session_state["pose_futures_map"]
    pending_futs    = st.session_state["pose_pending_futs"]
    completed_count = st.session_state["pose_completed_count"]
    log_lines       = st.session_state["pose_log_lines"]
    total           = st.session_state["pose_total"]
    progress_files  = st.session_state["pose_progress_files"]
    active_slots    = st.session_state["pose_active_slots"]
    bg_mode         = st.session_state.get("pose_bg_mode", False)

    # Header + stop button
    hcol1, hcol2 = st.columns([5, 1])
    with hcol1:
        if bg_mode:
            st.info("⚙️ Running in background — navigate freely, return here to check progress.")
        else:
            st.info("⚙️ Extraction in progress…")
    with hcol2:
        if st.button("■ Stop", type="secondary", use_container_width=True):
            _cancel_extraction()
            st.toast("Extraction stopped.", icon="⚠️")
            st.rerun()

    # Overall progress bar
    st.progress(
        completed_count / total if total > 0 else 0.0,
        text=f"{completed_count} / {total} completed",
    )

    # Poll futures — blocking in foreground, instant check in background
    poll_timeout = 0.4 if not bg_mode else 0.0
    done_set, still_pending = wait(
        pending_futs, timeout=poll_timeout, return_when=FIRST_COMPLETED
    )

    # Process completed futures
    for future in done_set:
        vp = futures_map[future]

        # Free the active slot
        for slot_i, vpath_str in list(active_slots.items()):
            if vpath_str == str(vp):
                del active_slots[slot_i]
                break

        # Log result
        try:
            result  = future.result()
            device  = result.get("device", "?")
            frames  = result.get("frames", 0)
            dur     = result.get("duration_sec", 0)
            kb      = result.get("output_size_bytes", 0) // 1024
            line    = f"✅ `{vp.name}` — {frames} frames, {dur:.1f}s, {kb} KB [{device}]"
        except Exception as exc:
            line = f"❌ `{vp.name}` — {exc}"
        log_lines = (log_lines + [line])[-100:]  # keep last 100 entries
        completed_count += 1

        # Assign freed slot to the next queued video
        for fut in still_pending:
            nxt = str(futures_map[fut])
            if nxt not in active_slots.values():
                used     = set(active_slots.keys())
                new_slot = next(i for i in range(len(futures_map) + 1) if i not in used)
                active_slots[new_slot] = nxt
                break

    # Persist updated state
    st.session_state["pose_pending_futs"]    = still_pending
    st.session_state["pose_active_slots"]    = active_slots
    st.session_state["pose_log_lines"]       = log_lines
    st.session_state["pose_completed_count"] = completed_count

    # Per-video progress bars (foreground only)
    if not bg_mode and active_slots:
        st.markdown("**Active workers:**")
        for vpath_str in active_slots.values():
            vp   = Path(vpath_str)
            pf   = progress_files.get(vpath_str)
            frac = _read_progress(pf) if pf else 0.0
            st.progress(min(frac, 1.0), text=f"⏳ {vp.name}  {frac*100:.0f}%")

    # Recent log
    if log_lines:
        st.markdown("\n\n".join(log_lines[-15:]))

    # ── All done? ──────────────────────────────────────────────────
    if not still_pending:
        try:
            executor.shutdown(wait=True)
        except Exception:
            pass
        for pf in progress_files.values():
            try:
                Path(str(pf)).unlink(missing_ok=True)
            except Exception:
                pass
        try:
            Path(str(st.session_state.get("pose_tmp_dir", ""))).rmdir()
        except Exception:
            pass

        # Refresh inventory
        inv["pose_extracted"] = inv["path"].apply(lambda p: pose_exists(Path(p), POSE_DIR))
        st.session_state["inventory"] = inv
        inv.to_csv(INVENTORY_CSV, index=False)
        n_done  = int(inv["pose_extracted"].sum())
        errors  = [l for l in log_lines if l.startswith("❌")]

        # Store completion message — shown after the rerun below
        st.session_state["pose_completion_msg"]    = (
            f"Done! {n_done} / {len(inv)} videos have pose data."
        )
        st.session_state["pose_completion_errors"] = errors

        # Clear extraction state + HD cache, then rerun to show idle UI + success banner
        for key in _POSE_SS_KEYS:
            st.session_state.pop(key, None)
        st.session_state.pop("pose_has_hd", None)
        st.rerun()

    # ── Continue ───────────────────────────────────────────────────
    # Background mode: refresh every 2s so the counter updates when the user
    # is watching this page, without blocking navigation elsewhere.
    time.sleep(2.0 if bg_mode else 0.05)
    st.rerun()

# ------------------------------------------------------------------ #
# Idle state — configuration UI
# ------------------------------------------------------------------ #

_av_ok = apple_vision_available()

# Three backends: Metal GPU (fastest), Apple Vision (ANE), MediaPipe CPU
_backend_options = [_BACKEND_METAL, _BACKEND_CPU]
if _av_ok:
    _backend_options.append(_BACKEND_AV)
_default_idx = 0  # Metal GPU is default (fastest on Apple Silicon)

backend = st.radio(
    "Extraction backend",
    _backend_options,
    index=_default_idx,
    horizontal=True,
    help="**Metal GPU** — fastest (269 fps, 8 threads, uses 40 GPU cores). "
         "**CPU** — stable fallback (XNNPACK, separate processes). "
         "**Apple Vision** — ANE-based (167 fps, 16 workers). "
         "All produce identical .pose file format.",
    key="pose_backend",
)
_use_apple_vision = (backend == _BACKEND_AV)
_use_metal_gpu = (backend == _BACKEND_METAL)

try:
    import psutil
    mem      = psutil.virtual_memory()
    free_gb  = mem.available / 1_073_741_824
    total_gb = mem.total     / 1_073_741_824
    suggested_cpu = recommend_workers(len(pending))
    st.caption(
        f"Free RAM: **{free_gb:.1f} GB** / {total_gb:.0f} GB"
    )
except ImportError:
    suggested_cpu = 2

if _use_metal_gpu:
    # Check if inventory has HD videos — Metal crashes with >2 threads on 1080p
    # Cached in session_state to avoid opening video files on every Streamlit rerun.
    if "pose_has_hd" not in st.session_state:
        _has_hd = False
        if not pending.empty:
            try:
                import cv2 as _cv2
                for _sp in pending.head(5)["path"].tolist():
                    _cap = _cv2.VideoCapture(str(_sp))
                    if _cap.isOpened() and _cap.get(_cv2.CAP_PROP_FRAME_WIDTH) > 720:
                        _has_hd = True
                    _cap.release()
                    if _has_hd:
                        break
            except Exception:
                pass
        st.session_state["pose_has_hd"] = _has_hd
    _has_hd = st.session_state["pose_has_hd"]
    if _has_hd:
        _default_workers = min(2, max(1, len(pending)))
        _max_workers = 4
        _help = ("HD videos detected — Metal CVPixelBuffer pool exhausts above 2-3 threads "
                 "at 1080p. Use 2 threads for stability (~195 fps).")
    else:
        _default_workers = min(8, max(1, len(pending)))
        _max_workers = 12
        _help = "Metal GPU threads share one Metal context. 8 threads optimal for small videos (~400 fps)."
elif _use_apple_vision:
    _default_workers = min(4, max(1, len(pending)))
    _max_workers = 16
    _help = "Apple Vision video workers (each runs Swift CLI subprocess)."
else:
    _default_workers = min(suggested_cpu, max(1, len(pending)))
    _max_workers = 8
    _help = "CPU workers run as separate processes (XNNPACK)."

n_workers = st.slider(
    "Parallel workers",
    min_value=1, max_value=_max_workers,
    value=_default_workers,
    help=_help,
)
if _use_metal_gpu:
    _hd_note = " HD detected — limited threads to prevent Metal crash." if _has_hd else ""
    st.caption(
        f"**{n_workers} thread(s) → Metal GPU (40 cores).**  "
        f"Threads share one Metal context.{_hd_note}"
    )
elif _use_apple_vision:
    frame_concurrent = st.slider(
        "Concurrent frames per video",
        min_value=1, max_value=16, value=4,
        help="GCD concurrent Vision inference within each video. "
             "ANE saturates at ~4 concurrent requests — higher values "
             "just use more memory with no speed gain.",
        key="pose_frame_concurrent",
    )
    st.caption(
        f"**{n_workers} worker(s) × {frame_concurrent} concurrent frames → Apple Vision (ANE).**  "
        f"Total in-flight: {n_workers * frame_concurrent} frames."
    )
else:
    st.caption(
        f"**{n_workers} process(es) → CPU (XNNPACK).**  "
        "Each process runs independently on the CPU cores."
    )

# ------------------------------------------------------------------ #
# Dual-view option (category-source side-by-side cameras)
# ------------------------------------------------------------------ #
split_dual = st.checkbox(
    "Split dual-view videos (category-source)",
    value=False,
    help="category-source videos have two camera angles side by side. "
         "Enable this to split each video and extract poses for both the "
         "60-degree (frontal) and 90-degree (profile) views separately. "
         "Output: {name}_60.pose and {name}_90.pose",
)

# Default crops for 1280x720 category-source videos (dark divider at ~col 640)
_DUAL_CROP_60 = (0, 0, 632, 720)      # left half — 60° frontal
_DUAL_CROP_90 = (648, 0, 632, 720)    # right half — 90° profile

# ------------------------------------------------------------------ #
# Video selection
# ------------------------------------------------------------------ #
st.subheader("Select videos to extract")

if split_dual:
    # Auto-filter to category-source videos only
    _kod_mask = inv["path"].str.contains("category-source", case=False, na=False)
    _kod_inv = inv[_kod_mask]
    if _kod_inv.empty:
        st.warning("No category-source videos found in inventory. Dual-view split requires category-source source videos.")
        st.stop()
    st.info(f"Filtered to **{len(_kod_inv)}** category-source videos for dual-view extraction.")

    # For dual-view, "done" means both _60.pose and _90.pose exist
    def _dual_done(p: str) -> bool:
        stem = Path(p).stem
        return (POSE_DIR / f"{stem}_60.pose").exists() and (POSE_DIR / f"{stem}_90.pose").exists()

    pending = _kod_inv[~_kod_inv["path"].apply(_dual_done)].copy()
    done = _kod_inv[_kod_inv["path"].apply(_dual_done)].copy()

if pending.empty and done.empty:
    st.success("No videos in inventory.")
    st.stop()

# Build a combined list: pending first, then done (for re-extract)
_sel_rows: list[dict] = []
for _, r in pending.iterrows():
    name = Path(str(r["path"])).name
    _sel_rows.append({
        "filename": name,
        "status": "pending",
        "duration_sec": r.get("duration_sec", 0),
        "fps": r.get("fps", 0),
        "file_size_mb": r.get("file_size_mb", 0),
        "_path": str(r["path"]),
    })
for _, r in done.iterrows():
    name = Path(str(r["path"])).name
    _sel_rows.append({
        "filename": name,
        "status": "done",
        "duration_sec": r.get("duration_sec", 0),
        "fps": r.get("fps", 0),
        "file_size_mb": r.get("file_size_mb", 0),
        "_path": str(r["path"]),
    })

all_filenames = [r["filename"] for r in _sel_rows]
pending_filenames = [r["filename"] for r in _sel_rows if r["status"] == "pending"]
_path_by_name = {r["filename"]: r["_path"] for r in _sel_rows}
_pending_set = set(pending_filenames)

# --- Selection mode: multiselect for small sets, radio for large ---
_MULTISELECT_THRESHOLD = 500

if len(all_filenames) <= _MULTISELECT_THRESHOLD:
    # Small inventory — use multiselect widget
    qcol1, qcol2, qcol3 = st.columns(3)
    with qcol1:
        if st.button(f"Select all pending ({len(pending_filenames)})", use_container_width=True):
            st.session_state["pose_video_multiselect"] = pending_filenames
            st.rerun()
    with qcol2:
        if st.button(f"Select all ({len(all_filenames)})", use_container_width=True):
            st.session_state["pose_video_multiselect"] = all_filenames
            st.rerun()
    with qcol3:
        if st.button("Clear selection", use_container_width=True):
            st.session_state["pose_video_multiselect"] = []
            st.rerun()

    if "pose_video_multiselect" not in st.session_state:
        st.session_state["pose_video_multiselect"] = pending_filenames

    selected_names = st.multiselect(
        "Videos to extract",
        options=all_filenames,
        help="Select one or more videos. Already-done videos will be re-extracted.",
        key="pose_video_multiselect",
    )
else:
    # Large inventory — radio selection to avoid DOM overload
    _selection_mode = st.radio(
        "Select videos",
        [
            f"All pending ({len(pending_filenames)})",
            f"All videos ({len(all_filenames)})",
        ],
        horizontal=True,
        key="pose_selection_mode",
    )
    if _selection_mode.startswith("All pending"):
        selected_names = pending_filenames
    else:
        selected_names = all_filenames

# Show selected info
n_selected = len(selected_names)
n_reextract = sum(1 for n in selected_names if n not in _pending_set)
if n_selected:
    info_parts = [f"**{n_selected}** video(s) selected"]
    if n_reextract:
        info_parts.append(f"{n_reextract} will be re-extracted (existing .pose replaced)")
    st.caption(" · ".join(info_parts))
else:
    st.info("Select at least one video to extract.")
    st.stop()

# ------------------------------------------------------------------ #
# Extract button
# ------------------------------------------------------------------ #
bcol1, bcol2 = st.columns([3, 2])
with bcol1:
    start_clicked = st.button(
        f"▶ Extract Selected ({n_selected})",
        type="primary", use_container_width=True,
    )
with bcol2:
    run_bg = st.checkbox(
        "Run in background",
        value=False,
        help="Extraction continues while you navigate to other pages. "
             "Return here to check progress or stop.",
    )

if not start_clicked:
    st.stop()

POSE_DIR.mkdir(parents=True, exist_ok=True)

if not _use_apple_vision:
    # Metal GPU and CPU both need MediaPipe models
    with st.spinner("Checking / downloading MediaPipe models (first run only)…"):
        try:
            ensure_models()
        except Exception as exc:
            st.error(f"Model download failed: {exc}")
            st.stop()

# Delete existing .pose files for re-extract selections
for name in selected_names:
    if name not in _pending_set:
        stem = Path(name).stem
        if split_dual:
            (POSE_DIR / f"{stem}_60.pose").unlink(missing_ok=True)
            (POSE_DIR / f"{stem}_90.pose").unlink(missing_ok=True)
        else:
            (POSE_DIR / f"{stem}.pose").unlink(missing_ok=True)

video_paths = [Path(_path_by_name[name]) for name in selected_names]

# Dual-view: detect actual video height for crop rectangles
if split_dual:
    import cv2 as _cv2
    _sample_cap = _cv2.VideoCapture(str(video_paths[0]))
    _vid_h = int(_sample_cap.get(_cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    _sample_cap.release()
    _crop_60 = (0, 0, 632, _vid_h)
    _crop_90 = (648, 0, 632, _vid_h)
    total = len(video_paths) * 2  # each video → 2 pose files
else:
    total = len(video_paths)

tmp_dir = Path(tempfile.mkdtemp(prefix="spj_pose_"))

# Start executor — NOT in a `with` block so it stays alive across reruns
# Select backend-specific worker, executor type, and third arg value
if _use_apple_vision:
    _worker_fn = _apple_batch_worker
    _ExecutorCls = ThreadPoolExecutor
    _third_arg = 0  # target_fps
elif _use_metal_gpu:
    _worker_fn = _metal_gpu_worker
    _ExecutorCls = ThreadPoolExecutor
    _third_arg = ""  # model_dir_str
else:
    _worker_fn = _batch_worker
    _ExecutorCls = ProcessPoolExecutor
    _third_arg = ""  # model_dir_str

if split_dual:
    # Two tasks per video: one for each crop (60° and 90°)
    # Each task gets a unique key based on output path for progress tracking
    args_list = []
    progress_files = {}
    _dual_task_labels: dict[str, str] = {}  # args_key → display label
    for vp in video_paths:
        for suffix, crop in [("_60", _crop_60), ("_90", _crop_90)]:
            out_path = POSE_DIR / f"{vp.stem}{suffix}.pose"
            task_key = str(out_path)  # unique per task
            pf = tmp_dir / f"{vp.stem}{suffix}.progress"
            progress_files[task_key] = pf
            _dual_task_labels[task_key] = f"{vp.name} → {vp.stem}{suffix}.pose"
            args_list.append(
                (str(vp), str(out_path), _third_arg, str(pf), crop)
            )
    if _use_apple_vision:
        args_list = [a + (frame_concurrent,) for a in args_list]
else:
    progress_files = {
        str(vp): tmp_dir / f"{vp.stem}.progress"
        for vp in video_paths
    }
    args_list = [
        (str(vp), str(POSE_DIR / f"{vp.stem}.pose"), _third_arg,
         str(progress_files[str(vp)]))
        for vp in video_paths
    ]
    if _use_apple_vision:
        args_list = [a + (frame_concurrent,) for a in args_list]

executor = _ExecutorCls(max_workers=n_workers)
if split_dual:
    # Use output path as unique task identifier (not input video — shared by 2 tasks)
    futures_map = {
        executor.submit(_worker_fn, args): Path(args[1])
        for args in args_list
    }
else:
    futures_map = {
        executor.submit(_worker_fn, args): Path(args[0])
        for args in args_list
    }

active_slots: dict[int, str] = {
    slot_i: str(vp)
    for slot_i, vp in enumerate(list(futures_map.values())[:n_workers])
}

st.session_state.update({
    "pose_state":           "running",
    "pose_executor":        executor,
    "pose_futures_map":     futures_map,
    "pose_pending_futs":    set(futures_map.keys()),
    "pose_completed_count": 0,
    "pose_log_lines":       [],
    "pose_total":           total,
    "pose_progress_files":  progress_files,
    "pose_active_slots":    active_slots,
    "pose_tmp_dir":         str(tmp_dir),
    "pose_bg_mode":         run_bg,
})

st.rerun()
