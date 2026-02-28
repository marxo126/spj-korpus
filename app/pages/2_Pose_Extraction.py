"""Pose Extraction page — parallel MediaPipe extraction using separate GPU processes."""
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from spj.inventory import pose_exists
from spj.pose import _batch_worker, ensure_models, recommend_workers

DATA_DIR = Path(__file__).parent.parent.parent / "data"
INVENTORY_CSV = DATA_DIR / "inventory.csv"
POSE_DIR = DATA_DIR / "pose"

st.header("🏃 Pose Extraction")
st.caption("Page 2/10 · Extract 543 body+hand+face keypoints from each video using MediaPipe.")
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
_zero_byte = [
    Path(r["path"]).name for _, r in inv.iterrows()
    if (POSE_DIR / f"{Path(r['path']).stem}.pose").exists()
    and (POSE_DIR / f"{Path(r['path']).stem}.pose").stat().st_size == 0
]
if _zero_byte:
    st.warning(
        f"**{len(_zero_byte)} pose file(s) are 0 bytes** (failed extraction). "
        f"They are counted as pending and will be re-extracted: "
        + ", ".join(f"`{n}`" for n in _zero_byte[:5])
        + ("…" if len(_zero_byte) > 5 else "")
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
        log_lines.append(line)
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

        # Clear extraction state, then rerun to show idle UI + success banner
        for key in _POSE_SS_KEYS:
            st.session_state.pop(key, None)
        st.rerun()

    # ── Continue ───────────────────────────────────────────────────
    # Background mode: refresh every 2s so the counter updates when the user
    # is watching this page, without blocking navigation elsewhere.
    time.sleep(2.0 if bg_mode else 0.05)
    st.rerun()

# ------------------------------------------------------------------ #
# Idle state — configuration UI
# ------------------------------------------------------------------ #
try:
    import psutil
    mem      = psutil.virtual_memory()
    free_gb  = mem.available / 1_073_741_824
    total_gb = mem.total     / 1_073_741_824
    suggested = recommend_workers(len(pending))
    st.caption(
        f"Free RAM: **{free_gb:.1f} GB** / {total_gb:.0f} GB  —  "
        f"suggested parallel workers: **{suggested}**"
    )
except ImportError:
    suggested = 2

n_workers = st.slider(
    "Parallel workers  (each = one GPU process)",
    min_value=1, max_value=8,
    value=min(suggested, max(1, len(pending))),
    help="Each worker is a separate OS process with its own Metal GPU context. "
         "More workers = more GPU cores in use simultaneously.",
)
st.caption(
    f"**{n_workers} worker(s) → CPU (XNNPACK) per process.**  "
    "MediaPipe's GPU delegate is officially supported on Ubuntu only — "
    "macOS GPU has known unbounded memory leaks (issues #5652, #6223). "
    "CPU + XNNPACK across the M4 Max's 16 performance cores is stable and fast."
)

# ------------------------------------------------------------------ #
# Pending table
# ------------------------------------------------------------------ #
st.subheader("Pending videos")
if pending.empty:
    st.success("All videos have been processed!")

if not pending.empty:
    show_cols = [c for c in ["filename", "duration_sec", "fps", "file_size_mb"]
                 if c in pending.columns]
    st.dataframe(pending[show_cols], use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ #
# Re-extract option for already-done videos
# ------------------------------------------------------------------ #
if not done.empty:
    with st.expander(f"🔄 Re-extract pose ({len(done)} done)", expanded=False):
        st.caption(
            "Select videos to re-extract. Their existing .pose files will be "
            "deleted and new ones generated."
        )
        reextract_names = [Path(str(p)).name for p in done["path"]]
        selected = st.multiselect(
            "Videos to re-extract",
            options=reextract_names,
            key="pose_reextract_sel",
        )
        if selected and st.button("🗑 Delete selected .pose files & re-extract"):
            for name in selected:
                stem = Path(name).stem
                pose_file = POSE_DIR / f"{stem}.pose"
                if pose_file.exists():
                    pose_file.unlink()
            # Refresh inventory
            inv["pose_extracted"] = inv["path"].apply(
                lambda p: pose_exists(Path(p), POSE_DIR)
            )
            st.session_state["inventory"] = inv
            st.rerun()

# ------------------------------------------------------------------ #
# Extract button
# ------------------------------------------------------------------ #
# Refresh pending after possible re-extract deletion
pending = inv[~inv["pose_extracted"]].copy()

if pending.empty:
    st.stop()

bcol1, bcol2 = st.columns([3, 2])
with bcol1:
    start_clicked = st.button(
        f"▶ Extract All ({len(pending)} pending)",
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

with st.spinner("Checking / downloading MediaPipe models (first run only)…"):
    try:
        ensure_models()
    except Exception as exc:
        st.error(f"Model download failed: {exc}")
        st.stop()

video_paths = [Path(r["path"]) for _, r in pending.iterrows()]
total = len(video_paths)

tmp_dir = Path(tempfile.mkdtemp(prefix="spj_pose_"))
progress_files = {
    str(vp): tmp_dir / f"{vp.stem}.progress"
    for vp in video_paths
}

args_list = [
    (str(vp), str(POSE_DIR / f"{vp.stem}.pose"), "", str(progress_files[str(vp)]))
    for vp in video_paths
]

# Start executor — NOT in a `with` block so it stays alive across reruns
executor    = ProcessPoolExecutor(max_workers=n_workers)
futures_map = {
    executor.submit(_batch_worker, args): Path(args[0])
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
