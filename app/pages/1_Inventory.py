"""Inventory page — scan video directory and display metadata table."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make src/ importable when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from spj.inventory import build_inventory, scan_videos

DATA_DIR = Path(__file__).parent.parent.parent / "data"
INVENTORY_CSV = DATA_DIR / "inventory.csv"
VIDEO_DIR_DEFAULT = str(DATA_DIR / "videos")

st.header("📂 Video Inventory")
st.caption("Page 1/10 · Register your video files so all other pages can find them.")
with st.expander("ℹ️ How to use this page", expanded=False):
    st.markdown("""
**Needs:**
- Video files on disk (`data/videos/` or any folder)
- Optional: download videos first on the Download page (page 4)

**Steps:**
1. Enter the path to your video folder (default: `data/videos/`).
2. Click **🔍 Scan Videos** — reads metadata for every `.mp4` / `.mov` / `.mkv` found.
3. After downloading new videos, come back here and scan again.
4. Click **↩ Reload saved CSV** to restore the last scan without re-scanning.
5. Tick checkboxes → Remove to drop entries (files on disk are not deleted).

**Creates:** `data/inventory.csv` (one row per video)
""")

# --- Directory input ---
video_dir_input = st.text_input(
    "Video directory",
    value=st.session_state.get("video_dir", VIDEO_DIR_DEFAULT),
    help="Absolute path to the folder containing your source videos (scanned recursively).",
)

col_scan, col_reload = st.columns([1, 1])
scan_clicked = col_scan.button("🔍 Scan Videos", type="primary")
reload_clicked = col_reload.button("↩ Reload saved CSV")

if scan_clicked:
    video_dir = Path(video_dir_input)
    if not video_dir.exists():
        st.error(f"Directory does not exist: `{video_dir}`")
    else:
        st.session_state["video_dir"] = str(video_dir)
        with st.spinner("Scanning videos and probing metadata…"):
            pose_dir = DATA_DIR / "pose"
            df = build_inventory(video_dir, pose_dir)
        if df.empty:
            st.warning("No video files found in that directory.")
        else:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(INVENTORY_CSV, index=False)
            st.session_state["inventory"] = df
            st.success(f"Found {len(df)} videos. Inventory saved to `data/inventory.csv`.")

elif reload_clicked:
    if INVENTORY_CSV.exists():
        df = pd.read_csv(INVENTORY_CSV)
        st.session_state["inventory"] = df
        st.success("Inventory reloaded from CSV.")
    else:
        st.warning("No saved inventory found. Click *Scan Videos* first.")

# --- Display table ---
inv: pd.DataFrame | None = st.session_state.get("inventory")

if inv is not None and not inv.empty:
    st.subheader("Videos")

    # Display columns — show ✅/⚠️ 0 B/❌ for pose status
    display = inv.copy()
    if "pose_extracted" in display.columns:
        pose_dir = DATA_DIR / "pose"

        def _pose_label(row):
            if row["pose_extracted"]:
                return "✅"
            stem = Path(row["path"]).stem
            pf = pose_dir / f"{stem}.pose"
            if pf.exists() and pf.stat().st_size == 0:
                return "⚠️ 0 B"
            return "❌"

        display["pose"] = display.apply(_pose_label, axis=1)
    if "duration_sec" in display.columns:
        display["duration"] = display["duration_sec"].map(
            lambda s: f"{int(s // 60)}m {int(s % 60)}s" if pd.notna(s) else "—"
        )
    if "width" in display.columns and "height" in display.columns:
        display["resolution"] = display.apply(
            lambda r: f"{int(r['width'])}×{int(r['height'])}"
            if pd.notna(r["width"]) and pd.notna(r["height"])
            else "—",
            axis=1,
        )

    show_cols = [c for c in ["filename", "duration", "fps", "resolution",
                              "file_size_mb", "codec", "pose"]
                 if c in display.columns]

    display["_sel"] = False
    edited = st.data_editor(
        display[["_sel"] + show_cols],
        column_config={"_sel": st.column_config.CheckboxColumn("✓", default=False, width="small")},
        hide_index=True,
        use_container_width=True,
        key="inventory_table",
    )

    selected_idx = edited.index[edited["_sel"]].tolist()
    if selected_idx:
        st.caption(f"{len(selected_idx)} video(s) selected")
        rc1, rc2 = st.columns([2, 3])
        with rc1:
            if st.button(f"🗑 Remove {len(selected_idx)} from inventory", type="primary"):
                inv = inv.drop(index=selected_idx).reset_index(drop=True)
                inv.to_csv(INVENTORY_CSV, index=False)
                st.session_state["inventory"] = inv
                st.success(f"Removed {len(selected_idx)} video(s) from inventory.")
                st.rerun()
        with rc2:
            st.caption("Removes from inventory only — video files on disk are not touched.")

    # --- Summary stats ---
    st.subheader("Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total videos", len(inv))
    total_h = inv["duration_sec"].sum() / 3600 if "duration_sec" in inv.columns else 0
    col2.metric("Total duration", f"{total_h:.1f} h")
    n_pose = int(inv["pose_extracted"].sum()) if "pose_extracted" in inv.columns else 0
    col3.metric("Pose extracted", n_pose)
    col4.metric("Missing pose", len(inv) - n_pose)
else:
    st.info("No inventory loaded. Enter a video directory above and click **Scan Videos**.")
