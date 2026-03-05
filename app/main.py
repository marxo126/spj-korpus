"""SPJ-Korpus Streamlit dashboard — entry point.

Launch with:
    uv run streamlit run app/main.py
"""
import sys
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="SPJ-Korpus",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

DATA_DIR = Path(__file__).parent.parent / "data"
INVENTORY_CSV = DATA_DIR / "inventory.csv"

# Load inventory into session_state so all pages can share it
if "inventory" not in st.session_state:
    if INVENTORY_CSV.exists():
        st.session_state["inventory"] = pd.read_csv(INVENTORY_CSV)
    else:
        st.session_state["inventory"] = None

st.title("🤟 SPJ-Korpus Dashboard")
st.markdown(
    """
Welcome to the **Slovak Sign Language (SPJ) Corpus** management dashboard.

Use the sidebar to navigate through the 12-page pipeline:

**Data Preparation** (pages 1–4)

| Page | Purpose |
|---|---|
| 1. **Inventory** | Scan and browse source videos |
| 2. **Pose Extraction** | Extract 543 body+hand+face keypoints (MediaPipe) |
| 3. **EAF Manager** | Create ELAN annotation files (.eaf) |
| 4. **Download** *(optional)* | Download sign language videos from YouTube |

**Annotation** (pages 5–7)

| Page | Purpose |
|---|---|
| 5. **Pre-Annotation** | Detect sign boundaries from wrist motion (kinematic) |
| 6. **Subtitles** | Find or OCR-extract Slovak subtitle text |
| 7. **Training Data** | Align pose + subtitles, review segments, export .npz |

**Training & Deployment** (pages 8–12)

| Page | Purpose |
|---|---|
| 8. **Training** | Split data, train Transformer model, manage checkpoints |
| 9. **Evaluation** | Evaluate models on test split, compare performance |
| 10. **Inference** | Predict glosses on new videos, write to EAF |
| 11. **Assistant** | AI chat assistant with pipeline context |
| 12. **AI Review** | Review AI predictions, approve/correct/skip glosses |

**Active learning loop:** Page 10 (Inference) → Page 12 (AI Review) → Page 7 (re-export) → Page 8 (retrain).

**Getting started:** Go to the Inventory page, enter your video directory, and click *Scan Videos*.
"""
)

# Summary ribbon if inventory is loaded
inv = st.session_state.get("inventory")
if inv is not None and not inv.empty:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Videos", len(inv))
    total_h = inv["duration_sec"].sum() / 3600 if "duration_sec" in inv.columns else 0
    col2.metric("Total hours", f"{total_h:.1f} h")
    n_pose = inv["pose_extracted"].sum() if "pose_extracted" in inv.columns else 0
    col3.metric("Pose extracted", int(n_pose))
    col4.metric("Remaining", int(len(inv) - n_pose))

    # ── Pipeline status dashboard ──────────────────────────────────────
    st.divider()
    st.subheader("Pipeline Status")

    def _pipeline_counts() -> list[dict]:
        """Scan data/ directories and return per-step status rows."""
        from spj.inventory import pose_exists
        from spj.ocr_subtitles import _vtt_has_cues

        pose_dir = DATA_DIR / "pose"
        subtitles_dir = DATA_DIR / "subtitles"
        annotations_dir = DATA_DIR / "annotations"
        training_dir = DATA_DIR / "training"
        models_dir = DATA_DIR / "models"

        n_videos = len(inv)
        stems = [Path(str(p)).stem for p in inv["path"]]

        # Pose: count valid (non-empty) files
        n_pose_valid = sum(
            1 for s in stems
            if (pose_dir / f"{s}.pose").exists()
            and (pose_dir / f"{s}.pose").stat().st_size > 0
        )
        n_pose_zero = sum(
            1 for s in stems
            if (pose_dir / f"{s}.pose").exists()
            and (pose_dir / f"{s}.pose").stat().st_size == 0
        )

        # EAF files
        n_eaf = sum(1 for s in stems if (annotations_dir / f"{s}.eaf").exists())

        # Subtitles: count files with actual cues
        n_sub = 0
        n_sub_empty = 0
        for s in stems:
            vtt = subtitles_dir / f"{s}.vtt"
            if vtt.exists():
                if _vtt_has_cues(vtt):
                    n_sub += 1
                else:
                    n_sub_empty += 1
            # Also check soft VTTs next to videos
            elif any(True for _ in []):
                pass  # soft VTTs counted separately by find_soft_vtt

        # Training data
        alignment_csv = training_dir / "alignment.csv"
        n_aligned = 0
        n_approved = 0
        if alignment_csv.exists():
            try:
                adf = pd.read_csv(alignment_csv)
                n_aligned = len(adf)
                n_approved = int((adf.get("status", pd.Series()) == "approved").sum())
            except Exception:
                pass

        # Models
        n_checkpoints = len(list(models_dir.glob("*.pt"))) if models_dir.exists() else 0

        def _status(done, total):
            if total == 0:
                return "pending"
            if done >= total:
                return "done"
            if done > 0:
                return "partial"
            return "pending"

        rows = [
            {"Step": "1. Inventory", "Count": f"{n_videos} videos", "Status": "done" if n_videos > 0 else "pending"},
            {"Step": "2. Pose", "Count": f"{n_pose_valid}/{n_videos} valid", "Status": _status(n_pose_valid, n_videos)},
            {"Step": "3. EAF files", "Count": f"{n_eaf}/{n_pose_valid}", "Status": _status(n_eaf, n_pose_valid)},
            {"Step": "5. Pre-Annotation", "Count": "—", "Status": "—"},
            {"Step": "6. Subtitles", "Count": f"{n_sub}/{n_videos} valid", "Status": _status(n_sub, n_videos)},
            {"Step": "7. Training Data", "Count": f"{n_approved} approved / {n_aligned} aligned", "Status": _status(n_approved, max(n_aligned, 1))},
            {"Step": "8–10. Models", "Count": f"{n_checkpoints} checkpoint(s)", "Status": "done" if n_checkpoints > 0 else "pending"},
        ]

        # Add warnings
        warnings = []
        if n_pose_zero > 0:
            warnings.append(f"**{n_pose_zero}** pose file(s) are **0 bytes** (failed extraction) — re-run Pose Extraction.")
        if n_sub_empty > 0:
            warnings.append(f"**{n_sub_empty}** subtitle file(s) are **empty stubs** (no cues) — re-run Subtitles.")

        return rows, warnings

    try:
        status_rows, status_warnings = _pipeline_counts()
        st.dataframe(
            pd.DataFrame(status_rows),
            hide_index=True,
            use_container_width=True,
        )
        for w in status_warnings:
            st.warning(w)
    except Exception as exc:
        st.caption(f"Could not compute pipeline status: {exc}")

else:
    st.info("No inventory loaded yet. Go to the **Inventory** page to scan your video directory.")
