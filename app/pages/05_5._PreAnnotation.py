"""Pre-annotation page — motion-based sign boundary detection.

Loads .pose files, detects sign segment boundaries from wrist velocity,
and writes placeholder annotations ("?") into the AI tiers of the matching
EAF file.  Annotators then open the EAF in ELAN and fill in the actual
glosses — the timing segments are already there.

No trained model is needed at this stage.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from spj.eaf import AI_TIERS, get_tier_stats, load_eaf
from spj.inventory import pose_exists
from spj.preannotate import preannotate_eaf

DATA_DIR       = Path(__file__).parent.parent.parent / "data"
INVENTORY_CSV  = DATA_DIR / "inventory.csv"
POSE_DIR       = DATA_DIR / "pose"
ANNOTATIONS_DIR = DATA_DIR / "annotations"


def _eaf_path(video_path: Path) -> Path:
    return ANNOTATIONS_DIR / f"{Path(video_path).stem}.eaf"


def _ai_annotation_count(eaf_path: Path) -> int:
    """Total annotations already in the three AI tiers."""
    try:
        stats = get_tier_stats(load_eaf(eaf_path))
        return sum(stats.get(t, 0) for t in AI_TIERS)
    except Exception:
        return -1


st.header("5. 🤖 Pre-Annotation")
st.caption(
    "Page 5/10 · Detect sign boundaries from wrist motion, write timing segments into AI EAF tiers. "
    "No trained model required."
)
with st.expander("ℹ️ How to use this page", expanded=False):
    st.markdown("""
**Needs:**
- Pose files (page 2 — Pose Extraction)
- EAF files (page 3 — EAF Manager)

Both must exist for the same video.

**Steps:**
1. Leave detection settings at defaults (they work for most SPJ videos).
2. Click **▶ Pre-annotate pending videos**.
3. Open the resulting `.eaf` in ELAN.
4. In ELAN: the `AI_Gloss_RH` and `AI_Gloss_LH` tiers already have timing segments with "?".
   Replace each "?" with the actual ID-gloss (e.g. `WATER-1`).

**Creates:** Updates `data/annotations/*.eaf` — adds `AI_Gloss_RH`, `AI_Gloss_LH`, `AI_Confidence` tiers.
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
        st.warning("No inventory found. Go to the **Inventory** page first.")
        st.stop()

if inv.empty:
    st.warning("Inventory is empty.")
    st.stop()

# Refresh pose / EAF flags
inv["pose_extracted"] = inv["path"].apply(lambda p: pose_exists(Path(p), POSE_DIR))
inv["eaf_exists"]     = inv["path"].apply(lambda p: _eaf_path(Path(p)).exists())
st.session_state["inventory"] = inv

# Videos ready for pre-annotation: have .pose AND .eaf
ready = inv[inv["pose_extracted"] & inv["eaf_exists"]].copy()

if ready.empty:
    st.info(
        "No videos are ready yet. A video needs both a **.pose** file "
        "(from Pose Extraction) and an **.eaf** file (from EAF Manager) "
        "before pre-annotation can run."
    )
    st.stop()

# ------------------------------------------------------------------ #
# Per-video AI annotation status
# ------------------------------------------------------------------ #
rows = []
for _, row in ready.iterrows():
    vpath = Path(row["path"])
    ep    = _eaf_path(vpath)
    ai_n  = _ai_annotation_count(ep)
    rows.append({
        "filename":     vpath.name,
        "duration_sec": row.get("duration_sec", ""),
        "ai_segments":  ai_n if ai_n >= 0 else "error",
        "status":       "done" if ai_n > 0 else ("error" if ai_n < 0 else "pending"),
        "_path":        str(vpath),
        "_eaf":         str(ep),
    })

status_df = pd.DataFrame(rows)

pending_df = status_df[status_df["status"] == "pending"]
done_df    = status_df[status_df["status"] == "done"]

col1, col2, col3 = st.columns(3)
col1.metric("Ready (pose + EAF)", len(status_df))
col2.metric("Pre-annotated",      len(done_df))
col3.metric("Pending",            len(pending_df))

st.subheader("Status")
display_cols = ["filename", "duration_sec", "ai_segments", "status"]
st.dataframe(
    status_df[display_cols],
    use_container_width=True,
    hide_index=True,
)

# ------------------------------------------------------------------ #
# Detection settings
# ------------------------------------------------------------------ #
with st.expander("Detection settings", expanded=False):
    st.markdown(
        "These control how sensitive the motion detector is. "
        "The defaults work well for most SPJ signing videos."
    )
    c1, c2 = st.columns(2)
    with c1:
        smooth_sigma = st.slider(
            "Smoothing (sigma, frames)",
            min_value=1.0, max_value=10.0, value=3.0, step=0.5,
            help="Higher = smoother signal, fewer spurious short segments.",
        )
        motion_threshold = st.slider(
            "Motion threshold (0–1)",
            min_value=0.05, max_value=0.50, value=0.15, step=0.01,
            help="Fraction of peak motion above which a frame is considered active. "
                 "Lower catches more subtle movement; higher ignores small hand shifts.",
        )
        min_gap_ms = st.slider(
            "Min gap to split (ms)",
            min_value=40, max_value=500, value=80, step=10,
            help="Gaps shorter than this between two active segments are merged.",
        )
    with c2:
        min_duration_ms = st.slider(
            "Min segment duration (ms)",
            min_value=50, max_value=800, value=150, step=25,
            help="Segments shorter than this are discarded as noise.",
        )
        max_duration_ms = st.slider(
            "Max segment duration (ms)",
            min_value=500, max_value=10000, value=4000, step=250,
            help="Segments longer than this are discarded (likely mis-detections).",
        )

detection_params = dict(
    smooth_sigma     = smooth_sigma,
    motion_threshold = motion_threshold,
    min_duration_ms  = min_duration_ms,
    max_duration_ms  = max_duration_ms,
    min_gap_ms       = min_gap_ms,
)

# ------------------------------------------------------------------ #
# Action buttons
# ------------------------------------------------------------------ #
if pending_df.empty:
    st.success("All ready videos are pre-annotated.")
    overwrite = st.checkbox(
        "Re-annotate (overwrite existing AI annotations)",
        value=False,
    )
    target_df = status_df if overwrite else pd.DataFrame()
else:
    bcol1, bcol2 = st.columns([3, 2])
    with bcol1:
        run_pending = st.button(
            f"▶ Pre-annotate {len(pending_df)} pending video(s)",
            type="primary",
            use_container_width=True,
        )
    with bcol2:
        overwrite = st.checkbox(
            "Overwrite existing AI annotations",
            value=False,
            help="If checked, existing AI_Gloss_RH / AI_Gloss_LH / AI_Confidence "
                 "annotations are cleared before new ones are written.",
        )
    if not run_pending:
        st.stop()
    target_df = pending_df if not overwrite else status_df

# ------------------------------------------------------------------ #
# Run pre-annotation
# ------------------------------------------------------------------ #
total   = len(target_df)
bar     = st.progress(0.0, text=f"0 / {total} done")
log_out = st.empty()

log_lines: list[str] = []
errors:    list[str] = []

for i, (_, row) in enumerate(target_df.iterrows()):
    vpath    = Path(row["_path"])
    eaf_path = Path(row["_eaf"])
    pose_path = POSE_DIR / f"{vpath.stem}.pose"

    bar.progress((i + 1) / total, text=f"{i + 1} / {total} — {vpath.name}")

    try:
        result = preannotate_eaf(
            pose_path, eaf_path,
            overwrite=overwrite,
            **detection_params,
        )
        rh = result["rh_segments"]
        lh = result["lh_segments"]
        dur = result["duration_sec"]
        line = (
            f"✅ `{vpath.name}` — "
            f"RH: {rh} segments, LH: {lh} segments  ({dur}s)"
        )
    except Exception as exc:
        line = f"❌ `{vpath.name}` — {exc}"
        errors.append(line)

    log_lines.append(line)
    log_out.markdown("\n\n".join(log_lines[-20:]))

bar.progress(1.0, text=f"{total} / {total} done")

if errors:
    with st.expander(f"Errors ({len(errors)})"):
        for e in errors:
            st.markdown(e)

ok = total - len(errors)
st.success(
    f"Pre-annotation complete — {ok} / {total} videos updated. "
    "Open the EAF files in ELAN to review and fill in the glosses."
)
