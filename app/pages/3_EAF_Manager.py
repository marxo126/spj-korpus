"""EAF Manager page — create and inspect ELAN annotation files."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from spj.eaf import (
    ALL_TIERS,
    create_empty_eaf,
    get_tier_stats,
    load_eaf,
    save_eaf,
)
from spj.inventory import pose_exists

DATA_DIR = Path(__file__).parent.parent.parent / "data"
INVENTORY_CSV = DATA_DIR / "inventory.csv"
POSE_DIR = DATA_DIR / "pose"
ANNOTATIONS_DIR = DATA_DIR / "annotations"

st.header("🗂 EAF Manager")
st.caption("Page 3/10 · Create ELAN annotation files (.eaf) ready for human annotation.")
with st.expander("ℹ️ How to use this page", expanded=False):
    st.markdown("""
**Needs:**
- Pose files extracted (page 2 — Pose Extraction)

**Steps:**
1. Click **📝 Create EAFs** — generates one empty `.eaf` per video that has a `.pose` file.
2. After running Pre-Annotation (page 5), return here to see annotation counts per tier.
3. Download any `.eaf` and open it in ELAN to review / fill in glosses.

**Creates:** `data/annotations/<videoname>.eaf` (one per video)

**After:** Open `.eaf` in ELAN → fill in `S1_Gloss_RH` / `S1_Gloss_LH` tiers.
`AI_Gloss` tiers are populated by page 5 (Pre-Annotation).
""")


def eaf_path(video_path: Path) -> Path:
    return ANNOTATIONS_DIR / f"{video_path.stem}.eaf"


def eaf_exists(video_path: Path) -> bool:
    return eaf_path(Path(video_path)).exists()


# Load inventory
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

# Refresh flags
inv["pose_extracted"] = inv["path"].apply(lambda p: pose_exists(Path(p), POSE_DIR))
inv["eaf_exists"] = inv["path"].apply(lambda p: eaf_exists(Path(p)))
st.session_state["inventory"] = inv

# Videos that have pose but no EAF yet
needs_eaf = inv[inv["pose_extracted"] & ~inv["eaf_exists"]].copy()
has_eaf = inv[inv["eaf_exists"]].copy()

col1, col2, col3 = st.columns(3)
col1.metric("Has pose", int(inv["pose_extracted"].sum()))
col2.metric("EAFs created", len(has_eaf))
col3.metric("Ready to annotate (no EAF)", len(needs_eaf))

# ------------------------------------------------------------------ #
# Create EAFs section
# ------------------------------------------------------------------ #
st.subheader("Create EAFs")
if needs_eaf.empty:
    st.info("No videos need new EAFs right now.")
else:
    show_cols = [c for c in ["filename", "duration_sec", "fps"] if c in needs_eaf.columns]
    st.dataframe(needs_eaf[show_cols], use_container_width=True, hide_index=True)

    if st.button("📝 Create EAFs for all listed videos", type="primary"):
        ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
        created = 0
        errors: list[str] = []
        for _, row in needs_eaf.iterrows():
            vpath = Path(row["path"])
            out = eaf_path(vpath)
            try:
                eaf = create_empty_eaf(vpath, out)
                save_eaf(eaf, out)
                created += 1
            except Exception as exc:
                errors.append(f"**{vpath.name}**: {exc}")

        inv["eaf_exists"] = inv["path"].apply(lambda p: eaf_exists(Path(p)))
        st.session_state["inventory"] = inv

        if errors:
            with st.expander(f"Errors ({len(errors)})"):
                for e in errors:
                    st.markdown(e)
        st.success(f"Created {created} EAF files in `data/annotations/`.")
        st.rerun()

# ------------------------------------------------------------------ #
# Existing EAFs section
# ------------------------------------------------------------------ #
st.subheader("Existing EAFs")
if has_eaf.empty:
    st.info("No EAF files found yet.")
else:
    eaf_rows: list[dict] = []
    for _, row in has_eaf.iterrows():
        vpath = Path(row["path"])
        ep = eaf_path(vpath)
        entry: dict = {"filename": vpath.name, "eaf": ep.name}
        try:
            loaded = load_eaf(ep)
            stats = get_tier_stats(loaded)
            entry["total_annotations"] = sum(stats.values())
            for tier, count in stats.items():
                entry[tier] = count
        except Exception as exc:
            entry["total_annotations"] = f"Error: {exc}"
        eaf_rows.append(entry)

    eaf_df = pd.DataFrame(eaf_rows)
    show_eaf_cols = ["filename", "total_annotations"] + [
        t for t in ALL_TIERS if t in eaf_df.columns
    ]
    st.dataframe(
        eaf_df[[c for c in show_eaf_cols if c in eaf_df.columns]],
        use_container_width=True,
        hide_index=True,
    )

    # Download a single EAF
    st.subheader("Download EAF")
    eaf_names = [eaf_path(Path(r["path"])).name for _, r in has_eaf.iterrows()]
    selected_eaf = st.selectbox("Select file to download", eaf_names)
    if selected_eaf:
        eaf_file = ANNOTATIONS_DIR / selected_eaf
        if eaf_file.exists():
            with open(eaf_file, "rb") as f:
                st.download_button(
                    label=f"⬇ Download {selected_eaf}",
                    data=f.read(),
                    file_name=selected_eaf,
                    mime="application/xml",
                )
