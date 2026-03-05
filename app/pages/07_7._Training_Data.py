"""Training Data page — pose-subtitle alignment, visual review, NPZ export.

Tab 1 (Align):   build alignment table from .pose + .vtt pairs
Tab 2 (Review):  frame-by-frame visual review with approve / skip / flag
Tab 3 (Sign-Word): sign-level pairing within approved segments
Tab 4 (Export):  export approved segments as float16 .npz training files
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from spj.glossary import Glossary, load_glossary, save_glossary, tokenize_slovak
from spj.training_data import (
    PST_AUTO_SUGGESTED,
    PST_PAIRED,
    PST_PENDING,
    PST_SKIPPED,
    build_alignment_table,
    create_manual_pairing,
    detect_signs_in_segment,
    export_segment_npz,
    export_sign_npz,
    extract_video_segment,
    filter_quality_labels,
    harvest_eaf_batch,
    import_single_sign_videos,
    load_alignment_csv,
    load_pairings_csv,
    normalize_label,
    pose_animation_html,
    pose_frame_figure,
    save_alignment_csv,
    save_pairings_csv,
    suggest_sign_pairings,
    synced_video_pose_html,
    write_training_config,
)

# Segment status constants
ST_PENDING  = "pending"
ST_APPROVED = "approved"
ST_SKIPPED  = "skipped"
ST_FLAGGED  = "flagged"

DATA_DIR      = Path(__file__).parent.parent.parent / "data"
INVENTORY_CSV = DATA_DIR / "inventory.csv"
POSE_DIR      = DATA_DIR / "pose"
SUBTITLES_DIR = DATA_DIR / "subtitles"
TRAINING_DIR  = DATA_DIR / "training"
ALIGNMENT_CSV = TRAINING_DIR / "alignment.csv"
PAIRINGS_CSV  = TRAINING_DIR / "pairings.csv"
GLOSSARY_JSON = TRAINING_DIR / "glossary.json"
MODELS_DIR    = DATA_DIR / "models"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_ms(ms: int) -> str:
    m   = ms // 60_000
    s   = (ms % 60_000) // 1_000
    rem = ms % 1_000
    return f"{m:02d}:{s:02d}.{rem:03d}"


def _video_aspect(video_path: str) -> float:
    """Return height/width ratio from inventory, or 0 if unknown."""
    inv = st.session_state.get("inventory")
    if inv is None:
        return 0
    match = inv[inv["path"] == video_path]
    if match.empty:
        name = Path(video_path).name
        match = inv[inv["filename"] == name] if "filename" in inv.columns else match
    if match.empty:
        return 0
    row = match.iloc[0]
    w_val = row.get("width")
    h_val = row.get("height")
    if pd.isna(w_val) or pd.isna(h_val):
        return 0
    w, h = int(w_val), int(h_val)
    return h / w if w > 0 else 0


_CLR_MAPPED  = ("#2d5a2d", "#a8e6a8")
_CLR_UNKNOWN = ("#5a5a2d", "#e6e6a8")


def _word_span(text: str, bg: str, fg: str, tooltip: str = "") -> str:
    title = f'title="{tooltip}" ' if tooltip else ""
    return (
        f'<span {title}style="background:{bg};color:{fg};'
        f'padding:2px 5px;border-radius:4px;'
        f'margin:1px;display:inline-block;">{text}</span>'
    )



def _get_glossary() -> Glossary:
    if "glossary" not in st.session_state:
        st.session_state["glossary"] = load_glossary(GLOSSARY_JSON)
    return st.session_state["glossary"]


def _get_pairings_df() -> pd.DataFrame:
    if "td_pairings_df" in st.session_state:
        return st.session_state["td_pairings_df"]
    if PAIRINGS_CSV.exists():
        pdf = load_pairings_csv(PAIRINGS_CSV)
        st.session_state["td_pairings_df"] = pdf
        return pdf
    return pd.DataFrame()


def _save_pairings(pairings_df: pd.DataFrame) -> None:
    """Save pairings to CSV + session state."""
    save_pairings_csv(pairings_df, PAIRINGS_CSV)
    st.session_state["td_pairings_df"] = pairings_df


def _render_glossary_html(word_matches: list[dict]) -> str:
    """Build color-coded HTML spans for glossary-matched words."""
    parts = []
    for wm in word_matches:
        if wm["mapped"]:
            tip = ", ".join(wm["glosses"])
            parts.append(_word_span(wm["raw"], *_CLR_MAPPED, tip))
        else:
            parts.append(_word_span(wm["raw"], *_CLR_UNKNOWN))
    return " ".join(parts)


# Pairing status icons (module-level constant, not rebuilt per iteration)
_PAIRING_ICONS = {
    PST_PENDING: "🟡",
    PST_PAIRED: "🟢",
    PST_SKIPPED: "⚪",
    PST_AUTO_SUGGESTED: "🤖",
}


@st.cache_resource
def _load_pose_cached(pose_path_str: str):
    """Load pose arrays once per video path, shared across all reruns."""
    from spj.preannotate import load_pose_arrays
    return load_pose_arrays(Path(pose_path_str))


@st.cache_data(max_entries=5, ttl=600)
def _extract_segment_cached(
    video_path_str: str, start_ms: int, end_ms: int,
) -> bytes | None:
    """Extract video segment via ffmpeg, cached (max 5 clips, 10 min TTL)."""
    return extract_video_segment(Path(video_path_str), start_ms, end_ms)


@st.cache_resource
def _load_model_cached(ckpt_path: str):
    """Load model checkpoint once, shared across reruns."""
    from spj.trainer import load_checkpoint
    return load_checkpoint(Path(ckpt_path))


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.header("7. 🎓 Training Data")
st.caption(
    "Align .pose files with subtitles or import glossed clips → review → "
    "export approved segments as float16 .npz files for SignBERT / OpenHands."
)
st.caption("Page 7/10 · Align pose + subtitles, review each segment, export .npz training files.")
with st.expander("ℹ️ How to use this page", expanded=False):
    st.markdown("""
**Needs:**
- Pose files (page 2 — Pose Extraction)
- Subtitle `.vtt` files (page 6) **OR** glossed single-sign video clips (partner-dictnary videos)

**Steps:**

**Tab 📐 Align**
1. **Subtitle-based:** Click **📐 Align all pairs** — matches subtitle timestamps to pose frames.
2. **Glossed clips:** Click **📥 Import glossed videos** — imports single-sign clips where
   each file = one sign (label parsed from filename, no subtitles needed).

**Tab 👁 Review** *(do after Align)*
2. Filter by *Pending* to see unreviewed segments.
3. Watch the video clip, check the pose stick figure with the frame slider.
4. Edit the text if OCR made a mistake.
5. Click **✅ Approve** (good), **⏭ Skip** (unusable), or **🚩 Flag** (needs later review).
   The page auto-advances to the next pending segment.

**Tab 📦 Export** *(do after Review)*
6. Click **📦 Export approved segments** — saves one `.npz` per approved segment.
7. Click **📄 Export CSV manifest** — saves `manifest.csv` for the dataset loader.

**Creates:** `data/training/export/*.npz` + `manifest.csv` + `training_config.json`
""")

# ---------------------------------------------------------------------------
# Load / cache alignment dataframe
# ---------------------------------------------------------------------------

def _get_alignment_df() -> pd.DataFrame | None:
    if "td_align_df" in st.session_state:
        return st.session_state["td_align_df"]
    if ALIGNMENT_CSV.exists():
        df = load_alignment_csv(ALIGNMENT_CSV)
        st.session_state["td_align_df"] = df
        return df
    return None


df = _get_alignment_df()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_align, tab_review, tab_signword, tab_export = st.tabs(
    ["📐 Align", "👁 Review", "✂ Sign-Word", "📦 Export"]
)

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — ALIGN
# ══════════════════════════════════════════════════════════════════════════
with tab_align:
    inv: pd.DataFrame | None = st.session_state.get("inventory")
    if inv is None:
        if INVENTORY_CSV.exists():
            inv = pd.read_csv(INVENTORY_CSV)
            st.session_state["inventory"] = inv

    if inv is None:
        st.warning("No inventory found. Go to the **Inventory** page first.")
    elif inv.empty:
        st.warning("Inventory is empty.")
    else:
        from spj.ocr_subtitles import get_subtitle_status

        stems = [Path(str(p)).stem for p in inv["path"]]
        has_pose = [
            (POSE_DIR / f"{s}.pose").exists()
            and (POSE_DIR / f"{s}.pose").stat().st_size > 0
            for s in stems
        ]
        has_vtt = [
            get_subtitle_status(Path(str(p)), SUBTITLES_DIR)["vtt_path"] is not None
            for p in inv["path"]
        ]

        inv_aug             = inv.copy()
        inv_aug["has_pose"] = has_pose
        inv_aug["has_vtt"]  = has_vtt
        inv_aug["paired"]   = inv_aug["has_pose"] & inv_aug["has_vtt"]

        paired_df   = inv_aug[inv_aug["paired"]]
        unpaired_df = inv_aug[~inv_aug["paired"]]

        c1, c2, c3 = st.columns(3)
        c1.metric("Total videos",        len(inv_aug))
        c2.metric("Paired (pose + vtt)", len(paired_df))
        c3.metric("Unpaired",            len(unpaired_df))

        if not paired_df.empty:
            st.subheader("Paired videos (ready to align)")
            st.dataframe(
                paired_df.assign(
                    filename=paired_df["path"].apply(lambda p: Path(p).name)
                )[["filename", "has_pose", "has_vtt"]],
                hide_index=True,
                use_container_width=True,
            )

        if not unpaired_df.empty:
            with st.expander(
                f"Unpaired videos ({len(unpaired_df)}) — missing .pose or .vtt"
            ):
                st.dataframe(
                    unpaired_df.assign(
                        filename=unpaired_df["path"].apply(lambda p: Path(p).name)
                    )[["filename", "has_pose", "has_vtt"]],
                    hide_index=True,
                    use_container_width=True,
                )

        if paired_df.empty:
            st.info(
                "No videos have both a .pose file and a .vtt file yet. "
                "Run Pose Extraction (page 2) and Subtitle Extraction (page 6) first."
            )
        else:
            with st.expander("⚙️ Alignment settings", expanded=False):
                ac1, ac2 = st.columns(2)
                with ac1:
                    dedup_enabled = st.checkbox(
                        "Merge duplicate OCR cues", value=True, key="td_dedup",
                    )
                    max_gap = st.number_input(
                        "Max gap between cues (ms)", value=2000, step=500,
                        key="td_max_gap",
                    )
                with ac2:
                    merge_short = st.checkbox(
                        "Merge short segments", value=True, key="td_merge_short",
                    )
                    min_seg = st.number_input(
                        "Min segment duration (ms)", value=1000, step=250,
                        key="td_min_seg",
                    )

            if st.button("📐 Align all pairs", type="primary"):
                with st.spinner("Aligning pose ↔ subtitles …"):
                    existing = st.session_state.get("td_align_df")
                    new_df, raw_cues, merged_segs = build_alignment_table(
                        paired_df,
                        POSE_DIR,
                        SUBTITLES_DIR,
                        existing_df=existing,
                        dedup=dedup_enabled,
                        merge_short=merge_short,
                        max_gap_ms=int(max_gap),
                        min_segment_ms=int(min_seg),
                    )

                if new_df.empty:
                    st.warning(
                        "No new segments produced — all pairs already aligned "
                        "or no subtitle cues found."
                    )
                else:
                    save_alignment_csv(new_df, ALIGNMENT_CSV)
                    st.session_state["td_align_df"] = new_df
                    df = new_df

                    if raw_cues > 0:
                        st.info(
                            f"VTT dedup: **{raw_cues}** raw cues → "
                            f"**{merged_segs}** segments "
                            f"({raw_cues - merged_segs} merged)"
                        )

                    total_segs  = len(new_df)
                    total_dur_s = ((new_df["end_ms"] - new_df["start_ms"]) / 1000).sum()
                    avg_dur_s   = total_dur_s / max(1, total_segs)

                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("Total segments", total_segs)
                    mc2.metric("Total duration", f"{total_dur_s / 60:.1f} min")
                    mc3.metric("Avg segment",    f"{avg_dur_s:.1f} s")

                    st.success(f"Alignment saved to `{ALIGNMENT_CSV}`")

    # ------------------------------------------------------------------
    # Import glossed single-sign videos (partner-dictnary clips, etc.)
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Import glossed videos (no subtitles needed)")
    st.caption(
        "Single-sign video clips where each file = one sign. "
        "Label is parsed from the filename: `{word_id}_{translation}.mp4`."
    )

    # Find subdirectories in data/videos/ that have matching .pose files
    VIDEO_DIR = DATA_DIR / "videos"
    _VIDEO_EXTS = ("*.mp4", "*.mkv", "*.webm", "*.mov")
    _gloss_dirs: list[tuple[str, bool]] = []  # (name, needs_recursive)
    if VIDEO_DIR.exists():
        for d in sorted(VIDEO_DIR.iterdir()):
            if d.is_dir():
                flat = any(d.glob(ext) for ext in _VIDEO_EXTS)
                if flat:
                    _gloss_dirs.append((d.name, False))
                elif any(d.rglob(_VIDEO_EXTS[0])):  # quick nested check
                    _gloss_dirs.append((d.name, True))

    if _gloss_dirs:
        sel_idx = st.selectbox(
            "Video folder",
            options=range(len(_gloss_dirs)),
            format_func=lambda i: _gloss_dirs[i][0],
            help="Subfolder inside data/videos/ containing glossed clips.",
            key="td_gloss_import_dir",
        )
        sel_dir, _needs_recursive = _gloss_dirs[sel_idx]
        gloss_video_dir = VIDEO_DIR / sel_dir
        _glob_fn = gloss_video_dir.rglob if _needs_recursive else gloss_video_dir.glob

        # Count how many have poses — batch check via set lookup
        _g_vids = sorted(
            f for ext in _VIDEO_EXTS for f in _glob_fn(ext)
        )
        _pose_stems = {f.stem for f in POSE_DIR.glob("*.pose") if f.stat().st_size > 0}
        _g_with_pose = [v for v in _g_vids if v.stem in _pose_stems]
        _mode = " (recursive)" if _needs_recursive else ""
        st.caption(
            f"**{len(_g_vids)}** video(s) in `{sel_dir}/`{_mode}, "
            f"**{len(_g_with_pose)}** with pose extracted"
        )

        icol1, icol2 = st.columns(2)
        with icol1:
            import_status = st.radio(
                "Import status",
                [ST_APPROVED, ST_PENDING],
                horizontal=True,
                help="'approved' auto-approves clean single-sign clips. "
                     "'pending' lets you review each one first.",
                key="td_gloss_import_status",
            )
        with icol2:
            split_words = st.checkbox(
                "Split multi-word labels",
                value=True,
                help="When a label has multiple words (e.g. 'baranie mäso'), "
                     "also create one segment per word (full phrase + each word). "
                     "Word segments are set to 'pending' for manual review.",
                key="td_gloss_split_words",
            )

        if st.button(
            f"📥 Import {len(_g_with_pose)} glossed videos",
            type="primary",
            disabled=len(_g_with_pose) == 0,
        ):
            with st.spinner("Importing glossed videos…"):
                existing = st.session_state.get("td_align_df")
                new_df, n_imported = import_single_sign_videos(
                    pose_dir=POSE_DIR,
                    video_dir=gloss_video_dir,
                    existing_df=existing,
                    status=import_status,
                    split_words=split_words,
                    recursive=_needs_recursive,
                )

            if n_imported == 0:
                st.warning("No new videos to import (all already in alignment table).")
            else:
                save_alignment_csv(new_df, ALIGNMENT_CSV)
                st.session_state["td_align_df"] = new_df
                df = new_df
                st.success(
                    f"Imported **{n_imported}** glossed video(s) as '{import_status}'. "
                    f"Total segments: **{len(new_df)}**."
                )
    else:
        st.info("No video subfolders found in `data/videos/`.")

    # Summary of current alignment (shown even without inventory)
    if df is not None and not df.empty:
        st.divider()
        st.subheader("Current alignment")
        sc = df["status"].value_counts()
        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric("Total",    len(df))
        cc2.metric("Pending",  int(sc.get(ST_PENDING,  0)))
        cc3.metric("Approved", int(sc.get(ST_APPROVED, 0)))
        cc4.metric("Skipped",  int(sc.get(ST_SKIPPED,  0)))

# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — REVIEW
# ══════════════════════════════════════════════════════════════════════════
with tab_review:
    df = st.session_state.get("td_align_df")

    if df is None or df.empty:
        st.info("No alignment data yet. Run alignment in the **Align** tab first.")
    else:
        # Filter
        filter_opt = st.radio(
            "Show",
            ["All", "Pending", "Approved", "Skipped", "Flagged"],
            horizontal=True,
            key="td_filter",
        )
        _fmap = {
            "All": None, "Pending": ST_PENDING,
            "Approved": ST_APPROVED, "Skipped": ST_SKIPPED, "Flagged": ST_FLAGGED,
        }
        fval     = _fmap[filter_opt]
        filtered = df if fval is None else df[df["status"] == fval]

        if filtered.empty:
            st.info(f"No segments with status '{fval}'.")
        else:
            # ── Video selector ─────────────────────────────────────
            filtered = filtered.copy()
            filtered["_vstem"] = filtered["video_path"].apply(
                lambda p: Path(p).stem
            )

            # Per-video stats, sorted by pending count (most work first)
            _vstats = (
                filtered.groupby("_vstem")
                .agg(
                    n_segs=("segment_id", "count"),
                    n_pending=("status", lambda x: (x == ST_PENDING).sum()),
                )
                .reset_index()
                .sort_values(["n_pending", "_vstem"], ascending=[False, True])
            )
            video_stems = _vstats["_vstem"].tolist()
            _vlabels = {
                r["_vstem"]: (
                    f"{r['_vstem'][:55]}{'…' if len(r['_vstem']) > 55 else ''}"
                    f"  ({r.n_pending} pending / {r.n_segs} total)"
                )
                for _, r in _vstats.iterrows()
            }

            # Validate current video selection
            cur_video = st.session_state.get("td_current_video")
            if cur_video not in video_stems:
                cur_video = video_stems[0]

            prev_video = cur_video
            cur_video = st.selectbox(
                "Video",
                video_stems,
                index=video_stems.index(cur_video),
                format_func=lambda v: _vlabels.get(v, v),
                key="td_current_video",
            )

            # Reset segment selection when video changes
            if cur_video != prev_video:
                st.session_state.pop("td_current_seg_id", None)

            # Filter to selected video
            video_df = filtered[filtered["_vstem"] == cur_video]

            # Per-video progress bar
            _vrow = _vstats[_vstats["_vstem"] == cur_video].iloc[0]
            n_reviewed = int(_vrow.n_segs - _vrow.n_pending)
            st.progress(
                n_reviewed / max(1, int(_vrow.n_segs)),
                text=f"{n_reviewed}/{int(_vrow.n_segs)} reviewed",
            )

            # Warn about single-cue entire-video segments
            if len(video_df) == 1:
                dur_ms = int(video_df.iloc[0]["end_ms"]) - int(
                    video_df.iloc[0]["start_ms"]
                )
                if dur_ms > 60_000:
                    st.warning(
                        f"This video has a single subtitle cue spanning "
                        f"{dur_ms / 1000:.0f}s (the entire video). This "
                        f"typically means soft subtitles. Consider "
                        f"**skipping** or re-extracting subtitles with "
                        f"OCR (page 6)."
                    )

            if video_df.empty:
                st.info("No segments match the filter for this video.")
            else:
                # ── Segment selector (scoped to video) ─────────────
                seg_ids = video_df["segment_id"].tolist()
                id_to_row = {
                    row["segment_id"]: row
                    for _, row in video_df.iterrows()
                }
                _icons = {
                    ST_PENDING: "🟡", ST_APPROVED: "🟢",
                    ST_SKIPPED: "⚪", ST_FLAGGED: "🔴",
                }

                def _seg_label_short(sid):
                    r = id_to_row[sid]
                    ts = (
                        f"{_fmt_ms(int(r['start_ms']))}"
                        f"→{_fmt_ms(int(r['end_ms']))}"
                    )
                    text = str(r["text"])[:50]
                    icon = _icons.get(str(r["status"]), "⚫")
                    return f"{icon} {ts} | {text}"

                # Resolve pending navigation (buttons set _td_nav_target
                # instead of writing directly to the widget-bound key).
                _nav = st.session_state.pop("_td_nav_target", None)
                if _nav is not None and _nav in seg_ids:
                    # Programmatic navigation: set widget key directly
                    st.session_state["td_current_seg_id"] = _nav

                # Resolve index for initial render (key not yet in state)
                _seg_index = 0
                if (
                    "td_current_seg_id" in st.session_state
                    and st.session_state["td_current_seg_id"] in seg_ids
                ):
                    _seg_index = seg_ids.index(
                        st.session_state["td_current_seg_id"]
                    )

                cur_id = st.selectbox(
                    "Segment",
                    seg_ids,
                    index=_seg_index,
                    format_func=_seg_label_short,
                    key="td_current_seg_id",
                )

                cur_row = id_to_row[cur_id]
                df_idx  = df.index[df["segment_id"] == cur_id][0]
                cur_pos = seg_ids.index(cur_id)

                # ── Reviewed timestamps (from session state) ─────────
                trim_key = f"trim_{cur_id}"
                if trim_key not in st.session_state:
                    st.session_state[trim_key] = (
                        int(cur_row["reviewed_start_ms"]),
                        int(cur_row["reviewed_end_ms"]),
                    )

                rev_start, rev_end = st.session_state[trim_key]

                # ── Load pose + compute frame range ───────────────────
                r_n_frames    = 0
                pose_loaded   = False
                is_zero_frames = True
                p_fps         = 25.0

                try:
                    p_data, p_conf, p_fps = _load_pose_cached(
                        str(cur_row["pose_path"])
                    )
                    T = p_data.shape[0]
                    r_frame_start = min(
                        int(rev_start * p_fps / 1000), T - 1
                    )
                    r_frame_end = min(
                        int(rev_end * p_fps / 1000), T - 1
                    )
                    r_n_frames    = max(0, r_frame_end - r_frame_start)
                    is_zero_frames = r_n_frames == 0
                    pose_loaded   = True
                except Exception as exc:
                    st.error(f"Pose load error: {exc}")

                # ── Segment info ──────────────────────────────────────
                if pose_loaded and r_n_frames > 0:
                    st.caption(
                        f"**{p_fps:.0f} fps** · "
                        f"{r_n_frames} frames · "
                        f"segment: {_fmt_ms(rev_start)} → "
                        f"{_fmt_ms(rev_end)} · "
                        f"duration: {(rev_end - rev_start) / 1000:.1f}s"
                    )
                elif is_zero_frames and pose_loaded:
                    st.warning(
                        "Segment has 0 frames at reviewed timestamps."
                    )

                # ── Subtitle text (prominent display) ─────────────────
                tx_key = f"tx_{cur_id}"
                if tx_key not in st.session_state:
                    default = (
                        str(cur_row["reviewed_text"]).strip()
                        or str(cur_row["text"])
                    )
                    st.session_state[tx_key] = default

                subtitle_text = st.session_state[tx_key]
                orig_start = int(cur_row["start_ms"])
                orig_end   = int(cur_row["end_ms"])
                st.markdown(
                    f'<div style="text-align:center; font-size:1.2em; '
                    f'padding:10px 16px; '
                    f'background:rgba(255,255,255,0.05); '
                    f'border-radius:8px; '
                    f'border-left:3px solid #ff6b6b; margin:4px 0;">'
                    f'<b>{subtitle_text}</b>'
                    f'<br><small style="opacity:0.5;">'
                    f'{_fmt_ms(orig_start)} → {_fmt_ms(orig_end)}'
                    f'</small></div>',
                    unsafe_allow_html=True,
                )

                # ── Sign-Word Glossary ─────────────────────────────────
                glossary = _get_glossary()
                word_matches = glossary.match_sentence(subtitle_text)
                n_mapped = sum(1 for w in word_matches if w["mapped"])
                n_total = len(word_matches)

                with st.expander(
                    f"Sign-Word Glossary ({n_mapped}/{n_total} words mapped)",
                    expanded=False,
                ):
                    # Color-coded sentence
                    if word_matches:
                        bg_m, fg_m = _CLR_MAPPED
                        bg_u, fg_u = _CLR_UNKNOWN
                        st.markdown(
                            '<div style="line-height:2.2;margin-bottom:8px;">'
                            + _render_glossary_html(word_matches) + "</div>"
                            + f'<small><span style="background:{bg_m};'
                            f'color:{fg_m};padding:1px 6px;border-radius:3px;">'
                            f'mapped</span> '
                            f'<span style="background:{bg_u};color:{fg_u};'
                            f'padding:1px 6px;border-radius:3px;">'
                            f'unknown</span></small>',
                            unsafe_allow_html=True,
                        )

                    # Word selector
                    if word_matches:
                        word_options = []
                        for i, wm in enumerate(word_matches):
                            if wm["mapped"]:
                                label = (
                                    f"[OK] {wm['raw']} -> "
                                    f"{', '.join(wm['glosses'])}"
                                )
                            else:
                                label = f"[??] {wm['raw']}"
                            word_options.append((i, label))

                        sel_idx = st.selectbox(
                            "Select word",
                            range(len(word_options)),
                            format_func=lambda i: word_options[i][1],
                            key=f"gw_sel_{cur_id}",
                        )
                        sel_word = word_matches[sel_idx]

                        if sel_word["mapped"]:
                            # Show gloss details
                            for gid in sel_word["glosses"]:
                                entry = glossary.get_entry(gid)
                                if entry:
                                    st.markdown(
                                        f"**{gid}** — lemma: "
                                        f"*{entry['lemma']}* "
                                        f"({entry.get('pos', '')})"
                                    )
                                    st.caption(
                                        "Forms: "
                                        + ", ".join(entry["forms"][:12])
                                    )
                                    # Add extra form
                                    new_form = st.text_input(
                                        "Add form",
                                        key=f"gw_af_{cur_id}_{gid}",
                                        placeholder="e.g. vodách",
                                    )
                                    if new_form and st.button(
                                        f"Add form to {gid}",
                                        key=f"gw_afb_{cur_id}_{gid}",
                                    ):
                                        glossary.add_form(gid, new_form)
                                        save_glossary(glossary, GLOSSARY_JSON)
                                        st.rerun()
                        else:
                            # Unknown word — offer to create gloss
                            raw_w = sel_word["raw"]
                            norm_w = sel_word["normalized"]
                            base_upper = norm_w.upper()
                            suggested = glossary.suggest_next_id(base_upper)

                            gc1, gc2 = st.columns(2)
                            with gc1:
                                new_gid = st.text_input(
                                    "Gloss ID",
                                    value=suggested,
                                    key=f"gw_nid_{cur_id}_{sel_idx}",
                                )
                            with gc2:
                                new_pos = st.selectbox(
                                    "POS",
                                    ["noun", "verb", "adj", "adv",
                                     "prep", "conj", "pron", "num",
                                     "part", "interj", "other"],
                                    key=f"gw_pos_{cur_id}_{sel_idx}",
                                )
                            extra_forms = st.text_input(
                                "Extra forms (comma-separated)",
                                key=f"gw_ef_{cur_id}_{sel_idx}",
                                placeholder="e.g. vody, vode, vodu",
                            )
                            if st.button(
                                f"Add **{new_gid}** for *{norm_w}*",
                                key=f"gw_add_{cur_id}_{sel_idx}",
                            ):
                                forms = [
                                    f.strip()
                                    for f in extra_forms.split(",")
                                    if f.strip()
                                ]
                                glossary.add_gloss(
                                    new_gid, norm_w, forms, new_pos,
                                )
                                save_glossary(glossary, GLOSSARY_JSON)
                                st.rerun()

                    # Metrics
                    st.divider()
                    gm1, gm2 = st.columns(2)
                    gm1.metric("Glosses", glossary.n_glosses)
                    gm2.metric("Word forms", glossary.n_forms)

                    # Browse full glossary
                    with st.expander("Browse full glossary"):
                        g_search = st.text_input(
                            "Search glossary",
                            key="gw_search",
                            placeholder="gloss ID or word form",
                        )
                        all_ids = glossary.gloss_ids
                        if g_search:
                            q = g_search.strip().lower()
                            all_ids = [
                                gid for gid in all_ids
                                if q in gid.lower()
                                or any(
                                    q in f.lower()
                                    for f in (
                                        glossary.get_entry(gid) or {}
                                    ).get("forms", [])
                                )
                            ]
                        for gid in all_ids[:50]:
                            entry = glossary.get_entry(gid)
                            if not entry:
                                continue
                            bc1, bc2 = st.columns([5, 1])
                            forms_str = ", ".join(entry["forms"][:8])
                            bc1.markdown(
                                f"**{gid}** — {entry['lemma']} "
                                f"({entry.get('pos', '')}) — "
                                f"{forms_str}"
                            )
                            if bc2.button(
                                "Del", key=f"gw_del_{gid}",
                            ):
                                glossary.remove_gloss(gid)
                                save_glossary(glossary, GLOSSARY_JSON)
                                st.rerun()
                        if len(all_ids) > 50:
                            st.caption(
                                f"Showing 50 of {len(all_ids)} — "
                                "use search to find more."
                            )
                        if not all_ids:
                            st.caption("No glosses yet." if not g_search
                                       else "No matches.")

                # ── Synced Video + Pose viewer ─────────────────────────
                import streamlit.components.v1 as stc

                video_path = Path(str(cur_row["video_path"]))
                _synced = False
                aspect = _video_aspect(str(video_path))

                if video_path.exists() and pose_loaded and r_n_frames > 0:
                    # Try synced viewer (ffmpeg extracts segment → base64)
                    seg_bytes = _extract_segment_cached(
                        str(video_path), rev_start, rev_end,
                    )
                    if seg_bytes is not None:
                        html = synced_video_pose_html(
                            seg_bytes, p_data, p_conf, p_fps,
                            r_frame_start, r_frame_end,
                        )
                        iframe_h = (
                            max(500, int(350 * aspect) + 80)
                            if aspect > 0 else 500
                        )
                        stc.html(html, height=iframe_h)
                        _synced = True
                    else:
                        st.warning(
                            "ffmpeg unavailable — video and pose "
                            "play independently (not synced)."
                        )

                if not _synced:
                    # Fallback: two-column independent playback
                    col_vid, col_pose = st.columns(2)
                    with col_vid:
                        if video_path.exists():
                            st.video(
                                str(video_path),
                                start_time=max(0, rev_start // 1000),
                            )
                        else:
                            st.warning(
                                f"Video not found: `{video_path.name}`"
                            )
                    with col_pose:
                        if pose_loaded and r_n_frames > 0:
                            anim_html = pose_animation_html(
                                p_data, p_conf, p_fps,
                                r_frame_start, r_frame_end, rev_start,
                                video_aspect=aspect,
                            )
                            stc.html(anim_html, height=600)
                        elif pose_loaded:
                            st.info("No frames in this segment.")

                # ── Edit controls ─────────────────────────────────────
                ec1, ec2 = st.columns([3, 5])
                with ec1:
                    st.text_area("Edit text", key=tx_key, height=68)
                with ec2:
                    seg_start = int(cur_row["start_ms"])
                    seg_end = int(cur_row["end_ms"])
                    trim_range = st.slider(
                        "Trim segment",
                        min_value=max(0, seg_start - 2000),
                        max_value=seg_end + 2000,
                        value=(rev_start, rev_end),
                        step=50,
                        format="%dms",
                        key=trim_key,
                    )
                    rev_start, rev_end = trim_range
                    st.caption(
                        f"{_fmt_ms(rev_start)} → {_fmt_ms(rev_end)}"
                        f"  ({rev_end - rev_start}ms)"
                    )
                reviewed_text = st.session_state[tx_key]

                # ── Action buttons ────────────────────────────────────
                if is_zero_frames:
                    st.caption(
                        "Cannot approve: 0 frames at reviewed "
                        "timestamps."
                    )

                b1, b2, b3, b4, _, b5, b6 = st.columns(
                    [2, 2, 2, 2, 1, 2, 2]
                )
                approve_btn = b1.button(
                    "Approve", key=f"app_{cur_id}", type="primary",
                    disabled=is_zero_frames,
                )
                skip_btn = b2.button("Skip", key=f"skp_{cur_id}")
                flag_btn = b3.button("Flag", key=f"flg_{cur_id}")
                merge_btn = b4.button(
                    "Merge↓", key=f"mrg_{cur_id}",
                    disabled=(cur_pos >= len(seg_ids) - 1),
                    help="Merge this segment with the next one",
                )
                prev_btn = b5.button(
                    "Prev", key=f"prv_{cur_id}",
                    disabled=(cur_pos == 0),
                )
                next_btn = b6.button(
                    "Next", key=f"nxt_{cur_id}",
                    disabled=(cur_pos >= len(seg_ids) - 1),
                )

                # Status badge
                st.caption(
                    f"{_icons.get(str(cur_row['status']), '⚫')} "
                    f"**{cur_row['status']}** — "
                    f"segment {cur_pos + 1} / {len(seg_ids)}"
                )

                # ── Action handlers ───────────────────────────────────
                def _commit(status: str, advance: bool) -> None:
                    df.loc[df_idx, "status"] = status
                    df.loc[df_idx, "reviewed_text"] = reviewed_text
                    df.loc[df_idx, "reviewed_start_ms"] = rev_start
                    df.loc[df_idx, "reviewed_end_ms"] = rev_end
                    st.session_state["td_align_df"] = df
                    save_alignment_csv(df, ALIGNMENT_CSV)
                    if advance:
                        # Stay within the same video (reuse stem filter)
                        pending_in_video = df[
                            (df["video_path"].apply(
                                lambda p: Path(p).stem
                            ) == cur_video)
                            & (df["status"] == ST_PENDING)
                        ]
                        if not pending_in_video.empty:
                            st.session_state["_td_nav_target"] = (
                                pending_in_video.iloc[0]["segment_id"]
                            )

                if approve_btn:
                    _commit(ST_APPROVED, advance=True)
                    st.rerun()
                if skip_btn:
                    _commit(ST_SKIPPED, advance=True)
                    st.rerun()
                if flag_btn:
                    _commit(ST_FLAGGED, advance=False)
                    st.rerun()
                if merge_btn and cur_pos < len(seg_ids) - 1:
                    next_id  = seg_ids[cur_pos + 1]
                    next_idx = df.index[
                        df["segment_id"] == next_id
                    ][0]
                    next_row = df.loc[next_idx]
                    # Combine text
                    cur_text  = str(df.loc[df_idx, "text"])
                    next_text = str(next_row["text"])
                    df.loc[df_idx, "text"] = (
                        cur_text + "\n" + next_text
                    )
                    # Extend end timestamps
                    df.loc[df_idx, "end_ms"] = int(
                        next_row["end_ms"]
                    )
                    df.loc[df_idx, "reviewed_end_ms"] = int(
                        next_row["end_ms"]
                    )
                    df.loc[df_idx, "frame_end"] = int(
                        next_row["frame_end"]
                    )
                    df.loc[df_idx, "n_frames"] = (
                        int(df.loc[df_idx, "frame_end"])
                        - int(df.loc[df_idx, "frame_start"])
                    )
                    # Update reviewed_text if it was set
                    _cr = df.loc[df_idx, "reviewed_text"]
                    _nr = next_row["reviewed_text"]
                    cur_rev = "" if pd.isna(_cr) else str(_cr).strip()
                    next_rev = "" if pd.isna(_nr) else str(_nr).strip()
                    if cur_rev or next_rev:
                        df.loc[df_idx, "reviewed_text"] = (
                            (cur_rev or cur_text)
                            + "\n"
                            + (next_rev or next_text)
                        )
                    # Delete the next segment
                    df = df.drop(next_idx).reset_index(drop=True)
                    st.session_state["td_align_df"] = df
                    save_alignment_csv(df, ALIGNMENT_CSV)
                    st.rerun()
                if prev_btn:
                    st.session_state["_td_nav_target"] = (
                        seg_ids[cur_pos - 1]
                    )
                    st.rerun()
                if next_btn:
                    st.session_state["_td_nav_target"] = (
                        seg_ids[cur_pos + 1]
                    )
                    st.rerun()

# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — SIGN-WORD PAIRING
# ══════════════════════════════════════════════════════════════════════════
with tab_signword:
    # ── EAF Harvest ───────────────────────────────────────────────
    with st.expander("Harvest EAF Annotations", expanded=False):
        st.markdown(
            "Bulk-import human-corrected **S1_Gloss_RH** / **S1_Gloss_LH** "
            "annotations from EAF files into pairings. This closes the "
            "active learning loop: AI predicts → annotator corrects in ELAN "
            "→ harvest → retrain."
        )
        annotations_dir = DATA_DIR / "annotations"
        if not annotations_dir.exists():
            st.warning(f"No annotations directory at `{annotations_dir}`")
        elif not INVENTORY_CSV.exists():
            st.warning("Inventory CSV not found. Run the **Inventory** page first.")
        else:
            if st.button("Harvest All EAFs", key="btn_harvest_eaf"):
                inv_df = pd.read_csv(INVENTORY_CSV)
                glossary = None
                if GLOSSARY_JSON.exists():
                    glossary = load_glossary(GLOSSARY_JSON)

                prog = st.progress(0.0, text="Scanning EAF files...")

                def _harvest_progress(current: int, total: int) -> None:
                    prog.progress(
                        current / max(total, 1),
                        text=f"Scanning EAF {current}/{total}...",
                    )

                result = harvest_eaf_batch(
                    annotations_dir=annotations_dir,
                    pose_dir=POSE_DIR,
                    inventory_df=inv_df,
                    pairings_path=PAIRINGS_CSV,
                    glossary=glossary,
                    progress_callback=_harvest_progress,
                )
                prog.empty()

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Files scanned", result["n_files_scanned"])
                c2.metric("With annotations", result["n_with_annotations"])
                c3.metric("New pairings", result["n_new_pairings"])
                c4.metric("Dupes skipped", result["n_skipped_dupes"])

                if result["n_new_pairings"] > 0:
                    st.success(f"Added {result['n_new_pairings']} new pairings!")
                    # Invalidate cached pairings
                    st.session_state.pop("td_pairings_df", None)
                elif result["n_with_annotations"] == 0:
                    st.info("No EAF files have human S1_Gloss annotations yet.")
                else:
                    st.info("All annotations already in pairings (0 new).")

    # ── Original Sign-Word tab content ────────────────────────────
    df = st.session_state.get("td_align_df")

    if df is None or df.empty:
        st.info("No alignment data yet. Run alignment in the **Align** tab first.")
    else:
        approved_df = df[df["status"] == ST_APPROVED]
        if approved_df.empty:
            st.info(
                "No approved segments yet. "
                "Approve segments in the **Review** tab first."
            )
        else:
            pairings_df = _get_pairings_df()

            # ── Video selector (same pattern as Review) ───────────
            approved_df = approved_df.copy()
            approved_df["_vstem"] = approved_df["video_path"].apply(
                lambda p: Path(p).stem
            )
            sw_video_stems = sorted(approved_df["_vstem"].unique())

            if not sw_video_stems:
                st.info("No approved segments available.")
            else:
                sw_cur_video = st.selectbox(
                    "Video",
                    sw_video_stems,
                    key="sw_current_video",
                )

                # Filter segments for this video
                sw_video_df = approved_df[
                    approved_df["_vstem"] == sw_cur_video
                ]
                sw_seg_ids = sw_video_df["segment_id"].tolist()
                sw_id_to_row = {
                    row["segment_id"]: row
                    for _, row in sw_video_df.iterrows()
                }

                # Pre-compute paired counts (O(N) once, not O(N*M) per selectbox)
                _paired_counts: dict[str, int] = {}
                if not pairings_df.empty:
                    _pm = pairings_df["status"].isin([PST_PAIRED, PST_AUTO_SUGGESTED])
                    _paired_counts = pairings_df[_pm].groupby("segment_id").size().to_dict()

                # Segment selector
                def _sw_seg_label(sid):
                    r = sw_id_to_row[sid]
                    ts = (
                        f"{_fmt_ms(int(r['start_ms']))}"
                        f"→{_fmt_ms(int(r['end_ms']))}"
                    )
                    text = str(r.get("reviewed_text", "") or r.get("text", ""))[:50]
                    n_paired = _paired_counts.get(sid, 0)
                    return f"{ts} | {text} ({n_paired} paired)"

                sw_cur_seg = st.selectbox(
                    "Segment (approved only)",
                    sw_seg_ids,
                    format_func=_sw_seg_label,
                    key="sw_current_seg_id",
                )

                sw_row = sw_id_to_row[sw_cur_seg]
                subtitle_text = (
                    str(sw_row.get("reviewed_text", "")).strip()
                    or str(sw_row.get("text", ""))
                )

                # ── Subtitle text display ─────────────────────────
                glossary = _get_glossary()
                word_matches = glossary.match_sentence(subtitle_text)

                # Color-coded sentence
                st.markdown(
                    '<div style="text-align:center;line-height:2.2;'
                    'padding:8px 12px;background:rgba(255,255,255,0.05);'
                    'border-radius:8px;margin:4px 0;">'
                    + _render_glossary_html(word_matches) + "</div>",
                    unsafe_allow_html=True,
                )

                # ── Load pose data ────────────────────────────────
                sw_pose_loaded = False
                try:
                    sw_p_data, sw_p_conf, sw_fps = _load_pose_cached(
                        str(sw_row["pose_path"])
                    )
                    sw_pose_loaded = True
                except Exception as exc:
                    st.error(f"Pose load error: {exc}")

                # ── Filter pairings to current segment ────────────
                seg_pairings = pd.DataFrame()
                if not pairings_df.empty:
                    seg_pairings = pairings_df[
                        pairings_df["segment_id"] == sw_cur_seg
                    ].copy()

                # ── Stats row ─────────────────────────────────────
                n_detected = len(seg_pairings)
                n_paired_seg = int(
                    seg_pairings["status"].isin(
                        [PST_PAIRED, PST_AUTO_SUGGESTED]
                    ).sum()
                ) if n_detected else 0
                n_suggested = int(
                    (seg_pairings["status"] == PST_AUTO_SUGGESTED).sum()
                ) if n_detected else 0

                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("Signs detected", n_detected)
                sc2.metric("Paired", n_paired_seg)
                sc3.metric("Auto-suggested", n_suggested)

                # ── Global progress bar (retrain thresholds) ──────
                total_paired = 0
                if not pairings_df.empty:
                    total_paired = int(
                        pairings_df["status"].isin(
                            [PST_PAIRED, PST_AUTO_SUGGESTED]
                        ).sum()
                    )

                # Retrain thresholds from CLAUDE.md
                thresholds = [
                    (500, "Fine-tune backbone"),
                    (2000, "v1 retrain"),
                    (5000, "v2 active learning"),
                    (10000, "v3 full retrain"),
                ]
                next_thresh = 500
                next_label = "Fine-tune backbone"
                for t, lbl in thresholds:
                    if total_paired < t:
                        next_thresh = t
                        next_label = lbl
                        break
                else:
                    next_thresh = thresholds[-1][0]
                    next_label = thresholds[-1][1]

                st.progress(
                    min(1.0, total_paired / next_thresh),
                    text=f"{total_paired}/{next_thresh} paired signs → {next_label}",
                )

                # ── Action buttons ────────────────────────────────
                btn_c1, btn_c2 = st.columns(2)
                detect_btn = btn_c1.button(
                    "Detect Signs",
                    type="primary",
                    disabled=not sw_pose_loaded,
                    key="sw_detect_btn",
                )

                # Find latest checkpoint for suggestions
                _ckpt_path = None
                if MODELS_DIR.exists():
                    ckpts = sorted(MODELS_DIR.glob("*.pt"))
                    if ckpts:
                        _ckpt_path = ckpts[-1]

                suggest_btn = btn_c2.button(
                    "Auto-Suggest",
                    disabled=(
                        not sw_pose_loaded
                        or seg_pairings.empty
                        or _ckpt_path is None
                    ),
                    key="sw_suggest_btn",
                    help=(
                        "No trained model found"
                        if _ckpt_path is None
                        else f"Using {_ckpt_path.name}"
                    ),
                )

                # ── Manual sign add ───────────────────────────────
                seg_start_ms = int(sw_row["reviewed_start_ms"])
                seg_end_ms = int(sw_row["reviewed_end_ms"])

                with st.expander("+ Add sign manually"):
                    mc1, mc2 = st.columns([1, 3])
                    with mc1:
                        manual_hand = st.selectbox(
                            "Hand",
                            ["right", "left"],
                            format_func=lambda h: "RH" if h == "right" else "LH",
                            key="sw_manual_hand",
                        )
                    with mc2:
                        manual_range = st.slider(
                            "Time range",
                            min_value=max(0, seg_start_ms - 500),
                            max_value=seg_end_ms + 500,
                            value=(seg_start_ms, seg_end_ms),
                            step=10,
                            format="%dms",
                            key="sw_manual_range",
                        )
                    if st.button("Add Sign", key="sw_add_manual", type="secondary"):
                        if sw_pose_loaded:
                            new_p = create_manual_pairing(
                                segment_id=sw_cur_seg,
                                video_path=str(sw_row["video_path"]),
                                pose_path=str(sw_row["pose_path"]),
                                hand=manual_hand,
                                sign_start_ms=manual_range[0],
                                sign_end_ms=manual_range[1],
                                fps=sw_fps,
                            )
                            new_pdf = pd.DataFrame([new_p])
                            pairings_df = pd.concat(
                                [pairings_df, new_pdf], ignore_index=True,
                            )
                            _save_pairings(pairings_df)
                            st.rerun()
                        else:
                            st.error("Pose data not loaded — cannot add sign.")

                # ── Detect handler ────────────────────────────────
                if detect_btn and sw_pose_loaded:
                    new_pairings = detect_signs_in_segment(
                        sw_p_data, sw_p_conf, sw_fps,
                        int(sw_row["reviewed_start_ms"]),
                        int(sw_row["reviewed_end_ms"]),
                        sw_cur_seg,
                        str(sw_row["video_path"]),
                        str(sw_row["pose_path"]),
                    )
                    if new_pairings:
                        new_pdf = pd.DataFrame(new_pairings)
                        # Remove existing pairings for this segment
                        if not pairings_df.empty:
                            pairings_df = pairings_df[
                                pairings_df["segment_id"] != sw_cur_seg
                            ]
                        pairings_df = pd.concat(
                            [pairings_df, new_pdf], ignore_index=True,
                        )
                        _save_pairings(pairings_df)
                        st.rerun()
                    else:
                        st.warning("No sign boundaries detected in this segment.")

                # ── Suggest handler ───────────────────────────────
                if suggest_btn and sw_pose_loaded and _ckpt_path and not seg_pairings.empty:
                    with st.spinner("Running model inference..."):
                        model, label_enc, ckpt_cfg = _load_model_cached(
                            str(_ckpt_path)
                        )
                        pairings_df = suggest_sign_pairings(
                            pairings_df,
                            sw_p_data, sw_fps,
                            model, label_enc, ckpt_cfg,
                            glossary,
                            subtitle_text,
                        )
                        _save_pairings(pairings_df)
                    st.rerun()

                # ── Display each detected sign ────────────────────
                if seg_pairings.empty and not detect_btn:
                    st.info(
                        "No signs detected for this segment yet. "
                        "Click **Detect Signs** to find sign boundaries."
                    )
                elif not seg_pairings.empty:
                    # Track which sign is expanded for pose viewer
                    sw_selected_sign = st.session_state.get("sw_selected_sign")

                    # Hoist invariants out of per-sign loop
                    import streamlit.components.v1 as stc
                    sw_video_path = Path(str(sw_row["video_path"]))
                    sw_word_options = tokenize_slovak(subtitle_text)

                    for pidx, (_, p_row) in enumerate(seg_pairings.iterrows()):
                        pid = p_row["pairing_id"]
                        hand_label = "RH" if p_row["hand"] == "right" else "LH"
                        status = str(p_row["status"])
                        icon = _PAIRING_ICONS.get(status, "⚫")

                        header = (
                            f"{icon} Sign {pidx + 1}: "
                            f"{_fmt_ms(int(p_row['sign_start_ms']))}"
                            f"→{_fmt_ms(int(p_row['sign_end_ms']))} "
                            f"({hand_label}) — {status}"
                        )

                        with st.expander(header, expanded=(pid == sw_selected_sign)):
                            # Show suggestion if present
                            if p_row.get("suggestion_gloss") and str(p_row["suggestion_gloss"]).strip():
                                sug_conf = float(p_row.get("suggestion_confidence", 0))
                                st.caption(
                                    f"Suggested: **{p_row['suggestion_gloss']}** "
                                    f"({sug_conf:.0%})"
                                )

                            # Mini pose viewer for this sign
                            if sw_pose_loaded:
                                sign_f_start = int(p_row["sign_frame_start"])
                                sign_f_end = int(p_row["sign_frame_end"])
                                sign_n_frames = sign_f_end - sign_f_start

                                if sign_n_frames > 0 and sw_video_path.exists():
                                    seg_bytes = _extract_segment_cached(
                                        str(sw_video_path),
                                        int(p_row["sign_start_ms"]),
                                        int(p_row["sign_end_ms"]),
                                    )
                                    if seg_bytes is not None:
                                        html = synced_video_pose_html(
                                            seg_bytes,
                                            sw_p_data, sw_p_conf, sw_fps,
                                            sign_f_start, sign_f_end,
                                        )
                                        stc.html(html, height=350)
                                    else:
                                        anim_html = pose_animation_html(
                                            sw_p_data, sw_p_conf, sw_fps,
                                            sign_f_start, sign_f_end,
                                            int(p_row["sign_start_ms"]),
                                            video_aspect=_video_aspect(
                                                str(sw_video_path)),
                                        )
                                        stc.html(anim_html, height=350)

                            # ── Time adjustment ────────────────────
                            sign_trim = st.slider(
                                "Sign timing",
                                min_value=max(0, seg_start_ms - 500),
                                max_value=seg_end_ms + 500,
                                value=(int(p_row["sign_start_ms"]),
                                       int(p_row["sign_end_ms"])),
                                step=10,
                                format="%dms",
                                key=f"sw_trim_{pid}",
                            )
                            st.caption(
                                f"{_fmt_ms(sign_trim[0])} → {_fmt_ms(sign_trim[1])}"
                            )

                            # ── Word pairing controls ─────────────
                            # Pre-select current words if paired
                            cur_word_str = str(p_row.get("word", ""))
                            current_words: list[str] = [
                                w for w in cur_word_str.split()
                                if w and w in sw_word_options
                            ]

                            wc1, wc2 = st.columns([3, 2])
                            with wc1:
                                sel_words = st.multiselect(
                                    "Words",
                                    sw_word_options,
                                    default=current_words,
                                    key=f"sw_words_{pid}",
                                )
                            with wc2:
                                # Gloss lookup uses first word; multi-word
                                # signs get a compound gloss via auto-create
                                word_glosses: list[str] = []
                                if sel_words:
                                    word_glosses = glossary.lookup(sel_words[0])
                                    if word_glosses:
                                        st.caption(
                                            "Gloss: "
                                            + ", ".join(word_glosses)
                                        )
                                    elif len(sel_words) > 1:
                                        st.caption("New compound gloss")
                                    else:
                                        st.caption("Unknown word")

                            # ── Note field ──────────────────────────
                            note = st.text_input(
                                "Note",
                                value=str(p_row.get("note", "") or ""),
                                key=f"sw_note_{pid}",
                                placeholder="e.g. 2 words → 1 sign",
                            )

                            # Action buttons
                            ac1, ac2, ac3 = st.columns(3)

                            pair_disabled = (
                                len(sel_words) == 0
                                or status == PST_PAIRED
                            )
                            confirm_label = "Confirm" if status == PST_AUTO_SUGGESTED else "Pair"
                            pair_btn = ac1.button(
                                confirm_label,
                                key=f"sw_pair_{pid}",
                                type="primary",
                                disabled=pair_disabled,
                            )
                            skip_btn = ac2.button(
                                "Skip",
                                key=f"sw_skip_{pid}",
                                disabled=(status == PST_SKIPPED),
                            )
                            unpair_btn = ac3.button(
                                "Unpair",
                                key=f"sw_unpair_{pid}",
                                disabled=(
                                    status not in (PST_PAIRED, PST_AUTO_SUGGESTED)
                                ),
                            )

                            # ── Pair handler ──────────────────────
                            if pair_btn and sel_words:
                                joined_word = " ".join(sel_words)

                                # Gloss lookup on first word
                                if word_glosses:
                                    gloss_id = word_glosses[0]
                                else:
                                    # Unknown word — auto-create gloss
                                    from spj.glossary import normalize_word
                                    nw = normalize_word(sel_words[0])
                                    gloss_id = glossary.suggest_next_id(
                                        nw.upper()
                                    )
                                    glossary.add_gloss(gloss_id, nw)
                                    save_glossary(glossary, GLOSSARY_JSON)

                                # Recompute frame indices from adjusted time
                                adj_frame_start = int(sign_trim[0] * sw_fps / 1000)
                                adj_frame_end = int(sign_trim[1] * sw_fps / 1000)

                                # Update the pairing
                                mask = pairings_df["pairing_id"] == pid
                                pairings_df.loc[mask, "word"] = joined_word
                                pairings_df.loc[mask, "gloss_id"] = gloss_id
                                pairings_df.loc[mask, "status"] = PST_PAIRED
                                pairings_df.loc[mask, "sign_start_ms"] = sign_trim[0]
                                pairings_df.loc[mask, "sign_end_ms"] = sign_trim[1]
                                pairings_df.loc[mask, "sign_frame_start"] = adj_frame_start
                                pairings_df.loc[mask, "sign_frame_end"] = adj_frame_end
                                pairings_df.loc[mask, "note"] = note
                                _save_pairings(pairings_df)
                                st.session_state["sw_selected_sign"] = pid
                                st.rerun()

                            # ── Skip handler ──────────────────────
                            if skip_btn:
                                mask = pairings_df["pairing_id"] == pid
                                pairings_df.loc[mask, "status"] = PST_SKIPPED
                                _save_pairings(pairings_df)
                                st.rerun()

                            # ── Unpair handler ────────────────────
                            if unpair_btn:
                                mask = pairings_df["pairing_id"] == pid
                                pairings_df.loc[mask, "word"] = ""
                                pairings_df.loc[mask, "gloss_id"] = ""
                                pairings_df.loc[mask, "status"] = PST_PENDING
                                _save_pairings(pairings_df)
                                st.rerun()

# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — EXPORT
# ══════════════════════════════════════════════════════════════════════════
with tab_export:
    df = st.session_state.get("td_align_df")

    if df is None or df.empty:
        st.info("No alignment data yet. Run alignment in the **Align** tab first.")
    else:
        sc = df["status"].value_counts()
        ec1, ec2, ec3, ec4 = st.columns(4)
        ec1.metric("Approved", int(sc.get(ST_APPROVED, 0)))
        ec2.metric("Skipped",  int(sc.get(ST_SKIPPED,  0)))
        ec3.metric("Pending",  int(sc.get(ST_PENDING,  0)))
        ec4.metric("Flagged",  int(sc.get(ST_FLAGGED,  0)))

        approved_df = df[df["status"] == ST_APPROVED]

        if approved_df.empty:
            st.info(
                "No approved segments yet. "
                "Review segments in the **Review** tab first."
            )
        else:
            export_mode = st.radio(
                "Export granularity",
                [
                    "Sentence-level (subtitle segments)",
                    "Sign-level (word pairings)",
                ],
                key="td_export_mode",
            )

            out_dir_str = st.text_input(
                "Output directory",
                value=str(TRAINING_DIR / "export"),
                key="td_out_dir",
            )
            out_dir = Path(out_dir_str)

            # Label quality filter
            with st.expander("Label quality filter", expanded=False):
                st.caption(
                    "Filter labels by minimum sample count. "
                    "Useful for training — labels with too few samples can't be learned."
                )
                fq1, fq2 = st.columns(2)
                do_normalize = fq1.checkbox(
                    "Normalize labels (lowercase, strip _N suffixes)",
                    value=True, key="td_normalize_labels",
                )
                min_samples = fq2.number_input(
                    "Min samples per label", 1, 100, 3,
                    key="td_min_samples",
                    help="Labels with fewer samples will be excluded",
                )

                # Preview filter effect
                if approved_df is not None and not approved_df.empty:
                    preview_df = approved_df.copy()
                    # Derive label column for preview
                    if "label" not in preview_df.columns:
                        preview_df["label"] = preview_df["reviewed_text"].where(
                            preview_df["reviewed_text"].str.strip() != "",
                            preview_df["text"],
                        )
                    if do_normalize:
                        preview_df["label"] = preview_df["label"].apply(normalize_label)
                    preview_df = preview_df[preview_df["label"].str.strip() != ""]
                    preview_df = preview_df[~preview_df["label"].isna()]
                    label_counts = preview_df["label"].value_counts()
                    valid = label_counts[label_counts >= min_samples]
                    fc1, fc2, fc3 = st.columns(3)
                    fc1.metric("Labels before", len(label_counts))
                    fc2.metric("Labels after", len(valid))
                    fc3.metric("Segments after", int(valid.sum()))

            if export_mode == "Sentence-level (subtitle segments)":
                # ── Sentence-level export (original) ──────────────
                col_e1, col_e2 = st.columns(2)

                with col_e1:
                    if st.button("Export approved segments", type="primary"):
                        out_dir.mkdir(parents=True, exist_ok=True)
                        bar    = st.progress(0.0, text="Exporting …")
                        errors: list[str] = []
                        paths:  list[str] = []

                        for i, (_, row) in enumerate(approved_df.iterrows(), 1):
                            bar.progress(
                                i / len(approved_df),
                                text=f"{i} / {len(approved_df)}",
                            )
                            try:
                                p = export_segment_npz(row, out_dir)
                                if p is not None:
                                    paths.append(p)
                                else:
                                    errors.append(
                                        f"`{row.get('segment_id', '?')}`: "
                                        "skipped (0-frame segment)"
                                    )
                            except Exception as exc:
                                errors.append(f"`{row.get('segment_id', '?')}`: {exc}")

                        bar.progress(1.0, text="Done")

                        if errors:
                            with st.expander(f"Export errors ({len(errors)})"):
                                for e in errors:
                                    st.markdown(e)

                        total_mb = sum(
                            Path(p).stat().st_size
                            for p in paths
                            if Path(p).exists()
                        ) / 1_048_576

                        cfg_path = write_training_config(out_dir, len(paths))
                        st.success(
                            f"Exported **{len(paths)}** .npz files — {total_mb:.1f} MB\n\n"
                            f"Training config: `{cfg_path.name}`\n\n"
                            f"Output: `{out_dir}`"
                        )

                with col_e2:
                    if st.button("Export CSV manifest"):
                        out_dir.mkdir(parents=True, exist_ok=True)
                        manifest = approved_df[[
                            "segment_id", "video_path",
                            "start_ms", "end_ms",
                            "reviewed_start_ms", "reviewed_end_ms",
                            "text", "reviewed_text",
                            "fps", "n_frames",
                        ]].copy()
                        manifest["label"] = manifest["reviewed_text"].where(
                            manifest["reviewed_text"].str.strip() != "",
                            manifest["text"],
                        )

                        # Apply quality filter if configured
                        if min_samples > 1:
                            before_count = len(manifest)
                            manifest = filter_quality_labels(
                                manifest,
                                min_samples=min_samples,
                                normalize=do_normalize,
                            )
                            st.info(
                                f"Quality filter: {before_count} → {len(manifest)} segments "
                                f"({len(manifest['label'].unique())} labels with {min_samples}+ samples)"
                            )

                        manifest_path = out_dir / "manifest.csv"
                        manifest.to_csv(manifest_path, index=False)
                        st.success(f"Manifest saved: `{manifest_path}`")

            else:
                # ── Sign-level export ─────────────────────────────
                exp_pairings = _get_pairings_df()

                if exp_pairings.empty:
                    st.info(
                        "No pairings yet. Use the **Sign-Word** tab to "
                        "detect and pair signs first."
                    )
                else:
                    paired_p = exp_pairings[
                        exp_pairings["status"].isin([PST_PAIRED, PST_AUTO_SUGGESTED])
                    ]
                    ep1, ep2 = st.columns(2)
                    ep1.metric("Paired signs", len(paired_p))
                    ep2.metric(
                        "Unique glosses",
                        int(paired_p["gloss_id"].nunique()) if not paired_p.empty else 0,
                    )

                    if paired_p.empty:
                        st.info("No paired signs to export.")
                    else:
                        col_se1, col_se2 = st.columns(2)

                        with col_se1:
                            if st.button("Export paired signs", type="primary"):
                                out_dir.mkdir(parents=True, exist_ok=True)
                                bar = st.progress(0.0, text="Exporting signs …")
                                errors: list[str] = []
                                paths: list[str] = []

                                # Group by pose_path for efficient loading
                                for pose_path, group in paired_p.groupby("pose_path"):
                                    try:
                                        p_data, p_conf, _ = _load_pose_cached(str(pose_path))
                                    except Exception as exc:
                                        for _, r in group.iterrows():
                                            errors.append(
                                                f"`{r.get('pairing_id', '?')}`: pose load error: {exc}"
                                            )
                                        continue

                                    for _, row in group.iterrows():
                                        try:
                                            p = export_sign_npz(
                                                row, p_data, p_conf, out_dir,
                                            )
                                            if p is not None:
                                                paths.append(p)
                                            else:
                                                errors.append(
                                                    f"`{row.get('pairing_id', '?')}`: "
                                                    "skipped (0 frames)"
                                                )
                                        except Exception as exc:
                                            errors.append(
                                                f"`{row.get('pairing_id', '?')}`: {exc}"
                                            )

                                    bar.progress(
                                        min(1.0, len(paths) / max(1, len(paired_p))),
                                        text=f"{len(paths)} / {len(paired_p)}",
                                    )

                                bar.progress(1.0, text="Done")

                                if errors:
                                    with st.expander(f"Export errors ({len(errors)})"):
                                        for e in errors:
                                            st.markdown(e)

                                total_mb = sum(
                                    Path(p).stat().st_size
                                    for p in paths
                                    if Path(p).exists()
                                ) / 1_048_576

                                cfg_path = write_training_config(out_dir, len(paths))
                                st.success(
                                    f"Exported **{len(paths)}** sign .npz files — "
                                    f"{total_mb:.1f} MB\n\n"
                                    f"Training config: `{cfg_path.name}`\n\n"
                                    f"Output: `{out_dir}`"
                                )

                        with col_se2:
                            if st.button("Export sign manifest"):
                                out_dir.mkdir(parents=True, exist_ok=True)
                                manifest = paired_p[[
                                    "pairing_id", "segment_id",
                                    "video_path", "pose_path", "hand",
                                    "sign_start_ms", "sign_end_ms",
                                    "sign_frame_start", "sign_frame_end",
                                    "word", "gloss_id", "fps",
                                ]].copy()
                                manifest["label"] = manifest["gloss_id"]
                                manifest_path = out_dir / "manifest.csv"
                                manifest.to_csv(manifest_path, index=False)
                                st.success(f"Manifest saved: `{manifest_path}`")
