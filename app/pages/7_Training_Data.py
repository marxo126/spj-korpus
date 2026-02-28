"""Training Data page — pose-subtitle alignment, visual review, NPZ export.

Tab 1 (Align):   build alignment table from .pose + .vtt pairs
Tab 2 (Review):  frame-by-frame visual review with approve / skip / flag
Tab 3 (Export):  export approved segments as float16 .npz training files
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from spj.glossary import Glossary, load_glossary, save_glossary
from spj.training_data import (
    build_alignment_table,
    export_segment_npz,
    load_alignment_csv,
    pose_animation_html,
    pose_frame_figure,
    save_alignment_csv,
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
GLOSSARY_JSON = TRAINING_DIR / "glossary.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_ms(ms: int) -> str:
    m   = ms // 60_000
    s   = (ms % 60_000) // 1_000
    rem = ms % 1_000
    return f"{m:02d}:{s:02d}.{rem:03d}"


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


@st.cache_resource
def _load_pose_cached(pose_path_str: str):
    """Load pose arrays once per video path, shared across all reruns."""
    from spj.preannotate import load_pose_arrays
    return load_pose_arrays(Path(pose_path_str))


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.header("🎓 Training Data")
st.caption(
    "Align .pose files with .vtt subtitles → visually review segments → "
    "export approved segments as float16 .npz files for SignBERT / OpenHands."
)
st.caption("Page 7/10 · Align pose + subtitles, review each segment, export .npz training files.")
with st.expander("ℹ️ How to use this page", expanded=False):
    st.markdown("""
**Needs:**
- Pose files (page 2 — Pose Extraction)
- Subtitle `.vtt` files (page 6 — Subtitles)

Both must exist for the same video.

**Steps:**

**Tab 📐 Align**
1. Click **📐 Align all pairs** — matches each subtitle timestamp to the pose frame range.
   Creates one row per subtitle cue in `data/training/alignment.csv`.

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

tab_align, tab_review, tab_export = st.tabs(["📐 Align", "👁 Review", "📦 Export"])

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

                # Validate / initialise current segment selection
                cur_id = st.session_state.get("td_current_seg_id")
                if cur_id not in seg_ids:
                    cur_id = seg_ids[0]
                    st.session_state["td_current_seg_id"] = cur_id

                cur_id = st.selectbox(
                    "Segment",
                    seg_ids,
                    index=seg_ids.index(cur_id),
                    format_func=_seg_label_short,
                    key="td_current_seg_id",
                )

                cur_row = id_to_row[cur_id]
                df_idx  = df.index[df["segment_id"] == cur_id][0]
                cur_pos = seg_ids.index(cur_id)

                # ── Reviewed timestamps (from session state) ─────────
                rs_key = f"rs_{cur_id}"
                re_key = f"re_{cur_id}"
                if rs_key not in st.session_state:
                    st.session_state[rs_key] = int(cur_row["reviewed_start_ms"])
                if re_key not in st.session_state:
                    st.session_state[re_key] = int(cur_row["reviewed_end_ms"])

                rev_start = int(st.session_state[rs_key])
                rev_end   = int(st.session_state[re_key])

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
                        html_parts = []
                        for wm in word_matches:
                            if wm["mapped"]:
                                tip = ", ".join(wm["glosses"])
                                html_parts.append(
                                    _word_span(wm["raw"], *_CLR_MAPPED, tip)
                                )
                            else:
                                html_parts.append(
                                    _word_span(wm["raw"], *_CLR_UNKNOWN)
                                )
                        bg_m, fg_m = _CLR_MAPPED
                        bg_u, fg_u = _CLR_UNKNOWN
                        st.markdown(
                            '<div style="line-height:2.2;margin-bottom:8px;">'
                            + " ".join(html_parts) + "</div>"
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

                # ── Two-column: Video | Pose animation ────────────────
                col_vid, col_pose = st.columns(2)

                with col_vid:
                    video_path = Path(str(cur_row["video_path"]))
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
                        import streamlit.components.v1 as stc
                        anim_html = pose_animation_html(
                            p_data, p_conf, p_fps,
                            r_frame_start, r_frame_end, rev_start,
                        )
                        stc.html(anim_html, height=600)
                    elif pose_loaded:
                        st.info("No frames in this segment.")

                # ── Edit controls ─────────────────────────────────────
                ec1, ec2, ec3 = st.columns([4, 2, 2])
                with ec1:
                    st.text_area("Edit text", key=tx_key, height=68)
                with ec2:
                    rev_start = st.number_input(
                        "Start (ms)", step=100, key=rs_key,
                    )
                with ec3:
                    rev_end = st.number_input(
                        "End (ms)", step=100, key=re_key,
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
                            st.session_state["td_current_seg_id"] = (
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
                    st.session_state["td_current_seg_id"] = (
                        seg_ids[cur_pos - 1]
                    )
                    st.rerun()
                if next_btn:
                    st.session_state["td_current_seg_id"] = (
                        seg_ids[cur_pos + 1]
                    )
                    st.rerun()

# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — EXPORT
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
            out_dir_str = st.text_input(
                "Output directory",
                value=str(TRAINING_DIR / "export"),
                key="td_out_dir",
            )
            out_dir = Path(out_dir_str)

            col_e1, col_e2 = st.columns(2)

            with col_e1:
                if st.button("📦 Export approved segments", type="primary"):
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
                if st.button("📄 Export CSV manifest"):
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
                    manifest_path = out_dir / "manifest.csv"
                    manifest.to_csv(manifest_path, index=False)
                    st.success(f"Manifest saved: `{manifest_path}`")
