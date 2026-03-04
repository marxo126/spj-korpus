"""AI Prepartner-dictn Review — one-prepartner-dictn-at-a-time annotator workflow.

Annotators see AI-predicted glosses (from Page 10 inference) with video+pose,
and approve / correct / skip with one click. Approved prepartner-dictns write directly
to pairings.csv as PST_PAIRED rows, ready for retraining.

Keyboard shortcuts: A=Approve, S=Skip, Z=Undo, Left/Right=Navigate
"""
import logging
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from spj.glossary import load_glossary
from spj.inference import read_prepartner-dictns_from_eaf
from spj.training_data import (
    GLOSS_RE,
    PST_PAIRED,
    PST_SKIPPED,
    make_pairing_dict,
    extract_video_segment,
    load_pairings_csv,
    parse_gloss_value,
    save_pairings_csv,
    synced_video_pose_html,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR       = Path(__file__).parent.parent.parent / "data"
INVENTORY_CSV  = DATA_DIR / "inventory.csv"
POSE_DIR       = DATA_DIR / "pose"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
PAIRINGS_CSV   = DATA_DIR / "training" / "pairings.csv"
GLOSSARY_JSON  = DATA_DIR / "training" / "glossary.json"

# Review status values (display-only — persistent status uses PST_* constants)
_RST_PENDING  = "pending"
_RST_APPROVED = "approved"
_RST_SKIPPED  = "skipped"

_STATUS_ICONS = {
    _RST_PENDING: "\U0001f7e1",    # yellow circle
    _RST_APPROVED: "\u2705",       # green check
    _RST_SKIPPED: "\u23ed\ufe0f",  # skip
}


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=120)
def _load_inventory() -> pd.DataFrame:
    return pd.read_csv(INVENTORY_CSV)


@st.cache_resource(max_entries=3)
def _load_pose_cached(pose_path_str: str):
    """Load pose arrays, bounded to 3 entries to limit memory."""
    from spj.preannotate import load_pose_arrays
    return load_pose_arrays(Path(pose_path_str))


@st.cache_data(max_entries=5, ttl=600)
def _extract_segment_cached(
    video_path_str: str, start_ms: int, end_ms: int,
) -> bytes | None:
    return extract_video_segment(Path(video_path_str), start_ms, end_ms)


@st.cache_data(ttl=300)
def _read_prepartner-dictns_cached(eaf_path_str: str) -> list[dict]:
    return read_prepartner-dictns_from_eaf(Path(eaf_path_str))


@st.cache_data(ttl=600, max_entries=3)
def _wrist_speed_cached(pose_path_str: str, hand: str):
    """Cached wrist velocity — avoids O(T) recomputation on every rerun."""
    import numpy as np
    from spj.preannotate import _wrist_speed, _gaussian_smooth, load_pose_arrays
    data, conf, fps = load_pose_arrays(Path(pose_path_str))
    # Wrist indices matching preannotate.py constants
    if hand == "right":
        speed = _wrist_speed(data, conf, 16, 54)  # _BODY_RIGHT_WRIST, _RIGHT_HAND_WRIST
    else:
        speed = _wrist_speed(data, conf, 15, 33)  # _BODY_LEFT_WRIST, _LEFT_HAND_WRIST
    return _gaussian_smooth(speed, sigma=2.0), fps


def _get_glossary():
    if "ar_glossary" not in st.session_state:
        st.session_state["ar_glossary"] = load_glossary(GLOSSARY_JSON)
    return st.session_state["ar_glossary"]


def _get_pairings_df() -> pd.DataFrame:
    if "ar_pairings_df" in st.session_state:
        return st.session_state["ar_pairings_df"]
    if PAIRINGS_CSV.exists():
        pdf = load_pairings_csv(PAIRINGS_CSV)
        st.session_state["ar_pairings_df"] = pdf
        return pdf
    return pd.DataFrame()


def _save_pairings(pairings_df: pd.DataFrame) -> None:
    save_pairings_csv(pairings_df, PAIRINGS_CSV)
    st.session_state["ar_pairings_df"] = pairings_df


def _fmt_ms(ms: int) -> str:
    return f"{ms / 1000.0:.1f}s"


# ---------------------------------------------------------------------------
# Build dedup set from existing pairings (vectorized)
# ---------------------------------------------------------------------------

def _build_dedup_set(pairings_df: pd.DataFrame) -> dict[tuple, dict]:
    """Map (stem, start_ms, hand) -> pairing row dict for dedup."""
    if pairings_df.empty:
        return {}
    df = pairings_df
    stems = df["video_path"].fillna("").apply(lambda p: Path(str(p)).stem)
    starts = df["sign_start_ms"].fillna(0).astype(int)
    hands = df["hand"].fillna("").astype(str)
    statuses = df["status"].fillna("").astype(str)
    words = df["word"].fillna("").astype(str)
    gloss_ids = df["gloss_id"].fillna("").astype(str)
    return {
        (stem, start, hand): {"status": status, "word": word, "gloss_id": gid}
        for stem, start, hand, status, word, gid
        in zip(stems, starts, hands, statuses, words, gloss_ids)
    }


# ---------------------------------------------------------------------------
# Find videos with AI prepartner-dictns
# ---------------------------------------------------------------------------

def _find_videos_with_prepartner-dictns(
    inventory_df: pd.DataFrame, dedup: dict[tuple, dict],
) -> list[dict]:
    """Scan annotations dir for EAF files containing AI prepartner-dictns."""
    if not ANNOTATIONS_DIR.exists():
        return []

    # Build inventory lookup: stem -> (video_path, fps) — vectorized
    vp_col = "video_path" if "video_path" in inventory_df.columns else "path"
    inv_lookup: dict[str, tuple[str, float]] = dict(zip(
        inventory_df[vp_col].apply(lambda p: Path(str(p)).stem),
        zip(inventory_df[vp_col].astype(str), inventory_df["fps"].astype(float)),
    ))

    videos = []
    for eaf_path in sorted(ANNOTATIONS_DIR.glob("*.eaf")):
        stem = eaf_path.stem
        pose_path = POSE_DIR / f"{stem}.pose"

        if not pose_path.exists() or pose_path.stat().st_size < 1000:
            continue
        if stem not in inv_lookup:
            continue

        try:
            preds = _read_prepartner-dictns_cached(str(eaf_path))
        except Exception as exc:
            logger.debug("Skipping %s: %s", eaf_path.name, exc)
            continue
        if not preds:
            continue

        video_path, fps = inv_lookup[stem]
        n_reviewed = sum(
            1 for p in preds
            if (stem, p["start_ms"], p["hand"]) in dedup
        )

        videos.append({
            "stem": stem,
            "video_path": video_path,
            "pose_path": str(pose_path),
            "eaf_path": str(eaf_path),
            "n_prepartner-dictns": len(preds),
            "n_reviewed": n_reviewed,
            "fps": fps,
        })

    return videos


# ---------------------------------------------------------------------------
# Undo support
# ---------------------------------------------------------------------------

def _push_undo(pairing_id: str, previous_row: dict | None) -> None:
    """Push undo entry — stores the previous state (None = was new)."""
    stack = st.session_state.setdefault("ar_undo_stack", [])
    stack.append({"pairing_id": pairing_id, "previous": previous_row})
    # Cap undo history
    if len(stack) > 50:
        stack[:] = stack[-50:]


def _pop_undo() -> dict | None:
    """Pop last undo entry, or None if stack empty."""
    stack = st.session_state.get("ar_undo_stack", [])
    return stack.pop() if stack else None


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.header("AI Prepartner-dictn Review")
st.caption(
    "Review AI-predicted glosses from inference (Page 10). "
    "Approve, correct, or skip each prepartner-dictn. "
    "Approved prepartner-dictns feed directly into retraining."
)

# Load inventory (cached)
if not INVENTORY_CSV.exists():
    st.warning("No inventory found. Run Page 1 first.")
    st.stop()

inventory_df = _load_inventory()
pairings_df = _get_pairings_df()
dedup = _build_dedup_set(pairings_df)

# Find videos with prepartner-dictns (cached at session level)
if "ar_videos" not in st.session_state:
    with st.spinner("Scanning for videos with AI prepartner-dictns..."):
        st.session_state["ar_videos"] = _find_videos_with_prepartner-dictns(
            inventory_df, dedup,
        )

videos = st.session_state["ar_videos"]

if not videos:
    st.info("No videos with AI prepartner-dictns found. Run inference on Page 10 first.")
    st.stop()

# Sort: most unreviewed first
videos.sort(key=lambda v: v["n_prepartner-dictns"] - v["n_reviewed"], reverse=True)

# ---------------------------------------------------------------------------
# Video selector — track by stem to survive re-sorting
# ---------------------------------------------------------------------------

video_labels = [
    f"{v['stem']} \u2014 {v['n_prepartner-dictns']} prepartner-dictns ({v['n_reviewed']} reviewed)"
    for v in videos
]

col_sel, col_stats = st.columns([3, 1])
with col_sel:
    selected_idx = st.selectbox(
        "Video",
        range(len(videos)),
        format_func=lambda i: video_labels[i],
        key="ar_video_select",
    )

vid = videos[selected_idx]
# Reset prepartner-dictn index if video changed
prev_stem = st.session_state.get("ar_video_stem")
if prev_stem and prev_stem != vid["stem"]:
    st.session_state.pop("ar_pred_idx", None)
    st.session_state.pop("_ar_nav_target", None)
st.session_state["ar_video_stem"] = vid["stem"]
preds = _read_prepartner-dictns_cached(vid["eaf_path"])

with col_stats:
    n_rev = vid["n_reviewed"]
    n_tot = vid["n_prepartner-dictns"]
    st.metric("Reviewed", f"{n_rev}/{n_tot}")

# Progress bar
if n_tot > 0:
    st.progress(n_rev / n_tot)

# ---------------------------------------------------------------------------
# Enrich prepartner-dictns with review status from pairings
# ---------------------------------------------------------------------------

for p in preds:
    key = (vid["stem"], p["start_ms"], p["hand"])
    if key in dedup:
        existing = dedup[key]
        status = existing["status"]
        if status == PST_PAIRED:
            p["review_status"] = _RST_APPROVED
        elif status == PST_SKIPPED:
            p["review_status"] = _RST_SKIPPED
        else:
            p["review_status"] = _RST_PENDING
        p["existing_word"] = existing.get("word", "")
        p["existing_gloss"] = existing.get("gloss_id", "")
    else:
        p["review_status"] = _RST_PENDING
        p["existing_word"] = ""
        p["existing_gloss"] = ""

# ---------------------------------------------------------------------------
# Prepartner-dictns timeline
# ---------------------------------------------------------------------------

with st.expander("Prepartner-dictns Timeline", expanded=True):
    from spj.inference import prepartner-dictns_timeline_figure
    try:
        pose_data, _, fps = _load_pose_cached(vid["pose_path"])
        duration_sec = pose_data.shape[0] / fps
    except Exception:
        duration_sec = max((p["end_ms"] for p in preds), default=5000) / 1000.0

    fig = prepartner-dictns_timeline_figure(preds, duration_sec)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Prepartner-dictn selector + navigation
# ---------------------------------------------------------------------------

# Navigation staging key (same pattern as page 7 Review tab)
if "_ar_nav_target" in st.session_state:
    nav_target = st.session_state.pop("_ar_nav_target")
    st.session_state["ar_pred_idx"] = nav_target
    # Reset trim sliders for new prepartner-dictn
    st.session_state.pop("ar_trim_range", None)

pred_labels = []
for i, p in enumerate(preds):
    icon = _STATUS_ICONS.get(p["review_status"], "\U0001f7e1")
    hand_label = "RH" if p["hand"] == "right" else "LH"
    label = (
        f"{icon} {hand_label} {_fmt_ms(p['start_ms'])}\u2192{_fmt_ms(p['end_ms'])} "
        f"| {p['predicted_gloss']} ({p['prepartner-dictn_confidence']:.0%})"
    )
    pred_labels.append(label)

# Default to first pending prepartner-dictn if no stored index
if "ar_pred_idx" not in st.session_state:
    default_idx = 0
    for i, p in enumerate(preds):
        if p["review_status"] == _RST_PENDING:
            default_idx = i
            break
    st.session_state["ar_pred_idx"] = default_idx

# Clamp to valid range (video may have changed; selectbox may store str)
max_idx = len(preds) - 1
try:
    st.session_state["ar_pred_idx"] = int(st.session_state["ar_pred_idx"])
except (ValueError, TypeError):
    st.session_state["ar_pred_idx"] = 0
if st.session_state["ar_pred_idx"] > max_idx:
    st.session_state["ar_pred_idx"] = 0

current_idx = st.selectbox(
    "Prepartner-dictn",
    range(len(preds)),
    format_func=lambda i: pred_labels[i],
    key="ar_pred_idx",
)

# Reset trim sliders if prepartner-dictn changed (selectbox or any other way)
if st.session_state.get("_ar_prev_pred_idx") != current_idx:
    st.session_state.pop("ar_trim_range", None)
st.session_state["_ar_prev_pred_idx"] = current_idx

pred = preds[current_idx]

# ---------------------------------------------------------------------------
# Prepartner-dictn info
# ---------------------------------------------------------------------------

hand_label = "Right Hand" if pred["hand"] == "right" else "Left Hand"
status_icon = _STATUS_ICONS.get(pred["review_status"], "\U0001f7e1")

st.markdown(
    f"**{status_icon} {pred['predicted_gloss']}** ({pred['prepartner-dictn_confidence']:.0%}) "
    f"\u2014 {hand_label}  "
    f"`{_fmt_ms(pred['start_ms'])}` \u2192 `{_fmt_ms(pred['end_ms'])}`"
)

if pred["review_status"] == _RST_APPROVED:
    st.success(
        f"Already reviewed: word=`{pred['existing_word']}` "
        f"gloss=`{pred['existing_gloss']}`"
    )
elif pred["review_status"] == _RST_SKIPPED:
    st.info("Previously skipped.")

# ---------------------------------------------------------------------------
# Load pose data (needed for video player AND motion energy)
# ---------------------------------------------------------------------------

_orig_start = pred["start_ms"]
_orig_end = pred["end_ms"]
_pose_loaded = False

try:
    p_data, p_conf, p_fps = _load_pose_cached(vid["pose_path"])
    _pose_loaded = True
except Exception as exc:
    st.error(f"Error loading pose: {exc}")
    p_fps = vid["fps"]

# ---------------------------------------------------------------------------
# Timeline trim — motion energy bar + range slider (video-editor style)
# ---------------------------------------------------------------------------

# Context: show wider window around the prepartner-dictn
_duration = _orig_end - _orig_start
_context_ms = max(2000, _duration * 2)
_slider_min = max(0, _orig_start - _context_ms)
_slider_max = _orig_end + _context_ms
_step = 40  # ~1 frame at 25fps

# Motion energy visualization (cached wrist velocity)
if _pose_loaded:
    try:
        _smooth_speed, _speed_fps = _wrist_speed_cached(vid["pose_path"], pred["hand"])
        ctx_frame_start = max(0, int(_slider_min * _speed_fps / 1000))
        ctx_frame_end = min(len(_smooth_speed), int(_slider_max * _speed_fps / 1000))
        ctx_speed = _smooth_speed[ctx_frame_start:ctx_frame_end]
    except Exception:
        ctx_speed = None

    if ctx_speed is not None and len(ctx_speed) > 0:
        max_speed = ctx_speed.max() if ctx_speed.max() > 0 else 1.0
        ctx_speed_norm = ctx_speed / max_speed

        # Build compact motion energy bar as inline HTML
        n_bars = min(200, len(ctx_speed_norm))
        if n_bars > 0:
            import numpy as np
            resampled = np.interp(
                np.linspace(0, len(ctx_speed_norm) - 1, n_bars),
                np.arange(len(ctx_speed_norm)),
                ctx_speed_norm,
            )
            # Color: low motion = dark, high motion = bright green
            bar_html_parts = []
            for v in resampled:
                g = int(80 + 175 * v)
                bar_html_parts.append(
                    f'<div style="flex:1;height:100%;background:rgb(0,{g},0);'
                    f'opacity:{0.3 + 0.7 * v}"></div>'
                )
            # Mark original prepartner-dictn region
            total_ms = _slider_max - _slider_min
            orig_left_pct = (_orig_start - _slider_min) / total_ms * 100
            orig_width_pct = (_orig_end - _orig_start) / total_ms * 100

            energy_html = f"""
            <div style="position:relative;height:28px;border-radius:4px;overflow:hidden;
                        display:flex;margin:0 0 -10px 0;background:#111">
                {''.join(bar_html_parts)}
                <div style="position:absolute;left:{orig_left_pct:.1f}%;width:{orig_width_pct:.1f}%;
                            top:0;height:100%;border-left:2px solid #ff4444;border-right:2px solid #ff4444;
                            background:rgba(255,68,68,0.15);pointer-events:none"></div>
                <div style="position:absolute;left:4px;top:2px;font-size:10px;color:#888;
                            pointer-events:none">Motion ({pred['hand'][0].upper()}H)</div>
            </div>
            """
            st.components.v1.html(energy_html, height=30)

# Range slider for trim (single slider, two handles)
trim_range = st.slider(
    "Trim",
    min_value=_slider_min,
    max_value=_slider_max,
    value=(_orig_start, _orig_end),
    step=_step,
    key="ar_trim_range",
    format="%dms",
    label_visibility="collapsed",
)
trim_start, trim_end = trim_range

if trim_start >= trim_end:
    trim_start, trim_end = _orig_start, _orig_end

_use_start_ms = trim_start
_use_end_ms = trim_end

# Show trim info only if trimmed
_was_trimmed = trim_start != _orig_start or trim_end != _orig_end
if _was_trimmed:
    st.caption(
        f"Trimmed: {_fmt_ms(trim_start)}\u2192{_fmt_ms(trim_end)} "
        f"(was {_fmt_ms(_orig_start)}\u2192{_fmt_ms(_orig_end)})"
    )

# ---------------------------------------------------------------------------
# Video + Pose player
# ---------------------------------------------------------------------------

if _pose_loaded:
    try:
        frame_start = int(_use_start_ms * p_fps / 1000)
        frame_end = int(_use_end_ms * p_fps / 1000)
        frame_start = max(0, min(frame_start, p_data.shape[0] - 1))
        frame_end = max(frame_start + 1, min(frame_end, p_data.shape[0]))

        # Add context frames (0.3s before/after)
        context_frames = int(0.3 * p_fps)
        display_frame_start = max(0, frame_start - context_frames)
        display_frame_end = min(p_data.shape[0], frame_end + context_frames)
        display_start_ms = int(display_frame_start / p_fps * 1000)
        display_end_ms = int(display_frame_end / p_fps * 1000)

        video_bytes = _extract_segment_cached(
            vid["video_path"], display_start_ms, display_end_ms,
        )
        if video_bytes:
            html = synced_video_pose_html(
                video_bytes, p_data, p_conf, p_fps,
                display_frame_start, display_frame_end,
            )
            st.components.v1.html(html, height=450)
        else:
            st.warning("Could not extract video segment.")
    except Exception as exc:
        st.error(f"Error loading pose/video: {exc}")

# ---------------------------------------------------------------------------
# Action buttons
# ---------------------------------------------------------------------------

def _next_pending_idx() -> int | None:
    """Find next pending prepartner-dictn after current index."""
    for i in range(current_idx + 1, len(preds)):
        if preds[i]["review_status"] == _RST_PENDING:
            return i
    # Wrap around
    for i in range(0, current_idx):
        if preds[i]["review_status"] == _RST_PENDING:
            return i
    return None


def _write_pairing(word: str, gloss_id: str, status: str, note: str) -> None:
    """Write a pairing row using trimmed timestamps."""
    stem = vid["stem"]
    p_fps = vid["fps"]
    hand = pred["hand"]

    # Use trimmed boundaries (may differ from original prepartner-dictn)
    start_ms = _use_start_ms
    end_ms = _use_end_ms
    if start_ms != pred["start_ms"] or end_ms != pred["end_ms"]:
        note = f"{note}|trimmed_{pred['start_ms']}-{pred['end_ms']}"

    pairing_id = f"{stem}_{start_ms:08d}_{hand[0]}_R"
    segment_id = f"{stem}_ai"

    row = make_pairing_dict(
        pairing_id=pairing_id,
        segment_id=segment_id,
        video_path=vid["video_path"],
        pose_path=vid["pose_path"],
        hand=hand,
        sign_start_ms=start_ms,
        sign_end_ms=end_ms,
        sign_frame_start=int(start_ms * p_fps / 1000),
        sign_frame_end=int(end_ms * p_fps / 1000),
        fps=p_fps,
        note=note,
        word=word,
        gloss_id=gloss_id,
        status=status,
        suggestion_gloss=pred["predicted_gloss"],
        suggestion_confidence=pred["prepartner-dictn_confidence"],
    )

    # Append to pairings (save previous for undo)
    pdf = _get_pairings_df()
    previous_row = None
    if not pdf.empty:
        mask = pdf["pairing_id"] == pairing_id
        if mask.any():
            previous_row = pdf.loc[mask].iloc[0].to_dict()
        pdf = pdf[~mask]
    new_row_df = pd.DataFrame([row])
    pdf = pd.concat([pdf, new_row_df], ignore_index=True) if not pdf.empty else new_row_df
    _save_pairings(pdf)
    _push_undo(pairing_id, previous_row)

    # Advance to next pending
    nxt = _next_pending_idx()
    if nxt is not None:
        st.session_state["_ar_nav_target"] = nxt


def _do_undo() -> bool:
    """Undo last action. Returns True if undo was performed."""
    entry = _pop_undo()
    if entry is None:
        return False
    pairing_id = entry["pairing_id"]
    previous = entry["previous"]
    pdf = _get_pairings_df()
    if not pdf.empty:
        pdf = pdf[pdf["pairing_id"] != pairing_id]
    if previous is not None:
        pdf = pd.concat([pdf, pd.DataFrame([previous])], ignore_index=True)
    _save_pairings(pdf)
    return True


# Action row: [input] [Save] [Skip] [Undo] [<] [>]
act_input, act_approve, act_skip, act_undo, act_prev, act_next = st.columns(
    [2.5, 1, 0.8, 0.7, 0.4, 0.4],
)
with act_input:
    # Show known word meanings as hint
    _glossary = _get_glossary()
    _cur_gloss = pred["predicted_gloss"]
    _entry = _glossary.get_entry(_cur_gloss)
    _words_hint = ", ".join(_entry.get("forms", [])) if _entry else ""
    _placeholder = f"{_cur_gloss}"
    if _words_hint:
        _placeholder += f" ({_words_hint})"

    correct_input = st.text_input(
        "Gloss",
        key="ar_correct_input",
        placeholder=_placeholder,
        label_visibility="collapsed",
        help="Leave empty = approve AI prepartner-dictn. Type to correct: word, GLOSS-ID, or word1, word2 (comma-separated meanings).",
    )


def _resolve_words_and_gloss() -> tuple[str, str]:
    """Resolve word(s) and gloss_id from user input.

    Input field supports:
      - Empty → approve AI prepartner-dictn as-is
      - "voda" → single word (looked up in glossary for gloss_id)
      - "WATER-1" → gloss ID directly
      - "voda, water, aqua" → multiple word meanings (comma-separated)
    """
    glossary = _get_glossary()
    raw = correct_input.strip()

    if not raw:
        # Approve AI prepartner-dictn
        gloss_id = pred["predicted_gloss"]
        word = ""
        entry = glossary.get_entry(gloss_id)
        if entry:
            forms = entry.get("forms", [])
            if forms:
                word = forms[0]
        if not word and not GLOSS_RE.match(gloss_id):
            word, gloss_id = parse_gloss_value(gloss_id, glossary)
        return word, gloss_id

    # Check for comma-separated multiple words
    if "," in raw:
        parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
        if parts:
            word = ", ".join(parts)
            # Use first word to find gloss_id
            _, gloss_id = parse_gloss_value(parts[0], glossary)
            if not gloss_id:
                gloss_id = pred["predicted_gloss"]
            # Register all forms in glossary
            if gloss_id:
                for w in parts:
                    glossary.add_form(gloss_id, w)
                from spj.glossary import save_glossary
                save_glossary(glossary, GLOSSARY_JSON)
            return word, gloss_id

    # Single value — word or gloss ID
    word, gloss_id = parse_gloss_value(raw, glossary)
    return word, gloss_id


with act_approve:
    # Single "Save" action: if gloss field is empty → approve AI prepartner-dictn as-is,
    # if gloss field has text → save the corrected gloss. No separate Approve/Correct.
    _has_correction = bool(correct_input.strip())
    _btn_label = "\u2705 Save (A)" if not _has_correction else "\u270f Save (A)"
    if st.button(_btn_label, use_container_width=True, type="primary"):
        word, gloss = _resolve_words_and_gloss()
        if not gloss and not word:
            st.toast("Could not resolve gloss", icon="\u26a0\ufe0f")
        else:
            note = "ai_review_corrected" if _has_correction else "ai_review_approved"
            _write_pairing(word, gloss, PST_PAIRED, note)
            st.rerun()

with act_skip:
    if st.button("\u23ed Skip (S)", use_container_width=True):
        _write_pairing("", "", PST_SKIPPED, "ai_review_skipped")
        st.rerun()

with act_undo:
    undo_stack = st.session_state.get("ar_undo_stack", [])
    if st.button(
        f"\u21a9 ({len(undo_stack)})",
        use_container_width=True,
        disabled=len(undo_stack) == 0,
    ):
        if _do_undo():
            st.rerun()

with act_prev:
    if st.button("\u25c0", use_container_width=True):
        new_idx = max(0, current_idx - 1)
        st.session_state["_ar_nav_target"] = new_idx
        st.rerun()

with act_next:
    if st.button("\u25b6", use_container_width=True):
        new_idx = min(len(preds) - 1, current_idx + 1)
        st.session_state["_ar_nav_target"] = new_idx
        st.rerun()

# ---------------------------------------------------------------------------
# Keyboard shortcuts (A=Approve, S=Skip, Z=Undo, Arrow keys=Navigate)
# ---------------------------------------------------------------------------

st.components.v1.html("""
<script>
document.addEventListener('keydown', function(e) {
    // Skip if user is typing in an input field
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    const buttons = parent.document.querySelectorAll('button[data-testid="stBaseButton-secondary"], button[data-testid="stBaseButton-primary"]');
    function clickButton(text) {
        for (const btn of buttons) {
            if (btn.textContent.includes(text)) { btn.click(); return; }
        }
    }

    switch(e.key) {
        case 'a': case 'A': clickButton('Approve'); break;
        case 's': case 'S': e.preventDefault(); clickButton('Skip'); break;
        case 'z': case 'Z': clickButton('Undo'); break;
        case 'ArrowLeft':  clickButton('\u25c0'); break;
        case 'ArrowRight': clickButton('\u25b6'); break;
    }
});
</script>
""", height=0)

# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

n_approved = sum(1 for p in preds if p["review_status"] == _RST_APPROVED)
n_skipped = sum(1 for p in preds if p["review_status"] == _RST_SKIPPED)
n_pending = sum(1 for p in preds if p["review_status"] == _RST_PENDING)

st.divider()
cols = st.columns(4)
cols[0].metric("\u2705 Approved", n_approved)
cols[1].metric("\u23ed Skipped", n_skipped)
cols[2].metric("\U0001f7e1 Pending", n_pending)
cols[3].caption("**Shortcuts:** A=Approve, S=Skip, Z=Undo, \u2190\u2192=Navigate")

# Refresh button
if st.button("Refresh video list"):
    st.session_state.pop("ar_videos", None)
    st.session_state.pop("ar_pairings_df", None)
    _read_prepartner-dictns_cached.clear()
    st.rerun()
