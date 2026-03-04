"""AI Prepartner-dictn Review — one-prepartner-dictn-at-a-time annotator workflow.

Annotators see AI-predicted glosses (from Page 10 inference) with video+pose
in a unified interactive timeline. Approve / correct / skip with one click.
Approved prepartner-dictns write directly to pairings.csv as PST_PAIRED rows.

Keyboard shortcuts: A=Save, S=Skip, Z=Undo
"""
import base64
import logging
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from spj.glossary import load_glossary
from spj.inference import read_prepartner-dictns_from_eaf
from spj.training_data import (
    CONNECTION_ARRAYS,
    GLOSS_RE,
    PST_PAIRED,
    PST_SKIPPED,
    encode_pose_data,
    extract_video_segment,
    load_pairings_csv,
    make_pairing_dict,
    parse_gloss_value,
    save_pairings_csv,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from components.timeline import spj_timeline

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
def _extract_segment_b64(
    video_path_str: str, start_ms: int, end_ms: int,
) -> str | None:
    """Extract video segment and return as base64 string (cached)."""
    raw = extract_video_segment(Path(video_path_str), start_ms, end_ms)
    if raw is None:
        return None
    return base64.b64encode(raw).decode("ascii")


@st.cache_data(ttl=300)
def _read_prepartner-dictns_cached(eaf_path_str: str) -> list[dict]:
    return read_prepartner-dictns_from_eaf(Path(eaf_path_str))


@st.cache_data(ttl=600, max_entries=3)
def _motion_energy_cached(pose_path_str: str, hand: str) -> tuple[list[float], float]:
    """Normalised wrist speed as a Python list, cached with b64 avoidance."""
    from spj.preannotate import _wrist_speed, _gaussian_smooth, load_pose_arrays
    data, conf, fps = load_pose_arrays(Path(pose_path_str))
    if hand == "right":
        speed = _wrist_speed(data, conf, 16, 54)
    else:
        speed = _wrist_speed(data, conf, 15, 33)
    smooth = _gaussian_smooth(speed, sigma=2.0)
    max_val = smooth.max() if smooth.max() > 0 else 1.0
    return (smooth / max_val).tolist(), fps


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
    st.session_state.pop("ar_dedup", None)  # invalidate cached dedup


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
    stack = st.session_state.setdefault("ar_undo_stack", [])
    stack.append({"pairing_id": pairing_id, "previous": previous_row})
    if len(stack) > 50:
        stack[:] = stack[-50:]


def _pop_undo() -> dict | None:
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
if "ar_dedup" not in st.session_state:
    st.session_state["ar_dedup"] = _build_dedup_set(pairings_df)
dedup = st.session_state["ar_dedup"]

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
# Video selector
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
    st.session_state.pop("ar_timeline_result", None)
st.session_state["ar_video_stem"] = vid["stem"]
preds = _read_prepartner-dictns_cached(vid["eaf_path"])

with col_stats:
    n_rev = vid["n_reviewed"]
    n_tot = vid["n_prepartner-dictns"]
    st.metric("Reviewed", f"{n_rev}/{n_tot}")

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
# Prepartner-dictn selector + navigation
# ---------------------------------------------------------------------------

# Handle navigation from timeline component (prepartner-dictn click)
timeline_result = st.session_state.get("ar_timeline_result")
if timeline_result and isinstance(timeline_result, dict):
    sel_idx = timeline_result.get("selected_pred_idx")
    if sel_idx is not None and sel_idx != st.session_state.get("ar_pred_idx"):
        st.session_state["_ar_nav_target"] = sel_idx

# Navigation staging key (same pattern as page 7 Review tab)
if "_ar_nav_target" in st.session_state:
    nav_target = st.session_state.pop("_ar_nav_target")
    st.session_state["ar_pred_idx"] = nav_target

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

# Clamp to valid range
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
# Load pose data + prepare video segment for timeline component
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

# Get trim values from timeline result, or default to prepartner-dictn bounds
_use_start_ms = _orig_start
_use_end_ms = _orig_end
if timeline_result and isinstance(timeline_result, dict):
    _tl_sel = timeline_result.get("selected_pred_idx")
    if _tl_sel == current_idx:
        _tr_s = timeline_result.get("trim_start_ms")
        _tr_e = timeline_result.get("trim_end_ms")
        if _tr_s is not None and _tr_e is not None and _tr_s < _tr_e:
            _use_start_ms = int(_tr_s)
            _use_end_ms = int(_tr_e)

# ---------------------------------------------------------------------------
# Unified Timeline Component (video + pose + interactive timeline)
# ---------------------------------------------------------------------------

if _pose_loaded:
    try:
        # Prepare video segment (with context for the current prepartner-dictn)
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

        video_b64 = _extract_segment_b64(
            vid["video_path"], display_start_ms, display_end_ms,
        )

        if video_b64:
            pos_b64, conf_b64, n_frames = encode_pose_data(
                p_data, p_conf, display_frame_start, display_frame_end,
            )

            # Motion energy (cached as normalised list)
            motion_energy: list[float] = []
            energy_fps = p_fps
            try:
                motion_energy, energy_fps = _motion_energy_cached(
                    vid["pose_path"], pred["hand"],
                )
            except Exception:
                pass

            duration_ms = int(p_data.shape[0] / p_fps * 1000)

            preds_json = [
                {
                    "start_ms": p["start_ms"],
                    "end_ms": p["end_ms"],
                    "hand": p["hand"],
                    "predicted_gloss": p["predicted_gloss"],
                    "prepartner-dictn_confidence": p["prepartner-dictn_confidence"],
                    "review_status": p.get("review_status", _RST_PENDING),
                }
                for p in preds
            ]

            result = spj_timeline(
                video_b64=video_b64,
                pos_b64=pos_b64,
                conf_b64=conf_b64,
                n_frames=n_frames,
                fps=p_fps,
                duration_ms=duration_ms,
                prepartner-dictns=preds_json,
                motion_energy=motion_energy,
                energy_fps=energy_fps,
                current_pred_idx=current_idx,
                connections=CONNECTION_ARRAYS,
                segment_start_ms=display_start_ms,
                segment_end_ms=display_end_ms,
                key="spj_timeline",
                height=520,
            )

            if result is not None:
                st.session_state["ar_timeline_result"] = result
        else:
            st.warning("Could not extract video segment.")
    except Exception as exc:
        st.error(f"Error loading pose/video: {exc}")

# Show trim info if trimmed
_was_trimmed = _use_start_ms != _orig_start or _use_end_ms != _orig_end
if _was_trimmed:
    st.caption(
        f"Trimmed: {_fmt_ms(_use_start_ms)}\u2192{_fmt_ms(_use_end_ms)} "
        f"(was {_fmt_ms(_orig_start)}\u2192{_fmt_ms(_orig_end)})"
    )

# ---------------------------------------------------------------------------
# Action buttons
# ---------------------------------------------------------------------------

def _next_pending_idx() -> int | None:
    """Find next pending prepartner-dictn after current index."""
    for i in range(current_idx + 1, len(preds)):
        if preds[i]["review_status"] == _RST_PENDING:
            return i
    for i in range(0, current_idx):
        if preds[i]["review_status"] == _RST_PENDING:
            return i
    return None


def _write_pairing(word: str, gloss_id: str, status: str, note: str) -> None:
    """Write a pairing row using trimmed timestamps."""
    stem = vid["stem"]
    p_fps_val = vid["fps"]
    hand = pred["hand"]

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
        sign_frame_start=int(start_ms * p_fps_val / 1000),
        sign_frame_end=int(end_ms * p_fps_val / 1000),
        fps=p_fps_val,
        note=note,
        word=word,
        gloss_id=gloss_id,
        status=status,
        suggestion_gloss=pred["predicted_gloss"],
        suggestion_confidence=pred["prepartner-dictn_confidence"],
    )

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
    glossary = _get_glossary()
    raw = correct_input.strip()

    if not raw:
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

    if "," in raw:
        parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
        if parts:
            word = ", ".join(parts)
            _, gloss_id = parse_gloss_value(parts[0], glossary)
            if not gloss_id:
                gloss_id = pred["predicted_gloss"]
            if gloss_id:
                for w in parts:
                    glossary.add_form(gloss_id, w)
                from spj.glossary import save_glossary
                save_glossary(glossary, GLOSSARY_JSON)
            return word, gloss_id

    word, gloss_id = parse_gloss_value(raw, glossary)
    return word, gloss_id


with act_approve:
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
# Keyboard shortcuts (A=Save, S=Skip, Z=Undo)
# ---------------------------------------------------------------------------

st.components.v1.html("""
<script>
document.addEventListener('keydown', function(e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    const buttons = parent.document.querySelectorAll('button[data-testid="stBaseButton-secondary"], button[data-testid="stBaseButton-primary"]');
    function clickButton(text) {
        for (const btn of buttons) {
            if (btn.textContent.includes(text)) { btn.click(); return; }
        }
    }

    switch(e.key) {
        case 'a': case 'A': clickButton('Save'); break;
        case 's': case 'S': e.preventDefault(); clickButton('Skip'); break;
        case 'z': case 'Z': clickButton('Undo'); break;
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
cols[3].caption("**Shortcuts:** A=Save, S=Skip, Z=Undo, \u2190\u2192=Navigate")

if st.button("Refresh video list"):
    for k in ("ar_videos", "ar_pairings_df", "ar_dedup", "ar_timeline_result"):
        st.session_state.pop(k, None)
    _read_prepartner-dictns_cached.clear()
    st.rerun()
