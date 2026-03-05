"""AI Prediction Review — one-prediction-at-a-time annotator workflow.

Annotators see AI-predicted glosses (from Page 10 inference) with video+pose
in a unified interactive timeline. Approve / correct / skip with one click.
Approved predictions write directly to pairings.csv as PST_PAIRED rows.

Keyboard shortcuts: A=Save, S=Skip, Z=Undo, C=Cut, X=Undo Cut
"""
import base64
import logging
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from spj.glossary import load_glossary, save_glossary
from spj.inference import read_predictions_from_eaf
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
SUBTITLES_DIR  = DATA_DIR / "subtitles"
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

# Cluster parameters for stable prediction switching
_CLUSTER_MAX_GAP_MS = 15000  # max gap between predictions to group them
_CLUSTER_MAX_DUR_MS = 45000  # max cluster duration


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
def _read_predictions_cached(eaf_path_str: str) -> list[dict]:
    return read_predictions_from_eaf(Path(eaf_path_str))


@st.cache_data(ttl=600, max_entries=3)
def _motion_energy_cached(pose_path_str: str, hand: str) -> tuple[list[float], float]:
    """Normalised wrist speed as a Python list, cached with b64 avoidance."""
    from spj.preannotate import compute_motion_energy, load_pose_arrays
    data, conf, fps = load_pose_arrays(Path(pose_path_str))
    energy = compute_motion_energy(data, conf, hand)
    return energy.tolist(), fps


@st.cache_data(ttl=600)
def _load_subtitles_for_video(stem: str, video_path: str = "") -> list[dict]:
    """Load subtitles from VTT file, falling back to EAF S1_Translation."""
    # Try VTT first (via get_subtitle_status which finds soft VTTs too)
    try:
        from spj.ocr_subtitles import get_subtitle_status, read_vtt
        vp = Path(video_path) if video_path else DATA_DIR / "videos" / f"{stem}.mp4"
        status = get_subtitle_status(vp, SUBTITLES_DIR)
        vtt_path = status.get("vtt_path")
        if vtt_path:
            return read_vtt(Path(vtt_path))
    except Exception:
        pass

    # Fallback: EAF S1_Translation tier
    eaf_path = ANNOTATIONS_DIR / f"{stem}.eaf"
    try:
        from spj.eaf import load_eaf
        tier_name = "S1_Translation"
        eaf = load_eaf(eaf_path)
        if tier_name in eaf.get_tier_names():
            subs = []
            for start, end, text in eaf.get_annotation_data_for_tier(tier_name):
                if text and text.strip():
                    subs.append({"start_ms": int(start), "end_ms": int(end), "text": text.strip()})
            return subs
    except Exception:
        pass

    return []


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
    st.session_state.pop("ar_videos", None)  # invalidate video list (stale n_reviewed)


def _fmt_ms(ms: int) -> str:
    return f"{ms / 1000.0:.1f}s"


# ---------------------------------------------------------------------------
# Prediction clustering — group nearby predictions to share one video extract
# ---------------------------------------------------------------------------

def _find_prediction_cluster(
    preds: list[dict], idx: int,
    max_gap_ms: int = _CLUSTER_MAX_GAP_MS,
    max_dur_ms: int = _CLUSTER_MAX_DUR_MS,
) -> tuple[int, int]:
    """Return (cluster_start_ms, cluster_end_ms) covering prediction at idx.

    Groups adjacent predictions within max_gap_ms, capped at max_dur_ms total.
    """
    if not preds:
        return (0, 1000)

    # Sort by start time for grouping
    sorted_preds = sorted(enumerate(preds), key=lambda x: x[1]["start_ms"])
    target_start = preds[idx]["start_ms"]
    target_end = preds[idx]["end_ms"]

    cluster_start = target_start
    cluster_end = target_end

    # Expand forward
    for _, p in sorted_preds:
        if p["start_ms"] > cluster_end + max_gap_ms:
            break
        if p["end_ms"] - cluster_start > max_dur_ms:
            break
        cluster_end = max(cluster_end, p["end_ms"])

    # Expand backward
    for _, p in reversed(sorted_preds):
        if p["end_ms"] < cluster_start - max_gap_ms:
            break
        if cluster_end - p["start_ms"] > max_dur_ms:
            break
        cluster_start = min(cluster_start, p["start_ms"])

    return (cluster_start, cluster_end)


# ---------------------------------------------------------------------------
# Build dedup set from existing pairings (vectorized)
# ---------------------------------------------------------------------------

def _build_dedup_set(pairings_df: pd.DataFrame) -> dict[tuple, dict]:
    """Map (stem, start_ms, hand) -> pairing row dict for dedup.

    For AI Review pairings (pairing_id ending with '_R'), the key uses the
    original prediction start encoded in the pairing_id, not sign_start_ms
    (which may be trimmed). This ensures dedup matches predictions correctly.
    """
    if pairings_df.empty:
        return {}
    df = pairings_df
    stems = df["video_path"].fillna("").apply(lambda p: Path(str(p)).stem)
    starts = df["sign_start_ms"].fillna(0).astype(int)
    hands = df["hand"].fillna("").astype(str)
    statuses = df["status"].fillna("").astype(str)
    words = df["word"].fillna("").astype(str)
    gloss_ids = df["gloss_id"].fillna("").astype(str)
    mouthings = df["mouthing"].fillna("").astype(str) if "mouthing" in df.columns else [""] * len(df)
    pairing_ids = df["pairing_id"].fillna("").astype(str)

    result = {}
    for pid, stem, start, hand, status, word, gid, mouth in zip(
        pairing_ids, stems, starts, hands, statuses, words, gloss_ids, mouthings,
    ):
        # For AI Review pairings, extract original prediction start from pairing_id
        # Format: {stem}_{start:08d}_{hand_char}_R
        if pid.endswith("_R"):
            parts = pid.rsplit("_", 3)  # [stem..., start_08d, hand_char, R]
            if len(parts) >= 4:
                try:
                    start = int(parts[-3])
                except (ValueError, IndexError):
                    pass
        val = {"status": status, "word": word, "gloss_id": gid, "mouthing": mouth}
        result[(stem, start, hand)] = val
    return result


# ---------------------------------------------------------------------------
# Find videos with AI predictions
# ---------------------------------------------------------------------------

def _find_videos_with_predictions(
    inventory_df: pd.DataFrame, dedup: dict[tuple, dict],
) -> list[dict]:
    """Scan annotations dir for EAF files containing AI predictions."""
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
            preds = _read_predictions_cached(str(eaf_path))
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
            "n_predictions": len(preds),
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

st.header("12. 🔍 AI Prediction Review")
st.caption(
    "Review AI-predicted glosses from inference (Page 10). "
    "Approve, correct, or skip each prediction. "
    "Approved predictions feed directly into retraining."
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

# Find videos with predictions (cached at session level)
if "ar_videos" not in st.session_state:
    with st.spinner("Scanning for videos with AI predictions..."):
        st.session_state["ar_videos"] = _find_videos_with_predictions(
            inventory_df, dedup,
        )

videos = st.session_state["ar_videos"]

if not videos:
    st.info("No videos with AI predictions found. Run inference on Page 10 first.")
    st.stop()

# Sort: most unreviewed first
videos.sort(key=lambda v: v["n_predictions"] - v["n_reviewed"], reverse=True)

# ---------------------------------------------------------------------------
# Video selector
# ---------------------------------------------------------------------------

video_labels = [
    f"{v['stem']} \u2014 {v['n_predictions']} predictions ({v['n_reviewed']} reviewed)"
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
# Reset prediction index if video changed
prev_stem = st.session_state.get("ar_video_stem")
if prev_stem and prev_stem != vid["stem"]:
    st.session_state.pop("ar_pred_idx", None)
    st.session_state.pop("_ar_nav_target", None)
    st.session_state.pop("ar_timeline_result", None)
    st.session_state.pop("_ar_trims", None)
    st.session_state.pop("_ar_cuts", None)
st.session_state["ar_video_stem"] = vid["stem"]
preds = _read_predictions_cached(vid["eaf_path"])

with col_stats:
    n_rev = vid["n_reviewed"]
    n_tot = vid["n_predictions"]
    st.metric("Reviewed", f"{n_rev}/{n_tot}")

if n_tot > 0:
    st.progress(n_rev / n_tot)

# ---------------------------------------------------------------------------
# Enrich predictions with review status from pairings
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
        p["existing_mouthing"] = existing.get("mouthing", "")
    else:
        p["review_status"] = _RST_PENDING
        p["existing_word"] = ""
        p["existing_gloss"] = ""
        p["existing_mouthing"] = ""

# ---------------------------------------------------------------------------
# Prediction selector + navigation
# ---------------------------------------------------------------------------

# Handle navigation from timeline component (prediction click)
timeline_result = st.session_state.get("ar_timeline_result")
if timeline_result and isinstance(timeline_result, dict) and timeline_result.get("mode") != "new_label":
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
        f"| {p['predicted_gloss']} ({p['prediction_confidence']:.0%})"
    )
    pred_labels.append(label)

# Default to first pending prediction if no stored index
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
    "Prediction",
    range(len(preds)),
    format_func=lambda i: pred_labels[i],
    key="ar_pred_idx",
)

pred = preds[current_idx]

# ---------------------------------------------------------------------------
# Prediction info
# ---------------------------------------------------------------------------

hand_label = "Right Hand" if pred["hand"] == "right" else "Left Hand"
status_icon = _STATUS_ICONS.get(pred["review_status"], "\U0001f7e1")

st.markdown(
    f"**{status_icon} {pred['predicted_gloss']}** ({pred['prediction_confidence']:.0%}) "
    f"\u2014 {hand_label}  "
    f"`{_fmt_ms(pred['start_ms'])}` \u2192 `{_fmt_ms(pred['end_ms'])}`"
)

if pred["review_status"] == _RST_APPROVED:
    _rev_info = (
        f"Already reviewed: word=`{pred['existing_word']}` "
        f"gloss=`{pred['existing_gloss']}`"
    )
    if pred.get("existing_mouthing"):
        _rev_info += f" mouthing=`{pred['existing_mouthing']}`"
    st.success(_rev_info)
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

# ---------------------------------------------------------------------------
# Unified Timeline Component (video + pose + interactive timeline)
# ---------------------------------------------------------------------------

# Trim values — check persisted trims first, fall back to original prediction bounds
_trims = st.session_state.get("_ar_trims", {})
if current_idx in _trims:
    _use_start_ms, _use_end_ms = _trims[current_idx]
else:
    _use_start_ms = _orig_start
    _use_end_ms = _orig_end

if _pose_loaded:
    try:
        # Find cluster of nearby predictions for stable switching
        cluster_start, cluster_end = _find_prediction_cluster(preds, current_idx)

        # Add 1s padding to cluster bounds
        cluster_start = max(0, cluster_start - 1000)
        total_frames = p_data.shape[0]
        total_dur_ms = int(total_frames / p_fps * 1000)
        cluster_end = min(total_dur_ms, cluster_end + 1000)

        frame_start = max(0, int(cluster_start * p_fps / 1000))
        frame_end = min(total_frames, int(cluster_end * p_fps / 1000))
        frame_end = max(frame_start + 1, frame_end)

        display_start_ms = int(frame_start / p_fps * 1000)
        display_end_ms = int(frame_end / p_fps * 1000)

        video_b64 = _extract_segment_b64(
            vid["video_path"], display_start_ms, display_end_ms,
        )

        if video_b64:
            pos_b64, conf_b64, n_frames = encode_pose_data(
                p_data, p_conf, frame_start, frame_end,
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
                    "prediction_confidence": p["prediction_confidence"],
                    "review_status": p.get("review_status", _RST_PENDING),
                }
                for p in preds
            ]

            # Load subtitles (VTT or EAF fallback), filtered to cluster range
            all_subs = _load_subtitles_for_video(vid["stem"], vid["video_path"])
            subs = [
                s for s in all_subs
                if s["end_ms"] > display_start_ms - 2000
                and s["start_ms"] < display_end_ms + 2000
            ]

            # Restore persisted cut points for this prediction
            _persisted_cuts = st.session_state.get("_ar_cuts", {}).get(current_idx, [])

            result = spj_timeline(
                video_b64=video_b64,
                pos_b64=pos_b64,
                conf_b64=conf_b64,
                n_frames=n_frames,
                fps=p_fps,
                duration_ms=duration_ms,
                predictions=preds_json,
                motion_energy=motion_energy,
                energy_fps=energy_fps,
                current_pred_idx=current_idx,
                connections=CONNECTION_ARRAYS,
                segment_start_ms=display_start_ms,
                segment_end_ms=display_end_ms,
                subtitles=subs,
                loaded_start_ms=display_start_ms,
                loaded_end_ms=display_end_ms,
                initial_trim_start_ms=_use_start_ms,
                initial_trim_end_ms=_use_end_ms,
                initial_cut_points=_persisted_cuts,
                key="spj_timeline",
                height=560,
            )

            # Read trim values from fresh result and persist in session state
            if result is not None:
                st.session_state["ar_timeline_result"] = result
                _tr_s = result.get("trim_start_ms")
                _tr_e = result.get("trim_end_ms")
                _tl_sel = result.get("selected_pred_idx")
                if (_tl_sel == current_idx and
                    _tr_s is not None and _tr_e is not None and _tr_s < _tr_e):
                    _use_start_ms = int(_tr_s)
                    _use_end_ms = int(_tr_e)
                    # Persist trim for this prediction
                    trims = st.session_state.setdefault("_ar_trims", {})
                    trims[current_idx] = (_use_start_ms, _use_end_ms)
                # Persist cut points for this prediction (cap at 20 entries)
                _res_cuts = result.get("cut_points_ms") or []
                cuts_store = st.session_state.setdefault("_ar_cuts", {})
                if _res_cuts:
                    cuts_store[current_idx] = [int(c) for c in _res_cuts]
                    if len(cuts_store) > 20:
                        oldest = min(k for k in cuts_store if k != current_idx)
                        cuts_store.pop(oldest, None)
                else:
                    cuts_store.pop(current_idx, None)
        else:
            st.warning("Could not extract video segment.")
    except Exception as exc:
        st.error(f"Error loading pose/video: {exc}")

# Store trim values in session state for action buttons
st.session_state["_ar_use_start_ms"] = _use_start_ms
st.session_state["_ar_use_end_ms"] = _use_end_ms

# Show trim info if trimmed
_was_trimmed = _use_start_ms != _orig_start or _use_end_ms != _orig_end
if _was_trimmed:
    st.caption(
        f"Trimmed: {_fmt_ms(_use_start_ms)}\u2192{_fmt_ms(_use_end_ms)} "
        f"(was {_fmt_ms(_orig_start)}\u2192{_fmt_ms(_orig_end)})"
    )

# ---------------------------------------------------------------------------
# Cut/Split mode — split prediction into multiple signs
# ---------------------------------------------------------------------------

_cut_points: list[int] = []
_timeline_res = st.session_state.get("ar_timeline_result")
if _timeline_res and isinstance(_timeline_res, dict) and _timeline_res.get("mode") != "new_label":
    _raw_cuts = _timeline_res.get("cut_points_ms") or []
    _cut_points = sorted(
        int(cp) for cp in _raw_cuts if _use_start_ms < cp < _use_end_ms
    )

if _cut_points:
    boundaries = [_use_start_ms] + _cut_points + [_use_end_ms]
    segments = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

    st.markdown(f"**Cut into {len(segments)} segments** (C=add cut, X=undo cut)")

    _cut_glossary = _get_glossary()
    _cut_glosses: list[str] = []
    _cut_mouths: list[str] = []

    for si, (seg_s, seg_e) in enumerate(segments):
        c1, c2 = st.columns([2, 1.5])
        with c1:
            g = st.text_input(
                f"Seg {si+1} gloss ({_fmt_ms(seg_s)}\u2192{_fmt_ms(seg_e)}, {seg_e-seg_s}ms)",
                key=f"cut_gloss_{si}",
                placeholder=pred["predicted_gloss"] if si == 0 else "Gloss",
            )
            _cut_glosses.append(g)
        with c2:
            m = st.text_input(
                f"Mouthing {si+1}",
                key=f"cut_mouth_{si}",
                placeholder="Mouthing",
                label_visibility="collapsed",
            )
            _cut_mouths.append(m)

    c_save, c_clear = st.columns([1, 1])
    with c_save:
        if st.button("Save All Segments", type="primary", use_container_width=True):
            _all_ok = True
            _resolved: list[tuple[str, str]] = []
            for si, raw in enumerate(_cut_glosses):
                raw = raw.strip()
                fallback = pred["predicted_gloss"] if si == 0 else ""
                if not raw and si > 0:
                    st.toast(f"Segment {si+1} is empty", icon="\u26a0\ufe0f")
                    _all_ok = False
                    break
                w, g = _resolve_gloss_input(raw, fallback, _cut_glossary)
                if w or g:
                    _resolved.append((w, g))
                else:
                    st.toast(f"Cannot resolve segment {si+1}: {raw}", icon="\u26a0\ufe0f")
                    _all_ok = False
                    break

            if _all_ok and _resolved:
                stem = vid["stem"]
                p_fps_val = vid["fps"]
                hand = pred["hand"]
                # Only annotate trim if overall bounds differ from prediction
                _pred_trimmed = (
                    _use_start_ms != pred["start_ms"]
                    or _use_end_ms != pred["end_ms"]
                )
                rows: list[dict] = []
                for si, ((seg_s, seg_e), (word, gloss_id)) in enumerate(
                    zip(segments, _resolved)
                ):
                    suffix = "_R" if si == 0 else "_C"
                    pid_start = pred["start_ms"] if si == 0 else seg_s
                    pairing_id = f"{stem}_{pid_start:08d}_{hand[0]}{suffix}"
                    note = "ai_review_cut"
                    if _pred_trimmed:
                        note += f"|trimmed_{pred['start_ms']}-{pred['end_ms']}"

                    rows.append(make_pairing_dict(
                        pairing_id=pairing_id,
                        segment_id=f"{stem}_ai",
                        video_path=vid["video_path"],
                        pose_path=vid["pose_path"],
                        hand=hand,
                        sign_start_ms=seg_s,
                        sign_end_ms=seg_e,
                        sign_frame_start=int(seg_s * p_fps_val / 1000),
                        sign_frame_end=int(seg_e * p_fps_val / 1000),
                        fps=p_fps_val,
                        note=note,
                        word=word,
                        gloss_id=gloss_id,
                        status=PST_PAIRED,
                        suggestion_gloss=pred["predicted_gloss"],
                        suggestion_confidence=pred["prediction_confidence"],
                        mouthing=_cut_mouths[si].strip(),
                    ))

                # Batch upsert — single CSV write
                pdf = _get_pairings_df()
                pids = {r["pairing_id"] for r in rows}
                if not pdf.empty:
                    pdf = pdf[~pdf["pairing_id"].isin(pids)]
                new_df = pd.DataFrame(rows)
                pdf = pd.concat([pdf, new_df], ignore_index=True) if not pdf.empty else new_df
                _save_pairings(pdf)
                for r in rows:
                    _push_undo(r["pairing_id"], None)

                # Clear persisted cuts
                cuts_store = st.session_state.get("_ar_cuts", {})
                cuts_store.pop(current_idx, None)
                _advance_after_save()
                st.rerun()

    with c_clear:
        if st.button("Clear Cuts", use_container_width=True):
            # Force timeline to re-render without cuts
            st.session_state.pop("ar_timeline_result", None)
            cuts_store = st.session_state.get("_ar_cuts", {})
            cuts_store.pop(current_idx, None)
            st.rerun()

# ---------------------------------------------------------------------------
# New Label mode (Alt+drag on timeline creates custom region)
# ---------------------------------------------------------------------------

_new_label_result = st.session_state.get("ar_timeline_result")
_is_new_label = (
    _new_label_result
    and isinstance(_new_label_result, dict)
    and _new_label_result.get("mode") == "new_label"
)

if _is_new_label:
    _nl_start = int(_new_label_result["custom_start_ms"])
    _nl_end = int(_new_label_result["custom_end_ms"])
    _nl_hand = _new_label_result["custom_hand"]
    _nl_hand_label = "RH" if _nl_hand == "right" else "LH"

    st.markdown(
        f"**New label region:** {_nl_hand_label} "
        f"`{_fmt_ms(_nl_start)}` \u2192 `{_fmt_ms(_nl_end)}`"
    )

    nl_col_input, nl_col_mouth, nl_col_btn, nl_col_cancel = st.columns([2, 1.5, 1, 1])
    with nl_col_input:
        new_label_input = st.text_input(
            "Gloss for new label",
            key="ar_new_label_input",
            placeholder="Type word or GLOSS-ID",
            label_visibility="collapsed",
        )
    with nl_col_mouth:
        new_label_mouthing = st.text_input(
            "Mouthing for new label",
            key="ar_new_label_mouthing",
            placeholder="Mouthing",
            label_visibility="collapsed",
        )
    with nl_col_btn:
        if st.button("Add Label", use_container_width=True, type="primary"):
            if new_label_input.strip():
                _glossary = _get_glossary()
                word, gloss_id = parse_gloss_value(new_label_input.strip(), _glossary)
                if word or gloss_id:
                    p_fps_val = vid["fps"]
                    row = make_pairing_dict(
                        pairing_id=f"{vid['stem']}_{_nl_start:08d}_{_nl_hand[0]}_N",
                        segment_id=f"{vid['stem']}_manual",
                        video_path=vid["video_path"],
                        pose_path=vid["pose_path"],
                        hand=_nl_hand,
                        sign_start_ms=_nl_start,
                        sign_end_ms=_nl_end,
                        sign_frame_start=int(_nl_start * p_fps_val / 1000),
                        sign_frame_end=int(_nl_end * p_fps_val / 1000),
                        fps=p_fps_val,
                        note="ai_review_new_label",
                        word=word,
                        gloss_id=gloss_id,
                        status=PST_PAIRED,
                        suggestion_gloss="",
                        suggestion_confidence=0.0,
                        mouthing=new_label_mouthing.strip(),
                    )
                    _upsert_pairing(row)
                    st.session_state.pop("ar_timeline_result", None)
                    st.rerun()
                else:
                    st.toast("Could not resolve gloss", icon="\u26a0\ufe0f")
            else:
                st.toast("Enter a word or gloss ID", icon="\u26a0\ufe0f")
    with nl_col_cancel:
        if st.button("Cancel", use_container_width=True):
            st.session_state.pop("ar_timeline_result", None)
            st.rerun()

# ---------------------------------------------------------------------------
# Action buttons
# ---------------------------------------------------------------------------

def _next_pending_idx() -> int | None:
    """Find next pending prediction after current index."""
    for i in range(current_idx + 1, len(preds)):
        if preds[i]["review_status"] == _RST_PENDING:
            return i
    for i in range(0, current_idx):
        if preds[i]["review_status"] == _RST_PENDING:
            return i
    return None


def _upsert_pairing(row: dict) -> None:
    """Insert or replace a pairing row by pairing_id, with undo support."""
    pairing_id = row["pairing_id"]
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


def _advance_after_save() -> None:
    """Clear persisted trim and advance to next pending prediction."""
    trims = st.session_state.get("_ar_trims", {})
    trims.pop(current_idx, None)
    nxt = _next_pending_idx()
    if nxt is not None:
        st.session_state["_ar_nav_target"] = nxt


def _write_pairing(word: str, gloss_id: str, status: str, note: str, mouthing: str = "") -> None:
    """Write a pairing row using trimmed timestamps."""
    stem = vid["stem"]
    p_fps_val = vid["fps"]
    hand = pred["hand"]

    # Read fresh trim values from session state (set after spj_timeline returns)
    start_ms = st.session_state.get("_ar_use_start_ms", _orig_start)
    end_ms = st.session_state.get("_ar_use_end_ms", _orig_end)
    if start_ms != pred["start_ms"] or end_ms != pred["end_ms"]:
        note = f"{note}|trimmed_{pred['start_ms']}-{pred['end_ms']}"

    # Use ORIGINAL prediction start for pairing_id so dedup key matches
    pairing_id = f"{stem}_{pred['start_ms']:08d}_{hand[0]}_R"

    row = make_pairing_dict(
        pairing_id=pairing_id,
        segment_id=f"{stem}_ai",
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
        suggestion_confidence=pred["prediction_confidence"],
        mouthing=mouthing,
    )
    _upsert_pairing(row)
    _advance_after_save()


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
        help="Leave empty = approve AI prediction. Type to correct: word, GLOSS-ID, or word1, word2 (comma-separated meanings).",
    )

# Mouthing row — set default from existing pairing when prediction changes
_mouth_default = pred.get("existing_mouthing", "")
if st.session_state.get("_ar_mouth_pred_idx") != current_idx:
    st.session_state["ar_mouthing_input"] = _mouth_default
    st.session_state["_ar_mouth_pred_idx"] = current_idx
mouthing_input = st.text_input(
    "Mouthing",
    key="ar_mouthing_input",
    placeholder="Mouthing (e.g. látka, koberec)",
    label_visibility="collapsed",
    help="Slovak mouthing for this sign (lowercase). Different mouthings distinguish homonymous signs.",
)


def _resolve_gloss_input(raw: str, fallback_gloss: str, glossary) -> tuple[str, str]:
    """Resolve raw text input to (word, gloss_id), defaulting to fallback_gloss when empty."""
    if not raw:
        gloss_id = fallback_gloss
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
                gloss_id = fallback_gloss
            if gloss_id:
                for w in parts:
                    glossary.add_form(gloss_id, w)
                save_glossary(glossary, GLOSSARY_JSON)
            return word, gloss_id

    return parse_gloss_value(raw, glossary)


with act_approve:
    _has_correction = bool(correct_input.strip())
    _btn_label = "\u2705 Save (A)" if not _has_correction else "\u270f Save (A)"
    if st.button(_btn_label, use_container_width=True, type="primary"):
        word, gloss = _resolve_gloss_input(correct_input.strip(), pred["predicted_gloss"], _get_glossary())
        if not gloss and not word:
            st.toast("Could not resolve gloss", icon="\u26a0\ufe0f")
        else:
            note = "ai_review_corrected" if _has_correction else "ai_review_approved"
            _write_pairing(word, gloss, PST_PAIRED, note, mouthing=mouthing_input.strip())
            st.rerun()

with act_skip:
    if st.button("\u23ed Skip (S)", use_container_width=True):
        _write_pairing("", "", PST_SKIPPED, "ai_review_skipped", mouthing="")
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
        case 'm': case 'M':
            e.preventDefault();
            var mouthInputs = parent.document.querySelectorAll('input[aria-label="Mouthing"]');
            if (mouthInputs.length > 0) mouthInputs[0].focus();
            break;
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
cols[3].caption(
    "**Shortcuts:** A=Save, S=Skip, Z=Undo, M=Mouthing, C=Cut, X=Undo Cut, \u2190\u2192=Navigate  \n"
    "**Timeline:** Drag=Pan, Scroll=HScroll, Ctrl+Scroll=Zoom, Alt+Drag=New Label"
)

if st.button("Refresh video list"):
    for k in ("ar_videos", "ar_pairings_df", "ar_dedup", "ar_timeline_result"):
        st.session_state.pop(k, None)
    _read_predictions_cached.clear()
    st.rerun()
