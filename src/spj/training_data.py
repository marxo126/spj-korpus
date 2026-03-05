"""Training data pipeline: pose-subtitle alignment, human review, and NPZ export.

Aligns .pose keypoint files with .vtt subtitle entries to produce labelled
training segments for SignBERT / OpenHands fine-tuning.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Skeleton connection constants (for 2D pose visualisation)
# ---------------------------------------------------------------------------

# Body landmark connections — upper body signing space.
# Includes arms (shoulder→elbow→hand_wrist) for body direction cues.
# Uses hand-model wrists (33=left, 54=right) instead of body-model
# wrists (15, 16) which diverge from hand positions.
# NOTE: elbows (13-14) are drawn but NOT included in viewport calculation
# to keep the viewport tight around the signing space.
BODY_CONNECTIONS = [
    (11, 12),           # shoulder span → body orientation
    (11, 13), (13, 33), # left arm: shoulder→elbow→hand_wrist
    (12, 14), (14, 54), # right arm: shoulder→elbow→hand_wrist
    (0, 11), (0, 12),   # nose→shoulders
]

# Hand landmark connections (MediaPipe hand model, 21 landmarks per hand)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

# Block offsets within the 543-landmark array
# body = 0..32 (33), left_hand = 33..53 (21), right_hand = 54..74 (21), face = 75..542 (468)
_BODY_OFFSET       = 0
_LEFT_HAND_OFFSET  = 33
_RIGHT_HAND_OFFSET = 54
_FACE_OFFSET       = 75

# ---------------------------------------------------------------------------
# Face landmark connections — official MediaPipe face_mesh_connections.py
# Source: github.com/google-ai-edge/mediapipe — face-local indices (0-467)
# _FACE_OFFSET is added at draw time.
# ---------------------------------------------------------------------------

# Lips (40 edges: outer + inner contour)
_FACE_LIPS = [
    (61,146),(146,91),(91,181),(181,84),(84,17),(17,314),(314,405),
    (405,321),(321,375),(375,291),(61,185),(185,40),(40,39),(39,37),
    (37,0),(0,267),(267,269),(269,270),(270,409),(409,291),
    (78,95),(95,88),(88,178),(178,87),(87,14),(14,317),(317,402),
    (402,318),(318,324),(324,308),(78,191),(191,80),(80,81),(81,82),
    (82,13),(13,312),(312,311),(311,310),(310,415),(415,308),
]
# Left eye (16 edges)
_FACE_LEFT_EYE = [
    (263,249),(249,390),(390,373),(373,374),(374,380),(380,381),
    (381,382),(382,362),(263,466),(466,388),(388,387),(387,386),
    (386,385),(385,384),(384,398),(398,362),
]
# Right eye (16 edges)
_FACE_RIGHT_EYE = [
    (33,7),(7,163),(163,144),(144,145),(145,153),(153,154),
    (154,155),(155,133),(33,246),(246,161),(161,160),(160,159),
    (159,158),(158,157),(157,173),(173,133),
]
# Left eyebrow (8 edges)
_FACE_LEFT_EYEBROW = [
    (276,283),(283,282),(282,295),(295,285),
    (300,293),(293,334),(334,296),(296,336),
]
# Right eyebrow (8 edges)
_FACE_RIGHT_EYEBROW = [
    (46,53),(53,52),(52,65),(65,55),
    (70,63),(63,105),(105,66),(66,107),
]
# Face oval (36 edges: closed loop around jaw + forehead)
_FACE_OVAL = [
    (10,338),(338,297),(297,332),(332,284),(284,251),(251,389),
    (389,356),(356,454),(454,323),(323,361),(361,288),(288,397),
    (397,365),(365,379),(379,378),(378,400),(400,377),(377,152),
    (152,148),(148,176),(176,149),(149,150),(150,136),(136,172),
    (172,58),(58,132),(132,93),(93,234),(234,127),(127,162),
    (162,21),(21,54),(54,103),(103,67),(67,109),(109,10),
]


# ---------------------------------------------------------------------------
# Landmark selection for training — drop irrelevant face mesh points
# ---------------------------------------------------------------------------

def _unique_face_indices(*edge_lists) -> list[int]:
    """Collect unique vertex indices from connection edge lists."""
    pts: set[int] = set()
    for edges in edge_lists:
        for a, b in edges:
            pts.add(a)
            pts.add(b)
    return sorted(pts)


# Face-local indices (0-467) for SL-relevant regions only
_SL_FACE_LIPS_LOCAL = _unique_face_indices(_FACE_LIPS)
_SL_FACE_LEFT_EYE_LOCAL = _unique_face_indices(_FACE_LEFT_EYE)
_SL_FACE_RIGHT_EYE_LOCAL = _unique_face_indices(_FACE_RIGHT_EYE)
_SL_FACE_LEFT_EYEBROW_LOCAL = _unique_face_indices(_FACE_LEFT_EYEBROW)
_SL_FACE_RIGHT_EYEBROW_LOCAL = _unique_face_indices(_FACE_RIGHT_EYEBROW)

# Add nose tip + nose bridge for face orientation (face-local indices)
_NOSE_LANDMARKS = [1, 2, 4, 5, 6, 168, 197]  # tip, bridge, sides

# Combined face-local indices for each preset level
_SL_FACE_LOCAL_COMPACT = sorted(
    set(_SL_FACE_LIPS_LOCAL) | set(_NOSE_LANDMARKS)
)
_SL_FACE_LOCAL_EXTENDED = sorted(
    set(_SL_FACE_LIPS_LOCAL) | set(_NOSE_LANDMARKS)
    | set(_SL_FACE_LEFT_EYE_LOCAL) | set(_SL_FACE_RIGHT_EYE_LOCAL)
    | set(_SL_FACE_LEFT_EYEBROW_LOCAL) | set(_SL_FACE_RIGHT_EYEBROW_LOCAL)
)
# Full preset uses the same face set as extended (all SL-relevant face regions)
_SL_FACE_LOCAL_FULL = _SL_FACE_LOCAL_EXTENDED
# Legacy alias used by other code
_SL_FACE_LOCAL = _SL_FACE_LOCAL_FULL

# Global indices (within the 543-landmark array)
_SL_FACE_GLOBAL_COMPACT = [_FACE_OFFSET + i for i in _SL_FACE_LOCAL_COMPACT]
_SL_FACE_GLOBAL_EXTENDED = [_FACE_OFFSET + i for i in _SL_FACE_LOCAL_EXTENDED]
_SL_FACE_GLOBAL = [_FACE_OFFSET + i for i in _SL_FACE_LOCAL]

# Body subset for compact/extended: only signing-relevant landmarks
_COMPACT_BODY = [0, 11, 12, 13, 14, 15, 16]  # nose, shoulders, elbows, wrists

# ---------------------------------------------------------------------------
# Landmark presets
# ---------------------------------------------------------------------------

# FULL: body(33) + hands(42) + lips + eyes + eyebrows + nose = 174
SL_LANDMARK_INDICES_FULL = (
    list(range(_BODY_OFFSET, _BODY_OFFSET + 33))       # 0-32: body
    + list(range(_LEFT_HAND_OFFSET, _LEFT_HAND_OFFSET + 21))  # 33-53: left hand
    + list(range(_RIGHT_HAND_OFFSET, _RIGHT_HAND_OFFSET + 21))  # 54-74: right hand
    + _SL_FACE_GLOBAL                                    # selected face points
)

# COMPACT: body(7) + hands(42) + lips + nose = 96
SL_LANDMARK_INDICES_COMPACT = (
    _COMPACT_BODY                                        # 7 body landmarks
    + list(range(_LEFT_HAND_OFFSET, _LEFT_HAND_OFFSET + 21))  # 33-53: left hand
    + list(range(_RIGHT_HAND_OFFSET, _RIGHT_HAND_OFFSET + 21))  # 54-74: right hand
    + _SL_FACE_GLOBAL_COMPACT                            # lips + nose
)

# EXTENDED: body(7) + hands(42) + lips + nose + eyes + eyebrows = 148
SL_LANDMARK_INDICES_EXTENDED = (
    _COMPACT_BODY                                        # 7 body landmarks
    + list(range(_LEFT_HAND_OFFSET, _LEFT_HAND_OFFSET + 21))  # 33-53: left hand
    + list(range(_RIGHT_HAND_OFFSET, _RIGHT_HAND_OFFSET + 21))  # 54-74: right hand
    + _SL_FACE_GLOBAL_EXTENDED                           # lips + nose + eyes + eyebrows
)

# Preset registry
SL_LANDMARK_PRESETS = {
    "compact": SL_LANDMARK_INDICES_COMPACT,
    "extended": SL_LANDMARK_INDICES_EXTENDED,
    "full": SL_LANDMARK_INDICES_FULL,
}

# Default preset — compact (96 landmarks, 288 input dim)
SL_LANDMARK_INDICES = SL_LANDMARK_INDICES_COMPACT

# Feature dimension: n_landmarks * 3 coords
SL_N_LANDMARKS = len(SL_LANDMARK_INDICES)
SL_INPUT_DIM = SL_N_LANDMARKS * 3


_DIM_TO_PRESET: dict[int, str] = {
    len(v) * 3: k for k, v in SL_LANDMARK_PRESETS.items()
}


def preset_from_input_dim(input_dim: int) -> str | None:
    """Return preset name matching the given input_dim, or None."""
    return _DIM_TO_PRESET.get(input_dim)


def _apply_landmark_filter(
    pose_slice: np.ndarray,
    conf_slice: np.ndarray,
    filter_landmarks: bool,
) -> tuple[np.ndarray, np.ndarray, int, list[int]]:
    """Apply SL landmark filtering to pose/confidence slices.

    Returns (filtered_pose, filtered_conf, n_landmarks, indices).
    """
    if filter_landmarks:
        idx = SL_LANDMARK_INDICES
        return pose_slice[:, idx, :], conf_slice[:, idx, :], SL_N_LANDMARKS, idx
    idx = list(range(543))
    return pose_slice, conf_slice, 543, idx


def select_sl_landmarks(
    pose: np.ndarray,
    indices: list[int] | None = None,
) -> np.ndarray:
    """Select only SL-relevant landmarks from a pose array.

    Args:
        pose: Shape (T, 543, 3) or (T, 1, 543, 3) — full landmark array.
        indices: Landmark indices to keep. Defaults to SL_LANDMARK_INDICES.

    Returns:
        Filtered array with shape (T, N, 3) where N = len(indices).
    """
    if indices is None:
        indices = SL_LANDMARK_INDICES

    if pose.ndim == 4:
        # (T, 1, 543, 3) → drop person dim first
        pose = pose[:, 0, :, :]

    return pose[:, indices, :]


# ---------------------------------------------------------------------------
# Label normalization and quality filtering
# ---------------------------------------------------------------------------

def normalize_label(label: str) -> str:
    """Normalize a training label: lowercase + strip trailing variant suffixes.

    Examples:
        "Voda_1" -> "voda"
        "HOUSE_3" -> "house"
        "čokoláda" -> "čokoláda"

    Distinct from glossary.normalize_word() which also strips punctuation.
    """
    if pd.isna(label):
        return ""
    s = str(label).strip().lower()
    # Strip trailing _N variant suffixes (e.g. "_1", "_23")
    s = re.sub(r'_\d+$', '', s)
    return s


def filter_quality_labels(
    manifest_df: pd.DataFrame,
    min_samples: int = 3,
    normalize: bool = True,
) -> pd.DataFrame:
    """Filter manifest to labels with at least min_samples occurrences.

    Args:
        manifest_df: DataFrame with training segments (from manifest.csv).
        min_samples: Minimum samples per label to keep (default 3).
        normalize: Apply normalize_label() (default True).

    Returns:
        Filtered DataFrame with 'label' and 'label_original' columns.
    """
    df = manifest_df.copy()

    # Derive label column defensively (lesson #5)
    if "label" not in df.columns:
        rt = df.get("reviewed_text", pd.Series(dtype=str))
        tx = df.get("text", pd.Series(dtype=str))
        df["label"] = rt.where(rt.str.strip() != "", tx)

    df["label_original"] = df["label"]

    if normalize:
        df["label"] = df["label"].apply(normalize_label)

    # Remove empty/NaN labels, then filter by min sample count (single mask)
    mask = df["label"].notna() & (df["label"].str.strip() != "")
    df = df[mask]

    label_counts = df["label"].value_counts()
    valid_labels = label_counts[label_counts >= min_samples].index
    return df[df["label"].isin(valid_labels)].copy()


# ---------------------------------------------------------------------------
# VTT deduplication / merging
# ---------------------------------------------------------------------------

def _text_similarity(a: str, b: str) -> float:
    """Character overlap ratio between two strings (0–1)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    longer = max(len(a), len(b))
    matches = sum(c1 == c2 for c1, c2 in zip(a, b))
    return matches / longer


def deduplicate_vtt_cues(
    cues: list[dict],
    max_gap_ms: int = 2000,
    similarity_threshold: float = 0.85,
) -> list[dict]:
    """Merge adjacent VTT cues with identical/similar text.

    OCR often produces near-duplicate cues: same subtitle text detected at
    slightly different frame windows.  This merges them into single cues.

    Args:
        cues: List of {start_ms, end_ms, text} from read_vtt().
        max_gap_ms: Maximum gap between cues to consider merging (default 2 s).
        similarity_threshold: Character overlap ratio to treat texts as "same".

    Returns:
        Deduplicated list — adjacent cues with similar text merged into one
        with start_ms from first, end_ms from last.
    """
    if not cues:
        return []

    sorted_cues = sorted(cues, key=lambda c: c["start_ms"])
    merged: list[dict] = [dict(sorted_cues[0])]  # copy first cue

    for cue in sorted_cues[1:]:
        prev = merged[-1]
        gap = cue["start_ms"] - prev["end_ms"]
        sim = _text_similarity(prev["text"], cue["text"])

        if gap <= max_gap_ms and sim >= similarity_threshold:
            # Merge: extend end, keep longer text
            prev["end_ms"] = max(prev["end_ms"], cue["end_ms"])
            if len(cue["text"]) > len(prev["text"]):
                prev["text"] = cue["text"]
        else:
            merged.append(dict(cue))

    return merged


def merge_short_segments(
    cues: list[dict],
    min_duration_ms: int = 1000,
    max_gap_ms: int = 500,
) -> list[dict]:
    """Merge segments shorter than *min_duration_ms* with their neighbours.

    Short segments get absorbed into the nearest neighbour (prefer next,
    then previous).  Only merges when the gap is ≤ *max_gap_ms*.
    """
    if len(cues) <= 1:
        return list(cues)

    result: list[dict] = [dict(c) for c in cues]  # working copies
    changed = True

    while changed:
        changed = False
        i = 0
        while i < len(result):
            dur = result[i]["end_ms"] - result[i]["start_ms"]
            if dur >= min_duration_ms or len(result) <= 1:
                i += 1
                continue

            # Try merge with next
            if i + 1 < len(result):
                gap = result[i + 1]["start_ms"] - result[i]["end_ms"]
                if gap <= max_gap_ms:
                    result[i]["end_ms"] = result[i + 1]["end_ms"]
                    result[i]["text"] = (
                        result[i]["text"] + "\n" + result[i + 1]["text"]
                    )
                    result.pop(i + 1)
                    changed = True
                    continue

            # Try merge with previous
            if i > 0:
                gap = result[i]["start_ms"] - result[i - 1]["end_ms"]
                if gap <= max_gap_ms:
                    result[i - 1]["end_ms"] = result[i]["end_ms"]
                    result[i - 1]["text"] = (
                        result[i - 1]["text"] + "\n" + result[i]["text"]
                    )
                    result.pop(i)
                    changed = True
                    continue

            i += 1

    return result


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def align_pose_to_subtitles(
    pose_path:  Path,
    vtt_path:   Path,
    video_path: Path,
    dedup: bool = True,
    merge_short: bool = True,
    max_gap_ms: int = 2000,
    min_segment_ms: int = 1000,
) -> tuple[list[dict], int]:
    """Align a .pose file to subtitle cues from a .vtt file.

    Returns (segments, raw_cue_count) where segments is one dict per
    (optionally deduplicated/merged) subtitle cue and raw_cue_count is
    the number of cues before dedup/merging.
    """
    from spj.preannotate import load_pose_arrays
    from spj.ocr_subtitles import read_vtt

    data, _conf, fps = load_pose_arrays(pose_path)
    T = data.shape[0]

    subs = read_vtt(vtt_path)
    raw_cue_count = len(subs)

    if dedup:
        subs = deduplicate_vtt_cues(subs, max_gap_ms=max_gap_ms)
    if merge_short:
        subs = merge_short_segments(subs, min_duration_ms=min_segment_ms)

    segments: list[dict] = []

    for sub in subs:
        start_ms = int(sub["start_ms"])
        end_ms   = int(sub["end_ms"])
        text     = sub["text"]

        frame_start = min(int(start_ms * fps / 1000), T - 1)
        frame_end   = min(int(end_ms   * fps / 1000), T - 1)
        n_frames    = max(0, frame_end - frame_start)

        segments.append({
            "video_path":        str(video_path),
            "pose_path":         str(pose_path),
            "vtt_path":          str(vtt_path),
            "start_ms":          start_ms,
            "end_ms":            end_ms,
            "frame_start":       frame_start,
            "frame_end":         frame_end,
            "text":              text,
            "fps":               float(fps),
            "n_frames":          n_frames,
            "status":            "pending",
            "reviewed_text":     "",
            "reviewed_start_ms": start_ms,
            "reviewed_end_ms":   end_ms,
        })

    return segments, raw_cue_count


def build_alignment_table(
    inv_df:        pd.DataFrame,
    pose_dir:      Path,
    subtitles_dir: Path,
    existing_df:   Optional[pd.DataFrame] = None,
    dedup: bool = True,
    merge_short: bool = True,
    max_gap_ms: int = 2000,
    min_segment_ms: int = 1000,
) -> tuple[pd.DataFrame, int, int]:
    """Build (or extend) the alignment table from inventory rows.

    For each video that has both a .pose and a .vtt file, calls
    align_pose_to_subtitles and assigns a unique segment_id.

    existing_df: already-aligned rows are skipped, preserving review state.

    Returns (DataFrame, total_raw_cues, total_merged_segments).
    """
    pose_dir      = Path(pose_dir)
    subtitles_dir = Path(subtitles_dir)

    existing_rows: list[pd.DataFrame] = []

    # Track which stems have already been processed (prevents re-processing
    # same video when alignment is run multiple times)
    processed_stems: set[str] = set()
    if existing_df is not None and not existing_df.empty:
        existing_rows.append(existing_df)
        for sid in existing_df["segment_id"]:
            parts = str(sid).rsplit("_", 1)
            if parts:
                processed_stems.add(parts[0])

    new_rows: list[dict] = []
    total_raw_cues = 0
    total_merged_segments = 0

    for _, row in inv_df.iterrows():
        video_path = Path(str(row["path"]))
        stem       = video_path.stem

        if stem in processed_stems:
            continue  # Already aligned — skip entirely

        pose_path = pose_dir      / f"{stem}.pose"

        if not (pose_path.exists() and pose_path.stat().st_size > 0):
            continue

        # Find best available VTT: OCR in subtitles_dir, or soft subs next to video
        from spj.ocr_subtitles import get_subtitle_status
        status = get_subtitle_status(video_path, subtitles_dir)
        vtt_path = status["vtt_path"]
        if vtt_path is None:
            continue

        try:
            segs, raw_count = align_pose_to_subtitles(
                pose_path, vtt_path, video_path,
                dedup=dedup, merge_short=merge_short,
                max_gap_ms=max_gap_ms, min_segment_ms=min_segment_ms,
            )
        except Exception as exc:
            logger.warning("Alignment failed for %s: %s", stem, exc)
            continue

        total_raw_cues += raw_count
        total_merged_segments += len(segs)

        for seg in segs:
            sid = f"{stem}_{seg['start_ms']:08d}"
            seg["segment_id"] = sid
            new_rows.append(seg)

        processed_stems.add(stem)

    new_df    = pd.DataFrame(new_rows) if new_rows else pd.DataFrame()
    all_parts = existing_rows + ([new_df] if not new_df.empty else [])

    result = pd.concat(all_parts, ignore_index=True) if all_parts else pd.DataFrame()
    return result, total_raw_cues, total_merged_segments


# ---------------------------------------------------------------------------
# Single-sign video import (partner-dictnary / glossed clips)
# ---------------------------------------------------------------------------

def import_single_sign_videos(
    pose_dir: Path,
    video_dir: Path,
    existing_df: Optional[pd.DataFrame] = None,
    label_source: str = "filename",
    status: str = "approved",
    split_words: bool = True,
) -> tuple[pd.DataFrame, int]:
    """Import single-sign videos as alignment rows (no subtitles needed).

    Each video = one segment covering the entire duration.
    Label is parsed from the filename pattern ``{word_id}_{translation}.ext``.

    When ``split_words=True`` and the label has multiple words (e.g.
    "baranie mäso"), the video is split into N+1 segments:
      1. Full video → "baranie mäso"  (the phrase)
      2. First portion → "baranie"     (individual word)
      3. Second portion → "mäso"       (individual word)

    Split points are estimated by dividing the video proportionally by word
    count, with a small overlap margin.

    Args:
        pose_dir: Directory containing .pose files.
        video_dir: Directory containing the source video files.
        existing_df: Already-aligned rows — videos already present are skipped.
        label_source: How to derive the label. Currently only "filename".
        status: Status to assign (default "approved" since these are clean clips).
        split_words: If True, also create per-word segments for multi-word labels.

    Returns:
        (merged_df, n_imported) — full alignment table and count of new videos.
    """
    from spj.preannotate import load_pose_arrays

    pose_dir = Path(pose_dir)
    video_dir = Path(video_dir)

    # Collect existing segment_ids to skip
    processed_stems: set[str] = set()
    existing_rows: list[pd.DataFrame] = []
    if existing_df is not None and not existing_df.empty:
        existing_rows.append(existing_df)
        for sid in existing_df["segment_id"]:
            parts = str(sid).rsplit("_", 1)
            if parts:
                processed_stems.add(parts[0])

    new_rows: list[dict] = []
    n_videos = 0

    # Find all video files in video_dir
    video_files = sorted(
        f for ext in ("*.mp4", "*.mkv", "*.webm", "*.mov")
        for f in video_dir.glob(ext)
    )

    for video_path in video_files:
        stem = video_path.stem
        if stem in processed_stems:
            continue

        pose_path = pose_dir / f"{stem}.pose"
        if not (pose_path.exists() and pose_path.stat().st_size > 0):
            continue

        # Parse label from filename: "{word_id}_{translation}" → translation
        parts = stem.split("_", 1)
        if len(parts) == 2 and parts[0].isdigit():
            label = parts[1].strip()
        else:
            label = stem  # fallback: use full stem

        if not label:
            continue

        try:
            data, _conf, fps = load_pose_arrays(pose_path)
        except Exception as exc:
            logger.warning("Failed to load pose for %s: %s", stem, exc)
            continue

        T = data.shape[0]
        if T == 0:
            continue

        duration_ms = int(T / fps * 1000)
        n_videos += 1

        # 1) Full-video segment (always created)
        sid = f"{stem}_00000000"
        new_rows.append({
            "segment_id":        sid,
            "video_path":        str(video_path),
            "pose_path":         str(pose_path),
            "vtt_path":          "",
            "start_ms":          0,
            "end_ms":            duration_ms,
            "frame_start":       0,
            "frame_end":         T - 1,
            "text":              label,
            "fps":               float(fps),
            "n_frames":          T,
            "status":            status,
            "reviewed_text":     label,
            "reviewed_start_ms": 0,
            "reviewed_end_ms":   duration_ms,
        })

        # 2) Per-word segments for multi-word labels
        words = label.split()
        if split_words and len(words) >= 2:
            # Trim ~10% from start/end (intro/outro idle frames)
            trim_ms = int(duration_ms * 0.10)
            active_start = trim_ms
            active_end = duration_ms - trim_ms
            active_dur = max(active_end - active_start, 1)

            for i, word in enumerate(words):
                w_start = active_start + int(active_dur * i / len(words))
                w_end = active_start + int(active_dur * (i + 1) / len(words))
                w_frame_start = min(int(w_start * fps / 1000), T - 1)
                w_frame_end = min(int(w_end * fps / 1000), T - 1)
                w_n_frames = max(0, w_frame_end - w_frame_start)

                w_sid = f"{stem}_{w_start:08d}"
                new_rows.append({
                    "segment_id":        w_sid,
                    "video_path":        str(video_path),
                    "pose_path":         str(pose_path),
                    "vtt_path":          "",
                    "start_ms":          w_start,
                    "end_ms":            w_end,
                    "frame_start":       w_frame_start,
                    "frame_end":         w_frame_end,
                    "text":              word,
                    "fps":               float(fps),
                    "n_frames":          w_n_frames,
                    "status":            "pending",
                    "reviewed_text":     word,
                    "reviewed_start_ms": w_start,
                    "reviewed_end_ms":   w_end,
                })

        processed_stems.add(stem)

    new_df = pd.DataFrame(new_rows) if new_rows else pd.DataFrame()
    all_parts = existing_rows + ([new_df] if not new_df.empty else [])
    result = pd.concat(all_parts, ignore_index=True) if all_parts else pd.DataFrame()

    return result, n_videos


# ---------------------------------------------------------------------------
# CSV persistence
# ---------------------------------------------------------------------------

_CSV_INT_COLS = [
    "start_ms", "end_ms", "frame_start", "frame_end",
    "n_frames", "reviewed_start_ms", "reviewed_end_ms",
]


def load_alignment_csv(path: Path) -> pd.DataFrame:
    """Load alignment CSV with correct dtypes."""
    df = pd.read_csv(path, dtype=str)

    for col in _CSV_INT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")

    if "fps" in df.columns:
        df["fps"] = pd.to_numeric(df["fps"], errors="coerce").astype("float64")

    if "reviewed_text" not in df.columns:
        df["reviewed_text"] = ""
    else:
        df["reviewed_text"] = df["reviewed_text"].fillna("")

    if "status" not in df.columns:
        df["status"] = "pending"
    else:
        df["status"] = df["status"].fillna("pending")

    if "segment_id" not in df.columns:
        df["segment_id"] = ""
    else:
        df["segment_id"] = df["segment_id"].fillna("")

    # Back-fill reviewed timestamps from originals if columns were missing
    if "reviewed_start_ms" not in df.columns:
        df["reviewed_start_ms"] = df["start_ms"]
    if "reviewed_end_ms" not in df.columns:
        df["reviewed_end_ms"] = df["end_ms"]

    return df


def save_alignment_csv(df: pd.DataFrame, path: Path) -> None:
    """Save alignment DataFrame to CSV, creating parent directories."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# NPZ export
# ---------------------------------------------------------------------------

def export_segment_npz(
    row: "pd.Series",
    output_dir: Path,
    filter_landmarks: bool = True,
) -> Optional[str]:
    """Export one approved segment as a float16 .npz training file.

    Arrays saved (when filter_landmarks=True, default):
        pose(T, N_SL, 3)       float16 — SL-relevant landmarks only
        confidence(T, N_SL, 1) float16 — per-landmark confidence
    where N_SL = SL_N_LANDMARKS (~147 instead of 543).

    When filter_landmarks=False (legacy):
        pose(T, 543, 3), confidence(T, 543, 1) — all landmarks.

    Also saves: text, fps, start_ms, end_ms, source_video, n_landmarks,
                landmark_indices (so downstream code knows the mapping).

    Uses reviewed timestamps / text if set, else falls back to originals.
    Returns the output file path string, or None for 0-frame segments.
    """
    from spj.preannotate import load_pose_arrays

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data, conf, fps = load_pose_arrays(Path(str(row["pose_path"])))
    T = data.shape[0]

    rev_start  = int(row["reviewed_start_ms"])
    rev_end    = int(row["reviewed_end_ms"])
    orig_start = int(row["start_ms"])

    frame_start = min(int(rev_start * fps / 1000), T - 1)
    frame_end   = min(int(rev_end   * fps / 1000), T - 1)

    if frame_end <= frame_start:
        logger.warning(
            "Skipping 0-frame segment: %s (start=%d, end=%d ms)",
            Path(str(row["video_path"])).name, rev_start, rev_end,
        )
        return None

    pose_slice = data[frame_start:frame_end, 0, :, :]   # (T, 543, 3)
    conf_slice = conf[frame_start:frame_end, 0, :, :]   # (T, 543, 1)

    pose_slice, conf_slice, n_landmarks, idx = _apply_landmark_filter(
        pose_slice, conf_slice, filter_landmarks,
    )

    text = str(row["reviewed_text"]).strip() or str(row["text"])
    stem = Path(str(row["video_path"])).stem

    out_path = output_dir / f"{stem}_{orig_start:08d}.npz"

    np.savez_compressed(
        str(out_path),
        pose         = pose_slice.astype(np.float16),
        confidence   = conf_slice.astype(np.float16),
        text             = text,
        fps              = float(fps),
        start_ms         = rev_start,
        end_ms           = rev_end,
        source_video     = str(row["video_path"]),
        n_landmarks      = n_landmarks,
        landmark_indices = np.array(idx, dtype=np.int16),
    )

    return str(out_path)


def write_training_config(
    output_dir: Path,
    n_segments: int,
    filter_landmarks: bool = True,
) -> Path:
    """Write a JSON training config optimised for M4 Max alongside the .npz files."""
    n_lm = SL_N_LANDMARKS if filter_landmarks else 543
    config = {
        "device": "mps",
        "batch_size": 256,
        "pin_memory": False,
        "precision": "fp16",
        "n_segments": n_segments,
        "n_landmarks": n_lm,
        "input_dim": n_lm * 3,
        "filter_landmarks": filter_landmarks,
        "pose_shape": [None, n_lm, 3],
        "confidence_shape": [None, n_lm, 1],
        "dataloader": {
            "num_workers": 0,
            "pin_memory": False,
            "prefetch_factor": None,
        },
        "notes": (
            f"M4 Max optimised: fp16 arrays, MPS backend, batch_size=256. "
            f"pin_memory=False (unified memory — no separate GPU VRAM to pin). "
            f"Landmarks: {n_lm} ({'SL-filtered' if filter_landmarks else 'all 543'}). "
            f"Input dim: {n_lm * 3}."
        ),
    }
    cfg_path = Path(output_dir) / "training_config.json"
    cfg_path.write_text(json.dumps(config, indent=2))
    return cfg_path


# ---------------------------------------------------------------------------
# Sign-level pairings: detection, export, suggestion, CSV persistence
# ---------------------------------------------------------------------------

_PAIRINGS_INT_COLS = [
    "sign_start_ms", "sign_end_ms", "sign_frame_start", "sign_frame_end",
]
_PAIRINGS_FLOAT_COLS = [
    "motion_confidence", "suggestion_confidence", "fps",
]
_PAIRINGS_STR_COLS = [
    "pairing_id", "segment_id", "video_path", "pose_path", "hand",
    "word", "gloss_id", "status", "suggestion_gloss", "note", "mouthing",
]

# Pairing status constants
PST_PENDING        = "pending"
PST_PAIRED         = "paired"
PST_SKIPPED        = "skipped"
PST_AUTO_SUGGESTED = "auto_suggested"

_VALID_HANDS = ("right", "left")


def make_pairing_dict(
    pairing_id: str,
    segment_id: str,
    video_path: str,
    pose_path: str,
    hand: str,
    sign_start_ms: int,
    sign_end_ms: int,
    sign_frame_start: int,
    sign_frame_end: int,
    fps: float,
    motion_confidence: float = 0.0,
    note: str = "",
    word: str = "",
    gloss_id: str = "",
    status: str = PST_PENDING,
    suggestion_gloss: str = "",
    suggestion_confidence: float = 0.0,
    mouthing: str = "",
) -> dict:
    """Build a single pairing dict matching the pairings CSV schema."""
    return {
        "pairing_id": pairing_id,
        "segment_id": segment_id,
        "video_path": video_path,
        "pose_path": pose_path,
        "hand": hand,
        "sign_start_ms": sign_start_ms,
        "sign_end_ms": sign_end_ms,
        "sign_frame_start": sign_frame_start,
        "sign_frame_end": sign_frame_end,
        "motion_confidence": motion_confidence,
        "word": word,
        "gloss_id": gloss_id,
        "status": status,
        "suggestion_gloss": suggestion_gloss,
        "suggestion_confidence": suggestion_confidence,
        "fps": float(fps),
        "note": note,
        "mouthing": mouthing,
    }


def detect_signs_in_segment(
    pose_data: np.ndarray,
    conf_data: np.ndarray,
    fps: float,
    seg_start_ms: int,
    seg_end_ms: int,
    segment_id: str,
    video_path: str,
    pose_path: str,
) -> list[dict]:
    """Detect individual sign boundaries within a subtitle segment.

    Slices pose data to the segment's frame range, runs kinematic sign
    detection, and returns a list of pairing dicts (one per detected sign)
    with absolute video timestamps.

    Args:
        pose_data: Full pose array (T, 1, 543, 3).
        conf_data: Full confidence array (T, 1, 543, 1).
        fps: Video frame rate.
        seg_start_ms: Segment start in ms (absolute video time).
        seg_end_ms: Segment end in ms (absolute video time).
        segment_id: Parent segment ID from alignment.csv.
        video_path: Source video path.
        pose_path: Source .pose path.

    Returns:
        List of dicts matching pairings CSV schema, all status="pending".
    """
    from spj.preannotate import detect_sign_segments

    T = pose_data.shape[0]
    seg_frame_start = min(int(seg_start_ms * fps / 1000), T - 1)
    seg_frame_end = min(int(seg_end_ms * fps / 1000), T - 1)

    if seg_frame_end <= seg_frame_start:
        return []

    # Slice keeping person dimension for detect_sign_segments (expects 4D)
    slice_data = pose_data[seg_frame_start:seg_frame_end]
    slice_conf = conf_data[seg_frame_start:seg_frame_end]

    segments = detect_sign_segments(slice_data, slice_conf, fps)

    stem = Path(video_path).stem
    pairings: list[dict] = []

    for hand in _VALID_HANDS:
        for start_ms_rel, end_ms_rel, motion_conf in segments.get(hand, []):
            # Convert relative (within-slice) timestamps to absolute video time
            abs_start_ms = seg_start_ms + start_ms_rel
            abs_end_ms = seg_start_ms + end_ms_rel

            # Frame indices (absolute)
            frame_start = min(int(abs_start_ms * fps / 1000), T - 1)
            frame_end = min(int(abs_end_ms * fps / 1000), T - 1)

            pairing_id = f"{stem}_{abs_start_ms:08d}_{hand[0]}"

            pairings.append(make_pairing_dict(
                pairing_id=pairing_id,
                segment_id=segment_id,
                video_path=video_path,
                pose_path=pose_path,
                hand=hand,
                sign_start_ms=abs_start_ms,
                sign_end_ms=abs_end_ms,
                sign_frame_start=frame_start,
                sign_frame_end=frame_end,
                fps=fps,
                motion_confidence=motion_conf,
            ))

    # Sort by start time
    pairings.sort(key=lambda p: (p["sign_start_ms"], p["hand"]))
    return pairings


_manual_counter = 0


def create_manual_pairing(
    segment_id: str,
    video_path: str,
    pose_path: str,
    hand: str,
    sign_start_ms: int,
    sign_end_ms: int,
    fps: float,
) -> dict:
    """Create a single manual pairing dict (same schema as detect output).

    Args:
        segment_id: Parent segment ID from alignment.csv.
        video_path: Source video path.
        pose_path: Source .pose path.
        hand: "right" or "left".
        sign_start_ms: Sign start in ms (absolute video time).
        sign_end_ms: Sign end in ms (absolute video time).
        fps: Video frame rate.

    Returns:
        Dict matching pairings CSV schema with status="pending".
        pairing_id uses ``_M<n>`` suffix to distinguish manual entries.

    Raises:
        ValueError: If *hand* is not "right" or "left".
    """
    if hand not in _VALID_HANDS:
        raise ValueError(f"hand must be one of {_VALID_HANDS}, got {hand!r}")

    global _manual_counter
    _manual_counter += 1

    stem = Path(video_path).stem
    frame_start = int(sign_start_ms * fps / 1000)
    frame_end = int(sign_end_ms * fps / 1000)
    pairing_id = f"{stem}_{sign_start_ms:08d}_{hand[0]}_M{_manual_counter}"

    return make_pairing_dict(
        pairing_id=pairing_id,
        segment_id=segment_id,
        video_path=video_path,
        pose_path=pose_path,
        hand=hand,
        sign_start_ms=sign_start_ms,
        sign_end_ms=sign_end_ms,
        sign_frame_start=frame_start,
        sign_frame_end=frame_end,
        fps=fps,
        note="manual",
    )


def export_sign_npz(
    pairing_row: "pd.Series",
    pose_data: np.ndarray,
    conf_data: np.ndarray,
    output_dir: Path,
    filter_landmarks: bool = True,
) -> Optional[str]:
    """Export one paired sign as a float16 .npz training file.

    Like export_segment_npz() but uses sign-level timestamps and saves
    gloss_id as label.

    When filter_landmarks=True (default), applies the same SL landmark
    filtering as export_segment_npz() — keeping only SL_LANDMARK_INDICES.

    Returns the output file path string, or None for 0-frame signs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fps = float(pairing_row["fps"])
    T = pose_data.shape[0]

    frame_start = int(pairing_row["sign_frame_start"])
    frame_end = int(pairing_row["sign_frame_end"])

    if frame_end <= frame_start or frame_start >= T:
        logger.warning(
            "Skipping 0-frame sign: %s", pairing_row.get("pairing_id", "?"),
        )
        return None

    frame_end = min(frame_end, T)

    # Remove person dimension: (T, 1, 543, 3) → (T, 543, 3)
    pose_slice = pose_data[frame_start:frame_end, 0, :, :]
    conf_slice = conf_data[frame_start:frame_end, 0, :, :]

    pose_slice, conf_slice, n_landmarks, idx = _apply_landmark_filter(
        pose_slice, conf_slice, filter_landmarks,
    )

    gloss_id = str(pairing_row["gloss_id"])
    word = str(pairing_row["word"])
    hand = str(pairing_row["hand"])

    out_path = output_dir / f"{pairing_row['pairing_id']}.npz"

    np.savez_compressed(
        str(out_path),
        pose=pose_slice.astype(np.float16),
        confidence=conf_slice.astype(np.float16),
        label=gloss_id,
        text=word,
        hand=hand,
        fps=fps,
        start_ms=int(pairing_row["sign_start_ms"]),
        end_ms=int(pairing_row["sign_end_ms"]),
        source_video=str(pairing_row["video_path"]),
        n_landmarks=n_landmarks,
        landmark_indices=np.array(idx, dtype=np.int16),
    )

    return str(out_path)


def suggest_sign_pairings(
    pairings_df: pd.DataFrame,
    pose_data: np.ndarray,
    fps: float,
    model: "PoseTransformerEncoder",
    label_encoder: "LabelEncoder",
    config: dict,
    glossary: "Glossary",
    subtitle_text: str,
) -> pd.DataFrame:
    """Auto-suggest sign-word pairings using a pre-loaded model.

    For each pending sign, runs model inference and cross-references the
    predicted gloss with words in the subtitle text.

    Args:
        pairings_df: Pairings dataframe (modified in place and returned).
        pose_data: Full pose array (T, 1, 543, 3).
        fps: Video frame rate.
        model: Pre-loaded PoseTransformerEncoder in eval mode.
        label_encoder: Pre-loaded LabelEncoder.
        config: Checkpoint config dict.
        glossary: Glossary instance for word↔gloss matching.
        subtitle_text: The subtitle text for this segment.

    Returns:
        Updated pairings_df with suggestions filled in.
    """
    try:
        from spj.inference import predict_segments
    except ImportError as exc:
        logger.warning("Cannot import inference: %s", exc)
        return pairings_df

    # Build segments dict in the format predict_segments expects
    pending_mask = pairings_df["status"].isin([PST_PENDING])
    if not pending_mask.any():
        return pairings_df

    segments: dict[str, list[tuple[int, int, float]]] = {"right": [], "left": []}
    for _, row in pairings_df[pending_mask].iterrows():
        hand = str(row["hand"])
        segments.setdefault(hand, []).append((
            int(row["sign_start_ms"]),
            int(row["sign_end_ms"]),
            float(row["motion_confidence"]),
        ))

    prepartner-dictns = predict_segments(
        model, label_encoder, segments, pose_data, fps,
        max_seq_len=config.get("max_seq_len", 300),
    )

    # Match subtitle words to glosses for cross-referencing
    word_matches = glossary.match_sentence(subtitle_text)

    # Build reverse map: gloss_id → word
    gloss_to_word: dict[str, str] = {}
    for wm in word_matches:
        for gid in wm.get("glosses", []):
            if gid not in gloss_to_word:
                gloss_to_word[gid] = wm["raw"]

    # Apply suggestions
    for pred in prepartner-dictns:
        pred_gloss = pred["predicted_gloss"]
        pred_conf = pred["prepartner-dictn_confidence"]
        hand = pred["hand"]
        start_ms = pred["start_ms"]

        mask = (
            (pairings_df["hand"] == hand)
            & (pairings_df["sign_start_ms"] == start_ms)
            & (pairings_df["status"] == PST_PENDING)
        )
        if not mask.any():
            continue

        idx = pairings_df.index[mask][0]
        pairings_df.loc[idx, "suggestion_gloss"] = pred_gloss
        pairings_df.loc[idx, "suggestion_confidence"] = pred_conf

        # If the predicted gloss maps to a word in the subtitle, auto-fill
        if pred_gloss in gloss_to_word:
            pairings_df.loc[idx, "word"] = gloss_to_word[pred_gloss]
            pairings_df.loc[idx, "gloss_id"] = pred_gloss
            pairings_df.loc[idx, "status"] = PST_AUTO_SUGGESTED

    return pairings_df


def load_pairings_csv(path: Path) -> pd.DataFrame:
    """Load pairings CSV with correct dtypes."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path, dtype=str)

    for col in _PAIRINGS_INT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")

    for col in _PAIRINGS_FLOAT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype("float64")

    # Fill NaN strings with empty
    for col in _PAIRINGS_STR_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("")

    if "status" in df.columns:
        df["status"] = df["status"].replace("", PST_PENDING)

    return df


def save_pairings_csv(df: pd.DataFrame, path: Path) -> None:
    """Save pairings DataFrame to CSV, creating parent directories."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Video segment extraction (ffmpeg → bytes)
# ---------------------------------------------------------------------------

_MAX_SEGMENT_SEC = 60


def extract_video_segment(
    video_path: Path,
    start_ms: int,
    end_ms: int,
    max_duration_sec: int = _MAX_SEGMENT_SEC,
) -> Optional[bytes]:
    """Extract a video segment as MP4 bytes using ffmpeg.

    Pipes the output to stdout so no temp files are created.
    Returns None on error (caller should fall back to st.video).
    """
    duration_ms = end_ms - start_ms
    if duration_ms <= 0:
        return None
    if duration_ms > max_duration_sec * 1000:
        logger.warning(
            "Segment too long (%d ms > %d s limit), truncating",
            duration_ms, max_duration_sec,
        )
        end_ms = start_ms + max_duration_sec * 1000
        duration_ms = end_ms - start_ms

    start_sec = start_ms / 1000.0
    duration_sec = duration_ms / 1000.0

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_sec:.3f}",
        "-i", str(video_path),
        "-t", f"{duration_sec:.3f}",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "28",
        "-an",
        "-movflags", "+faststart+frag_keyframe+empty_moov",
        "-f", "mp4",
        "pipe:1",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, timeout=30,
        )
        if result.returncode != 0:
            logger.warning("ffmpeg failed: %s", result.stderr[:200])
            return None
        if len(result.stdout) < 100:
            logger.warning("ffmpeg produced too-small output (%d bytes)", len(result.stdout))
            return None
        return result.stdout
    except FileNotFoundError:
        logger.warning("ffmpeg not found on PATH")
        return None
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg timed out after 30s")
        return None
    except Exception as exc:
        logger.warning("ffmpeg error: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Pose visualisation
# ---------------------------------------------------------------------------

def pose_frame_figure(
    frame_xyz:  np.ndarray,   # (543, 3)
    frame_conf: np.ndarray,   # (543,) or (543, 1)
) -> "go.Figure":
    """2D plotly scatter of body + hand + face landmarks for a single frame.

    Colours: body=royalblue, left_hand=limegreen, right_hand=crimson,
    lips=hotpink, eyes=cyan, eyebrows=orange, face_oval/dots=gold(semi-transparent).
    Y-axis flipped so head is at top. Black background.
    Landmarks with confidence < 0.1 are hidden.
    Auto-adapts aspect ratio to actual landmark bounding box.
    """
    import plotly.graph_objects as go

    conf    = frame_conf.ravel()   # (543,)
    visible = conf >= 0.1

    traces: list = []

    def _add_edges(connections: list, offset: int, color: str) -> None:
        xs: list = []
        ys: list = []
        for a, b in connections:
            ia, ib = a + offset, b + offset
            if ia < len(visible) and ib < len(visible) and visible[ia] and visible[ib]:
                xs += [frame_xyz[ia, 0], frame_xyz[ib, 0], None]
                ys += [1 - frame_xyz[ia, 1], 1 - frame_xyz[ib, 1], None]
        if xs:
            traces.append(go.Scatter(
                x=xs, y=ys,
                mode="lines",
                line=dict(color=color, width=2),
                showlegend=False,
                hoverinfo="skip",
            ))

    def _add_dots(start: int, end: int, color: str, name: str, size: int = 4) -> None:
        end = min(end, len(visible))
        xs = [frame_xyz[i, 0]       for i in range(start, end) if visible[i]]
        ys = [1 - frame_xyz[i, 1]   for i in range(start, end) if visible[i]]
        if xs:
            traces.append(go.Scatter(
                x=xs, y=ys,
                mode="markers",
                marker=dict(color=color, size=size),
                name=name,
                showlegend=False,
                hoverinfo="skip",
            ))

    # Body + hands
    _add_edges(BODY_CONNECTIONS, _BODY_OFFSET,       "royalblue")
    _add_edges(HAND_CONNECTIONS, _LEFT_HAND_OFFSET,  "limegreen")
    _add_edges(HAND_CONNECTIONS, _RIGHT_HAND_OFFSET, "crimson")

    _add_dots(_BODY_OFFSET,       _BODY_OFFSET + 33,       "royalblue", "body", size=3)
    _add_dots(_LEFT_HAND_OFFSET,  _LEFT_HAND_OFFSET + 21,  "limegreen", "left hand", size=3)
    _add_dots(_RIGHT_HAND_OFFSET, _RIGHT_HAND_OFFSET + 21, "crimson",   "right hand", size=3)

    # Face features — connections only, no individual dots
    _add_edges(_FACE_OVAL,           _FACE_OFFSET, "rgba(255,215,0,0.2)")
    _add_edges(_FACE_LIPS,           _FACE_OFFSET, "hotpink")
    _add_edges(_FACE_LEFT_EYE,       _FACE_OFFSET, "cyan")
    _add_edges(_FACE_RIGHT_EYE,      _FACE_OFFSET, "cyan")
    _add_edges(_FACE_LEFT_EYEBROW,   _FACE_OFFSET, "orange")
    _add_edges(_FACE_RIGHT_EYEBROW,  _FACE_OFFSET, "orange")

    # Auto-adapt axis ranges to visible landmark bounding box
    vis_x = [frame_xyz[i, 0]     for i in range(min(543, len(visible))) if visible[i]]
    vis_y = [1 - frame_xyz[i, 1] for i in range(min(543, len(visible))) if visible[i]]

    if vis_x and vis_y:
        x_min, x_max = min(vis_x), max(vis_x)
        y_min, y_max = min(vis_y), max(vis_y)
        pad_x = max(0.05, (x_max - x_min) * 0.05)
        pad_y = max(0.05, (y_max - y_min) * 0.05)
        x_range = [x_min - pad_x, x_max + pad_x]
        y_range = [y_min - pad_y, y_max + pad_y]
    else:
        x_range = [0, 1]
        y_range = [0, 1]

    fig = go.Figure(data=traces)
    fig.update_layout(
        height=450,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="black",
        plot_bgcolor="black",
        xaxis=dict(visible=False, range=x_range, scaleanchor="y"),
        yaxis=dict(visible=False, range=y_range),
    )
    return fig


# ---------------------------------------------------------------------------
# Real-time pose animation (HTML5 Canvas + JavaScript)
# ---------------------------------------------------------------------------

from string import Template as _Template

_POSE_ANIM_TEMPLATE = _Template(
    '<!DOCTYPE html>\n<html><head><style>'
    '*{margin:0;padding:0;box-sizing:border-box}'
    'body{background:transparent;font-family:-apple-system,BlinkMacSystemFont,sans-serif;color:#fafafa}'
    '#w{display:flex;flex-direction:column;align-items:center;'
    'width:fit-content;margin:0 auto;max-height:100vh}'
    'canvas{background:#000;border-radius:6px;display:block;cursor:pointer}'
    '#b{display:flex;align-items:center;gap:8px;padding:8px 2px;font-size:13px}'
    '#b button{background:#262730;border:1px solid #4a4a5a;color:#fafafa;'
    'border-radius:4px;padding:3px 12px;cursor:pointer;font-size:15px;line-height:1}'
    '#b button:hover{background:#3a3a4a}'
    '#b button.on{background:#ff6b6b33;border-color:#ff6b6b}'
    '#b input[type=range]{flex:1;accent-color:#ff6b6b;cursor:pointer}'
    '#b select{background:#262730;border:1px solid #4a4a5a;color:#fafafa;'
    'border-radius:4px;padding:2px 4px;font-size:12px;cursor:pointer}'
    '#tm{font-variant-numeric:tabular-nums;min-width:130px;text-align:center;'
    'font-size:12px;opacity:.7}'
    '#fp{font-size:11px;opacity:.5}'
    '</style></head><body>'
    '<div id="w"><canvas id="c"></canvas>'
    '<div id="b">'
    '<button id="pp" title="Play / Pause (Space)">&#x23F8;</button>'
    '<input type="range" id="sc" min="0" max="$max_frame" value="0" step="1">'
    '<span id="tm">00:00.000</span>'
    '<select id="sp">'
    '<option value="0.25">0.25x</option>'
    '<option value="0.5">0.5x</option>'
    '<option value="1" selected>1x</option>'
    '<option value="1.5">1.5x</option>'
    '<option value="2">2x</option>'
    '</select>'
    '<button id="lp" class="on" title="Loop">&#x21BB;</button>'
    '<span id="fp">$fps_display fps</span>'
    '</div></div>'
    '<script>'
    '(function(){'
    '"use strict";'
    'var NF=$n_frames,NL=543,FPS=$fps,SMS=$start_ms;'
    'var VX0=$vx_min,VX1=$vx_max,VY0=$vy_min,VY1=$vy_max;'
    'var VA=$video_aspect;'

    # decode base64 → Float32Array
    'function d64(s){var b=atob(s),a=new Uint8Array(b.length);'
    'for(var i=0;i<b.length;i++)a[i]=b.charCodeAt(i);'
    'return new Float32Array(a.buffer)}'
    'var P=d64("$pos_b64");var CF=d64("$conf_b64");'

    # connection arrays
    'var BC=$body_conn,HC=$hand_conn;'
    'var FL=$face_lips,FLE=$face_left_eye,FRE=$face_right_eye;'
    'var FLB=$face_left_brow,FRB=$face_right_brow,FO=$face_oval;'

    # canvas setup
    'var c=document.getElementById("c"),x=c.getContext("2d");'
    'var pp=document.getElementById("pp"),scr=document.getElementById("sc");'
    'var te=document.getElementById("tm"),se=document.getElementById("sp");'
    'var lb=document.getElementById("lp");'

    # Responsive canvas: use data's natural aspect, constrained by viewport
    # HiDPI: scale backing store by devicePixelRatio for sharp rendering
    'function rs(){'
    'var pw=c.parentElement.clientWidth;'
    'var maxH=Math.max(300,(window.innerHeight||800)-60);'
    'var da=VA>0?VA:(VY1-VY0)/(VX1-VX0);'  # video aspect or data aspect
    'var w=pw,h=Math.round(pw*da);'
    'if(h>maxH){h=maxH;w=Math.round(h/da)}'
    'var dpr=window.devicePixelRatio||1;'
    'c.width=Math.round(w*dpr);c.height=Math.round(h*dpr);'
    'c.style.width=w+"px";c.style.height=h+"px";'
    'x.setTransform(dpr,0,0,dpr,0,0)}'
    'window.addEventListener("resize",rs);rs();'

    # Fit-and-center: uniform scale that fits data in canvas + center offset
    # Use CSS (logical) size, not backing-store size, since ctx has DPR transform
    'var W,H,SC,OX,OY;'
    'function uw(){'
    'W=parseFloat(c.style.width)||c.width;'
    'H=parseFloat(c.style.height)||c.height;'
    'var dw=VX1-VX0,dh=VY1-VY0;'
    'SC=Math.min(W/dw,H/dh);'
    'OX=(W-dw*SC)/2;OY=(H-dh*SC)/2}'

    # coordinate mapping — fit-and-center with uniform scale
    'function mx(v){return(v-VX0)*SC+OX}'
    'function my(v){return(v-VY0)*SC+OY}'
    'function gp(f,i){var o=(f*NL+i)*2;return[P[o],P[o+1]]}'
    'function gc(f,i){return CF[f*NL+i]}'

    # draw edges
    'function el(f,cn,o,cl,lw){'
    'x.strokeStyle=cl;x.lineWidth=lw;x.beginPath();'
    'for(var k=0;k<cn.length;k++){'
    'var a=cn[k][0]+o,b=cn[k][1]+o;'
    'if(gc(f,a)<.1||gc(f,b)<.1)continue;'
    'var p=gp(f,a),q=gp(f,b);'
    'x.moveTo(mx(p[0]),my(p[1]));x.lineTo(mx(q[0]),my(q[1]))'
    '}x.stroke()}'

    # draw dots
    'function dt(f,s,e,cl,sz){'
    'x.fillStyle=cl;for(var i=s;i<e;i++){'
    'if(gc(f,i)<.1)continue;var p=gp(f,i);'
    'x.beginPath();x.arc(mx(p[0]),my(p[1]),sz,0,6.283);x.fill()}}'

    # draw full frame — sizes scale with canvas
    'function dr(f){uw();'
    'x.clearRect(0,0,W,H);x.fillStyle="#000";x.fillRect(0,0,W,H);'
    'if(f<0||f>=NF)return;'
    'var s=Math.max(1,W/200);'  # scale factor: 1px per 200px canvas width
    # body — shoulders + elbows (0-14)
    'el(f,BC,0,"royalblue",1.5*s);dt(f,0,15,"royalblue",2.5*s);'
    # hands — prominent for sign language
    'el(f,HC,33,"limegreen",1.2*s);dt(f,33,54,"limegreen",2*s);'
    'el(f,HC,54,"crimson",1.2*s);dt(f,54,75,"crimson",2*s);'
    # face — connections only, no dots
    'el(f,FO,75,"rgba(255,215,0,0.2)",0.8*s);'
    'el(f,FL,75,"hotpink",1.2*s);'
    'el(f,FLE,75,"cyan",1*s);el(f,FRE,75,"cyan",1*s);'
    'el(f,FLB,75,"orange",1*s);el(f,FRB,75,"orange",1*s)}'

    # playback state
    'var fr=0,pl=true,lo=true,spd=1,lt=0;'

    # format milliseconds
    'function fm(ms){'
    'var m=Math.floor(ms/60000),s=Math.floor((ms%60000)/1000),'
    'r=Math.round(ms%1000);'
    'return(m<10?"0":"")+m+":"+(s<10?"0":"")+s+"."+'
    '(r<100?(r<10?"00":"0"):"")+r}'

    # update UI
    'function ui(){scr.value=fr;'
    'var ms=Math.round(fr/FPS*1000);'
    'te.textContent=fm(ms+SMS)+" / $total_time";'
    'pp.innerHTML=pl?"&#x23F8;":"&#x25B6;";'
    'lb.className=lo?"on":""}'

    # animation loop
    'function tick(ts){if(pl){'
    'if(ts-lt>=1000/(FPS*spd)){lt=ts;fr++;'
    'if(fr>=NF){if(lo)fr=0;else{fr=NF-1;pl=false}}'
    'dr(fr);ui()}}'
    'requestAnimationFrame(tick)}'

    # init
    'dr(0);ui();requestAnimationFrame(tick);'

    # event handlers
    'pp.onclick=function(){pl=!pl;if(pl)lt=performance.now();ui()};'
    'scr.oninput=function(){fr=+scr.value;pl=false;dr(fr);ui()};'
    'se.onchange=function(){spd=+se.value};'
    'lb.onclick=function(){lo=!lo;ui()};'
    'c.onclick=function(){pp.click()};'
    'document.onkeydown=function(e){'
    'if(e.code==="Space"){e.preventDefault();pp.click()}'
    'else if(e.code==="ArrowLeft"){'
    'e.preventDefault();fr=Math.max(0,fr-1);pl=false;dr(fr);ui()}'
    'else if(e.code==="ArrowRight"){'
    'e.preventDefault();fr=Math.min(NF-1,fr+1);pl=false;dr(fr);ui()}};'

    '})();'
    '</script></body></html>'
)


def pose_animation_html(
    pose_data: np.ndarray,
    conf_data: np.ndarray,
    fps: float,
    frame_start: int,
    frame_end: int,
    start_ms: int = 0,
    video_aspect: float = 0,
) -> str:
    """Generate self-contained HTML/JS for real-time pose skeleton animation.

    Renders at actual pose FPS using requestAnimationFrame — no Streamlit
    reruns needed.  Use with ``st.components.v1.html(html, height=600)``.

    Same colour scheme as :func:`pose_frame_figure`: body=royalblue,
    left_hand=limegreen, right_hand=crimson, lips=hotpink, eyes=cyan,
    eyebrows=orange, face_oval=gold.

    Controls: play/pause, scrubber, speed (0.25×–2×), loop, keyboard
    (Space, ←/→ arrows).
    """
    import base64
    import json
    from string import Template

    n_frames = max(0, frame_end - frame_start)
    if n_frames == 0:
        return (
            '<div style="color:#fafafa;text-align:center;padding:40px">'
            'No frames in segment</div>'
        )

    # ── Extract segment data ──────────────────────────────────────────
    if pose_data.ndim == 4:
        seg_xy = pose_data[frame_start:frame_end, 0, :, :2].copy()
    else:
        seg_xy = pose_data[frame_start:frame_end, :, :2].copy()

    if conf_data.ndim == 4:
        seg_conf = conf_data[frame_start:frame_end, 0, :, 0].copy()
    elif conf_data.ndim == 3:
        seg_conf = (
            conf_data[frame_start:frame_end, 0, :].copy()
            if conf_data.shape[1] == 1
            else conf_data[frame_start:frame_end, :, 0].copy()
        )
    else:
        seg_conf = conf_data[frame_start:frame_end].copy()

    # ── Viewport: signing space — face + shoulders + hands ──
    # Body 0-12: face points + shoulders.  Hands 33-74: detailed hand model.
    # Elbows (13-14) excluded — they extend far below hands when arms
    # are bent upward for signing, stretching the viewport.
    sign_idx = list(range(0, 13)) + list(range(33, 75))
    signing_xy   = seg_xy[:, sign_idx, :]
    signing_conf = seg_conf[:, sign_idx]
    vis_mask = signing_conf >= 0.1
    vis_x = signing_xy[:, :, 0][vis_mask]
    vis_y = signing_xy[:, :, 1][vis_mask]
    pad = 0.05

    if len(vis_x) > 0:
        vx_min = float(vis_x.min()) - pad
        vx_max = float(vis_x.max()) + pad
        vy_min = float(vis_y.min()) - pad
        vy_max = float(vis_y.max()) + pad
    else:
        vx_min, vx_max, vy_min, vy_max = 0.0, 1.0, 0.0, 1.0

    if vx_max - vx_min < 0.1:
        cx = (vx_max + vx_min) / 2
        vx_min, vx_max = cx - 0.15, cx + 0.15
    if vy_max - vy_min < 0.1:
        cy = (vy_max + vy_min) / 2
        vy_min, vy_max = cy - 0.15, cy + 0.15

    # Viewport is the tight data bounding box (with padding).
    # The JavaScript fit-and-center logic adapts it to the actual canvas
    # dimensions at runtime — no need to pre-match aspect ratios here.

    # ── Encode as float32 binary → base64 ─────────────────────────────
    pos_b64 = base64.b64encode(
        seg_xy.astype(np.float32).tobytes()
    ).decode("ascii")
    conf_b64 = base64.b64encode(
        seg_conf.astype(np.float32).tobytes()
    ).decode("ascii")

    # ── Format helpers ────────────────────────────────────────────────
    total_ms = int(n_frames / fps * 1000)
    end_ms = start_ms + total_ms

    def _fmt(ms: int) -> str:
        m = ms // 60_000
        s = (ms % 60_000) // 1_000
        r = ms % 1_000
        return f"{m:02d}:{s:02d}.{r:03d}"

    return _POSE_ANIM_TEMPLATE.substitute(
        n_frames=n_frames,
        fps=fps,
        start_ms=start_ms,
        max_frame=max(0, n_frames - 1),
        total_time=_fmt(end_ms),
        fps_display=f"{fps:.0f}",
        vx_min=vx_min,
        vx_max=vx_max,
        vy_min=vy_min,
        vy_max=vy_max,
        video_aspect=video_aspect,
        pos_b64=pos_b64,
        conf_b64=conf_b64,
        body_conn=json.dumps(BODY_CONNECTIONS),
        hand_conn=json.dumps(HAND_CONNECTIONS),
        face_lips=json.dumps(_FACE_LIPS),
        face_left_eye=json.dumps(_FACE_LEFT_EYE),
        face_right_eye=json.dumps(_FACE_RIGHT_EYE),
        face_left_brow=json.dumps(_FACE_LEFT_EYEBROW),
        face_right_brow=json.dumps(_FACE_RIGHT_EYEBROW),
        face_oval=json.dumps(_FACE_OVAL),
    )


# ---------------------------------------------------------------------------
# Synced video + pose animation (single HTML component)
# ---------------------------------------------------------------------------

_SYNCED_ANIM_TEMPLATE = _Template(
    '<!DOCTYPE html>\n<html><head><style>'
    '*{margin:0;padding:0;box-sizing:border-box}'
    'body{background:transparent;font-family:-apple-system,BlinkMacSystemFont,sans-serif;color:#fafafa}'
    '#w{display:flex;flex-direction:column;align-items:center;width:100%;max-height:100vh}'
    '#vp{display:flex;gap:8px;width:100%;justify-content:center;align-items:flex-start}'
    'video{background:#000;border-radius:6px;flex:1;min-width:0;max-width:50%}'
    'canvas{background:#000;border-radius:6px}'
    '#ct{display:flex;align-items:center;gap:8px;padding:8px 4px;font-size:13px;width:100%}'
    '#ct button{background:#262730;border:1px solid #4a4a5a;color:#fafafa;'
    'border-radius:4px;padding:3px 12px;cursor:pointer;font-size:15px;line-height:1}'
    '#ct button:hover{background:#3a3a4a}'
    '#ct button.on{background:#ff6b6b33;border-color:#ff6b6b}'
    '#ct input[type=range]{flex:1;accent-color:#ff6b6b;cursor:pointer}'
    '#ct select{background:#262730;border:1px solid #4a4a5a;color:#fafafa;'
    'border-radius:4px;padding:2px 4px;font-size:12px;cursor:pointer}'
    '#tm{font-variant-numeric:tabular-nums;min-width:90px;text-align:center;font-size:12px;opacity:.7}'
    '#fp{font-size:11px;opacity:.5}'
    '</style></head><body>'
    '<div id="w">'
    '<div id="vp">'
    '<video id="vid" preload="auto" playsinline muted></video>'
    '<canvas id="c"></canvas>'
    '</div>'
    '<div id="ct">'
    '<button id="pp" title="Play / Pause (Space)">&#x25B6;</button>'
    '<input type="range" id="sc" min="0" max="1000" value="0" step="1">'
    '<span id="tm">00:00.000</span>'
    '<select id="sp">'
    '<option value="0.25">0.25x</option>'
    '<option value="0.5">0.5x</option>'
    '<option value="1" selected>1x</option>'
    '<option value="1.5">1.5x</option>'
    '<option value="2">2x</option>'
    '</select>'
    '<button id="lp" class="on" title="Loop">&#x21BB;</button>'
    '<span id="fp">$fps_display fps</span>'
    '</div></div>'
    '<script>'
    '(function(){'
    '"use strict";'
    'var NF=$n_frames,NL=543,PFPS=$fps;'
    'var VX0=$vx_min,VX1=$vx_max,VY0=$vy_min,VY1=$vy_max;'

    # decode base64 helpers
    'function d64(s){var b=atob(s),a=new Uint8Array(b.length);'
    'for(var i=0;i<b.length;i++)a[i]=b.charCodeAt(i);'
    'return new Float32Array(a.buffer)}'
    'var P=d64("$pos_b64");var CF=d64("$conf_b64");'

    # connection arrays
    'var BC=$body_conn,HC=$hand_conn;'
    'var FL=$face_lips,FLE=$face_left_eye,FRE=$face_right_eye;'
    'var FLB=$face_left_brow,FRB=$face_right_brow,FO=$face_oval;'

    # elements
    'var vid=document.getElementById("vid");'
    'var c=document.getElementById("c"),x=c.getContext("2d");'
    'var pp=document.getElementById("pp"),scr=document.getElementById("sc");'
    'var te=document.getElementById("tm"),se=document.getElementById("sp");'
    'var lb=document.getElementById("lp");'

    # set video source from base64
    'vid.src="data:video/mp4;base64,$video_b64";'

    # Compute panel size from video intrinsic aspect, size both elements
    'var vp=document.getElementById("vp");'
    'function rs(){'
    'var dpr=window.devicePixelRatio||1;'
    'var avW=Math.floor((vp.clientWidth-8)/2)||300;'
    'var avH=Math.max(200,(window.innerHeight||600)-80);'
    'var ar=(vid.videoWidth&&vid.videoHeight)'
    '?vid.videoHeight/vid.videoWidth'
    ':(VY1-VY0)/(VX1-VX0);'
    'var w=avW,h=Math.round(avW*ar);'
    'if(h>avH){h=avH;w=Math.round(h/ar)}'
    'vid.style.flex="none";vid.style.width=w+"px";vid.style.height=h+"px";'
    'c.width=Math.round(w*dpr);c.height=Math.round(h*dpr);'
    'c.style.width=w+"px";c.style.height=h+"px";'
    'x.setTransform(dpr,0,0,dpr,0,0)}'

    # non-uniform scaling: map [0,1]×[0,1] to full canvas (matches video frame)
    'var W,H,SX,SY;'
    'function uw(){'
    'W=parseFloat(c.style.width)||c.width;'
    'H=parseFloat(c.style.height)||c.height;'
    'SX=W/(VX1-VX0);SY=H/(VY1-VY0)}'

    # coordinate mapping (independent X/Y scale)
    'function mx(v){return(v-VX0)*SX}'
    'function my(v){return(v-VY0)*SY}'
    'function gp(f,i){var o=(f*NL+i)*2;return[P[o],P[o+1]]}'
    'function gc(f,i){return CF[f*NL+i]}'

    # draw edges
    'function el(f,cn,o,cl,lw){'
    'x.strokeStyle=cl;x.lineWidth=lw;x.beginPath();'
    'for(var k=0;k<cn.length;k++){'
    'var a=cn[k][0]+o,b=cn[k][1]+o;'
    'if(gc(f,a)<.1||gc(f,b)<.1)continue;'
    'var p=gp(f,a),q=gp(f,b);'
    'x.moveTo(mx(p[0]),my(p[1]));x.lineTo(mx(q[0]),my(q[1]))'
    '}x.stroke()}'

    # draw dots
    'function dt(f,s,e,cl,sz){'
    'x.fillStyle=cl;for(var i=s;i<e;i++){'
    'if(gc(f,i)<.1)continue;var p=gp(f,i);'
    'x.beginPath();x.arc(mx(p[0]),my(p[1]),sz,0,6.283);x.fill()}}'

    # draw full frame — sizes scale with canvas
    'function dr(f){uw();'
    'x.clearRect(0,0,W,H);x.fillStyle="#000";x.fillRect(0,0,W,H);'
    'if(f<0||f>=NF)return;'
    'var s=Math.max(1,W/200);'  # scale factor: 1px per 200px canvas width
    'el(f,BC,0,"royalblue",1.5*s);dt(f,0,15,"royalblue",2.5*s);'
    'el(f,HC,33,"limegreen",1.2*s);dt(f,33,54,"limegreen",2*s);'
    'el(f,HC,54,"crimson",1.2*s);dt(f,54,75,"crimson",2*s);'
    'el(f,FO,75,"rgba(255,215,0,0.2)",0.8*s);'
    'el(f,FL,75,"hotpink",1.2*s);'
    'el(f,FLE,75,"cyan",1*s);el(f,FRE,75,"cyan",1*s);'
    'el(f,FLB,75,"orange",1*s);el(f,FRB,75,"orange",1*s)}'

    # format time
    'function fm(s){'
    'var m=Math.floor(s/60),sec=Math.floor(s%60),ms=Math.round((s%1)*1000);'
    'return(m<10?"0":"")+m+":"+(sec<10?"0":"")+sec+"."+'
    '(ms<100?(ms<10?"00":"0"):"")+ms}'

    # sync: map video time → pose frame index
    'var dur=0;'
    'function syncFrame(){'
    'if(!dur)dur=vid.duration||1;'
    'var t=vid.currentTime;'
    'var f=Math.min(Math.floor(t*PFPS),NF-1);'
    'dr(Math.max(0,f));'
    'scr.value=Math.round(t/dur*1000);'
    'te.textContent=fm(t)+" / "+fm(dur);'
    'pp.innerHTML=vid.paused?"&#x25B6;":"&#x23F8;"}'

    # Primary sync: requestVideoFrameCallback (frame-accurate)
    'var hasRVFC="requestVideoFrameCallback" in HTMLVideoElement.prototype;'
    'if(hasRVFC){'
    'vid.requestVideoFrameCallback(function onFrame(now,meta){'
    'syncFrame();vid.requestVideoFrameCallback(onFrame)})}'

    # Fallback sync: timeupdate (~4Hz)
    'vid.addEventListener("timeupdate",syncFrame);'
    'vid.addEventListener("seeked",syncFrame);'

    # init on metadata loaded
    'vid.addEventListener("loadedmetadata",function(){'
    'dur=vid.duration;rs();dr(0);syncFrame()});'
    'vid.addEventListener("loadeddata",rs);'

    # controls
    'pp.onclick=function(){if(vid.paused)vid.play();else vid.pause();syncFrame()};'
    'scr.oninput=function(){if(dur)vid.currentTime=+scr.value/1000*dur;syncFrame()};'
    'se.onchange=function(){vid.playbackRate=+se.value};'
    'lb.onclick=function(){vid.loop=!vid.loop;lb.className=vid.loop?"on":""};'
    'vid.loop=true;'

    # keyboard
    'document.onkeydown=function(e){'
    'if(e.code==="Space"){e.preventDefault();pp.click()}'
    'else if(e.code==="ArrowLeft"){'
    'e.preventDefault();vid.currentTime=Math.max(0,vid.currentTime-1/PFPS);syncFrame()}'
    'else if(e.code==="ArrowRight"){'
    'e.preventDefault();vid.currentTime=Math.min(dur,vid.currentTime+1/PFPS);syncFrame()}};'

    # initial draw
    'dr(0);syncFrame();'
    'window.addEventListener("resize",function(){rs();syncFrame()});'

    '})();'
    '</script></body></html>'
)


def encode_pose_data(
    pose_data: np.ndarray,
    conf_data: np.ndarray,
    frame_start: int,
    frame_end: int,
) -> tuple[str, str, int]:
    """Encode a pose segment as base64 strings for JS rendering.

    Args:
        pose_data: Full pose array — (T,1,543,3) or (T,543,3).
        conf_data: Full confidence array — various shapes handled.
        frame_start: First frame index (inclusive).
        frame_end: Last frame index (exclusive).

    Returns:
        (pos_b64, conf_b64, n_frames) — base64-encoded xy coords,
        base64-encoded confidence, and frame count.
    """
    import base64

    n_frames = max(0, frame_end - frame_start)
    if n_frames == 0:
        return "", "", 0

    if pose_data.ndim == 4:
        seg_xy = pose_data[frame_start:frame_end, 0, :, :2].copy()
    else:
        seg_xy = pose_data[frame_start:frame_end, :, :2].copy()

    if conf_data.ndim == 4:
        seg_conf = conf_data[frame_start:frame_end, 0, :, 0].copy()
    elif conf_data.ndim == 3:
        seg_conf = (
            conf_data[frame_start:frame_end, 0, :].copy()
            if conf_data.shape[1] == 1
            else conf_data[frame_start:frame_end, :, 0].copy()
        )
    else:
        seg_conf = conf_data[frame_start:frame_end].copy()

    pos_b64 = base64.b64encode(
        seg_xy.astype(np.float32).tobytes()
    ).decode("ascii")
    conf_b64 = base64.b64encode(
        seg_conf.astype(np.float32).tobytes()
    ).decode("ascii")
    return pos_b64, conf_b64, n_frames


CONNECTION_ARRAYS: dict[str, list] = {
    "body_conn": BODY_CONNECTIONS,
    "hand_conn": HAND_CONNECTIONS,
    "face_lips": _FACE_LIPS,
    "face_left_eye": _FACE_LEFT_EYE,
    "face_right_eye": _FACE_RIGHT_EYE,
    "face_left_brow": _FACE_LEFT_EYEBROW,
    "face_right_brow": _FACE_RIGHT_EYEBROW,
    "face_oval": _FACE_OVAL,
}


def synced_video_pose_html(
    video_bytes: bytes,
    pose_data: np.ndarray,
    conf_data: np.ndarray,
    fps: float,
    frame_start: int,
    frame_end: int,
) -> str:
    """Generate HTML with synced <video> + <canvas> in a single component.

    The video element is the master clock; the canvas redraws on every frame
    via requestVideoFrameCallback (with timeupdate fallback).

    Use with ``st.components.v1.html(html, height=500)``.
    """
    import base64
    import json

    pos_b64, conf_b64, n_frames = encode_pose_data(
        pose_data, conf_data, frame_start, frame_end,
    )
    if n_frames == 0:
        return (
            '<div style="color:#fafafa;text-align:center;padding:40px">'
            'No frames in segment</div>'
        )

    video_b64 = base64.b64encode(video_bytes).decode("ascii")

    return _SYNCED_ANIM_TEMPLATE.substitute(
        n_frames=n_frames,
        fps=fps,
        fps_display=f"{fps:.0f}",
        vx_min=0.0,
        vx_max=1.0,
        vy_min=0.0,
        vy_max=1.0,
        pos_b64=pos_b64,
        conf_b64=conf_b64,
        video_b64=video_b64,
        body_conn=json.dumps(BODY_CONNECTIONS),
        hand_conn=json.dumps(HAND_CONNECTIONS),
        face_lips=json.dumps(_FACE_LIPS),
        face_left_eye=json.dumps(_FACE_LEFT_EYE),
        face_right_eye=json.dumps(_FACE_RIGHT_EYE),
        face_left_brow=json.dumps(_FACE_LEFT_EYEBROW),
        face_right_brow=json.dumps(_FACE_RIGHT_EYEBROW),
        face_oval=json.dumps(_FACE_OVAL),
    )


# ---------------------------------------------------------------------------
# EAF Harvest — bulk import of human-corrected S1_Gloss tiers
# ---------------------------------------------------------------------------

GLOSS_RE = re.compile(r"^[A-Z][A-Z0-9_-]+$")

_GLOSS_TIER_HANDS = {"S1_Gloss_RH": "right", "S1_Gloss_LH": "left"}


def parse_gloss_value(
    value: str,
    glossary: "Optional[Glossary]" = None,
) -> tuple[str, str]:
    """Parse a gloss annotation value into (word, gloss_id).

    UPPERCASE values matching GLOSS_RE are treated as gloss_id directly.
    Anything else is treated as a word and optionally looked up in the glossary.

    Returns:
        (word, gloss_id) tuple — one or both may be empty string.
    """
    value = value.strip()
    if not value:
        return "", ""
    if GLOSS_RE.match(value):
        return "", value
    word = value.lower()
    gloss_id = ""
    if glossary:
        matches = glossary.lookup(word)
        if matches:
            gloss_id = matches[0]
    return word, gloss_id


def harvest_eaf_annotations(
    eaf: "pympi.Eaf",
    stem: str,
    video_path: str,
    pose_path: str,
    fps: float,
    glossary: "Optional[Glossary]" = None,
) -> list[dict]:
    """Extract human S1_Gloss_RH / S1_Gloss_LH annotations as pairing dicts.

    Gloss values matching ``^[A-Z][A-Z0-9_-]+$`` are treated as gloss_id
    directly; anything else is treated as a word and looked up in the glossary
    via its reverse index (``glossary.lookup(word)``).

    Args:
        eaf:        Already-loaded pympi.Eaf instance.
        stem:       File stem (used for pairing_id / segment_id).
        video_path: Path string to the source video.
        pose_path:  Path string to the .pose file.
        fps:        Frame rate of the video.
        glossary:   Optional glossary for word→gloss_id lookup.

    Returns:
        List of pairing dicts (same schema as pairings.csv rows).
    """
    results: list[dict] = []
    present_tiers = eaf.get_tier_names()

    for tier_name, hand in _GLOSS_TIER_HANDS.items():
        if tier_name not in present_tiers:
            continue
        for start_ms, end_ms, value, *_ in eaf.get_annotation_data_for_tier(tier_name):
            value = str(value).strip() if not pd.isna(value) else ""
            if not value:
                continue

            word, gloss_id = parse_gloss_value(value, glossary)

            results.append(make_pairing_dict(
                pairing_id=f"{stem}_{start_ms:08d}_{hand[0]}_E",
                segment_id=f"{stem}_eaf",
                video_path=video_path,
                pose_path=pose_path,
                hand=hand,
                sign_start_ms=int(start_ms),
                sign_end_ms=int(end_ms),
                sign_frame_start=int(start_ms * fps / 1000),
                sign_frame_end=int(end_ms * fps / 1000),
                fps=fps,
                note="eaf_harvest",
                word=word,
                gloss_id=gloss_id,
                status=PST_PAIRED,
            ))

    return results


def harvest_eaf_batch(
    annotations_dir: Path,
    pose_dir: Path,
    inventory_df: pd.DataFrame,
    pairings_path: Path,
    glossary: "Optional[Glossary]" = None,
    progress_callback: "Optional[callable]" = None,
) -> dict:
    """Bulk-harvest human S1_Gloss annotations from all EAF files.

    Scans *annotations_dir* for .eaf files that have human-tier annotations,
    matches each to the inventory for video_path / fps, and appends new paired
    rows to *pairings_path* (deduplicating by stem+start_ms+hand).

    Args:
        annotations_dir: Directory containing .eaf files.
        pose_dir:        Directory containing .pose files.
        inventory_df:    Inventory DataFrame (must have ``filename``, ``fps`` columns).
        pairings_path:   Path to pairings.csv (created if missing).
        glossary:        Optional glossary for word→gloss_id lookup.
        progress_callback: Optional ``fn(current, total)`` for progress reporting.

    Returns:
        Dict with keys: n_files_scanned, n_with_annotations, n_new_pairings,
        n_skipped_dupes.
    """
    from spj.eaf import load_eaf

    annotations_dir = Path(annotations_dir)
    pose_dir = Path(pose_dir)
    eaf_files = sorted(annotations_dir.glob("*.eaf"))

    # Build inventory lookup: stem → (video_path, fps)  (vectorized)
    vp_col = "video_path" if "video_path" in inventory_df.columns else "path"
    inv_lookup: dict[str, tuple[str, float]] = dict(zip(
        inventory_df["filename"].apply(lambda f: Path(str(f)).stem),
        zip(inventory_df[vp_col].astype(str), inventory_df["fps"].astype(float)),
    ))

    n_scanned = 0
    n_with_ann = 0
    all_new: list[dict] = []
    n_dupes = 0
    existing_df: pd.DataFrame = pd.DataFrame()
    dedup_set: set[tuple[str, int, str]] | None = None  # lazy init

    n_total = len(eaf_files)
    gloss_tiers = tuple(_GLOSS_TIER_HANDS.keys())

    for i, eaf_path in enumerate(eaf_files):
        n_scanned += 1
        if progress_callback and (i % 50 == 0 or i == n_total - 1):
            progress_callback(i + 1, n_total)

        stem = eaf_path.stem

        # Check inventory first (cheap) before loading EAF (expensive)
        info = inv_lookup.get(stem)
        if not info:
            continue
        video_path, fps = info

        try:
            eaf = load_eaf(eaf_path)
        except Exception:
            logger.warning("Failed to load EAF: %s", eaf_path)
            continue

        # Check for human gloss tiers (tier_names is O(1))
        present = eaf.get_tier_names()
        if not any(t in present for t in gloss_tiers):
            continue

        # Harvest annotations — returns empty list if tiers exist but are empty
        pairings = harvest_eaf_annotations(
            eaf, stem, video_path, str(pose_dir / f"{stem}.pose"),
            fps, glossary,
        )
        if not pairings:
            continue
        n_with_ann += 1

        # Lazy-init dedup set on first file with annotations
        if dedup_set is None:
            existing_df = load_pairings_csv(pairings_path)
            if existing_df.empty:
                dedup_set = set()
            else:
                dedup_set = set(zip(
                    existing_df["video_path"].apply(lambda p: Path(p).stem),
                    existing_df["sign_start_ms"].astype(int),
                    existing_df["hand"].astype(str),
                ))

        for p in pairings:
            key = (stem, p["sign_start_ms"], p["hand"])
            if key in dedup_set:
                n_dupes += 1
                continue
            dedup_set.add(key)
            all_new.append(p)

    # Append and save
    if all_new:
        new_df = pd.DataFrame(all_new)
        if not existing_df.empty:
            combined = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined = new_df
        save_pairings_csv(combined, pairings_path)

    return {
        "n_files_scanned": n_scanned,
        "n_with_annotations": n_with_ann,
        "n_new_pairings": len(all_new),
        "n_skipped_dupes": n_dupes,
    }


def harvest_eaf_auto(data_dir: Path) -> Optional[dict]:
    """Convenience wrapper: resolve paths, load inputs, run harvest.

    Returns harvest result dict, or ``None`` if prerequisites are missing
    (no annotations dir or no inventory.csv).
    """
    data_dir = Path(data_dir)
    annotations_dir = data_dir / "annotations"
    inventory_path = data_dir / "inventory.csv"
    if not annotations_dir.exists() or not inventory_path.exists():
        return None

    glossary_path = data_dir / "training" / "glossary.json"
    glossary = None
    if glossary_path.exists():
        from spj.glossary import load_glossary
        glossary = load_glossary(glossary_path)

    return harvest_eaf_batch(
        annotations_dir=annotations_dir,
        pose_dir=data_dir / "pose",
        inventory_df=pd.read_csv(inventory_path),
        pairings_path=data_dir / "training" / "pairings.csv",
        glossary=glossary,
    )
