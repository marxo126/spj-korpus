"""Training data pipeline: pose-subtitle alignment, human review, and NPZ export.

Aligns .pose keypoint files with .vtt subtitle entries to produce labelled
training segments for SignBERT / OpenHands fine-tuning.
"""
from __future__ import annotations

import json
import logging
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

def export_segment_npz(row: "pd.Series", output_dir: Path) -> Optional[str]:
    """Export one approved segment as a float16 .npz training file.

    Arrays saved:
        pose(T,543,3)       float16 — normalised XYZ
        confidence(T,543,1) float16 — per-landmark confidence
    Scalars: text, fps, start_ms, end_ms, source_video

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

    text = str(row["reviewed_text"]).strip() or str(row["text"])
    stem = Path(str(row["video_path"])).stem

    out_path = output_dir / f"{stem}_{orig_start:08d}.npz"

    np.savez_compressed(
        str(out_path),
        pose         = pose_slice.astype(np.float16),
        confidence   = conf_slice.astype(np.float16),
        text         = text,
        fps          = float(fps),
        start_ms     = rev_start,
        end_ms       = rev_end,
        source_video = str(row["video_path"]),
    )

    return str(out_path)


def write_training_config(output_dir: Path, n_segments: int) -> Path:
    """Write a JSON training config optimised for M4 Max alongside the .npz files."""
    config = {
        "device": "mps",
        "batch_size": 256,
        "pin_memory": False,
        "precision": "fp16",
        "n_segments": n_segments,
        "pose_shape": [None, 543, 3],
        "confidence_shape": [None, 543, 1],
        "dataloader": {
            "num_workers": 0,
            "pin_memory": False,
            "prefetch_factor": None,
        },
        "notes": (
            "M4 Max optimised: fp16 arrays, MPS backend, batch_size=256. "
            "pin_memory=False (unified memory — no separate GPU VRAM to pin). "
            "Full dataset fits in 128 GB RAM — pre-load into dict at training start."
        ),
    }
    cfg_path = Path(output_dir) / "training_config.json"
    cfg_path.write_text(json.dumps(config, indent=2))
    return cfg_path


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

    _add_dots(_BODY_OFFSET,       _BODY_OFFSET + 33,       "royalblue", "body")
    _add_dots(_LEFT_HAND_OFFSET,  _LEFT_HAND_OFFSET + 21,  "limegreen", "left hand")
    _add_dots(_RIGHT_HAND_OFFSET, _RIGHT_HAND_OFFSET + 21, "crimson",   "right hand")

    # Face features
    _add_edges(_FACE_OVAL,           _FACE_OFFSET, "rgba(255,215,0,0.3)")
    _add_edges(_FACE_LIPS,           _FACE_OFFSET, "hotpink")
    _add_edges(_FACE_LEFT_EYE,       _FACE_OFFSET, "cyan")
    _add_edges(_FACE_RIGHT_EYE,      _FACE_OFFSET, "cyan")
    _add_edges(_FACE_LEFT_EYEBROW,   _FACE_OFFSET, "orange")
    _add_edges(_FACE_RIGHT_EYEBROW,  _FACE_OFFSET, "orange")

    _add_dots(_FACE_OFFSET, _FACE_OFFSET + 468, "rgba(255,215,0,0.4)", "face", size=2)

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
    'function rs(){'
    'var pw=c.parentElement.clientWidth;'
    'var maxH=Math.max(300,(window.innerHeight||800)-60);'
    'var da=(VY1-VY0)/(VX1-VX0);'  # data aspect
    'var w=pw,h=Math.round(pw*da);'
    'if(h>maxH){h=maxH;w=Math.round(h/da)}'
    'c.width=w;c.height=h}'
    'window.addEventListener("resize",rs);rs();'

    # Fit-and-center: uniform scale that fits data in canvas + center offset
    'var W,H,SC,OX,OY;'
    'function uw(){'
    'W=c.width;H=c.height;'
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

    # draw full frame
    'function dr(f){uw();'
    'x.clearRect(0,0,W,H);x.fillStyle="#000";x.fillRect(0,0,W,H);'
    # body — face + shoulders + elbows (0-14) for body direction
    'el(f,BC,0,"royalblue",3.5);dt(f,0,15,"royalblue",6);'
    # hands — prominent for sign language (detailed hand model)
    'el(f,HC,33,"limegreen",3);dt(f,33,54,"limegreen",5);'
    'el(f,HC,54,"crimson",3);dt(f,54,75,"crimson",5);'
    # face
    'el(f,FO,75,"rgba(255,215,0,0.25)",1);'
    'el(f,FL,75,"hotpink",2);'
    'el(f,FLE,75,"cyan",2);el(f,FRE,75,"cyan",2);'
    'el(f,FLB,75,"orange",2);el(f,FRB,75,"orange",2);'
    'dt(f,75,543,"rgba(255,215,0,0.35)",2)}'

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
