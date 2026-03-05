"""Motion-based sign boundary detection and EAF pre-annotation.

Algorithm:
  1. Load pose landmarks from .pose file (or .npz fallback).
  2. Compute per-frame wrist speed using body wrist + hand wrist landmarks,
     preferring the hand-specific landmark when the hand is confidently detected.
  3. Gaussian-smooth the speed signal (pure numpy, no scipy needed).
  4. Normalise to [0, 1] relative to the 99th-percentile peak and threshold.
  5. Group active frames into sign segments with min/max duration guards.
  6. Merge segments separated by short gaps to avoid over-splitting.
  7. Write (start_ms, end_ms, "?") into AI_Gloss_RH / AI_Gloss_LH EAF tiers.
     AI_Confidence tier gets the average normalised speed for each segment.

No trained model is required — purely kinematic.  Works at cold-start before
any annotated data exists.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Landmark indices in our 543-point array
# (33 body + 21 left hand + 21 right hand + 468 face)
# ---------------------------------------------------------------------------
_BODY_LEFT_WRIST  = 15   # BODY_15
_BODY_RIGHT_WRIST = 16   # BODY_16
_BODY_LEFT_ELBOW  = 13   # BODY_13
_BODY_RIGHT_ELBOW = 14   # BODY_14
_LEFT_HAND_WRIST  = 33   # LEFT_HAND_0  (first landmark of left hand block)
_RIGHT_HAND_WRIST = 54   # RIGHT_HAND_0 (first landmark of right hand block)


# ---------------------------------------------------------------------------
# Pose loading
# ---------------------------------------------------------------------------

def load_pose_arrays(pose_path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Load a .pose file and return (data, confidence, fps).

    data:       float32 ndarray shape (T, 1, 543, 3)  — normalised XYZ coords
    confidence: float32 ndarray shape (T, 1, 543, 1)  — per-landmark confidence
    fps:        frames per second

    Falls back to the .npz sidecar created when pose_format is unavailable.

    Raises FileNotFoundError if the file is missing, empty, or unreadable.
    """
    pose_path = Path(pose_path)

    if not pose_path.exists():
        raise FileNotFoundError(f"Pose file not found: {pose_path}")
    if pose_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Pose file is empty (0 bytes) — re-run extraction: {pose_path}"
        )

    try:
        from pose_format import Pose

        with open(pose_path, "rb") as fh:
            pose = Pose.read(fh.read())
        data = np.array(pose.body.data,       dtype=np.float32)   # (T, 1, N, 3)
        conf = np.array(pose.body.confidence, dtype=np.float32)   # may be (T, 1, N) or (T, 1, N, 1)
        fps  = float(pose.body.fps)
        # Ensure confidence has trailing dimension: (T, 1, N) -> (T, 1, N, 1)
        if conf.ndim == 3:
            conf = conf[..., np.newaxis]
        return data, conf, fps
    except Exception:
        pass

    npz_path = pose_path.with_suffix(".npz")
    if npz_path.exists() and npz_path.stat().st_size > 0:
        d    = np.load(str(npz_path))
        arr  = d["pose"].astype(np.float32)
        fps  = float(d["fps"])
        if arr.ndim == 3 and arr.shape[-1] >= 3:
            # (T, 543, 4) or (T, 543, 3)
            data = arr[:, np.newaxis, :, :3]           # (T, 1, 543, 3)
            if arr.shape[-1] >= 4:
                conf = arr[:, np.newaxis, :, 3:4]      # (T, 1, 543, 1)
            else:
                conf = np.ones_like(data[:, :, :, :1])  # default confidence 1.0
        elif arr.ndim == 4:
            # Already (T, 1, 543, 3+)
            data = arr[:, :, :, :3]
            if arr.shape[-1] >= 4:
                conf = arr[:, :, :, 3:4]
            else:
                conf = np.ones_like(data[:, :, :, :1])
        else:
            raise FileNotFoundError(
                f"Unexpected pose array shape {arr.shape} in {npz_path}"
            )
        return data, conf, fps

    raise FileNotFoundError(f"Cannot load pose data from {pose_path}")


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------

def _gaussian_smooth(signal: np.ndarray, sigma: float) -> np.ndarray:
    """1D Gaussian smoothing via numpy convolution (no scipy dependency)."""
    if sigma <= 0:
        return signal.copy()
    r = max(1, int(3 * sigma + 0.5))
    x = np.arange(-r, r + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(signal.astype(float), kernel, mode="same").astype(np.float32)


def _wrist_speed(
    data: np.ndarray,   # (T, 1, 543, 3)
    conf: np.ndarray,   # (T, 1, 543, 1)
    body_idx: int,
    hand_idx: int,
) -> np.ndarray:
    """Compute per-frame wrist speed (T,), confidence-masked.

    Uses the dedicated hand-wrist landmark when its confidence > 0.3,
    otherwise falls back to the body-pose wrist landmark.
    Transitions where either endpoint has near-zero confidence are zeroed
    out to avoid phantom motion spikes when landmarks jump to (0,0).
    """
    T = data.shape[0]

    body_pos  = data[:, 0, body_idx, :2]   # (T, 2)  — use XY only
    body_conf = conf[:, 0, body_idx, 0]    # (T,)
    hand_pos  = data[:, 0, hand_idx, :2]   # (T, 2)
    hand_conf = conf[:, 0, hand_idx, 0]    # (T,)

    use_hand = hand_conf > 0.3
    pos = np.where(use_hand[:, np.newaxis], hand_pos, body_pos)
    c   = np.where(use_hand, hand_conf, body_conf)

    speed = np.zeros(T, dtype=np.float32)
    if T > 1:
        diff   = np.diff(pos, axis=0)                       # (T-1, 2)
        raw    = np.linalg.norm(diff, axis=1).astype(np.float32)
        c_mask = np.minimum(c[:-1], c[1:])
        raw   *= (c_mask > 0.2).astype(np.float32)
        speed[1:] = raw

    return speed


def compute_motion_energy(
    data: np.ndarray,
    conf: np.ndarray,
    hand: str,
    sigma: float = 2.0,
) -> np.ndarray:
    """Normalised wrist speed for the given hand, smoothed and 0–1 scaled.

    Args:
        data: Pose data array (T, 1, 543, 3).
        conf: Confidence array (T, 1, 543, 1).
        hand: ``"right"`` or ``"left"``.
        sigma: Gaussian smoothing sigma.

    Returns:
        1-D float32 array of length T, values in [0, 1].
    """
    body_idx, hand_idx = (16, 54) if hand == "right" else (15, 33)
    speed = _wrist_speed(data, conf, body_idx, hand_idx)
    smooth = _gaussian_smooth(speed, sigma)
    mx = smooth.max()
    if mx > 0:
        smooth /= mx
    return smooth


def _find_segments(
    is_active: np.ndarray,
    fps: float,
    min_frames: int,
    max_frames: int,
    gap_frames: int,
) -> list[tuple[int, int]]:
    """Extract (start_frame, end_frame) pairs from a binary active-motion mask."""
    if not is_active.any():
        return []

    padded = np.concatenate([[False], is_active, [False]])
    starts = np.where(~padded[:-1] &  padded[1:])[0]
    ends   = np.where( padded[:-1] & ~padded[1:])[0]
    segments = list(zip(starts.tolist(), ends.tolist()))

    # Merge segments separated by a short inactive gap
    merged = [segments[0]]
    for s, e in segments[1:]:
        if s - merged[-1][1] <= gap_frames:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    return [(s, e) for s, e in merged if min_frames <= (e - s) <= max_frames]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_sign_segments(
    data: np.ndarray,
    conf: np.ndarray,
    fps: float,
    smooth_sigma: float = 3.0,
    motion_threshold: float = 0.15,
    min_duration_ms: int = 150,
    max_duration_ms: int = 4000,
    min_gap_ms: int = 80,
) -> dict[str, list[tuple[int, int, float]]]:
    """Detect sign segment boundaries from wrist motion.

    Args:
        data:              Pose data (T, 1, 543, 3).
        conf:              Confidence array (T, 1, 543, 1).
        fps:               Video frame rate.
        smooth_sigma:      Gaussian kernel sigma in frames. Higher = smoother,
                           fewer short segments.
        motion_threshold:  Fraction of peak motion above which a frame is
                           considered active. Lower = more sensitive.
        min_duration_ms:   Minimum segment length in ms (shorter are discarded).
        max_duration_ms:   Maximum segment length in ms (longer are discarded).
        min_gap_ms:        Gaps shorter than this between segments are merged.

    Returns:
        {'right': [(start_ms, end_ms, confidence), ...],
         'left':  [(start_ms, end_ms, confidence), ...]}
    """
    min_f = max(1, int(min_duration_ms * fps / 1000))
    max_f = max(2, int(max_duration_ms * fps / 1000))
    gap_f = max(1, int(min_gap_ms      * fps / 1000))

    result: dict[str, list[tuple[int, int, float]]] = {}

    for side, body_idx, hand_idx in [
        ("right", _BODY_RIGHT_WRIST, _RIGHT_HAND_WRIST),
        ("left",  _BODY_LEFT_WRIST,  _LEFT_HAND_WRIST),
    ]:
        speed = _wrist_speed(data, conf, body_idx, hand_idx)
        speed = _gaussian_smooth(speed, smooth_sigma)

        p99 = float(np.percentile(speed, 99))
        if p99 < 1e-6:
            result[side] = []
            continue
        speed_norm = speed / p99

        is_active = speed_norm > motion_threshold
        segs      = _find_segments(is_active, fps, min_f, max_f, gap_f)

        result[side] = [
            (
                int(s * 1000 / fps),
                int(e * 1000 / fps),
                round(float(min(1.0, speed_norm[s:e].mean())), 3),
            )
            for s, e in segs
        ]

    return result


def preannotate_eaf(
    pose_path: Path,
    eaf_path: Path,
    overwrite: bool = False,
    **detection_params,
) -> dict:
    """Run motion segmentation and populate the EAF AI tiers.

    Args:
        pose_path:  .pose file produced by the pose extraction pipeline.
        eaf_path:   EAF file to update in-place.
        overwrite:  If True, clear existing AI-tier annotations first.
        **detection_params: Forwarded to detect_sign_segments().

    Returns:
        {'rh_segments': int, 'lh_segments': int, 'duration_sec': float}
    """
    from spj.eaf import AI_TIERS, load_eaf, save_eaf

    data, conf_arr, fps = load_pose_arrays(pose_path)
    T            = data.shape[0]
    duration_sec = round(T / fps, 1) if fps > 0 else 0.0

    segments = detect_sign_segments(data, conf_arr, fps, **detection_params)

    eaf        = load_eaf(eaf_path)
    tier_names = eaf.get_tier_names()

    if overwrite:
        for t in AI_TIERS:
            if t in tier_names:
                eaf.remove_tier(t)
                eaf.add_tier(t)

    for start_ms, end_ms, score in segments["right"]:
        try:
            eaf.add_annotation("AI_Gloss_RH",   start_ms, end_ms, value="?")
            eaf.add_annotation("AI_Confidence", start_ms, end_ms, value=str(score))
        except Exception as exc:
            logger.debug("RH skip %d–%d: %s", start_ms, end_ms, exc)

    for start_ms, end_ms, score in segments["left"]:
        try:
            eaf.add_annotation("AI_Gloss_LH", start_ms, end_ms, value="?")
        except Exception as exc:
            logger.debug("LH skip %d–%d: %s", start_ms, end_ms, exc)

    save_eaf(eaf, eaf_path)

    return {
        "rh_segments": len(segments["right"]),
        "lh_segments": len(segments["left"]),
        "duration_sec": duration_sec,
    }
