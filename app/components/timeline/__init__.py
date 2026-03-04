"""SPJ Timeline — bidirectional Streamlit component for AI Review.

Combines video player, pose canvas, and interactive prediction timeline
into a single component. Returns user actions (trim changes, prediction
selection) back to Python via setComponentValue.
"""

from pathlib import Path

import streamlit.components.v1 as components

_COMPONENT_DIR = Path(__file__).parent
_component_func = components.declare_component("spj_timeline", path=str(_COMPONENT_DIR))


def spj_timeline(
    *,
    video_b64: str,
    pos_b64: str,
    conf_b64: str,
    n_frames: int,
    fps: float,
    duration_ms: int,
    predictions: list[dict],
    motion_energy: list[float],
    energy_fps: float,
    current_pred_idx: int,
    connections: dict[str, list],
    segment_start_ms: int,
    segment_end_ms: int,
    key: str = "spj_timeline",
    height: int = 520,
) -> dict | None:
    """Render the unified timeline component.

    Args:
        connections: Dict from ``CONNECTION_ARRAYS`` — keys: body_conn,
            hand_conn, face_lips, face_left_eye, face_right_eye,
            face_left_brow, face_right_brow, face_oval.

    Returns dict with keys:
        - trim_start_ms: int
        - trim_end_ms: int
        - selected_pred_idx: int
    or None if no interaction yet.
    """
    result = _component_func(
        video_b64=video_b64,
        pos_b64=pos_b64,
        conf_b64=conf_b64,
        n_frames=n_frames,
        fps=fps,
        duration_ms=duration_ms,
        predictions=predictions,
        motion_energy=motion_energy,
        energy_fps=energy_fps,
        current_pred_idx=current_pred_idx,
        segment_start_ms=segment_start_ms,
        segment_end_ms=segment_end_ms,
        **connections,
        key=key,
        default=None,
        height=height,
    )
    return result
