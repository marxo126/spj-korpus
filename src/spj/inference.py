"""Inference pipeline: gloss prepartner-dictn, EAF annotation, and timeline visualisation.

Takes a trained PoseTransformerEncoder, runs it on detected sign segments from
new videos, writes prepartner-dictns to EAF AI tiers, and provides timeline visualisation.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def predict_segments(
    model: "PoseTransformerEncoder",
    label_encoder: "LabelEncoder",
    segments: dict[str, list[tuple[int, int, float]]],
    pose_data: np.ndarray,   # (T, 1, 543, 3)
    fps: float,
    max_seq_len: int = 300,
    device: Optional[torch.device] = None,
) -> list[dict]:
    """Classify each detected sign segment using the trained model.

    Args:
        model: Trained PoseTransformerEncoder in eval mode.
        label_encoder: Maps indices to gloss labels.
        segments: Output of detect_sign_segments() — {'right': [...], 'left': [...]}.
        pose_data: Full pose array (T, 1, 543, 3).
        fps: Video frame rate.
        max_seq_len: Maximum sequence length the model expects.
        device: Torch device (inferred from model if None).

    Returns:
        List of dicts with keys:
            hand, start_ms, end_ms, motion_confidence,
            predicted_gloss, prepartner-dictn_confidence, top3_glosses
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    prepartner-dictns = []

    for hand in ("right", "left"):
        for start_ms, end_ms, motion_conf in segments.get(hand, []):
            frame_start = min(int(start_ms * fps / 1000), pose_data.shape[0] - 1)
            frame_end = min(int(end_ms * fps / 1000), pose_data.shape[0] - 1)

            if frame_end <= frame_start:
                continue

            # Extract segment pose
            segment_pose = pose_data[frame_start:frame_end, 0, :, :]  # (T_seg, 543, 3)
            T_seg = segment_pose.shape[0]

            # Flatten to (T_seg, 1629)
            features = segment_pose.reshape(T_seg, -1).astype(np.float32)

            # Pad or truncate
            if T_seg >= max_seq_len:
                features = features[:max_seq_len]
                mask = np.ones(max_seq_len, dtype=np.float32)
            else:
                pad_len = max_seq_len - T_seg
                features = np.concatenate([
                    features,
                    np.zeros((pad_len, features.shape[1]), dtype=np.float32),
                ], axis=0)
                mask = np.concatenate([
                    np.ones(T_seg, dtype=np.float32),
                    np.zeros(pad_len, dtype=np.float32),
                ])

            # Forward pass
            features_t = torch.from_numpy(features).unsqueeze(0).to(device)
            mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(features_t, mask_t)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            pred_idx = int(np.argmax(probs))
            pred_conf = float(probs[pred_idx])

            # Top-3
            top3_indices = np.argsort(probs)[-3:][::-1]
            top3 = [
                {"gloss": label_encoder.decode(int(i)), "confidence": round(float(probs[i]), 4)}
                for i in top3_indices
            ]

            prepartner-dictns.append({
                "hand": hand,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "motion_confidence": motion_conf,
                "predicted_gloss": label_encoder.decode(pred_idx),
                "prepartner-dictn_confidence": round(pred_conf, 4),
                "top3_glosses": top3,
            })

    # Sort by start time
    prepartner-dictns.sort(key=lambda p: p["start_ms"])
    return prepartner-dictns


def write_prepartner-dictns_to_eaf(
    prepartner-dictns: list[dict],
    eaf_path: Path,
    overwrite: bool = True,
) -> dict:
    """Write predicted glosses to EAF AI tiers.

    Populates AI_Gloss_RH, AI_Gloss_LH with predicted glosses,
    and AI_Confidence with prepartner-dictn confidence scores.

    Args:
        prepartner-dictns: Output of predict_segments().
        eaf_path: Path to EAF file to update.
        overwrite: If True, clear existing AI annotations first.

    Returns:
        {'rh_prepartner-dictns': int, 'lh_prepartner-dictns': int}
    """
    from spj.eaf import AI_TIERS, load_eaf, save_eaf

    eaf = load_eaf(eaf_path)
    tier_names = eaf.get_tier_names()

    if overwrite:
        for t in AI_TIERS:
            if t in tier_names:
                eaf.remove_tier(t)
                eaf.add_tier(t)

    rh_count = 0
    lh_count = 0

    for pred in prepartner-dictns:
        gloss = pred["predicted_gloss"]
        start_ms = pred["start_ms"]
        end_ms = pred["end_ms"]
        conf = str(round(pred["prepartner-dictn_confidence"], 3))
        hand = pred["hand"]

        tier = "AI_Gloss_RH" if hand == "right" else "AI_Gloss_LH"

        try:
            eaf.add_annotation(tier, start_ms, end_ms, value=gloss)
            eaf.add_annotation("AI_Confidence", start_ms, end_ms, value=conf)
            if hand == "right":
                rh_count += 1
            else:
                lh_count += 1
        except Exception as exc:
            logger.debug("Skip annotation %s %d-%d: %s", tier, start_ms, end_ms, exc)

    save_eaf(eaf, eaf_path)
    return {"rh_prepartner-dictns": rh_count, "lh_prepartner-dictns": lh_count}


def prepartner-dictns_timeline_figure(
    prepartner-dictns: list[dict],
    duration_sec: float,
) -> "go.Figure":
    """Plotly timeline of predicted glosses, color-coded by confidence.

    Args:
        prepartner-dictns: Output of predict_segments().
        duration_sec: Total video duration in seconds.

    Returns:
        Plotly Figure with horizontal bars for each prepartner-dictn.
    """
    import plotly.graph_objects as go

    if not prepartner-dictns:
        fig = go.Figure()
        fig.update_layout(title="No prepartner-dictns", height=200)
        return fig

    # Separate by hand
    rh_preds = [p for p in prepartner-dictns if p["hand"] == "right"]
    lh_preds = [p for p in prepartner-dictns if p["hand"] == "left"]

    fig = go.Figure()

    for preds, hand_label, y_offset, color in [
        (rh_preds, "Right Hand", 1, "crimson"),
        (lh_preds, "Left Hand", 0, "limegreen"),
    ]:
        for p in preds:
            start_s = p["start_ms"] / 1000
            end_s = p["end_ms"] / 1000
            conf = p["prepartner-dictn_confidence"]
            opacity = max(0.3, min(1.0, conf))

            fig.add_trace(go.Bar(
                x=[end_s - start_s],
                y=[hand_label],
                base=[start_s],
                orientation="h",
                marker=dict(color=color, opacity=opacity),
                text=f"{p['predicted_gloss']} ({conf:.0%})",
                textposition="inside",
                hovertemplate=(
                    f"<b>{p['predicted_gloss']}</b><br>"
                    f"Confidence: {conf:.1%}<br>"
                    f"{start_s:.2f}s – {end_s:.2f}s<br>"
                    f"<extra></extra>"
                ),
                showlegend=False,
            ))

    fig.update_layout(
        title="Predicted Glosses Timeline",
        xaxis_title="Time (s)",
        xaxis=dict(range=[0, duration_sec]),
        yaxis=dict(categoryorder="array", categoryarray=["Left Hand", "Right Hand"]),
        height=250,
        barmode="overlay",
        margin=dict(l=10, r=10, t=40, b=40),
    )
    return fig
