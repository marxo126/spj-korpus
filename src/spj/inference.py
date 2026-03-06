"""Inference pipeline: gloss prepartner-dictn, EAF annotation, and timeline visualisation.

Takes a trained PoseTransformerEncoder, runs it on detected sign segments from
new videos, writes prepartner-dictns to EAF AI tiers, and provides timeline visualisation.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

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
    landmark_indices: Optional[list[int]] = None,
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
        landmark_indices: If set, select only these landmarks before inference.
            Use SL_LANDMARK_INDICES from training_data to match training format.

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

            # Filter to SL-relevant landmarks if specified
            if landmark_indices is not None:
                segment_pose = segment_pose[:, landmark_indices, :]  # (T_seg, N, 3)

            T_seg = segment_pose.shape[0]

            # Flatten to (T_seg, N*3)
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


def predict_batch(
    model: "PoseTransformerEncoder",
    label_encoder: "LabelEncoder",
    pose_dir: Path,
    annotations_dir: Path,
    inventory_df: "pd.DataFrame",
    max_seq_len: int = 300,
    device: "Optional[torch.device]" = None,
    landmark_indices: "Optional[list[int]]" = None,
    min_pose_bytes: int = 10_000,
    progress_callback: "Optional[Callable[[int, int, str], None]]" = None,
) -> dict:
    """Run inference on all videos in the inventory.

    Args:
        model: Trained PoseTransformerEncoder in eval mode.
        label_encoder: Maps indices to gloss labels.
        pose_dir: Directory containing .pose files.
        annotations_dir: Directory for EAF output files.
        inventory_df: Inventory DataFrame with 'path' column.
        max_seq_len: Maximum sequence length the model expects.
        device: Torch device (inferred from model if None).
        landmark_indices: Landmark filter indices for SL-relevant subset.
        min_pose_bytes: Skip pose files smaller than this (corrupted).
        progress_callback: Called as progress_callback(idx, total, video_name).

    Returns:
        Dict with n_processed, n_skipped, n_errors, total_prepartner-dictns, results.
    """
    import pandas as pd
    from spj.preannotate import load_pose_arrays, detect_sign_segments
    from spj.eaf import create_empty_eaf, save_eaf

    pose_dir = Path(pose_dir)
    annotations_dir = Path(annotations_dir)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    results = []
    n_processed = 0
    n_skipped = 0
    n_errors = 0
    total_prepartner-dictns = 0
    total = len(inventory_df)

    for idx, (_, row) in enumerate(inventory_df.iterrows()):
        video_path = Path(str(row["path"]))
        stem = video_path.stem
        pose_path = pose_dir / f"{stem}.pose"

        if progress_callback:
            progress_callback(idx, total, stem)

        # Skip if no pose file
        if not pose_path.exists():
            results.append({"video": stem, "status": "skipped", "reason": "No pose file"})
            n_skipped += 1
            continue

        # Skip corrupted (small) pose files
        try:
            pose_size = pose_path.stat().st_size
        except OSError:
            results.append({"video": stem, "status": "skipped", "reason": "Cannot stat pose file"})
            n_skipped += 1
            continue

        if pose_size < min_pose_bytes:
            results.append({
                "video": stem, "status": "skipped",
                "reason": f"Pose file too small ({pose_size} bytes)",
            })
            n_skipped += 1
            continue

        try:
            pose_data, conf_data, fps = load_pose_arrays(pose_path)
            segments = detect_sign_segments(pose_data, conf_data, fps)
            n_segs = len(segments.get("right", [])) + len(segments.get("left", []))

            if n_segs == 0:
                results.append({"video": stem, "status": "ok", "n_prepartner-dictns": 0})
                n_processed += 1
                continue

            prepartner-dictns = predict_segments(
                model, label_encoder, segments, pose_data, fps,
                max_seq_len=max_seq_len, device=device,
                landmark_indices=landmark_indices,
            )

            # Ensure EAF exists
            eaf_path = annotations_dir / f"{stem}.eaf"
            if not eaf_path.exists():
                eaf = create_empty_eaf(video_path, eaf_path)
                save_eaf(eaf, eaf_path)

            write_result = write_prepartner-dictns_to_eaf(prepartner-dictns, eaf_path, overwrite=True)

            results.append({
                "video": stem,
                "status": "ok",
                "n_prepartner-dictns": len(prepartner-dictns),
                "rh": write_result.get("rh_prepartner-dictns", 0),
                "lh": write_result.get("lh_prepartner-dictns", 0),
            })
            total_prepartner-dictns += len(prepartner-dictns)
            n_processed += 1

        except Exception as exc:
            logger.warning("Inference failed for %s: %s", stem, exc)
            results.append({"video": stem, "status": "error", "message": str(exc)})
            n_errors += 1

    return {
        "n_processed": n_processed,
        "n_skipped": n_skipped,
        "n_errors": n_errors,
        "total_prepartner-dictns": total_prepartner-dictns,
        "results": results,
    }


def read_prepartner-dictns_from_eaf(eaf_path: Path) -> list[dict]:
    """Read AI-predicted glosses back from an EAF file.

    Reads AI_Gloss_RH → hand="right", AI_Gloss_LH → hand="left",
    and matches AI_Confidence annotations by timestamp overlap.

    Args:
        eaf_path: Path to the .eaf file.

    Returns:
        List of dicts sorted by start_ms, same schema as predict_segments():
            {hand, start_ms, end_ms, predicted_gloss, prepartner-dictn_confidence}
    """
    from spj.eaf import AI_TIERS, load_eaf

    eaf = load_eaf(eaf_path)
    tier_names = eaf.get_tier_names()

    # Read confidence annotations into O(1) lookup dict
    conf_exact: dict[tuple[int, int], float] = {}
    conf_by_start: dict[int, tuple[int, float]] = {}  # start_ms → (end_ms, value)
    if AI_TIERS[2] in tier_names:  # "AI_Confidence"
        for start, end, value, *_ in eaf.get_annotation_data_for_tier(AI_TIERS[2]):
            try:
                cv = float(value)
            except (ValueError, TypeError):
                cv = 0.0
            cs, ce = int(start), int(end)
            conf_exact[(cs, ce)] = cv
            conf_by_start[cs] = (ce, cv)

    def _find_confidence(start_ms: int, end_ms: int) -> float:
        """Find confidence — exact match first, then start-time fallback."""
        exact = conf_exact.get((start_ms, end_ms))
        if exact is not None:
            return exact
        nearby = conf_by_start.get(start_ms)
        if nearby:
            return nearby[1]
        return 0.0

    tier_hand_map = {AI_TIERS[0]: "right", AI_TIERS[1]: "left"}  # AI_Gloss_RH, AI_Gloss_LH
    prepartner-dictns: list[dict] = []

    for tier, hand in tier_hand_map.items():
        if tier not in tier_names:
            continue
        for start, end, value, *_ in eaf.get_annotation_data_for_tier(tier):
            gloss = str(value).strip() if value else ""
            if not gloss:
                continue
            start_ms = int(start)
            end_ms = int(end)
            prepartner-dictns.append({
                "hand": hand,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "predicted_gloss": gloss,
                "prepartner-dictn_confidence": _find_confidence(start_ms, end_ms),
            })

    prepartner-dictns.sort(key=lambda p: p["start_ms"])
    return prepartner-dictns


# ---------------------------------------------------------------------------
# Embedding index — pose similarity search
# ---------------------------------------------------------------------------

def build_embedding_index(
    model: "PoseTransformerEncoder",
    manifest_df: "pd.DataFrame",
    npz_dir: Path,
    max_seq_len: int = 300,
    device: Optional[torch.device] = None,
    batch_size: int = 256,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """Extract embeddings for all NPZ segments and build a similarity index.

    Args:
        model: Trained PoseTransformerEncoder (uses .encode() method).
        manifest_df: DataFrame with 'npz_file' and 'label' columns.
        npz_dir: Directory containing .npz segment files.
        max_seq_len: Padding/truncation length.
        device: Torch device.
        batch_size: Batch size for encoding.
        progress_callback: Called as progress_callback(done, total).

    Returns:
        Dict with:
            embeddings: np.ndarray (N, d_model) — L2-normalized
            labels: list[str] — label per segment
            npz_files: list[str] — npz filename per segment
    """
    import pandas as pd

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    npz_dir = Path(npz_dir)

    # Derive label column
    df = manifest_df.copy()
    if "label" not in df.columns:
        if "reviewed_text" in df.columns:
            df["label"] = df["reviewed_text"].where(
                df["reviewed_text"].astype(str).str.strip() != "", df["text"]
            )
        elif "text" in df.columns:
            df["label"] = df["text"]

    # Resolve npz file column (manifest may use npz_path or npz_file)
    npz_col = "npz_file"
    if npz_col not in df.columns and "npz_path" in df.columns:
        df[npz_col] = df["npz_path"].apply(lambda p: Path(p).name)

    # Filter valid rows
    valid = df.dropna(subset=[npz_col, "label"])
    valid = valid[valid["label"].astype(str).str.strip() != ""]

    all_embeddings = []
    all_labels = []
    all_files = []
    total = len(valid)

    for batch_start in range(0, total, batch_size):
        batch_rows = valid.iloc[batch_start:batch_start + batch_size]
        features_list = []
        masks_list = []
        batch_labels = []
        batch_files = []

        for _, row in batch_rows.iterrows():
            npz_name = str(row[npz_col])
            npz_path = npz_dir / npz_name
            if not npz_path.exists():
                continue

            data = np.load(str(npz_path))
            pose = data["pose"]  # (T, N_landmarks, 3)
            T = pose.shape[0]
            feat = pose.reshape(T, -1).astype(np.float32)

            if T >= max_seq_len:
                feat = feat[:max_seq_len]
                mask = np.ones(max_seq_len, dtype=np.float32)
            else:
                pad_len = max_seq_len - T
                feat = np.concatenate([feat, np.zeros((pad_len, feat.shape[1]), dtype=np.float32)])
                mask = np.concatenate([np.ones(T, dtype=np.float32), np.zeros(pad_len, dtype=np.float32)])

            features_list.append(feat)
            masks_list.append(mask)
            label = str(row["label"]).strip()
            batch_labels.append(label)
            batch_files.append(npz_name)

        if not features_list:
            continue

        features_t = torch.from_numpy(np.stack(features_list)).to(device)
        masks_t = torch.from_numpy(np.stack(masks_list)).to(device)

        with torch.no_grad():
            emb = model.encode(features_t, masks_t)  # (B, d_model)
            emb = emb.cpu().numpy()

        all_embeddings.append(emb)
        all_labels.extend(batch_labels)
        all_files.extend(batch_files)

        if progress_callback:
            progress_callback(min(batch_start + batch_size, total), total)

    if not all_embeddings:
        return {"embeddings": np.empty((0, 0)), "labels": [], "npz_files": []}

    embeddings = np.concatenate(all_embeddings, axis=0)
    # L2 normalize for cosine similarity via dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    embeddings = embeddings / norms

    return {
        "embeddings": embeddings,
        "labels": all_labels,
        "npz_files": all_files,
    }


def save_embedding_index(index: dict, path: Path) -> None:
    """Save embedding index to .npz file."""
    np.savez_compressed(
        str(path),
        embeddings=index["embeddings"],
        labels=np.array(index["labels"], dtype=object),
        npz_files=np.array(index["npz_files"], dtype=object),
    )
    logger.info("Saved embedding index (%d entries) to %s", len(index["labels"]), path)


def load_embedding_index(path: Path) -> dict:
    """Load embedding index from .npz file."""
    data = np.load(str(path), allow_pickle=True)
    return {
        "embeddings": data["embeddings"],
        "labels": data["labels"].tolist(),
        "npz_files": data["npz_files"].tolist(),
    }


def find_similar_signs(
    model: "PoseTransformerEncoder",
    query_pose: np.ndarray,
    index: dict,
    max_seq_len: int = 300,
    device: Optional[torch.device] = None,
    top_k: int = 5,
) -> list[dict]:
    """Find top-K most similar signs from the embedding index.

    Args:
        model: Same model used to build the index.
        query_pose: Pose array (T, N_landmarks, 3) — already landmark-filtered.
        index: Output of build_embedding_index() or load_embedding_index().
        max_seq_len: Padding/truncation length.
        device: Torch device.
        top_k: Number of results to return.

    Returns:
        List of dicts: [{label, similarity, npz_file}, ...]
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    embeddings = index["embeddings"]
    if embeddings.shape[0] == 0:
        return []

    # Prepare query
    T = query_pose.shape[0]
    feat = query_pose.reshape(T, -1).astype(np.float32)

    if T >= max_seq_len:
        feat = feat[:max_seq_len]
        mask = np.ones(max_seq_len, dtype=np.float32)
    else:
        pad_len = max_seq_len - T
        feat = np.concatenate([feat, np.zeros((pad_len, feat.shape[1]), dtype=np.float32)])
        mask = np.concatenate([np.ones(T, dtype=np.float32), np.zeros(pad_len, dtype=np.float32)])

    feat_t = torch.from_numpy(feat).unsqueeze(0).to(device)
    mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)

    with torch.no_grad():
        q_emb = model.encode(feat_t, mask_t).cpu().numpy()[0]

    # L2 normalize query
    q_norm = np.linalg.norm(q_emb)
    if q_norm > 1e-8:
        q_emb = q_emb / q_norm

    # Cosine similarity = dot product (both normalized)
    similarities = embeddings @ q_emb  # (N,)

    # Top-K (deduplicate by label — show each label only once)
    sorted_indices = np.argsort(similarities)[::-1]
    results = []
    seen_labels = set()
    for idx in sorted_indices:
        label = index["labels"][idx]
        if label in seen_labels:
            continue
        seen_labels.add(label)
        results.append({
            "label": label,
            "similarity": round(float(similarities[idx]), 4),
            "npz_file": index["npz_files"][idx],
        })
        if len(results) >= top_k:
            break

    return results


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
