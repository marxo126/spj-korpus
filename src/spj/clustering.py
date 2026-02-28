"""Pose feature clustering — group similar sign segments using pretrained encoder.

Implements the SignCLIP concept on pose data: extract features from a pretrained
encoder, cluster with k-means or HDBSCAN, and let humans name clusters to create
pseudo-labels (80-90% less manual annotation work).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

INPUT_DIM = 1629  # 543 landmarks * 3 coords


# ---------------------------------------------------------------------------
# Config & Result
# ---------------------------------------------------------------------------

@dataclass
class ClusterConfig:
    """Clustering configuration."""
    n_clusters: int = 0         # 0 = auto (HDBSCAN), >0 = k-means
    min_cluster_size: int = 5   # for HDBSCAN
    batch_size: int = 256
    max_seq_len: int = 300


@dataclass
class ClusterResult:
    """Clustering output."""
    segment_ids: list[str]
    features: np.ndarray        # (N, d_model) pooled features
    labels: np.ndarray          # (N,) cluster assignments
    n_clusters: int
    centroids: np.ndarray | None  # (K, d_model) for k-means, None for HDBSCAN


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(
    encoder: torch.nn.Module,
    manifest_df: pd.DataFrame,
    npz_dir: Path,
    config: ClusterConfig,
    device: torch.device | None = None,
) -> tuple[list[str], np.ndarray]:
    """Run pretrained encoder on all manifest segments, return pooled features.

    Uses the encoder (without classifier head) to extract d_model-dimensional
    feature vectors via masked mean pooling.

    Args:
        encoder: PoseTransformerEncoder or MaskedPoseModel with encoder weights.
        manifest_df: DataFrame with 'segment_id' column.
        npz_dir: Directory containing .npz segment files.
        config: Clustering config (batch_size, max_seq_len).
        device: Torch device (inferred from encoder if None).

    Returns:
        (segment_ids, features_array) where features_array is (N, d_model).
    """
    if device is None:
        device = next(encoder.parameters()).device

    encoder.eval()
    npz_dir = Path(npz_dir)
    max_seq_len = config.max_seq_len

    segment_ids = []
    all_features = []

    # Collect all segments that have NPZ files
    valid_rows = []
    for _, row in manifest_df.iterrows():
        seg_id = str(row["segment_id"])
        npz_path = npz_dir / f"{seg_id}.npz"
        if not npz_path.exists():
            candidates = list(npz_dir.glob(f"*{seg_id}*.npz"))
            npz_path = candidates[0] if candidates else None
        if npz_path and npz_path.exists():
            valid_rows.append((seg_id, npz_path))

    logger.info("Extracting features from %d segments", len(valid_rows))

    # Process in batches
    batch_features_list = []
    batch_masks_list = []
    batch_ids = []

    for seg_id, npz_path in valid_rows:
        d = np.load(str(npz_path))
        pose = d["pose"].astype(np.float32)  # (T, 543, 3)
        T = pose.shape[0]
        features = pose.reshape(T, -1)  # (T, 1629)

        # Pad or truncate
        if T >= max_seq_len:
            features = features[:max_seq_len]
            mask = np.ones(max_seq_len, dtype=np.float32)
        else:
            pad_len = max_seq_len - T
            features = np.concatenate([
                features,
                np.zeros((pad_len, INPUT_DIM), dtype=np.float32),
            ], axis=0)
            mask = np.concatenate([
                np.ones(T, dtype=np.float32),
                np.zeros(pad_len, dtype=np.float32),
            ])

        batch_features_list.append(features)
        batch_masks_list.append(mask)
        batch_ids.append(seg_id)

        # Process batch when full
        if len(batch_features_list) >= config.batch_size:
            feats = _encode_batch(
                encoder, batch_features_list, batch_masks_list, device,
            )
            all_features.append(feats)
            segment_ids.extend(batch_ids)
            batch_features_list, batch_masks_list, batch_ids = [], [], []

    # Process remaining
    if batch_features_list:
        feats = _encode_batch(
            encoder, batch_features_list, batch_masks_list, device,
        )
        all_features.append(feats)
        segment_ids.extend(batch_ids)

    if not all_features:
        return [], np.array([])

    features_array = np.concatenate(all_features, axis=0)  # (N, d_model)
    logger.info("Extracted features: shape %s", features_array.shape)
    return segment_ids, features_array


def _encode_batch(
    encoder: torch.nn.Module,
    features_list: list[np.ndarray],
    masks_list: list[np.ndarray],
    device: torch.device,
) -> np.ndarray:
    """Encode a batch of segments and return pooled features."""
    features_t = torch.from_numpy(np.stack(features_list)).to(device)
    masks_t = torch.from_numpy(np.stack(masks_list)).to(device)

    with torch.no_grad():
        # Check if this is a MaskedPoseModel (has recon_head) or PoseTransformerEncoder
        if hasattr(encoder, "recon_head"):
            # MaskedPoseModel — get encoder output
            _, hidden = encoder(features_t, masks_t)
        elif hasattr(encoder, "classifier"):
            # PoseTransformerEncoder — run through encoder layers, skip classifier
            h = encoder.input_proj(features_t)
            h = encoder.pos_enc(h)
            h = encoder.dropout(h)
            key_padding_mask = (masks_t == 0)
            hidden = encoder.transformer(h, src_key_padding_mask=key_padding_mask)
        else:
            raise ValueError("Unknown encoder type — expected MaskedPoseModel or PoseTransformerEncoder")

        # Masked mean pooling
        mask_expanded = masks_t.unsqueeze(-1)  # (B, S, 1)
        pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

    return pooled.cpu().numpy()


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_segments(
    features: np.ndarray,
    config: ClusterConfig,
) -> tuple[np.ndarray, np.ndarray | None]:
    """K-means (if n_clusters > 0) or HDBSCAN (if n_clusters == 0).

    Args:
        features: (N, d_model) feature array.
        config: ClusterConfig with n_clusters and min_cluster_size.

    Returns:
        (labels, centroids_or_None) where labels is (N,) integer array.
    """
    if features.shape[0] == 0:
        return np.array([], dtype=int), None

    if config.n_clusters > 0:
        # K-means
        from sklearn.cluster import KMeans
        km = KMeans(
            n_clusters=min(config.n_clusters, features.shape[0]),
            random_state=42,
            n_init=10,
        )
        labels = km.fit_predict(features)
        logger.info("K-means: %d clusters", config.n_clusters)
        return labels, km.cluster_centers_
    else:
        # HDBSCAN (available in scikit-learn >= 1.3)
        from sklearn.cluster import HDBSCAN
        hdb = HDBSCAN(
            min_cluster_size=config.min_cluster_size,
            min_samples=2,
        )
        labels = hdb.fit_predict(features)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        logger.info("HDBSCAN: %d clusters (%d noise points)",
                     n_clusters, (labels == -1).sum())
        return labels, None


# ---------------------------------------------------------------------------
# Cluster analysis
# ---------------------------------------------------------------------------

def cluster_summary(
    result: ClusterResult,
    manifest_df: pd.DataFrame,
) -> pd.DataFrame:
    """Per-cluster summary: size, sample texts, representative segment_ids.

    Returns a DataFrame with columns:
        cluster_id, size, sample_texts, representative_ids
    """
    # Build segment_id -> label mapping from manifest
    if "label" not in manifest_df.columns:
        manifest_df = manifest_df.copy()
        manifest_df["label"] = manifest_df["reviewed_text"].where(
            manifest_df["reviewed_text"].str.strip() != "",
            manifest_df.get("text", ""),
        )

    seg_to_label = {}
    for _, row in manifest_df.iterrows():
        seg_to_label[str(row["segment_id"])] = str(row.get("label", ""))

    summaries = []
    unique_labels = sorted(set(result.labels))

    for cluster_id in unique_labels:
        mask = result.labels == cluster_id
        indices = np.where(mask)[0]
        size = int(mask.sum())

        # Gather texts for this cluster
        texts = []
        rep_ids = []
        for idx in indices[:10]:  # sample up to 10
            seg_id = result.segment_ids[idx]
            rep_ids.append(seg_id)
            label = seg_to_label.get(seg_id, "")
            if label:
                texts.append(label)

        # Most common text in cluster
        text_counts = {}
        for t in texts:
            text_counts[t] = text_counts.get(t, 0) + 1
        sorted_texts = sorted(text_counts.items(), key=lambda x: -x[1])
        sample_texts = ", ".join(f"{t}({c})" for t, c in sorted_texts[:5])

        summaries.append({
            "cluster_id": int(cluster_id),
            "size": size,
            "sample_texts": sample_texts,
            "representative_ids": ", ".join(rep_ids[:5]),
        })

    return pd.DataFrame(summaries)


def apply_cluster_labels(
    result: ClusterResult,
    cluster_names: dict[int, str],
    manifest_df: pd.DataFrame,
) -> pd.DataFrame:
    """Write cluster_names as pseudo-labels into manifest DataFrame.

    Args:
        result: ClusterResult with segment_ids and labels.
        cluster_names: {cluster_id: "GLOSS_NAME"} mapping from user.
        manifest_df: Original manifest DataFrame.

    Returns:
        Updated manifest with 'label' column filled from cluster names.
        Only updates rows where the cluster has a name assigned.
    """
    df = manifest_df.copy()

    # Ensure label column exists
    if "label" not in df.columns:
        df["label"] = df["reviewed_text"].where(
            df["reviewed_text"].str.strip() != "",
            df.get("text", ""),
        )

    # Build segment_id -> cluster label mapping
    seg_to_cluster_label = {}
    for seg_id, cluster_id in zip(result.segment_ids, result.labels):
        if int(cluster_id) in cluster_names:
            name = cluster_names[int(cluster_id)]
            if name.strip():
                seg_to_cluster_label[seg_id] = name

    # Apply to manifest
    updated = 0
    for idx, row in df.iterrows():
        seg_id = str(row["segment_id"])
        if seg_id in seg_to_cluster_label:
            df.at[idx, "label"] = seg_to_cluster_label[seg_id]
            updated += 1

    logger.info("Applied cluster labels to %d/%d segments", updated, len(df))
    return df


def reduce_dimensions(
    features: np.ndarray,
    method: str = "tsne",
    n_components: int = 2,
) -> np.ndarray:
    """Reduce features to 2D for visualisation using t-SNE or PCA.

    Args:
        features: (N, d_model) feature array.
        method: "tsne" or "pca".
        n_components: Output dimensions (default 2).

    Returns:
        (N, n_components) reduced features.
    """
    if features.shape[0] == 0:
        return features

    if method == "tsne":
        from sklearn.manifold import TSNE
        perplexity = min(30, max(2, features.shape[0] - 1))
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    else:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components, random_state=42)

    return reducer.fit_transform(features)
