"""Compare training methods on the same data to find the best approach.

Tests multiple architectural improvements found in research:
1. Baseline — current PoseTransformerEncoder
2. + Velocity features (dx, dx2 from Kaggle 1st place)
3. + Drop z (xy only, Kaggle 1st place)
4. + Nose normalization (subtract nose, divide by std)
5. Conv1D + Transformer (Kaggle 1st place architecture)
6. Combined: velocity + normalize + Conv1D
7. 4-stream GCN-style (Joint + Bone + JointMotion + BoneMotion from AUTSL SAM-SLR)

Each method trains for a short run (20 epochs) on the same splits.
Results saved to data/method_comparison.json.

Usage:
    .venv/bin/python tools/method_comparison.py
    .venv/bin/python tools/method_comparison.py --epochs 30
    .venv/bin/python tools/method_comparison.py --methods baseline velocity conv1d
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
NPZ_DIR = DATA_DIR / "training" / "export"
SPLITS_DIR = DATA_DIR / "training" / "splits"

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

# Compact preset layout: [nose, L_shoulder, R_shoulder, L_elbow, R_elbow, L_wrist, R_wrist,
#                          LH(21), RH(21), face_lips+nose(47)]
NOSE_IDX = 0  # First landmark in compact preset = nose
BODY_SLICE = slice(0, 7)
LH_SLICE = slice(7, 28)
RH_SLICE = slice(28, 49)
FACE_SLICE = slice(49, 96)

# Bone connections for compact 96 landmarks (parent → child)
# Body: nose→shoulders, shoulders→elbows, elbows→wrists
BONE_PAIRS_BODY = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]
# Hands: wrist→fingers (standard 21-point hand topology)
_HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index
    (0, 9), (9, 10), (10, 11), (11, 12),  # middle
    (0, 13), (13, 14), (14, 15), (15, 16),# ring
    (0, 17), (17, 18), (18, 19), (19, 20),# pinky
]
BONE_PAIRS_LH = [(a + 7, b + 7) for a, b in _HAND_BONES]
BONE_PAIRS_RH = [(a + 28, b + 28) for a, b in _HAND_BONES]
ALL_BONE_PAIRS = BONE_PAIRS_BODY + BONE_PAIRS_LH + BONE_PAIRS_RH


class LabelEncoder:
    def __init__(self, labels: list[str] | None = None):
        self.label_to_idx: dict[str, int] = {}
        self.idx_to_label: dict[int, str] = {}
        if labels:
            self.fit(labels)

    def fit(self, labels: list[str]) -> "LabelEncoder":
        unique = sorted(set(labels))
        self.label_to_idx = {lab: i for i, lab in enumerate(unique)}
        self.idx_to_label = {i: lab for lab, i in self.label_to_idx.items()}
        return self

    def encode(self, label: str) -> int:
        return self.label_to_idx[label]

    @property
    def n_classes(self) -> int:
        return len(self.label_to_idx)


def load_splits(min_samples: int = 3):
    """Load train/val splits. Filter to labels with min_samples in train."""
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "val.csv")

    # Derive label column
    for df in [train_df, val_df]:
        if "label" not in df.columns:
            df["label"] = df["reviewed_text"].where(
                df["reviewed_text"].str.strip() != "", df["text"]
            )

    # Filter to labels with enough training samples
    counts = train_df["label"].value_counts()
    valid_labels = set(counts[counts >= min_samples].index)
    train_df = train_df[train_df["label"].isin(valid_labels)].reset_index(drop=True)
    val_df = val_df[val_df["label"].isin(valid_labels)].reset_index(drop=True)

    logger.info("Loaded %d train, %d val samples, %d classes (min %d samples)",
                len(train_df), len(val_df), len(valid_labels), min_samples)
    return train_df, val_df


# ---------------------------------------------------------------------------
# Feature transforms — applied at dataset level
# ---------------------------------------------------------------------------

def compute_velocity(pose: np.ndarray) -> np.ndarray:
    """Compute first-order and second-order velocity features.
    Input: (T, N, C) → Output: (T, N, C*3) = [pos, dx, dx2]
    """
    T = pose.shape[0]
    # dx: frame[t+1] - frame[t], padded at end
    dx = np.zeros_like(pose)
    if T > 1:
        dx[:-1] = pose[1:] - pose[:-1]
    # dx2: frame[t+2] - frame[t], padded at end
    dx2 = np.zeros_like(pose)
    if T > 2:
        dx2[:-2] = pose[2:] - pose[:-2]
    return np.concatenate([pose, dx, dx2], axis=-1)  # (T, N, C*3)


def drop_z(pose: np.ndarray) -> np.ndarray:
    """Keep only x,y coordinates. Input: (T, N, 3) → Output: (T, N, 2)"""
    return pose[:, :, :2]


def normalize_nose(pose: np.ndarray) -> np.ndarray:
    """Subtract nose position and divide by std (per-sample normalization)."""
    pose = pose.copy()
    # Nose is landmark 0
    nose = pose[:, NOSE_IDX:NOSE_IDX+1, :]  # (T, 1, C)
    # Replace NaN nose with 0.5 (center)
    nose_mean = np.nanmean(nose, axis=0, keepdims=True)  # (1, 1, C)
    nose_mean = np.where(np.isnan(nose_mean), 0.5, nose_mean)
    pose = pose - nose_mean
    std = np.nanstd(pose)
    if std > 1e-6:
        pose = pose / std
    return pose


def compute_bones(pose: np.ndarray) -> np.ndarray:
    """Compute bone vectors (child - parent) for all connected pairs.
    Input: (T, 96, C) → Output: (T, n_bones, C)
    """
    bones = np.zeros((pose.shape[0], len(ALL_BONE_PAIRS), pose.shape[2]), dtype=pose.dtype)
    for i, (parent, child) in enumerate(ALL_BONE_PAIRS):
        bones[:, i, :] = pose[:, child, :] - pose[:, parent, :]
    return bones


# ---------------------------------------------------------------------------
# Datasets for each method
# ---------------------------------------------------------------------------

def _pad_or_truncate(features: np.ndarray, max_seq_len: int):
    T = features.shape[0]
    if T >= max_seq_len:
        return features[:max_seq_len].astype(np.float32), np.ones(max_seq_len, dtype=np.float32)
    pad_len = max_seq_len - T
    padded = np.concatenate([features, np.zeros((pad_len, features.shape[1]), dtype=np.float32)])
    mask = np.concatenate([np.ones(T, dtype=np.float32), np.zeros(pad_len, dtype=np.float32)])
    return padded, mask


class MethodDataset(Dataset):
    """Generic dataset that applies a feature transform to pose data."""

    def __init__(self, manifest_df, label_encoder, max_seq_len, transform_fn):
        self.max_seq_len = max_seq_len
        self.label_encoder = label_encoder
        self.transform_fn = transform_fn
        self.data: list[tuple[np.ndarray, int]] = []

        for _, row in manifest_df.iterrows():
            label = str(row.get("label", "")).strip()
            if not label or label not in label_encoder.label_to_idx:
                continue
            npz_path = row.get("npz_path", "")
            if not npz_path or not Path(npz_path).exists():
                seg_id = str(row.get("segment_id", ""))
                npz_path = str(NPZ_DIR / f"{seg_id}.npz")
            if not Path(npz_path).exists():
                continue
            d = np.load(npz_path)
            pose = d["pose"].astype(np.float32)  # (T, 96, 3)
            self.data.append((pose, label_encoder.encode(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pose, label_idx = self.data[idx]
        features = self.transform_fn(pose)  # (T, F)
        features, mask = _pad_or_truncate(features, self.max_seq_len)
        return {
            "features": torch.from_numpy(features),
            "mask": torch.from_numpy(mask),
            "label": torch.tensor(label_idx, dtype=torch.long),
        }


# Transform functions — each takes (T, 96, 3) and returns (T, F) flat features

def transform_baseline(pose):
    """Baseline: flatten (T, 96, 3) → (T, 288)"""
    return pose.reshape(pose.shape[0], -1)

def transform_velocity(pose):
    """Add velocity: (T, 96, 3) → velocity → (T, 96, 9) → (T, 864)"""
    v = compute_velocity(pose)  # (T, 96, 9)
    return v.reshape(v.shape[0], -1)

def transform_xy_only(pose):
    """Drop z: (T, 96, 3) → (T, 96, 2) → (T, 192)"""
    xy = drop_z(pose)
    return xy.reshape(xy.shape[0], -1)

def transform_normalize(pose):
    """Nose normalization: (T, 96, 3) → normalize → (T, 288)"""
    n = normalize_nose(pose)
    return n.reshape(n.shape[0], -1)

def transform_xy_velocity(pose):
    """Drop z + velocity: (T, 96, 2) + dx + dx2 → (T, 96, 6) → (T, 576)"""
    xy = drop_z(pose)  # (T, 96, 2)
    v = compute_velocity(xy)  # (T, 96, 6)
    return v.reshape(v.shape[0], -1)

def transform_norm_xy_velocity(pose):
    """Normalize + drop z + velocity: best combination from papers."""
    n = normalize_nose(pose)
    xy = drop_z(n)  # (T, 96, 2)
    v = compute_velocity(xy)  # (T, 96, 6)
    return v.reshape(v.shape[0], -1)

def transform_4stream(pose):
    """4-stream: Joint(xy) + Bone(xy) + JointMotion(xy) + BoneMotion(xy).
    Approximates SAM-SLR's multi-stream approach in a single flat input.
    """
    n = normalize_nose(pose)
    xy = drop_z(n)  # (T, 96, 2)

    # Joint stream = positions
    joint = xy  # (T, 96, 2)

    # Bone stream = vectors between connected joints
    bones = compute_bones(n)[:, :, :2]  # (T, n_bones, 2)

    # Joint motion = velocity
    joint_motion = np.zeros_like(xy)
    if xy.shape[0] > 1:
        joint_motion[:-1] = xy[1:] - xy[:-1]

    # Bone motion = bone velocity
    bone_motion = np.zeros_like(bones)
    if bones.shape[0] > 1:
        bone_motion[:-1] = bones[1:] - bones[:-1]

    # Concatenate all streams: (T, 96*2 + n_bones*2 + 96*2 + n_bones*2)
    return np.concatenate([
        joint.reshape(joint.shape[0], -1),
        bones.reshape(bones.shape[0], -1),
        joint_motion.reshape(joint_motion.shape[0], -1),
        bone_motion.reshape(bone_motion.shape[0], -1),
    ], axis=-1)


# ---------------------------------------------------------------------------
# Model variants
# ---------------------------------------------------------------------------

class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class BaselineModel(nn.Module):
    """Standard PoseTransformerEncoder (current SPJ model)."""
    def __init__(self, input_dim, d_model, n_heads, d_ff, n_layers, n_classes, dropout, max_seq_len):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = SinusoidalPE(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x, mask=None):
        h = self.dropout(self.pos_enc(self.input_proj(x)))
        kpm = (mask == 0) if mask is not None else None
        h = self.transformer(h, src_key_padding_mask=kpm)
        if mask is not None:
            m = mask.unsqueeze(-1)
            h = (h * m).sum(1) / m.sum(1).clamp(min=1)
        else:
            h = h.mean(1)
        return self.classifier(h)


class CausalDWConv1D(nn.Module):
    """Causal depthwise conv1d (from Kaggle 1st place)."""
    def __init__(self, channels, kernel_size=17):
        super().__init__()
        self.pad = nn.ConstantPad1d((kernel_size - 1, 0), 0)
        self.dw_conv = nn.Conv1d(channels, channels, kernel_size, groups=channels, bias=False)

    def forward(self, x):
        # x: (B, T, C) → transpose → conv → transpose
        x = x.transpose(1, 2)  # (B, C, T)
        x = self.pad(x)
        x = self.dw_conv(x)
        return x.transpose(1, 2)  # (B, T, C)


class Conv1DBlock(nn.Module):
    """Efficient Conv1D block from Kaggle 1st place: expand → DWConv → BN → project + residual."""
    def __init__(self, dim, kernel_size=17, expand_ratio=2, drop_rate=0.2):
        super().__init__()
        expanded = dim * expand_ratio
        self.expand = nn.Linear(dim, expanded)
        self.act = nn.SiLU()
        self.dw_conv = CausalDWConv1D(expanded, kernel_size)
        self.bn = nn.BatchNorm1d(expanded)
        self.project = nn.Linear(expanded, dim)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        residual = x
        x = self.act(self.expand(x))
        x = self.dw_conv(x)
        x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        x = self.project(x)
        x = self.drop(x)
        return x + residual


class Conv1DTransformerModel(nn.Module):
    """Kaggle 1st place architecture: Conv1D blocks + Transformer blocks."""
    def __init__(self, input_dim, d_model, n_heads, d_ff, n_classes, dropout, max_seq_len):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_bn = nn.BatchNorm1d(d_model)
        self.pos_enc = SinusoidalPE(d_model, max_seq_len)

        # Block 1: 3× Conv1D → Transformer
        self.conv_block1 = nn.Sequential(
            Conv1DBlock(d_model, kernel_size=17, drop_rate=dropout),
            Conv1DBlock(d_model, kernel_size=17, drop_rate=dropout),
            Conv1DBlock(d_model, kernel_size=17, drop_rate=dropout),
        )
        layer1 = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True)
        self.transformer1 = nn.TransformerEncoder(layer1, num_layers=1, enable_nested_tensor=False)

        # Block 2: 3× Conv1D → Transformer
        self.conv_block2 = nn.Sequential(
            Conv1DBlock(d_model, kernel_size=17, drop_rate=dropout),
            Conv1DBlock(d_model, kernel_size=17, drop_rate=dropout),
            Conv1DBlock(d_model, kernel_size=17, drop_rate=dropout),
        )
        layer2 = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True)
        self.transformer2 = nn.TransformerEncoder(layer2, num_layers=1, enable_nested_tensor=False)

        self.top = nn.Linear(d_model, d_model * 2)
        self.classifier = nn.Linear(d_model * 2, n_classes)

    def forward(self, x, mask=None):
        h = self.input_proj(x)
        h = self.input_bn(h.transpose(1, 2)).transpose(1, 2)

        kpm = (mask == 0) if mask is not None else None

        h = self.conv_block1(h)
        h = self.transformer1(h, src_key_padding_mask=kpm)

        h = self.conv_block2(h)
        h = self.transformer2(h, src_key_padding_mask=kpm)

        h = self.top(h)

        # Global average pooling (masked)
        if mask is not None:
            m = mask.unsqueeze(-1)
            h = (h * m).sum(1) / m.sum(1).clamp(min=1)
        else:
            h = h.mean(1)
        return self.classifier(h)


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

METHODS = {
    "baseline": {
        "transform": transform_baseline,
        "model_cls": BaselineModel,
        "desc": "Current PoseTransformerEncoder (Linear → Transformer)",
    },
    "velocity": {
        "transform": transform_velocity,
        "model_cls": BaselineModel,
        "desc": "Baseline + velocity features (dx, dx2)",
    },
    "xy_only": {
        "transform": transform_xy_only,
        "model_cls": BaselineModel,
        "desc": "Drop z coordinate, use only (x, y)",
    },
    "normalize": {
        "transform": transform_normalize,
        "model_cls": BaselineModel,
        "desc": "Nose-centered normalization",
    },
    "xy_velocity": {
        "transform": transform_xy_velocity,
        "model_cls": BaselineModel,
        "desc": "Drop z + velocity (Kaggle 1st place input)",
    },
    "norm_xy_vel": {
        "transform": transform_norm_xy_velocity,
        "model_cls": BaselineModel,
        "desc": "Normalize + drop z + velocity (combined)",
    },
    "conv1d": {
        "transform": transform_norm_xy_velocity,
        "model_cls": Conv1DTransformerModel,
        "desc": "Conv1D+Transformer (Kaggle architecture) + norm+xy+vel",
    },
    "4stream": {
        "transform": transform_4stream,
        "model_cls": BaselineModel,
        "desc": "4-stream: Joint+Bone+JointMotion+BoneMotion (SAM-SLR style)",
    },
}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_method(
    method_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    label_encoder: LabelEncoder,
    epochs: int = 20,
    batch_size: int = 256,
    d_model: int = 128,
    n_heads: int = 4,
    d_ff: int = 256,
    n_layers: int = 3,
    dropout: float = 0.1,
    max_seq_len: int = 300,
    lr: float = 1e-3,
    device_str: str = "mps",
) -> dict:
    method = METHODS[method_name]
    logger.info("=" * 60)
    logger.info("Method: %s — %s", method_name, method["desc"])
    logger.info("=" * 60)

    # Build datasets
    t0 = time.time()
    train_ds = MethodDataset(train_df, label_encoder, max_seq_len, method["transform"])
    val_ds = MethodDataset(val_df, label_encoder, max_seq_len, method["transform"])
    load_time = time.time() - t0
    logger.info("Data loaded: %d train, %d val (%.1fs)", len(train_ds), len(val_ds), load_time)

    if len(train_ds) == 0:
        return {"method": method_name, "error": "No training samples"}

    # Detect input_dim
    sample = train_ds[0]
    input_dim = sample["features"].shape[-1]
    logger.info("Input dim: %d", input_dim)

    # Device
    if device_str == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif device_str == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    # Model
    model_cls = method["model_cls"]
    if model_cls == Conv1DTransformerModel:
        model = model_cls(input_dim, d_model, n_heads, d_ff, label_encoder.n_classes, dropout, max_seq_len)
    else:
        model = model_cls(input_dim, d_model, n_heads, d_ff, n_layers, label_encoder.n_classes, dropout, max_seq_len)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model params: %s", f"{n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_epoch = 0
    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch in train_loader:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            try:
                with torch.autocast(device_type=str(device).split(":")[0], dtype=torch.float16):
                    logits = model(features, mask)
                    loss = criterion(logits, labels)
            except Exception:
                logits = model(features, mask)
                loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / max(1, total)
        train_acc = correct / max(1, total)

        # Validate
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                mask = batch["mask"].to(device)
                labels = batch["label"].to(device)
                logits = model(features, mask)
                loss = criterion(logits, labels)
                v_loss += loss.item() * labels.size(0)
                v_correct += (logits.argmax(1) == labels).sum().item()
                v_total += labels.size(0)

        val_loss = v_loss / max(1, v_total)
        val_acc = v_correct / max(1, v_total)

        history["train_acc"].append(round(train_acc, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

        scheduler.step()

        if epoch % 5 == 0 or epoch == epochs:
            logger.info("  Epoch %d/%d — train_acc=%.4f val_acc=%.4f (best=%.4f @ep%d)",
                        epoch, epochs, train_acc, val_acc, best_val_acc, best_epoch)

    elapsed = time.time() - t_start

    result = {
        "method": method_name,
        "desc": method["desc"],
        "input_dim": input_dim,
        "n_params": n_params,
        "best_val_acc": round(best_val_acc, 4),
        "best_epoch": best_epoch,
        "final_train_acc": history["train_acc"][-1],
        "final_val_acc": history["val_acc"][-1],
        "elapsed_sec": round(elapsed, 1),
        "epochs": epochs,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_classes": label_encoder.n_classes,
    }
    logger.info("  → Best val_acc=%.4f at epoch %d (%.0fs)", best_val_acc, best_epoch, elapsed)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare training methods")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs per method")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-seq-len", type=int, default=300)
    parser.add_argument("--min-samples", type=int, default=3, help="Min samples per class")
    parser.add_argument("--methods", nargs="+", default=list(METHODS.keys()),
                        choices=list(METHODS.keys()), help="Methods to test")
    parser.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    args = parser.parse_args()

    # Load data
    train_df, val_df = load_splits(min_samples=args.min_samples)
    labels = sorted(train_df["label"].unique())
    label_encoder = LabelEncoder(labels)

    results = []
    for method_name in args.methods:
        result = train_one_method(
            method_name, train_df, val_df, label_encoder,
            epochs=args.epochs, batch_size=args.batch_size,
            d_model=args.d_model, lr=args.lr,
            max_seq_len=args.max_seq_len, device_str=args.device,
        )
        results.append(result)

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Method':<20} {'Input Dim':>10} {'Params':>10} {'Best Val':>10} {'@Epoch':>8} {'Time':>8}")
    print("-" * 80)
    for r in sorted(results, key=lambda x: x.get("best_val_acc", 0), reverse=True):
        print(f"{r['method']:<20} {r.get('input_dim', '?'):>10} {r.get('n_params', '?'):>10,} "
              f"{r.get('best_val_acc', 0):>10.4f} {r.get('best_epoch', 0):>8} {r.get('elapsed_sec', 0):>7.0f}s")
    print("=" * 80)

    # Save results
    out_path = DATA_DIR / "method_comparison.json"
    with open(out_path, "w") as f:
        json.dump({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "results": results}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
