"""Training pipeline: dataset, model, training loop, and checkpoints.

Provides a PoseTransformerEncoder for sign language gloss classification
from 543-landmark pose sequences, with stratified data splitting,
background-thread training, and checkpoint management.
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label encoder
# ---------------------------------------------------------------------------

class LabelEncoder:
    """Bidirectional mapping between string labels and integer indices."""

    def __init__(self, labels: Optional[list[str]] = None):
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

    def decode(self, idx: int) -> str:
        return self.idx_to_label[idx]

    @property
    def n_classes(self) -> int:
        return len(self.label_to_idx)

    def to_dict(self) -> dict:
        return {"label_to_idx": self.label_to_idx}

    @classmethod
    def from_dict(cls, d: dict) -> "LabelEncoder":
        enc = cls()
        enc.label_to_idx = d["label_to_idx"]
        enc.idx_to_label = {int(i): lab for lab, i in enc.label_to_idx.items()}
        return enc


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PoseSegmentDataset(Dataset):
    """Loads exported NPZ segments and pads to a fixed max_seq_len."""

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        npz_dir: Path,
        label_encoder: LabelEncoder,
        max_seq_len: int = 300,
    ):
        self.npz_dir = Path(npz_dir)
        self.max_seq_len = max_seq_len
        self.label_encoder = label_encoder
        self.items: list[tuple[str, int]] = []

        for _, row in manifest_df.iterrows():
            label = str(row.get("label", row.get("reviewed_text", row.get("text", "")))).strip()
            if not label or label not in label_encoder.label_to_idx:
                continue
            seg_id = str(row["segment_id"])
            self.items.append((seg_id, label_encoder.encode(label)))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        seg_id, label_idx = self.items[idx]
        npz_path = self._find_npz(seg_id)
        d = np.load(str(npz_path))

        pose = d["pose"].astype(np.float32)  # (T, N, 3) — N may be 543 or filtered
        T = pose.shape[0]

        # Flatten to (T, N*3)
        features = pose.reshape(T, -1)

        # Pad or truncate
        if T >= self.max_seq_len:
            features = features[:self.max_seq_len]
            mask = np.ones(self.max_seq_len, dtype=np.float32)
        else:
            pad_len = self.max_seq_len - T
            features = np.concatenate([
                features,
                np.zeros((pad_len, features.shape[1]), dtype=np.float32),
            ], axis=0)
            mask = np.concatenate([
                np.ones(T, dtype=np.float32),
                np.zeros(pad_len, dtype=np.float32),
            ])

        return {
            "features": torch.from_numpy(features),  # (max_seq_len, N*3)
            "mask": torch.from_numpy(mask),           # (max_seq_len,)
            "label": torch.tensor(label_idx, dtype=torch.long),
        }

    def _find_npz(self, seg_id: str) -> Path:
        p = self.npz_dir / f"{seg_id}.npz"
        if p.exists():
            return p
        # Fallback: search for matching file
        candidates = list(self.npz_dir.glob(f"*{seg_id}*.npz"))
        if candidates:
            return candidates[0]
        raise FileNotFoundError(f"NPZ not found for segment {seg_id} in {self.npz_dir}")


class AugmentedPoseDataset(Dataset):
    """Pose dataset with on-the-fly augmentation for few-shot sign learning.

    Each original sample produces ``n_augments`` augmented variants:
      - Random temporal crop (75-100% of original)
      - Random speed variation (0.8x-1.2x via frame resampling)
      - Gaussian noise on coordinates
      - Random scale jitter (0.9-1.1)

    The first variant (aug_idx=0) is always the unaugmented original.
    """

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        npz_dir: Path,
        label_encoder: "LabelEncoder",
        max_seq_len: int = 300,
        augment: bool = True,
        n_augments: int = 10,
    ):
        self.npz_dir = Path(npz_dir)
        self.max_seq_len = max_seq_len
        self.label_encoder = label_encoder
        self.augment = augment
        self.n_augments = n_augments if augment else 1

        self.data: list[tuple[np.ndarray, int]] = []
        skipped = 0

        for _, row in manifest_df.iterrows():
            label_str = str(row.get("label", row.get("reviewed_text", row.get("text", "")))).strip()
            if not label_str or label_str not in label_encoder.label_to_idx:
                skipped += 1
                continue

            seg_id = str(row.get("segment_id", Path(str(row.get("npz_path", ""))).stem))
            npz_path = self.npz_dir / f"{seg_id}.npz"
            if not npz_path.exists():
                skipped += 1
                continue

            d = np.load(str(npz_path))
            pose = d["pose"].astype(np.float32)  # (T, N, 3) — N may be filtered
            label_idx = label_encoder.encode(label_str)
            self.data.append((pose, label_idx))

        logger.info("AugmentedPoseDataset: %d segments loaded (%d skipped), %dx augment",
                     len(self.data), skipped, self.n_augments)

    def __len__(self) -> int:
        return len(self.data) * self.n_augments

    def __getitem__(self, idx: int) -> dict:
        real_idx = idx // self.n_augments
        aug_idx = idx % self.n_augments

        pose, label_idx = self.data[real_idx]

        if self.augment and aug_idx > 0:
            pose = self._augment(pose, aug_idx)

        T = pose.shape[0]
        features = pose.reshape(T, -1)  # (T, 1629)

        if T >= self.max_seq_len:
            features = features[:self.max_seq_len]
            mask = np.ones(self.max_seq_len, dtype=np.float32)
        else:
            pad_len = self.max_seq_len - T
            features = np.concatenate([
                features,
                np.zeros((pad_len, features.shape[1]), dtype=np.float32),
            ], axis=0)
            mask = np.concatenate([
                np.ones(T, dtype=np.float32),
                np.zeros(pad_len, dtype=np.float32),
            ])

        return {
            "features": torch.from_numpy(features),
            "mask": torch.from_numpy(mask),
            "label": torch.tensor(label_idx, dtype=torch.long),
        }

    def _augment(self, pose: np.ndarray, aug_idx: int) -> np.ndarray:
        rng = np.random.RandomState(aug_idx * 1000 + id(pose) % 10000)
        T = pose.shape[0]
        pose = pose.copy()

        # 1. Temporal crop (75-100%)
        crop_ratio = rng.uniform(0.75, 1.0)
        crop_len = max(5, int(T * crop_ratio))
        start = rng.randint(0, max(1, T - crop_len))
        pose = pose[start:start + crop_len]

        # 2. Speed variation via frame resampling (0.8x-1.2x)
        speed = rng.uniform(0.8, 1.2)
        n_landmarks = pose.shape[1]
        new_T = max(5, int(pose.shape[0] / speed))
        if new_T != pose.shape[0]:
            old_indices = np.linspace(0, pose.shape[0] - 1, new_T)
            new_pose = np.zeros((new_T, n_landmarks, 3), dtype=np.float32)
            for i, oi in enumerate(old_indices):
                lo = int(oi)
                hi = min(lo + 1, pose.shape[0] - 1)
                frac = oi - lo
                new_pose[i] = pose[lo] * (1 - frac) + pose[hi] * frac
            pose = new_pose

        # 3. Gaussian noise on coordinates
        noise_std = rng.uniform(0.001, 0.005)
        pose = pose + rng.randn(*pose.shape).astype(np.float32) * noise_std

        # 4. Random scale jitter (0.9-1.1)
        scale = rng.uniform(0.9, 1.1)
        pose = pose * scale

        return pose


# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------

def split_dataset(
    manifest_df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
    output_dir: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split of manifest into train/val/test sets.

    Returns (train_df, val_df, test_df) and optionally saves CSVs.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # Determine label column
    df = manifest_df.copy()
    if "label" not in df.columns:
        df["label"] = df["reviewed_text"].where(
            df["reviewed_text"].str.strip() != "",
            df["text"],
        )

    rng = np.random.RandomState(random_seed)
    train_parts, val_parts, test_parts = [], [], []

    for label, group in df.groupby("label"):
        n = len(group)
        indices = rng.permutation(n)
        n_train = max(1, int(n * train_ratio))
        n_val = max(0, int(n * val_ratio))

        train_parts.append(group.iloc[indices[:n_train]])
        val_parts.append(group.iloc[indices[n_train:n_train + n_val]])
        test_parts.append(group.iloc[indices[n_train + n_val:]])

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(output_dir / "train.csv", index=False)
        val_df.to_csv(output_dir / "val.csv", index=False)
        test_df.to_csv(output_dir / "test.csv", index=False)

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (non-learnable)."""

    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class PoseTransformerEncoder(nn.Module):
    """Transformer encoder for pose-sequence classification.

    Input:  (batch, max_seq_len, input_dim)
      -> Linear projection -> d_model
      -> Sinusoidal positional encoding
      -> N Transformer encoder layers
      -> Mean pooling (masked)
      -> Linear -> n_classes

    Default input_dim=1629 (543*3, all landmarks). With SL landmark
    filtering, use input_dim=SL_INPUT_DIM (~441 = ~147 landmarks * 3).
    """

    def __init__(
        self,
        input_dim: int = 1629,
        d_model: int = 256,
        n_heads: int = 4,
        d_ff: int = 512,
        n_layers: int = 4,
        n_classes: int = 10,
        dropout: float = 0.1,
        max_seq_len: int = 300,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
            mask: (batch, seq_len) — 1.0 for real frames, 0.0 for padding

        Returns:
            logits: (batch, n_classes)
        """
        h = self.input_proj(x)              # (B, S, d_model)
        h = self.pos_enc(h)
        h = self.dropout(h)

        # Transformer expects key_padding_mask: True = ignore
        if mask is not None:
            key_padding_mask = (mask == 0)   # (B, S) bool
        else:
            key_padding_mask = None

        h = self.transformer(h, src_key_padding_mask=key_padding_mask)

        # Masked mean pooling
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)  # (B, S, 1)
            h = (h * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            h = h.mean(dim=1)

        return self.classifier(h)


# ---------------------------------------------------------------------------
# Training config & state
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Hyperparameters for training."""
    backbone: str = "from_scratch"
    lr: float = 1e-3
    epochs: int = 50
    batch_size: int = 256
    d_model: int = 256
    n_heads: int = 4
    d_ff: int = 512
    n_layers: int = 4
    dropout: float = 0.1
    max_seq_len: int = 300
    weight_decay: float = 1e-4
    device: str = "mps"
    freeze_epochs: int = 0          # Phase 1 frozen epochs (0 = no freeze)
    unfreeze_lr: float = 3e-4       # Phase 2 learning rate
    patience: int = 0               # early stopping patience (0 = disabled)
    label_smoothing: float = 0.0    # cross-entropy label smoothing


@dataclass
class TrainingState:
    """Mutable state shared between training thread and Streamlit UI.

    All fields are simple Python types (GIL-safe reads from Streamlit thread).
    """
    running: bool = False
    finished: bool = False
    epoch: int = 0
    total_epochs: int = 0
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    train_accs: list[float] = field(default_factory=list)
    val_accs: list[float] = field(default_factory=list)
    best_val_acc: float = 0.0
    best_epoch: int = 0
    error: str = ""
    checkpoint_path: str = ""
    stop_requested: bool = False
    phase: int = 1
    phase_description: str = ""
    early_stopped: bool = False


# ---------------------------------------------------------------------------
# Pretrained backbone loading
# ---------------------------------------------------------------------------

def _load_pretrained_weights(model: PoseTransformerEncoder, pretrained_path: Path) -> None:
    """Load encoder weights from ssl_pretrain checkpoint.

    Matches keys: input_proj, pos_enc, transformer (encoder layers).
    Skips: classifier head (different n_classes), reconstruction head.
    """
    ckpt = torch.load(str(pretrained_path), map_location="cpu", weights_only=False)
    encoder_state = ckpt.get("encoder_state_dict", ckpt.get("model_state_dict", {}))

    model_state = model.state_dict()
    model_keys = set(model_state.keys())

    # Filter to matching keys with matching shapes
    compatible = {
        k: v for k, v in encoder_state.items()
        if k in model_keys and v.shape == model_state[k].shape
    }
    model.load_state_dict(compatible, strict=False)
    logger.info("Loaded %d/%d pretrained weights from %s",
                len(compatible), len(model_keys), pretrained_path)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _get_device(config: TrainingConfig) -> torch.device:
    if config.device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if config.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    npz_dir: Path,
    label_encoder: LabelEncoder,
    config: TrainingConfig,
    state: TrainingState,
    output_dir: Path,
    pretrained_path: Optional[Path] = None,
) -> None:
    """Train PoseTransformerEncoder. Designed to run in a background thread.

    Updates `state` in-place so the Streamlit UI can poll progress.
    Saves best checkpoint to output_dir.

    Args:
        pretrained_path: Optional path to a pretrained encoder checkpoint
            from ssl_pretrain.py. If provided, encoder weights are initialized
            from this checkpoint before training.
    """
    try:
        state.running = True
        state.total_epochs = config.epochs
        device = _get_device(config)
        logger.info("Training on device: %s", device)

        # Datasets
        train_ds = PoseSegmentDataset(train_df, npz_dir, label_encoder, config.max_seq_len)
        val_ds = PoseSegmentDataset(val_df, npz_dir, label_encoder, config.max_seq_len)

        if len(train_ds) == 0:
            state.error = "No training samples found. Check NPZ files and labels."
            state.finished = True
            state.running = False
            return

        train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        ) if len(val_ds) > 0 else None

        # Detect input_dim from the first NPZ file
        sample = train_ds[0]
        input_dim = sample["features"].shape[-1]
        logger.info("Detected input_dim=%d from NPZ data", input_dim)

        # Model
        model = PoseTransformerEncoder(
            input_dim=input_dim,
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            n_layers=config.n_layers,
            n_classes=label_encoder.n_classes,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len,
        )
        if pretrained_path:
            _load_pretrained_weights(model, pretrained_path)

        # Two-phase transfer: freeze encoder if freeze_epochs > 0
        if config.freeze_epochs > 0:
            for name, param in model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False
            state.phase = 1
            state.phase_description = f"Phase 1: Frozen encoder (epochs 1-{config.freeze_epochs})"
            logger.info("Phase 1: Frozen encoder, training classifier only")

        model = model.to(device)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs,
        )

        best_val_acc = 0.0
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, config.epochs + 1):
            if state.stop_requested:
                logger.info("Training stopped by user at epoch %d", epoch)
                break

            # Phase transition: unfreeze at freeze_epochs + 1
            if config.freeze_epochs > 0 and epoch == config.freeze_epochs + 1:
                for param in model.parameters():
                    param.requires_grad = True
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=config.unfreeze_lr,
                    weight_decay=config.weight_decay,
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=config.epochs - config.freeze_epochs,
                )
                state.phase = 2
                state.phase_description = f"Phase 2: All params unfrozen (lr={config.unfreeze_lr})"
                logger.info("Phase 2: Unfrozen all params, lr=%s", config.unfreeze_lr)

            state.epoch = epoch

            # --- Train ---
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for batch in train_loader:
                features = batch["features"].to(device)
                mask = batch["mask"].to(device)
                labels = batch["label"].to(device)

                optimizer.zero_grad()

                try:
                    with torch.autocast(device_type=str(device), dtype=torch.float16):
                        logits = model(features, mask)
                        loss = criterion(logits, labels)
                except Exception:
                    # Fallback to fp32 if autocast not supported for some ops
                    logits = model(features, mask)
                    loss = criterion(logits, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item() * labels.size(0)
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)

            train_loss = total_loss / max(1, total)
            train_acc = correct / max(1, total)
            state.train_losses.append(round(train_loss, 4))
            state.train_accs.append(round(train_acc, 4))

            # --- Validate ---
            val_loss = 0.0
            val_acc = 0.0
            if val_loader is not None:
                model.eval()
                v_loss = 0.0
                v_correct = 0
                v_total = 0
                with torch.no_grad():
                    for batch in val_loader:
                        features = batch["features"].to(device)
                        mask = batch["mask"].to(device)
                        labels = batch["label"].to(device)
                        logits = model(features, mask)
                        loss = criterion(logits, labels)
                        v_loss += loss.item() * labels.size(0)
                        v_correct += (logits.argmax(dim=1) == labels).sum().item()
                        v_total += labels.size(0)
                val_loss = v_loss / max(1, v_total)
                val_acc = v_correct / max(1, v_total)

            state.val_losses.append(round(val_loss, 4))
            state.val_accs.append(round(val_acc, 4))

            scheduler.step()

            # Save best checkpoint
            metric = val_acc if val_loader else train_acc
            if metric > best_val_acc:
                best_val_acc = metric
                state.best_val_acc = round(best_val_acc, 4)
                state.best_epoch = epoch

                ckpt_path = output_dir / "best_model.pt"
                save_checkpoint(
                    model, label_encoder, config, ckpt_path,
                    epoch=epoch, val_acc=best_val_acc,
                    n_classes=label_encoder.n_classes,
                    n_train=len(train_ds),
                    input_dim=input_dim,
                )
                state.checkpoint_path = str(ckpt_path)

            logger.info(
                "Epoch %d/%d — train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
                epoch, config.epochs, train_loss, train_acc, val_loss, val_acc,
            )

            # Early stopping
            if config.patience > 0:
                epochs_since_improvement = epoch - state.best_epoch
                if epochs_since_improvement >= config.patience:
                    state.early_stopped = True
                    logger.info("Early stopping at epoch %d (no improvement for %d epochs)",
                                epoch, config.patience)
                    break

        state.finished = True
        state.running = False

    except Exception as exc:
        state.error = str(exc)
        state.finished = True
        state.running = False
        logger.exception("Training failed")


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    label_encoder: LabelEncoder,
    config: TrainingConfig,
    path: Path,
    **extra_meta,
) -> None:
    """Save model checkpoint with metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "label_encoder": label_encoder.to_dict(),
        "config": asdict(config),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        **extra_meta,
    }, str(path))


def load_checkpoint(
    path: Path,
    device: Optional[str] = None,
) -> tuple[PoseTransformerEncoder, LabelEncoder, TrainingConfig, dict]:
    """Load checkpoint, returning (model, label_encoder, config, metadata).

    The model is loaded onto the specified device (or CPU by default).
    """
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)

    config = TrainingConfig(**ckpt["config"])
    label_encoder = LabelEncoder.from_dict(ckpt["label_encoder"])

    # Detect input_dim: check checkpoint metadata, or infer from weights
    input_dim = ckpt.get("input_dim", None)
    if input_dim is None:
        # Infer from input_proj weight shape: Linear(input_dim, d_model)
        proj_weight = ckpt["model_state_dict"].get("input_proj.weight")
        input_dim = proj_weight.shape[1] if proj_weight is not None else 1629

    model = PoseTransformerEncoder(
        input_dim=input_dim,
        d_model=config.d_model,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        n_layers=config.n_layers,
        n_classes=label_encoder.n_classes,
        dropout=config.dropout,
        max_seq_len=config.max_seq_len,
    )
    model.load_state_dict(ckpt["model_state_dict"])

    if device:
        dev = torch.device(device)
    elif config.device == "mps" and torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    model.to(dev)
    model.eval()

    meta = {k: v for k, v in ckpt.items()
            if k not in ("model_state_dict", "label_encoder", "config")}

    return model, label_encoder, config, meta


def list_checkpoints(models_dir: Path) -> list[dict]:
    """List all .pt checkpoints with their metadata."""
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return []

    checkpoints = []
    for pt_file in sorted(models_dir.glob("*.pt")):
        try:
            ckpt = torch.load(str(pt_file), map_location="cpu", weights_only=False)
            config = ckpt.get("config", {})
            label_enc = ckpt.get("label_encoder", {})
            checkpoints.append({
                "path": str(pt_file),
                "filename": pt_file.name,
                "backbone": config.get("backbone", "unknown"),
                "n_classes": len(label_enc.get("label_to_idx", {})),
                "val_acc": ckpt.get("val_acc", 0.0),
                "epoch": ckpt.get("epoch", 0),
                "timestamp": ckpt.get("timestamp", ""),
                "n_train": ckpt.get("n_train", 0),
                "d_model": config.get("d_model", 0),
                "n_layers": config.get("n_layers", 0),
            })
        except Exception as exc:
            logger.warning("Cannot read checkpoint %s: %s", pt_file, exc)
    return checkpoints


# Recommended defaults for category→word transfer learning
TRANSFER_DEFAULTS = {
    "freeze_epochs": 10,
    "unfreeze_lr": 3e-4,
    "patience": 25,
    "label_smoothing": 0.1,
}

# Minimum val_acc for a category checkpoint to be considered for transfer
_CAT_CKPT_MIN_ACC = 0.3


def find_category_checkpoints(
    models_dir: Path, min_val_acc: float = _CAT_CKPT_MIN_ACC,
) -> list[dict]:
    """Find category model checkpoints suitable for transfer learning.

    Looks for cat*.pt files with val_acc above the threshold.
    Returns sorted by val_acc descending (best first).
    """
    all_ckpts = list_checkpoints(models_dir)
    cats = [
        c for c in all_ckpts
        if c["filename"].startswith("cat") and c.get("val_acc", 0) > min_val_acc
    ]
    return sorted(cats, key=lambda c: c.get("val_acc", 0), reverse=True)
