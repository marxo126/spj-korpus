"""Self-supervised pretraining: Masked Pose Modeling.

Learns temporal dynamics of sign language pose sequences by masking random
frames and predicting the masked landmarks from context. No labels needed.

Concepts from SSVP-SLT (masked pretraining) and BERT (masked LM) applied
to our PoseTransformerEncoder architecture on pose data.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

INPUT_DIM = 1629  # 543 landmarks * 3 coords (X, Y, Z)


# ---------------------------------------------------------------------------
# Config & State
# ---------------------------------------------------------------------------

@dataclass
class PretrainConfig:
    """Hyperparameters for self-supervised pretraining."""
    mask_ratio: float = 0.15
    lr: float = 1e-4
    epochs: int = 30
    batch_size: int = 128
    d_model: int = 256
    n_heads: int = 4
    d_ff: int = 512
    n_layers: int = 4
    dropout: float = 0.1
    max_seq_len: int = 300
    weight_decay: float = 1e-4
    device: str = "mps"


@dataclass
class PretrainState:
    """Mutable state shared between pretrain thread and Streamlit UI.

    All fields are simple Python types (GIL-safe reads from Streamlit thread).
    """
    running: bool = False
    finished: bool = False
    epoch: int = 0
    total_epochs: int = 0
    losses: list[float] = field(default_factory=list)
    best_loss: float = float("inf")
    best_epoch: int = 0
    n_windows: int = 0
    error: str = ""
    checkpoint_path: str = ""
    stop_requested: bool = False


# ---------------------------------------------------------------------------
# Dataset — unlabeled pose windows
# ---------------------------------------------------------------------------

class UnlabeledPoseDataset(Dataset):
    """Load .pose files, sample sliding windows of max_seq_len frames.

    No labels needed — just pose sequences for self-supervised learning.
    Each .pose file contributes multiple overlapping windows.
    """

    def __init__(self, pose_dir: Path, max_seq_len: int = 300, stride: int = 150):
        self.max_seq_len = max_seq_len
        self.windows: list[tuple[Path, int]] = []  # (pose_file, start_frame)

        pose_dir = Path(pose_dir)
        pose_files = sorted(pose_dir.glob("*.pose"))

        for pf in pose_files:
            if pf.stat().st_size == 0:
                continue
            try:
                n_frames = self._count_frames(pf)
            except Exception:
                continue
            if n_frames < 10:
                continue

            # Sliding window
            for start in range(0, max(1, n_frames - max_seq_len + 1), stride):
                self.windows.append((pf, start))

            # If file is shorter than max_seq_len, still include one window
            if n_frames < max_seq_len and not self.windows or self.windows[-1][0] != pf:
                self.windows.append((pf, 0))

        logger.info("UnlabeledPoseDataset: %d windows from %d pose files",
                     len(self.windows), len(pose_files))

    @staticmethod
    def _count_frames(pose_path: Path) -> int:
        from pose_format import Pose
        with open(pose_path, "rb") as f:
            pose = Pose.read(f.read())
        return pose.body.data.shape[0]

    @staticmethod
    def _load_window(pose_path: Path, start: int, length: int) -> np.ndarray:
        from pose_format import Pose
        with open(pose_path, "rb") as f:
            pose = Pose.read(f.read())
        data = pose.body.data.numpy()  # (T, 1, 543, 3)
        data = data[:, 0, :, :]        # (T, 543, 3) — drop person dim
        T = data.shape[0]
        end = min(start + length, T)
        return data[start:end].astype(np.float32)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        pose_path, start = self.windows[idx]
        pose = self._load_window(pose_path, start, self.max_seq_len)
        T = pose.shape[0]

        # Flatten to (T, 1629)
        features = pose.reshape(T, -1)

        # Pad or truncate
        if T >= self.max_seq_len:
            features = features[:self.max_seq_len]
            mask = np.ones(self.max_seq_len, dtype=np.float32)
        else:
            pad_len = self.max_seq_len - T
            features = np.concatenate([
                features,
                np.zeros((pad_len, INPUT_DIM), dtype=np.float32),
            ], axis=0)
            mask = np.concatenate([
                np.ones(T, dtype=np.float32),
                np.zeros(pad_len, dtype=np.float32),
            ])

        return {
            "features": torch.from_numpy(features),  # (max_seq_len, 1629)
            "mask": torch.from_numpy(mask),           # (max_seq_len,)
        }


# ---------------------------------------------------------------------------
# Model — encoder + reconstruction head
# ---------------------------------------------------------------------------

class MaskedPoseModel(nn.Module):
    """PoseTransformerEncoder + reconstruction head for self-supervised pretraining.

    Architecture matches PoseTransformerEncoder (same layers) plus a linear
    reconstruction head that predicts masked frame landmarks.
    """

    def __init__(self, config: PretrainConfig):
        super().__init__()
        d_model = config.d_model

        # Encoder (same as PoseTransformerEncoder minus classifier)
        self.input_proj = nn.Linear(INPUT_DIM, d_model)
        self.pos_enc = _SinusoidalPE(d_model, config.max_seq_len)
        self.dropout = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.n_layers,
        )

        # Reconstruction head — predict masked frame features
        self.recon_head = nn.Linear(d_model, INPUT_DIM)

    def forward(
        self,
        features: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (B, S, 1629) — input pose features (with masked frames zeroed)
            seq_mask: (B, S) — 1.0 for real frames, 0.0 for padding

        Returns:
            reconstruction: (B, S, 1629) — predicted features for all frames
            encoder_output: (B, S, d_model) — hidden states
        """
        h = self.input_proj(features)
        h = self.pos_enc(h)
        h = self.dropout(h)

        key_padding_mask = (seq_mask == 0)
        h = self.transformer(h, src_key_padding_mask=key_padding_mask)

        reconstruction = self.recon_head(h)
        return reconstruction, h


class _SinusoidalPE(nn.Module):
    """Sinusoidal positional encoding (non-learnable). Same as trainer.py."""

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
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


# ---------------------------------------------------------------------------
# Pretraining loop
# ---------------------------------------------------------------------------

def _get_device(config: PretrainConfig) -> torch.device:
    if config.device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if config.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def pretrain_masked_pose(
    pose_dir: Path,
    config: PretrainConfig,
    state: PretrainState,
    output_dir: Path,
) -> str:
    """Background-threadable pretraining loop.

    Masks random frames of pose sequences and trains the encoder to
    reconstruct the masked landmarks. Updates `state` in-place so
    Streamlit can poll progress.

    Returns path to best checkpoint.
    """
    try:
        state.running = True
        state.total_epochs = config.epochs
        device = _get_device(config)
        logger.info("SSL pretraining on device: %s", device)

        # Dataset
        dataset = UnlabeledPoseDataset(pose_dir, config.max_seq_len)
        state.n_windows = len(dataset)

        if len(dataset) == 0:
            state.error = "No valid .pose files found for pretraining."
            state.finished = True
            state.running = False
            return ""

        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )

        # Model
        model = MaskedPoseModel(config).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs,
        )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        best_loss = float("inf")
        best_path = ""

        for epoch in range(1, config.epochs + 1):
            if state.stop_requested:
                logger.info("Pretraining stopped by user at epoch %d", epoch)
                break

            state.epoch = epoch
            model.train()
            total_loss = 0.0
            n_samples = 0

            for batch in loader:
                features = batch["features"].to(device)  # (B, S, 1629)
                seq_mask = batch["mask"].to(device)       # (B, S)
                B, S, D = features.shape

                # Create frame mask: randomly mask config.mask_ratio of real frames
                frame_mask = torch.zeros(B, S, device=device)
                for i in range(B):
                    real_len = int(seq_mask[i].sum().item())
                    if real_len < 2:
                        continue
                    n_mask = max(1, int(real_len * config.mask_ratio))
                    mask_indices = torch.randperm(real_len, device=device)[:n_mask]
                    frame_mask[i, mask_indices] = 1.0

                # Store original features for loss computation
                original = features.clone()

                # Zero out masked frames in input
                masked_input = features.clone()
                masked_input[frame_mask.bool()] = 0.0

                optimizer.zero_grad()

                try:
                    with torch.autocast(device_type=str(device), dtype=torch.float16):
                        reconstruction, _ = model(masked_input, seq_mask)
                        # MSE loss on masked frames only
                        mask_expanded = frame_mask.unsqueeze(-1)  # (B, S, 1)
                        diff = (reconstruction - original) ** 2   # (B, S, D)
                        loss = (diff * mask_expanded).sum() / mask_expanded.sum().clamp(min=1) / D
                except Exception:
                    reconstruction, _ = model(masked_input, seq_mask)
                    mask_expanded = frame_mask.unsqueeze(-1)
                    diff = (reconstruction - original) ** 2
                    loss = (diff * mask_expanded).sum() / mask_expanded.sum().clamp(min=1) / D

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item() * B
                n_samples += B

            epoch_loss = total_loss / max(1, n_samples)
            state.losses.append(round(epoch_loss, 6))
            scheduler.step()

            # Save best checkpoint (encoder weights only)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                state.best_loss = round(best_loss, 6)
                state.best_epoch = epoch

                best_path = str(output_dir / "pretrained_encoder.pt")
                _save_pretrain_checkpoint(model, config, best_path,
                                          epoch=epoch, loss=best_loss,
                                          n_windows=len(dataset))
                state.checkpoint_path = best_path

            logger.info("Pretrain epoch %d/%d — loss=%.6f (best=%.6f @ epoch %d)",
                        epoch, config.epochs, epoch_loss, best_loss, state.best_epoch)

        state.finished = True
        state.running = False
        return best_path

    except Exception as exc:
        state.error = str(exc)
        state.finished = True
        state.running = False
        logger.exception("Pretraining failed")
        return ""


def _save_pretrain_checkpoint(
    model: MaskedPoseModel,
    config: PretrainConfig,
    path: str,
    **extra_meta,
) -> None:
    """Save encoder weights (without reconstruction head) for downstream use."""
    # Extract only encoder keys (exclude recon_head)
    full_state = model.state_dict()
    encoder_keys = [k for k in full_state if not k.startswith("recon_head")]
    encoder_state = {k: full_state[k] for k in encoder_keys}

    torch.save({
        "encoder_state_dict": encoder_state,
        "full_state_dict": full_state,
        "config": {
            "d_model": config.d_model,
            "n_heads": config.n_heads,
            "d_ff": config.d_ff,
            "n_layers": config.n_layers,
            "dropout": config.dropout,
            "max_seq_len": config.max_seq_len,
            "mask_ratio": config.mask_ratio,
            "lr": config.lr,
        },
        "type": "pretrained_encoder",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        **extra_meta,
    }, path)


def load_pretrained_encoder(path: Path) -> dict:
    """Load pretrained encoder state_dict (without reconstruction head).

    Returns the checkpoint dict. The 'encoder_state_dict' key contains
    weights compatible with PoseTransformerEncoder (matching keys:
    input_proj, pos_enc, dropout, transformer).
    """
    return torch.load(str(path), map_location="cpu", weights_only=False)


def list_pretrain_checkpoints(models_dir: Path) -> list[dict]:
    """List pretrained encoder checkpoints in the given directory."""
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return []

    checkpoints = []
    for pt_file in sorted(models_dir.glob("pretrained_*.pt")):
        try:
            ckpt = torch.load(str(pt_file), map_location="cpu", weights_only=False)
            if ckpt.get("type") != "pretrained_encoder":
                continue
            config = ckpt.get("config", {})
            checkpoints.append({
                "path": str(pt_file),
                "filename": pt_file.name,
                "d_model": config.get("d_model", 0),
                "n_layers": config.get("n_layers", 0),
                "mask_ratio": config.get("mask_ratio", 0),
                "epoch": ckpt.get("epoch", 0),
                "loss": ckpt.get("loss", 0),
                "n_windows": ckpt.get("n_windows", 0),
                "timestamp": ckpt.get("timestamp", ""),
            })
        except Exception as exc:
            logger.warning("Cannot read pretrain checkpoint %s: %s", pt_file, exc)
    return checkpoints
