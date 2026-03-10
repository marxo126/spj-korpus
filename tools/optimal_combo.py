"""Optimal combo: Conv1D + norm_xy_velocity + mirror+rotation only.

Combines the best findings:
  - Architecture: conv1d_transformer (9x better than baseline transformer)
  - Features: norm_xy_velocity (nose-normalized, xy only, + velocity + acceleration)
  - Augmentation: mirror + rotation ONLY (augment search winner, 38.86% vs ~35.5%)
  - Data: unified_3plus (all sources, 3+ samples/label)

Pause/resume:
    touch data/optimal_combo.pause   — pause after current epoch
    Ctrl+C / SIGTERM                 — graceful stop
    Re-run to resume (not yet implemented — starts fresh)

Usage:
    .venv/bin/python tools/optimal_combo.py
    .venv/bin/python tools/optimal_combo.py --epochs 150
"""
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = DATA_DIR / "models"
PAUSE_PATH = DATA_DIR / "optimal_combo.pause"

_stop_requested = False


def _signal_handler(signum, frame):
    global _stop_requested
    _stop_requested = True
    name = signal.Signals(signum).name
    print(f"\n  [{name}] Stop requested — will finish after current epoch.")


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def main():
    from spj.trainer import (
        LabelEncoder, TrainingConfig, TrainingState, train_model,
    )

    # Parse args
    epochs = 100
    if "--epochs" in sys.argv:
        idx = sys.argv.index("--epochs")
        epochs = int(sys.argv[idx + 1])

    # Clear pause file
    if PAUSE_PATH.exists():
        PAUSE_PATH.unlink()
        print("Cleared pause file.")

    print("=" * 60)
    print("  OPTIMAL COMBO: Conv1D + norm_xy_vel + mirror+rotation")
    print("=" * 60)

    # Load unified 3plus splits
    splits_dir = DATA_DIR / "training" / "splits_unified_3plus"
    if not splits_dir.exists():
        print(f"ERROR: splits not found: {splits_dir}")
        print("Run overnight_conv1d.py first to generate unified splits.")
        sys.exit(1)

    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")

    all_labels = sorted(set(
        train_df["label"].astype(str).tolist() +
        val_df["label"].astype(str).tolist()
    ))
    label_encoder = LabelEncoder(all_labels)

    print(f"  Data: {len(train_df)} train, {len(val_df)} val, "
          f"{label_encoder.n_classes} classes")
    print(f"  Architecture: conv1d_transformer")
    print(f"  Features: norm_xy_velocity")
    print(f"  Augmentation: mirror + rotation ONLY (10x)")
    print(f"  Epochs: {epochs}")
    print()

    npz_dir = DATA_DIR / "training" / "export"

    config = TrainingConfig(
        epochs=epochs,
        batch_size=256,
        lr=0.0005,
        d_model=192,
        n_heads=4,
        n_layers=3,
        max_seq_len=300,
        augment=True,
        n_augments=10,
        patience=25,
        freeze_epochs=0,
        unfreeze_lr=3e-4,
        label_smoothing=0.1,
        weight_decay=1e-4,
        # Conv1D+Transformer + nose-normalized xy velocity
        model_type="conv1d_transformer",
        feature_mode="norm_xy_velocity",
        # Mirror + rotation ONLY (augment search winner)
        aug_temporal_crop=False,
        aug_speed=False,
        aug_noise=False,
        aug_scale=False,
        aug_mirror=True,
        aug_rotation=True,
        aug_joint_dropout=False,
        aug_temporal_mask=False,
    )

    state = TrainingState()
    output_dir = MODELS_DIR / "optimal_combo"
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_ckpt = output_dir / "resume_checkpoint.pt"

    if resume_ckpt.exists():
        print(f"  Resuming from {resume_ckpt}")

    # Monitor thread
    def _monitor():
        last_epoch = 0
        while not state.finished:
            if _stop_requested or PAUSE_PATH.exists():
                state.stop_requested = True
            if state.epoch > last_epoch:
                last_epoch = state.epoch
                vacc = state.val_accs[-1] if state.val_accs else 0
                tloss = state.train_losses[-1] if state.train_losses else 0
                best = max(state.val_accs) if state.val_accs else 0
                print(f"  Epoch {last_epoch}/{epochs} — "
                      f"val={vacc:.4f} best={best:.4f} loss={tloss:.4f}")
            time.sleep(10)

    monitor = threading.Thread(target=_monitor, daemon=True)
    monitor.start()

    t0 = time.time()
    train_model(
        train_df, val_df, npz_dir, label_encoder, config, state,
        output_dir, resume_path=resume_ckpt,
    )
    elapsed = time.time() - t0

    best_acc = max(state.val_accs) if state.val_accs else 0
    best_ep = state.val_accs.index(best_acc) + 1 if state.val_accs else 0

    print(f"\n{'=' * 60}")
    if state.error:
        print(f"  ERROR: {state.error}")
    else:
        print(f"  DONE in {elapsed / 60:.1f} min")
        print(f"  Best val acc: {best_acc:.4f} at epoch {best_ep}")

    # Rename checkpoint
    best_path = output_dir / "best_model.pt"
    if best_path.exists():
        final_name = f"optimal_conv1d_mirror_rot_ep{best_ep}_acc{best_acc:.4f}.pt"
        final_path = MODELS_DIR / final_name
        best_path.rename(final_path)
        print(f"  Saved: {final_name}")

    print("=" * 60)


if __name__ == "__main__":
    main()
