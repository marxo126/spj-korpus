"""Train Conv1D on 5+ samples/class subset for higher per-class accuracy.

Usage:
    nohup .venv/bin/python tools/train_5plus.py > data/train_5plus.log 2>&1 &
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
SPLITS_DIR = DATA_DIR / "training" / "splits_unified_5plus"
EXPORT_DIR = DATA_DIR / "training" / "export"

_stop_requested = False


def _sigint_handler(signum, frame):
    global _stop_requested
    if _stop_requested:
        os._exit(1)
    print("\n  Ctrl+C: stopping after current epoch...")
    _stop_requested = True


signal.signal(signal.SIGINT, _sigint_handler)


def run_training():
    from spj.trainer import (
        LabelEncoder, TrainingConfig, TrainingState, train_model,
    )

    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "val.csv")

    labels = sorted(set(train_df["label"].tolist() + val_df["label"].tolist()))
    label_encoder = LabelEncoder(labels)

    config = TrainingConfig(
        epochs=80,
        batch_size=256,
        lr=0.0005,
        d_model=192,
        n_heads=4,
        n_layers=3,
        max_seq_len=300,
        augment=True,
        n_augments=10,
        patience=20,
        freeze_epochs=0,
        label_smoothing=0.1,
        weight_decay=1e-4,
        model_type="conv1d_transformer",
        feature_mode="norm_xy_velocity",
        aug_mirror=True,
        aug_rotation=True,
        aug_temporal_crop=False,
        aug_speed=False,
        aug_noise=False,
        aug_scale=False,
        aug_joint_dropout=False,
        aug_temporal_mask=False,
    )

    state = TrainingState()
    output_dir = MODELS_DIR / "conv1d_5plus"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Conv1D 5+ samples/class")
    print(f"  {len(train_df)} train, {len(val_df)} val, {label_encoder.n_classes} classes")
    print(f"  Conv1D + norm_xy_velocity + mirror+rotation")
    print(f"{'='*60}")

    t0 = time.time()

    def _monitor():
        last_epoch = 0
        while not state.finished:
            if _stop_requested:
                state.stop_requested = True
            if state.epoch > last_epoch:
                last_epoch = state.epoch
                vacc = state.val_accs[-1] if state.val_accs else 0
                tloss = state.train_losses[-1] if state.train_losses else 0
                best = max(state.val_accs) if state.val_accs else 0
                print(f"  Epoch {last_epoch}/{config.epochs} — "
                      f"val={vacc:.4f} best={best:.4f} loss={tloss:.4f}")
            time.sleep(10)

    monitor = threading.Thread(target=_monitor, daemon=True)
    monitor.start()

    train_model(
        train_df, val_df, EXPORT_DIR, label_encoder, config, state,
        output_dir,
    )

    elapsed = time.time() - t0
    best_acc = max(state.val_accs) if state.val_accs else 0
    best_ep = state.val_accs.index(best_acc) + 1 if state.val_accs else 0

    print(f"\n  FINISHED in {elapsed/60:.1f} min")
    if state.error:
        print(f"  ERROR: {state.error}")
        return

    print(f"  Best val acc: {best_acc:.4f} at epoch {best_ep}")

    best_path = output_dir / "best_model.pt"
    if best_path.exists():
        import shutil
        final_name = f"conv1d_5plus_ep{best_ep}_acc{best_acc:.4f}.pt"
        final_path = MODELS_DIR / final_name
        shutil.copy2(best_path, final_path)
        print(f"  Saved: {final_name}")


if __name__ == "__main__":
    print("=" * 60)
    print("  Conv1D 5+ Samples/Class Training")
    print("=" * 60)
    run_training()
