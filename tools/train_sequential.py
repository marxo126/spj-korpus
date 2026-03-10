"""Sequential Conv1D training: one model at a time, full GPU.

Runs unified_3plus first (likely winner), then unified_2plus.
Each gets full MPS GPU + larger batch size = much faster per epoch.

Usage:
    nohup .venv/bin/python tools/train_sequential.py > data/train_sequential.log 2>&1 &
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

_stop_requested = False


def _sigint_handler(signum, frame):
    global _stop_requested
    if _stop_requested:
        os._exit(1)
    print("\n  Ctrl+C: stopping after current epoch...")
    _stop_requested = True


signal.signal(signal.SIGINT, _sigint_handler)


def run_experiment(name: str, splits_dir: Path, output_dir: Path):
    from spj.trainer import (
        LabelEncoder, TrainingConfig, TrainingState, train_model,
    )

    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")

    labels = sorted(set(train_df["label"].tolist() + val_df["label"].tolist()))
    label_encoder = LabelEncoder(labels)
    npz_dir = DATA_DIR / "training" / "export"

    config = TrainingConfig(
        epochs=100,
        batch_size=512,
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
        model_type="conv1d_transformer",
        feature_mode="norm_xy_velocity",
    )

    state = TrainingState()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  [{name}] {len(train_df)} train, {len(val_df)} val, "
          f"{label_encoder.n_classes} classes")
    print(f"  Conv1D+Transformer, norm_xy_velocity, batch=512, from scratch")
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
                print(f"  [{name}] Epoch {last_epoch}/{config.epochs} — "
                      f"val={vacc:.4f} best={best:.4f} loss={tloss:.4f}")
            time.sleep(10)

    monitor = threading.Thread(target=_monitor, daemon=True, name=f"mon-{name}")
    monitor.start()

    train_model(
        train_df, val_df, npz_dir, label_encoder, config, state,
        output_dir,
    )

    elapsed = time.time() - t0
    best_acc = max(state.val_accs) if state.val_accs else 0
    best_ep = state.val_accs.index(best_acc) + 1 if state.val_accs else 0

    print(f"\n  [{name}] FINISHED in {elapsed/60:.1f} min")
    if state.error:
        print(f"  [{name}] ERROR: {state.error}")
        return

    print(f"  [{name}] Best val acc: {best_acc:.4f} at epoch {best_ep}")

    best_path = output_dir / "best_model.pt"
    if best_path.exists():
        final_name = f"conv1d_{name}_ep{best_ep}_acc{best_acc:.4f}.pt"
        final_path = MODELS_DIR / final_name
        best_path.rename(final_path)
        print(f"  [{name}] Saved: {final_name}")


def main():
    print("=" * 60)
    print("  Sequential Conv1D Training (full GPU per model)")
    print("=" * 60)

    t_start = time.time()

    # Run unified_3plus first (faster, likely better)
    run_experiment(
        "unified_3plus",
        DATA_DIR / "training" / "splits_unified_3plus",
        MODELS_DIR / "conv1d_unified_3plus",
    )

    # Then unified_2plus (more classes, slower)
    if not _stop_requested:
        run_experiment(
            "unified_2plus",
            DATA_DIR / "training" / "splits_unified_2plus",
            MODELS_DIR / "conv1d_unified_2plus",
        )

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  All done in {total/3600:.1f} hours")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
