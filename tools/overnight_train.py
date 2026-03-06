"""Overnight training: two models with category transfer + augmentation.

Run 1: 3+ labels (4,844 classes) — higher per-class accuracy
Run 2: 2+ labels (5,352 classes) — broader coverage

Both use:
  - Category transfer from cat_v2_ep55_acc0.4493.pt
  - 10x data augmentation
  - Two-phase fine-tuning (10 frozen + 90 unfrozen)
  - Cosine LR schedule + patience=25

Usage:
    .venv/bin/python tools/overnight_train.py
"""
import os
import sys
import time
from pathlib import Path

# Force unbuffered output for nohup
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from spj.trainer import (
    LabelEncoder,
    PoseTransformerEncoder,
    TrainingConfig,
    TrainingState,
    train_model,
)

DATA_DIR = Path(__file__).parent.parent / "data"
EXPORT_DIR = DATA_DIR / "training" / "export"
MODELS_DIR = DATA_DIR / "models"
PRETRAINED = MODELS_DIR / "cat_v2_ep55_acc0.4493.pt"

RUNS = [
    {
        "name": "dualview_3plus",
        "splits_dir": DATA_DIR / "training" / "splits_3plus",
        "description": "3+ samples, 4844 labels, category transfer + 10x augment",
    },
    {
        "name": "dualview_2plus",
        "splits_dir": DATA_DIR / "training" / "splits_2plus",
        "description": "2+ samples, 5352 labels, category transfer + 10x augment",
    },
]


def run_training(name: str, splits_dir: Path, description: str):
    print(f"\n{'='*60}")
    print(f"  {name}: {description}")
    print(f"{'='*60}\n")

    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")

    # Derive label column
    for df in [train_df, val_df]:
        if "label" not in df.columns:
            df["label"] = df["reviewed_text"].where(
                df["reviewed_text"].str.strip() != "", df["text"]
            )

    all_labels = train_df["label"].dropna().astype(str).tolist()
    all_labels = [l for l in all_labels if l.strip()]
    label_encoder = LabelEncoder(all_labels)
    print(f"Labels: {label_encoder.n_classes}, Train: {len(train_df)}, Val: {len(val_df)}")

    config = TrainingConfig(
        backbone="from_scratch",
        lr=0.001,
        epochs=100,
        batch_size=256,
        d_model=256,
        n_heads=4,
        d_ff=512,
        n_layers=4,
        dropout=0.1,
        max_seq_len=300,
        weight_decay=1e-4,
        freeze_epochs=10,
        unfreeze_lr=3e-4,
        patience=25,
        label_smoothing=0.1,
        augment=True,
        n_augments=10,
    )

    state = TrainingState()
    t0 = time.time()

    import threading

    def _monitor():
        last_epoch = 0
        while state.running or not state.finished:
            if state.epoch > last_epoch:
                last_epoch = state.epoch
                vacc = state.val_accs[-1] if state.val_accs else 0
                tloss = state.train_losses[-1] if state.train_losses else 0
                phase = getattr(state, 'phase_description', '')
                print(f"  Epoch {last_epoch}/{config.epochs} — val_acc={vacc:.4f} train_loss={tloss:.4f} {phase}")
            time.sleep(5)

    monitor = threading.Thread(target=_monitor, daemon=True)
    monitor.start()

    train_model(
        train_df, val_df, EXPORT_DIR, label_encoder, config, state, MODELS_DIR,
        pretrained_path=PRETRAINED,
    )

    elapsed = time.time() - t0
    print(f"\n{name} finished in {elapsed/60:.1f} min")
    if state.error:
        print(f"  ERROR: {state.error}")
        return

    best_acc = max(state.val_accs) if state.val_accs else 0
    best_ep = state.val_accs.index(best_acc) + 1 if state.val_accs else 0
    print(f"  Best val acc: {best_acc:.4f} at epoch {best_ep}")
    print(f"  Final train loss: {state.train_losses[-1]:.4f}")

    # Rename best_model.pt to descriptive name
    best_path = MODELS_DIR / "best_model.pt"
    if best_path.exists():
        final_name = f"{name}_ep{best_ep}_acc{best_acc:.4f}.pt"
        final_path = MODELS_DIR / final_name
        best_path.rename(final_path)
        print(f"  Saved: {final_name}")


if __name__ == "__main__":
    if not PRETRAINED.exists():
        print(f"ERROR: Pretrained checkpoint not found: {PRETRAINED}")
        sys.exit(1)

    print(f"Overnight training — {len(RUNS)} runs")
    print(f"Pretrained: {PRETRAINED.name}")
    print(f"Augmentation: 10x")
    print(f"Device: MPS (Metal GPU)")

    t_start = time.time()
    for run in RUNS:
        run_training(**run)

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"All done in {total/3600:.1f} hours")
    print(f"Checkpoints in: {MODELS_DIR}")
