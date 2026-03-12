"""Conv1D category→word transfer learning.

Two-phase approach (mirrors the original transfer_cat2word success):
  Phase 1: Train Conv1D category model (116 kodifikácia categories)
  Phase 2: Transfer encoder → word-level Conv1D (unified data, 3+ samples)

This bridges the architecture gap: old category model (transformer, input_dim=288)
can't transfer to Conv1D (input_dim=576). Training a Conv1D category model first
restores the +48% relative gain from supervised transfer learning.

Usage:
    nohup .venv/bin/python tools/train_conv1d_transfer.py > data/train_conv1d_transfer.log 2>&1 &
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
from sklearn.model_selection import train_test_split

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


def _monitor(name, state, config):
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


def build_category_data():
    """Build category dataset from kodifikácia video→category mapping."""
    vid_dir = DATA_DIR / "videos" / "kodifikacia"
    export_dir = DATA_DIR / "training" / "export"

    # Map video stems to categories (parent folder name)
    categories = {}
    for cat_dir in sorted(vid_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        for vid in cat_dir.glob("*.mp4"):
            categories[vid.stem] = cat_dir.name

    # Find matching NPZ segments
    rows = []
    for stem, cat in categories.items():
        for npz in export_dir.glob(f"{stem}_*.npz"):
            rows.append({
                "npz_path": str(npz),
                "segment_id": npz.stem,
                "label": cat,
            })

    df = pd.DataFrame(rows)
    print(f"  Category data: {len(df)} segments, {df['label'].nunique()} categories")

    # Filter to categories with 3+ samples
    counts = df["label"].value_counts()
    valid = counts[counts >= 3].index
    df = df[df["label"].isin(valid)].copy()
    print(f"  After 3+ filter: {len(df)} segments, {df['label'].nunique()} categories")

    # Stratified split — handle small classes gracefully
    try:
        train_df, rest_df = train_test_split(
            df, test_size=0.3, stratify=df["label"], random_state=42,
        )
    except ValueError:
        train_df, rest_df = train_test_split(df, test_size=0.3, random_state=42)
    try:
        val_df, test_df = train_test_split(
            rest_df, test_size=0.5, stratify=rest_df["label"], random_state=42,
        )
    except ValueError:
        val_df, test_df = train_test_split(rest_df, test_size=0.5, random_state=42)

    # Save splits for reproducibility
    splits_dir = DATA_DIR / "training" / "splits_conv1d_category"
    splits_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)

    print(f"  Split: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
    return train_df, val_df


def run_phase(name, train_df, val_df, config, output_dir, pretrained_path=None):
    """Run a single training phase."""
    from spj.trainer import (
        LabelEncoder, TrainingState, train_model,
    )

    labels = sorted(set(train_df["label"].tolist() + val_df["label"].tolist()))
    label_encoder = LabelEncoder(labels)
    npz_dir = DATA_DIR / "training" / "export"

    state = TrainingState()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  [{name}] {len(train_df)} train, {len(val_df)} val, "
          f"{label_encoder.n_classes} classes")
    print(f"  Conv1D + norm_xy_velocity + mirror_rotation_only")
    if pretrained_path:
        print(f"  Transfer from: {pretrained_path.name}")
    print(f"{'='*60}")

    t0 = time.time()

    monitor = threading.Thread(
        target=_monitor, args=(name, state, config), daemon=True,
    )
    monitor.start()

    train_model(
        train_df, val_df, npz_dir, label_encoder, config, state,
        output_dir, pretrained_path=pretrained_path,
    )

    elapsed = time.time() - t0
    best_acc = max(state.val_accs) if state.val_accs else 0
    best_ep = state.val_accs.index(best_acc) + 1 if state.val_accs else 0

    print(f"\n  [{name}] FINISHED in {elapsed/60:.1f} min")
    if state.error:
        print(f"  [{name}] ERROR: {state.error}")
        return None

    print(f"  [{name}] Best val acc: {best_acc:.4f} at epoch {best_ep}")

    best_path = output_dir / "best_model.pt"
    if best_path.exists():
        final_name = f"conv1d_{name}_ep{best_ep}_acc{best_acc:.4f}.pt"
        final_path = MODELS_DIR / final_name
        import shutil
        shutil.copy2(best_path, final_path)
        print(f"  [{name}] Saved: {final_name}")
        return final_path

    return None


def main():
    from spj.trainer import TrainingConfig

    print("=" * 60)
    print("  Conv1D Category→Word Transfer Learning")
    print("=" * 60)

    t_start = time.time()

    # ── Phase 1: Category model ─────────────────────────────
    print("\n▶ Phase 1: Conv1D category model (kodifikácia categories)")
    train_df, val_df = build_category_data()

    cat_config = TrainingConfig(
        epochs=80,
        batch_size=512,
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
        # Mirror + rotation only (augment search winner for spatial)
        aug_mirror=True,
        aug_rotation=True,
        aug_temporal_crop=False,
        aug_speed=False,
        aug_noise=False,
        aug_scale=False,
        aug_joint_dropout=False,
        aug_temporal_mask=False,
    )

    cat_ckpt = run_phase(
        "category", train_df, val_df, cat_config,
        MODELS_DIR / "conv1d_category",
    )

    if _stop_requested or cat_ckpt is None:
        print("\n  Phase 1 failed or stopped. Cannot proceed to Phase 2.")
        return

    # ── Phase 2: Transfer to word-level ─────────────────────
    print(f"\n▶ Phase 2: Transfer category encoder → word-level")
    print(f"  Using category checkpoint: {cat_ckpt.name}")

    word_splits = DATA_DIR / "training" / "splits_unified_3plus"
    word_train = pd.read_csv(word_splits / "train.csv")
    word_val = pd.read_csv(word_splits / "val.csv")

    word_config = TrainingConfig(
        epochs=100,
        batch_size=512,
        lr=0.001,  # Higher lr for Phase 1 (frozen encoder)
        d_model=192,
        n_heads=4,
        n_layers=3,
        max_seq_len=300,
        augment=True,
        n_augments=10,
        patience=25,
        freeze_epochs=10,  # Phase 1: freeze encoder, train classifier
        unfreeze_lr=3e-4,  # Phase 2: unfreeze all, lower lr
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

    word_ckpt = run_phase(
        "word_transfer", word_train, word_val, word_config,
        MODELS_DIR / "conv1d_word_transfer",
        pretrained_path=cat_ckpt,
    )

    if word_ckpt:
        print(f"\n  ★ Word transfer model: {word_ckpt.name}")

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  All done in {total/3600:.1f} hours")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
