"""Overnight parallel training: Conv1D+Transformer on all data sources.

Runs 3 experiments IN PARALLEL using threads (MPS GPU shared):
  1. unified_2plus — all sources combined, 2+ samples/label
  2. unified_3plus — all sources combined, 3+ samples/label
  3. reference-dict_only  — reference-dict only (largest single source, 2+)

All use conv1d_transformer + norm_xy_velocity (9x better than baseline).
Category transfer from cat_v2_ep55.

Usage:
    nohup .venv/bin/python tools/overnight_conv1d.py > data/overnight_conv1d.log 2>&1 &
"""
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

# Force unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(threadName)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = DATA_DIR / "models"
PRETRAINED = MODELS_DIR / "cat_v2_ep55_acc0.4493.pt"

MIN_SAMPLES_2PLUS = 2
MIN_SAMPLES_3PLUS = 3

# Source manifests
SOURCES = {
    "partner-dict": DATA_DIR / "training" / "manifest_partner-dict.csv",
    "reference-dict": DATA_DIR / "training" / "manifest_reference-dict.csv",
    "climate-vocab": DATA_DIR / "training" / "manifest_climate-vocab.csv",
    "art-vocab": DATA_DIR / "training" / "manifest_art-vocab.csv",
    "fin-vocab": DATA_DIR / "training" / "manifest_fin-vocab.csv",
    "career-vocab": DATA_DIR / "training" / "manifest_career-vocab.csv",
}
EXISTING_SPLITS = DATA_DIR / "training" / "splits"

# Global stop flag
_stop_requested = False


def _sigint_handler(signum, frame):
    global _stop_requested
    if _stop_requested:
        print("\n  Aborting immediately.")
        os._exit(1)
    print("\n  Ctrl+C: will stop all runs after current epoch...")
    _stop_requested = True


signal.signal(signal.SIGINT, _sigint_handler)


def load_all_sources() -> pd.DataFrame:
    """Load and combine all source manifests."""
    dfs = []

    for source, path in SOURCES.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "source" not in df.columns:
            df["source"] = source
        dfs.append(df)
        print(f"  {source}: {len(df)} samples, {df['label'].nunique()} labels")

    # Add existing alignment-based splits
    for split in ["train.csv", "val.csv", "test.csv"]:
        p = EXISTING_SPLITS / split
        if p.exists():
            df = pd.read_csv(p)
            df["source"] = "existing"
            dfs.append(df)
    if any(d.get("source", pd.Series()).eq("existing").any() for d in dfs if "source" in d.columns):
        existing_total = sum(len(d) for d in dfs if d.get("source", pd.Series()).eq("existing").any())
        print(f"  existing: {existing_total} samples")

    combined = pd.concat(dfs, ignore_index=True)
    # Normalize labels
    combined["label"] = combined["label"].astype(str).str.lower().str.strip()
    return combined


def filter_min_samples(df: pd.DataFrame, min_samples: int) -> pd.DataFrame:
    counts = df["label"].value_counts()
    valid = counts[counts >= min_samples].index
    return df[df["label"].isin(valid)].copy()


def split_data(df: pd.DataFrame) -> tuple:
    from sklearn.model_selection import train_test_split

    n_labels = df["label"].nunique()

    try:
        train_df, rest_df = train_test_split(
            df, test_size=0.3,
            stratify=df["label"] if len(df) * 0.3 >= n_labels else None,
            random_state=42,
        )
        rest_labels = rest_df["label"].nunique()
        val_df, test_df = train_test_split(
            rest_df, test_size=0.5,
            stratify=rest_df["label"] if len(rest_df) * 0.5 >= rest_labels else None,
            random_state=42,
        )
    except ValueError:
        train_df, rest_df = train_test_split(df, test_size=0.3, random_state=42)
        val_df, test_df = train_test_split(rest_df, test_size=0.5, random_state=42)

    return train_df, val_df, test_df


def run_experiment(name: str, train_df: pd.DataFrame, val_df: pd.DataFrame,
                   n_labels: int, output_dir: Path):
    """Run a single training experiment (designed to run in a thread)."""
    from spj.trainer import (
        LabelEncoder, TrainingConfig, TrainingState, train_model,
    )

    label_encoder = LabelEncoder(
        sorted(set(train_df["label"].tolist() + val_df["label"].tolist()))
    )

    # Determine npz_dir (fallback — npz_path column has absolute paths)
    npz_dir = DATA_DIR / "training" / "export"

    # Train from scratch — old models use different architecture (can't transfer)
    config = TrainingConfig(
        epochs=100,
        batch_size=256,
        lr=0.0005,
        d_model=192,
        n_heads=4,
        n_layers=3,
        max_seq_len=300,
        augment=True,
        n_augments=10,
        patience=25,
        freeze_epochs=0,  # No freeze — training from scratch
        unfreeze_lr=3e-4,
        label_smoothing=0.1,
        weight_decay=1e-4,
        # Conv1D+Transformer + nose-normalized xy velocity
        model_type="conv1d_transformer",
        feature_mode="norm_xy_velocity",
    )

    state = TrainingState()
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_ckpt = output_dir / "resume_checkpoint.pt"

    if resume_ckpt.exists():
        print(f"[{name}] Resuming from {resume_ckpt}")

    print(f"\n[{name}] Starting: {len(train_df)} train, {len(val_df)} val, "
          f"{label_encoder.n_classes} classes")
    print(f"[{name}] Model: conv1d_transformer, feature: norm_xy_velocity, "
          f"d_model={config.d_model}, from scratch")

    t0 = time.time()

    # Monitor thread
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
        output_dir, resume_path=resume_ckpt,
    )

    elapsed = time.time() - t0
    best_acc = max(state.val_accs) if state.val_accs else 0
    best_ep = state.val_accs.index(best_acc) + 1 if state.val_accs else 0

    print(f"\n[{name}] FINISHED in {elapsed/60:.1f} min")
    if state.error:
        print(f"  [{name}] ERROR: {state.error}")
        return

    print(f"  [{name}] Best val acc: {best_acc:.4f} at epoch {best_ep}")

    # Rename checkpoint
    best_path = output_dir / "best_model.pt"
    if best_path.exists():
        final_name = f"conv1d_{name}_ep{best_ep}_acc{best_acc:.4f}.pt"
        final_path = MODELS_DIR / final_name
        best_path.rename(final_path)
        print(f"  [{name}] Saved: {final_name}")


def main():
    print("=" * 60)
    print("  Overnight Conv1D+Transformer Training (Parallel)")
    print("=" * 60)
    print(f"  Architecture: conv1d_transformer + norm_xy_velocity")
    print(f"  Transfer: {PRETRAINED.name}")
    print(f"  Augmentation: 10x")
    print()

    # Load all data
    print("Loading sources...")
    all_data = load_all_sources()
    print(f"\nTotal: {len(all_data)} samples, {all_data['label'].nunique()} labels")

    # Prepare experiments
    experiments = []

    # 1. Unified 2+ (all sources, 2+ samples/label)
    df_2plus = filter_min_samples(all_data, MIN_SAMPLES_2PLUS)
    if len(df_2plus) >= 20:
        train_df, val_df, test_df = split_data(df_2plus)
        # Save test split for later evaluation
        test_dir = DATA_DIR / "training" / "splits_unified_2plus"
        test_dir.mkdir(parents=True, exist_ok=True)
        test_df.to_csv(test_dir / "test.csv", index=False)
        train_df.to_csv(test_dir / "train.csv", index=False)
        val_df.to_csv(test_dir / "val.csv", index=False)
        experiments.append({
            "name": "unified_2plus",
            "train_df": train_df,
            "val_df": val_df,
            "n_labels": df_2plus["label"].nunique(),
            "output_dir": MODELS_DIR / "conv1d_unified_2plus",
        })
        print(f"\nExperiment 1: unified_2plus — {len(df_2plus)} samples, "
              f"{df_2plus['label'].nunique()} labels")

    # 2. Unified 3+ (all sources, 3+ samples/label)
    df_3plus = filter_min_samples(all_data, MIN_SAMPLES_3PLUS)
    if len(df_3plus) >= 20:
        train_df, val_df, test_df = split_data(df_3plus)
        test_dir = DATA_DIR / "training" / "splits_unified_3plus"
        test_dir.mkdir(parents=True, exist_ok=True)
        test_df.to_csv(test_dir / "test.csv", index=False)
        train_df.to_csv(test_dir / "train.csv", index=False)
        val_df.to_csv(test_dir / "val.csv", index=False)
        experiments.append({
            "name": "unified_3plus",
            "train_df": train_df,
            "val_df": val_df,
            "n_labels": df_3plus["label"].nunique(),
            "output_dir": MODELS_DIR / "conv1d_unified_3plus",
        })
        print(f"Experiment 2: unified_3plus — {len(df_3plus)} samples, "
              f"{df_3plus['label'].nunique()} labels")

    # 3. reference-dict only 2+ (largest single source)
    reference-dict_path = SOURCES["reference-dict"]
    if reference-dict_path.exists():
        reference-dict_df = pd.read_csv(reference-dict_path)
        reference-dict_df["label"] = reference-dict_df["label"].astype(str).str.lower().str.strip()
        reference-dict_2plus = filter_min_samples(reference-dict_df, MIN_SAMPLES_2PLUS)
        if len(reference-dict_2plus) >= 20:
            train_df, val_df, test_df = split_data(reference-dict_2plus)
            test_dir = DATA_DIR / "training" / "splits_reference-dict_2plus"
            test_dir.mkdir(parents=True, exist_ok=True)
            test_df.to_csv(test_dir / "test.csv", index=False)
            train_df.to_csv(test_dir / "train.csv", index=False)
            val_df.to_csv(test_dir / "val.csv", index=False)
            experiments.append({
                "name": "reference-dict_2plus",
                "train_df": train_df,
                "val_df": val_df,
                "n_labels": reference-dict_2plus["label"].nunique(),
                "output_dir": MODELS_DIR / "conv1d_reference-dict_2plus",
            })
            print(f"Experiment 3: reference-dict_2plus — {len(reference-dict_2plus)} samples, "
                  f"{reference-dict_2plus['label'].nunique()} labels")

    if not experiments:
        print("No experiments to run!")
        return

    print(f"\n{'='*60}")
    print(f"  Launching {len(experiments)} experiments in parallel")
    print(f"{'='*60}")

    t_start = time.time()

    # Launch all experiments as threads (MPS GPU is shared via Metal)
    threads = []
    for exp in experiments:
        t = threading.Thread(
            target=run_experiment,
            kwargs=exp,
            name=exp["name"],
            daemon=False,
        )
        threads.append(t)
        t.start()
        time.sleep(2)  # Stagger starts to avoid simultaneous data loading

    # Wait for all to finish
    for t in threads:
        t.join()

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  All {len(experiments)} experiments done in {total/3600:.1f} hours")
    print(f"  Checkpoints in: {MODELS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
