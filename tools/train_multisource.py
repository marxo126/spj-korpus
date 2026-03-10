"""Train per-source and unified models from all reference data.

Phase 2: Per-source models (partner-dict, reference-dict, existing)
Phase 3: Unified combined model

Usage:
    .venv/bin/python tools/train_multisource.py --phase 2  # per-source
    .venv/bin/python tools/train_multisource.py --phase 3  # unified
    .venv/bin/python tools/train_multisource.py --phase all
    .venv/bin/python tools/train_multisource.py --source partner-dict  # single source

Press Ctrl+C once to stop after current epoch (graceful).
Press Ctrl+C twice to abort immediately.
"""
import logging
import os
import signal
import sys
import time
from pathlib import Path

# Flush print output immediately
os.environ["PYTHONUNBUFFERED"] = "1"

# Configure logging to stdout so trainer.py progress is visible
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = DATA_DIR / "models"

# Minimum samples per label for training
MIN_SAMPLES = 2

# Global state for graceful stop
_active_state = None  # TrainingState of current training run
_stop_count = 0


def _sigint_handler(signum, frame):
    """Handle Ctrl+C: first press = graceful stop, second = abort."""
    global _stop_count
    _stop_count += 1
    if _stop_count == 1 and _active_state is not None:
        print("\n  ⏸ Ctrl+C: stopping after current epoch... (press again to abort)")
        _active_state.stop_requested = True
    else:
        print("\n  Aborting immediately.")
        sys.exit(1)


signal.signal(signal.SIGINT, _sigint_handler)

# Sources and their manifest files
SOURCES = {
    "partner-dict": DATA_DIR / "training" / "manifest_partner-dict.csv",
    "reference-dict": DATA_DIR / "training" / "manifest_reference-dict.csv",
    "climate-vocab": DATA_DIR / "training" / "manifest_climate-vocab.csv",
    "art-vocab": DATA_DIR / "training" / "manifest_art-vocab.csv",
    "fin-vocab": DATA_DIR / "training" / "manifest_fin-vocab.csv",
    "career-vocab": DATA_DIR / "training" / "manifest_career-vocab.csv",
    # Existing sources use the alignment-based pipeline
    "existing": DATA_DIR / "training" / "splits",  # has train.csv, val.csv, test.csv
}


def load_source_manifest(source: str) -> pd.DataFrame | None:
    """Load manifest for a source. Returns None if not available."""
    path = SOURCES.get(source)
    if not path:
        return None

    if source == "existing":
        # Combine existing train/val/test splits
        dfs = []
        for split in ["train.csv", "val.csv", "test.csv"]:
            p = path / split
            if p.exists():
                df = pd.read_csv(p)
                df["source"] = "existing"
                dfs.append(df)
        return pd.concat(dfs, ignore_index=True) if dfs else None

    if not path.exists():
        return None
    return pd.read_csv(path)


def filter_by_min_samples(df: pd.DataFrame, min_samples: int = MIN_SAMPLES) -> pd.DataFrame:
    """Keep only labels with >= min_samples."""
    counts = df["label"].value_counts()
    valid = counts[counts >= min_samples].index
    return df[df["label"].isin(valid)].copy()


def split_data(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15) -> tuple:
    """Split data into train/val/test. Stratified when possible, random otherwise."""
    from sklearn.model_selection import train_test_split

    n_labels = df["label"].nunique()
    test_size = 1 - train_ratio
    can_stratify = int(len(df) * test_size) >= n_labels

    try:
        train_df, rest_df = train_test_split(
            df, test_size=test_size,
            stratify=df["label"] if can_stratify else None,
            random_state=42,
        )
        val_ratio_adj = val_ratio / (1 - train_ratio)
        n_rest_labels = rest_df["label"].nunique()
        can_stratify_rest = int(len(rest_df) * (1 - val_ratio_adj)) >= n_rest_labels

        val_df, test_df = train_test_split(
            rest_df, test_size=1 - val_ratio_adj,
            stratify=rest_df["label"] if can_stratify_rest else None,
            random_state=42,
        )
    except ValueError:
        # Fallback: random split
        train_df, rest_df = train_test_split(df, test_size=test_size, random_state=42)
        val_df, test_df = train_test_split(rest_df, test_size=0.5, random_state=42)

    return train_df, val_df, test_df


def train_source_model(source: str, df: pd.DataFrame) -> Path | None:
    """Train a model on a single source's data."""
    from spj.trainer import (
        LabelEncoder, TrainingConfig, TrainingState, train_model,
    )

    df = filter_by_min_samples(df)
    if len(df) < 10:
        print(f"  {source}: too few samples ({len(df)}), skipping")
        return None

    labels = sorted(df["label"].unique())
    print(f"  {source}: {len(df)} samples, {len(labels)} labels")

    # Split
    try:
        train_df, val_df, test_df = split_data(df)
    except ValueError as e:
        print(f"  {source}: split failed ({e}), skipping")
        return None

    print(f"  Split: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")

    # Setup
    label_encoder = LabelEncoder(labels)

    # Determine npz_dir — use first file's parent as fallback
    if "npz_path" in df.columns:
        npz_dir = Path(df.iloc[0]["npz_path"]).parent
    else:
        npz_dir = DATA_DIR / "training" / "export"

    # Use category transfer if available
    cat_checkpoint = MODELS_DIR / "cat_v2_ep55_acc0.4493.pt"
    pretrained = cat_checkpoint if cat_checkpoint.exists() else None

    config = TrainingConfig(
        epochs=100,
        batch_size=512,
        lr=0.001 if pretrained else 0.0005,
        d_model=128,
        n_heads=4,
        n_layers=3,
        max_seq_len=300,
        augment=True,
        n_augments=10,
        patience=25,
        # Two-phase: freeze encoder first if using pretrained
        freeze_epochs=10 if pretrained else 0,
        # New architecture: Conv1D+Transformer + nose-normalized xy velocity
        model_type="conv1d_transformer",
        feature_mode="norm_xy_velocity",
    )

    global _active_state, _stop_count
    state = TrainingState()
    _active_state = state
    _stop_count = 0
    output_dir = MODELS_DIR / f"source_{source}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Training {source} model... (pretrained: {pretrained is not None})")
    print(f"  Press Ctrl+C to stop gracefully after current epoch.")
    t0 = time.time()

    import os
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    train_model(
        train_df, val_df, npz_dir, label_encoder, config, state,
        output_dir, pretrained_path=pretrained,
    )

    _active_state = None
    elapsed = time.time() - t0
    stopped = " (stopped by user)" if state.stop_requested else ""
    print(f"  Done: {elapsed/60:.1f} min, best val acc: {state.best_val_acc:.4f}{stopped}")

    if state.error:
        print(f"  ERROR: {state.error}")
        return None

    return output_dir


def train_unified_model(sources: dict[str, pd.DataFrame]) -> Path | None:
    """Train unified model on all sources combined."""
    from spj.trainer import (
        LabelEncoder, TrainingConfig, TrainingState, train_model,
    )

    # Combine all sources
    dfs = []
    for source, df in sources.items():
        if df is not None and len(df) > 0:
            df = df.copy()
            if "source" not in df.columns:
                df["source"] = source
            dfs.append(df)

    if not dfs:
        print("No data for unified model")
        return None

    combined = pd.concat(dfs, ignore_index=True)

    # Normalize labels across sources (lowercase, strip)
    combined["label"] = combined["label"].str.lower().str.strip()

    # Filter by min samples (cross-source)
    combined = filter_by_min_samples(combined)

    labels = sorted(combined["label"].unique())
    print(f"  Unified: {len(combined)} samples, {len(labels)} labels")

    # Show cross-source overlap
    source_labels = {}
    for source in combined["source"].unique():
        source_labels[source] = set(combined[combined["source"] == source]["label"])
    for s1 in source_labels:
        for s2 in source_labels:
            if s1 < s2:
                overlap = source_labels[s1] & source_labels[s2]
                if overlap:
                    print(f"  Label overlap {s1}↔{s2}: {len(overlap)}")

    # Split
    try:
        train_df, val_df, test_df = split_data(combined)
    except ValueError as e:
        print(f"  Split failed ({e})")
        return None

    print(f"  Split: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")

    label_encoder = LabelEncoder(labels)

    # Use first source's NPZ dir as fallback (npz_path column handles the rest)
    npz_dir = DATA_DIR / "training" / "export"

    cat_checkpoint = MODELS_DIR / "cat_v2_ep55_acc0.4493.pt"
    pretrained = cat_checkpoint if cat_checkpoint.exists() else None

    config = TrainingConfig(
        epochs=100,
        batch_size=512,
        lr=0.001 if pretrained else 0.0005,
        d_model=128,
        n_heads=4,
        n_layers=3,
        max_seq_len=300,
        augment=True,
        n_augments=10,
        patience=25,
        freeze_epochs=10 if pretrained else 0,
        # New architecture: Conv1D+Transformer + nose-normalized xy velocity
        model_type="conv1d_transformer",
        feature_mode="norm_xy_velocity",
    )

    global _active_state, _stop_count
    state = TrainingState()
    _active_state = state
    _stop_count = 0
    output_dir = MODELS_DIR / "unified"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Training unified model... (pretrained: {pretrained is not None})")
    print(f"  Press Ctrl+C to stop gracefully after current epoch.")
    t0 = time.time()

    import os
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    train_model(
        train_df, val_df, npz_dir, label_encoder, config, state,
        output_dir, pretrained_path=pretrained,
    )

    _active_state = None
    elapsed = time.time() - t0
    stopped = " (stopped by user)" if state.stop_requested else ""
    print(f"  Done: {elapsed/60:.1f} min, best val acc: {state.best_val_acc:.4f}{stopped}")

    if state.error:
        print(f"  ERROR: {state.error}")

    return output_dir


def main():
    phase = "all"
    source_filter = None

    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--phase" and i < len(sys.argv) - 1:
            phase = sys.argv[i + 1]
        elif arg.startswith("--phase="):
            phase = arg.split("=", 1)[1]
        elif arg == "--source" and i < len(sys.argv) - 1:
            source_filter = sys.argv[i + 1]
        elif arg.startswith("--source="):
            source_filter = arg.split("=", 1)[1]

    print("=" * 60)
    print("  Multi-Source Training Pipeline")
    print("=" * 60)

    # Load all manifests
    all_data = {}
    for source in SOURCES:
        if source_filter and source != source_filter:
            continue
        df = load_source_manifest(source)
        if df is not None and len(df) > 0:
            all_data[source] = df
            labels = df["label"].nunique() if "label" in df.columns else 0
            print(f"  {source}: {len(df)} samples, {labels} labels")
        else:
            print(f"  {source}: no data")

    # Phase 2: Per-source models
    if phase in ("2", "all") and not source_filter:
        print(f"\n{'='*60}")
        print("  Phase 2: Per-Source Models")
        print(f"{'='*60}")
        for source, df in all_data.items():
            if source == "existing":
                print(f"\n  Skipping 'existing' (already trained)")
                continue
            print(f"\n--- {source} ---")
            train_source_model(source, df)

    elif source_filter:
        if source_filter in all_data:
            print(f"\n--- Training {source_filter} ---")
            train_source_model(source_filter, all_data[source_filter])
        else:
            print(f"No data for {source_filter}")

    # Phase 3: Unified model
    if phase in ("3", "all") and not source_filter:
        print(f"\n{'='*60}")
        print("  Phase 3: Unified Combined Model")
        print(f"{'='*60}")
        train_unified_model(all_data)

    print(f"\n{'='*60}")
    print("  Done")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
