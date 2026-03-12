"""Extended landmark experiment: mixed-preset training.

Re-exports .pose sources at extended preset (148 landmarks), keeps
posunky/dictio at compact (96 landmarks) with zero-padding to 148.
Tests if eye/eyebrow landmarks improve sign recognition accuracy.

Usage:
    nohup .venv/bin/python tools/train_extended.py > data/train_extended.log 2>&1 &
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

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = DATA_DIR / "models"
EXPORT_DIR = DATA_DIR / "training" / "export"
EXPORT_EXT_DIR = DATA_DIR / "training" / "export_extended"

_stop_requested = False


def _sigint_handler(signum, frame):
    global _stop_requested
    if _stop_requested:
        os._exit(1)
    print("\n  Ctrl+C: stopping after current epoch...")
    _stop_requested = True


signal.signal(signal.SIGINT, _sigint_handler)


def reexport_extended():
    """Re-export 'existing' source segments at extended preset (148 landmarks).

    Reads original .pose files, selects 148 landmark indices, saves to export_extended/.
    Only re-exports segments that are in the unified_3plus splits.
    """
    from spj.preannotate import load_pose_arrays
    from spj.training_data import (
        SL_LANDMARK_INDICES_EXTENDED,
    )

    EXPORT_EXT_DIR.mkdir(parents=True, exist_ok=True)

    # Load manifest to know which segments to export
    splits_dir = DATA_DIR / "training" / "splits_unified_3plus"
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")
    test_df = pd.read_csv(splits_dir / "test.csv") if (splits_dir / "test.csv").exists() else pd.DataFrame()
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Filter to "existing" source (has .pose files)
    existing = all_df[all_df["source"] == "existing"].copy()
    print(f"  Re-export: {len(existing)} 'existing' segments to extended preset")

    # Find .pose files for each segment
    pose_dirs = [
        DATA_DIR / "pose" / "kodifikacia",
        DATA_DIR / "pose" / "spreadthesign",
    ]

    idx = np.array(SL_LANDMARK_INDICES_EXTENDED)
    done = 0
    skipped = 0
    errors = 0

    for _, row in existing.iterrows():
        if _stop_requested:
            break

        seg_id = str(row["segment_id"])
        out_path = EXPORT_EXT_DIR / f"{seg_id}.npz"

        # Skip if already exported
        if out_path.exists():
            done += 1
            continue

        # Parse video stem from segment_id (e.g., "kolko_90_00000000" → "kolko_90")
        parts = seg_id.split("_")
        # Last part is frame index, rest is video stem
        stem = "_".join(parts[:-1])

        # Find .pose file
        pose_path = None
        for pd_dir in pose_dirs:
            candidate = pd_dir / f"{stem}.pose"
            if candidate.exists():
                pose_path = candidate
                break

        if pose_path is None:
            skipped += 1
            continue

        try:
            data, conf, _fps = load_pose_arrays(pose_path)
            # data: (T, 1, 543, 3), conf: (T, 1, 543, 1)
            data = data[:, 0, :, :]  # (T, 543, 3)
            conf = conf[:, 0, :, :]  # (T, 543, 1)

            # Extract segment time range from original compact NPZ
            orig_npz = EXPORT_DIR / f"{seg_id}.npz"
            if orig_npz.exists():
                orig = np.load(orig_npz)
                orig_T = orig["pose"].shape[0]
                # Original segment is a slice of the full video
                # We need to find which frames — check if metadata exists
                if "start_frame" in orig.files:
                    start = int(orig["start_frame"])
                    end = start + orig_T
                else:
                    # Use full video length matching
                    # The segment was exported with compact landmarks, T frames
                    # Just take first orig_T frames (segments are typically short)
                    start = 0
                    end = min(orig_T, data.shape[0])
            else:
                start = 0
                end = data.shape[0]

            seg_data = data[start:end, idx, :]  # (seg_T, 148, 3)
            seg_conf = conf[start:end, idx, :]  # (seg_T, 148, 1)

            np.savez_compressed(
                out_path,
                pose=seg_data.astype(np.float32),
                confidence=seg_conf.astype(np.float32),
                landmark_indices=idx,
            )
            done += 1

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error {seg_id}: {e}")

        if done % 1000 == 0 and done > 0:
            print(f"  Re-exported {done}/{len(existing)} segments...")

    print(f"  Re-export complete: {done} done, {skipped} skipped (no .pose), {errors} errors")
    return done


def build_mixed_splits():
    """Build splits with extended NPZ for .pose sources, compact for posunky/dictio."""
    splits_dir = DATA_DIR / "training" / "splits_unified_3plus"
    out_dir = DATA_DIR / "training" / "splits_extended_3plus"
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ["train.csv", "val.csv", "test.csv"]:
        src_path = splits_dir / split_name
        if not src_path.exists():
            continue

        df = pd.read_csv(src_path)
        original_len = len(df)

        # For "existing" source, point to extended NPZ if available
        def update_path(row):
            if row.get("source") != "existing":
                return row["npz_path"]
            seg_id = str(row["segment_id"])
            ext_path = EXPORT_EXT_DIR / f"{seg_id}.npz"
            if ext_path.exists():
                return str(ext_path)
            return row["npz_path"]  # fallback to compact

        df["npz_path"] = df.apply(update_path, axis=1)

        # Update n_landmarks for extended samples
        def update_landmarks(row):
            if str(row["npz_path"]).startswith(str(EXPORT_EXT_DIR)):
                return 148
            return row.get("n_landmarks", 96)

        df["n_landmarks"] = df.apply(update_landmarks, axis=1)

        df.to_csv(out_dir / split_name, index=False)
        n_ext = (df["n_landmarks"] == 148).sum()
        print(f"  {split_name}: {len(df)} samples ({n_ext} extended, {len(df) - n_ext} compact)")

    return out_dir


def run_training(splits_dir: Path, name: str):
    """Train Conv1D model on mixed-preset data."""
    from spj.trainer import (
        LabelEncoder, TrainingConfig, TrainingState, train_model,
    )

    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")

    labels = sorted(set(train_df["label"].tolist() + val_df["label"].tolist()))
    label_encoder = LabelEncoder(labels)
    npz_dir = DATA_DIR / "training" / "export"

    config = TrainingConfig(
        epochs=30,  # Quick probe first
        batch_size=512,
        lr=0.0005,
        d_model=192,
        n_heads=4,
        n_layers=3,
        max_seq_len=300,
        augment=True,
        n_augments=10,
        patience=15,
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
    output_dir = MODELS_DIR / f"extended_{name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  [{name}] {len(train_df)} train, {len(val_df)} val, "
          f"{label_encoder.n_classes} classes")
    print(f"  Conv1D + norm_xy_velocity + extended landmarks (148)")
    print(f"  target_n_landmarks=148 (compact samples padded)")
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
        output_dir, target_n_landmarks=148,
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
        import shutil
        final_name = f"extended_{name}_ep{best_ep}_acc{best_acc:.4f}.pt"
        final_path = MODELS_DIR / final_name
        shutil.copy2(best_path, final_path)
        print(f"  [{name}] Saved: {final_name}")


def main():
    print("=" * 60)
    print("  Extended Landmark Experiment (148 landmarks)")
    print("=" * 60)

    t_start = time.time()

    # Step 1: Re-export existing source at extended preset
    print("\n▶ Step 1: Re-export .pose sources at extended (148 landmarks)")
    n_exported = reexport_extended()
    if _stop_requested:
        return

    # Step 2: Build mixed splits
    print("\n▶ Step 2: Build mixed-preset splits")
    splits_dir = build_mixed_splits()
    if _stop_requested:
        return

    # Step 3: Train
    print("\n▶ Step 3: Train Conv1D on mixed-preset data (30-epoch probe)")
    run_training(splits_dir, "3plus_probe")

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  All done in {total/3600:.1f} hours")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
