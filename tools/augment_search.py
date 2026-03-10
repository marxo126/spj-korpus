"""Augmentation search: greedy hill-climbing over augmentation combos.

Trains short runs (30 epochs) with different augmentation combinations.
Keeps the best checkpoint as baseline, then tries more combos.
Final winner gets a full 100-epoch training run.

Results logged to data/augment_search_log.json.

Pause/resume:
    touch data/augment_search.pause   — pauses after current probe finishes
    rm data/augment_search.pause      — allow resume
    Ctrl+C / SIGTERM                  — same as pause (graceful stop)
    Re-run the script to resume       — skips completed probes automatically

Usage:
    .venv/bin/python tools/augment_search.py
    .venv/bin/python tools/augment_search.py --quick   # 15-epoch probe runs
    .venv/bin/python tools/augment_search.py --full     # skip search, full train best combo
"""
import json
import signal
import sys
import threading
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from spj.trainer import (
    LabelEncoder,
    TrainingConfig,
    TrainingState,
    train_model,
)

DATA_DIR = Path(__file__).parent.parent / "data"
EXPORT_DIR = DATA_DIR / "training" / "export"
MODELS_DIR = DATA_DIR / "models"
SPLITS_DIR = DATA_DIR / "training" / "splits_2plus"  # 5,352 labels — broadest coverage
PRETRAINED = MODELS_DIR / "cat_v2_ep55_acc0.4493.pt"
LOG_PATH = DATA_DIR / "augment_search_log.json"
PAUSE_PATH = DATA_DIR / "augment_search.pause"

# Graceful shutdown flag — set by SIGINT/SIGTERM or pause file
_pause_requested = False


def _signal_handler(signum, frame):
    global _pause_requested
    _pause_requested = True
    name = signal.Signals(signum).name
    print(f"\n  [{name}] Pause requested — will stop after current probe finishes.")
    print(f"  Run script again to resume.")


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def is_pause_requested() -> bool:
    """Check if pause was requested via signal or pause file."""
    return _pause_requested or PAUSE_PATH.exists()


# Base config — same as overnight_train.py proven settings
BASE_CONFIG = dict(
    backbone="from_scratch",
    lr=0.001,
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
    patience=15,
    label_smoothing=0.1,
    augment=True,
    n_augments=10,
)

# Augmentation flag names
AUG_FLAGS = [
    "aug_temporal_crop", "aug_speed", "aug_noise", "aug_scale",
    "aug_mirror", "aug_rotation", "aug_joint_dropout", "aug_temporal_mask",
]

# Search combos — each is a dict of aug flags to override (default=True)
SEARCH_COMBOS = [
    # 0: Current baseline (original 4 augmentations only)
    {
        "name": "baseline_original4",
        "aug_mirror": False, "aug_rotation": False,
        "aug_joint_dropout": False, "aug_temporal_mask": False,
    },
    # 1: All 8 augmentations enabled
    {"name": "all8"},
    # 2: Original 4 + mirror only
    {
        "name": "original4_plus_mirror",
        "aug_rotation": False, "aug_joint_dropout": False, "aug_temporal_mask": False,
    },
    # 3: Original 4 + rotation only
    {
        "name": "original4_plus_rotation",
        "aug_mirror": False, "aug_joint_dropout": False, "aug_temporal_mask": False,
    },
    # 4: Original 4 + mirror + rotation
    {
        "name": "original4_plus_mirror_rotation",
        "aug_joint_dropout": False, "aug_temporal_mask": False,
    },
    # 5: Original 4 + dropout + mask (robustness pair)
    {
        "name": "original4_plus_dropout_mask",
        "aug_mirror": False, "aug_rotation": False,
    },
    # 6: Mirror + rotation + dropout (no temporal mask)
    {"name": "new3_no_tmask", "aug_temporal_mask": False},
    # 7: Mirror + rotation + temporal mask (no dropout)
    {"name": "new3_no_dropout", "aug_joint_dropout": False},
    # 8: All new, disable temporal crop (crop may hurt short signs)
    {"name": "all_new_no_crop", "aug_temporal_crop": False},
    # 9: All new, disable speed variation
    {"name": "all_new_no_speed", "aug_speed": False},
    # 10: Mirror + rotation only (no original augments)
    {
        "name": "mirror_rotation_only",
        "aug_temporal_crop": False, "aug_speed": False,
        "aug_noise": False, "aug_scale": False,
        "aug_joint_dropout": False, "aug_temporal_mask": False,
    },
    # 11: Heavier augmentation (20x)
    {"name": "all8_20x", "n_augments": 20},
    # 12: Lighter augmentation (5x) + all 8
    {"name": "all8_5x", "n_augments": 5},
]


def load_data():
    """Load train/val splits and build label encoder."""
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "val.csv")
    for df in [train_df, val_df]:
        if "label" not in df.columns:
            df["label"] = df["reviewed_text"].where(
                df["reviewed_text"].str.strip() != "", df["text"]
            )
    all_labels = train_df["label"].dropna().astype(str).tolist()
    all_labels = [l for l in all_labels if l.strip()]
    label_encoder = LabelEncoder(all_labels)
    print(f"Data: {len(train_df)} train, {len(val_df)} val, {label_encoder.n_classes} classes")
    return train_df, val_df, label_encoder


def _start_monitor(name: str, state: TrainingState, total_epochs: int,
                    interval: float = 5) -> threading.Thread:
    """Start a daemon thread that prints epoch progress."""
    last_report = [0]

    def _monitor():
        while state.running or not state.finished:
            if state.epoch > last_report[0]:
                last_report[0] = state.epoch
                vacc = state.val_accs[-1] if state.val_accs else 0
                tloss = state.train_losses[-1] if state.train_losses else 0
                print(f"  [{name}] Epoch {last_report[0]}/{total_epochs}"
                      f" — val_acc={vacc:.4f} train_loss={tloss:.4f}")
            time.sleep(interval)

    t = threading.Thread(target=_monitor, daemon=True)
    t.start()
    return t


def make_config(combo: dict, probe_epochs: int) -> TrainingConfig:
    """Build TrainingConfig from base + combo overrides."""
    params = {**BASE_CONFIG, "epochs": probe_epochs}
    # Apply augmentation flag overrides (default all True)
    for flag in AUG_FLAGS:
        params[flag] = combo.get(flag, True)
    if "n_augments" in combo:
        params["n_augments"] = combo["n_augments"]
    return TrainingConfig(**params)


def run_probe(name: str, combo: dict, train_df, val_df, label_encoder,
              probe_epochs: int) -> dict:
    """Run a short training probe and return results."""
    config = make_config(combo, probe_epochs)
    state = TrainingState()

    print(f"\n{'='*60}")
    print(f"  PROBE: {name}")
    flags = {f: combo.get(f, True) for f in AUG_FLAGS}
    n_aug = combo.get("n_augments", BASE_CONFIG["n_augments"])
    print(f"  Flags: {flags}")
    print(f"  n_augments: {n_aug}, epochs: {probe_epochs}")
    print(f"{'='*60}")

    _start_monitor(name, state, probe_epochs, interval=5)

    resume_ckpt = MODELS_DIR / f"resume_probe_{name}.pt"
    t0 = time.time()
    train_model(
        train_df, val_df, EXPORT_DIR, label_encoder, config, state, MODELS_DIR,
        pretrained_path=PRETRAINED, resume_path=resume_ckpt,
    )
    # Clean up resume checkpoint after successful probe
    if resume_ckpt.exists():
        resume_ckpt.unlink()
    elapsed = time.time() - t0

    best_acc = max(state.val_accs) if state.val_accs else 0
    best_ep = state.val_accs.index(best_acc) + 1 if state.val_accs else 0

    result = {
        "name": name,
        "best_val_acc": best_acc,
        "best_epoch": best_ep,
        "final_train_loss": state.train_losses[-1] if state.train_losses else 0,
        "elapsed_min": round(elapsed / 60, 1),
        "combo": combo,
        "n_augments": n_aug,
        "epochs": probe_epochs,
        "error": state.error,
    }

    if state.error:
        print(f"  ERROR: {state.error}")
    else:
        print(f"  Result: val_acc={best_acc:.4f} at epoch {best_ep}"
              f" ({elapsed/60:.1f} min)")

    # Rename checkpoint
    best_path = MODELS_DIR / "best_model.pt"
    if best_path.exists():
        probe_name = f"augsearch_{name}_ep{best_ep}_acc{best_acc:.4f}.pt"
        probe_path = MODELS_DIR / probe_name
        best_path.rename(probe_path)
        result["checkpoint"] = probe_name
        print(f"  Saved: {probe_name}")

    return result


def run_full_train(name: str, combo: dict, train_df, val_df, label_encoder):
    """Full 100-epoch training run with the winning augmentation combo."""
    config = make_config(combo, probe_epochs=100)
    # Full training: more patience
    config.patience = 25
    state = TrainingState()

    print(f"\n{'#'*60}")
    print(f"  FULL TRAIN: {name} (100 epochs)")
    print(f"{'#'*60}")

    _start_monitor(f"FULL:{name}", state, 100, interval=10)

    resume_ckpt = MODELS_DIR / "resume_full_train.pt"
    t0 = time.time()
    train_model(
        train_df, val_df, EXPORT_DIR, label_encoder, config, state, MODELS_DIR,
        pretrained_path=PRETRAINED, resume_path=resume_ckpt,
    )
    # Clean up resume checkpoint after successful full train
    if resume_ckpt.exists():
        resume_ckpt.unlink()
    elapsed = time.time() - t0

    best_acc = max(state.val_accs) if state.val_accs else 0
    best_ep = state.val_accs.index(best_acc) + 1 if state.val_accs else 0

    print(f"\n  FULL TRAIN DONE: val_acc={best_acc:.4f} at epoch {best_ep}"
          f" ({elapsed/60:.1f} min)")

    best_path = MODELS_DIR / "best_model.pt"
    if best_path.exists():
        final_name = f"augsearch_winner_{name}_ep{best_ep}_acc{best_acc:.4f}.pt"
        final_path = MODELS_DIR / final_name
        best_path.rename(final_path)
        print(f"  Saved: {final_name}")

    return {
        "name": name,
        "best_val_acc": best_acc,
        "best_epoch": best_ep,
        "elapsed_min": round(elapsed / 60, 1),
        "error": state.error,
    }


def main():
    quick = "--quick" in sys.argv
    full_only = "--full" in sys.argv
    probe_epochs = 15 if quick else 30

    if not PRETRAINED.exists():
        print(f"ERROR: Pretrained checkpoint not found: {PRETRAINED}")
        sys.exit(1)

    # Clear pause file if present (user is explicitly resuming)
    if PAUSE_PATH.exists():
        PAUSE_PATH.unlink()
        print("Cleared pause file — resuming.")

    train_df, val_df, label_encoder = load_data()

    # Load existing log if resuming
    log = {"probes": [], "winner": None, "full_train": None}
    if LOG_PATH.exists():
        log = json.loads(LOG_PATH.read_text())

    completed_names = {p["name"] for p in log.get("probes", [])}

    if not full_only:
        print(f"\nAugmentation search — {len(SEARCH_COMBOS)} combos,"
              f" {probe_epochs} epochs each")
        print(f"Already completed: {len(completed_names)}")

        t_start = time.time()
        for combo in SEARCH_COMBOS:
            name = combo["name"]
            if name in completed_names:
                print(f"\nSkipping {name} (already done)")
                continue

            result = run_probe(name, combo, train_df, val_df, label_encoder,
                               probe_epochs)
            log["probes"].append(result)

            # Save log after each probe (resumable)
            LOG_PATH.write_text(json.dumps(log, indent=2))

            # Check for pause between probes
            if is_pause_requested():
                print(f"\n  PAUSED after {name}. Log saved to {LOG_PATH}")
                print(f"  To resume: rm {PAUSE_PATH} && re-run this script")
                return

        total = time.time() - t_start
        print(f"\n{'='*60}")
        print(f"Search done in {total/3600:.1f} hours")

    # Check if full train already done
    if log.get("full_train") and not log["full_train"].get("error"):
        print(f"\nFull train already completed: val_acc={log['full_train']['best_val_acc']:.4f}")
        print("Nothing to do. Delete full_train from log to re-run.")
        return

    # Find winner
    valid_probes = [p for p in log["probes"] if not p.get("error")]
    if not valid_probes:
        print("No successful probes — nothing to do.")
        return

    valid_probes.sort(key=lambda p: p["best_val_acc"], reverse=True)

    print(f"\n{'='*60}")
    print("  RESULTS RANKED:")
    print(f"{'='*60}")
    for i, p in enumerate(valid_probes):
        marker = " ★" if i == 0 else ""
        print(f"  {i+1}. {p['name']}: val_acc={p['best_val_acc']:.4f}"
              f" (ep {p['best_epoch']}){marker}")

    winner = valid_probes[0]
    log["winner"] = winner
    LOG_PATH.write_text(json.dumps(log, indent=2))

    print(f"\nWinner: {winner['name']} — val_acc={winner['best_val_acc']:.4f}")

    # Check for pause before full train
    if is_pause_requested():
        print(f"\n  PAUSED before full train. Winner saved to {LOG_PATH}")
        print(f"  To resume full train: rm {PAUSE_PATH} && .venv/bin/python tools/augment_search.py --full")
        return

    # Full training run with winner combo
    winner_combo = winner["combo"]
    print("\nStarting full 100-epoch training with winner combo...")
    full_result = run_full_train(winner["name"], winner_combo,
                                  train_df, val_df, label_encoder)
    log["full_train"] = full_result
    LOG_PATH.write_text(json.dumps(log, indent=2))

    print(f"\n{'#'*60}")
    print(f"  FINAL: {full_result['best_val_acc']:.4f} val_acc"
          f" (was {winner['best_val_acc']:.4f} in probe)")
    print(f"  Log: {LOG_PATH}")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
