"""Active learning orchestrator — monitors sign count and triggers retraining.

Standalone module (no Streamlit dependency). Importable from MCP server,
Streamlit pages, or CLI.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

RETRAIN_THRESHOLDS = [500, 2_000, 5_000, 10_000]


@dataclass
class OrchestrationReport:
    """Result of an orchestration check or action."""
    paired_signs: int = 0
    unique_glosses: int = 0
    should_retrain: bool = False
    retrain_reason: str = ""
    current_threshold: int = 0
    next_threshold: int = 0
    checkpoint_path: Optional[str] = None
    val_acc: Optional[float] = None
    n_checkpoints: int = 0
    error: Optional[str] = None


def _count_paired_signs(pairings_path: Path) -> tuple[int, int]:
    """Count paired signs and unique glosses from pairings CSV.

    Returns (n_paired, n_unique_glosses).
    """
    import pandas as pd
    from spj.training_data import load_pairings_csv, PST_PAIRED

    if not pairings_path.exists():
        return 0, 0

    df = load_pairings_csv(pairings_path)
    if df.empty or "status" not in df.columns:
        return 0, 0

    paired = df[df["status"] == PST_PAIRED]
    n_paired = len(paired)
    n_glosses = paired["gloss_id"].nunique() if "gloss_id" in paired.columns else 0
    return n_paired, n_glosses


def _latest_checkpoint_n_train(models_dir: Path) -> int:
    """Get n_train from the most recent checkpoint, or 0 if none."""
    from spj.trainer import list_checkpoints

    checkpoints = list_checkpoints(models_dir)
    if not checkpoints:
        return 0

    # list_checkpoints returns sorted by filename; pick latest by timestamp
    latest = max(checkpoints, key=lambda c: c.get("timestamp", ""))
    return int(latest.get("n_train", 0))


def _find_threshold(n_paired: int, last_trained_at: int) -> tuple[bool, int, int, str]:
    """Determine if retraining should be triggered.

    Returns (should_retrain, current_threshold, next_threshold, reason).
    """
    # Find which thresholds have been crossed
    crossed = [t for t in RETRAIN_THRESHOLDS if n_paired >= t]
    current = crossed[-1] if crossed else 0

    # Find next threshold
    upcoming = [t for t in RETRAIN_THRESHOLDS if n_paired < t]
    next_t = upcoming[0] if upcoming else RETRAIN_THRESHOLDS[-1]

    # Check if we crossed a new threshold since last training
    newly_crossed = [t for t in RETRAIN_THRESHOLDS
                     if n_paired >= t > last_trained_at]

    if newly_crossed:
        trigger = newly_crossed[-1]
        reason = (
            f"Sign count ({n_paired}) crossed threshold {trigger}. "
            f"Last model was trained at {last_trained_at} signs."
        )
        return True, current, next_t, reason

    return False, current, next_t, ""


def check_retrain_status(
    pairings_path: Path,
    models_dir: Path,
) -> OrchestrationReport:
    """Check whether retraining should be triggered based on paired sign count.

    Compares current paired-sign count against RETRAIN_THRESHOLDS and the
    n_train recorded in the latest checkpoint.
    """
    try:
        from spj.trainer import list_checkpoints

        n_paired, n_glosses = _count_paired_signs(pairings_path)
        last_trained_at = _latest_checkpoint_n_train(models_dir)
        checkpoints = list_checkpoints(models_dir)
        should, current_t, next_t, reason = _find_threshold(n_paired, last_trained_at)

        return OrchestrationReport(
            paired_signs=n_paired,
            unique_glosses=n_glosses,
            should_retrain=should,
            retrain_reason=reason,
            current_threshold=current_t,
            next_threshold=next_t,
            n_checkpoints=len(checkpoints),
        )
    except Exception as exc:
        logger.exception("check_retrain_status failed")
        return OrchestrationReport(error=str(exc))


def run_retrain_cycle(
    data_dir: Path,
    config_overrides: Optional[dict] = None,
    transfer_from: Optional[str] = None,
) -> OrchestrationReport:
    """Full retrain cycle: export signs → split → train → evaluate.

    Runs synchronously (designed for MCP/CLI, not Streamlit).
    """
    try:
        import pandas as pd
        from spj.training_data import (
            load_pairings_csv, export_sign_npz, PST_PAIRED,
            harvest_eaf_auto,
        )
        from spj.preannotate import load_pose_arrays
        from spj.trainer import (
            LabelEncoder, TrainingConfig, TrainingState,
            split_dataset, train_model, list_checkpoints,
        )

        data_dir = Path(data_dir)
        pairings_path = data_dir / "training" / "pairings.csv"
        npz_dir = data_dir / "training" / "export"
        splits_dir = data_dir / "training" / "splits"
        models_dir = data_dir / "models"

        # Auto-harvest EAF annotations before counting paired signs
        try:
            harvest_result = harvest_eaf_auto(data_dir)
            if harvest_result and harvest_result["n_new_pairings"] > 0:
                logger.info(
                    "EAF harvest: %d new pairings from %d files",
                    harvest_result["n_new_pairings"],
                    harvest_result["n_with_annotations"],
                )
        except Exception as exc:
            logger.warning("EAF harvest failed (continuing): %s", exc)

        # 1. Load paired signs
        df = load_pairings_csv(pairings_path)
        paired = df[df["status"] == PST_PAIRED].copy()
        if paired.empty:
            return OrchestrationReport(error="No paired signs to train on")

        n_paired = len(paired)
        n_glosses = paired["gloss_id"].nunique()

        # 2. Export NPZ files for all paired signs
        logger.info("Exporting %d paired signs to NPZ...", n_paired)
        npz_dir.mkdir(parents=True, exist_ok=True)
        exported = 0
        # Group by pose_path to avoid reloading the same file
        for pose_path_str, group in paired.groupby("pose_path"):
            pose_path = Path(pose_path_str)
            if not pose_path.exists():
                logger.warning("Pose file not found: %s", pose_path)
                continue
            data, conf, fps = load_pose_arrays(pose_path)
            for _, row in group.iterrows():
                result = export_sign_npz(row, data, conf, npz_dir)
                if result:
                    exported += 1

        if exported == 0:
            return OrchestrationReport(
                paired_signs=n_paired,
                error="No NPZ files exported — check pose files",
            )

        # 3. Build manifest
        manifest_rows = []
        for _, row in paired.iterrows():
            npz_file = npz_dir / f"{row['pairing_id']}.npz"
            if npz_file.exists():
                manifest_rows.append({
                    "segment_id": row["pairing_id"],
                    "label": row["gloss_id"],
                    "text": row.get("word", ""),
                    "reviewed_text": row["gloss_id"],
                })
        manifest_df = pd.DataFrame(manifest_rows)
        manifest_df.to_csv(npz_dir / "manifest.csv", index=False)

        # 4. Apply quality filtering and split
        from spj.training_data import filter_quality_labels
        manifest_df = filter_quality_labels(manifest_df, min_samples=3)
        if manifest_df.empty:
            return OrchestrationReport(
                paired_signs=n_paired,
                error="No labels with 3+ samples after quality filtering",
            )

        train_df, val_df, test_df = split_dataset(
            manifest_df, output_dir=splits_dir,
        )

        # Build label encoder from filtered labels
        labels = manifest_df["label"].tolist()
        label_encoder = LabelEncoder(labels)

        config = TrainingConfig()
        if config_overrides:
            for k, v in config_overrides.items():
                if hasattr(config, k):
                    setattr(config, k, v)

        # 5. Auto-detect category model for transfer
        from spj.trainer import find_category_checkpoints, TRANSFER_DEFAULTS
        pretrained = None
        if transfer_from:
            p = Path(transfer_from)
            if not p.is_absolute():
                p = models_dir / transfer_from
            if p.exists():
                pretrained = p
        elif transfer_from is None:
            # Auto-detect: best category checkpoint
            cat_candidates = find_category_checkpoints(models_dir)
            if cat_candidates:
                best_cat = cat_candidates[0]  # already sorted best-first
                pretrained = Path(best_cat["path"])
                logger.info("Auto-detected category model: %s (val_acc=%.3f)",
                            best_cat["filename"], best_cat["val_acc"])

        # Set transfer-optimized config if using a pretrained model
        if pretrained and config.freeze_epochs == 0:
            for k, v in TRANSFER_DEFAULTS.items():
                setattr(config, k, v)

        # 6. Train (synchronous)
        state = TrainingState()
        train_model(
            train_df, val_df, npz_dir, label_encoder,
            config, state, models_dir, pretrained_path=pretrained,
        )

        if state.error:
            return OrchestrationReport(
                paired_signs=n_paired,
                unique_glosses=n_glosses,
                error=f"Training failed: {state.error}",
            )

        return OrchestrationReport(
            paired_signs=n_paired,
            unique_glosses=n_glosses,
            should_retrain=False,
            retrain_reason="Retrain completed successfully",
            checkpoint_path=state.checkpoint_path,
            val_acc=state.best_val_acc,
            n_checkpoints=len(list_checkpoints(models_dir)),
        )

    except Exception as exc:
        logger.exception("run_retrain_cycle failed")
        return OrchestrationReport(error=str(exc))


def select_best_checkpoint(
    models_dir: Path,
    evaluations_dir: Path,
    min_classes: int = 200,
) -> Optional[str]:
    """Select the best word-level checkpoint for inference.

    Prefers models with n_classes >= min_classes (skips category-only models).
    Among qualifying models, picks highest validation accuracy.
    Returns the checkpoint filename, or None.
    """
    from spj.trainer import list_checkpoints

    models_dir = Path(models_dir)
    evaluations_dir = Path(evaluations_dir)

    best_acc = -1.0
    best_name = None

    # Try evaluation reports first
    if evaluations_dir.exists():
        for json_file in evaluations_dir.glob("*_eval.json"):
            try:
                report = json.loads(json_file.read_text())
                n_cls = report.get("n_classes", 0)
                if n_cls < min_classes:
                    continue
                acc = report.get("accuracy", 0.0)
                if acc > best_acc:
                    best_acc = acc
                    stem = json_file.stem.replace("_eval", "")
                    best_name = f"{stem}.pt"
            except Exception:
                continue

    # Fallback to checkpoint metadata
    if best_name is None:
        checkpoints = list_checkpoints(models_dir)
        for ckpt in checkpoints:
            n_cls = ckpt.get("n_classes", 0)
            if n_cls < min_classes:
                continue
            acc = ckpt.get("val_acc", 0.0)
            if acc > best_acc:
                best_acc = acc
                best_name = ckpt.get("filename")

    return best_name


def run_inference_with_best(
    data_dir: Path,
    video_filenames: Optional[list[str]] = None,
) -> dict:
    """Run inference using the best checkpoint on specified (or all) videos.

    Returns dict with n_prepartner-dictns, results per video, and eaf_paths.
    """
    try:
        import pandas as pd
        from spj.trainer import load_checkpoint
        from spj.preannotate import load_pose_arrays, detect_sign_segments
        from spj.inference import predict_segments, write_prepartner-dictns_to_eaf

        data_dir = Path(data_dir)
        models_dir = data_dir / "models"
        evaluations_dir = data_dir / "evaluations"
        pose_dir = data_dir / "pose"
        annotations_dir = data_dir / "annotations"
        inv_path = data_dir / "inventory.csv"

        # Find best checkpoint
        best_name = select_best_checkpoint(models_dir, evaluations_dir)
        if not best_name:
            return {"status": "error", "message": "No checkpoints found"}

        ckpt_path = models_dir / best_name
        if not ckpt_path.exists():
            return {"status": "error", "message": f"Checkpoint not found: {best_name}"}

        model, label_encoder, config, meta = load_checkpoint(ckpt_path)

        # Detect landmark filtering from model's input dimension
        from spj.training_data import SL_LANDMARK_PRESETS, preset_from_input_dim
        input_dim = model.input_proj.weight.shape[1]
        preset_name = preset_from_input_dim(input_dim)
        lm_indices = SL_LANDMARK_PRESETS[preset_name] if preset_name else None

        # Load inventory
        if not inv_path.exists():
            return {"status": "error", "message": "No inventory.csv found"}

        inv_df = pd.read_csv(inv_path, dtype=str)

        # Filter videos
        if video_filenames:
            inv_df = inv_df[inv_df["filename"].isin(video_filenames)]

        results = []
        total_prepartner-dictns = 0

        for _, row in inv_df.iterrows():
            filename = str(row["filename"])
            pose_path = pose_dir / f"{Path(filename).stem}.pose"

            if not pose_path.exists():
                results.append({
                    "video": filename,
                    "status": "skipped",
                    "reason": "No pose file",
                })
                continue

            try:
                pose_data, conf_data, fps = load_pose_arrays(pose_path)
                segments = detect_sign_segments(pose_data, conf_data, fps)

                feature_mode = config.feature_mode
                prepartner-dictns = predict_segments(
                    model, label_encoder, segments, pose_data, fps,
                    max_seq_len=config.max_seq_len,
                    landmark_indices=lm_indices,
                    feature_mode=feature_mode,
                )

                # Write to EAF
                eaf_path = annotations_dir / f"{Path(filename).stem}.eaf"
                write_prepartner-dictns_to_eaf(prepartner-dictns, eaf_path)

                results.append({
                    "video": filename,
                    "status": "ok",
                    "n_prepartner-dictns": len(prepartner-dictns),
                    "eaf_path": str(eaf_path),
                })
                total_prepartner-dictns += len(prepartner-dictns)

            except Exception as exc:
                results.append({
                    "video": filename,
                    "status": "error",
                    "message": str(exc),
                })

        return {
            "status": "ok",
            "checkpoint": best_name,
            "n_videos": len(results),
            "n_prepartner-dictns": total_prepartner-dictns,
            "results": results,
        }

    except Exception as exc:
        logger.exception("run_inference_with_best failed")
        return {"status": "error", "message": str(exc)}
