"""MCP server exposing the SPJ pipeline as tools for Claude Code.

Run via stdio transport — Claude Code launches this as a subprocess.
All tool functions use lazy imports to keep startup fast.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# Project root (two levels up from src/spj/)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Ensure src/ is on the path so spj.* imports work
sys.path.insert(0, str(PROJECT_ROOT / "src"))

mcp = FastMCP(
    "spj-pipeline",
    instructions=(
        "SPJ Sign Language Corpus pipeline tools. "
        "Use these to inspect pipeline status, extract poses, "
        "align training data, train models, evaluate, and run inference."
    ),
)


# ---------------------------------------------------------------------------
# Tool: Pipeline status overview
# ---------------------------------------------------------------------------

@mcp.tool()
def spj_pipeline_status() -> dict:
    """Get a high-level overview of the entire SPJ pipeline status.

    Returns counts of videos, poses, segments, paired signs, checkpoints,
    and the next retraining threshold.
    """
    try:
        import pandas as pd
        from spj.trainer import list_checkpoints
        from spj.training_data import load_pairings_csv, PST_PAIRED
        from spj.glossary import load_glossary

        result = {
            "videos": 0,
            "poses_extracted": 0,
            "segments_total": 0,
            "segments_approved": 0,
            "signs_paired": 0,
            "unique_glosses": 0,
            "glossary_entries": 0,
            "checkpoints": 0,
            "latest_val_acc": None,
            "next_threshold": 500,
        }

        # Inventory
        inv_path = DATA_DIR / "inventory.csv"
        if inv_path.exists():
            inv = pd.read_csv(inv_path, dtype=str)
            result["videos"] = len(inv)

        # Pose files
        pose_dir = DATA_DIR / "pose"
        if pose_dir.exists():
            result["poses_extracted"] = len(list(pose_dir.glob("*.pose")))

        # Alignment / segments
        align_path = DATA_DIR / "training" / "alignment.csv"
        if align_path.exists():
            align = pd.read_csv(align_path, dtype=str)
            result["segments_total"] = len(align)
            if "status" in align.columns:
                result["segments_approved"] = int(
                    (align["status"] == "approved").sum()
                )

        # Pairings
        pairings_path = DATA_DIR / "training" / "pairings.csv"
        if pairings_path.exists():
            pairings = load_pairings_csv(pairings_path)
            if not pairings.empty:
                paired = pairings[pairings["status"] == PST_PAIRED]
                result["signs_paired"] = len(paired)
                if "gloss_id" in paired.columns:
                    result["unique_glosses"] = int(paired["gloss_id"].nunique())

        # Glossary
        glossary_path = DATA_DIR / "training" / "glossary.json"
        if glossary_path.exists():
            glossary = load_glossary(glossary_path)
            result["glossary_entries"] = len(glossary._data.get("glosses", {}))

        # Checkpoints
        models_dir = DATA_DIR / "models"
        checkpoints = list_checkpoints(models_dir)
        result["checkpoints"] = len(checkpoints)
        if checkpoints:
            latest = max(checkpoints, key=lambda c: c.get("timestamp", ""))
            result["latest_val_acc"] = latest.get("val_acc")

        # Next threshold
        from spj.orchestrator import RETRAIN_THRESHOLDS
        n_paired = result["signs_paired"]
        upcoming = [t for t in RETRAIN_THRESHOLDS if n_paired < t]
        result["next_threshold"] = upcoming[0] if upcoming else RETRAIN_THRESHOLDS[-1]

        return result

    except Exception as exc:
        return {"status": "error", "message": str(exc)}


# ---------------------------------------------------------------------------
# Tool: List videos
# ---------------------------------------------------------------------------

@mcp.tool()
def spj_list_videos(status_filter: str = "") -> list[dict]:
    """List videos from the inventory with their processing status.

    Args:
        status_filter: Optional filter — "has_pose", "no_pose", or empty for all.
    """
    try:
        import pandas as pd

        inv_path = DATA_DIR / "inventory.csv"
        if not inv_path.exists():
            return [{"status": "error", "message": "No inventory.csv found"}]

        df = pd.read_csv(inv_path, dtype=str)
        pose_dir = DATA_DIR / "pose"

        rows = []
        for _, row in df.iterrows():
            filename = str(row.get("filename", ""))
            stem = Path(filename).stem
            has_pose = (pose_dir / f"{stem}.pose").exists() if pose_dir.exists() else False

            if status_filter == "has_pose" and not has_pose:
                continue
            if status_filter == "no_pose" and has_pose:
                continue

            rows.append({
                "filename": filename,
                "has_pose": has_pose,
                "duration": row.get("duration", ""),
                "fps": row.get("fps", ""),
                "source": row.get("source", ""),
            })

        return rows

    except Exception as exc:
        return [{"status": "error", "message": str(exc)}]


# ---------------------------------------------------------------------------
# Tool: Extract pose
# ---------------------------------------------------------------------------

@mcp.tool()
def spj_extract_pose(video_filename: str) -> dict:
    """Extract MediaPipe pose landmarks from a single video.

    This is a long-running operation (2-30 min depending on video length).
    For batch extraction, use the Streamlit UI (page 2) instead.

    Args:
        video_filename: Filename from inventory (e.g. "video001.mp4").
    """
    try:
        from spj.pose import extract_pose

        video_dir = DATA_DIR / "videos"
        pose_dir = DATA_DIR / "pose"
        pose_dir.mkdir(parents=True, exist_ok=True)

        # Find the video file
        video_path = video_dir / video_filename
        if not video_path.exists():
            # Try searching subdirectories
            candidates = list(video_dir.rglob(video_filename))
            if not candidates:
                return {"status": "error", "message": f"Video not found: {video_filename}"}
            video_path = candidates[0]

        stem = video_path.stem
        pose_path = pose_dir / f"{stem}.pose"

        n_frames = extract_pose(video_path, pose_path)

        return {
            "status": "ok",
            "pose_path": str(pose_path),
            "n_frames": n_frames,
        }

    except Exception as exc:
        return {"status": "error", "message": str(exc)}


# ---------------------------------------------------------------------------
# Tool: Align segments
# ---------------------------------------------------------------------------

@mcp.tool()
def spj_align_segments(video_filename: str) -> dict:
    """Align pose data with subtitles for a video, producing training segments.

    Requires that both pose extraction and subtitle extraction have been
    completed for this video.

    Args:
        video_filename: Filename from inventory.
    """
    try:
        from spj.training_data import align_pose_to_subtitles

        stem = Path(video_filename).stem
        video_path = DATA_DIR / "videos" / video_filename
        pose_path = DATA_DIR / "pose" / f"{stem}.pose"
        vtt_path = DATA_DIR / "subtitles" / f"{stem}.vtt"

        for path, label in [(pose_path, "Pose"), (vtt_path, "Subtitles")]:
            if not path.exists():
                return {"status": "error", "message": f"{label} not found: {path.name}"}

        segments, raw_cue_count = align_pose_to_subtitles(
            pose_path, vtt_path, video_path,
        )

        return {
            "status": "ok",
            "n_segments": len(segments),
            "raw_cues": raw_cue_count,
            "video": video_filename,
        }

    except Exception as exc:
        return {"status": "error", "message": str(exc)}


# ---------------------------------------------------------------------------
# Tool: Detect signs in a segment
# ---------------------------------------------------------------------------

@mcp.tool()
def spj_detect_signs(segment_id: str) -> dict:
    """Detect individual sign boundaries within a subtitle segment.

    Args:
        segment_id: Segment ID from alignment.csv.
    """
    try:
        import pandas as pd
        from spj.training_data import detect_signs_in_segment, load_alignment_csv
        from spj.preannotate import load_pose_arrays

        align_path = DATA_DIR / "training" / "alignment.csv"
        if not align_path.exists():
            return {"status": "error", "message": "No alignment.csv found"}

        align_df = load_alignment_csv(align_path)
        match = align_df[align_df.index.astype(str) == segment_id]
        if match.empty:
            # Try matching by a segment_id column if it exists
            if "segment_id" in align_df.columns:
                match = align_df[align_df["segment_id"] == segment_id]
            if match.empty:
                return {"status": "error", "message": f"Segment {segment_id} not found"}

        row = match.iloc[0]
        pose_path = Path(str(row["pose_path"]))
        video_path = str(row["video_path"])

        if not pose_path.exists():
            return {"status": "error", "message": f"Pose file not found: {pose_path}"}

        data, conf, fps = load_pose_arrays(pose_path)

        pairings = detect_signs_in_segment(
            data, conf, fps,
            seg_start_ms=int(row["start_ms"]),
            seg_end_ms=int(row["end_ms"]),
            segment_id=segment_id,
            video_path=video_path,
            pose_path=str(pose_path),
        )

        return {
            "status": "ok",
            "n_signs": len(pairings),
            "signs": [
                {
                    "pairing_id": p["pairing_id"],
                    "hand": p["hand"],
                    "start_ms": p["sign_start_ms"],
                    "end_ms": p["sign_end_ms"],
                    "confidence": round(p["motion_confidence"], 3),
                }
                for p in pairings
            ],
        }

    except Exception as exc:
        return {"status": "error", "message": str(exc)}


# ---------------------------------------------------------------------------
# Tool: Pair a sign with a word/gloss
# ---------------------------------------------------------------------------

@mcp.tool()
def spj_pair_sign(pairing_id: str, word: str, gloss_id: str) -> dict:
    """Assign a word and gloss ID to a detected sign pairing.

    Args:
        pairing_id: The pairing ID from pairings.csv.
        word: The Slovak word (e.g. "voda").
        gloss_id: The ID-gloss (e.g. "WATER-1").
    """
    try:
        from spj.training_data import (
            load_pairings_csv, save_pairings_csv, PST_PAIRED,
        )

        pairings_path = DATA_DIR / "training" / "pairings.csv"
        if not pairings_path.exists():
            return {"status": "error", "message": "No pairings.csv found"}

        df = load_pairings_csv(pairings_path)
        mask = df["pairing_id"] == pairing_id
        if not mask.any():
            return {"status": "error", "message": f"Pairing {pairing_id} not found"}

        idx = df.index[mask][0]
        df.loc[idx, "word"] = word
        df.loc[idx, "gloss_id"] = gloss_id
        df.loc[idx, "status"] = PST_PAIRED

        save_pairings_csv(df, pairings_path)

        return {
            "status": "ok",
            "pairing_id": pairing_id,
            "gloss_id": gloss_id,
            "word": word,
        }

    except Exception as exc:
        return {"status": "error", "message": str(exc)}


# ---------------------------------------------------------------------------
# Tool: Export training data
# ---------------------------------------------------------------------------

@mcp.tool()
def spj_export_training() -> dict:
    """Export all paired signs as NPZ files for training.

    Creates manifest.csv alongside the NPZ files.
    """
    try:
        import pandas as pd
        from spj.training_data import (
            load_pairings_csv, export_sign_npz, PST_PAIRED,
        )
        from spj.preannotate import load_pose_arrays

        pairings_path = DATA_DIR / "training" / "pairings.csv"
        npz_dir = DATA_DIR / "training" / "export"

        if not pairings_path.exists():
            return {"status": "error", "message": "No pairings.csv found"}

        df = load_pairings_csv(pairings_path)
        paired = df[df["status"] == PST_PAIRED]
        if paired.empty:
            return {"status": "error", "message": "No paired signs to export"}

        npz_dir.mkdir(parents=True, exist_ok=True)
        exported = 0
        manifest_rows = []

        for pose_path_str, group in paired.groupby("pose_path"):
            pose_path = Path(pose_path_str)
            if not pose_path.exists():
                continue
            data, conf, fps = load_pose_arrays(pose_path)
            for _, row in group.iterrows():
                result = export_sign_npz(row, data, conf, npz_dir)
                if result:
                    exported += 1
                    manifest_rows.append({
                        "segment_id": row["pairing_id"],
                        "label": row["gloss_id"],
                        "text": row.get("word", ""),
                        "reviewed_text": row["gloss_id"],
                    })

        # Write manifest
        manifest_df = pd.DataFrame(manifest_rows)
        manifest_path = npz_dir / "manifest.csv"
        manifest_df.to_csv(manifest_path, index=False)

        return {
            "status": "ok",
            "n_exported": exported,
            "manifest_path": str(manifest_path),
        }

    except Exception as exc:
        return {"status": "error", "message": str(exc)}


# ---------------------------------------------------------------------------
# Tool: Train model
# ---------------------------------------------------------------------------

@mcp.tool()
def spj_train_model(
    epochs: int = 50,
    lr: float = 1e-3,
    backbone: str = "from_scratch",
) -> dict:
    """Train a PoseTransformerEncoder on exported training data.

    This is a long-running operation. For interactive training with
    live progress, use the Streamlit UI (page 8) instead.

    Args:
        epochs: Number of training epochs (default 50).
        lr: Learning rate (default 0.001).
        backbone: "from_scratch" or path to pretrained checkpoint.
    """
    try:
        import pandas as pd
        from spj.trainer import (
            LabelEncoder, TrainingConfig, TrainingState,
            split_dataset, train_model,
        )

        npz_dir = DATA_DIR / "training" / "export"
        splits_dir = DATA_DIR / "training" / "splits"
        models_dir = DATA_DIR / "models"

        manifest_path = npz_dir / "manifest.csv"
        if not manifest_path.exists():
            return {
                "status": "error",
                "message": "No manifest.csv — run spj_export_training first",
            }

        manifest_df = pd.read_csv(manifest_path)
        if manifest_df.empty:
            return {"status": "error", "message": "Manifest is empty"}

        # Derive label column
        if "label" not in manifest_df.columns:
            manifest_df["label"] = manifest_df["reviewed_text"].where(
                manifest_df["reviewed_text"].str.strip() != "",
                manifest_df["text"],
            )

        labels = manifest_df["label"].tolist()
        label_encoder = LabelEncoder(labels)

        train_df, val_df, test_df = split_dataset(
            manifest_df, output_dir=splits_dir,
        )

        config = TrainingConfig(
            epochs=epochs,
            lr=lr,
            backbone=backbone,
        )

        state = TrainingState()
        pretrained = None
        if backbone != "from_scratch":
            p = Path(backbone)
            if p.exists():
                pretrained = p

        train_model(
            train_df, val_df, npz_dir, label_encoder,
            config, state, models_dir, pretrained_path=pretrained,
        )

        if state.error:
            return {"status": "error", "message": state.error}

        return {
            "status": "ok",
            "checkpoint_path": state.checkpoint_path,
            "val_acc": state.best_val_acc,
            "best_epoch": state.best_epoch,
            "n_classes": label_encoder.n_classes,
            "n_train": len(train_df),
        }

    except Exception as exc:
        return {"status": "error", "message": str(exc)}


# ---------------------------------------------------------------------------
# Tool: Evaluate model
# ---------------------------------------------------------------------------

@mcp.tool()
def spj_evaluate_model(checkpoint_filename: str) -> dict:
    """Evaluate a trained model on the held-out test split.

    Args:
        checkpoint_filename: Filename of the .pt checkpoint in data/models/.
    """
    try:
        import pandas as pd
        from spj.trainer import load_checkpoint, LabelEncoder
        from spj.evaluator import evaluate_model, save_evaluation_report

        models_dir = DATA_DIR / "models"
        ckpt_path = models_dir / checkpoint_filename
        if not ckpt_path.exists():
            return {"status": "error", "message": f"Checkpoint not found: {checkpoint_filename}"}

        model, label_encoder, config, meta = load_checkpoint(ckpt_path)

        # Load test split
        test_path = DATA_DIR / "training" / "splits" / "test.csv"
        if not test_path.exists():
            return {"status": "error", "message": "No test.csv — train a model first"}

        test_df = pd.read_csv(test_path)
        npz_dir = DATA_DIR / "training" / "export"

        metrics = evaluate_model(
            model, label_encoder, test_df, npz_dir,
            max_seq_len=config.max_seq_len,
        )

        # Save report
        eval_dir = DATA_DIR / "evaluations"
        save_evaluation_report(metrics, checkpoint_filename, eval_dir)

        # Return summary (without large arrays)
        return {
            "status": "ok",
            "accuracy": metrics.get("accuracy"),
            "top3_accuracy": metrics.get("top3_accuracy"),
            "n_samples": metrics.get("n_samples"),
            "n_classes": metrics.get("n_classes"),
            "per_class_top5": sorted(
                metrics.get("per_class", []),
                key=lambda x: x.get("f1", 0),
                reverse=True,
            )[:5],
        }

    except Exception as exc:
        return {"status": "error", "message": str(exc)}


# ---------------------------------------------------------------------------
# Tool: Run inference
# ---------------------------------------------------------------------------

@mcp.tool()
def spj_run_inference(video_filename: str, checkpoint_filename: str) -> dict:
    """Run gloss prepartner-dictn on a video using a trained checkpoint.

    Writes prepartner-dictns to the video's EAF file (AI tiers).

    Args:
        video_filename: Video filename from inventory.
        checkpoint_filename: Checkpoint .pt filename in data/models/.
    """
    try:
        from spj.trainer import load_checkpoint
        from spj.preannotate import load_pose_arrays, detect_sign_segments
        from spj.inference import predict_segments, write_prepartner-dictns_to_eaf

        models_dir = DATA_DIR / "models"
        ckpt_path = models_dir / checkpoint_filename
        if not ckpt_path.exists():
            return {"status": "error", "message": f"Checkpoint not found: {checkpoint_filename}"}

        stem = Path(video_filename).stem
        pose_path = DATA_DIR / "pose" / f"{stem}.pose"
        if not pose_path.exists():
            return {"status": "error", "message": f"Pose not found for {video_filename}"}

        model, label_encoder, config, meta = load_checkpoint(ckpt_path)

        # Detect landmark filtering from model's input dimension
        from spj.training_data import SL_LANDMARK_PRESETS, preset_from_input_dim
        input_dim = model.input_proj.weight.shape[1]
        preset_name = preset_from_input_dim(input_dim)
        lm_indices = SL_LANDMARK_PRESETS[preset_name] if preset_name else None

        pose_data, conf_data, fps = load_pose_arrays(pose_path)
        segments = detect_sign_segments(pose_data, conf_data, fps)

        prepartner-dictns = predict_segments(
            model, label_encoder, segments, pose_data, fps,
            max_seq_len=config.max_seq_len,
            landmark_indices=lm_indices,
        )

        eaf_path = DATA_DIR / "annotations" / f"{stem}.eaf"
        write_prepartner-dictns_to_eaf(prepartner-dictns, eaf_path)

        return {
            "status": "ok",
            "n_prepartner-dictns": len(prepartner-dictns),
            "eaf_path": str(eaf_path),
            "prepartner-dictns": [
                {
                    "gloss": p["predicted_gloss"],
                    "confidence": round(p["prepartner-dictn_confidence"], 3),
                    "hand": p["hand"],
                    "start_ms": p["start_ms"],
                    "end_ms": p["end_ms"],
                }
                for p in prepartner-dictns[:20]  # Limit to first 20 for readability
            ],
        }

    except Exception as exc:
        return {"status": "error", "message": str(exc)}


# ---------------------------------------------------------------------------
# Tool: Glossary status
# ---------------------------------------------------------------------------

@mcp.tool()
def spj_glossary_status() -> dict:
    """Get the current state of the sign-word glossary.

    Returns gloss count, form count, and the full list of glosses with their forms.
    """
    try:
        from spj.glossary import load_glossary

        glossary_path = DATA_DIR / "training" / "glossary.json"
        if not glossary_path.exists():
            return {"n_glosses": 0, "n_forms": 0, "glosses": {}}

        glossary = load_glossary(glossary_path)
        data = glossary._data.get("glosses", {})

        n_forms = sum(len(entry.get("forms", [])) for entry in data.values())

        return {
            "n_glosses": len(data),
            "n_forms": n_forms,
            "glosses": data,
        }

    except Exception as exc:
        return {"status": "error", "message": str(exc)}


# ---------------------------------------------------------------------------
# Tool: Orchestrate (check / retrain / inference)
# ---------------------------------------------------------------------------

@mcp.tool()
def spj_orchestrate(action: str = "check") -> dict:
    """Active learning orchestration — check status, retrain, or run inference.

    Args:
        action: One of "check", "retrain", or "inference".
            - "check": Check if retraining should be triggered.
            - "retrain": Run a full retrain cycle (export → split → train).
            - "inference": Run inference with the best checkpoint on all videos.
    """
    try:
        from spj.orchestrator import (
            check_retrain_status,
            run_retrain_cycle,
            run_inference_with_best,
        )

        pairings_path = DATA_DIR / "training" / "pairings.csv"
        models_dir = DATA_DIR / "models"

        if action == "check":
            report = check_retrain_status(pairings_path, models_dir)
            return asdict(report)

        elif action == "retrain":
            report = run_retrain_cycle(DATA_DIR)
            return asdict(report)

        elif action == "inference":
            result = run_inference_with_best(DATA_DIR)
            return result

        else:
            return {"status": "error", "message": f"Unknown action: {action}. Use check/retrain/inference."}

    except Exception as exc:
        return {"status": "error", "message": str(exc)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")
