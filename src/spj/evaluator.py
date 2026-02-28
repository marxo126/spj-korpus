"""Model evaluation: test-set metrics, confusion matrix, and model comparison.

Evaluates trained PoseTransformerEncoder checkpoints on held-out test splits,
generates per-class metrics, and supports side-by-side model comparison.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def evaluate_model(
    model: "torch.nn.Module",
    label_encoder: "LabelEncoder",
    test_df: pd.DataFrame,
    npz_dir: Path,
    max_seq_len: int = 300,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
) -> dict:
    """Run model on test split and return detailed metrics.

    Returns dict with keys:
        accuracy, top3_accuracy, n_samples, n_classes,
        per_class (list of dicts with label, precision, recall, f1, support),
        confusion_matrix (list of lists),
        class_labels (list of str),
        all_preds (list of int),
        all_labels (list of int),
        all_confidences (list of float),
    """
    from spj.trainer import PoseSegmentDataset

    if device is None:
        device = next(model.parameters()).device

    ds = PoseSegmentDataset(test_df, npz_dir, label_encoder, max_seq_len)
    if len(ds) == 0:
        return {"error": "No test samples found", "accuracy": 0.0, "n_samples": 0}

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["label"]

            logits = model(features, mask)
            probs = torch.softmax(logits, dim=1).cpu()
            preds = logits.argmax(dim=1).cpu()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            all_probs.append(probs.numpy())

    all_preds_arr = np.array(all_preds)
    all_labels_arr = np.array(all_labels)
    all_probs_arr = np.concatenate(all_probs, axis=0)

    n_samples = len(all_labels)
    n_classes = label_encoder.n_classes
    class_labels = [label_encoder.decode(i) for i in range(n_classes)]

    # Accuracy
    accuracy = float((all_preds_arr == all_labels_arr).mean())

    # Top-3 accuracy
    top3_correct = 0
    for i in range(n_samples):
        top3_indices = np.argsort(all_probs_arr[i])[-3:]
        if all_labels_arr[i] in top3_indices:
            top3_correct += 1
    top3_accuracy = top3_correct / max(1, n_samples)

    # Confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for pred, true in zip(all_preds_arr, all_labels_arr):
        cm[true, pred] += 1

    # Per-class metrics
    per_class = []
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = int(cm[i, :].sum())

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)

        per_class.append({
            "label": class_labels[i],
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "support": support,
        })

    # Confidence of correct prepartner-dictns
    confidences = []
    for i in range(n_samples):
        confidences.append(float(all_probs_arr[i, all_preds_arr[i]]))

    return {
        "accuracy": round(accuracy, 4),
        "top3_accuracy": round(top3_accuracy, 4),
        "n_samples": n_samples,
        "n_classes": n_classes,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_labels": class_labels,
        "all_preds": all_preds,
        "all_labels": all_labels,
        "all_confidences": confidences,
    }


def save_evaluation_report(
    metrics: dict,
    checkpoint_name: str,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Save evaluation results as JSON + CSV.

    Returns (json_path, csv_path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(checkpoint_name).stem

    # JSON report (full metrics)
    json_path = output_dir / f"{stem}_eval.json"
    # Remove large arrays from JSON for readability
    json_data = {k: v for k, v in metrics.items()
                 if k not in ("all_preds", "all_labels", "all_confidences")}
    json_path.write_text(json.dumps(json_data, indent=2))

    # CSV per-class report
    csv_path = output_dir / f"{stem}_per_class.csv"
    if metrics.get("per_class"):
        pd.DataFrame(metrics["per_class"]).to_csv(csv_path, index=False)

    return json_path, csv_path


def confusion_matrix_figure(
    cm: list[list[int]],
    class_labels: list[str],
) -> "go.Figure":
    """Plotly heatmap of the confusion matrix."""
    import plotly.graph_objects as go

    cm_arr = np.array(cm)
    # Normalize for display (show percentages)
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    cm_pct = np.where(row_sums > 0, cm_arr / row_sums * 100, 0)

    # Build text annotations showing count (pct%)
    text = []
    for i in range(len(class_labels)):
        row = []
        for j in range(len(class_labels)):
            row.append(f"{cm_arr[i, j]}<br>({cm_pct[i, j]:.0f}%)")
        text.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=cm_pct,
        x=class_labels,
        y=class_labels,
        text=text,
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=True,
        colorbar=dict(title="%"),
    ))

    n = len(class_labels)
    height = max(400, min(800, 50 * n + 100))

    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="True",
        height=height,
        xaxis=dict(tickangle=-45),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def per_class_f1_figure(per_class: list[dict]) -> "go.Figure":
    """Bar chart of per-class F1 scores."""
    import plotly.graph_objects as go

    # Sort by F1 descending
    sorted_pc = sorted(per_class, key=lambda x: x["f1"], reverse=True)
    labels = [p["label"] for p in sorted_pc]
    f1s = [p["f1"] for p in sorted_pc]

    fig = go.Figure(data=go.Bar(
        x=labels,
        y=f1s,
        marker_color="steelblue",
    ))
    fig.update_layout(
        title="Per-Class F1 Score",
        xaxis_title="Class",
        yaxis_title="F1",
        yaxis=dict(range=[0, 1]),
        height=400,
        xaxis=dict(tickangle=-45),
    )
    return fig


def compare_models_table(
    evaluations: list[tuple[str, dict]],
) -> pd.DataFrame:
    """Build a comparison table from multiple (name, metrics) pairs."""
    rows = []
    for name, m in evaluations:
        rows.append({
            "Model": name,
            "Accuracy": m.get("accuracy", 0),
            "Top-3 Acc": m.get("top3_accuracy", 0),
            "N Samples": m.get("n_samples", 0),
            "N Classes": m.get("n_classes", 0),
        })
    return pd.DataFrame(rows)
