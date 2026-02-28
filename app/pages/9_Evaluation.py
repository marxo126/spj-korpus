"""Evaluation page — test-set metrics and model comparison.

Tab 1 (Evaluate):  Select checkpoint, run on test split, view metrics
Tab 2 (Compare):   Side-by-side comparison of multiple checkpoints
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

DATA_DIR     = Path(__file__).parent.parent.parent / "data"
TRAINING_DIR = DATA_DIR / "training"
EXPORT_DIR   = TRAINING_DIR / "export"
SPLITS_DIR   = TRAINING_DIR / "splits"
MODELS_DIR   = DATA_DIR / "models"
EVAL_DIR     = DATA_DIR / "evaluations"

st.header("📊 Evaluation")
st.caption("Page 9/10 · Evaluate trained models on the test split and compare performance.")
with st.expander("ℹ️ How to use this page", expanded=False):
    st.markdown("""
**Needs:**
- A trained model checkpoint in `data/models/` (from page 8)
- A test split CSV in `data/training/splits/` (from page 8)

**Tab 📈 Evaluate**
1. Select a checkpoint.
2. Click **Run Evaluation** — shows accuracy, top-3 accuracy, confusion matrix, per-class F1.
3. Export JSON + CSV report.

**Tab ⚖️ Compare**
1. Select 2+ checkpoints.
2. View side-by-side metrics table.
""")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_eval, tab_compare = st.tabs(["📈 Evaluate", "⚖️ Compare"])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — EVALUATE
# ══════════════════════════════════════════════════════════════════════════
with tab_eval:
    from spj.trainer import list_checkpoints

    ckpts = list_checkpoints(MODELS_DIR)

    if not ckpts:
        st.warning("No model checkpoints found. Train a model on page 8 first.")
    elif not (SPLITS_DIR / "test.csv").exists():
        st.warning("No test split found. Split the data on page 8 first.")
    else:
        selected_ckpt = st.selectbox(
            "Select checkpoint",
            [c["filename"] for c in ckpts],
            format_func=lambda f: f"{f} (acc: {next(c['val_acc'] for c in ckpts if c['filename'] == f):.2%})",
            key="eval_ckpt",
        )

        ckpt_info = next(c for c in ckpts if c["filename"] == selected_ckpt)

        ec1, ec2, ec3 = st.columns(3)
        ec1.metric("Backbone", ckpt_info.get("backbone", "?"))
        ec2.metric("Classes", ckpt_info.get("n_classes", 0))
        ec3.metric("Train val acc", f"{ckpt_info.get('val_acc', 0):.2%}")

        if st.button("▶ Run Evaluation", type="primary"):
            with st.spinner("Evaluating on test split…"):
                from spj.trainer import load_checkpoint
                from spj.evaluator import (
                    evaluate_model,
                    save_evaluation_report,
                    confusion_matrix_figure,
                    per_class_f1_figure,
                )

                ckpt_path = Path(ckpt_info["path"])
                model, label_encoder, config, meta = load_checkpoint(ckpt_path)

                test_df = pd.read_csv(SPLITS_DIR / "test.csv", dtype=str)
                if "label" not in test_df.columns:
                    test_df["label"] = test_df["reviewed_text"].where(
                        test_df["reviewed_text"].str.strip() != "",
                        test_df["text"],
                    )

                metrics = evaluate_model(
                    model, label_encoder, test_df, EXPORT_DIR,
                    max_seq_len=config.max_seq_len,
                    batch_size=config.batch_size,
                )

            if "error" in metrics:
                st.error(metrics["error"])
            else:
                st.session_state["eval_metrics"] = metrics
                st.session_state["eval_ckpt_name"] = selected_ckpt

                # Summary metrics
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
                mc2.metric("Top-3 Accuracy", f"{metrics['top3_accuracy']:.2%}")
                mc3.metric("Test Samples", metrics["n_samples"])
                mc4.metric("Classes", metrics["n_classes"])

                # Confusion matrix
                st.subheader("Confusion Matrix")
                cm_fig = confusion_matrix_figure(
                    metrics["confusion_matrix"],
                    metrics["class_labels"],
                )
                st.plotly_chart(cm_fig, use_container_width=True)

                # Per-class F1
                st.subheader("Per-Class F1 Score")
                f1_fig = per_class_f1_figure(metrics["per_class"])
                st.plotly_chart(f1_fig, use_container_width=True)

                # Per-class table
                with st.expander("Per-class details"):
                    st.dataframe(
                        pd.DataFrame(metrics["per_class"]),
                        hide_index=True,
                        use_container_width=True,
                    )

                # Export
                if st.button("💾 Save evaluation report"):
                    json_path, csv_path = save_evaluation_report(
                        metrics, selected_ckpt, EVAL_DIR,
                    )
                    st.success(
                        f"Report saved:\n- `{json_path.name}`\n- `{csv_path.name}`\n\n"
                        f"Directory: `{EVAL_DIR}`"
                    )

        # Show cached results if available
        elif "eval_metrics" in st.session_state:
            metrics = st.session_state["eval_metrics"]
            cached_name = st.session_state.get("eval_ckpt_name", "")
            if cached_name == selected_ckpt:
                from spj.evaluator import confusion_matrix_figure, per_class_f1_figure

                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
                mc2.metric("Top-3 Accuracy", f"{metrics['top3_accuracy']:.2%}")
                mc3.metric("Test Samples", metrics["n_samples"])
                mc4.metric("Classes", metrics["n_classes"])

                st.subheader("Confusion Matrix")
                st.plotly_chart(
                    confusion_matrix_figure(metrics["confusion_matrix"], metrics["class_labels"]),
                    use_container_width=True,
                )

                st.subheader("Per-Class F1 Score")
                st.plotly_chart(
                    per_class_f1_figure(metrics["per_class"]),
                    use_container_width=True,
                )


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — COMPARE
# ══════════════════════════════════════════════════════════════════════════
with tab_compare:
    from spj.trainer import list_checkpoints as _list_ckpts

    ckpts = _list_ckpts(MODELS_DIR)

    if len(ckpts) < 2:
        st.info("Need at least 2 checkpoints to compare. Train more models on page 8.")
    elif not (SPLITS_DIR / "test.csv").exists():
        st.warning("No test split found. Split the data on page 8 first.")
    else:
        selected = st.multiselect(
            "Select checkpoints to compare",
            [c["filename"] for c in ckpts],
            default=[c["filename"] for c in ckpts[:2]],
            key="compare_ckpts",
        )

        if len(selected) < 2:
            st.info("Select at least 2 checkpoints.")
        elif st.button("▶ Compare", type="primary"):
            from spj.trainer import load_checkpoint
            from spj.evaluator import evaluate_model, compare_models_table

            test_df = pd.read_csv(SPLITS_DIR / "test.csv", dtype=str)
            if "label" not in test_df.columns:
                test_df["label"] = test_df["reviewed_text"].where(
                    test_df["reviewed_text"].str.strip() != "",
                    test_df["text"],
                )

            evaluations = []
            progress = st.progress(0.0, text="Evaluating models…")

            for i, name in enumerate(selected):
                progress.progress(
                    (i + 1) / len(selected),
                    text=f"Evaluating {name}…",
                )
                ckpt_info = next(c for c in ckpts if c["filename"] == name)
                ckpt_path = Path(ckpt_info["path"])
                model, label_encoder, config, meta = load_checkpoint(ckpt_path)

                metrics = evaluate_model(
                    model, label_encoder, test_df, EXPORT_DIR,
                    max_seq_len=config.max_seq_len,
                )
                evaluations.append((name, metrics))

            progress.progress(1.0, text="Done")

            # Comparison table
            st.subheader("Model Comparison")
            comp_df = compare_models_table(evaluations)
            st.dataframe(comp_df, hide_index=True, use_container_width=True)

            # Overlay accuracy
            import plotly.graph_objects as go
            fig = go.Figure()
            colors = ["steelblue", "coral", "limegreen", "gold", "orchid"]
            for i, (name, metrics) in enumerate(evaluations):
                ckpt_info = next(c for c in ckpts if c["filename"] == name)
                fig.add_trace(go.Bar(
                    name=name,
                    x=["Accuracy", "Top-3 Accuracy"],
                    y=[metrics.get("accuracy", 0), metrics.get("top3_accuracy", 0)],
                    marker_color=colors[i % len(colors)],
                ))
            fig.update_layout(
                barmode="group",
                yaxis=dict(range=[0, 1], title="Score"),
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)
