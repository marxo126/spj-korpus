"""Training page — pretrain, cluster, split, train, and manage checkpoints.

Tab 1 (Pretrain):     Self-supervised pretraining on unlabeled pose data
Tab 2 (Cluster):      Extract features + cluster similar segments
Tab 3 (Data & Split): Load manifest, show label distribution, configure split
Tab 4 (Train):        Select backbone, set hyperparameters, train with live curves
Tab 5 (Checkpoints):  Browse saved .pt files with metadata
"""
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

DATA_DIR      = Path(__file__).parent.parent.parent / "data"
TRAINING_DIR  = DATA_DIR / "training"
EXPORT_DIR    = TRAINING_DIR / "export"
SPLITS_DIR    = TRAINING_DIR / "splits"
MODELS_DIR    = DATA_DIR / "models"
POSE_DIR      = DATA_DIR / "pose"
MANIFEST_CSV  = EXPORT_DIR / "manifest.csv"

st.header("Training")
st.caption("Page 8/10 · Pretrain, cluster, split data, train model, manage checkpoints.")
with st.expander("How to use this page", expanded=False):
    st.markdown("""
**Tab Pretrain**
- Self-supervised pretraining on all .pose files (no labels needed).
- Masks random frames and learns to reconstruct them.
- Produces pretrained encoder weights for better downstream training.

**Tab Cluster**
- Extract features from pretrained encoder, cluster similar segments.
- Name clusters to create pseudo-labels (much less manual work).

**Tab Data & Split**
1. Review label distribution and milestone progress.
2. Set train/val/test ratio (default 80/10/10).
3. Click **Split** to create stratified split CSVs.

**Tab Train**
1. Optionally select a pretrained backbone.
2. Adjust hyperparameters if needed (defaults are M4 Max optimised).
3. Click **Start Training** — live loss/accuracy curves update per epoch.

**Tab Checkpoints**
- Browse saved `.pt` files with metadata (backbone, classes, accuracy).
""")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_manifest() -> pd.DataFrame | None:
    if MANIFEST_CSV.exists():
        df = pd.read_csv(MANIFEST_CSV, dtype=str)
        if "label" not in df.columns:
            df["label"] = df["reviewed_text"].where(
                df["reviewed_text"].str.strip() != "",
                df["text"],
            )
        return df
    return None


def _count_pose_files() -> int:
    if not POSE_DIR.exists():
        return 0
    return len([f for f in POSE_DIR.glob("*.pose") if f.stat().st_size > 0])


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_pretrain, tab_cluster, tab_data, tab_train, tab_checkpoints = st.tabs(
    ["Pretrain", "Cluster", "Data & Split", "Train", "Checkpoints"]
)

# ======================================================================
# TAB 1 — PRETRAIN (Self-Supervised)
# ======================================================================
with tab_pretrain:
    n_pose = _count_pose_files()

    st.subheader("Masked Pose Modeling")
    st.caption(
        "Self-supervised pretraining: mask random frames of pose sequences "
        "and train the encoder to reconstruct them. No labels needed."
    )

    if n_pose == 0:
        st.warning("No .pose files found in `data/pose/`. Run Pose Extraction (page 2) first.")
    else:
        st.metric("Available .pose files", n_pose)

        # Check if pretraining is in progress
        pt_state = st.session_state.get("pt_state")

        if pt_state is not None and (pt_state.running or pt_state.finished):
            state = pt_state

            if state.error:
                st.error(f"Pretraining error: {state.error}")

            elif state.finished:
                st.success(
                    f"Pretraining complete! Best loss: **{state.best_loss:.6f}** "
                    f"at epoch {state.best_epoch}\n\n"
                    f"Checkpoint: `{state.checkpoint_path}`"
                )
                if st.button("Reset for new pretraining", key="pt_reset"):
                    st.session_state.pop("pt_state", None)
                    st.session_state.pop("pt_thread", None)
                    st.rerun()

            else:
                st.info(f"Pretraining epoch {state.epoch} / {state.total_epochs}...")
                st.progress(
                    state.epoch / max(1, state.total_epochs),
                    text=f"Epoch {state.epoch} / {state.total_epochs} ({state.n_windows} windows)",
                )
                if st.button("Stop pretraining", type="secondary", key="pt_stop"):
                    state.stop_requested = True
                    st.info("Stop requested — will finish current epoch.")

            # Live loss curve
            if state.losses:
                import plotly.graph_objects as go
                epochs = list(range(1, len(state.losses) + 1))
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=epochs, y=state.losses, name="Reconstruction Loss",
                    line=dict(color="steelblue"),
                ))
                fig.update_layout(
                    title="Pretraining Loss (MSE on masked frames)",
                    xaxis_title="Epoch", yaxis_title="Loss",
                    height=300,
                )
                st.plotly_chart(fig, use_container_width=True)

            # Auto-refresh while running
            if state.running:
                time.sleep(1.0)
                st.rerun()

        else:
            # Configuration UI
            with st.expander("Pretraining config", expanded=False):
                pc1, pc2 = st.columns(2)
                pt_mask_ratio = pc1.number_input(
                    "Mask ratio", 0.05, 0.5, 0.15, step=0.05, key="pt_mask",
                    help="Fraction of frames to mask per sequence",
                )
                pt_epochs = pc2.number_input("Epochs", 1, 200, 30, key="pt_epochs")
                pt_lr = pc1.number_input(
                    "Learning rate", 1e-5, 1e-2, 1e-4, format="%.1e", key="pt_lr",
                )
                pt_batch = pc2.number_input("Batch size", 8, 512, 128, step=8, key="pt_batch")
                pt_d_model = pc1.number_input("d_model", 32, 1024, 256, step=32, key="pt_dmodel")
                pt_n_layers = pc2.number_input("Layers", 1, 12, 4, key="pt_nlayers")

            if st.button("Start Pretraining", type="primary", key="pt_start"):
                from spj.ssl_pretrain import PretrainConfig, PretrainState, pretrain_masked_pose

                config = PretrainConfig(
                    mask_ratio=pt_mask_ratio,
                    lr=pt_lr,
                    epochs=pt_epochs,
                    batch_size=pt_batch,
                    d_model=pt_d_model,
                    n_layers=pt_n_layers,
                )
                state = PretrainState()
                st.session_state["pt_state"] = state

                thread = threading.Thread(
                    target=pretrain_masked_pose,
                    args=(POSE_DIR, config, state, MODELS_DIR),
                    daemon=True,
                )
                thread.start()
                st.session_state["pt_thread"] = thread
                st.rerun()

        # List existing pretrained checkpoints
        st.divider()
        st.subheader("Pretrained checkpoints")
        from spj.ssl_pretrain import list_pretrain_checkpoints
        pt_ckpts = list_pretrain_checkpoints(MODELS_DIR)
        if pt_ckpts:
            st.dataframe(
                pd.DataFrame(pt_ckpts)[[
                    "filename", "d_model", "n_layers", "mask_ratio",
                    "epoch", "loss", "n_windows", "timestamp",
                ]],
                hide_index=True, use_container_width=True,
            )
        else:
            st.info("No pretrained checkpoints yet.")


# ======================================================================
# TAB 2 — CLUSTER
# ======================================================================
with tab_cluster:
    st.subheader("Feature Clustering")
    st.caption(
        "Extract features from a pretrained encoder, cluster similar segments, "
        "and name clusters to create pseudo-labels."
    )

    manifest = _load_manifest()
    from spj.ssl_pretrain import list_pretrain_checkpoints
    pt_ckpts = list_pretrain_checkpoints(MODELS_DIR)

    if not pt_ckpts:
        st.warning("No pretrained checkpoints found. Run pretraining in the **Pretrain** tab first.")
    elif manifest is None:
        st.warning(
            "No `manifest.csv` found. Go to page 7 (Training Data) and export segments first."
        )
    elif not EXPORT_DIR.exists() or not list(EXPORT_DIR.glob("*.npz")):
        st.warning("No .npz segment files found. Export segments on page 7 first.")
    else:
        # Checkpoint selection
        selected_pt = st.selectbox(
            "Pretrained checkpoint",
            [c["filename"] for c in pt_ckpts],
            format_func=lambda f: (
                f"{f} — loss: {next(c['loss'] for c in pt_ckpts if c['filename'] == f):.6f}, "
                f"epoch: {next(c['epoch'] for c in pt_ckpts if c['filename'] == f)}"
            ),
            key="cl_ckpt",
        )

        # Clustering config
        with st.expander("Clustering config", expanded=False):
            cc1, cc2 = st.columns(2)
            cl_n_clusters = cc1.number_input(
                "Number of clusters", 0, 500, 0, key="cl_nclusters",
                help="0 = auto (HDBSCAN), >0 = k-means with fixed number",
            )
            cl_min_cluster = cc2.number_input(
                "Min cluster size (HDBSCAN)", 2, 50, 5, key="cl_minsize",
            )

        # Check for cached cluster result
        cluster_result = st.session_state.get("cl_result")

        if st.button("Extract Features & Cluster", type="primary", key="cl_run"):
            from spj.ssl_pretrain import MaskedPoseModel, PretrainConfig, load_pretrained_encoder
            from spj.clustering import (
                ClusterConfig, ClusterResult, extract_features, cluster_segments,
            )
            import torch

            ckpt_info = next(c for c in pt_ckpts if c["filename"] == selected_pt)
            ckpt_path = Path(ckpt_info["path"])

            with st.spinner("Loading pretrained encoder..."):
                ckpt = load_pretrained_encoder(ckpt_path)
                ckpt_config = ckpt.get("config", {})
                pt_config = PretrainConfig(
                    d_model=ckpt_config.get("d_model", 256),
                    n_heads=ckpt_config.get("n_heads", 4),
                    d_ff=ckpt_config.get("d_ff", 512),
                    n_layers=ckpt_config.get("n_layers", 4),
                    dropout=ckpt_config.get("dropout", 0.1),
                    max_seq_len=ckpt_config.get("max_seq_len", 300),
                )
                model = MaskedPoseModel(pt_config)
                model.load_state_dict(ckpt["full_state_dict"])
                device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
                model = model.to(device)
                model.eval()

            cl_config = ClusterConfig(
                n_clusters=cl_n_clusters,
                min_cluster_size=cl_min_cluster,
                max_seq_len=pt_config.max_seq_len,
            )

            with st.spinner("Extracting features from segments..."):
                seg_ids, features = extract_features(
                    model, manifest, EXPORT_DIR, cl_config, device,
                )

            if len(seg_ids) == 0:
                st.error("No segments could be loaded. Check NPZ files in export directory.")
            else:
                with st.spinner("Clustering..."):
                    labels, centroids = cluster_segments(features, cl_config)

                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

                cluster_result = ClusterResult(
                    segment_ids=seg_ids,
                    features=features,
                    labels=labels,
                    n_clusters=n_clusters,
                    centroids=centroids,
                )
                st.session_state["cl_result"] = cluster_result
                st.success(f"Clustered {len(seg_ids)} segments into {n_clusters} clusters.")

        # Display cluster results
        if cluster_result is not None:
            from spj.clustering import cluster_summary, reduce_dimensions, apply_cluster_labels

            n_clusters = cluster_result.n_clusters
            noise_count = int((cluster_result.labels == -1).sum())

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Segments", len(cluster_result.segment_ids))
            mc2.metric("Clusters", n_clusters)
            mc3.metric("Noise points", noise_count)

            # 2D scatter plot
            st.subheader("Cluster visualisation")
            viz_method = st.radio(
                "Dimensionality reduction", ["t-SNE", "PCA"],
                horizontal=True, key="cl_viz_method",
            )
            with st.spinner("Computing 2D projection..."):
                coords_2d = reduce_dimensions(
                    cluster_result.features,
                    method="tsne" if viz_method == "t-SNE" else "pca",
                )

            import plotly.express as px
            viz_df = pd.DataFrame({
                "x": coords_2d[:, 0],
                "y": coords_2d[:, 1],
                "cluster": [str(l) for l in cluster_result.labels],
                "segment_id": cluster_result.segment_ids,
            })
            fig = px.scatter(
                viz_df, x="x", y="y", color="cluster",
                hover_data=["segment_id"],
                title=f"{viz_method} projection ({n_clusters} clusters)",
                height=450,
            )
            fig.update_layout(legend_title="Cluster")
            st.plotly_chart(fig, use_container_width=True)

            # Cluster summary table
            st.subheader("Cluster details")
            summary_df = cluster_summary(cluster_result, manifest)
            st.dataframe(summary_df, hide_index=True, use_container_width=True)

            # Cluster naming
            st.subheader("Name clusters")
            st.caption(
                "Assign a gloss name to each cluster. Named clusters become pseudo-labels "
                "for training. Leave blank to skip a cluster."
            )

            cluster_names = {}
            # Load previously saved names if any
            saved_names = st.session_state.get("cl_names", {})

            unique_clusters = sorted(set(cluster_result.labels))
            unique_clusters = [c for c in unique_clusters if c != -1]  # exclude noise

            for cid in unique_clusters:
                size = int((cluster_result.labels == cid).sum())
                # Get sample texts for hint
                row = summary_df[summary_df["cluster_id"] == cid]
                hint = row["sample_texts"].values[0] if len(row) > 0 else ""
                default_val = saved_names.get(int(cid), "")

                name = st.text_input(
                    f"Cluster {cid} ({size} segments)",
                    value=default_val,
                    placeholder=hint[:80] if hint else "GLOSS_NAME",
                    key=f"cl_name_{cid}",
                )
                if name.strip():
                    cluster_names[int(cid)] = name.strip()

            # Save names to session state
            st.session_state["cl_names"] = cluster_names

            ac1, ac2 = st.columns(2)

            if ac1.button("Apply Labels to Manifest", type="primary", key="cl_apply"):
                if not cluster_names:
                    st.warning("No clusters named yet. Enter gloss names above.")
                else:
                    updated_df = apply_cluster_labels(
                        cluster_result, cluster_names, manifest,
                    )
                    # Save updated manifest
                    updated_df.to_csv(MANIFEST_CSV, index=False)
                    st.success(
                        f"Applied {len(cluster_names)} cluster labels to manifest. "
                        f"Updated `{MANIFEST_CSV}`."
                    )

            if ac2.button("Export Labeled Manifest", key="cl_export"):
                export_path = EXPORT_DIR / "manifest_clustered.csv"
                if manifest is not None:
                    manifest_fresh = pd.read_csv(MANIFEST_CSV, dtype=str)
                    manifest_fresh.to_csv(export_path, index=False)
                    st.success(f"Exported to `{export_path}`")


# ======================================================================
# TAB 3 — DATA & SPLIT
# ======================================================================
with tab_data:
    manifest = _load_manifest()

    if manifest is None:
        st.warning(
            "No `manifest.csv` found. Go to page 7 (Training Data) -> Export tab "
            "and click **Export CSV manifest** first."
        )
    else:
        # Derive label column
        labels = manifest["label"].dropna()
        labels = labels[labels.str.strip() != ""]

        if labels.empty:
            st.warning("Manifest has no labels. Review and export segments on page 7.")
        else:
            label_counts = labels.value_counts()
            total_signs = len(labels)
            n_classes = len(label_counts)

            st.subheader("Dataset overview")
            m1, m2 = st.columns(2)
            m1.metric("Total segments", total_signs)
            m2.metric("Unique labels", n_classes)

            # Milestone progress
            st.subheader("Annotation milestones")
            milestones = [500, 2000, 5000, 10000]
            cols = st.columns(len(milestones))
            for col, ms in zip(cols, milestones):
                pct = min(100, total_signs / ms * 100)
                col.metric(f"{ms:,} signs", f"{total_signs:,}")
                col.progress(pct / 100, text=f"{pct:.0f}%")

            # Label distribution chart
            st.subheader("Label distribution")
            import plotly.express as px
            top_n = min(50, len(label_counts))
            fig = px.bar(
                x=label_counts.head(top_n).index,
                y=label_counts.head(top_n).values,
                labels={"x": "Label", "y": "Count"},
            )
            fig.update_layout(height=350, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

            if len(label_counts) > top_n:
                st.caption(f"Showing top {top_n} of {len(label_counts)} labels.")

            # Split configuration
            st.divider()
            st.subheader("Data split")

            sc1, sc2, sc3 = st.columns(3)
            train_pct = sc1.number_input("Train %", 50, 95, 80, step=5, key="split_train")
            val_pct = sc2.number_input("Val %", 0, 30, 10, step=5, key="split_val")
            test_pct = sc3.number_input("Test %", 0, 30, 10, step=5, key="split_test")

            if abs(train_pct + val_pct + test_pct - 100) > 0.1:
                st.error(f"Ratios must sum to 100% (currently {train_pct + val_pct + test_pct}%)")
            else:
                if st.button("Split dataset", type="primary"):
                    from spj.trainer import split_dataset

                    train_df, val_df, test_df = split_dataset(
                        manifest,
                        train_ratio=train_pct / 100,
                        val_ratio=val_pct / 100,
                        test_ratio=test_pct / 100,
                        output_dir=SPLITS_DIR,
                    )
                    st.session_state["tr_train_df"] = train_df
                    st.session_state["tr_val_df"] = val_df
                    st.session_state["tr_test_df"] = test_df

                    rc1, rc2, rc3 = st.columns(3)
                    rc1.metric("Train", len(train_df))
                    rc2.metric("Val", len(val_df))
                    rc3.metric("Test", len(test_df))
                    st.success(f"Splits saved to `{SPLITS_DIR}`")

            # Show existing splits if available
            if (SPLITS_DIR / "train.csv").exists():
                existing_train = pd.read_csv(SPLITS_DIR / "train.csv")
                existing_val = pd.read_csv(SPLITS_DIR / "val.csv")
                existing_test = pd.read_csv(SPLITS_DIR / "test.csv")
                st.caption(
                    f"Current splits: train={len(existing_train)}, "
                    f"val={len(existing_val)}, test={len(existing_test)}"
                )

# ======================================================================
# TAB 4 — TRAIN
# ======================================================================
with tab_train:
    # Check prerequisites
    has_splits = (SPLITS_DIR / "train.csv").exists()

    if not has_splits:
        st.warning("No data splits found. Go to the **Data & Split** tab and split the dataset first.")
    else:
        # Load splits
        if "tr_train_df" not in st.session_state:
            st.session_state["tr_train_df"] = pd.read_csv(SPLITS_DIR / "train.csv", dtype=str)
        if "tr_val_df" not in st.session_state:
            st.session_state["tr_val_df"] = pd.read_csv(SPLITS_DIR / "val.csv", dtype=str)

        train_df = st.session_state["tr_train_df"]
        val_df = st.session_state["tr_val_df"]

        # Check if training is in progress
        training_state = st.session_state.get("tr_state")

        if training_state is not None and (training_state.running or training_state.finished):
            # -- Training in progress or just finished --
            state = training_state

            if state.error:
                st.error(f"Training error: {state.error}")

            elif state.finished:
                st.success(
                    f"Training complete! Best val accuracy: **{state.best_val_acc:.2%}** "
                    f"at epoch {state.best_epoch}\n\n"
                    f"Checkpoint: `{state.checkpoint_path}`"
                )
                if st.button("Reset for new training"):
                    st.session_state.pop("tr_state", None)
                    st.session_state.pop("tr_thread", None)
                    st.rerun()

            else:
                st.info(
                    f"Training epoch {state.epoch} / {state.total_epochs}..."
                )
                st.progress(
                    state.epoch / max(1, state.total_epochs),
                    text=f"Epoch {state.epoch} / {state.total_epochs}",
                )
                if st.button("Stop training", type="secondary"):
                    state.stop_requested = True
                    st.info("Stop requested — will finish current epoch.")

            # Live loss/accuracy curves
            if state.train_losses:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots

                fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "Accuracy"))
                epochs = list(range(1, len(state.train_losses) + 1))

                fig.add_trace(
                    go.Scatter(x=epochs, y=state.train_losses, name="Train Loss",
                               line=dict(color="steelblue")),
                    row=1, col=1,
                )
                if state.val_losses:
                    fig.add_trace(
                        go.Scatter(x=epochs, y=state.val_losses, name="Val Loss",
                                   line=dict(color="coral")),
                        row=1, col=1,
                    )
                fig.add_trace(
                    go.Scatter(x=epochs, y=state.train_accs, name="Train Acc",
                               line=dict(color="steelblue")),
                    row=1, col=2,
                )
                if state.val_accs:
                    fig.add_trace(
                        go.Scatter(x=epochs, y=state.val_accs, name="Val Acc",
                                   line=dict(color="coral")),
                        row=1, col=2,
                    )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

            # Auto-refresh while running
            if state.running:
                time.sleep(1.0)
                st.rerun()

        else:
            # -- Idle — configuration UI --
            st.subheader("Model configuration")

            # Backbone selection — now includes pretrained option
            from spj.ssl_pretrain import list_pretrain_checkpoints
            pt_ckpts = list_pretrain_checkpoints(MODELS_DIR)

            backbone_options = ["From scratch (Transformer)"]
            if pt_ckpts:
                backbone_options.insert(0, "Pretrained (SSL)")
            backbone_options.extend(["SignBERT", "OpenHands"])

            backbone = st.selectbox(
                "Backbone",
                backbone_options,
                index=0,
                help=(
                    "**Pretrained (SSL)**: Initialize from self-supervised pretrained weights.\n\n"
                    "**From scratch**: PoseTransformerEncoder trained on your data only.\n\n"
                    "**SignBERT / OpenHands**: Attempts to load external pretrained checkpoint."
                ),
                key="tr_backbone",
            )

            # Pretrained checkpoint selection
            pretrained_path = None
            if backbone == "Pretrained (SSL)" and pt_ckpts:
                selected_pt = st.selectbox(
                    "Pretrained checkpoint",
                    [c["filename"] for c in pt_ckpts],
                    format_func=lambda f: (
                        f"{f} — loss: {next(c['loss'] for c in pt_ckpts if c['filename'] == f):.6f}"
                    ),
                    key="tr_pt_ckpt",
                )
                pretrained_path = Path(
                    next(c["path"] for c in pt_ckpts if c["filename"] == selected_pt)
                )

            backbone_key_map = {
                "From scratch (Transformer)": "from_scratch",
                "Pretrained (SSL)": "pretrained_ssl",
                "SignBERT": "signbert",
                "OpenHands": "openhands",
            }
            backbone_key = backbone_key_map.get(backbone, "from_scratch")

            with st.expander("Hyperparameters", expanded=False):
                hc1, hc2 = st.columns(2)
                lr = hc1.number_input("Learning rate", 1e-5, 1e-1, 1e-3, format="%.1e", key="tr_lr")
                epochs = hc2.number_input("Epochs", 1, 500, 50, key="tr_epochs")
                batch_size = hc1.number_input("Batch size", 8, 1024, 256, step=8, key="tr_batch")
                d_model = hc2.number_input("d_model", 32, 1024, 256, step=32, key="tr_dmodel")
                n_layers = hc1.number_input("Layers", 1, 12, 4, key="tr_nlayers")
                n_heads = hc2.number_input("Heads", 1, 16, 4, key="tr_nheads")
                d_ff = hc1.number_input("d_ff", 64, 4096, 512, step=64, key="tr_dff")
                dropout = hc2.number_input("Dropout", 0.0, 0.5, 0.1, step=0.05, key="tr_dropout")
                max_seq_len = hc1.number_input("Max seq len", 50, 1000, 300, step=50, key="tr_maxseq")

            st.caption(
                f"Training: **{len(train_df)}** samples, "
                f"Validation: **{len(val_df)}** samples, "
                f"Device: **MPS** (M4 Max)"
            )

            if st.button("Start Training", type="primary"):
                from spj.trainer import (
                    LabelEncoder, TrainingConfig, TrainingState, train_model,
                )

                # Build label encoder from train split
                if "label" not in train_df.columns:
                    train_df["label"] = train_df["reviewed_text"].where(
                        train_df["reviewed_text"].str.strip() != "",
                        train_df["text"],
                    )
                if "label" not in val_df.columns:
                    val_df["label"] = val_df["reviewed_text"].where(
                        val_df["reviewed_text"].str.strip() != "",
                        val_df["text"],
                    )

                all_labels = train_df["label"].dropna().tolist()
                all_labels = [l for l in all_labels if str(l).strip()]
                label_encoder = LabelEncoder(all_labels)

                config = TrainingConfig(
                    backbone=backbone_key,
                    lr=lr,
                    epochs=epochs,
                    batch_size=batch_size,
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    n_layers=n_layers,
                    dropout=dropout,
                    max_seq_len=max_seq_len,
                )

                state = TrainingState()
                st.session_state["tr_state"] = state

                if backbone_key in ("signbert", "openhands"):
                    st.info(
                        f"Note: {backbone} pretrained checkpoint not available — "
                        f"training from scratch with Transformer architecture."
                    )

                thread = threading.Thread(
                    target=train_model,
                    args=(train_df, val_df, EXPORT_DIR, label_encoder, config, state, MODELS_DIR),
                    kwargs={"pretrained_path": pretrained_path},
                    daemon=True,
                )
                thread.start()
                st.session_state["tr_thread"] = thread
                st.rerun()


# ======================================================================
# TAB 5 — CHECKPOINTS
# ======================================================================
with tab_checkpoints:
    from spj.trainer import list_checkpoints

    ckpts = list_checkpoints(MODELS_DIR)

    if not ckpts:
        st.info("No checkpoints found in `data/models/`. Train a model first.")
    else:
        st.subheader(f"{len(ckpts)} checkpoint(s)")

        ckpt_df = pd.DataFrame(ckpts)
        display_cols = [c for c in [
            "filename", "backbone", "n_classes", "val_acc", "epoch", "n_train", "timestamp",
        ] if c in ckpt_df.columns]

        st.dataframe(
            ckpt_df[display_cols],
            hide_index=True,
            use_container_width=True,
        )

        # Detail view
        selected = st.selectbox(
            "View details",
            [c["filename"] for c in ckpts],
            key="ckpt_detail",
        )
        if selected:
            detail = next(c for c in ckpts if c["filename"] == selected)
            st.json(detail)
