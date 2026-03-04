"""Inference page — predict glosses on new videos and preview results.

Tab 1 (Predict):  Select checkpoint + videos, run inference, write to EAF
Tab 2 (Preview):  Video player + timeline + prepartner-dictns table
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

DATA_DIR        = Path(__file__).parent.parent.parent / "data"
INVENTORY_CSV   = DATA_DIR / "inventory.csv"
POSE_DIR        = DATA_DIR / "pose"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
MODELS_DIR      = DATA_DIR / "models"

st.header("🔮 Inference")
st.caption(
    "Page 10/10 · Predict glosses on new videos, write results to ELAN EAF files."
)
with st.expander("ℹ️ How to use this page", expanded=False):
    st.markdown("""
**Needs:**
- A trained model checkpoint in `data/models/` (from page 8)
- Videos with extracted pose files (pages 1-2)
- EAF annotation files (page 3)

**Tab 🎯 Predict**
1. Select a checkpoint.
2. Select video(s) from the inventory.
3. Click **▶ Run Inference** — detects sign segments, classifies each, writes to EAF.
4. Open the EAF file in ELAN to review AI prepartner-dictns.
5. Corrections flow back to training data via page 7 → retrain on page 8.

**Tab 👁 Preview**
- View prepartner-dictns timeline overlaid on the video.
- Browse the prepartner-dictns table.

**This closes the active learning loop:**
Inference → ELAN review → re-export on Page 7 → retrain on Page 8.
""")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_resource
def _load_model_cached(ckpt_path: str):
    """Load model checkpoint once, shared across reruns."""
    from spj.trainer import load_checkpoint
    return load_checkpoint(Path(ckpt_path))


@st.cache_resource
def _load_pose_cached(pose_path: str):
    """Load pose arrays once per video, shared across reruns."""
    from spj.preannotate import load_pose_arrays
    return load_pose_arrays(Path(pose_path))


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_predict, tab_preview = st.tabs(["🎯 Predict", "👁 Preview"])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════
with tab_predict:
    from spj.trainer import list_checkpoints

    ckpts = list_checkpoints(MODELS_DIR)

    if not ckpts:
        st.warning("No model checkpoints found. Train a model on page 8 first.")
    else:
        # Checkpoint selection
        selected_ckpt = st.selectbox(
            "Select checkpoint",
            [c["filename"] for c in ckpts],
            format_func=lambda f: (
                f"{f} — {next(c['n_classes'] for c in ckpts if c['filename'] == f)} classes, "
                f"acc: {next(c['val_acc'] for c in ckpts if c['filename'] == f):.2%}"
            ),
            key="inf_ckpt",
        )

        # Load inventory
        inv: pd.DataFrame | None = st.session_state.get("inventory")
        if inv is None:
            if INVENTORY_CSV.exists():
                inv = pd.read_csv(INVENTORY_CSV)
                st.session_state["inventory"] = inv

        if inv is None or inv.empty:
            st.warning("No inventory found. Go to page 1 (Inventory) first.")
        else:
            # Filter to videos with pose files
            stems = [Path(str(p)).stem for p in inv["path"]]
            has_pose = [(POSE_DIR / f"{s}.pose").exists() for s in stems]
            inv_aug = inv.copy()
            inv_aug["has_pose"] = has_pose
            inv_aug["stem"] = stems

            pose_ready = inv_aug[inv_aug["has_pose"]]

            if pose_ready.empty:
                st.warning("No videos have pose files. Run Pose Extraction (page 2) first.")
            else:
                # Video selection
                video_options = pose_ready["stem"].tolist()
                selected_videos = st.multiselect(
                    "Select video(s)",
                    video_options,
                    default=video_options[:1] if video_options else [],
                    key="inf_videos",
                )

                if not selected_videos:
                    st.info("Select at least one video to run inference.")

                bc1, bc2 = st.columns(2)
                run_selected = bc1.button(
                    "▶ Run Inference",
                    type="primary",
                    disabled=not selected_videos,
                )
                run_batch = bc2.button(
                    f"▶ Batch All ({len(pose_ready)})",
                    type="secondary",
                )

                if run_selected or run_batch:
                    if run_batch:
                        target_stems = pose_ready["stem"].tolist()
                    else:
                        target_stems = selected_videos

                    from spj.trainer import load_checkpoint
                    from spj.preannotate import load_pose_arrays, detect_sign_segments
                    from spj.inference import predict_segments, write_prepartner-dictns_to_eaf
                    from spj.eaf import create_empty_eaf

                    ckpt_info = next(c for c in ckpts if c["filename"] == selected_ckpt)
                    ckpt_path = Path(ckpt_info["path"])

                    with st.spinner("Loading model…"):
                        model, label_encoder, config, meta = _load_model_cached(str(ckpt_path))

                    # Detect landmark preset from model's input dimension
                    from spj.training_data import SL_LANDMARK_PRESETS, preset_from_input_dim
                    _input_dim = model.input_proj.weight.shape[1]
                    _preset_name = preset_from_input_dim(_input_dim)
                    _lm_indices = SL_LANDMARK_PRESETS[_preset_name] if _preset_name else None
                    if _preset_name:
                        st.caption(f"Landmark preset: **{_preset_name}** ({len(_lm_indices)} landmarks, dim={_input_dim})")

                    all_results = []
                    progress = st.progress(0.0, text="Running inference…")

                    # Build O(1) lookup to avoid O(N^2) DataFrame scan per video
                    _stem_rows = {r["stem"]: r for _, r in pose_ready.iterrows()}

                    for i, stem in enumerate(target_stems):
                        progress.progress(
                            (i + 1) / len(target_stems),
                            text=f"Processing {stem}… ({i + 1}/{len(target_stems)})",
                        )

                        video_row = _stem_rows[stem]
                        video_path = Path(str(video_row["path"]))
                        pose_path = POSE_DIR / f"{stem}.pose"
                        eaf_path = ANNOTATIONS_DIR / f"{stem}.eaf"

                        try:
                            # Load pose
                            data, conf, fps = load_pose_arrays(pose_path)
                            T = data.shape[0]
                            duration_sec = T / fps if fps > 0 else 0

                            # Detect sign segments (kinematic)
                            segments = detect_sign_segments(data, conf, fps)

                            n_segments = len(segments.get("right", [])) + len(segments.get("left", []))

                            if n_segments == 0:
                                all_results.append({
                                    "video": stem,
                                    "n_segments": 0,
                                    "mean_confidence": 0,
                                    "status": "No segments detected",
                                })
                                continue

                            # Classify segments
                            prepartner-dictns = predict_segments(
                                model, label_encoder, segments,
                                data, fps,
                                max_seq_len=config.max_seq_len,
                                landmark_indices=_lm_indices,
                            )

                            # Ensure EAF exists
                            if not eaf_path.exists():
                                ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
                                eaf = create_empty_eaf(video_path, eaf_path)
                                from spj.eaf import save_eaf
                                save_eaf(eaf, eaf_path)

                            # Write prepartner-dictns to EAF
                            write_result = write_prepartner-dictns_to_eaf(
                                prepartner-dictns, eaf_path, overwrite=True,
                            )

                            mean_conf = sum(
                                p["prepartner-dictn_confidence"] for p in prepartner-dictns
                            ) / max(1, len(prepartner-dictns))

                            all_results.append({
                                "video": stem,
                                "n_segments": len(prepartner-dictns),
                                "mean_confidence": round(mean_conf, 3),
                                "rh_prepartner-dictns": write_result["rh_prepartner-dictns"],
                                "lh_prepartner-dictns": write_result["lh_prepartner-dictns"],
                                "status": "OK",
                            })

                            # Cache prepartner-dictns for preview
                            st.session_state[f"inf_preds_{stem}"] = prepartner-dictns
                            st.session_state[f"inf_duration_{stem}"] = duration_sec

                        except Exception as exc:
                            all_results.append({
                                "video": stem,
                                "n_segments": 0,
                                "mean_confidence": 0,
                                "status": f"Error: {exc}",
                            })

                    progress.progress(1.0, text="Done")

                    # Summary table
                    st.subheader("Inference Results")
                    results_df = pd.DataFrame(all_results)
                    st.dataframe(results_df, hide_index=True, use_container_width=True)

                    ok_count = sum(1 for r in all_results if r["status"] == "OK")
                    total_preds = sum(r["n_segments"] for r in all_results)

                    st.success(
                        f"Processed **{ok_count}/{len(target_stems)}** videos, "
                        f"**{total_preds}** total prepartner-dictns written to EAF.\n\n"
                        f"Open the EAF files in **ELAN** to review AI prepartner-dictns. "
                        f"Corrections flow back to training data via page 7."
                    )

                    if run_batch:
                        mc1, mc2, mc3, mc4 = st.columns(4)
                        mc1.metric("Processed", ok_count)
                        mc2.metric("Skipped", sum(1 for r in all_results if r["status"] not in ("OK",) and not r["status"].startswith("Error")))
                        mc3.metric("Errors", sum(1 for r in all_results if r["status"].startswith("Error")))
                        mc4.metric("Prepartner-dictns", total_preds)

                    st.session_state["inf_results"] = all_results
                    st.session_state["inf_selected"] = target_stems


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — PREVIEW
# ══════════════════════════════════════════════════════════════════════════
with tab_preview:
    results = st.session_state.get("inf_results")

    if not results:
        st.info("No inference results yet. Run inference in the **Predict** tab first.")
    else:
        ok_videos = [r["video"] for r in results if r["status"] == "OK"]

        if not ok_videos:
            st.warning("No successful prepartner-dictns to preview.")
        else:
            selected_preview = st.selectbox(
                "Select video to preview",
                ok_videos,
                key="inf_preview_video",
            )

            if selected_preview:
                prepartner-dictns = st.session_state.get(f"inf_preds_{selected_preview}", [])
                duration_sec = st.session_state.get(f"inf_duration_{selected_preview}", 0)

                if not prepartner-dictns:
                    st.info("No prepartner-dictns cached for this video.")
                else:
                    # Video + pose viewer (synced if possible)
                    inv = st.session_state.get("inventory")
                    if inv is not None:
                        match = inv[inv["path"].apply(
                            lambda p: Path(str(p)).stem == selected_preview
                        )]
                        if not match.empty:
                            video_path = Path(str(match.iloc[0]["path"]))
                            pose_path = POSE_DIR / f"{selected_preview}.pose"

                            if video_path.exists() and pose_path.exists():
                                try:
                                    p_data, p_conf, p_fps = _load_pose_cached(
                                        str(pose_path),
                                    )
                                    T = p_data.shape[0]

                                    from spj.training_data import (
                                        extract_video_segment,
                                        synced_video_pose_html,
                                    )
                                    import streamlit.components.v1 as stc

                                    seg_bytes = extract_video_segment(
                                        video_path, 0,
                                        min(int(T / p_fps * 1000), 60_000),
                                    )
                                    if seg_bytes is not None:
                                        html = synced_video_pose_html(
                                            seg_bytes, p_data, p_conf,
                                            p_fps, 0, min(T, int(60 * p_fps)),
                                        )
                                        stc.html(html, height=500)
                                    else:
                                        st.video(str(video_path))
                                except Exception:
                                    st.video(str(video_path))
                            elif video_path.exists():
                                st.video(str(video_path))

                    # Timeline
                    from spj.inference import prepartner-dictns_timeline_figure

                    st.subheader("Prepartner-dictns Timeline")
                    timeline_fig = prepartner-dictns_timeline_figure(prepartner-dictns, duration_sec)
                    st.plotly_chart(timeline_fig, use_container_width=True)

                    # Prepartner-dictns table
                    st.subheader("Prepartner-dictns")
                    pred_rows = []
                    for p in prepartner-dictns:
                        top3_str = ", ".join(
                            f"{t['gloss']} ({t['confidence']:.0%})"
                            for t in p.get("top3_glosses", [])
                        )
                        pred_rows.append({
                            "Hand": p["hand"].capitalize(),
                            "Start": f"{p['start_ms'] / 1000:.2f}s",
                            "End": f"{p['end_ms'] / 1000:.2f}s",
                            "Predicted": p["predicted_gloss"],
                            "Confidence": f"{p['prepartner-dictn_confidence']:.0%}",
                            "Top 3": top3_str,
                        })

                    st.dataframe(
                        pd.DataFrame(pred_rows),
                        hide_index=True,
                        use_container_width=True,
                    )

                    st.info(
                        f"Total: **{len(prepartner-dictns)}** prepartner-dictns. "
                        f"Open `data/annotations/{selected_preview}.eaf` in ELAN to review."
                    )

                    # ----- Translation Section (MLX / Ollama) -----
                    st.divider()
                    st.subheader("Translate to Slovak")

                    # Detect available backends
                    from spj.mlx_translate import check_mlx
                    from spj.ollama_translate import check_ollama

                    mlx_ok, mlx_info = check_mlx()
                    ollama_ok, ollama_info = check_ollama()

                    backends = []
                    if mlx_ok:
                        backends.append("MLX (native Apple Silicon)")
                    if ollama_ok:
                        backends.append(f"Ollama (v{ollama_info})")

                    if not backends:
                        st.warning(
                            "No translation backend available.\n\n"
                            "**MLX (recommended for M4 Max):** `pip install mlx-lm`\n\n"
                            "**Ollama:** `brew install ollama` then `ollama serve`"
                        )
                    else:
                        backend = st.radio(
                            "Translation backend",
                            backends,
                            index=0,
                            horizontal=True,
                            key="trans_backend",
                        )
                        is_mlx = backend.startswith("MLX")

                        # Model selection
                        if is_mlx:
                            from spj.mlx_translate import list_recommended_models
                            rec_models = list_recommended_models()
                            model_name = st.selectbox(
                                "MLX model",
                                [m["name"] for m in rec_models],
                                format_func=lambda n: (
                                    f"{n.split('/')[-1]} — "
                                    f"{next((m['size'] for m in rec_models if m['name'] == n), '')} "
                                    f"({next((m['description'] for m in rec_models if m['name'] == n), '')})"
                                ),
                                key="mlx_model",
                            )
                        else:
                            from spj.ollama_translate import list_models
                            ollama_models = list_models()
                            if not ollama_models:
                                st.warning(
                                    "No Ollama models installed. "
                                    "Pull one with: `ollama pull llama3.2`"
                                )
                                model_name = None
                            else:
                                model_name = st.selectbox(
                                    "Ollama model",
                                    [m["name"] for m in ollama_models],
                                    key="ollama_model",
                                )

                        if model_name:
                            tc1, tc2 = st.columns(2)

                            if tc1.button("Translate prepartner-dictns", key="do_translate"):
                                if is_mlx:
                                    from spj.mlx_translate import batch_translate_prepartner-dictns as mlx_batch
                                    with st.spinner(f"Translating via MLX ({model_name.split('/')[-1]})..."):
                                        translated = mlx_batch(
                                            prepartner-dictns, model_id=model_name,
                                        )
                                else:
                                    from spj.ollama_translate import batch_translate_prepartner-dictns as ollama_batch
                                    with st.spinner(f"Translating via Ollama ({model_name})..."):
                                        translated = ollama_batch(
                                            prepartner-dictns, model=model_name,
                                        )

                                st.session_state[f"inf_translated_{selected_preview}"] = translated

                            # Show translations if available
                            translated = st.session_state.get(
                                f"inf_translated_{selected_preview}"
                            )
                            if translated:
                                st.subheader("Translations")
                                trans_rows = []
                                for p in translated:
                                    if "translation" in p:
                                        glosses_in_window = [
                                            t["predicted_gloss"] for t in translated
                                            if abs(t["start_ms"] - p["start_ms"]) < 5000
                                        ]
                                        trans_rows.append({
                                            "Time": f"{p['start_ms'] / 1000:.1f}s",
                                            "Glosses": " ".join(glosses_in_window[:8]),
                                            "Translation": p["translation"],
                                        })
                                if trans_rows:
                                    st.dataframe(
                                        pd.DataFrame(trans_rows),
                                        hide_index=True,
                                        use_container_width=True,
                                    )

                                # Write translations to EAF
                                if tc2.button("Write translations to EAF", key="write_trans_eaf"):
                                    eaf_path = ANNOTATIONS_DIR / f"{selected_preview}.eaf"
                                    if eaf_path.exists():
                                        from spj.eaf import load_eaf, save_eaf
                                        eaf = load_eaf(eaf_path)

                                        tier_name = "S1_Translation"
                                        if tier_name not in eaf.get_tier_names():
                                            eaf.add_tier(tier_name)

                                        written = 0
                                        for p in translated:
                                            if "translation" in p and p["translation"]:
                                                group_preds = [
                                                    t for t in translated
                                                    if abs(t["start_ms"] - p["start_ms"]) < 5000
                                                ]
                                                start_ms = min(t["start_ms"] for t in group_preds)
                                                end_ms = max(t["end_ms"] for t in group_preds)
                                                try:
                                                    eaf.add_annotation(
                                                        tier_name, start_ms, end_ms,
                                                        value=p["translation"],
                                                    )
                                                    written += 1
                                                except Exception:
                                                    pass

                                        save_eaf(eaf, eaf_path)
                                        st.success(
                                            f"Wrote {written} translation(s) to "
                                            f"`{eaf_path.name}` (S1_Translation tier)."
                                        )
                                    else:
                                        st.error(f"EAF file not found: {eaf_path}")
