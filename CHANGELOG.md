# Changelog

## [Unreleased]

### Added
- Dual-view video splitting for category-source — extract poses from both 60° and 90° camera angles separately
- `crop` parameter on `extract_pose()` — crop frames in-memory before MediaPipe processing
- `extract_pose_dual_view()` and `extract_pose_dual_view_batch()` in pose.py
- Standalone script `tools/split_category-source_poses.py` for batch dual-view extraction
- "Split dual-view" checkbox in Page 2 (Pose Extraction) — auto-filters to category-source videos
- Data augmentation wired into `train_model()` via `AugmentedPoseDataset` (10x variants: temporal crop, speed, noise, scale)
- `augment` and `n_augments` fields in `TrainingConfig`
- Augmentation controls in Page 8 (Training) with effective sample count display
- RH+LH unit grouping — auto-pair overlapping right/left hand prepartner-dictns as one review unit
- Sign type dropdown per unit: Sign (default), Classifier, Compound
- Classifier mode — each hand gets its own label independently
- Compound mode — link two sequential units as one meaning (shared compound ID)
- Multi-highlight timeline — paired prepartner-dictns highlighted simultaneously with synced trims/cuts
- Video-editor-style razor cut tool — split prepartner-dictns into visually separate blocks with gap
- Alt+drag to create new label regions on timeline
- Timeline tooltips for prepartner-dictns, trim handles, cut points
- Subtitle track display on timeline
- Right-click to delete cut points
- Keyboard shortcuts: C (cut at playhead), X (undo last cut)

### Changed
- AI Review: prepartner-dictn selectbox replaced with unit selectbox (shows RH+LH / RH / LH)
- Timeline component: `active_pred_indices` parameter for multi-prepartner-dictn highlighting
- Timeline cut visualization: split blocks with white borders replace overlay lines
- Scroll behavior: regular scroll = pan, Ctrl/Cmd+scroll = zoom (trackpad-native)
- Pose rendering: smaller dots, lower confidence threshold for hands (0.01)

## [0.1.0] - 2026-03-05

### Added
- 12-page Streamlit pipeline UI (Inventory, Pose Extraction, EAF Manager, Download, PreAnnotation, Subtitles, Training Data, Training, Evaluation, Inference, Assistant, AI Review)
- MediaPipe Holistic pose extraction with Metal GPU acceleration (~400 fps)
- Apple Vision backend for pose extraction (~67 fps)
- PoseTransformerEncoder model (500K params, 3-layer Transformer)
- Category transfer learning — two-phase fine-tuning (24.9% top-1 on 516 signs)
- Self-supervised masked pose pre-training (ssl_pretrain.py)
- Three landmark presets: compact (96), extended (148), full (174)
- Active learning orchestrator with milestone-based retraining
- EAF harvest — bulk import human annotations from ELAN files
- Multi-source video downloader (YouTube, partner-dictnary CSV, FTP, HTTP)
- AI Review page with synced video+pose player, trim handles, approve/correct workflow
- Interactive timeline component with zoom, pan, prepartner-dictn selection
- MCP server exposing 12 pipeline tools for Claude Code integration
- AI Assistant page for annotator chat (Anthropic API)
- SPJ glossary management with ID-glosses and Slovak word forms
- ELAN tier convention: S1_Gloss_RH/LH, AI_Gloss_RH/LH, AI_Confidence
- Training data export with NPZ segments and manifest
- Model evaluation with confusion matrices, per-class F1, model comparison
- Batch inference with prepartner-dictns written to EAF AI tiers
