# Changelog

## [0.2.0] - 2026-03-19

### Sign Collector (zber.spj.sk) — New
- Full web app for crowdsourcing sign recordings: register, record, validate, admin
- 68 vocabulary themes with 1,350+ signs seeded
- Camera recording with real-time quality checks (MediaPipe hand + face detection, brightness)
- Post-recording quality gate — analyzes 5 frames, requires face 4/5 + hands 3/5
- Auto-record mode — starts recording when hands + face detected for 1 sec
- Framing guide overlay for optimal sign recording position
- Researcher validation workflow — thumbs up/down voting (3 up = approve, 2 down = reject)
- Google OAuth login + email/password registration with email verification
- GDPR consent flow — 3 separate checkboxes (account, biometric video, research retention)
- Account deletion with video anonymization (GDPR data export endpoint)
- Admin panel: words, themes, videos, users, logs, stats, metrics tabs
- CSV bulk word import for admin
- Page view analytics with GDPR-safe IP hashing, device detection, bot filtering
- Admin "Metriky" tab — KPIs, hourly/daily charts, top pages, referrers, storage gauge, server health
- `/api/status.php` health check endpoint for uptime monitoring
- Variant recording support — when all signs done once, offers signs below target for additional variants
- Themes page: progress counter, completed themes sorted to bottom
- Offline upload fallback (LocalStorage queue for network interruptions)
- Dark mode + light mode with system preference detection
- Cookie consent banner
- PWA manifest
- Error logging system (PHP + JS errors → DB → admin panel)
- Security: CSRF protection, session hardening, CSP headers, XSS-safe output

### Sign Collector — Accessibility (WCAG 2.2 AA)
- Skip-to-content link, ARIA landmarks, keyboard navigation
- ARIA live regions for recording state, quality status, upload feedback
- Screen reader announcements for countdown, recording start/stop
- High contrast colors, minimum font sizes, reduced motion support
- Focus management on record button after camera reset
- Accessible forms with labels, required indicators, error messages

### Accessibility Audit Tool — New (`tools/a11y-audit/`)
- Static analysis tool for PHP/HTML/CSS/JS accessibility auditing
- 18 rule modules: contrast, color, typography, layout, structure, ARIA, focus, keyboard, interactive, forms, media, language, motion, cognitive, compliance, collector-specific
- PHP/HTML parser (BeautifulSoup + PHP block stripping)
- CSS parser (variables, keyframes, media queries)
- JS parser (event listeners, timers, ARIA manipulation)
- Terminal, JSON, and HTML dashboard reporters
- CLI entry point + Playwright browser bridge
- WCAG 2.2 AA configuration with customizable thresholds

### ML Pipeline — Conv1D Architecture
- `Conv1DTransformerEncoder` — Kaggle 1st-place architecture (Conv1D blocks + Transformer)
- 9x better than baseline in method comparison (12.9% vs 1.4% val acc, 20 epochs)
- Best combo: `conv1d_transformer` + `norm_xy_velocity` + mirror/rotation augmentation
- 4 feature modes: `raw`, `velocity`, `xy_velocity`, `norm_xy_velocity`
- Auto-detection of input_dim from NPZ files and checkpoint weights

### ML Pipeline — Data Augmentation
- 8 augmentation types: temporal crop, speed, noise, scale, mirror, rotation, joint dropout, temporal mask
- Per-augmentation enable/disable flags in `TrainingConfig`
- `tools/augment_search.py` — greedy hill-climbing over augmentation combos
- Result: mirror + rotation only = 38.9% val (best); all 8 = worse than baseline
- Augmentation controls in Page 8 (Training) with effective sample count

### ML Pipeline — Multi-Source Training
- 6 data sources: posunky (37K), kodifikácia (14K), spreadthesign (10K), dictio (9K), artsign (210), others
- Unified training splits: `splits_unified_3plus/` (8,005 classes), `splits_unified_2plus/` (14,296 classes)
- `tools/train_optimal.py` — best training recipe (Conv1D + mirror/rotation + all sources)
- Training resume support for interrupted runs

### ML Pipeline — Transfer Learning & Experiments
- Category → word transfer learning (two-phase: freeze encoder → unfreeze all)
- Conv1D transfer from category model to word model
- SSL pre-training experiment — marginal (+6% relative), supervised transfer far better (+48%)
- Extended landmarks (148) experiment — negative result, compact (96) is better
- 5+ samples/class experiment — negative vs 3+ baseline
- Augmentation search v2 — spatial augments saturated, real signer diversity needed

### AI Review Improvements
- RH+LH unit grouping — auto-pair overlapping hand predictions as one review unit
- Sign type dropdown: Sign (default), Classifier, Compound
- Compound mode — link sequential units as one meaning
- Multi-highlight timeline — paired predictions highlighted simultaneously
- Video-editor-style razor cut tool (C key), undo cut (X key)
- Alt+drag to create new label regions on timeline
- Timeline: subtitle track display, tooltips, right-click to delete cuts
- Pose similarity search for annotation support

### Pipeline — Other
- Dual-view video splitting for kodifikácia (60° + 90° camera angles)
- `crop` parameter on `extract_pose()` for in-memory frame cropping
- Interactive timeline component with zoom (Ctrl+scroll), pan (scroll), selection
- Streamlit postMessage protocol shim for timeline embedding

### Changed
- AI Review: prediction selectbox → unit selectbox (RH+LH / RH / LH)
- Timeline: `active_pred_indices` for multi-prediction highlighting
- Scroll behavior: regular scroll = pan, Ctrl/Cmd+scroll = zoom
- Pose rendering: smaller dots, lower hand confidence threshold (0.01)
- Quality checker stops during review, resumes on camera return (mobile battery fix)
- Registration "Škola pre nepočujúcich" now optional (for CODA, mainstream deaf)

### Fixed
- Reflected XSS in admin logs filters
- Safari video playback — HTTP Range requests, play/pause race condition
- Safari video autoplay on admin grid
- Dark mode contrast for terms page, consent cards
- Quality status badges flickering after recording (rAF loop not stopped)
- Themes page: PHP code rendered as text (broken `<?php` block after separator)

### Security
- CSRF on all state-changing endpoints
- Session regeneration on login
- `Strict-Transport-Security`, `X-Content-Type-Options`, `X-Frame-Options` headers
- `Content-Security-Policy` with MediaPipe/Google OAuth allowlist
- Password hashing with `PASSWORD_BCRYPT`
- Rate limiting on login and password reset (session-based)
- XSS fix in admin log filters (reflected via GET params)

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
- Multi-source video downloader (YouTube, CSV, FTP, HTTP)
- AI Review page with synced video+pose player, trim handles, approve/correct workflow
- Interactive timeline component with zoom, pan, prediction selection
- MCP server exposing 12 pipeline tools for Claude Code integration
- AI Assistant page for annotator chat (Anthropic API)
- SPJ glossary management with ID-glosses and Slovak word forms
- ELAN tier convention: S1_Gloss_RH/LH, AI_Gloss_RH/LH, AI_Confidence
- Training data export with NPZ segments and manifest
- Model evaluation with confusion matrices, per-class F1, model comparison
- Batch inference with predictions written to EAF AI tiers
