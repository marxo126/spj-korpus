# SPJ-Korpus — Claude Instructions

All detailed plans, phases, and rationale are in `PLAN.md`. This file captures only the
facts and rules most likely to cause mistakes if forgotten.

---

## Tech Stack

**Language:** Python 3.13 (managed by `uv`, venv at `.venv/`)
**Package manager:** `uv` (lockfile-based, fast)
**Run commands with:** `.venv/bin/python` (NOT `python` or `python3` — system Python is 3.14, project uses 3.13)

### Core dependencies

| Package | Purpose |
|---|---|
| `mediapipe` | Pose + face landmark extraction (Holistic) |
| `sign-language-processing` | `.pose` file format, pipeline utilities |
| `torch` | ML training and inference |
| `transformers` | Hugging Face models (AV-HuBERT, SignBERT, SignCLIP) |
| `datasets` | Hugging Face dataset loading/caching |
| `pympi-ling` | Read and write ELAN `.eaf` files — use this, not raw XML |
| `lxml` | XML fallback if pympi is insufficient |
| `ffmpeg-python` | Video decoding and frame extraction |
| `pandas` | Corpus metadata, annotation tables |
| `numpy` | Keypoint arrays and numerical ops |

### EAF files
Use **`pympi`** (`pympi-ling` on PyPI) for all ELAN EAF read/write — it is the standard in SL corpus research and understands the EAF schema natively. Do not hand-roll XML parsing for EAF files.

### No TypeScript
TypeScript is not used. A web corpus browser (Phase 5, optional) would be the only exception — treat it as a separate project if it ever happens.

---

## Language Scope

**SPJ only.** ISO 639-3 code: `svk`.

- Never include Czech SL (ČZJ), ASL, DGS, or any other SL data in the corpus itself.
- Other sign languages appear ONLY as transfer learning sources (backbone models).
- Do NOT mix other SL corpora into SPJ training sets.

---

## Data Licensing — What You Can and Cannot Do

| Source | ML Training | Notes |
|---|---|---|
| Own SPJ videos (~300) + Museum (100h+) | ✅ Yes | Fully owned — primary source |


---

## Models — What Exists vs. What Does Not

### Does NOT exist (never assume or try to use):
- **Czech Sign Language (ČZJ) pre-trained model** — no public checkpoint exists anywhere
- **SHuBERT public checkpoint** — paper is public but no checkpoint released as of 2025 (verify at shubert.pals.ttic.edu before use)
- **SignLLM model** — dataset is CC-BY-NC-4.0 but the model itself is NOT open-sourced
- **Pose → SignWriting tool** — no off-the-shelf tool converts pose to SignWriting; training a new model would be required

### Correct baselines to use instead:
- **SignBERT** (Hu et al., ICCV 2021) — MIT license, public checkpoint, hand-model-aware pretraining — primary transfer learning backbone
- **OpenHands ASL** (AI4Bharat) — Apache 2.0 — alternative backbone
- **SignCLIP** — multilingual embedding space (44 SLs) — frozen feature extractor option
- **AV-HuBERT** (Meta) — mouthing detection; public checkpoint on Hugging Face; verify checkpoint license before distributing derived models

---

## ELAN Tier Names — Exact Spelling, Always

Human annotation tiers:
```
S1_Gloss_RH        right-hand glosses
S1_Gloss_LH        left-hand glosses
S1_Translation     Slovak translation per utterance
S1_Mouthing        mouthed spoken words (see convention below)
S1_Mouth_Gesture   non-spoken mouth patterns (puffed cheeks, etc.)
S1_NonManual       other non-manual signals (eyebrows, head movement)
```

AI suggestion tiers (pre-populated by pipeline, reviewed by annotator):
```
AI_Gloss_RH        AI-suggested right-hand glosses with timestamps
AI_Gloss_LH        AI-suggested left-hand glosses with timestamps
AI_Confidence      confidence score per segment, float 0.0–1.0
```

Never invent variant names (e.g. `Gloss_R`, `RH_Gloss`, `AI_Confidence_Score`).

---

## Naming Conventions

**ID-glosses:** `UPPERCASE-NUMBER` — e.g. `WATER-1`, `HOUSE-3`

**Mouthings** (`S1_Mouthing`): lowercase Slovak — e.g. `voda` for the WATER sign, `dom` for HOUSE

**Mouth gestures** (`S1_Mouth_Gesture`): descriptive label — e.g. `puffed_cheeks`, `pursed_lips`

---

## SignWriting Path

The correct path is:
1. Annotate signs with ID-glosses (active learning pipeline)
2. Trained annotators write **original** SPJ SignWriting entries for each gloss
3. Use **partner-dict** (partner-university) and **SignPuddle** as notation learning guides only — they are references, not sources to copy

**partner-dict and SignPuddle are notation references like a musical score — you study them to
learn the system, then compose original SPJ entries from scratch.**

Output: SPJ SignWriting partner-dictnary built as a by-product of Phase 3 annotation. No ML
training step required or used for this.

---

## File Formats

- `.pose` — binary pose keypoint format (sign-language-processing library)
- `.eaf` — ELAN annotation format (XML, CLARIN standardized)

---

## Annotation Rules

**Priority order** (annotate in this order, do not skip ahead):
1. ID-glosses — `S1_Gloss_RH` + `S1_Gloss_LH`
2. Slovak translations per utterance — `S1_Translation`
3. Mouthings — `S1_Mouthing`
4. Non-manual signals — `S1_NonManual` + `S1_Mouth_Gesture`
5. HamNoSys (optional, selected signs only)
6. SignWriting (optional, see path above)

**No custom annotation UI.** All annotation happens in standard ELAN. The pipeline outputs
pre-populated `.eaf` files; annotators review/correct AI tiers in ELAN.

---

## Retraining — Trigger by Sign Count, NOT by Time

| Pool size | Action |
|---|---|
| 500 signs | Fine-tune SignBERT/OpenHands backbone on SPJ bootstrap data |
| 2,000 signs | v1 retrain — first SPJ-specific model |
| 5,000 signs | v2 retrain — begin active learning |
| 10,000+ signs | v3 full retrain + held-out test set evaluation |

Expected bootstrap accuracy: **~10–15% top-3** (not 20% — that was the old ČZJ estimate).

---

## Tool Licensing Constraints

- **OpenPose** — CMU academic/non-commercial only. Use **MediaPipe** for any pipeline that
  could have commercial application.
- **MMS-Player (DFKI)** — GPL-3.0 (copyleft — derived work must also be GPL).

---

## Output License

**CC BY-NC-SA 4.0** — required for compatibility with Corpus NGT (ShareAlike clause).

---

## Before Corpus Deposit or Publication

Confirm all three:
2. Written data-sharing agreement signed with **partner-ngo** (same scope)

---

## Pipeline Architecture (Pages 1–10)

| # | Page | Input | Output |
|---|------|-------|--------|
| 1 | Inventory | Video directory | `inventory.csv` |
| 2 | Pose Extraction | Inventory + videos | `.pose` files |
| 3 | EAF Manager | Inventory | `.eaf` files |
| 4 | Download | URL/playlist | Videos + subtitles |
| 5 | PreAnnotation | `.pose` files | AI EAF annotations (kinematic) |
| 6 | Subtitles | Videos | `.vtt` files |
| 7 | Training Data | `.pose` + `.vtt` | `alignment.csv` + `.npz` + `manifest.csv` |
| 8 | Training | `manifest.csv` + `.npz` | `.pt` checkpoints in `data/models/` |
| 9 | Evaluation | Checkpoint + test split | JSON/CSV reports in `data/evaluations/` |
| 10 | Inference | Checkpoint + videos | Predicted glosses written to EAF AI tiers |

**Active learning loop:** Page 10 → ELAN review → Page 7 re-export → Page 8 retrain.

---

## ML Pipeline Rules

### Pose Data Shapes — Critical
```
.pose file loaded:  (T, 1, 543, 3) data + (T, 1, 543, 1) confidence  ← has person dimension
.npz exported:      (T, 543, 3) pose + (T, 543, 1) confidence         ← NO person dimension
```
**Always check which shape you're working with.** Use `data[t, 0, :, :]` to drop the person dim from `.pose`, but `pose[t, :, :]` for `.npz`.

### M4 Max / MPS Training Rules
1. Use `threading.Thread` for background training — **NEVER** `multiprocessing.Process` (MPS tensors can't cross processes)
2. `pin_memory=False`, `num_workers=0` in DataLoader (unified memory — no separate GPU VRAM)
3. `torch.autocast("mps", dtype=torch.float16)` — wrap in try/except, fallback to fp32 (some ops unsupported)
4. `TrainingState` must use **simple Python types** only (int, float, list, str, bool) — GIL-safe reads from Streamlit thread

### Key Backend Modules

| Module | Public API |
|--------|-----------|
| `src/spj/trainer.py` | `LabelEncoder`, `PoseSegmentDataset`, `split_dataset()`, `PoseTransformerEncoder`, `TrainingConfig`, `TrainingState`, `train_model()`, `load_checkpoint()`, `list_checkpoints()` |
| `src/spj/evaluator.py` | `evaluate_model()`, `save_evaluation_report()`, `confusion_matrix_figure()`, `per_class_f1_figure()`, `compare_models_table()` |
| `src/spj/inference.py` | `predict_segments()`, `write_prepartner-dictns_to_eaf()`, `prepartner-dictns_timeline_figure()` |
| `src/spj/training_data.py` | `align_pose_to_subtitles()`, `build_alignment_table()`, `export_segment_npz()`, `write_training_config()` |
| `src/spj/preannotate.py` | `load_pose_arrays()`, `detect_sign_segments()`, `preannotate_eaf()` |
| `src/spj/eaf.py` | `load_eaf()`, `save_eaf()`, `create_empty_eaf()`, `add_ai_annotation()`, tier constants |
| `src/spj/pose.py` | `extract_pose()`, `extract_pose_batch()`, `ensure_models()` |
| `src/spj/glossary.py` | `Glossary`, `load_glossary()`, `save_glossary()`, `normalize_word()`, `tokenize_slovak()` |

### Model Architecture: PoseTransformerEncoder
```
Input: (batch, max_seq_len, 1629)     # 543 landmarks × 3 coords
  → Linear → d_model (256)
  → Sinusoidal positional encoding
  → 4× TransformerEncoderLayer (4 heads, d_ff=512)
  → Masked mean pooling
  → Linear → n_classes
```
Variable-length sequences padded to `max_seq_len`, attention mask ignores padding.

### Checkpoint Format (.pt)
```python
{
    "model_state_dict": ...,
    "label_encoder": {"label_to_idx": {"GLOSS": 0, ...}},
    "config": {TrainingConfig fields},
    "timestamp": "YYYY-MM-DD HH:MM:SS",
    "epoch": int,
    "val_acc": float,
    "n_classes": int,
    "n_train": int,
}
```

### Data Directories
```
data/training/splits/      # train.csv, val.csv, test.csv
data/training/export/       # .npz segments + manifest.csv
data/models/                # .pt checkpoints
data/evaluations/           # JSON + CSV evaluation reports
```

---

## Streamlit Page Patterns

### Page File Naming
`N_PageName.py` — Streamlit auto-sorts by numeric prefix. Always use this pattern.

### sys.path Pattern (every page starts with this)
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
```

### DATA_DIR Pattern
```python
DATA_DIR = Path(__file__).parent.parent.parent / "data"
```

### Background Thread Pattern (for long-running tasks)
- Use `threading.Thread(target=..., daemon=True)` — NOT ProcessPoolExecutor for ML
- Store `TrainingState` in `st.session_state` — UI polls simple Python fields
- Poll with `time.sleep(1.0)` + `st.rerun()` while `state.running`
- Thread sets `state.finished = True` when done

### ProcessPoolExecutor Pattern (for multi-video extraction)
- Executor stored in `st.session_state` (NOT in `with` block) so it survives reruns
- Progress polled via `wait(pending_futs, timeout=0.4, return_when=FIRST_COMPLETED)`
- Background mode: `time.sleep(2.0)` + `st.rerun()`; foreground: `time.sleep(0.05)`

### Streamlit Deprecation Warnings
- **NEVER** use `width="stretch"` — removed in modern Streamlit, use `use_container_width=True`
- Use `st.session_state` dict updates, not direct attribute assignment for complex state

---

## Mistakes & Lessons Learned (ML Pipeline)

### 1. System Python vs Project Python
**Mistake:** Using `python` or `python3` to run commands — system Python is 3.14, project uses 3.13 via uv.
**Rule:** Always use `.venv/bin/python` for any project commands.

### 2. Pose Data Person Dimension
**Mistake:** Assuming `.pose` and `.npz` have the same shape. `.pose` has shape `(T, 1, 543, 3)` (person dim at axis 1), while exported `.npz` training segments have `(T, 543, 3)` (person dim already removed).
**Rule:** Always verify which format you're loading. `load_pose_arrays()` returns the 4D shape with person dim. NPZ training segments have it removed.

### 3. MPS Autocast Fallback
**Mistake:** Assuming all torch ops work under `torch.autocast("mps")`.
**Rule:** Always wrap autocast in try/except and fall back to fp32. Some operations (e.g., certain attention variants) may not support float16 on MPS.

### 4. Training Must Use Threads, Not Processes
**Mistake:** Could have used `multiprocessing.Process` (like pose extraction does).
**Rule:** MPS tensors are bound to the creating process's Metal context. Training MUST use `threading.Thread`. Pose extraction can use ProcessPoolExecutor because each worker creates its own context.

### 5. Manifest Label Column
**Mistake:** `manifest.csv` from page 7 export may not have a `label` column — it uses `reviewed_text` and `text` separately.
**Rule:** Always derive the label column: `reviewed_text.where(reviewed_text.strip() != "", text)`. Do this defensively in every module that reads the manifest.

### 6. pose_format Confidence Shape is 3D, Not 4D
**Mistake:** `pose.body.confidence` returns `(T, 1, 543)` — 3D. Code assumed `(T, 1, 543, 1)` — 4D with trailing dimension. Downstream indexing `conf[:, 0, idx, 0]` crashed with "too many indices for array".
**Rule:** `load_pose_arrays()` now normalizes: `if conf.ndim == 3: conf = conf[..., np.newaxis]`. Never assume the trailing dimension exists — always go through `load_pose_arrays()`.

### 7. Empty .pose Files from Failed Extraction
**Mistake:** Pose extraction can fail silently, leaving 0-byte `.pose` files. These look like "pose extracted" in the inventory but crash on load.
**Rule:** `load_pose_arrays()` now checks `st_size == 0` and gives a clear error. The inventory page should also flag 0-byte files. Before training, verify files are non-empty.

### 8. Pandas NaN → String Conversion
**Mistake:** Using `str(value)` on a pandas `NaN` produces the string `"nan"`, which is truthy and non-empty — silently corrupts text fields.
**Rule:** Always use `pd.isna(value)` before converting to string: `"" if pd.isna(val) else str(val).strip()`.

### 9. Segment Status Constants
**Mistake:** Bare status strings (`"pending"`, `"approved"`, etc.) scattered across the Review tab — typo-prone, no single source of truth.
**Rule:** Use `ST_PENDING`, `ST_APPROVED`, `ST_SKIPPED`, `ST_FLAGGED` constants defined at top of `7_Training_Data.py`.

### 10. Glossary `get_entry()` Returns a Copy
**Mistake:** Returning the internal dict by reference from `get_entry()` allows callers to silently corrupt the reverse index by mutating forms.
**Rule:** `get_entry()` returns a shallow copy. Use `add_form()` / `add_gloss()` for mutations.
