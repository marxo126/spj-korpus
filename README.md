# SPJ-Korpus

**The first machine-learning-ready corpus toolkit for Slovak Sign Language (Slovensky posunkovy jazyk)**

Slovak Sign Language has ~5,000 native signers and ~20,000 users in Slovakia. No annotated corpus exists. No ML model exists. No big tech company will build recognition for a language this small. SPJ-Korpus fills every critical gap.

---

## What is this?

SPJ-Korpus is an end-to-end AI-assisted pipeline for building a sign language corpus — from raw video to trained recognition models. It combines:

- **Pose extraction** — MediaPipe Holistic (543 landmarks) with Metal GPU acceleration (~400 fps)
- **AI pre-annotation** — automatic sign boundary detection and gloss suggestion
- **Active learning loop** — AI suggests → annotators review in-app or in ELAN → corrections retrain the model → better suggestions
- **Training pipeline** — PoseTransformerEncoder with category transfer learning (24.9% top-1 on 516 signs)
- **Evaluation & inference** — model comparison, per-class metrics, prepartner-dictn output to ELAN
- **MCP server** — 12 pipeline tools for Claude Code integration

Built by a deaf developer using Claude Code.

---

## Pipeline Architecture

```
Video → MediaPipe Pose → .pose files → EAF Pre-annotation
                                            ↓
                                    Annotator review (ELAN)
                                            ↓
                                    Corrected annotations
                                            ↓
                                    Training data export (.npz)
                                            ↓
                                    Model training (PyTorch)
                                            ↓
                                    Better AI suggestions → faster annotation
```

### 12 Streamlit Pages

| # | Page | Purpose |
|---|------|---------|
| 1 | Inventory | Catalog videos, track extraction status |
| 2 | Pose Extraction | MediaPipe batch processing (Metal GPU / CPU / Apple Vision) |
| 3 | EAF Manager | Create and manage ELAN annotation files |
| 4 | Download | Download videos from URLs/playlists with subtitles |
| 5 | PreAnnotation | AI-generated sign boundaries from pose data |
| 6 | Subtitles | Extract and manage subtitle files |
| 7 | Training Data | Align pose segments to subtitles, export .npz training data, EAF harvest |
| 8 | Training | Train PoseTransformerEncoder models |
| 9 | Evaluation | Evaluate models with confusion matrices and per-class F1 |
| 10 | Inference | Run prepartner-dictns on new videos, write to ELAN AI tiers |
| 11 | Assistant | AI chat for annotators (requires Anthropic API key) |
| 12 | AI Review | Review AI prepartner-dictns with synced video+pose player, trim boundaries, approve/correct |

### Backend Modules

| Module | Purpose |
|--------|---------|
| `pose.py` | MediaPipe + Apple Vision pose extraction |
| `eaf.py` | ELAN EAF file read/write (via pympi) |
| `preannotate.py` | Kinematic sign boundary detection |
| `training_data.py` | Pose-subtitle alignment, NPZ export, landmark presets |
| `trainer.py` | PoseTransformerEncoder, training loop, checkpoints |
| `evaluator.py` | Model evaluation, confusion matrices, comparison |
| `inference.py` | Run prepartner-dictns, write to EAF AI tiers |
| `glossary.py` | SPJ glossary management with ID-glosses |
| `orchestrator.py` | Active learning orchestrator (milestone-based retraining) |
| `mcp_server.py` | MCP server exposing 12 pipeline tools |
| `ssl_pretrain.py` | Self-supervised masked pose pre-training |
| `clustering.py` | Sign clustering for exploratory analysis |
| `downloader.py` | Multi-source video download (YouTube, partner-dictnary CSV, FTP, HTTP) |

---

## Tech Stack

- **Python 3.13** (managed by `uv`)
- **PyTorch** — model training and inference
- **MediaPipe** — pose landmark extraction (Metal GPU accelerated)
- **Streamlit** — interactive pipeline UI
- **pympi-ling** — ELAN EAF file handling
- **sign-language-processing** — .pose file format
- **Hugging Face** — transformers, datasets

---

## Model Architecture

```
Input: (batch, max_seq_len, input_dim)    # 288 (compact) / 444 (extended) / 522 (full)
  → Linear projection → d_model (128)
  → Sinusoidal positional encoding
  → 3x TransformerEncoderLayer (4 heads, d_ff=256)
  → Masked mean pooling
  → Linear → n_classes
```

500K parameters. Compact model chosen over larger alternatives — larger models (2.2M params, d_model=256, 4 layers) consistently overfit on our few-shot data.

### Landmark Presets

| Preset | Body | Hands | Face | Total | input_dim |
|--------|------|-------|------|-------|-----------|
| **compact** (default) | 7 | 42 | 47 (lips+nose) | 96 | 288 |
| **extended** | 7 | 42 | 99 (+eyes+eyebrows) | 148 | 444 |
| **full** | 33 | 42 | 99 | 174 | 522 |

### Current Results (March 2026)

Training data: 13,638 NPZ segments from partner-dictnary videos (10K) + category vocabulary (4.7K).

**Word-level sign recognition (516 labels, 3+ samples each):**

| Model | Approach | Test Top-1 | Test Top-3 | Test Top-5 |
|-------|----------|------------|------------|------------|
| Baseline | From scratch | 16.8% | 25.4% | 27.6% |
| **Category Transfer** | **Category encoder → word fine-tune** | **24.9%** | **31.9%** | **35.1%** |
| SSL Transfer | Masked pose pre-training → fine-tune | 17.8% | 25.4% | 28.6% |
| Random | 1/516 | 0.2% | 0.6% | 1.0% |

**Key finding:** Supervised category transfer (+48% relative improvement over baseline) dramatically outperforms self-supervised pre-training (+6%) in this data regime. The category model's encoder already understands SPJ-specific motion patterns — this domain knowledge transfers far more effectively than generic temporal dynamics learned from unsupervised masking.

**Category-level classification (102 categories):** 44.9% val accuracy (45x random baseline).

**Expanded dataset (1,743 labels, 2+ samples):** 19.5% test top-1, 28.1% top-3 — covers 3.4x more signs at modest per-class accuracy cost.

### Transfer Learning Strategy

The best-performing approach uses two-phase fine-tuning from the category model:

1. **Phase 1** (epochs 1-10): Freeze encoder, train only the new classifier head (lr=0.001)
2. **Phase 2** (epochs 11-100): Unfreeze all parameters, lower learning rate (lr=0.0003) with cosine schedule

This transfers 27 of 29 encoder parameters from the category model (2 transformer layers match exactly; the 3rd layer initializes randomly; classifier is replaced entirely).

### Retraining Milestones

| Signs annotated | Action | Observed/Expected accuracy |
|----------------|--------|-------------------|
| 500 | Fine-tune on SPJ bootstrap data | **25-35% top-3** (observed) |
| 2,000 | v1 retrain — first SPJ-specific model | ~50-60% |
| 5,000 | v2 retrain — active learning begins | ~70-75% |
| 10,000+ | v3 — full evaluation | ~85-90% |

---

## Quick Start

```bash
# Clone
git clone https://github.com/marxo126/spj-korpus.git
cd spj-korpus

# Install dependencies (requires uv)
uv sync

# Create data directories
mkdir -p data/{videos,pose,annotations,subtitles,training,models,evaluations}

# Run the pipeline UI
.venv/bin/streamlit run app/main.py
```

### MCP Server (for Claude Code)

The pipeline is exposed as MCP tools. Add to your `.mcp.json`:

```json
{
  "mcpServers": {
    "spj-pipeline": {
      "command": ".venv/bin/python",
      "args": ["src/spj/mcp_server.py"]
    }
  }
}
```

---

## Data

**Video data is not included in this repository.** The corpus videos belong to partner organizations and are used under private agreements for ML training. The repository contains only the tools and pipeline code.

See [`data/README.md`](data/README.md) for details on data access.

---

## ELAN Tier Convention

Human annotation tiers (following DGS-Korpus conventions):

| Tier | Content |
|------|---------|
| `S1_Gloss_RH` | Right-hand glosses (ID-gloss format: `WATER-1`) |
| `S1_Gloss_LH` | Left-hand glosses |
| `S1_Translation` | Slovak translation per utterance |
| `S1_Mouthing` | Mouthed spoken words (lowercase Slovak: `voda`) |
| `S1_Mouth_Gesture` | Non-spoken mouth patterns |
| `S1_NonManual` | Other non-manual signals |

AI suggestion tiers (pre-populated by pipeline):

| Tier | Content |
|------|---------|
| `AI_Gloss_RH` | AI-suggested right-hand glosses |
| `AI_Gloss_LH` | AI-suggested left-hand glosses |
| `AI_Confidence` | Confidence score per segment (0.0-1.0) |

---

## Language Scope

**SPJ only.** ISO 639-3 code: `svk`.

Other sign languages appear only as transfer learning sources (backbone models). No other SL corpus data is mixed into SPJ training sets.

---

## Status

**Actively developing.** SPJ-Korpus is under active development. The pipeline is functional and processing real data.

Current state (March 2026):
- **15,000+ videos** ingested from multiple partner sources
- **13,638 training segments** exported as NPZ with compact landmarks
- **6 model checkpoints** trained — best achieves 24.9% top-1 on 516 signs (118x random)
- **Category→word transfer learning** validated as the most effective approach
- Active learning loop ready for deployment with deaf annotators

If the Slovak Sign Language pipeline succeeds, the toolkit is designed to expand to other sign languages — especially small/minority sign languages in Europe that are similarly underserved by technology. The architecture is language-agnostic; only the training data and annotation conventions are SPJ-specific.

## Vision

SPJ-Korpus is phase 1 of a 3-phase project:

1. **SPJ-Korpus** (this project) — build the annotated corpus and sign recognition model
2. **Training app** — free Slovak Sign Language training tool for the deaf community
3. **AI interpretation** — when human interpreters are unavailable (~30 for all of Slovakia), AI-powered sign language interpretation and real-time subtitles
4. **Expand to more languages** — adapt the pipeline for other minority sign languages across Europe

---

## Related Research

SPJ-Korpus builds on established research in pose-based sign language recognition:

- **[Google Isolated Sign Language Recognition](https://www.kaggle.com/competitions/asl-signs)** (Kaggle 2023) — Same approach: MediaPipe 543 landmarks → selected subset → Transformer encoder for isolated sign classification. Winning solutions validated that ~130 selected landmarks + 1D conv/Transformer architectures work well for this task.
- **[SignBERT](https://github.com/hubuwei/SignBERT)** (Hu et al., ICCV 2021) — Hand-model-aware self-supervised pretraining. Used as transfer learning backbone.
- **[OpenHands](https://github.com/AI4Bharat/OpenHands)** (AI4Bharat) — Apache 2.0 sign language recognition toolkit. Alternative backbone for transfer learning.
- **[SignCLIP](https://github.com/J22Melody/signclip)** — Multilingual sign language embedding space (44 SLs). Frozen feature extractor option.
- **[Corpus NGT](https://www.ru.nl/corpusngtuk/)** — Dutch Sign Language corpus. Annotation conventions and tier structure adapted for SPJ.
- **[DGS-Korpus](https://www.sign-lang.uni-hamburg.de/dgs-korpus/)** — German Sign Language corpus. ID-gloss methodology reference.

---

## License

Non-Commercial Open Source — free to use, modify, and distribute for non-commercial purposes. See [LICENSE](LICENSE) for details.

---

## Author

**Marek Kanas** — deaf developer, Slovakia

Built with [Claude Code](https://claude.ai/claude-code) by Anthropic.
