# SPJ-Korpus

**The first machine-learning-ready corpus toolkit for Slovak Sign Language (Slovensky posunkovy jazyk)**

Slovak Sign Language has ~5,000 native signers and ~20,000 users in Slovakia. No annotated corpus exists. No ML model exists. No big tech company will build recognition for a language this small. SPJ-Korpus fills every critical gap.

---

## What is this?

SPJ-Korpus is an end-to-end AI-assisted pipeline for building a sign language corpus — from raw video to trained recognition models. It combines:

- **Pose extraction** — MediaPipe Holistic (543 landmarks) with Metal GPU acceleration (~400 fps)
- **AI pre-annotation** — automatic sign boundary detection and gloss suggestion
- **Active learning loop** — AI suggests → deaf annotators correct in ELAN → corrections retrain the model → better suggestions
- **Training pipeline** — PoseTransformerEncoder with SignBERT/OpenHands transfer learning
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

### 11 Streamlit Pages

| # | Page | Purpose |
|---|------|---------|
| 1 | Inventory | Catalog videos, track extraction status |
| 2 | Pose Extraction | MediaPipe batch processing (Metal GPU / CPU / Apple Vision) |
| 3 | EAF Manager | Create and manage ELAN annotation files |
| 4 | Download | Download videos from URLs/playlists with subtitles |
| 5 | PreAnnotation | AI-generated sign boundaries from pose data |
| 6 | Subtitles | Extract and manage subtitle files |
| 7 | Training Data | Align pose segments to subtitles, export .npz training data |
| 8 | Training | Train PoseTransformerEncoder models |
| 9 | Evaluation | Evaluate models with confusion matrices and per-class F1 |
| 10 | Inference | Run prepartner-dictns on new videos, write to ELAN AI tiers |
| 11 | Assistant | AI chat for annotators (requires Anthropic API key) |

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
| `intl-vocab.py` | reference partner-dictnary partner-dictnary download |
| `downloader.py` | Video download with subtitle extraction |

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
  → Linear projection → d_model (256)
  → Sinusoidal positional encoding
  → 4x TransformerEncoderLayer (4 heads, d_ff=512)
  → Masked mean pooling
  → Linear → n_classes
```

### Landmark Presets

| Preset | Body | Hands | Face | Total | input_dim |
|--------|------|-------|------|-------|-----------|
| **compact** (default) | 7 | 42 | 47 (lips+nose) | 96 | 288 |
| **extended** | 7 | 42 | 99 (+eyes+eyebrows) | 148 | 444 |
| **full** | 33 | 42 | 99 | 174 | 522 |

### Retraining Milestones

| Signs annotated | Action | Expected accuracy |
|----------------|--------|-------------------|
| 500 | Fine-tune backbone on SPJ bootstrap | ~10-15% top-3 |
| 2,000 | v1 — first SPJ-specific model | ~50-60% |
| 5,000 | v2 — active learning begins | ~70-75% |
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

If the Slovak Sign Language pipeline succeeds, the toolkit is designed to expand to other sign languages — especially small/minority sign languages in Europe that are similarly underserved by technology. The architecture is language-agnostic; only the training data and annotation conventions are SPJ-specific.

## Vision

SPJ-Korpus is phase 1 of a 3-phase project:

1. **SPJ-Korpus** (this project) — build the annotated corpus and sign recognition model
2. **Training app** — free Slovak Sign Language training tool for the deaf community
3. **AI interpretation** — when human interpreters are unavailable (~30 for all of Slovakia), AI-powered sign language interpretation and real-time subtitles
4. **Expand to more languages** — adapt the pipeline for other minority sign languages across Europe

---

## License

This project is licensed under [CC BY-NC-SA 4.0](LICENSE) — Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International.

You are free to share and adapt this work for non-commercial purposes, with attribution, under the same license.

---

## Author

**Marek Kanas** — deaf developer, Slovakia
[partner-ngo](https://www.partner-ngo.eu)

Built with [Claude Code](https://claude.ai/claude-code) by Anthropic.
