# SPJ-Korpus Project Plan
## Slovak Sign Language Corpus & AI Translation System

**Goal:** Build the first machine-learning-ready corpus of Slovak Sign Language (SPJ), enabling AI-powered translation between written/spoken Slovak and signed video (avatar + real person captioning).

---

## Current State of SPJ

- **Full name:** Slovenský posunkový jazyk (SPJ)
- **Legal status:** Recognized by law since 1995 (Law 149/1995); codified 2020
- **Family:** French Sign Language family (related to Czech SL, Hungarian SL, Polish SL)
- **Community:** ~4,000 deaf users; ~30 professional interpreters
- **ISO 639-3:** svk

### Existing SPJ Resources

| Resource | Type | License | Notes |
|---|---|---|---|
| Own SPJ videos (~300) | Parallel corpus | ✅ Fully owned | SPJ video + Slovak subtitles — primary training source |
| Museum content (100h+) | Parallel corpus | ✅ Fully owned | Slovak exhibit text ↔ SPJ interpretation — largest owned dataset |
| Partner organizations | Video + subtitles | ✅ With agreements | SPJ videos with subtitles — written agreements required |
| SIGN-HUB (svk) | Linguistic docs | ⚠️ Mixed open/restricted | Grammar, ATLAS, life stories (EU H2020); apply for restricted content |
| Trnava University | Research | N/A | First SPJ study program, graduates since 2024 |

## Data Rights & Licensing

All training data requires written agreements from data owners before use. Own content (videos, museum) is fully owned. Partner data requires formal data-sharing agreements (scope: ML training, derivatives, publication).

### Reference Corpora (public, for transfer learning)

| Source | ML Training | Derivative Dataset | Publish Results |
|---|---|---|---|
| PHOENIX14T | ✅ Non-commercial | ✅ Non-commercial | ✅ |
| How2Sign | ✅ CC BY-NC 4.0 | ✅ CC BY-NC | ✅ |
| Corpus NGT | ✅ CC BY-NC-SA | ✅ (ShareAlike applies) | ✅ |
| BSL Corpus | ✅ With registration | ❌ | ✅ |
| DGS-Korpus | ✅ Likely (verify with IDGS Hamburg) | ⚠️ Verify | ✅ |

### Tool Licenses

| Tool | License | Commercial OK? |
|---|---|---|
| ELAN | GPL-3.0 | ✅ Yes |
| MediaPipe | Apache 2.0 | ✅ Yes |
| OpenPose | CMU academic/non-commercial | ❌ Academic only |
| sign-language-processing | MIT / Apache 2.0 | ✅ Yes |
| ANNIS | Apache 2.0 | ✅ Yes |
| MMS-Player (DFKI) | GPL-3.0 | ✅ Yes (copyleft) |

### Recommended Output License
**CC BY-NC-SA 4.0** — compatible with input corpus licenses (satisfies Corpus NGT ShareAlike requirement), allows research reuse, prevents commercial exploitation without permission.

---

### Critical Gaps
- No annotated corpus for ML/NLP
- No isolated sign recognition dataset
- No gloss-annotated continuous corpus
- No avatar/production system for SPJ
- No pose-keypoint dataset
- No published AI/NLP papers targeting SPJ

---

## Reference: Major Sign Language Corpora Worldwide

| Corpus | Language | Size | Signers | Annotation | Access |
|---|---|---|---|---|---|
| DGS-Korpus | German SL | 560h | 330 | ELAN+iLex (glosses, translations) | Research (ANNIS) |
| BSL Corpus | British SL | ~100h | UK-wide | ELAN (glosses, translations) | Open access |
| Corpus NGT | Dutch SL | 72h | 92 | ELAN + NGT SignBank | CC BY-NC-SA |
| Auslan Corpus | Australian SL | 300h+ | 100+ | ELAN + Auslan SignBank | ELAR (research) |
| PHOENIX14T | German SL | ~8K clips | 9 | Glosses + German text | Public |
| How2Sign | ASL | 80h | 11 | English text | Public |
| [public SL dataset] | 25+ languages | 3,207h | — | Subtitles | Public |

**Best model to follow:** DGS-Korpus (University of Hamburg) — most detailed methodology, publicly documented annotation conventions.

---

## Tools & Standards

### Annotation
- **ELAN** — standard annotation tool for all major SL corpora (free, open source)
  - Data format: EAF (XML-based, CLARIN standardized)
  - Tier structure: `S1_Gloss_RH`, `S1_Gloss_LH`, `S1_Translation`, `S1_Mouthing`, `S1_NonManual`
- **iLex** — University of Hamburg; corpus + lexicographic DB; integrates with ELAN
- **SignBank** — web app for sign lexicons; plugin for ELAN controlled vocabulary
- **ANNIS** — corpus search & visualization tool

### Notation
- **HamNoSys** — quasi-phonetic transcription (~200 symbols), Hamburg standard
- **SiGML** — XML wrapper of HamNoSys for driving 3D avatars
- **ID-glosses** — unique identifiers per sign (e.g., WATER-1); no universal standard

### SignWriting Resources
- **partner-dict** (partner-university Centre, partner-university University) — multilingual; integrates SignWriting + HamNoSys; free; use as a **notation learning reference** only — annotators study it to understand the system (hand shapes, location, movement encoding), then write original SPJ entries from scratch. This is NOT copying; SignWriting is a universal notation system like musical notation.
- **SignPuddle Online** — community sign partner-dictnaries including SPJ entries; free; reference for SPJ-specific notation conventions already established by the community
- **SignBank+** — cleaned multilingual text ↔ SignWriting dataset (open source) — structural reference only
- **signwriting Python package** — format utilities: parse, tokenize, visualize SignWriting (MIT license; format handling only — NOT a model)

**Important:** Pose → SignWriting directly is NOT possible without training a new model. No off-the-shelf tool exists for this path. The practical path is: annotate signs with ID-glosses (active learning pipeline) → trained annotators write original SPJ SignWriting entries for each gloss using partner-dict/SignPuddle as a notation guide → build SPJ SignWriting partner-dictnary as a by-product of Phase 3 annotation. No ML training needed for this step.

### Mouthing & Facial Analysis
- **AV-HuBERT** (Meta, 2022) — self-supervised audio-visual speech model; video-only mode works for mouthing transcription; public checkpoint on Hugging Face; transfer learning base for Slovak mouthing recognition; verify checkpoint license before distributing derived models
- **SLAN-tool** — semi-automatic sign language annotation tool; integrates with ELAN; outputs predicted mouthings as ELAN-reviewable tiers; open source (ACL 2022)
- **MediaPipe Face Mesh** (already included as part of Holistic) — 468 landmarks including full lip contours; input to both mouthing classifier and mouth gesture detector
- **SignMouth (2025)** — research reference showing mouthing fusion improves sign translation SOTA; validates including mouthing tier

**ELAN tiers for mouthings** (consistent with DGS-Korpus conventions):
- `S1_Mouthing` — mouthed spoken words (e.g., `voda` for the WATER sign)
- `S1_Mouth_Gesture` — non-spoken mouth patterns (puffed cheeks, pursed lips, etc.)

### Pose Estimation
- **MediaPipe Holistic (Google)** — 540+ keypoints (33 body + 42 hands + 468 face); real-time, free, cross-platform — recommended for SPJ pipeline
- **OpenPose (CMU)** — higher accuracy, GPU required, better for offline batch processing (CMU academic/non-commercial license — use MediaPipe for any commercially-sensitive pipeline)
- Output: `.pose` binary format (sign-language-processing library)

---

## AI-Assisted Annotation Workflow

Building an annotated corpus manually is extremely slow (~100 hours per video for full gloss annotation). The SPJ-Korpus uses a human-in-the-loop active learning pipeline to reduce this to ~10–20 hours per video by the mature phase.

### Pipeline Overview

```
Own SPJ videos (~300) + Museum content (100h+)   ← fully owned
    + Partner organization videos (with subtitles) ← with agreements
    + Phase 2 corpus recordings (new)
         ↓
    MediaPipe Holistic (pose extraction)
         ↓
    .pose files + EAF pre-annotation (AI tier)
         ↓
    Annotator review in ELAN
         ↓                              ↓ (parallel)
    Corrected EAF → training pool    MediaPipe face landmarks →
         ↓                            AV-HuBERT mouthing detection →
    Retrain sign classifier           SLAN-tool → ELAN S1_Mouthing tier
    (milestone-based)
         ↓
    Better AI suggestions → faster annotation loop
         ↓
    ID-gloss annotations → SignWriting (annotators write original SPJ entries using partner-dict/SignPuddle as notation guide)
         ↓
    SPJ SignWriting partner-dictnary (no ML training required for this step)
```

### How It Works

The AI does **not** replace the annotator — it pre-populates an ELAN tier (`AI_Gloss_RH` / `AI_Gloss_LH`) with candidate sign boundaries and gloss suggestions. The human annotator reviews, corrects, and approves each suggestion in standard ELAN. Corrections flow back into the training pool for the next retrain.

### Accuracy & Milestone Tracking

| Milestone | Training Signs | Expected Accuracy | Observed | Est. Annotation Speed |
|---|---|---|---|---|
| Bootstrap | 500 | ~10–15% top-3 | **31.9% top-3, 24.9% top-1** (category transfer) | 100h/video |
| v1 | 2,000 | ~50–60% | — | 40h/video |
| v2 | 5,000 | ~70–75% | — | 20h/video |
| v3 | 10,000+ | ~85–90% | — | 10h/video |

*Accuracy = correct gloss in top-3 AI suggestions. Speed estimates assume 60-minute video, 2-person annotation team.*

**Bootstrap results (March 2026):** Using category→word transfer learning (not external backbone), the bootstrap model achieved 2x the original estimate. The category model (102 SPJ grammatical categories, 44.9% val) provides a far more effective encoder than generic multilingual models. See README.md for full results table.

### Cold-Start Strategy

SPJ has no existing ML training data, and no public Czech Sign Language (ČZJ) pre-trained model exists.

**What worked (March 2026):** Internal category→word transfer learning. Trained a category classifier on 102 SPJ grammatical categories (from category-organized vocabulary, 5,193 videos). Then transferred the encoder weights to initialize the word-level model (516 signs). Result: 24.9% top-1, 31.9% top-3 — 2x the original bootstrap estimate.

**What didn't work as well:**
- Self-supervised pre-training (masked pose modeling on 13K segments): only +6% relative over baseline
- External backbones (SignBERT, OpenHands): impractical due to architecture mismatches (different landmark count, model format, unmaintained repos)
- Training on all labels at once (12K+ classes): 0% accuracy — most labels are singletons

**Lesson:** Domain-specific supervised features from a related SPJ task (categories) transfer far more effectively than either generic unsupervised learning or external cross-language backbones. Building internal data assets is more valuable than importing external models.

**Do NOT** assume a ready-made ČZJ model exists — it does not.

### ELAN Integration (No Custom Annotation UI)

Building a custom annotation interface is out of scope. Instead, the AI pipeline outputs ELAN-compatible `.eaf` files with pre-populated AI suggestion tiers:

- `AI_Gloss_RH` — AI-suggested right-hand glosses with timestamps
- `AI_Gloss_LH` — AI-suggested left-hand glosses with timestamps
- `AI_Confidence` — confidence score per segment (0.0–1.0), visible in ELAN

Annotators work entirely in standard ELAN, reviewing and correcting the AI tiers.

### Retraining Schedule (Milestone-Based)

Retraining is triggered by training pool size, **not** by time or number of videos. Early retraining on tiny datasets wastes resources and overfits.

| Trigger | Action |
|---|---|
| 500 annotated signs | Fine-tune generic backbone (SignBERT/OpenHands) on SPJ bootstrap data |
| 2,000 annotated signs | v1 retrain — first SPJ-specific model |
| 5,000 annotated signs | v2 retrain — improved accuracy, begin active learning |
| 10,000+ annotated signs | v3 full retrain + evaluation on held-out test set |

### SignWriting

SignWriting is an **optional** annotation tier for human readability and public-facing lexicon documentation. It is **not** the primary AI output target. The ML/NLP pipeline (SHuBERT, SignCLIP, Sign2GPT) operates on ID-gloss sequences, not SignWriting notation. SignWriting can be added post-hoc by annotators for selected signs in the SPJ lexicon.

---

## AI/ML Landscape

### Sign Language Recognition (Video → Text)

| Approach | Key Models | Notes |
|---|---|---|
| Isolated SLR | CNN-LSTM, Swin Transformer, SignKeyNet | 85–93% accuracy on benchmarks |
| Continuous SLR | CTC + Transformer, SignFormer-GCN | PHOENIX14T standard benchmark |
| Gloss-free SLT | Sign2GPT, GFSLT-VLP, SHuBERT | End-to-end, no gloss annotation needed |

**Best foundation model:** SHuBERT (ACL 2025) — self-supervised, trained on ~1,000h ASL, SOTA on How2Sign and OpenASL. Applicable to low-resource languages via transfer learning. (no public checkpoint as of 2025 — verify at shubert.pals.ttic.edu before use)

### Sign Language Production (Text → Sign)

| Approach | Key Models | Notes |
|---|---|---|
| Text → Gloss → Pose → Video | sign-language-processing pipeline | Open source, modular |
| LLM-based | SignLLM (ICCV 2025) | 8 sign languages, SOTA SLP (dataset CC-BY-NC-4.0; model not open-sourced — commercial partnership) |
| Autoregressive | T2S-GPT (ACL 2024) | Discrete token generation |
| Diffusion | SignGen (ECCV 2024) | Photo-realistic video |

**Practical fallback:** SignBERT (Hu et al., ICCV 2021) — MIT license, public checkpoint, hand-model-aware pretraining; usable immediately as transfer learning backbone.

### Avatar Systems
- **sign.mt** — real-time multilingual bidirectional; photorealistic avatars; open source pipeline
- **MMS-Player (DFKI)** — open-source avatar animation from notation (Blender, GPL-3.0)
- **SignSplat** — Gaussian Splatting on SMPL-X mesh, photorealistic; 2025
- **SignAvatar** — speech → signing avatar; trialled at Belgrade airport

### Key GitHub Resources
- https://github.com/sign-language-processing — full pipeline (pose, translation, avatar)
- https://github.com/sign/translate — real-time translation web app
- https://github.com/AI4Bharat/OpenHands — pose-based SLR, 6 languages
- https://github.com/ryanwongsa/Sign2GPT — gloss-free SLT
- https://github.com/ZhengdiYu/SignAvatars — 3D SL motion dataset
- https://github.com/DFKI-SignLanguage/MMS-Player — open-source avatar
- https://github.com/VIPL-SLP/awesome-sign-language-processing — curated resource list

---

## Project Phases

### Phase 1 — Community & Preparation (Months 1–6)
- [ ] Contact Slovak Deaf community organizations (Slovenský zväz sluchovo postihnutých)
- [ ] Partner with Trnava University (Roman Vojtechovský) as co-investigator
- [ ] Study DGS-Korpus annotation conventions (AP03-2018-01) — adapt for SPJ
- [ ] Set up ELAN with SPJ tier templates
- [ ] Catalog SPJ vocabulary from available partner-dictnaries; negotiate data-sharing agreements for ML training use
- [ ] Set up SignBank instance with SPJ ID-glosses

### Phase 2 — Data Collection (Months 6–18)
- [ ] Inventory and assess own corpus assets: ~300 SPJ videos (with Slovak subtitles) + 100+ hours museum SPJ content + partner organization videos — assess annotation coverage, extract MediaPipe pose as first training pool
- [ ] Formalize data-sharing agreements with partner organizations in writing (scope: ML training, derivative datasets, publication)
- [ ] Record 20–50 native Deaf SPJ signers (geographic spread: Bratislava, Košice, Banská Bystrica)
- [ ] Recording setup: 2–3 cameras, neutral background, 1080p / 50fps minimum
- [ ] Task types: free conversation (pairs), Frog Story, picture description, structured vocabulary
- [ ] Sociolinguistic metadata per signer (age, gender, city, age of acquisition, schooling)
- [ ] Video consent forms in SPJ + GDPR written consent
- [ ] Target: 20–50 hours initial corpus

### Phase 3 — Annotation (Months 12–30)

**Team:** 2 native Deaf SPJ annotators + 1 hearing linguist + 1 ML engineer (part-time from Month 10)

#### Stage A — Bootstrap (Months 12–14, fully manual)
- [ ] Annotate 5–10 videos by hand in ELAN — no AI assistance at this stage
- [ ] Establish inter-annotator reliability protocol (Cohen's κ target ≥ 0.8)
- [ ] Finalize ELAN tier template: `S1_Gloss_RH`, `S1_Gloss_LH`, `S1_Translation`, `S1_Mouthing`, `S1_NonManual`
- [ ] Establish mouthing annotation conventions for SPJ: mouthings in lowercase Slovak (e.g., `voda` for WATER sign); mouth gestures on separate tier `S1_Mouth_Gesture` (DGS-Korpus convention, adapted for SPJ)
- [ ] Produce ~500 annotated signs as initial training pool
- [ ] Build initial SPJ ID-gloss vocabulary in SignBank

#### Stage B — Transfer Bootstrap (Months 14–15)
- [ ] Use SignBERT (MIT, public checkpoint) or OpenHands ASL model (Apache 2.0) as pre-trained backbone — no public ČZJ model exists
- [ ] Fine-tune on ~500 SPJ bootstrap signs via transfer learning
- [ ] Build AI pre-annotation script: video → MediaPipe → `.eaf` with `AI_Gloss_RH` / `AI_Gloss_LH` tiers
- [ ] Validate: human spot-checks AI suggestions on 2 held-out videos before deploying to team

#### Stage C — Active Learning Loop (Months 15–30)
- [ ] AI pre-annotates each new video → outputs EAF file with AI suggestion tiers
- [ ] Annotators review and correct in ELAN (corrections tracked as diffs vs. AI tier)
- [ ] Corrected EAFs merged into training pool
- [ ] Retrain model at milestones: 2,000 → 5,000 → 10,000 annotated signs (see Retraining Schedule above)
- [ ] Track annotation speed (hours/video) and AI accuracy per retrain cycle

#### Annotation Priority Order
1. ID-glosses (both hands — `Gloss_RH`, `Gloss_LH`)
2. Slovak translations per utterance
3. Mouthings
4. Non-manual signals (eyebrows, mouth, head movement)
5. HamNoSys (optional — most time-intensive; add for selected signs)
6. SignWriting (optional — annotators create original SPJ entries using partner-dict/SignPuddle as a notation learning guide; no separate ML training required)

#### Quality Targets
- Inter-annotator agreement: Cohen's κ ≥ 0.8 on gloss tier
- AI suggestion accuracy: top-3 recall ≥ 70% by v2 milestone (5,000 signs)
- Target volume: 20–30 annotated hours by end of Phase 3

### Phase 4 — AI Infrastructure (Months 10–36)

*Note: Initial AI setup (Months 10–15) overlaps with Phase 2 and early Phase 3. The ML engineer is hired part-time from Month 10 to build the pose extraction pipeline and bootstrap the first sign classifier before Phase 3 annotation begins at scale.*

#### Months 10–15 (Setup & Bootstrap)
- [ ] Extract MediaPipe pose from own 300 SPJ videos + museum content — this is the first training pool, available before Phase 2 recordings begin
- [ ] Extract MediaPipe Holistic keypoints from available vocabulary videos → `.pose` files
- [ ] Build batch processing pipeline: video → pose → EAF pre-annotation
- [ ] Use SignBERT (MIT, public checkpoint) or OpenHands ASL model (Apache 2.0) as transfer learning baseline — no ČZJ model exists
- [ ] Build and test ELAN AI-tier output format (`AI_Gloss_RH`, `AI_Confidence`)
- [ ] Establish train/validation/test split protocol for SPJ data

#### Months 15–24 (Active Learning Infrastructure)
- [ ] Automate retrain pipeline: corrected EAF → training examples → model update
- [ ] Build training pool management system (track annotated sign count per milestone)
- [ ] Deploy v1 model (2,000 signs) and v2 model (5,000 signs)
- [ ] Evaluate on held-out test set; document accuracy and annotation speed per cycle
- [ ] Build mouthing detection pipeline: MediaPipe face landmarks → fine-tune AV-HuBERT on ~300 manually annotated SPJ mouthings → SLAN-tool integration for ELAN review → iterative retraining on corrected annotations. Expected accuracy: ~30–50% WER after fine-tuning (vs 60–70% zero-shot). Mouth gesture classifier: train separate SVM/CNN on ~100–200 labeled SPJ mouth gestures.

#### Months 24–36 (Full Pipeline)
- [ ] Build parallel corpus: SPJ video ↔ pose ↔ ID-glosses ↔ Slovak text
- [ ] Train continuous SLR model (CTC + Transformer on gloss sequences)
- [ ] Transfer learning from SHuBERT (ASL → SPJ) or SignCLIP (multilingual alignment)
- [ ] Build text-to-sign pipeline: Slovak text → gloss → pose → avatar video
- [ ] Create PHOENIX14T-style SPJ benchmark for standardized evaluation

### Phase 5 — Dissemination (Months 36+)
- [ ] Deposit corpus at CLARIN Slovakia or The Language Archive (MPI Nijmegen)
- [ ] Publish annotation conventions as working paper
- [ ] Submit to sign-lang@LREC workshop
- [ ] Register in SL Data Compendium (University of Hamburg)
- [ ] Apply for EU funding: Horizon Europe, Erasmus+, ERC Starting Grant
- [ ] Publish SPJ SignWriting partner-dictnary — original entries created during Phase 3 annotation using partner-dict/SignPuddle as notation reference; no training required

---

## Key Scientific Papers

| Paper | Venue | Year | Relevance |
|---|---|---|---|
| SHuBERT (Gueuwou et al.) | ACL 2025 | 2025 | Best foundation model for SLR/SLT |
| SignCLIP (Jiang et al.) | EMNLP 2024 | 2024 | 41-language text-sign alignment |
| Sign2GPT (Wong et al.) | ICLR 2024 | 2024 | Gloss-free SLT with LLMs |
| T2S-GPT (Chen et al.) | ACL 2024 | 2024 | Text → sign autoregressive |
| SignLLM (Fang et al.) | ICCV 2025 | 2025 | Multilingual SLP, 8 languages |
| SignAvatars (Yu et al.) | ECCV 2024 | 2024 | 3D holistic motion dataset |
| [public SL dataset] | arXiv 2024 | 2024 | 25+ SL, 3,207h |
| OpenHands (Selvaraj et al.) | ACL 2022 | 2022 | Pose-based SLR, low-resource SL |
| DGS-Korpus AP03-2018-01 | Hamburg 2018 | 2018 | Gold standard annotation conventions |
| Low-Resource Glossing + Augmentation | ACL 2025 | 2025 | Data augmentation for small corpora |
| SignBERT (Hu et al.) | ICCV 2021 | 2021 | MIT license, public checkpoint, hand-model-aware pretraining — practical transfer learning backbone |
| SignMouth (et al.) | arXiv 2025 | 2025 | Mouthing fusion for sign translation — SOTA on PHOENIX14T |
| Transfer VSR→Mouthing | arXiv 2025 | 2025 | Transfer learning from lip reading to sign language mouthing detection (DGS — directly applicable to SPJ) |

---

## Ethical Considerations

1. **Consent in SPJ** — video consent forms (not just written Slovak); participants must understand in their own language
2. **Anonymization paradox** — facial expressions carry grammar; full anonymization destroys data; use pose-only parallel versions
3. **Community involvement** — Deaf researchers as co-investigators, not just subjects ("Nothing about us without us")
4. **GDPR** — explicit consent, EU-compliant data management agreements required
5. **Small community** — signing styles can identify individuals; extra privacy care needed
6. **Open licensing** — CC BY-NC-SA recommended to maximize research reuse

---

## Quick Start (Immediate Next Steps)

1. Download and install ELAN: https://archive.mpi.nl/tla/elan
2. Browse DGS-Korpus annotation guidelines: https://www.dgs-korpus.de
3. Explore sign-language-processing toolkit: https://github.com/sign-language-processing
4. Catalog available SPJ video sources
5. Contact: Trnava University for academic partnership
6. Run MediaPipe Holistic on available SPJ videos as first pose extraction test
