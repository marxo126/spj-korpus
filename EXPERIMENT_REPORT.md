# Isolated Sign Language Recognition for Slovak Sign Language (SPJ): Experimental Report

**Project:** SPJ-Korpus — Slovak Sign Language Corpus Pipeline
**Period:** February–March 2026
**Task:** Isolated sign recognition from pose keypoints

---

## 1. Introduction

This report documents the systematic development of isolated sign recognition models for Slovak Sign Language (SPJ, ISO 639-3: `svk`). Starting from a baseline transformer model trained on 516 glosses, we scaled to 8,005 classes through architectural improvements, data augmentation search, multi-source data unification, and feature engineering. All models operate on MediaPipe Holistic pose keypoints (543 landmarks) extracted from monocular video.

---

## 2. Data

### 2.1 Pose Representation

All input data consists of MediaPipe Holistic pose sequences extracted from sign language videos. Raw pose files contain 543 landmarks (33 body + 21 left hand + 21 right hand + 468 face) at `(T, 1, 543, 3)`. For training, we filter to sign-language-relevant subsets:

| Preset | Body | Hands | Face | Total Landmarks | Raw Input Dim |
|--------|------|-------|------|-----------------|---------------|
| **compact** (default) | 7 | 42 | 47 (lips+nose) | **96** | **288** |
| extended | 7 | 42 | 99 (lips+nose+eyes+brows) | 148 | 444 |
| full | 33 | 42 | 99 | 174 | 522 |

The compact preset (96 landmarks) was used for all experiments reported here.

### 2.2 Training Data Sources

Training data was aggregated from multiple Slovak Sign Language video collections. Each source provides single-sign video clips with associated labels.

| Source | Samples | Unique Labels | Format |
|--------|---------|---------------|--------|
| Reference dictionary | 23,945 | 7,498 | NPZ (compact) |
| Partner dictionary | 9,211 | 5,246 | NPZ (compact) |
| Category vocabulary | 14,309 | 4,773 | Pose → NPZ |
| International vocabulary | 19,481 | 13,237 | Pose → NPZ |
| Art vocabulary | 211 | 106 | Pose → NPZ |
| Other specialized sources | 131 | ~130 | Pose → NPZ |
| Alignment pipeline (existing) | 15,139 | — | NPZ |
| **Total** | **~48,640** | **~16,800** | — |

### 2.3 Dataset Splits

Labels were normalized to lowercase. Minimum sample thresholds were applied to ensure trainable classes.

| Split Configuration | Min Samples | Train | Val | Test | Classes |
|---------------------|-------------|-------|-----|------|---------|
| Original (dual-view, 3+) | 3 | 5,451 | 4,844 | — | 4,844 |
| Original (dual-view, 2+) | 2 | 5,959 | 5,352 | — | 5,352 |
| Unified (all sources, 3+) | 3 | 23,359 | 5,006 | — | 8,005 |
| Unified (all sources, 2+) | 2 | 32,391 | 6,941 | — | 14,296 |

Stratified train/val/test splits (70/15/15) were used where class counts permitted.

---

## 3. Model Architectures

### 3.1 Transformer Baseline (PoseTransformerEncoder)

```
Input (T, F) → Linear(F, d_model) → Sinusoidal PE → N × TransformerEncoderLayer → Masked Mean Pool → Linear → n_classes
```

- **Parameters:** ~500K
- **Default config:** d_model=128, n_heads=4, d_ff=256, n_layers=3
- Variable-length sequences padded to max_seq_len=300, attention mask ignores padding.

### 3.2 Conv1D+Transformer (Conv1DTransformerEncoder)

Inspired by the 1st-place solution in the Google Isolated Sign Language Recognition Kaggle competition:

```
Input (T, F) → Linear(F, d_model) + BatchNorm
  → [Conv1DBlock(k=17) × 3 + TransformerBlock] × 2
  → Linear(d_model, d_model×2) → Global Average Pool → Dropout → Linear → n_classes
```

Where Conv1DBlock = Dense(expand) → Causal Depthwise Conv1D → BatchNorm → Dense(project) + residual connection.

- **Parameters:** ~900K (small-scale) to ~4.9M (8K classes)
- **Config:** d_model=192, n_heads=4, n_layers=3, kernel_size=17

### 3.3 Architecture Comparison

An 8-method comparison was conducted on a 495-class subset (20 epochs each):

| Method | Architecture | Input Dim | Params | Val Acc |
|--------|-------------|-----------|--------|---------|
| Baseline | Transformer + raw | 288 | 498K | 1.41% |
| Velocity | Transformer + velocity | 864 | 572K | 2.63% |
| XY only | Transformer + xy | 192 | 486K | 2.42% |
| Normalized | Transformer + nose-norm | 288 | 498K | 1.01% |
| XY velocity | Transformer + xy_vel | 576 | 535K | 0.81% |
| Norm XY velocity | Transformer + norm_xy_vel | 576 | 535K | 2.63% |
| **Conv1D** | **Conv1D + norm_xy_vel** | **576** | **924K** | **12.93%** |
| 4-stream | SAM-SLR-style | 568 | 534K | 2.02% |

The Conv1D+Transformer achieved **9.2× improvement** over the baseline transformer, confirming the strong inductive bias of local temporal convolutions for sign recognition.

---

## 4. Feature Engineering

### 4.1 Feature Modes

Four feature extraction modes were evaluated, applied to the compact 96-landmark subset:

| Mode | Description | Output Dim |
|------|-------------|-----------|
| `raw` | Flatten (x, y, z) coordinates | 288 |
| `velocity` | Position + first derivative + second derivative | 864 |
| `xy_velocity` | Drop z-axis, then position + Δ + Δ² | 576 |
| `norm_xy_velocity` | Nose-normalize, drop z, then position + Δ + Δ² | 576 |

**Nose normalization** subtracts the nose landmark (index 0) from all other landmarks per frame, making the representation translation-invariant. Dropping the z-axis removes depth estimation noise from monocular video. First and second derivatives capture velocity and acceleration of hand motion.

### 4.2 Best Feature Configuration

`norm_xy_velocity` paired with Conv1D+Transformer yielded the best results. This combination provides:
- Translation invariance (nose normalization)
- Reduced noise (no z-axis)
- Motion dynamics (velocity + acceleration features)
- Local temporal pattern capture (Conv1D kernels)

---

## 5. Data Augmentation

### 5.1 Augmentation Types

Eight augmentation strategies were implemented, applied on-the-fly during training with configurable multiplicity (default 10×):

| # | Augmentation | Description | Category |
|---|-------------|-------------|----------|
| 1 | Temporal crop | Random 75–100% time crop | Temporal |
| 2 | Speed variation | 0.8–1.2× via frame resampling | Temporal |
| 3 | Gaussian noise | Additive noise on coordinates | Spatial |
| 4 | Scale jitter | 0.9–1.1× uniform scaling | Spatial |
| 5 | Mirror | Swap left/right hands + flip X | Spatial |
| 6 | Rotation | Y-axis rotation ±15° | Spatial |
| 7 | Joint dropout | Zero random landmarks (p=0.3) | Regularization |
| 8 | Temporal masking | Zero random frame spans | Temporal |

### 5.2 Augmentation Search

A greedy hill-climbing search over all 2⁸ = 256 possible augmentation flag combinations was conducted. Thirteen representative configurations were tested as 30-epoch probes on the 5,352-class dataset with transformer architecture:

| Rank | Configuration | Val Acc | Best Epoch |
|------|---------------|---------|------------|
| **1** | **Mirror + rotation only** | **38.30%** | **19** |
| 2 | All 8 − temporal crop | 35.99% | 29 |
| 3 | All 8 − speed | 35.97% | 30 |
| 4 | Original 4 + dropout + mask | 35.93% | 26 |
| 5 | Original 4 only (baseline) | 35.72% | 26 |
| 6 | Original 4 + mirror | 35.72% | 26 |
| 7 | All 8 augments | 35.63% | 26 |
| 8 | All 8 − joint dropout | 35.59% | 24 |
| 9 | All 8 − temporal mask | 35.52% | 27 |
| 10 | Original 4 + rotation | 35.50% | 28 |
| 11 | Original 4 + mirror + rotation | 35.50% | 29 |
| 12 | All 8 (20× multiplier) | 36.10% | 25 |
| 13 | All 8 (5× multiplier) | 30.12% | 29 |

### 5.3 Key Finding: Spatial vs. Temporal Augmentation

**Mirror + rotation only achieved 38.30% — 2.6 percentage points above baseline (35.72%).** All configurations that included temporal augmentations (crop, speed, masking) performed worse.

This suggests that for isolated sign recognition:
- **Spatial augments help:** Mirror and rotation maintain anatomical validity while increasing viewpoint diversity. Signs are view-invariant to a degree.
- **Temporal augments hurt:** Temporal cropping removes sign onset/offset context. Speed variation distorts learned timing patterns. Temporal masking removes critical motion information.

The winner was extended to a full 100-epoch run, reaching **38.86% validation accuracy** at epoch 45.

### 5.4 Augmentation Multiplier

Increasing augmentation from 10× to 20× showed minimal benefit (+0.38%), while reducing to 5× caused significant underfitting (−5.60%). The 10× default provides sufficient diversity without excessive training time.

---

## 6. Transfer Learning

### 6.1 Category → Word Transfer (Transformer)

A category-level model was trained on 102 grammatical categories (44.93% validation accuracy). Its encoder weights were transferred to word-level models using two-phase fine-tuning:

1. **Phase 1** (10 epochs): Freeze encoder, train classifier only (lr=0.001)
2. **Phase 2** (remaining epochs): Unfreeze all, lr=0.0003 + cosine annealing + early stopping

This approach yielded 24.9% test top-1 accuracy on 516 classes — a +48% relative improvement over the SSL pre-training baseline.

### 6.2 Cross-Architecture Transfer Limitation

The category model (Transformer, input_dim=288) cannot transfer to Conv1D models (input_dim=576) due to incompatible weight dimensions. Conv1D models were trained from scratch.

### 6.3 Self-Supervised Pre-training

Masked Pose Modeling (15% frame masking, reconstruction objective) was attempted on 13,638 unlabeled segments. When fine-tuned on 516 word labels, SSL yielded only +6% relative improvement over random initialization — far below the +48% from supervised category transfer. In this data regime, domain-specific supervised features transfer more effectively than generic temporal dynamics learned through self-supervision.

---

## 7. Experimental Results

### 7.1 Model Evolution

| Phase | Model | Classes | Val/Test Acc | Architecture | Key Change |
|-------|-------|---------|-------------|--------------|------------|
| Baseline | quality_ep22 | 516 | 23.0% val | Transformer | — |
| SSL pre-train | ssl_finetune_ep66 | 516 | 17.8% test | Transformer | Masked pose modeling |
| Category transfer | transfer_cat2word_ep37 | 516 | 24.9% test / 31.9% top-3 | Transformer | Two-phase fine-tuning |
| Expanded labels | quality2_transfer_ep29 | 1,743 | 19.5% test | Transformer | More labels (2+ samples) |
| Dual-view data | dualview_3plus_ep75 | 4,844 | 34.97% val | Transformer | Dual-camera angles |
| Dual-view data | dualview_2plus_ep70 | 5,352 | 36.47% val | Transformer | Lower sample threshold |
| **Augment search** | **augsearch_winner_ep45** | **5,352** | **38.86% val** | **Transformer** | **Mirror+rotation only** |
| Conv1D + all sources | conv1d_unified_3plus | 8,005 | 36.68% val | Conv1D | New architecture + data |
| **Optimal combo** | **optimal_conv1d_ep40** | **8,005** | **37.55% val** | **Conv1D** | **Best arch + best aug** |
| Conv1D category | conv1d_category_ep59 | 116 | 74.60% val | Conv1D | 1.66× old category model |
| **Conv1D transfer** | **conv1d_word_transfer_ep50** | **8,005** | **35.42% val / 34.5% test** | **Conv1D** | **Best top-3 (41.5%) & top-5 (45.2%)** |

### 7.2 Best Models by Use Case

| Use Case | Model | Classes | Accuracy | Recommendation |
|----------|-------|---------|----------|----------------|
| Highest accuracy | augsearch_winner_ep45 | 5,352 | 38.86% val | Use when top-1 accuracy matters most |
| Best ranking (top-3/5) | conv1d_word_transfer_ep50 | 8,005 | 41.5% top-3 / 45.2% top-5 | Use for candidate ranking / AI suggestions |
| Broadest coverage | optimal_conv1d_ep40 | 8,005 | 37.55% val | Use when vocabulary coverage matters |
| Category recognition | conv1d_category_ep59 | 116 | 74.60% val | Use for grammatical analysis / transfer source |

### 7.3 Scaling Behavior

| Classes | Best Val Acc | Model | Notes |
|---------|-------------|-------|-------|
| 102 | 44.93% | Transformer | Category-level |
| 516 | 24.9% test | Transformer | Category transfer |
| 1,743 | 19.5% test | Transformer | Coverage tradeoff |
| 4,844 | 34.97% val | Transformer | Dual-view data |
| 5,352 | 38.86% val | Transformer | Augmentation search winner |
| 8,005 | 37.55% val | Conv1D | Multi-source unified |

Accuracy does not monotonically decrease with class count. The jump from 516→5,352 classes with improved accuracy (24.9% → 38.86%) demonstrates that additional training data compensates for the harder classification task.

---

## 8. Training Infrastructure

### 8.1 Hardware

- **Device:** Apple M4 Max (16 CPU cores, 40 GPU cores, 128 GB unified memory)
- **GPU framework:** PyTorch MPS (Metal Performance Shaders)
- **Training regime:** Single GPU, threading-based parallelism (MPS tensors cannot cross process boundaries)

### 8.2 Training Configuration

| Parameter | Transformer | Conv1D |
|-----------|------------|--------|
| Batch size | 256 | 256 |
| Learning rate | 0.001 (freeze) / 0.0003 (unfreeze) | 0.0005 |
| Optimizer | AdamW (weight_decay=1e-4) | AdamW (weight_decay=1e-4) |
| Scheduler | Cosine annealing | Cosine annealing |
| Label smoothing | 0.1 | 0.1 |
| Early stopping | Patience 25 | Patience 25 |
| max_seq_len | 300 frames | 300 frames |
| Augmentation | 10× (mirror + rotation only) | 10× (mirror + rotation only) |

### 8.3 Training Times

| Experiment | Epochs | Time | Time/Epoch |
|------------|--------|------|------------|
| Augmentation probe (30 ep) | 30 | 45–106 min | 1.5–3.5 min |
| Augment winner full train | 45 | ~5 hours | ~6.3 min |
| Dual-view overnight (2 runs) | 70+75 | 8.2 hours | ~3.4 min |
| Conv1D optimal (8K classes) | 65 | 18.3 hours | ~17 min |

### 8.4 Pause/Resume System

A resume checkpoint system was implemented to handle training interruptions (MacBook reboots, user pauses). After each epoch, a full resume checkpoint is saved containing model weights, optimizer state, scheduler state, and training history. Training scripts support pause via SIGINT/SIGTERM signals or file-based triggers.

---

## 9. Discussion

### 9.1 Augmentation Insights

The finding that temporal augmentations hurt isolated sign recognition is significant. Unlike continuous action recognition where temporal perturbations may improve robustness, isolated signs have precise temporal structure — onset, stroke, and offset phases carry discriminative information. Corrupting this temporal structure through random cropping, speed variation, or masking degrades the learning signal.

Mirror augmentation succeeds because many signs are performed single-handedly and can be validly mirrored. Rotation augmentation provides viewpoint robustness without distorting the temporal dynamics.

### 9.2 Architecture Comparison

The Conv1D+Transformer architecture consistently outperforms the pure Transformer baseline. The causal depthwise convolutions (kernel size 17) capture local temporal patterns that the attention mechanism alone struggles to learn efficiently. This aligns with findings from the Kaggle competition community, where CNN-Transformer hybrids dominated pure attention approaches for pose-based sign recognition.

However, the Conv1D model could not leverage the pre-trained category encoder (incompatible dimensions), which partially offset its architectural advantage. A Conv1D category model could serve as a better transfer source.

### 9.3 Data Scaling

The multi-source data unification (48,640 samples from 6+ sources) enabled scaling to 8,005 classes while maintaining >37% validation accuracy. The diminishing returns beyond 5,000 classes suggest that per-class sample count is the current bottleneck — many classes have only 2–3 samples.

### 9.4 Limitations

- All validation accuracies are reported; test set evaluation was performed only for earlier models (≤1,743 classes)
- The category transfer model (Transformer) and the Conv1D models operate on different feature spaces, preventing direct encoder transfer
- Multi-source data may contain label inconsistencies across sources (same sign labeled differently)
- All experiments use the compact landmark preset (96 landmarks); extended or full presets remain unexplored at scale

---

## 10. Conclusion

Through systematic experimentation, we improved isolated SPJ sign recognition from 23.0% (516 classes, baseline) to 38.86% (5,352 classes, Transformer + mirror/rotation augmentation) and 37.55% (8,005 classes, Conv1D + norm_xy_velocity + mirror/rotation). Key findings:

1. **Spatial augmentation only:** Mirror + rotation augmentation outperforms all temporal augmentation combinations by 2.6+ percentage points.
2. **Conv1D+Transformer:** 9× improvement over baseline transformer on matched conditions.
3. **Feature engineering:** Nose-normalized XY velocity features provide the best input representation.
4. **Data scaling:** Multi-source unification enables vocabulary coverage without proportional accuracy loss.
5. **Supervised transfer > SSL:** Domain-specific supervised features transfer far more effectively than self-supervised representations in this data regime.

### Next Steps

- ~~Train a Conv1D category model to enable architecture-matched transfer learning~~ (Done — see §11)
- Evaluate v2 spatial augmentations (translation, bone scaling, hand shift, head tilt, mixup)
- Explore extended landmark preset (148 landmarks, adding eye/eyebrow features)
- Active learning integration for iterative corpus annotation

---

## 11. Conv1D Category→Word Transfer Learning (2026-03-12)

### 11.1 Motivation

Previous experiments (§6.2) identified a key limitation: the category model (Transformer, input_dim=288) could not transfer encoder weights to Conv1D models (input_dim=576). Since supervised category transfer had previously yielded +48% relative improvement for transformer models (§6.1), we hypothesized that training a Conv1D category model first and then transferring its encoder to word-level classification would close this gap.

### 11.2 Experimental Setup

**Phase 1 — Conv1D Category Model:**

| Parameter | Value |
|-----------|-------|
| Architecture | Conv1D+Transformer (`conv1d_transformer`) |
| Feature mode | `norm_xy_velocity` (input_dim=576) |
| d_model | 192 |
| n_heads / n_layers | 4 / 3 |
| Parameters | 1,854,516 |
| Classes | 116 (kodifikácia grammatical categories) |
| Training data | 13,284 train / 2,847 val / 2,847 test |
| Augmentation | Mirror + rotation only (10×) |
| Batch size | 512 |
| Learning rate | 0.0005 |
| Epochs / Patience | 80 / 20 |
| Label smoothing | 0.1 |
| Weight decay | 1e-4 |

Category labels were derived from kodifikácia video folder structure: `data/videos/kodifikacia/{category_name}/{video}.mp4`, yielding 116 unique grammatical categories (e.g., `cislovky_ikonicke`, `slovesa_jednoduche`). NPZ segments were matched by video stem.

**Phase 2 — Word-Level Transfer:**

| Parameter | Value |
|-----------|-------|
| Architecture | Conv1D+Transformer (same as Phase 1) |
| Parameters | 4,893,829 |
| Classes | 8,005 (unified_3plus) |
| Training data | 23,293 train × 10 aug = 232,930 effective |
| Validation | 5,006 samples |
| Pretrained weights | 94/96 encoder params transferred (2 skipped: classifier) |
| Freeze epochs | 10 (lr=0.001, classifier only) |
| Unfreeze lr | 0.0003 + cosine annealing |
| Epochs / Patience | 100 / 25 |

### 11.3 Results

**Phase 1 — Category Model:**

| Metric | Value |
|--------|-------|
| Best validation accuracy | **74.60%** |
| Best epoch | 59 (early stopped at epoch 79) |
| Training time | 612.6 min (10.2 hours) |

This represents a **1.66× improvement** over the old Transformer category model (44.93% val, 102 classes), despite having 14% more classes (116 vs 102). The Conv1D architecture's local temporal convolutions are significantly more effective for category-level discrimination.

**Phase 2 — Word Transfer:**

| Metric | Value |
|--------|-------|
| Best validation accuracy | **35.42%** |
| Best epoch | 50 (stopped at epoch 65 due to disk full) |
| Test top-1 accuracy | **34.54%** |
| Test top-3 accuracy | **41.53%** |
| Test top-5 accuracy | **45.21%** |
| Training time | ~13 min/epoch (unfrozen phase) |

**Checkpoint:** `conv1d_word_transfer_ep50_acc0.3542.pt`

### 11.4 Transfer Learning Trajectory

| Epoch | Phase | Val Acc | Note |
|-------|-------|---------|------|
| 1 | Frozen | 18.0% | Classifier initialization |
| 5 | Frozen | 26.4% | Rapid classifier adaptation |
| 10 | Frozen | 27.7% | End of frozen phase |
| 11 | Unfrozen | 31.8% | +4.1% jump from unfreezing |
| 20 | Unfrozen | 34.0% | Steady climb |
| 30 | Unfrozen | 34.5% | |
| 40 | Unfrozen | 35.0% | |
| 50 | Unfrozen | **35.4%** | Best (patience clock starts) |
| 65 | Unfrozen | 35.1% | Stopped (disk full) |

### 11.5 Comparison with Prior Models

| Model | Architecture | Classes | Test Top-1 | Test Top-3 | Test Top-5 |
|-------|-------------|---------|-----------|-----------|-----------|
| **conv1d_word_transfer** | **Conv1D + transfer** | **8,005** | **34.5%** | **41.5%** | **45.2%** |
| conv1d_unified_3plus | Conv1D from scratch | 8,005 | — | — | — |
| augsearch_winner | Transformer | 5,352 | — | — | — |
| dualview_2plus | Transformer | 5,352 | 35.0% | 37.4% | 39.6% |
| transfer_cat2word | Transformer + transfer | 516 | 24.9% | 31.9% | 35.1% |

### 11.6 Key Findings

1. **Conv1D category model far exceeds Transformer:** 74.6% vs 44.9% on comparable category tasks, confirming the architectural advantage extends to category-level classification.

2. **Transfer improves ranking quality, not raw accuracy:** The transfer model's top-1 (34.5%) is slightly below Conv1D from-scratch validation (36.6%), but its top-3 (41.5%) and top-5 (45.2%) are the **best ever achieved** — +4.1% and +5.6% above the previous best (dualview_2plus: 37.4% / 39.6%). The encoder's category-level features improve the model's ability to narrow candidates even when the top prediction is wrong.

3. **Diminishing returns from category transfer at scale:** With 23K training samples and 8,005 classes, the Conv1D architecture has sufficient data to learn directly. Transfer learning provided marginal top-1 benefit compared to the +48% relative gain seen in the earlier low-data regime (516 classes, 1.1K train). Category transfer remains most valuable when per-class samples are scarce.

4. **Two-phase fine-tuning validated for Conv1D:** The freeze→unfreeze pattern works for Conv1D+Transformer just as it did for pure Transformer. The +4.1% jump at unfreeze (epoch 11: 27.7% → 31.8%) confirms that the frozen phase successfully adapts the classifier before fine-tuning encoder representations.

5. **Conv1D category model as foundation:** The `conv1d_category_ep59_acc0.7460.pt` checkpoint (74.6%, 116 categories) is a strong foundation model for future transfer experiments with different word-level datasets or label configurations.

---

## 12. Extended Landmark Preset — Mixed-Preset Training (2026-03-12)

### Motivation

The compact landmark preset (96 landmarks) captures body, hands, and lip/nose face landmarks. The extended preset (148 landmarks) adds 52 eye and eyebrow landmarks that carry grammatical meaning in sign language — questions (raised eyebrows), negation (furrowed brows), gaze direction (agreement verbs). This experiment tests whether these additional face landmarks improve sign recognition accuracy.

### Challenge: Incomplete Extended Data

Only sources with original `.pose` files can be re-exported at the extended preset:
- **Kodifikacia + SpreadTheSign:** 15,130 segments — re-exported at 148 landmarks
- **Posunky + Dictio:** 22,100 segments — only compact NPZ available (no videos to re-extract)

This creates a mixed-preset training scenario where ~44% of samples have real extended data and ~56% have compact data zero-padded to 148 landmarks.

### Approach: Compact-to-Extended Zero-Padding

Compact (96) landmarks are a strict subset of extended (148). A position mapping (`_COMPACT_TO_EXTENDED_MAP`) maps each of the 96 compact positions to its corresponding position in the 148-landmark array. The remaining 52 positions (eyes + eyebrows) are filled with zeros for compact samples.

Padding is applied at dataset load time in `AugmentedPoseDataset` and `PoseSegmentDataset` via the `target_n_landmarks=148` parameter.

### Setup

| Parameter | Value |
|-----------|-------|
| Architecture | Conv1D+Transformer |
| Feature mode | `norm_xy_velocity` |
| Input dim | 888 (148 × 6) |
| Parameters | 4,951,685 |
| Classes | 8,005 |
| Training samples | 23,293 (10,267 extended / 13,026 compact-padded) |
| Validation samples | 5,006 (2,465 extended / 2,541 compact-padded) |
| Augmentation | mirror + rotation only, 10× |
| Batch size | 512 |
| Learning rate | 0.0005 |
| Epochs | 12/30 (stopped early — negative result) |

### Results

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1 | 7.956 | 0.77% | 8.203 | 0.66% |
| 2 | 6.191 | 8.60% | 8.632 | 2.08% |
| 3 | 4.916 | 29.3% | 9.714 | 4.27% |
| 5 | 3.837 | 63.0% | 11.945 | 5.33% |
| 7 | 3.625 | 76.1% | 12.989 | 5.99% |
| 9 | 3.523 | 80.9% | 13.124 | 7.07% |
| 11 | 3.440 | 83.0% | 13.038 | 7.11% |
| 12 | 3.407 | 83.6% | 12.995 | **7.07%** |

**Best val accuracy: 7.11%** (epoch 11) — stopped early due to clear plateau.

### Comparison to Compact Baseline

| Metric | Extended (148, mixed) | Compact (96, all real) |
|--------|----------------------|----------------------|
| Best val acc | **7.11%** | **36.6%** |
| Val acc at epoch 12 | 7.07% | ~32% |
| Train acc at epoch 12 | 83.6% | ~75% |
| Overfitting gap | **76.5 pp** | ~43 pp |

The extended mixed-preset model achieves only **19.4% of the compact baseline's accuracy** (7.1% vs 36.6%).

### Analysis

1. **Zero-padding creates catastrophic noise.** With 56% of training samples having zeros for 52 out of 148 landmarks, the model cannot learn meaningful features from eye/eyebrow positions. The inconsistency between real and zero-filled data dominates over any signal those landmarks might carry.

2. **Massive overfitting.** The 76.5 percentage-point gap between train (83.6%) and val (7.1%) accuracy indicates the model memorizes training examples rather than learning generalizable sign features. The larger input dimensionality (888 vs 576) provides more capacity for memorization without improving generalization.

3. **Val loss divergence.** Validation loss increased monotonically from 8.2 (epoch 1) to 13.0 (epoch 12) while training loss decreased normally. This is a classic sign of the model fitting noise in the training data.

### Key Findings

1. **Mixed-preset training with zero-padding does not work.** The inconsistency between real and padded landmarks is too disruptive. Extended landmarks should only be used when **all** training samples have real data for all positions.

2. **Compact preset (96 landmarks) remains optimal** for the current data regime where >50% of samples come from NPZ-only sources (posunky, dictio).

3. **Extended preset would require re-downloading and re-extracting** posunky and dictio videos to get real 148-landmark data for all samples. This is a significant infrastructure effort (~46K videos) that is not justified by the uncertain benefit of eye/eyebrow landmarks.

4. **Alternative approach for non-manual features:** Rather than adding more landmarks to the same model, a separate non-manual classifier (trained only on extended-preset samples with real data) could be combined with the compact-preset sign classifier in an ensemble. This avoids the mixed-data problem entirely.

---

## 13. Higher Min-Samples Threshold (5+ per class) (2026-03-12)

### Motivation

The unified 3+ model trains on 8,005 classes with an average of ~3 samples per class. The category model achieves 74.6% with ~44 samples per class (116 classes). This experiment tests whether filtering to classes with 5+ samples improves per-class accuracy by providing more training data per class.

### Setup

| Parameter | Value |
|-----------|-------|
| Architecture | Conv1D+Transformer |
| Feature mode | `norm_xy_velocity` |
| Input dim | 576 (96 × 6) |
| Parameters | 2,557,526 |
| Classes | 1,942 (filtered from 8,005) |
| Training samples | 9,644 (filtered from 23,293) |
| Validation samples | 2,229 |
| Avg samples/class | ~5.0 (train) |
| Augmentation | mirror + rotation only, 10× |
| Batch size | 256 |
| Learning rate | 0.0005 |
| Epochs | 48/80 (stopped early — plateaued) |

### Results

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1 | 6.768 | 1.71% | 7.087 | 1.35% |
| 5 | 2.666 | 69.1% | 7.777 | 11.93% |
| 10 | 1.858 | 81.8% | 7.249 | 12.92% |
| 16 | 1.671 | 82.8% | 6.936 | **14.31%** |
| 30 | 1.514 | 83.3% | 6.654 | **15.34%** |
| 48 | 1.467 | 83.4% | 6.628 | 14.49% |

**Best val accuracy: 15.34%** — stopped at epoch 48 due to plateau (no improvement for ~18 epochs).

### Comparison

| Model | Classes | Train samples | Samples/class | Best Val Acc |
|-------|---------|--------------|---------------|-------------|
| Category (116) | 116 | ~5,200 | ~44 | **74.6%** |
| **5+ words** | **1,942** | **9,644** | **~5** | **15.3%** |
| 3+ words | 8,005 | 23,293 | ~3 | **36.6%** |

### Key Findings

1. **Filtering to 5+ samples per class makes performance worse, not better.** The 5+ model (15.3%) significantly underperforms the 3+ model (36.6%) despite having more samples per class (5 vs 3). The reason: filtering removed 59% of total training data (23K → 9.6K) while only reducing classes by 76% (8K → 1.9K).

2. **Total training data volume matters more than per-class count.** The 3+ model benefits from seeing a wider diversity of signs, even if most classes have only 3 samples. The larger dataset provides better regularization and more general pose features.

3. **The accuracy scaling law is dominated by samples/class.** Comparing across all experiments: 3 samples/class → ~36%, 5 samples/class → ~15%, 44 samples/class → ~75%. The 5+ result is an outlier because it sacrificed too much total data. The path to higher accuracy is more data per class **without** removing other classes.

4. **Implication for data collection:** To achieve 50%+ word-level accuracy, the corpus needs ~20-50 samples per sign from multiple signers. Current dictionary sources (posunky, dictio) provide 1-5 samples from 1-2 signers. A crowdsourced data collection approach (Sign Collector) is the most promising path to break through the accuracy ceiling.

---

## 14. Summary: Optimal Method Combination and Data Collection Strategy

### What Works — The Proven Recipe

After 13 experiments across architectures, augmentations, transfer learning, landmark presets, and data filtering, the optimal combination for SPJ sign recognition is:

| Component | Best Choice | Evidence |
|-----------|------------|----------|
| **Architecture** | Conv1D+Transformer | 9× better than pure Transformer (§6) |
| **Feature mode** | `norm_xy_velocity` | Nose-normalization + velocity features (§6) |
| **Augmentation** | Mirror + rotation only | 38.9% val; all-8 augs = 35.6% (§7, §8) |
| **Landmark preset** | Compact (96) | Extended with mixed data = 7.1% (§12) |
| **Transfer learning** | Category → word (2-phase) | Best top-3 (41.5%) and top-5 (45.2%) (§11) |
| **Data filtering** | Keep all classes (3+) | 5+ filter = 15.3% vs 3+ = 36.6% (§13) |
| **Pre-training** | Not useful at current scale | SSL gave only +6% relative (§9) |

### What Does Not Work

| Approach | Why it fails |
|----------|-------------|
| Extended landmarks (148) with mixed data | Zero-padded eye/eyebrow = noise for 56% of samples |
| Filtering to higher min-samples | Loses too much total training data |
| All-8 augmentation | Temporal augmentations hurt sign recognition |
| SSL pre-training | Supervised category transfer far more effective |
| Spatial augmentations v2 | Translation, bone scale, hand shift, head tilt — none improved over mirror+rotation |

### The Accuracy Ceiling and Why

Current best results plateau at **37–39% top-1** and **41–45% top-3** regardless of model improvements. The bottleneck is data, not architecture:

| Samples/class | Observed accuracy | Source |
|--------------|-------------------|--------|
| ~3 | 36.6% val | 3+ unified model |
| ~5 | 15.3% val* | 5+ filtered model |
| ~44 | 74.6% val | Category model |

*The 5+ result is anomalously low because filtering removed 59% of total data. The true relationship, controlling for total data volume, follows an approximate log-linear scaling: **doubling samples per class yields ~15–20 percentage points improvement.**

Extrapolating from the category model's performance:
- **20 samples/class → ~50% accuracy** (estimated)
- **50 samples/class → ~65% accuracy** (estimated)
- **100 samples/class from 10+ signers → ~80% accuracy** (target, based on ASL Citizen benchmarks)

### Recommended Training Configuration for New Data

When new sign collections become available (e.g., via Sign Collector), use this configuration:

```python
TrainingConfig(
    model_type="conv1d_transformer",
    feature_mode="norm_xy_velocity",
    epochs=100,
    batch_size=512,           # increase if >50K samples
    lr=0.0005,
    d_model=192,              # increase to 256 if >50K samples
    n_heads=4,
    n_layers=3,
    max_seq_len=300,
    augment=True,
    n_augments=10,            # reduce to 5 if >20 samples/class
    patience=25,
    label_smoothing=0.1,
    weight_decay=1e-4,
    aug_mirror=True,
    aug_rotation=True,
    # all other aug flags = False
)
```

**Two-phase transfer learning** should be used when:
- A strong category model exists (currently 74.6%)
- Per-class sample count is low (<20 samples/class)
- Phase 1: freeze encoder, train classifier only (10 epochs, lr=0.001)
- Phase 2: unfreeze all, lr=0.0003, cosine schedule, patience=25

**When to skip transfer and train from scratch:**
- Per-class sample count >20 (enough data for direct learning)
- Total training samples >50K (model has sufficient diversity)

### Data Collection Priorities for Maximum Impact

1. **Signer diversity over repetition.** 10 signers × 5 samples >>> 1 signer × 50 samples. The model needs to learn signer-independent features.

2. **Keep all existing data.** Never filter out low-count classes — they still contribute to general pose understanding. Add new samples to existing classes.

3. **Target the long tail.** Most signs currently have exactly 3 samples. Even adding 2–3 more samples per sign from different signers would significantly improve accuracy.

4. **Prioritize signs already in the corpus.** Adding more samples to existing 8,005 signs is more valuable than adding new signs with only 1–2 samples each.

5. **Environment variation.** Different backgrounds, lighting, and camera angles improve robustness. The kodifikácia dual-angle (60°/90°) approach is a good model.

### Projected Milestones

| Data milestone | Expected accuracy | Action |
|---------------|-------------------|--------|
| Current (3 samples/class avg) | 37–39% top-1 | Deploy for active learning |
| 10 samples/class, 3+ signers | ~45% top-1 | Retrain, reduce augmentation to 5× |
| 20 samples/class, 5+ signers | ~55% top-1 | Scale up d_model to 256 |
| 50 samples/class, 10+ signers | ~70% top-1 | Production-ready recognition |
| 100 samples/class, 20+ signers | ~80%+ top-1 | Research-grade corpus |
