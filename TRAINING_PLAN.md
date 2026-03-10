# Training Plan — Multi-Source + Unified Model

## Status: WAITING FOR DOWNLOADS

### Downloads Running
- [ ] partner-dict: ~10,622 videos (A_ front + B_ side) + 8,061 SignWriting PNGs
- [ ] reference-dict: ~37,395 YouTube videos (60° + 90° angles, unlisted)

### Existing Data (ready)
- intl-vocab: 10,008 videos, 19,481 segments, 13,237 labels
- category-source: 5,192 videos (dual-view _60/_90), 14,309 segments, 4,773 labels
- art-vocab: 421 videos, 211 segments, 106 labels
- fin-vocab: 49 videos, 46 segments
- career-vocab: 44 videos, 45 segments
- climate-vocab: 41 videos, 41 segments

---

## Phase 1: Process New Downloads

### 1a. partner-dict Pipeline
1. Pose extract all partner-dict videos (A_ + B_ views) → `data/pose/partner-dict/`
2. Create alignment: filename→label mapping from `partner-dict_metadata.csv` (name + translation)
3. Export NPZ segments
4. Add to manifest

### 1b. reference-dict Pipeline
1. Pose extract all reference-dict videos (60° + 90°) → `data/pose/reference-dict/`
2. Create alignment: filename→label from `reference-dict_sk_metadata.csv` (title/slug)
3. Export NPZ segments
4. Add to manifest

---

## Phase 2: Per-Source Models

Train separate models to see which sources contribute most:

| Model | Source | Expected Labels |
|-------|--------|----------------|
| model_partner-dict | partner-dict only | ~3,978 |
| model_reference-dict | reference-dict only | ~11,781 |
| model_intl-vocab | intl-vocab only | ~13,237 |
| model_category-source | category-source only | ~4,773 |
| model_art-vocab | art-vocab only | ~106 |

Each uses: category transfer → two-phase fine-tuning (same as best approach).
Include both angles (60°/90°, A_/B_) where available.

---

## Phase 3: Unified Combined Model

1. Normalize labels across all sources (lowercase, strip variants)
2. Cross-source label matching (same sign in multiple sources = more samples)
3. Combine all NPZ segments into unified manifest
4. Filter: labels with 2+ samples (cross-source counts)
5. Train with category transfer, two-phase, 10x augmentation
6. Expected: highest accuracy due to most data + multi-angle views

---

## Key Rules
- All angles (60°/90°, A_/B_, front/side) included as separate training samples
- Label normalization: lowercase, strip `_\d+$`, unify across sources
- Private research use only — separate from public training
- Use `.venv/bin/python` for all commands
- PYTORCH_ENABLE_MPS_FALLBACK=1 for MPS training
