# Data Directory

**Video data is not included in this repository.**

The corpus videos and related data belong to partner organizations and are used under private agreements for ML training and research purposes only.

## Directory Structure

```
data/
  videos/              # Source videos (not tracked)
    partner-dictnary/        #   Isolated sign vocabulary videos (~10K)
    categories/        #   Category-organized vocabulary (~5.2K)
    sentences/         #   Word + sentence pair videos
  pose/                # MediaPipe pose files .pose (not tracked)
  annotations/         # ELAN .eaf annotation files (not tracked)
  subtitles/           # Subtitle files .vtt (not tracked)
  training/            # Training data (not tracked)
    alignment.csv      #   25,642 rows (14,777 approved segments)
    export/            #   13,638 .npz training segments
    manifest.csv       #   Export manifest with metadata
    splits/            #   train.csv, val.csv, test.csv
  models/              # PyTorch .pt checkpoints (not tracked)
  evaluations/         # Evaluation reports (not tracked)
```

## Data Sources

| Source | Type | Videos | Access |
|--------|------|--------|--------|
| Own SPJ videos | Parallel corpus (SPJ + Slovak subtitles) | ~300 | Private — fully owned |
| Museum content | SPJ interpretation of exhibits | 100h+ | Private — fully owned |
| partner-dictnary videos | Isolated sign vocabulary (320x240, 50fps) | ~10,000 | Private — partner agreement |
| Category vocabulary | Category-organized vocabulary (720p, 25fps) | ~5,200 | Private — partner agreement |
| Sentence videos | Word + sentence pairs | ~420 | Private — partner agreement |
| Partner organization | SPJ videos with subtitles | — | Private — partner agreement |

## Model Checkpoints

| Checkpoint | Classes | Test Top-1 | Description |
|------------|---------|------------|-------------|
| `cat_v2_ep55_acc0.4493.pt` | 102 categories | — | Category classifier (encoder donor) |
| `quality_ep22_acc0.2297.pt` | 516 words | 16.8% | Baseline word model (from scratch) |
| `transfer_cat2word_ep37_acc0.2587.pt` | 516 words | **24.9%** | Best: category→word transfer |
| `ssl_pretrained_compact.pt` | — | — | SSL encoder (masked pose modeling) |
| `ssl_finetune_ep66_acc0.2394.pt` | 516 words | 17.8% | SSL→word transfer |
| `quality2_transfer_ep29_acc0.2259.pt` | 1,743 words | 19.5% | Expanded coverage (2+ samples) |

## Data Access

For research collaboration or data access requests, contact:

**Marek Kanas** — [partner-ngo](https://www.partner-ngo.eu)

Data sharing is subject to partner organization agreements and GDPR compliance.
