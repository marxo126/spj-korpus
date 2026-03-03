# Data Directory

**Video data is not included in this repository.**

The corpus videos and related data belong to partner organizations and are used under private agreements for ML training and research purposes only.

## Directory Structure

```
data/
  videos/          # Source videos (not tracked)
  pose/            # MediaPipe pose files .pose (not tracked)
  annotations/     # ELAN .eaf annotation files (not tracked)
  subtitles/       # Subtitle files .vtt (not tracked)
  training/        # Training splits and .npz exports (not tracked)
  models/          # PyTorch .pt checkpoints (not tracked)
  evaluations/     # Evaluation reports (not tracked)
```

## Data Sources

| Source | Type | Access |
|--------|------|--------|
| Own SPJ videos (~300) | Parallel corpus (SPJ + Slovak subtitles) | Private — fully owned |
| Museum content (100h+) | SPJ interpretation of exhibits | Private — fully owned |
| [partner organization] | SPJ videos with subtitles | Private — partner agreement |
| partner-ngo.eu | SPJ videos with subtitles | Private — partner agreement |

## Data Access

For research collaboration or data access requests, contact:

**Marek Kanas** — [partner-ngo](https://www.partner-ngo.eu)

Data sharing is subject to partner organization agreements and GDPR compliance.
