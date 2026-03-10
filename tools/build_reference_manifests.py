"""Build training manifests for reference partner-dictnary sources.

Each reference video = one sign. The compact NPZ files from compact_poses.py
are already the training segments. This script maps NPZ filenames to labels
using the metadata CSVs.

Usage:
    .venv/bin/python tools/build_reference_manifests.py
    .venv/bin/python tools/build_reference_manifests.py --source reference-dict
    .venv/bin/python tools/build_reference_manifests.py --source partner-dict
    .venv/bin/python tools/build_reference_manifests.py --source climate-vocab
    .venv/bin/python tools/build_reference_manifests.py --source art-vocab
"""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"


MANIFEST_FIELDS = ["npz_path", "segment_id", "text", "label", "n_frames", "n_landmarks", "source", "angle"]


def _fix_mojibake(text: str) -> str:
    """Fix latin-1 → iso-8859-2 mojibake from reference-dict metadata."""
    try:
        return text.encode("latin-1").decode("iso-8859-2")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text


def _normalize_label(text: str) -> str:
    """Normalize a label for training: lowercase, strip whitespace."""
    return text.strip().lower()


def _npz_info(npz_path: Path) -> tuple[int, int]:
    """Get (n_frames, n_landmarks) from an NPZ file."""
    try:
        d = np.load(str(npz_path))
        pose = d["pose"]
        return pose.shape[0], pose.shape[1]
    except Exception:
        return 0, 0


def build_reference-dict_manifest() -> Path:
    """Map reference-dict NPZ files to labels from metadata.

    Each reference-dict entry has multiple YouTube videos:
      - 1st = front view (60°), 2nd = side view (90°)
      - 3rd+ = additional variants (regional, dialectal, older forms)

    Same video can appear under multiple entries with different labels
    (same sign, different meanings / synonyms). All label-video pairs are
    included up to MAX_LABELS_PER_VIDEO to filter out generic context videos.
    """
    meta_path = DATA_DIR / "reference-dict_sk_metadata.csv"
    pose_dir = DATA_DIR / "pose" / "reference-dict"
    out_path = DATA_DIR / "training" / "manifest_reference-dict.csv"

    MAX_LABELS_PER_VIDEO = 20  # videos with more labels are generic context, not signs

    # Build mapping: yt_id → list of (label, angle)
    # Collect ALL label assignments per video
    yt_to_entries: dict[str, list[tuple[str, str]]] = {}

    with open(meta_path) as f:
        for row in csv.DictReader(f):
            yt_ids = json.loads(row["youtube_ids"])
            title = row.get("title", row["slug"])
            title = _fix_mojibake(title)
            title = re.sub(r"^Posunok:\s*", "", title, flags=re.IGNORECASE)
            label = _normalize_label(title)

            for idx, yt_id in enumerate(yt_ids):
                angle = "front" if idx == 0 else "side" if idx == 1 else "front"
                if yt_id not in yt_to_entries:
                    yt_to_entries[yt_id] = []
                yt_to_entries[yt_id].append((label, angle))

    # Deduplicate labels per video (same label from multiple entries)
    for yt_id in yt_to_entries:
        seen = set()
        deduped = []
        for label, angle in yt_to_entries[yt_id]:
            if label not in seen:
                seen.add(label)
                deduped.append((label, angle))
        yt_to_entries[yt_id] = deduped

    # Build reverse lookup: for each NPZ file, find which YT ID it contains
    # YT IDs can contain underscores (e.g., 3zRucFtg_-U), so simple split fails.
    # Instead, check if stem ends with any known YT ID.
    all_yt_ids = sorted(yt_to_entries.keys(), key=len, reverse=True)  # longest first

    rows = []
    seen_yt_ids = set()
    unmatched = 0
    duplicates = 0
    skipped_generic = 0

    for npz_path in sorted(pose_dir.glob("*.npz")):
        stem = npz_path.stem
        matched_yt = None
        for yt_id in all_yt_ids:
            if stem.endswith("_" + yt_id):
                matched_yt = yt_id
                break

        if not matched_yt:
            unmatched += 1
            continue

        # Deduplicate: same YouTube video may exist under multiple NPZ filenames
        if matched_yt in seen_yt_ids:
            duplicates += 1
            continue

        seen_yt_ids.add(matched_yt)
        entries = yt_to_entries[matched_yt]

        # Skip generic context videos with too many labels
        if len(entries) > MAX_LABELS_PER_VIDEO:
            skipped_generic += 1
            continue

        # Add one row per label for this video
        for label, angle in entries:
            rows.append({
                "npz_path": str(npz_path),
                "segment_id": stem,
                "text": label,
                "label": label,
                "source": "reference-dict",
                "angle": angle,
            })

    # Add frame/landmark info
    for row in rows:
        n_frames, n_landmarks = _npz_info(Path(row["npz_path"]))
        row["n_frames"] = n_frames
        row["n_landmarks"] = n_landmarks

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    labels = set(r["label"] for r in rows)
    unique_videos = len(seen_yt_ids) - skipped_generic
    print(f"reference-dict: {len(rows)} samples ({unique_videos} unique videos), "
          f"{len(labels)} unique labels, {duplicates} dup NPZ skipped, "
          f"{skipped_generic} generic videos skipped, {unmatched} unmatched")
    return out_path


def build_partner-dict_manifest() -> Path:
    """Map partner-dict NPZ files to labels from metadata."""
    meta_path = DATA_DIR / "partner-dict_metadata.csv"
    pose_dir = DATA_DIR / "pose" / "partner-dict"
    out_path = DATA_DIR / "training" / "manifest_partner-dict.csv"

    # Build lookup: video filename stem → (name, translation)
    lookup = {}
    with open(meta_path) as f:
        for row in csv.DictReader(f):
            videos = json.loads(row["videos"])
            name = row.get("name", "")
            translation = row.get("translation", "")
            for v in videos:
                # v = "videospj/A_word.mp4" or "videospj/B_word.mp4"
                fname = v.split("/")[-1]
                stem = Path(fname).stem  # "A_word" or "B_word"
                # Label: use name if it's not "spj-N" pattern, else translation
                if name and not re.match(r"^spj-\d+$", name):
                    label = _normalize_label(name)
                elif translation:
                    label = _normalize_label(translation)
                else:
                    label = _normalize_label(stem.split("_", 1)[1] if "_" in stem else stem)
                angle = "front" if stem.startswith("A_") else "side" if stem.startswith("B_") else "unknown"
                lookup[stem] = (label, angle, row["id"])

    rows = []
    unmatched = 0
    for npz_path in sorted(pose_dir.glob("*.npz")):
        stem = npz_path.stem
        if stem in lookup:
            label, angle, sign_id = lookup[stem]
            rows.append({
                "npz_path": str(npz_path),
                "segment_id": stem,
                "text": label,
                "label": label,
                "source": "partner-dict",
                "angle": angle,
            })
        else:
            unmatched += 1

    # Add frame/landmark info
    for row in rows:
        n_frames, n_landmarks = _npz_info(Path(row["npz_path"]))
        row["n_frames"] = n_frames
        row["n_landmarks"] = n_landmarks

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    labels = set(r["label"] for r in rows)
    print(f"partner-dict: {len(rows)} samples, {len(labels)} unique labels, {unmatched} unmatched")
    return out_path


def build_climate-vocab_manifest() -> Path:
    """Map climate-vocab NPZ files to labels."""
    meta_path = DATA_DIR / "climate-vocab_metadata.csv"
    pose_dir = DATA_DIR / "pose" / "climate-vocabaction"
    out_path = DATA_DIR / "training" / "manifest_climate-vocab.csv"

    if not meta_path.exists():
        print("climate-vocab: no metadata CSV found")
        return out_path

    lookup = {}
    with open(meta_path) as f:
        for row in csv.DictReader(f):
            stem = Path(row["filename"]).stem
            lookup[stem] = _normalize_label(row["term"])

    rows = []
    if pose_dir.exists():
        for npz_path in sorted(pose_dir.glob("*.npz")):
            stem = npz_path.stem
            if stem in lookup:
                rows.append({
                    "npz_path": str(npz_path),
                    "segment_id": stem,
                    "text": lookup[stem],
                    "label": lookup[stem],
                    "source": "climate-vocab",
                    "angle": "front",
                })

    # Add frame/landmark info
    for row in rows:
        n_frames, n_landmarks = _npz_info(Path(row["npz_path"]))
        row["n_frames"] = n_frames
        row["n_landmarks"] = n_landmarks

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    labels = set(r["label"] for r in rows)
    print(f"climate-vocab: {len(rows)} samples, {len(labels)} unique labels")
    return out_path


def build_vocab_manifest(source_name: str) -> Path:
    """Build manifest for vocab video sources (fin-vocab, career-vocab, climate-vocab)."""
    meta_path = DATA_DIR / "vocab_videos_metadata.csv"
    # Map source names to pose dirs
    pose_dir_map = {
        "fin-vocab": DATA_DIR / "pose" / "fin-vocab",
        "career-vocab": DATA_DIR / "pose" / "career-vocab",
        "climate-vocabaction": DATA_DIR / "pose" / "climate-vocabaction",
    }
    pose_dir = pose_dir_map.get(source_name)
    out_path = DATA_DIR / "training" / f"manifest_{source_name}.csv"

    if not meta_path.exists() or not pose_dir or not pose_dir.exists():
        print(f"{source_name}: no data found")
        return out_path

    # Build lookup from metadata
    lookup = {}
    with open(meta_path) as f:
        for row in csv.DictReader(f):
            if row["source"] == source_name:
                stem = Path(row["filename"]).stem
                lookup[stem] = _normalize_label(row["label"])

    rows = []
    for npz_path in sorted(pose_dir.glob("*.npz")):
        stem = npz_path.stem
        if stem in lookup:
            rows.append({
                "npz_path": str(npz_path),
                "segment_id": stem,
                "text": lookup[stem],
                "label": lookup[stem],
                "source": source_name,
                "angle": "front",
            })

    for row in rows:
        n_frames, n_landmarks = _npz_info(Path(row["npz_path"]))
        row["n_frames"] = n_frames
        row["n_landmarks"] = n_landmarks

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    labels = set(r["label"] for r in rows)
    print(f"{source_name}: {len(rows)} samples, {len(labels)} unique labels")
    return out_path


def build_art-vocab_manifest() -> Path:
    """Map art-vocab pose files to labels.

    art-vocab videos: {wp_id}_{wd|st}_{60|90}_{yt_id}.mp4
    Poses are in the main pose dir (not a subdir).
    We only use 'wd' (word) videos, not 'st' (sentence).
    """
    meta_path = DATA_DIR / "art-vocab_metadata.csv"
    video_dir = DATA_DIR / "videos" / "art-vocab"
    pose_dir = DATA_DIR / "pose"  # art-vocab poses in main pose dir
    out_path = DATA_DIR / "training" / "manifest_art-vocab.csv"

    if not meta_path.exists():
        print("art-vocab: no metadata CSV found")
        return out_path

    # Build lookup: wp_id → word
    lookup = {}
    with open(meta_path) as f:
        for row in csv.DictReader(f):
            lookup[row["wp_id"]] = _normalize_label(row["word"])

    # Find word videos and match to poses
    rows = []
    if video_dir.exists():
        for vid in sorted(video_dir.glob("*_wd_*.mp4")):
            stem = vid.stem
            parts = stem.split("_")
            wp_id = parts[0]
            if wp_id not in lookup:
                continue

            # Check for pose (could be .pose or .npz)
            pose_npz = pose_dir / f"{stem}.npz"
            pose_file = pose_dir / f"{stem}.pose"

            if pose_npz.exists():
                npz_path = pose_npz
            elif pose_file.exists():
                # Compact it
                npz_path = pose_npz  # will create below
            else:
                continue

            angle = "front" if "_60_" in stem else "side" if "_90_" in stem else "unknown"
            rows.append({
                "npz_path": str(npz_path),
                "segment_id": stem,
                "text": lookup[wp_id],
                "label": lookup[wp_id],
                "source": "art-vocab",
                "angle": angle,
            })

    # Add frame/landmark info (skip if npz doesn't exist yet)
    for row in rows:
        p = Path(row["npz_path"])
        if p.exists():
            n_frames, n_landmarks = _npz_info(p)
        else:
            n_frames, n_landmarks = 0, 0
        row["n_frames"] = n_frames
        row["n_landmarks"] = n_landmarks

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    labels = set(r["label"] for r in rows)
    print(f"art-vocab: {len(rows)} samples, {len(labels)} unique labels")
    return out_path


def main():
    source_filter = None
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--source" and i < len(sys.argv) - 1:
            source_filter = sys.argv[i + 1]
        elif arg.startswith("--source="):
            source_filter = arg.split("=", 1)[1]

    builders = {
        "reference-dict": build_reference-dict_manifest,
        "partner-dict": build_partner-dict_manifest,
        "climate-vocab": build_climate-vocab_manifest,
        "art-vocab": build_art-vocab_manifest,
        "fin-vocab": lambda: build_vocab_manifest("fin-vocab"),
        "career-vocab": lambda: build_vocab_manifest("career-vocab"),
    }

    if source_filter:
        if source_filter in builders:
            builders[source_filter]()
        else:
            print(f"Unknown source: {source_filter}")
    else:
        for name, builder in builders.items():
            builder()


if __name__ == "__main__":
    main()
