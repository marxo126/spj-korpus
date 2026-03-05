#!/usr/bin/env python3
"""Split Kodifikacia dual-view videos and extract poses for each half.

Kodifikacia videos (1280x720 or 1280x556) contain two camera angles:
  - Left half  (cols 0-631):   ~60 deg frontal view
  - Right half (cols 648-1279): ~90 deg profile view
  - Dark divider at cols 632-647

This script reads inventory.csv, filters to source=kodifikacia, checks which
videos already have _60.pose and _90.pose files, and extracts dual-view poses
for the rest.

Usage:
    .venv/bin/python tools/split_kodifikacia_poses.py
    .venv/bin/python tools/split_kodifikacia_poses.py --workers 4 --backend gpu
    .venv/bin/python tools/split_kodifikacia_poses.py --dry-run
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DATA_DIR = ROOT / "data"
POSE_DIR = DATA_DIR / "pose"
INVENTORY = DATA_DIR / "inventory.csv"

# Crop bounds for 1280-wide dual-view videos
CROP_LEFT = (0, 0, 632, 720)     # x, y, w, h — height adjusted per video
CROP_RIGHT = (648, 0, 632, 720)
SUFFIX_LEFT = "_60"
SUFFIX_RIGHT = "_90"


def main():
    parser = argparse.ArgumentParser(
        description="Extract dual-view poses from Kodifikacia videos",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel workers (default: auto)",
    )
    parser.add_argument(
        "--backend", choices=["gpu", "cpu"], default="gpu",
        help="Extraction backend (default: gpu = Metal GPU)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be processed without extracting",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process at most N videos (useful for testing)",
    )
    args = parser.parse_args()

    # Load inventory
    import pandas as pd

    if not INVENTORY.exists():
        print(f"ERROR: inventory not found at {INVENTORY}")
        sys.exit(1)

    df = pd.read_csv(INVENTORY)
    # Filter to kodifikacia — identified by 'kodifikacia' in the path
    if "source" in df.columns:
        kod = df[df["source"].str.lower() == "kodifikacia"].copy()
    elif "path" in df.columns:
        kod = df[df["path"].str.contains("kodifikac", case=False, na=False)].copy()
    else:
        print("ERROR: inventory.csv has neither 'source' nor 'path' column")
        sys.exit(1)

    print(f"Kodifikacia videos in inventory: {len(kod)}")

    if kod.empty:
        print("No Kodifikacia videos found. Nothing to do.")
        return

    # Determine video paths — inventory stores absolute paths
    if "path" in kod.columns:
        video_paths = [Path(str(p)) for p in kod["path"] if pd.notna(p)]
    elif "filename" in kod.columns:
        video_dir = DATA_DIR / "videos" / "kodifikacia"
        video_paths = [video_dir / str(f) for f in kod["filename"] if pd.notna(f)]
    else:
        print("ERROR: inventory has neither 'path' nor 'filename' column")
        sys.exit(1)

    # Filter to existing videos
    existing = [p for p in video_paths if p.exists()]
    print(f"Videos found on disk: {len(existing)} / {len(video_paths)}")

    # Check which already have both _60.pose and _90.pose
    POSE_DIR.mkdir(parents=True, exist_ok=True)
    todo = []
    for vpath in existing:
        stem = vpath.stem
        pose_left = POSE_DIR / f"{stem}{SUFFIX_LEFT}.pose"
        pose_right = POSE_DIR / f"{stem}{SUFFIX_RIGHT}.pose"
        if pose_left.exists() and pose_right.exists():
            # Both exist — check they're non-empty
            if pose_left.stat().st_size > 100 and pose_right.stat().st_size > 100:
                continue
        todo.append(vpath)

    print(f"Already extracted: {len(existing) - len(todo)}")
    print(f"Remaining to extract: {len(todo)}")

    if args.limit:
        todo = todo[:args.limit]
        print(f"Limited to: {len(todo)} videos")

    if not todo:
        print("Nothing to do.")
        return

    if args.dry_run:
        print("\n--- DRY RUN ---")
        for vpath in todo:
            print(f"  {vpath.name}")
        print(f"\nWould extract {len(todo)} videos × 2 views = {len(todo) * 2} pose files")
        return

    # Import and run extraction
    from spj.pose import extract_pose_dual_view_batch

    use_gpu = args.backend == "gpu"
    print(f"\nStarting extraction: backend={args.backend}, workers={args.workers or 'auto'}")
    t0 = time.time()

    results = extract_pose_dual_view_batch(
        video_paths=todo,
        output_dir=POSE_DIR,
        model_dir=None,
        n_workers=args.workers,
        use_gpu=use_gpu,
        crop_left=CROP_LEFT,
        crop_right=CROP_RIGHT,
        suffix_left=SUFFIX_LEFT,
        suffix_right=SUFFIX_RIGHT,
    )

    elapsed = time.time() - t0

    # Summarize results
    n_success = 0
    n_error = 0
    for res in results:
        if isinstance(res, list):
            for r in res:
                if "error" in r:
                    n_error += 1
                    print(f"  ERROR: {r.get('path', '?')}: {r['error']}")
                else:
                    n_success += 1
        elif isinstance(res, dict):
            if "error" in res:
                n_error += 1
                print(f"  ERROR: {res.get('path', '?')}: {res['error']}")
            else:
                n_success += 1

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Success: {n_success} pose files")
    print(f"  Errors:  {n_error}")
    print(f"  Output:  {POSE_DIR}")


if __name__ == "__main__":
    main()
