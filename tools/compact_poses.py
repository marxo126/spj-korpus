"""Convert .pose files to compact .npz (96 landmarks) and delete originals.

Runs alongside extract_and_cleanup.py to prevent disk space exhaustion.
Pose files average ~1 MB; compact NPZ averages ~200 KB (5x smaller).

Usage:
    .venv/bin/python tools/compact_poses.py --source reference-dict
    .venv/bin/python tools/compact_poses.py --source reference-dict --loop  # repeat every 60s
    .venv/bin/python tools/compact_poses.py --source partner-dict
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

DATA_DIR = Path(__file__).parent.parent / "data"

SOURCES = {
    "partner-dict": DATA_DIR / "pose" / "partner-dict",
    "reference-dict": DATA_DIR / "pose" / "reference-dict",
    "climate-vocab": DATA_DIR / "pose" / "climate-vocabaction",
    "career-vocab": DATA_DIR / "pose" / "career-vocab",
    "fin-vocab": DATA_DIR / "pose" / "fin-vocab",
}


def compact_poses(pose_dir: Path) -> tuple[int, int]:
    """Convert .pose → .npz (compact 96 landmarks), delete .pose.

    Returns (converted, skipped).
    """
    from spj.preannotate import load_pose_arrays
    from spj.training_data import SL_LANDMARK_INDICES

    converted = 0
    skipped = 0
    idx = np.array(SL_LANDMARK_INDICES)

    for pose_path in sorted(pose_dir.glob("*.pose")):
        npz_path = pose_path.with_suffix(".npz")
        if npz_path.exists():
            # Already converted — just delete the .pose
            pose_path.unlink()
            skipped += 1
            continue

        try:
            data, conf, fps = load_pose_arrays(pose_path)
            # data: (T, 1, 543, 3), conf: (T, 1, 543, 1)
            # Extract person dim, filter landmarks
            pose = data[:, 0, idx, :].astype(np.float16)  # (T, 96, 3)
            c = conf[:, 0, idx, :].astype(np.float16)     # (T, 96, 1)

            np.savez_compressed(npz_path, pose=pose, confidence=c, fps=np.float32(fps))
            pose_path.unlink()
            converted += 1
        except Exception as e:
            print(f"  ERROR {pose_path.name}: {e}")
            # Don't delete broken pose files
            continue

    return converted, skipped


def main():
    source = "reference-dict"
    loop = "--loop" in sys.argv
    for arg in sys.argv[1:]:
        if arg.startswith("--source"):
            idx = sys.argv.index(arg)
            if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("--"):
                source = sys.argv[idx + 1]
        if arg.startswith("--source="):
            source = arg.split("=", 1)[1]

    if source not in SOURCES:
        print(f"Unknown source: {source}")
        return

    pose_dir = SOURCES[source]
    print(f"Compacting {source} poses in {pose_dir}")

    while True:
        converted, skipped = compact_poses(pose_dir)
        npz_count = len(list(pose_dir.glob("*.npz")))
        pose_count = len(list(pose_dir.glob("*.pose")))
        print(f"  Converted: {converted}, Skipped (already npz): {skipped}, "
              f"Total NPZ: {npz_count}, Remaining .pose: {pose_count}")

        if not loop:
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
