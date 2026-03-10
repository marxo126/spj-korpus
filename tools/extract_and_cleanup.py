"""Extract poses from reference videos, then delete videos to save disk space.

Processes partner-dict and reference-dict videos: extract pose → verify → delete video.
Uses Metal GPU for fast extraction.

Usage:
    .venv/bin/python tools/extract_and_cleanup.py
    .venv/bin/python tools/extract_and_cleanup.py --source partner-dict
    .venv/bin/python tools/extract_and_cleanup.py --source reference-dict
    .venv/bin/python tools/extract_and_cleanup.py --dry-run
    .venv/bin/python tools/extract_and_cleanup.py --keep-videos
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

DATA_DIR = Path(__file__).parent.parent / "data"

SOURCES = {
    "partner-dict": {
        "video_dir": DATA_DIR / "reference" / "partner-dict" / "videos",
        "pose_dir": DATA_DIR / "pose" / "partner-dict",
    },
    "reference-dict": {
        "video_dir": DATA_DIR / "reference" / "reference-dict_sk",
        "pose_dir": DATA_DIR / "pose" / "reference-dict",
    },
}

# Metal GPU: 2 threads for HD, 8 for small videos (CLAUDE.md rule)
GPU_THREADS = 8  # small videos (320-480p) safe at 8; reduce to 2 for HD (>720p)
BATCH_SIZE = 50  # process in batches, delete after each


def process_source(source: str, dry_run: bool = False, keep_videos: bool = False):
    cfg = SOURCES[source]
    video_dir = cfg["video_dir"]
    pose_dir = cfg["pose_dir"]
    pose_dir.mkdir(parents=True, exist_ok=True)

    # Find videos that need pose extraction
    videos = sorted(video_dir.glob("*.mp4"))
    already_done = set()
    for p in pose_dir.glob("*.pose"):
        already_done.add(p.stem)

    to_process = [v for v in videos if v.stem not in already_done]
    skipped = len(videos) - len(to_process)

    print(f"\n{'='*60}")
    print(f"  {source}: {len(videos)} videos, {skipped} already extracted")
    print(f"  To process: {len(to_process)}")
    print(f"{'='*60}")

    if not to_process:
        if not keep_videos:
            # Delete videos that already have poses
            deleted = _cleanup_videos(video_dir, pose_dir, dry_run)
            print(f"  Cleaned up {deleted} videos (poses already exist)")
        return

    from spj.pose import extract_pose_batch

    extracted = 0
    failed = []
    deleted_count = 0

    for batch_start in range(0, len(to_process), BATCH_SIZE):
        batch = to_process[batch_start:batch_start + BATCH_SIZE]
        if dry_run:
            for v in batch:
                print(f"  [DRY] {v.name}")
            continue

        # Extract poses for batch
        video_list = [v for v in batch]

        try:
            extract_pose_batch(
                video_list, pose_dir,
                n_workers=GPU_THREADS,
                use_gpu=True,
            )
        except Exception as e:
            print(f"  Batch error: {e}")

        # Verify and cleanup
        for video_path in batch:
            pose_path = pose_dir / f"{video_path.stem}.pose"
            if pose_path.exists() and pose_path.stat().st_size > 0:
                extracted += 1
                if not keep_videos:
                    video_path.unlink()
                    deleted_count += 1
            else:
                failed.append(video_path.name)
                if pose_path.exists() and pose_path.stat().st_size == 0:
                    pose_path.unlink()

        done = min(batch_start + BATCH_SIZE, len(to_process))
        print(f"  Processed {done}/{len(to_process)} "
              f"(extracted: {extracted}, failed: {len(failed)}, "
              f"deleted videos: {deleted_count})", flush=True)

    # Final cleanup: delete videos that have valid poses
    if not keep_videos and not dry_run:
        extra_deleted = _cleanup_videos(video_dir, pose_dir, dry_run=False)
        deleted_count += extra_deleted

    print(f"\n  Done: extracted={extracted}, failed={len(failed)}, "
          f"videos deleted={deleted_count}")
    if failed:
        for f in failed[:10]:
            print(f"    FAIL: {f}")


def _cleanup_videos(video_dir: Path, pose_dir: Path, dry_run: bool) -> int:
    """Delete videos that have valid corresponding pose files."""
    deleted = 0
    for v in video_dir.glob("*.mp4"):
        pose = pose_dir / f"{v.stem}.pose"
        if pose.exists() and pose.stat().st_size > 0:
            if not dry_run:
                v.unlink()
            deleted += 1
    return deleted


def main():
    dry_run = "--dry-run" in sys.argv
    keep_videos = "--keep-videos" in sys.argv
    source_filter = None
    for arg in sys.argv[1:]:
        if arg.startswith("--source"):
            idx = sys.argv.index(arg)
            if idx + 1 < len(sys.argv):
                source_filter = sys.argv[idx + 1]

    # Also check --source=X format
    for arg in sys.argv[1:]:
        if arg.startswith("--source="):
            source_filter = arg.split("=", 1)[1]

    sources = [source_filter] if source_filter else list(SOURCES.keys())

    import shutil
    _, _, free = shutil.disk_usage("/")
    print(f"Disk free: {free / 1e9:.1f} GB")

    for source in sources:
        if source not in SOURCES:
            print(f"Unknown source: {source}")
            continue
        process_source(source, dry_run=dry_run, keep_videos=keep_videos)

    _, _, free_after = shutil.disk_usage("/")
    print(f"\nDisk free after: {free_after / 1e9:.1f} GB "
          f"(freed {(free_after - free) / 1e9:.1f} GB)")


if __name__ == "__main__":
    main()
