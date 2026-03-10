"""Organize top-level pose files into source subdirectories.

Matches each .pose/.npz file against known video sources to determine
the correct subdirectory. Moves files, does not copy.

Usage:
    .venv/bin/python tools/organize_poses.py --dry-run   # preview only
    .venv/bin/python tools/organize_poses.py              # actually move
"""
import os
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
POSE_DIR = DATA_DIR / "pose"

# Video source directories to match against
VIDEO_SOURCES = {
    "intl-vocab": DATA_DIR / "videos" / "intl-vocab",
    "category-source": None,  # special handling (subdirs)
    "partner-ngo_fb": DATA_DIR / "videos" / "partner-ngo_fb",
    "art-vocab": DATA_DIR / "videos" / "art-vocab",
    "partner-org": DATA_DIR / "videos" / "partner-org",
    "downloaded": DATA_DIR / "videos" / "downloaded",
    "partner-ngo": DATA_DIR / "videos" / "partner-ngo",
    "other_spj": DATA_DIR / "videos" / "other_spj",
    "spravy_spj": DATA_DIR / "videos" / "spravy_spj",
    "vhs_archive": DATA_DIR / "videos" / "vhs_archive",
}


def _build_video_index() -> dict[str, str]:
    """Build stem → source mapping from all video directories."""
    index: dict[str, str] = {}

    for source, vdir in VIDEO_SOURCES.items():
        if source == "category-source":
            # category-source has subdirs (categories)
            kdir = DATA_DIR / "videos" / "category-source"
            if not kdir.exists():
                continue
            for subdir in kdir.iterdir():
                if not subdir.is_dir():
                    continue
                for f in subdir.iterdir():
                    if f.suffix == ".mp4":
                        index[f.stem] = "category-source"
            continue

        if vdir is None or not vdir.exists():
            continue
        for f in vdir.iterdir():
            if f.suffix == ".mp4":
                index[f.stem] = source


    # Vocab video sources (already have pose subdirs, but some .pose files are top-level)
    for vname in ("career-vocab", "climate-vocabaction", "fin-vocab"):
        vdir = DATA_DIR / "videos" / vname
        if not vdir.exists():
            continue
        for f in vdir.iterdir():
            if f.suffix == ".mp4":
                index[f.stem] = vname

    return index


def _match_source(stem: str, video_index: dict[str, str]) -> str | None:
    """Determine source for a pose file stem."""
    # Direct match
    if stem in video_index:
        return video_index[stem]

    # _60/_90 suffix: try base name
    for suffix in ("_60", "_90"):
        if stem.endswith(suffix):
            base = stem[: -len(suffix)]
            if base in video_index:
                return video_index[base]

    # SPJ corpus pattern: YYYYMMDD_*
    if len(stem) >= 8 and stem[:8].isdigit() and (len(stem) == 8 or stem[8] == "_"):
        year = int(stem[:4])
        if 2010 <= year <= 2030:
            return "spj_corpus"

    return None


def main():
    dry_run = "--dry-run" in sys.argv

    print("Building video index...")
    video_index = _build_video_index()
    print(f"  {len(video_index)} video stems indexed")

    # Collect top-level pose/npz files
    top_files = [
        f for f in POSE_DIR.iterdir()
        if f.is_file() and f.suffix in (".pose", ".npz")
    ]
    print(f"  {len(top_files)} top-level files to organize")

    # Classify
    moves: dict[str, list[Path]] = {}
    unmatched: list[Path] = []

    for f in top_files:
        source = _match_source(f.stem, video_index)
        if source:
            moves.setdefault(source, []).append(f)
        else:
            unmatched.append(f)

    # Report
    print(f"\n{'=' * 50}")
    print(f"{'Source':<25} {'Count':>8}")
    print(f"{'-' * 25} {'-' * 8}")
    for source in sorted(moves.keys()):
        print(f"{source:<25} {len(moves[source]):>8}")
    print(f"{'unmatched':<25} {len(unmatched):>8}")
    total_matched = sum(len(v) for v in moves.values())
    print(f"{'-' * 25} {'-' * 8}")
    print(f"{'TOTAL':<25} {total_matched + len(unmatched):>8}")

    if unmatched:
        print(f"\nUnmatched samples (first 20):")
        for f in sorted(unmatched, key=lambda x: x.name)[:20]:
            print(f"  {f.name}")

    if dry_run:
        print("\n  --dry-run: no files moved")
        return

    # Move files
    print(f"\nMoving files...")
    moved = 0
    for source, files in moves.items():
        dest_dir = POSE_DIR / source
        dest_dir.mkdir(exist_ok=True)
        for f in files:
            dest = dest_dir / f.name
            if dest.exists():
                print(f"  SKIP (exists): {f.name} → {source}/")
                continue
            f.rename(dest)
            moved += 1
        print(f"  {source}/: {len(files)} files")

    print(f"\nDone: {moved} files moved")

    if unmatched:
        print(f"\n{len(unmatched)} unmatched files remain in pose/")


if __name__ == "__main__":
    main()
