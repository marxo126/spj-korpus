"""Video inventory — scan, probe metadata, persist to CSV."""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

import pandas as pd

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def scan_videos(root_dir: Path) -> list[Path]:
    """Recursively find all video files under root_dir."""
    root_dir = Path(root_dir)
    found: list[Path] = []
    for ext in VIDEO_EXTENSIONS:
        found.extend(root_dir.rglob(f"*{ext}"))
    return sorted(found)


def get_video_metadata(path: Path) -> dict:
    """Run ffprobe and return basic video metadata.

    Returns a dict with keys:
        path, filename, duration_sec, width, height, fps,
        file_size_mb, codec
    Returns placeholder values if ffprobe fails.
    """
    path = Path(path)
    result = {
        "path": str(path),
        "filename": path.name,
        "duration_sec": None,
        "width": None,
        "height": None,
        "fps": None,
        "file_size_mb": round(path.stat().st_size / 1_048_576, 2),
        "codec": None,
    }
    try:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                "-show_format",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        data = json.loads(probe.stdout)
        fmt = data.get("format", {})
        result["duration_sec"] = round(float(fmt.get("duration", 0)), 2)

        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                result["width"] = stream.get("width")
                result["height"] = stream.get("height")
                result["codec"] = stream.get("codec_name")
                fps_raw = stream.get("avg_frame_rate", "0/1")
                num, den = (int(x) for x in fps_raw.split("/"))
                result["fps"] = round(num / den, 2) if den else None
                break
    except Exception as exc:
        logger.warning("ffprobe failed for %s: %s", path.name, exc)
    return result


def build_inventory(root_dir: Path, pose_dir: Path | None = None) -> pd.DataFrame:
    """Scan root_dir and build a metadata DataFrame.

    Args:
        root_dir: Directory to scan for video files.
        pose_dir: Directory where .pose files are stored. If None,
                  uses root_dir/../pose relative to root_dir.

    Returns:
        DataFrame with columns:
            path, filename, duration_sec, width, height, fps,
            file_size_mb, codec, pose_extracted
    """
    root_dir = Path(root_dir)
    if pose_dir is None:
        pose_dir = root_dir.parent / "pose"

    videos = scan_videos(root_dir)
    if not videos:
        return pd.DataFrame(
            columns=[
                "path", "filename", "duration_sec", "width", "height",
                "fps", "file_size_mb", "codec", "pose_extracted",
            ]
        )

    rows = []
    for v in videos:
        meta = get_video_metadata(v)
        meta["pose_extracted"] = pose_exists(v, pose_dir)
        rows.append(meta)

    return pd.DataFrame(rows)


def pose_exists_for_row(row: dict, pose_dir: Path) -> bool:
    """Check whether a pose file exists for a DataFrame row."""
    return pose_exists(Path(row["path"]), pose_dir)


def pose_exists(video_path: Path, pose_dir: Path) -> bool:
    """Return True if a non-empty .pose file exists for the given video."""
    stem = Path(video_path).stem
    pose_file = Path(pose_dir) / f"{stem}.pose"
    return pose_file.exists() and pose_file.stat().st_size > 0


def load_or_create_inventory(
    csv_path: Path,
    video_dir: Path,
    pose_dir: Path | None = None,
) -> pd.DataFrame:
    """Load an existing inventory CSV or scan fresh if it doesn't exist.

    Args:
        csv_path: Path to persist the CSV.
        video_dir: Source video directory.
        pose_dir: Pose output directory.

    Returns:
        DataFrame as returned by build_inventory().
    """
    csv_path = Path(csv_path)
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return df
    df = build_inventory(video_dir, pose_dir)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df
