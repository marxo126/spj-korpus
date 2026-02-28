"""YouTube video downloader using yt-dlp.

Downloads videos + subtitles + metadata (title, description, uploader, etc.)
into the corpus video directory.

Usage:
    from spj.downloader import download_url, list_playlist

Licensing note (from CLAUDE.md):
    [partner organization] and InnoSign.eu are approved for ML training
    but written agreements are PENDING — flag before corpus deposit/publication.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Subtitle languages to attempt (in priority order)
SUBTITLE_LANGS = ["sk", "cs", "en"]


def _build_ydl_opts(
    output_dir: Path,
    progress_hook: Optional[Callable[[dict], None]] = None,
) -> dict:
    """Build yt-dlp options dict for corpus downloading."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output template: <upload_date>_<video_id>_<sanitized_title>.<ext>
    outtmpl = str(output_dir / "%(upload_date>%Y%m%d)s_%(id)s_%(title).80B.%(ext)s")

    opts: dict = {
        # Video quality: best single-file mp4 first, then best overall
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": outtmpl,
        # Subtitles — manual first, then auto-generated
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": SUBTITLE_LANGS,
        "subtitlesformat": "vtt/srt/best",
        # Metadata sidecar files
        "writeinfojson": True,       # full .info.json with all metadata
        "writedescription": True,    # .description text file
        "writethumbnail": False,     # thumbnails not needed for corpus
        # Post-processing
        "postprocessors": [
            {
                # Embed subs into mkv if mp4 embed fails
                "key": "FFmpegSubtitlesConvertor",
                "format": "vtt",
            },
        ],
        # Misc
        "ignoreerrors": True,        # skip unavailable videos in playlists
        "noplaylist": False,         # allow playlists
        "quiet": True,
        "no_warnings": False,
        "logger": logger,
    }

    if progress_hook is not None:
        opts["progress_hooks"] = [progress_hook]

    return opts


def list_playlist(url: str) -> list[dict]:
    """Fetch metadata for all entries in a playlist (no download).

    Args:
        url: YouTube playlist or channel URL.

    Returns:
        List of dicts, each with keys: id, title, url, duration, upload_date, uploader.
        Returns a single-item list for non-playlist URLs.
    """
    import yt_dlp

    opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": "in_playlist",
        "skip_download": True,
        "logger": logger,
    }
    entries: list[dict] = []
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
        if info is None:
            return []
        if "entries" in info:
            for e in info["entries"]:
                if e is None:
                    continue
                entries.append({
                    "id": e.get("id", ""),
                    "title": e.get("title", "(unknown)"),
                    "url": e.get("url") or f"https://www.youtube.com/watch?v={e.get('id', '')}",
                    "duration": e.get("duration"),
                    "upload_date": e.get("upload_date", ""),
                    "uploader": e.get("uploader", ""),
                })
        else:
            entries.append({
                "id": info.get("id", ""),
                "title": info.get("title", "(unknown)"),
                "url": url,
                "duration": info.get("duration"),
                "upload_date": info.get("upload_date", ""),
                "uploader": info.get("uploader", ""),
            })
    return entries


def download_url(
    url: str,
    output_dir: Path,
    progress_hook: Optional[Callable[[dict], None]] = None,
) -> list[dict]:
    """Download one video or an entire playlist.

    Downloads video file, subtitles (.vtt), and metadata (.info.json +
    .description) into output_dir.

    Args:
        url: YouTube video or playlist URL.
        output_dir: Directory to save files into (created if missing).
        progress_hook: Optional yt-dlp progress hook callable.
                       Receives a dict with keys: status, filename,
                       downloaded_bytes, total_bytes, speed, eta, etc.

    Returns:
        List of result dicts, one per downloaded video:
            id, title, description, uploader, upload_date,
            video_path, subtitle_paths, info_json_path, source_url
    """
    import yt_dlp

    output_dir = Path(output_dir)
    opts = _build_ydl_opts(output_dir, progress_hook)

    results: list[dict] = []

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if info is None:
            return results

        entries = info.get("entries") or [info]
        for entry in entries:
            if entry is None:
                continue
            result = _build_result(entry, output_dir)
            results.append(result)

    return results


def _build_result(entry: dict, output_dir: Path) -> dict:
    """Extract useful fields from a yt-dlp info dict entry."""
    video_id = entry.get("id", "")
    upload_date = entry.get("upload_date", "")

    # Find downloaded files by scanning output_dir for this video id
    video_path: Optional[str] = None
    subtitle_paths: list[str] = []
    info_json_path: Optional[str] = None

    for f in output_dir.iterdir():
        name = f.name
        if video_id not in name:
            continue
        if f.suffix in {".mp4", ".mkv", ".webm", ".mov"}:
            video_path = str(f)
        elif f.suffix in {".vtt", ".srt", ".ass"}:
            subtitle_paths.append(str(f))
        elif name.endswith(".info.json"):
            info_json_path = str(f)

    return {
        "id": video_id,
        "title": entry.get("title", "(unknown)"),
        "description": (entry.get("description") or "")[:500],  # truncate for display
        "uploader": entry.get("uploader", ""),
        "upload_date": upload_date,
        "duration_sec": entry.get("duration"),
        "source_url": entry.get("webpage_url") or entry.get("url", ""),
        "video_path": video_path,
        "subtitle_paths": subtitle_paths,
        "info_json_path": info_json_path,
    }


def read_info_json(json_path: Path) -> dict:
    """Load a yt-dlp .info.json sidecar file.

    Returns an empty dict if the file is missing or malformed.
    """
    try:
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}
