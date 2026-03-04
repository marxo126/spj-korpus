"""Multi-source video downloader for sign language corpus building.

Supported download sources:
  - YouTube (single videos, playlists, channels) via yt-dlp
  - partner-dictnary CSV (bulk individual sign videos) via yt-dlp
  - FTP servers (directory listing + parallel download) via curl
  - HTTP/direct URL download

Usage:
    from spj.downloader import download_url, list_playlist
    from spj.downloader import load_partner-dictnary_csv, download_partner-dictnary_batch
    from spj.downloader import list_ftp_directory, download_ftp_directory
    from spj.downloader import download_http_file, download_http_batch

Licensing note:
    All training data requires written agreements from data owners.
    Verify you have permission before downloading any source.
    Flag all content before corpus deposit or publication.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Subtitle languages to attempt (in priority order)
SUBTITLE_LANGS = ["sk", "cs", "en"]

# Video file extensions recognized across all download backends
VIDEO_EXTENSIONS = (".mp4", ".mov", ".webm", ".mkv")

# Download result status constants (see CLAUDE.md lesson #9)
DL_OK = "ok"
DL_SKIP = "skip"
DL_FAIL = "fail"

# ================================================================== #
#  Agreement / licensing check
# ================================================================== #

_AGREEMENT_FILENAME = "AGREEMENT.txt"


def check_agreement(source_dir: Path) -> bool:
    """Check if a data source directory has a written agreement marker.

    Looks for an AGREEMENT.txt file in the source directory.
    This is a simple convention: place AGREEMENT.txt (with any content
    describing the agreement terms) into the data directory to confirm
    that a written agreement exists for that source.

    Args:
        source_dir: Directory where downloaded data will be stored.

    Returns:
        True if AGREEMENT.txt exists in source_dir.
    """
    return (Path(source_dir) / _AGREEMENT_FILENAME).exists()


def agreement_warning(source_name: str) -> str:
    """Return a standard licensing warning string for a data source."""
    return (
        f"No written agreement found for '{source_name}'. "
        f"Place {_AGREEMENT_FILENAME} in the download directory to confirm "
        f"you have a written agreement from the data owner. "
        f"All training data requires explicit written permission."
    )


# ================================================================== #
#  Shared helpers
# ================================================================== #


def _curl_download(
    url: str,
    output_path: Path,
    extra_args: list[str] | None = None,
    timeout: int = 120,
    skip_existing: bool = True,
) -> dict:
    """Download a file via curl (shared by FTP and HTTP backends).

    Args:
        url: URL to download.
        output_path: Local path to save the file.
        extra_args: Additional curl arguments (e.g., ["--user", "u:p"]).
        timeout: Download timeout in seconds.
        skip_existing: Skip if file already exists and is non-empty.

    Returns:
        Dict with keys: status (DL_OK/DL_SKIP/DL_FAIL), file, size_bytes, error.
    """
    output_path = Path(output_path)
    rel_name = output_path.name

    if skip_existing and output_path.exists():
        sz = output_path.stat().st_size
        if sz > 0:
            return {"status": DL_SKIP, "file": rel_name, "size_bytes": sz}

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["curl", "-s", "-o", str(output_path), "--max-time", str(timeout)]
    if extra_args:
        cmd.extend(extra_args)
    cmd.append(url)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and output_path.exists():
        sz = output_path.stat().st_size
        if sz > 0:
            return {"status": DL_OK, "file": rel_name, "size_bytes": sz}

    return {
        "status": DL_FAIL,
        "file": rel_name,
        "size_bytes": 0,
        "error": result.stderr[:200] if result.stderr else f"curl exit code {result.returncode}",
    }


def _run_batch(
    items: list,
    worker_fn: Callable,
    n_workers: int = 4,
    progress_callback: Optional[Callable[[int, int, dict], None]] = None,
) -> list[dict]:
    """Run download tasks in parallel with progress tracking.

    Args:
        items: List of arguments to pass to worker_fn (each item is a single arg).
        worker_fn: Callable that takes one item and returns a result dict.
        n_workers: Number of parallel threads.
        progress_callback: Called with (done_count, total, result_dict).

    Returns:
        List of result dicts.
    """
    total = len(items)
    results: list[dict] = []
    done_count = 0

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(worker_fn, item): item for item in items}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            done_count += 1
            if progress_callback is not None:
                progress_callback(done_count, total, result)

    return results


# ================================================================== #
#  YouTube download (via yt-dlp)
# ================================================================== #


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


# ================================================================== #
#  partner-dictnary CSV download (bulk individual sign videos)
# ================================================================== #


def _sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a filename.

    Normalizes unicode, removes non-ASCII, replaces spaces/special chars
    with underscores, and truncates to 80 characters.
    """
    if pd.isna(name):
        return ""
    name = str(name).strip()
    # Normalize unicode (e.g., accented chars → base + combining → keep base)
    name = unicodedata.normalize("NFKD", name)
    # Keep only ASCII alphanumeric + basic punctuation
    name = re.sub(r"[^\w\s\-.]", "", name, flags=re.ASCII)
    name = re.sub(r"[\s]+", "_", name)
    name = name.strip("_.")
    return name[:80]


def load_partner-dictnary_csv(csv_path: Path) -> pd.DataFrame:
    """Load a partner-dictnary export CSV for bulk video download.

    Expected CSV columns (flexible — uses what's available):
      - word_id: unique identifier for each entry
      - translation: word/phrase in the target spoken language
      - video_url: URL to the sign video
      - word_class: grammatical category (Noun, Verb, etc.)

    Args:
        csv_path: Path to the partner-dictnary CSV export.

    Returns:
        DataFrame with standardized columns.
    """
    df = pd.read_csv(csv_path)

    # Normalize column names (some exports use different capitalization)
    col_map = {}
    for col in df.columns:
        lower = col.lower().strip()
        if lower in ("wordid", "word_id", "id"):
            col_map[col] = "word_id"
        elif lower in ("translation", "word", "gloss"):
            col_map[col] = "translation"
        elif lower in ("videourl", "video_url", "url", "video"):
            col_map[col] = "video_url"
        elif lower in ("wordclass", "word_class", "pos", "class"):
            col_map[col] = "word_class"

    if col_map:
        df = df.rename(columns=col_map)

    return df


def download_partner-dictnary_video(
    row: pd.Series,
    output_dir: Path,
    yt_dlp_cmd: str = "yt-dlp",
    skip_existing: bool = True,
) -> dict:
    """Download a single partner-dictnary video entry.

    Args:
        row: DataFrame row with word_id, translation, video_url columns.
        output_dir: Directory to save the video into.
        yt_dlp_cmd: Path to yt-dlp binary.
        skip_existing: Skip if output file already exists.

    Returns:
        Dict with keys: status (DL_OK/DL_SKIP/DL_FAIL), word_id, translation, path, error.
    """
    word_id = row.get("word_id", "")
    translation_raw = row.get("translation", "")
    translation = "" if pd.isna(translation_raw) else str(translation_raw).strip()
    video_url = row.get("video_url", "")

    base = {"word_id": word_id, "translation": translation}

    if pd.isna(video_url) or not str(video_url).strip():
        return {**base, "status": DL_FAIL, "path": None, "error": "No video URL"}

    safe_name = _sanitize_filename(translation_raw)
    filename = f"{word_id}_{safe_name}.mp4" if safe_name else f"{word_id}.mp4"
    out_path = Path(output_dir) / filename

    if skip_existing and out_path.exists():
        sz = out_path.stat().st_size
        if sz > 0:
            return {**base, "status": DL_SKIP, "path": str(out_path), "error": None}

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        proc = subprocess.run(
            [yt_dlp_cmd,
             "--format", "best[ext=mp4]/best",
             "--output", str(out_path),
             "--no-playlist",
             "--quiet",
             "--no-warnings",
             str(video_url)],
            capture_output=True, text=True, timeout=60,
        )
        if proc.returncode == 0 and out_path.exists():
            sz = out_path.stat().st_size
            if sz > 0:
                return {**base, "status": DL_OK, "path": str(out_path), "error": None}
        err = proc.stderr.strip()[-200:] if proc.stderr else "download failed"
        return {**base, "status": DL_FAIL, "path": None, "error": err}
    except subprocess.TimeoutExpired:
        return {**base, "status": DL_FAIL, "path": None, "error": "timeout"}
    except Exception as exc:
        return {**base, "status": DL_FAIL, "path": None, "error": str(exc)[:200]}


def download_partner-dictnary_batch(
    df: pd.DataFrame,
    output_dir: Path,
    n_workers: int = 4,
    yt_dlp_cmd: str = "yt-dlp",
    progress_callback: Optional[Callable[[int, int, dict], None]] = None,
    skip_existing: bool = True,
    delay: float = 0.3,
) -> list[dict]:
    """Download partner-dictnary videos in parallel from a DataFrame.

    Args:
        df: DataFrame with word_id, translation, video_url columns.
        output_dir: Directory to save videos into.
        n_workers: Number of parallel download threads.
        yt_dlp_cmd: Path to yt-dlp binary.
        progress_callback: Called with (done_count, total, result_dict).
        skip_existing: Skip already-downloaded videos.
        delay: Delay in seconds between actual download requests (skips are instant).

    Returns:
        List of result dicts (one per row in df).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _worker(row: pd.Series) -> dict:
        result = download_partner-dictnary_video(row, output_dir, yt_dlp_cmd, skip_existing)
        # Only delay when an actual network request was made (not skipped)
        if result["status"] != DL_SKIP:
            time.sleep(delay)
        return result

    rows = [row for _, row in df.iterrows()]
    return _run_batch(rows, _worker, n_workers, progress_callback)


# ================================================================== #
#  FTP download (via curl subprocess)
# ================================================================== #


def list_ftp_directory(
    ftp_url: str,
    username: str = "",
    password: str = "",
    timeout: int = 30,
) -> list[str]:
    """List files/directories in an FTP directory.

    Args:
        ftp_url: FTP URL to list (e.g., ftp://example.com/path/).
        username: FTP username (empty for anonymous).
        password: FTP password.
        timeout: Connection timeout in seconds.

    Returns:
        List of filenames/directory names.
    """
    cmd = ["curl", "-s", "--list-only", "--max-time", str(timeout)]
    if username:
        cmd.extend(["--user", f"{username}:{password}"])

    # Ensure trailing slash for directory listing
    url = ftp_url.rstrip("/") + "/"
    cmd.append(url)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FTP listing failed: {result.stderr[:200]}")

    entries = [
        line.strip()
        for line in result.stdout.strip().split("\n")
        if line.strip() and line.strip() not in (".", "..")
    ]
    return entries


def download_ftp_file(
    ftp_url: str,
    output_path: Path,
    username: str = "",
    password: str = "",
    timeout: int = 120,
    skip_existing: bool = True,
) -> dict:
    """Download a single file from FTP via curl."""
    extra = ["--user", f"{username}:{password}"] if username else []
    return _curl_download(ftp_url, output_path, extra, timeout, skip_existing)


def download_ftp_directory(
    base_url: str,
    output_dir: Path,
    username: str = "",
    password: str = "",
    n_workers: int = 8,
    file_extensions: tuple[str, ...] = VIDEO_EXTENSIONS,
    skip_existing: bool = True,
    progress_callback: Optional[Callable[[int, int, dict], None]] = None,
) -> list[dict]:
    """Download all matching files from an FTP directory tree.

    Scans the FTP base_url for subdirectories, lists files in each,
    and downloads matching files in parallel.

    Args:
        base_url: FTP base URL (e.g., ftp://example.com/videos/).
        output_dir: Local directory to save files into.
        username: FTP username.
        password: FTP password.
        n_workers: Number of parallel download threads.
        file_extensions: File extensions to download.
        skip_existing: Skip already-downloaded files.
        progress_callback: Called with (done_count, total, result_dict).

    Returns:
        List of result dicts.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_url = base_url.rstrip("/")

    # List top-level entries (categories/subdirectories)
    logger.info("Listing FTP directory: %s", base_url)
    top_entries = list_ftp_directory(base_url, username, password)

    # Build download list: scan subdirectories for matching files
    downloads: list[tuple[str, str, Path]] = []  # (url, category, output_path)

    for entry in top_entries:
        ext_lower = Path(entry).suffix.lower()
        if ext_lower in file_extensions:
            # Top-level file
            url = f"{base_url}/{entry}"
            downloads.append((url, "", output_dir / entry))
        else:
            # Assume it's a subdirectory — list its contents
            try:
                sub_files = list_ftp_directory(
                    f"{base_url}/{entry}", username, password
                )
                for fname in sub_files:
                    if Path(fname).suffix.lower() in file_extensions:
                        url = f"{base_url}/{entry}/{fname}"
                        downloads.append(
                            (url, entry, output_dir / entry / fname)
                        )
            except RuntimeError:
                logger.warning("Failed to list subdirectory: %s", entry)

    logger.info("Found %d files to download", len(downloads))

    def _worker(item: tuple[str, str, Path]) -> dict:
        url, category, out_path = item
        result = download_ftp_file(url, out_path, username, password, 120, skip_existing)
        result["category"] = category
        return result

    return _run_batch(downloads, _worker, n_workers, progress_callback)


# ================================================================== #
#  HTTP direct file download
# ================================================================== #


def download_http_file(
    url: str,
    output_path: Path,
    timeout: int = 120,
    skip_existing: bool = True,
) -> dict:
    """Download a file from an HTTP/HTTPS URL via curl."""
    return _curl_download(url, output_path, ["-L", "--fail"], timeout, skip_existing)


def download_http_batch(
    urls: list[tuple[str, Path]],
    n_workers: int = 4,
    skip_existing: bool = True,
    progress_callback: Optional[Callable[[int, int, dict], None]] = None,
) -> list[dict]:
    """Download multiple files from HTTP/HTTPS URLs in parallel.

    Args:
        urls: List of (url, output_path) tuples.
        n_workers: Number of parallel download threads.
        skip_existing: Skip already-downloaded files.
        progress_callback: Called with (done_count, total, result_dict).

    Returns:
        List of result dicts.
    """
    def _worker(item: tuple[str, Path]) -> dict:
        url, out_path = item
        return download_http_file(url, out_path, 120, skip_existing)

    return _run_batch(urls, _worker, n_workers, progress_callback)
