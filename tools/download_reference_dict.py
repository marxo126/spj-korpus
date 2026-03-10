"""Download all signs from reference-dict (Slovak Sign Language partner-dictnary).

Crawls the paginated listing, extracts metadata and YouTube video IDs,
then downloads videos via yt-dlp.

Usage:
    .venv/bin/python tools/download_reference-dict.py
    .venv/bin/python tools/download_reference-dict.py --dry-run
    .venv/bin/python tools/download_reference-dict.py --metadata-only
"""
import csv
import json
import re
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = DATA_DIR / "reference" / "reference-dict_sk"
META_PATH = DATA_DIR / "reference-dict_sk_metadata.csv"

NODE_PATH = "/Users/marekkanas/.nvm/versions/node/v24.12.0/bin/node"


def _fetch_page(url: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            return urllib.request.urlopen(url, timeout=15).read().decode("iso-8859-2")
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise


def collect_sign_urls() -> list[dict]:
    """Crawl paginated listing to collect all sign URLs and basic metadata."""
    base = "https://reference-dict/reference-dict/stranka-{}/"
    all_signs = {}
    consecutive_empty = 0

    for page in range(1, 1000):
        html = _fetch_page(base.format(page))
        # Pattern: /word-slug/posunok-ID/
        entries = re.findall(r'href="(/([^/]+)/posunok-(\d+)/)"', html)
        new_count = 0
        for href, slug, sid in entries:
            key = f"{slug}_{sid}"
            if key not in all_signs:
                all_signs[key] = {
                    "id": sid,
                    "slug": slug,
                    "url": f"https://reference-dict{href}",
                }
                new_count += 1

        if new_count == 0:
            consecutive_empty += 1
            if consecutive_empty >= 5:
                break
        else:
            consecutive_empty = 0

        if page % 50 == 0:
            print(f"  Page {page}: {len(all_signs)} unique signs")

    return list(all_signs.values())


def enrich_sign_data(signs: list[dict]) -> list[dict]:
    """Fetch each sign page to extract YouTube IDs and metadata."""
    for i, sign in enumerate(signs):
        html = _fetch_page(sign["url"])

        # YouTube embed IDs (usually 2: 60° and 90° angles)
        yt_ids = re.findall(r"youtube\.com/embed/([^?\"&]+)", html)
        sign["youtube_ids"] = list(dict.fromkeys(yt_ids))  # dedupe, preserve order

        # Also check for thumbnail YouTube IDs
        thumb_ids = re.findall(r"img\.youtube\.com/vi/([^/\"]+)", html)
        for tid in thumb_ids:
            if tid not in sign["youtube_ids"]:
                sign["youtube_ids"].append(tid)

        # Extract title from og:title
        title_m = re.search(r'og:title.*?content="([^"]+)"', html)
        sign["title"] = title_m.group(1).replace(" - reference-dict", "").strip() if title_m else sign["slug"]

        # Extract hand shape images
        shapes = re.findall(r'src="(/_images/tvaryruky/[^"]+)"', html)
        sign["hand_shapes"] = shapes

        if (i + 1) % 100 == 0:
            print(f"  Enriched {i + 1}/{len(signs)}...")

    return signs


COOKIE_FILE = Path("/tmp/yt_reference-dict_cookies.txt")
MAX_WORKERS = 8  # 2 workers per IP × 4 IPs (proven stable at 80/min)
PAUSE_EVERY = 200  # pause after this many downloads
PAUSE_SECONDS = 15  # seconds to pause

# NordVPN SOCKS5 proxies — each gives a different IP (2 workers per IP)
_NORD_USER = "kX3ACC7QVNZ5edxtgVT6qz7o"
_NORD_PASS = "qcENNWpow5CjeMQxYkiUwZ3j"
PROXIES = [
    None, None,  # direct IP (2 workers)
    f"socks5://{_NORD_USER}:{_NORD_PASS}@nl.socks.nordhold.net:1080",
    f"socks5://{_NORD_USER}:{_NORD_PASS}@nl.socks.nordhold.net:1080",
    f"socks5://{_NORD_USER}:{_NORD_PASS}@se.socks.nordhold.net:1080",
    f"socks5://{_NORD_USER}:{_NORD_PASS}@se.socks.nordhold.net:1080",
    f"socks5://{_NORD_USER}:{_NORD_PASS}@us.socks.nordhold.net:1080",
    f"socks5://{_NORD_USER}:{_NORD_PASS}@us.socks.nordhold.net:1080",
]


def _export_cookies() -> None:
    """Export Chrome cookies to file once (avoids per-video extraction)."""
    subprocess.run(
        [".venv/bin/yt-dlp", "--cookies-from-browser", "chrome",
         "--cookies", str(COOKIE_FILE), "--simulate",
         "https://www.youtube.com/watch?v=dQw4w9WgXcQ"],
        capture_output=True, timeout=30,
    )


def _download_one(yt_id: str, out_path: Path, proxy: str | None = None) -> tuple[str, bool, str]:
    """Download a single video. Returns (filename, success, error)."""
    url = f"https://www.youtube.com/watch?v={yt_id}"
    cmd = [
        ".venv/bin/yt-dlp",
        "--cookies", str(COOKIE_FILE),
        "--js-runtimes", f"node:{NODE_PATH}",
        "-f", "best[height<=720]/best",
        "-o", str(out_path),
        "--no-warnings", "-q",
    ]
    if proxy:
        cmd.extend(["--proxy", proxy])
    cmd.append(url)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            return (out_path.name, True, "")
        return (out_path.name, False, result.stderr.strip().split("\n")[-1])
    except subprocess.TimeoutExpired:
        return (out_path.name, False, "timeout")


def download_videos(signs: list[dict], dry_run: bool = False) -> int:
    """Download YouTube videos using 8 workers across 4 IPs (direct + NordVPN)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build work list
    jobs = []
    skipped = 0
    for sign in signs:
        for idx, yt_id in enumerate(sign["youtube_ids"]):
            angle = "60" if idx == 0 else "90" if idx == 1 else str(idx)
            fname = f"{sign['slug']}_{sign['id']}_{angle}_{yt_id}.mp4"
            out_path = OUT_DIR / fname

            if out_path.exists() and out_path.stat().st_size > 0:
                skipped += 1
                continue

            if dry_run:
                print(f"  [DRY] {fname}")
                continue

            jobs.append((yt_id, out_path))

    print(f"  To download: {len(jobs)}, Already exists: {skipped}")
    if not jobs or dry_run:
        return 0

    # Export cookies once
    _export_cookies()

    downloaded = 0
    failed = []
    lock = Lock()

    # Assign each job a proxy in round-robin across IPs
    def _submit(pool, idx, yt_id, out_path):
        proxy = PROXIES[idx % len(PROXIES)]
        return pool.submit(_download_one, yt_id, out_path, proxy)

    # Process in batches with pauses
    for batch_start in range(0, len(jobs), PAUSE_EVERY):
        batch = jobs[batch_start:batch_start + PAUSE_EVERY]
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {_submit(pool, batch_start + i, yt_id, out): yt_id
                       for i, (yt_id, out) in enumerate(batch)}
            for fut in as_completed(futures):
                fname, ok, err = fut.result()
                with lock:
                    if ok:
                        downloaded += 1
                        if downloaded % 50 == 0:
                            print(f"  Downloaded {downloaded}/{len(jobs)}...", flush=True)
                    else:
                        failed.append((fname, err))
        if batch_start + PAUSE_EVERY < len(jobs):
            print(f"  Pausing {PAUSE_SECONDS}s...", flush=True)
            time.sleep(PAUSE_SECONDS)

    print(f"  Downloaded: {downloaded}, Skipped: {skipped}, Failed: {len(failed)}")
    if failed:
        for f, err in failed[:10]:
            print(f"    FAIL: {f}: {err}")
    return downloaded


def _load_cached_metadata() -> list[dict] | None:
    """Load metadata from CSV if it exists."""
    if not META_PATH.exists():
        return None
    signs = []
    with open(META_PATH) as f:
        for row in csv.DictReader(f):
            row["youtube_ids"] = json.loads(row["youtube_ids"])
            row["hand_shapes"] = json.loads(row.get("hand_shapes", "[]"))
            signs.append(row)
    return signs


def _save_metadata(signs: list[dict]) -> None:
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(META_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "slug", "title", "url", "youtube_ids", "hand_shapes",
        ])
        writer.writeheader()
        for s in signs:
            writer.writerow({
                "id": s["id"],
                "slug": s["slug"],
                "title": s.get("title", ""),
                "url": s["url"],
                "youtube_ids": json.dumps(s["youtube_ids"]),
                "hand_shapes": json.dumps(s.get("hand_shapes", [])),
            })
    print(f"   Metadata saved: {META_PATH} ({len(signs)} entries)")


def main():
    dry_run = "--dry-run" in sys.argv
    metadata_only = "--metadata-only" in sys.argv
    force_rescan = "--rescan" in sys.argv

    print("=" * 60)
    print("  reference-dict — Slovak Sign Language partner-dictnary")
    print("=" * 60)

    # Use cached metadata if available
    signs = None if force_rescan else _load_cached_metadata()
    if signs:
        total_videos = sum(len(s["youtube_ids"]) for s in signs)
        print(f"\n   Using cached metadata: {len(signs)} signs, {total_videos} videos")
    else:
        print("\n1. Crawling sign listing...")
        signs = collect_sign_urls()
        print(f"   Found {len(signs)} unique signs")

        print("\n2. Fetching sign details...")
        signs = enrich_sign_data(signs)
        total_videos = sum(len(s["youtube_ids"]) for s in signs)
        print(f"   Total YouTube videos: {total_videos}")

        _save_metadata(signs)

    if metadata_only:
        print("\n   --metadata-only: skipping video download")
        return

    # Download videos
    print("\n3. Downloading videos...")
    download_videos(signs, dry_run=dry_run)

    print(f"\n{'=' * 60}")
    total_videos = sum(len(s["youtube_ids"]) for s in signs)
    print(f"  Done: {len(signs)} signs, {total_videos} videos")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
