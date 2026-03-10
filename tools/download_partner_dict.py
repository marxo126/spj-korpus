"""Download SPJ entries from partner-dict (partner-university University).

Scrapes entry pages for videos (front + side views), SignWriting FSW codes,
and metadata. SPJ entries have IDs in range 1-7000 (with gaps).

FOR PRIVATE RESEARCH USE ONLY — not for public distribution or commercial use.

Usage:
    .venv/bin/python tools/download_partner-dict.py
    .venv/bin/python tools/download_partner-dict.py --dry-run
    .venv/bin/python tools/download_partner-dict.py --metadata-only
"""
import csv
import json
import re
import ssl
import sys
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = DATA_DIR / "reference" / "partner-dict"
META_PATH = DATA_DIR / "partner-dict_metadata.csv"

# Skip SSL verification for partner-dict (bad cert)
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

MAX_ENTRY_ID = 7000


def _fetch(url: str) -> str:
    try:
        return urllib.request.urlopen(url, context=SSL_CTX, timeout=15).read().decode("utf-8")
    except Exception:
        return ""


def scan_entries() -> list[dict]:
    """Scan all SPJ entry IDs and extract metadata from detail pages."""
    entries = []

    for eid in range(1, MAX_ENTRY_ID + 1):
        html = _fetch(f"https://www.partner-dict/spj/show/{eid}")
        if not html or "videospj" not in html:
            continue

        # Video files (front A_ and side B_)
        videos = re.findall(r"files\.partner-dict\.info/(videospj/[^\"]+\.mp4)", html)
        # Filter to only SPJ videos (not CZJ cross-references)
        spj_videos = [v for v in videos if "videospj" in v]

        # SignWriting FSW codes
        fsw_codes = re.findall(
            r"sign\.partner-dict\.info/fsw/sign/png/([^\"]+?)-CG_white_", html
        )

        # Entry name / lemma
        name_m = re.search(r'class="entry-name[^"]*"[^>]*>([^<]+)', html)
        if not name_m:
            name_m = re.search(r"<h2[^>]*>([^<]+)</h2>", html)
        name = name_m.group(1).strip() if name_m else f"spj-{eid}"

        # Slovak translation text
        trans_m = re.search(r'translate-text[^>]*>([^<]+)', html)
        translation = trans_m.group(1).strip() if trans_m else ""

        entries.append({
            "id": eid,
            "name": name,
            "translation": translation,
            "videos": spj_videos,
            "fsw_codes": fsw_codes,
        })

        if len(entries) % 100 == 0:
            print(f"  Found {len(entries)} entries (scanned {eid}/{MAX_ENTRY_ID})...")

    return entries


def download_videos(entries: list[dict], dry_run: bool = False) -> int:
    """Download video files for all entries."""
    video_dir = OUT_DIR / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    failed = []

    for entry in entries:
        for video_path in entry["videos"]:
            fname = video_path.split("/")[-1]
            out_path = video_dir / fname

            if out_path.exists() and out_path.stat().st_size > 0:
                skipped += 1
                continue

            if dry_run:
                print(f"  [DRY] {fname}")
                continue

            url = f"https://files.partner-dict/{video_path}"
            try:
                resp = urllib.request.urlopen(url, context=SSL_CTX, timeout=30)
                with open(out_path, "wb") as fout:
                    fout.write(resp.read())
                downloaded += 1
                if downloaded % 50 == 0:
                    print(f"  Downloaded {downloaded}...")
            except Exception as e:
                failed.append((fname, str(e)))

    print(f"  Videos — Downloaded: {downloaded}, Skipped: {skipped}, Failed: {len(failed)}")
    if failed:
        for f, err in failed[:10]:
            print(f"    FAIL: {f}: {err}")
    return downloaded


def download_signwriting(entries: list[dict], dry_run: bool = False) -> int:
    """Download SignWriting PNG images."""
    sw_dir = OUT_DIR / "signwriting"
    sw_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0

    for entry in entries:
        for fsw in entry["fsw_codes"]:
            # Sanitize FSW code for filename
            safe_fsw = fsw.replace("/", "_")[:100]
            fname = f"spj_{entry['id']}_{safe_fsw}.png"
            out_path = sw_dir / fname

            if out_path.exists() and out_path.stat().st_size > 0:
                skipped += 1
                continue

            if dry_run:
                continue

            url = f"https://sign.partner-dict/fsw/sign/png/{fsw}-CG_white_"
            try:
                resp = urllib.request.urlopen(url, context=SSL_CTX, timeout=30)
                with open(out_path, "wb") as fout:
                    fout.write(resp.read())
                downloaded += 1
            except Exception:
                pass

    print(f"  SignWriting — Downloaded: {downloaded}, Skipped: {skipped}")
    return downloaded


def _load_cached_metadata() -> list[dict] | None:
    if not META_PATH.exists():
        return None
    entries = []
    with open(META_PATH) as f:
        for row in csv.DictReader(f):
            row["videos"] = json.loads(row["videos"])
            row["fsw_codes"] = json.loads(row["fsw_codes"])
            entries.append(row)
    return entries


def main():
    dry_run = "--dry-run" in sys.argv
    metadata_only = "--metadata-only" in sys.argv
    force_rescan = "--rescan" in sys.argv

    print("=" * 60)
    print("  partner-dict — SPJ (Slovak Sign Language)")
    print("  FOR PRIVATE RESEARCH USE ONLY")
    print("=" * 60)

    entries = None if force_rescan else _load_cached_metadata()
    if entries:
        total_videos = sum(len(e["videos"]) for e in entries)
        total_sw = sum(len(e["fsw_codes"]) for e in entries)
        print(f"\n   Using cached metadata: {len(entries)} entries, {total_videos} videos, {total_sw} SW")
    else:
        print(f"\n1. Scanning SPJ entries (IDs 1-{MAX_ENTRY_ID})...")
        entries = scan_entries()
        total_videos = sum(len(e["videos"]) for e in entries)
        total_sw = sum(len(e["fsw_codes"]) for e in entries)
        print(f"   Found {len(entries)} entries, {total_videos} videos, {total_sw} SignWriting codes")

        META_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(META_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "id", "name", "translation", "videos", "fsw_codes",
            ])
            writer.writeheader()
            for e in entries:
                writer.writerow({
                    "id": e["id"],
                    "name": e["name"],
                    "translation": e.get("translation", ""),
                    "videos": json.dumps(e["videos"]),
                    "fsw_codes": json.dumps(e["fsw_codes"]),
                })
        print(f"\n   Metadata saved: {META_PATH}")

    if metadata_only:
        print("\n   --metadata-only: skipping downloads")
        return

    print("\n2. Downloading videos...")
    download_videos(entries, dry_run=dry_run)

    print("\n3. Downloading SignWriting images...")
    download_signwriting(entries, dry_run=dry_run)

    print(f"\n{'=' * 60}")
    print(f"  Done: {len(entries)} entries")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
