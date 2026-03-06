"""Download SPJ vocabulary videos from partner-ngo project websites.

Sources:
  - REDACTED — financial literacy signs (Slovak subset)
  - REDACTED — career/job vocabulary signs
  - REDACTED — climate vocabulary signs (if any)

Usage:
    .venv/bin/python tools/download_vocab_videos.py
    .venv/bin/python tools/download_vocab_videos.py --dry-run
"""
import json
import re
import sys
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# Hungarian words/patterns to exclude from fin-vocab
_HUNGARIAN_PATTERNS = [
    "Adathalaszat", "Vasarloero", "Inflacio", "Deflacio", "Koezponti",
    "Ketlepcsos", "Szemelyes", "Penzuegyi", "Szemelyi", "Joevede",
    "Szemelyazonossag", "Refinanszirozas", "Jelzalog", "Hitel",
    "Lakasbiztositas", "kamat", "Fedezet", "Reszveny", "Befektet",
    "Koetveny", "Koetvany", "Kiegeszito", "Koetelezo", "Magan-fele",
    "Biztositasi", "Jogi-koelt", "Adossag", "Koeltseg", "Kamatlab",
    "Valtozo", "Tartalek", "Kolcson", "Haztartas", "joevedelem",
    "Csaladi", "Egyenleg", "Vasarlasi", "magatartas", "Ber",
    "Passziv", "Aktiv", "Tokejoevedelem", "Afa",
]
_GERMAN_PATTERNS = ["die-", "das-", "der-", "einen-", "kreditwuerdig"]
_ENGLISH_PATTERNS = [
    "income_", "balance_", "bond_", "budget_", "budgeting_", "credit_",
    "debt_", "household", "interest_", "investment_", "legal-", "mutual-",
    "passive-", "premium_", "private-", "rate_", "reserves_", "statutory-",
    "supplementary-", "variable-", "wage_", "Purchasing-", "active-income",
    "Household-",
]
_ITALIAN_PATTERNS = ["_FS_", "Potere-dacquisto"]


def _fetch_wp_videos(base_url: str) -> list[dict]:
    """Fetch all video media from WP REST API."""
    videos = []
    for page in range(1, 30):
        url = f"{base_url}/wp-json/wp/v2/media?media_type=video&per_page=100&page={page}&_fields=id,title,source_url"
        try:
            resp = urllib.request.urlopen(url)
            data = json.loads(resp.read())
            if not data:
                break
            videos.extend(data)
        except Exception:
            break
    return videos


def _is_slovak_financial(fname: str) -> bool:
    """Check if a fin-vocab video filename is Slovak."""
    low = fname.lower()
    for pat in _HUNGARIAN_PATTERNS + _GERMAN_PATTERNS + _ENGLISH_PATTERNS + _ITALIAN_PATTERNS:
        if pat.lower() in low:
            return False
    # Skip numbered prefix patterns (German/Hungarian: 33_08_, 45_11_)
    parts = fname.split("_")
    if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit():
        return False
    # Skip numbered files without Slovak words (NN_1.mp4 = unclear language)
    if re.match(r"^\d+_\d+\.mp4$", fname):
        return False
    # Skip numbered Hungarian (01-Aktiv, etc.)
    if re.match(r"^\d{2}-[A-Z]", fname):
        return False
    return True


def _is_slovak_career(fname: str) -> bool:
    """Check if a career-vocab video is Slovak SPJ.

    Slovak files use dash-separated names: NN-english-word-slovensky-nazov.mp4
    German files use underscore: NN_die_Wort_DGS.mp4 or NN_word.mp4
    """
    # German/Austrian DGS files: underscores between words, or _DGS suffix
    if "_DGS" in fname or "_dgs" in fname:
        return False
    # German files: NN_germanword pattern (underscores, no dashes)
    if "_" in fname and "-" not in fname:
        return False
    # Test/generic files
    if fname.startswith("Video-"):
        return False
    # German word in filename
    low = fname.lower()
    if "gastgewerbe" in low or "tourismus" in low:
        return False
    # Files ending with just "1.mp4" are English-only duplicates (no Slovak label)
    stem = fname.replace(".mp4", "")
    if stem.endswith("1") and not any(c in stem for c in "áéíóúýčďľňřšťžô"):
        # Check if it has a Slovak word after a dash
        parts = stem.split("-")
        # NN-english-word1.mp4 = English only, skip
        if len(parts) <= 3 and parts[-1].endswith("1"):
            return False
    # Slovak files have dashes: NN-english-slovakword.mp4
    if "-" in fname:
        return True
    return False


def _label_from_filename(fname: str, source: str, title: str = "") -> str:
    """Extract Slovak label from video filename or WP title."""
    stem = fname.replace(".mp4", "")
    if source == "fin-vocab":
        return stem.replace("-", " ").strip()
    elif source == "career-vocab":
        # WP title: "02 – rooms – miestnosti" → extract last part (Slovak)
        if "\u2013" in title or "&#8211;" in title:
            clean = title.replace("&#8211;", "\u2013")
            parts = [p.strip() for p in clean.split("\u2013")]
            if len(parts) >= 3:
                return parts[-1]
            elif len(parts) >= 2:
                return parts[-1]
        # Fallback: parse filename (NN-english-slovakword.mp4)
        parts = stem.split("-")
        if parts and (parts[0].isdigit() or parts[0].startswith("0")):
            parts = parts[1:]
        return " ".join(parts).strip()
    elif source == "climate-vocabaction":
        return stem.replace("-", " ").strip()
    return stem


def download_source(base_url: str, source_name: str, filter_fn, dry_run: bool = False) -> list[dict]:
    """Download videos from a WP site and return metadata list."""
    print(f"\n{'='*60}")
    print(f"  {source_name}: {base_url}")
    print(f"{'='*60}")

    out_dir = DATA_DIR / "videos" / source_name
    out_dir.mkdir(parents=True, exist_ok=True)

    all_videos = _fetch_wp_videos(base_url)
    print(f"  Total videos on site: {len(all_videos)}")

    sk_videos = [v for v in all_videos if filter_fn(v["source_url"].split("/")[-1])]
    print(f"  Slovak SPJ videos: {len(sk_videos)}")

    metadata = []
    downloaded = 0
    skipped = 0

    for v in sk_videos:
        url = v["source_url"]
        fname = url.split("/")[-1]
        title = v.get("title", {}).get("rendered", "")
        out_path = out_dir / fname
        label = _label_from_filename(fname, source_name, title=title)

        metadata.append({
            "source": source_name,
            "filename": fname,
            "label": label,
            "url": url,
            "path": str(out_path),
        })

        if out_path.exists():
            skipped += 1
            continue

        if dry_run:
            print(f"  [DRY] {fname} — {label}")
            continue

        try:
            # URL-encode non-ASCII characters in the filename part
            from urllib.parse import quote, urlparse
            parsed = urlparse(url)
            encoded_path = quote(parsed.path, safe="/")
            encoded_url = parsed._replace(path=encoded_path).geturl()
            urllib.request.urlretrieve(encoded_url, str(out_path))
            downloaded += 1
            if downloaded % 10 == 0:
                print(f"  Downloaded {downloaded}/{len(sk_videos)}...")
        except Exception as e:
            print(f"  ERROR downloading {fname}: {e}")

    print(f"  Downloaded: {downloaded}, Already existed: {skipped}")
    return metadata


def main():
    dry_run = "--dry-run" in sys.argv

    all_metadata = []

    # 1. Financial Signs (Slovak subset)
    meta = download_source(
        "https://www.REDACTED",
        "fin-vocab",
        _is_slovak_financial,
        dry_run=dry_run,
    )
    all_metadata.extend(meta)

    # 2. Career Paths Inclusive
    meta = download_source(
        "https://REDACTED",
        "career-vocab",
        _is_slovak_career,
        dry_run=dry_run,
    )
    all_metadata.extend(meta)

    # 3. Deaf Climate Action (check for sign videos)
    meta = download_source(
        "https://REDACTED",
        "climate-vocabaction",
        lambda f: True,
        dry_run=dry_run,
    )
    all_metadata.extend(meta)

    # Save metadata CSV
    if all_metadata and not dry_run:
        import csv
        csv_path = DATA_DIR / "vocab_videos_metadata.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["source", "filename", "label", "url", "path"])
            writer.writeheader()
            writer.writerows(all_metadata)
        print(f"\nMetadata saved: {csv_path} ({len(all_metadata)} entries)")

    print(f"\n{'='*60}")
    print(f"  Total SPJ videos: {len(all_metadata)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
