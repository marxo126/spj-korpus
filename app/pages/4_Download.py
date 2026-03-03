"""YouTube Download page — paste multiple URLs and download in parallel.

Each URL can be a single video or a full playlist.
yt-dlp fetches video + subtitles (sk/cs/en) + metadata per video.
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from spj.downloader import download_url, list_playlist

DATA_DIR      = Path(__file__).parent.parent.parent / "data"
VIDEO_DIR     = DATA_DIR / "videos"
INVENTORY_CSV = DATA_DIR / "inventory.csv"

st.header("⬇ YouTube Download")
st.caption("Page 4/10 (optional) · Download sign language videos from YouTube.")
with st.expander("ℹ️ How to use this page", expanded=False):
    st.markdown("""
**Needs:**
- Nothing — can be done at any time

**Steps:**
1. Choose a preset (partner-ngo, [partner organization]) or paste YouTube URLs manually — one per line.
2. Click **🔍 Preview** to see the video list and total duration before downloading.
3. Set **Parallel workers** (2–3 is usually fastest).
4. Click **⬇ Download all**.
5. When done, go to page 1 (Inventory) and scan again to register the new files.

**Creates:** `data/videos/downloaded/<videoname>.mp4` + soft subtitle `.vtt` files alongside each video

**Note:** Only download partner-ngo / [partner organization] content — check CLAUDE.md for licensing rules.
""")

# ------------------------------------------------------------------ #
# Licensing reminder
# ------------------------------------------------------------------ #
with st.expander("⚠️ Licensing reminder — read before downloading", expanded=False):
    st.markdown(
        """
| Source | Status |
|---|---|
| **partner-ngo.eu** | ✅ Approved for ML training — written agreement **PENDING** |
| **[partner organization]** | ✅ Approved for ML training — written agreement **PENDING** |
| **Other** | Check license before using for training |

**Flag all partner-ngo / [partner organization] content before corpus deposit or publication.**
Do NOT use [vocabulary reference] or [public SL dataset] for ML training.
"""
    )

# ------------------------------------------------------------------ #
# Quick-fill presets
# ------------------------------------------------------------------ #
SOURCE_PRESETS = {
    "— enter manually —": "",
    "partner-ngo.eu (YouTube channel)": "https://www.youtube.com/@partner-ngo",
    "[partner organization] (YouTube channel)": "https://www.youtube.com/@[partner organization]net",
}

preset = st.selectbox("Quick-fill preset (adds URL to the box below)", list(SOURCE_PRESETS.keys()))

# ------------------------------------------------------------------ #
# Multi-URL text area
# ------------------------------------------------------------------ #
st.markdown("**YouTube URLs** — one per line (videos, playlists, or channel URLs)")
default_area = st.session_state.get("dl_urls_text", "")
if preset and SOURCE_PRESETS[preset]:
    preset_url = SOURCE_PRESETS[preset]
    # Append preset URL if not already present
    if preset_url not in default_area:
        default_area = (default_area.rstrip("\n") + "\n" + preset_url).lstrip("\n")

urls_text = st.text_area(
    "URLs",
    value=default_area,
    height=140,
    placeholder="https://www.youtube.com/watch?v=...\nhttps://www.youtube.com/playlist?list=...\nhttps://www.youtube.com/@SomeChannel",
    label_visibility="collapsed",
    key="dl_urls_text",
)

# Parse URLs — skip blank lines and # comments
raw_urls = [u.strip() for u in urls_text.splitlines() if u.strip() and not u.strip().startswith("#")]
n_urls = len(raw_urls)

if n_urls:
    st.caption(f"{n_urls} URL(s) entered")
else:
    st.info("Paste one or more YouTube URLs above — one per line.")

# Output subfolder
subfolder = st.text_input(
    "Save into subfolder of data/videos/",
    value=st.session_state.get("dl_subfolder", "downloaded"),
    help="Files land in data/videos/<subfolder>/",
    key="dl_subfolder",
)
out_dir = VIDEO_DIR / subfolder

# ------------------------------------------------------------------ #
# Action buttons
# ------------------------------------------------------------------ #
if n_urls:
    col_preview, col_download = st.columns([1, 1])
    preview_clicked  = col_preview.button("🔍 Preview all URLs")
    download_clicked = col_download.button("⬇ Download all", type="primary")
else:
    preview_clicked = download_clicked = False

# ------------------------------------------------------------------ #
# Preview
# ------------------------------------------------------------------ #
if preview_clicked and raw_urls:
    all_entries: list[dict] = []
    errors: list[str] = []
    prog = st.progress(0.0, text="Fetching info…")

    for i, url in enumerate(raw_urls):
        prog.progress((i + 1) / n_urls, text=f"Fetching {i + 1}/{n_urls}: {url[:60]}…")
        try:
            entries = list_playlist(url)
            for e in entries:
                e["_source_url"] = url
            all_entries.extend(entries)
        except Exception as exc:
            errors.append(f"❌ `{url}` — {exc}")

    prog.empty()

    if all_entries:
        total_sec = sum(e.get("duration") or 0 for e in all_entries)
        st.success(f"Found **{len(all_entries)}** video(s) across {n_urls} URL(s)  —  ~{total_sec / 3600:.1f} h total")
        df = pd.DataFrame(all_entries)[["title", "uploader", "upload_date", "duration", "_source_url"]]
        df["duration"] = df["duration"].apply(
            lambda s: f"{int(s // 60)}m {int(s % 60)}s" if s else "—"
        )
        df["upload_date"] = df["upload_date"].apply(
            lambda d: f"{d[:4]}-{d[4:6]}-{d[6:]}" if d and len(d) == 8 else (d or "—")
        )
        df.rename(columns={"_source_url": "source"}, inplace=True)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.session_state["preview_entries_multi"] = all_entries

    if errors:
        with st.expander(f"Errors ({len(errors)})"):
            for e in errors:
                st.markdown(e)

# ------------------------------------------------------------------ #
# Download
# ------------------------------------------------------------------ #
if download_clicked and raw_urls:
    preview = st.session_state.get("preview_entries_multi", [])
    n_expected = len(preview) if preview else "?"

    st.markdown(f"Downloading **{n_urls}** URL(s) → `{out_dir}`  (~{n_expected} video(s) total)")
    st.markdown("Subtitles (sk/cs/en), title, description and `.info.json` included per video.")

    n_workers = st.session_state.get("dl_workers", 2)

    # Per-URL shared state (written by worker threads, read by main thread)
    url_progress: dict[str, dict] = {
        url: {"pct": 0.0, "text": "queued…", "done": False, "results": [], "error": None}
        for url in raw_urls
    }

    def _run_one(url: str) -> tuple[str, list[dict] | Exception]:
        def _hook(d: dict):
            status   = d.get("status", "")
            filename = Path(d.get("filename", "")).name
            if status == "downloading":
                total_b = d.get("total_bytes") or d.get("total_bytes_estimate") or 1
                dl_b    = d.get("downloaded_bytes", 0)
                speed   = (d.get("speed") or 0) / 1_048_576
                eta     = d.get("eta") or 0
                url_progress[url]["pct"]  = min(dl_b / total_b * 0.95, 0.95)
                url_progress[url]["text"] = f"{filename}  {speed:.1f} MB/s  ETA {eta}s"
            elif status == "finished":
                url_progress[url]["pct"]  = 0.98
                url_progress[url]["text"] = f"post-processing {filename}…"

        try:
            results = download_url(url, out_dir, progress_hook=_hook)
            return url, results
        except Exception as exc:
            return url, exc

    # Parallel workers slider (rendered before we block on pool)
    n_workers = st.slider(
        "Parallel download workers",
        min_value=1, max_value=4, value=2,
        help="Each worker downloads a separate URL concurrently. "
             "2–3 is usually fastest on a good connection.",
        key="dl_workers",
    )

    overall_bar  = st.progress(0.0, text=f"0 / {n_urls} URLs done")
    st.markdown("**Active downloads:**")
    url_placeholders: dict[str, object] = {}

    # Initialise progress bars for first batch
    for url in raw_urls[:n_workers]:
        url_placeholders[url] = st.empty()
        url_placeholders[url].progress(0.0, text=f"⏳ {url[:80]}")

    log_lines: list[str] = []
    errors:    list[str] = []
    completed = 0
    log_out   = st.empty()

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_one, url): url for url in raw_urls}
        pending = set(futures.keys())

        while pending:
            # Refresh per-URL progress bars
            for url, ph in list(url_placeholders.items()):
                info = url_progress[url]
                ph.progress(info["pct"], text=f"⏳ {url[:70]}  {info['text']}")

            done_now = {f for f in pending if f.done()}
            pending -= done_now

            for future in done_now:
                url         = futures[future]
                placeholder = url_placeholders.pop(url, None)
                if placeholder:
                    placeholder.empty()

                _, result = future.result()
                if isinstance(result, Exception):
                    line = f"❌ `{url[:70]}` — {result}"
                    errors.append(line)
                else:
                    n = len(result)
                    line = f"✅ `{url[:70]}` — {n} video(s)"

                log_lines.append(line)
                completed += 1
                overall_bar.progress(completed / n_urls, text=f"{completed} / {n_urls} URLs done")
                log_out.markdown("\n\n".join(log_lines[-20:]))

                # Assign freed slot to next queued URL
                next_urls = [futures[f] for f in pending if futures[f] not in url_placeholders]
                if next_urls:
                    nxt = next_urls[0]
                    url_placeholders[nxt] = st.empty()
                    url_placeholders[nxt].progress(0.0, text=f"⏳ {nxt[:80]}")

            time.sleep(0.3)

    overall_bar.progress(1.0, text=f"{n_urls} / {n_urls} URLs done")

    if errors:
        with st.expander(f"Errors ({len(errors)})"):
            for e in errors:
                st.markdown(e)

    ok = n_urls - len(errors)
    st.success(f"Download complete — {ok} / {n_urls} URLs finished. Files saved to `{out_dir}`.")

    # Invalidate in-memory inventory so next scan picks up new files
    st.session_state.pop("inventory", None)
    st.session_state.pop("preview_entries_multi", None)
    st.warning(
        "Inventory is **stale** — new files were downloaded. "
        "Go to the **Inventory** page and click **Scan Videos** to update."
    )

# ------------------------------------------------------------------ #
# intl-vocab partner-dictnary download
# ------------------------------------------------------------------ #
st.divider()
st.subheader("📖 intl-vocab partner-dictnary Download")
st.caption(
    "Download individual sign videos from intl-vocab.com using their "
    "partner-dictnary CSV export. Each video = one sign (~2-5s, 320x240)."
)

STS_CSV = DATA_DIR / "partner-dictnary_export_2026-02-26_10-01-57.csv"
STS_DIR = VIDEO_DIR / "intl-vocab"

if STS_CSV.exists():
    from spj.intl-vocab import load_partner-dictnary, download_batch, _sanitize_filename

    sts_df = load_partner-dictnary(STS_CSV)

    # Count already downloaded
    _downloaded_ids = set()
    if STS_DIR.exists():
        for f in STS_DIR.glob("*.mp4"):
            parts = f.stem.split("_", 1)
            if parts[0].isdigit():
                _downloaded_ids.add(int(parts[0]))

    _with_trans = sts_df[sts_df["translation"].notna() & (sts_df["translation"].str.strip() != "")]
    _remaining = _with_trans[~_with_trans["word_id"].isin(_downloaded_ids)]

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Total in CSV", len(sts_df))
    mc2.metric("Downloaded", len(_downloaded_ids))
    mc3.metric("Remaining", len(_remaining))

    # Word class filter
    wc_options = sorted(_remaining["word_class"].dropna().unique().tolist())
    selected_wc = st.multiselect(
        "Word classes to download",
        options=wc_options,
        default=[wc for wc in ["Noun", "Verb", "Adjective", "Adverb", "Numeral",
                                "Pronoun", "Conjunction", "Preposition", "Interjection"]
                 if wc in wc_options],
        key="sts_word_classes",
    )

    if selected_wc:
        filtered = _remaining[_remaining["word_class"].isin(selected_wc)]
    else:
        filtered = _remaining

    st.caption(f"**{len(filtered)}** entries to download after filtering")

    scol1, scol2 = st.columns(2)
    with scol1:
        sts_workers = st.slider("Workers", 1, 16, 8, key="sts_workers")
    with scol2:
        sts_delay = st.slider("Delay per request (s)", 0.1, 2.0, 0.3, 0.1, key="sts_delay")

    sts_limit = st.number_input(
        "Max videos to download (0 = all)",
        min_value=0, value=0, step=500, key="sts_limit",
    )

    if st.button("📥 Download intl-vocab", type="primary", disabled=len(filtered) == 0):
        dl_df = filtered.copy()
        if sts_limit > 0:
            dl_df = dl_df.head(sts_limit)

        st.markdown(f"Downloading **{len(dl_df)}** videos → `{STS_DIR}`")
        prog = st.progress(0.0, text="Starting…")
        log_area = st.empty()
        _ok = 0
        _fail = 0

        def _progress(done, total, res):
            nonlocal _ok, _fail
            if res["success"]:
                _ok += 1
            else:
                _fail += 1
            pct = done / total
            prog.progress(pct, text=f"{done}/{total} — ✅ {_ok} ok, ❌ {_fail} fail")

        results = download_batch(
            dl_df, STS_DIR,
            n_workers=sts_workers,
            yt_dlp_cmd=".venv/bin/yt-dlp",
            progress_callback=_progress,
            skip_existing=True,
            delay=sts_delay,
        )

        prog.progress(1.0, text="Done!")
        ok = sum(1 for r in results if r["success"])
        st.success(f"Downloaded **{ok}** / {len(results)} videos. "
                   f"Total in folder: ~{len(_downloaded_ids) + ok}")

        st.session_state.pop("inventory", None)
else:
    st.info(
        "No intl-vocab partner-dictnary CSV found. "
        "Place `partner-dictnary_export_*.csv` in `data/` to enable."
    )

# ------------------------------------------------------------------ #
# Files already on disk
# ------------------------------------------------------------------ #
st.divider()
st.subheader("Files in data/videos/")
if VIDEO_DIR.exists():
    all_videos = [
        v for ext in ("*.mp4", "*.mkv", "*.webm", "*.mov")
        for v in VIDEO_DIR.rglob(ext)
    ]
    if all_videos:
        rows = [
            {
                "filename":  v.name,
                "subfolder": str(v.parent.relative_to(VIDEO_DIR)),
                "size_mb":   round(v.stat().st_size / 1_048_576, 1),
            }
            for v in sorted(all_videos)
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(f"{len(all_videos)} video file(s) total")
    else:
        st.info("No video files yet.")
else:
    st.info("data/videos/ directory not found.")
