"""Download page — YouTube, partner-dictnary CSV, FTP, and direct HTTP downloads.

Multi-source video downloader for building the sign language corpus.
All sources require written agreements from data owners before use.
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from spj.downloader import (
    download_url, list_playlist,
    load_partner-dictnary_csv, download_partner-dictnary_batch,
    list_ftp_directory, download_ftp_directory,
    download_http_file, download_http_batch,
    check_agreement, agreement_warning,
    DL_OK, DL_SKIP, DL_FAIL, VIDEO_EXTENSIONS,
)

DATA_DIR      = Path(__file__).parent.parent.parent / "data"
VIDEO_DIR     = DATA_DIR / "videos"
INVENTORY_CSV = DATA_DIR / "inventory.csv"

st.header("4. ⬇ Download")
st.caption("Page 4/10 (optional) · Download sign language videos from multiple sources.")
with st.expander("ℹ️ How to use this page", expanded=False):
    st.markdown("""
**Supported sources:**
- **YouTube** — single videos, playlists, or channel URLs
- **partner-dictnary CSV** — bulk download individual sign videos from a CSV export
- **FTP** — download from FTP servers (category directories)
- **HTTP** — direct file download from URLs

**Licensing:**
All training data requires written agreements from data owners.
Place `AGREEMENT.txt` in each download directory to confirm you have permission.

**After downloading:** Go to page 1 (Inventory) and scan to register new files.
""")

# ------------------------------------------------------------------ #
# Licensing reminder
# ------------------------------------------------------------------ #
with st.expander("⚠️ Licensing reminder — read before downloading", expanded=False):
    st.markdown(
        """
All training data requires written agreements from data owners before use.

- Verify you have permission to use each source for ML training
- Place `AGREEMENT.txt` in each download directory to confirm agreement
- Flag all content before corpus deposit or publication
- Do NOT use third-party content without explicit written agreement
"""
    )

# ================================================================== #
#  Tab layout for different download sources
# ================================================================== #
tab_yt, tab_dict, tab_ftp, tab_http = st.tabs([
    "🎬 YouTube", "📖 partner-dictnary CSV", "📁 FTP", "🌐 HTTP"
])

# ================================================================== #
#  YouTube tab
# ================================================================== #
with tab_yt:
    st.markdown("**YouTube URLs** — one per line (videos, playlists, or channel URLs)")
    default_area = st.session_state.get("dl_urls_text", "")

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

    # Action buttons
    if n_urls:
        col_preview, col_download = st.columns([1, 1])
        preview_clicked  = col_preview.button("🔍 Preview all URLs")
        download_clicked = col_download.button("⬇ Download all", type="primary")
    else:
        preview_clicked = download_clicked = False

    # Preview
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

    # Download
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

# ================================================================== #
#  partner-dictnary CSV tab
# ================================================================== #
with tab_dict:
    st.markdown(
        "Download individual sign videos from a partner-dictnary CSV export. "
        "Each video = one sign (~2-5s)."
    )

    _DICT_CSV = DATA_DIR / "partner-dictnary_export.csv"
    _DICT_DIR = VIDEO_DIR / "partner-dictnary"

    # Look for any partner-dictnary_export*.csv in data/
    _dict_candidates = sorted(DATA_DIR.glob("partner-dictnary_export*.csv"))
    if _dict_candidates:
        _DICT_CSV = _dict_candidates[-1]  # use most recent

    if _DICT_CSV.exists():
        dict_df = load_partner-dictnary_csv(_DICT_CSV)
        st.caption(f"CSV: `{_DICT_CSV.name}` — {len(dict_df)} entries")

        # Agreement check
        if not check_agreement(_DICT_DIR):
            st.warning(agreement_warning("partner-dictnary"))

        # Count already downloaded
        _downloaded_ids = set()
        if _DICT_DIR.exists():
            for f in _DICT_DIR.glob("*.mp4"):
                parts = f.stem.split("_", 1)
                if parts[0].isdigit():
                    _downloaded_ids.add(int(parts[0]))

        _with_trans = dict_df[dict_df["translation"].notna() & (dict_df["translation"].str.strip() != "")]
        _remaining = _with_trans[~_with_trans["word_id"].isin(_downloaded_ids)]

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Total in CSV", len(dict_df))
        mc2.metric("Downloaded", len(_downloaded_ids))
        mc3.metric("Remaining", len(_remaining))

        # Word class filter
        if "word_class" in _remaining.columns:
            wc_options = sorted(_remaining["word_class"].dropna().unique().tolist())
            selected_wc = st.multiselect(
                "Word classes to download",
                options=wc_options,
                default=[wc for wc in ["Noun", "Verb", "Adjective", "Adverb", "Numeral",
                                        "Pronoun", "Conjunction", "Preposition", "Interjection"]
                         if wc in wc_options],
                key="dict_word_classes",
            )

            if selected_wc:
                filtered = _remaining[_remaining["word_class"].isin(selected_wc)]
            else:
                filtered = _remaining
        else:
            filtered = _remaining

        st.caption(f"**{len(filtered)}** entries to download after filtering")

        scol1, scol2 = st.columns(2)
        with scol1:
            dict_workers = st.slider("Workers", 1, 16, 8, key="dict_workers")
        with scol2:
            dict_delay = st.slider("Delay per request (s)", 0.1, 2.0, 0.3, 0.1, key="dict_delay")

        dict_limit = st.number_input(
            "Max videos to download (0 = all)",
            min_value=0, value=0, step=500, key="dict_limit",
        )

        if st.button("📥 Download partner-dictnary Videos", type="primary", disabled=len(filtered) == 0):
            dl_df = filtered.copy()
            if dict_limit > 0:
                dl_df = dl_df.head(dict_limit)

            st.markdown(f"Downloading **{len(dl_df)}** videos → `{_DICT_DIR}`")
            prog = st.progress(0.0, text="Starting…")
            _ok = 0
            _fail = 0

            def _dict_progress(done, total, res):
                nonlocal _ok, _fail
                if res["status"] == DL_OK or res["status"] == DL_SKIP:
                    _ok += 1
                else:
                    _fail += 1
                pct = done / total
                prog.progress(pct, text=f"{done}/{total} — ✅ {_ok} ok, ❌ {_fail} fail")

            results = download_partner-dictnary_batch(
                dl_df, _DICT_DIR,
                n_workers=dict_workers,
                yt_dlp_cmd=".venv/bin/yt-dlp",
                progress_callback=_dict_progress,
                skip_existing=True,
                delay=dict_delay,
            )

            prog.progress(1.0, text="Done!")
            ok = sum(1 for r in results if r["status"] in (DL_OK, DL_SKIP))
            st.success(f"Downloaded **{ok}** / {len(results)} videos. "
                       f"Total in folder: ~{len(_downloaded_ids) + ok}")

            st.session_state.pop("inventory", None)
    else:
        st.info(
            "No partner-dictnary CSV found. "
            "Place `partner-dictnary_export*.csv` in `data/` to enable."
        )

# ================================================================== #
#  FTP tab
# ================================================================== #
with tab_ftp:
    st.markdown(
        "Download videos from an FTP server. "
        "Scans subdirectories (categories) and downloads video files in parallel."
    )

    ftp_url = st.text_input(
        "FTP base URL",
        placeholder="ftp://example.com/videos/",
        key="ftp_url",
    )
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        ftp_user = st.text_input("FTP username", key="ftp_user")
    with fcol2:
        ftp_pass = st.text_input("FTP password", type="password", key="ftp_pass")

    ftp_subfolder = st.text_input(
        "Save into subfolder of data/videos/",
        value="ftp_download",
        key="ftp_subfolder",
    )
    ftp_out_dir = VIDEO_DIR / ftp_subfolder

    ftp_workers = st.slider("Parallel workers", 1, 32, 8, key="ftp_workers")

    # Agreement check
    if ftp_url.strip():
        if not check_agreement(ftp_out_dir):
            st.warning(agreement_warning(ftp_subfolder))

    fcol_preview, fcol_download = st.columns(2)
    ftp_preview = fcol_preview.button("🔍 Preview FTP", disabled=not ftp_url.strip())
    ftp_download = fcol_download.button("⬇ Download FTP", type="primary", disabled=not ftp_url.strip())

    if ftp_preview and ftp_url.strip():
        with st.spinner("Listing FTP directory…"):
            try:
                entries = list_ftp_directory(ftp_url.strip(), ftp_user, ftp_pass)
                st.success(f"Found **{len(entries)}** entries in top-level directory")

                # Try to count files in subdirectories
                total_files = 0
                cat_info: list[dict] = []
                for entry in entries:
                    try:
                        sub = list_ftp_directory(
                            f"{ftp_url.strip().rstrip('/')}/{entry}",
                            ftp_user, ftp_pass,
                        )
                        vids = [f for f in sub if Path(f).suffix.lower() in VIDEO_EXTENSIONS]
                        if vids:
                            cat_info.append({"directory": entry, "videos": len(vids)})
                            total_files += len(vids)
                    except RuntimeError:
                        cat_info.append({"directory": entry, "videos": "?"})

                if cat_info:
                    st.dataframe(pd.DataFrame(cat_info), use_container_width=True, hide_index=True)
                    st.caption(f"**{total_files}** video files total across {len(cat_info)} directories")
            except Exception as exc:
                st.error(f"FTP listing failed: {exc}")

    if ftp_download and ftp_url.strip():
        st.markdown(f"Downloading from `{ftp_url.strip()}` → `{ftp_out_dir}`")
        prog = st.progress(0.0, text="Starting…")
        _ftp_ok = 0
        _ftp_skip = 0
        _ftp_fail = 0

        def _ftp_progress(done, total, res):
            nonlocal _ftp_ok, _ftp_skip, _ftp_fail
            s = res.get("status", "")
            if s == DL_OK:
                _ftp_ok += 1
            elif s == DL_SKIP:
                _ftp_skip += 1
            else:
                _ftp_fail += 1
            pct = done / total if total > 0 else 1.0
            prog.progress(pct, text=f"{done}/{total} — ✅ {_ftp_ok} new, ⏭ {_ftp_skip} skip, ❌ {_ftp_fail} fail")

        try:
            results = download_ftp_directory(
                ftp_url.strip(), ftp_out_dir,
                username=ftp_user, password=ftp_pass,
                n_workers=ftp_workers,
                skip_existing=True,
                progress_callback=_ftp_progress,
            )
            prog.progress(1.0, text="Done!")
            n_ok = sum(1 for r in results if r["status"] == DL_OK)
            n_skip = sum(1 for r in results if r["status"] == DL_SKIP)
            n_fail = sum(1 for r in results if r["status"] == DL_FAIL)
            st.success(f"FTP download complete — {n_ok} new, {n_skip} skipped, {n_fail} failed.")
            st.session_state.pop("inventory", None)
        except Exception as exc:
            st.error(f"FTP download failed: {exc}")

# ================================================================== #
#  HTTP tab
# ================================================================== #
with tab_http:
    st.markdown(
        "Download video files directly from HTTP/HTTPS URLs. "
        "One URL per line. Each URL should point to a video file."
    )

    http_urls_text = st.text_area(
        "URLs",
        height=140,
        placeholder="https://example.com/video1.mp4\nhttps://example.com/video2.mp4",
        label_visibility="collapsed",
        key="http_urls_text",
    )

    http_subfolder = st.text_input(
        "Save into subfolder of data/videos/",
        value="http_download",
        key="http_subfolder",
    )
    http_out_dir = VIDEO_DIR / http_subfolder

    http_workers = st.slider("Parallel workers", 1, 16, 4, key="http_workers")

    http_urls = [u.strip() for u in http_urls_text.splitlines() if u.strip() and not u.strip().startswith("#")]

    if http_urls:
        st.caption(f"{len(http_urls)} URL(s) entered")

        # Agreement check
        if not check_agreement(http_out_dir):
            st.warning(agreement_warning(http_subfolder))

    if st.button("⬇ Download HTTP", type="primary", disabled=len(http_urls) == 0):
        # Build (url, output_path) pairs
        pairs = []
        for url in http_urls:
            filename = Path(url.split("?")[0].split("#")[0]).name or "download.mp4"
            pairs.append((url, http_out_dir / filename))

        st.markdown(f"Downloading **{len(pairs)}** file(s) → `{http_out_dir}`")
        prog = st.progress(0.0, text="Starting…")
        _http_ok = 0
        _http_fail = 0

        def _http_progress(done, total, res):
            nonlocal _http_ok, _http_fail
            if res["status"] == DL_OK:
                _http_ok += 1
            else:
                _http_fail += 1
            pct = done / total if total > 0 else 1.0
            prog.progress(pct, text=f"{done}/{total} — ✅ {_http_ok} ok, ❌ {_http_fail} fail")

        results = download_http_batch(
            pairs,
            n_workers=http_workers,
            skip_existing=True,
            progress_callback=_http_progress,
        )
        prog.progress(1.0, text="Done!")
        ok = sum(1 for r in results if r["status"] == DL_OK)
        st.success(f"Downloaded **{ok}** / {len(results)} files.")
        st.session_state.pop("inventory", None)

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
