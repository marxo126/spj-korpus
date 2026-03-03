"""Scrape spravy.spj.sk articles via WP REST API.

Each article has an embedded YouTube video (sign language) + Slovak text.
This gives us text-to-video alignment for training without needing YouTube subtitles.

Usage:
    from spj.spravy_scraper import scrape_all_articles, save_articles_csv
    articles = scrape_all_articles()
    save_articles_csv(articles, Path("data/spravy_spj_articles.csv"))
"""

import re
import time
import logging
from html import unescape
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

WP_API = "https://spravy.spj.sk/wp-json/wp/v2/posts"
PER_PAGE = 50  # max allowed by WP REST API


def _strip_html(html: str) -> str:
    """Remove HTML tags and decode entities, preserving paragraph breaks."""
    # Replace block-level tags with newlines
    text = re.sub(r"<(?:br|/p|/div|/h[1-6]|/li)[^>]*>", "\n", html, flags=re.I)
    # Remove all remaining tags
    text = re.sub(r"<[^>]+>", "", text)
    # Decode HTML entities
    text = unescape(text)
    # Collapse whitespace but keep paragraph breaks
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    return text.strip()


def _extract_youtube_id(html: str) -> Optional[str]:
    """Extract YouTube video ID from embedded iframe in post content."""
    # Pattern: youtube.com/embed/VIDEO_ID
    match = re.search(r"youtube\.com/embed/([A-Za-z0-9_-]{11})", html)
    if match:
        return match.group(1)
    # Fallback: youtube.com/watch?v=VIDEO_ID
    match = re.search(r"youtube\.com/watch\?v=([A-Za-z0-9_-]{11})", html)
    if match:
        return match.group(1)
    # Fallback: youtu.be/VIDEO_ID
    match = re.search(r"youtu\.be/([A-Za-z0-9_-]{11})", html)
    if match:
        return match.group(1)
    return None


def scrape_all_articles(
    delay: float = 0.5,
    max_pages: int = 100,
) -> list[dict]:
    """Scrape all articles from spravy.spj.sk via WP REST API.

    Returns list of dicts with keys:
        post_id, title, date, slug, youtube_id, text, url, categories
    """
    articles = []
    page = 1

    while page <= max_pages:
        logger.info(f"Fetching page {page}...")
        try:
            resp = requests.get(
                WP_API,
                params={"per_page": PER_PAGE, "page": page},
                timeout=30,
            )
            if resp.status_code == 400:
                # Past last page
                break
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Request failed on page {page}: {e}")
            break

        posts = resp.json()
        if not posts:
            break

        for post in posts:
            content_html = post.get("content", {}).get("rendered", "")
            title = _strip_html(post.get("title", {}).get("rendered", ""))
            text = _strip_html(content_html)
            youtube_id = _extract_youtube_id(content_html)

            articles.append({
                "post_id": post.get("id"),
                "title": title,
                "date": post.get("date", "")[:10],
                "slug": post.get("slug", ""),
                "youtube_id": youtube_id,
                "text": text,
                "url": post.get("link", ""),
                "categories": ",".join(str(c) for c in post.get("categories", [])),
            })

        total_pages = int(resp.headers.get("X-WP-TotalPages", max_pages))
        total_posts = int(resp.headers.get("X-WP-Total", 0))

        if page == 1:
            logger.info(f"Total posts: {total_posts}, pages: {total_pages}")

        if page >= total_pages:
            break

        page += 1
        time.sleep(delay)

    logger.info(f"Scraped {len(articles)} articles")
    return articles


def save_articles_csv(articles: list[dict], output_path: Path) -> Path:
    """Save scraped articles to CSV."""
    df = pd.DataFrame(articles)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} articles to {output_path}")

    # Stats
    with_video = df["youtube_id"].notna().sum()
    with_text = (df["text"].str.len() > 50).sum()
    logger.info(f"  With YouTube video: {with_video}")
    logger.info(f"  With text (>50 chars): {with_text}")

    return output_path


def match_videos_to_articles(
    articles_csv: Path,
    video_dir: Path,
) -> pd.DataFrame:
    """Match downloaded SpravySPJ videos to their article texts.

    Returns DataFrame with columns: video_file, youtube_id, title, date, text
    """
    df = pd.read_csv(articles_csv)

    # Build youtube_id → article mapping
    id_to_article = {}
    for _, row in df.iterrows():
        if pd.notna(row.get("youtube_id")):
            id_to_article[row["youtube_id"]] = row

    # Match video files
    matches = []
    unmatched_videos = []

    for mp4 in sorted(video_dir.glob("*.mp4")):
        # Filename: YYYYMMDD_VIDEOID_Title.mp4
        parts = mp4.stem.split("_", 2)
        if len(parts) >= 2:
            vid_id = parts[1]
            if vid_id in id_to_article:
                art = id_to_article[vid_id]
                matches.append({
                    "video_file": mp4.name,
                    "youtube_id": vid_id,
                    "title": art.get("title", ""),
                    "date": art.get("date", ""),
                    "text": art.get("text", ""),
                    "text_len": len(str(art.get("text", ""))),
                })
            else:
                unmatched_videos.append(mp4.name)

    logger.info(f"Matched {len(matches)} videos to articles, {len(unmatched_videos)} unmatched")
    return pd.DataFrame(matches)
