"""EAF read/write utilities using pympi-ling."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pympi

# Exact tier names — never vary these
HUMAN_TIERS = [
    "S1_Gloss_RH",
    "S1_Gloss_LH",
    "S1_Translation",
    "S1_Mouthing",
    "S1_Mouth_Gesture",
    "S1_NonManual",
]
AI_TIERS = [
    "AI_Gloss_RH",
    "AI_Gloss_LH",
    "AI_Confidence",
]
ALL_TIERS = HUMAN_TIERS + AI_TIERS


def create_empty_eaf(video_path: Path, output_path: Path) -> pympi.Eaf:
    """Create a new EAF file with all standard tiers linked to the given video.

    Args:
        video_path: Absolute path to the source video file.
        output_path: Where the .eaf will be saved.

    Returns:
        pympi.Eaf instance (not yet saved to disk — call save_eaf separately).
    """
    eaf = pympi.Eaf()
    # Link the media file
    eaf.add_linked_file(
        file_path=str(video_path.resolve()),
        mimetype=_mimetype(video_path),
    )
    # Add human annotation tiers
    for tier in HUMAN_TIERS:
        eaf.add_tier(tier)
    # Add AI suggestion tiers
    for tier in AI_TIERS:
        eaf.add_tier(tier)
    return eaf


def add_ai_annotation(
    eaf: pympi.Eaf,
    tier: str,
    start_ms: int,
    end_ms: int,
    value: str,
) -> None:
    """Add a single annotation to an AI tier.

    Args:
        eaf: The Eaf object to modify.
        tier: Must be one of AI_TIERS.
        start_ms: Annotation start time in milliseconds.
        end_ms: Annotation end time in milliseconds.
        value: Annotation value string.

    Raises:
        ValueError: If tier is not in AI_TIERS.
    """
    if tier not in AI_TIERS:
        raise ValueError(f"tier must be one of {AI_TIERS}, got {tier!r}")
    eaf.add_annotation(tier, start_ms, end_ms, value=value)


def save_eaf(eaf: pympi.Eaf, path: Path) -> None:
    """Save an EAF object to disk.

    Args:
        eaf: The Eaf object to save.
        path: Destination path (parent directory must exist).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    eaf.to_file(str(path))


def load_eaf(path: Path) -> pympi.Eaf:
    """Load an existing EAF file.

    Args:
        path: Path to the .eaf file.

    Returns:
        pympi.Eaf instance.
    """
    return pympi.Eaf(str(path))


def get_tier_stats(eaf: pympi.Eaf) -> dict[str, int]:
    """Count annotations per tier.

    Args:
        eaf: The Eaf object to inspect.

    Returns:
        Dict mapping tier name → annotation count.
        Tiers with zero annotations are included if they are standard tiers.
    """
    stats: dict[str, int] = {}
    present_tiers = eaf.get_tier_names()
    for tier in ALL_TIERS:
        if tier in present_tiers:
            stats[tier] = len(eaf.get_annotation_data_for_tier(tier))
        else:
            stats[tier] = 0
    return stats


def _mimetype(path: Path) -> str:
    ext = path.suffix.lower()
    mime_map = {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
    }
    return mime_map.get(ext, "video/mp4")
