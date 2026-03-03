"""Transcribe audio from video files using faster-whisper.

Extracts Slovak/Czech speech from video audio tracks and outputs
VTT subtitle files for use in the training pipeline.

Usage:
    from spj.transcribe import transcribe_video, transcribe_batch

    # Single video
    vtt_path = transcribe_video(Path("data/videos/vhs/video01.mp4"))

    # Batch
    results = transcribe_batch(Path("data/videos/vhs/"), language="sk")
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _extract_audio(video_path: Path, audio_path: Path) -> bool:
    """Extract audio track from video to WAV (16kHz mono)."""
    try:
        subprocess.run(
            [
                "ffmpeg", "-i", str(video_path),
                "-vn",                    # no video
                "-acodec", "pcm_s16le",   # 16-bit PCM
                "-ar", "16000",           # 16kHz (Whisper optimal)
                "-ac", "1",               # mono
                "-y",                     # overwrite
                str(audio_path),
            ],
            capture_output=True,
            timeout=300,
        )
        return audio_path.exists() and audio_path.stat().st_size > 0
    except Exception as e:
        logger.error(f"Audio extraction failed for {video_path.name}: {e}")
        return False


def _format_timestamp(seconds: float) -> str:
    """Format seconds as VTT timestamp HH:MM:SS.mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def transcribe_video(
    video_path: Path,
    output_dir: Optional[Path] = None,
    language: Optional[str] = None,
    model_size: str = "large-v3",
    device: str = "auto",
    beam_size: int = 5,
    word_timestamps: bool = False,
) -> Optional[Path]:
    """Transcribe a single video file, outputting a VTT subtitle file.

    Args:
        video_path: Path to video file (mp4, mkv, avi, etc.)
        output_dir: Where to save VTT. Defaults to same dir as video.
        language: Force language ("sk", "cs", etc.). None = auto-detect.
        model_size: Whisper model size. "large-v3" is best for Slovak.
        device: "auto", "cpu", or "cuda".
        beam_size: Beam search width.
        word_timestamps: Whether to include word-level timestamps.

    Returns:
        Path to generated VTT file, or None on failure.
    """
    from faster_whisper import WhisperModel

    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        return None

    if output_dir is None:
        output_dir = video_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    lang_suffix = language or "auto"
    vtt_path = output_dir / f"{video_path.stem}.{lang_suffix}.vtt"

    # Skip if already transcribed
    if vtt_path.exists() and vtt_path.stat().st_size > 0:
        logger.info(f"Already transcribed: {vtt_path.name}")
        return vtt_path

    logger.info(f"Transcribing {video_path.name} (model={model_size}, lang={language or 'auto'})...")

    # Extract audio to temp WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp_path = Path(tmp.name)

    if not _extract_audio(video_path, tmp_path):
        logger.error(f"Failed to extract audio from {video_path.name}")
        return None

    try:
        # Load model
        if device == "auto":
            device = "cpu"  # MPS not supported by CTranslate2, use CPU

        model = WhisperModel(
            model_size,
            device=device,
            compute_type="int8",  # fast on CPU
        )

        # Transcribe
        segments, info = model.transcribe(
            str(tmp_path),
            language=language,
            beam_size=beam_size,
            word_timestamps=word_timestamps,
            vad_filter=True,  # skip silence
        )

        detected_lang = info.language
        lang_prob = info.language_probability
        logger.info(f"Detected language: {detected_lang} ({lang_prob:.0%})")

        # Update VTT path with detected language if auto
        if language is None:
            vtt_path = output_dir / f"{video_path.stem}.{detected_lang}.vtt"

        # Write VTT
        with open(vtt_path, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for i, seg in enumerate(segments, 1):
                start = _format_timestamp(seg.start)
                end = _format_timestamp(seg.end)
                text = seg.text.strip()
                if text:
                    f.write(f"{i}\n")
                    f.write(f"{start} --> {end}\n")
                    f.write(f"{text}\n\n")

        logger.info(f"Saved: {vtt_path.name}")
        return vtt_path

    except Exception as e:
        logger.error(f"Transcription failed for {video_path.name}: {e}")
        return None
    finally:
        # Cleanup temp audio
        if tmp_path.exists():
            tmp_path.unlink()


def transcribe_batch(
    video_dir: Path,
    output_dir: Optional[Path] = None,
    language: Optional[str] = None,
    model_size: str = "large-v3",
    extensions: tuple[str, ...] = (".mp4", ".mkv", ".avi", ".mov", ".webm"),
    progress_callback=None,
) -> list[dict]:
    """Transcribe all videos in a directory.

    Args:
        video_dir: Directory containing video files.
        output_dir: Where to save VTTs. Defaults to same as video_dir.
        language: Force language or None for auto-detect.
        model_size: Whisper model size.
        extensions: Video file extensions to process.
        progress_callback: Optional fn(done, total, result_dict) called per video.

    Returns:
        List of result dicts with keys: video, vtt_path, success, language, error
    """
    videos = sorted(
        f for f in video_dir.iterdir()
        if f.suffix.lower() in extensions
    )

    if not videos:
        logger.warning(f"No video files found in {video_dir}")
        return []

    logger.info(f"Transcribing {len(videos)} videos from {video_dir}")
    results = []

    for i, video in enumerate(videos):
        try:
            vtt = transcribe_video(
                video,
                output_dir=output_dir,
                language=language,
                model_size=model_size,
            )
            result = {
                "video": video.name,
                "vtt_path": str(vtt) if vtt else None,
                "success": vtt is not None,
                "error": None,
            }
        except Exception as e:
            result = {
                "video": video.name,
                "vtt_path": None,
                "success": False,
                "error": str(e),
            }

        results.append(result)
        if progress_callback:
            progress_callback(i + 1, len(videos), result)

    ok = sum(1 for r in results if r["success"])
    logger.info(f"Transcribed {ok}/{len(videos)} videos")
    return results
