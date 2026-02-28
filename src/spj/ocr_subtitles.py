"""OCR-based subtitle extraction from hardsubbed sign-language videos.

Workflow:
  1. Detect subtitle region (bottom N % of frame).
  2. Sample every N frames (cheap pixel-diff check).
  3. When region changes: run OCR on the subtitle strip.
  4. Track when text changes → emit (start_ms, end_ms, text) entries.
  5. Save output as WebVTT alongside the video or in data/subtitles/.

Backend priority (auto-selected):
  1. Apple Vision via `ocrmac`  — macOS only, Neural Engine, best accuracy
  2. EasyOCR + MPS / CUDA       — GPU-accelerated on Apple Silicon / NVIDIA
  3. EasyOCR + CPU              — always available fallback

Falls back gracefully: ImportError is raised with a clear install instruction
so the UI can show a useful message.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Module-level EasyOCR reader cache — loading it is slow (~5 s), so we keep
# it alive between calls within the same process.
_reader_cache: dict[str, object] = {}

# Apple Vision language codes for ocrmac.
# Slovak (sk-SK) is NOT supported by Apple Vision — use Czech (cs-CZ) as fallback
# since Czech and Slovak share the same alphabet and are mutually intelligible.
_OCRMAC_LANG_MAP: dict[str, str] = {
    "sk": "cs-CZ",   # Slovak → Czech fallback (Apple Vision has no Slovak)
    "cs": "cs-CZ",
    "en": "en-US",
}


def _ocrmac_available() -> bool:
    """Return True if ocrmac (Apple Vision OCR) is importable."""
    try:
        from ocrmac.ocrmac import text_from_image as _  # noqa: F401
        return True
    except ImportError:
        return False


def _gpu_available() -> bool:
    """Return True if MPS (Apple Silicon) or CUDA GPU is usable by PyTorch."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return True
        if torch.cuda.is_available():
            return True
    except Exception:
        pass
    return False


def gpu_backend() -> str:
    """Return backend name for display.

    Priority: Apple Vision > EasyOCR+MPS > EasyOCR+CUDA > EasyOCR+CPU
    """
    if _ocrmac_available():
        return "Apple Vision"
    try:
        import torch
        if torch.backends.mps.is_available():
            return "MPS"
        if torch.cuda.is_available():
            return "CUDA"
    except Exception:
        pass
    return "CPU"


def ensure_reader(language: str = "sk") -> object | None:
    """Pre-warm (or return cached) OCR backend.

    When ocrmac is available (macOS), returns None immediately — Apple Vision
    needs no pre-warming; each call is stateless.

    Otherwise loads (or returns cached) EasyOCR reader with GPU auto-detection.
    Call this once before spawning parallel worker threads so all threads share
    the same already-loaded model rather than each downloading/loading it.
    """
    if _ocrmac_available():
        return None  # Apple Vision is stateless — nothing to pre-warm

    import warnings

    if language not in _reader_cache:
        import easyocr

        use_gpu = _gpu_available()
        logger.info("Loading EasyOCR reader lang=%s gpu=%s …", language, use_gpu)
        # Suppress the harmless 'pin_memory not supported on MPS' PyTorch warning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*pin_memory.*")
            _reader_cache[language] = easyocr.Reader(
                [language], gpu=use_gpu, verbose=False
            )
    return _reader_cache[language]


# ---------------------------------------------------------------------------
# VTT helpers
# ---------------------------------------------------------------------------

def _ms_to_vtt(ms: int) -> str:
    h   = ms // 3_600_000
    m   = (ms % 3_600_000) // 60_000
    s   = (ms % 60_000) // 1_000
    msr = ms % 1_000
    return f"{h:02d}:{m:02d}:{s:02d}.{msr:03d}"


def write_vtt(subtitles: list[dict], output_path: Path) -> None:
    """Write a list of {start_ms, end_ms, text} dicts to a WebVTT file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("WEBVTT\n\n")
        for i, sub in enumerate(subtitles, 1):
            fh.write(f"{i}\n")
            fh.write(f"{_ms_to_vtt(sub['start_ms'])} --> {_ms_to_vtt(sub['end_ms'])}\n")
            fh.write(f"{sub['text']}\n\n")


def read_vtt(vtt_path: Path) -> list[dict]:
    """Parse a WebVTT file into {start_ms, end_ms, text} dicts."""
    text = Path(vtt_path).read_text(encoding="utf-8")
    pattern = re.compile(
        r"(\d{2}:\d{2}:\d{2}\.\d{3})\s+-->\s+(\d{2}:\d{2}:\d{2}\.\d{3})\s*\n(.*?)(?=\n\n|\Z)",
        re.DOTALL,
    )
    results = []
    for m in pattern.finditer(text):
        start_ms = _vtt_to_ms(m.group(1))
        end_ms   = _vtt_to_ms(m.group(2))
        content  = m.group(3).strip()
        if content:
            results.append({"start_ms": start_ms, "end_ms": end_ms, "text": content})
    return results


def _vtt_to_ms(ts: str) -> int:
    h, m, rest = ts.split(":")
    s, ms = rest.split(".")
    return int(h) * 3_600_000 + int(m) * 60_000 + int(s) * 1_000 + int(ms)


# ---------------------------------------------------------------------------
# VTT discovery
# ---------------------------------------------------------------------------

def find_soft_vtt(video_path: Path) -> Path | None:
    """Return a yt-dlp soft-subtitle .vtt file alongside the video, if any.

    yt-dlp names subtitle files as:  <stem>.<lang>.vtt
    We prefer Slovak (sk) > Czech (cs) > any language.
    """
    vpath = Path(video_path)
    candidates = [c for c in vpath.parent.glob(f"{vpath.stem}*.vtt") if _vtt_has_cues(c)]
    for lang in ("sk", "cs", "en"):
        for c in candidates:
            if f".{lang}." in c.name or c.name.endswith(f".{lang}.vtt"):
                return c
    return candidates[0] if candidates else None


def ocr_vtt_path(video_path: Path, subtitles_dir: Path) -> Path:
    """Return the canonical OCR-output .vtt path for a given video."""
    return Path(subtitles_dir) / f"{Path(video_path).stem}.vtt"


def _vtt_has_cues(vtt_path: Path) -> bool:
    """Return True if a .vtt file has at least one timestamp cue."""
    try:
        return "-->" in vtt_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False


def get_subtitle_status(video_path: Path, subtitles_dir: Path) -> dict:
    """Return subtitle availability info for a video.

    Returns:
        {
          'has_soft':  bool,    # yt-dlp .vtt found next to video
          'has_ocr':   bool,    # OCR-generated .vtt in subtitles_dir
          'vtt_path':  Path | None,   # best available .vtt
          'source':    str,     # 'soft' | 'ocr' | 'none'
        }
    """
    soft = find_soft_vtt(video_path)
    ocr  = ocr_vtt_path(video_path, subtitles_dir)
    ocr_valid = ocr.exists() and _vtt_has_cues(ocr)
    if soft:
        return {"has_soft": True,  "has_ocr": ocr_valid, "vtt_path": soft, "source": "soft"}
    if ocr_valid:
        return {"has_soft": False, "has_ocr": True,       "vtt_path": ocr,  "source": "ocr"}
    return      {"has_soft": False, "has_ocr": False,      "vtt_path": None, "source": "none"}


# ---------------------------------------------------------------------------
# Apple Vision OCR helper
# ---------------------------------------------------------------------------

def _ocr_apple_vision(roi_bgr: np.ndarray, language: str) -> str:
    """Run Apple Vision OCR (via ocrmac) on a BGR image region.

    Apple Vision handles its own preprocessing — pass the raw colour strip.
    Thread-safe: each call creates an independent Vision request.
    """
    texts = [r[0] for r in _ocr_apple_vision_raw(roi_bgr, language) if r[0].strip()]
    return _clean_ocr_text(" ".join(texts))


def _ocr_apple_vision_raw(roi_bgr: np.ndarray, language: str) -> list[tuple[str, float, tuple]]:
    """Run Apple Vision OCR and return raw results with bounding boxes.

    Returns list of (text, confidence, (x, y, w, h)) tuples.
    Y coordinate is in Core Graphics convention: 0=bottom, 1=top of the ROI.
    """
    from PIL import Image
    from ocrmac.ocrmac import text_from_image

    rgb     = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    lang_code = _OCRMAC_LANG_MAP.get(language, "cs-CZ")
    return text_from_image(
        pil_img,
        language_preference=[lang_code],
        recognition_level="accurate",
    )


def _split_bilingual(results: list[tuple[str, float, tuple]]) -> tuple[str, str]:
    """Split OCR results into upper (primary) and lower (secondary) text by Y position.

    Apple Vision uses Core Graphics coords: y=0 is bottom, y=1 is top.
    Upper subtitle (higher Y) = primary language (Slovak).
    Lower subtitle (lower Y)  = secondary language (English).

    Returns (primary_text, secondary_text). Either may be empty.
    """
    if len(results) < 2:
        text = " ".join(r[0] for r in results if r[0].strip())
        return (_clean_ocr_text(text), "")

    # Sort by Y descending (top of image first in CG coords = higher Y)
    sorted_results = sorted(results, key=lambda r: r[2][1], reverse=True)
    ys = [r[2][1] for r in sorted_results]

    # Find the biggest gap between consecutive Y values
    max_gap = 0.0
    split_idx = len(sorted_results)
    for i in range(len(ys) - 1):
        gap = ys[i] - ys[i + 1]
        if gap > max_gap:
            max_gap = gap
            split_idx = i + 1

    # Only split if gap is significant (> 0.15 of region height)
    if max_gap < 0.15:
        text = " ".join(r[0] for r in results if r[0].strip())
        return (_clean_ocr_text(text), "")

    upper = [r[0] for r in sorted_results[:split_idx] if r[0].strip()]
    lower = [r[0] for r in sorted_results[split_idx:] if r[0].strip()]
    return (_clean_ocr_text(" ".join(upper)), _clean_ocr_text(" ".join(lower)))


# ---------------------------------------------------------------------------
# Frame preprocessing  (used by EasyOCR path only)
# ---------------------------------------------------------------------------

def _crop_region(frame: np.ndarray, region_fraction: float, region_anchor: str) -> np.ndarray:
    """Return the subtitle strip from a frame.

    Args:
        region_anchor: "bottom" — scan bottom N% (default).
                       "top"    — scan top N% (some videos put subs at top).
    """
    h = frame.shape[0]
    if region_anchor == "top":
        return frame[:int(h * region_fraction), :]
    return frame[int(h * (1.0 - region_fraction)):, :]


def _binarize_roi(roi_bgr: np.ndarray) -> np.ndarray:
    """Binarise a subtitle strip for EasyOCR.

    Not used by the Apple Vision path — Apple Vision does its own preprocessing.

    Handles two common hardsub styles:
      • White text on dark background  (common in Slovak educational videos)
      • Dark text on white/yellow bar  (less common)
    """
    w    = roi_bgr.shape[1]
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    # Decide polarity from the region's average brightness
    if gray.mean() < 100:
        # Dark bg, white text — binary threshold at high value
        _, binary = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
    elif gray.mean() > 160:
        # Light bg, dark text — invert so text becomes white
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    else:
        # Mid-tone — use Otsu's adaptive threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Upscale 2× — EasyOCR works better on larger images
    return cv2.resize(binary, (w * 2, binary.shape[0] * 2), interpolation=cv2.INTER_CUBIC)


def _preprocess_region(frame: np.ndarray, region_fraction: float,
                       region_anchor: str = "bottom") -> np.ndarray:
    """Crop and binarise a frame region for EasyOCR."""
    roi = _crop_region(frame, region_fraction, region_anchor)
    return _binarize_roi(roi)


def _text_changed(current: str, new: str, threshold: float = 0.80) -> bool:
    """Return True when the new text is substantially different from current."""
    if not current and not new:
        return False
    if not current or not new:
        return True
    # Simple character-overlap ratio
    longer  = max(len(current), len(new))
    matches = sum(c1 == c2 for c1, c2 in zip(current, new))
    return (matches / longer) < threshold


def _clean_ocr_text(text: str) -> str:
    """Remove common OCR noise from subtitle lines."""
    # Collapse whitespace
    text = " ".join(text.split())
    # Drop lines that are purely punctuation / single characters
    if len(text) < 2:
        return ""
    return text


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_subtitles(
    video_path: Path,
    output_vtt: Path,
    progress_callback: Optional[Callable[[float], None]] = None,
    region_fraction: float = 0.22,
    region_anchor: str = "bottom",
    diff_sample_n: int = 3,
    diff_threshold: float = 6.0,
    min_duration_ms: int = 400,
    change_threshold: float = 0.80,
    language: str = "sk",
    bilingual: bool = True,
) -> list[dict]:
    """Extract hardcoded subtitles from a video.

    Backend is auto-selected (Apple Vision > EasyOCR+MPS/CUDA > EasyOCR+CPU).

    Speed strategy — two-level sampling:
      1. Every `diff_sample_n` frames: compute mean pixel diff in the subtitle
         region (cheap numpy op, ~0.001 s/frame).
      2. Only when diff > `diff_threshold`: run OCR (expensive).

    For a 60-minute video at 25 fps this reduces OCR calls from ~18,000
    (naïve every-5-frame approach) to ~300–600 (just subtitle changes).

    Args:
        video_path:        Input video file.
        output_vtt:        Destination WebVTT file.
        progress_callback: Called with a float in [0, 1].
        region_fraction:   Fraction of frame height scanned.
        region_anchor:     "bottom" (default) or "top" — which edge to scan from.
        diff_sample_n:     Check pixel diff every N frames (3 ≈ 0.12 s at 25 fps).
        diff_threshold:    Mean absolute pixel diff (0–255) that triggers OCR.
                           ~6 works well for clean hardsubs; raise if too noisy.
        min_duration_ms:   Discard subtitle entries shorter than this.
        change_threshold:  Character-overlap ratio below which OCR result is
                           treated as new text (handles minor frame-to-frame noise).
        language:          Language code — 'sk' Slovak (default), 'cs', 'en'.
        bilingual:         When True and using Apple Vision, split dual-subtitle
                           videos into primary (upper) and secondary (lower) VTT
                           files.  Secondary file is written as <stem>.en.vtt.

    Returns:
        List of {start_ms, end_ms, text} dicts for the PRIMARY language
        (also written to output_vtt).  Secondary language written separately.

    Raises:
        ImportError: If neither ocrmac nor EasyOCR is installed.
        RuntimeError: If the video cannot be opened.
    """
    use_apple = _ocrmac_available()
    reader = None

    if not use_apple:
        bilingual = False  # bilingual split requires Apple Vision bbox data
        try:
            import easyocr  # noqa: F401
        except ImportError:
            raise ImportError(
                "No OCR backend available.  Run:  uv sync\n"
                "  • macOS: installs ocrmac (Apple Vision — fastest)\n"
                "  • all:   installs easyocr (GPU/CPU fallback)"
            )
        reader = ensure_reader(language)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    # Auto-increase region for portrait videos (9:16) — subtitles sit higher
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_h > frame_w and region_fraction < 0.35:
        logger.info(
            "Portrait video (%dx%d) — increasing region from %.0f%% to 35%%",
            frame_w, frame_h, region_fraction * 100,
        )
        region_fraction = 0.35

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    # Primary (upper / Slovak) and secondary (lower / English) subtitle streams
    subtitles: list[dict] = []
    subtitles_secondary: list[dict] = []
    current_text  = ""
    current_start = 0
    current_text_2  = ""
    current_start_2 = 0
    has_bilingual_data = False
    prev_gray: Optional[np.ndarray] = None
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            ts_ms = int(frame_idx * 1000 / fps)

            if frame_idx % diff_sample_n == 0:
                # --- cheap region diff check ---
                roi  = _crop_region(frame, region_fraction, region_anchor)
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                if prev_gray is None:
                    diff = diff_threshold + 1.0   # force OCR on first frame
                else:
                    diff = float(np.mean(np.abs(gray.astype(np.int16) - prev_gray.astype(np.int16))))

                prev_gray = gray

                if diff > diff_threshold:
                    text = ""
                    text_2 = ""
                    # --- expensive OCR only when region changed ---
                    try:
                        if use_apple and bilingual:
                            raw = _ocr_apple_vision_raw(roi, language)
                            text, text_2 = _split_bilingual(raw)
                            if text_2:
                                has_bilingual_data = True
                        elif use_apple:
                            text = _ocr_apple_vision(roi, language)
                        else:
                            # EasyOCR: binarise for better accuracy
                            processed = _binarize_roi(roi)
                            lines = reader.readtext(processed, detail=0, paragraph=True)
                            text  = _clean_ocr_text(" ".join(lines))
                    except Exception:
                        text = ""
                        text_2 = ""

                    # Primary stream
                    if _text_changed(current_text, text, change_threshold):
                        if current_text and (ts_ms - current_start) >= min_duration_ms:
                            subtitles.append({
                                "start_ms": current_start,
                                "end_ms":   ts_ms,
                                "text":     current_text,
                            })
                        current_text  = text
                        current_start = ts_ms

                    # Secondary stream (bilingual)
                    if bilingual and _text_changed(current_text_2, text_2, change_threshold):
                        if current_text_2 and (ts_ms - current_start_2) >= min_duration_ms:
                            subtitles_secondary.append({
                                "start_ms": current_start_2,
                                "end_ms":   ts_ms,
                                "text":     current_text_2,
                            })
                        current_text_2  = text_2
                        current_start_2 = ts_ms

                if progress_callback:
                    progress_callback(min(frame_idx / total_frames, 0.99))

            frame_idx += 1

    finally:
        cap.release()

    last_ms = int(frame_idx * 1000 / fps)
    if current_text and (last_ms - current_start) >= min_duration_ms:
        subtitles.append({"start_ms": current_start, "end_ms": last_ms, "text": current_text})
    if current_text_2 and (last_ms - current_start_2) >= min_duration_ms:
        subtitles_secondary.append({"start_ms": current_start_2, "end_ms": last_ms, "text": current_text_2})

    subtitles = [s for s in subtitles if s["text"].strip()]
    write_vtt(subtitles, output_vtt)

    # Write secondary language VTT if bilingual data was detected
    if has_bilingual_data and subtitles_secondary:
        subtitles_secondary = [s for s in subtitles_secondary if s["text"].strip()]
        secondary_vtt = output_vtt.parent / f"{output_vtt.stem}.en.vtt"
        write_vtt(subtitles_secondary, secondary_vtt)
        logger.info(
            "Bilingual: %d primary + %d secondary entries from %s",
            len(subtitles), len(subtitles_secondary), video_path.name,
        )
    else:
        logger.info("Extracted %d subtitle entries from %s", len(subtitles), video_path.name)

    if progress_callback:
        progress_callback(1.0)

    return subtitles


# ---------------------------------------------------------------------------
# Debug helper
# ---------------------------------------------------------------------------

def extract_debug_frame(
    video_path: Path,
    time_sec: float = 10.0,
    region_fraction: float = 0.22,
    region_anchor: str = "bottom",
    language: str = "sk",
) -> dict:
    """Extract one frame and run OCR on it — for diagnosing empty results.

    Returns a dict with RGB images and OCR text so the caller can display
    them directly with st.image():
        {
          "annotated_rgb":  np.ndarray  — full frame with the scanned region boxed in green
          "roi_rgb":        np.ndarray  — subtitle strip (colour)
          "preprocessed":   np.ndarray  — binary strip (what EasyOCR sees; None for Apple Vision)
          "ocr_text":       str
          "frame_size":     (w, h)
          "region_y":       (y0, y1)
          "error":          str | None
        }
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": f"Cannot open: {Path(video_path).name}"}

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frame = min(int(time_sec * fps), max(0, total_frames - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return {"error": f"Could not read frame at {time_sec:.0f}s"}

    h, w = frame.shape[:2]

    # Region boundaries
    if region_anchor == "top":
        y0, y1 = 0, int(h * region_fraction)
    else:
        y0, y1 = int(h * (1.0 - region_fraction)), h

    roi_bgr      = frame[y0:y1, :]
    preprocessed = _binarize_roi(roi_bgr)  # for display even on Apple Vision path

    # OCR
    try:
        if _ocrmac_available():
            text = _ocr_apple_vision(roi_bgr, language)
        else:
            reader = ensure_reader(language)
            lines  = reader.readtext(preprocessed, detail=0, paragraph=True)
            text   = _clean_ocr_text(" ".join(lines))
    except Exception as exc:
        text = f"(OCR error: {exc})"

    # Annotate frame: green rectangle + label
    annotated = frame.copy()
    cv2.rectangle(annotated, (0, y0), (w - 1, y1 - 1), (0, 220, 0), 3)
    label = f"region: {region_anchor}  {region_fraction:.0%}"
    cv2.putText(annotated, label,
                (8, max(y0 - 6, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2)

    return {
        "annotated_rgb": cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
        "roi_rgb":       cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB),
        "preprocessed":  preprocessed,
        "ocr_text":      text,
        "frame_size":    (w, h),
        "region_y":      (y0, y1),
        "error":         None,
    }
