"""Ollama integration — translate predicted sign language glosses to Slovak.

Uses a locally-running Ollama instance to convert gloss sequences into
natural Slovak sentences, and to suggest alternative glosses for uncertain
predictions (active learning support).
"""
from __future__ import annotations

import json
import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)

OLLAMA_BASE = "http://localhost:11434"
_TIMEOUT = 60  # seconds per request


# ---------------------------------------------------------------------------
# Ollama status
# ---------------------------------------------------------------------------

def check_ollama() -> tuple[bool, str]:
    """Ping Ollama API. Returns (available, version_or_error)."""
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/version", timeout=5)
        if r.ok:
            data = r.json()
            return True, data.get("version", "unknown")
        return False, f"HTTP {r.status_code}"
    except requests.ConnectionError:
        return False, "Connection refused — is Ollama running? (ollama serve)"
    except Exception as exc:
        return False, str(exc)


def list_models() -> list[dict]:
    """GET /api/tags — list of locally available models.

    Returns list of dicts with at least 'name' and 'size' keys.
    """
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=10)
        if r.ok:
            data = r.json()
            models = data.get("models", [])
            return [
                {
                    "name": m.get("name", "unknown"),
                    "size": m.get("size", 0),
                    "modified_at": m.get("modified_at", ""),
                }
                for m in models
            ]
        return []
    except Exception as exc:
        logger.warning("Failed to list Ollama models: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

def translate_glosses(
    glosses: list[str],
    model: str = "llama3.2",
    source_lang: str = "Slovak Sign Language",
    target_lang: str = "Slovak",
) -> str:
    """Send gloss sequence to Ollama, get natural language translation.

    Args:
        glosses: List of sign language glosses (e.g. ["HELLO", "HOW", "YOU"]).
        model: Ollama model name.
        source_lang: Source language description.
        target_lang: Target language for translation.

    Returns:
        Translated sentence in target language.
    """
    if not glosses:
        return ""

    gloss_str = " ".join(glosses)
    prompt = (
        f"You are a sign language translation assistant. "
        f"Translate this {source_lang} gloss sequence into a natural {target_lang} sentence. "
        f"Glosses are written in uppercase and represent individual signs. "
        f"Create a grammatically correct {target_lang} sentence that conveys the meaning.\n\n"
        f"Gloss sequence: {gloss_str}\n\n"
        f"Translation ({target_lang}):"
    )

    return _ollama_generate(prompt, model)


def batch_translate_predictions(
    predictions: list[dict],
    model: str = "llama3.2",
    window_ms: int = 5000,
) -> list[dict]:
    """Group predictions by time window, translate each group.

    Args:
        predictions: List of prediction dicts from predict_segments().
        model: Ollama model name.
        window_ms: Time window in ms for grouping glosses.

    Returns:
        Predictions with added 'translation' field on the first item of each group.
    """
    if not predictions:
        return predictions

    # Sort by start time
    sorted_preds = sorted(predictions, key=lambda p: p["start_ms"])

    # Group by time windows
    groups: list[list[int]] = []  # indices into sorted_preds
    current_group: list[int] = [0]
    group_end = sorted_preds[0]["start_ms"] + window_ms

    for i in range(1, len(sorted_preds)):
        if sorted_preds[i]["start_ms"] <= group_end:
            current_group.append(i)
        else:
            groups.append(current_group)
            current_group = [i]
            group_end = sorted_preds[i]["start_ms"] + window_ms
    groups.append(current_group)

    # Translate each group
    result = [dict(p) for p in sorted_preds]  # copy
    for group_indices in groups:
        glosses = [sorted_preds[i]["predicted_gloss"] for i in group_indices]
        try:
            translation = translate_glosses(glosses, model=model)
        except Exception as exc:
            logger.warning("Translation failed for group: %s", exc)
            translation = f"[Translation error: {exc}]"

        # Attach translation to first prediction in group
        result[group_indices[0]]["translation"] = translation

    return result


# ---------------------------------------------------------------------------
# Active learning support
# ---------------------------------------------------------------------------

def suggest_corrections(
    gloss: str,
    confidence: float,
    context_glosses: list[str],
    model: str = "llama3.2",
) -> list[str]:
    """Ask LLM to suggest alternative glosses when confidence is low.

    Useful for active learning — flag uncertain predictions and get
    linguistically plausible alternatives.

    Args:
        gloss: Current predicted gloss.
        confidence: Prediction confidence (0-1).
        context_glosses: Surrounding glosses for context.
        model: Ollama model name.

    Returns:
        List of suggested alternative glosses (up to 5).
    """
    context_str = " ".join(context_glosses) if context_glosses else "(no context)"
    prompt = (
        f"You are a sign language annotation assistant for Slovak Sign Language. "
        f"The AI predicted the gloss '{gloss}' with {confidence:.0%} confidence. "
        f"Context (surrounding glosses): {context_str}\n\n"
        f"Suggest up to 5 alternative glosses that might be correct instead. "
        f"Write each suggestion on a new line, uppercase, just the gloss name. "
        f"If the prediction seems correct, write 'CORRECT' as the first suggestion.\n\n"
        f"Suggestions:"
    )

    response = _ollama_generate(prompt, model)

    # Parse suggestions — one per line, uppercase
    suggestions = []
    for line in response.strip().split("\n"):
        line = line.strip().strip("-").strip("•").strip().upper()
        if line and len(line) < 50:  # sanity check
            suggestions.append(line)
    return suggestions[:5]


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _ollama_generate(prompt: str, model: str) -> str:
    """Call Ollama generate API and return the response text."""
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 256,
                },
            },
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        return data.get("response", "").strip()
    except requests.ConnectionError:
        raise ConnectionError(
            "Cannot connect to Ollama. Start it with: ollama serve"
        )
    except requests.Timeout:
        raise TimeoutError(f"Ollama request timed out after {_TIMEOUT}s")
    except Exception as exc:
        raise RuntimeError(f"Ollama API error: {exc}") from exc
