"""MLX-LM translation — translate sign language glosses to Slovak on Apple Silicon.

Uses mlx-lm to run language models natively on M4 Max unified memory.
No server process needed — loads model directly into Metal GPU memory.
Faster and more efficient than Ollama on Apple Silicon.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default model — small, fast, multilingual, 4-bit quantized for M4 Max
DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"

# Cache loaded model to avoid reloading on every call
_cached_model: tuple | None = None  # (model, tokenizer, model_id)


# ---------------------------------------------------------------------------
# MLX availability
# ---------------------------------------------------------------------------

def check_mlx() -> tuple[bool, str]:
    """Check if mlx-lm is installed and usable. Returns (available, info_or_error)."""
    try:
        import mlx.core as mx
        import mlx_lm
        return True, f"mlx-lm installed, Metal backend"
    except ImportError:
        return False, "mlx-lm not installed. Install with: pip install mlx-lm"
    except Exception as exc:
        return False, str(exc)


def list_recommended_models() -> list[dict]:
    """Return list of recommended MLX models for translation.

    These are pre-quantized models from mlx-community on Hugging Face,
    optimized for Apple Silicon.
    """
    return [
        {
            "name": "mlx-community/Llama-3.2-3B-Instruct-4bit",
            "size": "~2 GB",
            "description": "Fast, multilingual, good for translation",
        },
        {
            "name": "mlx-community/Llama-3.2-1B-Instruct-4bit",
            "size": "~0.7 GB",
            "description": "Smallest, fastest, less accurate",
        },
        {
            "name": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
            "size": "~4 GB",
            "description": "Larger, more accurate, slower",
        },
        {
            "name": "mlx-community/Qwen2.5-7B-Instruct-4bit",
            "size": "~4 GB",
            "description": "Strong multilingual, good for Slovak",
        },
    ]


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------

def load_mlx_model(model_id: str = DEFAULT_MODEL) -> tuple:
    """Load an MLX model and tokenizer. Cached — only loads once per model_id.

    Args:
        model_id: Hugging Face model ID (e.g. 'mlx-community/Llama-3.2-3B-Instruct-4bit')

    Returns:
        (model, tokenizer) tuple
    """
    global _cached_model

    if _cached_model is not None and _cached_model[2] == model_id:
        return _cached_model[0], _cached_model[1]

    from mlx_lm import load

    logger.info("Loading MLX model: %s", model_id)
    model, tokenizer = load(model_id)
    _cached_model = (model, tokenizer, model_id)
    logger.info("MLX model loaded: %s", model_id)
    return model, tokenizer


def unload_mlx_model() -> None:
    """Unload cached model to free memory."""
    global _cached_model
    _cached_model = None


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

def translate_glosses(
    glosses: list[str],
    model_id: str = DEFAULT_MODEL,
    source_lang: str = "Slovak Sign Language",
    target_lang: str = "Slovak",
) -> str:
    """Translate a gloss sequence to natural language using MLX.

    Args:
        glosses: List of sign language glosses (e.g. ["HELLO", "HOW", "YOU"]).
        model_id: Hugging Face model ID for mlx-lm.
        source_lang: Source language description.
        target_lang: Target language for translation.

    Returns:
        Translated sentence in target language.
    """
    if not glosses:
        return ""

    gloss_str = " ".join(glosses)
    system_msg = (
        f"You are a sign language translation assistant. "
        f"Translate {source_lang} gloss sequences into natural {target_lang} sentences. "
        f"Glosses are uppercase words representing individual signs. "
        f"Output only the {target_lang} translation, nothing else."
    )
    user_msg = f"Gloss sequence: {gloss_str}"

    return _mlx_generate(system_msg, user_msg, model_id)


def batch_translate_predictions(
    predictions: list[dict],
    model_id: str = DEFAULT_MODEL,
    window_ms: int = 5000,
) -> list[dict]:
    """Group predictions by time window, translate each group.

    Args:
        predictions: List of prediction dicts from predict_segments().
        model_id: Hugging Face model ID for mlx-lm.
        window_ms: Time window in ms for grouping glosses.

    Returns:
        Predictions with added 'translation' field on the first item of each group.
    """
    if not predictions:
        return predictions

    # Sort by start time
    sorted_preds = sorted(predictions, key=lambda p: p["start_ms"])

    # Group by time windows
    groups: list[list[int]] = []
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
    result = [dict(p) for p in sorted_preds]
    for group_indices in groups:
        glosses = [sorted_preds[i]["predicted_gloss"] for i in group_indices]
        try:
            translation = translate_glosses(glosses, model_id=model_id)
        except Exception as exc:
            logger.warning("MLX translation failed for group: %s", exc)
            translation = f"[Translation error: {exc}]"

        result[group_indices[0]]["translation"] = translation

    return result


def suggest_corrections(
    gloss: str,
    confidence: float,
    context_glosses: list[str],
    model_id: str = DEFAULT_MODEL,
) -> list[str]:
    """Ask MLX model to suggest alternative glosses when confidence is low.

    Args:
        gloss: Current predicted gloss.
        confidence: Prediction confidence (0-1).
        context_glosses: Surrounding glosses for context.
        model_id: Hugging Face model ID.

    Returns:
        List of suggested alternative glosses (up to 5).
    """
    context_str = " ".join(context_glosses) if context_glosses else "(no context)"
    system_msg = (
        "You are a sign language annotation assistant for Slovak Sign Language. "
        "Suggest alternative glosses when the AI prediction has low confidence."
    )
    user_msg = (
        f"The AI predicted '{gloss}' with {confidence:.0%} confidence. "
        f"Context: {context_str}\n"
        f"Suggest up to 5 alternative glosses, one per line, uppercase only. "
        f"If correct, write CORRECT first."
    )

    response = _mlx_generate(system_msg, user_msg, model_id)

    suggestions = []
    for line in response.strip().split("\n"):
        line = line.strip().strip("-").strip("*").strip().upper()
        if line and len(line) < 50:
            suggestions.append(line)
    return suggestions[:5]


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _mlx_generate(
    system_msg: str,
    user_msg: str,
    model_id: str,
    max_tokens: int = 256,
    temp: float = 0.3,
) -> str:
    """Generate text using MLX-LM with chat template."""
    from mlx_lm import generate

    model, tokenizer = load_mlx_model(model_id)

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    # Apply chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
    else:
        # Fallback for tokenizers without chat template
        prompt = f"{system_msg}\n\nUser: {user_msg}\nAssistant:"

    response = generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        temp=temp,
        verbose=False,
    )
    return response.strip()
