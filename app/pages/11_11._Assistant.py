"""Page 11 — AI Assistant for annotators.

Provides a conversational interface powered by Claude that is aware of the
current pipeline state (inventory, segments, pairings, models, glossary).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import json
import os

import streamlit as st

DATA_DIR = Path(__file__).parent.parent.parent / "data"

st.set_page_config(page_title="SPJ — Assistant", page_icon="💬", layout="wide")
st.title("11. 🤖 AI Assistant")
st.caption("Ask questions about annotation, pipeline status, or SPJ conventions.")


# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------

def _get_api_key() -> str | None:
    """Resolve Anthropic API key from env or Streamlit secrets."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Pipeline context builder
# ---------------------------------------------------------------------------

def _build_context() -> str:
    """Read-only snapshot of current pipeline status for the system prompt."""
    import pandas as pd

    parts: list[str] = []

    # Inventory
    try:
        inv_path = DATA_DIR / "inventory.csv"
        if inv_path.exists():
            inv = pd.read_csv(inv_path, dtype=str)
            n_videos = len(inv)
            pose_dir = DATA_DIR / "pose"
            n_poses = len(list(pose_dir.glob("*.pose"))) if pose_dir.exists() else 0
            parts.append(f"Videos: {n_videos} total, {n_poses} with pose extracted ({n_poses*100//max(1,n_videos)}%)")
    except Exception:
        pass

    # Alignment
    try:
        align_path = DATA_DIR / "training" / "alignment.csv"
        if align_path.exists():
            align = pd.read_csv(align_path, dtype=str)
            n_total = len(align)
            n_approved = int((align.get("status", pd.Series()) == "approved").sum())
            parts.append(f"Segments: {n_total} total, {n_approved} approved")
    except Exception:
        pass

    # Pairings
    try:
        pairings_path = DATA_DIR / "training" / "pairings.csv"
        if pairings_path.exists():
            from spj.training_data import load_pairings_csv, PST_PAIRED
            pairings = load_pairings_csv(pairings_path)
            if not pairings.empty:
                paired = pairings[pairings["status"] == PST_PAIRED]
                n_paired = len(paired)
                n_glosses = paired["gloss_id"].nunique() if "gloss_id" in paired.columns else 0
                parts.append(f"Paired signs: {n_paired} ({n_glosses} unique glosses)")
    except Exception:
        pass

    # Glossary
    try:
        glossary_path = DATA_DIR / "training" / "glossary.json"
        if glossary_path.exists():
            from spj.glossary import load_glossary
            glossary = load_glossary(glossary_path)
            n = len(glossary._data.get("glosses", {}))
            parts.append(f"Glossary: {n} entries")
    except Exception:
        pass

    # Checkpoints
    try:
        models_dir = DATA_DIR / "models"
        if models_dir.exists():
            from spj.trainer import list_checkpoints
            ckpts = list_checkpoints(models_dir)
            if ckpts:
                latest = max(ckpts, key=lambda c: c.get("timestamp", ""))
                parts.append(
                    f"Models: {len(ckpts)} checkpoints, "
                    f"latest val_acc={latest.get('val_acc', '?')}"
                )
            else:
                parts.append("Models: no checkpoints yet")
    except Exception:
        pass

    # Evaluations
    try:
        eval_dir = DATA_DIR / "evaluations"
        if eval_dir.exists():
            eval_files = list(eval_dir.glob("*_eval.json"))
            if eval_files:
                latest_eval = max(eval_files, key=lambda f: f.stat().st_mtime)
                report = json.loads(latest_eval.read_text())
                parts.append(
                    f"Latest evaluation: accuracy={report.get('accuracy', '?')}, "
                    f"top3={report.get('top3_accuracy', '?')}"
                )
    except Exception:
        pass

    if not parts:
        return "No pipeline data found yet. The pipeline has not been initialized."

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an AI assistant helping annotators work on the SPJ (Slovak Sign Language) corpus.
You have access to the current pipeline status and can answer questions about:
- Annotation conventions (tier names, ID-gloss format, priority order)
- Pipeline status (how many videos, segments, paired signs, models)
- What to do next in the annotation workflow
- SPJ-specific rules and constraints

## Pipeline Status
{context}

## Key Rules
- ELAN tier names are exact: S1_Gloss_RH, S1_Gloss_LH, S1_Translation, S1_Mouthing, S1_Mouth_Gesture, S1_NonManual
- AI tiers: AI_Gloss_RH, AI_Gloss_LH, AI_Confidence
- ID-glosses: UPPERCASE-NUMBER (e.g. WATER-1, HOUSE-3)
- Mouthings: lowercase Slovak (e.g. "voda", "dom")
- Mouth gestures: descriptive labels (e.g. "puffed_cheeks", "pursed_lips")
- Annotation priority: 1) ID-glosses, 2) Translations, 3) Mouthings, 4) Non-manual, 5) HamNoSys, 6) SignWriting

## Retraining Thresholds
- 500 signs: Fine-tune backbone
- 2,000 signs: v1 SPJ model
- 5,000 signs: v2 + active learning
- 10,000+ signs: v3 full retrain

## Data Licensing
- [vocabulary reference]: Vocabulary reference ONLY — never suggest using it for ML training
- [public SL dataset]: Reference/benchmarking only — never suggest for training

Be concise, helpful, and always reference exact tier names and conventions.
Answer in the language the user writes in (Slovak or English).
"""


# ---------------------------------------------------------------------------
# Chat UI
# ---------------------------------------------------------------------------

def main():
    api_key = _get_api_key()

    if not api_key:
        st.warning(
            "No Anthropic API key found. Set `ANTHROPIC_API_KEY` as an environment "
            "variable or add it to `.streamlit/secrets.toml`."
        )
        st.code(
            '# .streamlit/secrets.toml\nANTHROPIC_API_KEY = "sk-ant-..."',
            language="toml",
        )
        return

    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    # Sidebar controls
    with st.sidebar:
        st.subheader("Assistant Settings")
        model = st.selectbox(
            "Model",
            ["claude-sonnet-4-20250514", "claude-haiku-4-5-20251001"],
            index=0,
        )
        if st.button("Clear Chat"):
            st.session_state["chat_messages"] = []
            st.rerun()

        with st.expander("Pipeline Context", expanded=False):
            ctx = _build_context()
            st.text(ctx)

    # Display chat history
    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask about annotation, pipeline status, or conventions..."):
        # Add user message
        st.session_state["chat_messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Build system prompt with fresh context
        context = _build_context()
        system = _SYSTEM_PROMPT.format(context=context)

        # Stream response
        try:
            from anthropic import Anthropic

            client = Anthropic(api_key=api_key)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_response = ""

                with client.messages.stream(
                    model=model,
                    max_tokens=2048,
                    system=system,
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state["chat_messages"]
                    ],
                ) as stream:
                    for text in stream.text_stream:
                        full_response += text
                        placeholder.markdown(full_response + "▌")

                placeholder.markdown(full_response)

            st.session_state["chat_messages"].append(
                {"role": "assistant", "content": full_response}
            )

        except Exception as exc:
            st.error(f"API error: {exc}")


main()
