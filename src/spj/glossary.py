"""Sign-word glossary — maps Slovak words to sign glosses.

Maintains a JSON partner-dictnary that grows as the user maps words during review.
Each gloss entry has a lemma, inflected forms, POS tag, and optional notes.
A reverse index enables O(1) word→gloss lookups.

Schema (data/training/glossary.json):
    {
        "version": 1,
        "glosses": {
            "WATER-1": {
                "lemma": "voda",
                "forms": ["voda", "vody", "vode", "vodu", "vodou"],
                "pos": "noun",
                "notes": ""
            }
        }
    }
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

_EDGE_PUNCT = re.compile(r'^[\W_]+|[\W_]+$', re.UNICODE)


def normalize_word(word: str) -> str:
    """Lowercase, strip edge punctuation, preserve diacritics."""
    w = word.strip().lower()
    w = _EDGE_PUNCT.sub('', w)
    return w


def tokenize_slovak(text: str) -> list[str]:
    """Split text on whitespace, return raw tokens (before normalization)."""
    return text.split()


# ---------------------------------------------------------------------------
# Glossary class
# ---------------------------------------------------------------------------

class Glossary:
    """In-memory sign-word glossary with reverse index for fast lookups."""

    def __init__(self, data: dict | None = None):
        self._data = data or {"version": 1, "glosses": {}}
        self._reverse: dict[str, list[str]] = {}
        self._build_reverse_index()

    # -- Index ---------------------------------------------------------------

    def _build_reverse_index(self) -> None:
        """Build normalized_word → [gloss_ids] mapping."""
        rev: dict[str, list[str]] = {}
        for gid, entry in self._data.get("glosses", {}).items():
            for form in entry.get("forms", []):
                nf = normalize_word(form)
                if nf:
                    rev.setdefault(nf, []).append(gid)
        self._reverse = rev

    # -- Lookups -------------------------------------------------------------

    def lookup(self, word: str) -> list[str]:
        """Return gloss IDs matching a word (normalized)."""
        return self._reverse.get(normalize_word(word), [])

    def match_sentence(self, text: str) -> list[dict]:
        """Analyze each word in *text*, returning per-word match info.

        Returns list of dicts:
            {"raw": str, "normalized": str, "glosses": [str], "mapped": bool}
        """
        results = []
        for raw in tokenize_slovak(text):
            nw = normalize_word(raw)
            glosses = self._reverse.get(nw, [])
            results.append({
                "raw": raw,
                "normalized": nw,
                "glosses": glosses,
                "mapped": bool(glosses),
            })
        return results

    # -- Mutations -----------------------------------------------------------

    def add_gloss(
        self,
        gloss_id: str,
        lemma: str,
        forms: list[str] | None = None,
        pos: str = "",
        notes: str = "",
    ) -> None:
        """Add or overwrite a gloss entry."""
        gloss_id = gloss_id.strip().upper()
        if not gloss_id:
            raise ValueError("gloss_id must not be empty")
        all_forms = list(dict.fromkeys(
            [lemma] + (forms or [])
        ))  # deduplicate, preserve order
        self._data["glosses"][gloss_id] = {
            "lemma": lemma.strip(),
            "forms": [f.strip() for f in all_forms if f.strip()],
            "pos": pos.strip(),
            "notes": notes.strip(),
        }
        self._build_reverse_index()

    def add_form(self, gloss_id: str, form: str) -> bool:
        """Add an inflected form to an existing gloss. Returns True if added."""
        entry = self._data["glosses"].get(gloss_id)
        if entry is None:
            return False
        nf = form.strip()
        if not nf or nf in entry["forms"]:
            return False
        entry["forms"].append(nf)
        # Incremental reverse index update (O(1) instead of full rebuild)
        normalized = normalize_word(nf)
        if normalized:
            self._reverse.setdefault(normalized, []).append(gloss_id)
        return True

    def remove_gloss(self, gloss_id: str) -> bool:
        """Remove a gloss entry entirely. Returns True if it existed."""
        if gloss_id in self._data["glosses"]:
            del self._data["glosses"][gloss_id]
            self._build_reverse_index()
            return True
        return False

    # -- Helpers -------------------------------------------------------------

    def suggest_next_id(self, base: str) -> str:
        """Suggest next available ID: 'WORD' → 'WORD-1', or 'WORD-2' etc."""
        base = base.strip().upper()
        if not base:
            return "SIGN-1"
        existing = self._data["glosses"]
        # Try base-1, base-2, ...
        for n in range(1, 1000):
            candidate = f"{base}-{n}"
            if candidate not in existing:
                return candidate
        return f"{base}-999"

    def get_entry(self, gloss_id: str) -> dict | None:
        """Return a shallow copy of the entry dict for a gloss, or None."""
        entry = self._data["glosses"].get(gloss_id)
        if entry is None:
            return None
        return {**entry, "forms": list(entry.get("forms", []))}

    # -- Properties ----------------------------------------------------------

    @property
    def gloss_ids(self) -> list[str]:
        return sorted(self._data["glosses"].keys())

    @property
    def n_glosses(self) -> int:
        return len(self._data["glosses"])

    @property
    def n_forms(self) -> int:
        return sum(
            len(e.get("forms", []))
            for e in self._data["glosses"].values()
        )

    @property
    def raw_data(self) -> dict:
        return self._data


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def load_glossary(path: str | Path) -> Glossary:
    """Load glossary from JSON file. Returns empty glossary if file missing."""
    p = Path(path)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        return Glossary(data)
    return Glossary()


def save_glossary(glossary: Glossary, path: str | Path) -> None:
    """Write glossary to JSON file, creating parent dirs if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(glossary.raw_data, f, ensure_ascii=False, indent=2)
