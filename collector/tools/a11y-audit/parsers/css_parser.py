"""CSS parser — variables, keyframes, rules, and media queries."""
from __future__ import annotations

import re
from pathlib import Path

from parsers.models import CSSContext, CSSRule, CSSVariable, KeyframeAnimation

# ── Regex patterns ─────────────────────────────────────────────────────

# Custom property: --name: value;
_VAR_RE = re.compile(r"(--[\w-]+)\s*:\s*([^;]+);")

# Hex color (3, 4, 6, or 8 hex digits)
_HEX_RE = re.compile(r"^#[0-9a-fA-F]{3,8}$")

# var(--name) reference
_VAR_REF_RE = re.compile(r"var\((--[\w-]+)\)")

# @keyframes name { ... }
_KEYFRAMES_RE = re.compile(
    r"@keyframes\s+([\w-]+)\s*\{([^}]*(?:\{[^}]*\}[^}]*)*)\}",
    re.DOTALL,
)

# animation or animation-duration property
_ANIM_DURATION_RE = re.compile(r"([\d.]+)(ms|s)")
_ANIM_ITERATION_RE = re.compile(r"animation-iteration-count\s*:\s*([^;]+);")
_ANIM_SHORTHAND_RE = re.compile(r"animation\s*:[^;]*\b(infinite)\b")

# Media query blocks
_MEDIA_RE = re.compile(r"@media\s*\(([^)]+)\)", re.DOTALL)

# Dark mode contexts
_DARK_CONTEXT_RE = re.compile(
    r"(?:html\.dark|\.dark-mode|\.dark-theme|\[data-theme=['\"]dark['\"]\]"
    r"|@media\s*\(\s*prefers-color-scheme\s*:\s*dark\s*\))\s*\{",
)

# Simple CSS rule: selector { properties }
# Matches outermost braces, skipping @-rules and nested blocks
_RULE_RE = re.compile(
    r"^([^@{}\n][^{]*?)\{([^{}]*)\}",
    re.MULTILINE,
)


# ── Internal helpers ───────────────────────────────────────────────────

def _resolve_hex(value: str, all_vars: dict[str, str]) -> str | None:
    """Try to resolve a value to a hex color."""
    value = value.strip()
    if _HEX_RE.match(value):
        return value
    m = _VAR_REF_RE.match(value)
    if m and m.group(1) in all_vars:
        ref_val = all_vars[m.group(1)].strip()
        if _HEX_RE.match(ref_val):
            return ref_val
    return None


def _detect_mode(content: str, pos: int) -> str:
    """Determine if a position is inside a dark-mode context."""
    # Look backwards from pos for the nearest dark-mode opener
    prefix = content[:pos]
    # Find last dark-mode context opener
    dark_match = None
    for m in _DARK_CONTEXT_RE.finditer(prefix):
        dark_match = m
    if dark_match is None:
        return "light"
    # Count braces between the dark context opener and our position
    between = prefix[dark_match.end():]
    depth = 0
    for ch in between:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
    # If depth >= 0 we're still inside the dark block
    return "dark" if depth >= 0 else "light"


def _extract_variables(content: str) -> list[CSSVariable]:
    """Extract CSS custom properties with mode detection."""
    # First pass: collect all raw values for resolution
    raw_vars: dict[str, str] = {}
    for m in _VAR_RE.finditer(content):
        raw_vars[m.group(1)] = m.group(2).strip()

    variables: list[CSSVariable] = []
    for m in _VAR_RE.finditer(content):
        name = m.group(1)
        value = m.group(2).strip()
        mode = _detect_mode(content, m.start())
        resolved = _resolve_hex(value, raw_vars)
        variables.append(CSSVariable(
            name=name,
            value=value,
            resolved_hex=resolved,
            mode=mode,
        ))
    return variables


def _extract_keyframes(content: str) -> list[KeyframeAnimation]:
    """Extract @keyframes animations."""
    keyframes: list[KeyframeAnimation] = []
    for m in _KEYFRAMES_RE.finditer(content):
        name = m.group(1)
        line = content[:m.start()].count("\n") + 1

        # Try to find duration from nearby animation property
        duration_ms = None
        dur_match = _ANIM_DURATION_RE.search(content)
        if dur_match:
            val = float(dur_match.group(1))
            if dur_match.group(2) == "s":
                val *= 1000
            duration_ms = val

        # Detect iteration count
        iteration = "1"
        iter_match = _ANIM_ITERATION_RE.search(content)
        if iter_match:
            iteration = iter_match.group(1).strip()
        shorthand_match = _ANIM_SHORTHAND_RE.search(content)
        if shorthand_match:
            iteration = "infinite"

        keyframes.append(KeyframeAnimation(
            name=name,
            duration_ms=duration_ms,
            iteration_count=iteration,
            line=line,
        ))
    return keyframes


def _extract_rules(content: str) -> list[CSSRule]:
    """Extract CSS rules (selector + properties)."""
    rules: list[CSSRule] = []
    for m in _RULE_RE.finditer(content):
        selector = m.group(1).strip()
        if selector.startswith("@"):
            continue
        body = m.group(2)
        line = content[:m.start()].count("\n") + 1
        props: dict[str, str] = {}
        for prop_match in re.finditer(r"([\w-]+)\s*:\s*([^;]+);", body):
            props[prop_match.group(1)] = prop_match.group(2).strip()
        if props:
            rules.append(CSSRule(selector=selector, properties=props, line=line))
    return rules


def _extract_media_queries(content: str) -> tuple[list[str], bool, bool, bool]:
    """Extract media queries and detect accessibility-related ones."""
    queries: list[str] = []
    has_reduced_motion = False
    has_contrast = False
    has_forced_colors = False

    for m in _MEDIA_RE.finditer(content):
        query = m.group(1).strip()
        queries.append(query)
        if "prefers-reduced-motion" in query:
            has_reduced_motion = True
        if "prefers-contrast" in query:
            has_contrast = True
        if "forced-colors" in query:
            has_forced_colors = True

    return queries, has_reduced_motion, has_contrast, has_forced_colors


# ── Public API ─────────────────────────────────────────────────────────

def parse_css_string(content: str) -> CSSContext:
    """Parse a CSS string and return a CSSContext."""
    variables = _extract_variables(content)
    keyframes = _extract_keyframes(content)
    rules = _extract_rules(content)
    queries, reduced_motion, contrast, forced_colors = _extract_media_queries(content)

    return CSSContext(
        variables=variables,
        keyframes=keyframes,
        rules=rules,
        media_queries=queries,
        has_prefers_reduced_motion=reduced_motion,
        has_prefers_contrast=contrast,
        has_forced_colors=forced_colors,
        raw_content=content,
    )


def parse_css_file(path: str | Path) -> CSSContext:
    """Parse a CSS file and return a CSSContext."""
    path = Path(path)
    content = path.read_text(encoding="utf-8", errors="replace")
    return parse_css_string(content)
