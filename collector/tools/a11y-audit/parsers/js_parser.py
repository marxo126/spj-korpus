"""JS parser — event listeners, timers, and ARIA manipulation."""
from __future__ import annotations

import re
from pathlib import Path

from parsers.models import EventListener, FileContext, TimeoutCall

# ── Regex patterns ─────────────────────────────────────────────────────

# addEventListener('type', handler) or addEventListener("type", handler)
_ADD_EVENT_RE = re.compile(
    r"""\.addEventListener\(\s*(['"])([\w-]+)\1""",
)

# Inline event assignment: .onclick = , .onmouseover = , etc.
_INLINE_EVENT_RE = re.compile(
    r"""\.(on\w+)\s*=""",
)

# setTimeout(fn, ms) and setInterval(fn, ms)
_TIMEOUT_RE = re.compile(
    r"""(setTimeout|setInterval)\s*\(\s*(?:function\s*\([^)]*\)\s*\{[^}]*\}|[\w.]+)\s*,\s*(\d+)""",
    re.DOTALL,
)

# setTimeout/setInterval without parseable duration
_TIMEOUT_NO_DUR_RE = re.compile(
    r"""(setTimeout|setInterval)\s*\(""",
)

# Inline event handler attributes (for extracting event type)
_INLINE_HANDLER_PREFIX = "on"


# ── Internal helpers ───────────────────────────────────────────────────

def _extract_event_listeners(content: str, path: str) -> list[EventListener]:
    """Extract addEventListener calls."""
    listeners: list[EventListener] = []
    for m in _ADD_EVENT_RE.finditer(content):
        line = content[:m.start()].count("\n") + 1
        # Get surrounding code for context
        line_start = content.rfind("\n", 0, m.start()) + 1
        line_end = content.find("\n", m.end())
        if line_end == -1:
            line_end = len(content)
        code = content[line_start:line_end].strip()

        listeners.append(EventListener(
            event_type=m.group(2),
            line=line,
            code=code,
            file=path,
        ))
    return listeners


def _extract_inline_events(content: str, path: str) -> list[EventListener]:
    """Extract inline event assignments like .onclick = ..."""
    listeners: list[EventListener] = []
    for m in _INLINE_EVENT_RE.finditer(content):
        handler = m.group(1)
        # Strip 'on' prefix to get event type
        event_type = handler[2:] if handler.startswith("on") else handler
        line = content[:m.start()].count("\n") + 1
        line_start = content.rfind("\n", 0, m.start()) + 1
        line_end = content.find("\n", m.end())
        if line_end == -1:
            line_end = len(content)
        code = content[line_start:line_end].strip()

        listeners.append(EventListener(
            event_type=event_type,
            line=line,
            code=code,
            file=path,
        ))
    return listeners


def _extract_timeouts(content: str) -> list[TimeoutCall]:
    """Extract setTimeout and setInterval calls."""
    timeouts: list[TimeoutCall] = []
    seen_offsets: set[int] = set()

    # With parseable duration
    for m in _TIMEOUT_RE.finditer(content):
        line = content[:m.start()].count("\n") + 1
        line_start = content.rfind("\n", 0, m.start()) + 1
        line_end = content.find("\n", m.end())
        if line_end == -1:
            line_end = len(content)
        code = content[line_start:line_end].strip()

        timeouts.append(TimeoutCall(
            function=m.group(1),
            duration_ms=int(m.group(2)),
            line=line,
            code=code,
        ))
        seen_offsets.add(m.start())

    # Without parseable duration (variable or expression)
    for m in _TIMEOUT_NO_DUR_RE.finditer(content):
        if m.start() in seen_offsets:
            continue
        # Check this isn't a duplicate of an already-found one
        already = False
        for offset in seen_offsets:
            if abs(m.start() - offset) < 5:
                already = True
                break
        if already:
            continue

        line = content[:m.start()].count("\n") + 1
        line_start = content.rfind("\n", 0, m.start()) + 1
        line_end = content.find("\n", m.end())
        if line_end == -1:
            line_end = len(content)
        code = content[line_start:line_end].strip()

        timeouts.append(TimeoutCall(
            function=m.group(1),
            duration_ms=None,
            line=line,
            code=code,
        ))

    return timeouts


# ── Public API ─────────────────────────────────────────────────────────

def parse_js_string(content: str, path: str = "<string>") -> FileContext:
    """Parse a JS string and return a FileContext."""
    listeners = _extract_event_listeners(content, path)
    listeners.extend(_extract_inline_events(content, path))
    timeouts = _extract_timeouts(content)

    return FileContext(
        path=path,
        event_listeners=listeners,
        timeouts=timeouts,
        raw_content=content,
        lines=content.splitlines(),
    )


def parse_js_file(path: str | Path) -> FileContext:
    """Parse a JS file and return a FileContext."""
    path = Path(path)
    content = path.read_text(encoding="utf-8", errors="replace")
    return parse_js_string(content, str(path))
