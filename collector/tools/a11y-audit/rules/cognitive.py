"""WCAG 2.2.1, 3.2.x, 3.3.4, 3.3.7 — Cognitive accessibility rules."""
from __future__ import annotations

import re

from rules.base import BaseRule, Finding, Severity
from rules.helpers import iter_elements, iter_files
from parsers.models import PHP_EXTENSIONS, JS_EXTENSIONS

_SESSION_TIMEOUT_RE = re.compile(
    r"(session[_\-]?timeout|session[_\-]?expire|session[_\-]?lifetime|max[_\-]?idle)",
    re.IGNORECASE,
)
_DESTRUCTIVE_PATTERN = re.compile(
    r"\b(delete|remove|destroy|drop|truncate|reset|clear[_\-]?all)\b",
    re.IGNORECASE,
)
_CONFIRM_PATTERN = re.compile(
    r"\b(confirm\s*\(|modal|dialog|are\s+you\s+sure|naozaj|ste\s+si\s+ist)",
    re.IGNORECASE,
)
# DOM/IndexedDB operations that match _DESTRUCTIVE_PATTERN but aren't user-facing
_INTERNAL_DESTRUCTIVE_RE = re.compile(
    r"(\.\s*remove\s*\(|\.removeChild|\.removeAttribute|\.removeEventListener|"
    r"store\.delete|indexedDB|objectStore|\.removeItem|clearTimeout|clearInterval|"
    r"URL\.revokeObjectURL|async\s+remove\s*\(|await\s+this\.remove\s*\(|"
    r"await\s+self\.remove\s*\()",
)
_FOCUS_CHANGE_RE = re.compile(
    r"\b(window\.location|location\.href|location\.assign|\.submit\(\)|navigate)\b",
)
_INCLUDE_HEADER_RE = re.compile(
    r"""(?:require|include)(?:_once)?\s*[\(]?\s*['"]([^'"]*(?:header|footer|nav)[^'"]*)['"]""",
    re.IGNORECASE,
)
_CONFIRM_EMAIL_RE = re.compile(
    r"\b(confirm[_\-]?email|email[_\-]?confirm|verify[_\-]?email|retype[_\-]?email|repeat[_\-]?email)\b",
    re.IGNORECASE,
)


class CognitiveRule(BaseRule):
    id = "cognitive"
    name = "Cognitive Accessibility"
    wcag_criteria = ("2.2.1", "3.2.1", "3.2.2", "3.2.3", "3.2.4", "3.3.4", "3.3.7")
    standards = ("WCAG 2.2 AA",)

    def check(self, ctx, config) -> list[Finding]:
        findings: list[Finding] = []
        findings.extend(self._check_session_timeout(ctx))
        findings.extend(self._check_consistent_nav(ctx))
        findings.extend(self._check_on_focus_change(ctx))
        findings.extend(self._check_error_prevention(ctx))
        findings.extend(self._check_redundant_entry(ctx))
        return findings

    def _check_session_timeout(self, ctx) -> list[Finding]:
        """Check JS/PHP for session timeout patterns."""
        findings: list[Finding] = []
        for path, fc in iter_files(ctx, JS_EXTENSIONS + PHP_EXTENSIONS):
            content = fc.content
            for m in _SESSION_TIMEOUT_RE.finditer(content):
                # Try to find a numeric value nearby
                nearby = content[max(0, m.start() - 50):m.end() + 100]
                nums = re.findall(r"(\d{4,})", nearby)
                for num_str in nums:
                    ms = int(num_str)
                    # If it looks like milliseconds and < 20 min
                    if 1000 < ms < 1_200_000:
                        line_num = content[:m.start()].count("\n") + 1
                        findings.append(self._finding(
                            check_id="session-timeout",
                            severity=Severity.SERIOUS,
                            wcag="2.2.1",
                            wcag_name="Timing Adjustable",
                            message=f"Session timeout {ms}ms ({ms // 1000}s) — minimum 20 minutes recommended",
                            file=path,
                            line=line_num,
                            fix="Increase timeout to >= 20 min or provide extend/disable option",
                            impact=("cognitive",),
                        ))
                        break
        return findings

    def _check_consistent_nav(self, ctx) -> list[Finding]:
        """Check that PHP pages include same header/footer."""
        findings: list[Finding] = []
        includes_per_file: dict[str, set[str]] = {}

        for path, fc in iter_files(ctx, PHP_EXTENSIONS):
            # Skip includes/ directory itself
            if "includes/" in path or "include/" in path:
                continue
            includes: set[str] = set()
            for m in _INCLUDE_HEADER_RE.finditer(fc.content):
                includes.add(m.group(1).strip())
            if includes:
                includes_per_file[path] = includes

        if len(includes_per_file) < 2:
            return findings

        # Find the most common include set
        all_includes = set()
        for inc_set in includes_per_file.values():
            all_includes.update(inc_set)

        # Check each file includes the common header/footer
        for path, includes in includes_per_file.items():
            for common in all_includes:
                if "header" in common.lower() and common not in includes:
                    findings.append(self._finding(
                        check_id="consistent-nav",
                        severity=Severity.MODERATE,
                        wcag="3.2.3",
                        wcag_name="Consistent Navigation",
                        message=f"Page does not include common header '{common}'",
                        file=path,
                        line=0,
                        fix=f"Include '{common}' for consistent navigation",
                        impact=("cognitive",),
                    ))
        return findings

    def _check_on_focus_change(self, ctx) -> list[Finding]:
        """Check focus listeners that change page state."""
        findings: list[Finding] = []
        for path, fc in iter_files(ctx, JS_EXTENSIONS):
            for listener in fc.event_listeners:
                if listener.event_type != "focus":
                    continue
                # Check nearby lines for page-changing code
                start = max(0, listener.line - 1)
                end = min(len(fc.lines), listener.line + 5)
                nearby = "\n".join(fc.lines[start:end])
                if _FOCUS_CHANGE_RE.search(nearby):
                    findings.append(self._finding(
                        check_id="on-focus-change",
                        severity=Severity.SERIOUS,
                        wcag="3.2.1",
                        wcag_name="On Focus",
                        message="Focus event triggers page navigation or form submission",
                        file=path,
                        line=listener.line,
                        fix="Do not change context on focus — use explicit activation (click/Enter)",
                        impact=("motor", "cognitive"),
                    ))
        return findings

    def _check_error_prevention(self, ctx) -> list[Finding]:
        """Destructive actions should have confirmation."""
        findings: list[Finding] = []
        for path, fc in iter_files(ctx, JS_EXTENSIONS + PHP_EXTENSIONS):
            # Skip API endpoints and backend-only PHP files (no HTML output).
            # These are called by JS which handles confirmation on the frontend.
            if path.endswith(PHP_EXTENSIONS):
                if "/api/" in path:
                    continue
                # If file has very few HTML elements, it's likely a backend script
                if len(fc.elements) <= 2:
                    continue
            lines = fc.lines
            for i, line_text in enumerate(lines, 1):
                if not _DESTRUCTIVE_PATTERN.search(line_text):
                    continue
                # Skip internal DOM/IndexedDB operations (not user-facing)
                if _INTERNAL_DESTRUCTIVE_RE.search(line_text):
                    continue
                # Skip HTML/PHP comments (not user-facing actions)
                stripped = line_text.strip()
                if stripped.startswith("<!--") or stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
                    continue
                # Check surrounding lines for confirmation (wide range to cover
                # multi-line form tags where confirm() is on the <form> opener)
                start = max(0, i - 6)
                end = min(len(lines), i + 10)
                nearby = "\n".join(lines[start:end])
                if _CONFIRM_PATTERN.search(nearby):
                    continue
                findings.append(self._finding(
                    check_id="error-prevention",
                    severity=Severity.SERIOUS,
                    wcag="3.3.4",
                    wcag_name="Error Prevention (Legal, Financial, Data)",
                    message="Destructive action without confirmation dialog",
                    file=path,
                    line=i,
                    element=line_text.strip()[:80],
                    fix="Add confirmation dialog (confirm() or modal) before destructive actions",
                    impact=("cognitive",),
                ))
        return findings

    def _check_redundant_entry(self, ctx) -> list[Finding]:
        """Check for fields asking same info twice."""
        findings: list[Finding] = []
        for path, fc, elem in iter_elements(ctx):
            if elem.tag.lower() != "input":
                continue
            name = str(elem.attributes.get("name", "")).lower()
            if _CONFIRM_EMAIL_RE.search(name):
                findings.append(self._finding(
                    check_id="redundant-entry",
                    severity=Severity.MINOR,
                    wcag="3.3.7",
                    wcag_name="Redundant Entry",
                    message="Confirm/retype field detected — consider removing",
                    file=path,
                    line=elem.line,
                    element=f"<input name=\"{name}\">",
                    fix="Use auto-fill or verification instead of requiring re-entry",
                ))
        return findings
