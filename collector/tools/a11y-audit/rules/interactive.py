"""WCAG 1.4.13, 2.5.1-8 — Interactive content rules."""
from __future__ import annotations

import re

from rules.base import BaseRule, Finding, Severity
from rules.helpers import iter_files, parse_px
from parsers.models import JS_EXTENSIONS


_DRAG_PATTERNS = re.compile(
    r"\b(dragstart|dragend|dragover|dragenter|dragleave|drop|draggable)\b", re.IGNORECASE
)


class InteractiveRule(BaseRule):
    id = "interactive"
    name = "Interactive Content"
    wcag_criteria = ("1.4.13", "2.5.1", "2.5.2", "2.5.3", "2.5.4",
                     "2.5.5", "2.5.6", "2.5.7", "2.5.8")
    standards = ("WCAG 2.2 AA",)

    def check(self, ctx, config) -> list[Finding]:
        findings: list[Finding] = []
        findings.extend(self._check_hover_content(ctx))
        findings.extend(self._check_touch_target(ctx, config))
        findings.extend(self._check_drag_alternative(ctx))
        return findings

    def _check_hover_content(self, ctx) -> list[Finding]:
        """Check :hover that shows/hides content without :focus-within/:focus."""
        findings: list[Finding] = []
        visibility_props = {"display", "visibility", "opacity"}
        hover_rules: list[tuple[str, int]] = []
        focus_bases: set[str] = set()

        for rule in ctx.css.rules:
            sel = rule.selector
            # Collect focus-within/focus bases
            if ":focus-within" in sel or ":focus" in sel:
                base = re.split(r":focus(?:-within)?", sel)[0].strip()
                focus_bases.add(base)

            if ":hover" not in sel:
                continue
            # Check if any visibility property is changed
            changed = any(rule.properties.get(p) for p in visibility_props)
            if not changed:
                continue
            base = sel.split(":hover")[0].strip()
            hover_rules.append((base, rule.line))

        for base, line in hover_rules:
            if base not in focus_bases:
                findings.append(self._finding(
                    check_id="hover-content",
                    severity=Severity.MODERATE,
                    wcag="1.4.13",
                    wcag_name="Content on Hover or Focus",
                    message=f":hover shows/hides content without :focus-within — {base}:hover",
                    file="style.css",
                    line=line,
                    element=f"{base}:hover",
                    fix="Add corresponding :focus-within or :focus rule",
                    impact=("motor",),
                ))
        return findings

    def _check_touch_target(self, ctx, config) -> list[Finding]:
        """Check interactive element sizes < 48px."""
        findings: list[Finding] = []
        min_size = config.get("thresholds", {}).get("target_size_enhanced", 44)
        _interactive_re = re.compile(
            r"(?:^|[\s,>+~])(?:button|a(?=[.#\[:>\s,+~]|$)|input|select|textarea)"
            r"|\.btn|\[type=",
        )

        for rule in ctx.css.rules:
            sel = rule.selector.lower()
            # Strip CSS comments from selector for matching
            sel_clean = re.sub(r"/\*.*?\*/", "", sel).strip()
            # Skip media query rules (they may be responsive overrides)
            if "@media" in sel:
                continue
            is_interactive = bool(_interactive_re.search(sel_clean))
            if not is_interactive:
                continue
            props = rule.properties
            for dim_prop in ("min-height", "height", "min-width", "width"):
                val = props.get(dim_prop, "")
                if not val:
                    continue
                px = parse_px(val)
                if px is not None and 0 < px < min_size:
                    findings.append(self._finding(
                        check_id="touch-target",
                        severity=Severity.SERIOUS,
                        wcag="2.5.8",
                        wcag_name="Target Size (Minimum)",
                        message=f"Interactive element {dim_prop}:{val} below {min_size}px — {rule.selector}",
                        file="style.css",
                        line=rule.line,
                        element=rule.selector,
                        fix=f"Ensure minimum touch target size of {min_size}px",
                        impact=("motor",),
                    ))
                    break  # One finding per rule
        return findings

    def _check_drag_alternative(self, ctx) -> list[Finding]:
        """Flag drag-related patterns for manual review."""
        findings: list[Finding] = []
        for path, fc in iter_files(ctx, JS_EXTENSIONS):
            for i, line_text in enumerate(fc.lines, 1):
                stripped = line_text.strip()
                # Skip comments — "drop" in prose doesn't mean drag-and-drop
                if stripped.startswith("//") or stripped.startswith("*") or stripped.startswith("/*"):
                    continue
                if _DRAG_PATTERNS.search(line_text):
                    findings.append(self._finding(
                        check_id="drag-alternative",
                        severity=Severity.SERIOUS,
                        wcag="2.5.7",
                        wcag_name="Dragging Movements",
                        message="Drag interaction detected — verify single-pointer alternative exists",
                        file=path,
                        line=i,
                        element=line_text.strip()[:80],
                        fix="Provide a non-dragging alternative (e.g., buttons, select menus)",
                        impact=("motor",),
                    ))
                    break  # One finding per file
        return findings
