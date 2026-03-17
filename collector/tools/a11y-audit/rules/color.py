"""WCAG 1.4.1 / 1.3.3 — Color-only information rules."""
from __future__ import annotations

import re

from rules.base import BaseRule, Finding, Severity
from rules.helpers import iter_elements

_COLOR_WORDS = re.compile(r"(red|green|blue|orange|yellow|status|success|danger|warning)", re.I)
_ICON_CHARS = frozenset("✓✗✔✘!⚠●○◉✕✖✚⬤▶▲►")


class ColorRule(BaseRule):
    id = "color"
    name = "Color as Information"
    wcag_criteria = ("1.4.1", "1.3.3")
    standards = ("WCAG 2.2 AA",)

    def check(self, ctx, config) -> list[Finding]:
        findings: list[Finding] = []

        for path, fc, elem in iter_elements(ctx):
            classes = str(elem.attributes.get("class", ""))

            # color-only-status
            if _COLOR_WORDS.search(classes):
                if not self._has_icon_or_text(elem):
                    findings.append(self._finding(
                        check_id="color-only-status",
                        severity=Severity.SERIOUS,
                        wcag="1.4.1",
                        wcag_name="Use of Color",
                        message=f"Color-word class '{classes}' without icon or text indicator",
                        file=path,
                        line=elem.line,
                        element=f"<{elem.tag} class=\"{classes}\">",
                        fix="Add an icon, symbol, or text label alongside the color indicator",
                        impact=("color-blind",),
                    ))

            # error-color-only
            role = elem.attributes.get("role", "")
            if ("error" in classes.lower() or "danger" in classes.lower()
                    or role in ("alert", "status")):
                if not self._has_icon_or_text(elem):
                    findings.append(self._finding(
                        check_id="error-color-only",
                        severity=Severity.SERIOUS,
                        wcag="1.4.1",
                        wcag_name="Use of Color",
                        message=f"Error/status element relies on color only — {path}:{elem.line}",
                        file=path,
                        line=elem.line,
                        element=f"<{elem.tag}>",
                        fix="Add icon or text prefix (e.g. 'Error:') to convey status",
                        impact=("color-blind",),
                    ))

            # link-distinction
            if elem.tag == "a":
                findings.extend(self._check_link(ctx, path, elem))

        return findings

    def _check_link(self, ctx, path, elem) -> list[Finding]:
        css = ctx.css
        for rule in css.rules:
            sel = rule.selector.lower()
            if "a" not in sel:
                continue
            # Skip accessibility helper selectors (skip-link, sr-only, etc.)
            if "skip" in sel or "sr-only" in sel or "visually-hidden" in sel:
                continue
            props = rule.properties
            td = props.get("text-decoration", "").lower()
            if td == "none" and "border-bottom" not in props:
                # Check for other visual distinction (background, font-weight, border)
                has_distinction = any(
                    k in props for k in (
                        "border", "font-weight", "background",
                        "background-color", "outline", "box-shadow",
                    )
                )
                if has_distinction:
                    continue
                return [self._finding(
                    check_id="link-distinction",
                    severity=Severity.SERIOUS,
                    wcag="1.4.1",
                    wcag_name="Use of Color",
                    message=f"Links styled with text-decoration:none without alternative ({rule.selector})",
                    file="style.css",
                    line=rule.line,
                    element=rule.selector,
                    fix="Add underline, border-bottom, or other non-color visual cue for links",
                    impact=("color-blind",),
                )]
        return []

    @staticmethod
    def _has_icon_or_text(elem) -> bool:
        text = elem.text_content.strip()
        if text and any(c in _ICON_CHARS for c in text):
            return True
        if len(text) > 1:
            return True
        for child in elem.children:
            if child.tag in ("i", "svg", "img", "span"):
                child_cls = str(child.attributes.get("class", ""))
                if "icon" in child_cls.lower() or "fa-" in child_cls or "sr-only" in child_cls:
                    return True
                if child.tag == "img":
                    return True
        return False
