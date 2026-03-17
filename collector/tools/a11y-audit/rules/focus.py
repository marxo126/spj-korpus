"""WCAG 2.4.3, 2.4.7, 2.4.11 — Focus management rules."""
from __future__ import annotations

import re

from rules.base import BaseRule, Finding, Severity
from rules.helpers import iter_elements, iter_files, parse_color, contrast_ratio
from parsers.models import JS_EXTENSIONS


_INTERACTIVE_TAGS = frozenset(("button", "a", "input", "select", "textarea"))

_REPLACEMENT_PROPS = frozenset((
    "box-shadow", "border", "border-color", "border-bottom",
    "border-top", "border-left", "border-right", "background",
    "background-color", "text-decoration",
))


class FocusRule(BaseRule):
    id = "focus"
    name = "Focus Management"
    wcag_criteria = ("2.4.3", "2.4.7", "2.4.11")
    standards = ("WCAG 2.2 AA",)

    def check(self, ctx, config) -> list[Finding]:
        findings: list[Finding] = []
        findings.extend(self._check_outline_none(ctx))
        findings.extend(self._check_focus_visible(ctx))
        findings.extend(self._check_focus_indicator_contrast(ctx))
        findings.extend(self._check_tabindex_positive(ctx))
        findings.extend(self._check_focus_trap(ctx))
        return findings

    def _check_outline_none(self, ctx) -> list[Finding]:
        """Scan CSS for outline:none on :focus without replacement style."""
        findings: list[Finding] = []
        for rule in ctx.css.rules:
            sel = rule.selector
            if ":focus" not in sel and ":focus-visible" not in sel:
                continue
            props = rule.properties
            outline = props.get("outline", "").strip().lower()
            if outline not in ("none", "0", "0px", "0px none"):
                continue
            # Check for replacement style
            has_replacement = any(
                props.get(p) for p in _REPLACEMENT_PROPS if p != "outline"
            )
            if not has_replacement:
                findings.append(self._finding(
                    check_id="outline-none",
                    severity=Severity.CRITICAL,
                    wcag="2.4.7",
                    wcag_name="Focus Visible",
                    message=f"outline:none on {sel} without replacement focus style",
                    file="style.css",
                    line=rule.line,
                    element=sel,
                    fix="Add visible focus style (box-shadow, border, etc.) or remove outline:none",
                    impact=("blind", "motor"),
                ))
        return findings

    def _check_focus_visible(self, ctx) -> list[Finding]:
        """Check that interactive elements have :focus or :focus-visible CSS."""
        findings: list[Finding] = []
        # Collect selectors that have focus rules
        focus_selectors: set[str] = set()
        for rule in ctx.css.rules:
            if ":focus" in rule.selector or ":focus-visible" in rule.selector:
                # Extract base selector before :focus
                base = re.split(r":focus(?:-visible)?", rule.selector)[0].strip()
                focus_selectors.add(base)

        # Check common interactive selectors
        interactive_selectors = {"button", "a", "input", "select", "textarea",
                                 "[type=\"submit\"]", "[type=\"button\"]"}
        # If no focus rules at all for any interactive element, flag
        for sel in interactive_selectors:
            # Check if any focus selector covers this element
            has_focus = any(
                sel in fs or fs == "" or fs == "*"
                for fs in focus_selectors
            )
            if not has_focus and focus_selectors:
                # Only flag if there are some focus rules but not for this element
                continue
            if not focus_selectors:
                findings.append(self._finding(
                    check_id="focus-visible",
                    severity=Severity.SERIOUS,
                    wcag="2.4.7",
                    wcag_name="Focus Visible",
                    message="No :focus or :focus-visible CSS rules found for interactive elements",
                    file="style.css",
                    line=0,
                    fix="Add :focus-visible styles for interactive elements",
                    impact=("motor", "blind"),
                ))
                break  # One finding is enough
        return findings

    def _check_focus_indicator_contrast(self, ctx) -> list[Finding]:
        """If focus style uses a color, check >= 3:1 contrast."""
        findings: list[Finding] = []
        for rule in ctx.css.rules:
            if ":focus" not in rule.selector and ":focus-visible" not in rule.selector:
                continue
            props = rule.properties
            # Check box-shadow, border-color, outline-color for contrast
            for prop_name in ("box-shadow", "border-color", "outline-color"):
                val = props.get(prop_name, "")
                if not val:
                    continue
                # Try to extract color from value
                color_match = re.search(
                    r"(#[0-9a-fA-F]{3,8}|rgb\([^)]+\)|rgba\([^)]+\))", val
                )
                if not color_match:
                    continue
                color_str = color_match.group(1)
                try:
                    ratio = contrast_ratio(color_str, "#ffffff")
                except ValueError:
                    continue
                if ratio < 3.0:
                    findings.append(self._finding(
                        check_id="focus-indicator-contrast",
                        severity=Severity.SERIOUS,
                        wcag="2.4.11",
                        wcag_name="Focus Not Obscured (Minimum)",
                        message=f"Focus indicator contrast {ratio}:1 below 3:1 — {rule.selector}",
                        file="style.css",
                        line=rule.line,
                        element=rule.selector,
                        fix=f"Increase focus indicator color contrast (currently {color_str})",
                        impact=("low-vision",),
                    ))
        return findings

    def _check_tabindex_positive(self, ctx) -> list[Finding]:
        """Check for tabindex > 0."""
        findings: list[Finding] = []
        for path, fc, elem in iter_elements(ctx):
            tabindex = elem.attributes.get("tabindex")
            if tabindex is None:
                continue
            try:
                val = int(str(tabindex))
            except ValueError:
                continue
            if val > 0:
                findings.append(self._finding(
                    check_id="tabindex-positive",
                    severity=Severity.MODERATE,
                    wcag="2.4.3",
                    wcag_name="Focus Order",
                    message=f"Positive tabindex={val} disrupts natural focus order",
                    file=path,
                    line=elem.line,
                    element=f"<{elem.tag} tabindex=\"{val}\">",
                    fix="Use tabindex=\"0\" or \"-1\" instead of positive values",
                    impact=("blind",),
                ))
        return findings

    def _check_focus_trap(self, ctx) -> list[Finding]:
        """Check that role=dialog elements have focus management."""
        findings: list[Finding] = []
        has_dialog = False
        for path, fc, elem in iter_elements(ctx):
            role = str(elem.attributes.get("role", "")).lower()
            if role in ("dialog", "alertdialog"):
                has_dialog = True

        if not has_dialog:
            return findings

        # Check JS for focus trap patterns
        has_focus_trap = False
        for path, fc in iter_files(ctx, JS_EXTENSIONS):
            content = fc.content.lower()
            if "focustrap" in content or "focus-trap" in content:
                has_focus_trap = True
                break
            if "queryselectorall" in content and "tabindex" in content:
                has_focus_trap = True
                break
            # Check for manual focus management in modals
            if ("modal" in content or "dialog" in content) and ".focus()" in content:
                has_focus_trap = True
                break

        if not has_focus_trap:
            findings.append(self._finding(
                check_id="focus-trap",
                severity=Severity.SERIOUS,
                wcag="2.4.3",
                wcag_name="Focus Order",
                message="Dialog/modal found without focus trap management in JS",
                file="(project)",
                line=0,
                fix="Implement focus trapping inside modal dialogs",
                impact=("motor", "blind"),
            ))
        return findings
