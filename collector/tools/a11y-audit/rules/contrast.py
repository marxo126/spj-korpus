"""WCAG 1.4.3 / 1.4.11 — Contrast rules."""
from __future__ import annotations

from rules.base import BaseRule, Finding, Severity
from rules.helpers import (
    contrast_ratio,
    is_large_text,
    parse_color,
    parse_px,
    resolve_css_var,
)


class ContrastRule(BaseRule):
    id = "contrast"
    name = "Color Contrast"
    wcag_criteria = ("1.4.3", "1.4.11")
    standards = ("WCAG 2.2 AA",)

    def check(self, ctx, config) -> list[Finding]:
        findings: list[Finding] = []
        thresholds = config.get("thresholds", {})
        text_min = thresholds.get("contrast_text", 4.5)
        ui_min = thresholds.get("contrast_ui", 3.0)
        css = ctx.css

        for rule in css.rules:
            props = rule.properties
            fg = self._resolve(props.get("color"), css)
            bg = self._resolve(props.get("background-color") or props.get("background"), css)

            # text-contrast
            if fg and bg:
                try:
                    ratio = contrast_ratio(fg, bg)
                except ValueError:
                    continue
                threshold = text_min
                fs = props.get("font-size", "")
                bold = props.get("font-weight", "") in ("bold", "700", "800", "900")
                if fs:
                    px = parse_px(fs)
                    if px and is_large_text(px, bold):
                        threshold = 3.0
                if ratio < threshold:
                    findings.append(self._finding(
                        check_id="text-contrast",
                        severity=Severity.SERIOUS,
                        wcag="1.4.3",
                        wcag_name="Contrast (Minimum)",
                        message=f"Text contrast {ratio}:1 below {threshold}:1 — {rule.selector}",
                        file="style.css",
                        line=rule.line,
                        element=rule.selector,
                        fix=f"Increase contrast between color ({fg}) and background ({bg})",
                        impact=("low-vision",),
                    ))

            # ui-contrast — border/outline vs background
            if bg:
                for prop_name in ("border-color", "outline-color"):
                    border_val = self._resolve(props.get(prop_name), css)
                    if not border_val:
                        continue
                    try:
                        ratio = contrast_ratio(border_val, bg)
                    except ValueError:
                        continue
                    if ratio < ui_min:
                        findings.append(self._finding(
                            check_id="ui-contrast",
                            severity=Severity.SERIOUS,
                            wcag="1.4.11",
                            wcag_name="Non-text Contrast",
                            message=f"UI contrast {ratio}:1 below 3:1 — {rule.selector} {prop_name}",
                            file="style.css",
                            line=rule.line,
                            element=rule.selector,
                            fix=f"Increase contrast between {prop_name} ({border_val}) and background ({bg})",
                            impact=("low-vision",),
                        ))

        # dark-mode-contrast — re-check with dark mode variables
        findings.extend(self._check_dark_mode(css, text_min, ui_min))

        # placeholder-contrast
        findings.extend(self._check_placeholder(css, text_min))

        return findings

    def _check_dark_mode(self, css, text_min, ui_min) -> list[Finding]:
        findings: list[Finding] = []

        import re as _re

        def _clean_sel(s: str) -> str:
            """Strip comments and whitespace from selector for comparison."""
            s = _re.sub(r"/\*.*?\*/", "", s, flags=_re.DOTALL)
            return " ".join(s.split()).strip()

        # Build set of base selectors that have explicit html.dark overrides
        # (these override colors explicitly, so variable-based checks are wrong)
        dark_overridden: set[str] = set()
        for rule in css.rules:
            sel = _clean_sel(rule.selector)
            # Handle comma-separated selectors like "html.dark .form-group input, html.dark .form-group select"
            for part in sel.split(","):
                part = part.strip()
                if part.startswith("html.dark "):
                    base = part[len("html.dark "):].strip()
                    dark_overridden.add(base)

        for rule in css.rules:
            # Only check rules that explicitly set BOTH color and background
            props = rule.properties
            fg_raw = props.get("color")
            bg_raw = props.get("background-color") or props.get("background")
            if not fg_raw or not bg_raw:
                continue
            # Skip rules already under html.dark (they define the dark theme itself)
            if "html.dark" in rule.selector or ".dark " in rule.selector:
                continue
            # Check if ALL sub-selectors of this rule have dark overrides
            sel_clean = _clean_sel(rule.selector)
            sel_parts = [p.strip() for p in sel_clean.split(",")]
            all_overridden = all(p in dark_overridden for p in sel_parts)
            if all_overridden:
                continue
            fg = self._resolve(fg_raw, css, mode="dark")
            bg = self._resolve(bg_raw, css, mode="dark")
            if not fg or not bg:
                continue
            # Skip if neither value changed in dark mode (no CSS vars involved)
            if fg == self._resolve(fg_raw, css, mode="light") and bg == self._resolve(bg_raw, css, mode="light"):
                continue
            # Skip if fg and bg resolve to the same value (mis-paired)
            if fg == bg:
                continue
            try:
                ratio = contrast_ratio(fg, bg)
            except ValueError:
                continue
            if ratio < text_min:
                findings.append(self._finding(
                    check_id="dark-mode-contrast",
                    severity=Severity.MODERATE,
                    wcag="1.4.3",
                    wcag_name="Contrast (Minimum)",
                    message=f"Dark mode contrast {ratio}:1 below {text_min}:1 — {rule.selector}",
                    file="style.css",
                    line=rule.line,
                    element=rule.selector,
                    fix="Check dark mode color variables provide sufficient contrast",
                    impact=("low-vision",),
                ))
        return findings

    def _check_placeholder(self, css, text_min) -> list[Finding]:
        findings: list[Finding] = []
        for rule in css.rules:
            if "::placeholder" not in rule.selector:
                continue
            fg = self._resolve(rule.properties.get("color"), css)
            if not fg:
                continue
            bg = "#ffffff"  # assume white background unless specified
            try:
                ratio = contrast_ratio(fg, bg)
            except ValueError:
                continue
            if ratio < text_min:
                findings.append(self._finding(
                    check_id="placeholder-contrast",
                    severity=Severity.MODERATE,
                    wcag="1.4.3",
                    wcag_name="Contrast (Minimum)",
                    message=f"Placeholder contrast {ratio}:1 below {text_min}:1 — {rule.selector}",
                    file="style.css",
                    line=rule.line,
                    element=rule.selector,
                    fix="Increase placeholder text contrast",
                    impact=("low-vision", "cognitive"),
                ))
        return findings

    @staticmethod
    def _resolve(value: str | None, css, mode: str = "light") -> str | None:
        if not value:
            return None
        value = value.strip()
        if value.startswith("var("):
            var_name = value.split("(", 1)[1].rstrip(")")
            # handle fallback: var(--x, #fff)
            parts = var_name.split(",", 1)
            var_name = parts[0].strip()
            resolved = resolve_css_var(var_name, css, mode=mode)
            if resolved:
                return resolved
            if len(parts) > 1:
                return parts[1].strip()
            return None
        return value
