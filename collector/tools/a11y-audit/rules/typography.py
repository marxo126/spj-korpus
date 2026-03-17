"""WCAG 1.4.4 / 1.4.8 / 1.4.12 — Typography rules."""
from __future__ import annotations

import re

from rules.base import BaseRule, Finding, Severity


_PX_RE = re.compile(r"^([\d.]+)\s*px$", re.I)
_CH_RE = re.compile(r"^([\d.]+)\s*ch$", re.I)


class TypographyRule(BaseRule):
    id = "typography"
    name = "Typography"
    wcag_criteria = ("1.4.4", "1.4.8", "1.4.12")
    standards = ("WCAG 2.2 AA",)

    def check(self, ctx, config) -> list[Finding]:
        findings: list[Finding] = []
        thresholds = config.get("thresholds", {})
        min_font = thresholds.get("min_font_size", 14)
        min_lh = thresholds.get("min_line_height", 1.5)
        css = ctx.css

        for rule in css.rules:
            props = rule.properties

            # min-font-size
            fs = props.get("font-size", "")
            if fs:
                px = _parse_px(fs)
                if px is not None and px < min_font:
                    findings.append(self._finding(
                        check_id="min-font-size",
                        severity=Severity.MODERATE,
                        wcag="1.4.4",
                        wcag_name="Resize Text",
                        message=f"Font size {fs} below minimum {min_font}px — {rule.selector}",
                        file="style.css",
                        line=rule.line,
                        element=rule.selector,
                        fix=f"Use at least {min_font}px (or equivalent rem)",
                        impact=("low-vision",),
                    ))

            # relative-units
            if fs and fs.strip().lower().endswith("px"):
                findings.append(self._finding(
                    check_id="relative-units",
                    severity=Severity.MINOR,
                    wcag="1.4.4",
                    wcag_name="Resize Text",
                    message=f"Font size uses px units ({fs}) — {rule.selector}",
                    file="style.css",
                    line=rule.line,
                    element=rule.selector,
                    fix="Use rem or em instead of px for font-size",
                    impact=("low-vision",),
                ))

            # line-height
            lh = props.get("line-height", "")
            if lh:
                lh_val = _parse_unitless(lh)
                if lh_val is not None and lh_val < min_lh:
                    findings.append(self._finding(
                        check_id="line-height",
                        severity=Severity.MODERATE,
                        wcag="1.4.12",
                        wcag_name="Text Spacing",
                        message=f"Line height {lh} below {min_lh} — {rule.selector}",
                        file="style.css",
                        line=rule.line,
                        element=rule.selector,
                        fix=f"Set line-height to at least {min_lh}",
                        impact=("dyslexia", "low-vision"),
                    ))

            # letter-spacing
            ls = props.get("letter-spacing", "")
            if ls and ls.strip().startswith("-"):
                findings.append(self._finding(
                    check_id="letter-spacing",
                    severity=Severity.MODERATE,
                    wcag="1.4.12",
                    wcag_name="Text Spacing",
                    message=f"Negative letter-spacing ({ls}) — {rule.selector}",
                    file="style.css",
                    line=rule.line,
                    element=rule.selector,
                    fix="Avoid negative letter-spacing; use 0 or positive values",
                    impact=("dyslexia",),
                ))

            # text-justify
            ta = props.get("text-align", "")
            if ta.strip().lower() == "justify":
                findings.append(self._finding(
                    check_id="text-justify",
                    severity=Severity.MINOR,
                    wcag="1.4.8",
                    wcag_name="Visual Presentation",
                    message=f"text-align: justify used — {rule.selector}",
                    file="style.css",
                    line=rule.line,
                    element=rule.selector,
                    fix="Use text-align: left (or start) instead of justify",
                    impact=("dyslexia",),
                ))

            # line-length
            mw = props.get("max-width", "")
            if mw:
                ch_val = _parse_ch(mw)
                if ch_val is not None and ch_val > 80:
                    findings.append(self._finding(
                        check_id="line-length",
                        severity=Severity.MINOR,
                        wcag="1.4.8",
                        wcag_name="Visual Presentation",
                        message=f"Line length {mw} exceeds 80ch — {rule.selector}",
                        file="style.css",
                        line=rule.line,
                        element=rule.selector,
                        fix="Limit content width to 80ch or fewer",
                        impact=("dyslexia", "cognitive"),
                    ))

        return findings


def _parse_px(value: str) -> float | None:
    m = _PX_RE.match(value.strip())
    return float(m.group(1)) if m else None


def _parse_unitless(value: str) -> float | None:
    v = value.strip()
    try:
        return float(v)
    except ValueError:
        return None


def _parse_ch(value: str) -> float | None:
    m = _CH_RE.match(value.strip())
    return float(m.group(1)) if m else None
