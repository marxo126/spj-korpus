"""WCAG 2.2.2, 2.3.1, 2.3.3 — Motion and animation rules."""
from __future__ import annotations

import re

from rules.base import BaseRule, Finding, Severity
from rules.helpers import iter_files
from parsers.models import JS_EXTENSIONS

_FLASH_NAMES = re.compile(r"\b(blink|flash|strobe)\b", re.IGNORECASE)
_SCROLL_PATTERN = re.compile(r"\b(scrollTo|scrollBy|scrollIntoView)\b")
_TOAST_PATTERN = re.compile(
    r"(toast|notification|snackbar|alert|message)", re.IGNORECASE
)


class MotionRule(BaseRule):
    id = "motion"
    name = "Motion and Animation"
    wcag_criteria = ("2.2.2", "2.3.1", "2.3.3")
    standards = ("WCAG 2.2 AA",)

    def check(self, ctx, config) -> list[Finding]:
        findings: list[Finding] = []
        findings.extend(self._check_prefers_reduced_motion(ctx))
        findings.extend(self._check_animation_duration(ctx))
        findings.extend(self._check_flash_rate(ctx))
        findings.extend(self._check_auto_scroll(ctx))
        findings.extend(self._check_toast_duration(ctx, config))
        return findings

    def _check_prefers_reduced_motion(self, ctx) -> list[Finding]:
        """If CSS has animations, check for prefers-reduced-motion."""
        findings: list[Finding] = []
        css = ctx.css
        has_animation = bool(css.keyframes)
        if not has_animation:
            # Check for animation/transition in rules
            for rule in css.rules:
                props = rule.properties
                if "animation" in props or "transition" in props:
                    has_animation = True
                    break

        if has_animation and not css.has_prefers_reduced_motion:
            findings.append(self._finding(
                check_id="prefers-reduced-motion",
                severity=Severity.SERIOUS,
                wcag="2.3.3",
                wcag_name="Animation from Interactions",
                message="CSS has animations but no @media (prefers-reduced-motion) query",
                file="style.css",
                line=0,
                fix="Add @media (prefers-reduced-motion: reduce) to disable animations",
                impact=("vestibular", "cognitive"),
            ))
        return findings

    def _check_animation_duration(self, ctx) -> list[Finding]:
        """Check @keyframes with infinite iteration."""
        findings: list[Finding] = []
        # If the stylesheet already handles prefers-reduced-motion globally,
        # infinite animations are acceptable (they get disabled for users
        # who prefer reduced motion).
        if ctx.css.has_prefers_reduced_motion:
            return findings
        for kf in ctx.css.keyframes:
            if kf.iteration_count.lower() == "infinite":
                findings.append(self._finding(
                    check_id="animation-duration",
                    severity=Severity.MODERATE,
                    wcag="2.2.2",
                    wcag_name="Pause, Stop, Hide",
                    message=f"Animation '{kf.name}' has infinite iteration",
                    file="style.css",
                    line=kf.line,
                    element=f"@keyframes {kf.name}",
                    fix="Provide pause/stop control or limit animation iterations",
                ))
        return findings

    def _check_flash_rate(self, ctx) -> list[Finding]:
        """Check for animations named blink/flash/strobe."""
        findings: list[Finding] = []
        for kf in ctx.css.keyframes:
            if _FLASH_NAMES.search(kf.name):
                findings.append(self._finding(
                    check_id="flash-rate",
                    severity=Severity.CRITICAL,
                    wcag="2.3.1",
                    wcag_name="Three Flashes or Below Threshold",
                    message=f"Animation '{kf.name}' may cause seizures",
                    file="style.css",
                    line=kf.line,
                    element=f"@keyframes {kf.name}",
                    fix="Remove or limit flashing to fewer than 3 flashes per second",
                    impact=("epilepsy",),
                ))
        return findings

    def _check_auto_scroll(self, ctx) -> list[Finding]:
        """Check for JS scroll methods combined with CSS overflow."""
        findings: list[Finding] = []
        # Check if CSS has overflow:auto/scroll
        has_overflow = False
        for rule in ctx.css.rules:
            for prop in ("overflow", "overflow-x", "overflow-y"):
                val = rule.properties.get(prop, "").lower()
                if val in ("auto", "scroll"):
                    has_overflow = True
                    break

        if not has_overflow:
            return findings

        for path, fc in iter_files(ctx, JS_EXTENSIONS):
            for i, line_text in enumerate(fc.lines, 1):
                if _SCROLL_PATTERN.search(line_text):
                    findings.append(self._finding(
                        check_id="auto-scroll",
                        severity=Severity.MODERATE,
                        wcag="2.2.2",
                        wcag_name="Pause, Stop, Hide",
                        message="Programmatic scrolling detected — verify user can pause/stop",
                        file=path,
                        line=i,
                        element=line_text.strip()[:80],
                        fix="Ensure auto-scrolling content can be paused or stopped",
                    ))
                    break  # One per file
        return findings

    def _check_toast_duration(self, ctx, config) -> list[Finding]:
        """Check setTimeout for toast/notification hiding with short duration."""
        findings: list[Finding] = []
        min_ms = config.get("thresholds", {}).get("toast_min_ms", 5000)

        for path, fc in iter_files(ctx, JS_EXTENSIONS):
            for timeout in fc.timeouts:
                if timeout.duration_ms is None:
                    continue
                if timeout.duration_ms >= min_ms:
                    continue
                # Check if the timeout is related to toast/notification
                code = timeout.code.lower()
                if _TOAST_PATTERN.search(code):
                    findings.append(self._finding(
                        check_id="toast-duration",
                        severity=Severity.MODERATE,
                        wcag="2.2.1",
                        wcag_name="Timing Adjustable",
                        message=f"Toast/notification auto-hides in {timeout.duration_ms}ms (minimum {min_ms}ms)",
                        file=path,
                        line=timeout.line,
                        element=timeout.code[:80],
                        fix=f"Increase notification display time to at least {min_ms}ms",
                        impact=("cognitive",),
                    ))
        return findings
