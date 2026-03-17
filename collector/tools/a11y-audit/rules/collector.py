"""Domain-specific rules for the sign language video collector app."""
from __future__ import annotations

import re

from rules.base import BaseRule, Finding, Severity
from rules.helpers import iter_elements, iter_files, has_accessible_name
from parsers.models import JS_EXTENSIONS, PHP_EXTENSIONS

_CAMERA_PATTERN = re.compile(r"\b(getUserMedia|navigator\.mediaDevices)\b")
_CAMERA_ERROR = re.compile(r"\b(catch|onerror|NotAllowedError|NotFoundError|permission)\b", re.IGNORECASE)
_QUALITY_CLASSES = re.compile(r"\b(quality|status|badge|check-result)\b", re.IGNORECASE)
_CONSENT_PATTERN = re.compile(r"\b(consent|gdpr|súhlas|modal|dialog)\b", re.IGNORECASE)
_ESCAPE_PATTERN = re.compile(r"""(Escape|escape|27|keyCode\s*===?\s*27|key\s*===?\s*['"]Escape['"])""")
_MODAL_CLOSE = re.compile(r"\b(close|dismiss|hide|skry[tť])\b", re.IGNORECASE)
_FOCUS_RETURN = re.compile(r"\.focus\(\)")
_RECORDING_CLASSES = re.compile(r"\b(recording|timer|countdown|status)\b", re.IGNORECASE)
_OFFLINE_PATTERN = re.compile(r"\b(offline|retry|sync|navigator\.onLine|reconnect)\b", re.IGNORECASE)


class CollectorRule(BaseRule):
    id = "collector"
    name = "Sign Collector Domain Rules"
    wcag_criteria = ("1.1.1", "1.3.1", "2.1.2", "4.1.3")
    standards = ("WCAG 2.2 AA", "EN 301 549")

    def check(self, ctx, config) -> list[Finding]:
        findings: list[Finding] = []
        findings.extend(self._check_camera_fallback(ctx))
        findings.extend(self._check_quality_aria_live(ctx))
        findings.extend(self._check_consent_keyboard_trap(ctx))
        findings.extend(self._check_consent_focus_return(ctx))
        findings.extend(self._check_leaderboard_table(ctx))
        findings.extend(self._check_recording_status(ctx))
        findings.extend(self._check_timer_accessible(ctx))
        findings.extend(self._check_offline_notification(ctx))
        findings.extend(self._check_framing_guide_alt(ctx))
        return findings

    def _check_camera_fallback(self, ctx) -> list[Finding]:
        """Camera permission patterns should have error handling."""
        findings: list[Finding] = []
        for path, fc in iter_files(ctx, JS_EXTENSIONS):
            content = fc.content
            if not _CAMERA_PATTERN.search(content):
                continue
            if _CAMERA_ERROR.search(content):
                continue
            # Find line of getUserMedia
            for i, line_text in enumerate(fc.lines, 1):
                if _CAMERA_PATTERN.search(line_text):
                    findings.append(self._finding(
                        check_id="camera-fallback",
                        severity=Severity.SERIOUS,
                        wcag="4.1.3",
                        wcag_name="Status Messages",
                        message="getUserMedia without error/denial handling",
                        file=path,
                        line=i,
                        fix="Add catch block for camera permission denial with user-facing message",
                        impact=("blind", "motor"),
                    ))
                    break
        return findings

    def _check_quality_aria_live(self, ctx) -> list[Finding]:
        """Quality check elements should be in aria-live region."""
        findings: list[Finding] = []
        for path, fc, elem in iter_elements(ctx):
            cls = str(elem.attributes.get("class", ""))
            if not _QUALITY_CLASSES.search(cls):
                continue
            # Check if element or ancestor has aria-live
            aria_live = elem.attributes.get("aria-live")
            role = str(elem.attributes.get("role", "")).lower()
            if aria_live or role in ("status", "alert", "log"):
                continue
            findings.append(self._finding(
                check_id="quality-aria-live",
                severity=Severity.SERIOUS,
                wcag="4.1.3",
                wcag_name="Status Messages",
                message=f"Quality/status element without aria-live region",
                file=path,
                line=elem.line,
                element=f"<{elem.tag} class=\"{cls}\">",
                fix="Add aria-live=\"polite\" or role=\"status\" to announce updates",
                impact=("blind",),
            ))
        return findings

    def _check_consent_keyboard_trap(self, ctx) -> list[Finding]:
        """Consent/GDPR modals must have Escape key handler."""
        findings: list[Finding] = []
        # Check if there are consent/GDPR modal elements
        has_consent_modal = False
        for path, fc, elem in iter_elements(ctx):
            cls = str(elem.attributes.get("class", "")) + " " + str(elem.attributes.get("id", ""))
            role = str(elem.attributes.get("role", "")).lower()
            if _CONSENT_PATTERN.search(cls) and (role in ("dialog", "alertdialog") or "modal" in cls.lower()):
                has_consent_modal = True
                break
            if _CONSENT_PATTERN.search(cls) and elem.tag.lower() in ("div", "section"):
                has_consent_modal = True
                break

        if not has_consent_modal:
            return findings

        # Check JS for Escape key handler
        has_escape = False
        for path, fc in iter_files(ctx, JS_EXTENSIONS):
            if _ESCAPE_PATTERN.search(fc.content):
                has_escape = True
                break

        if not has_escape:
            findings.append(self._finding(
                check_id="consent-keyboard-trap",
                severity=Severity.CRITICAL,
                wcag="2.1.2",
                wcag_name="No Keyboard Trap",
                message="Consent/GDPR modal without Escape key handler",
                file="(project)",
                line=0,
                fix="Add Escape key listener to close consent modal",
                impact=("motor", "blind"),
            ))
        return findings

    def _check_consent_focus_return(self, ctx) -> list[Finding]:
        """Modal close should return focus."""
        findings: list[Finding] = []
        for path, fc in iter_files(ctx, JS_EXTENSIONS):
            content = fc.content
            if not _MODAL_CLOSE.search(content):
                continue
            if _FOCUS_RETURN.search(content):
                continue
            # Only flag if there are modals
            has_modal = any(
                "modal" in str(elem.attributes.get("class", "")).lower()
                or str(elem.attributes.get("role", "")).lower() in ("dialog", "alertdialog")
                for p, fc2, elem in iter_elements(ctx)
            )
            if has_modal:
                findings.append(self._finding(
                    check_id="consent-focus-return",
                    severity=Severity.MODERATE,
                    wcag="2.4.3",
                    wcag_name="Focus Order",
                    message="Modal close without .focus() call to return focus",
                    file=path,
                    line=0,
                    fix="Call .focus() on the trigger element after closing modal",
                ))
                break
        return findings

    def _check_leaderboard_table(self, ctx) -> list[Finding]:
        """Tables should have <th> with scope attribute."""
        findings: list[Finding] = []
        for path, fc, elem in iter_elements(ctx):
            if elem.tag.lower() != "table":
                continue
            has_th_with_scope = False
            has_th = False
            for child in _walk(elem):
                if child.tag.lower() == "th":
                    has_th = True
                    if child.attributes.get("scope"):
                        has_th_with_scope = True
            if has_th and not has_th_with_scope:
                findings.append(self._finding(
                    check_id="leaderboard-table",
                    severity=Severity.MODERATE,
                    wcag="1.3.1",
                    wcag_name="Info and Relationships",
                    message="<th> without scope attribute in table",
                    file=path,
                    line=elem.line,
                    element="<table>",
                    fix="Add scope=\"col\" or scope=\"row\" to <th> elements",
                    impact=("blind",),
                ))
        return findings

    def _check_recording_status(self, ctx) -> list[Finding]:
        """Recording state elements should have aria-live."""
        findings: list[Finding] = []
        for path, fc, elem in iter_elements(ctx):
            cls = str(elem.attributes.get("class", ""))
            elem_id = str(elem.attributes.get("id", ""))
            combined = cls + " " + elem_id
            if not _RECORDING_CLASSES.search(combined):
                continue
            # Skip generic status classes that aren't recording-specific
            if "recording" not in combined.lower() and "timer" not in combined.lower():
                continue
            aria_live = elem.attributes.get("aria-live")
            role = str(elem.attributes.get("role", "")).lower()
            if aria_live or role in ("status", "timer", "alert"):
                continue
            findings.append(self._finding(
                check_id="recording-status",
                severity=Severity.SERIOUS,
                wcag="4.1.3",
                wcag_name="Status Messages",
                message="Recording/timer element without aria-live or role=\"status\"",
                file=path,
                line=elem.line,
                element=f"<{elem.tag} class=\"{cls}\">",
                fix="Add aria-live=\"assertive\" or role=\"timer\" to announce state changes",
                impact=("blind",),
            ))
        return findings

    def _check_timer_accessible(self, ctx) -> list[Finding]:
        """Timer/countdown elements should have role=timer or aria-live."""
        findings: list[Finding] = []
        for path, fc, elem in iter_elements(ctx):
            cls = str(elem.attributes.get("class", ""))
            elem_id = str(elem.attributes.get("id", ""))
            combined = (cls + " " + elem_id).lower()
            if "timer" not in combined and "countdown" not in combined:
                continue
            role = str(elem.attributes.get("role", "")).lower()
            aria_live = elem.attributes.get("aria-live")
            if role == "timer" or aria_live:
                continue
            findings.append(self._finding(
                check_id="timer-accessible",
                severity=Severity.MODERATE,
                wcag="4.1.3",
                wcag_name="Status Messages",
                message="Timer/countdown element without role=\"timer\" or aria-live",
                file=path,
                line=elem.line,
                element=f"<{elem.tag}>",
                fix="Add role=\"timer\" or aria-live=\"polite\"",
                impact=("blind",),
            ))
        return findings

    def _check_offline_notification(self, ctx) -> list[Finding]:
        """Offline/retry patterns should announce to assistive tech."""
        findings: list[Finding] = []
        # Also check if PHP templates have aria-live regions for toasts/status
        has_template_announce = False
        for p, fc2, elem in iter_elements(ctx):
            aria_live = elem.attributes.get("aria-live")
            role = str(elem.attributes.get("role", "")).lower()
            if aria_live or role in ("status", "alert"):
                cls = str(elem.attributes.get("class", "")).lower()
                eid = str(elem.attributes.get("id", "")).lower()
                if any(kw in cls + " " + eid for kw in ("toast", "announce", "notification", "recording")):
                    has_template_announce = True
                    break

        for path, fc in iter_files(ctx, JS_EXTENSIONS):
            content = fc.content
            if not _OFFLINE_PATTERN.search(content):
                continue
            # Check if there's an aria-live or role=alert in JS or in templates
            has_announce = (
                has_template_announce
                or "aria-live" in content
                or 'role="alert"' in content
                or "role='alert'" in content
                or 'role="status"' in content
            )
            if not has_announce:
                for i, line_text in enumerate(fc.lines, 1):
                    if _OFFLINE_PATTERN.search(line_text):
                        findings.append(self._finding(
                            check_id="offline-notification",
                            severity=Severity.MODERATE,
                            wcag="4.1.3",
                            wcag_name="Status Messages",
                            message="Offline/retry pattern without assistive tech announcement",
                            file=path,
                            line=i,
                            fix="Use aria-live region or role=\"alert\" to announce connection status",
                            impact=("blind",),
                        ))
                        break
        return findings

    def _check_framing_guide_alt(self, ctx) -> list[Finding]:
        """SVG in framing/guide context should have text alternative."""
        findings: list[Finding] = []
        for path, fc, elem in iter_elements(ctx):
            if elem.tag.lower() != "svg":
                continue
            cls = str(elem.attributes.get("class", "")).lower()
            elem_id = str(elem.attributes.get("id", "")).lower()
            combined = cls + " " + elem_id
            if "frame" not in combined and "guide" not in combined and "overlay" not in combined:
                continue
            if has_accessible_name(elem):
                continue
            aria_hidden = str(elem.attributes.get("aria-hidden", "")).lower()
            if aria_hidden == "true":
                continue
            has_title = any(c.tag.lower() == "title" for c in elem.children)
            if has_title:
                continue
            findings.append(self._finding(
                check_id="framing-guide-alt",
                severity=Severity.MODERATE,
                wcag="1.1.1",
                wcag_name="Non-text Content",
                message="Framing guide SVG without text alternative",
                file=path,
                line=elem.line,
                element="<svg>",
                fix="Add aria-label or <title> to describe the framing guide",
                impact=("blind",),
            ))
        return findings


def _walk(elem):
    """Walk all descendants of an element."""
    for child in elem.children:
        yield child
        yield from _walk(child)
