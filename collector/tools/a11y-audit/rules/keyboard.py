"""WCAG 2.1.1, 2.1.2, 2.1.4 — Keyboard accessibility rules."""
from __future__ import annotations

import re

from rules.base import BaseRule, Finding, Severity
from rules.helpers import iter_elements, iter_files
from parsers.models import JS_EXTENSIONS

_BTN_TARGET_RE = re.compile(
    r"""getElementById\(['"][\w-]*(?:btn|button|submit|record)""", re.IGNORECASE
)


class KeyboardRule(BaseRule):
    id = "keyboard"
    name = "Keyboard Accessibility"
    wcag_criteria = ("2.1.1", "2.1.2", "2.1.4")
    standards = ("WCAG 2.2 AA",)

    def check(self, ctx, config) -> list[Finding]:
        findings: list[Finding] = []
        findings.extend(self._check_click_no_key(ctx))
        findings.extend(self._check_mouse_only_handler(ctx))
        findings.extend(self._check_accesskey_conflict(ctx))
        findings.extend(self._check_interactive_div(ctx))
        return findings

    def _check_click_no_key(self, ctx) -> list[Finding]:
        """Check onclick without onkeydown/onkeypress on same element."""
        findings: list[Finding] = []
        # HTML attributes
        for path, fc, elem in iter_elements(ctx):
            attrs = elem.attributes
            has_click = "onclick" in attrs
            has_key = "onkeydown" in attrs or "onkeypress" in attrs or "onkeyup" in attrs
            if has_click and not has_key:
                # Buttons and links handle Enter natively
                if elem.tag.lower() in ("button", "a", "input", "select", "textarea"):
                    continue
                findings.append(self._finding(
                    check_id="click-no-key",
                    severity=Severity.SERIOUS,
                    wcag="2.1.1",
                    wcag_name="Keyboard",
                    message=f"onclick without keyboard handler on <{elem.tag}>",
                    file=path,
                    line=elem.line,
                    element=f"<{elem.tag} onclick=\"...\">",
                    fix="Add onkeydown or onkeypress handler for keyboard users",
                    impact=("motor", "blind"),
                ))

        # JS event listeners — approximate by checking click listeners
        # without nearby keydown listeners on same target
        for path, fc in iter_files(ctx, JS_EXTENSIONS):
            click_lines: list[int] = []
            key_lines: list[int] = []
            for listener in fc.event_listeners:
                if listener.event_type == "click":
                    # Skip .onclick property assignments on button-like targets
                    # (buttons handle Enter/Space natively)
                    if ".onclick" in listener.code and _BTN_TARGET_RE.search(listener.code):
                        continue
                    click_lines.append(listener.line)
                elif listener.event_type in ("keydown", "keypress", "keyup"):
                    key_lines.append(listener.line)

            for cl in click_lines:
                # Check if there's a key listener within 10 lines
                has_nearby_key = any(abs(cl - kl) <= 10 for kl in key_lines)
                if not has_nearby_key:
                    findings.append(self._finding(
                        check_id="click-no-key",
                        severity=Severity.SERIOUS,
                        wcag="2.1.1",
                        wcag_name="Keyboard",
                        message="click event listener without paired keydown/keypress handler",
                        file=path,
                        line=cl,
                        fix="Add keydown/keypress event listener for keyboard accessibility",
                        impact=("motor", "blind"),
                    ))
        return findings

    def _check_mouse_only_handler(self, ctx) -> list[Finding]:
        """Check onmouseover/onmouseout without onfocus/onblur."""
        findings: list[Finding] = []
        for path, fc, elem in iter_elements(ctx):
            attrs = elem.attributes
            has_mouseover = "onmouseover" in attrs
            has_mouseout = "onmouseout" in attrs
            has_focus = "onfocus" in attrs
            has_blur = "onblur" in attrs

            if has_mouseover and not has_focus:
                findings.append(self._finding(
                    check_id="mouse-only-handler",
                    severity=Severity.SERIOUS,
                    wcag="2.1.1",
                    wcag_name="Keyboard",
                    message=f"onmouseover without onfocus on <{elem.tag}>",
                    file=path,
                    line=elem.line,
                    element=f"<{elem.tag}>",
                    fix="Add onfocus handler alongside onmouseover",
                    impact=("motor",),
                ))
            if has_mouseout and not has_blur:
                findings.append(self._finding(
                    check_id="mouse-only-handler",
                    severity=Severity.SERIOUS,
                    wcag="2.1.1",
                    wcag_name="Keyboard",
                    message=f"onmouseout without onblur on <{elem.tag}>",
                    file=path,
                    line=elem.line,
                    element=f"<{elem.tag}>",
                    fix="Add onblur handler alongside onmouseout",
                    impact=("motor",),
                ))
        return findings

    def _check_accesskey_conflict(self, ctx) -> list[Finding]:
        """Scan for duplicate accesskey attributes."""
        findings: list[Finding] = []
        seen: dict[str, tuple[str, int]] = {}  # key -> (file, line)
        for path, fc, elem in iter_elements(ctx):
            ak = elem.attributes.get("accesskey")
            if ak is None:
                continue
            key = str(ak).strip().lower()
            if not key:
                continue
            if key in seen:
                prev_file, prev_line = seen[key]
                findings.append(self._finding(
                    check_id="accesskey-conflict",
                    severity=Severity.MODERATE,
                    wcag="2.1.4",
                    wcag_name="Character Key Shortcuts",
                    message=f"Duplicate accesskey='{key}' (also at {prev_file}:{prev_line})",
                    file=path,
                    line=elem.line,
                    element=f"<{elem.tag} accesskey=\"{key}\">",
                    fix="Use unique accesskey values or remove duplicates",
                ))
            else:
                seen[key] = (path, elem.line)
        return findings

    def _check_interactive_div(self, ctx) -> list[Finding]:
        """div/span with onclick but no role or tabindex."""
        findings: list[Finding] = []
        for path, fc, elem in iter_elements(ctx):
            tag = elem.tag.lower()
            if tag not in ("div", "span"):
                continue
            attrs = elem.attributes
            has_click = "onclick" in attrs
            has_role = "role" in attrs
            has_tabindex = "tabindex" in attrs

            if has_click and not has_role and not has_tabindex:
                findings.append(self._finding(
                    check_id="interactive-div",
                    severity=Severity.SERIOUS,
                    wcag="2.1.1",
                    wcag_name="Keyboard",
                    message=f"<{tag}> with onclick but no role or tabindex",
                    file=path,
                    line=elem.line,
                    element=f"<{tag} onclick=\"...\">",
                    fix=f"Add role=\"button\" and tabindex=\"0\", or use a <button> element",
                    impact=("motor", "blind"),
                ))
        return findings
