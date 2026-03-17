"""WCAG 1.3.5, 3.3.1-8 — Form accessibility rules."""
from __future__ import annotations

import re

from rules.base import BaseRule, Finding, Severity
from rules.helpers import iter_elements, iter_files, has_accessible_name, line_of
from parsers.models import PHP_EXTENSIONS

_INPUT_TYPES_SKIP = frozenset(("hidden", "submit", "button", "image", "reset"))
_AUTOCOMPLETE_FIELDS = {
    "email": "email",
    "password": "current-password",
    "tel": "tel",
    "phone": "tel",
    "name": "name",
    "username": "username",
}
_REQUIRED_WORDS = ("*", "required", "povinné", "povinný", "povinne",
                   "povinny", "povinná", "povinna")
_ERROR_CLASSES = re.compile(r"\b(error|invalid|danger|alert-danger|form-error)\b", re.IGNORECASE)
_CAPTCHA_PATTERN = re.compile(r"\b(captcha|recaptcha|hcaptcha|g-recaptcha)\b", re.IGNORECASE)


class FormsRule(BaseRule):
    id = "forms"
    name = "Forms"
    wcag_criteria = ("1.3.5", "3.3.1", "3.3.2", "3.3.3", "3.3.4",
                     "3.3.7", "3.3.8")
    standards = ("WCAG 2.2 AA",)

    def check(self, ctx, config) -> list[Finding]:
        findings: list[Finding] = []
        findings.extend(self._check_input_label(ctx))
        findings.extend(self._check_label_for(ctx))
        findings.extend(self._check_autocomplete(ctx))
        findings.extend(self._check_error_identification(ctx))
        findings.extend(self._check_required_indicator(ctx))
        findings.extend(self._check_submit_button(ctx))
        findings.extend(self._check_accessible_auth(ctx))
        return findings

    def _check_input_label(self, ctx) -> list[Finding]:
        """Every input/select/textarea must have associated label or aria-label."""
        findings: list[Finding] = []
        # Collect all label[for] targets
        label_targets: set[str] = set()
        for path, fc, elem in iter_elements(ctx):
            if elem.tag.lower() == "label":
                for_val = elem.attributes.get("for")
                if for_val:
                    label_targets.add(str(for_val))

        for path, fc, elem in iter_elements(ctx):
            tag = elem.tag.lower()
            if tag not in ("input", "select", "textarea"):
                continue
            if tag == "input":
                input_type = str(elem.attributes.get("type", "text")).lower()
                if input_type in _INPUT_TYPES_SKIP:
                    continue

            # Check for label association
            elem_id = elem.attributes.get("id")
            has_label = (
                (elem_id and str(elem_id) in label_targets)
                or has_accessible_name(elem)
            )
            # Check if wrapped in a <label> (parent_tag)
            if elem.parent_tag and elem.parent_tag.lower() == "label":
                has_label = True

            if not has_label:
                findings.append(self._finding(
                    check_id="input-label",
                    severity=Severity.CRITICAL,
                    wcag="3.3.2",
                    wcag_name="Labels or Instructions",
                    message=f"<{tag}> without associated label or aria-label",
                    file=path,
                    line=elem.line,
                    element=f"<{tag}>",
                    fix="Add <label for=\"id\">, aria-label, or aria-labelledby",
                    impact=("blind",),
                ))
        return findings

    def _check_label_for(self, ctx) -> list[Finding]:
        """label[for] must match an existing element id in the same file."""
        findings: list[Finding] = []
        for path, fc in iter_files(ctx, PHP_EXTENSIONS):
            # Collect all ids in this file
            file_ids: set[str] = set()
            labels: list[tuple[str, int]] = []
            for elem in fc.elements:
                eid = elem.attributes.get("id")
                if eid:
                    file_ids.add(str(eid))
                if elem.tag.lower() == "label":
                    for_val = elem.attributes.get("for")
                    if for_val:
                        labels.append((str(for_val), elem.line))

            for for_val, line in labels:
                if for_val not in file_ids:
                    findings.append(self._finding(
                        check_id="label-for",
                        severity=Severity.SERIOUS,
                        wcag="3.3.2",
                        wcag_name="Labels or Instructions",
                        message=f"<label for=\"{for_val}\"> does not match any element id",
                        file=path,
                        line=line,
                        element=f"<label for=\"{for_val}\">",
                        fix=f"Ensure an element with id=\"{for_val}\" exists",
                    ))
        return findings

    def _check_autocomplete(self, ctx) -> list[Finding]:
        """Login/email/password inputs should have autocomplete attribute."""
        findings: list[Finding] = []
        for path, fc, elem in iter_elements(ctx):
            if elem.tag.lower() != "input":
                continue
            input_type = str(elem.attributes.get("type", "text")).lower()
            input_name = str(elem.attributes.get("name", "")).lower()
            autocomplete = elem.attributes.get("autocomplete")

            # Check type-based
            if input_type in ("email", "password", "tel") and not autocomplete:
                findings.append(self._finding(
                    check_id="autocomplete",
                    severity=Severity.MODERATE,
                    wcag="1.3.5",
                    wcag_name="Identify Input Purpose",
                    message=f"<input type=\"{input_type}\"> missing autocomplete attribute",
                    file=path,
                    line=elem.line,
                    element=f"<input type=\"{input_type}\">",
                    fix=f"Add autocomplete=\"{_AUTOCOMPLETE_FIELDS.get(input_type, input_type)}\"",
                    impact=("cognitive", "motor"),
                ))
                continue

            # Check name-based — skip if autocomplete="off" is explicit
            if input_type == "text" and not autocomplete:
                for keyword, ac_val in _AUTOCOMPLETE_FIELDS.items():
                    if keyword in input_name:
                        findings.append(self._finding(
                            check_id="autocomplete",
                            severity=Severity.MODERATE,
                            wcag="1.3.5",
                            wcag_name="Identify Input Purpose",
                            message=f"<input name=\"..{keyword}..\"> missing autocomplete attribute",
                            file=path,
                            line=elem.line,
                            element=f"<input name=\"{input_name}\">",
                            fix=f"Add autocomplete=\"{ac_val}\"",
                            impact=("cognitive", "motor"),
                        ))
                        break
        return findings

    def _check_error_identification(self, ctx) -> list[Finding]:
        """Elements with error/invalid class should have role=alert or aria-invalid."""
        findings: list[Finding] = []
        for path, fc, elem in iter_elements(ctx):
            cls = str(elem.attributes.get("class", ""))
            if not _ERROR_CLASSES.search(cls):
                continue
            role = str(elem.attributes.get("role", "")).lower()
            aria_invalid = elem.attributes.get("aria-invalid")
            aria_live = elem.attributes.get("aria-live")
            if role not in ("alert", "status") and not aria_invalid and not aria_live:
                findings.append(self._finding(
                    check_id="error-identification",
                    severity=Severity.SERIOUS,
                    wcag="3.3.1",
                    wcag_name="Error Identification",
                    message=f"Error element without role=\"alert\" or aria-invalid",
                    file=path,
                    line=elem.line,
                    element=f"<{elem.tag} class=\"{cls}\">",
                    fix="Add role=\"alert\" or aria-invalid=\"true\"",
                    impact=("blind", "cognitive"),
                ))
        return findings

    def _check_required_indicator(self, ctx) -> list[Finding]:
        """Required fields should indicate requirement beyond color."""
        findings: list[Finding] = []

        # Build a map of label[for] -> label text for the whole project
        label_texts: dict[str, str] = {}
        for _p, _fc, lelem in iter_elements(ctx):
            if lelem.tag.lower() == "label":
                for_val = lelem.attributes.get("for")
                if for_val:
                    label_texts[str(for_val)] = lelem.text_content

        for path, fc, elem in iter_elements(ctx):
            if elem.tag.lower() not in ("input", "select", "textarea"):
                continue
            required = elem.attributes.get("required")
            aria_required = str(elem.attributes.get("aria-required", "")).lower()
            if required is None and aria_required != "true":
                continue
            # Check if there's a visual indicator
            label = str(elem.attributes.get("aria-label", ""))
            title = str(elem.attributes.get("title", ""))
            indicator_texts = [label, title]

            # Check associated <label for="..."> text
            elem_id = elem.attributes.get("id")
            if elem_id and str(elem_id) in label_texts:
                indicator_texts.append(label_texts[str(elem_id)])

            # Check parent label text
            if elem.parent_tag and elem.parent_tag.lower() == "label":
                # parent text not directly available, but check nearby elements
                pass

            # Check for common required indicators in any associated text
            has_indicator = False
            for txt in indicator_texts:
                txt_lower = txt.lower()
                if any(w in txt_lower if w != "*" else w in txt
                       for w in _REQUIRED_WORDS):
                    has_indicator = True
                    break

            # aria-required="true" is a valid programmatic indicator
            if not has_indicator and aria_required == "true":
                has_indicator = True

            if has_indicator:
                continue
            findings.append(self._finding(
                check_id="required-indicator",
                severity=Severity.MODERATE,
                wcag="3.3.2",
                wcag_name="Labels or Instructions",
                message=f"Required <{elem.tag}> — verify visual indicator beyond color",
                file=path,
                line=elem.line,
                element=f"<{elem.tag} required>",
                fix="Add visual indicator (e.g., * or 'required' text) to the label",
            ))
        return findings

    def _check_submit_button(self, ctx) -> list[Finding]:
        """form elements should contain a submit button."""
        findings: list[Finding] = []
        for path, fc, elem in iter_elements(ctx):
            if elem.tag.lower() != "form":
                continue
            has_submit = self._has_submit_child(elem)
            if not has_submit:
                findings.append(self._finding(
                    check_id="submit-button",
                    severity=Severity.MODERATE,
                    wcag="3.3.2",
                    wcag_name="Labels or Instructions",
                    message="<form> without submit button",
                    file=path,
                    line=elem.line,
                    element="<form>",
                    fix="Add <button type=\"submit\"> or <input type=\"submit\">",
                ))
        return findings

    def _has_submit_child(self, elem) -> bool:
        for child in elem.children:
            tag = child.tag.lower()
            ctype = str(child.attributes.get("type", "")).lower()
            if tag == "button" and ctype in ("submit", ""):
                return True
            if tag == "input" and ctype == "submit":
                return True
            if self._has_submit_child(child):
                return True
        return False

    def _check_accessible_auth(self, ctx) -> list[Finding]:
        """Check for CAPTCHA without alternative."""
        findings: list[Finding] = []
        for path, fc in iter_files(ctx, PHP_EXTENSIONS):
            content = fc.content
            for m in _CAPTCHA_PATTERN.finditer(content):
                # Find line number
                line_num = line_of(content, m.start())
                findings.append(self._finding(
                    check_id="accessible-auth",
                    severity=Severity.SERIOUS,
                    wcag="3.3.8",
                    wcag_name="Accessible Authentication (Minimum)",
                    message="CAPTCHA detected — verify accessible alternative exists",
                    file=path,
                    line=line_num,
                    element=m.group(0),
                    fix="Provide alternative authentication (e.g., email link, passkey)",
                    impact=("cognitive",),
                ))
                break  # One finding per file
        return findings
