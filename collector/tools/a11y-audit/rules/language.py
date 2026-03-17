"""WCAG 3.1.1, 3.1.2 — Language rules."""
from __future__ import annotations

import re

from rules.base import BaseRule, Finding, Severity
from rules.helpers import iter_elements

_VALID_LANGS = frozenset({
    "af", "ar", "bg", "bn", "ca", "cs", "cy", "da", "de", "el", "en", "es",
    "et", "eu", "fa", "fi", "fr", "ga", "gl", "gu", "he", "hi", "hr", "hu",
    "hy", "id", "is", "it", "ja", "ka", "kk", "km", "kn", "ko", "lt", "lv",
    "mk", "ml", "mn", "mr", "ms", "my", "nb", "ne", "nl", "nn", "no", "pa",
    "pl", "pt", "ro", "ru", "si", "sk", "sl", "sq", "sr", "sv", "sw", "ta",
    "te", "th", "tl", "tr", "uk", "ur", "vi", "zh",
})


class LanguageRule(BaseRule):
    id = "language"
    name = "Language"
    wcag_criteria = ("3.1.1", "3.1.2")
    standards = ("WCAG 2.2 AA",)

    def check(self, ctx, config) -> list[Finding]:
        findings: list[Finding] = []
        findings.extend(self._check_html_lang(ctx))
        findings.extend(self._check_html_lang_valid(ctx))
        findings.extend(self._check_lang_change(ctx))
        return findings

    def _check_html_lang(self, ctx) -> list[Finding]:
        """<html> must have lang attribute."""
        findings: list[Finding] = []
        found_html = False
        for path, fc, elem in iter_elements(ctx):
            if elem.tag.lower() != "html":
                continue
            found_html = True
            lang = elem.attributes.get("lang")
            if not lang or not str(lang).strip():
                findings.append(self._finding(
                    check_id="html-lang",
                    severity=Severity.CRITICAL,
                    wcag="3.1.1",
                    wcag_name="Language of Page",
                    message="<html> element missing lang attribute",
                    file=path,
                    line=elem.line,
                    element="<html>",
                    fix="Add lang attribute, e.g. <html lang=\"sk\">",
                    impact=("blind",),
                ))
        # If no <html> element found at all, flag it
        if not found_html:
            findings.append(self._finding(
                check_id="html-lang",
                severity=Severity.CRITICAL,
                wcag="3.1.1",
                wcag_name="Language of Page",
                message="No <html> element with lang attribute found",
                file="(project)",
                line=0,
                fix="Add lang attribute to <html>, e.g. <html lang=\"sk\">",
                impact=("blind",),
            ))
        return findings

    def _check_html_lang_valid(self, ctx) -> list[Finding]:
        """lang value must be valid BCP 47."""
        findings: list[Finding] = []
        for path, fc, elem in iter_elements(ctx):
            if elem.tag.lower() != "html":
                continue
            lang = elem.attributes.get("lang")
            if not lang:
                continue
            lang_val = str(lang).strip().lower()
            # BCP 47: primary language subtag (first part before -)
            primary = lang_val.split("-")[0]
            if primary not in _VALID_LANGS:
                findings.append(self._finding(
                    check_id="html-lang-valid",
                    severity=Severity.SERIOUS,
                    wcag="3.1.1",
                    wcag_name="Language of Page",
                    message=f"Invalid lang value '{lang_val}'",
                    file=path,
                    line=elem.line,
                    element=f"<html lang=\"{lang_val}\">",
                    fix="Use a valid BCP 47 language code (e.g., 'sk', 'en', 'cs')",
                ))
        return findings

    def _check_lang_change(self, ctx) -> list[Finding]:
        """Check that elements with lang/xml:lang have valid values."""
        findings: list[Finding] = []
        for path, fc, elem in iter_elements(ctx):
            if elem.tag.lower() == "html":
                continue
            for attr in ("lang", "xml:lang"):
                val = elem.attributes.get(attr)
                if val is None:
                    continue
                lang_val = str(val).strip().lower()
                primary = lang_val.split("-")[0]
                if primary and primary not in _VALID_LANGS:
                    findings.append(self._finding(
                        check_id="lang-change",
                        severity=Severity.MINOR,
                        wcag="3.1.2",
                        wcag_name="Language of Parts",
                        message=f"Invalid {attr} value '{lang_val}' on <{elem.tag}>",
                        file=path,
                        line=elem.line,
                        element=f"<{elem.tag} {attr}=\"{lang_val}\">",
                        fix="Use a valid BCP 47 language code",
                    ))
        return findings
