"""EN 301 549 Annex A — Accessibility statement compliance rules."""
from __future__ import annotations

import re

from rules.base import BaseRule, Finding, Severity
from rules.helpers import iter_files
from parsers.models import PHP_EXTENSIONS

_DATE_PATTERN = re.compile(
    r"(\d{1,2}\.\d{1,2}\.\d{4}|\d{4}-\d{2}-\d{2}|"
    r"(?:január|február|marec|apríl|máj|jún|júl|august|september|október|november|december))",
    re.IGNORECASE,
)
_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_URL_PATTERN = re.compile(r"https?://[^\s\"'<>]+")


class ComplianceRule(BaseRule):
    id = "compliance"
    name = "Accessibility Statement Compliance"
    wcag_criteria = ()
    standards = ("EN 301 549",)

    def check(self, ctx, config) -> list[Finding]:
        # Find accessibility statement file
        statement_path = None
        statement_content = ""
        for path, fc in iter_files(ctx, PHP_EXTENSIONS):
            if "accessibility" in path.lower() or "pristupnost" in path.lower():
                statement_path = path
                statement_content = fc.content.lower()
                break

        if not statement_path:
            return [self._finding(
                check_id="conformance-status",
                severity=Severity.CRITICAL,
                wcag="",
                wcag_name="EN 301 549 Annex A",
                message="No accessibility statement file found",
                file="(project)",
                line=0,
                fix="Create an accessibility statement (accessibility.php or pristupnost.php)",
                impact=("blind", "cognitive", "motor"),
            )]

        findings: list[Finding] = []
        findings.extend(self._check_conformance_status(statement_path, statement_content))
        findings.extend(self._check_non_accessible(statement_path, statement_content))
        findings.extend(self._check_preparation_date(statement_path, statement_content))
        findings.extend(self._check_review_method(statement_path, statement_content))
        findings.extend(self._check_feedback(statement_path, statement_content))
        findings.extend(self._check_enforcement(statement_path, statement_content))
        findings.extend(self._check_scope(statement_path, statement_content))
        findings.extend(self._check_standards_ref(statement_path, statement_content))
        return findings

    def _has_keywords(self, content: str, keywords: list[str]) -> bool:
        return any(k in content for k in keywords)

    def _check_conformance_status(self, path: str, content: str) -> list[Finding]:
        keywords = ["conformance", "conformita", "zhoda", "čiastočn", "úpln"]
        if self._has_keywords(content, keywords):
            return []
        return [self._finding(
            check_id="conformance-status",
            severity=Severity.CRITICAL,
            wcag="",
            wcag_name="EN 301 549 Annex A",
            message="Accessibility statement missing conformance status",
            file=path,
            line=0,
            fix="State conformance level (full, partial, or non-conformant)",
        )]

    def _check_non_accessible(self, path: str, content: str) -> list[Finding]:
        keywords = ["nedostupn", "obmedzeni", "limitation", "issue", "problém"]
        if self._has_keywords(content, keywords):
            return []
        return [self._finding(
            check_id="non-accessible-content",
            severity=Severity.SERIOUS,
            wcag="",
            wcag_name="EN 301 549 Annex A",
            message="Accessibility statement missing non-accessible content section",
            file=path,
            line=0,
            fix="Document known accessibility limitations and issues",
        )]

    def _check_preparation_date(self, path: str, content: str) -> list[Finding]:
        if _DATE_PATTERN.search(content):
            return []
        return [self._finding(
            check_id="preparation-date",
            severity=Severity.SERIOUS,
            wcag="",
            wcag_name="EN 301 549 Annex A",
            message="Accessibility statement missing preparation/review date",
            file=path,
            line=0,
            fix="Add date when the statement was prepared or last reviewed",
        )]

    def _check_review_method(self, path: str, content: str) -> list[Finding]:
        keywords = ["hodnoteni", "posúdeni", "self-assessment", "audit", "test"]
        if self._has_keywords(content, keywords):
            return []
        return [self._finding(
            check_id="review-method",
            severity=Severity.MODERATE,
            wcag="",
            wcag_name="EN 301 549 Annex A",
            message="Accessibility statement missing review/assessment method",
            file=path,
            line=0,
            fix="Describe how accessibility was assessed (self-assessment, external audit, etc.)",
        )]

    def _check_feedback(self, path: str, content: str) -> list[Finding]:
        contact_keywords = ["kontakt", "contact", "email", "spätná väzba", "feedback"]
        has_contact = self._has_keywords(content, contact_keywords)
        has_email = bool(_EMAIL_PATTERN.search(content))
        if has_contact and has_email:
            return []
        return [self._finding(
            check_id="feedback-mechanism",
            severity=Severity.CRITICAL,
            wcag="",
            wcag_name="EN 301 549 Annex A",
            message="Accessibility statement missing feedback mechanism with contact email",
            file=path,
            line=0,
            fix="Add contact information (email) for accessibility feedback",
        )]

    def _check_enforcement(self, path: str, content: str) -> list[Finding]:
        keywords = ["dozor", "enforcement", "orgán", "komisár"]
        has_keywords = self._has_keywords(content, keywords)
        has_url = bool(_URL_PATTERN.search(content))
        if has_keywords or has_url:
            return []
        return [self._finding(
            check_id="enforcement-link",
            severity=Severity.SERIOUS,
            wcag="",
            wcag_name="EN 301 549 Annex A",
            message="Accessibility statement missing enforcement procedure link",
            file=path,
            line=0,
            fix="Add link to enforcement body or complaint procedure",
        )]

    def _check_scope(self, path: str, content: str) -> list[Finding]:
        keywords = ["rozsah", "scope", "stránk", "page", "funkcionalit"]
        if self._has_keywords(content, keywords):
            return []
        return [self._finding(
            check_id="scope-defined",
            severity=Severity.MODERATE,
            wcag="",
            wcag_name="EN 301 549 Annex A",
            message="Accessibility statement missing scope definition",
            file=path,
            line=0,
            fix="Define which pages and functionality the statement covers",
        )]

    def _check_standards_ref(self, path: str, content: str) -> list[Finding]:
        keywords = ["wcag", "en 301 549", "smernic", "directive"]
        if self._has_keywords(content, keywords):
            return []
        return [self._finding(
            check_id="standards-reference",
            severity=Severity.SERIOUS,
            wcag="",
            wcag_name="EN 301 549 Annex A",
            message="Accessibility statement missing standards reference",
            file=path,
            line=0,
            fix="Reference WCAG 2.2 and/or EN 301 549",
        )]
