"""WCAG 1.3.1 / 1.3.2 / 2.4.1-6 — Document structure rules."""
from __future__ import annotations

from rules.base import BaseRule, Finding, Severity
from rules.helpers import iter_elements, iter_files
from parsers.models import PHP_EXTENSIONS


class StructureRule(BaseRule):
    id = "structure"
    name = "Document Structure"
    wcag_criteria = ("1.3.1", "1.3.2", "2.4.1", "2.4.2", "2.4.6")
    standards = ("WCAG 2.2 AA",)

    def check(self, ctx, config) -> list[Finding]:
        findings: list[Finding] = []
        has_main = False
        has_nav = False
        has_footer = False

        # Collect per-file heading info and landmark presence
        headings_per_file: dict[str, list[tuple[int, int]]] = {}  # file -> [(level, line)]
        h1_per_file: dict[str, list[int]] = {}
        first_a_per_file: dict[str, tuple[int, str] | None] = {}
        has_title = False
        nav_labels: list[tuple[str, str, int]] = []  # (label, file, line)

        for path, fc, elem in iter_elements(ctx):
            tag = elem.tag.lower()

            if tag == "main":
                has_main = True
            elif tag == "nav":
                has_nav = True
                label = elem.attributes.get("aria-label", "")
                nav_labels.append((str(label) if label else "", path, elem.line))
            elif tag == "footer":
                has_footer = True
            elif tag == "title":
                if elem.text_content.strip():
                    has_title = True

            # headings
            if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
                level = int(tag[1])
                headings_per_file.setdefault(path, []).append((level, elem.line))
                if level == 1:
                    h1_per_file.setdefault(path, []).append(elem.line)

            # first <a> per file
            if tag == "a" and path not in first_a_per_file:
                href = str(elem.attributes.get("href", ""))
                first_a_per_file[path] = (elem.line, href)

        # landmarks
        if not has_main:
            findings.append(self._finding(
                check_id="landmarks",
                severity=Severity.SERIOUS,
                wcag="1.3.1",
                wcag_name="Info and Relationships",
                message="No <main> landmark found in any PHP file",
                file="(project)",
                line=0,
                fix="Add a <main> element to wrap primary content",
                impact=("blind", "cognitive"),
            ))
        if not has_nav:
            findings.append(self._finding(
                check_id="landmarks",
                severity=Severity.SERIOUS,
                wcag="1.3.1",
                wcag_name="Info and Relationships",
                message="No <nav> landmark found in any PHP file",
                file="(project)",
                line=0,
                fix="Add a <nav> element for navigation",
                impact=("blind", "cognitive"),
            ))

        # heading-hierarchy
        for path, headings in headings_per_file.items():
            levels = [h[0] for h in headings]
            for i in range(1, len(levels)):
                if levels[i] > levels[i - 1] + 1:
                    findings.append(self._finding(
                        check_id="heading-hierarchy",
                        severity=Severity.MODERATE,
                        wcag="1.3.1",
                        wcag_name="Info and Relationships",
                        message=f"Heading level skips from h{levels[i-1]} to h{levels[i]}",
                        file=path,
                        line=headings[i][1],
                        element=f"<h{levels[i]}>",
                        fix=f"Add an h{levels[i-1]+1} between h{levels[i-1]} and h{levels[i]}",
                        impact=("blind", "cognitive"),
                    ))

        # single-h1 (exclude includes/)
        for path, lines in h1_per_file.items():
            if "includes/" in path or "include/" in path:
                continue
            if len(lines) > 1:
                findings.append(self._finding(
                    check_id="single-h1",
                    severity=Severity.MODERATE,
                    wcag="1.3.1",
                    wcag_name="Info and Relationships",
                    message=f"Multiple <h1> elements found ({len(lines)})",
                    file=path,
                    line=lines[1],
                    element="<h1>",
                    fix="Use a single <h1> per page",
                ))

        # skip-link
        for path, fc in iter_files(ctx, PHP_EXTENSIONS):
            if "includes/" in path or "include/" in path:
                continue
            first = first_a_per_file.get(path)
            if first is None:
                continue
            line, href = first
            if not href.startswith("#"):
                findings.append(self._finding(
                    check_id="skip-link",
                    severity=Severity.SERIOUS,
                    wcag="2.4.1",
                    wcag_name="Bypass Blocks",
                    message="First <a> element is not a skip link",
                    file=path,
                    line=line,
                    element=f'<a href="{href}">',
                    fix='Add a skip link as the first element: <a href="#main">Skip to content</a>',
                    impact=("blind", "motor"),
                ))

        # page-title
        if not has_title:
            findings.append(self._finding(
                check_id="page-title",
                severity=Severity.SERIOUS,
                wcag="2.4.2",
                wcag_name="Page Titled",
                message="No <title> element found or title is empty",
                file="(project)",
                line=0,
                fix="Add a descriptive <title> element in <head>",
                impact=("blind",),
            ))

        # nav-aria-label
        if len(nav_labels) > 1:
            labels = [lbl for lbl, _, _ in nav_labels]
            for label, file, line in nav_labels:
                if not label:
                    findings.append(self._finding(
                        check_id="nav-aria-label",
                        severity=Severity.MODERATE,
                        wcag="1.3.1",
                        wcag_name="Info and Relationships",
                        message="Multiple <nav> elements but this one lacks aria-label",
                        file=file,
                        line=line,
                        element="<nav>",
                        fix="Add a unique aria-label to each <nav>",
                        impact=("blind",),
                    ))
            # Check for duplicate labels
            seen: set[str] = set()
            for label, file, line in nav_labels:
                if label and label in seen:
                    findings.append(self._finding(
                        check_id="nav-aria-label",
                        severity=Severity.MODERATE,
                        wcag="1.3.1",
                        wcag_name="Info and Relationships",
                        message=f"Duplicate nav aria-label: '{label}'",
                        file=file,
                        line=line,
                        element=f'<nav aria-label="{label}">',
                        fix="Use unique aria-label values for each <nav>",
                        impact=("blind",),
                    ))
                if label:
                    seen.add(label)

        # list-structure
        findings.extend(self._check_list_structure(ctx))

        return findings

    def _check_list_structure(self, ctx) -> list[Finding]:
        findings: list[Finding] = []
        for path, fc, elem in iter_elements(ctx):
            if elem.tag not in ("div", "span"):
                continue
            if len(elem.children) < 3:
                continue
            # Check if children share the same class (list-like pattern)
            child_classes = [str(c.attributes.get("class", "")) for c in elem.children]
            child_classes = [c for c in child_classes if c]
            if len(child_classes) >= 3:
                first = child_classes[0]
                if all(c == first for c in child_classes):
                    child_tags = {c.tag for c in elem.children}
                    if child_tags.issubset({"div", "span", "a", "p"}):
                        findings.append(self._finding(
                            check_id="list-structure",
                            severity=Severity.MINOR,
                            wcag="1.3.1",
                            wcag_name="Info and Relationships",
                            message=f"Div/span with repeated children looks like a list — {path}:{elem.line}",
                            file=path,
                            line=elem.line,
                            element=f"<{elem.tag}>",
                            fix="Use <ul>/<ol>/<li> for list-like content",
                        ))
        return findings
