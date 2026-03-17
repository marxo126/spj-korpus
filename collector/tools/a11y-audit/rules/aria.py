"""WCAG 4.1.2 — ARIA rules."""
from __future__ import annotations

from rules.base import BaseRule, Finding, Severity
from rules.helpers import iter_elements

_VALID_ROLES = frozenset({
    "alert", "alertdialog", "application", "article", "banner", "button",
    "cell", "checkbox", "columnheader", "combobox", "complementary",
    "contentinfo", "definition", "dialog", "directory", "document", "feed",
    "figure", "form", "grid", "gridcell", "group", "heading", "img", "link",
    "list", "listbox", "listitem", "log", "main", "marquee", "math", "menu",
    "menubar", "menuitem", "menuitemcheckbox", "menuitemradio", "navigation",
    "none", "note", "option", "presentation", "progressbar", "radio",
    "radiogroup", "region", "row", "rowgroup", "rowheader", "scrollbar",
    "search", "searchbox", "separator", "slider", "spinbutton", "status",
    "switch", "tab", "table", "tablist", "tabpanel", "term", "textbox",
    "timer", "toolbar", "tooltip", "tree", "treegrid", "treeitem",
})

_REDUNDANT_PAIRS: dict[str, str] = {
    "nav": "navigation",
    "main": "main",
    "header": "banner",
    "footer": "contentinfo",
    "aside": "complementary",
    "form": "form",
    "section": "region",
    "article": "article",
    "button": "button",
    "a": "link",
    "select": "listbox",
    "textarea": "textbox",
    "table": "table",
    "ul": "list",
    "ol": "list",
    "li": "listitem",
    "img": "img",
}

_REQUIRED_ATTRS: dict[str, list[str]] = {
    "checkbox": ["aria-checked"],
    "slider": ["aria-valuenow"],
}

_FOCUSABLE_TAGS = frozenset(("a", "button", "input", "select", "textarea"))

_VALID_LIVE_VALUES = frozenset(("polite", "assertive", "off"))


class AriaRule(BaseRule):
    id = "aria"
    name = "ARIA"
    wcag_criteria = ("4.1.2",)
    standards = ("WCAG 2.2 AA",)

    def check(self, ctx, config) -> list[Finding]:
        findings: list[Finding] = []

        for path, fc, elem in iter_elements(ctx):
            role = str(elem.attributes.get("role", "")).strip().lower()

            # valid-role
            if role and role not in _VALID_ROLES:
                findings.append(self._finding(
                    check_id="valid-role",
                    severity=Severity.CRITICAL,
                    wcag="4.1.2",
                    wcag_name="Name, Role, Value",
                    message=f"Invalid ARIA role '{role}'",
                    file=path,
                    line=elem.line,
                    element=f'<{elem.tag} role="{role}">',
                    fix=f"Use a valid ARIA role (see WAI-ARIA spec)",
                    impact=("blind",),
                ))

            # required-attrs
            if role in _REQUIRED_ATTRS:
                for attr in _REQUIRED_ATTRS[role]:
                    if attr not in elem.attributes:
                        findings.append(self._finding(
                            check_id="required-attrs",
                            severity=Severity.SERIOUS,
                            wcag="4.1.2",
                            wcag_name="Name, Role, Value",
                            message=f"role='{role}' missing required attribute '{attr}'",
                            file=path,
                            line=elem.line,
                            element=f'<{elem.tag} role="{role}">',
                            fix=f"Add {attr} attribute",
                            impact=("blind",),
                        ))

            # Also check input type for redundant role
            tag = elem.tag.lower()
            input_type = str(elem.attributes.get("type", "")).lower()
            if tag == "input":
                type_role_map = {
                    "text": "textbox",
                    "checkbox": "checkbox",
                    "radio": "radio",
                }
                implicit = type_role_map.get(input_type)
                if implicit and role == implicit:
                    findings.append(self._finding(
                        check_id="redundant-role",
                        severity=Severity.MINOR,
                        wcag="4.1.2",
                        wcag_name="Name, Role, Value",
                        message=f"Redundant role='{role}' on <input type='{input_type}'>",
                        file=path,
                        line=elem.line,
                        element=f'<input type="{input_type}" role="{role}">',
                        fix=f"Remove role='{role}' — <input type='{input_type}'> has implicit role",
                    ))

            # redundant-role (tag-level)
            if tag in _REDUNDANT_PAIRS and role == _REDUNDANT_PAIRS[tag]:
                findings.append(self._finding(
                    check_id="redundant-role",
                    severity=Severity.MINOR,
                    wcag="4.1.2",
                    wcag_name="Name, Role, Value",
                    message=f"Redundant role='{role}' on <{tag}>",
                    file=path,
                    line=elem.line,
                    element=f'<{tag} role="{role}">',
                    fix=f"Remove role='{role}' — <{tag}> has implicit role",
                ))

            # aria-hidden-focus
            hidden = elem.attributes.get("aria-hidden")
            if str(hidden).lower() == "true":
                if self._contains_focusable(elem):
                    findings.append(self._finding(
                        check_id="aria-hidden-focus",
                        severity=Severity.CRITICAL,
                        wcag="4.1.2",
                        wcag_name="Name, Role, Value",
                        message="aria-hidden='true' contains focusable elements",
                        file=path,
                        line=elem.line,
                        element=f"<{elem.tag} aria-hidden='true'>",
                        fix="Remove aria-hidden or add tabindex='-1' to focusable children",
                        impact=("blind",),
                    ))

            # aria-label-empty
            for attr in ("aria-label", "aria-labelledby"):
                val = elem.attributes.get(attr)
                if val is not None and str(val).strip() == "":
                    findings.append(self._finding(
                        check_id="aria-label-empty",
                        severity=Severity.SERIOUS,
                        wcag="4.1.2",
                        wcag_name="Name, Role, Value",
                        message=f"Empty {attr} attribute on <{elem.tag}>",
                        file=path,
                        line=elem.line,
                        element=f"<{elem.tag} {attr}=''>",
                        fix=f"Provide a meaningful value for {attr} or remove it",
                        impact=("blind",),
                    ))

            # live-region-valid
            live = elem.attributes.get("aria-live")
            if live is not None:
                live_val = str(live).strip().lower()
                if live_val and live_val not in _VALID_LIVE_VALUES:
                    findings.append(self._finding(
                        check_id="live-region-valid",
                        severity=Severity.MODERATE,
                        wcag="4.1.2",
                        wcag_name="Name, Role, Value",
                        message=f"Invalid aria-live value '{live_val}'",
                        file=path,
                        line=elem.line,
                        element=f'<{elem.tag} aria-live="{live_val}">',
                        fix="Use 'polite', 'assertive', or 'off'",
                    ))

        return findings

    def _contains_focusable(self, elem) -> bool:
        if elem.tag.lower() in _FOCUSABLE_TAGS:
            return True
        tabindex = elem.attributes.get("tabindex")
        if tabindex is not None:
            try:
                if int(str(tabindex)) >= 0:
                    return True
            except ValueError:
                pass
        for child in elem.children:
            if self._contains_focusable(child):
                return True
        return False
