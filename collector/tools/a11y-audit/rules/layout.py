"""WCAG 1.4.10 / 2.5.8 — Layout and target size rules."""
from __future__ import annotations

import re

from rules.base import BaseRule, Finding, Severity
from rules.helpers import is_interactive_selector, iter_elements, parse_px

_INTERACTIVE_TAGS = frozenset(("button", "a", "input", "select", "textarea"))


class LayoutRule(BaseRule):
    id = "layout"
    name = "Layout & Target Size"
    wcag_criteria = ("1.4.10", "2.5.8")
    standards = ("WCAG 2.2 AA",)

    def check(self, ctx, config) -> list[Finding]:
        findings: list[Finding] = []
        thresholds = config.get("thresholds", {})
        target_aa = thresholds.get("target_size_aa", 24)
        target_enh = thresholds.get("target_size_enhanced", 44)

        # viewport-meta
        for path, fc, elem in iter_elements(ctx):
            if elem.tag == "meta":
                name = str(elem.attributes.get("name", "")).lower()
                if name == "viewport":
                    content = str(elem.attributes.get("content", ""))
                    findings.extend(self._check_viewport(content, path, elem.line))

        # CSS-based checks
        css = ctx.css
        for rule in css.rules:
            props = rule.properties
            sel = rule.selector.lower()

            # target-size checks on interactive selectors
            sel_is_interactive = is_interactive_selector(sel)
            if sel_is_interactive:
                self._check_target_size(
                    rule, props, findings, target_aa, target_enh,
                )

            # spacing — only flag when margin/gap is explicitly set to 0
            # or negative, not when it's simply absent.
            # Skip elements inside flex/grid containers with gap (parent handles spacing).
            if sel_is_interactive:
                margin = props.get("margin", "")
                gap = props.get("gap", "")
                margin_val = margin.strip().rstrip(";").strip() if margin else ""
                gap_val = gap.strip().rstrip(";").strip() if gap else ""
                # Flag explicit zero or negative spacing
                is_zero_margin = margin_val in ("0", "0px", "0em", "0rem")
                is_zero_gap = gap_val in ("0", "0px", "0em", "0rem")
                has_negative = margin_val.lstrip().startswith("-")
                # Skip child elements in containers that use gap for spacing
                # (e.g., .radio-group input where parent .radio-group has gap)
                if is_zero_margin and not is_zero_gap and not has_negative:
                    # Check if parent selector likely has gap
                    parent_parts = sel.rsplit(" ", 1)
                    if len(parent_parts) > 1:
                        parent_sel = parent_parts[0].strip()
                        for pr in css.rules:
                            if pr.selector.strip() == parent_sel:
                                if pr.properties.get("gap") or pr.properties.get("flex-wrap"):
                                    is_zero_margin = False
                                    break
                if is_zero_margin or is_zero_gap or has_negative:
                    findings.append(self._finding(
                        check_id="spacing",
                        severity=Severity.MODERATE,
                        wcag="2.5.8",
                        wcag_name="Target Size (Minimum)",
                        message=f"Zero/negative margin/gap between interactive elements — {rule.selector}",
                        file="style.css",
                        line=rule.line,
                        element=rule.selector,
                        fix="Add margin or gap to ensure adequate spacing between targets",
                        impact=("motor",),
                    ))

            # reflow-320
            if sel in ("body", "html", ".container", ".wrapper", "main"):
                ox = props.get("overflow-x", "").lower()
                if ox == "hidden":
                    findings.append(self._finding(
                        check_id="reflow-320",
                        severity=Severity.SERIOUS,
                        wcag="1.4.10",
                        wcag_name="Reflow",
                        message=f"overflow-x: hidden on {rule.selector} may break 320px reflow",
                        file="style.css",
                        line=rule.line,
                        element=rule.selector,
                        fix="Remove overflow-x: hidden or use a media query fallback",
                        impact=("low-vision",),
                    ))
                w = props.get("width", "")
                mw = props.get("min-width", "")
                for val, prop in ((w, "width"), (mw, "min-width")):
                    px = parse_px(val)
                    if px is not None and px > 320:
                        findings.append(self._finding(
                            check_id="reflow-320",
                            severity=Severity.SERIOUS,
                            wcag="1.4.10",
                            wcag_name="Reflow",
                            message=f"Fixed {prop}: {val} on {rule.selector} exceeds 320px",
                            file="style.css",
                            line=rule.line,
                            element=rule.selector,
                            fix=f"Use max-width or relative units instead of fixed {prop}",
                            impact=("low-vision",),
                        ))

        return findings

    def _check_viewport(self, content: str, path: str, line: int) -> list[Finding]:
        findings: list[Finding] = []
        content_lower = content.lower().replace(" ", "")
        if "user-scalable=no" in content_lower:
            findings.append(self._finding(
                check_id="viewport-meta",
                severity=Severity.SERIOUS,
                wcag="1.4.4",
                wcag_name="Resize Text",
                message="Viewport disables user scaling (user-scalable=no)",
                file=path,
                line=line,
                element='<meta name="viewport">',
                fix="Remove user-scalable=no to allow pinch-to-zoom",
                impact=("low-vision", "motor"),
            ))
        m = re.search(r"maximum-scale\s*=\s*([\d.]+)", content_lower)
        if m:
            scale = float(m.group(1))
            if scale < 2:
                findings.append(self._finding(
                    check_id="viewport-meta",
                    severity=Severity.SERIOUS,
                    wcag="1.4.4",
                    wcag_name="Resize Text",
                    message=f"Viewport maximum-scale={scale} restricts zoom below 2x",
                    file=path,
                    line=line,
                    element='<meta name="viewport">',
                    fix="Set maximum-scale to at least 2 or remove the restriction",
                    impact=("low-vision", "motor"),
                ))
        return findings

    def _check_target_size(self, rule, props, findings, target_aa, target_enh):
        min_h = parse_px(props.get("min-height", ""))
        min_w = parse_px(props.get("min-width", ""))
        height = parse_px(props.get("height", ""))
        width = parse_px(props.get("width", ""))

        size = min(
            min_h or height or 999,
            min_w or width or 999,
        )
        if size == 999:
            # Check padding as proxy
            p = props.get("padding", "")
            px = parse_px(p)
            if px is not None and px < 8:
                findings.append(self._finding(
                    check_id="target-size-aa",
                    severity=Severity.SERIOUS,
                    wcag="2.5.8",
                    wcag_name="Target Size (Minimum)",
                    message=f"Small padding ({p}) may make target too small — {rule.selector}",
                    file="style.css",
                    line=rule.line,
                    element=rule.selector,
                    fix=f"Ensure interactive target is at least {target_aa}px",
                    impact=("motor",),
                ))
            return

        if size < target_aa:
            findings.append(self._finding(
                check_id="target-size-aa",
                severity=Severity.SERIOUS,
                wcag="2.5.8",
                wcag_name="Target Size (Minimum)",
                message=f"Target size {size}px below {target_aa}px — {rule.selector}",
                file="style.css",
                line=rule.line,
                element=rule.selector,
                fix=f"Set min-height/min-width to at least {target_aa}px",
                impact=("motor",),
            ))
        elif size < target_enh:
            findings.append(self._finding(
                check_id="target-size-enhanced",
                severity=Severity.MINOR,
                wcag="2.5.8",
                wcag_name="Target Size (Minimum)",
                message=f"Target size {size}px below enhanced {target_enh}px — {rule.selector}",
                file="style.css",
                line=rule.line,
                element=rule.selector,
                fix=f"Consider increasing to {target_enh}px for enhanced accessibility",
                impact=("motor",),
            ))

