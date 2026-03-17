"""Tests for Rules Group 1: contrast, color, typography, layout, structure, aria."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from parsers.models import (
    ParseContext, FileContext, ElementNode, CSSContext, CSSRule, CSSVariable,
)
from rules.base import Severity
from rules.contrast import ContrastRule
from rules.color import ColorRule
from rules.typography import TypographyRule
from rules.layout import LayoutRule
from rules.structure import StructureRule
from rules.aria import AriaRule


# ── Helpers ──────────────────────────────────────────────────────────────

def _ctx(
    elements: list[ElementNode] | None = None,
    css_rules: list[CSSRule] | None = None,
    css_vars: list[CSSVariable] | None = None,
    files: dict[str, FileContext] | None = None,
) -> ParseContext:
    css = CSSContext(
        rules=css_rules or [],
        variables=css_vars or [],
    )
    if files is None:
        fc = FileContext(path="index.php", elements=elements or [])
        files = {"index.php": fc}
    return ParseContext(files=files, css=css)


def _elem(tag, attrs=None, line=1, children=None, text_content="", parent_tag=None):
    return ElementNode(
        tag=tag,
        attributes=attrs or {},
        line=line,
        children=children or [],
        text_content=text_content,
        parent_tag=parent_tag,
    )


# ── ContrastRule ─────────────────────────────────────────────────────────

class TestContrastRule:
    rule = ContrastRule()

    def test_good_contrast(self):
        ctx = _ctx(css_rules=[
            CSSRule(selector=".text", properties={"color": "#000000", "background-color": "#ffffff"}, line=1),
        ])
        findings = self.rule.check(ctx, {})
        text_findings = [f for f in findings if f.check_id == "text-contrast"]
        assert len(text_findings) == 0

    def test_bad_contrast(self):
        ctx = _ctx(css_rules=[
            CSSRule(selector=".light", properties={"color": "#cccccc", "background-color": "#ffffff"}, line=5),
        ])
        findings = self.rule.check(ctx, {})
        text_findings = [f for f in findings if f.check_id == "text-contrast"]
        assert len(text_findings) == 1
        assert text_findings[0].severity == Severity.SERIOUS

    def test_large_text_lower_threshold(self):
        ctx = _ctx(css_rules=[
            CSSRule(selector="h1", properties={
                "color": "#767676", "background-color": "#ffffff", "font-size": "24px",
            }, line=1),
        ])
        findings = self.rule.check(ctx, {})
        text_findings = [f for f in findings if f.check_id == "text-contrast"]
        assert len(text_findings) == 0

    def test_ui_contrast_border(self):
        ctx = _ctx(css_rules=[
            CSSRule(selector="input", properties={
                "background-color": "#ffffff", "border-color": "#eeeeee",
            }, line=3),
        ])
        findings = self.rule.check(ctx, {})
        ui_findings = [f for f in findings if f.check_id == "ui-contrast"]
        assert len(ui_findings) == 1

    def test_placeholder_contrast(self):
        ctx = _ctx(css_rules=[
            CSSRule(selector="input::placeholder", properties={"color": "#dddddd"}, line=10),
        ])
        findings = self.rule.check(ctx, {})
        ph_findings = [f for f in findings if f.check_id == "placeholder-contrast"]
        assert len(ph_findings) == 1

    def test_css_var_resolution(self):
        ctx = _ctx(
            css_rules=[
                CSSRule(selector=".card", properties={
                    "color": "var(--text)", "background-color": "var(--bg)",
                }, line=1),
            ],
            css_vars=[
                CSSVariable(name="--text", value="#000", resolved_hex="#000000", mode="light"),
                CSSVariable(name="--bg", value="#fff", resolved_hex="#ffffff", mode="light"),
            ],
        )
        findings = self.rule.check(ctx, {})
        text_findings = [f for f in findings if f.check_id == "text-contrast"]
        assert len(text_findings) == 0


# ── ColorRule ────────────────────────────────────────────────────────────

class TestColorRule:
    rule = ColorRule()

    def test_color_only_status_violation(self):
        elem = _elem("span", {"class": "status-red"}, text_content="")
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        status_findings = [f for f in findings if f.check_id == "color-only-status"]
        assert len(status_findings) >= 1

    def test_color_status_with_icon_ok(self):
        elem = _elem("span", {"class": "status-green"}, text_content="✓ Done")
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        status_findings = [f for f in findings if f.check_id == "color-only-status"]
        assert len(status_findings) == 0

    def test_link_distinction(self):
        elem = _elem("a", {"href": "/page"})
        css_rules = [CSSRule(selector="a", properties={"text-decoration": "none"}, line=5)]
        ctx = _ctx(elements=[elem], css_rules=css_rules)
        findings = self.rule.check(ctx, {})
        link_findings = [f for f in findings if f.check_id == "link-distinction"]
        assert len(link_findings) >= 1

    def test_link_with_border_ok(self):
        elem = _elem("a", {"href": "/page"})
        css_rules = [CSSRule(selector="a", properties={
            "text-decoration": "none", "border-bottom": "1px solid",
        }, line=5)]
        ctx = _ctx(elements=[elem], css_rules=css_rules)
        findings = self.rule.check(ctx, {})
        link_findings = [f for f in findings if f.check_id == "link-distinction"]
        assert len(link_findings) == 0

    def test_error_color_only(self):
        elem = _elem("div", {"class": "error-message"}, text_content="")
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        err_findings = [f for f in findings if f.check_id == "error-color-only"]
        assert len(err_findings) >= 1


# ── TypographyRule ───────────────────────────────────────────────────────

class TestTypographyRule:
    rule = TypographyRule()

    def test_small_font_size(self):
        ctx = _ctx(css_rules=[
            CSSRule(selector=".small", properties={"font-size": "10px"}, line=3),
        ])
        findings = self.rule.check(ctx, {})
        fs_findings = [f for f in findings if f.check_id == "min-font-size"]
        assert len(fs_findings) == 1
        assert fs_findings[0].severity == Severity.MODERATE

    def test_ok_font_size(self):
        ctx = _ctx(css_rules=[
            CSSRule(selector="body", properties={"font-size": "16px"}, line=1),
        ])
        findings = self.rule.check(ctx, {})
        fs_findings = [f for f in findings if f.check_id == "min-font-size"]
        assert len(fs_findings) == 0

    def test_px_units_flagged(self):
        ctx = _ctx(css_rules=[
            CSSRule(selector="p", properties={"font-size": "16px"}, line=1),
        ])
        findings = self.rule.check(ctx, {})
        rel_findings = [f for f in findings if f.check_id == "relative-units"]
        assert len(rel_findings) == 1

    def test_rem_units_ok(self):
        ctx = _ctx(css_rules=[
            CSSRule(selector="p", properties={"font-size": "1rem"}, line=1),
        ])
        findings = self.rule.check(ctx, {})
        rel_findings = [f for f in findings if f.check_id == "relative-units"]
        assert len(rel_findings) == 0

    def test_low_line_height(self):
        ctx = _ctx(css_rules=[
            CSSRule(selector="p", properties={"line-height": "1.1"}, line=1),
        ])
        findings = self.rule.check(ctx, {})
        lh_findings = [f for f in findings if f.check_id == "line-height"]
        assert len(lh_findings) == 1

    def test_negative_letter_spacing(self):
        ctx = _ctx(css_rules=[
            CSSRule(selector="h1", properties={"letter-spacing": "-0.5px"}, line=1),
        ])
        findings = self.rule.check(ctx, {})
        ls_findings = [f for f in findings if f.check_id == "letter-spacing"]
        assert len(ls_findings) == 1

    def test_text_justify(self):
        ctx = _ctx(css_rules=[
            CSSRule(selector="p", properties={"text-align": "justify"}, line=1),
        ])
        findings = self.rule.check(ctx, {})
        tj_findings = [f for f in findings if f.check_id == "text-justify"]
        assert len(tj_findings) == 1

    def test_line_length_too_wide(self):
        ctx = _ctx(css_rules=[
            CSSRule(selector=".content", properties={"max-width": "100ch"}, line=1),
        ])
        findings = self.rule.check(ctx, {})
        ll_findings = [f for f in findings if f.check_id == "line-length"]
        assert len(ll_findings) == 1


# ── LayoutRule ───────────────────────────────────────────────────────────

class TestLayoutRule:
    rule = LayoutRule()

    def test_viewport_no_zoom(self):
        elem = _elem("meta", {"name": "viewport", "content": "width=device-width, user-scalable=no"})
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        vp_findings = [f for f in findings if f.check_id == "viewport-meta"]
        assert len(vp_findings) >= 1
        assert vp_findings[0].severity == Severity.SERIOUS

    def test_viewport_low_max_scale(self):
        elem = _elem("meta", {"name": "viewport", "content": "width=device-width, maximum-scale=1.0"})
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        vp_findings = [f for f in findings if f.check_id == "viewport-meta"]
        assert len(vp_findings) >= 1

    def test_viewport_ok(self):
        elem = _elem("meta", {"name": "viewport", "content": "width=device-width, initial-scale=1"})
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        vp_findings = [f for f in findings if f.check_id == "viewport-meta"]
        assert len(vp_findings) == 0

    def test_small_target_size(self):
        ctx = _ctx(css_rules=[
            CSSRule(selector="button.small", properties={"min-height": "16px", "min-width": "16px"}, line=1),
        ])
        findings = self.rule.check(ctx, {})
        ts_findings = [f for f in findings if f.check_id == "target-size-aa"]
        assert len(ts_findings) >= 1

    def test_reflow_fixed_width(self):
        ctx = _ctx(css_rules=[
            CSSRule(selector="body", properties={"min-width": "1024px"}, line=1),
        ])
        findings = self.rule.check(ctx, {})
        rf_findings = [f for f in findings if f.check_id == "reflow-320"]
        assert len(rf_findings) >= 1

    def test_reflow_overflow_hidden(self):
        ctx = _ctx(css_rules=[
            CSSRule(selector="body", properties={"overflow-x": "hidden"}, line=1),
        ])
        findings = self.rule.check(ctx, {})
        rf_findings = [f for f in findings if f.check_id == "reflow-320"]
        assert len(rf_findings) >= 1


# ── StructureRule ────────────────────────────────────────────────────────

class TestStructureRule:
    rule = StructureRule()

    def test_missing_landmarks(self):
        ctx = _ctx(elements=[_elem("div")])
        findings = self.rule.check(ctx, {})
        lm_findings = [f for f in findings if f.check_id == "landmarks"]
        assert len(lm_findings) >= 1

    def test_landmarks_present(self):
        elems = [_elem("main"), _elem("nav", {"aria-label": "Primary"}), _elem("footer")]
        ctx = _ctx(elements=elems)
        findings = self.rule.check(ctx, {})
        lm_findings = [f for f in findings if f.check_id == "landmarks"]
        assert len(lm_findings) == 0

    def test_heading_hierarchy_skip(self):
        elems = [_elem("h1", line=1), _elem("h3", line=5)]
        ctx = _ctx(elements=elems)
        findings = self.rule.check(ctx, {})
        hh_findings = [f for f in findings if f.check_id == "heading-hierarchy"]
        assert len(hh_findings) == 1

    def test_heading_hierarchy_ok(self):
        elems = [_elem("h1", line=1), _elem("h2", line=5), _elem("h3", line=10)]
        ctx = _ctx(elements=elems)
        findings = self.rule.check(ctx, {})
        hh_findings = [f for f in findings if f.check_id == "heading-hierarchy"]
        assert len(hh_findings) == 0

    def test_skip_link_present(self):
        elems = [_elem("a", {"href": "#main"}, line=1)]
        fc = FileContext(path="index.php", elements=elems)
        ctx = _ctx(files={"index.php": fc})
        findings = self.rule.check(ctx, {})
        sl_findings = [f for f in findings if f.check_id == "skip-link"]
        assert len(sl_findings) == 0

    def test_skip_link_missing(self):
        elems = [_elem("a", {"href": "/home"}, line=1)]
        fc = FileContext(path="index.php", elements=elems)
        ctx = _ctx(files={"index.php": fc})
        findings = self.rule.check(ctx, {})
        sl_findings = [f for f in findings if f.check_id == "skip-link"]
        assert len(sl_findings) == 1

    def test_page_title_present(self):
        elems = [_elem("title", text_content="My Page"), _elem("main"), _elem("nav")]
        ctx = _ctx(elements=elems)
        findings = self.rule.check(ctx, {})
        t_findings = [f for f in findings if f.check_id == "page-title"]
        assert len(t_findings) == 0

    def test_page_title_missing(self):
        ctx = _ctx(elements=[_elem("div")])
        findings = self.rule.check(ctx, {})
        t_findings = [f for f in findings if f.check_id == "page-title"]
        assert len(t_findings) == 1

    def test_multiple_h1(self):
        elems = [_elem("h1", line=1), _elem("h1", line=10)]
        fc = FileContext(path="page.php", elements=elems)
        ctx = _ctx(files={"page.php": fc})
        findings = self.rule.check(ctx, {})
        h1_findings = [f for f in findings if f.check_id == "single-h1"]
        assert len(h1_findings) == 1

    def test_nav_without_label(self):
        elems = [_elem("nav", line=1), _elem("nav", line=20)]
        ctx = _ctx(elements=elems)
        findings = self.rule.check(ctx, {})
        nav_findings = [f for f in findings if f.check_id == "nav-aria-label"]
        assert len(nav_findings) >= 1


# ── AriaRule ─────────────────────────────────────────────────────────────

class TestAriaRule:
    rule = AriaRule()

    def test_valid_role(self):
        elem = _elem("div", {"role": "button"})
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        vr_findings = [f for f in findings if f.check_id == "valid-role"]
        assert len(vr_findings) == 0

    def test_invalid_role(self):
        elem = _elem("div", {"role": "fancy-button"})
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        vr_findings = [f for f in findings if f.check_id == "valid-role"]
        assert len(vr_findings) == 1
        assert vr_findings[0].severity == Severity.CRITICAL

    def test_aria_hidden_focus_violation(self):
        child = _elem("button", line=5)
        parent = _elem("div", {"aria-hidden": "true"}, children=[child], line=3)
        ctx = _ctx(elements=[parent])
        findings = self.rule.check(ctx, {})
        ahf_findings = [f for f in findings if f.check_id == "aria-hidden-focus"]
        assert len(ahf_findings) == 1
        assert ahf_findings[0].severity == Severity.CRITICAL

    def test_aria_hidden_no_focusable_ok(self):
        child = _elem("span", line=5)
        parent = _elem("div", {"aria-hidden": "true"}, children=[child], line=3)
        ctx = _ctx(elements=[parent])
        findings = self.rule.check(ctx, {})
        ahf_findings = [f for f in findings if f.check_id == "aria-hidden-focus"]
        assert len(ahf_findings) == 0

    def test_empty_aria_label(self):
        elem = _elem("button", {"aria-label": ""})
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        al_findings = [f for f in findings if f.check_id == "aria-label-empty"]
        assert len(al_findings) == 1

    def test_nonempty_aria_label_ok(self):
        elem = _elem("button", {"aria-label": "Close"})
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        al_findings = [f for f in findings if f.check_id == "aria-label-empty"]
        assert len(al_findings) == 0

    def test_redundant_role(self):
        elem = _elem("nav", {"role": "navigation"})
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        rr_findings = [f for f in findings if f.check_id == "redundant-role"]
        assert len(rr_findings) == 1

    def test_required_attrs_missing(self):
        elem = _elem("div", {"role": "checkbox"})
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        ra_findings = [f for f in findings if f.check_id == "required-attrs"]
        assert len(ra_findings) == 1

    def test_required_attrs_present(self):
        elem = _elem("div", {"role": "checkbox", "aria-checked": "false"})
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        ra_findings = [f for f in findings if f.check_id == "required-attrs"]
        assert len(ra_findings) == 0

    def test_invalid_aria_live(self):
        elem = _elem("div", {"aria-live": "loud"})
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        lr_findings = [f for f in findings if f.check_id == "live-region-valid"]
        assert len(lr_findings) == 1

    def test_valid_aria_live(self):
        elem = _elem("div", {"aria-live": "polite"})
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        lr_findings = [f for f in findings if f.check_id == "live-region-valid"]
        assert len(lr_findings) == 0
