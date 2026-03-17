"""Tests for Rules Group 2: focus, keyboard, interactive, forms, media, language."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from parsers.models import (
    ParseContext, FileContext, ElementNode, CSSContext, CSSRule,
    EventListener,
)
from rules.base import Severity
from rules.focus import FocusRule
from rules.keyboard import KeyboardRule
from rules.interactive import InteractiveRule
from rules.forms import FormsRule
from rules.media import MediaRule
from rules.language import LanguageRule


# ── Helpers ──────────────────────────────────────────────────────────────

def _ctx(
    elements=None,
    css_rules=None,
    files=None,
    css_kwargs=None,
):
    css = CSSContext(
        rules=css_rules or [],
        **(css_kwargs or {}),
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


# ── FocusRule ────────────────────────────────────────────────────────────

class TestFocusRule:
    rule = FocusRule()

    def test_outline_none_violation(self):
        ctx = _ctx(css_rules=[
            CSSRule(selector="a:focus", properties={"outline": "none"}, line=5),
        ])
        findings = self.rule.check(ctx, {})
        on = [f for f in findings if f.check_id == "outline-none"]
        assert len(on) == 1
        assert on[0].severity == Severity.CRITICAL

    def test_outline_none_with_box_shadow_ok(self):
        ctx = _ctx(css_rules=[
            CSSRule(selector="a:focus", properties={
                "outline": "none", "box-shadow": "0 0 3px blue",
            }, line=5),
        ])
        findings = self.rule.check(ctx, {})
        on = [f for f in findings if f.check_id == "outline-none"]
        assert len(on) == 0

    def test_tabindex_positive(self):
        elem = _elem("div", {"tabindex": "5"})
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        tp = [f for f in findings if f.check_id == "tabindex-positive"]
        assert len(tp) == 1
        assert tp[0].severity == Severity.MODERATE

    def test_tabindex_zero_ok(self):
        elem = _elem("div", {"tabindex": "0"})
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        tp = [f for f in findings if f.check_id == "tabindex-positive"]
        assert len(tp) == 0

    def test_focus_visible_no_rules(self):
        ctx = _ctx(css_rules=[
            CSSRule(selector="button", properties={"color": "blue"}, line=1),
        ])
        findings = self.rule.check(ctx, {})
        fv = [f for f in findings if f.check_id == "focus-visible"]
        assert len(fv) == 1

    def test_focus_visible_with_rules_ok(self):
        ctx = _ctx(css_rules=[
            CSSRule(selector="button:focus-visible", properties={"outline": "2px solid"}, line=1),
        ])
        findings = self.rule.check(ctx, {})
        fv = [f for f in findings if f.check_id == "focus-visible"]
        assert len(fv) == 0


# ── KeyboardRule ─────────────────────────────────────────────────────────

class TestKeyboardRule:
    rule = KeyboardRule()

    def test_click_no_key_div(self):
        elem = _elem("div", {"onclick": "doSomething()"})
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        cnk = [f for f in findings if f.check_id == "click-no-key"]
        assert len(cnk) == 1

    def test_click_no_key_button_ok(self):
        elem = _elem("button", {"onclick": "doSomething()"})
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        cnk = [f for f in findings if f.check_id == "click-no-key"]
        assert len(cnk) == 0

    def test_click_with_keydown_ok(self):
        elem = _elem("div", {"onclick": "doSomething()", "onkeydown": "handleKey()"})
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        cnk = [f for f in findings if f.check_id == "click-no-key"]
        assert len(cnk) == 0

    def test_interactive_div_violation(self):
        elem = _elem("div", {"onclick": "handle()"})
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        idv = [f for f in findings if f.check_id == "interactive-div"]
        assert len(idv) == 1

    def test_interactive_div_with_role_ok(self):
        elem = _elem("div", {"onclick": "handle()", "role": "button", "tabindex": "0"})
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        idv = [f for f in findings if f.check_id == "interactive-div"]
        assert len(idv) == 0

    def test_mouse_only_handler(self):
        elem = _elem("div", {"onmouseover": "show()"})
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        moh = [f for f in findings if f.check_id == "mouse-only-handler"]
        assert len(moh) == 1

    def test_mouse_with_focus_ok(self):
        elem = _elem("div", {"onmouseover": "show()", "onfocus": "show()"})
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        moh = [f for f in findings if f.check_id == "mouse-only-handler"]
        assert len(moh) == 0

    def test_accesskey_duplicate(self):
        e1 = _elem("a", {"accesskey": "h"}, line=1)
        e2 = _elem("a", {"accesskey": "h"}, line=5)
        ctx = _ctx(elements=[e1, e2])
        findings = self.rule.check(ctx, {})
        ak = [f for f in findings if f.check_id == "accesskey-conflict"]
        assert len(ak) == 1


# ── InteractiveRule ──────────────────────────────────────────────────────

class TestInteractiveRule:
    rule = InteractiveRule()

    def test_hover_content_without_focus(self):
        ctx = _ctx(css_rules=[
            CSSRule(selector=".tooltip:hover .content", properties={"display": "block"}, line=3),
        ])
        findings = self.rule.check(ctx, {})
        hc = [f for f in findings if f.check_id == "hover-content"]
        assert len(hc) == 1

    def test_hover_content_with_focus_within_ok(self):
        ctx = _ctx(css_rules=[
            CSSRule(selector=".tooltip:hover .content", properties={"display": "block"}, line=3),
            CSSRule(selector=".tooltip:focus-within .content", properties={"display": "block"}, line=5),
        ])
        findings = self.rule.check(ctx, {})
        hc = [f for f in findings if f.check_id == "hover-content"]
        assert len(hc) == 0

    def test_touch_target_small(self):
        ctx = _ctx(css_rules=[
            CSSRule(selector="button.icon", properties={"min-height": "20px"}, line=1),
        ])
        findings = self.rule.check(ctx, {})
        tt = [f for f in findings if f.check_id == "touch-target"]
        assert len(tt) == 1

    def test_drag_alternative_flag(self):
        fc = FileContext(
            path="app.js",
            lines=["element.addEventListener('dragstart', handler);"],
            raw_content="element.addEventListener('dragstart', handler);",
        )
        ctx = _ctx(files={"app.js": fc})
        findings = self.rule.check(ctx, {})
        da = [f for f in findings if f.check_id == "drag-alternative"]
        assert len(da) == 1


# ── FormsRule ────────────────────────────────────────────────────────────

class TestFormsRule:
    rule = FormsRule()

    def test_input_without_label(self):
        elem = _elem("input", {"type": "text", "name": "foo"})
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        il = [f for f in findings if f.check_id == "input-label"]
        assert len(il) == 1
        assert il[0].severity == Severity.CRITICAL

    def test_input_with_label(self):
        label = _elem("label", {"for": "foo"})
        inp = _elem("input", {"type": "text", "id": "foo"})
        ctx = _ctx(elements=[label, inp])
        findings = self.rule.check(ctx, {})
        il = [f for f in findings if f.check_id == "input-label"]
        assert len(il) == 0

    def test_input_with_aria_label(self):
        inp = _elem("input", {"type": "text", "aria-label": "Search"})
        ctx = _ctx(elements=[inp])
        findings = self.rule.check(ctx, {})
        il = [f for f in findings if f.check_id == "input-label"]
        assert len(il) == 0

    def test_hidden_input_ok(self):
        inp = _elem("input", {"type": "hidden"})
        ctx = _ctx(elements=[inp])
        findings = self.rule.check(ctx, {})
        il = [f for f in findings if f.check_id == "input-label"]
        assert len(il) == 0

    def test_label_for_no_match(self):
        label = _elem("label", {"for": "missing"})
        ctx = _ctx(elements=[label])
        findings = self.rule.check(ctx, {})
        lf = [f for f in findings if f.check_id == "label-for"]
        assert len(lf) == 1

    def test_label_for_match_ok(self):
        label = _elem("label", {"for": "name"})
        inp = _elem("input", {"id": "name"})
        ctx = _ctx(elements=[label, inp])
        findings = self.rule.check(ctx, {})
        lf = [f for f in findings if f.check_id == "label-for"]
        assert len(lf) == 0

    def test_autocomplete_email(self):
        inp = _elem("input", {"type": "email", "name": "email"})
        ctx = _ctx(elements=[inp])
        findings = self.rule.check(ctx, {})
        ac = [f for f in findings if f.check_id == "autocomplete"]
        assert len(ac) == 1

    def test_autocomplete_present_ok(self):
        inp = _elem("input", {"type": "email", "autocomplete": "email"})
        ctx = _ctx(elements=[inp])
        findings = self.rule.check(ctx, {})
        ac = [f for f in findings if f.check_id == "autocomplete"]
        assert len(ac) == 0

    def test_error_identification(self):
        elem = _elem("div", {"class": "error-message"})
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        ei = [f for f in findings if f.check_id == "error-identification"]
        assert len(ei) == 1

    def test_error_with_role_alert_ok(self):
        elem = _elem("div", {"class": "error-message", "role": "alert"})
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        ei = [f for f in findings if f.check_id == "error-identification"]
        assert len(ei) == 0

    def test_submit_button_missing(self):
        form = _elem("form", children=[_elem("input", {"type": "text"})])
        ctx = _ctx(elements=[form])
        findings = self.rule.check(ctx, {})
        sb = [f for f in findings if f.check_id == "submit-button"]
        assert len(sb) == 1

    def test_submit_button_present(self):
        form = _elem("form", children=[
            _elem("input", {"type": "text"}),
            _elem("button", {"type": "submit"}),
        ])
        ctx = _ctx(elements=[form])
        findings = self.rule.check(ctx, {})
        sb = [f for f in findings if f.check_id == "submit-button"]
        assert len(sb) == 0


# ── MediaRule ────────────────────────────────────────────────────────────

class TestMediaRule:
    rule = MediaRule()

    def test_img_without_alt(self):
        elem = _elem("img", {"src": "photo.jpg"})
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        ia = [f for f in findings if f.check_id == "img-alt"]
        assert len(ia) == 1
        assert ia[0].severity == Severity.CRITICAL

    def test_img_with_alt(self):
        elem = _elem("img", {"src": "photo.jpg", "alt": "A photo"})
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        ia = [f for f in findings if f.check_id == "img-alt"]
        assert len(ia) == 0

    def test_img_decorative_alt_no_role(self):
        elem = _elem("img", {"src": "dot.png", "alt": ""})
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        da = [f for f in findings if f.check_id == "decorative-alt"]
        assert len(da) == 1

    def test_img_decorative_alt_with_role_ok(self):
        elem = _elem("img", {"src": "dot.png", "alt": "", "role": "presentation"})
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        da = [f for f in findings if f.check_id == "decorative-alt"]
        assert len(da) == 0

    def test_video_without_track(self):
        elem = _elem("video", {"src": "clip.mp4"})
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        vt = [f for f in findings if f.check_id == "video-track"]
        assert len(vt) == 1
        assert vt[0].severity == Severity.SERIOUS

    def test_video_with_captions(self):
        track = _elem("track", {"kind": "captions", "src": "sub.vtt"})
        video = _elem("video", children=[track])
        ctx = _ctx(elements=[video])
        findings = self.rule.check(ctx, {})
        vt = [f for f in findings if f.check_id == "video-track"]
        assert len(vt) == 0

    def test_video_autoplay_no_muted(self):
        elem = _elem("video", {"autoplay": True, "src": "bg.mp4"})
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        va = [f for f in findings if f.check_id == "video-autoplay"]
        assert len(va) == 1

    def test_video_autoplay_muted_ok(self):
        elem = _elem("video", {"autoplay": True, "muted": True, "src": "bg.mp4"})
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        va = [f for f in findings if f.check_id == "video-autoplay"]
        assert len(va) == 0

    def test_svg_no_label(self):
        elem = _elem("svg")
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        sa = [f for f in findings if f.check_id == "svg-accessible"]
        assert len(sa) == 1

    def test_svg_with_aria_hidden_ok(self):
        elem = _elem("svg", {"aria-hidden": "true"})
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        sa = [f for f in findings if f.check_id == "svg-accessible"]
        assert len(sa) == 0

    def test_svg_with_title_ok(self):
        title = _elem("title", text_content="Icon")
        elem = _elem("svg", children=[title])
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        sa = [f for f in findings if f.check_id == "svg-accessible"]
        assert len(sa) == 0


# ── LanguageRule ─────────────────────────────────────────────────────────

class TestLanguageRule:
    rule = LanguageRule()

    def test_missing_lang(self):
        elem = _elem("html")
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        hl = [f for f in findings if f.check_id == "html-lang"]
        assert len(hl) == 1
        assert hl[0].severity == Severity.CRITICAL

    def test_valid_lang(self):
        elem = _elem("html", {"lang": "sk"})
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        hl = [f for f in findings if f.check_id == "html-lang"]
        assert len(hl) == 0
        hv = [f for f in findings if f.check_id == "html-lang-valid"]
        assert len(hv) == 0

    def test_invalid_lang(self):
        elem = _elem("html", {"lang": "xyz"})
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        hv = [f for f in findings if f.check_id == "html-lang-valid"]
        assert len(hv) == 1

    def test_no_html_element(self):
        # Files without <html> (includes, API endpoints) should not be flagged
        ctx = _ctx(elements=[_elem("div")])
        findings = self.rule.check(ctx, {})
        hl = [f for f in findings if f.check_id == "html-lang"]
        assert len(hl) == 0

    def test_lang_subtag_ok(self):
        elem = _elem("html", {"lang": "sk-SK"})
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        hv = [f for f in findings if f.check_id == "html-lang-valid"]
        assert len(hv) == 0

    def test_lang_change_invalid(self):
        elem = _elem("span", {"lang": "xyz"})
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        lc = [f for f in findings if f.check_id == "lang-change"]
        assert len(lc) == 1

    def test_lang_change_valid_ok(self):
        elem = _elem("span", {"lang": "en"})
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        lc = [f for f in findings if f.check_id == "lang-change"]
        assert len(lc) == 0
