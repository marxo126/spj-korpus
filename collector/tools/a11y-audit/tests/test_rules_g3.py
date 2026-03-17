"""Tests for Rules Group 3: motion, cognitive, compliance, collector."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from parsers.models import (
    ParseContext, FileContext, ElementNode, CSSContext, CSSRule,
    KeyframeAnimation, TimeoutCall, EventListener,
)
from rules.base import Severity
from rules.motion import MotionRule
from rules.cognitive import CognitiveRule
from rules.compliance import ComplianceRule
from rules.collector import CollectorRule


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


# ── MotionRule ───────────────────────────────────────────────────────────

class TestMotionRule:
    rule = MotionRule()

    def test_missing_prefers_reduced_motion(self):
        ctx = _ctx(css_kwargs={
            "keyframes": [KeyframeAnimation(name="fade", duration_ms=300, iteration_count="1", line=5)],
            "has_prefers_reduced_motion": False,
        })
        findings = self.rule.check(ctx, {})
        prm = [f for f in findings if f.check_id == "prefers-reduced-motion"]
        assert len(prm) == 1
        assert prm[0].severity == Severity.SERIOUS

    def test_with_prefers_reduced_motion(self):
        ctx = _ctx(css_kwargs={
            "keyframes": [KeyframeAnimation(name="fade", duration_ms=300, iteration_count="1", line=5)],
            "has_prefers_reduced_motion": True,
        })
        findings = self.rule.check(ctx, {})
        prm = [f for f in findings if f.check_id == "prefers-reduced-motion"]
        assert len(prm) == 0

    def test_no_animations_ok(self):
        ctx = _ctx(css_kwargs={
            "keyframes": [],
            "has_prefers_reduced_motion": False,
        })
        findings = self.rule.check(ctx, {})
        prm = [f for f in findings if f.check_id == "prefers-reduced-motion"]
        assert len(prm) == 0

    def test_infinite_animation_with_reduced_motion(self):
        """Infinite animation is OK when prefers-reduced-motion is handled."""
        ctx = _ctx(css_kwargs={
            "keyframes": [KeyframeAnimation(name="spin", duration_ms=1000, iteration_count="infinite", line=10)],
            "has_prefers_reduced_motion": True,
        })
        findings = self.rule.check(ctx, {})
        ad = [f for f in findings if f.check_id == "animation-duration"]
        assert len(ad) == 0

    def test_infinite_animation_without_reduced_motion(self):
        """Infinite animation is flagged when no prefers-reduced-motion."""
        ctx = _ctx(css_kwargs={
            "keyframes": [KeyframeAnimation(name="spin", duration_ms=1000, iteration_count="infinite", line=10)],
            "has_prefers_reduced_motion": False,
        })
        findings = self.rule.check(ctx, {})
        ad = [f for f in findings if f.check_id == "animation-duration"]
        assert len(ad) == 1

    def test_finite_animation_ok(self):
        ctx = _ctx(css_kwargs={
            "keyframes": [KeyframeAnimation(name="slide", duration_ms=500, iteration_count="1", line=3)],
            "has_prefers_reduced_motion": True,
        })
        findings = self.rule.check(ctx, {})
        ad = [f for f in findings if f.check_id == "animation-duration"]
        assert len(ad) == 0

    def test_flash_name(self):
        ctx = _ctx(css_kwargs={
            "keyframes": [KeyframeAnimation(name="blink", duration_ms=200, iteration_count="infinite", line=1)],
            "has_prefers_reduced_motion": True,
        })
        findings = self.rule.check(ctx, {})
        fr = [f for f in findings if f.check_id == "flash-rate"]
        assert len(fr) == 1
        assert fr[0].severity == Severity.CRITICAL

    def test_normal_name_no_flash(self):
        ctx = _ctx(css_kwargs={
            "keyframes": [KeyframeAnimation(name="slideIn", duration_ms=300, iteration_count="1", line=1)],
            "has_prefers_reduced_motion": True,
        })
        findings = self.rule.check(ctx, {})
        fr = [f for f in findings if f.check_id == "flash-rate"]
        assert len(fr) == 0

    def test_toast_duration_short(self):
        fc = FileContext(
            path="app.js",
            lines=["// toast notification"],
            raw_content="// toast notification",
            timeouts=[TimeoutCall(function="setTimeout", duration_ms=2000, line=5, code="setTimeout(hideToast, 2000)")],
        )
        ctx = _ctx(files={"app.js": fc})
        findings = self.rule.check(ctx, {})
        td = [f for f in findings if f.check_id == "toast-duration"]
        assert len(td) == 1

    def test_toast_duration_ok(self):
        fc = FileContext(
            path="app.js",
            lines=["// toast notification"],
            raw_content="// toast notification",
            timeouts=[TimeoutCall(function="setTimeout", duration_ms=6000, line=5, code="setTimeout(hideToast, 6000)")],
        )
        ctx = _ctx(files={"app.js": fc})
        findings = self.rule.check(ctx, {})
        td = [f for f in findings if f.check_id == "toast-duration"]
        assert len(td) == 0


# ── CognitiveRule ────────────────────────────────────────────────────────

class TestCognitiveRule:
    rule = CognitiveRule()

    def test_error_prevention_no_confirm(self):
        fc = FileContext(
            path="admin.js",
            lines=["function deleteUser() {", "  fetch('/api/delete');", "}"],
            raw_content="function deleteUser() {\n  fetch('/api/delete');\n}",
        )
        ctx = _ctx(files={"admin.js": fc})
        findings = self.rule.check(ctx, {})
        ep = [f for f in findings if f.check_id == "error-prevention"]
        assert len(ep) >= 1

    def test_error_prevention_with_confirm_ok(self):
        fc = FileContext(
            path="admin.js",
            lines=["if (confirm('Are you sure?')) {", "  deleteRecord();", "}"],
            raw_content="if (confirm('Are you sure?')) {\n  deleteRecord();\n}",
        )
        ctx = _ctx(files={"admin.js": fc})
        findings = self.rule.check(ctx, {})
        ep = [f for f in findings if f.check_id == "error-prevention"]
        assert len(ep) == 0

    def test_redundant_entry(self):
        elem = _elem("input", {"type": "email", "name": "confirm_email"})
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        re_f = [f for f in findings if f.check_id == "redundant-entry"]
        assert len(re_f) == 1

    def test_on_focus_change(self):
        fc = FileContext(
            path="nav.js",
            lines=[
                "el.addEventListener('focus', function() {",
                "  window.location = '/page';",
                "});",
            ],
            raw_content="el.addEventListener('focus', function() {\n  window.location = '/page';\n});",
            event_listeners=[EventListener(event_type="focus", line=1, code="addEventListener('focus'...)", file="nav.js")],
        )
        ctx = _ctx(files={"nav.js": fc})
        findings = self.rule.check(ctx, {})
        ofc = [f for f in findings if f.check_id == "on-focus-change"]
        assert len(ofc) == 1


# ── ComplianceRule ───────────────────────────────────────────────────────

class TestComplianceRule:
    rule = ComplianceRule()

    def test_missing_statement(self):
        ctx = _ctx(elements=[_elem("div")])
        findings = self.rule.check(ctx, {})
        assert len(findings) == 1
        assert findings[0].check_id == "conformance-status"
        assert findings[0].severity == Severity.CRITICAL

    def test_complete_statement(self):
        content = """
        <h1>Vyhlásenie o prístupnosti</h1>
        <p>Stav zhoda: čiastočná conformance so štandardom WCAG 2.2.</p>
        <p>Nedostupné obmedzenia: niektoré obrázky nemajú alt text.</p>
        <p>Dátum posúdenia: 15.03.2026</p>
        <p>Hodnotenie bolo vykonané self-assessment.</p>
        <p>Kontakt pre spätnú väzbu: test@example.com</p>
        <p>Dozorný orgán: https://komisar.sk</p>
        <p>Rozsah: všetky stránky na doméne.</p>
        <p>Podľa smernice EN 301 549 a WCAG.</p>
        """
        fc = FileContext(
            path="accessibility.php",
            elements=[],
            raw_content=content.lower(),
            lines=content.lower().split("\n"),
        )
        ctx = _ctx(files={"accessibility.php": fc})
        findings = self.rule.check(ctx, {})
        assert len(findings) == 0

    def test_partial_statement(self):
        content = "vyhlásenie o prístupnosti. zhoda čiastočná."
        fc = FileContext(
            path="accessibility.php",
            elements=[],
            raw_content=content.lower(),
            lines=[content.lower()],
        )
        ctx = _ctx(files={"accessibility.php": fc})
        findings = self.rule.check(ctx, {})
        # Should have findings for missing sections
        check_ids = {f.check_id for f in findings}
        assert "non-accessible-content" in check_ids
        assert "preparation-date" in check_ids
        assert "feedback-mechanism" in check_ids


# ── CollectorRule ────────────────────────────────────────────────────────

class TestCollectorRule:
    rule = CollectorRule()

    def test_camera_no_error_handling(self):
        fc = FileContext(
            path="record.js",
            lines=["navigator.mediaDevices.getUserMedia({video: true})"],
            raw_content="navigator.mediaDevices.getUserMedia({video: true})",
        )
        ctx = _ctx(files={"record.js": fc})
        findings = self.rule.check(ctx, {})
        cf = [f for f in findings if f.check_id == "camera-fallback"]
        assert len(cf) == 1

    def test_camera_with_catch_ok(self):
        fc = FileContext(
            path="record.js",
            lines=[
                "navigator.mediaDevices.getUserMedia({video: true})",
                ".catch(err => showError(err));",
            ],
            raw_content="navigator.mediaDevices.getUserMedia({video: true})\n.catch(err => showError(err));",
        )
        ctx = _ctx(files={"record.js": fc})
        findings = self.rule.check(ctx, {})
        cf = [f for f in findings if f.check_id == "camera-fallback"]
        assert len(cf) == 0

    def test_quality_no_aria_live(self):
        elem = _elem("div", {"class": "quality-check badge"})
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        qa = [f for f in findings if f.check_id == "quality-aria-live"]
        assert len(qa) == 1

    def test_quality_with_aria_live_ok(self):
        elem = _elem("div", {"class": "quality-check", "aria-live": "polite"})
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        qa = [f for f in findings if f.check_id == "quality-aria-live"]
        assert len(qa) == 0

    def test_leaderboard_th_no_scope(self):
        th = _elem("th", text_content="Name")
        tr = _elem("tr", children=[th])
        table = _elem("table", children=[tr])
        ctx = _ctx(elements=[table])
        findings = self.rule.check(ctx, {})
        lt = [f for f in findings if f.check_id == "leaderboard-table"]
        assert len(lt) == 1

    def test_leaderboard_th_with_scope_ok(self):
        th = _elem("th", {"scope": "col"}, text_content="Name")
        tr = _elem("tr", children=[th])
        table = _elem("table", children=[tr])
        ctx = _ctx(elements=[table])
        findings = self.rule.check(ctx, {})
        lt = [f for f in findings if f.check_id == "leaderboard-table"]
        assert len(lt) == 0

    def test_timer_no_role(self):
        elem = _elem("div", {"class": "countdown-timer"})
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        ta = [f for f in findings if f.check_id == "timer-accessible"]
        assert len(ta) == 1

    def test_timer_with_role_ok(self):
        elem = _elem("div", {"class": "countdown-timer", "role": "timer"})
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        ta = [f for f in findings if f.check_id == "timer-accessible"]
        assert len(ta) == 0

    def test_recording_status_no_live(self):
        elem = _elem("span", {"class": "recording-indicator"})
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        rs = [f for f in findings if f.check_id == "recording-status"]
        assert len(rs) == 1

    def test_recording_status_with_live_ok(self):
        elem = _elem("span", {"class": "recording-indicator", "role": "status"})
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        rs = [f for f in findings if f.check_id == "recording-status"]
        assert len(rs) == 0

    def test_framing_guide_svg_no_alt(self):
        elem = _elem("svg", {"class": "frame-guide"})
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        fg = [f for f in findings if f.check_id == "framing-guide-alt"]
        assert len(fg) == 1

    def test_framing_guide_svg_with_label_ok(self):
        elem = _elem("svg", {"class": "frame-guide", "aria-label": "Framing guide overlay"})
        ctx = _ctx(elements=[elem])
        findings = self.rule.check(ctx, {})
        fg = [f for f in findings if f.check_id == "framing-guide-alt"]
        assert len(fg) == 0
