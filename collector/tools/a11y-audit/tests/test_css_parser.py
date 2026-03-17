"""Tests for CSS parser."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from parsers.css_parser import parse_css_string


def test_variable_extraction_light_mode():
    css = ":root { --primary: #3b82f6; --bg: #ffffff; }"
    ctx = parse_css_string(css)
    assert len(ctx.variables) == 2
    names = {v.name for v in ctx.variables}
    assert "--primary" in names
    assert "--bg" in names
    primary = next(v for v in ctx.variables if v.name == "--primary")
    assert primary.resolved_hex == "#3b82f6"
    assert primary.mode == "light"


def test_dark_mode_variable_detection():
    css = """
    :root { --bg: #ffffff; }
    html.dark { --bg: #1a1a1a; }
    """
    ctx = parse_css_string(css)
    dark_vars = [v for v in ctx.variables if v.mode == "dark"]
    assert len(dark_vars) >= 1
    assert dark_vars[0].name == "--bg"
    assert dark_vars[0].value == "#1a1a1a"


def test_dark_mode_via_media_query():
    css = """
    :root { --text: #333; }
    @media (prefers-color-scheme: dark) {
        :root { --text: #eee; }
    }
    """
    ctx = parse_css_string(css)
    dark_vars = [v for v in ctx.variables if v.mode == "dark"]
    assert len(dark_vars) >= 1


def test_keyframes_extraction():
    css = """
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .modal { animation: fadeIn 0.3s ease; }
    """
    ctx = parse_css_string(css)
    assert len(ctx.keyframes) == 1
    assert ctx.keyframes[0].name == "fadeIn"
    assert ctx.keyframes[0].duration_ms == 300.0


def test_prefers_reduced_motion_detection():
    css = """
    @media (prefers-reduced-motion: reduce) {
        * { animation: none !important; }
    }
    """
    ctx = parse_css_string(css)
    assert ctx.has_prefers_reduced_motion is True


def test_prefers_contrast_detection():
    css = "@media (prefers-contrast: high) { .btn { border: 2px solid; } }"
    ctx = parse_css_string(css)
    assert ctx.has_prefers_contrast is True


def test_forced_colors_detection():
    css = "@media (forced-colors: active) { .icon { forced-color-adjust: auto; } }"
    ctx = parse_css_string(css)
    assert ctx.has_forced_colors is True


def test_focus_rule_extraction():
    css = """
    .btn:focus { outline: 2px solid #3b82f6; outline-offset: 2px; }
    .btn:focus-visible { box-shadow: 0 0 0 3px rgba(59,130,246,0.5); }
    """
    ctx = parse_css_string(css)
    focus_rules = [r for r in ctx.rules if ":focus" in r.selector]
    assert len(focus_rules) >= 1
    assert "outline" in focus_rules[0].properties


def test_font_size_extraction():
    css = "body { font-size: 16px; line-height: 1.6; }"
    ctx = parse_css_string(css)
    body_rules = [r for r in ctx.rules if r.selector == "body"]
    assert len(body_rules) == 1
    assert body_rules[0].properties["font-size"] == "16px"
    assert body_rules[0].properties["line-height"] == "1.6"


def test_var_reference_resolution():
    css = ":root { --blue: #0000ff; --primary: var(--blue); }"
    ctx = parse_css_string(css)
    primary = next(v for v in ctx.variables if v.name == "--primary")
    assert primary.resolved_hex == "#0000ff"


def test_media_queries_collected():
    css = """
    @media (max-width: 768px) { .nav { display: none; } }
    @media (prefers-reduced-motion: reduce) { * { animation: none; } }
    """
    ctx = parse_css_string(css)
    assert len(ctx.media_queries) == 2
    assert "max-width: 768px" in ctx.media_queries
