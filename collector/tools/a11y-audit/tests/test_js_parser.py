"""Tests for JS parser."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from parsers.js_parser import parse_js_string


def test_add_event_listener_extraction():
    js = """
    btn.addEventListener('click', handleClick);
    input.addEventListener("keydown", onKey);
    """
    ctx = parse_js_string(js)
    assert len(ctx.event_listeners) == 2
    types = {el.event_type for el in ctx.event_listeners}
    assert "click" in types
    assert "keydown" in types


def test_inline_event_extraction():
    js = """
    element.onclick = function() { doStuff(); };
    card.onmouseover = highlight;
    """
    ctx = parse_js_string(js)
    types = {el.event_type for el in ctx.event_listeners}
    assert "click" in types
    assert "mouseover" in types


def test_set_timeout_extraction():
    js = "setTimeout(hideToast, 3000);"
    ctx = parse_js_string(js)
    assert len(ctx.timeouts) == 1
    assert ctx.timeouts[0].function == "setTimeout"
    assert ctx.timeouts[0].duration_ms == 3000


def test_set_interval_extraction():
    js = "setInterval(checkStatus, 5000);"
    ctx = parse_js_string(js)
    assert len(ctx.timeouts) == 1
    assert ctx.timeouts[0].function == "setInterval"
    assert ctx.timeouts[0].duration_ms == 5000


def test_timeout_with_anonymous_function():
    js = "setTimeout(function() { el.remove(); }, 1500);"
    ctx = parse_js_string(js)
    assert len(ctx.timeouts) == 1
    assert ctx.timeouts[0].duration_ms == 1500


def test_focus_blur_detection_in_raw_content():
    js = """
    el.addEventListener('focus', onFocus);
    el.addEventListener('blur', onBlur);
    """
    ctx = parse_js_string(js)
    types = {el.event_type for el in ctx.event_listeners}
    assert "focus" in types
    assert "blur" in types


def test_aria_manipulation_in_raw_content():
    js = """
    el.setAttribute('aria-expanded', 'true');
    el.setAttribute('aria-hidden', 'false');
    el.removeAttribute('aria-live');
    """
    ctx = parse_js_string(js)
    assert "aria-expanded" in ctx.raw_content
    assert "aria-hidden" in ctx.raw_content
    assert "aria-live" in ctx.raw_content


def test_line_numbers_correct():
    js = "// line 1\n// line 2\nbtn.addEventListener('click', fn);\n// line 4"
    ctx = parse_js_string(js)
    assert len(ctx.event_listeners) == 1
    assert ctx.event_listeners[0].line == 3


def test_multiple_listeners_same_element():
    js = """
    modal.addEventListener('keydown', trapFocus);
    modal.addEventListener('click', handleClick);
    modal.addEventListener('transitionend', cleanup);
    """
    ctx = parse_js_string(js)
    assert len(ctx.event_listeners) == 3
