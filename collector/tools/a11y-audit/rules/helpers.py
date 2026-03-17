"""Shared helper functions for accessibility rules."""
from __future__ import annotations

import re
from typing import Iterator

from parsers.models import (
    ParseContext, FileContext, ElementNode, CSSContext,
    PHP_EXTENSIONS, JS_EXTENSIONS, ALL_EXTENSIONS,
)

# ── Color math ──────────────────────────────────────────────────────────

_NAMED_COLORS: dict[str, tuple[int, int, int]] = {
    "white": (255, 255, 255), "black": (0, 0, 0),
    "red": (255, 0, 0), "green": (0, 128, 0), "blue": (0, 0, 255),
    "yellow": (255, 255, 0), "orange": (255, 165, 0), "purple": (128, 0, 128),
    "gray": (128, 128, 128), "grey": (128, 128, 128),
    "silver": (192, 192, 192), "maroon": (128, 0, 0),
    "navy": (0, 0, 128), "teal": (0, 128, 128), "aqua": (0, 255, 255),
    "fuchsia": (255, 0, 255), "lime": (0, 255, 0), "olive": (128, 128, 0),
    "transparent": (0, 0, 0),
}

_HEX_RE = re.compile(r"^#([0-9a-fA-F]{3,8})$")
_RGB_RE = re.compile(r"rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)")
_HSL_RE = re.compile(r"hsla?\(\s*([\d.]+)\s*,\s*([\d.]+)%\s*,\s*([\d.]+)%")


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = h[0] * 2 + h[1] * 2 + h[2] * 2
    if len(h) < 6:
        raise ValueError(f"Invalid hex: {hex_color!r}")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def hsl_to_rgb(h: float, s: float, l: float) -> tuple[int, int, int]:
    s /= 100.0
    l /= 100.0
    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = l - c / 2
    if 0 <= h < 60:
        r1, g1, b1 = c, x, 0.0
    elif 60 <= h < 120:
        r1, g1, b1 = x, c, 0.0
    elif 120 <= h < 180:
        r1, g1, b1 = 0.0, c, x
    elif 180 <= h < 240:
        r1, g1, b1 = 0.0, x, c
    elif 240 <= h < 300:
        r1, g1, b1 = x, 0.0, c
    else:
        r1, g1, b1 = c, 0.0, x
    return (round((r1 + m) * 255), round((g1 + m) * 255), round((b1 + m) * 255))


def relative_luminance(r: int, g: int, b: int) -> float:
    def _lin(v: int) -> float:
        rs = v / 255.0
        return rs / 12.92 if rs <= 0.04045 else ((rs + 0.055) / 1.055) ** 2.4
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def contrast_ratio(color1: str, color2: str) -> float:
    r1, g1, b1 = parse_color(color1)
    r2, g2, b2 = parse_color(color2)
    l1 = relative_luminance(r1, g1, b1)
    l2 = relative_luminance(r2, g2, b2)
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return round((lighter + 0.05) / (darker + 0.05), 1)


def parse_color(value: str) -> tuple[int, int, int]:
    value = value.strip().lower()
    if value in _NAMED_COLORS:
        return _NAMED_COLORS[value]
    m = _HEX_RE.match(value)
    if m:
        return hex_to_rgb(value)
    m = _RGB_RE.match(value)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = _HSL_RE.match(value)
    if m:
        return hsl_to_rgb(float(m.group(1)), float(m.group(2)), float(m.group(3)))
    raise ValueError(f"Cannot parse color: {value!r}")


def is_large_text(font_size_px: float, bold: bool = False) -> bool:
    return font_size_px >= 18 or (font_size_px >= 14 and bold)


# ── CSS helpers ─────────────────────────────────────────────────────────

def resolve_css_var(name: str, css_ctx: CSSContext, mode: str = "light") -> str | None:
    for var in css_ctx.variables:
        if var.name == name and var.mode == mode:
            return var.resolved_hex or var.value
    for var in css_ctx.variables:
        if var.name == name:
            return var.resolved_hex or var.value
    return None


# ── Iteration helpers ───────────────────────────────────────────────────

def iter_files(
    ctx: ParseContext,
    extensions: tuple[str, ...] = ALL_EXTENSIONS,
) -> Iterator[tuple[str, FileContext]]:
    for path, fc in ctx.files.items():
        if path.endswith(extensions):
            yield path, fc


def iter_elements(
    ctx: ParseContext,
    extensions: tuple[str, ...] = PHP_EXTENSIONS,
) -> Iterator[tuple[str, FileContext, ElementNode]]:
    for path, fc in iter_files(ctx, extensions):
        for elem in fc.elements:
            yield path, fc, elem


def has_accessible_name(elem: ElementNode) -> bool:
    return bool(
        elem.attributes.get("aria-label")
        or elem.attributes.get("aria-labelledby")
        or elem.attributes.get("title")
    )
