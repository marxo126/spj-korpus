"""Tests for helper functions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.helpers import (
    hex_to_rgb, hsl_to_rgb, relative_luminance, contrast_ratio,
    is_large_text, parse_color,
)


def test_hex_to_rgb():
    assert hex_to_rgb("#ffffff") == (255, 255, 255)
    assert hex_to_rgb("#000") == (0, 0, 0)
    assert hex_to_rgb("#DC2626") == (220, 38, 38)


def test_contrast_ratio_black_white():
    ratio = contrast_ratio("#000000", "#ffffff")
    assert ratio == 21.0


def test_contrast_ratio_similar():
    ratio = contrast_ratio("#767676", "#ffffff")
    assert ratio >= 4.5


def test_is_large_text():
    assert is_large_text(18, bold=False) is True
    assert is_large_text(14, bold=True) is True
    assert is_large_text(14, bold=False) is False
    assert is_large_text(12, bold=False) is False


def test_parse_color_hex():
    assert parse_color("#ff0000") == (255, 0, 0)


def test_parse_color_rgb():
    assert parse_color("rgb(255, 0, 0)") == (255, 0, 0)


def test_parse_color_named():
    assert parse_color("white") == (255, 255, 255)
    assert parse_color("black") == (0, 0, 0)


def test_parse_color_hsl():
    r, g, b = parse_color("hsl(0, 100%, 50%)")
    assert r == 255 and g == 0 and b == 0
