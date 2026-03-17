"""Tests for PHP/HTML parser."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from parsers.php_parser import parse_html_string, parse_php_file, _strip_php_blocks, _extract_echo_html
from parsers.models import PhpBlock


def test_simple_html_extracts_tags():
    html = '<div><button type="submit">OK</button></div>'
    ctx = parse_html_string(html)
    tags = [e.tag for e in ctx.elements]
    assert "button" in tags
    assert "div" in tags


def test_php_block_stripping_preserves_lines():
    content = (
        "<html>\n"
        "<?php echo 'hello'; ?>\n"
        "<div>after</div>\n"
    )
    stripped, blocks = _strip_php_blocks(content)
    assert len(blocks) == 1
    assert blocks[0].start_line == 2
    # The stripped content should have same number of lines
    assert stripped.count("\n") == content.count("\n")
    assert "<?php" not in stripped


def test_attribute_extraction():
    html = (
        '<input type="email" id="user-email" '
        'aria-label="Your email" required>'
    )
    ctx = parse_html_string(html)
    inputs = [e for e in ctx.elements if e.tag == "input"]
    assert len(inputs) >= 1
    inp = inputs[0]
    assert inp.attributes.get("type") == "email"
    assert inp.attributes.get("id") == "user-email"
    assert inp.attributes.get("aria-label") == "Your email"
    # 'required' is a boolean attribute in BS4
    assert "required" in inp.attributes


def test_suppression_comment_extraction():
    html = '<!-- a11y-suppress contrast:text-ratio -- low priority for now -->\n<p>hi</p>'
    ctx = parse_html_string(html)
    assert len(ctx.suppressions) == 1
    s = ctx.suppressions[0]
    assert s["rule"] == "contrast"
    assert s["check"] == "text-ratio"
    assert "low priority" in s["reason"]


def test_suppression_with_em_dash():
    html = '<!-- a11y-suppress forms:label \u2014 decorative input -->\n<input>'
    ctx = parse_html_string(html)
    assert len(ctx.suppressions) == 1
    assert ctx.suppressions[0]["reason"] == "decorative input"


def test_echo_html_extraction():
    blocks = [
        PhpBlock(start_line=1, end_line=1, content="""echo '<span class="badge">new</span>';"""),
        PhpBlock(start_line=2, end_line=2, content="$x = 42;"),
    ]
    html = _extract_echo_html(blocks)
    assert "<span" in html
    assert "badge" in html


def test_php_file_parsing(tmp_path):
    php_file = tmp_path / "test.php"
    php_file.write_text(
        '<?php $title = "Test"; ?>\n'
        "<html>\n"
        "<body>\n"
        '<div id="main">Hello</div>\n'
        "<?php echo '<a href=\"/\">Home</a>'; ?>\n"
        "</body>\n"
        "</html>\n"
    )
    ctx = parse_php_file(php_file)
    assert len(ctx.php_blocks) == 2
    tags = [e.tag for e in ctx.elements]
    assert "div" in tags
    # Echo-extracted anchor should also appear
    assert "a" in tags


def test_multiline_php_block():
    content = (
        "<p>before</p>\n"
        "<?php\n"
        "  if (true) {\n"
        "    echo 'hi';\n"
        "  }\n"
        "?>\n"
        "<p>after</p>\n"
    )
    stripped, blocks = _strip_php_blocks(content)
    assert len(blocks) == 1
    assert blocks[0].start_line == 2
    assert blocks[0].end_line == 6
    # Line count must be preserved
    assert stripped.count("\n") == content.count("\n")


def test_nested_elements_parent_tag():
    html = '<form><label><input type="text"></label></form>'
    ctx = parse_html_string(html)
    inputs = [e for e in ctx.elements if e.tag == "input"]
    assert len(inputs) >= 1
    assert inputs[0].parent_tag == "label"
