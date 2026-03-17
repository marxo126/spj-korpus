# Collector Accessibility Audit Tool — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a WCAG 2.2 AA + EN 301 549 accessibility audit tool for the SPJ Sign Language Collector app (PHP/HTML/JS/CSS).

**Architecture:** Static analysis tool that parses PHP/HTML files with BeautifulSoup, CSS with regex, and JS with regex. 16 rule modules check WCAG criteria. Optional Playwright runtime bridge. Three output formats: terminal, JSON, HTML dashboard.

**Tech Stack:** Python 3.13, BeautifulSoup4+lxml, PyYAML, Jinja2, colorama. Own venv at `collector/tools/a11y-audit/.venv/`.

**Spec:** `docs/superpowers/specs/2026-03-17-collector-a11y-audit-design.md`

**Reference Implementation:** `~/Coding-space/fluentiagrant-app/tools/a11y-audit/` — adapt patterns for PHP instead of TSX.

---

## Chunk 1: Foundation

### Task 1: Project scaffolding & dependencies

**Files:**
- Create: `collector/tools/a11y-audit/requirements.txt`
- Create: `collector/tools/a11y-audit/rules/__init__.py`
- Create: `collector/tools/a11y-audit/parsers/__init__.py`
- Create: `collector/tools/a11y-audit/reporters/__init__.py`
- Create: `collector/tools/a11y-audit/reporters/templates/.gitkeep`
- Create: `collector/tools/a11y-audit/reports/.gitignore`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p collector/tools/a11y-audit/{rules,parsers,reporters/templates,reports,tests}
```

- [ ] **Step 2: Write requirements.txt**

```
pyyaml>=6.0
jinja2>=3.1
colorama>=0.4
beautifulsoup4>=4.12
lxml>=5.0
```

- [ ] **Step 3: Create virtualenv and install deps**

```bash
cd collector/tools/a11y-audit
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

- [ ] **Step 4: Create __init__.py files and .gitignore**

Empty `__init__.py` for `rules/`, `parsers/`, `reporters/`, `tests/`.

`reports/.gitignore`:
```
*
!.gitignore
```

- [ ] **Step 5: Commit**

```bash
git add collector/tools/a11y-audit/
git commit -m "chore: scaffold collector a11y audit tool"
```

---

### Task 2: Base classes (rules/base.py)

**Files:**
- Create: `collector/tools/a11y-audit/rules/base.py`
- Create: `collector/tools/a11y-audit/tests/test_base.py`

- [ ] **Step 1: Write test**

```python
"""Tests for base rule classes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.base import Severity, Detection, Finding, BaseRule


def test_severity_values():
    assert Severity.CRITICAL.value == "critical"
    assert Severity.SERIOUS.value == "serious"
    assert Severity.MODERATE.value == "moderate"
    assert Severity.MINOR.value == "minor"


def test_finding_creation():
    f = Finding(
        rule_id="test",
        check_id="test-check",
        severity=Severity.CRITICAL,
        wcag="1.1.1",
        wcag_name="Non-text Content",
        message="Missing alt",
        file="index.php",
        line=10,
    )
    assert f.rule_id == "test"
    assert f.detection == Detection.STATIC
    assert f.suppressed is False


def test_base_rule_finding_factory():
    class TestRule(BaseRule):
        id = "test"
        name = "Test Rule"
        wcag_criteria = ("1.1.1",)
        standards = ("WCAG 2.2 AA",)

        def check(self, ctx, config):
            return [self._finding(
                check_id="demo",
                severity=Severity.MINOR,
                wcag="1.1.1",
                wcag_name="Test",
                message="Demo",
                file="x.php",
                line=1,
            )]

    rule = TestRule()
    findings = rule.check(None, {})
    assert len(findings) == 1
    assert findings[0].rule_id == "test"
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
cd collector/tools/a11y-audit && .venv/bin/python -m pytest tests/test_base.py -v
```

- [ ] **Step 3: Implement base.py**

Adapt from fluentiagrant `rules/base.py`. Key changes: use `tuple[str, ...]` for `impact` field instead of `list[str]`.

```python
"""Base classes for accessibility audit rules."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from parsers.models import ParseContext


class Severity(Enum):
    CRITICAL = "critical"
    SERIOUS = "serious"
    MODERATE = "moderate"
    MINOR = "minor"


class Detection(Enum):
    STATIC = "static"
    RUNTIME = "runtime"


@dataclass(frozen=True)
class Finding:
    rule_id: str
    check_id: str
    severity: Severity
    wcag: str
    wcag_name: str
    message: str
    file: str
    line: int
    element: str = ""
    fix: str = ""
    impact: tuple[str, ...] = ()
    detection: Detection = Detection.STATIC
    suppressed: bool = False
    suppression_reason: str = ""


class BaseRule(ABC):
    id: str = ""
    name: str = ""
    wcag_criteria: tuple[str, ...] = ()
    standards: tuple[str, ...] = ()

    @abstractmethod
    def check(self, ctx: "ParseContext", config: dict) -> list[Finding]:
        ...

    def _finding(
        self,
        check_id: str,
        severity: Severity,
        wcag: str,
        wcag_name: str,
        message: str,
        file: str,
        line: int,
        element: str = "",
        fix: str = "",
        impact: tuple[str, ...] = (),
    ) -> Finding:
        return Finding(
            rule_id=self.id,
            check_id=check_id,
            severity=severity,
            wcag=wcag,
            wcag_name=wcag_name,
            message=message,
            file=file,
            line=line,
            element=element,
            fix=fix,
            impact=impact,
            detection=Detection.STATIC,
        )
```

- [ ] **Step 4: Run test — expect PASS**

```bash
cd collector/tools/a11y-audit && .venv/bin/python -m pytest tests/test_base.py -v
```

- [ ] **Step 5: Commit**

```bash
git add collector/tools/a11y-audit/rules/base.py collector/tools/a11y-audit/tests/
git commit -m "feat(a11y): add base rule classes — Severity, Finding, BaseRule"
```

---

### Task 3: Data models (parsers/models.py)

**Files:**
- Create: `collector/tools/a11y-audit/parsers/models.py`

- [ ] **Step 1: Write models.py**

Adapted from fluentiagrant — remove TSX-specific fields (tailwind_classes, has_spread_props, condition), add PHP-specific fields.

```python
"""Shared data models for parse context."""
from __future__ import annotations

from dataclasses import dataclass, field

PHP_EXTENSIONS = (".php",)
JS_EXTENSIONS = (".js",)
CSS_EXTENSIONS = (".css",)
ALL_EXTENSIONS = (".php", ".js", ".css")


@dataclass
class ElementNode:
    """An HTML element extracted from a PHP file."""
    tag: str
    attributes: dict[str, str | bool | None]
    line: int
    children: list["ElementNode"] = field(default_factory=list)
    parent_tag: str | None = None
    text_content: str = ""


@dataclass
class PhpBlock:
    """A stripped PHP block with its original line range."""
    start_line: int
    end_line: int
    content: str


@dataclass
class TimeoutCall:
    """A setTimeout/setInterval call found in JS."""
    function: str
    duration_ms: int | None
    line: int
    code: str


@dataclass
class EventListener:
    """An addEventListener call found in JS."""
    event_type: str
    line: int
    code: str
    file: str


@dataclass
class FileContext:
    """Parsed content of a single file."""
    path: str
    elements: list[ElementNode] = field(default_factory=list)
    php_blocks: list[PhpBlock] = field(default_factory=list)
    timeouts: list[TimeoutCall] = field(default_factory=list)
    event_listeners: list[EventListener] = field(default_factory=list)
    raw_content: str = ""
    lines: list[str] = field(default_factory=list)
    suppressions: list[dict] = field(default_factory=list)

    @property
    def content(self) -> str:
        return self.raw_content or "\n".join(self.lines)

    def get_line(self, line_num: int) -> str:
        if 0 < line_num <= len(self.lines):
            return self.lines[line_num - 1]
        return ""


@dataclass
class CSSVariable:
    name: str
    value: str
    resolved_hex: str | None = None
    mode: str = "light"  # "light" or "dark"


@dataclass
class KeyframeAnimation:
    name: str
    duration_ms: float | None
    iteration_count: str
    line: int = 0


@dataclass
class CSSRule:
    """A CSS rule with selector and properties."""
    selector: str
    properties: dict[str, str]
    line: int


@dataclass
class CSSContext:
    variables: list[CSSVariable] = field(default_factory=list)
    keyframes: list[KeyframeAnimation] = field(default_factory=list)
    rules: list[CSSRule] = field(default_factory=list)
    media_queries: list[str] = field(default_factory=list)
    has_prefers_reduced_motion: bool = False
    has_prefers_contrast: bool = False
    has_forced_colors: bool = False
    raw_content: str = ""


@dataclass
class ParseContext:
    """Aggregate parse context for the entire project."""
    files: dict[str, FileContext] = field(default_factory=dict)
    css: CSSContext = field(default_factory=CSSContext)
    project_root: str = ""
```

- [ ] **Step 2: Commit**

```bash
git add collector/tools/a11y-audit/parsers/models.py
git commit -m "feat(a11y): add parser data models — FileContext, CSSContext, ParseContext"
```

---

### Task 4: Helpers (rules/helpers.py)

**Files:**
- Create: `collector/tools/a11y-audit/rules/helpers.py`
- Create: `collector/tools/a11y-audit/tests/test_helpers.py`

- [ ] **Step 1: Write test**

```python
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
    assert ratio >= 4.5  # WCAG AA threshold


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
```

- [ ] **Step 2: Run test — expect FAIL**

- [ ] **Step 3: Implement helpers.py**

```python
"""Shared helper functions for accessibility rules."""
from __future__ import annotations

import math
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


def resolve_css_var(name: str, css_ctx: CSSContext, mode: str = "light") -> str | None:
    """Resolve a CSS custom property name to its value."""
    for var in css_ctx.variables:
        if var.name == name and var.mode == mode:
            return var.resolved_hex or var.value
    # Try any mode as fallback
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
```

- [ ] **Step 4: Run test — expect PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(a11y): add helper functions — color math, PHP stripping, iterators"
```

---

### Task 5: Configuration (config.yaml)

**Files:**
- Create: `collector/tools/a11y-audit/config.yaml`

- [ ] **Step 1: Write config.yaml**

Per spec section 8. All paths relative to `root` (which resolves to `collector/`).

```yaml
project:
  name: "SPJ Collector (zber.spj.sk)"
  root: "../.."
  directories:
    pages: "."
    admin: "admin"
    includes: "includes"
    api: "api"
  css_file: "css/style.css"
  js_dir: "js"

standards:
  wcag: "2.2"
  conformance: "AA"
  en_301_549: true

severity:
  critical: error
  serious: error
  moderate: error
  minor: error

thresholds:
  contrast_text: 4.5
  contrast_large: 3.0
  contrast_ui: 3.0
  target_size_aa: 24
  target_size_enhanced: 44
  min_font_size: 14
  min_line_height: 1.5
  max_line_length: 80
  animation_flash_hz: 3
  toast_min_ms: 5000

modules:
  contrast: true
  color: true
  typography: true
  layout: true
  structure: true
  aria: true
  focus: true
  keyboard: true
  interactive: true
  forms: true
  media: true
  language: true
  motion: true
  cognitive: true
  compliance: true
  collector: true

include:
  - "*.php"
  - "js/*.js"
  - "css/*.css"
  - "admin/**/*.php"
  - "admin/api/**/*.php"
  - "api/**/*.php"
  - "includes/**/*.php"

exclude:
  - "uploads/**"
  - "tools/**"
  - ".git/**"
  - "sql/**"
  - "img/**"

suppressions: []

reports:
  output_dir: "tools/a11y-audit/reports"
  formats: [terminal, json, html]
  keep_history: 10
```

- [ ] **Step 2: Commit**

```bash
git commit -m "feat(a11y): add config.yaml with WCAG 2.2 AA thresholds"
```

---

## Chunk 2: Parsers

### Task 6: PHP/HTML parser (parsers/php_parser.py)

**Files:**
- Create: `collector/tools/a11y-audit/parsers/php_parser.py`
- Create: `collector/tools/a11y-audit/tests/test_php_parser.py`

- [ ] **Step 1: Write test**

```python
"""Tests for PHP/HTML parser."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from parsers.php_parser import parse_php_file, parse_html_string
from parsers.models import FileContext


def test_parse_simple_html():
    html = '<html lang="sk"><body><h1>Hello</h1></body></html>'
    fc = parse_html_string(html, "test.php")
    tags = [e.tag for e in fc.elements]
    assert "html" in tags
    assert "h1" in tags


def test_parse_with_php_blocks():
    content = '''<html lang="sk">
<?php
require_once 'includes/config.php';
$user = get_user();
?>
<body>
<h1><?= htmlspecialchars($title) ?></h1>
<img src="logo.png">
</body>
</html>'''
    fc = parse_html_string(content, "test.php")
    tags = [e.tag for e in fc.elements]
    assert "html" in tags
    assert "h1" in tags
    assert "img" in tags
    # Line numbers preserved
    img_elem = [e for e in fc.elements if e.tag == "img"][0]
    assert img_elem.line == 8


def test_attributes_extracted():
    html = '<input type="email" id="user_email" required aria-label="Email">'
    fc = parse_html_string(html, "test.php")
    inp = [e for e in fc.elements if e.tag == "input"][0]
    assert inp.attributes.get("type") == "email"
    assert inp.attributes.get("aria-label") == "Email"
    assert inp.attributes.get("required") is not None


def test_suppressions_extracted():
    html = '''<!-- a11y-suppress contrast:text-contrast -- decorative -->
<span class="watermark">Draft</span>'''
    fc = parse_html_string(html, "test.php")
    assert len(fc.suppressions) == 1
    assert fc.suppressions[0]["rule"] == "contrast"
    assert fc.suppressions[0]["check"] == "text-contrast"


def test_echo_html_extraction():
    content = '''<?php
echo '<div class="alert" role="alert">';
echo '<span>' . htmlspecialchars($msg) . '</span>';
echo '</div>';
?>'''
    fc = parse_html_string(content, "test.php")
    # Should extract elements from echo strings
    tags = [e.tag for e in fc.elements]
    assert "div" in tags
```

- [ ] **Step 2: Run test — expect FAIL**

- [ ] **Step 3: Implement php_parser.py**

```python
"""PHP/HTML parser — extracts HTML elements from PHP files using BeautifulSoup."""
from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from parsers.models import FileContext, ElementNode, PhpBlock

# Regex to strip PHP blocks
_PHP_BLOCK_RE = re.compile(r"<\?(?:php|=)(.*?)\?>", re.DOTALL)

# Regex to extract HTML from echo statements
_ECHO_HTML_RE = re.compile(
    r"""echo\s+['"](<[^'"]*>)['"]""",
    re.IGNORECASE,
)

# Regex for inline suppression comments
_SUPPRESS_RE = re.compile(
    r"<!--\s*a11y-suppress\s+([\w-]+):([\w-]+)\s*(?:--|—)\s*(.*?)\s*-->",
    re.IGNORECASE,
)


def _strip_php(content: str) -> tuple[str, list[PhpBlock]]:
    """Replace PHP blocks with whitespace, preserving line numbers."""
    blocks: list[PhpBlock] = []

    def _replace(m: re.Match) -> str:
        start_line = content[: m.start()].count("\n") + 1
        end_line = content[: m.end()].count("\n") + 1
        blocks.append(PhpBlock(
            start_line=start_line,
            end_line=end_line,
            content=m.group(0),
        ))
        # Preserve line count
        return "\n" * m.group(0).count("\n")

    stripped = _PHP_BLOCK_RE.sub(_replace, content)
    return stripped, blocks


def _extract_echo_html(php_blocks: list[PhpBlock]) -> list[tuple[str, int]]:
    """Extract HTML fragments from echo statements in PHP blocks."""
    fragments: list[tuple[str, int]] = []
    for block in php_blocks:
        for m in _ECHO_HTML_RE.finditer(block.content):
            html_frag = m.group(1)
            # Approximate line number
            line_in_block = block.content[: m.start()].count("\n")
            fragments.append((html_frag, block.start_line + line_in_block))
    return fragments


def _soup_to_elements(
    soup: BeautifulSoup,
    raw_content: str,
    base_line: int = 0,
) -> list[ElementNode]:
    """Convert BeautifulSoup tags to ElementNode list."""
    elements: list[ElementNode] = []

    for tag in soup.find_all(True):  # All tags
        if not isinstance(tag, Tag):
            continue

        attrs: dict[str, str | bool | None] = {}
        for key, val in tag.attrs.items():
            if isinstance(val, list):
                attrs[key] = " ".join(val)
            elif val is True or val == "":
                attrs[key] = True
            else:
                attrs[key] = str(val) if val is not None else None

        # Estimate line number from sourceline (BS4 provides this with lxml)
        line = getattr(tag, "sourceline", 0) or 0
        line += base_line

        parent_tag = tag.parent.name if tag.parent and isinstance(tag.parent, Tag) else None

        text = tag.string or ""
        if not text:
            # Direct text children only
            text = "".join(
                child.string for child in tag.children
                if isinstance(child, str) or (hasattr(child, "string") and child.name is None)
            ).strip()

        elements.append(ElementNode(
            tag=tag.name,
            attributes=attrs,
            line=line,
            parent_tag=parent_tag,
            text_content=text[:200],  # Truncate long text
        ))

    return elements


def _extract_suppressions(content: str) -> list[dict]:
    """Extract a11y-suppress comments."""
    suppressions = []
    for m in _SUPPRESS_RE.finditer(content):
        suppressions.append({
            "rule": m.group(1),
            "check": m.group(2),
            "reason": m.group(3).strip(),
            "line": content[: m.start()].count("\n") + 1,
        })
    return suppressions


def parse_html_string(content: str, path: str) -> FileContext:
    """Parse a string containing PHP/HTML into a FileContext."""
    raw_content = content
    lines = content.split("\n")

    # Extract suppressions before stripping
    suppressions = _extract_suppressions(content)

    # Strip PHP blocks
    stripped_html, php_blocks = _strip_php(content)

    # Parse main HTML
    soup = BeautifulSoup(stripped_html, "lxml")
    elements = _soup_to_elements(soup, stripped_html)

    # Extract HTML from echo statements
    echo_fragments = _extract_echo_html(php_blocks)
    for frag_html, frag_line in echo_fragments:
        try:
            frag_soup = BeautifulSoup(frag_html, "lxml")
            frag_elements = _soup_to_elements(frag_soup, frag_html, base_line=frag_line)
            elements.extend(frag_elements)
        except Exception:
            pass  # Skip malformed echo HTML

    return FileContext(
        path=path,
        elements=elements,
        php_blocks=php_blocks,
        raw_content=raw_content,
        lines=lines,
        suppressions=suppressions,
    )


def parse_php_file(file_path: str | Path) -> FileContext:
    """Parse a PHP file from disk."""
    path = Path(file_path)
    content = path.read_text(encoding="utf-8", errors="replace")
    rel_path = path.name  # Will be made relative by caller
    return parse_html_string(content, rel_path)
```

- [ ] **Step 4: Run test — expect PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(a11y): add PHP/HTML parser — BeautifulSoup + PHP block stripping"
```

---

### Task 7: CSS parser (parsers/css_parser.py)

**Files:**
- Create: `collector/tools/a11y-audit/parsers/css_parser.py`
- Create: `collector/tools/a11y-audit/tests/test_css_parser.py`

- [ ] **Step 1: Write test**

```python
"""Tests for CSS parser."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from parsers.css_parser import parse_css_string


def test_parse_variables():
    css = """:root {
    --red: #DC2626;
    --green: #16A34A;
    --bg: #FFFFFF;
}"""
    ctx = parse_css_string(css)
    names = [v.name for v in ctx.variables]
    assert "--red" in names
    assert "--green" in names


def test_dark_mode_variables():
    css = """html.dark {
    --bg: #0F172A;
    --text: #E2E8F0;
}"""
    ctx = parse_css_string(css)
    dark_vars = [v for v in ctx.variables if v.mode == "dark"]
    assert len(dark_vars) >= 2


def test_keyframes():
    css = """@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0; }
}"""
    ctx = parse_css_string(css)
    assert len(ctx.keyframes) == 1
    assert ctx.keyframes[0].name == "blink"


def test_prefers_reduced_motion():
    css = """@media (prefers-reduced-motion: reduce) {
    * { animation: none !important; }
}"""
    ctx = parse_css_string(css)
    assert ctx.has_prefers_reduced_motion is True


def test_focus_rules():
    css = """button:focus {
    outline: 2px solid var(--blue);
    outline-offset: 2px;
}"""
    ctx = parse_css_string(css)
    focus_rules = [r for r in ctx.rules if ":focus" in r.selector]
    assert len(focus_rules) >= 1


def test_font_size_extraction():
    css = """body { font-size: 18px; line-height: 1.5; }
.small { font-size: 12px; }"""
    ctx = parse_css_string(css)
    font_rules = [r for r in ctx.rules if "font-size" in r.properties]
    assert len(font_rules) >= 2
```

- [ ] **Step 2: Run test — expect FAIL**

- [ ] **Step 3: Implement css_parser.py**

```python
"""CSS parser — extracts variables, keyframes, rules, media queries."""
from __future__ import annotations

import re
from pathlib import Path

from parsers.models import CSSContext, CSSVariable, KeyframeAnimation, CSSRule

# ── Regex patterns ──────────────────────────────────────────────────────

_VAR_RE = re.compile(r"(--[\w-]+)\s*:\s*([^;]+);")
_KEYFRAME_RE = re.compile(r"@keyframes\s+([\w-]+)\s*\{", re.MULTILINE)
_MEDIA_RE = re.compile(r"@media\s*\(([^)]+)\)\s*\{", re.MULTILINE)
_RULE_RE = re.compile(
    r"(?:^|\})\s*([\w\s.#:>\[\]=~*,()+-]+?)\s*\{([^}]*)\}",
    re.MULTILINE,
)
_PROP_RE = re.compile(r"([\w-]+)\s*:\s*([^;]+);")
_HEX_RE = re.compile(r"#[0-9a-fA-F]{3,8}")

# Dark mode selectors
_DARK_SELECTORS = ("html.dark", ".dark", "@media (prefers-color-scheme: dark)")


def _resolve_hex(value: str) -> str | None:
    """Extract hex color from a CSS value if present."""
    m = _HEX_RE.search(value.strip())
    return m.group(0) if m else None


def _detect_mode(context_before: str) -> str:
    """Detect if we're inside a dark mode block."""
    # Count open/close braces to see if we're nested in a dark block
    for sel in _DARK_SELECTORS:
        if sel in context_before:
            # Check if the block is still open
            after_sel = context_before[context_before.rfind(sel):]
            opens = after_sel.count("{")
            closes = after_sel.count("}")
            if opens > closes:
                return "dark"
    return "light"


def parse_css_string(content: str) -> CSSContext:
    """Parse CSS content into a CSSContext."""
    variables: list[CSSVariable] = []
    keyframes: list[KeyframeAnimation] = []
    rules: list[CSSRule] = []
    media_queries: list[str] = []
    has_prefers_reduced_motion = False
    has_prefers_contrast = False
    has_forced_colors = False

    # Extract media queries
    for m in _MEDIA_RE.finditer(content):
        query = m.group(1).strip()
        media_queries.append(query)
        if "prefers-reduced-motion" in query:
            has_prefers_reduced_motion = True
        if "prefers-contrast" in query:
            has_prefers_contrast = True
        if "forced-colors" in query:
            has_forced_colors = True

    # Extract CSS variables
    for m in _VAR_RE.finditer(content):
        name = m.group(1)
        value = m.group(2).strip()
        mode = _detect_mode(content[: m.start()])
        variables.append(CSSVariable(
            name=name,
            value=value,
            resolved_hex=_resolve_hex(value),
            mode=mode,
        ))

    # Extract keyframes
    for m in _KEYFRAME_RE.finditer(content):
        name = m.group(1)
        line = content[: m.start()].count("\n") + 1
        # Try to find duration from animation properties referencing this keyframe
        duration_match = re.search(
            rf"animation[^;]*{re.escape(name)}[^;]*?([\d.]+)s",
            content,
        )
        duration_ms = float(duration_match.group(1)) * 1000 if duration_match else None
        keyframes.append(KeyframeAnimation(
            name=name,
            duration_ms=duration_ms,
            iteration_count="1",
            line=line,
        ))

    # Extract CSS rules (selector + properties)
    # Use a simpler approach: split by top-level braces
    _extract_rules(content, rules)

    return CSSContext(
        variables=variables,
        keyframes=keyframes,
        rules=rules,
        media_queries=media_queries,
        has_prefers_reduced_motion=has_prefers_reduced_motion,
        has_prefers_contrast=has_prefers_contrast,
        has_forced_colors=has_forced_colors,
        raw_content=content,
    )


def _extract_rules(content: str, rules: list[CSSRule]) -> None:
    """Extract CSS rules with selectors and properties."""
    # Remove @keyframes blocks first (they contain {} that confuse rule parsing)
    clean = re.sub(r"@keyframes\s+[\w-]+\s*\{[^}]*(?:\{[^}]*\}[^}]*)*\}", "", content)
    # Remove @media wrappers but keep inner rules
    clean = re.sub(r"@media\s*\([^)]*\)\s*\{", "", clean)

    for m in _RULE_RE.finditer(clean):
        selector = m.group(1).strip()
        body = m.group(2)
        if not selector or selector.startswith("@"):
            continue

        props: dict[str, str] = {}
        for pm in _PROP_RE.finditer(body):
            props[pm.group(1)] = pm.group(2).strip()

        if props:
            line = content.find(selector)
            line_num = content[:line].count("\n") + 1 if line >= 0 else 0
            rules.append(CSSRule(selector=selector, properties=props, line=line_num))


def parse_css_file(file_path: str | Path) -> CSSContext:
    """Parse a CSS file from disk."""
    path = Path(file_path)
    content = path.read_text(encoding="utf-8", errors="replace")
    return parse_css_string(content)
```

- [ ] **Step 4: Run test — expect PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(a11y): add CSS parser — variables, keyframes, rules, media queries"
```

---

### Task 8: JS parser (parsers/js_parser.py)

**Files:**
- Create: `collector/tools/a11y-audit/parsers/js_parser.py`
- Create: `collector/tools/a11y-audit/tests/test_js_parser.py`

- [ ] **Step 1: Write test**

```python
"""Tests for JS parser."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from parsers.js_parser import parse_js_string


def test_event_listeners():
    js = """
button.addEventListener('click', handleRecord);
video.addEventListener('mouseover', showControls);
"""
    fc = parse_js_string(js, "recorder.js")
    assert len(fc.event_listeners) == 2
    types = [e.event_type for e in fc.event_listeners]
    assert "click" in types
    assert "mouseover" in types


def test_timeouts():
    js = """
setTimeout(() => { showToast('Done'); }, 3000);
setInterval(updateTimer, 1000);
"""
    fc = parse_js_string(js, "recorder.js")
    assert len(fc.timeouts) == 2
    durations = [t.duration_ms for t in fc.timeouts]
    assert 3000 in durations
    assert 1000 in durations


def test_focus_calls():
    js = """
document.getElementById('input').focus();
modal.querySelector('.close-btn').blur();
"""
    fc = parse_js_string(js, "test.js")
    # Focus/blur tracked in raw content for rule analysis
    assert ".focus()" in fc.raw_content
    assert ".blur()" in fc.raw_content


def test_aria_manipulation():
    js = """
el.setAttribute('aria-hidden', 'true');
el.setAttribute('role', 'dialog');
el.removeAttribute('aria-label');
"""
    fc = parse_js_string(js, "test.js")
    assert "aria-hidden" in fc.raw_content
```

- [ ] **Step 2: Run test — expect FAIL**

- [ ] **Step 3: Implement js_parser.py**

```python
"""JS parser — extracts event listeners, timers, ARIA manipulation."""
from __future__ import annotations

import re
from pathlib import Path

from parsers.models import FileContext, TimeoutCall, EventListener

_EVENT_RE = re.compile(
    r"""\.addEventListener\(\s*['"](\w+)['"]""",
)
_ONCLICK_RE = re.compile(
    r"""\.on(\w+)\s*=\s*""",
)
_TIMEOUT_RE = re.compile(
    r"""(setTimeout|setInterval)\s*\(\s*(?:.*?),\s*(\d+)\s*\)""",
    re.DOTALL,
)


def parse_js_string(content: str, path: str) -> FileContext:
    """Parse JavaScript content into a FileContext."""
    lines = content.split("\n")
    event_listeners: list[EventListener] = []
    timeouts: list[TimeoutCall] = []

    # Extract addEventListener calls
    for m in _EVENT_RE.finditer(content):
        line = content[: m.start()].count("\n") + 1
        event_listeners.append(EventListener(
            event_type=m.group(1),
            line=line,
            code=lines[line - 1].strip() if line <= len(lines) else "",
            file=path,
        ))

    # Extract inline event handlers (onclick=, onmouseover=, etc.)
    for m in _ONCLICK_RE.finditer(content):
        event_type = m.group(1).lower()
        line = content[: m.start()].count("\n") + 1
        event_listeners.append(EventListener(
            event_type=event_type,
            line=line,
            code=lines[line - 1].strip() if line <= len(lines) else "",
            file=path,
        ))

    # Extract setTimeout/setInterval
    for m in _TIMEOUT_RE.finditer(content):
        line = content[: m.start()].count("\n") + 1
        timeouts.append(TimeoutCall(
            function=m.group(1),
            duration_ms=int(m.group(2)),
            line=line,
            code=lines[line - 1].strip() if line <= len(lines) else "",
        ))

    return FileContext(
        path=path,
        event_listeners=event_listeners,
        timeouts=timeouts,
        raw_content=content,
        lines=lines,
    )


def parse_js_file(file_path: str | Path) -> FileContext:
    """Parse a JS file from disk."""
    path = Path(file_path)
    content = path.read_text(encoding="utf-8", errors="replace")
    return parse_js_string(content, path.name)
```

- [ ] **Step 4: Run test — expect PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(a11y): add JS parser — event listeners, timers, ARIA manipulation"
```

---

## Chunk 3: Rules Group 1 (contrast, color, typography, layout, structure, aria)

### Task 9: Contrast rule (rules/contrast.py)

**Files:**
- Create: `collector/tools/a11y-audit/rules/contrast.py`

- [ ] **Step 1: Implement contrast.py**

WCAG 1.4.3 (text contrast) and 1.4.11 (non-text contrast). Reads CSS variables and checks foreground/background pairs in both light and dark modes. Uses `helpers.contrast_ratio()`.

Key checks:
- `text-contrast`: Pairs CSS `color` and `background-color` values, checks ≥ 4.5:1 (normal) or ≥ 3:1 (large text)
- `ui-contrast`: Checks `border-color` and `outline-color` vs backgrounds ≥ 3:1
- `dark-mode-contrast`: Same checks for dark mode variables
- `placeholder-contrast`: Checks `::placeholder` color contrast

- [ ] **Step 2: Commit**

---

### Task 10: Color rule (rules/color.py)

**Files:**
- Create: `collector/tools/a11y-audit/rules/color.py`

- [ ] **Step 1: Implement color.py**

WCAG 1.4.1 (color not sole indicator). Scans HTML for status elements using color classes without accompanying icon/text indicators.

Key checks:
- `color-only-status`: Flags elements with color-based CSS classes (e.g. `.status-ok`, `.text-red`) that lack icon/text siblings
- `link-distinction`: Checks `<a>` elements have underline or non-color distinguisher
- `error-color-only`: Error messages using red without icon

- [ ] **Step 2: Commit**

---

### Task 11: Typography rule (rules/typography.py)

**Files:**
- Create: `collector/tools/a11y-audit/rules/typography.py`

- [ ] **Step 1: Implement typography.py**

WCAG 1.4.4, 1.4.8, 1.4.12. Reads CSS rules for font-size, line-height, letter-spacing.

Key checks:
- `min-font-size`: No font-size below config `min_font_size` (14px)
- `relative-units`: Prefer rem/em over px for font-size
- `line-height`: line-height ≥ config `min_line_height` (1.5) for body text
- `letter-spacing`: No negative letter-spacing
- `text-justify`: No `text-align: justify`
- `line-length`: Max width ~80ch

- [ ] **Step 2: Commit**

---

### Task 12: Layout rule (rules/layout.py)

**Files:**
- Create: `collector/tools/a11y-audit/rules/layout.py`

- [ ] **Step 1: Implement layout.py**

WCAG 1.4.10, 2.5.8. Checks viewport meta, target sizes, spacing.

Key checks:
- `viewport-meta`: `<meta name="viewport">` allows zoom (`user-scalable=yes`, `maximum-scale` ≥ 2)
- `target-size-aa`: CSS min-height/min-width on interactive elements ≥ 24px
- `target-size-enhanced`: ≥ 44px (minor)
- `spacing`: Adequate spacing between interactive elements
- `reflow-320`: No `overflow-x: hidden` on body/html or fixed widths > 320px

- [ ] **Step 2: Commit**

---

### Task 13: Structure rule (rules/structure.py)

**Files:**
- Create: `collector/tools/a11y-audit/rules/structure.py`

- [ ] **Step 1: Implement structure.py**

WCAG 1.3.1, 1.3.2, 2.4.1-6. Checks HTML landmarks, heading hierarchy, skip link, page title.

Key checks:
- `landmarks`: `<main>`, `<nav>`, `<footer>` present
- `heading-hierarchy`: No skipped levels (h1→h3 without h2)
- `single-h1`: Exactly one `<h1>` per page
- `skip-link`: First `<a>` has `#main-content` or similar href
- `page-title`: `<title>` element present and non-empty
- `nav-aria-label`: Multiple `<nav>` elements have unique `aria-label`
- `list-structure`: Lists use semantic `<ul>`/`<ol>`

- [ ] **Step 2: Commit**

---

### Task 14: ARIA rule (rules/aria.py)

**Files:**
- Create: `collector/tools/a11y-audit/rules/aria.py`

- [ ] **Step 1: Implement aria.py**

WCAG 4.1.2. Validates ARIA attributes and roles.

Valid roles list, required attributes per role, redundant role/element combos.

Key checks:
- `valid-role`: Role attribute values are valid WAI-ARIA roles
- `required-attrs`: Required ARIA attributes present per role
- `redundant-role`: No `<nav role="navigation">` etc.
- `aria-hidden-focus`: `aria-hidden="true"` elements must not be focusable (no `<a>`, `<button>`, `<input>` inside)
- `aria-label-empty`: `aria-label` / `aria-labelledby` not empty string
- `live-region-valid`: `aria-live` values are "polite", "assertive", or "off"

- [ ] **Step 2: Commit**

---

**Commit all Chunk 3 rules:**

```bash
git add collector/tools/a11y-audit/rules/{contrast,color,typography,layout,structure,aria}.py
git commit -m "feat(a11y): add rule modules — contrast, color, typography, layout, structure, aria"
```

---

### Task 14b: Tests for Rules Group 1

**Files:**
- Create: `collector/tools/a11y-audit/tests/test_rules_g1.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for rules group 1: contrast, color, typography, layout, structure, aria."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from parsers.models import ParseContext, FileContext, ElementNode, CSSContext, CSSVariable, CSSRule
from rules.contrast import ContrastRule
from rules.color import ColorRule
from rules.typography import TypographyRule
from rules.layout import LayoutRule
from rules.structure import StructureRule
from rules.aria import AriaRule


def _make_ctx(elements=None, css_rules=None, css_vars=None, raw_content=""):
    fc = FileContext(
        path="test.php",
        elements=elements or [],
        raw_content=raw_content,
        lines=raw_content.split("\n"),
    )
    css = CSSContext(
        variables=css_vars or [],
        rules=css_rules or [],
    )
    return ParseContext(files={"test.php": fc}, css=css)


def test_structure_missing_landmarks():
    ctx = _make_ctx(elements=[
        ElementNode(tag="html", attributes={"lang": "sk"}, line=1),
        ElementNode(tag="body", attributes={}, line=2),
        ElementNode(tag="div", attributes={}, line=3),
    ])
    findings = StructureRule().check(ctx, {})
    check_ids = [f.check_id for f in findings]
    assert "landmarks" in check_ids


def test_structure_skip_link_present():
    ctx = _make_ctx(elements=[
        ElementNode(tag="html", attributes={"lang": "sk"}, line=1),
        ElementNode(tag="a", attributes={"href": "#main-content", "class": "skip-link"}, line=2),
        ElementNode(tag="main", attributes={"id": "main-content"}, line=3),
        ElementNode(tag="nav", attributes={"aria-label": "Main"}, line=4),
    ])
    findings = StructureRule().check(ctx, {})
    check_ids = [f.check_id for f in findings]
    assert "skip-link" not in check_ids


def test_aria_hidden_focusable():
    ctx = _make_ctx(elements=[
        ElementNode(tag="div", attributes={"aria-hidden": "true"}, line=1,
                    children=[ElementNode(tag="button", attributes={}, line=2)]),
    ])
    findings = AriaRule().check(ctx, {})
    check_ids = [f.check_id for f in findings]
    assert "aria-hidden-focus" in check_ids


def test_aria_valid_role():
    ctx = _make_ctx(elements=[
        ElementNode(tag="div", attributes={"role": "invalid-role"}, line=1),
    ])
    findings = AriaRule().check(ctx, {})
    check_ids = [f.check_id for f in findings]
    assert "valid-role" in check_ids


def test_typography_small_font():
    ctx = _make_ctx(css_rules=[
        CSSRule(selector=".small", properties={"font-size": "10px"}, line=5),
    ])
    config = {"thresholds": {"min_font_size": 14}}
    findings = TypographyRule().check(ctx, config)
    check_ids = [f.check_id for f in findings]
    assert "min-font-size" in check_ids


def test_layout_viewport_no_zoom():
    ctx = _make_ctx(elements=[
        ElementNode(tag="meta", attributes={"name": "viewport", "content": "width=device-width, user-scalable=no"}, line=1),
    ])
    findings = LayoutRule().check(ctx, {})
    check_ids = [f.check_id for f in findings]
    assert "viewport-meta" in check_ids
```

- [ ] **Step 2: Run tests — expect PASS**

```bash
cd collector/tools/a11y-audit && .venv/bin/python -m pytest tests/test_rules_g1.py -v
```

- [ ] **Step 3: Commit**

```bash
git commit -m "test(a11y): add tests for rules group 1"
```

---

## Chunk 4: Rules Group 2 (focus, keyboard, interactive, forms, media, language)

### Task 15: Focus rule (rules/focus.py)

Key checks: `outline-none`, `focus-visible`, `focus-indicator-contrast`, `tabindex-positive`, `focus-trap`.
Scans CSS for `:focus` / `:focus-visible` styles, checks `outline: none`/`outline: 0` without replacement. Scans HTML for `tabindex` > 0.

### Task 16: Keyboard rule (rules/keyboard.py)

Key checks: `click-no-key`, `mouse-only-handler`, `accesskey-conflict`, `interactive-div`.
Scans HTML `onclick` attrs without `onkeydown`/`onkeypress`. Scans JS `EventListener` objects for `click` without paired `keydown`. Checks `<div>`/`<span>` with click but no `role`/`tabindex`.

### Task 17: Interactive rule (rules/interactive.py)

Key checks: `hover-content`, `touch-target`, `drag-alternative`.
Scans CSS for hover-only visibility. Checks CSS target sizes. Flags drag-related JS.

### Task 18: Forms rule (rules/forms.py)

Key checks: `input-label`, `label-for`, `autocomplete`, `error-identification`, `required-indicator`, `submit-button`, `accessible-auth`.
Scans HTML `<input>`/`<select>`/`<textarea>` for associated `<label>` or `aria-label`. Checks `<label for>` matches. Checks `autocomplete` on login fields. Checks `<form>` has `<button type="submit">`.

### Task 19: Media rule (rules/media.py)

Key checks: `img-alt`, `decorative-alt`, `video-track`, `video-autoplay`, `svg-accessible`.
Scans `<img>` for `alt`. `<video>` for `<track kind="captions">` or ARIA alternative. `<svg>` for `<title>` or `aria-label`.

### Task 20: Language rule (rules/language.py)

Key checks: `html-lang`, `html-lang-valid`, `lang-change`.
Checks `<html>` has `lang` attribute with valid BCP 47 value.

**Implementation pattern for all:** Each is a class extending `BaseRule` with `check(ctx, config)` method. Iterate elements via `iter_elements()`, apply checks, return `Finding` list via `_finding()`.

- [ ] **Implement all 6 rules**
- [ ] **Commit**

```bash
git add collector/tools/a11y-audit/rules/{focus,keyboard,interactive,forms,media,language}.py
git commit -m "feat(a11y): add rule modules — focus, keyboard, interactive, forms, media, language"
```

---

### Task 20b: Tests for Rules Group 2

**Files:**
- Create: `collector/tools/a11y-audit/tests/test_rules_g2.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for rules group 2: focus, keyboard, interactive, forms, media, language."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from parsers.models import ParseContext, FileContext, ElementNode, CSSContext, CSSRule
from rules.forms import FormsRule
from rules.media import MediaRule
from rules.language import LanguageRule
from rules.focus import FocusRule


def _make_ctx(elements=None, css_rules=None, raw_content=""):
    fc = FileContext(
        path="test.php",
        elements=elements or [],
        raw_content=raw_content,
        lines=raw_content.split("\n"),
    )
    css = CSSContext(rules=css_rules or [])
    return ParseContext(files={"test.php": fc}, css=css)


def test_forms_input_without_label():
    ctx = _make_ctx(elements=[
        ElementNode(tag="input", attributes={"type": "text", "id": "name"}, line=1),
    ])
    findings = FormsRule().check(ctx, {})
    check_ids = [f.check_id for f in findings]
    assert "input-label" in check_ids


def test_forms_input_with_label():
    ctx = _make_ctx(elements=[
        ElementNode(tag="label", attributes={"for": "name"}, line=1),
        ElementNode(tag="input", attributes={"type": "text", "id": "name"}, line=2),
    ])
    findings = FormsRule().check(ctx, {})
    label_findings = [f for f in findings if f.check_id == "input-label"]
    assert len(label_findings) == 0


def test_media_img_no_alt():
    ctx = _make_ctx(elements=[
        ElementNode(tag="img", attributes={"src": "logo.png"}, line=1),
    ])
    findings = MediaRule().check(ctx, {})
    check_ids = [f.check_id for f in findings]
    assert "img-alt" in check_ids


def test_media_img_with_alt():
    ctx = _make_ctx(elements=[
        ElementNode(tag="img", attributes={"src": "logo.png", "alt": "SPJ Logo"}, line=1),
    ])
    findings = MediaRule().check(ctx, {})
    alt_findings = [f for f in findings if f.check_id == "img-alt"]
    assert len(alt_findings) == 0


def test_media_video_no_track():
    ctx = _make_ctx(elements=[
        ElementNode(tag="video", attributes={"src": "sign.mp4"}, line=1),
    ])
    findings = MediaRule().check(ctx, {})
    check_ids = [f.check_id for f in findings]
    assert "video-track" in check_ids


def test_language_missing_lang():
    ctx = _make_ctx(elements=[
        ElementNode(tag="html", attributes={}, line=1),
    ])
    findings = LanguageRule().check(ctx, {})
    check_ids = [f.check_id for f in findings]
    assert "html-lang" in check_ids


def test_language_valid_lang():
    ctx = _make_ctx(elements=[
        ElementNode(tag="html", attributes={"lang": "sk"}, line=1),
    ])
    findings = LanguageRule().check(ctx, {})
    lang_findings = [f for f in findings if f.check_id == "html-lang"]
    assert len(lang_findings) == 0


def test_focus_outline_none():
    ctx = _make_ctx(css_rules=[
        CSSRule(selector="button:focus", properties={"outline": "none"}, line=10),
    ])
    findings = FocusRule().check(ctx, {})
    check_ids = [f.check_id for f in findings]
    assert "outline-none" in check_ids
```

- [ ] **Step 2: Run tests — expect PASS**

- [ ] **Step 3: Commit**

```bash
git commit -m "test(a11y): add tests for rules group 2"
```

---

## Chunk 5: Rules Group 3 (motion, cognitive, compliance, collector)

### Task 21: Motion rule (rules/motion.py)

Key checks: `prefers-reduced-motion`, `animation-duration`, `flash-rate`, `auto-scroll`, `toast-duration`.
Checks CSS has `@media (prefers-reduced-motion)`. Checks `@keyframes` durations. Checks toast animation duration ≥ `toast_min_ms`.

### Task 22: Cognitive rule (rules/cognitive.py)

Key checks: `session-timeout`, `consistent-nav`, `on-focus-change`, `error-prevention`, `redundant-entry`.
Checks JS for session timeout durations. Checks PHP pages include consistent header/footer. Checks destructive actions have confirmation.

### Task 23: Compliance rule (rules/compliance.py)

EN 301 549 Annex A. Parses `accessibility-statement.php` for required elements:
- `conformance-status`: Searches for conformance declaration keywords
- `non-accessible-content`: Checks for issues/limitations section
- `preparation-date`: Date present
- `review-method`: Assessment method noted
- `feedback-mechanism`: Contact info present
- `enforcement-link`: Enforcement body link
- `scope-defined`: Coverage scope stated
- `standards-reference`: WCAG/EN 301 549 referenced

Outputs gaps as findings with fix suggestions.

### Task 24: Collector-specific rule (rules/collector.py)

Domain-specific checks for sign language video recording app:
- `camera-fallback`: Camera permission flow has denial fallback text
- `quality-aria-live`: Quality badges (MediaPipe) have `aria-live="polite"` container
- `consent-keyboard-trap`: Consent modals trap focus + Escape support
- `consent-focus-return`: Modal close returns focus
- `leaderboard-table`: Stats tables use `<th scope>`
- `recording-status`: Recording state changes have ARIA announcements
- `timer-accessible`: Timer has `aria-live` or `role="timer"`
- `offline-notification`: Offline status announced to assistive tech
- `framing-guide-alt`: SVG overlay has text alternative

- [ ] **Implement all 4 rules**
- [ ] **Commit**

```bash
git add collector/tools/a11y-audit/rules/{motion,cognitive,compliance,collector}.py
git commit -m "feat(a11y): add rule modules — motion, cognitive, compliance, collector"
```

---

### Task 24b: Tests for Rules Group 3

**Files:**
- Create: `collector/tools/a11y-audit/tests/test_rules_g3.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for rules group 3: motion, cognitive, compliance, collector."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from parsers.models import ParseContext, FileContext, ElementNode, CSSContext, KeyframeAnimation
from rules.motion import MotionRule
from rules.compliance import ComplianceRule
from rules.collector import CollectorRule


def _make_ctx(elements=None, css=None, raw_content="", path="test.php"):
    fc = FileContext(
        path=path,
        elements=elements or [],
        raw_content=raw_content,
        lines=raw_content.split("\n"),
    )
    return ParseContext(
        files={path: fc},
        css=css or CSSContext(),
    )


def test_motion_no_reduced_motion():
    css = CSSContext(
        keyframes=[KeyframeAnimation(name="blink", duration_ms=1000, iteration_count="infinite", line=1)],
        has_prefers_reduced_motion=False,
    )
    ctx = _make_ctx(css=css)
    findings = MotionRule().check(ctx, {})
    check_ids = [f.check_id for f in findings]
    assert "prefers-reduced-motion" in check_ids


def test_motion_with_reduced_motion():
    css = CSSContext(
        keyframes=[KeyframeAnimation(name="fade", duration_ms=500, iteration_count="1", line=1)],
        has_prefers_reduced_motion=True,
    )
    ctx = _make_ctx(css=css)
    findings = MotionRule().check(ctx, {})
    motion_findings = [f for f in findings if f.check_id == "prefers-reduced-motion"]
    assert len(motion_findings) == 0


def test_compliance_missing_statement():
    ctx = _make_ctx(path="index.php")
    findings = ComplianceRule().check(ctx, {})
    check_ids = [f.check_id for f in findings]
    # No accessibility-statement.php in files → should flag
    assert any("conformance" in cid or "feedback" in cid for cid in check_ids)


def test_collector_video_no_track():
    ctx = _make_ctx(
        elements=[ElementNode(tag="video", attributes={"id": "preview"}, line=10)],
        raw_content='<video id="preview" autoplay playsinline></video>',
    )
    findings = CollectorRule().check(ctx, {})
    # collector checks quality-aria-live, recording-status, etc.
    # video-track is in media.py, not collector.py
    assert isinstance(findings, list)
```

- [ ] **Step 2: Run tests — expect PASS**

- [ ] **Step 3: Commit**

```bash
git commit -m "test(a11y): add tests for rules group 3"
```

---

## Chunk 6: Reporters

### Task 25: Terminal reporter (reporters/terminal.py)

**Files:**
- Create: `collector/tools/a11y-audit/reporters/terminal.py`

Adapt from fluentiagrant. Colored output using colorama:
- Severity colors: RED=Critical, YELLOW=Serious, BLUE=Moderate, WHITE=Minor
- Icons: X, !, ~, i
- Summary with PASS/FAIL verdict
- Sort findings critical-first

### Task 26: JSON reporter (reporters/json_report.py)

**Files:**
- Create: `collector/tools/a11y-audit/reporters/json_report.py`

Adapt from fluentiagrant. Generates timestamped JSON with summary, violations, suppressed, compliance_gaps.
Supports `compare_reports()` for diffing against previous report.

### Task 27: HTML reporter (reporters/html_report.py + templates/report.html.j2)

**Files:**
- Create: `collector/tools/a11y-audit/reporters/html_report.py`
- Create: `collector/tools/a11y-audit/reporters/templates/report.html.j2`

Adapt from fluentiagrant. Self-contained HTML dashboard with:
- Summary cards by severity
- Findings grouped by module and WCAG criterion
- Compliance gaps checklist
- Expand/collapse controls
- Dark theme CSS

- [ ] **Implement all 3 reporters**
- [ ] **Commit**

```bash
git add collector/tools/a11y-audit/reporters/
git commit -m "feat(a11y): add reporters — terminal, JSON, HTML dashboard"
```

---

### Task 27b: Tests for reporters

**Files:**
- Create: `collector/tools/a11y-audit/tests/test_reporters.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for reporters."""
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.base import Finding, Severity, Detection
from reporters.json_report import finding_to_dict, compare_reports, report_json


def _sample_finding(rule_id="test", check_id="check", severity=Severity.MODERATE):
    return Finding(
        rule_id=rule_id,
        check_id=check_id,
        severity=severity,
        wcag="1.1.1",
        wcag_name="Non-text Content",
        message="Test finding",
        file="index.php",
        line=10,
    )


def test_finding_to_dict():
    f = _sample_finding()
    d = finding_to_dict(f)
    assert d["rule_id"] == "test"
    assert d["severity"] == "moderate"
    assert d["wcag"] == "1.1.1"
    assert d["detection"] == "static"


def test_compare_reports_new_and_fixed():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        prev = {
            "violations": [
                {"rule_id": "a", "check_id": "1", "file": "x.php", "line": 1},
                {"rule_id": "b", "check_id": "2", "file": "y.php", "line": 2},
            ]
        }
        json.dump(prev, f)
        prev_path = f.name

    current = [
        {"rule_id": "a", "check_id": "1", "file": "x.php", "line": 1},  # unchanged
        {"rule_id": "c", "check_id": "3", "file": "z.php", "line": 3},  # new
    ]
    result = compare_reports(current, prev_path)
    assert result["new_violations"] == 1
    assert result["fixed_violations"] == 1
    assert result["unchanged"] == 1

    Path(prev_path).unlink()


def test_report_json_output():
    findings = [_sample_finding()]
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"reports": {"output_dir": tmpdir}}
        path = report_json(findings, [], config)
        assert Path(path).exists()
        data = json.loads(Path(path).read_text())
        assert data["summary"]["moderate"] == 1
        assert len(data["violations"]) == 1
```

- [ ] **Step 2: Run tests — expect PASS**

- [ ] **Step 3: Commit**

```bash
git commit -m "test(a11y): add reporter tests — JSON output and comparison"
```

---

## Chunk 7: CLI & Integration

### Task 28: CLI entry point (audit.py)

**Files:**
- Create: `collector/tools/a11y-audit/audit.py`

Main orchestrator. Adapted from fluentiagrant `audit.py`:

1. Parse CLI args: `--config`, `--static-only`, `--runtime-only`, `--only`, `--files`, `--format`, `--compare`, `--verbose`
2. Load `config.yaml`
3. Resolve project root (relative to audit.py → `../../` = `collector/`)
4. Build `ParseContext`:
   - Glob include patterns, exclude patterns
   - Parse PHP files with `php_parser.parse_php_file()`
   - Parse CSS with `css_parser.parse_css_file()`
   - Parse JS files with `js_parser.parse_js_file()` (merge into context)
5. Load enabled rules from `rules/` based on config modules
6. Run each rule's `check()`, collect findings
7. Apply suppressions (config-based + inline)
8. If `--runtime`: run Playwright bridge
9. Generate reports (terminal, JSON, HTML per config)
10. Exit code: 0 if no active violations, 1 otherwise

```bash
#!/usr/bin/env python3
"""SPJ Collector Accessibility Audit Tool.

Usage:
    python audit.py                          # Full static audit
    python audit.py --runtime --base-url URL # Include runtime checks
    python audit.py --only contrast,forms    # Specific modules
    python audit.py --format json            # JSON output only
    python audit.py --compare prev.json      # Diff against previous
"""
```

- [ ] **Step 1: Implement audit.py**
- [ ] **Step 2: Test with dry run**

```bash
cd collector/tools/a11y-audit && .venv/bin/python audit.py --verbose
```

- [ ] **Step 3: Commit**

```bash
git add collector/tools/a11y-audit/audit.py
git commit -m "feat(a11y): add CLI entry point — audit.py orchestrator"
```

---

### Task 29: Playwright bridge (playwright_bridge.py)

**Files:**
- Create: `collector/tools/a11y-audit/playwright_bridge.py`

Optional runtime analysis. Only imported when `--runtime` flag is used.

- Launches headless Chromium
- Navigates to each page (configurable base URL)
- Runs axe-core via `page.evaluate()` (inject axe.min.js)
- Maps axe findings to `Finding` objects
- Deduplicates against static findings

Pages to test: index, record, validate, themes, progress, thanks, terms, consent, accessibility-statement, forgot-password, reset-password, verify-email, admin/index.

- [ ] **Step 1: Implement playwright_bridge.py**
- [ ] **Step 2: Commit**

```bash
git add collector/tools/a11y-audit/playwright_bridge.py
git commit -m "feat(a11y): add optional Playwright runtime bridge"
```

---

### Task 30: Run full audit & fix issues

- [ ] **Step 1: Run full audit on collector codebase**

```bash
cd collector/tools/a11y-audit && .venv/bin/python audit.py --format all --verbose
```

- [ ] **Step 2: Fix any tool crashes or false positives**
- [ ] **Step 3: Review HTML report in browser**
- [ ] **Step 4: Run tests**

```bash
cd collector/tools/a11y-audit && .venv/bin/python -m pytest tests/ -v
```

- [ ] **Step 5: Final commit**

```bash
git add -A collector/tools/a11y-audit/
git commit -m "fix(a11y): polish audit tool after first full run"
```

---

## Summary

| Chunk | Tasks | Files | Focus |
|-------|-------|-------|-------|
| 1: Foundation | 1-5 | base.py, models.py, helpers.py, config.yaml | Core classes, data models, color math |
| 2: Parsers | 6-8 | php_parser.py, css_parser.py, js_parser.py | Parse PHP/HTML, CSS, JS files |
| 3: Rules G1 | 9-14, 14b | contrast, color, typography, layout, structure, aria + tests | Visual & structural checks |
| 4: Rules G2 | 15-20, 20b | focus, keyboard, interactive, forms, media, language + tests | Interaction & content checks |
| 5: Rules G3 | 21-24, 24b | motion, cognitive, compliance, collector + tests | Temporal, EU compliance, domain-specific |
| 6: Reporters | 25-27, 27b | terminal, json, html + tests | Output formats |
| 7: Integration | 28-30 | audit.py, playwright_bridge.py | CLI, runtime, final polish |

**Total:** 34 tasks across 7 chunks. ~35 files to create.
