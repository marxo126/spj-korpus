"""PHP/HTML parser — strips PHP blocks, parses HTML with BeautifulSoup."""
from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup, Comment, Tag

from parsers.models import ElementNode, FileContext, PhpBlock

# ── PHP block patterns ─────────────────────────────────────────────────
_PHP_BLOCK_RE = re.compile(
    r"<\?(?:php|=)(.*?)(?:\?>|\Z)",
    re.DOTALL,
)

# Inline HTML inside echo/print: echo '<tag ...>';  or  echo "<tag ...>";
_ECHO_HTML_RE = re.compile(
    r"""(?:echo|print)\s+(['"])(.*?)\1\s*;""",
    re.DOTALL,
)

# Suppression comment: <!-- a11y-suppress rule:check — reason -->
_SUPPRESS_RE = re.compile(
    r"a11y-suppress\s+([\w-]+(?::[\w-]+)?)\s*(?:\u2014|-{1,2})\s*(.+)",
)


# ── Internal helpers ───────────────────────────────────────────────────

def _strip_php_blocks(content: str) -> tuple[str, list[PhpBlock]]:
    """Replace <?php ?> and <?= ?> blocks with newlines to preserve line numbers."""
    blocks: list[PhpBlock] = []
    result = content

    for m in _PHP_BLOCK_RE.finditer(content):
        block_text = m.group(0)
        start_offset = m.start()
        start_line = content[:start_offset].count("\n") + 1
        end_line = start_line + block_text.count("\n")

        blocks.append(PhpBlock(
            start_line=start_line,
            end_line=end_line,
            content=m.group(1).strip(),
        ))

    # Replace blocks with equivalent newlines to keep line numbers stable
    def _replace_with_newlines(m: re.Match) -> str:
        n = m.group(0).count("\n")
        return "\n" * n

    result = _PHP_BLOCK_RE.sub(_replace_with_newlines, result)
    return result, blocks


def _extract_echo_html(php_blocks: list[PhpBlock]) -> str:
    """Extract HTML fragments from echo statements in PHP blocks."""
    fragments: list[str] = []
    for block in php_blocks:
        for m in _ECHO_HTML_RE.finditer(block.content):
            html = m.group(2)
            if "<" in html:
                fragments.append(html)
    return "\n".join(fragments)


def _extract_suppressions(content: str) -> list[dict]:
    """Extract a11y-suppress comments from raw content."""
    suppressions: list[dict] = []
    for line_num, line in enumerate(content.splitlines(), 1):
        m = _SUPPRESS_RE.search(line)
        if m:
            target = m.group(1)
            reason = m.group(2).strip().rstrip("->").strip()
            parts = target.split(":", 1)
            suppressions.append({
                "rule": parts[0],
                "check": parts[1] if len(parts) > 1 else "*",
                "reason": reason,
                "line": line_num,
            })
    return suppressions


def _build_element_tree(soup: BeautifulSoup, source_lines: list[str]) -> list[ElementNode]:
    """Walk BeautifulSoup tree and build flat list of ElementNode."""
    elements: list[ElementNode] = []

    def _walk(tag: Tag, parent_tag: str | None = None) -> ElementNode | None:
        if not isinstance(tag, Tag):
            return

        attrs: dict[str, str | bool | None] = {}
        for key, value in tag.attrs.items():
            if isinstance(value, list):
                attrs[key] = " ".join(value)
            elif isinstance(value, bool):
                attrs[key] = value
            else:
                attrs[key] = str(value) if value is not None else None

        # Approximate line number from sourceline (BeautifulSoup with lxml)
        line = getattr(tag, "sourceline", None) or 0

        text = tag.string or ""
        if not text:
            text = tag.get_text(separator=" ", strip=True)[:200]

        node = ElementNode(
            tag=tag.name,
            attributes=attrs,
            line=line,
            parent_tag=parent_tag,
            text_content=str(text),
        )
        elements.append(node)

        for child in tag.children:
            if isinstance(child, Tag):
                child_node = _walk(child, tag.name)
                if child_node:
                    node.children.append(child_node)

        return node

    for child in soup.children:
        if isinstance(child, Tag):
            _walk(child)

    return elements


# ── Public API ─────────────────────────────────────────────────────────

def parse_html_string(content: str, path: str = "<string>") -> FileContext:
    """Parse an HTML string and return a FileContext."""
    lines = content.splitlines()
    soup = BeautifulSoup(content, "lxml")
    elements = _build_element_tree(soup, lines)
    suppressions = _extract_suppressions(content)

    return FileContext(
        path=path,
        elements=elements,
        raw_content=content,
        lines=lines,
        suppressions=suppressions,
    )


def parse_php_file(path: str | Path) -> FileContext:
    """Parse a PHP file: strip PHP blocks, parse HTML, extract echo HTML."""
    path = Path(path)
    content = path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()

    # Strip PHP blocks, preserving line numbers
    html_content, php_blocks = _strip_php_blocks(content)

    # Parse the remaining HTML
    soup = BeautifulSoup(html_content, "lxml")
    elements = _build_element_tree(soup, lines)

    # Also extract HTML from echo statements
    echo_html = _extract_echo_html(php_blocks)
    if echo_html:
        echo_soup = BeautifulSoup(echo_html, "lxml")
        echo_elements = _build_element_tree(echo_soup, [])
        elements.extend(echo_elements)

    suppressions = _extract_suppressions(content)

    return FileContext(
        path=str(path),
        elements=elements,
        php_blocks=php_blocks,
        raw_content=content,
        lines=lines,
        suppressions=suppressions,
    )
