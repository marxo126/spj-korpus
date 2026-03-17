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
    mode: str = "light"


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
