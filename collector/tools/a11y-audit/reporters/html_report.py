"""HTML reporter -- self-contained dashboard using Jinja2."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from rules.base import Finding, Severity, SEVERITY_ORDER

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _group_by_module(findings: list[Finding]) -> dict[str, list[Finding]]:
    """Group findings by rule module (rule_id prefix before first dot or full id)."""
    groups: dict[str, list[Finding]] = defaultdict(list)
    for f in findings:
        module = f.rule_id.split("-")[0] if "-" in f.rule_id else f.rule_id
        groups[module].append(f)
    return dict(sorted(groups.items()))


def _group_by_wcag(findings: list[Finding]) -> dict[str, list[Finding]]:
    """Group findings by WCAG criterion."""
    groups: dict[str, list[Finding]] = defaultdict(list)
    for f in findings:
        key = f"{f.wcag} {f.wcag_name}"
        groups[key].append(f)
    return dict(sorted(groups.items()))


def _wcag_understanding_url(criterion: str) -> str:
    """Build W3C Understanding WCAG 2.2 URL for a criterion.

    Maps e.g. '1.1.1' -> 'https://www.w3.org/WAI/WCAG22/Understanding/non-text-content'
    Falls back to the generic understanding page.
    """
    _SLUG_MAP = {
        "1.1.1": "non-text-content",
        "1.2.1": "audio-only-and-video-only-prerecorded",
        "1.2.2": "captions-prerecorded",
        "1.2.3": "audio-description-or-media-alternative-prerecorded",
        "1.2.5": "audio-description-prerecorded",
        "1.3.1": "info-and-relationships",
        "1.3.2": "meaningful-sequence",
        "1.3.3": "sensory-characteristics",
        "1.3.4": "orientation",
        "1.3.5": "identify-input-purpose",
        "1.4.1": "use-of-color",
        "1.4.2": "audio-control",
        "1.4.3": "contrast-minimum",
        "1.4.4": "resize-text",
        "1.4.5": "images-of-text",
        "1.4.10": "reflow",
        "1.4.11": "non-text-contrast",
        "1.4.12": "text-spacing",
        "1.4.13": "content-on-hover-or-focus",
        "2.1.1": "keyboard",
        "2.1.2": "no-keyboard-trap",
        "2.1.4": "character-key-shortcuts",
        "2.2.1": "timing-adjustable",
        "2.2.2": "pause-stop-hide",
        "2.3.1": "three-flashes-or-below-threshold",
        "2.4.1": "bypass-blocks",
        "2.4.2": "page-titled",
        "2.4.3": "focus-order",
        "2.4.4": "link-purpose-in-context",
        "2.4.5": "multiple-ways",
        "2.4.6": "headings-and-labels",
        "2.4.7": "focus-visible",
        "2.4.11": "focus-not-obscured-minimum",
        "2.5.1": "pointer-gestures",
        "2.5.2": "pointer-cancellation",
        "2.5.3": "label-in-name",
        "2.5.4": "motion-actuation",
        "2.5.7": "dragging-movements",
        "2.5.8": "target-size-minimum",
        "3.1.1": "language-of-page",
        "3.1.2": "language-of-parts",
        "3.2.1": "on-focus",
        "3.2.2": "on-input",
        "3.2.6": "consistent-help",
        "3.3.1": "error-identification",
        "3.3.2": "labels-or-instructions",
        "3.3.3": "error-suggestion",
        "3.3.7": "redundant-entry",
        "3.3.8": "accessible-authentication-minimum",
        "4.1.2": "name-role-value",
        "4.1.3": "status-messages",
    }
    slug = _SLUG_MAP.get(criterion)
    if slug:
        return f"https://www.w3.org/WAI/WCAG22/Understanding/{slug}"
    return "https://www.w3.org/WAI/WCAG22/Understanding/"


def report_html(
    findings: list[Finding],
    suppressed: list[Finding],
    config: dict,
) -> str:
    """Render HTML report, save to file, return file path."""
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y-%m-%d %H:%M UTC")
    date_str = now.strftime("%Y-%m-%d")

    all_findings = findings + suppressed
    n_files = len({f.file for f in all_findings}) if all_findings else 0
    n_rules = len({f.rule_id for f in all_findings}) if all_findings else 0

    counts = Counter(f.severity for f in findings)

    sorted_findings = sorted(
        findings,
        key=lambda f: (SEVERITY_ORDER[f.severity], f.file, f.line),
    )

    by_module = _group_by_module(findings)
    by_wcag = _group_by_wcag(findings)

    # Build WCAG URLs
    wcag_urls = {}
    for f in all_findings:
        if f.wcag not in wcag_urls:
            wcag_urls[f.wcag] = _wcag_understanding_url(f.wcag)

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=True,
    )
    template = env.get_template("report.html.j2")

    html = template.render(
        project_name=config.get("project", {}).get(
            "name", "SPJ Collector"
        ),
        timestamp=timestamp,
        n_files=n_files,
        n_rules=n_rules,
        counts={
            "critical": counts.get(Severity.CRITICAL, 0),
            "serious": counts.get(Severity.SERIOUS, 0),
            "moderate": counts.get(Severity.MODERATE, 0),
            "minor": counts.get(Severity.MINOR, 0),
            "total": len(findings),
        },
        findings=sorted_findings,
        by_module=by_module,
        by_wcag=by_wcag,
        suppressed=suppressed,
        wcag_urls=wcag_urls,
        severity_order=SEVERITY_ORDER,
    )

    reports_cfg = config.get("reports", {})
    output_dir = Path(reports_cfg.get("output_dir", "reports"))
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"audit-{date_str}.html"
    output_path.write_text(html, encoding="utf-8")

    return str(output_path)
