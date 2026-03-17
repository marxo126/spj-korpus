"""JSON reporter -- machine-readable audit output."""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from rules.base import Finding, Severity


def finding_to_dict(f: Finding) -> dict:
    """Serialize a Finding to a plain dict."""
    d: dict = {
        "rule_id": f.rule_id,
        "check_id": f.check_id,
        "severity": f.severity.value,
        "wcag": f.wcag,
        "wcag_name": f.wcag_name,
        "message": f.message,
        "file": f.file,
        "line": f.line,
        "detection": f.detection.value,
    }
    if f.element:
        d["element"] = f.element
    if f.fix:
        d["fix"] = f.fix
    if f.impact:
        d["impact"] = list(f.impact)
    return d


def _violation_key(v: dict) -> str:
    """Create a stable key for comparing violations across runs."""
    return f"{v['rule_id']}:{v['check_id']}:{v['file']}:{v['line']}"


def compare_reports(current_violations: list[dict], previous_path: str) -> dict:
    """Compare current violations against a previous report.

    Returns dict with new_violations, fixed_violations, unchanged counts
    and the corresponding lists.
    """
    prev_data = json.loads(Path(previous_path).read_text())
    prev_violations = prev_data.get("violations", [])

    prev_keys = {_violation_key(v) for v in prev_violations}
    curr_keys = {_violation_key(v) for v in current_violations}

    new_keys = curr_keys - prev_keys
    fixed_keys = prev_keys - curr_keys
    unchanged_keys = curr_keys & prev_keys

    new_list = [v for v in current_violations if _violation_key(v) in new_keys]
    fixed_list = [v for v in prev_violations if _violation_key(v) in fixed_keys]

    return {
        "previous_report": previous_path,
        "new_violations": len(new_list),
        "fixed_violations": len(fixed_list),
        "unchanged": len(unchanged_keys),
        "new": new_list,
        "fixed": fixed_list,
    }


def report_json(
    findings: list[Finding],
    suppressed: list[Finding],
    config: dict,
    compare_path: str | None = None,
) -> str:
    """Generate JSON report, save to file, return file path."""
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    date_str = now.strftime("%Y-%m-%d")

    all_findings = findings + suppressed
    n_files = len({f.file for f in all_findings}) if all_findings else 0
    n_rules = len({f.rule_id for f in all_findings}) if all_findings else 0

    counts = Counter(f.severity for f in findings)

    violations = [finding_to_dict(f) for f in findings]
    suppressed_list = [finding_to_dict(f) for f in suppressed]

    report: dict = {
        "meta": {
            "timestamp": timestamp,
            "version": "1.0.0",
            "wcag_version": config.get("standards", {}).get("wcag", "2.2"),
            "conformance_level": config.get("standards", {}).get(
                "conformance", "AA"
            ),
            "standards": ["WCAG 2.2", "EN 301 549 v3.2.1"],
            "files_scanned": n_files,
            "rules_checked": n_rules,
        },
        "summary": {
            "critical": counts.get(Severity.CRITICAL, 0),
            "serious": counts.get(Severity.SERIOUS, 0),
            "moderate": counts.get(Severity.MODERATE, 0),
            "minor": counts.get(Severity.MINOR, 0),
            "suppressed": len(suppressed),
            "total": len(findings),
        },
        "violations": violations,
        "suppressed": suppressed_list,
    }

    if compare_path:
        report["comparison"] = compare_reports(violations, compare_path)

    # Write to file
    reports_cfg = config.get("reports", {})
    output_dir = Path(reports_cfg.get("output_dir", "reports"))
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"audit-{date_str}.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    return str(output_path)
