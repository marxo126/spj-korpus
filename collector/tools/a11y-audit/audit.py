#!/usr/bin/env python3
"""SPJ Collector Accessibility Audit -- CLI entry point.

Usage:
    python audit.py                          # run with defaults from config.yaml
    python audit.py --format terminal -v     # verbose terminal output
    python audit.py --only contrast,color    # run specific modules only
    python audit.py --files "admin/*.php"    # audit specific files only
    python audit.py --runtime --base-url http://localhost:8080  # include Playwright checks
    python audit.py --compare reports/audit-2026-03-15.json     # compare with previous run
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

# Ensure package imports work when running from this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml  # noqa: E402

from parsers.models import ParseContext, CSSContext  # noqa: E402
from rules.base import Finding  # noqa: E402

# ---------------------------------------------------------------------------
# Rule registry
# ---------------------------------------------------------------------------

RULE_MAP: dict[str, tuple[str, str]] = {
    "contrast": ("rules.contrast", "ContrastRule"),
    "color": ("rules.color", "ColorRule"),
    "typography": ("rules.typography", "TypographyRule"),
    "layout": ("rules.layout", "LayoutRule"),
    "structure": ("rules.structure", "StructureRule"),
    "aria": ("rules.aria", "AriaRule"),
    "focus": ("rules.focus", "FocusRule"),
    "keyboard": ("rules.keyboard", "KeyboardRule"),
    "interactive": ("rules.interactive", "InteractiveRule"),
    "forms": ("rules.forms", "FormsRule"),
    "media": ("rules.media", "MediaRule"),
    "language": ("rules.language", "LanguageRule"),
    "motion": ("rules.motion", "MotionRule"),
    "cognitive": ("rules.cognitive", "CognitiveRule"),
    "compliance": ("rules.compliance", "ComplianceRule"),
    "collector": ("rules.collector", "CollectorRule"),
}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _find_config(explicit: str | None) -> Path:
    """Locate config.yaml -- explicit path, same dir, or parent dirs."""
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Config not found: {explicit}")

    here = Path(__file__).parent
    for candidate in [here / "config.yaml", here / "config.yml"]:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("config.yaml not found in audit tool directory")


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# File discovery and parsing
# ---------------------------------------------------------------------------

def _resolve_project_root(config: dict, config_path: Path) -> Path:
    """Resolve project.root relative to config file location."""
    raw = config.get("project", {}).get("root", ".")
    root = (config_path.parent / raw).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Project root not found: {root}")
    return root


def _discover_files(
    project_root: Path,
    include_patterns: list[str],
    exclude_patterns: list[str],
    file_filter: str | None = None,
) -> list[Path]:
    """Glob include patterns, subtract exclude matches, apply optional filter."""
    found: set[Path] = set()
    for pattern in include_patterns:
        for p in project_root.glob(pattern):
            if p.is_file():
                found.add(p)

    # Apply excludes
    excluded: set[Path] = set()
    for pattern in exclude_patterns:
        for p in project_root.glob(pattern):
            excluded.add(p)
    found -= excluded

    # Apply --files filter
    if file_filter:
        filter_matches: set[Path] = set()
        for p in project_root.glob(file_filter):
            filter_matches.add(p)
        found &= filter_matches

    return sorted(found)


def build_context(
    project_root: Path,
    config: dict,
    file_filter: str | None = None,
    verbose: bool = False,
) -> ParseContext:
    """Parse all project files into a ParseContext."""
    from parsers.php_parser import parse_php_file
    from parsers.css_parser import parse_css_file
    from parsers.js_parser import parse_js_file

    include = config.get("include", ["*.php", "js/*.js", "css/*.css"])
    exclude = config.get("exclude", [])

    files = _discover_files(project_root, include, exclude, file_filter)
    if verbose:
        print(f"  Found {len(files)} files to audit")

    ctx = ParseContext(project_root=str(project_root))
    css_ctx: CSSContext | None = None

    for fpath in files:
        rel = str(fpath.relative_to(project_root))
        suffix = fpath.suffix.lower()
        try:
            if suffix == ".php":
                fc = parse_php_file(str(fpath))
                fc.path = rel
                ctx.files[rel] = fc
            elif suffix == ".css":
                css_ctx = parse_css_file(str(fpath))
            elif suffix == ".js":
                fc = parse_js_file(str(fpath))
                fc.path = rel
                ctx.files[rel] = fc
        except Exception as e:
            if verbose:
                print(f"  [!] Parse error in {rel}: {e}")

    if css_ctx:
        ctx.css = css_ctx

    if verbose:
        n_php = sum(1 for f in ctx.files if f.endswith(".php"))
        n_js = sum(1 for f in ctx.files if f.endswith(".js"))
        print(f"  Parsed: {n_php} PHP, {n_js} JS, {'1 CSS' if css_ctx else '0 CSS'}")

    return ctx


# ---------------------------------------------------------------------------
# Rule loading and execution
# ---------------------------------------------------------------------------

def load_rules(config: dict, only: str | None = None, verbose: bool = False):
    """Import and instantiate enabled rule modules."""
    modules_cfg = config.get("modules", {})
    only_set = {m.strip() for m in only.split(",")} if only else None

    rules = []
    for name, (mod_path, cls_name) in RULE_MAP.items():
        # Skip if not enabled in config
        if not modules_cfg.get(name, True):
            continue
        # Skip if --only filter active and this module not in it
        if only_set and name not in only_set:
            continue
        try:
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            rules.append(cls())
            if verbose:
                print(f"  Loaded rule: {name}")
        except Exception as e:
            print(f"  [!] Could not load rule '{name}': {e}")
    return rules


def run_rules(rules, ctx: ParseContext, config: dict, verbose: bool = False) -> list[Finding]:
    """Execute all rules and collect findings."""
    all_findings: list[Finding] = []
    for rule in rules:
        try:
            findings = rule.check(ctx, config)
            all_findings.extend(findings)
            if verbose:
                print(f"  {rule.id}: {len(findings)} findings")
        except Exception as e:
            print(f"  [!] Rule '{rule.id}' crashed: {e}")
    return all_findings


# ---------------------------------------------------------------------------
# Suppressions
# ---------------------------------------------------------------------------

def _matches_suppression(finding: Finding, supp: dict) -> bool:
    """Check if a finding matches a suppression rule."""
    if "rule" in supp and supp["rule"] != finding.rule_id:
        return False
    if "check" in supp and supp["check"] != finding.check_id:
        return False
    if "file" in supp and supp["file"] != finding.file:
        return False
    if "line" in supp and supp["line"] != finding.line:
        return False
    return True


def apply_suppressions(
    findings: list[Finding],
    config: dict,
    ctx: ParseContext,
) -> tuple[list[Finding], list[Finding]]:
    """Split findings into active and suppressed lists.

    Checks both config-based suppressions and inline suppressions from
    FileContext.suppressions (extracted by the PHP parser).
    """
    config_supps = config.get("suppressions", []) or []

    # Build inline suppression lookup: file -> list[dict]
    inline_supps: dict[str, list[dict]] = {}
    for path, fc in ctx.files.items():
        if fc.suppressions:
            inline_supps[path] = fc.suppressions

    active: list[Finding] = []
    suppressed: list[Finding] = []

    for f in findings:
        reason = ""
        is_suppressed = False

        # Check config-based suppressions
        for supp in config_supps:
            if _matches_suppression(f, supp):
                reason = supp.get("reason", "config suppression")
                is_suppressed = True
                break

        # Check inline suppressions
        if not is_suppressed and f.file in inline_supps:
            for supp in inline_supps[f.file]:
                if _matches_suppression(f, supp):
                    reason = supp.get("reason", "inline suppression")
                    is_suppressed = True
                    break

        if is_suppressed:
            # Create a new Finding with suppressed=True (frozen dataclass)
            suppressed.append(Finding(
                rule_id=f.rule_id,
                check_id=f.check_id,
                severity=f.severity,
                wcag=f.wcag,
                wcag_name=f.wcag_name,
                message=f.message,
                file=f.file,
                line=f.line,
                element=f.element,
                fix=f.fix,
                impact=f.impact,
                detection=f.detection,
                suppressed=True,
                suppression_reason=reason,
            ))
        else:
            active.append(f)

    return active, suppressed


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def generate_reports(
    active: list[Finding],
    suppressed: list[Finding],
    config: dict,
    fmt: str | None = None,
    compare_path: str | None = None,
    verbose: bool = False,
) -> None:
    """Generate reports in the requested format(s)."""
    from reporters.terminal import report_terminal
    from reporters.json_report import report_json
    from reporters.html_report import report_html

    # Determine formats
    if fmt:
        formats = [fmt] if fmt != "all" else ["terminal", "json", "html"]
    else:
        formats = config.get("reports", {}).get("formats", ["terminal"])
        if isinstance(formats, str):
            formats = [formats]

    for f in formats:
        try:
            if f == "terminal":
                report_terminal(active, suppressed, config)
            elif f == "json":
                path = report_json(active, suppressed, config, compare_path)
                if verbose:
                    print(f"  JSON report: {path}")
            elif f == "html":
                path = report_html(active, suppressed, config)
                if verbose:
                    print(f"  HTML report: {path}")
            else:
                print(f"  [!] Unknown report format: {f}")
        except Exception as e:
            print(f"  [!] Report '{f}' failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SPJ Collector Accessibility Audit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", metavar="PATH",
        help="Path to config YAML (default: config.yaml in tool dir)",
    )
    parser.add_argument(
        "--static-only", action="store_true",
        help="Skip runtime (Playwright) checks",
    )
    parser.add_argument(
        "--runtime", action="store_true",
        help="Enable Playwright runtime checks",
    )
    parser.add_argument(
        "--base-url", metavar="URL",
        help="Base URL for runtime checks (e.g. http://localhost:8080)",
    )
    parser.add_argument(
        "--only", metavar="MODULES",
        help="Comma-separated list of modules to run (e.g. contrast,color)",
    )
    parser.add_argument(
        "--files", metavar="PATTERN",
        help="Glob pattern to filter files (e.g. 'admin/*.php')",
    )
    parser.add_argument(
        "--format", metavar="FORMAT", dest="format",
        choices=["terminal", "json", "html", "all"],
        help="Output format: terminal, json, html, all (default: from config)",
    )
    parser.add_argument(
        "--compare", metavar="PATH",
        help="Compare against a previous JSON report",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # 1. Load config
    if args.verbose:
        print("Loading config...")
    config_path = _find_config(args.config)
    config = load_config(config_path)
    if args.verbose:
        print(f"  Config: {config_path}")

    # 2. Resolve project root
    project_root = _resolve_project_root(config, config_path)
    if args.verbose:
        print(f"  Project root: {project_root}")

    # 3. Build parse context
    if args.verbose:
        print("Parsing files...")
    ctx = build_context(project_root, config, args.files, args.verbose)

    if not ctx.files:
        print("No files found to audit.")
        return 0

    # 4. Load rules
    if args.verbose:
        print("Loading rules...")
    rules = load_rules(config, args.only, args.verbose)
    if not rules:
        print("No rules to run.")
        return 0

    # 5. Run static checks
    if args.verbose:
        print("Running static checks...")
    findings = run_rules(rules, ctx, config, args.verbose)

    # 6. Runtime checks (optional)
    if args.runtime and not args.static_only:
        if not args.base_url:
            print("  [!] --base-url required for runtime checks")
        else:
            if args.verbose:
                print("Running runtime checks...")
            try:
                from playwright_bridge import run_runtime_audit
                runtime_findings = run_runtime_audit(args.base_url, config)
                findings.extend(runtime_findings)
                if args.verbose:
                    print(f"  Runtime: {len(runtime_findings)} findings")
            except ImportError:
                print("  [!] playwright_bridge not available. Skipping runtime checks.")

    # 7. Apply suppressions
    active, suppressed = apply_suppressions(findings, config, ctx)
    if args.verbose:
        print(f"  Active: {len(active)}, Suppressed: {len(suppressed)}")

    # 8. Generate reports
    # Resolve report output_dir relative to project root
    reports_cfg = config.get("reports", {})
    raw_output = reports_cfg.get("output_dir", "reports")
    abs_output = (project_root / raw_output).resolve()
    abs_output.mkdir(parents=True, exist_ok=True)
    reports_cfg["output_dir"] = str(abs_output)

    generate_reports(active, suppressed, config, args.format, args.compare, args.verbose)

    # 9. Exit code
    if active:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
