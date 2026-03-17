"""Terminal reporter -- colored CLI output using colorama."""
from __future__ import annotations

from collections import Counter

from colorama import Fore, Style, init

from rules.base import Finding, Severity, SEVERITY_ORDER

# Severity -> color and icon mapping
_SEVERITY_CONFIG = {
    Severity.CRITICAL: (Fore.RED, "X"),
    Severity.SERIOUS: (Fore.YELLOW, "!"),
    Severity.MODERATE: (Fore.BLUE, "~"),
    Severity.MINOR: (Fore.WHITE, "i"),
}


def report_terminal(
    findings: list[Finding],
    suppressed: list[Finding],
    config: dict,
) -> None:
    """Print colored terminal report."""
    init()  # Initialize colorama

    # Header
    print(f"\n{Style.BRIGHT}{'=' * 60}")
    print("  SPJ Collector -- Accessibility Audit")
    print(f"{'=' * 60}{Style.RESET_ALL}\n")

    all_findings = findings + suppressed
    n_files = len({f.file for f in all_findings})
    n_rules = len({f.rule_id for f in all_findings})
    print(f"  Files scanned: {n_files}")
    print(f"  Rules checked: {n_rules}\n")

    # Summary counts
    counts = Counter(f.severity for f in findings)
    for sev in Severity:
        color, icon = _SEVERITY_CONFIG[sev]
        count = counts.get(sev, 0)
        print(f"  {color}[{icon}] {sev.value.upper():10s} {count}{Style.RESET_ALL}")
    print()

    if not findings:
        print(
            f"  {Fore.GREEN}{Style.BRIGHT}"
            f"PASS -- No accessibility violations found"
            f"{Style.RESET_ALL}\n"
        )
        return

    # Sort: critical first, then serious, moderate, minor
    sorted_findings = sorted(
        findings,
        key=lambda f: (SEVERITY_ORDER[f.severity], f.file, f.line),
    )

    # Print each finding
    for f in sorted_findings:
        color, icon = _SEVERITY_CONFIG[f.severity]
        print(
            f"{color}[{icon}] {f.rule_id}/{f.check_id}"
            f" -- WCAG {f.wcag} ({f.wcag_name}){Style.RESET_ALL}"
        )
        print(f"    {f.message}")
        print(f"    {Fore.CYAN}{f.file}:{f.line}{Style.RESET_ALL}")
        if f.element:
            print(f"    {Style.DIM}{f.element[:120]}{Style.RESET_ALL}")
        if f.fix:
            print(f"    {Fore.GREEN}Fix: {f.fix}{Style.RESET_ALL}")
        if f.impact:
            print(f"    Impact: {', '.join(f.impact)}")
        print()

    # Suppressed
    if suppressed:
        print(
            f"  {Style.DIM}"
            f"({len(suppressed)} suppressed findings)"
            f"{Style.RESET_ALL}\n"
        )

    # Footer
    total = len(findings)
    suffix = "s" if total != 1 else ""
    print(
        f"  {Fore.RED}{Style.BRIGHT}"
        f"FAIL -- {total} accessibility violation{suffix} found"
        f"{Style.RESET_ALL}\n"
    )
