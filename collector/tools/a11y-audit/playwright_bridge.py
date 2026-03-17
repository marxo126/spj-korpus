"""Optional Playwright runtime bridge for accessibility audit.

Requires: pip install playwright && playwright install chromium

Usage: audit.py --runtime --base-url http://localhost:8080
"""
from __future__ import annotations

from rules.base import Finding, Severity, Detection

# Pages to test (relative to base URL)
PAGES = [
    "/",
    "/record.php",
    "/validate.php",
    "/themes.php",
    "/progress.php",
    "/thanks.php",
    "/terms.php",
    "/consent.php",
    "/accessibility-statement.php",
    "/forgot-password.php",
    "/reset-password.php",
    "/verify-email.php",
    "/admin/",
]


def run_runtime_audit(base_url: str, config: dict) -> list[Finding]:
    """Run Playwright + axe-core audit against a live server."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  [!] Playwright not installed. Run: pip install playwright && playwright install chromium")
        print("  [!] Skipping runtime checks.")
        return []

    findings: list[Finding] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Inject axe-core
        axe_script = _get_axe_script()
        if not axe_script:
            print("  [!] Could not load axe-core. Skipping runtime checks.")
            browser.close()
            return []

        for path in PAGES:
            url = f"{base_url.rstrip('/')}{path}"
            try:
                page.goto(url, wait_until="networkidle", timeout=10000)
            except Exception as e:
                print(f"  [!] Could not load {url}: {e}")
                continue

            # Run axe-core
            page.evaluate(axe_script)
            results = page.evaluate("async () => await axe.run()")

            for violation in results.get("violations", []):
                for node in violation.get("nodes", []):
                    findings.append(Finding(
                        rule_id="axe-core",
                        check_id=violation.get("id", "unknown"),
                        severity=_map_axe_severity(violation.get("impact", "moderate")),
                        wcag=_extract_wcag(violation),
                        wcag_name=violation.get("help", ""),
                        message=violation.get("description", ""),
                        file=path,
                        line=0,
                        element=node.get("html", "")[:200],
                        fix=node.get("failureSummary", ""),
                        detection=Detection.RUNTIME,
                    ))

        browser.close()

    return findings


def _map_axe_severity(impact: str) -> Severity:
    """Map axe-core impact level to our Severity enum."""
    return {
        "critical": Severity.CRITICAL,
        "serious": Severity.SERIOUS,
        "moderate": Severity.MODERATE,
        "minor": Severity.MINOR,
    }.get(impact, Severity.MODERATE)


def _extract_wcag(violation: dict) -> str:
    """Extract WCAG criterion number from axe-core tags."""
    tags = violation.get("tags", [])
    for tag in tags:
        if tag.startswith("wcag") and len(tag) > 4:
            # Convert wcag111 to 1.1.1
            digits = tag[4:]
            if len(digits) >= 3:
                return f"{digits[0]}.{digits[1]}.{digits[2:]}"
    return ""


def _get_axe_script() -> str | None:
    """Get axe-core script from CDN."""
    try:
        import urllib.request
        url = "https://cdnjs.cloudflare.com/ajax/libs/axe-core/4.9.1/axe.min.js"
        resp = urllib.request.urlopen(url, timeout=10)
        return resp.read().decode("utf-8")
    except Exception:
        return None
