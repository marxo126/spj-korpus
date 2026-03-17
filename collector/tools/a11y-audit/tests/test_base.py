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
