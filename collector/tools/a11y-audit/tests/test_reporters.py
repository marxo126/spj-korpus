"""Tests for reporters."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.base import Detection, Finding, Severity
from reporters.json_report import compare_reports, finding_to_dict, report_json
from reporters.html_report import report_html
from reporters.terminal import report_terminal


def _sample_finding(
    rule_id="test",
    check_id="check",
    severity=Severity.MODERATE,
    wcag="1.1.1",
    wcag_name="Non-text Content",
    file="index.php",
    line=10,
):
    return Finding(
        rule_id=rule_id,
        check_id=check_id,
        severity=severity,
        wcag=wcag,
        wcag_name=wcag_name,
        message="Test finding",
        file=file,
        line=line,
    )


# --- JSON reporter ---


def test_finding_to_dict():
    f = _sample_finding()
    d = finding_to_dict(f)
    assert d["rule_id"] == "test"
    assert d["severity"] == "moderate"
    assert d["wcag"] == "1.1.1"
    assert d["detection"] == "static"


def test_finding_to_dict_with_optional_fields():
    f = Finding(
        rule_id="r1",
        check_id="c1",
        severity=Severity.CRITICAL,
        wcag="2.1.1",
        wcag_name="Keyboard",
        message="msg",
        file="a.php",
        line=1,
        element="<div>",
        fix="Add role",
        impact=("blind", "motor"),
    )
    d = finding_to_dict(f)
    assert d["element"] == "<div>"
    assert d["fix"] == "Add role"
    assert d["impact"] == ["blind", "motor"]


def test_finding_to_dict_without_optional_fields():
    f = _sample_finding()
    d = finding_to_dict(f)
    assert "element" not in d
    assert "fix" not in d
    assert "impact" not in d


def test_compare_reports_new_and_fixed():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fp:
        prev = {
            "violations": [
                {"rule_id": "a", "check_id": "1", "file": "x.php", "line": 1},
                {"rule_id": "b", "check_id": "2", "file": "y.php", "line": 2},
            ]
        }
        json.dump(prev, fp)
        prev_path = fp.name

    current = [
        {"rule_id": "a", "check_id": "1", "file": "x.php", "line": 1},
        {"rule_id": "c", "check_id": "3", "file": "z.php", "line": 3},
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
        assert data["summary"]["total"] == 1
        assert len(data["violations"]) == 1
        assert data["meta"]["version"] == "1.0.0"


def test_report_json_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"reports": {"output_dir": tmpdir}}
        path = report_json([], [], config)
        data = json.loads(Path(path).read_text())
        assert data["summary"]["total"] == 0
        assert data["violations"] == []


def test_report_json_with_suppressed():
    findings = [_sample_finding()]
    suppressed = [_sample_finding(rule_id="sup", severity=Severity.MINOR)]
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"reports": {"output_dir": tmpdir}}
        path = report_json(findings, suppressed, config)
        data = json.loads(Path(path).read_text())
        assert data["summary"]["suppressed"] == 1
        assert len(data["suppressed"]) == 1


def test_report_json_with_compare():
    findings = [_sample_finding()]
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a previous report
        prev_path = Path(tmpdir) / "prev.json"
        prev_data = {"violations": []}
        prev_path.write_text(json.dumps(prev_data))

        config = {"reports": {"output_dir": tmpdir}}
        path = report_json(findings, [], config, compare_path=str(prev_path))
        data = json.loads(Path(path).read_text())
        assert "comparison" in data
        assert data["comparison"]["new_violations"] == 1
        assert data["comparison"]["fixed_violations"] == 0


# --- HTML reporter ---


def test_report_html_output():
    findings = [
        _sample_finding(),
        _sample_finding(rule_id="contrast-ratio", check_id="text", severity=Severity.CRITICAL),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"reports": {"output_dir": tmpdir}, "project": {"name": "Test"}}
        path = report_html(findings, [], config)
        assert Path(path).exists()
        html = Path(path).read_text()
        assert "Test -- Accessibility Audit" in html
        assert "WCAG 1.1.1" in html
        assert "critical" in html.lower()


def test_report_html_pass():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"reports": {"output_dir": tmpdir}}
        path = report_html([], [], config)
        html = Path(path).read_text()
        assert "PASS" in html


def test_report_html_with_suppressed():
    suppressed = [_sample_finding(rule_id="sup")]
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"reports": {"output_dir": tmpdir}}
        path = report_html([], suppressed, config)
        html = Path(path).read_text()
        assert "suppressed" in html.lower()


def test_report_html_groups_by_module():
    findings = [
        _sample_finding(rule_id="contrast-ratio"),
        _sample_finding(rule_id="contrast-ui"),
        _sample_finding(rule_id="focus-visible"),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"reports": {"output_dir": tmpdir}}
        path = report_html(findings, [], config)
        html = Path(path).read_text()
        # Should have both module groups
        assert "contrast" in html
        assert "focus" in html


def test_report_html_groups_by_wcag():
    findings = [
        _sample_finding(wcag="1.1.1", wcag_name="Non-text Content"),
        _sample_finding(wcag="2.1.1", wcag_name="Keyboard"),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"reports": {"output_dir": tmpdir}}
        path = report_html(findings, [], config)
        html = Path(path).read_text()
        assert "1.1.1" in html
        assert "2.1.1" in html


# --- Terminal reporter ---


def test_terminal_report_pass(capsys):
    report_terminal([], [], {})
    out = capsys.readouterr().out
    assert "PASS" in out


def test_terminal_report_findings(capsys):
    findings = [_sample_finding(severity=Severity.CRITICAL)]
    report_terminal(findings, [], {})
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "CRITICAL" in out
    assert "test/check" in out


def test_terminal_report_suppressed(capsys):
    suppressed = [_sample_finding()]
    report_terminal([], suppressed, {})
    out = capsys.readouterr().out
    # No active findings -> PASS, but mention suppressed
    assert "PASS" in out


def test_terminal_report_with_all_severities(capsys):
    findings = [
        _sample_finding(severity=Severity.CRITICAL, rule_id="r1"),
        _sample_finding(severity=Severity.SERIOUS, rule_id="r2"),
        _sample_finding(severity=Severity.MODERATE, rule_id="r3"),
        _sample_finding(severity=Severity.MINOR, rule_id="r4"),
    ]
    report_terminal(findings, [], {})
    out = capsys.readouterr().out
    assert "CRITICAL" in out
    assert "SERIOUS" in out
    assert "MODERATE" in out
    assert "MINOR" in out
