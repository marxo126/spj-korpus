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


SEVERITY_ORDER: dict["Severity", int] = {
    Severity.CRITICAL: 0,
    Severity.SERIOUS: 1,
    Severity.MODERATE: 2,
    Severity.MINOR: 3,
}

# Pseudo-filename for project-wide findings with no associated file.
PROJECT_WIDE = "(project)"


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
