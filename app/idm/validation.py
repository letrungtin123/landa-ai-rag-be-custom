"""Validation findings shared by the IDM stage validators."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class IdmIssue:
    """A deterministic finding: a stable code plus a content-free location."""

    code: str
    path: str
    severity: Severity = "error"

    def as_repair_item(self) -> dict[str, str]:
        return {"code": self.code, "path": self.path}


def errors(issues: Iterable[IdmIssue]) -> list[IdmIssue]:
    return [issue for issue in issues if issue.severity == "error"]


def warnings(issues: Iterable[IdmIssue]) -> list[IdmIssue]:
    return [issue for issue in issues if issue.severity == "warning"]


def code_counts(issues: Iterable[IdmIssue]) -> dict[str, int]:
    return dict(sorted(Counter(issue.code for issue in issues).items()))
