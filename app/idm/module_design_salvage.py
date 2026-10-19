"""Partial salvage of a W3/W4 module answer: the valid lessons are kept, the rejected ones are listed with
their codes (safe paths only) for the repair prompt and the attempt log.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from pydantic import ValidationError

from app.contracts.orchestration_v2.blueprint import LessonArchitectureV2
from app.contracts.orchestration_v2.provider import ChapterShardPlanV2
from app.idm.contracts import IdmLessonDesignV1, IdmPracticeTaskV1
from app.idm.module_autofix import autofix_lesson
from app.idm.module_design_lessons import normalize_lesson, validate_lesson
from app.idm.module_layout import ModuleScope, project_lesson
from app.idm.runtime import SCHEMA_INVALID_CODE, IdmRuntime, log_stage
from app.idm.validation import IdmIssue, errors

# Failing codes/paths logged per W4 attempt (QC run 8de1c76b, Q1d); paths are server-built, never content.
_MAX_LOGGED_ISSUES: Final = 24
_UNIT_PATH_RE: Final = re.compile(r"\.units\[(\d+)\]")


# --- orchestration -------------------------------------------------------------------------------
def _course_blocks_payload(scope: ModuleScope) -> list[dict[str, Any]]:
    payload = []
    for plan in scope.lesson_plans:
        for block_id in plan.block_ids:
            block, row = scope.blocks[block_id], scope.rows[block_id]
            payload.append(
                {
                    "block_id": block_id,
                    "lesson_key": plan.lesson_key,
                    "name": block.name,
                    "intent": block.intent,
                    "content_kind": block.content_kind,
                    "classification": row.classification,
                    "treatment": row.treatment,
                    "detail_level": row.detail_level,
                    "must_do_ids": row.must_do_ids,
                    "fact_keys": list(block.fact_keys),
                }
            )
    return payload


def _component_orders(lesson: IdmLessonDesignV1, issues: Sequence[IdmIssue]) -> list[str]:
    """The component types of each unit that fails the component order, as "units[i]=html,html,problem"."""

    orders: list[str] = []
    for issue in issues:
        match = _UNIT_PATH_RE.search(issue.path) if issue.code == "IDM_W4_COMPONENT_ORDER" else None
        if match is not None and int(match.group(1)) < len(lesson.units):
            unit = lesson.units[int(match.group(1))]
            orders.append(f"units[{match.group(1)}]=" + ",".join(component.type for component in unit.components))
    return orders


@dataclass
class _Salvage:
    accepted: dict[str, IdmLessonDesignV1] = field(default_factory=dict)
    failures: dict[str, list[IdmIssue]] = field(default_factory=dict)
    # Deterministic layout fixes applied to otherwise failing lessons (``module_autofix``), counted.
    fixes: Counter[str] = field(default_factory=Counter)
    practices: dict[str, list[IdmPracticeTaskV1]] = field(default_factory=dict)
    # lesson_key -> "units[i]=html,html,problem" of each unit still failing the component order after the autofix
    # (QC run ab8d67e1, R7: only CODE@path was logged, so the cause had to be guessed). Server enum values only.
    component_orders: dict[str, list[str]] = field(default_factory=dict)


def _salvage(
    lessons: Sequence[IdmLessonDesignV1],
    scope: ModuleScope,
    allowed: set[str],
    *,
    invalid: Mapping[str, list[IdmIssue]] | None = None,
) -> _Salvage:
    """Validate each lesson independently; return accepted lessons and per-lesson errors.

    A failing lesson first gets the deterministic layout fixes (QC run 8de1c76b, Q1c: two units sharing the
    only block of a lesson, components out of order, an uncovered block ...); it is accepted when they make it
    valid. ``invalid`` are lessons of a rejected answer that did not even parse (their schema paths).
    """

    result = _Salvage()
    by_key = {lesson.lesson_key: lesson for lesson in lessons}
    for position, plan in enumerate(scope.lesson_plans):
        lesson = by_key.get(plan.lesson_key)
        path = f"lessons[{position}]"
        if lesson is None:
            result.failures[plan.lesson_key] = (invalid or {}).get(plan.lesson_key) or [
                IdmIssue("IDM_W3_LESSON_SET_MISMATCH", path)]
            continue
        if lesson.practice_tasks:
            result.practices[plan.lesson_key] = list(lesson.practice_tasks)
        found = validate_lesson(lesson, plan, scope, allowed, path)
        if errors(found):
            fixed, fixes = autofix_lesson(lesson, plan, scope, allowed)
            refound = validate_lesson(fixed, plan, scope, allowed, path) if fixes else found
            if fixes and not errors(refound):
                lesson, found = fixed, refound
                result.fixes.update(fixes)
        if errors(found):
            result.failures[plan.lesson_key] = errors(found)
            orders = _component_orders(lesson, errors(found))
            if orders:
                result.component_orders[plan.lesson_key] = orders
        else:
            normalized = normalize_lesson(
                lesson, plan, scope.locale, [issue.code for issue in found if issue.severity == "warning"]
            )
            try:
                # Normalisation can shorten text below a bound; prove both projections before accepting.
                IdmLessonDesignV1.model_validate(normalized.model_dump(mode="json"))
                LessonArchitectureV2.model_validate(project_lesson(normalized, plan, scope))
            except ValidationError:
                result.failures[plan.lesson_key] = [IdmIssue("IDM_W4_PROJECTION_INVALID", path)]
            else:
                result.accepted[plan.lesson_key] = normalized
    if invalid is None and [lesson.lesson_key for lesson in lessons] != [plan.lesson_key
                                                                         for plan in scope.lesson_plans]:
        result.failures.setdefault("__set__", []).append(IdmIssue("IDM_W3_LESSON_SET_MISMATCH", "lessons"))
    return result


def _parse_rejected_lessons(value: Any, scope: ModuleScope) -> tuple[list[IdmLessonDesignV1], dict[str, list[IdmIssue]],
                                                                     dict[str, list[IdmPracticeTaskV1]]]:
    """The lessons of a schema-rejected W4 answer that are valid on their own (QC run 8de1c76b, Q1b).

    One over-long list in lesson 3 used to discard lessons 1 and 2 too; they are now checked like the lessons
    of an accepted answer. Lessons that do not parse are reported with their position; their valid practice
    tasks still reach the fallback layout.
    """

    raw = value.get("lessons") if isinstance(value, dict) else None
    known = {plan.lesson_key for plan in scope.lesson_plans}
    parsed: list[IdmLessonDesignV1] = []
    invalid: dict[str, list[IdmIssue]] = {}
    practices: dict[str, list[IdmPracticeTaskV1]] = {}
    for position, item in enumerate(raw if isinstance(raw, list) else []):
        key = item.get("lesson_key") if isinstance(item, dict) else None
        try:
            parsed.append(IdmLessonDesignV1.model_validate_json(json.dumps(item, ensure_ascii=False)))
            continue
        except ValidationError:
            if not isinstance(key, str) or key not in known:
                continue
        invalid[key] = [IdmIssue(SCHEMA_INVALID_CODE, f"lessons[{position}]")]
        tasks = item.get("practice_tasks") if isinstance(item, dict) else None
        valid_tasks = []
        for task in tasks if isinstance(tasks, list) else []:
            try:
                valid_tasks.append(IdmPracticeTaskV1.model_validate_json(json.dumps(task, ensure_ascii=False)))
            except ValidationError:
                continue
        if valid_tasks:
            practices[key] = valid_tasks
    return parsed, invalid, practices


def _issue_log(failures: Mapping[str, Sequence[IdmIssue]]) -> dict[str, list[str]]:
    """``CODE@path`` per failing lesson (server-built paths, never content), bounded."""

    logged: dict[str, list[str]] = {}
    budget = _MAX_LOGGED_ISSUES
    for lesson_key, issues in failures.items():
        items = [f"{issue.code}@{issue.path}" for issue in issues][:budget]
        if items:
            logged[lesson_key] = items
            budget -= len(items)
        if budget <= 0:
            break
    return logged


def _log_attempt(runtime: IdmRuntime, plan: ChapterShardPlanV2, *, call: int, kind: str, outcome: str,
                 salvage: _Salvage | None = None, schema_errors: Sequence[dict[str, Any]] = ()) -> None:
    """One line per W4 call (QC run 8de1c76b, Q1d: ch1.l2 failed twice and only the merged codes were logged)."""

    log_stage("idm_module_attempt", {
        "correlation_id": runtime.correlation_id, "stage": "idm_module", "chapter_key": plan.chapter_key,
        "shard_index": plan.shard_index, "call": call, "invocation_kind": kind, "outcome": outcome,
        "schema_errors": [f"{item.get('type')}@{'.'.join(str(part) for part in item.get('loc', []))}"
                          for item in schema_errors][:_MAX_LOGGED_ISSUES],
        "accepted_lessons": sorted(salvage.accepted) if salvage else [],
        "lesson_failures": _issue_log(salvage.failures) if salvage else {},
        "autofix": dict(sorted(salvage.fixes.items())) if salvage else {},
        "component_orders": dict(sorted(salvage.component_orders.items())) if salvage else {},
    })
