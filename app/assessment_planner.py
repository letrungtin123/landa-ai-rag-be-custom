from __future__ import annotations

"""Server-owned V5 assessment-plan compilation.

The Course Architect can declare lesson assessment intent, but it must not
choose canonical facts, provenance ownership, or an arbitrary assessment
anchor.  This module converts that declaration into the smallest safe set of
``knowledge_check`` semantic blocks before the coherence validator runs.

It deliberately works on a copy and returns typed outcomes.  The caller may
persist a compiled candidate only after the existing evidence, coherence and
canonical-fact validators pass.
"""

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Literal


AssessmentPlanStatus = Literal["READY", "NEEDS_SEMANTIC_RESOLUTION", "TERMINAL_GAP"]

TEACHING_INTENTS = frozenset({
    "concept_explanation", "definition", "example", "worked_example",
    "procedure", "comparison", "warning", "tip",
})


def _text_ids(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {
        item.strip()
        for item in value
        if isinstance(item, str) and item.strip()
    }


def _safe_path(chapter_index: int, lesson_index: int, unit_index: int | None = None) -> str:
    base = f"chapter_{chapter_index}.lesson_{lesson_index}"
    return f"{base}.unit_{unit_index}" if unit_index is not None else base


def _local_objective_ids(lesson: dict[str, Any]) -> set[str]:
    return {
        f"lo_{index}"
        for index, objective in enumerate(lesson.get("learning_objectives", []), start=1)
        if isinstance(objective, str) and objective.strip()
    }


def _candidate_semantic_descriptor(block: dict[str, Any]) -> dict[str, str]:
    """Return only bounded provider-facing semantic metadata, never evidence."""

    content = block.get("content")
    content = content if isinstance(content, dict) else {}
    descriptor: dict[str, str] = {"intent": str(block.get("intent") or "").strip()}
    for key in ("purpose", "learner_action"):
        value = content.get(key)
        if isinstance(value, str) and value.strip():
            descriptor[key] = value.strip()[:480]
    return descriptor


@dataclass(frozen=True)
class AssessmentTeachingCandidate:
    lesson_path: str
    unit_path: str
    unit_index: int
    block_id: str
    block_index: int
    objective_ref: str
    source_refs: tuple[str, ...]
    concept_ids: tuple[str, ...]
    primary_evidence_scope_ids: tuple[str, ...]
    semantic_descriptor: dict[str, str]
    alignment: Literal["fully_aligned", "semantic_candidate"]

    def safe_provider_value(self) -> dict[str, Any]:
        return {
            "objective_ref": self.objective_ref,
            "unit_path": self.unit_path,
            "teaching_block_id": self.block_id,
            "semantic_descriptor": dict(self.semantic_descriptor),
        }


@dataclass(frozen=True)
class AssessmentPlanIssue:
    code: str
    lesson_path: str
    objective_ref: str | None
    safe_reason: str
    candidate_count: int = 0

    def workflow_issue(self) -> dict[str, Any]:
        issue: dict[str, Any] = {
            "code": self.code,
            "severity": "error",
            "path": self.lesson_path,
            "repairable": self.code == "ASSESSMENT_SEMANTIC_SELECTION_REQUIRED",
            "safe_reason": self.safe_reason,
            "assessment_candidate_count": self.candidate_count,
        }
        if self.objective_ref:
            issue["objective_ids"] = [self.objective_ref]
        return issue


@dataclass(frozen=True)
class AssessmentPlanCompilation:
    status: AssessmentPlanStatus
    blueprint: dict[str, Any]
    issues: list[AssessmentPlanIssue] = field(default_factory=list)
    candidates: dict[tuple[str, str], tuple[AssessmentTeachingCandidate, ...]] = field(default_factory=dict)
    deterministic_insertions: int = 0
    existing_checks: int = 0

    @property
    def metrics(self) -> dict[str, int | str]:
        return {
            "assessment_plan_status": self.status,
            "assessment_plan_deterministic_insertions": self.deterministic_insertions,
            "assessment_plan_existing_checks": self.existing_checks,
            "assessment_plan_semantic_resolution_objectives": len(self.candidates),
            "assessment_plan_terminal_gaps": sum(1 for issue in self.issues if issue.code != "ASSESSMENT_SEMANTIC_SELECTION_REQUIRED"),
        }


def assessment_plan_fingerprint(
    blueprint: dict[str, Any],
    *,
    lesson_path: str,
    candidates: dict[tuple[str, str], tuple[AssessmentTeachingCandidate, ...]],
) -> str:
    """Fingerprint server-approved selection authority without customer text."""

    payload = {
        "lesson_path": lesson_path,
        "version": blueprint.get("architecture_contract_version"),
        "candidates": [
            {
                "objective_ref": objective_ref,
                "candidates": [
                    {
                        "unit_path": candidate.unit_path,
                        "block_id": candidate.block_id,
                        "primary_scope_ids": list(candidate.primary_evidence_scope_ids),
                        "concept_ids": list(candidate.concept_ids),
                    }
                    for candidate in options
                ],
            }
            for (path, objective_ref), options in sorted(candidates.items())
            if path == lesson_path
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:32]


def _server_block_id(unit: dict[str, Any], objective_refs: tuple[str, ...]) -> str:
    existing = {
        str(block.get("id") or "").strip()
        for block in unit.get("learning_blocks", [])
        if isinstance(block, dict) and str(block.get("id") or "").strip()
    }
    digest = hashlib.sha256("|".join(objective_refs).encode("utf-8")).hexdigest()[:10]
    base = f"lb_server_assessment_{digest}"
    identifier = base
    suffix = 2
    while identifier in existing:
        identifier = f"{base}_{suffix}"
        suffix += 1
    return identifier


def _insert_after(blocks: list[dict[str, Any]], block_id: str) -> int:
    for index, block in enumerate(blocks):
        if str(block.get("id") or "").strip() == block_id:
            return index + 1
    raise ValueError("assessment teaching block disappeared during copy-on-write compilation")


def _candidate_from_block(
    *,
    lesson_path: str,
    unit_path: str,
    unit_index: int,
    block: dict[str, Any],
    block_index: int,
    objective_ref: str,
    unit_objective_refs: set[str],
) -> AssessmentTeachingCandidate | None:
    if str(block.get("intent") or "").strip() not in TEACHING_INTENTS:
        return None
    primary_scope_ids = tuple(sorted(_text_ids(block.get("primary_evidence_scope_ids"))))
    if not primary_scope_ids:
        return None
    existing_objectives = _text_ids(block.get("learning_objective_refs"))
    if objective_ref in existing_objectives:
        alignment: Literal["fully_aligned", "semantic_candidate"] = "fully_aligned"
    elif objective_ref in unit_objective_refs:
        # It is source-scope safe, but the model must confirm the objective
        # association from compact semantic metadata before the server mutates.
        alignment = "semantic_candidate"
    else:
        return None
    block_id = str(block.get("id") or "").strip()
    if not block_id:
        return None
    return AssessmentTeachingCandidate(
        lesson_path=lesson_path,
        unit_path=unit_path,
        unit_index=unit_index,
        block_id=block_id,
        block_index=block_index,
        objective_ref=objective_ref,
        source_refs=tuple(sorted(_text_ids(block.get("source_refs")))),
        concept_ids=tuple(sorted(_text_ids(block.get("concept_ids")))),
        primary_evidence_scope_ids=primary_scope_ids,
        semantic_descriptor=_candidate_semantic_descriptor(block),
        alignment=alignment,
    )


def _group_insertions(
    planned: list[AssessmentTeachingCandidate],
) -> dict[tuple[str, str], list[AssessmentTeachingCandidate]]:
    groups: dict[tuple[str, str], list[AssessmentTeachingCandidate]] = {}
    for candidate in planned:
        groups.setdefault((candidate.unit_path, candidate.block_id), []).append(candidate)
    return groups


def compile_v5_assessment_plan(
    blueprint: dict[str, Any],
    *,
    selections: dict[tuple[str, str], tuple[str, str]] | None = None,
    lesson_paths: set[str] | None = None,
) -> AssessmentPlanCompilation:
    """Compile only V5 missing assessment blocks, without provenance mutation.

    ``selections`` is server-validated later by the semantic-delta guard and
    maps ``(lesson_path, objective_ref)`` to ``(unit_path, block_id)``.  A
    deterministic insertion occurs only where exactly one fully-aligned
    teaching block exists.  Ambiguous or merely semantic candidates are never
    selected by position or fallback order.
    """

    if blueprint.get("architecture_contract_version") != 5:
        return AssessmentPlanCompilation("READY", deepcopy(blueprint))

    candidate = deepcopy(blueprint)
    selections = selections or {}
    issues: list[AssessmentPlanIssue] = []
    unresolved: dict[tuple[str, str], tuple[AssessmentTeachingCandidate, ...]] = {}
    deterministic_insertions = existing_checks = 0

    for chapter_index, chapter in enumerate(candidate.get("chapters", []), start=1):
        if not isinstance(chapter, dict):
            continue
        for lesson_index, lesson in enumerate(chapter.get("lessons", []), start=1):
            if not isinstance(lesson, dict):
                continue
            lesson_path = _safe_path(chapter_index, lesson_index)
            if lesson_paths is not None and lesson_path not in lesson_paths:
                continue
            if lesson.get("assessment_required") is not True:
                continue
            required_refs = _text_ids(lesson.get("assessment_objective_refs"))
            valid_refs = _local_objective_ids(lesson)
            if not required_refs or not required_refs.issubset(valid_refs):
                issues.append(AssessmentPlanIssue(
                    "ASSESSMENT_PLAN_INVALID_OBJECTIVE_REF", lesson_path, None,
                    "INVALID_LOCAL_ASSESSMENT_OBJECTIVE_REF",
                ))
                continue

            units = lesson.get("units")
            if not isinstance(units, list) or not units:
                issues.append(AssessmentPlanIssue(
                    "ASSESSMENT_PLAN_NO_SAFE_ANCHOR", lesson_path, None,
                    "NO_LESSON_UNITS",
                ))
                continue
            flat: list[tuple[int, str, int, dict[str, Any], int, dict[str, Any], set[str]]] = []
            checks: list[dict[str, Any]] = []
            position = 0
            for unit_index, unit in enumerate(units, start=1):
                if not isinstance(unit, dict):
                    continue
                unit_path = _safe_path(chapter_index, lesson_index, unit_index)
                unit_refs = _text_ids(unit.get("learning_objective_refs"))
                blocks = unit.get("learning_blocks")
                if not isinstance(blocks, list):
                    continue
                for block_index, block in enumerate(blocks):
                    if not isinstance(block, dict):
                        continue
                    position += 1
                    flat.append((position, unit_path, unit_index, unit, block_index, block, unit_refs))
                    if str(block.get("intent") or "").strip() == "knowledge_check":
                        checks.append(block)
            if checks:
                existing_checks += len(checks)
                continue

            selected: list[AssessmentTeachingCandidate] = []
            for objective_ref in sorted(required_refs):
                fully: list[AssessmentTeachingCandidate] = []
                semantic: list[AssessmentTeachingCandidate] = []
                for _position, unit_path, unit_index, _unit, block_index, block, unit_refs in flat:
                    item = _candidate_from_block(
                        lesson_path=lesson_path,
                        unit_path=unit_path,
                        unit_index=unit_index,
                        block=block,
                        block_index=block_index,
                        objective_ref=objective_ref,
                        unit_objective_refs=unit_refs,
                    )
                    if item is None:
                        continue
                    (fully if item.alignment == "fully_aligned" else semantic).append(item)

                selected_address = selections.get((lesson_path, objective_ref))
                if selected_address is not None:
                    all_candidates = fully + semantic
                    approved = [
                        item for item in all_candidates
                        if (item.unit_path, item.block_id) == selected_address
                    ]
                    if len(approved) != 1:
                        issues.append(AssessmentPlanIssue(
                            "ASSESSMENT_PLAN_INVALID_SELECTION", lesson_path, objective_ref,
                            "SELECTION_OUTSIDE_SERVER_CANDIDATE_SET", len(all_candidates),
                        ))
                    else:
                        selected.append(approved[0])
                    continue
                # A fully aligned anchor is stronger deterministic evidence
                # than a merely scope-compatible candidate.  The latter must
                # not manufacture ambiguity or trigger a model call.
                if len(fully) == 1:
                    selected.append(fully[0])
                elif not fully and len(semantic) == 1:
                    unresolved[(lesson_path, objective_ref)] = tuple(semantic)
                elif fully or semantic:
                    unresolved[(lesson_path, objective_ref)] = tuple(fully + semantic)
                else:
                    issues.append(AssessmentPlanIssue(
                        "ASSESSMENT_PLAN_NO_SAFE_ANCHOR", lesson_path, objective_ref,
                        "NO_SAFE_TEACHING_ANCHOR", 0,
                    ))

            if any(issue.lesson_path == lesson_path for issue in issues) or any(
                key[0] == lesson_path for key in unresolved
            ):
                continue

            # The Node planner currently permits one primary interaction per
            # unit. Grouping only objectives whose exact anchor is the same
            # preserves that downstream contract without forcing artificial
            # component variety or cross-unit provenance changes.
            groups = _group_insertions(selected)
            unit_interactions: dict[str, int] = {}
            for unit_path, block_id in groups:
                unit_interactions[unit_path] = unit_interactions.get(unit_path, 0) + 1
                if unit_interactions[unit_path] > 1:
                    refs = tuple(sorted(item.objective_ref for item in groups[(unit_path, block_id)]))
                    issues.append(AssessmentPlanIssue(
                        "ASSESSMENT_PLAN_DOWNSTREAM_CAPABILITY_GAP", lesson_path, refs[0] if refs else None,
                        "MULTIPLE_ASSESSMENT_INTERACTIONS_IN_ONE_UNIT", len(groups[(unit_path, block_id)]),
                    ))
            if any(issue.lesson_path == lesson_path for issue in issues):
                continue

            for (unit_path, block_id), group in groups.items():
                unit = next(
                    unit for _position, path, _index, unit, _block_index, block, _refs in flat
                    if path == unit_path and str(block.get("id") or "").strip() == block_id
                )
                blocks = unit.get("learning_blocks")
                if not isinstance(blocks, list):
                    raise ValueError("V5 assessment compilation lost a validated learning-block list")
                teaching = next(block for block in blocks if str(block.get("id") or "").strip() == block_id)
                refs = tuple(sorted(item.objective_ref for item in group))
                new_block = {
                    "id": _server_block_id(unit, refs),
                    "intent": "knowledge_check",
                    "importance": "assessment",
                    "concept_ids": list(teaching.get("concept_ids") or []),
                    "primary_concept_ids": [],
                    "primary_evidence_scope_ids": [],
                    "supporting_evidence_scope_ids": sorted(_text_ids(teaching.get("primary_evidence_scope_ids"))),
                    "source_refs": list(teaching.get("source_refs") or []),
                    "learning_objective_refs": list(refs),
                    "content": {},
                }
                blocks.insert(_insert_after(blocks, block_id), new_block)
                deterministic_insertions += 1

    if issues:
        return AssessmentPlanCompilation(
            "TERMINAL_GAP", candidate, issues, unresolved,
            deterministic_insertions, existing_checks,
        )
    if unresolved:
        resolution_issues = [
            AssessmentPlanIssue(
                "ASSESSMENT_SEMANTIC_SELECTION_REQUIRED", lesson_path, objective_ref,
                "AMBIGUOUS_OR_SEMANTIC_TEACHING_ANCHOR", len(options),
            )
            for (lesson_path, objective_ref), options in sorted(unresolved.items())
        ]
        return AssessmentPlanCompilation(
            "NEEDS_SEMANTIC_RESOLUTION", candidate, resolution_issues, unresolved,
            deterministic_insertions, existing_checks,
        )
    return AssessmentPlanCompilation(
        "READY", candidate, [], {}, deterministic_insertions, existing_checks,
    )
