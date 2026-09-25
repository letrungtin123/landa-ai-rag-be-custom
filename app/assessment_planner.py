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
from app.component_capabilities import component_capabilities


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
    target_path: str | None = None
    knowledge_check_block_id: str | None = None
    candidate_rejection_counts: dict[str, int] = field(default_factory=dict)
    capacity_details: dict[str, int] = field(default_factory=dict)

    def workflow_issue(self) -> dict[str, Any]:
        issue: dict[str, Any] = {
            "code": self.code,
            "severity": "error",
            "path": self.lesson_path,
            "repairable": self.code in {
                "ASSESSMENT_SEMANTIC_SELECTION_REQUIRED",
                "ASSESSMENT_EXISTING_CHECK_ALIGNMENT_REQUIRED",
            },
            "safe_reason": self.safe_reason,
            "assessment_candidate_count": self.candidate_count,
        }
        if self.objective_ref:
            issue["objective_ids"] = [self.objective_ref]
        if self.target_path:
            issue["path"] = self.target_path
        if self.knowledge_check_block_id:
            issue["learning_block_ids"] = [self.knowledge_check_block_id]
        if self.candidate_rejection_counts:
            issue["assessment_candidate_rejection_counts"] = dict(self.candidate_rejection_counts)
        issue.update(self.capacity_details)
        return issue


@dataclass(frozen=True)
class AssessmentPlanCompilation:
    status: AssessmentPlanStatus
    blueprint: dict[str, Any]
    issues: list[AssessmentPlanIssue] = field(default_factory=list)
    candidates: dict[tuple[str, str], tuple[AssessmentTeachingCandidate, ...]] = field(default_factory=dict)
    deterministic_insertions: int = 0
    existing_checks: int = 0
    objective_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    objective_diagnostic_count: int = 0

    @property
    def metrics(self) -> dict[str, int | str]:
        return {
            "assessment_plan_status": self.status,
            "assessment_plan_deterministic_insertions": self.deterministic_insertions,
            "assessment_plan_proposed_insertions": self.deterministic_insertions,
            "assessment_plan_committed_insertions": 0 if self.status == "TERMINAL_GAP" else self.deterministic_insertions,
            "assessment_plan_existing_checks": self.existing_checks,
            "assessment_plan_semantic_resolution_objectives": len(self.candidates),
            "assessment_plan_terminal_gaps": sum(
                1 for issue in self.issues
                if issue.code not in {
                    "ASSESSMENT_SEMANTIC_SELECTION_REQUIRED",
                    "ASSESSMENT_EXISTING_CHECK_ALIGNMENT_REQUIRED",
                }
            ),
        }

    def safe_unit_diagnostics(self, baseline: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Bounded identity/count metadata; never instructional text/evidence."""
        result: list[dict[str, Any]] = []
        capabilities = component_capabilities(self.blueprint.get("component_capabilities"))
        for ci, chapter in enumerate(self.blueprint.get("chapters", []), 1):
            for li, lesson in enumerate(chapter.get("lessons", []), 1):
                if lesson.get("assessment_required") is not True:
                    continue
                required = _text_ids(lesson.get("assessment_objective_refs")) & _local_objective_ids(lesson)
                # Metadata coverage is not proof of grounded teaching alignment.
                check_refs = set().union(*(
                    _text_ids(block.get("learning_objective_refs"))
                    for unit in lesson.get("units", [])
                    for block in unit.get("learning_blocks", [])
                    if block.get("intent") == "knowledge_check"
                )) & required
                for ui, unit in enumerate(lesson.get("units", []), 1):
                    blocks = unit.get("learning_blocks") or []
                    checks = [b for b in blocks if b.get("intent") == "knowledge_check"]
                    original_blocks = []
                    if baseline is not None:
                        original_blocks = baseline['chapters'][ci - 1]['lessons'][li - 1]['units'][ui - 1].get('learning_blocks') or []
                    original_checks = [b for b in original_blocks if b.get('intent') == 'knowledge_check']
                    original_ids = {str(b.get('id') or '') for b in original_checks}
                    proposed_count = sum(str(b.get('id') or '') not in original_ids for b in checks)
                    result.append({
                        "lesson_path": _safe_path(ci, li), "unit_path": _safe_path(ci, li, ui),
                        "capability_version": 2 if capabilities else 1,
                        "assessment_capacity": capabilities.max_assessments_per_unit if capabilities else 1,
                        "required_objective_ids": sorted(_text_ids(lesson.get("assessment_objective_refs")))[:12],
                        "declared_assessed_objective_ids": sorted(check_refs)[:12],
                        "missing_assessment_objective_ids": sorted(required - check_refs)[:12],
                        "missing_assessment_objective_count": len(required - check_refs),
                        "candidate_check_ids": [str(b.get("id") or "")[:96] for b in checks][:12],
                        "candidate_check_count": len(checks),
                        **({
                            "existing_check_count": len(original_checks),
                            "proposed_assessment_count": proposed_count,
                            "committed_assessment_count": 0 if self.status == 'TERMINAL_GAP' else proposed_count,
                            "available_assessment_capacity": max(0, (capabilities.max_assessments_per_unit if capabilities else 1) - len(original_checks)),
                        } if baseline is not None else {}),
                        "primary_teaching_block_ids": [str(b.get("id") or "")[:96] for b in blocks if b.get("intent") in TEACHING_INTENTS and b.get("primary_evidence_scope_ids")][:12],
                    })
        return result


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


@dataclass(frozen=True)
class AssessmentTeachingEligibility:
    """One deterministic assessment-anchor decision without source content.

    ``base_eligible`` intentionally does not require the teaching block to
    already list the objective.  A semantic-delta repair may add that local
    association, but may never bypass lesson/order/provenance compatibility.
    """

    base_eligible: bool
    fully_aligned: bool
    reasons: tuple[str, ...]


def _record_anchor_diagnostic(
    diagnostic: dict[str, Any], *, unit_path: str, block_index: int,
    block: dict[str, Any], eligibility: AssessmentTeachingEligibility,
) -> None:
    """Bounded structural metadata only; hash provider IDs, never log content."""
    diagnostic["evaluated_block_count"] += 1
    if eligibility.base_eligible:
        diagnostic["base_eligible_count"] += 1
    if eligibility.fully_aligned:
        diagnostic["fully_aligned_count"] += 1
    if not eligibility.base_eligible:
        for reason in eligibility.reasons:
            counts = diagnostic["rejection_counts"]
            counts[reason] = counts.get(reason, 0) + 1
    if len(diagnostic["candidates"]) < 12:
        diagnostic["candidates"].append({
            "block_path": f"{unit_path}.block_{block_index + 1}",
            "block_id_hash": hashlib.sha256(str(block.get("id") or "").encode()).hexdigest()[:16],
            "base_eligible": eligibility.base_eligible,
            "fully_aligned": eligibility.fully_aligned,
            "reasons": list(eligibility.reasons),
        })
    else:
        diagnostic["omitted_candidate_count"] += 1


def evaluate_assessment_teaching_anchor(
    *,
    teaching_block: dict[str, Any],
    knowledge_check_block: dict[str, Any] | None,
    objective_refs: set[str],
    unit_objective_refs: set[str],
    precedes_check: bool = True,
) -> AssessmentTeachingEligibility:
    """Evaluate the one authoritative V5 teach→check eligibility contract.

    Reasons are bounded enums only.  They are suitable for correlated logs
    and tests, never for source or prompt disclosure.
    """

    reasons: list[str] = []
    if not objective_refs:
        reasons.append("OBJECTIVE_NOT_LOCAL")
    if not precedes_check:
        reasons.append("NOT_PRECEDING")
    if str(teaching_block.get("intent") or "").strip() not in TEACHING_INTENTS:
        reasons.append("INTENT_NOT_TEACHING")
    if not _text_ids(teaching_block.get("primary_evidence_scope_ids")):
        reasons.append("NO_PRIMARY_EVIDENCE")
    if not str(teaching_block.get("id") or "").strip():
        reasons.append("AMBIGUOUS_BLOCK_ADDRESS")

    teaching_objectives = _text_ids(teaching_block.get("learning_objective_refs"))
    if not objective_refs.issubset(teaching_objectives | unit_objective_refs):
        reasons.append("OBJECTIVE_OUTSIDE_UNIT_SCOPE")

    if knowledge_check_block is not None:
        teaching_concepts = _text_ids(teaching_block.get("concept_ids"))
        check_concepts = _text_ids(knowledge_check_block.get("concept_ids"))
        if not teaching_concepts or not check_concepts or not teaching_concepts.intersection(check_concepts):
            reasons.append("CONCEPT_MISMATCH")
        teaching_refs = _text_ids(teaching_block.get("source_refs"))
        check_refs = _text_ids(knowledge_check_block.get("source_refs"))
        if teaching_refs and check_refs and not teaching_refs.intersection(check_refs):
            reasons.append("SOURCE_REF_MISMATCH")

    base_eligible = not reasons
    fully_aligned = base_eligible and objective_refs.issubset(teaching_objectives)
    if base_eligible and not fully_aligned:
        reasons.append("OBJECTIVE_LINK_MISSING")
    return AssessmentTeachingEligibility(base_eligible, fully_aligned, tuple(reasons))


def _candidate_from_block(
    *,
    lesson_path: str,
    unit_path: str,
    unit_index: int,
    block: dict[str, Any],
    block_index: int,
    objective_ref: str,
    unit_objective_refs: set[str],
    knowledge_check_block: dict[str, Any] | None = None,
    precedes_check: bool = True,
) -> AssessmentTeachingCandidate | None:
    eligibility = evaluate_assessment_teaching_anchor(
        teaching_block=block,
        knowledge_check_block=knowledge_check_block,
        objective_refs={objective_ref},
        unit_objective_refs=unit_objective_refs,
        precedes_check=precedes_check,
    )
    if not eligibility.base_eligible:
        return None
    primary_scope_ids = tuple(sorted(_text_ids(block.get("primary_evidence_scope_ids"))))
    alignment: Literal["fully_aligned", "semantic_candidate"] = (
        "fully_aligned" if eligibility.fully_aligned else "semantic_candidate"
    )
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
    capabilities = component_capabilities(candidate.get("component_capabilities"))
    selections = selections or {}
    issues: list[AssessmentPlanIssue] = []
    unresolved: dict[tuple[str, str], tuple[AssessmentTeachingCandidate, ...]] = {}
    deterministic_insertions = existing_checks = 0
    objective_diagnostics: list[dict[str, Any]] = []
    objective_diagnostic_count = 0

    def begin_diagnostic(lesson_path: str, objective_ref: str, *, check_path: str | None = None) -> dict[str, Any]:
        nonlocal objective_diagnostic_count
        objective_diagnostic_count += 1
        diagnostic: dict[str, Any] = {
            "lesson_path": lesson_path, "objective_id": objective_ref,
            "assessment_state": "EXISTING_CHECK" if check_path else "MISSING_CHECK",
            "check_path": check_path,
            "evaluated_block_count": 0, "base_eligible_count": 0,
            "fully_aligned_count": 0, "rejection_counts": {},
            "candidates": [], "omitted_candidate_count": 0,
        }
        if len(objective_diagnostics) < 48:
            objective_diagnostics.append(diagnostic)
        return diagnostic

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

            if capabilities and not capabilities.assessment_enabled:
                issues.append(AssessmentPlanIssue(
                    "ASSESSMENT_PLAN_DOWNSTREAM_CAPABILITY_GAP", lesson_path, None,
                    "TENANT_ASSESSMENT_DISABLED",
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
            checks: list[tuple[int, str, int, dict[str, Any], int, dict[str, Any], set[str]]] = []
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
                        checks.append((position, unit_path, unit_index, unit, block_index, block, unit_refs))
            if checks:
                existing_checks += len(checks)
                if capabilities:
                    for unit_path in sorted({entry[1] for entry in checks}):
                        count = sum(entry[1] == unit_path for entry in checks)
                        if count > capabilities.max_assessments_per_unit:
                            issues.append(AssessmentPlanIssue(
                                "ASSESSMENT_PLAN_DOWNSTREAM_CAPABILITY_GAP", lesson_path, None,
                                "ASSESSMENT_INSTANCE_CAPACITY_EXCEEDED", target_path=unit_path,
                                capacity_details={"existing_check_count": count, "proposed_assessment_count": 0,
                                                  "assessment_capacity": capabilities.max_assessments_per_unit},
                            ))
                check_objectives = set().union(*(
                    _text_ids(block.get("learning_objective_refs"))
                    for _position, _path, _index, _unit, _block_index, block, _refs in checks
                )) & required_refs
                for (
                    check_position,
                    check_unit_path,
                    _check_unit_index,
                    _check_unit,
                    _check_block_index,
                    check_block,
                    _check_unit_refs,
                ) in checks:
                    check_id = str(check_block.get("id") or "").strip()
                    check_refs = _text_ids(check_block.get("learning_objective_refs")) & required_refs
                    for objective_ref in sorted(check_refs):
                        fully: list[AssessmentTeachingCandidate] = []
                        semantic: list[AssessmentTeachingCandidate] = []
                        rejection_counts: dict[str, int] = {}
                        diagnostic = begin_diagnostic(
                            lesson_path, objective_ref,
                            check_path=f"{check_unit_path}.block_{_check_block_index + 1}",
                        )
                        for (
                            teaching_position,
                            unit_path,
                            unit_index,
                            _unit,
                            block_index,
                            block,
                            unit_refs,
                        ) in flat:
                            eligibility = evaluate_assessment_teaching_anchor(
                                teaching_block=block,
                                knowledge_check_block=check_block,
                                objective_refs={objective_ref},
                                unit_objective_refs=unit_refs,
                                precedes_check=teaching_position < check_position,
                            )
                            _record_anchor_diagnostic(
                                diagnostic, unit_path=unit_path, block_index=block_index,
                                block=block, eligibility=eligibility,
                            )
                            if not eligibility.base_eligible:
                                for reason in eligibility.reasons:
                                    # OBJECTIVE_LINK_MISSING is explanatory for
                                    # a base-eligible candidate, never a reject.
                                    if reason != "OBJECTIVE_LINK_MISSING":
                                        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                                continue
                            item = _candidate_from_block(
                                lesson_path=lesson_path,
                                unit_path=unit_path,
                                unit_index=unit_index,
                                block=block,
                                block_index=block_index,
                                objective_ref=objective_ref,
                                unit_objective_refs=unit_refs,
                                knowledge_check_block=check_block,
                                precedes_check=teaching_position < check_position,
                            )
                            if item is not None:
                                (fully if item.alignment == "fully_aligned" else semantic).append(item)

                        supporting = _text_ids(check_block.get("supporting_evidence_scope_ids"))
                        grounded = bool(supporting.intersection(set().union(*(
                            set(item.primary_evidence_scope_ids) for item in fully
                        )))) if fully else False
                        diagnostic["grounded"] = grounded
                        if fully and grounded:
                            continue
                        if fully or semantic:
                            issues.append(AssessmentPlanIssue(
                                "ASSESSMENT_EXISTING_CHECK_ALIGNMENT_REQUIRED",
                                lesson_path,
                                objective_ref,
                                "EXISTING_CHECK_REQUIRES_OBJECTIVE_OR_GROUNDING_RECONCILIATION",
                                len(fully) + len(semantic),
                                target_path=check_unit_path,
                                knowledge_check_block_id=check_id or None,
                            ))
                        else:
                            issues.append(AssessmentPlanIssue(
                                "ASSESSMENT_PLAN_NO_SAFE_ANCHOR",
                                lesson_path,
                                objective_ref,
                                "NO_SAFE_TEACHING_ANCHOR_FOR_EXISTING_CHECK",
                                0,
                                target_path=check_unit_path,
                                knowledge_check_block_id=check_id or None,
                                candidate_rejection_counts=dict(sorted(rejection_counts.items())),
                            ))

                # Existing checks must not suppress another required objective.
                # A valid missing objective still uses the normal server-owned
                # per-objective insertion path below.
                required_refs = required_refs - check_objectives
                # Inspect missing objectives even when an existing check needs
                # alignment. Do not hide a terminal teaching gap behind a
                # repairable finding. The per-lesson commit guard below still
                # prohibits insertion while ANY issue remains unresolved.
                if not required_refs:
                    continue

            selected: list[AssessmentTeachingCandidate] = []
            for objective_ref in sorted(required_refs):
                fully: list[AssessmentTeachingCandidate] = []
                semantic: list[AssessmentTeachingCandidate] = []
                diagnostic = begin_diagnostic(lesson_path, objective_ref)
                for _position, unit_path, unit_index, _unit, block_index, block, unit_refs in flat:
                    eligibility = evaluate_assessment_teaching_anchor(
                        teaching_block=block, knowledge_check_block=None,
                        objective_refs={objective_ref}, unit_objective_refs=unit_refs,
                    )
                    _record_anchor_diagnostic(
                        diagnostic, unit_path=unit_path, block_index=block_index,
                        block=block, eligibility=eligibility,
                    )
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
                        candidate_rejection_counts=dict(sorted(diagnostic["rejection_counts"].items())),
                    ))

            if any(issue.lesson_path == lesson_path for issue in issues) or any(
                key[0] == lesson_path for key in unresolved
            ):
                continue

            # Group only exact anchors. Legacy retains one group/unit; the
            # Node-owned profile opts into bounded distinct instances without
            # moving provenance or changing instructional topology.
            groups = _group_insertions(selected)
            for unit_path in sorted({path for path, _block_id in groups}):
                proposed = sum(path == unit_path for path, _block_id in groups)
                existing = sum(entry[1] == unit_path for entry in checks) if capabilities else 0
                capacity = capabilities.max_assessments_per_unit if capabilities else 1
                if existing + proposed > capacity:
                    issues.append(AssessmentPlanIssue(
                        "ASSESSMENT_PLAN_DOWNSTREAM_CAPABILITY_GAP", lesson_path, None,
                        "ASSESSMENT_INSTANCE_CAPACITY_EXCEEDED" if capabilities else "MULTIPLE_ASSESSMENT_INTERACTIONS_IN_ONE_UNIT",
                        target_path=unit_path,
                        capacity_details={"anchor_group_count": proposed, "existing_check_count": existing,
                                          "proposed_assessment_count": proposed, "assessment_capacity": capacity},
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
                # Only an explicit, validated semantic selection may turn a
                # base-eligible anchor into a fully aligned one. Without this
                # link the inserted check immediately fails the next compile.
                # Unselected candidates never reach this commit branch.
                selected_semantic_refs = {
                    item.objective_ref for item in group
                    if item.alignment == "semantic_candidate"
                    and selections.get((lesson_path, item.objective_ref)) == (unit_path, block_id)
                }
                if selected_semantic_refs:
                    teaching["learning_objective_refs"] = sorted(
                        _text_ids(teaching.get("learning_objective_refs")) | selected_semantic_refs
                    )
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

    terminal_issues = [
        issue for issue in issues
        if issue.code not in {
            "ASSESSMENT_SEMANTIC_SELECTION_REQUIRED",
            "ASSESSMENT_EXISTING_CHECK_ALIGNMENT_REQUIRED",
        }
    ]
    if terminal_issues:
        return AssessmentPlanCompilation(
            "TERMINAL_GAP", candidate, issues, unresolved,
            deterministic_insertions, existing_checks,
            objective_diagnostics, objective_diagnostic_count,
        )
    if unresolved or issues:
        resolution_issues = [
            AssessmentPlanIssue(
                "ASSESSMENT_SEMANTIC_SELECTION_REQUIRED", lesson_path, objective_ref,
                "AMBIGUOUS_OR_SEMANTIC_TEACHING_ANCHOR", len(options),
            )
            for (lesson_path, objective_ref), options in sorted(unresolved.items())
        ]
        return AssessmentPlanCompilation(
            "NEEDS_SEMANTIC_RESOLUTION", candidate, [*issues, *resolution_issues], unresolved,
            deterministic_insertions, existing_checks,
            objective_diagnostics, objective_diagnostic_count,
        )
    return AssessmentPlanCompilation(
        "READY", candidate, [], {}, deterministic_insertions, existing_checks,
        objective_diagnostics, objective_diagnostic_count,
    )
