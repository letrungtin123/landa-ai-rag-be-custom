"""Lesson-generation workflow validation and repair patches."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Literal

from app.lesson_prompt_policy import lesson_output_language_policy
from app.services.lesson_author.architecture_shape import _workflow_issue
from app.services.lesson_author.errors import LessonAuthorProposalValidationError
from app.services.lesson_author.proposal_validation import validate_lesson_author_proposal_shape
from app.services.lesson_author.repair.guards import _repair_keeps_existing_source_scope
from app.services.lesson_author.source_context import MAX_WORKFLOW_REPAIR_TARGET_CHARS
from app.services.lesson_author.source_refs import validate_lesson_author_proposal_source_refs
from app.services.retrieval.source_coverage import source_coverage_metrics, validate_lesson_author_source_coverage
from app.workflows.contracts import (
    RepairTarget,
    WorkflowFailure,
    WorkflowIssue,
    WorkflowValidationResult,
    safe_workflow_path,
)


def _proposal_path_object(proposal: dict[str, Any], path: str) -> dict[str, Any] | None:
    match = re.fullmatch(r"lesson|chapter_(\d+)\.lesson_(\d+)(?:\.unit_(\d+)(?:\.component_(\d+))?)?", path)
    if match is None:
        return None
    if path == "lesson":
        return proposal
    chapters = proposal.get("chapters") if isinstance(proposal.get("chapters"), list) else []
    chapter_index, lesson_index = int(match.group(1)) - 1, int(match.group(2)) - 1
    if not 0 <= chapter_index < len(chapters) or not isinstance(chapters[chapter_index], dict):
        return None
    lessons = chapters[chapter_index].get("lessons") if isinstance(chapters[chapter_index].get("lessons"), list) else []
    if not 0 <= lesson_index < len(lessons) or not isinstance(lessons[lesson_index], dict):
        return None
    node: dict[str, Any] = lessons[lesson_index]
    if match.group(3) is None:
        return node
    units = node.get("units") if isinstance(node.get("units"), list) else []
    unit_index = int(match.group(3)) - 1
    if not 0 <= unit_index < len(units) or not isinstance(units[unit_index], dict):
        return None
    node = units[unit_index]
    if match.group(4) is None:
        return node
    components = node.get("components") if isinstance(node.get("components"), list) else node.get("blocks")
    component_index = int(match.group(4)) - 1
    return components[component_index] if isinstance(components, list) and 0 <= component_index < len(components) and isinstance(components[component_index], dict) else None


def validate_lesson_generation_workflow(
    proposal: dict[str, Any],
    source_coverage_manifest: dict[str, Any] | None,
    known_source_refs: set[str],
) -> WorkflowValidationResult:
    issues: list[WorkflowIssue] = []
    try:
        validate_lesson_author_proposal_shape(proposal)
    except LessonAuthorProposalValidationError as error:
        issues.append(_workflow_issue("LESSON_VALIDATION_FAILED", str(error), path="lesson"))
    try:
        validate_lesson_author_proposal_source_refs(proposal, known_source_refs)
    except LessonAuthorProposalValidationError as error:
        issues.append(_workflow_issue("INVALID_SOURCE_REF", str(error), path="lesson"))
    try:
        coverage = validate_lesson_author_source_coverage(proposal, source_coverage_manifest)
    except LessonAuthorProposalValidationError as error:
        coverage = source_coverage_metrics(proposal, source_coverage_manifest)
        issues.append(_workflow_issue("SOURCE_EVIDENCE_INSUFFICIENT", str(error), path="lesson"))
    return WorkflowValidationResult(issues, {"source_coverage": coverage.get("coverage_ratio")})


def build_lesson_generation_repair_prompt(
    *,
    proposal: dict[str, Any],
    targets: list[RepairTarget],
    evidence_context: str,
    locale: Literal["vi", "en"],
) -> str:
    snapshots = []
    for target in targets:
        node = _proposal_path_object(proposal, target["path"])
        if node is None:
            raise WorkflowFailure(
                "LESSON_REPAIR_INVALID",
                "A requested lesson repair path does not exist.",
                internal_code="LESSON_REPAIR_TARGET_MISSING",
                failure_stage="lesson_repair_target_snapshot",
                diagnostics={"repair_target_path": safe_workflow_path(target["path"])},
            )
        snapshots.append({
            "path": target["path"], "scope": target["scope"], "codes": target["codes"],
            "allowed_fields": target["allowed_fields"], "current": node,
        })
    serialized_targets = json.dumps(snapshots, ensure_ascii=False, separators=(",", ":"))
    if len(serialized_targets) > MAX_WORKFLOW_REPAIR_TARGET_CHARS:
        raise WorkflowFailure(
            "LESSON_REPAIR_SCOPE_TOO_LARGE",
            "The affected lesson scope is too large for a bounded repair prompt.",
            internal_code="LESSON_REPAIR_SCOPE_TOO_LARGE",
            failure_stage="lesson_repair_target_snapshot",
            diagnostics={"repair_target_count": len(targets)},
        )
    language = "Vietnamese" if locale == "vi" else "English"
    return "\n".join([
        "Repair only the listed parts of this source-grounded lesson proposal.",
        f"Write in {language}; return JSON only and do not include reasoning.",
        lesson_output_language_policy(locale),
        "Return {\"patches\":[{\"path\":\"...\",\"replacement\":{...}}]}. Every patch must match a listed path and only use allowed_fields. Do not change course hierarchy, source scope, assets, or unrelated components.",
        "SOURCE EVIDENCE (bounded to the approved lesson scope):",
        evidence_context,
        "REPAIR TARGETS:",
        serialized_targets,
    ])


def apply_lesson_generation_repair_patches(
    proposal: dict[str, Any],
    targets: list[RepairTarget],
    payload: dict[str, Any],
) -> dict[str, Any]:
    patches = payload.get("patches") if isinstance(payload.get("patches"), list) else []
    expected = {target["path"]: target for target in targets}
    result = deepcopy(proposal)
    seen: set[str] = set()
    for patch in patches:
        if not isinstance(patch, dict):
            raise WorkflowFailure(
                "LESSON_REPAIR_INVALID",
                "Repair response contains an invalid patch.",
                internal_code="LESSON_REPAIR_PATCH_INVALID",
                failure_stage="lesson_repair_patch_contract",
                diagnostics={"patch_count": len(patches)},
            )
        path = str(patch.get("path") or "")
        target = expected.get(path)
        replacement = patch.get("replacement")
        if target is None or path in seen:
            raise WorkflowFailure(
                "LESSON_REPAIR_INVALID",
                "Repair response tried to change an unapproved scope.",
                internal_code="LESSON_REPAIR_TARGET_OUT_OF_SCOPE",
                failure_stage="lesson_repair_target_whitelist",
                diagnostics={"repair_target_path": safe_workflow_path(path), "patch_count": len(patches)},
            )
        if not isinstance(replacement, dict):
            raise WorkflowFailure(
                "LESSON_REPAIR_INVALID",
                "Repair response contains an invalid replacement patch.",
                internal_code="LESSON_REPAIR_PATCH_INVALID",
                failure_stage="lesson_repair_patch_contract",
                diagnostics={"repair_target_path": safe_workflow_path(path), "patch_count": len(patches)},
            )
        if set(replacement) - set(target["allowed_fields"]):
            raise WorkflowFailure(
                "LESSON_REPAIR_INVALID",
                "Repair response tried to change fields outside its allowed scope.",
                internal_code="LESSON_REPAIR_FIELD_OUT_OF_SCOPE",
                failure_stage="lesson_repair_field_whitelist",
                diagnostics={"repair_target_path": safe_workflow_path(path), "patch_count": len(patches)},
            )
        original = _proposal_path_object(proposal, path)
        if original is None:
            raise WorkflowFailure(
                "LESSON_REPAIR_INVALID",
                "Repair target no longer exists.",
                internal_code="LESSON_REPAIR_TARGET_MISSING",
                failure_stage="lesson_repair_patch_apply",
                diagnostics={"repair_target_path": safe_workflow_path(path), "patch_count": len(patches)},
            )
        if not _repair_keeps_existing_source_scope(original, replacement):
            raise WorkflowFailure(
                "LESSON_REPAIR_INVALID",
                "Repair response tried to expand the approved source scope.",
                internal_code="LESSON_REPAIR_SOURCE_SCOPE_EXPANSION",
                failure_stage="lesson_repair_source_scope_guard",
                diagnostics={"repair_target_path": safe_workflow_path(path), "patch_count": len(patches)},
            )
        node = _proposal_path_object(result, path)
        if node is None:
            raise WorkflowFailure(
                "LESSON_REPAIR_INVALID",
                "Repair target no longer exists.",
                internal_code="LESSON_REPAIR_PATCH_APPLY_FAILED",
                failure_stage="lesson_repair_patch_apply",
                diagnostics={"repair_target_path": safe_workflow_path(path), "patch_count": len(patches)},
            )
        node.update(replacement)
        seen.add(path)
    if seen != set(expected):
        raise WorkflowFailure(
            "LESSON_REPAIR_INVALID",
            "Repair response did not repair every required target.",
            internal_code="LESSON_REPAIR_TARGET_MISSING",
            failure_stage="lesson_repair_target_whitelist",
            diagnostics={"patch_count": len(patches), "expected_target_count": len(expected), "applied_target_count": len(seen)},
        )
    return result
