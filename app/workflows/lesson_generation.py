from __future__ import annotations

"""Bounded self-built-RAG lesson/proposal orchestration.

The graph deliberately receives opaque callback functions. Source rows and
model prompts stay request-local in the endpoint closure rather than becoming
serializable graph state or a second authoritative component registry.
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Awaitable, Callable, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from .contracts import (
    RepairTarget,
    WorkflowFailure,
    WorkflowGenerationResult,
    WorkflowIssue,
    WorkflowValidationResult,
    append_progress,
    node_duration_update,
    progress_event,
    safe_workflow_issue_summary,
    safe_workflow_path,
    sanitized_workflow_diagnostics,
)


class LessonWorkflowState(TypedDict, total=False):
    request_context: dict[str, Any]
    evidence_summary: dict[str, Any]
    proposal: dict[str, Any]
    validation_issues: list[WorkflowIssue]
    repair_targets: list[RepairTarget]
    repair_attempts: int
    max_repair_attempts: int
    metrics: dict[str, Any]
    progress: list[dict[str, Any]]
    errors: list[WorkflowIssue]
    failure_code: str
    failure_internal_code: str
    failure_stage: str
    failure_repair_pass_number: int
    status: Literal["running", "ready", "failed"]
    usage: dict[str, Any]
    node_durations_ms: dict[str, int]


@dataclass(frozen=True)
class LessonGenerationWorkflowCallbacks:
    validate_contract: Callable[[], list[WorkflowIssue]]
    retrieve_evidence: Callable[[], Awaitable[dict[str, Any]]]
    validate_evidence: Callable[[dict[str, Any]], WorkflowValidationResult]
    generate_proposal: Callable[[], Awaitable[WorkflowGenerationResult]]
    validate_content: Callable[[dict[str, Any]], WorkflowValidationResult]
    validate_pedagogy: Callable[[dict[str, Any]], WorkflowValidationResult]
    validate_duplicates: Callable[[dict[str, Any]], WorkflowValidationResult]
    repair_content: Callable[[dict[str, Any], list[RepairTarget]], Awaitable[WorkflowGenerationResult]]
    # Endpoint-owned sink for metadata-only diagnostics. It is deliberately
    # optional so graph tests and non-HTTP callers keep the same behavior.
    emit_diagnostic: Callable[[dict[str, Any]], None] | None = None


NON_REPAIRABLE_CODES = {
    "SOURCE_SCOPE_INCOMPLETE",
    "SOURCE_EVIDENCE_INSUFFICIENT",
    "SOURCE_DOCUMENT_UNAVAILABLE",
    "SOURCE_OWNERSHIP_INVALID",
    "SOURCE_FACT_UNAVAILABLE",
    "TENANT_MISMATCH",
    "COMPONENT_NOT_PERMITTED",
    "ASSET_MISSING",
    "UNSUPPORTED_COMPONENT",
    "SECURITY_VALIDATION_FAILED",
}


def _lesson_scope(path: str) -> Literal["lesson", "unit", "component"]:
    if ".component_" in path:
        return "component"
    if ".unit_" in path:
        return "unit"
    return "lesson"


def _allowed_fields(scope: Literal["lesson", "unit", "component"]) -> list[str]:
    if scope == "component":
        return ["type", "title", "content", "metadata", "source_fact_ids"]
    if scope == "unit":
        return ["title", "source_refs", "source_fact_ids", "components"]
    return ["title", "source_refs", "units"]


def classify_lesson_repair_targets(issues: list[WorkflowIssue]) -> list[RepairTarget]:
    targets: dict[tuple[str, str], RepairTarget] = {}
    for issue in issues:
        if issue.get("severity") != "error" or str(issue.get("code") or "") in NON_REPAIRABLE_CODES:
            continue
        paths = [str(issue.get("path") or "lesson")]
        paths.extend(str(path) for path in issue.get("related_paths", []) if path)
        for path in paths:
            scope = _lesson_scope(path)
            key = (scope, path)
            target = targets.get(key)
            if target is None:
                target = {"scope": scope, "path": path, "codes": [], "allowed_fields": _allowed_fields(scope)}
                targets[key] = target
            code = str(issue.get("code") or "LESSON_VALIDATION_FAILED")
            if code not in target["codes"]:
                target["codes"].append(code)
    return list(targets.values())


def _blocking_issues(state: LessonWorkflowState) -> list[WorkflowIssue]:
    return [issue for issue in state.get("validation_issues", []) if issue.get("severity") == "error"]


def _failure_code(state: LessonWorkflowState) -> str:
    errors = list(state.get("errors", [])) + _blocking_issues(state)
    codes = {str(issue.get("code") or "") for issue in errors}
    if codes & NON_REPAIRABLE_CODES:
        return next(code for code in codes if code in NON_REPAIRABLE_CODES)
    if int(state.get("repair_attempts") or 0) >= int(state.get("max_repair_attempts") or 0):
        return "LESSON_REPAIR_EXHAUSTED"
    return str(state.get("failure_code") or "LESSON_VALIDATION_FAILED")


def build_lesson_generation_graph(callbacks: LessonGenerationWorkflowCallbacks):
    graph = StateGraph(LessonWorkflowState)

    def emit(
        state: LessonWorkflowState,
        stage: str,
        event: str,
        *,
        started_at: float | None = None,
        **metadata: Any,
    ) -> None:
        """Emit structural diagnostics only; never lesson/source/model text."""

        if callbacks.emit_diagnostic is None:
            return
        payload: dict[str, Any] = {
            "stage": stage,
            "event": event,
            "repair_pass_number": int(state.get("repair_attempts") or 0) + (1 if stage.startswith("lesson_repair") else 0),
            **metadata,
        }
        correlation_id = (state.get("request_context") or {}).get("correlation_id")
        if isinstance(correlation_id, str) and correlation_id:
            payload["correlation_id"] = correlation_id
        if started_at is not None:
            payload["duration_ms"] = max(0, round((perf_counter() - started_at) * 1000))
        callbacks.emit_diagnostic(payload)

    async def validate_contract(state: LessonWorkflowState) -> dict[str, Any]:
        started = perf_counter()
        issues = callbacks.validate_contract()
        emit(
            state, "lesson_contract_validation", "failed" if any(issue.get("severity") == "error" for issue in issues) else "passed",
            started_at=started,
            validation_finding_count=len(issues),
            findings=[safe_workflow_issue_summary(issue, repairable=str(issue.get("code") or "") not in NON_REPAIRABLE_CODES) for issue in issues[:32]],
        )
        return {
            "validation_issues": issues,
            "errors": [issue for issue in issues if issue.get("severity") == "error"],
            "progress": append_progress(state, progress_event("PLANNING_LESSON", "Planning lesson scope", "lesson_generation", current=1, total=9)),
            "node_durations_ms": node_duration_update(state, "validate_lesson_contract", started),
        }

    async def retrieve_evidence(state: LessonWorkflowState) -> dict[str, Any]:
        started = perf_counter()
        try:
            evidence = await callbacks.retrieve_evidence()
        except WorkflowFailure as error:
            issue: WorkflowIssue = {"code": error.code, "severity": "error", "message": error.message}
            emit(state, error.failure_stage or "lesson_evidence_retrieval", "failed", started_at=started,
                 internal_failure_code=error.internal_code, external_failure_code=error.code)
            return {"errors": [issue], "failure_code": error.code, "failure_internal_code": error.internal_code,
                    "failure_stage": error.failure_stage or "lesson_evidence_retrieval",
                    "node_durations_ms": node_duration_update(state, "retrieve_lesson_evidence", started)}
        return {
            "evidence_summary": evidence,
            "progress": append_progress(state, progress_event("RETRIEVING_EVIDENCE", "Retrieving lesson evidence", "lesson_generation", current=2, total=9)),
            "node_durations_ms": node_duration_update(state, "retrieve_lesson_evidence", started),
        }

    async def validate_evidence(state: LessonWorkflowState) -> dict[str, Any]:
        started = perf_counter()
        result = callbacks.validate_evidence(dict(state.get("evidence_summary") or {}))
        blocking = [issue for issue in result.issues if issue.get("severity") == "error"]
        emit(state, "lesson_evidence_validation", "failed" if blocking else "passed", started_at=started,
             validation_finding_count=len(result.issues), blocking_finding_count=len(blocking),
             findings=[safe_workflow_issue_summary(issue, repairable=str(issue.get("code") or "") not in NON_REPAIRABLE_CODES) for issue in result.issues[:32]])
        return {
            "validation_issues": result.issues,
            "metrics": result.metrics,
            "progress": append_progress(state, progress_event("VALIDATING_EVIDENCE", "Validating lesson evidence", "lesson_generation", current=3, total=9)),
            "node_durations_ms": node_duration_update(state, "validate_evidence", started),
        }

    async def generate_proposal(state: LessonWorkflowState) -> dict[str, Any]:
        started = perf_counter()
        try:
            generated = await callbacks.generate_proposal()
        except WorkflowFailure as error:
            issue: WorkflowIssue = {"code": error.code, "severity": "error", "message": error.message}
            emit(state, error.failure_stage or "lesson_generation", "failed", started_at=started,
                 internal_failure_code=error.internal_code, external_failure_code=error.code)
            return {"errors": [issue], "failure_code": error.code, "failure_internal_code": error.internal_code,
                    "failure_stage": error.failure_stage or "lesson_generation",
                    "node_durations_ms": node_duration_update(state, "generate_components", started)}
        return {
            "proposal": generated.value,
            "usage": generated.usage,
            "progress": append_progress(state, progress_event("GENERATING_COMPONENTS", "Generating selected lesson components", "lesson_generation", current=4, total=9)),
            "node_durations_ms": node_duration_update(state, "generate_components", started),
        }

    async def validate_content(state: LessonWorkflowState) -> dict[str, Any]:
        started = perf_counter()
        result = callbacks.validate_content(dict(state.get("proposal") or {}))
        blocking = [issue for issue in result.issues if issue.get("severity") == "error"]
        emit(state, "lesson_content_validation", "failed" if blocking else "passed", started_at=started,
             validation_finding_count=len(result.issues), blocking_finding_count=len(blocking),
             findings=[safe_workflow_issue_summary(issue, repairable=str(issue.get("code") or "") not in NON_REPAIRABLE_CODES) for issue in result.issues[:32]])
        return {
            "validation_issues": result.issues,
            "metrics": result.metrics,
            "progress": append_progress(state, progress_event("VALIDATING_CONTENT", "Validating generated lesson content", "lesson_generation", current=5, total=9)),
            "node_durations_ms": node_duration_update(state, "validate_content", started),
        }

    async def validate_pedagogy(state: LessonWorkflowState) -> dict[str, Any]:
        started = perf_counter()
        result = callbacks.validate_pedagogy(dict(state.get("proposal") or {}))
        blocking = [issue for issue in result.issues if issue.get("severity") == "error"]
        is_revalidation = int(state.get("repair_attempts") or 0) > 0
        emit(state, "lesson_pedagogical_revalidation" if is_revalidation else "lesson_pedagogical_validation",
             "failed" if blocking else "passed", started_at=started,
             validation_finding_count=len(result.issues), blocking_finding_count=len(blocking),
             **({"internal_failure_code": "LESSON_REPAIR_REVALIDATION_FAILED", "failure_stage": "lesson_pedagogical_revalidation"} if is_revalidation and blocking else {}),
             findings=[safe_workflow_issue_summary(issue, repairable=str(issue.get("code") or "") not in NON_REPAIRABLE_CODES) for issue in result.issues[:32]])
        return {
            "validation_issues": [*(state.get("validation_issues") or []), *result.issues],
            "metrics": {**dict(state.get("metrics") or {}), **result.metrics},
            **({"failure_internal_code": "LESSON_REPAIR_REVALIDATION_FAILED", "failure_stage": "lesson_pedagogical_revalidation"} if is_revalidation and blocking else {}),
            "progress": append_progress(state, progress_event("VALIDATING_PEDAGOGY", "Validating learning objectives and instructional treatment", "lesson_generation", current=6, total=9)),
            "node_durations_ms": node_duration_update(state, "validate_pedagogy", started),
        }

    async def validate_duplicates(state: LessonWorkflowState) -> dict[str, Any]:
        started = perf_counter()
        result = callbacks.validate_duplicates(dict(state.get("proposal") or {}))
        blocking = [issue for issue in result.issues if issue.get("severity") == "error"]
        emit(state, "lesson_duplicate_validation", "failed" if blocking else "passed", started_at=started,
             validation_finding_count=len(result.issues), blocking_finding_count=len(blocking),
             findings=[safe_workflow_issue_summary(issue, repairable=str(issue.get("code") or "") not in NON_REPAIRABLE_CODES) for issue in result.issues[:32]])
        return {
            "validation_issues": [*(state.get("validation_issues") or []), *result.issues],
            "metrics": {**dict(state.get("metrics") or {}), **result.metrics},
            "progress": append_progress(state, progress_event("CHECKING_DUPLICATION", "Checking generated lesson content for duplication", "lesson_generation", current=7, total=9)),
            "node_durations_ms": node_duration_update(state, "validate_duplicates", started),
        }

    async def classify_repair(state: LessonWorkflowState) -> dict[str, Any]:
        started = perf_counter()
        targets = classify_lesson_repair_targets(_blocking_issues(state))
        errors: list[WorkflowIssue] = []
        if not targets:
            errors.append({"code": "LESSON_VALIDATION_FAILED", "severity": "error", "message": "No safe scoped repair target was available."})
        emit(state, "lesson_repair_target_generation", "failed" if errors else "completed", started_at=started,
             repair_target_count=len(targets),
             repair_targets=[{"scope": target["scope"], "path": safe_workflow_path(target["path"]), "codes": list(target["codes"]), "allowed_fields": list(target["allowed_fields"])} for target in targets[:32]])
        return {"repair_targets": targets, "errors": errors, "node_durations_ms": node_duration_update(state, "classify_repair", started)}

    async def scoped_repair(state: LessonWorkflowState) -> dict[str, Any]:
        started = perf_counter()
        try:
            generated = await callbacks.repair_content(state["proposal"], state["repair_targets"])
        except WorkflowFailure as error:
            issue: WorkflowIssue = {"code": error.code, "severity": "error", "message": error.message}
            repair_pass = int(state.get("repair_attempts") or 0) + 1
            emit(state, error.failure_stage or "lesson_repair", "failed", started_at=started,
                 repair_pass_number=repair_pass, internal_failure_code=error.internal_code,
                 external_failure_code=error.code)
            return {"errors": [issue], "failure_code": error.code, "failure_internal_code": error.internal_code,
                    "failure_stage": error.failure_stage or "lesson_repair", "failure_repair_pass_number": repair_pass,
                    "node_durations_ms": node_duration_update(state, "scoped_repair", started)}
        emit(state, "lesson_repair", "completed", started_at=started,
             repair_pass_number=int(state.get("repair_attempts") or 0) + 1)
        return {
            "proposal": generated.value,
            "usage": generated.usage,
            "repair_attempts": int(state.get("repair_attempts") or 0) + 1,
            "progress": append_progress(state, progress_event("REPAIRING_CONTENT", "Repairing affected lesson content", "lesson_generation", current=8, total=9)),
            "node_durations_ms": node_duration_update(state, "scoped_repair", started),
        }

    async def finalize(state: LessonWorkflowState) -> dict[str, Any]:
        return {"status": "ready", "progress": append_progress(state, progress_event("PROPOSAL_READY", "Lesson proposal is ready for review", "lesson_generation", current=9, total=9))}

    async def fail(state: LessonWorkflowState) -> dict[str, Any]:
        code = _failure_code(state)
        internal_code = str(state.get("failure_internal_code") or code)
        failure_stage = str(state.get("failure_stage") or "lesson_validation")
        emit(state, failure_stage, "final_failure", internal_failure_code=internal_code,
             external_failure_code=code, failure_stage=failure_stage,
             repair_pass_number=int(state.get("failure_repair_pass_number") or state.get("repair_attempts") or 0))
        return {"status": "failed", "failure_code": code, "failure_internal_code": internal_code, "failure_stage": failure_stage}

    def after_contract(state: LessonWorkflowState) -> str:
        return "fail" if state.get("errors") else "retrieve_evidence"

    def after_retrieval(state: LessonWorkflowState) -> str:
        return "fail" if state.get("errors") else "validate_evidence"

    def after_evidence(state: LessonWorkflowState) -> str:
        return "fail" if _blocking_issues(state) else "generate_proposal"

    def after_generation(state: LessonWorkflowState) -> str:
        return "fail" if state.get("errors") else "validate_content"

    def after_content_validation(state: LessonWorkflowState) -> str:
        return "fail" if state.get("errors") else "validate_pedagogy"

    def after_pedagogy_validation(state: LessonWorkflowState) -> str:
        return "fail" if state.get("errors") else "validate_duplicates"

    def after_validation(state: LessonWorkflowState) -> str:
        errors = _blocking_issues(state)
        if not errors:
            return "finalize"
        codes = {str(issue.get("code") or "") for issue in errors}
        if codes & NON_REPAIRABLE_CODES:
            return "fail"
        if int(state.get("repair_attempts") or 0) >= int(state.get("max_repair_attempts") or 0):
            return "fail"
        return "classify_repair"

    def after_classification(state: LessonWorkflowState) -> str:
        return "fail" if state.get("errors") or not state.get("repair_targets") else "scoped_repair"

    def after_repair(state: LessonWorkflowState) -> str:
        return "fail" if state.get("errors") else "validate_content"

    graph.add_node("validate_contract", validate_contract)
    graph.add_node("retrieve_evidence", retrieve_evidence)
    graph.add_node("validate_evidence", validate_evidence)
    graph.add_node("generate_proposal", generate_proposal)
    graph.add_node("validate_content", validate_content)
    graph.add_node("validate_pedagogy", validate_pedagogy)
    graph.add_node("validate_duplicates", validate_duplicates)
    graph.add_node("classify_repair", classify_repair)
    graph.add_node("scoped_repair", scoped_repair)
    graph.add_node("finalize", finalize)
    graph.add_node("fail", fail)
    graph.add_edge(START, "validate_contract")
    graph.add_conditional_edges("validate_contract", after_contract, {"retrieve_evidence": "retrieve_evidence", "fail": "fail"})
    graph.add_conditional_edges("retrieve_evidence", after_retrieval, {"validate_evidence": "validate_evidence", "fail": "fail"})
    graph.add_conditional_edges("validate_evidence", after_evidence, {"generate_proposal": "generate_proposal", "fail": "fail"})
    graph.add_conditional_edges("generate_proposal", after_generation, {"validate_content": "validate_content", "fail": "fail"})
    graph.add_conditional_edges("validate_content", after_content_validation, {"validate_pedagogy": "validate_pedagogy", "fail": "fail"})
    graph.add_conditional_edges("validate_pedagogy", after_pedagogy_validation, {"validate_duplicates": "validate_duplicates", "fail": "fail"})
    graph.add_conditional_edges("validate_duplicates", after_validation, {"finalize": "finalize", "classify_repair": "classify_repair", "fail": "fail"})
    graph.add_conditional_edges("classify_repair", after_classification, {"scoped_repair": "scoped_repair", "fail": "fail"})
    graph.add_conditional_edges("scoped_repair", after_repair, {"validate_content": "validate_content", "fail": "fail"})
    graph.add_edge("finalize", END)
    graph.add_edge("fail", END)
    return graph.compile()


async def run_lesson_generation_workflow(
    callbacks: LessonGenerationWorkflowCallbacks,
    *,
    request_context: dict[str, Any],
    max_repair_attempts: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = perf_counter()
    graph = build_lesson_generation_graph(callbacks)
    state = await graph.ainvoke({
        "request_context": request_context,
        "repair_attempts": 0,
        "max_repair_attempts": max(0, max_repair_attempts),
        "status": "running",
        "progress": [],
        "errors": [],
        "node_durations_ms": {},
    })
    diagnostics = sanitized_workflow_diagnostics(state, workflow="lesson_generation", started_at=started)
    if state.get("status") != "ready":
        raise WorkflowFailure(
            str(state.get("failure_code") or "LESSON_VALIDATION_FAILED"),
            "Lesson generation workflow did not produce a valid proposal.",
            issues=list(state.get("validation_issues") or state.get("errors") or []),
            internal_code=str(state.get("failure_internal_code") or state.get("failure_code") or "LESSON_VALIDATION_FAILED"),
            failure_stage=str(state.get("failure_stage") or "lesson_validation"),
        )
    return dict(state["proposal"]), {**diagnostics, "usage": dict(state.get("usage") or {})}
