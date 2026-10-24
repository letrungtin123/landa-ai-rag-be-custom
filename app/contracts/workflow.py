"""The safe issue and failure contracts of the request-scoped authoring workflows.

Moved verbatim from ``app.workflows.contracts`` (refactor wave 3b): the staged unit validators and the orchestration
V2 unit service raise ``WorkflowFailure`` too, and strict code does not import the legacy pipeline.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

WorkflowSeverity = Literal["error", "warning", "info"]


class WorkflowIssue(TypedDict, total=False):
    code: str
    severity: WorkflowSeverity
    message: str
    path: str
    related_paths: list[str]
    objective_ids: list[str]
    learning_block_ids: list[str]
    component_ids: list[str]
    source_fact_ids: list[str]
    repairable: bool
    # Safe schema diagnostics only. These describe the contract and runtime
    # shape, never customer-authored values or source content.
    constraint: str
    expected_type: str
    actual_type: str
    validator: str
    schema_error_code: str
    # A structural schema path is server-generated validation metadata, never
    # customer/provider content. It lets repair policy distinguish an exact
    # lesson.units cardinality violation from other lesson-level arrays.
    schema_path: str
    # V5 Course Architecture repair scheduling is layer-aware. Only the
    # deterministic validator assigns this field; the provider never does.
    repair_layer: str
    # Canonical Source Map ownership metadata. These are server-generated
    # identifiers and structural counts only; they never carry source text or
    # provider-generated Blueprint content.
    concept_id: str
    concept_ids: list[str]
    section_id: str
    section_ids: list[str]
    expected_primary_owner_count: int
    actual_primary_owner_count: int
    ownership_level: str
    scope_classification: str
    coverage_state: str
    safe_reason: str
    # V5 evidence-scope diagnostics. These identifiers are server-generated
    # provenance keys; they carry no source text or provider content.
    evidence_scope_id: str
    evidence_scope_ids: list[str]
    eligible_repair_paths: list[str]
    # V5 assessment-plan compiler diagnostics are count/ID metadata only.
    # Candidate details are kept request-scoped and are never emitted to logs.
    assessment_candidate_count: int
    assessment_candidate_rejection_counts: dict[str, int]
    assessment_plan_fingerprint: str


class WorkflowFailure(RuntimeError):
    """A safe workflow failure category suitable for the service boundary."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        issues: list[WorkflowIssue] | None = None,
        internal_code: str | None = None,
        failure_stage: str | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.issues = issues or []
        # `code` remains the API-safe workflow category. More specific codes
        # are internal-only diagnostics and must never be inferred later from
        # this generic external category.
        self.internal_code = internal_code or code
        self.failure_stage = failure_stage
        self.diagnostics = diagnostics or {}
