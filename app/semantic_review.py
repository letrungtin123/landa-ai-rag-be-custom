from __future__ import annotations

"""Bounded semantic-review contracts and one-pass repair coordination.

This module deliberately does not call a model, mutate persistence, or replace
deterministic schema/security validation.  Provider access is injected by the
orchestration boundary so offline tests can prove routing and repair invariants
without network access.
"""

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import re
from typing import Any, Awaitable, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


SEMANTIC_REVIEW_CONTRACT_VERSION = "semantic-review-v1"
SEMANTIC_REVIEW_PROMPT_VERSION = "semantic-review-prompt-v1"
SEMANTIC_REPAIR_POLICY_VERSION = "semantic-scoped-repair-v2"
MAX_REVIEW_FINDINGS = 32
MAX_REPAIR_COMPONENTS = 4

SemanticCriterion = Literal[
    "evidence_fidelity",
    "source_completeness",
    "instructional_alignment",
    "assessment_quality",
    "explanation_quality",
    "component_fit",
]
SemanticFindingCode = Literal[
    "EVIDENCE_CONTRADICTION",
    "EVIDENCE_CONDITION_OMITTED",
    "REQUIRED_EVIDENCE_OMITTED",
    "STRUCTURED_RELATION_LOST",
    "OBJECTIVE_ALIGNMENT_GAP",
    "ASSESSMENT_ANSWER_UNGROUNDED",
    "ASSESSMENT_DISTRACTOR_INVALID",
    "EXPLANATION_INCOHERENT",
    "TERMINOLOGY_INCONSISTENT",
    "PREREQUISITE_GAP",
    "DUPLICATE_FILLER",
    "COMPONENT_SEMANTIC_MISMATCH",
    "REVIEW_EVIDENCE_INSUFFICIENT",
]
SemanticSeverity = Literal["critical", "major", "minor"]
SemanticVerdict = Literal["pass", "review_required"]

SERIOUS_SEVERITIES = frozenset({"critical", "major"})
REPAIRABLE_CODES = frozenset({
    "EVIDENCE_CONTRADICTION",
    "EVIDENCE_CONDITION_OMITTED",
    "REQUIRED_EVIDENCE_OMITTED",
    "STRUCTURED_RELATION_LOST",
    "OBJECTIVE_ALIGNMENT_GAP",
    "ASSESSMENT_ANSWER_UNGROUNDED",
    "ASSESSMENT_DISTRACTOR_INVALID",
    "EXPLANATION_INCOHERENT",
    "TERMINOLOGY_INCONSISTENT",
    "DUPLICATE_FILLER",
    "COMPONENT_SEMANTIC_MISMATCH",
})
_PATH = re.compile(r"^components\[(\d+)](?:\.[A-Za-z0-9_.\[\]-]+)?$")
_CODE = re.compile(r"^[A-Z][A-Z0-9_]{0,99}$")


class SemanticReviewFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    criterion: SemanticCriterion
    code: SemanticFindingCode
    severity: SemanticSeverity
    scope: Literal["unit", "component"]
    component_index: int | None = Field(default=None, ge=0, le=31)
    candidate_path: str = Field(min_length=1, max_length=256)
    source_fact_ids: list[str] = Field(default_factory=list, max_length=64)
    witness_summary: str = Field(min_length=1, max_length=500)
    repair_instruction: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_scope(self) -> "SemanticReviewFinding":
        if len(self.source_fact_ids) != len(set(self.source_fact_ids)) or any(
            not item or len(item) > 255 for item in self.source_fact_ids
        ):
            raise ValueError("semantic review source references are invalid")
        match = _PATH.fullmatch(self.candidate_path)
        if self.scope == "component":
            if self.component_index is None or match is None or int(match.group(1)) != self.component_index:
                raise ValueError("component finding scope is invalid")
        elif self.component_index is not None or self.candidate_path != "unit":
            raise ValueError("unit finding scope is invalid")
        return self


class SemanticReviewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: Literal["semantic-review-v1"]
    verdict: SemanticVerdict
    reviewed_component_indices: list[int] = Field(min_length=1, max_length=32)
    findings: list[SemanticReviewFinding] = Field(default_factory=list, max_length=MAX_REVIEW_FINDINGS)

    @model_validator(mode="after")
    def validate_verdict(self) -> "SemanticReviewResponse":
        if len(self.reviewed_component_indices) != len(set(self.reviewed_component_indices)):
            raise ValueError("reviewed component indices must be unique")
        serious = [finding for finding in self.findings if finding.severity in SERIOUS_SEVERITIES]
        if (self.verdict == "pass") != (not serious):
            raise ValueError("semantic review verdict does not match serious findings")
        reviewed = set(self.reviewed_component_indices)
        if any(
            finding.scope == "component" and finding.component_index not in reviewed
            for finding in self.findings
        ):
            raise ValueError("finding is outside the reviewed component scope")
        return self


@dataclass(frozen=True)
class SemanticReviewRunOutcome:
    unit: dict[str, Any]
    quality_state: Literal["validated", "review_required"]
    review_status: Literal["passed", "review_required", "unavailable"]
    first_review: SemanticReviewResponse | None
    scoped_review: SemanticReviewResponse | None
    repair_attempted: bool
    repair_applied: bool
    repair_component_indices: tuple[int, ...]
    failure_code: str | None = None


SemanticReviewer = Callable[[dict[str, Any], tuple[int, ...]], Awaitable[SemanticReviewResponse]]
SemanticRepairer = Callable[
    [dict[str, Any], tuple[int, ...], tuple[SemanticReviewFinding, ...]],
    Awaitable[dict[str, Any]],
]
DeterministicValidator = Callable[[dict[str, Any]], str | None]


def semantic_review_config_hash(
    *,
    mode: str,
    model: str,
    timeout_ms: int = 0,
    max_output_tokens: int = 0,
    provider_attempt_cap: int = 0,
) -> str:
    return sha256(json.dumps({
        "contract_version": SEMANTIC_REVIEW_CONTRACT_VERSION,
        "prompt_version": SEMANTIC_REVIEW_PROMPT_VERSION,
        "repair_policy_version": SEMANTIC_REPAIR_POLICY_VERSION,
        "mode": mode,
        "model": model,
        "timeout_ms": timeout_ms,
        "max_output_tokens": max_output_tokens,
        "provider_attempt_cap": provider_attempt_cap,
        "max_findings": MAX_REVIEW_FINDINGS,
        "max_repair_components": MAX_REPAIR_COMPONENTS,
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_semantic_review_response(
    value: Any,
    *,
    component_count: int,
    allowed_source_fact_ids: set[str],
    requested_component_indices: tuple[int, ...],
) -> SemanticReviewResponse:
    result = value if isinstance(value, SemanticReviewResponse) else SemanticReviewResponse.model_validate(value)
    requested = tuple(sorted(set(requested_component_indices)))
    if (
        component_count < 1
        or any(index < 0 or index >= component_count for index in requested)
        or tuple(sorted(result.reviewed_component_indices)) != requested
        or any(not set(finding.source_fact_ids) <= allowed_source_fact_ids for finding in result.findings)
        or (
            requested != tuple(range(component_count))
            and any(finding.scope == "unit" for finding in result.findings)
        )
    ):
        raise ValueError("semantic review authority is invalid")
    return result


def semantic_repair_targets(review: SemanticReviewResponse) -> tuple[int, ...]:
    targets = sorted({
        finding.component_index
        for finding in review.findings
        if (
            finding.severity in SERIOUS_SEVERITIES
            and finding.scope == "component"
            and finding.code in REPAIRABLE_CODES
            and finding.component_index is not None
        )
    })
    return tuple(targets[:MAX_REPAIR_COMPONENTS])


def build_semantic_review_prompt(
    *,
    unit: dict[str, Any],
    expected: dict[str, Any],
    manifest: dict[str, Any],
    component_indices: tuple[int, ...],
    locale: str,
) -> str:
    """Build a frozen reviewer prompt without granting source/identity authority."""

    components = unit.get("components") if isinstance(unit.get("components"), list) else []
    selected_components = [
        {"component_index": index, "component": components[index]}
        for index in component_indices
        if 0 <= index < len(components)
    ]
    facts = [
        {"fact_id": row.get("fact_id"), "text": row.get("text"), "locator": row.get("locator")}
        for row in manifest.get("facts", [])
        if isinstance(row, dict) and row.get("fact_id") in set(expected.get("source_fact_ids", []))
    ]
    evidence_bundle = manifest.get("source_evidence_bundle")
    contract = {
        "unit_title": expected.get("unit_title"),
        "unit_purpose": expected.get("unit_purpose"),
        "learning_objectives": expected.get("learning_objectives", []),
        "learning_objective_refs": expected.get("learning_objective_refs", []),
        "component_plan": expected.get("component_plan", []),
        "source_facts": facts,
        "source_evidence_bundle": evidence_bundle if isinstance(evidence_bundle, dict) else None,
        "candidate_components": selected_components,
    }
    return "\n\n".join([
        f"SEMANTIC REVIEW CONTRACT {SEMANTIC_REVIEW_CONTRACT_VERSION}; prompt {SEMANTIC_REVIEW_PROMPT_VERSION}.",
        "You are an independent instructional-content reviewer. Review only the supplied component indices. "
        "Do not redesign the course, invent facts, create source IDs, relax schema/security rules, or return a numeric score. "
        "Deterministic validation is authoritative and separate from this review.",
        "Evaluate: evidence fidelity including conditions/exceptions; completeness of required facts/tables/visual relations; "
        "objective-instruction-activity-assessment alignment; single-answer quiz grounding and distractor validity; "
        "coherence, terminology, prerequisites and duplicate filler; and whether the selected component type fits the semantics.",
        "Every finding must use one allowed code, identify critical/major/minor severity, point to unit or an exact components[N] path, "
        "cite only supplied fact IDs, give a concise witness summary, and give one bounded repair instruction. "
        "Use review_required iff at least one critical or major finding exists. Minor-only findings keep verdict pass.",
        f"Output language for witness summaries and repair instructions: {locale}.",
        "Immutable review input:\n" + json.dumps(contract, ensure_ascii=False, separators=(",", ":")),
    ])


def build_semantic_repair_prompt(
    *,
    unit: dict[str, Any],
    expected: dict[str, Any],
    manifest: dict[str, Any],
    component_indices: tuple[int, ...],
    findings: tuple[SemanticReviewFinding, ...],
    locale: str,
) -> str:
    components = unit.get("components") if isinstance(unit.get("components"), list) else []
    allowed_facts = set(expected.get("source_fact_ids", [])) | set(expected.get("supporting_evidence_fact_ids", []))
    evidence = [
        {"fact_id": row.get("fact_id"), "text": row.get("text")}
        for collection in ("facts", "supporting_evidence_facts")
        for row in manifest.get(collection, [])
        if isinstance(row, dict) and row.get("fact_id") in allowed_facts
    ]
    finding_payload = [finding.model_dump(mode="json") for finding in findings]
    baselines = [{"component_index": index, "component": components[index]} for index in component_indices]
    return "\n\n".join([
        f"SCOPED SEMANTIC REPAIR {SEMANTIC_REPAIR_POLICY_VERSION}.",
        "Repair only the server-addressed component slots. Preserve component type, identity, order, unit title, source ownership, "
        "covered-source claims and supporting-evidence IDs. Do not add/remove/reorder components or broaden evidence scope. "
        "Return payload fields only in the supplied response schema. Apply each repair instruction using only immutable evidence. "
        "Keep provenance private: never put source filenames, citations, page/slide/chunk locators, internal IDs, or source-attribution phrases into learner-facing fields.",
        f"Locale: {locale}.",
        "Semantic findings:\n" + json.dumps(finding_payload, ensure_ascii=False, separators=(",", ":")),
        "Authorized component baselines:\n" + json.dumps(baselines, ensure_ascii=False, separators=(",", ":")),
        "Immutable source evidence (provenance locators intentionally withheld):\n" + json.dumps({
            "facts": evidence,
        }, ensure_ascii=False, separators=(",", ":")),
    ])


def safe_semantic_review_summary(
    outcome: SemanticReviewRunOutcome,
    *,
    config_hash: str,
) -> dict[str, Any]:
    if outcome.scoped_review is not None and outcome.first_review is not None:
        repaired_indices = set(outcome.repair_component_indices)
        findings = [
            finding for finding in outcome.first_review.findings
            if finding.component_index not in repaired_indices
        ] + list(outcome.scoped_review.findings)
    else:
        review = outcome.first_review
        findings = [] if review is None else list(review.findings)
    counts = Counter(finding.severity for finding in findings)
    safe_findings = [{
        "criterion": finding.criterion,
        "code": finding.code,
        "severity": finding.severity,
        "scope": finding.scope,
        "component_index": finding.component_index,
        "candidate_path": finding.candidate_path,
        "source_fact_key_hashes": [sha256(value.encode()).hexdigest()[:16] for value in finding.source_fact_ids],
        "witness_sha256": sha256(finding.witness_summary.encode()).hexdigest(),
        "repair_instruction_sha256": sha256(finding.repair_instruction.encode()).hexdigest(),
    } for finding in findings]
    return {
        "contract_version": SEMANTIC_REVIEW_CONTRACT_VERSION,
        "config_hash": config_hash,
        "status": outcome.review_status,
        "quality_state": outcome.quality_state,
        "finding_counts": {severity: counts.get(severity, 0) for severity in ("critical", "major", "minor")},
        "findings": safe_findings,
        "repair_attempted": outcome.repair_attempted,
        "repair_applied": outcome.repair_applied,
        "repair_component_indices": list(outcome.repair_component_indices),
        "failure_code": outcome.failure_code,
    }


def _serious_findings(review: SemanticReviewResponse) -> list[SemanticReviewFinding]:
    return [finding for finding in review.findings if finding.severity in SERIOUS_SEVERITIES]


def _serious_finding_counts(findings: list[SemanticReviewFinding]) -> tuple[int, int]:
    return (
        sum(finding.severity == "critical" for finding in findings),
        sum(finding.severity == "major" for finding in findings),
    )


def _has_strict_repair_progress(
    before: list[SemanticReviewFinding],
    after: list[SemanticReviewFinding],
) -> bool:
    """Require fewer serious findings without introducing new critical risk.

    Lexicographic severity tuples are unsafe here: replacing one critical
    finding with several major findings would compare as an improvement.  A
    candidate is accepted only when its critical count does not increase and
    its total critical+major count strictly decreases.
    """

    before_critical, before_major = _serious_finding_counts(before)
    after_critical, after_major = _serious_finding_counts(after)
    return (
        after_critical <= before_critical
        and after_critical + after_major < before_critical + before_major
    )


async def run_bounded_semantic_review(
    unit: dict[str, Any],
    *,
    reviewer: SemanticReviewer,
    repairer: SemanticRepairer,
    deterministic_validate: DeterministicValidator,
    allow_repair: bool,
) -> SemanticReviewRunOutcome:
    """Review once, repair at most once, then re-review only affected slots.

    Any reviewer/repair/validation failure preserves a visible draft and marks
    it for review. A repaired candidate replaces the baseline only when it is
    deterministic-valid and the scoped serious-finding tuple strictly improves.
    """

    baseline = deepcopy(unit)
    components = baseline.get("components") if isinstance(baseline.get("components"), list) else []
    full_scope = tuple(range(len(components)))
    if not full_scope:
        return SemanticReviewRunOutcome(
            baseline, "review_required", "unavailable", None, None, False, False, (),
            "SEMANTIC_REVIEW_SCOPE_INVALID",
        )
    try:
        first = await reviewer(deepcopy(baseline), full_scope)
    except Exception:
        return SemanticReviewRunOutcome(
            baseline, "review_required", "unavailable", None, None, False, False, (),
            "SEMANTIC_REVIEW_UNAVAILABLE",
        )
    if first.verdict == "pass":
        return SemanticReviewRunOutcome(
            baseline, "validated", "passed", first, None, False, False, (), None,
        )
    targets = semantic_repair_targets(first)
    if not allow_repair or not targets:
        return SemanticReviewRunOutcome(
            baseline, "review_required", "review_required", first, None, False, False, targets, None,
        )
    targeted_findings = tuple(
        finding for finding in _serious_findings(first)
        if finding.component_index in targets
    )
    try:
        repaired = await repairer(deepcopy(baseline), targets, targeted_findings)
        deterministic_failure = deterministic_validate(repaired)
        if deterministic_failure:
            return SemanticReviewRunOutcome(
                baseline, "review_required", "review_required", first, None, True, False, targets,
                "SEMANTIC_REPAIR_DETERMINISTIC_RECHECK_FAILED",
            )
        scoped = await reviewer(deepcopy(repaired), targets)
    except Exception:
        return SemanticReviewRunOutcome(
            baseline, "review_required", "review_required", first, None, True, False, targets,
            "SEMANTIC_REPAIR_UNAVAILABLE",
        )
    carried_findings = [
        finding for finding in first.findings
        if finding.component_index not in targets
    ]
    if len(carried_findings) + len(scoped.findings) > MAX_REVIEW_FINDINGS:
        return SemanticReviewRunOutcome(
            baseline, "review_required", "review_required", first, None, True, False, targets,
            "SEMANTIC_REPAIR_REVIEW_OVERFLOW",
        )
    before = [finding for finding in _serious_findings(first) if finding.component_index in targets]
    after = _serious_findings(scoped)
    improved = _has_strict_repair_progress(before, after)
    if not improved:
        return SemanticReviewRunOutcome(
            baseline, "review_required", "review_required", first, scoped, True, False, targets,
            "SEMANTIC_REPAIR_NO_PROGRESS",
        )
    unresolved_outside_scope = any(
        finding.component_index not in targets for finding in _serious_findings(first)
    )
    quality_state: Literal["validated", "review_required"] = (
        "validated" if scoped.verdict == "pass" and not unresolved_outside_scope else "review_required"
    )
    return SemanticReviewRunOutcome(
        deepcopy(repaired), quality_state,
        "passed" if quality_state == "validated" else "review_required",
        first, scoped, True, True, targets, None,
    )


def evaluate_semantic_review_predictions(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate frozen predictions against separately-authored oracle labels."""

    if not cases:
        raise ValueError("semantic review benchmark is empty")
    ids: set[str] = set()
    counts = Counter()
    by_split: dict[str, Counter] = {}
    all_human_reviewed = True
    for case in cases:
        case_id = str(case.get("case_id") or "")
        split = str(case.get("split") or "")
        oracle = case.get("oracle")
        prediction = case.get("prediction")
        if (
            not case_id or case_id in ids or split not in {"development", "held_out"}
            or not isinstance(oracle, dict) or not isinstance(prediction, dict)
            or type(oracle.get("acceptable")) is not bool
            or type(oracle.get("serious_error")) is not bool
            or oracle.get("label_status") not in {"human_reviewed", "provisional_product_spec"}
            or prediction.get("verdict") not in {"pass", "review_required"}
            or not case.get("document_group") or not case.get("course_group")
        ):
            raise ValueError("semantic review benchmark case is invalid")
        ids.add(case_id)
        all_human_reviewed &= oracle["label_status"] == "human_reviewed"
        bucket = by_split.setdefault(split, Counter())
        predicted_pass = prediction["verdict"] == "pass"
        if oracle["serious_error"]:
            counts["serious_total"] += 1
            bucket["serious_total"] += 1
            if predicted_pass:
                counts["serious_false_pass"] += 1
                bucket["serious_false_pass"] += 1
        if oracle["acceptable"]:
            counts["acceptable_total"] += 1
            bucket["acceptable_total"] += 1
            if not predicted_pass:
                counts["acceptable_false_fail"] += 1
                bucket["acceptable_false_fail"] += 1
        counts["case_count"] += 1
        bucket["case_count"] += 1

    def rate(numerator: int, denominator: int) -> float | None:
        return round(numerator / denominator, 6) if denominator else None

    def summary(counter: Counter) -> dict[str, Any]:
        result = dict(counter)
        result["serious_false_pass_rate"] = rate(counter["serious_false_pass"], counter["serious_total"])
        result["acceptable_false_fail_rate"] = rate(counter["acceptable_false_fail"], counter["acceptable_total"])
        return result

    document_groups = {str(case["document_group"]) for case in cases}
    course_groups = {str(case["course_group"]) for case in cases}
    return {
        "contract_version": "semantic-review-benchmark-v1",
        **summary(counts),
        "by_split": {split: summary(counter) for split, counter in sorted(by_split.items())},
        "document_group_count": len(document_groups),
        "course_group_count": len(course_groups),
        "all_labels_human_reviewed": all_human_reviewed,
        "release_claim_eligible": all_human_reviewed and "held_out" in by_split,
    }
