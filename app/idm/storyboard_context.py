"""Inputs of the W5 unit writer (``app.idm.storyboard``): the injected staged-writer helpers, the brief checked
against the contract, the IDM output budget, what Node's acceptance reads from the contract, the staged
validator's ``expected`` dict, the draft a unit is settled in, and the code sets that decide how a finding is
settled (repaired, targeted, kept for review).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Protocol

from pydantic import BaseModel, ValidationError

from app.contracts.orchestration_v2.provider import UnitGenerationContractV2
from app.idm.contracts import IdmUnitBriefV1, brief_hash_of
from app.idm.diagram import idm_diagram_relationships
from app.idm.faq_checks import FAQ_VALUE_CODES
from app.idm.fidelity import brief_mode, budget_words
from app.idm.fidelity_checks import FIDELITY_CODES
from app.idm.mcq import ANSWER_LENGTH_CUE_CODE, QUESTION_INCOMPLETE_CODE
from app.idm.node_acceptance import AcceptanceContext, AcceptanceFinding
from app.idm.policy import (
    EXTRA_WORDS_PER_MUST_KNOW_BLOCK,
    MAX_GENERATED_WORDS,
    MIN_GENERATED_WORDS,
    SEGMENT_WORD_BUDGET,
    VISIBLE_CHARS_PER_WORD,
    WORKSHEET_COMPONENT_TYPE,
)
from app.idm.qa import (
    ANSWER_CONTESTED_CODE,
    ANSWER_LEAK_CODE,
    CALLOUT_UNGROUNDED_CODE,
    FAQ_RESTATES_HTML_CODE,
    FAQ_TITLE_MISMATCH_CODE,
    FRAMEWORK_INCOMPLETE_CODE,
    SOURCE_ARTIFACT_CODE,
    WORKSHEET_INCOMPLETE_CODE,
    JudgeMode,
)
from app.idm.quiz_text import QUIZ_CODES, QUIZ_REVIEW_CODES
from app.idm.runtime import IdmStageError
from app.idm.worksheet_input import WORKSHEET_INPUT_CODE
from app.instructional_density import INSTRUCTIONAL_DENSITY_POLICY_VERSION

_SLOT_RE: Final = re.compile(r"components\[(\d+)\]")
_MS: Final = 1000
# Log/prompt-safe tokens: a validator code and a JSON path (never provider text).
_SAFE_CODE_RE: Final = re.compile(r"^[A-Z][A-Z0-9_]{2,99}$")
_SAFE_PATH_RE: Final = re.compile(r"^[A-Za-z0-9_.\[\]]{1,160}$")
_MAX_LOGGED_DETAILS: Final = 16
_COVERAGE_CODES: Final = frozenset({
    "COMPONENT_COVERAGE_MISSING", "COMPONENT_COVERAGE_INCOMPLETE", "INVALID_FACT_ID_ARRAY",
})
VERBATIM_COPY_CODE: Final = "IDM_W5_VERBATIM_COPY"
# A worksheet that misses part of its structure keeps the provider slot for author review: the
# source-locked html fallback would replace the practice with plain explanation.
# An option copying the html before the question (QC course 364564, N3) is a review note once the repair
# could not fix it: the source-locked question would drop the provider's scenario practice. So are an
# ungrounded callout that cannot become a paragraph (N9), FAQ items that restate the html while too few
# others remain, an FAQ title that does not match its questions (N11) and a title that promises a framework
# the html does not list (N5): the source-locked rebuild would replace the whole teaching slot for a problem
# the author fixes in a minute.
# QLT-5 C: a copy (slot or sentence level), explanation or new-fact finding of a teaching slot is a review note too:
# the source-locked rebuild would answer "too close to the source" with the source itself (plan 2026-10-13 §2.2,
# point 5). QLT-5 D: so is an FAQ that recalls, copies a source sentence or states a new number or name: the
# source-locked FAQ is made of verbatim source sentences.
# QC feedback 2026-10-21 (A1/A2): the quiz-text findings (an explanation without its principle or reasons, a stem in one
# block) too: the source-locked question would replace a good question for a layout or wording problem.
# Review m18: not a question over its cap (Node cuts it at 1,000 characters): after its repair it falls back.
_REVIEW_FIRST_CODES: Final = frozenset({WORKSHEET_INCOMPLETE_CODE, WORKSHEET_INPUT_CODE, ANSWER_LEAK_CODE,
                                        ANSWER_CONTESTED_CODE,
                                        CALLOUT_UNGROUNDED_CODE, FAQ_RESTATES_HTML_CODE, FAQ_TITLE_MISMATCH_CODE,
                                        FRAMEWORK_INCOMPLETE_CODE, SOURCE_ARTIFACT_CODE, VERBATIM_COPY_CODE,
                                        *FIDELITY_CODES, *FAQ_VALUE_CODES, *QUIZ_REVIEW_CODES})
# Findings whose repair is dropped when it breaks a shared or Node rule: they end as review notes anyway.
# Run 5496155c: a worksheet repair that adds its input data but breaks a rule (the length budget) is dropped too.
_REVIEW_ONLY_CODES: Final = frozenset({*FIDELITY_CODES, *FAQ_VALUE_CODES, *QUIZ_REVIEW_CODES, WORKSHEET_INPUT_CODE})
# Findings a deterministic step settles when the repair does not (a list inserted, an FAQ item dropped, a callout
# turned into prose, a review note): their repair alone is a targeted one, a scoped rewrite at low thinking. QC run
# ab8d67e1 (R3): two leak repairs at medium thinking were cut by the unit deadline.
# QLT-8: a question cut before it asks anything is one targeted repair too (the whole stem is rewritten).
_TARGETED_REPAIR_CODES: Final = frozenset({FRAMEWORK_INCOMPLETE_CODE, ANSWER_LENGTH_CUE_CODE, ANSWER_LEAK_CODE,
                                           ANSWER_CONTESTED_CODE, FAQ_RESTATES_HTML_CODE, FAQ_TITLE_MISMATCH_CODE,
                                           CALLOUT_UNGROUNDED_CODE, SOURCE_ARTIFACT_CODE, QUESTION_INCOMPLETE_CODE,
                                           *FIDELITY_CODES, *FAQ_VALUE_CODES, *QUIZ_CODES})
# A non-targeted repair that would not finish before the unit deadline is not started (R3).
REPAIR_SKIPPED_CODE: Final = "IDM_W5_REPAIR_SKIPPED_BUDGET"
_FALLBACK_INVALID: Final = "ORCHESTRATION_V2_UNIT_FALLBACK_INVALID"


def safe_error_details(error: BaseException) -> list[dict[str, Any]]:
    """The validator code and path behind a rejected answer, in ``ValidationError``-like form.

    Shared validators raise ``LessonAuthorProposalValidationError(code)`` (the code is the
    message), ``LessonAuthorProposalValidationError(message, code=..., path=...)`` or
    ``WorkflowFailure`` with ``diagnostics.validation_finding``; only tokens that look like a
    code or a JSON path are kept, so provider text can never reach a log line or a prompt.
    """

    diagnostics = getattr(error, "diagnostics", None)
    finding = diagnostics.get("validation_finding") if isinstance(diagnostics, dict) else None
    candidates: list[tuple[Any, Any]] = []
    if isinstance(finding, dict):
        candidates.append((finding.get("code"), finding.get("path")))
    # A code raised as the message carries no location (its ``path`` is the class default "unit").
    candidates.append((str(error), None))
    candidates.append((getattr(error, "code", None), getattr(error, "path", None)))
    for code, path in candidates:
        if isinstance(code, str) and _SAFE_CODE_RE.fullmatch(code):
            return [{"type": code, "loc": [path] if isinstance(path, str) and _SAFE_PATH_RE.fullmatch(path) else []}]
    if isinstance(error, json.JSONDecodeError):
        return [{"type": "json_invalid", "loc": []}]
    return [{"type": type(error).__name__, "loc": []}]


def _safe_path(path: Any) -> str | None:
    return path if isinstance(path, str) and _SAFE_PATH_RE.fullmatch(path) else None


class UnitFinding(Protocol):
    code: str
    path: str


@dataclass(frozen=True)
class IdmUnitDeps:
    """Shared staged-writer helpers injected by the service layer."""

    build_instance_model: Callable[[list[dict[str, Any]]], type[BaseModel]]
    bind_instance_payload: Callable[[Any, dict[str, Any]], dict[str, Any]]
    validate_unit: Callable[[dict[str, Any], dict[str, Any]], UnitFinding | None]
    build_repair_model: Callable[[dict[str, Any], list[int], list[int]], type[BaseModel]]
    decode_repair: Callable[[str, dict[str, Any], list[int], list[int], dict[str, Any]], dict[str, Any]]
    merge_repair: Callable[[dict[str, Any], dict[str, Any], list[int], list[int]], dict[str, Any]]
    source_locked_unit: dict[str, Any] | None
    purity_context: dict[str, Any]
    evidence_review_required: bool
    judge_mode: JudgeMode
    recoverable_errors: tuple[type[Exception], ...]
    # Per-slot source-locked rebuilds in plan order; ``None`` marks a slot whose evidence cannot
    # rebuild that type (prose-only ``problem``, ``la_faq`` without explicit conditions, ...).
    # Optional for injected deps: derived from ``source_locked_unit`` when absent.
    source_locked_components: Sequence[dict[str, Any] | None] | None = None
    # Observer of the ``recoverable_errors`` a W5 step turns into an invalid answer: (stage, code, error).
    on_recoverable_error: Callable[[str, str, BaseException], None] | None = None


def brief_purpose(brief: IdmUnitBriefV1) -> dict[str, str] | None:
    """The course purpose a brief carries (QLT-5), as plan data for the writer and the judge."""

    if brief.purpose_profile is None:
        return None
    return {"profile": brief.purpose_profile, "summary": brief.purpose_summary or ""}


def parse_brief(contract: UnitGenerationContractV2) -> IdmUnitBriefV1:
    """Validate the brief and check it against the contract slots (spec §7.7.2 step 1)."""

    raw = contract.idm_unit_brief
    try:
        brief = IdmUnitBriefV1.model_validate(raw)
    except ValidationError as error:
        raise IdmStageError("IDM_W5_BRIEF_CONTRACT_MISMATCH") from error
    if not isinstance(raw, dict) or brief_hash_of(raw) != brief.brief_hash:
        raise IdmStageError("IDM_W5_BRIEF_CONTRACT_MISMATCH")
    plans = contract.component_plan
    if len(brief.components) != len(plans) or any(
            slot.component_plan_id != plan.component_plan_id or slot.type != plan.type
            or slot.owned_fact_keys != plan.source_fact_ids
            or slot.supporting_fact_keys != plan.supporting_evidence_fact_ids
            for slot, plan in zip(brief.components, plans, strict=True)):
        raise IdmStageError("IDM_W5_BRIEF_CONTRACT_MISMATCH")
    return brief


def output_budget(brief: IdmUnitBriefV1, contract: UnitGenerationContractV2) -> dict[str, Any]:
    """Segment-based teaching budget (spec §7.7.3); never the 1.75x-source rule. An explained unit (QLT-5 C) gets
    ``EXPLAIN_BUDGET_PERCENT`` of it; Node's ``idmUnitOutputBudget`` computes the same numbers from the brief."""

    blocks = {treatment.block_id for slot in brief.components for treatment in slot.treatments}
    words = SEGMENT_WORD_BUDGET[brief.unit_segment] + EXTRA_WORDS_PER_MUST_KNOW_BLOCK * max(0, len(blocks) - 1)
    words = budget_words(max(MIN_GENERATED_WORDS, min(MAX_GENERATED_WORDS, words)), brief_mode(brief),
                         brief.unit_segment)
    source_chars = sum(len(fact.fact_text) for fact in contract.source_facts)
    return {"policy_version": INSTRUCTIONAL_DENSITY_POLICY_VERSION, "source_content_chars": source_chars,
            "source_estimated_words": len(" ".join(fact.fact_text for fact in contract.source_facts).split()),
            "max_visible_chars": words * VISIBLE_CHARS_PER_WORD, "max_words": words}


def acceptance_context(contract: UnitGenerationContractV2, brief: IdmUnitBriefV1) -> AcceptanceContext:
    """What Node's unit acceptance reads from the contract (``acceptOrchestrationV2GeneratedUnit``)."""

    budget = output_budget(brief, contract)
    return AcceptanceContext(
        unit_title=contract.unit_title, unit_source_fact_ids=tuple(contract.unit_source_fact_ids),
        plans=tuple(plan.model_dump(mode="json") for plan in contract.component_plan),
        exact_identifiers=tuple(value for fact in contract.source_facts for value in (fact.fact_key, fact.source_ref)
                                if value),
        budget={"max_words": budget["max_words"], "max_visible_chars": budget["max_visible_chars"]},
        worksheet_slots=frozenset(index for index, slot in enumerate(brief.components)
                                  if slot.role == "practice" and slot.type == WORKSHEET_COMPONENT_TYPE))


@dataclass
class NodeAcceptanceUnitFinding:
    """A Node acceptance rule in the shape of a shared-validator finding (``code``, ``path``)."""

    code: str
    path: str
    acceptance: AcceptanceFinding

    @classmethod
    def of(cls, acceptance: AcceptanceFinding) -> NodeAcceptanceUnitFinding:
        return cls(acceptance.code, acceptance.path, acceptance)


def build_idm_expected(contract: UnitGenerationContractV2, brief: IdmUnitBriefV1, deps: IdmUnitDeps,
                       locale: str) -> dict[str, Any]:
    """The staged validator's ``expected`` dict with the IDM output budget."""

    supporting = list(dict.fromkeys(fact_id for plan in contract.component_plan
                                    for fact_id in plan.supporting_evidence_fact_ids))
    text_by_id = {fact.fact_key: fact.fact_text for fact in contract.source_facts}
    # The same step construction as the IDM diagram fallback (app.idm.diagram), so a provider
    # diagram is never required to draw a slogan or heading the fallback would refuse (run e869f43a).
    diagrams = {
        plan.component_plan_id: idm_diagram_relationships(
            text_by_id[fact_id] for fact_id in dict.fromkeys([*plan.source_fact_ids,
                                                              *plan.supporting_evidence_fact_ids])
            if fact_id in text_by_id)
        for plan in contract.component_plan if plan.type == "la_diagram"
    }
    return {
        "unit_title": contract.unit_title, "unit_purpose": contract.unit_purpose,
        "source_fact_ids": list(contract.unit_source_fact_ids), "supporting_evidence_fact_ids": supporting,
        "component_types": [plan.type for plan in contract.component_plan],
        "component_plan": [plan.model_dump(mode="json") for plan in contract.component_plan],
        "diagram_relationships_by_plan_id": diagrams,
        "learning_objectives": list(contract.lesson_learning_objectives),
        "learning_objective_refs": list(contract.unit_learning_objective_refs), "locale": locale,
        "learner_content_purity": deps.purity_context, "instructional_output_budget": output_budget(brief, contract),
    }


def _strip_learning_block_ids(value: Any) -> Any:
    """Teaching-group ids are server-owned; V2 IDM plans carry none."""

    slots = value.get("components") if isinstance(value, dict) else None
    # A writer answer addresses slots by key ({"c0": ...}); a decoded repair delta is a list.
    payloads = slots.values() if isinstance(slots, dict) else slots if isinstance(slots, list) else []
    for payload in payloads:
        semantic = payload.get("semantic_content") if isinstance(payload, dict) else None
        sections = semantic.get("sections") if isinstance(semantic, dict) else None
        for section in sections if isinstance(sections, list) else []:
            if isinstance(section, dict):
                section["learning_block_ids"] = []
    return value


def _slot_of(finding: UnitFinding | None) -> int | None:
    match = _SLOT_RE.search(finding.path) if finding is not None else None
    return int(match.group(1)) if match else None


@dataclass
class _Draft:
    unit: dict[str, Any]
    fallback_slots: list[int]
    repair_applied: bool = False
    # Provider slots kept for author review: the shared payload validator accepts them, only IDM
    # pedagogy checks (spec §7.7.2(b)) still fail, and no source-locked rebuild exists for the slot.
    review_slots: list[int] = field(default_factory=list)
    # Why each fallback or review slot was settled that way (validator codes; QC 234653, D16).
    reasons: dict[int, list[str]] = field(default_factory=dict)
