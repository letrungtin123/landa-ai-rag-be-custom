"""IDM ``generate_unit`` task: W5 storyboard writer + W6 judge (spec §7.7).

The writer fills exactly the server-owned component slots of the unit contract.
Acceptance reuses the shared staged validators (injected through
:class:`IdmUnitDeps`, so ``app.idm`` never imports ``app.main``), then Node's own
revision-0 acceptance rules (:mod:`app.idm.node_acceptance`), and adds the IDM
checks. A failing slot gets one scoped repair, then the source-locked fallback; a
unit Node would reject is never returned (run c2e5ac41).
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Protocol

from pydantic import BaseModel, ValidationError

from app.idm.contracts import IdmUnitBriefV1, brief_hash_of
from app.idm.diagram import idm_diagram_relationships
from app.idm.framework import build_promise, framework_coverage, framework_support_lines, with_framework_list
from app.idm.html_rules import (
    DENSITY_CODE,
    HtmlRuleViolation,
    html_rule_violations,
    minimum_visible_chars,
    normalize_html_semantic,
)
from app.idm.mcq import normalize_single_choice
from app.idm.node_acceptance import (
    AcceptanceContext,
    AcceptanceFinding,
    acceptance_log,
    node_acceptance_findings,
)
from app.idm.policy import (
    EXTRA_WORDS_PER_MUST_KNOW_BLOCK,
    IDM_FAQ_MIN_ITEMS,
    IDM_TARGETED_REPAIR_MIN_SECONDS,
    IDM_UNIT_WRITER_MAX_OUTPUT_TOKENS,
    MAX_GENERATED_WORDS,
    MIN_GENERATED_WORDS,
    SEGMENT_WORD_BUDGET,
    THINKING_TARGETED_REPAIR,
    THINKING_W5,
    VISIBLE_CHARS_PER_WORD,
    WORKSHEET_COMPONENT_TYPE,
)
from app.idm.prompts import (
    COMPACT_UNIT,
    html_slot_rules,
    html_violation_line,
    repair_suffix,
    rule_line,
    truncation_suffix,
    unit_repair_suffix,
    unit_writer_prompt,
)
from app.idm.qa import (
    ANSWER_LEAK_CODE,
    CALLOUT_TO_PROSE_CODE,
    CALLOUT_UNGROUNDED_CODE,
    FAQ_ITEMS_DROPPED_CODE,
    FAQ_RESTATES_HTML_CODE,
    FAQ_TITLE_MISMATCH_CODE,
    FAQ_UNGROUNDED_CODE,
    FRAMEWORK_INCOMPLETE_CODE,
    FRAMEWORK_LIST_INSERTED_CODE,
    WORKSHEET_INCOMPLETE_CODE,
    JudgeMode,
    JudgeOutcome,
    blocking_count,
    build_unit_author_note,
    build_unit_quality,
    callouts_as_paragraphs,
    deterministic_slot_findings,
    final_unit_findings,
    has_practice_slot,
    html_before,
    repair_targets,
    restated_faq_items,
    run_judge,
    settled_codes,
    ungrounded_callouts,
    ungrounded_faq_items,
)
from app.idm.runtime import (
    IdmBudgetError,
    IdmProviderError,
    IdmResponseInvalidError,
    IdmRuntime,
    IdmStageError,
    ThinkingLevel,
    idm_generate,
    log_stage,
    record_deterministic_fallback,
    repair_thinking,
)
from app.idm.text import evidence_index
from app.idm.worksheet import COMPACTED_CODE as WORKSHEET_COMPACTED_CODE
from app.idm.worksheet import compact_worksheet, worksheet_fallback_html
from app.instructional_density import INSTRUCTIONAL_DENSITY_POLICY_VERSION
from app.lesson_author_orchestration_v2_provider import UnitGenerationContractV2
from app.ordered_learning_content import bind_provider_semantic_versions

_SLOT_RE: Final = re.compile(r"components\[(\d+)\]")
_COVERAGE_CODES: Final = frozenset({
    "COMPONENT_COVERAGE_MISSING", "COMPONENT_COVERAGE_INCOMPLETE", "INVALID_FACT_ID_ARRAY",
})
_MS: Final = 1000
# Log/prompt-safe tokens: a validator code and a JSON path (never provider text).
_SAFE_CODE_RE: Final = re.compile(r"^[A-Z][A-Z0-9_]{2,99}$")
_SAFE_PATH_RE: Final = re.compile(r"^[A-Za-z0-9_.\[\]]{1,160}$")
_MAX_LOGGED_DETAILS: Final = 16
_FIELD_OF_PATH_RE: Final = re.compile(r"^components\[\d+\]\.?")
# A worksheet that misses part of its structure keeps the provider slot for author review: the
# source-locked html fallback would replace the practice with plain explanation.
# An option copying the html before the question (QC course 364564, N3) is a review note once the repair
# could not fix it: the source-locked question would drop the provider's scenario practice. So are an
# ungrounded callout that cannot become a paragraph (N9), FAQ items that restate the html while too few
# others remain, an FAQ title that does not match its questions (N11) and a title that promises a framework
# the html does not list (N5): the source-locked rebuild would replace the whole teaching slot for a problem
# the author fixes in a minute.
_REVIEW_FIRST_CODES: Final = frozenset({WORKSHEET_INCOMPLETE_CODE, ANSWER_LEAK_CODE, CALLOUT_UNGROUNDED_CODE,
                                        FAQ_RESTATES_HTML_CODE, FAQ_TITLE_MISMATCH_CODE,
                                        FRAMEWORK_INCOMPLETE_CODE})
# Findings a deterministic step settles when the repair does not: their repair alone is a targeted one.
_TARGETED_REPAIR_CODES: Final = frozenset({FRAMEWORK_INCOMPLETE_CODE})
# FAQ findings settled by dropping the items concerned while the slot keeps two items.
_FAQ_PRUNE_CODES: Final = frozenset({FAQ_UNGROUNDED_CODE, FAQ_RESTATES_HTML_CODE})
_MAX_HINT_FACT_KEYS: Final = 24
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
    """Segment-based teaching budget (spec §7.7.3); never the 1.75x-source rule."""

    blocks = {treatment.block_id for slot in brief.components for treatment in slot.treatments}
    words = SEGMENT_WORD_BUDGET[brief.unit_segment] + EXTRA_WORDS_PER_MUST_KNOW_BLOCK * max(0, len(blocks) - 1)
    words = max(MIN_GENERATED_WORDS, min(MAX_GENERATED_WORDS, words))
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


class IdmUnitWriter:
    """State for one unit: contract, brief, deps, runtime and the evolving draft."""

    def __init__(self, contract: UnitGenerationContractV2, brief: IdmUnitBriefV1, deps: IdmUnitDeps,
                 runtime: IdmRuntime) -> None:
        self.contract, self.brief, self.deps, self.runtime = contract, brief, deps, runtime
        self.plans = [plan.model_dump(mode="json") for plan in contract.component_plan]
        self.expected = build_idm_expected(contract, brief, deps, runtime.locale)
        # Source-locked fallback content restates whole facts; the IDM teaching
        # budget applies to authored content only, so fallback validation drops it.
        self.expected_fallback = {**self.expected, "instructional_output_budget": None}
        self.text_by_id = {fact.fact_key: fact.fact_text for fact in contract.source_facts}
        self.owned_text = [" ".join(self.text_by_id[key] for key in slot.owned_fact_keys if key in self.text_by_id)
                           for slot in brief.components]
        # Everything the writer was given (SOURCE_FACTS + LESSON_CONTEXT_FACTS); numbers of the approved
        # plan (Must Do, purpose, practice sentences) may be repeated too.
        self.evidence = evidence_index(
            [*(fact.fact_text for fact in contract.source_facts),
             *(item.fact_text for item in brief.lesson_context_facts)],
            number_texts=[brief.lesson_objective, brief.unit_purpose, *brief.lesson_practice_sentences])
        # A unit or html slot title that names "N <items>" promises an overview of all N items (QC 364564, N5). The
        # items come from the unit's facts or, for an orientation unit whose blocks only introduce the framework,
        # from the server-written support item W4 attached with the names of every item (QC run 8de1c76b, Q3).
        teaching = [slot for slot in brief.components if slot.type == "html" and slot.role != "practice"]
        self.framework_support = framework_support_lines(item.brief for slot in teaching for item in slot.support_items)
        self.framework = build_promise(
            [contract.unit_title, *(slot.title for slot in teaching)],
            [*self.framework_support, *(fact.fact_text for fact in contract.source_facts)])
        self.framework_inserted = 0
        # The worksheet left out its worked example to fit the budget (Q2).
        self.worksheet_compacted = False
        self.codes: list[str] = []
        # Provider, time-budget and repair failures behind a fallback (codes only, never provider text).
        self.failure_codes: list[str] = []
        self.faq_items_dropped = 0
        # FAQ items dropped because their answer only repeated the html above them (N11).
        self.faq_items_restated = 0
        # FAQ items dropped because Node's workspace schema rejects them (not a grounding problem).
        self.faq_items_invalid = 0
        # Ungrounded callouts turned into plain paragraphs after the repair could not ground them (N9).
        self.callouts_converted = 0
        self.budget: dict[str, Any] = self.expected["instructional_output_budget"]
        self.min_html_chars = minimum_visible_chars(self.budget.get("source_content_chars"))
        # Deterministic html fixes applied before validation (counted, never content).
        self.html_fixes: Counter[str] = Counter()
        # Node's revision-0 acceptance rules: never return a unit Node rejects (run c2e5ac41).
        self.acceptance = acceptance_context(contract, brief)
        self.last_acceptance: list[AcceptanceFinding] = []
        self.acceptance_seen: list[str] = []

    # -- prompts ----------------------------------------------------------------------------
    def writer_prompt(self) -> str:
        brief = self.brief
        slots = [{
            "slot": f"c{index}", "type": slot.type, "role": slot.role, "title": slot.title,
            "support_items": [item.model_dump(mode="json") for item in slot.support_items],
            "practice": None if slot.practice is None else {
                "sentence": slot.practice.sentence, "criteria": slot.practice.criteria_fact_keys,
                "feedback_focus": slot.practice.feedback_focus.model_dump(mode="json"),
                "scenario_origin": slot.practice.scenario_origin,
            },
            "treatments": [item.model_dump(mode="json") for item in slot.treatments],
            "owned_fact_keys": slot.owned_fact_keys, "supporting_fact_keys": slot.supporting_fact_keys,
        } for index, slot in enumerate(brief.components)]
        return unit_writer_prompt(
            self.runtime.locale, course_title=brief.course_title, audience=brief.target_audience,
            lesson_title=brief.lesson_title, lesson_objective=brief.lesson_objective,
            practice_sentences=brief.lesson_practice_sentences, previous_title=brief.previous_lesson_title,
            next_title=brief.next_lesson_title,
            unit_brief={"unit_title": self.contract.unit_title, "segment": brief.unit_segment,
                        "purpose": brief.unit_purpose, "slots": slots,
                        "job_aid_signpost": brief.job_aid_signpost},
            facts=[(fact.fact_key, fact.fact_text, None) for fact in self.contract.source_facts],
            context_facts=[(item.fact_key, item.fact_text, None) for item in brief.lesson_context_facts],
            html_rules=html_slot_rules(max_words=self.budget.get("max_words"),
                                       max_chars=self.budget.get("max_visible_chars"),
                                       min_chars=self.min_html_chars)
            if any(slot.type == "html" for slot in brief.components) else "",
        )

    def html_violations(self, component: Any) -> list[HtmlRuleViolation]:
        if not isinstance(component, dict) or component.get("type") != "html":
            return []
        return html_rule_violations(component.get("semantic_content"), budget=self.budget,
                                    min_chars=self.min_html_chars)

    def rule_lines(self, unit: dict[str, Any], issues: Sequence[tuple[str, int]],
                   locations: Mapping[tuple[str, int], str]) -> list[str]:
        """Precise failing rules per slot for the repair prompt (codes, locations, server numbers)."""

        components = unit.get("components", [])
        lines: list[str] = []
        for index in sorted({index for _code, index in issues}):
            slot = f"c{index}"
            component = components[index] if 0 <= index < len(components) else None
            violations = self.html_violations(component)
            only_density = bool(violations) and all(item.code == DENSITY_CODE for item in violations)
            lines.extend(html_violation_line(slot, item, only_density=only_density) for item in violations)
            explained = {item.code for item in violations}
            if any(item.structural for item in violations):
                explained.add("HTML_SEMANTIC_INVALID")
            if any(item.code == "HTML_PRESENTATION_MARKUP" for item in violations):
                explained.add("HTML_PRESENTATION_FORBIDDEN")
            acceptance_lines = self.acceptance_rule_lines(index)
            lines.extend(acceptance_lines)
            if acceptance_lines:
                explained.update(item.code for item in self.last_acceptance if item.component_index == index)
            for code, issue_index in issues:
                if issue_index == index and code not in explained:
                    if code in _FAQ_PRUNE_CODES and isinstance(component, dict):
                        lines.append(self.faq_rule_line(slot, index, component, code, unit))
                    elif code == CALLOUT_UNGROUNDED_CODE and isinstance(component, dict):
                        lines.append(self.callout_rule_line(slot, index, component))
                    elif code == FRAMEWORK_INCOMPLETE_CODE:
                        lines.append(self.framework_rule_line(slot, unit))
                    else:
                        field = _FIELD_OF_PATH_RE.sub("", locations.get((code, index), ""))
                        lines.append(rule_line(slot, code, f"{slot}.{field}" if field else slot))
                    explained.add(code)
        return lines

    def faq_rule_line(self, slot: str, index: int, component: dict[str, Any], code: str = FAQ_UNGROUNDED_CODE,
                      unit: dict[str, Any] | None = None) -> str:
        """Targeted repair: which items to rewrite and which facts they may use (keys, never text)."""

        found = (restated_faq_items(component, html_before(unit or {}, index)) if code == FAQ_RESTATES_HTML_CODE
                 else ungrounded_faq_items(component, self.evidence))
        items = ", ".join(f"{slot}.items[{item}]" for item in found)
        return rule_line(slot, code, items or slot) + self.fact_hint(index)

    def callout_rule_line(self, slot: str, index: int, component: dict[str, Any]) -> str:
        """Targeted repair of the callouts the facts do not support (QC course 364564, N9)."""

        blocks = ", ".join(f"{slot}.semantic_content.sections[{section}].blocks[{block}]"
                           for section, block in ungrounded_callouts(component, self.evidence))
        return rule_line(slot, CALLOUT_UNGROUNDED_CODE, blocks or slot) + self.fact_hint(index)

    def framework_rule_line(self, slot: str, unit: dict[str, Any]) -> str:
        """The promised and the listed item counts (server-measured numbers, never the item names)."""

        line = rule_line(slot, FRAMEWORK_INCOMPLETE_CODE, slot)
        if self.framework is None:
            return line
        components = [item for item in unit.get("components", []) if isinstance(item, dict)]
        teaching = [component for index, component in enumerate(components) if component.get("type") == "html"
                    and not (index < len(self.brief.components) and self.brief.components[index].role == "practice")]
        shown = framework_coverage(self.framework, teaching, html_before(unit, len(components)))
        where = (" named in the support item of the teaching html slot in UNIT_BRIEF" if self.framework_support
                 else " numbered in SOURCE_FACTS" if self.framework.labels else "")
        return line + (f" (the title promises {self.framework.count} items; the html shows {shown}; list every "
                       f"item{where} in one bullets block)")

    def fact_hint(self, index: int) -> str:
        brief_slot = self.brief.components[index] if index < len(self.brief.components) else None
        keys = list(dict.fromkeys([*brief_slot.owned_fact_keys, *brief_slot.supporting_fact_keys]))[
            :_MAX_HINT_FACT_KEYS] if brief_slot else []
        listed = ", ".join(f"[{key}]" for key in keys)
        return f" Start from this slot's facts: {listed}." if keys else ""

    # -- acceptance --------------------------------------------------------------------------
    def normalize_component(self, index: int, component: Any) -> Any:
        """Deterministic, meaning-preserving fixes before validation: html (``app.idm.html_rules``) and the
        seeded option order of a single-choice question (``app.idm.mcq``, QC course 364564 N2)."""

        plan_type = self.plans[index]["type"] if 0 <= index < len(self.plans) else None
        if plan_type == "problem" and isinstance(component, dict):
            normalized, codes = normalize_single_choice(component, str(self.plans[index]["component_plan_id"]))
            self.runtime.adjustments.update(codes)
            return normalized
        if plan_type != "html" or not isinstance(component, dict) or not isinstance(
                component.get("semantic_content"), dict):
            return component
        semantic, fixes = normalize_html_semantic(component["semantic_content"])
        if not fixes:
            return component
        self.html_fixes.update(fixes)
        self.runtime.adjustments.update({f"w5_html_{key}": count for key, count in fixes.items()})
        return {**component, "semantic_content": semantic}

    def bind(self, text: str) -> dict[str, Any]:
        try:
            value = _strip_learning_block_ids(bind_provider_semantic_versions(json.loads(text)))
            unit = self.deps.bind_instance_payload(value, self.expected)
        except self.deps.recoverable_errors as error:
            raise IdmResponseInvalidError("IDM_W5_INSTANCE_INVALID", safe_error_details(error)) from error
        components = [self.normalize_component(index, component)
                      for index, component in enumerate(unit.get("components", []))]
        return {**unit, "components": components, "component_plan": list(self.plans)}

    def decode_repair(self, text: str, unit: dict[str, Any], targets: list[int],
                      coverage: list[int]) -> dict[str, Any]:
        """The addressed slot payloads of a repair answer, prepared exactly like a writer answer.

        The repair wire schema hides the server-owned ``semantic_content.version``; without the
        stamp every html repair failed the shared reader (run e869f43a: 4 x IDM_W5_REPAIR_INVALID).
        """

        delta: dict[str, Any] = _strip_learning_block_ids(bind_provider_semantic_versions(
            self.deps.decode_repair(text, unit, targets, coverage, {})))
        changes = delta.get("components") if isinstance(delta, dict) else None
        if isinstance(changes, list):
            delta = {**delta, "components": [
                self.normalize_component(change["component_index"], change)
                if isinstance(change, dict) and type(change.get("component_index")) is int else change
                for change in changes]}
        return delta

    def delta_details(self, delta: Any) -> list[dict[str, Any]]:
        """Safe html rule codes and paths of the slots a rejected repair answer addressed."""

        changes = delta.get("components") if isinstance(delta, dict) else None
        details: list[dict[str, Any]] = []
        for change in changes if isinstance(changes, list) else []:
            index = change.get("component_index") if isinstance(change, dict) else None
            if type(index) is not int or not 0 <= index < len(self.plans) or "semantic_content" not in change:
                continue
            component = {"type": self.plans[index]["type"], "semantic_content": change["semantic_content"]}
            details.extend({"type": item.code, "loc": [f"components[{index}].{item.location}"]}
                           for item in self.html_violations(component))
        return details[:_MAX_LOGGED_DETAILS]

    def log_findings(self, answer: str, draft: _Draft, finding: UnitFinding | None,
                     idm: Sequence[tuple[str, int]]) -> None:
        """One safe line per accepted-but-invalid writer/repair answer: codes, paths, numbers."""

        slots = sorted({index for _code, index in idm} | ({slot} if (slot := _slot_of(finding)) is not None else set()))
        components = draft.unit.get("components", [])
        html_rules = [item.as_log(index) for index in slots if 0 <= index < len(components)
                      for item in self.html_violations(components[index])]
        log_stage("idm_unit_validation", {
            "correlation_id": self.runtime.correlation_id, "unit_path": self.contract.unit_path, "answer": answer,
            "finding": None if finding is None else f"{finding.code}@{_safe_path(finding.path) or ''}",
            "idm_findings": [f"{code}@components[{index}]" for code, index in idm],
            "html_rules": html_rules[:_MAX_LOGGED_DETAILS], "html_fixes": dict(sorted(self.html_fixes.items())),
            "node_acceptance": acceptance_log(self.last_acceptance, _MAX_LOGGED_DETAILS),
            "fallback_slots": list(draft.fallback_slots),
        })

    def problems(self, draft: _Draft) -> tuple[UnitFinding | None, list[tuple[str, int]]]:
        """First shared-validator finding plus IDM slot findings (fallback slots exempt)."""

        expected = self.expected_fallback if draft.fallback_slots else self.expected
        finding: UnitFinding | None = self.deps.validate_unit(draft.unit, expected)
        self.last_acceptance = []
        if finding is None:
            # Node rejects the whole unit on the first of these; each is settled like a shared-validator finding.
            self.last_acceptance = self.node_acceptance(draft)
            finding = NodeAcceptanceUnitFinding.of(self.last_acceptance[0]) if self.last_acceptance else None
        exempt = {*draft.fallback_slots, *draft.review_slots}
        idm = [(item.code, item.component_index)
               for item in deterministic_slot_findings(draft.unit, self.brief, self.owned_text, self.evidence,
                                                       self.framework)
               if item.component_index not in exempt]
        return finding, idm

    def node_acceptance(self, draft: _Draft, *, whole_fallback: bool = False) -> list[AcceptanceFinding]:
        """Node acceptance findings of ``draft`` (budget only on provider content, as Node measures it)."""

        findings = node_acceptance_findings(draft.unit, self.acceptance,
                                            provider_validated=not whole_fallback and not draft.fallback_slots)
        self.acceptance_seen.extend(item for item in acceptance_log(findings) if item not in self.acceptance_seen)
        return findings

    def acceptance_rule_lines(self, index: int) -> list[str]:
        """Repair lines for the Node rules a slot breaks (the finer detail code where Node's code is coarse)."""

        slot = f"c{index}"
        lines: list[str] = []
        for item in self.last_acceptance:
            if item.component_index != index:
                continue
            location = ", ".join(f"{slot}.items[{position}]" for position in item.items) or slot
            lines.append(rule_line(slot, item.detail or item.code, location))
        return lines

    def prune_items(self, draft: _Draft, index: int, finding: AcceptanceFinding) -> _Draft | None:
        """Drop the FAQ items Node would reject while the slot keeps its minimum item count."""

        components = list(draft.unit.get("components", []))
        component = components[index] if 0 <= index < len(components) else None
        items = component.get("items") if isinstance(component, dict) else None
        if (not isinstance(component, dict) or component.get("type") != "la_faq" or not isinstance(items, list)
                or not finding.items):
            return None
        drop = set(finding.items)
        keep = [item for position, item in enumerate(items) if position not in drop]
        if len(keep) < IDM_FAQ_MIN_ITEMS:
            return None
        components[index] = {**component, "items": keep}
        self.faq_items_invalid += len(items) - len(keep)
        self.runtime.adjustments["w5_faq_items_invalid_dropped"] += len(items) - len(keep)
        return _Draft({**draft.unit, "components": components}, list(draft.fallback_slots), draft.repair_applied,
                      list(draft.review_slots), dict(draft.reasons))

    # -- provider steps -----------------------------------------------------------------------
    async def write(self, repair: str = "", thinking: ThinkingLevel = THINKING_W5) -> dict[str, Any]:
        return await idm_generate(
            self.runtime, stage="idm_w5_writer", prompt=self.writer_prompt() + repair,
            response_schema=self.deps.build_instance_model(self.plans), parse=self.bind,
            max_output_tokens=IDM_UNIT_WRITER_MAX_OUTPUT_TOKENS, thinking_level=thinking,
            invocation_kind="writer" if not repair else "repair",
        )

    async def repair_slots(self, draft: _Draft, issues: Sequence[tuple[str, int]],
                           locations: Mapping[tuple[str, int], str] | None = None,
                           thinking: ThinkingLevel = THINKING_W5) -> _Draft:
        targets = sorted({index for _code, index in issues})
        coverage = sorted({index for code, index in issues if code in _COVERAGE_CODES})
        model = self.deps.build_repair_model(draft.unit, targets, coverage)

        def parse(text: str) -> dict[str, Any]:
            delta: Any = None
            try:
                delta = self.decode_repair(text, draft.unit, targets, coverage)
                return self.deps.merge_repair(draft.unit, delta, targets, coverage)
            except self.deps.recoverable_errors as error:
                details = (safe_error_details(error) + self.delta_details(delta))[:_MAX_LOGGED_DETAILS]
                raise IdmResponseInvalidError("IDM_W5_REPAIR_INVALID", details) from error

        prompt = self.writer_prompt() + unit_repair_suffix(
            [{"code": code, "path": f"components[{index}]"} for code, index in issues],
            self.rule_lines(draft.unit, issues, locations or {}))
        unit = await idm_generate(
            self.runtime, stage="idm_w5_repair", prompt=prompt, response_schema=model, parse=parse,
            max_output_tokens=IDM_UNIT_WRITER_MAX_OUTPUT_TOKENS, thinking_level=thinking,
            invocation_kind="repair",
        )
        return _Draft({**unit, "component_plan": list(self.plans)}, list(draft.fallback_slots), True,
                      list(draft.review_slots), dict(draft.reasons))

    def fallback_component(self, index: int) -> dict[str, Any] | None:
        slots = self.deps.source_locked_components
        if slots is None:
            source = self.deps.source_locked_unit
            slots = source.get("components", []) if source is not None else []
        component = slots[index] if 0 <= index < len(slots) else None
        return dict(component) if component is not None else None

    def is_worksheet(self, index: int) -> bool:
        slot = self.brief.components[index] if 0 <= index < len(self.brief.components) else None
        return slot is not None and slot.role == "practice" and slot.type == WORKSHEET_COMPONENT_TYPE

    def worksheet_fallback(self, index: int) -> dict[str, Any] | None:
        """The source-locked component of a worksheet slot as a structured form (QC run 8de1c76b, Q2): the
        fields in one table, the closing lines and a self-check list; ``None`` when the facts hold no form."""

        base = self.fallback_component(index)
        if base is None or not self.is_worksheet(index):
            return None
        plan = self.plans[index]
        keys = list(dict.fromkeys([*plan["source_fact_ids"], *plan["supporting_evidence_fact_ids"]]))
        practice = self.brief.components[index].practice
        structured = worksheet_fallback_html(str(base.get("title") or self.contract.unit_title),
                                             [self.text_by_id[key] for key in keys if key in self.text_by_id],
                                             practice.sentence if practice else None, self.runtime.locale)
        return {**base, "html": structured} if structured else None

    def fallback_slot(self, draft: _Draft, index: int, reasons: Sequence[str] = ()) -> _Draft | None:
        component = self.fallback_component(index)
        if component is None:
            return None
        components = list(draft.unit["components"])
        worksheet = self.worksheet_fallback(index)
        if worksheet is not None:
            components[index] = worksheet
            candidate = _Draft({**draft.unit, "components": components}, sorted({*draft.fallback_slots, index}))
            finding, _idm = self.problems(candidate)
            # The structured form is used only when the validators accept it in that slot.
            if _slot_of(finding) != index:
                component = worksheet
                self.runtime.adjustments["w5_worksheet_structured_fallback"] += 1
        components[index] = component
        record_deterministic_fallback(self.runtime, stage="idm_w5_slot", code="IDM_W5_SLOT_FALLBACK")
        return _Draft({**draft.unit, "components": components}, sorted({*draft.fallback_slots, index}),
                      draft.repair_applied, [slot for slot in draft.review_slots if slot != index],
                      {**draft.reasons, index: list(dict.fromkeys(reasons))})

    def keep_for_review(self, draft: _Draft, index: int, reasons: Sequence[str]) -> _Draft:
        self.codes.extend(reasons)
        return _Draft(draft.unit, list(draft.fallback_slots), draft.repair_applied,
                      sorted({*draft.review_slots, index}), {**draft.reasons, index: list(dict.fromkeys(reasons))})

    def prune_faq(self, draft: _Draft, index: int) -> _Draft | None:
        """Drop the FAQ items the facts do not support or that repeat the html above them (N11) while the
        slot keeps its minimum item count."""

        components = list(draft.unit.get("components", []))
        component = components[index] if 0 <= index < len(components) else None
        items = component.get("items") if isinstance(component, dict) else None
        if not isinstance(component, dict) or component.get("type") != "la_faq" or not isinstance(items, list):
            return None
        ungrounded = set(ungrounded_faq_items(component, self.evidence))
        restated = set(restated_faq_items(component, html_before(draft.unit, index))) - ungrounded
        drop = ungrounded | restated
        keep = [item for position, item in enumerate(items) if position not in drop]
        if not drop or len(keep) < IDM_FAQ_MIN_ITEMS:
            return None
        components[index] = {**component, "items": keep}
        self.faq_items_dropped += len(ungrounded)
        self.faq_items_restated += len(restated)
        self.codes.append(FAQ_ITEMS_DROPPED_CODE)
        for key, count in (("w5_faq_items_dropped", len(ungrounded)), ("w5_faq_items_restated_dropped", len(restated))):
            if count:
                self.runtime.adjustments[key] += count
        return _Draft({**draft.unit, "components": components}, list(draft.fallback_slots), draft.repair_applied,
                      list(draft.review_slots), dict(draft.reasons))

    def compact_worksheet(self, draft: _Draft, index: int) -> _Draft | None:
        """An over-budget worksheet without its worked example (then with one-sentence guidance cells), when that
        fits every html rule and Node's budget and the worksheet stays complete (QC run 8de1c76b, Q2)."""

        components = list(draft.unit.get("components", []))
        component = components[index] if 0 <= index < len(components) else None
        if not isinstance(component, dict) or not self.is_worksheet(index):
            return None
        compacted = compact_worksheet(component, lambda value: not self.html_violations(value))
        if compacted is None:
            return None
        components[index] = compacted
        candidate = _Draft({**draft.unit, "components": components}, list(draft.fallback_slots),
                           draft.repair_applied, list(draft.review_slots), dict(draft.reasons))
        finding, idm = self.problems(candidate)
        if _slot_of(finding) == index or any(slot == index for _code, slot in idm):
            return None
        self.worksheet_compacted = True
        self.codes.append(WORKSHEET_COMPACTED_CODE)
        self.runtime.adjustments["w5_worksheet_compacted"] += 1
        return candidate

    def insert_framework_list(self, draft: _Draft, index: int) -> _Draft | None:
        """The teaching html with one more section listing every promised item by name (QC run 8de1c76b, Q3), when
        the names are known (unit facts or the W4 support item) and the slot stays within every html rule."""

        if self.framework is None or len(self.framework.labels) < self.framework.count:
            return None
        components = list(draft.unit.get("components", []))
        component = components[index] if 0 <= index < len(components) else None
        if not isinstance(component, dict) or component.get("type") != "html":
            return None
        listed = with_framework_list(component, self.framework, self.runtime.locale)
        if listed is None or self.html_violations(listed):
            return None
        components[index] = listed
        candidate = _Draft({**draft.unit, "components": components}, list(draft.fallback_slots),
                           draft.repair_applied, list(draft.review_slots), dict(draft.reasons))
        finding, idm = self.problems(candidate)
        if _slot_of(finding) == index or (FRAMEWORK_INCOMPLETE_CODE, index) in idm:
            return None
        self.framework_inserted += len(self.framework.labels)
        self.codes.append(FRAMEWORK_LIST_INSERTED_CODE)
        self.runtime.adjustments["w5_framework_list_inserted"] += 1
        return candidate

    def callouts_to_prose(self, draft: _Draft, index: int) -> _Draft | None:
        """Turn the callouts the facts do not support into plain paragraphs (QC course 364564, N9), when the
        slot stays valid that way; the text no longer reads as a quotation and the note asks the SME."""

        components = list(draft.unit.get("components", []))
        component = components[index] if 0 <= index < len(components) else None
        if not isinstance(component, dict):
            return None
        positions = ungrounded_callouts(component, self.evidence)
        converted = callouts_as_paragraphs(component, positions)
        if not positions or self.html_violations(converted):
            return None
        components[index] = converted
        candidate = _Draft({**draft.unit, "components": components}, list(draft.fallback_slots),
                           draft.repair_applied, list(draft.review_slots), dict(draft.reasons))
        # Another IDM finding of the slot is settled by the next pass; a validator or Node finding means the
        # paragraph form is not acceptable here.
        finding, idm = self.problems(candidate)
        if _slot_of(finding) == index or (CALLOUT_UNGROUNDED_CODE, index) in idm:
            return None
        self.callouts_converted += len(positions)
        self.codes.append(CALLOUT_TO_PROSE_CODE)
        self.runtime.adjustments["w5_callouts_to_prose"] += len(positions)
        return candidate

    async def settle(self, draft: _Draft) -> _Draft | None:
        """Repair once, then replace failing slots with the source-locked fallback.

        A slot without a source-locked rebuild that only fails IDM pedagogy checks is kept for
        author review (``review_slots``); a shared-validator failure there still discards the draft.
        """

        finding, idm = self.problems(draft)
        if finding is None and not idm:
            return draft
        self.log_findings("writer", draft, finding, idm)
        # An over-budget worksheet is compacted before any repair call (QC run 8de1c76b, Q2: 550 then 452 words
        # against 400, then the raw source fallback); the repair handles what compaction cannot.
        if finding is not None and finding.code == DENSITY_CODE and (slot := _slot_of(finding)) is not None:
            compacted = self.compact_worksheet(draft, slot)
            if compacted is not None:
                self.codes.append(finding.code)
                draft = compacted
                finding, idm = self.problems(draft)
                if finding is None and not idm:
                    return draft
        self.codes.extend([finding.code] if finding is not None else [])
        self.codes.extend(code for code, _index in idm)
        issues = list(idm)
        slot = _slot_of(finding)
        if finding is not None and slot is None:
            return None
        locations: dict[tuple[str, int], str] = {}
        if finding is not None and slot is not None:
            issues.append((finding.code, slot))
            locations[(finding.code, slot)] = _safe_path(finding.path) or ""
        # Every slot Node would reject is repaired in the same call, not only the first one.
        issues.extend(dict.fromkeys((item.code, item.component_index) for item in self.last_acceptance[1:]
                                    if item.component_index is not None
                                    and (item.code, item.component_index) not in issues))
        # A failed repair (budget, provider, invalid answer) falls through to slot fallback. A repair of findings a
        # deterministic step settles anyway (a missing framework list, a length cue) is a targeted one: low thinking,
        # and none at all when too little time is left for it (QC run 8de1c76b, Q3: the framework repair of
        # "Bản đồ 5 chuyển dịch" timed out at 52.6 s against the 120 s unit deadline).
        targeted = finding is None and {code for code, _index in issues} <= _TARGETED_REPAIR_CODES
        repaired = False
        if targeted and self.runtime.remaining_seconds() < IDM_TARGETED_REPAIR_MIN_SECONDS:
            self.runtime.adjustments["w5_targeted_repair_skipped_budget"] += 1
        else:
            try:
                draft = await self.repair_slots(draft, issues, locations,
                                                thinking=THINKING_TARGETED_REPAIR if targeted else THINKING_W5)
                repaired = True
            except (IdmBudgetError, IdmResponseInvalidError) as error:
                self.failure_codes.append(error.code)  # falls through to the per-slot fallback below
            except IdmProviderError as error:
                if error.terminal:
                    raise
                self.failure_codes.append(error.code)
        # Each pass settles one step of one slot (Node item pruning, FAQ pruning or callouts to prose, then
        # fallback or review), so the bound never limits.
        for attempt in range(3 * len(self.plans) + 1):
            finding, idm = self.problems(draft)
            if finding is None and not idm:
                return draft
            if repaired and attempt == 0:
                self.log_findings("repair", draft, finding, idm)
            bad = _slot_of(finding) if finding is not None else idm[0][1]
            if bad is None or bad in draft.fallback_slots:
                return None
            reasons = [*([finding.code] if finding is not None else []),
                       *(code for code, index in idm if index == bad)]
            if isinstance(finding, NodeAcceptanceUnitFinding):
                # A safe deterministic alternative first: drop only the FAQ items Node would reject.
                pruned = self.prune_items(draft, bad, finding.acceptance)
                if pruned is not None:
                    self.codes.append(finding.code)
                    draft = pruned
                    continue
            if finding is None:
                # Still-ungrounded FAQ answers (R6) and answers that repeat the html above (N11) are dropped
                # while the slot keeps two items.
                pruned = self.prune_faq(draft, bad) if set(reasons) & _FAQ_PRUNE_CODES else None
                if pruned is not None:
                    draft = pruned
                    continue
                # Still-ungrounded callouts become plain paragraphs (N9).
                converted = self.callouts_to_prose(draft, bad) if CALLOUT_UNGROUNDED_CODE in reasons else None
                if converted is not None:
                    draft = converted
                    continue
                # A framework the html still does not list gets the item names as a list (Q3).
                listed = self.insert_framework_list(draft, bad) if FRAMEWORK_INCOMPLETE_CODE in reasons else None
                if listed is not None:
                    draft = listed
                    continue
                if set(reasons) <= _REVIEW_FIRST_CODES:
                    draft = self.keep_for_review(draft, bad, reasons)
                    continue
            compacted = (self.compact_worksheet(draft, bad)
                         if finding is not None and finding.code == DENSITY_CODE else None)
            if compacted is not None:
                draft = compacted
                continue
            replaced = self.fallback_slot(draft, bad, reasons)
            if replaced is None:
                if finding is not None:
                    return None
                # No source-locked rebuild exists for this slot (spec §7.7.4 "returns None") and the
                # shared validator accepts its payload: keep the provider slot as a reviewable draft
                # instead of discarding the whole unit (which would end in FALLBACK_INVALID).
                draft = self.keep_for_review(draft, bad, reasons)
                continue
            draft = replaced
        return None


def _whole_fallback(writer: IdmUnitWriter) -> dict[str, Any]:
    source = writer.deps.source_locked_unit
    worksheets = {index: component for index in range(len(writer.plans))
                  if (component := writer.worksheet_fallback(index)) is not None}
    if source is not None and worksheets:
        # A worksheet slot keeps its form structure in the whole-unit fallback too, when the validators accept it.
        structured = {**source, "components": [worksheets.get(index, component)
                                               for index, component in enumerate(source.get("components", []))]}
        if writer.deps.validate_unit(structured, writer.expected_fallback) is None and not writer.node_acceptance(
                _Draft(dict(structured), list(range(len(writer.plans)))), whole_fallback=True):
            source = structured
    if source is None or writer.deps.validate_unit(source, writer.expected_fallback) is not None:
        raise IdmStageError(_FALLBACK_INVALID)
    rejected = writer.node_acceptance(_Draft(dict(source), list(range(len(writer.plans)))), whole_fallback=True)
    if rejected:
        _log_node_rejection(writer, "whole_fallback", rejected)
        raise IdmStageError(_FALLBACK_INVALID)
    record_deterministic_fallback(writer.runtime, stage="idm_w5_unit", code="IDM_W5_UNIT_FALLBACK")
    return dict(source)


def _log_node_rejection(writer: IdmUnitWriter, stage: str, findings: Sequence[AcceptanceFinding]) -> None:
    """The same check/code/path Node would log as ``acceptance_*`` (never content)."""

    log_stage("idm_unit_node_acceptance", {
        "correlation_id": writer.runtime.correlation_id, "unit_path": writer.contract.unit_path, "stage": stage,
        "findings": acceptance_log(findings, _MAX_LOGGED_DETAILS),
    })


async def run_idm_unit(
    *,
    contract: UnitGenerationContractV2,
    runtime: IdmRuntime,
    deps: IdmUnitDeps,
    fallback_only: bool,
) -> dict[str, Any]:
    """Return the ``/unit`` response for an IDM unit in the legacy envelope shape."""

    started = time.perf_counter()
    brief = parse_brief(contract)
    writer = IdmUnitWriter(contract, brief, deps, runtime)
    ai_drafted = any(slot.practice is not None and slot.practice.scenario_origin == "ai_drafted"
                     for slot in brief.components)
    draft: _Draft | None = None
    judge = JudgeOutcome("not_run")
    if fallback_only:
        writer.failure_codes.append("IDM_W5_FALLBACK_ONLY")
    else:
        try:
            async with asyncio.timeout(max(0.001, runtime.remaining_seconds())):
                draft = await _provider_draft(writer)
        except TimeoutError:
            writer.failure_codes.append("IDM_DEADLINE_EXCEEDED")
            draft = None
        except IdmBudgetError as error:
            writer.failure_codes.append(error.code)
            draft = None
        except IdmProviderError as error:
            # A definitive rejection (credentials, request) must reach Node's fast-fail path.
            if error.terminal:
                raise
            writer.failure_codes.append(error.code)
            draft = None
        if draft is not None:
            # QA runs on its own clock: running out of time never discards the accepted unit.
            try:
                async with asyncio.timeout(max(0.001, runtime.remaining_seconds())):
                    draft, judge = await _judge_and_repair(writer, draft, deps.judge_mode)
            except TimeoutError:
                judge = JudgeOutcome("skipped_budget")
    if draft is not None and (rejected := writer.node_acceptance(draft)):
        # Unreachable while settle() applies the same rules; never hand Node a unit it rejects.
        _log_node_rejection(writer, "final", rejected)
        writer.failure_codes.append("IDM_NODE_ACCEPTANCE_REJECTED")
        draft = None
    whole_fallback = draft is None
    if draft is None:
        draft = _Draft(_whole_fallback(writer), list(range(len(writer.plans))))
    usage_complete = runtime.usage.complete and runtime.provider_failure_code is None
    if whole_fallback:
        usage: dict[str, int] = {"inputTokens": 0, "outputTokens": 0, "embeddingTokens": 0, "totalTokens": 0}
        usage_complete = fallback_only
        usage_source = "deterministic_fallback" if fallback_only else "reserved_upper_bound"
    else:
        usage = runtime.usage.as_usage()
        usage_source = "provider" if usage_complete else "reserved_upper_bound"
    # Node admits the validated lane only for provider-accounted content: a draft kept after a failed
    # repair or judge call (reserved upper-bound accounting) is a reviewable structured draft.
    provider_lane = usage_source == "provider" and not whole_fallback and not draft.fallback_slots
    # A callout turned into a paragraph still states what the facts do not (N9): the SME confirms it.
    reviewable = (not provider_lane or bool(draft.review_slots) or writer.callouts_converted > 0
                  or writer.framework_inserted > 0
                  or deps.evidence_review_required
                  or blocking_count(judge.findings) > 0
                  or (ai_drafted and _q5(judge) != "pass"))
    # N8: "still warns" is recomputed on the returned unit; codes settled on the way are listed apart.
    remaining = [] if whole_fallback else final_unit_findings(draft.unit, brief, writer.owned_text, writer.evidence,
                                                              draft.fallback_slots, framework=writer.framework)
    fixed = [] if whole_fallback else settled_codes(
        writer.codes, remaining, {code for index in draft.fallback_slots for code in draft.reasons.get(index, ())})
    note = build_unit_author_note(
        locale=runtime.locale, judge=judge, deterministic_codes=writer.codes, fallback_slots=draft.fallback_slots,
        ai_drafted=ai_drafted, slot_reasons=draft.reasons, review_slots=draft.review_slots,
        slot_types=[plan["type"] for plan in writer.plans], failure_codes=writer.failure_codes,
        whole_fallback=whole_fallback, faq_items_dropped=writer.faq_items_dropped,
        faq_items_invalid=writer.faq_items_invalid, remaining=remaining, fixed_codes=fixed,
        faq_items_restated=writer.faq_items_restated, callouts_to_prose=writer.callouts_converted,
        framework_list_inserted=writer.framework_inserted, worksheet_compacted=writer.worksheet_compacted)
    quality = build_unit_quality(mode=deps.judge_mode, judge=judge, repair_applied=draft.repair_applied,
                                 deterministic_codes=[*writer.codes, *(item.code for item in remaining)],
                                 author_note=note)
    unit = {**draft.unit, "idm_quality": quality.model_dump(mode="json")}
    log_stage("idm_unit_quality", {
        "correlation_id": runtime.correlation_id, "unit_path": contract.unit_path,
        "judge_status": judge.status, "finding_counts": quality.finding_counts.model_dump(),
        "repair_applied": draft.repair_applied, "fallback_slots": draft.fallback_slots,
        "review_slots": draft.review_slots, "html_fixes": dict(sorted(writer.html_fixes.items())),
        "whole_fallback": whole_fallback, "codes": sorted(set(writer.codes)),
        "remaining_codes": sorted({item.code for item in remaining}), "fixed_codes": sorted(set(fixed)),
        "failure_codes": sorted(set(writer.failure_codes)), "faq_items_dropped": writer.faq_items_dropped,
        "faq_items_invalid": writer.faq_items_invalid, "faq_items_restated": writer.faq_items_restated,
        "callouts_to_prose": writer.callouts_converted, "framework_list_inserted": writer.framework_inserted,
        "slot_reasons": {str(index): codes for index, codes in sorted(draft.reasons.items())},
        "node_acceptance": writer.acceptance_seen[:_MAX_LOGGED_DETAILS],
        "call_count": runtime.usage.calls, "duration_ms": int((time.perf_counter() - started) * _MS),
    })
    return {
        "contract_version": 2, "source_snapshot_hash": contract.source_snapshot_hash,
        "unit_path": contract.unit_path, "unit": unit, "usage": usage, "usage_complete": usage_complete,
        "usage_source": usage_source,
        "content_origin": "provider_validated" if provider_lane else "structured_fallback",
        "quality_state": "review_required" if reviewable else "validated",
        "attempt_trace": runtime.trace,
    }


def _q5(judge: JudgeOutcome) -> str:
    severities = [finding.severity for finding in judge.findings if finding.criterion == "Q5_grounded_criteria"]
    if judge.status in {"not_run", "skipped_budget", "failed"}:
        return "unknown"
    if not severities:
        return "unknown"
    return "pass" if all(severity == "pass" for severity in severities) else "fail"


async def _provider_draft(writer: IdmUnitWriter) -> _Draft | None:
    repair = ""
    thinking: ThinkingLevel = THINKING_W5
    for _attempt in (1, 2):
        try:
            unit = await writer.write(repair, thinking)
        except IdmResponseInvalidError as error:
            writer.codes.append(error.code)
            # A cut answer is retried shorter and with less thinking, never with the same prompt; any
            # other rejection names the validator codes and paths behind it (never the answer itself).
            details = [{"code": str(item.get("type")), "path": ".".join(str(part) for part in item.get("loc", []))}
                       for item in error.errors if isinstance(item, dict)]
            repair = (truncation_suffix(COMPACT_UNIT) if error.code == "IDM_RESPONSE_TRUNCATED"
                      else repair_suffix([{"code": error.code, "path": "components"}, *details]))
            thinking = repair_thinking(error.code, thinking)
            continue
        return await writer.settle(_Draft(unit, []))
    return None


async def _judge_and_repair(writer: IdmUnitWriter, draft: _Draft, mode: JudgeMode) -> tuple[_Draft, JudgeOutcome]:
    facts = [(key, writer.text_by_id[key]) for key in writer.contract.unit_source_fact_ids
             if key in writer.text_by_id]
    criteria = {key for slot in writer.brief.components if slot.practice
                for key in slot.practice.criteria_fact_keys}
    facts.extend((item.fact_key, item.fact_text) for item in writer.brief.lesson_context_facts
                 if item.fact_key in criteria)
    practice_slot = has_practice_slot(writer.brief)
    summary = {
        "unit_title": writer.contract.unit_title, "segment": writer.brief.unit_segment,
        "purpose": writer.brief.unit_purpose, "must_do": writer.brief.lesson_objective,
        # Only this unit's own practice: the lesson's practice sentences made the judge grade practice that
        # sits in another unit and reject teach-only units (QC course 364564, N7).
        "practices": [slot.practice.sentence for slot in writer.brief.components if slot.practice is not None],
        "has_practice_slot": practice_slot,
        "slots": [{"index": index, "type": slot.type, "role": slot.role, "title": slot.title,
                   "treatments": [item.model_dump(mode="json") for item in slot.treatments]}
                  for index, slot in enumerate(writer.brief.components)],
    }
    judge = await run_judge(writer.runtime, mode=mode, plan_summary=summary, facts=facts, unit=draft.unit,
                            practice_slot=practice_slot)
    targets = [index for index in repair_targets(judge.findings) if index not in draft.fallback_slots]
    if mode != "repair" or not targets:
        return draft, judge
    issues = [(f"IDM_W6_{finding.criterion.split('_')[0]}", finding.component_index)
              for finding in judge.findings
              if finding.component_index in targets and finding.severity in {"major", "critical"}]
    try:
        repaired = await writer.repair_slots(draft, [(code, index) for code, index in issues if index is not None])
    except (IdmBudgetError, IdmResponseInvalidError, IdmProviderError):
        return draft, judge
    finding, idm = writer.problems(repaired)
    if finding is not None or idm:
        return draft, judge
    second = await run_judge(writer.runtime, mode=mode, plan_summary=summary, facts=facts, unit=repaired.unit,
                             practice_slot=practice_slot)
    if second.status in {"pass", "review_required", "reject"} and (
            blocking_count(second.findings) < blocking_count(judge.findings)):
        return repaired, second
    return draft, judge
