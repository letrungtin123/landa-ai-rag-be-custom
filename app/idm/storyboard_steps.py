"""Deterministic steps of the W5 unit writer (``app.idm.storyboard``).

The state of one unit, the html, title and option fixes made before validation, the shared, Node and IDM
checks of a draft, and the settle steps that need no provider call: Node item pruning, FAQ pruning, worksheet
compaction, framework lists, callouts to prose, slot fallback, whole questions kept and acronyms explained.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from typing import Any

from app.content.ordered_learning_content import bind_provider_semantic_versions
from app.contracts.orchestration_v2.provider import UnitGenerationContractV2
from app.idm.acronyms import KNOWN_ABBREVIATIONS, AcronymGuide, abbreviations, explained, learner_texts, with_expansion
from app.idm.answer_checks import fact_table_rows
from app.idm.artifacts import clean_question, clean_semantic
from app.idm.budget import writer_thinking
from app.idm.contracts import IdmUnitBriefV1
from app.idm.faq_checks import FaqChecker, extra_recall_items, question_types
from app.idm.fidelity import brief_mode
from app.idm.fidelity_checks import FidelityChecker
from app.idm.framework import build_promise, framework_support_lines, is_framework_brief, with_framework_list
from app.idm.html_rules import (
    MAX_HEADING_CHARS,
    HtmlRuleViolation,
    html_rule_violations,
    minimum_visible_chars,
    normalize_html_semantic,
)
from app.idm.mcq import (
    free_key_letter,
    normalize_single_choice,
    practice_key_shift,
    question_complete,
    question_ordinal,
)
from app.idm.node_acceptance import AcceptanceFinding, acceptance_log, node_acceptance_findings
from app.idm.org_context import plan_terms
from app.idm.org_grounding import withheld_details
from app.idm.policy import FAQ_COMPONENT_TYPE, FAQ_DEEP_TYPES, IDM_FAQ_MIN_ITEMS, WORKSHEET_COMPONENT_TYPE
from app.idm.qa import (
    CALLOUT_TO_PROSE_CODE,
    CALLOUT_UNGROUNDED_CODE,
    FAQ_ITEMS_DROPPED_CODE,
    FRAMEWORK_INCOMPLETE_CODE,
    FRAMEWORK_LIST_INSERTED_CODE,
    callouts_as_paragraphs,
    deterministic_slot_findings,
    has_practice_slot,
    html_before,
    restated_faq_items,
    ungrounded_callouts,
)
from app.idm.qa_checks import CALLOUT_REPEAT_DROPPED, repeated_callouts, without_blocks
from app.idm.quiz_text import lay_out_quiz
from app.idm.runtime import IdmResponseInvalidError, IdmRuntime, ThinkingLevel, log_stage, record_deterministic_fallback
from app.idm.source_locked import FAQ_FALLBACK_REPEATED_CODE, faq_answers_repeat
from app.idm.storyboard_context import (
    _MAX_LOGGED_DETAILS,
    _MS,
    IdmUnitDeps,
    NodeAcceptanceUnitFinding,
    UnitFinding,
    _Draft,
    _safe_path,
    _slot_of,
    _strip_learning_block_ids,
    acceptance_context,
    build_idm_expected,
    safe_error_details,
)
from app.idm.text import evidence_index, is_suggested_criteria
from app.idm.titles import TitleCaser
from app.idm.worksheet import COMPACTED_CODE as WORKSHEET_COMPACTED_CODE
from app.idm.worksheet import compact_worksheet, label_suggested_self_check, worksheet_fallback_html


class IdmUnitWriterSteps:
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
        given = [*(fact.fact_text for fact in contract.source_facts),
                 *(item.fact_text for item in brief.lesson_context_facts)]
        self.evidence = evidence_index(
            given, number_texts=[brief.lesson_objective, brief.unit_purpose, *brief.lesson_practice_sentences])
        # QLT-5 C: the content mode of the unit and its checks of the teaching html slots. Names and numbers of the
        # approved plan's titles may be repeated; drafted plan text (support briefs, scenarios) adds none.
        self.mode = brief_mode(brief)
        # QLT-7: the server-written framework briefs (item names and descriptions taken from the facts) are plan text
        # too: the orientation unit of run fae66524 listed "SHIFT 4", "SHIFT 5" from them and got IDM_W5_NEW_FACT.
        server_briefs = [item.brief for slot in brief.components for item in slot.support_items
                         if is_framework_brief(item.brief)]
        # The audience is plan text as well: a drafted example now names the learner's role from it ("một CEO").
        plan_texts = [brief.course_title, brief.module_title, brief.lesson_title, brief.lesson_objective,
                      brief.unit_purpose, contract.unit_title, *(slot.title for slot in brief.components),
                      *server_briefs, brief.target_audience]
        # The organization context's glossary terms, product and learner-group names are plan text, never evidence;
        # its other codes and numbers the facts do not state are never known, even when the plan repeats them.
        background = plan_terms(brief.organization_context)
        withheld = withheld_details(brief.organization_context, given)
        self.fidelity = FidelityChecker.build(brief, self.text_by_id, self.evidence, locale=runtime.locale,
                                              plan_texts=plan_texts, background_texts=background, withheld=withheld)
        # QLT-5 D: the FAQ plan and the facts each FAQ answer is grounded in (its anchors, not the whole lesson).
        self.faq = FaqChecker.build(
            brief, {**{item.fact_key: item.fact_text for item in brief.lesson_context_facts}, **self.text_by_id},
            plan_texts=[*plan_texts, *brief.lesson_practice_sentences, *background], withheld=withheld)
        self.has_faq = any(slot.type == FAQ_COMPONENT_TYPE for slot in brief.components)
        # QLT-7: the table rows of the facts (a contested key is read against them too) and the sentence case of the
        # slot titles, keeping the proper names the facts and the plan write.
        self.fact_rows = fact_table_rows(given)
        self.titles = TitleCaser.from_texts([*given, *plan_texts])
        # QLT-8: the full forms the facts give ("SPD (Standard Product Dev)") and the server's short list.
        self.acronyms = AcronymGuide.build(given, runtime.locale)
        self.acronyms_explained = 0
        # A unit or html slot title that names "N <items>" promises an overview of all N items (QC 364564, N5). The
        # items come from the unit's facts or, for an orientation unit whose blocks only introduce the framework,
        # from the server-written support item W4 attached with the names of every item (QC run 8de1c76b, Q3).
        teaching = [slot for slot in brief.components if slot.type == "html" and slot.role != "practice"]
        self.framework_support = framework_support_lines(item.brief for slot in teaching for item in slot.support_items)
        # The items come from one group: the support item, or the facts of one source block (QC run ab8d67e1, R1).
        blocks: dict[str, list[str]] = {}
        for fact in contract.source_facts:
            blocks.setdefault(fact.scope_key, []).append(fact.fact_text)
        self.framework = build_promise([contract.unit_title, *(slot.title for slot in teaching)],
                                       [self.framework_support, *blocks.values()])
        self.framework_inserted = 0
        # The worksheet was shortened to fit the budget (Q2); QLT-9: whether its worked example stayed.
        self.worksheet_compacted = False
        self.worksheet_example_kept = False
        self.codes: list[str] = []
        # Provider, time-budget and repair failures behind a fallback (codes only, never provider text).
        self.failure_codes: list[str] = []
        self.faq_items_dropped = 0
        # FAQ items dropped because their answer only repeated the html above them (N11).
        self.faq_items_restated = 0
        # QLT-5 D: FAQ items dropped because they copied a source sentence, or asked for a fact again past the one
        # recall question allowed.
        self.faq_items_copied = 0
        self.faq_items_recall = 0
        # QLT-7: FAQ items dropped because they answered an open SME question.
        self.faq_items_open = 0
        # FAQ items dropped because Node's workspace schema rejects them (not a grounding problem).
        self.faq_items_invalid = 0
        # Ungrounded callouts turned into plain paragraphs after the repair could not ground them (N9).
        self.callouts_converted = 0
        # Slots whose source-locked FAQ repeats one sentence (logged once each).
        self.faq_fallback_repeated: set[int] = set()
        self.budget: dict[str, Any] = self.expected["instructional_output_budget"]
        self.min_html_chars = minimum_visible_chars(self.budget.get("source_content_chars"))
        # Deterministic html fixes applied before validation (counted, never content).
        self.html_fixes: Counter[str] = Counter()
        # Node's revision-0 acceptance rules: never return a unit Node rejects (run c2e5ac41).
        self.acceptance = acceptance_context(contract, brief)
        self.last_acceptance: list[AcceptanceFinding] = []
        self.acceptance_seen: list[str] = []

    def html_violations(self, component: Any) -> list[HtmlRuleViolation]:
        if not isinstance(component, dict) or component.get("type") != "html":
            return []
        return html_rule_violations(component.get("semantic_content"), budget=self.budget,
                                    min_chars=self.min_html_chars)

    # -- acceptance --------------------------------------------------------------------------
    def normalize_component(self, index: int, component: Any) -> Any:
        """Deterministic, meaning-preserving fixes before validation: html (``app.idm.html_rules``) and the
        seeded option order of a single-choice question (``app.idm.mcq``, QC course 364564 N2)."""

        plan = self.plans[index] if 0 <= index < len(self.plans) else None
        # A slot that owns no fact (a supporting question or FAQ) claims no coverage: the plan holds ownership, not
        # the answer (QC run 61068de6: SUPPORTING_COMPONENT_CLAIMS_OWNERSHIP on a question without a source-locked
        # rebuild ended the unit in FALLBACK_INVALID and the run in needs_action).
        if (plan is not None and plan.get("component_plan_id") and not plan.get("source_fact_ids")
                and isinstance(component, dict) and component.get("covered_source_fact_ids")):
            component = {**component, "covered_source_fact_ids": []}
            self.runtime.adjustments["w5_supporting_claim_cleared"] += 1
        plan_type = plan["type"] if plan is not None else None
        # QLT-7: the slot title in sentence case (Vietnamese), keeping acronyms and the facts' proper names.
        if (self.runtime.locale == "vi" and isinstance(component, dict) and isinstance(component.get("title"), str)
                and (titled := self.titles.title(component["title"])) != component["title"]):
            component = {**component, "title": titled}
            self.runtime.adjustments["w5_title_sentence_case"] += 1
        # PDF marks in learner text (QC run ab8d67e1, R5): "Row N:" labels and a repeated word go first.
        artefacts: Counter[str] = Counter()
        if plan_type in {"problem", "la_faq"} and isinstance(component, dict):
            component = clean_question(component, self.evidence.pairs, artefacts)
            self.count_artefacts(artefacts)
        if plan_type == "problem" and isinstance(component, dict):
            # The key's letter: the one W4 gave the slot (QLT-9, balanced and shuffled over the course); for a design
            # written before it, the question's place in the course plus the shift encoded in its practice id (Q4),
            # or, for a question without a practice, a letter seeded by the source and the unit.
            ordinal = question_ordinal(self.contract.unit_path)
            slot = self.brief.components[index] if index < len(self.brief.components) else None
            if slot is not None and slot.key_letter is not None:
                ordinal = slot.key_letter
            elif ordinal is not None and slot is not None and slot.practice is not None:
                ordinal += practice_key_shift(slot.practice.practice_id)
            elif ordinal is not None:
                ordinal = free_key_letter(self.contract.source_snapshot_hash, self.contract.unit_path)
            normalized, codes = normalize_single_choice(component, str(self.plans[index]["component_plan_id"]),
                                                        ordinal)
            # QC feedback A1/A2 (K1): the stem and the explanation in lines, after the options got their letters.
            normalized, laid_out = lay_out_quiz(normalized)
            self.runtime.adjustments.update(codes + laid_out)
            return normalized
        if plan_type != "html" or not isinstance(component, dict) or not isinstance(
                component.get("semantic_content"), dict):
            return component
        semantic, fixes = normalize_html_semantic(component["semantic_content"])
        if isinstance(semantic, dict):
            semantic = clean_semantic(semantic, self.evidence.pairs, self.runtime.locale, artefacts)
            # QC feedback A3: a callout that only repeats the slot's paragraphs, lists or tables goes (its text stays).
            if (repeats := repeated_callouts(semantic)) and (trimmed := without_blocks(semantic, repeats)) is not None:
                semantic = trimmed
                fixes[CALLOUT_REPEAT_DROPPED] += len(repeats)
        fixes.update(artefacts)
        component = {**component, "semantic_content": semantic} if fixes else component
        # QLT-5 C: an example that does not repeat the facts carries the illustrative label (explain mode, teaching
        # slots), so a drafted example never reads as a fact.
        component, labelled = self.fidelity.label_examples(index, component)
        if labelled:
            fixes["example_labelled"] += labelled
        # A5: the self-check of a worksheet whose criteria the AI suggested says so in its heading.
        if self.suggested_criteria(index) and isinstance(component.get("semantic_content"), dict) and (
                suggested := label_suggested_self_check(component["semantic_content"], self.runtime.locale,
                                                        MAX_HEADING_CHARS)) is not None:
            component = {**component, "semantic_content": suggested}
            fixes["suggested_criteria_labelled"] += 1
        if not fixes:
            return component
        self.html_fixes.update(fixes)
        self.runtime.adjustments.update({f"w5_html_{key}": count for key, count in fixes.items()})
        return component

    def count_artefacts(self, artefacts: Counter[str]) -> None:
        self.runtime.adjustments.update({f"w5_text_{key}": count for key, count in artefacts.items()})

    def bind(self, text: str) -> dict[str, Any]:
        try:
            value = _strip_learning_block_ids(bind_provider_semantic_versions(json.loads(text)))
            unit = self.deps.bind_instance_payload(value, self.expected)
        except self.deps.recoverable_errors as error:
            self.report_recoverable("idm_w5_bind", "IDM_W5_INSTANCE_INVALID", error)
            raise IdmResponseInvalidError("IDM_W5_INSTANCE_INVALID", safe_error_details(error)) from error
        components = [self.normalize_component(index, component)
                      for index, component in enumerate(unit.get("components", []))]
        return {**unit, "components": components, "component_plan": list(self.plans)}

    def report_recoverable(self, stage: str, code: str, error: BaseException) -> None:
        """Hand an error this step turns into an invalid answer to the service's observer (type, stage,
        traceback), so a programming error is not mistaken for a model answer."""

        if self.deps.on_recoverable_error is not None:
            self.deps.on_recoverable_error(stage, code, error)

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
                                                       self.framework, fidelity=self.fidelity, faq_check=self.faq,
                                                       fact_rows=self.fact_rows)
               if item.component_index not in exempt]
        return finding, idm

    def node_acceptance(self, draft: _Draft, *, whole_fallback: bool = False) -> list[AcceptanceFinding]:
        """Node acceptance findings of ``draft`` (budget only on provider content, as Node measures it)."""

        findings = node_acceptance_findings(draft.unit, self.acceptance,
                                            provider_validated=not whole_fallback and not draft.fallback_slots)
        self.acceptance_seen.extend(item for item in acceptance_log(findings) if item not in self.acceptance_seen)
        return findings

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

    # -- time budget (QC run ab8d67e1, R3) ------------------------------------------------------
    def log_budget(self, step: str, thinking: ThinkingLevel | None) -> None:
        """One line per budget decision: the step, the seconds left and the level chosen (numbers only)."""

        log_stage("idm_unit_budget", {
            "correlation_id": self.runtime.correlation_id, "unit_path": self.contract.unit_path, "step": step,
            "remaining_ms": int(self.runtime.remaining_seconds() * _MS), "thinking": thinking or "skipped",
        })

    def writer_level(self, preferred: ThinkingLevel, step: str) -> ThinkingLevel:
        level = writer_thinking(self.runtime.remaining_seconds(), repair_room=has_practice_slot(self.brief),
                                preferred=preferred)
        if level != preferred:
            self.runtime.adjustments["w5_writer_thinking_lowered_budget"] += 1
        self.log_budget(step, level)
        return level

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

    def suggested_criteria(self, index: int | None = None) -> bool:
        """A5: the worksheet slot ``index`` (any practice slot of the unit without one) practises criteria the AI
        suggested from what the lesson teaches (its feedback criterion carries the label)."""

        slots = (self.brief.components if index is None else
                 [self.brief.components[index]] if 0 <= index < len(self.brief.components) and self.is_worksheet(index)
                 else [])
        return any(slot.practice is not None and is_suggested_criteria(slot.practice.feedback_focus.criterion)
                   for slot in slots)

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

    def repeated_faq_fallback(self, index: int) -> list[str]:
        """``[FAQ_FALLBACK_REPEATED_CODE]`` when the source-locked FAQ of slot ``index`` says one sentence twice (its
        facts hold a single item, and ``idm_source_faq`` keeps the slot buildable): a reason for author review, logged
        once per slot (the slot index only, never content)."""

        component = self.fallback_component(index)
        if (component is None or component.get("type") != FAQ_COMPONENT_TYPE
                or not faq_answers_repeat(component.get("items"))):
            return []
        if index not in self.faq_fallback_repeated:
            self.faq_fallback_repeated.add(index)
            self.codes.append(FAQ_FALLBACK_REPEATED_CODE)
            log_stage("idm_unit_faq_fallback_repeated", {
                "correlation_id": self.runtime.correlation_id, "unit_path": self.contract.unit_path, "slot": index,
            })
        return [FAQ_FALLBACK_REPEATED_CODE]

    def fallback_slot(self, draft: _Draft, index: int, reasons: Sequence[str] = ()) -> _Draft | None:
        component = self.fallback_component(index)
        if component is None:
            return None
        reasons = [*reasons, *self.repeated_faq_fallback(index)]
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
        """Drop the FAQ items that repeat the html above them (N11), copy a source sentence or state a number or a
        name their facts do not (QLT-5 D), and the recall questions past the one allowed when a clarifying question
        stays, while the slot keeps its minimum item count."""

        components = list(draft.unit.get("components", []))
        component = components[index] if 0 <= index < len(components) else None
        items = component.get("items") if isinstance(component, dict) else None
        if (not isinstance(component, dict) or component.get("type") != FAQ_COMPONENT_TYPE
                or not isinstance(items, list)):
            return None
        restated = set(restated_faq_items(component, html_before(draft.unit, index)))
        copied = set(self.faq.restated(component)) - restated
        # QLT-7: an item that answers an open SME question, or invents a reason for a threshold, goes too.
        unresolved = set(self.faq.open_items(component)) - restated - copied
        new_facts = (set(self.faq.with_new_facts(index, component)) | set(self.faq.invented_reasons(index, component))
                     ) - restated - copied - unresolved
        drop = restated | copied | new_facts | unresolved
        types = question_types(component)
        kept = [position for position in range(len(items)) if position not in drop]
        kept_types = [types[position] for position in kept]
        recall = ({kept[position] for position in extra_recall_items(kept_types)}
                  if any(kind in FAQ_DEEP_TYPES for kind in kept_types) else set())
        drop |= recall
        keep = [item for position, item in enumerate(items) if position not in drop]
        if not drop or len(keep) < IDM_FAQ_MIN_ITEMS:
            return None
        components[index] = {**component, "items": keep}
        self.faq_items_dropped += len(new_facts)
        self.faq_items_restated += len(restated)
        self.faq_items_copied += len(copied)
        self.faq_items_recall += len(recall)
        self.faq_items_open += len(unresolved)
        self.codes.append(FAQ_ITEMS_DROPPED_CODE)
        for key, count in (("w5_faq_items_dropped", len(new_facts)), ("w5_faq_items_restated_dropped", len(restated)),
                           ("w5_faq_items_copied_dropped", len(copied)),
                           ("w5_faq_items_recall_dropped", len(recall)),
                           ("w5_faq_items_open_question_dropped", len(unresolved))):
            if count:
                self.runtime.adjustments[key] += count
        return _Draft({**draft.unit, "components": components}, list(draft.fallback_slots), draft.repair_applied,
                      list(draft.review_slots), dict(draft.reasons))

    def compact_worksheet(self, draft: _Draft, index: int) -> _Draft | None:
        """An over-budget worksheet made shorter, its worked example kept as long as possible (QLT-9: shorter
        guidance cells, then two example entries, then without other sections, only then without the example), when
        that fits every html rule and Node's budget and the worksheet stays complete (QC run 8de1c76b, Q2)."""

        components = list(draft.unit.get("components", []))
        component = components[index] if 0 <= index < len(components) else None
        if not isinstance(component, dict) or not self.is_worksheet(index):
            return None

        def with_slot(value: dict[str, Any]) -> _Draft:
            return _Draft({**draft.unit, "components": [*components[:index], value, *components[index + 1:]]},
                          list(draft.fallback_slots), draft.repair_applied, list(draft.review_slots),
                          dict(draft.reasons))

        def fits(value: dict[str, Any]) -> bool:
            if self.html_violations(value):
                return False
            finding, idm = self.problems(with_slot(value))
            return _slot_of(finding) != index and not any(slot == index for _code, slot in idm)

        compacted = compact_worksheet(component, fits)
        if compacted is None:
            return None
        value, keeps_example = compacted
        self.worksheet_compacted = True
        self.worksheet_example_kept = keeps_example
        self.codes.append(WORKSHEET_COMPACTED_CODE)
        self.runtime.adjustments["w5_worksheet_compacted_example_kept" if keeps_example
                                 else "w5_worksheet_compacted"] += 1
        return with_slot(value)

    def insert_framework_list(self, draft: _Draft, index: int) -> _Draft | None:
        """The teaching html with one more section listing every promised item by name (QC run 8de1c76b, Q3), when
        the names are known (unit facts or the W4 support item) and the slot stays within every html rule."""

        if self.framework is None or len(self.framework.labels) != self.framework.count:
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

    def keep_whole_questions(self, before: _Draft, after: _Draft) -> _Draft:
        """A question a repair cut short gets its pre-repair version back (QLT-8, run 0a11aa9e c1.l2.u4: the leak
        repair ended the stem at "… với các tiêu chí của 5 trục chuyển dịch" and the cut stem reached the course); the
        finding that started the repair is then settled as before (review note or fallback)."""

        old = before.unit.get("components", [])
        new = list(after.unit.get("components", []))
        restored = 0
        for index, component in enumerate(new):
            previous = old[index] if index < len(old) else None
            if (isinstance(component, dict) and component.get("type") == "problem" and isinstance(previous, dict)
                    and question_complete(previous.get("question"))
                    and not question_complete(component.get("question"))):
                new[index] = previous
                restored += 1
        if not restored:
            return after
        self.runtime.adjustments["w5_cut_question_reverted"] += restored
        return _Draft({**after.unit, "components": new}, list(after.fallback_slots), after.repair_applied,
                      list(after.review_slots), dict(after.reasons))

    def explain_acronyms(self, draft: _Draft) -> tuple[_Draft, list[str]]:
        """QLT-8: the full form in brackets at the first use of each abbreviation in the unit's provider html, when the
        facts or the server's short list give it and the slot keeps every rule it met; and the abbreviations nobody
        explains, for the author note (``app.idm.acronyms``)."""

        components = list(draft.unit.get("components", []))
        texts = [text for component in components if isinstance(component, dict)
                 for text in learner_texts(component)]
        unexplained: list[str] = []
        for token in abbreviations(texts):
            full = self.acronyms.expansion(token)
            if token in KNOWN_ABBREVIATIONS or explained(token, full, texts):
                continue
            if full is None:
                if self.acronyms.worth_noting(token):
                    unexplained.append(token)
                continue
            for index, component in enumerate(components):
                if (index in draft.fallback_slots or not isinstance(component, dict)
                        or component.get("type") != "html"):
                    continue
                expanded = with_expansion(component, token, full)
                if expanded is None:
                    continue
                candidate = _Draft({**draft.unit, "components": [*components[:index], expanded,
                                                                 *components[index + 1:]]},
                                   list(draft.fallback_slots), draft.repair_applied, list(draft.review_slots),
                                   dict(draft.reasons))
                if not self.html_violations(expanded) and self.keeps_findings(draft, candidate, index):
                    components[index] = expanded
                    draft = candidate
                    self.acronyms_explained += 1
                    self.runtime.adjustments["w5_acronym_explained"] += 1
                break
        return draft, unexplained

    def keeps_findings(self, before: _Draft, after: _Draft, index: int) -> bool:
        """``after`` fails no shared, Node or IDM rule of slot ``index`` that ``before`` passed."""

        old_finding, old_idm = self.problems(before)
        new_finding, new_idm = self.problems(after)
        if new_finding is not None and (old_finding is None or (new_finding.code, new_finding.path)
                                        != (old_finding.code, old_finding.path)):
            return False
        return {code for code, slot in new_idm if slot == index} <= {code for code, slot in old_idm if slot == index}
