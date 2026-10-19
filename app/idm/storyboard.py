"""IDM ``generate_unit`` task: W5 storyboard writer + W6 judge (spec §7.7).

The writer fills exactly the server-owned component slots of the unit contract.
Acceptance reuses the shared staged validators (injected through
:class:`IdmUnitDeps`, so ``app.idm`` never imports ``app.main``), then Node's own
revision-0 acceptance rules (:mod:`app.idm.node_acceptance`), and adds the IDM
checks. A failing slot gets one scoped repair, then the source-locked fallback; a
unit Node would reject is never returned (run c2e5ac41).

The unit inputs (deps, brief, budget, draft, settle code sets) are in ``app.idm.storyboard_context`` and the
writer's state and deterministic steps in ``app.idm.storyboard_steps``; the prompts, the provider steps, the
settlement and the unit flow are here.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Mapping, Sequence
from typing import Any, Final

from app.contracts.orchestration_v2.provider import UnitGenerationContractV2
from app.idm.answer_checks import contested_option, unit_teaching
from app.idm.budget import judge_repair_fits, repair_thinking_for
from app.idm.faq_checks import (
    FAQ_INVENTED_REASON_CODE,
    FAQ_OPEN_QUESTION_CODE,
    FAQ_RESTATES_SOURCE_CODE,
    FAQ_TYPE_MIX_CODE,
    FAQ_VALUE_CODES,
    RECALL,
    faq_writer_rules,
    question_types,
)
from app.idm.fidelity import slot_marks, writer_fidelity_rules
from app.idm.fidelity_checks import (
    EXAMPLE_ROLE_CODE,
    FIDELITY_CODES,
    NEW_FACT_CODE,
    SENTENCE_COPY_CODE,
    SENTENCE_LENGTH_CODE,
    length_report,
)
from app.idm.framework import framework_coverage
from app.idm.html_rules import DENSITY_CODE
from app.idm.mcq import ANSWER_LENGTH_CUE_CODE, OPTION_LETTERS, option_length_stats
from app.idm.node_acceptance import AcceptanceFinding, acceptance_log
from app.idm.policy import (
    FAQ_DEEP_TYPES,
    IDM_MCQ_LENGTH_TARGET_RATIO,
    IDM_SENTENCE_LENGTH_MAX_AVERAGE_WORDS,
    IDM_SENTENCE_LENGTH_MAX_WORDS,
    IDM_UNIT_WRITER_MAX_OUTPUT_TOKENS,
    THINKING_W5,
)
from app.idm.prompts import (
    COMPACT_UNIT,
    example_opening,
    html_slot_rules,
    html_violation_line,
    repair_suffix,
    rule_line,
    truncation_suffix,
    unit_repair_suffix,
    unit_writer_prompt,
)
from app.idm.qa import (
    ANSWER_CONTESTED_CODE,
    CALLOUT_UNGROUNDED_CODE,
    FAQ_RESTATES_HTML_CODE,
    FRAMEWORK_INCOMPLETE_CODE,
    NOTICE_REASONS,
    REVIEW_REASON_EVIDENCE,
    REVIEW_REASON_FALLBACK,
    REVIEW_REASON_JUDGE,
    REVIEW_REASON_KEPT_SLOTS,
    REVIEW_REASON_SCENARIO,
    REVIEW_REASON_SERVER_EDIT,
    REVIEW_REASON_STEP_INCOMPLETE,
    REVIEW_REASON_STYLE,
    STYLE_CODES,
    JudgeMode,
    JudgeOutcome,
    advisory_slot_findings,
    blocking_count,
    build_unit_author_note,
    build_unit_quality,
    final_unit_findings,
    has_practice_slot,
    html_before,
    repair_targets,
    restated_faq_items,
    run_judge,
    settled_codes,
    ungrounded_callouts,
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
from app.idm.storyboard_context import (
    _COVERAGE_CODES,
    _FALLBACK_INVALID,
    _MAX_LOGGED_DETAILS,
    _MS,
    _REVIEW_FIRST_CODES,
    _REVIEW_ONLY_CODES,
    _TARGETED_REPAIR_CODES,
    REPAIR_SKIPPED_CODE,
    IdmUnitDeps,
    NodeAcceptanceUnitFinding,
    _Draft,
    _safe_path,
    _slot_of,
    acceptance_context,
    brief_purpose,
    build_idm_expected,
    output_budget,
    parse_brief,
    safe_error_details,
)
from app.idm.storyboard_steps import IdmUnitWriterSteps

_FIELD_OF_PATH_RE: Final = re.compile(r"^components\[\d+\]\.?")
# FAQ findings settled by dropping the items concerned while the slot keeps two items.
_FAQ_PRUNE_CODES: Final = frozenset({FAQ_RESTATES_HTML_CODE, *FAQ_VALUE_CODES})
_MAX_HINT_FACT_KEYS: Final = 24
# Locations named per content-mode repair line (server-built JSON paths, never text).
_MAX_HINT_LOCATIONS: Final = 8


class IdmUnitWriter(IdmUnitWriterSteps):
    """The W5 writer of one unit: prompts, provider steps and the settlement of a draft (``IdmUnitWriterSteps``
    holds the state and the deterministic steps)."""

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
            **slot_marks(brief, index),
            **({"faq_items": [item.model_dump(mode="json") for item in slot.faq_items]} if slot.faq_items else {}),
        } for index, slot in enumerate(brief.components)]
        material: dict[str, Any] = ({"faq_material": brief.faq_material.model_dump(mode="json")}
                                    if brief.faq_material is not None else {})
        # QLT-7: the open SME questions the FAQ must leave unanswered, listed apart from the material.
        if self.has_faq and self.faq.open_questions:
            material["open_questions"] = list(self.faq.open_questions)
        return unit_writer_prompt(
            self.runtime.locale, course_title=brief.course_title, audience=brief.target_audience,
            lesson_title=brief.lesson_title, lesson_objective=brief.lesson_objective,
            practice_sentences=brief.lesson_practice_sentences, previous_title=brief.previous_lesson_title,
            next_title=brief.next_lesson_title,
            unit_brief={"unit_title": self.contract.unit_title, "segment": brief.unit_segment,
                        "purpose": brief.unit_purpose, "slots": slots,
                        "job_aid_signpost": brief.job_aid_signpost, **material},
            facts=[(fact.fact_key, fact.fact_text, None) for fact in self.contract.source_facts],
            context_facts=[(item.fact_key, item.fact_text, None) for item in brief.lesson_context_facts],
            html_rules=html_slot_rules(max_words=self.budget.get("max_words"),
                                       max_chars=self.budget.get("max_visible_chars"),
                                       min_chars=self.min_html_chars, explain=self.mode == "explain")
            if any(slot.type == "html" for slot in brief.components) else "",
            course_purpose=brief_purpose(brief),
            fidelity_rules=writer_fidelity_rules(self.mode, self.runtime.locale, self.fidelity.roles),
            faq_rules=faq_writer_rules(self.mode) if self.has_faq else "",
            example_opening=example_opening(self.contract.unit_path, self.runtime.locale),
        )

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
            lines.extend(html_violation_line(slot, item, only_density=only_density, explain=self.mode == "explain")
                         for item in violations)
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
                    elif code == ANSWER_LENGTH_CUE_CODE and isinstance(component, dict):
                        lines.append(self.length_cue_line(slot, component))
                    elif code == ANSWER_CONTESTED_CODE and isinstance(component, dict):
                        lines.append(self.contested_rule_line(slot, index, component, unit))
                    elif (code in {SENTENCE_COPY_CODE, NEW_FACT_CODE, SENTENCE_LENGTH_CODE, EXAMPLE_ROLE_CODE}
                          and isinstance(component, dict)):
                        lines.append(self.fidelity_rule_line(slot, index, component, code))
                    else:
                        field = _FIELD_OF_PATH_RE.sub("", locations.get((code, index), ""))
                        lines.append(rule_line(slot, code, f"{slot}.{field}" if field else slot))
                    explained.add(code)
        return lines

    def faq_rule_line(self, slot: str, index: int, component: dict[str, Any], code: str,
                      unit: dict[str, Any] | None = None) -> str:
        """Targeted repair: which items to rewrite, the server-measured type mix and the facts each item is planned
        on (keys and numbers, never text)."""

        hint = self.faq.plan_hint(index) or self.fact_hint(index)
        if code == FAQ_TYPE_MIX_CODE:
            types = question_types(component)
            deep = sum(kind in FAQ_DEEP_TYPES for kind in types)
            return (rule_line(slot, code, f"{slot}.items") + f" ({types.count(RECALL)} recall question(s), {deep} why, "
                    "what-if or common-mistake question(s))" + hint)
        found = (restated_faq_items(component, html_before(unit or {}, index)) if code == FAQ_RESTATES_HTML_CODE
                 else self.faq.restated(component) if code == FAQ_RESTATES_SOURCE_CODE
                 else self.faq.open_items(component) if code == FAQ_OPEN_QUESTION_CODE
                 else self.faq.invented_reasons(index, component) if code == FAQ_INVENTED_REASON_CODE
                 else self.faq.with_new_facts(index, component))
        items = ", ".join(f"{slot}.items[{item}]" for item in found)
        return rule_line(slot, code, items or slot) + hint

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

    def contested_rule_line(self, slot: str, index: int, component: dict[str, Any], unit: dict[str, Any]) -> str:
        """Which option's taught criteria the case also meets (its served letter, never content) (R2)."""

        components = [item for item in unit.get("components", []) if isinstance(item, dict)]
        faq = [entry for item in components if item.get("type") == "la_faq"
               for entry in item.get("items") or [] if isinstance(entry, dict)]
        rival = contested_option(component, unit_teaching(
            [item for item in components[:index] if item.get("type") == "html"], faq, self.fact_rows))
        letter = f" option {OPTION_LETTERS[rival]}" if rival is not None and rival < len(OPTION_LETTERS) else ""
        return rule_line(slot, ANSWER_CONTESTED_CODE, f"{slot}.question") + (
            f" (the case also meets the criteria the unit gives for{letter or ' another option'})")

    def fidelity_rule_line(self, slot: str, index: int, component: dict[str, Any], code: str) -> str:
        """Targeted repair of a content-mode finding: the locations and, for a copy or long sentences, the
        server-measured numbers; for a drafted example of another level, the roles of the audience (QLT-7)."""

        if code == SENTENCE_COPY_CODE:
            report = self.fidelity.report(index, component)
            limit = self.fidelity.copy_limit(index) or 0.0
            locations = list(dict.fromkeys(report.copied))[:_MAX_HINT_LOCATIONS]
            measured = (f" ({len(report.copied)} of {report.judged} judged sentences repeat the facts word for word; "
                        f"at most {round(limit * 100)}% may)")
        elif code == SENTENCE_LENGTH_CODE:
            lengths = length_report(component)
            locations = list(lengths.long_locations)[:_MAX_HINT_LOCATIONS]
            measured = (f" ({lengths.sentences} prose sentences of {lengths.average:.1f} words on average, the longest "
                        f"{lengths.longest}; at most {IDM_SENTENCE_LENGTH_MAX_AVERAGE_WORDS} on average and "
                        f"{IDM_SENTENCE_LENGTH_MAX_WORDS} for any sentence)")
        elif code == EXAMPLE_ROLE_CODE:
            locations = self.fidelity.role_locations(component)[:_MAX_HINT_LOCATIONS]
            roles = self.fidelity.roles.phrases
            measured = f" (roles of TARGET_AUDIENCE: {', '.join(roles)})" if roles else ""
        else:
            locations = self.fidelity.new_fact_locations(component)[:_MAX_HINT_LOCATIONS]
            measured = ""
        where = ", ".join(f"{slot}.{location}" for location in locations)
        return rule_line(slot, code, where or slot) + measured

    def length_cue_line(self, slot: str, component: dict[str, Any]) -> str:
        """The server-measured option lengths and the length to aim for (QLT-7): the key at most
        ``IDM_MCQ_LENGTH_TARGET_RATIO`` x the average distractor."""

        stats = option_length_stats(component)
        line = rule_line(slot, ANSWER_LENGTH_CUE_CODE, f"{slot}.choices")
        if stats is None:
            return line
        key, mean, longest = stats
        target = round(IDM_MCQ_LENGTH_TARGET_RATIO * mean)
        return line + (f" (the correct option has {key} characters, the other options {round(mean)} on average and "
                       f"{longest} at most: bring the correct option to about {target} characters or less, or "
                       "lengthen the distractors with plausible detail)")

    def fact_hint(self, index: int) -> str:
        brief_slot = self.brief.components[index] if index < len(self.brief.components) else None
        keys = list(dict.fromkeys([*brief_slot.owned_fact_keys, *brief_slot.supporting_fact_keys]))[
            :_MAX_HINT_FACT_KEYS] if brief_slot else []
        listed = ", ".join(f"[{key}]" for key in keys)
        return f" Start from this slot's facts: {listed}." if keys else ""

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
                self.report_recoverable("idm_w5_repair", "IDM_W5_REPAIR_INVALID", error)
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

    async def budgeted_repair(self, draft: _Draft, issues: Sequence[tuple[str, int]],
                              locations: Mapping[tuple[str, int], str], *, targeted: bool) -> tuple[_Draft, bool]:
        """One repair call at the thinking level the time left allows, or none (R3); ``(draft, repaired)``.

        A failed repair (budget, provider, invalid answer) leaves the draft for the per-slot settlement."""

        level = repair_thinking_for(self.runtime.remaining_seconds(), targeted=targeted, preferred=THINKING_W5)
        self.log_budget("repair_targeted" if targeted else "repair", level)
        if level is None:
            self.runtime.adjustments["w5_targeted_repair_skipped_budget" if targeted
                                     else "w5_repair_skipped_budget"] += 1
            if not targeted:
                self.failure_codes.append(REPAIR_SKIPPED_CODE)
            return draft, False
        try:
            return await self.repair_slots(draft, issues, locations, thinking=level), True
        except (IdmBudgetError, IdmResponseInvalidError) as error:
            self.failure_codes.append(error.code)
        except IdmProviderError as error:
            if error.terminal:
                raise
            self.failure_codes.append(error.code)
        return draft, False

    async def settle(self, draft: _Draft) -> _Draft | None:
        """Repair once, then replace failing slots with the source-locked fallback.

        A slot without a source-locked rebuild that only fails IDM pedagogy checks is kept for
        author review (``review_slots``); a shared-validator failure there still discards the draft.
        """

        finding, idm = self.problems(draft)
        # A correct option far longer than the others gets one repair too (QC run 8de1c76b, Q4); left by it, it is
        # a review note as before.
        cues = [(item.code, item.component_index) for item in advisory_slot_findings(
            draft.unit, {*draft.fallback_slots, *draft.review_slots})]
        if finding is None and not idm and not cues:
            return draft
        self.log_findings("writer", draft, finding, [*idm, *cues])
        # An over-budget worksheet is compacted before any repair call (QC run 8de1c76b, Q2: 550 then 452 words
        # against 400, then the raw source fallback); the repair handles what compaction cannot.
        if finding is not None and finding.code == DENSITY_CODE and (slot := _slot_of(finding)) is not None:
            compacted = self.compact_worksheet(draft, slot)
            if compacted is not None:
                self.codes.append(finding.code)
                draft = compacted
                finding, idm = self.problems(draft)
                if finding is None and not idm and not cues:
                    return draft
        self.codes.extend([finding.code] if finding is not None else [])
        self.codes.extend(code for code, _index in [*idm, *cues])
        issues = [*idm, *cues]
        original = draft
        # Only notes to settle: a repair that breaks anything else is dropped (the unit was acceptable before).
        notes_only = finding is None and not idm
        # Only content-mode or FAQ findings (QLT-5 C, D): they end as review notes, so a repair that makes a slot fail a
        # shared or Node rule (a longer explanation over the budget) is dropped instead of replacing the slot with the
        # source-locked copy.
        review_only = finding is None and bool(idm) and {code for code, _index in idm} <= _REVIEW_ONLY_CODES
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
        # deterministic step settles anyway (a missing framework list, a length cue, a leak, an FAQ item) is a
        # targeted one: low thinking, and none at all when too little time is left for it (QC run 8de1c76b, Q3). Any
        # other repair gets the thinking level the remaining time leaves room for (QC run ab8d67e1, R3: two leak
        # repairs started at medium with ~36 s left and were cut by the 120 s unit deadline).
        targeted = finding is None and {code for code, _index in issues} <= _TARGETED_REPAIR_CODES
        draft, repaired = await self.budgeted_repair(draft, issues, locations, targeted=targeted)
        if repaired:
            draft = self.keep_whole_questions(original, draft)
        if repaired and notes_only and any(self.problems(draft)):
            self.runtime.adjustments["w5_note_repair_reverted"] += 1
            draft = original
        elif repaired and review_only and self.problems(draft)[0] is not None:
            fidelity = {code for code, _index in idm} <= FIDELITY_CODES
            self.runtime.adjustments["w5_fidelity_repair_reverted" if fidelity else "w5_faq_repair_reverted"] += 1
            draft = original
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
                # FAQ answers that repeat the html above (N11), copy a source sentence or state a new number or name,
                # and recall questions past the allowance (QLT-5 D) are dropped while the slot keeps two items.
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
    # Off the event loop only outside the two deadlines below: a timeout must never leave a thread changing the writer.
    writer = await runtime.run_cpu(IdmUnitWriter, contract, brief, deps, runtime)
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
    unexplained_acronyms: list[str] = []
    if draft is not None:
        draft, unexplained_acronyms = await runtime.run_cpu(writer.explain_acronyms, draft)
    if draft is not None and (rejected := await runtime.run_cpu(writer.node_acceptance, draft)):
        # Unreachable while settle() applies the same rules; never hand Node a unit it rejects.
        _log_node_rejection(writer, "final", rejected)
        writer.failure_codes.append("IDM_NODE_ACCEPTANCE_REJECTED")
        draft = None
    whole_fallback = draft is None
    if draft is None:
        draft = _Draft(await runtime.run_cpu(_whole_fallback, writer), list(range(len(writer.plans))))
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
    # QLT-8: each reason is named in the note (QC run 0a11aa9e: 3 units marked for review without one). QLT-9 (QC run
    # ffaf7318: 14 of 18 units marked): slots kept with wording findings only and source pages with images or tables are
    # notices ("Lưu ý"), not a review mark.
    style_only = bool(draft.review_slots) and all(set(draft.reasons.get(index, ())) <= STYLE_CODES
                                                  for index in draft.review_slots)
    reasons = [reason for reason, applies in (
        (REVIEW_REASON_FALLBACK, not whole_fallback and bool(draft.fallback_slots)),
        (REVIEW_REASON_STEP_INCOMPLETE, not whole_fallback and usage_source != "provider"),
        (REVIEW_REASON_KEPT_SLOTS, bool(draft.review_slots) and not style_only),
        (REVIEW_REASON_STYLE, style_only),
        (REVIEW_REASON_SERVER_EDIT, writer.callouts_converted > 0 or writer.framework_inserted > 0),
        (REVIEW_REASON_EVIDENCE, deps.evidence_review_required),
        (REVIEW_REASON_JUDGE, blocking_count(judge.findings) > 0),
        (REVIEW_REASON_SCENARIO, ai_drafted and _q5(judge) != "pass"),
    ) if applies]
    review_reasons = [reason for reason in reasons if reason not in NOTICE_REASONS]
    notices = [reason for reason in reasons if reason in NOTICE_REASONS]
    reviewable = not provider_lane or bool(review_reasons)
    # N8: "still warns" is recomputed on the returned unit; codes settled on the way are listed apart.
    remaining = [] if whole_fallback else await runtime.run_cpu(
        final_unit_findings, draft.unit, brief, writer.owned_text, writer.evidence, draft.fallback_slots,
        framework=writer.framework, fidelity=writer.fidelity, faq=writer.faq, fact_rows=writer.fact_rows)
    components = draft.unit.get("components", [])
    # Labelled examples of provider slots only: a source-locked slot carries none.
    drafted = {} if whole_fallback else {index: count for index, count in writer.fidelity.drafted_examples(
        components).items() if index not in draft.fallback_slots}
    # FAQ answers that explain beyond the words of their facts: the SME checks they add no rule (QLT-5 D).
    inferred = {} if whole_fallback else writer.faq.inferred_counts(components, draft.fallback_slots)
    fixed = [] if whole_fallback else settled_codes(
        writer.codes, remaining, {code for index in draft.fallback_slots for code in draft.reasons.get(index, ())})
    note = build_unit_author_note(
        locale=runtime.locale, judge=judge, deterministic_codes=writer.codes, fallback_slots=draft.fallback_slots,
        ai_drafted=ai_drafted, slot_reasons=draft.reasons, review_slots=draft.review_slots,
        slot_types=[plan["type"] for plan in writer.plans], failure_codes=writer.failure_codes,
        whole_fallback=whole_fallback, faq_items_dropped=writer.faq_items_dropped,
        faq_items_invalid=writer.faq_items_invalid, remaining=remaining, fixed_codes=fixed,
        faq_items_restated=writer.faq_items_restated, callouts_to_prose=writer.callouts_converted,
        framework_list_inserted=writer.framework_inserted, worksheet_compacted=writer.worksheet_compacted,
        worksheet_example_kept=writer.worksheet_example_kept,
        drafted_examples=drafted, faq_items_copied=writer.faq_items_copied, faq_items_recall=writer.faq_items_recall,
        faq_inferred=inferred, faq_items_open=writer.faq_items_open,
        review_reasons=review_reasons if reviewable and not whole_fallback else (),
        unexplained_acronyms=unexplained_acronyms, notices=() if whole_fallback else notices)
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
        "faq_items_copied": writer.faq_items_copied, "faq_items_recall": writer.faq_items_recall,
        "faq_items_open": writer.faq_items_open,
        "faq": {} if whole_fallback else writer.faq.metrics(components),
        "callouts_to_prose": writer.callouts_converted, "framework_list_inserted": writer.framework_inserted,
        "acronyms_explained": writer.acronyms_explained, "acronyms_unexplained": len(unexplained_acronyms),
        "review_reasons": review_reasons if reviewable else [], "notices": notices,
        "slot_reasons": {str(index): codes for index, codes in sorted(draft.reasons.items())},
        "node_acceptance": writer.acceptance_seen[:_MAX_LOGGED_DETAILS],
        "fidelity": {"mode": writer.mode} if whole_fallback else writer.fidelity.metrics(components),
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
    for attempt in (1, 2):
        # The writer thinks at medium only when its p90 time still leaves room for a repair and the judge (R3).
        thinking = writer.writer_level(thinking, "writer" if attempt == 1 else "writer_retry")
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
        # QLT-5: Q11 judges the unit against the course purpose (null for a brief without one).
        "course_purpose": brief_purpose(writer.brief),
        # QLT-5 C: Q12 judges how an explained unit explains (not_applicable in the other modes).
        "fidelity_mode": writer.mode,
        # QLT-5 D: Q13 judges the value of the unit's FAQ (not_applicable without one).
        "has_faq_slot": writer.has_faq,
        "slots": [{"index": index, "type": slot.type, "role": slot.role, "title": slot.title,
                   "treatments": [item.model_dump(mode="json") for item in slot.treatments]}
                  for index, slot in enumerate(writer.brief.components)],
    }
    judge = await run_judge(writer.runtime, mode=mode, plan_summary=summary, facts=facts, unit=draft.unit,
                            practice_slot=practice_slot, faq_slot=writer.has_faq)
    targets = [index for index in repair_targets(judge.findings) if index not in draft.fallback_slots]
    if mode != "repair" or not targets:
        return draft, judge
    # The repair and the second judge must both fit; otherwise the findings stay as review notes (R3).
    level = (repair_thinking_for(writer.runtime.remaining_seconds(), targeted=False, preferred=THINKING_W5)
             if judge_repair_fits(writer.runtime.remaining_seconds()) else None)
    writer.log_budget("judge_repair", level)
    if level is None:
        writer.runtime.adjustments["w6_repair_skipped_budget"] += 1
        return draft, judge
    issues = [(f"IDM_W6_{finding.criterion.split('_')[0]}", finding.component_index)
              for finding in judge.findings
              if finding.component_index in targets and finding.severity in {"major", "critical"}]
    try:
        repaired = await writer.repair_slots(draft, [(code, index) for code, index in issues if index is not None],
                                             thinking=level)
    except (IdmBudgetError, IdmResponseInvalidError, IdmProviderError):
        return draft, judge
    finding, idm = writer.problems(repaired)
    if finding is not None or idm:
        return draft, judge
    second = await run_judge(writer.runtime, mode=mode, plan_summary=summary, facts=facts, unit=repaired.unit,
                             practice_slot=practice_slot, faq_slot=writer.has_faq)
    if second.status in {"pass", "review_required", "reject"} and (
            blocking_count(second.findings) < blocking_count(judge.findings)):
        return repaired, second
    return draft, judge


# The names other modules and the tests import from this path; the unit inputs moved to storyboard_context and
# the deterministic steps of the writer to storyboard_steps.
__all__ = [
    "IdmUnitDeps", "IdmUnitWriter", "acceptance_context", "build_idm_expected", "output_budget", "parse_brief",
    "run_idm_unit", "safe_error_details",
]
