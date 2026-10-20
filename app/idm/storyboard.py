"""IDM ``generate_unit`` task: W5 storyboard writer + W6 judge (spec §7.7).

The writer fills exactly the server-owned component slots of the unit contract.
Acceptance reuses the shared staged validators (injected through
:class:`IdmUnitDeps`, so ``app.idm`` never imports ``app.main``), then Node's own
revision-0 acceptance rules (:mod:`app.idm.node_acceptance`), and adds the IDM
checks. A failing slot gets one scoped repair (a leak, a new fact or a contested key a
second, targeted one when time allows), then the source-locked fallback; a repair never
leaves a slot worse than it found it, and a unit Node would reject is never returned (run c2e5ac41).

The unit inputs (deps, brief, budget, draft, settle code sets) are in ``app.idm.storyboard_context``, the
writer's state and deterministic steps in ``app.idm.storyboard_steps``, its repair decisions in
``app.idm.storyboard_repair`` and the judge and review reasons in ``app.idm.storyboard_judge``; the prompts, the
provider steps, the settlement and the unit flow are here.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Mapping, Sequence
from typing import Any, Final

from app.contracts.orchestration_v2.provider import UnitGenerationContractV2
from app.idm.answer_checks import contested_option, unit_teaching
from app.idm.budget import repair_thinking_for, second_repair_fits
from app.idm.faq_checks import (
    FAQ_INVENTED_REASON_CODE,
    FAQ_NEW_FACT_CODE,
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
from app.idm.org_context import writer_block
from app.idm.policy import (
    FAQ_DEEP_TYPES,
    IDM_MCQ_LENGTH_TARGET_RATIO,
    IDM_SENTENCE_LENGTH_MAX_AVERAGE_WORDS,
    IDM_SENTENCE_LENGTH_MAX_WORDS,
    IDM_UNIT_WRITER_MAX_OUTPUT_TOKENS,
    THINKING_TARGETED_REPAIR,
    THINKING_W5,
)
from app.idm.prompts import (
    COMPACT_UNIT,
    example_opening,
    html_slot_rules,
    html_violation_line,
    new_fact_hint,
    repair_suffix,
    rule_line,
    second_repair_line,
    truncation_suffix,
    unit_repair_suffix,
    unit_writer_prompt,
)
from app.idm.qa import (
    ANSWER_CONTESTED_CODE,
    CALLOUT_UNGROUNDED_CODE,
    FAQ_RESTATES_HTML_CODE,
    FRAMEWORK_INCOMPLETE_CODE,
    JudgeMode,
    JudgeOutcome,
    advisory_slot_findings,
    build_unit_author_note,
    build_unit_quality,
    final_unit_findings,
    html_before,
    restated_faq_items,
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
    UnitFinding,
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
from app.idm.storyboard_judge import judge_and_repair as _judge_and_repair
from app.idm.storyboard_judge import unit_review_reasons
from app.idm.storyboard_repair import IdmUnitRepairSteps

_FIELD_OF_PATH_RE: Final = re.compile(r"^components\[\d+\]\.?")
# FAQ findings settled by dropping the items concerned while the slot keeps two items.
_FAQ_PRUNE_CODES: Final = frozenset({FAQ_RESTATES_HTML_CODE, *FAQ_VALUE_CODES})
_MAX_HINT_FACT_KEYS: Final = 24
# Locations named per content-mode repair line (server-built JSON paths, never text).
_MAX_HINT_LOCATIONS: Final = 8


class IdmUnitWriter(IdmUnitRepairSteps):
    """The W5 writer of one unit: prompts, provider steps and the settlement of a draft (``IdmUnitWriterSteps``
    holds the state and the deterministic steps, ``IdmUnitRepairSteps`` the repair decisions)."""

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
            organization_context=writer_block(brief.organization_context),
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
        on (keys and numbers); for a new fact, the new numbers and names of the writer's answer (owner P2)."""

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
        tokens = self.faq.new_fact_tokens(index, component) if code == FAQ_NEW_FACT_CODE else {}
        return (rule_line(slot, code, items or slot)
                + new_fact_hint(token for item in found for token in tokens.get(item, ())) + hint)

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
        server-measured numbers; for a drafted example of another level, the roles of the audience (QLT-7); for a new
        fact, the numbers and names of the writer's answer the facts do not state (owner P2, 2026-10-20)."""

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
            hits = [hit for hit in self.fidelity.new_fact_hits(component) if hit.review_tokens][:_MAX_HINT_LOCATIONS]
            locations = [hit.location for hit in hits]
            measured = new_fact_hint(token for hit in hits for token in hit.review_tokens)
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
                           thinking: ThinkingLevel = THINKING_W5, *, second: bool = False) -> _Draft:
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

        rules = self.rule_lines(draft.unit, issues, locations or {})
        if second:
            # Owner P3: the second targeted repair says what the first one left and how to change the case.
            rules.extend(line for code, index in issues if (line := second_repair_line(f"c{index}", code)))
        prompt = self.writer_prompt() + unit_repair_suffix(
            [{"code": code, "path": f"components[{index}]"} for code, index in issues], rules)
        unit = await idm_generate(
            self.runtime, stage="idm_w5_repair", prompt=prompt, response_schema=model, parse=parse,
            max_output_tokens=IDM_UNIT_WRITER_MAX_OUTPUT_TOKENS, thinking_level=thinking,
            invocation_kind="repair",
        )
        return _Draft({**unit, "component_plan": list(self.plans)}, list(draft.fallback_slots), True,
                      list(draft.review_slots), dict(draft.reasons))

    async def repair_with_retry(self, draft: _Draft, issues: Sequence[tuple[str, int]],
                                locations: Mapping[tuple[str, int], str], thinking: ThinkingLevel, *,
                                targeted: bool, second: bool = False) -> _Draft:
        """``repair_slots``, sent once more after a transient provider failure of the call when the time left allows
        it (owner P6, ``retry_level``)."""

        failed_before = self.runtime.provider_failure_code
        try:
            return await self.repair_slots(draft, issues, locations, thinking, second=second)
        except IdmProviderError as error:
            level = self.retry_level(error, targeted=targeted, preferred=thinking, failed_before=failed_before)
            if level is None:
                raise
        return await self.repair_slots(draft, issues, locations, level, second=second)

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
            return await self.repair_with_retry(draft, issues, locations, level, targeted=targeted), True
        except (IdmBudgetError, IdmResponseInvalidError) as error:
            self.failure_codes.append(error.code)
        except IdmProviderError as error:
            if error.terminal:
                raise
            self.failure_codes.append(error.code)
        return draft, False

    async def settle(self, draft: _Draft) -> _Draft | None:
        """Repair once (a leak, a new fact or a contested key a second time when the time left allows), then replace
        failing slots with the source-locked fallback.

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
        if finding is not None and _slot_of(finding) is None:
            return None
        draft, repaired = await self.first_repair(draft, finding, idm, cues)
        if repaired:
            draft = await self.second_repair(draft)
        return self.settle_slots(draft, repaired=repaired)

    async def first_repair(self, draft: _Draft, finding: UnitFinding | None, idm: Sequence[tuple[str, int]],
                           cues: Sequence[tuple[str, int]]) -> tuple[_Draft, bool]:
        """The one repair call for the writer's findings; ``(draft, repaired)``. A slot it makes worse keeps the
        writer's version (owner P1, ``guard_repair``)."""

        original = draft
        # Owner P7: the FAQ slots with wording findings only are settled without a repair call (pruned or noted).
        issues = self.without_style_faq([*idm, *cues])
        # Only notes to settle: a repair that breaks anything else is dropped (the unit was acceptable before).
        notes_only = finding is None and not idm
        # Only content-mode or FAQ findings (QLT-5 C, D): they end as review notes, so a repair that makes a slot fail a
        # shared or Node rule (a longer explanation over the budget) is dropped instead of replacing the slot with the
        # source-locked copy.
        review_only = finding is None and bool(idm) and {code for code, _index in idm} <= _REVIEW_ONLY_CODES
        locations: dict[tuple[str, int], str] = {}
        if finding is not None and (slot := _slot_of(finding)) is not None:
            issues.append((finding.code, slot))
            locations[(finding.code, slot)] = _safe_path(finding.path) or ""
        # Every slot Node would reject is repaired in the same call, not only the first one.
        issues.extend(dict.fromkeys((item.code, item.component_index) for item in self.last_acceptance[1:]
                                    if item.component_index is not None
                                    and (item.code, item.component_index) not in issues))
        if not issues:
            return draft, False
        # A failed repair (budget, provider, invalid answer) falls through to slot fallback. A repair of findings a
        # deterministic step settles anyway (a missing framework list, a length cue, a leak, an FAQ item) is a
        # targeted one: low thinking, and none at all when too little time is left for it (QC run 8de1c76b, Q3). Any
        # other repair gets the thinking level the remaining time leaves room for (QC run ab8d67e1, R3: two leak
        # repairs started at medium with ~36 s left and were cut by the 120 s unit deadline).
        targeted = finding is None and {code for code, _index in issues} <= _TARGETED_REPAIR_CODES
        draft, repaired = await self.budgeted_repair(draft, issues, locations, targeted=targeted)
        if not repaired:
            return draft, False
        draft = self.guard_repair(original, self.keep_whole_questions(original, draft), "w5")
        if notes_only and any(self.problems(draft)):
            self.runtime.adjustments["w5_note_repair_reverted"] += 1
            draft = original
        elif review_only and self.problems(draft)[0] is not None:
            fidelity = {code for code, _index in idm} <= FIDELITY_CODES
            self.runtime.adjustments["w5_fidelity_repair_reverted" if fidelity else "w5_faq_repair_reverted"] += 1
            draft = original
        return draft, True

    async def second_repair(self, draft: _Draft) -> _Draft:
        """Owner P3: one more targeted repair of the leaks, new facts and contested keys the first repair left, when it
        and the judge still fit; the slots it makes worse keep their version (P1)."""

        issues = self.second_repair_issues(draft)
        if not issues:
            return draft
        if not second_repair_fits(self.runtime.remaining_seconds(), judge=self.deps.judge_mode != "off"):
            self.log_budget("repair_second", None)
            self.runtime.adjustments["w5_second_repair_skipped_budget"] += 1
            return draft
        self.log_budget("repair_second", THINKING_TARGETED_REPAIR)
        try:
            repaired = await self.repair_with_retry(draft, issues, {}, THINKING_TARGETED_REPAIR, targeted=True,
                                                    second=True)
        except (IdmBudgetError, IdmResponseInvalidError) as error:
            self.failure_codes.append(error.code)
            return draft
        except IdmProviderError as error:
            if error.terminal:
                raise
            self.failure_codes.append(error.code)
            return draft
        return self.guard_repair(draft, self.keep_whole_questions(draft, repaired), "w5_second")

    def settle_slots(self, draft: _Draft, *, repaired: bool) -> _Draft | None:
        """Settle each slot still failing after the repairs (Node item pruning, FAQ pruning, callouts to prose, a
        framework list, compaction, then fallback or review); ``None`` when the unit cannot be returned."""

        # Each pass settles one step of one slot (Node item pruning, FAQ pruning or callouts to prose, then
        # fallback or review), so the bound never limits.
        for attempt in range(3 * len(self.plans) + 1):
            finding, idm = self.problems(draft)
            if finding is None and not idm:
                # Owner P3: a slot kept for review that a later step settled loses its mark.
                return self.refresh_review_marks(draft)
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
    for index in range(len(writer.plans)):  # an FAQ that says one sentence twice is a reason in the note
        writer.repeated_faq_fallback(index)
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
    draft, judge = await _written(writer, fallback_only=fallback_only, mode=deps.judge_mode)
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
    components = draft.unit.get("components", [])
    # Labelled examples of provider slots only: a source-locked slot carries none. Owner decision 2026-10-20: a number
    # the facts do not state inside one is a notice, not a review mark.
    drafted, example_numbers = ({}, {}) if whole_fallback else (
        {index: count for index, count in writer.fidelity.drafted_examples(components).items()
         if index not in draft.fallback_slots},
        {index: count for index, count in writer.fidelity.example_numbers(components).items()
         if index not in draft.fallback_slots})
    review_reasons, notices = unit_review_reasons(writer, draft, judge, whole_fallback=whole_fallback,
                                                  usage_source=usage_source, ai_drafted=ai_drafted,
                                                  example_numbers=bool(example_numbers))
    reviewable = not provider_lane or bool(review_reasons)
    # N8: "still warns" is recomputed on the returned unit; codes settled on the way are listed apart.
    remaining = [] if whole_fallback else await runtime.run_cpu(
        final_unit_findings, draft.unit, brief, writer.owned_text, writer.evidence, draft.fallback_slots,
        framework=writer.framework, fidelity=writer.fidelity, faq=writer.faq, fact_rows=writer.fact_rows)
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
        worksheet_example_kept=writer.worksheet_example_kept, drafted_examples=drafted,
        example_numbers=example_numbers, faq_items_copied=writer.faq_items_copied,
        faq_items_recall=writer.faq_items_recall, faq_inferred=inferred, faq_items_open=writer.faq_items_open,
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


async def _written(writer: IdmUnitWriter, *, fallback_only: bool, mode: JudgeMode) -> tuple[_Draft | None,
                                                                                            JudgeOutcome]:
    """The settled and judged provider draft, or ``None`` when the whole unit falls back (the codes say why)."""

    runtime = writer.runtime
    if fallback_only:
        writer.failure_codes.append("IDM_W5_FALLBACK_ONLY")
        return None, JudgeOutcome("not_run")
    try:
        async with asyncio.timeout(max(0.001, runtime.remaining_seconds())):
            draft = await _provider_draft(writer)
    except TimeoutError:
        writer.failure_codes.append("IDM_DEADLINE_EXCEEDED")
        return None, JudgeOutcome("not_run")
    except IdmBudgetError as error:
        writer.failure_codes.append(error.code)
        return None, JudgeOutcome("not_run")
    except IdmProviderError as error:
        # A definitive rejection (credentials, request) must reach Node's fast-fail path.
        if error.terminal:
            raise
        writer.failure_codes.append(error.code)
        return None, JudgeOutcome("not_run")
    if draft is None:
        return None, JudgeOutcome("not_run")
    # QA runs on its own clock: running out of time never discards the accepted unit.
    try:
        async with asyncio.timeout(max(0.001, runtime.remaining_seconds())):
            return await _judge_and_repair(writer, draft, mode)
    except TimeoutError:
        return draft, JudgeOutcome("skipped_budget")


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


# The names other modules and the tests import from this path; the unit inputs moved to storyboard_context, the
# deterministic steps of the writer to storyboard_steps and storyboard_repair, and the judge to storyboard_judge.
__all__ = [
    "IdmUnitDeps", "IdmUnitWriter", "acceptance_context", "build_idm_expected", "output_budget", "parse_brief",
    "run_idm_unit", "safe_error_details",
]
