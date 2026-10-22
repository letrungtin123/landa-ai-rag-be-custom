"""W6 judge of a settled W5 unit and, in the ``repair`` judge mode, one repair of the slots it finds a major or critical
problem in (spec §7.7.5; ``app.idm.storyboard``), and the review reasons and notices of the returned unit.

Owner P4 (2026-10-20): ``repair`` is the default judge mode (``AI_RAG_IDM_JUDGE_MODE`` still overrides it). The
repair runs only when it and a second judge still fit (``budget.judge_repair_fits``, about 50 s); it follows the same
rules as the W5 repairs: one retry after a transient provider failure, the slots it makes worse back in their judged
version (``IdmUnitRepairSteps.guard_repair``), and a slot kept for review that the repair fixed loses its mark. The
repair is kept only when the second judge finds fewer major or critical problems.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.idm.budget import judge_repair_fits, repair_thinking_for
from app.idm.org_context import judge_block
from app.idm.policy import THINKING_W5
from app.idm.qa import (
    NOTICE_REASONS,
    REVIEW_REASON_EVIDENCE,
    REVIEW_REASON_EXAMPLE_NUMBERS,
    REVIEW_REASON_FALLBACK,
    REVIEW_REASON_JUDGE,
    REVIEW_REASON_KEPT_SLOTS,
    REVIEW_REASON_SCENARIO,
    REVIEW_REASON_SERVER_EDIT,
    REVIEW_REASON_SOURCE,
    REVIEW_REASON_STEP_INCOMPLETE,
    REVIEW_REASON_STYLE,
    STYLE_CODES,
    JudgeMode,
    JudgeOutcome,
    blocking_count,
    has_practice_slot,
    repair_targets,
    run_judge,
)
from app.idm.runtime import IdmBudgetError, IdmProviderError, IdmResponseInvalidError
from app.idm.storyboard_context import _Draft, brief_purpose

if TYPE_CHECKING:
    from app.idm.storyboard import IdmUnitWriter


def judge_inputs(writer: IdmUnitWriter) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    """The plan summary and the facts the judge reads (never the lesson's other practice)."""

    facts = [(key, writer.text_by_id[key]) for key in writer.contract.unit_source_fact_ids
             if key in writer.text_by_id]
    criteria = {key for slot in writer.brief.components if slot.practice
                for key in slot.practice.criteria_fact_keys}
    facts.extend((item.fact_key, item.fact_text) for item in writer.brief.lesson_context_facts
                 if item.fact_key in criteria)
    summary = {
        "unit_title": writer.contract.unit_title, "segment": writer.brief.unit_segment,
        "purpose": writer.brief.unit_purpose, "must_do": writer.brief.lesson_objective,
        # Only this unit's own practice: the lesson's practice sentences made the judge grade practice that
        # sits in another unit and reject teach-only units (QC course 364564, N7).
        "practices": [slot.practice.sentence for slot in writer.brief.components if slot.practice is not None],
        "has_practice_slot": has_practice_slot(writer.brief),
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
    return summary, facts


async def judge_and_repair(writer: IdmUnitWriter, draft: _Draft, mode: JudgeMode) -> tuple[_Draft, JudgeOutcome]:
    summary, facts = judge_inputs(writer)
    practice_slot = has_practice_slot(writer.brief)
    background = judge_block(writer.brief.organization_context)
    judge = await run_judge(writer.runtime, mode=mode, plan_summary=summary, facts=facts, unit=draft.unit,
                            practice_slot=practice_slot, faq_slot=writer.has_faq, organization_context=background)
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
        repaired = await writer.repair_with_retry(
            draft, [(code, index) for code, index in issues if index is not None], {}, level, targeted=False)
    except (IdmBudgetError, IdmResponseInvalidError, IdmProviderError):
        return draft, judge
    # A slot the repair made worse keeps its judged version; anything else still failing drops the repair (a second
    # judge that ran out of time leaves the first judgement standing).
    repaired = writer.guard_repair(draft, writer.keep_whole_questions(draft, repaired), "w6")
    finding, idm = writer.problems(repaired)
    if repaired is draft or finding is not None or idm:
        return draft, judge
    repaired = writer.refresh_review_marks(repaired)
    second = await run_judge(writer.runtime, mode=mode, plan_summary=summary, facts=facts, unit=repaired.unit,
                             practice_slot=practice_slot, faq_slot=writer.has_faq, organization_context=background)
    if second.status in {"pass", "review_required", "reject"} and (
            blocking_count(second.findings) < blocking_count(judge.findings)):
        return repaired, second
    return draft, judge


def unit_review_reasons(writer: IdmUnitWriter, draft: _Draft, judge: JudgeOutcome, *, whole_fallback: bool,
                        usage_source: str, ai_drafted: bool, example_numbers: bool) -> tuple[list[str], list[str]]:
    """``(review reasons, notices)`` of the returned unit (QLT-8: each one is named in the note)."""

    # A callout turned into a paragraph still states what the facts do not (N9): the SME confirms it.
    # QLT-9 (QC run ffaf7318: 14 of 18 units marked for review): slots kept with wording findings only and source pages
    # with images or tables are notices ("Lưu ý"), not a review mark; so are numbers inside AI-drafted examples (owner
    # decision 2026-10-20).
    style_only = bool(draft.review_slots) and all(set(draft.reasons.get(index, ())) <= STYLE_CODES
                                                  for index in draft.review_slots)
    reasons = [reason for reason, applies in (
        (REVIEW_REASON_FALLBACK, not whole_fallback and bool(draft.fallback_slots)),
        (REVIEW_REASON_STEP_INCOMPLETE, not whole_fallback and usage_source != "provider"),
        (REVIEW_REASON_KEPT_SLOTS, bool(draft.review_slots) and not style_only),
        (REVIEW_REASON_STYLE, style_only),
        (REVIEW_REASON_SERVER_EDIT, writer.callouts_converted > 0 or writer.framework_inserted > 0),
        (REVIEW_REASON_EVIDENCE, writer.deps.evidence_review_required),
        (REVIEW_REASON_JUDGE, blocking_count(judge.findings) > 0),
        (REVIEW_REASON_SCENARIO, ai_drafted and q5_verdict(judge) != "pass"),
        (REVIEW_REASON_EXAMPLE_NUMBERS, example_numbers),
        (REVIEW_REASON_SOURCE, bool(writer.source_flags)),
    ) if applies]
    return ([reason for reason in reasons if reason not in NOTICE_REASONS],
            [reason for reason in reasons if reason in NOTICE_REASONS])


def q5_verdict(judge: JudgeOutcome) -> str:
    """"pass" or "fail" of the judge's Q5 (grounded criteria) answers; "unknown" without a judgement or a Q5 answer."""

    severities = [finding.severity for finding in judge.findings if finding.criterion == "Q5_grounded_criteria"]
    if judge.status in {"not_run", "skipped_budget", "failed"}:
        return "unknown"
    if not severities:
        return "unknown"
    return "pass" if all(severity == "pass" for severity in severities) else "fail"
