"""W6 judge (spec §7.7.5): the criteria, the provider call on the learner view of the unit, and the stored
quality summary Node reads (nine criteria; the later ones are recorded under them).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal

from app.idm.contracts import (
    IdmFindingCountsV1,
    IdmJudgeFindingV1,
    IdmJudgeResponseV1,
    IdmUnitBriefV1,
    IdmUnitQualityV1,
    JudgeAnswerCriterion,
    JudgeCriterion,
    JudgeSeverity,
)
from app.idm.policy import (
    GRADED_PRACTICE_TYPES,
    IDM_JUDGE_MAX_OUTPUT_TOKENS,
    IDM_JUDGE_MIN_REMAINING_SECONDS,
    THINKING_W6,
)
from app.idm.prompts import judge_prompt
from app.idm.qa_checks import _visible
from app.idm.runtime import IdmBudgetError, IdmProviderError, IdmResponseInvalidError, IdmRuntime, idm_call
from app.idm.text import sanitize_author_text

JudgeMode = Literal["off", "observe", "repair"]
JudgeStatus = Literal["not_run", "pass", "review_required", "reject", "skipped_budget", "failed"]
CRITERIA: Final[tuple[JudgeCriterion, ...]] = (
    "Q1_support_sufficient", "Q2_not_copied", "Q3_practice_complete", "Q4_feedback_teaches",
    "Q5_grounded_criteria", "Q6_alignment", "Q7_cognitive_load", "Q8_language", "Q9_traceability",
)
# QC course 364564 (N7): the unit title "Tổng quan 5 chuyển dịch" taught something else and no criterion
# caught it. The judge answers a tenth criterion; the stored summary (Node reads exactly the nine keys
# above) records it under Q9_traceability, so the cross-language contract is unchanged.
TITLE_CRITERION: Final = "Q10_title_matches"
# QLT-5: the unit serves the course purpose (an onboarding unit that trains selling fails it). Recorded under
# Q9_traceability like Q10, in the judge's mode (observe by default), so the stored contract stays unchanged.
PURPOSE_CRITERION: Final = "Q11_purpose_alignment"
# QLT-5 C: an explained unit (content mode explain) explains its teaching for the learner. Stored under Q2_not_copied
# (the same question: taught, not pasted), so the stored contract stays unchanged; observe mode like the others.
EXPLAINED_CRITERION: Final = "Q12_explained"
# QLT-5 D: the FAQ explains (why, what if, a common mistake) instead of restating the source or the html. Stored under
# Q2_not_copied (restated, not taught), observe mode like Q10-Q12; "not_applicable" for a unit without an la_faq slot.
FAQ_VALUE_CRITERION: Final = "Q13_faq_value"
JUDGE_CRITERIA: Final[tuple[JudgeAnswerCriterion, ...]] = (*CRITERIA, TITLE_CRITERION, PURPOSE_CRITERION,
                                                          EXPLAINED_CRITERION, FAQ_VALUE_CRITERION)
_STORED_CRITERION: Final[dict[str, JudgeCriterion]] = {
    **{criterion: criterion for criterion in CRITERIA}, TITLE_CRITERION: "Q9_traceability",
    PURPOSE_CRITERION: "Q9_traceability", EXPLAINED_CRITERION: "Q2_not_copied", FAQ_VALUE_CRITERION: "Q2_not_copied"}
# QC course 364564 (N7): two teach-only units (html + FAQ) were rejected for Q3/Q4/Q6 because the judge saw
# the lesson's practice. A unit without a practice slot cannot fail these: they are "not_applicable",
# which never counts as a finding and never reaches the stored summary.
PRACTICE_CRITERIA: Final = frozenset({"Q3_practice_complete", "Q4_feedback_teaches", "Q6_alignment"})
NOT_APPLICABLE: Final = "not_applicable"
_STORED_SEVERITY: Final[dict[str, JudgeSeverity]] = {
    "pass": "pass", "minor": "minor", "major": "major", "critical": "critical"}
_SEVERITY_RANK: Final = {"pass": 0, "minor": 1, "major": 2, "critical": 3}


def learner_view(unit: dict[str, Any]) -> list[dict[str, Any]]:
    """Learner-visible text per component for the judge; no identifiers."""

    view = []
    for index, component in enumerate(item for item in unit.get("components", []) if isinstance(item, dict)):
        entry: dict[str, Any] = {"index": index, "type": component.get("type"), "title": component.get("title")}
        if component.get("type") == "problem":
            entry["question"] = component.get("question")
            entry["choices"] = [{"text": choice.get("text"), "correct": choice.get("correct")}
                                for choice in component.get("choices") or [] if isinstance(choice, dict)]
            entry["explanation"] = component.get("explanation")
        else:
            entry["text"] = _visible(component)[:6000]
        view.append(entry)
    return view


def worst_counts(findings: Sequence[IdmJudgeFindingV1]) -> IdmFindingCountsV1:
    counts = Counter(finding.severity for finding in findings)
    return IdmFindingCountsV1(minor=counts["minor"], major=counts["major"], critical=counts["critical"])


def criteria_summary(findings: Sequence[IdmJudgeFindingV1]) -> dict[JudgeCriterion, JudgeSeverity]:
    """Worst severity per stored criterion; "not_applicable" is left out and Q10 counts as Q9 (N7)."""

    summary: dict[JudgeCriterion, JudgeSeverity] = {}
    for finding in findings:
        severity = _STORED_SEVERITY.get(finding.severity)
        if severity is None:
            continue
        criterion = _STORED_CRITERION[finding.criterion]
        current = summary.get(criterion, "pass")
        if _SEVERITY_RANK[severity] >= _SEVERITY_RANK[current]:
            summary[criterion] = severity
    return summary


def has_practice_slot(brief: IdmUnitBriefV1) -> bool:
    """The unit holds a practice (role practice, a worksheet) or a question the learner answers."""

    return any(slot.role == "practice" or slot.type in GRADED_PRACTICE_TYPES for slot in brief.components)


def settle_applicability(findings: Sequence[IdmJudgeFindingV1], *, practice_slot: bool,
                         faq_slot: bool = True) -> list[IdmJudgeFindingV1]:
    """Q3/Q4/Q6 of a unit without a practice slot (and Q13 of a unit without an la_faq slot) become "not_applicable"
    whatever the judge answered; "not_applicable" on any other criterion is no verdict and is dropped (QC course
    364564, N7)."""

    settled: list[IdmJudgeFindingV1] = []
    for finding in findings:
        if ((finding.criterion in PRACTICE_CRITERIA and not practice_slot)
                or (finding.criterion == FAQ_VALUE_CRITERION and not faq_slot)):
            settled.append(finding.model_copy(update={"severity": NOT_APPLICABLE}))
        elif finding.severity != NOT_APPLICABLE:
            settled.append(finding)
    return settled


@dataclass
class JudgeOutcome:
    status: JudgeStatus
    findings: list[IdmJudgeFindingV1] = field(default_factory=list)


async def run_judge(
    runtime: IdmRuntime,
    *,
    mode: JudgeMode,
    plan_summary: dict[str, Any],
    facts: Sequence[tuple[str, str]],
    unit: dict[str, Any],
    practice_slot: bool = True,
    faq_slot: bool = True,
) -> JudgeOutcome:
    """One bounded judge call; never raises for provider/budget/response problems.

    ``practice_slot`` is False for a unit without a practice or question slot: its practice criteria
    are not applicable (QC course 364564, N7); ``faq_slot`` is False for a unit without an la_faq slot (Q13).
    """

    if mode == "off":
        return JudgeOutcome("not_run")
    if runtime.remaining_seconds() < IDM_JUDGE_MIN_REMAINING_SECONDS:
        return JudgeOutcome("skipped_budget")
    prompt = judge_prompt(runtime.locale, plan_summary=plan_summary,
                          facts=[(key, text, None) for key, text in facts], unit_content=learner_view(unit))
    try:
        response = await idm_call(runtime, stage="idm_w6_judge", prompt=prompt, response_model=IdmJudgeResponseV1,
                                  max_output_tokens=IDM_JUDGE_MAX_OUTPUT_TOKENS, thinking_level=THINKING_W6,
                                  invocation_kind="evaluator")
    except IdmBudgetError:
        return JudgeOutcome("skipped_budget")
    except IdmProviderError:
        # The judge observes; a provider problem never discards an accepted unit.
        return JudgeOutcome("failed")
    except IdmResponseInvalidError:
        return JudgeOutcome("failed")
    component_count = len(unit.get("components", []))
    findings = [finding.model_copy(update={"witness": sanitize_author_text(finding.witness, 300)})
                for finding in settle_applicability(response.findings, practice_slot=practice_slot,
                                                    faq_slot=faq_slot)
                if finding.component_index is None or finding.component_index < component_count]
    counts = worst_counts(findings)
    status: JudgeStatus = "reject" if counts.critical else "review_required" if counts.major else "pass"
    return JudgeOutcome(status, findings)


def build_unit_quality(
    *,
    mode: JudgeMode,
    judge: JudgeOutcome,
    repair_applied: bool,
    deterministic_codes: Sequence[str],
    author_note: str,
) -> IdmUnitQualityV1:
    return IdmUnitQualityV1(
        judge_mode=mode, judge_status=judge.status, finding_counts=worst_counts(judge.findings),
        criteria=criteria_summary(judge.findings), repair_applied=repair_applied,
        deterministic_codes=sorted(set(deterministic_codes))[:32], author_note=author_note,
    )


def repair_targets(findings: Sequence[IdmJudgeFindingV1]) -> list[int]:
    return sorted({finding.component_index for finding in findings
                   if finding.severity in {"major", "critical"} and finding.component_index is not None})


def blocking_count(findings: Sequence[IdmJudgeFindingV1]) -> int:
    return sum(finding.severity in {"major", "critical"} for finding in findings)
