"""W5 deterministic checks and the W6 judge (spec §7.7.2(b), §7.7.5)."""

from __future__ import annotations

import re
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
    JudgeCriterion,
    JudgeSeverity,
)
from app.idm.policy import (
    IDM_JUDGE_MAX_OUTPUT_TOKENS,
    IDM_JUDGE_MIN_REMAINING_SECONDS,
    IDM_UNIT_AUTHOR_NOTE_MAX_CHARS,
    THINKING_W6,
)
from app.idm.prompts import judge_prompt
from app.idm.runtime import IdmBudgetError, IdmProviderError, IdmResponseInvalidError, IdmRuntime, idm_call
from app.idm.text import idm_fold, ngram_overlap, sanitize_author_text
from app.learner_content_purity import component_learner_text

JudgeMode = Literal["off", "observe", "repair"]
JudgeStatus = Literal["not_run", "pass", "review_required", "reject", "skipped_budget", "failed"]

MIN_FEEDBACK_CHARS: Final = 60
MIN_EXPLAINED_OPTIONS: Final = 2
VERBATIM_MIN_CHARS: Final = 300
VERBATIM_MAX_OVERLAP: Final = 0.7
ANSWER_LEAK_MIN_CHARS: Final = 12
CRITERIA: Final[tuple[JudgeCriterion, ...]] = (
    "Q1_support_sufficient", "Q2_not_copied", "Q3_practice_complete", "Q4_feedback_teaches",
    "Q5_grounded_criteria", "Q6_alignment", "Q7_cognitive_load", "Q8_language", "Q9_traceability",
)
_CRITERION_LABEL_VI: Final = {
    "Q1_support_sufficient": "phần hỗ trợ chưa đủ cho bài luyện tập",
    "Q2_not_copied": "nội dung còn chép nguyên văn tài liệu",
    "Q3_practice_complete": "bài luyện tập chưa đầy đủ",
    "Q4_feedback_teaches": "phản hồi chưa giải thích từng lựa chọn",
    "Q5_grounded_criteria": "đáp án chưa bám tiêu chí trong tài liệu",
    "Q6_alignment": "bài luyện tập chưa khớp loại hành động của Must Do",
    "Q7_cognitive_load": "lượng lý thuyết quá dài",
    "Q8_language": "ngôn ngữ chưa phù hợp người học",
    "Q9_traceability": "có khối không phục vụ Must Do",
}
_CRITERION_LABEL_EN: Final = {
    "Q1_support_sufficient": "support is not enough for the practice",
    "Q2_not_copied": "content is copied from the source",
    "Q3_practice_complete": "the practice is incomplete",
    "Q4_feedback_teaches": "feedback does not explain each option",
    "Q5_grounded_criteria": "the answer does not follow the source criteria",
    "Q6_alignment": "the practice does not match the Must Do action",
    "Q7_cognitive_load": "too much theory in a row",
    "Q8_language": "language does not suit the learner",
    "Q9_traceability": "a block does not serve the Must Do",
}
_SEVERITY_RANK: Final = {"pass": 0, "minor": 1, "major": 2, "critical": 3}
_OPTION_LABEL_RE: Final = re.compile(r"(?:^|[\s(;,.])([A-F])\s*(?:[—\-:.)]|là\b|is\b)")


@dataclass(frozen=True)
class SlotFinding:
    code: str
    component_index: int


def _visible(component: dict[str, Any]) -> str:
    return " ".join(component_learner_text(component).split())


def _correct_choices(component: dict[str, Any]) -> list[dict[str, Any]]:
    choices = component.get("choices")
    if not isinstance(choices, list):
        return []
    return [choice for choice in choices if isinstance(choice, dict) and choice.get("correct") is True]


def _explained_option_count(component: dict[str, Any]) -> int:
    explanation = str(component.get("explanation") or "")
    choices = [choice for choice in component.get("choices") or [] if isinstance(choice, dict)]
    labels = {match.group(1) for match in _OPTION_LABEL_RE.finditer(explanation)}
    folded = idm_fold(explanation)
    texts = sum(1 for choice in choices
                if len(idm_fold(str(choice.get("text") or ""))) >= ANSWER_LEAK_MIN_CHARS
                and idm_fold(str(choice.get("text") or ""))[:40] in folded)
    return max(len(labels), texts)


def deterministic_slot_findings(
    unit: dict[str, Any], brief: IdmUnitBriefV1, owned_text_by_slot: Sequence[str],
) -> list[SlotFinding]:
    """IDM checks on top of the shared staged validator (spec §7.7.2(b))."""

    components = [item for item in unit.get("components", []) if isinstance(item, dict)]
    findings: list[SlotFinding] = []
    preceding_html = ""
    for index, component in enumerate(components):
        kind = component.get("type")
        if kind == "html":
            text = _visible(component)
            preceding_html += " " + text
            slot = brief.components[index] if index < len(brief.components) else None
            treatments = {item.treatment for item in slot.treatments} if slot else set()
            if (len(text) >= VERBATIM_MIN_CHARS and treatments and treatments != {"keep"}
                    and index < len(owned_text_by_slot)
                    and ngram_overlap(text, owned_text_by_slot[index]) > VERBATIM_MAX_OVERLAP):
                findings.append(SlotFinding("IDM_W5_VERBATIM_COPY", index))
        elif kind == "problem":
            explanation = " ".join(str(component.get("explanation") or "").split())
            correct = _correct_choices(component)
            if len(correct) != 1 or _explained_option_count(component) < MIN_EXPLAINED_OPTIONS:
                findings.append(SlotFinding("IDM_W5_PRACTICE_INCOMPLETE", index))
            if len(explanation) < MIN_FEEDBACK_CHARS or idm_fold(explanation).strip(" .!") in {
                    "dung", "sai", "correct", "incorrect", "dung roi", "chua dung"}:
                findings.append(SlotFinding("IDM_W5_FEEDBACK_NOT_TEACHING", index))
            if correct:
                answer = idm_fold(str(correct[0].get("text") or ""))
                html = idm_fold(preceding_html)
                if (len(answer) >= ANSWER_LEAK_MIN_CHARS and answer in html
                        and ("dap an" in html or "answer" in html)):
                    findings.append(SlotFinding("IDM_W5_ANSWER_LEAK", index))
    return findings


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
    summary: dict[JudgeCriterion, JudgeSeverity] = {}
    for finding in findings:
        current = summary.get(finding.criterion, "pass")
        if _SEVERITY_RANK[finding.severity] >= _SEVERITY_RANK[current]:
            summary[finding.criterion] = finding.severity
    return summary


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
) -> JudgeOutcome:
    """One bounded judge call; never raises for provider/budget/response problems."""

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
                for finding in response.findings
                if finding.component_index is None or finding.component_index < component_count]
    counts = worst_counts(findings)
    status: JudgeStatus = "reject" if counts.critical else "review_required" if counts.major else "pass"
    return JudgeOutcome(status, findings)


def build_unit_author_note(
    *,
    locale: str,
    judge: JudgeOutcome,
    deterministic_codes: Sequence[str],
    fallback_slots: Sequence[int],
    ai_drafted: bool,
) -> str:
    """Template note for the unit ``implementation_notes``; no IDs, no ``<``/``>``."""

    vi = locale == "vi"
    labels = _CRITERION_LABEL_VI if vi else _CRITERION_LABEL_EN
    problems = [finding for finding in judge.findings if finding.severity in {"major", "critical"}]
    parts: list[str] = []
    if judge.status in {"pass", "review_required", "reject"}:
        if problems:
            described = "; ".join(f"{labels[finding.criterion]} ({finding.criterion[:2]})"
                                  for finding in problems[:4])
            parts.append(f"QA tự động: {len(problems)} vấn đề cần xem — {described}." if vi
                         else f"Automatic QA: {len(problems)} issue(s) to review — {described}.")
        else:
            parts.append("QA tự động: không phát hiện vấn đề lớn." if vi
                         else "Automatic QA: no major issue found.")
    elif judge.status == "skipped_budget":
        parts.append("QA tự động: bỏ qua do hết thời gian." if vi else "Automatic QA: skipped (time budget).")
    elif judge.status == "failed":
        parts.append("QA tự động: không chạy được, cần rà soát thủ công." if vi
                     else "Automatic QA: unavailable, review manually.")
    if fallback_slots:
        parts.append(f"{len(fallback_slots)} khối dùng bản dự phòng dựng từ tài liệu — cần biên tập." if vi
                     else f"{len(fallback_slots)} block(s) use the source-based fallback — edit them.")
    if deterministic_codes:
        parts.append("Kiểm tra tự động còn cảnh báo: " + ", ".join(sorted(set(deterministic_codes))[:6]) + "."
                     if vi else "Automatic checks still warn: " + ", ".join(sorted(set(deterministic_codes))[:6]) + ".")
    if ai_drafted:
        parts.append("Tình huống minh hoạ do AI soạn, cần SME xác nhận tính thực tế." if vi
                     else "The illustrative scenario was drafted by AI; the SME should confirm it is realistic.")
    return sanitize_author_text(" ".join(parts), IDM_UNIT_AUTHOR_NOTE_MAX_CHARS)


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


__all__ = [
    "CRITERIA", "JudgeOutcome", "SlotFinding", "blocking_count", "build_unit_author_note", "build_unit_quality",
    "deterministic_slot_findings", "learner_view", "repair_targets", "run_judge",
]
