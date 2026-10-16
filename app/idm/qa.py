"""W5 deterministic checks and the W6 judge (spec §7.7.2(b), §7.7.5)."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal

from app.idm.answer_checks import (
    contested_option,
    quoted_key,
    restates_example,
    restates_joined_example,
    restates_list,
    unit_teaching,
)
from app.idm.artifacts import cut_fragments
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
from app.idm.faq_checks import (
    FAQ_INVENTED_REASON_CODE,
    FAQ_NEW_FACT_CODE,
    FAQ_OPEN_QUESTION_CODE,
    FAQ_RESTATES_SOURCE_CODE,
    FAQ_TYPE_MIX_CODE,
    FaqChecker,
)
from app.idm.fidelity_checks import (
    EXAMPLE_ROLE_CODE,
    EXPLANATION_MISSING_CODE,
    NEW_FACT_CODE,
    SENTENCE_COPY_CODE,
    SENTENCE_LENGTH_CODE,
    FidelityChecker,
)
from app.idm.framework import FrameworkPromise, framework_listed
from app.idm.mcq import (
    ANSWER_LENGTH_CUE_CODE,
    OPTION_LETTERS,
    QUESTION_INCOMPLETE_CODE,
    answer_length_cue,
    labelled_letters,
    question_complete,
)
from app.idm.policy import (
    FAQ_TITLE_GENERIC_WORDS,
    GRADED_PRACTICE_TYPES,
    IDM_ANSWER_LEAK_MIN_NGRAMS,
    IDM_ANSWER_LEAK_MIN_SHARE,
    IDM_ANSWER_LEAK_NGRAM,
    IDM_ANSWER_LEAK_PAIR_MIN_HITS,
    IDM_ANSWER_LEAK_PAIR_MIN_MARGIN,
    IDM_ANSWER_LEAK_PAIR_MIN_SHARE,
    IDM_FAQ_RESTATE_MIN_NGRAMS,
    IDM_FAQ_RESTATE_MIN_SHARE,
    IDM_FAQ_RESTATE_NGRAM,
    IDM_FAQ_TITLE_MIN_KEY_WORDS,
    IDM_FAQ_TITLE_MIN_SHARED_SHARE,
    IDM_JUDGE_MAX_OUTPUT_TOKENS,
    IDM_JUDGE_MIN_REMAINING_SECONDS,
    IDM_UNIT_AUTHOR_NOTE_MAX_CHARS,
    IDM_WORKSHEET_MIN_GROUNDED_SHARE,
    IDM_WORKSHEET_MIN_LIST_ITEMS,
    THINKING_W6,
    WORKSHEET_COMPONENT_TYPE,
)
from app.idm.prompts import judge_prompt
from app.idm.runtime import IdmBudgetError, IdmProviderError, IdmResponseInvalidError, IdmRuntime, idm_call
from app.idm.text import (
    EvidenceIndex,
    content_pairs,
    content_words,
    grounding_verdict,
    idm_fold,
    ngram_overlap,
    sanitize_author_text,
    word_ngrams,
)
from app.learner_content_purity import component_learner_text

JudgeMode = Literal["off", "observe", "repair"]
JudgeStatus = Literal["not_run", "pass", "review_required", "reject", "skipped_budget", "failed"]

MIN_FEEDBACK_CHARS: Final = 60
# An option is named by its text when the explanation contains the start of it (accent-folded).
_OPTION_TEXT_PREFIX: Final = 40
VERBATIM_MIN_CHARS: Final = 300
VERBATIM_MAX_OVERLAP: Final = 0.7
ANSWER_LEAK_MIN_CHARS: Final = 12
# A key and at least one distractor to compare it with.
_MIN_LEAK_OPTIONS: Final = 2
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
    TITLE_CRITERION: "tiêu đề không khớp nội dung được dạy",
    PURPOSE_CRITERION: "nội dung chưa phục vụ đúng mục đích khoá học",
    EXPLAINED_CRITERION: "phần lý thuyết chưa được giảng giải cho người học",
    FAQ_VALUE_CRITERION: "phần hỏi đáp chỉ nhắc lại nội dung, chưa giải thích vì sao hay lỗi hay gặp",
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
    TITLE_CRITERION: "a title does not match what the unit teaches",
    PURPOSE_CRITERION: "the content does not serve the course purpose",
    EXPLAINED_CRITERION: "the theory is not explained for the learner",
    FAQ_VALUE_CRITERION: "the FAQ only restates the content instead of explaining why or common mistakes",
}
_SEVERITY_RANK: Final = {"pass": 0, "minor": 1, "major": 2, "critical": 3}
FAQ_ITEMS_DROPPED_CODE: Final = "IDM_W5_FAQ_ITEMS_DROPPED"
WORKSHEET_INCOMPLETE_CODE: Final = "IDM_W5_WORKSHEET_INCOMPLETE"
ANSWER_LEAK_CODE: Final = "IDM_W5_ANSWER_LEAK"
# QC run ab8d67e1 (R2): the question's case meets another option's taught criteria as well as the key's.
ANSWER_CONTESTED_CODE: Final = "IDM_W5_ANSWER_CONTESTED"
# QC run ab8d67e1 (R5): a list item, row label or cell of the html ends cut as the PDF line did ("Best-in", "(KPI /").
SOURCE_ARTIFACT_CODE: Final = "IDM_W5_SOURCE_ARTIFACT"
# QC course 364564: N9 (a callout that reads as a quotation states what the facts do not), N11 (FAQ items
# that repeat the html above; an FAQ title that does not match its questions).
CALLOUT_UNGROUNDED_CODE: Final = "IDM_W5_CALLOUT_UNGROUNDED"
CALLOUT_TO_PROSE_CODE: Final = "IDM_W5_CALLOUT_TO_PROSE"
FAQ_RESTATES_HTML_CODE: Final = "IDM_W5_FAQ_RESTATES_HTML"
FAQ_TITLE_MISMATCH_CODE: Final = "IDM_W5_FAQ_TITLE_MISMATCH"
# N5: the unit or html slot title names "N <items>" ("Tổng quan 5 chuyển dịch") but the html does not list them.
FRAMEWORK_INCOMPLETE_CODE: Final = "IDM_W5_FRAMEWORK_INCOMPLETE"
# QC run 8de1c76b (Q3): the list of the framework's items inserted by the server when the repair did not add it.
FRAMEWORK_LIST_INSERTED_CODE: Final = "IDM_W5_FRAMEWORK_LIST_INSERTED"
# A warning block renders as <blockquote> (app.idm.node_acceptance); "callout"/"note" are its aliases.
CALLOUT_KIND: Final = "warning"


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


def unexplained_options(component: dict[str, Any]) -> list[int]:
    """Options the explanation never names, by letter or by their own text (QC course 364564, N10).

    The prompt asks for "A - ...; B - ..." with every option; an explanation that skips one cannot
    tell the learner why that option is right or wrong. Letters are read as ``app.idm.mcq`` reads them
    when it relabels the explanation after the seeded shuffle, so they are the served letters.
    """

    explanation = str(component.get("explanation") or "")
    choices = [choice for choice in component.get("choices") or [] if isinstance(choice, dict)]
    letters = labelled_letters(explanation)
    folded = idm_fold(explanation)
    missing = []
    for index, choice in enumerate(choices):
        text = idm_fold(str(choice.get("text") or ""))
        by_letter = index < len(OPTION_LETTERS) and OPTION_LETTERS[index] in letters
        if not by_letter and not (len(text) >= ANSWER_LEAK_MIN_CHARS and text[:_OPTION_TEXT_PREFIX] in folded):
            missing.append(index)
    return missing


def html_before(unit: dict[str, Any], index: int) -> str:
    """Visible text of the html slots before slot ``index``: what the learner has just read."""

    components = [item for item in unit.get("components", []) if isinstance(item, dict)]
    return " ".join(_visible(component) for component in components[:index] if component.get("type") == "html")


def restated_faq_items(component: dict[str, Any], preceding_html: str) -> list[int]:
    """FAQ items whose answer repeats the html shown before them in the unit (QC course 364564, N11).

    Same measure as the answer leak (accent-folded word 4-grams), with its own calibration
    (``IDM_FAQ_RESTATE_*``): an FAQ answers a misconception, an edge case or a "what if"; one that only
    restates a row or paragraph the learner has just read adds nothing.
    """

    items = component.get("items")
    if not isinstance(items, list) or not preceding_html.strip():
        return []
    answers = [str(item.get("answer") or "") if isinstance(item, dict) else "" for item in items]
    return [index for index, answer in enumerate(answers)
            if len(word_ngrams(answer, IDM_FAQ_RESTATE_NGRAM)) >= IDM_FAQ_RESTATE_MIN_NGRAMS
            and ngram_overlap(answer, preceding_html, IDM_FAQ_RESTATE_NGRAM) >= IDM_FAQ_RESTATE_MIN_SHARE]


def faq_title_mismatch(component: dict[str, Any]) -> bool:
    """The FAQ title's key words (content words that are not FAQ boilerplate) are mostly absent from its
    questions and answers ("… trong họp giao ban" over items about something else; QC 364564, N11)."""

    items = component.get("items")
    key = set(content_words(str(component.get("title") or ""))) - FAQ_TITLE_GENERIC_WORDS
    if not isinstance(items, list) or len(key) < IDM_FAQ_TITLE_MIN_KEY_WORDS:
        return False
    said = {word for item in items if isinstance(item, dict)
            for word in content_words(f"{item.get('question') or ''} {item.get('answer') or ''}")}
    return len(key & said) < IDM_FAQ_TITLE_MIN_SHARED_SHARE * len(key)


def _sections(component: dict[str, Any]) -> list[dict[str, Any]]:
    semantic = component.get("semantic_content")
    sections = semantic.get("sections") if isinstance(semantic, dict) else None
    return [section for section in sections if isinstance(section, dict)] if isinstance(sections, list) else []


def ungrounded_callouts(component: dict[str, Any], evidence: EvidenceIndex) -> list[tuple[int, int]]:
    """(section, block) of every callout whose text does not restate the facts (QC course 364564, N9).

    A warning block renders as a blockquote that learners read as a quotation or an official rule; 3 of
    7 were sentences the writer made up. The test is the FAQ grounding verdict.
    """

    return [(section_index, block_index)
            for section_index, section in enumerate(_sections(component))
            for block_index, block in enumerate(section.get("blocks") or [])
            if isinstance(block, dict) and block.get("kind") == CALLOUT_KIND
            and grounding_verdict(str(block.get("text") or ""), evidence) != "grounded"]


def callouts_as_paragraphs(component: dict[str, Any], positions: Collection[tuple[int, int]]) -> dict[str, Any]:
    """``component`` with the callouts at ``positions`` turned into plain paragraphs (same text)."""

    semantic = component.get("semantic_content")
    if not isinstance(semantic, dict) or not positions:
        return component
    sections = []
    for section_index, section in enumerate(_sections(component)):
        blocks = [{**block, "kind": "paragraph"}
                  if (section_index, block_index) in positions and isinstance(block, dict) else block
                  for block_index, block in enumerate(section.get("blocks") or [])]
        sections.append({**section, "blocks": blocks})
    return {**component, "semantic_content": {**semantic, "sections": sections}}


def _worksheet_blocks(component: dict[str, Any]) -> list[dict[str, Any]]:
    return [block for section in _sections(component)
            for block in section.get("blocks") or [] if isinstance(block, dict)]


def worksheet_complete(component: dict[str, Any], evidence: EvidenceIndex | None) -> bool:
    """A worksheet (html slot with role practice) states the task, gives the template the learner fills
    in (table rows or steps) and ends with a self-check list whose items come from the criteria facts."""

    blocks = _worksheet_blocks(component)

    def entries(block: dict[str, Any], field_name: str) -> list[Any]:
        value = block.get(field_name)
        return value if isinstance(value, list) else []

    task = any(block.get("kind") == "task" and str(block.get("text") or "").strip() for block in blocks)
    template = any((block.get("kind") == "table" and len(entries(block, "rows")) >= IDM_WORKSHEET_MIN_LIST_ITEMS)
                   or (block.get("kind") == "steps" and len(entries(block, "items")) >= IDM_WORKSHEET_MIN_LIST_ITEMS)
                   for block in blocks)
    checklists = [entries(block, "items") for block in blocks
                  if block.get("kind") == "bullets" and len(entries(block, "items")) >= IDM_WORKSHEET_MIN_LIST_ITEMS]
    if not (task and template and checklists):
        return False
    if evidence is None:
        return True
    items = [str(item) for item in checklists[-1]]
    grounded = sum(grounding_verdict(item, evidence) == "grounded" for item in items)
    return grounded >= IDM_WORKSHEET_MIN_GROUNDED_SHARE * len(items)


def deterministic_slot_findings(
    unit: dict[str, Any], brief: IdmUnitBriefV1, owned_text_by_slot: Sequence[str],
    evidence: EvidenceIndex | None = None, framework: FrameworkPromise | None = None, *,
    fidelity: FidelityChecker | None = None, faq_check: FaqChecker | None = None,
    fact_rows: Sequence[tuple[str, str]] = (),
) -> list[SlotFinding]:
    """IDM checks on top of the shared staged validator (spec §7.7.2(b)).

    With ``evidence`` (the facts the writer was given), callouts must restate them and a worksheet's self-check
    list must come from them. With ``framework`` (the "N <items>" a title of the unit promises), the unit's
    teaching html must list every item (reported on its first html slot). With ``fidelity`` (QLT-5 C), the
    teaching html slots follow the unit's content mode (``fidelity_checks``); the slot-level verbatim check then
    only runs in the structured mode, as before. With ``faq_check`` (QLT-5 D), an la_faq slot mixes its question types,
    does not copy a source fact and states no number or name its basis facts do not (``faq_checks``). ``fact_rows``
    (QLT-7): the table rows of the facts the writer was given, read with the html rows for a contested key.
    """

    slot_copy_check = fidelity is None or fidelity.mode == "structured"

    components = [item for item in unit.get("components", []) if isinstance(item, dict)]
    findings: list[SlotFinding] = []
    preceding_html = ""
    # The FAQ answers of the unit, wherever the FAQ sits: a key must agree with them (QC run ab8d67e1, R2).
    faq = [item for component in components if component.get("type") == "la_faq"
           for item in component.get("items") or [] if isinstance(item, dict)]
    for index, component in enumerate(components):
        kind = component.get("type")
        slot = brief.components[index] if index < len(brief.components) else None
        if kind == "html":
            text = _visible(component)
            preceding_html += " " + text
            treatments = {item.treatment for item in slot.treatments} if slot else set()
            if (slot_copy_check and len(text) >= VERBATIM_MIN_CHARS and treatments and treatments != {"keep"}
                    and index < len(owned_text_by_slot)
                    and ngram_overlap(text, owned_text_by_slot[index]) > VERBATIM_MAX_OVERLAP):
                findings.append(SlotFinding("IDM_W5_VERBATIM_COPY", index))
            if (slot is not None and slot.role == "practice" and slot.type == WORKSHEET_COMPONENT_TYPE
                    and not worksheet_complete(component, evidence)):
                findings.append(SlotFinding(WORKSHEET_INCOMPLETE_CODE, index))
            if evidence is not None and ungrounded_callouts(component, evidence):
                findings.append(SlotFinding(CALLOUT_UNGROUNDED_CODE, index))
            if cut_fragments(component.get("semantic_content")):
                findings.append(SlotFinding(SOURCE_ARTIFACT_CODE, index))
        elif kind == "la_faq":
            if faq_check is not None:
                findings.extend(SlotFinding(code, index) for code in faq_check.findings(index, component))
            if restated_faq_items(component, preceding_html):
                findings.append(SlotFinding(FAQ_RESTATES_HTML_CODE, index))
            if faq_title_mismatch(component):
                findings.append(SlotFinding(FAQ_TITLE_MISMATCH_CODE, index))
        elif kind == "problem":
            explanation = " ".join(str(component.get("explanation") or "").split())
            correct = _correct_choices(component)
            if len(correct) != 1 or unexplained_options(component):
                findings.append(SlotFinding("IDM_W5_PRACTICE_INCOMPLETE", index))
            # QLT-8: a case without its question (run 0a11aa9e c1.l2.u4, cut by a repair) cannot be answered.
            if not question_complete(component.get("question")):
                findings.append(SlotFinding(QUESTION_INCOMPLETE_CODE, index))
            if len(explanation) < MIN_FEEDBACK_CHARS or idm_fold(explanation).strip(" .!") in {
                    "dung", "sai", "correct", "incorrect", "dung roi", "chua dung"}:
                findings.append(SlotFinding("IDM_W5_FEEDBACK_NOT_TEACHING", index))
            if correct:
                answer = idm_fold(str(correct[0].get("text") or ""))
                html = idm_fold(preceding_html)
                teaching = unit_teaching([item for item in components[:index] if item.get("type") == "html"], faq,
                                         fact_rows)
                if ((len(answer) >= ANSWER_LEAK_MIN_CHARS and answer in html and ("dap an" in html or "answer" in html))
                        or copied_options(component, preceding_html) or reworded_key(component, preceding_html)
                        or quoted_key(component, teaching) or restates_example(component, teaching)
                        or restates_list(component, teaching) or restates_joined_example(component, teaching)):
                    findings.append(SlotFinding(ANSWER_LEAK_CODE, index))
                if contested_option(component, teaching) is not None:
                    findings.append(SlotFinding(ANSWER_CONTESTED_CODE, index))
    # The overview is taught, not practised: a worksheet html slot neither promises nor lists it.
    html_slots = [index for index, component in enumerate(components) if component.get("type") == "html"
                  and not (index < len(brief.components) and brief.components[index].role == "practice")]
    if framework is not None and html_slots and not framework_listed(
            framework, [components[index] for index in html_slots], preceding_html):
        findings.append(SlotFinding(FRAMEWORK_INCOMPLETE_CODE, html_slots[0]))
    if fidelity is not None:
        findings.extend(SlotFinding(code, index) for code, index in fidelity.findings(components))
    return findings


def copied_options(component: dict[str, Any], preceding_text: str) -> list[int]:
    """Options that copy the html shown before the question, when that tells the answer (QC 364564, N3).

    An option copies the text when most of its word 4-grams (accent-folded) appear in it
    (``IDM_ANSWER_LEAK_*``): the correct option copying the worked example ("ví dụ đạt chuẩn"), a
    distractor copying the bad example. When every option copies the text (a recognition question
    over a list that was taught), matching tells nothing and nothing is returned.
    """

    choices = [choice for choice in component.get("choices") or [] if isinstance(choice, dict)]
    if not preceding_text.strip() or not choices:
        return []
    copied = [index for index, choice in enumerate(choices)
              if len(word_ngrams(str(choice.get("text") or ""), IDM_ANSWER_LEAK_NGRAM)) >= IDM_ANSWER_LEAK_MIN_NGRAMS
              and ngram_overlap(str(choice.get("text") or ""), preceding_text,
                                IDM_ANSWER_LEAK_NGRAM) >= IDM_ANSWER_LEAK_MIN_SHARE]
    return copied if len(copied) < len(choices) else []


def _distinctive_share(pairs: set[tuple[str, str]], others: Sequence[set[tuple[str, str]]],
                       question: set[tuple[str, str]], preceding: set[tuple[str, str]]) -> tuple[float, int]:
    distinctive = pairs - question - set().union(*others)
    hits = len(distinctive & preceding)
    return (hits / len(distinctive) if distinctive else 0.0), hits


def reworded_key(component: dict[str, Any], preceding_text: str) -> bool:
    """The correct option rewords the html shown before the question (QC run 8de1c76b, Q4).

    The 4-gram check misses a key that restates the example or quotation in other words ("Nếu không có hành
    động thì thay đổi tư duy chỉ là cảm hứng nhất thời …" after the blockquote it paraphrases). Here the key's
    distinctive content-word pairs (not in any distractor nor in the question) are looked up in the html before
    it: a leak when enough of them are there and clearly more than for any distractor
    (``IDM_ANSWER_LEAK_PAIR_*``, calibrated on the run's 9 questions).
    """

    choices = [choice for choice in component.get("choices") or [] if isinstance(choice, dict)]
    correct = [index for index, choice in enumerate(choices) if choice.get("correct") is True]
    if len(correct) != 1 or len(choices) < _MIN_LEAK_OPTIONS or not preceding_text.strip():
        return False
    preceding = content_pairs(preceding_text)
    question = content_pairs(str(component.get("question") or ""))
    pairs = [content_pairs(str(choice.get("text") or "")) for choice in choices]

    def share(index: int) -> tuple[float, int]:
        return _distinctive_share(pairs[index], [item for position, item in enumerate(pairs) if position != index],
                                  question, preceding)

    key_share, hits = share(correct[0])
    others = [share(index)[0] for index in range(len(choices)) if index != correct[0]]
    return (hits >= IDM_ANSWER_LEAK_PAIR_MIN_HITS and key_share >= IDM_ANSWER_LEAK_PAIR_MIN_SHARE
            and key_share - max(others) >= IDM_ANSWER_LEAK_PAIR_MIN_MARGIN)


def advisory_slot_findings(unit: dict[str, Any], exempt: Collection[int] = ()) -> list[SlotFinding]:
    """Review notes that never trigger a repair or a fallback: the length cue of a question (N2)."""

    return [SlotFinding(ANSWER_LENGTH_CUE_CODE, index)
            for index, component in enumerate(item for item in unit.get("components", []) if isinstance(item, dict))
            if index not in exempt and component.get("type") == "problem" and answer_length_cue(component)]


def final_unit_findings(unit: dict[str, Any], brief: IdmUnitBriefV1, owned_text_by_slot: Sequence[str],
                        evidence: EvidenceIndex | None, fallback_slots: Collection[int], *,
                        framework: FrameworkPromise | None = None, fidelity: FidelityChecker | None = None,
                        faq: FaqChecker | None = None, fact_rows: Sequence[tuple[str, str]] = ()) -> list[SlotFinding]:
    """What the returned unit still has (QC course 364564, N8): the IDM checks of its provider slots plus the
    review-only notes. A code seen before a repair is not in it unless the final unit still fails it."""

    exempt = set(fallback_slots)
    found = [item for item in deterministic_slot_findings(unit, brief, owned_text_by_slot, evidence, framework,
                                                          fidelity=fidelity, faq_check=faq, fact_rows=fact_rows)
             if item.component_index not in exempt]
    return [*found, *advisory_slot_findings(unit, exempt)]


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


_SLOT_LABEL_VI: Final = {"html": "Lý thuyết", "problem": "Quiz", "la_faq": "Hỏi đáp", "la_sortable": "Sắp xếp",
                         "la_crossword": "Ô chữ", "la_diagram": "Sơ đồ"}
_SLOT_LABEL_EN: Final = {"html": "Theory", "problem": "Quiz", "la_faq": "FAQ", "la_sortable": "Sortable",
                         "la_crossword": "Crossword", "la_diagram": "Diagram"}
_SAFE_NOTE_CODE_RE: Final = re.compile(r"^[A-Z][A-Z0-9_]{2,95}$")
_MAX_NOTE_CODES: Final = 6


def _codes(values: Sequence[str]) -> str:
    return ", ".join(list(dict.fromkeys(code for code in values if _SAFE_NOTE_CODE_RE.fullmatch(code)))[
        :_MAX_NOTE_CODES])


def _slot_reasons(slots: Sequence[int], reasons: Mapping[int, Sequence[str]], slot_types: Sequence[str],
                  vi: bool) -> str:
    labels = _SLOT_LABEL_VI if vi else _SLOT_LABEL_EN
    parts = []
    for index in sorted(slots):
        kind = slot_types[index] if 0 <= index < len(slot_types) else ""
        label = f" ({labels[kind]})" if kind in labels else ""
        codes = _codes(reasons.get(index, ()))
        parts.append(f"{'khối' if vi else 'block'} {index + 1}{label}" + (f" — {codes}" if codes else ""))
    return "; ".join(parts)


def build_unit_author_note(
    *,
    locale: str,
    judge: JudgeOutcome,
    deterministic_codes: Sequence[str],
    fallback_slots: Sequence[int],
    ai_drafted: bool,
    slot_reasons: Mapping[int, Sequence[str]] | None = None,
    review_slots: Sequence[int] = (),
    slot_types: Sequence[str] = (),
    failure_codes: Sequence[str] = (),
    whole_fallback: bool = False,
    faq_items_dropped: int = 0,
    faq_items_invalid: int = 0,
    remaining: Sequence[SlotFinding] | None = None,
    fixed_codes: Sequence[str] = (),
    faq_items_restated: int = 0,
    callouts_to_prose: int = 0,
    framework_list_inserted: int = 0,
    worksheet_compacted: bool = False,
    drafted_examples: Mapping[int, int] | None = None,
    faq_items_copied: int = 0,
    faq_items_recall: int = 0,
    faq_inferred: Mapping[int, int] | None = None,
    faq_items_open: int = 0,
    review_reasons: Sequence[str] = (),
    unexplained_acronyms: Sequence[str] = (),
) -> str:
    """Template note for the unit ``implementation_notes``; no IDs, no ``<``/``>``.

    Every fallback says why (QC course 234653, D16): the codes behind each replaced or kept slot, the
    provider/time/budget code behind a whole-unit fallback and a failed automatic repair.

    ``remaining`` are the findings of the unit as returned and ``fixed_codes`` the codes found and
    settled during generation (QC course 364564, N8: codes seen before a repair were printed as
    still open). Without ``remaining`` every deterministic code counts as still open.

    ``drafted_examples`` (QLT-5 C): the examples labelled "Ví dụ minh hoạ" per slot; the author checks each one.
    ``faq_items_copied``/``faq_items_recall`` (QLT-5 D): FAQ items dropped because they copied a source sentence or only
    asked for a fact again; ``faq_inferred``: per FAQ slot, the answers that explain beyond the words of their facts
    (the SME checks that they add no rule); ``faq_items_open`` (QLT-7): FAQ items dropped because they answered an
    open SME question.

    ``review_reasons`` (QLT-8): why the unit is marked for review (``REVIEW_REASON_*``), stated first in plain words;
    ``unexplained_acronyms``: abbreviations the learner text uses whose full form neither the facts nor the server's
    short list give, for the SME to spell out.
    """

    vi = locale == "vi"
    labels = _CRITERION_LABEL_VI if vi else _CRITERION_LABEL_EN
    reasons = slot_reasons or {}
    problems = [finding for finding in judge.findings if finding.severity in {"major", "critical"}]
    parts: list[str] = []
    if line := _review_reasons_line(review_reasons, vi):
        parts.append(line)
    if judge.status in {"pass", "review_required", "reject"}:
        if problems:
            described = "; ".join(f"{labels[finding.criterion]} ({finding.criterion.split('_')[0]})"
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
    failures = _codes([*failure_codes, *deterministic_codes])
    if whole_fallback:
        parts.append(("Cả bài dùng bản dự phòng dựng từ tài liệu — cần biên tập." if vi
                      else "The whole lesson uses the source-based fallback — edit it.")
                     + ((f" Lý do: {failures}." if vi else f" Reason: {failures}.") if failures else ""))
    elif fallback_slots:
        parts.append(f"{len(fallback_slots)} khối dùng bản dự phòng dựng từ tài liệu — cần biên tập." if vi
                     else f"{len(fallback_slots)} block(s) use the source-based fallback — edit them.")
        explained = [index for index in fallback_slots if _codes(reasons.get(index, ()))]
        if explained:
            parts.append(("Lý do: " if vi else "Reason: ") + _slot_reasons(explained, reasons, slot_types, vi) + ".")
    if review_slots and not whole_fallback:
        parts.append(("Giữ bản AI để tác giả rà soát: " if vi else "AI draft kept for author review: ")
                     + _slot_reasons(review_slots, reasons, slot_types, vi) + ".")
    repair_failures = _codes(failure_codes)
    if repair_failures and not whole_fallback and (fallback_slots or review_slots):
        parts.append(f"Sửa tự động không thành công: {repair_failures}." if vi
                     else f"Automatic repair did not succeed: {repair_failures}.")
    if faq_items_dropped:
        parts.append(f"Đã bỏ {faq_items_dropped} câu hỏi đáp có nội dung ngoài tài liệu." if vi
                     else f"{faq_items_dropped} FAQ item(s) with claims outside the source were removed.")
    if faq_items_restated:
        parts.append(f"Đã bỏ {faq_items_restated} câu hỏi đáp chỉ nhắc lại nội dung vừa học." if vi
                     else f"{faq_items_restated} FAQ item(s) that only repeated the content just taught were removed.")
    if faq_items_copied:
        parts.append(f"Đã bỏ {faq_items_copied} câu hỏi đáp gần như chép nguyên câu tài liệu." if vi
                     else f"{faq_items_copied} FAQ item(s) that nearly copied a source sentence were removed.")
    if faq_items_recall:
        parts.append(f"Đã bỏ {faq_items_recall} câu hỏi đáp chỉ hỏi lại kiến thức đã học." if vi
                     else f"{faq_items_recall} FAQ item(s) that only asked for a fact again were removed.")
    if faq_items_open:
        parts.append(f"Đã bỏ {faq_items_open} câu hỏi đáp trả lời câu hỏi còn chờ SME xác nhận." if vi
                     else f"{faq_items_open} FAQ item(s) that answered a question still open for the SME were removed.")
    if faq_items_invalid:
        parts.append(f"Đã bỏ {faq_items_invalid} câu hỏi đáp hệ thống không lưu được (ký tự góc nhọn, câu hỏi trùng)."
                     if vi else f"{faq_items_invalid} FAQ item(s) the course editor cannot store (angle brackets, "
                                "repeated question) were removed.")
    if callouts_to_prose:
        parts.append(f"Đã chuyển {callouts_to_prose} khung trích dẫn/lưu ý không có trong tài liệu thành đoạn văn "
                     "thường — cần SME xác nhận nội dung." if vi
                     else f"{callouts_to_prose} callout(s) not found in the source were turned into plain paragraphs "
                          "— the SME should confirm their content.")
    if framework_list_inserted:
        parts.append(f"Đã chèn danh sách {framework_list_inserted} thành phần của khung (tên lấy từ tài liệu) vào phần "
                     "lý thuyết — tác giả nên biên tập cho liền mạch." if vi
                     else f"A list of the framework's {framework_list_inserted} items (names from the source) was "
                          "inserted into the theory block — edit it into the text.")
    if worksheet_compacted:
        parts.append("Phiếu thực hành dài hơn giới hạn nên đã bỏ phần ví dụ mẫu (hoặc rút gọn gợi ý trong bảng) — "
                     "tác giả có thể bổ sung lại một ví dụ ngắn." if vi
                     else "The worksheet was over its length budget, so its worked example (or the extra guidance in "
                          "its table) was left out — the author may add a short example back.")
    if not whole_fallback:
        parts.extend(_check_lines(vi, deterministic_codes, remaining, fixed_codes, slot_types))
    if drafted_examples and not whole_fallback:
        parts.append(_drafted_examples_line(drafted_examples, slot_types, vi))
    if faq_inferred and not whole_fallback:
        parts.append(_faq_inferred_line(faq_inferred, slot_types, vi))
    # A source-locked unit carries no drafted scenario; the review reason already says it when it is one.
    if ai_drafted and not whole_fallback and REVIEW_REASON_SCENARIO not in review_reasons:
        parts.append("Tình huống minh hoạ do AI soạn, cần SME xác nhận tính thực tế." if vi
                     else "The illustrative scenario was drafted by AI; the SME should confirm it is realistic.")
    if unexplained_acronyms and not whole_fallback:
        parts.append(_acronyms_line(unexplained_acronyms, vi))
    return sanitize_author_text(" ".join(parts), IDM_UNIT_AUTHOR_NOTE_MAX_CHARS)


def _drafted_examples_line(examples: Mapping[int, int], slot_types: Sequence[str], vi: bool) -> str:
    """One plain line per unit: how many AI-drafted examples, and in which slots."""

    labels = _SLOT_LABEL_VI if vi else _SLOT_LABEL_EN
    total = sum(examples.values())
    where = ", ".join(f"{'khối' if vi else 'block'} {index + 1}"
                      + (f" ({labels[slot_types[index]]})" if index < len(slot_types) and slot_types[index] in labels
                         else "") for index in sorted(examples))
    return (f"Ví dụ minh hoạ do AI soạn — cần kiểm tra: {total} ví dụ ({where})." if vi
            else f"Illustrative examples drafted by AI — please check: {total} example(s) ({where}).")


def _faq_inferred_line(inferred: Mapping[int, int], slot_types: Sequence[str], vi: bool) -> str:
    """One plain line: FAQ answers that explain beyond the words of their facts, for the SME to check."""

    labels = _SLOT_LABEL_VI if vi else _SLOT_LABEL_EN
    total = sum(inferred.values())
    where = ", ".join(f"{'khối' if vi else 'block'} {index + 1}"
                      + (f" ({labels[slot_types[index]]})" if index < len(slot_types) and slot_types[index] in labels
                         else "") for index in sorted(inferred))
    return (f"Câu trả lời hỏi đáp có phần AI suy luận từ tài liệu — SME cần kiểm tra không có quy định mới: "
            f"{total} câu ({where})." if vi
            else f"FAQ answers with AI reasoning drawn from the source — the SME should check they add no new rule: "
                 f"{total} answer(s) ({where}).")


# Content-mode findings (QLT-5 C) in plain words, without codes.
_PLAIN_HINT_VI: Final = {
    SENTENCE_COPY_CODE: "nhiều câu gần như chép nguyên văn tài liệu — nên viết lại bằng lời dễ hiểu và giải thích thêm",
    EXPLANATION_MISSING_CODE: "phần lý thuyết mới nêu lại nội dung tài liệu — nên thêm giải thích vì sao quan trọng "
                              "và một ví dụ",
    NEW_FACT_CODE: "có số liệu hoặc tên riêng không có trong tài liệu — cần kiểm tra và bỏ nếu không đúng",
    FAQ_TYPE_MIX_CODE: "phần hỏi đáp chủ yếu hỏi lại kiến thức — nên có câu hỏi vì sao, nếu… thì sao hoặc lỗi hay gặp",
    FAQ_RESTATES_SOURCE_CODE: "câu trả lời hỏi đáp gần như chép nguyên câu tài liệu — nên giải thích bằng lời của mình",
    FAQ_NEW_FACT_CODE: "câu hỏi đáp có số liệu hoặc tên riêng không có trong tài liệu — cần kiểm tra và bỏ nếu không "
                       "đúng",
    # QLT-7 (QC run fae66524).
    SENTENCE_LENGTH_CODE: "nhiều câu quá dài — nên tách thành các câu ngắn, mỗi câu một ý",
    EXAMPLE_ROLE_CODE: "ví dụ minh hoạ kể về một người không thuộc đối tượng học (ví dụ trưởng nhóm trong khoá học cho "
                       "lãnh đạo) — nên đổi thành người đúng vai trò của học viên",
    FAQ_OPEN_QUESTION_CODE: "câu hỏi đáp trả lời một câu hỏi còn chờ SME xác nhận — chỉ giữ khi SME đã trả lời",
    FAQ_INVENTED_REASON_CODE: "câu hỏi đáp tự đưa ra lý do cho một con số hoặc ngưỡng mà tài liệu không giải thích — "
                              "cần SME xác nhận hoặc bỏ",
}
_PLAIN_HINT_EN: Final = {
    SENTENCE_COPY_CODE: "many sentences copy the source almost word for word — rewrite them in plain words and explain "
                        "them",
    EXPLANATION_MISSING_CODE: "the theory only restates the source — add why it matters and an example",
    NEW_FACT_CODE: "a number or a name that is not in the source — check it and remove it if it is wrong",
    FAQ_TYPE_MIX_CODE: "the FAQ mostly asks for facts again — add a why, what-if or common-mistake question",
    FAQ_RESTATES_SOURCE_CODE: "an FAQ answer nearly copies a source sentence — explain it in your own words",
    FAQ_NEW_FACT_CODE: "an FAQ item states a number or a name that is not in the source — check it and remove it if it "
                       "is wrong",
    SENTENCE_LENGTH_CODE: "many sentences are too long — split them into short sentences, one idea each",
    EXAMPLE_ROLE_CODE: "an illustrative example is about someone outside the audience (a team lead in a course for "
                       "executives) — give it a person in the learners' own role",
    FAQ_OPEN_QUESTION_CODE: "an FAQ item answers a question the SME has not settled yet — keep it only once the SME "
                            "answers it",
    FAQ_INVENTED_REASON_CODE: "an FAQ item gives its own reason for a number or threshold the source does not "
                              "explain — the SME should confirm it or remove it",
}


_REVIEW_HINT_VI: Final = {
    ANSWER_LEAK_CODE: "một phương án gần như chép lại ví dụ hoặc nội dung ngay trước câu hỏi — người học có thể "
                      "chọn theo trí nhớ thay vì áp dụng tiêu chí",
    ANSWER_LENGTH_CUE_CODE: "đáp án đúng dài hơn hẳn các phương án khác — người học có thể đoán theo độ dài",
    ANSWER_CONTESTED_CODE: "tình huống của câu hỏi cũng khớp tiêu chí mà bài học (bảng, danh sách hoặc phần hỏi đáp) "
                           "nêu cho một phương án khác — cần sửa tình huống để chỉ một phương án đúng",
    CALLOUT_UNGROUNDED_CODE: "khung trích dẫn/lưu ý nêu điều không tìm thấy trong tài liệu — chỉ giữ khi SME xác "
                             "nhận, hoặc đổi thành đoạn văn thường",
    FAQ_RESTATES_HTML_CODE: "câu hỏi đáp chỉ nhắc lại nội dung vừa học — nên thay bằng ngộ nhận, trường hợp đặc biệt "
                            "hoặc tình huống nếu… thì",
    FAQ_TITLE_MISMATCH_CODE: "tiêu đề phần hỏi đáp không khớp các câu hỏi bên trong",
    SOURCE_ARTIFACT_CODE: "có dòng hoặc ô bảng bị cắt dở như trong tài liệu PDF (ví dụ \"Best-in\", \"(KPI /\") — "
                          "cần viết lại cho trọn ý",
    FRAMEWORK_INCOMPLETE_CODE: "tiêu đề nêu một khung gồm nhiều thành phần nhưng nội dung chưa liệt kê đủ các thành "
                               "phần đó — bổ sung bản đồ tổng quan hoặc đổi tiêu đề theo nội dung",
    QUESTION_INCOMPLETE_CODE: "đề câu hỏi bị cụt, chưa có câu hỏi ở cuối — cần viết lại cho trọn câu hỏi",
}
_REVIEW_HINT_EN: Final = {
    ANSWER_LEAK_CODE: "an option nearly copies the example or text shown right before the question — learners can "
                      "match it instead of applying the criterion",
    ANSWER_LENGTH_CUE_CODE: "the correct option is much longer than the others — learners can guess it by length",
    ANSWER_CONTESTED_CODE: "the question's case also meets the criteria the unit (its table, list or FAQ) gives for "
                           "another option — change the case so only one option is correct",
    CALLOUT_UNGROUNDED_CODE: "a quotation or callout states something not found in the source — keep it only if the "
                             "SME confirms it, or turn it into a plain paragraph",
    FAQ_RESTATES_HTML_CODE: "FAQ items only repeat what was just taught — replace them with a misconception, an edge "
                            "case or a what-if",
    FAQ_TITLE_MISMATCH_CODE: "the FAQ title does not match its questions",
    SOURCE_ARTIFACT_CODE: "a list item or table cell ends cut as the PDF line did (\"Best-in\", \"(KPI /\") — "
                          "complete it",
    FRAMEWORK_INCOMPLETE_CODE: "the title names a framework of several items but the content does not list all of "
                               "them — add the overview or retitle it to what it teaches",
    QUESTION_INCOMPLETE_CODE: "the question stops before it asks anything — write the question out in full",
}

# QLT-8 (QC run 0a11aa9e: 3 units marked "Cần xem" without a reason): why a unit needs the author's review, in plain
# words, one per reason the server marks it for. Reasons another note line explains in detail say "chi tiết bên dưới".
REVIEW_REASON_FALLBACK: Final = "fallback"
REVIEW_REASON_KEPT_SLOTS: Final = "kept_slots"
REVIEW_REASON_STEP_INCOMPLETE: Final = "step_incomplete"
REVIEW_REASON_SERVER_EDIT: Final = "server_edit"
REVIEW_REASON_EVIDENCE: Final = "evidence"
REVIEW_REASON_JUDGE: Final = "judge"
REVIEW_REASON_SCENARIO: Final = "ai_scenario"
_REVIEW_REASON_VI: Final = {
    REVIEW_REASON_FALLBACK: "có khối dùng bản dự phòng dựng từ tài liệu",
    REVIEW_REASON_KEPT_SLOTS: "có khối AI viết chưa đạt hết kiểm tra tự động (chi tiết bên dưới)",
    REVIEW_REASON_STEP_INCOMPLETE: "một bước sửa hoặc kiểm tra tự động không hoàn tất",
    REVIEW_REASON_SERVER_EDIT: "hệ thống đã tự chèn hoặc đổi một phần nội dung (chi tiết bên dưới)",
    REVIEW_REASON_EVIDENCE: "tài liệu nguồn của phần này có hình ảnh hoặc bảng hệ thống chưa đọc chắc chắn — cần đối "
                            "chiếu với tài liệu gốc",
    REVIEW_REASON_JUDGE: "kiểm tra tự động phát hiện vấn đề lớn (chi tiết bên dưới)",
    REVIEW_REASON_SCENARIO: "câu hỏi dùng tình huống do AI soạn, SME cần xác nhận tình huống và đáp án",
}
_REVIEW_REASON_EN: Final = {
    REVIEW_REASON_FALLBACK: "some blocks use the source-based fallback",
    REVIEW_REASON_KEPT_SLOTS: "some AI blocks did not pass every automatic check (details below)",
    REVIEW_REASON_STEP_INCOMPLETE: "an automatic repair or check step did not finish",
    REVIEW_REASON_SERVER_EDIT: "the system inserted or changed part of the content (details below)",
    REVIEW_REASON_EVIDENCE: "the source of this section has images or tables the system could not read with "
                            "certainty — check them against the original document",
    REVIEW_REASON_JUDGE: "the automatic review found a major issue (details below)",
    REVIEW_REASON_SCENARIO: "the question uses an AI-drafted scenario; the SME should confirm the scenario and the "
                            "answer",
}
_MAX_NOTE_ACRONYMS: Final = 8


def _review_reasons_line(reasons: Sequence[str], vi: bool) -> str:
    texts = _REVIEW_REASON_VI if vi else _REVIEW_REASON_EN
    described = [texts[reason] for reason in dict.fromkeys(reasons) if reason in texts]
    if not described:
        return ""
    return ("Cần tác giả xem lại vì: " if vi else "Needs the author's review because: ") + "; ".join(described) + "."


def _acronyms_line(acronyms: Sequence[str], vi: bool) -> str:
    listed = ", ".join(list(dict.fromkeys(acronyms))[:_MAX_NOTE_ACRONYMS])
    return (f"Chữ viết tắt chưa có nghĩa đầy đủ trong tài liệu — SME bổ sung nghĩa ở lần xuất hiện đầu: {listed}." if vi
            else f"Abbreviations without their full form in the source — the SME should add it where each first "
                 f"appears: {listed}.")


def _check_lines(vi: bool, deterministic_codes: Sequence[str], remaining: Sequence[SlotFinding] | None,
                 fixed_codes: Sequence[str], slot_types: Sequence[str]) -> list[str]:
    """The "Đã tự sửa" line (settled during generation), one review line per hinted finding, then what still
    warns on the returned unit."""

    lines: list[str] = []
    fixed = _codes(sorted(set(fixed_codes)))
    if fixed:
        lines.append(f"Đã tự sửa: {fixed}." if vi else f"Automatically fixed: {fixed}.")
    still = list(deterministic_codes) if remaining is None else [item.code for item in remaining]
    hints = _REVIEW_HINT_VI if vi else _REVIEW_HINT_EN
    labels = _SLOT_LABEL_VI if vi else _SLOT_LABEL_EN
    plain = _PLAIN_HINT_VI if vi else _PLAIN_HINT_EN
    for code, hint in [*hints.items(), *plain.items()]:
        slots = sorted({item.component_index for item in remaining or () if item.code == code})
        if not slots:
            continue
        where = ", ".join(f"{'khối' if vi else 'block'} {index + 1}"
                          + (f" ({labels[slot_types[index]]})" if index < len(slot_types)
                             and slot_types[index] in labels else "") for index in slots)
        if code in plain:
            lines.append(f"Cần xem {where}: {hint}." if vi else f"Review {where}: {hint}.")
        else:
            lines.append(f"Cần xem ({code}, {where}): {hint}." if vi else f"Review ({code}, {where}): {hint}.")
    warn = sorted({code for code in still if remaining is None or (code not in hints and code not in plain)})[
        :_MAX_NOTE_CODES]
    if warn:
        lines.append("Kiểm tra tự động còn cảnh báo: " + ", ".join(warn) + "." if vi
                     else "Automatic checks still warn: " + ", ".join(warn) + ".")
    return lines


def settled_codes(seen: Sequence[str], remaining: Sequence[SlotFinding],
                  settled_elsewhere: Collection[str] = ()) -> list[str]:
    """Codes found during generation that the returned unit no longer has (N8 "đã tự sửa").

    ``settled_elsewhere`` are codes another note line already explains (a slot fallback's reason,
    the FAQ items dropped), so they are not listed twice.
    """

    open_codes = {item.code for item in remaining}
    return [code for code in dict.fromkeys(seen)
            if code not in open_codes and code not in settled_elsewhere and code not in _NOTE_LINE_CODES]


# Codes of a deterministic fix that has its own note line ("Đã bỏ …", "Đã chuyển …").
_NOTE_LINE_CODES: Final = frozenset({FAQ_ITEMS_DROPPED_CODE, CALLOUT_TO_PROSE_CODE, FRAMEWORK_LIST_INSERTED_CODE,
                                      "IDM_W5_WORKSHEET_COMPACTED"})


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
    "ANSWER_CONTESTED_CODE", "ANSWER_LEAK_CODE", "CALLOUT_KIND", "CALLOUT_TO_PROSE_CODE", "CALLOUT_UNGROUNDED_CODE",
    "CRITERIA", "EXPLAINED_CRITERION", "FAQ_ITEMS_DROPPED_CODE", "FAQ_RESTATES_HTML_CODE", "FAQ_TITLE_MISMATCH_CODE",
    "FAQ_VALUE_CRITERION",
    "FRAMEWORK_INCOMPLETE_CODE", "FRAMEWORK_LIST_INSERTED_CODE", "JUDGE_CRITERIA", "NOT_APPLICABLE",
    "PRACTICE_CRITERIA", "PURPOSE_CRITERION", "REVIEW_REASON_EVIDENCE", "REVIEW_REASON_FALLBACK", "REVIEW_REASON_JUDGE",
    "REVIEW_REASON_KEPT_SLOTS", "REVIEW_REASON_SCENARIO", "REVIEW_REASON_SERVER_EDIT", "REVIEW_REASON_STEP_INCOMPLETE",
    "SOURCE_ARTIFACT_CODE", "TITLE_CRITERION", "WORKSHEET_INCOMPLETE_CODE",
    "JudgeOutcome",
    "SlotFinding", "advisory_slot_findings", "blocking_count",
    "build_unit_author_note", "build_unit_quality", "callouts_as_paragraphs", "copied_options",
    "deterministic_slot_findings", "faq_title_mismatch", "final_unit_findings",
    "has_practice_slot", "html_before", "learner_view", "repair_targets", "restated_faq_items", "reworded_key",
    "run_judge",
    "settle_applicability", "settled_codes", "unexplained_options", "ungrounded_callouts", "worksheet_complete",
]
