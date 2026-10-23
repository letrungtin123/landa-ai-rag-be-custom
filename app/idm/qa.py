"""W5 deterministic checks and the W6 judge (spec §7.7.2(b), §7.7.5).

The checks live in ``app.idm.qa_checks`` and the judge in ``app.idm.qa_judge``; this module builds the unit
author note and keeps the names other modules import from ``app.idm.qa``.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from typing import Any, Final, NamedTuple

from app.idm.answer_near import ANSWER_ITEM_LENGTH_CODE, ANSWER_NEAR_EXAMPLE_CODE
from app.idm.faq_checks import (
    FAQ_HYPE_CODE,
    FAQ_INVENTED_REASON_CODE,
    FAQ_NEW_FACT_CODE,
    FAQ_OPEN_QUESTION_CODE,
    FAQ_RESTATES_SOURCE_CODE,
    FAQ_TYPE_MIX_CODE,
)
from app.idm.fidelity_checks import (
    EXAMPLE_ROLE_CODE,
    EXPLANATION_MISSING_CODE,
    NEW_FACT_CODE,
    SENTENCE_COPY_CODE,
    SENTENCE_LENGTH_CODE,
)
from app.idm.mcq import ANSWER_LENGTH_CUE_CODE, QUESTION_INCOMPLETE_CODE
from app.idm.note_codes import contract_code_text
from app.idm.policy import IDM_UNIT_AUTHOR_NOTE_MAX_CHARS, IDM_UNIT_SOURCE_FLAGS_MAX_CHARS
from app.idm.qa_checks import (
    ANSWER_CONTESTED_CODE,
    ANSWER_LEAK_CODE,
    CALLOUT_KIND,
    CALLOUT_TO_PROSE_CODE,
    CALLOUT_UNGROUNDED_CODE,
    FAQ_ITEMS_DROPPED_CODE,
    FAQ_RESTATES_HTML_CODE,
    FAQ_TITLE_MISMATCH_CODE,
    FRAMEWORK_INCOMPLETE_CODE,
    FRAMEWORK_LIST_INSERTED_CODE,
    SOURCE_ARTIFACT_CODE,
    WORKSHEET_INCOMPLETE_CODE,
    SlotFinding,
    advisory_slot_findings,
    callouts_as_paragraphs,
    copied_options,
    deterministic_slot_findings,
    faq_title_mismatch,
    final_unit_findings,
    html_before,
    restated_faq_items,
    reworded_key,
    unexplained_options,
    ungrounded_callouts,
    worksheet_complete,
)
from app.idm.qa_judge import (
    CRITERIA,
    EXPLAINED_CRITERION,
    FAQ_VALUE_CRITERION,
    JUDGE_CRITERIA,
    NOT_APPLICABLE,
    PRACTICE_CRITERIA,
    PURPOSE_CRITERION,
    TITLE_CRITERION,
    JudgeMode,
    JudgeOutcome,
    blocking_count,
    build_unit_quality,
    criteria_summary,
    has_practice_slot,
    learner_view,
    repair_targets,
    run_judge,
    settle_applicability,
    worst_counts,
)
from app.idm.quiz_text import (
    EXPLANATION_FILLER_CODE,
    EXPLANATION_LAYOUT_CODE,
    EXPLANATION_LENGTH_CODE,
    EXPLANATION_NAMES_ONLY_CODE,
    EXPLANATION_RESTATES_CODE,
    EXPLANATION_VERDICT_CODE,
    QUESTION_FEW_OPTIONS_CODE,
    QUESTION_ITEM_LETTERS_CODE,
    QUESTION_ITEM_NOUN_CODE,
    QUESTION_LAYOUT_CODE,
    QUESTION_LENGTH_CODE,
    QUESTION_LIMIT_CODE,
    QUESTION_TWICE_CODE,
    QUIZ_STYLE_CODES,
)
from app.idm.review_hints import REVIEW_HINT_EN, REVIEW_HINT_VI
from app.idm.source_fallback import glued_line, placeholder_line
from app.idm.source_flags import fit_flags
from app.idm.source_locked import FAQ_FALLBACK_REPEATED_CODE
from app.idm.text import sanitize_author_text
from app.idm.unit_note_lines import (
    SCENARIO_LINE,
    SLOT_LABEL_EN,
    SLOT_LABEL_VI,
    SUGGESTED_CRITERIA_LINE,
    callout_lines,
    classroom_line,
    drafted_examples_line,
    dropped_faq_lines,
    example_numbers_line,
    fallback_input_line,
    faq_inferred_line,
    framework_line,
    supposition_line,
    worksheet_line,
)
from app.idm.worksheet_input import EXAMPLE_REUSE_CODE, WORKSHEET_INPUT_CODE

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


_SLOT_LABEL_VI: Final = SLOT_LABEL_VI
_SLOT_LABEL_EN: Final = SLOT_LABEL_EN
_SAFE_NOTE_CODE_RE: Final = re.compile(r"^[A-Z][A-Z0-9_]{2,95}$")
# The hype words a note line names (words of the fixed lists of ``app.idm.policy``, never other provider text).
_MAX_HYPE_WORDS: Final = 4
_MAX_NOTE_CODES: Final = 6


# QLT-9 (QC run ffaf7318: the notes still carried machine codes): what each code means, in a few plain words. The codes
# stay in ``deterministic_codes`` for the tools; the author reads words.
_CODE_TEXT_VI: Final[dict[str, str]] = {
    ANSWER_LEAK_CODE: "phương án lộ đáp án",
    ANSWER_NEAR_EXAMPLE_CODE: "đáp án lặp lại câu chữ ngay phía trên", ANSWER_ITEM_LENGTH_CODE: "ý của đáp án dài nhất",
    ANSWER_LENGTH_CUE_CODE: "đáp án đúng dài hơn hẳn",
    ANSWER_CONTESTED_CODE: "đáp án có thể tranh cãi",
    QUESTION_INCOMPLETE_CODE: "câu hỏi bị cụt",
    "IDM_W5_PRACTICE_INCOMPLETE": "bài luyện tập thiếu phần",
    "IDM_W5_FEEDBACK_NOT_TEACHING": "lời giải chưa giảng giải",
    SENTENCE_COPY_CODE: "câu chép gần nguyên văn tài liệu",
    SENTENCE_LENGTH_CODE: "câu quá dài",
    EXPLANATION_MISSING_CODE: "thiếu phần giảng giải",
    EXAMPLE_ROLE_CODE: "ví dụ chưa đúng vai trò người học",
    NEW_FACT_CODE: "số liệu hoặc tên riêng không có trong tài liệu",
    "IDM_W5_VERBATIM_COPY": "đoạn chép nguyên văn",
    FAQ_RESTATES_HTML_CODE: "hỏi đáp nhắc lại bài",
    FAQ_RESTATES_SOURCE_CODE: "hỏi đáp chép tài liệu",
    FAQ_TYPE_MIX_CODE: "hỏi đáp chỉ hỏi lại kiến thức",
    FAQ_NEW_FACT_CODE: "hỏi đáp có số liệu hoặc tên ngoài tài liệu",
    FAQ_OPEN_QUESTION_CODE: "hỏi đáp trả lời điều tài liệu chưa chốt",
    FAQ_INVENTED_REASON_CODE: "hỏi đáp tự đặt lý do",
    FAQ_TITLE_MISMATCH_CODE: "tiêu đề hỏi đáp chưa khớp",
    FAQ_HYPE_CODE: "hỏi đáp dùng chữ thổi phồng",
    FAQ_ITEMS_DROPPED_CODE: "bỏ bớt câu hỏi đáp",
    CALLOUT_UNGROUNDED_CODE: "khung trích dẫn không có trong tài liệu",
    CALLOUT_TO_PROSE_CODE: "đổi khung trích dẫn thành đoạn văn",
    FRAMEWORK_INCOMPLETE_CODE: "chưa liệt kê đủ thành phần của khung",
    FRAMEWORK_LIST_INSERTED_CODE: "chèn danh sách thành phần",
    SOURCE_ARTIFACT_CODE: "chữ bị cắt như trong PDF",
    WORKSHEET_INCOMPLETE_CODE: "phiếu thực hành thiếu phần",
    WORKSHEET_INPUT_CODE: "phiếu thực hành thiếu dữ liệu đầu vào",
    EXAMPLE_REUSE_CODE: "ví dụ mẫu dùng lại dữ liệu của bài tập",
    "IDM_W5_WORKSHEET_COMPACTED": "rút gọn phiếu thực hành",
    "HTML_INSTRUCTIONAL_DENSITY_EXCEEDED": "phần lý thuyết dài quá giới hạn",
    "IDM_W5_REPAIR_INVALID": "lần sửa của AI sai định dạng",
    "IDM_W5_REPAIR_SKIPPED_BUDGET": "không đủ thời gian để sửa",
    "IDM_DEADLINE_EXCEEDED": "hết thời gian",
    "IDM_W5_INSTANCE_INVALID": "AI trả lời sai định dạng",
    "IDM_RESPONSE_TRUNCATED": "câu trả lời của AI bị cắt",
    "IDM_W5_FALLBACK_ONLY": "lần chạy lại không dùng AI",
    FAQ_FALLBACK_REPEATED_CODE: "hỏi đáp dự phòng lặp lại cùng một câu của tài liệu",
    # QC feedback 2026-10-21 (A1/A2).
    EXPLANATION_LAYOUT_CODE: "lời giải chưa đúng bố cục",
    EXPLANATION_VERDICT_CODE: "lời giải ghi đúng sai ngược với đáp án",
    EXPLANATION_FILLER_CODE: "lời giải dùng chữ sáo rỗng",
    EXPLANATION_LENGTH_CODE: "lời giải quá dài",
    EXPLANATION_RESTATES_CODE: "lời giải chỉ chép lại phương án",
    QUESTION_LAYOUT_CODE: "đề câu hỏi dồn thành một khối",
    QUESTION_LENGTH_CODE: "đề câu hỏi quá dài",
    QUESTION_LIMIT_CODE: "đề câu hỏi dài hơn mức hệ thống nhận",
    QUESTION_ITEM_LETTERS_CODE: "các ý trong đề đặt tên bằng chữ cái của đáp án",
    QUESTION_ITEM_NOUN_CODE: "các ý trong đề đặt tên giống phương án trả lời",
    QUESTION_TWICE_CODE: "đề hỏi hai lần",
    QUESTION_FEW_OPTIONS_CODE: "câu hỏi thiếu phương án",
    EXPLANATION_NAMES_ONLY_CODE: "lời giải chỉ nhắc tên ý, không nói nội dung",
}
_CODE_TEXT_EN: Final[dict[str, str]] = {
    ANSWER_LEAK_CODE: "an option gives the answer away",
    ANSWER_NEAR_EXAMPLE_CODE: "a key in the words just above", ANSWER_ITEM_LENGTH_CODE: "the key's item is the longest",
    ANSWER_LENGTH_CUE_CODE: "the correct option is much longer",
    ANSWER_CONTESTED_CODE: "a contestable answer",
    QUESTION_INCOMPLETE_CODE: "a cut question",
    "IDM_W5_PRACTICE_INCOMPLETE": "an incomplete practice",
    "IDM_W5_FEEDBACK_NOT_TEACHING": "feedback that does not teach",
    SENTENCE_COPY_CODE: "sentences copied from the source",
    SENTENCE_LENGTH_CODE: "sentences that are too long",
    EXPLANATION_MISSING_CODE: "no explanation",
    EXAMPLE_ROLE_CODE: "an example about another role",
    NEW_FACT_CODE: "a number or name the source does not state",
    "IDM_W5_VERBATIM_COPY": "a copied passage",
    FAQ_RESTATES_HTML_CODE: "FAQ that repeats the lesson",
    FAQ_RESTATES_SOURCE_CODE: "FAQ that copies the source",
    FAQ_TYPE_MIX_CODE: "FAQ that only asks for facts",
    FAQ_NEW_FACT_CODE: "FAQ with a number or name outside the source",
    FAQ_OPEN_QUESTION_CODE: "FAQ answering a point the source has not settled",
    FAQ_INVENTED_REASON_CODE: "FAQ with its own reason",
    FAQ_TITLE_MISMATCH_CODE: "an FAQ title that does not match",
    FAQ_HYPE_CODE: "FAQ answers with hype words",
    FAQ_ITEMS_DROPPED_CODE: "FAQ items removed",
    CALLOUT_UNGROUNDED_CODE: "a callout not in the source",
    CALLOUT_TO_PROSE_CODE: "a callout turned into a paragraph",
    FRAMEWORK_INCOMPLETE_CODE: "a framework not listed in full",
    FRAMEWORK_LIST_INSERTED_CODE: "a list of items added",
    SOURCE_ARTIFACT_CODE: "text cut as in the PDF",
    WORKSHEET_INCOMPLETE_CODE: "an incomplete worksheet",
    WORKSHEET_INPUT_CODE: "a worksheet without its input data",
    EXAMPLE_REUSE_CODE: "a worked example on the task's own data",
    "IDM_W5_WORKSHEET_COMPACTED": "a shortened worksheet",
    "HTML_INSTRUCTIONAL_DENSITY_EXCEEDED": "theory over its length",
    "IDM_W5_REPAIR_INVALID": "an AI repair in the wrong format",
    "IDM_W5_REPAIR_SKIPPED_BUDGET": "no time left to repair",
    "IDM_DEADLINE_EXCEEDED": "out of time",
    "IDM_W5_INSTANCE_INVALID": "an AI answer in the wrong format",
    "IDM_RESPONSE_TRUNCATED": "a cut AI answer",
    "IDM_W5_FALLBACK_ONLY": "a rerun without AI",
    FAQ_FALLBACK_REPEATED_CODE: "a fallback FAQ that repeats one source sentence",
    EXPLANATION_LAYOUT_CODE: "feedback not in its layout",
    EXPLANATION_VERDICT_CODE: "feedback that contradicts the answer key",
    EXPLANATION_FILLER_CODE: "feedback with empty praise",
    EXPLANATION_LENGTH_CODE: "feedback that is too long",
    EXPLANATION_RESTATES_CODE: "feedback that only repeats the option",
    QUESTION_LAYOUT_CODE: "a question written as one block",
    QUESTION_LENGTH_CODE: "a question that is too long",
    QUESTION_LIMIT_CODE: "a question longer than the system accepts",
    QUESTION_ITEM_LETTERS_CODE: "items in the question named with the answer letters",
    QUESTION_ITEM_NOUN_CODE: "items in the question named like the answer options",
    QUESTION_TWICE_CODE: "a question asked twice",
    QUESTION_FEW_OPTIONS_CODE: "a question with too few options",
    EXPLANATION_NAMES_ONLY_CODE: "feedback that names an item without its content",
}
# A code without its own words, by family.
_CODE_FAMILY_VI: Final = (("HTML_", "lỗi định dạng phần lý thuyết"), ("DIAGRAM_", "lỗi sơ đồ"),
                          ("FAQ_", "lỗi phần hỏi đáp"), ("IDM_W6_", "kiểm tra chất lượng tự động"),
                          ("IDM_PROVIDER", "AI không phản hồi"), ("AI_PROVIDER", "AI không phản hồi"),
                          ("IDM_BUDGET", "hết hạn mức gọi AI"))
_CODE_FAMILY_EN: Final = (("HTML_", "a theory format problem"), ("DIAGRAM_", "a diagram problem"),
                          ("FAQ_", "an FAQ problem"), ("IDM_W6_", "the automatic review"),
                          ("IDM_PROVIDER", "the AI did not answer"), ("AI_PROVIDER", "the AI did not answer"),
                          ("IDM_BUDGET", "the AI budget ran out"))


def _code_text(code: str, vi: bool) -> str:
    texts = _CODE_TEXT_VI if vi else _CODE_TEXT_EN
    if code in texts:
        return texts[code]
    if (contract := contract_code_text(code, vi)) is not None:  # run 2f8781d6: never "một kiểm tra tự động khác"
        return contract
    family = next((text for prefix, text in (_CODE_FAMILY_VI if vi else _CODE_FAMILY_EN) if code.startswith(prefix)),
                  None)
    return family or ("một kiểm tra tự động khác" if vi else "another automatic check")


def _codes(values: Sequence[str], vi: bool = True) -> str:
    """The plain words of ``values`` (QLT-9), without repeats."""

    codes = [code for code in dict.fromkeys(values) if _SAFE_NOTE_CODE_RE.fullmatch(code)][:_MAX_NOTE_CODES]
    return ", ".join(dict.fromkeys(_code_text(code, vi) for code in codes))


def _block_names(slots: Sequence[int], vi: bool) -> str:
    """"khối 1, khối 3": the blocks without the label of their type (a worksheet is no "Lý thuyết", review 4)."""

    return ", ".join(f"{'khối' if vi else 'block'} {index + 1}" for index in sorted(slots))


def _slot_reasons(slots: Sequence[int], reasons: Mapping[int, Sequence[str]], slot_types: Sequence[str],
                  vi: bool) -> str:
    labels = _SLOT_LABEL_VI if vi else _SLOT_LABEL_EN
    parts = []
    for index in sorted(slots):
        kind = slot_types[index] if 0 <= index < len(slot_types) else ""
        label = f" ({labels[kind]})" if kind in labels else ""
        codes = _codes(reasons.get(index, ()), vi)
        parts.append(f"{'khối' if vi else 'block'} {index + 1}{label}" + (f": {codes}" if codes else ""))
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
    callouts_repeating: int = 0,
    worksheet_compacted: bool = False,
    worksheet_example_kept: bool = False,
    drafted_examples: Mapping[int, int] | None = None,
    example_numbers: Mapping[int, int] | None = None,
    faq_items_copied: int = 0,
    faq_items_recall: int = 0,
    faq_inferred: Mapping[int, int] | None = None,
    faq_items_open: int = 0,
    review_reasons: Sequence[str] = (),
    unexplained_acronyms: Sequence[str] = (),
    notices: Sequence[str] = (),
    suggested_criteria: bool = False,
    source_flags: Sequence[str] = (),
    source_notes: Sequence[str] = (),
    placeholder_slots: Sequence[int] = (),
    glued: Sequence[str] = (),
    suppositions: int = 0,
    fallback_inputs: Sequence[int] = (),
    hype_words: Mapping[int, Sequence[str]] | None = None,
    classroom: Mapping[int, Sequence[str]] | None = None,
) -> str:
    """Template note for the unit ``implementation_notes``; no IDs, no ``<``/``>``.

    Every fallback says why (QC course 234653, D16): the codes behind each replaced or kept slot, the
    provider/time/budget code behind a whole-unit fallback and a failed automatic repair.

    ``remaining`` are the findings of the unit as returned and ``fixed_codes`` the codes found and
    settled during generation (QC course 364564, N8: codes seen before a repair were printed as
    still open). Without ``remaining`` every deterministic code counts as still open.

    ``drafted_examples`` (QLT-5 C): the examples labelled "Ví dụ minh hoạ" per slot; the author checks each one;
    ``example_numbers``: those that state a number the facts do not (owner decision 2026-10-20: a notice, not a review).
    ``faq_items_copied``/``faq_items_recall`` (QLT-5 D): FAQ items dropped because they copied a source sentence or only
    asked for a fact again; ``faq_inferred``: per FAQ slot, the answers that explain beyond the words of their facts
    (the SME checks that they add no rule); ``faq_items_open`` (QLT-7): FAQ items dropped because they answered an
    open SME question.

    ``review_reasons`` (QLT-8): why the unit is marked for review (``REVIEW_REASON_*``), stated first in plain words;
    ``unexplained_acronyms``: abbreviations the learner text uses whose full form neither the facts nor the server's
    short list give, for the SME to spell out. ``notices`` (QLT-9): what the author should know about a unit that is
    not marked for review (``NOTICE_REASONS``), stated as "Lưu ý".

    ``suggested_criteria`` (A5): the unit's worksheet has criteria the AI suggested from what the lesson teaches.
    ``source_flags`` (owner decision 2026-10-22): one sentence per contradictory or out-of-date point of the source,
    right after the review reasons; ``source_notes`` (run 5496155c): one per unclear point, an edit note after the
    notices. ``placeholder_slots`` (run 5496155c): the slots of a whole-unit fallback that only repeat sentences of the
    lesson (no source structure rebuilt them); ``glued``: words the PDF glued together that a fallback slot shows;
    ``suppositions``: drafted cases told as a real past event that now open with "Giả sử"; ``fallback_inputs``
    (review of dfb6db2): fallback worksheets whose input data the practice plan only describes.
    """

    vi = locale == "vi"
    problems = [finding for finding in judge.findings if finding.severity in {"major", "critical"}]
    judged = _judge_line(judge, problems, _CRITERION_LABEL_VI if vi else _CRITERION_LABEL_EN, vi)
    slot_lines, named = _slot_lines(vi, slot_reasons or {}, slot_types, whole_fallback=whole_fallback,
                                    fallback_slots=fallback_slots, review_slots=review_slots,
                                    failure_codes=failure_codes, deterministic_codes=deterministic_codes,
                                    placeholder_slots=placeholder_slots, remaining=remaining,
                                    fallback_inputs=fallback_inputs)
    checks = _Checks("", [], []) if whole_fallback else _check_lines(
        vi, deterministic_codes, remaining, fixed_codes, slot_types, named=named, hype_words=hype_words or {})
    # Run 5496155c: a reason says "chi tiết bên dưới" only when a line below gives the details.
    kept_line = bool(review_slots) and not whole_fallback
    detailed = {reason for reason, shown in (
        (REVIEW_REASON_SOURCE, bool(source_flags)), (REVIEW_REASON_JUDGE, bool(problems) and bool(judged)),
        (REVIEW_REASON_KEPT_SLOTS, kept_line), (REVIEW_REASON_STYLE, kept_line),
        (REVIEW_REASON_SERVER_EDIT, callouts_to_prose > 0 or framework_list_inserted > 0),
        (REVIEW_REASON_EXAMPLE_NUMBERS, bool(example_numbers) and not whole_fallback)) if shown}
    content = not whole_fallback
    # Review of dfb6db2: what the author must check first ("Cần xem"), then what to polish ("Lưu ý", "Nên biên tập"),
    # then what was done; the note's character budget only ever cuts the end.
    # The fallback and kept-slot lines lead a unit marked for review; a unit with notices only opens with them.
    first = bool(review_reasons or fallback_slots or whole_fallback)
    review = [_reasons_line(review_reasons, vi, detailed, _REVIEW_LEAD[vi]),
              *fit_flags(source_flags, IDM_UNIT_SOURCE_FLAGS_MAX_CHARS, locale), judged,
              *(slot_lines if first else ()), *checks.review]
    edits = [_reasons_line(notices, vi, detailed, _NOTICE_LEAD[vi]),
             *fit_flags(source_notes, IDM_UNIT_SOURCE_FLAGS_MAX_CHARS, locale, soft=True),
             *(() if first else slot_lines), *checks.polish,
             *dropped_faq_lines(vi, dropped=faq_items_dropped, restated=faq_items_restated, copied=faq_items_copied,
                                recall=faq_items_recall, unresolved=faq_items_open, invalid=faq_items_invalid),
             *callout_lines(vi, to_prose=callouts_to_prose, repeating=callouts_repeating),
             supposition_line(suppositions, vi) if suppositions else "",
             framework_line(framework_list_inserted, vi) if framework_list_inserted else "",
             worksheet_line(vi, example_kept=worksheet_example_kept) if worksheet_compacted else "",
             drafted_examples_line(drafted_examples, slot_types, vi) if drafted_examples and content else "",
             example_numbers_line(example_numbers, slot_types, vi) if example_numbers and content else "",
             faq_inferred_line(faq_inferred, slot_types, vi) if faq_inferred and content else "",
             # A source-locked unit carries no drafted scenario; the review reason already says it when it is one.
             SCENARIO_LINE[vi] if ai_drafted and content and REVIEW_REASON_SCENARIO not in review_reasons else "",
             _acronyms_line(unexplained_acronyms, vi) if unexplained_acronyms and content else "",
             SUGGESTED_CRITERIA_LINE[vi] if suggested_criteria else "", glued_line(glued, vi) if glued else "",
             classroom_line(classroom, slot_types, vi) if classroom and content else ""]
    # Run 2f8781d6 (A4): never "QA tự động: không phát hiện vấn đề lớn." (it stood on a unit whose question gave its
    # answer away); a clean unit's note says only what was fixed, or nothing.
    return sanitize_author_text(" ".join(part for part in [*review, *edits, checks.fixed] if part),
                                IDM_UNIT_AUTHOR_NOTE_MAX_CHARS)


def _slot_lines(vi: bool, reasons: Mapping[int, Sequence[str]], slot_types: Sequence[str], *, whole_fallback: bool,
                fallback_slots: Sequence[int], review_slots: Sequence[int], failure_codes: Sequence[str],
                deterministic_codes: Sequence[str], placeholder_slots: Sequence[int],
                remaining: Sequence[SlotFinding] | None, fallback_inputs: Sequence[int] = (),
                ) -> tuple[list[str], set[str]]:
    """The lines of the fallback and kept slots and why, and the plain words they name, which the "còn cảnh báo"
    line does not repeat. Run 5496155c: a kept slot's finding with its own "Cần xem"/"Nên biên tập" line below is not
    named twice."""

    if whole_fallback:
        failures = _codes([*failure_codes, *deterministic_codes], vi)
        lines = [("Cả bài dùng bản dự phòng dựng từ tài liệu — cần biên tập." if vi
                  else "The whole lesson uses the source-based fallback — edit it.")
                 + ((f" Lý do: {failures}." if vi else f" Reason: {failures}.") if failures else "")]
        if placeholder_slots:
            lines.append(placeholder_line(_slot_reasons(placeholder_slots, {}, slot_types, vi), vi))
        if fallback_inputs:
            lines.append(fallback_input_line(_block_names(fallback_inputs, vi), vi))
        return lines, set()
    lines = []
    if fallback_slots:
        lines.append(f"{len(fallback_slots)} khối dùng bản dự phòng dựng từ tài liệu — cần biên tập." if vi
                     else f"{len(fallback_slots)} block(s) use the source-based fallback — edit them.")
        explained = [index for index in fallback_slots if _codes(reasons.get(index, ()), vi)]
        if explained:
            lines.append(("Lý do: " if vi else "Reason: ") + _slot_reasons(explained, reasons, slot_types, vi) + ".")
        if placeholder_slots:  # run 2f8781d6: a slot no source structure rebuilds falls back alone
            lines.append(placeholder_line(_slot_reasons(placeholder_slots, {}, slot_types, vi), vi))
        if fallback_inputs:
            lines.append(fallback_input_line(_block_names(fallback_inputs, vi), vi))
    hinted = {(item.component_index, item.code) for item in remaining or ()
              if item.code in _REVIEW_HINT_VI or item.code in _PLAIN_HINT_VI}
    kept = {index: [code for code in reasons.get(index, ()) if (index, code) not in hinted] for index in review_slots}
    if review_slots:
        lines.append(("Giữ bản AI để tác giả rà soát: " if vi else "AI draft kept for author review: ")
                     + _slot_reasons(review_slots, kept, slot_types, vi) + ".")
    repair_failures = _codes(failure_codes, vi)
    if repair_failures and (fallback_slots or review_slots):
        lines.append(f"Sửa tự động không thành công: {repair_failures}." if vi
                     else f"Automatic repair did not succeed: {repair_failures}.")
    named = {_code_text(code, vi) for codes in [*(reasons.get(index, ()) for index in fallback_slots),
                                                *kept.values()] for code in codes}
    return lines, named


def _judge_line(judge: JudgeOutcome, problems: Sequence[Any], labels: Mapping[str, str], vi: bool) -> str:
    """The automatic review's line: its major problems, or why it did not run; nothing when it found none."""

    if judge.status in {"pass", "review_required", "reject"}:
        if problems:
            described = "; ".join(dict.fromkeys(labels[finding.criterion] for finding in problems[:4]))
            return (f"QA tự động: {len(problems)} vấn đề cần xem — {described}." if vi
                    else f"Automatic QA: {len(problems)} issue(s) to review — {described}.")
    elif judge.status == "skipped_budget":
        return "QA tự động: bỏ qua do hết thời gian." if vi else "Automatic QA: skipped (time budget)."
    elif judge.status == "failed":
        return ("QA tự động: không chạy được, cần rà soát thủ công." if vi
                else "Automatic QA: unavailable, review manually.")
    return ""


# Content-mode findings (QLT-5 C) in plain words, without codes.
_PLAIN_HINT_VI: Final = {
    # Run 2f8781d6 (B6): soft give-aways of the key, left after their one repair.
    ANSWER_NEAR_EXAMPLE_CODE: "đáp án đúng (hoặc ý nó chỉ tới) lặp lại câu chữ của ví dụ, bảng hoặc tình huống ngay "
                              "phía trên — người học có thể chọn theo chữ; nên giữ nguyên tắc nhưng đổi chi tiết tình "
                              "huống và cách diễn đạt",
    ANSWER_ITEM_LENGTH_CODE: "ý mà đáp án đúng chỉ tới dài hơn hẳn các ý khác — người học có thể đoán theo độ dài, nên "
                             "cân độ dài các ý",
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
    FAQ_OPEN_QUESTION_CODE: "câu hỏi đáp trả lời một điều tài liệu chưa chốt — chỉ giữ khi tác giả đã kiểm tra",
    FAQ_INVENTED_REASON_CODE: "câu hỏi đáp tự đưa ra lý do cho một con số hoặc ngưỡng mà tài liệu không giải thích — "
                              "tác giả nên kiểm tra lại hoặc bỏ",
    # QC feedback 2026-10-21 (A1/A2): wording of the question and its feedback, to polish.
    EXPLANATION_FILLER_CODE: "lời giải dùng chữ sáo rỗng hoặc thổi phồng{words} — nên thay bằng lý do cụ thể lấy từ "
                             "bài học",
    FAQ_HYPE_CODE: "câu trả lời hỏi đáp dùng chữ thổi phồng{words} — nên thay bằng lời giải thích cụ thể",
    EXPLANATION_LENGTH_CODE: "lời giải dài quá giới hạn — nên rút gọn: một câu nêu nguyên tắc, rồi mỗi phương án một "
                             "câu ngắn",
    QUESTION_LAYOUT_CODE: "đề câu hỏi còn dồn thành một khối — nên để một hai câu bối cảnh, mỗi ý của danh sách một "
                          "dòng và câu hỏi ở dòng cuối",
    QUESTION_LENGTH_CODE: "đề câu hỏi dài — nên rút gọn phần bối cảnh hoặc các ý liệt kê",
    QUESTION_ITEM_LETTERS_CODE: "các ý trong đề được đặt tên bằng chữ A, B, C — dễ nhầm với chữ cái của đáp án, nên "
                                "đổi thành tên khác kèm số (Cách 1, Đề xuất 1) ở cả đề, đáp án và lời giải",
    QUESTION_ITEM_NOUN_CODE: "các ý trong đề được gọi là «Phương án 1, 2…» nên đáp án đọc thành «A: Phương án 2» — "
                             "nên đổi tên ý (Cách 1, Đề xuất 1, Tình huống 1) ở cả đề, đáp án và lời giải",
    QUESTION_TWICE_CODE: "đề hỏi hai lần (câu hỏi ở dòng bối cảnh hoặc trước danh sách và ở dòng cuối) — chỉ giữ một "
                         "câu hỏi ở dòng cuối",
    QUESTION_FEW_OPTIONS_CODE: "câu hỏi chỉ có 3 phương án vì đề chỉ liệt kê 3 ý — nên thêm một ý hợp lý thứ tư và "
                               "phương án, lời giải tương ứng",
    EXPLANATION_NAMES_ONLY_CODE: "đáp án chỉ ghi tên ý (như «Cách 2») mà lời giải không nói ý đó là gì — nên nêu nội "
                                 "dung của ý trong lời giải",
}
_PLAIN_HINT_EN: Final = {
    ANSWER_NEAR_EXAMPLE_CODE: "the correct option (or the item it names) repeats the words of the example, table or "
                              "case just above — learners can match words; keep the principle, change the case "
                              "details and the wording",
    ANSWER_ITEM_LENGTH_CODE: "the item the correct option names is clearly longer than the others — learners can guess "
                             "by length; balance the items",
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
    FAQ_OPEN_QUESTION_CODE: "an FAQ item answers a point the source has not settled — keep it only once the author "
                            "has checked it",
    FAQ_INVENTED_REASON_CODE: "an FAQ item gives its own reason for a number or threshold the source does not "
                              "explain — the author should check it or remove it",
    EXPLANATION_FILLER_CODE: "the feedback uses empty praise or hype{words} — replace it with a concrete reason taken "
                             "from the lesson",
    FAQ_HYPE_CODE: "FAQ answers use hype words{words} — replace them with a plain explanation",
    EXPLANATION_LENGTH_CODE: "the feedback is over its length — shorten it: one sentence with the principle, then one "
                             "short sentence per option",
    QUESTION_LAYOUT_CODE: "the question is still one block of text — keep one or two context sentences, each list item "
                          "on its own line and the question on the last line",
    QUESTION_LENGTH_CODE: "the question is long — shorten the context or the listed items",
    QUESTION_ITEM_LETTERS_CODE: "the items in the question are named with the letters A, B, C, which read like the "
                                "answer letters — give them another name with a number (Approach 1) in the "
                                "question, the options and the feedback",
    QUESTION_ITEM_NOUN_CODE: "the items in the question are called «Option 1, 2…», so a choice reads «A: Option 2» — "
                             "rename them (Approach 1, Proposal 1) in the question, the options and the feedback",
    QUESTION_TWICE_CODE: "the question is asked twice (on a context line or before the list, and on the last line) — "
                         "keep one question on the last line",
    QUESTION_FEW_OPTIONS_CODE: "the question has only 3 options because the case lists 3 items — add a plausible "
                               "fourth item with its option and feedback",
    EXPLANATION_NAMES_ONLY_CODE: "an option only names an item («Approach 2») and its feedback never says what it is — "
                                 "say the item's content in the feedback",
}


_REVIEW_HINT_VI: Final = REVIEW_HINT_VI
_REVIEW_HINT_EN: Final = REVIEW_HINT_EN

# QLT-8 (QC run 0a11aa9e: 3 units marked "Cần xem" without a reason): why a unit needs the author's review, in plain
# words, one per reason the server marks it for. Reasons another note line explains in detail say "chi tiết bên dưới".
REVIEW_REASON_FALLBACK: Final = "fallback"
REVIEW_REASON_KEPT_SLOTS: Final = "kept_slots"
REVIEW_REASON_STEP_INCOMPLETE: Final = "step_incomplete"
REVIEW_REASON_SERVER_EDIT: Final = "server_edit"
REVIEW_REASON_EVIDENCE: Final = "evidence"
REVIEW_REASON_JUDGE: Final = "judge"
REVIEW_REASON_SCENARIO: Final = "ai_scenario"
# Owner decision 2026-10-22: the source of the unit's blocks is contradictory or out of date in places (the note quotes
# each point, ``app.idm.source_flags``); the old W2 rule held such a block for an SME. Since run 5496155c an unclear
# point is an edit note only.
REVIEW_REASON_SOURCE: Final = "source_check"
# QLT-9 (QC run ffaf7318: 14 of 18 units marked, so the mark meant little): reasons that are a notice ("Lưu ý"), not a
# review mark: the source pages with images or tables, and AI slots kept with style findings only (long or copied
# sentences, an FAQ title, an FAQ that recalls) - the content is correct, the author may polish it.
REVIEW_REASON_STYLE: Final = "style_kept"
# Owner decision 2026-10-20: a number the facts do not state inside an AI-drafted example ("Ví dụ minh hoạ: …") is a
# notice; outside one it stays a review finding (IDM_W5_NEW_FACT).
REVIEW_REASON_EXAMPLE_NUMBERS: Final = "example_numbers"
NOTICE_REASONS: Final = frozenset({REVIEW_REASON_EVIDENCE, REVIEW_REASON_STYLE, REVIEW_REASON_EXAMPLE_NUMBERS})
# Findings about the wording, not the correctness: a slot kept with only these is a notice ("Nên biên tập").
STYLE_CODES: Final = frozenset({
    SENTENCE_COPY_CODE, SENTENCE_LENGTH_CODE, "IDM_W5_VERBATIM_COPY", EXPLANATION_MISSING_CODE, EXAMPLE_ROLE_CODE,
    FAQ_TITLE_MISMATCH_CODE, FAQ_TYPE_MIX_CODE, FAQ_RESTATES_HTML_CODE, FAQ_RESTATES_SOURCE_CODE,
    ANSWER_LENGTH_CUE_CODE, FAQ_HYPE_CODE, *QUIZ_STYLE_CODES, ANSWER_NEAR_EXAMPLE_CODE, ANSWER_ITEM_LENGTH_CODE,
})
_REVIEW_REASON_VI: Final = {
    REVIEW_REASON_FALLBACK: "có khối dùng bản dự phòng dựng từ tài liệu",
    REVIEW_REASON_KEPT_SLOTS: "có khối AI viết chưa đạt hết kiểm tra tự động (chi tiết bên dưới)",
    REVIEW_REASON_STEP_INCOMPLETE: "một bước sửa hoặc kiểm tra tự động không hoàn tất",
    REVIEW_REASON_SERVER_EDIT: "hệ thống đã tự chèn hoặc đổi một phần nội dung (chi tiết bên dưới)",
    REVIEW_REASON_EVIDENCE: "tài liệu nguồn của phần này có trang chủ yếu là hình ảnh (bảng hoặc sơ đồ dạng ảnh) hệ "
                            "thống chưa đọc được — cần đối chiếu với tài liệu gốc",
    REVIEW_REASON_JUDGE: "kiểm tra tự động phát hiện vấn đề lớn (chi tiết bên dưới)",
    REVIEW_REASON_SCENARIO: "câu hỏi dùng tình huống do AI soạn, tác giả nên kiểm tra lại tình huống và đáp án",
    REVIEW_REASON_SOURCE: "tài liệu nguồn có chỗ ghi khác nhau hoặc có thể đã cũ (chi tiết bên dưới)",
    REVIEW_REASON_STYLE: "có khối AI nên biên tập lại câu chữ (chi tiết bên dưới)",
    REVIEW_REASON_EXAMPLE_NUMBERS: "ví dụ minh hoạ do AI soạn có số liệu không có trong tài liệu (chi tiết bên dưới)",
}
_REVIEW_REASON_EN: Final = {
    REVIEW_REASON_FALLBACK: "some blocks use the source-based fallback",
    REVIEW_REASON_KEPT_SLOTS: "some AI blocks did not pass every automatic check (details below)",
    REVIEW_REASON_STEP_INCOMPLETE: "an automatic repair or check step did not finish",
    REVIEW_REASON_SERVER_EDIT: "the system inserted or changed part of the content (details below)",
    REVIEW_REASON_EVIDENCE: "the source of this section has pages that are mostly an image (a table or diagram as a "
                            "picture) the system could not read — check them against the original document",
    REVIEW_REASON_JUDGE: "the automatic review found a major issue (details below)",
    REVIEW_REASON_SCENARIO: "the question uses an AI-drafted scenario; the author should check the scenario and "
                            "the answer",
    REVIEW_REASON_SOURCE: "the source differs or may be out of date in places (details below)",
    REVIEW_REASON_STYLE: "some AI blocks need their wording polished (details below)",
    REVIEW_REASON_EXAMPLE_NUMBERS: "AI-drafted illustrative examples state numbers that are not in the source (details "
                                   "below)",
}
_MAX_NOTE_ACRONYMS: Final = 8


_REVIEW_LEAD: Final = {True: "Cần tác giả xem lại vì: ", False: "Needs the author's review because: "}
_NOTICE_LEAD: Final = {True: "Lưu ý: ", False: "Note: "}
_DETAILS_MARK: Final = {True: " (chi tiết bên dưới)", False: " (details below)"}
_FIXED_LEAD: Final = {True: "Đã tự sửa:", False: "Automatically fixed:"}


def _reasons_line(reasons: Sequence[str], vi: bool, detailed: Collection[str], lead: str) -> str:
    """``lead`` and the plain words of ``reasons``; "(chi tiết bên dưới)" only for those in ``detailed`` (a line
    below gives their details, run 5496155c)."""

    texts = _REVIEW_REASON_VI if vi else _REVIEW_REASON_EN
    described = [texts[reason] if reason in detailed else texts[reason].replace(_DETAILS_MARK[vi], "")
                 for reason in dict.fromkeys(reasons) if reason in texts]
    return f"{lead}{'; '.join(described)}." if described else ""


def _acronyms_line(acronyms: Sequence[str], vi: bool) -> str:
    listed = ", ".join(list(dict.fromkeys(acronyms))[:_MAX_NOTE_ACRONYMS])
    return (f"Chữ viết tắt chưa có nghĩa đầy đủ trong tài liệu — tác giả nên bổ sung nghĩa ở lần xuất hiện đầu: "
            f"{listed}." if vi
            else f"Abbreviations without their full form in the source — the author should add it where each first "
                 f"appears: {listed}.")


class _Checks(NamedTuple):
    """The "Đã tự sửa" line, the lines to check ("Cần xem", "còn cảnh báo") and the lines to polish."""

    fixed: str
    review: list[str]
    polish: list[str]


def _check_lines(vi: bool, deterministic_codes: Sequence[str], remaining: Sequence[SlotFinding] | None,
                 fixed_codes: Sequence[str], slot_types: Sequence[str], *, named: Collection[str] = (),
                 hype_words: Mapping[int, Sequence[str]] | None = None) -> _Checks:
    """The "Đã tự sửa" line (settled during generation), one line per hinted finding (to check or to polish), then
    what still warns on the returned unit and no line above names (``named``: the plain words of the slot lines).
    ``hype_words`` (run 2f8781d6): slot -> the hype words its text uses, which their line names."""

    lines: list[str] = []
    polish: list[str] = []
    fixed = _codes(sorted(set(fixed_codes)), vi)
    still = list(deterministic_codes) if remaining is None else [item.code for item in remaining]
    hints = _REVIEW_HINT_VI if vi else _REVIEW_HINT_EN
    labels = _SLOT_LABEL_VI if vi else _SLOT_LABEL_EN
    plain = _PLAIN_HINT_VI if vi else _PLAIN_HINT_EN
    for code, template in [*hints.items(), *plain.items()]:
        slots = sorted({item.component_index for item in remaining or () if item.code == code})
        if not slots:
            continue
        where = ", ".join(f"{'khối' if vi else 'block'} {index + 1}"
                          + (f" ({labels[slot_types[index]]})" if index < len(slot_types)
                             and slot_types[index] in labels else "") for index in slots)
        found = list(dict.fromkeys(word for index in slots for word in (hype_words or {}).get(index, ())))
        hint = template.replace("{words}", f" ({', '.join(f'«{word}»' for word in found[:_MAX_HYPE_WORDS])})"
                                if found else "")
        # QLT-9: plain words only, the code stays in ``deterministic_codes``; a wording finding is to polish.
        if code in STYLE_CODES:
            polish.append(f"Nên biên tập {where}: {hint}." if vi else f"Polish {where}: {hint}.")
        else:
            lines.append(f"Cần xem {where}: {hint}." if vi else f"Review {where}: {hint}.")
    warn = _codes(sorted({code for code in still if (remaining is None or (code not in hints and code not in plain))
                          and _code_text(code, vi) not in named}), vi)
    if warn:
        lines.append(f"Kiểm tra tự động còn cảnh báo: {warn}." if vi else f"Automatic checks still warn: {warn}.")
    return _Checks((f"Đã tự sửa: {fixed}." if vi else f"Automatically fixed: {fixed}.") if fixed else "", lines, polish)


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


# JudgeMode, criteria_summary and worst_counts moved to app.idm.qa_judge; app.idm.storyboard and the tests still
# import them from here.
__all__ = [
    "ANSWER_CONTESTED_CODE", "ANSWER_LEAK_CODE", "CALLOUT_KIND", "CALLOUT_TO_PROSE_CODE", "CALLOUT_UNGROUNDED_CODE",
    "CRITERIA", "EXPLAINED_CRITERION", "FAQ_ITEMS_DROPPED_CODE", "FAQ_RESTATES_HTML_CODE",
    "FAQ_TITLE_MISMATCH_CODE", "FAQ_VALUE_CRITERION", "FRAMEWORK_INCOMPLETE_CODE", "FRAMEWORK_LIST_INSERTED_CODE",
    "JUDGE_CRITERIA", "NOTICE_REASONS", "NOT_APPLICABLE", "PRACTICE_CRITERIA", "PURPOSE_CRITERION",
    "REVIEW_REASON_EVIDENCE", "REVIEW_REASON_EXAMPLE_NUMBERS", "REVIEW_REASON_FALLBACK", "REVIEW_REASON_JUDGE",
    "REVIEW_REASON_KEPT_SLOTS",
    "REVIEW_REASON_SCENARIO", "REVIEW_REASON_SERVER_EDIT", "REVIEW_REASON_SOURCE", "REVIEW_REASON_STEP_INCOMPLETE",
    "REVIEW_REASON_STYLE",
    "SOURCE_ARTIFACT_CODE", "TITLE_CRITERION", "WORKSHEET_INCOMPLETE_CODE", "JudgeMode", "JudgeOutcome", "SlotFinding",
    "advisory_slot_findings", "blocking_count", "build_unit_author_note", "build_unit_quality",
    "callouts_as_paragraphs", "copied_options", "criteria_summary", "deterministic_slot_findings", "faq_title_mismatch",
    "final_unit_findings", "has_practice_slot", "html_before", "learner_view", "repair_targets",
    "restated_faq_items", "reworded_key", "run_judge", "settle_applicability", "settled_codes",
    "unexplained_options", "ungrounded_callouts", "worksheet_complete", "worst_counts",
]
