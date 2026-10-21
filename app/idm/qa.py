"""W5 deterministic checks and the W6 judge (spec §7.7.2(b), §7.7.5).

The checks live in ``app.idm.qa_checks`` and the judge in ``app.idm.qa_judge``; this module builds the unit
author note and keeps the names other modules import from ``app.idm.qa``.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from typing import Final

from app.idm.faq_checks import (
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
from app.idm.policy import IDM_UNIT_AUTHOR_NOTE_MAX_CHARS
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
from app.idm.source_locked import FAQ_FALLBACK_REPEATED_CODE
from app.idm.text import sanitize_author_text

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


_SLOT_LABEL_VI: Final = {"html": "Lý thuyết", "problem": "Quiz", "la_faq": "Hỏi đáp", "la_sortable": "Sắp xếp",
                         "la_crossword": "Ô chữ", "la_diagram": "Sơ đồ"}
_SLOT_LABEL_EN: Final = {"html": "Theory", "problem": "Quiz", "la_faq": "FAQ", "la_sortable": "Sortable",
                         "la_crossword": "Crossword", "la_diagram": "Diagram"}
_SAFE_NOTE_CODE_RE: Final = re.compile(r"^[A-Z][A-Z0-9_]{2,95}$")
_MAX_NOTE_CODES: Final = 6


# QLT-9 (QC run ffaf7318: the notes still carried machine codes): what each code means, in a few plain words. The codes
# stay in ``deterministic_codes`` for the tools; the author reads words.
_CODE_TEXT_VI: Final[dict[str, str]] = {
    ANSWER_LEAK_CODE: "phương án lộ đáp án",
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
    FAQ_OPEN_QUESTION_CODE: "hỏi đáp trả lời câu còn chờ SME",
    FAQ_INVENTED_REASON_CODE: "hỏi đáp tự đặt lý do",
    FAQ_TITLE_MISMATCH_CODE: "tiêu đề hỏi đáp chưa khớp",
    FAQ_ITEMS_DROPPED_CODE: "bỏ bớt câu hỏi đáp",
    CALLOUT_UNGROUNDED_CODE: "khung trích dẫn không có trong tài liệu",
    CALLOUT_TO_PROSE_CODE: "đổi khung trích dẫn thành đoạn văn",
    FRAMEWORK_INCOMPLETE_CODE: "chưa liệt kê đủ thành phần của khung",
    FRAMEWORK_LIST_INSERTED_CODE: "chèn danh sách thành phần",
    SOURCE_ARTIFACT_CODE: "chữ bị cắt như trong PDF",
    WORKSHEET_INCOMPLETE_CODE: "phiếu thực hành thiếu phần",
    "IDM_W5_WORKSHEET_COMPACTED": "rút gọn phiếu thực hành",
    "HTML_INSTRUCTIONAL_DENSITY_EXCEEDED": "phần lý thuyết dài quá giới hạn",
    "IDM_W5_REPAIR_INVALID": "lần sửa của AI sai định dạng",
    "IDM_W5_REPAIR_SKIPPED_BUDGET": "không đủ thời gian để sửa",
    "IDM_DEADLINE_EXCEEDED": "hết thời gian",
    "IDM_W5_INSTANCE_INVALID": "AI trả lời sai định dạng",
    "IDM_RESPONSE_TRUNCATED": "câu trả lời của AI bị cắt",
    "IDM_W5_FALLBACK_ONLY": "lần chạy lại không dùng AI",
    FAQ_FALLBACK_REPEATED_CODE: "hỏi đáp dự phòng lặp lại cùng một câu của tài liệu",
}
_CODE_TEXT_EN: Final[dict[str, str]] = {
    ANSWER_LEAK_CODE: "an option gives the answer away",
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
    FAQ_OPEN_QUESTION_CODE: "FAQ answering an open SME question",
    FAQ_INVENTED_REASON_CODE: "FAQ with its own reason",
    FAQ_TITLE_MISMATCH_CODE: "an FAQ title that does not match",
    FAQ_ITEMS_DROPPED_CODE: "FAQ items removed",
    CALLOUT_UNGROUNDED_CODE: "a callout not in the source",
    CALLOUT_TO_PROSE_CODE: "a callout turned into a paragraph",
    FRAMEWORK_INCOMPLETE_CODE: "a framework not listed in full",
    FRAMEWORK_LIST_INSERTED_CODE: "a list of items added",
    SOURCE_ARTIFACT_CODE: "text cut as in the PDF",
    WORKSHEET_INCOMPLETE_CODE: "an incomplete worksheet",
    "IDM_W5_WORKSHEET_COMPACTED": "a shortened worksheet",
    "HTML_INSTRUCTIONAL_DENSITY_EXCEEDED": "theory over its length",
    "IDM_W5_REPAIR_INVALID": "an AI repair in the wrong format",
    "IDM_W5_REPAIR_SKIPPED_BUDGET": "no time left to repair",
    "IDM_DEADLINE_EXCEEDED": "out of time",
    "IDM_W5_INSTANCE_INVALID": "an AI answer in the wrong format",
    "IDM_RESPONSE_TRUNCATED": "a cut AI answer",
    "IDM_W5_FALLBACK_ONLY": "a rerun without AI",
    FAQ_FALLBACK_REPEATED_CODE: "a fallback FAQ that repeats one source sentence",
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
    family = next((text for prefix, text in (_CODE_FAMILY_VI if vi else _CODE_FAMILY_EN) if code.startswith(prefix)),
                  None)
    return family or ("một kiểm tra tự động khác" if vi else "another automatic check")


def _codes(values: Sequence[str], vi: bool = True) -> str:
    """The plain words of ``values`` (QLT-9), without repeats."""

    codes = [code for code in dict.fromkeys(values) if _SAFE_NOTE_CODE_RE.fullmatch(code)][:_MAX_NOTE_CODES]
    return ", ".join(dict.fromkeys(_code_text(code, vi) for code in codes))


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

    ``suggested_criteria`` (A5): the unit's practice has criteria the AI suggested from what the lesson teaches.
    """

    vi = locale == "vi"
    labels = _CRITERION_LABEL_VI if vi else _CRITERION_LABEL_EN
    reasons = slot_reasons or {}
    problems = [finding for finding in judge.findings if finding.severity in {"major", "critical"}]
    parts: list[str] = []
    if line := _review_reasons_line(review_reasons, vi):
        parts.append(line)
    if line := _notices_line(notices, vi):
        parts.append(line)
    if judge.status in {"pass", "review_required", "reject"}:
        if problems:
            described = "; ".join(dict.fromkeys(labels[finding.criterion] for finding in problems[:4]))
            parts.append(f"QA tự động: {len(problems)} vấn đề cần xem — {described}." if vi
                         else f"Automatic QA: {len(problems)} issue(s) to review — {described}.")
        elif not review_reasons:
            # QLT-9: never "no major issue" on a unit marked for review.
            parts.append("QA tự động: không phát hiện vấn đề lớn." if vi
                         else "Automatic QA: no major issue found.")
    elif judge.status == "skipped_budget":
        parts.append("QA tự động: bỏ qua do hết thời gian." if vi else "Automatic QA: skipped (time budget).")
    elif judge.status == "failed":
        parts.append("QA tự động: không chạy được, cần rà soát thủ công." if vi
                     else "Automatic QA: unavailable, review manually.")
    failures = _codes([*failure_codes, *deterministic_codes], vi)
    if whole_fallback:
        parts.append(("Cả bài dùng bản dự phòng dựng từ tài liệu — cần biên tập." if vi
                      else "The whole lesson uses the source-based fallback — edit it.")
                     + ((f" Lý do: {failures}." if vi else f" Reason: {failures}.") if failures else ""))
    elif fallback_slots:
        parts.append(f"{len(fallback_slots)} khối dùng bản dự phòng dựng từ tài liệu — cần biên tập." if vi
                     else f"{len(fallback_slots)} block(s) use the source-based fallback — edit them.")
        explained = [index for index in fallback_slots if _codes(reasons.get(index, ()), vi)]
        if explained:
            parts.append(("Lý do: " if vi else "Reason: ") + _slot_reasons(explained, reasons, slot_types, vi) + ".")
    if review_slots and not whole_fallback:
        parts.append(("Giữ bản AI để tác giả rà soát: " if vi else "AI draft kept for author review: ")
                     + _slot_reasons(review_slots, reasons, slot_types, vi) + ".")
    repair_failures = _codes(failure_codes, vi)
    if repair_failures and not whole_fallback and (fallback_slots or review_slots):
        parts.append(f"Sửa tự động không thành công: {repair_failures}." if vi
                     else f"Automatic repair did not succeed: {repair_failures}.")
    parts.extend(_dropped_faq_lines(vi, dropped=faq_items_dropped, restated=faq_items_restated,
                                    copied=faq_items_copied, recall=faq_items_recall, unresolved=faq_items_open,
                                    invalid=faq_items_invalid))
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
        parts.append(_worksheet_line(vi, example_kept=worksheet_example_kept))
    if not whole_fallback:
        parts.extend(_check_lines(vi, deterministic_codes, remaining, fixed_codes, slot_types))
    if drafted_examples and not whole_fallback:
        parts.append(_drafted_examples_line(drafted_examples, slot_types, vi))
    if example_numbers and not whole_fallback:
        parts.append(_example_numbers_line(example_numbers, slot_types, vi))
    if faq_inferred and not whole_fallback:
        parts.append(_faq_inferred_line(faq_inferred, slot_types, vi))
    # A source-locked unit carries no drafted scenario; the review reason already says it when it is one.
    if ai_drafted and not whole_fallback and REVIEW_REASON_SCENARIO not in review_reasons:
        parts.append("Tình huống minh hoạ do AI soạn, cần SME xác nhận tính thực tế." if vi
                     else "The illustrative scenario was drafted by AI; the SME should confirm it is realistic.")
    if unexplained_acronyms and not whole_fallback:
        parts.append(_acronyms_line(unexplained_acronyms, vi))
    if suggested_criteria:
        parts.append(_SUGGESTED_CRITERIA_LINE[vi])
    return sanitize_author_text(" ".join(parts), IDM_UNIT_AUTHOR_NOTE_MAX_CHARS)


def _worksheet_line(vi: bool, *, example_kept: bool) -> str:
    """The note of a worksheet the server shortened to fit its budget (QC run 8de1c76b, Q2; QLT-9)."""

    if example_kept:
        return ("Phiếu thực hành dài hơn giới hạn nên đã rút gọn gợi ý trong bảng (ví dụ mẫu vẫn giữ, có thể còn hai "
                "dòng) — tác giả có thể bổ sung lại gợi ý." if vi
                else "The worksheet was over its length budget, so the guidance in its table was shortened (the worked "
                     "example stays, possibly with two entries) — the author may add the guidance back.")
    return ("Phiếu thực hành dài hơn giới hạn nên đã bỏ phần ví dụ mẫu (hoặc rút gọn gợi ý trong bảng) — tác giả có "
            "thể bổ sung lại một ví dụ ngắn." if vi
            else "The worksheet was over its length budget, so its worked example (or the extra guidance in its table) "
                 "was left out — the author may add a short example back.")


# A5 (owner decision 2026-10-22): the practice of the unit has criteria the AI suggested; keyed by "Vietnamese".
_SUGGESTED_CRITERIA_LINE: Final = {
    True: "Tài liệu không nêu tiêu chí chấm cho bài thực hành này: AI gợi ý tiêu chí từ nội dung đã dạy (ghi là tiêu "
          "chí gợi ý) — tác giả nên xem lại.",
    False: "The source states no grading criteria for this practice: the AI suggested criteria from what the lesson "
           "teaches (marked as suggested criteria) — the author should check them.",
}


def _drafted_examples_line(examples: Mapping[int, int], slot_types: Sequence[str], vi: bool) -> str:
    """One plain line per unit: how many AI-drafted examples, and in which slots."""

    labels = _SLOT_LABEL_VI if vi else _SLOT_LABEL_EN
    total = sum(examples.values())
    where = ", ".join(f"{'khối' if vi else 'block'} {index + 1}"
                      + (f" ({labels[slot_types[index]]})" if index < len(slot_types) and slot_types[index] in labels
                         else "") for index in sorted(examples))
    return (f"Ví dụ minh hoạ do AI soạn — cần kiểm tra: {total} ví dụ ({where})." if vi
            else f"Illustrative examples drafted by AI — please check: {total} example(s) ({where}).")


def _example_numbers_line(numbers: Mapping[int, int], slot_types: Sequence[str], vi: bool) -> str:
    """One plain line: the AI-drafted examples that state a number the facts do not (a notice, never a review mark)."""

    labels = _SLOT_LABEL_VI if vi else _SLOT_LABEL_EN
    where = ", ".join(f"{'khối' if vi else 'block'} {index + 1}"
                      + (f" ({labels[slot_types[index]]})" if index < len(slot_types) and slot_types[index] in labels
                         else "") for index in sorted(numbers))
    total = sum(numbers.values())
    return (f"Ví dụ minh hoạ do AI soạn có số liệu không có trong tài liệu, chỉ để minh hoạ — cần kiểm tra lại số "
            f"liệu: {total} ví dụ ({where})." if vi
            else f"Illustrative examples drafted by AI state numbers that are not in the source, for illustration only "
                 f"— check the numbers: {total} example(s) ({where}).")


def _dropped_faq_lines(vi: bool, *, dropped: int, restated: int, copied: int, recall: int, unresolved: int,
                       invalid: int) -> list[str]:
    """The "Đã bỏ …" lines of the FAQ items removed during generation, one per reason."""

    lines = [
        (dropped, f"Đã bỏ {dropped} câu hỏi đáp có nội dung ngoài tài liệu." if vi
         else f"{dropped} FAQ item(s) with claims outside the source were removed."),
        (restated, f"Đã bỏ {restated} câu hỏi đáp chỉ nhắc lại nội dung vừa học." if vi
         else f"{restated} FAQ item(s) that only repeated the content just taught were removed."),
        (copied, f"Đã bỏ {copied} câu hỏi đáp gần như chép nguyên câu tài liệu." if vi
         else f"{copied} FAQ item(s) that nearly copied a source sentence were removed."),
        (recall, f"Đã bỏ {recall} câu hỏi đáp chỉ hỏi lại kiến thức đã học." if vi
         else f"{recall} FAQ item(s) that only asked for a fact again were removed."),
        (unresolved, f"Đã bỏ {unresolved} câu hỏi đáp trả lời câu hỏi còn chờ SME xác nhận." if vi
         else f"{unresolved} FAQ item(s) that answered a question still open for the SME were removed."),
        (invalid, f"Đã bỏ {invalid} câu hỏi đáp hệ thống không lưu được (ký tự góc nhọn, câu hỏi trùng)." if vi
         else f"{invalid} FAQ item(s) the course editor cannot store (angle brackets, repeated question) were "
              "removed."),
    ]
    return [line for count, line in lines if count]


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
    ANSWER_LENGTH_CUE_CODE,
})
_REVIEW_REASON_VI: Final = {
    REVIEW_REASON_FALLBACK: "có khối dùng bản dự phòng dựng từ tài liệu",
    REVIEW_REASON_KEPT_SLOTS: "có khối AI viết chưa đạt hết kiểm tra tự động (chi tiết bên dưới)",
    REVIEW_REASON_STEP_INCOMPLETE: "một bước sửa hoặc kiểm tra tự động không hoàn tất",
    REVIEW_REASON_SERVER_EDIT: "hệ thống đã tự chèn hoặc đổi một phần nội dung (chi tiết bên dưới)",
    REVIEW_REASON_EVIDENCE: "tài liệu nguồn của phần này có hình ảnh hoặc bảng hệ thống chưa đọc chắc chắn — cần đối "
                            "chiếu với tài liệu gốc",
    REVIEW_REASON_JUDGE: "kiểm tra tự động phát hiện vấn đề lớn (chi tiết bên dưới)",
    REVIEW_REASON_SCENARIO: "câu hỏi dùng tình huống do AI soạn, SME cần xác nhận tình huống và đáp án",
    REVIEW_REASON_STYLE: "có khối AI còn câu dài hoặc câu gần chép tài liệu, nên biên tập lại (chi tiết bên dưới)",
    REVIEW_REASON_EXAMPLE_NUMBERS: "ví dụ minh hoạ do AI soạn có số liệu không có trong tài liệu (chi tiết bên dưới)",
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
    REVIEW_REASON_STYLE: "some AI blocks keep long or copied sentences; polish them (details below)",
    REVIEW_REASON_EXAMPLE_NUMBERS: "AI-drafted illustrative examples state numbers that are not in the source (details "
                                   "below)",
}
_MAX_NOTE_ACRONYMS: Final = 8


def _review_reasons_line(reasons: Sequence[str], vi: bool) -> str:
    texts = _REVIEW_REASON_VI if vi else _REVIEW_REASON_EN
    described = [texts[reason] for reason in dict.fromkeys(reasons) if reason in texts]
    if not described:
        return ""
    return ("Cần tác giả xem lại vì: " if vi else "Needs the author's review because: ") + "; ".join(described) + "."


def _notices_line(notices: Sequence[str], vi: bool) -> str:
    texts = _REVIEW_REASON_VI if vi else _REVIEW_REASON_EN
    described = [texts[reason] for reason in dict.fromkeys(notices) if reason in texts]
    return (("Lưu ý: " if vi else "Note: ") + "; ".join(described) + ".") if described else ""


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
    fixed = _codes(sorted(set(fixed_codes)), vi)
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
        # QLT-9: plain words only, the code stays in ``deterministic_codes``; a wording finding is to polish.
        if code in STYLE_CODES:
            lines.append(f"Nên biên tập {where}: {hint}." if vi else f"Polish {where}: {hint}.")
        else:
            lines.append(f"Cần xem {where}: {hint}." if vi else f"Review {where}: {hint}.")
    warn = _codes(sorted({code for code in still if remaining is None or (code not in hints and code not in plain)}),
                  vi)
    if warn:
        lines.append(f"Kiểm tra tự động còn cảnh báo: {warn}." if vi else f"Automatic checks still warn: {warn}.")
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


# JudgeMode, criteria_summary and worst_counts moved to app.idm.qa_judge; app.idm.storyboard and the tests still
# import them from here.
__all__ = [
    "ANSWER_CONTESTED_CODE", "ANSWER_LEAK_CODE", "CALLOUT_KIND", "CALLOUT_TO_PROSE_CODE", "CALLOUT_UNGROUNDED_CODE",
    "CRITERIA", "EXPLAINED_CRITERION", "FAQ_ITEMS_DROPPED_CODE", "FAQ_RESTATES_HTML_CODE",
    "FAQ_TITLE_MISMATCH_CODE", "FAQ_VALUE_CRITERION", "FRAMEWORK_INCOMPLETE_CODE", "FRAMEWORK_LIST_INSERTED_CODE",
    "JUDGE_CRITERIA", "NOTICE_REASONS", "NOT_APPLICABLE", "PRACTICE_CRITERIA", "PURPOSE_CRITERION",
    "REVIEW_REASON_EVIDENCE", "REVIEW_REASON_EXAMPLE_NUMBERS", "REVIEW_REASON_FALLBACK", "REVIEW_REASON_JUDGE",
    "REVIEW_REASON_KEPT_SLOTS",
    "REVIEW_REASON_SCENARIO", "REVIEW_REASON_SERVER_EDIT", "REVIEW_REASON_STEP_INCOMPLETE", "REVIEW_REASON_STYLE",
    "SOURCE_ARTIFACT_CODE", "TITLE_CRITERION", "WORKSHEET_INCOMPLETE_CODE", "JudgeMode", "JudgeOutcome", "SlotFinding",
    "advisory_slot_findings", "blocking_count", "build_unit_author_note", "build_unit_quality",
    "callouts_as_paragraphs", "copied_options", "criteria_summary", "deterministic_slot_findings", "faq_title_mismatch",
    "final_unit_findings", "has_practice_slot", "html_before", "learner_view", "repair_targets",
    "restated_faq_items", "reworded_key", "run_judge", "settle_applicability", "settled_codes",
    "unexplained_options", "ungrounded_callouts", "worksheet_complete", "worst_counts",
]
