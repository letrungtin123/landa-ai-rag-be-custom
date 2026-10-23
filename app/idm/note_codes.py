"""Plain words for the shared-validator and Node contract codes in a unit note (run 2f8781d6, A4).

The note of c2.l3.u3 read "Lý do: khối 2 (Quiz): một kiểm tra tự động khác." for ``PROBLEM_CHOICE_INVALID`` (an option
under 8 characters). Every code the shared staged validator, its provider binding and the mirror of Node's acceptance
can report now has a few plain words (``contract_code_text``): its own for the common ones, else those of its family;
the codes stay in ``deterministic_codes`` for the tools.
"""

from __future__ import annotations

from typing import Final

_CODE_TEXT_VI: Final = {
    "PROBLEM_CHOICE_INVALID": "một phương án trả lời quá ngắn hoặc thiếu nội dung",
    "PROBLEM_CHOICES_INVALID": "một phương án trả lời quá ngắn hoặc thiếu nội dung",
    "PROBLEM_CHOICE_COUNT": "câu hỏi có số phương án không hợp lệ",
    "PROBLEM_CHOICES_NOT_DISTINCT": "hai phương án trả lời gần như trùng nhau",
    "PROBLEM_DUPLICATE_CHOICES": "hai phương án trả lời trùng nhau",
    "PROBLEM_CORRECT_COUNT": "câu hỏi không có đúng một đáp án",
    "PROBLEM_CORRECT_ANSWER_INVALID": "câu hỏi không có đúng một đáp án",
    "PROBLEM_CORRECT_BOILERPLATE": "đáp án đúng không phải nội dung bài học",
    "PROBLEM_CONTENT_INCOMPLETE": "câu hỏi thiếu đề hoặc thiếu phương án",
    "PROBLEM_QUESTION_REQUIRED": "câu hỏi thiếu đề",
    "PROBLEM_QUESTION_INCOMPLETE": "đề câu hỏi quá ngắn",
    "PROBLEM_EXPLANATION_MISSING": "câu hỏi thiếu lời giải",
    "PROBLEM_EXPLANATION_INCOMPLETE": "lời giải quá ngắn",
    "HTML_DUPLICATE_BLOCK": "phần lý thuyết lặp lại cùng một đoạn, ý hoặc ô bảng",
    "REQUIRED_ARTIFACT_NOT_PRESERVED": "thiếu danh sách hoặc bảng mà kế hoạch của bài yêu cầu",
    "IDM_NODE_ACCEPTANCE_REJECTED": "hệ thống quản lý khoá học không nhận nội dung này",
    "IDM_W5_BRIEF_CONTRACT_MISMATCH": "khối chưa khớp kế hoạch của bài",
}
_CODE_TEXT_EN: Final = {
    "PROBLEM_CHOICE_INVALID": "an answer option that is too short or empty",
    "PROBLEM_CHOICES_INVALID": "an answer option that is too short or empty",
    "PROBLEM_CHOICE_COUNT": "a question with a wrong number of options",
    "PROBLEM_CHOICES_NOT_DISTINCT": "two answer options that are nearly the same",
    "PROBLEM_DUPLICATE_CHOICES": "two answer options that are the same",
    "PROBLEM_CORRECT_COUNT": "a question without exactly one correct option",
    "PROBLEM_CORRECT_ANSWER_INVALID": "a question without exactly one correct option",
    "PROBLEM_CORRECT_BOILERPLATE": "a correct option that is no lesson content",
    "PROBLEM_CONTENT_INCOMPLETE": "a question without its stem or its options",
    "PROBLEM_QUESTION_REQUIRED": "a question without its stem",
    "PROBLEM_QUESTION_INCOMPLETE": "a question stem that is too short",
    "PROBLEM_EXPLANATION_MISSING": "a question without its feedback",
    "PROBLEM_EXPLANATION_INCOMPLETE": "feedback that is too short",
    "HTML_DUPLICATE_BLOCK": "theory that repeats the same paragraph, item or table cell",
    "REQUIRED_ARTIFACT_NOT_PRESERVED": "a list or table the unit plan asks for is missing",
    "IDM_NODE_ACCEPTANCE_REJECTED": "the course system does not accept this content",
    "IDM_W5_BRIEF_CONTRACT_MISMATCH": "a block that does not match the unit plan",
}
# By family (the longest prefix first).
_FAMILY_VI: Final = (
    ("IDM_HTML_", "lỗi định dạng phần lý thuyết"), ("PROBLEM_", "lỗi câu hỏi trắc nghiệm"),
    ("SORTABLE_", "lỗi bài sắp xếp"), ("CROSSWORD_", "lỗi ô chữ"),
    ("WORKSPACE_", "nội dung khối không đúng định dạng hệ thống nhận"),
    ("LEARNER_CONTENT_", "nội dung cho học viên còn dấu vết của tài liệu (tên tệp, số trang)"),
    ("SUPPORTING_COMPONENT_", "khối chưa khớp kế hoạch của bài"), ("COMPONENT_", "khối chưa khớp kế hoạch của bài"),
    ("RESPONSE_", "khối chưa khớp kế hoạch của bài"), ("APPROVED_", "khối chưa khớp kế hoạch của bài"),
    ("UNIT_", "bài chưa khớp kế hoạch"), ("CHAPTER_", "bài chưa khớp kế hoạch"),
    ("INSTANCE_", "nội dung AI trả về sai cấu trúc"),
)
_FAMILY_EN: Final = (
    ("IDM_HTML_", "a theory format problem"), ("PROBLEM_", "a quiz problem"),
    ("SORTABLE_", "a sorting exercise problem"),
    ("CROSSWORD_", "a crossword problem"), ("WORKSPACE_", "a block in a format the system does not accept"),
    ("LEARNER_CONTENT_", "learner text with traces of the source (file names, page numbers)"),
    ("SUPPORTING_COMPONENT_", "a block that does not match the unit plan"),
    ("COMPONENT_", "a block that does not match the unit plan"),
    ("RESPONSE_", "a block that does not match the unit plan"),
    ("APPROVED_", "a block that does not match the unit plan"), ("UNIT_", "a unit that does not match its plan"),
    ("CHAPTER_", "a unit that does not match its plan"), ("INSTANCE_", "an AI answer in the wrong structure"),
)
# The shape checks of the shared validator and of Node's normalisation (a field, a list, an id array …).
_STRUCTURE_CODES: Final = frozenset({
    "ARRAY_REQUIRED", "CARDINALITY", "DUPLICATE_REFERENCE", "EMPTY_STRING_ITEM", "FIELD_MISSING", "FIELD_REQUIRED",
    "FIELD_TYPE_INVALID", "INSTRUCTIONAL_OUTPUT_BUDGET_INVALID", "INVALID_COVERAGE_ID_ARRAY", "INVALID_ENUM",
    "INVALID_FACT_ID_ARRAY", "INVALID_ID_ARRAY", "LESSON_VALIDATION_FAILED", "MEMBERSHIP_MISMATCH",
    "MISSING_OR_INVALID_UNIT", "MISSING_OWNED_REFERENCE", "NEGATIVE_INDEX", "NON_STRING_ITEM",
    "NORMALIZATION_COMPONENT_TYPE", "ORDER_ONLY", "OUT_OF_SCOPE_REFERENCE", "RESPONSE_SCHEMA_INVALID", "TEXT_TOO_LONG",
    "TEXT_TOO_SHORT",
})
_STRUCTURE_TEXT: Final = {True: "nội dung AI trả về sai cấu trúc", False: "an AI answer in the wrong structure"}


def contract_code_text(code: str, vi: bool) -> str | None:
    """The plain words of a validator or contract ``code``, or ``None`` for a code of another kind."""

    texts = _CODE_TEXT_VI if vi else _CODE_TEXT_EN
    if code in texts:
        return texts[code]
    if code in _STRUCTURE_CODES:
        return _STRUCTURE_TEXT[vi]
    return next((text for prefix, text in (_FAMILY_VI if vi else _FAMILY_EN) if code.startswith(prefix)), None)


__all__ = ["contract_code_text"]
