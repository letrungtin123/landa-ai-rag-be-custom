"""Lines of the unit author note, one per topic: what the server changed (callouts, a shortened worksheet, FAQ items
dropped) and what the author should check (AI-drafted examples, their numbers, inferred FAQ answers, suggested
criteria). Moved verbatim out of ``app.idm.qa`` (STD-5 file size); ``qa.build_unit_author_note`` puts them in order.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

SLOT_LABEL_VI: Final = {"html": "Lý thuyết", "problem": "Quiz", "la_faq": "Hỏi đáp", "la_sortable": "Sắp xếp",
                         "la_crossword": "Ô chữ", "la_diagram": "Sơ đồ"}
SLOT_LABEL_EN: Final = {"html": "Theory", "problem": "Quiz", "la_faq": "FAQ", "la_sortable": "Sortable",
                         "la_crossword": "Crossword", "la_diagram": "Diagram"}


def callout_lines(vi: bool, *, to_prose: int, repeating: int) -> list[str]:
    """The notes of callouts the server changed: ungrounded ones turned into paragraphs (N9), and those that only
    repeated the text of their slot, removed (QC feedback A3)."""

    lines = []
    if to_prose:
        lines.append(f"Đã chuyển {to_prose} khung trích dẫn/lưu ý không có trong tài liệu thành đoạn văn thường — tác "
                     "giả nên kiểm tra lại nội dung." if vi
                     else f"{to_prose} callout(s) not found in the source were turned into plain paragraphs — the "
                          "author should check their content.")
    if repeating:
        lines.append(f"Đã bỏ {repeating} khung trích dẫn/lưu ý chỉ nhắc lại đoạn văn có sẵn trong bài." if vi
                     else f"{repeating} callout(s) that only repeated the text around them were removed.")
    return lines


SCENARIO_LINE: Final = {
    True: "Tình huống minh hoạ do AI soạn, tác giả nên kiểm tra lại tính thực tế.",
    False: "The illustrative scenario was drafted by AI; the author should check it is realistic.",
}


def framework_line(count: int, vi: bool) -> str:
    """The note of the framework list the server inserted into the theory block (QC run 8de1c76b, Q3)."""

    return (f"Đã chèn danh sách {count} thành phần của khung (tên lấy từ tài liệu) vào phần lý thuyết — tác giả nên "
            "biên tập cho liền mạch." if vi
            else f"A list of the framework's {count} items (names from the source) was inserted into the theory "
                 "block — edit it into the text.")


def fallback_input_line(where: str, vi: bool) -> str:
    """Review of dfb6db2 (item 5a): the fallback worksheets (``where``) whose practice plan only describes the data
    the learner works on ("Danh sách 5 yêu cầu …" without the five requests)."""

    return (f"Phiếu thực hành dự phòng ở {where} chỉ mô tả dữ liệu đầu vào, chưa có dữ liệu để học viên làm bài — tác "
            "giả cần bổ sung dữ liệu." if vi
            else f"The fallback worksheet in {where} only describes its input data, without the data learners work "
                 "on — the author must add it.")


def supposition_line(count: int, vi: bool) -> str:
    """The note of the drafted cases told as a real past event that now open with "Giả sử" (run 5496155c)."""

    return (f"Đã thêm «Giả sử» trước {count} tình huống AI soạn được kể như chuyện đã xảy ra (như «Trong quý vừa "
            "qua…») — tác giả nên đọc lại." if vi
            else f"Added «Suppose that» before {count} AI-drafted case(s) told as a past event (such as «Last "
                 "quarter…») — the author should read them again.")


def worksheet_line(vi: bool, *, example_kept: bool) -> str:
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


# A5 (owner decision 2026-10-22): the worksheet of the unit has criteria the AI suggested; keyed by "Vietnamese". Run
# 5496155c: the learner reads "Tự kiểm tra" only, so this line is where the author learns it.
SUGGESTED_CRITERIA_LINE: Final = {
    True: "Tài liệu không nêu tiêu chí chấm cho phiếu thực hành của bài này: phần Tự kiểm tra dùng tiêu chí AI gợi ý "
          "từ nội dung đã dạy — tác giả nên xem lại.",
    False: "The source states no grading criteria for this unit's worksheet: its Self-check uses criteria the AI "
           "suggested from what the lesson teaches — the author should check them.",
}


def drafted_examples_line(examples: Mapping[int, int], slot_types: Sequence[str], vi: bool) -> str:
    """One plain line per unit: how many AI-drafted examples, and in which slots."""

    labels = SLOT_LABEL_VI if vi else SLOT_LABEL_EN
    total = sum(examples.values())
    where = ", ".join(f"{'khối' if vi else 'block'} {index + 1}"
                      + (f" ({labels[slot_types[index]]})" if index < len(slot_types) and slot_types[index] in labels
                         else "") for index in sorted(examples))
    return (f"Ví dụ minh hoạ do AI soạn — cần kiểm tra: {total} ví dụ ({where})." if vi
            else f"Illustrative examples drafted by AI — please check: {total} example(s) ({where}).")


def example_numbers_line(numbers: Mapping[int, int], slot_types: Sequence[str], vi: bool) -> str:
    """One plain line: the AI-drafted examples that state a number the facts do not (a notice, never a review mark)."""

    labels = SLOT_LABEL_VI if vi else SLOT_LABEL_EN
    where = ", ".join(f"{'khối' if vi else 'block'} {index + 1}"
                      + (f" ({labels[slot_types[index]]})" if index < len(slot_types) and slot_types[index] in labels
                         else "") for index in sorted(numbers))
    total = sum(numbers.values())
    return (f"Ví dụ minh hoạ do AI soạn có số liệu không có trong tài liệu, chỉ để minh hoạ — cần kiểm tra lại số "
            f"liệu: {total} ví dụ ({where})." if vi
            else f"Illustrative examples drafted by AI state numbers that are not in the source, for illustration only "
                 f"— check the numbers: {total} example(s) ({where}).")


def dropped_faq_lines(vi: bool, *, dropped: int, restated: int, copied: int, recall: int, unresolved: int,
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
        (unresolved, f"Đã bỏ {unresolved} câu hỏi đáp trả lời điều tài liệu chưa chốt." if vi
         else f"{unresolved} FAQ item(s) that answered a point the source has not settled were removed."),
        (invalid, f"Đã bỏ {invalid} câu hỏi đáp hệ thống không lưu được (ký tự góc nhọn, câu hỏi trùng)." if vi
         else f"{invalid} FAQ item(s) the course editor cannot store (angle brackets, repeated question) were "
              "removed."),
    ]
    return [line for count, line in lines if count]


def faq_inferred_line(inferred: Mapping[int, int], slot_types: Sequence[str], vi: bool) -> str:
    """One plain line: FAQ answers that explain beyond the words of their facts, for the SME to check."""

    labels = SLOT_LABEL_VI if vi else SLOT_LABEL_EN
    total = sum(inferred.values())
    where = ", ".join(f"{'khối' if vi else 'block'} {index + 1}"
                      + (f" ({labels[slot_types[index]]})" if index < len(slot_types) and slot_types[index] in labels
                         else "") for index in sorted(inferred))
    return (f"Câu trả lời hỏi đáp có phần AI suy luận từ tài liệu — tác giả nên kiểm tra không có quy định mới: "
            f"{total} câu ({where})." if vi
            else f"FAQ answers with AI reasoning drawn from the source — the author should check they add no new "
                 "rule: "
                 f"{total} answer(s) ({where}).")


def classroom_line(details: Mapping[int, Sequence[str]], slot_types: Sequence[str], vi: bool) -> str:
    """Run 2f8781d6 (B8): the slots that still carry the time limit, the count or the group or homework wording of a
    classroom exercise of the source ("10 phút", "đúng 2"), an edit note."""

    labels = SLOT_LABEL_VI if vi else SLOT_LABEL_EN
    where = ", ".join(f"{'khối' if vi else 'block'} {index + 1}" + (f" ({labels[slot_types[index]]})"
                      if index < len(slot_types) and slot_types[index] in labels else "") for index in sorted(details))
    words = ", ".join(f"«{word}»" for word in dict.fromkeys(word for index in sorted(details)
                                                           for word in details[index]))
    return (f"Nên biên tập {where}: còn chi tiết của bài tập trên lớp trong tài liệu ({words}) — nên viết thành bài "
            "tự luyện trên công ty của người học, bỏ thời lượng, số lượng phải chọn và cách làm theo nhóm hoặc ở nhà."
            if vi else
            f"Polish {where}: details of a classroom exercise of the source remain ({words}) — make it a self-paced "
            "practice on the learner's own company, without its time limit, the number to pick and the group or "
            "homework wording.")


__all__ = [
    "SCENARIO_LINE",
    "SLOT_LABEL_EN",
    "SLOT_LABEL_VI",
    "SUGGESTED_CRITERIA_LINE",
    "callout_lines",
    "classroom_line",
    "drafted_examples_line",
    "dropped_faq_lines",
    "example_numbers_line",
    "fallback_input_line",
    "faq_inferred_line",
    "framework_line",
    "supposition_line",
    "worksheet_line",
]
