"""The "Cần xem" / "Review" hints of the unit author note, per finding code (what to check and why).

Moved verbatim out of ``app.idm.qa`` (STD-5 file size, run 2f8781d6); ``qa._check_lines`` reads them.
"""

from __future__ import annotations

from typing import Final

from app.idm.mcq import ANSWER_LENGTH_CUE_CODE, QUESTION_INCOMPLETE_CODE
from app.idm.qa_checks import (
    ANSWER_CONTESTED_CODE,
    ANSWER_LEAK_CODE,
    CALLOUT_UNGROUNDED_CODE,
    FAQ_RESTATES_HTML_CODE,
    FAQ_TITLE_MISMATCH_CODE,
    FRAMEWORK_INCOMPLETE_CODE,
    SOURCE_ARTIFACT_CODE,
)
from app.idm.quiz_text import EXPLANATION_LAYOUT_CODE, EXPLANATION_RESTATES_CODE, EXPLANATION_VERDICT_CODE
from app.idm.worksheet_input import EXAMPLE_REUSE_CODE, WORKSHEET_INPUT_CODE

REVIEW_HINT_VI: Final = {
    ANSWER_LEAK_CODE: "một phương án gần như chép lại ví dụ hoặc nội dung ngay trước câu hỏi — người học có thể "
                      "chọn theo trí nhớ thay vì áp dụng tiêu chí",
    ANSWER_LENGTH_CUE_CODE: "đáp án đúng dài hơn hẳn các phương án khác — người học có thể đoán theo độ dài",
    ANSWER_CONTESTED_CODE: "tình huống của câu hỏi cũng khớp tiêu chí mà bài học (bảng, danh sách hoặc phần hỏi đáp) "
                           "nêu cho một phương án khác — cần sửa tình huống để chỉ một phương án đúng",
    CALLOUT_UNGROUNDED_CODE: "khung trích dẫn/lưu ý nêu điều không tìm thấy trong tài liệu — chỉ giữ khi tác giả đã "
                             "kiểm tra, hoặc đổi thành đoạn văn thường",
    FAQ_RESTATES_HTML_CODE: "câu hỏi đáp chỉ nhắc lại nội dung vừa học — nên thay bằng ngộ nhận, trường hợp đặc biệt "
                            "hoặc tình huống nếu… thì",
    FAQ_TITLE_MISMATCH_CODE: "tiêu đề phần hỏi đáp không khớp các câu hỏi bên trong",
    SOURCE_ARTIFACT_CODE: "có dòng hoặc ô bảng bị cắt dở như trong tài liệu PDF (ví dụ \"Best-in\", \"(KPI /\") — "
                          "cần viết lại cho trọn ý",
    FRAMEWORK_INCOMPLETE_CODE: "tiêu đề nêu một khung gồm nhiều thành phần nhưng nội dung chưa liệt kê đủ các thành "
                               "phần đó — bổ sung bản đồ tổng quan hoặc đổi tiêu đề theo nội dung",
    QUESTION_INCOMPLETE_CODE: "đề câu hỏi bị cụt, chưa có câu hỏi ở cuối — cần viết lại cho trọn câu hỏi",
    # Run 5496155c: a worksheet that names data it does not give cannot be done.
    WORKSHEET_INPUT_CODE: "phiếu thực hành nhắc tới dữ liệu để làm bài (danh sách, nội dung phỏng vấn, các yêu cầu) "
                          "nhưng không đưa dữ liệu đó — cần bổ sung để người học làm được bài",
    EXAMPLE_REUSE_CODE: "ví dụ mẫu của phiếu thực hành giải chính các mục dữ liệu đầu vào của bài tập nên đã cho sẵn "
                        "đáp án — cần đổi ví dụ sang trường hợp khác",
    # QC feedback 2026-10-21 (A1): the feedback must tell the learner why each option is right or wrong.
    EXPLANATION_LAYOUT_CODE: "lời giải chưa theo bố cục: một dòng nêu nguyên tắc, rồi mỗi phương án một dòng «A - Đúng "
                             "vì …» hoặc «A - Sai vì …» — cần viết lại để người học thấy vì sao đúng, vì sao sai",
    EXPLANATION_VERDICT_CODE: "lời giải ghi một phương án là đúng hoặc sai ngược với đáp án — cần sửa cho khớp đáp án",
    EXPLANATION_RESTATES_CODE: "lời giải chỉ chép lại nội dung phương án, chưa nêu lý do — cần nêu vì sao đúng hoặc "
                               "sai dựa trên bài học",
}
REVIEW_HINT_EN: Final = {
    ANSWER_LEAK_CODE: "an option nearly copies the example or text shown right before the question — learners can "
                      "match it instead of applying the criterion",
    ANSWER_LENGTH_CUE_CODE: "the correct option is much longer than the others — learners can guess it by length",
    ANSWER_CONTESTED_CODE: "the question's case also meets the criteria the unit (its table, list or FAQ) gives for "
                           "another option — change the case so only one option is correct",
    CALLOUT_UNGROUNDED_CODE: "a quotation or callout states something not found in the source — keep it only once "
                             "the author has checked it, or turn it into a plain paragraph",
    FAQ_RESTATES_HTML_CODE: "FAQ items only repeat what was just taught — replace them with a misconception, an edge "
                            "case or a what-if",
    FAQ_TITLE_MISMATCH_CODE: "the FAQ title does not match its questions",
    SOURCE_ARTIFACT_CODE: "a list item or table cell ends cut as the PDF line did (\"Best-in\", \"(KPI /\") — "
                          "complete it",
    FRAMEWORK_INCOMPLETE_CODE: "the title names a framework of several items but the content does not list all of "
                               "them — add the overview or retitle it to what it teaches",
    QUESTION_INCOMPLETE_CODE: "the question stops before it asks anything — write the question out in full",
    WORKSHEET_INPUT_CODE: "the worksheet refers to data to work on (a list, an interview, requests) but does not give "
                          "it — add it so learners can do the task",
    EXAMPLE_REUSE_CODE: "the worksheet's worked example solves the task's own input entries, so it gives the answer "
                        "away — move the example to another case",
    EXPLANATION_LAYOUT_CODE: "the feedback is not laid out as one line with the principle, then one line per option "
                             "(«A - Correct because …» or «A - Incorrect because …») — rewrite it so learners see why "
                             "each option is right or wrong",
    EXPLANATION_VERDICT_CODE: "the feedback calls an option right or wrong against the answer key — make it match the "
                              "key",
    EXPLANATION_RESTATES_CODE: "the feedback only repeats the option, it gives no reason — say why it is right or "
                               "wrong from what the lesson taught",
}


__all__ = ["REVIEW_HINT_EN", "REVIEW_HINT_VI"]
