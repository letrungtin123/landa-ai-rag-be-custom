"""Fixes from the QC report of the real AI ID run 5496155c (AI db022e7, 2026-10-22, 7,2/10).

Every case below is synthetic text with the shape of a defect of that run (no course text of the run); every provider
is a local fake.
"""

from __future__ import annotations

import json
import re
import unittest
from typing import Any

from app.idm.contracts import IdmIssueV1, brief_hash_of
from app.idm.notes import build_course_notes
from app.idm.prompts_module import module_prompt
from app.idm.prompts_unit import _WORKSHEET_RULE
from app.idm.quiz_text import (
    EXPLANATION_NAMES_ONLY_CODE,
    QUESTION_ITEM_NOUN_CODE,
    QUESTION_TWICE_CODE,
    explanation_report,
    lay_out_quiz,
    number_item_letters,
    question_report,
)
from app.idm.source_flags import (
    REVIEW_FLAG_TYPES,
    SOFT_FLAG_TYPES,
    SourceFact,
    clean_quote,
    flag_sentence,
    flag_sentences,
    is_transition,
)
from app.idm.source_locked import render_idm_source_locked_html
from app.idm.text import glued_words
from app.idm.worksheet import plain_self_check
from app.idm.worksheet_input import InputReference, input_hint, missing_input
from tests.support.idm_test_support import rehash_contract
from tests.unit.idm.test_idm_fallback_matrix import procedure_steps, unit_findings, with_fact_texts
from tests.unit.idm.test_idm_qc_feedback import (
    CONFLICT,
    LINES,
    SUGGESTED,
    flagged_escalate_body,
    suggested_worksheet_body,
)
from tests.unit.idm.test_idm_storyboard import (
    JUDGE,
    REPAIR,
    WRITER,
    FakeGenerate,
    StoryboardEndpointTestCase,
    body_for,
    escalate_body,
    escalate_writer,
    judge,
    severity_body,
    severity_writer,
    slot_repair,
    worksheet_body,
    worksheet_writer,
)

# The shapes of the run's flags: a cut bracket and a lone letter of one slide (p.43), a line the PDF glued and cut
# ("… bộchân dung … của", p.46), and a transition sentence of the presenter (p.20).
BRACKET = SourceFact("(Chữ", 43)
LETTER = SourceFact("I", 43)
GLUED = SourceFact("Lập bộhồ sơ các nhóm nhà cung cấp của", 46)
TRANSITION = SourceFact("Chúng ta lại đi sâu thêm một bước nữa, 5 nhóm vai trò…", 20)
CLEAN = SourceFact("Nhân viên cần phản hồi khách hàng trong vòng 24 giờ.", 4)
FACTS = {"bracket": BRACKET, "letter": LETTER, "glued": GLUED, "transition": TRANSITION, "clean": CLEAN,
         "other": SourceFact("Khiếu nại cấp 2 được phản hồi trong vòng 48 giờ.", 7)}


def issue(kind: str, *keys: str) -> IdmIssueV1:
    return IdmIssueV1.model_validate({"type": kind, "note": "ghi chú của AI", "fact_keys": list(keys)})


class SourceFlagQuoteTests(unittest.TestCase):
    """Item 1: only a conflict or an out-of-date point is "Cần xem"; an unclear one is a soft edit note; a quote is
    never a fragment; a transition sentence is no point; one sentence per point of a block."""

    def test_glued_words_are_found_never_split(self) -> None:
        self.assertEqual(glued_words("Lập bộhồ sơ, khiến họphải từchối và bỏlỡmột cơ hội"),
                         ["bộhồ", "họphải", "từchối", "bỏlỡmột"])
        self.assertEqual(glued_words("Nguyễn khuyến khích người quản lý ưu tiên quốc gia, Customer OPEX B2B"), [])

    def test_a_quote_is_never_a_fragment(self) -> None:
        for fragment in ("(Chữ", "I", GLUED.text, "Mô hình gồm năm bước và", "12 | 3 | 4 | 5 | 6", TRANSITION.text):
            self.assertFalse(clean_quote(fragment), fragment)
        self.assertTrue(clean_quote(CLEAN.text))
        # A quote the excerpt cut ends with an ellipsis, never on a dangling word.
        self.assertTrue(clean_quote("Nhân viên cần phản hồi khách hàng của…"))
        self.assertTrue(is_transition(TRANSITION.text))
        self.assertFalse(is_transition("Chúng ta phải phản hồi khách hàng trong 24 giờ."))

    def test_an_unclear_point_names_the_part_and_page_without_a_clean_quote(self) -> None:
        sentence = flag_sentence("Các thành phần của sơ đồ dịch vụ", issue("unclear", "bracket", "letter"), FACTS, "vi")
        self.assertEqual(sentence, "Tài liệu ghi chưa rõ ở phần “Các thành phần của sơ đồ dịch vụ” (trang 43) — nên "
                                   "đọc lại khi biên tập.")
        self.assertEqual(flag_sentence("Bài tập nhóm", issue("unclear", "glued"), FACTS, "en"),
                         "The source is unclear in “Bài tập nhóm” (page 46) — reread it when editing.")

    def test_a_transition_sentence_is_no_point(self) -> None:
        self.assertIsNone(flag_sentence("Ma trận thấu cảm", issue("unclear", "transition"), FACTS, "vi"))
        self.assertEqual(flag_sentences([("Ma trận thấu cảm", [issue("unclear", "transition")])], FACTS, "vi"), [])

    def test_a_conflict_quotes_only_its_clean_facts(self) -> None:
        sentence = flag_sentence("Thời hạn phản hồi", issue("conflict", "bracket", "clean"), FACTS, "vi")
        self.assertEqual(sentence, "Tài liệu ghi khác nhau ở phần “Thời hạn phản hồi”: “Nhân viên cần phản hồi khách "
                                   "hàng trong vòng 24 giờ.” (trang 4). Hãy kiểm tra trước khi áp dụng.")
        alone = flag_sentence("Thời hạn phản hồi", issue("outdated", "bracket", "letter"), FACTS, "vi")
        self.assertEqual(alone, "Tài liệu có thể đã cũ ở phần “Thời hạn phản hồi” (trang 43). Hãy kiểm tra trước khi "
                                "áp dụng.")

    def test_one_sentence_per_kind_and_part_without_repeats(self) -> None:
        blocks = [("Sơ đồ dịch vụ", [issue("unclear", "bracket"), issue("unclear", "letter"),
                                     issue("unclear", "glued")]),
                  ("Sơ đồ dịch vụ", [issue("unclear", "bracket")]),
                  ("Thời hạn phản hồi", [issue("conflict", "clean", "other"), issue("conflict", "other", "clean")])]
        self.assertEqual(flag_sentences(blocks, FACTS, "vi", SOFT_FLAG_TYPES),
                         ["Tài liệu ghi chưa rõ ở phần “Sơ đồ dịch vụ” (trang 43, 46) — nên đọc lại khi biên tập."])
        review = flag_sentences(blocks, FACTS, "vi", REVIEW_FLAG_TYPES)
        self.assertEqual(len(review), 1)
        self.assertEqual(review[0].count("24 giờ"), 1)
        self.assertEqual(review[0].count("48 giờ"), 1)

    def test_the_course_note_keeps_the_unclear_points_apart(self) -> None:
        note = build_course_notes(
            locale="vi", ai_proposed=False, total_minutes=30, lesson_count=2, module_count=1, reference_block_count=0,
            excluded_block_count=0, fallback_stages=[],
            source_checks=flag_sentences([("Thời hạn phản hồi", [issue("conflict", "clean", "other")])], FACTS, "vi"),
            source_unclear=flag_sentences([("Sơ đồ dịch vụ", [issue("unclear", "bracket")])], FACTS, "vi",
                                          SOFT_FLAG_TYPES))
        checks = note.split("\n• Chỗ tài liệu cần kiểm tra trước khi áp dụng:\n")[1].split("\n• ")[0]
        unclear = note.split("\n• Chỗ tài liệu ghi chưa rõ, nên đọc lại khi biên tập:\n")[1].split("\n• ")[0]
        self.assertIn("Tài liệu ghi khác nhau ở phần “Thời hạn phản hồi”", checks)
        self.assertNotIn("chưa rõ", checks)
        self.assertEqual(unclear, "  - Tài liệu ghi chưa rõ ở phần “Sơ đồ dịch vụ” (trang 43) — nên đọc lại khi biên "
                                  "tập.")


class SourceFlagUnitTests(StoryboardEndpointTestCase):
    """Item 1 at the unit: an unclear point leaves the unit validated with a soft note; a conflict still marks it."""

    async def test_an_unclear_point_is_an_edit_note_only(self) -> None:
        unclear = {"type": "unclear", "note": "Không rõ.", "fact_keys": CONFLICT["fact_keys"]}
        body = flagged_escalate_body([unclear])
        status, data, _ = await self.post(body, FakeGenerate(**{WRITER: [escalate_writer(body)], JUDGE: [judge()]}))
        self.assertEqual(status, 200)
        self.assertEqual(data["quality_state"], "validated")
        note = data["unit"]["idm_quality"]["author_note"]
        self.assertIn("Tài liệu ghi chưa rõ ở phần “Những lỗi nào nhân viên thường mắc khi xử lý khiếu nại?”: “", note)
        self.assertIn("— nên đọc lại khi biên tập.", note)
        self.assertNotIn("Cần tác giả xem lại", note)
        self.assertNotIn("Hãy kiểm tra trước khi áp dụng", note)

    async def test_a_conflict_still_marks_the_unit(self) -> None:
        body = flagged_escalate_body([CONFLICT])
        status, data, _ = await self.post(body, FakeGenerate(**{WRITER: [escalate_writer(body)], JUDGE: [judge()]}))
        self.assertEqual(status, 200)
        self.assertEqual(data["quality_state"], "review_required")
        self.assertIn("Hãy kiểm tra trước khi áp dụng.", data["unit"]["idm_quality"]["author_note"])


def self_check(heading: str, *items: str) -> dict[str, Any]:
    return {"sections": [{"heading": "Nhiệm vụ", "blocks": [{"kind": "task", "text": "Làm phiếu."}]},
                         {"heading": heading, "blocks": [{"kind": "bullets", "items": list(items)}]}]}


class SelfCheckHeadingTests(StoryboardEndpointTestCase):
    """Item 2: learners read "Tự kiểm tra" ("Self-check"); that the AI suggested the criteria is said in the author
    note only, and only on a unit with a worksheet whose criteria the AI suggested."""

    def test_the_label_goes_from_the_heading_and_the_items(self) -> None:
        for heading, plain in (("Tự kiểm tra (tiêu chí gợi ý)", "Tự kiểm tra"),
                               ("Tự kiểm tra bảng của bạn (Tiêu chí gợi ý)", "Tự kiểm tra bảng của bạn"),
                               ("Tự kiểm tra – tiêu chí gợi ý", "Tự kiểm tra"),  # noqa: RUF001 - an en dash
                               ("Tiêu chí gợi ý", "Tự kiểm tra")):
            result = plain_self_check(self_check(heading, "Tiêu chí gợi ý: Chọn đúng 2 sáng kiến khả thi.", "Rõ ràng"),
                                      "vi")
            assert result is not None
            self.assertEqual(result["sections"][1]["heading"], plain, heading)
            self.assertEqual(result["sections"][1]["blocks"][0]["items"], ["Chọn đúng 2 sáng kiến khả thi.", "Rõ ràng"])
            self.assertEqual(result["sections"][0], self_check(heading)["sections"][0])
        english = plain_self_check(self_check("Self-check (suggested criteria)", "a"), "en")
        self.assertEqual(english["sections"][1]["heading"] if english else None, "Self-check")
        # Other words that mention criteria stay as written.
        kept = self_check("Tiêu chí tự kiểm tra bảng của bạn", "Tiêu chí chấm: đủ ba cột.")
        self.assertIsNone(plain_self_check(kept, "vi"))

    async def test_a_labelled_self_check_reaches_the_learner_plain(self) -> None:
        body = suggested_worksheet_body()
        writer = worksheet_writer(body)
        sections = writer["components"]["c0"]["semantic_content"]["sections"]
        sections[-1]["heading"] = "Tự kiểm tra (tiêu chí gợi ý)"
        items = sections[-1]["blocks"][0]["items"]
        sections[-1]["blocks"][0]["items"] = [f"Tiêu chí gợi ý: {item}" for item in items]
        status, data, _ = await self.post(body, FakeGenerate(**{WRITER: [writer], JUDGE: [judge()]}))
        self.assertEqual(status, 200)
        served = data["unit"]["components"][0]["semantic_content"]["sections"][-1]
        self.assertEqual(served["heading"], "Tự kiểm tra")
        self.assertEqual(served["blocks"][0]["items"], items)
        self.assertNotIn("gợi ý", json.dumps(data["unit"]["components"], ensure_ascii=False))
        self.assertIn("phần Tự kiểm tra dùng tiêu chí AI gợi ý", data["unit"]["idm_quality"]["author_note"])

    async def test_a_unit_without_a_worksheet_gets_no_criteria_line(self) -> None:
        # The shape of c1.l1.u3: a question whose practice has suggested criteria, no worksheet in the unit.
        body = severity_body()
        brief = body["unit_contract"]["idm_unit_brief"]
        brief["components"][1]["practice"]["feedback_focus"]["criterion"] = SUGGESTED
        brief["brief_hash"] = brief_hash_of(brief)
        rehash_contract(body["unit_contract"])
        writer = severity_writer(body, explanation=LINES)
        status, data, _ = await self.post(body, FakeGenerate(**{WRITER: [writer], JUDGE: [judge()]}))
        self.assertEqual(status, 200)
        self.assertNotIn("tiêu chí", data["unit"]["idm_quality"]["author_note"].lower())


# The shapes of the run's fallback units (synthetic text). A slide table the PDF reader broke over lines: the header row
# wrapped inside a cell, row labels cut before their bracket, a label word on its own line (c1.l3.u2).
BROKEN_TABLE = {
    "d1-c5-f1": "1.3 TỪ BẢN ĐỒ CẤP ĐỘ TỚI CÁCH XỬ LÝ KHIẾU NẠI",
    "d1-c5-f2": "Mỗi khiếu nại có một mức độ riêng. Mức độ quyết định ai xử lý và trong bao lâu.",
    "d1-c5-f3": "Row 1: Cấp độ | DẤU HIỆU(Khách hàng nói gì?) | CÁCH",
    "d1-c5-f4": "XỬ LÝ(Ai làm gì?) | THỜI HẠN(Bao lâu?)",
    "d1-c5-f5": "Row 2: 1. THẤP",
    "d1-c5-f6": "(Ảnh hưởng nhỏ) | Không thiệt hại tài chính. | Nhân viên tự xử lý. | Trong ngày.",
    "d1-c5-f7": "Row 3: 2. TRUNG BÌNH",
    "d1-c5-f8": "(Ảnh hưởng vừa) | Thiệt hại dưới 50 triệu đồng. | Báo trưởng nhóm. | Trong 24 giờ.",
    "d1-c5-f9": "Row 4: 3. CAO / NGHIÊM",
    "d1-c5-f10": "TRỌNG",
    "d1-c5-f11": "(Ảnh hưởng lớn) | Liên quan an toàn hoặc pháp lý. | Escalate cho quản lý. | Ngay lập tức.",
}
# A profile slide (c1.l3.u3, c1.l3.u4): a header row alone, a label cut before its bracket closes and its "* " bullet,
# the next column's label after a "|" with a list marker, and a word the PDF glued together.
BROKEN_PROFILE = {
    "d1-c2-f1": "1.3 HỒ SƠ KHÁCH HÀNG (VD1: TRƯỞNG PHÒNG MUA HÀNG)",
    "d1-c2-f2": "Row 1: VIỆC CẦN LÀM - Mục tiêu | NỖI ĐAU - Điều cản trở | LỢI ÍCH - Điều mong đợi",
    "d1-c2-f3": "Row 2: Về Chức năng (Functional Job)",
    "d1-c2-f4": "Chọn nhà cung cấp đúng hạn, đúng ngân sách và đúng quy trình mua sắm của công ty.",
    "d1-c2-f5": "Về Ý nghĩa / Xã hội (Social/Meaningful",
    "d1-c3-f1": "Job): * Được cấp trên tin tưởng giao các gói mua sắm lớn hơn trong năm tới.",
    "d1-c3-f2": "Kiểm soát rủi ro khi đổi nhà cung cấp mới cho dự án trọng điểm. | - Về Chức năng (Functional Pain)",
    "d1-c3-f3": "Hồ sơ chào hàng dài dòng, thiếu số liệu so sánh nên phảiđọc lại nhiều lần.",
    "d1-c3-f4": "Bị trễ tiến độ vì báo giá thiếu thông tin quan trọng. | -Về Chức năng (Functional Gain)",
}
# Classroom exercise slides only (c2.l2.u2, c3.l2.u1): no sentence of the unit states anything.
EXERCISES = {
    "d1-c8-f1": "BÀI TẬP 10 phút: CHỌN CÁCH XỬ LÝ",
    "d1-c8-f2": "Câu hỏi thảo luận: Nhóm bạn đã gặp khiếu nại nào khó xử lý nhất?",
    "d1-c8-f3": "Thực hành: Các nhóm được phát 4 khiếu nại mẫu và chọn ra đúng 2 khiếu nại cần escalate ngay",
}


def facts_of(body: dict[str, Any], texts: dict[str, str]) -> dict[str, Any]:
    keys = [fact["fact_key"] for fact in body["unit_contract"]["source_facts"]]
    return with_fact_texts(body, {key: text for key, text in zip(keys, texts.values(), strict=False)})


class FallbackOnlyRunShapeTests(StoryboardEndpointTestCase):
    """The lead's verification of run 5496155c: 9 of 10 real unit requests answered fallback_only with 422 (already at
    db022e7), and a fallback slot reached learners with broken table text (c1.l3.u3). Every shape below gives a valid,
    clean, provider-free unit marked for review, with a plain reason."""

    async def fallback(self, body: dict[str, Any], locale: str = "vi") -> dict[str, Any]:
        request = {**body, "locale": locale, "fallback_only": True}
        status, data, provider = await self.post(request)
        self.assertEqual((status, provider.calls), (200, []), data.get("detail"))
        self.assert_envelope(data, "structured_fallback", "review_required", "deterministic_fallback")
        self.assertEqual(unit_findings(request, data["unit"]), (None, []))
        for component in data["unit"]["components"]:
            text = re.sub(r"<[^>]+>", "\n", str(component.get("html") or ""))
            self.assertNotRegex(text, r"(?m)^Row \d|\s\|\s|(?<=\s)\*(?=\s)|\): \*")
        return data

    async def test_a_broken_slide_table_is_one_table(self) -> None:
        data = await self.fallback(facts_of(procedure_steps(), BROKEN_TABLE))
        html = data["unit"]["components"][0]["html"]
        self.assertIn("<h3>1.3 Từ bản đồ cấp độ tới cách xử lý khiếu nại</h3>", html)
        self.assertEqual(html.count("<table>"), 1)
        self.assertEqual(html.count("<tr>"), 4)
        self.assertIn("<th>DẤU HIỆU(Khách hàng nói gì?)</th><th>CÁCH XỬ LÝ(Ai làm gì?)</th>", html)
        self.assertIn("<td>3. CAO / NGHIÊM TRỌNG (Ảnh hưởng lớn)</td>", html)
        self.assertNotIn("<h3>TRỌNG", html)

    async def test_a_profile_slide_reads_as_headings_and_text(self) -> None:
        for locale in ("vi", "en"):
            data = await self.fallback(facts_of(body_for(0, 0, 0), BROKEN_PROFILE), locale)
            html = data["unit"]["components"][0]["html"]
            self.assertIn("<h3>1.3 Hồ sơ khách hàng (VD1: trưởng phòng mua hàng)</h3>", html)
            self.assertIn("<ul><li>VIỆC CẦN LÀM - Mục tiêu</li><li>NỖI ĐAU - Điều cản trở</li>", html)
            self.assertIn("<strong>Về Ý nghĩa / Xã hội (Social/Meaningful Job):</strong> Được cấp trên tin tưởng", html)
            self.assertIn("<h3>Về Chức năng (Functional Pain)</h3>", html)
            self.assertNotIn("(Social/Meaningful</", html)
            # A glued word stays as written and the note names it.
            self.assertIn("phảiđọc", html)
            note = data["unit"]["idm_quality"]["author_note"]
            self.assertIn("«phảiđọc»", note)
            # The question slot holds no structure: it asks about a sentence of the lesson, the note says to rewrite it.
            question = data["unit"]["components"][1]
            self.assertEqual(question["question"].splitlines()[1],
                             "- Chọn nhà cung cấp đúng hạn, đúng ngân sách và đúng quy trình mua sắm của công ty.")
            self.assertEqual(sum(choice["correct"] for choice in question["choices"]), 1)
            self.assertIn("khối 2 (Quiz)" if locale == "vi" else "block 2 (Quiz)", note)

    async def test_exercise_slides_fall_back_on_the_lessons_sentences(self) -> None:
        body = facts_of(escalate_body(), EXERCISES)
        data = await self.fallback(body)
        context = [fact["fact_text"] for fact in body["unit_contract"]["idm_unit_brief"]["lesson_context_facts"]]
        question, faq = data["unit"]["components"]
        self.assertIn(question["question"].splitlines()[1][2:], context)
        self.assertTrue(all(item["answer"] in context for item in faq["items"]))
        self.assertIn("Tài liệu không đủ dữ kiện để dựng khối 1 (Quiz); khối 2 (Hỏi đáp)",
                      data["unit"]["idm_quality"]["author_note"])


class SourceLockedSlideTextTests(unittest.TestCase):
    """Item 10 at the renderer: what the PDF reader leaves of a profile slide reads as clean text (synthetic lines with
    the shapes of c1.l3.u3)."""

    def test_cut_labels_and_cells_are_mended_never_shown_raw(self) -> None:
        html = render_idm_source_locked_html("Hồ sơ mẫu", [
            "Row 1: VIỆC CẦN LÀM - Mục tiêu | NỖI ĐAU - Điều cản trở | LỢI ÍCH - Điều mong đợi",
            "Row 2: Về Chức năng (Functional Job): * Chọn nhà cung cấp đúng hạn và đúng ngân sách của công ty.",
            "Về Ý nghĩa / Xã hội (Social/Meaningful",
            "Job): * Được cấp trên tin tưởng giao gói mua sắm lớn hơn. | - Về Chức năng (Functional Pain): * Bị",
            "\"bủa vây\" bởi hồ sơ chào hàng dài dòng, thiếu số liệu so sánh.",
            "Pain): * Sợ bị coi là một \"nút thắt\"",
            "(bottleneck) làm chậm tiến độ của cả phòng.",
        ], locale="vi")
        text = re.sub(r"<[^>]+>", "\n", html)
        self.assertNotRegex(text, r"(?m)\s\|\s|(?<=\s)\*(?=\s)|\): \*|^(?:Job|Pain)\)")
        self.assertIn("<ul><li>VIỆC CẦN LÀM - Mục tiêu</li>", html)
        self.assertIn("Về Ý nghĩa / Xã hội (Social/Meaningful Job):</strong> Được cấp trên tin tưởng", html)
        self.assertIn("Về Chức năng (Functional Pain):</strong> Bị &quot;bủa vây&quot; bởi hồ sơ", html)
        self.assertIn("<p>Sợ bị coi là một &quot;nút thắt&quot; (bottleneck) làm chậm tiến độ của cả phòng.</p>", html)
        self.assertNotIn("<h3>", html)


def worksheet(task: str, *extra: dict[str, Any], template_rows: int = 4) -> dict[str, Any]:
    """A worksheet html slot: the task, ``extra`` blocks after it, a template table, a worked example, a self-check."""

    def block(kind: str, **fields: Any) -> dict[str, Any]:
        return {"kind": kind, "text": fields.get("text"), "items": fields.get("items", []),
                "rows": fields.get("rows", [])}

    rows = [{"label": f"Ô {number}", "value": "Ghi gì vào ô này?"} for number in range(1, template_rows + 1)]
    return {"type": "html", "semantic_content": {"sections": [
        {"heading": "Nhiệm vụ", "blocks": [block("task", text=task), *extra]},
        {"heading": "Biểu mẫu cần điền", "blocks": [block("table", rows=rows)]},
        {"heading": "Ví dụ mẫu đối chiếu", "blocks": [block("table", rows=rows[:2])]},
        {"heading": "Tự kiểm tra", "blocks": [block("bullets", items=["Đủ các ô.", "Đúng vai trò."])]},
    ]}}


def bullets(count: int) -> dict[str, Any]:
    return {"kind": "bullets", "text": None, "rows": [],
            "items": [f"Đề xuất {number}: một yêu cầu mẫu của khách hàng." for number in range(1, count + 1)]}


INTERVIEW = ("Tổng giám đốc chia sẻ: tôi cần các khoản đầu tư có tỷ suất hoàn vốn rõ ràng, tôi lo đội ngũ không theo "
             "kịp thay đổi và tôi muốn công ty được đối tác nhìn nhận là đơn vị dẫn đầu trong ngành.")


class WorksheetInputTests(StoryboardEndpointTestCase):
    """Item 3: a worksheet gives the data its task refers to (a list, an interview, requests); one that does not is
    repaired once, then kept for the author's review with a plain note."""

    def test_a_task_without_its_data_is_found(self) -> None:
        # The run's three shapes: a list of people, an interview, five requests (only a 3-step list and the template).
        for task, extra in (("Bạn hãy đọc danh sách nhân sự tham gia dự án, sau đó phân loại từng người.", ()),
                            ("Hãy bóc tách nội dung phỏng vấn thành ma trận ba tầng.", ()),
                            ("Từ danh sách 5 yêu cầu của khách hàng, hãy chọn ra 2 sáng kiến khả thi nhất.",
                             (bullets(3),))):
            reference = missing_input(worksheet(task, *extra))
            self.assertIsNotNone(reference, task)
        self.assertEqual(missing_input(worksheet("Từ danh sách 5 yêu cầu, hãy chọn 2.", bullets(3))),
                         InputReference("items", 5))
        self.assertIn("(5 entries)", input_hint(InputReference("items", 5)))

    def test_a_task_with_its_data_or_without_a_reference_is_left_alone(self) -> None:
        self.assertIsNone(missing_input(worksheet("Từ danh sách 5 yêu cầu, hãy chọn 2 sáng kiến.", bullets(5))))
        self.assertIsNone(missing_input(worksheet("Hãy bóc tách nội dung phỏng vấn thành ma trận ba tầng.",
                                                  {"kind": "paragraph", "text": INTERVIEW, "items": [], "rows": []})))
        # Data written as the first table, before the template.
        table = {"kind": "table", "text": None, "items": [],
                 "rows": [{"label": f"Người {n}", "value": "Trợ lý giám đốc."} for n in range(1, 4)]}
        self.assertIsNone(missing_input(worksheet("Bạn hãy đọc danh sách nhân sự dưới đây và phân loại.", table)))
        # An output to produce is no input: "lập danh sách", "chọn ra 2".
        for task in ("Lập danh sách 5 khách hàng mục tiêu của bạn.", "Chọn ra đúng 2 sáng kiến khả thi nhất."):
            self.assertIsNone(missing_input(worksheet(task)))

    async def test_the_writer_adds_the_data_once_else_the_unit_is_kept_for_review(self) -> None:
        body = worksheet_body()
        writer = worksheet_writer(body)
        writer["components"]["c0"]["semantic_content"]["sections"][0]["blocks"].pop()  # the three complaints go
        fixed = worksheet_writer(body)
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [slot_repair(fixed, 0)], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        self.assertEqual(provider.names, [WRITER, REPAIR, JUDGE])
        self.assertIn("c0 IDM_W5_WORKSHEET_INPUT_MISSING at c0: the worksheet's task refers to data the learner works "
                      "on", provider.calls[1]["prompt"])
        self.assertIn("(the task refers to the items (3 entries) the worksheet does not give)",
                      provider.calls[1]["prompt"])
        self.assertEqual(data["quality_state"], "validated")
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [slot_repair(writer, 0)], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual((status, data["quality_state"]), (200, "review_required"))
        note = data["unit"]["idm_quality"]["author_note"]
        self.assertIn("phiếu thực hành nhắc tới dữ liệu để làm bài (danh sách, nội dung phỏng vấn, các yêu cầu) nhưng "
                      "không đưa dữ liệu đó", note)
        self.assertNotIn("source_locked_fallback", data["unit"]["components"][0])

    def test_the_prompts_ask_for_the_data(self) -> None:
        self.assertIn('(heading "Dữ liệu đầu vào" / "Input data")', _WORKSHEET_RULE)
        self.assertIn("writes it out in full", module_prompt(
            "vi", module_plan={}, lesson_plans=[], audience="", objectives=[], must_dos=[], course_blocks=[],
            facts=[], allowed_components=["html"]))


def stem_quiz(question: str, choices: list[tuple[str, bool]], explanation: str) -> dict[str, Any]:
    return {"type": "problem", "problem_type": "multiple_choice", "question": question,
            "choices": [{"text": text, "correct": correct} for text, correct in choices], "explanation": explanation}


# The shape of c2.l2.u2: items "Phương án 1-4", options that only name them, explanation lines without their content.
NOUN_ITEMS = stem_quiz(
    "Bạn nhận được bốn đề xuất lời cam kết sau:\n- Phương án 1: Dành cho trưởng phòng mua hàng, giảm thời gian duyệt "
    "báo giá nhờ cổng duyệt tự động.\n- Phương án 2: Giải pháp quản lý kho toàn diện giúp tiết kiệm chi phí.\n- "
    "Phương án 3: Hỗ trợ các chuỗi bán lẻ tăng doanh thu mà không tốn thời gian.\n- Phương án 4: Ứng dụng độc quyền "
    "giúp nhân viên kinh doanh học sản phẩm mới trong 5 phút mỗi ngày.\nĐề xuất nào đủ ba thành phần?",
    [("Phương án 2", False), ("Phương án 1", True), ("Phương án 3", False)],
    "Nguyên tắc: lời cam kết đủ ba thành phần.\nA - Sai vì phương án thiếu đối tượng mục tiêu cụ thể.\nB - Đúng vì "
    "phương án nêu đủ ba phần.\nC - Sai vì phương án thiếu cách làm khác biệt.")
# The shapes of c2.l4.u2 (a question before the list and on the last line) and c3.l1.u1 (one in the context line).
ASKED_BEFORE_LIST = ("Khách hàng chờ báo giá 5 ngày.\nCách nào xử lý tận gốc điểm đau này?\n- Cách 1: Gửi thư xin "
                     "lỗi.\n- Cách 2: Duyệt báo giá tự động trong ngày.\nCách nào đúng nhất?")
ASKED_IN_CONTEXT = ("Giả sử khách hàng phàn nàn vì đơn kích hoạt trễ. Nhận định nào đúng?\n- Nhận định 1: Do thái độ "
                    "nhân viên.\n- Nhận định 2: Do hệ thống hậu trường.\nBạn đồng ý với nhận định nào?")


class StemNamingTests(StoryboardEndpointTestCase):
    """Items 4 and 5: stem items are never named like the answer options, an option that only names an item gets an
    explanation that says what the item is, and a stem asks one question, on its last line."""

    def test_items_named_like_the_options_are_found(self) -> None:
        laid = lay_out_quiz(NOUN_ITEMS)[0]
        self.assertIn(QUESTION_ITEM_NOUN_CODE, question_report(laid))
        self.assertEqual(explanation_report(laid)[EXPLANATION_NAMES_ONLY_CODE],
                         ["the lines of A, B, C never say what their item is"])
        renamed = json.loads(json.dumps(NOUN_ITEMS, ensure_ascii=False).replace("Phương án", "Cách"))
        renamed["explanation"] = ("Nguyên tắc: lời cam kết đủ ba thành phần.\nA - Sai vì quản lý kho chưa nêu đối "
                                  "tượng mục tiêu.\nB - Đúng vì trưởng phòng mua hàng, duyệt báo giá và cổng duyệt tự "
                                  "động đủ ba phần.\nC - Sai vì tăng doanh thu chưa nêu cách làm khác biệt.")
        laid = lay_out_quiz(renamed)[0]
        self.assertEqual({**question_report(laid), **explanation_report(laid)}, {})

    def test_a_lettered_item_gets_a_neutral_noun_only_when_certain(self) -> None:
        lettered = stem_quiz("Bối cảnh:\n- Lựa chọn A: giảm giá\n- Lựa chọn B: tặng quà\nLựa chọn nào giữ khách?",
                             [("Lựa chọn A", False), ("Lựa chọn B", True)], "Nguyên tắc: x.\nA - Sai.\nB - Đúng.")
        self.assertEqual(number_item_letters(lettered)[0]["question"],
                         "Bối cảnh:\n- Cách 1: giảm giá\n- Cách 2: tặng quà\nLựa chọn nào giữ khách?")
        english = stem_quiz("Plans:\n- Option A: cut prices\n- Option B: add a gift\nWhich option keeps margin?",
                            [("Option A", False), ("Option B", True)], "Principle: x.\nA - No.\nB - Yes.")
        self.assertEqual([choice["text"] for choice in number_item_letters(english)[0]["choices"]],
                         ["Approach 1", "Approach 2"])
        # Another generic label keeps its noun; a name with a letter and the article "a" are never touched.
        story = stem_quiz("Bối cảnh:\n- Kịch bản A: chờ\n- Kịch bản B: gọi lại\nChọn?",
                          [("Kịch bản A", True), ("Kịch bản B", False)], "")
        self.assertEqual(number_item_letters(story)[0]["question"], "Bối cảnh:\n- Kịch bản 1: chờ\n- Kịch bản 2: gọi "
                                                                    "lại\nChọn?")
        for name in ("Vitamin A", "Hạng B"):
            kept = stem_quiz(f"Bối cảnh:\n- {name}: một\n- {name[:-1]}C: hai\nChọn?", [(name, True)], "")
            self.assertEqual(number_item_letters(kept), (kept, 0))

    def test_a_question_asked_twice_is_found(self) -> None:
        for question in (ASKED_BEFORE_LIST, ASKED_IN_CONTEXT):
            self.assertIn(QUESTION_TWICE_CODE, question_report({"question": question}), question)
        once = ("Khách hỏi: “Bao giờ có báo giá?”. Bạn cần trả lời theo quy trình.\n- Cách 1: Có cần duyệt lại?\n"
                "- Cách 2: Gửi ngay.\nCách nào đúng?")
        self.assertNotIn(QUESTION_TWICE_CODE, question_report({"question": once}))

    async def test_one_repair_then_an_edit_note(self) -> None:
        body = severity_body()
        writer = severity_writer(body, explanation=LINES)
        writer["components"]["c1"]["question"] = (
            "Khách hàng gọi lần thứ hai vì đơn giao trễ, chưa có thiệt hại tài chính. Bạn xếp cấp nào?\n- Phương án 1: "
            "Xếp cấp 1 vì chưa có thiệt hại.\n- Phương án 2: Xếp cấp 2 vì khách phàn nàn lần thứ hai.\n- Phương án 3: "
            "Xếp cấp 3 và escalate ngay.\nKhiếu nại này thuộc cấp nào?")
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [slot_repair(writer, 1)], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        self.assertEqual(provider.names, [WRITER, REPAIR, JUDGE])
        prompt = provider.calls[1]["prompt"]
        self.assertIn("c1 IDM_W5_QUESTION_ITEM_NOUN at c1.question", prompt)
        self.assertIn("c1 IDM_W5_QUESTION_ASKED_TWICE at c1.question", prompt)
        self.assertEqual(data["quality_state"], "validated")
        note = data["unit"]["idm_quality"]["author_note"]
        self.assertIn("Nên biên tập khối 2 (Quiz): các ý trong đề được gọi là «Phương án 1, 2…»", note)
        self.assertIn("đề hỏi hai lần", note)
