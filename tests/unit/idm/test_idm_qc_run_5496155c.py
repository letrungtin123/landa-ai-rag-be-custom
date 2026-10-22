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
    WRITER,
    FakeGenerate,
    StoryboardEndpointTestCase,
    body_for,
    escalate_body,
    escalate_writer,
    judge,
    severity_body,
    severity_writer,
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
