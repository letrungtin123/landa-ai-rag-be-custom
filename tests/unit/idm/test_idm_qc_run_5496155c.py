"""Fixes from the QC report of the real AI ID run 5496155c (AI db022e7, 2026-10-22, 7,2/10).

Every case below is synthetic text with the shape of a defect of that run (no course text of the run); every provider
is a local fake.
"""

from __future__ import annotations

import json
import re
import unittest
from collections import Counter
from typing import Any

from app.idm.answer_checks import copies_example_measure, resolved_choices, stem_copies_answer, unit_teaching
from app.idm.contracts import IdmIssueV1, brief_hash_of
from app.idm.faq_checks import faq_writer_rules
from app.idm.html_rules import visible_text
from app.idm.hypothetical import SUPPOSITION_ADDED, as_supposition, suppose_semantic
from app.idm.notes import build_course_notes
from app.idm.prompts_judge import judge_prompt
from app.idm.prompts_module import module_prompt
from app.idm.prompts_unit import _WORKSHEET_RULE, unit_writer_prompt
from app.idm.qa import (
    REVIEW_REASON_EVIDENCE,
    REVIEW_REASON_JUDGE,
    REVIEW_REASON_KEPT_SLOTS,
    REVIEW_REASON_SOURCE,
    JudgeOutcome,
    SlotFinding,
    build_unit_author_note,
)
from app.idm.qa_checks import WORKSHEET_INCOMPLETE_CODE
from app.idm.quiz_text import (
    EXPLANATION_FILLER_CODE,
    EXPLANATION_NAMES_ONLY_CODE,
    QUESTION_FEW_OPTIONS_CODE,
    QUESTION_ITEM_NOUN_CODE,
    QUESTION_TWICE_CODE,
    explanation_report,
    hype_phrases,
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
from app.idm.storyboard import build_idm_expected, parse_brief
from app.idm.text import evidence_index, glued_words
from app.idm.visual_pages import evidence_needs_check, unread_image_pages
from app.idm.worksheet import compact_worksheet, plain_self_check, plan_worksheet_html
from app.idm.worksheet_input import InputReference, input_hint, missing_input, reuse_hint, reused_entries
from app.schemas.orchestration_v2 import RagLessonAuthorUnitV2Request
from app.services.orchestration_v2 import unit as unit_service
from tests.support.idm_test_support import rehash_contract
from tests.unit.idm.test_idm_fallback_matrix import procedure_steps, unit_findings, with_fact_texts
from tests.unit.idm.test_idm_faq import EXPLAINED, WHAT_IF, faq_writer
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
        # Run 2f8781d6 (N1): the cut label's end gets back the one label start the slide cut the same way, so the row
        # stays under its own layer (it used to lose its label and read as part of the row above).
        self.assertIn("<p><strong>Về Ý nghĩa / Xã hội (Social/Meaningful Pain):</strong> Sợ bị coi là một &quot;nút "
                      "thắt&quot; (bottleneck) làm chậm tiến độ của cả phòng.</p>", html)
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
        # R3 of the paid verification: three options that only name the listed items are one too few.
        laid = lay_out_quiz(renamed)[0]
        self.assertEqual({**question_report(laid), **explanation_report(laid)},
                         {QUESTION_FEW_OPTIONS_CODE: ["3 options that name the listed items (write 4)"]})
        renamed["choices"].append({"text": "Cách 4", "correct": False})
        renamed["explanation"] += "\nD - Sai vì học sản phẩm mới trong 5 phút chưa nêu đối tượng mua hàng."
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
            "Khách hàng gọi lần thứ hai vì đơn giao trễ, chưa có thiệt hại tài chính. Nhóm bạn bàn cách nào?\n- "
            "Phương án 1: Gọi lại khách ngay trong ngày.\n- Phương án 2: Gửi thư xin lỗi kèm mã giảm giá.\n- Phương "
            "án 3: Chờ khách gọi thêm một lần nữa.\nKhiếu nại này thuộc cấp nào?")
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


def html_slot(*sections: tuple[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {"type": "html", "semantic_content": {"sections": [
        {"heading": heading, "blocks": [{"text": None, "items": [], "rows": [], **block} for block in blocks]}
        for heading, blocks in sections]}}


def rows(*pairs: tuple[str, str]) -> dict[str, Any]:
    return {"kind": "table", "rows": [{"label": label, "value": value} for label, value in pairs]}


# c1.l2.u2: the worked example right above the question, whose answer is a role.
ROLE_EXAMPLE = html_slot(("Ví dụ phân loại một nhân sự", [rows(
    ("Nhân sự và chức danh", "Chị Hoa - Thư ký Giám đốc"), ("Việc làm", "Sàng lọc thư mời và đặt lịch gặp đối tác"),
    ("Vai trò", "Người gác cổng (Gatekeeper)"))]))
ROLE_QUIZ = stem_quiz(
    "Một dự án ghi nhận ba nhân sự:\n- Nhân sự 1: Thư ký Giám đốc sàng lọc thư mời và đặt lịch gặp đối tác.\n- Nhân "
    "sự 2: Kỹ sư vận hành máy hằng ngày.\n- Nhân sự 3: Chuyên gia độc lập đánh giá thiết bị.\nCách xếp nào đúng?",
    [("Nhân sự 1 là Người gác cổng, Nhân sự 2 là Người sử dụng, Nhân sự 3 là Người ảnh hưởng", True),
     ("Nhân sự 1 là Người quyết định, Nhân sự 2 là Người sử dụng, Nhân sự 3 là Phòng mua hàng", False),
     ("Nhân sự 1 là Người ảnh hưởng, Nhân sự 2 là Kho vận, Nhân sự 3 là Người quyết định", False)], "")
# c1.l3.u2: a listed item copies the cell of the row whose layer the key names.
LAYER_TABLE = html_slot(("Ba tầng góc nhìn", [rows(
    ("Tầng Chức năng", "Lợi ích là xong việc nhanh, tiết kiệm và chính xác."),
    ("Tầng Ý nghĩa", "Lợi ích là được ghi nhận, được tôn vinh trước tập thể và để lại dấu ấn."))]))
LAYER_QUIZ = stem_quiz(
    "Một trưởng phòng chia sẻ:\n- Phản hồi 1: Lo bị chậm tiến độ.\n- Phản hồi 2: Mong được ghi nhận, tôn vinh trước "
    "tập thể và để lại dấu ấn.\nPhản hồi 2 thuộc tầng nào?",
    [("Lợi ích ở tầng Chức năng", False), ("Lợi ích ở tầng Ý nghĩa", True), ("Nỗi đau ở tầng Chức năng", False)], "")
# c2.l4.u2: the key's item repeats the worked example's figure.
FIGURE_EXAMPLE = html_slot(("Ví dụ mẫu tại một cột mốc", [rows(
    ("Điểm đau", "Hồ sơ hợp đồng chờ duyệt nhiều ngày."),
    ("Giải pháp", "Duyệt hồ sơ mẫu trực tuyến và trả kết quả trong 3 giờ làm việc."))]))
FIGURE_QUIZ = stem_quiz(
    "Khách hàng chờ báo giá 5 ngày.\n- Cách 1: Gọi điện xin lỗi khi báo giá trễ.\n- Cách 2: Tạo và duyệt báo giá trên "
    "cổng trực tuyến dưới 3 giờ.\n- Cách 3: Gửi thêm hồ sơ năng lực khi khách chờ.\nCách nào xử lý tận gốc điểm đau?",
    [("Cách 1", False), ("Cách 2", True), ("Cách 3", False)], "")


class LeakThroughTheStemTests(unittest.TestCase):
    """Item 6: the give-away sat in the stem's listed items, or behind a key that only names an item."""

    def test_an_item_copying_the_example_or_a_row_gives_the_answer(self) -> None:
        for html, quiz in ((ROLE_EXAMPLE, ROLE_QUIZ), (LAYER_TABLE, LAYER_QUIZ)):
            teaching = unit_teaching([html], [])
            laid = lay_out_quiz(quiz)[0]
            self.assertTrue(stem_copies_answer(resolved_choices(laid), teaching, laid), quiz["question"])

    def test_a_key_that_only_names_an_item_is_read_as_the_item(self) -> None:
        laid = lay_out_quiz(FIGURE_QUIZ)[0]
        read = resolved_choices(laid)
        self.assertTrue(next(choice["text"] for choice in read["choices"] if choice["correct"]).startswith("Tạo và"))
        self.assertEqual([choice["text"] for choice in laid["choices"]], ["Cách 1", "Cách 2", "Cách 3"])
        teaching = unit_teaching([FIGURE_EXAMPLE], [])
        self.assertTrue(copies_example_measure(read, teaching))
        self.assertFalse(copies_example_measure(laid, teaching))

    def test_a_case_in_its_own_words_is_no_leak(self) -> None:
        own = stem_quiz(
            "Một dự án ghi nhận ba nhân sự:\n- Nhân sự 1: Người nhận cuộc gọi đầu tiên và chọn cuộc hẹn nào chuyển "
            "lên.\n"
            "- Nhân sự 2: Kỹ sư vận hành máy hằng ngày.\n- Nhân sự 3: Chuyên gia độc lập đánh giá thiết bị.\nCách xếp "
            "nào đúng?", [(text, correct) for text, correct in
                          ((choice["text"], choice["correct"]) for choice in ROLE_QUIZ["choices"])], "")
        laid = lay_out_quiz(own)[0]
        self.assertFalse(stem_copies_answer(resolved_choices(laid), unit_teaching([ROLE_EXAMPLE], []), laid))
        other_figure = json.loads(json.dumps(FIGURE_QUIZ, ensure_ascii=False).replace("dưới 3 giờ", "trong ngày"))
        laid = lay_out_quiz(other_figure)[0]
        self.assertFalse(copies_example_measure(resolved_choices(laid), unit_teaching([FIGURE_EXAMPLE], [])))


class LeakAndContestEndpointTests(StoryboardEndpointTestCase):
    """Items 6 and 7 through the unit: a stem item copying the taught table is repaired as a leak; the writer and the
    judge are told to check every option of a formula question."""

    async def test_an_item_copying_the_taught_row_gets_the_leak_repair(self) -> None:
        body = severity_body()
        writer = severity_writer(body, explanation=LINES)
        writer["components"]["c1"]["question"] = (
            "Bối cảnh:\n- Khiếu nại 1: Thiệt hại tài chính dưới 50 triệu đồng hoặc khách hàng phàn nàn lần thứ hai.\n"
            "- Khiếu nại 2: Khách hàng hỏi giờ mở cửa của cửa hàng.\nKhiếu nại 1 thuộc cấp độ nào?")
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [slot_repair(writer, 1)] * 2, JUDGE: [judge()]})
        status, _data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        self.assertIn("c1 IDM_W5_ANSWER_LEAK at c1", provider.calls[1]["prompt"])
        self.assertIn("a listed item of the question copies the worked example, a table row",
                      provider.calls[1]["prompt"])

    def test_the_writer_and_the_judge_check_every_option_of_a_formula(self) -> None:
        self.assertIn("each\n  distractor visibly lacks or misuses ONE named part", PRACTICE_RULES)
        self.assertIn('no "chi phí cực cao", "vượt ngân sách"', PRACTICE_RULES)
        judge_text = judge_prompt("vi", plan_summary={}, facts=[], unit_content=[])
        self.assertIn("Check EVERY\n   option against the definition, formula or criterion the unit teaches",
                      judge_text)
        self.assertIn("it has every part, as the key does", judge_text)


PRACTICE_RULES = unit_writer_prompt(
    "vi", course_title="", audience="", lesson_title="", lesson_objective="", practice_sentences=[],
    previous_title=None, next_title=None, unit_brief={}, facts=[], context_facts=[])


# The run's hype words, in synthetic sentences; the facts use two of them (the run's source did too).
HYPE_FACTS = evidence_index(["Đây là cuộc chiến sinh tử của doanh nghiệp.",
                             "Chiến lược Say No giúp đội ngũ tập trung nguồn lực."])


class HypeWordTests(StoryboardEndpointTestCase):
    """Item 8: hype words go from explanations and FAQ answers even when the source uses them; a quoted name of a
    source concept keeps its words."""

    def test_a_hype_word_counts_even_when_the_source_uses_it(self) -> None:
        self.assertEqual(hype_phrases("A - Sai vì đây là vấn đề sinh tử.", HYPE_FACTS), ["sinh tử"])
        self.assertEqual(hype_phrases("B - Đúng vì xử lý triệt để và vượt trội.", HYPE_FACTS),
                         ["triệt để", "vượt trội"])
        self.assertEqual(hype_phrases("Absolutely the right call.", HYPE_FACTS), ["absolutely"])
        self.assertEqual(hype_phrases("B - Đúng vì phản hồi khách trong 24 giờ.", HYPE_FACTS), [])

    def test_only_a_quoted_source_name_keeps_its_words(self) -> None:
        quoted = "B - Đúng vì áp dụng chiến lược “Say No” với yêu cầu ngoài phạm vi."
        self.assertEqual(hype_phrases(quoted, HYPE_FACTS), [])
        # Not a name the facts write, not quoted, or a whole quoted sentence: the word counts.
        self.assertEqual(hype_phrases(quoted), ["say no"])
        self.assertEqual(hype_phrases("B - Đúng vì bạn dám Say No với khách.", HYPE_FACTS), ["say no"])
        sentence = "C - Sai vì tài liệu ghi “đây là cuộc chiến sinh tử của doanh nghiệp”."
        self.assertEqual(hype_phrases(sentence, HYPE_FACTS), ["sinh tử"])

    async def test_an_faq_answer_with_hype_is_repaired_once_then_an_edit_note(self) -> None:
        body = escalate_body()
        hyped = EXPLAINED + " Đây là sai lầm sinh tử, cần xử lý triệt để."
        writer = faq_writer(body, ("Vì sao không nên hứa hoàn tiền khi khách vừa gọi đến?", hyped),
                            ("Điều gì xảy ra nếu xử lý khiếu nại mà bỏ qua yếu tố truyền thông?", WHAT_IF))
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [slot_repair(writer, 1)], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        self.assertEqual(provider.names, [WRITER, REPAIR, JUDGE])
        self.assertIn("c1 IDM_W5_FAQ_HYPE at c1: the answers use the hype words quoted below, banned even when the "
                      "source uses them", provider.calls[1]["prompt"])
        self.assertIn('(remove: "sinh tử", "triệt để")', provider.calls[1]["prompt"])
        self.assertEqual(provider.calls[1]["thinking_level"], "low")
        # The item stays (a wording finding never drops it); the author polishes it.
        self.assertEqual(data["unit"]["components"][1]["items"][0]["answer"], hyped)
        self.assertEqual(data["quality_state"], "validated")
        self.assertIn("Nên biên tập khối 2 (Hỏi đáp): câu trả lời hỏi đáp dùng chữ thổi phồng",
                      data["unit"]["idm_quality"]["author_note"])

    def test_the_prompts_ban_hype_in_explanations_and_faq_answers(self) -> None:
        self.assertIn('No empty praise, slogan or hype word instead of a reason, even when the source uses it ("sinh '
                      'tử",', PRACTICE_RULES)
        self.assertIn('No hype word, even when the source uses it ("sinh tử", "dũng cảm", "đập tan", "triệt để", '
                      '"khổng lồ", "tuyệt đối", "vượt trội", "Say No")', faq_writer_rules("explain"))


class SuppositionTests(StoryboardEndpointTestCase):
    """Item 8: a drafted case told as a real past event ("Trong quý vừa qua, …") opens with "Giả sử"."""

    def test_only_a_past_time_opening_the_facts_do_not_use_is_supposed(self) -> None:
        changed = {
            "Trong quý vừa qua, bộ phận kinh doanh đã áp dụng quy trình bốn bước.":
                "Giả sử trong quý vừa qua, bộ phận kinh doanh đã áp dụng quy trình bốn bước.",
            "Khách hàng phàn nàn. Năm ngoái, công ty mất ba khách lớn.":
                "Khách hàng phàn nàn. Giả sử năm ngoái, công ty mất ba khách lớn.",
            "Trong năm qua, công ty ký ba hợp đồng.": "Giả sử trong năm qua, công ty ký ba hợp đồng.",
        }
        for text, expected in changed.items():
            self.assertEqual(as_supposition(text, "vi", None), (expected, 1))
        self.assertEqual(as_supposition("Last quarter, the team missed two deadlines.", "en", None),
                         ("Suppose that last quarter, the team missed two deadlines.", 1))
        for text in ("Tháng trước bạn nhận được đơn.", "Quý trước là quý tốt nhất.",
                     "Giả sử trong quý vừa qua, khách chờ báo giá.", "Doanh số trong quý vừa qua tăng."):
            self.assertEqual(as_supposition(text, "vi", None), (text, 0))
        sourced = "Trong quý vừa qua, doanh số tăng 5%."
        self.assertEqual(as_supposition(sourced, "vi", evidence_index([sourced])), (sourced, 0))

    def test_texts_items_and_cells_change_never_headings_or_labels(self) -> None:
        slot = html_slot(("Trong quý vừa qua", [
            {"kind": "paragraph", "text": "Trong quý vừa qua, một chuyên viên đã hoàn thiện mẫu phân tích."},
            {"kind": "bullets", "items": ["Năm ngoái, khách chờ duyệt hợp đồng nhiều ngày.", "Khách cần báo giá."]},
            rows(("Trong quý vừa qua", "Trong quý vừa qua, khách hàng gặp nút thắt khi chờ phê duyệt."))]))
        counts: Counter[str] = Counter()
        semantic = suppose_semantic(slot["semantic_content"], "vi", None, counts)
        self.assertEqual(counts[SUPPOSITION_ADDED], 3)
        section = semantic["sections"][0]
        self.assertEqual(section["heading"], "Trong quý vừa qua")
        self.assertTrue(section["blocks"][0]["text"].startswith("Giả sử trong quý vừa qua, một chuyên viên"))
        self.assertEqual(section["blocks"][1]["items"][1], "Khách cần báo giá.")
        self.assertEqual(section["blocks"][2]["rows"][0]["label"], "Trong quý vừa qua")
        self.assertTrue(section["blocks"][2]["rows"][0]["value"].startswith("Giả sử trong quý vừa qua, khách hàng"))
        self.assertIs(suppose_semantic(semantic, "vi", None, counts), semantic)

    async def test_the_unit_reads_as_a_supposition_and_the_author_is_told(self) -> None:
        body = severity_body()
        writer = severity_writer(body, explanation=LINES)
        writer["components"]["c0"]["semantic_content"]["sections"][0]["blocks"][0]["text"] = (
            "Trong quý vừa qua, một cửa hàng nhận nhiều khiếu nại về giao hàng. Mỗi khiếu nại thuộc một trong ba cấp "
            "độ.")
        writer["components"]["c1"]["question"] = (
            "Trong quý vừa qua, một khách hàng gọi lần thứ hai vì đơn hàng giao trễ, chưa có thiệt hại tài chính.\n"
            "Khiếu nại này thuộc cấp độ nào?")
        provider = FakeGenerate(**{WRITER: [writer], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        html, question = data["unit"]["components"][0], data["unit"]["components"][1]["question"]
        self.assertIn("Giả sử trong quý vừa qua, một cửa hàng", json.dumps(html, ensure_ascii=False))
        self.assertTrue(question.startswith("Giả sử trong quý vừa qua, một khách hàng"))
        self.assertIn("Đã thêm «Giả sử» trước 2 tình huống AI soạn được kể như chuyện đã xảy ra",
                      data["unit"]["idm_quality"]["author_note"])


def note(**changes: Any) -> str:
    base: dict[str, Any] = {"locale": "vi", "judge": JudgeOutcome("pass"), "deterministic_codes": [],
                            "fallback_slots": [], "ai_drafted": False, "slot_types": ["html", "problem", "la_faq"]}
    return build_unit_author_note(**{**base, **changes})


NO_ISSUE = "QA tự động: không phát hiện vấn đề lớn."


class AuthorNoteNoiseTests(unittest.TestCase):
    """Item 9: "no major issue" next to warnings, "(chi tiết bên dưới)" without details and reasons named twice."""

    def test_no_major_issue_only_on_a_note_that_asks_nothing_else(self) -> None:
        self.assertEqual(note(), NO_ISSUE)
        fixed = note(fixed_codes=["IDM_W5_EXPLANATION_LENGTH"], remaining=[])
        self.assertEqual(fixed, NO_ISSUE + " Đã tự sửa: lời giải quá dài.")
        unclear = "Tài liệu ghi chưa rõ ở phần “Ba cấp độ”: trang 4 — nên đọc lại khi biên tập."
        for changes in ({"notices": [REVIEW_REASON_EVIDENCE]}, {"ai_drafted": True}, {"source_notes": [unclear]},
                        {"unexplained_acronyms": ["TNA"]},
                        {"fixed_codes": ["IDM_W5_EXPLANATION_LENGTH"], "remaining": [], "ai_drafted": True}):
            with self.subTest(changes=sorted(changes)):
                self.assertNotIn("không phát hiện vấn đề lớn", note(**changes))

    def test_details_below_only_when_a_line_below_gives_them(self) -> None:
        # A source reason whose points were all cut, a judge reason without a major finding, kept slots of a unit that
        # fell back as a whole: nothing below details them.
        bare = note(review_reasons=[REVIEW_REASON_SOURCE, REVIEW_REASON_JUDGE])
        self.assertTrue(bare.startswith("Cần tác giả xem lại vì: tài liệu nguồn có chỗ ghi khác nhau hoặc có thể đã "
                                        "cũ; kiểm tra tự động phát hiện vấn đề lớn."), bare)
        whole = note(review_reasons=[REVIEW_REASON_KEPT_SLOTS], review_slots=[1], whole_fallback=True,
                     slot_reasons={1: [WORKSHEET_INCOMPLETE_CODE]})
        self.assertNotIn("chi tiết bên dưới", whole)
        flagged = note(review_reasons=[REVIEW_REASON_SOURCE],
                       source_flags=["Tài liệu ghi khác nhau ở phần “Ba cấp độ”: “Cấp 2 được phản hồi trong 48 giờ” "
                                     "(trang 7). Hãy kiểm tra trước khi áp dụng."])
        self.assertIn("có thể đã cũ (chi tiết bên dưới). Tài liệu ghi khác nhau", flagged)

    def test_a_kept_slots_reason_is_named_once(self) -> None:
        # c1.l2.u2: "Giữ bản AI …: phiếu thực hành thiếu phần." then "còn cảnh báo: phiếu thực hành thiếu phần."
        worksheet = note(review_reasons=[REVIEW_REASON_KEPT_SLOTS], review_slots=[0],
                         slot_reasons={0: [WORKSHEET_INCOMPLETE_CODE]},
                         remaining=[SlotFinding(WORKSHEET_INCOMPLETE_CODE, 0)])
        self.assertIn("Giữ bản AI để tác giả rà soát: khối 1 (Lý thuyết): phiếu thực hành thiếu phần.", worksheet)
        self.assertNotIn("còn cảnh báo", worksheet)
        self.assertNotIn("không phát hiện vấn đề lớn", worksheet)
        # c2.l2.u1: the kept reason and its own "Nên biên tập" line said the same.
        filler = note(review_slots=[1], slot_reasons={1: [EXPLANATION_FILLER_CODE]},
                      remaining=[SlotFinding(EXPLANATION_FILLER_CODE, 1)])
        self.assertIn("Giữ bản AI để tác giả rà soát: khối 2 (Quiz). Nên biên tập khối 2 (Quiz): lời giải dùng chữ "
                      "sáo rỗng", filler)
        self.assertEqual(filler.count("sáo rỗng"), 1)


# The run's slide deck: a photo beside the text of a slide (12-37 % of the page) and picture pages (41-99 %).
SHA = "b" * 64


def region(page: int, share: float | None, *, asset: str = SHA, status: str = "unreviewed",
           box: tuple[float, float, float, float] = (0.5, 0.2, 0.95, 0.9)) -> dict[str, Any]:
    raw: dict[str, Any] = {"region_kind": "embedded_image" if share is not None else "vector_composition",
                           "asset_revision": asset, "locator": {"page": page, "bbox_normalized": list(box)},
                           "observation": {"status": status, "facts": []},
                           "inference": {"status": "not_performed", "claims": []}}
    if share is not None:
        raw["page_area_ratio"] = share
    return raw


class ImagePageTests(unittest.TestCase):
    """Item 9: the note's image warning only for a page that is mostly an image nobody has read."""

    def test_only_a_page_mostly_an_unread_image_counts(self) -> None:
        decorative = [{"visual_regions": [region(14, 0.35)]}, {"visual_regions": [region(15, 0.27, asset="c" * 64)]}]
        self.assertEqual(unread_image_pages(decorative), [])
        self.assertFalse(evidence_needs_check(decorative))
        # The same photo carried by two facts counts once; two photos of one page add up.
        self.assertEqual(unread_image_pages([{"visual_regions": [region(21, 0.3)]},
                                             {"visual_regions": [region(21, 0.3)]}]), [])
        self.assertEqual(unread_image_pages([{"visual_regions": [region(29, 0.25), region(29, 0.2, asset="d" * 64)]}]),
                         [29])
        self.assertEqual(unread_image_pages([{"visual_regions": [region(39, 0.69)]}]), [39])
        # A region somebody observed is read; a vector drawing counts by its box.
        self.assertEqual(unread_image_pages([{"visual_regions": [region(39, 0.69, status="observed")]}]), [])
        self.assertEqual(unread_image_pages([{"visual_regions": [region(12, None, box=(0.1, 0.1, 0.9, 0.9))]}]), [12])
        # A legacy document: nothing tells its pages apart, the warning stays.
        self.assertTrue(evidence_needs_check([{"source_evidence_status": "legacy_review_required"}]))

    def test_the_unit_warns_only_for_an_image_page(self) -> None:
        for share, expected in ((0.3, False), (0.7, True)):
            body = severity_body()
            for fact in body["unit_contract"]["source_facts"]:
                fact["locator"] = {**(fact.get("locator") or {}), "visual_regions": [region(7, share)]}
            rehash_contract(body["unit_contract"])
            request = RagLessonAuthorUnitV2Request.model_validate(body)
            with self.subTest(share=share):
                self.assertEqual(unit_service._idm_unit_deps(request).evidence_review_required, expected)


def flat(text: str) -> str:
    return " ".join(text.split())


class NamingFidelityTests(unittest.TestCase):
    """Item 11: a framework and its parts keep the source's names; no part named as the whole, no invented axes or
    steps (the run renamed a map after one of its parts and gave a two-group framework axes and steps of its own)."""

    def test_the_writer_and_the_module_keep_the_source_names(self) -> None:
        self.assertIn("Name a framework, model, matrix or process and each of its parts exactly as SOURCE_FACTS name "
                      "it (their term and language, or the translation the facts themselves give): never call a part "
                      "by the name of the whole or the whole by the name of one part, never rename it with your own "
                      "metaphor, and never give it axes, dimensions, criteria, steps or stages the facts do not list",
                      flat(PRACTICE_RULES))
        module = module_prompt("vi", module_plan={}, lesson_plans=[], audience="", objectives=[], must_dos=[],
                               course_blocks=[], facts=[], allowed_components=["html"])
        self.assertIn("A title that names a framework, model or process, or one of its parts, uses the name the facts "
                      "give it: never a part's name for the whole, never a new metaphor, never axes or steps the facts "
                      "do not list.", flat(module))



def worksheet_section(heading: str, *blocks: dict[str, Any]) -> dict[str, Any]:
    return {"heading": heading, "learning_block_ids": [],
            "blocks": [{"text": None, "items": [], "rows": [], **block} for block in blocks]}


INPUT_HEADING = "Dữ liệu đầu vào"
# The shape of c1.l2.u2 in the paid run: the worksheet with its new input section was 411 words against 400 and its
# self-check items were not the criteria facts (incomplete); the repair answered the same.
GUIDANCE = (" Đọc kỹ mô tả của khách hàng, ghi lại thiệt hại, số lần phàn nàn và các yếu tố an toàn, pháp lý, truyền "
            "thông trước khi chọn cấp độ cho từng trường hợp.")


def long_worksheet(body: dict[str, Any]) -> dict[str, Any]:
    writer = worksheet_writer(body)
    task, template, example, check = writer["components"]["c0"]["semantic_content"]["sections"]
    complaints = task["blocks"].pop()
    for row in template["blocks"][0]["rows"]:
        row["value"] += GUIDANCE * 4
    check["blocks"][0]["items"] = ["Bảng của bạn trình bày gọn gàng, dễ đọc.",
                                   "Bạn hoàn thành bảng trong thời gian quy định."]
    writer["components"]["c0"]["semantic_content"]["sections"] = [
        task, worksheet_section(INPUT_HEADING, complaints), template, example, check]
    return writer


class WorksheetBudgetTests(StoryboardEndpointTestCase):
    """R1 of the paid verification of 3ed380d: a worksheet over Node's budget that also has a soft finding fell back
    to its loose source lines; it is compacted, keeps its input data and is kept for the author's review."""

    async def test_an_over_budget_incomplete_worksheet_is_compacted_and_kept(self) -> None:
        body = worksheet_body()
        writer = long_worksheet(body)
        request = RagLessonAuthorUnitV2Request.model_validate(body)
        contract = request.unit_contract
        budget = build_idm_expected(contract, parse_brief(contract), unit_service._idm_unit_deps(request),
                                    "vi")["instructional_output_budget"]
        semantic = writer["components"]["c0"]["semantic_content"]
        self.assertGreater(len(re.findall(r"\w+", visible_text(semantic))), budget["max_words"])
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [slot_repair(writer, 0)], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200, data.get("detail"))
        self.assertEqual(provider.names, [WRITER, REPAIR, JUDGE])
        self.assert_envelope(data, "provider_validated", "review_required", "provider")
        slot = data["unit"]["components"][0]
        self.assertNotIn("source_locked_fallback", slot)
        headings = [section["heading"] for section in slot["semantic_content"]["sections"]]
        self.assertIn(INPUT_HEADING, headings)
        self.assertEqual(len(slot["semantic_content"]["sections"][headings.index(INPUT_HEADING)]["blocks"][0]["items"]),
                         3)
        self.assertLessEqual(len(re.findall(r"\w+", visible_text(slot["semantic_content"]))), budget["max_words"])
        note = data["unit"]["idm_quality"]["author_note"]
        self.assertIn("Giữ bản AI để tác giả rà soát: khối 1 (Lý thuyết): phiếu thực hành thiếu phần.", note)
        self.assertIn("Phiếu thực hành dài hơn giới hạn nên đã rút gọn", note)
        self.assertNotIn("bản dự phòng", note)

    def test_compaction_never_drops_the_input_data(self) -> None:
        writer = long_worksheet(worksheet_body())
        slot = writer["components"]["c0"]
        # Only the shortest candidate fits: the task, the input, the template (short rows) and the self-check.
        compacted = compact_worksheet(slot, lambda value: "Ví dụ" not in json.dumps(value, ensure_ascii=False))
        self.assertIsNotNone(compacted)
        assert compacted is not None
        headings = [section["heading"] for section in compacted[0]["semantic_content"]["sections"]]
        self.assertIn(INPUT_HEADING, headings)
        self.assertFalse(compacted[1])

    def test_a_worksheet_without_a_form_falls_back_on_its_practice_plan(self) -> None:
        plan = {"sentence": "Phân loại năm nhân sự của dự án vào đúng vai trò trong hội đồng mua hàng.",
                "context_input": "Một dự án gồm trợ lý giám đốc, trưởng phòng tài chính, kỹ sư trưởng và trưởng kho.",
                "learner_action": "Ghi vai trò của từng nhân sự.", "result": "Bảng phân loại năm nhân sự.",
                "criterion": "Tiêu chí gợi ý: mỗi nhân sự đúng một vai trò theo bài học."}
        html = plan_worksheet_html("Phiếu phân loại vai trò", plan, "vi") or ""
        self.assertEqual(re.findall(r"<h3>([^<]+)</h3>", html),
                         ["Nhiệm vụ", "Dữ liệu đầu vào", "Cách làm", "Kết quả cần có", "Tự kiểm tra"])
        self.assertIn("<li>Mỗi nhân sự đúng một vai trò theo bài học.</li>", html)
        self.assertNotIn("gợi ý", html)
        # The facts' complete statements stay word for word, what the learner applies.
        rule = "Người quyết định là người duyệt ngân sách cuối cùng của dự án mua sắm."
        with_rule = plan_worksheet_html("Phiếu phân loại vai trò", plan, "vi", [rule]) or ""
        self.assertIn(f"<h3>Nội dung cần áp dụng</h3><ul><li>{rule}</li></ul><h3>Cách làm</h3>", with_rule)
        self.assertIsNone(plan_worksheet_html("Phiếu", {"sentence": " "}, "vi"))

    async def test_fallback_only_never_shows_the_loose_source_lines(self) -> None:
        # c1.l2.u2's facts held a cut slide heading and a discussion question, no form: here the slide heading.
        question = "Câu hỏi thảo luận 5 phút: Nhóm bạn còn những nhóm khách hàng nào khác?"
        body = facts_of(worksheet_body(), {"a": question})
        status, data, provider = await self.post({**body, "fallback_only": True})
        self.assertEqual((status, provider.calls), (200, []), data.get("detail"))
        html = data["unit"]["components"][0]["html"]
        self.assertIn("<h3>Nhiệm vụ</h3>", html)
        self.assertIn("<h3>Dữ liệu đầu vào</h3>", html)
        self.assertNotIn("Câu hỏi thảo luận", html)
        self.assertEqual(data["unit"]["components"][0]["source_locked_fallback"], True)


# c1.l3.u4 of the paid run: the worked example solved the three interview entries of the input data.
REUSED = {"semantic_content": {"sections": [
    worksheet_section("Nhiệm vụ", {"kind": "task", "text": "Phân loại các ý kiến phỏng vấn vào ba tầng."}),
    worksheet_section("Dữ liệu phỏng vấn đầu vào", {"kind": "bullets", "items": [
        "Ý kiến 1: Tôi cần duyệt khoản đầu tư có tỷ suất hoàn vốn cao và phân bổ nguồn lực chuẩn xác.",
        "Ý kiến 2: Tôi lo áp lực khi là người chịu trách nhiệm cuối cùng cho kết quả kinh doanh.",
        "Ý kiến 3: Tôi muốn xây một tổ chức tự vận hành và để lại giá trị lâu dài."]}),
    worksheet_section("Mẫu phân tích hoàn chỉnh", {"kind": "table", "rows": [
        {"label": "Ý kiến 1 (duyệt đầu tư hoàn vốn cao)", "value": "Việc chức năng: phân bổ nguồn lực và dòng tiền."},
        {"label": "Ý kiến 2 (áp lực trách nhiệm)", "value": "Nỗi đau cảm xúc: áp lực của người chịu trách nhiệm."},
        {"label": "Ý kiến 3", "value": "Việc xã hội: một tổ chức tự vận hành, giá trị lâu dài."}]}),
    worksheet_section("Tự kiểm tra", {"kind": "bullets", "items": ["Mỗi ý kiến đúng một tầng.", "Có đủ ba tầng."]}),
]}}


class ExampleReuseTests(StoryboardEndpointTestCase):
    """R2 of the paid verification of 3ed380d: a worked example on the task's own data gives the answer away."""

    def test_an_example_on_the_tasks_entries_is_found(self) -> None:
        self.assertEqual(reused_entries(REUSED), [1, 2, 3])
        self.assertEqual(reuse_hint([1, 2, 3]), " (input entries 1, 2, 3 reappear in the worked example)")
        # c2.l2.u2: the example's statement copies the strength the learner is asked to use.
        copied = json.loads(json.dumps(REUSED, ensure_ascii=False))
        sections = copied["semantic_content"]["sections"]
        sections[1]["blocks"][0]["items"] = ["Thế mạnh được khách hàng khen ngợi: ứng dụng công nghệ AI kết hợp "
                                             "phương pháp thiết kế bài giảng hiện đại."]
        sections[2]["blocks"][0]["rows"] = [{"label": "Câu tuyên ngôn", "value": (
            "Giúp giám đốc nhân sự rút ngắn thời gian nhờ ứng dụng công nghệ AI kết hợp thiết kế bài giảng hiện đại.")}]
        self.assertEqual(reused_entries(copied), [1])

    def test_an_example_on_other_data_is_no_reuse(self) -> None:
        other = json.loads(json.dumps(REUSED, ensure_ascii=False))
        other["semantic_content"]["sections"][2]["blocks"][0]["rows"] = [
            {"label": "Phát biểu mẫu (trưởng phòng mua hàng)", "value": "Việc chức năng: rút ngắn thời gian duyệt "
                                                                          "báo giá."},
            {"label": "Phát biểu mẫu (giám đốc vận hành)", "value": "Nỗi đau cảm xúc: sợ gián đoạn dây chuyền."}]
        self.assertEqual(reused_entries(other), [])
        # Without an input or an example section nothing is measured.
        self.assertEqual(reused_entries({"semantic_content": {"sections": REUSED["semantic_content"]["sections"][:2]}}),
                         [])

    async def test_the_writer_moves_the_example_once_else_the_unit_is_reviewed(self) -> None:
        body = worksheet_body()
        writer = worksheet_writer(body)
        sections = writer["components"]["c0"]["semantic_content"]["sections"]
        complaints = sections[0]["blocks"].pop()
        sections.insert(1, worksheet_section(INPUT_HEADING, complaints))
        example = sections[3]
        example["blocks"][0]["text"] = ("Ví dụ: khách hàng bị thiệt hại tài chính dưới 50 triệu đồng vì lô hàng giao "
                                        "thiếu: ghi cấp 2 và thông báo trưởng nhóm.")
        self.assertEqual(reused_entries(writer["components"]["c0"]), [2])
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [slot_repair(writer, 0)], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200, data.get("detail"))
        self.assertIn("c0 IDM_W5_WORKSHEET_EXAMPLE_REUSES_INPUT at c0: the worked example works on entries of the "
                      "task's own input data", provider.calls[1]["prompt"])
        self.assertIn("(input entries 2 reappear in the worked example)", provider.calls[1]["prompt"])
        self.assertNotIn("source_locked_fallback", data["unit"]["components"][0])
        self.assertEqual(data["quality_state"], "review_required")
        self.assertIn("Cần xem khối 1 (Lý thuyết): ví dụ mẫu của phiếu thực hành giải chính các mục dữ liệu đầu vào",
                      data["unit"]["idm_quality"]["author_note"])

    def test_the_prompt_keeps_the_example_off_the_tasks_data(self) -> None:
        self.assertIn("The worked example works on other data: never one of the input entries, not even shortened or "
                      "renamed", flat(_WORKSHEET_RULE))
        self.assertIn("and the input data counts toward it", flat(_WORKSHEET_RULE))


class OptionCountTests(unittest.TestCase):
    """R3 of the paid verification of 3ed380d (c2.l4.u2): three listed items named by three options."""

    def test_options_that_name_three_items_are_one_too_few(self) -> None:
        three = stem_quiz("Giả sử khách hàng lo hệ thống mới làm gián đoạn vận hành.\n- Đề xuất 1: Gửi hồ sơ năng lực."
                          "\n- Đề xuất 2: Chạy thử có giới hạn kèm cam kết chuyển dữ liệu an toàn.\n- Đề xuất 3: Mời "
                          "dự hội thảo một ngày.\nĐề xuất nào phù hợp nhất?",
                          [("Đề xuất 3", False), ("Đề xuất 2", True), ("Đề xuất 1", False)], "")
        laid = lay_out_quiz(three)[0]
        self.assertEqual(question_report(laid)[QUESTION_FEW_OPTIONS_CODE],
                         ["3 options that name the listed items (write 4)"])
        # Options that state their content may stay three (the prompt's exception); four named options are fine.
        stated = {**three, "choices": [{"text": "Chạy thử có giới hạn kèm cam kết", "correct": True},
                                       {"text": "Gửi hồ sơ năng lực", "correct": False},
                                       {"text": "Mời dự hội thảo", "correct": False}]}
        self.assertNotIn(QUESTION_FEW_OPTIONS_CODE, question_report(lay_out_quiz(stated)[0]))
        four = {**three, "question": three["question"].replace("\nĐề xuất nào", "\n- Đề xuất 4: Giảm giá 5%.\nĐề "
                                                                                 "xuất nào"),
                "choices": [*three["choices"], {"text": "Đề xuất 4", "correct": False}]}
        self.assertNotIn(QUESTION_FEW_OPTIONS_CODE, question_report(lay_out_quiz(four)[0]))

    def test_the_prompt_lists_one_item_per_option(self) -> None:
        self.assertIn("options that name the listed items have one item each, so the case lists 4 items for 4 options",
                      flat(PRACTICE_RULES))
