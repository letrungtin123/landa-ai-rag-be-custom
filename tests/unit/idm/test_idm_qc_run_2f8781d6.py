"""Fixes from the QC report of the real AI ID run 2f8781d6 (AI 7f67137, 2026-10-23, 7,1/10), wave A.

Every case below is synthetic text with the shape of a defect of that run (no course text of the run); every provider
is a local fake.
"""

from __future__ import annotations

import html as html_lib
import re
import unittest
from pathlib import Path
from typing import Any, ClassVar

from app.content.instructional_quality import clean_source_facts
from app.idm import prompts_unit
from app.idm.acronyms import explained
from app.idm.fallback_lines import faq_topic, restore_stops, short_name, split_glued, tidy_facts
from app.idm.faq_checks import FAQ_HYPE_CODE
from app.idm.node_acceptance import html_quality_code
from app.idm.prompts_unit import rule_line
from app.idm.qa import JudgeOutcome, SlotFinding, _code_text, build_unit_author_note
from app.idm.quiz_letters import expand_short_item_labels
from app.idm.quiz_text import EXPLANATION_FILLER_CODE
from app.idm.slot_words import hype_words
from app.idm.sortable import without_order_labels
from app.idm.source_fallback import fallback_glued_words, idm_statement_faq
from app.idm.source_locked import render_idm_source_locked_html
from app.idm.storyboard import IdmUnitWriter, parse_brief
from app.idm.visual_pages import evidence_needs_check, page_texts, unread_image_pages
from app.schemas.orchestration_v2 import RagLessonAuthorUnitV2Request
from app.services.orchestration_v2 import unit as unit_service
from app.services.orchestration_v2.source_locked import render_source_locked_html
from tests.support.idm_test_support import make_runtime, rehash_contract
from tests.support.log_support import logged_events
from tests.unit.idm.test_idm_fallback_matrix import rehashed, unit_findings
from tests.unit.idm.test_idm_qc_run_5496155c import region
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
    slot_repair,
)

HEADER = "Row 1: VIỆC CẦN LÀM - Mục tiêu | NỖI ĐAU - Điều cản trở | LỢI ÍCH - Điều mong đợi"
FUNCTIONAL_JOB = "Về Chức năng (Functional Job)"
EMOTIONAL_JOB = "Về Cảm xúc (Emotional Job)"
SOCIAL_PAIN = "Về Ý nghĩa / Xã hội (Social/Meaningful Pain)"
EMOTIONAL_GAIN = "Về Cảm xúc (Emotional Gain)"
SOCIAL_GAIN = "Về Ý nghĩa / Xã hội (Social/Meaningful Gain)"
# Three profiles with the shapes of pages 24-26 of run 2f8781d6: the first one written inline with a label the PDF cut
# ("(Social/Meaningful" + "Job): *") and the ends of two more labels whose starts the PDF reader dropped as repeats
# ("Pain): *", "Gain): *"); the second and third ones with the same labels on their own lines (the second one writes
# the two cut labels in full, as page 25 does: review 5, L2).
PROFILE_FACTS = [
    "1.3 HỒ SƠ KHÁCH HÀNG B2B (VD1: TRƯỞNG PHÒNG MUA HÀNG)",
    HEADER,
    f"Row 2: {FUNCTIONAL_JOB}: * Chọn nhà cung cấp đúng hạn và đúng ngân sách của bộ phận nội bộ.Phân loại hồ sơ chào "
    "hàng trước khi trình lên cấp trên.",
    "Về Ý nghĩa / Xã hội (Social/Meaningful",
    "Job): * Được cấp trên tin tưởng giao gói mua sắm lớn hơn. | - Về Chức năng (Functional Pain): * Bị",
    "\"bủa vây\" bởi hồ sơ chào hàng dài dòng khiến họphải đọc lại nhiều lần.",
    "Pain): * Sợ bị coi là một \"nút thắt\"",
    "(bottleneck) làm chậm tiến độ của cả phòng.",
    "Gain): * Được đồng nghiệp nể trọng vì chọn đúng đối tác tốt.",
    "1.3 HỒ SƠ KHÁCH HÀNG B2B (VD2: GIÁM ĐỐC TÀI CHÍNH)",
    HEADER,
    f"Row 2: {FUNCTIONAL_JOB}",
    "Phê duyệt khoản đầu tư có tỷ suất hoàn vốn rõ ràng cho công ty.",
    EMOTIONAL_JOB,
    "Muốn chắc chắn họ phải kiểm soát được dòng tiền trước mỗi quyết định lớn.",
    SOCIAL_PAIN,
    "Sợ bị đánh giá là người quản lý lỗi thời trước hội đồng quản trị.",
    EMOTIONAL_GAIN,
    "An tâm khi đội ngũ tự đồng bộ số liệu tại từng điểm chạm.",
    SOCIAL_GAIN,
    "Được hội đồng quản trị ghi nhận vì dẫn dắt công ty chuyển mình.",
    "1.3 HỒ SƠ KHÁCH HÀNG B2B (VD3: NHÂN VIÊN KHO)",
    HEADER,
    f"Row 2: {FUNCTIONAL_JOB}",
    "Hoàn thành chỉ tiêu xuất kho hằng ngày nhanh và chính xác.",
    EMOTIONAL_JOB,
    "Muốn làm việc thoải mái, không căng thẳng với công cụ mới.",
    SOCIAL_PAIN,
    "Sợ bị mang tiếng chậm chạp trong mắt trưởng ca và đồng nghiệp.",
    EMOTIONAL_GAIN,
    "Nhẹ nhõm khi công việc trôi chảy, không còn lỗi vặt của hệ thống.",
]
# Each row and the label it must sit under (the source's own, in source order).
ROWS = [
    ("Chọn nhà cung cấp đúng hạn", FUNCTIONAL_JOB),
    ("Được cấp trên tin tưởng", "Về Ý nghĩa / Xã hội (Social/Meaningful Job)"),
    ("bởi hồ sơ chào hàng dài dòng", "Về Chức năng (Functional Pain)"),
    ("Sợ bị coi là một", SOCIAL_PAIN),
    ("Được đồng nghiệp nể trọng", SOCIAL_GAIN),
    # Review 5 (L3): the second profile repeats labels of the first one (the leads of its rows), so all its labels
    # carry its name, as the third one's do.
    ("Phê duyệt khoản đầu tư", f"Giám đốc tài chính – {FUNCTIONAL_JOB}"),  # noqa: RUF001 - the en dash
    ("Muốn chắc chắn họ phải", f"Giám đốc tài chính – {EMOTIONAL_JOB}"),  # noqa: RUF001 - the en dash
    ("Sợ bị đánh giá là người quản lý", f"Giám đốc tài chính – {SOCIAL_PAIN}"),  # noqa: RUF001 - the en dash
    ("An tâm khi đội ngũ", f"Giám đốc tài chính – {EMOTIONAL_GAIN}"),  # noqa: RUF001 - the en dash
    ("Được hội đồng quản trị ghi nhận", f"Giám đốc tài chính – {SOCIAL_GAIN}"),  # noqa: RUF001 - the en dash
    ("Hoàn thành chỉ tiêu xuất kho", f"Nhân viên kho – {FUNCTIONAL_JOB}"),  # noqa: RUF001 - the en dash
    ("Muốn làm việc thoải mái", f"Nhân viên kho – {EMOTIONAL_JOB}"),  # noqa: RUF001 - the en dash
    ("Sợ bị mang tiếng chậm chạp", f"Nhân viên kho – {SOCIAL_PAIN}"),  # noqa: RUF001 - the en dash
    ("Nhẹ nhõm khi công việc", f"Nhân viên kho – {EMOTIONAL_GAIN}"),  # noqa: RUF001 - the en dash
]
REPO_ROOT = Path(__file__).resolve().parents[3]
_LABEL_RE = re.compile(r"\((?:Functional|Emotional|Social/Meaningful) (?:Job|Pain|Gain)\)")


def label_of_rows(html: str) -> dict[str, str]:
    """Row start -> the label it reads under: the last label shown before it (a heading, a paragraph that is only a
    label, or the bold lead of its own paragraph)."""

    current = ""
    found: dict[str, str] = {}
    for tag, inner in re.findall(r"<(h3|p|li|td|th)>(.*?)</\1>", html):
        lead = re.match(r"<strong>(.*?):</strong>\s*(.*)", inner)
        text = html_lib.unescape(re.sub(r"<[^>]+>", "", lead.group(2) if lead else inner))
        if lead:
            current = html_lib.unescape(lead.group(1))
        elif _LABEL_RE.search(text) and len(text) < 90 and tag in {"h3", "p"}:
            current = text
            continue
        for start, _label in ROWS:
            if start in text:
                found[start] = current
    return found


def many_facts_body(texts: list[str]) -> dict[str, Any]:
    """lsn_003 unit 1 (one html slot) with ``texts`` as its facts, in order."""

    body = body_for(1, 0, 0)
    contract = body["unit_contract"]
    template = contract["source_facts"][0]
    facts = [{**template, "fact_key": f"d1-c6-f{index}", "fact_text": text} for index, text in enumerate(texts, 1)]
    keys = [fact["fact_key"] for fact in facts]
    contract["source_facts"] = facts
    contract["unit_source_fact_ids"] = keys
    contract["component_plan"][0]["source_fact_ids"] = keys
    contract["idm_unit_brief"]["components"][0]["owned_fact_keys"] = keys
    return rehashed(body)


class RepeatedLabelFallbackTests(unittest.TestCase):
    """A1 (N1, heavy): the html fallback keeps a row label the profiles repeat, so every row stays under its own label,
    in source order; the result passes the duplicate rule Python and Node share."""

    def test_every_row_stays_under_its_own_label(self) -> None:
        html = render_idm_source_locked_html("Bộ hồ sơ mẫu", PROFILE_FACTS, locale="vi")
        self.assertEqual(label_of_rows(html), dict(ROWS))
        # The rows in source order.
        positions = [html.find(start) for start, _label in ROWS]
        self.assertEqual(positions, sorted(positions))
        self.assertNotIn(-1, positions)
        # No label nor table header repeats, so the Node-mirrored quality rule accepts the slot.
        self.assertIsNone(html_quality_code(html))
        self.assertEqual(html.count("VIỆC CẦN LÀM - Mục tiêu"), 1)

    def test_a_label_is_prefixed_only_under_a_new_section(self) -> None:
        html = render_idm_source_locked_html("Bộ hồ sơ mẫu", PROFILE_FACTS, locale="vi")
        self.assertEqual(html.count(f"Nhân viên kho – {EMOTIONAL_JOB}"), 1)  # noqa: RUF001 - the en dash
        self.assertNotIn(f"Trưởng phòng mua hàng – {FUNCTIONAL_JOB}", html)  # noqa: RUF001 - the first profile
        # A label repeated inside one section (a running header) is dropped as before.
        repeated = render_idm_source_locked_html("Hồ sơ", [*PROFILE_FACTS[9:15], EMOTIONAL_JOB, "Muốn cảm thấy yên "
                                                          "tâm khi duyệt mọi khoản chi lớn."], locale="vi")
        self.assertEqual(repeated.count(EMOTIONAL_JOB), 1)

    def test_text_the_pdf_glued_is_mended_where_it_is_certain(self) -> None:
        html = render_idm_source_locked_html("Bộ hồ sơ mẫu", PROFILE_FACTS, locale="vi")
        self.assertIn("bộ phận nội bộ. Phân loại hồ sơ", html)
        # "họ" and "phải" are words of the slot's facts: "họphải" has one split only.
        self.assertIn("khiến họ phải đọc lại", html)
        self.assertEqual(fallback_glued_words([{"type": "html", "html": html}], [0]), [])

    def test_the_whole_unit_fallback_passes_python_and_node(self) -> None:
        body = many_facts_body(PROFILE_FACTS)
        deps = unit_service._idm_unit_deps(RagLessonAuthorUnitV2Request.model_validate(body))
        unit = deps.source_locked_unit
        assert unit is not None
        self.assertEqual(unit_findings(body, unit), (None, []))
        self.assertEqual(label_of_rows(unit["components"][0]["html"]), dict(ROWS))

    def test_the_legacy_renderer_is_unchanged(self) -> None:
        # R3: the shared renderer gets the cleaned facts and repeats as before.
        legacy = render_source_locked_html("Bộ hồ sơ mẫu", clean_source_facts(PROFILE_FACTS,
                                                                           preserve_table_numeric=True), locale="vi")
        self.assertNotIn("Nhân viên kho –", legacy)  # noqa: RUF001 - the en dash
        self.assertNotIn("nội bộ. Phân loại", legacy)


class FallbackTextTests(unittest.TestCase):
    """A1: the space after a full stop and the glued words, only where the fix is certain."""

    def test_a_full_stop_gets_its_space_only_between_sentences(self) -> None:
        self.assertEqual(restore_stops("vận hành nội bộ.Phân loại hồ sơ"), "vận hành nội bộ. Phân loại hồ sơ")
        self.assertEqual(restore_stops("thợ săn (salespeople).Cảm thấy"), "thợ săn (salespeople). Cảm thấy")
        self.assertEqual(restore_stops("đúng thứ anh cần\").Nâng tầm"), "đúng thứ anh cần\"). Nâng tầm")
        for kept in ("Văn phòng tại TP.HCM và Hà Nội", "tăng 3.5 lần", "v.v.Các mục khác", "xem www.Example.com",
                     "gửi an.Nguyen@example.com", "https://example.com/Trang", "ISO.Tiêu chuẩn", "năm 2024.Doanh"):
            self.assertEqual(restore_stops(kept), kept, kept)

    def test_a_glued_word_is_split_only_where_the_facts_show_how(self) -> None:
        facts = ["Khiến họphải đọc lại.", "Họ phải kiểm tra hồ sơ.", "Từ chối nhẹ nhàng thì bỏlỡ cơ hội."]
        tidy = tidy_facts(facts)
        self.assertEqual(tidy[0], "Khiến họ phải đọc lại.")
        # "bỏ" and "lỡ" never stand alone in these facts: the word stays as written.
        self.assertIn("bỏlỡ", tidy[2])
        # Two possible splits: none is taken.
        self.assertEqual(split_glued("họphải", frozenset({"họ", "phải", "họp", "hải"})), "họphải")
        self.assertEqual(split_glued("Họphải", frozenset({"họ", "phải"})), "Họ phải")

    def test_the_short_name_of_a_section(self) -> None:
        self.assertEqual(short_name("1.3 HỒ SƠ KHÁCH HÀNG B2B (VD3: NGƯỜI SỬ DỤNG TRỰC TIẾP)"),
                         "Người sử dụng trực tiếp")
        self.assertEqual(short_name("2.3 LỘ TRÌNH TRẢI NGHIỆM KHÁCH HÀNG (WORKBOOK)"),
                         "Lộ trình trải nghiệm khách hàng")
        # Review 5 (M3): a step's words without its marker (a prefixed label is never a numbered step).
        self.assertEqual(short_name("BƯỚC 2: XỬ LÝ KHIẾU NẠI VIP"), "Xử lý khiếu nại VIP")
        self.assertEqual(short_name("BƯỚC 2"), "Bước 2")
        self.assertIsNone(short_name("1.3 " + "RẤT DÀI " * 10))

    def test_two_sections_of_one_short_name_never_repeat_a_label(self) -> None:
        facts = ["HỒ SƠ (VD1: KHO HÀNG)", EMOTIONAL_JOB, "Muốn làm việc thoải mái với công cụ mới.",
                 "HỒ SƠ (VD2: KHO HÀNG)", EMOTIONAL_JOB, "Muốn được hướng dẫn rõ ràng khi đổi ca.",
                 "HỒ SƠ (VD3: KHO HÀNG)", EMOTIONAL_JOB, "Muốn tự tin khi nhập số liệu tồn kho."]
        html = render_idm_source_locked_html("Hồ sơ", facts, locale="vi")
        self.assertIsNone(html_quality_code(html))
        self.assertEqual(html.count(f"Kho hàng – {EMOTIONAL_JOB}"), 1)  # noqa: RUF001 - the en dash


class FallbackOnlyProfileTests(StoryboardEndpointTestCase):
    """A1 through the endpoint: a profile unit answered ``fallback_only`` is valid and keeps every row's label."""

    async def test_fallback_only(self) -> None:
        body = {**many_facts_body(PROFILE_FACTS), "fallback_only": True}
        status, data, provider = await self.post(body)
        self.assertEqual((status, provider.calls), (200, []), data.get("detail"))
        self.assertEqual(unit_findings(body, data["unit"]), (None, []))
        self.assertEqual(label_of_rows(data["unit"]["components"][0]["html"]), dict(ROWS))
        # No glued word left, so the note names none.
        self.assertNotIn("dính liền", data["unit"]["idm_quality"]["author_note"])



def item_quiz(label: str, *, names: tuple[str, ...] | None = None, extra: str = "") -> dict[str, Any]:
    """A single-choice quiz whose options only name the stem's items "<label> 1" … "<label> 4"."""

    items = "\n".join(f"- {label} {number}: Phương pháp số {number} để chia giai đoạn của hành trình."
                      for number in range(1, 5))
    return {"problem_type": "multiple_choice",
            "question": f"Giả sử bạn đọc bốn cách làm sau:\n{items}\nCách nào đúng?{extra}",
            "choices": [{"text": name, "correct": position == 0}
                        for position, name in enumerate(names or tuple(f"{label} {n}" for n in range(1, 5)))],
            "explanation": f"Nguyên tắc: chia theo thời gian.\nA - Đúng vì {label} 1 chia theo thời gian.\nB - Sai vì "
                           f"{label} 2 bỏ tâm thế.\nC - Sai vì {label} 3 gộp điểm đau.\nD - Sai vì {label} 4 xếp sai."}


class ShortItemLabelTests(unittest.TestCase):
    """A2: an option that only names a short stem item ("Ý 1") is under the shared validator's 8 characters; the label
    is written out in a longer noun everywhere the quiz uses it, never "Phương án"."""

    def test_the_label_is_written_out_in_the_items_and_options(self) -> None:
        for label, longer in (("Ý", "Ý kiến"), ("Mục", "Mục số"), ("Câu", "Câu số"), ("Bước", "Bước số"),
                              ("Item", "Item no.")):
            quiz, count = expand_short_item_labels(item_quiz(label))
            self.assertEqual(count, 8, label)
            self.assertEqual([choice["text"] for choice in quiz["choices"]], [f"{longer} {n}" for n in range(1, 5)])
            self.assertIn(f"- {longer} 2: Phương pháp số 2", quiz["question"])
            # Review 5 (M1): never a word inside a sentence; the explanation is left as written.
            self.assertEqual(quiz["explanation"], item_quiz(label)["explanation"])
            self.assertTrue(all(len(choice["text"]) >= 8 for choice in quiz["choices"]))
            self.assertNotIn("Phương án", quiz["question"])

    def test_left_as_written_when_it_is_not_certain(self) -> None:
        for quiz in (item_quiz("Ý", names=("Ý 1", "Ý 2", "Ý 3", "Ý 5")),  # an option names no listed item
                     item_quiz("Ý", names=("Ý kiến 1", "Ý kiến 2", "Ý kiến 3", "Ý kiến 4")),  # no short option
                     item_quiz("Nhóm")):  # not a short item label of the list
            self.assertEqual(expand_short_item_labels(quiz), (quiz, 0))
        skipped = item_quiz("Ý")
        skipped["question"] = skipped["question"].replace("- Ý 3:", "- Ý 7:")  # items not numbered 1, 2, 3 …
        self.assertEqual(expand_short_item_labels(skipped), (skipped, 0))


class ProblemSlotFallbackTests(StoryboardEndpointTestCase):
    """A2: a too-short option that the repair leaves sends only its question slot to the fallback; the FAQ keeps the AI
    draft (run 2f8781d6, c2.l3.u3: the whole unit fell back)."""

    async def test_a_short_option_sends_only_its_slot_to_the_fallback(self) -> None:
        body = escalate_body()
        writer = escalate_writer(body)
        writer["components"]["c0"]["choices"][0]["text"] = "Có"
        repair = slot_repair(writer, 0)
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [repair], JUDGE: [judge(), judge()]})
        with self.assertLogs("app.idm", level="INFO") as logs:
            status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200, data.get("detail"))
        self.assertEqual((data["content_origin"], data["quality_state"]), ("structured_fallback", "review_required"))
        question, faq = data["unit"]["components"]
        self.assertTrue(question["source_locked_fallback"])
        self.assertNotIn("source_locked_fallback", faq)
        self.assertEqual(faq["items"], writer["components"]["c1"]["items"])
        quality = logged_events(logs.records, "idm_unit_quality")[-1]
        self.assertEqual((quality["whole_fallback"], quality["fallback_slots"]), (False, [0]))
        note = data["unit"]["idm_quality"]["author_note"]
        self.assertIn("1 khối dùng bản dự phòng dựng từ tài liệu", note)
        self.assertIn("Tài liệu không đủ dữ kiện để dựng khối 1 (Quiz)", note)
        # A4: the reason in plain words, never "một kiểm tra tự động khác".
        self.assertIn("Lý do: khối 1 (Quiz): một phương án trả lời quá ngắn hoặc thiếu nội dung.", note)
        self.assertNotIn("kiểm tra tự động khác", note)
        self.assertEqual(unit_findings(body, data["unit"])[1], [])

    async def test_short_item_names_pass_once_written_out(self) -> None:
        body = escalate_body()
        writer = escalate_writer(body)
        writer["components"]["c0"].update({key: value for key, value in item_quiz("Ý").items()
                                            if key in {"question", "choices", "explanation"}})
        writer["components"]["c0"]["choices"] = writer["components"]["c0"]["choices"][:3]
        writer["components"]["c0"]["explanation"] = writer["components"]["c0"]["explanation"].rsplit("\nD", 1)[0]
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [slot_repair(writer, 0)], JUDGE: [judge(), judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200, data.get("detail"))
        question = data["unit"]["components"][0]
        self.assertNotIn("source_locked_fallback", question)
        self.assertEqual(sorted(choice["text"] for choice in question["choices"]), ["Ý kiến 1", "Ý kiến 2", "Ý kiến 3"])


class FallbackFaqQuestionTests(unittest.TestCase):
    """A2: a fallback FAQ item never quotes its own slot title ("… về 'Hỏi đáp: …'"); review 5 (M2): a statement is
    asked about by its place, as before, never by its own label."""

    STATEMENTS = ("Tầng 2: Tâm thế và kỳ vọng của khách hàng tại mỗi cột mốc của hành trình mua hàng.",
                  "Bản đồ hành trình giúp đội ngũ nhìn thấy điểm đau lớn nhất ở từng giai đoạn của khách hàng.")

    def test_the_question_names_the_topic(self) -> None:
        self.assertEqual(faq_topic("Hỏi đáp: Thực hành Bản đồ hành trình"), "Thực hành Bản đồ hành trình")
        self.assertEqual(faq_topic("FAQ: Journey map practice"), "Journey map practice")
        self.assertEqual(faq_topic("Giải đáp thắc mắc về điểm chạm"), "Giải đáp thắc mắc về điểm chạm")
        items = idm_statement_faq("Hỏi đáp: Thực hành Bản đồ hành trình", self.STATEMENTS, locale="vi")
        assert items is not None
        self.assertEqual([item["question"] for item in items], [
            "Người học cần ghi nhớ gì về 'Thực hành Bản đồ hành trình'?",
            "Điểm thứ 2 cần lưu ý trong 'Thực hành Bản đồ hành trình' là gì?"])
        self.assertEqual([item["answer"] for item in items], list(self.STATEMENTS))
        english = idm_statement_faq("FAQ: Journey map practice", self.STATEMENTS[1:] * 1 + (
            "The journey map shows the biggest pain point of every stage the customer goes through.",), locale="en")
        assert english is not None
        self.assertEqual(english[0]["question"], "What should learners remember about 'Journey map practice'?")
        self.assertNotIn("Hỏi đáp:", " ".join(item["question"] for item in items))


class SortableOrderLabelTests(StoryboardEndpointTestCase):
    """A3: the learner UI shuffles a sortable's items, so their leading ordinal labels never stay (VI and EN)."""

    def test_leading_ordinal_labels_go(self) -> None:
        cases = {
            "vi": (["Tầng 1: Cột mốc hành trình theo trục thời gian", "Tầng 2: Tâm thế và kỳ vọng của khách hàng",
                    "Tầng 3: Điểm đau trọng yếu ở từng giai đoạn", "Tầng 4: Điểm chạm và giải pháp tạo cảm xúc"],
                   ["Cột mốc hành trình theo trục thời gian", "Tâm thế và kỳ vọng của khách hàng",
                    "Điểm đau trọng yếu ở từng giai đoạn", "Điểm chạm và giải pháp tạo cảm xúc"]),
            "vi-stages": (["Giai đoạn 1 - xác định nhóm khách hàng mục tiêu",
                           "Giai đoạn 2 - Phân tích hành trình mua", "Giai đoạn 3 - Thiết kế điểm chạm tạo khác biệt",
                           "Giai đoạn 4 - Đo lường kết quả sau ba tháng"],
                          ["Xác định nhóm khách hàng mục tiêu", "Phân tích hành trình mua",
                           "Thiết kế điểm chạm tạo khác biệt", "Đo lường kết quả sau ba tháng"]),
            "vi-brackets": (["(1) Xác định nhóm khách hàng mục tiêu", "(2) Phân tích hành trình mua",
                             "(3) Thiết kế điểm chạm tạo khác biệt", "(4) Đo lường kết quả sau ba tháng"],
                            ["Xác định nhóm khách hàng mục tiêu", "Phân tích hành trình mua",
                             "Thiết kế điểm chạm tạo khác biệt", "Đo lường kết quả sau ba tháng"]),
            "en": (["Step 1: Map the customer milestones", "Step 2: Capture the mindset at each stage",
                    "Step 3 - List the pain points per stage", "Step 4) Design the touchpoints and fixes"],
                   ["Map the customer milestones", "Capture the mindset at each stage",
                    "List the pain points per stage", "Design the touchpoints and fixes"]),
        }
        for name, (items, expected) in cases.items():
            component, count = without_order_labels({"question_text": "Sắp xếp", "items": [{"text": t} for t in items]})
            self.assertEqual([item["text"] for item in component["items"]], expected, name)
            self.assertEqual(count, 4, name)

    def test_items_stay_as_written_when_stripping_would_break_them(self) -> None:
        for items in (["Bước 1: Lập", "Bước 2: Duyệt", "Bước 3: Gửi đi"],  # under the 14 characters of a step
                      ["1.5 triệu khách hàng mới trong quý", "Hai kênh bán hàng chính của công ty",
                       "Ba nhóm sản phẩm chủ lực năm nay"],  # a figure is no ordinal
                      ["Tầng 1: Điểm chạm của khách hàng", "Tầng 2: Điểm chạm của khách hàng",
                       "Tầng 3: Điểm đau lớn nhất của khách"],  # two items would read the same
                      # Review 5 (M4): labels of several forms are no single ordinal prefix.
                      ["Bước 1: xác định nhóm khách hàng mục tiêu", "Giai đoạn 2 - Phân tích hành trình mua",
                       "3. Thiết kế điểm chạm tạo khác biệt", "(4) Đo lường kết quả sau ba tháng"]):
            component = {"question_text": "Sắp xếp", "items": [{"text": text} for text in items]}
            self.assertEqual(without_order_labels(component), (component, 0))

    async def test_the_writer_and_the_fallback_items_carry_no_order(self) -> None:
        # The writer's items, as they are bound before validation.
        body = body_for(2, 0, 1)
        request = RagLessonAuthorUnitV2Request.model_validate(body)
        writer = IdmUnitWriter(request.unit_contract, parse_brief(request.unit_contract),
                               unit_service._idm_unit_deps(request), make_runtime(None))
        written = writer.normalize_component(1, {"type": "la_sortable", "question_text": "Sắp xếp các bước", "items": [
            {"text": f"Bước {number}: {step}"} for number, step in enumerate(
                ("Thực hiện xử lý hoặc chuyển hồ sơ", "Phản hồi kết quả cho khách hàng",
                 "Xác nhận khách hàng đồng ý", "Đóng hồ sơ và lưu trữ trên CRM"), 9)]})
        self.assertEqual(written["items"][0]["text"], "Thực hiện xử lý hoặc chuyển hồ sơ")
        self.assertEqual(writer.runtime.adjustments["w5_sortable_order_labels_stripped"], 4)
        # The source-locked sortable of golden lsn_004 u2 writes "Bước 9: …" before each step.
        status, data, _ = await self.post({**body, "fallback_only": True})
        self.assertEqual(status, 200, data.get("detail"))
        items = [item["text"] for item in data["unit"]["components"][1]["items"]]
        self.assertTrue(items)
        self.assertFalse([text for text in items if re.match(r"(?i)bước\s*\d", text)])
        self.assertEqual(unit_findings({**body, "fallback_only": True}, data["unit"]), (None, []))


class AuthorNoteTests(unittest.TestCase):
    """A4: the unit note's lines that misled the author in run 2f8781d6."""

    def note(self, **changes: Any) -> str:
        base: dict[str, Any] = {"locale": "vi", "judge": JudgeOutcome("pass"), "deterministic_codes": [],
                                "fallback_slots": [], "ai_drafted": False, "slot_types": ["html", "problem", "la_faq"]}
        return build_unit_author_note(**{**base, **changes})

    def test_no_major_issue_line_is_gone(self) -> None:
        self.assertEqual(self.note(), "")
        self.assertEqual(self.note(locale="en"), "")
        self.assertEqual(self.note(fixed_codes=["IDM_W5_EXPLANATION_LENGTH"], remaining=[]),
                         "Đã tự sửa: lời giải quá dài.")
        self.assertNotIn("không phát hiện vấn đề lớn", self.note(notices=["evidence"]))

    def test_an_abbreviation_with_its_full_form_before_a_dash_is_explained(self) -> None:
        for text in ("Nhận diện khoảnh khắc sự thật (Moments of Truth - MOT) quyết định gắn bó.",
                     "Nhận diện khoảnh khắc sự thật (Moments of Truth – MOT) quyết định gắn bó.",  # noqa: RUF001
                     "Moments of Truth - MOT quyết định khách hàng ở lại hay rời đi.",
                     "Khoảnh khắc sự thật (MOT) quyết định khách hàng ở lại.",
                     "MOT (Moments of Truth) quyết định khách hàng ở lại."):
            self.assertTrue(explained("MOT", None, [text]), text)
        for text in ("MOT quyết định khách hàng ở lại.", "Lean - MOT là hai khái niệm.", "Bước 2 - MOT"):
            self.assertFalse(explained("MOT", None, [text]), text)

    def test_the_hype_line_names_the_words_found(self) -> None:
        note = self.note(remaining=[SlotFinding(EXPLANATION_FILLER_CODE, 1), SlotFinding(FAQ_HYPE_CODE, 2)],
                         hype_words={1: ["hoàn hảo"], 2: ["đập tan", "vượt trội"]})
        self.assertIn("Nên biên tập khối 2 (Quiz): lời giải dùng chữ sáo rỗng hoặc thổi phồng («hoàn hảo»)", note)
        self.assertIn("Nên biên tập khối 3 (Hỏi đáp): câu trả lời hỏi đáp dùng chữ thổi phồng («đập tan», «vượt trội»)",
                      note)
        for example in ("sinh tử", "triệt để", "tuyệt đối"):
            self.assertNotIn(example, note)
        bare = self.note(remaining=[SlotFinding(EXPLANATION_FILLER_CODE, 1)])
        self.assertIn("lời giải dùng chữ sáo rỗng hoặc thổi phồng — nên thay", bare)
        english = self.note(locale="en", remaining=[SlotFinding(FAQ_HYPE_CODE, 2)], hype_words={2: ["absolutely"]})
        self.assertIn("FAQ answers use hype words («absolutely»)", english)

    def test_the_writer_reads_the_hype_words_of_each_slot(self) -> None:
        body = escalate_body()
        request = RagLessonAuthorUnitV2Request.model_validate(body)
        writer = IdmUnitWriter(request.unit_contract, parse_brief(request.unit_contract),
                               unit_service._idm_unit_deps(request), make_runtime(None))
        components = escalate_writer(body)["components"]
        question = {**components["c0"], "type": "problem", "explanation": (
            "Nguyên tắc: an toàn là tiêu chí bắt buộc.\nA - Sai vì hứa hoàn tiền trước.\nB - Đúng vì giải pháp này ăn "
            "khớp hoàn hảo với tiêu chí an toàn.\nC - Sai vì khách VIP không đổi tiêu chí.")}
        faq = {**components["c1"], "type": "la_faq"}
        self.assertEqual(hype_words([question, faq], (), writer.evidence, writer.faq), {0: ["hoàn hảo"]})
        self.assertEqual(hype_words([question, faq], (0,), writer.evidence, writer.faq), {})

    def test_every_contract_code_has_plain_words(self) -> None:
        codes: set[str] = set()
        for path in ("app/services/staged_unit/validation.py", "app/services/staged_unit/provider_schemas.py",
                     "app/idm/node_acceptance.py", "app/idm/node_acceptance_html.py",
                     "app/idm/node_acceptance_problems.py", "app/idm/node_acceptance_widgets.py"):
            source = (REPO_ROOT / path).read_text(encoding="utf-8")
            codes |= set(re.findall(r'"([A-Z][A-Z0-9]*_[A-Z0-9_]{2,})"', source))
        generic = sorted(code for code in codes - {"WORKSPACE_"}
                         if _code_text(code, True) == "một kiểm tra tự động khác")
        self.assertEqual(generic, [])
        self.assertEqual(_code_text("PROBLEM_CHOICE_INVALID", True),
                         "một phương án trả lời quá ngắn hoặc thiếu nội dung")
        self.assertEqual(_code_text("PROBLEM_CHOICE_INVALID", False), "an answer option that is too short or empty")
        # An unknown code still reads as before.
        self.assertEqual(_code_text("SOMETHING_ELSE", True), "một kiểm tra tự động khác")


class ImageCoverPageTests(unittest.TestCase):
    """A4: the image-page warning never fires for a section cover (a title over a picture: pages 12, 45) and still
    fires for pages whose picture holds the content (36, 39, and 29 where the value curve sits beside one sentence)."""

    TEXTS: ClassVar[dict[int, list[str]]] = {
        12: ["Row 1: | MODULE 1: THẤU HIỂU HIỆN TRẠNG",
             "Góc nhìn Outside – In: Khách hàng thực sự đang trải qua điều gì?"],  # noqa: RUF001
        45: ["Row 1: | BÀI TẬP VỀ NHÀ"],
        39: ["2.4 MA TRẬN ƯU TIÊN HÀNH ĐỘNG"],
        36: ["2.3 LỘ TRÌNH TRẢI NGHIỆM KHÁCH HÀNG (WORKBOOK)", "1", "2", "3", "4"],
        29: ["1.5 THIẾT KẾ ĐƯỜNG CONG GIÁ TRỊ - VALUE CURVE", "Thiết kế trải nghiệm xuất sắc không phải là tăng cường "
             "mọi thứ, mà là tập trung nguồn lực vào điểm tạo ra khác biệt."],
        14: ["1.1. PHÂN NHÓM KHÁCH HÀNG (SEGMENTATION)", "Phân khúc là quá trình gom các khách hàng có đặc điểm chung.",
             "Quy tắc MECE giúp các nhóm không trùng nhau và không bỏ sót ai.",
             "Strategic Targeting chọn nhóm nào để phục vụ dựa trên giá trị."],
    }

    def regions(self) -> list[dict[str, Any]]:
        shares = {12: [0.95], 45: [0.95], 39: [0.69], 36: [0.6], 29: [0.18, 0.19], 14: [0.35]}
        return [{"visual_regions": [region(page, share, asset=f"{page:02d}{index}" * 21)
                                    for index, share in enumerate(values)]} for page, values in shares.items()]

    def test_covers_never_count_content_pictures_do(self) -> None:
        self.assertEqual(unread_image_pages(self.regions(), self.TEXTS), [29, 36, 39])
        # Without the page texts, the rule of run 5496155c.
        self.assertEqual(unread_image_pages(self.regions()), [12, 36, 39, 45])
        self.assertFalse(evidence_needs_check(self.regions()[:2], (), self.TEXTS))
        self.assertTrue(evidence_needs_check(self.regions()[4:], (), self.TEXTS))
        self.assertEqual(page_texts([(12, "a"), (None, "b"), (12, "c")]), {12: ["a", "c"]})

    def test_the_unit_reads_the_texts_of_its_pages(self) -> None:
        for page, expected in ((45, False), (39, True)):
            body = severity_body()
            facts = body["unit_contract"]["source_facts"]
            for position, fact in enumerate(facts):
                fact["source_page"] = page
                fact["fact_text"] = self.TEXTS[page][0] if position == 0 else "OPEX"
                fact["locator"] = {**(fact.get("locator") or {}),
                                   "visual_regions": [region(page, 0.95 if page == 45 else 0.69)]}
            rehash_contract(body["unit_contract"])
            request = RagLessonAuthorUnitV2Request.model_validate(body)
            with self.subTest(page=page):
                self.assertEqual(unit_service._idm_unit_deps(request).evidence_review_required, expected)


class RepeatedRowLabelGuidanceTests(unittest.TestCase):
    """B5: the writer and the repair are told how to show row labels several sections share; the duplicate rule Python
    and Node share backs the guidance."""

    @staticmethod
    def profiles(labels: list[str]) -> str:
        return "".join(f"<h2>Hồ sơ {name}</h2><table><tbody><tr><th>{label}</th><td>Việc {name} cần làm mỗi "
                       f"ngày.</td></tr></tbody></table>" for name, label in zip(("A", "B"), labels, strict=True))

    def test_repeated_labels_fail_prefixed_labels_pass(self) -> None:
        self.assertEqual(html_quality_code(self.profiles(["Về chức năng công việc", "Về chức năng công việc"])),
                         "HTML_DUPLICATE_BLOCK")
        self.assertIsNone(html_quality_code(self.profiles(["Hồ sơ A - Về chức năng công việc",
                                                           "Hồ sơ B - Về chức năng công việc"])))

    def test_the_writer_and_the_repair_are_told(self) -> None:
        self.assertIn("When several sections share the same\n  row labels (profiles, cases, layers)",
                      prompts_unit._REPEAT_RULE)
        self.assertIn(prompts_unit._REPEAT_RULE, writer_prompt_of(escalate_body()))
        self.assertIn("start it with the section's name", rule_line("c0", "HTML_DUPLICATE_BLOCK", "c0"))


def writer_prompt_of(body: dict[str, Any]) -> str:
    request = RagLessonAuthorUnitV2Request.model_validate(body)
    return IdmUnitWriter(request.unit_contract, parse_brief(request.unit_contract),
                         unit_service._idm_unit_deps(request), make_runtime(None)).writer_prompt()


if __name__ == "__main__":
    unittest.main()
