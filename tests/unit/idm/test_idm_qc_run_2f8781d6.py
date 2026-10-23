"""Fixes from the QC report of the real AI ID run 2f8781d6 (AI 7f67137, 2026-10-23, 7,1/10), wave A.

Every case below is synthetic text with the shape of a defect of that run (no course text of the run); every provider
is a local fake.
"""

from __future__ import annotations

import html as html_lib
import re
import unittest
from typing import Any

from app.content.instructional_quality import clean_source_facts
from app.idm.fallback_lines import restore_stops, short_name, split_glued, tidy_facts
from app.idm.node_acceptance import html_quality_code
from app.idm.source_fallback import fallback_glued_words
from app.idm.source_locked import render_idm_source_locked_html
from app.schemas.orchestration_v2 import RagLessonAuthorUnitV2Request
from app.services.orchestration_v2 import unit as unit_service
from app.services.orchestration_v2.source_locked import render_source_locked_html
from tests.unit.idm.test_idm_fallback_matrix import rehashed, unit_findings
from tests.unit.idm.test_idm_storyboard import StoryboardEndpointTestCase, body_for

HEADER = "Row 1: VIỆC CẦN LÀM - Mục tiêu | NỖI ĐAU - Điều cản trở | LỢI ÍCH - Điều mong đợi"
FUNCTIONAL_JOB = "Về Chức năng (Functional Job)"
EMOTIONAL_JOB = "Về Cảm xúc (Emotional Job)"
SOCIAL_PAIN = "Về Ý nghĩa / Xã hội (Social/Meaningful Pain)"
EMOTIONAL_GAIN = "Về Cảm xúc (Emotional Gain)"
# Three profiles with the shapes of pages 24-26 of run 2f8781d6: the first one written inline with a label the PDF cut
# ("(Social/Meaningful" + "Job): *") and the ends of two more labels whose starts the PDF reader dropped as repeats
# ("Pain): *", "Gain): *"); the second and third ones with the same labels on their own lines.
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
    ("Được đồng nghiệp nể trọng", "Về Ý nghĩa / Xã hội (Social/Meaningful Gain)"),
    ("Phê duyệt khoản đầu tư", FUNCTIONAL_JOB),
    ("Muốn chắc chắn họ phải", EMOTIONAL_JOB),
    ("Sợ bị đánh giá là người quản lý", SOCIAL_PAIN),
    ("An tâm khi đội ngũ", EMOTIONAL_GAIN),
    ("Hoàn thành chỉ tiêu xuất kho", f"Nhân viên kho – {FUNCTIONAL_JOB}"),  # noqa: RUF001 - the en dash
    ("Muốn làm việc thoải mái", f"Nhân viên kho – {EMOTIONAL_JOB}"),  # noqa: RUF001 - the en dash
    ("Sợ bị mang tiếng chậm chạp", f"Nhân viên kho – {SOCIAL_PAIN}"),  # noqa: RUF001 - the en dash
    ("Nhẹ nhõm khi công việc", f"Nhân viên kho – {EMOTIONAL_GAIN}"),  # noqa: RUF001 - the en dash
]
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
        self.assertNotIn(f"Giám đốc tài chính – {EMOTIONAL_JOB}", html)  # noqa: RUF001 - the en dash
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
        self.assertEqual(short_name("BƯỚC 2: XỬ LÝ KHIẾU NẠI VIP"), "Bước 2: xử lý khiếu nại VIP")
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


if __name__ == "__main__":
    unittest.main()
