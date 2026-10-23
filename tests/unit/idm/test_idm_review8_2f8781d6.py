"""Fixes from the independent review 8 of the review 7 fixes (d74c42f..830ef32), run 2f8781d6.

Every case is synthetic text with the shape of the reviewer's reproduction scripts; every provider is a local fake.
"""

from __future__ import annotations

import itertools
import unittest

from app.idm.acronyms import explained
from app.idm.node_acceptance import html_quality_code
from app.idm.quiz_letters import expand_short_item_labels
from app.idm.source_locked import render_idm_source_locked_html
from app.idm.source_names import NameGuide
from app.idm.visual_pages import unread_image_pages
from tests.support.log_support import logged_events
from tests.unit.idm.test_idm_qc_run_2f8781d6 import many_facts_body
from tests.unit.idm.test_idm_qc_run_5496155c import region
from tests.unit.idm.test_idm_review_dfb6db2 import GUIDANCE
from tests.unit.idm.test_idm_storyboard import (
    JUDGE,
    REPAIR,
    WRITER,
    FakeGenerate,
    StoryboardEndpointTestCase,
    judge,
    slot_repair,
    worksheet_body,
    worksheet_writer,
)

# M1: two step sections, each with the same numbered labels ("1. Mục tiêu của bước này"); the renderer lists a
# numbered label with its step title, so a bare label repeated its list item.
STEP_FACTS = ["BƯỚC 1: CHUẨN BỊ", "1. Mục tiêu của bước này", "Thu thập đủ hồ sơ khách hàng trước buổi gặp đầu tiên.",
              "2. Công cụ cần dùng", "Bảng kiểm hồ sơ và mẫu email xác nhận lịch hẹn.", "BƯỚC 2: THỰC HIỆN",
              "1. Mục tiêu của bước này", "Trình bày giải pháp phù hợp với đúng nhu cầu đã ghi nhận.",
              "2. Công cụ cần dùng", "Bộ slide giới thiệu và bảng so sánh phương án."]
PROFILE_FACTS = ["HỒ SƠ KHÁCH HÀNG (VD1: KẾ TOÁN TRƯỞNG)", "1. Mục tiêu công việc",
                 "Đóng sổ sách đúng hạn vào cuối mỗi tháng.", "2. Nỗi lo lớn nhất",
                 "Sai lệch số liệu khi kiểm toán cuối năm.", "HỒ SƠ KHÁCH HÀNG (VD2: GIÁM ĐỐC TÀI CHÍNH)",
                 "1. Mục tiêu công việc", "Phê duyệt khoản đầu tư có hiệu quả rõ ràng.", "2. Nỗi lo lớn nhất",
                 "Dòng tiền âm trong quý đầu tiên."]

# The reviewer's sweep of 3780 layouts around a repeated label under its section title (p1_fuzz.py): no layout may
# repeat a paragraph, as none did at 345663c.
_NAMES = ("KẾ TOÁN TRƯỞNG", "GIÁM ĐỐC TÀI CHÍNH", "NHÂN VIÊN KHO")
_TITLES = ("HỒ SƠ KHÁCH HÀNG (VD{i}: {n})", "BƯỚC {i}: {n}", "{i}. {n}", "PHẦN {i} - {n}", "1.{i} {n}", "{n}")
_TAILS = (None, "Đây là câu kết của phần {i} với đủ chữ để thành đoạn văn.", "8{i}%", "Ghi chú {i}",
          "- Ý phụ thứ {i} của phần trước.", "Row 1: Cột A{i} | Cột B{i}", "{i}. Gọi lại cho khách trong ngày thứ {i}.",
          "Ghi chú (phần {i}", "Lưu ý: kiểm tra lại số liệu phần {i} trước khi gửi.")
_LABELS = ("Mục tiêu công việc", "1. Mục tiêu công việc", "Mục tiêu (Job Goal)", "Bước 1: Mục tiêu công việc",
           "Mục tiêu công việc: hoàn thành báo cáo {i} đúng hạn.")
_FOLLOWING = ("Hoàn thành công việc số {i} đúng hạn và đúng số lượng mỗi ngày.", "- Hoàn thành việc số {i} đúng hạn.",
              "1. Hoàn thành việc số {i} đúng hạn.", "Row 1: Việc {i} | Hạn {i}", "Thách thức số {i}", "7{i}%",
              "2. Hoàn thành việc số {i} đúng hạn.")


def sweep_layouts() -> list[tuple[str, list[str], bool]]:
    """Every (name, facts, ordered_list) of the reviewer's sweep."""

    layouts = []
    for title, tail, label, following, ordered in itertools.product(_TITLES, _TAILS, _LABELS, _FOLLOWING,
                                                                     (False, True)):
        facts: list[str] = []
        for i, name in enumerate(_NAMES, 1):
            if tail is not None and i > 1:
                facts.append(tail.format(i=i))
            facts += [title.format(i=i, n=name), label.format(i=i), following.format(i=i)]
        layouts.append((f"{title}|{tail}|{label}|{following}|{ordered}", facts, ordered))
    return layouts


class NumberedLabelTests(StoryboardEndpointTestCase):
    """M1: a numbered label keeps its section name, since the renderer may list it with the title."""

    def test_numbered_labels_never_repeat(self) -> None:
        for facts in (STEP_FACTS, PROFILE_FACTS):
            for required in ([], [{"type": "ordered_list"}]):
                html = render_idm_source_locked_html("Hồ sơ", facts, locale="vi", required_artifacts=required)
                self.assertIsNone(html_quality_code(html), (facts[0], required))
        html = render_idm_source_locked_html("Hồ sơ", STEP_FACTS, locale="vi")
        self.assertIn("Thực hiện – 1. Mục tiêu của bước này", html)  # noqa: RUF001 - the en dash

    def test_the_layout_sweep_never_repeats_a_paragraph(self) -> None:
        layouts = sweep_layouts()
        self.assertEqual(len(layouts), 3780)
        failing = [name for name, facts, ordered in layouts if html_quality_code(render_idm_source_locked_html(
            "Hồ sơ", facts, locale="vi", required_artifacts=[{"type": "ordered_list"}] if ordered else []))]
        self.assertEqual(failing, [])

    async def test_fallback_only_answers(self) -> None:
        for facts in (STEP_FACTS, PROFILE_FACTS):
            status, data, provider = await self.post({**many_facts_body(facts), "fallback_only": True})
            self.assertEqual((status, provider.calls), (200, []), data.get("detail"))
            self.assertEqual(data["content_origin"], "structured_fallback")


def labelled_quiz(label: str, explanation: str, hints: list[str] | None = None) -> dict[str, object]:
    items = "\n".join(f"- {label} {number}: {text}" for number, text in enumerate(
        ("Gọi lại cho khách trong ngày.", "Gửi thư xin lỗi theo mẫu chuẩn.", "Chuyển ngay cho quản lý cấp cao.",
         "Hẹn khách mang máy đến cửa hàng."), 1))
    return {"problem_type": "multiple_choice",
            "question": f"Giả sử khách báo máy sấy bốc khói. Bạn cân nhắc bốn ý:\n{items}\nPhương án nào đúng?",
            "choices": [{"text": f"{label} {number}", "correct": number == 3} for number in (3, 1, 2, 4)],
            "explanation": explanation, "hints": hints or []}


class LabelWordTests(unittest.TestCase):
    """L1: a word that only shares the label ("lưu ý 2", "Chú ý 2", "LƯU Ý 2", "5 Bước 3 phút") never changes."""

    def test_lower_case_stem_label(self) -> None:
        quiz = labelled_quiz("ý", "Nguyên tắc: cần lưu ý 2 điều: an toàn trước.\nA - Đúng vì ý 3 an toàn.\nB - Sai.\n"
                                  "C - Sai.\nD - Sai.", ["Chú ý 2 điều trước khi chọn."])
        out, _count = expand_short_item_labels(quiz)
        self.assertEqual(out["explanation"].splitlines()[:2],
                         ["Nguyên tắc: cần lưu ý 2 điều: an toàn trước.", "A - Đúng vì ý kiến 3 an toàn."])
        self.assertEqual(out["hints"], ["Chú ý 2 điều trước khi chọn."])
        listed = labelled_quiz("mục", "Nguyên tắc: xem danh mục 2 trong sổ tay.\nA - Đúng vì mục 3 an toàn.\nB - Sai.\n"
                                      "C - Sai.\nD - Sai.")
        self.assertEqual(expand_short_item_labels(listed)[0]["explanation"].splitlines()[:2],
                         ["Nguyên tắc: xem danh mục 2 trong sổ tay.", "A - Đúng vì mục số 3 an toàn."])

    def test_capitals_and_counts_stay(self) -> None:
        quiz = labelled_quiz("Ý", "Nguyên tắc: LƯU Ý 2 ĐIỀU QUAN TRỌNG.\nA - Đúng vì Ý 3 an toàn.\nB - Sai.\nC - Sai.\n"
                                  "D - Sai.")
        self.assertEqual(expand_short_item_labels(quiz)[0]["explanation"].splitlines()[:2],
                         ["Nguyên tắc: LƯU Ý 2 ĐIỀU QUAN TRỌNG.", "A - Đúng vì Ý kiến 3 an toàn."])
        steps = labelled_quiz("Bước", "Nguyên tắc: Bước 1 luôn trước Bước 2; quy trình 5 Bước 3 phút.\nA - Đúng.\n"
                                      "B - Sai.\nC - Sai.\nD - Sai.")
        self.assertEqual(expand_short_item_labels(steps)[0]["explanation"].splitlines()[0],
                         "Nguyên tắc: Bước số 1 luôn trước Bước số 2; quy trình 5 Bước 3 phút.")


class EnglishHeadingTests(unittest.TestCase):
    """L2: a Title Case English heading with a subtitle, and an English gloss of a Vietnamese name the source writes,
    name no tool; a name at the start of a sentence still does."""

    def test_headings_and_glosses(self) -> None:
        guide = NameGuide.build(["Ma trận ưu tiên giúp sắp xếp công việc.", "Bản đồ thấu cảm gồm bốn ô.",
                                 "Ma trận SWOT giúp …"])
        for text in ("Building Your Prioritization Matrix: a quick guide",
                     "Building Your Prioritization Matrix: hướng dẫn từng bước", "Bản đồ thấu cảm (Empathy Map)",
                     "Ma trận ưu tiên (Priority Matrix)", "Bản Đồ Thấu Cảm (Empathy Map)"):
            self.assertEqual(guide.missing(text), [], text)
        for text, found in (("Empathy Map giúp gì cho bạn?", ["Empathy Map"]),
                            ("Eisenhower Matrix: xếp việc theo mức gấp", ["Eisenhower Matrix"]),
                            ("Hãy dùng Eisenhower Matrix để xếp việc", ["Eisenhower Matrix"]),
                            ("Bản đồ thấu cảm (Eisenhower Matrix)", ["Eisenhower Matrix"]),
                            ("Ma trận thấu cảm (Empathy Matrix)", ["Ma trận thấu cảm", "Empathy Matrix"])):
            self.assertEqual(guide.missing(text), found, text)


class TwoWordFullFormTests(unittest.TestCase):
    """L3: "Customer Experience - CX" (the established X of "Experience") and the facts' own two-word pairing
    ("Tiếp thị - MKT") explain the abbreviation again; a two-word tail of another phrase still does not."""

    def test_two_word_full_forms(self) -> None:
        for unit, source, token in (
                ("Customer Experience - CX là trọng tâm.", "Customer Experience (CX) matters.", "CX"),
                ("We track Customer Experience - CX weekly.", "", "CX"),
                ("Tiếp thị - MKT phối hợp.", "Phòng Tiếp thị (MKT) phụ trách.", "MKT")):
            self.assertTrue(explained(token, None, [unit], source_texts=[source] if source else []), unit)
        source = ["Trải nghiệm khách hàng (CX) là cách khách cảm nhận doanh nghiệp."]
        self.assertFalse(explained("CX", None, ["Chăm sóc khách hàng - CX là trọng tâm."], source_texts=source))
        self.assertFalse(explained("CX", None, ["Customer Journey - CX là trọng tâm."]))


class BreadcrumbLastPartTests(unittest.TestCase):
    """L4: only the last part of a breadcrumb scope title starts the page's section, so a picture page inside a module
    keeps its warning."""

    def test_a_page_inside_a_module_warns(self) -> None:
        texts = {44: ["MODULE 2: CHIẾN LƯỢC"]}
        crumb = f" {chr(0x203A)} "
        for scope, warned in ((crumb.join(("Khóa học Trải nghiệm khách hàng", "MODULE 2: CHIẾN LƯỢC",
                                           "2.1 Phân tích SWOT")), [44]),
                              (crumb.join(("MODULE 2: CHIẾN LƯỢC", "2.1 Phân tích SWOT")), [44]),
                              (crumb.join(("Khóa học Trải nghiệm khách hàng", "MODULE 2: CHIẾN LƯỢC")), [])):
            locator = {"scope_title": scope, "visual_regions": [region(44, 0.92, asset="44a" * 21)]}
            self.assertEqual(unread_image_pages([locator], texts), warned, scope)


class DroppedCompactionTests(StoryboardEndpointTestCase):
    """L5: a judge repair the guard compacted and the second judge does not keep leaves no compaction line."""

    async def run_case(self, second: dict[str, object]) -> tuple[list[str], str]:
        body = worksheet_body()
        writer = worksheet_writer(body)
        repaired = worksheet_writer(body)
        for row in repaired["components"]["c0"]["semantic_content"]["sections"][1]["blocks"][0]["rows"]:
            row["value"] += GUIDANCE * 4
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [slot_repair(repaired, 0)],
                                   JUDGE: [judge(("Q7_cognitive_load", "major", 0)), second]})
        with self.assertLogs("app.idm", level="INFO") as logs:
            status, data, _ = await self.post(body, provider, mode="repair")
        self.assertEqual(status, 200, data.get("detail"))
        return (logged_events(logs.records, "idm_unit_quality")[-1]["codes"],
                data["unit"]["idm_quality"]["author_note"])

    async def test_a_dropped_repair_leaves_no_compaction_line(self) -> None:
        codes, note = await self.run_case(judge(("Q7_cognitive_load", "major", 0)))
        self.assertNotIn("IDM_W5_WORKSHEET_COMPACTED", codes)
        self.assertNotIn("rút gọn", note)
        codes, note = await self.run_case(judge())
        self.assertIn("IDM_W5_WORKSHEET_COMPACTED", codes)
        self.assertIn("rút gọn", note)


if __name__ == "__main__":
    unittest.main()
