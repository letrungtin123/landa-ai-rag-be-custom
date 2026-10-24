"""QLT-6 preserve_source: line rules, page furniture, the verbatim renderer and the exact-text check."""

from __future__ import annotations

import unittest
from typing import Any

from app.contracts.orchestration_v2.provider import SourceSnapshotFactV2
from app.idm.policy import IDM_PRESERVE_TABLE_MAX_COLUMNS, IDM_PRESERVE_TABLE_MAX_ROWS
from app.idm.preserve_lines import (
    all_caps,
    expected_tokens,
    heading_hint,
    html_tokens,
    is_render_heading,
    keyword_level,
    numbering_depth,
    page_furniture,
    table_cells,
    title_eligible,
    tokens,
    visible_text,
)
from app.idm.preserve_render import (
    KEPT_SOURCE_MARKER,
    PRESERVE_TEXT_ADDED,
    PRESERVE_TEXT_CHANGED,
    PRESERVE_TEXT_MISSING,
    preserve_text_finding,
    purity_report,
    render_preserved_html,
)
from tests.golden import idm_preserve_golden as pg

NBSP = chr(0xA0)
EN_DASH = chr(0x2013)
IDEOGRAPHIC = chr(0x3000)


def fact(number: int, text: str, page: int | None = 1) -> SourceSnapshotFactV2:
    return SourceSnapshotFactV2(document_id=pg.DOCUMENT_ID, fact_key=f"d1-c1-f{number}", scope_key="scope3_00",
                                fact_text=text, source_ref=None, source_page=page, source_chunk=0, locator={})


def pages(*lines_per_page: list[str]) -> list[SourceSnapshotFactV2]:
    facts: list[SourceSnapshotFactV2] = []
    for page, lines in enumerate(lines_per_page, start=1):
        facts.extend(fact(len(facts) + 1, line, page) for line in lines)
    return facts


class FurnitureTests(unittest.TestCase):
    def test_golden_document_drops_page_numbers_and_repeated_headers_only(self) -> None:
        furniture = page_furniture(pg.source_facts())
        self.assertEqual(sorted(furniture.reasons), sorted([*pg.PAGE_NUMBER_KEYS, *pg.REPEATED_HEADER_KEYS]))
        self.assertEqual((furniture.page_numbers, furniture.repeated_lines), (4, 3))
        self.assertEqual(len(pg.content_texts()), 27)
        # The header's first occurrence stays (a running header often repeats the real heading).
        self.assertIn(pg.HEADER, pg.content_texts())

    def test_page_number_shapes(self) -> None:
        for text in ("7", "- 7 -", f"{EN_DASH} 12 {EN_DASH}", "Trang 3", "trang 3/40", "Page 4 of 10", "12/40",
                     "TRANG 5"):
            self.assertEqual(page_furniture([fact(1, text)]).reasons, {"d1-c1-f1": "page_number"}, text)
        for text in ("2025", "Trang 3: Quy trình", "3. Lập kế hoạch", "Page layout", "Bước 7"):
            self.assertEqual(page_furniture([fact(1, text)]).reasons, {}, text)

    def test_a_repeated_line_needs_three_pages_and_a_page_edge(self) -> None:
        def body(page: int) -> list[str]:
            return [f"Trang {page}, đoạn nội dung số {number}, đủ dài để không phải đầu trang." for number in range(6)]

        two_pages = pages(["Đầu trang", *body(1)], ["Đầu trang", *body(2)])
        self.assertEqual(page_furniture(two_pages).reasons, {})
        three = pages(["Đầu trang", *body(1)], ["Đầu trang", *body(2)], [*body(3), "Đầu trang"])
        self.assertEqual(page_furniture(three).repeated_lines, 2)
        middle = pages([*body(1)[:3], "Giữa trang", *body(1)[3:]], [*body(2)[:3], "Giữa trang", *body(2)[3:]],
                       [*body(3)[:3], "Giữa trang", *body(3)[3:]])
        self.assertEqual(page_furniture(middle).reasons, {}, "a repeated line inside a page is content")

    def test_labels_long_lines_tables_and_pageless_facts_are_never_furniture(self) -> None:
        def body(page: int) -> list[str]:
            return [f"Câu nội dung thứ {number} trong trang {page} của tài liệu." for number in range(5)]

        label = pages(["Lưu ý:", *body(1)], ["Lưu ý:", *body(2)], ["Lưu ý:", *body(3)])
        self.assertEqual(page_furniture(label).reasons, {})
        long_line = ("Dòng này dài hơn một trăm ký tự nên được coi là nội dung của tài liệu chứ không phải đầu trang "
                     "lặp lại.")
        repeated_long = pages([long_line, *body(1)], [long_line, *body(2)], [long_line, *body(3)])
        self.assertEqual(page_furniture(repeated_long).reasons, {})
        table = pages(["Row 1: A | B", *body(1)], ["Row 1: A | B", *body(2)], ["Row 1: A | B", *body(3)])
        self.assertEqual(page_furniture(table).reasons, {})
        pageless = [fact(number, "Đầu trang", None) for number in range(1, 5)]
        self.assertEqual(page_furniture(pageless).reasons, {})


class LineRuleTests(unittest.TestCase):
    def test_visible_text_of_table_rows(self) -> None:
        self.assertEqual(table_cells("Row 2: A | Hằng tuần | 92%"), ["A", "Hằng tuần", "92%"])
        self.assertEqual(table_cells("ROW 3 :  x|y | z"), ["x|y", "z"])
        self.assertEqual(table_cells("Row 1: a " + chr(0xA6) + " b | c"), ["a | b", "c"])
        self.assertIsNone(table_cells("Rows 1: a | b"))
        self.assertEqual(visible_text("Row 1: Nhóm | Tần suất"), "Nhóm Tần suất")
        self.assertEqual(visible_text("Không phải bảng | chỉ là chữ"), "Không phải bảng | chỉ là chữ")
        self.assertEqual(expected_tokens(["Row 1: a | b c", "x  y"]), ["a", "b", "c", "x", "y"])

    def test_tokens_use_the_ecmascript_white_space(self) -> None:
        self.assertEqual(tokens(f"a{NBSP}b　c\tdd"), ["a", "b", "c", "dd"])
        self.assertEqual(tokens(f" {chr(0xFEFF)}x "), ["x"])
        self.assertEqual(tokens("a\x1cb"), ["a\x1cb"], "Python-only white space is not a separator")

    def test_html_tokens_decode_entities_once(self) -> None:
        self.assertEqual(html_tokens("<p>a &amp;lt; b</p><p>&quot;c&quot; &#39;d&#x27; e&nbsp;f</p>"),
                         ["a", "&lt;", "b", '"c"', "'d'", "e", "f"])
        self.assertEqual(html_tokens("<p>&#1114112; &copy; &#65;</p>"), ["&#1114112;", "&copy;", "A"])
        self.assertEqual(html_tokens("<h2>x</h2><table><tr><td>a</td><td>b</td></tr></table>"), ["x", "a", "b"])

    def test_heading_candidates(self) -> None:
        cases: dict[str, int | None] = {
            "Chương 1: Tổng quan": 1, "CHAPTER II Overview": 1, "Phần 3": 1, "Module 4 - Sales": 1,
            "MODUN 1: CHANGE MINDSET": 1, "Học phần 2": 1, "Tuần 1": 1, "Bài 2: Lợi ích": 2, "Lesson 5": 2,
            "Unit 1.2 Basics": 2, "Mục 3 Quy định": 3, "Section 7": 3, "Bài tập về nhà": None,
            "Phần mềm quản lý": None, "Chương trình đào tạo năm nay rất quan trọng.": None,
        }
        for text, level in cases.items():
            self.assertEqual(keyword_level(text), level, text)
        self.assertEqual(numbering_depth("1.2 Vì sao cần phản hồi"), 2)
        self.assertEqual(numbering_depth("1.2.3. Chi tiết"), 3)
        self.assertEqual(numbering_depth("1. Giới thiệu"), 1)
        self.assertEqual(numbering_depth("IV. Kết luận"), 1)
        self.assertIsNone(numbering_depth("1. Mở tài khoản rồi đăng nhập vào hệ thống."))
        self.assertTrue(all_caps("CHANGE MINDSET"))
        self.assertFalse(all_caps("Change Mindset"))
        self.assertTrue(is_render_heading("3.1 Ví dụ áp dụng"))
        self.assertFalse(is_render_heading("1. Mở tài khoản"))

    def test_heading_hints_shown_to_the_model(self) -> None:
        texts = ["KHOÁ HỌC", "Bài 1: Mở đầu", "1.1 Khái niệm", "Đoạn văn dài " * 10, "Ngắn",
                 "Một đoạn văn bản dài " * 5]
        self.assertEqual([heading_hint(texts, index) for index in range(len(texts))],
                         ["H", "H2", "H2", "-", "H", "-"])

    def test_title_eligibility(self) -> None:
        self.assertTrue(title_eligible("Bài 1: Phản hồi là gì"))
        for text in ("AB", "x" * 181, "Bài <1>", "Row 1: a | b", " ".join(["từ"] * 31)):
            self.assertFalse(title_eligible(text), text)


class RendererTests(unittest.TestCase):
    def check(self, title: str, texts: list[str]) -> str:
        html = render_preserved_html(title, texts)
        self.assertIsNone(preserve_text_finding(texts, html))
        return html

    def test_paragraphs_headings_and_lists_keep_every_word_and_marker(self) -> None:
        texts = ["Bài 1: Phản hồi là gì", "MODUN 1", "Một phản hồi tốt có:", "• Cụ thể", "1. Đầu tiên",
                 "a) Thứ hai", "- Gạch đầu dòng", "1.2 Mục con"]
        html = self.check("Bài 1: Phản hồi là gì", texts)
        self.assertEqual(html, "<h2>Bài 1: Phản hồi là gì</h2><h3>MODUN 1</h3><p>Một phản hồi tốt có:</p>"
                               "<p>• Cụ thể</p><p>1. Đầu tiên</p><p>a) Thứ hai</p><p>- Gạch đầu dòng</p>"
                               "<h3>1.2 Mục con</h3>")
        self.assertNotIn("<ul>", html)
        self.assertNotIn("<ol>", html)

    def test_a_continued_unit_has_no_title_heading(self) -> None:
        html = self.check("Bài 1 (tiếp theo)", ["Đoạn tiếp theo của bài."])
        self.assertEqual(html, "<p>Đoạn tiếp theo của bài.</p>")

    def test_tables_keep_every_cell(self) -> None:
        texts = ["Row 1: Nhóm | Tần suất | Tỷ lệ", "Row 2: A | Hằng tuần", "Row 3: B | Hằng tháng | 74%",
                 "Sau bảng."]
        html = self.check("Bảng", texts)
        self.assertEqual(html, "<table><thead><tr><th>Nhóm</th><th>Tần suất</th><th>Tỷ lệ</th></tr></thead>"
                               "<tbody><tr><td>A</td><td>Hằng tuần</td><td></td></tr><tr><td>B</td><td>Hằng tháng"
                               "</td><td>74%</td></tr></tbody></table><p>Sau bảng.</p>")
        continued = self.check("Bảng", ["Row 7: x | y", "Row 8: z | w"])
        self.assertTrue(continued.startswith("<table><tbody><tr><td>x</td>"), "no header row mid-table")

    def test_table_limits_never_drop_a_cell(self) -> None:
        rows = [f"Row {number}: a{number} | b{number}" for number in range(1, IDM_PRESERVE_TABLE_MAX_ROWS + 6)]
        html = self.check("Bảng", rows)
        self.assertEqual(html.count("<table>"), 2)
        wide = "Row 1: " + " | ".join(f"c{number}" for number in range(IDM_PRESERVE_TABLE_MAX_COLUMNS + 1))
        html = self.check("Bảng", ["Row 1: a | b", wide, "Row 3: c | d"])
        self.assertEqual(html.count("<table>"), 2)
        self.assertIn("<p>c0 c1", html)
        single = self.check("Bảng", ["Row 1: Chỉ một ô", "Row 2: Ô khác"])
        self.assertEqual(single, "<p>Chỉ một ô</p><p>Ô khác</p>")
        # The parser's cell separator is " | ": a row of empty cells keeps its lone bar, the same text on both sides.
        self.assertEqual(self.check("Bảng", ["Row 1:  |  ", "Văn bản"]), "<p>|</p><p>Văn bản</p>")
        self.assertEqual(self.check("Bảng", ["Row 1:   ", "Văn bản"]), "<p>Văn bản</p>")
        self.assertEqual(self.check("Bảng", ["Row 1:", "Văn bản"]), "<p>Row 1:</p><p>Văn bản</p>")

    def test_special_characters_are_escaped_and_read_back(self) -> None:
        texts = ['Dùng thẻ <b> & "ngoặc" với \'nháy\'', f"Khoảng{NBSP}trắng   kép"]
        html = self.check("X", texts)
        self.assertIn("&lt;b&gt; &amp; &quot;ngoặc&quot;", html)
        self.assertNotIn("<b>", html)


class ExactTextTests(unittest.TestCase):
    TEXTS: tuple[str, ...] = ("Bài 1: Mở đầu", "Phản hồi giúp người nhận cải thiện.", "Row 1: A | B")

    def base(self) -> str:
        return render_preserved_html("Bài 1: Mở đầu", list(self.TEXTS))

    def test_the_rendered_unit_passes_and_tag_or_space_changes_do_not_matter(self) -> None:
        html = self.base()
        self.assertIsNone(preserve_text_finding(list(self.TEXTS), html))
        relaxed = html.replace("<p>", "<p>  ").replace("</h2>", "</h2>\n")
        self.assertIsNone(preserve_text_finding(list(self.TEXTS), relaxed))

    def test_any_added_removed_reordered_or_changed_word_is_rejected(self) -> None:
        html = self.base()
        cases: dict[str, tuple[str, str]] = {
            "added": (html.replace("cải thiện.", "cải thiện. Ví dụ minh hoạ."), PRESERVE_TEXT_ADDED),
            "title added": ("<h2>Giới thiệu</h2>" + html, PRESERVE_TEXT_ADDED),
            "removed": (html.replace("<th>B</th>", "<th></th>"), PRESERVE_TEXT_MISSING),
            "reordered": (html.replace("<th>A</th><th>B</th>", "<th>B</th><th>A</th>"), PRESERVE_TEXT_CHANGED),
            "changed": (html.replace("Phản hồi", "Góp ý"), PRESERVE_TEXT_CHANGED),
            "punctuation": (html.replace("cải thiện.", "cải thiện"), PRESERVE_TEXT_CHANGED),
            "case": (html.replace("Mở đầu", "mở đầu"), PRESERVE_TEXT_CHANGED),
        }
        for name, (mutated, code) in cases.items():
            self.assertEqual(preserve_text_finding(list(self.TEXTS), mutated), code, name)


class PurityReportTests(unittest.TestCase):
    def test_kept_text_is_reported_never_changed(self) -> None:
        texts = ["Nguồn: Báo cáo nhân sự 2025, trang 12.", "Xem thêm tài liệu: huong-dan.pdf"]
        html = render_preserved_html("X", texts)
        self.assertIn("trang 12", html)
        self.assertIn("huong-dan.pdf", html)
        report = purity_report(texts, html, ())
        self.assertEqual(report[0], KEPT_SOURCE_MARKER)
        self.assertTrue(report[1].startswith("IDM_PRESERVE_KEPT_HTML_"))

    def test_punctuation_trims_are_not_reported(self) -> None:
        texts = ["Một phản hồi tốt có ba đặc điểm:"]
        self.assertEqual(purity_report(texts, render_preserved_html("X", texts), ()), [])

    def test_golden_units_report_only_the_flagged_lines(self) -> None:
        texts: list[Any] = pg.content_texts()
        self.assertEqual(purity_report(texts[:9], render_preserved_html("Bài 1: Phản hồi là gì", texts[:9]), ()), [])


if __name__ == "__main__":
    unittest.main()
