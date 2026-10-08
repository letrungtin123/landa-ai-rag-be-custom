"""QC D9 / QLT-1 R2: multi-column PDF pages are read column by column (``AI_RAG_PDF_COLUMN_AWARE``).

Pure geometry cases for ``column_reading_order`` plus synthetic PDFs generated with PyMuPDF.
"""

from __future__ import annotations

import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pymupdf

from app import main
from app.infra.pdf_layout import Box, baseline_order, column_reading_order


def lines(*rows: tuple[float, float, float, float]) -> list[Box]:
    return list(rows)


class ColumnOrderGeometryTests(unittest.TestCase):
    def test_single_column_pages_keep_the_baseline(self) -> None:
        prose = lines((60, 100, 540, 112), (60, 116, 540, 128), (60, 132, 200, 144), (60, 160, 540, 172))
        self.assertIsNone(column_reading_order(prose))
        # A right-aligned page number or date is not a column (too narrow).
        self.assertIsNone(column_reading_order([*prose, (480, 780, 540, 792)]))
        self.assertIsNone(column_reading_order([]))
        self.assertIsNone(column_reading_order([(10, 10, 10, 20), (10, 30, 10, 40)]))  # zero width

    def test_two_columns_read_left_then_right_with_full_width_blocks_in_place(self) -> None:
        boxes = lines(
            (60, 40, 540, 60),     # 0 full-width title
            (320, 100, 360, 114),  # 1 right header, slightly higher
            (60, 102, 110, 116),   # 2 left header
            (60, 130, 280, 142),   # 3 left body
            (320, 130, 540, 142),  # 4 right body
            (60, 300, 540, 312),   # 5 full-width summary
            (60, 330, 280, 342),   # 6 left after summary
            (320, 330, 540, 342),  # 7 right after summary
        )
        self.assertEqual(baseline_order(boxes), [0, 1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(column_reading_order(boxes), [0, 2, 3, 1, 4, 5, 6, 7])

    def test_stacked_halves_are_not_columns(self) -> None:
        # A right-aligned line above a left-aligned paragraph: no vertical overlap, nothing to reorder.
        self.assertIsNone(column_reading_order(lines((330, 60, 540, 72), (60, 100, 290, 112), (60, 116, 290, 128))))

    def test_three_columns(self) -> None:
        boxes = lines((40, 100, 200, 112), (220, 98, 380, 110), (400, 96, 560, 108),
                      (40, 116, 200, 128), (220, 114, 380, 126), (400, 112, 560, 124))
        self.assertEqual(column_reading_order(boxes), [0, 3, 1, 4, 2, 5])


def make_pdf(path: Path, build: Callable[[Any], None]) -> None:
    document = pymupdf.open()
    build(document.new_page(width=600, height=800))
    document.save(path)
    document.close()


def two_column_page(page: Any) -> None:
    page.insert_text((60, 60), "Comparison of the two states", fontsize=18)
    page.insert_text((320, 118), "After", fontsize=14)  # PyMuPDF merges it with "Before" into one block
    page.insert_text((60, 120), "Before", fontsize=14)
    for number in range(1, 4):
        page.insert_text((60, 132 + 18 * number), f"Old step {number} done by hand", fontsize=11)
        page.insert_text((320, 132 + 18 * number), f"New step {number} done by the tool", fontsize=11)
    page.insert_text((60, 400), "Summary paragraph that spans the whole width of the page for every reader.",
                     fontsize=11)


def one_column_page(page: Any) -> None:
    page.insert_text((60, 60), "Chapter One Overview", fontsize=24)
    for number in range(6):
        page.insert_text((60, 110 + 16 * number), f"Body line {number} with enough words to read like prose.",
                         fontsize=11)
    page.insert_text((60, 230), "Short last line.", fontsize=11)
    page.insert_text((480, 760), "Page 3", fontsize=9)


class PdfExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name)

    def extract(self, build: Callable[[Any], None], *, column_aware: bool) -> list[main.ExtractedSection]:
        path = self.dir / f"page-{column_aware}.pdf"
        make_pdf(path, build)
        with patch.object(main.settings, "pdf_column_aware", column_aware):
            return main.extract_pdf(path)

    def test_one_column_page_is_unchanged_byte_for_byte(self) -> None:
        [aware] = self.extract(one_column_page, column_aware=True)
        [legacy] = self.extract(one_column_page, column_aware=False)
        self.assertEqual((aware.text, aware.metadata), (legacy.text, legacy.metadata))
        self.assertEqual(aware.metadata["reading_order"], "top_to_bottom_left_to_right")

    def test_two_column_page_reads_the_left_column_then_the_right_one(self) -> None:
        [legacy] = self.extract(two_column_page, column_aware=False)
        # The historical order interleaves the columns and puts "After" before "Before".
        self.assertLess(legacy.text.index("After"), legacy.text.index("Before"))
        self.assertLess(legacy.text.index("New step 1"), legacy.text.index("Old step 2"))

        [aware] = self.extract(two_column_page, column_aware=True)
        self.assertEqual(aware.metadata["reading_order"], "columns_left_to_right")
        self.assertEqual(aware.text.split("\n\n"), [
            "Comparison of the two states",
            "Before", "Old step 1 done by hand", "Old step 2 done by hand", "Old step 3 done by hand",
            "After", "New step 1 done by the tool", "New step 2 done by the tool", "New step 3 done by the tool",
            "Summary paragraph that spans the whole width of the page for every reader.",
        ])
        # Headings are taken per column fragment, so the merged block no longer yields "After Before".
        titles = [candidate["title"] for candidate in aware.metadata["heading_candidates"]]
        self.assertEqual(titles, ["Comparison of the two states", "Before", "After"])
        self.assertIn("After Before", [candidate["title"] for candidate in legacy.metadata["heading_candidates"]])

    def test_multi_line_blocks_inside_a_column_keep_their_line_breaks(self) -> None:
        def page(target: Any) -> None:
            target.insert_text((60, 100), "Left first line\nLeft second line\nLeft third line", fontsize=11)
            target.insert_text((330, 98), "Right first line\nRight second line", fontsize=11)

        [aware] = self.extract(page, column_aware=True)
        self.assertEqual(aware.text, "Left first line\nLeft second line\nLeft third line\n\n"
                                     "Right first line\nRight second line")


if __name__ == "__main__":
    unittest.main()
