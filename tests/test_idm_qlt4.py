"""QC run ab8d67e1 (course-v1:nesso+4645+2026, 2026-10-09, 8.2/10) regressions for QLT-4.

* R1: the framework list of QLT-3 Q3 was attached and inserted wrongly: "4 lực đẩy" (c1.l1.u1) got 7 table rows
  "Row 1: TRỤC VẬN HÀNH … Row 8: ĐỈNH CAO" (the PDF label "Row" taken for the stem, the first stem used when none
  named the noun, ">= N" items merged from two tables, an insertion that never counted), and "3 giai đoạn"
  (c5.l1.u2) got the 4 fields of the commitment form.

The facts are those of the run (``fixtures/qlt4_run_ab8d67e1_blocks.json``); nothing reaches the network.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

from app.idm.contracts import IdmLessonDesignV1, IdmLessonPlanV1
from app.idm.framework import (
    build_promise,
    framework_listed,
    has_list_of,
    stem_names_noun,
    unit_framework_brief,
    with_framework_list,
    without_row_label,
)
from app.idm.module_design import attach_framework_items
from app.idm.module_layout import ModuleScope
from tests import idm_golden_module as gm
from tests.test_idm_qlt3 import block, row

RUN = json.loads((Path(__file__).parent / "fixtures" / "qlt4_run_ab8d67e1_blocks.json").read_text("utf-8"))
CHAPTERS: dict[str, dict[str, Any]] = RUN["chapters"]


def groups(chapter: str, block_ids: list[str] | None = None) -> list[list[str]]:
    """The facts of each block of a chapter of the run (all its blocks by default), one list per block."""

    blocks = CHAPTERS[chapter]
    return [list(blocks[block_id]["facts"].values()) for block_id in block_ids or list(blocks)]


def chapter_scope(chapter: str) -> ModuleScope:
    """One lesson plan over every block of the chapter: what W4 sees as the shard."""

    blocks = CHAPTERS[chapter]
    plan = IdmLessonPlanV1.model_validate({
        "lesson_key": "lsn_001", "kind": "learning", "title": "Bài 1", "primary_must_do_id": None,
        "secondary_must_do_ids": [], "block_ids": list(blocks), "est_screens": 4, "est_minutes": 10,
        "ordering_rationale": "Theo tài liệu."})
    return ModuleScope(
        blocks={block_id: block(block_id, list(item["facts"])) for block_id, item in blocks.items()},
        rows={block_id: row(block_id, ["md_1"], "must_know") for block_id in blocks},
        scope_of_block={block_id: "idmcb_" + "c" * 32 for block_id in blocks},
        fact_text={key: text for item in blocks.values() for key, text in item["facts"].items()},
        lesson_plans=[plan], must_do_statement={}, locale="vi")


def teaching_lesson(unit_title: str, html_title: str, block_ids: list[str]) -> IdmLessonDesignV1:
    return IdmLessonDesignV1.model_validate(gm._lesson("lsn_001", "Bài 1", "Mục tiêu bài", [], [
        gm._unit(1, "context_explain", unit_title, block_ids, [
            gm._component(1, "html", "explain", html_title, block_ids)])]))


def html_with(*blocks: dict[str, Any]) -> dict[str, Any]:
    return {"type": "html", "semantic_content": {"version": 2, "sections": [
        {"heading": "Bối cảnh là gì?", "learning_block_ids": [], "blocks": list(blocks)}]}}


FORCES = ["Bước nhảy vọt AI & Digital: công nghệ định hình lại cấu trúc chi phí.",
          "VUCA & BANI: môi trường siêu biến động.", "Hành vi khách hàng mới: yêu cầu phản hồi tức thì.",
          "Cạnh tranh phi truyền thống: đối thủ từ nền tảng xuyên biên giới."]


class FrameworkListTests(unittest.TestCase):
    """R1: no "Row" stem, a stem that names the promised noun, exactly N items of one table (or one heading per
    block), and no insertion into an html that already lists N entries."""

    def test_the_four_forces_get_no_list_of_table_rows(self) -> None:
        # c1.l1.u1: the unit owns cb_0004 (the 4 forces, unnumbered) and cb_0008. Before: "Liệt kê đủ 4 lực đẩy:
        # Row 1: TRỤC VẬN HÀNH; …; Row 8: ĐỈNH CAO" from cb_0006 and cb_0017.
        titles = ["4 lực đẩy thị trường và bản chất nền tảng của Best-in-Class",
                  "Bối cảnh 4 lực đẩy tái cấu trúc và công thức cốt lõi BiC"]
        self.assertIsNone(unit_framework_brief(titles, groups("chapter-1", ["cb_0004", "cb_0008"]),
                                               groups("chapter-1"), "vi"))
        lessons, attached = attach_framework_items([teaching_lesson(*titles, ["cb_0004", "cb_0008"])],
                                                   chapter_scope("chapter-1"))
        self.assertEqual((attached, lessons[0].units[0].components[0].support_items), (0, []))
        # W5: no numbered items, so the writer's list of the 4 forces is the overview.
        promise = build_promise(titles, groups("chapter-1", ["cb_0004", "cb_0008"]))
        assert promise is not None
        self.assertEqual((promise.count, promise.items, promise.labels), (4, (), ()))
        self.assertTrue(framework_listed(promise, [html_with({"kind": "bullets", "items": FORCES})], ""))

    def test_three_stages_get_no_fields_of_the_commitment_form(self) -> None:
        # c5.l1.u2: "30 Ngày: SEE DIFFERENT" puts the number first; the only numbered stem of the chapter was
        # "Row 1-4" of the commitment form (cb_0032), which the writer then wrote up as "4 yếu tố" of each stage.
        titles = ["Phân bổ mục tiêu Quick-Win qua 3 giai đoạn CEO 90-Day Challenge",
                  "Cấu trúc 3 giai đoạn: See Different - Think Different - Act Different"]
        self.assertIsNone(unit_framework_brief(titles, groups("chapter-5", ["cb_0030"]), groups("chapter-5"), "vi"))
        promise = build_promise(titles, groups("chapter-5"))
        assert promise is not None
        self.assertEqual(promise.labels, ())

    def test_the_orientation_unit_of_the_run_still_gets_the_five_shifts(self) -> None:
        # c2.l1.u1 was right in the run: one SHIFT heading per block of chapter 2.
        brief = unit_framework_brief(["Bản đồ 5 chuyển dịch tư duy sống còn của lãnh đạo 5.0"],
                                     groups("chapter-2", ["cb_0019"]), groups("chapter-2"), "vi")
        self.assertEqual(brief, "Liệt kê đủ 5 chuyển dịch: SHIFT 1: THINK BIG; SHIFT 2: THINK DIFFERENT; "
                                "SHIFT 3: THINK CUSTOMER; SHIFT 4: THINK SYSTEM; SHIFT 5: THINK ECOSYSTEM")

    def test_a_table_row_label_is_never_a_stem_and_the_ladder_is_read_whole(self) -> None:
        self.assertEqual(without_row_label("Row 2: Bậc 01 | Vietnam Manufacturing"), "Bậc 01 | Vietnam Manufacturing")
        self.assertFalse(stem_names_noun("row", "lực đẩy"))
        self.assertFalse(stem_names_noun("row", "row"))
        self.assertTrue(stem_names_noun("shift", "chuyển dịch"))
        self.assertTrue(stem_names_noun("bac", "nấc"))
        self.assertTrue(stem_names_noun("mindset shift", "shifts"))
        self.assertFalse(stem_names_noun("buoc", "lực đẩy"))
        promise = build_promise(["6 nấc thang giá trị The Made-in-World Ladder"], groups("chapter-1", ["cb_0017"]))
        assert promise is not None
        self.assertEqual(promise.labels, tuple(f"Bậc {n}: Vietnam {name}" for n, name in enumerate(
            ["Manufacturing", "Quality", "Value", "Know-how", "Innovation", "Brand"], start=1)))

    def test_items_are_never_merged_from_two_tables_or_counted_past_the_promise(self) -> None:
        def labels(fact_groups: list[list[str]]) -> tuple[str, ...]:
            promise = build_promise(["4 bước"], fact_groups)
            assert promise is not None
            return promise.labels

        rows_a = [f"Row {n}: Bước {n} | Việc {n}" for n in range(1, 4)]
        self.assertEqual(labels([rows_a, ["Row 1: Bước 4 | Việc thêm"]]), ())
        # Five numbered steps do not answer a promise of four, nor three of four.
        five = [f"Bước {n}: việc {n}" for n in range(1, 6)]
        self.assertEqual((labels([five]), labels([five[:3]])), ((), ()))
        # One heading per block is a framework taught block by block; a block with two of them is not.
        headed = [[f"Bước {n}: việc {n}", "Giải thích"] for n in range(1, 5)]
        self.assertEqual(len(labels(headed)), 4)
        self.assertEqual(labels([*headed[:2], [*headed[2], *headed[3]]]), ())

    def test_nothing_is_inserted_into_an_html_that_already_lists_the_items(self) -> None:
        promise = build_promise(["4 bước"], [[f"Bước {n}: việc {n}" for n in range(1, 5)]])
        assert promise is not None
        listed = html_with({"kind": "paragraph", "text": "Bốn bước như sau."}, {"kind": "steps", "items": FORCES})
        self.assertTrue(has_list_of(listed, 4))
        self.assertIsNone(with_framework_list(listed, promise, "vi"))
        prose = html_with({"kind": "paragraph", "text": "Bốn bước như sau."})
        self.assertFalse(has_list_of(prose, 4))
        inserted = with_framework_list(prose, promise, "vi")
        assert inserted is not None
        self.assertEqual(inserted["semantic_content"]["sections"][1]["blocks"][0]["items"],
                         [f"Bước {n}: việc {n}" for n in range(1, 5)])
