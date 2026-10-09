"""QC run ab8d67e1 (course-v1:nesso+4645+2026, 2026-10-09, 8.2/10) regressions for QLT-4.

* R1: the framework list of QLT-3 Q3 was attached and inserted wrongly: "4 lực đẩy" (c1.l1.u1) got 7 table rows
  "Row 1: TRỤC VẬN HÀNH … Row 8: ĐỈNH CAO" (the PDF label "Row" taken for the stem, the first stem used when none
  named the noun, ">= N" items merged from two tables, an insertion that never counted), and "3 giai đoạn"
  (c5.l1.u2) got the 4 fields of the commitment form.

The facts are those of the run (``fixtures/qlt4_run_ab8d67e1_blocks.json``); nothing reaches the network.
"""

from __future__ import annotations

import dataclasses
import json
import unittest
from pathlib import Path
from typing import Any

from app.idm.answer_checks import UnitTeaching, contested_option, quoted_key, unit_teaching
from app.idm.budget import judge_repair_fits, repair_thinking_for, writer_thinking
from app.idm.contracts import IdmLessonDesignV1, IdmLessonPlanV1, IdmUnitQualityV1
from app.idm.framework import (
    build_promise,
    framework_listed,
    has_list_of,
    stem_names_noun,
    unit_framework_brief,
    with_framework_list,
    without_row_label,
)
from app.idm.html_rules import visible_text
from app.idm.module_autofix import (
    COMPONENT_ORDER_CODE,
    QUESTION_ONLY_BLOCK_CODE,
    TAUGHT_BEFORE_QUESTION_CODE,
    autofix_lesson,
    teach_before_questions,
)
from app.idm.module_design import attach_framework_items, validate_lesson
from app.idm.module_layout import ModuleScope
from app.idm.prompts import judge_prompt
from app.idm.qa import copied_options, reworded_key
from app.idm.runtime import IdmRuntime
from app.idm.storyboard import run_idm_unit
from app.idm.validation import errors
from app.schemas.orchestration_v2 import RagLessonAuthorUnitV2Request
from app.services.orchestration_v2 import unit as unit_service
from tests import idm_golden_module as gm
from tests.test_idm_module_design import ALLOWED, component, design_module, edited, scope_of
from tests.test_idm_qlt3 import block, row
from tests.test_idm_storyboard import (
    JUDGE,
    REPAIR,
    WRITER,
    FakeGenerate,
    StoryboardEndpointTestCase,
    judge,
    severity_body,
    severity_writer,
    slot_repair,
    worksheet_body,
    worksheet_writer,
)

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


# --- R7: a teaching html next to the worksheet html no longer costs a W4 repair call -------------------------------
def worksheet_beside_teaching(payload: dict[str, Any]) -> None:
    """lsn_002 (one block, cb_0005) written as the 4 lessons of chapters 2 and 5 of the run: a teaching html and the
    worksheet html (role practice) in the one unit, then the question that checks it and the FAQ."""

    teaching = component("html", "explain", ["cb_0005"])
    teaching["support_items"] = [{"kind": "explain_concept", "brief": "Ba cấp độ và dấu hiệu.", "block_id": "cb_0005"}]
    payload["lessons"][1]["units"] = [gm._unit(1, "practice_feedback", "Đánh giá mức độ", ["cb_0005"], [
        teaching, component("html", "practice", ["cb_0005"], practice_id="pt_1"),
        component("problem", "practice", ["cb_0005"], practice_id="pt_1"),
        component("la_faq", "clarify", ["cb_0005"])])]


class ComponentOrderAutofixTests(unittest.IsolatedAsyncioTestCase):
    def test_the_worksheet_takes_over_the_teaching_html_of_its_unit(self) -> None:
        scope = scope_of(0)
        lesson = IdmLessonDesignV1.model_validate(edited("mod_01", worksheet_beside_teaching)["lessons"][1])
        plan = scope.lesson_plans[1]
        self.assertEqual({issue.code for issue in errors(validate_lesson(lesson, plan, scope, ALLOWED,
                                                                         "lessons[1]"))}, {"IDM_W4_COMPONENT_ORDER"})
        fixed, codes = autofix_lesson(lesson, plan, scope, ALLOWED)
        self.assertEqual(codes, {COMPONENT_ORDER_CODE: 1})
        self.assertEqual(errors(validate_lesson(fixed, plan, scope, ALLOWED, "lessons[1]")), [])
        components = fixed.units[0].components
        self.assertEqual([(c.type, c.role, c.practice_id, c.component_index) for c in components],
                         [("html", "practice", "pt_1", 1), ("problem", "practice", "pt_1", 2),
                          ("la_faq", "clarify", None, 3)])
        # What the teaching html carried reaches the worksheet's brief.
        self.assertEqual([item.brief for item in components[0].support_items], ["Ba cấp độ và dấu hiệu."])

    async def test_no_repair_call_is_needed_and_two_practices_of_one_type_still_get_one(self) -> None:
        with self.assertLogs("app.idm", level="INFO") as logs:
            result, provider, runtime = await design_module(0, [edited("mod_01", worksheet_beside_teaching)])
        self.assertEqual((len(provider.calls), result["shard"]["idm_design"]["stage_origin"]), (1, "provider"))
        self.assertEqual(runtime.adjustments[COMPONENT_ORDER_CODE], 1)
        self.assertNotIn("component_orders\": {\"", "".join(logs.output))

        def two_questions(payload: dict[str, Any]) -> None:
            worksheet_beside_teaching(payload)
            components = payload["lessons"][1]["units"][0]["components"]
            components[0] = component("problem", "practice", ["cb_0005"], practice_id="pt_2")
            payload["lessons"][1]["practice_tasks"].append(
                {**payload["lessons"][1]["practice_tasks"][0], "practice_id": "pt_2"})

        with self.assertLogs("app.idm", level="INFO") as logs:
            result, provider, _ = await design_module(
                0, [edited("mod_01", two_questions), gm.MODULE_RESPONSES["mod_01"]])
        self.assertEqual(len(provider.calls), 2)
        attempts = [json.loads(line.split("lesson_author_idm ", 1)[1]) for line in logs.output
                    if "idm_module_attempt" in line]
        # The failing unit's component types are logged (server enum values only).
        self.assertEqual(attempts[0]["component_orders"], {"lsn_002": ["units[0]=problem,html,problem,la_faq"]})


# --- R4: facts a question tests are taught before it -----------------------------------------------------------------
class TaughtBeforeQuestionTests(unittest.IsolatedAsyncioTestCase):
    def test_blocks_only_the_question_used_join_the_teaching_html(self) -> None:
        # c1.l1.u2 of the run: the html taught cb_0013 (the BiC house) and the knowledge check alone listed cb_0014
        # (12 facts "khi móng tư duy đổi / không đổi") and cb_0015 ("90% thất bại"), so Node gave their facts to the
        # question and they appeared only in its explanation.
        scope = chapter_scope("chapter-1")
        lesson = IdmLessonDesignV1.model_validate(gm._lesson("lsn_001", "Bài 1", "Mục tiêu bài", [], [
            gm._unit(1, "context_explain", "Kiến trúc Ngôi nhà BiC và nền móng Change Mindset",
                     ["cb_0013", "cb_0014", "cb_0015"], [
                         gm._component(1, "html", "explain", "Kiến trúc Ngôi nhà Năng lực Best-in-Class", ["cb_0013"]),
                         gm._component(2, "problem", "clarify", "Nguyên nhân gốc rễ của thất bại chuyển đổi",
                                       ["cb_0014", "cb_0015"])])]))
        fixed, attached, untaught = teach_before_questions(lesson, scope.lesson_plans[0])
        self.assertEqual((attached, untaught), (2, 0))
        self.assertEqual([c.block_ids for c in fixed.units[0].components],
                         [["cb_0013", "cb_0014", "cb_0015"], ["cb_0014", "cb_0015"]])
        # Nothing to do when the html already teaches every block of the question.
        self.assertEqual(teach_before_questions(fixed, scope.lesson_plans[0])[1:], (0, 0))

    async def test_w4_teaches_before_testing_and_notes_what_it_cannot(self) -> None:
        def question_block(payload: dict[str, Any]) -> None:
            payload["lessons"][0]["units"][0]["components"][0]["block_ids"] = ["cb_0003"]

        result, provider, runtime = await design_module(0, [edited("mod_01", question_block)])
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(runtime.adjustments[TAUGHT_BEFORE_QUESTION_CODE], 1)
        unit = result["shard"]["idm_design"]["lessons"][0]["units"][0]
        self.assertEqual(unit["components"][0]["block_ids"], ["cb_0003", "cb_0004"])
        # Node gives each fact to the first component listing its block: the html (first) now lists both.
        plans = result["shard"]["lessons"][0]["units"][0]["component_plan"]
        self.assertEqual(plans[0]["source_scope_ids"], plans[1]["source_scope_ids"])

        def question_alone(payload: dict[str, Any]) -> None:
            payload["lessons"][0]["units"][1]["components"].pop()  # the FAQ that also presented cb_0011

        with self.assertLogs("app.idm", level="INFO") as logs:
            result, _, _ = await design_module(1, [edited("mod_02", question_alone)])
        lesson = result["shard"]["idm_design"]["lessons"][0]
        self.assertIn("chỉ xuất hiện trong lời giải của câu hỏi", lesson["notes"])
        completed = next(json.loads(line.split("lesson_author_idm ", 1)[1]) for line in logs.output
                         if "idm_stage_completed" in line)
        self.assertEqual(completed["error_codes"].get(QUESTION_ONLY_BLOCK_CODE), 1)


# --- R3: writer, repair and judge fit the 120 s unit request ---------------------------------------------------------
class ClockedGenerate(FakeGenerate):
    """The fake provider, each call taking the given seconds of a fake clock (writer, repair, judge)."""

    def __init__(self, seconds: dict[str, float], **queues: list[Any]) -> None:
        super().__init__(**queues)
        self.seconds = seconds
        self.now = 1_000.0

    def clock(self) -> float:
        return self.now

    async def __call__(self, api_key: str, model: str, prompt: str, **options: Any) -> tuple[str, Any]:
        answer = await super().__call__(api_key, model, prompt, **options)
        self.now += self.seconds[self.calls[-1]["family"]]
        return answer


def leaking_worksheet(body: dict[str, Any]) -> dict[str, Any]:
    """The worksheet unit with a key that restates the worked example above it (as c2.l5.u2 of the run)."""

    writer = worksheet_writer(body)
    writer["components"]["c0"]["semantic_content"]["sections"][2]["blocks"][0]["text"] = (
        "Ví dụ mẫu: khách quen gọi lại lần hai vì hàng giao muộn; nhân viên ghi nhận cấp hai rồi báo trưởng nhóm.")
    choices = writer["components"]["c1"]["choices"]
    choices[1]["text"] = "Báo trưởng nhóm, ghi nhận cấp hai: khách quen gọi lại lần hai vì hàng giao muộn"
    choices[2]["text"] = "Cấp 3 vì khách dọa đăng lên mạng xã hội"
    return writer


FIXED_CHOICES = [{"text": "Cấp 1 vì chưa có thiệt hại tài chính", "correct": False},
                 {"text": "Cấp 2, vì cùng một người đã gọi lại để khiếu nại", "correct": True},
                 {"text": "Cấp 3 vì khách dọa đăng lên mạng xã hội", "correct": False}]


def budget_lines(output: list[str]) -> list[dict[str, Any]]:
    return [json.loads(line.split("lesson_author_idm ", 1)[1]) for line in output if "idm_unit_budget" in line]


class UnitBudgetTests(unittest.IsolatedAsyncioTestCase):
    def test_levels_follow_the_time_left(self) -> None:
        # Node's 120 s request leaves 115 s: medium p90 80 s + low repair 20 s + judge 25 s + 5 s headroom = 130 s.
        self.assertEqual(writer_thinking(115.0, repair_room=True, preferred="medium"), "low")
        self.assertEqual(writer_thinking(115.0, repair_room=False, preferred="medium"), "medium")
        self.assertEqual(writer_thinking(295.0, repair_room=True, preferred="medium"), "medium")
        self.assertEqual(writer_thinking(295.0, repair_room=True, preferred="low"), "low")
        # A leak repair is targeted: low thinking, not started below 30 s.
        self.assertEqual(repair_thinking_for(36.5, targeted=True, preferred="medium"), "low")
        self.assertIsNone(repair_thinking_for(29.0, targeted=True, preferred="medium"))
        # Any other repair: medium with room for it and the judge, else low, else none.
        self.assertEqual(repair_thinking_for(105.0, targeted=False, preferred="medium"), "medium")
        self.assertEqual(repair_thinking_for(60.0, targeted=False, preferred="medium"), "low")
        self.assertIsNone(repair_thinking_for(24.0, targeted=False, preferred="medium"))
        self.assertEqual((judge_repair_fits(49.0), judge_repair_fits(50.0)), (False, True))

    @staticmethod
    async def run_unit(provider: ClockedGenerate, body: dict[str, Any], *, seconds: float = 115.0,
                       judge_mode: str = "observe") -> dict[str, Any]:
        request = RagLessonAuthorUnitV2Request.model_validate(body)
        deps = dataclasses.replace(unit_service._idm_unit_deps(request), judge_mode=judge_mode)
        runtime = IdmRuntime(generate=provider, api_key="test-key", model="fake-model", locale="vi",
                             deadline=provider.clock() + seconds, clock=provider.clock)
        return await run_idm_unit(contract=request.unit_contract, runtime=runtime, deps=deps, fallback_only=False)

    async def test_the_leak_repair_of_a_slow_writer_still_fits_with_the_judge(self) -> None:
        # c2.l5.u2: the writer took 78.5 s and the medium leak repair was cut 31.4 s later. Now the writer of a
        # question unit thinks low, and even at 78.5 s the leak repair is a low-thinking call the judge follows.
        body = worksheet_body()
        writer = leaking_worksheet(body)
        provider = ClockedGenerate({WRITER: 78.5, REPAIR: 7.0, JUDGE: 8.0}, **{
            WRITER: [writer], REPAIR: [slot_repair(writer, 1, choices=FIXED_CHOICES)], JUDGE: [judge()]})
        with self.assertLogs("app.idm", level="INFO") as logs:
            result = await self.run_unit(provider, body)
        self.assertEqual(provider.names, [WRITER, REPAIR, JUDGE])
        self.assertEqual([call["thinking_level"] for call in provider.calls], ["low", "low", "low"])
        self.assertEqual((result["content_origin"], result["usage_source"]), ("provider_validated", "provider"))
        self.assertEqual(result["unit"]["idm_quality"]["judge_status"], "pass")
        self.assertEqual([(item["step"], item["thinking"], item["remaining_ms"]) for item in budget_lines(logs.output)],
                         [("writer", "low", 115_000), ("repair_targeted", "low", 36_500)])

    async def test_a_repair_that_cannot_finish_is_not_started(self) -> None:
        # A shared-validator finding (feedback that does not teach) with 22 s left: no call cut by the deadline; the
        # slot is settled by the deterministic steps and the note says why.
        body = severity_body()
        writer = severity_writer(body)
        writer["components"]["c1"]["explanation"] = "Đúng."
        provider = ClockedGenerate({WRITER: 93.0, REPAIR: 7.0, JUDGE: 8.0}, **{WRITER: [writer], JUDGE: [judge()]})
        with self.assertLogs("app.idm", level="INFO") as logs:
            result = await self.run_unit(provider, body)
        self.assertEqual(provider.names, [WRITER])
        self.assertEqual(budget_lines(logs.output)[-1]["thinking"], "skipped")
        self.assertIn("IDM_W5_REPAIR_SKIPPED_BUDGET", result["unit"]["idm_quality"]["author_note"])

    async def test_a_judge_repair_needs_room_for_the_second_judge(self) -> None:
        body = severity_body()
        provider = ClockedGenerate({WRITER: 70.0, REPAIR: 7.0, JUDGE: 8.0}, **{
            WRITER: [severity_writer(body)], JUDGE: [judge(("Q4_feedback_teaches", "major", 1))]})
        with self.assertLogs("app.idm", level="INFO") as logs:
            result = await self.run_unit(provider, body, judge_mode="repair")
        # 45 s left after the writer, 37 s after the judge: no repair of its finding (that needs 50 s with the second
        # judge), the finding stays a review note.
        self.assertEqual(provider.names, [WRITER, JUDGE])
        self.assertEqual(result["unit"]["idm_quality"]["judge_status"], "review_required")
        self.assertEqual(budget_lines(logs.output)[-1], {**budget_lines(logs.output)[-1], "step": "judge_repair",
                                                        "thinking": "skipped"})


# --- R2: leaks the pair check missed and keys the unit itself contradicts ---------------------------------------------
RUN_MCQ = json.loads((Path(__file__).parent / "fixtures" / "qlt4_run_ab8d67e1_mcq.json").read_text("utf-8"))
QLT3_MCQ = json.loads((Path(__file__).parent / "fixtures" / "qlt3_run_8de1c76b_mcq.json").read_text("utf-8"))


def teaching_of(question: dict[str, Any]) -> UnitTeaching:
    return unit_teaching([{"type": "html", "semantic_content": semantic} for semantic in question["preceding_html"]],
                         question["faq"])


def as_problem(question: dict[str, Any]) -> dict[str, Any]:
    return {"question": question["question"], "choices": question["choices"]}


def html_slot(*blocks: dict[str, Any]) -> dict[str, Any]:
    return {"type": "html", "semantic_content": {"sections": [{"heading": "Nội dung", "blocks": list(blocks)}]}}


CONTESTED_CASE = ("Một khách hàng phàn nàn rằng sản phẩm gây mất an toàn, liên quan pháp lý và truyền thông đã đưa "
                  "tin. Khiếu nại này thuộc cấp độ nào?")


class AnswerCheckTests(StoryboardEndpointTestCase):
    def test_the_leaks_and_the_contested_key_of_the_run_are_found_and_nothing_else(self) -> None:
        leaks = set()
        contested = {}
        for question in RUN_MCQ["questions"]:
            text = " ".join(visible_text(semantic) for semantic in question["preceding_html"])
            problem = as_problem(question)
            if copied_options(problem, text) or reworded_key(problem, text) or quoted_key(problem,
                                                                                          teaching_of(question)):
                leaks.add(question["unit"])
            rival = contested_option(problem, teaching_of(question))
            if rival is not None:
                contested[question["unit"]] = question["choices"][rival]["text"].split(",")[0]
        # c4.l1.u1 (the key restates the blockquote) was missed by the pair check; the other two were found already.
        self.assertEqual(leaks, {"c2.l5.u2", "c4.l1.u1", "c5.l1.u2"})
        quote = next(question for question in RUN_MCQ["questions"] if question["unit"] == "c4.l1.u1")
        self.assertTrue(quoted_key(as_problem(quote), teaching_of(quote)))
        # c1.l2.u2: "Bậc 01" for a case with ISO 9001 and CE, which the unit's row and FAQ give to "Bậc 02".
        self.assertEqual(contested, {"c1.l2.u2": "Bậc 02 - Vietnam Quality"})
        # The 9 questions of run 8de1c76b (as one block of text) raise nothing new.
        for question in QLT3_MCQ["questions"]:
            teaching = unit_teaching([html_slot({"kind": "paragraph", "text": question["preceding_html_text"]})], [])
            self.assertFalse(quoted_key(as_problem(question), teaching), question["unit"])
            self.assertIsNone(contested_option(as_problem(question), teaching), question["unit"])

    def test_short_keys_and_copied_cases(self) -> None:
        teaching = unit_teaching([html_slot(
            {"kind": "bullets", "items": ["Hiện đại hóa: áp dụng chuẩn quản trị quốc tế.",
                                          "Việt Nam hóa: chuyển hóa chuẩn toàn cầu cho phù hợp SME Việt."]},
            {"kind": "table", "rows": [{"label": "Nhóm gia công", "value": "Làm theo bản vẽ của đối tác."},
                                       {"label": "Nhóm làm chủ công nghệ", "value": "Sở hữu quy trình riêng."},
                                       {"label": "Nhóm thương hiệu", "value": "Khách tìm mua nhờ uy tín."}]},
            {"kind": "paragraph", "text": "Ví dụ: Công ty May An Phát nhận bản vẽ và nguyên phụ liệu từ đối tác, chỉ "
                                          "cắt may và hưởng biên lợi nhuận 4% nên thuộc nhóm gia công."})], [])
        # The only option the html shows, word for word: a cue, not a test of the criterion.
        only_one = {"question": "Trụ cột nào đòi hỏi áp dụng chuẩn quản trị quốc tế?", "choices": [
            {"text": "Hiện đại hóa", "correct": True}, {"text": "Số hóa", "correct": False},
            {"text": "Tự động hóa", "correct": False}]}
        self.assertTrue(quoted_key(only_one, teaching))
        # Every option taught: no cue, unless the question copies the one line that holds the key (a lookup).
        both = {**only_one, "choices": [{"text": "Hiện đại hóa", "correct": True},
                                        {"text": "Việt Nam hóa", "correct": False}]}
        self.assertTrue(quoted_key(both, teaching))
        reworded = {**both, "question": "Trụ cột nào giúp doanh nghiệp làm theo cách quản lý tốt nhất của thế giới?"}
        self.assertFalse(quoted_key(reworded, teaching))
        # The case copies the example that names the answer.
        copied_case = {"question": "Công ty May An Phát nhận bản vẽ và nguyên phụ liệu từ đối tác, chỉ cắt may và "
                                   "hưởng biên lợi nhuận 4%. Công ty thuộc nhóm nào?", "choices": [
            {"text": "Nhóm gia công", "correct": True}, {"text": "Nhóm làm chủ công nghệ", "correct": False},
            {"text": "Nhóm thương hiệu", "correct": False}]}
        self.assertTrue(quoted_key(copied_case, teaching))
        new_case = {**copied_case, "question": "Xưởng Minh Long tự thiết kế mẫu, mua vật tư trong nước và bán dưới "
                                               "thương hiệu riêng. Xưởng thuộc nhóm nào?"}
        self.assertFalse(quoted_key(new_case, teaching))

    async def test_a_contested_key_gets_a_targeted_repair_naming_the_rival(self) -> None:
        body = severity_body()
        writer = severity_writer(body)
        writer["components"]["c1"]["question"] = CONTESTED_CASE
        writer["components"]["c1"]["choices"][1]["text"] = "Cấp 2 vì khách hàng phàn nàn"
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [slot_repair(severity_writer(body), 1)],
                                   JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        self.assertEqual(provider.names, [WRITER, REPAIR, JUDGE])
        repair = provider.calls[1]
        self.assertEqual(repair["thinking_level"], "low")
        self.assertIn("c1 IDM_W5_ANSWER_CONTESTED at c1.question", repair["prompt"])
        self.assertIn("(the case also meets the criteria the unit gives for option ", repair["prompt"])
        quality = self.assert_envelope(data, "provider_validated", "validated", "provider")
        self.assertIn("Đã tự sửa: IDM_W5_ANSWER_CONTESTED.", quality.author_note)

    async def test_a_contested_key_the_repair_keeps_is_left_for_the_author(self) -> None:
        body = severity_body()
        writer = severity_writer(body)
        writer["components"]["c1"]["question"] = CONTESTED_CASE
        writer["components"]["c1"]["choices"][1]["text"] = "Cấp 2 vì khách hàng phàn nàn"
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [slot_repair(writer, 1)], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = IdmUnitQualityV1.model_validate(data["unit"]["idm_quality"])
        self.assertEqual(data["quality_state"], "review_required")
        self.assertIn("Cần xem (IDM_W5_ANSWER_CONTESTED, khối 2 (Quiz)): tình huống của câu hỏi cũng khớp tiêu chí",
                      quality.author_note)

    def test_the_judge_checks_one_answer_against_the_unit(self) -> None:
        prompt = judge_prompt("vi", plan_summary={}, facts=[], unit_content=[])
        self.assertIn("whose key contradicts an FAQ answer of this unit, is major", prompt)
