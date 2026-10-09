"""QC run 8de1c76b (course-v1:LAndA2+235245+2026, 2026-10-09, 7.8/10) regressions for QLT-3.

* Q1: 3 of 4 chapters lost a lesson to the deterministic fallback: list bounds the prompt never stated
  (33 Canvas criteria facts, bound 24; an 8-point media brief, bound 6) rejected whole W4 answers, a schema
  rejection used up one of the two attempts, single-block lessons were split into units sharing the block,
  and the failing paths of each attempt were never logged.
* Q5: two Must Dos had no practice: the fallback kept the provider practice only for kind "do", and
  "Ký cam kết …" was marked "decide"; ``learning_activities`` promised the held practice anyway.

The provider is the local fake; nothing reaches the network.
"""

from __future__ import annotations

import json
import unittest
from dataclasses import replace
from typing import Any

from app.idm.contracts import (
    IdmBlueprintRowV1,
    IdmContentBlockV1,
    IdmLessonDesignV1,
    IdmLessonPlanV1,
    IdmPracticeTaskV1,
    IdmW3W4ModuleResponseV1,
)
from app.idm.limits import answer_limits
from app.idm.module_autofix import (
    BLOCK_ATTACHED_CODE,
    BLOCK_COVERED_CODE,
    COMPONENT_ORDER_CODE,
    CRITERIA_TRIMMED_CODE,
    FORMAT_TO_HTML_CODE,
    UNITS_MERGED_CODE,
    autofix_lesson,
    ranked_criteria,
    trim_module_answer,
)
from app.idm.module_design import validate_lesson
from app.idm.module_layout import ModuleScope, fallback_lesson, must_do_unit_position, project_lesson
from app.idm.policy import AI_DRAFTED_MARKER_EN, AI_DRAFTED_MARKER_VI, IDM_PRACTICE_MAX_CRITERIA_FACTS
from app.idm.prompts import module_prompt
from app.idm.validation import errors
from tests import idm_golden_module as gm
from tests.idm_golden import key, keys
from tests.test_idm_module_design import ALLOWED, STAGE, component, design_module, edited, scope_of

# The 33 facts of block cb_0022 ("Cách điền biểu mẫu CEO Change Mindset Canvas"), run 8de1c76b.
CANVAS_FACTS = [
    "CÔNG CỤ THỰC CHIẾN", "CEO CHANGE MINDSET CANVAS • 9 Ô TỰ ĐÁNH GIÁ", "Biểu Mẫu CEO Change Mindset Canvas",
    "Đây là công cụ bắt buộc mỗi học viên phải hoàn thành trước khi rời khỏi Modun 1. Hãy trả lời trung thực, "
    "ngắn gọn và dứt khoát vào 9 ô dưới đây để thiết lập bản đồ chuyển đổi cho chính mình",
    "MINDSET CẦN BỎ", "KHÁT VỌNG MỚI", "GIÁ TRỊ KHÁCH HÀNG",
    "Niềm tin hay thói quen quản trị kinh nghiệm cũ nào đang kìm hãm doanh nghiệp mà anh/chị cam kết từ bỏ ngay?",
    "Mục tiêu tăng trưởng vượt chuẩn ngành lớn nhất trong 3 năm tới của doanh nghiệp là gì?",
    "Khách hàng mục tiêu đang mong muốn giá trị vượt trội nào mà thị trường hiện tại chưa phục vụ tốt?",
    "[ Ghi rõ 01 tư duy cũ cần đoạn tuyệt... ]", "[ Xác định mục tiêu định lượng cụ thể... ]",
    "[ 01 giá trị mang lại Customer Delight... ]",
    "NĂNG LỰC BIC", "VIỆC CẦN DỪNG", "VIỆC CẦN BẮT ĐẦU",
    "Doanh nghiệp bắt buộc phải xây dựng năng lực lõi nào để dẫn đầu phân khúc ngành đã chọn?",
    "Những hoạt động kinh doanh, dòng sản phẩm hoặc cuộc họp kém hiệu quả nào cần dừng lại ngay?",
    "Hành động hoặc kỷ luật quản trị hệ thống mới nào cần phải được áp dụng ngay vào sáng thứ Hai tuần tới?",
    "[ Năng lực vận hành / R&D / Sản xuất... ]", "[ Cắt bỏ lãng phí để giải phóng nguồn lực... ]",
    "[ Kỷ luật phòng ngừa / Đo lường dữ liệu... ]",
    "ỨNG DỤNG AI", "KẾT NỐI HỆ SINH THÁI", "LỜI HỨA TOÀN CẦU",
    "Quy trình hay điểm nghẽn nào trong doanh nghiệp sẽ được giải quyết hoặc tăng gấp đôi năng suất bằng AI?",
    "Nguồn lực chiến lược bên ngoài nào (Chuyên gia, Vốn, Công nghệ) cần kết nối qua ERA5.0?",
    "Chuẩn mực chất lượng hoặc chứng chỉ toàn cầu cao nhất mà doanh nghiệp cam kết đạt được là gì?",
    "[ Xác định đối tác chiến lược bên ngoài... ]", "[ 01 use case ứng dụng AI cụ thể... ]",
    "[ Cam kết chuẩn mực chất lượng... ]", "XÁC NHẬN CỦA NGƯỜI ĐỨNG ĐẦU DOANH NGHIỆP",
    "\"Tôi cam kết rằng 9 nội dung trên đây là định hướng hành động thật của tôi và Ban điều hành, không phải lý "
    "thuyết đối phó trên lớp học.\"",
]
CANVAS_KEYS = [f"d1-c9-f{number}" for number in range(1, len(CANVAS_FACTS) + 1)]
CANVAS_LABELS = {"CÔNG CỤ THỰC CHIẾN", "MINDSET CẦN BỎ", "KHÁT VỌNG MỚI", "GIÁ TRỊ KHÁCH HÀNG", "NĂNG LỰC BIC",
                 "VIỆC CẦN DỪNG", "VIỆC CẦN BẮT ĐẦU", "ỨNG DỤNG AI", "LỜI HỨA TOÀN CẦU"}


def block(block_id: str, fact_keys: list[str]) -> IdmContentBlockV1:
    return IdmContentBlockV1(block_id=block_id, section_id="sec_001", name="Biểu mẫu", summary="Biểu mẫu.",
                             intent="do", support_role=None, content_kind="procedure", fact_keys=fact_keys,
                             issues=[], gaps=[], sme_questions=[], origin="provider")


def row(block_id: str, must_dos: list[str], classification: str = "must_do") -> IdmBlueprintRowV1:
    return IdmBlueprintRowV1.model_validate({
        "block_id": block_id, "lo_id": "lo_4", "must_do_ids": must_dos, "classification": classification,
        "placement": "course", "treatment": "keep", "detail_level": "Đủ để điền.", "hold": False,
        "hold_reason": None, "sme_question": None, "combine_into": None, "separate_into": [],
        "rationale": "Biểu mẫu của Must Do."})


def canvas_scope() -> tuple[ModuleScope, IdmLessonPlanV1]:
    """Chapter 3 of the QC run: one lesson (lsn_010), one block (cb_0022, 33 facts) for md_9 (kind do)."""

    plan = IdmLessonPlanV1.model_validate({
        "lesson_key": "lsn_010", "kind": "learning", "title": "Thiết lập bản đồ tư duy hành động",
        "primary_must_do_id": "md_9", "secondary_must_do_ids": [], "block_ids": ["cb_0022"], "est_screens": 4,
        "est_minutes": 10, "ordering_rationale": "Một khối."})
    scope = ModuleScope(
        blocks={"cb_0022": block("cb_0022", CANVAS_KEYS)}, rows={"cb_0022": row("cb_0022", ["md_9"])},
        scope_of_block={"cb_0022": "idmcb_" + "a" * 32},
        fact_text=dict(zip(CANVAS_KEYS, CANVAS_FACTS, strict=True)), lesson_plans=[plan],
        must_do_statement={"md_9": "Điền đủ 9 ô nội dung trên biểu mẫu CEO Change Mindset Canvas"}, locale="vi",
        must_do_kind={"md_9": "do"})
    return scope, plan


def practice(criteria: list[str]) -> dict[str, Any]:
    return {**gm._practice("Cho hiện trạng doanh nghiệp, người học điền 9 ô của biểu mẫu để xác định việc cần làm",
                           criteria), "practice_id": "pt_1"}


class ListBoundTests(unittest.IsolatedAsyncioTestCase):
    def test_canvas_criteria_keep_the_statements_of_the_must_do_block(self) -> None:
        # Q1a: the practice of lsn_010 listed all 33 facts of cb_0022; the bound is 24 on both sides (the Node
        # contract reads the same bound), so the list is cut, never the answer: the 9 short labels go first.
        scope, plan = canvas_scope()
        kept = ranked_criteria(CANVAS_KEYS, scope, plan)
        self.assertEqual(len(kept), IDM_PRACTICE_MAX_CRITERIA_FACTS)
        self.assertEqual(kept, [key_ for key_, text in zip(CANVAS_KEYS, CANVAS_FACTS, strict=True)
                                if text not in CANVAS_LABELS])
        # Duplicates go before anything else; a list within the bound is only de-duplicated.
        self.assertEqual(ranked_criteria([*CANVAS_KEYS[:5], *CANVAS_KEYS[:5]], scope, plan), CANVAS_KEYS[:5])
        # Facts outside the lesson rank last, after the labels of the Must Do block.
        foreign = [f"d1-c1-f{number}" for number in range(1, 6)]
        self.assertTrue(set(ranked_criteria([*foreign, *CANVAS_KEYS], scope, plan)).isdisjoint(foreign))

    def test_the_whole_answer_is_cut_before_strict_validation(self) -> None:
        scope, _ = canvas_scope()
        answer = {"lessons": [{"lesson_key": "lsn_010", "practice_tasks": [practice(list(CANVAS_KEYS))]}]}
        value, codes = trim_module_answer(json.loads(json.dumps(answer)), scope)
        self.assertEqual(len(value["lessons"][0]["practice_tasks"][0]["criteria_fact_keys"]), 24)
        self.assertEqual(codes, {CRITERIA_TRIMMED_CODE: 1})
        untouched, none = trim_module_answer({"lessons": [{"practice_tasks": [{"criteria_fact_keys": [1] * 30}]}]},
                                             scope)
        self.assertEqual((len(untouched["lessons"][0]["practice_tasks"][0]["criteria_fact_keys"]), none),
                         (30, {}))

    async def test_over_long_lists_no_longer_reject_the_w4_answer(self) -> None:
        # Q1a end to end: 30 criteria keys (duplicates of the lesson facts) and an 8-point media brief (the 8
        # Quick-Win job aid) are cut; the writer answer is accepted on the first call.
        def overflow(payload: dict[str, Any]) -> None:
            lesson = payload["lessons"][0]
            lesson["practice_tasks"][0]["criteria_fact_keys"] = [*keys(2, 2, 3), *keys(3, 2, 6)] * 5
            lesson["units"][0]["media_brief"] = {
                "type": "static_infographic", "title": "8 cam kết Quick-Win",
                "content_points": [f"Cam kết {number}" for number in range(1, 9)],
                "context_description": "Một bảng kiểm treo tại phòng họp.", "rationale": "Dễ tra khi làm việc."}

        result, provider, runtime = await design_module(0, [edited("mod_01", overflow)])
        design = result["shard"]["idm_design"]
        self.assertEqual((len(provider.calls), design["stage_origin"]), (1, "provider"))
        lesson = design["lessons"][0]
        self.assertEqual(lesson["practice_tasks"][0]["criteria_fact_keys"], [*keys(2, 2, 3), *keys(3, 2, 6)])
        self.assertEqual(lesson["units"][0]["media_brief"]["content_points"],
                         [f"Cam kết {number}" for number in range(1, 7)])
        self.assertEqual(runtime.adjustments[CRITERIA_TRIMMED_CODE], 1)
        self.assertEqual(runtime.adjustments["IDM_RESPONSE_LIST_TRIMMED"], 1)

    def test_every_list_and_text_bound_is_stated_in_the_w4_prompt(self) -> None:
        limits = answer_limits(IdmW3W4ModuleResponseV1)
        for expected in ("criteria_fact_keys at most 24 items", "content_points 1-6 items",
                         "learner_action 3-200 chars", "components 1-4 items", "units[]: title 3-180 chars",
                         "brief 3-300 chars"):
            self.assertIn(expected, limits)
        prompt = module_prompt("vi", module_plan={}, lesson_plans=[], audience="Nhân viên", objectives=[],
                               must_dos=[], course_blocks=[], facts=[], allowed_components=["html"], limits=limits)
        self.assertIn("10. LIMITS of the answer", prompt)
        self.assertIn(limits, prompt)
        self.assertIn("a lesson with\n   ONE block has exactly ONE unit", prompt)


class SchemaRepairTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def bad_enum(payload: dict[str, Any]) -> None:
        payload["lessons"][1]["practice_tasks"][0]["bloom"] = "memorise"

    async def test_schema_rejection_gets_its_own_repair_and_keeps_the_content_attempt(self) -> None:
        # Q1b: the writer answer fails the schema (lesson 2); lesson 1 is salvaged at once, the schema repair answer
        # still has a content error, and the content repair that follows is the third call, not a fallback.
        foreign = edited("mod_01", lambda p: p["lessons"][1]["practice_tasks"][0]["criteria_fact_keys"].append(
            key(2, 2)))
        with self.assertLogs("app.idm", level="INFO") as logs:
            result, provider, runtime = await design_module(
                0, [edited("mod_01", self.bad_enum), foreign, gm.MODULE_RESPONSES["mod_01"]])
        design = result["shard"]["idm_design"]
        self.assertEqual((design["stage_origin"], len(provider.calls)), ("provider", 3))
        self.assertIn('"path":"lessons.1.practice_tasks.0.bloom"', provider.calls[1]["prompt"])
        self.assertIn('"IDM_W3_PRACTICE_CRITERIA_FOREIGN"', provider.calls[2]["prompt"])
        self.assertEqual([(event["invocation_kind"], event["outcome"]) for event in runtime.trace],
                         [("writer", "failed"), ("repair", "succeeded"), ("repair", "succeeded")])
        attempts = [json.loads(line.split("lesson_author_idm ", 1)[1]) for line in logs.output
                    if "idm_module_attempt" in line]
        self.assertEqual([(item["call"], item["outcome"]) for item in attempts],
                         [(1, "idm_response_schema_invalid"), (2, "validated"), (3, "validated")])
        self.assertEqual(attempts[0]["accepted_lessons"], ["lsn_001"])
        self.assertEqual(attempts[0]["lesson_failures"], {"lsn_002": ["IDM_RESPONSE_SCHEMA_INVALID@lessons[1]"]})
        self.assertEqual(attempts[0]["schema_errors"], ["literal_error@lessons.1.practice_tasks.0.bloom"])
        # Q1d: the exact code and path of each failing lesson, per attempt (codes and paths only).
        self.assertEqual(attempts[1]["lesson_failures"], {"lsn_002": [
            "IDM_W3_PRACTICE_CRITERIA_FOREIGN@lessons[1].practice_tasks[0].criteria_fact_keys"]})

    async def test_only_one_schema_repair_is_free(self) -> None:
        bad = edited("mod_01", self.bad_enum)
        result, provider, _ = await design_module(0, [bad, bad, bad])
        self.assertEqual(len(provider.calls), 3)
        design = result["shard"]["idm_design"]
        # The valid lesson of the rejected answers is kept; only the lesson that never parsed falls back.
        self.assertEqual(design["stage_origin"], "partial_fallback")
        self.assertEqual(design["lessons"][0]["practice_tasks"][0]["sentence"],
                         gm.MODULE_RESPONSES["mod_01"]["lessons"][0]["practice_tasks"][0]["sentence"])
        self.assertIn("Bố cục dự phòng tự động", design["lessons"][1]["notes"])


class LayoutAutofixTests(unittest.IsolatedAsyncioTestCase):
    def fixed(self, change: Any, chapter_index: int = 0, position: int = 1) -> tuple[IdmLessonDesignV1, Any]:
        module_key = ("mod_01", "mod_02", "mod_03")[chapter_index]
        scope = scope_of(chapter_index)
        lesson = IdmLessonDesignV1.model_validate(edited(module_key, change)["lessons"][position])
        plan = scope.lesson_plans[position]
        self.assertTrue(errors(validate_lesson(lesson, plan, scope, ALLOWED, "lessons[1]")))
        result, codes = autofix_lesson(lesson, plan, scope, ALLOWED)
        self.assertEqual(errors(validate_lesson(result, plan, scope, ALLOWED, "lessons[1]")), [])
        return result, codes

    @staticmethod
    def split_single_block(worksheet: bool) -> Any:
        def change(payload: dict[str, Any]) -> None:
            lesson = payload["lessons"][1]  # lsn_002: one block (cb_0005)
            practice_unit = ([component("html", "practice", ["cb_0005"], practice_id="pt_1")] if worksheet else [])
            lesson["units"] = [
                gm._unit(1, "context_explain", "Dấu hiệu của từng cấp độ", ["cb_0005"], [
                    component("html", "explain", ["cb_0005"])]),
                gm._unit(2, "practice_feedback", "Luyện tập xác định cấp độ", ["cb_0005"], [
                    *practice_unit, component("problem", "practice", ["cb_0005"], practice_id="pt_1"),
                    component("la_faq", "clarify", ["cb_0005"])]),
            ]
        return change

    def test_single_block_lesson_split_into_two_units_is_merged(self) -> None:
        # Q1c: lsn_010/lsn_012 (one block each) became an explanation unit + a practice unit on the same block:
        # IDM_W4_UNIT_BLOCK_PARTITION in both attempts. Node requires each block in exactly one unit, so the
        # units are merged into one unit that teaches and practises.
        lesson, codes = self.fixed(self.split_single_block(worksheet=False))
        self.assertEqual(codes, {UNITS_MERGED_CODE: 1})
        self.assertEqual(len(lesson.units), 1)
        unit = lesson.units[0]
        self.assertEqual(([c.type for c in unit.components], unit.segment, unit.title),
                         (["html", "problem", "la_faq"], "practice_feedback", "Dấu hiệu của từng cấp độ"))
        self.assertEqual([c.component_index for c in unit.components], [1, 2, 3])

    def test_worksheet_wins_over_the_explanation_html_of_the_same_block(self) -> None:
        lesson, codes = self.fixed(self.split_single_block(worksheet=True))
        self.assertEqual(codes, {UNITS_MERGED_CODE: 1})
        self.assertEqual([(c.type, c.role, c.practice_id) for c in lesson.units[0].components],
                         [("html", "practice", "pt_1"), ("problem", "practice", "pt_1"), ("la_faq", "clarify", None)])

    def test_order_uncovered_and_missing_blocks_and_formats_without_evidence(self) -> None:
        # Q1c/Q1d: ch1.l2 failed twice on IDM_W4_COMPONENT_ORDER, IDM_W4_FORMAT_EVIDENCE (a diagram the facts give no
        # relation for) and IDM_W4_UNIT_BLOCKS_UNCOVERED; each is fixed by the server without a provider call.
        def reverse(payload: dict[str, Any]) -> None:
            payload["lessons"][1]["units"][0]["components"].reverse()

        _, codes = self.fixed(reverse)
        self.assertEqual(codes, {COMPONENT_ORDER_CODE: 1})

        def uncovered(payload: dict[str, Any]) -> None:
            for item in payload["lessons"][0]["units"][0]["components"]:
                item["block_ids"] = ["cb_0003"]

        lesson, codes = self.fixed(uncovered, position=0)
        self.assertEqual(codes, {BLOCK_COVERED_CODE: 1})
        self.assertEqual(lesson.units[0].components[0].block_ids, ["cb_0003", "cb_0004"])

        def missing(payload: dict[str, Any]) -> None:
            unit = payload["lessons"][0]["units"][0]
            unit["block_ids"] = ["cb_0003"]
            for item in unit["components"]:
                item["block_ids"] = ["cb_0003"]

        lesson, codes = self.fixed(missing, position=0)
        self.assertEqual(codes, {BLOCK_ATTACHED_CODE: 1})
        self.assertEqual(lesson.units[0].block_ids, ["cb_0003", "cb_0004"])

        def diagram(payload: dict[str, Any]) -> None:
            payload["lessons"][0]["units"][0]["components"].insert(
                1, component("la_diagram", "show", ["cb_0009"]))

        lesson, codes = self.fixed(diagram, chapter_index=1, position=0)
        self.assertEqual(codes, {FORMAT_TO_HTML_CODE: 1})
        self.assertEqual([c.type for c in lesson.units[0].components], ["html"])

    def test_unfixable_layouts_are_left_to_the_provider_repair(self) -> None:
        def two_practices(payload: dict[str, Any]) -> None:
            lesson = payload["lessons"][1]
            lesson["practice_tasks"].append({**lesson["practice_tasks"][0], "practice_id": "pt_2"})
            lesson["units"] = [
                gm._unit(1, "practice_feedback", "Một", ["cb_0005"], [
                    component("problem", "practice", ["cb_0005"], practice_id="pt_1")]),
                gm._unit(2, "practice_feedback", "Hai", ["cb_0005"], [
                    component("problem", "practice", ["cb_0005"], practice_id="pt_2")]),
            ]

        scope = scope_of(0)
        lesson = IdmLessonDesignV1.model_validate(edited("mod_01", two_practices)["lessons"][1])
        self.assertEqual(autofix_lesson(lesson, scope.lesson_plans[1], scope, ALLOWED), (lesson, {}))

    async def test_merged_lesson_is_accepted_without_a_repair_call(self) -> None:
        result, provider, runtime = await design_module(0, [edited("mod_01", self.split_single_block(False))])
        self.assertEqual((len(provider.calls), result["shard"]["idm_design"]["stage_origin"]), (1, "provider"))
        self.assertEqual(runtime.adjustments[UNITS_MERGED_CODE], 1)
        self.assertEqual(len(result["shard"]["lessons"][1]["units"]), 1)
        self.assertEqual(provider.calls[0]["schema"], STAGE)


class FallbackPracticeTests(unittest.IsolatedAsyncioTestCase):
    """Q5: md_2 ("Đối chiếu … 4 trục", decide) and md_11 ("Ký cam kết …", marked decide) had no practice at all."""

    @staticmethod
    def provider_practices(chapter_index: int, position: int = 0) -> list[IdmPracticeTaskV1]:
        payload = gm.MODULE_RESPONSES[("mod_01", "mod_02", "mod_03")[chapter_index]]
        return IdmLessonDesignV1.model_validate(payload["lessons"][position]).practice_tasks

    def test_deciding_must_do_keeps_its_practice_as_a_scenario_question(self) -> None:
        scope = scope_of(1)  # lsn_003, md_3 "Quyết định tự xử lý hay escalate" (decide), units cb_0009 / cb_0011
        plan = scope.lesson_plans[0]
        for locale, note in (("vi", "được giữ thành câu hỏi tình huống"), ("en", "kept as a scenario question")):
            lesson = fallback_lesson(plan, replace(scope, locale=locale),
                                     provider_practices=self.provider_practices(1))
            practice = lesson.practice_tasks[0]
            self.assertEqual((practice.hold, practice.hold_question), (False, None))
            # The unit that owns the Must Do block (cb_0009, must_do) at or after the criteria units.
            owner = lesson.units[must_do_unit_position(lesson.units, scope, plan.primary_must_do_id,
                                                       practice.criteria_fact_keys)]
            self.assertEqual([(c.type, c.role, c.practice_id) for c in owner.components],
                             [("html", "explain", None), ("problem", "practice", "pt_1")])
            self.assertEqual(owner.segment, "practice_feedback")
            self.assertTrue(str(owner.components[1].author_review.example_scenario).startswith(
                AI_DRAFTED_MARKER_VI if locale == "vi" else AI_DRAFTED_MARKER_EN))
            self.assertIn(note, lesson.notes)
            projected = project_lesson(lesson, plan, scope)
            self.assertEqual(projected["learning_activities"], [practice.sentence])
            self.assertEqual(errors(validate_lesson(lesson, plan, scope, ALLOWED, "lessons[0]")), [])

    def test_signing_a_commitment_is_practised_with_a_worksheet_even_when_marked_decide(self) -> None:
        scope = scope_of(1)
        plan = scope.lesson_plans[0]
        signing = replace(scope, must_do_statement={**scope.must_do_statement, "md_3": (
            "Ký cam kết bản Action Commitment chỉ định rõ 01 năng lực Best-in-Class, chỉ số KPI/OKR và người "
            "chịu trách nhiệm")})
        lesson = fallback_lesson(plan, signing, provider_practices=self.provider_practices(1))
        self.assertFalse(lesson.practice_tasks[0].hold)
        owner = lesson.units[must_do_unit_position(lesson.units, signing, plan.primary_must_do_id,
                                                   lesson.practice_tasks[0].criteria_fact_keys)]
        self.assertEqual([(c.type, c.role) for c in owner.components],
                         [("html", "practice"), ("problem", "practice")])
        self.assertIn("phiếu thực hành (worksheet)", lesson.notes)

    def test_learning_activities_never_promise_a_held_practice(self) -> None:
        scope = scope_of(1)
        plan = scope.lesson_plans[0]
        held = [practice.model_copy(update={"hold": True}) for practice in self.provider_practices(1)]
        lesson = fallback_lesson(plan, scope, provider_practices=held)
        self.assertTrue(all(practice.hold for practice in lesson.practice_tasks))
        activities = project_lesson(lesson, plan, scope)["learning_activities"]
        self.assertEqual(len(activities), 1)
        self.assertIn("Chưa có bài luyện tập", activities[0])
        self.assertNotIn(held[0].sentence, activities[0])

    async def test_fallback_of_a_deciding_must_do_opens_no_obligation(self) -> None:
        def foreign(payload: dict[str, Any]) -> None:
            payload["lessons"][0]["practice_tasks"][0]["criteria_fact_keys"].append(key(2, 2))

        bad = edited("mod_02", foreign)
        result, _, _ = await design_module(1, [bad, bad])
        shard = result["shard"]
        self.assertEqual(shard["idm_design"]["stage_origin"], "deterministic_fallback")
        self.assertEqual(shard["assessment_obligations"], [])
        lesson = shard["idm_design"]["lessons"][0]
        self.assertIn("problem", [c["type"] for u in lesson["units"] for c in u["components"]])
        self.assertEqual(shard["lessons"][0]["learning_activities"], [lesson["practice_tasks"][0]["sentence"]])


class PracticeTaskBoundTests(unittest.TestCase):
    def test_policy_bound_matches_the_contract(self) -> None:
        schema = IdmPracticeTaskV1.model_json_schema()["properties"]["criteria_fact_keys"]
        self.assertEqual(schema["maxItems"], IDM_PRACTICE_MAX_CRITERIA_FACTS)


if __name__ == "__main__":
    unittest.main()
