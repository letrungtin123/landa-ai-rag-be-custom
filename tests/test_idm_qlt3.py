"""QC run 8de1c76b (course-v1:LAndA2+235245+2026, 2026-10-09, 7.8/10) regressions for QLT-3.

* Q1: 3 of 4 chapters lost a lesson to the deterministic fallback: list bounds the prompt never stated
  (33 Canvas criteria facts, bound 24; an 8-point media brief, bound 6) rejected whole W4 answers, a schema
  rejection used up one of the two attempts, single-block lessons were split into units sharing the block,
  and the failing paths of each attempt were never logged.
* Q3: the orientation unit "Bản đồ 5 chuyển dịch" owned only the introduction block, so the writer could not
  list the five shifts, and its repair timed out against the 120 s unit deadline.
* Q2: the Canvas worksheet exceeded the 400-word budget twice and became raw PDF text.
* Q4: keys at D in 6 of 9 questions, the key the longest option in 8 of 9, three reworded leaks missed.
* Q5: two Must Dos had no practice: the fallback kept the provider practice only for kind "do", and
  "Ký cam kết …" was marked "decide"; ``learning_activities`` promised the held practice anyway.

The provider is the local fake; nothing reaches the network.
"""

from __future__ import annotations

import itertools
import json
import re
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from app.idm.contracts import (
    IdmBlueprintRowV1,
    IdmContentBlockV1,
    IdmLessonDesignV1,
    IdmLessonPlanV1,
    IdmPracticeTaskV1,
    IdmUnitQualityV1,
    IdmW3W4ModuleResponseV1,
    brief_hash_of,
)
from app.idm.framework import framework_support_brief, framework_support_lines
from app.idm.limits import answer_limits
from app.idm.mcq import (
    OPTION_LETTERS,
    encode_key_letters,
    normalize_single_choice,
    option_order,
    practice_key_shift,
    question_ordinal,
)
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
from app.idm.module_design import attach_framework_items, validate_lesson
from app.idm.module_layout import ModuleScope, fallback_lesson, must_do_unit_position, project_lesson
from app.idm.policy import AI_DRAFTED_MARKER_EN, AI_DRAFTED_MARKER_VI, IDM_PRACTICE_MAX_CRITERIA_FACTS
from app.idm.prompts import module_prompt
from app.idm.qa import reworded_key
from app.idm.validation import errors
from app.idm.worksheet import worksheet_fallback_html
from tests import idm_golden_module as gm
from tests.idm_golden import key, keys
from tests.idm_test_support import rehash_contract
from tests.test_idm_module_design import ALLOWED, STAGE, component, design_module, edited, scope_of
from tests.test_idm_storyboard import (
    JUDGE,
    REPAIR,
    WRITER,
    FakeGenerate,
    StoryboardEndpointTestCase,
    escalate_body,
    escalate_writer,
    judge,
    served,
    severity_body,
    severity_writer,
    slot_repair,
    worksheet_body,
    worksheet_writer,
)

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


# Facts of chapter 2 of the QC run: cb_0015 only introduces the framework; each shift has its own block.
SHIFT_INTRO = ["CHƯƠNG 05", "BIC 5 MINDSET SHIFTS • 5 BƯỚC CHUYỂN DỊCH TƯ DUY (1)",
               "5 Chuyển Dịch Tư Duy Sống Còn Của Lãnh Đạo 5.0",
               "Để bước qua cánh cổng chuyển đổi, CEO không thể chỉ dừng lại ở sự đồng cảm chung chung."]
SHIFT_HEADS = [
    "SHIFT 1: THINK BIG — Từ Tư Duy \"Tồn Tại\" Sang \"Khát Vọng Lớn\"",
    "SHIFT 2: THINK DIFFERENT — Từ Benchmark Nội Địa Sang Chuẩn Best-in-Class",
    "SHIFT 3: THINK CUSTOMER — Từ Tiêu Chuẩn Kỹ Thuật Sang \"Customer Delight\"",
    "SHIFT 4: THINK SYSTEM — Thoát Bẫy \"Chữa Cháy\" Sang Vận Hành Xuất Sắc",
    "SHIFT 5: THINK ECOSYSTEM — Từ \"Tự Sở Hữu Hữu Hạn\" Sang \"Siêu Kết Nối ERA5.0\"",
]
SHIFT_BRIEF = ("Liệt kê đủ 5 chuyển dịch: SHIFT 1: THINK BIG; SHIFT 2: THINK DIFFERENT; SHIFT 3: THINK CUSTOMER; "
               "SHIFT 4: THINK SYSTEM; SHIFT 5: THINK ECOSYSTEM")


def shift_scope() -> ModuleScope:
    """Module 2 of the QC run: lsn_004 (cb_0015 intro + cb_0016 Shift 1), then one lesson per other shift."""

    texts: dict[str, str] = {}
    blocks: dict[str, IdmContentBlockV1] = {}
    rows: dict[str, IdmBlueprintRowV1] = {}
    contents = [SHIFT_INTRO, *([head, "Tư duy cũ", "Tư duy mới"] for head in SHIFT_HEADS)]
    for index, facts in enumerate(contents, start=15):
        block_id = f"cb_{index:04d}"
        keys_ = [f"d1-c7-f{index * 10 + position}" for position in range(len(facts))]
        texts.update(zip(keys_, facts, strict=True))
        blocks[block_id] = block(block_id, keys_)
        rows[block_id] = row(block_id, [f"md_{max(1, index - 13)}"], "must_know" if index == 15 else "must_do")
    plans = [IdmLessonPlanV1.model_validate({
        "lesson_key": f"lsn_{number:03d}", "kind": "learning", "title": f"Bài {number}", "primary_must_do_id": None,
        "secondary_must_do_ids": [], "block_ids": block_ids, "est_screens": 3, "est_minutes": 6,
        "ordering_rationale": "Theo khung."})
        for number, block_ids in ((4, ["cb_0015", "cb_0016"]), (5, ["cb_0017"]), (6, ["cb_0018"]), (7, ["cb_0019"]),
                                  (8, ["cb_0020"]))]
    return ModuleScope(blocks=blocks, rows=rows, scope_of_block={key_: "idmcb_" + "b" * 32 for key_ in blocks},
                       fact_text=texts, lesson_plans=plans, must_do_statement={}, locale="vi")


def orientation_lesson(title: str = "Bản đồ 5 chuyển dịch", html_title: str = "Năm chuyển dịch là gì",
                       ) -> IdmLessonDesignV1:
    return IdmLessonDesignV1.model_validate(gm._lesson("lsn_004", "Think Big", "Xác lập tham vọng", [
        practice([key("7", 0)])], [
        gm._unit(1, "context_explain", title, ["cb_0015"], [
            gm._component(1, "html", "explain", html_title, ["cb_0015"])]),
        gm._unit(2, "practice_feedback", "Think Big", ["cb_0016"], [
            gm._component(1, "html", "explain", "Từ tồn tại sang khát vọng lớn", ["cb_0016"]),
            gm._component(2, "problem", "practice", "Bạn chọn gì?", ["cb_0016"], practice_id="pt_1")]),
    ]))


class FrameworkOrientationTests(StoryboardEndpointTestCase):
    """Q3: the orientation unit "Bản đồ 5 chuyển dịch" did not list the 5 shifts: its block held only the
    introduction, and the repair hit the 120 s unit deadline."""

    def test_w4_hands_the_orientation_unit_every_item_name(self) -> None:
        scope = shift_scope()
        lessons, attached = attach_framework_items([orientation_lesson()], scope)
        self.assertEqual(attached, 1)
        support = lessons[0].units[0].components[0].support_items
        self.assertEqual([(item.kind, item.brief, item.block_id) for item in support],
                         [("explain_concept", SHIFT_BRIEF, "cb_0015")])
        self.assertEqual(framework_support_lines([support[0].brief]), [head.split(" — ")[0] for head in SHIFT_HEADS])
        # A unit whose own facts number the items, or whose title promises nothing, is left as it is.
        self.assertEqual(lessons[0].units[1].components[0].support_items, [])
        self.assertEqual(attach_framework_items([orientation_lesson("Tổng quan", "Giới thiệu")], scope)[1], 0)
        # The html slot title alone ("Năm chuyển dịch là gì") is a promise too.
        self.assertEqual(attach_framework_items([orientation_lesson("Tổng quan")], scope)[1], 1)
        # English lead; names shortened, then labels only, to stay within the 300-character bound.
        self.assertTrue(str(framework_support_brief("5 shifts", ["SHIFT 1: THINK BIG"] * 5, "en")).startswith(
            "List all 5 shifts: SHIFT 1: THINK BIG; "))
        long_names = [f"Bước {number}: " + "rất dài " * 30 for number in range(1, 9)]
        brief = framework_support_brief("8 bước", long_names, "vi")
        self.assertEqual(brief, "Liệt kê đủ 8 bước: " + "; ".join(f"Bước {n}: rất dài rất" for n in range(1, 9)))

    @staticmethod
    def orientation_body(remaining_ms: int | None = None) -> dict[str, Any]:
        body = severity_body()
        contract = body["unit_contract"]
        contract["unit_title"] = "Bản đồ 5 chuyển dịch"
        brief = contract["idm_unit_brief"]
        brief["components"][0]["support_items"] = [
            {"kind": "explain_concept", "brief": SHIFT_BRIEF, "block_id": "cb_0005"}]
        brief["brief_hash"] = brief_hash_of(brief)
        rehash_contract(contract)
        if remaining_ms is not None:
            body["remaining_workflow_budget_ms"] = remaining_ms
        return body

    async def test_w5_lists_the_items_after_a_targeted_repair(self) -> None:
        body = self.orientation_body()
        writer = severity_writer(body)
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [slot_repair(writer, 0)], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        self.assertIn(SHIFT_BRIEF, provider.calls[0]["prompt"])
        # The repair is targeted (low thinking) and points the writer at the support item, not at source text.
        self.assertEqual(provider.calls[1]["thinking_level"], "low")
        self.assertIn("list every item named in the support item of the teaching html slot in UNIT_BRIEF",
                      provider.calls[1]["prompt"])
        quality = self.assert_envelope(data, "provider_validated", "review_required", "provider")
        sections = data["unit"]["components"][0]["semantic_content"]["sections"]
        self.assertEqual((sections[1]["heading"], sections[1]["blocks"][0]["items"]),
                         ("5 chuyển dịch gồm những gì?", [head.split(" — ")[0] for head in SHIFT_HEADS]))
        self.assertIn("Đã chèn danh sách 5 thành phần của khung", quality.author_note)

    async def test_without_time_for_a_repair_the_list_is_inserted_at_once(self) -> None:
        # 34 s for the request (29 s after the response headroom): below the 30 s a targeted repair needs.
        body = self.orientation_body(remaining_ms=34_000)
        provider = FakeGenerate(**{WRITER: [severity_writer(body)], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        self.assertEqual(provider.names, [WRITER, JUDGE])
        sections = data["unit"]["components"][0]["semantic_content"]["sections"]
        self.assertEqual(len(sections[1]["blocks"][0]["items"]), 5)
        self.assertEqual(data["quality_state"], "review_required")

    async def test_unknown_item_names_keep_the_slot_for_review(self) -> None:
        body = self.orientation_body()
        brief = body["unit_contract"]["idm_unit_brief"]
        brief["components"][0]["support_items"] = []
        brief["brief_hash"] = brief_hash_of(brief)
        rehash_contract(body["unit_contract"])
        writer = severity_writer(body)
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [slot_repair(writer, 0)], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = IdmUnitQualityV1.model_validate(data["unit"]["idm_quality"])
        self.assertIn("Cần xem (IDM_W5_FRAMEWORK_INCOMPLETE", quality.author_note)
        self.assertEqual(len(data["unit"]["components"][0]["semantic_content"]["sections"]), 1)


def long_paragraphs(tag: str, count: int, words: int = 150) -> list[dict[str, Any]]:
    """Distinct filler paragraphs (a long worked example), never a repeated block."""

    return [{"kind": "paragraph", "text": " ".join(f"{tag}{number}chu{index}" for index in range(words)) + ".",
             "items": [], "rows": []} for number in range(count)]


COMMITMENT_FACTS = [
    "PHẦN KẾT", "BẢN CAM KẾT HÀNH ĐỘNG CỦA LÃNH ĐẠO",
    "Tôi, người đứng đầu doanh nghiệp, cam kết dẫn dắt doanh nghiệp thực hiện cuộc chuyển đổi thực chất.",
    "Row 1: 01 Năng Lực Sẽ Trở Thành Best-in", "Row 2: 01 Hành Động Đo Được Trong 90 Ngày; ......................",
    "Row 3: Chỉ Số Đo Lường Thành Công (KPI /", "Row 4: Người Phụ Trách Triển Khai (Owner); ...................",
    "NGƯỜI LÃNH ĐẠO CAM KẾT (CEO / FOUNDER)", "Ký, ghi rõ họ tên & Đóng dấu doanh nghiệp",
]


class WorksheetBudgetTests(StoryboardEndpointTestCase):
    """Q2: the Canvas worksheet was written at 550 then 452 words (budget 400) and became raw PDF text."""

    def test_the_canvas_fallback_is_one_table_of_its_nine_fields(self) -> None:
        html = str(worksheet_fallback_html("Phiếu thực hành: Canvas", CANVAS_FACTS, "Cho hiện trạng, người học điền "
                                           "9 ô để xác định việc cần làm", "vi"))
        self.assertEqual(html.count("<table>"), 1)
        self.assertEqual(re.findall(r"<tr><td>([^<]*)</td>", html), [
            "Mindset cần bỏ", "Khát vọng mới", "Giá trị khách hàng", "Năng lực BIC", "Việc cần dừng",
            "Việc cần bắt đầu", "Ứng dụng AI", "Kết nối hệ sinh thái", "Lời hứa toàn cầu"])
        self.assertIn("<th>Ô cần điền</th><th>Câu hỏi gợi ý</th><th>Cần ghi</th>", html)
        self.assertIn("<td>Ghi rõ 01 tư duy cũ cần đoạn tuyệt</td>", html)
        self.assertIn("<h3>Xác nhận của người đứng đầu doanh nghiệp</h3>", html)
        self.assertIn("<li>Mindset cần bỏ: Ghi rõ 01 tư duy cũ cần đoạn tuyệt</li>", html)
        # No upper-case label list, no placeholder paragraph, no heading line of the PDF repeated as text.
        self.assertNotIn("[", html)
        self.assertNotIn("MINDSET", html)
        self.assertNotIn("CÔNG CỤ THỰC CHIẾN", html)

    def test_a_row_form_and_a_reference_table(self) -> None:
        html = str(worksheet_fallback_html("Cam kết", COMMITMENT_FACTS, None, "en"))
        self.assertEqual(re.findall(r"<tr><td>([^<]*)</td>", html), [
            "01 Năng Lực Sẽ Trở Thành Best-in", "01 Hành Động Đo Được Trong 90 Ngày",
            "Chỉ Số Đo Lường Thành Công (KPI /", "Người Phụ Trách Triển Khai (Owner)"])
        self.assertIn("<td>(write on your own copy)</td>", html)
        self.assertIn("<li>Filled in: 01 Hành Động Đo Được Trong 90 Ngày</li>", html)
        self.assertNotIn("......", html)
        # A reference table (several filled cells per row) is not a form: the generic fallback renders it.
        severity = ["Row 1: Cấp độ | Dấu hiệu | Cách xử lý", "Row 2: Cấp 1 | Ảnh hưởng thấp | Nhân viên tự xử lý",
                    "Row 3: Cấp 2 | Thiệt hại dưới 50 triệu | Thông báo trưởng nhóm"]
        self.assertIsNone(worksheet_fallback_html("Cấp độ", severity, None, "vi"))

    async def test_over_long_worked_example_is_left_out_without_a_repair_call(self) -> None:
        body = worksheet_body()
        writer = worksheet_writer(body)
        sections = writer["components"]["c0"]["semantic_content"]["sections"]
        sections[2]["blocks"] = long_paragraphs("vidu", 3)
        provider = FakeGenerate(**{WRITER: [writer], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        self.assertEqual(provider.names, [WRITER, JUDGE])
        quality = self.assert_envelope(data, "provider_validated", "validated", "provider")
        kept = data["unit"]["components"][0]["semantic_content"]["sections"]
        self.assertEqual([section["heading"] for section in kept],
                         [sections[0]["heading"], sections[1]["heading"], sections[3]["heading"]])
        self.assertIn("đã bỏ phần ví dụ mẫu", quality.author_note)
        self.assertIn("IDM_W5_WORKSHEET_COMPACTED", quality.deterministic_codes)

    async def test_worksheet_still_over_budget_falls_back_to_a_structured_form(self) -> None:
        body = worksheet_body()
        contract = body["unit_contract"]
        texts = ["Hãy trả lời ngắn gọn vào từng ô của phiếu dưới đây.", "MINDSET CẦN BỎ", "KHÁT VỌNG MỚI",
                 "Niềm tin cũ nào đang kìm hãm doanh nghiệp?", "Mục tiêu tăng trưởng lớn nhất trong 3 năm tới là gì?"]
        for fact, text in zip(contract["source_facts"], texts, strict=True):
            fact["fact_text"] = text
        rehash_contract(contract)
        writer = worksheet_writer(body)
        writer["components"]["c0"]["semantic_content"]["sections"][3]["blocks"] = [
            {"kind": "bullets", "text": None, "rows": [],
             "items": [" ".join(f"k{number}t{index}" for index in range(90)) for number in range(6)]}]
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [slot_repair(writer, 0)], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        self.assertEqual(data["content_origin"], "structured_fallback")
        component = data["unit"]["components"][0]
        self.assertTrue(component["source_locked_fallback"])
        self.assertEqual(re.findall(r"<tr><td>([^<]*)</td>", component["html"]), ["Mindset cần bỏ", "Khát vọng mới"])
        self.assertIn("<h3>Tự kiểm tra</h3>", component["html"])


RUN_MCQ = json.loads((Path(__file__).parent / "fixtures" / "qlt3_run_8de1c76b_mcq.json").read_text("utf-8"))


class SingleChoiceTests(StoryboardEndpointTestCase):
    """Q4: keys at D in 6 of 9 questions (independent shuffles), the key the longest option in 8 of 9 (a note
    only), three reworded leaks the 4-gram check missed."""

    @staticmethod
    def chapter(practice_units: list[int]) -> list[IdmLessonDesignV1]:
        """One lesson per entry, its practice question in the given unit (the others teach)."""

        lessons = []
        for number, practice_unit in enumerate(practice_units, start=1):
            units = []
            for unit_number in range(1, practice_unit + 1):
                components = [gm._component(1, "html", "explain", "Nội dung", ["cb_0005"])]
                if unit_number == practice_unit:
                    components.append(gm._component(2, "problem", "practice", "Câu hỏi", ["cb_0005"],
                                                    practice_id="pt_1"))
                units.append(gm._unit(unit_number, "practice_feedback", f"Unit {unit_number}", ["cb_0005"],
                                      components))
            lessons.append(IdmLessonDesignV1.model_validate(gm._lesson(
                f"lsn_{number:03d}", f"Bài {number}", "Mục tiêu bài", [practice([key(4, 2)])], units)))
        return lessons

    @staticmethod
    def served_letters(lessons: list[IdmLessonDesignV1], chapter_number: int) -> list[str]:
        """The key letter each question gets in W5: its place in the course plus its practice's shift."""

        letters = []
        for lesson_number, lesson in enumerate(lessons, start=1):
            for unit_number, unit in enumerate(lesson.units, start=1):
                for item in unit.components:
                    if item.type == "problem":
                        ordinal = cast("int", question_ordinal(
                            f"chapter_{chapter_number}.lesson_{lesson_number}.unit_{unit_number}"))
                        letters.append(OPTION_LETTERS[(ordinal + practice_key_shift(item.practice_id)) % 4])
        return letters

    def test_w4_cycles_the_keys_of_a_chapter(self) -> None:
        # Q4: the practice questions of chapter 2 of the run sat in units 2, 1, 1, 1, 2 of its lessons. With the
        # place in the course alone two of them shared a letter; W4 numbers each practice so the keys cycle.
        lessons = self.chapter([2, 1, 1, 1, 2])
        plain = self.served_letters(lessons, 2)
        self.assertEqual(plain, ["D", "C", "D", "A", "D"])
        encoded, renumbered = encode_key_letters(lessons, chapter_order=1, first_letter=3)
        self.assertEqual(self.served_letters(encoded, 2), ["D", "A", "B", "C", "D"])
        self.assertEqual(renumbered, 3)
        self.assertEqual([lesson.practice_tasks[0].practice_id for lesson in encoded],
                         ["pt_1", "pt_3", "pt_3", "pt_3", "pt_1"])
        # Every component follows the renumbered practice; nothing else changes.
        for lesson in encoded:
            referenced = {c.practice_id for unit in lesson.units for c in unit.components if c.practice_id}
            self.assertEqual(referenced, {lesson.practice_tasks[0].practice_id})
        self.assertEqual([lesson.model_dump(exclude={"practice_tasks", "units"}) for lesson in encoded],
                         [lesson.model_dump(exclude={"practice_tasks", "units"}) for lesson in lessons])
        # A design written before the encoding (pt_1 everywhere) keeps the plain letters of its place.
        self.assertEqual(practice_key_shift("pt_1"), 0)
        self.assertEqual(practice_key_shift(None), 0)

    async def test_the_run_design_gets_balanced_keys(self) -> None:
        # The golden chapters through the real W4 path: the keys of a chapter's practice questions cycle.
        result, _, runtime = await design_module(1, [gm.MODULE_RESPONSES["mod_02"]])
        lessons = [IdmLessonDesignV1.model_validate(item) for item in result["shard"]["idm_design"]["lessons"]]
        self.assertEqual(len(self.served_letters(lessons, 2)), 1)
        self.assertLessEqual(runtime.adjustments["IDM_W4_KEY_LETTERS_ENCODED"], 1)

    def test_the_place_in_the_course_alone_never_repeats_a_letter_three_times(self) -> None:
        # Knowledge checks (no practice) and later shards use the place in the course only. The lessons of a
        # chapter cycle the letters; two consecutive lessons never share one.
        self.assertEqual([OPTION_LETTERS[cast("int", question_ordinal(f"chapter_2.lesson_{n}.unit_1")) % 4]
                          for n in range(1, 6)], ["B", "C", "D", "A", "B"])
        for first, second in itertools.product(range(1, 4), repeat=2):
            for lesson in range(1, 6):
                self.assertNotEqual(
                    cast("int", question_ordinal(f"chapter_1.lesson_{lesson}.unit_{first}")) % 4,
                    cast("int", question_ordinal(f"chapter_1.lesson_{lesson + 1}.unit_{second}")) % 4)
        # Never three in a row for any chapter of four lessons of 1-3 units with a question in every, the first or
        # the last unit.
        for counts in itertools.product((1, 2, 3), repeat=4):
            for pick in (lambda n: range(1, n + 1), lambda n: [1], lambda n: [n]):
                sequence = [cast("int", question_ordinal(f"chapter_{chapter}.lesson_{lesson}.unit_{unit}")) % 4
                            for chapter in (1, 2, 3) for lesson, count in enumerate(counts, start=1)
                            for unit in pick(count)]
                self.assertFalse(any(a == b == c for a, b, c in zip(sequence, sequence[1:], sequence[2:],
                                                                     strict=False)), counts)
        self.assertIsNone(question_ordinal("lesson_1"))

    def test_the_key_goes_to_its_letter_and_the_explanation_follows(self) -> None:
        component = {"problem_type": "multiple_choice", "question": "Khiếu nại này thuộc cấp nào?", "choices": [
            {"text": "Cấp 1 vì chưa có thiệt hại", "correct": False},
            {"text": "Cấp 2 vì khách phàn nàn lần thứ hai", "correct": True},
            {"text": "Cấp 3 vì cần escalate ngay", "correct": False},
            {"text": "Không thuộc cấp nào vì chỉ là góp ý", "correct": False}],
            "explanation": "A - sai vì chưa đủ; B - đúng vì phàn nàn lần thứ hai; C - sai vì không có yếu tố an toàn; "
                           "D - sai vì khách có yêu cầu xử lý."}
        for ordinal in range(8):
            served, codes = normalize_single_choice(component, "cp2_" + "c" * 32, ordinal)
            key_ = next(index for index, choice in enumerate(served["choices"]) if choice["correct"])
            self.assertEqual(key_, ordinal % 4)
            self.assertEqual(served["choices"][key_]["text"], "Cấp 2 vì khách phàn nàn lần thứ hai")
            self.assertIn(f"{OPTION_LETTERS[key_]} - đúng vì phàn nàn lần thứ hai", served["explanation"])
            self.assertEqual(codes["w5_problem_key_position_balanced"], 1)
        # Without a single key (or an ordinal) the seeded shuffle of QLT-2 applies unchanged.
        two_keys = {**component, "choices": [{**choice, "correct": True} for choice in component["choices"]]}
        self.assertEqual(normalize_single_choice(two_keys, "cp2_" + "c" * 32, 3)[0]["choices"],
                         [two_keys["choices"][old] for old in option_order("cp2_" + "c" * 32, 4)])

    def test_reworded_leaks_of_the_run_are_found_and_the_others_are_not(self) -> None:
        found = {question["unit"] for question in RUN_MCQ["questions"]
                 if reworded_key({"question": question["question"], "choices": question["choices"]},
                                 question["preceding_html_text"])}
        # The three leaks of the QC report, plus the worksheet check whose key copies the template's confirmation
        # sentence ("định hướng hành động thật … không phải lý thuyết đối phó").
        self.assertEqual(found, {*RUN_MCQ["leaks_per_qc_report"], "c3.l1.u1"})
        # A taught rule applied to the question's case is not a leak (the golden fixture of QLT-2, N3).
        rule = ("Khiếu nại thuộc cấp 2 khi khách hàng phàn nàn lần thứ hai.")
        golden = {"question": "Khách hàng gọi lần thứ hai vì giao trễ. Khiếu nại này thuộc cấp độ nào?", "choices": [
            {"text": "Cấp 1 vì chưa có thiệt hại tài chính", "correct": False},
            {"text": "Cấp 2 vì khách hàng phàn nàn lần thứ hai", "correct": True},
            {"text": "Cấp 3 vì cần escalate ngay cho quản lý", "correct": False}]}
        self.assertFalse(reworded_key(golden, rule))
        self.assertFalse(reworded_key(golden, ""))

    async def test_a_reworded_key_is_repaired_like_a_copied_one(self) -> None:
        body = worksheet_body()
        writer = worksheet_writer(body)
        # The worked example is restated in the key in another order (few 4-grams survive) while the distractors
        # are new: only the reworded-key check finds it.
        writer["components"]["c0"]["semantic_content"]["sections"][2]["blocks"][0]["text"] = (
            "Ví dụ mẫu: khách quen gọi lại lần hai vì hàng giao muộn; nhân viên ghi nhận cấp hai rồi báo trưởng "
            "nhóm.")
        choices = writer["components"]["c1"]["choices"]
        choices[1]["text"] = "Báo trưởng nhóm, ghi nhận cấp hai: khách quen gọi lại lần hai vì hàng giao muộn"
        choices[2]["text"] = "Cấp 3 vì khách dọa đăng lên mạng xã hội"
        fixed = slot_repair(writer, 1, choices=[
            {"text": "Cấp 1 vì chưa có thiệt hại tài chính", "correct": False},
            {"text": "Cấp 2, vì cùng một người đã gọi lại để khiếu nại", "correct": True},
            {"text": "Cấp 3 vì khách dọa đăng lên mạng xã hội", "correct": False}])
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [fixed], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        self.assertIn('{"code":"IDM_W5_ANSWER_LEAK","path":"components[1]"}', provider.calls[1]["prompt"])
        quality = IdmUnitQualityV1.model_validate(data["unit"]["idm_quality"])
        self.assertIn("IDM_W5_ANSWER_LEAK", quality.author_note.split("Đã tự sửa: ")[1])

    async def test_a_length_cue_the_repair_fixes_is_settled(self) -> None:
        body = escalate_body()
        writer = escalate_writer(body)
        writer["components"]["c0"]["choices"][0]["text"] = "Hứa hoàn tiền ngay để giữ chân khách VIP"
        writer["components"]["c0"]["choices"][2]["text"] = "Chỉ thông báo trưởng nhóm vì đây là khách VIP"
        fixed = slot_repair(escalate_writer(body), 0)
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [fixed], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        self.assertEqual(provider.names, [WRITER, REPAIR, JUDGE])
        quality = self.assert_envelope(data, "provider_validated", "validated", "provider")
        self.assertIn("Đã tự sửa: IDM_W5_ANSWER_LENGTH_CUE.", quality.author_note)
        self.assertNotIn("Cần xem (IDM_W5_ANSWER_LENGTH_CUE", quality.author_note)

    async def test_a_note_repair_that_breaks_the_question_is_dropped(self) -> None:
        body = escalate_body()
        writer = escalate_writer(body)
        writer["components"]["c0"]["choices"][0]["text"] = "Hứa hoàn tiền ngay để giữ chân khách VIP"
        writer["components"]["c0"]["choices"][2]["text"] = "Chỉ thông báo trưởng nhóm vì đây là khách VIP"
        broken = slot_repair(writer, 0, explanation="Đúng!")
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [broken], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "validated", "provider")
        self.assertEqual(data["unit"]["components"][0]["explanation"],
                         served(body, writer, 0)["explanation"])
        self.assertIn("Cần xem (IDM_W5_ANSWER_LENGTH_CUE", quality.author_note)


class PracticeTaskBoundTests(unittest.TestCase):
    def test_policy_bound_matches_the_contract(self) -> None:
        schema = IdmPracticeTaskV1.model_json_schema()["properties"]["criteria_fact_keys"]
        self.assertEqual(schema["maxItems"], IDM_PRACTICE_MAX_CRITERIA_FACTS)


if __name__ == "__main__":
    unittest.main()
