"""IDM ``chapter_blueprint`` (W3 treatment + W4 lesson layout) on the golden fixture (spec §6.7, §7.6, §8.2).

Every provider call goes to ``FakeIdmProvider``; nothing reaches the network.
"""

from __future__ import annotations

import dataclasses
import json
import unittest
from collections.abc import Callable
from typing import Any

from app.idm.contracts import (
    IdmLessonDesignV1,
    IdmLessonPlanV1,
    IdmModuleContextV1,
    IdmShardDesignV1,
    design_hash_of,
)
from app.idm.module_design import PRACTICE_COMPONENT_TYPES, normalize_lesson, run_idm_module_design, validate_lesson
from app.idm.module_layout import ModuleScope, build_module_scope, fallback_lesson
from app.idm.policy import AI_DRAFTED_MARKER_EN, AI_DRAFTED_MARKER_VI
from app.idm.runtime import IdmProviderError, IdmStageError
from app.idm.validation import IdmIssue
from app.instructional_quality import build_source_grounded_single_choice
from app.lesson_author_orchestration_v2 import ChapterBlueprintShardV2
from tests import idm_golden as g
from tests import idm_golden_module as gm
from tests.idm_golden import FakeIdmProvider, key, keys
from tests.idm_test_support import assert_node_trace, golden_design, make_runtime, module_inputs, module_response

STAGE = "IdmW3W4ModuleResponseV1"
ALLOWED = set(gm.ALLOWED_TYPES)
MODULES = ("mod_01", "mod_02", "mod_03")


def edited(module_key: str, change: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    payload = module_response(module_key)
    change(payload)
    return payload


def lesson_of(payload: dict[str, Any], position: int) -> IdmLessonDesignV1:
    return IdmLessonDesignV1.model_validate(payload["lessons"][position])


def scope_of(chapter_index: int) -> ModuleScope:
    context, plan, facts = module_inputs(chapter_index)
    return build_module_scope(context, plan, facts)


def check(chapter_index: int, payload: dict[str, Any], position: int = 0, *,
          allowed: set[str] = ALLOWED, scope: ModuleScope | None = None) -> list[IdmIssue]:
    scope = scope or scope_of(chapter_index)
    return validate_lesson(lesson_of(payload, position), scope.lesson_plans[position], scope, allowed,
                           f"lessons[{position}]")


def error_codes(issues: list[IdmIssue]) -> set[str]:
    return {issue.code for issue in issues if issue.severity == "error"}


def component(ctype: str, role: str, block_ids: list[str], *, practice_id: str | None = None) -> dict[str, Any]:
    return gm._component(1, ctype, role, f"Thành phần {ctype}", block_ids, practice_id=practice_id)


async def design_module(chapter_index: int, responses: list[Any], *, seconds: float = 500.0,
                        context_change: Callable[[dict[str, Any]], None] | None = None,
                        ) -> tuple[dict[str, Any], FakeIdmProvider, Any]:
    design_context, plan, facts = module_inputs(chapter_index)
    if context_change is not None:
        raw = design_context.model_dump(mode="json")
        context_change(raw)
        design_context = IdmModuleContextV1.model_validate(raw)
    provider = FakeIdmProvider({STAGE: responses})
    runtime = make_runtime(provider, seconds=seconds)
    _, skeleton = golden_design()
    result = await run_idm_module_design(context=design_context, skeleton=skeleton, plan=plan, facts=facts,
                                         runtime=runtime)
    return result, provider, runtime


class ValidateLessonTests(unittest.TestCase):
    def test_every_golden_lesson_is_valid(self) -> None:
        for chapter_index, module_key in enumerate(MODULES):
            payload = module_response(module_key)
            for position in range(len(payload["lessons"])):
                self.assertEqual(check(chapter_index, payload, position), [], (module_key, position))

    def test_practice_codes(self) -> None:
        def no_practice(p: dict[str, Any]) -> None:
            p["lessons"][0]["practice_tasks"] = []
            p["lessons"][0]["units"][0]["components"].pop()

        def no_criteria(p: dict[str, Any]) -> None:
            p["lessons"][0]["practice_tasks"][0]["criteria_fact_keys"] = []

        def foreign(p: dict[str, Any]) -> None:
            p["lessons"][0]["practice_tasks"][0]["criteria_fact_keys"].append(key(4, 2))

        def no_component(p: dict[str, Any]) -> None:
            p["lessons"][0]["units"][0]["components"].pop()

        def held_with_component(p: dict[str, Any]) -> None:
            p["lessons"][0]["practice_tasks"][0].update(hold=True, hold_question="Nhóm nào đúng với phản ánh này?")

        cases = {
            "IDM_W3_PRACTICE_MISSING": no_practice,
            "IDM_W3_PRACTICE_CRITERIA_MISSING": no_criteria,
            "IDM_W3_PRACTICE_CRITERIA_FOREIGN": foreign,
            "IDM_W3_PRACTICE_COMPONENT_MISSING": no_component,
            "IDM_W3_HELD_PRACTICE_HAS_COMPONENT": held_with_component,
        }
        for code, change in cases.items():
            with self.subTest(code=code):
                self.assertEqual(error_codes(check(0, edited("mod_01", change))), {code})

    def test_unknown_practice_reference_and_duplicate_practice_ids(self) -> None:
        def wrong_id(p: dict[str, Any]) -> None:
            p["lessons"][0]["units"][0]["components"][1]["practice_id"] = "pt_2"

        def not_a_practice_type(p: dict[str, Any]) -> None:
            p["lessons"][0]["units"][0]["components"][1]["type"] = "la_faq"

        def duplicate(p: dict[str, Any]) -> None:
            p["lessons"][0]["practice_tasks"].append(dict(p["lessons"][0]["practice_tasks"][0]))

        for change in (wrong_id, not_a_practice_type):
            issues = check(0, edited("mod_01", change))
            self.assertIn(IdmIssue("IDM_W3_PRACTICE_UNKNOWN", "lessons[0].units[0].components[1].practice_id"), issues)
            self.assertIn("IDM_W3_PRACTICE_COMPONENT_MISSING", error_codes(issues))
        issues = check(0, edited("mod_01", duplicate))
        self.assertIn(IdmIssue("IDM_W3_PRACTICE_UNKNOWN", "lessons[0].practice_tasks"), issues)
        self.assertEqual(PRACTICE_COMPONENT_TYPES, {"problem", "la_sortable", "la_crossword"})

    def test_sentence_format_is_a_warning_in_both_locales(self) -> None:
        def no_shape(p: dict[str, Any]) -> None:
            p["lessons"][0]["practice_tasks"][0]["sentence"] = "Phân loại đúng nhóm của mọi khiếu nại tiếp nhận"

        issues = check(0, edited("mod_01", no_shape))
        self.assertEqual(issues, [IdmIssue("IDM_W3_PRACTICE_SENTENCE_FORMAT",
                                           "lessons[0].practice_tasks[0].sentence", "warning")])
        english = dataclasses.replace(scope_of(0), locale="en")
        self.assertEqual([issue.code for issue in check(0, module_response("mod_01"), scope=english)],
                         ["IDM_W3_PRACTICE_SENTENCE_FORMAT"])

        def english_sentence(p: dict[str, Any]) -> None:
            p["lessons"][0]["practice_tasks"][0]["sentence"] = (
                "Given a customer report, the learner decides its group to route it correctly")

        self.assertEqual(check(0, edited("mod_01", english_sentence), scope=english), [])

    def test_layout_codes(self) -> None:
        def partition(p: dict[str, Any]) -> None:
            unit = p["lessons"][0]["units"][0]
            unit["block_ids"] = ["cb_0003"]
            for item in unit["components"]:
                item["block_ids"] = ["cb_0003"]

        def outside(p: dict[str, Any]) -> None:
            p["lessons"][0]["units"][0]["components"][0]["block_ids"] = ["cb_0009", "cb_0011"]

        def uncovered(p: dict[str, Any]) -> None:
            for item in p["lessons"][0]["units"][0]["components"]:
                item["block_ids"] = ["cb_0003"]

        def html_not_first(p: dict[str, Any]) -> None:
            p["lessons"][0]["units"][0]["components"].reverse()

        def faq_not_last(p: dict[str, Any]) -> None:
            p["lessons"][0]["units"][1]["components"].reverse()

        def duplicate_type(p: dict[str, Any]) -> None:
            p["lessons"][0]["units"][0]["components"].append(component("html", "show", ["cb_0009"]))

        cases = [("mod_01", partition, "IDM_W4_UNIT_BLOCK_PARTITION"),
                 ("mod_02", outside, "IDM_W4_COMPONENT_BLOCKS_OUTSIDE_UNIT"),
                 ("mod_01", uncovered, "IDM_W4_UNIT_BLOCKS_UNCOVERED"),
                 ("mod_01", html_not_first, "IDM_W4_COMPONENT_ORDER"),
                 ("mod_02", faq_not_last, "IDM_W4_COMPONENT_ORDER"),
                 ("mod_02", duplicate_type, "IDM_W4_COMPONENT_ORDER")]
        for module_key, change, code in cases:
            with self.subTest(change=change.__name__):
                self.assertEqual(error_codes(check(MODULES.index(module_key), edited(module_key, change))), {code})

    def test_format_evidence_for_sortable_crossword_and_diagram(self) -> None:
        for ctype in ("la_sortable", "la_crossword"):
            def practice_type(p: dict[str, Any], ctype: str = ctype) -> None:
                p["lessons"][0]["units"][0]["components"][1]["type"] = ctype

            issues = check(0, edited("mod_01", practice_type))
            self.assertEqual(issues, [IdmIssue("IDM_W4_FORMAT_EVIDENCE", "lessons[0].units[0].components[1].type")])

        def diagram_without_relations(p: dict[str, Any]) -> None:
            p["lessons"][0]["units"][0]["components"][0].update(type="la_diagram", role="show")

        self.assertEqual(error_codes(check(1, edited("mod_02", diagram_without_relations))),
                         {"IDM_W4_FORMAT_EVIDENCE"})

        def diagram_with_relations(p: dict[str, Any]) -> None:
            p["lessons"][1]["units"][0]["components"].insert(1, component("la_diagram", "show", ["cb_0005"]))

        self.assertEqual(check(0, edited("mod_01", diagram_with_relations), 1), [])

    def test_practice_before_support(self) -> None:
        def early(p: dict[str, Any]) -> None:
            lesson = p["lessons"][0]
            lesson["practice_tasks"][0]["criteria_fact_keys"] = keys(8, 2, 3)
            first, second = lesson["units"]
            first["components"].append(second["components"].pop(0) | {"block_ids": ["cb_0009"]})

        issues = check(1, edited("mod_02", early))
        self.assertEqual(issues, [IdmIssue("IDM_W4_PRACTICE_BEFORE_SUPPORT", "lessons[0].practice_tasks[0]")])

    def test_theory_run_is_a_warning(self) -> None:
        def long_theory(p: dict[str, Any]) -> None:
            lesson = p["lessons"][0]
            lesson["units"] = [
                gm._unit(1, "context_explain", "Bốn bước đầu", ["cb_0006"], [
                    component("html", "explain", ["cb_0006"]), component("la_faq", "clarify", ["cb_0006"])]),
                gm._unit(2, "example", "Bốn bước tiếp", ["cb_0007"], [
                    component("html", "show", ["cb_0007"]), component("la_faq", "clarify", ["cb_0007"])]),
                gm._unit(3, "practice_feedback", "Hoàn tất", ["cb_0008"], [
                    component("html", "explain", ["cb_0008"]), component("problem", "show", ["cb_0008"]),
                    component("la_sortable", "practice", ["cb_0008"], practice_id="pt_1")]),
            ]

        issues = check(2, edited("mod_03", long_theory))
        self.assertEqual(issues, [IdmIssue("IDM_W4_THEORY_RUN", "lessons[0].units[2].components[1]", "warning")])

    def test_type_not_allowed_and_missing_purpose(self) -> None:
        issues = check(0, module_response("mod_01"), allowed=ALLOWED - {"problem"})
        self.assertEqual(error_codes(issues), {"IDM_W4_COMPONENT_TYPE_NOT_ALLOWED"})

        def no_purpose(p: dict[str, Any]) -> None:
            p["lessons"][0]["units"][0]["components"][0]["author_review"]["purpose"] = None

        self.assertEqual(check(0, edited("mod_01", no_purpose)),
                         [IdmIssue("IDM_W4_AUTHOR_REVIEW_PURPOSE_MISSING",
                                   "lessons[0].units[0].components[0].author_review")])


class NormalizeLessonTests(unittest.TestCase):
    def plan(self, chapter_index: int, position: int) -> IdmLessonPlanV1:
        return scope_of(chapter_index).lesson_plans[position]

    def test_renumbers_and_sanitises_text(self) -> None:
        def messy(p: dict[str, Any]) -> None:
            lesson = p["lessons"][0]
            lesson["title"] = "Quyết định <escalate>\nngay"
            lesson["notes"] = "Ghi chú <b>quan trọng</b>"
            for unit_index, unit in enumerate(lesson["units"]):
                unit["unit_index"] = 7 + unit_index
                for component_index, item in enumerate(unit["components"]):
                    item["component_index"] = 3 + component_index
                    item["rationale"] = "Dòng 1\nDòng <2>"

        lesson = normalize_lesson(lesson_of(edited("mod_02", messy), 0), self.plan(1, 0), "vi", [])
        self.assertEqual([unit.unit_index for unit in lesson.units], [1, 2])
        self.assertEqual([[c.component_index for c in unit.components] for unit in lesson.units], [[1], [1, 2]])
        self.assertEqual(lesson.title, "Quyết định ‹escalate› ngay")  # noqa: RUF001 - sanitised glyphs
        self.assertEqual(lesson.units[0].components[0].rationale, "Dòng 1 Dòng ‹2›")  # noqa: RUF001
        dumped = json.dumps(lesson.model_dump(mode="json"), ensure_ascii=False)
        self.assertNotIn("<", dumped)
        self.assertNotIn(">", dumped)

    def test_ai_drafted_marker_only_on_ai_drafted_practice_components(self) -> None:
        plan = self.plan(1, 0)
        lesson = normalize_lesson(lesson_of(module_response("mod_02"), 0), plan, "vi", [])
        practice = lesson.units[1].components[0].author_review.example_scenario
        self.assertEqual(practice, f"{AI_DRAFTED_MARKER_VI} Một tình huống khiếu nại cụ thể")
        self.assertIsNone(lesson.units[0].components[0].author_review.example_scenario)
        self.assertIsNone(lesson.units[1].components[1].author_review.example_scenario)

        def scenario(p: dict[str, Any]) -> None:
            p["lessons"][0]["units"][1]["components"][0]["author_review"]["example_scenario"] = "Khách VIP gọi"

        english = normalize_lesson(lesson_of(edited("mod_02", scenario), 0), plan, "en", [])
        self.assertEqual(english.units[1].components[0].author_review.example_scenario,
                         f"{AI_DRAFTED_MARKER_EN} Khách VIP gọi")
        again = normalize_lesson(english, plan, "en", [])
        self.assertEqual(again.units[1].components[0].author_review.example_scenario,
                         f"{AI_DRAFTED_MARKER_EN} Khách VIP gọi")
        source = normalize_lesson(lesson_of(module_response("mod_03"), 0), self.plan(2, 0), "vi", [])
        self.assertIsNone(source.units[1].components[1].author_review.example_scenario)

    def test_job_aid_drops_practices_and_warnings_reach_notes(self) -> None:
        def job_aid_practice(p: dict[str, Any]) -> None:
            p["lessons"][1]["practice_tasks"] = [p["lessons"][0]["practice_tasks"][0]]

        job_aid = normalize_lesson(lesson_of(edited("mod_03", job_aid_practice), 1), self.plan(2, 1), "vi", [])
        self.assertEqual(job_aid.practice_tasks, [])
        warnings = ["IDM_W4_THEORY_RUN", "IDM_W3_PRACTICE_SENTENCE_FORMAT"]
        vi = normalize_lesson(lesson_of(module_response("mod_01"), 0), self.plan(0, 0), "vi", warnings)
        self.assertTrue(vi.notes.startswith("Bloom: Vận dụng"))
        self.assertIn("Cảnh báo: có hơn 5 khối giải thích liên tiếp", vi.notes)
        self.assertIn("Lưu ý: câu Practice chưa đủ", vi.notes)
        en = normalize_lesson(lesson_of(module_response("mod_01"), 0), self.plan(0, 0), "en", warnings)
        self.assertIn("Warning: more than 5 explanation blocks", en.notes)
        self.assertIn("Note: a practice sentence lacks", en.notes)
        plain = normalize_lesson(lesson_of(module_response("mod_01"), 0), self.plan(0, 0), "vi", [])
        self.assertEqual(plain.notes, "Bloom: Vận dụng · ước tính 4 khối.")


class ModuleScopeTests(unittest.TestCase):
    def test_golden_scope_lookups(self) -> None:
        scope = scope_of(2)
        self.assertEqual(scope.locale, "vi")
        self.assertEqual([plan.lesson_key for plan in scope.lesson_plans], ["lsn_004", "lsn_005"])
        self.assertEqual(scope.lesson_fact_keys(scope.lesson_plans[0]), set(keys(5, 1, 13)))
        self.assertEqual(scope.must_do_statement["md_4"], "Thực hiện các bước tiếp nhận khiếu nại đúng trình tự")

    def test_scope_and_fact_mismatches_are_rejected(self) -> None:
        context, plan, facts = module_inputs(2)
        raw = context.model_dump(mode="json")

        def variant(change: Callable[[dict[str, Any]], None]) -> IdmModuleContextV1:
            copy = json.loads(json.dumps(raw, ensure_ascii=False))
            change(copy)
            return IdmModuleContextV1.model_validate(copy)

        def duplicate(c: dict[str, Any]) -> None:
            c["module"]["lessons"][1]["block_ids"].append("cb_0006")

        def unknown(c: dict[str, Any]) -> None:
            c["module"]["lessons"][1]["block_ids"].append("cb_0099")

        def no_row(c: dict[str, Any]) -> None:
            c["blueprint"] = [row for row in c["blueprint"] if row["block_id"] != "cb_0013"]

        foreign_fact = facts[0].model_copy(update={"fact_key": key(4, 2), "scope_key": "scope3_04"})
        cases = [
            (variant(duplicate), plan, facts), (variant(unknown), plan, facts), (variant(no_row), plan, facts),
            (context, plan.model_copy(update={"source_scope_ids": list(reversed(plan.source_scope_ids))}), facts),
            (context, plan, [*facts, foreign_fact]),
            (context, plan, facts[:-1]),
            (context, plan, [*facts, facts[0].model_copy(update={"fact_key": "extra"})]),
        ]
        for index, (item_context, item_plan, item_facts) in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(IdmStageError) as caught:
                build_module_scope(item_context, item_plan, item_facts)
            self.assertEqual(caught.exception.code, "IDM_MODULE_CONTEXT_INVALID")


class FallbackLessonTests(unittest.TestCase):
    def test_large_lesson_groups_blocks_and_job_aid_has_no_practice(self) -> None:
        scope = scope_of(2)
        extra = [f"cb_{n:04d}" for n in range(100, 111)]
        wide = dataclasses.replace(
            scope,
            blocks={**scope.blocks, **{block: scope.blocks["cb_0006"].model_copy(update={"block_id": block})
                                       for block in extra}},
            rows={**scope.rows, **dict.fromkeys(extra, scope.rows["cb_0006"])})
        blocks = ["cb_0006", "cb_0007", *extra]
        plan = scope.lesson_plans[0].model_copy(update={"block_ids": blocks})
        lesson = fallback_lesson(plan, wide)
        self.assertEqual(len(lesson.units), 7)
        self.assertEqual([block for unit in lesson.units for block in unit.block_ids], blocks)
        job_aid = fallback_lesson(scope.lesson_plans[1], dataclasses.replace(scope, locale="en"))
        self.assertEqual(job_aid.practice_tasks, [])
        self.assertEqual({unit.segment for unit in job_aid.units}, {"job_aid"})
        self.assertEqual({c.role for unit in job_aid.units for c in unit.components}, {"job_aid"})
        self.assertIn("Automatic fallback layout", job_aid.notes)

    def test_practice_targets_last_unit_without_primary_must_do(self) -> None:
        scope = scope_of(2)
        plan = scope.lesson_plans[0].model_copy(update={"primary_must_do_id": "md_9"})
        lesson = fallback_lesson(plan, scope)
        self.assertEqual([c.type for c in lesson.units[-1].components], ["html", "problem"])
        self.assertEqual(lesson.practice_tasks[0].criteria_fact_keys, keys(5, 10, 13))


class RunModuleDesignTests(unittest.IsolatedAsyncioTestCase):
    async def test_golden_modules_are_provider_validated(self) -> None:
        for chapter_index, module_key in enumerate(MODULES):
            result, provider, runtime = await design_module(chapter_index, [module_response(module_key)])
            shard = result["shard"]
            design = shard["idm_design"]
            self.assertEqual((result["content_origin"], result["quality_state"], result["usage_source"]),
                             ("provider_validated", "validated", "provider"))
            self.assertTrue(result["usage_complete"])
            ChapterBlueprintShardV2.model_validate({k: v for k, v in shard.items() if k != "idm_design"})
            IdmShardDesignV1.model_validate(design)
            self.assertEqual(design["design_hash"], design_hash_of(design))
            self.assertEqual(design["stage_origin"], "provider")
            self.assertEqual(shard["assessment_obligations"], [])
            self.assertEqual(len(provider.calls), 1)
            self.assertEqual(provider.calls[0]["thinking_level"], "high")
            assert_node_trace(self, runtime.trace)
            plans = scope_of(chapter_index).lesson_plans
            for lesson, projected, plan in zip(design["lessons"], shard["lessons"], plans, strict=True):
                roles = [c["role"] for unit in lesson["units"] for c in unit["components"]]
                if plan.kind == "job_aid":
                    self.assertNotIn("practice", roles)
                    self.assertEqual(projected["learning_activities"],
                                     [f"Tra cứu {lesson['title']} khi thực hiện công việc"])
                    self.assertEqual(projected["assessment"], "Không đánh giá — tài liệu hỗ trợ khi làm việc")
                else:
                    self.assertIn("practice", roles)
                    self.assertEqual(projected["learning_activities"],
                                     [practice["sentence"] for practice in lesson["practice_tasks"]])

    async def test_partial_salvage_falls_back_only_for_the_invalid_lesson(self) -> None:
        def foreign(p: dict[str, Any]) -> None:
            p["lessons"][1]["practice_tasks"][0]["criteria_fact_keys"].append(key(2, 2))

        bad = edited("mod_01", foreign)
        result, provider, runtime = await design_module(0, [bad, bad])
        design = result["shard"]["idm_design"]
        self.assertEqual(design["stage_origin"], "partial_fallback")
        self.assertEqual((result["content_origin"], result["quality_state"]),
                         ("structured_fallback", "review_required"))
        self.assertEqual(design["lessons"][0]["practice_tasks"][0]["sentence"],
                         bad["lessons"][0]["practice_tasks"][0]["sentence"])
        self.assertIn("Bố cục dự phòng tự động", design["lessons"][1]["notes"])
        self.assertEqual(design["lessons"][1]["practice_tasks"][0]["criteria_fact_keys"], keys(4, 2, 5))
        self.assertTrue(design["lessons"][1]["practice_tasks"][0]["hold"])
        self.assertEqual(len(result["shard"]["assessment_obligations"]), 1)
        self.assertEqual([call["schema"] for call in provider.calls], [STAGE, STAGE])
        first, second = (call["prompt"] for call in provider.calls)
        suffix = second[len(first):]
        self.assertTrue(second.startswith(first))
        self.assertIn('"IDM_W3_PRACTICE_CRITERIA_FOREIGN"', suffix)
        self.assertIn("lessons[1].practice_tasks[0].criteria_fact_keys", suffix)
        for fact in g.source_facts():
            self.assertNotIn(fact.fact_text, suffix)
        self.assertNotIn(key(2, 2), suffix)
        self.assertEqual([(e["invocation_kind"], e["outcome"]) for e in runtime.trace],
                         [("writer", "succeeded"), ("repair", "succeeded"), ("deterministic", "fallback")])
        self.assertEqual(runtime.trace[-1]["failure_code"], "IDM_MODULE_LESSON_FALLBACK")
        assert_node_trace(self, runtime.trace)

    async def test_invalid_answers_fall_back_for_every_lesson_with_grounded_checks(self) -> None:
        grounded_lessons = []
        for chapter_index in range(3):
            result, provider, runtime = await design_module(chapter_index, ["{}", "not json"])
            design = result["shard"]["idm_design"]
            self.assertEqual(design["stage_origin"], "deterministic_fallback")
            self.assertEqual(result["content_origin"], "structured_fallback")
            self.assertEqual(result["usage_source"], "provider")
            self.assertIn('"code":"missing"', provider.calls[1]["prompt"])
            scope = scope_of(chapter_index)
            for lesson, plan in zip(design["lessons"], scope.lesson_plans, strict=True):
                problems = [(u, c) for u in lesson["units"] for c in u["components"] if c["type"] == "problem"]
                target = next((u for u in lesson["units"] if any(
                    plan.primary_must_do_id in scope.rows[b].must_do_ids for b in u["block_ids"])), lesson["units"][-1])
                grounded = build_source_grounded_single_choice(
                    plan.title, scope.block_texts(target["block_ids"]), locale="vi") is not None
                expected = plan.kind == "learning" and grounded
                self.assertEqual(bool(problems), expected, plan.lesson_key)
                grounded_lessons += [plan.lesson_key] if problems else []
                for unit, item in problems:
                    self.assertEqual(item["practice_id"], "pt_1")
                    criteria = set(lesson["practice_tasks"][0]["criteria_fact_keys"])
                    self.assertTrue(criteria <= {k for b in unit["block_ids"] for k in scope.blocks[b].fact_keys})
            ChapterBlueprintShardV2.model_validate({k: v for k, v in result["shard"].items() if k != "idm_design"})
            assert_node_trace(self, runtime.trace)
        self.assertEqual(grounded_lessons, ["lsn_002", "lsn_004"])

    async def test_invalid_then_valid_answer_is_provider_validated(self) -> None:
        result, provider, _ = await design_module(1, ["{}", module_response("mod_02")])
        self.assertEqual(result["shard"]["idm_design"]["stage_origin"], "provider")
        self.assertIn("REPAIR_REQUIREMENTS", provider.calls[1]["prompt"])

    async def test_missing_or_reordered_lessons(self) -> None:
        missing = edited("mod_01", lambda p: p["lessons"].pop())
        result, provider, _ = await design_module(0, [missing, missing])
        self.assertEqual(result["shard"]["idm_design"]["stage_origin"], "partial_fallback")
        self.assertIn('"IDM_W3_LESSON_SET_MISMATCH"', provider.calls[1]["prompt"])
        reordered = edited("mod_01", lambda p: p["lessons"].reverse())
        result, provider, _ = await design_module(0, [reordered])
        self.assertEqual([lesson["lesson_key"] for lesson in result["shard"]["idm_design"]["lessons"]],
                         ["lsn_001", "lsn_002"])
        self.assertEqual(len(provider.calls), 1)

    async def test_provider_failures_and_budget(self) -> None:
        transient = IdmProviderError("AI_PROVIDER_UNAVAILABLE", terminal=False)
        result, provider, runtime = await design_module(1, [transient])
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual((result["usage_source"], result["usage_complete"]), ("reserved_upper_bound", False))
        self.assertEqual(result["shard"]["idm_design"]["stage_origin"], "deterministic_fallback")
        self.assertEqual(runtime.trace[0]["failure_code"], "AI_PROVIDER_UNAVAILABLE")
        assert_node_trace(self, runtime.trace)
        with self.assertRaises(IdmProviderError) as caught:
            await design_module(1, [IdmProviderError("AI_PROVIDER_REQUEST_REJECTED", terminal=True)])
        self.assertTrue(caught.exception.terminal)
        result, provider, _ = await design_module(1, [], seconds=5)
        self.assertEqual(provider.calls, [])
        self.assertEqual(result["shard"]["idm_design"]["stage_origin"], "deterministic_fallback")

    async def test_held_practice_becomes_an_assessment_obligation(self) -> None:
        def hold(p: dict[str, Any]) -> None:
            lesson = p["lessons"][0]
            lesson["practice_tasks"][0].update(hold=True, hold_question="Khách VIP có yếu tố pháp lý thì sao?")
            lesson["units"][1]["components"].pop(0)

        result, _, _ = await design_module(1, [edited("mod_02", hold)])
        shard = result["shard"]
        self.assertEqual(result["content_origin"], "provider_validated")
        self.assertEqual(len(shard["assessment_obligations"]), 1)
        obligation = shard["assessment_obligations"][0]
        unit = shard["lessons"][0]["units"][obligation["unit_index"] - 1]
        self.assertLessEqual(obligation["component_index"], 3)
        self.assertEqual((obligation["lesson_index"], obligation["unit_index"], obligation["component_index"]),
                         (1, 1, 2))
        self.assertTrue(set(obligation["relevant_scope_ids"]) <= set(unit["source_scope_ids"]))
        self.assertTrue(set(obligation["learning_objective_refs"]) <= set(unit["learning_objective_refs"]))
        self.assertEqual(obligation["relevant_evidence_fact_ids"], keys(6, 1, 6))
        self.assertEqual((obligation["required_assessment_kind"], obligation["unresolved_reason"]),
                         ("single_choice", "ASSESSMENT_SOURCE_CHECK_REQUIRED"))
        self.assertRegex(obligation["planned_slot_key"], r"^ao2_[a-f0-9]{32}$")
        ChapterBlueprintShardV2.model_validate({k: v for k, v in shard.items() if k != "idm_design"})

    async def test_held_practice_beyond_slot_three_has_no_obligation(self) -> None:
        def hold_full_unit(p: dict[str, Any]) -> None:
            lesson = p["lessons"][0]
            lesson["practice_tasks"][0].update(hold=True)
            lesson["units"][0]["components"] += [component("problem", "show", ["cb_0009"]),
                                                 component("la_faq", "clarify", ["cb_0009"])]
            lesson["units"][1]["components"].pop(0)

        result, _, _ = await design_module(1, [edited("mod_02", hold_full_unit)])
        self.assertEqual(result["shard"]["idm_design"]["stage_origin"], "provider")
        self.assertEqual(result["shard"]["assessment_obligations"], [])

    # Regression: two held practices of one unit used to share one slot (duplicated planned_slot_key,
    # ChapterBlueprintShardV2 ValidationError -> HTTP 500). Each now takes its own component_index.
    async def test_two_held_practices_in_one_lesson_still_project_a_valid_shard(self) -> None:
        def holds(count: int) -> Callable[[dict[str, Any]], None]:
            def change(p: dict[str, Any]) -> None:
                lesson = p["lessons"][0]
                first = lesson["practice_tasks"][0]
                first.update(hold=True)
                lesson["practice_tasks"] += [{**first, "practice_id": f"pt_{n}"} for n in range(2, count + 1)]
                lesson["units"][1]["components"].pop(0)
            return change

        result, _, _ = await design_module(1, [edited("mod_02", holds(2))])
        obligations = result["shard"]["assessment_obligations"]
        self.assertEqual([(o["unit_index"], o["component_index"]) for o in obligations], [(1, 2), (1, 3)])
        self.assertEqual(len({o["planned_slot_key"] for o in obligations}), 2)
        result, _, _ = await design_module(1, [edited("mod_02", holds(3))])
        self.assertEqual([o["component_index"] for o in result["shard"]["assessment_obligations"]], [2, 3])

    # Regression (spec §7.6.4/§7.6.5): a lesson that still fails after the repair keeps the provider's
    # practice tasks as holds (criteria filtered to lesson facts) instead of a synthesised check.
    async def test_partial_fallback_keeps_the_provider_practice_as_hold(self) -> None:
        def bad_layout(p: dict[str, Any]) -> None:
            p["lessons"][1]["units"][0]["components"].reverse()
            p["lessons"][1]["practice_tasks"][0]["criteria_fact_keys"].append(key(2, 2))

        bad = edited("mod_01", bad_layout)
        result, _, _ = await design_module(0, [bad, bad])
        shard = result["shard"]
        lesson = shard["idm_design"]["lessons"][1]
        practice = lesson["practice_tasks"][0]
        self.assertEqual((practice["sentence"], practice["hold"]),
                         (bad["lessons"][1]["practice_tasks"][0]["sentence"], True))
        self.assertEqual(practice["criteria_fact_keys"], keys(4, 2, 5))
        self.assertIn("SME xác nhận tiêu chí", practice["hold_question"])
        self.assertNotIn("problem", [c["type"] for u in lesson["units"] for c in u["components"]])
        self.assertEqual([(o["lesson_index"], o["unit_index"], o["component_index"])
                          for o in shard["assessment_obligations"]], [(2, 1, 2)])
        ChapterBlueprintShardV2.model_validate({k: v for k, v in shard.items() if k != "idm_design"})

    # Regression (fixed, spec §7.6.3): a Job Aid lesson has ``practice_tasks=[]`` and only html ± la_faq
    # units. A job-aid practice component is now rejected (IDM_W3_JOB_AID_HAS_PRACTICE) so the shard never
    # carries a dangling ``practice_id`` or an assessed problem in a "not assessed" job aid.
    async def test_job_aid_lesson_never_keeps_a_practice_component(self) -> None:
        def job_aid_practice(p: dict[str, Any]) -> None:
            lesson = p["lessons"][1]
            lesson["practice_tasks"] = [gm._practice(
                "Cho một khiếu nại, người học chọn mã khiếu nại để ghi nhận đúng", keys(10, 2, 6))]
            lesson["units"][0]["components"].append(component("problem", "practice", ["cb_0013"], practice_id="pt_1"))

        bad = edited("mod_03", job_aid_practice)
        # The lesson is rejected (IDM_W3_JOB_AID_HAS_PRACTICE), repaired once, then laid out by fallback.
        result, _, _ = await design_module(2, [bad, bad])
        for lesson in result["shard"]["idm_design"]["lessons"]:
            practices = {item["practice_id"] for item in lesson["practice_tasks"]}
            referenced = {c["practice_id"] for u in lesson["units"] for c in u["components"] if c["practice_id"]}
            self.assertLessEqual(referenced, practices, lesson["lesson_key"])

if __name__ == "__main__":
    unittest.main()
