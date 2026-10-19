"""QLT-5 E: a practice format the facts give no evidence for is converted by the server.

In 3 consecutive real runs a lesson fell back to the deterministic design because W4 planned a PRACTICE la_sortable,
la_crossword or la_diagram the facts cannot support (IDM_W4_FORMAT_EVIDENCE at lessons[0].units[0].components[1].type)
and the provider repair failed too. A teaching slot of that kind was already folded into the html; a practice now
becomes a single-choice scenario question (a practice that decides, recognises or chooses) or a worksheet checked by a
question on a sample answer, keeping its practice task.

The provider is the local fake; nothing reaches the network.
"""

from __future__ import annotations

import unittest
from typing import Any

from app.idm.contracts import IdmLessonDesignV1
from app.idm.module_autofix import PRACTICE_FORMAT_CODE, autofix_lesson, practice_format_note
from app.idm.validation import IdmIssue, errors
from tests.golden import idm_golden_module as gm
from tests.unit.idm.test_idm_module_design import ALLOWED, check, component, design_module, edited, scope_of

FORMAT_EVIDENCE = IdmIssue("IDM_W4_FORMAT_EVIDENCE", "lessons[0].units[0].components[1].type")


def practice_as(kind: str) -> Any:
    """lsn_001 unit 1: its scenario practice (pt_1, "Chọn cách xử lý") planned as ``kind`` over blocks without steps,
    definitions or relations: the shape of the failing runs."""

    def change(payload: dict[str, Any]) -> None:
        payload["lessons"][0]["units"][0]["components"][1]["type"] = kind

    return change


def doing(kind: str) -> Any:
    """lsn_002 unit 1: its practice turned into producing a table (not a decision), planned as ``kind``."""

    def change(payload: dict[str, Any]) -> None:
        lesson = payload["lessons"][1]
        lesson["practice_tasks"][0].update(
            learner_action="Lập bảng xử lý cho từng cấp độ",
            sentence="Cho mô tả ba khiếu nại, người học lập bảng xử lý cho từng cấp độ để báo đúng người")
        lesson["units"][0]["components"][1]["type"] = kind

    return change


def fixed(change: Any, position: int = 0, allowed: set[str] = ALLOWED) -> tuple[IdmLessonDesignV1, Any]:
    scope = scope_of(0)
    lesson = IdmLessonDesignV1.model_validate(edited("mod_01", change)["lessons"][position])
    return autofix_lesson(lesson, scope.lesson_plans[position], scope, allowed)


class PracticeFormatTests(unittest.TestCase):
    def test_the_failing_shape_is_reproduced(self) -> None:
        for kind in ("la_sortable", "la_crossword"):
            with self.subTest(kind=kind):
                self.assertEqual(check(0, edited("mod_01", practice_as(kind))), [FORMAT_EVIDENCE])

    def test_a_deciding_practice_becomes_a_scenario_question(self) -> None:
        original = IdmLessonDesignV1.model_validate(edited("mod_01", lambda _p: None)["lessons"][0])
        for kind in ("la_sortable", "la_crossword", "la_diagram"):
            with self.subTest(kind=kind):
                lesson, codes = fixed(practice_as(kind))
                self.assertEqual(codes, {PRACTICE_FORMAT_CODE: 1})
                self.assertEqual(validate(lesson, 0), [])
                practice_slot = lesson.units[0].components[1]
                self.assertEqual((practice_slot.type, practice_slot.role, practice_slot.practice_id),
                                 ("problem", "practice", "pt_1"))
                self.assertIn("thay cho dạng", practice_slot.rationale)
                # The linked practice task (criteria, feedback focus) is untouched.
                self.assertEqual(lesson.practice_tasks, original.practice_tasks)
                self.assertIn("đã được đổi thành câu hỏi tình huống một lựa chọn", lesson.notes)

    def test_a_producing_practice_becomes_a_worksheet_with_its_check(self) -> None:
        lesson, codes = fixed(doing("la_crossword"), position=1)
        self.assertEqual(codes[PRACTICE_FORMAT_CODE], 1)
        unit = lesson.units[0]
        # The worksheet html takes over the teaching html of the unit (one html per unit) and gets its check.
        self.assertEqual([(c.type, c.role, c.practice_id) for c in unit.components],
                         [("html", "practice", "pt_1"), ("problem", "practice", "pt_1")])
        self.assertEqual(unit.components[0].title, "Phiếu thực hành: Lập bảng xử lý cho từng cấp độ")
        self.assertEqual(unit.components[1].title, "Đối chiếu bài mẫu với tiêu chí")
        self.assertEqual(validate(lesson, 1), [])
        self.assertIn("phiếu thực hành kèm câu hỏi đối chiếu bài mẫu", lesson.notes)

    def test_an_existing_check_of_the_same_practice_is_kept(self) -> None:
        def with_check(payload: dict[str, Any]) -> None:
            doing("la_sortable")(payload)
            unit = payload["lessons"][1]["units"][0]
            unit["components"].append(component("problem", "practice", ["cb_0005"], practice_id="pt_1"))
            unit["components"][1]["component_index"] = 2

        lesson, _codes = fixed(with_check, position=1)
        self.assertEqual([(c.type, c.practice_id) for c in lesson.units[0].components],
                         [("html", "pt_1"), ("problem", "pt_1")])

    def test_without_an_allowed_conversion_the_repair_decides(self) -> None:
        scope = scope_of(0)
        lesson = IdmLessonDesignV1.model_validate(edited("mod_01", practice_as("la_sortable"))["lessons"][0])
        no_question = {"html", "la_faq", "la_sortable", "la_crossword", "la_diagram"}
        self.assertEqual(autofix_lesson(lesson, scope.lesson_plans[0], scope, no_question), (lesson, {}))

        def held(payload: dict[str, Any]) -> None:
            practice_as("la_sortable")(payload)
            payload["lessons"][0]["practice_tasks"][0].update(hold=True, hold_question="Tiêu chí nào áp dụng ở đây?")

        held_lesson = IdmLessonDesignV1.model_validate(edited("mod_01", held)["lessons"][0])
        self.assertEqual(autofix_lesson(held_lesson, scope.lesson_plans[0], scope, ALLOWED), (held_lesson, {}))

    def test_the_sentence_names_the_action_when_the_learner_action_does_not(self) -> None:
        def sentence_only(payload: dict[str, Any]) -> None:
            practice_as("la_sortable")(payload)
            payload["lessons"][0]["practice_tasks"][0]["learner_action"] = "Xem xét phản ánh"

        lesson, _codes = fixed(sentence_only)
        # "người học xác định có phải khiếu nại ...": a recognition, so a question.
        self.assertEqual(lesson.units[0].components[1].type, "problem")

    def test_the_note_is_plain_in_both_locales(self) -> None:
        self.assertEqual(practice_format_note([("la_sortable", "problem")], "en"),
                         "Note: the ordering practice became a single-choice scenario question because the source has "
                         "no steps in a stated order; the practice task, criteria and feedback are unchanged.")
        self.assertNotIn("IDM_", practice_format_note([("la_diagram", "html")], "vi"))


def validate(lesson: IdmLessonDesignV1, position: int) -> list[IdmIssue]:
    return errors(check(0, {"lessons": [None] * position + [lesson.model_dump(mode="json")]}, position))


class PracticeFormatModuleTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_lesson_is_accepted_without_a_repair_or_a_fallback(self) -> None:
        result, provider, runtime = await design_module(0, [edited("mod_01", practice_as("la_sortable"))])
        design = result["shard"]["idm_design"]
        self.assertEqual((len(provider.calls), design["stage_origin"]), (1, "provider"))
        self.assertEqual(runtime.adjustments[PRACTICE_FORMAT_CODE], 1)
        self.assertEqual(result["content_origin"], "provider_validated")
        unit = result["shard"]["lessons"][0]["units"][0]
        self.assertEqual([c["type"] for c in unit["component_plan"]], ["html", "problem"])
        practice = design["lessons"][0]["practice_tasks"][0]
        golden = gm.MODULE_RESPONSES["mod_01"]["lessons"][0]["practice_tasks"][0]
        self.assertEqual((practice["sentence"], practice["criteria_fact_keys"], practice["feedback_focus"]),
                         (golden["sentence"], golden["criteria_fact_keys"], golden["feedback_focus"]))
        self.assertNotIn("Bố cục dự phòng", design["lessons"][0]["notes"])
