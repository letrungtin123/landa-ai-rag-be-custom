"""Strict IDM data contracts (spec §6, §15.1): closed models, stripped strings, ids, bounds and hashes."""

from __future__ import annotations

import hashlib
import json
import unittest
from typing import Any

from pydantic import BaseModel, ValidationError

from app.idm.contracts import (
    IdmAudienceV1,
    IdmAuthorNotesV1,
    IdmBlockLoLinkV1,
    IdmBlockScopeV1,
    IdmBlueprintRowV1,
    IdmContentBlockV1,
    IdmCourseDesignV1,
    IdmCourseSkeletonRequestV1,
    IdmDispositionV1,
    IdmGapV1,
    IdmHoldItemV1,
    IdmIssueV1,
    IdmLearningObjectiveV1,
    IdmLessonPlanV1,
    IdmModulePlanV1,
    IdmMustDoV1,
    IdmProjectContextV1,
    IdmSourceDocumentRefV1,
    IdmTokenAllowanceV1,
    IdmW1BlockDraftV1,
    IdmW1ReduceResponseV1,
    IdmW1SectionResponseV1,
    IdmW2BlueprintResponseV1,
    IdmW4CourseResponseV1,
    brief_hash_of,
    design_hash_of,
)
from app.lesson_author_orchestration_v2 import canonical_hash
from tests.idm_golden import W1_REDUCE, W1_SECTION, W2_BLUEPRINT, W4_COURSE, project_context, source_facts
from tests.test_idm_course_design import golden_design

HEX32 = "0123456789abcdef" * 2


def lesson_payload() -> dict[str, Any]:
    return dict(W4_COURSE["modules"][0]["lessons"][0])


def module_payload() -> dict[str, Any]:
    return dict(W4_COURSE["modules"][0])


def block_payload() -> dict[str, Any]:
    draft = {name: value for name, value in W1_SECTION["blocks"][1].items() if name != "local_id"}
    return {**draft, "block_id": "cb_0002", "section_id": "sec_001", "origin": "provider"}


def scope_payload() -> dict[str, Any]:
    return {"scope_key": "idmcb_" + HEX32, "block_id": "cb_0003", "title": "Thế nào là một khiếu nại?",
            "source_ref": "src-2", "fact_count": 1, "content_chars": 10, "fact_keys": ["d1-c2-f1"]}


def valid_payloads() -> list[tuple[type[BaseModel], dict[str, Any]]]:
    return [
        (IdmSourceDocumentRefV1, {"document_id": "d1", "name": "a.pdf", "type": "pdf"}),
        (IdmProjectContextV1, project_context()),
        (IdmGapV1, {"type": "missing_example", "note": "Thiếu ví dụ."}),
        (IdmIssueV1, {"type": "conflict", "note": "24 giờ và 48 giờ.", "fact_keys": ["d1-c7-f2"]}),
        (IdmW1BlockDraftV1, dict(W1_SECTION["blocks"][0])),
        (IdmW1SectionResponseV1, W1_SECTION),
        (IdmContentBlockV1, block_payload()),
        (IdmW1ReduceResponseV1, W1_REDUCE),
        (IdmBlueprintRowV1, dict(W2_BLUEPRINT["rows"][0])),
        (IdmW2BlueprintResponseV1, W2_BLUEPRINT),
        (IdmLessonPlanV1, lesson_payload()),
        (IdmModulePlanV1, module_payload()),
        (IdmW4CourseResponseV1, W4_COURSE),
        (IdmDispositionV1, {"fact_key": "d1-c0-f1", "disposition": "noise", "block_id": None, "reason": "x"}),
        (IdmBlockScopeV1, scope_payload()),
        (IdmHoldItemV1, {"block_id": "cb_0010", "name": "Thời hạn", "reason": "Mâu thuẫn",
                         "sme_question": "24 hay 48 giờ?", "blocked_must_do_ids": ["md_5"]}),
        (IdmTokenAllowanceV1, {"input_tokens": 10, "output_tokens": 10}),
        (IdmAuthorNotesV1, {"course": "Ghi chú", "modules": {"mod_01": "m"}, "lessons": {"lsn_001": "l"}}),
    ]


def rejected(model: type[BaseModel], payload: dict[str, Any]) -> list[str]:
    try:
        model.model_validate(payload)
    except ValidationError as error:
        return [str(item["type"]) for item in error.errors()]
    return []


class ClosedModelTests(unittest.TestCase):
    def test_every_model_accepts_its_valid_payload(self) -> None:
        for model, payload in valid_payloads():
            with self.subTest(model=model.__name__):
                self.assertEqual(rejected(model, payload), [])

    def test_every_model_rejects_unknown_keys(self) -> None:
        for model, payload in valid_payloads():
            with self.subTest(model=model.__name__):
                self.assertIn("extra_forbidden", rejected(model, {**payload, "unexpected": 1}))

    def test_nested_models_reject_unknown_keys(self) -> None:
        payload = {**W1_SECTION, "blocks": [{**W1_SECTION["blocks"][0], "score": 3}, *W1_SECTION["blocks"][1:]]}
        self.assertIn("extra_forbidden", rejected(IdmW1SectionResponseV1, payload))
        rows = [{**W2_BLUEPRINT["rows"][0], "priority": "high"}]
        self.assertIn("extra_forbidden", rejected(IdmW2BlueprintResponseV1, {"rows": rows}))

    def test_strict_mode_never_coerces_types(self) -> None:
        self.assertTrue(rejected(IdmProjectContextV1, {**project_context(), "duration_target_minutes": "30"}))
        self.assertTrue(rejected(IdmBlueprintRowV1, {**W2_BLUEPRINT["rows"][0], "hold": "false"}))
        self.assertTrue(rejected(IdmLessonPlanV1, {**lesson_payload(), "est_minutes": 5.0}))
        self.assertTrue(rejected(IdmGapV1, {"type": "missing_example", "note": 123}))
        self.assertTrue(rejected(IdmW1BlockDraftV1, {**W1_SECTION["blocks"][0], "intent": "learn"}))


class StringNormalisationTests(unittest.TestCase):
    def test_strings_are_stripped_before_validation(self) -> None:
        self.assertEqual(IdmGapV1(type="missing_example", note="  Thiếu ví dụ.  ").note, "Thiếu ví dụ.")
        link = IdmBlockLoLinkV1(block_id=" cb_0001 ", lo_id="lo_1", relation="direct")
        self.assertEqual(link.block_id, "cb_0001")
        draft = IdmW1BlockDraftV1.model_validate({**W1_SECTION["blocks"][0], "sme_questions": ["  Hỏi gì?  "]})
        self.assertEqual(draft.sme_questions, ["Hỏi gì?"])

    def test_length_bounds_apply_after_stripping(self) -> None:
        # "  ab  " is six characters raw but only two after stripping: below the minimum of 3.
        self.assertIn("string_too_short", rejected(IdmW1BlockDraftV1, {**W1_SECTION["blocks"][0], "name": "  ab  "}))

    def test_blank_required_strings_are_rejected(self) -> None:
        cases: list[tuple[type[BaseModel], dict[str, Any]]] = [
            (IdmSourceDocumentRefV1, {"document_id": "d1", "name": "   "}),
            (IdmGapV1, {"type": "missing_step", "note": " \n\t "}),
            (IdmAudienceV1, {"description": " " * 20, "origin": "client"}),
            (IdmW1BlockDraftV1, {**W1_SECTION["blocks"][0], "fact_keys": ["   "]}),
            (IdmW1BlockDraftV1, {**W1_SECTION["blocks"][0], "sme_questions": ["   "]}),
            (IdmProjectContextV1, {**project_context(), "learning_objectives": ["   "]}),
            (IdmLessonPlanV1, {**lesson_payload(), "ordering_rationale": "  "}),
            (IdmBlueprintRowV1, {**W2_BLUEPRINT["rows"][0], "detail_level": "   "}),
            (IdmDispositionV1, {"fact_key": "  ", "disposition": "noise", "block_id": None}),
            (IdmAuthorNotesV1, {"course": "c", "modules": {"mod_01": "   "}, "lessons": {}}),
        ]
        for model, payload in cases:
            with self.subTest(model=model.__name__, payload=payload):
                self.assertIn("string_too_short", rejected(model, payload))

    def test_nullable_strings_may_be_blank(self) -> None:
        context = IdmProjectContextV1.model_validate({**project_context(), "course_title_hint": "   "})
        self.assertEqual(context.course_title_hint, "")
        self.assertIsNone(IdmProjectContextV1.model_validate(project_context()).target_audience)


class IdentifierPatternTests(unittest.TestCase):
    def check(self, model: type[BaseModel], base: dict[str, Any], field: str, good: list[str],
              bad: list[str]) -> None:
        for value in good:
            with self.subTest(field=field, value=value):
                self.assertEqual(rejected(model, {**base, field: value}), [])
        for value in bad:
            with self.subTest(field=field, value=value):
                self.assertIn("string_pattern_mismatch", rejected(model, {**base, field: value}))

    def test_identifier_patterns(self) -> None:
        link = {"block_id": "cb_0001", "lo_id": "lo_1", "relation": "direct"}
        self.check(IdmBlockLoLinkV1, link, "block_id", ["cb_0001", "cb_9999"],
                   ["cb_1", "cb_00001", "CB_0001", "cb_abcd", "block_1"])
        self.check(IdmBlockLoLinkV1, link, "lo_id", ["lo_1", "lo_12", "lo_99"], ["lo_0", "lo_01", "lo_100", "LO_1"])
        must_do = {"must_do_id": "md_1", "lo_id": "lo_1", "statement": "Phân loại khiếu nại", "kind": "do",
                   "bloom": "apply"}
        self.check(IdmMustDoV1, must_do, "must_do_id", ["md_1", "md_40"], ["md_0", "md_100", "md_a"])
        self.check(IdmContentBlockV1, block_payload(), "section_id", ["sec_001", "sec_016"],
                   ["sec_01", "sec_0001", "section_1"])
        self.check(IdmLessonPlanV1, lesson_payload(), "lesson_key", ["lsn_001", "lsn_120"],
                   ["lsn_01", "lsn_1000", "lesson_1"])
        self.check(IdmModulePlanV1, module_payload(), "module_key", ["mod_01", "mod_24"], ["mod_1", "mod_001"])
        self.check(IdmBlockScopeV1, scope_payload(), "scope_key", ["idmcb_" + HEX32],
                   ["idmcb_" + HEX32.upper(), "idmcb_" + HEX32[:31], "idmcb_" + "g" * 32, "scope3_01"])
        self.check(IdmW1BlockDraftV1, dict(W1_SECTION["blocks"][0]), "local_id", ["b1", "b40", "b999"],
                   ["b0", "b1000", "B1", "cb_0001"])

    def test_id_lists_are_checked_item_by_item(self) -> None:
        payload = {**W2_BLUEPRINT["rows"][2], "must_do_ids": ["md_1", "md_x"]}
        self.assertIn("string_pattern_mismatch", rejected(IdmBlueprintRowV1, payload))
        payload = {**lesson_payload(), "block_ids": ["cb_0003", "cb-0004"]}
        self.assertIn("string_pattern_mismatch", rejected(IdmLessonPlanV1, payload))


class LengthBoundTests(unittest.TestCase):
    def test_string_and_list_bounds(self) -> None:
        block = dict(W1_SECTION["blocks"][0])
        context = project_context()
        document = context["source_documents"][0]
        cases: list[tuple[type[BaseModel], dict[str, Any], str, Any, bool]] = [
            (IdmW1BlockDraftV1, block, "name", "abc", True),
            (IdmW1BlockDraftV1, block, "name", "x" * 180, True),
            (IdmW1BlockDraftV1, block, "name", "x" * 181, False),
            (IdmW1BlockDraftV1, block, "summary", "x" * 300, True),
            (IdmW1BlockDraftV1, block, "summary", "x" * 301, False),
            (IdmW1BlockDraftV1, block, "fact_keys", [], False),
            (IdmW1BlockDraftV1, block, "sme_questions", ["Câu hỏi?"] * 5, True),
            (IdmW1BlockDraftV1, block, "sme_questions", ["Câu hỏi?"] * 6, False),
            (IdmW1BlockDraftV1, block, "sme_questions", ["x" * 401], False),
            (IdmW1SectionResponseV1, W1_SECTION, "blocks", [], False),
            (IdmW1SectionResponseV1, W1_SECTION, "blocks", [block] * 41, False),
            (IdmLearningObjectiveV1, W1_REDUCE["learning_objectives"][0], "statement", "x" * 9, False),
            (IdmLearningObjectiveV1, W1_REDUCE["learning_objectives"][0], "statement", "x" * 10, True),
            (IdmMustDoV1, W1_REDUCE["must_dos"][0], "statement", "x" * 4, False),
            (IdmMustDoV1, W1_REDUCE["must_dos"][0], "statement", "x" * 280, True),
            (IdmMustDoV1, W1_REDUCE["must_dos"][0], "statement", "x" * 281, False),
            (IdmLessonPlanV1, lesson_payload(), "est_screens", 1, False),
            (IdmLessonPlanV1, lesson_payload(), "est_screens", 30, True),
            (IdmLessonPlanV1, lesson_payload(), "est_screens", 31, False),
            (IdmLessonPlanV1, lesson_payload(), "est_minutes", 60, True),
            (IdmLessonPlanV1, lesson_payload(), "est_minutes", 61, False),
            (IdmLessonPlanV1, lesson_payload(), "secondary_must_do_ids", ["md_2", "md_3", "md_4"], False),
            (IdmLessonPlanV1, lesson_payload(), "block_ids", ["cb_0001"] * 41, False),
            (IdmModulePlanV1, module_payload(), "performance_goal", "x" * 9, False),
            (IdmProjectContextV1, context, "duration_target_minutes", 4, False),
            (IdmProjectContextV1, context, "duration_target_minutes", 5, True),
            (IdmProjectContextV1, context, "duration_target_minutes", 600, True),
            (IdmProjectContextV1, context, "duration_target_minutes", 601, False),
            (IdmProjectContextV1, context, "source_documents", [], False),
            (IdmProjectContextV1, context, "source_documents", [document] * 21, False),
            (IdmProjectContextV1, context, "learning_objectives", ["Phân loại khiếu nại"] * 9, False),
            (IdmProjectContextV1, context, "locale", "fr", False),
            (IdmW4CourseResponseV1, W4_COURSE, "course_summary", "x" * 19, False),
            (IdmW4CourseResponseV1, W4_COURSE, "prerequisites", ["x"] * 11, False),
            (IdmW4CourseResponseV1, W4_COURSE, "prerequisites", ["x" * 301], False),
            (IdmTokenAllowanceV1, {"input_tokens": 1, "output_tokens": 1}, "output_tokens", 131_072, True),
            (IdmTokenAllowanceV1, {"input_tokens": 1, "output_tokens": 1}, "output_tokens", 131_073, False),
            (IdmTokenAllowanceV1, {"input_tokens": 1, "output_tokens": 1}, "input_tokens", 0, False),
            (IdmAuthorNotesV1, {"course": "c", "modules": {}, "lessons": {}}, "course", "x" * 7_000, True),
            (IdmAuthorNotesV1, {"course": "c", "modules": {}, "lessons": {}}, "course", "x" * 7_001, False),
            (IdmAuthorNotesV1, {"course": "c", "modules": {}, "lessons": {}}, "modules", {"m": "x" * 3_001}, False),
            (IdmAuthorNotesV1, {"course": "c", "modules": {}, "lessons": {}}, "lessons", {"l": "x" * 2_001}, False),
            (IdmBlockScopeV1, scope_payload(), "fact_count", 0, False),
            (IdmBlockScopeV1, scope_payload(), "content_chars", 0, False),
        ]
        for model, base, field, value, valid in cases:
            with self.subTest(model=model.__name__, field=field, size=len(value) if hasattr(value, "__len__")
                              else value):
                self.assertEqual(rejected(model, {**base, field: value}) == [], valid)

    def test_skeleton_request_bounds(self) -> None:
        request = {"pipeline_version": "idm-1", "project_context": project_context(),
                   "source_facts": [source_facts()[0].model_dump()],
                   "token_allowance": {"input_tokens": 1, "output_tokens": 1}, "remaining_budget_ms": 30_000}
        self.assertEqual(rejected(IdmCourseSkeletonRequestV1, request), [])
        self.assertTrue(rejected(IdmCourseSkeletonRequestV1, {**request, "pipeline_version": "v2-legacy"}))
        self.assertTrue(rejected(IdmCourseSkeletonRequestV1, {**request, "remaining_budget_ms": 29_999}))
        self.assertTrue(rejected(IdmCourseSkeletonRequestV1, {**request, "source_facts": []}))


class HashTests(unittest.TestCase):
    def test_design_hash_is_canonical_stable_and_excludes_itself(self) -> None:
        payload = {"b": [1, {"y": 2, "x": 1}], "a": "Tiếng Việt", "design_hash": "first"}
        digest = design_hash_of(payload)
        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertEqual(digest, design_hash_of({**payload, "design_hash": "second"}))
        self.assertEqual(digest, design_hash_of({"a": "Tiếng Việt", "b": [1, {"x": 1, "y": 2}]}))
        self.assertEqual(digest, canonical_hash({"a": "Tiếng Việt", "b": [1, {"x": 1, "y": 2}]}))
        node_encoding = json.dumps({"a": "Tiếng Việt", "b": [1, {"x": 1, "y": 2}]}, ensure_ascii=False,
                                   sort_keys=True, separators=(",", ":"))
        self.assertEqual(digest, hashlib.sha256(node_encoding.encode("utf-8")).hexdigest())
        self.assertNotEqual(digest, design_hash_of({**payload, "a": "Tieng Viet"}))
        self.assertNotEqual(design_hash_of({"a": 1, "brief_hash": "x"}), design_hash_of({"a": 1}))

    def test_brief_hash_excludes_only_its_own_key(self) -> None:
        payload = {"lesson_title": "Bài 1", "brief_hash": "old"}
        self.assertEqual(brief_hash_of(payload), brief_hash_of({**payload, "brief_hash": "new"}))
        self.assertEqual(brief_hash_of(payload), canonical_hash({"lesson_title": "Bài 1"}))
        self.assertNotEqual(brief_hash_of({**payload, "design_hash": "x"}), brief_hash_of(payload))


class GoldenDesignRoundTripTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.design = golden_design()

    def test_json_round_trip_is_lossless(self) -> None:
        text = self.design.model_dump_json()
        restored = IdmCourseDesignV1.model_validate_json(text)
        self.assertEqual(restored, self.design)
        self.assertEqual(restored.model_dump_json(), text)
        self.assertEqual(IdmCourseDesignV1.model_validate(json.loads(text)), self.design)

    def test_design_hash_covers_every_field_but_itself(self) -> None:
        payload = self.design.model_dump(mode="json")
        self.assertEqual(design_hash_of(payload), self.design.design_hash)
        payload["notes"]["course"] += " Đã sửa."
        self.assertNotEqual(design_hash_of(payload), self.design.design_hash)

    def test_design_rejects_malformed_identity_fields(self) -> None:
        payload = self.design.model_dump(mode="json")
        self.assertTrue(rejected(IdmCourseDesignV1, {**payload, "design_hash": "abc"}))
        self.assertTrue(rejected(IdmCourseDesignV1, {**payload, "source_snapshot_hash": "A" * 64}))
        self.assertTrue(rejected(IdmCourseDesignV1, {**payload, "pipeline_version": "idm-2"}))
        self.assertTrue(rejected(IdmCourseDesignV1, {**payload, "stage_origins": {"w2": "llm"}}))
        self.assertTrue(rejected(IdmCourseDesignV1, {**payload, "dispositions": []}))
        self.assertIn("extra_forbidden", rejected(IdmCourseDesignV1, {**payload, "chapters": []}))


if __name__ == "__main__":
    unittest.main()
