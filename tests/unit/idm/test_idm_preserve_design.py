"""QLT-6 preserve_source: course design, module stage and unit stage of a kept document, and the whole chain.

Only the course skeleton calls the provider (the one segmentation call); module and unit stages are deterministic, and
every learner-visible word of every unit is the document's own text, in order.
"""

from __future__ import annotations

import copy
import functools
import json
import unittest
from typing import Any
from unittest.mock import patch

from app.contracts.orchestration_v2.blueprint import CourseSkeletonV2, canonical_hash
from app.contracts.orchestration_v2.provider import SourceSnapshotFactV2, UnitGenerationContractV2
from app.idm.contracts import IdmCourseDesignV1, IdmCourseSkeletonRequestV1, IdmShardDesignV1, brief_hash_of
from app.idm.course_design import run_idm_course_design
from app.idm.preserve_design import (
    EMPTY_CODE,
    HTML_NOT_ALLOWED_CODE,
    MODE_MISMATCH_CODE,
    run_preserve_module_design,
)
from app.idm.preserve_lines import expected_tokens, html_tokens
from app.idm.preserve_unit import TEXT_MISMATCH_CODE, UNIT_INVALID_CODE, run_preserve_unit
from app.idm.runtime import IdmProviderError, IdmStageError
from app.schemas.orchestration_v2 import RagLessonAuthorChapterShardV2Request, RagLessonAuthorUnitV2Request
from app.services.orchestration_v2.chapter_shard import lesson_author_orchestration_v2_chapter_shard
from app.services.orchestration_v2.unit import lesson_author_orchestration_v2_unit
from tests import idm_golden as g
from tests.golden import idm_golden_module as gm
from tests.golden import idm_preserve_golden as pg
from tests.golden.idm_golden_unit import build_unit_request
from tests.support.idm_test_support import _run_sync, make_runtime


def course_request(facts: list[SourceSnapshotFactV2] | None = None, **context: Any) -> IdmCourseSkeletonRequestV1:
    return IdmCourseSkeletonRequestV1.model_validate({
        "pipeline_version": "idm-1", "project_context": {**pg.project_context(), **context},
        "source_facts": [fact.model_dump() for fact in facts or pg.source_facts()],
        "token_allowance": {"input_tokens": 1_800_000, "output_tokens": 131_072}, "remaining_budget_ms": 500_000,
    })


def run_course(provider: g.FakeIdmProvider, request: IdmCourseSkeletonRequestV1 | None = None) -> dict[str, Any]:
    body = request or course_request()
    return _run_sync(lambda: run_idm_course_design(body, source_snapshot_hash=pg.SNAPSHOT_HASH,
                                                   runtime=make_runtime(provider), parallelism=4, max_sections=16))


@functools.cache
def _golden_json() -> str:
    return json.dumps(run_course(pg.preserve_provider()), ensure_ascii=False)


def golden() -> tuple[dict[str, Any], IdmCourseDesignV1, CourseSkeletonV2]:
    result = json.loads(_golden_json())
    return result, IdmCourseDesignV1.model_validate(result["idm"]), CourseSkeletonV2.model_validate(result["skeleton"])


def shard_of(design: IdmCourseDesignV1, skeleton: CourseSkeletonV2, chapter: int) -> dict[str, Any]:
    context, plan, facts = gm.module_context(design, skeleton, chapter, pg.source_facts())
    return run_preserve_module_design(context=context, skeleton=skeleton, plan=plan, facts=facts)


def unit_request(design: IdmCourseDesignV1, shard: IdmShardDesignV1, chapter: int, lesson: int, unit: int,
                 *, design_mode: str | None = "preserve_source") -> dict[str, Any]:
    request = build_unit_request(design, shard, chapter_index=chapter, lesson_position=lesson, unit_position=unit,
                                 facts=pg.source_facts())
    contract = request["unit_contract"]
    contract["idm_unit_brief"]["design_mode"] = design_mode
    contract["idm_unit_brief"]["brief_hash"] = brief_hash_of(contract["idm_unit_brief"])
    contract["contract_hash"] = canonical_hash({key: value for key, value in contract.items()
                                               if key != "contract_hash"})
    request["fallback_only"] = True
    return request


def contract_of(request: dict[str, Any]) -> UnitGenerationContractV2:
    return UnitGenerationContractV2.model_validate(request["unit_contract"])


class CourseDesignTests(unittest.TestCase):
    def test_one_segmentation_call_and_the_document_structure_verbatim(self) -> None:
        provider = pg.preserve_provider()
        result = run_course(provider)
        self.assertEqual([call["schema"] for call in provider.calls], ["IdmPreserveSegmentationV1"])
        self.assertEqual((result["content_origin"], result["quality_state"], result["usage_source"]),
                         ("provider_validated", "validated", "provider"))
        design = IdmCourseDesignV1.model_validate(result["idm"])
        self.assertEqual(design.idm_contract_version, 3)
        self.assertEqual(design.stage_origins, {"preserve_segmentation": "provider"})
        self.assertEqual([module.title for module in design.modules],
                         ["MODUN 1: VÌ SAO PHẢN HỒI QUAN TRỌNG", "MODUN 2: CÁCH ĐƯA RA PHẢN HỒI"])
        self.assertEqual([lesson.title for module in design.modules for lesson in module.lessons],
                         ["Bài 1: Phản hồi là gì", "Bài 2: Lợi ích của phản hồi", "Bài 3: Mô hình SBI",
                          "Bài 4: Những lỗi thường gặp"])
        self.assertEqual([block.name for block in design.blocks],
                         ["Bài 1: Phản hồi là gì", "Bài 2: Lợi ích của phản hồi", "Bài 3: Mô hình SBI",
                          "3.1 Ví dụ áp dụng", "Bài 4: Những lỗi thường gặp"])
        self.assertTrue(all(row.treatment == "keep" and row.placement == "course" for row in design.blueprint))

    def test_every_fact_is_accounted_for_once_and_furniture_is_noise(self) -> None:
        _result, design, skeleton = golden()
        dispositions = {item.fact_key: item for item in design.dispositions}
        self.assertEqual(set(dispositions), {fact.fact_key for fact in pg.source_facts()})
        noise = {key for key, item in dispositions.items() if item.disposition == "noise"}
        self.assertEqual(noise, {*pg.PAGE_NUMBER_KEYS, *pg.REPEATED_HEADER_KEYS})
        self.assertEqual(dispositions[pg.PAGE_NUMBER_KEYS[0]].reason, "Số trang")
        scoped = [key for scope in design.block_scopes for key in scope.fact_keys]
        self.assertEqual(scoped, [fact.fact_key for fact in pg.source_facts() if fact.fact_key not in noise])
        self.assertEqual(sorted(key for chapter in skeleton.chapters for key in chapter.source_scope_ids),
                         sorted(scope.scope_key for scope in design.block_scopes))

    def test_pedagogy_is_skipped_behind_author_only_placeholders(self) -> None:
        _result, design, skeleton = golden()
        self.assertEqual(design.target_audience.origin, "client")
        self.assertTrue(design.target_audience.description.startswith("Không áp dụng"))
        self.assertEqual([(item.origin, item.statement.startswith("Không áp dụng"))
                          for item in design.learning_objectives], [("client", True)])
        self.assertEqual((design.purpose.profile, design.purpose.fidelity_mode, design.purpose.template_gaps),
                         ("general", "keep", []))
        self.assertTrue(all(not block.sme_questions and not block.issues for block in design.blocks))
        self.assertEqual(design.hold_items, [])
        self.assertEqual(skeleton.title, "Kỹ năng phản hồi")
        self.assertNotIn("Đề xuất bởi AI", skeleton.target_audience)

    def test_author_notes_in_plain_words(self) -> None:
        _result, design, _skeleton = golden()
        lines = design.notes.course.split("\n")
        self.assertEqual(lines[0], "Chế độ: Giữ nguyên nội dung tài liệu — nội dung được giữ nguyên từng chữ, AI chỉ "
                                   "chia chương và bài.")
        self.assertIn("AI chia theo các tiêu đề có trong tài liệu", lines[1])
        self.assertEqual(lines[2], "• Đã bỏ 4 dòng số trang và 3 dòng đầu trang hoặc chân trang lặp lại. Không bỏ "
                                   "dòng nào khác.")
        self.assertEqual(design.notes.modules["mod_01"], "Chương giữ nguyên theo tài liệu: 2 bài.")
        self.assertTrue(design.notes.lessons["lsn_003"].startswith("Bài giữ nguyên theo tài liệu: 2 phần"))
        for text in (design.notes.course, *design.notes.modules.values(), *design.notes.lessons.values()):
            self.assertNotRegex(text, r"[<>]|IDM_|PRESERVE")

    def test_fallback_structure_is_recorded_and_reviewable(self) -> None:
        provider = pg.preserve_provider(IdmProviderError("AI_PROVIDER_UNAVAILABLE", terminal=False))
        result = run_course(provider, course_request(locale="en"))
        design = IdmCourseDesignV1.model_validate(result["idm"])
        self.assertEqual(design.stage_origins, {"preserve_segmentation": "deterministic_fallback"})
        self.assertEqual((result["content_origin"], result["quality_state"]),
                         ("structured_fallback", "review_required"))
        self.assertIn("the system split it at the headings it found", design.notes.course)
        self.assertTrue(design.notes.course.startswith("Mode: Keep the document as it is"))
        self.assertEqual(len(design.blocks), 4, "the fallback has no sub-unit for 3.1 (not a marker level)")

    def test_a_document_of_furniture_only_is_refused(self) -> None:
        facts = [SourceSnapshotFactV2(document_id=pg.DOCUMENT_ID, fact_key=f"d1-c0-f{number}", scope_key="scope3_00",
                                      fact_text=str(number), source_ref=None, source_page=number, source_chunk=0,
                                      locator={}) for number in range(1, 4)]
        provider = pg.preserve_provider()
        with self.assertRaises(IdmStageError) as raised:
            run_course(provider, course_request(facts))
        self.assertEqual((raised.exception.code, provider.calls), (EMPTY_CODE, []))


class ModuleStageTests(unittest.TestCase):
    def test_one_html_slot_per_unit_and_no_pedagogy(self) -> None:
        _result, design, skeleton = golden()
        response = shard_of(design, skeleton, 1)
        self.assertEqual((response["usage"]["totalTokens"], response["usage_source"], response["content_origin"]),
                         (0, "provider", "structured_fallback"))
        shard = response["shard"]
        self.assertEqual(shard["assessment_obligations"], [])
        lessons = IdmShardDesignV1.model_validate(shard["idm_design"]).lessons
        self.assertEqual([[unit.title for unit in lesson.units] for lesson in lessons],
                         [["Bài 3: Mô hình SBI", "3.1 Ví dụ áp dụng"], ["Bài 4: Những lỗi thường gặp"]])
        for lesson in lessons:
            self.assertEqual(lesson.practice_tasks, [])
            for unit in lesson.units:
                self.assertEqual([(slot.type, slot.role, slot.block_ids) for slot in unit.components],
                                 [("html", "explain", unit.block_ids)])
        self.assertEqual(shard["lessons"][0]["learning_activities"], ["Đọc nội dung giữ nguyên theo tài liệu."])

    def test_the_route_never_calls_the_provider(self) -> None:
        _result, design, skeleton = golden()
        context, plan, facts = gm.module_context(design, skeleton, 0, pg.source_facts())
        request = RagLessonAuthorChapterShardV2Request.model_validate({
            "tenant_id": "00000000-0000-4000-8000-0000000000a1", "kb_id": "00000000-0000-4000-8000-0000000000b1",
            "conversation_id": "00000000-0000-4000-8000-0000000000c1", "target": "lesson_author",
            "model": "gemini-3.8-flash", "max_output_tokens": 65_536, "embedding_model": "gemini-embedding-001",
            "system_prompt": "x", "user_message": "x", "locale": "vi", "api_key": "test-key",
            "correlation_id": "00000000-0000-4000-8000-0000000000e2",
            "source_documents": [{"document_id": pg.DOCUMENT_ID, "kb_id": "00000000-0000-4000-8000-0000000000b1",
                                  "name": "ky-nang-phan-hoi.pdf", "type": "pdf", "status": "ready"}],
            "contract_version": 2, "skeleton": skeleton.model_dump(mode="json"), "shard_plan": plan.model_dump(),
            "source_facts": [fact.model_dump(mode="json") for fact in facts], "max_attempts": 2,
            "idm_module_context": context.model_dump(mode="json"),
        })
        with patch("app.services.provider.generate_content", side_effect=AssertionError("provider called")):
            response = _run_sync(lambda: lesson_author_orchestration_v2_chapter_shard(request))
        self.assertEqual(len(response["shard"]["lessons"]), 2)

    def test_guards(self) -> None:
        _result, design, skeleton = golden()
        context, plan, facts = gm.module_context(design, skeleton, 0, pg.source_facts())
        no_html = context.model_copy(update={"allowed_component_types": ["problem"]})
        with self.assertRaises(IdmStageError) as raised:
            run_preserve_module_design(context=no_html, skeleton=skeleton, plan=plan, facts=facts)
        self.assertEqual(raised.exception.code, HTML_NOT_ALLOWED_CODE)
        designed = context.model_copy(update={"project_context": context.project_context.model_copy(
            update={"design_mode": "instructional_design"})})
        with self.assertRaises(IdmStageError) as raised:
            run_preserve_module_design(context=designed, skeleton=skeleton, plan=plan, facts=facts)
        self.assertEqual(raised.exception.code, MODE_MISMATCH_CODE)


def all_unit_requests() -> list[dict[str, Any]]:
    _result, design, skeleton = golden()
    requests = []
    for chapter in range(len(design.modules)):
        shard = IdmShardDesignV1.model_validate(shard_of(design, skeleton, chapter)["shard"]["idm_design"])
        requests.extend(unit_request(design, shard, chapter, lesson_index, unit_index)
                        for lesson_index, lesson in enumerate(shard.lessons) for unit_index in range(len(lesson.units)))
    return requests


class UnitStageTests(unittest.TestCase):
    def test_every_unit_shows_exactly_its_lines_and_the_course_covers_every_line_once(self) -> None:
        covered: list[str] = []
        for request in all_unit_requests():
            contract = contract_of(request)
            result = run_preserve_unit(contract=contract, locale="vi", fallback_only=True)
            self.assertEqual((result["usage_source"], result["usage_complete"], result["usage"]["totalTokens"]),
                             ("deterministic_fallback", True, 0))
            self.assertEqual((result["content_origin"], result["quality_state"]), ("structured_fallback",
                                                                                  "review_required"))
            unit = result["unit"]
            self.assertNotIn("source_locked_fallback", unit, "a kept unit is the intended output, not a fallback")
            [component] = unit["components"]
            texts = [fact.fact_text for fact in contract.source_facts]
            self.assertEqual(html_tokens(component["html"]), expected_tokens(texts))
            self.assertEqual(component["source_fact_ids"], contract.unit_source_fact_ids)
            covered.extend(contract.unit_source_fact_ids)
            quality = unit["idm_quality"]
            self.assertEqual((quality["judge_mode"], quality["judge_status"]), ("off", "not_run"))
            self.assertEqual(quality["deterministic_codes"][0], "IDM_PRESERVE_SOURCE")
        noise = {*pg.PAGE_NUMBER_KEYS, *pg.REPEATED_HEADER_KEYS}
        self.assertEqual(covered, [fact.fact_key for fact in pg.source_facts() if fact.fact_key not in noise])

    def test_kept_source_mentions_are_reported_to_the_author(self) -> None:
        reports = {}
        for request in all_unit_requests():
            result = run_preserve_unit(contract=contract_of(request), locale="vi", fallback_only=True)
            reports[request["unit_contract"]["unit_title"]] = result["unit"]["idm_quality"]
        self.assertIn("IDM_PRESERVE_KEPT_SOURCE_MARKER", reports["Bài 2: Lợi ích của phản hồi"]["deterministic_codes"])
        self.assertIn("IDM_PRESERVE_KEPT_HTML_SOURCE_FILENAME",
                      reports["Bài 4: Những lỗi thường gặp"]["deterministic_codes"])
        self.assertIn("được giữ đúng như tài liệu", reports["Bài 4: Những lỗi thường gặp"]["author_note"])
        self.assertEqual(reports["3.1 Ví dụ áp dụng"]["author_note"],
                         "Nội dung giữ nguyên từng chữ theo tài liệu; AI không viết thêm gì.")

    def test_a_dispatched_caller_settles_zero_provider_usage(self) -> None:
        result = run_preserve_unit(contract=contract_of(all_unit_requests()[0]), locale="en", fallback_only=False)
        self.assertEqual((result["usage_source"], result["usage_complete"]), ("provider", True))
        self.assertTrue(result["unit"]["idm_quality"]["author_note"].startswith("Content kept word for word"))

    def test_a_unit_that_is_not_a_kept_single_html_slot_is_refused(self) -> None:
        request = all_unit_requests()[0]
        designed = copy.deepcopy(request)
        designed["unit_contract"]["idm_unit_brief"]["design_mode"] = "instructional_design"
        designed["unit_contract"]["idm_unit_brief"]["brief_hash"] = brief_hash_of(
            designed["unit_contract"]["idm_unit_brief"])
        contract = designed["unit_contract"]
        contract["contract_hash"] = canonical_hash({key: value for key, value in contract.items()
                                                   if key != "contract_hash"})
        with self.assertRaises(IdmStageError) as raised:
            run_preserve_unit(contract=contract_of(designed), locale="vi", fallback_only=True)
        self.assertEqual(raised.exception.code, UNIT_INVALID_CODE)

    def test_a_renderer_that_changed_a_word_is_never_returned(self) -> None:
        contract = contract_of(all_unit_requests()[0])
        with patch("app.idm.preserve_unit.render_preserved_html", return_value="<p>Một câu khác.</p>"), \
                self.assertRaises(IdmStageError) as raised:
            run_preserve_unit(contract=contract, locale="vi", fallback_only=True)
        self.assertEqual(raised.exception.code, TEXT_MISMATCH_CODE)

    def test_an_oversized_unit_fails_instead_of_being_cut(self) -> None:
        contract = contract_of(all_unit_requests()[0])
        huge = "<p>" + "x " * 12_000 + "</p>"
        with patch("app.idm.preserve_unit.render_preserved_html", return_value=huge), \
                patch("app.idm.preserve_unit.preserve_text_finding", return_value=None), \
                self.assertRaises(IdmStageError) as raised:
            run_preserve_unit(contract=contract, locale="vi", fallback_only=True)
        self.assertEqual(raised.exception.code, UNIT_INVALID_CODE)

    def test_the_route_never_calls_the_provider_and_records_no_fallback(self) -> None:
        request = RagLessonAuthorUnitV2Request.model_validate(all_unit_requests()[1])
        with patch("app.services.provider.generate_content", side_effect=AssertionError("provider called")), \
                patch("app.services.orchestration_v2.unit.record_fallback") as recorded:
            response = _run_sync(lambda: lesson_author_orchestration_v2_unit(request))
        self.assertEqual(response["usage_source"], "deterministic_fallback")
        recorded.assert_not_called()
        broken = all_unit_requests()[1]
        broken["unit_contract"]["idm_unit_brief"]["brief_hash"] = "0" * 64
        broken["unit_contract"]["contract_hash"] = canonical_hash(
            {key: value for key, value in broken["unit_contract"].items() if key != "contract_hash"})
        with self.assertRaises(Exception) as raised:
            _run_sync(lambda: lesson_author_orchestration_v2_unit(RagLessonAuthorUnitV2Request.model_validate(broken)))
        self.assertEqual(getattr(raised.exception, "status_code", None), 422)


if __name__ == "__main__":
    unittest.main()
