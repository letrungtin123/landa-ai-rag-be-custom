"""Golden W3/W4 module responses for the complaint-handling fixture and a Node-like context builder.

``module_context`` mirrors what Node sends for one chapter shard (spec §7.6.1,
§12.3): the module plan restricted to the shard lessons, their blocks, blueprint
rows and block scopes, and the facts remapped to block-scope keys.
"""

from __future__ import annotations

from typing import Any

from app.idm.contracts import IdmCourseDesignV1, IdmModuleContextV1
from app.lesson_author_orchestration_v2 import CourseSkeletonV2
from app.lesson_author_orchestration_v2_provider import ChapterShardPlanV2, SourceSnapshotFactV2
from tests.idm_golden import key, keys

ALLOWED_TYPES = ["html", "problem", "la_faq", "la_sortable", "la_crossword", "la_diagram"]


def _review(purpose: str, scenario: str | None = None, navigation: str | None = None) -> dict[str, Any]:
    return {"purpose": purpose, "example_scenario": scenario, "visual_asset": None,
            "user_behavior_navigation": navigation}


def _component(index: int, ctype: str, role: str, title: str, block_ids: list[str], *,
               practice_id: str | None = None, scenario: str | None = None) -> dict[str, Any]:
    return {"component_index": index, "type": ctype, "role": role, "title": title,
            "rationale": f"Chọn {ctype} vì phù hợp mục đích {role}.", "block_ids": block_ids,
            "practice_id": practice_id, "support_items": [],
            "author_review": _review(f"Phục vụ {role} cho Must Do của mục.", scenario, "Đọc, chọn rồi xem phản hồi.")}


def _practice(sentence: str, criteria: list[str], *, origin: str = "ai_drafted", hold: bool = False,
              question: str | None = None) -> dict[str, Any]:
    return {"practice_id": "pt_1", "sentence": sentence, "context_input": "Một tình huống khiếu nại cụ thể",
            "learner_action": "Chọn cách xử lý", "result": "Quyết định đúng tiêu chí", "bloom": "apply",
            "criteria_fact_keys": criteria, "scenario_origin": origin, "hold": hold, "hold_question": question,
            "feedback_focus": {"criterion": "Tiêu chí của quy trình", "rationale": "Vì sao phương án đúng tốt hơn",
                               "improvement": "Cần đối chiếu tiêu chí trước khi quyết định"}}


def _unit(index: int, segment: str, title: str, block_ids: list[str],
          components: list[dict[str, Any]]) -> dict[str, Any]:
    return {"unit_index": index, "segment": segment, "title": title,
            "purpose": "Người học làm được bước này của Must Do.", "block_ids": block_ids,
            "components": components, "media_brief": None}


def _lesson(lesson_key: str, title: str, objective: str, practices: list[dict[str, Any]],
            units: list[dict[str, Any]]) -> dict[str, Any]:
    return {"lesson_key": lesson_key, "title": title, "objective": objective,
            "learning_objectives": [objective], "practice_tasks": practices,
            "assessment": "Tiêu chí phản hồi sau phần Must Know, cuối mục.", "units": units,
            "notes": "Bloom: Vận dụng · ước tính 4 khối."}


MODULE_RESPONSES: dict[str, dict[str, Any]] = {
    "mod_01": {"lessons": [
        _lesson("lsn_001", "Phân loại khiếu nại theo nhóm", "Phân loại khiếu nại theo nhóm", [
            _practice("Cho một phản ánh của khách hàng, người học xác định có phải khiếu nại và thuộc nhóm nào để "
                      "chuyển đúng hướng xử lý", [*keys(2, 2, 3), *keys(3, 2, 6)]),
        ], [
            _unit(1, "context_explain", "Nhận diện khiếu nại và nhóm", ["cb_0003", "cb_0004"], [
                _component(1, "html", "explain", "Khiếu nại là gì và có những nhóm nào", ["cb_0003", "cb_0004"]),
                _component(2, "problem", "practice", "Phản ánh này là gì?", ["cb_0003", "cb_0004"],
                           practice_id="pt_1"),
            ]),
        ]),
        _lesson("lsn_002", "Đánh giá mức độ nghiêm trọng", "Đánh giá mức độ nghiêm trọng của khiếu nại", [
            _practice("Cho mô tả thiệt hại của một khiếu nại, người học xác định cấp độ để chọn cách xử lý phù hợp",
                      keys(4, 2, 5)),
        ], [
            _unit(1, "context_explain", "Ba cấp độ nghiêm trọng", ["cb_0005"], [
                _component(1, "html", "explain", "Dấu hiệu của từng cấp độ", ["cb_0005"]),
                _component(2, "problem", "practice", "Khiếu nại này thuộc cấp nào?", ["cb_0005"],
                           practice_id="pt_1"),
            ]),
        ]),
    ]},
    "mod_02": {"lessons": [
        _lesson("lsn_003", "Quyết định tự xử lý hay escalate", "Quyết định tự xử lý hay escalate", [
            _practice("Cho một khiếu nại có yếu tố pháp lý, người học quyết định tự xử lý hay escalate để bảo vệ "
                      "khách hàng và công ty", keys(6, 2, 6)),
        ], [
            _unit(1, "context_explain", "Tiêu chí bắt buộc escalate", ["cb_0009"], [
                _component(1, "html", "explain", "Khi nào bắt buộc escalate", ["cb_0009"]),
            ]),
            _unit(2, "practice_feedback", "Tình huống quyết định escalate", ["cb_0011"], [
                _component(1, "problem", "practice", "Bạn sẽ làm gì với khiếu nại này?", ["cb_0011"],
                           practice_id="pt_1"),
                _component(2, "la_faq", "clarify", "Những nhầm lẫn thường gặp", ["cb_0011"]),
            ]),
        ]),
    ]},
    "mod_03": {"lessons": [
        _lesson("lsn_004", "Thực hiện quy trình tiếp nhận khiếu nại",
                "Thực hiện các bước tiếp nhận khiếu nại đúng trình tự", [
                    _practice("Cho các bước tiếp nhận bị xáo trộn, người học sắp xếp lại đúng trình tự để không bỏ "
                              "sót bước", keys(5, 2, 13), origin="source"),
                ], [
            _unit(1, "context_explain", "Tám bước đầu của quy trình", ["cb_0006", "cb_0007"], [
                _component(1, "html", "explain", "Tiếp nhận và định hướng xử lý", ["cb_0006", "cb_0007"]),
            ]),
            _unit(2, "practice_feedback", "Hoàn tất và sắp xếp trình tự", ["cb_0008"], [
                _component(1, "html", "show", "Bốn bước hoàn tất hồ sơ", ["cb_0008"]),
                _component(2, "la_sortable", "practice", "Sắp xếp các bước", ["cb_0008"], practice_id="pt_1"),
            ]),
        ]),
        {"lesson_key": "lsn_005", "title": "Bảng tra mã khiếu nại và đầu mối liên hệ",
         "objective": "Tra cứu mã khiếu nại và đầu mối khi làm việc",
         "learning_objectives": ["Tra cứu mã khiếu nại và đầu mối khi làm việc"], "practice_tasks": [],
         "assessment": "Không đánh giá.", "units": [
             _unit(1, "job_aid", "Bảng tra nhanh", ["cb_0013", "cb_0014"], [
                 _component(1, "html", "job_aid", "Mã khiếu nại và đầu mối", ["cb_0013", "cb_0014"]),
             ]),
         ], "notes": "Job Aid cuối khoá."},
    ]},
}


def chapter_plan(skeleton: CourseSkeletonV2, chapter_index: int) -> ChapterShardPlanV2:
    chapter = skeleton.chapters[chapter_index]
    return ChapterShardPlanV2(
        chapter_key=chapter.chapter_key, order=chapter.order, shard_index=0, shard_count=1,
        source_scope_ids=list(chapter.source_scope_ids), source_fact_count=0, source_content_chars=0,
    )


def module_context(
    design: IdmCourseDesignV1, skeleton: CourseSkeletonV2, chapter_index: int,
    facts: list[SourceSnapshotFactV2],
) -> tuple[IdmModuleContextV1, ChapterShardPlanV2, list[SourceSnapshotFactV2]]:
    module = design.modules[chapter_index]
    block_ids = [block_id for lesson in module.lessons for block_id in lesson.block_ids]
    scope_of_fact = {fact_key: scope.scope_key for scope in design.block_scopes
                     if scope.block_id in block_ids for fact_key in scope.fact_keys}
    shard_facts = [fact.model_copy(update={"scope_key": scope_of_fact[fact.fact_key]})
                   for fact in facts if fact.fact_key in scope_of_fact]
    plan = chapter_plan(skeleton, chapter_index)
    plan = plan.model_copy(update={"source_fact_count": len(shard_facts),
                                   "source_content_chars": sum(len(fact.fact_text) for fact in shard_facts)})
    offset = sum(len(item.lessons) for item in design.modules[:chapter_index])
    context = IdmModuleContextV1.model_validate({
        "pipeline_version": "idm-1", "project_context": design.project_context.model_dump(mode="json"),
        "target_audience": design.target_audience.model_dump(mode="json"),
        "module": module.model_dump(mode="json"),
        "learning_objectives": [item.model_dump(mode="json") for item in design.learning_objectives],
        "must_dos": [item.model_dump(mode="json") for item in design.must_dos],
        "blocks": [item.model_dump(mode="json") for item in design.blocks if item.block_id in block_ids],
        "blueprint": [item.model_dump(mode="json") for item in design.blueprint if item.block_id in block_ids],
        "block_scopes": [item.model_dump(mode="json") for item in design.block_scopes
                         if item.block_id in block_ids],
        "lesson_index_offset": offset, "allowed_component_types": ALLOWED_TYPES,
        "token_allowance": {"input_tokens": 400_000, "output_tokens": 131_072},
        "remaining_budget_ms": 500_000, "design_hash": design.design_hash,
    })
    return context, plan, shard_facts


__all__ = ["MODULE_RESPONSES", "chapter_plan", "key", "module_context"]
