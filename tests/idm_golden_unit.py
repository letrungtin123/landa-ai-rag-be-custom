"""Unit contracts with an IDM brief for the golden fixture, built the way Node builds them.

Mirrors ``prepareOrchestrationV2UnitGenerationContract`` + the IDM brief rules of
spec §8.3 closely enough for Python-side tests; the Node builder is the
production authority and the cross-language test checks parity.
"""

from __future__ import annotations

import hashlib
from typing import Any

from app.idm.contracts import IdmCourseDesignV1, IdmShardDesignV1, brief_hash_of
from app.lesson_author_orchestration_v2 import canonical_hash
from app.lesson_author_orchestration_v2_provider import SourceSnapshotFactV2

PURPOSE_BY_TYPE = {"html": "explain", "problem": "assess", "la_sortable": "sequence", "la_diagram": "relationship",
                   "la_crossword": "terminology", "la_faq": "clarify"}
ASSEMBLY_HASH = hashlib.sha256(b"idm-golden-assembly").hexdigest()


def build_unit_request(
    design: IdmCourseDesignV1,
    shard: IdmShardDesignV1,
    *,
    chapter_index: int,
    lesson_position: int,
    unit_position: int,
    facts: list[SourceSnapshotFactV2],
    tenant_id: str = "00000000-0000-4000-8000-0000000000a1",
    remaining_ms: int = 300_000,
) -> dict[str, Any]:
    lesson = shard.lessons[lesson_position]
    unit = lesson.units[unit_position]
    scope_of_block = {scope.block_id: scope.scope_key for scope in design.block_scopes}
    block_facts = {block.block_id: list(block.fact_keys) for block in design.blocks}
    rows = {row.block_id: row for row in design.blueprint}
    fact_by_key = {fact.fact_key: fact for fact in facts}
    unit_fact_keys = [key for block_id in unit.block_ids for key in block_facts[block_id]]
    source_facts = [fact_by_key[key].model_copy(update={"scope_key": scope_of_block[block_id]}).model_dump(mode="json")
                    for block_id in unit.block_ids for key in block_facts[block_id]]
    refs = [f"lo_{position}" for position in range(1, len(lesson.learning_objectives) + 1)]
    owned: set[str] = set()
    plans = []
    for component in unit.components:
        facts_in_scope = [key for block_id in component.block_ids for key in block_facts[block_id]]
        own = [key for key in unit_fact_keys if key in facts_in_scope and key not in owned]
        support = [key for key in unit_fact_keys if key in facts_in_scope and key in owned]
        owned.update(own)
        plan_id = "cp2_" + hashlib.sha256(
            f"{lesson.lesson_key}:{unit.unit_index}:{component.component_index}".encode()).hexdigest()[:32]
        plans.append({
            "component_plan_id": plan_id, "type": component.type, "title": component.title,
            "rationale": component.rationale, "purpose": PURPOSE_BY_TYPE[component.type],
            "source_fact_ids": own, "supporting_evidence_fact_ids": support, "learning_objective_refs": refs,
            "source_scope_ids": [scope_of_block[block_id] for block_id in component.block_ids],
            "content_requirements": [], "learning_block_ids": [], "required_artifacts": [],
        })
    practices = {practice.practice_id: practice for practice in lesson.practice_tasks}
    module = design.modules[chapter_index]
    lessons_in_course = [item for item in module.lessons]
    position = next(index for index, item in enumerate(lessons_in_course) if item.lesson_key == lesson.lesson_key)
    lesson_blocks = [block_id for item in lesson.units for block_id in item.block_ids]
    context_keys = [key for block_id in lesson_blocks if block_id not in unit.block_ids
                    for key in block_facts[block_id]]
    brief: dict[str, Any] = {
        "pipeline_version": "idm-1", "course_title": "Xử lý khiếu nại khách hàng đúng quy trình",
        "target_audience": design.target_audience.description, "module_title": module.title,
        "lesson_title": lesson.title, "lesson_objective": lesson.objective,
        "lesson_practice_sentences": [practice.sentence for practice in lesson.practice_tasks][:3],
        "previous_lesson_title": lessons_in_course[position - 1].title if position > 0 else None,
        "next_lesson_title": lessons_in_course[position + 1].title if position + 1 < len(lessons_in_course) else None,
        "unit_segment": unit.segment, "unit_purpose": unit.purpose,
        "components": [{
            "component_plan_id": plan["component_plan_id"], "type": component.type, "role": component.role,
            "title": component.title,
            "support_items": [item.model_dump(mode="json") for item in component.support_items],
            "practice": practices[component.practice_id].model_dump(mode="json") if component.practice_id else None,
            "treatments": [{"block_id": block_id, "treatment": rows[block_id].treatment,
                            "detail_level": rows[block_id].detail_level} for block_id in component.block_ids],
            "owned_fact_keys": plan["source_fact_ids"], "supporting_fact_keys": plan["supporting_evidence_fact_ids"],
        } for component, plan in zip(unit.components, plans, strict=True)],
        "lesson_context_facts": [{"fact_key": key, "fact_text": fact_by_key[key].fact_text}
                                 for key in context_keys][:80],
        "job_aid_signpost": None,
    }
    brief["brief_hash"] = brief_hash_of(brief)
    contract: dict[str, Any] = {
        "contract_version": 2, "unit_content_policy_version": "unit-content-v4-alignment-1",
        "source_snapshot_hash": design.source_snapshot_hash, "assembly_hash": ASSEMBLY_HASH,
        "chapter_key": shard.chapter_key,
        "unit_path": f"chapter_{chapter_index + 1}.lesson_{lesson_position + 1}.unit_{unit_position + 1}",
        "chapter_title": module.title, "lesson_title": lesson.title,
        "lesson_learning_objectives": list(lesson.learning_objectives), "unit_title": unit.title,
        "unit_purpose": unit.purpose, "unit_learning_objective_refs": refs,
        "unit_source_scope_ids": [scope_of_block[block_id] for block_id in unit.block_ids],
        "unit_source_fact_ids": unit_fact_keys, "component_plan": plans, "source_facts": source_facts,
        "idm_unit_brief": brief,
    }
    contract["contract_hash"] = canonical_hash(contract)
    return {
        "tenant_id": tenant_id, "kb_id": "00000000-0000-4000-8000-0000000000b1",
        "conversation_id": "00000000-0000-4000-8000-0000000000c1", "target": "lesson_author",
        "model": "gemini-3.8-flash", "max_output_tokens": 65_536, "embedding_model": "gemini-embedding-001",
        "system_prompt": "x", "user_message": "Generate unit", "locale": "vi",
        "correlation_id": "00000000-0000-4000-8000-0000000000e1", "api_key": "test-provider-key",
        "source_documents": [{"document_id": facts[0].document_id, "kb_id": "00000000-0000-4000-8000-0000000000b1",
                              "name": "quy-trinh-khieu-nai.pdf", "type": "pdf", "status": "ready"}],
        "contract_version": 2, "unit_contract": contract, "remaining_workflow_budget_ms": remaining_ms,
    }


def writer_response_escalate_practice(plans: list[dict[str, Any]]) -> dict[str, Any]:
    """Writer answer for lsn_003 unit 2: scenario MCQ + FAQ on common mistakes."""

    return {"components": {
        "c0": {
            "title": "Khách VIP phàn nàn về hoá đơn sai",
            "selection_rationale": "Tình huống quyết định escalate theo tiêu chí bắt buộc.",
            "covered_source_fact_ids": plans[0]["source_fact_ids"],
            "problem_type": "multiple_choice",
            "question": "Một khách hàng VIP gọi đến, cho biết sản phẩm vừa giao gây chập điện và có nguy cơ cháy. "
                        "Khách yêu cầu được hoàn tiền ngay. Bạn nên làm gì trước tiên?",
            "choices": [
                {"text": "Hứa hoàn tiền ngay để giữ chân khách VIP", "correct": False},
                {"text": "Escalate ngay cho quản lý vì khiếu nại liên quan đến an toàn", "correct": True},
                {"text": "Chỉ thông báo trưởng nhóm vì đây là khách VIP", "correct": False},
            ],
            "explanation": "Tiêu chí: khiếu nại liên quan an toàn bắt buộc phải escalate. A — sai vì không được hứa "
                           "bồi thường trước khi phân loại mức độ; B — đúng vì yếu tố an toàn là tiêu chí bắt buộc; "
                           "C — sai vì quy định VIP chỉ áp dụng khi không có yếu tố bắt buộc escalate.",
        },
        "c1": {
            "title": "Những nhầm lẫn thường gặp khi xử lý khiếu nại",
            "selection_rationale": "Làm rõ lỗi thường gặp liên quan quyết định escalate.",
            "covered_source_fact_ids": [],
            "items": [
                {"question": "Có nên hứa hoàn tiền khi khách vừa gọi đến không?",
                 "answer": "Không. Cần xác minh đơn hàng và phân loại mức độ trước, nếu không công ty có thể phải "
                           "chi trả cho đơn hàng không thuộc diện bồi thường."},
                {"question": "Vì sao khiếu nại cấp 3 hay bị ghi nhầm thành cấp 2?",
                 "answer": "Vì nhân viên bỏ qua yếu tố truyền thông. Hãy luôn kiểm tra yếu tố an toàn, pháp lý và "
                           "truyền thông trước khi chốt cấp độ."},
            ],
        },
    }}
