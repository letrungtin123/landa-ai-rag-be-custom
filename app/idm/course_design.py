"""IDM ``course_skeleton`` task: W0/W1 → W1-reduce → W2 → W4 → assembly (spec §5.1, §7.1-§7.5)."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Final

from app.idm.architecture import AssemblyInput, assemble_idm_course_design, project_course_skeleton, run_w4
from app.idm.blueprint import blocked_must_dos, build_dispositions, hold_items, run_w2
from app.idm.content_map import (
    FactIndex,
    apply_w1_reduce,
    map_sections,
    page_furniture_keys,
    rekey_blocks,
    run_w1_reduce,
)
from app.idm.contracts import IdmCourseSkeletonRequestV1, StageOrigin
from app.idm.policy import (
    IDM_FALLBACK_SECTION_SHARE_FOR_STAGE_FALLBACK,
    IDM_FALLBACK_STAGES_FOR_REVIEW,
    IDM_MAX_BLOCKS,
    IDM_SINGLE_TASK_MAX_SOURCE_CHARS,
    IDM_W1_REDUCE_MAX_OUTPUT_TOKENS,
    IDM_W2_MAX_OUTPUT_TOKENS,
    IDM_W4_COURSE_MAX_OUTPUT_TOKENS,
)
from app.idm.runtime import IdmRuntime, IdmStageError, idm_tail_reserve, log_stage
from app.idm.signals import IdmCapacityError, SectionPlan, compute_fact_signals, plan_idm_sections
from app.lesson_author_orchestration_v2_provider import SourceSnapshotFactV2

_MS: Final = 1000


def _validate_source(request: IdmCourseSkeletonRequestV1) -> None:
    keys = [fact.fact_key for fact in request.source_facts]
    if len(keys) != len(set(keys)):
        raise IdmStageError("IDM_SOURCE_FACTS_INVALID")
    if sum(len(fact.fact_text) for fact in request.source_facts) > IDM_SINGLE_TASK_MAX_SOURCE_CHARS:
        raise IdmCapacityError("IDM_SOURCE_EXCEEDS_SINGLE_TASK_CAPACITY")


def _prepare(
    facts: list[SourceSnapshotFactV2], document_names: dict[str, str], max_sections: int,
) -> tuple[SectionPlan, FactIndex, set[str]]:
    """CPU-bound signal/section work (seconds on large sources), run in a worker thread."""

    signals = compute_fact_signals(facts)
    section_plan = plan_idm_sections(facts, signals, document_names, max_sections=max_sections)
    return section_plan, FactIndex.build(facts, signals), page_furniture_keys(facts)


async def run_idm_course_design(
    request: IdmCourseSkeletonRequestV1,
    *,
    source_snapshot_hash: str,
    runtime: IdmRuntime,
    parallelism: int,
    max_sections: int,
) -> dict[str, Any]:
    """Return the ``/course-skeleton`` response: legacy envelope plus ``idm`` (spec §5.6)."""

    started = time.perf_counter()
    _validate_source(request)
    context = request.project_context
    facts = request.source_facts
    document_names = {item.document_id: item.name for item in context.source_documents}
    section_plan, index, furniture = await asyncio.to_thread(_prepare, facts, document_names, max_sections)
    log_stage("idm_stage_started", {"correlation_id": runtime.correlation_id, "stage": "idm_course_design",
                                    "section_count": len(section_plan.sections), "fact_count": len(facts)})

    # Later stages keep their minimum admissible share of the task allowance (runtime.idm_tail_reserve).
    tail = idm_tail_reserve(IDM_W1_REDUCE_MAX_OUTPUT_TOKENS, IDM_W2_MAX_OUTPUT_TOKENS,
                            IDM_W4_COURSE_MAX_OUTPUT_TOKENS)
    section_results = await map_sections(
        runtime, section_plan.sections, index, context.model_dump(mode="json"), furniture,
        parallelism=parallelism, tail_reserve_tokens=tail,
    )
    blocks, noise = rekey_blocks(section_results, index)
    if not blocks:
        raise IdmStageError("IDM_COURSE_HAS_NO_TEACHABLE_CONTENT")
    if len(blocks) > IDM_MAX_BLOCKS:
        raise IdmCapacityError("IDM_SOURCE_EXCEEDS_SINGLE_TASK_CAPACITY")
    fallback_sections = sum(result.origin != "provider" for result in section_results)
    w1_origin: StageOrigin = (
        "provider" if fallback_sections == 0 else
        "deterministic_fallback" if fallback_sections == len(section_results) else "partial_fallback")

    reduce_response, reduce_origin, reduce_codes = await run_w1_reduce(
        runtime, blocks, section_plan.sections, context,
        tail_reserve_tokens=idm_tail_reserve(IDM_W2_MAX_OUTPUT_TOKENS, IDM_W4_COURSE_MAX_OUTPUT_TOKENS),
        index=index,
    )
    blocks, links = apply_w1_reduce(reduce_response, blocks, index, context.locale)
    objectives = list(reduce_response.learning_objectives)
    must_dos = list(reduce_response.must_dos)

    blueprint = await run_w2(
        runtime, context=context, blocks=blocks, links=links, objectives=objectives, must_dos=must_dos,
        index=index, tail_reserve_tokens=idm_tail_reserve(IDM_W4_COURSE_MAX_OUTPUT_TOKENS),
    )
    blocked = blocked_must_dos(blueprint.rows, must_dos)
    holds = hold_items(blueprint.rows, blueprint.blocks, blocked)
    dispositions = build_dispositions(blueprint.rows, blueprint.blocks, noise, index)

    block_chars = {block.block_id: sum(len(index.text[key]) for key in block.fact_keys)
                   for block in blueprint.blocks}
    architecture = await run_w4(
        runtime, context=context, audience=reduce_response.target_audience, objectives=objectives,
        must_dos=must_dos, blocks=blueprint.blocks, rows=blueprint.rows, blocked=blocked,
        block_chars=block_chars,
    )
    stage_origins: dict[str, StageOrigin] = {
        "w1_map": w1_origin, "w1_reduce": reduce_origin, "w2": blueprint.origin, "w4": architecture.origin,
    }
    design = await asyncio.to_thread(assemble_idm_course_design, AssemblyInput(
        source_snapshot_hash=source_snapshot_hash, context=context,
        snapshot_fact_keys=[fact.fact_key for fact in facts],
        fact_text={fact.fact_key: fact.fact_text for fact in facts},
        fact_source_ref={fact.fact_key: fact.source_ref for fact in facts},
        audience=reduce_response.target_audience, objectives=objectives, must_dos=must_dos,
        blocks=blueprint.blocks, links=blueprint.links, rows=blueprint.rows, blocked=blocked, holds=holds,
        plan=architecture.plan, dispositions=dispositions, stage_origins=stage_origins,
    ))
    skeleton = await asyncio.to_thread(project_course_skeleton, design, architecture.plan)

    fallback_stage_count = sum([
        fallback_sections / max(1, len(section_results)) > IDM_FALLBACK_SECTION_SHARE_FOR_STAGE_FALLBACK,
        reduce_origin == "deterministic_fallback",
        blueprint.origin == "deterministic_fallback",
        architecture.origin == "deterministic_fallback",
    ])
    reviewable = fallback_stage_count >= IDM_FALLBACK_STAGES_FOR_REVIEW
    usage_complete = runtime.usage.complete and runtime.provider_failure_code is None
    counts: dict[str, int] = {}
    for item in design.dispositions:
        counts[item.disposition] = counts.get(item.disposition, 0) + 1
    log_stage("idm_stage_completed", {
        "correlation_id": runtime.correlation_id, "stage": "idm_course_design",
        "section_count": len(section_plan.sections), "call_count": runtime.usage.calls,
        "duration_ms": int((time.perf_counter() - started) * _MS), "usage": runtime.usage.as_usage(),
        "origin": stage_origins, "block_count": len(design.blocks), "lesson_count": sum(
            len(module.lessons) for module in design.modules), "module_count": len(design.modules),
        "disposition_counts": counts, "section_warnings": section_plan.warnings,
        "error_codes": {"w1_reduce": reduce_codes, "w2": blueprint.codes, "w4": architecture.codes,
                        "w1_map": _merge([result.issue_codes for result in section_results])},
        "warnings": architecture.warnings, "response_adjustments": dict(sorted(runtime.adjustments.items())),
    })
    return {
        "contract_version": 2,
        "skeleton": skeleton.model_dump(mode="json"),
        "idm": design.model_dump(mode="json"),
        "usage": runtime.usage.as_usage(),
        "usage_complete": usage_complete,
        "usage_source": "provider" if usage_complete else "reserved_upper_bound",
        "content_origin": "structured_fallback" if reviewable else "provider_validated",
        "quality_state": "review_required" if reviewable else "validated",
    }


def _merge(items: list[dict[str, int]]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for item in items:
        for code, count in item.items():
            merged[code] = merged.get(code, 0) + count
    return dict(sorted(merged.items()))
