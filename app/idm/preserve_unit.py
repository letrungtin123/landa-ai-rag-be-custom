"""Unit task of the "keep the document as it is" mode (QLT-6 ``preserve_source``): no provider call.

The unit's lines are rendered word for word (``app.idm.preserve_render``) into its single html slot, checked for the
exact text, then by the same structural rules Node's acceptance applies (``app.idm.node_acceptance`` with the
instructional-quality rules off, as Node does for a kept unit). The answer is the deterministic envelope Node admits
on its provider-free path (``fallback_only``): ``usage_source`` "deterministic_fallback", no usage. A unit that cannot
be kept exactly fails with a stable code instead of being changed.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Final

from app.idm.contracts import IdmFindingCountsV1, IdmUnitQualityV1
from app.idm.node_acceptance import acceptance_log, node_acceptance_findings
from app.idm.preserve_design import PRESERVE_MODE
from app.idm.preserve_render import KEPT_PREFIX, preserve_text_finding, purity_report, render_preserved_html
from app.idm.runtime import IdmStageError, log_stage
from app.idm.storyboard import acceptance_context, parse_brief
from app.lesson_author_orchestration_v2_provider import UnitGenerationContractV2

PRESERVE_SOURCE_CODE: Final = "IDM_PRESERVE_SOURCE"
UNIT_INVALID_CODE: Final = "IDM_PRESERVE_UNIT_INVALID"
TEXT_MISMATCH_CODE: Final = "IDM_PRESERVE_TEXT_MISMATCH"
_MAX_CODES: Final = 32
_MAX_LOGGED: Final = 16
_ZERO_USAGE: Final = {"inputTokens": 0, "outputTokens": 0, "embeddingTokens": 0, "totalTokens": 0}


def is_preserve_unit(contract: UnitGenerationContractV2) -> bool:
    """The unit belongs to a kept document (its brief carries the launch mode; strict parsing comes later)."""

    brief = contract.idm_unit_brief
    return isinstance(brief, dict) and brief.get("design_mode") == PRESERVE_MODE


def unit_note(locale: str, kept: bool) -> str:
    """The author note of a kept unit, in plain words."""

    if locale == "vi":
        note = "Nội dung giữ nguyên từng chữ theo tài liệu; AI không viết thêm gì."
        if kept:
            note += (" Một số chỗ thường được lược bỏ khi soạn bài (số trang, tên tệp, dòng ghi nguồn, đoạn lặp lại "
                     "hoặc dãy ký tự lặp) được giữ đúng như tài liệu; bạn có thể tự xoá khi duyệt nếu không cần.")
        return note
    note = "Content kept word for word from the document; AI added nothing."
    if kept:
        note += (" Some parts that are usually left out of lessons (page numbers, file names, source lines, repeated "
                 "passages or repeated characters) are kept exactly as in the document; you can delete them while "
                 "reviewing if they are not needed.")
    return note


def run_preserve_unit(*, contract: UnitGenerationContractV2, locale: str, fallback_only: bool,
                      correlation_id: str | None = None) -> dict[str, Any]:
    """The ``/unit`` answer of a kept unit."""

    brief = parse_brief(contract)
    plans = contract.component_plan
    if (brief.design_mode != PRESERVE_MODE or len(plans) != 1 or plans[0].type != "html"
            or list(plans[0].source_fact_ids) != list(contract.unit_source_fact_ids)
            or plans[0].supporting_evidence_fact_ids):
        raise IdmStageError(UNIT_INVALID_CODE)
    plan = plans[0]
    texts = [fact.fact_text for fact in contract.source_facts]
    html = render_preserved_html(contract.unit_title, texts)
    if preserve_text_finding(texts, html) is not None:
        raise IdmStageError(TEXT_MISMATCH_CODE)
    component = {
        "type": "html", "title": plan.title, "component_plan_id": plan.component_plan_id,
        "source_fact_ids": list(plan.source_fact_ids), "covered_source_fact_ids": list(plan.source_fact_ids),
        "supporting_evidence_fact_ids": [], "learning_objective_refs": list(plan.learning_objective_refs),
        "source_locked_fallback": True,
        "selection_rationale": ("Nội dung giữ nguyên từng chữ theo tài liệu." if locale == "vi"
                                else "Content kept word for word from the document."),
        "html": html,
    }
    unit: dict[str, Any] = {
        "title": contract.unit_title, "source_fact_ids": list(contract.unit_source_fact_ids),
        "supporting_evidence_fact_ids": [], "component_plan": [plan.model_dump(mode="json")],
        "components": [component],
    }
    context = dataclasses.replace(acceptance_context(contract, brief), budget=None, preserve_source=True)
    findings = node_acceptance_findings(unit, context, provider_validated=False)
    if findings:
        log_stage("idm_unit_node_acceptance", {"correlation_id": correlation_id, "unit_path": contract.unit_path,
                                               "stage": "preserve", "findings": acceptance_log(findings, _MAX_LOGGED)})
        raise IdmStageError(UNIT_INVALID_CODE)
    report = purity_report(texts, html, context.exact_identifiers)
    quality = IdmUnitQualityV1(
        judge_mode="off", judge_status="not_run", finding_counts=IdmFindingCountsV1(minor=0, major=0, critical=0),
        criteria={}, repair_applied=False, deterministic_codes=[PRESERVE_SOURCE_CODE, *report][:_MAX_CODES],
        author_note=unit_note(locale, bool(report)))
    unit["idm_quality"] = quality.model_dump(mode="json")
    log_stage("idm_unit_quality", {"correlation_id": correlation_id, "unit_path": contract.unit_path,
                                   "mode": PRESERVE_MODE, "fact_count": len(texts), "html_chars": len(html),
                                   "kept_codes": [code.removeprefix(KEPT_PREFIX) for code in report],
                                   "call_count": 0})
    return {
        "contract_version": 2, "source_snapshot_hash": contract.source_snapshot_hash,
        "unit_path": contract.unit_path, "unit": unit, "usage": dict(_ZERO_USAGE),
        # Node's provider-free path; a caller that still fenced a dispatch settles zero provider usage.
        "usage_complete": True, "usage_source": "deterministic_fallback" if fallback_only else "provider",
        "content_origin": "structured_fallback", "quality_state": "review_required", "attempt_trace": [],
    }
