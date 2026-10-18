"""Mirror of Node's revision-0 unit acceptance, applied before an IDM unit is returned.

Node accepts a ``/unit`` answer only after it normalizes the unit
(``chat.service.normalizeLessonAuthorProposal``), checks it
(``lesson-author-content-contract.logic.lessonAuthorGeneratedUnitCoverageFinding``, the IDM W5
budget, ``lesson-author-workspace-component.logic.workspaceComponentContent``) and stops at the
first failure with ``ORCHESTRATION_V2_UNIT_BASELINE_INVALID`` (or ``..._NORMALIZATION_INVALID``).
The shared staged validator does not cover every one of those rules: run c2e5ac41 returned two
worksheet units that Python validated and Node rejected, and the unit was lost because the
``fallback_only`` retry had no fallback for its slots.

This module re-implements those checks on the unit exactly as Python is about to return it, so the
IDM writer can repair or replace the failing slot instead. Every finding carries the check and the
reason code Node logs (``acceptance_check`` / ``acceptance_code`` / ``acceptance_path`` in the
worker log) and, where Node only has a coarse code, a finer ``detail`` for the repair prompt.

JavaScript semantics are reproduced where they matter: string lengths in UTF-16 code units, the
ECMAScript white-space set, ``String.prototype.slice`` on code units, ASCII ``\\d``/``\\w``/``\\b``.
``tests/test_idm_node_acceptance.py`` pins the rules and the backend test
``lesson-author-idm-acceptance.cross-language.test.ts`` (through ``tests/idm_contract_bridge.py``) keeps
the verdicts in parity with the real Node code. Diagram geometry is computed by Node itself and is
not mirrored; only the diagram text rules are.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Final

from app.idm.node_acceptance_html import (
    _normalized_html,
    html_contract_code,
    html_quality_code,
    idm_budget_code,
    idm_html_visible_text,
    render_semantic_html,
    sanitize_lesson_author_html,
    workspace_html_code,
)
from app.idm.node_acceptance_problems import (
    PAYLOAD_INVALID,
    _problem_xml,
    plain_text_valid,
    single_choice_problem_code,
    workspace_problem_code,
)
from app.idm.node_acceptance_text import (
    AcceptanceCheck,
    AcceptanceContext,
    AcceptanceFinding,
    _list,
    _nullish,
    _record,
    _Rejected,
    instructional_fold,
    js_len,
    js_slice,
    visible_html_text,
)
from app.idm.node_acceptance_widgets import _crossword_findings, _diagram_findings, _faq_findings, _sortable_findings


# --- the unit ----------------------------------------------------------------------------------------
def _exact_ids(value: Any, expected: Sequence[str]) -> bool:
    return (isinstance(value, list) and len(value) == len(expected) and all(isinstance(item, str) for item in value)
            and len(set(value)) == len(value) and all(item in expected for item in value))


def _response_code(component: dict[str, Any], context: AcceptanceContext, index: int) -> str | None:
    """``readOrchestrationV2UnitProviderResponse`` / ``acceptOrchestrationV2GeneratedUnit`` slot identity."""

    metadata = _record(component.get("metadata"))
    plan = context.plans[index]

    def claimed(name: str) -> Any:
        return _nullish(component.get(name), metadata.get(name))

    if (component.get("type") != plan.get("type")
            or claimed("component_plan_id") != plan.get("component_plan_id")):
        return "RESPONSE_COMPONENT_IDENTITY"
    owned = list(plan.get("source_fact_ids") or [])
    if (not _exact_ids(claimed("source_fact_ids"), owned) or not _exact_ids(claimed("covered_source_fact_ids"), owned)
            or not _exact_ids(claimed("supporting_evidence_fact_ids"),
                              list(plan.get("supporting_evidence_fact_ids") or []))):
        return "RESPONSE_FACT_IDS"
    refs = claimed("learning_objective_refs")
    if refs is not None and not _exact_ids(refs, list(plan.get("learning_objective_refs") or [])):
        return "RESPONSE_FACT_IDS"
    return None


_CHECKERS: Final[Mapping[str, Callable[[dict[str, Any]], None]]] = {
    "la_faq": _faq_findings, "la_sortable": _sortable_findings, "la_crossword": _crossword_findings,
    "la_diagram": _diagram_findings,
}


def _merge_nested(component: dict[str, Any]) -> dict[str, Any]:
    nested = _record(component.get("content"))
    return {**nested, **component} if nested else component


def node_acceptance_findings(unit: Mapping[str, Any], context: AcceptanceContext, *,
                             provider_validated: bool) -> list[AcceptanceFinding]:
    """Every rule Node would reject ``unit`` for, in Node's order (the first one is what Node reports)."""

    components = _list(unit.get("components"))
    if (unit.get("title") != context.unit_title or len(components) != len(context.plans)
            or not _exact_ids(unit.get("source_fact_ids"), list(context.unit_source_fact_ids))):
        return [AcceptanceFinding("response", "RESPONSE_UNIT_IDENTITY", None)]
    normalization: list[AcceptanceFinding] = []
    response: list[AcceptanceFinding] = []
    coverage: list[AcceptanceFinding] = []
    budget: list[AcceptanceFinding] = []
    workspace: list[AcceptanceFinding] = []
    for index, raw in enumerate(components):
        if (code := _response_code(_record(raw), context, index)) is not None:
            response.append(AcceptanceFinding("response", code, index))
        component = _merge_nested(_record(raw))
        kind = component.get("type")
        plan = context.plans[index]
        try:
            if kind == "html":
                html = _normalized_html(component)
            elif kind == "problem":
                xml = _problem_xml(component)
            elif isinstance(kind, str) and kind in _CHECKERS:
                _CHECKERS[kind](component)
            else:
                raise _Rejected("NORMALIZATION_COMPONENT_TYPE")
        except _Rejected as error:
            check: AcceptanceCheck = "workspace_component" if error.code.startswith("WORKSPACE_") else "normalization"
            target = workspace if check == "workspace_component" else normalization
            target.append(AcceptanceFinding(check, error.code, index, error.detail, error.items))
            continue
        if kind == "html":
            if (code := html_contract_code(html, plan.get("required_artifacts") or [])) is None \
                    and not context.preserve_source:
                code = html_quality_code(html, context.exact_identifiers, worksheet=index in context.worksheet_slots)
            if code is not None:
                coverage.append(AcceptanceFinding("coverage", code, index))
            if (detail := workspace_html_code(html)) is not None:
                workspace.append(AcceptanceFinding("workspace_component", PAYLOAD_INVALID, index, detail))
            if provider_validated and context.budget is not None and (
                    code := idm_budget_code(_record(raw), context.budget)) is not None:
                budget.append(AcceptanceFinding("idm_budget", code, index))
        elif kind == "problem":
            code = single_choice_problem_code(xml) if xml is not None else "PROBLEM_NOT_SINGLE_CHOICE"
            if code is not None:
                coverage.append(AcceptanceFinding("coverage", code, index))
            if xml is not None and (detail := workspace_problem_code(xml)) is not None:
                workspace.append(AcceptanceFinding("workspace_component", PAYLOAD_INVALID, index, detail))
    # Node rejects at the first failing stage; every finding of it is kept so all slots can be repaired.
    for stage in (normalization, response, coverage, budget, workspace):
        if stage:
            return [*stage, *(item for later in (normalization, response, coverage, budget, workspace)
                              if later is not stage for item in later)]
    return []


def acceptance_log(findings: Iterable[AcceptanceFinding], limit: int = 16) -> list[str]:
    return [finding.as_log() for finding in findings][:limit]


__all__ = [
    "AcceptanceContext", "AcceptanceFinding", "acceptance_log", "html_contract_code", "html_quality_code",
    "idm_html_visible_text", "instructional_fold", "js_len", "js_slice", "node_acceptance_findings",
    "plain_text_valid", "render_semantic_html", "sanitize_lesson_author_html", "single_choice_problem_code",
    "visible_html_text", "workspace_problem_code",
]
