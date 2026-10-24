"""Small prompt contracts; no provider, retrieval or acceptance policy here."""
import re
from typing import Any, Literal

from app.core.text import casefold_strip_marks_and_d
# Moved in refactor wave 3b; re-exported here for the legacy pipeline and its tests.
from app.content.teaching_policy import (
    LESSON_INSTRUCTIONAL_POLICY_VERSION as LESSON_INSTRUCTIONAL_POLICY_VERSION,
    lesson_instructional_quality_policy as lesson_instructional_quality_policy,
)
from app.workflows.contracts import WorkflowFailure

ARCHITECT_POLICY_MAX_CHARS = 12_000
LESSON_LANGUAGE_POLICY_VERSION = "lesson-language-1"
_COMPONENT_PURPOSES = {
    "html": "Teach assigned knowledge, preserving conditions and distinctions. Explain once; use a table or list instead of repeating the same paragraph.",
    "problem": "Assess the mapped learner action using knowledge taught here; explain the answer from evidence, not an invented exception.",
    "la_faq": "Clarify source-supported distinctions or likely confusion; do not copy nearby teaching. FAQ stays last in the unit.",
    "la_crossword": "Reinforce evidenced terminology only. Accurate clues are not proof that the learner has learned every procedure or condition associated with the terms.",
    "la_diagram": "Show evidenced membership, hierarchy or relationships with accurate edge meaning; do not infer causation or order from proximity.",
    "la_sortable": "Practice a source-established sequence. Do not impose order on independent precautions or unrelated sections.",
}


def _ids(value: Any) -> list[str]:
    return list(dict.fromkeys(item for item in value if isinstance(item, str) and item)) if isinstance(value, list) else []


def instructional_action_intents(text: str) -> list[str]:
    """Conservative EN/VI lexical hints, not a cognitive/psychometric score."""
    folded = casefold_strip_marks_and_d(text)
    patterns = {
        "calculation": r"\b(?:calculate|compute|tinh toan|tinh diem)\b",
        "analysis": r"\b(?:analyse|analyze|phan tich)\b",
        "selection": r"\b(?:select|choose|lua chon|chon bien phap)\b",
        "procedure": r"\b(?:perform|thuc hien)\b",
    }
    return [name for name, pattern in patterns.items() if re.search(pattern, folded)]


def component_instructional_brief(expected: dict[str, Any]) -> dict[str, Any]:
    """Read-only projection of the approved plan, not a planner or fact allocator.

    Objectives already appear in the instructional contract; reference that local
    array rather than repeating private text per component. No fuzzy ID recovery.
    Missing bindings stay visible and must not become fabricated associations.
    """
    objectives = expected.get("learning_objectives") or []
    valid_refs = {f"lo_{i}" for i, objective in enumerate(objectives, 1)
                  if isinstance(objective, str) and objective.strip()}
    blocks = {b.get("id"): b for b in expected.get("learning_blocks", []) if isinstance(b, dict)}
    components = []
    for index, plan in enumerate(expected.get("component_plan", [])):
        if not isinstance(plan, dict):
            continue
        refs = _ids(plan.get("learning_objective_refs"))
        origin = "component_plan"
        if not refs:
            refs = list(dict.fromkeys(ref for block_id in _ids(plan.get("learning_block_ids"))
                                     for ref in _ids(blocks.get(block_id, {}).get("learning_objective_refs"))))
            origin = "linked_learning_blocks" if refs else "unspecified"
        kind = plan.get("type")
        components.append({
            "component_index": index,
            "type": kind,
            "objective_binding_origin": origin,
            "local_objective_refs": [ref for ref in refs if ref in valid_refs],
            "objective_actions": [{"objective_ref": ref,
                                   "actions": instructional_action_intents(objectives[int(ref[3:]) - 1])}
                                  for ref in refs if ref in valid_refs],
            "unresolved_objective_ref_count": sum(ref not in valid_refs for ref in refs),
            "owned_fact_count": len(_ids(plan.get("source_fact_ids"))),
            "supporting_fact_count": len(_ids(plan.get("supporting_evidence_fact_ids"))),
            "instructional_role": (
                "teach" if kind in {"html", "la_diagram"}
                else "check" if kind == "problem"
                else "practice" if kind in {"la_sortable", "la_crossword"}
                else "clarify"
            ),
            "checklist": _COMPONENT_PURPOSES.get(kind, "Follow the selected component contract; do not invent evidence."),
        })
    return {"version": LESSON_INSTRUCTIONAL_POLICY_VERSION,
            "authority": "read_only_approved_plan",
            "unit_action_hints": instructional_action_intents(str(expected.get("unit_title") or "") + " " + str(expected.get("unit_purpose") or "")),
            "objective_reference_rule": "lo_N addresses item N in lesson_learning_objectives; never assign missing links yourself.",
            "coverage_rule": "Canonical ownership and declared coverage are contract metadata, not proof of semantic completeness. Keep them unchanged. Every owning component must represent its facts through the selected treatment; supporting evidence alone never proves coverage.",
            "components": components}


def instructional_contract_review_signals(expected: dict[str, Any]) -> dict[str, Any]:
    """Safe structural review signals, NOT a factual evaluator or acceptance gate.

    Deliberately accepts only the server plan: no provider content or source text
    can enter diagnostics. These observations cannot prove semantic alignment.
    """
    brief = component_instructional_brief(expected)
    findings = []
    for component in brief["components"]:
        codes = []
        if component["unresolved_objective_ref_count"]:
            codes.append("OBJECTIVE_BINDING_UNRESOLVED_REVIEW")
        if component["type"] == "problem" and not component["local_objective_refs"]:
            codes.append("ASSESSMENT_OBJECTIVE_BINDING_MISSING_REVIEW")
        for code in codes:
            findings.append({"code": code, "severity": "review", "component_index": component["component_index"],
                             "path": f"components[{component['component_index']}]",
                             "owned_fact_count": component["owned_fact_count"],
                             "supporting_fact_count": component["supporting_fact_count"]})
    return {"policy_version": LESSON_INSTRUCTIONAL_POLICY_VERSION,
            "diagnostic_scope": "approved_plan_structure_only", "blocking": False,
            "semantic_fidelity": "not_measured", "semantic_coverage": "not_measured",
            "component_count": len(brief["components"]), "review_signal_count": len(findings), "findings": findings}


def bounded_architect_policy(policy: str) -> str:
    """Node composes whole teaching sections. Never truncate its safety policy."""
    policy = policy.strip()
    if len(policy) > ARCHITECT_POLICY_MAX_CHARS:
        raise WorkflowFailure(
            "ARCHITECTURE_SCOPE_INCOMPLETE",
            "The server architecture policy exceeds its bounded context allowance.",
            internal_code="ARCH_POLICY_BUDGET_EXCEEDED",
            failure_stage="course_architect_policy_composition",
            diagnostics={"policy_chars": len(policy), "policy_max_chars": ARCHITECT_POLICY_MAX_CHARS},
        )
    return policy


def lesson_output_language_policy(locale: Literal["vi", "en"]) -> str:
    language = "Vietnamese (vi), with diacritics" if locale == "vi" else "English (en)"
    return "\n".join([
        f"SERVER OUTPUT LANGUAGE POLICY: {LESSON_LANGUAGE_POLICY_VERSION}.",
        f"Write newly generated learner-facing text in {language}, regardless of source language or UI labels.",
        "This applies to explanations, headings, examples, FAQ questions/answers, problem questions/choices/feedback, crossword clues, diagram labels, and sortable instructions/items.",
        "Preserve approved course/chapter/lesson/unit titles exactly. Do not translate or change IDs, keys, component_plan_id, block/objective/concept IDs, source refs, canonical fact IDs, PRIMARY/SUPPORTING evidence or component types. Language is not authority to change the approved scope or plan.",
        "Preserve proper names and source terminology when needed. Crossword clues must describe the exact answer term and its normalized spelling; do not translate an answer independently from its clue. Preserve quoted evidence in its source language and identify it as a source quotation, not a translated factual claim.",
        "Do not add factual claims during translation. Respect the same source, schema, semantic HTML and repair-target restrictions. No styles, arbitrary classes, scripts or media assets.",
    ])
