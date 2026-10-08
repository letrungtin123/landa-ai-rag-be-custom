"""Deterministic source-locked components of orchestration-v2 units."""

from __future__ import annotations

import re
from typing import Any, Callable

from app.instructional_quality import (
    build_source_grounded_single_choice,
    clean_source_facts,
    ordered_source_steps,
    render_source_locked_html,
    source_relationship_pairs,
    source_term_definitions,
)
from app.learner_content_purity import sanitize_source_fact_for_learner
from app.lesson_author_orchestration_v2_provider import UnitComponentPlanV2, UnitGenerationContractV2
from app.services.lesson_author.staged.source_locked import (
    _source_locked_faq_answers,
    _source_locked_faq_question,
    _source_locked_html_facts,
)


def _orchestration_v2_source_fragments(values: list[str], *, minimum: int = 3) -> list[str]:
    """Return deterministic, source-ordered display fragments without inventing facts."""

    fragments: list[str] = []
    seen: set[str] = set()
    for value in values:
        for candidate in _source_locked_html_facts([value]):
            for sentence in re.split(r"(?<=[.!?;])\s+|[\r\n]+", candidate):
                normalized = re.sub(r"\s+", " ", sentence).strip(" -\u2022")
                key = normalized.casefold()
                if normalized and key not in seen:
                    seen.add(key)
                    fragments.append(normalized[:500])
    if not fragments:
        return []
    if len(fragments) >= minimum:
        return fragments
    words = re.sub(r"\s+", " ", " ".join(values)).strip().split(" ")
    if len(words) >= minimum * 3:
        width = max(3, (len(words) + minimum - 1) // minimum)
        for offset in range(0, len(words), width):
            candidate = " ".join(words[offset:offset + width]).strip()
            key = candidate.casefold()
            if candidate and key not in seen:
                seen.add(key)
                fragments.append(candidate[:500])
            if len(fragments) >= minimum:
                break
    return fragments


def _orchestration_v2_sortable_steps(fact_texts: list[str], locale: str) -> list[str]:
    """Return only a real, source-evidenced sequence.

    Fabricating review actions from arbitrary prose made the interaction valid
    structurally but instructionally meaningless. Unsupported sortable plans
    are removed before the immutable unit contract is published.
    """

    return ordered_source_steps(fact_texts, locale=locale)


def _orchestration_v2_crossword_words(
    title: str,
    fact_texts: list[str],
    locale: str,
) -> list[dict[str, Any]]:
    """Build crossword rows only from explicit source term definitions."""

    return [
        {"answer": answer, "clue": definition, "hint": None}
        for answer, definition in source_term_definitions(fact_texts)
    ]


def _orchestration_v2_source_locked_html(
    title: str,
    fact_texts: list[str],
    locale: str,
    required_artifacts: list[dict[str, Any]] | None = None,
    *,
    renderer: Callable[..., str] = render_source_locked_html,
) -> str:
    """Render source facts once, preserving semantic tables and lists."""

    return renderer(
        title,
        fact_texts,
        locale=locale,
        required_artifacts=required_artifacts or [],
    )


def _orchestration_v2_source_relationship_diagram(
    title: str,
    fact_texts: list[str],
    locale: str,
) -> dict[str, Any]:
    """Build a bounded diagram without inferring relationships from proximity.

    Explicit table pairs and arrow/include relations are authoritative.  A
    numbered source procedure may still use the existing sequential rendering.
    The final star-shaped branch exists only for replaying a legacy approved
    contract; the CP10 architecture compiler no longer selects a new diagram
    when neither an explicit relationship nor sequence exists.
    """

    relationships = source_relationship_pairs(fact_texts)
    if relationships:
        nodes: list[dict[str, Any]] = []
        node_indexes: dict[str, int] = {}

        def node_index(label: str, shape: str) -> int:
            key = re.sub(r"[^\w\d]+", " ", label.casefold()).strip()
            existing = node_indexes.get(key)
            if existing is not None:
                return existing
            node_indexes[key] = len(nodes)
            nodes.append({"label": label[:500], "shape": shape, "tooltip": label[:500]})
            return len(nodes) - 1

        edges: list[dict[str, Any]] = []
        for left, right, relation in relationships:
            source = node_index(left, "rectangle")
            target = node_index(right, "rounded")
            edges.append({
                "source": source,
                "target": target,
                **({"label": relation[:120]} if relation else {}),
            })
        return {"name": title, "nodes": nodes, "edges": edges}

    sequence = ordered_source_steps(fact_texts, locale=locale)
    if sequence:
        nodes = [
            {"label": label[:500], "shape": "rounded", "tooltip": label[:500]}
            for label in sequence[:10]
        ]
        return {
            "name": title,
            "nodes": nodes,
            "edges": [
                {
                    "source": index,
                    "target": index + 1,
                    "label": "Next" if locale == "en" else "Tiếp theo",
                }
                for index in range(len(nodes) - 1)
            ],
        }

    fragments = _orchestration_v2_source_fragments(fact_texts)
    child_labels = fragments[:6] or [fact_texts[0][:500]]
    nodes = [{"label": title[:500], "shape": "ellipse", "tooltip": title[:500]}]
    nodes.extend(
        {"label": label[:500], "shape": "rounded", "tooltip": label[:500]}
        for label in child_labels
    )
    return {
        "name": title,
        "nodes": nodes,
        "edges": [
            {
                "source": 0,
                "target": child_index,
                "label": "Content" if locale == "en" else "Nội dung",
            }
            for child_index in range(1, len(nodes))
        ],
    }


def build_orchestration_v2_source_locked_unit(
    contract: UnitGenerationContractV2,
    locale: str,
    *,
    components: list[dict[str, Any] | None] | None = None,
) -> dict[str, Any] | None:
    """Build an exact V2 component inventory from its immutable source ledger.

    This path is intentionally deterministic and provider-free. It preserves
    every server-owned component instance and only places escaped source text
    or generic learner instructions into payloads. The normal Python and Node
    validators remain authoritative before anything can be committed.
    """

    slots = (build_orchestration_v2_source_locked_components(contract, locale)
             if components is None else components)
    if len(slots) != len(contract.component_plan) or any(component is None for component in slots):
        return None
    return {
        "title": (
            sanitize_source_fact_for_learner(contract.unit_title)
            or ("Learning content" if locale == "en" else "Nội dung học tập")
        ),
        "source_fact_ids": list(contract.unit_source_fact_ids),
        "supporting_evidence_fact_ids": list(dict.fromkeys(
            fact_id
            for plan in contract.component_plan
            for fact_id in plan.supporting_evidence_fact_ids
        )),
        "component_plan": [plan.model_dump(mode="json") for plan in contract.component_plan],
        "components": [component for component in slots if component is not None],
        "source_locked_fallback": True,
    }


def build_orchestration_v2_source_locked_components(
    contract: UnitGenerationContractV2,
    locale: str,
    *,
    html_renderer: Callable[..., str] = render_source_locked_html,
    single_choice_builder: Callable[..., dict[str, Any] | None] = build_source_grounded_single_choice,
    diagram_builder: Callable[..., dict[str, Any] | None] | None = None,
    faq_builder: Callable[..., list[dict[str, str]] | None] | None = None,
) -> list[dict[str, Any] | None]:
    """One source-locked component per server-owned plan slot, in plan order.

    A slot is ``None`` when its locked evidence cannot rebuild that component
    type deterministically (for example a prose-only ``problem`` or a ``la_faq``
    without explicit source conditions). Other slots stay usable on their own.
    The defaults are the legacy builders; IDM units inject their own variants
    (``app.idm.source_locked``) so legacy output stays byte-identical.
    """

    fact_by_id = {fact.fact_key: fact.fact_text for fact in contract.source_facts}
    return [_orchestration_v2_source_locked_component(contract, plan, fact_by_id, locale,
                                                      html_renderer=html_renderer,
                                                      single_choice_builder=single_choice_builder,
                                                      diagram_builder=diagram_builder, faq_builder=faq_builder)
            for plan in contract.component_plan]


def _orchestration_v2_source_locked_component(
    contract: UnitGenerationContractV2,
    plan: UnitComponentPlanV2,
    fact_by_id: dict[str, str],
    locale: str,
    *,
    html_renderer: Callable[..., str] = render_source_locked_html,
    single_choice_builder: Callable[..., dict[str, Any] | None] = build_source_grounded_single_choice,
    diagram_builder: Callable[..., dict[str, Any] | None] | None = None,
    faq_builder: Callable[..., list[dict[str, str]] | None] | None = None,
) -> dict[str, Any] | None:
    evidence_fact_ids = list(dict.fromkeys([
        *plan.source_fact_ids,
        *plan.supporting_evidence_fact_ids,
    ]))
    fact_texts = clean_source_facts(
        [fact_by_id[fact_id] for fact_id in evidence_fact_ids if fact_id in fact_by_id],
        preserve_table_numeric=True,
    )
    if not fact_texts or any(not value for value in fact_texts):
        return None
    title = (
        sanitize_source_fact_for_learner(plan.title)
        or sanitize_source_fact_for_learner(contract.unit_title)
        or ("Learning content" if locale == "en" else "Nội dung học tập")
    )
    rationale = ("Provider output was unavailable; this reviewable draft is reconstructed from the locked source facts."
                 if locale == "en" else
                 "Kết quả từ mô hình chưa khả dụng; bản nháp để rà soát này được dựng từ các dữ kiện nguồn đã khóa.")
    component: dict[str, Any] = {
        "type": plan.type,
        "title": title,
        "component_plan_id": plan.component_plan_id,
        "source_fact_ids": list(plan.source_fact_ids),
        "covered_source_fact_ids": list(plan.source_fact_ids),
        "supporting_evidence_fact_ids": list(plan.supporting_evidence_fact_ids),
        "learning_objective_refs": list(plan.learning_objective_refs),
        "source_locked_fallback": True,
        "selection_rationale": rationale,
    }
    if plan.type == "html":
        component["html"] = _orchestration_v2_source_locked_html(
            title,
            fact_texts,
            locale,
            plan.required_artifacts,
            renderer=html_renderer,
        )
    elif plan.type == "problem":
        problem = single_choice_builder(title, fact_texts, locale=locale)
        if problem is None:
            return None
        component.update(problem)
    elif plan.type == "la_faq" and faq_builder is not None:
        # IDM units inject a builder that also uses labelled source definitions (run c2e5ac41).
        items = faq_builder(title, fact_texts, locale=locale)
        if items is None:
            return None
        component["items"] = items
    elif plan.type == "la_faq":
        answers = _source_locked_faq_answers(fact_texts)
        if len(answers) < 2:
            return None
        component["items"] = [{
            "question": _source_locked_faq_question(answer, title, item_index, locale),
            "answer": answer,
        } for item_index, answer in enumerate(answers)]
    elif plan.type == "la_sortable":
        source_steps = _orchestration_v2_sortable_steps(fact_texts, locale)
        if len(source_steps) < 3:
            return None
        component.update({
            "question_text": ("Arrange the procedure steps in the correct order."
                              if locale == "en" else "Sắp xếp các bước thực hiện theo đúng trình tự."),
            "items": [{"text": item} for item in source_steps[:10]],
        })
    elif plan.type == "la_crossword":
        words = _orchestration_v2_crossword_words(title, fact_texts, locale)
        if len(words) < 3:
            return None
        component["words"] = words
    elif plan.type == "la_diagram":
        if diagram_builder is None:
            component.update(_orchestration_v2_source_relationship_diagram(
                title,
                fact_texts,
                locale,
            ))
        else:
            # IDM units inject a step-only builder that returns None when the evidence has no steps.
            diagram = diagram_builder(title, fact_texts, locale)
            if diagram is None:
                return None
            component.update(diagram)
    else:
        return None
    return component
