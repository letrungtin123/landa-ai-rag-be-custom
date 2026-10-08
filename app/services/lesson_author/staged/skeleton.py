"""Staged skeleton: unit matching, blueprint-locked skeletons and unit batches."""

from __future__ import annotations

import json
import re
from typing import Any

from fastapi import HTTPException

from app.component_capabilities import validate_instance_plan
from app.instructional_density import INSTRUCTIONAL_DENSITY_POLICY_VERSION
from app.schemas.lesson_author import RagLessonAuthorRequest
from app.services.lesson_author.architecture_shape import (
    _is_v4_supporting_factless_unit,
    _is_v5_supporting_factless_unit,
)
from app.services.lesson_author.errors import LessonAuthorProposalValidationError
from app.services.lesson_author.staged.plan import (
    STAGED_LESSON_AUTHOR_CONTEXT_THRESHOLD,
    STAGED_LESSON_AUTHOR_MIN_OUTPUT_TOKENS,
    STAGED_LESSON_AUTHOR_UNITS_PER_BATCH,
    build_staged_component_plan,
    ensure_staged_chapter_component_diversity,
    normalize_staged_component_type,
)
from app.source_structure import strip_source_range_suffix


def parse_lesson_author_json_value(text: str, label: str) -> Any:
    """Parse the first complete JSON value without accepting trailing prose."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, (dict, list)):
            return value
    raise HTTPException(status_code=502, detail=f"AI không trả JSON {label} hợp lệ.")


def should_stage_lesson_author_proposal(
    request: RagLessonAuthorRequest,
    context: str,
) -> bool:
    """Stage chapter creation before the provider can truncate one large JSON."""
    # The backend owns the operation contract.  Do not let a heuristic silently
    # downgrade an explicitly staged chapter request back to one giant response.
    if request.generation_mode == "staged":
        return True
    if request.generation_mode == "single":
        return False
    if request.operation == "create" and request.target_type == "chapter":
        return True
    if request.max_output_tokens < STAGED_LESSON_AUTHOR_MIN_OUTPUT_TOKENS:
        return False
    # Blueprint-driven drafting carries the selected node in
    # target_scope_instruction while outline_context is intentionally empty.
    # For a new chapter request there may be no selected node at all; an
    # explicit create signal is enough to stage a large source context. Edit,
    # rename and delete requests stay on the normal path unless the server
    # supplied an authoritative scope.
    has_authoritative_scope = any(
        str(getattr(request, field, "") or "").strip()
        for field in ("outline_context", "target_scope_instruction")
    )
    has_create_signal = bool(re.search(
        r"\b(?:tạo|tao|soạn|soan|viết|viet|xây dựng|xay dung|bổ sung|bo sung|thêm|them|create|draft|add)\b",
        request.user_message,
        flags=re.IGNORECASE,
    ))
    has_chapter_signal = bool(re.search(
        r"\b(?:chương|chuong|chapter|section)\b",
        request.user_message,
        flags=re.IGNORECASE,
    ))
    if not has_authoritative_scope and not has_create_signal:
        return False
    if not has_chapter_signal:
        return False
    # A new chapter has no selected outline scope, so context size alone is
    # not a safe signal: the course context and schema can still push a single
    # response over the provider's structured-output limit. Stage all explicit
    # new-chapter requests; retain the size gate for scoped requests.
    if not has_authoritative_scope:
        return has_create_signal
    return has_create_signal and len(context) >= STAGED_LESSON_AUTHOR_CONTEXT_THRESHOLD


def lesson_author_title_key(value: Any) -> str:
    raw = strip_source_range_suffix(str(value or "")).casefold()
    raw = re.sub(r"^(?:chương|chuong|chapter|bài|bai|lesson|unit|mục|muc|section)\s+[0-9ivxlcdm.:-]+", "", raw)
    return re.sub(r"[^0-9a-zà-ỹ]+", " ", raw, flags=re.UNICODE).strip()


def staged_unit_candidates(value: Any) -> list[Any]:
    """Share the supported unit envelopes between initial and recovery parsing."""
    if isinstance(value, dict) and "units" in value:
        # Never accept an ambiguous unit plus batch envelope.
        if "title" in value or "components" in value:
            return []
        return value["units"] if isinstance(value["units"], list) else []
    if isinstance(value, dict):
        return [value]
    return value if isinstance(value, list) else []


def staged_unit_match_diagnostics(generated_units: list[Any], expected_title: str) -> dict[str, Any]:
    """Identity-only diagnostics: never emit provider labels or content."""
    title = expected_title.strip()
    key = lesson_author_title_key(title)
    matches = [u for u in generated_units if isinstance(u, dict) and isinstance(u.get("title"), str)
               and title and (lesson_author_title_key(u["title"]) == key if key else u["title"].strip() == title)]
    reason = ("INVALID_EXPECTED_TITLE" if not title else
              "NO_UNIT_CANDIDATES" if not generated_units else
              "UNIT_TITLE_NOT_MATCHED" if not matches else
              "AMBIGUOUS_UNIT_TITLE" if len(matches) != 1 else "MATCHED")
    return {"reason": reason, "candidate_count": len(generated_units), "matching_count": len(matches)}


def match_staged_unit_by_title(generated_units: list[Any], expected_title: str) -> dict[str, Any] | None:
    """Match one approved title, never position or an ambiguous last-write win.

    Label-only titles (e.g. ``Unit 1``) have an empty normalized key. They
    still have an exact identity; do not spend another provider call when
    that exact non-empty title was returned. Content/evidence validation is
    unchanged and runs on the matched candidate immediately afterwards.
    """
    title = expected_title.strip()
    if not title:
        return None
    key = lesson_author_title_key(title)
    matches = [
        unit for unit in generated_units
        if isinstance(unit, dict) and isinstance(unit.get("title"), str)
        and (lesson_author_title_key(unit["title"]) == key if key else unit["title"].strip() == title)
    ]
    return matches[0] if len(matches) == 1 else None


def _blueprint_draft_architecture(request: RagLessonAuthorRequest) -> dict[str, Any] | None:
    architecture = request.blueprint_architecture
    if architecture is None:
        return None
    value = architecture.model_dump()
    lessons = value.get("lessons")
    if not isinstance(lessons, list) or not lessons:
        raise LessonAuthorProposalValidationError("Blueprint content architecture does not contain lessons.")
    for lesson in lessons:
        if not isinstance(lesson, dict) or not isinstance(lesson.get("units"), list) or not lesson["units"]:
            raise LessonAuthorProposalValidationError("Blueprint content architecture does not contain draftable units.")
        for unit in lesson["units"]:
            plan = unit.get("component_plan") if isinstance(unit, dict) else None
            if (
                not isinstance(plan, list)
                or (
                    not plan
                    and not (
                        value.get("architecture_contract_version") in {4, 5}
                        and isinstance(unit, dict)
                        and (
                            _is_v5_supporting_factless_unit(unit)
                            if value.get("architecture_contract_version") == 5
                            else _is_v4_supporting_factless_unit(unit)
                        )
                    )
                )
            ):
                raise LessonAuthorProposalValidationError("Blueprint unit does not contain a component plan.")
    return value


def _exact_blueprint_identifier_list(
    values: Any,
    *,
    label: str,
    max_items: int,
    max_length: int = 96,
) -> list[str]:
    """Preserve canonical Blueprint IDs exactly or reject the draft contract."""

    if not isinstance(values, list) or len(values) > max_items:
        raise LessonAuthorProposalValidationError(f"{label} is not a valid Blueprint identifier list.")
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise LessonAuthorProposalValidationError(f"{label} contains an invalid Blueprint identifier.")
        identifier = value.strip()
        if not identifier or len(identifier) > max_length:
            raise LessonAuthorProposalValidationError(f"{label} contains an invalid Blueprint identifier.")
        if identifier not in seen:
            seen.add(identifier)
            result.append(identifier)
    return result


def _exact_local_objective_refs(values: Any, *, label: str, objective_count: int) -> list[str]:
    refs = _exact_blueprint_identifier_list(values, label=label, max_items=12, max_length=16)
    for ref in refs:
        match = re.fullmatch(r"lo_([1-9][0-9]*)", ref)
        if match is None or int(match.group(1)) > objective_count:
            raise LessonAuthorProposalValidationError(f"{label} contains an unresolved local learning objective reference.")
    return refs


def _blueprint_instructional_density_contract(unit: dict[str, Any]) -> dict[str, Any]:
    """Preserve only the server-authored V3 output budget on staged units."""

    if unit.get("instructional_density_policy_version") != INSTRUCTIONAL_DENSITY_POLICY_VERSION:
        return {}
    values = {
        "source_content_chars": unit.get("source_content_chars"),
        "source_estimated_words": unit.get("source_estimated_words"),
        "max_generated_visible_chars": unit.get("max_generated_visible_chars"),
        "max_generated_words": unit.get("max_generated_words"),
    }
    if any(type(value) is not int or value < 0 for value in values.values()):
        raise LessonAuthorProposalValidationError("Instructional density output budget is invalid.")
    if values["max_generated_visible_chars"] < 1 or values["max_generated_words"] < 1:
        raise LessonAuthorProposalValidationError("Instructional density output budget is invalid.")
    return {
        "instructional_density_policy_version": INSTRUCTIONAL_DENSITY_POLICY_VERSION,
        **values,
    }


def _locked_component_plan(
    component_plan: list[dict[str, Any]],
    source_fact_ids: list[str],
    supporting_evidence_fact_ids: list[str] | None = None,
    *,
    strict_ownership: bool = False,
    instance_contract: bool = False,
) -> list[dict[str, Any]]:
    instance_contract = instance_contract or any(p.get("component_plan_id") for p in component_plan if isinstance(p, dict))
    if instance_contract:
        try:
            validate_instance_plan(component_plan)
        except ValueError as error:
            raise LessonAuthorProposalValidationError(str(error)) from error
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    supporting_fact_ids = list(dict.fromkeys(
        str(fact_id).strip()
        for fact_id in (supporting_evidence_fact_ids or [])
        if str(fact_id).strip()
    ))
    if not source_fact_ids and not supporting_fact_ids:
        raise LessonAuthorProposalValidationError("A Blueprint component plan requires canonical ownership or resolved read-only supporting evidence.")
    for plan_value in component_plan:
        component_type = normalize_staged_component_type(
            plan_value.get("type") if isinstance(plan_value, dict) else None,
        )
        if instance_contract and not component_type:
            raise LessonAuthorProposalValidationError("COMPONENT_PLAN_TYPE_INVALID")
        if not component_type or (not instance_contract and component_type in seen):
            continue
        seen.add(component_type)
        plan = plan_value if isinstance(plan_value, dict) else {}
        if instance_contract:
            owned = plan.get("source_fact_ids") or []
            supporting = plan.get("supporting_evidence_fact_ids") or []
            if not isinstance(owned, list) or not isinstance(supporting, list) or any(not isinstance(v, str) for v in [*owned, *supporting]):
                raise LessonAuthorProposalValidationError("COMPONENT_INSTANCE_EVIDENCE_INVALID")
            if set(owned) - set(source_fact_ids) or set(supporting) - set(supporting_fact_ids):
                raise LessonAuthorProposalValidationError("COMPONENT_INSTANCE_EVIDENCE_OUT_OF_SCOPE")
        assigned_fact_ids = [
            str(fact_id).strip()
            for fact_id in plan.get("source_fact_ids", [])
            if str(fact_id).strip() in source_fact_ids
        ]
        assigned_supporting_fact_ids = [
            str(fact_id).strip()
            for fact_id in plan.get("supporting_evidence_fact_ids", [])
            if str(fact_id).strip() in supporting_fact_ids
        ]
        if not source_fact_ids:
            if assigned_fact_ids:
                raise LessonAuthorProposalValidationError("A supporting-only Blueprint component must not claim canonical Source Fact ownership.")
            if not assigned_supporting_fact_ids:
                if strict_ownership:
                    raise LessonAuthorProposalValidationError("A V5 supporting Blueprint component is missing resolved read-only evidence.")
                assigned_supporting_fact_ids = supporting_fact_ids
        elif not assigned_fact_ids and not (instance_contract and assigned_supporting_fact_ids):
            if strict_ownership:
                raise LessonAuthorProposalValidationError("A V5 Blueprint component is missing its explicit canonical Source Fact ownership.")
            assigned_fact_ids = source_fact_ids if component_type == "html" else [source_fact_ids[0]]
        normalized.append({
            "type": component_type,
            **({"component_plan_id": plan["component_plan_id"], "learning_objective_refs": list(plan.get("learning_objective_refs") or [])} if instance_contract else {}),
            "title": str(plan.get("title") or "").strip()[:180],
            "rationale": str(plan.get("rationale") or "").strip()[:240],
            "purpose": str(plan.get("purpose") or "").strip()[:32],
            "reason_code": str(plan.get("reason_code") or "").strip()[:80],
            "learning_block_ids": _exact_blueprint_identifier_list(
                plan.get("learning_block_ids", []),
                label="component_plan.learning_block_ids",
                max_items=12,
            ),
            "source_fact_ids": list(dict.fromkeys(assigned_fact_ids)),
            "supporting_evidence_fact_ids": list(dict.fromkeys(assigned_supporting_fact_ids)),
            "content_requirements": [
                str(value).strip()[:500]
                for value in plan.get("content_requirements", [])
                if isinstance(value, str) and value.strip()
            ][:8],
            "required_artifacts": [
                artifact for artifact in plan.get("required_artifacts", [])
                if isinstance(artifact, dict)
            ][:6],
        })
    ordered_types = [plan["type"] for plan in normalized]
    if "html" in ordered_types and ordered_types[0] != "html":
        raise LessonAuthorProposalValidationError("Blueprint unit component plan must place html first when selected.")
    if "la_faq" in ordered_types and ordered_types[-1] != "la_faq":
        raise LessonAuthorProposalValidationError("Blueprint unit component plan must place FAQ last when selected.")
    if source_fact_ids and not set(ordered_types).intersection({"html", "la_diagram", "la_sortable", "la_crossword"}):
        raise LessonAuthorProposalValidationError("Blueprint unit component plan requires substantive instruction before checks or clarification.")
    return normalized[:4]


def _apply_blueprint_architecture_to_skeleton(
    skeleton: dict[str, Any],
    request: RagLessonAuthorRequest,
) -> dict[str, Any]:
    architecture = _blueprint_draft_architecture(request)
    if architecture is None:
        return skeleton
    chapters = skeleton.get("chapters") if isinstance(skeleton.get("chapters"), list) else []
    if len(chapters) != 1 or not isinstance(chapters[0], dict):
        raise LessonAuthorProposalValidationError("Staged skeleton must contain exactly one approved Blueprint chapter.")
    expected_lessons = architecture["lessons"]
    actual_lessons = chapters[0].get("lessons") if isinstance(chapters[0].get("lessons"), list) else []
    if len(actual_lessons) != len(expected_lessons):
        raise LessonAuthorProposalValidationError("Staged skeleton changed the approved Blueprint lesson count.")

    locked_lessons: list[dict[str, Any]] = []
    for lesson_index, (actual_value, expected_value) in enumerate(zip(actual_lessons, expected_lessons)):
        if not isinstance(actual_value, dict) or not isinstance(expected_value, dict):
            raise LessonAuthorProposalValidationError("Staged skeleton contains an invalid Blueprint lesson.")
        if lesson_author_title_key(actual_value.get("title")) != lesson_author_title_key(expected_value.get("title")):
            raise LessonAuthorProposalValidationError(f"Staged skeleton changed Blueprint lesson {lesson_index + 1}.")
        actual_units = actual_value.get("units") if isinstance(actual_value.get("units"), list) else []
        expected_units = expected_value.get("units") if isinstance(expected_value.get("units"), list) else []
        if len(actual_units) != len(expected_units):
            raise LessonAuthorProposalValidationError(
                f"Staged skeleton changed the approved unit count for Blueprint lesson {lesson_index + 1}.",
            )
        locked_units: list[dict[str, Any]] = []
        for unit_index, (actual_unit, expected_unit) in enumerate(zip(actual_units, expected_units)):
            if not isinstance(actual_unit, dict) or not isinstance(expected_unit, dict):
                raise LessonAuthorProposalValidationError("Staged skeleton contains an invalid Blueprint unit.")
            if lesson_author_title_key(actual_unit.get("title")) != lesson_author_title_key(expected_unit.get("title")):
                raise LessonAuthorProposalValidationError(
                    f"Staged skeleton changed Blueprint unit {lesson_index + 1}.{unit_index + 1}.",
                )
            actual_source_fact_ids = [
                str(fact_id).strip()
                for fact_id in actual_unit.get("source_fact_ids", [])
                if str(fact_id).strip()
            ]
            expected_source_fact_ids = [
                str(fact_id).strip()
                for fact_id in expected_unit.get("source_fact_ids", [])
                if str(fact_id).strip()
            ]
            supporting_evidence_fact_ids = [
                str(fact_id).strip()
                for fact_id in expected_unit.get("supporting_evidence_fact_ids", [])
                if str(fact_id).strip()
            ]
            source_fact_ids = expected_source_fact_ids or actual_source_fact_ids
            supporting_factless = (
                architecture.get("architecture_contract_version") in {4, 5}
                and (
                    _is_v5_supporting_factless_unit(expected_unit)
                    if architecture.get("architecture_contract_version") == 5
                    else _is_v4_supporting_factless_unit(expected_unit)
                )
            )
            if not source_fact_ids:
                if not supporting_factless or not supporting_evidence_fact_ids:
                    raise LessonAuthorProposalValidationError(
                        "Blueprint unit is missing its persisted source fact allocation or resolved supporting evidence.",
                    )
            locked_units.append({
                "title": str(expected_unit.get("title") or "").strip(),
                "purpose": str(expected_unit.get("purpose") or "").strip()[:500],
                "concept_ids": _exact_blueprint_identifier_list(expected_unit.get("concept_ids", []), label="unit.concept_ids", max_items=24),
                "learning_objective_refs": _exact_local_objective_refs(
                    expected_unit.get("learning_objective_refs", []),
                    label="unit.learning_objective_refs",
                    objective_count=len(expected_value.get("learning_objectives", [])),
                ),
                "learning_blocks": [dict(block) for block in expected_unit.get("learning_blocks", []) if isinstance(block, dict)][:12],
                "source_fact_ids": source_fact_ids,
                "supporting_evidence_fact_ids": supporting_evidence_fact_ids,
                "component_plan": _locked_component_plan(
                    expected_unit.get("component_plan", []),
                    source_fact_ids,
                    supporting_evidence_fact_ids,
                    strict_ownership=architecture.get("architecture_contract_version") == 5,
                    instance_contract=bool(architecture.get("component_capabilities")),
                ),
                **_blueprint_instructional_density_contract(expected_unit),
                "_blueprint_component_plan_locked": True,
                "_v5_evidence_contract": architecture.get("architecture_contract_version") == 5,
            })
        locked_lessons.append({
            "title": str(expected_value.get("title") or "").strip(),
            "learning_objectives": [str(value).strip()[:300] for value in expected_value.get("learning_objectives", []) if str(value).strip()][:12],
            "primary_concept_ids": _exact_blueprint_identifier_list(expected_value.get("primary_concept_ids", []), label="lesson.primary_concept_ids", max_items=24),
            "supporting_concept_ids": _exact_blueprint_identifier_list(expected_value.get("supporting_concept_ids", []), label="lesson.supporting_concept_ids", max_items=24),
            "assessment_required": expected_value.get("assessment_required") is True,
            "assessment_objective_refs": _exact_local_objective_refs(
                expected_value.get("assessment_objective_refs", []),
                label="lesson.assessment_objective_refs",
                objective_count=len(expected_value.get("learning_objectives", [])),
            ),
            "units": locked_units,
        })
    return {
        "chapters": [{
            "title": str(architecture.get("chapter_title") or "").strip(),
            "lessons": locked_lessons,
        }],
    }


def _build_blueprint_locked_staged_skeleton(
    request: RagLessonAuthorRequest,
    manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    """Fallback that preserves the approved Blueprint topology and source coverage."""
    architecture = _blueprint_draft_architecture(request)
    if architecture is None:
        raise LessonAuthorProposalValidationError("Blueprint content architecture is unavailable.")
    facts = [
        fact
        for fact in (manifest or {}).get("facts", [])
        if isinstance(fact, dict) and str(fact.get("fact_id") or "").strip()
    ]
    supporting_manifest_facts = [
        fact
        for fact in (manifest or {}).get("supporting_evidence_facts", [])
        if isinstance(fact, dict) and str(fact.get("fact_id") or "").strip()
    ]
    if not facts and not supporting_manifest_facts:
        raise LessonAuthorProposalValidationError("Không có source fact để phục hồi cấu trúc Blueprint.")

    # A persisted Blueprint has already assigned source facts to each unit.
    # Preserve that contract verbatim in the source-locked path: redistributing
    # facts by source_ref can otherwise leave a component plan owning only a
    # subset of the facts attached to its fallback unit.
    manifest_fact_ids = {
        str(fact.get("fact_id") or "").strip()
        for fact in facts
        if str(fact.get("fact_id") or "").strip()
    }
    supporting_manifest_fact_ids = {
        str(fact.get("fact_id") or "").strip()
        for fact in supporting_manifest_facts
        if str(fact.get("fact_id") or "").strip()
    }
    locked_lessons: list[dict[str, Any]] = []
    missing_blueprint_fact_ids: set[str] = set()
    has_complete_blueprint_allocation = True
    for lesson in architecture["lessons"]:
        locked_units: list[dict[str, Any]] = []
        for unit in lesson["units"]:
            fact_ids = list(dict.fromkeys(
                str(fact_id).strip()
                for fact_id in unit.get("source_fact_ids", [])
                if str(fact_id).strip()
            ))
            supporting_evidence_fact_ids = list(dict.fromkeys(
                str(fact_id).strip()
                for fact_id in unit.get("supporting_evidence_fact_ids", [])
                if str(fact_id).strip()
            ))
            supporting_factless = (
                architecture.get("architecture_contract_version") in {4, 5}
                and (
                    _is_v5_supporting_factless_unit(unit)
                    if architecture.get("architecture_contract_version") == 5
                    else _is_v4_supporting_factless_unit(unit)
                )
            )
            if not fact_ids and (not supporting_factless or not supporting_evidence_fact_ids):
                has_complete_blueprint_allocation = False
                break
            missing_blueprint_fact_ids.update(set(fact_ids) - manifest_fact_ids)
            missing_blueprint_fact_ids.update(set(supporting_evidence_fact_ids) - supporting_manifest_fact_ids)
            locked_units.append({
                "title": str(unit.get("title") or "").strip(),
                "purpose": str(unit.get("purpose") or "").strip()[:500],
                "concept_ids": _exact_blueprint_identifier_list(unit.get("concept_ids", []), label="unit.concept_ids", max_items=24),
                "learning_objective_refs": _exact_local_objective_refs(
                    unit.get("learning_objective_refs", []),
                    label="unit.learning_objective_refs",
                    objective_count=len(lesson.get("learning_objectives", [])),
                ),
                "learning_blocks": [dict(block) for block in unit.get("learning_blocks", []) if isinstance(block, dict)][:12],
                "source_fact_ids": fact_ids,
                "supporting_evidence_fact_ids": supporting_evidence_fact_ids,
                "component_plan": _locked_component_plan(
                    unit.get("component_plan", []),
                    fact_ids,
                    supporting_evidence_fact_ids,
                    strict_ownership=architecture.get("architecture_contract_version") == 5,
                    instance_contract=bool(architecture.get("component_capabilities")),
                ),
                **_blueprint_instructional_density_contract(unit),
                "_blueprint_component_plan_locked": True,
                "_v5_evidence_contract": architecture.get("architecture_contract_version") == 5,
            })
        if not has_complete_blueprint_allocation:
            break
        locked_lessons.append({
            "title": str(lesson.get("title") or "").strip(),
            "learning_objectives": [str(value).strip()[:300] for value in lesson.get("learning_objectives", []) if str(value).strip()][:12],
            "primary_concept_ids": _exact_blueprint_identifier_list(lesson.get("primary_concept_ids", []), label="lesson.primary_concept_ids", max_items=24),
            "supporting_concept_ids": _exact_blueprint_identifier_list(lesson.get("supporting_concept_ids", []), label="lesson.supporting_concept_ids", max_items=24),
            "assessment_required": lesson.get("assessment_required") is True,
            "assessment_objective_refs": _exact_local_objective_refs(
                lesson.get("assessment_objective_refs", []),
                label="lesson.assessment_objective_refs",
                objective_count=len(lesson.get("learning_objectives", [])),
            ),
            "units": locked_units,
        })

    if has_complete_blueprint_allocation:
        if missing_blueprint_fact_ids:
            raise LessonAuthorProposalValidationError(
                "Không thể phục hồi Blueprint vì thiếu source fact đã được duyệt: "
                f"{', '.join(sorted(missing_blueprint_fact_ids)[:12])}.",
            )
        return {
            "chapters": [{
                "title": str(architecture.get("chapter_title") or "").strip(),
                "lessons": locked_lessons,
            }],
        }

    # Compatibility fallback for Blueprints created before source fact
    # allocation was persisted. New Blueprint-driven drafts always use the
    # immutable allocation above.
    planned_units: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for lesson_index, lesson in enumerate(architecture["lessons"]):
        for unit in lesson["units"]:
            planned_units.append((lesson_index, lesson, unit))
    assignments: list[list[dict[str, Any]]] = [[] for _ in planned_units]
    for fact_index, fact in enumerate(facts):
        source_ref = str(fact.get("source_ref") or "").strip().casefold()
        eligible = [
            index
            for index, (_lesson_index, lesson, unit) in enumerate(planned_units)
            if source_ref and source_ref in {
                *(str(value).strip().casefold() for value in unit.get("source_refs", [])),
                *(str(value).strip().casefold() for value in lesson.get("source_refs", [])),
                *(str(value).strip().casefold() for value in architecture.get("source_refs", [])),
            }
        ]
        candidates = eligible or list(range(len(planned_units)))
        target_index = min(candidates, key=lambda index: (len(assignments[index]), index))
        assignments[target_index].append(fact)

    for unit_index, assigned in enumerate(assignments):
        if assigned:
            continue
        donor_index = max(range(len(assignments)), key=lambda index: len(assignments[index]))
        if assignments[donor_index]:
            assignments[unit_index].append(assignments[donor_index][-1])

    lessons: list[dict[str, Any]] = []
    cursor = 0
    for lesson in architecture["lessons"]:
        units: list[dict[str, Any]] = []
        for unit in lesson["units"]:
            fact_ids = [str(fact["fact_id"]).strip() for fact in assignments[cursor]]
            units.append({
                "title": str(unit.get("title") or "").strip(),
                "source_fact_ids": fact_ids,
                "component_plan": _locked_component_plan(unit.get("component_plan", []), fact_ids),
                **_blueprint_instructional_density_contract(unit),
                "_blueprint_component_plan_locked": True,
            })
            cursor += 1
        lessons.append({"title": str(lesson.get("title") or "").strip(), "units": units})
    return {"chapters": [{"title": architecture["chapter_title"], "lessons": lessons}]}


def extract_lesson_author_unit_batches(
    skeleton: dict[str, Any],
    source_coverage_manifest: dict[str, Any] | None = None,
    locale: str = "vi",
) -> list[list[dict[str, Any]]]:
    units: list[dict[str, Any]] = []
    chapters = skeleton.get("chapters") if isinstance(skeleton.get("chapters"), list) else []
    for chapter_index, chapter_value in enumerate(chapters[:1], start=1):
        if not isinstance(chapter_value, dict):
            continue
        chapter_title = str(chapter_value.get("title") or "").strip()
        lessons = chapter_value.get("lessons") if isinstance(chapter_value.get("lessons"), list) else []
        for lesson_index, lesson_value in enumerate(lessons, start=1):
            if not isinstance(lesson_value, dict):
                continue
            lesson_title = str(lesson_value.get("title") or "").strip()
            raw_units = lesson_value.get("units") if isinstance(lesson_value.get("units"), list) else []
            for unit_index, unit_value in enumerate(raw_units, start=1):
                if not isinstance(unit_value, dict):
                    continue
                if unit_value.get("_blueprint_component_plan_locked") is True:
                    source_fact_ids = [
                        str(fact_id).strip()
                        for fact_id in (unit_value.get("source_fact_ids") or [])
                        if str(fact_id).strip()
                    ]
                    supporting_evidence_fact_ids = [
                        str(fact_id).strip()
                        for fact_id in (unit_value.get("supporting_evidence_fact_ids") or [])
                        if str(fact_id).strip()
                    ]
                    component_plan = _locked_component_plan(
                        unit_value.get("component_plan", []),
                        source_fact_ids,
                        supporting_evidence_fact_ids,
                        strict_ownership=unit_value.get("_v5_evidence_contract") is True,
                    )
                else:
                    component_plan = build_staged_component_plan(
                        unit_value,
                        source_coverage_manifest,
                        locale,
                    )
                unit_value["component_plan"] = component_plan
                component_types = [item["type"] for item in component_plan]
                units.append(
                    {
                        "unit_path": f"chapter_{chapter_index}.lesson_{lesson_index}.unit_{unit_index}",
                        "chapter_title": chapter_title,
                        "lesson_title": lesson_title,
                        "learning_objectives": [
                            str(value).strip()[:300]
                            for value in lesson_value.get("learning_objectives", [])
                            if str(value).strip()
                        ][:12],
                        "assessment_required": lesson_value.get("assessment_required") is True,
                        "assessment_objective_refs": [
                            str(value).strip()[:80]
                            for value in lesson_value.get("assessment_objective_refs", [])
                            if str(value).strip()
                        ][:12],
                        "unit_title": str(unit_value.get("title") or "").strip(),
                        "unit_purpose": str(unit_value.get("purpose") or "").strip()[:500],
                        "concept_ids": [
                            str(value).strip()[:96]
                            for value in unit_value.get("concept_ids", [])
                            if str(value).strip()
                        ][:24],
                        "learning_objective_refs": [
                            str(value).strip()[:80]
                            for value in unit_value.get("learning_objective_refs", [])
                            if str(value).strip()
                        ][:12],
                        "learning_blocks": [dict(block) for block in unit_value.get("learning_blocks", []) if isinstance(block, dict)][:12],
                        "component_types": component_types[:4],
                        "component_plan": component_plan[:4],
                        "locale": locale,
                        "component_plan_locked": unit_value.get("_blueprint_component_plan_locked") is True,
                        "source_fact_ids": [
                            str(fact_id).strip()
                            for fact_id in (unit_value.get("source_fact_ids") or [])
                            if str(fact_id).strip()
                        ],
                        "supporting_evidence_fact_ids": [
                            str(fact_id).strip()
                            for fact_id in (unit_value.get("supporting_evidence_fact_ids") or [])
                            if str(fact_id).strip()
                        ],
                        "strict_v5_evidence": unit_value.get("_v5_evidence_contract") is True,
                        "instructional_output_budget": (
                            {
                                "policy_version": INSTRUCTIONAL_DENSITY_POLICY_VERSION,
                                "source_content_chars": unit_value["source_content_chars"],
                                "source_estimated_words": unit_value["source_estimated_words"],
                                "max_visible_chars": unit_value["max_generated_visible_chars"],
                                "max_words": unit_value["max_generated_words"],
                            }
                            if unit_value.get("instructional_density_policy_version")
                            == INSTRUCTIONAL_DENSITY_POLICY_VERSION
                            else None
                        ),
                    }
                )
    ensure_staged_chapter_component_diversity(
        units,
        source_coverage_manifest,
        locale,
    )
    return [
        units[index : index + STAGED_LESSON_AUTHOR_UNITS_PER_BATCH]
        for index in range(0, len(units), STAGED_LESSON_AUTHOR_UNITS_PER_BATCH)
    ]


def validate_staged_skeleton_source_facts(
    batches: list[list[dict[str, Any]]],
    manifest: dict[str, Any] | None,
) -> None:
    if not manifest:
        return
    required = {
        str(fact.get("fact_id") or "").strip()
        for fact in manifest.get("facts", [])
        if isinstance(fact, dict) and str(fact.get("fact_id") or "").strip()
    }
    assigned = {
        fact_id
        for batch in batches
        for unit in batch
        for fact_id in unit.get("source_fact_ids", [])
        if isinstance(fact_id, str) and fact_id.strip()
    }
    invalid = sorted(assigned - required)
    missing = sorted(required - assigned)
    if invalid or missing:
        raise LessonAuthorProposalValidationError(
            "Staged skeleton không phân bổ đúng source fact. "
            f"Thiếu: {', '.join(missing[:12]) or 'none'}; "
            f"không hợp lệ: {', '.join(invalid[:12]) or 'none'}."
        )
