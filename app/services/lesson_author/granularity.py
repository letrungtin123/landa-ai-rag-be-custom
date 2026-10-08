"""Blueprint source granularity: chapter fact partitions, boundaries and fact-id allocation."""

from __future__ import annotations

import re
from typing import Any

from app.lesson_author_blueprint import LessonAuthorBlueprintValidationError, ensure_lesson_author_blueprint_faqs
from app.services.lesson_author.proposal_validation import _normalize_structural_title
from app.services.lesson_author.staged.plan import _source_locked_ordered_items
from app.services.retrieval.query import parse_source_range
from app.services.retrieval.source_coverage import _source_page_number


def _blueprint_scope_refs(chapter: dict[str, Any]) -> set[str]:
    refs = {
        str(value).strip().casefold()
        for value in chapter.get("source_refs", []) or []
        if str(value).strip()
    }
    for lesson in chapter.get("lessons", []) or []:
        if not isinstance(lesson, dict):
            continue
        refs.update(
            str(value).strip().casefold()
            for value in lesson.get("source_refs", []) or []
            if str(value).strip()
        )
        for unit in lesson.get("units", []) or []:
            if not isinstance(unit, dict):
                continue
            refs.update(
                str(value).strip().casefold()
                for value in unit.get("source_refs", []) or []
                if str(value).strip()
            )
    return refs


def _blueprint_chapter_facts(
    chapter: dict[str, Any],
    manifest: dict[str, Any] | None,
    structure_nodes: list[dict[str, Any]] | None,
    *,
    allow_all: bool = False,
) -> list[dict[str, Any]]:
    """Resolve a Blueprint chapter to the ordered fact checklist it owns."""
    facts = [
        fact for fact in (manifest or {}).get("facts", [])
        if isinstance(fact, dict) and str(fact.get("fact_id") or "").strip()
    ]
    refs = _blueprint_scope_refs(chapter)
    if not refs:
        return facts if allow_all else []
    matching_ranges: list[tuple[str, int, int]] = []
    for node in structure_nodes or []:
        if not isinstance(node, dict):
            continue
        source_ref = str(node.get("source_ref") or "").strip().casefold()
        parent_ref = str(node.get("parent_source_ref") or "").strip().casefold()
        if source_ref not in refs and parent_ref not in refs:
            continue
        page_range = parse_source_range(str(node.get("title") or ""))
        document_id = str(node.get("document_id") or "")
        if page_range:
            matching_ranges.append((document_id, page_range[0], page_range[1]))

    matched: list[dict[str, Any]] = []
    for fact in facts:
        source_ref = str(fact.get("source_ref") or "").strip().casefold()
        if source_ref and source_ref in refs:
            matched.append(fact)
            continue
        page = _source_page_number(fact.get("source_page"))
        document_id = str(fact.get("document_id") or "")
        if page is not None and any(
            (not range_document_id or range_document_id == document_id)
            and start_page <= page <= end_page
            for range_document_id, start_page, end_page in matching_ranges
        ):
            matched.append(fact)
    return matched


def _blueprint_chapter_scope_titles(chapter: dict[str, Any]) -> list[str]:
    """Return the semantic labels that define a Blueprint chapter's scope."""
    values = [chapter.get("title")]
    for lesson in chapter.get("lessons", []) or []:
        if not isinstance(lesson, dict):
            continue
        values.append(lesson.get("title"))
        for unit in lesson.get("units", []) or []:
            if isinstance(unit, dict):
                values.append(unit.get("title"))
    titles: list[str] = []
    seen: set[str] = set()
    for value in values:
        title = re.sub(r"[^\w\s]", " ", str(value or "").casefold())
        title = re.sub(r"\s+", " ", title).strip()
        if len(title) >= 10 and title not in seen:
            seen.add(title)
            titles.append(title)
    return titles


def _trailing_source_page_group(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep source pages atomic when moving a clearly detected boundary."""
    if not facts:
        return []
    tail = facts[-1]
    tail_key = (
        str(tail.get("document_id") or ""),
        _source_page_number(tail.get("source_page")),
    )
    start = len(facts) - 1
    while start > 0:
        previous = facts[start - 1]
        previous_key = (
            str(previous.get("document_id") or ""),
            _source_page_number(previous.get("source_page")),
        )
        if previous_key != tail_key:
            break
        start -= 1
    return facts[start:]


def _page_group_belongs_to_next_blueprint_chapter(
    facts: list[dict[str, Any]],
    next_chapter: dict[str, Any],
) -> bool:
    """Detect a heading on a boundary page that names the next chapter.

    TOC-derived source ranges occasionally include the opening slide of the
    following chapter. Move only a whole trailing page when its own source
    text explicitly names a next-chapter lesson or unit; never infer this
    from loose keyword overlap.
    """
    next_titles = _blueprint_chapter_scope_titles(next_chapter)
    if not next_titles:
        return False
    for fact in facts:
        raw_text = str(fact.get("text") or "")
        for line in re.split(r"[\r\n]+|(?<=[.!?])\s+", raw_text):
            candidate = re.sub(r"^\s*(?:[\u2022*-]|\d+[.)])\s*", "", line.casefold())
            candidate = re.sub(r"[^\w\s]", " ", candidate)
            candidate = re.sub(r"\s+", " ", candidate).strip()
            if len(candidate) < 10:
                continue
            if any(candidate in title or title in candidate for title in next_titles):
                return True
    return False


def _reconcile_blueprint_chapter_boundaries(
    chapter_allocations: list[tuple[int, list[dict[str, Any]], list[dict[str, Any]]]],
    chapters: list[dict[str, Any]],
) -> list[tuple[int, list[dict[str, Any]], list[dict[str, Any]]]]:
    """Correct only explicit TOC range overlaps before persisting fact IDs."""
    reconciled = list(chapter_allocations)
    for index in range(len(reconciled) - 1):
        chapter_index, units, current_facts = reconciled[index]
        next_chapter_index, next_units, next_facts = reconciled[index + 1]
        boundary_page = _trailing_source_page_group(current_facts)
        next_chapter = chapters[next_chapter_index] if next_chapter_index < len(chapters) else None
        if (
            not boundary_page
            or not isinstance(next_chapter, dict)
            or not _page_group_belongs_to_next_blueprint_chapter(boundary_page, next_chapter)
        ):
            continue
        boundary_ids = {str(fact.get("fact_id") or "").strip() for fact in boundary_page}
        if not boundary_ids or any(
            str(fact.get("fact_id") or "").strip() in boundary_ids
            for fact in next_facts
        ):
            continue
        reconciled[index] = (chapter_index, units, current_facts[:-len(boundary_page)])
        reconciled[index + 1] = (next_chapter_index, next_units, boundary_page + next_facts)
    return reconciled


def _partition_blueprint_facts(
    facts: list[dict[str, Any]],
    count: int,
) -> list[list[dict[str, Any]]]:
    """Partition ordered source facts without splitting a page when possible."""
    if not facts or count <= 0:
        return []
    count = min(count, len(facts))
    page_groups: list[list[dict[str, Any]]] = []
    current_key: tuple[str, int | None, str] | None = None
    for fact in facts:
        key = (
            str(fact.get("document_id") or ""),
            _source_page_number(fact.get("source_page")),
            str(fact.get("source_ref") or ""),
        )
        if not page_groups or key != current_key:
            page_groups.append([])
            current_key = key
        page_groups[-1].append(fact)
    if len(page_groups) >= count:
        base, remainder = divmod(len(page_groups), count)
        groups: list[list[dict[str, Any]]] = []
        cursor = 0
        for index in range(count):
            size = base + (1 if index < remainder else 0)
            groups.append([fact for group in page_groups[cursor:cursor + size] for fact in group])
            cursor += size
        return groups
    base, remainder = divmod(len(facts), count)
    groups = []
    cursor = 0
    for index in range(count):
        size = base + (1 if index < remainder else 0)
        groups.append(facts[cursor:cursor + size])
        cursor += size
    return [group for group in groups if group]


def _blueprint_fact_group_title(facts: list[dict[str, Any]], index: int, locale: str) -> str:
    fallback = f"Nội dung trọng tâm {index}" if locale != "en" else f"Core content {index}"
    text = re.sub(r"\s+", " ", str(facts[0].get("text") or "")).strip() if facts else ""
    if not text:
        return fallback
    candidate = text.split(":", 1)[0].strip()
    if len(candidate) < 8 or len(candidate) > 110:
        candidate = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0].strip()
    return _normalize_structural_title(candidate[:160], fallback)


def ensure_blueprint_source_granularity(
    blueprint: dict[str, Any],
    manifest: dict[str, Any] | None,
    structure_nodes: list[dict[str, Any]] | None,
    locale: str,
) -> dict[str, Any]:
    """Prevent a substantial source chapter from being persisted as one unit.

    This is a deterministic guard for the failure mode where a single TOC
    heading contains several pages of definitions, outcomes, and a model.
    Normal Blueprint output remains untouched; only a one-unit chapter with
    multiple source page groups is expanded into source-named lessons.
    """
    chapters = blueprint.get("chapters") if isinstance(blueprint.get("chapters"), list) else []
    for chapter_index, chapter_value in enumerate(chapters):
        if not isinstance(chapter_value, dict):
            continue
        lessons = chapter_value.get("lessons") if isinstance(chapter_value.get("lessons"), list) else []
        units = [
            unit
            for lesson in lessons if isinstance(lesson, dict)
            for unit in (lesson.get("units") if isinstance(lesson.get("units"), list) else [])
            if isinstance(unit, dict)
        ]
        facts = _blueprint_chapter_facts(
            chapter_value,
            manifest,
            structure_nodes,
            allow_all=len(chapters) == 1,
        )
        distinct_pages = {
            (str(fact.get("document_id") or ""), _source_page_number(fact.get("source_page")))
            for fact in facts
        }
        desired_units = min(3, len(distinct_pages), len(facts))
        if len(units) != 1 or desired_units < 2:
            continue
        seed_lesson = lessons[0] if lessons and isinstance(lessons[0], dict) else {}
        seed_unit = units[0]
        partitions = _partition_blueprint_facts(facts, desired_units)
        expanded_units: list[dict[str, Any]] = []
        for group_index, group in enumerate(partitions, start=1):
            title = _blueprint_fact_group_title(group, group_index, locale)
            component_plan = [{
                "type": "html",
                "title": title,
                "rationale": (
                    "Explains this complete source-backed concept before practice."
                    if locale == "en" else "Giải thích trọn vẹn cụm kiến thức nguồn trước khi thực hành."
                ),
            }]
            group_text = " ".join(str(fact.get("text") or "") for fact in group)
            supports_process_diagram = bool(re.search(
                r"\b(?:bước|buoc|step|steps|giai đoạn|giai doan|phase|phases|"
                r"quy trình|quy trinh|process|workflow|trình tự|trinh tu|sequence|"
                r"mô hình triển khai|mo hinh trien khai|implementation model|"
                r"framework|chu trình|chu trinh|cycle)\b",
                f"{chapter_value.get('title') or ''} {title} {group_text}".casefold(),
                flags=re.IGNORECASE,
            ))
            if len(_source_locked_ordered_items(group)) >= 3 and supports_process_diagram:
                component_plan.append({
                    "type": "la_diagram",
                    "title": title,
                    "rationale": (
                        "Visualizes the explicit source sequence or model."
                        if locale == "en" else "Trực quan hóa chuỗi bước hoặc mô hình được nêu rõ trong nguồn."
                    ),
                })
            expanded_units.append({
                "title": title,
                "source_refs": list(seed_unit.get("source_refs") or seed_lesson.get("source_refs") or chapter_value.get("source_refs") or []),
                "component_plan": component_plan,
            })
        chapter_value["lessons"] = [{
            "title": _normalize_structural_title(
                str(seed_lesson.get("title") or ""),
                "Nội dung trọng tâm" if locale != "en" else "Core content",
            ),
            "objective": str(seed_lesson.get("objective") or (
                "Người học có thể giải thích và vận dụng các nội dung nguồn của chương."
                if locale != "en" else "Learners can explain and apply the chapter's source-backed content."
            )),
            "learning_activities": list(seed_lesson.get("learning_activities") or (
                ["Đọc hiểu cụm kiến thức nguồn và thực hành truy hồi."]
                if locale != "en" else ["Review the source-backed concepts and practice retrieval."]
            )),
            "assessment": str(seed_lesson.get("assessment") or (
                "Kiểm tra khả năng vận dụng chính xác các fact nguồn."
                if locale != "en" else "Check accurate application of the source facts."
            )),
            "source_refs": list(seed_lesson.get("source_refs") or chapter_value.get("source_refs") or []),
            "units": expanded_units,
        }]
    return ensure_lesson_author_blueprint_faqs(blueprint, locale)


def apply_phase_one_blueprint_component_contract(
    blueprint: dict[str, Any],
) -> dict[str, Any]:
    """Attach deterministic component ownership after the server allocates unit facts."""
    architecture_contract_version = blueprint.get("architecture_contract_version")
    server_allocation = blueprint.get("source_fact_allocation")
    has_complete_server_allocation = (
        architecture_contract_version in {4, 5}
        and isinstance(server_allocation, dict)
        and server_allocation.get("version") in {"source-fact-allocation-v2", "source-fact-allocation-v3"}
        and server_allocation.get("authority") == "server"
        and server_allocation.get("complete") is True
    )
    purpose_by_type = {
        "html": "explain",
        "problem": "assess",
        "la_faq": "clarify",
        "la_sortable": "sequence",
        "la_crossword": "terminology",
        "la_diagram": "relationship",
    }
    default_requirement = {
        "html": "Explain every assigned source fact accurately and preserve required source structure.",
        "problem": "Assess understanding of the assigned source facts without adding unsupported facts.",
        "la_faq": "Clarify source-grounded questions using the assigned source facts.",
        "la_sortable": "Preserve the source-supported order of the assigned procedure.",
        "la_crossword": "Practice only source-supported terminology represented by the assigned facts.",
        "la_diagram": "Show the source-supported relationship or flow represented by the assigned facts.",
    }
    valid_purposes = set(purpose_by_type.values())
    valid_artifacts = {"ordered_list", "checklist", "table", "warning", "requirement", "exception", "comparison"}
    for chapter in blueprint.get("chapters", []):
        if not isinstance(chapter, dict):
            continue
        for lesson in chapter.get("lessons", []):
            if not isinstance(lesson, dict):
                continue
            for unit in lesson.get("units", []):
                if not isinstance(unit, dict):
                    continue
                fact_ids = list(dict.fromkeys(
                    str(fact_id).strip()
                    for fact_id in unit.get("source_fact_ids", [])
                    if str(fact_id).strip()
                ))
                if not fact_ids:
                # A v4/v5 architecture can contain a supporting or
                    # reinforcement unit that is not the canonical owner of
                    # any fact.  Once the server has proven complete global
                    # allocation, that unit must reach the Blueprint
                    # Validator instead of being misreported as an allocator
                    # failure.  The validator remains responsible for
                    # deciding whether the empty ownership is acceptable.
                    if has_complete_server_allocation:
                        learning_blocks = unit.get("learning_blocks")
                        if not isinstance(learning_blocks, list) or not learning_blocks:
                            raise LessonAuthorBlueprintValidationError(
                                "BLUEPRINT_INVALID_SCHEMA",
                                "A Source Map Blueprint unit requires semantic learning blocks.",
                            )
                        unit["component_plan"] = []
                        continue
                    raise LessonAuthorBlueprintValidationError(
                        "BLUEPRINT_SOURCE_COVERAGE_INCOMPLETE",
                        "A Phase-1 Blueprint unit cannot have an empty source fact allocation.",
                    )
                # Phase 3 has already selected semantic learning blocks with
                # fact ownership.  Preserve that independent representation;
                # Node's Phase-2 planner is the component authority.
                if architecture_contract_version in {3, 4, 5}:
                    learning_blocks = unit.get("learning_blocks")
                    if not isinstance(learning_blocks, list) or not learning_blocks:
                        raise LessonAuthorBlueprintValidationError(
                            "BLUEPRINT_INVALID_SCHEMA",
                            "A Source Map Blueprint unit requires semantic learning blocks.",
                        )
                    unit["component_plan"] = []
                    continue
                plan = unit.get("component_plan") if isinstance(unit.get("component_plan"), list) else []
                if not plan:
                    plan = [{
                        "type": "html",
                        "title": str(unit.get("title") or "Nội dung học tập").strip(),
                        "rationale": "Explain the source-backed learning content.",
                    }]
                non_html_cursor = 0
                normalized_plan: list[dict[str, Any]] = []
                for plan_value in plan:
                    if not isinstance(plan_value, dict):
                        continue
                    component_type = str(plan_value.get("type") or "").strip().casefold()
                    if component_type not in purpose_by_type:
                        raise LessonAuthorBlueprintValidationError(
                            "BLUEPRINT_INVALID_SCHEMA",
                            f"Unsupported Phase-1 component type: {component_type}.",
                        )
                    # The explanatory block is the source-complete owner. Each
                    # supporting component receives one deterministic fact so
                    # it cannot become a disconnected decorative activity.
                    owned_fact_ids = fact_ids if component_type == "html" else [fact_ids[non_html_cursor % len(fact_ids)]]
                    if component_type != "html":
                        non_html_cursor += 1
                    purpose = str(plan_value.get("purpose") or "").strip().casefold()
                    if purpose not in valid_purposes:
                        purpose = purpose_by_type[component_type]
                    requirements = [
                        str(value).strip()[:500]
                        for value in plan_value.get("content_requirements", [])
                        if isinstance(value, str) and value.strip()
                    ][:8]
                    artifacts: list[dict[str, Any]] = []
                    for artifact_value in plan_value.get("required_artifacts", []):
                        if not isinstance(artifact_value, dict):
                            continue
                        artifact_type = str(artifact_value.get("type") or "").strip().casefold()
                        if artifact_type not in valid_artifacts:
                            continue
                        minimum = artifact_value.get("minimum_items")
                        artifacts.append({
                            "type": artifact_type,
                            **({"minimum_items": min(minimum, 100)} if isinstance(minimum, int) and minimum > 0 else {}),
                        })
                    normalized_plan.append({
                        "type": component_type,
                        "title": str(plan_value.get("title") or "").strip()[:180],
                        "rationale": str(plan_value.get("rationale") or "").strip()[:240],
                        "purpose": purpose,
                        "source_fact_ids": owned_fact_ids,
                        "content_requirements": requirements or [default_requirement[component_type]],
                        **({"required_artifacts": artifacts[:6]} if artifacts else {}),
                    })
                owned = {
                    fact_id
                    for plan_value in normalized_plan
                    for fact_id in plan_value.get("source_fact_ids", [])
                }
                missing = [fact_id for fact_id in fact_ids if fact_id not in owned]
                if missing:
                    raise LessonAuthorBlueprintValidationError(
                        "BLUEPRINT_SOURCE_COVERAGE_INCOMPLETE",
                        f"A Phase-1 Blueprint unit has unowned source facts: {', '.join(missing[:8])}.",
                    )
                unit["component_plan"] = normalized_plan
    blueprint["content_contract_version"] = 1
    return blueprint


def allocate_blueprint_source_fact_ids(
    blueprint: dict[str, Any],
    manifest: dict[str, Any] | None,
    structure_nodes: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Persist the exact fact allocation that detailed drafting must satisfy."""
    all_facts = [
        fact for fact in (manifest or {}).get("facts", [])
        if isinstance(fact, dict) and str(fact.get("fact_id") or "").strip()
    ]
    if not all_facts:
        raise LessonAuthorBlueprintValidationError(
            "BLUEPRINT_SOURCE_COVERAGE_EMPTY",
            "Blueprint cannot be approved without source facts for detailed authoring.",
        )
    if blueprint.get("architecture_contract_version") in {3, 4, 5}:
        required_ids = {str(fact["fact_id"]).strip() for fact in all_facts}
        assigned_ids: list[str] = [
            str(fact_id).strip()
            for chapter in blueprint.get("chapters", []) if isinstance(chapter, dict)
            for lesson in chapter.get("lessons", []) if isinstance(lesson, dict)
            for unit in lesson.get("units", []) if isinstance(unit, dict)
            for fact_id in unit.get("source_fact_ids", []) if str(fact_id).strip()
        ]
        assigned_set = set(assigned_ids)
        if assigned_set != required_ids or len(assigned_ids) != len(assigned_set):
            raise LessonAuthorBlueprintValidationError(
                "BLUEPRINT_SOURCE_COVERAGE_INCOMPLETE",
                "Course Architect must allocate every Source Map fact exactly once across its units.",
            )
        return apply_phase_one_blueprint_component_contract(blueprint)
    chapters = blueprint.get("chapters") if isinstance(blueprint.get("chapters"), list) else []
    chapter_allocations: list[tuple[int, list[dict[str, Any]], list[dict[str, Any]]]] = []
    for chapter_index, chapter_value in enumerate(chapters):
        if not isinstance(chapter_value, dict):
            continue
        units = [
            unit
            for lesson in chapter_value.get("lessons", []) if isinstance(lesson, dict)
            for unit in (lesson.get("units") if isinstance(lesson.get("units"), list) else [])
            if isinstance(unit, dict)
        ]
        facts = _blueprint_chapter_facts(
            chapter_value,
            manifest,
            structure_nodes,
            allow_all=len(chapters) == 1,
        )
        if not units:
            raise LessonAuthorBlueprintValidationError(
                "BLUEPRINT_SOURCE_COVERAGE_INCOMPLETE",
                f"Blueprint chapter {chapter_index + 1} has no draftable units.",
            )
        chapter_allocations.append((chapter_index, units, facts))

    chapter_allocations = _reconcile_blueprint_chapter_boundaries(
        chapter_allocations,
        [chapter if isinstance(chapter, dict) else {} for chapter in chapters],
    )

    required_ids = {str(fact["fact_id"]).strip() for fact in all_facts}
    resolved_fact_ids = [
        str(fact["fact_id"]).strip()
        for _chapter_index, _units, facts in chapter_allocations
        for fact in facts
    ]
    resolved_ids = set(resolved_fact_ids)
    # A raw DOCX/PDF without a usable TOC cannot reliably map generated chapter
    # titles to source refs. Do not fail a valid request or silently omit the
    # unmatched tail: allocate the full ordered manifest across the approved
    # units. Explicit source ranges still keep their chapter-local allocation.
    needs_unscoped_allocation = (
        any(not facts for _chapter_index, _units, facts in chapter_allocations)
        or resolved_ids != required_ids
        or len(resolved_fact_ids) != len(resolved_ids)
    )
    if needs_unscoped_allocation:
        all_units = [
            unit
            for _chapter_index, units, _facts in chapter_allocations
            for unit in units
        ]
        if len(all_facts) < len(all_units):
            raise LessonAuthorBlueprintValidationError(
                "BLUEPRINT_SOURCE_COVERAGE_INCOMPLETE",
                "Blueprint contains more draftable units than the selected source has distinct facts.",
            )
        for unit, group in zip(all_units, _partition_blueprint_facts(all_facts, len(all_units))):
            fact_ids = [str(fact["fact_id"]).strip() for fact in group]
            unit["source_fact_ids"] = fact_ids
        return apply_phase_one_blueprint_component_contract(blueprint)

    assigned_ids: set[str] = set()
    for chapter_index, units, facts in chapter_allocations:
        if len(facts) < len(units):
            raise LessonAuthorBlueprintValidationError(
                "BLUEPRINT_SOURCE_COVERAGE_INCOMPLETE",
                f"Blueprint chapter {chapter_index + 1} contains more units than its resolved source facts.",
            )
        for unit, group in zip(units, _partition_blueprint_facts(facts, len(units))):
            fact_ids = [str(fact["fact_id"]).strip() for fact in group]
            unit["source_fact_ids"] = fact_ids
            assigned_ids.update(fact_ids)
    missing = required_ids - assigned_ids
    if missing:
        raise LessonAuthorBlueprintValidationError(
            "BLUEPRINT_SOURCE_COVERAGE_INCOMPLETE",
            f"Blueprint did not allocate all source facts: {', '.join(sorted(missing)[:12])}.",
        )
    return apply_phase_one_blueprint_component_contract(blueprint)
