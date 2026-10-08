"""Course-architecture evidence and semantic scope validation and source-map fact allocation."""

from __future__ import annotations

from typing import Any

from app.lesson_author_blueprint import MAX_SERVER_OWNED_SOURCE_FACT_IDS_PER_SCOPE, LessonAuthorBlueprintValidationError
from app.services.lesson_author.architecture_shape import _workflow_issue
from app.workflows.contracts import (
    WorkflowIssue,
    WorkflowValidationResult,
    safe_workflow_issue_summary,
    safe_workflow_path,
)


def _source_fact_allocation_values(value: Any, key: str) -> set[str]:
    values: set[str] = set()
    if isinstance(value, dict):
        for raw_value in value.get(key, []) if isinstance(value.get(key), list) else []:
            if isinstance(raw_value, str) and raw_value.strip():
                values.add(raw_value.strip())
        content = value.get("content")
        if isinstance(content, dict):
            for raw_value in content.get(key, []) if isinstance(content.get(key), list) else []:
                if isinstance(raw_value, str) and raw_value.strip():
                    values.add(raw_value.strip())
    return values


def validate_course_architecture_evidence_scope(
    blueprint: dict[str, Any],
    source_map: dict[str, Any],
) -> WorkflowValidationResult:
    """Validate v5 primary/supporting evidence-scope semantics before allocation.

    Evidence *scope* is the authoritative ownership boundary in v5.  A
    concept can be taught or reinforced by several lessons, while each
    canonical evidence scope has exactly one primary semantic-block owner.
    Supporting references are grounded references only and never expand into
    canonical Fact ownership.
    """
    if blueprint.get("architecture_contract_version") != 5:
        return WorkflowValidationResult()

    sections = {
        str(section.get("id") or "").strip(): section
        for section in source_map.get("sections", [])
        if isinstance(section, dict) and str(section.get("id") or "").strip()
    }
    concepts = {
        str(concept.get("id") or "").strip(): concept
        for concept in source_map.get("concepts", [])
        if isinstance(concept, dict) and str(concept.get("id") or "").strip()
    }
    scopes = {
        str(scope.get("id") or "").strip(): scope
        for scope in source_map.get("source_evidence_scopes", [])
        if isinstance(scope, dict) and str(scope.get("id") or "").strip()
    }
    issues: list[WorkflowIssue] = []

    def add(
        code: str,
        message: str,
        *,
        path: str,
        learning_block_id: str | None = None,
        scope_id: str | None = None,
        eligible_repair_paths: list[str] | None = None,
        repairable: bool | None = None,
    ) -> None:
        issue = _workflow_issue(code, message, path=path)
        if learning_block_id:
            issue["learning_block_ids"] = [learning_block_id]
        if scope_id:
            issue["evidence_scope_id"] = scope_id
        if eligible_repair_paths:
            issue["eligible_repair_paths"] = sorted(set(eligible_repair_paths))[:12]
        if repairable is not None:
            issue["repairable"] = repairable
        issues.append(issue)

    def ref_matches_scope(scope: dict[str, Any], refs: set[str]) -> bool:
        if not refs:
            return True
        section_id = str(scope.get("section_id") or "").strip()
        visited: set[str] = set()
        while section_id and section_id not in visited:
            visited.add(section_id)
            section = sections.get(section_id)
            if not isinstance(section, dict):
                return False
            if str(section.get("source_ref") or "").strip() in refs:
                return True
            section_id = str(section.get("parent_id") or "").strip()
        return False

    primary_owners: dict[str, list[tuple[str, str, str]]] = {}
    referenced_scopes: set[str] = set()
    # This is a server-only, structural index used solely to select a safe
    # local repair destination for a missing V5 primary owner. It never maps
    # or reveals canonical facts and deliberately rejects ambiguous choices.
    block_candidates: list[dict[str, Any]] = []
    for chapter_index, chapter in enumerate(blueprint.get("chapters", []), start=1):
        if not isinstance(chapter, dict):
            continue
        chapter_refs = _source_fact_allocation_values(chapter, "source_refs")
        chapter_concepts = _source_fact_allocation_values(chapter, "concept_ids")
        for lesson_index, lesson in enumerate(chapter.get("lessons", []), start=1):
            if not isinstance(lesson, dict):
                continue
            lesson_path = f"chapter_{chapter_index}.lesson_{lesson_index}"
            lesson_refs = _source_fact_allocation_values(lesson, "source_refs") or chapter_refs
            lesson_concepts = (
                _source_fact_allocation_values(lesson, "primary_concept_ids")
                | _source_fact_allocation_values(lesson, "supporting_concept_ids")
                | chapter_concepts
            )
            for unit_index, unit in enumerate(lesson.get("units", []), start=1):
                if not isinstance(unit, dict):
                    continue
                unit_path = f"{lesson_path}.unit_{unit_index}"
                unit_refs = _source_fact_allocation_values(unit, "source_refs") or lesson_refs
                unit_concepts = _source_fact_allocation_values(unit, "concept_ids") or lesson_concepts
                blocks = unit.get("learning_blocks") if isinstance(unit.get("learning_blocks"), list) else []
                if not blocks:
                    add("MISSING_INSTRUCTIONAL_SCOPE", "A v5 unit must contain semantic learning blocks.", path=unit_path)
                    continue
                for block_index, block in enumerate(blocks, start=1):
                    if not isinstance(block, dict):
                        add("INVALID_EVIDENCE_SCOPE_REFERENCE", "A v5 semantic learning block must be an object.", path=f"{unit_path}.block_{block_index}")
                        continue
                    block_path = f"{unit_path}.block_{block_index}"
                    block_id = str(block.get("id") or "").strip()
                    block_concepts = _source_fact_allocation_values(block, "concept_ids")
                    block_refs = _source_fact_allocation_values(block, "source_refs") or unit_refs
                    block_candidates.append({
                        "unit_path": unit_path,
                        "block_path": block_path,
                        "block_id": block_id,
                        "concept_ids": block_concepts,
                        "source_refs": block_refs,
                    })
                    primary = _source_fact_allocation_values(block, "primary_evidence_scope_ids")
                    supporting = _source_fact_allocation_values(block, "supporting_evidence_scope_ids")
                    if not primary and not supporting:
                        add("MISSING_EVIDENCE_SCOPE_REFERENCE", "A v5 semantic learning block must reference a primary or supporting evidence scope.", path=block_path, learning_block_id=block_id or None)
                    if primary & supporting:
                        add("EVIDENCE_SCOPE_OWNERSHIP_OVERLAP", "One semantic learning block cannot primary-own and supporting-reference the same evidence scope.", path=block_path, learning_block_id=block_id or None)
                    scope_documents: set[str] = set()
                    for scope_id, ownership in [*( (scope_id, "primary") for scope_id in primary ), *( (scope_id, "supporting") for scope_id in supporting )]:
                        scope = scopes.get(scope_id)
                        referenced_scopes.add(scope_id)
                        if not isinstance(scope, dict):
                            add("UNKNOWN_EVIDENCE_SCOPE", "A semantic learning block selected an evidence scope outside the Source Map.", path=block_path, learning_block_id=block_id or None, scope_id=scope_id)
                            continue
                        scope_documents.add(str(scope.get("document_id") or ""))
                        scope_concepts = {
                            str(value).strip() for value in scope.get("concept_ids", [])
                            if isinstance(value, str) and value.strip()
                        }
                        if scope_concepts and not scope_concepts.issubset(block_concepts):
                            add("EVIDENCE_SCOPE_CONCEPT_MISMATCH", "Evidence scope concepts must remain inside the semantic block concept scope.", path=block_path, learning_block_id=block_id or None, scope_id=scope_id)
                        if scope_concepts and not scope_concepts.issubset(unit_concepts):
                            add("EVIDENCE_SCOPE_CONCEPT_MISMATCH", "Evidence scope concepts must remain inside the unit concept scope.", path=unit_path, learning_block_id=block_id or None, scope_id=scope_id)
                        if block_refs and not ref_matches_scope(scope, block_refs):
                            add("EVIDENCE_SCOPE_SOURCE_MISMATCH", "Evidence scope is outside this semantic block's canonical source scope.", path=block_path, learning_block_id=block_id or None, scope_id=scope_id)
                        if ownership == "primary":
                            primary_owners.setdefault(scope_id, []).append((unit_path, block_id, block_path))
                    if len(scope_documents) > 1:
                        add("CROSS_DOCUMENT_EVIDENCE_SCOPE_CLAIM", "One semantic learning block cannot combine evidence scopes from different documents.", path=block_path, learning_block_id=block_id or None)

    for scope_id, scope in scopes.items():
        if not isinstance(scope.get("source_fact_ids"), list) or not scope.get("source_fact_ids"):
            continue
        owners = primary_owners.get(scope_id, [])
        if not owners:
            scope_concepts = {
                str(value).strip()
                for value in scope.get("concept_ids", [])
                if isinstance(value, str) and value.strip()
            }
            eligible = [
                candidate
                for candidate in block_candidates
                if (not scope_concepts or scope_concepts.issubset(candidate["concept_ids"]))
                and ref_matches_scope(scope, candidate["source_refs"])
            ]
            eligible_unit_paths = sorted({str(candidate["unit_path"]) for candidate in eligible})
            eligible_block_paths = sorted({str(candidate["block_path"]) for candidate in eligible})
            if len(eligible_unit_paths) == 1:
                add(
                    "MISSING_PRIMARY_EVIDENCE_SCOPE_OWNER",
                    "A canonical evidence scope has no primary semantic learning-block owner.",
                    path=eligible_unit_paths[0],
                    scope_id=scope_id,
                    eligible_repair_paths=eligible_block_paths,
                    repairable=True,
                )
            else:
                # There is no deterministic smallest patch target. A repair
                # must not choose an arbitrary chapter/unit just to force
                # coverage, so stop before any broad provider repair.
                add(
                    "MISSING_PRIMARY_EVIDENCE_SCOPE_OWNER",
                    "A canonical evidence scope has no uniquely eligible primary semantic learning-block owner.",
                    path="course",
                    scope_id=scope_id,
                    eligible_repair_paths=eligible_block_paths,
                    repairable=False,
                )
                add(
                    "EVIDENCE_SCOPE_REPAIR_TARGET_UNRESOLVED",
                    "The immutable Source Map cannot identify one safe local target for evidence-scope ownership repair.",
                    path="course",
                    scope_id=scope_id,
                    eligible_repair_paths=eligible_block_paths,
                    repairable=False,
                )
        elif len(owners) != 1:
            add("DUPLICATE_PRIMARY_EVIDENCE_SCOPE_OWNER", "A canonical evidence scope has multiple primary semantic learning-block owners.", path=owners[0][2], learning_block_id=owners[0][1] or None, scope_id=scope_id)

    unknown_concepts: set[str] = set()
    for chapter in blueprint.get("chapters", []):
        if not isinstance(chapter, dict):
            continue
        unknown_concepts.update(
            concept_id for concept_id in _source_fact_allocation_values(chapter, "concept_ids")
            if concept_id not in concepts
        )
        for lesson in chapter.get("lessons", []):
            if not isinstance(lesson, dict):
                continue
            for key in ("primary_concept_ids", "supporting_concept_ids"):
                unknown_concepts.update(
                    concept_id for concept_id in _source_fact_allocation_values(lesson, key)
                    if concept_id not in concepts
                )
            for unit in lesson.get("units", []):
                if not isinstance(unit, dict):
                    continue
                unknown_concepts.update(
                    concept_id for concept_id in _source_fact_allocation_values(unit, "concept_ids")
                    if concept_id not in concepts
                )
                for block in unit.get("learning_blocks", []):
                    if isinstance(block, dict):
                        unknown_concepts.update(
                            concept_id for concept_id in _source_fact_allocation_values(block, "concept_ids")
                            if concept_id not in concepts
                        )
    for concept_id in sorted(unknown_concepts):
        add("UNKNOWN_CONCEPT_ID", "Course architecture selected a concept outside the Source Map.", path="course")

    required_scopes = {
        scope_id for scope_id, scope in scopes.items()
        if isinstance(scope.get("source_fact_ids"), list) and scope.get("source_fact_ids")
    }
    return WorkflowValidationResult(
        issues=issues,
        metrics={
            "canonical_evidence_scope_count": len(required_scopes),
            "referenced_evidence_scope_count": len(required_scopes & referenced_scopes),
            "primary_owned_evidence_scope_count": len(required_scopes & set(primary_owners)),
            "evidence_scope_coverage": round(len(required_scopes & set(primary_owners)) / max(1, len(required_scopes)), 4),
        },
    )


def validate_course_architecture_semantic_scope(
    blueprint: dict[str, Any],
    source_map: dict[str, Any],
) -> WorkflowValidationResult:
    """Validate canonical semantic ownership before any fact is allocated.

    The Course Architect may choose only canonical Source Map concept IDs and
    source references.  This gate deliberately operates on the hierarchy,
    not individual facts, so a missing or ambiguous owner becomes a small,
    explainable architecture finding instead of hundreds of derived failures.
    """
    if blueprint.get("architecture_contract_version") != 4:
        return WorkflowValidationResult()

    sections = {
        str(section.get("id") or "").strip(): section
        for section in source_map.get("sections", [])
        if isinstance(section, dict) and str(section.get("id") or "").strip()
    }
    concepts = {
        str(concept.get("id") or "").strip(): concept
        for concept in source_map.get("concepts", [])
        if isinstance(concept, dict) and str(concept.get("id") or "").strip()
    }
    known_source_refs = {
        str(section.get("source_ref") or "").strip()
        for section in sections.values()
        if str(section.get("source_ref") or "").strip()
    }
    concept_sections = {
        concept_id: {
            str(section_id).strip()
            for section_id in concept.get("source_section_ids", [])
            if isinstance(section_id, str) and section_id.strip()
        }
        for concept_id, concept in concepts.items()
    }
    issues: list[WorkflowIssue] = []

    def add(
        code: str,
        message: str,
        *,
        path: str,
        learning_block_id: str | None = None,
        related_paths: list[str] | None = None,
        concept_id: str | None = None,
        concept_ids: list[str] | None = None,
        section_id: str | None = None,
        section_ids: list[str] | None = None,
        expected_primary_owner_count: int | None = None,
        actual_primary_owner_count: int | None = None,
        ownership_level: str | None = None,
        scope_classification: str | None = None,
        coverage_state: str | None = None,
        safe_reason: str | None = None,
    ) -> None:
        issue: WorkflowIssue = {"code": code, "severity": "error", "message": message, "path": path}
        if learning_block_id:
            issue["learning_block_ids"] = [learning_block_id]
        if related_paths:
            issue["related_paths"] = sorted({safe_workflow_path(item) for item in related_paths})
        if concept_id:
            issue["concept_id"] = concept_id
        if concept_ids:
            issue["concept_ids"] = sorted({item for item in concept_ids if item})[:12]
        if section_id:
            issue["section_id"] = section_id
        if section_ids:
            issue["section_ids"] = sorted({item for item in section_ids if item})[:12]
        if expected_primary_owner_count is not None:
            issue["expected_primary_owner_count"] = expected_primary_owner_count
        if actual_primary_owner_count is not None:
            issue["actual_primary_owner_count"] = actual_primary_owner_count
        if ownership_level:
            issue["ownership_level"] = ownership_level
        if scope_classification:
            issue["scope_classification"] = scope_classification
        if coverage_state:
            issue["coverage_state"] = coverage_state
        if safe_reason:
            issue["safe_reason"] = safe_reason
        issues.append(issue)

    def source_ref_matches_section(section_id: str, source_refs: set[str]) -> bool:
        if not source_refs:
            return True
        current = section_id
        visited: set[str] = set()
        while current and current not in visited:
            visited.add(current)
            section = sections.get(current)
            if not isinstance(section, dict):
                return False
            if str(section.get("source_ref") or "").strip() in source_refs:
                return True
            current = str(section.get("parent_id") or "").strip()
        return False

    def validate_scope_values(
        *,
        concept_ids: set[str],
        source_refs: set[str],
        path: str,
        learning_block_id: str | None = None,
        require_scope: bool = True,
    ) -> None:
        if require_scope and not concept_ids and not source_refs:
            add("MISSING_INSTRUCTIONAL_SCOPE", "Instructional node has no canonical concept or source scope.", path=path, learning_block_id=learning_block_id)
            return
        for concept_id in sorted(concept_ids):
            if concept_id not in concepts:
                add("UNKNOWN_CONCEPT_ID", "Instructional node selected a concept ID outside the Source Map.", path=path, learning_block_id=learning_block_id)
        for source_ref in sorted(source_refs):
            if source_ref not in known_source_refs:
                add("UNKNOWN_SOURCE_REF", "Instructional node selected a source reference outside the Source Map.", path=path, learning_block_id=learning_block_id)
        if source_refs:
            for concept_id in sorted(concept_ids & concepts.keys()):
                source_sections = concept_sections.get(concept_id, set())
                if source_sections and not any(source_ref_matches_section(section_id, source_refs) for section_id in source_sections):
                    add("CONCEPT_SOURCE_SCOPE_MISMATCH", "Concept ownership conflicts with this node's canonical source scope.", path=path, learning_block_id=learning_block_id)

    primary_destinations: dict[str, list[str]] = {}
    concept_scope_paths: dict[str, list[str]] = {}
    section_scope_paths: dict[str, list[str]] = {}

    def record_scope(
        *,
        path: str,
        concept_ids: set[str],
        source_refs: set[str],
    ) -> None:
        safe_path = safe_workflow_path(path)
        for concept_id in concept_ids & concepts.keys():
            concept_scope_paths.setdefault(concept_id, []).append(safe_path)
            for section_id in concept_sections.get(concept_id, set()):
                section_scope_paths.setdefault(section_id, []).append(safe_path)
        for section_id in sections:
            if source_refs and source_ref_matches_section(section_id, source_refs):
                section_scope_paths.setdefault(section_id, []).append(safe_path)

    def nearest_scope_paths(paths: list[str]) -> list[str]:
        """Return deterministic, most-specific structural paths only."""

        unique = {safe_workflow_path(path) for path in paths if safe_workflow_path(path)}
        return sorted(unique, key=lambda path: (-path.count("."), path))[:12]

    for chapter_index, chapter in enumerate(blueprint.get("chapters", []), start=1):
        if not isinstance(chapter, dict):
            continue
        chapter_path = f"chapter_{chapter_index}"
        chapter_concepts = _source_fact_allocation_values(chapter, "concept_ids")
        chapter_refs = _source_fact_allocation_values(chapter, "source_refs")
        record_scope(path=chapter_path, concept_ids=chapter_concepts, source_refs=chapter_refs)
        validate_scope_values(concept_ids=chapter_concepts, source_refs=chapter_refs, path=chapter_path)
        for lesson_index, lesson in enumerate(chapter.get("lessons", []), start=1):
            if not isinstance(lesson, dict):
                continue
            lesson_path = f"{chapter_path}.lesson_{lesson_index}"
            lesson_primary = _source_fact_allocation_values(lesson, "primary_concept_ids")
            lesson_supporting = _source_fact_allocation_values(lesson, "supporting_concept_ids")
            lesson_refs = _source_fact_allocation_values(lesson, "source_refs")
            record_scope(
                path=lesson_path,
                concept_ids=lesson_primary | lesson_supporting,
                source_refs=lesson_refs,
            )
            validate_scope_values(
                concept_ids=lesson_primary | lesson_supporting,
                source_refs=lesson_refs,
                path=lesson_path,
            )
            if lesson_primary and chapter_concepts and not lesson_primary.issubset(chapter_concepts):
                add("CONCEPT_SOURCE_SCOPE_MISMATCH", "Lesson primary concept is outside the chapter's canonical concept scope.", path=lesson_path)
            descendant_block_primary: set[str] = set()
            for unit_index, unit in enumerate(lesson.get("units", []), start=1):
                if not isinstance(unit, dict):
                    continue
                unit_path = f"{lesson_path}.unit_{unit_index}"
                unit_concepts = _source_fact_allocation_values(unit, "concept_ids")
                unit_primary = _source_fact_allocation_values(unit, "primary_concept_ids")
                unit_refs = _source_fact_allocation_values(unit, "source_refs")
                record_scope(path=unit_path, concept_ids=unit_concepts, source_refs=unit_refs)
                validate_scope_values(concept_ids=unit_concepts, source_refs=unit_refs, path=unit_path)
                if not unit_primary.issubset(unit_concepts):
                    add("CONCEPT_SOURCE_SCOPE_MISMATCH", "Unit primary concept must be declared in unit concept_ids.", path=unit_path)
                if not unit_primary.issubset(lesson_primary):
                    add("CONCEPT_SOURCE_SCOPE_MISMATCH", "Unit primary concept must remain within the lesson's primary ownership.", path=unit_path)
                blocks = unit.get("learning_blocks") if isinstance(unit.get("learning_blocks"), list) else []
                if not blocks:
                    add("MISSING_INSTRUCTIONAL_SCOPE", "Unit has no semantic learning block that can own source-grounded instruction.", path=unit_path)
                for block_index, block in enumerate(blocks, start=1):
                    if not isinstance(block, dict):
                        continue
                    block_path = f"{unit_path}.block_{block_index}"
                    block_id = str(block.get("id") or "").strip()
                    block_concepts = _source_fact_allocation_values(block, "concept_ids")
                    block_primary = _source_fact_allocation_values(block, "primary_concept_ids")
                    descendant_block_primary.update(block_primary)
                    block_refs = _source_fact_allocation_values(block, "source_refs")
                    record_scope(path=block_path, concept_ids=block_concepts, source_refs=block_refs)
                    validate_scope_values(
                        concept_ids=block_concepts,
                        source_refs=block_refs,
                        path=block_path,
                        learning_block_id=block_id or None,
                    )
                    if not block_primary.issubset(block_concepts):
                        add("CONCEPT_SOURCE_SCOPE_MISMATCH", "Learning-block primary concept must be declared in block concept_ids.", path=block_path, learning_block_id=block_id or None)
                    if not block_primary.issubset(unit_primary):
                        add("CONCEPT_SOURCE_SCOPE_MISMATCH", "Learning-block primary concept must remain within the unit's primary ownership.", path=block_path, learning_block_id=block_id or None)
                    for concept_id in block_primary & concepts.keys():
                        primary_destinations.setdefault(concept_id, []).append(block_path)
            # A lesson's primary concept is an instructional promise, not
            # decorative metadata. Canonical facts can only be allocated to a
            # primary semantic block, so that block must be a descendant of
            # the lesson declaring the primary scope.
            if lesson_primary - descendant_block_primary:
                add(
                    "LESSON_PRIMARY_OWNERSHIP_WITHOUT_PRIMARY_UNIT",
                    "Lesson primary ownership has no descendant primary unit and semantic learning block.",
                    path=lesson_path,
                )

    required_concepts = {
        concept_id
        for concept_id, concept in concepts.items()
        if isinstance(concept.get("source_fact_ids"), list) and concept.get("source_fact_ids")
    }
    covered_sections: set[str] = set()
    for concept_id in sorted(required_concepts):
        destinations = sorted(set(primary_destinations.get(concept_id, [])))
        source_section_ids = sorted(concept_sections.get(concept_id, set()))
        nearest_paths = nearest_scope_paths(concept_scope_paths.get(concept_id, []))
        nearest_path = nearest_paths[0] if nearest_paths else "course"
        if not destinations:
            add(
                "ARCHITECTURE_SCOPE_INCOMPLETE",
                "A canonical Source Map concept with source facts has no primary instructional destination.",
                path=nearest_path,
                related_paths=nearest_paths,
                concept_id=concept_id,
                section_id=source_section_ids[0] if source_section_ids else None,
                section_ids=source_section_ids,
                expected_primary_owner_count=1,
                actual_primary_owner_count=0,
                ownership_level="learning_block",
                scope_classification="concept_primary_ownership",
                coverage_state="missing",
                safe_reason="NO_PRIMARY_DESTINATION",
            )
            continue
        if len(destinations) != 1:
            add(
                "AMBIGUOUS_INSTRUCTIONAL_SCOPE",
                "A canonical Source Map concept has multiple primary instructional destinations.",
                path=destinations[0],
                related_paths=destinations,
                concept_id=concept_id,
                section_id=source_section_ids[0] if source_section_ids else None,
                section_ids=source_section_ids,
                expected_primary_owner_count=1,
                actual_primary_owner_count=len(destinations),
                ownership_level="learning_block",
                scope_classification="concept_primary_ownership",
                coverage_state="ambiguous",
                safe_reason="MULTIPLE_PRIMARY_DESTINATIONS",
            )
            continue
        covered_sections.update(concept_sections.get(concept_id, set()))

    required_sections = {
        str(section.get("id") or "").strip()
        for section in sections.values()
        if isinstance(section.get("source_fact_ids"), list) and section.get("source_fact_ids")
    }
    for section_id in sorted(required_sections - covered_sections):
        scoped_concepts = sorted(
            concept_id
            for concept_id, section_ids in concept_sections.items()
            if section_id in section_ids and concept_id in required_concepts
        )
        nearest_paths = nearest_scope_paths(section_scope_paths.get(section_id, []))
        primary_count = sum(
            len(set(primary_destinations.get(concept_id, [])))
            for concept_id in scoped_concepts
        )
        add(
            "ARCHITECTURE_SCOPE_INCOMPLETE",
            "A canonical Source Map section with source facts has no eligible primary instructional destination.",
            path=nearest_paths[0] if nearest_paths else "course",
            related_paths=nearest_paths,
            concept_id=scoped_concepts[0] if len(scoped_concepts) == 1 else None,
            concept_ids=scoped_concepts,
            section_id=section_id,
            expected_primary_owner_count=1,
            actual_primary_owner_count=primary_count,
            ownership_level="learning_block",
            scope_classification="section_coverage",
            coverage_state="missing",
            safe_reason="NO_ELIGIBLE_PRIMARY_DESTINATION",
        )

    return WorkflowValidationResult(
        issues=issues,
        metrics={
            "canonical_section_count": len(required_sections),
            "covered_section_count": len(required_sections & covered_sections),
            "canonical_concept_count": len(required_concepts),
            "covered_concept_count": len({concept_id for concept_id, destinations in primary_destinations.items() if len(set(destinations)) == 1}),
        },
    )


_SEMANTIC_SCOPE_DIAGNOSTIC_FIELDS = (
    "related_paths",
    "concept_id",
    "concept_ids",
    "section_id",
    "section_ids",
    "expected_primary_owner_count",
    "actual_primary_owner_count",
    "ownership_level",
    "scope_classification",
    "coverage_state",
    "safe_reason",
)


def _semantic_scope_preallocation_metadata(issue: WorkflowIssue) -> dict[str, Any]:
    """Persist safe semantic diagnostics through allocation/workflow validation."""

    summary = safe_workflow_issue_summary(issue, repairable=False)
    return {
        "code": str(issue.get("code") or "ARCHITECTURE_SCOPE_INCOMPLETE"),
        "path": safe_workflow_path(issue.get("path")),
        **{
            key: summary[key]
            for key in _SEMANTIC_SCOPE_DIAGNOSTIC_FIELDS
            if key in summary
        },
        **({"learning_block_ids": list(issue.get("learning_block_ids") or [])[:12]} if issue.get("learning_block_ids") else {}),
    }


def allocate_source_map_evidence_scope_facts(
    blueprint: dict[str, Any],
    source_map: dict[str, Any],
    manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    """Expand server-owned v5 primary evidence scopes into canonical Facts.

    This allocator intentionally has no similarity, positional, or coverage
    fallback.  The Architect chooses semantic *scope IDs* only; the immutable
    Source Map is the sole authority for their member Fact IDs. Supporting
    references stay on blocks for grounding, but never receive Fact ownership.
    """
    if blueprint.get("architecture_contract_version") != 5:
        return blueprint
    if "source_fact_allocation" in blueprint or "source_evidence_scope_allocation" in blueprint:
        raise LessonAuthorBlueprintValidationError(
            "BLUEPRINT_PROVIDER_FACT_OWNERSHIP_FORBIDDEN",
            "Evidence-scope and Source Fact allocation must be created only by the server allocator.",
        )

    manifest_facts = {
        str(fact.get("fact_id") or "").strip(): fact
        for fact in (manifest or {}).get("facts", [])
        if isinstance(fact, dict) and str(fact.get("fact_id") or "").strip()
    }
    scopes = {
        str(scope.get("id") or "").strip(): scope
        for scope in source_map.get("source_evidence_scopes", [])
        if isinstance(scope, dict) and str(scope.get("id") or "").strip()
    }
    semantic = validate_course_architecture_evidence_scope(blueprint, source_map)
    semantic_errors = [issue for issue in semantic.issues if issue.get("severity") == "error"]
    if semantic_errors:
        return {
            **blueprint,
            "source_evidence_scope_allocation": {
                "version": "source-evidence-scope-allocation-v1",
                "authority": "server",
                "architecture_contract_version": 5,
                "required_count": len(scopes),
                "allocated_count": 0,
                "complete": False,
                "allocations": [],
                "unallocated": [],
                "preallocation_validation": [_semantic_scope_preallocation_metadata(issue) for issue in semantic_errors[:80]],
                "semantic_scope_metrics": semantic.metrics,
            },
            "source_fact_allocation": {
                "version": "source-fact-allocation-v3",
                "authority": "server",
                "architecture_contract_version": 5,
                "required_count": len(manifest_facts),
                "allocated_count": 0,
                "complete": False,
                "allocations": [],
                "unallocated": [],
                "preallocation_validation": [_semantic_scope_preallocation_metadata(issue) for issue in semantic_errors[:80]],
                "semantic_scope_metrics": semantic.metrics,
            },
        }

    scope_targets: dict[str, tuple[dict[str, Any], dict[str, Any], str, str]] = {}
    for chapter_index, chapter in enumerate(blueprint.get("chapters", []), start=1):
        if not isinstance(chapter, dict):
            continue
        for lesson_index, lesson in enumerate(chapter.get("lessons", []), start=1):
            if not isinstance(lesson, dict):
                continue
            for unit_index, unit in enumerate(lesson.get("units", []), start=1):
                if not isinstance(unit, dict):
                    continue
                unit_path = f"chapter_{chapter_index}.lesson_{lesson_index}.unit_{unit_index}"
                blocks = unit.get("learning_blocks") if isinstance(unit.get("learning_blocks"), list) else []
                unit["source_fact_ids"] = []
                unit["primary_evidence_scope_ids"] = []
                unit["supporting_evidence_scope_ids"] = []
                for block in blocks:
                    if not isinstance(block, dict):
                        continue
                    if "source_fact_ids" in block or "covered_source_fact_ids" in block:
                        raise LessonAuthorBlueprintValidationError(
                            "BLUEPRINT_PROVIDER_FACT_OWNERSHIP_FORBIDDEN",
                            "Course Architect semantic blocks must not submit canonical source fact IDs.",
                        )
                    block["source_fact_ids"] = []
                    block_id = str(block.get("id") or "").strip()
                    for scope_id in _source_fact_allocation_values(block, "primary_evidence_scope_ids"):
                        scope_targets[scope_id] = (unit, block, unit_path, block_id)

    scope_allocations: list[dict[str, Any]] = []
    scope_unallocated: list[dict[str, str]] = []
    fact_allocations: list[dict[str, str]] = []
    fact_unallocated: list[dict[str, str]] = []
    seen_fact_ids: set[str] = set()
    for scope_id, scope in sorted(scopes.items()):
        target = scope_targets.get(scope_id)
        scope_fact_ids = [
            str(value).strip() for value in scope.get("source_fact_ids", [])
            if isinstance(value, str) and str(value).strip()
        ]
        if target is None:
            scope_unallocated.append({"evidence_scope_id": scope_id, "code": "UNALLOCATED_EVIDENCE_SCOPE", "path": "course"})
            continue
        unit, block, unit_path, block_id = target
        if not block_id or not scope_fact_ids:
            scope_unallocated.append({"evidence_scope_id": scope_id, "code": "EVIDENCE_SCOPE_PROVENANCE_INVALID", "path": unit_path})
            continue
        if any(fact_id not in manifest_facts for fact_id in scope_fact_ids):
            scope_unallocated.append({"evidence_scope_id": scope_id, "code": "EVIDENCE_SCOPE_PROVENANCE_INVALID", "path": unit_path})
            continue
        if any(fact_id in seen_fact_ids for fact_id in scope_fact_ids):
            scope_unallocated.append({"evidence_scope_id": scope_id, "code": "DUPLICATE_SOURCE_FACT_SCOPE_MEMBERSHIP", "path": unit_path})
            continue
        if (
            len(unit["source_fact_ids"]) + len(scope_fact_ids) > MAX_SERVER_OWNED_SOURCE_FACT_IDS_PER_SCOPE
            or len(block["source_fact_ids"]) + len(scope_fact_ids) > MAX_SERVER_OWNED_SOURCE_FACT_IDS_PER_SCOPE
        ):
            scope_unallocated.append({"evidence_scope_id": scope_id, "code": "ARCHITECTURE_FACT_CAPACITY_EXCEEDED", "path": unit_path})
            continue
        seen_fact_ids.update(scope_fact_ids)
        unit["primary_evidence_scope_ids"].append(scope_id)
        unit["source_fact_ids"].extend(scope_fact_ids)
        block["source_fact_ids"].extend(scope_fact_ids)
        scope_allocations.append({
            "evidence_scope_id": scope_id,
            "unit_path": unit_path,
            "learning_block_id": block_id,
            "basis": "PRIMARY_EVIDENCE_SCOPE",
            "evidence_char_count": int(scope.get("evidence_char_count") or 0),
            "evidence_token_estimate": int(scope.get("evidence_token_estimate") or 0),
        })
        fact_allocations.extend({
            "fact_id": fact_id,
            "unit_path": unit_path,
            "learning_block_id": block_id,
            "evidence_scope_id": scope_id,
            "basis": "PRIMARY_EVIDENCE_SCOPE",
        } for fact_id in scope_fact_ids)

    for chapter in blueprint.get("chapters", []):
        if not isinstance(chapter, dict):
            continue
        for lesson in chapter.get("lessons", []):
            if not isinstance(lesson, dict):
                continue
            for unit in lesson.get("units", []):
                if not isinstance(unit, dict):
                    continue
                supporting: set[str] = set()
                for block in unit.get("learning_blocks", []) if isinstance(unit.get("learning_blocks"), list) else []:
                    if isinstance(block, dict):
                        supporting.update(_source_fact_allocation_values(block, "supporting_evidence_scope_ids"))
                unit["primary_evidence_scope_ids"] = sorted(set(unit.get("primary_evidence_scope_ids") or []))
                unit["supporting_evidence_scope_ids"] = sorted(supporting)
                unit["source_fact_ids"] = sorted(set(unit.get("source_fact_ids") or []))
                for block in unit.get("learning_blocks", []) if isinstance(unit.get("learning_blocks"), list) else []:
                    if isinstance(block, dict):
                        block["source_fact_ids"] = sorted(set(block.get("source_fact_ids") or []))

    for fact_id in sorted(set(manifest_facts) - seen_fact_ids):
        fact_unallocated.append({"fact_id": fact_id, "code": "UNALLOCATED_SOURCE_FACT", "path": "course"})
    source_scope_complete = not scope_unallocated and len(scope_allocations) == len(scopes)
    source_fact_complete = not fact_unallocated and len(fact_allocations) == len(manifest_facts)
    return {
        **blueprint,
        "source_evidence_scope_allocation": {
            "version": "source-evidence-scope-allocation-v1",
            "authority": "server",
            "architecture_contract_version": 5,
            "required_count": len(scopes),
            "allocated_count": len(scope_allocations),
            "complete": source_scope_complete,
            "allocations": scope_allocations,
            "unallocated": scope_unallocated,
            "evidence_char_count": sum(int(scope.get("evidence_char_count") or 0) for scope in scopes.values()),
            "evidence_token_estimate": sum(int(scope.get("evidence_token_estimate") or 0) for scope in scopes.values()),
        },
        "source_fact_allocation": {
            "version": "source-fact-allocation-v3",
            "authority": "server",
            "architecture_contract_version": 5,
            "required_count": len(manifest_facts),
            "allocated_count": len(fact_allocations),
            "complete": source_scope_complete and source_fact_complete,
            "allocations": fact_allocations,
            "unallocated": fact_unallocated,
        },
    }


def allocate_source_map_architecture_facts(
    blueprint: dict[str, Any],
    source_map: dict[str, Any],
    manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    """Bind each canonical fact only where deterministic ownership supports it.

    This is intentionally not a coverage-filling partitioner. A fact needs a
    unique supported chain from its source section/reference to a unit and a
    semantic block. Ambiguous or unmatched facts remain explicit validation
    failures, allowing the existing bounded architecture repair path to act
    on the smallest safe scope instead of assigning them positionally.
    """
    if blueprint.get("architecture_contract_version") == 5:
        return allocate_source_map_evidence_scope_facts(blueprint, source_map, manifest)
    if blueprint.get("architecture_contract_version") != 4:
        return blueprint
    source_facts = [
        fact for fact in (manifest or {}).get("facts", [])
        if isinstance(fact, dict) and str(fact.get("fact_id") or "").strip()
    ]
    map_facts = {
        str(fact.get("id") or "").strip(): fact
        for fact in source_map.get("facts", []) if isinstance(fact, dict) and str(fact.get("id") or "").strip()
    }
    concepts = {
        str(concept.get("id") or "").strip(): concept
        for concept in source_map.get("concepts", []) if isinstance(concept, dict) and str(concept.get("id") or "").strip()
    }
    sections = {
        str(section.get("id") or "").strip(): section
        for section in source_map.get("sections", []) if isinstance(section, dict) and str(section.get("id") or "").strip()
    }
    fact_by_id = {str(fact.get("fact_id") or "").strip(): fact for fact in source_facts}

    # v4 is deliberately semantic-only until this function completes.  The
    # provider cannot steer canonical evidence ownership through IDs it saw in
    # a prompt.  A caller must strip a previous *server* allocation before
    # reallocation after a scoped repair; seeing one here is a contract error.
    if "source_fact_allocation" in blueprint:
        raise LessonAuthorBlueprintValidationError(
            "BLUEPRINT_PROVIDER_FACT_OWNERSHIP_FORBIDDEN",
            "Source Fact allocation must be created only by the server allocator.",
        )

    semantic_scope = validate_course_architecture_semantic_scope(blueprint, source_map)
    semantic_errors = [issue for issue in semantic_scope.issues if issue.get("severity") == "error"]
    if semantic_errors:
        # Do not derive hundreds of fact-level failures from an invalid
        # architecture.  The semantic prerequisite is a server validation,
        # not provider-owned allocation metadata.
        return {
            **blueprint,
            "source_fact_allocation": {
                "version": "source-fact-allocation-v2",
                "authority": "server",
                "architecture_contract_version": 4,
                "required_count": len(fact_by_id),
                "allocated_count": 0,
                "complete": False,
                "allocations": [],
                "unallocated": [],
                "preallocation_validation": [
                    _semantic_scope_preallocation_metadata(issue)
                    for issue in semantic_errors[:80]
                ],
                "semantic_scope_metrics": semantic_scope.metrics,
            },
        }

    units: list[dict[str, Any]] = []
    for chapter_index, chapter in enumerate(blueprint.get("chapters", []), start=1):
        if not isinstance(chapter, dict):
            continue
        chapter_refs = {
            value.strip() for value in chapter.get("source_refs", [])
            if isinstance(value, str) and value.strip()
        }
        chapter_concepts = {
            value.strip() for value in chapter.get("concept_ids", [])
            if isinstance(value, str) and value.strip()
        }
        for lesson_index, lesson in enumerate(chapter.get("lessons", []), start=1):
            if not isinstance(lesson, dict):
                continue
            lesson_direct_refs = {
                value.strip() for value in lesson.get("source_refs", [])
                if isinstance(value, str) and value.strip()
            }
            primary_concepts = {
                value.strip() for value in lesson.get("primary_concept_ids", [])
                if isinstance(value, str) and value.strip()
            }
            supporting_concepts = {
                value.strip() for value in lesson.get("supporting_concept_ids", [])
                if isinstance(value, str) and value.strip()
            }
            lesson_concepts = primary_concepts | supporting_concepts
            for unit_index, unit in enumerate(lesson.get("units", []), start=1):
                if not isinstance(unit, dict):
                    continue
                path = f"chapter_{chapter_index}.lesson_{lesson_index}.unit_{unit_index}"
                unit_refs = {
                    value.strip() for value in unit.get("source_refs", [])
                    if isinstance(value, str) and value.strip()
                }
                unit_concepts = {
                    value.strip() for value in unit.get("concept_ids", [])
                    if isinstance(value, str) and value.strip()
                }
                unit_primary_concepts = {
                    value.strip() for value in unit.get("primary_concept_ids", [])
                    if isinstance(value, str) and value.strip()
                }
                blocks = [block for block in unit.get("learning_blocks", []) if isinstance(block, dict)]
                for block in blocks:
                    if "source_fact_ids" in block or "covered_source_fact_ids" in block:
                        raise LessonAuthorBlueprintValidationError(
                            "BLUEPRINT_PROVIDER_FACT_OWNERSHIP_FORBIDDEN",
                            "Course Architect semantic blocks must not submit canonical source fact IDs.",
                        )
                    block["source_fact_ids"] = []
                if "source_fact_ids" in unit or "covered_source_fact_ids" in unit:
                    raise LessonAuthorBlueprintValidationError(
                        "BLUEPRINT_PROVIDER_FACT_OWNERSHIP_FORBIDDEN",
                        "Course Architect units must not submit canonical source fact IDs.",
                    )
                unit["source_fact_ids"] = []
                units.append({
                    "path": path,
                    "unit": unit,
                    "blocks": blocks,
                    "chapter_refs": chapter_refs,
                    "chapter_concepts": chapter_concepts,
                    "lesson_refs": lesson_direct_refs,
                    "lesson_concepts": lesson_concepts,
                    "primary_concepts": primary_concepts,
                    "unit_refs": unit_refs,
                    "unit_concepts": unit_concepts,
                    "unit_primary_concepts": unit_primary_concepts,
                })

    concept_section_index = {
        concept_id: {
            section_id
            for section_id in concept.get("source_section_ids", [])
            if isinstance(section_id, str) and section_id
        }
        for concept_id, concept in concepts.items()
    }

    def concept_sections(concept_ids: set[str]) -> set[str]:
        return set().union(*(concept_section_index.get(concept_id, set()) for concept_id in concept_ids))

    def fact_matches_source_scope(fact: dict[str, Any], source_refs: set[str]) -> bool:
        """Match a fact's source section or one of its documented ancestors."""
        if not source_refs:
            return False
        source_ref = str(fact.get("source_ref") or "").strip()
        if source_ref in source_refs:
            return True
        section_id = str(fact.get("section_id") or "").strip()
        visited: set[str] = set()
        while section_id and section_id not in visited:
            visited.add(section_id)
            section = sections.get(section_id)
            if not isinstance(section, dict):
                return False
            if str(section.get("source_ref") or "").strip() in source_refs:
                return True
            section_id = str(section.get("parent_id") or "").strip()
        return False

    def unit_basis(unit_context: dict[str, Any], fact: dict[str, Any]) -> tuple[int, str] | None:
        section_id = str(fact.get("section_id") or "").strip()
        # A direct unit source scope is an explicit exclusion boundary. A
        # primary concept cannot move a fact across it merely to increase
        # coverage. Parent source sections are accepted for child facts.
        if unit_context["unit_refs"] and not fact_matches_source_scope(fact, unit_context["unit_refs"]):
            return None
        if not unit_context["unit_refs"] and unit_context["lesson_refs"] and not fact_matches_source_scope(fact, unit_context["lesson_refs"]):
            return None
        if not unit_context["unit_refs"] and not unit_context["lesson_refs"] and unit_context["chapter_refs"] and not fact_matches_source_scope(fact, unit_context["chapter_refs"]):
            return None
        if section_id and section_id in concept_sections(unit_context["unit_primary_concepts"]):
            return 9, "OWNERSHIP_MATCH"
        if unit_context["unit_refs"] and fact_matches_source_scope(fact, unit_context["unit_refs"]):
            return 6, "SOURCE_REF_MATCH"
        if section_id and section_id in concept_sections(unit_context["unit_concepts"]):
            return 5, "CONCEPT_MATCH"
        if section_id and section_id in concept_sections(unit_context["primary_concepts"]):
            return 4, "OWNERSHIP_MATCH"
        if unit_context["lesson_refs"] and fact_matches_source_scope(fact, unit_context["lesson_refs"]):
            return 3, "SOURCE_REF_MATCH"
        if section_id and section_id in concept_sections(unit_context["lesson_concepts"]):
            return 2, "OWNERSHIP_MATCH"
        if unit_context["chapter_refs"] and fact_matches_source_scope(fact, unit_context["chapter_refs"]):
            return 1, "SECTION_MATCH"
        if section_id and section_id in concept_sections(unit_context["chapter_concepts"]):
            return 1, "SECTION_MATCH"
        return None

    def block_basis(block: dict[str, Any], fact: dict[str, Any]) -> tuple[int, str] | None:
        section_id = str(fact.get("section_id") or "").strip()
        # Only a block that explicitly declares a canonical primary concept
        # may own the fact. Reinforcement/practice blocks can reference the
        # concept but must never steal the server-owned canonical allocation.
        if section_id and section_id in concept_sections(_source_fact_allocation_values(block, "primary_concept_ids")):
            return 10, "OWNERSHIP_MATCH"
        return None

    allocations: list[dict[str, str]] = []
    unallocated: list[dict[str, str]] = []
    rejection_counts: dict[str, int] = {}

    def reject(fact_id: str, code: str, path: str) -> None:
        """Record only stable operational allocation metadata, never source text."""

        unallocated.append({"fact_id": fact_id, "code": code, "path": path})
        rejection_counts[code] = rejection_counts.get(code, 0) + 1
    for fact_id, manifest_fact in fact_by_id.items():
        fact = map_facts.get(fact_id)
        if fact is None:
            reject(fact_id, "SOURCE_MAP_PROVENANCE_INVALID", "course")
            continue
        candidates: list[tuple[tuple[int, int], dict[str, Any], dict[str, Any], str]] = []
        for unit_context in units:
            architecture_match = unit_basis(unit_context, fact)
            if architecture_match is None:
                continue
            for block in unit_context["blocks"]:
                semantic_match = block_basis(block, fact)
                if semantic_match is None:
                    continue
                candidates.append((
                    (architecture_match[0], semantic_match[0]),
                    unit_context,
                    block,
                    semantic_match[1] if semantic_match[0] >= architecture_match[0] else architecture_match[1],
                ))
        if not candidates:
            reject(fact_id, "UNALLOCATED_SOURCE_FACT", "course")
            continue
        best_rank = max(candidate[0] for candidate in candidates)
        best = [candidate for candidate in candidates if candidate[0] == best_rank]
        unique_targets = {(candidate[1]["path"], str(candidate[2].get("id") or "")) for candidate in best}
        if len(unique_targets) != 1:
            reject(fact_id, "AMBIGUOUS_SOURCE_FACT_OWNERSHIP", "course")
            continue
        _rank, unit_context, block, basis = best[0]
        if (
            len(unit_context["unit"]["source_fact_ids"]) >= MAX_SERVER_OWNED_SOURCE_FACT_IDS_PER_SCOPE
            or len(block["source_fact_ids"]) >= MAX_SERVER_OWNED_SOURCE_FACT_IDS_PER_SCOPE
        ):
            reject(fact_id, "ARCHITECTURE_FACT_CAPACITY_EXCEEDED", unit_context["path"])
            continue
        unit_context["unit"]["source_fact_ids"].append(fact_id)
        block["source_fact_ids"].append(fact_id)
        allocations.append({
            "fact_id": fact_id,
            "unit_path": unit_context["path"],
            "learning_block_id": str(block.get("id") or ""),
            "basis": basis,
        })

    # A majority of a complete source section in one semantic block can be a
    # valid allocation, but is a useful architecture-quality warning.  This
    # is deliberately source-relative (not a raw-fact cap) and never changes
    # canonical allocation completeness.
    facts_by_section: dict[str, int] = {}
    targets_by_section: dict[str, set[tuple[str, str]]] = {}
    for item in allocations:
        fact = map_facts.get(item["fact_id"], {})
        section_id = str(fact.get("section_id") or "").strip()
        if not section_id:
            continue
        facts_by_section[section_id] = facts_by_section.get(section_id, 0) + 1
        targets_by_section.setdefault(section_id, set()).add((item["unit_path"], item["learning_block_id"]))
    quality_findings = [
        {
            "code": "INSTRUCTIONAL_SCOPE_COARSE",
            "severity": "warning",
            "path": next(iter(targets))[0],
            "section_id": section_id,
            "allocated_fact_count": fact_count,
        }
        for section_id, fact_count in sorted(facts_by_section.items())
        for targets in [targets_by_section.get(section_id, set())]
        if len(facts_by_section) > 1
        and fact_count * 2 > len(fact_by_id)
        and len(targets) == 1
    ]
    allocation = {
        "version": "source-fact-allocation-v2",
        "authority": "server",
        "architecture_contract_version": 4,
        "required_count": len(fact_by_id),
        "allocated_count": len(allocations),
        "complete": not unallocated,
        "allocations": allocations,
        "unallocated": unallocated,
        "rejection_counts": rejection_counts,
        "quality_findings": quality_findings,
    }
    return {**blueprint, "source_fact_allocation": allocation}


def source_fact_allocation_diagnostics(blueprint: dict[str, Any]) -> dict[str, Any]:
    """Summarise server allocation only; never emit fact IDs or source text."""

    allocation = blueprint.get("source_fact_allocation")
    if not isinstance(allocation, dict):
        return {
            "allocation_available": False,
            "canonical_fact_count": 0,
            "allocated_fact_count": 0,
            "unallocated_fact_count": 0,
            "allocation_complete": False,
            "allocation_basis_counts": {},
        }
    basis_counts: dict[str, int] = {}
    for item in allocation.get("allocations", []):
        if not isinstance(item, dict):
            continue
        basis = str(item.get("basis") or "UNKNOWN")[:64]
        basis_counts[basis] = basis_counts.get(basis, 0) + 1
    unallocated = allocation.get("unallocated") if isinstance(allocation.get("unallocated"), list) else []
    scope_allocation = blueprint.get("source_evidence_scope_allocation")
    scope_diagnostics = {
        "evidence_scope_allocation_available": isinstance(scope_allocation, dict),
        "canonical_evidence_scope_count": int(scope_allocation.get("required_count") or 0) if isinstance(scope_allocation, dict) else 0,
        "allocated_evidence_scope_count": int(scope_allocation.get("allocated_count") or 0) if isinstance(scope_allocation, dict) else 0,
        "evidence_scope_allocation_complete": bool(scope_allocation.get("complete")) if isinstance(scope_allocation, dict) else False,
    }
    return {
        "allocation_available": True,
        "canonical_fact_count": int(allocation.get("required_count") or 0),
        "allocated_fact_count": int(allocation.get("allocated_count") or 0),
        "unallocated_fact_count": len(unallocated),
        "allocation_complete": bool(allocation.get("complete")),
        "allocation_basis_counts": basis_counts,
        "allocation_rejection_counts": {
            str(code)[:96]: int(count)
            for code, count in (allocation.get("rejection_counts") or {}).items()
            if isinstance(code, str) and isinstance(count, int)
        },
        "allocation_quality_codes": sorted({
            str(item.get("code") or "")[:96]
            for item in allocation.get("quality_findings", [])
            if isinstance(item, dict) and str(item.get("code") or "")
        }),
        **scope_diagnostics,
    }
