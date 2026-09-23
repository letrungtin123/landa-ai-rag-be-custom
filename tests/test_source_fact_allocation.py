from __future__ import annotations

import copy
import json
import unittest

from app.main import (
    apply_course_architecture_repair_patches,
    allocate_blueprint_source_fact_ids,
    allocate_source_map_architecture_facts,
    build_course_architecture_repair_prompt,
    deterministic_factless_unit_removal_repair,
    validate_course_architecture_repair_candidate,
    validate_course_architecture_semantic_scope,
    validate_course_architecture_workflow,
)
from app.lesson_author_blueprint import (
    LessonAuthorBlueprintValidationError,
    parse_and_validate_lesson_author_blueprint,
    validate_lesson_author_blueprint,
)
from app.source_map import build_source_map
from app.workflows.contracts import WorkflowFailure, WorkflowGenerationResult, WorkflowValidationResult
from app.workflows.course_architecture import (
    CourseArchitectureWorkflowCallbacks,
    classify_course_repair_targets,
    run_course_architecture_workflow,
)


def _nodes(section_count: int) -> list[dict[str, object]]:
    return [
        {
            "document_id": "doc-1",
            "document_name": "Allocation guide",
            "source_ref": f"src-{index:03d}",
            "title": f"Section {index}",
            "level": 1,
            "order": index,
            "logical_page": index,
        }
        for index in range(1, section_count + 1)
    ]


def _manifest(fact_count: int, section_count: int) -> dict[str, object]:
    return {
        "truncated": False,
        "fact_scope_complete": True,
        "total_fact_count": fact_count,
        "represented_fact_count": fact_count,
        "facts": [
            {
                "fact_id": f"fact-{index:04d}",
                "document_id": "doc-1",
                "source_ref": f"src-{((index - 1) % section_count) + 1:03d}",
                "source_page": ((index - 1) % section_count) + 1,
                "source_chunk": (index - 1) % 36,
                "text": f"Canonical source fact {index}.",
            }
            for index in range(1, fact_count + 1)
        ],
    }


def _manifest_with_section_counts(section_counts: list[int]) -> dict[str, object]:
    facts: list[dict[str, object]] = []
    fact_index = 1
    for section_index, count in enumerate(section_counts, start=1):
        for offset in range(count):
            facts.append({
                "fact_id": f"fact-{fact_index:04d}",
                "document_id": "doc-1",
                "source_ref": f"src-{section_index:03d}",
                "source_page": section_index,
                "source_chunk": offset % 36,
                "text": f"Canonical source fact {fact_index}.",
            })
            fact_index += 1
    return {
        "truncated": False,
        "fact_scope_complete": True,
        "total_fact_count": len(facts),
        "represented_fact_count": len(facts),
        "facts": facts,
    }


def _concepts_by_ref(source_map: dict[str, object]) -> dict[str, str]:
    section_by_id = {
        str(section["id"]): str(section["source_ref"])
        for section in source_map["sections"]  # type: ignore[index]
    }
    result: dict[str, str] = {}
    for concept in source_map["concepts"]:  # type: ignore[index]
        section_ids = concept.get("source_section_ids", [])
        if section_ids:
            result[section_by_id[str(section_ids[0])]] = str(concept["id"])
    return result


def _blueprint_for_refs(
    source_map: dict[str, object],
    refs: list[str],
) -> dict[str, object]:
    concept_by_ref = _concepts_by_ref(source_map)
    chapters: list[dict[str, object]] = []
    for index, source_ref in enumerate(refs, start=1):
        concept_id = concept_by_ref[source_ref]
        chapters.append({
            "title": f"Chapter {index}",
            "objective": f"Explain content from {source_ref}.",
            "learning_objectives": ["lo_1"],
            "concept_ids": [concept_id],
            "source_refs": [source_ref],
            "lessons": [{
                "title": f"Lesson {index}",
                "objective": f"Identify and explain content from {source_ref}.",
                "learning_objectives": [f"Identify the source-grounded concept in {source_ref}."],
                "learning_activities": ["Review the source evidence."],
                "assessment": "Use a source-grounded check.",
                "primary_concept_ids": [concept_id],
                "supporting_concept_ids": [],
                "prerequisite_concept_ids": [],
                "assessment_required": False,
                "assessment_objective_refs": [],
                "source_refs": [source_ref],
                "units": [{
                    "title": f"Unit {index}",
                    "purpose": f"Teach the concept owned by {source_ref}.",
                    "concept_ids": [concept_id],
                    "primary_concept_ids": [concept_id],
                    "learning_objective_refs": ["lo_1"],
                    "source_refs": [source_ref],
                    "learning_blocks": [{
                        "id": f"lb_{index}",
                        "intent": "concept_explanation",
                        "importance": "core",
                        "concept_ids": [concept_id],
                        "primary_concept_ids": [concept_id],
                        "source_refs": [source_ref],
                        "learning_objective_refs": ["lo_1"],
                        "content": {},
                    }],
                }],
            }],
        })
    return {
        "architecture_contract_version": 4,
        "content_contract_version": 1,
        "title": "Allocation guide",
        "summary": "A source-owned blueprint.",
        "target_audience": "Operators",
        "prerequisites": [],
        "learning_outcomes": [
            "Identify source-grounded concepts.",
            "Explain source-grounded requirements.",
            "Apply source-grounded requirements.",
        ],
        "course_outcomes": [
            "Identify source-grounded concepts.",
            "Explain source-grounded requirements.",
            "Apply source-grounded requirements.",
        ],
        "assessment_strategy": "Use source-grounded checks.",
        "assumptions": [],
        "chapters": chapters,
    }


def _append_supporting_unit_without_canonical_facts(
    blueprint: dict[str, object],
    source_map: dict[str, object],
) -> None:
    """Model a valid supporting unit, not a second canonical fact owner."""

    concept_id = _concepts_by_ref(source_map)["src-001"]
    lesson = blueprint["chapters"][0]["lessons"][0]  # type: ignore[index]
    lesson["units"].append({
        "title": "Reinforce the core concept",
        "purpose": "Practice the already-owned concept without taking canonical fact ownership.",
        "concept_ids": [concept_id],
        "primary_concept_ids": [],
        "learning_objective_refs": ["lo_1"],
        "source_refs": ["src-001"],
        "learning_blocks": [{
            "id": "lb_supporting",
            "intent": "practice",
            "importance": "supporting",
            "concept_ids": [concept_id],
            "primary_concept_ids": [],
            "source_refs": ["src-001"],
            "learning_objective_refs": ["lo_1"],
            "content": {},
        }],
    })


def _append_factless_supporting_block(
    blueprint: dict[str, object],
    source_map: dict[str, object],
) -> None:
    """Add valid reinforcement that must not become a canonical fact owner."""

    concept_id = _concepts_by_ref(source_map)["src-001"]
    primary_unit = blueprint["chapters"][0]["lessons"][0]["units"][0]  # type: ignore[index]
    primary_unit["learning_blocks"].append({
        "id": "lb_supporting_block",
        "intent": "practice",
        "importance": "supporting",
        "concept_ids": [concept_id],
        "primary_concept_ids": [],
        "source_refs": ["src-001"],
        "learning_objective_refs": ["lo_1"],
        "content": {},
    })


def _semantic_unit_without_server_facts(unit: dict[str, object]) -> dict[str, object]:
    """Build a provider-safe semantic replacement for a synthetic repair test."""

    def remove_server_fields(value: object) -> object:
        if isinstance(value, dict):
            return {
                str(key): remove_server_fields(child)
                for key, child in value.items()
                if key not in {"source_fact_ids", "covered_source_fact_ids", "source_fact_allocation", "component_plan"}
            }
        if isinstance(value, list):
            return [remove_server_fields(child) for child in value]
        return copy.deepcopy(value)

    result = remove_server_fields(unit)
    assert isinstance(result, dict)
    return result


def _semantic_blueprint_without_server_facts(blueprint: dict[str, object]) -> dict[str, object]:
    """Build a provider-shaped candidate without retaining server allocation."""

    def remove_server_fields(value: object) -> object:
        if isinstance(value, dict):
            return {
                str(key): remove_server_fields(child)
                for key, child in value.items()
                if key not in {"source_fact_ids", "covered_source_fact_ids", "source_fact_allocation", "component_plan"}
            }
        if isinstance(value, list):
            return [remove_server_fields(child) for child in value]
        return copy.deepcopy(value)

    result = remove_server_fields(blueprint)
    assert isinstance(result, dict)
    return result


def _unit_source_fact_ownership_target(path: str) -> dict[str, object]:
    return {
        "scope": "unit",
        "path": path,
        "codes": ["UNIT_SOURCE_FACT_OWNERSHIP_REQUIRED"],
        "allowed_fields": ["primary_concept_ids", "learning_blocks"],
        "diagnostics": [],
        "allowed_operations": ["replace", "remove_unit"],
    }


class SourceFactAllocationTests(unittest.IsolatedAsyncioTestCase):
    def test_v4_architect_and_post_allocation_contracts_accept_factless_supporting_block(self) -> None:
        manifest = _manifest(2, 1)
        source_map = build_source_map(_nodes(1), manifest)
        semantic = _blueprint_for_refs(source_map, ["src-001"])
        _append_factless_supporting_block(semantic, source_map)

        # The Architect owns semantic purpose only, so no canonical fact IDs
        # are present in the provider-shaped v4 contract.
        architect_validated = validate_lesson_author_blueprint(
            semantic,
            require_source_fact_ownership=False,
            forbid_provider_fact_ownership=True,
        )
        self.assertEqual(len(architect_validated["chapters"][0]["lessons"][0]["units"][0]["learning_blocks"]), 2)

        finalized = allocate_blueprint_source_fact_ids(
            allocate_source_map_architecture_facts(semantic, source_map, manifest),
            manifest,
            _nodes(1),
        )
        blocks = finalized["chapters"][0]["lessons"][0]["units"][0]["learning_blocks"]  # type: ignore[index]
        self.assertTrue(blocks[0]["source_fact_ids"])
        self.assertEqual(blocks[1]["source_fact_ids"], [])
        validation = validate_course_architecture_workflow(finalized, source_map, manifest, {"src-001"})
        self.assertFalse([issue for issue in validation.issues if issue["severity"] == "error"])

    def test_v4_schema_failures_include_safe_exact_path_and_constraint(self) -> None:
        manifest = _manifest(2, 1)
        source_map = build_source_map(_nodes(1), manifest)
        base = _blueprint_for_refs(source_map, ["src-001"])
        block = base["chapters"][0]["lessons"][0]["units"][0]["learning_blocks"][0]  # type: ignore[index]
        unit = base["chapters"][0]["lessons"][0]["units"][0]  # type: ignore[index]

        malformed: list[tuple[str, dict[str, object], str, str]] = []
        missing_purpose = copy.deepcopy(base)
        del missing_purpose["chapters"][0]["lessons"][0]["units"][0]["purpose"]  # type: ignore[index]
        malformed.append(("missing", missing_purpose, "chapters[0].lessons[0].units[0].purpose", "TYPE_STRING"))
        wrong_blocks = copy.deepcopy(base)
        wrong_blocks["chapters"][0]["lessons"][0]["units"][0]["learning_blocks"] = {}  # type: ignore[index]
        malformed.append(("wrong_type", wrong_blocks, "chapters[0].lessons[0].units[0].learning_blocks", "ARRAY_LENGTH"))
        invalid_intent = copy.deepcopy(base)
        invalid_intent["chapters"][0]["lessons"][0]["units"][0]["learning_blocks"][0]["intent"] = "unsupported"  # type: ignore[index]
        malformed.append(("invalid_enum", invalid_intent, "chapters[0].lessons[0].units[0].learning_blocks[0].intent", "ENUM_SEMANTIC_LEARNING_BLOCK_INTENT"))
        empty_blocks = copy.deepcopy(base)
        empty_blocks["chapters"][0]["lessons"][0]["units"][0]["learning_blocks"] = []  # type: ignore[index]
        malformed.append(("empty_collection", empty_blocks, "chapters[0].lessons[0].units[0].learning_blocks", "ARRAY_LENGTH"))
        malformed_content = copy.deepcopy(base)
        malformed_content["chapters"][0]["lessons"][0]["units"][0]["learning_blocks"][0]["content"] = []  # type: ignore[index]
        malformed.append(("nested_shape", malformed_content, "chapters[0].lessons[0].units[0].learning_blocks[0].content", "TYPE_OBJECT"))

        for name, candidate, expected_path, expected_constraint in malformed:
            with self.subTest(name=name), self.assertRaises(LessonAuthorBlueprintValidationError) as raised:
                validate_lesson_author_blueprint(
                    candidate,
                    require_source_fact_ownership=False,
                    forbid_provider_fact_ownership=True,
                )
            diagnostic = raised.exception.safe_diagnostic()
            self.assertEqual(diagnostic["path"], expected_path)
            self.assertEqual(diagnostic["constraint"], expected_constraint)
            self.assertNotIn("Canonical source fact", str(diagnostic))

        # Keep static analyzers honest about the fixture shape without
        # recording the values in test diagnostics.
        self.assertEqual(block["intent"], "concept_explanation")
        self.assertIn("purpose", unit)

    def test_v4_objective_references_must_be_exact_local_ids(self) -> None:
        manifest = _manifest(2, 1)
        source_map = build_source_map(_nodes(1), manifest)
        candidate = _blueprint_for_refs(source_map, ["src-001"])
        unit = candidate["chapters"][0]["lessons"][0]["units"][0]  # type: ignore[index]
        unit["learning_objective_refs"] = ["Explain the source-grounded concept in src-001."]
        with self.assertRaises(LessonAuthorBlueprintValidationError) as raised:
            validate_lesson_author_blueprint(
                candidate,
                require_source_fact_ownership=False,
                forbid_provider_fact_ownership=True,
            )
        self.assertEqual(raised.exception.code, "BLUEPRINT_INVALID_SCHEMA")
        self.assertEqual(raised.exception.constraint, "LOCAL_LEARNING_OBJECTIVE_REFERENCE")

    def test_post_allocation_schema_finding_is_scoped_and_repair_diagnostic_is_safe(self) -> None:
        manifest = _manifest(2, 1)
        source_map = build_source_map(_nodes(1), manifest)
        finalized = allocate_blueprint_source_fact_ids(
            allocate_source_map_architecture_facts(_blueprint_for_refs(source_map, ["src-001"]), source_map, manifest),
            manifest,
            _nodes(1),
        )
        finalized["chapters"][0]["lessons"][0]["units"][0]["learning_blocks"][0]["content"] = []  # type: ignore[index]

        validation = validate_course_architecture_workflow(finalized, source_map, manifest, {"src-001"})
        self.assertEqual(len(validation.issues), 1)
        issue = validation.issues[0]
        self.assertEqual(issue["code"], "BLUEPRINT_INVALID_SCHEMA")
        self.assertEqual(issue["path"], "chapter_1.lesson_1.unit_1")
        self.assertEqual(issue["validator"], "lesson_author_blueprint.validate_lesson_author_blueprint")
        self.assertEqual(issue["schema_error_code"], "BLUEPRINT_INVALID_SCHEMA")
        self.assertEqual(issue["constraint"], "TYPE_OBJECT")
        self.assertEqual(issue["expected_type"], "object")
        self.assertEqual(issue["actual_type"], "array[length=0]")
        targets = classify_course_repair_targets(validation.issues)
        self.assertEqual(targets[0]["path"], "chapter_1.lesson_1.unit_1")
        self.assertEqual(targets[0]["diagnostics"], [{
            "code": "BLUEPRINT_INVALID_SCHEMA",
            "path": "chapter_1.lesson_1.unit_1",
            "validator": "lesson_author_blueprint.validate_lesson_author_blueprint",
            "schema_error_code": "BLUEPRINT_INVALID_SCHEMA",
            "schema_path": "chapters[0].lessons[0].units[0].learning_blocks[0].content",
            "constraint": "TYPE_OBJECT",
            "expected_type": "object",
            "actual_type": "array[length=0]",
        }])

    def test_scoped_schema_repair_revalidates_after_server_reallocation(self) -> None:
        manifest = _manifest(2, 1)
        source_map = build_source_map(_nodes(1), manifest)
        valid = allocate_blueprint_source_fact_ids(
            allocate_source_map_architecture_facts(_blueprint_for_refs(source_map, ["src-001"]), source_map, manifest),
            manifest,
            _nodes(1),
        )
        broken = copy.deepcopy(valid)
        broken["chapters"][0]["lessons"][0]["units"][0]["learning_blocks"][0]["content"] = []  # type: ignore[index]
        targets = classify_course_repair_targets(
            validate_course_architecture_workflow(broken, source_map, manifest, {"src-001"}).issues,
        )
        semantic_unit = _semantic_unit_without_server_facts(
            valid["chapters"][0]["lessons"][0]["units"][0],  # type: ignore[index]
        )
        repaired = apply_course_architecture_repair_patches(broken, targets, {
            "patches": [{
                "path": "chapter_1.lesson_1.unit_1",
                "replacement": {"learning_blocks": semantic_unit["learning_blocks"]},
            }],
        })
        reallocated = allocate_blueprint_source_fact_ids(
            allocate_source_map_architecture_facts(repaired, source_map, manifest),
            manifest,
            _nodes(1),
        )
        validation = validate_course_architecture_workflow(reallocated, source_map, manifest, {"src-001"})
        self.assertFalse([issue for issue in validation.issues if issue["severity"] == "error"])
        self.assertEqual(reallocated["source_fact_allocation"]["allocated_count"], 2)  # type: ignore[index]

    def test_section_a_fact_never_moves_to_unrelated_section_b_to_fill_coverage(self) -> None:
        manifest = _manifest(2, 2)
        source_map = build_source_map(_nodes(2), manifest)
        # Reverse architecture order to prove allocation does not follow input position.
        allocated = allocate_source_map_architecture_facts(
            _blueprint_for_refs(source_map, ["src-002", "src-001"]), source_map, manifest,
        )

        allocations = {item["fact_id"]: item for item in allocated["source_fact_allocation"]["allocations"]}  # type: ignore[index]
        self.assertEqual(allocations["fact-0001"]["unit_path"], "chapter_2.lesson_1.unit_1")
        self.assertEqual(allocations["fact-0002"]["unit_path"], "chapter_1.lesson_1.unit_1")
        self.assertEqual(allocations["fact-0001"]["basis"], "OWNERSHIP_MATCH")

    def test_concept_ownership_wins_over_arbitrary_positional_assignment(self) -> None:
        manifest = _manifest(4, 2)
        source_map = build_source_map(_nodes(2), manifest)
        blueprint = _blueprint_for_refs(source_map, ["src-002", "src-001"])
        # Remove explicit refs: the only deterministic evidence is concept
        # ownership, so the reverse architecture order cannot affect it.
        for chapter in blueprint["chapters"]:  # type: ignore[index]
            chapter["source_refs"] = []
            lesson = chapter["lessons"][0]
            lesson["source_refs"] = []
            unit = lesson["units"][0]
            unit["source_refs"] = []
            unit["learning_blocks"][0]["source_refs"] = []
        allocated = allocate_source_map_architecture_facts(
            blueprint, source_map, manifest,
        )

        allocations = allocated["source_fact_allocation"]["allocations"]  # type: ignore[index]
        section_a_paths = {item["unit_path"] for item in allocations if item["fact_id"] in {"fact-0001", "fact-0003"}}
        section_b_paths = {item["unit_path"] for item in allocations if item["fact_id"] in {"fact-0002", "fact-0004"}}
        self.assertEqual(section_a_paths, {"chapter_2.lesson_1.unit_1"})
        self.assertEqual(section_b_paths, {"chapter_1.lesson_1.unit_1"})
        self.assertTrue(all(item["basis"] == "OWNERSHIP_MATCH" for item in allocations))

    def test_uat_scale_371_facts_are_allocated_with_provenance_when_matching_architecture_exists(self) -> None:
        # This mirrors the production regression shape: six canonical source
        # sections, with the largest section owning 233 of 371 facts.
        manifest = _manifest_with_section_counts([25, 10, 233, 23, 31, 49])
        source_map = build_source_map(_nodes(6), manifest)
        semantic = validate_course_architecture_semantic_scope(
            _blueprint_for_refs(source_map, [f"src-{index:03d}" for index in range(1, 7)]),
            source_map,
        )
        self.assertFalse(semantic.errors)
        self.assertEqual(semantic.metrics["covered_section_count"], 6)
        self.assertEqual(semantic.metrics["covered_concept_count"], 6)
        allocated = allocate_source_map_architecture_facts(
            _blueprint_for_refs(source_map, [f"src-{index:03d}" for index in range(1, 7)]), source_map, manifest,
        )

        result = allocated["source_fact_allocation"]  # type: ignore[index]
        self.assertTrue(result["complete"])
        self.assertEqual(result["required_count"], 371)
        self.assertEqual(result["allocated_count"], 371)
        self.assertEqual(result["version"], "source-fact-allocation-v2")
        self.assertEqual(result["authority"], "server")
        self.assertEqual({item["fact_id"] for item in result["allocations"]}, {fact["fact_id"] for fact in manifest["facts"]})
        self.assertTrue(all(item["basis"] in {"CONCEPT_MATCH", "SOURCE_REF_MATCH", "OWNERSHIP_MATCH", "SECTION_MATCH"} for item in result["allocations"]))
        finalized = allocate_blueprint_source_fact_ids(allocated, manifest, _nodes(6))
        validation = validate_course_architecture_workflow(
            finalized,
            source_map,
            manifest,
            {f"src-{index:03d}" for index in range(1, 7)},
        )
        self.assertFalse([issue for issue in validation.issues if issue["severity"] == "error"])
        self.assertIn("INSTRUCTIONAL_SCOPE_COARSE", [issue["code"] for issue in validation.issues])

    def test_complete_v4_allocation_accepts_proven_factless_supporting_unit(self) -> None:
        # The allocator remains globally complete without inventing an
        # unrelated assignment for a supporting/reinforcement unit. The
        # persisted v4 contract keeps its semantic block without fabricating
        # component provenance.
        manifest = _manifest_with_section_counts([25, 10, 233, 23, 31, 49])
        source_map = build_source_map(_nodes(6), manifest)
        blueprint = _blueprint_for_refs(source_map, [f"src-{index:03d}" for index in range(1, 7)])
        _append_supporting_unit_without_canonical_facts(blueprint, source_map)

        semantic = validate_course_architecture_semantic_scope(blueprint, source_map)
        self.assertFalse(semantic.errors)
        allocated = allocate_source_map_architecture_facts(blueprint, source_map, manifest)
        allocation = allocated["source_fact_allocation"]  # type: ignore[index]
        self.assertTrue(allocation["complete"])
        self.assertEqual(allocation["allocated_count"], 371)
        self.assertEqual(allocation["required_count"], 371)
        self.assertEqual(allocation["unallocated"], [])
        self.assertIn("INSTRUCTIONAL_SCOPE_COARSE", [item["code"] for item in allocation["quality_findings"]])

        finalized = allocate_blueprint_source_fact_ids(allocated, manifest, _nodes(6))
        supporting_unit = finalized["chapters"][0]["lessons"][0]["units"][1]  # type: ignore[index]
        self.assertEqual(supporting_unit["source_fact_ids"], [])
        self.assertEqual(supporting_unit["component_plan"], [])

        validation = validate_course_architecture_workflow(
            finalized,
            source_map,
            manifest,
            {f"src-{index:03d}" for index in range(1, 7)},
        )
        errors = [item for item in validation.issues if item["severity"] == "error"]
        self.assertEqual(errors, [])
        self.assertNotIn("UNIT_SOURCE_FACT_OWNERSHIP_REQUIRED", [item["code"] for item in validation.issues])
        self.assertNotIn("LESSON_SOURCE_FACT_MISSING", [item["code"] for item in validation.issues])

    def test_factless_primary_block_fails_closed(self) -> None:
        manifest = _manifest(2, 1)
        source_map = build_source_map(_nodes(1), manifest)
        blueprint = _blueprint_for_refs(source_map, ["src-001"])
        primary_block = blueprint["chapters"][0]["lessons"][0]["units"][0]["learning_blocks"][0]  # type: ignore[index]
        primary_block["primary_concept_ids"] = [primary_block["concept_ids"][0]]
        finalized = allocate_blueprint_source_fact_ids(
            allocate_source_map_architecture_facts(blueprint, source_map, manifest),
            manifest,
            _nodes(1),
        )
        finalized["chapters"][0]["lessons"][0]["units"][0]["source_fact_ids"] = []  # type: ignore[index]
        finalized["chapters"][0]["lessons"][0]["units"][0]["learning_blocks"][0]["source_fact_ids"] = []  # type: ignore[index]
        with self.assertRaises(LessonAuthorBlueprintValidationError) as raised:
            validate_lesson_author_blueprint(finalized)
        self.assertEqual(raised.exception.code, "BLUEPRINT_PRIMARY_BLOCK_SOURCE_FACT_OWNERSHIP_REQUIRED")

    def test_valid_supporting_unit_does_not_create_a_repair_target(self) -> None:
        manifest = _manifest_with_section_counts([25, 10, 233, 23, 31, 49])
        source_map = build_source_map(_nodes(6), manifest)
        known_source_refs = {f"src-{index:03d}" for index in range(1, 7)}
        blueprint = _blueprint_for_refs(source_map, sorted(known_source_refs))
        _append_supporting_unit_without_canonical_facts(blueprint, source_map)
        finalized = allocate_blueprint_source_fact_ids(
            allocate_source_map_architecture_facts(blueprint, source_map, manifest),
            manifest,
            _nodes(6),
        )
        result = validate_course_architecture_workflow(finalized, source_map, manifest, known_source_refs)
        self.assertFalse([item for item in result.issues if item["severity"] == "error"])
        self.assertEqual(classify_course_repair_targets(result.issues), [])

    def test_valid_factless_supporting_unit_keeps_371_allocated_facts_without_removal(self) -> None:
        manifest = _manifest_with_section_counts([25, 10, 233, 23, 31, 49])
        source_map = build_source_map(_nodes(6), manifest)
        known_source_refs = {f"src-{index:03d}" for index in range(1, 7)}
        blueprint = _blueprint_for_refs(source_map, sorted(known_source_refs))
        _append_supporting_unit_without_canonical_facts(blueprint, source_map)
        baseline = allocate_blueprint_source_fact_ids(
            allocate_source_map_architecture_facts(blueprint, source_map, manifest), manifest, _nodes(6),
        )
        self.assertEqual(len(baseline["chapters"][0]["lessons"][0]["units"]), 2)  # type: ignore[index]
        self.assertTrue(baseline["source_fact_allocation"]["complete"])  # type: ignore[index]
        self.assertEqual(baseline["source_fact_allocation"]["allocated_count"], 371)  # type: ignore[index]
        result = validate_course_architecture_workflow(baseline, source_map, manifest, known_source_refs)
        self.assertFalse([issue for issue in result.issues if issue["severity"] == "error"])

    def test_multiple_valid_supporting_units_do_not_trigger_legacy_removal(self) -> None:
        manifest = _manifest(2, 1)
        source_map = build_source_map(_nodes(1), manifest)
        blueprint = _blueprint_for_refs(source_map, ["src-001"])
        _append_supporting_unit_without_canonical_facts(blueprint, source_map)
        second_supporting = copy.deepcopy(blueprint["chapters"][0]["lessons"][0]["units"][1])  # type: ignore[index]
        second_supporting["title"] = "Reinforce the core concept again"
        second_supporting["learning_blocks"][0]["id"] = "lb_supporting_2"
        blueprint["chapters"][0]["lessons"][0]["units"].append(second_supporting)  # type: ignore[index]
        baseline = allocate_blueprint_source_fact_ids(
            allocate_source_map_architecture_facts(blueprint, source_map, manifest), manifest, _nodes(1),
        )
        result = validate_course_architecture_workflow(baseline, source_map, manifest, {"src-001"})
        self.assertFalse([issue for issue in result.issues if issue["severity"] == "error"])
        self.assertEqual(classify_course_repair_targets(result.issues), [])
        self.assertEqual(len(baseline["chapters"][0]["lessons"][0]["units"]), 3)  # type: ignore[index]

    def test_factless_supporting_unit_does_not_make_a_standalone_lesson_valid(self) -> None:
        manifest = _manifest(2, 1)
        source_map = build_source_map(_nodes(1), manifest)
        blueprint = _blueprint_for_refs(source_map, ["src-001"])
        _append_supporting_unit_without_canonical_facts(blueprint, source_map)
        supporting_unit = copy.deepcopy(blueprint["chapters"][0]["lessons"][0]["units"][1])  # type: ignore[index]
        standalone_lesson = copy.deepcopy(blueprint["chapters"][0]["lessons"][0])  # type: ignore[index]
        standalone_lesson["title"] = "Invalid standalone reinforcement"
        standalone_lesson["primary_concept_ids"] = []
        standalone_lesson["units"] = [supporting_unit]
        blueprint["chapters"][0]["lessons"].append(standalone_lesson)  # type: ignore[index]
        baseline = allocate_blueprint_source_fact_ids(
            allocate_source_map_architecture_facts(blueprint, source_map, manifest), manifest, _nodes(1),
        )
        result = validate_course_architecture_workflow(baseline, source_map, manifest, {"src-001"})
        # v4 permits a supporting *unit* with no canonical facts only within
        # an otherwise instructional lesson. It does not waive the existing
        # lesson-level structural requirement for local objectives.
        self.assertIn("BLUEPRINT_INVALID_SCHEMA", [issue["code"] for issue in result.errors])
        self.assertEqual(len(baseline["chapters"][0]["lessons"]), 2)  # type: ignore[index]

    def test_lesson_primary_without_descendant_primary_block_fails_preallocation(self) -> None:
        manifest = _manifest(2, 1)
        source_map = build_source_map(_nodes(1), manifest)
        blueprint = _blueprint_for_refs(source_map, ["src-001"])
        unit = blueprint["chapters"][0]["lessons"][0]["units"][0]  # type: ignore[index]
        unit["primary_concept_ids"] = []
        unit["learning_blocks"][0]["primary_concept_ids"] = []

        validation = validate_course_architecture_semantic_scope(blueprint, source_map)

        self.assertIn("LESSON_PRIMARY_OWNERSHIP_WITHOUT_PRIMARY_UNIT", [issue["code"] for issue in validation.errors])
        allocated = allocate_source_map_architecture_facts(blueprint, source_map, manifest)
        self.assertIn("LESSON_PRIMARY_OWNERSHIP_WITHOUT_PRIMARY_UNIT", {
            issue["code"] for issue in allocated["source_fact_allocation"]["preallocation_validation"]  # type: ignore[index]
        })

    def test_empty_parent_with_lesson_primary_scope_fails_closed_without_removal(self) -> None:
        manifest = _manifest(2, 1)
        source_map = build_source_map(_nodes(1), manifest)
        blueprint = _blueprint_for_refs(source_map, ["src-001"])
        _append_supporting_unit_without_canonical_facts(blueprint, source_map)
        baseline = allocate_blueprint_source_fact_ids(
            allocate_source_map_architecture_facts(blueprint, source_map, manifest), manifest, _nodes(1),
        )
        supporting_unit = copy.deepcopy(baseline["chapters"][0]["lessons"][0]["units"][1])  # type: ignore[index]
        standalone_lesson = copy.deepcopy(baseline["chapters"][0]["lessons"][0])  # type: ignore[index]
        standalone_lesson["title"] = "Unresolved primary lesson"
        standalone_lesson["units"] = [supporting_unit]
        baseline["chapters"][0]["lessons"].append(standalone_lesson)  # type: ignore[index]

        with self.assertRaises(WorkflowFailure) as raised:
            deterministic_factless_unit_removal_repair(
                baseline,
                [_unit_source_fact_ownership_target("chapter_1.lesson_2.unit_1")],
                source_map=source_map,
                source_coverage_manifest=manifest,
                known_source_refs={"src-001"},
                source_structure_nodes=_nodes(1),
            )

        self.assertEqual(raised.exception.code, "ARCHITECTURE_LESSON_PRIMARY_SCOPE_UNRESOLVED")
        self.assertEqual(raised.exception.internal_code, "ARCH_REPAIR_LESSON_PRIMARY_SCOPE_UNRESOLVED")

    def test_factless_unit_with_primary_ownership_is_not_auto_removed(self) -> None:
        manifest = _manifest(2, 1)
        source_map = build_source_map(_nodes(1), manifest)
        blueprint = _blueprint_for_refs(source_map, ["src-001"])
        _append_supporting_unit_without_canonical_facts(blueprint, source_map)
        baseline = allocate_blueprint_source_fact_ids(
            allocate_source_map_architecture_facts(blueprint, source_map, manifest), manifest, _nodes(1),
        )
        concept_id = _concepts_by_ref(source_map)["src-001"]
        baseline["chapters"][0]["lessons"][0]["units"][1]["primary_concept_ids"] = [concept_id]  # type: ignore[index]

        self.assertIsNone(deterministic_factless_unit_removal_repair(
            baseline,
            [_unit_source_fact_ownership_target("chapter_1.lesson_1.unit_2")],
            source_map=source_map,
            source_coverage_manifest=manifest,
            known_source_refs={"src-001"},
            source_structure_nodes=_nodes(1),
        ))

    def test_factless_unit_with_primary_learning_block_is_not_auto_removed(self) -> None:
        manifest = _manifest(2, 1)
        source_map = build_source_map(_nodes(1), manifest)
        blueprint = _blueprint_for_refs(source_map, ["src-001"])
        _append_supporting_unit_without_canonical_facts(blueprint, source_map)
        baseline = allocate_blueprint_source_fact_ids(
            allocate_source_map_architecture_facts(blueprint, source_map, manifest), manifest, _nodes(1),
        )
        concept_id = _concepts_by_ref(source_map)["src-001"]
        baseline["chapters"][0]["lessons"][0]["units"][1]["learning_blocks"][0]["primary_concept_ids"] = [concept_id]  # type: ignore[index]

        self.assertIsNone(deterministic_factless_unit_removal_repair(
            baseline,
            [_unit_source_fact_ownership_target("chapter_1.lesson_1.unit_2")],
            source_map=source_map,
            source_coverage_manifest=manifest,
            known_source_refs={"src-001"},
            source_structure_nodes=_nodes(1),
        ))

    def test_destructive_candidate_that_regresses_371_allocations_fails_closed_and_baseline_is_unchanged(self) -> None:
        manifest = _manifest_with_section_counts([25, 10, 233, 23, 31, 49])
        source_map = build_source_map(_nodes(6), manifest)
        known_source_refs = {f"src-{index:03d}" for index in range(1, 7)}
        blueprint = _blueprint_for_refs(source_map, sorted(known_source_refs))
        _append_supporting_unit_without_canonical_facts(blueprint, source_map)
        baseline = allocate_blueprint_source_fact_ids(
            allocate_source_map_architecture_facts(blueprint, source_map, manifest), manifest, _nodes(6),
        )
        candidate = _semantic_blueprint_without_server_facts(baseline)
        valid_unit = candidate["chapters"][0]["lessons"][0]["units"][0]  # type: ignore[index]
        valid_unit["primary_concept_ids"] = []
        valid_unit["learning_blocks"][0]["primary_concept_ids"] = []
        candidate = allocate_source_map_architecture_facts(candidate, source_map, manifest)

        self.assertFalse(candidate["source_fact_allocation"]["complete"])  # type: ignore[index]
        self.assertIn("ARCHITECTURE_SCOPE_INCOMPLETE", {
            issue["code"] for issue in candidate["source_fact_allocation"]["preallocation_validation"]  # type: ignore[index]
        })
        self.assertTrue(baseline["source_fact_allocation"]["complete"])  # type: ignore[index]
        self.assertEqual(baseline["source_fact_allocation"]["allocated_count"], 371)  # type: ignore[index]

    def test_patch_for_unrelated_valid_unit_is_rejected_at_mutation_guard(self) -> None:
        manifest = _manifest(2, 1)
        source_map = build_source_map(_nodes(1), manifest)
        blueprint = _blueprint_for_refs(source_map, ["src-001"])
        _append_supporting_unit_without_canonical_facts(blueprint, source_map)
        baseline = allocate_blueprint_source_fact_ids(
            allocate_source_map_architecture_facts(blueprint, source_map, manifest), manifest, _nodes(1),
        )
        targets = classify_course_repair_targets(
            validate_course_architecture_workflow(baseline, source_map, manifest, {"src-001"}).issues,
        )

        with self.assertRaises(WorkflowFailure) as raised:
            apply_course_architecture_repair_patches(baseline, targets, {
                "patches": [{
                    "path": "chapter_1.lesson_1.unit_1",
                    "replacement": {"primary_concept_ids": []},
                }],
            })
        self.assertEqual(raised.exception.code, "ARCHITECTURE_REPAIR_SCOPE_VIOLATION")
        self.assertEqual(raised.exception.internal_code, "ARCH_REPAIR_TARGET_OUT_OF_SCOPE")

    def test_remove_unit_cannot_delete_a_fact_bearing_or_primary_owner_unit(self) -> None:
        manifest = _manifest(2, 1)
        source_map = build_source_map(_nodes(1), manifest)
        blueprint = _blueprint_for_refs(source_map, ["src-001"])
        baseline = allocate_blueprint_source_fact_ids(
            allocate_source_map_architecture_facts(blueprint, source_map, manifest), manifest, _nodes(1),
        )
        target = [{
            "scope": "unit",
            "path": "chapter_1.lesson_1.unit_1",
            "codes": ["UNIT_SOURCE_FACT_OWNERSHIP_REQUIRED"],
            "allowed_fields": ["primary_concept_ids", "learning_blocks"],
            "diagnostics": [],
            "allowed_operations": ["replace", "remove_unit"],
        }]

        with self.assertRaises(WorkflowFailure) as raised:
            apply_course_architecture_repair_patches(baseline, target, {
                "patches": [{"path": "chapter_1.lesson_1.unit_1", "operation": "remove_unit"}],
            })
        self.assertEqual(raised.exception.code, "ARCHITECTURE_REPAIR_SCOPE_VIOLATION")

    def test_targeted_candidate_that_creates_ambiguous_primary_ownership_fails_preallocation(self) -> None:
        manifest = _manifest(2, 1)
        source_map = build_source_map(_nodes(1), manifest)
        blueprint = _blueprint_for_refs(source_map, ["src-001"])
        _append_supporting_unit_without_canonical_facts(blueprint, source_map)
        baseline = allocate_blueprint_source_fact_ids(
            allocate_source_map_architecture_facts(blueprint, source_map, manifest), manifest, _nodes(1),
        )
        concept_id = _concepts_by_ref(source_map)["src-001"]
        candidate = _semantic_blueprint_without_server_facts(baseline)
        target_unit = candidate["chapters"][0]["lessons"][0]["units"][1]  # type: ignore[index]
        target_unit["primary_concept_ids"] = [concept_id]
        target_unit["learning_blocks"][0]["primary_concept_ids"] = [concept_id]
        candidate = allocate_source_map_architecture_facts(candidate, source_map, manifest)

        self.assertFalse(candidate["source_fact_allocation"]["complete"])  # type: ignore[index]
        self.assertIn("AMBIGUOUS_INSTRUCTIONAL_SCOPE", {
            issue["code"] for issue in candidate["source_fact_allocation"]["preallocation_validation"]  # type: ignore[index]
        })

    def test_multiple_supporting_units_preserve_complete_allocation_without_repair_targets(self) -> None:
        manifest = _manifest(2, 1)
        source_map = build_source_map(_nodes(1), manifest)
        blueprint = _blueprint_for_refs(source_map, ["src-001"])
        _append_supporting_unit_without_canonical_facts(blueprint, source_map)
        second_supporting = copy.deepcopy(blueprint["chapters"][0]["lessons"][0]["units"][1])  # type: ignore[index]
        second_supporting["title"] = "Reinforce the core concept again"
        second_supporting["learning_blocks"][0]["id"] = "lb_supporting_2"
        blueprint["chapters"][0]["lessons"][0]["units"].append(second_supporting)  # type: ignore[index]
        baseline = allocate_blueprint_source_fact_ids(
            allocate_source_map_architecture_facts(blueprint, source_map, manifest), manifest, _nodes(1),
        )
        targets = classify_course_repair_targets(
            validate_course_architecture_workflow(baseline, source_map, manifest, {"src-001"}).issues,
        )
        self.assertEqual(targets, [])
        self.assertEqual(baseline["source_fact_allocation"]["allocated_count"], 2)  # type: ignore[index]

    async def test_valid_factless_supporting_unit_bypasses_repair(self) -> None:
        manifest = _manifest(2, 1)
        source_map = build_source_map(_nodes(1), manifest)
        blueprint = _blueprint_for_refs(source_map, ["src-001"])
        _append_supporting_unit_without_canonical_facts(blueprint, source_map)
        finalized = allocate_blueprint_source_fact_ids(
            allocate_source_map_architecture_facts(blueprint, source_map, manifest),
            manifest,
            _nodes(1),
        )

        async def architect(_source_map: dict[str, object]) -> WorkflowGenerationResult:
            return WorkflowGenerationResult(copy.deepcopy(finalized))

        async def repair(
            _current: dict[str, object],
            _targets: list[dict[str, object]],
            _source_map: dict[str, object],
        ) -> WorkflowGenerationResult:
            raise AssertionError("A valid supporting unit must not enter repair.")

        callbacks = CourseArchitectureWorkflowCallbacks(
            validate_source_scope=lambda: [],
            build_source_map=lambda: source_map,
            architect=architect,
            validate_blueprint=lambda candidate, _source_map: validate_course_architecture_workflow(
                candidate, source_map, manifest, {"src-001"},
            ),
            repair_blueprint=repair,  # type: ignore[arg-type]
        )
        _blueprint, workflow = await run_course_architecture_workflow(
            callbacks,
            request_context={"correlation_id": "f7e48b51-01d1-4fd7-a578-5d7e749f70e9"},
            max_repair_attempts=2,
        )
        self.assertEqual(workflow["status"], "ready")
        self.assertEqual(workflow["repair_count"], 0)

    def test_genuinely_unallocated_fact_still_fails_closed(self) -> None:
        manifest = _manifest(2, 2)
        source_map = build_source_map(_nodes(2), manifest)
        allocated = allocate_source_map_architecture_facts(
            _blueprint_for_refs(source_map, ["src-001"]), source_map, manifest,
        )

        allocation = allocated["source_fact_allocation"]  # type: ignore[index]
        self.assertFalse(allocation["complete"])
        validation = validate_course_architecture_workflow(allocated, source_map, manifest, {"src-001", "src-002"})
        self.assertIn("ARCHITECTURE_SCOPE_INCOMPLETE", [issue["code"] for issue in validation.issues])

    def test_capacity_exceeded_remains_fail_closed_in_workflow_validation(self) -> None:
        manifest = _manifest(513, 1)
        source_map = build_source_map(_nodes(1), manifest)
        allocated = allocate_source_map_architecture_facts(
            _blueprint_for_refs(source_map, ["src-001"]), source_map, manifest,
        )

        allocation = allocated["source_fact_allocation"]  # type: ignore[index]
        self.assertFalse(allocation["complete"])
        validation = validate_course_architecture_workflow(allocated, source_map, manifest, {"src-001"})
        self.assertIn("ARCHITECTURE_FACT_CAPACITY_EXCEEDED", [issue["code"] for issue in validation.issues])

    def test_server_owned_fact_serialization_capacity_accepts_239_240_and_241_facts(self) -> None:
        for count in (239, 240, 241):
            with self.subTest(count=count):
                manifest = _manifest(count, 1)
                source_map = build_source_map(_nodes(1), manifest)
                allocated = allocate_source_map_architecture_facts(
                    _blueprint_for_refs(source_map, ["src-001"]), source_map, manifest,
                )
                result = allocated["source_fact_allocation"]  # type: ignore[index]
                self.assertTrue(result["complete"])
                self.assertEqual(result["allocated_count"], count)
                unit = allocated["chapters"][0]["lessons"][0]["units"][0]  # type: ignore[index]
                self.assertEqual(len(unit["source_fact_ids"]), count)

    def test_server_owned_fact_serialization_capacity_fails_explicitly_above_512(self) -> None:
        manifest = _manifest(513, 1)
        source_map = build_source_map(_nodes(1), manifest)
        allocated = allocate_source_map_architecture_facts(
            _blueprint_for_refs(source_map, ["src-001"]), source_map, manifest,
        )
        result = allocated["source_fact_allocation"]  # type: ignore[index]
        self.assertFalse(result["complete"])
        self.assertEqual(result["allocated_count"], 512)
        self.assertEqual(result["unallocated"], [{
            "fact_id": "fact-0513",
            "code": "ARCHITECTURE_FACT_CAPACITY_EXCEEDED",
            "path": "chapter_1.lesson_1.unit_1",
        }])

    def test_invalid_semantic_scope_fails_before_fact_level_allocation(self) -> None:
        manifest = _manifest(2, 2)
        source_map = build_source_map(_nodes(2), manifest)
        candidate = _blueprint_for_refs(source_map, ["src-001", "src-002"])
        first_unit = candidate["chapters"][0]["lessons"][0]["units"][0]  # type: ignore[index]
        first_block = first_unit["learning_blocks"][0]
        first_unit["concept_ids"] = ["concept-does-not-exist"]
        first_block["source_refs"] = ["src-404"]
        findings = validate_course_architecture_semantic_scope(candidate, source_map)
        self.assertIn("UNKNOWN_CONCEPT_ID", [item["code"] for item in findings.issues])
        self.assertIn("UNKNOWN_SOURCE_REF", [item["code"] for item in findings.issues])
        allocated = allocate_source_map_architecture_facts(candidate, source_map, manifest)
        result = allocated["source_fact_allocation"]  # type: ignore[index]
        self.assertEqual(result["allocations"], [])
        self.assertEqual(result["unallocated"], [])
        self.assertTrue(result["preallocation_validation"])

    def test_unmatched_source_fact_is_not_assigned_and_fails_validation(self) -> None:
        manifest = _manifest(2, 2)
        source_map = build_source_map(_nodes(2), manifest)
        allocated = allocate_source_map_architecture_facts(
            _blueprint_for_refs(source_map, ["src-001"]), source_map, manifest,
        )

        result = allocated["source_fact_allocation"]  # type: ignore[index]
        self.assertFalse(result["complete"])
        self.assertEqual(result["unallocated"], [])
        self.assertIn("ARCHITECTURE_SCOPE_INCOMPLETE", {
            item["code"] for item in result["preallocation_validation"]
        })
        validation = validate_course_architecture_workflow(allocated, source_map, manifest, {"src-001", "src-002"})
        self.assertIn("ARCHITECTURE_SCOPE_INCOMPLETE", [issue["code"] for issue in validation.issues])

    def test_1000_plus_fact_allocation_is_bounded_and_stable(self) -> None:
        # One semantic scope per section proves scale without a positional
        # fallback or model-declared fact IDs.
        manifest = _manifest(1_001, 12)
        source_map = build_source_map(_nodes(12), manifest)
        blueprint = _blueprint_for_refs(source_map, [f"src-{section:03d}" for section in range(1, 13)])

        first = allocate_source_map_architecture_facts(copy.deepcopy(blueprint), source_map, manifest)
        second = allocate_source_map_architecture_facts(copy.deepcopy(blueprint), source_map, manifest)
        self.assertTrue(first["source_fact_allocation"]["complete"])
        self.assertEqual(first["source_fact_allocation"], second["source_fact_allocation"])
        self.assertEqual(first["source_fact_allocation"]["allocated_count"], 1_001)

    def test_provider_hallucinated_or_missing_fact_ids_cannot_become_allocation_input(self) -> None:
        manifest = _manifest(2, 2)
        source_map = build_source_map(_nodes(2), manifest)
        candidate = _blueprint_for_refs(source_map, ["src-001", "src-002"])
        first_block = candidate["chapters"][0]["lessons"][0]["units"][0]["learning_blocks"][0]  # type: ignore[index]
        first_block["source_fact_ids"] = ["[fact_x]", "fact-does-not-exist"]
        with self.assertRaises(LessonAuthorBlueprintValidationError) as raised:
            parse_and_validate_lesson_author_blueprint(
                json.dumps(candidate),
                require_source_fact_ownership=False,
                forbid_provider_fact_ownership=True,
            )
        self.assertEqual(raised.exception.code, "BLUEPRINT_PROVIDER_FACT_OWNERSHIP_FORBIDDEN")
        with self.assertRaises(LessonAuthorBlueprintValidationError) as allocator_raised:
            allocate_source_map_architecture_facts(candidate, source_map, manifest)
        self.assertEqual(allocator_raised.exception.code, "BLUEPRINT_PROVIDER_FACT_OWNERSHIP_FORBIDDEN")

        semantic_only = _blueprint_for_refs(source_map, ["src-001", "src-002"])
        parsed = parse_and_validate_lesson_author_blueprint(
            json.dumps(semantic_only),
            require_source_fact_ownership=False,
            forbid_provider_fact_ownership=True,
        )
        self.assertNotIn("source_fact_ids", parsed["chapters"][0]["lessons"][0]["units"][0])
        allocated = allocate_source_map_architecture_facts(parsed, source_map, manifest)
        self.assertTrue(allocated["source_fact_allocation"]["complete"])

    def test_ambiguous_primary_semantic_ownership_fails_closed_before_fact_allocation(self) -> None:
        manifest = _manifest(1, 1)
        source_map = build_source_map(_nodes(1), manifest)
        candidate = _blueprint_for_refs(source_map, ["src-001", "src-001"])
        expected_concept_id = _concepts_by_ref(source_map)["src-001"]
        expected_section_id = next(
            str(section["id"])
            for section in source_map["sections"]  # type: ignore[index]
            if section["source_ref"] == "src-001"
        )
        semantic = validate_course_architecture_semantic_scope(candidate, source_map)
        ambiguous = next(issue for issue in semantic.issues if issue["code"] == "AMBIGUOUS_INSTRUCTIONAL_SCOPE")
        self.assertEqual(ambiguous["concept_id"], expected_concept_id)
        self.assertEqual(ambiguous["section_id"], expected_section_id)
        self.assertEqual(ambiguous["path"], "chapter_1.lesson_1.unit_1.block_1")
        self.assertEqual(ambiguous["related_paths"], [
            "chapter_1.lesson_1.unit_1.block_1",
            "chapter_2.lesson_1.unit_1.block_1",
        ])
        self.assertEqual(ambiguous["expected_primary_owner_count"], 1)
        self.assertEqual(ambiguous["actual_primary_owner_count"], 2)
        self.assertEqual(ambiguous["ownership_level"], "learning_block")
        self.assertEqual(ambiguous["safe_reason"], "MULTIPLE_PRIMARY_DESTINATIONS")
        allocated = allocate_source_map_architecture_facts(candidate, source_map, manifest)
        result = allocated["source_fact_allocation"]  # type: ignore[index]
        self.assertFalse(result["complete"])
        self.assertEqual(result["unallocated"], [])
        persisted = next(item for item in result["preallocation_validation"] if item["code"] == "AMBIGUOUS_INSTRUCTIONAL_SCOPE")
        self.assertEqual(persisted["concept_id"], expected_concept_id)
        self.assertEqual(persisted["related_paths"], ambiguous["related_paths"])

    def test_missing_primary_scope_reports_exact_canonical_concept_and_nearest_block_path(self) -> None:
        manifest = _manifest(2, 2)
        source_map = build_source_map(_nodes(2), manifest)
        candidate = _blueprint_for_refs(source_map, ["src-001", "src-002"])
        missing_lesson = candidate["chapters"][1]["lessons"][0]  # type: ignore[index]
        missing_unit = missing_lesson["units"][0]
        missing_block = missing_unit["learning_blocks"][0]
        concept_id = missing_lesson["primary_concept_ids"][0]
        expected_section_id = next(
            str(section["id"])
            for section in source_map["sections"]  # type: ignore[index]
            if section["source_ref"] == "src-002"
        )
        missing_lesson["primary_concept_ids"] = []
        missing_unit["primary_concept_ids"] = []
        missing_block["primary_concept_ids"] = []

        semantic = validate_course_architecture_semantic_scope(candidate, source_map)
        missing = next(
            issue
            for issue in semantic.issues
            if issue["code"] == "ARCHITECTURE_SCOPE_INCOMPLETE"
            and issue.get("scope_classification") == "concept_primary_ownership"
        )
        self.assertEqual(missing["concept_id"], concept_id)
        self.assertEqual(missing["section_id"], expected_section_id)
        self.assertEqual(missing["path"], "chapter_2.lesson_1.unit_1.block_1")
        self.assertEqual(missing["expected_primary_owner_count"], 1)
        self.assertEqual(missing["actual_primary_owner_count"], 0)
        self.assertEqual(missing["coverage_state"], "missing")
        self.assertEqual(missing["safe_reason"], "NO_PRIMARY_DESTINATION")

        allocated = allocate_source_map_architecture_facts(candidate, source_map, manifest)
        self.assertEqual(allocated["source_fact_allocation"]["allocations"], [])  # type: ignore[index]
        validation = validate_course_architecture_workflow(allocated, source_map, manifest, {"src-001", "src-002"})
        propagated = next(
            issue
            for issue in validation.issues
            if issue["code"] == "ARCHITECTURE_SCOPE_INCOMPLETE"
            and issue.get("scope_classification") == "concept_primary_ownership"
        )
        self.assertEqual(propagated["concept_id"], concept_id)
        self.assertEqual(propagated["path"], "chapter_2.lesson_1.unit_1.block_1")

    def test_scoped_repair_cannot_edit_or_retain_server_owned_fact_ids(self) -> None:
        manifest = _manifest(1, 1)
        source_map = build_source_map(_nodes(1), manifest)
        semantic = _blueprint_for_refs(source_map, ["src-001"])
        finalized = allocate_source_map_architecture_facts(semantic, source_map, manifest)
        target = [{
            "scope": "unit", "path": "chapter_1.lesson_1.unit_1",
            "codes": ["LESSON_TOO_THIN"],
            "allowed_fields": ["purpose", "concept_ids", "source_refs", "learning_objective_refs", "learning_blocks"],
        }]
        with self.assertRaises(WorkflowFailure):
            apply_course_architecture_repair_patches(finalized, target, {
                "patches": [{"path": "chapter_1.lesson_1.unit_1", "replacement": {"source_fact_ids": ["fact-0001"]}}],
            })
        repaired = apply_course_architecture_repair_patches(finalized, target, {
            "patches": [{"path": "chapter_1.lesson_1.unit_1", "replacement": {"purpose": "Refined source-grounded purpose."}}],
        })
        self.assertNotIn("source_fact_allocation", repaired)
        self.assertNotIn("source_fact_ids", repaired["chapters"][0]["lessons"][0]["units"][0])  # type: ignore[index]


class AllocationPropagationWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_371_factless_unit_repair_bypasses_provider(self) -> None:
        manifest = _manifest_with_section_counts([25, 10, 233, 23, 31, 49])
        source_map = build_source_map(_nodes(6), manifest)
        known_source_refs = {f"src-{index:03d}" for index in range(1, 7)}
        blueprint = _blueprint_for_refs(source_map, sorted(known_source_refs))
        _append_supporting_unit_without_canonical_facts(blueprint, source_map)
        baseline = allocate_blueprint_source_fact_ids(
            allocate_source_map_architecture_facts(blueprint, source_map, manifest), manifest, _nodes(6),
        )
        provider_call_count = 0

        async def architect(_source_map: dict[str, object]) -> WorkflowGenerationResult:
            return WorkflowGenerationResult(copy.deepcopy(baseline))

        async def deterministic_repair(
            current: dict[str, object],
            targets: list[dict[str, object]],
            _source_map: dict[str, object],
        ) -> WorkflowGenerationResult | None:
            repaired = deterministic_factless_unit_removal_repair(
                current,
                targets,  # type: ignore[arg-type]
                source_map=source_map,
                source_coverage_manifest=manifest,
                known_source_refs=known_source_refs,
                source_structure_nodes=_nodes(6),
            )
            return WorkflowGenerationResult(repaired) if repaired is not None else None

        async def provider_repair(
            _current: dict[str, object],
            _targets: list[dict[str, object]],
            _source_map: dict[str, object],
        ) -> WorkflowGenerationResult:
            nonlocal provider_call_count
            provider_call_count += 1
            raise AssertionError("deterministic factless-unit repair must not call the provider")

        callbacks = CourseArchitectureWorkflowCallbacks(
            validate_source_scope=lambda: [],
            build_source_map=lambda: source_map,
            architect=architect,
            validate_blueprint=lambda candidate, _source_map: validate_course_architecture_workflow(
                candidate, source_map, manifest, known_source_refs,
            ),
            repair_blueprint=provider_repair,  # type: ignore[arg-type]
            deterministic_repair=deterministic_repair,  # type: ignore[arg-type]
        )
        repaired, workflow = await run_course_architecture_workflow(
            callbacks,
            request_context={"correlation_id": "3e164d54-b892-4b97-a249-93165feb2d6d"},
            max_repair_attempts=2,
        )

        self.assertEqual(provider_call_count, 0)
        self.assertEqual(workflow["status"], "ready")
        self.assertTrue(repaired["source_fact_allocation"]["complete"])
        self.assertEqual(repaired["source_fact_allocation"]["allocated_count"], 371)

    async def test_valid_supporting_unit_bypasses_provider_without_legacy_removal(self) -> None:
        manifest = _manifest(2, 1)
        source_map = build_source_map(_nodes(1), manifest)
        blueprint = _blueprint_for_refs(source_map, ["src-001"])
        _append_supporting_unit_without_canonical_facts(blueprint, source_map)
        baseline = allocate_blueprint_source_fact_ids(
            allocate_source_map_architecture_facts(blueprint, source_map, manifest), manifest, _nodes(1),
        )
        provider_call_count = 0

        async def architect(_source_map: dict[str, object]) -> WorkflowGenerationResult:
            return WorkflowGenerationResult(copy.deepcopy(baseline))

        async def deterministic_repair(
            current: dict[str, object],
            targets: list[dict[str, object]],
            _source_map: dict[str, object],
        ) -> WorkflowGenerationResult | None:
            repaired = deterministic_factless_unit_removal_repair(
                current,
                targets,  # type: ignore[arg-type]
                source_map=source_map,
                source_coverage_manifest=manifest,
                known_source_refs={"src-001"},
                source_structure_nodes=_nodes(1),
            )
            return WorkflowGenerationResult(repaired) if repaired is not None else None

        async def provider_repair(
            _current: dict[str, object],
            _targets: list[dict[str, object]],
            _source_map: dict[str, object],
        ) -> WorkflowGenerationResult:
            nonlocal provider_call_count
            provider_call_count += 1
            raise AssertionError("valid supporting unit must not call the provider")

        callbacks = CourseArchitectureWorkflowCallbacks(
            validate_source_scope=lambda: [],
            build_source_map=lambda: source_map,
            architect=architect,
            validate_blueprint=lambda candidate, _source_map: validate_course_architecture_workflow(
                candidate, source_map, manifest, {"src-001"},
            ),
            repair_blueprint=provider_repair,  # type: ignore[arg-type]
            deterministic_repair=deterministic_repair,  # type: ignore[arg-type]
        )
        repaired, workflow = await run_course_architecture_workflow(
            callbacks,
            request_context={"correlation_id": "f5930c49-8ec1-4a70-9ddd-817c98e32cb5"},
            max_repair_attempts=2,
        )

        self.assertEqual(provider_call_count, 0)
        self.assertEqual(workflow["status"], "ready")
        self.assertEqual(len(repaired["chapters"][0]["lessons"]), 1)
        self.assertEqual(len(repaired["chapters"][0]["lessons"][0]["units"]), 2)
        self.assertTrue(repaired["source_fact_allocation"]["complete"])

    async def test_unchanged_schema_repair_is_reported_as_no_progress(self) -> None:
        manifest = _manifest(2, 1)
        source_map = build_source_map(_nodes(1), manifest)
        broken = allocate_blueprint_source_fact_ids(
            allocate_source_map_architecture_facts(_blueprint_for_refs(source_map, ["src-001"]), source_map, manifest),
            manifest,
            _nodes(1),
        )
        broken["chapters"][0]["lessons"][0]["units"][0]["learning_blocks"][0]["content"] = []  # type: ignore[index]

        async def architect(_source_map: dict[str, object]) -> WorkflowGenerationResult:
            return WorkflowGenerationResult(copy.deepcopy(broken))

        async def repair(
            current: dict[str, object],
            _targets: list[dict[str, object]],
            _source_map: dict[str, object],
        ) -> WorkflowGenerationResult:
            return WorkflowGenerationResult(copy.deepcopy(current))

        callbacks = CourseArchitectureWorkflowCallbacks(
            validate_source_scope=lambda: [],
            build_source_map=lambda: source_map,
            architect=architect,
            validate_blueprint=lambda candidate, _source_map: validate_course_architecture_workflow(
                candidate, source_map, manifest, {"src-001"},
            ),
            repair_blueprint=repair,  # type: ignore[arg-type]
        )
        with self.assertRaises(WorkflowFailure) as raised:
            await run_course_architecture_workflow(
                callbacks,
                request_context={"correlation_id": "74fac563-459e-4ac9-a5f9-2d1a9464f1cc"},
                max_repair_attempts=2,
            )
        self.assertEqual(raised.exception.code, "ARCHITECTURE_REPAIR_NO_PROGRESS")
        self.assertEqual(raised.exception.internal_code, "ARCH_REPAIR_NO_PROGRESS")

    async def test_complete_uat_shaped_allocation_reaches_blueprint_validation(self) -> None:
        manifest = _manifest_with_section_counts([25, 10, 233, 23, 31, 49])
        source_map = build_source_map(_nodes(6), manifest)
        blueprint = _blueprint_for_refs(source_map, [f"src-{index:03d}" for index in range(1, 7)])
        _append_supporting_unit_without_canonical_facts(blueprint, source_map)
        validation_calls: list[dict[str, object]] = []

        async def architect(_source_map: dict[str, object]) -> WorkflowGenerationResult:
            allocated = allocate_source_map_architecture_facts(copy.deepcopy(blueprint), source_map, manifest)
            return WorkflowGenerationResult(allocate_blueprint_source_fact_ids(allocated, manifest, _nodes(6)))

        def validate(candidate: dict[str, object], _source_map: dict[str, object]) -> WorkflowValidationResult:
            validation_calls.append(candidate)
            return WorkflowValidationResult()

        callbacks = CourseArchitectureWorkflowCallbacks(
            validate_source_scope=lambda: [],
            build_source_map=lambda: source_map,
            architect=architect,
            validate_blueprint=validate,
            repair_blueprint=lambda _blueprint, _targets, _source_map: None,  # type: ignore[arg-type]
        )
        _result, diagnostics = await run_course_architecture_workflow(
            callbacks,
            request_context={"correlation_id": "f7e48b51-01d1-4fd7-a578-5d7e749f70e9"},
            max_repair_attempts=2,
        )

        self.assertEqual(len(validation_calls), 1)
        self.assertEqual(diagnostics["status"], "ready")


if __name__ == "__main__":
    unittest.main()
