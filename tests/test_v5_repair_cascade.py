from __future__ import annotations

"""Regression coverage for the V5 fallback/repair cascade UAT failure."""

import copy
import json
import unittest
from unittest.mock import AsyncMock, patch

from google import genai
from google.genai import models, types

from app.lesson_author_blueprint import (
    COURSE_ARCHITECTURE_REPAIR_RESPONSE_SCHEMA,
    LessonAuthorBlueprintValidationError,
    build_v5_semantic_delta_repair_response_schema,
    parse_lesson_author_blueprint_candidate,
    validate_lesson_author_blueprint,
)
from app.main import (
    AiUsage,
    LessonAuthorBlueprintGenerationError,
    _workflow_issue_from_blueprint_validation_error,
    _v5_deterministic_evidence_alignment_candidate,
    _v5_deterministic_assessment_alignment_payload,
    _v5_prepare_evidence_alignment_targets,
    _v5_prepare_assessment_alignment_targets,
    allocate_blueprint_source_fact_ids,
    allocate_source_map_architecture_facts,
    apply_course_architecture_repair_patches,
    assert_v5_immutable_source_context,
    assert_v5_scoped_repair_target_bound,
    build_course_architect_prompt,
    build_v5_scoped_repair_source_context,
    create_v5_immutable_source_context,
    generate_validated_lesson_author_blueprint,
    validate_course_architecture_evidence_scope,
    validate_course_architecture_workflow,
    validate_v5_instructional_coherence,
    validate_v5_post_allocation_instructional_depth,
)
from app.source_map import build_source_map
from app.workflows.contracts import (
    WorkflowFailure,
    WorkflowGenerationResult,
    WorkflowValidationResult,
)
from app.workflows.course_architecture import (
    CourseArchitectureWorkflowCallbacks,
    classify_course_repair_targets,
    run_course_architecture_workflow,
)
from tests.test_evidence_scope_allocation_v5 import _manifest, _nodes, _v5_blueprint
from tests.test_lesson_author_blueprint import blueprint_request


def _node_at(blueprint: dict[str, object], path: str) -> dict[str, object]:
    # Keep the test independent from private production path helpers.
    import re

    match = re.fullmatch(r"chapter_(\d+)\.lesson_(\d+)\.unit_(\d+)", path)
    assert match is not None
    chapter_index, lesson_index, unit_index = (int(value) - 1 for value in match.groups())
    return blueprint["chapters"][chapter_index]["lessons"][lesson_index]["units"][unit_index]  # type: ignore[index,return-value]


class V5RepairCascadeTests(unittest.IsolatedAsyncioTestCase):
    def _context(
        self,
        fact_count: int = 371,
        section_count: int = 6,
        *,
        evidence_scope_count: int | None = None,
    ):
        manifest = _manifest(fact_count, section_count)
        source_map = build_source_map(_nodes(section_count), manifest, locale="en")
        if evidence_scope_count is not None:
            self._split_evidence_scopes_to_count(source_map, evidence_scope_count)
        return create_v5_immutable_source_context(source_map, manifest), source_map, manifest

    def _split_evidence_scopes_to_count(
        self,
        source_map: dict[str, object],
        target_count: int,
    ) -> None:
        """Build a deterministic 17-scope fixture without changing fact ownership."""

        scopes = source_map["source_evidence_scopes"]  # type: ignore[index]
        assert isinstance(scopes, list)
        split_index = 1
        while len(scopes) < target_count:
            candidates = [
                scope for scope in scopes
                if isinstance(scope, dict) and len(scope.get("source_fact_ids") or []) >= 2
            ]
            if not candidates:
                raise AssertionError("fixture cannot create the requested bounded evidence-scope count")
            scope = max(candidates, key=lambda item: len(item["source_fact_ids"]))
            fact_ids = list(scope["source_fact_ids"])
            midpoint = len(fact_ids) // 2
            scope["source_fact_ids"] = fact_ids[:midpoint]
            sibling = copy.deepcopy(scope)
            sibling["id"] = f"{scope['id']}:split:{split_index}"
            sibling["source_fact_ids"] = fact_ids[midpoint:]
            scopes.append(sibling)
            split_index += 1
        coverage = source_map.get("coverage")
        if isinstance(coverage, dict):
            coverage["evidence_scope_count"] = len(scopes)

    def _remove_primary_scope_owners(self, blueprint: dict[str, object], *, count: int) -> list[str]:
        removed: list[str] = []
        for chapter in blueprint["chapters"]:  # type: ignore[index]
            for lesson in chapter["lessons"]:
                for unit in lesson["units"]:
                    for block in unit["learning_blocks"]:
                        primary = block["primary_evidence_scope_ids"]
                        while primary and len(removed) < count:
                            removed.append(primary.pop())
                        if len(removed) == count:
                            return removed
        raise AssertionError("fixture does not contain enough primary V5 scope owners")

    def _replacement_blocks_for_target(
        self,
        blueprint: dict[str, object],
        target: dict[str, object],
        source_map: dict[str, object],
    ) -> list[dict[str, object]]:
        unit = _node_at(blueprint, str(target["path"]))
        blocks = copy.deepcopy(unit["learning_blocks"])
        scopes = {
            str(scope["id"]): scope
            for scope in source_map["source_evidence_scopes"]  # type: ignore[index]
        }
        for scope_id in target.get("allowed_evidence_scope_ids", []):
            scope = scopes[str(scope_id)]
            for block in blocks:
                if set(scope["concept_ids"]).issubset(set(block["concept_ids"])):
                    block["primary_evidence_scope_ids"] = sorted(set(block["primary_evidence_scope_ids"]) | {str(scope_id)})
                    break
            else:
                raise AssertionError("test target did not have a compatible semantic block")
        return blocks

    def _exact_evidence_mismatch_fixture(self) -> tuple[dict[str, object], dict[str, object], dict[str, object], list[dict[str, object]]]:
        """Mirror the UAT shape: an evidence mismatch in chapter_4 lesson_1 unit_2."""

        _context, source_map, manifest = self._context(371, 6, evidence_scope_count=17)
        candidate = _v5_blueprint(source_map)
        lesson = candidate["chapters"][3]["lessons"][0]  # type: ignore[index]
        first_unit = lesson["units"][0]
        second_unit = copy.deepcopy(first_unit)
        first_block = first_unit["learning_blocks"][0]
        second_block = second_unit["learning_blocks"][0]
        # Retain exactly one primary scope owner globally. The first unit is a
        # grounded supporting treatment; only the second target block has a
        # missing concept declaration.
        first_block["primary_evidence_scope_ids"] = []
        first_block["supporting_evidence_scope_ids"] = list(second_block["primary_evidence_scope_ids"])
        second_unit["title"] = "Evidence alignment detail"
        second_block["id"] = "lb_exact_evidence_unit_2"
        second_block["concept_ids"] = []
        lesson["units"].append(second_unit)

        issues = validate_course_architecture_evidence_scope(candidate, source_map).errors
        for issue in issues:
            issue["repair_layer"] = "EVIDENCE_SEMANTIC"
        targets = classify_course_repair_targets(issues, repair_layer="EVIDENCE_SEMANTIC")
        return candidate, source_map, manifest, targets

    def test_exact_uat_evidence_mismatch_repairs_concepts_only_and_allocates_371_facts(self) -> None:
        candidate, source_map, manifest, targets = self._exact_evidence_mismatch_fixture()
        self.assertEqual([target["path"] for target in targets], ["chapter_4.lesson_1.unit_2"])
        self.assertEqual(targets[0]["semantic_operations"], ["align_concepts_to_evidence"])
        prepared = _v5_prepare_evidence_alignment_targets(candidate, targets, source_map)
        self.assertEqual(len(prepared[0]["primary_concept_options"]), 1)
        before_ownership = {
            block["id"]: (
                list(block["primary_evidence_scope_ids"]),
                list(block["supporting_evidence_scope_ids"]),
            )
            for unit in candidate["chapters"][3]["lessons"][0]["units"]  # type: ignore[index]
            for block in unit["learning_blocks"]
        }

        repaired = _v5_deterministic_evidence_alignment_candidate(candidate, prepared)
        self.assertIsNotNone(repaired)
        assert repaired is not None
        self.assertFalse(validate_course_architecture_evidence_scope(repaired, source_map).errors)
        after_ownership = {
            block["id"]: (
                list(block["primary_evidence_scope_ids"]),
                list(block["supporting_evidence_scope_ids"]),
            )
            for unit in repaired["chapters"][3]["lessons"][0]["units"]  # type: ignore[index]
            for block in unit["learning_blocks"]
        }
        self.assertEqual(after_ownership, before_ownership)
        allocated = allocate_source_map_architecture_facts(repaired, source_map, manifest)
        self.assertEqual(allocated["source_fact_allocation"]["allocated_count"], 371)  # type: ignore[index]
        self.assertTrue(allocated["source_fact_allocation"]["complete"])  # type: ignore[index]
        self.assertEqual(allocated["source_fact_allocation"]["unallocated"], [])  # type: ignore[index]
        self.assertEqual(len(allocated["source_fact_allocation"]["allocations"]), 371)  # type: ignore[index]

    def test_evidence_alignment_rejects_unknown_or_out_of_lesson_concepts_and_extra_fields(self) -> None:
        candidate, source_map, _manifest_value, targets = self._exact_evidence_mismatch_fixture()
        prepared = _v5_prepare_evidence_alignment_targets(candidate, targets, source_map)
        target = prepared[0]
        valid = {
            "path": target["path"],
            "operation": "align_concepts_to_evidence",
            "concept_ids": list(target["required_concept_ids"]),
            "primary_concept_ids": list(target["primary_concept_options"][0]),
        }
        invalid_cases = {
            "unknown": {**valid, "concept_ids": ["concept_unknown"]},
            "outside_lesson": {**valid, "concept_ids": [
                next(concept["id"] for concept in source_map["concepts"]  # type: ignore[index]
                     if concept["id"] not in target["allowed_concept_ids"]),
            ]},
            "evidence_scope_field": {**valid, "primary_evidence_scope_ids": ["forbidden"]},
            "learning_blocks_field": {**valid, "learning_blocks": []},
        }
        for name, patch in invalid_cases.items():
            with self.subTest(name=name):
                with self.assertRaises(WorkflowFailure) as raised:
                    apply_course_architecture_repair_patches(candidate, prepared, {"patches": [patch]})
                if name == "unknown":
                    self.assertEqual(raised.exception.internal_code, "ARCH_REPAIR_INVALID_CONCEPT_ID")
                    self.assertEqual(raised.exception.diagnostics["guard_reason"], "INVALID_CONCEPT_ID")
                elif name == "outside_lesson":
                    self.assertEqual(raised.exception.internal_code, "ARCH_REPAIR_CONCEPT_NOT_COMPATIBLE_WITH_EVIDENCE_SCOPE")
                else:
                    self.assertEqual(raised.exception.diagnostics["guard_reason"], "DISALLOWED_OUTER_FIELD")

    def test_evidence_alignment_fails_closed_without_a_compatible_lesson_concept(self) -> None:
        candidate, source_map, _manifest_value, targets = self._exact_evidence_mismatch_fixture()
        lesson = candidate["chapters"][3]["lessons"][0]  # type: ignore[index]
        lesson["primary_concept_ids"] = []
        lesson["supporting_concept_ids"] = []
        with self.assertRaises(WorkflowFailure) as raised:
            _v5_prepare_evidence_alignment_targets(candidate, targets, source_map)
        self.assertEqual(raised.exception.internal_code, "ARCH_REPAIR_NO_COMPATIBLE_CONCEPT")
        self.assertEqual(raised.exception.diagnostics["guard_reason"], "NO_COMPATIBLE_CONCEPT")

    def test_evidence_alignment_multiple_primary_candidates_uses_typed_provider_path(self) -> None:
        candidate, source_map, _manifest_value, targets = self._exact_evidence_mismatch_fixture()
        lesson = candidate["chapters"][3]["lessons"][0]  # type: ignore[index]
        target_unit = lesson["units"][1]
        target_block = target_unit["learning_blocks"][0]
        target_scope_id = target_block["primary_evidence_scope_ids"][0]
        target_scope = next(scope for scope in source_map["source_evidence_scopes"] if scope["id"] == target_scope_id)  # type: ignore[index]
        alternate = next(
            concept["id"] for concept in source_map["concepts"]  # type: ignore[index]
            if concept["id"] not in target_scope["concept_ids"]
        )
        target_scope["concept_ids"] = [*target_scope["concept_ids"], alternate]
        lesson["primary_concept_ids"] = [*lesson["primary_concept_ids"], alternate]
        target_unit["primary_concept_ids"] = []
        target_block["primary_concept_ids"] = []
        prepared = _v5_prepare_evidence_alignment_targets(candidate, targets, source_map)
        self.assertEqual(len(prepared[0]["primary_concept_options"]), 2)
        self.assertIsNone(_v5_deterministic_evidence_alignment_candidate(candidate, prepared))
        selected = prepared[0]["primary_concept_options"][1]
        repaired = apply_course_architecture_repair_patches(candidate, prepared, {"patches": [{
            "path": prepared[0]["path"],
            "operation": "align_concepts_to_evidence",
            "concept_ids": prepared[0]["required_concept_ids"],
            "primary_concept_ids": selected,
        }]})
        self.assertEqual(
            repaired["chapters"][3]["lessons"][0]["units"][1]["primary_concept_ids"],  # type: ignore[index]
            selected,
        )

    def test_evidence_alignment_has_a_dedicated_provider_schema(self) -> None:
        schema = build_v5_semantic_delta_repair_response_schema({"align_concepts_to_evidence"})
        patch = schema.properties["patches"].items
        self.assertEqual(set(patch.required), {"path", "operation", "concept_ids", "primary_concept_ids"})
        self.assertNotIn("learning_blocks", patch.properties)
        self.assertNotIn("source_refs", patch.properties)
        self.assertNotIn("primary_evidence_scope_ids", patch.properties)

    async def test_v5_provider_counter_remains_monotonic_after_post_provider_guard_failure(self) -> None:
        _context, source_map, _manifest_value = self._context(20, 2)
        candidate = _v5_blueprint(source_map)
        issue = {
            "code": "EVIDENCE_SCOPE_CONCEPT_MISMATCH",
            "severity": "error",
            "message": "mismatch",
            "path": "chapter_1.lesson_1.unit_1",
            "repair_layer": "EVIDENCE_SEMANTIC",
        }
        events: list[dict[str, object]] = []

        async def architect(_source_map: dict[str, object]) -> WorkflowGenerationResult:
            return WorkflowGenerationResult(copy.deepcopy(candidate))

        async def repair(
            _blueprint: dict[str, object],
            _targets: list[dict[str, object]],
            _source_map: dict[str, object],
        ) -> WorkflowGenerationResult:
            raise WorkflowFailure(
                "ARCHITECTURE_REPAIR_INVALID",
                "provider returned an invalid typed patch",
                internal_code="ARCH_REPAIR_CONCEPT_NOT_COMPATIBLE_WITH_EVIDENCE_SCOPE",
                failure_stage="architecture_repair_semantic_delta_guard",
                diagnostics={
                    "guard_reason": "CONCEPT_NOT_COMPATIBLE_WITH_EVIDENCE_SCOPE",
                    "executed_provider_call_count": 1,
                },
            )

        with self.assertRaises(WorkflowFailure) as raised:
            await run_course_architecture_workflow(
                CourseArchitectureWorkflowCallbacks(
                    validate_source_scope=lambda: [],
                    build_source_map=lambda: copy.deepcopy(source_map),
                    architect=architect,
                    validate_blueprint=lambda _candidate, _map: WorkflowValidationResult([dict(issue)]),
                    repair_blueprint=repair,
                    emit_diagnostic=events.append,
                ),
                request_context={},
                max_repair_attempts=2,
            )
        self.assertEqual(raised.exception.diagnostics["repair_provider_calls"], 1)
        guard_events = [event for event in events if event.get("stage") == "architecture_repair_semantic_delta_guard"]
        self.assertTrue(guard_events)
        self.assertEqual(guard_events[-1]["total_repair_provider_calls"], 1)
        self.assertEqual(guard_events[-1]["provider_calls_executed"], 1)

    def test_371_fact_context_is_immutable_through_local_missing_owner_repair(self) -> None:
        context, source_map, manifest = self._context()
        candidate = _v5_blueprint(source_map)
        self._remove_primary_scope_owners(candidate, count=4)

        semantic = validate_course_architecture_evidence_scope(candidate, context.source_map_copy())
        missing = [issue for issue in semantic.errors if issue["code"] == "MISSING_PRIMARY_EVIDENCE_SCOPE_OWNER"]
        self.assertEqual(len(missing), 4)
        self.assertTrue(all(issue["path"].startswith("chapter_") for issue in missing))
        targets = classify_course_repair_targets(missing)
        self.assertLessEqual(len(targets), 4)
        self.assertTrue(all(target["scope"] == "unit" for target in targets))
        self.assertTrue(all(target["allowed_fields"] == ["learning_blocks"] for target in targets))

        payload = {"patches": [
            {
                "path": target["path"],
                "operation": "replace",
                "replacement": {"learning_blocks": self._replacement_blocks_for_target(candidate, target, source_map)},
            }
            for target in targets
        ]}
        repaired = apply_course_architecture_repair_patches(candidate, targets, payload)
        self.assertFalse(validate_course_architecture_evidence_scope(repaired, context.source_map_copy()).errors)
        assert_v5_immutable_source_context(context, stage="test_after_patch")

        allocated = allocate_source_map_architecture_facts(
            repaired,
            context.source_map_copy(),
            context.manifest_copy(),
        )
        finalized = allocate_blueprint_source_fact_ids(allocated, context.manifest_copy(), _nodes(6))
        self.assertEqual(finalized["source_fact_allocation"]["required_count"], 371)  # type: ignore[index]
        self.assertEqual(finalized["source_fact_allocation"]["allocated_count"], 371)  # type: ignore[index]
        self.assertTrue(finalized["source_fact_allocation"]["complete"])  # type: ignore[index]
        self.assertFalse(validate_course_architecture_workflow(
            finalized, context.source_map_copy(), context.manifest_copy(), {f"src-{index:03d}" for index in range(1, 7)},
        ).errors)

    def test_unknown_scope_in_local_ownership_patch_fails_closed(self) -> None:
        context, source_map, _manifest_value = self._context(20, 2)
        candidate = _v5_blueprint(source_map)
        self._remove_primary_scope_owners(candidate, count=1)
        missing = [issue for issue in validate_course_architecture_evidence_scope(candidate, context.source_map_copy()).errors if issue["code"] == "MISSING_PRIMARY_EVIDENCE_SCOPE_OWNER"]
        target = classify_course_repair_targets(missing)[0]
        blocks = self._replacement_blocks_for_target(candidate, target, source_map)
        blocks[0]["primary_evidence_scope_ids"].append("unknown_scope")
        with self.assertRaises(WorkflowFailure) as raised:
            apply_course_architecture_repair_patches(candidate, [target], {
                "patches": [{"path": target["path"], "operation": "replace", "replacement": {"learning_blocks": blocks}}],
            })
        self.assertEqual(raised.exception.internal_code, "ARCH_REPAIR_SOURCE_SCOPE_EXPANSION")

    def test_four_units_is_a_lesson_local_schema_target_and_does_not_touch_other_chapters(self) -> None:
        _context, source_map, _manifest_value = self._context(20, 2)
        candidate = _v5_blueprint(source_map)
        lesson = candidate["chapters"][0]["lessons"][0]  # type: ignore[index]
        lesson["units"] = [copy.deepcopy(lesson["units"][0]) for _ in range(4)]
        raw = parse_lesson_author_blueprint_candidate(json.dumps(candidate), forbid_provider_fact_ownership=True)
        with self.assertRaises(LessonAuthorBlueprintValidationError) as raised:
            validate_lesson_author_blueprint(raw, require_source_fact_ownership=False, forbid_provider_fact_ownership=True)
        self.assertEqual(raised.exception.path, "chapters[0].lessons[0].units")
        issue = _workflow_issue_from_blueprint_validation_error(raised.exception)
        issue["repairable"] = True
        target = classify_course_repair_targets([issue])[0]
        self.assertEqual(target["scope"], "lesson")
        self.assertEqual(target["path"], "chapter_1.lesson_1")
        self.assertEqual(target["allowed_fields"], ["units"])
        before_second_chapter = copy.deepcopy(candidate["chapters"][1])  # type: ignore[index]
        repaired = apply_course_architecture_repair_patches(candidate, [target], {
            "patches": [{
                "path": "chapter_1.lesson_1",
                "operation": "replace",
                "replacement": {"units": copy.deepcopy(lesson["units"][:3])},
            }],
        })
        self.assertEqual(repaired["chapters"][1], before_second_chapter)  # type: ignore[index]
        self.assertEqual(len(repaired["chapters"][0]["lessons"][0]["units"]), 3)  # type: ignore[index]

    def test_exact_schema_whitelist_rejects_unrelated_lesson_metadata(self) -> None:
        _context, source_map, _manifest_value = self._context(20, 2)
        candidate = _v5_blueprint(source_map)
        lesson = candidate["chapters"][0]["lessons"][0]  # type: ignore[index]
        lesson["units"] = [copy.deepcopy(lesson["units"][0]) for _ in range(4)]
        raw = parse_lesson_author_blueprint_candidate(json.dumps(candidate), forbid_provider_fact_ownership=True)
        with self.assertRaises(LessonAuthorBlueprintValidationError) as raised:
            validate_lesson_author_blueprint(raw, require_source_fact_ownership=False, forbid_provider_fact_ownership=True)
        issue = _workflow_issue_from_blueprint_validation_error(raised.exception)
        issue["repairable"] = True
        target = classify_course_repair_targets([issue])[0]
        with self.assertRaises(WorkflowFailure) as rejected:
            apply_course_architecture_repair_patches(candidate, [target], {
                "patches": [{
                    "path": target["path"],
                    "operation": "replace",
                    "replacement": {
                        "title": "unrelated mutation",
                        "units": copy.deepcopy(lesson["units"][:3]),
                    },
                }],
            })
        self.assertEqual(rejected.exception.internal_code, "ARCH_REPAIR_SCOPE_VIOLATION")

    def test_finding_specific_unit_whitelists_are_minimal(self) -> None:
        semantic = classify_course_repair_targets([{
            "code": "EVIDENCE_SCOPE_CONCEPT_MISMATCH", "severity": "error",
            "message": "mismatch", "path": "chapter_1.lesson_1.unit_1",
        }])[0]
        coherence = classify_course_repair_targets([{
            "code": "ACTION_OBJECTIVE_INSTRUCTION_MISMATCH", "severity": "error",
            "message": "coherence", "path": "chapter_1.lesson_1.unit_1",
        }])[0]
        self.assertEqual(semantic["allowed_fields"], ["concept_ids", "primary_concept_ids", "source_refs", "learning_blocks"])
        self.assertEqual(coherence["allowed_fields"], ["purpose", "learning_objective_refs", "learning_blocks"])

    def test_assessment_repair_is_limited_to_the_affected_knowledge_check_unit(self) -> None:
        assessment = classify_course_repair_targets([{
            "code": "ASSESSMENT_OBJECTIVE_NOT_COVERED", "severity": "error",
            "message": "assessment", "path": "chapter_1.lesson_1.unit_1",
        }])[0]
        grounding = classify_course_repair_targets([{
            "code": "ASSESSMENT_EVIDENCE_NOT_GROUNDED", "severity": "error",
            "message": "grounding", "path": "chapter_1.lesson_1.unit_1",
        }])[0]
        self.assertEqual(assessment["allowed_fields"], ["learning_blocks"])
        self.assertEqual(grounding["allowed_fields"], ["learning_blocks"])

        _context, source_map, _manifest_value = self._context(20, 2)
        candidate = _v5_blueprint(source_map)
        original = copy.deepcopy(candidate)
        for forbidden_field, value in {
            "title": "must not be mutable",
            "estimated_minutes": 30,
            "primary_concept_ids": ["concept_1"],
            "prerequisite_concept_ids": ["concept_1"],
            "source_refs": ["src-001"],
        }.items():
            with self.subTest(forbidden_field=forbidden_field):
                with self.assertRaises(WorkflowFailure) as rejected:
                    apply_course_architecture_repair_patches(candidate, [assessment], {
                        "patches": [{
                            "path": assessment["path"], "operation": "replace",
                            "replacement": {forbidden_field: value},
                        }],
                    })
                self.assertEqual(rejected.exception.internal_code, "ARCH_REPAIR_SCOPE_VIOLATION")
        self.assertEqual(candidate, original)

    def _coherence_repair_fixture(self) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        _context, source_map, manifest = self._context(371, 6, evidence_scope_count=17)
        candidate = _v5_blueprint(source_map)

        action_lesson = candidate["chapters"][0]["lessons"][0]  # type: ignore[index]
        action_unit = action_lesson["units"][0]
        action_lesson["objective"] = "Perform the source-grounded procedure."
        action_unit["purpose"] = "Perform the required procedure safely."

        assessment_lesson = candidate["chapters"][1]["lessons"][0]  # type: ignore[index]
        assessment_unit = assessment_lesson["units"][0]
        teaching = copy.deepcopy(assessment_unit["learning_blocks"][0])
        knowledge_check = {
            "id": "lb_assessment_check",
            "intent": "knowledge_check",
            "importance": "assessment",
            "concept_ids": list(teaching["concept_ids"]),
            "primary_concept_ids": [],
            "primary_evidence_scope_ids": [],
            "supporting_evidence_scope_ids": list(teaching["primary_evidence_scope_ids"]),
            "source_refs": list(teaching["source_refs"]),
            "learning_objective_refs": ["lo_1"],
            "content": {},
        }
        # The check follows its teaching block but lacks objective references
        # and supporting evidence. A typed delta can repair only that block.
        knowledge_check["learning_objective_refs"] = []
        assessment_unit["learning_blocks"] = [teaching, knowledge_check]
        assessment_lesson["assessment_required"] = True
        assessment_lesson["assessment_objective_refs"] = ["lo_1"]
        return candidate, source_map, manifest

    def _post_allocation_depth_fixture(self) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        """One realistic V5 lesson that is evidence-rich but instructionally thin."""

        _context, source_map, manifest = self._context(371, 6, evidence_scope_count=17)
        candidate = _v5_blueprint(source_map)
        lesson = candidate["chapters"][0]["lessons"][0]  # type: ignore[index]
        unit = lesson["units"][0]
        lesson["learning_objectives"] = [
            "Identify the source-grounded safety control.",
            "Explain the documented purpose of the safety control.",
        ]
        unit["learning_objective_refs"] = ["lo_1", "lo_2"]
        unit["learning_blocks"][0]["learning_objective_refs"] = ["lo_1", "lo_2"]
        allocated = allocate_blueprint_source_fact_ids(
            allocate_source_map_architecture_facts(candidate, source_map, manifest),
            manifest,
            _nodes(6),
        )
        return allocated, source_map, manifest

    def test_post_allocation_depth_repair_derives_support_without_new_fact_ownership(self) -> None:
        finalized, source_map, manifest = self._post_allocation_depth_fixture()
        self.assertFalse(validate_v5_instructional_coherence(finalized).errors)
        issues = validate_v5_post_allocation_instructional_depth(finalized).errors
        self.assertEqual([issue["code"] for issue in issues], ["INSTRUCTIONAL_DEPTH_INSUFFICIENT"])
        target = classify_course_repair_targets(
            issues,
            repair_layer="POST_ALLOCATION_INSTRUCTIONAL_DEPTH",
        )[0]
        self.assertEqual(target["scope"], "lesson")
        self.assertEqual(target["semantic_operations"], ["add_instructional_support_block"])
        self.assertEqual(target["allowed_unit_paths"], ["chapter_1.lesson_1.unit_1"])
        anchor_id = target["allowed_block_ids"][0]
        repaired = apply_course_architecture_repair_patches(finalized, [target], {"patches": [{
            "path": target["path"],
            "operation": "add_instructional_support_block",
            "unit_path": "chapter_1.lesson_1.unit_1",
            "after_block_id": anchor_id,
            "intent": "worked_example",
            "learning_objective_refs": ["lo_1", "lo_2"],
            "content": {"purpose": "Illustrate the taught safety control in its documented context."},
        }]})
        blocks = _node_at(repaired, "chapter_1.lesson_1.unit_1")["learning_blocks"]
        support = next(block for block in blocks if block["id"].startswith("lb_server_instructional_support_"))
        anchor = next(block for block in blocks if block["id"] == anchor_id)
        self.assertEqual(support["primary_evidence_scope_ids"], [])
        self.assertEqual(support["supporting_evidence_scope_ids"], sorted(anchor["primary_evidence_scope_ids"]))
        self.assertEqual(support.get("source_fact_ids", []), [])
        reallocated = allocate_blueprint_source_fact_ids(
            allocate_source_map_architecture_facts(repaired, source_map, manifest),
            manifest,
            _nodes(6),
        )
        allocation = reallocated["source_fact_allocation"]  # type: ignore[index]
        self.assertEqual(allocation["required_count"], 371)
        self.assertEqual(allocation["allocated_count"], 371)
        self.assertEqual(allocation["unallocated"], [])
        self.assertFalse(validate_course_architecture_workflow(
            reallocated, source_map, manifest, {f"src-{index:03d}" for index in range(1, 7)},
        ).errors)

    def test_post_allocation_depth_provider_context_excludes_evidence_scope_identifiers(self) -> None:
        finalized, source_map, _manifest_value = self._post_allocation_depth_fixture()
        target = classify_course_repair_targets(
            validate_v5_post_allocation_instructional_depth(finalized).errors,
            repair_layer="POST_ALLOCATION_INSTRUCTIONAL_DEPTH",
        )[0]
        context = build_v5_scoped_repair_source_context(
            blueprint=finalized,
            targets=[target],  # type: ignore[list-item]
            source_map=source_map,
        )
        self.assertIn('"v5_post_allocation_depth_repair":true', context)
        self.assertNotIn('"source_evidence_scopes"', context)
        self.assertNotIn('"id":', context)

    def test_post_allocation_depth_repair_rejects_cross_lesson_objective_and_provenance_fields(self) -> None:
        finalized, _source_map, _manifest_value = self._post_allocation_depth_fixture()
        target = classify_course_repair_targets(
            validate_v5_post_allocation_instructional_depth(finalized).errors,
            repair_layer="POST_ALLOCATION_INSTRUCTIONAL_DEPTH",
        )[0]
        anchor_id = target["allowed_block_ids"][0]
        base_patch = {
            "path": target["path"], "operation": "add_instructional_support_block",
            "unit_path": "chapter_1.lesson_1.unit_1", "after_block_id": anchor_id,
            "intent": "worked_example", "learning_objective_refs": ["lo_1"],
            "content": {"purpose": "Apply the taught safety control."},
        }
        cases = [
            ({**base_patch, "unit_path": "chapter_2.lesson_1.unit_1"}, "TARGET_BOUNDARY_MUTATION"),
            ({**base_patch, "learning_objective_refs": ["lo_999"]}, "INVALID_OBJECTIVE_REF"),
            ({**base_patch, "intent": "knowledge_check"}, "INVALID_SEMANTIC_INTENT"),
            ({**base_patch, "source_fact_ids": ["fact-0001"]}, "DISALLOWED_OUTER_FIELD"),
        ]
        for patch, reason in cases:
            with self.subTest(reason=reason):
                with self.assertRaises(WorkflowFailure) as raised:
                    apply_course_architecture_repair_patches(finalized, [target], {"patches": [patch]})
                self.assertEqual(raised.exception.diagnostics["guard_reason"], reason)

    def test_post_allocation_depth_repair_fails_closed_without_unique_grounded_anchor(self) -> None:
        finalized, _source_map, _manifest_value = self._post_allocation_depth_fixture()
        target = classify_course_repair_targets(
            validate_v5_post_allocation_instructional_depth(finalized).errors,
            repair_layer="POST_ALLOCATION_INSTRUCTIONAL_DEPTH",
        )[0]
        anchor_id = target["allowed_block_ids"][0]
        unit = _node_at(finalized, "chapter_1.lesson_1.unit_1")
        anchor = unit["learning_blocks"][0]
        anchor["primary_evidence_scope_ids"] = []
        patch = {
            "path": target["path"], "operation": "add_instructional_support_block",
            "unit_path": "chapter_1.lesson_1.unit_1", "after_block_id": anchor_id,
            "intent": "worked_example", "learning_objective_refs": ["lo_1"],
            "content": {"purpose": "Apply the taught safety control."},
        }
        with self.assertRaises(WorkflowFailure) as raised:
            apply_course_architecture_repair_patches(finalized, [target], {"patches": [patch]})
        self.assertEqual(raised.exception.diagnostics["guard_reason"], "NO_COMPATIBLE_SUPPORTING_EVIDENCE")

    async def test_exact_uat_sequence_repairs_pre_coherence_then_post_allocation_depth_within_cap(self) -> None:
        """Three action repairs, allocation 371/371, then three depth deltas.

        This is intentionally provider-free: it verifies graph scheduling and
        server-side delta application rather than invoking Gemini.
        """

        context, source_map, manifest = self._context(371, 6, evidence_scope_count=17)
        candidate = _v5_blueprint(source_map)
        for chapter_index in range(3):
            lesson = candidate["chapters"][chapter_index]["lessons"][0]  # type: ignore[index]
            unit = lesson["units"][0]
            lesson["objective"] = "Perform the documented safety procedure."
            unit["purpose"] = "Perform the required safety procedure."
        for chapter_index in range(3, 6):
            lesson = candidate["chapters"][chapter_index]["lessons"][0]  # type: ignore[index]
            unit = lesson["units"][0]
            lesson["learning_objectives"] = [
                "Identify the source-grounded safety control.",
                "Explain the documented purpose of the safety control.",
            ]
            unit["learning_objective_refs"] = ["lo_1", "lo_2"]
            unit["learning_blocks"][0]["learning_objective_refs"] = ["lo_1", "lo_2"]
        repairs: list[list[dict[str, object]]] = []

        def validate(blueprint: dict[str, object], _source_map: dict[str, object]) -> WorkflowValidationResult:
            pre = validate_v5_instructional_coherence(blueprint)
            if pre.errors:
                return pre
            allocated = allocate_blueprint_source_fact_ids(
                allocate_source_map_architecture_facts(blueprint, context.source_map_copy(), context.manifest_copy()),
                context.manifest_copy(),
                _nodes(6),
            )
            blueprint.clear()
            blueprint.update(allocated)
            return validate_course_architecture_workflow(
                allocated,
                context.source_map_copy(),
                context.manifest_copy(),
                {f"src-{index:03d}" for index in range(1, 7)},
            )

        async def architect(_source_map: dict[str, object]) -> WorkflowGenerationResult:
            return WorkflowGenerationResult(copy.deepcopy(candidate))

        async def repair(
            blueprint: dict[str, object],
            targets: list[dict[str, object]],
            _source_map: dict[str, object],
        ) -> WorkflowGenerationResult:
            repairs.append(copy.deepcopy(targets))
            operation = str(targets[0]["semantic_operations"][0])
            patches: list[dict[str, object]] = []
            for target in targets:
                if operation == "set_block_intent":
                    patches.append({
                        "path": target["path"], "operation": operation,
                        "block_id": target["allowed_block_ids"][0], "intent": "procedure",
                    })
                elif operation == "add_instructional_support_block":
                    unit_path = target["allowed_unit_paths"][0]
                    patches.append({
                        "path": target["path"], "operation": operation,
                        "unit_path": unit_path, "after_block_id": target["allowed_block_ids"][0],
                        "intent": "worked_example", "learning_objective_refs": ["lo_1", "lo_2"],
                        "content": {"purpose": "Illustrate the source-grounded control in context."},
                    })
                else:
                    raise AssertionError(f"unexpected operation: {operation}")
            return WorkflowGenerationResult(apply_course_architecture_repair_patches(
                blueprint,
                targets,  # type: ignore[arg-type]
                {"patches": patches},
            ))

        blueprint, diagnostics = await run_course_architecture_workflow(
            CourseArchitectureWorkflowCallbacks(
                validate_source_scope=lambda: [],
                build_source_map=lambda: context.source_map_copy(),
                architect=architect,
                validate_blueprint=validate,
                repair_blueprint=repair,
            ),
            request_context={},
            max_repair_attempts=2,
        )
        self.assertEqual([targets[0]["repair_layer"] for targets in repairs], [
            "PRE_ALLOCATION_COHERENCE",
            "POST_ALLOCATION_INSTRUCTIONAL_DEPTH",
        ])
        self.assertEqual([len(targets) for targets in repairs], [3, 3])
        self.assertEqual(diagnostics["repair_provider_calls"], 2)
        self.assertEqual(diagnostics["repair_layer_attempts"], {
            "PRE_ALLOCATION_COHERENCE": 1,
            "POST_ALLOCATION_INSTRUCTIONAL_DEPTH": 1,
        })
        allocation = blueprint["source_fact_allocation"]
        self.assertEqual(allocation["required_count"], 371)
        self.assertEqual(allocation["allocated_count"], 371)
        self.assertEqual(allocation["unallocated"], [])
        self.assertTrue(allocation["complete"])

    async def test_post_allocation_depth_no_progress_fails_closed_after_one_scoped_repair(self) -> None:
        _finalized, source_map, manifest = self._post_allocation_depth_fixture()
        candidate = _v5_blueprint(source_map)
        lesson = candidate["chapters"][0]["lessons"][0]  # type: ignore[index]
        unit = lesson["units"][0]
        lesson["learning_objectives"] = [
            "Identify the source-grounded safety control.",
            "Explain the documented purpose of the safety control.",
        ]
        unit["learning_objective_refs"] = ["lo_1", "lo_2"]
        unit["learning_blocks"][0]["learning_objective_refs"] = ["lo_1", "lo_2"]
        repairs = 0

        def validate(blueprint: dict[str, object], _source_map: dict[str, object]) -> WorkflowValidationResult:
            if "source_fact_allocation" not in blueprint:
                allocated = allocate_blueprint_source_fact_ids(
                    allocate_source_map_architecture_facts(blueprint, source_map, manifest),
                    manifest,
                    _nodes(6),
                )
                blueprint.clear()
                blueprint.update(allocated)
            return validate_course_architecture_workflow(
                blueprint, source_map, manifest, {f"src-{index:03d}" for index in range(1, 7)},
            )

        async def architect(_source_map: dict[str, object]) -> WorkflowGenerationResult:
            return WorkflowGenerationResult(copy.deepcopy(candidate))

        async def no_progress_repair(
            blueprint: dict[str, object],
            _targets: list[dict[str, object]],
            _source_map: dict[str, object],
        ) -> WorkflowGenerationResult:
            nonlocal repairs
            repairs += 1
            return WorkflowGenerationResult(blueprint)

        with self.assertRaises(WorkflowFailure) as raised:
            await run_course_architecture_workflow(
                CourseArchitectureWorkflowCallbacks(
                    validate_source_scope=lambda: [],
                    build_source_map=lambda: copy.deepcopy(source_map),
                    architect=architect,
                    validate_blueprint=validate,
                    repair_blueprint=no_progress_repair,
                ),
                request_context={},
                max_repair_attempts=2,
            )
        self.assertEqual(repairs, 1)
        self.assertEqual(raised.exception.internal_code, "ARCH_REPAIR_NO_PROGRESS")
        self.assertEqual(raised.exception.failure_stage, "blueprint_revalidation")

    def test_latest_uat_two_patch_invalid_intent_rejects_before_candidate_mutation(self) -> None:
        candidate, source_map, _manifest_value = self._coherence_repair_fixture()
        self.assertFalse(validate_course_architecture_evidence_scope(candidate, source_map).errors)
        issues = validate_v5_instructional_coherence(candidate).errors
        self.assertEqual(
            {issue["code"] for issue in issues},
            {"ACTION_OBJECTIVE_INSTRUCTION_MISMATCH", "ASSESSMENT_OBJECTIVE_NOT_COVERED", "ASSESSMENT_EVIDENCE_NOT_GROUNDED"},
        )
        targets = classify_course_repair_targets(issues, repair_layer="PRE_ALLOCATION_COHERENCE")
        self.assertEqual(len(targets), 2)
        before = copy.deepcopy(candidate)
        action_target = next(target for target in targets if "ACTION_OBJECTIVE_INSTRUCTION_MISMATCH" in target["codes"])
        assessment_target = next(target for target in targets if "ASSESSMENT_OBJECTIVE_NOT_COVERED" in target["codes"])
        action_block_id = action_target["allowed_block_ids"][0]
        assessment_block_id = assessment_target["allowed_block_ids"][0]
        payload = {"patches": [
            {"path": action_target["path"], "operation": "set_block_intent", "block_id": action_block_id, "intent": "guided_procedure"},
            {"path": assessment_target["path"], "operation": "repair_knowledge_check", "block_id": assessment_block_id, "learning_objective_refs": ["lo_1"]},
        ]}
        with self.assertRaises(WorkflowFailure) as rejected:
            apply_course_architecture_repair_patches(candidate, targets, payload)
        self.assertEqual(rejected.exception.internal_code, "ARCH_REPAIR_INVALID_SEMANTIC_INTENT")
        self.assertEqual(rejected.exception.failure_stage, "architecture_repair_semantic_delta_guard")
        self.assertEqual(rejected.exception.diagnostics["repair_target_path"], action_target["path"])
        self.assertEqual(rejected.exception.diagnostics["guard_reason"], "INVALID_SEMANTIC_INTENT")
        self.assertEqual(candidate, before)
        # Allocation is intentionally a later stage and never receives this
        # rejected repair candidate.
        self.assertFalse("source_fact_allocation" in candidate)

    def test_valid_coherence_repair_preserves_schema_then_allocates_all_371_facts(self) -> None:
        candidate, source_map, manifest = self._coherence_repair_fixture()
        issues = validate_v5_instructional_coherence(candidate).errors
        targets = classify_course_repair_targets(issues, repair_layer="PRE_ALLOCATION_COHERENCE")
        action_target = next(target for target in targets if "ACTION_OBJECTIVE_INSTRUCTION_MISMATCH" in target["codes"])
        assessment_target = next(target for target in targets if "ASSESSMENT_OBJECTIVE_NOT_COVERED" in target["codes"])

        action_before = copy.deepcopy(_node_at(candidate, str(action_target["path"]))["learning_blocks"])
        action_block_id = action_target["allowed_block_ids"][0]
        assessment_block_id = assessment_target["allowed_block_ids"][0]
        repaired = apply_course_architecture_repair_patches(candidate, targets, {"patches": [
            {"path": action_target["path"], "operation": "set_block_intent", "block_id": action_block_id, "intent": "procedure"},
            {"path": assessment_target["path"], "operation": "repair_knowledge_check", "block_id": assessment_block_id, "learning_objective_refs": ["lo_1"]},
        ]})
        action_after = _node_at(repaired, str(action_target["path"]))["learning_blocks"]
        changed = next(block for block in action_after if block["id"] == action_block_id)
        original = next(block for block in action_before if block["id"] == action_block_id)
        self.assertEqual(changed["intent"], "procedure")
        for field in ("id", "concept_ids", "primary_concept_ids", "source_refs", "primary_evidence_scope_ids", "supporting_evidence_scope_ids"):
            self.assertEqual(changed[field], original[field])
        validate_lesson_author_blueprint(
            repaired,
            require_source_fact_ownership=False,
            forbid_provider_fact_ownership=True,
        )
        self.assertFalse(validate_course_architecture_evidence_scope(repaired, source_map).errors)
        self.assertFalse(validate_v5_instructional_coherence(repaired).errors)
        allocated = allocate_blueprint_source_fact_ids(
            allocate_source_map_architecture_facts(repaired, source_map, manifest),
            manifest,
            _nodes(6),
        )
        allocation = allocated["source_fact_allocation"]  # type: ignore[index]
        self.assertEqual(allocation["required_count"], 371)
        self.assertEqual(allocation["allocated_count"], 371)
        self.assertEqual(allocation["unallocated"], [])
        allocated_fact_ids = [item["fact_id"] for item in allocation["allocations"]]
        self.assertEqual(len(allocated_fact_ids), len(set(allocated_fact_ids)))

    def _cross_unit_assessment_alignment_fixture(self) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        """One earlier base-eligible teaching anchor and one later check unit."""

        _context, source_map, manifest = self._context(371, 6, evidence_scope_count=17)
        candidate = _v5_blueprint(source_map)
        lesson = candidate["chapters"][0]["lessons"][0]  # type: ignore[index]
        teaching_unit = lesson["units"][0]
        teaching = teaching_unit["learning_blocks"][0]
        # It is source/concept compatible and primary-evidence-grounded, but
        # lacks the selected objective: base eligible, not fully aligned.
        teaching["learning_objective_refs"] = []
        check_unit = copy.deepcopy(teaching_unit)
        check_unit["title"] = "Check the taught control"
        check_unit["primary_concept_ids"] = []
        check = check_unit["learning_blocks"][0]
        check.update({
            "id": "lb_later_check",
            "intent": "knowledge_check",
            "importance": "assessment",
            "primary_concept_ids": [],
            "primary_evidence_scope_ids": [],
            "supporting_evidence_scope_ids": [],
            "learning_objective_refs": [],
            "content": {},
        })
        lesson["units"].append(check_unit)
        lesson["assessment_required"] = True
        lesson["assessment_objective_refs"] = ["lo_1"]
        return candidate, source_map, manifest

    def test_cross_unit_base_anchor_repairs_deterministically_without_provider_and_allocates_371(self) -> None:
        candidate, source_map, manifest = self._cross_unit_assessment_alignment_fixture()
        issues = validate_v5_instructional_coherence(candidate).errors
        self.assertEqual(
            {issue["code"] for issue in issues},
            {"ASSESSMENT_OBJECTIVE_NOT_COVERED", "ASSESSMENT_EVIDENCE_NOT_GROUNDED"},
        )
        target = classify_course_repair_targets(issues, repair_layer="PRE_ALLOCATION_COHERENCE")[0]
        self.assertEqual(target["path"], "chapter_1.lesson_1.unit_2")
        prepared = _v5_prepare_assessment_alignment_targets(candidate, [target])
        self.assertEqual(prepared[0]["semantic_operations"], ["repair_assessment_alignment"])
        self.assertTrue(prepared[0]["deterministic_semantic_delta"])
        candidate_record = prepared[0]["assessment_alignment_candidates"][0]
        self.assertEqual(candidate_record["teaching_block_path"], "chapter_1.lesson_1.unit_1")
        payload = _v5_deterministic_assessment_alignment_payload(prepared)
        self.assertIsNotNone(payload)
        repaired = apply_course_architecture_repair_patches(candidate, prepared, payload or {"patches": []})
        teaching = repaired["chapters"][0]["lessons"][0]["units"][0]["learning_blocks"][0]  # type: ignore[index]
        check = repaired["chapters"][0]["lessons"][0]["units"][1]["learning_blocks"][0]  # type: ignore[index]
        self.assertEqual(teaching["learning_objective_refs"], ["lo_1"])
        self.assertEqual(check["learning_objective_refs"], ["lo_1"])
        self.assertEqual(check["primary_evidence_scope_ids"], [])
        self.assertEqual(check["supporting_evidence_scope_ids"], sorted(teaching["primary_evidence_scope_ids"]))
        self.assertFalse(validate_v5_instructional_coherence(repaired).errors)
        allocated = allocate_blueprint_source_fact_ids(
            allocate_source_map_architecture_facts(repaired, source_map, manifest),
            manifest,
            _nodes(6),
        )
        allocation = allocated["source_fact_allocation"]  # type: ignore[index]
        self.assertEqual(allocation["allocated_count"], 371)
        self.assertEqual(allocation["unallocated"], [])
        self.assertEqual(len({item["fact_id"] for item in allocation["allocations"]}), 371)

    def test_fully_aligned_anchor_retains_existing_knowledge_check_operation(self) -> None:
        candidate, _source_map, _manifest = self._cross_unit_assessment_alignment_fixture()
        teaching = candidate["chapters"][0]["lessons"][0]["units"][0]["learning_blocks"][0]  # type: ignore[index]
        teaching["learning_objective_refs"] = ["lo_1"]
        issues = validate_v5_instructional_coherence(candidate).errors
        target = classify_course_repair_targets(issues, repair_layer="PRE_ALLOCATION_COHERENCE")[0]
        prepared = _v5_prepare_assessment_alignment_targets(candidate, [target])
        self.assertEqual(prepared[0]["semantic_operations"], ["repair_knowledge_check"])
        self.assertNotIn("deterministic_semantic_delta", prepared[0])

    def test_assessment_alignment_rejects_unapproved_cross_unit_teaching_id_and_extra_provenance(self) -> None:
        candidate, _source_map, _manifest = self._cross_unit_assessment_alignment_fixture()
        target = classify_course_repair_targets(
            validate_v5_instructional_coherence(candidate).errors,
            repair_layer="PRE_ALLOCATION_COHERENCE",
        )[0]
        prepared = _v5_prepare_assessment_alignment_targets(candidate, [target])
        base = _v5_deterministic_assessment_alignment_payload(prepared)["patches"][0]  # type: ignore[index]
        for patch, reason in (
            ({**base, "teaching_block_id": "lb_unapproved"}, "TARGET_BOUNDARY_MUTATION"),
            ({**base, "learning_objective_refs": ["lo_999"]}, "INVALID_OBJECTIVE_REF"),
            ({**base, "source_refs": []}, "DISALLOWED_OUTER_FIELD"),
            ({**base, "primary_evidence_scope_ids": ["scope_fake"]}, "DISALLOWED_OUTER_FIELD"),
            ({**base, "source_fact_ids": ["fact_fake"]}, "DISALLOWED_OUTER_FIELD"),
        ):
            with self.subTest(reason=reason):
                with self.assertRaises(WorkflowFailure) as raised:
                    apply_course_architecture_repair_patches(candidate, prepared, {"patches": [patch]})
                self.assertEqual(raised.exception.diagnostics["guard_reason"], reason)

    def test_assessment_alignment_fails_before_provider_when_no_base_anchor_exists(self) -> None:
        candidate, _source_map, _manifest = self._cross_unit_assessment_alignment_fixture()
        teaching = candidate["chapters"][0]["lessons"][0]["units"][0]["learning_blocks"][0]  # type: ignore[index]
        teaching["primary_evidence_scope_ids"] = []
        target = classify_course_repair_targets(
            validate_v5_instructional_coherence(candidate).errors,
            repair_layer="PRE_ALLOCATION_COHERENCE",
        )[0]
        with self.assertRaises(WorkflowFailure) as raised:
            _v5_prepare_assessment_alignment_targets(candidate, [target])
        self.assertEqual(raised.exception.internal_code, "ARCH_REPAIR_NO_VALID_TEACHING_ANCHOR")
        self.assertEqual(raised.exception.diagnostics["guard_reason"], "NO_VALID_TEACHING_ANCHOR")

    def test_multiple_base_anchors_require_one_narrow_provider_choice(self) -> None:
        candidate, _source_map, _manifest = self._cross_unit_assessment_alignment_fixture()
        lesson = candidate["chapters"][0]["lessons"][0]  # type: ignore[index]
        teaching = lesson["units"][0]
        primary = list(teaching["learning_blocks"][0]["primary_evidence_scope_ids"])
        self.assertGreaterEqual(len(primary), 2)
        teaching["learning_blocks"][0]["primary_evidence_scope_ids"] = [primary[0]]
        second_teaching = copy.deepcopy(teaching)
        second_teaching["title"] = "Additional source-grounded teaching"
        second_teaching["primary_concept_ids"] = []
        second_block = second_teaching["learning_blocks"][0]
        second_block["id"] = "lb_second_base_anchor"
        second_block["primary_evidence_scope_ids"] = primary[1:]
        lesson["units"].insert(1, second_teaching)
        target = classify_course_repair_targets(
            validate_v5_instructional_coherence(candidate).errors,
            repair_layer="PRE_ALLOCATION_COHERENCE",
        )[0]
        prepared = _v5_prepare_assessment_alignment_targets(candidate, [target])
        self.assertEqual(prepared[0]["semantic_operations"], ["repair_assessment_alignment"])
        self.assertIsNot(prepared[0].get("deterministic_semantic_delta"), True)
        self.assertEqual(len(prepared[0]["assessment_alignment_candidates"]), 2)
        schema = build_v5_semantic_delta_repair_response_schema({"repair_assessment_alignment"})
        patch = schema.properties["patches"].items
        self.assertEqual(set(patch.required), {
            "path", "operation", "knowledge_check_block_id", "teaching_block_id", "learning_objective_refs",
        })
        self.assertFalse({"source_refs", "source_fact_ids", "primary_evidence_scope_ids", "concept_ids"}.intersection(patch.properties))
        selected = prepared[0]["assessment_alignment_candidates"][1]
        repaired = apply_course_architecture_repair_patches(candidate, prepared, {"patches": [{
            "path": prepared[0]["path"],
            "operation": "repair_assessment_alignment",
            "knowledge_check_block_id": selected["knowledge_check_block_id"],
            "teaching_block_id": selected["teaching_block_id"],
            "learning_objective_refs": selected["learning_objective_refs"],
        }]})
        self.assertFalse(validate_v5_instructional_coherence(repaired).errors)

    def test_cross_lesson_teaching_is_not_an_assessment_alignment_candidate(self) -> None:
        candidate, _source_map, _manifest = self._cross_unit_assessment_alignment_fixture()
        lesson = candidate["chapters"][0]["lessons"][0]  # type: ignore[index]
        # Make the only same-lesson teaching block ineligible. A valid block
        # elsewhere in the course must not expand this typed target.
        lesson["units"][0]["learning_blocks"][0]["primary_evidence_scope_ids"] = []
        other = candidate["chapters"][1]["lessons"][0]["units"][0]["learning_blocks"][0]  # type: ignore[index]
        self.assertTrue(other["primary_evidence_scope_ids"])
        target = classify_course_repair_targets(
            validate_v5_instructional_coherence(candidate).errors,
            repair_layer="PRE_ALLOCATION_COHERENCE",
        )[0]
        with self.assertRaises(WorkflowFailure) as raised:
            _v5_prepare_assessment_alignment_targets(candidate, [target])
        self.assertEqual(raised.exception.diagnostics["guard_reason"], "NO_VALID_TEACHING_ANCHOR")

    async def test_deterministic_assessment_preflight_uses_zero_provider_calls(self) -> None:
        candidate, source_map, manifest = self._cross_unit_assessment_alignment_fixture()
        provider_called = False

        def validate(blueprint: dict[str, object], _source_map: dict[str, object]) -> WorkflowValidationResult:
            coherence = validate_v5_instructional_coherence(blueprint)
            if coherence.errors:
                return coherence
            allocated = allocate_blueprint_source_fact_ids(
                allocate_source_map_architecture_facts(blueprint, source_map, manifest),
                manifest,
                _nodes(6),
            )
            blueprint.clear()
            blueprint.update(allocated)
            return validate_course_architecture_workflow(
                blueprint, source_map, manifest, {f"src-{index:03d}" for index in range(1, 7)},
            )

        async def architect(_source_map: dict[str, object]) -> WorkflowGenerationResult:
            return WorkflowGenerationResult(copy.deepcopy(candidate))

        async def provider_repair(
            _blueprint: dict[str, object], _targets: list[dict[str, object]], _source_map: dict[str, object],
        ) -> WorkflowGenerationResult:
            nonlocal provider_called
            provider_called = True
            raise AssertionError("deterministic assessment alignment must not call the provider")

        async def deterministic_repair(
            blueprint: dict[str, object], targets: list[dict[str, object]], _source_map: dict[str, object],
        ) -> WorkflowGenerationResult | None:
            payload = _v5_deterministic_assessment_alignment_payload(targets)  # type: ignore[arg-type]
            if payload is None:
                return None
            return WorkflowGenerationResult(apply_course_architecture_repair_patches(
                blueprint, targets, payload,  # type: ignore[arg-type]
            ))

        blueprint, diagnostics = await run_course_architecture_workflow(
            CourseArchitectureWorkflowCallbacks(
                validate_source_scope=lambda: [],
                build_source_map=lambda: copy.deepcopy(source_map),
                architect=architect,
                validate_blueprint=validate,
                repair_blueprint=provider_repair,
                prepare_repair_targets=lambda blueprint, targets, _source_map: _v5_prepare_assessment_alignment_targets(
                    blueprint, targets,
                ),
                deterministic_repair=deterministic_repair,
            ),
            request_context={},
            max_repair_attempts=2,
        )
        self.assertFalse(provider_called)
        self.assertEqual(diagnostics["repair_provider_calls"], 0)
        allocation = blueprint["source_fact_allocation"]
        self.assertEqual(allocation["allocated_count"], 371)
        self.assertEqual(allocation["unallocated"], [])

    def test_semantic_delta_rejects_block_objective_and_provenance_mutations(self) -> None:
        candidate, _source_map, _manifest_value = self._coherence_repair_fixture()
        targets = classify_course_repair_targets(
            validate_v5_instructional_coherence(candidate).errors,
            repair_layer="PRE_ALLOCATION_COHERENCE",
        )
        action_target = next(target for target in targets if "ACTION_OBJECTIVE_INSTRUCTION_MISMATCH" in target["codes"])
        assessment_target = next(target for target in targets if "ASSESSMENT_OBJECTIVE_NOT_COVERED" in target["codes"])
        cases = [
            (
                action_target,
                {"path": action_target["path"], "operation": "set_block_intent", "block_id": "unknown_block", "intent": "procedure"},
                "ARCH_REPAIR_INVALID_BLOCK_ID",
                "INVALID_BLOCK_ID",
            ),
            (
                action_target,
                {"path": action_target["path"], "operation": "set_block_intent", "block_id": action_target["allowed_block_ids"][0], "intent": "procedure", "source_refs": []},
                "ARCH_REPAIR_SCOPE_VIOLATION",
                "DISALLOWED_OUTER_FIELD",
            ),
            (
                assessment_target,
                {"path": assessment_target["path"], "operation": "repair_knowledge_check", "block_id": assessment_target["allowed_block_ids"][0], "learning_objective_refs": ["lo_999"]},
                "ARCH_REPAIR_INVALID_OBJECTIVE_REF",
                "INVALID_OBJECTIVE_REF",
            ),
        ]
        for target, patch, internal_code, reason in cases:
            with self.subTest(reason=reason):
                with self.assertRaises(WorkflowFailure) as raised:
                    apply_course_architecture_repair_patches(candidate, [target], {"patches": [patch]})
                self.assertEqual(raised.exception.internal_code, internal_code)
                self.assertEqual(raised.exception.diagnostics["guard_reason"], reason)

        # A block ID belonging to another unit is not a valid target block even
        # if its shape otherwise resembles a semantic block.
        other_unit = candidate["chapters"][1]["lessons"][0]["units"][0]  # type: ignore[index]
        other_unit["learning_blocks"][0]["id"] = "lb_other_unit"
        with self.assertRaises(WorkflowFailure) as cross_unit:
            apply_course_architecture_repair_patches(candidate, [action_target], {"patches": [{
                "path": action_target["path"], "operation": "set_block_intent",
                "block_id": "lb_other_unit", "intent": "procedure",
            }]})
        self.assertEqual(cross_unit.exception.diagnostics["guard_reason"], "INVALID_BLOCK_ID")

        # Two deltas for one exact target are a broad/unrelated mutation, and
        # copy-on-write keeps the original candidate intact.
        before = copy.deepcopy(candidate)
        with self.assertRaises(WorkflowFailure) as duplicate_target:
            apply_course_architecture_repair_patches(candidate, [action_target], {"patches": [
                {"path": action_target["path"], "operation": "set_block_intent", "block_id": action_target["allowed_block_ids"][0], "intent": "procedure"},
                {"path": action_target["path"], "operation": "set_block_intent", "block_id": action_target["allowed_block_ids"][0], "intent": "example"},
            ]})
        self.assertEqual(duplicate_target.exception.diagnostics["guard_reason"], "TARGET_BOUNDARY_MUTATION")
        self.assertEqual(candidate, before)

    def test_add_knowledge_check_derives_supporting_evidence_server_side(self) -> None:
        candidate, source_map, _manifest_value = self._coherence_repair_fixture()
        assessment_lesson = candidate["chapters"][1]["lessons"][0]  # type: ignore[index]
        assessment_unit = assessment_lesson["units"][0]
        assessment_unit["learning_blocks"] = [assessment_unit["learning_blocks"][0]]
        issues = validate_v5_instructional_coherence(candidate).errors
        required = [issue for issue in issues if issue["code"] == "ASSESSMENT_BLOCK_REQUIRED"]
        self.assertEqual(len(required), 1)
        target = classify_course_repair_targets(required, repair_layer="PRE_ALLOCATION_COHERENCE")[0]
        self.assertEqual(target["scope"], "unit")
        self.assertEqual(target["semantic_operations"], ["add_knowledge_check"])
        teaching_id = target["allowed_block_ids"][0]
        repaired = apply_course_architecture_repair_patches(candidate, [target], {"patches": [{
            "path": target["path"],
            "operation": "add_knowledge_check",
            "after_block_id": teaching_id,
            "learning_objective_refs": ["lo_1"],
        }]})
        blocks = _node_at(repaired, target["path"])["learning_blocks"]
        check = next(block for block in blocks if block["intent"] == "knowledge_check")
        teaching = next(block for block in blocks if block["id"] == teaching_id)
        self.assertEqual(check["primary_evidence_scope_ids"], [])
        self.assertEqual(check["supporting_evidence_scope_ids"], sorted(teaching["primary_evidence_scope_ids"]))
        self.assertEqual(check.get("source_fact_ids", []), [])
        self.assertFalse(validate_course_architecture_evidence_scope(repaired, source_map).errors)
        self.assertNotIn(
            "ASSESSMENT_BLOCK_REQUIRED",
            {issue["code"] for issue in validate_v5_instructional_coherence(repaired).errors},
        )

    def test_add_knowledge_check_fails_closed_without_unique_compatible_evidence(self) -> None:
        candidate, _source_map, _manifest_value = self._coherence_repair_fixture()
        lesson = candidate["chapters"][1]["lessons"][0]  # type: ignore[index]
        unit = lesson["units"][0]
        unit["learning_blocks"] = [unit["learning_blocks"][0]]
        teaching = unit["learning_blocks"][0]
        target = {
            "scope": "unit", "path": "chapter_2.lesson_1.unit_1",
            "codes": ["ASSESSMENT_BLOCK_REQUIRED"], "allowed_fields": [],
            "allowed_operations": ["add_knowledge_check"],
            "semantic_operations": ["add_knowledge_check"],
            "allowed_block_ids": [teaching["id"]], "allowed_objective_ids": ["lo_1"],
        }
        teaching["primary_evidence_scope_ids"] = []
        with self.assertRaises(WorkflowFailure) as no_evidence:
            apply_course_architecture_repair_patches(candidate, [target], {"patches": [{
                "path": target["path"], "operation": "add_knowledge_check",
                "after_block_id": teaching["id"], "learning_objective_refs": ["lo_1"],
            }]})
        self.assertEqual(no_evidence.exception.diagnostics["guard_reason"], "NO_COMPATIBLE_SUPPORTING_EVIDENCE")

        candidate, _source_map, _manifest_value = self._coherence_repair_fixture()
        lesson = candidate["chapters"][1]["lessons"][0]  # type: ignore[index]
        unit = lesson["units"][0]
        unit["learning_blocks"] = [unit["learning_blocks"][0]]
        teaching = unit["learning_blocks"][0]
        second = copy.deepcopy(teaching)
        second["id"] = "lb_second_eligible_teaching"
        unit["learning_blocks"].append(second)
        target["allowed_block_ids"] = [teaching["id"]]
        with self.assertRaises(WorkflowFailure) as ambiguous:
            apply_course_architecture_repair_patches(candidate, [target], {"patches": [{
                "path": target["path"], "operation": "add_knowledge_check",
                "after_block_id": teaching["id"], "learning_objective_refs": ["lo_1"],
            }]})
        self.assertEqual(ambiguous.exception.diagnostics["guard_reason"], "AMBIGUOUS_SUPPORTING_EVIDENCE")

    def test_repair_response_schema_uses_the_canonical_semantic_intent_enum(self) -> None:
        patches = COURSE_ARCHITECTURE_REPAIR_RESPONSE_SCHEMA.properties["patches"]
        replacement = patches.items.properties["replacement"]
        learning_blocks = replacement.properties["learning_blocks"]
        intent = learning_blocks.items.properties["intent"]
        self.assertIn("procedure", intent.enum)
        self.assertIn("knowledge_check", intent.enum)
        self.assertNotIn("guided_procedure", intent.enum)

    def test_semantic_delta_response_schema_exposes_no_provenance_or_full_blocks(self) -> None:
        schema = build_v5_semantic_delta_repair_response_schema({"set_block_intent"})
        patches = schema.properties["patches"]
        variant = patches.items
        self.assertNotIn("replacement", variant.properties)
        self.assertNotIn("learning_blocks", variant.properties)
        self.assertNotIn("source_refs", variant.properties)
        self.assertNotIn("primary_evidence_scope_ids", variant.properties)
        self.assertNotIn("source_fact_ids", variant.properties)
        with self.assertRaises(ValueError):
            build_v5_semantic_delta_repair_response_schema({"set_block_intent", "repair_knowledge_check"})

    def test_post_allocation_depth_schema_exposes_only_provider_owned_semantics(self) -> None:
        schema = build_v5_semantic_delta_repair_response_schema({"add_instructional_support_block"})
        variant = schema.properties["patches"].items
        self.assertEqual(set(variant.required), {
            "path", "operation", "unit_path", "after_block_id", "intent",
            "learning_objective_refs", "content",
        })
        self.assertNotIn("source_fact_ids", variant.properties)
        self.assertNotIn("source_refs", variant.properties)
        self.assertNotIn("primary_evidence_scope_ids", variant.properties)
        self.assertNotIn("supporting_evidence_scope_ids", variant.properties)
        self.assertNotIn("concept_ids", variant.properties)

    def test_repair_response_schema_serializes_with_the_installed_gemini_sdk(self) -> None:
        client = genai.Client(api_key="test-key")
        payload = models._GenerateContentConfig_to_mldev(
            client._api_client,
            types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=COURSE_ARCHITECTURE_REPAIR_RESPONSE_SCHEMA,
            ),
        )
        self.assertIn("responseSchema", payload)

    def test_semantic_delta_schema_serializes_with_the_installed_gemini_sdk(self) -> None:
        client = genai.Client(api_key="test-key")
        schema = build_v5_semantic_delta_repair_response_schema({"set_block_intent"})
        payload = models._GenerateContentConfig_to_mldev(
            client._api_client,
            types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        self.assertIn("responseSchema", payload)

    def test_v5_architect_prompt_prevents_the_three_latest_repair_causes(self) -> None:
        prompt = build_course_architect_prompt(
            blueprint_request(),
            "representative evidence",
            source_map_context="bounded source-map descriptor",
        )
        self.assertIn("Every lesson must contain one to three units, never four or more.", prompt)
        self.assertIn("objective hành động phải có procedure hoặc treatment thực hành phù hợp, không chỉ concept_explanation", prompt)
        self.assertIn("Mọi scope được tham chiếu phải thuộc cùng document/section/concept scope của block", prompt)

    async def test_latest_uat_layers_each_receive_one_bounded_repair_then_allocate_371_facts(self) -> None:
        context, source_map, manifest = self._context(371, 6, evidence_scope_count=17)
        self.assertEqual(context.canonical_fact_count, 371)
        self.assertEqual(context.evidence_scope_count, 17)
        candidate = _v5_blueprint(source_map)
        repairs: list[list[dict[str, object]]] = []
        validation_call = 0

        def finding(code: str, path: str, layer: str, **metadata: object) -> dict[str, object]:
            return {
                "code": code, "severity": "error", "message": code,
                "path": path, "repairable": True, "repair_layer": layer,
                **metadata,
            }

        def validate(blueprint: dict[str, object], _source_map: dict[str, object]) -> WorkflowValidationResult:
            nonlocal validation_call
            validation_call += 1
            if validation_call == 1:
                return WorkflowValidationResult([finding(
                    "BLUEPRINT_INVALID_SCHEMA", "chapter_2.lesson_1", "SCHEMA",
                    constraint="ARRAY_LENGTH", schema_path="chapters[1].lessons[0].units",
                )])
            if validation_call == 2:
                return WorkflowValidationResult([finding(
                    "EVIDENCE_SCOPE_CONCEPT_MISMATCH", "chapter_4.lesson_1.unit_1", "EVIDENCE_SEMANTIC",
                )])
            if validation_call == 3:
                return WorkflowValidationResult([
                    finding("ACTION_OBJECTIVE_INSTRUCTION_MISMATCH", path, "PRE_ALLOCATION_COHERENCE")
                    for path in (
                        "chapter_2.lesson_1.unit_1",
                        "chapter_3.lesson_1.unit_1",
                        "chapter_5.lesson_1.unit_1",
                    )
                ])
            allocated = allocate_source_map_architecture_facts(
                blueprint,
                context.source_map_copy(),
                context.manifest_copy(),
            )
            finalized = allocate_blueprint_source_fact_ids(allocated, context.manifest_copy(), _nodes(6))
            blueprint.clear()
            blueprint.update(finalized)
            return validate_course_architecture_workflow(
                finalized,
                context.source_map_copy(),
                context.manifest_copy(),
                {f"src-{index:03d}" for index in range(1, 7)},
            )

        async def architect(_source_map: dict[str, object]) -> WorkflowGenerationResult:
            return WorkflowGenerationResult(copy.deepcopy(candidate))

        async def repair(
            blueprint: dict[str, object],
            targets: list[dict[str, object]],
            _source_map: dict[str, object],
        ) -> WorkflowGenerationResult:
            repairs.append(copy.deepcopy(targets))
            return WorkflowGenerationResult(blueprint)

        blueprint, diagnostics = await run_course_architecture_workflow(
            CourseArchitectureWorkflowCallbacks(
                validate_source_scope=lambda: [],
                build_source_map=lambda: copy.deepcopy(source_map),
                architect=architect,
                validate_blueprint=validate,
                repair_blueprint=repair,
            ),
            request_context={},
            max_repair_attempts=2,
        )

        self.assertEqual(validation_call, 4)
        self.assertEqual(len(repairs), 3)
        self.assertEqual([targets[0]["repair_layer"] for targets in repairs], [
            "SCHEMA", "EVIDENCE_SEMANTIC", "PRE_ALLOCATION_COHERENCE",
        ])
        self.assertEqual(repairs[0][0]["allowed_fields"], ["units"])
        self.assertEqual(len(repairs[2]), 3)
        allocation = blueprint["source_fact_allocation"]
        self.assertEqual(allocation["required_count"], 371)
        self.assertEqual(allocation["allocated_count"], 371)
        self.assertEqual(len(allocation["unallocated"]), 0)
        self.assertTrue(allocation["complete"])
        self.assertEqual(diagnostics["repair_provider_calls"], 3)
        self.assertEqual(diagnostics["repair_layer_attempts"], {
            "SCHEMA": 1,
            "EVIDENCE_SEMANTIC": 1,
            "PRE_ALLOCATION_COHERENCE": 1,
        })

    async def test_v5_invalid_architect_attempts_never_take_legacy_source_locked_fallback(self) -> None:
        context, _source_map, _manifest_value = self._context(20, 2)
        usage = AiUsage(inputTokens=1, outputTokens=1, totalTokens=2)
        with patch("app.main.generate_content", new=AsyncMock(side_effect=[("{", usage), ("{", usage)])) as generate:
            with self.assertRaises(LessonAuthorBlueprintGenerationError) as raised:
                await generate_validated_lesson_author_blueprint(
                    blueprint_request(),
                    "SERVER MODE: COURSE_BLUEPRINT.",
                    allow_server_fact_allocation=True,
                    v5_immutable_source_context=context,
                )
        self.assertEqual(generate.await_count, 2)
        self.assertEqual(raised.exception.code, "V5_SOURCE_CONTEXT_FALLBACK_UNSAFE")
        self.assertEqual(raised.exception.usage.totalTokens, 4)
        self.assertEqual(context.canonical_fact_count, 20)

    async def test_v5_four_unit_architect_response_is_deferred_to_one_local_repair_not_regenerated(self) -> None:
        context, source_map, _manifest_value = self._context(20, 2)
        candidate = _v5_blueprint(source_map)
        lesson = candidate["chapters"][0]["lessons"][0]  # type: ignore[index]
        lesson["units"] = [copy.deepcopy(lesson["units"][0]) for _ in range(4)]
        usage = AiUsage(inputTokens=4, outputTokens=8, totalTokens=12)
        with patch("app.main.generate_content", new=AsyncMock(return_value=(json.dumps(candidate), usage))) as generate:
            deferred, returned_usage = await generate_validated_lesson_author_blueprint(
                blueprint_request(),
                "SERVER MODE: COURSE_BLUEPRINT.",
                allow_server_fact_allocation=True,
                v5_immutable_source_context=context,
            )
        self.assertEqual(generate.await_count, 1)
        self.assertEqual(len(deferred["chapters"][0]["lessons"][0]["units"]), 4)  # type: ignore[index]
        self.assertEqual(returned_usage.totalTokens, 12)

    def test_degraded_twenty_target_candidate_is_rejected_before_provider_repair(self) -> None:
        context, _source_map, _manifest_value = self._context(371, 6)
        targets = [{
            "scope": "unit", "path": f"chapter_1.lesson_1.unit_{index}",
            "codes": ["LESSON_TOO_THIN"], "allowed_fields": ["learning_blocks"],
            "diagnostics": [], "allowed_operations": ["replace"],
        } for index in range(1, 21)]
        with self.assertRaises(WorkflowFailure) as raised:
            assert_v5_scoped_repair_target_bound(targets, context)
        self.assertEqual(raised.exception.code, "ARCHITECTURE_REPAIR_SCOPE_TOO_LARGE")
        self.assertEqual(raised.exception.internal_code, "V5_REPAIR_TARGET_SET_TOO_LARGE")

    def test_context_mutation_is_detected_before_repair_or_allocation(self) -> None:
        context, _source_map, _manifest_value = self._context(20, 2)
        # The object is request-scoped and never passed to provider code. This
        # negative control proves any accidental server-side mutation fails.
        context.source_map["source_evidence_scopes"][0]["source_fact_ids"].append("fact-injected")  # type: ignore[index]
        with self.assertRaises(WorkflowFailure) as raised:
            assert_v5_immutable_source_context(context, stage="test_mutation")
        self.assertEqual(raised.exception.internal_code, "V5_SOURCE_CONTEXT_MUTATED")


if __name__ == "__main__":
    unittest.main()
