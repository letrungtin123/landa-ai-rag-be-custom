from __future__ import annotations

import copy
import unittest

from google import genai
from google.genai import models, types

from app.assessment_planner import compile_v5_assessment_plan
from app.lesson_author_blueprint import build_v5_semantic_delta_repair_response_schema
from app.main import (
    _v5_prepare_semantic_repair_targets,
    allocate_source_map_architecture_facts,
    apply_course_architecture_repair_patches,
)
from app.source_map import build_source_map
from app.workflows.course_architecture import classify_course_repair_targets
from tests.test_evidence_scope_allocation_v5 import _manifest, _nodes, _v5_blueprint


class V5AssessmentPlannerTests(unittest.TestCase):
    def _candidate(self, *, facts: int = 40, sections: int = 2) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        manifest = _manifest(facts, sections)
        source_map = build_source_map(_nodes(sections), manifest, locale="en")
        return _v5_blueprint(source_map), source_map, manifest

    @staticmethod
    def _enable_assessment(lesson: dict[str, object], refs: list[str]) -> None:
        lesson["assessment_required"] = True
        lesson["assessment_objective_refs"] = refs

    def test_one_fully_aligned_anchor_compiles_without_provider_selection(self) -> None:
        candidate, _source_map, _manifest_value = self._candidate()
        lesson = candidate["chapters"][0]["lessons"][0]  # type: ignore[index]
        self._enable_assessment(lesson, ["lo_1"])

        compiled = compile_v5_assessment_plan(candidate)

        self.assertEqual(compiled.status, "READY")
        self.assertEqual(compiled.deterministic_insertions, 1)
        blocks = compiled.blueprint["chapters"][0]["lessons"][0]["units"][0]["learning_blocks"]  # type: ignore[index]
        check = blocks[-1]
        self.assertEqual(check["intent"], "knowledge_check")
        self.assertEqual(check["primary_evidence_scope_ids"], [])
        self.assertEqual(check["supporting_evidence_scope_ids"], blocks[0]["primary_evidence_scope_ids"])
        self.assertEqual(check["learning_objective_refs"], ["lo_1"])

    def test_distributed_objectives_compile_to_their_own_anchors(self) -> None:
        candidate, _source_map, _manifest_value = self._candidate()
        lesson = candidate["chapters"][0]["lessons"][0]  # type: ignore[index]
        unit_one = lesson["units"][0]
        unit_two = copy.deepcopy(candidate["chapters"][1]["lessons"][0]["units"][0])  # type: ignore[index]
        unit_one["learning_objective_refs"] = ["lo_1"]
        unit_two["learning_objective_refs"] = ["lo_2"]
        unit_two["learning_blocks"][0]["id"] = "lb_objective_two"
        unit_two["learning_blocks"][0]["learning_objective_refs"] = ["lo_2"]
        lesson["learning_objectives"] = ["Identify the first concept.", "Explain the second concept."]
        lesson["units"].append(unit_two)
        self._enable_assessment(lesson, ["lo_1", "lo_2"])

        compiled = compile_v5_assessment_plan(candidate)

        self.assertEqual(compiled.status, "READY")
        units = compiled.blueprint["chapters"][0]["lessons"][0]["units"]  # type: ignore[index]
        self.assertEqual(units[0]["learning_blocks"][-1]["learning_objective_refs"], ["lo_1"])
        self.assertEqual(units[1]["learning_blocks"][-1]["learning_objective_refs"], ["lo_2"])

    def test_multiple_fully_aligned_anchors_requires_semantic_resolution(self) -> None:
        candidate, _source_map, _manifest_value = self._candidate()
        lesson = candidate["chapters"][0]["lessons"][0]  # type: ignore[index]
        unit = lesson["units"][0]
        alternate = copy.deepcopy(unit["learning_blocks"][0])
        alternate["id"] = "lb_alternate"
        unit["learning_blocks"].append(alternate)
        self._enable_assessment(lesson, ["lo_1"])

        compiled = compile_v5_assessment_plan(candidate)

        self.assertEqual(compiled.status, "NEEDS_SEMANTIC_RESOLUTION")
        self.assertEqual(len(compiled.candidates), 1)
        self.assertEqual(len(next(iter(compiled.candidates.values()))), 2)

    def test_single_semantic_candidate_requires_resolution_not_automatic_linking(self) -> None:
        candidate, _source_map, _manifest_value = self._candidate()
        lesson = candidate["chapters"][0]["lessons"][0]  # type: ignore[index]
        unit = lesson["units"][0]
        unit["learning_blocks"][0]["learning_objective_refs"] = []
        self._enable_assessment(lesson, ["lo_1"])

        compiled = compile_v5_assessment_plan(candidate)

        self.assertEqual(compiled.status, "NEEDS_SEMANTIC_RESOLUTION")
        candidate_value = next(iter(compiled.candidates.values()))[0]
        self.assertEqual(candidate_value.alignment, "semantic_candidate")

    def test_zero_safe_anchor_is_a_typed_terminal_gap(self) -> None:
        candidate, _source_map, _manifest_value = self._candidate()
        lesson = candidate["chapters"][0]["lessons"][0]  # type: ignore[index]
        lesson["units"][0]["learning_blocks"][0]["primary_evidence_scope_ids"] = []
        self._enable_assessment(lesson, ["lo_1"])

        compiled = compile_v5_assessment_plan(candidate)

        self.assertEqual(compiled.status, "TERMINAL_GAP")
        self.assertEqual(compiled.issues[0].code, "ASSESSMENT_PLAN_NO_SAFE_ANCHOR")
        self.assertFalse(compiled.issues[0].workflow_issue()["repairable"])

    def test_invalid_local_assessment_objective_fails_closed(self) -> None:
        candidate, _source_map, _manifest_value = self._candidate()
        lesson = candidate["chapters"][0]["lessons"][0]  # type: ignore[index]
        self._enable_assessment(lesson, ["lo_99"])

        compiled = compile_v5_assessment_plan(candidate)

        self.assertEqual(compiled.status, "TERMINAL_GAP")
        self.assertEqual(compiled.issues[0].code, "ASSESSMENT_PLAN_INVALID_OBJECTIVE_REF")

    def test_existing_valid_check_is_preserved_and_compilation_is_idempotent(self) -> None:
        candidate, _source_map, _manifest_value = self._candidate()
        lesson = candidate["chapters"][0]["lessons"][0]  # type: ignore[index]
        self._enable_assessment(lesson, ["lo_1"])
        first = compile_v5_assessment_plan(candidate)
        self.assertEqual(first.status, "READY")
        second = compile_v5_assessment_plan(first.blueprint)
        self.assertEqual(second.status, "READY")
        self.assertEqual(second.deterministic_insertions, 0)
        self.assertEqual(second.blueprint, first.blueprint)

    def test_semantic_resolver_cannot_select_unapproved_anchor_or_mutate_provenance(self) -> None:
        candidate, source_map, _manifest_value = self._candidate()
        lesson = candidate["chapters"][0]["lessons"][0]  # type: ignore[index]
        unit = lesson["units"][0]
        unit["learning_blocks"][0]["learning_objective_refs"] = []
        self._enable_assessment(lesson, ["lo_1"])
        compiled = compile_v5_assessment_plan(candidate)
        issues = [issue.workflow_issue() for issue in compiled.issues]
        for issue in issues:
            issue["repair_layer"] = "PRE_ALLOCATION_COHERENCE"
        targets = classify_course_repair_targets(issues, repair_layer="PRE_ALLOCATION_COHERENCE")
        prepared = _v5_prepare_semantic_repair_targets(candidate, targets, source_map)
        target = prepared[0]
        self.assertEqual(set(target["assessment_plan_objectives"]), {"lo_1"})
        self.assertTrue(target["assessment_plan_objectives"]["lo_1"])

        invalid_payload = {"patches": [{
            "path": target["path"],
            "operation": "select_assessment_teaching_alignment",
            "selections": [{
                "objective_ref": "lo_1", "decision": "SELECT",
                "unit_path": "chapter_2.lesson_1.unit_1", "teaching_block_id": "lb_2",
            }],
        }]}
        from app.workflows.contracts import WorkflowFailure
        with self.assertRaises(WorkflowFailure) as raised:
            apply_course_architecture_repair_patches(candidate, prepared, invalid_payload)
        self.assertEqual(raised.exception.internal_code, "ARCH_REPAIR_TARGET_OUT_OF_SCOPE")

        approved = target["assessment_plan_candidates"][0]
        valid_payload = {"patches": [{
            "path": target["path"],
            "operation": "select_assessment_teaching_alignment",
            "selections": [{
                "objective_ref": approved["objective_ref"], "decision": "SELECT",
                "unit_path": approved["unit_path"], "teaching_block_id": approved["teaching_block_id"],
            }],
        }]}
        repaired = apply_course_architecture_repair_patches(candidate, prepared, valid_payload)
        repaired_check = repaired["chapters"][0]["lessons"][0]["units"][0]["learning_blocks"][-1]  # type: ignore[index]
        self.assertEqual(repaired_check["primary_evidence_scope_ids"], [])
        self.assertEqual(repaired_check["supporting_evidence_scope_ids"], unit["learning_blocks"][0]["primary_evidence_scope_ids"])

    def test_compiled_assessment_keeps_371_fact_allocation_complete(self) -> None:
        candidate, source_map, manifest = self._candidate(facts=371, sections=6)
        lesson = candidate["chapters"][0]["lessons"][0]  # type: ignore[index]
        self._enable_assessment(lesson, ["lo_1"])

        compiled = compile_v5_assessment_plan(candidate)
        allocated = allocate_source_map_architecture_facts(compiled.blueprint, source_map, manifest)

        self.assertEqual(compiled.status, "READY")
        self.assertEqual(allocated["source_fact_allocation"]["allocated_count"], 371)  # type: ignore[index]
        self.assertEqual(allocated["source_fact_allocation"]["unallocated"], [])  # type: ignore[index]

    def test_semantic_selection_schema_exposes_no_provenance_and_serializes(self) -> None:
        schema = build_v5_semantic_delta_repair_response_schema({"select_assessment_teaching_alignment"})
        patch = schema.properties["patches"].items
        self.assertEqual(set(patch.required), {"path", "operation", "selections"})
        self.assertNotIn("source_refs", patch.properties)
        self.assertNotIn("primary_evidence_scope_ids", patch.properties)
        self.assertNotIn("source_fact_ids", patch.properties)
        client = genai.Client(api_key="test-key")
        payload = models._GenerateContentConfig_to_mldev(
            client._api_client,
            types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        self.assertIn("responseSchema", payload)


if __name__ == "__main__":
    unittest.main()
