from __future__ import annotations

import unittest

from app.main import (
    RagLessonAuthorDraftArchitecture,
    allocate_blueprint_source_fact_ids,
    allocate_source_map_architecture_facts,
    format_approved_lesson_quality_contract,
    validate_v5_instructional_coherence,
    validate_course_architecture_workflow,
)
from app.source_map import build_source_map
from tests.test_evidence_scope_allocation_v5 import _manifest, _nodes, _v5_blueprint


class V5BlueprintInstructionalCoherenceTests(unittest.TestCase):
    def _finalized(self, fact_count: int = 12) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        manifest = _manifest(fact_count, 1)
        source_map = build_source_map(_nodes(1), manifest, locale="en")
        candidate = _v5_blueprint(source_map)
        allocated = allocate_source_map_architecture_facts(candidate, source_map, manifest)
        finalized = allocate_blueprint_source_fact_ids(allocated, manifest, _nodes(1))
        return finalized, source_map, manifest

    def _validation_codes(self, blueprint: dict[str, object], source_map: dict[str, object], manifest: dict[str, object]) -> set[str]:
        return {
            str(issue["code"])
            for issue in validate_course_architecture_workflow(blueprint, source_map, manifest, {"src-001"}).issues
        }

    def test_assessment_required_lesson_needs_grounded_knowledge_check_before_draft(self) -> None:
        blueprint, source_map, manifest = self._finalized()
        lesson = blueprint["chapters"][0]["lessons"][0]  # type: ignore[index]
        lesson["assessment_required"] = True
        lesson["assessment_objective_refs"] = ["lo_1"]
        self.assertIn("ASSESSMENT_BLOCK_REQUIRED", self._validation_codes(blueprint, source_map, manifest))

    def test_teach_then_grounded_knowledge_check_is_coherent_and_facts_remain_primary_only(self) -> None:
        blueprint, source_map, manifest = self._finalized()
        lesson = blueprint["chapters"][0]["lessons"][0]  # type: ignore[index]
        unit = lesson["units"][0]
        lesson["assessment_required"] = True
        lesson["assessment_objective_refs"] = ["lo_1"]
        teaching = unit["learning_blocks"][0]
        unit["learning_blocks"].append({
            "id": "lb_check",
            "intent": "knowledge_check",
            "importance": "assessment",
            "concept_ids": teaching["concept_ids"],
            "primary_concept_ids": [],
            "primary_evidence_scope_ids": [],
            "supporting_evidence_scope_ids": list(teaching["primary_evidence_scope_ids"]),
            "source_refs": list(teaching["source_refs"]),
            "learning_objective_refs": ["lo_1"],
            "content": {},
            "source_fact_ids": [],
        })
        # Allocation metadata/fact ownership was produced before the optional
        # reinforcement block is added. The supporting block remains factless.
        codes = self._validation_codes(blueprint, source_map, manifest)
        self.assertNotIn("ASSESSMENT_BLOCK_REQUIRED", codes)
        self.assertNotIn("ASSESSMENT_OBJECTIVE_NOT_COVERED", codes)
        self.assertNotIn("ASSESSMENT_EVIDENCE_NOT_GROUNDED", codes)
        self.assertEqual(unit["learning_blocks"][1]["source_fact_ids"], [])

    def test_knowledge_check_cannot_take_second_primary_scope_ownership(self) -> None:
        blueprint, source_map, manifest = self._finalized()
        lesson = blueprint["chapters"][0]["lessons"][0]  # type: ignore[index]
        unit = lesson["units"][0]
        teaching = unit["learning_blocks"][0]
        lesson["assessment_required"] = True
        lesson["assessment_objective_refs"] = ["lo_1"]
        unit["learning_blocks"].append({
            "id": "lb_bad_check", "intent": "knowledge_check", "importance": "assessment",
            "concept_ids": teaching["concept_ids"], "primary_concept_ids": [],
            "primary_evidence_scope_ids": list(teaching["primary_evidence_scope_ids"]),
            "supporting_evidence_scope_ids": [], "source_refs": list(teaching["source_refs"]),
            "learning_objective_refs": ["lo_1"], "content": {}, "source_fact_ids": [],
        })
        self.assertIn("DUPLICATE_PRIMARY_EVIDENCE_SCOPE_OWNER", self._validation_codes(blueprint, source_map, manifest))

    def test_knowledge_check_missing_assessment_objective_is_rejected(self) -> None:
        blueprint, source_map, manifest = self._finalized()
        lesson = blueprint["chapters"][0]["lessons"][0]  # type: ignore[index]
        unit = lesson["units"][0]
        teaching = unit["learning_blocks"][0]
        lesson["assessment_required"] = True
        lesson["assessment_objective_refs"] = ["lo_1"]
        unit["learning_blocks"].append({
            "id": "lb_unaligned_check", "intent": "knowledge_check", "importance": "assessment",
            "concept_ids": teaching["concept_ids"], "primary_concept_ids": [],
            "primary_evidence_scope_ids": [], "supporting_evidence_scope_ids": list(teaching["primary_evidence_scope_ids"]),
            "source_refs": list(teaching["source_refs"]), "learning_objective_refs": [], "content": {}, "source_fact_ids": [],
        })
        self.assertIn("ASSESSMENT_OBJECTIVE_NOT_COVERED", self._validation_codes(blueprint, source_map, manifest))

    def test_low_complexity_informational_unit_can_stay_one_explanation(self) -> None:
        blueprint, source_map, manifest = self._finalized(2)
        codes = self._validation_codes(blueprint, source_map, manifest)
        self.assertNotIn("INSTRUCTIONAL_DEPTH_INSUFFICIENT", codes)
        self.assertNotIn("ACTION_OBJECTIVE_INSTRUCTION_MISMATCH", codes)

    def test_multi_objective_substantial_lesson_cannot_be_one_generic_explanation(self) -> None:
        blueprint, source_map, manifest = self._finalized(12)
        lesson = blueprint["chapters"][0]["lessons"][0]  # type: ignore[index]
        lesson["learning_objectives"] = ["Identify the source-grounded scope.", "Explain the source-grounded procedure."]
        self.assertIn("INSTRUCTIONAL_DEPTH_INSUFFICIENT", self._validation_codes(blueprint, source_map, manifest))

    def test_action_objective_cannot_be_satisfied_by_generic_explanation_only(self) -> None:
        blueprint, source_map, manifest = self._finalized()
        lesson = blueprint["chapters"][0]["lessons"][0]  # type: ignore[index]
        lesson["objective"] = "Perform the source-grounded procedure."
        lesson["units"][0]["purpose"] = "Perform the required procedure safely."
        self.assertIn("ACTION_OBJECTIVE_INSTRUCTION_MISMATCH", self._validation_codes(blueprint, source_map, manifest))

    def test_action_requirement_stays_with_the_unit_objective_scope(self) -> None:
        blueprint, _source_map, _manifest_value = self._finalized()
        lesson = blueprint["chapters"][0]["lessons"][0]  # type: ignore[index]
        informational = lesson["units"][0]
        informational["purpose"] = "Identify the documented safety control."
        informational["learning_objective_refs"] = ["lo_1"]
        informational["learning_blocks"][0]["learning_objective_refs"] = ["lo_1"]
        action = {
            **informational,
            "title": "Perform the documented procedure",
            "purpose": "Perform the documented procedure safely.",
            "learning_objective_refs": ["lo_2"],
            "learning_blocks": [
                {
                    **informational["learning_blocks"][0],
                    "id": "lb_action",
                    "learning_objective_refs": ["lo_2"],
                },
            ],
        }
        lesson["objective"] = "Perform the full documented procedure."
        lesson["learning_objectives"] = [
            "Identify the documented safety control.",
            "Perform the documented procedure safely.",
        ]
        lesson["units"].append(action)

        issues = validate_v5_instructional_coherence(blueprint).errors
        action_paths = [
            issue["path"]
            for issue in issues
            if issue["code"] == "ACTION_OBJECTIVE_INSTRUCTION_MISMATCH"
        ]
        self.assertEqual(action_paths, ["chapter_1.lesson_1.unit_2"])

    def test_v5_approved_draft_contract_preserves_scope_semantics_for_staged_generation(self) -> None:
        architecture = RagLessonAuthorDraftArchitecture.model_validate({
            "architecture_contract_version": 5,
            "chapter_title": "Chapter",
            "lessons": [{
                "title": "Lesson", "learning_objectives": ["Identify the control."],
                "assessment_required": True, "assessment_objective_refs": ["lo_1"],
                "units": [{
                    "title": "Teach", "purpose": "Teach the control.", "concept_ids": ["concept_1"],
                    "primary_evidence_scope_ids": ["scope_primary"],
                    "supporting_evidence_scope_ids": ["scope_support"],
                    "learning_objective_refs": ["lo_1"], "source_fact_ids": ["fact_1"],
                    "learning_blocks": [{
                        "id": "lb_1", "intent": "concept_explanation", "learning_objective_refs": ["lo_1"],
                        "primary_evidence_scope_ids": ["scope_primary"],
                        "supporting_evidence_scope_ids": ["scope_support"], "source_fact_ids": ["fact_1"],
                    }],
                }],
            }],
        })
        contract = format_approved_lesson_quality_contract(architecture)
        self.assertIn('"primary_evidence_scope_ids":["scope_primary"]', contract)
        self.assertIn('"supporting_evidence_scope_ids":["scope_support"]', contract)


if __name__ == "__main__":
    unittest.main()
