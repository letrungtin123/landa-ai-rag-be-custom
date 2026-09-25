from __future__ import annotations

import copy
import json
import unittest
from unittest.mock import AsyncMock

from google import genai
from google.genai import models, types

from app.assessment_planner import compile_v5_assessment_plan
from app.lesson_author_blueprint import build_v5_semantic_delta_repair_response_schema
from app.main import (
    _architecture_repair_preserves_unaffected_snapshot,
    _v5_deterministic_assessment_alignment_payload,
    _v5_prepare_semantic_repair_targets,
    allocate_source_map_architecture_facts,
    apply_course_architecture_repair_patches,
)
from app.source_map import build_source_map
from app.workflows.course_architecture import classify_course_repair_targets
from app.workflows.course_architecture import CourseArchitectureWorkflowCallbacks, run_course_architecture_workflow
from app.workflows.contracts import WorkflowFailure, WorkflowGenerationResult, WorkflowValidationResult, safe_workflow_issue_summary
from tests.test_evidence_scope_allocation_v5 import _manifest, _nodes, _v5_blueprint


class V5AssessmentPlannerTests(unittest.TestCase):
    def _alignment_and_missing_objectives(self, *, terminal: bool = True):
        candidate, source_map, manifest = self._candidate(facts=371, sections=6)
        lesson = candidate['chapters'][0]['lessons'][0]
        unit = lesson['units'][0]
        teaching = unit['learning_blocks'][0]
        lesson['learning_objectives'] = ['Identify A.', 'Explain B.', 'Describe C.']
        self._enable_assessment(lesson, ['lo_1', 'lo_2', 'lo_3'])
        unit['learning_objective_refs'] = ['lo_1', 'lo_2'] if terminal else ['lo_1', 'lo_2', 'lo_3']
        teaching['learning_objective_refs'] = ['lo_2'] if terminal else ['lo_2', 'lo_3']
        unit['learning_blocks'].append({
            'id': 'existing_check', 'intent': 'knowledge_check', 'importance': 'assessment',
            'concept_ids': list(teaching['concept_ids']), 'primary_concept_ids': [],
            'primary_evidence_scope_ids': [], 'supporting_evidence_scope_ids': [],
            'source_refs': list(teaching['source_refs']), 'learning_objective_refs': ['lo_1'],
            'content': {},
        })
        candidate['component_capabilities'] = {
            'version': 2, 'max_components_per_unit': 4,
            'max_assessments_per_unit': 3, 'assessment_enabled': True,
        }
        return candidate, source_map, manifest

    def test_alignment_does_not_hide_missing_objective_terminal_gap(self) -> None:
        candidate, _, _ = self._alignment_and_missing_objectives()
        original = copy.deepcopy(candidate)
        compiled = compile_v5_assessment_plan(candidate)
        self.assertEqual(compiled.status, 'TERMINAL_GAP')
        self.assertEqual([(i.code, i.objective_ref) for i in compiled.issues], [
            ('ASSESSMENT_EXISTING_CHECK_ALIGNMENT_REQUIRED', 'lo_1'),
            ('ASSESSMENT_PLAN_NO_SAFE_ANCHOR', 'lo_3'),
        ])
        self.assertEqual(compiled.deterministic_insertions, 0)
        self.assertEqual(candidate, original)
        self.assertEqual(compiled.blueprint, original)
        gap = compiled.issues[1]
        self.assertFalse(gap.workflow_issue()['repairable'])
        self.assertEqual(gap.candidate_rejection_counts['OBJECTIVE_OUTSIDE_UNIT_SCOPE'], 2)
        self.assertEqual(gap.candidate_rejection_counts['NO_PRIMARY_EVIDENCE'], 1)
        summary = safe_workflow_issue_summary(gap.workflow_issue(), repairable=False)
        self.assertEqual(summary['objective_ids'], ['lo_3'])
        self.assertEqual(summary['assessment_candidate_rejection_counts'], gap.candidate_rejection_counts)

    def test_alignment_then_missing_safe_assessment_preserves_coverage_and_371_facts(self) -> None:
        candidate, source_map, manifest = self._alignment_and_missing_objectives(terminal=False)
        original = copy.deepcopy(candidate)
        compiled = compile_v5_assessment_plan(candidate)
        self.assertEqual(compiled.status, 'NEEDS_SEMANTIC_RESOLUTION')
        self.assertEqual(compiled.deterministic_insertions, 0)
        issues = [dict(i.workflow_issue(), repair_layer='PRE_ALLOCATION_COHERENCE') for i in compiled.issues]
        targets = _v5_prepare_semantic_repair_targets(candidate, classify_course_repair_targets(
            issues, repair_layer='PRE_ALLOCATION_COHERENCE',
        ), source_map)
        payload = _v5_deterministic_assessment_alignment_payload(targets)
        self.assertIsNotNone(payload)
        repaired = apply_course_architecture_repair_patches(candidate, targets, payload)
        final = compile_v5_assessment_plan(repaired)
        self.assertEqual(final.status, 'READY')
        self.assertEqual(final.deterministic_insertions, 1)
        blocks = final.blueprint['chapters'][0]['lessons'][0]['units'][0]['learning_blocks']
        checks = [b for b in blocks if b['intent'] == 'knowledge_check']
        self.assertEqual(len(checks), 2)
        self.assertEqual(set().union(*(set(b['learning_objective_refs']) for b in checks)), {'lo_1', 'lo_2', 'lo_3'})
        self.assertTrue(all(not b['primary_evidence_scope_ids'] for b in checks))
        self.assertEqual(candidate, original)
        self.assertEqual(compile_v5_assessment_plan(final.blueprint).blueprint, final.blueprint)
        allocated = allocate_source_map_architecture_facts(final.blueprint, source_map, manifest)
        self.assertEqual(allocated['source_fact_allocation']['allocated_count'], 371)
        self.assertEqual(allocated['source_fact_allocation']['unallocated'], [])

    def test_anchor_diagnostics_are_bounded_and_do_not_contain_provider_content(self) -> None:
        candidate, _, _ = self._alignment_and_missing_objectives()
        lesson = candidate['chapters'][0]['lessons'][0]
        unit = lesson['units'][0]
        marker = 'PRIVATE-DO-NOT-LOG'
        unit['learning_blocks'][0]['id'] = marker
        unit['learning_blocks'][0]['content'] = {'purpose': marker}
        extra = copy.deepcopy(unit['learning_blocks'][1])
        extra['intent'] = 'reflection'
        extra['id'] = marker
        unit['learning_blocks'].extend(copy.deepcopy(extra) for _ in range(20))
        # A large synthetic in-memory candidate exercises telemetry bounds;
        # this is not a claim that this oversized structure passes schema.
        candidate['chapters'][0]['lessons'] = [copy.deepcopy(lesson) for _ in range(20)]
        compiled = compile_v5_assessment_plan(candidate)
        self.assertEqual(len(compiled.objective_diagnostics), 48)
        self.assertEqual(compiled.objective_diagnostic_count, 60)
        for row in compiled.objective_diagnostics:
            self.assertEqual(len(row['candidates']), 12)
            self.assertEqual(row['omitted_candidate_count'], 10)
            self.assertEqual(row['evaluated_block_count'], 22)
        self.assertNotIn(marker, json.dumps(compiled.objective_diagnostics))
        self.assertNotIn('source_refs', json.dumps(compiled.objective_diagnostics))
        self.assertEqual(compiled.objective_diagnostics, compile_v5_assessment_plan(candidate).objective_diagnostics)

    def test_alignment_and_missing_semantic_objective_are_both_planned_without_widening_authority(self) -> None:
        candidate, source_map, manifest = self._alignment_and_missing_objectives(terminal=False)
        unit = candidate['chapters'][0]['lessons'][0]['units'][0]
        # lo_3 is within the unit, but not yet taught by the only base anchor.
        unit['learning_blocks'][0]['learning_objective_refs'] = ['lo_2']
        compiled = compile_v5_assessment_plan(candidate)
        self.assertEqual(compiled.status, 'NEEDS_SEMANTIC_RESOLUTION')
        self.assertEqual({(i.code, i.objective_ref) for i in compiled.issues}, {
            ('ASSESSMENT_EXISTING_CHECK_ALIGNMENT_REQUIRED', 'lo_1'),
            ('ASSESSMENT_SEMANTIC_SELECTION_REQUIRED', 'lo_3'),
        })
        issues = [dict(i.workflow_issue(), repair_layer='PRE_ALLOCATION_COHERENCE') for i in compiled.issues]
        targets = _v5_prepare_semantic_repair_targets(candidate, classify_course_repair_targets(
            issues, repair_layer='PRE_ALLOCATION_COHERENCE',
        ), source_map)
        alignment = next(t for t in targets if t['scope'] == 'unit')
        selection = next(t for t in targets if t['scope'] == 'lesson')
        self.assertEqual(alignment['allowed_objective_ids'], ['lo_1'])
        self.assertEqual(selection['allowed_objective_ids'], ['lo_3'])
        payload = _v5_deterministic_assessment_alignment_payload([alignment])
        choice = selection['assessment_plan_candidates'][0]
        payload['patches'].append({
            'path': selection['path'], 'operation': 'select_assessment_teaching_alignment',
            'selections': [{
                'objective_ref': 'lo_3', 'decision': 'SELECT',
                'unit_path': choice['unit_path'], 'teaching_block_id': choice['teaching_block_id'],
            }],
        })
        repaired = apply_course_architecture_repair_patches(candidate, targets, payload)
        final = compile_v5_assessment_plan(repaired)
        self.assertEqual(final.status, 'READY')
        allocated = allocate_source_map_architecture_facts(final.blueprint, source_map, manifest)
        self.assertEqual(allocated['source_fact_allocation']['allocated_count'], 371)

    def test_safe_summary_rejects_untrusted_reasons_and_objective_text(self) -> None:
        summary = safe_workflow_issue_summary({
            'code': 'ASSESSMENT_PLAN_NO_SAFE_ANCHOR', 'path': 'chapter_1.lesson_1',
            'objective_ids': ['lo_3', 'private objective text', 'lo_0'],
            'assessment_candidate_rejection_counts': {
                'NO_PRIMARY_EVIDENCE': 2, 'private text': 1, 'NOT_PRECEDING': True,
                'CONCEPT_MISMATCH': -1,
            },
        }, repairable=False)
        self.assertEqual(summary['objective_ids'], ['lo_3'])
        self.assertEqual(summary['assessment_candidate_rejection_counts'], {'NO_PRIMARY_EVIDENCE': 2})

    def test_nested_snapshot_targets_are_order_independent_and_unrelated_changes_rejected(self) -> None:
        baseline, _, _ = self._alignment_and_missing_objectives(terminal=False)
        candidate = copy.deepcopy(baseline)
        candidate['chapters'][0]['lessons'][0]['units'][0]['learning_blocks'][0]['learning_objective_refs'].append('lo_1')
        lesson = 'chapter_1.lesson_1'
        unit = lesson + '.unit_1'
        # Explicit order sequences exercise both former set iteration orders.
        for paths in ([lesson, unit], [unit, lesson], {lesson, unit}):
            self.assertTrue(_architecture_repair_preserves_unaffected_snapshot(
                baseline, candidate, replacement_paths=paths, removal_paths=set(),
            ))
            bad = copy.deepcopy(candidate)
            bad['chapters'][1]['lessons'][0]['title'] = 'Unauthorized change'
            self.assertFalse(_architecture_repair_preserves_unaffected_snapshot(
                baseline, bad, replacement_paths=paths, removal_paths=set(),
            ))
        self.assertFalse(_architecture_repair_preserves_unaffected_snapshot(
            baseline, candidate, replacement_paths={lesson, lesson + '.unit_99'}, removal_paths=set(),
        ))

    def test_coverage_diagnostics_distinguish_declared_checks_from_missing_objectives(self) -> None:
        candidate, _, _ = self._alignment_and_missing_objectives()
        compiled = compile_v5_assessment_plan(candidate)
        unit = compiled.safe_unit_diagnostics(candidate)[0]
        self.assertEqual(unit['declared_assessed_objective_ids'], ['lo_1'])
        self.assertEqual(unit['missing_assessment_objective_ids'], ['lo_2', 'lo_3'])
        diagnostics = compiled.objective_diagnostics
        self.assertEqual(diagnostics[0]['assessment_state'], 'EXISTING_CHECK')
        self.assertFalse(diagnostics[0]['grounded'])
        self.assertEqual(diagnostics[2]['objective_id'], 'lo_3')
        self.assertEqual(diagnostics[2]['base_eligible_count'], 0)

    def test_profile_two_preserves_two_assessments_in_one_unit(self) -> None:
        candidate, source_map, manifest = self._candidate(facts=90, sections=1)
        lesson = candidate['chapters'][0]['lessons'][0]
        unit = lesson['units'][0]
        second = copy.deepcopy(unit['learning_blocks'][0])
        second['id'] = 'second_teaching'
        second['learning_objective_refs'] = ['lo_2']
        scopes = unit['learning_blocks'][0]['primary_evidence_scope_ids']
        self.assertGreater(len(scopes), 1)
        unit['learning_blocks'][0]['primary_evidence_scope_ids'] = scopes[:1]
        second['primary_evidence_scope_ids'] = scopes[1:]
        unit['learning_blocks'].append(second)
        unit['learning_objective_refs'] = ['lo_1', 'lo_2']
        lesson['learning_objectives'] = ['Identify A.', 'Explain B.']
        self._enable_assessment(lesson, ['lo_1', 'lo_2'])
        candidate['component_capabilities'] = {
            'version': 2, 'max_components_per_unit': 4,
            'max_assessments_per_unit': 3, 'assessment_enabled': True,
        }
        compiled = compile_v5_assessment_plan(candidate)
        self.assertEqual(compiled.status, 'READY')
        checks = [b for b in compiled.blueprint['chapters'][0]['lessons'][0]['units'][0]['learning_blocks'] if b['intent'] == 'knowledge_check']
        self.assertEqual(len(checks), 2)
        self.assertEqual([b['learning_objective_refs'] for b in checks], [['lo_1'], ['lo_2']])
        self.assertEqual(compile_v5_assessment_plan(compiled.blueprint).blueprint, compiled.blueprint)
        from app.main import validate_course_architecture_evidence_scope, validate_v5_instructional_coherence
        self.assertFalse(validate_course_architecture_evidence_scope(compiled.blueprint, source_map).errors)
        self.assertFalse(validate_v5_instructional_coherence(compiled.blueprint).errors)
        allocated = allocate_source_map_architecture_facts(compiled.blueprint, source_map, manifest)
        self.assertEqual(allocated['source_fact_allocation']['allocated_count'], 90)

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

    def test_existing_check_without_prior_teaching_is_not_reported_ready(self) -> None:
        candidate, _source_map, _manifest_value = self._candidate()
        lesson = candidate["chapters"][0]["lessons"][0]  # type: ignore[index]
        unit = lesson["units"][0]
        teaching = unit["learning_blocks"][0]
        unit["learning_blocks"] = [{
            "id": "lb_existing_check_only", "intent": "knowledge_check",
            "importance": "assessment", "concept_ids": list(teaching["concept_ids"]),
            "primary_concept_ids": [], "primary_evidence_scope_ids": [],
            "supporting_evidence_scope_ids": [], "source_refs": list(teaching["source_refs"]),
            "learning_objective_refs": ["lo_1"], "content": {},
        }]
        self._enable_assessment(lesson, ["lo_1"])

        compiled = compile_v5_assessment_plan(candidate)

        self.assertEqual(compiled.status, "TERMINAL_GAP")
        self.assertEqual(compiled.issues[0].code, "ASSESSMENT_PLAN_NO_SAFE_ANCHOR")
        self.assertEqual(compiled.issues[0].safe_reason, "NO_SAFE_TEACHING_ANCHOR_FOR_EXISTING_CHECK")
        self.assertFalse(compiled.issues[0].workflow_issue()["repairable"])

    def test_existing_check_before_teaching_fails_closed_with_order_reason(self) -> None:
        candidate, _source_map, _manifest_value = self._candidate()
        lesson = candidate["chapters"][0]["lessons"][0]  # type: ignore[index]
        unit = lesson["units"][0]
        teaching = unit["learning_blocks"][0]
        unit["learning_blocks"] = [{
            "id": "lb_check_first", "intent": "knowledge_check", "importance": "assessment",
            "concept_ids": list(teaching["concept_ids"]), "primary_concept_ids": [],
            "primary_evidence_scope_ids": [], "supporting_evidence_scope_ids": [],
            "source_refs": list(teaching["source_refs"]), "learning_objective_refs": ["lo_1"], "content": {},
        }, teaching]
        self._enable_assessment(lesson, ["lo_1"])

        compiled = compile_v5_assessment_plan(candidate)

        self.assertEqual(compiled.status, "TERMINAL_GAP")
        self.assertGreaterEqual(compiled.issues[0].candidate_rejection_counts.get("NOT_PRECEDING", 0), 1)

    def test_existing_check_with_base_anchor_requires_scoped_reconciliation(self) -> None:
        candidate, _source_map, _manifest_value = self._candidate()
        lesson = candidate["chapters"][0]["lessons"][0]  # type: ignore[index]
        unit = lesson["units"][0]
        teaching = unit["learning_blocks"][0]
        teaching["learning_objective_refs"] = []
        unit["learning_blocks"].append({
            "id": "lb_existing_check", "intent": "knowledge_check", "importance": "assessment",
            "concept_ids": list(teaching["concept_ids"]), "primary_concept_ids": [],
            "primary_evidence_scope_ids": [], "supporting_evidence_scope_ids": [],
            "source_refs": list(teaching["source_refs"]), "learning_objective_refs": ["lo_1"], "content": {},
        })
        self._enable_assessment(lesson, ["lo_1"])

        compiled = compile_v5_assessment_plan(candidate)

        self.assertEqual(compiled.status, "NEEDS_SEMANTIC_RESOLUTION")
        issue = compiled.issues[0]
        self.assertEqual(issue.code, "ASSESSMENT_EXISTING_CHECK_ALIGNMENT_REQUIRED")
        self.assertTrue(issue.workflow_issue()["repairable"])
        self.assertEqual(issue.workflow_issue()["path"], "chapter_1.lesson_1.unit_1")
        self.assertEqual(issue.workflow_issue()["learning_block_ids"], ["lb_existing_check"])

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


class AssessmentGapWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_terminal_missing_anchor_blocks_scheduler_before_any_repair_or_allocation(self) -> None:
        fixture = V5AssessmentPlannerTests()
        candidate, source_map, _ = fixture._alignment_and_missing_objectives()
        provider = AsyncMock(side_effect=AssertionError('must not call repair provider'))
        deterministic = AsyncMock(side_effect=AssertionError('must not repair past a terminal gap'))
        events = []

        def validate(blueprint, _source_map):
            result = compile_v5_assessment_plan(blueprint)
            return WorkflowValidationResult([
                dict(issue.workflow_issue(), repair_layer='PRE_ALLOCATION_COHERENCE')
                for issue in result.issues
            ], result.metrics)

        with self.assertRaises(WorkflowFailure) as raised:
            await run_course_architecture_workflow(CourseArchitectureWorkflowCallbacks(
                validate_source_scope=lambda: [], build_source_map=lambda: source_map,
                architect=AsyncMock(return_value=WorkflowGenerationResult(copy.deepcopy(candidate))),
                validate_blueprint=validate, repair_blueprint=provider,
                deterministic_repair=deterministic, emit_diagnostic=events.append,
            ), request_context={'correlation_id': 'synthetic-assessment-gap'}, max_repair_attempts=2)
        provider.assert_not_awaited()
        deterministic.assert_not_awaited()
        self.assertEqual(raised.exception.failure_stage, 'blueprint_validation')
        self.assertTrue(any(i.get('code') == 'ASSESSMENT_PLAN_NO_SAFE_ANCHOR' for i in raised.exception.issues))
        self.assertFalse(any(e.get('stage') == 'repair_target_generation' for e in events))
        self.assertNotIn('source_fact_allocation', candidate)


if __name__ == "__main__":
    unittest.main()
