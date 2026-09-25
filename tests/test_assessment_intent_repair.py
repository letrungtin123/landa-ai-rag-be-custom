"""Synthetic regression for a primary owner rejected only by teaching intent.

No private UAT output or live provider calls. The original UAT intent was not
logged; introduction is one representative canonical non-teaching intent.
"""
import copy
import json
import unittest
from unittest.mock import AsyncMock

from google import genai
from google.genai import models, types

from app.assessment_planner import (
    assessment_intent_repair_options, compile_v5_assessment_plan,
    evaluate_assessment_teaching_anchor,
)
from app.lesson_author_blueprint import build_v5_semantic_delta_repair_response_schema
from app.main import (
    _semantic_delta_target_snapshot, _v5_deterministic_assessment_alignment_payload,
    _v5_prepare_semantic_repair_targets, allocate_source_map_architecture_facts,
    apply_course_architecture_repair_patches, validate_course_architecture_evidence_scope,
    validate_v5_instructional_coherence,
)
from app.source_map import build_source_map
from app.workflows.contracts import WorkflowFailure, WorkflowGenerationResult, WorkflowValidationResult
from app.workflows.course_architecture import (
    CourseArchitectureWorkflowCallbacks, V5_MAX_PROVIDER_REPAIR_CALLS,
    classify_course_repair_targets, run_course_architecture_workflow,
)
from tests.test_evidence_scope_allocation_v5 import _manifest, _nodes, _v5_blueprint


def fixture(facts=371):
    manifest = _manifest(facts, 6)
    source_map = build_source_map(_nodes(6), manifest, locale='en')
    candidate = _v5_blueprint(source_map)
    lesson = candidate['chapters'][0]['lessons'][0]
    refs = ['lo_1', 'lo_2', 'lo_3']
    lesson.update(learning_objectives=['Identify A.', 'Explain B.', 'Describe C.'],
                  assessment_required=True, assessment_objective_refs=refs.copy())
    unit = lesson['units'][0]
    unit['learning_objective_refs'] = refs.copy()
    teaching = unit['learning_blocks'][0]
    teaching.update(intent='introduction', learning_objective_refs=refs.copy(),
                    content={'purpose': 'Explain the evidence-backed definitions.',
                             'learner_action': 'Identify and describe the concepts.'})
    check = copy.deepcopy(teaching)
    check.update(id='check_existing', intent='knowledge_check', importance='assessment',
                 primary_concept_ids=[], primary_evidence_scope_ids=[],
                 supporting_evidence_scope_ids=teaching['primary_evidence_scope_ids'].copy(), content={})
    unit['learning_blocks'].append(check)
    return candidate, source_map, manifest


def unit_of(candidate):
    return candidate['chapters'][0]['lessons'][0]['units'][0]


def targets_for(candidate, source_map):
    compiled = compile_v5_assessment_plan(candidate)
    issues = [dict(i.workflow_issue(), repair_layer='PRE_ALLOCATION_COHERENCE') for i in compiled.issues]
    return _v5_prepare_semantic_repair_targets(candidate, classify_course_repair_targets(
        issues, repair_layer='PRE_ALLOCATION_COHERENCE'), source_map)


def payload_for(targets):
    return {'patches': [{
        'path': target['path'], 'operation': 'repair_assessment_alignment',
        'knowledge_check_block_id': target['allowed_block_ids'][0],
        'teaching_selections': [{
            'teaching_block_id': c['teaching_block_id'],
            'learning_objective_refs': c['learning_objective_refs'].copy(),
            **({'intent': 'concept_explanation'} if c.get('allowed_intents') else {}),
        } for c in target['assessment_alignment_candidates']],
    } for target in targets]}


class AssessmentIntentRepairTests(unittest.TestCase):
    def test_potential_repair_does_not_relax_anchor_predicate(self):
        candidate, source_map, _ = fixture()
        teaching, check = unit_of(candidate)['learning_blocks']
        kwargs = dict(teaching_block=teaching, knowledge_check_block=check,
                      objective_refs={'lo_1'}, unit_objective_refs={'lo_1'}, precedes_check=True)
        eligibility = evaluate_assessment_teaching_anchor(**kwargs)
        self.assertFalse(eligibility.base_eligible)
        self.assertFalse(eligibility.fully_aligned)
        self.assertEqual(eligibility.reasons, ('INTENT_NOT_TEACHING',))
        self.assertIn('concept_explanation', assessment_intent_repair_options(**kwargs, same_unit=True))
        compiled = compile_v5_assessment_plan(candidate)
        self.assertEqual(compiled.status, 'NEEDS_SEMANTIC_RESOLUTION')
        self.assertEqual(len(compiled.issues), 3)
        self.assertTrue(all(i.code == 'ASSESSMENT_EXISTING_CHECK_ALIGNMENT_REQUIRED' and i.workflow_issue()['repairable'] for i in compiled.issues))
        self.assertTrue(all(d['intent_repair_candidate_count'] == 1 for d in compiled.objective_diagnostics))
        targets = targets_for(candidate, source_map)
        self.assertEqual(len(targets), 1)
        self.assertIsNone(_v5_deterministic_assessment_alignment_payload(targets))

    def test_repaired_three_objectives_revalidate_and_allocate_371_once(self):
        candidate, source_map, manifest = fixture()
        baseline = copy.deepcopy(candidate)
        targets = targets_for(candidate, source_map)
        repaired = apply_course_architecture_repair_patches(candidate, targets, payload_for(targets))
        expected = copy.deepcopy(baseline)
        unit_of(expected)['learning_blocks'][0]['intent'] = 'concept_explanation'
        self.assertEqual(repaired, expected)
        self.assertEqual(candidate, baseline)
        self.assertFalse(validate_course_architecture_evidence_scope(repaired, source_map).errors)
        self.assertFalse(validate_v5_instructional_coherence(repaired).errors)
        self.assertEqual(compile_v5_assessment_plan(repaired).status, 'READY')
        allocated = allocate_source_map_architecture_facts(repaired, source_map, manifest)
        allocation = allocated['source_fact_allocation']
        self.assertEqual(allocation['allocated_count'], 371)
        self.assertEqual(allocation['unallocated'], [])
        self.assertTrue(allocation['complete'])
        self.assertEqual(len({a['fact_id'] for a in allocation['allocations']}), 371)

    def test_source_order_scope_and_non_instructional_roles_still_fail_closed(self):
        cases = {
            'primary_missing': lambda t, c: t.update(primary_evidence_scope_ids=[]),
            'source_mismatch': lambda t, c: t.update(source_refs=['other_source']),
            'concept_mismatch': lambda t, c: t.update(concept_ids=['other_concept']),
            'unknown_intent': lambda t, c: t.update(intent='arbitrary_role'),
            'assessment_role': lambda t, c: t.update(intent='knowledge_check'),
            'media_role': lambda t, c: t.update(intent='media_reference'),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name):
                candidate, _, _ = fixture()
                teaching, check = unit_of(candidate)['learning_blocks']
                mutate(teaching, check)
                self.assertEqual(assessment_intent_repair_options(
                    teaching_block=teaching, knowledge_check_block=check,
                    objective_refs={'lo_1'}, unit_objective_refs={'lo_1'},
                    precedes_check=True, same_unit=True), ())
                self.assertEqual(compile_v5_assessment_plan(candidate).status, 'TERMINAL_GAP')
        for before, same_unit in [(False, True), (True, False)]:
            candidate, _, _ = fixture()
            teaching, check = unit_of(candidate)['learning_blocks']
            self.assertFalse(assessment_intent_repair_options(
                teaching_block=teaching, knowledge_check_block=check,
                objective_refs={'lo_1'}, unit_objective_refs={'lo_1'},
                precedes_check=before, same_unit=same_unit))

    def test_cross_unit_role_change_is_not_authorized(self):
        candidate, _, _ = fixture()
        lesson = candidate['chapters'][0]['lessons'][0]
        new_unit = copy.deepcopy(unit_of(candidate))
        new_unit['learning_blocks'] = [unit_of(candidate)['learning_blocks'].pop()]
        lesson['units'].append(new_unit)
        self.assertEqual(compile_v5_assessment_plan(candidate).status, 'TERMINAL_GAP')

    def test_invalid_deltas_are_typed_and_transactional(self):
        candidate, source_map, _ = fixture()
        baseline = copy.deepcopy(candidate)
        targets = targets_for(candidate, source_map)
        for name in ['missing_intent', 'invalid_intent', 'conflicting_intent', 'fact_injection', 'wrong_block', 'wrong_objective', 'wrong_path']:
            with self.subTest(name=name):
                payload = payload_for(targets)
                patch = payload['patches'][0]
                selection = patch['teaching_selections'][0]
                if name == 'missing_intent':
                    selection.pop('intent')
                elif name == 'invalid_intent':
                    selection['intent'] = 'knowledge_check'
                elif name == 'conflicting_intent':
                    patch['teaching_selections'][1]['intent'] = 'definition'
                elif name == 'fact_injection':
                    selection['source_fact_ids'] = ['invented']
                elif name == 'wrong_block':
                    selection['teaching_block_id'] = 'lb_2'
                elif name == 'wrong_objective':
                    selection['learning_objective_refs'] = ['lo_99']
                else:
                    patch['path'] = 'chapter_2.lesson_1.unit_1'
                with self.assertRaises(WorkflowFailure) as raised:
                    apply_course_architecture_repair_patches(candidate, targets, payload)
                if 'intent' in name:
                    self.assertEqual(raised.exception.internal_code, 'ARCH_REPAIR_INVALID_SEMANTIC_INTENT')
                self.assertEqual(candidate, baseline)

    def test_apply_rechecks_evidence_instead_of_trusting_preflight(self):
        candidate, source_map, _ = fixture()
        targets = targets_for(candidate, source_map)
        unit_of(candidate)['learning_blocks'][0]['source_refs'] = ['changed_source']
        baseline = copy.deepcopy(candidate)
        with self.assertRaises(WorkflowFailure) as raised:
            apply_course_architecture_repair_patches(candidate, targets, payload_for(targets))
        self.assertEqual(raised.exception.internal_code, 'ARCH_REPAIR_NO_COMPATIBLE_SUPPORTING_EVIDENCE')
        self.assertEqual(candidate, baseline)

    def test_ordinary_alignment_payload_remains_compatible_without_intent(self):
        candidate, source_map, _ = fixture()
        teaching = unit_of(candidate)['learning_blocks'][0]
        teaching.update(intent='concept_explanation', learning_objective_refs=[])
        targets = targets_for(candidate, source_map)
        self.assertTrue(all(not c.get('allowed_intents') for c in targets[0]['assessment_alignment_candidates']))
        repaired = apply_course_architecture_repair_patches(candidate, targets, payload_for(targets))
        self.assertEqual(compile_v5_assessment_plan(repaired).status, 'READY')
        invalid = payload_for(targets)
        invalid['patches'][0]['teaching_selections'][0]['intent'] = 'definition'
        with self.assertRaises(WorkflowFailure):
            apply_course_architecture_repair_patches(candidate, targets, invalid)

    def test_provider_context_bounded_and_not_in_diagnostics(self):
        candidate, source_map, _ = fixture()
        marker = 'PRIVATE-INSTRUCTIONAL-DESCRIPTOR'
        unit_of(candidate)['learning_blocks'][0]['content']['purpose'] = marker * 100
        targets = targets_for(candidate, source_map)
        snapshot = _semantic_delta_target_snapshot(unit_of(candidate), targets[0])
        for c in snapshot['assessment_alignment_candidates']:
            self.assertLessEqual(len(c['semantic_descriptor']['purpose']), 480)
            self.assertIn('objective_text', c)
        serialized = json.dumps(snapshot)
        self.assertNotIn('source_fact_ids', serialized)
        self.assertNotIn('primary_evidence_scope_ids', serialized)
        diagnostics = json.dumps(compile_v5_assessment_plan(candidate).objective_diagnostics)
        self.assertNotIn(marker, diagnostics)
        self.assertIn('INTENT_NOT_TEACHING', diagnostics)

    def test_selected_schema_serializes_with_optional_bounded_intent(self):
        schema = build_v5_semantic_delta_repair_response_schema({'repair_assessment_alignment'})
        selection = schema.properties['patches'].items.properties['teaching_selections'].items
        self.assertNotIn('intent', selection.required)
        self.assertEqual(len(selection.properties['intent'].enum), 8)
        self.assertNotIn('source_fact_ids', selection.properties)
        client = genai.Client(api_key='test-key')
        payload = models._GenerateContentConfig_to_mldev(client._api_client,
            types.GenerateContentConfig(response_mime_type='application/json', response_schema=schema))
        self.assertIn('responseSchema', payload)


class AssessmentIntentWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def run_fixture(self, mode):
        candidate, source_map, manifest = fixture()
        if mode == 'valid':
            unit_of(candidate)['learning_blocks'][0]['intent'] = 'concept_explanation'
        validations = []
        allocations = []

        def validate(blueprint, source):
            compiled = compile_v5_assessment_plan(blueprint)
            validations.append(compiled.status)
            issues = [dict(i.workflow_issue(), repair_layer='PRE_ALLOCATION_COHERENCE') for i in compiled.issues]
            if not issues:
                self.assertFalse(validate_course_architecture_evidence_scope(blueprint, source).errors)
                self.assertFalse(validate_v5_instructional_coherence(blueprint).errors)
                allocations.append(allocate_source_map_architecture_facts(blueprint, source, manifest))
            return WorkflowValidationResult(issues, compiled.metrics)

        async def repair(blueprint, targets, source):
            if mode == 'no_progress':
                return WorkflowGenerationResult(copy.deepcopy(blueprint))
            return WorkflowGenerationResult(apply_course_architecture_repair_patches(blueprint, targets, payload_for(targets)))

        provider = AsyncMock(side_effect=repair)
        callbacks = CourseArchitectureWorkflowCallbacks(
            validate_source_scope=lambda: [], build_source_map=lambda: source_map,
            architect=AsyncMock(return_value=WorkflowGenerationResult(candidate)),
            validate_blueprint=validate, repair_blueprint=provider,
            prepare_repair_targets=_v5_prepare_semantic_repair_targets,
        )
        self.assertEqual(V5_MAX_PROVIDER_REPAIR_CALLS, 3)
        if mode == 'no_progress':
            with self.assertRaises(WorkflowFailure) as raised:
                await run_course_architecture_workflow(callbacks, request_context={}, max_repair_attempts=2)
            self.assertEqual(raised.exception.internal_code, 'ARCH_REPAIR_NO_PROGRESS')
            self.assertFalse(allocations)
            self.assertEqual(provider.await_count, 1)
        else:
            await run_course_architecture_workflow(callbacks, request_context={}, max_repair_attempts=2)
            self.assertEqual(provider.await_count, 0 if mode == 'valid' else 1)
            self.assertEqual(validations[-1], 'READY')
            self.assertEqual(allocations[-1]['source_fact_allocation']['allocated_count'], 371)

    async def test_provider_choice_revalidates_then_allocates(self):
        await self.run_fixture('repair')

    async def test_valid_blueprint_bypasses_repair(self):
        await self.run_fixture('valid')

    async def test_no_progress_remains_bounded_and_cannot_finalize(self):
        await self.run_fixture('no_progress')
