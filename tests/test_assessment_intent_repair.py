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
    assessment_intent_repair_options,
    compile_v5_assessment_plan,
    evaluate_assessment_teaching_anchor,
)
from app.lesson_author_blueprint import build_v5_semantic_delta_repair_response_schema
from app.main import (
    _semantic_delta_target_snapshot,
    _v5_deterministic_assessment_alignment_payload,
    _v5_prepare_semantic_repair_targets,
    apply_course_architecture_repair_patches,
)
from app.services.lesson_author.architecture_validation import validate_v5_instructional_coherence
from app.services.lesson_author.evidence_scope import (
    allocate_source_map_architecture_facts,
    validate_course_architecture_evidence_scope,
)
from app.source_map import build_source_map
from app.workflows.contracts import WorkflowFailure, WorkflowGenerationResult, WorkflowValidationResult
from app.workflows.course_architecture import (
    V5_MAX_PROVIDER_REPAIR_CALLS,
    CourseArchitectureWorkflowCallbacks,
    classify_course_repair_targets,
    run_course_architecture_workflow,
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


def missing_check_fixture(intent='introduction', facts=415):
    candidate, source_map, manifest = fixture(facts)
    unit_of(candidate)['learning_blocks'].pop()
    unit_of(candidate)['learning_blocks'][0]['intent'] = intent
    return candidate, source_map, manifest


def missing_check_payload(targets):
    patches = []
    for target in targets:
        selections = []
        for objective in target['allowed_objective_ids']:
            option = next(c for c in target['assessment_plan_candidates'] if c['objective_ref'] == objective)
            selections.append({
                'objective_ref': objective, 'decision': 'SELECT',
                'unit_path': option['unit_path'], 'teaching_block_id': option['teaching_block_id'],
                **({'intent': 'concept_explanation'} if option.get('allowed_intents') else {}),
            })
        patches.append({'path': target['path'], 'operation': 'select_assessment_teaching_alignment',
                        'selections': selections})
    return {'patches': patches}


class MissingCheckIntentRepairTests(unittest.TestCase):
    def test_missing_check_requires_explicit_semantic_resolution_not_automatic_relabel(self):
        for intent in ['introduction', 'practice', 'reflection']:
            with self.subTest(intent=intent):
                candidate, source_map, _ = missing_check_fixture(intent)
                baseline = copy.deepcopy(candidate)
                compiled = compile_v5_assessment_plan(candidate)
                self.assertEqual(compiled.status, 'NEEDS_SEMANTIC_RESOLUTION')
                self.assertTrue(all(i.code == 'ASSESSMENT_SEMANTIC_SELECTION_REQUIRED' for i in compiled.issues))
                self.assertTrue(all(d['intent_repair_candidate_count'] == 1 for d in compiled.objective_diagnostics))
                self.assertEqual(candidate, baseline)
                self.assertEqual(unit_of(compiled.blueprint)['learning_blocks'][0]['intent'], intent)
                targets = targets_for(candidate, source_map)
                self.assertEqual(targets[0]['path'], 'chapter_1.lesson_1')
                self.assertTrue(all(c['allowed_intents'] for c in targets[0]['assessment_plan_candidates']))
                direct_selections = {(p, ref): (options[0].unit_path, options[0].block_id)
                                     for (p, ref), options in compiled.candidates.items()}
                self.assertEqual(compile_v5_assessment_plan(candidate, selections=direct_selections).status, 'TERMINAL_GAP')

    def test_delta_revalidates_preserves_provenance_and_allocates_once(self):
        for facts in [371, 415]:
            with self.subTest(facts=facts):
                candidate, source_map, manifest = missing_check_fixture(facts=facts)
                baseline = copy.deepcopy(candidate)
                targets = targets_for(candidate, source_map)
                repaired = apply_course_architecture_repair_patches(candidate, targets, missing_check_payload(targets))
                self.assertEqual(candidate, baseline)
                teaching, check = unit_of(repaired)['learning_blocks']
                expected = dict(unit_of(baseline)['learning_blocks'][0], intent='concept_explanation')
                self.assertEqual(teaching, expected)
                self.assertEqual(check['intent'], 'knowledge_check')
                self.assertEqual(check['primary_evidence_scope_ids'], [])
                self.assertEqual(set(check['supporting_evidence_scope_ids']), set(teaching['primary_evidence_scope_ids']))
                self.assertFalse(check.get('source_fact_ids'))
                self.assertEqual(repaired['chapters'][1:], baseline['chapters'][1:])
                self.assertEqual(compile_v5_assessment_plan(repaired).status, 'READY')
                self.assertFalse(validate_course_architecture_evidence_scope(repaired, source_map).errors)
                self.assertFalse(validate_v5_instructional_coherence(repaired).errors)
                allocation = allocate_source_map_architecture_facts(repaired, source_map, manifest)['source_fact_allocation']
                self.assertTrue(allocation['complete'])
                self.assertEqual(allocation['allocated_count'], facts)
                self.assertEqual(allocation['unallocated'], [])
                self.assertEqual(len({a['fact_id'] for a in allocation['allocations']}), facts)

    def test_other_missing_prerequisites_remain_terminal(self):
        for field, value in [('primary_evidence_scope_ids', []), ('concept_ids', []),
                             ('source_refs', []), ('intent', 'unknown'), ('intent', 'media_reference'), ('id', '')]:
            with self.subTest(field=field, value=value):
                candidate, _, _ = missing_check_fixture()
                unit_of(candidate)['learning_blocks'][0][field] = value
                self.assertEqual(compile_v5_assessment_plan(candidate).status, 'TERMINAL_GAP')
        candidate, _, _ = missing_check_fixture()
        unit_of(candidate)['learning_objective_refs'] = []
        unit_of(candidate)['learning_blocks'][0]['learning_objective_refs'] = []
        self.assertEqual(compile_v5_assessment_plan(candidate).status, 'TERMINAL_GAP')

    def test_valid_anchor_preferred_without_extra_repair(self):
        candidate, _, _ = missing_check_fixture()
        unit_of(candidate)['learning_blocks'][0]['intent'] = 'concept_explanation'
        compiled = compile_v5_assessment_plan(candidate)
        self.assertEqual(compiled.status, 'READY')
        self.assertFalse(compiled.candidates)
        self.assertEqual(len(unit_of(compiled.blueprint)['learning_blocks']), 2)

    def test_multiple_candidates_select_exact_unit_without_mutating_others(self):
        candidate, source_map, manifest = missing_check_fixture('practice')
        lesson = candidate['chapters'][0]['lessons'][0]
        first = unit_of(candidate)
        second = copy.deepcopy(first)
        second_block = second['learning_blocks'][0]
        second_block.update(id='reflection_owner', intent='reflection')
        scopes = first['learning_blocks'][0]['primary_evidence_scope_ids']
        self.assertGreater(len(scopes), 1)
        second_block['primary_evidence_scope_ids'] = scopes[1:]
        first['learning_blocks'][0]['primary_evidence_scope_ids'] = scopes[:1]
        lesson['units'].append(second)
        baseline = copy.deepcopy(candidate)
        compiled = compile_v5_assessment_plan(candidate)
        self.assertEqual(compiled.status, 'NEEDS_SEMANTIC_RESOLUTION')
        self.assertTrue(all(len(options) == 2 for options in compiled.candidates.values()))
        targets = targets_for(candidate, source_map)
        payload = missing_check_payload(targets)
        for selection in payload['patches'][0]['selections']:
            selection.update(unit_path='chapter_1.lesson_1.unit_2', teaching_block_id='reflection_owner')
        repaired = apply_course_architecture_repair_patches(candidate, targets, payload)
        self.assertEqual(unit_of(repaired), unit_of(baseline))
        self.assertEqual(candidate, baseline)
        selected_blocks = repaired['chapters'][0]['lessons'][0]['units'][1]['learning_blocks']
        self.assertEqual([b['intent'] for b in selected_blocks], ['concept_explanation', 'knowledge_check'])
        self.assertEqual(selected_blocks[0], dict(second_block, intent='concept_explanation'))
        self.assertFalse(validate_course_architecture_evidence_scope(repaired, source_map).errors)
        self.assertFalse(validate_v5_instructional_coherence(repaired).errors)
        self.assertEqual(allocate_source_map_architecture_facts(repaired, source_map, manifest)
                         ['source_fact_allocation']['allocated_count'], 415)

    def test_ordinary_missing_check_selection_cannot_change_intent(self):
        candidate, source_map, _ = missing_check_fixture('concept_explanation')
        unit_of(candidate)['learning_blocks'][0]['learning_objective_refs'] = []
        targets = targets_for(candidate, source_map)
        payload = missing_check_payload(targets)
        self.assertTrue(all('intent' not in s for s in payload['patches'][0]['selections']))
        repaired = apply_course_architecture_repair_patches(candidate, targets, payload)
        self.assertEqual(compile_v5_assessment_plan(repaired).status, 'READY')
        payload['patches'][0]['selections'][0]['intent'] = 'definition'
        with self.assertRaises(WorkflowFailure) as caught:
            apply_course_architecture_repair_patches(candidate, targets, payload)
        self.assertEqual(caught.exception.internal_code, 'ARCH_REPAIR_SCOPE_VIOLATION')

    def test_invalid_selections_are_transactional(self):
        candidate, source_map, _ = missing_check_fixture()
        baseline = copy.deepcopy(candidate)
        targets = targets_for(candidate, source_map)
        for case in ['missing_intent', 'bad_intent', 'conflicting_intent', 'wrong_block',
                     'wrong_unit', 'wrong_objective', 'provenance', 'no_match']:
            with self.subTest(case=case):
                payload = missing_check_payload(targets)
                selection = payload['patches'][0]['selections'][0]
                if case == 'missing_intent':
                    selection.pop('intent')
                elif case == 'bad_intent':
                    selection['intent'] = 'knowledge_check'
                elif case == 'conflicting_intent':
                    payload['patches'][0]['selections'][1]['intent'] = 'definition'
                elif case == 'wrong_block':
                    selection['teaching_block_id'] = 'unapproved'
                elif case == 'wrong_unit':
                    selection['unit_path'] = 'chapter_2.lesson_1.unit_1'
                elif case == 'wrong_objective':
                    selection['objective_ref'] = 'lo_99'
                elif case == 'provenance':
                    selection['primary_evidence_scope_ids'] = ['invented']
                else:
                    payload['patches'][0]['selections'][0] = {'objective_ref': selection['objective_ref'], 'decision': 'NO_MATCH'}
                with self.assertRaises(WorkflowFailure) as caught:
                    apply_course_architecture_repair_patches(candidate, targets, payload)
                if 'intent' in case:
                    self.assertEqual(caught.exception.internal_code, 'ARCH_REPAIR_INVALID_SEMANTIC_INTENT')
                if case == 'no_match':
                    self.assertEqual(caught.exception.internal_code, 'ARCH_REPAIR_ASSESSMENT_NO_MATCH')
                self.assertEqual(candidate, baseline)

    def test_candidate_fingerprint_rechecks_source_and_instructional_descriptor(self):
        for field, value in [('source_refs', ['different_source']), ('content', {'purpose': 'Different purpose'})]:
            with self.subTest(field=field):
                candidate, source_map, _ = missing_check_fixture()
                targets = targets_for(candidate, source_map)
                unit_of(candidate)['learning_blocks'][0][field] = value
                baseline = copy.deepcopy(candidate)
                with self.assertRaises(WorkflowFailure):
                    apply_course_architecture_repair_patches(candidate, targets, missing_check_payload(targets))
                self.assertEqual(candidate, baseline)

    def test_selection_schema_and_snapshot_only_expose_bounded_semantic_authority(self):
        candidate, source_map, _ = missing_check_fixture()
        target = targets_for(candidate, source_map)[0]
        snapshot = _semantic_delta_target_snapshot(candidate['chapters'][0]['lessons'][0], target)
        self.assertTrue(all(c.get('allowed_intents') for c in snapshot['assessment_plan_candidates']))
        schema = build_v5_semantic_delta_repair_response_schema({'select_assessment_teaching_alignment'})
        selection = schema.properties['patches'].items.properties['selections'].items
        self.assertNotIn('intent', selection.required)
        self.assertEqual(len(selection.properties['intent'].enum), 8)
        self.assertNotIn('source_fact_ids', selection.properties)
        client = genai.Client(api_key='test-key')
        payload = models._GenerateContentConfig_to_mldev(client._api_client,
            types.GenerateContentConfig(response_mime_type='application/json', response_schema=schema))
        self.assertIn('responseSchema', payload)


class AssessmentIntentWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def run_fixture(self, mode):
        missing = mode.startswith('missing_')
        candidate, source_map, manifest = missing_check_fixture(facts=371) if missing else fixture()
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
            if mode in {'no_progress', 'missing_no_progress'}:
                return WorkflowGenerationResult(copy.deepcopy(blueprint))
            payload = missing_check_payload(targets) if missing else payload_for(targets)
            return WorkflowGenerationResult(apply_course_architecture_repair_patches(blueprint, targets, payload))

        provider = AsyncMock(side_effect=repair)
        callbacks = CourseArchitectureWorkflowCallbacks(
            validate_source_scope=lambda: [], build_source_map=lambda: source_map,
            architect=AsyncMock(return_value=WorkflowGenerationResult(candidate)),
            validate_blueprint=validate, repair_blueprint=provider,
            prepare_repair_targets=_v5_prepare_semantic_repair_targets,
        )
        self.assertEqual(V5_MAX_PROVIDER_REPAIR_CALLS, 3)
        if mode in {'no_progress', 'missing_no_progress'}:
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

    async def test_missing_check_reuses_one_bounded_repair_call(self):
        await self.run_fixture('missing_repair')

    async def test_missing_check_no_progress_cannot_finalize(self):
        await self.run_fixture('missing_no_progress')
