"""Offline regression: HTML teaching-group gap + Diagram 26/32 coverage."""
import asyncio
from copy import deepcopy
import json
import unittest
from unittest.mock import AsyncMock, patch

from app import main
from app.lesson_author_provider_schema import staged_provider_response_model
from app.workflows.contracts import WorkflowFailure
from tests.test_staged_ordered_writer import uat_fixture
from tests.test_checkpoint_component_quality_repair import instance_wire
from tests.test_chapter_checkpoint import checkpoint_result
from tests.staged_schema_probe import capture_sdk_body
from tests import test_checkpoint_provider_sdk as sdk_fixture


def fixture():
    request, valid, scope, manifest = uat_fixture()
    # Keep the non-contiguous real failing target layout, using synthetic data.
    for items in (valid['components'], scope['component_plan']):
        items[2], items[3] = items[3], items[2]
    # An unchanged supporting ordering activity avoids placing FAQ before a
    # Diagram (Node correctly keeps FAQ last). No production plan is changed.
    valid['components'][2].update(type='la_sortable', question_text='Put the documented procedure in order.',
        items=[{'text': 'Inspect the documented conditions before starting.'},
               {'text': 'Perform the authorized task following the instructions.'},
               {'text': 'Record the verified result after completing the task.'}])
    scope['component_plan'][2].update(type='la_sortable', reason_code='ORDERING_PRACTICE')
    diagram, plan = valid['components'][3], scope['component_plan'][3]
    owned = valid['source_fact_ids'][:32]
    plan.update(source_fact_ids=owned[:], supporting_evidence_fact_ids=[], reason_code='RELATIONSHIP_VISUALIZATION')
    diagram.update(source_fact_ids=owned[:], covered_source_fact_ids=owned[:], supporting_evidence_fact_ids=[])
    scope['component_plan'][0]['learning_block_ids'] = ['teach_a']
    scope['learning_blocks'] = [{'id': 'teach_a', 'intent': 'concept_explanation',
        'learning_objective_refs': ['lo_1'], 'source_fact_ids': valid['source_fact_ids'][:]}]
    for section in valid['components'][0]['semantic_content']['sections']:
        section['learning_block_ids'] = ['teach_a']
    scope['component_types'] = [p['type'] for p in scope['component_plan']]
    architecture = request.blueprint_architecture.model_dump()
    architecture['lessons'][0]['units'][0] = scope
    request = request.model_copy(update={'blueprint_architecture': type(request.blueprint_architecture).model_validate(architecture)})
    broken = deepcopy(valid)
    for section in broken['components'][0]['semantic_content']['sections']:
        section['learning_block_ids'] = []
    broken['components'][3]['covered_source_fact_ids'] = owned[:26]
    diagram_payload = {k: deepcopy(v) for k, v in diagram.items() if k in main.STAGED_COMPONENT_PAYLOAD_FIELDS['la_diagram']}
    diagram_payload['nodes'][0]['tooltip'] = 'Inspect documented conditions before the authorized action.'
    delta = {'components': {
        'c0': {'semantic_content': deepcopy(valid['components'][0]['semantic_content'])},
        'c3': {**diagram_payload, 'covered_source_fact_ids': owned[:]},
    }}
    return request, valid, broken, scope, manifest, delta


async def endpoint_fixture(delta_mutator=None):
    request, valid, broken, scope, manifest, delta = fixture()
    if delta_mutator:
        delta_mutator(delta)
    provider = AsyncMock(side_effect=[(json.dumps(instance_wire(broken)), main.AiUsage()), (json.dumps(delta), main.AiUsage())])
    with patch('app.main.generate_content', provider):
        result = await checkpoint_result(request, manifest)
        final = await checkpoint_result(request.model_copy(update={
            'checkpoint_action': 'validate_chapter', 'checkpoint_unit_index': None,
            'checkpoint_units': [main.ChapterCheckpointUnit(unit_index=0, unit=result['unit'])]}), manifest)
    return result, final, provider, request, valid, broken, scope


class MultiComponentRepairSlotTests(unittest.TestCase):
    def test_sdk_http_roundtrip_and_raw_extra_keys_cannot_be_silently_dropped(self):
        for mode in ('valid', 'extra_slot', 'extra_field', 'duplicate'):
            request, _, broken, _, manifest, delta = fixture()
            for payload in delta['components'].values():
                payload.update(title=None, selection_rationale=None)
            delta['components']['c0']['html'] = None
            delta['components']['c3'].setdefault('name', None)
            # Prove the SDK parsed branch is eligible, not just raw fallback.
            model = main.build_staged_multi_repair_model(broken, [0, 3], [3])
            model.model_validate(delta)
            if mode == 'extra_slot': delta['components']['PRIVATE'] = {}
            elif mode == 'extra_field': delta['components']['c3']['PRIVATE'] = 'PRIVATE'
            text = json.dumps(delta)
            if mode == 'duplicate':
                text = text.replace('"c0":', '"c0": {}, "c0":', 1)
            def body(text):
                return sdk_fixture.response(200, {'candidates': [{'content': {'parts': [{'text': text}]}, 'finishReason': 'STOP'}],
                    'usageMetadata': {'promptTokenCount': 10, 'candidatesTokenCount': 20, 'totalTokenCount': 30}})
            with patch('httpx.Client.send', side_effect=[body(json.dumps(instance_wire(broken))), body(text)]) as http, self.assertLogs('app.main', 'INFO') as logs:
                result = sdk_fixture.CheckpointProviderSDKTests().send(request, manifest)
            self.assertEqual(http.call_count, 2)
            self.assertEqual(result.status_code, 200 if mode == 'valid' else 422, result.text)
            self.assertNotIn('PRIVATE', '\n'.join(logs.output) + result.text)
            if mode == 'valid': self.assertEqual(result.json()['status'], 'unit_ready')
            else: self.assertEqual(result.json()['detail']['internal_failure_code'], 'CHAPTER_COMPONENT_REPAIR_EXHAUSTED')

    def test_uat_two_targets_pass_real_unit_and_full_chapter_acceptance(self):
        request, valid, broken, scope, manifest, delta = fixture()
        self.assertEqual(main.staged_component_repair_targets(broken, scope), [0, 3])
        self.assertEqual(main.validate_staged_unit_content(broken, scope, strict_payload=True).code, 'HTML_TEACHING_GROUP_MISSING')
        self.assertEqual(main.staged_coverage_repair_diagnostics(broken, [3])[0]['missing_count'], 6)
        before = deepcopy(broken)
        with self.assertLogs('app.main', 'INFO') as logs:
            result, final, provider, *_ = asyncio.run(endpoint_fixture())
        self.assertEqual(final['status'], 'ready')
        self.assertEqual(provider.await_count, 2)
        actual = result['unit']['components']
        self.assertEqual(actual[1:3], valid['components'][1:3])
        self.assertEqual(actual[3]['covered_source_fact_ids'], valid['components'][3]['source_fact_ids'])
        for i in (0, 3):
            for key in ('type', 'component_plan_id', 'source_fact_ids', 'supporting_evidence_fact_ids'):
                self.assertEqual(actual[i][key], valid['components'][i][key])
        self.assertEqual(broken, before)
        self.assertIsNone(main.validate_staged_unit_content(result['unit'], scope, strict_payload=True))
        self.assertEqual(provider.await_args.kwargs['request_timeout_ms'], 180000)
        self.assertEqual(provider.await_args.kwargs['max_output_tokens'], provider.await_args_list[0].kwargs['max_output_tokens'])
        self.assertIn('"expected_slot_count": 2', '\n'.join(logs.output))
        self.assertIn('"missing_slot_count": 0', '\n'.join(logs.output))

    def test_projected_actual_sdk_wire_keeps_required_typed_slots_and_per_target_claim_enum(self):
        _, _, broken, _, _, _ = fixture()
        model = main.build_staged_multi_repair_model(broken, [0, 3], [3])
        projected, _ = staged_provider_response_model(model)
        wire = capture_sdk_body(projected)['generationConfig']['responseSchema']
        slots = wire['properties']['components']
        self.assertEqual(slots['type'], 'OBJECT')
        self.assertEqual(set(slots['required']), {'c0', 'c3'})
        html, diagram = slots['properties']['c0'], slots['properties']['c3']
        self.assertIn('semantic_content', html['properties'])
        self.assertNotIn('nodes', html['properties'])
        self.assertNotIn('covered_source_fact_ids', html['properties'])
        self.assertNotIn('semantic_content', diagram['properties'])
        self.assertIn('covered_source_fact_ids', diagram['required'])
        self.assertEqual(diagram['properties']['covered_source_fact_ids']['items']['enum'], broken['components'][3]['source_fact_ids'])
        for field in ('component_index', 'source_fact_ids', 'supporting_evidence_fact_ids', 'component_plan_id'):
            self.assertNotIn('"' + field + '"', json.dumps(wire))
        self.assertEqual(wire, capture_sdk_body(projected)['generationConfig']['responseSchema'])

    def test_slot_inventory_and_duplicate_keys_fail_with_safe_counts_not_values(self):
        _, _, broken, _, _, delta = fixture()
        cases = [({'components': []}, 'COMPONENT_REPAIR_SLOTS_REQUIRED'),
                 ({'components': {'c0': {}}}, 'COMPONENT_REPAIR_SLOT_INVENTORY_INVALID'),
                 ({'components': {**delta['components'], 'PRIVATE_KEY': {}}}, 'COMPONENT_REPAIR_SLOT_INVENTORY_INVALID'),
                 ({'components': {'c0': [], 'c3': {}}}, 'COMPONENT_REPAIR_NOT_OBJECT'),
                 ({'components': {'c0': {'source_fact_ids': ['PRIVATE']}, 'c3': {}}}, 'COMPONENT_REPAIR_SLOT_FIELD_FORBIDDEN'),
                 ({'components': {'c0': {'component_index': 3}, 'c3': {}}}, 'COMPONENT_REPAIR_SLOT_FIELD_FORBIDDEN'),
                 ({'components': {'c0': {'nodes': []}, 'c3': {}}}, 'COMPONENT_REPAIR_SLOT_FIELD_FORBIDDEN')]
        before = deepcopy(broken)
        for value, code in cases:
            diagnostics = {}
            with self.subTest(code=code), self.assertRaisesRegex(main.LessonAuthorProposalValidationError, code):
                main.decode_staged_multi_repair(json.dumps(value), broken, [0, 3], [3], diagnostics)
            self.assertNotIn('PRIVATE', json.dumps(diagnostics))
            self.assertEqual(diagnostics['expected_slot_count'], 2)
        for text in ('{"components":{"c0":{},"c0":{},"c3":{}}}', '{"components":{},"components":{}}'):
            diagnostics = {}
            with self.assertRaisesRegex(main.LessonAuthorProposalValidationError, 'COMPONENT_REPAIR_DUPLICATE_JSON_KEY'):
                main.decode_staged_multi_repair(text, broken, [0, 3], [3], diagnostics)
            self.assertEqual(diagnostics['duplicate_key_count'], 1)
        self.assertEqual(broken, before)

    def test_bad_slot_or_bad_content_never_third_call_or_partial_commit(self):
        for mode in ('missing', 'foreign', 'claim_only', 'foreign_claim', 'html_still_missing'):
            request, _, broken, _, manifest, delta = fixture()
            if mode == 'missing': del delta['components']['c3']
            elif mode == 'foreign': delta['components']['PRIVATE'] = {}
            elif mode == 'claim_only': delta['components']['c3'] = {'covered_source_fact_ids': broken['components'][3]['source_fact_ids']}
            elif mode == 'foreign_claim': delta['components']['c3']['covered_source_fact_ids'].append('PRIVATE')
            else: delta['components']['c0']['semantic_content'] = broken['components'][0]['semantic_content']
            provider = AsyncMock(side_effect=[(json.dumps(instance_wire(broken)), main.AiUsage()), (json.dumps(delta), main.AiUsage())])
            before = deepcopy(broken)
            with patch('app.main.generate_content', provider), patch('app.main.build_source_locked_html_unit') as fallback:
                with self.assertRaises(WorkflowFailure) as caught:
                    asyncio.run(checkpoint_result(request, manifest))
            self.assertEqual(caught.exception.internal_code, 'CHAPTER_COMPONENT_REPAIR_EXHAUSTED')
            self.assertEqual(provider.await_count, 2)
            self.assertNotIn('PRIVATE', json.dumps(caught.exception.diagnostics))
            self.assertIn('repair_slot_diagnostics', caught.exception.diagnostics)
            self.assertEqual(broken, before)
            fallback.assert_not_called()

    def test_all_component_types_serialize_and_duplicate_types_keep_separate_slots(self):
        kinds = ['html', 'problem', 'la_faq', 'la_sortable', 'la_crossword', 'la_diagram', 'html']
        baseline = {'components': [{'type': kind, 'source_fact_ids': [f'fact-{i}']} for i, kind in enumerate(kinds)]}
        model = main.build_staged_multi_repair_model(baseline, list(range(len(kinds))), [0, 6])
        projected, _ = staged_provider_response_model(model)
        slots = capture_sdk_body(projected)['generationConfig']['responseSchema']['properties']['components']
        self.assertEqual(slots['required'], [f'c{i}' for i in range(len(kinds))])
        for i, kind in enumerate(kinds):
            props = slots['properties'][f'c{i}']['properties']
            self.assertLessEqual(set(props), main.STAGED_COMPONENT_PAYLOAD_FIELDS[kind] | {'title', 'selection_rationale', 'covered_source_fact_ids'})
        self.assertEqual(slots['properties']['c0']['properties']['covered_source_fact_ids']['items']['enum'], ['fact-0'])
        self.assertEqual(slots['properties']['c6']['properties']['covered_source_fact_ids']['items']['enum'], ['fact-6'])


if __name__ == '__main__':
    import sys
    if '--node-fixture' in sys.argv:
        unit_result, result, provider, request, *_ = asyncio.run(endpoint_fixture())
        print(json.dumps({'unit_result': unit_result, 'result': result, 'request': request.model_dump(), 'provider_calls': provider.await_count}))
    else:
        unittest.main()
