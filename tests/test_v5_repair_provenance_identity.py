from __future__ import annotations

import copy
import json
import unittest

from app.assessment_planner import compile_v5_assessment_plan
from app.main import (
    _assert_v5_primary_provenance_preserved,
    _v5_prepare_semantic_repair_targets,
    allocate_source_map_architecture_facts,
    apply_course_architecture_repair_patches,
    validate_course_architecture_evidence_scope,
    validate_v5_instructional_coherence,
)
from app.source_map import build_source_map
from app.workflows.contracts import WorkflowFailure
from app.workflows.course_architecture import classify_course_repair_targets
from tests.test_evidence_scope_allocation_v5 import _manifest, _nodes, _v5_blueprint


class V5RepairProvenanceIdentityTests(unittest.TestCase):
    def fixture(self, count=371):
        manifest = _manifest(count, 6)
        source_map = build_source_map(_nodes(6), manifest, locale='en')
        blueprint = _v5_blueprint(source_map)
        lesson = blueprint['chapters'][0]['lessons'][0]
        unit = lesson['units'][0]
        first = unit['learning_blocks'][0]
        scopes = first['primary_evidence_scope_ids']
        self.assertGreater(len(scopes), 1)
        second = copy.deepcopy(first)
        second['id'] = 'second_teaching'
        second['primary_evidence_scope_ids'] = scopes[1:]
        first['primary_evidence_scope_ids'] = scopes[:1]
        unit['learning_blocks'].append(second)
        lesson['assessment_required'] = True
        lesson['assessment_objective_refs'] = ['lo_1']
        return blueprint, source_map, manifest

    @staticmethod
    def blocks(blueprint):
        return blueprint['chapters'][0]['lessons'][0]['units'][0]['learning_blocks']

    def test_actual_semantic_selection_inserts_before_existing_block_and_retains_complete_allocation(self):
        for count in (371, 1001):
            with self.subTest(fact_count=count):
                blueprint, source_map, manifest = self.fixture(count)
                baseline = copy.deepcopy(blueprint)
                compilation = compile_v5_assessment_plan(blueprint)
                self.assertEqual(compilation.status, 'NEEDS_SEMANTIC_RESOLUTION')
                issues = [dict(i.workflow_issue(), repair_layer='PRE_ALLOCATION_COHERENCE') for i in compilation.issues]
                targets = _v5_prepare_semantic_repair_targets(blueprint, classify_course_repair_targets(
                    issues, repair_layer='PRE_ALLOCATION_COHERENCE',
                ), source_map)
                target = targets[0]
                first = self.blocks(blueprint)[0]
                payload = {'patches': [{
                    'path': target['path'], 'operation': 'select_assessment_teaching_alignment',
                    'selections': [{
                        'objective_ref': 'lo_1', 'decision': 'SELECT',
                        'unit_path': target['path'] + '.unit_1', 'teaching_block_id': first['id'],
                    }],
                }]}
                repaired = apply_course_architecture_repair_patches(blueprint, targets, payload)
                blocks = self.blocks(repaired)
                self.assertEqual(blocks[1]['intent'], 'knowledge_check')
                self.assertEqual(blocks[2], self.blocks(baseline)[1])
                self.assertEqual(blocks[0], self.blocks(baseline)[0])
                self.assertEqual(blocks[1]['primary_evidence_scope_ids'], [])
                self.assertEqual(blocks[1]['supporting_evidence_scope_ids'], first['primary_evidence_scope_ids'])
                self.assertFalse(validate_course_architecture_evidence_scope(repaired, source_map).errors)
                self.assertFalse(validate_v5_instructional_coherence(repaired).errors)
                self.assertEqual(compile_v5_assessment_plan(repaired).status, 'READY')
                self.assertEqual(compile_v5_assessment_plan(repaired).blueprint, repaired)
                allocated = allocate_source_map_architecture_facts(copy.deepcopy(repaired), source_map, manifest)
                self.assertEqual(allocated['source_fact_allocation']['allocated_count'], count)
                self.assertEqual(allocated['source_fact_allocation']['unallocated'], [])
                self.assertTrue(allocated['source_fact_allocation']['complete'])
                self.assertEqual(blueprint, baseline)
                self.assertEqual(apply_course_architecture_repair_patches(blueprint, targets, payload), repaired)

    def test_provenance_guard_allows_factless_insertions_without_comparing_array_positions(self):
        baseline, _, _ = self.fixture()
        for position in (0, 1, 2):
            with self.subTest(position=position):
                candidate = copy.deepcopy(baseline)
                self.blocks(candidate).insert(position, {
                    'id': 'server_check', 'intent': 'knowledge_check',
                    'primary_evidence_scope_ids': [],
                    'source_refs': list(self.blocks(baseline)[0]['source_refs']),
                })
                _assert_v5_primary_provenance_preserved(baseline, candidate)
        # This helper proves provenance only; operation/target/domain guards
        # separately authorize where insertion may happen.

    def assert_rejected(self, baseline, candidate, reason, field=None):
        with self.assertRaises(WorkflowFailure) as raised:
            _assert_v5_primary_provenance_preserved(baseline, candidate, patch_count=2)
        failure = raised.exception
        self.assertEqual(failure.internal_code, 'ARCH_REPAIR_SCOPE_VIOLATION')
        self.assertEqual(failure.failure_stage, 'architecture_repair_semantic_delta_guard')
        self.assertEqual(failure.diagnostics['guard_reason'], reason)
        self.assertEqual(failure.diagnostics['patch_count'], 2)
        if field:
            self.assertEqual(failure.diagnostics['outer_field'], field)
        return failure

    def test_existing_provenance_changes_are_still_rejected(self):
        baseline, _, _ = self.fixture()
        for field in ('primary_evidence_scope_ids', 'source_refs'):
            for value in ([], ['unknown'], None):
                with self.subTest(field=field, value=value):
                    candidate = copy.deepcopy(baseline)
                    self.blocks(candidate)[0][field] = value
                    self.assert_rejected(baseline, candidate, 'PRIMARY_OWNERSHIP_MUTATION', field)
        candidate = copy.deepcopy(baseline)
        a, b = self.blocks(candidate)
        a['primary_evidence_scope_ids'], b['primary_evidence_scope_ids'] = b['primary_evidence_scope_ids'], a['primary_evidence_scope_ids']
        self.assert_rejected(baseline, candidate, 'PRIMARY_OWNERSHIP_MUTATION')

    def test_missing_renamed_moved_and_reordered_blocks_fail_closed(self):
        baseline, _, _ = self.fixture()
        for action in ('delete', 'rename', 'move', 'reorder'):
            with self.subTest(action=action):
                candidate = copy.deepcopy(baseline)
                blocks = self.blocks(candidate)
                if action == 'delete':
                    blocks.pop()
                elif action == 'rename':
                    blocks[0]['id'] = 'renamed'
                elif action == 'move':
                    candidate['chapters'][1]['lessons'][0]['units'][0]['learning_blocks'].append(blocks.pop())
                else:
                    blocks.reverse()
                self.assert_rejected(baseline, candidate,
                    'EXISTING_BLOCK_ORDER_CHANGED' if action == 'reorder' else 'EXISTING_BLOCK_MISSING_OR_MOVED')

    def test_duplicate_missing_ids_and_whitespace_aliases_never_collapse(self):
        baseline, _, _ = self.fixture()
        for identifier in (None, '', '  ', self.blocks(baseline)[0]['id'], ' ' + self.blocks(baseline)[0]['id'] + ' '):
            with self.subTest(identifier=identifier):
                candidate = copy.deepcopy(baseline)
                self.blocks(candidate)[1]['id'] = identifier
                reason = 'MISSING_BLOCK_ID' if identifier is None or not identifier.strip() else 'DUPLICATE_BLOCK_ID'
                self.assert_rejected(baseline, candidate, reason)
                self.assert_rejected(candidate, baseline, reason)

    def test_new_primary_ownership_is_rejected_and_equal_ids_in_different_units_do_not_collide(self):
        baseline, _, _ = self.fixture()
        baseline['chapters'][1]['lessons'][0]['units'][0]['learning_blocks'][0]['id'] = self.blocks(baseline)[0]['id']
        _assert_v5_primary_provenance_preserved(baseline, copy.deepcopy(baseline))
        candidate = copy.deepcopy(baseline)
        new = copy.deepcopy(self.blocks(candidate)[0])
        new['id'] = 'new_primary'
        self.blocks(candidate).append(new)
        self.assert_rejected(baseline, candidate, 'NEW_BLOCK_PRIMARY_OWNERSHIP')

    def test_non_block_provenance_stays_protected_and_diagnostics_are_content_free(self):
        baseline, _, _ = self.fixture()
        marker = 'PRIVATE title / source / id'
        self.blocks(baseline)[0]['id'] = marker
        candidate = copy.deepcopy(baseline)
        candidate['chapters'][0]['lessons'][0]['source_refs'] = [marker]
        failure = self.assert_rejected(baseline, candidate, 'PRIMARY_OWNERSHIP_MUTATION', 'source_refs')
        self.assertNotIn(marker, json.dumps(failure.diagnostics))
        self.assertNotIn(marker, str(failure))
        self.assertEqual(failure.diagnostics['repair_target_path'], 'chapter_1.lesson_1')


if __name__ == '__main__':
    unittest.main()
