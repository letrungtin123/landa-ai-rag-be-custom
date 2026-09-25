from __future__ import annotations

import asyncio
import copy
import json
import sys
import unittest
from unittest.mock import AsyncMock, patch

from app.component_capabilities import ComponentCapabilities, validate_instance_plan
from app.assessment_planner import compile_v5_assessment_plan
from app.main import (
    AiUsage, RagLessonAuthorRequest, _locked_component_plan,
    allocate_source_map_architecture_facts, generate_staged_lesson_author_proposal, build_source_locked_unit,
    validate_course_architecture_evidence_scope, validate_v5_instructional_coherence,
    validate_staged_unit_content, validate_course_architecture_workflow,
)
from app.lesson_quality import validate_lesson_pedagogical_quality
from app.source_map import build_source_map
from tests.test_evidence_scope_allocation_v5 import _manifest, _nodes, _v5_blueprint


PROFILE = {"version": 2, "max_components_per_unit": 4, "max_assessments_per_unit": 3, "assessment_enabled": True}


def compiled_fixture(count: int = 90) -> dict:
    manifest = _manifest(count, 1)
    source_map = build_source_map(_nodes(1), manifest, locale="en")
    blueprint = _v5_blueprint(source_map)
    blueprint["component_capabilities"] = PROFILE.copy()
    lesson = blueprint["chapters"][0]["lessons"][0]
    unit = lesson["units"][0]
    unit["title"] = "Safety requirements and procedure"
    first = unit["learning_blocks"][0]
    second = copy.deepcopy(first)
    second.update(id="second_teaching", learning_objective_refs=["lo_2"])
    scopes = first["primary_evidence_scope_ids"]
    first["primary_evidence_scope_ids"] = scopes[:1]
    second["primary_evidence_scope_ids"] = scopes[1:]
    unit["learning_blocks"].append(second)
    unit["learning_objective_refs"] = ["lo_1", "lo_2"]
    lesson.update(learning_objectives=["Identify the documented safety requirement.", "Explain the documented safety procedure."],
                  assessment_required=True, assessment_objective_refs=["lo_1", "lo_2"])
    compiled = compile_v5_assessment_plan(blueprint)
    assert compiled.status == "READY", compiled.issues
    assert not validate_course_architecture_evidence_scope(compiled.blueprint, source_map).errors
    assert not validate_v5_instructional_coherence(compiled.blueprint).errors
    allocated = allocate_source_map_architecture_facts(compiled.blueprint, source_map, manifest)
    assert allocated["source_fact_allocation"]["allocated_count"] == count
    assert not validate_course_architecture_workflow(allocated, source_map, manifest, {"src-001"}).errors
    return {"blueprint": allocated, "source_map": source_map, "manifest": manifest}


def draft_fixture(payload: dict) -> dict:
    architecture = payload["architecture"]
    manifest = payload["manifest"]
    unit = architecture["lessons"][0]["units"][0]
    components = []
    for index, plan in enumerate(unit["component_plan"]):
        content = {
            "type": plan["type"], "component_plan_id": plan["component_plan_id"],
            "source_fact_ids": plan.get("source_fact_ids", []),
            "covered_source_fact_ids": plan.get("source_fact_ids", []),
            "supporting_evidence_fact_ids": plan.get("supporting_evidence_fact_ids", []),
        }
        if plan["type"] == "html":
            content["html"] = "<p>" + " ".join(fact["text"] for fact in manifest["facts"]) + "</p>"
        else:
            question = "What kind of definition does the source evidence provide?" if index == 1 else "Which procedure is documented before beginning the task?"
            content.update(problem_type="short_text", question=question,
                           answer="source-backed definition" if index == 1 else "source-backed procedure", explanation="The prior teaching explains this documented requirement.")
        components.append(content)
    generated = {"title": unit["title"], "source_fact_ids": unit["source_fact_ids"],
                 "supporting_evidence_fact_ids": unit.get("supporting_evidence_fact_ids", []), "components": components}
    request = RagLessonAuthorRequest(
        tenant_id="11111111-1111-4111-8111-111111111111", kb_id="22222222-2222-4222-8222-222222222222",
        conversation_id="33333333-3333-4333-8333-333333333333", target="lesson_author", model="mock-only",
        embedding_model="mock-only", system_prompt="synthetic", user_message="Draft approved unit", output_schema_hint="synthetic",
        api_key="test-key", max_output_tokens=8192, blueprint_architecture=architecture,
    )
    provider = AsyncMock(return_value=(json.dumps(generated), AiUsage(inputTokens=1, outputTokens=1, totalTokens=2)))
    with patch("app.main.generate_content", new=provider):
        proposal, _usage = asyncio.run(generate_staged_lesson_author_proposal(
            request, "Synthetic context", "", "", source_rows=[], source_coverage_manifest=manifest,
        ))
    assert provider.await_count == 1
    quality = validate_lesson_pedagogical_quality(proposal, architecture)
    assert quality.status != "FAIL", quality.findings
    return {"proposal": proposal, "provider_calls": provider.await_count}


class ComponentInstanceContractTests(unittest.TestCase):
    def test_profile_existing_plus_new_checks_and_capacity_fail_closed(self):
        original = compiled_fixture()['blueprint']
        unit = original['chapters'][0]['lessons'][0]['units'][0]
        # Keep one existing check, compile the missing second obligation.
        unit['learning_blocks'] = [b for b in unit['learning_blocks']
                                   if not (b['intent'] == 'knowledge_check' and b['learning_objective_refs'] == ['lo_2'])]
        before = copy.deepcopy(original)
        result = compile_v5_assessment_plan(original)
        self.assertEqual(original, before)
        self.assertEqual(result.status, 'READY')
        self.assertEqual(result.existing_checks, 1)
        self.assertEqual(result.deterministic_insertions, 1)
        diagnostic = result.safe_unit_diagnostics(original)[0]
        self.assertEqual(diagnostic['existing_check_count'], 1)
        self.assertEqual(diagnostic['proposed_assessment_count'], 1)
        self.assertNotIn('text', diagnostic)
        existing = next(b for b in unit['learning_blocks'] if b['intent'] == 'knowledge_check')
        unit['learning_blocks'] += [dict(copy.deepcopy(existing), id=f'existing_{n}') for n in range(2)]
        rejected = compile_v5_assessment_plan(original)
        self.assertEqual(rejected.status, 'TERMINAL_GAP')
        issue = next(i for i in rejected.issues if i.code == 'ASSESSMENT_PLAN_DOWNSTREAM_CAPABILITY_GAP')
        self.assertEqual(issue.capacity_details['existing_check_count'], 3)
        self.assertEqual(issue.capacity_details['proposed_assessment_count'], 1)
        self.assertEqual(rejected.metrics['assessment_plan_committed_insertions'], 0)
        self.assertFalse(issue.workflow_issue()['repairable'])

    def test_profile_disabled_assessment_and_same_anchor_objectives(self):
        original = compiled_fixture()['blueprint']
        original['component_capabilities']['assessment_enabled'] = False
        rejected = compile_v5_assessment_plan(original)
        self.assertEqual(rejected.status, 'TERMINAL_GAP')
        self.assertEqual(rejected.issues[0].safe_reason, 'TENANT_ASSESSMENT_DISABLED')
        original['component_capabilities']['assessment_enabled'] = True
        unit = original['chapters'][0]['lessons'][0]['units'][0]
        teaching = [b for b in unit['learning_blocks'] if b['intent'] != 'knowledge_check']
        teaching[0]['primary_evidence_scope_ids'] += teaching[1]['primary_evidence_scope_ids']
        teaching[0]['learning_objective_refs'] = ['lo_1', 'lo_2']
        unit['learning_blocks'] = [teaching[0]]
        result = compile_v5_assessment_plan(original)
        checks = [b for b in result.blueprint['chapters'][0]['lessons'][0]['units'][0]['learning_blocks'] if b['intent'] == 'knowledge_check']
        self.assertEqual(result.status, 'READY')
        self.assertEqual(len(checks), 1)
        self.assertEqual(checks[0]['learning_objective_refs'], ['lo_1', 'lo_2'])

    def test_profile_scale_preserves_all_canonical_ownership(self):
        for count in (371, 1001):
            with self.subTest(count=count):
                manifest = _manifest(count, 6)
                if count == 371:
                    # Synthetic provenance/weight fixture, not private UAT text:
                    # dense first section plus five smaller source sections.
                    cursor = 0
                    for section, size in enumerate((233, 28, 28, 28, 27, 27), 1):
                        for fact in manifest['facts'][cursor:cursor + size]:
                            fact.update(source_ref=f'src-{section:03d}', source_page=section)
                            if section == 1:
                                fact['text'] = ('Documented safety condition and approved procedural evidence. ' * 6)[:320]
                        cursor += size
                source_map = build_source_map(_nodes(6), manifest, locale='en')
                if count == 371:
                    self.assertEqual(len(source_map['source_evidence_scopes']), 17)
                candidate = _v5_blueprint(source_map)
                candidate['component_capabilities'] = PROFILE.copy()
                for chapter in candidate['chapters']:
                    lesson = chapter['lessons'][0]
                    lesson.update(assessment_required=True, assessment_objective_refs=['lo_1'])
                compiled = compile_v5_assessment_plan(candidate)
                self.assertEqual(compiled.status, 'READY')
                self.assertFalse(validate_course_architecture_evidence_scope(compiled.blueprint, source_map).errors)
                self.assertFalse(validate_v5_instructional_coherence(compiled.blueprint).errors)
                first = allocate_source_map_architecture_facts(copy.deepcopy(compiled.blueprint), source_map, manifest)
                second = allocate_source_map_architecture_facts(copy.deepcopy(compiled.blueprint), source_map, manifest)
                self.assertEqual(first, second)
                allocation = first['source_fact_allocation']
                self.assertEqual(allocation['allocated_count'], count)
                self.assertEqual(allocation['unallocated'], [])
                self.assertEqual(len({item['fact_id'] for item in allocation['allocations']}), count)
                self.assertFalse(validate_course_architecture_workflow(first, source_map, manifest, {f'src-{i:03d}' for i in range(1, 7)}).errors)

    def plans(self):
        return [{"component_plan_id": "cp2_" + str(n) * 32, "type": t, "title": "Fixture", "rationale": "Fixture",
                 "source_fact_ids": ["fact_a"] if t == "html" else [],
                 "supporting_evidence_fact_ids": [] if t == "html" else ["fact_a"]}
                for n, t in enumerate(["html", "problem", "problem"])]

    def test_locked_plan_preserves_instances_and_legacy_dedup(self):
        plans = self.plans()
        locked = _locked_component_plan(plans, ["fact_a"], ["fact_a"], strict_ownership=True, instance_contract=True)
        self.assertEqual([p['component_plan_id'] for p in locked], [p['component_plan_id'] for p in plans])
        legacy = [{k: v for k, v in p.items() if k != 'component_plan_id'} for p in plans]
        self.assertEqual(len(_locked_component_plan(legacy, ['fact_a'], ['fact_a'])), 2)

    def test_instance_missing_duplicate_malformed_and_cross_source_fail(self):
        for change in ('missing', 'duplicate', 'malformed', 'outside'):
            plans = self.plans()
            if change == 'missing': plans[1].pop('component_plan_id')
            if change == 'duplicate': plans[1]['component_plan_id'] = plans[2]['component_plan_id']
            if change == 'malformed': plans[1]['component_plan_id'] = 'cp2_arbitrary'
            if change == 'outside': plans[1]['supporting_evidence_fact_ids'] = ['alien_fact']
            with self.subTest(change=change), self.assertRaises(ValueError):
                _locked_component_plan(plans, ['fact_a'], ['fact_a'], strict_ownership=True, instance_contract=True)

    def test_generated_instance_mismatch_fails(self):
        plans = self.plans()
        expected = {'component_plan': plans, 'component_types': ['html', 'problem', 'problem'],
                    'source_fact_ids': ['fact_a'], 'supporting_evidence_fact_ids': ['fact_a']}
        components = [dict(p, covered_source_fact_ids=p['source_fact_ids']) for p in plans]
        components[0]['html'] = '<p>' + 'Documented safety procedure. ' * 40 + '</p>'
        for p in components[1:]: p.update(question='What is the documented requirement?', problem_type='short_text', answer='Safety')
        unit = {'components': components, 'source_fact_ids': ['fact_a'], 'supporting_evidence_fact_ids': ['fact_a']}
        self.assertIsNone(validate_staged_unit_content(unit, expected))
        components[1]['component_plan_id'] = components[2]['component_plan_id']
        self.assertEqual(validate_staged_unit_content(unit, expected), 'COMPONENT_PLAN_INSTANCE_MISMATCH')

    def test_profile_is_strict_and_not_a_provider_budget_override(self):
        for value in ({**PROFILE, 'max_components_per_unit': 100}, {**PROFILE, 'assessment_enabled': 'true'}, {**PROFILE, 'token_limit': 999}):
            with self.assertRaises(ValueError): ComponentCapabilities.model_validate(value)

    def test_instance_plan_cannot_use_legacy_type_keyed_fallback(self):
        self.assertIsNone(build_source_locked_unit({'component_plan': self.plans()}, {}))


if __name__ == '__main__':
    # Narrow deterministic subprocess bridge for the cross-language test; never HTTP/DB.
    payload = json.load(sys.stdin)
    print(json.dumps(compiled_fixture() if payload['mode'] == 'compile' else draft_fixture(payload)))
