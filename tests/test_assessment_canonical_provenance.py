"""Offline regression for Sep29 assessment gaps; no private output/DB/provider IO."""
import copy
import json
import unittest
from unittest.mock import AsyncMock, patch

from app.assessment_planner import compile_v5_assessment_plan, materialize_assessment_source_refs
from app.assessment_selection_contract import build_assessment_selection_contract
from app.component_capabilities import ComponentCapabilities
from app.schemas.common import AiUsage
from app.services.lesson_author.architecture_validation import validate_v5_instructional_coherence
from app.services.lesson_author.blueprint_workflow import lesson_author_blueprint
from app.services.lesson_author.evidence_scope import (
    allocate_source_map_architecture_facts,
    validate_course_architecture_evidence_scope,
)
from app.services.lesson_author.repair.semantic_delta import apply_course_architecture_repair_patches
from app.source_map import build_source_map
from app.workflows.contracts import WorkflowFailure
from tests.test_assessment_intent_repair import targets_for
from tests.test_assessment_selection_contract import selected_choices
from tests.test_evidence_scope_allocation_v5 import _manifest, _nodes, _v5_blueprint
from tests.test_lesson_author_blueprint import blueprint_request


def fixture(count=415, locale="en"):
    nodes, manifest = _nodes(7), _manifest(count, 7)
    source_map = build_source_map(nodes, manifest, locale=locale)
    blueprint = _v5_blueprint(source_map)
    for ci in (2, 5):
        lesson = blueprint["chapters"][ci]["lessons"][0]
        lesson.update(assessment_required=True, assessment_objective_refs=["lo_1", "lo_2"])
        lesson["learning_objectives"].append("Describe the remaining source-backed concepts.")
        block = lesson["units"][0]["learning_blocks"][0]
        second = copy.deepcopy(lesson["units"][0])
        second.update(title="Other objective", learning_objective_refs=["lo_2"])
        second["learning_blocks"][0].update(
            id=f"other_{ci}", learning_objective_refs=["lo_2"],
            primary_evidence_scope_ids=block["primary_evidence_scope_ids"][1:],
        )
        assert second["learning_blocks"][0]["primary_evidence_scope_ids"]
        lesson["units"].append(second)
        block["primary_evidence_scope_ids"] = block["primary_evidence_scope_ids"][:1]
        block.update(intent="relationship_visualization", source_refs=[],
                     content={"purpose": "Explain the documented relationships.",
                              "relationship_evidence": True, "relationship_count": 3})
    return blueprint, source_map, manifest, nodes


def owner(blueprint, ci=2):
    return blueprint["chapters"][ci]["lessons"][0]["units"][0]["learning_blocks"][0]


class CanonicalAssessmentProvenanceTests(unittest.TestCase):
    def test_reproduces_terminal_gap_and_materializes_only_canonical_refs(self):
        blueprint, source_map, _, _ = fixture()
        baseline = copy.deepcopy(blueprint)
        self.assertFalse(validate_course_architecture_evidence_scope(blueprint, source_map).errors)
        before = compile_v5_assessment_plan(blueprint)
        self.assertEqual(before.status, "TERMINAL_GAP")
        self.assertEqual([i.lesson_path for i in before.issues], ["chapter_3.lesson_1", "chapter_6.lesson_1"])
        self.assertTrue(all(d["intent_repair_candidate_count"] == 0 for d in before.objective_diagnostics))
        self.assertEqual(before.objective_diagnostics[0]["candidates"][0]["missing_direct_provenance"], ["MISSING_SOURCE_REFS"])
        self.assertEqual(before.objective_diagnostics[0]["candidates"][1]["reasons"], ["OBJECTIVE_OUTSIDE_UNIT_SCOPE"])
        projected, diagnostics, issues = materialize_assessment_source_refs(blueprint, source_map)
        self.assertFalse(issues)
        self.assertEqual(len(diagnostics), 2)
        self.assertEqual(blueprint, baseline)
        self.assertEqual(owner(projected)["source_refs"], ["src-003"])
        self.assertEqual(owner(projected, 5)["source_refs"], ["src-006"])
        self.assertEqual(owner(projected)["intent"], "relationship_visualization")
        self.assertEqual(compile_v5_assessment_plan(projected).status, "NEEDS_SEMANTIC_RESOLUTION")
        self.assertEqual(materialize_assessment_source_refs(projected, source_map), (projected, [], []))

    def test_typed_selection_preserves_diagram_provenance_and_complete_allocation(self):
        for count in (371, 415, 1001):
            with self.subTest(count=count):
                blueprint, source_map, manifest, _ = fixture(count)
                projected, _, issues = materialize_assessment_source_refs(blueprint, source_map)
                self.assertFalse(issues)
                baseline = copy.deepcopy(projected)
                targets = targets_for(projected, source_map)
                contract = build_assessment_selection_contract(targets, max_context_chars=16000)
                payload = contract.decode_text(json.dumps(selected_choices(contract)))
                repaired = apply_course_architecture_repair_patches(projected, targets, payload)
                self.assertEqual(projected, baseline)
                self.assertEqual(repaired, apply_course_architecture_repair_patches(projected, targets, payload))
                for ci in (2, 5):
                    blocks = repaired["chapters"][ci]["lessons"][0]["units"][0]["learning_blocks"]
                    self.assertEqual([b["intent"] for b in blocks], ["concept_explanation", "knowledge_check", "relationship_visualization"])
                    self.assertEqual(blocks[0], dict(owner(baseline, ci), intent="concept_explanation"))
                    visual = blocks[2]
                    self.assertEqual(visual["content"], owner(baseline, ci)["content"])
                    self.assertEqual(visual["primary_evidence_scope_ids"], [])
                    self.assertEqual(set(visual["supporting_evidence_scope_ids"]), set(blocks[0]["primary_evidence_scope_ids"]))
                    self.assertFalse(visual.get("source_fact_ids"))
                for ci in (0, 1, 3, 4, 6):
                    self.assertEqual(repaired["chapters"][ci], baseline["chapters"][ci])
                self.assertFalse(validate_course_architecture_evidence_scope(repaired, source_map).errors)
                self.assertFalse(validate_v5_instructional_coherence(repaired).errors)
                self.assertEqual(compile_v5_assessment_plan(repaired).status, "READY")
                allocated = allocate_source_map_architecture_facts(repaired, source_map, manifest)["source_fact_allocation"]
                self.assertTrue(allocated["complete"])
                self.assertEqual(allocated["allocated_count"], count)
                self.assertEqual(allocated["unallocated"], [])
                self.assertEqual(len({a["fact_id"] for a in allocated["allocations"]}), count)

    def test_invalid_canonical_inputs_are_transactional_and_fail_closed(self):
        for mode in ("unknown", "duplicate", "no_refs", "cross_document", "concept_mismatch"):
            with self.subTest(mode=mode):
                blueprint, source_map, _, _ = fixture()
                block = owner(blueprint)
                scope = next(s for s in source_map["source_evidence_scopes"] if s["id"] == block["primary_evidence_scope_ids"][0])
                if mode == "unknown":
                    block["primary_evidence_scope_ids"].append("not-canonical")
                elif mode == "duplicate":
                    source_map["source_evidence_scopes"].append(copy.deepcopy(scope))
                elif mode == "no_refs":
                    scope["source_ref"] = ""
                elif mode == "cross_document":
                    extra = copy.deepcopy(scope)
                    extra.update(id="other-doc-scope", document_id="other-doc")
                    source_map["source_evidence_scopes"].append(extra)
                    block["primary_evidence_scope_ids"].append(extra["id"])
                else:
                    block["concept_ids"] = []
                baseline = copy.deepcopy(blueprint)
                projected, _, issues = materialize_assessment_source_refs(blueprint, source_map)
                self.assertTrue(issues)
                self.assertEqual(projected, baseline)
                self.assertEqual(blueprint, baseline)
                self.assertTrue(all(not i.workflow_issue()["repairable"] for i in issues))

    def test_explicit_mismatch_and_cross_unit_objective_still_fail(self):
        blueprint, source_map, _, _ = fixture()
        owner(blueprint)["source_refs"] = ["src-001"]
        projected, _, _ = materialize_assessment_source_refs(blueprint, source_map)
        self.assertEqual(owner(projected)["source_refs"], ["src-001"])
        self.assertTrue(validate_course_architecture_evidence_scope(projected, source_map).errors)
        blueprint, source_map, _, _ = fixture()
        lesson = blueprint["chapters"][2]["lessons"][0]
        lesson["units"][0]["learning_objective_refs"] = []
        owner(blueprint)["learning_objective_refs"] = []
        projected, _, _ = materialize_assessment_source_refs(blueprint, source_map)
        self.assertEqual(compile_v5_assessment_plan(projected).status, "TERMINAL_GAP")

    def test_no_match_and_stale_authority_do_not_create_support_or_check(self):
        blueprint, source_map, _, _ = fixture()
        projected, _, _ = materialize_assessment_source_refs(blueprint, source_map)
        targets = targets_for(projected, source_map)
        contract = build_assessment_selection_contract(targets, max_context_chars=16000)
        baseline = copy.deepcopy(projected)
        no_match = contract.decode_text(json.dumps({"choices": dict.fromkeys(contract.bindings, "NO_MATCH")}))
        with self.assertRaises(WorkflowFailure):
            apply_course_architecture_repair_patches(projected, targets, no_match)
        self.assertEqual(projected, baseline)
        owner(projected)["content"] = {"purpose": "Different semantics"}
        with self.assertRaises(WorkflowFailure):
            apply_course_architecture_repair_patches(projected, targets, contract.decode_text(json.dumps(selected_choices(contract))))

    def test_visual_id_collision_is_typed_and_transactional(self):
        blueprint, source_map, _, _ = fixture()
        projected, _, _ = materialize_assessment_source_refs(blueprint, source_map)
        targets = targets_for(projected, source_map)
        contract = build_assessment_selection_contract(targets, max_context_chars=16000)
        payload = contract.decode_text(json.dumps(selected_choices(contract)))
        repaired = apply_course_architecture_repair_patches(projected, targets, payload)
        occupied_id = repaired["chapters"][2]["lessons"][0]["units"][0]["learning_blocks"][2]["id"]
        existing = copy.deepcopy(owner(projected))
        existing.update(id=occupied_id, intent="summary", primary_evidence_scope_ids=[],
                        supporting_evidence_scope_ids=owner(projected)["primary_evidence_scope_ids"].copy())
        projected["chapters"][2]["lessons"][0]["units"][0]["learning_blocks"].append(existing)
        targets = targets_for(projected, source_map)
        contract = build_assessment_selection_contract(targets, max_context_chars=16000)
        baseline = copy.deepcopy(projected)
        with self.assertRaises(WorkflowFailure) as caught:
            apply_course_architecture_repair_patches(projected, targets, contract.decode_text(json.dumps(selected_choices(contract))))
        self.assertEqual(caught.exception.internal_code, "ARCH_REPAIR_INVALID_BLOCK_ID")
        self.assertEqual(caught.exception.diagnostics["guard_reason"], "SERVER_VISUAL_BLOCK_ID_COLLISION")
        self.assertEqual(projected, baseline)

    def test_legacy_and_already_explicit_references_are_unchanged(self):
        for version in (3, 4):
            blueprint, source_map, _, _ = fixture()
            blueprint["architecture_contract_version"] = version
            self.assertEqual(materialize_assessment_source_refs(blueprint, source_map), (blueprint, [], []))
        _, source_map, _, _ = fixture()
        blueprint = _v5_blueprint(source_map)
        self.assertEqual(materialize_assessment_source_refs(blueprint, source_map), (blueprint, [], []))


class CanonicalAssessmentEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_endpoint_projects_refs_repairs_two_lessons_and_finalizes(self):
        for count, locale in ((415, "vi"), (1001, "en")):
            with self.subTest(count=count, locale=locale):
                candidate, source_map, manifest, nodes = fixture(count, locale)
                projected, _, _ = materialize_assessment_source_refs(candidate, source_map)
                contract = build_assessment_selection_contract(targets_for(projected, source_map), max_context_chars=16000)
                choices = selected_choices(contract)
                request = blueprint_request().model_copy(update={
                    "locale": locale, "correlation_id": "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
                    "component_capabilities": ComponentCapabilities(version=2, max_components_per_unit=4,
                        max_assessments_per_unit=3, assessment_enabled=True),
                })
                calls, events = [], []

                async def provider(*args, **kwargs):
                    calls.append(kwargs)
                    if len(calls) == 1:
                        payload = candidate
                    elif len(calls) == 2:
                        self.assertEqual(set(kwargs["response_schema"].properties), {"choices"})
                        payload = choices
                    else:
                        self.fail("Unexpected extra provider call")
                    return json.dumps(payload), AiUsage(inputTokens=10, outputTokens=10, totalTokens=20)

                def capture(message, *args, **kwargs):
                    if message == "lesson_author_blueprint_diagnostic %s":
                        events.append(json.loads(args[0]))

                structure = {"source_structure_nodes": nodes, "source_coverage_manifest": manifest,
                             "known_source_refs": {n["source_ref"] for n in nodes}}
                with patch("app.services.retrieval.search.retrieve_chunks", new=AsyncMock(return_value=([], AiUsage(), structure))), \
                     patch("app.services.provider.generate_content", new=AsyncMock(side_effect=provider)), \
                     patch("app.services.lesson_author.blueprint_workflow.logger.info", side_effect=capture):
                    result = await lesson_author_blueprint(request, pool=object())
                allocation = result["blueprint"]["source_fact_allocation"]
                self.assertEqual(allocation["allocated_count"], count)
                self.assertTrue(allocation["complete"])
                self.assertEqual(allocation["unallocated"], [])
                self.assertEqual(len(calls), 2)
                self.assertEqual(result["workflow"]["repair_provider_calls"], 1)
                self.assertTrue(any(e.get("stage") == "v5_assessment_provenance" and e.get("materialized_block_count") == 2 for e in events))
                self.assertTrue(any(e["event"] == "final_validation_passed" for e in events))
                self.assertTrue(all(e["correlation_id"] == request.correlation_id for e in events))
                self.assertNotIn("Explain the documented relationships", json.dumps(events))
