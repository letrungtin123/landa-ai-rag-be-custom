"""Regression for an objective-link candidate hiding a safe intent repair.

Synthetic metadata only; production endpoint is exercised with provider/DB IO
disabled. A test choice models a semantic decision, not a production fallback.
"""
import copy
import json
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from app.assessment_planner import compile_v5_assessment_plan, materialize_assessment_source_refs
from app.assessment_selection_contract import build_assessment_selection_contract
from app.component_capabilities import ComponentCapabilities
from app.main import lesson_author_blueprint
from app.schemas.common import AiUsage
from app.services.lesson_author.architecture_validation import validate_v5_instructional_coherence
from app.services.lesson_author.evidence_scope import (
    allocate_source_map_architecture_facts,
    validate_course_architecture_evidence_scope,
)
from app.services.lesson_author.repair.semantic_delta import apply_course_architecture_repair_patches
from app.workflows.contracts import WorkflowFailure
from tests.test_assessment_canonical_provenance import fixture
from tests.test_assessment_intent_repair import targets_for
from tests.test_assessment_selection_contract import selected_choices
from tests.test_lesson_author_blueprint import blueprint_request


def mixed_fixture(count=415, locale="en"):
    candidate, source_map, manifest, nodes = fixture(count, locale)
    candidate["component_capabilities"] = dict(version=2, max_components_per_unit=4,
        max_assessments_per_unit=3, assessment_enabled=True)
    for ci in (2, 5):
        lesson = candidate["chapters"][ci]["lessons"][0]
        unit, second = lesson["units"]
        unit["learning_objective_refs"] = ["lo_1", "lo_2"]
        comparison = second["learning_blocks"][0]
        comparison.update(intent="comparison", content={"purpose": "Compare the other documented concepts."})
        unit["learning_blocks"].append(comparison)
        lesson["units"] = [unit]
    return candidate, source_map, manifest, nodes


class AssessmentCandidateInventoryTests(unittest.TestCase):
    def test_objective_link_candidate_cannot_hide_intent_candidate(self):
        candidate, source_map, _, _ = mixed_fixture()
        candidate, _, _ = materialize_assessment_source_refs(candidate, source_map)
        compiled = compile_v5_assessment_plan(candidate)
        self.assertEqual(compiled.status, "NEEDS_SEMANTIC_RESOLUTION")
        for path in ("chapter_3.lesson_1", "chapter_6.lesson_1"):
            options = compiled.candidates[(path, "lo_1")]
            self.assertEqual({c.alignment for c in options}, {"semantic_candidate", "intent_candidate"})
            self.assertEqual(len(options), 2)
            diagnostic = next(d for d in compiled.objective_diagnostics if d["lesson_path"] == path and d["objective_id"] == "lo_1")
            self.assertEqual(diagnostic["offered_candidate_count"], 2)
            self.assertEqual(diagnostic["offered_intent_candidate_count"], 1)

    def test_candidate_matrix_is_complete_deterministic_and_strict(self):
        for semantic_count in (0, 1, 2):
            for intent_count in (0, 1, 2):
                for fully_count in (0, 1, 2):
                    with self.subTest(semantic=semantic_count, intent=intent_count, fully=fully_count):
                        candidate, source_map, _, _ = mixed_fixture()
                        candidate, _, _ = materialize_assessment_source_refs(candidate, source_map)
                        lesson = candidate["chapters"][2]["lessons"][0]
                        lesson["assessment_objective_refs"] = ["lo_1"]
                        unit = lesson["units"][0]
                        template = copy.deepcopy(unit["learning_blocks"][0])
                        blocks = []
                        for kind, count in (("semantic", semantic_count), ("intent", intent_count), ("fully", fully_count)):
                            for index in range(count):
                                block = copy.deepcopy(template)
                                block.update(id=f"{kind}_{index}", intent="relationship_visualization" if kind == "intent" else "concept_explanation",
                                    learning_objective_refs=["lo_2"] if kind == "semantic" else ["lo_1"])
                                blocks.append(block)
                        unit["learning_blocks"] = blocks
                        path = "chapter_3.lesson_1"
                        result = compile_v5_assessment_plan(candidate, lesson_paths={path})
                        self.assertEqual(result.candidates, compile_v5_assessment_plan(candidate, lesson_paths={path}).candidates)
                        if fully_count == 1:
                            self.assertEqual(result.status, "READY")
                            self.assertFalse(result.candidates)
                        elif fully_count > 1:
                            self.assertEqual(len(result.candidates[(path, "lo_1")]), fully_count + semantic_count)
                        elif semantic_count + intent_count:
                            self.assertEqual(len(result.candidates[(path, "lo_1")]), semantic_count + intent_count)
                        else:
                            self.assertEqual(result.status, "TERMINAL_GAP")
        # No same-unit/provenance eligibility is weakened to enlarge the set.
        candidate, source_map, _, _ = mixed_fixture()
        candidate, _, _ = materialize_assessment_source_refs(candidate, source_map)
        block = candidate["chapters"][2]["lessons"][0]["units"][0]["learning_blocks"][0]
        block["primary_evidence_scope_ids"] = []
        options = compile_v5_assessment_plan(candidate).candidates[("chapter_3.lesson_1", "lo_1")]
        self.assertEqual([c.alignment for c in options], ["semantic_candidate"])

    def test_mixed_choice_passes_allocation_without_changing_other_anchor(self):
        for count in (371, 415, 1001):
            with self.subTest(count=count):
                candidate, source_map, manifest, _ = mixed_fixture(count)
                candidate, _, _ = materialize_assessment_source_refs(candidate, source_map)
                baseline = copy.deepcopy(candidate)
                targets = targets_for(candidate, source_map)
                contract = build_assessment_selection_contract(targets, max_context_chars=16000)
                payload = contract.decode_text(json.dumps(selected_choices(contract)))
                repaired = apply_course_architecture_repair_patches(candidate, targets, payload)
                self.assertEqual(candidate, baseline)
                self.assertEqual(repaired, apply_course_architecture_repair_patches(candidate, targets, payload))
                for ci in (2, 5):
                    original_blocks = baseline["chapters"][ci]["lessons"][0]["units"][0]["learning_blocks"]
                    blocks = repaired["chapters"][ci]["lessons"][0]["units"][0]["learning_blocks"]
                    self.assertEqual(next(b for b in blocks if b["id"] == original_blocks[1]["id"]), original_blocks[1])
                    self.assertEqual(sum(b["intent"] == "relationship_visualization" for b in blocks), 1)
                    self.assertEqual(sum(b["intent"] == "knowledge_check" for b in blocks), 2)
                self.assertFalse(validate_course_architecture_evidence_scope(repaired, source_map).errors)
                self.assertFalse(validate_v5_instructional_coherence(repaired).errors)
                self.assertEqual(compile_v5_assessment_plan(repaired).status, "READY")
                allocation = allocate_source_map_architecture_facts(repaired, source_map, manifest)["source_fact_allocation"]
                self.assertTrue(allocation["complete"])
                self.assertEqual(allocation["allocated_count"], count)
                self.assertEqual(allocation["unallocated"], [])
                self.assertEqual(len({a["fact_id"] for a in allocation["allocations"]}), count)

    def test_no_match_remains_terminal_not_arbitrary_selection(self):
        candidate, source_map, _, _ = mixed_fixture()
        candidate, _, _ = materialize_assessment_source_refs(candidate, source_map)
        baseline = copy.deepcopy(candidate)
        targets = targets_for(candidate, source_map)
        contract = build_assessment_selection_contract(targets, max_context_chars=16000)
        payload = contract.decode_text(json.dumps({"choices": dict.fromkeys(contract.bindings, "NO_MATCH")}))
        with self.assertRaises(WorkflowFailure) as caught:
            apply_course_architecture_repair_patches(candidate, targets, payload)
        self.assertEqual(caught.exception.internal_code, "ARCH_REPAIR_ASSESSMENT_NO_MATCH")
        self.assertEqual(candidate, baseline)


class AssessmentCandidateEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def run_endpoint(self, *, count, locale, no_match=False):
        candidate, source_map, manifest, nodes = mixed_fixture(count, locale)
        projected, _, _ = materialize_assessment_source_refs(candidate, source_map)
        contract = build_assessment_selection_contract(targets_for(projected, source_map), max_context_chars=16000)
        choices = {"choices": dict.fromkeys(contract.bindings, "NO_MATCH")} if no_match else selected_choices(contract)
        request = blueprint_request().model_copy(update={
            "locale": locale, "correlation_id": "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
            "component_capabilities": ComponentCapabilities(**candidate["component_capabilities"]),
        })
        calls, events = [], []

        async def provider(*args, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                payload = candidate
            elif len(calls) == 2:
                self.assertEqual(set(kwargs["response_schema"].properties), {"choices"})
                self.assertEqual(kwargs["request_timeout_ms"], 300000)
                payload = choices
            else:
                self.fail("Unexpected additional provider call")
            return json.dumps(payload), AiUsage(inputTokens=10, outputTokens=10, totalTokens=20)

        def capture(message, *args, **kwargs):
            if message == "lesson_author_blueprint_diagnostic %s":
                events.append(json.loads(args[0]))

        structure = {"source_structure_nodes": nodes, "source_coverage_manifest": manifest,
                     "known_source_refs": {n["source_ref"] for n in nodes}}
        with patch("app.services.retrieval.search.retrieve_chunks", new=AsyncMock(return_value=([], AiUsage(), structure))), \
             patch("app.services.provider.generate_content", new=AsyncMock(side_effect=provider)), \
             patch("app.main.logger.info", side_effect=capture):
            try:
                result = await lesson_author_blueprint(request, pool=object())
            except HTTPException as error:
                result = error
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(e["correlation_id"] == request.correlation_id for e in events))
        self.assertNotIn("Compare the other documented concepts", json.dumps(events))
        return result, events

    async def test_mixed_candidate_production_endpoint_finalizes_vi_and_en(self):
        for count, locale in ((415, "vi"), (1001, "en")):
            with self.subTest(count=count, locale=locale):
                result, events = await self.run_endpoint(count=count, locale=locale)
                self.assertNotIsInstance(result, HTTPException, getattr(result, "detail", None))
                allocation = result["blueprint"]["source_fact_allocation"]
                self.assertEqual(allocation["allocated_count"], count)
                self.assertTrue(allocation["complete"])
                self.assertEqual(allocation["unallocated"], [])
                self.assertEqual(result["workflow"]["repair_provider_calls"], 1)
                self.assertTrue(any(e["event"] == "final_validation_passed" for e in events))

    async def test_no_match_endpoint_stops_once_without_allocation(self):
        result, events = await self.run_endpoint(count=415, locale="vi", no_match=True)
        self.assertIsInstance(result, HTTPException)
        self.assertEqual(result.detail["internal_failure_code"], "ARCH_REPAIR_ASSESSMENT_NO_MATCH")
        self.assertEqual(result.detail["total_repair_provider_calls"], 1)
        self.assertFalse(any(e["event"] == "final_validation_passed" for e in events))
