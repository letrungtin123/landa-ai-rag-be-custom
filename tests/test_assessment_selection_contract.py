"""Offline provider choices -> existing mutation guards -> final allocation."""
import copy
import json
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from app.assessment_planner import compile_v5_assessment_plan
from app.assessment_selection_contract import VERSION, build_assessment_selection_contract
from app.component_capabilities import ComponentCapabilities
from app.main import apply_course_architecture_repair_patches, lesson_author_blueprint
from app.schemas.common import AiUsage
from app.services.lesson_author.architecture_validation import validate_v5_instructional_coherence
from app.services.lesson_author.evidence_scope import (
    allocate_source_map_architecture_facts,
    validate_course_architecture_evidence_scope,
)
from app.source_map import build_source_map
from app.workflows.contracts import WorkflowFailure
from tests.staged_schema_probe import capture_sdk_body
from tests.test_assessment_intent_repair import missing_check_fixture, missing_check_payload, targets_for, unit_of
from tests.test_evidence_scope_allocation_v5 import _manifest, _nodes, _v5_blueprint
from tests.test_lesson_author_blueprint import blueprint_request


def selected_choices(contract):
    # Test provider explicitly chooses a teaching treatment; production never
    # automatically selects first/unique candidates or changes their intent.
    return {"choices": {slot: next(key for key, value in binding["options"].items()
            if value.get("intent") == "concept_explanation") for slot, binding in contract.bindings.items()}}


class AssessmentSelectionContractTests(unittest.TestCase):
    def test_decode_preserves_existing_guard_and_complete_allocation(self):
        for count in (371, 415, 1001):
            with self.subTest(count=count):
                candidate, source_map, manifest = missing_check_fixture(facts=count)
                baseline = copy.deepcopy(candidate)
                targets = targets_for(candidate, source_map)
                contract = build_assessment_selection_contract(targets, max_context_chars=16000)
                decoded = contract.decode_text(json.dumps(selected_choices(contract)))
                self.assertEqual(decoded, missing_check_payload(targets))
                repaired = apply_course_architecture_repair_patches(candidate, targets, decoded)
                self.assertEqual(candidate, baseline)
                self.assertFalse(validate_course_architecture_evidence_scope(repaired, source_map).errors)
                self.assertFalse(validate_v5_instructional_coherence(repaired).errors)
                self.assertEqual(compile_v5_assessment_plan(repaired).status, "READY")
                allocation = allocate_source_map_architecture_facts(repaired, source_map, manifest)["source_fact_allocation"]
                self.assertTrue(allocation["complete"])
                self.assertEqual(allocation["allocated_count"], count)
                self.assertEqual(allocation["unallocated"], [])
                self.assertEqual(len({a["fact_id"] for a in allocation["allocations"]}), count)
                for field in ("source_refs", "primary_evidence_scope_ids", "supporting_evidence_scope_ids", "concept_ids"):
                    self.assertEqual(unit_of(repaired)["learning_blocks"][0][field], unit_of(baseline)["learning_blocks"][0][field])

    def test_sdk_wire_is_required_enum_choices_not_mutation_payloads(self):
        candidate, source_map, _ = missing_check_fixture()
        targets = targets_for(candidate, source_map)
        contract = build_assessment_selection_contract(targets, max_context_chars=16000)
        wire = capture_sdk_body(contract.schema)["generationConfig"]["responseSchema"]
        self.assertEqual(set(wire["properties"]), {"choices"})
        slots = wire["properties"]["choices"]
        self.assertEqual(set(slots["required"]), set(contract.bindings))
        for slot, node in slots["properties"].items():
            self.assertEqual(node["type"], "STRING")
            self.assertEqual(node["enum"], list(contract.bindings[slot]["options"]))
        for forbidden in ("source_fact_ids", "source_refs", "unit_path", "teaching_block_id", "operation", "selections"):
            self.assertNotIn(forbidden, json.dumps(wire))
        self.assertEqual(contract.descriptors, build_assessment_selection_contract(targets, max_context_chars=16000).descriptors)
        self.assertIn("Vietnamese", contract.prompt("vi"))
        self.assertIn("English", contract.prompt("en"))
        with self.assertRaises(WorkflowFailure) as caught:
            build_assessment_selection_contract(targets, max_context_chars=10)
        self.assertEqual(caught.exception.diagnostics["guard_reason"], "SELECTION_CONTEXT_BUDGET_EXCEEDED")

    def test_malformed_choices_fail_typed_without_private_values_or_partial_apply(self):
        candidate, source_map, _ = missing_check_fixture()
        targets = targets_for(candidate, source_map)
        contract = build_assessment_selection_contract(targets, max_context_chars=16000)
        valid = selected_choices(contract)
        cases = ["not JSON PRIVATE", json.dumps([]), json.dumps({**valid, "PRIVATE": "PRIVATE"}),
                 json.dumps({"choices": {}}), json.dumps({"choices": {**valid["choices"], "PRIVATE": "PRIVATE"}}),
                 json.dumps({"choices": {**valid["choices"], "s0": "PRIVATE"}}),
                 json.dumps({"choices": {**valid["choices"], "s0": None}}),
                 json.dumps({"choices": {**valid["choices"], "s0": valid["choices"]["s1"]}}),
                 '{"choices":{},"choices":{}}', '{"choices":{"s0":"NO_MATCH","s0":"NO_MATCH"}}']
        for value in cases:
            with self.subTest(value=value), self.assertRaises(WorkflowFailure) as caught:
                contract.decode_text(value)
            self.assertNotIn("PRIVATE", str(caught.exception) + json.dumps(caught.exception.diagnostics))
            self.assertIn(caught.exception.internal_code, {"ARCH_REPAIR_JSON_INVALID", "ARCH_REPAIR_SELECTION_CONTRACT_INVALID"})
        with self.assertRaises(WorkflowFailure):
            contract.decode_text(json.dumps(missing_check_payload(targets)))

    def test_no_match_and_stale_authority_still_fail_existing_guard(self):
        candidate, source_map, _ = missing_check_fixture()
        targets = targets_for(candidate, source_map)
        contract = build_assessment_selection_contract(targets, max_context_chars=16000)
        payload = contract.decode_text(json.dumps({"choices": dict.fromkeys(contract.bindings, "NO_MATCH")}))
        with self.assertRaises(WorkflowFailure) as caught:
            apply_course_architecture_repair_patches(candidate, targets, payload)
        self.assertEqual(caught.exception.internal_code, "ARCH_REPAIR_ASSESSMENT_NO_MATCH")
        payload = contract.decode_text(json.dumps(selected_choices(contract)))
        unit_of(candidate)["learning_blocks"][0]["source_refs"] = ["different-source"]
        baseline = copy.deepcopy(candidate)
        with self.assertRaises(WorkflowFailure):
            apply_course_architecture_repair_patches(candidate, targets, payload)
        self.assertEqual(candidate, baseline)


async def selection_endpoint_fixture(*, invalid=False, count=415, locale="en"):
    nodes, manifest = _nodes(7), _manifest(count, 7)
    source_map = build_source_map(nodes, manifest, locale=locale)
    candidate = _v5_blueprint(source_map)
    for chapter_index in (2, 5):
        lesson = candidate["chapters"][chapter_index]["lessons"][0]
        lesson.update(assessment_required=True, assessment_objective_refs=["lo_1"])
        lesson["units"][0]["learning_blocks"][0].update(
            intent="introduction", content={"purpose": "Explain the documented concepts and their meaning."})
    targets = targets_for(candidate, source_map)
    assert [t["path"] for t in targets] == ["chapter_3.lesson_1", "chapter_6.lesson_1"]
    contract = build_assessment_selection_contract(targets, max_context_chars=16000)
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
            schema = kwargs["response_schema"]
            assert set(schema.properties) == {"choices"}
            assert schema.properties["choices"].required == ["s0", "s1"]
            assert kwargs["max_output_tokens"] == min(request.max_output_tokens, 16384)
            assert kwargs["request_timeout_ms"] == 300000
            payload = {**choices, "PRIVATE_EXTRA": "PRIVATE_CONTENT"} if invalid else choices
        else:
            raise AssertionError("Unexpected additional provider call")
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
    return result, calls, events


class AssessmentSelectionEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_target_production_endpoint_reaches_final_allocation(self):
        for count, locale in ((415, "vi"), (1001, "en")):
            with self.subTest(count=count):
                result, calls, events = await selection_endpoint_fixture(count=count, locale=locale)
                self.assertNotIsInstance(result, HTTPException, getattr(result, "detail", None))
                allocation = result["blueprint"]["source_fact_allocation"]
                self.assertEqual(allocation["allocated_count"], count)
                self.assertTrue(allocation["complete"])
                self.assertEqual(allocation["unallocated"], [])
                self.assertEqual(len({a["fact_id"] for a in allocation["allocations"]}), count)
                self.assertEqual(len(calls), 2)
                self.assertEqual(result["workflow"]["repair_provider_calls"], 1)
                self.assertTrue(any(e.get("selection_contract_version") == VERSION for e in events))
                self.assertTrue(any(e["event"] == "final_validation_passed" for e in events))
                self.assertTrue(all(e["correlation_id"] == "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee" for e in events))

    async def test_bad_selection_fails_once_with_safe_structured_diagnostics(self):
        result, calls, events = await selection_endpoint_fixture(invalid=True)
        self.assertIsInstance(result, HTTPException)
        self.assertEqual(result.detail["internal_failure_code"], "ARCH_REPAIR_SELECTION_CONTRACT_INVALID")
        self.assertEqual(result.detail["total_repair_provider_calls"], 1)
        self.assertEqual(len(calls), 2)
        self.assertNotIn("PRIVATE_", json.dumps(events))
        self.assertTrue(any(e.get("guard_reason") == "SELECTION_ENVELOPE_INVALID" for e in events))
