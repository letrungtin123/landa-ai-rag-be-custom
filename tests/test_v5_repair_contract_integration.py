"""Offline production endpoint -> graph -> allocation -> acceptance fixture.

Only retrieval and Gemini are mocked. No private UAT payloads, database writes
or network requests. The Node suite consumes the returned production response.
"""
from __future__ import annotations

import asyncio
import copy
import json
import sys
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from google import genai
from google.genai import models, types

from app.lesson_author_blueprint import (
    ACTION_OBJECTIVE_REPAIR_INTENTS,
    INSTRUCTIONAL_SUPPORT_REPAIR_INTENTS,
    SEMANTIC_LEARNING_BLOCK_INTENTS,
    V5_SEMANTIC_DELTA_REPAIR_OPERATIONS,
    build_v5_semantic_delta_repair_response_schema,
    semantic_delta_required_fields,
    INSTRUCTIONAL_SUPPORT_TEXT_MAX_CHARS,
)
from app.main import (
    AiUsage,
    apply_course_architecture_repair_patches,
    build_course_architecture_repair_prompt,
    lesson_author_blueprint,
    match_staged_unit_by_title,
    validate_v5_post_allocation_instructional_depth,
)
from app.workflows.contracts import WorkflowFailure
from app.workflows.course_architecture import classify_course_repair_targets
from tests.test_evidence_scope_allocation_v5 import _manifest, _nodes, _v5_blueprint
from tests.test_lesson_author_blueprint import blueprint_request
from app.source_map import build_source_map
from app.component_capabilities import ComponentCapabilities


async def endpoint_fixture(count=371, *, intent="worked_example", thin=True, locale="en"):
    manifest = _manifest(count, 6)
    nodes = _nodes(6)
    source_map = build_source_map(nodes, manifest, locale=locale)
    candidate = _v5_blueprint(source_map)
    lesson = candidate["chapters"][0]["lessons"][0]
    unit = lesson["units"][0]
    if thin:
        lesson["learning_objectives"] = [
            "Identify the source-grounded safety control.",
            "Explain the documented purpose of the safety control.",
        ]
        unit["learning_objective_refs"] = ["lo_1", "lo_2"]
        unit["learning_blocks"][0]["learning_objective_refs"] = ["lo_1", "lo_2"]
    lesson["assessment_required"] = True
    lesson["assessment_objective_refs"] = ["lo_1"]
    request = blueprint_request().model_copy(update={
        "locale": locale, "correlation_id": "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
        "component_capabilities": ComponentCapabilities(
            version=2, max_components_per_unit=4, max_assessments_per_unit=3, assessment_enabled=True,
        ),
    })
    calls = []
    events = []

    async def provider(*args, **kwargs):
        # Fail this test if orchestration accidentally retries or generates
        # extra content. All real JSON parsing, guards and validators execute.
        calls.append(kwargs)
        if len(calls) == 1:
            payload = candidate
        elif len(calls) == 2 and thin:
            schema = kwargs["response_schema"].properties["patches"].items
            assert schema.properties["operation"].enum == ["add_instructional_support_block"]
            assert set(schema.properties["intent"].enum) == INSTRUCTIONAL_SUPPORT_REPAIR_INTENTS
            payload = {"patches": [{
                "path": "chapter_1.lesson_1", "operation": "add_instructional_support_block",
                "unit_path": "chapter_1.lesson_1.unit_1", "after_block_id": "lb_1",
                "intent": intent, "learning_objective_refs": ["lo_1", "lo_2"],
                "content": {"purpose": "Demonstrate the documented control and explain its safety implication."},
            }]}
        else:
            raise AssertionError("Unexpected additional provider call")
        return json.dumps(payload), AiUsage(inputTokens=10, outputTokens=10, totalTokens=20)

    def capture(message, *args, **kwargs):
        if message == "lesson_author_blueprint_diagnostic %s":
            events.append(json.loads(args[0]))

    structure = {"source_structure_nodes": nodes, "source_coverage_manifest": manifest,
                 "known_source_refs": {n["source_ref"] for n in nodes}}
    with patch("app.main.retrieve_chunks", new=AsyncMock(return_value=([], AiUsage(), structure))), \
         patch("app.main.generate_content", new=AsyncMock(side_effect=provider)), \
         patch("app.main.logger.info", side_effect=capture):
        result = await lesson_author_blueprint(request, pool=object())
    return result, calls, events


class RepairContractIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def test_staged_title_matching_is_exact_when_normalized_key_is_empty_and_never_positional(self):
        for title in ("Unit 1", "Bài 1", "Mục 2", "章一"):
            with self.subTest(title=title):
                valid = {"title": title, "components": []}
                self.assertIs(match_staged_unit_by_title([valid], title), valid)
                self.assertIsNone(match_staged_unit_by_title([{"title": "Unit 99"}], title))
                self.assertIsNone(match_staged_unit_by_title([valid, dict(valid)], title))
        self.assertIsNone(match_staged_unit_by_title([{"title": ""}], ""))
        self.assertIsNone(match_staged_unit_by_title([{"title": None}], "Unit 1"))
        self.assertIsNone(match_staged_unit_by_title([{"title": "Safety"}, {"title": "Unit 1 Safety"}], "Safety"))
        self.assertEqual(match_staged_unit_by_title([{"title": "Unit 1 Safety"}], "Safety"), {"title": "Unit 1 Safety"})

    async def test_production_endpoint_repairs_depth_revalidates_and_preserves_all_facts(self):
        for count, locale in ((371, "vi"), (1001, "en")):
            with self.subTest(count=count, locale=locale):
                result, calls, events = await endpoint_fixture(count, locale=locale)
                blueprint = result["blueprint"]
                allocation = blueprint["source_fact_allocation"]
                self.assertTrue(allocation["complete"])
                self.assertEqual(allocation["allocated_count"], count)
                self.assertEqual(allocation["unallocated"], [])
                self.assertEqual(len({a["fact_id"] for a in allocation["allocations"]}), count)
                self.assertEqual(len(calls), 2)
                self.assertEqual(result["workflow"]["repair_provider_calls"], 1)
                self.assertFalse(validate_v5_post_allocation_instructional_depth(blueprint).errors)
                blocks = blueprint["chapters"][0]["lessons"][0]["units"][0]["learning_blocks"]
                self.assertEqual([b["intent"] for b in blocks], ["concept_explanation", "worked_example", "knowledge_check"])
                self.assertEqual(blocks[1]["source_fact_ids"], [])
                self.assertEqual(blocks[1]["primary_evidence_scope_ids"], [])
                self.assertEqual(set(blocks[1]["supporting_evidence_scope_ids"]), set(blocks[0]["primary_evidence_scope_ids"]))
                event_names = [e["event"] for e in events]
                self.assertIn("post_allocation_instructional_depth_failed", event_names)
                self.assertIn("final_validation_passed", event_names)
                self.assertTrue(all(e["correlation_id"] == "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee" for e in events))

    async def test_already_valid_blueprint_does_not_repair(self):
        result, calls, _ = await endpoint_fixture(thin=False)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["workflow"]["repair_provider_calls"], 0)

    async def test_invalid_intent_still_fails_closed_without_retry_or_result(self):
        for intent in ("faq", "knowledge_check", "PRIVATE_PROVIDER_VALUE"):
            with self.subTest(intent=intent):
                with self.assertRaises(HTTPException) as raised:
                    await endpoint_fixture(intent=intent)
                self.assertEqual(raised.exception.detail["internal_failure_code"], "ARCH_REPAIR_INVALID_SEMANTIC_INTENT")
                self.assertEqual(raised.exception.detail["total_repair_provider_calls"], 1)

    async def test_support_schema_prompt_and_guard_have_identical_intent_authority(self):
        # Use a real endpoint result, then remove ONLY its supporting treatment
        # to recreate the depth finding against a complete server allocation.
        result, _, _ = await endpoint_fixture()
        baseline = result["blueprint"]
        blocks = baseline["chapters"][0]["lessons"][0]["units"][0]["learning_blocks"]
        blocks[:] = [b for b in blocks if b["intent"] != "worked_example"]
        targets = classify_course_repair_targets(validate_v5_post_allocation_instructional_depth(baseline).errors,
                                                repair_layer="POST_ALLOCATION_INSTRUCTIONAL_DEPTH")
        self.assertEqual(len(targets), 1)
        schema = build_v5_semantic_delta_repair_response_schema({"add_instructional_support_block"})
        self.assertEqual(set(schema.properties["patches"].items.properties["intent"].enum), INSTRUCTIONAL_SUPPORT_REPAIR_INTENTS)
        prompt = build_course_architecture_repair_prompt(blueprint=baseline, targets=targets, source_map_context="{}", locale="en")
        self.assertIn('"allowed_support_intents":' + json.dumps(sorted(INSTRUCTIONAL_SUPPORT_REPAIR_INTENTS), separators=(",", ":")), prompt)
        original = copy.deepcopy(baseline)
        for intent in sorted(SEMANTIC_LEARNING_BLOCK_INTENTS | {"PRIVATE_PROVIDER_VALUE", ""}):
            payload = {"patches": [{"path": targets[0]["path"], "operation": "add_instructional_support_block",
                "unit_path": "chapter_1.lesson_1.unit_1", "after_block_id": "lb_1", "intent": intent,
                "learning_objective_refs": ["lo_1", "lo_2"], "content": {"purpose": "Explain the documented safety implication."}}]}
            if intent in INSTRUCTIONAL_SUPPORT_REPAIR_INTENTS:
                apply_course_architecture_repair_patches(baseline, targets, payload)
            else:
                with self.assertRaises(WorkflowFailure) as raised:
                    apply_course_architecture_repair_patches(baseline, targets, payload)
                self.assertEqual(raised.exception.internal_code, "ARCH_REPAIR_INVALID_SEMANTIC_INTENT")
                self.assertEqual(raised.exception.diagnostics["selected_intent"], intent if intent in SEMANTIC_LEARNING_BLOCK_INTENTS else "UNKNOWN")
                self.assertNotIn("PRIVATE_PROVIDER_VALUE", json.dumps(raised.exception.diagnostics))
            self.assertEqual(baseline, original)

    def test_all_operation_schemas_serialize_without_provenance_and_with_narrow_intents(self):
        client = genai.Client(api_key="offline-test-key")
        expected_intents = {"set_block_intent": ACTION_OBJECTIVE_REPAIR_INTENTS,
                            "add_instructional_support_block": INSTRUCTIONAL_SUPPORT_REPAIR_INTENTS}
        for operation in V5_SEMANTIC_DELTA_REPAIR_OPERATIONS:
            with self.subTest(operation=operation):
                schema = build_v5_semantic_delta_repair_response_schema({operation})
                props = schema.properties["patches"].items.properties
                self.assertEqual(props["operation"].enum, [operation])
                self.assertEqual(semantic_delta_required_fields(operation), set(props))
                self.assertFalse(set(props) & {"source_fact_ids", "source_refs", "primary_evidence_scope_ids", "supporting_evidence_scope_ids", "learning_blocks"})
                if operation in expected_intents:
                    self.assertEqual(set(props["intent"].enum), expected_intents[operation])
                if operation == "add_instructional_support_block":
                    for field in ("purpose", "learner_action"):
                        self.assertEqual(props["content"].properties[field].max_length, INSTRUCTIONAL_SUPPORT_TEXT_MAX_CHARS)
                        self.assertEqual(props["content"].properties[field].min_length, 1)
                models._GenerateContentConfig_to_mldev(client._api_client, types.GenerateContentConfig(response_schema=schema))


if __name__ == "__main__":
    if "--node-fixture" in sys.argv:
        result, _, _ = asyncio.run(endpoint_fixture())
        result["test_manifest"] = _manifest(371, 6)
        print(json.dumps(result))
    else:
        unittest.main()
