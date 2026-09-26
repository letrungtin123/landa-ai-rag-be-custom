"""Offline UAT58f4904a regression: typed slots, strict ownership and local repair."""
import asyncio
from copy import deepcopy
import json
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from google import genai
from google.genai import models, types

from app import main
from app.workflows.contracts import WorkflowFailure
from tests.test_staged_recovery_identity import fixture
from tests.test_checkpoint_component_quality_repair import instance_wire
from tests.test_chapter_checkpoint import fixture as checkpoint_fixture, checkpoint_result


def checkpoint_instance_fixture():
    base, unit, scope, manifest = fixture()
    architecture = base.blueprint_architecture.model_dump()
    plans = architecture["lessons"][0]["units"][0]["component_plan"]
    plans[2]["reason_code"] = "RELATIONSHIP_VISUALIZATION"
    plans[3]["reason_code"] = "FAQ_ANTICIPATED_QUESTIONS"
    base = base.model_copy(update={"blueprint_architecture": type(base.blueprint_architecture).model_validate(architecture)})
    checkpoint, _, _ = checkpoint_fixture(count=1)
    data = {**base.model_dump(), "operation": "create", "target_type": "chapter", "generation_mode": "staged",
            "source_documents": checkpoint.model_dump()["source_documents"], "correlation_id": checkpoint.correlation_id,
            "checkpoint_action": "generate_unit", "checkpoint_unit_index": 0, "remaining_workflow_budget_ms": 400_000}
    return main.RagLessonAuthorCheckpointRequest.model_validate(data), unit, scope, manifest


class StagedInstanceOutputTests(unittest.TestCase):
    def test_sdk_has_per_instance_required_payloads_without_ownership_or_union(self):
        _, _, scope, _ = fixture()
        plans = deepcopy(scope["component_plan"])
        schema = main.build_staged_instance_response_model(plans)
        client = genai.Client(api_key="test-key")
        wire = models._GenerateContentConfig_to_mldev(client._api_client, types.GenerateContentConfig(response_schema=schema))["responseSchema"]
        slots = wire["properties"]["components"]
        self.assertEqual(set(slots.required), {f"c{i}" for i in range(4)})
        for key in ("nodes", "edges"):
            self.assertIn(key, slots.properties["c2"].required)
        self.assertEqual(slots.properties["c2"].properties["edges"].min_items, 1)
        self.assertEqual(set(slots.properties["c3"].properties["items"].items.properties), {"question", "answer"})
        self.assertIn("items", slots.properties["c3"].required)
        self.assertIn("semantic_content", slots.properties["c0"].required)
        for slot in slots.properties.values():
            self.assertFalse({"type", "source_fact_ids", "supporting_evidence_fact_ids", "component_plan_id"}.intersection(slot.properties))
            self.assertIn("covered_source_fact_ids", slot.required)
        self.assertTrue(main.staged_response_schema_diagnostics(schema)["schema_valid"])
        for kinds in (("html", "la_sortable", "la_crossword", "la_faq"), ("html", "html", "problem", "la_faq")):
            extra_plans = [{**deepcopy(plans[i]), "type": kind} for i, kind in enumerate(kinds)]
            other = main.build_staged_instance_response_model(extra_plans)
            other_wire = models._GenerateContentConfig_to_mldev(client._api_client, types.GenerateContentConfig(response_schema=other))["responseSchema"]
            other_slots = other_wire["properties"]["components"].properties
            if kinds[1] == "la_sortable":
                self.assertEqual(set(other_slots["c1"].properties["items"].items.properties), {"text"})
                self.assertEqual(set(other_slots["c3"].properties["items"].items.properties), {"question", "answer"})
                self.assertIn("items", other_slots["c1"].required)
                self.assertIn("words", other_slots["c2"].required)
            else:
                self.assertIn("semantic_content", other_slots["c1"].required)

    def test_binding_is_server_owned_without_auto_filling_coverage(self):
        _, valid, scope, _ = fixture()
        expected = {**scope, "unit_title": valid["title"]}
        wire = instance_wire(valid)
        wire["components"]["c0"]["covered_source_fact_ids"] = []
        before = deepcopy(wire)
        bound = main.bind_staged_instance_payload(wire, expected)
        self.assertEqual(bound["components"][0]["source_fact_ids"], scope["component_plan"][0]["source_fact_ids"])
        self.assertEqual(bound["components"][0]["covered_source_fact_ids"], [])
        self.assertEqual(main.staged_component_repair_guard(bound, expected)["code"], "COMPONENT_COVERAGE_INCOMPLETE")
        self.assertEqual(wire, before)
        del wire["components"]["c0"]["covered_source_fact_ids"]
        bound = main.bind_staged_instance_payload(wire, expected)
        self.assertNotIn("covered_source_fact_ids", bound["components"][0])
        self.assertEqual(main.validate_staged_unit_content(bound, expected, strict_payload=True).code, "COMPONENT_COVERAGE_MISSING")

    def test_unknown_slots_or_protected_fields_are_never_silently_dropped(self):
        _, valid, scope, _ = fixture()
        expected = {**scope, "unit_title": valid["title"]}
        for change in ("slot", "missing", "owned", "identity", "type", "asset", "envelope"):
            wire = instance_wire(valid)
            if change == "slot": wire["components"]["PRIVATE_SLOT"] = {}
            elif change == "missing": del wire["components"]["c2"]
            elif change == "envelope": wire["PRIVATE_FIELD"] = "PRIVATE_TEXT"
            else:
                key = {"owned": "source_fact_ids", "identity": "component_plan_id", "type": "type", "asset": "asset_url"}[change]
                wire["components"]["c2"][key] = "PRIVATE_VALUE"
            with self.assertRaises(WorkflowFailure) as failure:
                main.bind_staged_instance_payload(wire, expected)
            self.assertNotIn("PRIVATE", json.dumps(failure.exception.diagnostics))

    def test_every_repair_authority_rejection_has_a_safe_reason(self):
        _, valid, scope, _ = fixture()
        cases = [("source_fact_ids", ["PRIVATE"], "COMPONENT_EVIDENCE_MEMBERSHIP_INVALID"),
                 ("supporting_evidence_fact_ids", ["PRIVATE"], "COMPONENT_EVIDENCE_MEMBERSHIP_INVALID"),
                 ("component_plan_id", "PRIVATE", "COMPONENT_PLAN_INSTANCE_MISMATCH"),
                 ("type", "PRIVATE", "COMPONENT_TYPE_PLAN_MISMATCH"),
                 ("covered_source_fact_ids", None, "INVALID_COVERAGE_ID_ARRAY"),
                 ("covered_source_fact_ids", ["PRIVATE"], "COMPONENT_COVERAGE_OUT_OF_SCOPE"),
                 ("covered_source_fact_ids", ["fact-0"], "SUPPORTING_COMPONENT_CLAIMS_OWNERSHIP")]
        for key, value, reason in cases:
            broken = deepcopy(valid)
            broken["components"][2][key] = value
            diagnostic = main.staged_component_repair_guard(broken, scope)
            self.assertEqual(diagnostic["code"], reason)
            self.assertEqual(main.staged_component_repair_targets(broken, scope), [])
            self.assertNotIn("PRIVATE", json.dumps(diagnostic))
        broken = deepcopy(valid)
        broken["components"][0]["covered_source_fact_ids"] = []
        self.assertEqual(main.staged_component_repair_guard(broken, scope)["code"], "COMPONENT_COVERAGE_INCOMPLETE")
        broken["components"].pop()
        self.assertEqual(main.staged_component_repair_guard(broken, scope)["code"], "COMPONENT_COUNT_MISMATCH")

    def test_uat_missing_edges_repairs_diagram_only_and_final_chapter_validates(self):
        request, valid, scope, manifest = checkpoint_instance_fixture()
        broken = instance_wire(valid)
        del broken["components"]["c2"]["edges"]
        delta = {"components": [{"component_index": 2, "edges": valid["components"][2]["edges"]}]}
        provider = AsyncMock(side_effect=[(json.dumps(broken), main.AiUsage()), (json.dumps(delta), main.AiUsage())])
        with patch("app.main.generate_content", provider), self.assertLogs("app.main", "INFO") as logs:
            result = asyncio.run(checkpoint_result(request, manifest))
            final_request = request.model_copy(update={"checkpoint_action": "validate_chapter", "checkpoint_unit_index": None,
                "checkpoint_units": [main.ChapterCheckpointUnit(unit_index=0, unit=result["unit"])]})
            final = asyncio.run(checkpoint_result(final_request, manifest))
        self.assertEqual(provider.await_count, 2)
        self.assertEqual(result["status"], "unit_ready")
        self.assertEqual(final["status"], "ready")
        for i, actual in enumerate(result["unit"]["components"]):
            self.assertEqual(actual["source_fact_ids"], scope["component_plan"][i]["source_fact_ids"])
            self.assertEqual(actual["supporting_evidence_fact_ids"], scope["component_plan"][i]["supporting_evidence_fact_ids"])
            for key in main.STAGED_COMPONENT_PAYLOAD_FIELDS[actual["type"]]:
                self.assertEqual(actual.get(key), valid["components"][i].get(key))
        self.assertIn('"repair_component_indices": [2]', "\n".join(logs.output))
        self.assertIn('"repair_guard_finding": null', "\n".join(logs.output))

    def test_invalid_coverage_still_fails_before_repair_with_reason(self):
        request, valid, _, manifest = checkpoint_instance_fixture()
        wire = instance_wire(valid)
        wire["components"]["c2"].pop("edges")
        wire["components"]["c2"]["covered_source_fact_ids"] = ["fact-0"]
        provider = AsyncMock(return_value=(json.dumps(wire), main.AiUsage()))
        with patch("app.main.generate_content", provider), self.assertLogs("app.main", "INFO") as logs:
            with self.assertRaises(WorkflowFailure) as failure:
                asyncio.run(checkpoint_result(request, manifest))
        self.assertEqual(provider.await_count, 1)
        self.assertEqual(failure.exception.diagnostics["repair_guard_finding"]["code"], "SUPPORTING_COMPONENT_CLAIMS_OWNERSHIP")
        self.assertIn("SUPPORTING_COMPONENT_CLAIMS_OWNERSHIP", "\n".join(logs.output))

    def test_http_endpoint_returns_typed_safe_binding_failure(self):
        request, _, _, manifest = checkpoint_instance_fixture()
        provider = AsyncMock(return_value=(json.dumps({"components": {"PRIVATE_SLOT": {"PRIVATE_DATA": "PRIVATE_TEXT"}}}), main.AiUsage()))
        async def generation(req, _pool):
            return await checkpoint_result(req, manifest)
        async def send():
            main.app.dependency_overrides[main.get_db] = lambda: None
            try:
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
                    return await client.post("/v1/lesson-author/chapter-checkpoint", json=request.model_dump(),
                                             headers={"X-Landa-AI-Service-Token": "fixture-internal-token"})
            finally:
                main.app.dependency_overrides.pop(main.get_db, None)
        with patch.object(main.settings, "service_token", "fixture-internal-token"), patch("app.main.generate_content", provider), \
                patch("app.main.lesson_author_proposal", generation), self.assertLogs("app.main", "INFO") as logs:
            result = asyncio.run(send())
        self.assertEqual(result.status_code, 422)
        self.assertEqual(result.json()["detail"]["failure_stage"], "chapter_component_binding")
        self.assertEqual(result.json()["detail"]["correlation_id"], request.correlation_id)
        self.assertIn("INSTANCE_SLOT_INVENTORY_INVALID", "\n".join(logs.output))
        self.assertNotIn("PRIVATE", result.text + "\n".join(logs.output))
        self.assertEqual(provider.await_count, 1)


if __name__ == "__main__":
    unittest.main()
