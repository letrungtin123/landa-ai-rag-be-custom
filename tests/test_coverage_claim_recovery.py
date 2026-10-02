"""Malformed claims are not ownership; offline acceptance + actual SDK wire."""
import asyncio
from copy import deepcopy
import json
import unittest
from unittest.mock import AsyncMock, patch

from app import main
from app.workflows.contracts import WorkflowFailure
from app.lesson_author_provider_schema import staged_provider_response_model
from tests.test_staged_ordered_writer import uat_fixture
from tests.test_checkpoint_component_quality_repair import instance_wire
from tests.test_chapter_checkpoint import checkpoint_result
from tests import test_checkpoint_provider_sdk as sdk_fixture
from tests.staged_schema_probe import capture_sdk_body, visit_schema


def repaired_payload(unit):
    semantic = deepcopy(unit["components"][0]["semantic_content"])
    semantic["sections"][0]["blocks"][0]["text"] += " Verify the documented instruction before taking action."
    return {"components": [{"component_index": 0, "semantic_content": semantic,
                            "covered_source_fact_ids": unit["source_fact_ids"][:]}]}


class CoverageClaimRecoveryTests(unittest.TestCase):
    def test_shape_reason_enums_and_counts_are_safe(self):
        cases = [(None, "ARRAY_REQUIRED"), ({"PRIVATE_KEY": "PRIVATE"}, "ARRAY_REQUIRED"),
                 ([{}, 1], "NON_STRING_ITEM"), (["", "  "], "EMPTY_STRING_ITEM"),
                 (["f", "f"], "DUPLICATE_REFERENCE"), (["PRIVATE"], "OUT_OF_SCOPE_REFERENCE"),
                 ([], "MISSING_OWNED_REFERENCE")]
        for claim, reason in cases:
            baseline = {"source_fact_ids": ["f"], "covered_source_fact_ids": claim}
            before = deepcopy(baseline)
            finding = main.staged_coverage_claim_diagnostic(baseline)
            self.assertIn(reason, finding["reason_codes"])
            self.assertNotIn("PRIVATE", json.dumps(finding))
            self.assertEqual(baseline, before)
        self.assertEqual(main.staged_coverage_claim_diagnostic({"source_fact_ids": ["f"]})["reason_codes"][0], "FIELD_MISSING")

    def test_malformed_claim_recovers_once_and_passes_full_chapter_without_mutation(self):
        for mode in ("duplicate", "null", "object", "non_string", "empty_string", "empty", "missing", "partial", "scalar"):
            with self.subTest(mode=mode):
                request, unit, scope, manifest = uat_fixture()
                broken = instance_wire(unit)
                facts = unit["source_fact_ids"]
                claims = {"duplicate": facts + facts[:1], "null": None, "object": {"PRIVATE_KEY": "PRIVATE"},
                          "non_string": facts + [{}], "empty_string": facts + [""], "empty": [],
                          "partial": facts[:47], "scalar": facts[0]}
                if mode == "missing":
                    broken["components"]["c0"].pop("covered_source_fact_ids")
                else:
                    broken["components"]["c0"]["covered_source_fact_ids"] = claims[mode]
                before = deepcopy(broken)
                delta = repaired_payload(unit)
                provider = AsyncMock(side_effect=[(json.dumps(broken), main.AiUsage()), (json.dumps(delta), main.AiUsage())])
                with patch("app.main.generate_content", provider), self.assertLogs("app.main", "INFO") as logs:
                    result = asyncio.run(checkpoint_result(request, manifest))
                    final_request = request.model_copy(update={"checkpoint_action": "validate_chapter", "checkpoint_unit_index": None,
                        "checkpoint_units": [main.ChapterCheckpointUnit(unit_index=0, unit=result["unit"])]})
                    final = asyncio.run(checkpoint_result(final_request, manifest))
                self.assertEqual(provider.await_count, 2)
                self.assertEqual(final["status"], "ready")
                actual = result["unit"]
                self.assertEqual(actual["components"][1:], unit["components"][1:])
                for key in ("source_fact_ids", "supporting_evidence_fact_ids", "component_plan_id", "type"):
                    self.assertEqual(actual["components"][0][key], unit["components"][0][key])
                self.assertEqual(actual["components"][0]["covered_source_fact_ids"], facts)
                self.assertEqual(actual["components"][0]["semantic_content"], delta["components"][0]["semantic_content"])
                self.assertEqual(before, broken)
                safe = "\n".join(logs.output)
                self.assertIn('"coverage_repair_component_indices": [0]', safe)
                self.assertIn('"coverage_claim_findings":', safe)
                self.assertIn(request.correlation_id, safe)
                self.assertNotIn("PRIVATE", safe)

    def test_unknown_reference_supporting_claim_and_any_ownership_mutation_never_repair(self):
        for mode in ("unknown", "bracket_id", "whitespace_id", "supporting", "other_component", "ownership"):
            with self.subTest(mode=mode):
                request, unit, scope, manifest = uat_fixture()
                wire = instance_wire(unit)
                facts = unit["source_fact_ids"]
                wire["components"]["c0"]["covered_source_fact_ids"] = facts + facts[:1]
                if mode in ("unknown", "bracket_id", "whitespace_id"):
                    wire["components"]["c0"]["covered_source_fact_ids"].append(
                        "PRIVATE_UNKNOWN" if mode == "unknown" else f"[{facts[0]}]" if mode == "bracket_id" else f" {facts[0]}")
                elif mode == "supporting": wire["components"]["c1"]["covered_source_fact_ids"] = facts[:1]
                elif mode == "other_component": wire["components"]["c2"]["component_plan_id"] = "PRIVATE_TARGET"
                else: wire["components"]["c0"]["source_fact_ids"] = facts
                provider = AsyncMock(return_value=(json.dumps(wire), main.AiUsage()))
                with patch("app.main.generate_content", provider), self.assertLogs("app.main", "INFO") as logs:
                    with self.assertRaises(WorkflowFailure) as failure:
                        asyncio.run(checkpoint_result(request, manifest))
                self.assertEqual(provider.await_count, 1)
                self.assertEqual(failure.exception.internal_code, "CHAPTER_UNIT_CONTRACT_REJECTED")
                self.assertNotIn("PRIVATE", "\n".join(logs.output))

    def test_repair_rejects_bad_claim_content_scope_and_third_provider_call(self):
        for mode in ("duplicate", "unknown", "empty", "null", "id_only", "same_content", "wrong_target", "ownership", "bad_html"):
            with self.subTest(mode=mode):
                request, unit, _, manifest = uat_fixture()
                wire = instance_wire(unit)
                wire["components"]["c0"]["covered_source_fact_ids"] = unit["source_fact_ids"] * 2
                delta = repaired_payload(unit)
                change = delta["components"][0]
                if mode == "duplicate": change["covered_source_fact_ids"] *= 2
                elif mode == "unknown": change["covered_source_fact_ids"].append("PRIVATE")
                elif mode == "empty": change["covered_source_fact_ids"] = []
                elif mode == "null": change["covered_source_fact_ids"] = None
                elif mode == "id_only": change.pop("semantic_content")
                elif mode == "same_content": change["semantic_content"] = unit["components"][0]["semantic_content"]
                elif mode == "wrong_target": change["component_index"] = 1
                elif mode == "ownership": change["source_fact_ids"] = unit["source_fact_ids"]
                else: change["semantic_content"] = {"version": 2, "sections": []}
                provider = AsyncMock(side_effect=[(json.dumps(wire), main.AiUsage()), (json.dumps(delta), main.AiUsage())])
                with patch("app.main.generate_content", provider), patch("app.main.build_source_locked_html_unit") as fallback:
                    with self.assertRaises(WorkflowFailure) as failure:
                        asyncio.run(checkpoint_result(request, manifest))
                self.assertEqual(provider.await_count, 2)
                self.assertEqual(failure.exception.internal_code, "CHAPTER_COMPONENT_REPAIR_EXHAUSTED")
                self.assertNotIn("PRIVATE", json.dumps(failure.exception.diagnostics))
                if mode == "duplicate":
                    self.assertIn("DUPLICATE_REFERENCE", failure.exception.diagnostics["coverage_claim_findings"][0]["reason_codes"])
                fallback.assert_not_called()

    def test_actual_sdk_wire_allows_only_owned_ids_initial_and_repair(self):
        request, unit, _, _ = uat_fixture()
        plans = [p.model_dump() for p in request.blueprint_architecture.lessons[0].units[0].component_plan]
        models = [main.build_staged_instance_response_model(plans), main.build_staged_lesson_content_response_model(
            ["html"], payload_only=True, coverage_repair=True, coverage_allowed_ids=unit["source_fact_ids"])]
        for model in models:
            projected, _ = staged_provider_response_model(model)
            wire = capture_sdk_body(projected)["generationConfig"]["responseSchema"]
            claims = [node for path, node in visit_schema(wire) if path.endswith("covered_source_fact_ids")]
            self.assertEqual(claims[0]["items"]["enum"], unit["source_fact_ids"])
            self.assertNotIn("source_fact_ids", json.dumps(wire).replace("covered_source_fact_ids", ""))

    def test_actual_sdk_malformed_claim_raw_fallback_then_repair_and_final_acceptance(self):
        request, unit, _, manifest = uat_fixture()
        wire = instance_wire(unit)
        wire["components"]["c0"]["covered_source_fact_ids"] = None
        delta = repaired_payload(unit)
        del delta["components"][0]["semantic_content"]["version"]
        def body(value):
            return sdk_fixture.response(200, {"candidates": [{"content": {"parts": [{"text": json.dumps(value)}]}, "finishReason": "STOP"}],
                                  "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 20, "totalTokenCount": 30}})
        harness = sdk_fixture.CheckpointProviderSDKTests()
        with patch("httpx.Client.send", side_effect=[body(wire), body(delta)]) as http:
            result = harness.send(request, manifest)
            self.assertEqual(result.status_code, 200, result.text)
            final_request = request.model_copy(update={"checkpoint_action": "validate_chapter", "checkpoint_unit_index": None,
                "checkpoint_units": [main.ChapterCheckpointUnit(unit_index=0, unit=result.json()["unit"])]})
            final = harness.send(final_request, manifest)
        self.assertEqual(final.status_code, 200)
        self.assertEqual(final.json()["status"], "ready")
        self.assertEqual(http.call_count, 2)

    def test_exact_scope_guard_never_trims_or_reassigns_foreign_claims(self):
        _, unit, scope, _ = uat_fixture()
        facts = unit["source_fact_ids"]
        for claim in (facts + [f" {facts[0]}"], facts + [f"[{facts[0]}]"], facts + ["PRIVATE_OTHER_SECTION"]):
            broken = deepcopy(unit)
            broken["components"][0]["covered_source_fact_ids"] = claim
            self.assertEqual(main.staged_component_repair_targets(broken, scope), [])
            finding = main.validate_staged_unit_content(broken, scope, strict_payload=True)
            self.assertEqual(finding.code, "COMPONENT_COVERAGE_OUT_OF_SCOPE")
            self.assertFalse(finding.repairable)
        # A malformed claim cannot hide mutation of a later component's owner.
        broken = deepcopy(unit)
        broken["components"][0]["covered_source_fact_ids"] = None
        broken["components"][2]["source_fact_ids"] = facts[:1]
        self.assertEqual(main.staged_component_repair_targets(broken, scope), [])

    def test_1001_reference_schema_and_diagnostics_are_deterministic(self):
        facts = [f"synthetic-{i}" for i in range(1001)]
        model = main.build_staged_lesson_content_response_model(["html"], payload_only=True,
                    coverage_repair=True, coverage_allowed_ids=facts)
        projected, _ = staged_provider_response_model(model)
        first = capture_sdk_body(projected)
        self.assertEqual(first, capture_sdk_body(projected))
        self.assertLess(len(json.dumps(first)), 40000)
        component = {"source_fact_ids": facts, "covered_source_fact_ids": facts + facts[:1]}
        before = deepcopy(component)
        finding = main.staged_coverage_claim_diagnostic(component)
        self.assertEqual(finding, main.staged_coverage_claim_diagnostic(component))
        self.assertEqual(finding["duplicate_count"], 1)
        self.assertEqual(finding["missing_count"], 0)
        self.assertTrue(main.staged_coverage_claim_repairable(component))
        self.assertEqual(component, before)
