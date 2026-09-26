"""Offline UAT8760cefd: 36 unit facts, crossword 14/23, nine missing.

Synthetic payloads test contracts, not a claim of semantic/factual accuracy.
No DB, network, provider or live UAT execution.
"""
import asyncio
from copy import deepcopy
import json
import unittest
from unittest.mock import AsyncMock, patch

from google import genai
from google.genai import models, types

from app import main
from app.workflows.contracts import WorkflowFailure
from tests.test_staged_instance_output import checkpoint_instance_fixture
from tests.test_checkpoint_component_quality_repair import instance_wire
from tests.test_chapter_checkpoint import checkpoint_result
from tests.test_lesson_prompt_policy import content_payload


def coverage_fixture():
    request, unit, scope, _ = checkpoint_instance_fixture()
    facts = [f"synthetic-fact-{i}" for i in range(36)]
    owned_by_crossword = facts[13:]
    unit.update(source_fact_ids=facts[:], supporting_evidence_fact_ids=facts[:13])
    scope.update(source_fact_ids=facts[:], supporting_evidence_fact_ids=facts[:13])
    unit["components"][2] = content_payload("la_crossword", "vi")
    for i, (component, plan) in enumerate(zip(unit["components"], scope["component_plan"])):
        owned = facts if i == 0 else owned_by_crossword if i == 2 else []
        supporting = facts[:13] if i in (1, 3) else []
        plan.update(type=component["type"], source_fact_ids=owned[:], supporting_evidence_fact_ids=supporting[:])
        component.update(component_plan_id=plan["component_plan_id"], source_fact_ids=owned[:],
                         covered_source_fact_ids=owned[:], supporting_evidence_fact_ids=supporting[:])
    scope["component_plan"][2]["reason_code"] = "TERMINOLOGY_REINFORCEMENT"
    scope["component_plan"][3]["reason_code"] = "FAQ_ANTICIPATED_QUESTIONS"
    scope["component_types"] = [p["type"] for p in scope["component_plan"]]
    architecture = request.blueprint_architecture.model_dump()
    architecture["lessons"][0]["units"][0] = scope
    request = request.model_copy(update={"blueprint_architecture": type(request.blueprint_architecture).model_validate(architecture)})
    manifest = {"facts": [{"fact_id": f, "text": "Synthetic instruction: check conditions, act, then confirm the result."} for f in facts]}
    manifest["supporting_evidence_facts"] = deepcopy(manifest["facts"][:13])
    broken = deepcopy(unit)
    broken["components"][2]["covered_source_fact_ids"] = owned_by_crossword[:14]
    words = deepcopy(unit["components"][2]["words"])
    words[0]["clue"] = "Kiểm tra điều kiện đã được phê duyệt trước khi thực hiện quy trình."
    delta = {"components": [{"component_index": 2, "words": words, "covered_source_fact_ids": owned_by_crossword[:]}]}
    return request, unit, broken, scope, manifest, delta


class CheckpointCoverageRepairTests(unittest.TestCase):
    def test_partial_claim_is_not_ownership_mutation_and_only_crossword_is_targeted(self):
        _, _, broken, scope, _, _ = coverage_fixture()
        self.assertIsNone(main.staged_component_repair_guard(broken, scope, allow_partial_coverage=True))
        finding = main.validate_staged_unit_content(broken, scope, strict_payload=True)
        self.assertEqual(finding.code, "COMPONENT_COVERAGE_INCOMPLETE")
        self.assertTrue(finding.repairable)
        self.assertEqual(main.staged_component_repair_targets(broken, scope), [2])
        self.assertEqual(main.staged_coverage_repair_diagnostics(broken, [2]), [{
            "code": "COMPONENT_COVERAGE_INCOMPLETE", "component_index": 2, "component_type": "la_crossword",
            "owned_count": 23, "covered_owned_count": 14, "missing_count": 9,
        }])

    def test_scoped_content_repair_revalidates_full_chapter_without_auto_filling(self):
        request, valid, broken, scope, manifest, delta = coverage_fixture()
        before = deepcopy(broken)
        provider = AsyncMock(side_effect=[(json.dumps(instance_wire(broken)), main.AiUsage()), (json.dumps(delta), main.AiUsage())])
        with patch("app.main.generate_content", provider), self.assertLogs("app.main", "INFO") as logs:
            result = asyncio.run(checkpoint_result(request, manifest))
            final_request = request.model_copy(update={"checkpoint_action": "validate_chapter", "checkpoint_unit_index": None,
                "checkpoint_units": [main.ChapterCheckpointUnit(unit_index=0, unit=result["unit"])]})
            final = asyncio.run(checkpoint_result(final_request, manifest))
        self.assertEqual(final["status"], "ready")
        self.assertEqual(provider.await_count, 2)
        self.assertEqual(broken, before)
        actual = result["unit"]["components"]
        for i in (0, 1, 3):
            self.assertEqual(actual[i], valid["components"][i])
        for key in ("source_fact_ids", "supporting_evidence_fact_ids", "component_plan_id", "type"):
            self.assertEqual(actual[2][key], before["components"][2][key])
        self.assertEqual(actual[2]["words"], delta["components"][0]["words"])
        self.assertEqual(actual[2]["covered_source_fact_ids"], delta["components"][0]["covered_source_fact_ids"])
        self.assertIsNone(main.validate_staged_unit_content(result["unit"], scope, strict_payload=True))
        repair = provider.await_args_list[1]
        self.assertEqual(repair.kwargs["request_timeout_ms"], 180000)
        wire = json.dumps(repair.kwargs["response_schema"].model_json_schema())
        self.assertIn('"covered_source_fact_ids"', wire)
        for field in ("source_fact_ids", "supporting_evidence_fact_ids", "component_plan_id", "nodes", "semantic_content"):
            self.assertNotIn(f'"{field}"', wire)
        safe = "\n".join(logs.output)
        self.assertIn('"coverage_repair_component_indices": [2]', safe)
        self.assertIn('"missing_count": 9', safe)
        self.assertIn(main.STAGED_COMPONENT_COVERAGE_REPAIR_CONTRACT_VERSION, safe)
        self.assertNotIn(delta["components"][0]["words"][0]["clue"], safe)

    def test_bad_claim_or_id_only_repair_fails_closed_in_existing_one_call_budget(self):
        for mutation, code in (("id_only", "COMPONENT_REPAIR_COVERAGE_WITHOUT_CONTENT"),
                               ("same_content", "COMPONENT_REPAIR_COVERAGE_WITHOUT_CONTENT"),
                               ("incomplete", "COMPONENT_REPAIR_COVERAGE_CLAIM_INVALID"),
                               ("unknown", "COMPONENT_REPAIR_COVERAGE_CLAIM_INVALID"),
                               ("other_unit_fact", "COMPONENT_REPAIR_COVERAGE_CLAIM_INVALID"),
                               ("duplicate", "COMPONENT_REPAIR_COVERAGE_CLAIM_INVALID"),
                               ("invalid_payload", "COMPONENT_PAYLOAD_SCHEMA_INVALID"),
                               ("ownership", "COMPONENT_REPAIR_PROTECTED_FIELD_EMITTED"),
                               ("other_component", "COMPONENT_REPAIR_TARGET_INVALID")):
            with self.subTest(mutation=mutation):
                request, _, broken, _, manifest, delta = coverage_fixture()
                change = delta["components"][0]
                if mutation == "id_only": change.pop("words")
                elif mutation == "same_content": change["words"] = deepcopy(broken["components"][2]["words"])
                elif mutation == "incomplete": change["covered_source_fact_ids"].pop()
                elif mutation == "unknown": change["covered_source_fact_ids"].append("PRIVATE_UNKNOWN_FACT")
                elif mutation == "other_unit_fact": change["covered_source_fact_ids"].append("synthetic-fact-0")
                elif mutation == "duplicate": change["covered_source_fact_ids"].append(change["covered_source_fact_ids"][0])
                elif mutation == "invalid_payload": change["words"] = change["words"][:2]
                elif mutation == "ownership": change["source_fact_ids"] = ["PRIVATE_UNKNOWN_FACT"]
                elif mutation == "other_component": change["component_index"] = 0
                provider = AsyncMock(side_effect=[(json.dumps(instance_wire(broken)), main.AiUsage()), (json.dumps(delta), main.AiUsage())])
                with patch("app.main.generate_content", provider), patch("app.main.build_source_locked_html_unit") as fallback:
                    with self.assertRaises(WorkflowFailure) as failure:
                        asyncio.run(checkpoint_result(request, manifest))
                self.assertEqual(provider.await_count, 2)
                self.assertEqual(failure.exception.internal_code, "CHAPTER_COMPONENT_REPAIR_EXHAUSTED")
                self.assertEqual(failure.exception.diagnostics["repair_failure_code"], code)
                fallback.assert_not_called()

    def test_unknown_fact_or_invalid_other_component_blocks_repair_for_entire_unit(self):
        for mutation in ("unknown", "duplicate", "support_claim", "ownership"):
            request, _, broken, scope, manifest, _ = coverage_fixture()
            if mutation == "unknown": broken["components"][2]["covered_source_fact_ids"].append("PRIVATE_UNKNOWN_FACT")
            elif mutation == "duplicate": broken["components"][2]["covered_source_fact_ids"] *= 2
            elif mutation == "support_claim": broken["components"][3]["covered_source_fact_ids"] = ["synthetic-fact-0"]
            else: broken["components"][2]["source_fact_ids"] = ["PRIVATE_UNKNOWN_FACT"]
            self.assertEqual(main.staged_component_repair_targets(broken, scope), [])
            # Wire binding rejects ownership injection before the repair guard.
            wire = instance_wire(broken)
            if mutation == "ownership": wire["components"]["c2"]["source_fact_ids"] = ["PRIVATE_UNKNOWN_FACT"]
            provider = AsyncMock(return_value=(json.dumps(wire), main.AiUsage()))
            with patch("app.main.generate_content", provider):
                with self.assertRaises(WorkflowFailure):
                    asyncio.run(checkpoint_result(request, manifest))
            self.assertEqual(provider.await_count, 1)

    def test_coverage_edits_require_explicit_target_authority_and_are_atomic(self):
        _, _, broken, _, _, delta = coverage_fixture()
        before = deepcopy(broken)
        with self.assertRaisesRegex(main.LessonAuthorProposalValidationError, "PROTECTED_FIELD"):
            main.merge_staged_component_payload_delta(broken, delta, [2])
        delta["components"][0]["words"] = []
        with self.assertRaises(main.LessonAuthorProposalValidationError):
            main.merge_staged_component_payload_delta(broken, delta, [2], coverage_targets=[2])
        self.assertEqual(broken, before)

    def test_valid_output_does_not_repair(self):
        request, valid, _, _, manifest, _ = coverage_fixture()
        provider = AsyncMock(return_value=(json.dumps(instance_wire(valid)), main.AiUsage()))
        with patch("app.main.generate_content", provider):
            result = asyncio.run(checkpoint_result(request, manifest))
        self.assertEqual(result["status"], "unit_ready")
        self.assertEqual(provider.await_count, 1)

    def test_actual_sdk_serializes_selected_repair_schema_without_protected_fields(self):
        schema = main.build_staged_lesson_content_response_model(["la_crossword"], payload_only=True, coverage_repair=True)
        client = genai.Client(api_key="offline-placeholder")
        wire = models._GenerateContentConfig_to_mldev(client._api_client, types.GenerateContentConfig(response_schema=schema))["responseSchema"]
        fields = wire["properties"]["components"].items.properties
        self.assertIn("covered_source_fact_ids", fields)
        self.assertIn("words", fields)
        self.assertFalse({"source_fact_ids", "component_plan_id", "supporting_evidence_fact_ids", "type"}.intersection(fields))


if __name__ == "__main__":
    unittest.main()
