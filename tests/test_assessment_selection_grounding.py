"""Repair must see real canonical signal even when Architect content is {}."""
import copy
import json
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from app.assessment_planner import materialize_assessment_source_refs
from app.assessment_selection_contract import build_assessment_selection_contract
from app.component_capabilities import ComponentCapabilities
from app.lesson_author_blueprint import _semantic_learning_block_response_schema
from app.schemas.common import AiUsage
from app.services.lesson_author.blueprint_workflow import lesson_author_blueprint
from app.services.lesson_author.repair.semantic_delta import apply_course_architecture_repair_patches
from app.workflows.contracts import WorkflowFailure
from tests.test_assessment_canonical_provenance import fixture
from tests.test_assessment_intent_repair import targets_for
from tests.test_assessment_selection_contract import selected_choices
from tests.test_lesson_author_blueprint import blueprint_request


def schema_realistic_fixture(count=415, locale="en"):
    candidate, source_map, manifest, nodes = fixture(count, locale)
    # Architect's content schema contains treatment flags, NOT purpose/action
    # prose. Do not give test candidates descriptors production cannot emit.
    for chapter in candidate["chapters"]:
        for lesson in chapter["lessons"]:
            for unit in lesson["units"]:
                for block in unit["learning_blocks"]:
                    block["content"] = {}
                    if block["intent"] == "relationship_visualization":
                        block["intent"] = "introduction"
    for fact in manifest["facts"]:
        fact["text"] = "SOURCE_ONLY_SIGNAL: Identify the defined concept, explain its relationships, then apply the documented procedure."
    return candidate, source_map, manifest, nodes


def grounded(candidate, source_map, manifest, *, budget=16000):
    projected, _, issues = materialize_assessment_source_refs(candidate, source_map)
    assert not issues
    targets = targets_for(projected, source_map)
    return projected, targets, build_assessment_selection_contract(
        targets, max_context_chars=budget, blueprint=projected, source_map=source_map, manifest=manifest,
    )


class AssessmentSelectionGroundingTests(unittest.TestCase):
    def test_real_schema_excludes_old_descriptors_but_repair_receives_evidence(self):
        properties = _semantic_learning_block_response_schema().properties["content"].properties
        self.assertNotIn("purpose", properties)
        self.assertNotIn("learner_action", properties)
        candidate, source_map, manifest, _ = schema_realistic_fixture()
        baseline = copy.deepcopy((candidate, source_map, manifest))
        projected, targets, contract = grounded(candidate, source_map, manifest)
        old_style = build_assessment_selection_contract(targets, max_context_chars=16000)
        self.assertNotIn("SOURCE_ONLY_SIGNAL", old_style.descriptors)
        self.assertIn("SOURCE_ONLY_SIGNAL", contract.descriptors)
        self.assertEqual((candidate, source_map, manifest), baseline)
        body = json.loads(contract.descriptors)
        self.assertTrue(body["canonical_evidence"])
        for objective in body["objectives"].values():
            for option in objective["candidates"]:
                self.assertEqual(set(option["semantic_descriptor"]), {"intent"})
                self.assertTrue(option["unit_title"])
                self.assertTrue(option["unit_purpose"])
                self.assertTrue(option["evidence_keys"])
                self.assertTrue(all(k in body["canonical_evidence"] for k in option["evidence_keys"]))
        for item in manifest["facts"]:
            self.assertNotIn(item["fact_id"], contract.descriptors)
        for scope in source_map["source_evidence_scopes"]:
            self.assertNotIn(scope["id"], contract.descriptors)
        self.assertNotIn("SOURCE_ONLY_SIGNAL", json.dumps(contract.diagnostics))
        self.assertGreater(contract.diagnostics["selection_evidence_chars"], 0)
        self.assertEqual(contract.descriptors, grounded(candidate, source_map, manifest)[2].descriptors)
        self.assertEqual(contract.bindings, old_style.bindings)

    def test_samples_cover_beginning_middle_end_only_of_approved_primary_scopes(self):
        candidate, source_map, manifest, _ = schema_realistic_fixture(1001)
        for index, fact in enumerate(manifest["facts"]):
            fact["text"] = f"SAMPLE_{index}: " + "Evidence. " * 100
        projected, targets, contract = grounded(candidate, source_map, manifest)
        allowed_ids = set()
        for target in targets:
            for option in target["assessment_plan_candidates"]:
                ci, li, ui = [int(part.split("_")[-1]) - 1 for part in option["unit_path"].split(".")]
                block = next(b for b in projected["chapters"][ci]["lessons"][li]["units"][ui]["learning_blocks"] if b["id"] == option["teaching_block_id"])
                allowed_ids.update(block["primary_evidence_scope_ids"])
        evidence = json.loads(contract.descriptors)["canonical_evidence"]
        expected = set()
        facts = {f["fact_id"]: f for f in manifest["facts"]}
        for scope in source_map["source_evidence_scopes"]:
            if scope["id"] in allowed_ids:
                ids = scope["source_fact_ids"]
                expected.update(facts[ids[i]]["text"][:280] for i in {0, len(ids)//2, len(ids)-1})
        actual = {s for e in evidence.values() for s in e["representative_evidence"]}
        self.assertEqual(actual, expected)
        self.assertTrue(all(not e["sample_is_exhaustive"] for e in evidence.values()))
        self.assertEqual(len(evidence), len(allowed_ids))
        self.assertLessEqual(len(contract.descriptors), 16000)

    def test_invalid_manifest_and_scope_fail_before_provider_without_private_content(self):
        for mode in ("missing_fact", "duplicate_fact", "wrong_document", "blank_evidence", "unknown_scope", "concept_mismatch"):
            with self.subTest(mode=mode):
                candidate, source_map, manifest, _ = schema_realistic_fixture()
                projected, _, _ = materialize_assessment_source_refs(candidate, source_map)
                targets = targets_for(projected, source_map)
                block = projected["chapters"][2]["lessons"][0]["units"][0]["learning_blocks"][0]
                scope = next(s for s in source_map["source_evidence_scopes"] if s["id"] == block["primary_evidence_scope_ids"][0])
                fid = scope["source_fact_ids"][0]
                fact = next(f for f in manifest["facts"] if f["fact_id"] == fid)
                if mode == "missing_fact":
                    manifest["facts"].remove(fact)
                elif mode == "duplicate_fact":
                    manifest["facts"].append(copy.deepcopy(fact))
                elif mode == "wrong_document":
                    fact["document_id"] = "OTHER_PRIVATE_DOCUMENT"
                elif mode == "blank_evidence":
                    for f in manifest["facts"]:
                        f["text"] = ""
                elif mode == "unknown_scope":
                    block["primary_evidence_scope_ids"] = ["UNKNOWN_PRIVATE_SCOPE"]
                else:
                    block["concept_ids"] = []
                with self.assertRaises(WorkflowFailure) as caught:
                    build_assessment_selection_contract(targets, max_context_chars=16000, blueprint=projected, source_map=source_map, manifest=manifest)
                self.assertEqual(caught.exception.failure_stage, "architecture_repair_target_snapshot")
                self.assertNotIn("PRIVATE", str(caught.exception) + json.dumps(caught.exception.diagnostics))

    def test_bound_is_fail_closed_not_silent_evidence_truncation_and_no_match_unchanged(self):
        candidate, source_map, manifest, _ = schema_realistic_fixture()
        with self.assertRaises(WorkflowFailure) as caught:
            grounded(candidate, source_map, manifest, budget=200)
        self.assertEqual(caught.exception.diagnostics["guard_reason"], "SELECTION_CONTEXT_BUDGET_EXCEEDED")
        projected, targets, contract = grounded(candidate, source_map, manifest)
        baseline = copy.deepcopy(projected)
        delta = contract.decode_text(json.dumps({"choices": dict.fromkeys(contract.bindings, "NO_MATCH")}))
        with self.assertRaises(WorkflowFailure) as caught:
            apply_course_architecture_repair_patches(projected, targets, delta)
        self.assertEqual(caught.exception.internal_code, "ARCH_REPAIR_ASSESSMENT_NO_MATCH")
        self.assertEqual(projected, baseline)


class GroundedSelectionEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_schema_realistic_endpoint_sends_canonical_signal_and_finishes(self):
        for count, locale in ((415, "vi"), (1001, "en")):
            with self.subTest(count=count, locale=locale):
                candidate, source_map, manifest, nodes = schema_realistic_fixture(count, locale)
                _, _, contract = grounded(candidate, source_map, manifest)
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
                        # This assertion fails on the previous production path,
                        # whose prompt exposed only the intent label.
                        prompt = kwargs.get("prompt") or next((arg for arg in args if isinstance(arg, str) and "ASSESSMENT" in arg), "")
                        self.assertIn("SOURCE_ONLY_SIGNAL", prompt)
                        self.assertEqual(kwargs["request_timeout_ms"], 300000)
                        self.assertEqual(set(kwargs["response_schema"].properties), {"choices"})
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
                    result = await lesson_author_blueprint(request, pool=object())
                self.assertEqual(result["blueprint"]["source_fact_allocation"]["allocated_count"], count)
                self.assertTrue(result["blueprint"]["source_fact_allocation"]["complete"])
                self.assertEqual(result["workflow"]["repair_provider_calls"], 1)
                self.assertEqual(len(calls), 2)
                self.assertTrue(any(e.get("selection_evidence_chars", 0) > 0 for e in events))
                self.assertTrue(all(e["correlation_id"] == request.correlation_id for e in events))
                self.assertNotIn("SOURCE_ONLY_SIGNAL", json.dumps(events))
