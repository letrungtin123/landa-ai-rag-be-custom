"""Offline regressions for UAT 58c9ce81: payload repair, not unit regeneration."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import json
import unittest
from unittest.mock import AsyncMock, patch

from google import genai
from google.genai import models, types

from app.main import (
    AiUsage, LessonAuthorProposalValidationError,
    build_staged_lesson_content_response_model, generate_staged_lesson_author_proposal,
    match_staged_unit_by_title, staged_unit_candidates, staged_unit_match_diagnostics,
    staged_component_repair_targets, staged_fact_membership_equal, staged_payload_diagnostics,
    validate_staged_unit_content,
)
from tests.test_component_instance_contract import PROFILE
from tests.test_lesson_prompt_policy import content_payload, request_and_unit


def fixture():
    request, unit = request_and_unit("problem", "vi")
    architecture = request.blueprint_architecture.model_dump()
    architecture["component_capabilities"] = PROFILE
    scope = architecture["lessons"][0]["units"][0]
    facts = [f"fact-{i}" for i in range(25)]
    scope["source_fact_ids"] = list(facts)
    scope["supporting_evidence_fact_ids"] = list(facts)
    unit["source_fact_ids"] = list(reversed(facts))
    unit["supporting_evidence_fact_ids"] = list(reversed(facts))
    unit["components"] += [content_payload(t, "vi") for t in ("la_diagram", "la_faq")]
    unit["components"][2].update(nodes=[{"label": "Inspect", "shape": "rounded"}, {"label": "Act", "shape": "rounded"}],
                                  edges=[{"source": 0, "target": 1}])
    scope["component_plan"] = []
    for index, component in enumerate(unit["components"]):
        owned = facts if index == 0 else []
        supporting = [] if index == 0 else facts
        plan = {
            "component_plan_id": "cp2_" + str(index + 1) * 32,
            "type": component["type"], "title": "Approved instance", "rationale": "Synthetic fixture",
            "source_fact_ids": list(owned), "supporting_evidence_fact_ids": list(supporting),
        }
        scope["component_plan"].append(plan)
        component.update(component_plan_id=plan["component_plan_id"], source_fact_ids=list(reversed(owned)),
                         covered_source_fact_ids=list(reversed(owned)), supporting_evidence_fact_ids=list(reversed(supporting)))
    request = request.model_copy(update={"blueprint_architecture": type(request.blueprint_architecture).model_validate(architecture)})
    scope["component_types"] = [c["type"] for c in unit["components"]]
    manifest = {"facts": [{"fact_id": f, "text": "Synthetic source-backed instruction."} for f in facts]}
    manifest["supporting_evidence_facts"] = deepcopy(manifest["facts"])
    return request, unit, scope, manifest


class StagedRecoveryIdentityTests(unittest.TestCase):
    def test_membership_is_exact_duplicate_free_and_order_independent(self):
        self.assertTrue(staged_fact_membership_equal(["f2", "f1"], ["f1", "f2"]))
        self.assertTrue(staged_fact_membership_equal([], []))
        for invalid in (["f1"], ["f1", "f2", "f2"], ["f1", "[f2]"], ["f1", " f2"],
                        ["f1", "unknown"], ["f1", {}], ["f1", None], None, "f1", ["", "f2"]):
            self.assertFalse(staged_fact_membership_equal(invalid, ["f1", "f2"]))

    def test_uat_order_only_repairs_diagram_and_preserves_every_other_byte(self):
        request, valid, scope, manifest = fixture()
        broken = deepcopy(valid)
        broken["components"][2]["nodes"] = []
        before = deepcopy(broken)
        self.assertEqual(staged_component_repair_targets(broken, scope), [2])
        delta = {"components": [{"component_index": 2, "nodes": valid["components"][2]["nodes"], "edges": valid["components"][2]["edges"]}]}
        provider = AsyncMock(side_effect=[(json.dumps(broken), AiUsage()), (json.dumps(delta), AiUsage())])
        with patch("app.main.generate_content", provider), self.assertLogs("app.main", level="INFO") as logs:
            result, _ = asyncio.run(generate_staged_lesson_author_proposal(request, "Synthetic", "", "", source_rows=[], source_coverage_manifest=manifest))
        actual = result["chapters"][0]["lessons"][0]["units"][0]
        self.assertEqual(actual["components"], valid["components"])
        # Final unit envelope is rebuilt from the approved server skeleton.
        self.assertEqual(actual["source_fact_ids"], scope["source_fact_ids"])
        self.assertEqual(broken, before)
        self.assertEqual(provider.await_count, 2)
        self.assertEqual(provider.await_args_list[1].kwargs["request_timeout_ms"], 180000)
        schema = provider.await_args_list[1].kwargs["response_schema"].model_json_schema()
        self.assertEqual(set(schema["properties"]), {"components"})
        serialized = json.dumps(schema)
        for forbidden in ("source_fact_ids", "supporting_evidence_fact_ids", "choices", "semantic_content"):
            self.assertNotIn(f'"{forbidden}"', serialized)
        self.assertIn('"repair_scope": "components"', "\n".join(logs.output))
        self.assertIn('"reason": "CARDINALITY"', "\n".join(logs.output))
        self.assertIn('"status": "PASS"', "\n".join(logs.output))

    def test_scope_or_identity_mutation_never_qualifies_for_payload_only_repair(self):
        _, unit, scope, _ = fixture()
        unit["components"][2]["nodes"] = []
        for path, value in (("source_fact_ids", ["unknown"]), ("supporting_evidence_fact_ids", ["unknown"])):
            bad = deepcopy(unit)
            bad[path] = value
            self.assertEqual(staged_component_repair_targets(bad, scope), [])
        for key, value in (("source_fact_ids", ["fact-0"]), ("supporting_evidence_fact_ids", ["fact-0"]),
                           ("covered_source_fact_ids", ["fact-0"]), ("component_plan_id", "outside"), ("type", "html")):
            bad = deepcopy(unit)
            bad["components"][2][key] = value
            self.assertEqual(staged_component_repair_targets(bad, scope), [])
        for key in ("source_fact_ids", "supporting_evidence_fact_ids", "covered_source_fact_ids"):
            bad = deepcopy(unit)
            bad["components"][0][key] = ["fact-0", "fact-0"]
            self.assertEqual(staged_component_repair_targets(bad, scope), [])
            self.assertIn("INVALID_FACT_ID_ARRAY", validate_staged_unit_content(bad, scope))

    def test_diagram_only_wire_bounds_and_title_enum_survive_actual_sdk(self):
        client = genai.Client(api_key="test-key")
        for delta in (False, True):
            model = build_staged_lesson_content_response_model(["la_diagram"], payload_only=delta, expected_unit_title="Mục đã duyệt")
            wire = models._GenerateContentConfig_to_mldev(client._api_client, types.GenerateContentConfig(response_schema=model))["responseSchema"]
            component = wire["properties"]["components"].items
            self.assertEqual((component.properties["nodes"].min_items, component.properties["nodes"].max_items), (2, 20))
            self.assertEqual((component.properties["edges"].min_items, component.properties["edges"].max_items), (1, 40))
            self.assertIn("nodes", component.required)
            if not delta:
                self.assertEqual(wire["properties"]["title"].enum, ["Mục đã duyệt"])
        # Mixed schemas still permit other component types' empty node arrays.
        mixed = build_staged_lesson_content_response_model(["html", "la_diagram"])
        wire = models._GenerateContentConfig_to_mldev(client._api_client, types.GenerateContentConfig(response_schema=mixed))["responseSchema"]
        self.assertNotIn("nodes", wire["properties"]["components"].items.required)

    def test_safe_diagram_diagnostics_have_paths_not_private_values(self):
        payload = {"type": "la_diagram", "nodes": [{"label": "PRIVATE", "shape": "PRIVATE_ENUM"}, {"label": "PRIVATE", "shape": "rounded"}],
                   "edges": [{"source": "PRIVATE", "target": -1}]}
        findings = staged_payload_diagnostics({"components": [payload]})
        safe = json.dumps(findings)
        self.assertNotIn("PRIVATE", safe)
        self.assertIn("nodes[0].shape", safe)
        self.assertIn("INVALID_ENUM", safe)
        self.assertIn("edges[0].source", safe)
        self.assertIn("NEGATIVE_INDEX", safe)
        payload["nodes"] = [{"label": None, "shape": None}] * 20
        self.assertLessEqual(len(staged_payload_diagnostics({"components": [payload]})[0]["shape_findings"]), 8)

    def test_unit_envelopes_share_exact_unambiguous_resolution(self):
        unit = {"title": "Approved", "components": []}
        for envelope in (unit, [unit], {"units": [unit]}):
            candidates = staged_unit_candidates(envelope)
            self.assertIs(match_staged_unit_by_title(candidates, "Approved"), unit)
        for envelope, reason in (({"units": []}, "NO_UNIT_CANDIDATES"), ({"units": "private"}, "NO_UNIT_CANDIDATES"),
                                 ({"units": [unit], "title": "Approved"}, "NO_UNIT_CANDIDATES"),
                                 ({"title": "PRIVATE"}, "UNIT_TITLE_NOT_MATCHED"), ([unit, deepcopy(unit)], "AMBIGUOUS_UNIT_TITLE")):
            candidates = staged_unit_candidates(envelope)
            self.assertIsNone(match_staged_unit_by_title(candidates, "Approved"))
            diagnostic = staged_unit_match_diagnostics(candidates, "Approved")
            self.assertEqual(diagnostic["reason"], reason)
            self.assertNotIn("PRIVATE", json.dumps(diagnostic))

    def test_full_unit_recovery_uses_same_envelope_handling_and_title_constraint(self):
        request, valid, _, manifest = fixture()
        # Force genuine unit/plan recovery, not payload-only repair.
        broken = {**deepcopy(valid), "components": []}
        provider = AsyncMock(side_effect=[(json.dumps(broken), AiUsage()), (json.dumps({"units": [valid]}), AiUsage())])
        with patch("app.main.generate_content", provider):
            result, _ = asyncio.run(generate_staged_lesson_author_proposal(request, "Synthetic", "", "", source_rows=[], source_coverage_manifest=manifest))
        self.assertEqual(result["chapters"][0]["lessons"][0]["units"][0]["components"], valid["components"])
        self.assertEqual(provider.await_count, 2)
        for call in provider.await_args_list:
            schema = call.kwargs["response_schema"].model_json_schema()
            title = schema["$defs"]["StagedUnitTitle"]
            self.assertEqual(title["enum"], [valid["title"]])

    def test_invalid_repair_fails_closed_within_same_budget_and_precise_reason(self):
        request, valid, _, manifest = fixture()
        cases = [
            ({**deepcopy(valid), "components": []}, {"title": "PRIVATE_WRONG_TITLE", "components": []}, "UNIT_TITLE_NOT_MATCHED"),
            ({**deepcopy(valid), "components": []}, {"units": [valid, deepcopy(valid)]}, "AMBIGUOUS_UNIT_TITLE"),
        ]
        broken = deepcopy(valid)
        broken["components"][2]["nodes"] = []
        cases.append((broken, {"components": [{"component_index": 2, "nodes": [], "edges": []}]}, "CARDINALITY"))
        cases.append((broken, {"components": [{"component_index": 2, "source_fact_ids": ["unknown"]}]}, "COMPONENT_REPAIR_PROTECTED_FIELD_EMITTED"))
        for initial, recovery, reason in cases:
            provider = AsyncMock(side_effect=[(json.dumps(initial), AiUsage()), (json.dumps(recovery), AiUsage())])
            with patch("app.main.generate_content", provider), self.assertLogs("app.main", level="INFO") as logs:
                with self.assertRaises(LessonAuthorProposalValidationError):
                    asyncio.run(generate_staged_lesson_author_proposal(request, "Synthetic", "", "", source_rows=[], source_coverage_manifest=manifest))
            self.assertEqual(provider.await_count, 2)
            self.assertIn(reason, "\n".join(logs.output))
            self.assertNotIn("PRIVATE_WRONG_TITLE", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
