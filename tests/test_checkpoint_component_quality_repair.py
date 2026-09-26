"""Synthetic offline chapter3 regressions; no DB, HTTP or real provider."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import json
import unittest
from unittest.mock import AsyncMock, patch

from app import main
from app.workflows.contracts import WorkflowFailure
from tests.test_staged_recovery_identity import fixture as original_fixture
from tests.test_chapter_checkpoint import fixture as endpoint_fixture


def fixture():
    request, unit, scope, _ = original_fixture()
    facts = [f"synthetic-fact-{i}" for i in range(71)]
    unit.update(source_fact_ids=facts[:], supporting_evidence_fact_ids=facts[:])
    scope.update(source_fact_ids=facts[:], supporting_evidence_fact_ids=facts[:])
    for i, (component, plan) in enumerate(zip(unit["components"], scope["component_plan"])):
        owned, supporting = (facts[:], []) if i == 0 else ([], facts[:])
        plan.update(source_fact_ids=owned, supporting_evidence_fact_ids=supporting)
        component.update(source_fact_ids=owned[:], covered_source_fact_ids=owned[:], supporting_evidence_fact_ids=supporting[:])
    component = unit["components"][2]
    component.pop("nodes", None)
    component.pop("edges", None)
    component.update(type="la_sortable", question_text="Put the documented procedure in order.", items=[
        {"text": "Inspect the documented conditions before starting."},
        {"text": "Perform the authorized task following the instructions."},
        {"text": "Record the verified result after completing the task."},
    ])
    scope["component_plan"][2]["type"] = "la_sortable"
    scope["component_types"][2] = "la_sortable"
    architecture = request.blueprint_architecture.model_dump()
    architecture["lessons"][0]["units"][0] = scope
    request = request.model_copy(update={"blueprint_architecture": type(request.blueprint_architecture).model_validate(architecture),
                                         "correlation_id": "55555555-5555-4555-8555-555555555555"})
    manifest = {"facts": [{"fact_id": f, "text": "Synthetic documented instruction only."} for f in facts]}
    manifest["supporting_evidence_facts"] = deepcopy(manifest["facts"])
    return request, unit, scope, manifest


def run(request, manifest):
    return asyncio.run(main.generate_staged_lesson_author_proposal(request, "Synthetic", "", "", source_rows=[],
                     source_coverage_manifest=manifest, checkpoint_unit_index=0))


def instance_wire(unit):
    """Provider-shaped fixture for the server-owned checkpoint output contract."""
    return {"components": {f"c{i}": {k: deepcopy(v) for k, v in component.items()
              if k in main.STAGED_COMPONENT_PAYLOAD_FIELDS[component["type"]] | {"title", "selection_rationale", "covered_source_fact_ids"}}
            for i, component in enumerate(unit["components"])}}


class CheckpointComponentQualityRepairTests(unittest.TestCase):
    def test_valid_dense_unit_bypasses_repair(self):
        request, unit, scope, manifest = fixture()
        self.assertIsNone(main.validate_staged_unit_content(unit, scope, strict_payload=True))
        provider = AsyncMock(return_value=(json.dumps(instance_wire(unit)), main.AiUsage()))
        with patch("app.main.generate_content", provider):
            result, _ = run(request, manifest)
        self.assertEqual(provider.await_count, 1)
        self.assertEqual(len(result["unit"]["source_fact_ids"]), 71)

    def test_sortable_schema_repair_preserves_good_components_and_provenance(self):
        request, valid, scope, manifest = fixture()
        broken = deepcopy(valid)
        broken["components"][2]["items"] = [{"text": "PRIVATE_ONLY_TWO_ITEMS"}] * 2
        baseline = deepcopy(broken)
        self.assertEqual(main.staged_component_repair_targets(broken, scope), [2])
        delta = {"components": [{"component_index": 2, "items": valid["components"][2]["items"]}]}
        provider = AsyncMock(side_effect=[(json.dumps(instance_wire(broken)), main.AiUsage()), (json.dumps(delta), main.AiUsage())])
        with patch("app.main.generate_content", provider), self.assertLogs("app.main", "INFO") as logs:
            result, _ = run(request, manifest)
        self.assertEqual(result["unit"]["components"], valid["components"])
        self.assertEqual(broken, baseline)
        self.assertEqual(provider.await_count, 2)
        wire = json.dumps(provider.await_args.kwargs["response_schema"].model_json_schema())
        for forbidden in ("semantic_content", "source_fact_ids", "choices", "question", "answer"):
            self.assertNotIn(f'"{forbidden}"', wire)
        safe = "\n".join(logs.output)
        self.assertIn('"reason": "CARDINALITY"', safe)
        self.assertNotIn("PRIVATE_ONLY_TWO_ITEMS", safe)
        self.assertIn('"validation_finding": {"code": "COMPONENT_PAYLOAD_SCHEMA_INVALID"', safe)

    def test_shape_valid_instructional_failures_repair_only_weak_component(self):
        for target, code in ((0, "HTML_INSUFFICIENT_DEPTH"), (2, "SORTABLE_STEP_FRAGMENT"), (3, "FAQ_CLARIFICATION_INCOMPLETE")):
            with self.subTest(code=code):
                request, valid, scope, manifest = fixture()
                broken = deepcopy(valid)
                if target == 0:
                    broken["components"][0]["semantic_content"] = {"paragraphs": ["PRIVATE_THIN"]}
                    fields = {"semantic_content": valid["components"][0]["semantic_content"]}
                elif target == 2:
                    broken["components"][2]["items"][0]["text"] = "inspect the authorized conditions before starting."
                    fields = {"items": valid["components"][2]["items"]}
                else:
                    broken["components"][3]["items"][0]["answer"] = "PRIVATE_THIN"
                    fields = {"items": valid["components"][3]["items"]}
                finding = main.validate_staged_unit_content(broken, scope, strict_payload=True)
                self.assertEqual(finding.code, code)
                self.assertEqual(main.staged_component_repair_targets(broken, scope), [target])
                delta = {"components": [{"component_index": target, **fields}]}
                provider = AsyncMock(side_effect=[(json.dumps(instance_wire(broken)), main.AiUsage()), (json.dumps(delta), main.AiUsage())])
                with patch("app.main.generate_content", provider), self.assertLogs("app.main", "INFO") as logs:
                    result, _ = run(request, manifest)
                self.assertEqual(result["unit"]["components"], valid["components"])
                self.assertEqual(provider.await_count, 2)
                self.assertNotIn("PRIVATE_THIN", "\n".join(logs.output))
                self.assertIn(code, "\n".join(logs.output))

    def test_sortable_shape_metadata_is_bounded_and_redacted(self):
        cases = [(None, "ARRAY_REQUIRED"), ([], "CARDINALITY"),
                 ([{"PRIVATE_KEY": "PRIVATE_VALUE"}] * 3, "FIELD_REQUIRED"),
                 ([{"text": "PRIVATE" * 100}] * 3, "TEXT_TOO_LONG"),
                 (["PRIVATE_VALUE"] * 3, "FIELD_TYPE_INVALID")]
        for items, reason in cases:
            result = main.staged_sortable_shape_diagnostics({"items": items})
            self.assertEqual(result[0]["reason"], reason)
            self.assertLessEqual(len(result), 8)
            self.assertNotIn("PRIVATE", json.dumps(result))

    def test_unknown_or_changed_scope_never_regenerates_whole_checkpoint_unit(self):
        for mutation in ("facts", "instance", "missing", "null_components"):
            request, valid, _, manifest = fixture()
            broken = instance_wire(valid)
            if mutation == "facts": broken["components"]["c0"]["source_fact_ids"] = ["PRIVATE_UNKNOWN_FACT"]
            elif mutation == "instance": broken["components"]["c0"]["component_plan_id"] = "PRIVATE_ID"
            elif mutation == "null_components": broken["components"] = None
            else: broken["components"] = []
            provider = AsyncMock(return_value=(json.dumps(broken), main.AiUsage()))
            with patch("app.main.generate_content", provider), self.assertLogs("app.main", "INFO") as logs:
                with self.assertRaises(WorkflowFailure) as failure:
                    run(request, manifest)
            self.assertEqual(failure.exception.internal_code, "CHAPTER_UNIT_CONTRACT_REJECTED")
            self.assertEqual(provider.await_count, 1)
            self.assertNotIn("PRIVATE", "\n".join(logs.output))

    def test_bad_delta_cannot_inject_scope_or_touch_other_components_and_budget_stays_two(self):
        for change in ({"component_index": 2, "source_fact_ids": ["PRIVATE_NEW_FACT"]},
                       {"component_index": 0, "semantic_content": {"paragraphs": ["PRIVATE_REPLACEMENT"]}},
                       {"component_index": 2, "items": []}):
            request, valid, _, manifest = fixture()
            broken = deepcopy(valid)
            broken["components"][2]["items"] = []
            baseline = deepcopy(broken)
            provider = AsyncMock(side_effect=[(json.dumps(instance_wire(broken)), main.AiUsage()),
                                             (json.dumps({"components": [change]}), main.AiUsage())])
            with patch("app.main.generate_content", provider), patch("app.main.build_source_locked_html_unit") as fallback:
                with self.assertRaises(WorkflowFailure) as failure:
                    run(request, manifest)
            self.assertEqual(failure.exception.internal_code, "CHAPTER_COMPONENT_REPAIR_EXHAUSTED")
            self.assertEqual(failure.exception.failure_stage, "chapter_component_repair_revalidation")
            self.assertIn(failure.exception.diagnostics["repair_failure_code"], {
                "COMPONENT_REPAIR_PROTECTED_FIELD_EMITTED", "COMPONENT_REPAIR_TARGET_INVALID", "COMPONENT_PAYLOAD_SCHEMA_INVALID"})
            self.assertEqual(provider.await_count, 2)
            fallback.assert_not_called()
            self.assertEqual(broken, baseline)

    def test_instructional_diagnostics_handle_invalid_structure_and_remain_bounded(self):
        for value in (None, {}, {"components": None}, {"components": 5}):
            self.assertEqual(main.staged_instructional_diagnostics(value), [])
        result = main.staged_instructional_diagnostics({"components": [{"type": "la_sortable", "items": []}] * 40})
        self.assertEqual(len(result), 16)

    def test_endpoint_preserves_typed_terminal_stage_and_metadata_without_private_payload(self):
        request, _, _ = endpoint_fixture()
        failure = WorkflowFailure("LESSON_VALIDATION_FAILED", "Safe failure", internal_code="CHAPTER_COMPONENT_REPAIR_EXHAUSTED",
                                  failure_stage="chapter_component_repair_revalidation",
                                  diagnostics={"repair_scope": "components", "repair_component_indices": [2],
                                               "validation_finding": {"code": "SORTABLE_STEP_FRAGMENT", "path": "components[2].items", "repairable": True}})
        with patch("app.main.lesson_author_proposal", AsyncMock(side_effect=failure)), self.assertLogs("app.main", "INFO") as logs:
            with self.assertRaises(main.HTTPException) as result:
                asyncio.run(main.lesson_author_chapter_checkpoint(request, pool=None))
        self.assertEqual(result.exception.status_code, 422)
        self.assertEqual(result.exception.detail["internal_failure_code"], failure.internal_code)
        self.assertIn("SORTABLE_STEP_FRAGMENT", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
