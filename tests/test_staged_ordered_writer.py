"""New writer vs legacy reader, UAT mixed HTML + 47/54 claim; offline only."""
import asyncio
from copy import deepcopy
import json
import unittest
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError
from app import main
from app.lesson_author_provider_schema import staged_provider_response_model
from app.ordered_learning_content import bind_provider_semantic_versions, semantic_shape_diagnostics, ProviderSemanticVersionError
from app.workflows.contracts import WorkflowFailure
from tests.staged_schema_probe import capture_sdk_body, visit_schema
from tests.test_staged_instance_output import checkpoint_instance_fixture
from tests.test_checkpoint_component_quality_repair import instance_wire
from tests.test_chapter_checkpoint import checkpoint_result


def uat_fixture():
    request, unit, scope, _ = checkpoint_instance_fixture()
    facts = [f"synthetic-fact-{i}" for i in range(54)]
    unit.update(source_fact_ids=facts[:], supporting_evidence_fact_ids=facts[:])
    scope.update(source_fact_ids=facts[:], supporting_evidence_fact_ids=facts[:])
    for index, (component, plan) in enumerate(zip(unit["components"], scope["component_plan"])):
        owned, supporting = (facts[:], []) if index == 0 else ([], facts[:])
        plan.update(source_fact_ids=owned, supporting_evidence_fact_ids=supporting)
        component.update(source_fact_ids=owned[:], covered_source_fact_ids=owned[:], supporting_evidence_fact_ids=supporting[:])
    architecture = request.blueprint_architecture.model_dump()
    architecture["lessons"][0]["units"][0] = scope
    request = request.model_copy(update={"blueprint_architecture": type(request.blueprint_architecture).model_validate(architecture)})
    manifest = {"facts": [{"fact_id": f, "text": "Synthetic source instruction for an offline contract test."} for f in facts]}
    manifest["supporting_evidence_facts"] = deepcopy(manifest["facts"])
    return request, unit, scope, manifest


class StagedOrderedWriterTests(unittest.TestCase):
    def test_real_sdk_generation_and_repair_expose_only_sections(self):
        request, _, _, _ = uat_fixture()
        plans = [p.model_dump() for p in request.blueprint_architecture.lessons[0].units[0].component_plan]
        models = [main.build_staged_instance_response_model(plans)]
        models += [main.build_staged_lesson_content_response_model(kinds, payload_only=repair, coverage_repair=repair)
                   for kinds in (["html"], ["html", "problem", "la_faq"])
                   for repair in (False, True)]
        for model in models:
            projected, _ = staged_provider_response_model(model)
            wire = capture_sdk_body(projected)["generationConfig"]["responseSchema"]
            semantics = [node for path, node in visit_schema(wire) if path.endswith(".semantic_content")]
            self.assertTrue(semantics)
            for semantic in semantics:
                self.assertEqual(set(semantic["properties"]), {"sections"})
                self.assertEqual(semantic["required"], ["sections"])
                self.assertEqual(semantic["properties"]["sections"]["type"], "ARRAY")
                self.assertIn("items", semantic["properties"]["sections"])

    def test_server_stamps_format_in_typed_and_raw_fallback_without_dropping(self):
        _, unit, _, _ = uat_fixture()
        payload = deepcopy(unit["components"][0]["semantic_content"])
        del payload["version"]
        before = deepcopy(payload)
        typed = main.StagedOrderedSemanticOutput.model_validate(payload)
        self.assertEqual(typed.model_dump(exclude_unset=True)["version"], 2)
        response = {"components": {"c0": {"semantic_content": payload}}}
        bound = bind_provider_semantic_versions(response)
        self.assertEqual(bound["components"]["c0"]["semantic_content"], {"version": 2, **payload})
        self.assertEqual(payload, before)
        for field in ("heading", "paragraphs", "PRIVATE_KEY"):
            bad = {**payload, field: "PRIVATE_VALUE"}
            with self.assertRaises(ValidationError):
                main.StagedOrderedSemanticOutput.model_validate(bad)
            raw = bind_provider_semantic_versions({"components": [{"type": "html", "semantic_content": bad}]})
            self.assertEqual(raw["components"][0]["semantic_content"][field], "PRIVATE_VALUE")
            self.assertEqual(main.staged_payload_diagnostics(raw)[0]["code"], "HTML_SEMANTIC_INVALID")

    def test_wrong_provider_version_rejected_but_legacy_reader_preserved(self):
        legacy = {"heading": "Legacy", "paragraphs": ["Stored explanation."]}
        self.assertIsNone(main.semantic_learning_visible_text(legacy)[1])
        main.StagedSemanticContent.model_validate(legacy)
        for version in (1, 3, "2", True):
            bad = {"components": [{"semantic_content": {"version": version, "sections": []}}]}
            with self.assertRaises(ProviderSemanticVersionError):
                bind_provider_semantic_versions(bad)
        with self.assertRaises(ValidationError):
            main.StagedOrderedSemanticOutput.model_validate(legacy)

    def test_mixed_html_and_seven_missing_claims_repair_atomically(self):
        request, valid, scope, manifest = uat_fixture()
        broken = instance_wire(valid)
        bad = broken["components"]["c0"]
        bad["semantic_content"]["heading"] = "PRIVATE_CONFLICT"
        bad["semantic_content"]["paragraphs"] = ["PRIVATE_OLD_BODY"]
        bad["covered_source_fact_ids"] = valid["source_fact_ids"][:47]
        replacement = deepcopy(valid["components"][0]["semantic_content"])
        del replacement["version"]  # Actual new wire shape, server supplies it.
        delta = {"components": [{"component_index": 0, "semantic_content": replacement,
                                  "covered_source_fact_ids": valid["source_fact_ids"][:]}]}
        before = deepcopy(broken)
        provider = AsyncMock(side_effect=[(json.dumps(broken), main.AiUsage()), (json.dumps(delta), main.AiUsage())])
        with patch("app.main.generate_content", provider), self.assertLogs("app.main", "INFO") as logs:
            result = asyncio.run(checkpoint_result(request, manifest))
            final_request = request.model_copy(update={
                "checkpoint_action": "validate_chapter", "checkpoint_unit_index": None,
                "checkpoint_units": [main.ChapterCheckpointUnit(unit_index=0, unit=result["unit"])],
            })
            final = asyncio.run(checkpoint_result(final_request, manifest))
        self.assertEqual(final["status"], "ready")
        self.assertEqual(provider.await_count, 2)
        self.assertEqual(result["status"], "unit_ready")
        actual = result["unit"]
        self.assertEqual(actual["components"], valid["components"])
        self.assertEqual(actual["components"][0]["semantic_content"]["version"], 2)
        self.assertEqual(len(actual["components"][0]["covered_source_fact_ids"]), 54)
        self.assertEqual(broken, before)
        events = [json.loads(line[line.index("{"):]) for line in logs.output if "lesson_author_component_validation " in line]
        self.assertEqual(events[0]["repair_component_indices"], [0])
        self.assertEqual(events[0]["coverage_findings"][0]["missing_count"], 7)
        self.assertEqual(events[0]["findings"][0]["semantic_shape"]["legacy_fields_populated"], ["heading", "paragraphs"])
        self.assertEqual(events[-1]["status"], "PASS")
        safe = json.dumps(events)
        self.assertNotIn("PRIVATE_", safe)
        self.assertIn(request.correlation_id, safe)
        self.assertIn("staged_lesson_content_recovery", safe)

    def test_repeat_mixed_repair_fails_closed_at_existing_two_call_budget(self):
        request, unit, _, manifest = uat_fixture()
        wire = instance_wire(unit)
        wire["components"]["c0"]["semantic_content"]["heading"] = "PRIVATE_CONFLICT"
        delta = {"components": [{"component_index": 0, "semantic_content": wire["components"]["c0"]["semantic_content"]}]}
        provider = AsyncMock(side_effect=[(json.dumps(wire), main.AiUsage()), (json.dumps(delta), main.AiUsage())])
        with patch("app.main.generate_content", provider), self.assertRaises(WorkflowFailure) as failure:
            asyncio.run(checkpoint_result(request, manifest))
        self.assertEqual(provider.await_count, 2)
        self.assertEqual(failure.exception.internal_code, "CHAPTER_COMPONENT_REPAIR_EXHAUSTED")
        self.assertEqual(failure.exception.diagnostics["repair_failure_code"], "HTML_SEMANTIC_INVALID")
        self.assertNotIn("PRIVATE_", json.dumps(failure.exception.diagnostics))

    def test_coverage_ids_only_never_repair_missing_teaching(self):
        _, unit, _, _ = uat_fixture()
        unit["components"][0]["covered_source_fact_ids"] = unit["source_fact_ids"][:47]
        delta = {"components": [{"component_index": 0, "covered_source_fact_ids": unit["source_fact_ids"][:]}]}
        with self.assertRaisesRegex(main.LessonAuthorProposalValidationError, "COVERAGE_WITHOUT_CONTENT"):
            main.merge_staged_component_payload_delta(unit, delta, [0], coverage_targets=[0])

    def test_diagnostics_use_allowlisted_names_counts_and_no_raw_values(self):
        shape = semantic_shape_diagnostics({"version": "PRIVATE_VERSION", "heading": "PRIVATE_TEXT",
                                           "PRIVATE_KEY": "PRIVATE_TEXT", "sections": [{"blocks": [{}, {}]}]})
        self.assertEqual(shape["legacy_fields_populated"], ["heading"])
        self.assertEqual(shape["unknown_field_count"], 1)
        self.assertEqual((shape["section_count"], shape["block_count"]), (1, 2))
        self.assertNotIn("PRIVATE", json.dumps(shape))
