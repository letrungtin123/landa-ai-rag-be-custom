from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, patch

from google import genai
from google.genai import models, types
from pydantic import create_model, ValidationError

from app.main import (
    AiUsage,
    RagLessonAuthorRequest,
    build_staged_lesson_content_response_model,
    generate_staged_lesson_author_proposal,
    semantic_learning_visible_text,
    staged_response_schema_diagnostics,
    validate_staged_unit_content,
    staged_component_payload_code,
    staged_component_contract_prompt,
    staged_component_repair_targets,
    merge_staged_component_repair,
    merge_staged_component_payload_delta,
    staged_evidence_scope_diagnostics,
    staged_payload_diagnostics,
    ARCHITECT_COMPONENT_OPPORTUNITY_POLICY,
    LessonAuthorProposalValidationError,
)
from app.workflows.contracts import WorkflowFailure


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "v5_chapter_content_quality_fixture.json"


def staged_request() -> RagLessonAuthorRequest:
    return RagLessonAuthorRequest(
        tenant_id="11111111-1111-4111-8111-111111111111",
        kb_id="22222222-2222-4222-8222-222222222222",
        conversation_id="33333333-3333-4333-8333-333333333333",
        target="lesson_author",
        model="gemini-3.5-flash",
        max_output_tokens=8192,
        embedding_model="gemini-embedding-001",
        system_prompt="test",
        user_message="Soạn chi tiết chương đã chọn.",
        output_schema_hint="schema",
        api_key="test-key",
    )


class StagedLessonProviderBoundaryTests(unittest.TestCase):
    def test_semantic_html_wire_fields_survive_sdk_in_generation_and_delta(self):
        client = genai.Client(api_key="test-key")
        for repair in (False, True):
            for selected in (["html"], ["html", "problem", "la_faq"]):
                model = build_staged_lesson_content_response_model(selected, payload_only=repair)
                wire = models._GenerateContentConfig_to_mldev(client._api_client, types.GenerateContentConfig(response_schema=model))["responseSchema"]
                component = wire["properties"]["components"].items
                semantic = component.properties["semantic_content"]
                self.assertEqual(set(semantic.properties), {"heading", "paragraphs", "bullet_points", "ordered_steps", "warnings", "comparison_rows"})
                self.assertIn("paragraphs", semantic.required)
                self.assertEqual(semantic.properties["paragraphs"].min_items, 1)
                for key in ("paragraphs", "bullet_points", "ordered_steps", "warnings"):
                    self.assertEqual(semantic.properties[key].items.type, types.Type.STRING)
                self.assertEqual(set(semantic.properties["comparison_rows"].items.required), {"label", "value"})
                if selected == ["html"]:
                    self.assertIn("semantic_content", component.required)
                self.assertTrue(staged_response_schema_diagnostics(model)["schema_valid"])
        delta = build_staged_lesson_content_response_model(["html"], payload_only=True)
        for semantic in ({}, None, {"paragraphs": []}, {"paragraphs": [{}]}):
            with self.assertRaises(ValidationError):
                delta.model_validate({"components": [{"component_index": 0, "title": None, "selection_rationale": None, "html": None, "semantic_content": semantic}]})

    def test_uat_empty_semantic_html_delta_repaired_and_failure_reason_retained(self):
        from tests.test_component_instance_contract import PROFILE
        from tests.test_lesson_prompt_policy import request_and_unit
        request, generated = request_and_unit("problem", "vi")
        architecture = request.blueprint_architecture.model_dump()
        architecture["component_capabilities"] = PROFILE
        expected = architecture["lessons"][0]["units"][0]
        for i, (plan, component) in enumerate(zip(expected["component_plan"], generated["components"])):
            plan["component_plan_id"] = component["component_plan_id"] = "cp2_" + str(i + 1) * 32
        request = request.model_copy(update={"blueprint_architecture": type(request.blueprint_architecture).model_validate(architecture)})
        broken = deepcopy(generated)
        broken["components"][0]["semantic_content"] = {}
        manifest = {"facts": [{"fact_id": f, "text": "Source-supported synthetic instruction."} for f in expected["source_fact_ids"]]}
        for repaired in (True, False):
            delta = {"components": [{"component_index": 0, "semantic_content": generated["components"][0]["semantic_content"] if repaired else {}}]}
            provider = AsyncMock(side_effect=[(json.dumps(broken), AiUsage()), (json.dumps(delta), AiUsage())])
            with patch("app.main.generate_content", provider), self.assertLogs("app.main", level="INFO") as logs:
                if repaired:
                    result, _ = asyncio.run(generate_staged_lesson_author_proposal(request, "Synthetic", "", "", source_rows=[], source_coverage_manifest=manifest))
                    self.assertEqual(result["chapters"][0]["lessons"][0]["units"][0]["components"], generated["components"])
                else:
                    with self.assertRaises(LessonAuthorProposalValidationError):
                        asyncio.run(generate_staged_lesson_author_proposal(request, "Synthetic", "", "", source_rows=[], source_coverage_manifest=manifest))
            self.assertEqual(provider.await_count, 2)
            if not repaired:
                event = next(line for line in logs.output if '"stage": "staged_repair_revalidation"' in line)
                self.assertIn("semantic_shape_reason", event)
                self.assertIn("non-empty object", event)

    @staticmethod
    def payloads() -> list[dict]:
        return [
            {"type": "html", "semantic_content": {"heading": "Explain", "paragraphs": ["Check the initial conditions before performing the approved task. Record the result of the inspection and act only when the conditions are satisfied. After completing the task, confirm the result against the documented requirement. These three stages have distinct purposes: establish readiness, perform the action, and verify completion."]}},
            {"type": "problem", "problem_type": "multiple_choice", "question": "Which action is taught?", "choices": [{"text": "Inspect", "correct": True}, {"text": "Skip", "correct": False}]},
            {"type": "la_faq", "items": [{"question": "When?", "answer": "Before the task."}, {"question": "Why?", "answer": "To check conditions."}]},
            {"type": "la_sortable", "question_text": "Order the taught steps.", "items": [{"text": x} for x in ["Inspect", "Act", "Confirm"]]},
            {"type": "la_crossword", "words": [{"answer": a, "clue": c} for a, c in [("CHECK", "Inspect conditions"), ("ACT", "Perform a step"), ("CONFIRM", "Verify completion")]]},
            {"type": "la_diagram", "nodes": [{"label": x, "shape": "rounded"} for x in ["Check", "Act"]], "edges": [{"source": 0, "target": 1}]},
        ]

    def test_six_typed_contracts_valid_and_invalid_payloads(self):
        for payload in self.payloads():
            with self.subTest(component=payload["type"]):
                self.assertIsNone(staged_component_payload_code(payload))
                empty = {"type": payload["type"]}
                self.assertIsNotNone(staged_component_payload_code(empty))
                unsafe = deepcopy(payload)
                if payload["type"] == "html":
                    unsafe["semantic_content"] = {"paragraphs": ["<script>alert(1)</script>"]}
                elif payload["type"] == "problem":
                    unsafe["choices"][0]["correct"] = False
                elif payload["type"] == "la_faq":
                    unsafe["items"][1]["question"] = unsafe["items"][0]["question"]
                elif payload["type"] == "la_sortable":
                    unsafe["items"][1] = unsafe["items"][0]
                elif payload["type"] == "la_crossword":
                    unsafe["words"][1]["answer"] = "Chéck"
                else:
                    unsafe["edges"][0]["target"] = 100
                self.assertIsNotNone(staged_component_payload_code(unsafe))

    def test_problem_subtypes_and_answer_cardinality(self):
        base = self.payloads()[1]
        for subtype in ("multiple_choice", "multiple_select"):
            candidate = {**deepcopy(base), "problem_type": subtype}
            self.assertIsNone(staged_component_payload_code(candidate))
            candidate["choices"][1]["correct"] = True
            self.assertEqual(staged_component_payload_code(candidate), "PROBLEM_CORRECT_ANSWER_INVALID" if subtype == "multiple_choice" else None)
        for subtype, answer in (("short_text", "Check"), ("numerical", "0"), ("dropdown", "Check")):
            c = {"type": "problem", "problem_type": subtype, "question": "Check?", "answer": answer, "options": ["Check", "Skip"]}
            self.assertIsNone(staged_component_payload_code(c))
            c["answer"] = ""
            self.assertIsNotNone(staged_component_payload_code(c))
        for invalid in (None, [], [{"label": "Unknown shape", "correct": True}], [{"text": "A", "correct": "true"}, {"text": "B", "correct": False}]):
            self.assertIsNotNone(staged_component_payload_code({**base, "choices": invalid}))

    def test_sdk_wire_schema_contains_typed_choices_and_shared_item_fields(self):
        client = genai.Client(api_key="test-key")
        response_model = build_staged_lesson_content_response_model(["problem", "la_faq", "la_sortable"])
        wire = models._GenerateContentConfig_to_mldev(client._api_client, types.GenerateContentConfig(response_schema=response_model))["responseSchema"]
        component = wire["properties"]["components"].items
        self.assertEqual(set(component.properties["type"].enum), {"problem", "la_faq", "la_sortable"})
        self.assertEqual(set(component.properties["choices"].items.required), {"text", "correct"})
        self.assertEqual(set(component.properties["items"].items.properties), {"question", "answer", "text"})
        self.assertNotIn("nodes", component.properties)

    def test_scoped_repair_preserves_good_component_and_provenance(self):
        html, problem = deepcopy(self.payloads()[:2])
        for index, c in enumerate([html, problem], 1):
            c.update(component_plan_id="cp2_" + str(index) * 32, source_fact_ids=["f1"], covered_source_fact_ids=["f1"], supporting_evidence_fact_ids=[])
        baseline = {"title": "Unit", "source_fact_ids": ["f1"], "supporting_evidence_fact_ids": [], "components": [html, {**problem, "choices": []}]}
        before = deepcopy(baseline)
        expected = {"source_fact_ids": ["f1"], "supporting_evidence_fact_ids": [], "component_plan": [html, problem]}
        targets = staged_component_repair_targets(baseline, expected)
        self.assertEqual(targets, [1])
        repair = {**deepcopy(baseline), "components": [problem]}
        result = merge_staged_component_repair(baseline, repair, targets)
        self.assertEqual(result["components"][0], html)
        self.assertEqual(baseline, before)
        for key in ("source_fact_ids", "covered_source_fact_ids", "supporting_evidence_fact_ids"):
            bad = deepcopy(repair)
            bad["components"][0][key] = ["outside-fact"]
            with self.assertRaisesRegex(LessonAuthorProposalValidationError, "AUTHORITY_CHANGED"):
                merge_staged_component_repair(baseline, bad, targets)
        with self.assertRaisesRegex(LessonAuthorProposalValidationError, "COUNT_INVALID"):
            merge_staged_component_repair(baseline, {**repair, "components": [html, problem]}, targets)
        for forbidden in ("source_locked_fallback", "metadata", "primary_evidence_scope_ids", "learning_block_ids"):
            bad = deepcopy(repair)
            bad["components"][0][forbidden] = True
            with self.assertRaisesRegex(LessonAuthorProposalValidationError, "FIELD_NOT_ALLOWED"):
                merge_staged_component_repair(baseline, bad, targets)

    def test_scoped_repair_runs_inside_existing_single_recovery_budget(self):
        from tests.test_component_instance_contract import PROFILE
        from tests.test_lesson_prompt_policy import request_and_unit
        request, unit = request_and_unit("problem", "en")
        architecture = request.blueprint_architecture.model_dump()
        architecture["component_capabilities"] = PROFILE
        plans = architecture["lessons"][0]["units"][0]["component_plan"]
        for n, (plan, c) in enumerate(zip(plans, unit["components"]), 1):
            plan["component_plan_id"] = c["component_plan_id"] = "cp2_" + str(n) * 32
        request = request.model_copy(update={"blueprint_architecture": type(request.blueprint_architecture).model_validate(architecture)})
        broken = deepcopy(unit)
        broken["components"][1].update(problem_type="multiple_choice", choices=[])
        correct = {**broken["components"][1], **self.payloads()[1]}
        repair = {"components": [{"component_index": 1, **{k: v for k, v in correct.items()
                   if k not in {"type", "component_plan_id", "source_fact_ids", "covered_source_fact_ids", "supporting_evidence_fact_ids"}}}]}
        manifest = {"facts": [{"fact_id": "fact-1", "source_page": 1, "text": "Check the conditions before acting."}]}
        provider = AsyncMock(side_effect=[(json.dumps(broken), AiUsage()), (json.dumps(repair), AiUsage())])
        with patch("app.main.generate_content", new=provider):
            result, _ = asyncio.run(generate_staged_lesson_author_proposal(request, "Synthetic", "", "", source_rows=[], source_coverage_manifest=manifest))
        self.assertEqual(provider.await_count, 2)
        produced = result["chapters"][0]["lessons"][0]["units"][0]["components"]
        self.assertEqual(produced[0], unit["components"][0])
        self.assertEqual(produced[1]["choices"], correct["choices"])
        self.assertIn("SCOPED COMPONENT REPAIR", provider.call_args_list[1].args[2])
        model = provider.call_args_list[1].kwargs["response_schema"].model_json_schema()
        self.assertEqual(set(model["properties"]), {"components"})
        failed_delta = deepcopy(repair)
        failed_delta["components"][0]["choices"] = []
        failed = AsyncMock(side_effect=[(json.dumps(broken), AiUsage()), (json.dumps(failed_delta), AiUsage())])
        with patch("app.main.generate_content", new=failed):
            with self.assertRaises(LessonAuthorProposalValidationError):
                asyncio.run(generate_staged_lesson_author_proposal(request, "Synthetic", "", "", source_rows=[], source_coverage_manifest=manifest))
        self.assertEqual(failed.await_count, 2)

    def test_delta_wire_contract_excludes_all_provider_provenance(self):
        for types_ in (["html"], ["problem"], ["la_faq", "la_sortable"], ["la_crossword", "la_diagram"]):
            model = build_staged_lesson_content_response_model(types_, payload_only=True)
            client = genai.Client(api_key="test-key")
            wire = models._GenerateContentConfig_to_mldev(client._api_client, types.GenerateContentConfig(response_schema=model))["responseSchema"]
            self.assertEqual(set(wire["properties"]), {"components"})
            fields = wire["properties"]["components"].items.properties
            self.assertIn("component_index", fields)
            for key in ("type", "component_plan_id", "source_fact_ids", "covered_source_fact_ids", "supporting_evidence_fact_ids"):
                self.assertNotIn(key, fields)
            self.assertTrue(staged_response_schema_diagnostics(model)["schema_valid"])

    def test_three_batch_chapter_repairs_last_html_without_reemitting_evidence(self):
        from tests.test_component_instance_contract import PROFILE
        from tests.test_lesson_prompt_policy import request_and_unit
        request, template = request_and_unit("problem", "vi")
        architecture = request.blueprint_architecture.model_dump()
        architecture["component_capabilities"] = PROFILE
        template_scope = deepcopy(architecture["lessons"][0]["units"][0])
        scopes, generated, manifest = [], [], {"facts": []}
        for index, count in enumerate((25, 10, 74)):
            scope, unit = deepcopy(template_scope), deepcopy(template)
            facts = [f"p{index + 1}-f{n + 1}" for n in range(count)]
            scope["title"] = unit["title"] = f"Synthetic unit {index + 1}"
            scope["source_fact_ids"] = unit["source_fact_ids"] = facts
            scope["supporting_evidence_fact_ids"] = unit["supporting_evidence_fact_ids"] = facts
            for component_index, (plan, component) in enumerate(zip(scope["component_plan"], unit["components"])):
                plan["component_plan_id"] = component["component_plan_id"] = "cp2_" + str(index * 2 + component_index + 1) * 32
                owned = facts if component_index == 0 else []
                support = [] if component_index == 0 else facts
                plan["source_fact_ids"] = component["source_fact_ids"] = component["covered_source_fact_ids"] = owned
                plan["supporting_evidence_fact_ids"] = component["supporting_evidence_fact_ids"] = support
            manifest["facts"].extend({"fact_id": f, "source_page": index + 1, "text": "Check conditions before acting."} for f in facts)
            scopes.append(scope)
            generated.append(unit)
        architecture["lessons"][0]["units"] = scopes
        manifest["supporting_evidence_facts"] = deepcopy(manifest["facts"])
        request = request.model_copy(update={"blueprint_architecture": type(request.blueprint_architecture).model_validate(architecture)})
        broken = deepcopy(generated[2])
        broken["components"][0]["semantic_content"] = {"paragraphs": [{"text": "Invalid string-array item"}]}
        delta = {"components": [{"component_index": 0, "semantic_content": generated[2]["components"][0]["semantic_content"]}]}
        provider = AsyncMock(side_effect=[(json.dumps(value), AiUsage()) for value in [generated[0], generated[1], broken, delta]])
        with patch("app.main.generate_content", provider):
            result, _ = asyncio.run(generate_staged_lesson_author_proposal(request, "Synthetic", "", "", source_rows=[], source_coverage_manifest=manifest))
        self.assertEqual(provider.await_count, 4)
        units = result["chapters"][0]["lessons"][0]["units"]
        for actual, original in zip(units, generated):
            self.assertEqual(actual["components"], original["components"])
            self.assertEqual(actual["source_fact_ids"], original["source_fact_ids"])
            self.assertEqual(actual["supporting_evidence_fact_ids"], original["supporting_evidence_fact_ids"])

    def test_payload_delta_preserves_large_server_owned_scope_and_good_components(self):
        for count in (74, 126, 371, 1001):
            facts = [f"f{i}" for i in range(count)]
            html, problem = deepcopy(self.payloads()[:2])
            html.update(source_fact_ids=facts, covered_source_fact_ids=facts, supporting_evidence_fact_ids=list(reversed(facts)), component_plan_id="cp2_" + "a" * 32)
            baseline = {"title": "Unit", "source_fact_ids": facts, "supporting_evidence_fact_ids": list(reversed(facts)),
                        "learning_blocks": [{"id": "server-owned"}], "components": [{**html, "semantic_content": {"paragraphs": [{}]}}, problem]}
            before = deepcopy(baseline)
            delta = {"components": [{"component_index": 0, "semantic_content": html["semantic_content"]}]}
            result = merge_staged_component_payload_delta(baseline, delta, [0])
            self.assertEqual(baseline, before)
            self.assertEqual(result["components"][1], problem)
            self.assertEqual(result["components"][0], html)
            for field in ("title", "source_fact_ids", "supporting_evidence_fact_ids", "learning_blocks"):
                self.assertEqual(result[field], baseline[field])

    def test_delta_rejects_provenance_injection_addresses_and_forbidden_fields(self):
        baseline = {"title": "Unit", "source_fact_ids": ["f1"], "supporting_evidence_fact_ids": ["f2"], "components": deepcopy(self.payloads()[:2])}
        good = {"component_index": 0, "semantic_content": self.payloads()[0]["semantic_content"]}
        for forbidden in ("source_fact_ids", "covered_source_fact_ids", "supporting_evidence_fact_ids", "component_plan_id", "type", "primary_evidence_scope_ids", "metadata"):
            with self.subTest(field=forbidden), self.assertRaises(LessonAuthorProposalValidationError):
                merge_staged_component_payload_delta(baseline, {"components": [{**good, forbidden: ["outside"]}]}, [0])
        for index in (-1, 1, 999, "0", True, None):
            with self.subTest(index=index), self.assertRaisesRegex(LessonAuthorProposalValidationError, "TARGET_INVALID"):
                merge_staged_component_payload_delta(baseline, {"components": [{**good, "component_index": index}]}, [0])
        for outer in ("source_fact_ids", "supporting_evidence_fact_ids", "title", "metadata"):
            with self.assertRaisesRegex(LessonAuthorProposalValidationError, "ENVELOPE_FORBIDDEN"):
                merge_staged_component_payload_delta(baseline, {"components": [good], outer: []}, [0])
        before = deepcopy(baseline)
        with self.assertRaises(LessonAuthorProposalValidationError):
            merge_staged_component_payload_delta(baseline, {"components": [good, {"component_index": 1, "choices": []}]}, [0, 1])
        self.assertEqual(baseline, before)
        with self.assertRaisesRegex(LessonAuthorProposalValidationError, "TARGET_INVALID"):
            merge_staged_component_payload_delta(baseline, {"components": [good, good]}, [0, 1])

    def test_scope_and_semantic_diagnostics_are_specific_without_private_values(self):
        expected = {"source_fact_ids": ["a", "b"], "supporting_evidence_fact_ids": ["secret_fact_1"]}
        actual = {"source_fact_ids": ["b", "a"], "supporting_evidence_fact_ids": ["PRIVATE_SOURCE_MARKER"]}
        findings = staged_evidence_scope_diagnostics(actual, expected)
        self.assertEqual(findings[0]["reason"], "ORDER_ONLY")
        self.assertEqual(findings[1]["path"], "unit.supporting_evidence_fact_ids")
        self.assertEqual(findings[1]["missing_count"], 1)
        self.assertEqual(findings[1]["unexpected_count"], 1)
        semantic = staged_payload_diagnostics({"components": [{"type": "html", "semantic_content": {"paragraphs": [{"text": "PRIVATE_SOURCE_MARKER"}]}}]})
        self.assertEqual(semantic[0]["code"], "HTML_SEMANTIC_INVALID")
        self.assertIn("paragraphs", semantic[0]["semantic_shape_reason"])
        self.assertIn("non-string item", semantic[0]["semantic_shape_reason"])
        self.assertNotIn("PRIVATE_SOURCE_MARKER", json.dumps(findings + semantic))
        self.assertNotIn("secret_fact_1", json.dumps(findings + semantic))

    def test_selected_prompts_and_diagnostics_do_not_expose_content(self):
        prompt = staged_component_contract_prompt(["problem"])
        self.assertIn('"text"', prompt)
        self.assertNotIn("la_faq:", prompt)
        self.assertIn("final learning activity", ARCHITECT_COMPONENT_OPPORTUNITY_POLICY)
        self.assertIn("insufficient, omit", ARCHITECT_COMPONENT_OPPORTUNITY_POLICY)
        self.assertIn("Never duplicate primary ownership", ARCHITECT_COMPONENT_OPPORTUNITY_POLICY)
        diagnostic = staged_payload_diagnostics({"components": [{"type": "problem", "question": "PRIVATE_SOURCE_MARKER", "problem_type": "multiple_choice", "choices": []}]})
        self.assertNotIn("PRIVATE_SOURCE_MARKER", json.dumps(diagnostic))
        self.assertEqual(diagnostic[0]["choice_count"], 0)

    def test_synthetic_fixture_preserves_the_planned_scale_and_supporting_shape(self) -> None:
        fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

        self.assertEqual(fixture["chapter"]["lesson_count"], 9)
        self.assertEqual(fixture["chapter"]["unit_count"], 9)
        self.assertEqual(fixture["chapter"]["dense_unit"]["fact_count"], 90)
        self.assertEqual(fixture["evidence"]["scope_count"], 17)
        self.assertEqual(fixture["evidence"]["canonical_fact_count"], 371)
        self.assertEqual(fixture["scale"]["canonical_fact_count"], 1001)
        self.assertEqual(fixture["chapter"]["supporting_only_assessment"]["owned_fact_count"], 0)
        self.assertTrue(fixture["chapter"]["supporting_only_assessment"]["supporting_scope_ids"])

    def test_installed_sdk_reproduces_schema_no_text_defect_without_network(self) -> None:
        with self.assertRaises(TypeError):
            types.GenerateContentResponse._from_response(
                response={"candidates": []},
                kwargs={"config": {"response_schema": types.Schema(type=types.Type.OBJECT)}},
            )

    def test_typed_stage_two_model_serializes_and_handles_empty_candidates_without_type_error(self) -> None:
        response_model = build_staged_lesson_content_response_model(["html", "problem"])
        client = genai.Client(api_key="test-key")
        payload = models._GenerateContentConfig_to_mldev(
            client._api_client,
            types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_model,
            ),
        )
        self.assertIn("responseSchema", payload)

        response = types.GenerateContentResponse._from_response(
            response={"candidates": []},
            kwargs={"config": {"response_schema": response_model}},
        )
        self.assertIsNone(response.parsed)
        self.assertIsNone(response.text)

    def test_all_enabled_stage_two_arrays_keep_items_after_installed_sdk_serialization(self) -> None:
        response_model = build_staged_lesson_content_response_model([
            "html", "problem", "la_faq", "la_sortable", "la_crossword", "la_diagram",
        ])
        client = genai.Client(api_key="test-key")
        payload = models._GenerateContentConfig_to_mldev(
            client._api_client,
            types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_model,
            ),
        )
        root = payload["responseSchema"]
        component = root["properties"]["components"].items
        for field in (
            "choices", "options", "items", "ordered_items", "steps", "words", "nodes", "edges",
        ):
            array_schema = component.properties[field]
            self.assertEqual(array_schema.type, types.Type.ARRAY, field)
            self.assertIsNotNone(array_schema.items, field)
            self.assertIsNot(array_schema.nullable, True, field)

        diagnostics = staged_response_schema_diagnostics(response_model)
        self.assertTrue(diagnostics["schema_valid"])
        self.assertEqual(diagnostics["array_missing_items_count"], 0)
        self.assertEqual(diagnostics["nullable_array_count"], 0)

    def test_each_enabled_component_schema_is_individually_sdk_serializable(self) -> None:
        client = genai.Client(api_key="test-key")
        expected_array_fields = {
            "html": set(),
            "problem": {"choices", "options"},
            "la_faq": {"items"},
            "la_sortable": {"items", "ordered_items", "steps"},
            "la_crossword": {"words"},
            "la_diagram": {"nodes", "edges"},
        }

        for component_type, array_fields in expected_array_fields.items():
            with self.subTest(component_type=component_type):
                response_model = build_staged_lesson_content_response_model([component_type])
                payload = models._GenerateContentConfig_to_mldev(
                    client._api_client,
                    types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=response_model,
                    ),
                )
                component = payload["responseSchema"]["properties"]["components"].items
                for field in array_fields:
                    array_schema = component.properties[field]
                    self.assertEqual(array_schema.type, types.Type.ARRAY, field)
                    self.assertIsNotNone(array_schema.items, field)
                    self.assertIsNot(array_schema.nullable, True, field)
                diagnostics = staged_response_schema_diagnostics(response_model)
                self.assertTrue(diagnostics["schema_valid"])
                self.assertEqual(diagnostics["array_missing_items_count"], 0)
                self.assertEqual(diagnostics["nullable_array_count"], 0)

    def test_stage_two_schema_preflight_rejects_nullable_array_before_provider(self) -> None:
        malformed = create_model("MalformedStageTwoSchema", values=(list[str] | None, ...))
        diagnostics = staged_response_schema_diagnostics(malformed)
        self.assertFalse(diagnostics["schema_valid"])
        self.assertEqual(diagnostics["nullable_array_count"], 1)
        self.assertEqual(
            diagnostics["safe_invalid_schema_paths"],
            ["response_schema.values"],
        )

    def test_typed_stage_two_model_keeps_only_selected_component_fields(self) -> None:
        response_model = build_staged_lesson_content_response_model(["html"])
        component_model = response_model.model_fields["components"].annotation.__args__[0]
        fields = set(component_model.model_fields)

        self.assertIn("semantic_content", fields)
        self.assertIn("html", fields)
        self.assertNotIn("question", fields)
        self.assertNotIn("nodes", fields)

    def test_stage_two_sdk_deserialization_failure_is_typed_and_not_retried(self) -> None:
        skeleton = json.dumps({
            "chapters": [{
                "title": "Synthetic chapter",
                "lessons": [{
                    "title": "Synthetic lesson",
                    "units": [{
                        "title": "Synthetic unit",
                        "source_fact_ids": ["fixture-fact-1"],
                        "component_plan": [{
                            "type": "html",
                            "source_fact_ids": ["fixture-fact-1"],
                        }],
                    }],
                }],
            }],
        })
        provider = AsyncMock(side_effect=[
            (skeleton, AiUsage()),
            TypeError("the JSON object must be str, bytes or bytearray, not NoneType"),
        ])
        manifest = {"facts": [{
            "fact_id": "fixture-fact-1",
            "source_page": 1,
            "text": "Synthetic evidence only for an offline provider-boundary regression test.",
        }]}

        with patch("app.main.generate_content", new=provider):
            with self.assertRaises(WorkflowFailure) as raised:
                asyncio.run(generate_staged_lesson_author_proposal(
                    staged_request(),
                    "Synthetic context",
                    "",
                    "",
                    source_rows=[],
                    source_coverage_manifest=manifest,
                ))

        self.assertEqual(raised.exception.code, "PROVIDER_ERROR")
        self.assertEqual(raised.exception.internal_code, "LESSON_PROVIDER_RESPONSE_DESERIALIZATION_FAILED")
        self.assertEqual(provider.await_count, 2)

    def test_stage_two_provider_schema_rejection_is_typed_and_not_retried(self) -> None:
        request = RagLessonAuthorRequest(**{
            **staged_request().model_dump(),
            "blueprint_architecture": {
                "architecture_contract_version": 5,
                "chapter_title": "Synthetic chapter",
                "lessons": [{
                    "title": "Synthetic lesson",
                    "units": [{
                        "title": "Synthetic unit",
                        "source_fact_ids": ["fixture-fact-1"],
                        "component_plan": [{
                            "type": "html",
                            "title": "Explanation",
                            "rationale": "Explain the assigned source fact.",
                            "source_fact_ids": ["fixture-fact-1"],
                        }],
                    }],
                }],
            },
        })
        manifest = {"facts": [{
            "fact_id": "fixture-fact-1", "source_page": 1,
            "text": "Synthetic evidence for a provider schema rejection test.",
        }]}

        class ProviderSchemaError(RuntimeError):
            status_code = 400

        provider = AsyncMock(side_effect=ProviderSchemaError("response_schema array items missing"))
        with patch("app.main.generate_content", new=provider):
            with self.assertRaises(WorkflowFailure) as raised:
                asyncio.run(generate_staged_lesson_author_proposal(
                    request, "Synthetic context", "", "", source_rows=[], source_coverage_manifest=manifest,
                ))

        self.assertEqual(raised.exception.code, "PROVIDER_ERROR")
        self.assertEqual(raised.exception.internal_code, "LESSON_PROVIDER_REQUEST_SCHEMA_INVALID")
        self.assertEqual(raised.exception.failure_stage, "staged_lesson_provider_schema_request")
        self.assertEqual(provider.await_count, 1)

    def test_v5_blueprint_skips_provider_topology_and_uses_stable_unit_paths(self) -> None:
        request = RagLessonAuthorRequest(**{
            **staged_request().model_dump(),
            "blueprint_architecture": {
                "architecture_contract_version": 5,
                "chapter_title": "Synthetic chapter",
                "lessons": [{
                    "title": "Synthetic lesson",
                    "units": [
                        {
                            "title": "Repeated title",
                            "source_fact_ids": ["fixture-fact-1"],
                            "component_plan": [{
                                "type": "html",
                                "title": "Explanation",
                                "rationale": "Explain the assigned source fact.",
                                "source_fact_ids": ["fixture-fact-1"],
                            }],
                        },
                        {
                            "title": "Repeated title",
                            "source_fact_ids": ["fixture-fact-2"],
                            "component_plan": [{
                                "type": "html",
                                "title": "Explanation",
                                "rationale": "Explain the assigned source fact.",
                                "source_fact_ids": ["fixture-fact-2"],
                            }],
                        },
                    ],
                }],
            },
        })
        manifest = {
            "facts": [
                {"fact_id": "fixture-fact-1", "source_page": 1, "text": "First synthetic source fact."},
                {"fact_id": "fixture-fact-2", "source_page": 2, "text": "Second synthetic source fact."},
            ],
        }

        def generated_unit(fact_id: str) -> tuple[str, AiUsage]:
            return json.dumps({
                "title": "Repeated title",
                "source_fact_ids": [fact_id],
                "components": [{
                    "type": "html",
                    "title": "Explanation",
                    "html": "<p>" + ("Synthetic source-grounded explanation. " * 20) + "</p>",
                    "source_fact_ids": [fact_id],
                    "covered_source_fact_ids": [fact_id],
                }],
            }), AiUsage(inputTokens=1, outputTokens=1, totalTokens=2)

        provider = AsyncMock(side_effect=[
            generated_unit("fixture-fact-1"),
            generated_unit("fixture-fact-2"),
        ])
        with patch("app.main.generate_content", new=provider):
            proposal, _usage = asyncio.run(generate_staged_lesson_author_proposal(
                request,
                "Synthetic context",
                "",
                "",
                source_rows=[],
                source_coverage_manifest=manifest,
            ))

        # Only the two bounded content calls are permitted: V5 topology must
        # not be recreated by a Stage-1 provider request.
        self.assertEqual(provider.await_count, 2)
        units = proposal["chapters"][0]["lessons"][0]["units"]
        self.assertEqual([unit["source_fact_ids"] for unit in units], [
            ["fixture-fact-1"],
            ["fixture-fact-2"],
        ])

    def test_semantic_only_html_is_validated_without_legacy_html_and_never_clipped(self) -> None:
        semantic_content = {
            "heading": "Synthetic heading",
            "paragraphs": ["Synthetic source-grounded explanation. " * 12],
            "ordered_steps": ["Prepare the approved source evidence before the learner action."],
        }
        expected = {
            "component_types": ["html"],
            "source_fact_ids": ["fixture-fact-1"],
            "component_plan": [{"type": "html", "source_fact_ids": ["fixture-fact-1"]}],
        }
        unit = {
            "title": "Semantic unit",
            "source_fact_ids": ["fixture-fact-1"],
            "components": [{
                "type": "html",
                "semantic_content": semantic_content,
                "source_fact_ids": ["fixture-fact-1"],
                "covered_source_fact_ids": ["fixture-fact-1"],
            }],
        }

        self.assertIsNone(validate_staged_unit_content(unit, expected))
        _visible, failure = semantic_learning_visible_text({
            "ordered_steps": [f"Step {index}" for index in range(21)],
        })
        self.assertIn("20-item", failure or "")

    def test_v5_supporting_only_unit_uses_read_only_evidence_without_claiming_it(self) -> None:
        request = RagLessonAuthorRequest(**{
            **staged_request().model_dump(),
            "blueprint_architecture": {
                "architecture_contract_version": 5,
                "chapter_title": "Supporting evidence chapter",
                "lessons": [{
                    "title": "Supporting evidence lesson",
                    "units": [
                        {
                            "title": "Teaching unit",
                            "source_fact_ids": ["fixture-fact-1"],
                            "component_plan": [{
                                "type": "html", "title": "Teach", "rationale": "Teach the owned fact.",
                                "source_fact_ids": ["fixture-fact-1"],
                                "supporting_evidence_fact_ids": [],
                            }],
                        },
                        {
                            "title": "Supporting assessment",
                            "source_fact_ids": [],
                            "supporting_evidence_fact_ids": ["fixture-fact-1"],
                            "learning_blocks": [{
                                "id": "lb_supporting_check",
                                "supporting_evidence_scope_ids": ["scope-01"],
                            }],
                            "component_plan": [{
                                "type": "problem", "title": "Check", "rationale": "Check taught evidence.",
                                "source_fact_ids": [],
                                "supporting_evidence_fact_ids": ["fixture-fact-1"],
                            }],
                        },
                    ],
                }],
            },
        })
        manifest = {
            "facts": [{
                "fact_id": "fixture-fact-1", "source_page": 1,
                "text": "The owned fact supplies the exact evidence for the supporting assessment.",
            }],
            "supporting_evidence_facts": [{
                "fact_id": "fixture-fact-1", "source_page": 1,
                "text": "The owned fact supplies the exact evidence for the supporting assessment.",
            }],
        }
        teaching = json.dumps({
            "title": "Teaching unit", "source_fact_ids": ["fixture-fact-1"], "supporting_evidence_fact_ids": [],
            "components": [{
                "type": "html", "html": "<p>" + ("The owned source fact is explained before a learner check. " * 12) + "</p>",
                "source_fact_ids": ["fixture-fact-1"], "covered_source_fact_ids": ["fixture-fact-1"],
                "supporting_evidence_fact_ids": [],
            }],
        })
        supporting_check = json.dumps({
            "title": "Supporting assessment", "source_fact_ids": [], "supporting_evidence_fact_ids": ["fixture-fact-1"],
            "components": [{
                "type": "problem", "problem_type": "short_text", "question": "What does the taught evidence require?",
                "answer": "Use the owned source fact.", "explanation": "The answer follows the taught evidence.",
                "source_fact_ids": [], "covered_source_fact_ids": [],
                "supporting_evidence_fact_ids": ["fixture-fact-1"],
            }],
        })
        provider = AsyncMock(side_effect=[
            (teaching, AiUsage(inputTokens=1, outputTokens=1, totalTokens=2)),
            (supporting_check, AiUsage(inputTokens=1, outputTokens=1, totalTokens=2)),
        ])
        with patch("app.main.generate_content", new=provider):
            proposal, _usage = asyncio.run(generate_staged_lesson_author_proposal(
                request, "Synthetic context", "", "", source_rows=[], source_coverage_manifest=manifest,
            ))

        supporting_unit = proposal["chapters"][0]["lessons"][0]["units"][1]
        self.assertEqual(supporting_unit["source_fact_ids"], [])
        self.assertEqual(supporting_unit["supporting_evidence_fact_ids"], ["fixture-fact-1"])
        self.assertEqual(provider.await_count, 2)


if __name__ == "__main__":
    unittest.main()
