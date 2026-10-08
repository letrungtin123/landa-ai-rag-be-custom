"""Offline unit-boundary and completion tests. No DB or real provider requests."""
from __future__ import annotations

import asyncio
import json
import unittest
from copy import deepcopy
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import HTTPException
from pydantic import ValidationError

from app import lesson_quality, main
from app.api import deps as api_deps
from app.lesson_author_checkpoint import ChapterCheckpointUnit, assemble_checkpoint_chapter, select_checkpoint_unit
from app.schemas.common import AiUsage
from app.schemas.lesson_author import RagLessonAuthorCheckpointRequest, RagLessonAuthorRequest
from app.services.lesson_author import chapter_checkpoint as chapter_checkpoint_service
from app.services.lesson_author import checkpoint as checkpoint_service
from app.services.lesson_author.staged import skeleton as staged_skeleton
from app.services.lesson_author.staged import source_locked as staged_source_locked
from app.workflows.contracts import WorkflowFailure, WorkflowValidationResult
from tests.test_lesson_prompt_policy import request_and_unit


def fixture(count=5, *, action="generate_unit", index=0):
    base, template = request_and_unit("html", "en")
    architecture = base.blueprint_architecture.model_dump()
    architecture["lessons"][0]["units"] = []
    units = []
    facts = []
    subjects = [
        "Observe equipment condition and identify hazards before beginning work. Inspection records describe machinery surroundings and affected operators.",
        "Compare risk probability against impact using the approved rating matrix. Scores distinguish priority levels and guide the selected control strategy.",
        "Isolate electrical energy at the authorized switch. Attach a lock and tag then verify absence of voltage before maintenance begins.",
        "Store chemical containers upright within secondary containment. Read handling labels and segregate incompatible substances in designated cabinets.",
        "Choose protective footwear suitable for the task. Check sizing, sole damage and material integrity before entering the designated work area.",
    ]
    for i in range(count):
        fact = f"fixture-fact-{i}"
        text = subjects[i % len(subjects)]
        generated = deepcopy(template)
        generated.update(title="Repeated title", source_fact_ids=[fact], supporting_evidence_fact_ids=[])
        generated["components"][0].update(source_fact_ids=[fact], covered_source_fact_ids=[fact], supporting_evidence_fact_ids=[],
                                            semantic_content={"version": 2, "sections": [{"heading": f"Topic {i}",
                                                "learning_block_ids": [], "blocks": [{"kind": "paragraph", "text": text * 5}]}]})
        units.append(generated)
        facts.append({"fact_id": fact, "source_page": i + 1, "text": text})
        architecture["lessons"][0]["units"].append({"title": "Repeated title", "source_fact_ids": [fact],
            "component_plan": [{"type": "html", "title": "Explanation", "rationale": "Teach the assigned evidence.",
                                "source_fact_ids": [fact]}]})
    data = {**base.model_dump(), "max_output_tokens": 30_000, "blueprint_architecture": architecture, "target_type": "chapter", "operation": "create",
            "generation_mode": "staged", "correlation_id": "55555555-5555-4555-8555-555555555555",
            "source_documents": [{"document_id": "66666666-6666-4666-8666-666666666666", "kb_id": base.kb_id,
                                  "name": "fixture.pdf", "type": "pdf", "status": "ready"}],
            "checkpoint_action": action, "remaining_workflow_budget_ms": 400_000,
            "checkpoint_unit_index": index if action == "generate_unit" else None,
            "checkpoint_units": [] if action == "generate_unit" else [{"unit_index": i, "unit": unit} for i, unit in enumerate(units)]}
    return RagLessonAuthorCheckpointRequest.model_validate(data), units, {"facts": facts}


def provider_result(unit, *, known=True, retry=False):
    async def generate(*args, **kwargs):
        callback = kwargs.get("on_provider_telemetry")
        if callback:
            if retry:
                callback({"event": "provider_transient_retry", "usage_source": "unavailable"})
            callback({"provider_http_status": 200, "provider_finish_reason": "FinishReason.STOP",
                      "usage_source": "provider" if known else "local_estimate",
                      "provider_input_tokens": 10 if known else None, "provider_output_tokens": 20 if known else None,
                      "provider_total_tokens": 30 if known else None})
        return json.dumps(unit), AiUsage(inputTokens=10, outputTokens=20, totalTokens=30)
    return AsyncMock(side_effect=generate)


async def checkpoint_result(request, manifest, events=None, retrieval_usage=None):
    return await checkpoint_service.build_lesson_author_checkpoint_result(
        request, context="Synthetic context", source_outline="", source_coverage="", rows=[], manifest=manifest,
        known_source_refs=set(), retrieval={}, retrieval_usage=retrieval_usage or AiUsage(), elapsed_ms=0,
        emit=(events if events is not None else []).append,
    )


class ChapterCheckpointTests(unittest.TestCase):
    def test_explicit_html_block_mismatch_fails_before_provider_without_mutating_plan(self):
        request, _, manifest = fixture(count=1)
        architecture = request.blueprint_architecture.model_dump()
        lesson = architecture["lessons"][0]
        lesson["learning_objectives"] = ["Explain the first concept.", "Explain the relationship."]
        unit = lesson["units"][0]
        unit["learning_objective_refs"] = ["lo_1", "lo_2"]
        unit["learning_blocks"] = [
            {"id": "block_a", "intent": "concept_explanation", "learning_objective_refs": ["lo_1"], "source_fact_ids": ["fixture-fact-0"]},
            {"id": "block_b", "intent": "relationship_visualization", "learning_objective_refs": ["lo_2"], "source_fact_ids": ["fixture-fact-0"]},
        ]
        unit["component_plan"][0]["learning_block_ids"] = ["block_a"]
        request = RagLessonAuthorCheckpointRequest.model_validate({**request.model_dump(), "blueprint_architecture": architecture})
        before = request.model_dump()
        events = []
        with patch("app.services.provider.generate_content", AsyncMock()) as provider, self.assertRaises(WorkflowFailure) as raised:
            asyncio.run(checkpoint_result(request, manifest, events))
        provider.assert_not_called()
        self.assertEqual(raised.exception.internal_code, "CHAPTER_INSTRUCTIONAL_PLAN_INVALID")
        self.assertEqual(events[0]["target_paths"], ["chapter_1.lesson_1.unit_1"])
        self.assertEqual(events[0]["validation_codes"], ["OBJECTIVE_INSTRUCTION_PLAN_MISMATCH"])
        self.assertEqual(request.model_dump(), before)
        self.assertNotIn("Explain", json.dumps(events))
        unit["component_plan"][0]["learning_block_ids"].append("block_b")
        self.assertFalse(lesson_quality.instructional_plan_validation_result(architecture).errors)
        unit["learning_blocks"][1]["intent"] = "knowledge_check"
        self.assertTrue(lesson_quality.instructional_plan_validation_result(architecture).errors)

    def test_request_rejects_non_v5_wrong_operation_and_provider_owned_state(self):
        request, _, _ = fixture()
        for update in ({"operation": "delete"}, {"target_type": "unit"}, {"correlation_id": None},
                       {"source_documents": []}, {"checkpoint_unit_index": 5}, {"checkpoint_unit_index": True},
                       {"checkpoint_unit_index": "1"}, {"lease_token": "not a provider field"},
                       {"remaining_workflow_budget_ms": 600_000}, {"remaining_workflow_budget_ms": 0},
                       {"blueprint_architecture": {**request.blueprint_architecture.model_dump(), "architecture_contract_version": 4}}):
            with self.subTest(update=list(update)), self.assertRaises(ValidationError):
                RagLessonAuthorCheckpointRequest.model_validate({**request.model_dump(), **update})

    def test_completion_inventory_rejects_duplicate_missing_and_out_of_scope_indices(self):
        request, _, _ = fixture(action="validate_chapter")
        original = request.model_dump()
        for units in (original["checkpoint_units"][:-1], original["checkpoint_units"][:-1] + [original["checkpoint_units"][0]],
                      [{**item, "unit_index": item["unit_index"] + 1} for item in original["checkpoint_units"]]):
            with self.assertRaises(ValidationError):
                RagLessonAuthorCheckpointRequest.model_validate({**original, "checkpoint_units": units})

    def test_unit_response_is_exact_selected_position_not_duplicate_title(self):
        request, units, manifest = fixture(index=3)
        provider = provider_result(units[3])
        with patch("app.services.provider.generate_content", provider):
            result = asyncio.run(checkpoint_result(request, manifest))
        self.assertEqual(provider.await_count, 1)
        self.assertEqual(result["unit_index"], 3)
        self.assertEqual(result["unit_path"], "chapter_1.lesson_1.unit_4")
        self.assertEqual(result["unit"]["source_fact_ids"], ["fixture-fact-3"])
        self.assertNotIn("proposal", result)
        self.assertNotIn("chapters", result)
        self.assertEqual(result["status"], "unit_ready")
        self.assertTrue(result["usage_complete"])

    def test_previous_completed_units_are_not_regenerated_on_explicit_new_selection(self):
        generated = []
        for index in (3, 4):
            request, units, manifest = fixture(index=index)
            provider = provider_result(units[index])
            with patch("app.services.provider.generate_content", provider):
                result = asyncio.run(checkpoint_result(request, manifest))
            generated.append(result["unit_index"])
            self.assertEqual(provider.await_count, 1)
        self.assertEqual(generated, [3, 4])  # Node persistence/resume is tested separately.

    def test_selected_unit_budget_ignores_unrelated_dense_unit(self):
        request, units, manifest = fixture(index=0)
        architecture = request.blueprint_architecture.model_dump()
        dense = [f"dense-{i}" for i in range(30)]
        architecture["lessons"][0]["units"][4]["source_fact_ids"] += dense
        architecture["lessons"][0]["units"][4]["component_plan"][0]["source_fact_ids"] += dense
        manifest["facts"] += [{"fact_id": f, "text": "Other unit evidence.", "source_page": 5} for f in dense]
        request = RagLessonAuthorCheckpointRequest.model_validate({**request.model_dump(), "blueprint_architecture": architecture})
        provider = provider_result(units[0])
        with patch("app.services.provider.generate_content", provider):
            asyncio.run(checkpoint_result(request, manifest))
        self.assertEqual(provider.await_args.kwargs["max_output_tokens"], min(request.max_output_tokens, 8192))
        self.assertLessEqual(provider.await_args.kwargs["request_timeout_ms"], 180_000)

    def test_remaining_invocation_budget_does_not_reset_for_each_call(self):
        request, units, manifest = fixture()
        request = request.model_copy(update={"remaining_workflow_budget_ms": 12_000})
        provider = provider_result(units[0])
        with patch("app.services.provider.generate_content", provider):
            asyncio.run(checkpoint_result(request, manifest))
        self.assertLessEqual(provider.await_args.kwargs["request_timeout_ms"], 12_000)
        self.assertGreater(provider.await_args.kwargs["request_timeout_ms"], 0)

    def test_timeout_is_terminal_provider_error_without_identical_request_retry(self):
        request, _, _ = fixture(index=3)
        failure = HTTPException(504, {"code": "AI_PROVIDER_TIMEOUT", "message": "safe"})
        provider_boundary = AsyncMock(side_effect=failure)
        with patch("app.services.lesson_author.proposal.lesson_author_proposal", provider_boundary):
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(chapter_checkpoint_service.lesson_author_chapter_checkpoint(request, pool=None))
        self.assertEqual(provider_boundary.await_count, 1)
        self.assertEqual(raised.exception.detail["code"], "PROVIDER_ERROR")
        self.assertEqual(raised.exception.detail["internal_failure_code"], "AI_PROVIDER_TIMEOUT")
        self.assertEqual(raised.exception.detail["failure_stage"], "chapter_checkpoint_provider")

    def test_outer_deadline_is_safe_and_does_not_retry(self):
        request, _, _ = fixture()
        request = request.model_copy(update={"remaining_workflow_budget_ms": 5})
        async def slow(*args):
            await asyncio.sleep(1)
        boundary = AsyncMock(side_effect=slow)
        with patch("app.services.lesson_author.proposal.lesson_author_proposal", boundary), self.assertRaises(HTTPException) as raised:
            asyncio.run(chapter_checkpoint_service.lesson_author_chapter_checkpoint(request, pool=None))
        self.assertEqual(raised.exception.detail["internal_failure_code"], "AI_STAGED_LESSON_WORKFLOW_TIMEOUT")
        self.assertEqual(boundary.await_count, 1)

    def test_endpoint_preserves_evidence_gate_before_checkpoint_generation(self):
        request, _, _ = fixture()
        with patch("app.services.retrieval.search.retrieve_chunks", AsyncMock(return_value=([], AiUsage(), {}))), \
             patch("app.services.retrieval.search.target_source_scope_is_incomplete", return_value=False), \
             patch("app.services.lesson_author.checkpoint.build_lesson_author_checkpoint_result", AsyncMock()) as generation:
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(chapter_checkpoint_service.lesson_author_chapter_checkpoint(request, pool=None))
        generation.assert_not_called()
        self.assertEqual(raised.exception.detail["code"], "SOURCE_EVIDENCE_INSUFFICIENT")
        self.assertEqual(raised.exception.detail["failure_stage"], "chapter_checkpoint_evidence_validation")

    def test_estimated_or_retried_provider_usage_does_not_claim_complete_accounting(self):
        request, units, manifest = fixture()
        for known, retry in ((False, False), (True, True)):
            with patch("app.services.provider.generate_content", provider_result(units[0], known=known, retry=retry)):
                result = asyncio.run(checkpoint_result(request, manifest))
            self.assertFalse(result["usage_complete"])
            self.assertNotEqual(result["usage_source"], "provider")
        with patch("app.services.provider.generate_content", provider_result(units[0])):
            result = asyncio.run(checkpoint_result(request, manifest, retrieval_usage=AiUsage(embeddingTokens=4, totalTokens=4)))
        self.assertFalse(result["usage_complete"])

    def test_full_completion_uses_all_existing_quality_gates_and_no_generation(self):
        request, _, manifest = fixture(count=1, action="validate_chapter")
        events = []
        with patch("app.services.provider.generate_content", AsyncMock(side_effect=AssertionError("No provider allowed"))) as provider:
            result = asyncio.run(checkpoint_result(request, manifest, events))
        provider.assert_not_called()
        self.assertEqual(result["status"], "ready")
        self.assertEqual([e["stage"] for e in events], ["chapter_checkpoint_content_validation",
                         "chapter_checkpoint_pedagogical_validation", "chapter_checkpoint_duplication_validation", "chapter_checkpoint_finalize"])
        self.assertEqual(result["retrieval"]["source_coverage_status"], "complete")

    def test_full_completion_rejects_changed_component_or_fact_scope(self):
        request, _, manifest = fixture(count=1, action="validate_chapter")
        for field, value in (("source_fact_ids", ["not-owned"]), ("title", "Renamed unit")):
            changed = request.model_dump()
            changed["checkpoint_units"][0]["unit"][field] = value
            with self.assertRaises(WorkflowFailure) as raised:
                asyncio.run(checkpoint_result(RagLessonAuthorCheckpointRequest.model_validate(changed), manifest))
            self.assertEqual(raised.exception.internal_code, "CHAPTER_CHECKPOINT_UNIT_INVALID")

    def test_full_quality_failure_cannot_finalize_or_repair_immutable_units(self):
        request, _, manifest = fixture(count=1, action="validate_chapter")
        bad = WorkflowValidationResult([{"code": "ASSESSMENT_NOT_ALIGNED", "severity": "error"}])
        with patch("app.services.lesson_author.checkpoint.pedagogical_validation_result", return_value=bad), patch("app.services.provider.generate_content", AsyncMock()) as provider:
            with self.assertRaises(WorkflowFailure) as raised:
                asyncio.run(checkpoint_result(request, manifest))
        self.assertEqual(raised.exception.failure_stage, "chapter_checkpoint_pedagogical_validation")
        provider.assert_not_called()

    def test_duplicate_explanations_fail_the_full_chapter_gate(self):
        request, _, manifest = fixture(count=2, action="validate_chapter")
        first = request.checkpoint_units[0].unit["components"][0]["semantic_content"]
        request.checkpoint_units[1].unit["components"][0]["semantic_content"] = deepcopy(first)
        with self.assertRaises(WorkflowFailure) as raised:
            asyncio.run(checkpoint_result(request, manifest))
        self.assertEqual(raised.exception.failure_stage, "chapter_checkpoint_duplication_validation")

    def test_assembly_is_indexed_transactional_and_never_partial(self):
        request, units, manifest = fixture(count=2)
        skeleton = staged_source_locked.build_source_locked_staged_skeleton(request, manifest)
        expected = [u for b in staged_skeleton.extract_lesson_author_unit_batches(skeleton, manifest) for u in b]
        before = deepcopy(skeleton)
        checkpoints = [ChapterCheckpointUnit(unit_index=i, unit=u) for i, u in enumerate(units)]
        result = assemble_checkpoint_chapter(skeleton, expected, list(reversed(checkpoints)))
        self.assertEqual(result["chapters"][0]["lessons"][0]["units"][1]["source_fact_ids"], ["fixture-fact-1"])
        self.assertEqual(skeleton, before)
        with self.assertRaises(ValueError):
            assemble_checkpoint_chapter(skeleton, expected, checkpoints[:1])
        with self.assertRaises(ValueError):
            select_checkpoint_unit([expected], True)

    def test_request_errors_never_echo_provider_key_or_checkpoint_content(self):
        request, _, _ = fixture()
        data = request.model_dump()
        data.update(api_key="synthetic-secret-do-not-echo", checkpoint_unit_index=100)
        async def request_http():
            main.app.dependency_overrides[api_deps.require_internal_token] = lambda: None
            main.app.dependency_overrides[api_deps.get_db] = lambda: None
            try:
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
                    return await client.post("/v1/lesson-author/chapter-checkpoint", json=data)
            finally:
                main.app.dependency_overrides.pop(api_deps.require_internal_token, None)
                main.app.dependency_overrides.pop(api_deps.get_db, None)
        response = asyncio.run(request_http())
        self.assertEqual(response.status_code, 422)
        self.assertNotIn("synthetic-secret", response.text)
        self.assertEqual(response.json()["detail"]["code"], "CHAPTER_CHECKPOINT_CONTRACT_INVALID")

    def test_internal_token_is_still_required_before_generation(self):
        request, _, _ = fixture()
        async def request_http():
            main.app.dependency_overrides[api_deps.get_db] = lambda: None
            try:
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
                    payload = request.model_dump(mode="json")
                    payload["api_key"] = request.api_key.get_secret_value()
                    return await client.post("/v1/lesson-author/chapter-checkpoint", json=payload)
            finally:
                main.app.dependency_overrides.pop(api_deps.get_db, None)
        with patch.object(main.settings, "service_token", "test-only-internal-token"), patch("app.services.provider.generate_content", AsyncMock()) as provider:
            response = asyncio.run(request_http())
        self.assertIn(response.status_code, (401, 403))
        provider.assert_not_called()

    def test_supporting_only_unit_keeps_zero_canonical_owned_facts(self):
        request, units, manifest = fixture(count=2, index=1)
        architecture = request.blueprint_architecture.model_dump()
        scope = architecture["lessons"][0]["units"][1]
        scope.update(source_fact_ids=[], supporting_evidence_fact_ids=["fixture-fact-0"],
                     learning_blocks=[{"id": "supporting_check", "supporting_evidence_scope_ids": ["scope-1"]}])
        scope["component_plan"] = [{"type": "problem", "title": "Check", "rationale": "Check taught evidence.",
                                    "source_fact_ids": [], "supporting_evidence_fact_ids": ["fixture-fact-0"]}]
        units[1].update(source_fact_ids=[], supporting_evidence_fact_ids=["fixture-fact-0"], components=[{
            "type": "problem", "problem_type": "multiple_choice",
            "question": "What must be inspected before work begins?",
            "choices": [
                {"text": "The equipment condition.", "correct": True},
                {"text": "Only the final incident report.", "correct": False},
                {"text": "An unrelated administrative record.", "correct": False},
            ],
            "explanation": "The evidence calls for inspecting the equipment condition before work begins.",
            "source_fact_ids": [], "covered_source_fact_ids": [], "supporting_evidence_fact_ids": ["fixture-fact-0"],
        }])
        manifest["facts"] = manifest["facts"][:1]
        manifest["supporting_evidence_facts"] = deepcopy(manifest["facts"])
        request = RagLessonAuthorCheckpointRequest.model_validate({**request.model_dump(), "blueprint_architecture": architecture})
        with patch("app.services.provider.generate_content", provider_result(units[1])):
            result = asyncio.run(checkpoint_result(request, manifest))
        self.assertEqual(result["unit"]["source_fact_ids"], [])
        self.assertEqual(result["unit"]["supporting_evidence_fact_ids"], ["fixture-fact-0"])

    def test_regular_proposal_model_does_not_activate_checkpoint_from_extra_fields(self):
        request, _, _ = fixture()
        regular = RagLessonAuthorRequest.model_validate(request.model_dump())
        self.assertNotIsInstance(regular, RagLessonAuthorCheckpointRequest)
        self.assertNotIn("checkpoint_action", regular.model_dump())


if __name__ == "__main__":
    unittest.main()
