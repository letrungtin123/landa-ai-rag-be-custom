from __future__ import annotations

import asyncio
import json
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, patch

from google import genai
from google.genai import models, types

from app.main import (
    AiUsage,
    RagLessonAuthorRequest,
    build_staged_lesson_content_response_model,
    generate_staged_lesson_author_proposal,
    semantic_learning_visible_text,
    validate_staged_unit_content,
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
