from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

from app.lesson_prompt_policy import bounded_architect_policy, lesson_output_language_policy, lesson_instructional_quality_policy
from app.main import (
    AiUsage, build_course_architect_prompt, build_lesson_generation_repair_prompt,
    generate_staged_lesson_author_proposal, generate_validated_lesson_author_blueprint,
    enrich_lesson_author_blueprint_media_review,
)
from app.workflows.contracts import WorkflowFailure
from tests.test_lesson_author_blueprint import blueprint_request, proposal_request, valid_blueprint


def content_payload(component_type: str, locale: str) -> dict:
    # Synthetic transport/roundtrip fixture, not a factual-quality evaluation.
    vi = locale == "vi"
    explanation = ("Kiểm tra điều kiện trước khi thực hiện quy trình đã được xác nhận. " if vi else
                   "Check the conditions before performing the approved procedure. ") * 12
    fields = {
        "html": {"semantic_content": {"heading": "Kiểm tra" if vi else "Check", "paragraphs": [explanation]}},
        "problem": {"problem_type": "short_text", "question": "Bước nào trước tiên?" if vi else "Which step is first?",
                    "answer": "Kiểm tra" if vi else "Check", "explanation": explanation},
        "la_faq": {"items": [
            {"question": "Khi nào kiểm tra điều kiện?" if vi else "When should conditions be checked?", "answer": explanation},
            {"question": "Vì sao cần giữ đúng thứ tự?" if vi else "Why preserve the approved order?", "answer": explanation},
        ]},
        "la_sortable": {"items": ["Kiểm tra điều kiện ban đầu.", "Thực hiện bước đã phê duyệt.", "Xác nhận kết quả cuối cùng."] if vi else
                        ["Check the initial conditions.", "Perform the approved action.", "Confirm the final result."]},
        "la_crossword": {"words": [{"answer": answer, "clue": clue} for answer, clue in
                                   [("CHECK", "Kiểm tra điều kiện" if vi else "Inspect conditions"),
                                    ("ACT", "Thực hiện bước" if vi else "Perform the step"),
                                    ("CONFIRM", "Xác nhận kết quả" if vi else "Verify the result")]]},
        "la_diagram": {"nodes": [{"id": "node-1", "label": "Kiểm tra" if vi else "Check"},
                                 {"id": "node-2", "label": "Thực hiện" if vi else "Act"}],
                       "edges": [{"source": "node-1", "target": "node-2"}]},
    }[component_type]
    return {"type": component_type, "title": "Học liệu" if vi else "Learning material",
            "source_fact_ids": ["fact-1"], "covered_source_fact_ids": ["fact-1"], **fields}


def request_and_unit(component_type: str, locale: str):
    types = ["html"] if component_type == "html" else ["html", component_type]
    generated = {"title": "Locked source unit", "source_fact_ids": ["fact-1"],
                 "components": [content_payload(kind, locale) for kind in types]}
    request = proposal_request(locale=locale, blueprint_architecture={
        "architecture_contract_version": 5, "chapter_title": "Locked source chapter",
        "lessons": [{"title": "Locked source lesson", "units": [{
            "title": generated["title"], "source_fact_ids": ["fact-1"],
            "component_plan": [{"type": kind, "title": "Approved plan", "rationale": "Synthetic fixture",
                                "source_fact_ids": ["fact-1"]} for kind in types],
        }]}],
    })
    return request, generated


class LessonPromptPolicyTests(unittest.TestCase):
    def test_teaching_policy_allows_synthesis_not_new_facts_or_asset_claims(self):
        policy = lesson_instructional_quality_policy()
        for phrase in ("synthesize questions", "Never create canonical IDs",
                       "Put FAQ last", "absent image", "specific uncertainty", "non-instructional source note",
                       "component instances", "not always place the correct option first"):
            self.assertIn(phrase, policy)

    def test_architect_never_silently_cuts_mandatory_policy(self):
        exact = "x" * 12000
        self.assertEqual(bounded_architect_policy(exact), exact)
        self.assertIn(exact + "\n</STORED_SYSTEM_PROMPT>", build_course_architect_prompt(blueprint_request(exact), ""))
        with self.assertRaises(WorkflowFailure) as raised:
            build_course_architect_prompt(blueprint_request("x" * 17563), "")
        self.assertEqual(raised.exception.internal_code, "ARCH_POLICY_BUDGET_EXCEEDED")
        self.assertEqual(raised.exception.failure_stage, "course_architect_policy_composition")
        self.assertEqual(raised.exception.diagnostics["policy_chars"], 17563)

    def test_architect_final_provider_receives_complete_policy_and_language(self):
        for locale in ("vi", "en"):
            request = blueprint_request("SERVER POLICY VERSION: v5-blueprint-policy-1.\n<STORED_TEACHING_POLICY>\nSafe teaching policy.\n</STORED_TEACHING_POLICY>").model_copy(update={"locale": locale})
            prompt = build_course_architect_prompt(request, "Synthetic evidence only.")
            provider = AsyncMock(return_value=(json.dumps(valid_blueprint()), AiUsage()))
            with patch("app.main.generate_content", provider):
                asyncio.run(generate_validated_lesson_author_blueprint(request, prompt))
            self.assertEqual(provider.await_count, 1)
            sent = provider.await_args.args[2]
            self.assertIn(request.system_prompt, sent)
            self.assertIn("tiếng Việt có dấu" if locale == "vi" else "English", sent)
            self.assertIn("không trả estimated_minutes", sent)
            self.assertIn("architecture_contract_version=5", sent)
            self.assertEqual(provider.await_args.kwargs["max_output_tokens"], request.max_output_tokens)
            self.assertEqual(provider.await_args.kwargs["request_timeout_ms"], 300000)

    def test_four_way_language_matrix_at_actual_stage_two_boundary_and_six_payload_roundtrips(self):
        for source_locale in ("vi", "en"):
            source = "Kiểm tra, thực hiện, xác nhận theo đúng thứ tự." if source_locale == "vi" else "Check, act, then confirm in the approved order."
            for output_locale in ("vi", "en"):
                for component_type in ("html", "problem", "la_faq", "la_crossword", "la_diagram", "la_sortable"):
                    with self.subTest(source=source_locale, output=output_locale, component=component_type):
                        request, generated = request_and_unit(component_type, output_locale)
                        provider = AsyncMock(return_value=(json.dumps(generated, ensure_ascii=False), AiUsage()))
                        with patch("app.main.generate_content", provider):
                            proposal, _ = asyncio.run(generate_staged_lesson_author_proposal(
                                request, source, "", source_rows=[], source_coverage_manifest={
                                    "facts": [{"fact_id": "fact-1", "source_page": 1, "text": source}],
                                },
                            ))
                        self.assertEqual(provider.await_count, 1)
                        sent = provider.await_args.args[2]
                        self.assertTrue(sent.startswith(lesson_output_language_policy(output_locale)))
                        self.assertIn(source, sent)
                        self.assertIn("Locked source unit", sent)
                        self.assertIn("crossword clues, diagram labels", sent)
                        self.assertEqual(provider.await_args.kwargs["request_timeout_ms"], 180000)
                        # Only selected payload schema (plus shared contract fields).
                        schema = provider.await_args.kwargs["response_schema"].model_json_schema()
                        schema_text = json.dumps(schema)
                        if component_type != "la_crossword":
                            self.assertNotIn('"words":', schema_text)
                        unit = proposal["chapters"][0]["lessons"][0]["units"][0]
                        self.assertEqual(unit["title"], generated["title"])
                        self.assertEqual(unit["source_fact_ids"], ["fact-1"])
                        self.assertEqual(unit["components"], generated["components"])

    def test_recovery_uses_same_language_policy_without_extra_retry(self):
        request, generated = request_and_unit("html", "en")
        invalid = {**generated, "components": []}
        provider = AsyncMock(side_effect=[(json.dumps(invalid), AiUsage()), (json.dumps(generated), AiUsage())])
        with patch("app.main.generate_content", provider):
            proposal, _ = asyncio.run(generate_staged_lesson_author_proposal(
                request, "Kiểm tra điều kiện.", "", source_rows=[],
                source_coverage_manifest={"facts": [{"fact_id": "fact-1", "text": "Kiểm tra điều kiện."}]},
            ))
        self.assertEqual(provider.await_count, 2)
        for call in provider.await_args_list:
            self.assertTrue(call.args[2].startswith(lesson_output_language_policy("en")))
            self.assertIn(lesson_instructional_quality_policy(), call.args[2])
        self.assertIn("SERVER STAGE 2 RECOVERY", provider.await_args_list[1].args[2])
        self.assertEqual(proposal["chapters"][0]["title"], "Locked source chapter")

    def test_legacy_skeleton_and_skeleton_recovery_also_receive_controlled_locale(self):
        from tests.test_staged_lesson_timeout import staged_request, staged_content
        request = staged_request().model_copy(update={"locale": "en"})
        provider = AsyncMock(side_effect=[("{", AiUsage()), ("{", AiUsage()), (staged_content(), AiUsage())])
        with patch("app.main.generate_content", provider):
            asyncio.run(generate_staged_lesson_author_proposal(
                request, "Synthetic source", "", source_rows=[], source_coverage_manifest={
                    "facts": [{"fact_id": "p10-f1", "source_page": 10, "text": "Nhận diện mối nguy"}],
                },
            ))
        self.assertEqual(provider.await_count, 3)
        for call in provider.await_args_list:
            self.assertIn(lesson_output_language_policy("en"), call.args[2])
        self.assertNotIn("request_timeout_ms", provider.await_args_list[0].kwargs)
        self.assertNotIn("request_timeout_ms", provider.await_args_list[1].kwargs)
        self.assertIn("SERVER STAGE 1 RECOVERY", provider.await_args_list[1].args[2])

    def test_scoped_lesson_repair_preserves_language_and_target_contract(self):
        proposal = {"chapters": [{"lessons": [{"units": [{"components": [content_payload("problem", "vi")]}]}]}]}
        for locale in ("vi", "en"):
            prompt = build_lesson_generation_repair_prompt(
                proposal=proposal, locale=locale, evidence_context="Synthetic evidence",
                targets=[{"path": "chapter_1.lesson_1.unit_1.component_1", "scope": "component",
                          "codes": ["COMPONENT_INVALID"], "allowed_fields": ["question"]}],
            )
            self.assertIn(lesson_output_language_policy(locale), prompt)
            self.assertIn("only use allowed_fields", prompt)
            self.assertIn("Do not change course hierarchy, source scope", prompt)

    def test_media_framing_labels_original_language_without_fake_translation(self):
        for locale in ("vi", "en"):
            blueprint = {"architecture_contract_version": 5, "chapters": [{"lessons": [{"units": [{
                "title": "Locked title", "source_fact_ids": ["fact-1"], "learning_blocks": [{"intent": "procedure"}],
            }]}]}]}
            enriched, _ = enrich_lesson_author_blueprint_media_review(
                blueprint, {"facts": [{"fact_id": "fact-1", "text": "Nguyên văn nguồn."}]}, locale,
            )
            outline = enriched["chapters"][0]["lessons"][0]["units"][0]["media_plan"]["content_outline"]
            self.assertIn("Nguyên văn nguồn.", outline)
            self.assertIn("ngôn ngữ gốc" if locale == "vi" else "original language", outline)


if __name__ == "__main__":
    unittest.main()
