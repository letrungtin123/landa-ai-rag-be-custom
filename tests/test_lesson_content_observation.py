"""Synthetic EN/VI quality-risk regressions; no customer data or paid calls."""
import asyncio
from copy import deepcopy
import json
import unittest
from unittest.mock import AsyncMock, patch

from app.lesson_content_observation import observe_lesson_content, MAX_CANDIDATES, MAX_FACTS, MAX_FINDINGS
from app.lesson_prompt_policy import component_instructional_brief, lesson_instructional_quality_policy


def fixture(fact="Do not operate equipment before controls are verified.", teaching=None):
    expected = {"source_fact_ids": ["f1"], "component_plan": [{"type": "html", "source_fact_ids": ["f1"]}],
                "learning_objectives": [], "learning_objective_refs": []}
    unit = {"components": [{"type": "html", "source_fact_ids": ["f1"], "covered_source_fact_ids": ["f1"],
                            "semantic_content": {"paragraphs": [fact if teaching is None else teaching]}}]}
    return unit, expected, {"facts": [{"fact_id": "f1", "text": fact}]}


def codes(report):
    return set(report["finding_counts"])


class ContentObservationTests(unittest.TestCase):
    def test_lexical_witness_is_located_but_never_semantic_pass(self):
        args = fixture()
        before = deepcopy(args)
        result = observe_lesson_content(*args)
        self.assertEqual(result["fact_state_counts"], {"LEXICAL_WITNESS": 1, "NOT_ASSESSED": 0})
        self.assertEqual(result["fact_witnesses"][0]["candidate_path"], "components[0].semantic_content.paragraphs[0]")
        self.assertEqual(result["semantic_coverage"], "not_measured")
        self.assertFalse(result["blocking"])
        self.assertEqual(args, before)

    def test_declared_full_coverage_and_id_in_metadata_do_not_prove_teaching(self):
        report = observe_lesson_content(*fixture(teaching="A different topic is discussed here."))
        self.assertIn("FACT_TEACHING_NOT_LOCATED", codes(report))
        self.assertEqual(report["fact_state_counts"]["REVIEW_REQUIRED"], 1)

    def test_negation_loss_in_english_and_vietnamese(self):
        for source, text in [
            ("Do not operate equipment before controls are verified.", "Operate equipment before controls are verified."),
            ("Không vận hành thiết bị khi chưa được kiểm soát.", "Vận hành thiết bị khi được kiểm soát."),
        ]:
            self.assertIn("FACT_NEGATION_DIFFERENCE_REVIEW", codes(observe_lesson_content(*fixture(source, text))))

    def test_negated_candidate_does_not_count_as_positive_literal_witness(self):
        source = "Operate equipment after approved safety checks."
        report = observe_lesson_content(*fixture(source, "Do not operate equipment after approved safety checks."))
        self.assertIn("FACT_NEGATION_DIFFERENCE_REVIEW", codes(report))
        self.assertNotIn("LEXICAL_WITNESS", report["fact_state_counts"])

    def test_arithmetic_mismatch_is_observed_but_not_a_domain_fact_judgement(self):
        unit, expected, manifest = fixture()
        unit["components"][0]["semantic_content"]["paragraphs"].append("Hypothetical example: 3 x 4 = 11.")
        self.assertIn("ARITHMETIC_EXAMPLE_RESULT_MISMATCH", codes(observe_lesson_content(unit, expected, manifest)))
        unit["components"][0]["semantic_content"]["paragraphs"][-1] = "Hypothetical example: 1.5 x 4 = 6 points"
        result = observe_lesson_content(unit, expected, manifest)
        self.assertNotIn("ARITHMETIC_EXAMPLE_RESULT_MISMATCH", codes(result))
        self.assertEqual(result["semantic_fidelity"], "not_measured")

    def test_incorrect_arithmetic_distractor_is_not_a_teaching_error(self):
        unit, expected, manifest = fixture()
        unit["components"].append({"type": "problem", "question": "Which calculation is correct?",
            "choices": [{"text": "3 x 4 = 11", "correct": False}, {"text": "3 x 4 = 12", "correct": True}]})
        self.assertNotIn("ARITHMETIC_EXAMPLE_RESULT_MISMATCH", codes(observe_lesson_content(unit, expected, manifest)))

    def test_changed_number_and_decimal_are_reviewed(self):
        for source, text in [
            ("The approved inspection interval is 30 days.", "The approved inspection interval is 60 days."),
            ("Maintain the stated distance of 1.5 metres.", "Maintain the stated distance of 1-5 metres."),
        ]:
            self.assertIn("FACT_NUMBER_DIFFERENCE_REVIEW", codes(observe_lesson_content(*fixture(source, text))))

    def test_changed_material_and_condition_require_review(self):
        for source, text in [
            ("Tiết kiệm điện nước và nguyên liệu.", "Tiết kiệm điện nước và nhiên liệu."),
            ("Use protective gloves when handling chemical containers.", "Use protective gloves when handling containers."),
            ("Maintain the stated distance of 15 metres.", "Maintain the stated distance of 15 centimetres."),
        ]:
            self.assertIn("FACT_PARAPHRASE_OR_DETAIL_REVIEW", codes(observe_lesson_content(*fixture(source, text))))

    def test_correct_paraphrase_or_translation_is_review_not_proven_failure(self):
        report = observe_lesson_content(*fixture("Do not operate equipment before controls are verified.",
                                                "Chỉ dùng máy sau khi đã xác nhận biện pháp kiểm soát."))
        self.assertFalse(report["blocking"])
        self.assertEqual(report["fact_state_counts"]["REVIEW_REQUIRED"], 1)
        self.assertNotIn("FAIL", json.dumps(report))

    def test_quiz_distractors_and_faq_cannot_substitute_for_teaching(self):
        unit, expected, manifest = fixture(teaching="Other topic only.")
        source = manifest["facts"][0]["text"]
        unit["components"].extend([{"type": "problem", "choices": [{"text": source, "correct": False}]},
                                   {"type": "la_faq", "items": [{"question": "Clarification?", "answer": source}]}])
        expected["component_plan"].extend([{"type": "problem"}, {"type": "la_faq"}])
        self.assertIn("FACT_TEACHING_NOT_LOCATED", codes(observe_lesson_content(unit, expected, manifest)))

    def test_other_component_source_owner_does_not_supply_witness(self):
        unit, expected, manifest = fixture()
        expected["component_plan"][0]["source_fact_ids"] = ["different-fact"]
        self.assertIn("FACT_TEACHING_NOT_LOCATED", codes(observe_lesson_content(unit, expected, manifest)))

    def test_unknown_and_duplicate_manifest_and_short_fragment_are_unassessed(self):
        unit, expected, manifest = fixture()
        for facts, code in [([], "FACT_TEXT_UNAVAILABLE"),
                            (manifest["facts"] * 2, "FACT_MANIFEST_AMBIGUOUS"),
                            ([{"fact_id": "f1", "text": "HIRA"}], "FACT_FRAGMENT_REQUIRES_REVIEW")]:
            report = observe_lesson_content(unit, expected, {"facts": facts})
            self.assertEqual(report["fact_state_counts"]["NOT_ASSESSED"], 1)
            self.assertIn(code, codes(report))

    def test_supporting_only_unit_does_not_require_new_owned_facts(self):
        unit, expected, manifest = fixture()
        expected["source_fact_ids"] = []
        expected["supporting_evidence_fact_ids"] = ["f1"]
        report = observe_lesson_content(unit, expected, manifest)
        self.assertEqual(report["owned_fact_count"], 0)
        self.assertEqual(report["findings"], [])

    def test_1001_fact_fixture_deterministic_bounded_complete_inventory(self):
        texts = [f"Synthetic instruction number {i} requires checking approved operating conditions." for i in range(1001)]
        ids = [f"f{i}" for i in range(1001)]
        unit = {"components": [{"type": "html", "semantic_content": {"paragraphs": [" ".join(texts[i:i+3]) for i in range(0, 1001, 3)]}}]}
        expected = {"source_fact_ids": ids, "component_plan": [{"type": "html", "source_fact_ids": ids}]}
        manifest = {"facts": [{"fact_id": f, "text": t} for f, t in zip(ids, texts)]}
        report = observe_lesson_content(unit, expected, manifest)
        self.assertEqual(report, observe_lesson_content(unit, expected, manifest))
        self.assertEqual(report["fact_state_counts"]["LEXICAL_WITNESS"], 1001)
        self.assertLessEqual(report["candidate_comparisons"], 1001 * MAX_CANDIDATES)
        self.assertEqual(len(report["fact_witnesses"]) + report["omitted_witness_count"], 1001)

    def test_budget_exhaustion_is_visible_and_does_not_declare_missing(self):
        unit, expected, manifest = fixture(teaching="x" * 4001)
        report = observe_lesson_content(unit, expected, manifest)
        self.assertIn("FACT_OBSERVATION_BUDGET", codes(report))
        self.assertTrue(report["content_scan_limited"])
        expected["source_fact_ids"] = [f"f{i}" for i in range(MAX_FACTS + 3)]
        report = observe_lesson_content(unit, expected, {"facts": []})
        self.assertEqual(report["fact_state_counts"]["NOT_ASSESSED"], MAX_FACTS + 3)
        self.assertLessEqual(len(report["findings"]), MAX_FINDINGS)
        self.assertGreater(report["omitted_finding_count"], 0)

    def test_content_hashes_paths_counts_only_in_diagnostics(self):
        report = observe_lesson_content(*fixture("PRIVATE_TOKEN private source phrase only."))
        serialized = json.dumps(report)
        self.assertNotIn("PRIVATE_TOKEN", serialized)
        self.assertNotIn("private source", serialized)
        self.assertNotIn('"f1"', serialized)

    def test_missing_calculation_task_and_worked_example_detected_in_both_languages(self):
        for objective in ("Calculate the risk score", "Tính toán điểm rủi ro"):
            unit, expected, manifest = fixture()
            expected.update(learning_objectives=[objective], learning_objective_refs=["lo_1"])
            report = observe_lesson_content(unit, expected, manifest)
            self.assertIn("ACTION_ASSESSMENT_NOT_PLANNED_REVIEW", codes(report))
            self.assertIn("WORKED_CALCULATION_NOT_OBSERVED", codes(report))

    def test_unit_calculation_title_is_a_hint_without_reassigning_objectives(self):
        unit, expected, manifest = fixture()
        expected["unit_title"] = "Đánh giá và Tính toán Mức độ Rủi ro"
        report = observe_lesson_content(unit, expected, manifest)
        finding = next(f for f in report["findings"] if f["code"] == "ACTION_ASSESSMENT_NOT_PLANNED_REVIEW")
        self.assertIsNone(finding["objective_ref"])

    def test_calculation_check_uses_bound_objective_and_question_not_feedback(self):
        unit, expected, manifest = fixture()
        expected.update(learning_objectives=["Calculate the risk score", "Identify terms"], learning_objective_refs=["lo_1"])
        expected["component_plan"].append({"type": "problem", "learning_objective_refs": ["lo_1"]})
        unit["components"].append({"type": "problem", "question": "What is risk?", "explanation": "Calculate 3 x 4 = 12"})
        self.assertIn("CALCULATION_TASK_NOT_OBSERVED", codes(observe_lesson_content(unit, expected, manifest)))
        unit["components"][1]["question"] = "Hypothetical inputs 3 and 4: calculate the risk score."
        unit["components"][0]["semantic_content"]["paragraphs"].append("Hypothetical example: 2 x 4 = 8.")
        result = observe_lesson_content(unit, expected, manifest)
        self.assertNotIn("CALCULATION_TASK_NOT_OBSERVED", codes(result))
        self.assertNotIn("WORKED_CALCULATION_NOT_OBSERVED", codes(result))
        expected["component_plan"][1]["learning_objective_refs"] = ["lo_2"]
        self.assertIn("ACTION_ASSESSMENT_NOT_PLANNED_REVIEW", codes(observe_lesson_content(unit, expected, manifest)))

    def test_unrelated_lesson_objective_does_not_create_unit_demand(self):
        unit, expected, manifest = fixture()
        expected.update(learning_objectives=["Identify the terms", "Calculate risk"], learning_objective_refs=["lo_1"])
        self.assertNotIn("WORKED_CALCULATION_NOT_OBSERVED", codes(observe_lesson_content(unit, expected, manifest)))

    def test_analysis_recall_question_is_only_a_review_signal(self):
        unit, expected, manifest = fixture()
        expected.update(learning_objectives=["Phân tích tình huống"], learning_objective_refs=["lo_1"])
        expected["component_plan"].append({"type": "problem", "learning_objective_refs": ["lo_1"]})
        unit["components"].append({"type": "problem", "question": "JSA là gì?"})
        result = observe_lesson_content(unit, expected, manifest)
        self.assertIn("APPLICATION_TASK_REQUIRES_REVIEW", codes(result))
        self.assertFalse(result["blocking"])

    def test_visible_ids_detected_without_rewriting_payload_and_choice_positions_counted(self):
        unit, expected, manifest = fixture()
        unit["components"].append({"type": "problem", "question": "What is the first step?", "explanation": "Refer to p5-f2.",
                                   "choices": [{"text": "Check", "correct": True}, {"text": "Check", "correct": False}]})
        before = deepcopy(unit)
        result = observe_lesson_content(unit, expected, manifest)
        self.assertIn("LEARNER_INTERNAL_ID_VISIBLE", codes(result))
        self.assertIn("ASSESSMENT_DUPLICATE_CHOICES_REVIEW", codes(result))
        self.assertEqual(result["correct_choice_position_counts"], {"0": 1})
        self.assertEqual(unit, before)

    def test_prompt_has_action_brief_and_no_new_component_authority(self):
        unit, expected, _ = fixture()
        expected.update(learning_objectives=["Tính toán rủi ro"], learning_objective_refs=["lo_1"])
        expected["component_plan"][0]["learning_objective_refs"] = ["lo_1"]
        brief = component_instructional_brief(expected)
        self.assertEqual(brief["components"][0]["objective_actions"], [{"objective_ref": "lo_1", "actions": ["calculation"]}])
        policy = lesson_instructional_quality_policy()
        self.assertIn("never add a component outside the plan", policy)
        self.assertIn("actual teaching paragraph", policy)
        self.assertIn("Do not place internal fact IDs", policy)

    def test_actual_stage_two_logs_shadow_result_without_extra_call_or_payload_change(self):
        from app import main
        from tests.test_lesson_prompt_policy import request_and_unit
        request, generated = request_and_unit("html", "vi")
        provider = AsyncMock(return_value=(json.dumps(generated), main.AiUsage()))
        with patch("app.main.generate_content", provider), patch("app.main.logger.info") as log:
            proposal, _ = asyncio.run(main.generate_staged_lesson_author_proposal(
                request, "Private evidence", "", source_rows=[],
                source_coverage_manifest={"facts": [{"fact_id": "fact-1", "text": "Private evidence is not taught in this fixture."}]}))
        reports = [json.loads(c.args[1]) for c in log.call_args_list if c.args[0] == "lesson_author_content_observation %s"]
        self.assertEqual(len(reports), 1)
        self.assertFalse(reports[0]["blocking"])
        self.assertEqual(reports[0]["conversation_id"], request.conversation_id)
        self.assertIn("unit_path", reports[0])
        self.assertEqual(provider.await_count, 1)
        self.assertEqual(proposal["chapters"][0]["lessons"][0]["units"][0]["components"], generated["components"])
        self.assertIn('"objective_actions"', provider.await_args.args[2])

    def test_shadow_exception_does_not_fail_generation_or_expose_error_body(self):
        from app import main
        from tests.test_lesson_prompt_policy import request_and_unit
        request, generated = request_and_unit("html", "en")
        provider = AsyncMock(return_value=(json.dumps(generated), main.AiUsage()))
        with patch("app.main.generate_content", provider), patch("app.main.logger.info") as log, \
                patch("app.main.observe_lesson_content", side_effect=ValueError("PRIVATE_EXCEPTION")):
            asyncio.run(main.generate_staged_lesson_author_proposal(request, "Evidence", "", source_rows=[],
                source_coverage_manifest={"facts": [{"fact_id": "fact-1", "text": "Evidence"}]}))
        reports = [c.args[1] for c in log.call_args_list if c.args[0] == "lesson_author_content_observation %s"]
        self.assertIn("CONTENT_OBSERVATION_UNAVAILABLE", reports[0])
        self.assertNotIn("PRIVATE_EXCEPTION", reports[0])
        self.assertEqual(provider.await_count, 1)

    def test_existing_scoped_repair_keeps_two_calls_and_observes_final_checkpoint_once(self):
        from app import main
        from tests.test_checkpoint_coverage_repair import coverage_fixture
        from tests.test_checkpoint_component_quality_repair import instance_wire
        from tests.test_chapter_checkpoint import checkpoint_result
        request, valid, broken, scope, manifest, delta = coverage_fixture()
        provider = AsyncMock(side_effect=[(json.dumps(instance_wire(broken)), main.AiUsage()),
                                          (json.dumps(delta), main.AiUsage())])
        with patch("app.main.generate_content", provider), patch("app.main.logger.info") as log:
            result = asyncio.run(checkpoint_result(request, manifest))
        self.assertEqual(result["status"], "unit_ready")
        self.assertEqual(provider.await_count, 2)
        for i in (0, 1, 3):
            self.assertEqual(result["unit"]["components"][i], valid["components"][i])
        events = [json.loads(c.args[1]) for c in log.call_args_list if c.args[0] == "lesson_author_content_observation %s"]
        self.assertEqual(len(events), 1)
        self.assertFalse(events[0]["blocking"])
        self.assertEqual(events[0]["checkpoint_unit_index"], 0)
        self.assertIn('"objective_actions"', provider.await_args_list[1].args[2])


if __name__ == "__main__":
    unittest.main()
