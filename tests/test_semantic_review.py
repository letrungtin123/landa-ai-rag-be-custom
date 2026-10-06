"""Provider-free CP4 contract, repair and evaluation regressions."""

import asyncio
from copy import deepcopy
import json
from pathlib import Path
import unittest

from app.semantic_review import (
    SemanticReviewResponse,
    build_semantic_review_prompt,
    evaluate_semantic_review_predictions,
    run_bounded_semantic_review,
    safe_semantic_review_summary,
    semantic_review_config_hash,
    validate_semantic_review_response,
)


FIXTURE = Path(__file__).parent / "fixtures" / "ai_id_cp4_semantic_review_benchmark.json"


def finding(code="EVIDENCE_CONTRADICTION", severity="critical", index=0):
    return {
        "criterion": "evidence_fidelity",
        "code": code,
        "severity": severity,
        "scope": "component",
        "component_index": index,
        "candidate_path": f"components[{index}].semantic_content.sections[0]",
        "source_fact_ids": ["f1"],
        "witness_summary": "The candidate changes a required source condition.",
        "repair_instruction": "Restore the source condition in this component only.",
    }


def review(verdict="review_required", findings=None, indices=None):
    return SemanticReviewResponse.model_validate({
        "contract_version": "semantic-review-v1",
        "verdict": verdict,
        "reviewed_component_indices": [0] if indices is None else list(indices),
        "findings": [finding()] if findings is None else findings,
    })


def unit():
    return {
        "title": "Risk controls",
        "source_fact_ids": ["f1"],
        "supporting_evidence_fact_ids": [],
        "components": [{
            "type": "html", "component_plan_id": "cp1", "source_fact_ids": ["f1"],
            "covered_source_fact_ids": ["f1"], "supporting_evidence_fact_ids": [],
            "semantic_content": {"version": 2, "sections": [{"heading": "Controls", "learning_block_ids": [],
                "blocks": [{"kind": "paragraph", "text": "Unsafe draft."}]}]},
        }],
    }


class SemanticReviewContractTests(unittest.TestCase):
    def test_response_scope_and_authority_are_strict(self):
        accepted = validate_semantic_review_response(
            review(), component_count=1, allowed_source_fact_ids={"f1"}, requested_component_indices=(0,),
        )
        self.assertEqual(accepted.verdict, "review_required")
        bad = review().model_dump(mode="json")
        bad["findings"][0]["source_fact_ids"] = ["invented"]
        with self.assertRaisesRegex(ValueError, "authority"):
            validate_semantic_review_response(
                bad, component_count=1, allowed_source_fact_ids={"f1"}, requested_component_indices=(0,),
            )
        bad = review().model_dump(mode="json")
        bad["reviewed_component_indices"] = [1]
        with self.assertRaises(ValueError):
            validate_semantic_review_response(
                bad, component_count=2, allowed_source_fact_ids={"f1"}, requested_component_indices=(0,),
            )
        partial_unit_finding = review("review_required", [{
            **finding(code="PREREQUISITE_GAP", severity="major"),
            "criterion": "instructional_alignment", "scope": "unit",
            "component_index": None, "candidate_path": "unit",
        }], (0,))
        with self.assertRaisesRegex(ValueError, "authority"):
            validate_semantic_review_response(
                partial_unit_finding, component_count=2, allowed_source_fact_ids={"f1"},
                requested_component_indices=(0,),
            )

    def test_pass_cannot_hide_serious_finding(self):
        payload = review().model_dump(mode="json")
        payload["verdict"] = "pass"
        with self.assertRaisesRegex(ValueError, "verdict"):
            SemanticReviewResponse.model_validate(payload)

    def test_prompt_keeps_deterministic_authority_separate(self):
        prompt = build_semantic_review_prompt(
            unit=unit(),
            expected={"unit_title": "Risk controls", "source_fact_ids": ["f1"],
                      "component_plan": [{"type": "html", "source_fact_ids": ["f1"]}]},
            manifest={"facts": [{"fact_id": "f1", "text": "Controls are required."}]},
            component_indices=(0,), locale="en",
        )
        self.assertIn("Deterministic validation is authoritative", prompt)
        self.assertIn("Do not redesign the course", prompt)
        self.assertIn('"component_index":0', prompt)

    def test_review_pass_does_not_call_repair(self):
        calls = []

        async def reviewer(candidate, indices):
            calls.append(("review", indices))
            return review("pass", [], indices)

        async def repairer(*_args):
            calls.append(("repair",))
            return unit()

        outcome = asyncio.run(run_bounded_semantic_review(
            unit(), reviewer=reviewer, repairer=repairer,
            deterministic_validate=lambda _candidate: None, allow_repair=True,
        ))
        self.assertEqual(outcome.quality_state, "validated")
        self.assertEqual(calls, [("review", (0,))])

    def test_exactly_one_scoped_repair_and_affected_rereview(self):
        baseline = unit()
        calls = []

        async def reviewer(candidate, indices):
            calls.append(("review", indices, candidate["components"][0]["semantic_content"]["sections"][0]["blocks"][0]["text"]))
            return review("review_required", [finding()], indices) if len(calls) == 1 else review("pass", [], indices)

        async def repairer(candidate, indices, findings):
            calls.append(("repair", indices, tuple(item.code for item in findings)))
            candidate["components"][0]["semantic_content"]["sections"][0]["blocks"][0]["text"] = "Controls are required."
            return candidate

        outcome = asyncio.run(run_bounded_semantic_review(
            baseline, reviewer=reviewer, repairer=repairer,
            deterministic_validate=lambda _candidate: None, allow_repair=True,
        ))
        self.assertTrue(outcome.repair_attempted)
        self.assertTrue(outcome.repair_applied)
        self.assertEqual(outcome.repair_component_indices, (0,))
        self.assertEqual(outcome.quality_state, "validated")
        self.assertEqual([call[0] for call in calls], ["review", "repair", "review"])
        self.assertEqual(baseline, unit())

    def test_failed_or_no_progress_repair_preserves_visible_baseline(self):
        baseline = unit()
        original = deepcopy(baseline)
        reviews = 0

        async def reviewer(_candidate, indices):
            nonlocal reviews
            reviews += 1
            return review("review_required", [finding()], indices)

        async def repairer(candidate, _indices, _findings):
            candidate["components"][0]["semantic_content"]["sections"][0]["blocks"][0]["text"] = "Still unsafe."
            return candidate

        outcome = asyncio.run(run_bounded_semantic_review(
            baseline, reviewer=reviewer, repairer=repairer,
            deterministic_validate=lambda _candidate: None, allow_repair=True,
        ))
        self.assertEqual(reviews, 2)
        self.assertEqual(outcome.unit, original)
        self.assertFalse(outcome.repair_applied)
        self.assertEqual(outcome.failure_code, "SEMANTIC_REPAIR_NO_PROGRESS")

    def test_repair_cannot_trade_one_critical_for_multiple_major_findings(self):
        baseline = unit()
        reviews = 0

        async def reviewer(_candidate, indices):
            nonlocal reviews
            reviews += 1
            if reviews == 1:
                return review("review_required", [finding()], indices)
            return review("review_required", [
                finding(code="EVIDENCE_CONDITION_OMITTED", severity="major"),
                finding(code="REQUIRED_EVIDENCE_OMITTED", severity="major"),
            ], indices)

        async def repairer(candidate, _indices, _findings):
            candidate["components"][0]["semantic_content"]["sections"][0]["blocks"][0]["text"] = "Changed."
            return candidate

        outcome = asyncio.run(run_bounded_semantic_review(
            baseline, reviewer=reviewer, repairer=repairer,
            deterministic_validate=lambda _candidate: None, allow_repair=True,
        ))
        self.assertEqual(outcome.unit, baseline)
        self.assertFalse(outcome.repair_applied)
        self.assertEqual(outcome.failure_code, "SEMANTIC_REPAIR_NO_PROGRESS")

    def test_repaired_summary_keeps_unresolved_unit_findings(self):
        baseline = unit()
        reviews = 0
        unit_finding = {
            **finding(code="PREREQUISITE_GAP", severity="major"),
            "criterion": "instructional_alignment", "scope": "unit",
            "component_index": None, "candidate_path": "unit",
        }

        async def reviewer(_candidate, indices):
            nonlocal reviews
            reviews += 1
            return (review("review_required", [finding(), unit_finding], indices)
                    if reviews == 1 else review("pass", [], indices))

        async def repairer(candidate, _indices, _findings):
            candidate["components"][0]["semantic_content"]["sections"][0]["blocks"][0]["text"] = "Fixed."
            return candidate

        outcome = asyncio.run(run_bounded_semantic_review(
            baseline, reviewer=reviewer, repairer=repairer,
            deterministic_validate=lambda _candidate: None, allow_repair=True,
        ))
        summary = safe_semantic_review_summary(
            outcome, config_hash=semantic_review_config_hash(mode="repair", model="frozen-model"),
        )
        self.assertTrue(outcome.repair_applied)
        self.assertEqual(outcome.quality_state, "review_required")
        self.assertEqual(summary["finding_counts"]["major"], 1)
        self.assertEqual(summary["findings"][0]["scope"], "unit")

    def test_reviewer_failure_preserves_draft_and_reports_hashes_only(self):
        baseline = unit()

        async def reviewer(_candidate, _indices):
            raise RuntimeError("PRIVATE_SOURCE_TEXT")

        async def repairer(*_args):
            raise AssertionError("repair must not run")

        outcome = asyncio.run(run_bounded_semantic_review(
            baseline, reviewer=reviewer, repairer=repairer,
            deterministic_validate=lambda _candidate: None, allow_repair=True,
        ))
        summary = safe_semantic_review_summary(
            outcome, config_hash=semantic_review_config_hash(mode="repair", model="frozen-model"),
        )
        self.assertEqual(outcome.unit, baseline)
        self.assertEqual(outcome.quality_state, "review_required")
        self.assertEqual(summary["failure_code"], "SEMANTIC_REVIEW_UNAVAILABLE")
        self.assertNotIn("PRIVATE_SOURCE_TEXT", json.dumps(summary))

    def test_repair_disabled_keeps_draft_reviewable(self):
        repair_called = False

        async def reviewer(_candidate, indices):
            return review("review_required", [finding()], indices)

        async def repairer(*_args):
            nonlocal repair_called
            repair_called = True
            return unit()

        outcome = asyncio.run(run_bounded_semantic_review(
            unit(), reviewer=reviewer, repairer=repairer,
            deterministic_validate=lambda _candidate: None, allow_repair=False,
        ))
        self.assertFalse(repair_called)
        self.assertFalse(outcome.repair_attempted)
        self.assertEqual(outcome.quality_state, "review_required")

    def test_frozen_benchmark_reports_false_pass_false_fail_without_release_claim(self):
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        before = evaluate_semantic_review_predictions(fixture["before_cases"])
        after = evaluate_semantic_review_predictions(fixture["after_cases"])
        self.assertEqual(before["serious_false_pass_rate"], 1.0)
        self.assertEqual(after["serious_false_pass_rate"], 0.0)
        self.assertEqual(after["acceptable_false_fail_rate"], 0.0)
        self.assertEqual(after["document_group_count"], 4)
        self.assertEqual(after["course_group_count"], 4)
        self.assertFalse(after["all_labels_human_reviewed"])
        self.assertFalse(after["release_claim_eligible"])


if __name__ == "__main__":
    unittest.main()
