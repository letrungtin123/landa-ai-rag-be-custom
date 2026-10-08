from __future__ import annotations

import json
import unittest
from pathlib import Path

from app.instructional_density import (
    DEFAULT_INSTRUCTIONAL_DENSITY_BUDGET,
    INSTRUCTIONAL_DENSITY_POLICY_VERSION,
    InstructionalDensityBudget,
    InstructionalDensityProfile,
    evaluate_instructional_density,
    instructional_output_budget,
    partition_chunk_facts_for_density,
    profile_instructional_texts,
)
from app.lesson_author_orchestration_v2_provider import MAX_UNIT_SOURCE_FACTS

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "hse_instructional_density_fixture.json"


class InstructionalDensityContractTests(unittest.TestCase):
    def test_hse_regression_requires_multiple_instructional_units(self) -> None:
        fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

        decision = evaluate_instructional_density(
            InstructionalDensityProfile(**fixture["profile"]),
        )

        self.assertEqual(decision.policy_version, INSTRUCTIONAL_DENSITY_POLICY_VERSION)
        self.assertEqual(decision.split_required, fixture["expected"]["split_required"])
        self.assertEqual(decision.recommended_unit_count, fixture["expected"]["recommended_unit_count"])
        self.assertEqual(list(decision.reason_codes), fixture["expected"]["reason_codes"])

    def test_balanced_unit_stays_intact(self) -> None:
        decision = evaluate_instructional_density(InstructionalDensityProfile(
            canonical_fact_count=24,
            source_chars=6_000,
            estimated_words=900,
            learning_objective_count=2,
            independent_topic_count=1,
        ))

        self.assertFalse(decision.split_required)
        self.assertEqual(decision.recommended_unit_count, 1)
        self.assertEqual(decision.reason_codes, ())

    def test_largest_overage_determines_stable_minimum_split_count(self) -> None:
        decision = evaluate_instructional_density(InstructionalDensityProfile(
            canonical_fact_count=97,
            source_chars=12_001,
            estimated_words=1_600,
            learning_objective_count=3,
            independent_topic_count=2,
        ))

        self.assertEqual(decision.recommended_unit_count, 3)
        self.assertEqual(decision.reason_codes, (
            "UNIT_FACT_DENSITY_EXCEEDED",
            "UNIT_SOURCE_TEXT_DENSITY_EXCEEDED",
        ))

    def test_provider_transport_capacity_is_not_a_pedagogical_budget(self) -> None:
        self.assertGreater(
            MAX_UNIT_SOURCE_FACTS,
            DEFAULT_INSTRUCTIONAL_DENSITY_BUDGET.max_canonical_facts,
        )

    def test_output_budget_has_safe_floor_and_hard_ceiling(self) -> None:
        sparse = instructional_output_budget(InstructionalDensityProfile(
            canonical_fact_count=1,
            source_chars=80,
            estimated_words=12,
            learning_objective_count=1,
            independent_topic_count=1,
        ))
        dense = instructional_output_budget(InstructionalDensityProfile(
            canonical_fact_count=48,
            source_chars=12_000,
            estimated_words=1_600,
            learning_objective_count=3,
            independent_topic_count=2,
        ))

        self.assertEqual((sparse.max_visible_chars, sparse.max_words), (4_000, 600))
        self.assertEqual((dense.max_visible_chars, dense.max_words), (18_000, 2_400))

    def test_invalid_profiles_and_budgets_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            InstructionalDensityProfile(
                canonical_fact_count=-1,
                source_chars=0,
                estimated_words=0,
                learning_objective_count=0,
                independent_topic_count=0,
            )
        with self.assertRaisesRegex(ValueError, "must be positive"):
            InstructionalDensityBudget(max_canonical_facts=0)

    def test_four_paginated_chunks_merge_into_density_safe_scope_lanes(self) -> None:
        chunks = [
            [f"Chunk {chunk_index} canonical fact {fact_index}." for fact_index in range(30)]
            for chunk_index in range(4)
        ]
        partitioned = [partition_chunk_facts_for_density(chunk) for chunk in chunks]

        self.assertTrue(all(
            [fact for lane in chunk_lanes for fact in lane] == original
            for chunk_lanes, original in zip(partitioned, chunks)
        ))
        for lane_index in range(max(len(chunk_lanes) for chunk_lanes in partitioned)):
            merged = [
                fact
                for chunk_lanes in partitioned
                if lane_index < len(chunk_lanes)
                for fact in chunk_lanes[lane_index]
            ]
            decision = evaluate_instructional_density(profile_instructional_texts(
                merged,
                learning_objective_count=1,
                independent_topic_count=1,
            ))
            self.assertFalse(decision.split_required)


if __name__ == "__main__":
    unittest.main()
