from __future__ import annotations

import unittest

from tests.cp3a_source_benchmark import (
    _anchor_matches,
    _real_held_out_document_count,
    _visual_observation_metrics,
)


class Cp3aSourceBenchmarkTests(unittest.TestCase):
    def test_anchor_matching_tolerates_layout_punctuation_without_reordering_tokens(self) -> None:
        candidate = (
            "01 Năng Lực Sẽ Trở Thành Best-in- "
            "................................ Class:"
        )

        self.assertTrue(_anchor_matches(
            "01 Năng Lực Sẽ Trở Thành Best-in-Class",
            candidate,
        ))
        self.assertFalse(_anchor_matches("Class Best-in", candidate))

    def test_real_held_out_count_is_derived_from_document_split(self) -> None:
        oracle = {
            "documents": [
                {"split": "development_regression"},
                {"split": "held_out_document"},
                {"split": "held_out_document"},
            ],
            "held_out_document_requirement": {"current_real_document_count": 99},
        }

        self.assertEqual(_real_held_out_document_count(oracle), 2)

    def test_visual_metrics_require_task_and_human_observation_without_provider(self) -> None:
        labels = [{
            "page": 3,
            "critical_elements": [{"kind": "relational_vector_visual", "count": 1}],
            "reviewed_visual_facts": ["Three connected boxes are visible."],
        }]
        readiness = {
            "source": {"sha256": "a" * 64},
            "pages": [{
                "page": 3,
                "visual_regions": [{
                    "region_kind": "vector_composition",
                    "asset_revision": "b" * 64,
                    "locator": {
                        "page": 3,
                        "bbox_normalized": [0.1, 0.2, 0.8, 0.9],
                    },
                }],
            }],
        }

        metrics = _visual_observation_metrics(labels, readiness)

        self.assertEqual(metrics["task_page_recall"], 1.0)
        self.assertEqual(metrics["observation_page_recall"], 1.0)
        self.assertEqual(metrics["provider_call_count"], 0)
        self.assertTrue(metrics["non_blocking"])


if __name__ == "__main__":
    unittest.main()
