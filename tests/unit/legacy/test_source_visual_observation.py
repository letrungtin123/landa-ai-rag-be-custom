from __future__ import annotations

import unittest

from app.source_visual_observation import (
    SHADOW_VISUAL_PIPELINE_VERSION,
    build_shadow_visual_observation_plan,
    build_visual_cache_key,
    record_human_visual_observation,
)


def _readiness(*, regions: list[dict[str, object]]) -> dict[str, object]:
    return {
        "source": {"sha256": "a" * 64},
        "pages": [{"page": 4, "visual_regions": regions}],
    }


def _region(*, asset: str = "b" * 64) -> dict[str, object]:
    return {
        "region_kind": "vector_composition",
        "asset_revision": asset,
        "locator": {
            "page": 4,
            "bbox_normalized": [0.1, 0.2, 0.8, 0.9],
        },
    }


class SourceVisualObservationTests(unittest.TestCase):
    def test_shadow_plan_is_non_blocking_and_never_calls_provider(self) -> None:
        plan = build_shadow_visual_observation_plan(
            _readiness(regions=[_region()]),
            tenant_scope="tenant-a",
        )

        self.assertEqual(plan["pipeline_version"], SHADOW_VISUAL_PIPELINE_VERSION)
        self.assertFalse(plan["blocking"])
        self.assertEqual(plan["draft_visibility"], "preserved")
        self.assertEqual(plan["provider_call_count"], 0)
        self.assertEqual(plan["scheduled_region_count"], 1)
        task = plan["tasks"][0]
        self.assertEqual(task["provider"]["policy"], "disabled")
        self.assertFalse(task["provider"]["called"])
        self.assertNotIn("tenant-a", str(plan))

    def test_cache_key_is_tenant_source_asset_locator_and_config_scoped(self) -> None:
        values = {
            "source_revision": "a" * 64,
            "asset_revision": "b" * 64,
            "locator": {"page": 4, "bbox_normalized": [0.1, 0.2, 0.8, 0.9]},
            "locale": "vi",
            "observer_config_version": "human-review-v1",
        }
        first = build_visual_cache_key(tenant_scope="tenant-a", **values)
        replay = build_visual_cache_key(tenant_scope="tenant-a", **values)
        other_tenant = build_visual_cache_key(tenant_scope="tenant-b", **values)
        other_config = build_visual_cache_key(
            tenant_scope="tenant-a",
            **{**values, "observer_config_version": "human-review-v2"},
        )

        self.assertEqual(first, replay)
        self.assertNotEqual(first["cache_key"], other_tenant["cache_key"])
        self.assertNotEqual(first["cache_key"], other_config["cache_key"])

    def test_invalid_region_degrades_to_review_without_hiding_draft(self) -> None:
        plan = build_shadow_visual_observation_plan(
            _readiness(regions=[_region(asset="invalid")]),
            tenant_scope="tenant-a",
        )

        self.assertFalse(plan["blocking"])
        self.assertEqual(plan["draft_visibility"], "preserved")
        self.assertEqual(plan["provider_call_count"], 0)
        self.assertEqual(plan["tasks"], [])
        self.assertEqual(plan["diagnostics"][0]["code"], "VISUAL_REGION_INVALID")

    def test_region_limit_is_bounded_and_reported(self) -> None:
        regions = [
            {
                **_region(asset=f"{index + 1:064x}"),
                "locator": {
                    "page": 4,
                    "bbox_normalized": [0.1, 0.1 + index * 0.1, 0.8, 0.15 + index * 0.1],
                },
            }
            for index in range(3)
        ]
        plan = build_shadow_visual_observation_plan(
            _readiness(regions=regions),
            tenant_scope="tenant-a",
            max_regions=2,
        )

        self.assertEqual(plan["scheduled_region_count"], 2)
        self.assertEqual(plan["total_region_count"], 3)
        self.assertEqual(plan["diagnostics"][0]["code"], "VISUAL_REGION_LIMIT_REACHED")

    def test_human_observation_keeps_fact_and_inference_separate(self) -> None:
        plan = build_shadow_visual_observation_plan(
            _readiness(regions=[_region()]),
            tenant_scope="tenant-a",
        )
        receipt = record_human_visual_observation(
            plan["tasks"][0],
            observed_facts=["A three by three grid is visible."],
            inference_claims=["The grid may be intended as a learner canvas."],
            observer_role="reviewer",
        )

        self.assertFalse(receipt["blocking"])
        self.assertEqual(receipt["provider_call_count"], 0)
        self.assertEqual(receipt["record"]["observation"]["status"], "observed")
        self.assertEqual(
            receipt["record"]["inference"]["status"],
            "requires_verification",
        )


if __name__ == "__main__":
    unittest.main()
