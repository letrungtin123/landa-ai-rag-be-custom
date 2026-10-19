"""Usage counters of the legacy AI ID V1 routes (audit F-05): one count and one log event per legacy call."""

from __future__ import annotations

import json
import logging
import unittest
from pathlib import Path

from fastapi.routing import APIRoute

from app.core.logging import SERVICE_LOGGER_NAME
from app.core.metrics import REGISTRY
from app.main import app
from app.services import legacy_routes
from app.services.legacy_routes import LEGACY_ROUTES, record_legacy_route_call
from tests.log_support import logged_events
from tests.test_route_snapshots import replay

SNAPSHOTS = Path(__file__).resolve().parent / "snapshots" / "routes"
METRIC = "ai_rag_legacy_route_calls_total"
# Route snapshot case -> the legacy route it counts (None: a current route or an IDM request, never counted).
CASES: dict[str, str | None] = {
    "post_proposal_single_json": legacy_routes.PROPOSAL_ROUTE,
    "post_blueprint_v5_valid": legacy_routes.BLUEPRINT_ROUTE,
    "post_chapter_checkpoint_generate_unit": legacy_routes.CHAPTER_CHECKPOINT_ROUTE,
    "post_v2_course_skeleton_legacy": legacy_routes.COURSE_SKELETON_ROUTE,
    "post_v2_course_skeleton_idm": None,
    "post_v2_chapter_shard_legacy": legacy_routes.CHAPTER_SHARD_ROUTE,
    "post_v2_chapter_shard_idm": None,
    "post_v2_unit_legacy_fallback_only": legacy_routes.UNIT_ROUTE,
    "post_v2_unit_idm": None,
    "post_v2_source_snapshot_page": None,
    "post_chat_answer": None,
}


def counts() -> dict[str, float]:
    return {route: REGISTRY.get_sample_value(METRIC, {"route": route}) or 0.0 for route in LEGACY_ROUTES}


class LegacyRouteUsageTests(unittest.TestCase):
    def test_each_legacy_call_counts_once_and_logs_only_its_route(self) -> None:
        for name, route in CASES.items():
            with self.subTest(case=name):
                case = json.loads((SNAPSHOTS / f"{name}.json").read_text(encoding="utf-8"))
                before = counts()
                with self.assertLogs(SERVICE_LOGGER_NAME, level=logging.INFO) as captured:
                    logging.getLogger(SERVICE_LOGGER_NAME).info("marker")
                    replay(case)
                after = counts()
                expected = {key: value + (1 if key == route else 0) for key, value in before.items()}
                self.assertEqual(after, expected)
                events = logged_events(captured.records, "legacy_route_called")
                self.assertEqual([event.get("route") for event in events], [route] if route else [])
                for event in events:
                    self.assertLessEqual(set(event) - {"event", "message"}, {"route"})

    def test_rejected_requests_are_not_counted(self) -> None:
        case = json.loads((SNAPSHOTS / "post_v2_course_skeleton_legacy.json").read_text(encoding="utf-8"))
        case["request"]["json"].pop("source_snapshot_hash")
        before = counts()
        self.assertEqual(replay(case)["status"], 422)
        self.assertEqual(counts(), before)

    def test_labels_are_registered_route_templates(self) -> None:
        paths = {route.path for route in app.routes if isinstance(route, APIRoute)}
        self.assertLessEqual(LEGACY_ROUTES, paths)
        self.assertEqual(len(LEGACY_ROUTES), 6)
        with self.assertRaises(ValueError):
            record_legacy_route_call("/v1/chat")


if __name__ == "__main__":
    unittest.main()
