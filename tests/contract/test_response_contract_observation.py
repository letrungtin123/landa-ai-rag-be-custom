"""The IDM generation answers are never re-validated on the response path (behaviour unchanged since 2.6).

A service answer that does not fit its published model still goes out unchanged (200, the bytes of the
``dict[str, Any]`` encoder); the mismatch is only counted and logged with field locations after the answer.
"""

from __future__ import annotations

import copy
import json
import logging
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
from pydantic import BaseModel, ConfigDict, TypeAdapter

from app.core.logging import SERVICE_LOGGER_NAME
from app.core.metrics import REGISTRY
from app.schemas.orchestration_v2 import CourseSkeletonV2Response
from app.services.legacy_routes import CHAPTER_SHARD_ROUTE, COURSE_SKELETON_ROUTE, UNIT_ROUTE
from app.services.orchestration_v2 import chapter_shard, course_skeleton, unit
from app.services.response_contracts import OBSERVED_ROUTES, OTHER_ROUTE, observe_response_contract
from tests.contract.test_route_snapshots import replay
from tests.support.log_support import logged_events

SNAPSHOTS = Path(__file__).resolve().parents[1] / "snapshots" / "routes"
METRIC = "ai_rag_response_contract_mismatch_total"
UNTYPED: TypeAdapter[dict[str, Any]] = TypeAdapter(dict[str, Any])
# Route -> (a recorded case of it, the service function its handler awaits).
ROUTES: dict[str, tuple[str, Any, str]] = {
    COURSE_SKELETON_ROUTE: ("post_v2_course_skeleton_legacy", course_skeleton,
                            "lesson_author_orchestration_v2_course_skeleton"),
    CHAPTER_SHARD_ROUTE: ("post_v2_chapter_shard_legacy", chapter_shard,
                          "lesson_author_orchestration_v2_chapter_shard"),
    UNIT_ROUTE: ("post_v2_unit_legacy_fallback_only", unit, "lesson_author_orchestration_v2_unit"),
}


@contextmanager
def raw_responses() -> Iterator[list[httpx.Response]]:
    """The raw responses the snapshot harness receives (its own result is parsed JSON)."""

    sent: list[httpx.Response] = []
    original = httpx.AsyncClient.request

    async def request(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> httpx.Response:
        response = await original(self, *args, **kwargs)
        sent.append(response)
        return response

    with patch.object(httpx.AsyncClient, "request", request):
        yield sent


def mismatches(route: str) -> float:
    return REGISTRY.get_sample_value(METRIC, {"route": route}) or 0.0


def case(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((SNAPSHOTS / f"{name}.json").read_text(encoding="utf-8"))
    return loaded


def invalid(body: dict[str, Any]) -> dict[str, Any]:
    # Out of the published contract: an unknown provenance value and an unknown top-level key.
    return {**copy.deepcopy(body), "usage_source": "something_new", "unexpected": {"secret": "value"}}


class ResponseContractObservationTests(unittest.TestCase):
    def test_a_payload_outside_the_contract_is_sent_unchanged_and_counted(self) -> None:
        for route, (name, module, function) in ROUTES.items():
            with self.subTest(route=route):
                recorded = case(name)
                payload = invalid(recorded["expected"]["body"])

                async def service(*_args: Any, payload: dict[str, Any] = payload) -> dict[str, Any]:
                    return copy.deepcopy(payload)

                before = mismatches(route)
                with patch.object(module, function, service), raw_responses() as sent, \
                        self.assertLogs(SERVICE_LOGGER_NAME, level=logging.WARNING) as logs:
                    result = replay(recorded)
                self.assertEqual(result["status"], 200)
                # The exact bytes the dict[str, Any] encoder gives for the payload: nothing was re-validated.
                self.assertEqual(sent[-1].content, UNTYPED.dump_json(payload))
                self.assertEqual(mismatches(route), before + 1)
                (event,) = logged_events(logs.records, "response_contract_mismatch")
                self.assertEqual(event["route"], route)
                self.assertIn("usage_source", event["locations"])
                self.assertIn("unexpected", event["locations"])
                self.assertNotIn("something_new", json.dumps(event))
                self.assertNotIn("secret", json.dumps(event))

    def test_recorded_answers_fit_their_contract(self) -> None:
        for route, (name, _module, _function) in ROUTES.items():
            with self.subTest(route=route):
                before = mismatches(route)
                self.assertEqual(replay(case(name))["status"], 200)
                self.assertEqual(mismatches(route), before)

    def test_the_check_never_raises_and_keeps_labels_bounded(self) -> None:
        before = mismatches(OTHER_ROUTE)
        with self.assertLogs(SERVICE_LOGGER_NAME, level=logging.WARNING):
            observe_response_contract("/v1/unknown", CourseSkeletonV2Response, None)
        self.assertEqual(mismatches(OTHER_ROUTE), before + 1)

        class Broken(CourseSkeletonV2Response):
            @classmethod
            def model_validate(cls, *_args: Any, **_kwargs: Any) -> Any:
                raise RuntimeError("model bug")

        before = mismatches(UNIT_ROUTE)
        with self.assertLogs(SERVICE_LOGGER_NAME, level=logging.WARNING):
            observe_response_contract(UNIT_ROUTE, Broken, {})
        self.assertEqual(mismatches(UNIT_ROUTE), before + 1)
        self.assertEqual(len(OBSERVED_ROUTES), 3)

    def test_logged_locations_keep_only_indexes_and_plain_identifiers(self) -> None:
        class Probe(BaseModel):
            model_config = ConfigDict(extra="forbid")

            items: list[int]
            notes: dict[str, int]

        longest = "k" * 64
        payload = {
            "items": [1, "x"],
            "notes": {"plain_key": "x", longest: "x", longest + "k": "x", "Ignore previous rules: a@b.example": "x"},
            "unknown text, from the model": 1,
        }
        with self.assertLogs(SERVICE_LOGGER_NAME, level=logging.WARNING) as logs:
            observe_response_contract(UNIT_ROUTE, Probe, payload)
        (event,) = logged_events(logs.records, "response_contract_mismatch")
        self.assertEqual(sorted(event["locations"]),
                         sorted(["items.1", "notes.plain_key", f"notes.{longest}", "notes.<key>", "notes.<key>",
                                 "<key>"]))
        text = json.dumps(event)
        for leaked in ("Ignore", "a@b.example", "unknown text", longest + "k"):
            self.assertNotIn(leaked, text)


if __name__ == "__main__":
    unittest.main()
