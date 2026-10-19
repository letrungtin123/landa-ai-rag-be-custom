"""PRD-2 typed responses (STD-3) must serialize to exactly the JSON the routes returned before.

FastAPI validates a route's return value against its response model and dumps it in JSON mode.
These cases feed each model the dict shape its service returns (including branches the route
snapshots do not reach, such as a handled indexing failure) and require the identical JSON
document back: same keys, same order, same values.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

from fastapi.routing import APIRoute
from pydantic import TypeAdapter, ValidationError

from app.main import app
from app.schemas.chat import ChatResponse
from app.schemas.common import AiUsage
from app.schemas.health import HealthResponse, ServiceMetaResponse
from app.schemas.kb import INDEX_DOCUMENT_RESPONSE, DeleteResponse, IndexDocumentResponse
from app.schemas.orchestration_v2 import (
    ChapterShardV2Response,
    CourseSkeletonV2Response,
    SourceSnapshotV2Response,
    UnitV2Response,
)

USAGE = AiUsage(inputTokens=3, outputTokens=2, embeddingTokens=1, totalTokens=6).model_dump()


def wire(adapter: TypeAdapter[Any], value: dict[str, Any]) -> str:
    """The JSON text FastAPI produces for ``value`` under ``adapter``'s response model."""
    return json.dumps(adapter.dump_python(adapter.validate_python(value), mode="json", by_alias=True))


CASES: list[tuple[str, TypeAdapter[Any], dict[str, Any]]] = [
    ("healthz", TypeAdapter(HealthResponse), {"status": "ok"}),
    ("meta", TypeAdapter(ServiceMetaResponse), {
        "service": "landa-ai-rag", "build_sha": "unknown", "api_version": "0.1.0",
        "contracts": {"idm_contract_version": 1, "orchestration_v2_contract_version": 2, "x": "v-1"},
        "capabilities": {"index_source_download_url": True, "legacy_storage_service_key": False},
        "database": {"state": "idle"},
        "schema_check": {"status": "ok", "code": None, "missing_count": 0, "missing": [], "checked_at": 1.5},
        "prompt_templates_hash": "c" * 64,
    }),
    ("index_learned", INDEX_DOCUMENT_RESPONSE, {
        "status": "learned", "chunk_count": 2, "structure_source": None, "structure_confidence": 0,
        "structure_node_count": 0, "diagnostics": {"sections": 1, "ratio": 0.5, "kinds": ["text"]}, "usage": USAGE,
    }),
    ("index_failed", INDEX_DOCUMENT_RESPONSE, {
        "status": "error", "chunk_count": 0, "usage": AiUsage().model_dump(), "error_reason": "INDEX_DOCUMENT_FAILED",
    }),
    ("delete", TypeAdapter(DeleteResponse), {"deleted": True}),
    ("chat", TypeAdapter(ChatResponse), {
        "text": "answer", "usage": USAGE,
        "sources": [{"document_id": "d", "page": 3, "score": 0.75, "section": None}],
        "retrieval": {"candidate_count": 4, "used": [1, 2], "notes": None},
    }),
    ("source_snapshot", TypeAdapter(SourceSnapshotV2Response), {
        "contract_version": 2, "source_snapshot_hash": "a" * 64, "source_revision": "b" * 64,
        "source_authority": {"mode": "locked", "confidence": 1.0, "chapters": []},
        "facts": [{"fact_key": "f1", "page": None}], "next_cursor": {"document_id": "d", "chunk_no": 3},
        "has_more": True, "page_content_bytes": 120, "usage": AiUsage().model_dump(),
    }),
]


# Step 2.6: the generation answers, fed the bodies the route snapshots recorded (every IDM and legacy branch there).
ROUTE_SNAPSHOTS = Path(__file__).resolve().parents[1] / "snapshots" / "routes"
GENERATION_RESPONSES: dict[str, type[Any]] = {
    "post_v2_course_skeleton_idm": CourseSkeletonV2Response,
    "post_v2_course_skeleton_legacy": CourseSkeletonV2Response,
    "post_v2_course_skeleton_legacy_fallback": CourseSkeletonV2Response,
    "post_v2_chapter_shard_idm": ChapterShardV2Response,
    "post_v2_chapter_shard_legacy": ChapterShardV2Response,
    "post_v2_unit_idm": UnitV2Response,
    "post_v2_unit_idm_fallback_only": UnitV2Response,
    "post_v2_unit_legacy_fallback_only": UnitV2Response,
    "post_v2_unit_legacy_invalid_provider": UnitV2Response,
}


def snapshot_body(name: str) -> dict[str, Any]:
    case = json.loads((ROUTE_SNAPSHOTS / f"{name}.json").read_text(encoding="utf-8"))
    body: dict[str, Any] = case["expected"]["body"]
    return body


CASES += [(name, TypeAdapter(model), snapshot_body(name)) for name, model in GENERATION_RESPONSES.items()]
# Semantic review on: the unit answer gains a trailing ``semantic_review`` summary.
CASES.append(("unit_semantic_review", TypeAdapter(UnitV2Response), {
    **snapshot_body("post_v2_unit_legacy_fallback_only"),
    "semantic_review": {"mode": "observe", "config_hash": "d" * 64, "findings": [], "score": 0.5, "x": None},
}))


class ResponseModelTests(unittest.TestCase):
    def test_response_models_reproduce_the_service_json(self) -> None:
        for name, adapter, value in CASES:
            with self.subTest(case=name):
                self.assertEqual(wire(adapter, value), json.dumps(value))

    def test_response_models_reject_unknown_keys(self) -> None:
        # extra="forbid": a new service key must be added to the model, never silently dropped.
        for name, adapter, value in CASES:
            with self.subTest(case=name), self.assertRaises(ValidationError):
                adapter.validate_python({**value, "unexpected": 1})

    def test_typed_routes_declare_their_response_models(self) -> None:
        models = {route.path: route.response_model for route in app.routes if isinstance(route, APIRoute)}
        self.assertEqual(models["/healthz"], HealthResponse)
        self.assertEqual(models["/v1/meta"], ServiceMetaResponse)
        self.assertEqual(models["/v1/kb/documents/index"], IndexDocumentResponse)
        self.assertEqual(models["/v1/kb/documents/delete"], DeleteResponse)
        self.assertEqual(models["/v1/kb/delete"], DeleteResponse)
        self.assertEqual(models["/v1/chat"], ChatResponse)
        self.assertEqual(models["/v1/lesson-author/orchestration-v2/source-snapshot"], SourceSnapshotV2Response)
        # The generation routes answer the unvalidated service dict; their models are observed after the answer.
        for name in ("course-skeleton", "chapter-shard", "unit"):
            self.assertEqual(models[f"/v1/lesson-author/orchestration-v2/{name}"], dict[str, Any])

    def test_generation_responses_keep_the_untyped_wire_bytes(self) -> None:
        # Before step 2.6 these routes declared ``dict[str, Any]``, so FastAPI encoded the service dict with that
        # adapter's dump_json. The typed model must produce the same bytes (key order, floats, escapes).
        untyped: TypeAdapter[dict[str, Any]] = TypeAdapter(dict[str, Any])
        floats = {"attempt_trace": [{"score": 1e-05, "big": 1e16, "text": "Bước " + chr(0x2028) + " \"x\" </script>"}]}
        bodies = [(name, model, snapshot_body(name)) for name, model in GENERATION_RESPONSES.items()]
        bodies.append(("floats", UnitV2Response, {**snapshot_body("post_v2_unit_idm"), **floats}))
        for name, model, body in bodies:
            with self.subTest(case=name):
                typed: TypeAdapter[Any] = TypeAdapter(model)
                self.assertEqual(typed.dump_json(typed.validate_python(body), by_alias=True),
                                 untyped.dump_json(untyped.validate_python(body), by_alias=True))

    def test_generation_responses_keep_absent_add_ons_absent(self) -> None:
        skeleton = CourseSkeletonV2Response.model_validate(snapshot_body("post_v2_course_skeleton_legacy"))
        self.assertNotIn("idm", skeleton.model_dump(mode="json"))
        shard = ChapterShardV2Response.model_validate(snapshot_body("post_v2_chapter_shard_legacy"))
        self.assertNotIn("idm_design", shard.model_dump(mode="json")["shard"])
        unit = UnitV2Response.model_validate(snapshot_body("post_v2_unit_idm"))
        self.assertNotIn("semantic_review", unit.model_dump(mode="json"))


if __name__ == "__main__":
    unittest.main()
