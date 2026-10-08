"""PRD-2 typed responses (STD-3) must serialize to exactly the JSON the routes returned before.

FastAPI validates a route's return value against its response model and dumps it in JSON mode.
These cases feed each model the dict shape its service returns (including branches the route
snapshots do not reach, such as a handled indexing failure) and require the identical JSON
document back: same keys, same order, same values.
"""

from __future__ import annotations

import json
import unittest
from typing import Any

from fastapi.routing import APIRoute
from pydantic import TypeAdapter, ValidationError

from app.main import app
from app.schemas.chat import ChatResponse
from app.schemas.common import AiUsage
from app.schemas.health import HealthResponse, ServiceMetaResponse
from app.schemas.kb import INDEX_DOCUMENT_RESPONSE, DeleteResponse, IndexDocumentResponse
from app.schemas.orchestration_v2 import SourceSnapshotV2Response

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


if __name__ == "__main__":
    unittest.main()
