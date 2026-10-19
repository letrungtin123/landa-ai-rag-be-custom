"""The public contract surface the Node backend uses (refactor step 2.6, audit F-07).

Two parts:

* The Python helpers the backend's cross-language tests import (``python -c "from app.main import <name>"``).
  They live here so that ``app.main`` is only the application factory; ``app.main`` re-exports these names until
  the backend imports them from this module instead.
* The HTTP contract: every POST route with its top-level request model and, for the typed routes, its response
  model, exported as one JSON Schema document (``contracts/landa-ai-rag.http.schema.json``) that Node can compare
  against. ``tests/test_contract_bridge.py`` checks that the committed file matches these models and the routes.

Regenerate the schema file after a deliberate contract change with ``python -m app.contract_bridge``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, TypeAdapter
from pydantic.json_schema import JsonSchemaMode

from app.schemas.chat import ChatResponse, RagChatRequest
from app.schemas.common import AiUsage
from app.schemas.kb import (
    DeleteResponse,
    IndexDocumentResponse,
    RagDeleteDocumentRequest,
    RagDeleteKbRequest,
    RagIndexRequest,
)
from app.schemas.lesson_author import (
    RagLessonAuthorBlueprintRequest,
    RagLessonAuthorCheckpointRequest,
    RagLessonAuthorRequest,
)
from app.schemas.orchestration_v2 import (
    ChapterShardV2Response,
    CourseSkeletonV2Response,
    RagLessonAuthorChapterShardV2Request,
    RagLessonAuthorCourseSkeletonV2Request,
    RagLessonAuthorSourceSnapshotV2Request,
    RagLessonAuthorUnitV2Request,
    SourceSnapshotV2Response,
    UnitV2Response,
)
from app.services.lesson_author.architecture_validation import validate_v5_instructional_coherence
from app.services.lesson_author.checkpoint import build_lesson_author_checkpoint_result
from app.services.lesson_author.proposal_validation import semantic_learning_visible_text
from app.services.provider import generate_content
from app.services.staged_unit.provider_schemas import staged_component_payload_code

__all__ = [
    "CONTRACT_SCHEMA_PATH",
    "ROUTE_CONTRACTS",
    "AiUsage",
    "RagLessonAuthorCheckpointRequest",
    "RagLessonAuthorCourseSkeletonV2Request",
    "build_lesson_author_checkpoint_result",
    "contract_schema_text",
    "generate_content",
    "semantic_learning_visible_text",
    "staged_component_payload_code",
    "validate_v5_instructional_coherence",
    "write_contract_schema",
]

CONTRACT_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "contracts" / "landa-ai-rag.http.schema.json"
SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"

# path -> (request model, response type or None while the route still answers an untyped object).
# The legacy lesson-author routes (proposal, chapter checkpoint, blueprint) stay untyped until PRD-3 removes them.
ROUTE_CONTRACTS: dict[str, tuple[type[BaseModel], Any]] = {
    "/v1/kb/documents/index": (RagIndexRequest, IndexDocumentResponse),
    "/v1/kb/documents/delete": (RagDeleteDocumentRequest, DeleteResponse),
    "/v1/kb/delete": (RagDeleteKbRequest, DeleteResponse),
    "/v1/chat": (RagChatRequest, ChatResponse),
    "/v1/lesson-author/chapter-checkpoint": (RagLessonAuthorCheckpointRequest, None),
    "/v1/lesson-author/proposal": (RagLessonAuthorRequest, None),
    "/v1/lesson-author/blueprint": (RagLessonAuthorBlueprintRequest, None),
    "/v1/lesson-author/orchestration-v2/source-snapshot": (
        RagLessonAuthorSourceSnapshotV2Request, SourceSnapshotV2Response),
    "/v1/lesson-author/orchestration-v2/course-skeleton": (
        RagLessonAuthorCourseSkeletonV2Request, CourseSkeletonV2Response),
    "/v1/lesson-author/orchestration-v2/chapter-shard": (RagLessonAuthorChapterShardV2Request, ChapterShardV2Response),
    "/v1/lesson-author/orchestration-v2/unit": (RagLessonAuthorUnitV2Request, UnitV2Response),
}


def contract_schema() -> dict[str, Any]:
    """The JSON Schema document of every POST route (requests and typed responses, validation mode).

    All models share one ``$defs``. Optional response fields that a service leaves out are optional properties,
    never ``null``.
    """

    inputs: list[tuple[tuple[str, str], JsonSchemaMode, TypeAdapter[Any]]] = []
    for path, (request_model, response_type) in ROUTE_CONTRACTS.items():
        inputs.append(((path, "request"), "validation", TypeAdapter(request_model)))
        if response_type is not None:
            inputs.append(((path, "response"), "validation", TypeAdapter(response_type)))
    schemas, top_level = TypeAdapter.json_schemas(inputs, ref_template="#/$defs/{model}")
    routes = {
        path: {"request": schemas[((path, "request"), "validation")],
               "response": schemas.get(((path, "response"), "validation"))}
        for path in ROUTE_CONTRACTS
    }
    return {
        "$schema": SCHEMA_DIALECT,
        "$comment": "Generated by `python -m app.contract_bridge` from the Pydantic models; do not edit by hand.",
        "routes": routes,
        "$defs": top_level.get("$defs", {}),
    }


def contract_schema_text() -> str:
    return json.dumps(contract_schema(), ensure_ascii=False, indent=1, sort_keys=False) + "\n"


def write_contract_schema(path: Path = CONTRACT_SCHEMA_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contract_schema_text(), encoding="utf-8", newline="\n")
    return path


if __name__ == "__main__":
    write_contract_schema(Path(sys.argv[1]) if len(sys.argv) > 1 else CONTRACT_SCHEMA_PATH)
