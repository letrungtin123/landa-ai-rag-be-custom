"""Request and response models of the knowledge-base indexing and deletion routes."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, TypeAdapter, field_validator
from pydantic.json_schema import SkipJsonSchema

from app.schemas.common import validate_uuid_string


class RagIndexRequest(BaseModel):
    tenant_id: str
    kb_id: str
    document_id: str
    embedding_model: str
    embedding_dimensions: int = 768
    api_key: SecretStr = Field(repr=False)
    # SEP-1: short-lived URL the backend signed for kb_documents.file_path. It embeds a bearer
    # token, so it is a secret (never logged, never echoed). Absent: legacy service-key download.
    source_download_url: SecretStr | None = Field(default=None, repr=False, max_length=8_192)
    # Rebuild even when the document's active index already matches this request (an admin "learn
    # again"). By default an unchanged index is kept and answered as skipped, with zero usage.
    force: bool = False

    @field_validator("tenant_id", "kb_id", "document_id")
    @classmethod
    def validate_required_uuid(cls, value: str, info: Any) -> str:
        return validate_uuid_string(value, info.field_name) or ""


class RagDeleteDocumentRequest(BaseModel):
    tenant_id: str
    kb_id: str
    document_id: str

    @field_validator("tenant_id", "kb_id", "document_id")
    @classmethod
    def validate_required_uuid(cls, value: str, info: Any) -> str:
        return validate_uuid_string(value, info.field_name) or ""


class RagDeleteKbRequest(BaseModel):
    tenant_id: str
    kb_id: str

    @field_validator("tenant_id", "kb_id")
    @classmethod
    def validate_required_uuid(cls, value: str, info: Any) -> str:
        return validate_uuid_string(value, info.field_name) or ""


# ---- responses (field order is the wire order) ------------------------------------------------
# ``skip_reason`` of an index request answered by the document's unchanged active index.
INDEX_SKIP_REASON_UNCHANGED = "INDEX_UNCHANGED"


def _absent(value: object) -> bool:
    return value is None


class IndexLearnedResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["learned"]
    chunk_count: int
    structure_source: Any
    structure_confidence: Any
    structure_node_count: int
    diagnostics: dict[str, Any]
    usage: dict[str, Any]
    # Present only when the active index already held exactly what this request would build: nothing
    # was embedded or written and ``usage`` is zero. Absent on every rebuilt index.
    skipped: Literal[True] | SkipJsonSchema[None] = Field(default=None, exclude_if=_absent)
    skip_reason: Literal["INDEX_UNCHANGED"] | SkipJsonSchema[None] = Field(default=None, exclude_if=_absent)


class IndexFailedResponse(BaseModel):
    """A handled indexing failure: the index row is marked with the safe ``error_reason`` code."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["error"]
    chunk_count: int
    usage: dict[str, Any]
    error_reason: str


IndexDocumentResponse = Annotated[IndexLearnedResponse | IndexFailedResponse, Field(discriminator="status")]
INDEX_DOCUMENT_RESPONSE: TypeAdapter[IndexLearnedResponse | IndexFailedResponse] = TypeAdapter(IndexDocumentResponse)


class DeleteResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deleted: bool
