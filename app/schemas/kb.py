"""Request and response models of the knowledge-base indexing and deletion routes."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, TypeAdapter, field_validator

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
class IndexLearnedResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["learned"]
    chunk_count: int
    structure_source: Any
    structure_confidence: Any
    structure_node_count: int
    diagnostics: dict[str, Any]
    usage: dict[str, Any]


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
