"""Request models of the knowledge-base indexing and deletion routes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, SecretStr, field_validator

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
