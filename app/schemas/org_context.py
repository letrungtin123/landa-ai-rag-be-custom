"""Request and response models of ``POST /v1/lesson-author/org-context/distill`` (ORG_CONTEXT spec §4)."""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from app.idm.org_profile import IdmOrgProfileV1
from app.schemas.common import validate_uuid_string

# Context documents one distillation reads (spec §4).
MAX_CONTEXT_DOCUMENTS = 50
# Bounds of the source text one distillation reads. Node's default (LESSON_AUTHOR_ORG_CONTEXT_DISTILL_MAX_SOURCE_CHARS)
# is 1,500,000; the upper bound keeps one request's source text and map calls bounded.
MIN_SOURCE_CHARS = 1_000
MAX_SOURCE_CHARS = 2_000_000


class RagOrgContextDistillRequest(BaseModel):
    """The tenant's context documents to draft the organization profile from (the backend's distill worker).

    ``document_ids``: the learned context documents in the order the profile reads them; ``locale``: the language of
    the profile; ``max_source_chars``: the most source text read (an even sample beyond it). ``model`` and ``api_key``
    are the tenant's provider settings, as on every generation route; ``correlation_id`` is operational metadata.
    """

    model_config = ConfigDict(extra="forbid")

    tenant_id: str
    kb_id: str
    document_ids: list[str] = Field(min_length=1, max_length=MAX_CONTEXT_DOCUMENTS)
    locale: Literal["vi", "en"]
    max_source_chars: int = Field(ge=MIN_SOURCE_CHARS, le=MAX_SOURCE_CHARS, strict=True)
    model: str = Field(min_length=1, max_length=255)
    api_key: SecretStr = Field(repr=False)
    correlation_id: str | None = None

    @field_validator("tenant_id", "kb_id")
    @classmethod
    def validate_required_uuid(cls, value: str, info: Any) -> str:
        return validate_uuid_string(value, info.field_name) or ""

    @field_validator("correlation_id")
    @classmethod
    def validate_optional_correlation_uuid(cls, value: str | None) -> str | None:
        return validate_uuid_string(value, "correlation_id")

    @field_validator("document_ids")
    @classmethod
    def validate_document_ids(cls, value: list[str]) -> list[str]:
        # The canonical (lower-case) form: the database answers document ids in it.
        ids = [str(UUID(validate_uuid_string(item, "document_ids") or "")) for item in value]
        if len(set(ids)) != len(ids):
            raise ValueError("document_ids phải khác nhau.")
        return ids


class OrgContextDistillResponse(BaseModel):
    """The drafted profile (field order is the wire order).

    ``source_document_ids``: the requested documents that had learned text (request order); ``source_chars``: the
    characters of source text the profile was drafted from; ``truncated``: the source was longer than
    ``max_source_chars``, so an even sample of it was read. ``usage`` is the provider usage of every call
    (``usage_complete`` false when a call's usage is unknown), ``provider_calls`` the number of calls.
    """

    model_config = ConfigDict(extra="forbid")

    profile: IdmOrgProfileV1
    source_document_ids: list[str]
    source_chars: int
    truncated: bool
    usage: dict[str, int]
    usage_complete: bool
    provider_calls: int
