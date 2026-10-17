"""Shared request validators and the provider usage model."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, SerializerFunctionWrapHandler, model_serializer


def validate_uuid_string(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} không được để trống.")
    try:
        UUID(text)
    except ValueError as exc:
        raise ValueError(f"{field_name} không hợp lệ.") from exc
    return text


class AiUsage(BaseModel):
    """Provider usage as the provider bills it: input includes tool-use prompt tokens, output includes
    thinking tokens. cachedInputTokens is the part of input served from the context cache (billed at
    the cache rate); it is sent only when non-zero, so the wire shape is unchanged without a cache hit."""

    inputTokens: int = 0
    outputTokens: int = 0
    embeddingTokens: int = 0
    totalTokens: int = 0
    cachedInputTokens: int = 0

    @model_serializer(mode="wrap")
    def _omit_zero_cache(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data: dict[str, Any] = handler(self)
        if not data.get("cachedInputTokens"):
            data.pop("cachedInputTokens", None)
        return data
