"""Shared request validators and the provider usage model."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel


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
    inputTokens: int = 0
    outputTokens: int = 0
    embeddingTokens: int = 0
    totalTokens: int = 0
