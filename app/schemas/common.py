"""Shared request validators. The provider usage model is ``app.infra.gemini.usage.AiUsage``."""

from __future__ import annotations

from uuid import UUID


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

