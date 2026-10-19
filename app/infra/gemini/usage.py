"""Usage accounting of provider calls: token counts as Google bills them, local estimates otherwise."""

from __future__ import annotations

from typing import Any

from app.schemas.common import AiUsage


def estimate_tokens(text: str | None) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def normalize_usage(
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    embedding_tokens: int = 0,
    total_tokens: int | None = None,
    cached_input_tokens: int = 0,
) -> AiUsage:
    total = total_tokens if total_tokens is not None else input_tokens + output_tokens + embedding_tokens
    inputs = max(0, int(input_tokens))
    return AiUsage(
        inputTokens=inputs,
        outputTokens=max(0, int(output_tokens)),
        embeddingTokens=max(0, int(embedding_tokens)),
        totalTokens=max(0, int(total)),
        cachedInputTokens=min(inputs, max(0, int(cached_input_tokens))),
    )


def combine_usage(*items: AiUsage) -> AiUsage:
    return normalize_usage(
        input_tokens=sum(item.inputTokens for item in items),
        output_tokens=sum(item.outputTokens for item in items),
        embedding_tokens=sum(item.embeddingTokens for item in items),
        total_tokens=sum(item.totalTokens for item in items),
        cached_input_tokens=sum(item.cachedInputTokens for item in items),
    )


def _token_count(meta: Any, name: str) -> int | None:
    """A usage counter of the provider response, or None when it is absent (never a mock or a bool)."""

    value = getattr(meta, name, None)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def usage_from_google_response(response: Any, fallback_prompt: str, fallback_output: str) -> AiUsage:
    """Usage as Google bills it: thinking tokens are output (not part of the candidate count) and
    tool-use prompt tokens are input; the cached part of the input is kept apart."""

    meta = getattr(response, "usage_metadata", None)
    prompt_tokens = _token_count(meta, "prompt_token_count")
    candidate_tokens = _token_count(meta, "candidates_token_count")
    total_tokens = _token_count(meta, "total_token_count")
    input_tokens = (prompt_tokens if prompt_tokens is not None else estimate_tokens(fallback_prompt)) \
        + (_token_count(meta, "tool_use_prompt_token_count") or 0)
    output_tokens = (candidate_tokens if candidate_tokens is not None else estimate_tokens(fallback_output)) \
        + (_token_count(meta, "thoughts_token_count") or 0)
    return normalize_usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=max(total_tokens, input_tokens + output_tokens) if total_tokens is not None else None,
        cached_input_tokens=_token_count(meta, "cached_content_token_count") or 0,
    )
