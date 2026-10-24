"""Gemini calls of the service: text generation (structured or not) and embeddings.

Callers reach ``generate_content`` / ``embed_texts`` through ``app.services.provider`` so a single
patch target (``app.services.provider.generate_content``) fakes the provider everywhere.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Callable, Coroutine
from time import perf_counter
from typing import Any, Literal

from google.genai import types
from pydantic import BaseModel, SecretStr

from app.core import metrics
from app.core.config import settings
from app.core.logging import SERVICE_LOGGER_NAME, log_event
from app.infra.gemini import pool
from app.infra.gemini.telemetry import emit_safe_provider_telemetry, provider_response_telemetry
from app.infra.gemini.transport import call_provider_with_timeout
from app.infra.gemini.usage import combine_usage, estimate_tokens, normalize_usage, usage_from_google_response
from app.lesson_author_blueprint import describe_lesson_author_blueprint_response
from app.schemas.common import AiUsage

logger = logging.getLogger(SERVICE_LOGGER_NAME)

# The most contents one embedding request carries.
MAX_EMBEDDING_BATCH = 100


DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"


LEGACY_EMBEDDING_MODEL_ALIASES = {
    "text-embedding-004": DEFAULT_EMBEDDING_MODEL,
}


def require_provider_api_key(api_key: str | SecretStr) -> str:
    value = api_key.get_secret_value().strip() if isinstance(api_key, SecretStr) else api_key.strip()
    if not value:
        raise ValueError("Google AI Studio API key chưa được cấu hình.")
    return value


def get_embedding_values(item: Any) -> list[float]:
    values = getattr(item, "values", None)
    if values is None and isinstance(item, dict):
        values = item.get("values")
    if values is None:
        raise ValueError("Google AI Studio không trả embedding hợp lệ.")
    return [float(value) for value in values]


def normalize_embedding_model(model: str) -> str:
    return LEGACY_EMBEDDING_MODEL_ALIASES.get(model.strip(), model.strip())


def embedding_batch_size(model: str) -> int:
    if model == "gemini-embedding-2":
        return 1
    return max(1, min(settings.embedding_batch_size, MAX_EMBEDDING_BATCH))


async def embed_text_batch(
    api_key: str | SecretStr,
    model: str,
    contents: list[str],
    *,
    task_type: str | None = None,
    output_dimensionality: int = 768,
) -> tuple[list[list[float]], AiUsage]:
    effective_model = normalize_embedding_model(model)
    safe_api_key = require_provider_api_key(api_key)

    def run() -> Any:
        config_args: dict[str, Any] = {"outputDimensionality": output_dimensionality}
        if effective_model == "gemini-embedding-001" and task_type:
            config_args["taskType"] = task_type
        config = types.EmbedContentConfig(
            **config_args, http_options=pool.request_http_options(settings.provider_request_timeout_ms))
        with pool.client_pool.lease(safe_api_key) as client:
            return client.models.embed_content(model=effective_model, contents=contents, config=config)

    response = await call_provider_with_timeout(run, effective_model, operation="embed")
    raw_embeddings = getattr(response, "embeddings", None)
    if raw_embeddings is None and isinstance(response, dict):
        raw_embeddings = response.get("embeddings")
    if raw_embeddings is None:
        raw_embeddings = [response]
    embeddings = [get_embedding_values(item) for item in raw_embeddings]
    usage = normalize_usage(embedding_tokens=sum(estimate_tokens(content) for content in contents))
    metrics.PROVIDER_TOKENS.labels(operation="embed", kind="embedding", stage=metrics.current_provider_stage()).inc(
        usage.embeddingTokens)
    return embeddings, usage


async def embed_texts(
    api_key: str | SecretStr,
    model: str,
    contents: list[str],
    *,
    task_type: str | None = None,
    output_dimensionality: int = 768,
) -> tuple[list[list[float]], AiUsage]:
    if not contents:
        return [], AiUsage()

    effective_model = normalize_embedding_model(model)
    batch_size = embedding_batch_size(effective_model)
    batches = [contents[start : start + batch_size] for start in range(0, len(contents), batch_size)]
    results: list[tuple[list[list[float]], AiUsage]] = [([], AiUsage())] * len(batches)
    failures: dict[int, Exception] = {}
    next_batches = iter(range(len(batches)))

    async def embed_batches() -> None:
        # The workers share one iterator: batches start in order and each runs once.
        for index in next_batches:
            try:
                results[index] = await embed_text_batch(
                    api_key,
                    effective_model,
                    batches[index],
                    task_type=task_type,
                    output_dimensionality=output_dimensionality,
                )
            except Exception as error:
                failures[index] = error
                raise

    # Audit 2026-10-24, P4: up to AI_RAG_EMBEDDING_CONCURRENCY batches at once (each still takes a provider slot,
    # its own retries and deadline in call_provider_with_timeout); the vectors keep the order of ``contents``.
    workers = min(settings.embedding_concurrency, len(batches))
    if workers == 1:
        await embed_batches()
    else:
        await _until_first_failure([embed_batches() for _ in range(workers)])
    if failures:
        # The error of the first failed batch, as the sequential loop raised it.
        raise failures[min(failures)]
    embeddings: list[list[float]] = []
    usage = AiUsage()
    for batch_embeddings, batch_usage in results:
        embeddings.extend(batch_embeddings)
        usage = combine_usage(usage, batch_usage)
    return embeddings, usage


async def _until_first_failure(work: list[Coroutine[Any, Any, None]]) -> None:
    """Run ``work`` concurrently until all of it is done or one part fails, then cancel the rest and wait for it.

    The failures are the caller's to raise. A cancelled caller cancels every part and waits for them before its
    cancellation goes on, so no provider call outlives its request.
    """

    tasks = [asyncio.ensure_future(item) for item in work]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def _emit_request_size(
    on_provider_telemetry: Callable[[dict[str, Any]], None],
    *,
    model: str,
    prompt: str,
    response_schema: types.Schema | type[BaseModel] | None,
    max_output_tokens: int,
    provider_timeout_ms: int,
) -> None:
    # Size measurements only. Never emit schema text, prompts or local token
    # estimates under provider usage. Failure to measure must not affect AI.
    try:
        schema_object = (
            response_schema.model_json_schema()
            if isinstance(response_schema, type) and issubclass(response_schema, BaseModel)
            else response_schema.model_dump(mode="json", exclude_none=True)
            if isinstance(response_schema, BaseModel) else None
        )
        schema_wire = (json.dumps(schema_object, ensure_ascii=False, separators=(",", ":"))
                       if schema_object is not None else "")
        emit_safe_provider_telemetry(on_provider_telemetry, {
            "stage": "provider_transport", "event": "provider_request_started",
            "model": model, "prompt_chars": len(prompt),
            "prompt_bytes": len(prompt.encode("utf-8")),
            "response_schema_chars": len(schema_wire),
            "schema_measurement_source": "local_declared_schema_not_sdk_wire",
            "response_schema_sha256": hashlib.sha256(schema_wire.encode("utf-8")).hexdigest(),
            "configured_max_output_tokens": max_output_tokens,
            "provider_timeout_ms": provider_timeout_ms,
            "usage_source": "unavailable",
        })
    except Exception:
        emit_safe_provider_telemetry(on_provider_telemetry, {
            "stage": "provider_transport", "event": "request_size_unavailable",
            "usage_source": "unavailable",
        })


def _response_text(
    response: Any,
    response_schema: types.Schema | type[BaseModel] | None,
    *,
    model: str,
    on_provider_telemetry: Callable[[dict[str, Any]], None] | None,
) -> str:
    """The answer text: the SDK's parsed structured answer serialised again, else the raw text."""

    text = getattr(response, "text", "") or ""
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        if isinstance(parsed, BaseModel) and not getattr(response_schema, "retain_raw_provider_text", False):
            # Do not synthesize optional response fields from Pydantic defaults.
            # In particular, v4 Course Architect output must stay semantic-only
            # until the deterministic server allocator injects canonical facts.
            text = json.dumps(parsed.model_dump(mode="json", exclude_unset=True), ensure_ascii=False,
                              separators=(",", ":"))
        elif isinstance(parsed, (dict, list)):
            text = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    elif response_schema is not None:
        diagnostics = describe_lesson_author_blueprint_response(text)
        candidates = getattr(response, "candidates", None) or []
        if on_provider_telemetry is not None:
            emit_safe_provider_telemetry(on_provider_telemetry, {
                "stage": "provider_response",
                "event": "structured_response_unparsed",
                "schema": getattr(response_schema, "__name__", type(response_schema).__name__),
                "model": model,
                "candidate_count": len(candidates),
                "finish_reasons": [str(getattr(candidate, "finish_reason", None))[:80]
                                   for candidate in candidates[:3]],
                "has_prompt_feedback": bool(getattr(response, "prompt_feedback", None)),
                # This helper already reports only structural counters/types.
                "response_diagnostics": diagnostics,
            })
        else:
            log_event(logger, logging.WARNING, "structured_response_unparsed",
                      schema=getattr(response_schema, "__name__", type(response_schema).__name__), model=model,
                      candidates=len(candidates),
                      finish_reasons=[str(getattr(candidate, "finish_reason", None))[:80]
                                      for candidate in candidates[:3]],
                      prompt_feedback=bool(getattr(response, "prompt_feedback", None)), diagnostics=diagnostics)
    return text


async def generate_content(
    api_key: str | SecretStr,
    model: str,
    prompt: str,
    *,
    max_output_tokens: int,
    json_mode: bool = False,
    response_schema: types.Schema | type[BaseModel] | None = None,
    thinking_config: types.ThinkingConfig | dict[str, Any] | None = None,
    thinking_level: Literal["low", "medium", "high"] | None = None,
    request_timeout_ms: int | None = None,
    on_provider_telemetry: Callable[[dict[str, Any]], None] | None = None,
    rate_limit_max_wait_ms: int | None = None,
) -> tuple[str, AiUsage]:
    safe_api_key = require_provider_api_key(api_key)
    if response_schema is not None and not json_mode:
        raise ValueError("response_schema requires JSON mode.")
    provider_timeout_ms = max(1, request_timeout_ms or settings.provider_request_timeout_ms)

    model_id = model.strip().lower().split("/")[-1]
    gemini_38 = model_id == "gemini-3.8-flash"

    def run() -> Any:
        # The deadline travels with the request: the pooled client of this key serves every timeout.
        config: dict[str, Any] = {"max_output_tokens": max_output_tokens,
                                  "http_options": pool.request_http_options(provider_timeout_ms)}
        if gemini_38:
            # Gemini 3.8 rejects legacy sampling parameters. Use its supported
            # reasoning-level contract and deliberately ignore old
            # include-thoughts-only configs supplied by legacy call sites.
            config["thinking_config"] = {
                "thinking_level": thinking_level or settings.gemini_38_thinking_level,
            }
        else:
            config["temperature"] = settings.generation_temperature
        if json_mode:
            config["response_mime_type"] = "application/json"
        if response_schema is not None:
            config["response_schema"] = response_schema
        if thinking_config is not None and not gemini_38:
            config["thinking_config"] = thinking_config
        with pool.client_pool.lease(safe_api_key) as client:
            return client.models.generate_content(model=model, contents=prompt, config=config)

    provider_started = perf_counter()
    last_provider_attempt = 0
    if on_provider_telemetry is not None:
        _emit_request_size(on_provider_telemetry, model=model, prompt=prompt, response_schema=response_schema,
                           max_output_tokens=max_output_tokens, provider_timeout_ms=provider_timeout_ms)

    def on_provider_diagnostic(metadata: dict[str, Any]) -> None:
        nonlocal last_provider_attempt
        provider_attempt = metadata.get("provider_attempt")
        if type(provider_attempt) is int and provider_attempt > 0:
            last_provider_attempt = provider_attempt
        emit_safe_provider_telemetry(
            on_provider_telemetry,
            {
                "stage": "provider_transport",
                "configured_max_output_tokens": max_output_tokens,
                "duration_ms": int((perf_counter() - provider_started) * 1000),
                **metadata,
            },
        )

    rate_limit_options: dict[str, Any] = (
        {"rate_limit_max_wait_ms": rate_limit_max_wait_ms} if rate_limit_max_wait_ms is not None else {}
    )
    response = await call_provider_with_timeout(
        run,
        model,
        request_timeout_ms=provider_timeout_ms,
        on_provider_diagnostic=on_provider_diagnostic if on_provider_telemetry is not None else None,
        **rate_limit_options,
    )
    # Capture real provider metadata before SDK response access/parsing can fail.
    # An HTTP success is not a validated lesson, nor permission to persist it.
    received = provider_response_telemetry(response, model=model, max_output_tokens=max_output_tokens,
                                          prompt=prompt, response_text="",
                                          duration_ms=round((perf_counter() - provider_started) * 1000))
    emit_safe_provider_telemetry(on_provider_telemetry, {
        **{key: value for key, value in received.items()
           if key.startswith("provider_") or key in {"model", "duration_ms", "configured_max_output_tokens"}},
        "usage_source": "provider" if received["usage_source"] == "provider" else "unavailable",
        "event": "provider_response_received",
        "provider_attempt": last_provider_attempt or 1,
    })
    text = _response_text(response, response_schema, model=model, on_provider_telemetry=on_provider_telemetry)
    emit_safe_provider_telemetry(
        on_provider_telemetry,
        provider_response_telemetry(
            response,
            model=model,
            max_output_tokens=max_output_tokens,
            prompt=prompt,
            response_text=text,
            duration_ms=max(0, round((perf_counter() - provider_started) * 1000)),
        ),
    )
    usage = usage_from_google_response(response, prompt, text)
    stage = metrics.current_provider_stage()
    metrics.PROVIDER_TOKENS.labels(operation="generate", kind="input", stage=stage).inc(usage.inputTokens)
    metrics.PROVIDER_TOKENS.labels(operation="generate", kind="output", stage=stage).inc(usage.outputTokens)
    return text, usage
