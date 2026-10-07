"""Provider runtime shared by every IDM stage (spec §7.0).

``app.idm`` never imports ``app.main``: the service layer builds an
:class:`IdmRuntime` around its ``generate_content`` transport and injects it. The
runtime enforces the task deadline and the Node-granted token allowance before a
call is scheduled, keeps an attempt trace in the shape Node admits, and turns a
provider response into a strictly validated server model.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Final, Literal, Protocol

from pydantic import BaseModel, ValidationError

from app.idm.policy import (
    IDM_CALL_HEADROOM_SECONDS,
    IDM_CHARS_PER_TOKEN_ESTIMATE,
    IDM_MAX_ATTEMPT_TRACE_EVENTS,
    IDM_MIN_CALL_REMAINING_SECONDS,
    IDM_PROVIDER_CALL_TIMEOUT_MS,
)
from app.lesson_author_orchestration_v2_provider import _project_orchestration_v2_provider_schema

logger = logging.getLogger("app.idm")

ThinkingLevel = Literal["low", "medium", "high"]
InvocationKind = Literal["writer", "evaluator", "repair"]

# Provider failures after which no further call in the same task can succeed.
TRANSIENT_PROVIDER_CODES: Final = frozenset({
    "AI_PROVIDER_TIMEOUT", "AI_PROVIDER_UNAVAILABLE", "AI_PROVIDER_QUOTA_EXHAUSTED", "SERVICE_BUSY",
})
_FAILURE_CODE_RE: Final = re.compile(r"[^A-Z0-9_]")
_MAX_FAILURE_CODE_CHARS: Final = 100
_MS_PER_SECOND: Final = 1000
_MAX_INVOCATION_INDEX: Final = 64


class UsageLike(Protocol):
    inputTokens: int
    outputTokens: int
    totalTokens: int


class GenerateFn(Protocol):
    def __call__(
        self,
        api_key: str,
        model: str,
        prompt: str,
        *,
        max_output_tokens: int,
        json_mode: bool,
        response_schema: type[BaseModel],
        thinking_level: ThinkingLevel,
        request_timeout_ms: int,
        on_provider_telemetry: Callable[[dict[str, Any]], None] | None,
    ) -> Awaitable[tuple[str, UsageLike]]: ...


class IdmError(Exception):
    """Base for IDM failures that carry a stable, content-free code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class IdmBudgetError(IdmError):
    """The deadline or the token allowance does not admit another provider call."""


class IdmStageError(IdmError):
    """A stage cannot produce a valid result (no fallback is possible)."""


class IdmProviderError(IdmError):
    """The provider transport failed; ``terminal`` failures must reach the caller."""

    def __init__(self, code: str, *, terminal: bool, http_status: int = 502) -> None:
        super().__init__(code)
        self.terminal = terminal
        self.http_status = http_status


class IdmResponseInvalidError(IdmError):
    """The provider answered but the answer is not valid JSON for the server model."""

    def __init__(self, code: str, errors: list[dict[str, Any]]) -> None:
        super().__init__(code)
        self.errors = errors


@dataclass(frozen=True)
class IdmTokenAllowance:
    input_tokens: int
    output_tokens: int


@dataclass
class IdmUsageLedger:
    """Running token totals. All stage coroutines share one event loop thread."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reserved_output_tokens: int = 0
    calls: int = 0
    complete: bool = True

    def as_usage(self) -> dict[str, int]:
        return {"inputTokens": self.input_tokens, "outputTokens": self.output_tokens,
                "embeddingTokens": 0, "totalTokens": self.total_tokens}


@dataclass
class IdmRuntime:
    generate: GenerateFn
    api_key: str = field(repr=False)
    model: str
    locale: Literal["vi", "en"]
    deadline: float
    correlation_id: str | None = None
    token_allowance: IdmTokenAllowance | None = None
    usage: IdmUsageLedger = field(default_factory=IdmUsageLedger)
    trace: list[dict[str, Any]] = field(default_factory=list)
    provider_failure_code: str | None = None
    provider_call_timeout_ms: int = IDM_PROVIDER_CALL_TIMEOUT_MS
    clock: Callable[[], float] = time.monotonic
    _invocations: dict[str, int] = field(default_factory=dict)

    def remaining_seconds(self) -> float:
        return self.deadline - self.clock()

    def next_invocation_index(self, kind: str) -> int:
        self._invocations[kind] = self._invocations.get(kind, 0) + 1
        return self._invocations[kind]


_WIRE_CACHE: dict[type[BaseModel], type[BaseModel]] = {}
_CONTRACTS_MODULE: Final = "app.idm.contracts"


def idm_provider_response_model[ModelT: BaseModel](server_model: type[ModelT]) -> type[ModelT]:
    """Provider-only schema projection; the server model still validates everything.

    It reuses the V2 projection (no ``additionalProperties`` and no numeric/length
    bounds) and also drops ``pattern``: identifiers are re-checked server side and
    a regex keyword is not part of the documented provider schema subset.
    """

    cacheable = server_model.__module__ == _CONTRACTS_MODULE
    cached = _WIRE_CACHE.get(server_model) if cacheable else None
    if cached is not None:
        return cached  # type: ignore[return-value]  # cache is keyed by the same class

    def strip_patterns(node: Any) -> None:
        if isinstance(node, dict):
            if isinstance(node.get("pattern"), str):
                del node["pattern"]
            for value in node.values():
                strip_patterns(value)
        elif isinstance(node, list):
            for value in node:
                strip_patterns(value)

    class IdmProviderWire(server_model):  # type: ignore[valid-type,misc]
        @classmethod
        def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
            projected = _project_orchestration_v2_provider_schema(super().model_json_schema(*args, **kwargs))[0]
            strip_patterns(projected)
            if "title" in projected:
                projected["title"] = server_model.__name__
            return projected

    IdmProviderWire.__name__ = f"{server_model.__name__}IdmWire"
    IdmProviderWire.__qualname__ = IdmProviderWire.__name__
    if cacheable:
        # Per-request writer/repair models are created fresh; caching them would leak.
        _WIRE_CACHE[server_model] = IdmProviderWire
    return IdmProviderWire


def estimate_prompt_tokens(prompt: str) -> int:
    return math.ceil(len(prompt) / IDM_CHARS_PER_TOKEN_ESTIMATE)


def admitted_output_tokens(runtime: IdmRuntime, *, prompt: str, max_output_tokens: int,
                           reserve_after_tokens: int = 0) -> int:
    """Check deadline and allowance before a call; return the output cap to request.

    ``reserve_after_tokens`` keeps room for later stages of the same task. The
    output cap is never raised above the stage cap; it shrinks to what is left.
    """

    if runtime.provider_failure_code is not None:
        raise IdmProviderError(runtime.provider_failure_code, terminal=False)
    if runtime.remaining_seconds() < IDM_MIN_CALL_REMAINING_SECONDS:
        raise IdmBudgetError("IDM_DEADLINE_EXCEEDED")
    allowance = runtime.token_allowance
    if allowance is None:
        return max_output_tokens
    ledger = runtime.usage
    if ledger.input_tokens + estimate_prompt_tokens(prompt) > allowance.input_tokens:
        raise IdmBudgetError("IDM_TASK_TOKEN_BUDGET_EXCEEDED")
    available = (allowance.output_tokens - ledger.output_tokens - ledger.reserved_output_tokens
                 - reserve_after_tokens)
    minimum = max(1, max_output_tokens // 2)
    if available < minimum:
        raise IdmBudgetError("IDM_TASK_TOKEN_BUDGET_EXCEEDED")
    return min(max_output_tokens, available)


def _trace(runtime: IdmRuntime, event: dict[str, Any]) -> None:
    if len(runtime.trace) < IDM_MAX_ATTEMPT_TRACE_EVENTS:
        runtime.trace.append({"sequence": len(runtime.trace) + 1, **event})


def _failure_code(value: str) -> str:
    code = _FAILURE_CODE_RE.sub("_", value.upper())[:_MAX_FAILURE_CODE_CHARS]
    return code if code and code[0].isalpha() else "IDM_PROVIDER_FAILED"


async def idm_generate[ResultT](
    runtime: IdmRuntime,
    *,
    stage: str,
    prompt: str,
    response_schema: type[BaseModel],
    parse: Callable[[str], ResultT],
    max_output_tokens: int,
    thinking_level: ThinkingLevel,
    invocation_kind: InvocationKind = "writer",
    reserve_after_tokens: int = 0,
) -> ResultT:
    """Run one bounded provider call and return ``parse(text)``.

    Raises ``IdmBudgetError`` before dispatch, ``IdmProviderError`` for transport
    failures (transient ones also stop later calls of the task) and
    ``IdmResponseInvalidError`` when ``parse`` rejects the answer. ``parse`` may
    raise ``ValidationError``/``ValueError`` or ``IdmResponseInvalidError``.
    """

    output_cap = admitted_output_tokens(runtime, prompt=prompt, max_output_tokens=max_output_tokens,
                                        reserve_after_tokens=reserve_after_tokens)
    remaining_ms = int(runtime.remaining_seconds() * _MS_PER_SECOND)
    timeout_ms = max(1, min(runtime.provider_call_timeout_ms, remaining_ms))
    invocation_index = min(runtime.next_invocation_index(invocation_kind), _MAX_INVOCATION_INDEX)
    observed: dict[str, int] = {}
    started = time.perf_counter()

    def capture(metadata: dict[str, Any]) -> None:
        for key in ("provider_input_tokens", "provider_output_tokens", "provider_total_tokens"):
            value = metadata.get(key)
            if type(value) is int and value >= 0:
                observed[key] = value

    def event(outcome: str, event_code: str, failure_code: str | None) -> dict[str, Any]:
        return {
            "invocation_kind": invocation_kind, "invocation_index": invocation_index, "provider_attempt": 1,
            "phase": stage, "outcome": outcome, "event_code": event_code,
            "failure_stage": stage if failure_code else None, "failure_code": failure_code, "failure_path": None,
            "provider_dispatched": True, "usage_source": "provider_reported" if observed else "unknown",
            "observed_usage": dict(observed),
            "duration_ms": int((time.perf_counter() - started) * _MS_PER_SECOND), "diagnostics": {},
        }

    runtime.usage.reserved_output_tokens += output_cap
    runtime.usage.calls += 1
    # The transport may wait for a provider slot and retry once; this outer bound keeps the
    # whole call inside the task deadline with headroom for the response to reach Node.
    outer_seconds = max(0.001, runtime.remaining_seconds() - IDM_CALL_HEADROOM_SECONDS)
    try:
        try:
            async with asyncio.timeout(outer_seconds):
                text, usage = await runtime.generate(
                    runtime.api_key, runtime.model, prompt,
                    max_output_tokens=output_cap, json_mode=True,
                    response_schema=idm_provider_response_model(response_schema),
                    thinking_level=thinking_level, request_timeout_ms=timeout_ms,
                    on_provider_telemetry=capture,
                )
        except TimeoutError as timeout:
            raise IdmProviderError("AI_PROVIDER_TIMEOUT", terminal=False, http_status=504) from timeout
    except IdmProviderError as error:
        if not error.terminal:
            runtime.provider_failure_code = error.code
        runtime.usage.complete = False
        _trace(runtime, event("failed", "provider_request_failed", _failure_code(error.code)))
        raise
    except asyncio.CancelledError:
        runtime.usage.complete = False
        raise
    finally:
        runtime.usage.reserved_output_tokens -= output_cap
    input_tokens = max(0, int(usage.inputTokens))
    total_tokens = max(0, int(usage.totalTokens))
    runtime.usage.input_tokens += input_tokens
    # Thinking tokens are billed as output but are not in the candidate count.
    runtime.usage.output_tokens += max(int(usage.outputTokens), total_tokens - input_tokens, 0)
    runtime.usage.total_tokens += max(0, int(usage.totalTokens))
    try:
        result = parse(text)
    except IdmResponseInvalidError as error:
        _trace(runtime, event("failed", "response_invalid", _failure_code(error.code)))
        raise
    except (ValidationError, ValueError) as error:
        _trace(runtime, event("failed", "response_invalid", "IDM_RESPONSE_SCHEMA_INVALID"))
        raise IdmResponseInvalidError("IDM_RESPONSE_SCHEMA_INVALID", _safe_validation_errors(error)) from error
    _trace(runtime, event("succeeded", "provider_response_received", None))
    return result


async def idm_call[ModelT: BaseModel](
    runtime: IdmRuntime,
    *,
    stage: str,
    prompt: str,
    response_model: type[ModelT],
    max_output_tokens: int,
    thinking_level: ThinkingLevel,
    invocation_kind: InvocationKind = "writer",
    reserve_after_tokens: int = 0,
) -> ModelT:
    """``idm_generate`` that validates the answer strictly with ``response_model``."""

    return await idm_generate(
        runtime, stage=stage, prompt=prompt, response_schema=response_model,
        parse=response_model.model_validate_json, max_output_tokens=max_output_tokens,
        thinking_level=thinking_level, invocation_kind=invocation_kind, reserve_after_tokens=reserve_after_tokens,
    )


def _safe_validation_errors(error: Exception) -> list[dict[str, Any]]:
    """Location and type only: never the provider's content."""

    if isinstance(error, ValidationError):
        return [{"type": str(item.get("type")), "loc": [str(part) for part in item.get("loc", ())][:8]}
                for item in error.errors(include_url=False, include_input=False)[:12]]
    return [{"type": type(error).__name__, "loc": []}]


def record_deterministic_fallback(runtime: IdmRuntime, *, stage: str, code: str) -> None:
    _trace(runtime, {
        "invocation_kind": "deterministic", "invocation_index": 1, "provider_attempt": None,
        "phase": stage, "outcome": "fallback", "event_code": "deterministic_fallback",
        "failure_stage": stage, "failure_code": _failure_code(code), "failure_path": None,
        "provider_dispatched": False, "usage_source": "unknown", "observed_usage": {},
        "duration_ms": 0, "diagnostics": {},
    })


def log_stage(event: str, payload: dict[str, Any]) -> None:
    """One JSON line per stage event; payloads carry counts and codes only."""

    logger.info("lesson_author_idm %s", json.dumps({"event": event, **payload}, sort_keys=True, default=str))
