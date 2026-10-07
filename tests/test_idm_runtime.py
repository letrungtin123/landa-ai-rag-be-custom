"""Provider runtime shared by every IDM stage (spec §7.0): budget admission, wire schema, trace.

The provider is a local fake; nothing reaches the network.
"""

from __future__ import annotations

import asyncio
import gc
import json
import unittest
import weakref
from typing import Any

from pydantic import BaseModel, ValidationError, create_model

from app.idm.contracts import IdmW3W4ModuleResponseV1
from app.idm.runtime import (
    IdmBudgetError,
    IdmProviderError,
    IdmResponseInvalidError,
    IdmRuntime,
    IdmTokenAllowance,
    admitted_output_tokens,
    estimate_prompt_tokens,
    idm_call,
    idm_generate,
    idm_provider_response_model,
    log_stage,
    record_deterministic_fallback,
)
from tests.idm_golden import FakeUsage
from tests.idm_test_support import assert_node_trace

BOUND_KEYWORDS = {"pattern", "additionalProperties", "minItems", "maxItems", "minLength", "maxLength", "minimum",
                  "maximum"}


class Answer(BaseModel):
    model_config = {"extra": "forbid", "strict": True}
    value: str


def runtime(generate: Any = None, *, seconds: float = 100.0, allowance: tuple[int, int] | None = None,
            timeout_ms: int = 180_000) -> IdmRuntime:
    return IdmRuntime(generate=generate, api_key="secret-key", model="m", locale="vi", deadline=seconds,
                      clock=lambda: 0.0, provider_call_timeout_ms=timeout_ms,
                      token_allowance=IdmTokenAllowance(*allowance) if allowance else None)


class Provider:
    """Answers in order; an exception instance is raised; records the call options."""

    def __init__(self, *answers: Any, telemetry: dict[str, Any] | None = None) -> None:
        self.answers = list(answers)
        self.telemetry = telemetry
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, api_key: str, model: str, prompt: str, **options: Any) -> tuple[str, FakeUsage]:
        self.calls.append({"api_key": api_key, "model": model, "prompt": prompt, **options})
        if self.telemetry is not None:
            options["on_provider_telemetry"](self.telemetry)
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer, FakeUsage(100, 20)


def schema_keys(node: Any) -> set[str]:
    if isinstance(node, dict):
        return set(node) | {key for value in node.values() for key in schema_keys(value)}
    if isinstance(node, list):
        return {key for value in node for key in schema_keys(value)}
    return set()


class AdmissionTests(unittest.TestCase):
    def test_deadline_below_eight_seconds_is_rejected(self) -> None:
        with self.assertRaises(IdmBudgetError) as caught:
            admitted_output_tokens(runtime(seconds=7.99), prompt="x", max_output_tokens=1_000)
        self.assertEqual(caught.exception.code, "IDM_DEADLINE_EXCEEDED")
        self.assertEqual(admitted_output_tokens(runtime(seconds=8.0), prompt="x", max_output_tokens=1_000), 1_000)

    def test_input_allowance(self) -> None:
        prompt = "x" * 350
        self.assertEqual(estimate_prompt_tokens(prompt), 100)
        self.assertEqual(admitted_output_tokens(runtime(allowance=(100, 10_000)), prompt=prompt,
                                                max_output_tokens=1_000), 1_000)
        used = runtime(allowance=(100, 10_000))
        used.usage.input_tokens = 1
        with self.assertRaises(IdmBudgetError) as caught:
            admitted_output_tokens(used, prompt=prompt, max_output_tokens=1_000)
        self.assertEqual(caught.exception.code, "IDM_TASK_TOKEN_BUDGET_EXCEEDED")

    def test_output_shrinks_to_what_is_left_but_not_below_half_the_cap(self) -> None:
        item = runtime(allowance=(1_000_000, 30_000))
        self.assertEqual(admitted_output_tokens(item, prompt="p", max_output_tokens=24_000), 24_000)
        item.usage.output_tokens = 10_000
        self.assertEqual(admitted_output_tokens(item, prompt="p", max_output_tokens=24_000), 20_000)
        item.usage.reserved_output_tokens = 8_000
        self.assertEqual(admitted_output_tokens(item, prompt="p", max_output_tokens=24_000), 12_000)
        item.usage.reserved_output_tokens = 8_001
        with self.assertRaises(IdmBudgetError) as caught:
            admitted_output_tokens(item, prompt="p", max_output_tokens=24_000)
        self.assertEqual(caught.exception.code, "IDM_TASK_TOKEN_BUDGET_EXCEEDED")

    def test_reserve_after_is_kept_for_later_stages(self) -> None:
        item = runtime(allowance=(1_000_000, 30_000))
        self.assertEqual(admitted_output_tokens(item, prompt="p", max_output_tokens=24_000,
                                                reserve_after_tokens=10_000), 20_000)
        with self.assertRaises(IdmBudgetError):
            admitted_output_tokens(item, prompt="p", max_output_tokens=24_000, reserve_after_tokens=18_001)
        self.assertEqual(admitted_output_tokens(item, prompt="p", max_output_tokens=1), 1)

    def test_no_allowance_admits_the_stage_cap_and_failed_provider_blocks(self) -> None:
        item = runtime()
        self.assertEqual(admitted_output_tokens(item, prompt="p" * 10_000_000, max_output_tokens=7), 7)
        item.provider_failure_code = "AI_PROVIDER_TIMEOUT"
        with self.assertRaises(IdmProviderError) as caught:
            admitted_output_tokens(item, prompt="p", max_output_tokens=7)
        self.assertEqual((caught.exception.code, caught.exception.terminal), ("AI_PROVIDER_TIMEOUT", False))


class ProviderWireModelTests(unittest.TestCase):
    def test_wire_schema_drops_unsupported_keywords_and_keeps_the_title(self) -> None:
        wire = idm_provider_response_model(IdmW3W4ModuleResponseV1)
        self.assertIs(idm_provider_response_model(IdmW3W4ModuleResponseV1), wire)
        self.assertTrue(issubclass(wire, IdmW3W4ModuleResponseV1))
        self.assertEqual(wire.__name__, "IdmW3W4ModuleResponseV1IdmWire")
        schema = wire.model_json_schema()
        self.assertEqual(schema["title"], "IdmW3W4ModuleResponseV1")
        self.assertFalse(schema_keys(schema) & BOUND_KEYWORDS)
        server = IdmW3W4ModuleResponseV1.model_json_schema()
        self.assertTrue({"pattern", "additionalProperties", "minLength", "maxItems"} <= schema_keys(server))

    def test_server_model_still_validates_strictly(self) -> None:
        with self.assertRaises(ValidationError):
            IdmW3W4ModuleResponseV1.model_validate_json(json.dumps({"lessons": [], "extra": 1}))
        with self.assertRaises(ValidationError):
            Answer.model_validate_json(json.dumps({"value": "a", "other": 1}))

    # Regression (fixed): the wire cache is a module-level dict keyed by the server model class. The W5 writer and the
    # slot repair build a fresh pydantic model per unit/call (build_staged_instance_response_model,
    # build_staged_multi_repair_model), so every /unit request leaks those classes for the life of
    # the worker. Evidence: app/idm/runtime.py ``_WIRE_CACHE: dict[type[BaseModel], type[BaseModel]]``;
    # a per-request model stays alive after the request (weakref below is never cleared).
    def test_per_request_models_are_not_retained(self) -> None:
        def per_request() -> type[BaseModel]:
            return create_model("PerRequestSlots", answer=(str, ...))

        control = per_request()
        control_ref = weakref.ref(control)
        del control
        gc.collect()
        self.assertIsNone(control_ref())
        model = per_request()
        model_ref = weakref.ref(model)
        idm_provider_response_model(model).model_json_schema()
        del model
        gc.collect()
        self.assertIsNone(model_ref())


class GenerateTests(unittest.IsolatedAsyncioTestCase):
    async def call(self, item: IdmRuntime, *, kind: str = "writer", stage: str = "idm_test",
                   max_output_tokens: int = 1_000) -> Answer:
        return await idm_call(item, stage=stage, prompt="prompt", response_model=Answer,
                              max_output_tokens=max_output_tokens, thinking_level="medium",
                              invocation_kind=kind)  # type: ignore[arg-type]

    async def test_success_records_usage_options_and_trace(self) -> None:
        provider = Provider('{"value": "ok"}', telemetry={"provider_input_tokens": 90, "provider_output_tokens": -1,
                                                           "provider_total_tokens": True, "other": 3})
        item = runtime(provider, seconds=20.0, allowance=(1_000, 5_000), timeout_ms=180_000)
        self.assertEqual((await self.call(item)).value, "ok")
        call = provider.calls[0]
        self.assertEqual((call["api_key"], call["model"], call["max_output_tokens"], call["json_mode"]),
                         ("secret-key", "m", 1_000, True))
        self.assertEqual((call["thinking_level"], call["request_timeout_ms"]), ("medium", 20_000))
        self.assertEqual(call["response_schema"].__name__, "AnswerIdmWire")
        self.assertEqual(item.usage.as_usage(), {"inputTokens": 100, "outputTokens": 20, "embeddingTokens": 0,
                                                 "totalTokens": 120})
        self.assertEqual((item.usage.calls, item.usage.reserved_output_tokens, item.usage.complete), (1, 0, True))
        event = item.trace[0]
        self.assertEqual((event["phase"], event["outcome"], event["event_code"], event["invocation_kind"]),
                         ("idm_test", "succeeded", "provider_response_received", "writer"))
        self.assertEqual((event["usage_source"], event["observed_usage"]),
                         ("provider_reported", {"provider_input_tokens": 90}))
        self.assertTrue(event["provider_dispatched"])
        assert_node_trace(self, item.trace)

    async def test_transient_provider_error_stops_later_calls(self) -> None:
        provider = Provider(IdmProviderError("ai-provider timeout", terminal=False), '{"value": "never"}')
        item = runtime(provider)
        with self.assertRaises(IdmProviderError):
            await self.call(item)
        self.assertEqual(item.provider_failure_code, "ai-provider timeout")
        self.assertFalse(item.usage.complete)
        self.assertEqual(item.usage.reserved_output_tokens, 0)
        with self.assertRaises(IdmProviderError) as caught:
            await self.call(item, kind="evaluator")
        self.assertFalse(caught.exception.terminal)
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(len(item.trace), 1)
        self.assertEqual((item.trace[0]["event_code"], item.trace[0]["failure_code"], item.trace[0]["failure_stage"]),
                         ("provider_request_failed", "AI_PROVIDER_TIMEOUT", "idm_test"))
        assert_node_trace(self, item.trace)

    async def test_terminal_provider_error_does_not_block_the_task(self) -> None:
        provider = Provider(IdmProviderError("AI_PROVIDER_REQUEST_REJECTED", terminal=True), '{"value": "ok"}')
        item = runtime(provider)
        with self.assertRaises(IdmProviderError):
            await self.call(item)
        self.assertIsNone(item.provider_failure_code)
        self.assertEqual((await self.call(item)).value, "ok")

    async def test_invalid_answers_carry_location_only(self) -> None:
        provider = Provider('{"value": "PRIVATE-SOURCE", "leak": "PRIVATE-SOURCE"}', "not json")
        item = runtime(provider)
        for _ in range(2):
            with self.assertRaises(IdmResponseInvalidError) as caught:
                await self.call(item, kind="repair")
            self.assertEqual(caught.exception.code, "IDM_RESPONSE_SCHEMA_INVALID")
            self.assertNotIn("PRIVATE-SOURCE", json.dumps(caught.exception.errors))
            self.assertTrue(all(set(error) == {"type", "loc"} for error in caught.exception.errors))
        self.assertEqual([e["event_code"] for e in item.trace], ["response_invalid", "response_invalid"])
        self.assertEqual([e["invocation_index"] for e in item.trace], [1, 2])
        assert_node_trace(self, item.trace)

    async def test_parse_errors_from_custom_parsers(self) -> None:
        def reject(_text: str) -> Any:
            raise IdmResponseInvalidError("idm w5 instance-invalid", [])

        def value_error(_text: str) -> Any:
            raise ValueError("PRIVATE")

        item = runtime(Provider("{}", "{}"))
        with self.assertRaises(IdmResponseInvalidError) as caught:
            await idm_generate(item, stage="idm_w5_writer", prompt="p", response_schema=Answer, parse=reject,
                               max_output_tokens=10, thinking_level="low")
        self.assertEqual(caught.exception.code, "idm w5 instance-invalid")
        with self.assertRaises(IdmResponseInvalidError) as caught:
            await idm_generate(item, stage="idm_w5_writer", prompt="p", response_schema=Answer, parse=value_error,
                               max_output_tokens=10, thinking_level="low")
        self.assertEqual(caught.exception.errors, [{"type": "ValueError", "loc": []}])
        self.assertEqual([e["failure_code"] for e in item.trace],
                         ["IDM_W5_INSTANCE_INVALID", "IDM_RESPONSE_SCHEMA_INVALID"])
        assert_node_trace(self, item.trace)

    async def test_cancellation_marks_usage_incomplete(self) -> None:
        item = runtime(Provider(asyncio.CancelledError()))
        with self.assertRaises(asyncio.CancelledError):
            await self.call(item)
        self.assertFalse(item.usage.complete)
        self.assertEqual(item.usage.reserved_output_tokens, 0)

    async def test_trace_sequence_indices_and_caps(self) -> None:
        item = runtime(Provider(*(['{"value": "ok"}'] * 70)))
        for _ in range(66):
            await self.call(item, kind="evaluator")
        record_deterministic_fallback(item, stage="idm_w5_unit", code="IDM_W5_UNIT_FALLBACK")
        self.assertEqual(len(item.trace), 64)
        self.assertEqual([event["sequence"] for event in item.trace], list(range(1, 65)))
        self.assertEqual(item.trace[-1]["invocation_index"], 64)
        assert_node_trace(self, item.trace)

    async def test_failure_codes_are_sanitised_and_fallback_events_are_admitted(self) -> None:
        item = runtime(Provider(IdmProviderError("1-bad", terminal=True), IdmProviderError("x" * 150, terminal=True)))
        for _ in range(2):
            with self.assertRaises(IdmProviderError):
                await self.call(item)
        record_deterministic_fallback(item, stage="idm_module", code="idm.module-lesson fallback")
        codes = [event["failure_code"] for event in item.trace]
        self.assertEqual(codes, ["IDM_PROVIDER_FAILED", "X" * 100, "IDM_MODULE_LESSON_FALLBACK"])
        fallback = item.trace[-1]
        self.assertEqual((fallback["invocation_kind"], fallback["outcome"], fallback["provider_dispatched"],
                          fallback["provider_attempt"]), ("deterministic", "fallback", False, None))
        assert_node_trace(self, item.trace)

    async def test_log_stage_emits_one_json_line(self) -> None:
        with self.assertLogs("app.idm", "INFO") as logs:
            log_stage("idm_stage_completed", {"stage": "idm_module", "count": 2})
        line = logs.records[0].getMessage()
        self.assertTrue(line.startswith("lesson_author_idm "))
        self.assertEqual(json.loads(line.split(" ", 1)[1]),
                         {"event": "idm_stage_completed", "stage": "idm_module", "count": 2})


if __name__ == "__main__":
    unittest.main()
