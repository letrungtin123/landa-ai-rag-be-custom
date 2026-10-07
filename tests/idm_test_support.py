"""Shared helpers for the IDM module/unit/endpoint tests (golden fixture, Appendix B).

Builds the golden course design once per process (deterministic, no network) and
offers the Node attempt-trace admission check as a Python mirror of
``lesson-author-orchestration-v2-attempt.logic.ts``.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import json
import re
import time
import unittest
from collections.abc import Awaitable, Callable
from typing import Any

from app.idm.contracts import IdmCourseDesignV1, IdmCourseSkeletonRequestV1, IdmShardDesignV1
from app.idm.course_design import run_idm_course_design
from app.idm.module_design import run_idm_module_design
from app.idm.runtime import IdmRuntime, IdmTokenAllowance
from app.lesson_author_orchestration_v2 import CourseSkeletonV2, canonical_hash
from app.lesson_author_orchestration_v2_provider import SourceSnapshotFactV2
from tests import idm_golden as g
from tests import idm_golden_module as gm

ALLOWANCE = (1_800_000, 131_072)

# Mirror of the Node admission rules for ``attempt_trace`` events.
NODE_TRACE_KEYS = frozenset({
    "sequence", "invocation_kind", "invocation_index", "provider_attempt", "phase", "outcome", "event_code",
    "failure_stage", "failure_code", "failure_path", "provider_dispatched", "usage_source", "observed_usage",
    "duration_ms", "diagnostics",
})
NODE_INVOCATION_KINDS = frozenset({"writer", "evaluator", "repair", "deterministic"})
NODE_OUTCOMES = frozenset({"started", "succeeded", "failed", "retrying", "fallback", "unknown", "requeued"})
NODE_USAGE_SOURCES = frozenset({"provider_reported", "estimated", "unknown"})
NODE_OBSERVED_KEYS = frozenset({"provider_input_tokens", "provider_output_tokens", "provider_total_tokens"})
_LOWER_TOKEN = re.compile(r"^[a-z][a-z0-9_]{0,95}$")
_FAILURE_TOKEN = re.compile(r"^[A-Z][A-Z0-9_]{0,99}$")


def assert_node_trace(case: unittest.TestCase, events: list[dict[str, Any]]) -> None:
    """Every event must pass ``readOrchestrationV2AttemptTrace`` unchanged."""

    case.assertLessEqual(len(events), 64)
    for index, event in enumerate(events):
        case.assertEqual(set(event), NODE_TRACE_KEYS)
        case.assertEqual(event["sequence"], index + 1)
        case.assertIn(event["invocation_kind"], NODE_INVOCATION_KINDS)
        case.assertTrue(1 <= event["invocation_index"] <= 64)
        case.assertTrue(event["provider_attempt"] is None or 1 <= event["provider_attempt"] <= 8)
        case.assertRegex(event["phase"], _LOWER_TOKEN)
        case.assertIn(event["outcome"], NODE_OUTCOMES)
        case.assertRegex(event["event_code"], _LOWER_TOKEN)
        case.assertIsInstance(event["provider_dispatched"], bool)
        case.assertIn(event["usage_source"], NODE_USAGE_SOURCES)
        case.assertTrue(set(event["observed_usage"]) <= NODE_OBSERVED_KEYS)
        case.assertEqual(event["diagnostics"], {})
        case.assertTrue(0 <= event["duration_ms"] <= 3_600_000)
        if event["outcome"] in {"failed", "fallback"}:
            case.assertRegex(event["failure_stage"], _LOWER_TOKEN)
            case.assertRegex(event["failure_code"], _FAILURE_TOKEN)
        else:
            case.assertIsNone(event["failure_stage"])
            case.assertIsNone(event["failure_code"])
            case.assertIsNone(event["failure_path"])


def make_runtime(provider: Any, *, seconds: float = 500.0, locale: str = "vi",
                 allowance: tuple[int, int] | None = ALLOWANCE) -> IdmRuntime:
    return IdmRuntime(generate=provider, api_key="test-key", model="fake-model", locale=locale,  # type: ignore[arg-type]
                      deadline=time.monotonic() + seconds,
                      token_allowance=IdmTokenAllowance(*allowance) if allowance else None)


def _run_sync[T](factory: Callable[[], Awaitable[T]]) -> T:
    """Run a coroutine to completion, also from inside a running event loop (async tests)."""

    async def main() -> T:
        return await factory()

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(main())
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        return pool.submit(asyncio.run, main()).result()


@functools.cache
def _design_json() -> tuple[str, str]:
    request = IdmCourseSkeletonRequestV1.model_validate({
        "pipeline_version": "idm-1", "project_context": g.project_context(),
        "source_facts": [fact.model_dump() for fact in g.source_facts()],
        "token_allowance": {"input_tokens": ALLOWANCE[0], "output_tokens": ALLOWANCE[1]},
        "remaining_budget_ms": 500_000,
    })
    result = _run_sync(lambda: run_idm_course_design(request, source_snapshot_hash=g.SNAPSHOT_HASH,
                                                     runtime=make_runtime(g.golden_provider()),
                                                     parallelism=4, max_sections=16))
    return json.dumps(result["idm"], ensure_ascii=False), json.dumps(result["skeleton"], ensure_ascii=False)


def golden_design() -> tuple[IdmCourseDesignV1, CourseSkeletonV2]:
    """Fresh copies of the golden course design and skeleton (computed once)."""

    design, skeleton = _design_json()
    return IdmCourseDesignV1.model_validate_json(design), CourseSkeletonV2.model_validate_json(skeleton)


def module_inputs(chapter_index: int) -> tuple[Any, Any, list[SourceSnapshotFactV2]]:
    design, skeleton = golden_design()
    return gm.module_context(design, skeleton, chapter_index, g.source_facts())


def module_response(module_key: str) -> dict[str, Any]:
    return json.loads(json.dumps(gm.MODULE_RESPONSES[module_key], ensure_ascii=False))


@functools.cache
def _shard_json(chapter_index: int) -> str:
    design, skeleton = golden_design()
    context, plan, facts = gm.module_context(design, skeleton, chapter_index, g.source_facts())
    response = module_response(design.modules[chapter_index].module_key)
    provider = g.FakeIdmProvider({"IdmW3W4ModuleResponseV1": [response]})
    result = _run_sync(lambda: run_idm_module_design(context=context, skeleton=skeleton, plan=plan, facts=facts,
                                                     runtime=make_runtime(provider)))
    return json.dumps(result["shard"]["idm_design"], ensure_ascii=False)


def golden_shard(chapter_index: int) -> IdmShardDesignV1:
    return IdmShardDesignV1.model_validate_json(_shard_json(chapter_index))


def rehash_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Recompute the unit contract hash exactly as ``UnitGenerationContractV2`` checks it."""

    base = {key: value for key, value in contract.items() if key != "contract_hash"}
    if base.get("unit_content_policy_version") is None:
        base.pop("unit_content_policy_version", None)
    if base.get("idm_unit_brief") is None:
        base.pop("idm_unit_brief", None)
    contract["contract_hash"] = canonical_hash(base)
    return contract


__all__ = [
    "ALLOWANCE", "assert_node_trace", "golden_design", "golden_shard", "make_runtime", "module_inputs",
    "module_response", "rehash_contract",
]
