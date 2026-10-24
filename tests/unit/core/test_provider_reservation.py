"""Provider slots kept for chat (audit 2026-10-24, P9): background calls never hold all of them."""

from __future__ import annotations

import asyncio
import contextlib
import threading
import unittest
from typing import Any
from unittest.mock import patch

from app.core import metrics
from app.core.concurrency import (
    ConcurrencyRuntime,
    WorkloadLimiter,
    current_provider_priority,
    interactive_provider_calls,
)
from app.core.errors import AppError
from app.infra.gemini.transport import call_provider_with_timeout
from app.services import provider as provider_service
from app.services import runtime as runtime_service
from app.services.chat import service as chat_service
from tests.unit.services.test_characterization_retrieval_chat import (
    DOC_A,
    FakePool,
    PinnedTestCase,
    chunk,
    make_request,
)


def metric(name: str, labels: dict[str, str]) -> float:
    return metrics.REGISTRY.get_sample_value(name, labels) or 0.0


async def hold(limiter: WorkloadLimiter, count: int, *, background: bool, stack: contextlib.AsyncExitStack) -> None:
    for _ in range(count):
        await stack.enter_async_context(limiter.slot(background=background))


async def busy(limiter: WorkloadLimiter, *, background: bool) -> bool:
    """True when the limiter answers SERVICE_BUSY instead of a slot."""

    try:
        async with limiter.slot(background=background):
            return False
    except AppError as error:
        return error.code == "SERVICE_BUSY"


class ReservedSlotTests(unittest.TestCase):
    def limiter(self, limit: int = 8, reserved: int = 2) -> WorkloadLimiter:
        return WorkloadLimiter("provider", limit, acquire_timeout_seconds=0.05, reserved=reserved)

    def test_background_work_leaves_the_reserved_slots_to_chat(self) -> None:
        limiter = self.limiter()
        in_use = {"workload": "provider"}
        rejections_before = metric("ai_rag_limiter_rejections_total", in_use)

        async def scenario() -> list[Any]:
            async with contextlib.AsyncExitStack() as stack:
                await hold(limiter, 6, background=True, stack=stack)
                seen = [metric("ai_rag_limiter_in_use", in_use), await busy(limiter, background=True)]
                await hold(limiter, 2, background=False, stack=stack)  # chat takes the two reserved slots
                seen += [metric("ai_rag_limiter_in_use", in_use), await busy(limiter, background=False)]
            return seen

        in_use_before = metric("ai_rag_limiter_in_use", in_use)
        self.assertEqual(asyncio.run(scenario()), [in_use_before + 6, True, in_use_before + 8, True])
        self.assertEqual(metric("ai_rag_limiter_rejections_total", in_use), rejections_before + 2)
        self.assertEqual(metric("ai_rag_limiter_in_use", in_use), in_use_before)

    def test_chat_may_take_any_free_slot(self) -> None:
        limiter = self.limiter()

        async def scenario() -> tuple[bool, bool]:
            async with contextlib.AsyncExitStack() as stack:
                await hold(limiter, 8, background=False, stack=stack)
                return await busy(limiter, background=False), await busy(limiter, background=True)

        self.assertEqual(asyncio.run(scenario()), (True, True))

    def test_a_background_caller_that_times_out_gives_its_background_slot_back(self) -> None:
        limiter = self.limiter(limit=3, reserved=1)

        async def scenario() -> bool:
            async with contextlib.AsyncExitStack() as chat:
                await hold(limiter, 3, background=False, stack=chat)  # every slot taken by chat
                for _ in range(5):  # each waits with a background slot, then times out on the full limiter
                    self.assertTrue(await busy(limiter, background=True))
            async with contextlib.AsyncExitStack() as background:
                await hold(limiter, 2, background=True, stack=background)  # both background slots still there
                return await busy(limiter, background=True)

        self.assertTrue(asyncio.run(scenario()))

    def test_without_a_reservation_every_caller_shares_the_limit(self) -> None:
        limiter = self.limiter(limit=2, reserved=0)

        async def scenario() -> bool:
            async with contextlib.AsyncExitStack() as stack:
                await hold(limiter, 2, background=True, stack=stack)
                return await busy(limiter, background=False)

        self.assertTrue(asyncio.run(scenario()))

    def test_the_reservation_always_leaves_background_work_a_slot(self) -> None:
        with self.assertRaises(ValueError):
            WorkloadLimiter("provider", 2, 1.0, reserved=2)
        for limit, reserved, expected in ((8, 2, 2), (2, 2, 1), (1, 2, 0), (8, 0, 0), (3, 255, 2)):
            with self.subTest(limit=limit, reserved=reserved):
                runtime = ConcurrencyRuntime(provider_limit=limit, index_limit=1, cpu_workers=1,
                                             acquire_timeout_seconds=1, extraction_executor="thread",
                                             provider_reserved_slots=reserved)
                self.assertEqual(runtime.provider.reserved, expected)
        self.assertEqual(runtime_service.concurrency.provider.reserved, 2)  # the default of 8 slots, 2 kept for chat


class ProviderPriorityTests(unittest.TestCase):
    def test_calls_are_background_unless_marked_interactive(self) -> None:
        async def nested() -> str:
            return current_provider_priority()

        async def scenario() -> list[str]:
            seen = [current_provider_priority()]
            with interactive_provider_calls():
                seen += [current_provider_priority(), await asyncio.ensure_future(nested())]
            return [*seen, current_provider_priority()]

        self.assertEqual(asyncio.run(scenario()), ["background", "interactive", "interactive", "background"])

    def test_the_transport_takes_a_reserved_slot_only_for_chat(self) -> None:
        runtime = ConcurrencyRuntime(provider_limit=2, index_limit=1, cpu_workers=1, acquire_timeout_seconds=30,
                                     extraction_executor="thread", provider_reserved_slots=1)
        release = threading.Event()

        def blocked_call() -> str:
            release.wait(10)
            return "background"

        async def scenario() -> list[Any]:
            first = asyncio.ensure_future(call_provider_with_timeout(blocked_call, "model", request_timeout_ms=5_000))
            while runtime.provider._gates()[1]._value:  # noqa: ASYNC110 - wait for its background slot
                await asyncio.sleep(0.01)
            outcomes: list[Any] = []
            try:
                await call_provider_with_timeout(lambda: "second", "model", request_timeout_ms=300)
            except AppError as error:
                outcomes.append(error.code)
            with interactive_provider_calls():
                outcomes.append(await call_provider_with_timeout(lambda: "chat", "model", request_timeout_ms=5_000))
            release.set()
            outcomes.append(await first)
            return outcomes

        try:
            with patch.object(runtime_service, "concurrency", runtime):
                self.assertEqual(asyncio.run(scenario()), ["SERVICE_BUSY", "chat", "background"])
        finally:
            release.set()
            runtime.shutdown()

    def test_a_busy_background_call_is_still_service_busy(self) -> None:
        # SERVICE_BUSY keeps its code and status for background callers (AI ID and indexing retry on it).
        runtime = ConcurrencyRuntime(provider_limit=1, index_limit=1, cpu_workers=1, acquire_timeout_seconds=0.05,
                                     extraction_executor="thread", provider_reserved_slots=0)

        async def scenario() -> tuple[str, int]:
            async with runtime.provider.slot(background=True):
                try:
                    await call_provider_with_timeout(lambda: None, "model", request_timeout_ms=1_000)
                except AppError as error:
                    return error.code, error.http_status
            raise AssertionError("expected SERVICE_BUSY")

        with patch.object(runtime_service, "concurrency", runtime):
            self.assertEqual(asyncio.run(scenario()), ("SERVICE_BUSY", 503))


class ChatPriorityTests(PinnedTestCase):
    def test_chat_marks_its_provider_calls_interactive(self) -> None:
        priorities: list[str] = []
        embed, generate = self.embed, self.generate

        async def recorded_embed(*args: Any, **kwargs: Any) -> Any:
            priorities.append(current_provider_priority())
            return await embed(*args, **kwargs)

        async def recorded_generate(*args: Any, **kwargs: Any) -> Any:
            priorities.append(current_provider_priority())
            return await generate(*args, **kwargs)

        pool = FakePool(vector=[chunk(DOC_A, 1, "Lockout tagout isolates hazardous energy.", score=0.8)])
        with patch.object(provider_service, "embed_texts", recorded_embed), \
                patch.object(provider_service, "generate_content", recorded_generate):
            result = asyncio.run(chat_service.chat(make_request(), pool=pool))
        self.assertEqual(result["text"], "Generated answer")
        self.assertEqual(priorities, ["interactive", "interactive"])  # the question's embedding and the answer
        self.assertEqual(current_provider_priority(), "background")


if __name__ == "__main__":
    unittest.main()
