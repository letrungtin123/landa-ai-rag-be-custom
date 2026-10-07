"""Bounded concurrency, CPU offloading and deadline helpers.

The service runs long provider calls, document parsing and CPU-heavy
validation next to latency-sensitive routes. Everything heavy goes through a
named workload limiter so one large request cannot starve the event loop or
exhaust provider quota for every other caller.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import logging
import os
import threading
import weakref
from collections.abc import AsyncIterator, Callable
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Literal, ParamSpec, TypeVar

from app.core import metrics
from app.core.errors import AppError

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")

WorkloadName = Literal["provider", "index", "cpu"]
SERVICE_BUSY_MESSAGE = "The AI service is busy. Please retry shortly."


def default_cpu_workers() -> int:
    """Leave one core for the event loop; never fewer than one worker."""
    return max(1, min(32, (os.cpu_count() or 2) - 1))


class WorkloadLimiter:
    """Per-event-loop semaphore with a bounded wait.

    asyncio primitives bind to the loop that first waits on them. Tests and
    tooling may run several loops in one process, so each loop gets its own
    semaphore with the same limit.
    """

    def __init__(self, name: WorkloadName, limit: int, acquire_timeout_seconds: float) -> None:
        if limit < 1:
            raise ValueError("Workload limit must be at least one.")
        self.name = name
        self.limit = limit
        self.acquire_timeout_seconds = max(0.0, acquire_timeout_seconds)
        self._semaphores: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
            weakref.WeakKeyDictionary()
        )
        self._lock = threading.Lock()

    def _semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        with self._lock:
            semaphore = self._semaphores.get(loop)
            if semaphore is None:
                semaphore = asyncio.Semaphore(self.limit)
                self._semaphores[loop] = semaphore
            return semaphore

    @asynccontextmanager
    async def slot(self, *, max_wait_seconds: float | None = None) -> AsyncIterator[None]:
        semaphore = self._semaphore()
        wait = self.acquire_timeout_seconds
        if max_wait_seconds is not None:
            wait = max(0.0, min(wait, max_wait_seconds))
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=wait)
        except TimeoutError:
            metrics.LIMITER_REJECTIONS.labels(workload=self.name).inc()
            logger.warning(
                "workload_limiter_saturated",
                extra={"event": "workload_limiter_saturated", "workload": self.name, "limit": self.limit},
            )
            raise AppError("SERVICE_BUSY", 503, SERVICE_BUSY_MESSAGE) from None
        metrics.LIMITER_IN_USE.labels(workload=self.name).inc()
        try:
            yield
        finally:
            semaphore.release()
            metrics.LIMITER_IN_USE.labels(workload=self.name).dec()


class ConcurrencyRuntime:
    """Process-wide limiters and executors, configured from Settings."""

    def __init__(
        self,
        *,
        provider_limit: int,
        index_limit: int,
        cpu_workers: int,
        acquire_timeout_seconds: float,
        extraction_executor: Literal["thread", "process"],
    ) -> None:
        self.provider = WorkloadLimiter("provider", provider_limit, acquire_timeout_seconds)
        self.index = WorkloadLimiter("index", index_limit, acquire_timeout_seconds)
        self.cpu = WorkloadLimiter("cpu", cpu_workers, acquire_timeout_seconds)
        self.cpu_workers = cpu_workers
        self.extraction_executor_kind = extraction_executor
        self._thread_executor: ThreadPoolExecutor | None = None
        self._process_executor: ProcessPoolExecutor | None = None
        self._executor_lock = threading.Lock()

    def _threads(self) -> ThreadPoolExecutor:
        with self._executor_lock:
            if self._thread_executor is None:
                self._thread_executor = ThreadPoolExecutor(
                    max_workers=self.cpu_workers,
                    thread_name_prefix="ai-rag-cpu",
                )
            return self._thread_executor

    def _processes(self) -> ProcessPoolExecutor:
        with self._executor_lock:
            if self._process_executor is None:
                self._process_executor = ProcessPoolExecutor(max_workers=self.cpu_workers)
            return self._process_executor

    async def _run(self, executor: Executor, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        loop = asyncio.get_running_loop()
        async with self.cpu.slot():
            if isinstance(executor, ProcessPoolExecutor):
                return await loop.run_in_executor(executor, functools.partial(func, *args, **kwargs))
            # Copy context so request/correlation IDs reach logs emitted in the thread.
            context = contextvars.copy_context()
            return await loop.run_in_executor(executor, functools.partial(context.run, func, *args, **kwargs))

    async def run_cpu(self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        """Run CPU-heavy or blocking work off the event loop under the CPU limit."""
        return await self._run(self._threads(), func, *args, **kwargs)

    async def run_extraction(self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        """Run untrusted document extraction; a process pool isolates parser crashes."""
        executor: Executor = self._processes() if self.extraction_executor_kind == "process" else self._threads()
        return await self._run(executor, func, *args, **kwargs)

    def shutdown(self) -> None:
        with self._executor_lock:
            if self._thread_executor is not None:
                self._thread_executor.shutdown(wait=False, cancel_futures=True)
                self._thread_executor = None
            if self._process_executor is not None:
                self._process_executor.shutdown(wait=False, cancel_futures=True)
                self._process_executor = None


def deadline_seconds(milliseconds: int) -> float:
    return max(0.001, milliseconds / 1000)
