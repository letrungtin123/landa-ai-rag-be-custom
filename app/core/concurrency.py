"""Bounded concurrency, CPU offloading and deadline helpers.

The service runs long provider calls, document parsing and CPU-heavy
validation next to latency-sensitive routes. Everything heavy goes through a
named workload limiter so one large request cannot starve the event loop or
exhaust provider quota for every other caller.
"""

from __future__ import annotations

import asyncio
import contextlib
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
from app.core.child_process import ChildOutcome, ChildProcess
from app.core.errors import AppError

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")

WorkloadName = Literal["provider", "index", "cpu"]
ExtractionExecutor = Literal["thread", "process", "isolated"]
SERVICE_BUSY_MESSAGE = "The AI service is busy. Please retry shortly."
# How long a cancelled caller waits for its killed child process to be gone before giving the CPU slot back.
CHILD_KILL_WAIT_SECONDS = 10.0


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
        extraction_executor: ExtractionExecutor,
    ) -> None:
        self.provider = WorkloadLimiter("provider", provider_limit, acquire_timeout_seconds)
        self.index = WorkloadLimiter("index", index_limit, acquire_timeout_seconds)
        self.cpu = WorkloadLimiter("cpu", cpu_workers, acquire_timeout_seconds)
        self.cpu_workers = cpu_workers
        self.extraction_executor_kind = extraction_executor
        self._thread_executor: ThreadPoolExecutor | None = None
        self._process_executor: ProcessPoolExecutor | None = None
        self._provider_executor: ThreadPoolExecutor | None = None
        self._executor_lock = threading.Lock()
        self._child_runs: set[_ChildRun] = set()

    def _threads(self) -> ThreadPoolExecutor:
        with self._executor_lock:
            if self._thread_executor is None:
                self._thread_executor = ThreadPoolExecutor(
                    max_workers=self.cpu_workers,
                    thread_name_prefix="ai-rag-cpu",
                )
            return self._thread_executor

    def _provider_threads(self) -> ThreadPoolExecutor:
        with self._executor_lock:
            if self._provider_executor is None:
                self._provider_executor = ThreadPoolExecutor(
                    max_workers=self.provider.limit,
                    thread_name_prefix="ai-rag-provider",
                )
            return self._provider_executor

    def _processes(self) -> ProcessPoolExecutor:
        with self._executor_lock:
            if self._process_executor is None:
                self._process_executor = ProcessPoolExecutor(max_workers=self.cpu_workers)
            return self._process_executor

    async def _run(self, executor: Executor, func: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> T:
        loop = asyncio.get_running_loop()
        async with self.cpu.slot():
            if isinstance(executor, ProcessPoolExecutor):
                return await loop.run_in_executor(executor, functools.partial(func, *args, **kwargs))
            # Copy context so request/correlation IDs reach logs emitted in the thread.
            context = contextvars.copy_context()
            return await loop.run_in_executor(executor, functools.partial(context.run, func, *args, **kwargs))

    async def run_cpu(self, func: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> T:
        """Run CPU-heavy or blocking work off the event loop under the CPU limit."""
        return await self._run(self._threads(), func, *args, **kwargs)

    async def run_provider_call(self, func: Callable[[], T]) -> T:
        """Run one synchronous provider SDK call on the provider threads (audit 2026-10-24, P5).

        One thread per provider slot, apart from the default executor that ``asyncio.to_thread`` shares with every
        other blocking call: a chat call never waits for a thread behind an AI ID run's calls or a storage download.
        The caller holds the provider slot (``WorkloadLimiter``) and bounds the await; a call whose await timed out
        keeps its thread until the SDK's own request timeout (the same deadline) ends it.
        """
        loop = asyncio.get_running_loop()
        # Copy context so request/correlation IDs and the provider stage reach logs emitted in the thread.
        context = contextvars.copy_context()
        return await loop.run_in_executor(self._provider_threads(), functools.partial(context.run, func))

    async def run_extraction(self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        """Run untrusted document extraction in this process (``thread``) or a process pool (``process``).

        The ``isolated`` executor runs each document in its own child process instead
        (``app.services.ingestion.extract_isolated`` through ``run_in_child``); here it means the thread executor.
        """
        executor: Executor = self._processes() if self.extraction_executor_kind == "process" else self._threads()
        return await self._run(executor, func, *args, **kwargs)

    async def run_in_child(self, module: str, request: bytes, *, memory_limit_bytes: int, max_output_bytes: int,
                           environment: dict[str, str] | None = None) -> ChildOutcome:
        """Run ``python -m module`` in a fresh child process under the CPU limit (audit 2026-10-24, M2).

        The child gets ``request`` on stdin; its stdout is the outcome's output. A cancelled await (the caller's
        deadline, a disconnect, the service stopping) kills the child and everything it started, and the CPU slot is
        given back once the child is gone, so a parser that never returns cannot keep a slot. ``environment`` is added
        to the child's minimal environment (``app.core.child_process.child_environment``).
        """
        run = _ChildRun(module, request, memory_limit_bytes=memory_limit_bytes, max_output_bytes=max_output_bytes,
                        environment=environment)
        async with self.cpu.slot():
            with self._executor_lock:
                self._child_runs.add(run)
            try:
                # The thread starts the child and blocks on its pipes; the event loop only waits.
                future = self._threads().submit(contextvars.copy_context().run, run.run)
                try:
                    return await asyncio.wrap_future(future)
                except asyncio.CancelledError:
                    run.cancel()
                    with contextlib.suppress(Exception, asyncio.CancelledError):
                        await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(future)), CHILD_KILL_WAIT_SECONDS)
                    raise
            finally:
                with self._executor_lock:
                    self._child_runs.discard(run)

    def shutdown(self) -> None:
        with self._executor_lock:
            child_runs, self._child_runs = list(self._child_runs), set()
        for run in child_runs:
            run.cancel()
        with self._executor_lock:
            if self._thread_executor is not None:
                self._thread_executor.shutdown(wait=False, cancel_futures=True)
                self._thread_executor = None
            if self._process_executor is not None:
                self._process_executor.shutdown(wait=False, cancel_futures=True)
                self._process_executor = None
            if self._provider_executor is not None:
                self._provider_executor.shutdown(wait=False, cancel_futures=True)
                self._provider_executor = None


class _ChildRun:
    """One ``run_in_child`` call: the child is started and read on a worker thread and killed from any thread."""

    def __init__(self, module: str, request: bytes, *, memory_limit_bytes: int, max_output_bytes: int,
                 environment: dict[str, str] | None) -> None:
        self._module = module
        self._request = request
        self._memory_limit_bytes = memory_limit_bytes
        self._max_output_bytes = max_output_bytes
        self._environment = environment
        self._lock = threading.Lock()
        self._child: ChildProcess | None = None
        self._cancelled = False

    def run(self) -> ChildOutcome:
        child = ChildProcess(self._module, memory_limit_bytes=self._memory_limit_bytes, environment=self._environment)
        try:
            with self._lock:
                self._child = child
                cancelled = self._cancelled
            if cancelled:  # cancelled while the child was starting: it never gets the request
                child.kill()
            return child.exchange(self._request, max_output_bytes=self._max_output_bytes)
        finally:
            child.close()

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True
            child = self._child
        if child is not None:
            child.kill()


def deadline_seconds(milliseconds: int) -> float:
    return max(0.001, milliseconds / 1000)
