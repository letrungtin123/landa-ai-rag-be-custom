"""Real memory and event-loop lag of the service process, as Prometheus gauges on /metrics (audit 2026-10-24, D2).

PM2 restarts on the working set it reads on Windows (30.6 MB measured when the process held 113.8 MB), so the
service reports its own numbers, read when /metrics is scraped:

- ``ai_rag_process_memory_bytes{kind="private"}``: memory only this process holds. Windows: private bytes (the
  commit charge, ``PROCESS_MEMORY_COUNTERS_EX.PrivateUsage``); Linux: anonymous resident memory plus swap
  (``RssAnon`` + ``VmSwap`` of /proc/self/status).
- ``ai_rag_process_memory_bytes{kind="working_set"}``: resident memory (Windows working set, Linux ``VmRSS``).
- ``ai_rag_event_loop_lag_seconds{window="last"|"max_1m"}``: how late the event loop woke up for a timer, in the
  last sample and at most over the last minute. A CPU-bound call on the loop (a regex, a large JSON) shows here
  before it shows as timeouts.

A value the platform does not give is NaN. Label values are fixed.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Final, Literal

from prometheus_client import Gauge

from app.core.metrics import REGISTRY

MemoryKind = Literal["private", "working_set"]
MEMORY_KINDS: Final[tuple[MemoryKind, ...]] = ("private", "working_set")
LAG_SAMPLE_SECONDS: Final = 0.5
LAG_WINDOW_SECONDS: Final = 60.0
_KIB = 1024

PROCESS_MEMORY = Gauge(
    "ai_rag_process_memory_bytes",
    "Memory of the service process: private (Windows private bytes, Linux anonymous resident plus swap) and "
    "working_set (resident).",
    ("kind",),
    registry=REGISTRY,
)
EVENT_LOOP_LAG = Gauge(
    "ai_rag_event_loop_lag_seconds",
    "How late the event loop ran a timer: in the last sample and the most over the last minute.",
    ("window",),
    registry=REGISTRY,
)


def parse_proc_status(text: str) -> dict[MemoryKind, int]:
    """The memory of /proc/self/status (Linux): working set ``VmRSS``, private ``RssAnon`` + ``VmSwap``."""

    fields: dict[str, int] = {}
    for line in text.splitlines():
        name, _, value = line.partition(":")
        number, _, unit = value.strip().partition(" ")
        if name in {"VmRSS", "RssAnon", "VmSwap"} and unit.strip() == "kB" and number.isdecimal():
            fields[name] = int(number) * _KIB
    memory: dict[MemoryKind, int] = {}
    if "VmRSS" in fields:
        memory["working_set"] = fields["VmRSS"]
    if "RssAnon" in fields:
        memory["private"] = fields["RssAnon"] + fields.get("VmSwap", 0)
    return memory


if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    class _ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t)]

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    _kernel32.K32GetProcessMemoryInfo.argtypes = (wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD)
    _kernel32.K32GetProcessMemoryInfo.restype = wintypes.BOOL

    def process_memory() -> dict[MemoryKind, int]:
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not _kernel32.K32GetProcessMemoryInfo(_kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return {}
        return {"private": int(counters.PrivateUsage), "working_set": int(counters.WorkingSetSize)}
else:
    _PROC_STATUS = Path("/proc/self/status")

    def process_memory() -> dict[MemoryKind, int]:
        try:
            return parse_proc_status(_PROC_STATUS.read_text(encoding="ascii", errors="replace"))
        except OSError:
            return {}


def _memory_reader(kind: MemoryKind) -> float:
    value = process_memory().get(kind)
    return math.nan if value is None else float(value)


for _kind in MEMORY_KINDS:
    PROCESS_MEMORY.labels(kind=_kind).set_function(functools.partial(_memory_reader, _kind))


class EventLoopLagMonitor:
    """Samples the lag of the running loop every ``interval`` seconds while started."""

    def __init__(self, interval: float = LAG_SAMPLE_SECONDS, window: float = LAG_WINDOW_SECONDS) -> None:
        self.interval = interval
        self.window = window
        self._samples: deque[tuple[float, float]] = deque()
        self._last = math.nan
        self._task: asyncio.Task[None] | None = None
        EVENT_LOOP_LAG.labels(window="last").set_function(lambda: self._last)
        EVENT_LOOP_LAG.labels(window="max_1m").set_function(self.max_lag)

    def record(self, lag: float, now: float | None = None) -> None:
        moment = time.monotonic() if now is None else now
        self._last = lag
        self._samples.append((moment, lag))
        while self._samples and self._samples[0][0] < moment - self.window:
            self._samples.popleft()

    def max_lag(self) -> float:
        cutoff = time.monotonic() - self.window
        recent = [lag for moment, lag in self._samples if moment >= cutoff]
        return max(recent) if recent else math.nan

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            expected = loop.time() + self.interval
            await asyncio.sleep(self.interval)
            self.record(max(0.0, loop.time() - expected))

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._run(), name="ai-rag-event-loop-lag")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


event_loop_lag = EventLoopLagMonitor()

__all__ = ["EVENT_LOOP_LAG", "PROCESS_MEMORY", "EventLoopLagMonitor", "event_loop_lag", "parse_proc_status",
           "process_memory"]
