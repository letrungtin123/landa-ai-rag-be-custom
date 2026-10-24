"""Process memory and event-loop lag gauges on /metrics (audit 2026-10-24, D2)."""

from __future__ import annotations

import asyncio
import math
import sys
import time
import unittest
from unittest.mock import patch

from app.core import metrics, process_metrics
from app.core.process_metrics import EventLoopLagMonitor, parse_proc_status

PROC_STATUS = """Name:\tpython
Umask:\t0022
State:\tS (sleeping)
VmPeak:\t  912340 kB
VmSize:\t  899112 kB
VmRSS:\t  154236 kB
RssAnon:\t  120004 kB
RssFile:\t   34200 kB
RssShmem:\t      32 kB
VmData:\t  402932 kB
VmSwap:\t    2048 kB
Threads:\t9
"""


def sample(name: str, labels: dict[str, str]) -> float | None:
    return metrics.REGISTRY.get_sample_value(name, labels)


def exposition() -> str:
    return metrics.render_metrics()[0].decode("utf-8")


class ProcessMemoryTests(unittest.TestCase):
    def test_linux_status_gives_resident_and_private_memory(self) -> None:
        self.assertEqual(parse_proc_status(PROC_STATUS),
                         {"working_set": 154_236 * 1024, "private": (120_004 + 2_048) * 1024})
        self.assertEqual(parse_proc_status("VmRSS:\t 10 kB\nRssAnon:\t 4 kB\n"),
                         {"working_set": 10 * 1024, "private": 4 * 1024})  # no swap line: none swapped
        self.assertEqual(parse_proc_status("VmRSS:\t 10 MB\nRssAnon:\tlots kB\nVmSwap: ² kB\n"), {})
        self.assertEqual(parse_proc_status(""), {})

    def test_the_gauges_read_this_process_at_scrape_time(self) -> None:
        memory = process_metrics.process_memory()
        if sys.platform in {"win32", "linux"}:
            self.assertGreater(memory["private"], 1024 * 1024)
            self.assertGreater(memory["working_set"], 1024 * 1024)
        for kind in ("private", "working_set"):
            value = sample("ai_rag_process_memory_bytes", {"kind": kind})
            self.assertIsNotNone(value)
            if sys.platform in {"win32", "linux"}:
                self.assertGreater(value, 1024 * 1024)
        held = bytearray(64 * 1024 * 1024)
        try:
            grown = sample("ai_rag_process_memory_bytes", {"kind": "private"})
        finally:
            del held
        if sys.platform in {"win32", "linux"}:
            self.assertGreater(grown, memory["private"] + 32 * 1024 * 1024)

    def test_a_platform_without_numbers_reports_nan(self) -> None:
        with patch.object(process_metrics, "process_memory", dict):
            value = sample("ai_rag_process_memory_bytes", {"kind": "private"})
        self.assertTrue(value is not None and math.isnan(value))


class EventLoopLagTests(unittest.TestCase):
    def test_a_blocked_loop_shows_as_lag(self) -> None:
        monitor = EventLoopLagMonitor(interval=0.02)

        async def scenario() -> None:
            monitor.start()
            monitor.start()  # already running: still one sampler
            await asyncio.sleep(0.1)
            time.sleep(0.3)  # noqa: ASYNC251 - on purpose: a CPU-bound call blocking the loop
            await asyncio.sleep(0.1)
            await monitor.stop()
            await monitor.stop()

        asyncio.run(scenario())
        self.assertGreaterEqual(sample("ai_rag_event_loop_lag_seconds", {"window": "max_1m"}) or 0.0, 0.25)
        last = sample("ai_rag_event_loop_lag_seconds", {"window": "last"})
        self.assertTrue(last is not None and 0.0 <= last < 0.25)

    def test_the_window_forgets_old_samples(self) -> None:
        monitor = EventLoopLagMonitor(interval=0.5, window=60.0)
        self.assertTrue(math.isnan(monitor.max_lag()))
        now = time.monotonic()
        monitor.record(2.0, now - 120)
        monitor.record(0.5, now - 30)
        monitor.record(0.1, now)
        self.assertEqual(monitor.max_lag(), 0.5)
        self.assertEqual(len(monitor._samples), 2)  # the sample older than the window is gone

    def test_the_metrics_page_has_bounded_labels(self) -> None:
        lines = [line for line in exposition().splitlines()
                 if line.startswith(("ai_rag_process_memory_bytes", "ai_rag_event_loop_lag_seconds"))]
        labels = sorted(line.split("{", 1)[1].split("}", 1)[0] for line in lines)
        self.assertEqual(labels, ['kind="private"', 'kind="working_set"', 'window="last"', 'window="max_1m"'])


if __name__ == "__main__":
    unittest.main()
