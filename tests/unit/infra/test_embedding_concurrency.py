"""Embedding batches run concurrently, up to AI_RAG_EMBEDDING_CONCURRENCY at once (audit 2026-10-24, P4).

A fake provider replaces ``app.infra.gemini.calls.embed_text_batch`` (one call per batch): no network.
"""

from __future__ import annotations

import asyncio
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

from app.core.config import settings
from app.core.errors import ProviderError
from app.infra.gemini import calls
from app.schemas.common import AiUsage
from app.services import provider as provider_service

TEXTS = [f"text {number}" for number in range(14)]


class FakeProvider:
    """Answers each text with ``[its number]``; batches that start later finish sooner."""

    def __init__(self, *, fail_batch: int | None = None, block: bool = False) -> None:
        self.fail_batch = fail_batch
        self.block = block
        self.in_flight = 0
        self.max_in_flight = 0
        self.started: list[int] = []
        self.finished: list[int] = []
        self.cancelled: list[int] = []
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, api_key: Any, model: str, contents: list[str], *, task_type: str | None = None,
                       output_dimensionality: int = 768) -> tuple[list[list[float]], AiUsage]:
        batch = len(self.started)
        self.started.append(batch)
        self.calls.append({"model": model, "task_type": task_type, "dims": output_dimensionality})
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(3600 if self.block else 0.01 * (8 - batch % 8))
            if batch == self.fail_batch:
                raise ProviderError("AI_PROVIDER_UNAVAILABLE", "AI provider hiện không khả dụng. Vui lòng thử lại sau.")
        except asyncio.CancelledError:
            self.cancelled.append(batch)
            raise
        finally:
            self.in_flight -= 1
        self.finished.append(batch)
        vectors = [[float(text.split()[1])] for text in contents]
        return vectors, AiUsage(embeddingTokens=3 * len(contents), totalTokens=3 * len(contents))


def embed(fake: FakeProvider, *, concurrency: int, batch_size: int = 2) -> tuple[list[list[float]], AiUsage]:
    with patch.object(calls, "embed_text_batch", fake), patch.object(settings, "embedding_concurrency", concurrency), \
            patch.object(settings, "embedding_batch_size", batch_size):
        return asyncio.run(provider_service.embed_texts("key", "gemini-embedding-001", TEXTS,
                                                        task_type="RETRIEVAL_DOCUMENT", output_dimensionality=768))


class EmbeddingConcurrencyTests(unittest.TestCase):
    def test_vectors_keep_the_order_of_the_texts_whatever_order_the_batches_finish_in(self) -> None:
        for concurrency in (1, 2, 3, 16):
            with self.subTest(concurrency=concurrency):
                fake = FakeProvider()
                vectors, usage = embed(fake, concurrency=concurrency)
                self.assertEqual(vectors, [[float(number)] for number in range(len(TEXTS))])
                self.assertEqual(usage, AiUsage(embeddingTokens=3 * len(TEXTS), totalTokens=3 * len(TEXTS)))
                self.assertEqual(sorted(fake.started), list(range(7)))
                # Batches start in order; each one once, with the arguments of the request.
                self.assertEqual(fake.started, list(range(7)))
                self.assertEqual({(call["model"], call["task_type"], call["dims"]) for call in fake.calls},
                                 {("gemini-embedding-001", "RETRIEVAL_DOCUMENT", 768)})
                if concurrency > 1:
                    self.assertNotEqual(fake.finished, sorted(fake.finished))  # they did finish out of order

    def test_at_most_the_configured_number_of_batches_run_at_once(self) -> None:
        for concurrency, expected in ((1, 1), (2, 2), (3, 3), (16, 7)):
            with self.subTest(concurrency=concurrency):
                fake = FakeProvider()
                embed(fake, concurrency=concurrency)
                self.assertEqual(fake.max_in_flight, expected)

    def test_a_failed_batch_raises_its_error_and_cancels_the_others(self) -> None:
        fake = FakeProvider(fail_batch=2)
        with self.assertRaises(ProviderError) as raised:
            embed(fake, concurrency=2)
        self.assertEqual(raised.exception.detail["code"], "AI_PROVIDER_UNAVAILABLE")
        self.assertEqual(fake.in_flight, 0)
        self.assertNotIn(2, fake.finished)
        # Nothing starts after the failure; what was running is cancelled, not left behind.
        self.assertLess(max(fake.started), 6)
        self.assertEqual(set(fake.started), set(fake.finished) | set(fake.cancelled) | {2})

    def test_the_failure_is_the_one_of_the_first_failed_batch(self) -> None:
        first = ProviderError("AI_PROVIDER_RATE_LIMITED", "rate")
        second = ProviderError("AI_PROVIDER_UNAVAILABLE", "down")

        async def failing(_key: Any, _model: str, contents: list[str], **_kwargs: Any) -> Any:
            await asyncio.sleep(0)
            raise first if contents[0] == "text 0" else second

        with patch.object(calls, "embed_text_batch", failing), patch.object(settings, "embedding_concurrency", 2), \
                patch.object(settings, "embedding_batch_size", 2), self.assertRaises(ProviderError) as raised:
            asyncio.run(provider_service.embed_texts("key", "gemini-embedding-001", TEXTS))
        self.assertIs(raised.exception, first)

    def test_cancelling_the_caller_cancels_every_running_batch(self) -> None:
        fake = FakeProvider(block=True)

        async def scenario() -> None:
            task = asyncio.ensure_future(provider_service.embed_texts("key", "gemini-embedding-001", TEXTS))
            while len(fake.started) < 2:  # noqa: ASYNC110 - polling a fake, no real waiting
                await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        with patch.object(calls, "embed_text_batch", fake), patch.object(settings, "embedding_concurrency", 2), \
                patch.object(settings, "embedding_batch_size", 2):
            asyncio.run(scenario())
        self.assertEqual((sorted(fake.cancelled), fake.in_flight, fake.finished), ([0, 1], 0, []))

    def test_each_batch_keeps_its_own_retries_through_the_transport(self) -> None:
        # One 503 for the second batch, then success: the transport retries that batch alone.
        transient = RuntimeError("503 UNAVAILABLE")
        transient.code = 503  # type: ignore[attr-defined]
        attempts: list[list[str]] = []

        def embed_content(*, model: str, contents: list[str], config: Any) -> Any:
            attempts.append(list(contents))
            if contents == ["text 2", "text 3"] and attempts.count(contents) == 1:
                raise transient
            return MagicMock(embeddings=[MagicMock(values=[float(text.split()[1])]) for text in contents])

        client = MagicMock()
        client.models.embed_content.side_effect = embed_content
        with patch("app.infra.gemini.pool.genai.Client", MagicMock(return_value=client)), \
                patch("app.infra.gemini.retry.provider_retry_delay_seconds", return_value=0.0), \
                patch.object(settings, "embedding_concurrency", 2), patch.object(settings, "embedding_batch_size", 2):
            vectors, _usage = asyncio.run(provider_service.embed_texts("retry-key", "gemini-embedding-001", TEXTS[:6]))
        self.assertEqual(vectors, [[float(number)] for number in range(6)])
        self.assertEqual(attempts.count(["text 2", "text 3"]), 2)
        self.assertEqual(len(attempts), 4)


if __name__ == "__main__":
    unittest.main()
