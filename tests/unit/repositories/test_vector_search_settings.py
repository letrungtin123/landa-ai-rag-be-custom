"""Per-query HNSW settings of the vector search (audit 2026-10-24, P3).

The HNSW index covers every tenant and the tenant/KB/active-index filters apply after it: a scan stopping at
pgvector's 40 candidates returned 14 of 32 rows on the local database. The query raises hnsw.ef_search to its
limit and, with pgvector 0.8+, scans on in distance order, both for its own transaction only.
"""

from __future__ import annotations

import asyncio
import unittest
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import patch

from app.core.config import settings
from app.repositories import retrieval
from app.services.retrieval.search import hnsw_ef_search


class RecordingPool:
    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[RecordingPool]:
        self.events.append(("acquire", None))
        yield self

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[RecordingPool]:
        self.events.append(("begin", None))
        yield self
        self.events.append(("commit", None))

    async def execute(self, query: str, *args: Any) -> str:
        self.events.append(("execute", (" ".join(query.split()), args)))
        return "SELECT 1"

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        self.events.append(("fetch", args))
        return [{"chunk_no": 1}]


def search(pool: RecordingPool, *, ef_search: int, iterative_scan: Any = "strict_order") -> list[Any]:
    return asyncio.run(retrieval.fetch_vector_matches(
        pool, "tenant", "kb", None, "[0.1,0.2]", 32, "gemini-embedding-001", 768,
        ef_search=ef_search, iterative_scan=iterative_scan))


class VectorSearchSettingsTests(unittest.TestCase):
    def test_settings_are_set_for_the_query_transaction_only(self) -> None:
        pool = RecordingPool()
        self.assertEqual(search(pool, ef_search=96), [{"chunk_no": 1}])
        self.assertEqual([name for name, _ in pool.events], ["acquire", "begin", "execute", "fetch", "commit"])
        statement, args = pool.events[2][1]
        self.assertEqual(args, ("96", "strict_order"))
        self.assertIn("set_config('hnsw.ef_search', $1::text, true)", statement)
        self.assertIn("current_setting('hnsw.iterative_scan', true) IS NOT NULL", statement)
        self.assertIn("THEN set_config('hnsw.iterative_scan', $2::text, true)", statement)
        self.assertIn("FROM (SELECT '[1]'::vector) AS loaded", statement)
        self.assertEqual(pool.events[3][1], ("tenant", "kb", None, "[0.1,0.2]", 32, "gemini-embedding-001", 768))

    def test_zero_keeps_the_database_settings_and_one_statement(self) -> None:
        pool = RecordingPool()
        search(pool, ef_search=0)
        self.assertEqual([name for name, _ in pool.events], ["fetch"])

    def test_ef_search_is_at_least_the_limit_within_pgvector_bounds(self) -> None:
        self.assertEqual([hnsw_ef_search(limit) for limit in (1, 32, 40, 96, 1_000, 25_000)],
                         [40, 40, 40, 96, 1_000, 1_000])
        with patch.object(settings, "vector_ef_search", 120):
            self.assertEqual([hnsw_ef_search(limit) for limit in (32, 200)], [120, 200])
        with patch.object(settings, "vector_ef_search", 0):
            self.assertEqual(hnsw_ef_search(96), 0)


if __name__ == "__main__":
    unittest.main()
