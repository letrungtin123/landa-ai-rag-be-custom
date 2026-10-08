"""SQL of the readiness probe."""

from __future__ import annotations

from typing import Any

from app.repositories.executor import SqlExecutor


async def ping(pool: SqlExecutor) -> Any:
    """Cheapest round trip proving the pool can run a statement."""
    return await pool.fetchval(
        """
        SELECT 1
        """,
    )
