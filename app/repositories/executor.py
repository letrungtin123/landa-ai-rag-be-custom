"""The statement-executing surface repositories need from an asyncpg pool or connection."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Protocol


class SqlExecutor(Protocol):
    """Structural type of ``asyncpg.Pool`` / ``asyncpg.Connection`` (and the test fakes)."""

    async def fetch(self, query: str, *args: Any) -> list[Any]: ...

    async def fetchrow(self, query: str, *args: Any) -> Any: ...

    async def fetchval(self, query: str, *args: Any) -> Any: ...

    async def execute(self, query: str, *args: Any) -> str: ...

    async def executemany(self, command: str, args: Iterable[Sequence[Any]]) -> None: ...
