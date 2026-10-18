"""Reusable Gemini SDK clients.

Creating a ``genai.Client`` per call rebuilds HTTP connection pools and TLS
sessions for every provider request. One client is cached per API key in a
small LRU; API keys are only ever held as SHA-256 digests in the cache key,
never logged.

The timeout is not a client option: every call passes its own deadline as
request ``http_options`` (``request_http_options``), so calls with different
deadlines (IDM shrinks the last calls of a task to the remaining budget) share
one client and its connection pool.

A client is leased for the duration of one SDK call (``client_pool.lease``).
A worker thread may still be inside a call after its awaiting coroutine timed
out, so a client that leaves the cache (LRU eviction or ``clear()``) is closed
only once no lease holds it.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

DEFAULT_MAX_CLIENTS = 32


@dataclass(slots=True)
class _PooledClient:
    client: Any
    leases: int = 0
    # Out of the cache: closed when the last lease ends.
    retired: bool = False


class GeminiClientPool:
    def __init__(self, max_clients: int = DEFAULT_MAX_CLIENTS) -> None:
        if max_clients < 1:
            raise ValueError("max_clients must be at least one.")
        self.max_clients = max_clients
        self._clients: OrderedDict[tuple[str, int], _PooledClient] = OrderedDict()
        self._lock = threading.Lock()

    @contextmanager
    def lease(self, api_key: str) -> Iterator[Any]:
        """The cached client of ``api_key``, held for one SDK call (never closed while held)."""

        entry = self._acquire(api_key)
        try:
            yield entry.client
        finally:
            self._release(entry)

    def _acquire(self, api_key: str) -> _PooledClient:
        # The factory identity is part of the key so instrumented or patched
        # client factories (tests, probes) never receive a stale cached client.
        factory = genai.Client
        key = (hashlib.sha256(api_key.encode("utf-8")).hexdigest(), id(factory))
        with self._lock:
            entry = self._clients.get(key)
            if entry is not None:
                self._clients.move_to_end(key)
                entry.leases += 1
                return entry
        # Built outside the lock: client construction must not serialise other keys' calls.
        created = factory(api_key=api_key)
        idle: list[Any] = []
        with self._lock:
            entry = self._clients.get(key)
            if entry is not None:
                # Another thread cached a client for this key meanwhile; ours was never used.
                self._clients.move_to_end(key)
                idle.append(created)
            else:
                entry = _PooledClient(created)
                self._clients[key] = entry
                while len(self._clients) > self.max_clients:
                    _, evicted = self._clients.popitem(last=False)
                    evicted.retired = True
                    if evicted.leases == 0:
                        idle.append(evicted.client)
            entry.leases += 1
        for client in idle:
            _close_quietly(client)
        return entry

    def _release(self, entry: _PooledClient) -> None:
        with self._lock:
            entry.leases -= 1
            close = entry.retired and entry.leases == 0
        if close:
            _close_quietly(entry.client)

    def clear(self) -> None:
        """Drop every cached client; idle ones close now, leased ones when their call returns."""

        with self._lock:
            entries = list(self._clients.values())
            self._clients.clear()
            idle = []
            for entry in entries:
                entry.retired = True
                if entry.leases == 0:
                    idle.append(entry.client)
        for client in idle:
            _close_quietly(client)

    def __len__(self) -> int:
        with self._lock:
            return len(self._clients)


def _close_quietly(client: Any) -> None:
    close = getattr(client, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception:  # closing a cached client must never fail a request
        logger.debug("gemini_client_close_failed", extra={"event": "gemini_client_close_failed"})


def request_http_options(timeout_ms: int) -> types.HttpOptions:
    """Transport options of one call: its own deadline (the pooled client carries no timeout).

    The SDK applies it to that HTTP request only (httpx timeout and the ``X-Server-Timeout``
    header); it is never part of the request body.
    """

    return types.HttpOptions(timeout=max(1, int(timeout_ms)))


client_pool = GeminiClientPool()
