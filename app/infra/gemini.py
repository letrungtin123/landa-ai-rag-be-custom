"""Reusable Gemini SDK clients.

Creating a ``genai.Client`` per call rebuilds HTTP connection pools and TLS
sessions for every provider request. Clients are cached per (API key, timeout)
in a small LRU. API keys are only ever held as SHA-256 digests in the cache
key, never logged.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import OrderedDict
from typing import Any

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

DEFAULT_MAX_CLIENTS = 32


class GeminiClientPool:
    def __init__(self, max_clients: int = DEFAULT_MAX_CLIENTS) -> None:
        if max_clients < 1:
            raise ValueError("max_clients must be at least one.")
        self.max_clients = max_clients
        self._clients: OrderedDict[tuple[str, int, int], Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, api_key: str, timeout_ms: int) -> Any:
        # The factory identity is part of the key so instrumented or patched
        # client factories (tests, probes) never receive a stale cached client.
        factory = genai.Client
        key = (hashlib.sha256(api_key.encode("utf-8")).hexdigest(), int(timeout_ms), id(factory))
        with self._lock:
            client = self._clients.get(key)
            if client is not None:
                self._clients.move_to_end(key)
                return client
        client = factory(api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_ms)))
        evicted: list[Any] = []
        with self._lock:
            existing = self._clients.get(key)
            if existing is not None:
                self._clients.move_to_end(key)
                evicted.append(client)
                client = existing
            else:
                self._clients[key] = client
                while len(self._clients) > self.max_clients:
                    _, old = self._clients.popitem(last=False)
                    evicted.append(old)
        for old in evicted:
            _close_quietly(old)
        return client

    def clear(self) -> None:
        with self._lock:
            clients = list(self._clients.values())
            self._clients.clear()
        for client in clients:
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


client_pool = GeminiClientPool()


def gemini_client(api_key: str, timeout_ms: int) -> Any:
    return client_pool.get(api_key, timeout_ms)
