"""The Gemini provider: client pool, calls, retry budget, error classification and usage accounting.

* ``pool``: one SDK client per API key, leased per call (``client_pool``, ``request_http_options``);
* ``calls``: ``generate_content`` and the embeddings, reached through ``app.services.provider``;
* ``transport``: one bounded call (deadline, provider slot, transient retries, 429 waits, metrics);
* ``retry``: attempts, backoff and server retry hints; ``errors``: content-free failure classification;
* ``usage`` and ``telemetry``: token accounting and safe response-derived telemetry.

Failures are ``app.core.errors.ProviderError`` with a code of ``app.core.error_codes``.

This package replaced the module ``app/infra/gemini.py`` (refactor wave 2b, step 2.4): the pool API
stays importable from here. Patch ``app.infra.gemini.pool.client_pool``, which the calls read.
"""

from app.infra.gemini.pool import DEFAULT_MAX_CLIENTS, GeminiClientPool, client_pool, request_http_options

__all__ = ["DEFAULT_MAX_CLIENTS", "GeminiClientPool", "client_pool", "request_http_options"]
