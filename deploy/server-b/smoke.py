"""Post-deploy smoke test. Runs INSIDE the ai-rag container, read from stdin:

    docker compose exec -T ai-rag python - https://<server-b-private-dns>:8443 \
        --cafile /etc/landa-ai/tls/internal-ca.pem < smoke.py      # through nginx (TLS + proxy)
    docker exec -i <container> python - http://127.0.0.1:8010 < smoke.py   # direct (CI)

Checks: /healthz 200, /readyz 200, /metrics 401 without auth, /metrics 200 with an HMAC-signed
request. Through nginx this proves the internal certificate, that the proxy forwards the path
unchanged (the HMAC signs it) and that the configured key verifies. It uses the container's own
AI_RAG_SERVICE_HMAC_SECRETS and prints status codes only, never secrets or signatures.
"""

from __future__ import annotations

import argparse
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
import uuid
from http import HTTPStatus
from urllib.parse import urlsplit

from app.core.security import (
    HMAC_KEY_ID_HEADER,
    HMAC_REQUEST_ID_HEADER,
    HMAC_SIGNATURE_HEADER,
    HMAC_TIMESTAMP_HEADER,
    parse_hmac_secrets,
    sign_hmac_request,
)


def _opener(cafile: str | None) -> urllib.request.OpenerDirector:
    context = ssl.create_default_context(cafile=cafile) if cafile else ssl.create_default_context()
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=context),
    )


def _status(opener: urllib.request.OpenerDirector, url: str, headers: dict[str, str]) -> tuple[int, bytes]:
    # main() only accepts http/https base URLs.
    request = urllib.request.Request(url, headers=headers, method="GET")  # noqa: S310
    try:
        with opener.open(request, timeout=15) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, b""


def _signed_headers(path: str) -> dict[str, str] | None:
    credentials = parse_hmac_secrets(os.environ.get("AI_RAG_SERVICE_HMAC_SECRETS", ""))
    if not credentials:
        return None
    credential = next(iter(credentials.values()))
    timestamp = str(int(time.time()))
    return {
        HMAC_KEY_ID_HEADER: credential.key_id,
        HMAC_TIMESTAMP_HEADER: timestamp,
        HMAC_REQUEST_ID_HEADER: str(uuid.uuid4()),
        HMAC_SIGNATURE_HEADER: sign_hmac_request(credential.secret, timestamp, "GET", path, b""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="landa-ai-rag smoke test")
    parser.add_argument("base_url", help="origin only, e.g. https://ai-rag.internal:8443 (no path)")
    parser.add_argument("--cafile", default=None, help="internal CA bundle for https base URLs")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    parts = urlsplit(base)
    if parts.scheme not in {"http", "https"} or not parts.netloc or parts.path:
        print("FAIL base_url must be an http(s) origin without a path (the backend signs the full path)")
        return 2
    opener = _opener(args.cafile)
    failures = 0

    def check(name: str, ok: bool, detail: str) -> None:
        nonlocal failures
        print(f"{'PASS' if ok else 'FAIL'} {name} {detail}")
        failures += 0 if ok else 1

    status, _ = _status(opener, f"{base}/healthz", {})
    check("/healthz", status == HTTPStatus.OK, f"status={status}")
    status, _ = _status(opener, f"{base}/readyz", {})
    check("/readyz", status == HTTPStatus.OK, f"status={status}")
    status, _ = _status(opener, f"{base}/metrics", {})
    check("/metrics unsigned", status == HTTPStatus.UNAUTHORIZED, f"status={status} (expect 401)")

    headers = _signed_headers("/metrics")
    if headers is None:
        print("SKIP /metrics signed (AI_RAG_SERVICE_HMAC_SECRETS is empty; production must use hmac)")
    else:
        status, body = _status(opener, f"{base}/metrics", headers)
        check(
            "/metrics signed",
            status == HTTPStatus.OK and b"ai_rag_http_requests_total" in body,
            f"status={status}",
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
