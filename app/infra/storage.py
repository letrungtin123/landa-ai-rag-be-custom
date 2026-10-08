"""Download a stored document through a backend-signed URL (SEP-1 #3) under an outbound allowlist (#4).

The backend signs a short-lived URL for exactly one object and sends it with the index request,
so this service needs no storage credential. Before any byte is fetched the URL must:

* use https (plain http only towards a loopback host) and carry no user info or fragment;
* point to an origin in the allowlist (``AI_RAG_STORAGE_ALLOWED_ORIGINS``, default the origin of
  ``SUPABASE_URL``);
* name the expected object: the decoded path ends with ``/{bucket}/{kb_documents.file_path}``,
  whose first segment is the tenant id (checked by the caller).

The body is streamed to a file under a byte cap; redirects are followed only within the same
origin. TLS is always verified. The URL embeds a bearer token, so it is never logged or echoed.
"""

from __future__ import annotations

import logging
import ssl
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

import httpx

from app.core.errors import AppError, DocumentLimitError
from app.infra.net import is_loopback_host

logger = logging.getLogger(__name__)

DEFAULT_PORTS = {"https": 443, "http": 80}
MAX_REDIRECTS = 3
CONNECT_TIMEOUT_SECONDS = 10.0
# Status codes after which the same request may succeed later (the backend re-signs on retry).
TRANSIENT_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})


def storage_error(code: str, http_status: int, message: str) -> AppError:
    return AppError(code=code, http_status=http_status, safe_message=message)


def _rejected(reason: str) -> AppError:
    logger.warning("storage_url_rejected", extra={"event": "storage_url_rejected", "reason": reason})
    return storage_error("STORAGE_URL_REJECTED", 422, "The document download URL is not allowed.")


def normalize_origin(url: str) -> str:
    """``scheme://host:port`` in lower case with the default port made explicit; ValueError if invalid."""
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    if scheme not in DEFAULT_PORTS or not host or parts.username is not None or parts.password is not None:
        raise ValueError("invalid origin")
    port = parts.port or DEFAULT_PORTS[scheme]
    rendered_host = f"[{host}]" if ":" in host else host
    return f"{scheme}://{rendered_host}:{port}"


def _origin_is_safe(origin: str) -> bool:
    parts = urlsplit(origin)
    return parts.scheme == "https" or is_loopback_host(parts.hostname or "")


def parse_allowed_origins(configured: str, supabase_url: str) -> frozenset[str]:
    """Explicit origins must all be valid (ValueError otherwise); the SUPABASE_URL fallback is dropped
    with a warning when it is not usable, so legacy deployments still start."""
    entries = [item.strip() for item in configured.split(",") if item.strip()]
    if entries:
        origins = {normalize_origin(entry) for entry in entries}
        unsafe = sorted(origin for origin in origins if not _origin_is_safe(origin))
        if unsafe:
            raise ValueError("plain http storage origins are allowed only for a loopback host")
        return frozenset(origins)
    if not supabase_url.strip():
        return frozenset()
    try:
        origin = normalize_origin(supabase_url)
    except ValueError:
        logger.warning("storage_origin_fallback_unusable", extra={"event": "storage_origin_fallback_unusable"})
        return frozenset()
    if not _origin_is_safe(origin):
        logger.warning("storage_origin_fallback_unusable", extra={"event": "storage_origin_fallback_unusable"})
        return frozenset()
    return frozenset({origin})


@dataclass(frozen=True, slots=True)
class StoragePolicy:
    allowed_origins: frozenset[str]
    bucket: str
    max_bytes: int
    timeout_seconds: float
    ca_file: str = ""


def validate_source_url(url: str, *, policy: StoragePolicy, object_path: str) -> str:
    """Return the origin of an acceptable signed URL or raise ``STORAGE_URL_REJECTED``."""
    try:
        parts = urlsplit(url)
        origin = normalize_origin(url)
    except ValueError:
        raise _rejected("malformed") from None
    if parts.fragment:
        raise _rejected("fragment")
    if origin not in policy.allowed_origins or not _origin_is_safe(origin):
        raise _rejected("origin_not_allowed")
    raw_path = parts.path
    if "\\" in raw_path or "%2f" in raw_path.lower() or "%5c" in raw_path.lower():
        raise _rejected("encoded_separator")
    decoded = unquote(raw_path)
    if any(segment in {".", ".."} for segment in decoded.split("/")):
        raise _rejected("dot_segment")
    normalized_object = object_path.replace("\\", "/").lstrip("/")
    bucket = policy.bucket.strip("/")
    if not normalized_object.strip() or not decoded.endswith(f"/{bucket}/{normalized_object}"):
        raise _rejected("object_mismatch")
    return origin


def _status_error(status_code: int) -> AppError:
    logger.warning(
        "storage_download_failed",
        extra={"event": "storage_download_failed", "storage_status": status_code},
    )
    if status_code in TRANSIENT_STATUS_CODES or httpx.codes.is_server_error(status_code):
        return storage_error("STORAGE_DOWNLOAD_FAILED", 502, "The document could not be downloaded from storage.")
    return storage_error("STORAGE_OBJECT_UNAVAILABLE", 422,
                         "The document is not available in storage (missing object or expired link).")


def _verify_argument(ca_file: str) -> ssl.SSLContext | bool:
    return ssl.create_default_context(cafile=ca_file) if ca_file else True


async def download_to_file(
    url: str,
    destination: Path,
    *,
    policy: StoragePolicy,
    object_path: str,
    transport: httpx.AsyncBaseTransport | None = None,
) -> int:
    """Stream the object to ``destination`` and return its size in bytes."""
    origin = validate_source_url(url, policy=policy, object_path=object_path)
    current = url
    timeout = httpx.Timeout(policy.timeout_seconds, connect=min(CONNECT_TIMEOUT_SECONDS, policy.timeout_seconds))
    client = httpx.AsyncClient(
        follow_redirects=False,
        timeout=timeout,
        trust_env=False,  # no proxy/netrc from the environment: the allowlist is the only egress rule
        verify=_verify_argument(policy.ca_file),
        transport=transport,
    )
    try:
        async with client:
            for _hop in range(MAX_REDIRECTS + 1):
                async with client.stream("GET", current, headers={"Accept-Encoding": "identity"}) as response:
                    if response.is_redirect:
                        location = response.headers.get("location", "")
                        target = urljoin(current, location) if location else ""
                        try:
                            same_origin = bool(target) and normalize_origin(target) == origin
                        except ValueError:
                            same_origin = False
                        if not same_origin:
                            raise storage_error("STORAGE_REDIRECT_REJECTED", 422,
                                                "The storage server redirected to another origin.")
                        current = target
                        continue
                    if response.status_code != httpx.codes.OK:
                        raise _status_error(response.status_code)
                    declared = response.headers.get("content-length", "")
                    if declared.isdigit() and int(declared) > policy.max_bytes:
                        raise DocumentLimitError()
                    total = 0
                    with destination.open("wb") as handle:
                        # Decoded bytes are counted, so a compressed body cannot expand past the cap.
                        async for chunk in response.aiter_bytes():
                            total += len(chunk)
                            if total > policy.max_bytes:
                                raise DocumentLimitError()
                            handle.write(chunk)
                    return total
            raise storage_error("STORAGE_REDIRECT_REJECTED", 422, "The storage server redirected too many times.")
    except (httpx.HTTPError, httpx.InvalidURL, httpx.StreamError) as error:
        # Chained exceptions are dropped (``from None``): their messages may embed the signed URL.
        logger.warning(
            "storage_download_failed",
            extra={"event": "storage_download_failed", "error_type": type(error).__name__},
        )
        raise storage_error("STORAGE_DOWNLOAD_FAILED", 502,
                            "The document could not be downloaded from storage.") from None
