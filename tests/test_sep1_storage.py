"""SEP-1 #3/#4: signed-URL document download under an outbound allowlist, and its use by the index route.

Offline: every HTTP exchange goes through ``httpx.MockTransport``. The signed URLs below are shaped
like supabase-js ``createSignedUrl`` output (``encodeURI`` of ``{url}/storage/v1/object/sign/...``).
"""

from __future__ import annotations

import asyncio
import functools
import logging
import ssl
import tempfile
import unittest
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch
from urllib.parse import quote

import certifi
import httpx
from pydantic import ValidationError

from app import main
from app.core.errors import AppError, DocumentLimitError
from app.infra import storage
from tests.test_characterization_ingestion import (
    DOC_ID,
    KB_ID,
    TENANT_ID,
    FakeDb,
    FakeEmbedder,
    document_row,
    index_request,
)
from app.schemas.kb import RagIndexRequest

ORIGIN = "https://storage.internal:8443"
BUCKET = "landa-storage"
OBJECT_PATH = f"{TENANT_ID}/kb-files/20261008/1791417600_bao_cao.pdf"
TOKEN = "eyJhbGciOiJIUzI1NiJ9.signed-token-must-never-be-logged"


def signed_url(object_path: str = OBJECT_PATH, *, origin: str = ORIGIN, bucket: str = BUCKET) -> str:
    return f"{origin}/storage/v1/object/sign/{bucket}/{quote(object_path, safe='/')}?token={TOKEN}"


def policy(**overrides: Any) -> storage.StoragePolicy:
    values: dict[str, Any] = {"allowed_origins": frozenset({"https://storage.internal:8443"}), "bucket": BUCKET,
                              "max_bytes": 1024, "timeout_seconds": 5.0}
    values.update(overrides)
    return storage.StoragePolicy(**values)


def transport(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


class OriginPolicyTests(unittest.TestCase):
    def test_origins_are_normalised_with_explicit_ports(self) -> None:
        self.assertEqual(storage.normalize_origin("HTTPS://Storage.Internal/x"), "https://storage.internal:443")
        self.assertEqual(storage.normalize_origin("http://127.0.0.1:54321/"), "http://127.0.0.1:54321")
        self.assertEqual(storage.normalize_origin("https://[::1]:9000"), "https://[::1]:9000")
        for bad in ("ftp://x", "https://", "https://user:pw@host", "not a url", "https://host:99999"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                storage.normalize_origin(bad)

    def test_allowlist_defaults_to_the_supabase_url_origin(self) -> None:
        self.assertEqual(storage.parse_allowed_origins("", "http://127.0.0.1:54321"),
                         frozenset({"http://127.0.0.1:54321"}))
        self.assertEqual(storage.parse_allowed_origins("", ""), frozenset())
        self.assertEqual(
            storage.parse_allowed_origins(" https://a.internal , https://b.internal:8443 ", "http://127.0.0.1:1"),
            frozenset({"https://a.internal:443", "https://b.internal:8443"}),
        )

    def test_plain_http_is_refused_unless_loopback(self) -> None:
        with self.assertRaises(ValueError):
            storage.parse_allowed_origins("http://10.0.0.5:8000", "")
        with self.assertRaises(ValueError):
            storage.parse_allowed_origins("https://ok.internal,::bad::", "")
        with self.assertLogs("app.infra.storage", logging.WARNING) as logs:
            self.assertEqual(storage.parse_allowed_origins("", "http://10.0.0.5:8000"), frozenset())
            self.assertEqual(storage.parse_allowed_origins("", "::bad::"), frozenset())
        self.assertIn("storage_origin_fallback_unusable", "\n".join(logs.output))
        self.assertEqual(storage.parse_allowed_origins("http://localhost:54321", ""),
                         frozenset({"http://localhost:54321"}))


class SignedUrlValidationTests(unittest.TestCase):
    def test_a_supabase_signed_url_for_the_expected_object_is_accepted(self) -> None:
        self.assertEqual(storage.validate_source_url(signed_url(), policy=policy(), object_path=OBJECT_PATH),
                         "https://storage.internal:8443")
        # encodeURI keeps non-ASCII names percent-encoded; the decoded path is compared.
        vietnamese = f"{TENANT_ID}/kb-files/báo cáo.pdf"
        storage.validate_source_url(signed_url(vietnamese), policy=policy(), object_path=vietnamese)

    def test_urls_outside_the_contract_are_rejected_without_echoing_them(self) -> None:
        other_tenant = "55555555-5555-4555-8555-555555555555"
        cases = {
            "other_origin": signed_url(origin="https://evil.example"),
            "same_host_other_port": signed_url(origin="https://storage.internal:9443"),
            "plain_http": signed_url(origin="http://storage.internal:8443"),
            "user_info": signed_url(origin="https://user@storage.internal:8443"),
            "fragment": signed_url() + "#frag",
            "other_object": signed_url(f"{TENANT_ID}/kb-files/other.pdf"),
            "other_tenant": signed_url(f"{other_tenant}/kb-files/20261008/1791417600_bao_cao.pdf"),
            "other_bucket": signed_url(bucket="private-bucket"),
            "encoded_slash": f"{ORIGIN}/storage/v1/object/sign/{BUCKET}/{TENANT_ID}%2Fkb-files%2Fx.pdf?token=t",
            "dot_segment": f"{ORIGIN}/storage/v1/object/sign/{BUCKET}/../{BUCKET}/{OBJECT_PATH}?token=t",
            "malformed": "https://storage.internal:notaport/x",
        }
        for name, url in cases.items():
            with self.subTest(case=name), self.assertLogs("app.infra.storage", logging.WARNING) as logs, \
                    self.assertRaises(AppError) as caught:
                storage.validate_source_url(url, policy=policy(), object_path=OBJECT_PATH)
            self.assertEqual((caught.exception.code, caught.exception.http_status), ("STORAGE_URL_REJECTED", 422))
            self.assertNotIn(TOKEN, "\n".join(logs.output) + str(caught.exception))
        with self.assertRaises(AppError):
            storage.validate_source_url(signed_url(), policy=policy(), object_path=" ")


class DownloadTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.destination = Path(temp.name) / "source.pdf"

    def download(self, handler: Callable[[httpx.Request], httpx.Response], **policy_overrides: Any) -> int:
        return asyncio.run(storage.download_to_file(
            signed_url(), self.destination, policy=policy(**policy_overrides), object_path=OBJECT_PATH,
            transport=transport(handler),
        ))

    def test_streams_the_object_to_disk(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, content=b"%PDF-1.7 body")

        self.assertEqual(self.download(handler), 13)
        self.assertEqual(self.destination.read_bytes(), b"%PDF-1.7 body")
        self.assertEqual(seen[0].headers["accept-encoding"], "identity")
        self.assertEqual(seen[0].url.params["token"], TOKEN)

    def test_size_cap_applies_to_declared_and_streamed_bodies(self) -> None:
        with self.assertRaises(DocumentLimitError):
            self.download(lambda request: httpx.Response(200, content=b"x" * 2048))  # Content-Length declared

        async def body() -> AsyncIterator[bytes]:
            yield b"x" * 600
            yield b"y" * 600

        def chunked(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=body())  # streamed, no Content-Length

        with self.assertRaises(DocumentLimitError):
            self.download(chunked)

    def test_redirects_are_followed_only_within_the_same_origin(self) -> None:
        def same_origin(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith(".pdf"):
                return httpx.Response(302, headers={"location": "/storage/v1/object/final?token=t2"})
            return httpx.Response(200, content=b"ok")

        self.assertEqual(self.download(same_origin), 2)
        for location in ("https://evil.example/steal", "", "https://user@storage.internal:8443/x"):
            with self.subTest(location=location), self.assertRaises(AppError) as caught:
                self.download(lambda request, value=location: httpx.Response(302, headers={"location": value}))
            self.assertEqual(caught.exception.code, "STORAGE_REDIRECT_REJECTED")
        # httpx itself refuses an unparsable Location: a storage protocol failure, not a redirect.
        with self.assertRaises(AppError) as caught:
            self.download(lambda request: httpx.Response(302, headers={"location": "https://h:notaport/x"}))
        self.assertEqual(caught.exception.code, "STORAGE_DOWNLOAD_FAILED")

        def endless(request: httpx.Request) -> httpx.Response:
            return httpx.Response(307, headers={"location": f"/loop/{len(str(request.url))}"})

        with self.assertRaises(AppError) as caught:
            self.download(endless)
        self.assertEqual(caught.exception.code, "STORAGE_REDIRECT_REJECTED")

    def test_storage_failures_map_to_safe_codes(self) -> None:
        for status, code, http_status in ((404, "STORAGE_OBJECT_UNAVAILABLE", 422),
                                          (400, "STORAGE_OBJECT_UNAVAILABLE", 422),
                                          (429, "STORAGE_DOWNLOAD_FAILED", 502),
                                          (503, "STORAGE_DOWNLOAD_FAILED", 502)):
            with self.subTest(status=status), self.assertRaises(AppError) as caught:
                self.download(lambda request, value=status: httpx.Response(value, text="denied"))
            self.assertEqual((caught.exception.code, caught.exception.http_status), (code, http_status))

    def test_transport_errors_never_leak_the_signed_url(self) -> None:
        def broken(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError(f"cannot reach {request.url}", request=request)

        with self.assertLogs("app.infra.storage", logging.WARNING) as logs, self.assertRaises(AppError) as caught:
            self.download(broken)
        self.assertEqual(caught.exception.code, "STORAGE_DOWNLOAD_FAILED")
        self.assertTrue(caught.exception.__suppress_context__)
        self.assertIsNone(caught.exception.__cause__)
        self.assertNotIn(TOKEN, "\n".join(logs.output))

    def test_rejected_urls_never_reach_the_network(self) -> None:
        handler = Mock(side_effect=AssertionError("no request expected"))
        with self.assertRaises(AppError):
            asyncio.run(storage.download_to_file(
                signed_url(origin="https://evil.example"), self.destination, policy=policy(),
                object_path=OBJECT_PATH, transport=transport(handler),
            ))
        handler.assert_not_called()

    def test_custom_ca_file_builds_a_verifying_context(self) -> None:
        context = storage._verify_argument(certifi.where())
        self.assertIsInstance(context, ssl.SSLContext)
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)  # type: ignore[union-attr]
        self.assertIs(storage._verify_argument(""), True)


class IndexRouteSignedUrlTests(unittest.TestCase):
    """The index route downloads through the signed URL before it creates the index row."""

    def run_index(self, db: FakeDb, request: RagIndexRequest,
                  handler: Callable[[httpx.Request], httpx.Response] | None = None) -> tuple[Any, Mock, list[str]]:
        legacy = Mock(side_effect=AssertionError("legacy download not expected"))
        original = storage.download_to_file
        injected = functools.partial(original, transport=transport(handler or (lambda r: httpx.Response(200))))
        captured: list[str] = []

        class Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                captured.append(f"{record.getMessage()} {record.__dict__}")

        handler_obj = Capture()
        logging.getLogger("app").addHandler(handler_obj)
        try:
            with patch("app.main.embed_texts", new=FakeEmbedder()), \
                    patch("app.main.download_storage_object", new=legacy), \
                    patch.object(main.storage_infra, "download_to_file", new=injected), \
                    patch.object(main.settings, "supabase_url", ORIGIN), \
                    patch.object(main.settings, "storage_allowed_origins", ""), \
                    patch.object(main.settings, "supabase_storage_bucket", BUCKET):
                try:
                    outcome: Any = asyncio.run(main.index_document(request, pool=db))
                except Exception as error:
                    outcome = error
        finally:
            logging.getLogger("app").removeHandler(handler_obj)
        return outcome, legacy, captured

    def test_signed_url_download_feeds_the_extractor(self) -> None:
        db = FakeDb(document=document_row(name="notes.txt", file_path=f"{TENANT_ID}/kb-files/notes.txt",
                                           content=None))
        request = index_request(source_download_url=signed_url(f"{TENANT_ID}/kb-files/notes.txt"))
        self.assertNotIn(TOKEN, repr(request))
        result, legacy, logs = self.run_index(
            db, request, lambda r: httpx.Response(200, content=b"Signed body text for the index."),
        )
        self.assertEqual(result["status"], "learned")
        legacy.assert_not_called()
        [(rows,)] = db.args_for("insert_chunk")
        self.assertEqual(rows[0][5], "Signed body text for the index.")
        downloaded = [line for line in logs if "rag_index_downloaded" in line]
        self.assertIn("'storage_source': 'signed_url'", downloaded[0])
        self.assertFalse([line for line in logs if TOKEN in line])

    def test_storage_rejections_fail_before_any_index_row(self) -> None:
        cases = {
            "origin_not_allowed": (signed_url(f"{TENANT_ID}/kb-files/notes.txt", origin="https://evil.example"),
                                   None, "STORAGE_URL_REJECTED"),
            "object_missing": (signed_url(f"{TENANT_ID}/kb-files/notes.txt"),
                               lambda r: httpx.Response(404), "STORAGE_OBJECT_UNAVAILABLE"),
            "storage_down": (signed_url(f"{TENANT_ID}/kb-files/notes.txt"),
                             lambda r: httpx.Response(503), "STORAGE_DOWNLOAD_FAILED"),
        }
        for name, (url, handler, code) in cases.items():
            with self.subTest(case=name):
                db = FakeDb(document=document_row(file_path=f"{TENANT_ID}/kb-files/notes.txt", content=None))
                outcome, _, logs = self.run_index(db, index_request(source_download_url=url), handler)
                self.assertIsInstance(outcome, AppError)
                self.assertEqual(outcome.code, code)
                self.assertEqual(db.labels(), ["load_document"])
                self.assertFalse([line for line in logs if TOKEN in line])

    def test_without_url_or_service_key_the_request_is_refused(self) -> None:
        db = FakeDb(document=document_row(file_path=f"{TENANT_ID}/kb-files/notes.txt", content=None))
        with (
            patch.object(main, "supabase_client", None),
            patch("app.main.embed_texts", new=FakeEmbedder()),
            self.assertRaises(AppError) as caught,
        ):
            asyncio.run(main.index_document(index_request(), pool=db))
        self.assertEqual((caught.exception.code, caught.exception.http_status), ("SOURCE_DOWNLOAD_URL_REQUIRED", 422))
        self.assertEqual(db.labels(), ["load_document"])

    def test_legacy_service_key_path_still_downloads_with_the_sdk(self) -> None:
        bucket = Mock()
        bucket.download.return_value = b"legacy body"
        client = Mock()
        client.storage.from_.return_value = bucket
        with patch.object(main, "supabase_client", client):
            self.assertEqual(main.download_storage_object(f"{TENANT_ID}/kb-files/notes.txt"), b"legacy body")
        client.storage.from_.assert_called_once_with(main.settings.supabase_storage_bucket)

    def test_the_signed_url_is_a_bounded_secret_field(self) -> None:
        with self.assertRaises(ValidationError):
            index_request(source_download_url="https://x/" + "a" * 9000)
        self.assertIsNone(index_request().source_download_url)
        self.assertEqual(KB_ID, index_request().kb_id)
        self.assertEqual(DOC_ID, index_request().document_id)


if __name__ == "__main__":
    unittest.main()
