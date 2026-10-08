"""Download of a document's source file (backend-signed URL, or the legacy storage client)."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import cast

from app.core.config import settings
from app.core.document_limits import assert_document_size
from app.core.errors import AppError
from app.core.logging import SERVICE_LOGGER_NAME
from app.infra import storage as storage_infra
from app.schemas.kb import RagIndexRequest
from app.services import runtime as service_runtime
from app.services.runtime import storage_policy

logger = logging.getLogger(SERVICE_LOGGER_NAME)


def download_storage_object(storage_path: str) -> bytes:
    """Legacy download with SUPABASE_SERVICE_KEY, used only when the request has no signed URL."""
    if service_runtime.supabase_client is None:
        raise AppError(
            "SOURCE_DOWNLOAD_URL_REQUIRED", 422,
            "The index request must include a signed source download URL.",
        )
    bucket = service_runtime.supabase_client.storage.from_(settings.supabase_storage_bucket)
    return cast(bytes, bucket.download(storage_path))


async def fetch_index_source(request: RagIndexRequest, storage_path: str, destination: Path) -> int:
    """Copy the stored document to ``destination`` (no index row exists yet) and return its size."""
    if request.source_download_url is not None:
        size = await storage_infra.download_to_file(
            request.source_download_url.get_secret_value(),
            destination,
            policy=storage_policy(),
            object_path=storage_path,
        )
        source = "signed_url"
    else:
        raw = await asyncio.to_thread(download_storage_object, storage_path)
        size = len(raw)
        assert_document_size(size, maximum=settings.max_document_bytes)
        destination.write_bytes(raw)
        source = "service_key"
    logger.info(
        "rag_index_downloaded",
        extra={"event": "rag_index_downloaded", "tenant_id": request.tenant_id,
               "document_id": request.document_id, "bytes": size, "storage_source": source},
    )
    return size
