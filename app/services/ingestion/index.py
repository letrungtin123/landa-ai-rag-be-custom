"""Document indexing: extract, chunk, embed and persist one knowledge-base document."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Any

import asyncpg
from fastapi import HTTPException

from app.core import metrics
from app.core.concurrency import deadline_seconds
from app.core.config import settings
from app.core.document_limits import (
    UnsupportedDocumentTypeError,
    assert_document_size,
    assert_tenant_storage_path,
    document_source_suffix,
)
from app.core.errors import AppError, DocumentLimitError
from app.core.logging import SERVICE_LOGGER_NAME
from app.repositories import indexing as index_repository
from app.repositories.indexing import delete_previous_structure_nodes_if_available, persist_structure_nodes_if_available
from app.repositories.pgvector import vector_literal
from app.schemas.common import AiUsage
from app.schemas.kb import RagIndexRequest
from app.services import provider
from app.services import runtime as service_runtime
from app.services.ingestion import extract as extraction
from app.services.ingestion.chunking import build_chunks, build_index_diagnostics
from app.services.ingestion.extract import ExtractedSection, index_document_temp_path
from app.services.ingestion.storage import fetch_index_source
from app.services.provider import http_error_detail, normalize_embedding_model
from app.services.text import clean_text
from app.source_structure import analyze_source_structure

logger = logging.getLogger(SERVICE_LOGGER_NAME)


SAFE_ERROR_CODE_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]{2,95}$")


# Provider answers after which the same index request may succeed later (F5).
RETRYABLE_PROVIDER_HTTP_STATUSES = frozenset({429, 502, 503, 504})


def retryable_provider_failure(exc: BaseException) -> AppError | None:
    """Map a transient provider failure (quota/rate limit/unavailable/timeout) to a retryable 503."""
    if not isinstance(exc, HTTPException) or exc.status_code not in RETRYABLE_PROVIDER_HTTP_STATUSES:
        return None
    detail = http_error_detail(exc)
    code = str(detail.get("code") or "")
    message = detail.get("message")
    return AppError(
        code if SAFE_ERROR_CODE_PATTERN.fullmatch(code) else "AI_PROVIDER_UNAVAILABLE",
        503,
        message if isinstance(message, str) and message.strip()
        else "AI provider hiện không khả dụng. Vui lòng thử lại sau.",
    )


async def index_document(request: RagIndexRequest, pool: asyncpg.Pool) -> dict[str, Any]:
    # Bounded concurrency: a saturated indexer answers SERVICE_BUSY (503) so the
    # backend's durable KB worker retries later instead of piling up memory.
    async with service_runtime.concurrency.index.slot():
        return await _index_document(request, pool)


async def _index_document(request: RagIndexRequest, pool: asyncpg.Pool) -> dict[str, Any]:
    if request.embedding_dimensions != 768:
        raise HTTPException(status_code=400, detail="RAG hiện chỉ hỗ trợ embedding 768 chiều.")
    row = await index_repository.load_document(pool, request.tenant_id, request.kb_id, request.document_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy tài liệu Knowledge Base.")
    # F4: whitespace-only content has nothing to learn; reject it before any index row exists.
    if row["file_path"] is None and not clean_text(str(row["content"] or "")):
        raise HTTPException(status_code=400, detail="Tài liệu không có file nguồn hoặc nội dung để học.")

    index_id: str | None = None
    raw_bytes: int | None = None
    effective_embedding_model = normalize_embedding_model(request.embedding_model)
    deadline = asyncio.timeout(deadline_seconds(settings.index_deadline_ms))
    try:
        async with deadline:
            logger.info(
                "rag_index_started",
                extra={"event": "rag_index_started", "tenant_id": request.tenant_id,
                       "kb_id": request.kb_id, "document_id": request.document_id},
            )
            with tempfile.TemporaryDirectory() as temp_dir:
                # F1/F3: ownership, type, size and the download are settled before the index row is
                # created, so a rejected request never supersedes the document's running index.
                source_path: Path | None = None
                if row["file_path"]:
                    storage_path = str(row["file_path"])
                    assert_tenant_storage_path(storage_path, request.tenant_id)
                    # Routed by the stored object's extension; unknown types are rejected (F1).
                    suffix = document_source_suffix(storage_path)
                    source_path = index_document_temp_path(temp_dir, str(row["id"]), f"source{suffix}")
                    raw_bytes = await fetch_index_source(request, storage_path, source_path)
                else:
                    raw_bytes = len(str(row["content"] or "").encode("utf-8"))
                    assert_document_size(raw_bytes, maximum=settings.max_document_bytes)
                index_id = await index_repository.start_index_row(pool, row, effective_embedding_model)
                if source_path is not None:
                    sections = await service_runtime.concurrency.run_extraction(extraction.extract_sections, source_path, source_path.name)
                else:
                    sections = [ExtractedSection(text=clean_text(row["content"] or ""))]
            logger.info(
                "rag_index_extracted",
                extra={"event": "rag_index_extracted", "tenant_id": request.tenant_id,
                       "document_id": request.document_id, "section_count": len(sections)},
            )
            structure = await service_runtime.concurrency.run_cpu(analyze_source_structure, sections)
            chunks = await service_runtime.concurrency.run_cpu(build_chunks, sections, structure)
            if not chunks:
                raise ValueError("Không tạo được đoạn kiến thức nào từ tài liệu.")
            index_diagnostics = build_index_diagnostics(
                sections,
                chunks,
                raw_bytes=raw_bytes,
                file_name=row["name"] or None,
            )
            logger.info(
                "rag_index_chunked",
                extra={"event": "rag_index_chunked", "tenant_id": request.tenant_id,
                       "document_id": request.document_id, "chunk_count": len(chunks)},
            )

            embeddings, embedding_usage = await provider.embed_texts(
                request.api_key,
                effective_embedding_model,
                [chunk["content"] for chunk in chunks],
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=request.embedding_dimensions,
            )
            if len(embeddings) != len(chunks):
                raise ValueError("Số lượng embedding không khớp số đoạn kiến thức.")
            logger.info(
                "rag_index_embedded",
                extra={"event": "rag_index_embedded", "tenant_id": request.tenant_id,
                       "document_id": request.document_id, "embedding_count": len(embeddings)},
            )

            content_sha = hashlib.sha256("\n\n".join(chunk["content"] for chunk in chunks).encode("utf-8")).hexdigest()
            async with pool.acquire() as conn:
                async with conn.transaction():
                    index_status = await index_repository.lock_index_status(conn, index_id, row["id"])
                    if index_status != "running":
                        raise ValueError("Phiên học tài liệu đã bị thay thế bởi phiên mới hơn.")
                    # One pipelined executemany inside the same transaction (SEP-1 #7): one round trip
                    # per batch instead of one per chunk when the database is on another server.
                    await index_repository.insert_chunks(
                        conn,
                        [
                            (
                                row["tenant_id"],
                                row["kb_id"],
                                row["id"],
                                index_id,
                                chunk_no,
                                chunk["content"],
                                chunk["content_hash"],
                                chunk["token_count"],
                                chunk["page"],
                                chunk["section"],
                                json.dumps(
                                    {
                                        "source_name": row["name"],
                                        "document_type": row["type"],
                                        **(chunk.get("metadata") or {}),
                                    },
                                    ensure_ascii=False,
                                ),
                                vector_literal(embedding),
                            )
                            for chunk_no, (chunk, embedding) in enumerate(zip(chunks, embeddings))
                        ],
                    )
                    await persist_structure_nodes_if_available(conn, row, index_id, structure)
                    await delete_previous_structure_nodes_if_available(conn, row, index_id)
                    await index_repository.deactivate_other_indexes(conn, row["id"], index_id)
                    await index_repository.mark_index_learned(conn, index_id, row["id"], content_sha, len(chunks))
                    await index_repository.delete_inactive_indexes(conn, row["id"], index_id)

            logger.info(
                "rag_index_completed",
                extra={"event": "rag_index_completed", "tenant_id": request.tenant_id,
                       "document_id": request.document_id, "chunk_count": len(chunks)},
            )
            return {
                "status": "learned",
                "chunk_count": len(chunks),
                "structure_source": structure.get("structure_source"),
                "structure_confidence": structure.get("confidence"),
                "structure_node_count": len(structure.get("nodes", [])),
                "diagnostics": index_diagnostics,
                "usage": embedding_usage.model_dump(),
            }
    except Exception as exc:
        retryable = retryable_provider_failure(exc)
        if isinstance(exc, TimeoutError) and deadline.expired():
            error_code = "INDEX_DEADLINE_EXCEEDED"
            metrics.DEADLINE_EXCEEDED.labels(route="/v1/kb/documents/index").inc()
        elif retryable is not None:
            error_code = retryable.code
        else:
            error_code = exc.code if isinstance(exc, AppError) else "INDEX_DOCUMENT_FAILED"
        if isinstance(exc, (DocumentLimitError, UnsupportedDocumentTypeError)):
            metrics.DOCUMENT_LIMIT_REJECTIONS.labels(code=exc.code).inc()
        await index_repository.mark_index_error(pool, index_id, error_code)
        logger.exception(
            "rag_index_failed",
            extra={"event": "rag_index_failed", "tenant_id": request.tenant_id,
                   "document_id": request.document_id, "error_code": error_code,
                   "error_type": type(exc).__name__},
        )
        if isinstance(exc, DocumentLimitError):
            raise
        if index_id is None and isinstance(exc, AppError):
            # Rejected before any index row (type, ownership, storage): keep the HTTP status.
            raise
        if retryable is not None:
            # F5: a transient provider failure is a retryable 503 carrying the provider code, so the
            # backend's durable worker retries it instead of reading HTTP 200 + status "error".
            raise retryable from None
        if isinstance(exc, AppError) and exc.http_status >= 500:
            raise  # e.g. SERVICE_BUSY from the provider limiter: retryable as well
        return {
            "status": "error",
            "chunk_count": 0,
            "usage": AiUsage().model_dump(),
            "error_reason": error_code,
        }
