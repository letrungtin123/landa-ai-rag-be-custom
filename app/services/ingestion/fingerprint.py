"""Fingerprint of a document index: what a re-index compares to skip rebuilding an unchanged index.

The fingerprint is stored in ``rag_document_indexes.content_sha256`` when an index is learned. It hashes
everything the index stores except the embedding vectors (every chunk row value and the source
structure the structure nodes come from), the embedding model and size the vectors were made with,
and the pipeline versions. A request whose fingerprint equals the active index's would store exactly
the same rows again, so the embedding call (the paid step) can be skipped.

The fingerprint is taken from the pipeline's output, not from the source bytes: a parser, chunker or
library change that alters any stored value changes it even when no version constant was bumped,
and a change that alters nothing (most documents after a parser fix) keeps it. The versions are
included as well, so bumping one rebuilds every index. Indexes learned before the fingerprint
existed hold a plain hash of their chunk text, which never equals a fingerprint: they rebuild once.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from app.content.source_structure import PARSER_VERSION
from app.core.config import settings
from app.services.ingestion.chunking import CHUNKER_VERSION, SOURCE_EVIDENCE_PROPAGATION_VERSION
from app.services.ingestion.extract import STRUCTURED_EXTRACTION_VERSION

# Bump when something the stored rows do not show changes what an index holds: the text sent to the
# embedding model, its task type, or how vectors are stored (``vector_literal``).
INDEX_FINGERPRINT_VERSION = "rag-index-fingerprint-v1"
# The embedding task of every indexed chunk (part of the fingerprint through the versions).
EMBEDDING_TASK_TYPE = "RETRIEVAL_DOCUMENT"

# (chunk_no, content, content_hash, token_count, source_page, source_section, metadata JSON): the
# values of one ``rag_chunks`` row besides its ids and embedding, in insert order.
StoredChunk = tuple[int, str, str, int, int | None, str | None, str]


def stored_chunk_values(document: Mapping[str, Any], chunks: Sequence[Mapping[str, Any]]) -> list[StoredChunk]:
    """The values each chunk row of ``document`` stores, built once for the fingerprint and the insert."""
    return [
        (
            chunk_no,
            chunk["content"],
            chunk["content_hash"],
            chunk["token_count"],
            chunk["page"],
            chunk["section"],
            json.dumps(
                {
                    "source_name": document["name"],
                    "document_type": document["type"],
                    **(chunk.get("metadata") or {}),
                },
                ensure_ascii=False,
            ),
        )
        for chunk_no, chunk in enumerate(chunks)
    ]


def index_pipeline_versions() -> dict[str, Any]:
    """Versions and settings that decide what an index holds besides its source."""
    return {
        "fingerprint": INDEX_FINGERPRINT_VERSION,
        "extraction": STRUCTURED_EXTRACTION_VERSION,
        "structure_parser": PARSER_VERSION,
        "chunker": CHUNKER_VERSION,
        "evidence_contract": SOURCE_EVIDENCE_PROPAGATION_VERSION,
        "chunk_max_chars": settings.chunk_max_chars,
        "chunk_overlap_chars": settings.chunk_overlap_chars,
        "embedding_task_type": EMBEDDING_TASK_TYPE,
    }


def index_fingerprint(
    *,
    embedding_model: str,
    embedding_dimensions: int,
    structure: Mapping[str, Any],
    chunks: Sequence[StoredChunk],
) -> str:
    """SHA-256 hex digest of one index (see the module docstring); CPU work, run it off the event loop.

    Fed as JSON lines, one per chunk, so a large document is never serialized into one string.
    """
    digest = hashlib.sha256()
    _feed(digest, {
        "versions": index_pipeline_versions(),
        "embedding": {"model": embedding_model, "dimensions": embedding_dimensions},
        "structure": structure,
        "chunk_count": len(chunks),
    })
    for chunk in chunks:
        _feed(digest, chunk)
    return digest.hexdigest()


def _feed(digest: Any, value: Any) -> None:
    # Canonical JSON: sorted keys and fixed separators; a JSON text never contains a raw newline.
    digest.update(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(b"\n")
