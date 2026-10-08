"""Chunking of extracted sections and the bounded index diagnostics stored with an index row."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.services.ingestion.extract import STRUCTURED_EXTRACTION_VERSION, ExtractedSection
from app.services.provider import estimate_tokens
from app.services.text import clean_text
from app.source_structure import analyze_source_structure, chunk_structure_metadata

SOURCE_EVIDENCE_PROPAGATION_VERSION = "source-evidence-propagation-v1"
SOURCE_EVIDENCE_READY = "ready"
SOURCE_EVIDENCE_LEGACY_REVIEW_REQUIRED = "legacy_review_required"


def split_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            lines = paragraph.splitlines()
            if lines and lines[0].strip().upper() == "[TABLE]" and any(
                re.match(r"^Row\s+\d+\s*:", line.strip(), flags=re.IGNORECASE)
                for line in lines[1:]
            ):
                # A table row is an atomic relation. Split only between rows;
                # character slicing can detach a value from its source column.
                table_chunk = "[TABLE]"
                for raw_line in lines[1:]:
                    line = raw_line.strip()
                    if not line:
                        continue
                    candidate = f"{table_chunk}\n{line}"
                    if len(candidate) > max_chars and table_chunk != "[TABLE]":
                        chunks.append(table_chunk)
                        table_chunk = f"[TABLE]\n{line}"
                    else:
                        table_chunk = candidate
                if table_chunk != "[TABLE]":
                    chunks.append(table_chunk)
                continue
            for start in range(0, len(paragraph), max_chars - overlap_chars):
                chunks.append(paragraph[start : start + max_chars].strip())
            continue
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current.strip())
            tail = current[-overlap_chars:].strip() if overlap_chars > 0 else ""
            current = f"{tail}\n\n{paragraph}".strip() if tail else paragraph
    if current:
        chunks.append(current.strip())
    return [chunk for chunk in chunks if chunk]


def build_chunks(
    sections: list[ExtractedSection],
    structure: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    effective_structure = structure or analyze_source_structure(sections)
    source_evidence_revision = hashlib.sha256(json.dumps({
        "extraction_version": STRUCTURED_EXTRACTION_VERSION,
        "structure": effective_structure,
        "sections": [
            {
                "page": section.page,
                "section": section.section,
                "content_kinds": section.metadata.get("content_kinds", []),
                "table_count": section.metadata.get("table_count", 0),
                "reading_order": section.metadata.get("reading_order"),
                "visual_regions": section.metadata.get("visual_regions", []),
            }
            for section in sections
        ],
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    for section in sections:
        for text in split_text(section.text, settings.chunk_max_chars, settings.chunk_overlap_chars):
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            metadata = chunk_structure_metadata(
                effective_structure,
                page=section.page,
                text=text,
                include_outline=not chunks,
            )
            # Heading candidates have already been compiled into the canonical
            # source structure. Do not repeat private source titles in every
            # chunk's metadata row.
            extraction_metadata = {
                key: value
                for key, value in section.metadata.items()
                if key != "heading_candidates"
            }
            table_count = len(re.findall(r"(?m)^\[TABLE\]\s*$", text))
            table_only = bool(table_count) and all(
                not line.strip()
                or line.strip().upper() == "[TABLE]"
                or re.match(r"^Row\s+\d+\s*:", line.strip(), flags=re.IGNORECASE)
                for line in text.splitlines()
            )
            content_kinds = {
                str(value).strip().casefold()
                for value in extraction_metadata.get("content_kinds", [])
                if isinstance(value, str) and value.strip()
            }
            content_kinds.discard("table")
            if table_only:
                content_kinds.discard("text")
            elif text.strip():
                content_kinds.add("text")
            if table_count:
                content_kinds.add("table")
            extraction_metadata["content_kinds"] = sorted(content_kinds)
            extraction_metadata["table_count"] = table_count
            extraction_metadata["structured_evidence_contract_version"] = SOURCE_EVIDENCE_PROPAGATION_VERSION
            extraction_metadata["source_evidence_revision"] = source_evidence_revision
            metadata = {**extraction_metadata, **metadata}
            metadata = {key: value for key, value in metadata.items() if value is not None}
            chunks.append(
                {
                    "content": text,
                    "page": section.page,
                    "section": section.section,
                    "token_count": estimate_tokens(text),
                    "content_hash": content_hash,
                    "metadata": metadata,
                }
            )
    return chunks


def build_index_diagnostics(
    sections: list[ExtractedSection],
    chunks: list[dict[str, Any]],
    *,
    raw_bytes: int | None = None,
    file_name: str | None = None,
) -> dict[str, Any]:
    """Return bounded, non-content diagnostics for auditing extraction loss."""
    extracted_chars = sum(len(clean_text(section.text)) for section in sections)
    indexed_chars = sum(len(str(chunk.get("content") or "")) for chunk in chunks)
    candidate_chunk_count = sum(
        len(split_text(section.text, settings.chunk_max_chars, settings.chunk_overlap_chars))
        for section in sections
    )
    pages = sorted({section.page for section in sections if section.page is not None})
    sections_without_page = sum(1 for section in sections if section.page is None)
    structured_section_count = sum(
        1 for section in sections
        if section.metadata.get("extraction_version") == STRUCTURED_EXTRACTION_VERSION
    )
    extracted_table_count = sum(
        max(0, int(section.metadata.get("table_count") or 0))
        for section in sections
    )
    extracted_heading_count = sum(
        len(section.metadata.get("heading_candidates") or [])
        for section in sections
    )
    evidence_revisions = {
        str((chunk.get("metadata") or {}).get("source_evidence_revision") or "").strip().casefold()
        for chunk in chunks
        if re.fullmatch(
            r"[0-9a-f]{64}",
            str((chunk.get("metadata") or {}).get("source_evidence_revision") or "").strip().casefold(),
        )
    }
    evidence_revision_chunk_count = sum(
        1 for chunk in chunks
        if re.fullmatch(
            r"[0-9a-f]{64}",
            str((chunk.get("metadata") or {}).get("source_evidence_revision") or "").strip().casefold(),
        )
    )
    warnings: list[str] = []
    if file_name and Path(file_name).suffix.lower() == ".pdf":
        warnings.append("PDF_TEXT_LAYER_ONLY; OCR_OR_EMBEDDED_IMAGE_TEXT_IS_NOT_EXTRACTED")
    if file_name and Path(file_name).suffix.lower() == ".pptx":
        warnings.append("PPTX_TEXT_SHAPES_ONLY; CHARTS_IMAGES_AND_SPEAKER_NOTES_ARE_NOT_EXTRACTED")
    if extracted_chars == 0:
        warnings.append("EXTRACTION_EMPTY")
    if indexed_chars == 0:
        warnings.append("INDEX_CONTENT_EMPTY")
    if candidate_chunk_count > len(chunks):
        warnings.append("DUPLICATE_CHUNKS_DEDUPLICATED")
    if len(evidence_revisions) != 1 or evidence_revision_chunk_count != len(chunks):
        warnings.append("STRUCTURED_EVIDENCE_REVISION_INCONSISTENT")
    # Chunk overlap intentionally makes indexed_chars larger than extracted
    # chars. The useful loss signal here is the number of sections and chunks,
    # not a misleading character ratio.
    return {
        "file_name": file_name,
        "raw_bytes": raw_bytes,
        "extracted_section_count": len(sections),
        "extracted_page_count": len(pages),
        "extracted_pages": pages[:200],
        "structured_evidence_contract_version": SOURCE_EVIDENCE_PROPAGATION_VERSION,
        "source_evidence_revision": next(iter(evidence_revisions), None) if len(evidence_revisions) == 1 else None,
        "source_evidence_revision_chunk_count": evidence_revision_chunk_count,
        "sections_without_page": sections_without_page,
        "structured_section_count": structured_section_count,
        "extracted_table_count": extracted_table_count,
        "extracted_heading_count": extracted_heading_count,
        "extracted_chars": extracted_chars,
        "chunk_count": len(chunks),
        "candidate_chunk_count": candidate_chunk_count,
        "deduplicated_chunk_count": max(0, candidate_chunk_count - len(chunks)),
        "indexed_chars": indexed_chars,
        "indexed_tokens": sum(int(chunk.get("token_count") or 0) for chunk in chunks),
        "warnings": warnings,
    }
