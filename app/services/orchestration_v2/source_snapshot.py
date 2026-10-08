"""Orchestration-v2 source snapshot: one deterministic source page per request, no provider call."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any, Literal

import asyncpg

from app.core.logging import SERVICE_LOGGER_NAME
from app.instructional_density import (
    INSTRUCTIONAL_DENSITY_POLICY_VERSION,
    SOURCE_SCOPE_CHUNKS_PER_GROUP,
    partition_chunk_facts_for_density,
)
from app.lesson_author_orchestration_v2 import canonical_hash as orchestration_v2_canonical_hash
from app.lesson_author_orchestration_v2_provider import (
    SourceOutlineAuthorityV2,
    SourceOutlineChapterV2,
    SourceSnapshotFactV2,
)
from app.repositories import source_snapshot as snapshot_repository
from app.schemas.common import AiUsage
from app.schemas.orchestration_v2 import RagLessonAuthorSourceSnapshotV2Request
from app.services.ingestion.chunking import SOURCE_EVIDENCE_LEGACY_REVIEW_REQUIRED, SOURCE_EVIDENCE_READY
from app.services.orchestration_v2.common import _orchestration_v2_http_error
from app.services.provider import normalize_embedding_model
from app.services.retrieval.query import decode_json_object
from app.services.retrieval.source_coverage import extract_source_coverage_facts
from app.services.text import clean_text
from app.source_chapter_policy import resolve_source_chapter_policy

logger = logging.getLogger(SERVICE_LOGGER_NAME)


async def lesson_author_orchestration_v2_source_snapshot(
    request: RagLessonAuthorSourceSnapshotV2Request,
    pool: asyncpg.Pool,
) -> dict[str, Any]:
    """Return one deterministic source page without a provider call or whole-source materialization."""

    document_ids = sorted(document.document_id for document in request.source_documents)
    cursor = request.cursor
    if cursor is not None and str(cursor["document_id"]) not in set(document_ids):
        raise _orchestration_v2_http_error(
            "SOURCE_CURSOR_INVALID", "The source cursor is outside the selected source authority.",
        )
    indexes = await snapshot_repository.fetch_learned_indexes(
        pool,
        request.tenant_id,
        request.kb_id,
        document_ids,
        normalize_embedding_model(request.embedding_model),
        request.embedding_dimensions,
    )
    if len(indexes) != len(document_ids) or {str(row["document_id"]) for row in indexes} != set(document_ids):
        raise _orchestration_v2_http_error(
            "SOURCE_REVISION_UNAVAILABLE", "The selected learned source revision is unavailable.",
        )
    structure_rows = await snapshot_repository.fetch_index_evidence_summary(
        pool,
        request.tenant_id,
        request.kb_id,
        [str(row["index_id"]) for row in indexes],
    )
    structures_by_document = {
        str(row["document_id"]): decode_json_object(row["source_structure"])
        for row in structure_rows
        if row.get("source_structure") is not None
    }
    structure_rows_by_document = {str(row["document_id"]): row for row in structure_rows}
    evidence_revisions_by_document: dict[str, str] = {}
    evidence_status_by_document: dict[str, str] = {}
    for index_row in indexes:
        document_id = str(index_row["document_id"])
        declared_chunks = int(index_row["chunk_count"] or 0)
        summary = structure_rows_by_document.get(document_id) or {}
        revision = str(summary.get("source_evidence_revision") or "").strip().casefold()
        valid_revision = revision if re.fullmatch(r"[0-9a-f]{64}", revision) else None
        actual_chunks = int(summary.get("actual_chunk_count") if summary.get("actual_chunk_count") is not None
                            else declared_chunks)
        revision_chunks = int(summary.get("evidence_revision_chunk_count")
                              if summary.get("evidence_revision_chunk_count") is not None
                              else declared_chunks if valid_revision else 0)
        revision_count = int(summary.get("evidence_revision_distinct_count")
                             if summary.get("evidence_revision_distinct_count") is not None
                             else 1 if valid_revision else 0)
        if declared_chunks < 1 or actual_chunks != declared_chunks:
            raise _orchestration_v2_http_error(
                "SOURCE_EVIDENCE_REVISION_INCONSISTENT",
                "The learned source index does not match its declared chunk inventory.",
                status_code=409,
            )
        if revision_chunks == 0 and revision_count == 0 and valid_revision is None:
            evidence_status_by_document[document_id] = SOURCE_EVIDENCE_LEGACY_REVIEW_REQUIRED
        elif (revision_chunks == declared_chunks and revision_count == 1
              and valid_revision is not None):
            evidence_status_by_document[document_id] = SOURCE_EVIDENCE_READY
            evidence_revisions_by_document[document_id] = valid_revision
        else:
            raise _orchestration_v2_http_error(
                "SOURCE_EVIDENCE_REVISION_INCONSISTENT",
                "The learned source index contains mixed structured-evidence revisions.",
                status_code=409,
            )
    policy_documents = [{
        "document_id": document_id,
        "structure": structures_by_document.get(document_id),
    } for document_id in document_ids]
    chapter_policy = resolve_source_chapter_policy(policy_documents)
    policy_mode = str(chapter_policy.get("mode") or "NEEDS_STRUCTURE_REVIEW")
    authority_mode: Literal["locked", "model_designed", "needs_review"] = (
        "locked" if policy_mode.startswith("SOURCE_LOCKED")
        else "model_designed" if policy_mode == "MODEL_DESIGNED"
        else "needs_review"
    )
    authority_source: Literal["toc", "headings", "none", "ambiguous"] = (
        "toc" if policy_mode == "SOURCE_LOCKED_TOC"
        else "headings" if policy_mode == "SOURCE_LOCKED_HEADINGS"
        else "none" if authority_mode == "model_designed"
        else "ambiguous"
    )
    chapters = [SourceOutlineChapterV2(
        order=index,
        document_id=str(chapter["document_id"]),
        source_ref=str(chapter["source_ref"]),
        title=str(chapter["title"]),
    ) for index, chapter in enumerate(chapter_policy.get("chapters", []))] if authority_mode == "locked" else []
    confidences = [float(structure.get("confidence") or 0) for structure in structures_by_document.values()
                   if isinstance(structure, dict)]
    reason_codes = [str(value)[:120] for value in chapter_policy.get("reason_codes", [])[:32]]
    legacy_document_count = sum(
        status == SOURCE_EVIDENCE_LEGACY_REVIEW_REQUIRED
        for status in evidence_status_by_document.values()
    )
    if legacy_document_count and "STRUCTURED_EVIDENCE_REVISION_MISSING" not in reason_codes:
        reason_codes = [*reason_codes[:31], "STRUCTURED_EVIDENCE_REVISION_MISSING"]
        logger.warning("lesson_author_orchestration_v2 %s", json.dumps({
            "event": "source_snapshot_legacy_evidence_review_required",
            "correlation_id": request.correlation_id,
            "source_snapshot_hash": request.source_snapshot_hash,
            "legacy_document_count": legacy_document_count,
            "selected_document_count": len(document_ids),
        }, sort_keys=True))
    authority_payload = {
        "mode": authority_mode,
        "source": authority_source,
        "complete": bool(chapter_policy.get("complete")),
        "confidence": min(confidences) if confidences else 0.0,
        "reason_codes": reason_codes,
        "chapters": [chapter.model_dump(mode="json") for chapter in chapters],
    }
    source_authority = SourceOutlineAuthorityV2(
        **authority_payload,
        structure_hash=orchestration_v2_canonical_hash(authority_payload),
    )
    revision_payload: list[dict[str, Any]] = []
    for row in indexes:
        document_id = str(row["document_id"])
        revision_entry: dict[str, Any] = {
            "document_id": document_id,
            "index_id": str(row["index_id"]),
            "content_sha256": str(row["content_sha256"] or ""),
            "chunk_count": int(row["chunk_count"] or 0),
            "embedding_model": str(row["embedding_model"]),
            "embedding_dimensions": int(row["embedding_dimensions"]),
            "source_evidence_status": evidence_status_by_document[document_id],
        }
        # Preserve the exact legacy revision for already learned documents.
        # A new revision is materialized only after re-indexing has produced
        # explicit structured-evidence metadata.
        evidence_revision = evidence_revisions_by_document.get(document_id)
        if evidence_revision is not None:
            revision_entry["source_evidence_revision"] = evidence_revision
        revision_payload.append(revision_entry)
    source_revision = orchestration_v2_canonical_hash(revision_payload)
    if request.expected_source_revision and request.expected_source_revision != source_revision:
        raise _orchestration_v2_http_error(
            "SOURCE_REVISION_CHANGED", "The learned source revision changed while paging.", status_code=409,
        )

    index_ids = [str(row["index_id"]) for row in indexes]
    after_document_id = str(cursor["document_id"]) if cursor else None
    after_chunk_no = int(cursor["chunk_no"]) if cursor else -1
    row_limit = 128
    fetched = await snapshot_repository.fetch_source_page(
        pool,
        request.tenant_id,
        request.kb_id,
        index_ids,
        after_document_id,
        after_chunk_no,
        row_limit + 1,
    )
    document_order = {document_id: index + 1 for index, document_id in enumerate(document_ids)}
    facts: list[SourceSnapshotFactV2] = []
    content_bytes = 0
    last_cursor: dict[str, Any] | None = None
    has_more = len(fetched) > row_limit
    for raw_row in fetched[:row_limit]:
        row = dict(raw_row)
        document_id = str(row["document_id"])
        chunk_no = int(row["chunk_no"])
        metadata = decode_json_object(row.get("metadata")) or {}
        content_kinds = {
            str(value).strip().casefold()
            for value in metadata.get("content_kinds", [])
            if isinstance(value, str) and value.strip()
        } if isinstance(metadata.get("content_kinds"), list) else set()
        table_count = max(0, min(100, int(metadata.get("table_count") or 0))) \
            if str(metadata.get("table_count") or "0").isdigit() else 0
        fact_texts = extract_source_coverage_facts(
            clean_text(str(row.get("content") or "")),
            preserve_table_numeric=table_count > 0 or "table" in content_kinds,
        )
        page = row.get("source_page") if isinstance(row.get("source_page"), int) and row.get("source_page") > 0 else None
        source_ref = str(metadata.get("source_ref") or row.get("source_section") or "").strip()[:255] or None
        heading = metadata.get("heading_path")
        if isinstance(heading, list):
            heading = " › ".join(
                str(value).strip()[:120] for value in heading[:8] if str(value).strip()
            )
        scope_title = str(heading or row.get("source_section") or "").strip()
        if not scope_title:
            suffix = f" · page {page}" if page is not None else f" · chunk {chunk_no + 1}"
            scope_title = f"{str(row.get('document_name') or 'Source document')}{suffix}"
        scope_title = scope_title[:500]
        visual_regions = metadata.get("visual_regions")
        if not isinstance(visual_regions, list):
            visual_regions = []
        visual_regions = [region for region in visual_regions[:4] if isinstance(region, dict)]
        visual_prompt_text = str(metadata.get("visual_prompt_text") or "").strip()[:600] or None
        # Scope V3 is a deterministic density lane. Four adjacent chunks share
        # one scope only when their same-index fact partitions remain within a
        # complete instructional-unit budget. This is pagination independent:
        # no cross-request mutable counter can change scope identity.
        scope_group = chunk_no // SOURCE_SCOPE_CHUNKS_PER_GROUP
        if source_ref:
            scope_owner = f"source-ref:{source_ref}"
        else:
            scope_owner = "title:" + hashlib.sha256(scope_title.encode("utf-8")).hexdigest()[:24]
        candidates: list[SourceSnapshotFactV2] = []
        fact_index = 0
        try:
            fact_partitions = partition_chunk_facts_for_density(fact_texts)
        except ValueError as error:
            raise _orchestration_v2_http_error(
                "SOURCE_FACT_DENSITY_INVALID",
                "One canonical source fact exceeds the instructional density contract.",
            ) from error
        for lane_index, partition in enumerate(fact_partitions):
            scope_locator = f"{scope_owner}:chunk-group:{scope_group}:lane:{lane_index}"
            scope_key = "scope3_" + hashlib.sha256(
                f"{document_id}\x1e{scope_locator}".encode("utf-8"),
            ).hexdigest()[:32]
            # Chunk metadata describes every structure found anywhere in the
            # chunk. Density partitioning creates independent scopes, so
            # copying a chunk-level table marker onto a prose-only lane makes
            # downstream contracts require a table that cannot be faithfully
            # reconstructed from that lane. Keep only structure represented by
            # the canonical facts in this partition.
            partition_table_row_count = sum(
                bool(re.match(r"^Row\s+\d+\s*:\s*.+\|.+$", text, re.IGNORECASE))
                for text in partition
            )
            partition_has_table = partition_table_row_count >= 2
            partition_content_kinds = set(content_kinds)
            if partition_has_table:
                partition_content_kinds.add("table")
            else:
                partition_content_kinds.discard("table")
            partition_table_count = max(1, table_count) if partition_has_table else 0
            for text in partition:
                fact_index += 1
                candidates.append(SourceSnapshotFactV2(
                    document_id=document_id,
                    fact_key=f"d{document_order[document_id]}-c{chunk_no + 1}-f{fact_index}",
                    scope_key=scope_key,
                    fact_text=text,
                    source_ref=source_ref,
                    source_page=page,
                    source_chunk=chunk_no,
                    locator={
                        "index_id": str(row["index_id"]),
                        "source_revision": source_revision,
                        "scope_title": scope_title,
                        "parser_version": str(metadata.get("parser_version") or "")[:80] or None,
                        "content_kinds": sorted(partition_content_kinds)[:8],
                        "table_count": partition_table_count,
                        "source_evidence_revision": (
                            str(metadata.get("source_evidence_revision") or "").strip().casefold()
                            if re.fullmatch(
                                r"[0-9a-f]{64}",
                                str(metadata.get("source_evidence_revision") or "").strip().casefold(),
                            )
                            else None
                        ),
                        "source_evidence_status": evidence_status_by_document[document_id],
                        "structured_evidence_contract_version": str(
                            metadata.get("structured_evidence_contract_version") or ""
                        )[:80] or None,
                        "visual_regions": visual_regions,
                        "visual_prompt_text": visual_prompt_text,
                        "instructional_density_policy_version": INSTRUCTIONAL_DENSITY_POLICY_VERSION,
                        "scope_partition_lane": lane_index,
                    },
                ))
        candidate_bytes = sum(len(fact.fact_text.encode("utf-8")) for fact in candidates)
        exceeds = (len(facts) + len(candidates) > request.page_max_facts
                   or content_bytes + candidate_bytes > request.page_max_bytes)
        if exceeds and facts:
            has_more = True
            break
        if exceeds:
            raise _orchestration_v2_http_error(
                "SOURCE_CHUNK_EXCEEDS_PAGE", "One source chunk exceeds the bounded page contract.",
            )
        facts.extend(candidates)
        content_bytes += candidate_bytes
        last_cursor = {"document_id": document_id, "chunk_no": chunk_no}

    if has_more and last_cursor is None:
        raise _orchestration_v2_http_error(
            "SOURCE_CURSOR_STALLED", "The source cursor could not advance.",
        )
    facts_wire = [fact.model_dump(mode="json") for fact in facts]
    logger.info("lesson_author_orchestration_v2 %s", json.dumps({
        "event": "source_snapshot_page_ready", "correlation_id": request.correlation_id,
        "source_snapshot_hash": request.source_snapshot_hash, "source_revision": source_revision,
        "source_fact_count": len(facts_wire), "source_content_bytes": content_bytes,
        "source_evidence_ready_document_count": len(document_ids) - legacy_document_count,
        "source_evidence_legacy_document_count": legacy_document_count,
        "has_more": has_more, "provider_call_count": 0,
    }, sort_keys=True))
    return {
        "contract_version": 2,
        "source_snapshot_hash": request.source_snapshot_hash,
        "source_revision": source_revision,
        "source_authority": source_authority.model_dump(mode="json"),
        "facts": facts_wire,
        "next_cursor": last_cursor if has_more else None,
        "has_more": has_more,
        "page_content_bytes": content_bytes,
        "usage": AiUsage().model_dump(),
    }
