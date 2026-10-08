"""Source coverage manifest: source facts, table-of-contents filtering and target source refs."""

from __future__ import annotations

import json
import re
from typing import Any

from app.core.config import settings
from app.instructional_quality import TABLE_ROW_RE, clean_source_facts, is_non_instructional_source_line
from app.schemas.chat import RagChatRequest
from app.services.lesson_author.errors import LessonAuthorProposalValidationError
from app.services.retrieval.query import decode_json_object, parse_source_range
from app.services.text import clean_text

MAX_SOURCE_COVERAGE_FACT_CHARS = 420


# This applies only to the legacy/local prompt formatter below. It is never a
# definition of canonical source completeness.
MAX_SOURCE_COVERAGE_MANIFEST_CHARS = 24000


SOURCE_COVERAGE_MARKER_RE = re.compile(
    r"^\s*(?:[•●▪◦*-]\s+|(?:\d+(?:\.\d+)*|[IVXLCDM]+)[.)]\s+)(?P<text>.+?)\s*$",
    re.UNICODE | re.IGNORECASE | re.MULTILINE,
)


SOURCE_COVERAGE_MARKER_ONLY_RE = re.compile(
    r"^\s*(?:[•●▪◦*-]|(?:\d+(?:\.\d+)*|[IVXLCDM]+)[.)])\s*$",
    re.UNICODE | re.IGNORECASE | re.MULTILINE,
)


SOURCE_REF_RE = re.compile(r"\bsrc-\d+\b", re.IGNORECASE)


def is_likely_source_cover_page(text: str) -> bool:
    """Exclude branded cover metadata from the source-content checklist."""
    folded = clean_text(text).casefold()
    if not folded:
        return False
    has_branding = any(
        marker in folded
        for marker in ("www.", "biên soạn bởi", "prepared by", "copyright")
    )
    has_explicit_content = bool(
        SOURCE_COVERAGE_MARKER_RE.search(folded)
        or SOURCE_COVERAGE_MARKER_ONLY_RE.search(folded)
        or "nội dung chương trình" in folded
        or "table of contents" in folded
    )
    return has_branding and not has_explicit_content


def _split_bounded_source_text(text: str) -> list[str]:
    """Split source text without dropping characters at the fact-size limit."""
    remaining = re.sub(r"\s+", " ", text).strip()
    fragments: list[str] = []
    while remaining:
        if len(remaining) <= MAX_SOURCE_COVERAGE_FACT_CHARS:
            fragments.append(remaining)
            break
        window = remaining[: MAX_SOURCE_COVERAGE_FACT_CHARS + 1]
        lower_bound = MAX_SOURCE_COVERAGE_FACT_CHARS // 2
        break_at = max(
            window.rfind(marker, lower_bound, MAX_SOURCE_COVERAGE_FACT_CHARS + 1)
            for marker in (". ", "; ", ": ", ", ", " ")
        )
        if break_at < lower_bound:
            break_at = MAX_SOURCE_COVERAGE_FACT_CHARS
        elif window[break_at: break_at + 2] in {". ", "; ", ": ", ", "}:
            break_at += 1
        fragment = remaining[:break_at].strip()
        if not fragment:
            fragment = remaining[:MAX_SOURCE_COVERAGE_FACT_CHARS]
            break_at = len(fragment)
        fragments.append(fragment)
        remaining = remaining[break_at:].strip()
    return fragments


def _is_wrapped_source_line(previous: str, current: str) -> bool:
    """Return whether an extractor split one sentence across visual lines.

    PDF and slide extractors frequently wrap a sentence at an arbitrary visual
    position. A lower-case continuation is strong evidence of that wrap, while
    a bullet, numbered item, or a new heading must remain a separate fact.
    """
    if not previous or not current:
        return False
    if re.match(r"^\s*(?:[-*•●▪◦]|\d+\s*[.)-])\s+", current):
        return False
    if re.search(r"[.!?;:]\s*$", previous):
        return False
    return current[:1].islower() or bool(re.match(r"^[,;:)\]]", current))


def _merge_wrapped_source_lines(lines: list[str]) -> list[str]:
    """Join visual PDF line wraps without collapsing independently stated facts."""
    merged: list[str] = []
    for line in lines:
        if merged and _is_wrapped_source_line(merged[-1], line):
            merged[-1] = f"{merged[-1]} {line}".strip()
        else:
            merged.append(line)
    return merged


# Phrases that mark a chunk as carrying a table of contents (unchanged trigger), and the extra
# title phrase recognised inside such a chunk.
SOURCE_TOC_TRIGGER_PHRASES = ("nội dung chương trình", "table of contents", "table of content")


SOURCE_TOC_TITLE_PHRASES = (*SOURCE_TOC_TRIGGER_PHRASES, "mục lục")


# A table-of-contents title is a short line; a sentence that merely mentions the phrase is content.
SOURCE_TOC_TITLE_MAX_WORDS = 8


SOURCE_TOC_ENTRY_MAX_CHARS = 120
SOURCE_TOC_LEADER_RE = re.compile(r"(?:\.{3,}|…{2,}|·{3,}|_{3,})\s*\d{1,4}\s*$")
SOURCE_TOC_PAGE_SUFFIX_RE = re.compile(r"\S\s+\d{1,4}$")


SOURCE_TOC_NUMBERED_ENTRY_RE = re.compile(
    r"^(?:(?:chương|chuong|chapter|phần|phan|part|module|modun|bài|bai|lesson)\s*(?:\d{1,3}|[ivxlc]{1,6})\b"
    r"|\d{1,3}(?:\.\d{1,3})*[.)]?\s)",
    re.IGNORECASE,
)


SOURCE_TOC_CHAPTER_LABEL_RE = re.compile(
    r"(?:chương|chuong|chapter|phần|phan|part|module|modun)\s*(?:\d{1,3}|[ivxlc]{1,6})[.:]?",
    re.IGNORECASE,
)


def _is_source_toc_title(line: str) -> bool:
    folded = line.casefold()
    return (any(phrase in folded for phrase in SOURCE_TOC_TITLE_PHRASES)
            and len(line.split()) <= SOURCE_TOC_TITLE_MAX_WORDS)


def _is_source_toc_entry(line: str) -> bool:
    if SOURCE_TOC_LEADER_RE.search(line):
        return True
    if len(line) > SOURCE_TOC_ENTRY_MAX_CHARS or re.search(r"[.!?;:]$", line):
        return False  # a sentence is content, even inside a TOC page
    return bool(line.isdigit() or SOURCE_TOC_NUMBERED_ENTRY_RE.match(line)
                or SOURCE_TOC_PAGE_SUFFIX_RE.search(line)
                or (TABLE_ROW_RE.match(line) and re.search(r"\|\s*\d{1,4}\s*$", line)))


def _drop_source_toc_lines(lines: list[str]) -> list[str]:
    """Remove a table of contents (title, entries, chapter labels) and keep the rest of the page.

    Dropping the whole chunk lost every other fact printed on a TOC page (QC course 234653:
    positioning statement, target learners and the six expected competencies of page 2).
    The TOC region starts at a short title line containing a TOC phrase and continues while
    lines look like entries (dot leaders, page numbers, numbered or chapter entries). After a
    TOC title, dot-leader entries and bare chapter labels ("CHƯƠNG 02") are navigation wherever
    they appear in the chunk; chapter titles and outcome sentences between them stay as content.
    """

    result: list[str] = []
    in_toc = False
    seen_title = False
    for line in lines:
        if _is_source_toc_title(line):
            in_toc = seen_title = True
            continue
        if in_toc and _is_source_toc_entry(line):
            continue
        in_toc = False
        if seen_title and (SOURCE_TOC_LEADER_RE.search(line) or SOURCE_TOC_CHAPTER_LABEL_RE.fullmatch(line)):
            continue
        result.append(line)
    return result


def extract_source_coverage_facts(text: str, *, preserve_table_numeric: bool = False) -> list[str]:
    """Extract bounded semantic facts while repairing visual PDF line wraps."""
    normalized = clean_text(text)
    if not normalized:
        return []
    folded = normalized.casefold()
    lines = [re.sub(r"\s+", " ", raw_line).strip() for raw_line in normalized.splitlines()]
    if any(phrase in folded for phrase in SOURCE_TOC_TRIGGER_PHRASES):
        lines = _drop_source_toc_lines([line for line in lines if line])
    source_lines = [
        line
        for line in lines
        if line
        and not SOURCE_COVERAGE_MARKER_ONLY_RE.fullmatch(line)
        and not is_non_instructional_source_line(line, preserve_table_numeric=preserve_table_numeric)
    ]
    facts: list[str] = []
    for line in _merge_wrapped_source_lines(source_lines):
        facts.extend(_split_bounded_source_text(line))
    return clean_source_facts(facts, preserve_table_numeric=preserve_table_numeric)


def _source_heading_signature(value: Any) -> str:
    text = re.sub(r"\s+", " ", clean_text(str(value or ""))).strip().casefold()
    return re.sub(r"^(?:(?:\d+(?:\.\d+)*|[ivxlcdm]+)[.)]\s*)", "", text).strip()


def _append_unique_source_line(lines: list[str], line: str) -> None:
    value = re.sub(r"\s+", " ", line).strip()
    if not value:
        return
    signature = value.casefold()
    for index, existing in enumerate(lines):
        existing_signature = existing.casefold()
        if signature == existing_signature:
            return
        if len(signature) >= 24 and signature in existing_signature:
            return
        if len(existing_signature) >= 24 and existing_signature in signature:
            lines[index] = value
            return
    lines.append(value)


def _build_unpaginated_source_sections(
    rows: list[dict[str, Any]],
    structure_nodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Map DOCX/HTML chunks to inferred headings and remove chunk overlap."""
    nodes_by_document: dict[str, list[dict[str, Any]]] = {}
    for node in structure_nodes:
        if not isinstance(node, dict):
            continue
        document_id = str(node.get("document_id") or "")
        source_ref = str(node.get("source_ref") or "").strip().casefold()
        signature = _source_heading_signature(node.get("title"))
        if document_id and source_ref and signature:
            nodes_by_document.setdefault(document_id, []).append({
                **node,
                "source_ref": source_ref,
                "heading_signature": signature,
            })
    for nodes in nodes_by_document.values():
        nodes.sort(key=lambda item: (-len(str(item["heading_signature"])), int(item.get("order") or 0)))

    sections: list[dict[str, Any]] = []
    section_indexes: dict[tuple[str, str], int] = {}
    current_ref_by_document: dict[str, str] = {}
    for row in sorted(
        rows,
        key=lambda item: (str(item.get("document_id") or ""), int(item.get("chunk_no") or 0)),
    ):
        if _source_page_number(row.get("source_page")) is not None:
            continue
        document_id = str(row.get("document_id") or "")
        chunk_no = int(row.get("chunk_no") or 0)
        metadata = decode_json_object(row.get("metadata")) or {}
        metadata_ref = str(metadata.get("source_ref") or "").strip().casefold()
        if metadata_ref:
            current_ref_by_document.setdefault(document_id, metadata_ref)
        for raw_line in clean_text(str(row.get("content") or "")).splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip()
            if not line:
                continue
            line_signature = _source_heading_signature(line)
            matched_node = next(
                (
                    node
                    for node in nodes_by_document.get(document_id, [])
                    if len(line_signature) >= 8
                    and (
                        node["heading_signature"] in line_signature
                        or line_signature in node["heading_signature"]
                    )
                ),
                None,
            )
            if matched_node:
                current_ref_by_document[document_id] = str(matched_node["source_ref"])
            source_ref = current_ref_by_document.get(document_id, metadata_ref)
            section_key = source_ref or f"chunk-{chunk_no}"
            index_key = (document_id, section_key)
            section_index = section_indexes.get(index_key)
            if section_index is None:
                section_index = len(sections)
                section_indexes[index_key] = section_index
                sections.append({
                    "document_id": document_id,
                    "document_name": str(row.get("document_name") or "Tài liệu nguồn"),
                    "source_ref": source_ref or None,
                    "source_chunk": chunk_no,
                    "lines": [],
                })
            _append_unique_source_line(sections[section_index]["lines"], line)
    return sections


def _expand_target_source_refs(
    target_source_refs: set[str],
    structure_nodes: list[dict[str, Any]],
) -> set[str]:
    expanded = {value.casefold() for value in target_source_refs if value}
    changed = True
    while changed:
        changed = False
        for node in structure_nodes:
            source_ref = str(node.get("source_ref") or "").strip().casefold()
            parent_ref = str(node.get("parent_source_ref") or "").strip().casefold()
            if source_ref and parent_ref in expanded and source_ref not in expanded:
                expanded.add(source_ref)
                changed = True
    return expanded


def lesson_author_target_source_refs(request: RagChatRequest) -> set[str]:
    if request.target != "lesson_author":
        return set()
    values = "\n".join([
        str(getattr(request, "target_scope_instruction", "") or ""),
        str(getattr(request, "outline_context", "") or ""),
    ])
    return {match.group(0).casefold() for match in SOURCE_REF_RE.finditer(values)}


def lesson_author_blueprint_draft_fact_ids(request: RagChatRequest) -> set[str]:
    """Return the exact persisted fact coverage for a Blueprint chapter draft."""
    architecture = getattr(request, "blueprint_architecture", None)
    if architecture is None:
        return set()
    fact_ids: set[str] = set()
    for lesson in architecture.lessons:
        for unit in lesson.units:
            fact_ids.update(
                fact_id.strip()
                for fact_id in unit.source_fact_ids
                if isinstance(fact_id, str) and fact_id.strip()
            )
    return fact_ids


def lesson_author_blueprint_draft_supporting_fact_ids(request: RagChatRequest) -> set[str]:
    """Return V5 read-only supporting evidence without changing ownership."""
    architecture = getattr(request, "blueprint_architecture", None)
    if architecture is None or architecture.architecture_contract_version != 5:
        return set()
    return {
        fact_id.strip()
        for lesson in architecture.lessons
        for unit in lesson.units
        for fact_id in unit.supporting_evidence_fact_ids
        if isinstance(fact_id, str) and fact_id.strip()
    }


def restrict_blueprint_draft_source_manifest(
    manifest: dict[str, Any],
    expected_fact_ids: set[str],
    supporting_evidence_fact_ids: set[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Keep owned facts and separately retain V5 read-only evidence."""
    supporting_ids = supporting_evidence_fact_ids or set()
    facts = [
        fact
        for fact in manifest.get("facts", [])
        if isinstance(fact, dict) and str(fact.get("fact_id") or "").strip() in expected_fact_ids
    ]
    supporting_facts = [
        fact
        for fact in manifest.get("facts", [])
        if isinstance(fact, dict) and str(fact.get("fact_id") or "").strip() in supporting_ids
    ]
    available_fact_ids = {
        str(fact.get("fact_id") or "").strip()
        for fact in [*facts, *supporting_facts]
    }
    missing_fact_ids = sorted((expected_fact_ids | supporting_ids) - available_fact_ids)
    return {
        **manifest,
        "facts": facts,
        "supporting_evidence_facts": supporting_facts,
        # The source snapshot can have unrelated pages beyond the selected
        # chapter. They do not make this draft incomplete when every assigned
        # fact is present.
        "truncated": bool(missing_fact_ids),
    }, missing_fact_ids


def _resolve_paginated_source_refs(
    rows: list[dict[str, Any]],
    structure_nodes: list[dict[str, Any]],
    target_scopes: list[dict[str, Any]],
) -> set[str]:
    """Return source refs backed by actual page chunks in the selected scope."""
    page_rows = [
        {
            "document_id": str(row.get("document_id") or ""),
            "page": _source_page_number(row.get("source_page")),
        }
        for row in rows
        if clean_text(str(row.get("content") or ""))
    ]
    resolved: set[str] = set()

    def has_chunk(document_id: str, start_page: int, end_page: int) -> bool:
        return any(
            row["page"] is not None
            and (not document_id or row["document_id"] == document_id)
            and start_page <= int(row["page"]) <= end_page
            for row in page_rows
        )

    for scope in target_scopes:
        source_ref = str(scope.get("source_ref") or "").strip().casefold()
        document_id = str(scope.get("document_id") or "")
        start_page = _source_page_number(scope.get("start_page"))
        end_page = _source_page_number(scope.get("end_page"))
        if source_ref and start_page is not None and end_page is not None and has_chunk(document_id, start_page, end_page):
            resolved.add(source_ref)

    for node in structure_nodes:
        source_ref = str(node.get("source_ref") or "").strip().casefold()
        page_range = parse_source_range(str(node.get("title") or ""))
        if not source_ref:
            continue
        # Durable source-structure nodes carry canonical page provenance. A
        # heading does not need to repeat "page 2-4" in its learner-facing
        # title to prove that it belongs to that source page. Prefer the
        # explicit title range only for legacy rows; otherwise resolve the
        # exact node from its stored logical/physical page. Missing page
        # provenance remains unresolved and the caller still validates every
        # server-owned fact ID before generation.
        if page_range is None:
            node_page = _source_page_number(node.get("logical_page"))
            if node_page is None:
                node_page = _source_page_number(node.get("page"))
            if node_page is not None:
                page_range = (node_page, node_page)
        if page_range and has_chunk(str(node.get("document_id") or ""), *page_range):
            resolved.add(source_ref)
    return resolved


def build_source_coverage_manifest(
    rows: list[dict[str, Any]],
    *,
    structure_nodes: list[dict[str, Any]] | None = None,
    target_source_refs: set[str] | None = None,
    target_scopes: list[dict[str, Any]] | None = None,
    canonical_max_chars: int | None = None,
) -> dict[str, Any]:
    """Create a canonical source-fact manifest for the selected source scope.

    The input scope is already bounded by the full-source chunk contract. A
    configurable serialized-character budget then protects request memory.
    We continue enumerating after that budget is exhausted so capacity failure
    is explicit (`total_fact_count` vs `represented_fact_count`) rather than
    silently turning a flat fact count into false completeness.
    """
    structure_nodes = [node for node in (structure_nodes or []) if isinstance(node, dict)]
    requested_refs = {value.casefold() for value in (target_source_refs or set()) if value}
    expanded_refs = _expand_target_source_refs(requested_refs, structure_nodes)
    document_ids = list(dict.fromkeys(
        str(row.get("document_id") or "") for row in rows
    ))
    document_order = {document_id: index + 1 for index, document_id in enumerate(document_ids)}
    multi_document = len(document_ids) > 1
    page_parts: dict[tuple[str, int], list[str]] = {}
    for row in sorted(
        rows,
        key=lambda item: (
            int(item.get("source_page") or 0),
            int(item.get("chunk_no") or 0),
        ),
    ):
        page = row.get("source_page")
        if not isinstance(page, int) or page <= 0:
            continue
        content = clean_text(str(row.get("content") or ""))
        if content:
            page_parts.setdefault((str(row.get("document_id") or ""), page), []).append(content)

    facts: list[dict[str, Any]] = []
    canonical_limit = max(
        1,
        canonical_max_chars
        if isinstance(canonical_max_chars, int)
        else settings.source_coverage_canonical_max_chars,
    )
    canonical_size = 0
    total_fact_count = 0
    canonical_capacity_exceeded = False

    def append_fact(candidate: dict[str, Any]) -> None:
        nonlocal canonical_size, total_fact_count, canonical_capacity_exceeded
        total_fact_count += 1
        estimated_size = (
            len(str(candidate.get("fact_id") or ""))
            + len(str(candidate.get("text") or ""))
            + len(str(candidate.get("source_ref") or ""))
            + 128
        )
        if canonical_size + estimated_size > canonical_limit:
            canonical_capacity_exceeded = True
            return
        facts.append(candidate)
        canonical_size += estimated_size
    first_page_by_document = {
        document_id: min(page for row_document_id, page in page_parts if row_document_id == document_id)
        for document_id, _page in page_parts
    }
    for document_id, page in sorted(page_parts, key=lambda item: (item[0], item[1])):
        page_text = "\n".join(page_parts[(document_id, page)])
        if page == first_page_by_document[document_id] and is_likely_source_cover_page(page_text):
            continue
        prefix = (
            f"d{document_order[document_id]}-p{page}"
            if multi_document
            else f"p{page}"
        )
        for fact_index, text in enumerate(extract_source_coverage_facts(page_text), start=1):
            append_fact({
                "fact_id": f"{prefix}-f{fact_index}",
                "source_page": page,
                "document_id": document_id or None,
                "text": text,
            })

    unpaginated_rows = [
        row for row in rows if _source_page_number(row.get("source_page")) is None
    ]
    sections = _build_unpaginated_source_sections(unpaginated_rows, structure_nodes)
    resolved_refs = {
        str(section.get("source_ref") or "").casefold()
        for section in sections
        if str(section.get("source_ref") or "").strip()
    }
    resolved_refs.update(_resolve_paginated_source_refs(
        rows,
        structure_nodes,
        [scope for scope in (target_scopes or []) if isinstance(scope, dict)],
    ))
    scoped_sections = [
        section
        for section in sections
        if not expanded_refs or str(section.get("source_ref") or "").casefold() in expanded_refs
    ]
    for section_index, section in enumerate(scoped_sections, start=1):
        source_ref = str(section.get("source_ref") or "").strip().casefold()
        chunk_no = int(section.get("source_chunk") or 0)
        locator_prefix = source_ref or f"c{chunk_no + 1}"
        document_id = str(section.get("document_id") or "")
        prefix = (
            f"d{document_order[document_id]}-{locator_prefix}"
            if multi_document
            else locator_prefix
        )
        fact_texts = extract_source_coverage_facts("\n".join(section.get("lines") or []))
        for fact_index, text in enumerate(fact_texts, start=1):
            append_fact({
                "fact_id": f"{prefix}-f{fact_index}",
                "source_page": None,
                "source_chunk": chunk_no,
                "source_ref": source_ref or None,
                "document_id": section.get("document_id"),
                "text": text,
            })

    return {
        "facts": facts,
        "total_fact_count": total_fact_count,
        "represented_fact_count": len(facts),
        "canonical_size_chars": canonical_size,
        "canonical_max_chars": canonical_limit,
        "fact_scope_complete": not canonical_capacity_exceeded,
        "incomplete_reason": "SOURCE_FACT_EXTRACTION_CAPACITY_EXCEEDED" if canonical_capacity_exceeded else None,
        "pages": sorted({page for _document_id, page in page_parts}),
        "chunks": sorted({
            int(section.get("source_chunk") or 0)
            for section in scoped_sections
        }),
        "target_source_refs": sorted(requested_refs),
        "resolved_source_refs": sorted(resolved_refs & expanded_refs) if expanded_refs else sorted(resolved_refs),
        "scope_unresolved": bool(requested_refs - resolved_refs),
        # Compatibility field for existing local-drafting callers. It now
        # means a genuine canonical capacity failure, never "more than N
        # facts" or a formatter/prompt detail limit.
        "truncated": canonical_capacity_exceeded,
    }


def format_source_coverage_manifest(manifest: dict[str, Any] | None) -> str:
    if not manifest:
        return ""
    facts = manifest.get("facts") if isinstance(manifest.get("facts"), list) else []
    if not facts:
        return ""
    lines = [
        "MANDATORY SOURCE COVERAGE CHECKLIST (internal provenance IDs):",
        "Mọi fact_id phải được gán vào ít nhất một unit qua source_fact_ids. Không dùng source_refs để thay thế checklist này.",
    ]
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        fact_id = str(fact.get("fact_id") or "").strip()
        text = re.sub(r"\s+", " ", str(fact.get("text") or "")).strip()
        page = _source_page_number(fact.get("source_page"))
        if not fact_id or not text:
            continue
        source_ref = str(fact.get("source_ref") or "").strip()
        chunk_no = fact.get("source_chunk")
        if page is not None:
            locator = f"trang/slide {page}"
        elif source_ref:
            locator = f"mục nguồn {source_ref}"
        else:
            locator = f"đoạn nguồn {int(chunk_no or 0) + 1}"
        line = f"- [{fact_id}] {locator}: {text}"
        if sum(len(item) + 1 for item in lines) + len(line) > MAX_SOURCE_COVERAGE_MANIFEST_CHARS:
            lines.append("- [MANIFEST_TRUNCATED] Phần còn lại phải được xử lý ở lượt tiếp theo; không được tuyên bố đã bao phủ toàn bộ nguồn.")
            break
        lines.append(line)
    return "\n".join(lines)


def format_course_architecture_coverage_contract(manifest: dict[str, Any] | None) -> str:
    """Describe complete canonical coverage without flattening fact text to Gemini.

    Course Architect receives the bounded hierarchical Source Map separately.
    It designs semantic architecture; the server later assigns every canonical
    fact only where document/section/concept ownership supports that mapping.
    """
    if not manifest:
        return ""
    total = manifest.get("total_fact_count")
    represented = manifest.get("represented_fact_count")
    if not isinstance(total, int):
        total = len(manifest.get("facts") or [])
    if not isinstance(represented, int):
        represented = len(manifest.get("facts") or [])
    payload = {
        "canonical_source_fact_count": total,
        "canonical_represented_fact_count": represented,
        "fact_scope_complete": bool(manifest.get("fact_scope_complete", not manifest.get("truncated"))),
        "allocation_policy": "The server assigns canonical source_fact_ids after architecture using source-section, concept, source-reference and semantic-block ownership. Do not invent fact IDs or enumerate the whole canonical manifest.",
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def collect_lesson_author_source_fact_ids(proposal: dict[str, Any]) -> set[str]:
    collected: set[str] = set()

    def collect(value: Any) -> None:
        if isinstance(value, str):
            if value.strip():
                collected.add(value.strip())
            return
        if isinstance(value, list):
            for item in value:
                collect(item)
            return
        if isinstance(value, dict):
            fact_id = value.get("fact_id")
            if isinstance(fact_id, str) and fact_id.strip():
                collected.add(fact_id.strip())

    for chapter in proposal.get("chapters", []) if isinstance(proposal.get("chapters"), list) else []:
        if not isinstance(chapter, dict):
            continue
        for lesson in chapter.get("lessons", []) if isinstance(chapter.get("lessons"), list) else []:
            if not isinstance(lesson, dict):
                continue
            for unit in lesson.get("units", []) if isinstance(lesson.get("units"), list) else []:
                if not isinstance(unit, dict):
                    continue
                collect(unit.get("source_fact_ids"))
                collect(unit.get("source_coverage"))
                for component in unit.get("components", []) if isinstance(unit.get("components"), list) else []:
                    if isinstance(component, dict):
                        collect(component.get("source_fact_ids"))
                        collect(component.get("source_coverage"))
    collect(proposal.get("source_coverage"))
    return collected


def source_coverage_metrics(
    proposal: dict[str, Any],
    manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    required = {
        str(fact.get("fact_id"))
        for fact in (manifest or {}).get("facts", [])
        if isinstance(fact, dict) and str(fact.get("fact_id") or "").strip()
    }
    declared = collect_lesson_author_source_fact_ids(proposal)
    covered = required & declared
    missing = sorted(required - declared)
    invalid = sorted(declared - required)
    return {
        "required_count": len(required),
        "covered_count": len(covered),
        "coverage_ratio": round(len(covered) / len(required), 4) if required else None,
        "missing_fact_ids": missing[:40],
        "invalid_fact_ids": invalid[:40],
        "status": "not_applicable" if not required else "complete" if not missing and not invalid else "incomplete",
    }


def validate_lesson_author_source_coverage(
    proposal: dict[str, Any],
    manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    metrics = source_coverage_metrics(proposal, manifest)
    if metrics["status"] == "incomplete":
        missing = ", ".join(metrics["missing_fact_ids"][:12]) or "none"
        invalid = ", ".join(metrics["invalid_fact_ids"][:12]) or "none"
        raise LessonAuthorProposalValidationError(
            "Nguồn chưa được bao phủ đầy đủ. Thiếu fact_id: "
            f"{missing}; fact_id không hợp lệ: {invalid}."
        )
    return metrics


def _source_page_number(value: Any) -> int | None:
    try:
        page = int(value)
    except (TypeError, ValueError):
        return None
    return page if page > 0 else None


def _source_chunk_number(value: Any) -> int | None:
    try:
        chunk = int(value)
    except (TypeError, ValueError):
        return None
    return chunk if chunk >= 0 else None
