"""Query normalization, keyword patterns, retrieval limits and row merging."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import asyncpg

from app.core.config import settings
from app.schemas.chat import RagChatRequest
from app.services.text import clean_text
from app.source_structure import strip_source_range_suffix

SOURCE_RANGE_RE = re.compile(
    r"\b(?:từ|tu|from)\s+(?:slide|slides|trang|page|pages)\s+(\d+)\s+"
    r"(?:đến|den|to)\s+(?:(?:slide|slides|trang|page|pages)\s+)?(\d+)\b",
    flags=re.IGNORECASE,
)


KEYWORD_STOPWORDS = {
    "anh",
    "ban",
    "bang",
    "bằng",
    "bao",
    "bi",
    "biet",
    "biết",
    "cac",
    "các",
    "cho",
    "cua",
    "của",
    "duoc",
    "được",
    "gi",
    "gì",
    "gioi",
    "giới",
    "hay",
    "hien",
    "hiện",
    "hoi",
    "hỏi",
    "khong",
    "không",
    "la",
    "là",
    "lai",
    "lại",
    "mot",
    "một",
    "nay",
    "này",
    "nhung",
    "những",
    "noi",
    "nói",
    "tao",
    "the",
    "thế",
    "thi",
    "thì",
    "thong",
    "thông",
    "tin",
    "toi",
    "tôi",
    "trong",
    "ve",
    "về",
    "voi",
    "với",
    "you",
    "your",
    "the",
    "and",
    "for",
    "from",
    "that",
    "this",
    "what",
    "about",
    "please",
}


def normalize_query_text(value: str) -> str:
    return clean_text(value).strip()


def make_like_pattern(value: str) -> str:
    cleaned = normalize_query_text(value).replace("%", " ").replace("*", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return f"%{cleaned}%" if cleaned else ""


def build_keyword_patterns(value: str) -> tuple[list[str], list[str], list[str]]:
    raw = normalize_query_text(value)
    normalized = normalize_query_text(raw.replace("_", " ").replace("-", " ").replace(".", " "))

    phrase_patterns: list[str] = []
    for candidate in [raw, normalized]:
        if len(candidate) >= 3:
            pattern = make_like_pattern(candidate)
            if pattern and pattern not in phrase_patterns:
                phrase_patterns.append(pattern)

    terms: list[str] = []
    seen: set[str] = set()
    token_text = f"{raw} {normalized}".lower()
    for token in re.findall(r"[0-9A-Za-zÀ-ỹ_.-]{3,}", token_text, flags=re.UNICODE):
        candidates = [token]
        candidates.extend(part for part in re.split(r"[_\-.]+", token) if part)
        for candidate in candidates:
            cleaned = candidate.strip("._- ").lower()
            if len(cleaned) < 3 or cleaned in KEYWORD_STOPWORDS or cleaned in seen:
                continue
            seen.add(cleaned)
            terms.append(cleaned)
            if len(terms) >= 12:
                break
        if len(terms) >= 12:
            break

    term_patterns = [make_like_pattern(term) for term in terms]
    return phrase_patterns, [pattern for pattern in term_patterns if pattern], terms


def retrieval_limits(request: RagChatRequest) -> dict[str, int]:
    if request.target == "lesson_author":
        return {
            "top_k": max(1, settings.lesson_author_top_k),
            "max_context_chars": max(1, settings.lesson_author_max_context_chars),
            "max_chunks_per_document": max(1, settings.lesson_author_max_chunks_per_document),
        }
    return {
        "top_k": max(1, settings.top_k),
        "max_context_chars": max(1, settings.max_context_chars),
        "max_chunks_per_document": max(1, settings.retrieval_max_chunks_per_document),
    }


def retrieval_candidate_limit(request: RagChatRequest) -> int:
    multiplier = max(1, settings.retrieval_candidate_multiplier)
    top_k = retrieval_limits(request)["top_k"]
    return max(top_k, top_k * multiplier)


def build_retrieval_query_texts(request: RagChatRequest) -> list[str]:
    """Add authoring scope hints to semantic retrieval without changing chat Q&A."""
    values = [request.user_message]
    if request.target == "lesson_author":
        values.extend([
            getattr(request, "outline_context", ""),
            getattr(request, "target_scope_instruction", ""),
        ])
    queries: list[str] = []
    seen: set[str] = set()
    for value in values:
        query = normalize_query_text(str(value or ""))
        if not query:
            continue
        query = query[:6000]
        signature = retrieval_text_signature(query)
        if signature in seen:
            continue
        seen.add(signature)
        queries.append(query)
    return queries or [request.user_message]


def parse_source_range(value: str) -> tuple[int, int] | None:
    match = SOURCE_RANGE_RE.search(value or "")
    if not match:
        return None
    start, end = int(match.group(1)), int(match.group(2))
    return (start, end) if start <= end else (end, start)


def build_target_source_scopes(
    request: RagChatRequest,
    structure_context: dict[str, Any],
) -> list[dict[str, Any]]:
    """Resolve a selected authoring chapter to its explicit TOC page range."""
    if (
        request.target != "lesson_author"
        or structure_context.get("structure_source") != "toc"
        or not getattr(request, "target_scope_instruction", "").strip()
    ):
        return []
    user_scope_text = normalize_query_text(
        " ".join(
            [
                request.user_message,
                getattr(request, "target_scope_instruction", ""),
            ]
        )
    ).casefold()
    outline_scope_text = normalize_query_text(getattr(request, "outline_context", "")).casefold()
    haystack = " ".join(filter(None, [user_scope_text, outline_scope_text])).casefold()
    query_terms = set(re.findall(r"[0-9A-Za-zÀ-ỹ]{3,}", haystack, flags=re.UNICODE))
    authoritative_nodes = [
        node
        for node in (structure_context.get("authoritative_source_nodes", []) or [])
        if isinstance(node, dict) and str(node.get("title") or "").strip()
    ]
    authoritative_nodes.sort(
        key=lambda node: (
            int(node.get("logical_page") or node.get("page") or 0),
            int(node.get("order") or 0),
            str(node.get("source_ref") or ""),
        ),
    )

    # The current user turn is the strongest identifier. This prevents a
    # full outline pasted into context from making every TOC title appear to
    # match the request.
    chapter_match = re.search(
        r"\b(?:chương|chuong|chapter)\s*(?:số\s*|so\s*)?(\d+)\b",
        user_scope_text,
        flags=re.IGNORECASE,
    )
    if chapter_match:
        chapter_index = int(chapter_match.group(1))
        if 1 <= chapter_index <= len(authoritative_nodes):
            node = authoritative_nodes[chapter_index - 1]
            raw_title = str(node.get("title") or "").strip()
            page_range = parse_source_range(raw_title)
            if page_range:
                return [{
                    "document_id": str(node.get("document_id") or ""),
                    "document_name": str(node.get("document_name") or "Tài liệu nguồn"),
                    "source_ref": str(node.get("source_ref") or "") or None,
                    "title": strip_source_range_suffix(raw_title).strip(),
                    "start_page": page_range[0],
                    "end_page": page_range[1],
                }]

    ranked_matches: list[tuple[float, dict[str, Any]]] = []
    for node in authoritative_nodes:
        raw_title = str(node.get("title") or "").strip()
        semantic_title = strip_source_range_suffix(raw_title).strip()
        if not semantic_title:
            continue
        title_fold = normalize_query_text(semantic_title).casefold()
        title_terms = set(re.findall(r"[0-9A-Za-zÀ-ỹ]{3,}", title_fold, flags=re.UNICODE))
        overlap = len(title_terms & query_terms) / max(1, len(title_terms))
        if title_fold not in haystack and overlap < 0.6:
            continue
        page_range = parse_source_range(raw_title)
        if not page_range:
            continue
        ranked_matches.append(
            (
                overlap + (0.25 if title_fold in haystack else 0),
                {
                    "document_id": str(node.get("document_id") or ""),
                    "document_name": str(node.get("document_name") or "Tài liệu nguồn"),
                    "source_ref": str(node.get("source_ref") or "") or None,
                    "title": semantic_title,
                    "start_page": page_range[0],
                    "end_page": page_range[1],
                },
            )
        )
    ranked_matches.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in ranked_matches[:1]]


def retrieval_text_signature(value: str) -> str:
    compact = re.sub(r"\s+", " ", clean_text(value).lower()).strip()
    return hashlib.sha256(compact[:1600].encode("utf-8")).hexdigest()


def row_passes_retrieval_threshold(row: dict[str, Any]) -> bool:
    score = float(row.get("score") or 0)
    vector_score = float(row.get("vector_score") or 0)
    keyword_score = float(row.get("keyword_score") or 0)
    if keyword_score >= settings.retrieval_keyword_min_score:
        return True
    if max(score, vector_score) >= settings.retrieval_min_score:
        return True
    return False


def apply_retrieval_quality_controls(
    rows: list[dict[str, Any]],
    limit: int,
    *,
    max_chunks_per_document: int | None = None,
) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    chunks_per_document: dict[str, int] = {}
    seen_content: set[str] = set()
    max_chunks_per_document = max_chunks_per_document or settings.retrieval_max_chunks_per_document
    max_chunks_per_document = max(1, max_chunks_per_document)

    for row in rows:
        if not row_passes_retrieval_threshold(row):
            continue
        document_id = str(row.get("document_id") or "")
        if chunks_per_document.get(document_id, 0) >= max_chunks_per_document:
            continue
        signature = retrieval_text_signature(str(row.get("content") or ""))
        if signature in seen_content:
            continue

        seen_content.add(signature)
        chunks_per_document[document_id] = chunks_per_document.get(document_id, 0) + 1
        accepted.append(row)
        if len(accepted) >= limit:
            break

    return accepted


def merge_retrieval_rows(
    vector_rows: list[asyncpg.Record],
    keyword_rows: list[asyncpg.Record],
    limit: int,
    scope_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}

    def add_row(row: asyncpg.Record) -> None:
        item = dict(row)
        key = f"{item.get('document_id')}:{item.get('chunk_no')}"
        vector_score = float(item.get("vector_score") or 0)
        keyword_score = float(item.get("keyword_score") or 0)
        score = max(float(item.get("score") or 0), vector_score, keyword_score)
        method = str(item.get("method") or "unknown")
        existing = merged.get(key)
        if not existing:
            item["score"] = score
            item["vector_score"] = vector_score
            item["keyword_score"] = keyword_score
            item["methods"] = [method]
            merged[key] = item
            return

        existing["score"] = max(float(existing.get("score") or 0), score)
        existing["vector_score"] = max(float(existing.get("vector_score") or 0), vector_score)
        existing["keyword_score"] = max(float(existing.get("keyword_score") or 0), keyword_score)
        methods = set(existing.get("methods") or [])
        methods.add(method)
        existing["methods"] = sorted(methods)
        existing["method"] = "hybrid" if len(methods) > 1 else next(iter(methods))

    for row in vector_rows:
        add_row(row)
    for row in keyword_rows:
        add_row(row)
    for row in scope_rows or []:
        add_row(row)

    return sorted(
        merged.values(),
        key=lambda item: (
            float(item.get("score") or 0),
            float(item.get("keyword_score") or 0),
            float(item.get("vector_score") or 0),
        ),
        reverse=True,
    )[:limit]


def decode_json_object(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def decode_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []
