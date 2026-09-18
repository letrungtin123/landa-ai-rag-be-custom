from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable


PARSER_VERSION = "source-structure-v2"
MAX_NODE_TITLE_CHARS = 220
MAX_NODES = 400
TOC_MARKERS = (
    "mục lục",
    "muc luc",
    "table of contents",
    "contents",
    "nội dung chương trình",
    "noi dung chuong trinh",
)
ARABIC_HEADING_RE = re.compile(
    r"^\s*((?:\d+\.)*\d+)[.)]?\s+(.{3,220}?)\s*$",
    re.UNICODE,
)
ROMAN_HEADING_RE = re.compile(
    r"^\s*([IVXLCDM]{1,8})[.)]\s+(.{3,220}?)\s*$",
    re.UNICODE,
)
TOC_LINE_RE = re.compile(
    r"^\s*(.+?)(?:\.{2,}|\s{2,})\s*(\d{1,4})\s*$",
    re.UNICODE,
)
TOC_PAGE_ONLY_RE = re.compile(r"^\s*(\d{1,4})\s*$", re.UNICODE)
NUMBERED_ENTRY_RE = re.compile(
    r"^\s*((?:\d+\.)*\d+|[IVXLCDM]{1,8})(?:[.)]\s*|\s{2,})(.*)$",
    re.IGNORECASE | re.UNICODE,
)
SLIDE_RANGE_RE = re.compile(
    r"\btừ\s+(?:slide|trang)\s+(\d+)\s+đến\s+(?:slide|trang)\s+(\d+)\b",
    re.IGNORECASE | re.UNICODE,
)
SOURCE_RANGE_SUFFIX_RE = re.compile(
    r"\s*\(\s*(?:(?:từ|tu|from)\s+)?(?:slides?|trang|pages?)\s+\d{1,4}\s*"
    r"(?:(?:đến|den|to)\s+(?:(?:slides?|trang|pages?)\s+)?\d{1,4}"
    r"|[-\u2013\u2014]\s*(?:(?:slides?|trang|pages?)\s+)?\d{1,4})\s*\)\s*$",
    re.IGNORECASE | re.UNICODE,
)
CHAPTER_HEADING_RE = re.compile(
    r"^\s*(chương|chuong|chapter|phần|phan|part)\s+([0-9IVXLCDM]+)\s*[:.)-]?\s*(.*?)\s*$",
    re.IGNORECASE | re.UNICODE,
)


@dataclass(frozen=True)
class SourceStructureNode:
    source_ref: str
    title: str
    level: int
    order: int
    page: int | None = None
    logical_page: int | None = None
    number_label: str | None = None
    parent_source_ref: str | None = None
    node_type: str = "section"
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_ref": self.source_ref,
            "title": self.title,
            "level": self.level,
            "order": self.order,
            "page": self.page,
            "logical_page": self.logical_page,
            "number_label": self.number_label,
            "parent_source_ref": self.parent_source_ref,
            "node_type": self.node_type,
            "confidence": round(self.confidence, 4),
        }


def _clean_line(value: str) -> str:
    value = value.replace("\u00a0", " ")
    value = value.replace("\ufffd", "")
    value = re.sub(r"[\x00-\x1f\x7f]+", " ", value)
    value = re.sub(r"[ \t]+", " ", value)
    return value.strip(" \t•·-_")


def strip_source_range_suffix(value: str) -> str:
    """Remove only trailing slide/page range metadata from a structural title."""
    title = _clean_line(value)
    for _ in range(3):
        next_title = SOURCE_RANGE_SUFFIX_RE.sub("", title).strip()
        if next_title == title:
            break
        if not next_title:
            return ""
        title = next_title
    return title


def _normalise(value: str) -> str:
    return re.sub(r"[^\w\d]+", " ", value.casefold(), flags=re.UNICODE).strip()


def _has_toc_marker(text: str) -> bool:
    markers = {_normalise(marker) for marker in TOC_MARKERS}
    return any(_normalise(_clean_line(line)) in markers for line in text.splitlines())


def _has_toc_signal(text: str) -> bool:
    """Detect explicit and layout-extracted table-of-contents pages."""
    if _has_toc_marker(text):
        return True
    lines = [_clean_line(line) for line in text.splitlines()]
    numbered_count = sum(1 for line in lines if NUMBERED_ENTRY_RE.match(line))
    slide_count = sum(1 for line in lines if SLIDE_RANGE_RE.search(line))
    dotted_count = sum(1 for line in lines if TOC_LINE_RE.match(line))
    page_only_count = sum(1 for line in lines if TOC_PAGE_ONLY_RE.fullmatch(line))
    # Require both a numbered sequence and page/slide evidence so a normal
    # numbered body list is not promoted to an authoritative outline.
    return numbered_count >= 2 and (slide_count >= 2 or dotted_count >= 2 or page_only_count >= 2)


def _looks_like_heading(line: str) -> bool:
    if not line or len(line) > 160:
        return False
    if line.endswith((".", ";", ":", ",", "?", "!")):
        return False
    words = line.split()
    if not 2 <= len(words) <= 18:
        return False
    letters = [char for char in line if char.isalpha()]
    if not letters:
        return False
    uppercase_ratio = sum(char.isupper() for char in letters) / len(letters)
    # Sentence-case lines are usually body text in extracted PDFs. Keep the
    # fallback conservative; numbered/chapter headings are parsed separately.
    return uppercase_ratio >= 0.72


def _number_depth(label: str) -> int:
    if re.fullmatch(r"[IVXLCDM]+", label, flags=re.IGNORECASE):
        return 1
    return max(1, min(6, label.count(".") + 1))


def _parse_explicit_heading(line: str, page: int | None) -> tuple[str, int, str | None, int | None] | None:
    chapter_match = CHAPTER_HEADING_RE.match(line)
    if chapter_match:
        number_label = chapter_match.group(2).strip()
        title = chapter_match.group(3).strip() or f"{chapter_match.group(1).title()} {number_label}"
        return title[:MAX_NODE_TITLE_CHARS], 1, number_label[:32], None

    arabic_match = ARABIC_HEADING_RE.match(line)
    if arabic_match:
        label = arabic_match.group(1)
        title = arabic_match.group(2).strip(" .:-")
        if len(title) >= 3:
            return title[:MAX_NODE_TITLE_CHARS], _number_depth(label), label[:32], None

    roman_match = ROMAN_HEADING_RE.match(line)
    if roman_match:
        label = roman_match.group(1).upper()
        title = roman_match.group(2).strip(" .:-")
        if len(title) >= 3:
            return title[:MAX_NODE_TITLE_CHARS], 1, label, None

    return None


def _parse_toc_line(line: str) -> tuple[str, int | None] | None:
    match = TOC_LINE_RE.match(line)
    if not match:
        return None
    title = _clean_line(match.group(1))
    if not title or len(title) < 3 or len(title) > MAX_NODE_TITLE_CHARS:
        return None
    if _normalise(title) in {_normalise(marker) for marker in TOC_MARKERS}:
        return None
    return title, int(match.group(2))


def _toc_candidate(
    title: str,
    *,
    page: int | None,
    logical_page: int | None,
    has_chapter: bool,
    number_label: str | None = None,
) -> dict[str, Any] | None:
    title = _clean_line(title)
    if not title or _normalise(title) in {_normalise(marker) for marker in TOC_MARKERS}:
        return None
    explicit = _parse_explicit_heading(title, page)
    if explicit:
        normalized_title, level, number_label, _ = explicit
        node_type = "chapter" if level == 1 else "section"
    else:
        normalized_title = title[:MAX_NODE_TITLE_CHARS]
        level = _number_depth(number_label) if number_label else (2 if has_chapter else 1)
        node_type = "chapter" if level == 1 else "section"
    range_match = SLIDE_RANGE_RE.search(normalized_title)
    if logical_page is None and range_match:
        logical_page = int(range_match.group(1))
    return {
        "title": normalized_title,
        "level": level,
        "number_label": number_label,
        "page": page,
        "logical_page": logical_page,
        "node_type": node_type,
    }


def _extract_toc_candidates(text: str, page: int | None) -> list[dict[str, Any]]:
    lines = [_clean_line(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    candidates: list[dict[str, Any]] = []
    has_chapter = False
    active_label: str | None = None
    active_lines: list[str] = []
    active_logical_page: int | None = None

    def flush_active() -> None:
        nonlocal active_label, active_lines, active_logical_page, has_chapter
        if not active_lines:
            active_label = None
            active_logical_page = None
            return
        candidate = _toc_candidate(
            " ".join(active_lines),
            page=page,
            logical_page=active_logical_page,
            has_chapter=has_chapter,
            number_label=active_label,
        )
        if candidate:
            has_chapter = has_chapter or candidate["node_type"] == "chapter"
            candidates.append(candidate)
        active_label = None
        active_lines = []
        active_logical_page = None

    index = 0
    while index < len(lines):
        line = lines[index]
        if _normalise(line) in {_normalise(marker) for marker in TOC_MARKERS}:
            # In the supplied PDF the marker is a footer after the numbered
            # entries. It also appears before entries in other documents.
            if active_lines or candidates:
                flush_active()
                break
            index += 1
            continue

        parsed = _parse_toc_line(line)
        if parsed:
            flush_active()
            title, logical_page = parsed
            candidate = _toc_candidate(
                title,
                page=page,
                logical_page=logical_page,
                has_chapter=has_chapter,
            )
            if candidate:
                has_chapter = has_chapter or candidate["node_type"] == "chapter"
                candidates.append(candidate)
            index += 1
            continue

        numbered = NUMBERED_ENTRY_RE.match(line)
        if numbered:
            flush_active()
            active_label = numbered.group(1).upper()
            remainder = _clean_line(numbered.group(2) or "")
            if remainder:
                active_lines.append(remainder)
            index += 1
            continue

        if active_label is not None:
            page_match = TOC_PAGE_ONLY_RE.fullmatch(line)
            if page_match and active_lines:
                active_logical_page = int(page_match.group(1))
            elif line:
                active_lines.append(line)
            index += 1
            continue

        next_page = TOC_PAGE_ONLY_RE.fullmatch(lines[index + 1]) if index + 1 < len(lines) else None
        if next_page:
            candidate = _toc_candidate(
                line,
                page=page,
                logical_page=int(next_page.group(1)),
                has_chapter=has_chapter,
            )
            if candidate:
                has_chapter = has_chapter or candidate["node_type"] == "chapter"
                candidates.append(candidate)
            index += 2
            continue
        index += 1
    flush_active()
    return candidates


def _set_parent_refs(nodes: list[SourceStructureNode]) -> list[SourceStructureNode]:
    stack: list[SourceStructureNode] = []
    result: list[SourceStructureNode] = []
    for node in nodes:
        stack = [item for item in stack if item.level < node.level]
        parent = stack[-1].source_ref if stack else None
        updated = SourceStructureNode(
            **{**node.to_dict(), "parent_source_ref": parent},
        )
        result.append(updated)
        stack.append(updated)
    return result


def _build_nodes(candidates: Iterable[dict[str, Any]], confidence: float) -> list[SourceStructureNode]:
    nodes: list[SourceStructureNode] = []
    seen: set[tuple[str, int | None, int]] = set()
    for candidate in candidates:
        title = _clean_line(str(candidate.get("title") or ""))[:MAX_NODE_TITLE_CHARS]
        if not title:
            continue
        level = max(1, min(6, int(candidate.get("level") or 1)))
        page = candidate.get("page")
        logical_page = candidate.get("logical_page")
        dedupe_key = (_normalise(title), page, level)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        order = len(nodes)
        nodes.append(
            SourceStructureNode(
                source_ref=f"src-{order + 1:03d}",
                title=title,
                level=level,
                order=order,
                page=page if isinstance(page, int) and page > 0 else None,
                logical_page=logical_page if isinstance(logical_page, int) and logical_page > 0 else None,
                number_label=str(candidate.get("number_label") or "")[:32] or None,
                node_type=str(candidate.get("node_type") or "section"),
                confidence=confidence,
            ),
        )
        if len(nodes) >= MAX_NODES:
            break
    return _set_parent_refs(nodes)


def analyze_source_structure(sections: Iterable[Any]) -> dict[str, Any]:
    page_lines: list[tuple[str, int | None]] = []
    toc_page_indexes: set[int] = set()
    material = list(sections)
    for section_index, section in enumerate(material):
        page = getattr(section, "page", None)
        text = str(getattr(section, "text", "") or "")
        lines = [_clean_line(line) for line in text.splitlines()]
        lines = [line for line in lines if line]
        page_lines.extend((line, page) for line in lines)
        if _has_toc_signal(text):
            toc_page_indexes.add(section_index)

    toc_candidates: list[dict[str, Any]] = []
    for section_index in toc_page_indexes:
        section = material[section_index]
        page = getattr(section, "page", None)
        toc_candidates.extend(
            _extract_toc_candidates(
                str(getattr(section, "text", "") or ""),
                page,
            ),
        )

    warnings: list[str] = []
    if len(toc_candidates) >= 2:
        nodes = _build_nodes(toc_candidates, 0.95)
        structure_source = "toc"
    else:
        heading_candidates: list[dict[str, Any]] = []
        for line, page in page_lines:
            parsed = _parse_explicit_heading(line, page)
            if parsed:
                title, level, number_label, _ = parsed
                heading_candidates.append(
                    {
                        "title": title,
                        "level": level,
                        "number_label": number_label,
                        "page": page,
                        "node_type": "chapter" if level == 1 else "section",
                    },
                )
            elif _looks_like_heading(line):
                heading_candidates.append(
                    {"title": line, "level": 1, "page": page, "node_type": "section"},
                )
        nodes = _build_nodes(heading_candidates, 0.76 if heading_candidates else 0.35)
        structure_source = "heading_inferred" if nodes else "semantic_inferred"

    if not nodes:
        first_line = next((line for line, _ in page_lines if len(line) >= 3), "Tài liệu nguồn")
        nodes = _build_nodes(
            [{"title": first_line[:MAX_NODE_TITLE_CHARS], "level": 1, "node_type": "document"}],
            0.35,
        )
        warnings.append("NO_EXPLICIT_HEADINGS")
    if structure_source == "heading_inferred":
        warnings.append("TOC_NOT_FOUND_HEADING_INFERRED")
    if structure_source == "semantic_inferred":
        warnings.append("NO_EXPLICIT_HEADINGS")

    return {
        "parser_version": PARSER_VERSION,
        "structure_source": structure_source,
        "confidence": round(min(1.0, max(0.0, max((node.confidence for node in nodes), default=0.0))), 4),
        "warnings": list(dict.fromkeys(warnings)),
        "nodes": [node.to_dict() for node in nodes],
    }


def structure_outline(structure: dict[str, Any], *, max_chars: int = 12000, locale: str = "vi") -> str:
    nodes = structure.get("nodes") if isinstance(structure, dict) else []
    if not isinstance(nodes, list) or not nodes:
        return ""
    source = str(structure.get("structure_source") or "semantic_inferred")
    confidence = float(structure.get("confidence") or 0)
    heading = (
        f"Cấu trúc nguồn ({source}, độ tin cậy {round(confidence * 100)}%):"
        if locale != "en"
        else f"Source structure ({source}, confidence {round(confidence * 100)}%):"
    )
    lines = [heading]
    for node in nodes:
        if not isinstance(node, dict):
            continue
        # Keep raw node titles in storage for provenance, but expose semantic
        # titles to the authoring model so source ranges are not copied into
        # Chapter/Lesson/Unit names.
        title = strip_source_range_suffix(str(node.get("title") or ""))
        if not title:
            continue
        prefix = str(node.get("number_label") or "").strip()
        if prefix:
            prefix += " "
        indent = "  " * max(0, min(5, int(node.get("level") or 1) - 1))
        ref = str(node.get("source_ref") or "")
        page = node.get("logical_page") or node.get("page")
        suffix = f" (trang {page})" if page else ""
        line = f"{indent}[{ref}] {prefix}{title}{suffix}"
        if sum(len(item) + 1 for item in lines) + len(line) > max_chars:
            lines.append("... cấu trúc nguồn đã được rút gọn ..." if locale != "en" else "... source structure truncated ...")
            break
        lines.append(line)
    return "\n".join(lines)


def compact_structure(structure: dict[str, Any]) -> dict[str, Any]:
    return {
        "parser_version": str(structure.get("parser_version") or PARSER_VERSION),
        "structure_source": str(structure.get("structure_source") or "semantic_inferred"),
        "confidence": float(structure.get("confidence") or 0),
        "warnings": [str(item)[:120] for item in structure.get("warnings", []) if str(item).strip()][:8],
        "nodes": [
            {
                key: node.get(key)
                for key in (
                    "source_ref",
                    "title",
                    "level",
                    "order",
                    "page",
                    "logical_page",
                    "number_label",
                    "parent_source_ref",
                    "node_type",
                    "confidence",
                )
                if node.get(key) is not None
            }
            for node in structure.get("nodes", [])[:MAX_NODES]
            if isinstance(node, dict) and str(node.get("title") or "").strip()
        ],
    }


def chunk_structure_metadata(
    structure: dict[str, Any],
    *,
    page: int | None,
    text: str,
    include_outline: bool,
) -> dict[str, Any]:
    nodes = [node for node in structure.get("nodes", []) if isinstance(node, dict)]
    selected: dict[str, Any] | None = None
    normalized_text = _normalise(text)

    # A TOC node describes a range, so body chunks usually do not contain the
    # chapter title verbatim. Resolve the owning top-level node by page first;
    # otherwise retrieval cannot enforce a selected chapter scope after a
    # durable re-index.
    if page is not None and structure.get("structure_source") == "toc":
        top_level_nodes = [
            node
            for node in nodes
            if int(node.get("level") or 1) == 1
            and isinstance(node.get("logical_page") or node.get("page"), int)
        ]
        top_level_nodes.sort(
            key=lambda node: int(node.get("logical_page") or node.get("page") or 0),
        )
        for node in top_level_nodes:
            start_page = int(node.get("logical_page") or node.get("page") or 0)
            if start_page <= page:
                selected = node
            else:
                break

    for node in nodes:
        if selected is not None:
            break
        title = _normalise(str(node.get("title") or ""))
        if title and title in normalized_text:
            selected = node
            break
    if selected is None and page is not None and structure.get("structure_source") == "heading_inferred":
        for node in nodes:
            node_page = node.get("page")
            if isinstance(node_page, int) and node_page <= page:
                selected = node
            elif isinstance(node_page, int) and node_page > page:
                break

    metadata: dict[str, Any] = {
        "structure_source": structure.get("structure_source"),
        "structure_confidence": structure.get("confidence"),
        "parser_version": structure.get("parser_version") or PARSER_VERSION,
    }
    if selected:
        metadata["source_ref"] = selected.get("source_ref")
        metadata["heading_path"] = selected.get("title")
    if include_outline:
        metadata["source_structure"] = compact_structure(structure)
    return metadata
