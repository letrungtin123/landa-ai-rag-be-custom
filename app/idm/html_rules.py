"""HTML slot rules of the W5 writer: safe diagnostics and a deterministic normaliser.

An ``html`` slot is written as ``semantic_content`` (ordered sections of typed blocks) and
rendered by Node. The shared staged validator stays authoritative
(``app.services.lesson_author.proposal_validation.semantic_learning_visible_text`` on top of
``app.ordered_learning_content.flatten_ordered_content``, ``staged_component_payload_code``
and ``staged_instructional_finding``). It reports only the first failure and only as
``HTML_SEMANTIC_INVALID``, while the provider wire schema drops every length and count
bound, so the model never learns which rule it broke (run e869f43a: every html repair
was rejected). This module mirrors those rules so the IDM writer can

* state them up front and name the exact failing rule, location and measured value in a
  repair prompt (``app.idm.prompts``),
* log safe rule codes and paths (never content), and
* apply deterministic, meaning-preserving fixes before validation
  (:func:`normalize_html_semantic`): it never invents, rewrites or drops text, it only
  moves text into the block shape the renderer accepts and drops empty or duplicated values.

``tests/test_idm_html_rules.py`` keeps the mirror in parity with the shared validator.
"""

from __future__ import annotations

import copy
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Final

# --- limits (mirror proposal_validation.SEMANTIC_LEARNING_HTML_LIMITS and flatten_ordered_content) ---
ORDERED_CONTENT_VERSION: Final = 2
MAX_SECTIONS: Final = 12
MAX_BLOCKS_PER_SECTION: Final = 12
MAX_HEADING_CHARS: Final = 240
MAX_SECTION_REFS: Final = 24
MAX_SECTION_REF_CHARS: Final = 160
# Fallback minimum when a contract carries no source length (MIN_STAGED_LESSON_AUTHOR_HTML_TEXT_CHARS).
DEFAULT_MIN_VISIBLE_CHARS: Final = 320
MIN_VISIBLE_CHARS_FLOOR: Final = 20
# Whole-slot groups after flattening: group -> (max entries, max characters per entry).
GROUP_LIMITS: Final[Mapping[str, tuple[int, int]]] = {
    "paragraphs": (12, 2_000),
    "bullet_points": (20, 800),
    "ordered_steps": (20, 1_000),
    "warnings": (8, 1_000),
}
MAX_ROWS: Final = 30
MAX_ROW_LABEL_CHARS: Final = 500
MAX_ROW_VALUE_CHARS: Final = 1_000

BLOCK_FIELD: Final[Mapping[str, str]] = {
    "paragraph": "text", "task": "text", "warning": "text", "bullets": "items", "steps": "items", "table": "rows",
}
GROUP_OF_KIND: Final[Mapping[str, str]] = {
    "paragraph": "paragraphs", "task": "paragraphs", "warning": "warnings", "bullets": "bullet_points",
    "steps": "ordered_steps", "table": "comparison_rows",
}
_BLOCK_KEYS: Final = frozenset({"kind", "text", "items", "rows"})
_SECTION_KEYS: Final = frozenset({"heading", "learning_block_ids", "blocks"})
# Top-level keys flatten_ordered_content tolerates (legacy ones only while empty).
_TOP_KEYS: Final = frozenset({"version", "sections", "heading", "paragraphs", "bullet_points", "ordered_steps",
                              "warnings", "comparison_rows"})
LEGACY_FIELDS: Final = ("heading", "paragraphs", "bullet_points", "ordered_steps", "warnings", "comparison_rows",
                        "bullets", "steps", "warning", "table_rows")
_TOO_MANY_CODE: Final[Mapping[str, str]] = {
    "paragraphs": "HTML_TOO_MANY_PARAGRAPHS", "bullet_points": "HTML_TOO_MANY_BULLETS",
    "ordered_steps": "HTML_TOO_MANY_STEPS", "warnings": "HTML_TOO_MANY_WARNINGS",
}
_GROUP_UNIT: Final[Mapping[str, str]] = {
    "paragraphs": "paragraph+task blocks", "bullet_points": "bullet items", "ordered_steps": "step items",
    "warnings": "warning blocks",
}
# Same expression as staged.provider_schemas.staged_component_payload_code (applied to the JSON text).
PRESENTATION_RE: Final = re.compile(
    r"<\s*/?\s*(?:script|iframe|style|div|span)\b|\b(?:style|class|onerror|onclick)\s*=", re.IGNORECASE)
_WORD_RE: Final = re.compile(r"\w+", re.UNICODE)

DENSITY_CODE: Final = "HTML_INSTRUCTIONAL_DENSITY_EXCEEDED"
DEPTH_CODE: Final = "HTML_INSUFFICIENT_DEPTH"
PRESENTATION_CODE: Final = "HTML_PRESENTATION_MARKUP"
_MAX_VIOLATIONS: Final = 16


@dataclass(frozen=True)
class HtmlRuleViolation:
    """One broken rule: a stable code, a JSON location inside the slot payload, server numbers."""

    code: str
    location: str
    # (unit, measured, limit) triples; numbers only, never provider text.
    metrics: tuple[tuple[str, int, int], ...] = ()

    @property
    def structural(self) -> bool:
        """True for the rules the shared validator reports as ``HTML_SEMANTIC_INVALID``."""

        return self.code not in {DENSITY_CODE, DEPTH_CODE, PRESENTATION_CODE}

    def as_log(self, slot: int) -> str:
        suffix = "".join(f";{unit.replace(' ', '_')}={measured}/{limit}" for unit, measured, limit in self.metrics)
        return f"{self.code}@components[{slot}].{self.location}{suffix}"


def minimum_visible_chars(source_content_chars: Any) -> int:
    """The shared validator's minimum visible text of an authored html slot."""

    if type(source_content_chars) is int and source_content_chars > 0:
        return max(MIN_VISIBLE_CHARS_FLOOR, min(DEFAULT_MIN_VISIBLE_CHARS, source_content_chars // 2))
    return DEFAULT_MIN_VISIBLE_CHARS


# --- diagnostics -------------------------------------------------------------------------
def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _section_violations(section: Any, path: str, groups: dict[str, list[Any]]) -> list[HtmlRuleViolation]:
    found: list[HtmlRuleViolation] = []
    if not isinstance(section, dict) or set(section) - _SECTION_KEYS:
        return [HtmlRuleViolation("HTML_SECTION_INVALID", path)]
    heading = section.get("heading")
    if not isinstance(heading, str) or not heading.strip():
        found.append(HtmlRuleViolation("HTML_HEADING_INVALID", f"{path}.heading"))
    elif len(heading.strip()) > MAX_HEADING_CHARS:
        found.append(HtmlRuleViolation("HTML_HEADING_INVALID", f"{path}.heading",
                                       (("characters", len(heading.strip()), MAX_HEADING_CHARS),)))
    refs = section.get("learning_block_ids", [])
    if (not isinstance(refs, list) or len(refs) > MAX_SECTION_REFS or len(set(map(str, refs))) != len(refs)
            or any(not _nonempty(ref) or len(ref) > MAX_SECTION_REF_CHARS for ref in refs)):
        found.append(HtmlRuleViolation("HTML_SECTION_INVALID", f"{path}.learning_block_ids"))
    blocks = section.get("blocks")
    if not isinstance(blocks, list) or not 1 <= len(blocks) <= MAX_BLOCKS_PER_SECTION:
        count = len(blocks) if isinstance(blocks, list) else 0
        found.append(HtmlRuleViolation("HTML_BLOCK_COUNT", f"{path}.blocks",
                                       (("blocks", count, MAX_BLOCKS_PER_SECTION),)))
        if not isinstance(blocks, list):
            return found
    for index, block in enumerate(blocks):
        found.extend(_block_violations(block, f"{path}.blocks[{index}]", groups))
    return found


def _block_violations(block: Any, path: str, groups: dict[str, list[Any]]) -> list[HtmlRuleViolation]:
    if not isinstance(block, dict) or block.get("kind") not in BLOCK_FIELD:
        return [HtmlRuleViolation("HTML_BLOCK_KIND_INVALID", f"{path}.kind")]
    if set(block) - _BLOCK_KEYS:
        return [HtmlRuleViolation("HTML_BLOCK_FIELDS_MIXED", path)]
    kind = block["kind"]
    field = BLOCK_FIELD[kind]
    if any(block.get(key) for key in {"text", "items", "rows"} - {field}):
        return [HtmlRuleViolation("HTML_BLOCK_FIELDS_MIXED", path)]
    data = block.get(field)
    if field == "text":
        if not _nonempty(data):
            return [HtmlRuleViolation("HTML_BLOCK_CONTENT_REQUIRED", f"{path}.text")]
        groups[GROUP_OF_KIND[kind]].append((f"{path}.text", data))
        return []
    if not isinstance(data, list) or not data:
        return [HtmlRuleViolation("HTML_BLOCK_CONTENT_REQUIRED", f"{path}.{field}")]
    groups[GROUP_OF_KIND[kind]].extend((f"{path}.{field}[{index}]", item) for index, item in enumerate(data))
    return []


def _group_violations(groups: dict[str, list[Any]]) -> list[HtmlRuleViolation]:
    found: list[HtmlRuleViolation] = []
    for group, (max_items, max_chars) in GROUP_LIMITS.items():
        entries = groups[group]
        if len(entries) > max_items:
            found.append(HtmlRuleViolation(_TOO_MANY_CODE[group], "semantic_content.sections",
                                           ((_GROUP_UNIT[group], len(entries), max_items),)))
        for path, item in entries:
            if not _nonempty(item):
                found.append(HtmlRuleViolation("HTML_ITEM_INVALID", path))
            elif len(item.strip()) > max_chars:
                found.append(HtmlRuleViolation("HTML_TEXT_TOO_LONG", path,
                                               (("characters", len(item.strip()), max_chars),)))
    rows = groups["comparison_rows"]
    if len(rows) > MAX_ROWS:
        found.append(HtmlRuleViolation("HTML_TOO_MANY_ROWS", "semantic_content.sections",
                                       (("table rows", len(rows), MAX_ROWS),)))
    for path, row in rows:
        label = row.get("label") if isinstance(row, dict) else None
        value = row.get("value") if isinstance(row, dict) else None
        if not isinstance(label, str) or not isinstance(value, str) or not label.strip() or not value.strip():
            found.append(HtmlRuleViolation("HTML_ROW_INVALID", path))
        elif len(label.strip()) > MAX_ROW_LABEL_CHARS or len(value.strip()) > MAX_ROW_VALUE_CHARS:
            found.append(HtmlRuleViolation("HTML_ROW_TOO_LONG", path, (
                ("label characters", len(label.strip()), MAX_ROW_LABEL_CHARS),
                ("value characters", len(value.strip()), MAX_ROW_VALUE_CHARS))))
    return found


def structural_violations(semantic: Any) -> list[HtmlRuleViolation]:
    """Every ordered-content rule ``semantic`` breaks (the shared validator stops at the first)."""

    root = "semantic_content"
    if not isinstance(semantic, dict) or not semantic:
        return [HtmlRuleViolation("HTML_SEMANTIC_REQUIRED", root)]
    found: list[HtmlRuleViolation] = []
    version = semantic.get("version", ORDERED_CONTENT_VERSION)
    if type(version) is not int or version != ORDERED_CONTENT_VERSION:  # IDM stamps it on every answer
        found.append(HtmlRuleViolation("HTML_VERSION_INVALID", f"{root}.version"))
    found.extend(HtmlRuleViolation("HTML_LEGACY_FIELDS", f"{root}.{key}") for key in LEGACY_FIELDS
                 if semantic.get(key))
    if set(semantic) - _TOP_KEYS - set(LEGACY_FIELDS):
        found.append(HtmlRuleViolation("HTML_UNKNOWN_FIELD", root))
    elif set(semantic) - _TOP_KEYS:
        # Empty "bullets"/"steps"/"warning"/"table_rows" keys are still unknown to the v2 reader.
        found.extend(HtmlRuleViolation("HTML_LEGACY_FIELDS", f"{root}.{key}")
                     for key in sorted(set(semantic) - _TOP_KEYS) if not semantic.get(key))
    sections = semantic.get("sections")
    if not isinstance(sections, list) or not 1 <= len(sections) <= MAX_SECTIONS:
        count = len(sections) if isinstance(sections, list) else 0
        found.append(HtmlRuleViolation("HTML_SECTION_COUNT", f"{root}.sections", (("sections", count, MAX_SECTIONS),)))
        if not isinstance(sections, list):
            return found[:_MAX_VIOLATIONS]
    groups: dict[str, list[Any]] = {group: [] for group in (*GROUP_LIMITS, "comparison_rows")}
    for index, section in enumerate(sections):
        found.extend(_section_violations(section, f"{root}.sections[{index}]", groups))
    found.extend(_group_violations(groups))
    return found[:_MAX_VIOLATIONS]


def visible_text(semantic: Any) -> str:
    """Renderer-visible text exactly as the shared density check measures it (headings excluded)."""

    if not isinstance(semantic, dict) or not isinstance(semantic.get("sections"), list):
        return ""
    flat: dict[str, list[str]] = {group: [] for group in (*GROUP_LIMITS, "comparison_rows")}
    for section in semantic["sections"]:
        for block in section.get("blocks", []) if isinstance(section, dict) else []:
            if not isinstance(block, dict) or block.get("kind") not in BLOCK_FIELD:
                continue
            kind = block["kind"]
            field = BLOCK_FIELD[kind]
            if field == "text":
                flat[GROUP_OF_KIND[kind]].append(str(block.get("text") or "").strip())
            elif field == "items":
                flat[GROUP_OF_KIND[kind]].extend(str(item).strip() for item in block.get("items") or [])
            else:
                for row in block.get("rows") or []:
                    if isinstance(row, dict):
                        flat["comparison_rows"].extend([str(row.get("label") or "").strip(),
                                                        str(row.get("value") or "").strip()])
    return " ".join(item for group in (*GROUP_LIMITS, "comparison_rows") for item in flat[group])


def html_rule_violations(
    semantic: Any, *, budget: Mapping[str, Any] | None = None, min_chars: int | None = None,
) -> list[HtmlRuleViolation]:
    """Structure first (as the validator); length and markup rules once the structure is valid."""

    found = structural_violations(semantic)
    if not isinstance(semantic, dict):
        return found
    if PRESENTATION_RE.search(json.dumps(semantic)):
        found.insert(0, HtmlRuleViolation(PRESENTATION_CODE, "semantic_content"))
    if any(item.structural for item in found):
        return found[:_MAX_VIOLATIONS]
    text = visible_text(semantic)
    words = len(_WORD_RE.findall(text))
    if min_chars is not None and len(text) < min_chars:
        found.append(HtmlRuleViolation(DEPTH_CODE, "semantic_content", (("characters", len(text), min_chars),)))
    max_chars = budget.get("max_visible_chars") if budget else None
    max_words = budget.get("max_words") if budget else None
    if type(max_chars) is int and type(max_words) is int and (len(text) > max_chars or words > max_words):
        found.append(HtmlRuleViolation(DENSITY_CODE, "semantic_content",
                                       (("words", words, max_words), ("characters", len(text), max_chars))))
    return found[:_MAX_VIOLATIONS]



# --- normaliser ----------------------------------------------------------------------------
_KIND_ALIASES: Final[Mapping[str, str]] = {
    "text": "paragraph", "para": "paragraph", "p": "paragraph",
    "list": "bullets", "bullet": "bullets", "bullet_list": "bullets", "bullet_points": "bullets",
    "unordered_list": "bullets", "ul": "bullets", "checklist": "bullets",
    "step": "steps", "ordered_list": "steps", "numbered_list": "steps", "numbered": "steps", "ol": "steps",
    "note": "warning", "callout": "warning", "caution": "warning", "alert": "warning",
    "comparison": "table", "comparison_table": "table",
}
_HEADING_KINDS: Final = frozenset({"heading", "subheading", "sub_heading", "title", "h2", "h3", "h4"})
_TEXT_KINDS: Final = frozenset({"paragraph", "task", "warning"})
_LIST_KINDS: Final = frozenset({"bullets", "steps"})
# Inline or wrapper markup around plain text (never a document element such as script or style).
_WRAPPER_TAG_RE: Final = re.compile(
    r"<\s*/?\s*(?:div|span|p|font|b|strong|em|i|u|small|mark|section|article)\b[^<>]*>", re.IGNORECASE)
_BREAK_TAG_RE: Final = re.compile(r"<\s*br\s*/?\s*>", re.IGNORECASE)
_HEADING_TAG_RE: Final = re.compile(r"^\s*<\s*h[1-6]\b[^<>]*>(.*?)<\s*/\s*h[1-6]\s*>\s*$", re.IGNORECASE | re.DOTALL)
_MARKDOWN_HEADING_RE: Final = re.compile(r"^\s*#{1,6}\s+(\S.*?)\s*$", re.DOTALL)
_BOLD_HEADING_RE: Final = re.compile(r"^\s*\*\*(\S[^*\n]*?)\*\*\s*$")
_LIST_TAG_RE: Final = re.compile(r"^\s*<\s*(ul|ol)\b[^<>]*>(.*)<\s*/\s*(?:ul|ol)\s*>\s*$", re.IGNORECASE | re.DOTALL)
_LIST_ITEM_TAG_RE: Final = re.compile(r"<\s*li\b[^<>]*>(.*?)(?=<\s*/?\s*li\b|$)", re.IGNORECASE | re.DOTALL)
_CLOSE_ITEM_RE: Final = re.compile(r"<\s*/\s*li\s*>", re.IGNORECASE)
_ANY_TAG_RE: Final = re.compile(r"<\s*/?\s*[a-z][a-z0-9]*\b[^<>]*>", re.IGNORECASE)
_BULLET_LINE_RE: Final = re.compile(r"^\s*[-•*●▪◦+]\s+(\S.*?)\s*$")
_NUMBER_LINE_RE: Final = re.compile(r"^\s*\d{1,2}\s*[.)]\s+(\S.*?)\s*$")
_MIN_TEXT_LIST_LINES: Final = 2
_PAIR: Final = 2

# A normalised block is ("block", value) or ("heading", text): a sub-heading that opens a section.
_Piece = tuple[str, Any]


def _empty(value: Any) -> bool:
    if isinstance(value, str):
        return not value.strip()
    return value is None or (isinstance(value, (list, dict)) and not value)


def _fold(value: str) -> str:
    text = unicodedata.normalize("NFD", value.casefold())
    return " ".join("".join(char for char in text if unicodedata.category(char) != "Mn").split())


def _strings(node: Any) -> Iterable[str]:
    if isinstance(node, str):
        yield node
    elif isinstance(node, list):
        for item in node:
            yield from _strings(item)
    elif isinstance(node, dict):
        for item in node.values():
            yield from _strings(item)


def _unwrap(value: str) -> str:
    """Drop inline/wrapper tags (and their attributes) around plain text; keep every word."""

    text = _WRAPPER_TAG_RE.sub("", _BREAK_TAG_RE.sub(" ", value))
    return " ".join(text.split()) if text != value else value


def _heading_text(value: str, *, bold: bool = False) -> str | None:
    """The text of a lone heading (``<h3>X</h3>``, ``### X``; ``**X**`` when ``bold``), else ``None``."""

    patterns = (_HEADING_TAG_RE, _MARKDOWN_HEADING_RE, *((_BOLD_HEADING_RE,) if bold else ()))
    for pattern in patterns:
        match = pattern.match(value)
        if match:
            inner = _unwrap(match.group(1)).strip()
            if inner and "\n" not in inner and not _ANY_TAG_RE.search(inner):
                return inner
    return None


def _text_list(value: str) -> tuple[str, list[str]] | None:
    """A text that is only a list (``<ul><li>``, ``- a`` lines, ``1. a`` lines) -> (kind, items)."""

    match = _LIST_TAG_RE.match(value)
    if match:
        items = [_unwrap(_CLOSE_ITEM_RE.sub("", item)).strip() for item in _LIST_ITEM_TAG_RE.findall(match.group(2))]
        items = [item for item in items if item]
        if items and not any(_ANY_TAG_RE.search(item) for item in items):
            return ("steps" if match.group(1).casefold() == "ol" else "bullets"), items
        return None
    lines = [line for line in value.splitlines() if line.strip()]
    if len(lines) < _MIN_TEXT_LIST_LINES:
        return None
    for pattern, kind in ((_BULLET_LINE_RE, "bullets"), (_NUMBER_LINE_RE, "steps")):
        matches = [pattern.match(line) for line in lines]
        if all(matches):
            return kind, [found.group(1) for found in matches if found]
    return None


def _block(kind: str, *, text: str | None = None, items: list[Any] | None = None,
           rows: list[Any] | None = None) -> dict[str, Any]:
    return {"kind": kind, "text": text, "items": items or [], "rows": rows or []}


def _clean_items(items: list[Any], fixes: Counter[str]) -> list[Any]:
    cleaned: list[Any] = []
    for item in items:
        if not isinstance(item, str):
            cleaned.append(item)
            continue
        unwrapped = _unwrap(item)
        if unwrapped != item:
            fixes["markup_unwrapped"] += 1
        if not unwrapped.strip():
            fixes["empty_item_dropped"] += 1
            continue
        cleaned.append(unwrapped)
    return cleaned


def _clean_rows(rows: list[Any], fixes: Counter[str]) -> list[Any]:
    cleaned: list[Any] = []
    for raw in rows:
        row = raw
        if isinstance(row, list) and len(row) == _PAIR and all(isinstance(cell, str) for cell in row):
            row = {"label": row[0], "value": row[1]}
            fixes["row_pair_converted"] += 1
        if isinstance(row, dict):
            row = dict(row)
            for key in ("label", "value"):
                if isinstance(row.get(key), str) and (unwrapped := _unwrap(row[key])) != row[key]:
                    row[key] = unwrapped
                    fixes["markup_unwrapped"] += 1
            for key in [key for key in row if key not in {"label", "value"} and _empty(row[key])]:
                del row[key]
                fixes["empty_field_dropped"] += 1
            if set(row) <= {"label", "value"} and _empty(row.get("label")) and _empty(row.get("value")):
                fixes["empty_row_dropped"] += 1
                continue
        cleaned.append(row)
    return cleaned


def _same_block(candidate: dict[str, Any], original: dict[str, Any]) -> bool:
    return set(original) <= _BLOCK_KEYS and all(
        (candidate.get(key) or None) == (original.get(key) or None) for key in _BLOCK_KEYS)


def _split_block(kind: str, text: str | None, items: list[Any], rows: list[Any]) -> list[_Piece]:
    """One block per filled field, the introduction sentence first (no text changes)."""

    has_text = _nonempty(text)
    pieces: list[dict[str, Any]] = []
    if kind in _TEXT_KINDS:
        if has_text:
            pieces.append(_block(kind, text=text))
        if items and kind == "warning" and not has_text:
            pieces.extend(_block("warning", text=item) for item in items)
        elif items:
            pieces.append(_block("bullets", items=items))
    elif kind in _LIST_KINDS:
        if has_text and items:
            pieces.append(_block("paragraph", text=text))
        elif has_text and text is not None:
            listed = _text_list(text)
            pieces.append(_block(kind, items=listed[1]) if listed else _block("paragraph", text=text))
        if items:
            pieces.append(_block(kind, items=items))
    else:  # table
        if has_text:
            pieces.append(_block("paragraph", text=text))
        if items:
            pieces.append(_block("bullets", items=items))
    if rows:
        pieces.append(_block("table", rows=rows))
    return [("block", piece) for piece in pieces]


def _normalize_block(block: Any, fixes: Counter[str]) -> list[_Piece]:
    if not isinstance(block, dict):
        return [("block", block)]
    original = block
    counted = sum(fixes.values())
    block = dict(block)
    kind = block.get("kind")
    if isinstance(kind, str) and kind not in BLOCK_FIELD:
        folded = kind.strip().casefold()
        if (folded in _HEADING_KINDS and _nonempty(block.get("text")) and not block.get("items")
                and not block.get("rows")):
            fixes["subheading_block"] += 1
            return [("heading", _unwrap(block["text"]).strip())]
        if folded in _KIND_ALIASES:
            kind = block["kind"] = _KIND_ALIASES[folded]
            fixes["block_kind_alias"] += 1
    for key in [key for key in block if key not in _BLOCK_KEYS and _empty(block[key])]:
        del block[key]
        fixes["empty_field_dropped"] += 1
    text, items, rows = block.get("text"), block.get("items") or [], block.get("rows") or []
    if (kind not in BLOCK_FIELD or set(block) - _BLOCK_KEYS or not isinstance(items, list)
            or not isinstance(rows, list) or (text is not None and not isinstance(text, str))):
        return [("block", block)]  # unknown kinds, fields and value types stay for the validator to name
    if text is not None and (unwrapped := _unwrap(text)) != text:
        text = unwrapped
        fixes["markup_unwrapped"] += 1
    items, rows = _clean_items(items, fixes), _clean_rows(rows, fixes)
    if kind in _TEXT_KINDS and text is not None and _nonempty(text) and not items and not rows:
        heading = _heading_text(text)
        if heading is not None:
            fixes["subheading_block"] += 1
            return [("heading", heading)]
        listed = _text_list(text) if kind == "paragraph" else None
        if listed is not None:
            fixes["list_items_from_text"] += 1
            return [("block", _block(listed[0], items=listed[1]))]
    pieces = _split_block(kind, text, items, rows)
    if not pieces:
        fixes["empty_block_dropped"] += 1
        return []
    if len(pieces) == 1 and _same_block(pieces[0][1], original):
        return [("block", original)]
    if len(pieces) > 1:
        fixes["block_fields_split"] += 1
    elif pieces[0][1]["kind"] != kind:
        fixes["block_kind_fixed"] += 1
    elif sum(fixes.values()) == counted:
        fixes["block_fields_fixed"] += 1  # e.g. a list written into "text" moved into "items"
    return pieces


def _normalize_section(section: Any, fixes: Counter[str], *, split: bool) -> list[Any]:
    if not isinstance(section, dict):
        return [section]
    section = dict(section)
    if not _nonempty(section.get("heading")) and _nonempty(section.get("title")):
        section["heading"] = section.pop("title")
        fixes["section_heading_alias"] += 1
    for key in [key for key in section if key not in _SECTION_KEYS and _empty(section[key])]:
        del section[key]
        fixes["empty_field_dropped"] += 1
    heading = section.get("heading")
    if isinstance(heading, str):
        cleaned = _heading_text(heading, bold=True) or _unwrap(heading)
        if cleaned != heading:
            section["heading"] = cleaned
            fixes["heading_markup_stripped"] += 1
    blocks = section.get("blocks")
    if not isinstance(blocks, list):
        return [section]
    with_refs = "learning_block_ids" in section
    current: list[Any] = []
    section["blocks"] = current
    result = [section]
    for block in blocks:
        for piece_kind, value in _normalize_block(block, fixes):
            if piece_kind == "block":
                current.append(value)
            elif not _nonempty(result[-1].get("heading")):
                result[-1]["heading"] = value
            elif not split or not current:
                current.append(_block("paragraph", text=value))  # kept as text: no new section here
            else:
                current = []
                result.append({"heading": value, **({"learning_block_ids": []} if with_refs else {}),
                               "blocks": current})
    if len(result) > 1 and not result[-1]["blocks"]:
        # A trailing sub-heading introduces nothing: keep its text as the last paragraph.
        tail = result.pop()
        result[-1]["blocks"].append(_block("paragraph", text=tail["heading"]))
    fixes["subheading_to_section"] += len(result) - 1
    return result


def _merge_adjacent(sections: list[Any], kind: str, max_chars: int, *, only: int | None = None) -> bool:
    """Join the shortest adjacent pair of ``kind`` text blocks of one section (text unchanged)."""

    best: tuple[int, int, int] | None = None
    for s_index, section in enumerate(sections):
        blocks = section.get("blocks") if isinstance(section, dict) else None
        if (only is not None and s_index != only) or not isinstance(blocks, list):
            continue
        for index in range(len(blocks) - 1):
            left, right = blocks[index], blocks[index + 1]
            if not all(isinstance(item, dict) and item.get("kind") == kind and _nonempty(item.get("text"))
                       and not item.get("items") and not item.get("rows") for item in (left, right)):
                continue
            size = len(left["text"].strip()) + 1 + len(right["text"].strip())
            if size <= max_chars and (best is None or size < best[0]):
                best = (size, s_index, index)
    if best is None:
        return False
    _size, s_index, index = best
    blocks = sections[s_index]["blocks"]
    joined = f"{blocks[index]['text'].strip()} {blocks[index + 1]['text'].strip()}"
    blocks[index : index + 2] = [_block(kind, text=joined)]
    return True


def _merge_lists(section: dict[str, Any]) -> bool:
    blocks = section["blocks"]
    for index in range(len(blocks) - 1):
        left, right = blocks[index], blocks[index + 1]
        if (isinstance(left, dict) and isinstance(right, dict) and left.get("kind") in _LIST_KINDS
                and left.get("kind") == right.get("kind") and not left.get("text") and not right.get("text")
                and isinstance(left.get("items"), list) and isinstance(right.get("items"), list)):
            blocks[index : index + 2] = [_block(left["kind"], items=[*left["items"], *right["items"]])]
            return True
    return False


def _count(sections: list[Any], kinds: frozenset[str]) -> int:
    return sum(1 for section in sections if isinstance(section, dict) and isinstance(section.get("blocks"), list)
               for block in section["blocks"] if isinstance(block, dict) and block.get("kind") in kinds)


def _fit_whole_slot_limits(sections: list[Any], fixes: Counter[str]) -> None:
    paragraphs, paragraph_chars = GROUP_LIMITS["paragraphs"]
    while (_count(sections, frozenset({"paragraph", "task"})) > paragraphs
           and _merge_adjacent(sections, "paragraph", paragraph_chars)):
        fixes["paragraphs_merged"] += 1
    warnings, warning_chars = GROUP_LIMITS["warnings"]
    while _count(sections, frozenset({"warning"})) > warnings and _merge_adjacent(sections, "warning", warning_chars):
        fixes["warnings_merged"] += 1
    for index, section in enumerate(sections):
        while (isinstance(section, dict) and isinstance(section.get("blocks"), list)
               and len(section["blocks"]) > MAX_BLOCKS_PER_SECTION
               and (_merge_adjacent(sections, "paragraph", paragraph_chars, only=index) or _merge_lists(section))):
            fixes["blocks_merged"] += 1


def _legacy_to_section(value: dict[str, Any], fixes: Counter[str]) -> None:
    """An answer with only legacy fields and a heading becomes one ordered section (same text, same order)."""

    heading = value.get("heading")
    if not _nonempty(heading):
        return
    blocks: list[dict[str, Any]] = []
    consumed = ["heading"]
    for key, kind in (("paragraphs", "paragraph"), ("bullet_points", "bullets"), ("bullets", "bullets"),
                      ("ordered_steps", "steps"), ("steps", "steps"), ("warnings", "warning"),
                      ("warning", "warning"), ("comparison_rows", "table"), ("table_rows", "table")):
        if key not in value:
            continue
        entries = value[key]
        entries = [entries] if isinstance(entries, str) else entries
        if not isinstance(entries, list):
            return  # an unexpected shape stays for the validator to name
        consumed.append(key)
        if kind in {"paragraph", "warning"}:
            blocks.extend(_block(kind, text=item) for item in entries if isinstance(item, str) and item.strip())
        elif entries:
            blocks.append(_block(kind, rows=list(entries)) if kind == "table" else _block(kind, items=list(entries)))
    if not blocks:
        return
    for key in consumed:
        value.pop(key, None)
    value["sections"] = [{"heading": heading, "blocks": blocks}]
    fixes["legacy_converted"] += 1


def _drop_duplicated_legacy(value: dict[str, Any], fixes: Counter[str]) -> None:
    """Legacy fields next to sections are dropped only when the sections already hold every string."""

    section_text = {_fold(text) for text in _strings(value.get("sections"))}
    for key in LEGACY_FIELDS:
        if key in value and not _empty(value[key]):
            strings = [_fold(text) for text in _strings(value[key]) if text.strip()]
            if strings and all(text in section_text for text in strings):
                del value[key]
                fixes["legacy_duplicate_dropped"] += 1


def _normalize_sections(sections: list[Any], fixes: Counter[str], *, split: bool) -> list[Any]:
    normalized: list[Any] = []
    for section in sections:
        normalized.extend(_normalize_section(section, fixes, split=split))
    kept: list[Any] = []
    for section in normalized:
        blocks = section.get("blocks") if isinstance(section, dict) else None
        if isinstance(blocks, list) and not blocks and not _nonempty(section.get("heading")):
            fixes["empty_section_dropped"] += 1
            continue
        previous = kept[-1].get("blocks") if kept and isinstance(kept[-1], dict) else None
        if (isinstance(blocks, list) and blocks and not _nonempty(section.get("heading"))
                and isinstance(previous, list) and len(previous) + len(blocks) <= MAX_BLOCKS_PER_SECTION):
            previous.extend(blocks)  # an untitled continuation of the previous section
            fixes["untitled_section_merged"] += 1
            continue
        kept.append(section)
    return kept


def normalize_html_semantic(semantic: Any) -> tuple[Any, Counter[str]]:
    """Deterministic, meaning-preserving fixes of the commonest ordered-content violations.

    Returns the (possibly unchanged) value and the fixes applied. It unwraps inline/wrapper
    markup, drops empty values and empty blocks, moves an introduction sentence out of a
    list/table block into its own paragraph, turns a list written as text into a list block,
    turns a sub-heading block into a section heading, drops legacy fields that only repeat the
    sections, and joins adjacent paragraphs (text unchanged) to fit the whole-slot counts. It
    never rewrites, shortens or invents learner text.
    """

    fixes: Counter[str] = Counter()
    if not isinstance(semantic, dict):
        return semantic, fixes
    value = copy.deepcopy(semantic)
    for key in [key for key in value if key not in {"version", "sections"} and _empty(value[key])]:
        del value[key]
        fixes["empty_field_dropped"] += 1
    if not value.get("sections"):
        _legacy_to_section(value, fixes)
    elif isinstance(value["sections"], list):
        _drop_duplicated_legacy(value, fixes)
    sections = value.get("sections")
    if isinstance(sections, list):
        attempt: Counter[str] = Counter()
        kept = _normalize_sections(sections, attempt, split=True)
        if len(kept) > MAX_SECTIONS >= len(sections):
            attempt = Counter()  # sub-headings stay paragraphs rather than overflow the section limit
            kept = _normalize_sections(sections, attempt, split=False)
        _fit_whole_slot_limits(kept, attempt)
        fixes.update(attempt)
        value["sections"] = kept
    fixes = Counter({key: count for key, count in fixes.items() if count > 0})
    if not fixes:
        return semantic, fixes
    return value, fixes
