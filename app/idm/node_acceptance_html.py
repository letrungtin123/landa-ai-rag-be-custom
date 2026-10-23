"""HTML rules of the Node acceptance mirror (``app.idm.node_acceptance``): the sanitizer, the html
contract and quality codes, the semantic-content renderer, the workspace html limits and the visible text the
IDM W5 budget counts.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any, Final

from app.core.markup import replace_tags
from app.idm.node_acceptance_text import (
    _ENTITY_RE,
    _JS_WS,
    _JS_WS_RUN,
    _js_number,
    _js_truthy,
    _list,
    _nullish,
    _record,
    _Rejected,
    escape_html_text,
    instructional_fold,
    js_len,
    js_trim,
    strip_html,
    visible_html_text,
)

# --- html: sanitizer, contract, instructional quality -------------------------------------------
_SAFE_HTML_TAGS: Final = frozenset({"h2", "h3", "p", "ul", "ol", "li", "strong", "blockquote", "table", "thead",
                                    "tbody", "tr", "th", "td"})
_DROPPED_ELEMENTS: Final = tuple(
    re.compile(rf"<{tag}[^>]*>[\s\S]*?</{tag}>", re.IGNORECASE | re.ASCII)
    for tag in ("script", "style", "iframe", "object", "embed"))
_SANITIZE_TAG_RE: Final = re.compile(f"</?([a-z0-9]+)(?:[{_JS_WS}][^>]*)?>", re.IGNORECASE | re.ASCII)
_CONTRACT_TAG_RE: Final = re.compile(r"</?([a-z0-9]+)>", re.IGNORECASE | re.ASCII)
MAX_UNIT_HTML_CHARS: Final = 20_000
MIN_UNIT_HTML_TEXT_CHARS: Final = 180


def sanitize_lesson_author_html(value: Any) -> str:
    """``sanitizeLessonAuthorHtml``."""

    text = value if isinstance(value, str) else ""
    for pattern in _DROPPED_ELEMENTS:
        text = pattern.sub("", text)

    def keep(match: re.Match[str]) -> str:
        tag = match.group(1).lower()
        if tag not in _SAFE_HTML_TAGS:
            return ""
        return f"</{tag}>" if match.group(0).startswith("</") else f"<{tag}>"

    return js_trim(_SANITIZE_TAG_RE.sub(keep, text))


def _count_tags(html: str, tag: str) -> int:
    return len(re.findall(f"<{tag}(?:[{_JS_WS}][^>]*)?>", html, flags=re.IGNORECASE | re.ASCII))


def _count_list_items(html: str, tag: str) -> int:
    segments = re.findall(f"<{tag}(?:[{_JS_WS}][^>]*)?>([\\s\\S]*?)</{tag}>", html, flags=re.IGNORECASE | re.ASCII)
    return sum(_count_tags(segment, "li") for segment in segments)


def _artifact_missing(html: str, artifact: Mapping[str, Any]) -> bool:
    minimum = artifact.get("minimum_items") or 1
    kind = artifact.get("type")
    if kind == "ordered_list":
        return _count_list_items(html, "ol") < minimum
    if kind == "checklist":
        return _count_list_items(html, "ul") < minimum
    if kind in {"table", "comparison"}:
        return not (_count_tags(html, "table") > 0 and _count_tags(html, "tr") >= max(2, minimum))
    if kind in {"warning", "requirement", "exception"}:
        return _count_tags(html, "blockquote") < minimum
    return False


def html_contract_code(html: str, required_artifacts: Iterable[Mapping[str, Any]] = ()) -> str | None:
    """``lessonAuthorHtmlContractFinding`` code."""

    if not js_trim(html):
        return "HTML_EMPTY"
    stack: list[str] = []
    cursor = 0
    for match in _CONTRACT_TAG_RE.finditer(html):
        if not stack and visible_html_text(html[cursor:match.start()]):
            return "HTML_TEXT_OUTSIDE_ROOT"
        tag = match.group(1).lower()
        if tag not in _SAFE_HTML_TAGS:
            return "HTML_UNSUPPORTED_TAG"
        if match.group(0).startswith("</"):
            if not stack or stack.pop() != tag:
                return "HTML_INVALID_NESTING"
        else:
            stack.append(tag)
        cursor = match.end()
    if stack:
        return "HTML_UNCLOSED_TAG"
    if visible_html_text(html[cursor:]):
        return "HTML_TEXT_OUTSIDE_ROOT"
    if "<li>" in html and not re.search(r"<(?:ul|ol)>", html, re.IGNORECASE | re.ASCII):
        return "HTML_LIST_ITEM_OUTSIDE_LIST"
    if re.search(r"<(?:th|td)>", html, re.IGNORECASE | re.ASCII) and not re.search(r"<tr>", html, re.I | re.A):
        return "HTML_CELL_OUTSIDE_ROW"
    if re.search(r"<tr>", html, re.IGNORECASE | re.ASCII) and not re.search(r"<table>", html, re.I | re.A):
        return "HTML_ROW_OUTSIDE_TABLE"
    if any(_artifact_missing(html, artifact) for artifact in required_artifacts):
        return "HTML_REQUIRED_ARTIFACT_MISSING"
    return None


_REVIEW_PHRASES: Final = (
    "ra soat y trong tai lieu nguon", "theo dung thu tu xuat hien trong tai lieu nguon",
    "duoc giu nguyen de nguoi dung ra soat theo nguon", "review the source point", "displayed order in the source",
    "retained for source review",
)
# Run on the folded text, whose only white space is " ": ASCII `\b`, `\d` and `\s` are exact.
_ATTRIBUTION_RES: Final = tuple(re.compile(pattern, re.IGNORECASE | re.ASCII) for pattern in (
    r"\b(?:theo|dua tren|trich tu)\s+(?:tai lieu|nguon|source|document)\b",
    r"\b(?:trong|inside)\s+(?:tai lieu nguon|source document)\b",
    r"\b(?:tai lieu|document|source)\s+(?:neu|mo ta|states?|describes?)\b",
    "\\b(?:nguon|source|tai lieu nguon)\\s*[:\uff1a]",
))
_LOCATOR_RE: Final = re.compile(
    r"\b(?:trang|page|slide|chunk|doan nguon|muc nguon)\s*(?:so|number|no\.?|#)?\s*[:#-]?\s*\d{1,6}\b",
    re.IGNORECASE | re.ASCII)
_FILENAME_RE: Final = re.compile(
    f"(?:^|[{_JS_WS}])[^<>\\n]{{0,160}}\\.(?:pdf|pptx?|docx?|xlsx?|csv|txt|rtf)(?=\\Z|[{_JS_WS}]|[),.;:])",
    re.IGNORECASE)
_INTERNAL_ID_RE: Final = re.compile(
    f"(?:^|[{_JS_WS}])(?:p[0-9]+[-_]f[0-9]+|src[-_][0-9]+(?:[-_]f[0-9]+)?|"
    f"(?:component|block|fact|scope)[-_](?:[a-z0-9]+[-_]?){{1,8}})(?=\\Z|[{_JS_WS}]|[),.;:])",
    re.IGNORECASE)
_OCR_NOISE_RE: Final = re.compile(r"(.)\1{5,}", re.IGNORECASE | re.DOTALL)
_SEGMENT_RE: Final = re.compile(f"<(p|li|th|td|blockquote)>[{_JS_WS}]*([^<]+?)[{_JS_WS}]*</", re.IGNORECASE)
_URL_ONLY_RE: Final = re.compile(f"^(?:https?://|www\\.)[^{_JS_WS}]+\\Z", re.IGNORECASE)
_EMAIL_ONLY_RE: Final = re.compile(r"^[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\Z", re.ASCII)
_PHONE_ONLY_RE: Final = re.compile(f"^\\+?[0-9][0-9{_JS_WS}().-]{{7,}}[0-9]\\Z")
_THANKS_RE: Final = re.compile("^(?:thank you|thanks|c\u1ea3m \u01a1n|xin c\u1ea3m \u01a1n)\\Z", re.IGNORECASE)
MIN_DUPLICATE_SEGMENT_CHARS: Final = 12


def html_quality_code(html: str, exact_identifiers: Iterable[str] = (), *, worksheet: bool = False) -> str | None:
    """``lessonAuthorHtmlInstructionalQualityFinding`` code (worksheets may repeat table cells)."""

    visible = visible_html_text(html)
    folded = instructional_fold(visible)
    if any(phrase in folded for phrase in _REVIEW_PHRASES):
        return "HTML_SOURCE_REVIEW_COPY"
    if any(pattern.search(folded) for pattern in _ATTRIBUTION_RES):
        return "HTML_SOURCE_ATTRIBUTION"
    if _LOCATOR_RE.search(folded):
        return "HTML_SOURCE_LOCATOR"
    if _FILENAME_RE.search(visible):
        return "HTML_SOURCE_FILENAME"
    if _INTERNAL_ID_RE.search(visible):
        return "HTML_INTERNAL_IDENTIFIER"
    for identifier in exact_identifiers:
        candidate = js_trim(identifier) if isinstance(identifier, str) else ""
        if js_len(candidate) < 3:  # noqa: PLR2004 - the Node identifier floor
            continue
        if re.search(rf"(?:^|[^\w]){re.escape(candidate)}(?=\Z|[^\w])", visible, re.IGNORECASE):
            return "HTML_INTERNAL_IDENTIFIER"
    if _OCR_NOISE_RE.search(visible):
        return "HTML_OCR_NOISE"
    segments = [(tag.lower(), js_trim(_JS_WS_RUN.sub(" ", _ENTITY_RE.sub(" ", text))))
                for tag, text in _SEGMENT_RE.findall(html)]
    segments = [(tag, text) for tag, text in segments if text]
    for _tag, text in segments:
        if _URL_ONLY_RE.search(text) or _EMAIL_ONLY_RE.search(text) or _THANKS_RE.search(text):
            return "HTML_BOILERPLATE"
    meaningful = [instructional_fold(text) for tag, text in segments
                  if not (worksheet and tag in {"th", "td"})]
    meaningful = [text for text in meaningful if js_len(text) >= MIN_DUPLICATE_SEGMENT_CHARS]
    if len(set(meaningful)) != len(meaningful):
        return "HTML_DUPLICATE_BLOCK"
    return None


# --- html: semantic renderer (renderSemanticLearningHtml) ------------------------------------------
_LEGACY_KEYS: Final = ("heading", "paragraphs", "bullet_points", "ordered_steps", "warnings", "comparison_rows")
_BLOCK_GROUP: Final = {"paragraph": "paragraphs", "task": "paragraphs", "bullets": "bullet_points",
                       "steps": "ordered_steps", "warning": "warnings", "table": "comparison_rows"}
_TEXT_LIMITS: Final = {"paragraphs": (12, 2_000), "bullet_points": (20, 800), "ordered_steps": (20, 1_000),
                       "warnings": (8, 1_000)}
_MAX_SECTIONS: Final = 12
_MAX_BLOCKS: Final = 12
_MAX_HEADING: Final = 240
_MAX_REFS: Final = 24
_MAX_REF_CHARS: Final = 160
_MAX_ROWS: Final = 30
_MAX_ROW_LABEL: Final = 500
_MAX_ROW_VALUE: Final = 1_000


def _flatten_ordered(content: dict[str, Any]) -> dict[str, Any]:
    """``flattenOrderedLearningContent`` (raises ``ValueError`` like Node throws)."""

    version = content.get("version")
    if not (_js_number(version) and version == 2):  # noqa: PLR2004 - ordered content version
        sections = content.get("sections")
        if ((version is not None and not (_js_number(version) and version == 1))
                or (len(sections) > 0 if isinstance(sections, list) else sections is not None)):
            raise ValueError("version")
        return content
    if any(key not in {"version", "sections", *_LEGACY_KEYS} for key in content) or any(
            (len(content[key]) > 0 if isinstance(content[key], list) else _js_truthy(content[key]))
            for key in _LEGACY_KEYS if key in content):
        raise ValueError("mixed")
    sections = content.get("sections")
    if not isinstance(sections, list) or not 1 <= len(sections) <= _MAX_SECTIONS:
        raise ValueError("sections")
    flat: dict[str, list[Any]] = {"paragraphs": [], "bullet_points": [], "ordered_steps": [], "warnings": [],
                                  "comparison_rows": []}
    for raw in sections:
        section = _record(raw)
        heading = section.get("heading")
        if (any(key not in {"heading", "learning_block_ids", "blocks"} for key in section)
                or not isinstance(heading, str) or not js_trim(heading) or js_len(js_trim(heading)) > _MAX_HEADING):
            raise ValueError("heading")
        refs = _nullish(section.get("learning_block_ids"), [])
        if (not isinstance(refs, list) or len(refs) > _MAX_REFS
                or any(not isinstance(ref, str) or not js_trim(ref) or js_len(ref) > _MAX_REF_CHARS for ref in refs)
                or len({json.dumps(ref) for ref in refs}) != len(refs)):
            raise ValueError("refs")
        blocks = section.get("blocks")
        if not isinstance(blocks, list) or not 1 <= len(blocks) <= _MAX_BLOCKS:
            raise ValueError("blocks")
        for raw_block in blocks:
            block = _record(raw_block)
            kind = block.get("kind")
            if (not isinstance(kind, str) or kind not in _BLOCK_GROUP
                    or any(key not in {"kind", "text", "items", "rows"} for key in block)):
                raise ValueError("kind")
            data_field = "rows" if kind == "table" else "items" if kind in {"bullets", "steps"} else "text"
            if any((len(block[key]) > 0 if isinstance(block[key], list) else _js_truthy(block[key]))
                   for key in ("text", "items", "rows") if key != data_field and key in block):
                raise ValueError("mixed block")
            data = block.get(data_field)
            if data_field == "text" and not (isinstance(data, str) and js_trim(data)):
                raise ValueError("text")
            if data_field != "text" and not (isinstance(data, list) and data):
                raise ValueError("items")
            flat[_BLOCK_GROUP[kind]].extend([data] if data_field == "text" else _list(data))
    return flat


def _semantic_payload_invalid(value: dict[str, Any]) -> bool:
    """``validateSemanticLearningHtmlPayload`` (true when Node reports a failure)."""

    try:
        content = _flatten_ordered(value)
    except ValueError:
        return True
    if not content:
        return True
    renderable = False
    if "heading" in content:
        heading = content["heading"]
        if not isinstance(heading, str) or not js_trim(heading) or js_len(js_trim(heading)) > _MAX_HEADING:
            return True
        renderable = True
    for label, primary, alias in (("paragraphs", "paragraphs", None), ("bullet_points", "bullet_points", "bullets"),
                                  ("ordered_steps", "ordered_steps", "steps"), ("warnings", "warnings", "warning")):
        items = _nullish(content.get(primary), content.get(alias) if alias else None)
        if items is None:
            continue
        max_items, max_chars = _TEXT_LIMITS[label]
        if not isinstance(items, list) or len(items) > max_items or any(
                not isinstance(item, str) or not js_trim(item) or js_len(js_trim(item)) > max_chars for item in items):
            return True
        renderable = renderable or bool(items)
    rows = _nullish(content.get("comparison_rows"), content.get("table_rows"))
    if rows is not None:
        if not isinstance(rows, list) or len(rows) > _MAX_ROWS:
            return True
        for raw_row in rows:
            row = _record(raw_row)
            label = js_trim(row["label"]) if isinstance(row.get("label"), str) else ""
            value_text = js_trim(row["value"]) if isinstance(row.get("value"), str) else ""
            if not label or not value_text or js_len(label) > _MAX_ROW_LABEL or js_len(value_text) > _MAX_ROW_VALUE:
                return True
        renderable = renderable or bool(rows)
    return not renderable


def _semantic_texts(value: Any) -> list[str]:
    return [js_trim(item) for item in _list(value) if isinstance(item, str) and js_trim(item)]


def render_semantic_html(value: Any) -> str | None:
    """``renderSemanticLearningHtml``; raises ``_Rejected`` where Node throws."""

    content = _record(value)
    if not content:
        return None
    if _semantic_payload_invalid(content):
        raise _Rejected("HTML_SEMANTIC_INVALID")
    fragments: list[str] = []
    if _js_number(content.get("version")) and content["version"] == 2:  # noqa: PLR2004 - ordered content version
        for raw in content["sections"]:
            section = _record(raw)
            fragments.append(f"<h2>{escape_html_text(section.get('heading'))}</h2>")
            for raw_block in section["blocks"]:
                block = _record(raw_block)
                if block["kind"] == "table":
                    cells = "".join(f"<tr><th>{escape_html_text(_record(row).get('label'))}</th>"
                                    f"<td>{escape_html_text(_record(row).get('value'))}</td></tr>"
                                    for row in block["rows"])
                    fragments.append(f"<table><tbody>{cells}</tbody></table>")
                elif block["kind"] in {"bullets", "steps"}:
                    tag = "ol" if block["kind"] == "steps" else "ul"
                    items = "".join(f"<li>{escape_html_text(item)}</li>" for item in block["items"])
                    fragments.append(f"<{tag}>{items}</{tag}>")
                else:
                    tag = "blockquote" if block["kind"] == "warning" else "p"
                    fragments.append(f"<{tag}>{escape_html_text(block.get('text'))}</{tag}>")
        rendered = sanitize_lesson_author_html("".join(fragments))
        if html_contract_code(rendered):
            raise _Rejected("HTML_SEMANTIC_RENDER_INVALID")
        return rendered
    heading = js_trim(content["heading"]) if isinstance(content.get("heading"), str) else ""
    rows_value = _nullish(content.get("comparison_rows"), content.get("table_rows"))
    rows = [f"<tr><th>{escape_html_text(js_trim(_record(row)['label']))}</th>"
            f"<td>{escape_html_text(js_trim(_record(row)['value']))}</td></tr>"
            for row in (rows_value if isinstance(rows_value, list) else [])
            if isinstance(_record(row).get("label"), str) and isinstance(_record(row).get("value"), str)
            and js_trim(_record(row)["label"]) and js_trim(_record(row)["value"])]
    bullets = _semantic_texts(_nullish(content.get("bullet_points"), content.get("bullets")))
    steps = _semantic_texts(_nullish(content.get("ordered_steps"), content.get("steps")))
    parts = [
        f"<h2>{escape_html_text(heading)}</h2>" if heading else "",
        *(f"<p>{escape_html_text(item)}</p>" for item in _semantic_texts(content.get("paragraphs"))),
        f"<ul>{''.join(f'<li>{escape_html_text(item)}</li>' for item in bullets)}</ul>" if bullets else "",
        f"<ol>{''.join(f'<li>{escape_html_text(item)}</li>' for item in steps)}</ol>" if steps else "",
        *(f"<blockquote>{escape_html_text(item)}</blockquote>"
          for item in _semantic_texts(_nullish(content.get("warnings"), content.get("warning")))),
        f"<table><tbody>{''.join(rows)}</tbody></table>" if rows else "",
    ]
    output = "".join(part for part in parts if part)
    if not output:
        return None
    rendered = sanitize_lesson_author_html(output)
    if html_contract_code(rendered):
        raise _Rejected("HTML_SEMANTIC_RENDER_INVALID")
    return rendered


def _normalized_html(component: dict[str, Any]) -> str:
    semantic = _nullish(component.get("semantic_content"), component.get("semanticContent"),
                        component.get("structured_content"))
    rendered = render_semantic_html(semantic)
    raw = rendered if rendered is not None else _nullish(component.get("html"), component.get("data"),
                                                         component.get("content"))
    html = sanitize_lesson_author_html(raw)
    if html_contract_code(html):
        raise _Rejected("HTML_CONTRACT_INVALID")
    minimum = 1 if component.get("source_locked_fallback") is True else MIN_UNIT_HTML_TEXT_CHARS
    if js_len(strip_html(html)) < minimum:
        raise _Rejected("HTML_TOO_THIN")
    if js_len(html) > MAX_UNIT_HTML_CHARS:
        raise _Rejected("HTML_TOO_LARGE")
    return html


_COURSE_HTML_MAX_BYTES: Final = 250_000
_COURSE_HTML_MAX_ROWS: Final = 100
_COURSE_HTML_MAX_COLUMNS: Final = 50
_COURSE_HTML_MAX_CELLS: Final = 10_000


def workspace_html_code(html: str) -> str | None:
    """``sanitizeCourseHtmlData`` size limits (its tag allow-list is wider than the html contract)."""

    if len(html.encode("utf-8", "surrogatepass")) > _COURSE_HTML_MAX_BYTES:
        return "HTML_TOO_LARGE"
    cells = 0
    for table in re.findall(r"<table\b[^>]*>[\s\S]*?</table>", html, re.IGNORECASE | re.ASCII):
        if len(re.findall(r"<tr\b[^>]*>", table, re.I | re.A)) > _COURSE_HTML_MAX_ROWS:
            return "HTML_TABLE_TOO_LARGE"
        for row in re.findall(r"<tr\b[^>]*>[\s\S]*?</tr>", table, re.IGNORECASE | re.ASCII):
            row_cells = len(re.findall(r"<(?:td|th)\b[^>]*>", row, re.IGNORECASE | re.ASCII))
            if row_cells > _COURSE_HTML_MAX_COLUMNS:
                return "HTML_TABLE_TOO_LARGE"
            cells += row_cells
    return "HTML_TABLE_TOO_LARGE" if cells > _COURSE_HTML_MAX_CELLS else None


# --- IDM W5 budget (idmPythonHtmlVisibleText / idmOutputBudgetFinding) ------------------------------
def _node_word_count(text: str) -> int:
    r"""``idmPythonWordCount``: runs of letters, digits and ``_`` (``[\p{L}\p{N}_]+``)."""

    count, inside = 0, False
    for char in text:
        word = char == "_" or unicodedata.category(char)[0] in {"L", "N"}
        count += word and not inside
        inside = word
    return count


def idm_html_visible_text(component: Mapping[str, Any]) -> str | None:
    semantic = component.get("semantic_content")
    if semantic is None:
        raw = next((value for value in (component.get("html"), component.get("data"), component.get("content"))
                    if _js_truthy(value)), None)
        html = raw if isinstance(raw, str) else ""
        return re.sub(r"\s+", " ", replace_tags(html, " ")).strip()
    if not isinstance(semantic, dict) or not semantic:
        return None
    try:
        flat = _flatten_ordered(semantic)
    except ValueError:
        return None
    visible: list[str] = []
    if flat.get("heading") is not None:
        heading = flat["heading"]
        if not isinstance(heading, str) or not heading.strip():
            return None
        visible.append(heading.strip())
    for key, alias in (("paragraphs", "paragraphs"), ("bullet_points", "bullets"), ("ordered_steps", "steps"),
                       ("warnings", "warning")):
        items = flat[key] if key in flat else flat.get(alias)
        if items is None:
            continue
        if not isinstance(items, list):
            return None
        for item in items:
            if not isinstance(item, str) or not item.strip():
                return None
            visible.append(item.strip())
    rows = flat["comparison_rows"] if "comparison_rows" in flat else flat.get("table_rows")
    if rows is not None:
        if not isinstance(rows, list):
            return None
        for row in rows:
            label, value = _record(row).get("label"), _record(row).get("value")
            if not isinstance(label, str) or not isinstance(value, str) or not label.strip() or not value.strip():
                return None
            visible.extend([label.strip(), value.strip()])
    return " ".join(visible) if visible else None


def idm_budget_code(component: Mapping[str, Any], budget: Mapping[str, int]) -> str | None:
    text = idm_html_visible_text(component)
    if text is None:
        return "IDM_HTML_VISIBLE_TEXT_INVALID"
    if len(text) > budget["max_visible_chars"] or _node_word_count(text) > budget["max_words"]:
        return "IDM_HTML_DENSITY_EXCEEDED"
    return None
