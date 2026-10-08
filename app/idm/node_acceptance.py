"""Mirror of Node's revision-0 unit acceptance, applied before an IDM unit is returned.

Node accepts a ``/unit`` answer only after it normalizes the unit
(``chat.service.normalizeLessonAuthorProposal``), checks it
(``lesson-author-content-contract.logic.lessonAuthorGeneratedUnitCoverageFinding``, the IDM W5
budget, ``lesson-author-workspace-component.logic.workspaceComponentContent``) and stops at the
first failure with ``ORCHESTRATION_V2_UNIT_BASELINE_INVALID`` (or ``..._NORMALIZATION_INVALID``).
The shared staged validator does not cover every one of those rules: run c2e5ac41 returned two
worksheet units that Python validated and Node rejected, and the unit was lost because the
``fallback_only`` retry had no fallback for its slots.

This module re-implements those checks on the unit exactly as Python is about to return it, so the
IDM writer can repair or replace the failing slot instead. Every finding carries the check and the
reason code Node logs (``acceptance_check`` / ``acceptance_code`` / ``acceptance_path`` in the
worker log) and, where Node only has a coarse code, a finer ``detail`` for the repair prompt.

JavaScript semantics are reproduced where they matter: string lengths in UTF-16 code units, the
ECMAScript white-space set, ``String.prototype.slice`` on code units, ASCII ``\\d``/``\\w``/``\\b``.
``tests/test_idm_node_acceptance.py`` pins the rules and the backend test
``lesson-author-idm-acceptance.cross-language.test.ts`` (through ``tests/idm_contract_bridge.py``) keeps
the verdicts in parity with the real Node code. Diagram geometry is computed by Node itself and is
not mirrored; only the diagram text rules are.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal

AcceptanceCheck = Literal["normalization", "response", "coverage", "idm_budget", "workspace_component"]

# --- JavaScript text semantics ------------------------------------------------------------------
# ECMAScript WhiteSpace + LineTerminator: what `\s` and `String.prototype.trim` use.
_JS_WS: Final = "\t\n\v\f\r \u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
_JS_WS_RUN: Final = re.compile(f"[{_JS_WS}]+")
_JS_TRIM: Final = re.compile(f"^[{_JS_WS}]+|[{_JS_WS}]+$")
_ENTITY_RE: Final = re.compile(r"&(?:[a-z]+|#\d+|#x[a-f0-9]+);", re.IGNORECASE | re.ASCII)
_TAG_RE: Final = re.compile(r"<[^>]+>")
_COMBINING_RE: Final = re.compile("[\u0300-\u036f]")
_BMP_MAX: Final = 0xFFFF
_SURROGATES: Final = re.compile("[\ud800-\udfff]")


def js_len(value: str) -> int:
    """``String.prototype.length``: UTF-16 code units."""

    return sum(2 if ord(char) > _BMP_MAX else 1 for char in value)


def js_trim(value: str) -> str:
    return _JS_TRIM.sub("", value)


def js_slice(value: str, end: int) -> str:
    """``value.slice(0, end)`` on UTF-16 code units (may leave a lone high surrogate, as JS does)."""

    if js_len(value) <= end:
        return value
    return value.encode("utf-16-le", "surrogatepass")[: 2 * end].decode("utf-16-le", "surrogatepass")


def _js_truthy(value: Any) -> bool:
    if isinstance(value, list):
        return True  # callers test arrays by length first, as Node does
    if isinstance(value, dict):
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0 and not (isinstance(value, float) and math.isnan(value))
    if isinstance(value, str):
        return value != ""
    return value is not None


def _js_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _js_string(value: Any) -> str:
    """``String(value)`` for the JSON values a provider answer can hold."""

    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e21:  # noqa: PLR2004 - JS Number#toString
        return str(int(value))
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    return "[object Object]" if isinstance(value, dict) else ",".join(_js_string(item) for item in value)


def _record(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _nullish(*values: Any) -> Any:
    """``a ?? b ?? c``."""

    for value in values:
        if value is not None:
            return value
    return None


def read_string(value: Any, fallback: str, max_length: int) -> str:
    raw = js_trim(value) if isinstance(value, str) else ""
    return js_slice(raw or fallback, max_length)


def read_scalar_string(value: Any, fallback: str, max_length: int) -> str:
    raw = ""
    if isinstance(value, str):
        raw = js_trim(value)
    elif (_js_number(value) and math.isfinite(value)) or isinstance(value, bool):
        raw = _js_string(value)
    return js_slice(raw or fallback, max_length)


def visible_html_text(value: str) -> str:
    """``visibleHtmlText`` of lesson-author-content-contract.logic."""

    return js_trim(_JS_WS_RUN.sub(" ", _ENTITY_RE.sub(" ", _TAG_RE.sub(" ", value))))


def instructional_fold(value: str) -> str:
    """``normalizedInstructionalText``: NFD, no U+0300-036F marks, lower case, letters/digits only."""

    text = _COMBINING_RE.sub("", unicodedata.normalize("NFD", value)).lower()
    words: list[str] = []
    run: list[str] = []
    for char in text:
        if unicodedata.category(char)[0] in {"L", "N"}:
            run.append(char)
        elif run:
            words.append("".join(run))
            run = []
    if run:
        words.append("".join(run))
    return " ".join(words)


def strip_html(value: str) -> str:
    """``stripHtml`` of chat.service (the normalizer's thin-content measure)."""

    text = re.sub(r"<style[^>]*>[\s\S]*?</style>", "", value, flags=re.IGNORECASE | re.ASCII)
    text = re.sub(r"<script[^>]*>[\s\S]*?</script>", "", text, flags=re.IGNORECASE | re.ASCII)
    text = _TAG_RE.sub(" ", text)
    for entity, replacement in (("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'),
                                ("&#39;", "'")):
        text = re.sub(re.escape(entity), replacement, text, flags=re.IGNORECASE | re.ASCII)
    return js_trim(_JS_WS_RUN.sub(" ", text))


def escape_html_text(value: Any) -> str:
    """``escapeHtmlText`` of the semantic renderer."""

    text = "" if value is None else value if isinstance(value, str) else _js_string(value)
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
            .replace("'", "&#39;"))


def escape_xml(value: Any) -> str:
    text = value if isinstance(value, str) else _js_string(value)
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
            .replace("'", "&apos;"))


# --- findings ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class AcceptanceFinding:
    """A rule Node would reject the unit for; ``component_index`` is ``None`` for a unit-level rule."""

    check: AcceptanceCheck
    code: str
    component_index: int | None
    # A finer, safe reason for the repair prompt where Node's code is coarse.
    detail: str | None = None
    # Item positions (as in the returned slot payload) that break the rule, for item-level pruning.
    items: tuple[int, ...] = ()

    @property
    def path(self) -> str:
        return "unit" if self.component_index is None else f"components[{self.component_index}]"

    def as_log(self) -> str:
        return f"{self.check}:{self.code}@{self.path}" + (f"#{self.detail}" if self.detail else "")


class _Rejected(Exception):
    def __init__(self, code: str, detail: str | None = None, items: Sequence[int] = ()) -> None:
        super().__init__(code)
        self.code, self.detail, self.items = code, detail, tuple(items)


@dataclass(frozen=True)
class AcceptanceContext:
    """The parts of the server-owned unit contract the Node checks read."""

    unit_title: str
    unit_source_fact_ids: tuple[str, ...]
    plans: tuple[Mapping[str, Any], ...]
    exact_identifiers: tuple[str, ...]
    # IDM W5 budget (``max_words``/``max_visible_chars``); ``None`` for a legacy contract.
    budget: Mapping[str, int] | None = None
    # Plan indexes of IDM worksheets (html slot with role practice).
    worksheet_slots: frozenset[int] = field(default_factory=frozenset)


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


# --- problem -----------------------------------------------------------------------------------------
_MULTI_SELECT: Final = frozenset({"multiple_select", "multi_select", "checkbox", "checkboxes", "choiceresponse"})
_DROPDOWN: Final = frozenset({"dropdown", "option", "select", "option_response", "optionresponse"})
_NUMERICAL: Final = frozenset({"numerical", "numeric", "number", "numerical_response", "numericalresponse"})
_SHORT_TEXT: Final = frozenset({"short_text", "short_answer", "text", "string", "string_response",
                                "stringresponse", "free_text"})
_ANSWER_FIELDS: Final = ("answers", "correct_answers", "correctAnswers", "answer", "correct_answer", "correctAnswer",
                         "expected_answer", "expectedAnswer", "value")
_MAX_QUESTION: Final = 1_000
_MAX_EXPLANATION: Final = 1_500
_MAX_CHOICE: Final = 500
_MAX_ANSWER: Final = 500
_MAX_NUMERIC_ANSWER: Final = 120
_MAX_CHOICES: Final = 6
_MAX_DROPDOWN_OPTIONS: Final = 8
_MIN_CHOICES: Final = 2


def _subtype(component: dict[str, Any]) -> str:
    raw = read_scalar_string(_nullish(component.get("problem_type"), component.get("subtype"),
                                      component.get("response_type"), component.get("type")), "multiple_choice", 60)
    kind = re.sub(f"[{_JS_WS}-]+", "_", raw.lower())
    for names, name in ((_MULTI_SELECT, "multiple_select"), (_DROPDOWN, "dropdown"), (_NUMERICAL, "numerical"),
                        (_SHORT_TEXT, "short_text")):
        if kind in names:
            return name
    return "multiple_choice"


def _comparable(value: str) -> str:
    return js_trim(value).lower()


def _answers(component: dict[str, Any], max_length: int = _MAX_ANSWER) -> list[str]:
    for source in (component.get(name) for name in _ANSWER_FIELDS):
        values = source if isinstance(source, list) else [source]
        answers = [read_scalar_string(_nullish(_record(item).get("answer"), _record(item).get("text"),
                                               _record(item).get("label"), _record(item).get("value"), item),
                                      "", max_length) for item in values]
        unique = list(dict.fromkeys(answer for answer in answers if answer))
        if unique:
            return unique
    return []


def _choices(raw_choices: list[Any], answers: list[str], *, single: bool, limit: int) -> list[tuple[str, bool]]:
    answer_set = {_comparable(answer) for answer in answers}
    choices: list[tuple[str, bool]] = []
    for value in raw_choices:
        if isinstance(value, str) or _js_number(value):
            text = read_scalar_string(value, "", _MAX_CHOICE)
            choices.append((text, _comparable(text) in answer_set))
            continue
        choice = _record(value)
        correct_value = _nullish(choice.get("correct"), choice.get("is_correct"), choice.get("answer"))
        text = read_scalar_string(_nullish(choice.get("text"), choice.get("label"), choice.get("value"),
                                           choice.get("answer")), "", _MAX_CHOICE)
        choices.append((text, correct_value is True or _js_string(correct_value).lower() == "true"
                        or (bool(answer_set) and _comparable(text) in answer_set)))
    choices = [choice for choice in choices if choice[0]][:limit]
    if len(choices) < _MIN_CHOICES:
        raise _Rejected("PROBLEM_CHOICE_COUNT")
    if len({_comparable(text) for text, _correct in choices}) != len(choices):
        raise _Rejected("PROBLEM_DUPLICATE_CHOICES")
    correct = sum(1 for _text, flag in choices if flag)
    if not correct or (single and correct != 1):
        raise _Rejected("PROBLEM_CORRECT_ANSWER_INVALID")
    return choices


def _problem_xml(component: dict[str, Any]) -> str | None:
    """``normalizeProblemComponent``: the single-choice XML, or ``None`` for another (valid) subtype."""

    kind = _subtype(component)
    question = read_string(_nullish(component.get("question"), component.get("prompt"), component.get("label")), "",
                           _MAX_QUESTION)
    if not question:
        raise _Rejected("PROBLEM_QUESTION_REQUIRED")
    explanation = read_string(_nullish(component.get("explanation"), component.get("solution")), "", _MAX_EXPLANATION)
    if kind in {"numerical", "short_text"}:
        if not _answers(component, _MAX_NUMERIC_ANSWER if kind == "numerical" else _MAX_ANSWER):
            raise _Rejected("PROBLEM_ANSWER_REQUIRED")
        return None
    if kind == "dropdown":
        options = component.get("options") if isinstance(component.get("options"), list) else component.get("choices")
        _choices(options if isinstance(options, list) else [], _answers(component), single=True,
                 limit=_MAX_DROPDOWN_OPTIONS)
        return None
    raw_choices = _list(component.get("choices"))
    choices = _choices(raw_choices, _answers(component), single=kind != "multiple_select", limit=_MAX_CHOICES)
    if kind == "multiple_select":
        return None
    solution = (f'\n  <solution><div class="detailed-solution"><p>{escape_xml(explanation)}</p></div></solution>'
                if explanation else "")
    return "\n".join([
        "<problem>", "  <multiplechoiceresponse>", f"    <label>{escape_xml(question)}</label>",
        '    <choicegroup type="MultipleChoice">',
        "\n".join(f'      <choice correct="{"true" if correct else "false"}">{escape_xml(text)}</choice>'
                  for text, correct in choices),
        "    </choicegroup>", f"  </multiplechoiceresponse>{solution}", "</problem>",
    ])


_LABEL_RE: Final = re.compile(r"<label>([\s\S]*?)</label>", re.IGNORECASE | re.ASCII)
_CHOICE_RE: Final = re.compile(f'<choice[{_JS_WS}]+correct="(true|false)">([\\s\\S]*?)</choice>', re.IGNORECASE)
_SOLUTION_RE: Final = re.compile(r"<solution>[\s\S]*?<p>([\s\S]*?)</p>[\s\S]*?</solution>", re.IGNORECASE | re.ASCII)
MIN_QUESTION_CHARS: Final = 20
MIN_CHOICE_CHARS: Final = 8
MIN_EXPLANATION_CHARS: Final = 20
_MIN_SINGLE_CHOICES: Final = 3


def single_choice_problem_code(xml: Any) -> str | None:
    """``lessonAuthorSingleChoiceProblemFinding`` code."""

    if not isinstance(xml, str) or not js_trim(xml):
        return "PROBLEM_XML_EMPTY"
    if (not re.search(r"<multiplechoiceresponse>", xml, re.I | re.A)
            or not re.search(f'<choicegroup[{_JS_WS}]+type="MultipleChoice">', xml, re.IGNORECASE)):
        return "PROBLEM_NOT_SINGLE_CHOICE"
    if re.search(r"<(?:stringresponse|numericalresponse|optionresponse|checkboxgroup|choiceresponse)>", xml,
                 re.IGNORECASE | re.ASCII):
        return "PROBLEM_UNSUPPORTED_RESPONSE"
    labels = _LABEL_RE.findall(xml)
    if len(labels) != 1 or js_len(visible_html_text(labels[0])) < MIN_QUESTION_CHARS:
        return "PROBLEM_QUESTION_INCOMPLETE"
    choices = _CHOICE_RE.findall(xml)
    if not _MIN_SINGLE_CHOICES <= len(choices) <= _MAX_CHOICES:
        return "PROBLEM_CHOICE_COUNT"
    if sum(1 for flag, _text in choices if flag.lower() == "true") != 1:
        return "PROBLEM_CORRECT_COUNT"
    texts = [visible_html_text(text) for _flag, text in choices]
    folded = [instructional_fold(text) for text in texts]
    if any(js_len(text) < MIN_CHOICE_CHARS for text in texts) or len(set(folded)) != len(folded):
        return "PROBLEM_CHOICES_NOT_DISTINCT"
    correct = next(text for (flag, _raw), text in zip(choices, texts, strict=True) if flag.lower() == "true")
    if _URL_ONLY_RE.search(correct) or _EMAIL_ONLY_RE.search(correct) or _PHONE_ONLY_RE.search(correct):
        return "PROBLEM_CORRECT_BOILERPLATE"
    solution = _SOLUTION_RE.search(xml)
    if not solution or js_len(visible_html_text(solution.group(1))) < MIN_EXPLANATION_CHARS:
        return "PROBLEM_EXPLANATION_MISSING"
    return None


# --- workspace payload checks (lesson-author-workspace-component / -problem) -------------------------
PAYLOAD_INVALID: Final = "WORKSPACE_COMPONENT_PAYLOAD_INVALID"
REFERENCE_INVALID: Final = "WORKSPACE_COMPONENT_REFERENCE_INVALID"
_XML_TEXT_BAD: Final = re.compile("[\u0000-\u0008\u000b\u000c\u000e-\u001f\ufffe\uffff]")
_PLAIN_TEXT_BAD: Final = re.compile("[<>\u0000-\u0008\u000b\u000c\u000e-\u001f]")
_MAX_WORKSPACE_TEXT: Final = 8_000
_MAX_PROBLEM_TEXT: Final = 4_000
_MAX_PROBLEM_XML: Final = 100_000
_ENTITY_DECODE: Final = {"amp": "&", "lt": "<", "gt": ">", "quot": '"', "apos": "'"}


def _lone_surrogate(value: str) -> bool:
    return bool(_SURROGATES.search(value))


def _xml_text(value: str) -> bool:
    return not _XML_TEXT_BAD.search(value) and not _lone_surrogate(value)


def plain_text_valid(value: Any) -> bool:
    """The workspace ``text`` schema of FAQ/sortable/crossword/diagram: 1-8000 units, no ``<``/``>``/C0."""

    return (isinstance(value, str) and 1 <= js_len(value) <= _MAX_WORKSPACE_TEXT and bool(js_trim(value))
            and not _PLAIN_TEXT_BAD.search(value))


def _decode_xml_text(value: str) -> str:
    if re.search(r"[<>]", value) or re.search(r"&(?!(?:amp|lt|gt|quot|apos);)", value):
        raise _Rejected(PAYLOAD_INVALID, "PROBLEM_XML_UNSUPPORTED")
    return re.sub(r"&(amp|lt|gt|quot|apos);", lambda match: _ENTITY_DECODE[match.group(1)], value)


def _problem_text(value: str, maximum: int, *, required: bool) -> bool:
    return js_len(value) <= maximum and _xml_text(value) and (not required or (js_len(value) >= 1
                                                                                 and bool(js_trim(value))))


def workspace_problem_code(xml: str) -> str | None:
    """``decodeWorkspaceProblem`` + ``readWorkspaceProblem`` for the single-choice XML; a detail code."""

    if js_len(xml) > _MAX_PROBLEM_XML:
        return "PROBLEM_XML_UNSUPPORTED"
    outer = re.fullmatch(f"[{_JS_WS}]*<problem>[{_JS_WS}]*([\\s\\S]*?)[{_JS_WS}]*</problem>[{_JS_WS}]*", xml)
    if not outer:
        return "PROBLEM_XML_UNSUPPORTED"
    body, explanation = outer.group(1), ""
    try:
        solution = re.search(f'[{_JS_WS}]*<solution><div class="detailed-solution"><p>([^<]*)</p></div></solution>'
                             f"[{_JS_WS}]*\\Z", body)
        if solution:
            explanation, body = _decode_xml_text(solution.group(1)), body[:solution.start()]
        match = re.fullmatch(
            f"[{_JS_WS}]*<multiplechoiceresponse>[{_JS_WS}]*<label>([^<]*)</label>[{_JS_WS}]*"
            f'<choicegroup(?: type="MultipleChoice")?>([\\s\\S]*?)</choicegroup>[{_JS_WS}]*'
            f"</multiplechoiceresponse>[{_JS_WS}]*", body)
        if not match:
            return "PROBLEM_XML_UNSUPPORTED"
        choices: list[tuple[str, bool]] = []

        def take(found: re.Match[str]) -> str:
            choices.append((_decode_xml_text(found.group(2)), found.group(1) == "true"))
            return ""

        remaining = re.sub(r'<choice correct="(true|false)">([^<]*)</choice>', take, match.group(2))
        if js_trim(remaining):
            return "PROBLEM_XML_UNSUPPORTED"
        question = _decode_xml_text(match.group(1))
    except _Rejected as error:
        return error.detail
    if (not _problem_text(question, _MAX_PROBLEM_TEXT, required=True)
            or not _problem_text(explanation, _MAX_WORKSPACE_TEXT, required=False)
            or not _MIN_CHOICES <= len(choices) <= _MAX_DROPDOWN_OPTIONS
            or any(not _problem_text(text, _MAX_PROBLEM_TEXT, required=True) for text, _correct in choices)):
        return "PROBLEM_TEXT_INVALID"
    labels = [unicodedata.normalize("NFKC", text).lower() for text, _correct in choices]
    if len(set(labels)) != len(labels) or sum(1 for _text, correct in choices if correct) != 1:
        return "PROBLEM_CHOICES_NOT_DISTINCT"
    return None


def _faq_items(component: dict[str, Any]) -> list[tuple[int, str, str]]:
    """``normalizeFaqComponent`` items as (slot position, question, answer)."""

    raw_items = _list(component.get("items"))
    items = []
    for position, raw in enumerate(raw_items):
        item = _record(raw)
        question = read_string(_nullish(item.get("question"), item.get("q")), "", 500)
        answer = read_string(_nullish(item.get("answer"), item.get("a"), item.get("content")), "", 2_000)
        if question and answer:
            items.append((position, question, answer))
    return items[:8]


def _unique_key(value: str) -> str:
    return js_trim(unicodedata.normalize("NFKC", value)).lower()


def _later_duplicates(keys: Sequence[tuple[int, str]]) -> list[int]:
    seen: set[str] = set()
    duplicates: list[int] = []
    for position, key in keys:
        if key in seen:
            duplicates.append(position)
        seen.add(key)
    return duplicates


def _faq_findings(component: dict[str, Any]) -> None:
    items = _faq_items(component)
    if len(items) < _MIN_CHOICES:
        raise _Rejected("FAQ_ITEM_COUNT")
    bad = [position for position, question, answer in items
           if not plain_text_valid(question) or not plain_text_valid(answer)]
    if bad:
        raise _Rejected(PAYLOAD_INVALID, "FAQ_TEXT_FORBIDDEN_CHARACTER", bad)
    duplicates = _later_duplicates([(position, _unique_key(question)) for position, question, _answer in items])
    if duplicates:
        raise _Rejected(REFERENCE_INVALID, "FAQ_DUPLICATE_QUESTION", duplicates)


_SORTABLE_DEFAULT_QUESTION: Final = "Sap xep cac muc theo dung thu tu."
_MIN_SORTABLE_ITEMS: Final = 3
_MAX_SORTABLE_ITEMS: Final = 10


def _sortable_findings(component: dict[str, Any]) -> None:
    raw_items: list[Any] = next((_list(component[key]) for key in ("items", "ordered_items", "steps")
                                 if isinstance(component.get(key), list)), [])
    items = []
    for raw in raw_items:
        record = _record(raw)
        text = (read_string(raw, "", 500) if isinstance(raw, str)
                else read_string(_nullish(record.get("text"), record.get("label"), record.get("title")), "", 500))
        if text:
            items.append(text)
    items = items[:_MAX_SORTABLE_ITEMS]
    if len(items) < _MIN_SORTABLE_ITEMS:
        raise _Rejected("SORTABLE_ITEM_COUNT")
    question = read_string(_nullish(component.get("question_text"), component.get("question"),
                                    component.get("prompt")), _SORTABLE_DEFAULT_QUESTION, 500)
    if not plain_text_valid(question) or any(not plain_text_valid(item) for item in items):
        raise _Rejected(PAYLOAD_INVALID, "SORTABLE_TEXT_FORBIDDEN_CHARACTER")
    if _later_duplicates([(index, _unique_key(item)) for index, item in enumerate(items)]):
        raise _Rejected(REFERENCE_INVALID, "SORTABLE_DUPLICATE_ITEM")


_CROSSWORD_ANSWER_RE: Final = re.compile(r"^[A-Z0-9]{2,24}\Z")
_MIN_CROSSWORD_WORDS: Final = 3
_MAX_CROSSWORD_WORDS: Final = 10
_MAX_CROSSWORD_ANSWER: Final = 24


def _crossword_answer(value: Any) -> str:
    text = re.sub("\u0111", "D", read_string(value, "", 80), flags=re.IGNORECASE)
    text = _COMBINING_RE.sub("", unicodedata.normalize("NFD", text)).upper()
    return re.sub(r"[^A-Z0-9]", "", text)[:_MAX_CROSSWORD_ANSWER]


def _crossword_findings(component: dict[str, Any]) -> None:
    raw_words = _list(component.get("words"))
    words = []
    for index, raw in enumerate(raw_words):
        word = _record(raw)
        answer = _crossword_answer(_nullish(word.get("answer"), word.get("term"), word.get("text")))
        clue = read_string(_nullish(word.get("clue"), word.get("definition"), word.get("hint")), "", 500)
        hint = read_string(word.get("hint"), "", 500)
        if len(answer) >= _MIN_CHOICES and clue:
            words.append((index, answer, clue, hint))
    words = words[:_MAX_CROSSWORD_WORDS]
    if len(words) < _MIN_CROSSWORD_WORDS:
        raise _Rejected("CROSSWORD_WORD_COUNT")
    if any(not _CROSSWORD_ANSWER_RE.search(answer) or not plain_text_valid(clue) or re.search("[<>]", hint)
           for _row, answer, clue, hint in words):
        raise _Rejected(PAYLOAD_INVALID, "CROSSWORD_TEXT_FORBIDDEN_CHARACTER")
    # The renderer keeps the source row of every word; a dropped word leaves a gap Node rejects.
    if any(row != position for position, (row, _answer, _clue, _hint) in enumerate(words)) or _later_duplicates(
            [(position, _unique_key(answer)) for position, (_row, answer, _clue, _hint) in enumerate(words)]):
        raise _Rejected(REFERENCE_INVALID, "CROSSWORD_WORD_LAYOUT")


_DIAGRAM_ESCAPED_NEWLINE: Final = re.compile(r"(?:\\r\\n|\\n|\\r|(?<![^\W_])/n(?![^\W_]))", re.IGNORECASE)
_DIAGRAM_CONTROL: Final = re.compile("[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f\u2028\u2029]")
_MIN_DIAGRAM_NODES: Final = 2
_MAX_DIAGRAM_NODES: Final = 12


def _diagram_text(value: Any, maximum: int) -> str:
    """``normalizeDiagramDisplayText``."""

    if not isinstance(value, str):
        return ""
    text = _DIAGRAM_CONTROL.sub(" ", _DIAGRAM_ESCAPED_NEWLINE.sub(" ", value))
    return js_slice(js_trim(_JS_WS_RUN.sub(" ", text)), maximum)


def _diagram_findings(component: dict[str, Any]) -> None:
    """Node lays the diagram out itself; only its text rules are mirrored."""

    raw_nodes = _list(component.get("nodes"))
    texts: list[str] = []
    labels = 0
    for raw in raw_nodes:
        node = _record(raw)
        label = _diagram_text(read_string(raw, "", 120) if isinstance(raw, str) else read_string(
            _nullish(node.get("label"), node.get("title"), node.get("name")), "", 120), 120)
        if not label:
            continue
        labels += 1
        if labels > _MAX_DIAGRAM_NODES:
            break
        tooltip = read_string(_nullish(node.get("tooltip"), node.get("description"), node.get("summary")), label, 500)
        texts.extend([label, _diagram_text(tooltip, 500)])
    if labels < _MIN_DIAGRAM_NODES:
        raise _Rejected("DIAGRAM_NODE_COUNT")
    texts.append(_diagram_text(read_string(_nullish(component.get("name"), component.get("title")), "Main Diagram",
                                           120), 120))
    edges = _list(component.get("edges"))
    texts.extend(label for edge in edges if (label := _diagram_text(read_string(_record(edge).get("label"), "", 120),
                                                                     120)))
    if any(re.search("[<>]", text) for text in texts):
        raise _Rejected(PAYLOAD_INVALID, "DIAGRAM_TEXT_FORBIDDEN_CHARACTER")


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
        return re.sub(r"\s+", " ", _TAG_RE.sub(" ", html)).strip()
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


# --- the unit ----------------------------------------------------------------------------------------
def _exact_ids(value: Any, expected: Sequence[str]) -> bool:
    return (isinstance(value, list) and len(value) == len(expected) and all(isinstance(item, str) for item in value)
            and len(set(value)) == len(value) and all(item in expected for item in value))


def _response_code(component: dict[str, Any], context: AcceptanceContext, index: int) -> str | None:
    """``readOrchestrationV2UnitProviderResponse`` / ``acceptOrchestrationV2GeneratedUnit`` slot identity."""

    metadata = _record(component.get("metadata"))
    plan = context.plans[index]

    def claimed(name: str) -> Any:
        return _nullish(component.get(name), metadata.get(name))

    if (component.get("type") != plan.get("type")
            or claimed("component_plan_id") != plan.get("component_plan_id")):
        return "RESPONSE_COMPONENT_IDENTITY"
    owned = list(plan.get("source_fact_ids") or [])
    if (not _exact_ids(claimed("source_fact_ids"), owned) or not _exact_ids(claimed("covered_source_fact_ids"), owned)
            or not _exact_ids(claimed("supporting_evidence_fact_ids"),
                              list(plan.get("supporting_evidence_fact_ids") or []))):
        return "RESPONSE_FACT_IDS"
    refs = claimed("learning_objective_refs")
    if refs is not None and not _exact_ids(refs, list(plan.get("learning_objective_refs") or [])):
        return "RESPONSE_FACT_IDS"
    return None


_CHECKERS: Final[Mapping[str, Callable[[dict[str, Any]], None]]] = {
    "la_faq": _faq_findings, "la_sortable": _sortable_findings, "la_crossword": _crossword_findings,
    "la_diagram": _diagram_findings,
}


def _merge_nested(component: dict[str, Any]) -> dict[str, Any]:
    nested = _record(component.get("content"))
    return {**nested, **component} if nested else component


def node_acceptance_findings(unit: Mapping[str, Any], context: AcceptanceContext, *,
                             provider_validated: bool) -> list[AcceptanceFinding]:
    """Every rule Node would reject ``unit`` for, in Node's order (the first one is what Node reports)."""

    components = _list(unit.get("components"))
    if (unit.get("title") != context.unit_title or len(components) != len(context.plans)
            or not _exact_ids(unit.get("source_fact_ids"), list(context.unit_source_fact_ids))):
        return [AcceptanceFinding("response", "RESPONSE_UNIT_IDENTITY", None)]
    normalization: list[AcceptanceFinding] = []
    response: list[AcceptanceFinding] = []
    coverage: list[AcceptanceFinding] = []
    budget: list[AcceptanceFinding] = []
    workspace: list[AcceptanceFinding] = []
    for index, raw in enumerate(components):
        if (code := _response_code(_record(raw), context, index)) is not None:
            response.append(AcceptanceFinding("response", code, index))
        component = _merge_nested(_record(raw))
        kind = component.get("type")
        plan = context.plans[index]
        try:
            if kind == "html":
                html = _normalized_html(component)
            elif kind == "problem":
                xml = _problem_xml(component)
            elif isinstance(kind, str) and kind in _CHECKERS:
                _CHECKERS[kind](component)
            else:
                raise _Rejected("NORMALIZATION_COMPONENT_TYPE")
        except _Rejected as error:
            check: AcceptanceCheck = "workspace_component" if error.code.startswith("WORKSPACE_") else "normalization"
            target = workspace if check == "workspace_component" else normalization
            target.append(AcceptanceFinding(check, error.code, index, error.detail, error.items))
            continue
        if kind == "html":
            if (code := html_contract_code(html, plan.get("required_artifacts") or [])) is None:
                code = html_quality_code(html, context.exact_identifiers, worksheet=index in context.worksheet_slots)
            if code is not None:
                coverage.append(AcceptanceFinding("coverage", code, index))
            if (detail := workspace_html_code(html)) is not None:
                workspace.append(AcceptanceFinding("workspace_component", PAYLOAD_INVALID, index, detail))
            if provider_validated and context.budget is not None and (
                    code := idm_budget_code(_record(raw), context.budget)) is not None:
                budget.append(AcceptanceFinding("idm_budget", code, index))
        elif kind == "problem":
            code = single_choice_problem_code(xml) if xml is not None else "PROBLEM_NOT_SINGLE_CHOICE"
            if code is not None:
                coverage.append(AcceptanceFinding("coverage", code, index))
            if xml is not None and (detail := workspace_problem_code(xml)) is not None:
                workspace.append(AcceptanceFinding("workspace_component", PAYLOAD_INVALID, index, detail))
    # Node rejects at the first failing stage; every finding of it is kept so all slots can be repaired.
    for stage in (normalization, response, coverage, budget, workspace):
        if stage:
            return [*stage, *(item for later in (normalization, response, coverage, budget, workspace)
                              if later is not stage for item in later)]
    return []


def acceptance_log(findings: Iterable[AcceptanceFinding], limit: int = 16) -> list[str]:
    return [finding.as_log() for finding in findings][:limit]


__all__ = [
    "AcceptanceContext", "AcceptanceFinding", "acceptance_log", "html_contract_code", "html_quality_code",
    "idm_html_visible_text", "instructional_fold", "js_len", "js_slice", "node_acceptance_findings",
    "plain_text_valid", "render_semantic_html", "sanitize_lesson_author_html", "single_choice_problem_code",
    "visible_html_text", "workspace_problem_code",
]
