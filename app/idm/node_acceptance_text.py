"""JavaScript text semantics and the finding types of the Node acceptance mirror (``app.idm.node_acceptance``).

Node measures strings in UTF-16 code units, trims with the ECMAScript white-space set and slices on code
units; these helpers reproduce that, so a Python verdict matches the Node one character for character.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal

from app.core.markup import replace_tags

AcceptanceCheck = Literal["normalization", "response", "coverage", "idm_budget", "workspace_component"]


# --- JavaScript text semantics ------------------------------------------------------------------
# ECMAScript WhiteSpace + LineTerminator: what `\s` and `String.prototype.trim` use.
_JS_WS: Final = "\t\n\v\f\r \u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
_JS_WS_RUN: Final = re.compile(f"[{_JS_WS}]+")
_JS_TRIM: Final = re.compile(f"^[{_JS_WS}]+|[{_JS_WS}]+$")
_ENTITY_RE: Final = re.compile(r"&(?:[a-z]+|#\d+|#x[a-f0-9]+);", re.IGNORECASE | re.ASCII)
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

    return js_trim(_JS_WS_RUN.sub(" ", _ENTITY_RE.sub(" ", replace_tags(value, " "))))


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
    text = replace_tags(text, " ")
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
    # A kept document (QLT-6 preserve_source): Node skips the instructional-quality rules of generated html
    # (``enforceGeneratedContentQuality: false``) and checks the exact text instead (``app.idm.preserve_render``).
    preserve_source: bool = False
