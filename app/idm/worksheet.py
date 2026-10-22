"""Worksheet slots that do not fit their budget (QC run 8de1c76b, Q2).

The CEO Change Mindset Canvas worksheet (9 cells, 33 facts) was written at 550 and then 452 words against
the 400-word budget of its unit, and the slot fell back to the source-locked html: upper-case labels in
lists and headings, nine cells torn apart, every "[ … ]" placeholder in one paragraph. The budget is
Node's acceptance rule too (``idmUnitOutputBudget``), so a larger worksheet budget needs both sides
(reported, not changed here). Instead:

* :func:`compact_worksheet` brings an over-long worksheet within the budget: first the second and later sentences of
  the template's guidance cells go, then the worked example is cut to two filled entries, then the other optional
  sections go, and only then the worked example (QLT-9). The task, every template field and the self-check list
  stay, so the worksheet stays complete (``qa.worksheet_complete``).
* :func:`worksheet_fallback_html` is the fallback of a worksheet slot: the form's fields as one table
  (field, guiding question, what to write) built from the source lines in their order, the source
  introduction and closing lines as paragraphs, and a self-check list from the placeholders. It returns
  ``None`` when the facts hold no form, and the generic source-locked html is used.

Both are deterministic and never add a claim the source does not make.
"""

from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Callable, Sequence
from typing import Any, Final

from app.content.instructional_quality import TABLE_ROW_RE
from app.idm.acronyms import keeps_capitals
from app.idm.text import idm_fold, trim_at_boundary

COMPACTED_CODE: Final = "IDM_W5_WORKSHEET_COMPACTED"
_PLACEHOLDER_RE: Final = re.compile(r"^\[\s*(.+?)\s*\]$")
_FILL_LEADER_RE: Final = re.compile(r"[.…_·]{4,}")
_TRAILING_ELLIPSIS_RE: Final = re.compile(r"\s*(?:\.{2,}|…)\s*$")
_SENTENCE_END_RE: Final = re.compile(r"(?<=[.!?])\s+")
_CELL_SPLIT_RE: Final = re.compile(r"\s*[;|]\s*")
_LABEL_MAX_WORDS: Final = 8
_HEADING_MAX_WORDS: Final = 12
_UPPER_SHARE: Final = 0.6
_TITLE_CASE_SHARE: Final = 0.8
_MIN_ROWS: Final = 2
_MAX_GROUP: Final = 6
_GUIDANCE_CELL_CHARS: Final = 160
_TEMPLATE_KINDS: Final = frozenset({"table", "steps"})
# The worked example (QLT-9): its heading names an example, or a text opens with an example label.
_EXAMPLE_HEADING_WORDS: Final = ("vi du", "example", "mau hoan thanh", "bai mau", "tham khao")
_EXAMPLE_OPENING_RE: Final = re.compile(r"^\s*(?:ví dụ|illustrative example|worked example|example)\b", re.IGNORECASE)
_TEXT_BLOCKS: Final = frozenset({"paragraph", "task", "warning"})
_EXAMPLE_ENTRIES: Final = 2
_TEXT: Final[dict[str, dict[str, str]]] = {
    "vi": {"task": "Nhiệm vụ", "how": "Điền từng ô dưới đây trên bản của bạn, rồi tự kiểm tra theo danh sách cuối "
                                      "phiếu.",
           "template": "Mẫu cần điền", "field": "Ô cần điền", "question": "Câu hỏi gợi ý", "hint": "Cần ghi",
           "blank": "(ghi trên bản của bạn)", "after": "Lưu ý khi hoàn thành", "check": "Tự kiểm tra",
           "filled": "Đã điền"},
    "en": {"task": "Task", "how": "Fill in each field below on your own copy, then check it against the list at the "
                                  "end.",
           "template": "Template", "field": "Field", "question": "Guiding question", "hint": "What to write",
           "blank": "(write on your own copy)", "after": "Before you finish", "check": "Self-check",
           "filled": "Filled in"},
}


# --- compaction ------------------------------------------------------------------------------------------------
def _blocks(section: dict[str, Any]) -> list[dict[str, Any]]:
    blocks = section.get("blocks")
    return [block for block in blocks if isinstance(block, dict)] if isinstance(blocks, list) else []


def _kinds(section: dict[str, Any]) -> set[str]:
    return {str(block.get("kind")) for block in _blocks(section)}


def _essential(sections: list[dict[str, Any]]) -> set[int]:
    """Sections a worksheet cannot lose: the task, the first template and the closing self-check list."""

    keep = {index for index, section in enumerate(sections) if "task" in _kinds(section)}
    template = next((index for index, section in enumerate(sections) if _kinds(section) & _TEMPLATE_KINDS), None)
    check = next((index for index in range(len(sections) - 1, -1, -1) if "bullets" in _kinds(sections[index])), None)
    keep.update(index for index in (template, check) if index is not None)
    return keep


def _first_sentence(value: str) -> str:
    return trim_at_boundary(_SENTENCE_END_RE.split(value.strip())[0], _GUIDANCE_CELL_CHARS)


def _short_rows(section: dict[str, Any]) -> dict[str, Any]:
    blocks = []
    for block in _blocks(section):
        rows = block.get("rows")
        blocks.append({**block, "rows": [{**row, "value": _first_sentence(str(row.get("value")))}
                                         if isinstance(row, dict) and str(row.get("value") or "").strip() else row
                                         for row in rows]}
                      if block.get("kind") == "table" and isinstance(rows, list) else block)
    return {**section, "blocks": blocks}


def _is_example(section: dict[str, Any]) -> bool:
    """The worked example: a heading that names an example, or a text that opens with an example label."""

    if any(word in idm_fold(str(section.get("heading") or "")) for word in _EXAMPLE_HEADING_WORDS):
        return True
    return any(block.get("kind") in _TEXT_BLOCKS and _EXAMPLE_OPENING_RE.match(str(block.get("text") or ""))
               for block in _blocks(section))


def _two_entries(section: dict[str, Any]) -> dict[str, Any]:
    """The worked example cut to its first ``_EXAMPLE_ENTRIES`` filled rows or items."""

    blocks = []
    for block in _blocks(section):
        rows, items = block.get("rows"), block.get("items")
        if isinstance(rows, list) and len(rows) > _EXAMPLE_ENTRIES:
            blocks.append({**block, "rows": rows[:_EXAMPLE_ENTRIES]})
        elif isinstance(items, list) and len(items) > _EXAMPLE_ENTRIES:
            blocks.append({**block, "items": items[:_EXAMPLE_ENTRIES]})
        else:
            blocks.append(block)
    return {**section, "blocks": blocks}


def compact_worksheet(component: dict[str, Any], fits: Callable[[dict[str, Any]], bool],
                      ) -> tuple[dict[str, Any], bool] | None:
    """The first compaction of a worksheet html slot that ``fits`` (the slot's html rules) and whether it keeps the
    worked example, or ``None``.

    QLT-9 (QC run ffaf7318: 3 of 5 worksheets lost their worked example at 569, 600 and 846 words against 540): the
    example goes last. In order: one-sentence guidance cells; then the example cut to two filled entries; then
    without the other optional sections (an introduction), the example kept; only then without the example, as before.
    """

    semantic = component.get("semantic_content")
    sections = semantic.get("sections") if isinstance(semantic, dict) else None
    if not isinstance(semantic, dict) or not isinstance(sections, list):
        return None
    sections = [section for section in sections if isinstance(section, dict)]
    keep = _essential(sections)
    example = {index for index, section in enumerate(sections) if index not in keep and _is_example(section)}
    short = [_short_rows(section) for section in sections]
    trimmed = [_two_entries(section) if index in example else section for index, section in enumerate(short)]
    with_example = [section for index, section in enumerate(trimmed) if index in keep or index in example]
    kept = [section for index, section in enumerate(sections) if index in keep]
    candidates = [(short, True), (trimmed, True), (with_example, True), (kept, False),
                  ([_short_rows(section) for section in kept], False)]
    for candidate, keeps_example in candidates:
        if not candidate or candidate == sections or (keeps_example and not example):
            continue
        compacted = {**component, "semantic_content": {**semantic, "sections": candidate}}
        if fits(compacted):
            return compacted, keeps_example
    return None


# --- structured fallback ---------------------------------------------------------------------------------------
def _clean(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split())


def _placeholder(line: str) -> str | None:
    match = _PLACEHOLDER_RE.match(line)
    if match is None:
        return None
    return _TRAILING_ELLIPSIS_RE.sub("", match.group(1)).strip() or None


def _upper_share(line: str) -> float:
    letters = [char for char in line if char.isalpha()]
    return sum(char.isupper() for char in letters) / len(letters) if letters else 0.0


def _is_label(line: str) -> bool:
    """A form field label: short, upper case, no sentence punctuation ("MINDSET CẦN BỎ")."""

    return (len(line.split()) <= _LABEL_MAX_WORDS and not line.endswith(("?", ".", "!", ":"))
            and _upper_share(line) >= _UPPER_SHARE and _placeholder(line) is None)


def _is_heading(line: str) -> bool:
    """A heading line of the form ("XÁC NHẬN CỦA NGƯỜI ĐỨNG ĐẦU", "Biểu Mẫu CEO Change Mindset Canvas")."""

    words = line.split()
    if len(words) > _HEADING_MAX_WORDS or line.endswith(("?", ".", "!", ":", '"', "”")):
        return False
    capitalised = sum(word[:1].isupper() for word in words if word[:1].isalpha())
    return _upper_share(line) >= _UPPER_SHARE or capitalised >= _TITLE_CASE_SHARE * len(words)


def _display_label(label: str) -> str:
    """"MINDSET CẦN BỎ" -> "Mindset cần bỏ", "YẾU TỐ AN TOÀN" -> "Yếu tố an toàn"; abbreviations and codes ("AI",
    "BIC", "KPI", "ERA5.0") keep their capitals wherever they stand (``acronyms.keeps_capitals``: a short ASCII word
    such as "AN" is a Vietnamese syllable, not an abbreviation)."""

    if _upper_share(label) < _UPPER_SHARE:
        return label
    text = " ".join(word if keeps_capitals(word) else word.lower() for word in label.split())
    return text[:1].upper() + text[1:]


def _table_row(line: str) -> tuple[str, str, str] | None:
    """A form line of a source table: "Row 2: 01 Hành Động Đo Được Trong 90 Ngày; ......" -> (field, "", "").

    Only a row with one filled cell (the field; the rest blank or a dot leader) is a field to fill in; a row
    with several filled cells ("Row 2: Cấp 1 | Ảnh hưởng thấp | …") is reference content, not a form.
    """

    match = TABLE_ROW_RE.match(line)
    if match is None:
        return None
    cells = [cell for cell in (_FILL_LEADER_RE.sub("", cell).strip()
                               for cell in _CELL_SPLIT_RE.split(match.group(2))) if cell]
    return (cells[0], "", "") if len(cells) == 1 else None


def _form_rows(lines: list[str]) -> tuple[list[tuple[str, str, str]], set[int]]:
    """(field, guiding question, what to write) per form cell and the line indexes they use.

    A run of k labels followed by k questions (and optionally k placeholders) is k cells in source order (the
    PDF prints a row of three cells as three labels, three questions, three placeholders); "Row N: field; …"
    lines are cells too.
    """

    rows: list[tuple[str, str, str]] = []
    used: set[int] = set()
    index = 0
    while index < len(lines):
        table = _table_row(lines[index])
        if table is not None:
            rows.append(table)
            used.add(index)
            index += 1
            continue
        labels = 0
        while index + labels < len(lines) and labels < _MAX_GROUP and _is_label(lines[index + labels]):
            labels += 1
        questions = 0
        while (questions < labels and index + labels + questions < len(lines)
               and lines[index + labels + questions].endswith("?")):
            questions += 1
        if not labels or questions != labels:
            index += max(1, labels)
            continue
        start = index + 2 * labels
        hints = [_placeholder(lines[start + offset]) if start + offset < len(lines) else None
                 for offset in range(labels)]
        complete = all(hint is not None for hint in hints)
        rows.extend((_display_label(lines[index + offset]), lines[index + labels + offset],
                     str(hints[offset]) if complete else "") for offset in range(labels))
        end = start + (labels if complete else 0)
        used.update(range(index, end))
        index = end
    return rows, used


def worksheet_fallback_html(title: str, fact_texts: Sequence[str], task: str | None, locale: str) -> str | None:
    """A structured worksheet from the form in ``fact_texts``, or ``None`` when they hold no form."""

    text = _TEXT["vi" if locale == "vi" else "en"]
    lines = [_clean(value) for value in fact_texts if _clean(value)]
    rows, used = _form_rows(lines)
    if len(rows) < _MIN_ROWS:
        return None
    first = min(used)
    # Lines before the form introduce it (its headings are the slot title already); lines after it close it.
    before = [line for index, line in enumerate(lines[:first]) if not _is_heading(line)]
    after = [line for index, line in enumerate(lines) if index > first and index not in used]
    hints = any(hint for _field, _question, hint in rows)
    questions = any(question for _field, question, _hint in rows)
    columns = [text["field"], *([text["question"]] if questions else []), *([text["hint"]] if hints else [])]
    if len(columns) == 1:
        columns.append(text["hint"])

    def cells(field: str, question: str, hint: str) -> list[str]:
        values = [field, *([question] if questions else []), *([hint] if hints else [])]
        return values if len(values) > 1 else [field, text["blank"]]

    parts = [f"<h2>{html.escape(title)}</h2>", f"<h3>{text['task']}</h3>"]
    if task:
        parts.append(f"<p>{html.escape(task)}</p>")
    parts.append(f"<p>{text['how']}</p>")
    parts.extend(f"<p>{html.escape(line)}</p>" for line in before)
    parts.append(f"<h3>{text['template']}</h3>")
    parts.append("<table><thead><tr>" + "".join(f"<th>{html.escape(cell)}</th>" for cell in columns)
                 + "</tr></thead><tbody>" + "".join(
                     "<tr>" + "".join(f"<td>{html.escape(cell or text['blank'])}</td>" for cell in cells(*row))
                     + "</tr>" for row in rows) + "</tbody></table>")
    if after:
        lead = after[0] if _is_heading(after[0]) else None
        parts.append(f"<h3>{html.escape(_display_label(lead) if lead else text['after'])}</h3>")
        parts.extend(f"<p>{html.escape(line)}</p>" for line in (after[1:] if lead else after))
    checks = ([f"{field}: {hint}" for field, _question, hint in rows if hint]
              or [f"{text['filled']}: {field}" for field, _question, _hint in rows])
    parts.append(f"<h3>{text['check']}</h3>")
    parts.append("<ul>" + "".join(f"<li>{html.escape(item)}</li>" for item in checks) + "</ul>")
    return "".join(parts)


# Run 5496155c (owner decision 2026-10-22): learners read "Tự kiểm tra" / "Self-check", never "(tiêu chí gợi ý)";
# that the AI suggested the criteria is said in the author note only. The label is removed wherever the AI still wrote
# it: after the heading (in brackets or after a dash or colon), as the whole heading, before an item.
_SELF_CHECK: Final = {"vi": "Tự kiểm tra", "en": "Self-check"}
_LABEL: Final = r"(?:tiêu chí gợi ý|suggested criteria|suggested criterion)"
_DASHES: Final = "-" + "".join(map(chr, (0x2013, 0x2014)))
_HEADING_LABEL_RE: Final = re.compile(rf"\s*(?:[(\[]\s*{_LABEL}\s*[)\]]|[{_DASHES}:]\s*{_LABEL})\s*$", re.IGNORECASE)
_WHOLE_LABEL_RE: Final = re.compile(rf"^\s*{_LABEL}\s*$", re.IGNORECASE)
_ITEM_LABEL_RE: Final = re.compile(rf"^\s*{_LABEL}\s*[:{_DASHES}]\s*(?=\S)", re.IGNORECASE)


def plain_self_check(semantic: dict[str, Any], locale: str) -> dict[str, Any] | None:
    """``semantic`` with its self-check section (the last section holding a bullets list) free of the suggested-criteria
    label, or ``None`` when it has none: a heading "Tự kiểm tra bảng của bạn (tiêu chí gợi ý)" becomes "Tự kiểm tra
    bảng của bạn", a heading that is only the label becomes "Tự kiểm tra", an item "Tiêu chí gợi ý: Chọn …" becomes
    "Chọn …". Only that fixed label goes; every other word stays."""

    sections = semantic.get("sections")
    if not isinstance(sections, list):
        return None
    positions = [index for index, section in enumerate(sections) if isinstance(section, dict)
                 and any(block.get("kind") == "bullets" for block in _blocks(section))]
    if not positions:
        return None
    section = sections[positions[-1]]
    heading = unicodedata.normalize("NFC", str(section.get("heading") or ""))
    plain = _HEADING_LABEL_RE.sub("", heading).strip()
    if _WHOLE_LABEL_RE.match(plain) or (heading.strip() and not plain):
        plain = _SELF_CHECK["vi" if locale == "vi" else "en"]
    blocks = []
    for block in section.get("blocks") or []:
        items = block.get("items") if isinstance(block, dict) else None
        blocks.append({**block, "items": [_ITEM_LABEL_RE.sub("", unicodedata.normalize("NFC", item))
                                          if isinstance(item, str) else item for item in items]}
                      if isinstance(block, dict) and block.get("kind") == "bullets" and isinstance(items, list)
                      else block)
    if plain == heading and blocks == section.get("blocks"):
        return None
    updated = [*sections[:positions[-1]], {**section, "heading": plain, "blocks": blocks},
               *sections[positions[-1] + 1:]]
    return {**semantic, "sections": updated}


__all__ = ["COMPACTED_CODE", "compact_worksheet", "plain_self_check", "worksheet_fallback_html"]
