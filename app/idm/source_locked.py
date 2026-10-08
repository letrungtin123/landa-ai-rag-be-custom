"""Source-locked fallback content for IDM units (QC course 234653, defects D8, D11, D12).

When the W5 writer cannot produce a slot, the unit shows the locked source facts. The
shared renderer and question builder in ``app.instructional_quality`` serve the legacy
pipeline and stay byte-identical (spec §11.6); IDM units get these variants through
``IdmUnitDeps``/the service layer instead:

* HTML: wrapped PDF lines are joined again, a short line becomes a heading only when it
  introduces content (no runs of ``<h3>``), a lone numbered heading is not a one-item
  list, tables stay where the source put them and carry no generic "data table" label,
  markdown emphasis markers are removed.
* Single choice: the question names the term as the source wrote it (never its crossword
  answer form), shows at most four options and explains why the other options are wrong.
* FAQ: explicit source conditions first (the legacy rule), then labelled source definitions
  ("Label: complete sentence"); each answer is the source sentence itself (run c2e5ac41: a
  worksheet unit whose FAQ had no rebuild failed its ``fallback_only`` retry with 422).

Both stay deterministic and never add a claim the source does not make.
"""

from __future__ import annotations

import hashlib
import html
import re
from collections.abc import Iterable
from typing import Any, Final

from app.idm.text import idm_fold
from app.instructional_quality import (
    CALLOUT_RE,
    CHECKLIST_ITEM_RE,
    MIN_FAQ_QUESTION_CHARS,
    ORDERED_STEP_RE,
    TABLE_ROW_RE,
    TERM_DEFINITION_RE,
    build_source_locked_single_choice,
    clean_source_facts,
    faq_answer_is_complete,
    normalize_visible_text,
    ordered_source_steps,
    parse_structured_table_rows,
    source_clarification_signals,
    source_term_definitions,
)

_BULLET_RE: Final = re.compile(r"^[•●▪◦*-]\s+(.+)$")
_INLINE_STEP_RE: Final = re.compile(r"(?<!\w)(?:(?:step|bước|buoc)\s*)?\d{1,3}\s*[:.)-]\s+", re.IGNORECASE)
_TERM_LEAD_RE: Final = re.compile(r"^([^:]{2,100}):\s+(.+)$")
_TERMINAL_RE: Final = re.compile(r"[.!?;:…]$")
# "*Labor-intensive*", "**Race to the bottom**": markdown emphasis copied from the PDF.
_EMPHASIS_RE: Final = re.compile(r"(?<![\w*])\*{1,2}([^*\n]{1,200}?)\*{1,2}(?![\w*])")
# A line that ends on a connector or a comma continues on the next visual line.
_CONNECTOR_END_RE: Final = re.compile(
    r"(?:,|[-\u2013\u2014]|\b(?:va|hoac|cua|trong|cho|voi|de|la|cac|nhung|mot|tu|tai|theo|ve|nhu|khi|neu|"
    r"and|or|of|to|in|for|with|the|a|an|by|on|at|from))$",
)
_CONTINUATION_PUNCTUATION: Final = "([,;"
_SEMANTIC_KEY_RE: Final = re.compile(r"[^\w\d]+")
_STEP_LABEL_RE: Final = re.compile(r"^(?:Step|Bước)\s+\d+\s*:\s*", re.IGNORECASE)

# A label (heading candidate) is short and carries no sentence punctuation.
_LABEL_MAX_CHARS: Final = 90
_LABEL_MAX_WORDS: Final = 12
# Two labels in a row read as one heading ("TRỤ CỘT 01 - Business Excellence"); three or
# more are a list of labels (card titles), never a run of headings.
_HEADING_MAX_LABELS: Final = 2
_HEADING_JOINER: Final = " - "
# A wrapped line is only joined to a previous line that is long enough to be a sentence.
_WRAP_MIN_PREVIOUS_CHARS: Final = 40
_WRAP_TAIL_MAX_WORDS: Final = 4
# Title Case headings ("Chuyển Dịch Tư Duy Sống Còn Của Lãnh Đạo 5.0") are not sentences.
_SENTENCE_LOWER_WORD_SHARE: Final = 0.5
_MIN_INLINE_STEPS: Final = 2
_MIN_TITLE_SUFFIX_CHARS: Final = 8
_NUMBER_PREFIX_RE: Final = re.compile(r"^(?:[0-9]+ )+")
_MIN_TABLE_CELLS: Final = 2
_MAX_TABLE_CELLS: Final = 12
_MAX_TABLE_ROWS: Final = 40
_FILL_LEADER_RE: Final = re.compile(r"[.…_·]{4,}")

_MAX_GROUNDED_CHOICES: Final = 4
_MIN_GROUNDED_CHOICES: Final = 3
_MIN_TABLE_ROWS_FOR_QUESTION: Final = 4
_MAX_TERM_WORDS: Final = 8
_MAX_EXPLANATION_CHARS: Final = 1000
_MAX_QUESTION_CHARS: Final = 500
_MAX_OPTION_CHARS: Final = 500


def _key(value: str) -> str:
    return _SEMANTIC_KEY_RE.sub(" ", idm_fold(value)).strip()


def _inline(value: str) -> str:
    match = _TERM_LEAD_RE.match(value)
    if not match:
        return html.escape(value)
    return f"<strong>{html.escape(match.group(1))}:</strong> {html.escape(match.group(2))}"


def _structural(line: str) -> bool:
    return bool(TABLE_ROW_RE.match(line) or ORDERED_STEP_RE.match(line) or _BULLET_RE.match(line)
                or CHECKLIST_ITEM_RE.match(line) or CALLOUT_RE.match(line))


def _is_label(line: str) -> bool:
    return (len(line) <= _LABEL_MAX_CHARS and len(line.split()) <= _LABEL_MAX_WORDS
            and not _TERMINAL_RE.search(line) and not TABLE_ROW_RE.match(line)
            and not _BULLET_RE.match(line) and not CHECKLIST_ITEM_RE.match(line) and not CALLOUT_RE.match(line))


def _split_inline_steps(value: str) -> list[str]:
    starts = list(_INLINE_STEP_RE.finditer(value))
    if len(starts) < _MIN_INLINE_STEPS:
        return [value]
    result = [value[: starts[0].start()].strip()] if value[: starts[0].start()].strip() else []
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(value)
        if candidate := value[match.start():end].strip():
            result.append(candidate)
    return result


def _sentence_like(value: str) -> bool:
    words = [word for word in value.split() if word[:1].isalpha()]
    return bool(words) and sum(word[:1].islower() for word in words) / len(words) >= _SENTENCE_LOWER_WORD_SHARE


def _continues(previous: str, current: str) -> bool:
    """True when ``current`` is the rest of the sentence ``previous`` started (a PDF wrap)."""

    if (_TERMINAL_RE.search(previous) or TABLE_ROW_RE.match(previous) or _structural(current)
            or not _sentence_like(previous)):
        return False
    if _CONNECTOR_END_RE.search(idm_fold(previous)):
        return True
    if len(previous) < _WRAP_MIN_PREVIOUS_CHARS:
        return False
    if current[:1].islower() or current[:1].isdigit() or current[:1] in _CONTINUATION_PUNCTUATION:
        return True
    # "… Lean, Six" + "Sigma.": a short tail that ends the sentence.
    return len(current.split()) <= _WRAP_TAIL_MAX_WORDS and bool(_TERMINAL_RE.search(current))


def _continues_row(previous: str, current: str) -> bool:
    """A table cell wrapped onto the next line: "… loại bỏ triệt để các lãng phí" + "(Muda) và …"."""

    if not TABLE_ROW_RE.match(previous) or _TERMINAL_RE.search(previous) or _structural(current):
        return False
    return (current[:1].islower() or current[:1] in _CONTINUATION_PUNCTUATION
            or (len(current.split()) <= _WRAP_TAIL_MAX_WORDS and bool(_TERMINAL_RE.search(current))))


def _prepare_lines(title: str, values: Iterable[Any]) -> list[str]:
    facts = clean_source_facts([normalize_visible_text(value) for value in values], preserve_table_numeric=True)
    title_key = _key(title)
    lines: list[str] = []
    repeats: list[str] = []
    for fact in facts:
        text = normalize_visible_text(_EMPHASIS_RE.sub(r"\1", fact))
        key = _key(text)
        # The unit title is already the <h2>; "30 Ngày: SEE DIFFERENT" under "Ngày: SEE DIFFERENT" too.
        if not key:
            continue
        # Only a number may precede the title ("30 Ngày: SEE DIFFERENT" for the block "Ngày: SEE
        # DIFFERENT"); "WHY CHANGE? • BỐI CẢNH MỚI" is not a repeat of "Bối cảnh mới".
        if key == title_key or (len(title_key) >= _MIN_TITLE_SUFFIX_CHARS and _is_label(text)
                                and _NUMBER_PREFIX_RE.sub("", key) == title_key):
            repeats.append(text)
            continue
        for part in _split_inline_steps(text):
            if lines and (_continues(lines[-1], part) or _continues_row(lines[-1], part)):
                lines[-1] = f"{lines[-1]} {part}"
            else:
                lines.append(part)
    # A unit whose only fact repeats its title still shows that fact (the shared validator
    # requires visible source text beyond the title).
    return lines or repeats


def _row_cells(row: str) -> list[str]:
    match = TABLE_ROW_RE.match(row)
    if match is None:
        return []
    return [part.strip().replace("¦", "|") for part in re.split(r"\s+\|\s+", match.group(2)) if part.strip()]


def _table(rows: list[str]) -> str:
    """Rows of one source table, in place; a one-cell row (a form line) is a paragraph."""

    parts: list[str] = []
    grid: list[list[str]] = []

    def flush() -> None:
        if not grid:
            return
        width = max(len(row) for row in grid)
        padded = [row + [""] * (width - len(row)) for row in grid[:_MAX_TABLE_ROWS]]
        head = "<thead><tr>" + "".join(f"<th>{html.escape(cell)}</th>" for cell in padded[0]) + "</tr></thead>"
        body = "".join("<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
                       for row in padded[1:])
        parts.append(f"<table>{head}{f'<tbody>{body}</tbody>' if body else ''}</table>")
        grid.clear()

    for row in rows:
        cells = _row_cells(row)
        if len(cells) >= _MIN_TABLE_CELLS:
            grid.append(cells[:_MAX_TABLE_CELLS])
            continue
        flush()
        # "01 Hành Động Đo Được Trong 90 Ngày;......": a fill-in line, shown without its dot leader.
        text = _FILL_LEADER_RE.sub("", " ".join(cells)).strip()
        if text:
            parts.append(f"<p>{_inline(text)}</p>")
    flush()
    return "".join(parts)


def _list_run(lines: list[str], index: int, keep_single_ordered: bool) -> tuple[str, int] | None:
    """Render the bullet/checklist/numbered run starting at ``index``; ``None`` when it is not one."""

    line = lines[index]
    patterns = ((ORDERED_STEP_RE, "ol", 2), (CHECKLIST_ITEM_RE, "ul", 1), (_BULLET_RE, "ul", 1))
    for pattern, tag, group in patterns:
        if not pattern.match(line):
            continue
        items = []
        end = index
        while end < len(lines) and (match := pattern.match(lines[end])):
            items.append(match.group(group).strip())
            end += 1
        if tag == "ol" and len(items) == 1 and not keep_single_ordered:
            return None  # "1. Thế Giới Mới: …" alone is a numbered heading, not a one-item list
        return f"<{tag}>" + "".join(f"<li>{_inline(item)}</li>" for item in items) + f"</{tag}>", end
    return None


def render_idm_source_locked_html(
    title: str,
    values: Iterable[Any],
    *,
    locale: str,
    required_artifacts: Iterable[dict[str, Any]] = (),
) -> str:
    """Render locked source facts as a readable, source-ordered HTML fragment."""

    del locale  # the IDM variant adds no generated labels, so it needs no locale-specific copy
    required = {str(item.get("type") or "").strip().casefold() for item in required_artifacts
                if isinstance(item, dict)}
    # The shared validator counts <li> inside <ol>; when the plan requires an ordered list,
    # lone numbered lines keep the legacy list form so the required artifact is preserved.
    keep_single_ordered = "ordered_list" in required
    lines = _prepare_lines(title, values)
    parts = [f"<h2>{html.escape(title)}</h2>"]
    index = 0
    while index < len(lines):
        line = lines[index]
        if TABLE_ROW_RE.match(line):
            end = index
            while end < len(lines) and TABLE_ROW_RE.match(lines[end]):
                end += 1
            parts.append(_table(lines[index:end]))
            index = end
            continue
        rendered = _list_run(lines, index, keep_single_ordered)
        if rendered is not None:
            parts.append(rendered[0])
            index = rendered[1]
            continue
        if CALLOUT_RE.match(line):
            parts.append(f"<blockquote>{_inline(line)}</blockquote>")
            index += 1
            continue
        if _is_label(line):
            end = index
            # A numbered run ("1. …", "2. …") after the labels is a list of its own.
            while end < len(lines) and _is_label(lines[end]) and (
                    end == index or _list_run(lines, end, keep_single_ordered) is None):
                end += 1
            labels = lines[index:end]
            has_body = end < len(lines)
            if has_body and len(labels) <= _HEADING_MAX_LABELS:
                parts.append(f"<h3>{html.escape(_HEADING_JOINER.join(labels))}</h3>")
            elif len(labels) == 1:
                parts.append(f"<p>{_inline(labels[0])}</p>")
            elif has_body:
                # Card titles or a chapter label above a section heading: the labels are a list,
                # the last one introduces the content that follows.
                parts.append("<ul>" + "".join(f"<li>{_inline(label)}</li>" for label in labels[:-1]) + "</ul>")
                parts.append(f"<h3>{html.escape(labels[-1])}</h3>")
            else:
                parts.append("<ul>" + "".join(f"<li>{_inline(label)}</li>" for label in labels) + "</ul>")
            index = end
            continue
        parts.append(f"<p>{_inline(line)}</p>")
        index += 1
    return "".join(parts)


# --- single choice -----------------------------------------------------------------------------
def _display_terms(values: Iterable[str]) -> dict[str, str]:
    """Crossword answer form -> the term as the source wrote it (first occurrence)."""

    result: dict[str, str] = {}
    for value in values:
        if TABLE_ROW_RE.match(value) or ORDERED_STEP_RE.match(value):
            continue
        match = TERM_DEFINITION_RE.match(value)
        if not match:
            continue
        term = normalize_visible_text(match.group(1)).strip(" .;,-\u2013\u2014")
        if term and len(term.split()) <= _MAX_TERM_WORDS:
            answer = re.sub(r"[^A-Za-z0-9]", "", idm_fold(term)).upper()
            result.setdefault(answer, term)
    return result


def _explain_others(lead: str, others: list[str], locale: str) -> str:
    """Feedback that teaches: the criterion, then why each other option is wrong."""

    lead = lead.rstrip()
    if not _TERMINAL_RE.search(lead):
        lead += "."
    if others:
        joined = "; ".join(others)
        lead += (f" The other options describe something else: {joined}." if locale == "en"
                 else f" Các lựa chọn còn lại mô tả nội dung khác: {joined}.")
    return lead[:_MAX_EXPLANATION_CHARS]


def _choices(options: list[str], correct: str, seed: str) -> list[dict[str, Any]]:
    bounded = [normalize_visible_text(option)[:_MAX_OPTION_CHARS] for option in options]
    # Reproducible, and the correct option is never systematically first.
    offset = 1 + int.from_bytes(hashlib.sha256(seed.encode("utf-8")).digest()[:2], "big") % (len(bounded) - 1)
    ordered = bounded[offset:] + bounded[:offset]
    correct_key = _key(correct)
    return [{"text": option, "correct": _key(option) == correct_key} for option in ordered]


def _distinct(options: list[str]) -> bool:
    return len({_key(option) for option in options}) >= _MIN_GROUNDED_CHOICES


def idm_source_grounded_single_choice(title: str, values: Iterable[Any], *, locale: str) -> dict[str, Any] | None:
    """One verifiable single-choice question built only from source structure (or ``None``)."""

    materialized = list(values)
    authored = build_source_locked_single_choice(title, materialized, locale=locale)
    if authored is not None:
        return authored
    facts = clean_source_facts(materialized, preserve_table_numeric=True)
    en = locale == "en"
    definitions = source_term_definitions(facts)
    if len(definitions) >= _MIN_GROUNDED_CHOICES:
        display = _display_terms(facts)
        named = [(display.get(answer, answer), definition)
                 for answer, definition in definitions[:_MAX_GROUNDED_CHOICES]]
        term, correct = named[0]
        options = [definition for _term, definition in named]
        if _distinct(options):
            others = [(f"that is “{other}”" if en else f"đó là “{other}”") for other, _text in named[1:]]
            return {
                "problem_type": "multiple_choice",
                "question": (f"Which description correctly matches the term {term}?" if en
                             else f"Mô tả nào phù hợp nhất với thuật ngữ {term}?")[:_MAX_QUESTION_CHARS],
                "choices": _choices(options, correct, f"definition:{title}:{term}"),
                "explanation": _explain_others(f"{term} is defined as: {correct}" if en
                                               else f"{term} được xác định là: {correct}", others, locale),
            }
    rows = parse_structured_table_rows(facts)
    if len(rows) >= _MIN_TABLE_ROWS_FOR_QUESTION and len(rows[0]) >= _MIN_TABLE_CELLS:
        header, body = rows[0], [row for row in rows[1:] if len(row) >= _MIN_TABLE_CELLS]
        shown = body[:_MAX_GROUNDED_CHOICES]
        if len(shown) >= _MIN_GROUNDED_CHOICES and _distinct([row[1] for row in shown]):
            key, correct = shown[0][0], shown[0][1]
            return {
                "problem_type": "multiple_choice",
                "question": (f"According to {header[0]}, which {header[1]} corresponds to {key}?" if en
                             else f"Theo {header[0]}, {header[1]} nào tương ứng với {key}?")[:_MAX_QUESTION_CHARS],
                "choices": _choices([row[1] for row in shown], correct, f"table:{title}:{key}"),
                "explanation": _explain_others(f"{key} corresponds to {correct}." if en
                                               else f"{key} tương ứng với {correct}.",
                                               [f"{row[1]} ↔ {row[0]}" for row in shown[1:]], locale),
            }
    steps = ordered_source_steps(facts, locale=locale)
    if len(steps) >= _MIN_GROUNDED_CHOICES:
        actions = [_STEP_LABEL_RE.sub("", step) for step in steps][:_MAX_GROUNDED_CHOICES]
        if _distinct(actions):
            return {
                "problem_type": "multiple_choice",
                "question": ("Which action comes first in the procedure?" if en
                             else "Hoạt động nào được thực hiện đầu tiên trong quy trình?"),
                "choices": _choices(actions, actions[0], f"procedure:{title}"),
                "explanation": _explain_others(
                    f"The procedure begins with: {actions[0]}" if en else f"Quy trình bắt đầu bằng: {actions[0]}",
                    [(f"“{action}” is a later step" if en else f"“{action}” là bước sau") for action in actions[1:]],
                    locale),
            }
    return None


_FAQ_MAX_ITEMS: Final = 4
_FAQ_MAX_ANSWER_CHARS: Final = 600
_FAQ_MIN_ITEMS: Final = 2
_FAQ_LABEL_MIN_CHARS: Final = 6
_FAQ_LABEL_MAX_CHARS: Final = 100
_FAQ_LABEL_TRIM: Final = " .;,-*" + chr(0x2013) + chr(0x2014) + chr(0x2022)
# Learner text of an interactive component never carries angle brackets or C0 controls (Node workspace schema).
_FAQ_UNSAFE_RE: Final = re.compile("[<>" + "".join(chr(code) for code in (*range(0x00, 0x09), 0x0B, 0x0C,
                                                                           *range(0x0E, 0x20))) + "]")
_LEAD_LABEL_RE: Final = re.compile(r"[:.!?]")


def _faq_question(label: str, title: str, index: int, locale: str) -> str:
    en = locale == "en"
    if _FAQ_LABEL_MIN_CHARS <= len(label) <= _FAQ_LABEL_MAX_CHARS:
        return f"What should learners understand about {label}?" if en else f"Người học cần hiểu gì về {label}?"
    if index == 0:
        return f"What should learners remember about '{title}'?" if en else f"Người học cần ghi nhớ gì về '{title}'?"
    return (f"Which point {index + 1} needs attention in '{title}'?" if en
            else f"Điểm thứ {index + 1} cần lưu ý trong '{title}' là gì?")


def idm_source_faq(title: str, values: Iterable[Any], *, locale: str) -> list[dict[str, str]] | None:
    """FAQ items of a source-locked IDM ``la_faq`` slot (or ``None`` below two items).

    Each answer is one complete source sentence: an explicit condition/exception first (the
    legacy rule), then the definition of a labelled fact ("Quy luật chi phí: Đầu tư 1 đồng ...").
    The question only names the label, so no claim is added; title-case headings are skipped.
    """

    facts = clean_source_facts(list(values), preserve_table_numeric=True)
    pairs = [(_LEAD_LABEL_RE.split(answer, maxsplit=1)[0].strip(), answer)
             for answer in source_clarification_signals(facts)]
    for value in facts:
        if TABLE_ROW_RE.match(value) or ORDERED_STEP_RE.match(value):
            continue
        match = TERM_DEFINITION_RE.match(value)
        if not match:
            continue
        label = normalize_visible_text(match.group(1)).strip(_FAQ_LABEL_TRIM)
        definition = normalize_visible_text(match.group(2)).strip()
        if (label and len(label.split()) <= _LABEL_MAX_WORDS and faq_answer_is_complete(definition)
                and _sentence_like(definition)):
            pairs.append((label, definition))
    items: list[dict[str, str]] = []
    answers: set[str] = set()
    questions: set[str] = set()
    for label, text in pairs:
        answer = text[:_FAQ_MAX_ANSWER_CHARS].strip()
        question = _faq_question(label, title, len(items), locale)
        if (not _key(answer) or _key(answer) in answers or _key(question) in questions
                or len(question) < MIN_FAQ_QUESTION_CHARS or not faq_answer_is_complete(answer)
                or _FAQ_UNSAFE_RE.search(question + answer)):
            continue
        answers.add(_key(answer))
        questions.add(_key(question))
        items.append({"question": question, "answer": answer})
        if len(items) == _FAQ_MAX_ITEMS:
            break
    return items if len(items) >= _FAQ_MIN_ITEMS else None

