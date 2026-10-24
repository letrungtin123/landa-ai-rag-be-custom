"""Source-locked fallback content for IDM units (QC course 234653, defects D8, D11, D12).

When the W5 writer cannot produce a slot, the unit shows the locked source facts. The
shared renderer and question builder in ``app.content.instructional_quality`` serve the legacy
pipeline and stay byte-identical (spec §11.6); IDM units get these variants through
``IdmUnitDeps``/the service layer instead:

* HTML: wrapped PDF lines are joined again, a short line becomes a heading only when it
  introduces content (no runs of ``<h3>``), a lone numbered heading is not a one-item
  list, tables stay where the source put them and carry no generic "data table" label,
  markdown emphasis markers are removed. Run 5496155c (c1.l3.u3: "Về Ý nghĩa / Xã hội
  (Social/Meaningful" as a heading, an orphan "Job): *", stray "|" and "*", a heading in
  capitals): a line is joined to the next ones until its brackets close, a cell of a table
  row the PDF broke over several lines ends at its "|" and loses its list marker, a table
  of a header row alone is a list, a stray "*" goes, and a heading in capitals is written
  in sentence case (codes and abbreviations keep theirs). Run 2f8781d6: ``app.idm.fallback_lines``.
* Single choice: the question names the term as the source wrote it (never its crossword
  answer form), shows at most four options and explains why the other options are wrong.
* FAQ: explicit source conditions first (the legacy rule), then labelled source definitions
  ("Label: complete sentence"); each answer is the source sentence itself (run c2e5ac41: a
  worksheet unit whose FAQ had no rebuild failed its ``fallback_only`` retry with 422), said once.

Both stay deterministic and never add a claim the source does not make.
"""

from __future__ import annotations

import hashlib
import html
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Final

from app.content.instructional_quality import (
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
from app.idm import fallback_lines
from app.idm.acronyms import keeps_capitals
from app.idm.fallback_lines import ORPHAN_LABEL_END_RE, OrphanLabels, faq_topic, prefix_repeated_labels, source_facts
from app.idm.text import idm_fold
from app.idm.titles import TitleCaser

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
# A quotation mark too (run 5496155c: "… Một trải nghiệm" + "\"Wow\" ở tuyến trước …").
_CONTINUATION_PUNCTUATION: Final = "([,;\"" + chr(0x201C)
_OPENING_CONTINUATION: Final = "(\"" + chr(0x201C)
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
# A label of fewer letters ("I", a lone letter of a slide graphic) is no heading and no text.
_MIN_HEADING_LETTERS: Final = 2
_NUMBER_PREFIX_RE: Final = re.compile(r"^(?:[0-9]+ )+")
_MIN_TABLE_CELLS: Final = 2
_MAX_TABLE_CELLS: Final = 12
_MAX_TABLE_ROWS: Final = 40
_FILL_LEADER_RE: Final = re.compile(r"[.…_·]{4,}")
# Run 5496155c: what the PDF reader left of a slide table. A "*" bullet after a label ("Job): * Sàng lọc") or alone; the
# cell separator inside a line that is no table row ("… chung. | - Về Chức năng (Functional Pain): * Bị") and the list
# marker that opens the next cell ("- Về", "-Về").
# Review of dfb6db2 (item 4): only such a bullet star (after a label or a sentence end) or one ending a line goes; a
# star between operands ("Doanh thu trung bình * Số năm", "(Tác động + Mức khẩn) * Khả thi") is the formula's.
_STAR_RE: Final = re.compile(r"(?<=[:;.])[ \t]*\*[ \t]+|[ \t]+\*[ \t]*$")
_CELL_SEPARATOR_RE: Final = re.compile(r"[ \t]+\|[ \t]+")
# A "-" before a digit is a sign ("| -5% | +3% |"), never a list marker (review of dfb6db2, item 4).
_CELL_MARKER_RE: Final = re.compile(r"^(?:[*•][ \t]*|-[ \t]*(?![\d\s]))(?=\S)")
STATEMENT_END_RE: Final = re.compile(r"[.!…]$")
# A classroom exercise or discussion prompt of a slide ("BÀI TẬP 10 phút", "Câu hỏi thảo luận: …"), accent-folded: no
# statement of the lesson and no FAQ answer (run 5496155c).
EXERCISE_RE: Final = re.compile(r"^\W*(?:bai tap|cau hoi|thao luan|thuc hanh|exercise|question|discussion|homework)\b")
_SENTENCE_INSIDE_RE: Final = re.compile(r"[.!?…][ \t]+\S")
_CLOSING_MARKS: Final = "\"')" + chr(0x201D)
# A word that introduces the quotation on the next line ("… cần tránh câu" + "“Việc này …”").
_QUOTE_LEAD_RE: Final = re.compile(r"\b(?:cau|loi|rang|noi|hoi|chu|khau hieu|thong diep|cum tu|says?|said|asks?"
                                   r"|asked|phrase|words?|slogan)$")
_WORD_MARKS: Final = ".,;:!?()[]\"'" + chr(0x201C) + chr(0x201D)
# A line whose brackets stay open is joined to at most this many following lines, this long in all.
_BRACKET_JOIN_LINES: Final = 4
_BRACKET_JOIN_CHARS: Final = 400
# A heading in capitals (this share of its letters) is written in sentence case.
_UPPER_LABEL_SHARE: Final = 0.6

_MAX_GROUNDED_CHOICES: Final = 4
_MIN_GROUNDED_CHOICES: Final = 3
_MIN_TABLE_ROWS_FOR_QUESTION: Final = 4
_MAX_TERM_WORDS: Final = 8
_MAX_EXPLANATION_CHARS: Final = 1000
_MAX_QUESTION_CHARS: Final = 500
_MAX_OPTION_CHARS: Final = 500
_MAX_REASON_CHARS: Final = 160
_OPTION_LETTERS: Final = "ABCDEF"
_LINE_FEED: Final = "\n"


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


def _whole_quote(line: str) -> bool:
    """A line that is one quotation ("“Tôi cần …”"), never a quoted word that starts a longer line (review 4)."""

    text = line.strip()
    closing = {chr(0x201C): chr(0x201D), '"': '"'}.get(text[:1])
    return closing is not None and len(text) > 1 and text.find(closing, 1) == len(text) - 1


def _is_label(line: str) -> bool:
    """A short line without closing punctuation (a closing mark may follow it), never a whole quotation (item 10)."""

    return (len(core := fallback_lines.label_core(line)) <= _LABEL_MAX_CHARS and len(core.split()) <= _LABEL_MAX_WORDS
            and not _whole_quote(line) and not _TERMINAL_RE.search(line.rstrip(_CLOSING_MARKS))
            and not TABLE_ROW_RE.match(line)
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
    """True when ``current`` is the rest of the sentence ``previous`` started (a PDF wrap). Run 5496155c: a labelled
    line ("Về Chức năng (Functional Pain): Bị") is read by its text after the label, and a line that opens with a
    bracket or a quotation mark continues an unfinished sentence of any length ("… cổ chai" + "(bottleneck) …")."""

    labelled = _TERM_LEAD_RE.match(previous)
    if (_TERMINAL_RE.search(previous) or TABLE_ROW_RE.match(previous) or _structural(current)
            or not (labelled or _sentence_like(previous))):
        return False
    folded = idm_fold(previous)
    if _CONNECTOR_END_RE.search(folded):
        return True
    # Item 10 (narrowed by review 4): a whole-line quotation after a list item or a heading stands apart; a quotation
    # the sentence introduces ("… cần tránh câu" + "“…”") or a labelled line ("…(Functional Pain): Bị") goes on.
    if (_whole_quote(current) and not _QUOTE_LEAD_RE.search(folded)
            and (_structural(previous) or (_is_label(previous) and not labelled))):
        return False
    if current[:1] in _OPENING_CONTINUATION:
        return True
    if len(previous) < _WRAP_MIN_PREVIOUS_CHARS or (labelled and not _sentence_like(previous)):
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


def _open_brackets(line: str) -> int:
    return line.count("(") - line.count(")")


def _closes(previous: str, current: str, joined: int) -> bool:
    """``current`` closes the bracket ``previous`` left open ("Về Ý nghĩa / Xã hội (Social/Meaningful" + "Job): …",
    "(Chữ" + "I" + "Implementation trong G-STIC): …"), within the join bounds; never a new table row."""

    return (_open_brackets(previous) > 0 and joined < _BRACKET_JOIN_LINES and not TABLE_ROW_RE.match(current)
            and not _whole_quote(current) and len(previous) + len(current) < _BRACKET_JOIN_CHARS)


def _row_goes_on(previous: str, current: str) -> bool:
    """``current`` holds the next cells of the table row ``previous`` the PDF broke ("Row 1: Tầng góc nhìn | JOB TO
    BE" + "DONE(…) | PAINS(…) | GAINS(…)")."""

    return (bool(TABLE_ROW_RE.match(previous)) and not _TERMINAL_RE.search(previous)
            and not TABLE_ROW_RE.match(current) and _CELL_SEPARATOR_RE.search(current) is not None)


def _trails_off(text: str) -> bool:
    """A text that ends a sentence and then stops in the middle of the next one ("… hay CSKH. Nếu hệthống"): the PDF cut
    it; never an option or an answer (run 5496155c)."""

    return bool(_SENTENCE_INSIDE_RE.search(text)) and not STATEMENT_END_RE.search(text.rstrip(_CLOSING_MARKS))


def _wrapped_word(line: str) -> bool:
    """A few words without closing punctuation: the end of a cell the PDF moved to its own line ("HỘI")."""

    return len(line.split()) <= _WRAP_TAIL_MAX_WORDS and not _TERMINAL_RE.search(line) and not _structural(line)


def _cells(text: str) -> list[str]:
    """A line that is no table row, cut at the cell separators the PDF reader left in it, each cell without its list
    marker; a table row whole."""

    if TABLE_ROW_RE.match(text) or not _CELL_SEPARATOR_RE.search(text):
        return [text]
    return [cell for part in _CELL_SEPARATOR_RE.split(text) if (cell := _CELL_MARKER_RE.sub("", part.strip()))]


def _prepare_lines(title: str, values: Iterable[Any], *, labels: bool = False, ordered: bool = False) -> list[str]:
    facts = source_facts(values, _is_label, keep_labels=labels)
    title_key = _key(title)
    lines: list[str] = []
    repeats: list[str] = []
    joined, orphans = 0, OrphanLabels(fallback_lines.written_labels(facts))
    for fact in facts:
        text = normalize_visible_text(_STAR_RE.sub(" ", _EMPHASIS_RE.sub(r"\1", fact)))
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
        if lines and _row_goes_on(lines[-1], text):
            lines[-1] = f"{lines[-1]} {text}"
            joined += 1
            continue
        # "Row 4: 3. Ý NGHĨA / XÃ" + "HỘI" + "(Định vị - Tầm nhìn) | …": the short line between is the row's too.
        if (len(lines) > 1 and _wrapped_word(lines[-1]) and _row_goes_on(f"{lines[-2]} {lines[-1]}", text)):
            lines[-2:] = [f"{lines[-2]} {lines[-1]} {text}"]
            continue
        # A table row keeps its cells: "Row 2: 1. CHỨC NĂNG" is no step "1.".
        for raw in (cell for value in _cells(text)
                    for cell in ([value] if TABLE_ROW_RE.match(value) else _split_inline_steps(value))):
            orphan = _open_brackets(raw) < 0 and not (lines and _closes(lines[-1], raw, joined))
            part = (orphans.restore(raw) if labels else ORPHAN_LABEL_END_RE.sub("", raw)) if orphan else raw
            if lines and (_continues(lines[-1], part) or _continues_row(lines[-1], part)
                          or _closes(lines[-1], part, joined)):
                orphans.note(lines[-1], part)
                lines[-1] = f"{lines[-1]} {part}"
                joined += 1
            else:
                lines.append(part)
                joined = 0
    # A unit whose only fact repeats its title still shows that fact (the shared validator
    # requires visible source text beyond the title).
    return prefix_repeated_labels(lines or repeats, _is_label, ordered=ordered) if labels else lines or repeats


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
        _flush_or_list(grid, parts, flush)
        # "01 Hành Động Đo Được Trong 90 Ngày;......": a fill-in line, shown without its dot leader.
        text = _FILL_LEADER_RE.sub("", " ".join(cells)).strip()
        if text:
            parts.append(f"<p>{_inline(text)}</p>")
    _flush_or_list(grid, parts, flush)
    return "".join(parts)


def _flush_or_list(grid: list[list[str]], parts: list[str], flush: Callable[[], None]) -> None:
    """Close the table in progress; a table of its header row alone (the PDF broke its other rows into lines, run
    5496155c) is the list of its cells instead."""

    if len(grid) == 1:
        cells = [cell for cell in grid[0] if cell.strip()]
        grid.clear()
        if cells:
            parts.append("<ul>" + "".join(f"<li>{_inline(cell)}</li>" for cell in cells) + "</ul>")
        return
    flush()


@dataclass(frozen=True)
class _Names:
    """Known place and personal names, runs of capitalised Vietnamese words inside a sentence (``TitleCaser``) and
    single brand-like words a sentence capitalises ("công ty Vinamilk"); never "từng Điểm chạm" (review 4)."""

    caser: TitleCaser
    words: frozenset[str]

    @classmethod
    def of(cls, lines: Sequence[str]) -> _Names:
        words: set[str] = set()
        lower = {core for line in lines for token in line.split() if (core := token.strip(_WORD_MARKS)).islower()}
        for line in lines:
            if _is_label(line) or not _sentence_like(line):
                continue
            tokens = line.split()
            for position, token in enumerate(tokens[1:], 1):
                core = token.strip(_WORD_MARKS)
                if (len(core) > 1 and core.isascii() and core[0].isupper() and core[1:].islower()
                        and not tokens[position - 1].endswith((".", ":", "!", "?"))):
                    words.add(core.lower())
        return cls(TitleCaser.from_texts(lines), frozenset(words - lower))

    def positions(self, lowered: Sequence[str]) -> set[int]:
        return self.caser.name_positions(lowered) | {index for index, word in enumerate(lowered) if word in self.words}


def _capitalised(word: str) -> str:
    letter = next((index for index, char in enumerate(word) if char.isalpha()), None)
    return word.lower() if letter is None else word[:letter] + word[letter].upper() + word[letter + 1:].lower()


def _sentence_case(label: str, names: _Names | None = None) -> str:
    """A heading in capitals in sentence case: "1.3 HỒ SƠ KHÁCH HÀNG B2B (VD2: NGƯỜI RA QUYẾT ĐỊNH)" -> "1.3 Hồ sơ
    khách hàng B2B (VD2: người ra quyết định)"; codes and abbreviations keep their capitals (``keeps_capitals``), and
    the ``names`` of the unit their capital (item 11): "CHI NHÁNH VINAMILK TẠI ĐÀ NẴNG" -> "Chi nhánh Vinamilk tại Đà
    Nẵng"."""

    letters = [char for char in label if char.isalpha()]
    if not letters or sum(char.isupper() for char in letters) / len(letters) < _UPPER_LABEL_SHARE:
        return label
    raw = label.split()
    kept = (names or _Names(TitleCaser(), frozenset())).positions([word.strip(_WORD_MARKS).lower() for word in raw])
    words = [word if keeps_capitals(word) else (_capitalised(word) if position in kept else word.lower())
             for position, word in enumerate(raw)]
    first = next((index for index, word in enumerate(words) if any(char.isalpha() for char in word)), None)
    if first is not None and not keeps_capitals(words[first]):
        word = words[first]
        letter = next(index for index, char in enumerate(word) if char.isalpha())
        words[first] = word[:letter] + word[letter].upper() + word[letter + 1:]
    return " ".join(words)


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
    lines = _prepare_lines(title, values, labels=True, ordered=keep_single_ordered)
    names = _Names.of(lines)
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
            # A label whose brackets never close is a cut line, and a lone letter no heading: both stay text.
            labels = [_sentence_case(label, names) for label in lines[index:end]]
            if any(_open_brackets(label) or sum(char.isalpha() for char in label) < _MIN_HEADING_LETTERS
                   for label in labels):
                # Review of dfb6db2 (item 4): a figure ("85%", "+32", "2024", "4P") is never dropped as a lone letter.
                parts.extend(f"<p>{_inline(label)}</p>" for label in labels
                             if sum(char.isalpha() for char in label) >= _MIN_HEADING_LETTERS
                             or any(char.isdigit() for char in label))
                index = end
                continue
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


def _lower_first(label: str) -> str:
    """A header label inside a sentence: its first letter in lower case unless it starts an acronym ("Dấu hiệu" ->
    "dấu hiệu", "KPI" stays)."""

    return label[:1].lower() + label[1:] if label[1:2].islower() else label


def _feedback(principle: str, choices: list[dict[str, Any]], reasons: dict[str, str], locale: str) -> str:
    """Feedback in the shape the writer is asked for (QC feedback 2026-10-21, A1; owner decision 2026-10-22 for the
    fallback): the principle line, then one line per option in letter order, "A - Đúng vì …" / "A - Sai vì …" (English
    "Correct/Incorrect because …"). Every reason names only what the source structure the question was built from says
    about that option (``reasons``: option key -> reason); the principle is cut to what the lines leave."""

    en = locale == "en"
    lines = []
    for index, choice in enumerate(choices):
        verdict = (("Correct because" if en else "Đúng vì") if choice["correct"]
                   else ("Incorrect because" if en else "Sai vì"))
        reason = reasons.get(_key(str(choice["text"])), "")
        lines.append(f"{_OPTION_LETTERS[index]} - {verdict} {_reason_text(reason)}")
    body = _LINE_FEED.join(lines)
    lead = " ".join(principle.split())
    if not _TERMINAL_RE.search(lead):
        lead += "."
    room = _MAX_EXPLANATION_CHARS - len(body) - 1
    if len(lead) > room:
        lead = lead[:max(room - 1, 0)].rsplit(" ", 1)[0].rstrip(" ,;:") + "…"
    return lead + _LINE_FEED + body


def _reason_text(reason: str) -> str:
    """One reason within ``_MAX_REASON_CHARS``, cut after a whole word with an ellipsis (review m8: "lần đ.")."""

    reason = " ".join(reason.split())
    if len(reason) > _MAX_REASON_CHARS:
        reason = reason[:_MAX_REASON_CHARS].rsplit(" ", 1)[0].rstrip(" ,;:") + "…"
    return reason if _TERMINAL_RE.search(reason) else reason + "."


def _choices(options: list[str], correct: str, seed: str) -> list[dict[str, Any]]:
    bounded = [normalize_visible_text(option)[:_MAX_OPTION_CHARS] for option in options]
    # Reproducible, and the correct option is never systematically first.
    offset = 1 + int.from_bytes(hashlib.sha256(seed.encode("utf-8")).digest()[:2], "big") % (len(bounded) - 1)
    ordered = bounded[offset:] + bounded[:offset]
    correct_key = _key(correct)
    return [{"text": option, "correct": _key(option) == correct_key} for option in ordered]


def _distinct(options: list[str]) -> bool:
    return len({_key(option) for option in options}) >= _MIN_GROUNDED_CHOICES


def _option_key(option: str) -> str:
    """The key of an option as ``_choices`` shows it (normalised and bounded), for its reason."""

    return _key(normalize_visible_text(option)[:_MAX_OPTION_CHARS])


def _authored_feedback(question: dict[str, Any], locale: str) -> dict[str, Any]:
    """A question the source itself writes (question, options, key and explanation): its explanation becomes the
    principle line and every option gets its line; the source decides, so the reason is the key the source gives."""

    choices = [choice for choice in question.get("choices") or [] if isinstance(choice, dict)]
    letter = next((_OPTION_LETTERS[index] for index, choice in enumerate(choices) if choice.get("correct")), "")
    en = locale == "en"
    reasons = {_key(str(choice["text"])): (("this is the answer the source gives" if en
                                            else "đây là đáp án tài liệu nêu") if choice.get("correct")
                                           else (f"the source gives {letter} as the answer" if en
                                                 else f"tài liệu nêu đáp án là {letter}"))
               for choice in choices}
    return {**question, "explanation": _feedback(str(question.get("explanation") or ""), choices, reasons, locale)}


def idm_source_grounded_single_choice(title: str, values: Iterable[Any], *, locale: str) -> dict[str, Any] | None:
    """One verifiable single-choice question built only from source structure (or ``None``). Its explanation has the
    shape the writer is asked for (``_feedback``): the principle line, then one line per option with its reason."""

    materialized = list(values)
    authored = build_source_locked_single_choice(title, materialized, locale=locale)
    if authored is not None:
        return _authored_feedback(authored, locale)
    facts = clean_source_facts(materialized, preserve_table_numeric=True)
    # Definitions and table rows are read from the lines the PDF wrapped joined again (run 5496155c: an option ended
    # "… Nếu hệthống", a table of four rows read as one).
    joined = _prepare_lines("", facts)
    en = locale == "en"
    definitions = [(term, text) for term, text in source_term_definitions(joined) if not _trails_off(text)]
    if len(definitions) >= _MIN_GROUNDED_CHOICES:
        display = _display_terms(joined)
        named = [(display.get(answer, answer), definition)
                 for answer, definition in definitions[:_MAX_GROUNDED_CHOICES]]
        term, correct = named[0]
        options = [definition for _term, definition in named]
        if _distinct(options):
            choices = _choices(options, correct, f"definition:{title}:{term}")
            reasons = {_option_key(text): (f"this is the definition of {term}" if en
                                           else f"đây là định nghĩa của {term}") if other == term
                       else (f"this defines {other}, not {term}" if en
                             else f"đây là định nghĩa của {other}, không phải của {term}")
                       for other, text in reversed(named)}
            return {
                "problem_type": "multiple_choice",
                "question": (f"Which description correctly matches the term {term}?" if en
                             else f"Mô tả nào phù hợp nhất với thuật ngữ {term}?")[:_MAX_QUESTION_CHARS],
                "choices": choices,
                "explanation": _feedback(f"{term} is defined as: {correct}" if en
                                         else f"{term} được xác định là: {correct}", choices, reasons, locale),
            }
    rows = parse_structured_table_rows(joined)
    if len(rows) >= _MIN_TABLE_ROWS_FOR_QUESTION and len(rows[0]) >= _MIN_TABLE_CELLS:
        header, body = rows[0], [row for row in rows[1:] if len(row) >= _MIN_TABLE_CELLS]
        shown = body[:_MAX_GROUNDED_CHOICES]
        if len(shown) >= _MIN_GROUNDED_CHOICES and _distinct([row[1] for row in shown]):
            key, correct = shown[0][0], shown[0][1]
            label = _lower_first(header[1])
            choices = _choices([row[1] for row in shown], correct, f"table:{title}:{key}")
            reasons = {_option_key(row[1]): (f"this is the {label} of {key}" if en
                                             else f"đây là {label} của {key}") if row[0] == key
                       else (f"this is the {label} of {row[0]}, not of {key}" if en
                             else f"đây là {label} của {row[0]}, không phải của {key}")
                       for row in reversed(shown)}
            return {
                "problem_type": "multiple_choice",
                "question": (f"According to {header[0]}, which {header[1]} corresponds to {key}?" if en
                             else f"Theo {header[0]}, {header[1]} nào tương ứng với {key}?")[:_MAX_QUESTION_CHARS],
                "choices": choices,
                "explanation": _feedback(f"{key} corresponds to {correct}." if en
                                         else f"{key} tương ứng với {correct}.", choices, reasons, locale),
            }
    steps = ordered_source_steps(facts, locale=locale)
    if len(steps) >= _MIN_GROUNDED_CHOICES:
        actions = [_STEP_LABEL_RE.sub("", step) for step in steps][:_MAX_GROUNDED_CHOICES]
        if _distinct(actions):
            choices = _choices(actions, actions[0], f"procedure:{title}")
            reasons = {_option_key(action): (("this is the first step of the procedure" if en
                                              else "đây là bước đầu tiên của quy trình") if number == 1
                                             else (f"this is step {number} of the procedure, not the first one" if en
                                                   else f"đây là bước thứ {number} của quy trình, không phải bước đầu "
                                                        "tiên"))
                       for number, action in reversed(list(enumerate(actions, start=1)))}
            return {
                "problem_type": "multiple_choice",
                "question": ("Which action comes first in the procedure?" if en
                             else "Hoạt động nào được thực hiện đầu tiên trong quy trình?"),
                "choices": choices,
                "explanation": _feedback(
                    f"The procedure begins with: {actions[0]}" if en else f"Quy trình bắt đầu bằng: {actions[0]}",
                    choices, reasons, locale),
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
# A source-locked FAQ kept with one sentence twice because the facts hold no second item (see idm_source_faq).
FAQ_FALLBACK_REPEATED_CODE: Final = "IDM_W5_FAQ_FALLBACK_REPEATED"


def _faq_question(label: str, title: str, index: int, locale: str) -> str:
    en, title = locale == "en", faq_topic(title)  # run 2f8781d6: never "… về 'Hỏi đáp: …'"
    if _FAQ_LABEL_MIN_CHARS <= len(label) <= _FAQ_LABEL_MAX_CHARS:
        return f"What should learners understand about {label}?" if en else f"Người học cần hiểu gì về {label}?"
    if index == 0:
        return f"What should learners remember about '{title}'?" if en else f"Người học cần ghi nhớ gì về '{title}'?"
    return (f"Which point {index + 1} needs attention in '{title}'?" if en
            else f"Điểm thứ {index + 1} cần lưu ý trong '{title}' là gì?")


def _faq_candidates(facts: list[str]) -> list[tuple[str, str, str]]:
    """(source fact, label the question names, answer text) in the order items are chosen: explicit
    conditions first, whole sentence (the legacy rule), then the definitions of labelled facts. Run 5496155c: an
    answer is never a question or a sentence the PDF cut, and a classroom exercise of the slides ("Thực hành: …") comes
    last."""

    candidates = [(answer, _LEAD_LABEL_RE.split(answer, maxsplit=1)[0].strip(), answer)
                  for answer in source_clarification_signals(facts)]
    exercises: list[tuple[str, str, str]] = []
    for value in facts:
        if TABLE_ROW_RE.match(value) or ORDERED_STEP_RE.match(value):
            continue
        match = TERM_DEFINITION_RE.match(value)
        if not match:
            continue
        label = normalize_visible_text(match.group(1)).strip(_FAQ_LABEL_TRIM)
        definition = normalize_visible_text(match.group(2)).strip()
        if (label and len(label.split()) <= _LABEL_MAX_WORDS and faq_answer_is_complete(definition)
                and _sentence_like(definition) and not _trails_off(definition) and "?" not in definition):
            (exercises if EXERCISE_RE.match(idm_fold(label)) else candidates).append((value, label, definition))
    return [*candidates, *exercises]


def _repeats(answer: str, others: Iterable[str]) -> bool:
    """``answer`` says what one of ``others`` says, or the reverse ("Lưu ý: X" and "X")."""

    key = f" {_key(answer)} "
    return any(key in f" {_key(other)} " or f" {_key(other)} " in key for other in others)


def _faq_item(items: list[dict[str, str]], label: str, text: str, title: str, locale: str, *,
              distinct: bool) -> dict[str, str] | None:
    """The next item for ``text`` under the source-locked rules, or ``None``; ``distinct``: it must not say
    what an item already says."""

    answer = text[:_FAQ_MAX_ANSWER_CHARS].strip()
    question = _faq_question(label, title, len(items), locale)
    if (not _key(answer) or _key(answer) in {_key(item["answer"]) for item in items}
            or _key(question) in {_key(item["question"]) for item in items}
            or len(question) < MIN_FAQ_QUESTION_CHARS or not faq_answer_is_complete(answer)
            or _FAQ_UNSAFE_RE.search(question + answer)
            or (distinct and _repeats(answer, (item["answer"] for item in items)))):
        return None
    return {"question": question, "answer": answer}


def _faq_items(candidates: list[tuple[str, str, str]], title: str, locale: str, *,
               distinct: bool) -> list[dict[str, str]]:
    """Up to ``_FAQ_MAX_ITEMS`` items; ``distinct``: one per source fact, none repeating another (without it,
    the rule of commit d4ff677, where "Lưu ý: X" also gave "X" as a second item)."""

    items: list[dict[str, str]] = []
    sources: set[str] = set()
    for source, label, text in candidates:
        if distinct and _key(source) in sources:
            continue
        item = _faq_item(items, label, text, title, locale, distinct=distinct)
        if item is None:
            continue
        sources.add(_key(source))
        items.append(item)
        if len(items) == _FAQ_MAX_ITEMS:
            break
    return items


def faq_answers_repeat(items: Any) -> bool:
    """One answer of an FAQ payload says what another says (the kept repeat of a lone caution)."""

    if not isinstance(items, list):
        return False
    answers = [str(item.get("answer") or "") for item in items if isinstance(item, dict)]
    return any(_repeats(answer, answers[:index]) for index, answer in enumerate(answers))


def idm_source_faq(title: str, values: Iterable[Any], *, locale: str,
                   unit_values: Iterable[Any] = ()) -> list[dict[str, str]] | None:
    """FAQ items of a source-locked IDM ``la_faq`` slot (or ``None`` below two items).

    Each answer is one complete source sentence: an explicit condition/exception first (the
    legacy rule), then the definition of a labelled fact ("Quy luật chi phí: Đầu tư 1 đồng ...").
    The question only names the label, so no claim is added; title-case headings are skipped.
    A source fact gives at most one item and no answer repeats another: a labelled caution
    ("Lưu ý: Không được ...") is both a condition and a labelled fact, and keeps its whole sentence.

    A slot that the rule without that check could build stays buildable (``fallback_only`` must not
    turn into a 422): a missing second item is another complete sentence of the slot, else of the
    unit (``unit_values``), under a "point N" question; with none, the repeated item stays and the
    unit keeps the slot for author review (``FAQ_FALLBACK_REPEATED_CODE``).
    """

    # Run 5496155c: answers come from the wrapped lines joined again, never from a line the PDF cut.
    facts = _prepare_lines("", clean_source_facts(list(values), preserve_table_numeric=True))
    candidates = _faq_candidates(facts)
    items = _faq_items(candidates, title, locale, distinct=True)
    if len(items) >= _FAQ_MIN_ITEMS:
        return items
    repeated = _faq_items(candidates, title, locale, distinct=False)
    if len(repeated) < _FAQ_MIN_ITEMS:
        return None
    for value in (*facts, *_prepare_lines("", clean_source_facts(list(unit_values), preserve_table_numeric=True))):
        if len(items) >= _FAQ_MIN_ITEMS:
            break
        if TABLE_ROW_RE.match(value) or ORDERED_STEP_RE.match(value) or not _sentence_like(value):
            continue
        if (item := _faq_item(items, "", value, title, locale, distinct=True)) is not None:
            items.append(item)
    return items if len(items) >= _FAQ_MIN_ITEMS else repeated
