"""Verbatim html of a kept unit and its exact-text check (QLT-6 ``preserve_source``).

The renderer never rewrites, merges, drops or reorders a word. Each source line is one block, in source order:

* a heading line (chapter/lesson/section marker, "1.2" numbering, ALL CAPS) is a section heading: ``<h2>`` when it
  is the unit's own title line, ``<h3>`` otherwise;
* consecutive table rows (``Row N: a | b | c``) are one ``<table>``, one ``<tr>`` per row and one cell per value, the
  first row as the header when the run starts at ``Row 1``; a run longer than the course html limit continues in a
  new table, a row wider than it is shown as a line of its cells;
* any other line is a ``<p>``. List lines keep their own marker or number as text ("• ...", "1. ...", "a) ..."):
  wrapping them in ``<ul>``/``<ol>`` would add a second bullet or renumber them in the lesson view.

``preserve_text_finding`` is the exact-text check: the visible words of the html must equal the visible words of the
unit's lines, in order (``app.idm.preserve_lines``). The learner-content purity rules (page and source mentions, file
names, repeated blocks, ...) never change a kept text: ``purity_report`` only names what they would have flagged so
the author can decide.
"""

from __future__ import annotations

import html as html_lib
from collections.abc import Sequence
from typing import Final

from app.idm.node_acceptance import html_quality_code, instructional_fold
from app.idm.policy import IDM_PRESERVE_TABLE_MAX_COLUMNS, IDM_PRESERVE_TABLE_MAX_ROWS
from app.idm.preserve_lines import (
    TABLE_ROW_RE,
    expected_tokens,
    html_tokens,
    is_render_heading,
    normalize_line,
    table_cells,
)
from app.learner_content_purity import sanitize_source_fact_for_learner

PRESERVE_TEXT_ADDED: Final = "PRESERVE_TEXT_ADDED"
PRESERVE_TEXT_MISSING: Final = "PRESERVE_TEXT_MISSING"
PRESERVE_TEXT_CHANGED: Final = "PRESERVE_TEXT_CHANGED"
# Report codes (``IdmUnitQualityV1.deterministic_codes``): a purity rule would have changed or rejected kept text.
KEPT_SOURCE_MARKER: Final = "IDM_PRESERVE_KEPT_SOURCE_MARKER"
KEPT_PREFIX: Final = "IDM_PRESERVE_KEPT_"
_MIN_TABLE_CELLS: Final = 2


def _escape(text: str) -> str:
    return html_lib.escape(text, quote=True)


def _row_number(text: str) -> int:
    match = TABLE_ROW_RE.fullmatch(text)
    return int(match.group(1)) if match is not None else 0


def _table(rows: Sequence[str]) -> list[str]:
    """One or more ``<table>`` for a run of table rows; a row wider than the html limit is a line of its cells."""

    parts: list[str] = []
    grid: list[list[str]] = []
    header = _row_number(rows[0]) == 1

    def flush() -> None:
        nonlocal header
        if not grid:
            return
        width = max(len(row) for row in grid)
        padded = [row + [""] * (width - len(row)) for row in grid]
        head = ""
        body = padded
        if header:
            head = "<thead><tr>" + "".join(f"<th>{_escape(cell)}</th>" for cell in padded[0]) + "</tr></thead>"
            body = padded[1:]
        rows_html = "".join("<tr>" + "".join(f"<td>{_escape(cell)}</td>" for cell in row) + "</tr>" for row in body)
        parts.append(f"<table>{head}{f'<tbody>{rows_html}</tbody>' if rows_html else ''}</table>")
        grid.clear()
        header = False

    for row in rows:
        cells = table_cells(row) or []
        if len(cells) > IDM_PRESERVE_TABLE_MAX_COLUMNS:
            flush()
            parts.append(f"<p>{_escape(' '.join(cells))}</p>")
            continue
        if len(grid) == IDM_PRESERVE_TABLE_MAX_ROWS:
            flush()
        grid.append(cells or [""])
    flush()
    return parts


def _table_run(lines: Sequence[str], start: int) -> int:
    """End (exclusive) of the table-row run starting at ``start``; ``start`` when it is not a real table."""

    end = start
    while end < len(lines) and table_cells(lines[end]) is not None:
        end += 1
    widest = max((len(table_cells(line) or []) for line in lines[start:end]), default=0)
    return end if widest >= _MIN_TABLE_CELLS else start


def render_preserved_html(unit_title: str, fact_texts: Sequence[str]) -> str:
    """The unit's lines as html, word for word and in order."""

    title = normalize_line(unit_title)
    parts: list[str] = []
    index = 0
    while index < len(fact_texts):
        text = fact_texts[index]
        end = _table_run(fact_texts, index)
        if end > index:
            parts.extend(_table(fact_texts[index:end]))
            index = end
            continue
        cells = table_cells(text)
        visible = " ".join(cells) if cells is not None else normalize_line(text)
        index += 1
        if not visible:
            continue
        if cells is None and index == 1 and visible == title:
            parts.append(f"<h2>{_escape(visible)}</h2>")
        elif cells is None and is_render_heading(text):
            parts.append(f"<h3>{_escape(visible)}</h3>")
        else:
            parts.append(f"<p>{_escape(visible)}</p>")
    return "".join(parts)


def preserve_text_finding(fact_texts: Sequence[str], html: str) -> str | None:
    """``None`` when the html shows exactly the lines' words in order; else what differs (added/missing/changed)."""

    expected = expected_tokens(fact_texts)
    actual = html_tokens(html)
    if actual == expected:
        return None
    if len(actual) > len(expected):
        return PRESERVE_TEXT_ADDED
    if len(actual) < len(expected):
        return PRESERVE_TEXT_MISSING
    return PRESERVE_TEXT_CHANGED


def purity_report(fact_texts: Sequence[str], html: str, exact_identifiers: Sequence[str]) -> list[str]:
    """Report codes for kept text a learner-content purity rule would have changed or rejected (never applied)."""

    codes: list[str] = []
    # Only a removed word or number counts (the sanitizer also trims edge punctuation such as a closing ":").
    if any(instructional_fold(sanitize_source_fact_for_learner(text)) != instructional_fold(text)
           for text in fact_texts):
        codes.append(KEPT_SOURCE_MARKER)
    quality = html_quality_code(html, exact_identifiers)
    if quality is not None:
        codes.append(KEPT_PREFIX + quality)
    return codes
