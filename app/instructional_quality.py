from __future__ import annotations

import html
import re
import unicodedata
from typing import Any, Iterable

from app.learner_content_purity import (
    is_provenance_only_source_line,
    sanitize_source_fact_for_learner,
)


TABLE_MARKER_RE = re.compile(r"^\[TABLE\]$", re.IGNORECASE)
TABLE_ROW_RE = re.compile(r"^Row\s+(\d+)\s*:\s*(.+)$", re.IGNORECASE)
ORDERED_STEP_RE = re.compile(
    r"^(?:(?:step|bước|buoc)\s*)?(\d{1,3})\s*[:.)-]\s*(.+)$",
    re.IGNORECASE,
)
TERM_DEFINITION_RE = re.compile(r"^([^:]{2,80})\s*:\s*(.{20,})$")

_URL_ONLY_RE = re.compile(r"^(?:https?://|www\.)\S+$", re.IGNORECASE)
_EMAIL_ONLY_RE = re.compile(r"^[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}$")
_PHONE_ONLY_RE = re.compile(r"^(?:\+?\d[\d\s().-]{7,}\d)$")
_REPEATED_GLYPH_RE = re.compile(r"(.)\1{5,}", re.IGNORECASE)
_CONTACT_LABEL_RE = re.compile(
    r"^(?:tel|telephone|phone|hotline|email|e-mail|website|web|địa\s*chỉ|dia\s*chi|address)\s*[:.-]",
    re.IGNORECASE,
)

_NON_INSTRUCTIONAL_EXACT = {
    "thank you",
    "thanks",
    "cảm ơn",
    "cam on",
    "xin cảm ơn",
    "xin cam on",
    "the end",
}

_GENERIC_REVIEW_PHRASES = (
    "rà soát ý trong tài liệu nguồn",
    "theo đúng thứ tự xuất hiện trong tài liệu nguồn",
    "được giữ nguyên để người dùng rà soát theo nguồn",
    "review the source point",
    "displayed order in the source",
    "retained for source review",
)


def normalize_visible_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _fold(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value.casefold())
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def is_non_instructional_source_line(value: Any, *, preserve_table_numeric: bool = False) -> bool:
    """Conservatively reject extraction debris before it becomes canonical evidence.

    This only removes standalone presentation/contact artifacts and malformed
    fragments. Domain statements containing a URL, phone number, or address
    remain available; emergency contact instructions must not be discarded.
    """

    text = normalize_visible_text(value).strip("\ufeff")
    if not text or TABLE_MARKER_RE.fullmatch(text):
        return True
    if is_provenance_only_source_line(text):
        return True
    if TABLE_ROW_RE.match(text):
        return False
    folded = _fold(text).strip(" .,:;!?-_–—")
    if folded in _NON_INSTRUCTIONAL_EXACT:
        return True
    if _URL_ONLY_RE.fullmatch(text) or _EMAIL_ONLY_RE.fullmatch(text):
        return True
    if _CONTACT_LABEL_RE.match(text):
        return True
    if _PHONE_ONLY_RE.fullmatch(text) and not preserve_table_numeric:
        return True
    if _REPEATED_GLYPH_RE.search(text):
        return True
    letters = sum(char.isalpha() for char in text)
    digits = sum(char.isdigit() for char in text)
    if letters == 0:
        return not (preserve_table_numeric and digits > 0)
    if letters < 3 and len(text) > 8:
        return True
    return False


def clean_source_facts(values: Iterable[Any], *, preserve_table_numeric: bool = False) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = sanitize_source_fact_for_learner(normalize_visible_text(value))
        if is_non_instructional_source_line(text, preserve_table_numeric=preserve_table_numeric):
            continue
        key = _fold(text)
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def parse_structured_table_rows(values: Iterable[Any]) -> list[list[str]]:
    rows: list[tuple[int, list[str]]] = []
    for value in values:
        match = TABLE_ROW_RE.match(normalize_visible_text(value))
        if not match:
            continue
        cells = [cell.strip().replace("¦", "|") for cell in re.split(r"\s+\|\s+", match.group(2))]
        cells = [cell for cell in cells if cell]
        if len(cells) >= 2:
            rows.append((int(match.group(1)), cells[:12]))
    rows.sort(key=lambda item: item[0])
    return [cells for _index, cells in rows[:40]]


def _semantic_line_key(value: str) -> str:
    return re.sub(r"[^\w\d]+", " ", _fold(value)).strip()


def _inline(value: str) -> str:
    match = re.match(r"^([^:]{2,100}):\s+(.+)$", value)
    if not match:
        return html.escape(value)
    return f"<strong>{html.escape(match.group(1))}:</strong> {html.escape(match.group(2))}"


def render_source_locked_html(
    title: str,
    values: Iterable[Any],
    *,
    locale: str,
    required_artifacts: Iterable[dict[str, Any]] = (),
) -> str:
    """Render a bounded semantic document while preserving source order.

    The source-locked path is a recovery path, so it must not rewrite facts or
    regroup unrelated lines merely to make the page look structured.  Keeping
    headings, their following bullets, and inline numbered procedures in source
    order also prevents a long numbered source line from collapsing to a title-
    only HTML component.
    """

    raw = [normalize_visible_text(value) for value in values]
    facts = clean_source_facts(raw, preserve_table_numeric=True)
    rows = parse_structured_table_rows(facts)
    table_facts = {normalize_visible_text(value) for value in facts if TABLE_ROW_RE.match(value)}
    prose = [value for value in facts if value not in table_facts]
    title_key = _semantic_line_key(title)
    prose = [value for value in prose if _semantic_line_key(value) != title_key]

    def split_inline_steps(value: str) -> list[str]:
        starts = list(re.finditer(
            r"(?<!\w)(?:(?:step|bước|buoc)\s*)?\d{1,3}\s*[:.)-]\s+",
            value,
            flags=re.IGNORECASE,
        ))
        if len(starts) < 2:
            return [value]
        result: list[str] = []
        prefix = value[:starts[0].start()].strip()
        if prefix:
            result.append(prefix)
        for index, match in enumerate(starts):
            end = starts[index + 1].start() if index + 1 < len(starts) else len(value)
            candidate = value[match.start():end].strip()
            if candidate:
                result.append(candidate)
        return result

    lines = [part for value in prose for part in split_inline_steps(value)]
    parts = [f"<h3>{html.escape(title)}</h3>"]
    index = 0
    while index < len(lines):
        line = lines[index]
        ordered_match = ORDERED_STEP_RE.match(line)
        bullet_match = re.match(r"^[•●▪◦*-]\s+(.+)$", line)
        if ordered_match or bullet_match:
            tag = "ol" if ordered_match else "ul"
            items: list[str] = []
            while index < len(lines):
                match = (
                    ORDERED_STEP_RE.match(lines[index])
                    if tag == "ol"
                    else re.match(r"^[•●▪◦*-]\s+(.+)$", lines[index])
                )
                if not match:
                    break
                items.append(f"<li>{_inline(match.group(2 if tag == 'ol' else 1).strip())}</li>")
                index += 1
            parts.append(f"<{tag}>{''.join(items)}</{tag}>")
            continue

        word_count = len(line.split())
        is_heading = (
            len(line) <= 140
            and 2 <= word_count <= 18
            and not re.search(r"[.!?;:]$", line)
        )
        if is_heading:
            parts.append(f"<h3>{html.escape(line)}</h3>")
            index += 1
            body_lines: list[str] = []
            while index < len(lines):
                candidate = lines[index]
                if (
                    len(candidate) > 180
                    or not re.search(r"[.!?;:]$", candidate)
                    or ORDERED_STEP_RE.match(candidate)
                    or re.match(r"^[•●▪◦*-]\s+", candidate)
                ):
                    break
                body_lines.append(candidate)
                index += 1
            if len(body_lines) >= 2:
                parts.append("<ul>" + "".join(f"<li>{_inline(value)}</li>" for value in body_lines) + "</ul>")
            elif body_lines:
                parts.append(f"<p>{_inline(body_lines[0])}</p>")
            continue

        parts.append(f"<p>{_inline(line)}</p>")
        index += 1

    required_types = {
        str(item.get("type") or "").strip().casefold()
        for item in required_artifacts
        if isinstance(item, dict)
    }
    if rows:
        width = max(len(row) for row in rows)
        normalized_rows = [row + [""] * (width - len(row)) for row in rows]
        header = normalized_rows[0]
        body = normalized_rows[1:]
        parts.append(f"<h3>{html.escape('Bảng dữ liệu' if locale != 'en' else 'Structured data')}</h3>")
        parts.append("<table><thead><tr>" + "".join(f"<th>{html.escape(cell)}</th>" for cell in header) + "</tr></thead>")
        if body:
            parts.append("<tbody>" + "".join(
                "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
                for row in body
            ) + "</tbody>")
        parts.append("</table>")
    elif "table" in required_types or "comparison" in required_types:
        # The quality gate will reject this fallback. Keeping the HTML valid
        # makes the exact missing artifact observable instead of fabricating it.
        pass

    return "".join(parts)


def ordered_source_steps(values: Iterable[Any], *, locale: str) -> list[str]:
    discovered: list[tuple[int, str]] = []
    for value in clean_source_facts(values, preserve_table_numeric=True):
        candidates = re.split(r"(?=(?<!\w)(?:step|bước|buoc)?\s*\d{1,3}\s*[:.)-]\s+)", value, flags=re.IGNORECASE)
        for candidate in candidates:
            match = ORDERED_STEP_RE.match(candidate.strip())
            if match and len(match.group(2).strip()) >= 10:
                discovered.append((int(match.group(1)), match.group(2).strip()))
    unique: dict[int, str] = {}
    for order, value in discovered:
        unique.setdefault(order, value)
    if len(unique) < 3:
        return []
    label = "Step" if locale == "en" else "Bước"
    return [f"{label} {index}: {value}"[:500] for index, value in sorted(unique.items())[:10]]


def source_term_definitions(values: Iterable[Any]) -> list[tuple[str, str]]:
    """Extract explicit term-definition pairs without turning steps into vocabulary.

    Crossword fallback must never pick arbitrary frequent words or manufacture
    generic SOURCE/LESSON entries. Only a concise term followed by a substantive
    source definition is eligible.
    """

    result: list[tuple[str, str]] = []
    seen_answers: set[str] = set()
    for value in clean_source_facts(values, preserve_table_numeric=True):
        if TABLE_ROW_RE.match(value) or ORDERED_STEP_RE.match(value):
            continue
        match = TERM_DEFINITION_RE.match(value)
        if not match:
            continue
        term = normalize_visible_text(match.group(1)).strip(" .;,-–—")
        definition = normalize_visible_text(match.group(2)).strip()
        if not term or len(term.split()) > 8 or re.match(r"^(?:step|bước|buoc)\b", _fold(term)):
            continue
        folded = unicodedata.normalize("NFD", term.replace("đ", "d").replace("Đ", "D"))
        answer = re.sub(
            r"[^A-Za-z0-9]",
            "",
            "".join(char for char in folded if unicodedata.category(char) != "Mn"),
        ).upper()
        if not 2 <= len(answer) <= 24 or answer in seen_answers:
            continue
        seen_answers.add(answer)
        result.append((answer, definition[:500]))
        if len(result) == 10:
            break
    return result


def source_clarification_signals(values: Iterable[Any]) -> list[str]:
    """Return only explicit conditions/exceptions/cautions that justify FAQ.

    Two unrelated substantive statements are not an FAQ opportunity.  This
    conservative signal is shared by architecture admission and deterministic
    fallback so optional components cannot depend on model creativity alone.
    """

    result: list[str] = []
    seen: set[str] = set()
    for value in clean_source_facts(values, preserve_table_numeric=True):
        folded = _fold(value).strip().lstrip("•*- ")
        if not re.match(
            r"^(?:neu|khi|tru khi|chi khi|khong duoc|luu y|canh bao|ngoai le|"
            r"if|when|unless|only if|must not|do not|warning|caution|exception)\b",
            folded,
        ):
            continue
        key = _semantic_line_key(folded)
        if key and key not in seen:
            seen.add(key)
            result.append(value)
        if len(result) == 8:
            break
    return result


def build_source_locked_single_choice(
    title: str,
    values: Iterable[Any],
    *,
    locale: str,
) -> dict[str, Any] | None:
    """Recover only an explicit source-authored single-choice assessment.

    A prose fact is not enough evidence for plausible wrong answers.  The old
    recovery path paired one true source sentence with three generic false
    statements and therefore turned a missing assessment into a learner-facing
    quiz that had never existed in the source.  Deterministic recovery is now
    deliberately narrower: question, choices, answer key and explanation must
    all be present in the locked evidence.
    """

    del title, locale
    raw_lines: list[str] = []
    marker = re.compile(
        r"(?=(?:question|câu\s*hỏi|cau\s*hoi|correct\s*answer|answer|đáp\s*án|dap\s*an|"
        r"explanation|rationale|giải\s*thích|giai\s*thich)\s*(?:\d+)?\s*[:.)-]|"
        r"[A-F]\s*[.)-]\s+)",
        re.IGNORECASE,
    )
    for value in values:
        for physical_line in re.split(r"[\r\n]+", str(value or "")):
            for candidate in marker.split(physical_line):
                normalized = normalize_visible_text(candidate)
                if normalized:
                    raw_lines.append(normalized)

    question_re = re.compile(
        r"^(?:question|câu\s*hỏi|cau\s*hoi)\s*(?:\d+)?\s*[:.)-]\s*(.{12,500})$",
        re.IGNORECASE,
    )
    option_re = re.compile(r"^([A-F])\s*[.)-]\s*(.{3,500})$", re.IGNORECASE)
    answer_re = re.compile(
        r"^(?:correct\s*answer|answer|đáp\s*án|dap\s*an)\s*[:.)-]\s*([A-F])\s*$",
        re.IGNORECASE,
    )
    explanation_re = re.compile(
        r"^(?:explanation|rationale|giải\s*thích|giai\s*thich)\s*[:.)-]\s*(.{20,1000})$",
        re.IGNORECASE,
    )
    question: str | None = None
    answer_label: str | None = None
    explanation: str | None = None
    options: list[tuple[str, str]] = []
    for line in raw_lines:
        if question is None and (match := question_re.match(line)):
            question = match.group(1).strip()
            continue
        if (match := option_re.match(line)):
            options.append((match.group(1).upper(), match.group(2).strip()))
            continue
        if answer_label is None and (match := answer_re.match(line)):
            answer_label = match.group(1).upper()
            continue
        if explanation is None and (match := explanation_re.match(line)):
            explanation = match.group(1).strip()

    labels = [label for label, _text in options]
    choice_texts = [text for _label, text in options]
    if (question is None or answer_label is None or explanation is None
            or not 3 <= len(options) <= 6 or len(set(labels)) != len(labels)
            or labels != [chr(ord("A") + index) for index in range(len(labels))]
            or answer_label not in labels
            or len({_semantic_line_key(text) for text in choice_texts}) != len(choice_texts)):
        return None
    return {
        "problem_type": "multiple_choice",
        "question": question,
        "choices": [
            {"text": text, "correct": label == answer_label}
            for label, text in options
        ],
        "explanation": explanation,
    }


def contains_generic_review_language(value: Any) -> bool:
    folded = _fold(normalize_visible_text(value))
    return any(_fold(marker) in folded for marker in _GENERIC_REVIEW_PHRASES)
