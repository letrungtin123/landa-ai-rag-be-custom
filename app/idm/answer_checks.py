"""Single-choice checks against what the unit itself shows (QC run ab8d67e1, R2).

The 4-gram copy check and the distinctive-pair check (``qa.copied_options``, ``qa.reworded_key``) need long keys
with many distinctive word pairs. The run showed three more ways a question gives itself away or has no single
answer, each read against the passages of the html slots before the question (a sentence of a paragraph or
quotation, a list item, a table row) and the unit's FAQ answers:

* a key that paraphrases ONE passage in the same word order (c4.l1.u1 restated its blockquote);
* a short key that is the only option the html shows, or a case that copies an example passage holding the key;
* a contested key: the options name labels the unit teaches ("Bậc 01", "Bậc 02" …) and the question's case meets
  another label's criteria at least as well as the key's (c1.l2.u2: "Bậc 01" for a case with ISO and CE while
  the FAQ right after says ISO, CE, FDA only reach "Bậc 02").

QC run fae66524 (QLT-7) added two more:

* a key that restates the unit's own worked example ("Ví dụ minh hoạ" and the blocks it introduces): c2.l3.u1's key
  "Chủ động gửi hướng dẫn cá nhân hóa và dịch vụ tri ân bất ngờ …" after the worksheet example "Cung cấp video hướng
  dẫn cá nhân hóa và gửi quà tri ân bất ngờ" (``restates_example``);
* a case that literally meets a criterion only another label's row states, read from the html table rows and the
  table-row facts ("Row 3: Bậc 02 | Vietnam Quality | … (ISO, CE, FDA)"): c1.l1.u2 keyed "Bậc 01" for a case "đạt
  chứng nhận ISO 9001 và chứng chỉ CE" (``contested_option``, ``fact_table_rows``).

QC run 0a11aa9e (QLT-8) added the keys that join several shown items, which the one-passage measures never see:

* a key that strings together the items of ONE list or table shown before it (c1.l1.u1: the five list heads in one
  sentence, ``restates_list``);
* a key that joins several passages of the worked example (c2.l5.u1: three rows of the worksheet example,
  ``restates_joined_example``).

Run 5496155c (QC 2026-10-22) found the give-away in the stem itself, which the key-only measures never read:

* ``resolved_choices``: an option that is only a listed item's name ("Phương án 2") is measured by that item's text, so
  every measure above reads what the key really says (c3.l1.u2: "Phương án 1" pointed to the worked example's rows);
* ``stem_copies_answer``: a listed item copies the worked example or a table row right above the question and the key
  names that example's answer or that row's label (c1.l2.u2: "Trợ lý Tổng giám đốc … lọc hồ sơ" from the example whose
  answer was "Người gác cổng"; c1.l3.u2: "vinh danh đúng tầm … kiến tạo di sản" from the row of the key's layer);
* ``copies_example_measure``: the key states a number with its unit from the worked example ("2 giờ") that neither the
  case nor any distractor states (c2.l4.u2).

Pure functions on the writer's payload; thresholds and their calibration live in ``app.idm.policy``.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from app.idm.policy import (
    IDM_ANSWER_CONTESTED_MIN_CODES,
    IDM_ANSWER_CONTESTED_MIN_PAIRS,
    IDM_ANSWER_CONTESTED_MIN_WORDS,
    IDM_ANSWER_EXAMPLE_BAG_MIN_MARGIN,
    IDM_ANSWER_EXAMPLE_BAG_MIN_SHARE,
    IDM_ANSWER_EXAMPLE_MAX_DISTRACTOR_SHARE,
    IDM_ANSWER_EXAMPLE_MIN_SHARE,
    IDM_ANSWER_EXAMPLE_WORDS_MIN_KEY_WORDS,
    IDM_ANSWER_EXAMPLE_WORDS_MIN_MARGIN,
    IDM_ANSWER_EXAMPLE_WORDS_MIN_SHARE,
    IDM_ANSWER_JOINED_MIN_ITEMS,
    IDM_ANSWER_JOINED_MIN_KEY_PAIRS,
    IDM_ANSWER_LIST_MIN_MARGIN,
    IDM_ANSWER_LIST_MIN_SHARE,
    IDM_ANSWER_QUOTE_MIN_KEY_WORDS,
    IDM_ANSWER_QUOTE_MIN_MARGIN,
    IDM_ANSWER_QUOTE_MIN_SHARE,
    IDM_ANSWER_SHORT_DISTRACTOR_MAX_SHARE,
    IDM_ANSWER_SHORT_KEY_MAX_WORDS,
    IDM_ANSWER_STEM_COPY_MIN_SHARE,
    IDM_ANSWER_STEM_ITEM_COPY_SHARE,
    IDM_ANSWER_STEM_ITEM_MARGIN,
    IDM_ANSWER_STEM_ITEM_MIN_PAIRS,
    IDM_ANSWER_STEM_ITEM_MIN_WORDS,
    IDM_ANSWER_STEM_ITEM_WORD_SHARE,
    IDM_ANSWER_STEM_PASSAGE_MIN_WORDS,
)
from app.idm.quiz_text import named_items
from app.idm.text import content_pairs, content_words, idm_fold, number_spans
from app.idm.worksheet_input import is_input_section

_WORD_RE: Final = re.compile(r"\w+", re.UNICODE)
# A worked example in the html: a text that opens with an example label, the blocks after it in its section, and every
# block of a section whose heading names an example ("Ví dụ minh họa mẫu hoàn thành").
_EXAMPLE_LABEL_RE: Final = re.compile(
    r"^\s*[\"'(]*\s*(?:ví dụ minh hoạ|ví dụ minh họa|ví dụ|illustrative example|worked example|for example|example)"
    r"\s*[:,]\s*", re.IGNORECASE)
_EXAMPLE_HEADING_WORDS: Final = ("vi du", "example", "mau hoan thanh", "bai mau")
# A table-row fact: "Row 3: Bậc 02 | Vietnam Quality | Đạt chuẩn …" (the PDF reader's row label is optional).
_ROW_LABEL_RE: Final = re.compile(r"^\s*row\s+\d{1,3}\s*:\s*", re.IGNORECASE)
_MIN_ROW_CELLS: Final = 2
# A first cell of at most this many words ("Bậc 01") is a rung or a code: the second cell completes its label.
_SHORT_LABEL_WORDS: Final = 3
# A code a row states: "ISO", "CE", "FDA", "5S", "CR-04" (all capitals, at least two characters).
_CODE_RE: Final = re.compile(r"(?<![\w-])[A-Z][A-Z0-9]{1,9}(?![\w-])")
# Words that negate the rest of their clause in the case ("chưa đạt ISO", "không tối ưu được chi phí ẩn", "without
# ISO"); a clause ends at a comma, semicolon, colon or sentence mark.
_NEGATIONS: Final = frozenset({"khong", "chua", "chang", "thieu", "not", "no", "without", "lacks", "lacking"})
_CLAUSE_RE: Final = re.compile(r"[,;:.!?]")
# "bac 01" -> ("bac", 1): the rungs of one ladder share the stem.
_RUNG_RE: Final = re.compile(r"^(?P<stem>[^\W\d_]+(?:\s+[^\W\d_]+)?)\s+0*(?P<number>\d{1,2})$")
_SENTENCE_RE: Final = re.compile(r"(?<=[.!?;])\s+")
# The head of a taught label: "Bậc 01 - Vietnam Manufacturing" -> "Bậc 01", "Hiện đại hóa: …" -> "Hiện đại hóa".
_HEAD_END_RE: Final = re.compile(r"\s+[-–—|]\s+|\s*:\s*|\s*\(")  # noqa: RUF001 - dashes
# An option that starts with a numbered label ("Cấp 2 vì …", "Bậc 01 - …") applies a taught rule to the case.
_NUMBERED_LABEL_RE: Final = re.compile(r"^\s*[^\W\d_]+(?:\s+[^\W\d_]+)?\s+0?\d{1,2}(?!\d)")
_MIN_HEAD_CHARS: Final = 3
# A label is looked for at the start of an option (its first words), never in its reason.
_HEAD_SEARCH_CHARS: Final = 60
# A key that applies a label opens with it, after at most this many words ("Thuộc Bậc 02 …", "Cấp 2 …").
_LABEL_OPENING_WORDS: Final = 2
_MIN_OPTIONS: Final = 2
_LIST_KINDS: Final = frozenset({"bullets", "steps"})
_TEXT_KINDS: Final = frozenset({"paragraph", "task", "warning"})


@dataclass(frozen=True)
class UnitTeaching:
    """What a question can be read against: html passages before it, taught (label, criteria) pairs, FAQ text; the
    passages of the worked examples shown before it and the (label, criteria) rows of the table-row facts (QLT-7)."""

    passages: tuple[str, ...]
    rows: tuple[tuple[str, str], ...]
    faq_sentences: tuple[str, ...]
    examples: tuple[str, ...] = ()
    fact_rows: tuple[tuple[str, str], ...] = ()
    # QLT-8: the items of each list or table block before the question, block by block ("label value" per table row).
    lists: tuple[tuple[str, ...], ...] = ()


def _sections(component: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    semantic = component.get("semantic_content")
    sections = semantic.get("sections") if isinstance(semantic, dict) else None
    for section in sections if isinstance(sections, list) else []:
        if isinstance(section, dict):
            yield section


def _blocks(component: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    for section in _sections(component):
        for block in section.get("blocks") or []:
            if isinstance(block, dict):
                yield block


def _worksheet(component: Mapping[str, Any]) -> bool:
    """A worksheet html slot (it states a task): only its input data is no taught content (review 4: a theory section
    "Các loại dữ liệu khách hàng" or an example "Ví dụ minh họa với dữ liệu đơn hàng" is taught)."""

    return any(block.get("kind") == "task" for block in _blocks(component))


def _block_texts(block: Mapping[str, Any]) -> list[str]:
    """The learner texts of a block: its sentences, list items or table values (a row's value is the filled entry)."""

    kind = block.get("kind")
    if kind in _TEXT_KINDS:
        return [part for part in _SENTENCE_RE.split(str(block.get("text") or "")) if part.strip()]
    if kind in _LIST_KINDS:
        return [str(item) for item in block.get("items") or []]
    if kind == "table":
        return [str(row.get("value") or "") for row in block.get("rows") or [] if isinstance(row, dict)]
    return []


def example_passages(html_before: Sequence[Mapping[str, Any]]) -> list[str]:
    """The passages of the worked examples the html before a question shows (QLT-7): a text opening with an example
    label ("Ví dụ minh hoạ: …", the label left out), the blocks after it in its section, and every block of a section
    whose heading names an example."""

    passages: list[str] = []
    for component in html_before:
        worksheet = _worksheet(component)
        for section in _sections(component):
            if worksheet and is_input_section(section):
                continue
            inside = any(word in idm_fold(str(section.get("heading") or "")) for word in _EXAMPLE_HEADING_WORDS)
            for block in section.get("blocks") or []:
                if not isinstance(block, dict):
                    continue
                texts = _block_texts(block)
                opening = _EXAMPLE_LABEL_RE.match(texts[0]) if texts and block.get("kind") in _TEXT_KINDS else None
                if opening is not None:
                    texts = [texts[0][opening.end():], *texts[1:]]
                    inside = True
                if inside:
                    passages.extend(text for text in texts if text.strip())
    return passages


def fact_table_rows(texts: Iterable[str]) -> list[tuple[str, str]]:
    """(label, criteria) of the table-row facts ("Row 3: Bậc 02 | Vietnam Quality | Đạt chuẩn …" -> ("Bậc 02 -
    Vietnam Quality", "Đạt chuẩn …")); a header row (every cell in capitals) is left out."""

    rows: list[tuple[str, str]] = []
    for text in texts:
        cells = [cell.strip() for cell in _ROW_LABEL_RE.sub("", text, count=1).split("|") if cell.strip()]
        if len(cells) < _MIN_ROW_CELLS or all(cell == cell.upper() for cell in cells):
            continue
        if len(cells) > _MIN_ROW_CELLS and len(cells[0].split()) <= _SHORT_LABEL_WORDS:
            rows.append((f"{cells[0]} - {cells[1]}", " ".join(cells[2:])))
        else:
            rows.append((cells[0], " ".join(cells[1:])))
    return rows


def unit_teaching(html_before: Sequence[Mapping[str, Any]], faq: Sequence[Mapping[str, Any]],
                  fact_rows: Sequence[tuple[str, str]] = ()) -> UnitTeaching:
    """The passages and taught labels of the html slots before a question, the sentences of the unit's FAQ, the
    worked-example passages and ``fact_rows`` (``fact_table_rows`` of the facts the writer was given).

    Review of dfb6db2 (item 1, r5 c3.l2.u1): a worksheet's input data ("Yêu cầu 1: …") is no taught content: a
    question that practises on the same entries gives nothing away, the worksheet holds no answer for them."""

    passages: list[str] = []
    rows: list[tuple[str, str]] = []
    lists: list[tuple[str, ...]] = []
    for component in html_before:
        worksheet = _worksheet(component)
        for block in (block for section in _sections(component) if not (worksheet and is_input_section(section))
                      for block in section.get("blocks") or [] if isinstance(block, dict)):
            kind = block.get("kind")
            if kind in _TEXT_KINDS:
                passages.extend(_block_texts(block))
            elif kind in _LIST_KINDS:
                items = [str(item) for item in block.get("items") or []]
                lists.append(tuple(items))
                for item in items:
                    passages.append(item)
                    label, separator, value = item.partition(":")
                    if separator and value.strip():
                        rows.append((label, value))
            elif kind == "table":
                cells: list[str] = []
                for row in block.get("rows") or []:
                    if isinstance(row, dict):
                        label, value = str(row.get("label") or ""), str(row.get("value") or "")
                        passages.append(f"{label}: {value}")
                        rows.append((label, value))
                        cells.append(f"{label} {value}")
                lists.append(tuple(cells))
    sentences = [part for item in faq for part in _SENTENCE_RE.split(str(item.get("answer") or "")) if part.strip()]
    return UnitTeaching(tuple(passages), tuple(rows), tuple(sentences), tuple(example_passages(html_before)),
                        tuple(fact_rows), tuple(lists))


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(idm_fold(text))


def _ordered_share(part: Sequence[str], whole: Sequence[str]) -> float:
    """Share of ``part``'s words found in ``whole`` in the same order (longest common subsequence)."""

    if not part or not whole or not set(part) & set(whole):
        return 0.0
    previous = [0] * (len(whole) + 1)
    for word in part:
        current = [0]
        for position, other in enumerate(whole):
            current.append(previous[position] + 1 if word == other else max(previous[position + 1], current[-1]))
        previous = current
    return previous[-1] / len(part)


def _choices(component: Mapping[str, Any]) -> tuple[list[str], int | None]:
    choices = [choice for choice in component.get("choices") or [] if isinstance(choice, dict)]
    correct = [index for index, choice in enumerate(choices) if choice.get("correct") is True]
    return [str(choice.get("text") or "") for choice in choices], (correct[0] if len(correct) == 1 else None)


def _head(label: str) -> str:
    return idm_fold(_HEAD_END_RE.split(label.strip(), maxsplit=1)[0]).strip(" .")


def _names(folded: str, head: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(head)}(?!\w)", folded) is not None


def labelled_options(texts: Sequence[str], rows: Sequence[tuple[str, str]]) -> dict[int, str]:
    """Option index -> the head of the one taught label it starts with ("bac 01")."""

    heads = {head for label, _value in rows if len(head := _head(label)) >= _MIN_HEAD_CHARS}
    mapped: dict[int, str] = {}
    for index, text in enumerate(texts):
        start = idm_fold(text)[:_HEAD_SEARCH_CHARS]
        named = [head for head in heads if _names(start, head)]
        longest = max(named, key=len) if named else None
        # One label only: an option naming two ("Bậc 02 chưa tới Bậc 03") is not a label of either.
        if longest is not None and all(head in longest for head in named):
            mapped[index] = longest
    return mapped


def quoted_key(component: Mapping[str, Any], teaching: UnitTeaching) -> bool:
    """The key gives itself away against ONE passage shown before the question (R2):

    * it restates the passage in the same word order (a quotation, a list item, a row) far more than any distractor
      restates any passage (keys of 8 words or more that do not name a taught label);
    * a short key is shown word for word while no distractor appears in the html (the only familiar option);
    * the question's case copies an example passage that holds the key and no distractor.
    """

    texts, key = _choices(component)
    passages = [tokens for passage in teaching.passages if (tokens := _tokens(passage))]
    if key is None or len(texts) < _MIN_OPTIONS or not passages:
        return False
    options = [_tokens(text) for text in texts]
    best = [max(_ordered_share(option, passage) for passage in passages) for option in options]
    others = max(best[index] for index in range(len(options)) if index != key)
    words = len(options[key])
    if words <= IDM_ANSWER_SHORT_KEY_MAX_WORDS and best[key] == 1.0 and others <= IDM_ANSWER_SHORT_DISTRACTOR_MAX_SHARE:
        return True
    # A key that names a taught label restates its rule by design ("Cấp 2 vì khách phàn nàn lần thứ hai").
    labelled = (key in labelled_options(texts, teaching.rows)
                or sum(bool(_NUMBERED_LABEL_RE.match(text)) for text in texts) >= _MIN_OPTIONS)
    if (not labelled and words >= IDM_ANSWER_QUOTE_MIN_KEY_WORDS and best[key] >= IDM_ANSWER_QUOTE_MIN_SHARE
            and best[key] - others >= IDM_ANSWER_QUOTE_MIN_MARGIN):
        return True
    stem = _tokens(str(component.get("question") or ""))
    for passage in passages:
        holds = [_ordered_share(option, passage) >= IDM_ANSWER_QUOTE_MIN_SHARE for option in options]
        if (len(passage) >= IDM_ANSWER_STEM_PASSAGE_MIN_WORDS and holds[key] and sum(holds) == 1
                and _ordered_share(passage, stem) >= IDM_ANSWER_STEM_COPY_MIN_SHARE):
            return True
    return False


def restates_example(component: Mapping[str, Any], teaching: UnitTeaching) -> bool:
    """The key restates a worked example shown before the question (QLT-7, c2.l3.u1): it holds at least
    ``IDM_ANSWER_EXAMPLE_MIN_SHARE`` of one example passage's words in order while no distractor holds more than
    ``IDM_ANSWER_EXAMPLE_MAX_DISTRACTOR_SHARE`` of it. The learner recognises the example instead of applying the
    criterion."""

    texts, key = _choices(component)
    if key is None or len(texts) < _MIN_OPTIONS:
        return False
    # A key that names a taught label applies its rule to the case ("Cấp 2: thông báo trưởng nhóm vì …"), as the worked
    # example of a worksheet does for its own case: the same words, no give-away (see ``quoted_key``).
    if key in labelled_options(texts, teaching.rows) or sum(
            bool(_NUMBERED_LABEL_RE.match(text)) for text in texts) >= _MIN_OPTIONS:
        return False
    options = [_tokens(text) for text in texts]
    for passage in teaching.examples:
        words = _tokens(passage)
        if len(words) < IDM_ANSWER_STEM_PASSAGE_MIN_WORDS:
            continue
        shares = [_ordered_share(words, option) for option in options]
        if shares[key] >= IDM_ANSWER_EXAMPLE_MIN_SHARE and all(
                share <= IDM_ANSWER_EXAMPLE_MAX_DISTRACTOR_SHARE for index, share in enumerate(shares) if index != key):
            return True
    return False


def _labelled_key(texts: Sequence[str], key: int, teaching: UnitTeaching) -> bool:
    """A key that OPENS with a taught label applies its rule to the case by design ("Cấp 2: thông báo trưởng nhóm vì
    …", "Bậc 02 (Vietnam Quality), vì …"). A key that names labels further in strings them together (QLT-8 c1.l1.u1:
    "Xây dựng sản phẩm đồng nghĩa với chất lượng đỉnh cao, uy tín tuyệt đối, …") and is measured."""

    if sum(bool(_NUMBERED_LABEL_RE.match(text)) for text in texts) >= _MIN_OPTIONS:
        return True
    heads = {head for label, _value in [*teaching.rows, *teaching.fact_rows]
             if len(head := _head(label)) >= _MIN_HEAD_CHARS}
    opening = idm_fold(texts[key])
    return any(re.match(rf"^\W*(?:\w+\W+){{0,{_LABEL_OPENING_WORDS}}}{re.escape(head)}(?!\w)", opening)
               for head in heads)


def _joined(texts: Sequence[str], key: int, parts: Sequence[str]) -> tuple[float, float, int]:
    """The key's share of its content-word pairs found in ``parts`` read as one text, the best distractor's share, and
    how many of the parts the key draws on."""

    pairs = [content_pairs(text) for text in texts]
    part_pairs = [content_pairs(part) for part in parts]
    # The pairs most parts share are the frame of a table, not its content: "Yêu cầu:" and "Nguy cơ cần chặn:" open
    # every row, and a key that applies one row's criterion is no joined list (QLT-9, ffaf7318 c3.l1.u1).
    counts = Counter(pair for items in part_pairs for pair in items)
    boilerplate = {pair for pair, count in counts.items() if len(part_pairs) > _MIN_OPTIONS
                   and count * 2 > len(part_pairs)}
    whole: set[tuple[str, str]] = set().union(*part_pairs) - boilerplate if part_pairs else set()

    def share(index: int) -> float:
        return len(pairs[index] & whole) / len(pairs[index]) if pairs[index] else 0.0

    others = max((share(index) for index in range(len(texts)) if index != key), default=0.0)
    return share(key), others, sum(bool(pairs[key] & items & whole) for items in part_pairs)


def restates_list(component: Mapping[str, Any], teaching: UnitTeaching) -> bool:
    """The key strings together several items of ONE list or table shown before the question (QLT-8, c1.l1.u1 of run
    0a11aa9e: the five heads of the list in one sentence). The one-passage measure (``quoted_key``) reads each item
    alone and never sees it."""

    texts, key = _choices(component)
    if (key is None or len(texts) < _MIN_OPTIONS or _labelled_key(texts, key, teaching)
            or len(content_pairs(texts[key])) < IDM_ANSWER_JOINED_MIN_KEY_PAIRS):
        return False
    for items in teaching.lists:
        mine, others, touched = _joined(texts, key, items)
        if (touched >= IDM_ANSWER_JOINED_MIN_ITEMS and mine >= IDM_ANSWER_LIST_MIN_SHARE
                and mine - others >= IDM_ANSWER_LIST_MIN_MARGIN):
            return True
    return False


def restates_joined_example(component: Mapping[str, Any], teaching: UnitTeaching) -> bool:
    """The key joins several passages of the worked example shown before it (QLT-8, c2.l5.u1 of run 0a11aa9e: three
    filled rows of the worksheet example in one option). ``restates_example`` reads one passage at a time."""

    texts, key = _choices(component)
    if (key is None or len(texts) < _MIN_OPTIONS or not teaching.examples or _labelled_key(texts, key, teaching)
            or len(content_pairs(texts[key])) < IDM_ANSWER_JOINED_MIN_KEY_PAIRS):
        return False
    mine, others, touched = _joined(texts, key, teaching.examples)
    if (touched >= IDM_ANSWER_JOINED_MIN_ITEMS and mine >= IDM_ANSWER_EXAMPLE_BAG_MIN_SHARE
            and mine - others >= IDM_ANSWER_EXAMPLE_BAG_MIN_MARGIN):
        return True
    # QLT-9: the same rows rewritten in other words keep the example's words, not its word pairs (ffaf7318 c3.l2.u2).
    words = [set(content_words(text)) for text in texts]
    if len(words[key]) < IDM_ANSWER_EXAMPLE_WORDS_MIN_KEY_WORDS:
        return False
    passages = [set(content_words(passage)) for passage in teaching.examples]
    shown: set[str] = set().union(*passages)
    shares = [len(option & shown) / len(option) if option else 0.0 for option in words]
    best_other = max((share for index, share in enumerate(shares) if index != key), default=0.0)
    return (sum(bool(words[key] & passage) for passage in passages) >= IDM_ANSWER_JOINED_MIN_ITEMS
            and shares[key] >= IDM_ANSWER_EXAMPLE_WORDS_MIN_SHARE
            and shares[key] - best_other >= IDM_ANSWER_EXAMPLE_WORDS_MIN_MARGIN)


def _rung(head: str) -> tuple[str, int] | None:
    match = _RUNG_RE.match(head)
    return (match.group("stem"), int(match.group("number"))) if match else None


def _affirmed(text: str) -> tuple[set[str], set[tuple[str, str]]]:
    """The folded words and content pairs of ``text`` outside a negated clause part: a negation ("chưa", "không",
    "without") negates the rest of its clause."""

    words: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    for clause in _CLAUSE_RE.split(text):
        tokens = _tokens(clause)
        stop = next((index for index, token in enumerate(tokens) if token in _NEGATIONS), len(tokens))
        words.update(tokens[:stop])
        pairs.update(content_pairs(" ".join(tokens[:stop])))
    return words, pairs


def _literal_rival(component: Mapping[str, Any], mapped: Mapping[int, str], key: int,
                   rows: Sequence[tuple[str, str]]) -> int | None:
    """A distractor whose row criteria the case states literally (QLT-7, c1.l1.u2): codes (ISO, CE) and content pairs
    of that row that no other option's row has, not negated in the case. On a ladder only a higher rung counts."""

    values: dict[str, list[str]] = {head: [] for head in set(mapped.values())}
    for label, value in rows:
        if (head := _head(label)) in values:
            values[head].append(value)
    codes = {head: {code.lower() for text in texts for code in _CODE_RE.findall(text)}
             for head, texts in values.items()}
    pairs = {head: set().union(*(content_pairs(text) for text in texts)) for head, texts in values.items()}
    words, case_pairs = _affirmed(str(component.get("question") or ""))
    key_head, key_rung = mapped[key], _rung(mapped[key])
    for index, head in sorted(mapped.items()):
        rung = _rung(head)
        if head == key_head or (rung and key_rung and rung[0] == key_rung[0] and rung[1] < key_rung[1]):
            continue
        others = [other for other in values if other != head]
        own_codes = codes[head] - set().union(*(codes[other] for other in others)) - set(content_words(head))
        own_pairs = pairs[head] - set().union(*(pairs[other] for other in others))
        code_hits, pair_hits = len(own_codes & words), len(own_pairs & case_pairs)
        if (code_hits >= IDM_ANSWER_CONTESTED_MIN_CODES or pair_hits >= IDM_ANSWER_CONTESTED_MIN_PAIRS
                or (code_hits and pair_hits)):
            return index
    return None


def contested_option(component: Mapping[str, Any], teaching: UnitTeaching) -> int | None:
    """A distractor whose taught criteria the question's case meets at least as well as the key's (R2), or that the
    case meets literally (QLT-7), or None.

    Only for options that name labels the unit teaches (html table rows and "label: …" items, and the table-row facts
    the writer was given): the criteria of a label are its row values and the sentences of the html and of the unit's
    FAQ answers that name it and no other option's label.
    """

    texts, key = _choices(component)
    rows = [*teaching.rows, *teaching.fact_rows]
    mapped = labelled_options(texts, rows)
    if key is None or key not in mapped or len(set(mapped.values())) < _MIN_OPTIONS:
        return None
    heads = set(mapped.values())
    criteria: dict[str, list[str]] = {head: [] for head in heads}
    for label, value in rows:
        if (head := _head(label)) in criteria:
            criteria[head].append(value)
    for sentence in (*teaching.passages, *teaching.faq_sentences):
        folded = idm_fold(sentence)
        named = [head for head in heads if _names(folded, head)]
        if len(named) == 1:
            criteria[named[0]].append(sentence)
    label_words = {word for head in heads for word in content_words(head)}
    case = set(content_words(str(component.get("question") or "")))

    def score(head: str) -> int:
        return max((len((set(content_words(text)) - label_words) & case) for text in criteria[head]), default=0)

    key_score = score(mapped[key])
    rivals = [(score(head), index) for index, head in mapped.items() if head != mapped[key]]
    best, rival = max(rivals) if rivals else (0, None)
    if best >= max(IDM_ANSWER_CONTESTED_MIN_WORDS, key_score):
        return rival
    return _literal_rival(component, mapped, key, rows)


# A number and the word after it ("2 giờ", "50 triệu"): what an example states, measured.
_MEASURE_UNIT_RE: Final = re.compile(r"[ \t]*([^\W\d_]+)")


def resolved_choices(component: Mapping[str, Any]) -> dict[str, Any]:
    """``component`` with every option that is only a listed item's name ("Phương án 2", "Cách 2") read as that item's
    text, for the measures (the served options never change)."""

    items = named_items(str(component.get("question") or ""))
    choices = component.get("choices")
    if len(items) < _MIN_OPTIONS or not isinstance(choices, list):
        return dict(component)
    resolved = [{**choice, "text": items.get(idm_fold(str(choice.get("text") or "")).strip(" .:;,"),
                                             choice.get("text"))} if isinstance(choice, dict) else choice
                for choice in choices]
    return {**component, "choices": resolved}


def _measures(text: str) -> set[tuple[str, str]]:
    found = set()
    for _raw, normal, _start, end in number_spans(text):
        if unit := _MEASURE_UNIT_RE.match(text, end):
            found.add((normal, idm_fold(unit.group(1))))
    return found


def _case_lines(question: str) -> str:
    """The stem without its listed items: the case the question gives."""

    return " ".join(line for line in question.splitlines() if not line.strip().startswith("-"))


def copies_example_measure(component: Mapping[str, Any], teaching: UnitTeaching) -> bool:
    """The key states a number with its unit from the worked example ("2 giờ") that neither the case nor any
    distractor states (run 5496155c, c2.l4.u2): the learner picks the example's figure."""

    texts, key = _choices(component)
    if key is None or len(texts) < _MIN_OPTIONS or not teaching.examples:
        return False
    shown = set().union(*(_measures(passage) for passage in teaching.examples))
    others = set().union(*(_measures(text) for index, text in enumerate(texts) if index != key))
    # Review of dfb6db2 (item 2): a number the lesson teaches as its rule ("phản hồi trong 24 giờ") is the answer the
    # learner applies, not the example's figure.
    taught = set().union(*(_measures(passage) for passage in [
        *(text for text in teaching.passages if not any(example in text for example in teaching.examples)),
        *teaching.faq_sentences, *(value for _label, value in teaching.fact_rows)]))
    return bool((_measures(texts[key]) & shown) - _measures(_case_lines(str(component.get("question") or "")))
                - others - taught)


def _item_share(text: str, passage: str) -> float:
    pairs = content_pairs(text)
    return len(pairs & content_pairs(passage)) / len(pairs) if len(pairs) >= IDM_ANSWER_STEM_ITEM_MIN_PAIRS else 0.0


def _copies(text: str, passage: str) -> bool:
    """``text`` (a listed item) copies ``passage``: most of its content-word pairs, or nearly all its content words."""

    words = set(content_words(text))
    return _item_share(text, passage) >= IDM_ANSWER_STEM_ITEM_COPY_SHARE or (
        len(words) >= IDM_ANSWER_STEM_ITEM_MIN_WORDS
        and len(words & set(content_words(passage))) / len(words) >= IDM_ANSWER_STEM_ITEM_WORD_SHARE)


def stem_copies_answer(component: Mapping[str, Any], teaching: UnitTeaching, original: Mapping[str, Any]) -> bool:
    """A listed item of the stem copies the worked example or a table row shown before the question, and the key gives
    the answer that copy points to (run 5496155c): it names the row's label, or a pair of the example's words the case
    does not use and some distractor lacks; or the key is only that item's name (``original``, the served options) and
    its item copies more than every other item."""

    texts, key = _choices(component)
    items = named_items(str(component.get("question") or ""))
    if key is None or len(texts) < _MIN_OPTIONS or not items:
        return False
    case = content_pairs(str(component.get("question") or ""))
    example = " ".join(teaching.examples)
    groups = [(_head(label), value) for label, value in teaching.rows if len(_head(label)) >= _MIN_HEAD_CHARS]
    folded = [idm_fold(text) for text in texts]
    key_pairs = content_pairs(texts[key])
    for name, item in items.items():
        if example and _copies(item, example):
            # The example's words the key adds to the case: its answer, unless every distractor says it too.
            answer = (key_pairs & content_pairs(example)) - case
            if any(pair not in content_pairs(text) for pair in answer
                   for index, text in enumerate(texts) if index != key):
                return True
        for head, value in groups:
            if (_copies(item, value) and _names(folded[key], head)
                    and not all(_names(text, head) for text in folded)):
                return True
        served = idm_fold(str((original.get("choices") or [{}])[key].get("text") or "")).strip(" .:;,")
        if served == name and example:
            share = _item_share(item, example)
            rest = [_item_share(other, example) for other_name, other in items.items() if other_name != name]
            if (share >= IDM_ANSWER_STEM_ITEM_COPY_SHARE
                    and share - max(rest, default=0.0) >= IDM_ANSWER_STEM_ITEM_MARGIN):
                return True
    return False


__all__ = [
    "UnitTeaching", "contested_option", "copies_example_measure", "example_passages", "fact_table_rows",
    "labelled_options", "quoted_key", "resolved_choices", "restates_example", "restates_joined_example",
    "restates_list", "stem_copies_answer", "unit_teaching",
]
