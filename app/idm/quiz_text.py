"""Layout and checks of the text of a single-choice question (QC feedback 2026-10-21, A1/A2; contract K1).

The QC department found the explanations rambling and hollow ("B - Đúng vì phương án này dũng cảm Say No … tạo điểm
khác biệt tuyệt đối"; three definitions restated before "A - Sai vì …", all in one paragraph) and the stems a wall of
text ("…như sau: - Phân khúc X: … - Phân khúc Y: … - Phân khúc Z: … Theo quy tắc … bạn nên đưa ra quyết định nào?").
The question and the explanation are plain text whose line feeds are the only layout (K1): in the question a line
that starts with "- " is a list item and the last line is the question; the explanation's first line is the principle
and every other line one option, "A - Đúng vì …" / "A - Sai vì …".

* ``lay_out_quiz`` (deterministic, after the options were shuffled): an inline list introduced by a colon gets one line
  per item, the question sentence moves to its own last line, and an explanation written as "principle A - …; B - …"
  gets one line per option with the separator "A - ". Only whitespace, list markers and the segment punctuation
  change; never a word.
* ``quiz_findings``: what the layout cannot fix, each a code the W5 writer repairs once (a targeted repair), then a
  review note: the explanation's layout and verdicts, its empty praise, its length, an option line that only repeats
  its option; the stem's layout and length.

Limits and word lists live in ``app.idm.policy``; ``quiz_hint`` gives the repair prompt the measured numbers.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any, Final

from app.idm.mcq import OPTION_LETTERS, question_complete
from app.idm.policy import (
    IDM_MCQ_EXPLANATION_MAX_CHARS,
    IDM_MCQ_EXPLANATION_OPTION_MAX_SENTENCES,
    IDM_MCQ_EXPLANATION_OPTION_MAX_WORDS,
    IDM_MCQ_EXPLANATION_PRINCIPLE_MAX_SENTENCES,
    IDM_MCQ_EXPLANATION_PRINCIPLE_MAX_WORDS,
    IDM_MCQ_EXPLANATION_RESTATE_MIN_WORDS,
    IDM_MCQ_EXPLANATION_RESTATE_NGRAM,
    IDM_MCQ_EXPLANATION_RESTATE_SHARE,
    IDM_MCQ_STEM_CONTEXT_MAX_SENTENCES,
    IDM_MCQ_STEM_MAX_CHARS,
    IDM_MCQ_STEM_MAX_WORDS,
    QUIZ_FILLER_EN,
    QUIZ_FILLER_VI,
)
from app.idm.text import EvidenceIndex, idm_fold, ngram_overlap

EXPLANATION_LAYOUT_CODE: Final = "IDM_W5_EXPLANATION_LAYOUT"
EXPLANATION_VERDICT_CODE: Final = "IDM_W5_EXPLANATION_VERDICT"
EXPLANATION_FILLER_CODE: Final = "IDM_W5_EXPLANATION_FILLER"
EXPLANATION_LENGTH_CODE: Final = "IDM_W5_EXPLANATION_LENGTH"
EXPLANATION_RESTATES_CODE: Final = "IDM_W5_EXPLANATION_RESTATES_OPTION"
QUESTION_LAYOUT_CODE: Final = "IDM_W5_QUESTION_LAYOUT"
QUESTION_LENGTH_CODE: Final = "IDM_W5_QUESTION_LENGTH"
QUIZ_CODES: Final = frozenset({EXPLANATION_LAYOUT_CODE, EXPLANATION_VERDICT_CODE, EXPLANATION_FILLER_CODE,
                               EXPLANATION_LENGTH_CODE, EXPLANATION_RESTATES_CODE, QUESTION_LAYOUT_CODE,
                               QUESTION_LENGTH_CODE})
QUESTION_CODES: Final = frozenset({QUESTION_LAYOUT_CODE, QUESTION_LENGTH_CODE})
# Findings about the wording, not the correctness: a slot kept with only these is a polish notice.
QUIZ_STYLE_CODES: Final = frozenset({EXPLANATION_FILLER_CODE, EXPLANATION_LENGTH_CODE, QUESTION_LAYOUT_CODE,
                                     QUESTION_LENGTH_CODE})
QUESTION_LAID_OUT: Final = "w5_problem_question_laid_out"
EXPLANATION_LAID_OUT: Final = "w5_problem_explanation_laid_out"

LIST_MARKER: Final = "- "
OPTION_SEPARATOR: Final = " - "
_SINGLE_CHOICE_TYPES: Final = frozenset({"multiple_choice", "single_choice"})
# An inline list item: a marker between spaces before an item that starts with a capital letter or a digit.
_INLINE_MARKER_RE: Final = re.compile(r"[ \t]+[-–•][ \t]+(?=\S)")  # noqa: RUF001 - the en dash is a marker
_MIN_INLINE_ITEMS: Final = 2
# A list item written with another marker at a line start (a bullet, a star or an en dash).
_OTHER_MARKER_RE: Final = re.compile(r"^[•*–][ \t]+(?=\S)")  # noqa: RUF001 - the en dash is a marker
# An option label at a line start: "A - ", "A: ", "A. ", "A) " or a dash other than the hyphen.
_LINE_LABEL_RE: Final = re.compile(r"^([A-F])[ \t]*(?:-|–|—|:|\.|\))[ \t]+(?=\S)")  # noqa: RUF001 - dashes
# The start of one option's segment inside a paragraph (as ``app.idm.mcq`` reads it).
_SEGMENT_RE: Final = re.compile(r"(?:^|(?<=[\s;,.(]))([A-F])[ \t]*(?:[—–:.)]|-(?=\s))[ \t]*")  # noqa: RUF001
# The verdict and the reason connector right after the label.
_VERDICT_RE: Final = re.compile(
    r"^(?:(?P<vi>đúng|sai|chưa đúng)(?:[ \t]*[,:])?[ \t]+(?:là vì|bởi vì|vì|bởi|do)"
    r"|(?P<en>correct|incorrect|right|wrong)(?:[ \t]*[,:])?[ \t]+(?:because|since|as))(?!\w)", re.IGNORECASE)
_POSITIVE_VERDICTS: Final = frozenset({"đúng", "correct", "right"})
# A sentence ends at . ! ? or … followed by a space and a capital letter, a digit or an opening quote; never after an
# abbreviation ("TP.", "v.v.", "e.g.").
_SENTENCE_END_RE: Final = re.compile(r"(?<=[.!?…])[\"'”’)\]]*[ \t]+")  # noqa: RUF001 - closing quotes
_ABBREVIATION_RE: Final = re.compile(r"^(?:[A-ZĐ]{1,2}|(?i:v\.v|e\.g|i\.e|mr|ms|dr|no|vs|tr))\.$")
_SENTENCE_STARTS: Final = "\"'“‘(-"  # noqa: RUF001 - opening quotes
_END_PUNCTUATION: Final = ".!?…"
_TRAILING_JOINERS: Final = ";, \t"
_MAX_HINT_ITEMS: Final = 6


def _normal(text: str) -> str:
    return unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")


def _words(text: str) -> int:
    return len(text.split())


def sentence_spans(text: str) -> list[str]:
    """The sentences of one line of ``text``, in order (punctuation kept)."""

    pieces: list[str] = []
    start = 0
    for match in _SENTENCE_END_RE.finditer(text):
        following = text[match.end():match.end() + 1]
        head = text[start:match.start()].split()
        if not (following.isupper() or following.isdigit() or following in _SENTENCE_STARTS) or (
                head and _ABBREVIATION_RE.fullmatch(head[-1])):
            continue
        pieces.append(text[start:match.end()].strip())
        start = match.end()
    pieces.append(text[start:].strip())
    return [piece for piece in pieces if piece]


def _lines(text: str) -> list[str]:
    return [line.strip() for line in _normal(text).split("\n") if line.strip()]


# --- layout -------------------------------------------------------------------------------------------------------
def _split_inline_list(line: str) -> list[str]:
    """A line with a list introduced by a colon written inline (": - X … - Y …") as the lead and one "- " line per
    item; the line unchanged otherwise."""

    markers = [match for match in _INLINE_MARKER_RE.finditer(line)
               if line[match.end()].isupper() or line[match.end()].isdigit()]
    lead = line[:markers[0].start()].rstrip() if markers else ""
    if len(markers) < _MIN_INLINE_ITEMS or not lead.endswith(":"):
        return [line]
    bounds = [match.end() for match in markers] + [len(line)]
    items = [line[bounds[index]:markers[index + 1].start() if index + 1 < len(markers) else len(line)].strip()
             for index in range(len(markers))]
    return [lead, *(LIST_MARKER + item for item in items if item)]


def _question_alone(lines: list[str]) -> list[str]:
    """The question sentence (the last sentence, when it asks) on its own last line."""

    if not lines:
        return lines
    parts = sentence_spans(lines[-1])
    if len(parts) < _MIN_INLINE_ITEMS or not question_complete(parts[-1]):
        return lines
    head = lines[-1][:len(lines[-1]) - len(parts[-1])].rstrip()
    return [*lines[:-1], head, parts[-1]]


def layout_question(text: str) -> str:
    """``text`` with an inline list on lines of their own, other list markers as "- " and the question on its own last
    line; only whitespace and list markers change."""

    lines: list[str] = []
    for line in _lines(text):
        for part in _split_inline_list(line):
            lines.append(_OTHER_MARKER_RE.sub(LIST_MARKER, part))
    return "\n".join(_question_alone(lines))


def _option_line(segment: str) -> str:
    """One option's segment as its line: the label separated by " - ", no trailing joiner, a closing full stop."""

    line = segment.strip().rstrip(_TRAILING_JOINERS)
    line = _LINE_LABEL_RE.sub(lambda match: match.group(1) + OPTION_SEPARATOR, line, count=1)
    return line if line[-1:] in _END_PUNCTUATION else line + "."


def _labelled_lines(text: str) -> str:
    return "\n".join(_option_line(line) if _LINE_LABEL_RE.match(line) else line for line in _lines(text))


def layout_explanation(text: str, option_count: int) -> str:
    """``text`` as the principle line and one line per option in letter order, when its options are named once each
    at the start of a segment (out of their order only when every segment is one sentence, so no remark moves with
    another option); otherwise ``text`` with only the separators of its labelled lines normalised."""

    normal = _normal(text).strip()
    letters = list(OPTION_LETTERS[:option_count])
    starts = [match for match in _SEGMENT_RE.finditer(normal) if match.group(1) in letters]
    if sorted(match.group(1) for match in starts) != letters:
        return _labelled_lines(normal)
    bounds = [match.start() for match in starts] + [len(normal)]
    lines = {match.group(1): _option_line(" ".join(normal[bounds[index]:bounds[index + 1]].split()))
             for index, match in enumerate(starts)}
    if [match.group(1) for match in starts] != letters and any(
            len(sentence_spans(line)) > IDM_MCQ_EXPLANATION_OPTION_MAX_SENTENCES for line in lines.values()):
        return _labelled_lines(normal)
    principle = " ".join(normal[:bounds[0]].split())
    return "\n".join([*([principle] if principle else []), *(lines[letter] for letter in letters)])


def lay_out_quiz(component: dict[str, Any]) -> tuple[dict[str, Any], Counter[str]]:
    """A single-choice ``component`` with its question and explanation laid out (``layout_question``,
    ``layout_explanation``) and the counts of what changed."""

    codes: Counter[str] = Counter()
    choices = component.get("choices")
    kind = str(component.get("problem_type") or "multiple_choice").strip().lower()
    if kind not in _SINGLE_CHOICE_TYPES or not isinstance(choices, list):
        return component, codes
    result = dict(component)
    question = component.get("question")
    if isinstance(question, str) and (laid := layout_question(question)) and laid != question:
        result["question"] = laid
        codes[QUESTION_LAID_OUT] += 1
    explanation = component.get("explanation")
    if isinstance(explanation, str) and (laid := layout_explanation(explanation, len(choices))) and (
            laid != explanation):
        result["explanation"] = laid
        codes[EXPLANATION_LAID_OUT] += 1
    return result, codes


# --- checks -------------------------------------------------------------------------------------------------------
def _option_lines(lines: Sequence[str], count: int) -> list[tuple[str, str]] | None:
    """(letter, text after the label) of every option line, when the lines are the principle and one line per option
    in order A, B, C …; ``None`` otherwise."""

    if len(lines) != count + 1 or _LINE_LABEL_RE.match(lines[0]):
        return None
    found = [(match.group(1), line[match.end():]) for line in lines[1:] if (match := _LINE_LABEL_RE.match(line))]
    return found if [letter for letter, _rest in found] == list(OPTION_LETTERS[:count]) else None


def _choices(component: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [choice for choice in component.get("choices") or [] if isinstance(choice, dict)]


def filler_phrases(text: str, evidence: EvidenceIndex | None = None) -> list[str]:
    """The empty-praise phrases of ``text`` (``QUIZ_FILLER_VI``/``_EN``) its facts do not use themselves."""

    lowered = _normal(text).lower()
    found = []
    for phrase in (*QUIZ_FILLER_VI, *QUIZ_FILLER_EN):
        if not re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", lowered):
            continue
        tokens = idm_fold(phrase).split()
        in_facts = evidence is not None and (
            all(pair in evidence.pairs for pair in pairwise(tokens)) if len(tokens) > 1
            else tokens[0] in evidence.words)
        if not in_facts:
            found.append(phrase)
    # "khác biệt tuyệt đối" also contains "tuyệt đối": name the longer phrase once (and "say no" is in both lists).
    unique = list(dict.fromkeys(found))
    return [phrase for phrase in unique if not any(phrase != other and phrase in other for other in unique)]


def _too_long(lines: Sequence[str], options: list[tuple[str, str]] | None, text: str) -> list[str]:
    """What of the explanation is over its limits, as repair-prompt facts ("principle 52 words (at most 35)")."""

    over: list[str] = []
    if lines and not _LINE_LABEL_RE.match(lines[0]):
        words, sentences = _words(lines[0]), len(sentence_spans(lines[0]))
        if words > IDM_MCQ_EXPLANATION_PRINCIPLE_MAX_WORDS or sentences > IDM_MCQ_EXPLANATION_PRINCIPLE_MAX_SENTENCES:
            over.append(f"principle {words} words in {sentences} sentence(s) (at most "
                        f"{IDM_MCQ_EXPLANATION_PRINCIPLE_MAX_WORDS} words, "
                        f"{IDM_MCQ_EXPLANATION_PRINCIPLE_MAX_SENTENCES} sentences)")
    for letter, rest in options or ():
        words, sentences = _words(rest), len(sentence_spans(rest))
        if words > IDM_MCQ_EXPLANATION_OPTION_MAX_WORDS or sentences > IDM_MCQ_EXPLANATION_OPTION_MAX_SENTENCES:
            over.append(f"option {letter} {words} words in {sentences} sentence(s) (at most "
                        f"{IDM_MCQ_EXPLANATION_OPTION_MAX_WORDS} words, one sentence)")
    if len(text) > IDM_MCQ_EXPLANATION_MAX_CHARS:
        over.append(f"{len(text)} characters (at most {IDM_MCQ_EXPLANATION_MAX_CHARS})")
    return over


def _restating(options: Sequence[tuple[str, str]], choices: Sequence[dict[str, Any]]) -> list[str]:
    """Letters of the option lines whose reason mostly repeats the option's own words."""

    found = []
    for (letter, rest), choice in zip(options, choices, strict=False):
        option = str(choice.get("text") or "")
        reason = _VERDICT_RE.sub("", rest.strip(), count=1)
        if (_words(option) >= IDM_MCQ_EXPLANATION_RESTATE_MIN_WORDS
                and ngram_overlap(option, reason, IDM_MCQ_EXPLANATION_RESTATE_NGRAM)
                >= IDM_MCQ_EXPLANATION_RESTATE_SHARE):
            found.append(letter)
    return found


def _verdicts(options: Sequence[tuple[str, str]], choices: Sequence[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """(letters whose line has no verdict and reason, letters whose verdict contradicts the answer key)."""

    missing, wrong = [], []
    for (letter, rest), choice in zip(options, choices, strict=False):
        match = _VERDICT_RE.match(rest.strip())
        if match is None:
            missing.append(letter)
            continue
        verdict = (match.group("vi") or match.group("en") or "").lower()
        if (verdict in _POSITIVE_VERDICTS) != (choice.get("correct") is True):
            wrong.append(letter)
    return missing, wrong


def explanation_report(component: Mapping[str, Any], evidence: EvidenceIndex | None = None) -> dict[str, list[str]]:
    """Code -> the measured facts behind it (letters, phrases, numbers) for the explanation of ``component``."""

    text = _normal(str(component.get("explanation") or "")).strip()
    choices = _choices(component)
    lines = _lines(text)
    options = _option_lines(lines, len(choices))
    report: dict[str, list[str]] = {}
    if options is None:
        report[EXPLANATION_LAYOUT_CODE] = [f"{len(lines)} line(s); it needs 1 principle line and {len(choices)} "
                                           "option lines in order"]
    else:
        missing, wrong = _verdicts(options, choices)
        if missing:
            report[EXPLANATION_LAYOUT_CODE] = [f"no verdict and reason on the lines of {', '.join(missing)}"]
        if wrong:
            report[EXPLANATION_VERDICT_CODE] = [f"the verdict of {', '.join(wrong)} contradicts the answer key"]
        if restating := _restating(options, choices):
            report[EXPLANATION_RESTATES_CODE] = [f"the lines of {', '.join(restating)} repeat their option"]
    if fillers := filler_phrases(text, evidence):
        report[EXPLANATION_FILLER_CODE] = fillers[:_MAX_HINT_ITEMS]
    if over := _too_long(lines, options, text):
        report[EXPLANATION_LENGTH_CODE] = over[:_MAX_HINT_ITEMS]
    return report


def question_report(component: Mapping[str, Any]) -> dict[str, list[str]]:
    """Code -> the measured facts behind it for the question (stem) of ``component``."""

    lines = _lines(str(component.get("question") or ""))
    report: dict[str, list[str]] = {}
    layout: list[str] = []
    if any(len(_split_inline_list(line)) > 1 for line in lines):
        layout.append("a list written inside one line")
    if lines and len(sentence_spans(lines[-1])) > 1:
        layout.append("the question sentence is not alone on the last line")
    context = sum(len(sentence_spans(line)) for line in lines[:-1] if not line.startswith(LIST_MARKER))
    if context > IDM_MCQ_STEM_CONTEXT_MAX_SENTENCES:
        layout.append(f"{context} context sentences (at most {IDM_MCQ_STEM_CONTEXT_MAX_SENTENCES})")
    if layout:
        report[QUESTION_LAYOUT_CODE] = layout
    words, chars = sum(_words(line) for line in lines), len("\n".join(lines))
    if words > IDM_MCQ_STEM_MAX_WORDS or chars > IDM_MCQ_STEM_MAX_CHARS:
        report[QUESTION_LENGTH_CODE] = [f"{words} words, {chars} characters (at most {IDM_MCQ_STEM_MAX_WORDS} words "
                                        f"and {IDM_MCQ_STEM_MAX_CHARS} characters)"]
    return report


def quiz_findings(component: Mapping[str, Any], evidence: EvidenceIndex | None = None, *,
                  explanation: bool = True) -> list[str]:
    """The quiz-text codes of a single-choice ``component`` (none for another kind or a question without options):
    what its layout (``lay_out_quiz``, applied to every provider slot) does not fix. Without ``explanation`` only the
    stem is checked (the caller already reports an explanation that misses an option or teaches nothing)."""

    kind = str(component.get("problem_type") or "multiple_choice").strip().lower()
    if kind not in _SINGLE_CHOICE_TYPES or not _choices(component):
        return []
    laid, _codes = lay_out_quiz(dict(component))
    return [*question_report(laid), *(explanation_report(laid, evidence) if explanation else ())]


def quiz_hint(code: str, component: Mapping[str, Any], evidence: EvidenceIndex | None = None) -> str:
    """The server-measured facts behind ``code`` for its repair line: letters, numbers and the empty-praise phrases of
    the fixed word list (never other provider text)."""

    laid, _codes = lay_out_quiz(dict(component))
    report = {**question_report(laid), **explanation_report(laid, evidence)}
    items = report.get(code, [])
    if code == EXPLANATION_FILLER_CODE:
        return " (remove: " + ", ".join(f'"{phrase}"' for phrase in items) + ")" if items else ""
    return f" ({'; '.join(items)})" if items else ""


__all__ = [
    "EXPLANATION_FILLER_CODE", "EXPLANATION_LAID_OUT", "EXPLANATION_LAYOUT_CODE", "EXPLANATION_LENGTH_CODE",
    "EXPLANATION_RESTATES_CODE", "EXPLANATION_VERDICT_CODE", "LIST_MARKER", "OPTION_SEPARATOR", "QUESTION_CODES",
    "QUESTION_LAID_OUT",
    "QUESTION_LAYOUT_CODE", "QUESTION_LENGTH_CODE", "QUIZ_CODES", "QUIZ_STYLE_CODES", "explanation_report",
    "filler_phrases", "lay_out_quiz", "layout_explanation", "layout_question", "question_report", "quiz_findings",
    "quiz_hint", "sentence_spans",
]
