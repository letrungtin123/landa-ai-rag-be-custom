"""Question rules of the Node acceptance mirror (``app.idm.node_acceptance``): the problem XML Node
builds, the single-choice question code and the workspace problem text limits.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Final

from app.idm.node_acceptance_html import _EMAIL_ONLY_RE, _PHONE_ONLY_RE, _URL_ONLY_RE
from app.idm.node_acceptance_text import (
    _JS_WS,
    _SURROGATES,
    _js_number,
    _js_string,
    _list,
    _nullish,
    _record,
    _Rejected,
    escape_xml,
    instructional_fold,
    js_len,
    js_trim,
    read_scalar_string,
    read_string,
    visible_html_text,
)

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
