"""Source lines and fallback FAQ text of a source-locked unit.

Moved verbatim from ``app.services.lesson_author.staged.source_locked`` (refactor wave 3b), where they were private:
the orchestration V2 source-locked builder uses them too.
"""

from __future__ import annotations

import re

from app.content.instructional_quality import source_clarification_signals


def source_locked_html_facts(fact_texts: list[str]) -> list[str]:
    """Split raw source facts into displayable source lines without rewriting them."""
    lines: list[str] = []
    for fact_text in fact_texts:
        text = re.sub(r"\s+", " ", fact_text).strip()
        if not text:
            continue
        numbered_starts = list(re.finditer(r"(?<!\w)\d+[.)]\s+", text))
        if not numbered_starts:
            lines.append(text)
            continue
        prefix = text[:numbered_starts[0].start()].strip()
        if prefix:
            lines.append(prefix)
        for index, match in enumerate(numbered_starts):
            end = numbered_starts[index + 1].start() if index + 1 < len(numbered_starts) else len(text)
            item = text[match.start():end].strip()
            if item:
                lines.append(item)
    return lines


def source_locked_faq_question(answer: str, title: str, index: int, locale: str) -> str:
    label = re.split(r"[:.!?]", answer, maxsplit=1)[0].strip()
    if len(label) >= 6 and len(label) <= 100:
        return (
            f"What should learners understand about {label}?"
            if locale == "en"
            else f"Người học cần hiểu gì về {label}?"
        )
    if locale == "en":
        return (f"What should learners remember about '{title}'?" if index == 0
                else f"Which condition needs attention in '{title}'?")
    return f"Người học cần ghi nhớ gì về '{title}'?" if index == 0 else f"Điều kiện nào cần lưu ý trong '{title}'?"


def source_locked_faq_answers(fact_texts: list[str]) -> list[str]:
    """Use only explicit source conditions/exceptions for a fallback FAQ."""
    answers = source_clarification_signals(fact_texts)
    return [answer[:600] for answer in answers[:2]] if len(answers) >= 2 else []
