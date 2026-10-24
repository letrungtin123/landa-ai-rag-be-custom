"""The staged component types and the names a provider may use for them.

Moved verbatim from ``app.services.lesson_author.staged.plan`` (refactor wave 3b).
"""

from __future__ import annotations

from typing import Any

STAGED_COMPONENT_TYPE_ALIASES = {
    "html": "html",
    "problem": "problem",
    "quiz": "problem",
    "question": "problem",
    "multiple_choice": "problem",
    "multiple-select": "problem",
    "multiple_select": "problem",
    "multi_choice": "problem",
    "multi_select": "problem",
    "mcq": "problem",
    "dropdown": "problem",
    "select": "problem",
    "numerical": "problem",
    "numeric": "problem",
    "short_text": "problem",
    "short-answer": "problem",
    "short_answer": "problem",
    "la_problem": "problem",
    "la_faq": "la_faq",
    "faq": "la_faq",
    "la_sortable": "la_sortable",
    "sortable": "la_sortable",
    "ordering": "la_sortable",
    "la_crossword": "la_crossword",
    "crossword": "la_crossword",
    "vocabulary": "la_crossword",
    "la_diagram": "la_diagram",
    "diagram": "la_diagram",
    "flowchart": "la_diagram",
    "mindmap": "la_diagram",
}


def normalize_staged_component_type(value: Any) -> str | None:
    raw = str(value or "").strip().casefold().replace(" ", "_")
    return STAGED_COMPONENT_TYPE_ALIASES.get(raw)
