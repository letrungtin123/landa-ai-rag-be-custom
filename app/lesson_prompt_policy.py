"""Small prompt contracts; no provider, retrieval or acceptance policy here."""
from typing import Literal

from app.workflows.contracts import WorkflowFailure

ARCHITECT_POLICY_MAX_CHARS = 12_000
LESSON_LANGUAGE_POLICY_VERSION = "lesson-language-1"


def bounded_architect_policy(policy: str) -> str:
    """Node composes whole teaching sections. Never truncate its safety policy."""
    policy = policy.strip()
    if len(policy) > ARCHITECT_POLICY_MAX_CHARS:
        raise WorkflowFailure(
            "ARCHITECTURE_SCOPE_INCOMPLETE",
            "The server architecture policy exceeds its bounded context allowance.",
            internal_code="ARCH_POLICY_BUDGET_EXCEEDED",
            failure_stage="course_architect_policy_composition",
            diagnostics={"policy_chars": len(policy), "policy_max_chars": ARCHITECT_POLICY_MAX_CHARS},
        )
    return policy


def lesson_output_language_policy(locale: Literal["vi", "en"]) -> str:
    language = "Vietnamese (vi), with diacritics" if locale == "vi" else "English (en)"
    return "\n".join([
        f"SERVER OUTPUT LANGUAGE POLICY: {LESSON_LANGUAGE_POLICY_VERSION}.",
        f"Write newly generated learner-facing text in {language}, regardless of source language or UI labels.",
        "This applies to explanations, headings, examples, FAQ questions/answers, problem questions/choices/feedback, crossword clues, diagram labels, and sortable instructions/items.",
        "Preserve approved course/chapter/lesson/unit titles exactly. Do not translate or change IDs, keys, component_plan_id, block/objective/concept IDs, source refs, canonical fact IDs, PRIMARY/SUPPORTING evidence or component types. Language is not authority to change the approved scope or plan.",
        "Preserve proper names and source terminology when needed. Crossword clues must describe the exact answer term and its normalized spelling; do not translate an answer independently from its clue. Preserve quoted evidence in its source language and identify it as a source quotation, not a translated factual claim.",
        "Do not add factual claims during translation. Respect the same source, schema, semantic HTML and repair-target restrictions. No styles, arbitrary classes, scripts or media assets.",
    ])
