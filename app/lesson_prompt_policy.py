"""Small prompt contracts; no provider, retrieval or acceptance policy here."""
from typing import Literal

from app.workflows.contracts import WorkflowFailure

ARCHITECT_POLICY_MAX_CHARS = 12_000
LESSON_LANGUAGE_POLICY_VERSION = "lesson-language-1"
LESSON_INSTRUCTIONAL_POLICY_VERSION = "evidence-teaching-2"


def lesson_instructional_quality_policy() -> str:
    """Pedagogical synthesis is allowed; domain claims and ownership are not."""
    return "\n".join([
        f"EVIDENCE-BOUNDED TEACHING POLICY: {LESSON_INSTRUCTIONAL_POLICY_VERSION}.",
        "Turn available knowledge into useful learning, not a longer transcription. Explain concept, meaning, application conditions and learner takeaway where the evidence supports them. Scale depth to the available knowledge; do not pad with repeated paragraphs or invent missing detail.",
        "You may synthesize questions, comparisons, practice instructions and explanations from the approved evidence even if the source contains no FAQ or exercise. Every answer, ordering relationship, definition and diagram edge must be supported by that evidence. A pedagogical inference is not a new source fact. Never create canonical IDs or broaden evidence ownership.",
        "For each selected FAQ, create 2-3 useful, distinct clarification questions about distinctions, conditions, decisions or cautions taught here. No verbatim paragraph copies or generic filler. Put FAQ last. Do not force unsupported exceptions or explanations of why when the source states only what.",
        "A problem must address its mapped objectives, not just an easy neighbouring definition. For procedural/application objectives use a source-supported decision or clearly labelled instructional calculation using the source rule. Do not present a hypothetical example as an observed source event. Use plausible distinct distractors, avoid absurd giveaways and do not always place the correct option first.",
        "Sortable uses only explicit source order. Diagram distinguishes membership/hierarchy from temporal/causal edges; co-occurrence alone does not establish causation. Crossword uses only source-defined terms with accurate clues; spelling normalization must not change the term's meaning.",
        "Do not ask learners to inspect an absent image, video or unspecified Scenario 1-4. Use the available textual scenario instead where sufficient; otherwise state the local missing evidence briefly without inventing its contents. Missing optional illustrative material must not turn usable teaching into an empty lesson.",
        "Source contradictions must not become unambiguous quiz answers: label the specific uncertainty for review and teach the undisputed material. Never silently choose a disputed threshold or invent a resolution.",
        "Do not teach document footers, promotional contacts, THANK YOU slides or page furniture as learning objectives or assessment content. Preserve mandated evidence ownership/coverage; when such source material is assigned, distinguish it as a brief non-instructional source note, not a lesson or activity. Never silently delete assigned fact IDs.",
        "Keep the approved component instances and all payload/security constraints unchanged. Only generate types selected by the server. This policy does not authorize skipping mandatory content, returning invalid payloads or fabricating assets.",
    ])


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
