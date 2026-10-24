"""The evidence-bounded teaching policy every lesson writer prompt starts with.

Moved verbatim from ``app.lesson_prompt_policy`` (refactor wave 3b): the staged unit provider schemas put it in the
content instruction, and strict code does not import the legacy modules.
"""

from __future__ import annotations

LESSON_INSTRUCTIONAL_POLICY_VERSION = "evidence-teaching-6"


def lesson_instructional_quality_policy() -> str:
    """Pedagogical synthesis is allowed; domain claims and ownership are not."""
    return "\n".join([
        f"EVIDENCE-BOUNDED TEACHING POLICY: {LESSON_INSTRUCTIONAL_POLICY_VERSION}.",
        "Turn available knowledge into useful learning, not a longer transcription. Explain concept, meaning, "
        "application conditions and learner takeaway where the evidence supports them. Scale depth to the available "
        "knowledge; do not pad with repeated paragraphs or invent missing detail.",
        "You may synthesize questions, comparisons, practice instructions and explanations from the approved evidence "
        "even if the source contains no FAQ or exercise. Every answer, ordering relationship, definition and diagram "
        "edge must be supported by that evidence. A pedagogical inference is not a new source fact. Never create "
        "canonical IDs or broaden evidence ownership.",
        "Preserve source conditions, negations, quantities, units and technical distinctions in both teaching and "
        "answers. Absence from the source is NOT an exemption, prohibition or permission. Do not infer 'not required',"
        " 'safe', 'always' or 'only' from an omitted case. In particular, never invent an exemption to a safety "
        "procedure to manufacture a quiz distractor or correct answer.",
        "Do not concatenate independent source sections into a new mandatory procedure. Each asserted temporal, causal"
        " or prerequisite relationship needs evidence; separate topics may be compared without inventing an order. "
        "Hypothetical practice inputs must be labelled hypothetical, use an evidenced rule and must not introduce new "
        "domain thresholds or exceptions.",
        "For each selected FAQ, create 2-3 useful, distinct clarification questions about distinctions, conditions, "
        "decisions or cautions taught here. No verbatim paragraph copies or generic filler. Put FAQ last. Do not force"
        " unsupported exceptions or explanations of why when the source states only what.",
        "A problem must address its mapped objectives, not just an easy neighbouring definition. For "
        "procedural/application objectives use a source-supported decision or clearly labelled instructional "
        "calculation using the source rule. Do not present a hypothetical example as an observed source event. Use "
        "plausible distinct distractors, avoid absurd giveaways and do not always place the correct option first.",
        "Match the learner action, not merely the topic: calculation objectives require applying the evidenced "
        "formula; analysis objectives require distinguishing relevant conditions; identification objectives may use "
        "recall. Feedback must explain why the answer follows from what was taught. When evidence cannot support the "
        "required action, disclose the local limitation rather than inventing a rule or silently substituting a "
        "vocabulary question.",
        "For calculation practice, teach one worked example using the source formula and explicitly hypothetical "
        "inputs, then ask the learner to compute with another set of inputs if an approved problem is available. If no"
        " problem is selected, include a short self-check and explained answer inside the existing HTML; never add a "
        "component outside the plan. Avoid disputed source boundary values. Analysis and selection practice must "
        "require a justified choice from the available evidence, not merely repeat a definition.",
        "Before finalizing, check each assigned fact against an actual teaching paragraph, step or table row, not "
        "merely its ID in coverage metadata. Preserve essential qualifiers and list members. Keep all provenance in "
        "structured metadata only. Do not place internal fact IDs, component/block IDs, source filenames, "
        "page/slide/chunk locators, citations, or phrases such as 'theo tài liệu nguồn' in learner-facing content. "
        "Give activities accurate names; sortable is ordering practice, not crossword.",
        "Sortable uses only explicit source order. Diagram distinguishes membership/hierarchy from temporal/causal "
        "edges; co-occurrence alone does not establish causation. Crossword uses only source-defined terms with "
        "accurate clues; spelling normalization must not change the term's meaning.",
        "Do not ask learners to inspect an absent image, video or unspecified Scenario 1-4. Use the available textual "
        "scenario instead where sufficient; otherwise state the local missing evidence briefly without inventing its "
        "contents. Missing optional illustrative material must not turn usable teaching into an empty lesson.",
        "Source contradictions must not become unambiguous quiz answers: label the specific uncertainty for review and"
        " teach the undisputed material. Never silently choose a disputed threshold or invent a resolution.",
        "Do not teach document footers, promotional contacts, THANK YOU slides or page furniture as learning "
        "objectives or assessment content. Preserve mandated evidence ownership/coverage; when such source material is"
        " assigned, distinguish it as a brief non-instructional source note, not a lesson or activity. Never silently "
        "delete assigned fact IDs.",
        "Keep the approved component instances and all payload/security constraints unchanged. Only generate types "
        "selected by the server. This policy does not authorize skipping mandatory content, returning invalid payloads"
        " or fabricating assets.",
    ])
