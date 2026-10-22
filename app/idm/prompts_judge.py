"""The W6 judge prompt: review of one generated unit against its plan."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from app.idm.prompts_common import (
    Locale,
    _json,
    learner_fact_lines,
    preamble,
)
from app.prompt_safety import untrusted_block


def judge_prompt(
    locale: Locale,
    *,
    plan_summary: dict[str, Any],
    facts: Iterable[tuple[str, str, Sequence[str] | None]],
    unit_content: list[dict[str, Any]],
    organization_context: str = "",
) -> str:
    """``organization_context``: ``app.idm.org_context.judge_block`` (empty without an organization context)."""

    return preamble(locale) + f"""
TASK: Review ONE generated unit against its approved plan as a senior instructional-design reviewer. Do not
rewrite content. For each criterion return pass | minor | major | critical (or not_applicable, only as allowed
below), the component index (0-based) when the issue is in one component, and a witness of at most 300
characters quoting the unit. Judge only this unit: PLAN.practices lists the practice of this unit only, and
PLAN.has_practice_slot says whether this unit contains a practice or question slot. Practice that sits in
another unit of the lesson is never a finding here.
 Q1_support_sufficient: a learner could do this unit's practice using only what this lesson taught (a unit
   without a practice slot: its teaching achieves PLAN.purpose).
 Q2_not_copied: rewritten for the learner, not pasted source (exact rules/definitions are allowed).
 Q3_practice_complete: context, task, input, one correct answer and feedback are present and consistent; a
   question whose text stops before it asks anything is critical. A worksheet (html slot with role practice) has no
   single answer: check its task, template, worked example and self-check criteria, and the problem that checks it;
   a worked example that breaks one of its own worksheet's self-check criteria (a KPI without a number when the
   criteria ask for a measurable one) is major.
 Q4_feedback_teaches: feedback names the criterion and explains why each option is right or wrong; the
   explanation of EACH option agrees with that option's own text and with the answer key (an explanation that
   calls a correct-by-the-criterion option wrong, or a wrong option right, is major). Its first line states the
   principle the unit taught and each option has its own line with the reason; a reason that only repeats the option,
   praises it with empty words ("dũng cảm", "tuyệt đối", "Say No") or restates several definitions instead of the
   reason is major; feedback that is correct but long or wordy is minor.
 Q5_grounded_criteria: the correct answer follows from SOURCE_FACTS; no invented rule, threshold or exception;
   a quotation or callout states only what SOURCE_FACTS say. Exactly one option is correct by the criteria this
   unit teaches: a question whose case also meets the criteria the unit's tables, lists or FAQ answers give for
   another option, or whose key contradicts an FAQ answer of this unit, is major; so is a correct option that
   restates the example, quotation or template shown right before the question, or strings together the items of a
   list shown before it, and a case that repeats the worked example so its answer is the example's answer. Check EVERY
   option against the definition, formula or criterion the unit teaches, one by one: a distractor that also meets it
   (for a formula or framework of required parts: it has every part, as the key does) or a key that misses a part
   makes more than one option defensible: major, with the option named in the witness.
 Q6_alignment: the practice matches the Must Do action type and Bloom level in the plan.
 Q3, Q4 and Q6 grade this unit's own practice: when PLAN.has_practice_slot is false return not_applicable for
   them. not_applicable is never allowed for any other criterion except Q11, Q12 and Q13 as stated below.
 Q7_cognitive_load: no long theory runs; context under 20%; detail matches detail_level.
 Q8_language: clear, suited to the audience, terms explained, no internal IDs or citations.
 Q9_traceability: every component serves the stated Must Do, practice or support.
 Q10_title_matches: PLAN.unit_title, the slot titles and the section headings name what the content actually
   teaches; a title that promises a topic or a list ("Tổng quan 5 chuyển dịch") the content does not present
   is major; an FAQ title that does not match its questions is major.
 Q11_purpose_alignment: the unit serves PLAN.course_purpose (what the course is for and for whom): content that
   pulls the unit towards another purpose (an onboarding unit that trains selling or consulting, a compliance unit
   that loosens a rule) is major. When PLAN.course_purpose is null return not_applicable for Q11.
 Q12_explained: when PLAN.fidelity_mode is "explain", every teaching html slot explains its content for the
   learner: the main idea in plain words, why it matters to them and a short example or situation (an example
   labelled "Ví dụ minh hoạ" or "Illustrative example" may be drafted, but it adds no number, rule or name); a
   teaching slot that only restates the source is major. For any other PLAN.fidelity_mode return not_applicable
   for Q12.
 Q13_faq_value: when PLAN.has_faq_slot is true, the la_faq slot adds value: its questions ask why, what happens if,
   what goes wrong or how to apply (at most one plain recall question), and each answer explains from the facts
   instead of restating a source sentence or the html above it; an FAQ that only restates the content is major; an
   answer that adds a number, rule or name the facts do not state is major. When PLAN.has_faq_slot is false return
   not_applicable for Q13.
Return one finding per criterion (Q1-Q13). verdict = pass (no major/critical) | review_required (any major) |
reject (any critical); not_applicable never counts.

{organization_context}{untrusted_block("PLAN_CONTEXT", "PLAN=" + _json(plan_summary))}
{untrusted_block("SOURCE_FACTS", learner_fact_lines(facts))}
{untrusted_block("UNIT_CONTENT", _json(unit_content))}
"""
