"""The W3/W4 module prompt: lesson design of one module."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Final

from app.idm.policy import IDM_MEDIA_CONTEXT_MIN_WORDS, SUGGESTED_CRITERIA_LABEL_EN, SUGGESTED_CRITERIA_LABEL_VI
from app.idm.prompts_common import (
    Locale,
    _json,
    fact_lines,
    locale_name,
    preamble,
)
from app.prompt_safety import untrusted_block

# QC feedback 2026-10-21 (A4): the context_description of a media brief is a designer brief.
_MEDIA_BRIEF_RULE: Final = f"""\
   media_brief: null unless a short video or static infographic clearly helps; then a concrete brief with at most
   6 content_points. Its context_description is the brief a designer draws from, at least
   {IDM_MEDIA_CONTEXT_MIN_WORDS} words: the layout (how the image or the video is divided: panels, columns, rows or
   scenes, and the reading order), then one element per content point named by its number ("Khung 1" / "Panel 1",
   "Cảnh 1" / "Scene 1") with the exact text and numbers shown on it, taken from that point; arrows or connectors and
   their direction only where the points are steps or a cause and its effect. Never a mood or a setting alone ("Một
   bảng kiểm treo tại phòng họp")."""


def _criteria_rule(locale: Locale) -> str:
    """Step 2 of the module prompt. A5 (owner decision 2026-10-22): a practice is never held for an SME; one whose
    facts state no criteria gets suggested criteria drawn from what the lesson teaches, labelled as such."""

    label = SUGGESTED_CRITERIA_LABEL_VI if locale == "vi" else SUGGESTED_CRITERIA_LABEL_EN
    return (f"""2. criteria_fact_keys: the source facts that decide what a correct answer is; non-empty; at most 24 """
            f"""(when more
   apply, keep the statements of the Must Do block that decide right and wrong, not its labels or headings); only
   facts of this lesson's blocks. Never hold a practice: hold is always false and hold_question null. When the facts
   state no criteria that decide a correct answer, still design the practice as its own practice unit: put in
   criteria_fact_keys the facts of this lesson's blocks the practice applies (what the lesson teaches about it),
   start feedback_focus.criterion with "{label}: " and state there criteria drawn only from that taught content
   (never a rule, number or threshold the facts do not state); the practice is a WORKSHEET checked by a problem when
   the worksheet is allowed, else a problem, and like any practice it writes out the data its task works on (a
   classroom exercise of the slides becomes one the learner can do alone). Contexts, characters and example
   situations may be drafted by you
   (scenario_origin "ai_drafted") but must not introduce new rules, numbers or exceptions.""")


def module_prompt(
    locale: Locale,
    *,
    module_plan: dict[str, Any],
    lesson_plans: list[dict[str, Any]],
    audience: str,
    objectives: list[dict[str, Any]],
    must_dos: list[dict[str, Any]],
    course_blocks: list[dict[str, Any]],
    facts: Iterable[tuple[str, str, Sequence[str] | None]],
    allowed_components: Sequence[str],
    scenario_chat: bool = False,
    limits: str = "",
    purpose_rules: str = "",
    course_purpose: dict[str, Any] | None = None,
    fidelity_rules: str = "",
    organization_context: str = "",
) -> str:
    """``purpose_rules``: ``app.idm.purpose.module_purpose_rules`` (empty for profiles that keep the general
    rules); ``course_purpose``: the design's purpose as plan data; ``fidelity_rules``:
    ``app.idm.fidelity.module_fidelity_rules`` (empty unless the content mode is explain); ``organization_context``:
    ``app.idm.org_context.design_block`` (empty without an organization context)."""

    scenario_clause = (
        "; a multi-turn conversation where each reply changes the outcome -> la_scenario_chat" if scenario_chat else ""
    )
    # QC run 8de1c76b (Q1): the provider schema carries no bounds, so they are stated here.
    limits_block = (f"10. LIMITS of the answer (characters of text, items of lists); never exceed them:\n{limits}\n"
                    if limits else "")
    return preamble(locale) + f"""
TASK: For ONE module, design each listed lesson (Week 3 Instructional Treatment + Week 4 lesson layout).
Return the lessons in exactly the order and with exactly the lesson_key values of LESSONS.
{purpose_rules}{fidelity_rules}{organization_context}For each lesson:
1. PRACTICE FIRST. Write 1-3 practice tasks as one sentence
   "Cho [bối cảnh/đầu vào], người học [hành động] để [kết quả/quyết định]"
   ("Given [context/input], the learner [action] to produce/decide [result]"), plus context_input,
   learner_action, result and Bloom level. Ids pt_1, pt_2, pt_3 inside each lesson. context_input is what the
   learner works on; when it is data to read (a list, an interview, requests, cases), keep it short (at most 6
   entries): the practice unit writes it out in full, never "the team's own list" or data the learner does not have.
   The practice must use the same kind of action as the Must Do (classify, sequence, decide, apply a rule, choose a
   message ...). Do not turn everything into recall questions.
   Must Do of kind "do" that produces an output (fill in a form or canvas, draft a plan, write a commitment, map
   resources, redesign a step; not classify, identify or choose), and a Must Do marked "decide" whose verb
   produces one (sign a commitment, draw up a list): the main practice lets the learner DO it. Use
   la_sortable when the action is a procedure whose order the source states; otherwise use a WORKSHEET, because no
   allowed component records a free-text answer: in the practice unit put an html component with role "practice"
   and the practice_id (the learner completes the task on their own copy: task and input, the template to fill in,
   a short worked example, a self-check list built only from criteria_fact_keys), followed in the same unit by a
   problem with role "practice" and the same practice_id in which the learner judges a sample answer against those
   criteria (single choice). Must Do of kind "decide" and recognition of Must Know keep problem (scenario or
   knowledge check).
{_criteria_rule(locale)}
3. SUPPORTING INFO - the minimum needed for the practice: explain_concept, explain_principle (criteria/rules),
   example, non_example, worked_example, demonstration. Ask "if this were removed, could the learner still do the
   practice?" - if yes, leave it out or leave it for the Job Aid.
4. FEEDBACK FOCUS: criterion (what decides right/wrong), rationale (why the better choice is better),
   improvement (what to fix). Never just "Correct/Incorrect".
5. FORMAT last, simplest that works, ONLY from ALLOWED_COMPONENTS:
   short principle -> html; compare concepts -> html table; relationships/structure/process map -> la_diagram
   (only when the facts state the relations); annotated or worked example -> html; a decision or policy, a condition
   or exception, or a common mistake the facts state -> la_faq (only then: the server removes an la_faq whose unit
   has none of these; the wrong options of the unit's own question are no reason on their own, and a lesson keeps at
   most one la_faq that rests only on a likely confusion; never an FAQ that only recalls facts); ordering steps ->
   la_sortable (only with source order); terminology to memorise ->
   la_crossword (at least 3 source definitions with short terms); knowledge check of a Must Know -> problem
   (single choice); decision in a short situation -> problem as a scenario question{scenario_clause}; producing
   an output (Must Do of kind "do") -> worksheet: html with role practice + problem that checks a sample answer.
6. LAYOUT into units (each unit is a mini-cycle of at most 4 components; at most one component of each type per
   unit; html first; la_faq last): context (short, under 20% of the lesson) -> Must Know explain/show -> example ->
   practice -> feedback -> apply/next step. Typical units: "context_explain" (+ knowledge check), "example",
   "practice_feedback" (the main practice), "summary_apply" (optional). Merge units when the lesson is short.
   Units split a lesson by WHOLE blocks: every block of the lesson belongs to exactly one unit, so a lesson with
   ONE block has exactly ONE unit that both teaches and practises (html first, then the practice components,
   la_faq last); never repeat a block in a second unit to practise it. When the practice of such a unit is a
   worksheet, the worksheet html is the unit's only html (its support_items carry what the practice needs);
   never a second, teaching html in the same unit. Every component lists the blocks it uses
   (from its unit) and every block of a unit is used by at least one of its components. The main practice sits in a unit
   that contains at least one block of its Must Do. Facts that decide a practice must be taught in the same or an
   earlier unit: every block a problem uses is also used by the html placed before it in its unit (a question
   tests what that html teaches and is never the only component that presents a block). Never more than 5
   explanation components in a row without a question or practice.
   A practice component has role "practice" and practice_id set; other components have practice_id null. Only
   problem, la_sortable, la_crossword and the worksheet html may have role "practice"; a worksheet html always has
   the problem that checks it in the same unit.
   component_index is the 1-based position inside the unit; unit_index is 1-based inside the lesson.
   Unit titles name what the unit actually teaches (never "Tổng quan 5 chuyển dịch" over a unit about something
   else). Lesson, unit and component titles are in sentence case (a capital for the first word only; acronyms,
   codes and names as the facts write them). When an objective this module serves, or COURSE_PURPOSE, names an
   enumerated framework whose items this module's blocks hold ("5 chuyển dịch", "6 trụ cột", "3 giai đoạn"), the
   first lesson that serves it opens with ONE orientation unit
   (segment context_explain, an html component) whose title names the framework with its number
   ("Bản đồ 5 chuyển dịch") and whose html lists every item of it; later units and lessons teach the items one by
   one.
7. For every component write the storyboard development notes in {locale_name(locale)} (author_review):
   purpose (which Must Do / practice / support it serves - required), example_scenario (the example or situation
   used; start with "[AI soạn — tác giả kiểm tra lại]" / "[AI-drafted — author to check]" when you drafted it),
   visual_asset (suggested visual or null), user_behavior_navigation (what the learner does: read, choose, drag,
   retry once, continue, open the Job Aid).
8. Lesson fields: objective = the main Must Do as an observable action; learning_objectives = the objective
   statements this lesson serves; assessment = feedback criterion + where the practice sits (after Must Know /
   end of lesson / end of module); notes = Bloom level, estimated blocks and minutes, ordering rationale (no
   question for a subject-matter expert).
{_MEDIA_BRIEF_RULE}
9. Lessons of kind job_aid: 1-3 units of segment "job_aid" with an html checklist / lookup table / decision tree
   (and optionally la_faq); no practice tasks. Contact lists: names and extensions only, no emails or URLs.
{limits_block}
{untrusted_block("PLAN_CONTEXT", chr(10).join([
    "COURSE_PURPOSE=" + _json(course_purpose),
    "MODULE=" + _json(module_plan),
    "LESSONS=" + _json(lesson_plans),
    "TARGET_AUDIENCE=" + _json(audience),
    "OBJECTIVES=" + _json(objectives),
    "MUST_DOS=" + _json(must_dos),
]))}
{untrusted_block("BLOCK_CATALOG", "COURSE_BLOCKS=" + _json(course_blocks))}
{untrusted_block("SOURCE_FACTS", fact_lines(facts))}
ALLOWED_COMPONENTS={_json(list(allowed_components))}
"""
