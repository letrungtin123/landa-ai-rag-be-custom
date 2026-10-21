"""Course-level IDM prompts: W1 section map, W1 reduce, W2 blueprint and W4 course layout."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from app.idm.fidelity import DEFAULT_MODE, w2_treatment_rules
from app.idm.prompts_common import (
    Locale,
    _json,
    fact_lines,
    locale_name,
    preamble,
)
from app.idm.purpose import DEFAULT_BACKGROUND_RULE, DEFAULT_SUMMARY_RULE, DEFAULT_TITLE_RULE
from app.prompt_safety import untrusted_block


def w1_section_prompt(
    locale: Locale,
    *,
    project_context: dict[str, Any],
    section_id: str,
    section_title_path: str,
    facts: Iterable[tuple[str, str, Sequence[str] | None]],
) -> str:
    return preamble(locale) + f"""
TASK: Week 1 - build the SME Content Map for ONE section of the source.

1. Scan first. Use headings, tables, numbered steps, rule words ("luôn", "không được", "bắt buộc",
   "trừ trường hợp", "must", "never", "always", "unless"), examples/cases, emphasised and repeated content
   (see the signal flags: H heading, T table row, S numbered step, R rule word, E example/case, M emphasis,
   D duplicate candidate, V visual reference) to see the structure before grouping.
2. Group facts into Content Blocks. 1 Content Block = 1 clear purpose = answers one question OR supports one
   action OR one decision.
   - intent: "know" (what the learner needs to know), "do" (what the learner must perform),
     "decide" (a decision with criteria/exceptions).
   - support_role (only for supporting blocks, otherwise null): example | common_mistake | checklist | job_aid |
     practice_material.
   - content_kind and how to chunk it:
     concept -> one concept or a group of concepts serving one question;
     procedure -> one stage / group of steps with the same purpose (NOT one block per step);
     skill -> one observable behaviour; decision -> the decision + its criteria + exceptions;
     policy -> rules, prohibitions, obligations, exceptions; mindset -> message, expected behaviour, consequence;
     system -> one task on a system; case -> context, decision, action, result; reference -> one lookup need.
   - MERGE facts that answer the same question. SPLIT when parts would need different presentation, practice,
     feedback, format or priority. Too big: the name joins two purposes with "và"/"and"; it serves unrelated
     objectives; one part needs explanation and another a scenario; it mixes must-know with lookup material;
     it mixes concept + procedure + assessment. Too small: a single sentence or bullet with no standalone meaning,
     or several tiny blocks answering the same question.
   - name: a specific question or action phrase a reviewer understands immediately
     (good: "Khi nào phải chuyển khiếu nại cho quản lý?"; bad: "Giới thiệu", "Thông tin chung", "Nội dung quy trình").
   - summary: at most 300 characters, neutral, only what the facts say.
3. Every fact_key in this section must be assigned exactly once: to one block (fact_keys, in source order) or to
   noise with a reason: page_furniture | contact_or_promo | toc_or_title_only | duplicate_verbatim | unreadable.
   Teaching value is NOT decided here; only true non-content goes to noise. Background or history text is still
   a block (Week 2 decides to remove it). A heading line belongs to the block it introduces.
   A formula or definition of a term ("X = A + B", "X là ...", "X is ...") is its own block (intent know,
   content_kind concept, named "X là gì?" / "What is X?"), never merged with the history, origin or background
   narrative around it, even on the same page: Week 2 may drop the history but must keep the definition.
4. Record issues only with concrete evidence and the fact_keys involved: unclear | duplicate | conflict |
   too_general | too_detailed | outdated.
5. Record content gaps that will matter for examples, practice, feedback or assessment: missing_example |
   missing_exception | missing_criteria | missing_step | missing_sample_output | missing_common_mistake |
   missing_feedback_basis | missing_unit_variation.
   Look explicitly for common mistakes, misconceptions and exceptions. When the source states one (a "lỗi thường
   gặp" / "common mistakes" part, a wrong action shown as an example, "lưu ý", "cảnh báo", "trừ khi", "không áp dụng
   khi", "unless", "except"), keep it: mistakes and misconceptions as a block with support_role common_mistake, an
   exception inside the decision or policy block it limits. When a decision, policy or procedure block has none in
   the source, record the gap missing_common_mistake (and missing_exception when the rule clearly needs one); never
   write a mistake or exception the source does not state.
6. Write SME questions that point at the exact rule, example or gap
   (good: "Ngoài an toàn và pháp lý, giá trị giao dịch có phải tiêu chí bắt buộc escalate không?";
   bad: "Phần nào quan trọng?").
7. Do NOT decide keep/remove, do NOT choose formats, do NOT write course content.
Use local ids b1, b2, ... for blocks.

{untrusted_block("PLAN_CONTEXT", "PROJECT_CONTEXT=" + _json(project_context))}
SECTION={section_id}
{untrusted_block("SOURCE_FACTS", "SECTION_TITLE=" + section_title_path + chr(10) + fact_lines(facts))}
"""


def w1_reduce_prompt(locale: Locale, *, project_context: dict[str, Any], catalog: list[dict[str, Any]],
                     purpose_step: str = "", organization_context: str = "") -> str:
    """``purpose_step`` (``app.idm.purpose.w1_purpose_step``): the author intent block and the purpose profiles;
    ``organization_context``: ``app.idm.org_context.design_block`` (empty without an organization context)."""

    return preamble(locale) + f"""
TASK: Consolidate the section Content Maps into ONE course-level SME Content Map, decide the course purpose, then
propose the learning objectives and the Must Do list for that purpose.
{purpose_step}{organization_context}
1. Merge duplicate blocks across sections (same purpose): keep one block_id, list the merged ids, give a reason.
   Flag cross-section conflicts (blocks that give different instructions) with a specific SME question.
2. Target audience: if PROJECT_CONTEXT.target_audience is null, propose one that fits the course purpose (role,
   experience level, work context, constraints) with origin "ai_proposed"; otherwise copy it with origin "client".
3. Learning objectives: if PROJECT_CONTEXT.learning_objectives is not empty, copy them verbatim in order with
   origin "client". Otherwise propose 3-8 performance-based, measurable objectives, one action each:
   "{{learner}} có thể {{observable verb}} {{object}} {{condition or criterion when the source provides it}}".
   Never use unmeasurable main verbs (hiểu, biết, nắm được, nắm rõ, làm quen, understand, know, learn about,
   be aware of) beyond the forms the OBJECTIVE POLICY of the course purpose allows. Give each a Bloom level
   (remember | understand | apply | analyze | evaluate | create); prefer apply/analyze/evaluate when the source
   describes work actions or decisions, unless the OBJECTIVE POLICY says otherwise. Objectives must be achievable
   with the source content. Use ids lo_1, lo_2, ... in order.
4. Must Do (1-5 per objective): concrete actions or decisions performed at work, as the MUST DO POLICY of the
   course purpose describes them ("Phân loại khiếu nại theo mức độ", "Quyết định tự xử lý hay escalate"). A Must
   Do is not a topic. kind: do | decide. Ids md_1, md_2, ...
   Every objective needs at least one Must Do. Every explicit action item of the source ("HÀNH ĐỘNG CEO",
   "Hành động", "Việc cần làm", "Action", "Next step"; blocks listed with action_items in BLOCK_CATALOG) is a
   Must Do candidate: make it a Must Do of the objective it serves, or name it inside a Must Do statement;
   link its block to that objective with relation direct. When an objective names an enumerated framework
   ("5 chuyển dịch"), the action items of all its parts are candidates, not only some of them.
5. lo_links: list a link ONLY when a block serves an objective, with relation direct (needed to perform it) or
   supporting (helps). Do not list context, unrelated or unknown links: a block without a link is treated as
   unrelated and becomes a Remove candidate later - do not delete it.

{untrusted_block("PLAN_CONTEXT", "PROJECT_CONTEXT=" + _json(project_context))}
{untrusted_block("BLOCK_CATALOG", _json(catalog))}
"""


def w2_blueprint_prompt(
    locale: Locale,
    *,
    project_context: dict[str, Any],
    objectives: list[dict[str, Any]],
    must_dos: list[dict[str, Any]],
    catalog: list[dict[str, Any]],
    purpose_rules: str = "",
    background_rule: str = DEFAULT_BACKGROUND_RULE,
    treatment_rules: str = "",
    organization_context: str = "",
) -> str:
    """``purpose_rules``/``background_rule``: ``app.idm.purpose.w2_purpose_rules`` (the author intent and what the
    course purpose keeps); the defaults are the job-performance rules of the general profile. ``treatment_rules``:
    ``app.idm.fidelity.w2_treatment_rules`` (every treatment defined, then the content mode; default structured).
    ``organization_context``: ``app.idm.org_context.design_block`` (empty without an organization context)."""

    treatments = treatment_rules or w2_treatment_rules(DEFAULT_MODE)

    return preamble(locale) + f"""
TASK: Week 2 - decide, for EVERY block in BLOCK_CATALOG, what happens to it. Return exactly one row per block.
{purpose_rules}{organization_context}Decision flow (apply in order):
  a. Does the block serve an objective or a Must Do? No -> remove (useful but not needed, unrelated, outdated,
     duplicated, wrong audience, too specialised).
  b. Must the learner REMEMBER it to perform the Must Do correctly? No -> reference with treatment
     move_to_reference (look up when needed) or convert_to_job_aid (used often while working: checklist,
     decision tree, code table, form guide, contact list, response templates).
  c. Yes -> keep it in the course with the detail the learner needs to act correctly; leave out detail the Must
     Do does not need.
  d. Unclear, conflicting or outdated content that affects correctness, or a decision block that lacks the
     criteria needed to judge practice -> hold = true with hold_reason and one specific sme_question.
classification: must_do | must_know | reference | remove (never nice_to_know: the course keeps no list of left-out
content)
placement: course (must_do, must_know) | reference_job_aid (reference) | excluded (remove)
{treatments}detail_level: one sentence on what to keep ("Định nghĩa ngắn + dấu hiệu phân biệt 3 cấp độ";
"Chỉ 6 nhóm bước chính; ngoại lệ hiếm chuyển sang Job Aid").
Rules: repeated content is not automatically important; SME emphasis is a signal to check, not a decision;
full legal text, code lists, contact lists, forms and rare exceptions -> reference/job aid; {background_rule};
"remove" is not a judgement of quality. Never classify as
remove a block that defines or gives the formula of a term that appears in the course title
(COURSE_NAME in AUTHOR_COURSE_INTENT or a source document name), an objective or a Must Do: it is must_know
(or reference), with a detail_level that keeps the definition and drops the history around it.
must_know rows list the must_do_ids they enable; must_do rows list the Must Do they represent; lo_id is the
objective the block serves (null when none). Only must_do/must_know rows may hold.
combine: set combine_into (another block id with the same classification). separate: separate_into =
[{{name, intent, fact_keys}}] that partition the block's facts exactly.
List in blocked_must_do_ids every Must Do that has no course block left that is not on hold.

{untrusted_block("PLAN_CONTEXT", "PROJECT_CONTEXT=" + _json(project_context) + chr(10)
                 + "OBJECTIVES=" + _json(objectives) + chr(10) + "MUST_DOS=" + _json(must_dos))}
{untrusted_block("BLOCK_CATALOG", _json(catalog))}
"""


def w4_course_prompt(
    locale: Locale,
    *,
    project_context: dict[str, Any],
    audience: dict[str, Any],
    objectives: list[dict[str, Any]],
    must_dos: list[dict[str, Any]],
    course_blocks: list[dict[str, Any]],
    reference_blocks: list[dict[str, Any]],
    blocked_must_dos: list[str],
    duration_target_minutes: int | None,
    purpose_rules: str = "",
    course_purpose: dict[str, Any] | None = None,
    titles: str = DEFAULT_TITLE_RULE,
    summary: str = DEFAULT_SUMMARY_RULE,
    organization_context: str = "",
) -> str:
    """``purpose_rules``, ``titles`` and ``summary`` come from ``app.idm.purpose`` (server-owned); ``course_purpose``
    is plan data (profile and the W1-reduce summary); ``organization_context``: ``app.idm.org_context.design_block``
    (empty without an organization context)."""

    rules = purpose_rules + organization_context
    return preamble(locale) + f"""
TASK: Week 4 (course level) - group Must Dos into Modules and Lessons and order them.
{rules}- Module = lessons serving one larger performance. Lesson = helps the learner do ONE main Must Do; a
  closely related second Must Do may share the lesson only when it uses the same practice and the same support at
  the same moment. One objective may need several lessons. Each lesson lists the course blocks it uses; every course
  block and every reference block belongs to exactly one lesson; the content_chars of one lesson's blocks add up
  to at most 90000. Every Must Do that is not blocked is the primary
  or secondary Must Do of exactly one learning lesson.
- Order by prerequisite first, then by the real work process. Shared foundations first. When an objective names
  an enumerated framework ("5 chuyển dịch", "6 trụ cột"), the block that presents the whole framework belongs to
  the first lesson that serves that objective, so that lesson can open with an overview of all its items.
- Linear path only. Do not invent routing, pre-tests or role branches.
- Estimate each lesson (screens, minutes) from Must Do complexity, risk of error and practice intensity - not from
  source length. If DURATION_TARGET_MINUTES is given, fit it: cut detail the Must Dos do not need first, move
  lookup content to the Job Aid, keep core practice; never remove the practice of a high-risk Must Do.
- Reference/Job Aid blocks go into lessons of kind "job_aid" (primary_must_do_id null, no course blocks): at the
  end of the module that uses them, or one final job_aid lesson at the end of the course when several modules use
  them. Learning lessons never contain reference blocks.
- Titles in {locale_name(locale)}: {titles} Write every title in sentence case: a capital for the first word (and
  after a colon) only; acronyms, codes and names keep the case the facts give them ("BiC", "ERA5.0", "Made-in-World").
- Also write: course_title (a learner-facing name that reflects AUTHOR_COURSE_INTENT and the course purpose: keep
  the author's course name when it already names the course well; never a name that changes the purpose),
  course_summary ({summary}), assessment_strategy (where knowledge checks, practices and end-of-module practices
  sit), prerequisites, pattern_sections (see COURSE PATTERN; [] when there is none).
Keys: mod_01, mod_02, ... ; lesson keys lsn_001, lsn_002, ... across the whole course in order.

{untrusted_block("PLAN_CONTEXT", chr(10).join([
    "PROJECT_CONTEXT=" + _json(project_context),
    "COURSE_PURPOSE=" + _json(course_purpose),
    "TARGET_AUDIENCE=" + _json(audience),
    "OBJECTIVES=" + _json(objectives),
    "MUST_DOS=" + _json(must_dos),
    "BLOCKED_MUST_DOS=" + _json(blocked_must_dos),
    "DURATION_TARGET_MINUTES=" + _json(duration_target_minutes),
]))}
{untrusted_block("BLOCK_CATALOG", "COURSE_BLOCKS=" + _json(course_blocks) + chr(10)
                 + "REFERENCE_BLOCKS=" + _json(reference_blocks))}
"""
