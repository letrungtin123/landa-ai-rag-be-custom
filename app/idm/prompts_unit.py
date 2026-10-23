"""The W5 unit writer prompt, its HTML slot rules and the repair rule lines."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from typing import Any, Final

from app.idm.fidelity import DEFAULT_MODE, writer_fidelity_rules
from app.idm.html_rules import (
    DENSITY_CODE,
    GROUP_LIMITS,
    MAX_BLOCKS_PER_SECTION,
    MAX_HEADING_CHARS,
    MAX_ROW_LABEL_CHARS,
    MAX_ROW_VALUE_CHARS,
    MAX_ROWS,
    MAX_SECTIONS,
    HtmlRuleViolation,
)
from app.idm.policy import (
    IDM_MCQ_EXPLANATION_MAX_CHARS,
    IDM_MCQ_EXPLANATION_OPTION_MAX_WORDS,
    IDM_MCQ_EXPLANATION_PRINCIPLE_MAX_WORDS,
    IDM_MCQ_OPTION_COUNT,
    IDM_MCQ_STEM_ITEM_MAX_WORDS,
    IDM_MCQ_STEM_MAX_ITEMS,
    IDM_MCQ_STEM_PROSE_MAX_WORDS,
)
from app.idm.prompts_common import (
    Locale,
    _json,
    learner_fact_lines,
    locale_name,
    preamble,
    repair_suffix,
)
from app.prompt_safety import untrusted_block

_PURPOSE_LINE: Final = ("- Keep the unit on COURSE_PURPOSE (what the course is for and for whom): choose the wording, "
                        "emphasis and examples that serve it; it never adds facts.\n")


# The text of a single-choice slot (QC feedback 2026-10-21, A1/A2; contract K1): a stem in lines and an explanation of
# one principle line and one line per option. The server lays out what it safely can and checks the rest
# (``app.idm.quiz_text``); the limits are those of ``app.idm.policy``.
_QUIZ_TEXT_RULE: Final = f"""\
- QUIZ TEXT of a single-choice slot: plain text whose only layout is the line feed ("\\n" inside the JSON string),
  never markdown or HTML.
  question: the whole case and its question: one or two short context sentences and the question sentence, together
  at most {IDM_MCQ_STEM_PROSE_MAX_WORDS} words; when the case lists several items (groups, offers, figures, steps; at
  most {IDM_MCQ_STEM_MAX_ITEMS}, each at most {IDM_MCQ_STEM_ITEM_MAX_WORDS} words), each item on its own line starting
  "- ", never labelled with the option letters A-H (after the options are shuffled they read as answer letters) and
  never named with the options' own noun ("Phương án", "Lựa chọn", "Option": the answer options A-D are the
  "phương án"): name them "Cách 1", "Đề xuất 1", "Tình huống 1", "Yêu cầu 1", "Nhân sự 1" or with a short
  descriptive name. Ask ONE question, once: the question sentence itself alone on the last line, ending with "?";
  no context line and no line before the list asks it (or another question) too. An option states its content (what
  the learner chooses); an option that only names an item ("Cách 2") is allowed, but then its explanation line says
  what that item is or does, never only "phương án này"; options that name the listed items have one item each, so
  the case lists {IDM_MCQ_OPTION_COUNT} items for {IDM_MCQ_OPTION_COUNT} options.
  explanation: line 1 the principle of the lesson that decides the answer (one sentence, two at most, at most
  {IDM_MCQ_EXPLANATION_PRINCIPLE_MAX_WORDS} words; never restate several definitions); then one line per option in
  the order A, B, C …: "A - Đúng vì …" or "A - Sai vì …" (English: "A - Correct because …" or "A - Incorrect
  because …"), one sentence of at most {IDM_MCQ_EXPLANATION_OPTION_MAX_WORDS} words giving the REASON from what this
  unit taught (the rule, criterion or step the option meets or breaks); it never repeats the option's own words and
  agrees with that option's text and with the answer key (never call a distractor that meets the criterion wrong, or
  the reverse). No empty praise, slogan or hype word instead of a reason, even when the source uses it ("sinh tử",
  "dũng cảm", "đập tan", "triệt để", "khổng lồ", "tuyệt đối", "vượt trội", "Say No", "brave", "absolutely"; only a
  quoted name of a source concept keeps its words); at most {IDM_MCQ_EXPLANATION_MAX_CHARS} characters in all. The
  shape, in JSON:
  "question": "Giả sử doanh nghiệp của bạn có ba nhóm khách hàng như sau:\\n- Nhóm X: …\\n- Nhóm Y: …\\nBạn nên làm gì?"
  "explanation": "Nguyên tắc: …\\nA - Sai vì …\\nB - Đúng vì …\\nC - Sai vì …\""""
# A worksheet slot (role practice). A5 (owner decision 2026-10-22): criteria the AI suggested because the source states
# none restate what the lesson teaches. Run 5496155c: the learner reads "Tự kiểm tra" only; the author note says the
# criteria are suggested (the server removes a label the writer still adds, ``worksheet.plain_self_check``).
_WORKSHEET_RULE: Final = """\
- An html slot whose role is "practice" is a WORKSHEET for its practice (the learner works on their own copy;
  nothing is graded automatically): a section whose heading names the task with a "task" block (what to produce,
  from which input); a section with the template to complete as a "table" block (label = the field or cell to
  fill, value = the guiding question or what a good entry contains) or as "steps"; a section with a short worked
  example of one or two filled entries (illustrative when the practice is ai_drafted; it never adds a rule); and a
  final section headed "Tự kiểm tra" / "Self-check" with one "bullets" block whose items are the criteria, each
  taken only from the practice criteria facts. When the practice's feedback_focus criterion starts with "Tiêu chí gợi
  ý" / "Suggested criteria" (the source states no grading criteria), its items restate what the lesson teaches, never
  a new rule, number or threshold; the heading stays "Tự kiểm tra" / "Self-check" and no learner text says "tiêu chí
  gợi ý" / "suggested criteria" (the author is told apart). The worked example meets every self-check item of its
  own worksheet (when an item asks for a measurable KPI, the example's KPI has a number). Do not reveal the answer of
  the problem slot that follows it; that problem asks the learner to judge a sample entry against the same criteria
  (exactly one option meets them). When the task refers to data the learner works on (a list of people, an
  interview, requests, cases, figures), the worksheet gives that data in full in its own section right after the task
  (heading "Dữ liệu đầu vào" / "Input data"), as a bullets list or a table with exactly the entries the task names
  (at most 6), about a case other than the worked example's; never refer to data the worksheet does not show. The
  worked example works on other data: never one of the input entries, not even shortened or renamed (an example that
  solves the learner's own entries gives the answer away). A worksheet has the same length budget as any html slot,
  and the input data counts toward it, so keep it lean: one template row per field (its value one guiding question
  of at most 15 words), input entries of one sentence of at most 20 words each, a worked example of at most two
  filled entries and at most 6 self-check items; a worksheet still over the budget loses its worked example."""
# Run 2f8781d6 (B5): three profiles that share their row labels ("Về Chức năng (Functional Job)" …) repeated those
# labels as identical cells, the duplicate rule Python and Node share (HTML_DUPLICATE_BLOCK) rejected the slot twice and
# it fell back to the source text.
_REPEAT_RULE: Final = """\
- In one html slot never repeat the same paragraph, list item, table cell or callout; a worksheet may repeat its
  template's row labels (and blank-cell guidance) in the worked example table. When several sections share the same
  row labels (profiles, cases, layers), put the label in the section heading or start it with the section's name
  ("Người gác cổng - Chức năng"), never an identical label twice."""
# Run 2f8781d6 (B7): the course was for sales and marketing staff, yet only 4 of 12 questions put the learner in that
# role, 4 made the leadership the actor and the source's own association ("hội viên", "Ban Giao thương") stayed the
# setting.
_ROLE_RULE: Final = """\
- Quiz and worksheet scenarios put the learner ("bạn" / "you") in the role TARGET_AUDIENCE names, in the learner's own
  business setting (their company, their customers): the learner is the one who acts or decides. Actors outside that
  role (the source's own organisation, its members or departments, a level the audience is not) appear only as the
  customer or a stakeholder; the facts and names the unit teaches stay as the source states them."""
# Run 2f8781d6 (B8): "BÀI TẬP 10 phút … chọn ra đúng 2 sáng kiến" became "đội ngũ cần 10 phút phản biện", "gồm đúng 2
# sáng kiến" (the server notes what is left: app.idm.slot_words).
_SELF_PACED_RULE: Final = """\
- The course is self-paced e-learning: no classroom instructions (no "trước khi rời lớp/học phần", "thảo luận nhóm
  tại lớp", "giảng viên sẽ …"). A classroom exercise of the facts ("BÀI TẬP 10 phút …", "Câu hỏi thảo luận …", "Bài
  tập về nhà theo nhóm") becomes a self-paced practice on the learner's own company: teach the method it practises and
  drop its time limit, its group, discussion or homework wording and the number of items it picks (unless a
  PRACTICE_TASKS sentence keeps that number); never write them as rules of real work ("đội ngũ cần 10 phút phản
  biện", "chọn đúng 2 sáng kiến")."""
# N9 and QC feedback 2026-10-21 (A3): a callout quotes the source and adds a point its slot does not already make.
_CALLOUT_RULE: Final = """\
- A "warning" block is shown to the learner as a quotation/callout: use it only for a rule, warning or
  statement that SOURCE_FACTS state, quoted or closely restated. Never write your own maxim, slogan, consequence
  or rule as a warning; such text is a plain paragraph, or is left out. A callout adds a point the slot's
  paragraphs, lists and tables do not already make: never the paragraph or list above it again, in the same or
  other words (the server removes a callout that only repeats them)."""


def _opening_line(example_opening: str) -> str:
    return (f'; in this unit the first drafted example or case opens with "{example_opening}" (still about the '
            "learner's own role)" if example_opening else "")


def unit_writer_prompt(
    locale: Locale,
    *,
    course_title: str,
    audience: str,
    lesson_title: str,
    lesson_objective: str,
    practice_sentences: Sequence[str],
    previous_title: str | None,
    next_title: str | None,
    unit_brief: dict[str, Any],
    facts: Iterable[tuple[str, str, Sequence[str] | None]],
    context_facts: Iterable[tuple[str, str, Sequence[str] | None]],
    html_rules: str = "",
    course_purpose: dict[str, Any] | None = None,
    fidelity_rules: str = "",
    faq_rules: str = "",
    example_opening: str = "",
    organization_context: str = "",
) -> str:
    """``fidelity_rules``: ``app.idm.fidelity.writer_fidelity_rules`` (the content mode; default structured);
    ``faq_rules``: ``app.idm.faq_checks.faq_writer_rules`` for a unit with an la_faq slot (QLT-5 D), else empty;
    ``example_opening`` (QLT-9): how this unit opens its first drafted example or case (``example_opening``);
    ``organization_context``: ``app.idm.org_context.writer_block`` (empty without an organization context)."""

    content_mode = fidelity_rules or writer_fidelity_rules(DEFAULT_MODE, locale)
    opening_line = _opening_line(example_opening)
    purpose_line = _PURPOSE_LINE if course_purpose else ""
    return preamble(locale) + f"""
TASK: Week 5 - write the development-ready storyboard content for ONE unit. The plan is approved; do not
redesign it. Produce exactly the component slots in UNIT_BRIEF (c0, c1, ...), in order, as payloads matching the
schema.
{html_rules}
Writing rules (learner-facing, {locale_name(locale)}):
{purpose_line}- Write for TARGET_AUDIENCE: short sentences, active voice, direct instructions; explain a necessary
  term the first time; split long text into short paragraphs, bullets, steps or tables; each heading says which
  question the section answers.
{content_mode}- Apply each block treatment: keep -> preserve wording; condense -> remove repetition and long
  introductions; rewrite -> plain language with the same meaning; explain -> plain language, why it matters to the
  learner and a short example. Keep every condition, negation, quantity, unit and exception that matters for the
  Must Do. Teach what the practice needs (detail_level).
- Context building stays under 20% of the unit text.
- Practice slots of type problem, la_sortable or la_crossword contain context, task, the input the learner works with,
  exactly one correct answer, and feedback that teaches. Single-choice: 4 options (3 only when the criterion leaves
  just two plausible mistakes); each distractor is a plausible near-miss a learner of TARGET_AUDIENCE could choose,
  wrong by the stated criterion: a common mistake, the old practice the facts reject, a criterion only half met or a
  step in the wrong order (never an obviously wrong, extreme or absurd option: no "chi phí cực cao", "vượt ngân sách",
  "gửi email nhắc khách kiên nhẫn chờ"; each one is a reasonable choice that applies the taught criterion wrongly); no
  "all/none of the above", correct option not always first, every option of comparable length and detail (the correct
  one is not the longest option: at least one distractor is as long or longer), and the correct option never reuses
  the wording of the example, quotation, worked example or template shown before the question, nor strings together
  the items of a list or table shown before it (it applies the criterion to the question's own case); the question's
  case is not the worked example's case: other facts decide it and its answer is not the worked example's answer; the
  case and its listed items never copy the words of the worked example, a table row or a list item shown before it
  (the learner applies the criterion, never matches words; a worksheet's input data is no taught content: the question
  may practise on the same entries, the worksheet gives no answer for them), and the key never repeats a number or
  detail of the worked example the case does not give; the case never names the cause, rule or criterion the key is
  about (the learner finds it from the case's facts). A question on a formula, definition or
  framework of several required parts ("đối tượng + lợi ích + cách làm khác biệt"): the key has every part; each
  distractor visibly lacks or misuses ONE named part (a different one each) and its explanation line names that
  part; never a distractor that also has every part. The question and the explanation follow QUIZ TEXT below. Never
  reveal the answer before the question. A question tests only what the html slots before it teach: every fact its
  key or explanation relies on is stated in that html first (the explanation recalls it, it never teaches it for the
  first time). A classification question (options naming taught labels, rows or stages) describes a case that meets the
  criteria of exactly one option and contradicts no FAQ answer of the unit; no option copies a table row or list item.
{_QUIZ_TEXT_RULE}
{_WORKSHEET_RULE}
{faq_rules}{_CALLOUT_RULE}
- Never invent rules, thresholds, criteria, labels or consequences the facts do not state, not even as the
  labels of a template or the edges of a diagram.
- Name a framework, model, matrix or process and each of its parts exactly as SOURCE_FACTS name it (their term and
  language, or the translation the facts themselves give): never call a part by the name of the whole or the whole
  by the name of one part, never rename it with your own metaphor, and never give it axes, dimensions, criteria,
  steps or stages the facts do not list (a matrix keeps the facts' axes and cells, a process its own steps, in their
  number and order).
- The facts keep marks of the PDF page: never copy them. No "Row N:" labels, no word cut at a line end
  ("Best-in" is "Best-in-Class"; drop a cut "(KPI /"), no Title Case Copied From A Slide (write slot titles,
  headings and labels in sentence case, keep acronyms and names), no word written twice in a row.
- Every slot title and section heading names what that slot or section actually teaches. When UNIT_BRIEF or a
  slot title names an enumerated framework ("5 chuyển dịch", "6 trụ cột", "3 giai đoạn"), the html lists every
  item of it, by the name the facts use, in one bullets, steps or table block. A support item that starts with
  "Liệt kê đủ" / "List all" names every item of that framework (taught in later units): list all of them, each with
  one short plain line on what it changes or is about (the support item that starts with "Ý chính của từng" / "Main
  idea of each" gives it, in the same order), without teaching them in depth here; the slot still follows the
  CONTENT MODE for the framework as a whole (why it matters to the learner, a short example).
- Use drafted scenarios/examples only where the brief marks them ai_drafted (a practice with scenario_origin
  ai_drafted, or a slot with "examples": "ai_drafted"); they never add rules, numbers or names, and their main person
  has the learner's own role (TARGET_AUDIENCE), never one of another level (no team lead in a course for executives).
  Vary how examples and cases open (not every one "Giả sử bạn …": "Hãy hình dung bạn …", "Nếu bạn …", "Hãy đặt
  mình vào tình huống: bạn …"){opening_line}. A drafted case is a supposition ("Giả sử …", "Hãy hình dung …", "Nếu
  …"), never told as a real past event ("Trong quý vừa qua …", "Năm ngoái, …"; the server writes "Giả sử" before
  such an opening).
{_ROLE_RULE}
{_SELF_PACED_RULE}
- An abbreviation the learner may not know gets its full form in brackets the first time this unit uses it, only as
  SOURCE_FACTS write it; never invent a full form.
- LESSON_CONTEXT_FACTS are read-only background from earlier units; use them for consistency and for the
  correctness criteria, do not re-teach them.
- No file names, page or slide numbers, citations, internal IDs, "theo tài liệu nguồn", emails or URLs; never a
  number right after the word trang/trạng, page or slide (write "Hiện trạng: có 3 lần", not "Hiện trạng: 3 lần").
- Text of la_faq, la_sortable, la_crossword and la_diagram slots never contains the characters "<" or ">".
{_REPEAT_RULE}
- When UNIT_BRIEF.job_aid_signpost is set, point the learner to the Job Aid in one sentence.
- covered_source_fact_ids: list exactly the owned fact keys given for that slot.
- title of each slot: a short learner-facing title; selection_rationale: one sentence for the author.

{organization_context}{untrusted_block("PLAN_CONTEXT", chr(10).join([
    "COURSE=" + _json(course_title),
    *(["COURSE_PURPOSE=" + _json(course_purpose)] if course_purpose else []),
    "TARGET_AUDIENCE=" + _json(audience),
    "LESSON=" + _json(lesson_title),
    "MUST_DO=" + _json(lesson_objective),
    "PRACTICE_TASKS=" + _json(list(practice_sentences)),
    "PREVIOUS=" + _json(previous_title),
    "NEXT=" + _json(next_title),
    "UNIT_BRIEF=" + _json(unit_brief),
]))}
{untrusted_block("SOURCE_FACTS", learner_fact_lines(facts))}
{untrusted_block("LESSON_CONTEXT_FACTS", learner_fact_lines(context_facts))}
"""


def html_slot_rules(*, max_words: int | None, max_chars: int | None, min_chars: int | None,
                    explain: bool = False) -> str:
    """The ordered-content rules the server enforces on every ``html`` slot (stated up front); ``explain``: the
    budget leaves room for the explanation and the example of an explained unit (QLT-5 C)."""

    paragraphs, paragraph_chars = GROUP_LIMITS["paragraphs"]
    bullets, bullet_chars = GROUP_LIMITS["bullet_points"]
    steps, step_chars = GROUP_LIMITS["ordered_steps"]
    warnings, warning_chars = GROUP_LIMITS["warnings"]
    length = ""
    if max_words is not None and max_chars is not None:
        length = (f"\n- Length of EACH html slot: at most {max_words} words and at most {max_chars} visible "
                  "characters (paragraph, task and warning text, list items and table cells; headings are not "
                  "counted; every space-separated word or syllable counts)")
        if min_chars is not None:
            length += f"; at least {min_chars} visible characters"
        length += (". Stay inside the budget: teach what the practice needs, with the explanation and the example of "
                   "each main idea." if explain else ". Stay well inside the budget: teach only what the practice "
                                                     "needs.")
    return f"""
HTML SLOT FORMAT (type "html"). The server renders the HTML itself and rejects an answer that breaks ANY rule:
- Write the slot only as semantic_content = {{"sections": [...]}} and set "html" to null. Every string is plain
  text: no HTML tags (no <p>, <div>, <span>, <h1>-<h6>, <ul>, <li>, <br>, <table>), no markdown (#, **, "- "
  bullets inside a text), no style or class attributes.
- Headings: one heading level only. Each section has exactly one "heading" (plain text, 1-{MAX_HEADING_CHARS}
  characters, saying which question the section answers). Never put a sub-heading inside blocks: start a new
  section instead. 1-{MAX_SECTIONS} sections per slot, 1-{MAX_BLOCKS_PER_SECTION} blocks per section.
- Block structure: kind paragraph, task or warning fills only "text"; kind bullets or steps fills only "items"
  (non-empty strings; steps in order); kind table fills only "rows", each {{"label": ..., "value": ...}} with both
  non-empty. Leave the other fields empty (text null, items [], rows []). A sentence that introduces a list or a
  table is its own paragraph block placed before it. No empty strings, no empty blocks.
- Whole-slot limits, counted over ALL sections together: at most {paragraphs} paragraph+task blocks (each at most
  {paragraph_chars} characters); at most {bullets} bullet items (each at most {bullet_chars}); at most {steps} step
  items (each at most {step_chars}); at most {warnings} warning blocks (each at most {warning_chars}); at most
  {MAX_ROWS} table rows (label at most {MAX_ROW_LABEL_CHARS}, value at most {MAX_ROW_VALUE_CHARS} characters).
  Group parallel points into one bullets block instead of many short paragraphs.
- No FAQ section and no question-and-answer pairs inside html (questions belong to la_faq or problem slots).{length}
"""


# QLT-9 (QC run ffaf7318: 12 of 14 drafted examples opened "Một CEO …"): each unit is written alone, so the server
# gives it one opening, chosen from the unit's place in the course. Run 2f8781d6 (B7): every opening puts the learner in
# the case (no leadership as the actor; no "Trong quý vừa qua", which became "Giả sử trong quý vừa qua"). Review 5 (L4):
# each one a supposition with the learner in it (no "Tại một doanh nghiệp …", no "Bạn vừa nhận được …" told as real).
# Review 6 (L8): in several forms, so that about one unit in three opens with "Giả sử" (85 % did).
_EXAMPLE_OPENINGS: Final[dict[str, tuple[str, ...]]] = {
    "vi": ("Giả sử bạn …", "Hãy hình dung bạn …", "Nếu bạn …", "Trong vai trò của mình, nếu bạn …",
           "Hãy đặt mình vào tình huống: bạn …", "Giả sử khách hàng của bạn …"),
    "en": ("Suppose you …", "Imagine you …", "If you …", "In your role, if you …", "Picture yourself …",
           "Suppose your customer …"),
}


def example_opening(unit_path: str, locale: Locale) -> str:
    """The opening of this unit's first drafted example or case (reproducible per unit)."""

    openings = _EXAMPLE_OPENINGS["vi" if locale == "vi" else "en"]
    draw = int.from_bytes(hashlib.sha256(f"opening:{unit_path}".encode()).digest()[:4], "big")
    return openings[draw % len(openings)]


# Short, server-owned rule text per validator code for repair prompts (never provider text).
UNIT_RULE_TEXT: Final[dict[str, str]] = {
    "HTML_SEMANTIC_INVALID": "semantic_content must follow the HTML SLOT FORMAT above",
    "HTML_SEMANTIC_REQUIRED": 'write the slot as semantic_content {"sections": [...]} with html null',
    "HTML_VERSION_INVALID": "do not write a version field; the server sets it",
    "HTML_LEGACY_FIELDS": 'semantic_content holds only "sections": put heading/paragraphs/bullet_points/'
                          "ordered_steps/warnings/comparison_rows content into section blocks",
    "HTML_UNKNOWN_FIELD": 'semantic_content holds only "sections"',
    "HTML_SECTION_COUNT": f"1-{MAX_SECTIONS} sections per slot",
    "HTML_SECTION_INVALID": 'a section has only "heading" and "blocks"',
    "HTML_HEADING_INVALID": f"each section heading is plain text of 1-{MAX_HEADING_CHARS} characters",
    "HTML_BLOCK_COUNT": f"each section has 1-{MAX_BLOCKS_PER_SECTION} blocks; start a new section or merge short "
                        "paragraphs",
    "HTML_BLOCK_KIND_INVALID": "block kind is one of paragraph, task, warning, bullets, steps, table",
    "HTML_BLOCK_FIELDS_MIXED": "a block fills only the field of its kind (paragraph/task/warning: text; "
                               "bullets/steps: items; table: rows); put an introduction sentence in its own "
                               "paragraph block before the list or table",
    "HTML_BLOCK_CONTENT_REQUIRED": "the field of the block's kind needs non-empty content; remove empty blocks",
    "HTML_ITEM_INVALID": "every text and list item is a non-empty plain string",
    "HTML_TEXT_TOO_LONG": "shorten this text or split it into a list",
    "HTML_ROW_INVALID": 'every table row is {"label", "value"} with both non-empty',
    "HTML_ROW_TOO_LONG": f"table label at most {MAX_ROW_LABEL_CHARS} characters, value at most {MAX_ROW_VALUE_CHARS}",
    "HTML_TOO_MANY_PARAGRAPHS": "too many paragraph+task blocks in the whole slot (all sections together): merge "
                                "related short paragraphs or turn parallel points into one bullets block",
    "HTML_TOO_MANY_BULLETS": "too many bullet items in the whole slot: keep only the points the practice needs",
    "HTML_TOO_MANY_STEPS": "too many step items in the whole slot",
    "HTML_TOO_MANY_WARNINGS": "too many warning blocks in the whole slot: merge related warnings",
    "HTML_TOO_MANY_ROWS": "too many table rows in the whole slot",
    "HTML_PRESENTATION_MARKUP": "plain text only: no HTML tags (div, span, style, script, iframe) and no style or "
                                "class attributes",
    "HTML_PRESENTATION_FORBIDDEN": "plain text only: no HTML tags (div, span, style, script, iframe) and no style "
                                   "or class attributes",
    "HTML_CONTENT_REQUIRED": "fill semantic_content with at least one section",
    "HTML_INSTRUCTIONAL_DENSITY_EXCEEDED": "condense the slot to the length budget in the HTML SLOT FORMAT",
    "HTML_INSUFFICIENT_DEPTH": "explain the owned facts in more visible text (see the length rule above)",
    "HTML_FAQ_BOUNDARY_VIOLATION": "no FAQ heading and no question-and-answer pairs inside html",
    "HTML_GENERIC_REVIEW_COPY": "write instruction for the learner, not notes about reviewing the source",
    "HTML_NON_INSTRUCTIONAL_CONTACT_COPY": "remove URLs, websites and e-mail addresses",
    "HTML_OCR_NOISE": "remove runs of repeated characters",
    "HTML_DUPLICATE_BLOCK": "do not repeat the same paragraph, list item or table cell (a worksheet may repeat only "
                            "its template's table cells); when several sections share the same row labels (profiles, "
                            "cases, layers), put the label in the section heading or start it with the section's name "
                            "(\"Người gác cổng - Chức năng\"), never an identical paragraph, list item or cell",
    # Node's revision-0 acceptance rules (app.idm.node_acceptance; codes as Node logs them).
    "HTML_SOURCE_REVIEW_COPY": "write instruction for the learner, not notes about reviewing the source",
    "HTML_SOURCE_ATTRIBUTION": 'no attribution to the source ("theo tài liệu", "dựa trên nguồn", "source states")',
    "HTML_SOURCE_LOCATOR": 'never put a number right after the word trang/trạng, page, slide or chunk (even after '
                           '":" or "số"); it reads as a page reference: write words in between ("Hiện trạng: có 3 lần '
                           '...")',
    "HTML_SOURCE_FILENAME": "no file names in learner text",
    "HTML_INTERNAL_IDENTIFIER": "no fact keys, source refs or internal IDs in learner text",
    "HTML_BOILERPLATE": "no paragraph, item or cell that is only a URL, an e-mail address or a thank-you line",
    "HTML_SEMANTIC_RENDER_INVALID": "semantic_content breaks the ordered-content format (see HTML SLOT FORMAT)",
    "HTML_CONTRACT_INVALID": "plain text only inside the ordered blocks",
    "HTML_TOO_THIN": "the slot needs at least 180 visible characters",
    "HTML_TOO_LARGE": "shorten the slot",
    "HTML_TABLE_TOO_LARGE": "split the table: at most 100 rows",
    "IDM_HTML_DENSITY_EXCEEDED": "condense the slot to the length budget in the HTML SLOT FORMAT",
    "IDM_HTML_VISIBLE_TEXT_INVALID": "semantic_content breaks the ordered-content format (see HTML SLOT FORMAT)",
    "PROBLEM_NOT_SINGLE_CHOICE": 'problem_type is "multiple_choice" with exactly one correct choice',
    "PROBLEM_UNSUPPORTED_RESPONSE": 'problem_type is "multiple_choice" with exactly one correct choice',
    "PROBLEM_QUESTION_INCOMPLETE": "one complete question of at least 20 characters",
    "PROBLEM_CHOICE_COUNT": "3-6 choices",
    "PROBLEM_CORRECT_COUNT": "exactly one choice with correct true",
    "PROBLEM_CHOICES_NOT_DISTINCT": "every choice at least 8 characters and clearly different from the others (not "
                                    "only in case, accents or punctuation)",
    "PROBLEM_CORRECT_BOILERPLATE": "the correct choice is an instructional statement, not a URL, e-mail or phone",
    "PROBLEM_EXPLANATION_MISSING": "an explanation of at least 20 characters that states the source criterion",
    "PROBLEM_ANSWER_REQUIRED": 'problem_type is "multiple_choice" with exactly one correct choice',
    "PROBLEM_TEXT_INVALID": "question, choices and explanation are plain text without control characters",
    "PROBLEM_XML_UNSUPPORTED": "question, choices and explanation are plain text without control characters",
    "FAQ_ITEM_COUNT": "at least two question-and-answer items",
    "FAQ_TEXT_FORBIDDEN_CHARACTER": 'questions and answers are plain text: never the characters "<" or ">" (write '
                                    '"→" or words such as "nhỏ hơn"/"less than")',
    "SORTABLE_ITEM_COUNT": "at least three ordered items",
    "SORTABLE_TEXT_FORBIDDEN_CHARACTER": 'the question and items are plain text: never the characters "<" or ">"',
    "SORTABLE_DUPLICATE_ITEM": "items must differ",
    "CROSSWORD_WORD_COUNT": "at least three terms with clues",
    "CROSSWORD_TEXT_FORBIDDEN_CHARACTER": 'clues and hints are plain text: never the characters "<" or ">"',
    "CROSSWORD_WORD_LAYOUT": "every term is a distinct word of 2-24 letters or digits with a clue",
    "DIAGRAM_NODE_COUNT": "at least two labelled nodes",
    "DIAGRAM_TEXT_FORBIDDEN_CHARACTER": 'labels, tooltips and edge labels are plain text: never "<" or ">"',
    "RESPONSE_COMPONENT_IDENTITY": "keep the slot type and order of UNIT_BRIEF",
    "RESPONSE_FACT_IDS": "covered_source_fact_ids lists exactly the owned fact keys of this slot",
    "REQUIRED_ARTIFACT_NOT_PRESERVED": "keep the list, table or warning structure the plan requires for this slot",
    "LEARNER_CONTENT_SOURCE_ATTRIBUTION": "no attribution to the source document in learner text",
    "LEARNER_CONTENT_SOURCE_LOCATOR": "no page, slide or section locators in learner text",
    "LEARNER_CONTENT_SOURCE_FILENAME": "no file names in learner text",
    "LEARNER_CONTENT_INTERNAL_IDENTIFIER": "no fact keys or internal IDs in learner text",
    "COMPONENT_PAYLOAD_SCHEMA_INVALID": "fill every required field of this slot type with valid values",
    "PROBLEM_SINGLE_CHOICE_REQUIRED": 'problem_type is "multiple_choice" with exactly one correct choice',
    "PROBLEM_QUESTION_REQUIRED": "write the question",
    "PROBLEM_CONTENT_INCOMPLETE": "a complete question (at least 20 characters) and 3-6 choices",
    "PROBLEM_CHOICE_INVALID": "every choice is a complete option of at least 8 characters",
    "PROBLEM_CHOICES_INVALID": "every choice has non-empty text",
    "PROBLEM_DUPLICATE_CHOICES": "choices must differ",
    "PROBLEM_CORRECT_ANSWER_INVALID": "distinct choices and exactly one choice with correct true",
    "PROBLEM_EXPLANATION_INCOMPLETE": "an explanation of at least 20 characters that states the source criterion",
    "FAQ_ITEM_COUNT_INVALID": "at least two question-and-answer items",
    "FAQ_ITEM_INVALID": 'every item is {"question", "answer"}',
    "FAQ_DUPLICATE_QUESTION": "questions must differ",
    "FAQ_CLARIFICATION_INCOMPLETE": "each question at least 14 characters; each answer a complete sentence of at "
                                    "least 40 characters that starts with a capital letter",
    "FAQ_GENERIC_REVIEW_COPY": "answer a learner question, not a note about reviewing the source",
    "SORTABLE_STEP_INCOMPLETE": "at least three complete steps of at least 14 characters each",
    "SORTABLE_DUPLICATE_STEP": "steps must differ",
    "SORTABLE_STEP_FRAGMENT": "every step is a complete step that starts with a capital letter",
    "SORTABLE_GENERIC_SOURCE_ORDER": "order a real procedure, not the display order of the source",
    "DIAGRAM_SOURCE_RELATION_MISSING": "keep every step chain the owned facts write with arrows (A -> B -> C) as "
                                       "edges in the same direction between nodes labelled with exactly those terms",
    "COMPONENT_COVERAGE_MISSING": "covered_source_fact_ids lists exactly the owned fact keys of this slot",
    "COMPONENT_COVERAGE_INCOMPLETE": "covered_source_fact_ids lists exactly the owned fact keys of this slot",
    "INVALID_FACT_ID_ARRAY": "covered_source_fact_ids lists exactly the owned fact keys of this slot",
    "IDM_W5_PRACTICE_INCOMPLETE": "exactly one correct choice, and an explanation of the principle line and one "
                                  "line per option in letter order (\"A - Đúng vì …\" / \"A - Sai vì …\") that "
                                  "says why it is right or wrong, in agreement with that option's text and the "
                                  "answer key",
    "IDM_W5_FEEDBACK_NOT_TEACHING": "feedback of at least 60 characters that names the criterion; never only "
                                    "Correct/Incorrect",
    "IDM_W5_ANSWER_LENGTH_CUE": "the correct option is clearly the longest, so learners can pick it by length: "
                                "balance the lengths (shorten the correct option, or give the distractors the same "
                                "level of plausible detail, each still wrong by the criterion), keep the answer key, "
                                "and keep the explanation's reason for every option",
    "IDM_W5_ANSWER_LEAK": "an option of this question copies or restates the example, worked example, quotation, "
                          "list item, table row or template shown before it, or strings several items of one list, "
                          "table or worked example together (or the question's case copies the example that names "
                          "the answer), or a listed item of the question copies the worked example, a table row or "
                          "a number of the example the key repeats: rewrite THAT option or item (and its part of the "
                          "explanation) in new words or about a different case so the learner must apply the "
                          "criterion, and give every distractor the same footing in what was taught; keep the earlier "
                          "html as it is",
    # Run 2f8781d6 (B9): a tool named by words SOURCE_FACTS do not use (app.idm.source_names), repaired once.
    "IDM_W5_NAME_NOT_IN_SOURCE": "the slot title or a section heading names a tool, model, matrix, map or framework "
                                 "by words SOURCE_FACTS never use (quoted after \"new:\"): give it the name the facts "
                                 "use, or say plainly what it is, the same in every title and heading of the slot; "
                                 "keep the content",
    # Run 2f8781d6 (B6): soft give-aways (app.idm.answer_near), repaired once.
    "IDM_W5_ANSWER_NEAR_EXAMPLE": "the correct option, or the listed item it names, repeats the words of the example, "
                                  "table row, list item or case right before it (or the case names the need in the "
                                  "key's own words), so the learner can match words: keep the principle the question "
                                  "tests and the answer key, change the case details and the wording of the case, that "
                                  "option or item and its explanation line; keep the earlier html as it is",
    "IDM_W5_ANSWER_ITEM_LENGTH_CUE": "the listed item the correct option names is clearly the longest item of the "
                                     "case, so learners can pick it by length: balance the items (shorten that item or "
                                     "give the others the same level of plausible detail, each still wrong by the "
                                     "criterion); keep the names, the options and the answer key",
    "IDM_W5_ANSWER_CONTESTED": "the question's case also meets the criteria this unit's html table, list, FAQ "
                               "answers or the table rows of SOURCE_FACTS give for another option (for example a code "
                               "or condition only that option's row states), so more than one option is defensible: "
                               "rewrite the case itself (its facts, not only its wording) so it meets the criteria of "
                               "exactly one option and contradicts no FAQ answer of this unit, keep the whole question "
                               "ending with its question sentence, and keep the explanation's reason for every option; "
                               "for a formula or framework of required parts, each distractor lacks one named part",
    "IDM_W5_QUESTION_INCOMPLETE": "the question stops before it asks anything: write the whole case and end the "
                                  "question with the question sentence itself (ending with \"?\"); keep the options, "
                                  "the answer key and the explanation consistent with it",
    "IDM_W5_VERBATIM_COPY": "rewrite for the learner instead of copying the source text",
    # QC feedback 2026-10-21 (A1/A2): the QUIZ TEXT rules (``app.idm.quiz_text``; the numbers follow the code).
    "IDM_W5_EXPLANATION_LAYOUT": "write the explanation in lines: line 1 the principle of the lesson that decides the "
                                 "answer, then one line per option in the order A, B, C …: \"A - Đúng vì …\" or "
                                 "\"A - Sai vì …\" (English \"A - Correct because …\" / \"A - Incorrect because "
                                 "…\"), each giving the reason; keep the options and the answer key",
    "IDM_W5_EXPLANATION_VERDICT": "make the verdict of every option line match the answer key: only the correct "
                                  "option's line says Đúng / Correct, every other line Sai / Incorrect, each with the "
                                  "reason the unit teaches",
    "IDM_W5_EXPLANATION_FILLER": "replace the empty praise, slogan or hype word quoted below (even when the source "
                                 "uses it) with the concrete reason the unit teaches (the rule, criterion or step the "
                                 "option meets or breaks)",
    "IDM_W5_FAQ_HYPE": "the answers use the hype words quoted below, banned even when the source uses them: say the "
                       "plain reason or consequence the facts give instead; keep the questions, their types and every "
                       "fact",
    "IDM_W5_EXPLANATION_LENGTH": f"shorten the explanation: the principle line one sentence (two at most) of at most "
                                 f"{IDM_MCQ_EXPLANATION_PRINCIPLE_MAX_WORDS} words, each option line one sentence of "
                                 f"at most {IDM_MCQ_EXPLANATION_OPTION_MAX_WORDS} words, at most "
                                 f"{IDM_MCQ_EXPLANATION_MAX_CHARS} characters in all; keep every option's reason",
    "IDM_W5_EXPLANATION_RESTATES_OPTION": "the listed option lines only repeat their option: say WHY the option is "
                                          "right or wrong from the rule, criterion or step the unit taught, without "
                                          "the option's own words",
    "IDM_W5_QUESTION_LAYOUT": "lay the question out in lines: one or two short context sentences; each listed item "
                              "on its own line starting \"- \"; the question sentence alone on the last line; keep "
                              "the case's facts, the options and the answer key",
    "IDM_W5_QUESTION_OVER_LIMIT": "shorten the question to well under 1000 characters (the system cuts a longer "
                                  "one): keep the facts that decide the answer and the question sentence; keep the "
                                  "options and the answer key",
    "IDM_W5_QUESTION_ITEM_LETTERS": "the question's listed items are named with the option letters A-H (\"Phương "
                                    "án A\"): name them with another noun and a number (\"Cách 1\", \"Đề xuất "
                                    "1\") or a short name, and use the same names in the options and the "
                                    "explanation; keep the facts and the answer key",
    # Run 5496155c (owner decision 2026-10-22): items, questions and explanation lines that confuse the learner.
    "IDM_W5_QUESTION_ITEM_NOUN": "the question's listed items are named with the noun of the answer options "
                                 "(\"Phương án 1\", \"Lựa chọn 1\", \"Option 1\"), so a choice reads \"A: "
                                 "Phương án 2\": name them with another noun (\"Cách 1\", \"Đề xuất 1\", \"Tình "
                                 "huống 1\", \"Yêu cầu 1\") or a short name, the same in the options and the "
                                 "explanation; keep the facts and the answer key",
    "IDM_W5_QUESTION_ASKED_TWICE": "the question asks twice (a context line or the line before the list asks too): "
                                   "keep ONE question sentence, alone on the last line; the earlier lines only give "
                                   "the case; keep the facts, the options and the answer key",
    "IDM_W5_EXPLANATION_NAMES_ONLY": "the listed option lines explain an option that only names a listed item "
                                     "(\"Cách 2\") without saying what that item is: say in each line what the "
                                     "item proposes and why it meets or breaks the criterion; keep the answer key",
    "IDM_W5_QUESTION_LENGTH": f"shorten the question: the context and the question sentence together at most "
                              f"{IDM_MCQ_STEM_PROSE_MAX_WORDS} words, at most {IDM_MCQ_STEM_MAX_ITEMS} listed items of "
                              f"at most {IDM_MCQ_STEM_ITEM_MAX_WORDS} words each; keep the facts that "
                              "decide the answer and the question sentence; keep the options and the answer key",
    # QLT-5 C: the content mode checks of a teaching html slot (``app.idm.fidelity_checks``).
    "IDM_W5_SENTENCE_COPY": "too many sentences repeat the facts word for word: rewrite the listed sentences in your "
                            "own plain words and explain them (definitions, rules and numbers stay exact); add no "
                            "fact",
    "IDM_W5_EXPLANATION_MISSING": "this teaching slot only states the facts: add the explanation the CONTENT MODE asks "
                                  "for (the main idea in plain words, why it matters to the learner, a short example; "
                                  "a drafted example starts with its illustrative-example label) without adding facts",
    "IDM_W5_SENTENCE_LENGTH": "the listed texts hold sentences that are too long: split each into short sentences of "
                              "one idea each (about 15-20 words); keep their meaning and every exact rule, definition "
                              "and number; add no fact",
    "IDM_W5_EXAMPLE_ROLE": "the listed drafted example is about a person of another level than TARGET_AUDIENCE: make "
                           "its main person someone in the learner's own role, keep the rest of the example; add no "
                           "name, number or rule",
    "IDM_W5_NEW_FACT": "the listed text states a number, date, amount or a person, company or product name the facts "
                       "do not state (the words quoted after \"new:\", copied from your answer): remove each of them "
                       "or use what the facts say, and change nothing else in that text (a drafted example uses "
                       "roles, never names or new numbers)",
    "IDM_W5_SOURCE_ARTIFACT": "a list item, row label or table cell ends cut as the PDF line did (an open \"(\", a "
                              "trailing \"/\" or a word such as \"Best-in\"): write it out in full from the facts "
                              "or drop the cut part",
    "IDM_W5_FAQ_RESTATES_HTML": "these items only repeat the html above: replace each listed item with a common "
                                "misconception, an edge case or a what-if question that SOURCE_FACTS or "
                                "LESSON_CONTEXT_FACTS answer",
    # QLT-5 D: the FAQ checks (``app.idm.faq_checks``); an answer explains, it is never asked to restate.
    "IDM_W5_FAQ_RESTATES_SOURCE": "the listed answers repeat a source sentence almost word for word: rewrite each in "
                                  "2-3 sentences (at most about 60 words) of your own that explain why, what happens "
                                  "if, or what goes wrong, based on its facts; add no number, rule or name",
    "IDM_W5_FAQ_TYPE_MIX": "the questions mostly ask for facts again: keep at most one recall question and make at "
                           "least one a why, what-if or common-mistake question on the planned facts, each answered in "
                           "2-3 explaining sentences (at most about 60 words)",
    "IDM_W5_FAQ_NEW_FACT": "the listed items state a number, date, amount or a person, company or product name their "
                           "facts do not state (the words quoted after \"new:\", copied from your answer): remove each "
                           "of them or use what the facts say; explaining in your own words stays allowed",
    "IDM_W5_FAQ_OPEN_QUESTION": "the listed items ask or answer a point the source has not settled (OPEN_QUESTIONS): "
                                "replace each with another why, what-if or common-mistake question the planned facts "
                                "answer",
    "IDM_W5_FAQ_INVENTED_REASON": "the listed items give a reason for a number or threshold that their facts do not "
                                  "explain: ask instead what the facts do say about it (what it requires, what happens "
                                  "below it) or another question; never invent why",
    "IDM_W5_FAQ_TITLE_MISMATCH": "the la_faq title names what its questions are about: retitle it from its questions",
    "IDM_W5_CALLOUT_UNGROUNDED": "a warning block is shown as a quotation/callout: quote or closely restate what "
                                 "SOURCE_FACTS state, or make it a paragraph block; never a rule, threshold or "
                                 "consequence the facts do not state",
    "IDM_W5_FRAMEWORK_INCOMPLETE": "the unit or slot title names a framework of several items: list every item by "
                                   "the name the facts use in one bullets, steps or table block of this html slot, "
                                   "or make the slot title and headings name what the slot teaches",
    "IDM_W5_WORKSHEET_INCOMPLETE": "a worksheet slot needs a task block, the template to complete (table rows or "
                                   "steps) and a final self-check bullets block whose items are the practice "
                                   "criteria taken from the facts",
    # Run 5496155c: "đọc danh sách nhân sự …" without the list (``app.idm.worksheet_input``).
    "IDM_W5_WORKSHEET_EXAMPLE_REUSES_INPUT": "the worked example works on entries of the task's own input data "
                                             "(listed below), so it gives the learner the answer: rewrite the worked "
                                             "example on another case or other entries (never one of the input "
                                             "entries, not even shortened or renamed); keep the task, the input data, "
                                             "the template and the self-check",
    "IDM_W5_QUESTION_FEW_OPTIONS": f"the options only name the case's listed items and there are fewer than "
                                   f"{IDM_MCQ_OPTION_COUNT}: add one more plausible item to the case (its own line, "
                                   f"the same naming), its option and its explanation line, so the question has "
                                   f"{IDM_MCQ_OPTION_COUNT} options; keep the other items and the answer key",
    "IDM_W5_WORKSHEET_INPUT_MISSING": "the worksheet's task refers to data the learner works on (a list, an interview, "
                                      "requests or cases) that the worksheet does not give: add that data in full in "
                                      "its own section right after the task (heading \"Dữ liệu đầu vào\" / \"Input "
                                      "data\"), as a bullets list or a table with exactly the number of entries the "
                                      "task names, about a case other than the worked example's; drafted entries add "
                                      "no rule or number the facts do not state; keep the task, the template, the "
                                      "worked example and the self-check",
    "IDM_W6_Q1": "a learner can do the practice using only what this lesson taught",
    "IDM_W6_Q2": "rewrite for the learner; do not paste the source (exact rules and definitions may stay)",
    "IDM_W6_Q3": "context, task, input, one correct answer and feedback are present and consistent; the question ends "
                 "with its question sentence; a worksheet's worked example meets every one of its self-check items",
    "IDM_W6_Q4": "feedback names the criterion in its first line, then gives one line per option with the reason it "
                 "is right or wrong (from what the unit taught, not the option's own words); the explanation of each "
                 "option agrees with that option's text and with the answer key",
    "IDM_W6_Q5": "the correct answer follows from the source facts; no invented rule, threshold or exception",
    "IDM_W6_Q6": "the practice matches the Must Do action type and Bloom level",
    "IDM_W6_Q7": "no long theory runs; context under 20%; detail matches detail_level",
    "IDM_W6_Q8": "clear language for the audience; terms explained; no internal IDs or citations",
    "IDM_W6_Q9": "every component serves the stated Must Do, practice or support",
    "IDM_W6_Q10": "make the slot title and section headings name what the content teaches, or teach what the "
                  "title promises (every item of a list it names)",
    "IDM_W6_Q11": "keep the slot on COURSE_PURPOSE: teach what the course is for, for its audience; rewrite or drop "
                  "content that pulls it towards another purpose, without adding facts",
    "IDM_W6_Q12": "explain the content for the learner: the main idea in plain words, why it matters and a short "
                  "example; rules, definitions and numbers stay exact; add no fact",
    "IDM_W6_Q13": "make the la_faq explain instead of restating: why, what happens if, or what goes wrong, based on "
                  "the facts, at most one recall question; add no number, rule or name",
}
# Owner P3 (2026-10-20): the second targeted repair of a slot that still gives its answer away, still states a new
# number or name, or still has a contested key after the first repair. A leak rewritten inside the same case tends to
# stay close to the worked example it copied, so the second attempt moves the case to another setting.
SECOND_REPAIR_TEXT: Final[dict[str, str]] = {
    "IDM_W5_ANSWER_LEAK": "SECOND ATTEMPT: the first rewrite still gives the answer away. Set the question's case in a "
                          "different industry AND at a different touchpoint (another moment of the work, another "
                          "person or document) than every example, worked example, quotation, list and table shown "
                          "before it, then write every option fresh for that case: the key applies the criterion to "
                          "the new case and restates no item, row or example of the html; every distractor is a "
                          "plausible mistake on the same case; keep the earlier html as it is",
    "IDM_W5_NEW_FACT": "SECOND ATTEMPT: the listed texts still state the quoted numbers or names: delete each quoted "
                       "word or replace it with the exact words of the facts; add no other number or name",
    "IDM_W5_ANSWER_CONTESTED": "SECOND ATTEMPT: the case still meets the criteria of another option: change the "
                               "facts of the case (not only its wording) so that the criteria of exactly one option "
                               "hold and every other option is clearly ruled out by a criterion the unit teaches; keep "
                               "the whole question ending with its question sentence and the explanation's reason for "
                               "every option",
}
# A condensed slot aims below the hard limit, so a near miss does not fail again.
DENSITY_TARGET_SHARE: Final = 0.9
_MAX_RULE_LINES: Final = 40
# The new numbers and names a repair line quotes (owner P2, 2026-10-20): tokens of the writer's own answer (word
# characters, spaces, digits, "." and ","), so they cannot carry markup; bounded in number and length.
_MAX_HINT_TOKENS: Final = 8
_MAX_HINT_TOKEN_CHARS: Final = 40
_HINT_TOKEN_RE: Final = re.compile(r"^[\w .,]{1,40}$")


def _metrics_text(violation: HtmlRuleViolation) -> str:
    measured = ", ".join(f"{value} {unit}" for unit, value, _limit in violation.metrics)
    limits = ", ".join(f"{limit} {unit}" for unit, _value, limit in violation.metrics)
    return f"measured {measured}; limit {limits}"


def html_violation_line(slot: str, violation: HtmlRuleViolation, *, only_density: bool = False,
                        explain: bool = False) -> str:
    """One repair line: slot, code, JSON location, the rule and the server-measured numbers. ``explain``: an
    explained unit cuts repetition first and keeps its explanations and examples (QLT-5 C)."""

    location = f"{slot}.{violation.location}"
    if violation.code == DENSITY_CODE and violation.metrics:
        targets = " and ".join(f"{int(limit * DENSITY_TARGET_SHARE)} {unit}"
                               for unit, _value, limit in violation.metrics)
        cut = ("cut repetition first, then long introductions and source detail the practice does not need; keep the "
               "explanation and the example of each main idea" if explain else
               "cut repetition, long introductions and examples the brief does not ask for")
        line = (f"{slot} {violation.code} at {location}: {_metrics_text(violation)}. Condense this slot to at "
                f"most {targets}: {cut}; keep every rule, condition, number and exception of the owned facts; prefer "
                "short bullets to long paragraphs; do not move content into other slots.")
        if only_density:
            line += " Everything else in this slot is valid: keep its sections and their order, only shorten it."
        return line
    rule = UNIT_RULE_TEXT.get(violation.code)
    numbers = f" ({_metrics_text(violation)})" if violation.metrics else ""
    return f"{slot} {violation.code} at {location}" + (f": {rule}" if rule else "") + numbers


def rule_line(slot: str, code: str, location: str) -> str:
    """A repair line for a validator code without a finer IDM diagnosis."""

    rule = UNIT_RULE_TEXT.get(code)
    return f"{slot} {code} at {location}" + (f": {rule}" if rule else "")


def new_fact_hint(tokens: Iterable[str]) -> str:
    """' (new: "Think Fast", "2")': the new numbers and names a repair must remove, JSON-quoted (owner P2)."""

    quoted = [_json(token) for token in dict.fromkeys(token.strip()[:_MAX_HINT_TOKEN_CHARS] for token in tokens)
              if _HINT_TOKEN_RE.fullmatch(token)][:_MAX_HINT_TOKENS]
    return f" (new: {', '.join(quoted)})" if quoted else ""


def second_repair_line(slot: str, code: str) -> str:
    """The extra line of the second targeted repair of ``code`` in ``slot`` (owner P3); "" for another code."""

    text = SECOND_REPAIR_TEXT.get(code)
    return f"{slot} {code}: {text}" if text else ""


def unit_repair_suffix(issues: Sequence[dict[str, str]], rules: Sequence[str]) -> str:
    """``repair_suffix`` plus the precise failing rules per slot (code, location, rule, numbers)."""

    suffix = repair_suffix(issues)
    if not rules:
        return suffix
    lines = "\n".join(f"- {line}" for line in list(dict.fromkeys(rules))[:_MAX_RULE_LINES])
    return (suffix + "FAILING_RULES (fix every one; a location is a JSON path inside your answer, c0 = "
            "components.c0):\n" + lines + "\n")
