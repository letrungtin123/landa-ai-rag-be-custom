"""Content modes (QLT-5 C): how closely learner text follows the source wording.

Measured on real runs (plan 2026-10-13 §2.2), learner text read like the source file: W2 had no explaining
treatment (72% condense, 21% keep, 7% rewrite), the writer could add an example only to an AI-drafted practice and
the anti-copy check never fired. Each purpose profile carries a content mode that every later stage now follows as
policy, not only as prompt text:

* ``keep``: the wording is the content (compliance: rules, procedures, approved text). No explanation, no rewrite
  of a rule, no copy check.
* ``structured``: today's behaviour, organised for learning (sales, general, and every design stored before the
  mode existed).
* ``explain``: adaptive teaching (onboarding, product, soft skills). Each teaching slot gives the main idea in plain
  words, why it matters, a short example (AI-drafted allowed when labelled "Ví dụ minh hoạ:") and a common
  confusion; exact wording only for definitions, rules and numbers.

The server derives the mode from the profile; a clear request in the author's course description to keep the source
wording only ever tightens it to ``keep``. Authors never choose it. Every rule text here is server-owned.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Final

from app.idm.contracts import (
    FidelityMode,
    IdmBlueprintRowV1,
    IdmBriefComponentV1,
    IdmContentBlockV1,
    IdmLessonDesignV1,
    IdmPurposeV1,
    IdmSupportItemV1,
    IdmUnitBriefV1,
    Treatment,
    UnitSegment,
)
from app.idm.example_roles import AudienceRoles, role_rule
from app.idm.policy import (
    AI_DRAFTED_MARKER_EN,
    AI_DRAFTED_MARKER_VI,
    EXAMPLE_LABEL_EN,
    EXAMPLE_LABEL_VI,
    EXPLAIN_BUDGET_PERCENT,
    MAX_SUPPORT_ITEMS,
)
from app.idm.text import idm_fold, single_line

DEFAULT_MODE: Final[FidelityMode] = "structured"
TREATMENT_POLICY_CODE: Final = "IDM_W2_TREATMENT_POLICY"
TEACHING_SUPPORT_CODE: Final = "IDM_W4_TEACHING_SUPPORT_ADDED"
_PERCENT: Final = 100

# --- Resolution ----------------------------------------------------------------------------------------------------
# Folded phrases of the author's description that ask for the source wording. "giữ nguyên" alone is not one: an
# onboarding description may ask to "giữ nguyên giá trị cốt lõi" (keep the core values).
_KEEP_PHRASES: Final = (
    "nguyen van", "giu nguyen cau chu", "giu nguyen noi dung", "giu nguyen van ban", "giu nguyen loi van",
    "giu nguyen tai lieu", "giu nguyen tu ngu", "giu nguyen cach dien dat", "giu dung cau chu",
    "khong thay doi cau chu", "khong duoc thay doi noi dung", "verbatim", "word for word", "exact wording",
    "original wording", "keep the wording", "do not paraphrase",
)
# A phrase with one of these folded words among the four before it is a negation ("không cần giữ nguyên văn", "no
# need to be verbatim").
_NEGATIONS: Final = frozenset({"khong", "chang", "not", "no", "without", "never", "dont"})
_NEGATION_WINDOW: Final = 4
_NON_WORD_RE: Final = re.compile(r"[^0-9a-z]+")


def description_asks_to_keep(description: str | None) -> bool:
    """The author's description asks to keep the source wording ("giữ nguyên câu chữ", "nguyên văn", "verbatim")."""

    words = f" {_NON_WORD_RE.sub(' ', idm_fold(description or '')).strip()} "
    for phrase in _KEEP_PHRASES:
        start = words.find(f" {phrase} ")
        while start >= 0:
            before = words[:start].split()[-_NEGATION_WINDOW:]
            if not _NEGATIONS & set(before):
                return True
            start = words.find(f" {phrase} ", start + 1)
    return False


def resolve_fidelity(profile_mode: FidelityMode, description: str | None) -> FidelityMode:
    """The profile's mode, tightened to ``keep`` when the author's description asks for the source wording."""

    return "keep" if description_asks_to_keep(description) else profile_mode


def purpose_mode(purpose: IdmPurposeV1 | None) -> FidelityMode:
    """The mode a design follows; a design without one (contract 1, or stored before QLT-5 C) keeps today's rules."""

    return purpose.fidelity_mode if purpose is not None and purpose.fidelity_mode is not None else DEFAULT_MODE


def brief_mode(brief: IdmUnitBriefV1) -> FidelityMode:
    return brief.fidelity_mode or DEFAULT_MODE


# --- W2: treatments ------------------------------------------------------------------------------------------------
# Block kinds whose wording decides correct performance: never explained by default, never rewritten in keep mode.
_RULE_KINDS: Final = frozenset({"policy", "procedure", "decision", "system", "reference"})
# Must Know kinds an explained course teaches with "explain" unless W2 kept their wording.
_EXPLAINED_KINDS: Final = frozenset({"concept", "mindset", "case", "skill"})
_TREATMENTS: Final = (
    "treatment of a course row (must_do, must_know), one of:\n"
    "  keep = preserve the source wording (approved rules, definitions, legal or contract text, exact steps);\n"
    "  condense = remove repetition and long introductions; what stays keeps its meaning and key wording;\n"
    "  rewrite = plain language with the same meaning (dense, technical or badly written text);\n"
    "  explain = plain language + why it matters to the learner + a short example (a concept, principle, value, "
    "mindset or case the learner must understand, not only read);\n"
    "  combine = merge into another block of the same classification (set combine_into);\n"
    "  separate = split into parts that need different presentation (set separate_into).\n"
    "treatment of a reference row: move_to_reference | convert_to_job_aid; of a nice_to_know or remove row: remove.\n")
_W2_MODE_RULES: Final[dict[FidelityMode, str]] = {
    "explain": ("CONTENT MODE explain (the course explains, it does not transcribe): a must_know concept, principle, "
                "value, mindset or case uses explain (the default); a rule, definition, step, number or threshold "
                "keeps its wording (keep or condense) and is still explained right after it is quoted."),
    "structured": ("CONTENT MODE structured (organise the source for learning): use keep, condense, rewrite, combine "
                   "or separate; never explain."),
    "keep": ("CONTENT MODE keep (the source wording is the content): rules, steps, conditions, definitions and "
             "numbers use keep (condense only removes repetition); never rewrite or explain them."),
}


def w2_treatment_rules(mode: FidelityMode) -> str:
    """Every treatment defined, then what the content mode allows and defaults to."""

    return f"{_TREATMENTS}{_W2_MODE_RULES[mode]}\n"


def _policy_treatment(row: IdmBlueprintRowV1, kind: str | None, mode: FidelityMode) -> Treatment:
    treatment = row.treatment
    if row.placement != "course":
        return treatment
    rule = kind in _RULE_KINDS
    if mode == "explain":
        explained = (row.classification == "must_know" and treatment in {"condense", "rewrite"}
                     and kind in _EXPLAINED_KINDS)
        return "explain" if explained else treatment
    if mode == "structured":
        return "rewrite" if treatment == "explain" else treatment
    if treatment == "explain":
        return "keep" if rule else "condense"
    return "keep" if treatment == "rewrite" and rule else treatment


def enforce_treatments(rows: Sequence[IdmBlueprintRowV1], blocks: Sequence[IdmContentBlockV1],
                       mode: FidelityMode) -> tuple[list[IdmBlueprintRowV1], int]:
    """The rows with the treatments the content mode allows (and its default), and how many changed.

    explain: a Must Know concept, mindset, case or skill W2 condensed or rewrote is explained. structured: "explain"
    becomes "rewrite" (no example). keep: a rule, step, decision or system block is never rewritten or explained
    (it keeps its wording); another block W2 wanted to explain is condensed.
    """

    kinds = {block.block_id: block.content_kind for block in blocks}
    result: list[IdmBlueprintRowV1] = []
    changed = 0
    for row in rows:
        treatment = _policy_treatment(row, kinds.get(row.block_id), mode)
        changed += treatment != row.treatment
        result.append(row if treatment == row.treatment else row.model_copy(update={"treatment": treatment}))
    return result, changed


# --- W3/W4: teaching supports --------------------------------------------------------------------------------------
# Support kinds that explain beyond the facts: an example, a worked example, a comparison (non_example) or why it
# matters (explain_principle).
TEACHING_SUPPORT_KINDS: Final = frozenset({"example", "worked_example", "non_example", "explain_principle"})
_NOT_TEACHING_ROLES: Final = frozenset({"practice", "job_aid"})


def module_fidelity_rules(mode: FidelityMode, locale: str) -> str:
    """Module stage: in explain mode every teaching html plans an example, a comparison or why it matters."""

    if mode != "explain":
        return ""
    marker = AI_DRAFTED_MARKER_VI if locale == "vi" else AI_DRAFTED_MARKER_EN
    return ("CONTENT MODE explain: every teaching html component (role explain, show, clarify or summary) lists at "
            "least one support item of kind example, worked_example, non_example (a comparison or contrast) or "
            "explain_principle (why it matters to the learner). An example the source does not give may be drafted "
            f'by you: start its brief with "{marker}"; it shows the idea in a work situation of TARGET_AUDIENCE, about '
            "a person in the learner's own role (never a role of another level, such as a team lead in a course for "
            "executives), and never adds a number, rule, threshold or a person, company or product name.\n")


def _example_brief(name: str, locale: str) -> str:
    topic = single_line(name, 120)
    if locale == "vi":
        return single_line(f'{AI_DRAFTED_MARKER_VI} Ví dụ minh hoạ ngắn về "{topic}" trong công việc của người học; '
                           "không thêm số liệu, quy định hay tên riêng.", 300)
    return single_line(f'{AI_DRAFTED_MARKER_EN} A short illustrative example of "{topic}" in the learner\'s work; no '
                       "new numbers, rules or names.", 300)


def plan_teaching_supports(lessons: Sequence[IdmLessonDesignV1], block_names: Mapping[str, str],
                           mode: FidelityMode, locale: str) -> tuple[list[IdmLessonDesignV1], int]:
    """Explain mode: a teaching html component without an example, comparison or why-it-matters support item gets
    an AI-drafted example item about its first block (the last item gives way when the list is full)."""

    if mode != "explain":
        return list(lessons), 0
    added = 0
    result: list[IdmLessonDesignV1] = []
    for lesson in lessons:
        units = []
        for unit in lesson.units:
            components = list(unit.components)
            for position, component in enumerate(components):
                if (unit.segment == "job_aid" or component.type != "html" or component.role in _NOT_TEACHING_ROLES
                        or any(item.kind in TEACHING_SUPPORT_KINDS for item in component.support_items)):
                    continue
                block_id = component.block_ids[0]
                item = IdmSupportItemV1(kind="example", brief=_example_brief(block_names.get(block_id, unit.title),
                                                                             locale), block_id=block_id)
                kept = list(component.support_items)[:MAX_SUPPORT_ITEMS - 1]
                components[position] = component.model_copy(update={"support_items": [*kept, item]})
                added += 1
            units.append(unit.model_copy(update={"components": components}))
        result.append(lesson.model_copy(update={"units": units}))
    return result, added


# --- W5: writer rules, slot marks and budget -----------------------------------------------------------------------
def is_teaching_slot(slot: IdmBriefComponentV1, segment: UnitSegment) -> bool:
    """An html slot that teaches (not a worksheet, not a job aid)."""

    return slot.type == "html" and slot.role not in _NOT_TEACHING_ROLES and segment != "job_aid"


def teaching_slots(brief: IdmUnitBriefV1) -> list[int]:
    return [index for index, slot in enumerate(brief.components) if is_teaching_slot(slot, brief.unit_segment)]


def example_label(locale: str) -> str:
    return EXAMPLE_LABEL_VI if locale == "vi" else EXAMPLE_LABEL_EN


def writer_fidelity_rules(mode: FidelityMode, locale: str, roles: AudienceRoles | None = None) -> str:
    """The content-mode rules of the W5 writer (one bullet of its writing rules). ``roles``: the roles a drafted
    example may give its main person (QLT-7, ``example_roles.audience_roles``); the generic rule without them."""

    if mode == "keep":
        return ("- CONTENT MODE keep: the wording of the facts is the content (rules, procedures, approved text). Keep "
                "every rule, step, condition, definition, number and exception in the exact words of the facts; you "
                "may split them into sections and lists, order them, add headings and one short lead-in sentence per "
                "section. Never reword, summarise or explain a rule in other words, and never add an example, "
                "consequence or advice the facts do not state.\n")
    if mode == "structured":
        return ("- CONTENT MODE structured: organise the facts for learning. Do not paste source paragraphs: restate "
                "descriptive text in short sentences (on average at most 20 words); definitions, rules and numbers "
                "keep their exact wording. Add an example only where the facts give one or the brief marks it "
                "ai_drafted.\n")
    vi = locale == "vi"
    means = '"nghĩa là …", "tức là …"' if vi else '"this means …", "in other words …"'
    # QLT-7: generic words ("để", "giúp", "vì") alone explain nothing; the W5 check counts only these devices.
    clarifiers = ('"nghĩa là", "tức là", "nói cách khác", "điều này có nghĩa là", "vì vậy", "nhờ đó"' if vi
                  else '"this means", "in other words", "that is why", "as a result"')
    generic = '"để", "giúp", "vì", "nên"' if vi else '"to", "helps", "because", "so"'
    # QLT-7 (QC run fae66524): the fixed roles "một nhân viên mới, một khách hàng, trưởng nhóm" put a team lead into
    # 11 of 17 examples of a course for CEOs; the roles now come from the audience.
    roles_text = role_rule(roles or AudienceRoles(frozenset(), ()))
    return (
        '- CONTENT MODE explain: teach, do not transcribe. Every teaching html slot (marked "examples": "ai_drafted" '
        "in UNIT_BRIEF) follows this pattern for its main idea(s):\n"
        "  1. the main idea in plain words, in your own short sentences: one idea per sentence, on average at most 20 "
        "words, never more than 35 (split a long sentence in two);\n"
        "  2. why it matters to this learner: what it changes in their work (COURSE_PURPOSE, TARGET_AUDIENCE);\n"
        "  3. a short example or situation: one from SOURCE_FACTS when they give one; otherwise draft one and start "
        f'it with "{example_label(locale)}". The main person of a drafted example has {roles_text}; never a role of '
        "another level (no team lead or junior staff in a course for executives), never a person, company or "
        "product name, a number, percentage, amount, date, threshold or rule the facts do not state;\n"
        "  4. a common confusion or mistake, only when the facts support one.\n"
        "  Definitions, rules, steps, numbers and thresholds keep the exact wording of the facts, each followed by a "
        f"sentence that explains it in plain words ({means}). Never paste a source paragraph as it is: at most "
        "one in five of a slot's sentences may repeat the facts word for word, not counting the exact definitions, "
        f"rules and numbers. Explain with a labelled example, a clarifying phrase ({clarifiers}) or a sentence that "
        f"says why; a generic word ({generic}) alone explains nothing.\n")


def slot_marks(brief: IdmUnitBriefV1, index: int) -> dict[str, str]:
    """Extra writer-view keys of slot ``index``: an explained unit's teaching slot may hold an AI-drafted example
    (the practice-only ``ai_drafted`` mark extended to teaching, QLT-5 C)."""

    slot = brief.components[index]
    if brief_mode(brief) == "explain" and is_teaching_slot(slot, brief.unit_segment):
        return {"examples": "ai_drafted"}
    return {}


def budget_words(base_words: int, mode: FidelityMode, segment: UnitSegment) -> int:
    """The W5 word budget of a unit in ``mode`` (Node: ``idmUnitOutputBudget``)."""

    if mode == "explain" and segment != "job_aid":
        return base_words * EXPLAIN_BUDGET_PERCENT // _PERCENT
    return base_words


# --- Author notes --------------------------------------------------------------------------------------------------
_MODE_NOTE_VI: Final[dict[FidelityMode, str]] = {
    "explain": ("• Cách trình bày nội dung: diễn giải dễ hiểu — mỗi phần lý thuyết nêu ý chính bằng lời dễ hiểu, vì "
                'sao quan trọng và có ví dụ; ví dụ do AI soạn được ghi "Ví dụ minh hoạ" và cần kiểm tra.'),
    "structured": ("• Cách trình bày nội dung: sắp xếp lại cho dễ học — rút gọn, câu ngắn, giữ đúng ý của tài liệu."),
    "keep": ("• Cách trình bày nội dung: giữ nguyên câu chữ của tài liệu cho quy định, quy trình và định nghĩa — chỉ "
             "chia khối, sắp xếp và thêm tiêu đề."),
}
_MODE_NOTE_EN: Final[dict[FidelityMode, str]] = {
    "explain": ("• Content style: explained for the learner — each theory part gives the main idea in plain words, why "
                'it matters and an example; examples drafted by AI are marked "Illustrative example" and need a '
                "check."),
    "structured": ("• Content style: organised for learning — condensed, short sentences, the meaning of the source "
                   "kept."),
    "keep": ("• Content style: the source wording is kept for rules, procedures and definitions — only split into "
             "parts, ordered and given headings."),
}


def fidelity_note_line(mode: FidelityMode, locale: str, *, requested: bool = False) -> str:
    """The course-note line saying how the content was written; ``requested``: the description asked for it."""

    if locale == "vi":
        return _MODE_NOTE_VI[mode] + (" (theo yêu cầu trong mô tả khoá học)" if requested else "")
    return _MODE_NOTE_EN[mode] + (" (as the course description asks)" if requested else "")


__all__ = [
    "DEFAULT_MODE", "TEACHING_SUPPORT_CODE", "TEACHING_SUPPORT_KINDS", "TREATMENT_POLICY_CODE", "brief_mode",
    "budget_words", "description_asks_to_keep", "enforce_treatments", "example_label", "fidelity_note_line",
    "is_teaching_slot", "module_fidelity_rules", "plan_teaching_supports", "purpose_mode", "resolve_fidelity",
    "slot_marks", "teaching_slots", "w2_treatment_rules", "writer_fidelity_rules",
]
