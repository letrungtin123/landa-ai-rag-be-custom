"""W5 deterministic slot checks (spec §7.7.2(b)): answer leaks, copied or reworded options, FAQ items that
restate the html, ungrounded callouts, incomplete worksheets and framework lists, and the codes they report.

The W5 writer (``app.idm.storyboard``) repairs or settles every finding; the author note (``app.idm.qa``)
names the ones that remain.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any, Final

from app.content.learner_content_purity import component_learner_text
from app.idm.answer_checks import (
    contested_option,
    copies_example_measure,
    quoted_key,
    resolved_choices,
    restates_example,
    restates_joined_example,
    restates_list,
    stem_copies_answer,
    unit_teaching,
)
from app.idm.answer_near import ANSWER_ITEM_LENGTH_CODE, ANSWER_NEAR_EXAMPLE_CODE, item_length_cue, near_example
from app.idm.artifacts import cut_fragments
from app.idm.contracts import IdmUnitBriefV1
from app.idm.faq_checks import FaqChecker
from app.idm.fidelity_checks import FidelityChecker
from app.idm.framework import FrameworkPromise, framework_listed
from app.idm.mcq import (
    ANSWER_LENGTH_CUE_CODE,
    OPTION_LETTERS,
    QUESTION_INCOMPLETE_CODE,
    answer_length_cue,
    labelled_letters,
    question_complete,
)
from app.idm.policy import (
    FAQ_TITLE_GENERIC_WORDS,
    IDM_ANSWER_LEAK_MIN_NGRAMS,
    IDM_ANSWER_LEAK_MIN_SHARE,
    IDM_ANSWER_LEAK_NGRAM,
    IDM_ANSWER_LEAK_PAIR_MIN_HITS,
    IDM_ANSWER_LEAK_PAIR_MIN_MARGIN,
    IDM_ANSWER_LEAK_PAIR_MIN_SHARE,
    IDM_CALLOUT_REPEAT_MIN_WORDS,
    IDM_CALLOUT_REPEAT_NGRAM,
    IDM_CALLOUT_REPEAT_SHARE,
    IDM_FAQ_RESTATE_MIN_NGRAMS,
    IDM_FAQ_RESTATE_MIN_SHARE,
    IDM_FAQ_RESTATE_NGRAM,
    IDM_FAQ_TITLE_MIN_KEY_WORDS,
    IDM_FAQ_TITLE_MIN_SHARED_SHARE,
    IDM_WORKSHEET_MIN_GROUNDED_SHARE,
    IDM_WORKSHEET_MIN_LIST_ITEMS,
    WORKSHEET_COMPONENT_TYPE,
)
from app.idm.quiz_text import quiz_findings
from app.idm.source_names import NAME_NOT_IN_SOURCE_CODE, NameGuide
from app.idm.text import (
    EvidenceIndex,
    content_pairs,
    content_words,
    grounding_verdict,
    idm_fold,
    ngram_overlap,
    word_ngrams,
)
from app.idm.worksheet_input import EXAMPLE_REUSE_CODE, WORKSHEET_INPUT_CODE, missing_input, reused_entries

MIN_FEEDBACK_CHARS: Final = 60
# An option is named by its text when the explanation contains the start of it (accent-folded).
_OPTION_TEXT_PREFIX: Final = 40
VERBATIM_MIN_CHARS: Final = 300
VERBATIM_MAX_OVERLAP: Final = 0.7
ANSWER_LEAK_MIN_CHARS: Final = 12
# A key and at least one distractor to compare it with.
_MIN_LEAK_OPTIONS: Final = 2
FAQ_ITEMS_DROPPED_CODE: Final = "IDM_W5_FAQ_ITEMS_DROPPED"
WORKSHEET_INCOMPLETE_CODE: Final = "IDM_W5_WORKSHEET_INCOMPLETE"
ANSWER_LEAK_CODE: Final = "IDM_W5_ANSWER_LEAK"
# QC run ab8d67e1 (R2): the question's case meets another option's taught criteria as well as the key's.
ANSWER_CONTESTED_CODE: Final = "IDM_W5_ANSWER_CONTESTED"
# QC run ab8d67e1 (R5): a list item, row label or cell of the html ends cut as the PDF line did ("Best-in", "(KPI /").
SOURCE_ARTIFACT_CODE: Final = "IDM_W5_SOURCE_ARTIFACT"
# QC course 364564: N9 (a callout that reads as a quotation states what the facts do not), N11 (FAQ items
# that repeat the html above; an FAQ title that does not match its questions).
CALLOUT_UNGROUNDED_CODE: Final = "IDM_W5_CALLOUT_UNGROUNDED"
CALLOUT_TO_PROSE_CODE: Final = "IDM_W5_CALLOUT_TO_PROSE"
FAQ_RESTATES_HTML_CODE: Final = "IDM_W5_FAQ_RESTATES_HTML"
FAQ_TITLE_MISMATCH_CODE: Final = "IDM_W5_FAQ_TITLE_MISMATCH"
# N5: the unit or html slot title names "N <items>" ("Tổng quan 5 chuyển dịch") but the html does not list them.
FRAMEWORK_INCOMPLETE_CODE: Final = "IDM_W5_FRAMEWORK_INCOMPLETE"
# QC run 8de1c76b (Q3): the list of the framework's items inserted by the server when the repair did not add it.
FRAMEWORK_LIST_INSERTED_CODE: Final = "IDM_W5_FRAMEWORK_LIST_INSERTED"
# A warning block renders as <blockquote> (app.idm.node_acceptance); "callout"/"note" are its aliases.
CALLOUT_KIND: Final = "warning"
# The html fix that counts the callouts dropped for repeating their slot (QC feedback A3).
CALLOUT_REPEAT_DROPPED: Final = "callout_repeat_dropped"
# What makes a callout a new point (review M3): a number, or a negation or exception word.
_NUMBER_RE: Final = re.compile(r"\d+(?:[.,:]\d+)*")
_NEGATION_RE: Final = re.compile(
    r"(?<!\w)(?:không|chưa|chẳng|cấm|đừng|ngoại trừ|trừ khi|trừ|not|no|never|except|unless|cannot|without|"
    r"don't|doesn't|do not|does not)(?!\w)")


@dataclass(frozen=True)
class SlotFinding:
    code: str
    component_index: int


def _visible(component: dict[str, Any]) -> str:
    return " ".join(component_learner_text(component).split())


def _correct_choices(component: dict[str, Any]) -> list[dict[str, Any]]:
    choices = component.get("choices")
    if not isinstance(choices, list):
        return []
    return [choice for choice in choices if isinstance(choice, dict) and choice.get("correct") is True]


def unexplained_options(component: dict[str, Any]) -> list[int]:
    """Options the explanation never names, by letter or by their own text (QC course 364564, N10).

    The prompt asks for "A - ...; B - ..." with every option; an explanation that skips one cannot
    tell the learner why that option is right or wrong. Letters are read as ``app.idm.mcq`` reads them
    when it relabels the explanation after the seeded shuffle, so they are the served letters.
    """

    explanation = str(component.get("explanation") or "")
    choices = [choice for choice in component.get("choices") or [] if isinstance(choice, dict)]
    letters = labelled_letters(explanation)
    folded = idm_fold(explanation)
    missing = []
    for index, choice in enumerate(choices):
        text = idm_fold(str(choice.get("text") or ""))
        by_letter = index < len(OPTION_LETTERS) and OPTION_LETTERS[index] in letters
        if not by_letter and not (len(text) >= ANSWER_LEAK_MIN_CHARS and text[:_OPTION_TEXT_PREFIX] in folded):
            missing.append(index)
    return missing


def html_before(unit: dict[str, Any], index: int) -> str:
    """Visible text of the html slots before slot ``index``: what the learner has just read."""

    components = [item for item in unit.get("components", []) if isinstance(item, dict)]
    return " ".join(_visible(component) for component in components[:index] if component.get("type") == "html")


def restated_faq_items(component: dict[str, Any], preceding_html: str) -> list[int]:
    """FAQ items whose answer repeats the html shown before them in the unit (QC course 364564, N11).

    Same measure as the answer leak (accent-folded word 4-grams), with its own calibration
    (``IDM_FAQ_RESTATE_*``): an FAQ answers a misconception, an edge case or a "what if"; one that only
    restates a row or paragraph the learner has just read adds nothing.
    """

    items = component.get("items")
    if not isinstance(items, list) or not preceding_html.strip():
        return []
    answers = [str(item.get("answer") or "") if isinstance(item, dict) else "" for item in items]
    return [index for index, answer in enumerate(answers)
            if len(word_ngrams(answer, IDM_FAQ_RESTATE_NGRAM)) >= IDM_FAQ_RESTATE_MIN_NGRAMS
            and ngram_overlap(answer, preceding_html, IDM_FAQ_RESTATE_NGRAM) >= IDM_FAQ_RESTATE_MIN_SHARE]


def faq_title_mismatch(component: dict[str, Any], taught: str = "") -> bool:
    """The FAQ title's key words (content words that are not FAQ boilerplate) are mostly absent from its
    questions and answers ("… trong họp giao ban" over items about something else; QC 364564, N11).

    ``taught`` (QLT-9): what the unit speaks of besides the FAQ. A title that names the unit's topic in the unit's words
    ("Giải đáp về đối sánh nội địa …" over items about "đối thủ trong nước", QC run ffaf7318: 2 of 2 findings were
    false alarms) matches; only a title about something neither the items nor the unit speak of does not. Since
    2026-10-20 (owner P7) the unit is every other slot (the html after the FAQ and the question too) and the facts the
    FAQ is built on (``unit_context``); a title word counts as the unit's only inside a word pair the unit writes too
    (Vietnamese words are syllables: the "giao" of "đơn hàng giao trễ" is not the "giao" of "họp giao ban")."""

    items = component.get("items")
    title = str(component.get("title") or "")
    key = set(content_words(title)) - FAQ_TITLE_GENERIC_WORDS
    if not isinstance(items, list) or len(key) < IDM_FAQ_TITLE_MIN_KEY_WORDS:
        return False
    said = {word for item in items if isinstance(item, dict)
            for word in content_words(f"{item.get('question') or ''} {item.get('answer') or ''}")}
    pairs = set(word_ngrams(taught, 2))
    said |= {word for pair in word_ngrams(title, 2) if pair in pairs for word in pair}
    return len(key & said) < IDM_FAQ_TITLE_MIN_SHARED_SHARE * len(key)


def unit_context(components: Sequence[dict[str, Any]], index: int, faq_check: FaqChecker | None) -> str:
    """The learner text of every slot of the unit but ``index`` and the facts slot ``index`` is built on."""

    texts = [_visible(component) for position, component in enumerate(components) if position != index]
    return " ".join([*texts, *(faq_check.slot_texts(index) if faq_check is not None else ())])


def _sections(component: dict[str, Any]) -> list[dict[str, Any]]:
    semantic = component.get("semantic_content")
    sections = semantic.get("sections") if isinstance(semantic, dict) else None
    return [section for section in sections if isinstance(section, dict)] if isinstance(sections, list) else []


def ungrounded_callouts(component: dict[str, Any], evidence: EvidenceIndex) -> list[tuple[int, int]]:
    """(section, block) of every callout whose text does not restate the facts (QC course 364564, N9).

    A warning block renders as a blockquote that learners read as a quotation or an official rule; 3 of
    7 were sentences the writer made up. The test is the FAQ grounding verdict.
    """

    return [(section_index, block_index)
            for section_index, section in enumerate(_sections(component))
            for block_index, block in enumerate(section.get("blocks") or [])
            if isinstance(block, dict) and block.get("kind") == CALLOUT_KIND
            and grounding_verdict(str(block.get("text") or ""), evidence) != "grounded"]


def callouts_as_paragraphs(component: dict[str, Any], positions: Collection[tuple[int, int]]) -> dict[str, Any]:
    """``component`` with the callouts at ``positions`` turned into plain paragraphs (same text)."""

    semantic = component.get("semantic_content")
    if not isinstance(semantic, dict) or not positions:
        return component
    sections = []
    for section_index, section in enumerate(_sections(component)):
        blocks = [{**block, "kind": "paragraph"}
                  if (section_index, block_index) in positions and isinstance(block, dict) else block
                  for block_index, block in enumerate(section.get("blocks") or [])]
        sections.append({**section, "blocks": blocks})
    return {**component, "semantic_content": {**semantic, "sections": sections}}


def _block_pieces(block: dict[str, Any]) -> list[str]:
    """The visible text pieces of one semantic block: its text, each list item, each table row."""

    rows = [f"{row.get('label') or ''} {row.get('value') or ''}" for row in block.get("rows") or []
            if isinstance(row, dict)]
    return [piece for piece in [str(block.get("text") or ""), *map(str, block.get("items") or []), *rows]
            if piece.strip()]


def _new_point(callout: str, before: str) -> bool:
    """The callout states a number or a negation that ``before`` does not (review M3): a new point, never a repeat."""

    numbers = set(_NUMBER_RE.findall(before))
    negations = set(_NEGATION_RE.findall(before.lower()))
    return (any(number not in numbers for number in _NUMBER_RE.findall(callout))
            or any(word not in negations for word in _NEGATION_RE.findall(callout.lower())))


def repeated_callouts(semantic: Any) -> list[tuple[int, int]]:
    """(section, block) of every callout that only repeats the paragraphs before it in its section (QC feedback
    2026-10-21, A3: a quotation or callout box adds a new point, never the paragraph above it). Review M3: a callout
    with its own number or negation is kept (``_new_point``); otherwise it repeats when at least
    ``IDM_CALLOUT_REPEAT_SHARE`` of its word trigrams are in the blocks before it in the same section (paragraphs, list
    items, table rows and the callouts kept before it)."""

    sections = semantic.get("sections") if isinstance(semantic, dict) else None
    if not isinstance(sections, list):
        return []
    found: list[tuple[int, int]] = []
    for section_index, section in enumerate(sections):
        blocks = section.get("blocks") if isinstance(section, dict) else None
        before: list[str] = []
        for block_index, block in enumerate(blocks if isinstance(blocks, list) else []):
            if not isinstance(block, dict):
                continue
            text = str(block.get("text") or "")
            if (block.get("kind") == CALLOUT_KIND and len(text.split()) >= IDM_CALLOUT_REPEAT_MIN_WORDS and before
                    and not _new_point(text, " ".join(before))):
                known = {gram for piece in before for gram in word_ngrams(piece, IDM_CALLOUT_REPEAT_NGRAM)}
                grams = word_ngrams(text, IDM_CALLOUT_REPEAT_NGRAM)
                if grams and sum(gram in known for gram in grams) / len(grams) >= IDM_CALLOUT_REPEAT_SHARE:
                    found.append((section_index, block_index))
                    continue
            before.extend(_block_pieces(block))
    return found


def without_blocks(semantic: dict[str, Any], positions: Collection[tuple[int, int]]) -> dict[str, Any] | None:
    """``semantic`` without the blocks at ``positions`` and without a section they leave empty; ``None`` when no
    section would remain."""

    sections = []
    for section_index, section in enumerate(semantic.get("sections") or []):
        if not isinstance(section, dict):
            sections.append(section)
            continue
        blocks = [block for block_index, block in enumerate(section.get("blocks") or [])
                  if (section_index, block_index) not in positions]
        if blocks:
            sections.append({**section, "blocks": blocks})
    return {**semantic, "sections": sections} if sections else None


def _worksheet_blocks(component: dict[str, Any]) -> list[dict[str, Any]]:
    return [block for section in _sections(component)
            for block in section.get("blocks") or [] if isinstance(block, dict)]


def worksheet_complete(component: dict[str, Any], evidence: EvidenceIndex | None) -> bool:
    """A worksheet (html slot with role practice) states the task, gives the template the learner fills
    in (table rows or steps) and ends with a self-check list whose items come from the criteria facts."""

    blocks = _worksheet_blocks(component)

    def entries(block: dict[str, Any], field_name: str) -> list[Any]:
        value = block.get(field_name)
        return value if isinstance(value, list) else []

    task = any(block.get("kind") == "task" and str(block.get("text") or "").strip() for block in blocks)
    template = any((block.get("kind") == "table" and len(entries(block, "rows")) >= IDM_WORKSHEET_MIN_LIST_ITEMS)
                   or (block.get("kind") == "steps" and len(entries(block, "items")) >= IDM_WORKSHEET_MIN_LIST_ITEMS)
                   for block in blocks)
    checklists = [entries(block, "items") for block in blocks
                  if block.get("kind") == "bullets" and len(entries(block, "items")) >= IDM_WORKSHEET_MIN_LIST_ITEMS]
    if not (task and template and checklists):
        return False
    if evidence is None:
        return True
    items = [str(item) for item in checklists[-1]]
    grounded = sum(grounding_verdict(item, evidence) == "grounded" for item in items)
    return grounded >= IDM_WORKSHEET_MIN_GROUNDED_SHARE * len(items)


def deterministic_slot_findings(
    unit: dict[str, Any], brief: IdmUnitBriefV1, owned_text_by_slot: Sequence[str],
    evidence: EvidenceIndex | None = None, framework: FrameworkPromise | None = None, *,
    fidelity: FidelityChecker | None = None, faq_check: FaqChecker | None = None,
    fact_rows: Sequence[tuple[str, str]] = (), names: NameGuide | None = None,
) -> list[SlotFinding]:
    """IDM checks on top of the shared staged validator (spec §7.7.2(b)).

    With ``evidence`` (the facts the writer was given), callouts must restate them and a worksheet's self-check
    list must come from them. With ``framework`` (the "N <items>" a title of the unit promises), the unit's
    teaching html must list every item (reported on its first html slot). With ``fidelity`` (QLT-5 C), the
    teaching html slots follow the unit's content mode (``fidelity_checks``); the slot-level verbatim check then
    only runs in the structured mode, as before. With ``faq_check`` (QLT-5 D), an la_faq slot mixes its question types,
    does not copy a source fact and states no number or name its basis facts do not (``faq_checks``). ``fact_rows``
    (QLT-7): the table rows of the facts the writer was given, read with the html rows for a contested key.
    """

    slot_copy_check = fidelity is None or fidelity.mode == "structured"

    components = [item for item in unit.get("components", []) if isinstance(item, dict)]
    findings: list[SlotFinding] = []
    preceding_html = ""
    # The FAQ answers of the unit, wherever the FAQ sits: a key must agree with them (QC run ab8d67e1, R2).
    faq = [item for component in components if component.get("type") == "la_faq"
           for item in component.get("items") or [] if isinstance(item, dict)]
    for index, component in enumerate(components):
        kind = component.get("type")
        slot = brief.components[index] if index < len(brief.components) else None
        # Run 2f8781d6 (B9): a tool named in a title or heading by words the source does not use ("ma trận thấu cảm").
        if names is not None and names.slot_names(component):
            findings.append(SlotFinding(NAME_NOT_IN_SOURCE_CODE, index))
        if kind == "html":
            text = _visible(component)
            preceding_html += " " + text
            treatments = {item.treatment for item in slot.treatments} if slot else set()
            if (slot_copy_check and len(text) >= VERBATIM_MIN_CHARS and treatments and treatments != {"keep"}
                    and index < len(owned_text_by_slot)
                    and ngram_overlap(text, owned_text_by_slot[index]) > VERBATIM_MAX_OVERLAP):
                findings.append(SlotFinding("IDM_W5_VERBATIM_COPY", index))
            worksheet = slot is not None and slot.role == "practice" and slot.type == WORKSHEET_COMPONENT_TYPE
            if worksheet and not worksheet_complete(component, evidence):
                findings.append(SlotFinding(WORKSHEET_INCOMPLETE_CODE, index))
            # Run 5496155c: a worksheet whose task refers to a list, an interview or requests it does not give.
            if worksheet and missing_input(component) is not None:
                findings.append(SlotFinding(WORKSHEET_INPUT_CODE, index))
            # R2 of the paid verification of 3ed380d: a worked example on the task's own data answers the task.
            if worksheet and reused_entries(component):
                findings.append(SlotFinding(EXAMPLE_REUSE_CODE, index))
            if evidence is not None and ungrounded_callouts(component, evidence):
                findings.append(SlotFinding(CALLOUT_UNGROUNDED_CODE, index))
            if cut_fragments(component.get("semantic_content")):
                findings.append(SlotFinding(SOURCE_ARTIFACT_CODE, index))
        elif kind == "la_faq":
            if faq_check is not None:
                findings.extend(SlotFinding(code, index) for code in faq_check.findings(index, component))
            if restated_faq_items(component, preceding_html):
                findings.append(SlotFinding(FAQ_RESTATES_HTML_CODE, index))
            if faq_title_mismatch(component, unit_context(components, index, faq_check)):
                findings.append(SlotFinding(FAQ_TITLE_MISMATCH_CODE, index))
        elif kind == "problem":
            explanation = " ".join(str(component.get("explanation") or "").split())
            correct = _correct_choices(component)
            incomplete = len(correct) != 1 or bool(unexplained_options(component))
            if incomplete:
                findings.append(SlotFinding("IDM_W5_PRACTICE_INCOMPLETE", index))
            # QLT-8: a case without its question (run 0a11aa9e c1.l2.u4, cut by a repair) cannot be answered.
            if not question_complete(component.get("question")):
                findings.append(SlotFinding(QUESTION_INCOMPLETE_CODE, index))
            not_teaching = len(explanation) < MIN_FEEDBACK_CHARS or idm_fold(explanation).strip(" .!") in {
                "dung", "sai", "correct", "incorrect", "dung roi", "chua dung"}
            # QC feedback 2026-10-21 (A1/A2): the stem and the explanation in their lines, short and with a reason (an
            # explanation already found incomplete or not teaching is repaired for that first).
            findings.extend(SlotFinding(code, index) for code in quiz_findings(
                component, evidence, explanation=not incomplete and not not_teaching))
            if not_teaching:
                findings.append(SlotFinding("IDM_W5_FEEDBACK_NOT_TEACHING", index))
            if correct:
                answer = idm_fold(str(correct[0].get("text") or ""))
                html = idm_fold(preceding_html)
                teaching = unit_teaching([item for item in components[:index] if item.get("type") == "html"], faq,
                                         fact_rows)
                # Run 5496155c: an option that is only a listed item's name is measured by that item's text.
                read = resolved_choices(component)
                if ((len(answer) >= ANSWER_LEAK_MIN_CHARS and answer in html and ("dap an" in html or "answer" in html))
                        or copied_options(read, preceding_html) or reworded_key(read, preceding_html)
                        or quoted_key(read, teaching) or restates_example(read, teaching)
                        or restates_list(read, teaching) or restates_joined_example(read, teaching)
                        or stem_copies_answer(read, teaching, component) or copies_example_measure(read, teaching)):
                    findings.append(SlotFinding(ANSWER_LEAK_CODE, index))
                # Run 2f8781d6 (B6): a key the words right above give away, short of a leak: a soft finding.
                elif near_example(component, [item for item in components[:index] if item.get("type") == "html"]):
                    findings.append(SlotFinding(ANSWER_NEAR_EXAMPLE_CODE, index))
                if contested_option(component, teaching) is not None:
                    findings.append(SlotFinding(ANSWER_CONTESTED_CODE, index))
    # The overview is taught, not practised: a worksheet html slot neither promises nor lists it.
    html_slots = [index for index, component in enumerate(components) if component.get("type") == "html"
                  and not (index < len(brief.components) and brief.components[index].role == "practice")]
    if framework is not None and html_slots and not framework_listed(
            framework, [components[index] for index in html_slots], preceding_html):
        findings.append(SlotFinding(FRAMEWORK_INCOMPLETE_CODE, html_slots[0]))
    if fidelity is not None:
        findings.extend(SlotFinding(code, index) for code, index in fidelity.findings(components))
    return findings


def copied_options(component: dict[str, Any], preceding_text: str) -> list[int]:
    """Options that copy the html shown before the question, when that tells the answer (QC 364564, N3).

    An option copies the text when most of its word 4-grams (accent-folded) appear in it
    (``IDM_ANSWER_LEAK_*``): the correct option copying the worked example ("ví dụ đạt chuẩn"), a
    distractor copying the bad example. When every option copies the text (a recognition question
    over a list that was taught), matching tells nothing and nothing is returned.
    """

    choices = [choice for choice in component.get("choices") or [] if isinstance(choice, dict)]
    if not preceding_text.strip() or not choices:
        return []
    copied = [index for index, choice in enumerate(choices)
              if len(word_ngrams(str(choice.get("text") or ""), IDM_ANSWER_LEAK_NGRAM)) >= IDM_ANSWER_LEAK_MIN_NGRAMS
              and ngram_overlap(str(choice.get("text") or ""), preceding_text,
                                IDM_ANSWER_LEAK_NGRAM) >= IDM_ANSWER_LEAK_MIN_SHARE]
    return copied if len(copied) < len(choices) else []


def _distinctive_share(pairs: set[tuple[str, str]], others: Sequence[set[tuple[str, str]]],
                       question: set[tuple[str, str]], preceding: set[tuple[str, str]]) -> tuple[float, int]:
    distinctive = pairs - question - set().union(*others)
    hits = len(distinctive & preceding)
    return (hits / len(distinctive) if distinctive else 0.0), hits


def reworded_key(component: dict[str, Any], preceding_text: str) -> bool:
    """The correct option rewords the html shown before the question (QC run 8de1c76b, Q4).

    The 4-gram check misses a key that restates the example or quotation in other words ("Nếu không có hành
    động thì thay đổi tư duy chỉ là cảm hứng nhất thời …" after the blockquote it paraphrases). Here the key's
    distinctive content-word pairs (not in any distractor nor in the question) are looked up in the html before
    it: a leak when enough of them are there and clearly more than for any distractor
    (``IDM_ANSWER_LEAK_PAIR_*``, calibrated on the run's 9 questions).
    """

    choices = [choice for choice in component.get("choices") or [] if isinstance(choice, dict)]
    correct = [index for index, choice in enumerate(choices) if choice.get("correct") is True]
    if len(correct) != 1 or len(choices) < _MIN_LEAK_OPTIONS or not preceding_text.strip():
        return False
    preceding = content_pairs(preceding_text)
    question = content_pairs(str(component.get("question") or ""))
    pairs = [content_pairs(str(choice.get("text") or "")) for choice in choices]

    def share(index: int) -> tuple[float, int]:
        return _distinctive_share(pairs[index], [item for position, item in enumerate(pairs) if position != index],
                                  question, preceding)

    key_share, hits = share(correct[0])
    others = [share(index)[0] for index in range(len(choices)) if index != correct[0]]
    return (hits >= IDM_ANSWER_LEAK_PAIR_MIN_HITS and key_share >= IDM_ANSWER_LEAK_PAIR_MIN_SHARE
            and key_share - max(others) >= IDM_ANSWER_LEAK_PAIR_MIN_MARGIN)


def advisory_slot_findings(unit: dict[str, Any], exempt: Collection[int] = ()) -> list[SlotFinding]:
    """Review notes that never trigger a fallback (one targeted repair, then a note): the length cue of a question
    (N2), and the key's listed item clearly the longest item of the case (run 2f8781d6, B6)."""

    found: list[SlotFinding] = []
    for index, component in enumerate(item for item in unit.get("components", []) if isinstance(item, dict)):
        if index in exempt or component.get("type") != "problem":
            continue
        if answer_length_cue(component):
            found.append(SlotFinding(ANSWER_LENGTH_CUE_CODE, index))
        elif item_length_cue(component) is not None:
            found.append(SlotFinding(ANSWER_ITEM_LENGTH_CODE, index))
    return found


def final_unit_findings(unit: dict[str, Any], brief: IdmUnitBriefV1, owned_text_by_slot: Sequence[str],
                        evidence: EvidenceIndex | None, fallback_slots: Collection[int], *,
                        framework: FrameworkPromise | None = None, fidelity: FidelityChecker | None = None,
                        faq: FaqChecker | None = None, fact_rows: Sequence[tuple[str, str]] = (),
                        names: NameGuide | None = None) -> list[SlotFinding]:
    """What the returned unit still has (QC course 364564, N8): the IDM checks of its provider slots plus the
    review-only notes. A code seen before a repair is not in it unless the final unit still fails it."""

    exempt = set(fallback_slots)
    found = [item for item in deterministic_slot_findings(unit, brief, owned_text_by_slot, evidence, framework,
                                                          fidelity=fidelity, faq_check=faq, fact_rows=fact_rows,
                                                          names=names)
             if item.component_index not in exempt]
    return [*found, *advisory_slot_findings(unit, exempt)]
