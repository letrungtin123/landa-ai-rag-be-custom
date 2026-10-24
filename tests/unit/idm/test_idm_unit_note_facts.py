"""``build_unit_author_note(UnitNoteFacts)`` writes the note of the 40-keyword function it replaced (refactor wave 3b).

``reference_author_note`` is that function, frozen verbatim (app/idm/qa.py at 2a1c0cb) on the same helpers. Every note
the unit pipeline builds for the golden and replay fixtures (the units of the golden and the kept-document designs,
fallback-only and with the writer unavailable, the recorded writer answers and the malformed-answer probes of
tools/qc_replay.py), and one note per non-default fact in both locales, must be equal to the reference note.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import unittest
from collections.abc import Mapping, Sequence
from typing import Any
from unittest.mock import patch

from app.core.config import settings
from app.idm import storyboard
from app.idm.contracts import IdmJudgeFindingV1
from app.idm.policy import IDM_UNIT_AUTHOR_NOTE_MAX_CHARS, IDM_UNIT_SOURCE_FLAGS_MAX_CHARS
from app.idm.qa import (
    _CRITERION_LABEL_EN,
    _CRITERION_LABEL_VI,
    _NOTICE_LEAD,
    _REVIEW_LEAD,
    REVIEW_REASON_EXAMPLE_NUMBERS,
    REVIEW_REASON_JUDGE,
    REVIEW_REASON_KEPT_SLOTS,
    REVIEW_REASON_SCENARIO,
    REVIEW_REASON_SERVER_EDIT,
    REVIEW_REASON_SOURCE,
    REVIEW_REASON_STYLE,
    SCENARIO_LINE,
    SUGGESTED_CRITERIA_LINE,
    JudgeOutcome,
    SlotFinding,
    UnitNoteFacts,
    _acronyms_line,
    _check_lines,
    _Checks,
    _judge_line,
    _reasons_line,
    _slot_lines,
    build_unit_author_note,
    callout_lines,
    classroom_line,
    drafted_examples_line,
    dropped_faq_lines,
    example_numbers_line,
    faq_inferred_line,
    fit_flags,
    framework_line,
    glued_line,
    supposition_line,
    title_names_line,
    worksheet_line,
)
from app.idm.text import sanitize_author_text
from app.schemas.orchestration_v2 import RagLessonAuthorUnitV2Request
from app.services import provider as provider_service
from app.services.orchestration_v2 import unit as unit_service
from tests import idm_golden as g
from tests.golden.idm_golden_unit import build_unit_request
from tests.support.idm_test_support import golden_design, golden_shard
from tests.unit.idm.test_idm_preserve_design import all_unit_requests as preserve_unit_requests
from tests.unit.idm.test_idm_storyboard import (
    JUDGE,
    WRITER,
    FakeGenerate,
    escalate_body,
    escalate_writer,
    judge,
    severity_body,
    severity_writer,
    unavailable,
)


def reference_author_note(
    *,
    locale: str,
    judge: JudgeOutcome,
    deterministic_codes: Sequence[str],
    fallback_slots: Sequence[int],
    ai_drafted: bool,
    slot_reasons: Mapping[int, Sequence[str]] | None = None,
    review_slots: Sequence[int] = (),
    slot_types: Sequence[str] = (),
    failure_codes: Sequence[str] = (),
    whole_fallback: bool = False,
    faq_items_dropped: int = 0,
    faq_items_invalid: int = 0,
    remaining: Sequence[SlotFinding] | None = None,
    fixed_codes: Sequence[str] = (),
    faq_items_restated: int = 0,
    callouts_to_prose: int = 0,
    framework_list_inserted: int = 0,
    callouts_repeating: int = 0,
    worksheet_compacted: bool = False,
    worksheet_example_kept: bool = False,
    drafted_examples: Mapping[int, int] | None = None,
    example_numbers: Mapping[int, int] | None = None,
    faq_items_copied: int = 0,
    faq_items_recall: int = 0,
    faq_inferred: Mapping[int, int] | None = None,
    faq_items_open: int = 0,
    review_reasons: Sequence[str] = (),
    unexplained_acronyms: Sequence[str] = (),
    notices: Sequence[str] = (),
    suggested_criteria: bool = False,
    source_flags: Sequence[str] = (),
    source_notes: Sequence[str] = (),
    placeholder_slots: Sequence[int] = (),
    glued: Sequence[str] = (), title_names: Sequence[str] = (),
    suppositions: int = 0,
    fallback_inputs: Sequence[int] = (),
    hype_words: Mapping[int, Sequence[str]] | None = None,
    classroom: Mapping[int, Sequence[str]] | None = None,
    name_words: Mapping[int, Sequence[str]] | None = None,
) -> str:
    """``app.idm.qa.build_unit_author_note`` at 2a1c0cb, verbatim apart from this docstring."""

    vi = locale == "vi"
    problems = [finding for finding in judge.findings if finding.severity in {"major", "critical"}]
    judged = _judge_line(judge, problems, _CRITERION_LABEL_VI if vi else _CRITERION_LABEL_EN, vi)
    slot_lines, named = _slot_lines(vi, slot_reasons or {}, slot_types, whole_fallback=whole_fallback,
                                    fallback_slots=fallback_slots, review_slots=review_slots,
                                    failure_codes=failure_codes, deterministic_codes=deterministic_codes,
                                    placeholder_slots=placeholder_slots, remaining=remaining,
                                    fallback_inputs=fallback_inputs)
    checks = _Checks("", [], []) if whole_fallback else _check_lines(
        vi, deterministic_codes, remaining, fixed_codes, slot_types, named=named, hype_words=hype_words or {},
        name_words=name_words or {})
    # Run 5496155c: a reason says "chi tiết bên dưới" only when a line below gives the details.
    kept_line = bool(review_slots) and not whole_fallback
    detailed = {reason for reason, shown in (
        (REVIEW_REASON_SOURCE, bool(source_flags)), (REVIEW_REASON_JUDGE, bool(problems) and bool(judged)),
        (REVIEW_REASON_KEPT_SLOTS, kept_line), (REVIEW_REASON_STYLE, kept_line),
        (REVIEW_REASON_SERVER_EDIT, callouts_to_prose > 0 or framework_list_inserted > 0),
        (REVIEW_REASON_EXAMPLE_NUMBERS, bool(example_numbers) and not whole_fallback)) if shown}
    content = not whole_fallback
    # Review of dfb6db2: what the author must check first ("Cần xem"), then what to polish ("Lưu ý", "Nên biên tập"),
    # then what was done; the note's character budget only ever cuts the end.
    # The fallback and kept-slot lines lead a unit marked for review; a unit with notices only opens with them.
    first = bool(review_reasons or fallback_slots or whole_fallback)
    review = [_reasons_line(review_reasons, vi, detailed, _REVIEW_LEAD[vi]),
              *fit_flags(source_flags, IDM_UNIT_SOURCE_FLAGS_MAX_CHARS, locale), judged,
              *(slot_lines if first else ()), *checks.review]
    edits = [_reasons_line(notices, vi, detailed, _NOTICE_LEAD[vi]),
             *fit_flags(source_notes, IDM_UNIT_SOURCE_FLAGS_MAX_CHARS, locale, soft=True),
             *(() if first else slot_lines), *checks.polish,
             *dropped_faq_lines(vi, dropped=faq_items_dropped, restated=faq_items_restated, copied=faq_items_copied,
                                recall=faq_items_recall, unresolved=faq_items_open, invalid=faq_items_invalid),
             *callout_lines(vi, to_prose=callouts_to_prose, repeating=callouts_repeating),
             supposition_line(suppositions, vi) if suppositions else "",
             framework_line(framework_list_inserted, vi) if framework_list_inserted else "",
             worksheet_line(vi, example_kept=worksheet_example_kept) if worksheet_compacted else "",
             drafted_examples_line(drafted_examples, slot_types, vi) if drafted_examples and content else "",
             example_numbers_line(example_numbers, slot_types, vi) if example_numbers and content else "",
             faq_inferred_line(faq_inferred, slot_types, vi) if faq_inferred and content else "",
             # A source-locked unit carries no drafted scenario; the review reason already says it when it is one.
             SCENARIO_LINE[vi] if ai_drafted and content and REVIEW_REASON_SCENARIO not in review_reasons else "",
             _acronyms_line(unexplained_acronyms, vi) if unexplained_acronyms and content else "",
             SUGGESTED_CRITERIA_LINE[vi] if suggested_criteria else "", glued_line(glued, vi) if glued else "",
             classroom_line(classroom, slot_types, vi) if classroom and content else "",
             title_names_line(title_names, vi) if title_names and content else ""]
    # Run 2f8781d6 (A4): never "QA tự động: không phát hiện vấn đề lớn." (it stood on a unit whose question gave its
    # answer away); a clean unit's note says only what was fixed, or nothing.
    return sanitize_author_text(" ".join(part for part in [*review, *edits, checks.fixed] if part),
                                IDM_UNIT_AUTHOR_NOTE_MAX_CHARS)



def reference_of(facts: UnitNoteFacts) -> str:
    return reference_author_note(**{field.name: getattr(facts, field.name) for field in dataclasses.fields(facts)})


def run_unit(body: dict[str, Any], provider: Any) -> None:
    request = RagLessonAuthorUnitV2Request.model_validate(body)
    with patch.object(provider_service, "generate_content", provider), patch.object(settings, "idm_judge_mode",
                                                                                     "observe"):
        asyncio.run(unit_service.lesson_author_orchestration_v2_unit(request))


def fixture_runs() -> list[tuple[dict[str, Any], Any]]:
    """The unit requests and fake providers of tools/qc_replay.py, and the units of the kept-document design."""

    design, _skeleton = golden_design()
    runs: list[tuple[dict[str, Any], Any]] = []
    for chapter_index in range(len(design.modules)):
        shard = golden_shard(chapter_index)
        for lesson_position, lesson in enumerate(shard.lessons):
            for unit_position in range(len(lesson.units)):
                body = build_unit_request(design, shard, chapter_index=chapter_index, lesson_position=lesson_position,
                                          unit_position=unit_position, facts=g.source_facts())
                runs.append(({**copy.deepcopy(body), "fallback_only": True}, FakeGenerate()))
                runs.append((copy.deepcopy(body), FakeGenerate(**{WRITER: [unavailable()] * 4})))
    for body, writer in ((escalate_body(), escalate_writer), (severity_body(), severity_writer)):
        runs.append((body, FakeGenerate(**{WRITER: [writer(body)], JUDGE: [judge()]})))
    body = severity_body()
    good = severity_writer(body)
    components = good["components"]
    for answer in ([], {"components": [components["c0"], components["c1"]]},
                   {"components": {"c0": None, "c1": components["c1"]}},
                   {"components": {"c0": "text", "c1": components["c1"]}},
                   {"components": {**components, "c1": {**components["c1"], "choices": None}}},
                   {"components": {**components, "c0": {**components["c0"], "semantic_content": None}}},
                   {"components": {"c1": components["c1"]}}):
        runs.append((copy.deepcopy(body), FakeGenerate(**{WRITER: [copy.deepcopy(answer)] * 4, JUDGE: [judge()] * 2})))
    runs.extend(({**request, "fallback_only": True}, FakeGenerate()) for request in preserve_unit_requests())
    return runs


# One non-default value per optional fact; a fact added to UnitNoteFacts must get one here.
SAMPLES: dict[str, Any] = {
    "slot_reasons": {0: ["IDM_W5_NEW_FACT", "IDM_W5_SENTENCE_COPY"], 1: ["IDM_W5_ANSWER_LEAK"]},
    "review_slots": [1],
    "slot_types": ["html", "problem", "la_faq"],
    "failure_codes": ["IDM_W5_REPAIR_INVALID", "AI_PROVIDER_TIMEOUT"],
    "whole_fallback": True,
    "faq_items_dropped": 2,
    "faq_items_invalid": 1,
    "remaining": [SlotFinding("IDM_W5_SENTENCE_COPY", 0), SlotFinding("IDM_W5_NEW_FACT", 1)],
    "fixed_codes": ["IDM_W5_SENTENCE_LENGTH"],
    "faq_items_restated": 1,
    "callouts_to_prose": 2,
    "framework_list_inserted": 1,
    "callouts_repeating": 1,
    "worksheet_compacted": True,
    "worksheet_example_kept": True,
    "drafted_examples": {0: 2},
    "example_numbers": {0: 1},
    "faq_items_copied": 1,
    "faq_items_recall": 1,
    "faq_inferred": {2: 1},
    "faq_items_open": 1,
    "review_reasons": [REVIEW_REASON_SOURCE, REVIEW_REASON_JUDGE, REVIEW_REASON_KEPT_SLOTS, REVIEW_REASON_SCENARIO],
    "unexplained_acronyms": ["KPI", "CX"],
    "notices": [REVIEW_REASON_STYLE, REVIEW_REASON_EXAMPLE_NUMBERS, REVIEW_REASON_SERVER_EDIT],
    "suggested_criteria": True,
    "source_flags": ["Tài liệu ghi hai thời hạn khác nhau cho cùng một bước."],
    "source_notes": ["Tài liệu chưa nói rõ ai duyệt bước cuối."],
    "placeholder_slots": [0],
    "glued": ["kháchhàng"],
    "title_names": ["Ma trận thấu cảm"],
    "suppositions": 1,
    "fallback_inputs": [1],
    "hype_words": {0: ["tuyệt vời"]},
    "classroom": {0: ["Thực hành: chia nhóm"]},
    "name_words": {1: ["Anh Minh"]},
}


class UnitNoteFactsTests(unittest.TestCase):
    def test_the_pipeline_notes_of_every_fixture_equal_the_reference(self) -> None:
        built: list[UnitNoteFacts] = []

        def recording(facts: UnitNoteFacts) -> str:
            built.append(facts)
            return build_unit_author_note(facts)

        with patch.object(storyboard, "build_unit_author_note", recording):
            for body, provider in fixture_runs():
                run_unit(body, provider)
        self.assertGreaterEqual(len(built), 20)
        notes = [build_unit_author_note(facts) for facts in built]
        self.assertEqual(notes, [reference_of(facts) for facts in built])
        self.assertGreater(sum(1 for note in notes if note), 10)
        self.assertGreater(len(set(notes)), 5)

    def test_one_note_per_non_default_fact_equals_the_reference(self) -> None:
        optional = {field.name for field in dataclasses.fields(UnitNoteFacts)
                    if field.default is not dataclasses.MISSING}
        self.assertEqual(set(SAMPLES), optional)
        problems = JudgeOutcome("review_required", [
            IdmJudgeFindingV1(criterion="Q10_title_matches", severity="major", component_index=None, witness="w"),
            IdmJudgeFindingV1(criterion="Q2_not_copied", severity="critical", component_index=0, witness="w"),
            IdmJudgeFindingV1(criterion="Q8_language", severity="minor", component_index=1, witness="w")])
        checked = 0
        for locale in ("vi", "en"):
            for judged in (JudgeOutcome("pass"), JudgeOutcome("not_run"), problems):
                for drafted in (False, True):
                    base = UnitNoteFacts(locale=locale, judge=judged, deterministic_codes=["IDM_W5_SENTENCE_COPY"],
                                         fallback_slots=[0], ai_drafted=drafted, slot_types=["html", "problem"])
                    variants = [base, *(dataclasses.replace(base, **{name: value}) for name, value in SAMPLES.items()),
                                dataclasses.replace(base, **{name: value for name, value in SAMPLES.items()
                                                             if name != "whole_fallback"}),
                                dataclasses.replace(base, **SAMPLES)]
                    for facts in variants:
                        with self.subTest(locale=locale, judge=judged.status, drafted=drafted):
                            self.assertEqual(build_unit_author_note(facts), reference_of(facts))
                        checked += 1
        self.assertEqual(checked, 2 * 3 * 2 * (len(SAMPLES) + 3))


if __name__ == "__main__":
    unittest.main()
