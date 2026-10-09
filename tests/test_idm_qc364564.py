"""QC course 364564 (2026-10-08) regressions through the real ``/unit`` route and the IDM helpers.

Package A (deterministic MCQ and note fixes):

* N2: the correct option was the longest in 8 of 9 questions (a length cue, review note only).
* N3: an option copied the worked example of the worksheet shown just before the question.
* N8: codes found before a successful repair were printed as "còn cảnh báo" on a unit that passed.

Package B (prompts, judge and grounding):

* N7: the W6 judge rejected two teach-only units for practice criteria (it saw the lesson's practice).
* N9: three blockquotes (warning blocks) looked like quotations but stated what the source never says.
* N10: an explanation contradicted its option; explanations must name every option.
* N11: FAQ items repeated the table just taught; two FAQ titles did not match their questions.
* N4: the block defining BiC (the course's "BiC 5.0") was removed as history and the author was not told.
* N5: "5 chuyển dịch" was never presented as a whole; the unit titled "Tổng quan 5 chuyển dịch" taught
  something else; the source's "HÀNH ĐỘNG CEO" items were not Must Do candidates.

The provider is the local ``FakeGenerate``; nothing reaches the network.
"""

from __future__ import annotations

import dataclasses
import json
import unittest
from typing import Any
from unittest.mock import AsyncMock

from app.idm.blueprint import KeptDefinition, keep_term_definitions
from app.idm.content_map import action_items, reduce_catalog, validate_w1_reduce
from app.idm.contracts import (
    IdmCourseDesignV1,
    IdmJudgeFindingV1,
    IdmLessonDesignV1,
    IdmProjectContextV1,
    IdmUnitBriefV1,
)
from app.idm.framework import build_promise, framework_coverage, framework_listed, framework_promise
from app.idm.mcq import labelled_letters, normalize_single_choice
from app.idm.module_design import with_orientation_note
from app.idm.notes import KeptDefinitionNote, SkippedBlockNote, build_course_notes
from app.idm.qa import (
    JudgeOutcome,
    SlotFinding,
    build_unit_author_note,
    callouts_as_paragraphs,
    criteria_summary,
    deterministic_slot_findings,
    faq_title_mismatch,
    html_before,
    restated_faq_items,
    settle_applicability,
    unexplained_options,
    ungrounded_callouts,
)
from app.idm.storyboard import run_idm_unit
from app.idm.terms import course_terms, defined_terms, defines, mentions
from app.idm.text import evidence_index
from app.schemas.common import AiUsage
from app.schemas.orchestration_v2 import RagLessonAuthorUnitV2Request
from app.services.orchestration_v2 import unit as unit_service
from tests import idm_golden as g
from tests import idm_golden_unit as gu
from tests.idm_contract_bridge import writer_answer
from tests.idm_test_support import golden_design, golden_shard, make_runtime, module_response
from tests.test_idm_content_map import CONTEXT as CONTENT_CONTEXT
from tests.test_idm_content_map import golden_blocks as golden_content_blocks
from tests.test_idm_content_map import reduce_response as golden_reduce_response
from tests.test_idm_course_design import golden_design as golden_course_design
from tests.test_idm_course_design import golden_request as golden_course_request
from tests.test_idm_course_design import run_design as run_course_design
from tests.test_idm_module_design import design_module
from tests.test_idm_storyboard import (
    JUDGE,
    REPAIR,
    WRITER,
    FakeGenerate,
    StoryboardEndpointTestCase,
    body_for,
    escalate_body,
    escalate_writer,
    judge,
    severity_body,
    severity_writer,
    slot_repair,
    worksheet_body,
    worksheet_writer,
)

EXAMPLE_COPY = "Ví dụ đạt chuẩn: Cấp 2 vì khách hàng phàn nàn lần thứ hai."


def copying_worksheet(body: dict[str, Any]) -> dict[str, Any]:
    """The worksheet's worked example states the correct option of the problem that checks it."""

    writer = worksheet_writer(body)
    sections = writer["components"]["c0"]["semantic_content"]["sections"]
    sections[2]["blocks"][0]["text"] = EXAMPLE_COPY
    return writer


class AnswerLeakTests(StoryboardEndpointTestCase):
    async def test_option_copying_the_worked_example_is_repaired(self) -> None:
        body = worksheet_body()
        writer = copying_worksheet(body)
        fixed = slot_repair(writer, 1, choices=[
            {"text": "Cấp 1 vì chưa có thiệt hại tài chính", "correct": False},
            {"text": "Cấp 2, vì cùng một khách đã gọi phàn nàn lại", "correct": True},
            {"text": "Cấp 3 vì cần escalate ngay cho quản lý", "correct": False}])
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [fixed], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "validated", "provider")
        self.assertEqual(provider.names, [WRITER, REPAIR, JUDGE])
        self.assertIn('{"code":"IDM_W5_ANSWER_LEAK","path":"components[1]"}', provider.calls[1]["prompt"])
        # The repair targets the copying option, never the worksheet shown before it (package A report).
        self.assertIn("c1 IDM_W5_ANSWER_LEAK at c1: an option of this question copies or restates the example, "
                      "quotation, list item, table row or template shown before it", provider.calls[1]["prompt"])
        self.assertIn("rewrite THAT option", provider.calls[1]["prompt"])
        self.assertIn("Đã tự sửa: IDM_W5_ANSWER_LEAK.", quality.author_note)
        self.assertNotIn("còn cảnh báo", quality.author_note)
        self.assertEqual(quality.deterministic_codes, ["IDM_W5_ANSWER_LEAK"])

    async def test_leak_the_repair_cannot_fix_is_kept_for_review_with_a_note(self) -> None:
        body = worksheet_body()
        writer = copying_worksheet(body)
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [slot_repair(writer, 1)], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "review_required", "provider")
        self.assertNotIn("source_locked_fallback", data["unit"]["components"][1])
        self.assertIn("Giữ bản AI để tác giả rà soát: khối 2 (Quiz) — IDM_W5_ANSWER_LEAK.", quality.author_note)
        self.assertIn("Cần xem (IDM_W5_ANSWER_LEAK, khối 2 (Quiz)): một phương án gần như chép lại ví dụ",
                      quality.author_note)
        self.assertNotIn("Đã tự sửa", quality.author_note)


class LengthCueTests(StoryboardEndpointTestCase):
    async def test_length_cue_the_repair_leaves_is_a_review_note(self) -> None:
        # QC run 8de1c76b (Q4): the cue gets one targeted repair; when the options stay as they were, it is the
        # review note of QLT-2 (N2), never a fallback.
        body = escalate_body()
        writer = escalate_writer(body)
        writer["components"]["c0"]["choices"][0]["text"] = "Hứa hoàn tiền ngay để giữ chân khách VIP"
        writer["components"]["c0"]["choices"][2]["text"] = "Chỉ thông báo trưởng nhóm vì đây là khách VIP"
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [slot_repair(writer, 0)], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "validated", "provider")
        self.assertEqual(provider.names, [WRITER, REPAIR, JUDGE])
        self.assertEqual(provider.calls[1]["thinking_level"], "low")
        self.assertIn("c0 IDM_W5_ANSWER_LENGTH_CUE at c0.choices: the correct option is much longer than every other "
                      "option", provider.calls[1]["prompt"])
        self.assertIn("(the correct option has 60 characters, the longest other option 45)",
                      provider.calls[1]["prompt"])
        self.assertEqual(quality.deterministic_codes, ["IDM_W5_ANSWER_LENGTH_CUE"])
        self.assertIn("Cần xem (IDM_W5_ANSWER_LENGTH_CUE, khối 1 (Quiz)): đáp án đúng dài hơn hẳn",
                      quality.author_note)
        self.assertNotIn("còn cảnh báo", quality.author_note)


class FinalCodeTests(StoryboardEndpointTestCase):
    async def test_codes_fixed_by_the_repair_are_not_reported_as_remaining(self) -> None:
        # QC course 364564 (N8): "thực trạng 30" reads as a page locator to Node; the repair removed it and the
        # incomplete explanation, yet the note said both still warned.
        body = severity_body()
        good = severity_writer(body)
        writer = severity_writer(body, "Tiêu chí: khách hàng phàn nàn lần thứ hai là dấu hiệu của cấp 2 nên cần "
                                       "thông báo trưởng nhóm ngay.")
        paragraph = writer["components"]["c0"]["semantic_content"]["sections"][0]["blocks"][0]
        paragraph["text"] = "Thực trạng 30 khiếu nại mỗi tháng cho thấy cần phân cấp. " + paragraph["text"]
        repair = {"components": {**slot_repair(good, 0)["components"], **slot_repair(good, 1)["components"]}}
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [repair], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "validated", "provider")
        self.assertEqual(provider.names, [WRITER, REPAIR, JUDGE])
        self.assertEqual(quality.deterministic_codes, ["HTML_SOURCE_LOCATOR", "IDM_W5_PRACTICE_INCOMPLETE"])
        self.assertEqual(quality.author_note, "QA tự động: không phát hiện vấn đề lớn. "
                                              "Đã tự sửa: HTML_SOURCE_LOCATOR, IDM_W5_PRACTICE_INCOMPLETE. "
                                              "Tình huống minh hoạ do AI soạn, cần SME xác nhận tính thực tế.")

    async def test_english_note_lists_fixed_codes(self) -> None:
        body = {**severity_body(), "locale": "en"}
        good = severity_writer(body)
        writer = severity_writer(body, "Tiêu chí: khách hàng phàn nàn lần thứ hai là dấu hiệu của cấp 2 nên cần "
                                       "thông báo trưởng nhóm ngay.")
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [slot_repair(good, 1)], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "validated", "provider")
        self.assertIn("Automatically fixed: IDM_W5_PRACTICE_INCOMPLETE.", quality.author_note)
        self.assertNotIn("still warn", quality.author_note)


def teach_only_body() -> dict[str, Any]:
    """lsn_003 unit 1: one html slot that teaches the escalation criteria; the lesson's practice is in unit 2."""

    return body_for(1, 0, 0)


def finding(criterion: str, severity: str, index: int | None = 0) -> IdmJudgeFindingV1:
    return IdmJudgeFindingV1.model_validate({"criterion": criterion, "severity": severity,
                                             "component_index": index, "witness": "w"})


class JudgeScopeTests(StoryboardEndpointTestCase):
    """N7: the judge grades the unit's own practice; a teach-only unit has no practice criteria to fail."""

    async def test_teach_only_unit_has_no_practice_findings(self) -> None:
        # The QC false positive: Q3/Q4/Q6 critical on a unit without a practice slot made it "reject".
        body = teach_only_body()
        lesson_practice = body["unit_contract"]["idm_unit_brief"]["lesson_practice_sentences"][0]
        verdict = judge(("Q3_practice_complete", "critical", 0), ("Q4_feedback_teaches", "critical", 0),
                        ("Q6_alignment", "critical", 0))
        provider = FakeGenerate(**{WRITER: [writer_answer(body)], JUDGE: [verdict]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "validated", "provider")
        self.assertEqual(quality.judge_status, "pass")
        self.assertEqual(quality.finding_counts.model_dump(), {"minor": 0, "major": 0, "critical": 0})
        self.assertFalse({"Q3_practice_complete", "Q4_feedback_teaches", "Q6_alignment"} & set(quality.criteria))
        self.assertEqual(quality.author_note, "QA tự động: không phát hiện vấn đề lớn.")
        prompt = provider.calls[1]["prompt"]
        self.assertIn('"practices":[],"has_practice_slot":false', prompt)
        self.assertIn('"unit_title":"Tiêu chí bắt buộc escalate"', prompt)
        self.assertNotIn(lesson_practice, prompt.split("PLAN=")[1])
        self.assertIn("return not_applicable for\n   them", prompt)

    async def test_practice_unit_gets_only_its_own_practice_and_keeps_its_findings(self) -> None:
        body = escalate_body()
        provider = FakeGenerate(**{WRITER: [escalate_writer(body)],
                                   JUDGE: [judge(("Q3_practice_complete", "major", 0))]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "review_required", "provider")
        self.assertEqual(quality.criteria["Q3_practice_complete"], "major")
        practice = body["unit_contract"]["idm_unit_brief"]["components"][0]["practice"]["sentence"]
        plan = provider.calls[1]["prompt"].split("PLAN=")[1]
        self.assertIn(f'"practices":{json.dumps([practice], ensure_ascii=False)},"has_practice_slot":true', plan)

    async def test_title_criterion_is_stored_under_traceability(self) -> None:
        body = escalate_body()
        verdict = judge(("Q9_traceability", "minor", None))
        verdict["findings"].append({"criterion": "Q10_title_matches", "severity": "major", "component_index": 1,
                                    "witness": "Tiêu đề hứa một nội dung khác"})
        provider = FakeGenerate(**{WRITER: [escalate_writer(body)], JUDGE: [verdict]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "review_required", "provider")
        self.assertEqual((quality.judge_status, quality.criteria["Q9_traceability"]), ("review_required", "major"))
        self.assertNotIn("Q10_title_matches", quality.criteria)
        self.assertIn("tiêu đề không khớp nội dung được dạy (Q10)", quality.author_note)


class JudgeAggregationTests(unittest.TestCase):
    def test_not_applicable_never_counts(self) -> None:
        answered = [finding("Q3_practice_complete", "critical"), finding("Q4_feedback_teaches", "not_applicable"),
                    finding("Q5_grounded_criteria", "not_applicable"), finding("Q8_language", "minor")]
        teach_only = settle_applicability(answered, practice_slot=False)
        self.assertEqual([(item.criterion, item.severity) for item in teach_only],
                         [("Q3_practice_complete", "not_applicable"), ("Q4_feedback_teaches", "not_applicable"),
                          ("Q8_language", "minor")])
        self.assertEqual(criteria_summary(teach_only), {"Q8_language": "minor"})
        with_practice = settle_applicability(answered, practice_slot=True)
        self.assertEqual([item.criterion for item in with_practice], ["Q3_practice_complete", "Q8_language"])
        self.assertEqual(criteria_summary([finding("Q10_title_matches", "major"), finding("Q9_traceability", "minor")]),
                         {"Q9_traceability": "major"})

    def test_title_criterion_note_in_both_locales(self) -> None:
        outcome = JudgeOutcome("review_required", [finding("Q10_title_matches", "major")])
        notes = [build_unit_author_note(locale=locale, judge=outcome, deterministic_codes=[], fallback_slots=[],
                                        ai_drafted=False) for locale in ("vi", "en")]
        self.assertEqual(notes, ["QA tự động: 1 vấn đề cần xem — tiêu đề không khớp nội dung được dạy (Q10).",
                                 "Automatic QA: 1 issue(s) to review — a title does not match what the unit teaches "
                                 "(Q10)."])


QC_MINDSET_OPTIONS = ("Mindset cần bỏ: chỉ làm đúng việc được giao",
                      "Mindset cần bỏ: đổ lỗi cho nhân viên khi có sự cố",
                      "Mindset cần bỏ: ngại thay đổi quy trình đã quen",
                      "Mindset cần bỏ: luôn tin vào kinh nghiệm lâu năm")


def four_options(explanation: str) -> dict[str, Any]:
    return {"type": "problem", "problem_type": "multiple_choice", "question": "Mindset nào cần bỏ trước tiên?",
            "choices": [{"text": text, "correct": index == 1} for index, text in enumerate(QC_MINDSET_OPTIONS)],
            "explanation": explanation}


class ExplanationCoverageTests(StoryboardEndpointTestCase):
    """N10: every option is explained; the letters are read after the seeded shuffle relabels them."""

    def test_every_option_must_be_named(self) -> None:
        complete = ("Tiêu chí: mindset cần bỏ là mindset cản trở cải tiến. A — sai vì đây là việc cần làm đúng; "
                    "B — đúng vì đổ lỗi chặn cải tiến; C — sai vì chỉ là thói quen; D — sai vì kinh nghiệm vẫn có ích.")
        self.assertEqual(unexplained_options(four_options(complete)), [])
        self.assertEqual(unexplained_options(four_options(complete.split("; D", maxsplit=1)[0] + ".")), [3])
        self.assertEqual(unexplained_options(four_options("Đáp án đúng là B. Phương án A và C sai vì chưa đúng tiêu "
                                                          "chí cải tiến của bài.")), [3])
        # "loại A" names a category, not option A.
        self.assertEqual(unexplained_options(four_options("Loại A là khách VIP; phương án B đúng; C, D sai.")), [0])
        by_text = four_options("Chỉ làm đúng việc được giao là sai; đổ lỗi cho nhân viên khi có sự cố là đúng; "
                               "ngại thay đổi quy trình đã quen là sai; luôn tin vào kinh nghiệm lâu năm là sai.")
        for choice in by_text["choices"]:
            choice["text"] = choice["text"].removeprefix("Mindset cần bỏ: ")
        self.assertEqual(unexplained_options(by_text), [])

    def test_shuffled_options_keep_a_complete_explanation_complete(self) -> None:
        explanation = ("Tiêu chí: mindset cần bỏ là mindset cản trở cải tiến. A — sai vì đây là việc cần làm đúng; "
                       "B — đúng vì đổ lỗi chặn cải tiến; C — sai vì chỉ là thói quen; D — sai vì kinh nghiệm vẫn có "
                       "ích.")
        for seed in ("cp2_" + "1" * 32, "cp2_" + "2" * 32, "cp2_" + "3" * 32):
            served = normalize_single_choice(four_options(explanation), seed)[0]
            self.assertEqual(unexplained_options(served), [], seed)
            self.assertEqual(labelled_letters(served["explanation"]), {"A", "B", "C", "D"})

    async def test_explanation_missing_an_option_is_repaired(self) -> None:
        body = escalate_body()
        writer = escalate_writer(body)
        good = writer["components"]["c0"]["explanation"]
        writer["components"]["c0"]["explanation"] = good.split("; C", maxsplit=1)[0] + "."
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [slot_repair(escalate_writer(body), 0)],
                                   JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "validated", "provider")
        self.assertEqual(provider.names, [WRITER, REPAIR, JUDGE])
        self.assertIn("c0 IDM_W5_PRACTICE_INCOMPLETE at c0: exactly one correct choice, and an explanation that names "
                      "EVERY option by its letter", provider.calls[1]["prompt"])
        self.assertIn("Đã tự sửa: IDM_W5_PRACTICE_INCOMPLETE.", quality.author_note)


# --- N9: callouts must restate the facts -----------------------------------------------------------
INVENTED_CALLOUT = "Doanh nghiệp không thay đổi tư duy sẽ có nguy cơ bị thị trường đào thải rất cao."
SOURCE_CALLOUT = "Cấp 3 liên quan an toàn, pháp lý hoặc truyền thông: escalate ngay cho quản lý và bộ phận Pháp chế."


def with_callout(writer: dict[str, Any], text: str) -> dict[str, Any]:
    blocks = writer["components"]["c0"]["semantic_content"]["sections"][0]["blocks"]
    blocks.append({"kind": "warning", "text": text, "items": [], "rows": []})
    return writer


class CalloutGroundingTests(StoryboardEndpointTestCase):
    def test_only_callouts_the_facts_do_not_state_are_flagged(self) -> None:
        body = severity_body()
        evidence = evidence_index([fact["fact_text"] for fact in body["unit_contract"]["source_facts"]])
        writer = with_callout(with_callout(severity_writer(body), SOURCE_CALLOUT), INVENTED_CALLOUT)
        component = writer["components"]["c0"]
        self.assertEqual(ungrounded_callouts(component, evidence), [(0, 3)])
        prose = callouts_as_paragraphs(component, [(0, 3)])
        kinds = [block["kind"] for block in prose["semantic_content"]["sections"][0]["blocks"]]
        self.assertEqual(kinds, ["paragraph", "table", "warning", "paragraph"])
        self.assertEqual(prose["semantic_content"]["sections"][0]["blocks"][3]["text"], INVENTED_CALLOUT)
        brief = IdmUnitBriefV1.model_validate(body["unit_contract"]["idm_unit_brief"])
        value = {"components": [{**component, "type": "html"}]}
        self.assertEqual(deterministic_slot_findings(value, brief, [""], evidence),
                         [SlotFinding("IDM_W5_CALLOUT_UNGROUNDED", 0)])
        self.assertEqual(deterministic_slot_findings(value, brief, [""]), [])

    async def test_invented_callout_is_repaired_from_the_facts(self) -> None:
        body = severity_body()
        writer = with_callout(severity_writer(body), INVENTED_CALLOUT)
        fixed = slot_repair(with_callout(severity_writer(body), SOURCE_CALLOUT), 0)
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [fixed], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "validated", "provider")
        self.assertIn("c0 IDM_W5_CALLOUT_UNGROUNDED at c0.semantic_content.sections[0].blocks[2]: a warning block is "
                      "shown as a quotation/callout", provider.calls[1]["prompt"])
        self.assertNotIn(INVENTED_CALLOUT, provider.calls[1]["prompt"].split("REPAIR_REQUIREMENTS")[1])
        self.assertIn("Đã tự sửa: IDM_W5_CALLOUT_UNGROUNDED.", quality.author_note)
        self.assertIn('A "warning" block is shown to the learner as a quotation/callout', provider.calls[0]["prompt"])

    async def test_callout_still_invented_after_the_repair_becomes_a_paragraph_for_review(self) -> None:
        body = severity_body()
        writer = with_callout(severity_writer(body), INVENTED_CALLOUT)
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [slot_repair(writer, 0)], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "review_required", "provider")
        blocks = data["unit"]["components"][0]["semantic_content"]["sections"][0]["blocks"]
        self.assertEqual((blocks[2]["kind"], blocks[2]["text"]), ("paragraph", INVENTED_CALLOUT))
        self.assertNotIn("source_locked_fallback", data["unit"]["components"][0])
        self.assertIn("Đã chuyển 1 khung trích dẫn/lưu ý không có trong tài liệu thành đoạn văn thường — cần SME xác "
                      "nhận nội dung.", quality.author_note)
        self.assertNotIn("còn cảnh báo", quality.author_note)
        self.assertEqual(quality.deterministic_codes, ["IDM_W5_CALLOUT_TO_PROSE", "IDM_W5_CALLOUT_UNGROUNDED"])

    async def test_english_note_for_a_converted_callout(self) -> None:
        body = {**severity_body(), "locale": "en"}
        writer = with_callout(severity_writer(body), INVENTED_CALLOUT)
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [slot_repair(writer, 0)], JUDGE: [judge()]})
        _, data, _ = await self.post(body, provider)
        quality = self.assert_envelope(data, "provider_validated", "review_required", "provider")
        self.assertIn("1 callout(s) not found in the source were turned into plain paragraphs — the SME should "
                      "confirm their content.", quality.author_note)


# --- N11: FAQ items add value and the FAQ title matches them ----------------------------------------
RESTATED = ("Cấp 2 là khi thiệt hại tài chính dưới 50 triệu đồng hoặc khách hàng phàn nàn lần thứ hai; khi đó "
            "thông báo trưởng nhóm.")
EDGE_CASE = ("Thiệt hại tài chính dưới 50 triệu đồng đã là dấu hiệu của cấp 2, nên vẫn thông báo trưởng nhóm dù khách "
             "hàng chưa phàn nàn lần thứ hai.")
MISCONCEPTION = ("Không. Khi khiếu nại liên quan an toàn, pháp lý hoặc truyền thông thì đó là cấp 3 và phải escalate "
                 "ngay cho quản lý, dù chưa có thiệt hại tài chính.")
FAQ_TITLE = "Những nhầm lẫn thường gặp về cấp độ"


def faq_after_html_body() -> dict[str, Any]:
    """lsn_002 unit 1 (html + problem) with an la_faq slot after the question, as Node would plan it."""

    design, _ = golden_design()
    shard = golden_shard(0)
    lesson = shard.lessons[1]
    unit = lesson.units[0]
    faq = unit.components[0].model_copy(update={"component_index": 3, "type": "la_faq", "role": "clarify",
                                                 "title": FAQ_TITLE})
    unit = unit.model_copy(update={"components": [*unit.components, faq]})
    shard = shard.model_copy(update={"lessons": [shard.lessons[0], lesson.model_copy(update={"units": [unit]}),
                                                 *shard.lessons[2:]]})
    return gu.build_unit_request(design, shard, chapter_index=0, lesson_position=1, unit_position=0,
                                 facts=g.source_facts())


def faq_writer(body: dict[str, Any], *answers: str, title: str = FAQ_TITLE) -> dict[str, Any]:
    writer = severity_writer(body)
    questions = ("Khiếu nại cấp 2 là khi nào?", "Có thiệt hại tài chính nhỏ nhưng khách mới phàn nàn lần đầu thì sao?",
                 "Chưa có thiệt hại tài chính thì luôn là cấp 1 phải không?")
    writer["components"]["c2"] = {
        "title": title, "selection_rationale": "Làm rõ nhầm lẫn về cấp độ.", "covered_source_fact_ids": [],
        "items": [{"question": question, "answer": answer}
                  for question, answer in zip(questions, answers, strict=False)]}
    return writer


class FaqValueTests(StoryboardEndpointTestCase):
    def test_restated_items_and_title_mismatch(self) -> None:
        body = faq_after_html_body()
        writer = faq_writer(body, RESTATED, EDGE_CASE, MISCONCEPTION)
        html_text = html_before({"components": [{"type": "html", **writer["components"]["c0"]}]}, 1)
        self.assertEqual(restated_faq_items(writer["components"]["c2"], html_text), [0])
        self.assertEqual(restated_faq_items(writer["components"]["c2"], ""), [])
        self.assertFalse(faq_title_mismatch(writer["components"]["c2"]))
        self.assertFalse(faq_title_mismatch(escalate_writer(escalate_body())["components"]["c1"]))
        meeting = faq_writer(body, EDGE_CASE, MISCONCEPTION, title="Câu hỏi thường gặp trong họp giao ban")
        self.assertTrue(faq_title_mismatch(meeting["components"]["c2"]))
        # Boilerplate only ("Câu hỏi thường gặp") says nothing to match.
        self.assertFalse(faq_title_mismatch({**meeting["components"]["c2"], "title": "Câu hỏi thường gặp"}))

    async def test_item_repeating_the_html_is_dropped_after_the_repair(self) -> None:
        body = faq_after_html_body()
        writer = faq_writer(body, RESTATED, EDGE_CASE, MISCONCEPTION)
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [slot_repair(writer, 2)], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "validated", "provider")
        self.assertIn("c2 IDM_W5_FAQ_RESTATES_HTML at c2.items[0]: these items only repeat the html above",
                      provider.calls[1]["prompt"])
        self.assertEqual([item["answer"] for item in data["unit"]["components"][2]["items"]],
                         [EDGE_CASE, MISCONCEPTION])
        self.assertIn("Đã bỏ 1 câu hỏi đáp chỉ nhắc lại nội dung vừa học.", quality.author_note)
        self.assertIn("Each la_faq item adds value: a common misconception, an edge case", provider.calls[0]["prompt"])

    async def test_too_few_items_left_keeps_the_faq_for_review(self) -> None:
        body = faq_after_html_body()
        writer = faq_writer(body, RESTATED, EDGE_CASE)
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [slot_repair(writer, 2)], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "review_required", "provider")
        self.assertEqual(len(data["unit"]["components"][2]["items"]), 2)
        self.assertIn("Cần xem (IDM_W5_FAQ_RESTATES_HTML, khối 3 (Hỏi đáp)): câu hỏi đáp chỉ nhắc lại nội dung vừa "
                      "học", quality.author_note)

    async def test_mismatched_title_is_repaired(self) -> None:
        body = faq_after_html_body()
        writer = faq_writer(body, EDGE_CASE, MISCONCEPTION, title="Câu hỏi thường gặp trong họp giao ban")
        fixed = slot_repair(faq_writer(body, EDGE_CASE, MISCONCEPTION), 2)
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [fixed], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "validated", "provider")
        self.assertIn("c2 IDM_W5_FAQ_TITLE_MISMATCH at c2: the la_faq title names what its questions are about",
                      provider.calls[1]["prompt"])
        self.assertEqual(data["unit"]["components"][2]["title"], FAQ_TITLE)
        self.assertIn("Đã tự sửa: IDM_W5_FAQ_TITLE_MISMATCH.", quality.author_note)


class CalloutKeptForReviewTests(unittest.IsolatedAsyncioTestCase):
    """N9: when the paragraph form is not acceptable (here the plan requires the warning), the slot stays."""

    @dataclasses.dataclass(frozen=True)
    class Finding:
        code: str
        path: str

    async def test_callout_that_cannot_become_a_paragraph_is_kept_for_review(self) -> None:
        body = severity_body()
        request = RagLessonAuthorUnitV2Request.model_validate(body)
        base = unit_service._idm_unit_deps(request)

        def validate(unit: dict[str, Any], expected: dict[str, Any]) -> Any:
            first = unit["components"][0]
            kinds = [block["kind"] for section in first["semantic_content"]["sections"] for block in section["blocks"]]
            if not first.get("source_locked_fallback") and "warning" not in kinds:
                return self.Finding("REQUIRED_ARTIFACT_NOT_PRESERVED", "components[0].semantic_content")
            return base.validate_unit(unit, expected)

        deps = dataclasses.replace(base, validate_unit=validate, judge_mode="off")
        writer = with_callout(severity_writer(body), INVENTED_CALLOUT)
        usage = AiUsage(inputTokens=1, outputTokens=1, totalTokens=2)
        generate = AsyncMock(side_effect=[(json.dumps(writer, ensure_ascii=False), usage),
                                          (json.dumps(slot_repair(writer, 0), ensure_ascii=False), usage)])
        result = await run_idm_unit(contract=request.unit_contract, runtime=make_runtime(generate, allowance=None),
                                    deps=deps, fallback_only=False)
        self.assertEqual((result["content_origin"], result["quality_state"]), ("provider_validated", "review_required"))
        blocks = result["unit"]["components"][0]["semantic_content"]["sections"][0]["blocks"]
        self.assertEqual((blocks[-1]["kind"], blocks[-1]["text"]), ("warning", INVENTED_CALLOUT))
        note = result["unit"]["idm_quality"]["author_note"]
        self.assertIn("Giữ bản AI để tác giả rà soát: khối 1 (Lý thuyết) — IDM_W5_CALLOUT_UNGROUNDED.", note)
        self.assertIn("Cần xem (IDM_W5_CALLOUT_UNGROUNDED, khối 1 (Lý thuyết)): khung trích dẫn/lưu ý nêu điều không "
                      "tìm thấy trong tài liệu", note)


# --- N5: an enumerated framework is presented as a whole; action items are Must Do candidates --------
SHIFT_FACTS = ("SHIFT 1: TỪ TƯ DUY LÀM THUÊ SANG TƯ DUY LÀM CHỦ",
               "Shift 2 – Think Customer: lấy khách hàng làm trung tâm",  # noqa: RUF001 - a PDF dash
               "Shift 3: Think System", "Shift 4: Think Big", "Shift 5: Think Partner",
               "Năm 2015, công ty mở rộng lên 40 nhân sự.")


def html_component(*blocks: dict[str, Any]) -> dict[str, Any]:
    return {"type": "html", "semantic_content": {"sections": [{"heading": "Bản đồ", "blocks": list(blocks)}]}}


class FrameworkTests(unittest.TestCase):
    def test_promise_needs_a_number_and_a_framework_noun(self) -> None:
        self.assertEqual(framework_promise(["Tổng quan 5 chuyển dịch"]), (5, "chuyển dịch", "5 chuyển dịch"))
        self.assertEqual(framework_promise(["Ba cấp độ nghiêm trọng"]), (3, "cấp độ", "Ba cấp độ"))
        self.assertEqual(framework_promise(["6 trụ cột ERA", "x"]), (6, "trụ cột", "6 trụ cột"))
        self.assertEqual(framework_promise(["8 Quick-Win đầu tiên"]), (8, "quick-win", "8 Quick-Win"))
        # "sau" (after) is not "sáu" (six); a year is not a count; one item is not a framework.
        for text in ("Việc cần làm sau bước kiểm tra", "Năm 2015 phòng mở rộng", "1 bước duy nhất", "Shift 1"):
            self.assertIsNone(framework_promise([text]), text)

    def test_numbered_items_and_listing(self) -> None:
        promise = build_promise(["Tổng quan 5 chuyển dịch"], [SHIFT_FACTS])
        assert promise is not None
        self.assertEqual([label for label, _name in promise.items], [f"shift {n}" for n in range(1, 6)])
        listed = "Shift 1 làm chủ; Think Customer; Shift 3 hệ thống; Think Big; Think Partner."
        self.assertTrue(framework_listed(promise, [], listed))
        # The QC unit: five rows about the operating model's axes, not the five shifts.
        axes = html_component({"kind": "table", "rows": [{"label": f"Trục {n}", "value": "v"} for n in range(5)]})
        self.assertEqual(framework_coverage(promise, [axes], "Trục 0 Trục 1 Trục 2 Trục 3 Trục 4"), 0)
        self.assertFalse(framework_listed(promise, [axes], "Trục 0 Trục 1 Trục 2 Trục 3 Trục 4"))
        # Facts that number no items: one list or table with every item is enough.
        bare = build_promise(["Tổng quan 5 chuyển dịch"], [["Doanh nghiệp cần chuyển dịch tư duy."]])
        assert bare is not None and bare.items == ()
        self.assertTrue(framework_listed(bare, [axes], ""))
        self.assertFalse(framework_listed(bare, [html_component({"kind": "bullets", "items": ["a", "b"]})], ""))

    def test_only_teaching_html_is_checked(self) -> None:
        body = severity_body()
        brief = IdmUnitBriefV1.model_validate(body["unit_contract"]["idm_unit_brief"])
        facts = [fact["fact_text"] for fact in body["unit_contract"]["source_facts"]]
        promise = build_promise([body["unit_contract"]["unit_title"]], [facts])
        assert promise is not None
        self.assertEqual(promise.count, 3)
        prose = severity_writer(body)["components"]["c0"]
        prose["semantic_content"]["sections"][0]["blocks"] = [
            {"kind": "paragraph", "text": LEVELS_IN_WORDS, "items": [], "rows": []}]
        value = {"components": [{**prose, "type": "html"}]}
        self.assertEqual(deterministic_slot_findings(value, brief, [""], framework=promise),
                         [SlotFinding("IDM_W5_FRAMEWORK_INCOMPLETE", 0)])
        listed = {"components": [{**severity_writer(body)["components"]["c0"], "type": "html"}]}
        self.assertEqual(deterministic_slot_findings(listed, brief, [""], framework=promise), [])
        worksheet = IdmUnitBriefV1.model_validate(worksheet_body()["unit_contract"]["idm_unit_brief"])
        # A worksheet html slot (role practice) neither promises nor lists the overview.
        self.assertEqual(deterministic_slot_findings(value, worksheet, [""], framework=promise),
                         [SlotFinding("IDM_W5_WORKSHEET_INCOMPLETE", 0)])


LEVELS_IN_WORDS = ("Mỗi khiếu nại thuộc một trong ba cấp độ, và cấp độ quyết định ai xử lý: nhân viên, trưởng nhóm "
                   "hoặc quản lý. Hãy đọc kỹ thiệt hại tài chính, số lần khách hàng phàn nàn và các yếu tố an toàn, "
                   "pháp lý hay truyền thông trước khi chọn cấp độ cho khiếu nại.")


def levels_in_words(body: dict[str, Any]) -> dict[str, Any]:
    writer = severity_writer(body)
    writer["components"]["c0"]["semantic_content"]["sections"][0]["blocks"] = [
        {"kind": "paragraph", "text": LEVELS_IN_WORDS, "items": [], "rows": []}]
    return writer


class FrameworkUnitTests(StoryboardEndpointTestCase):
    async def test_title_promising_three_levels_gets_the_overview(self) -> None:
        body = severity_body()  # unit title "Ba cấp độ nghiêm trọng"
        provider = FakeGenerate(**{WRITER: [levels_in_words(body)], REPAIR: [slot_repair(severity_writer(body), 0)],
                                   JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "validated", "provider")
        self.assertIn("c0 IDM_W5_FRAMEWORK_INCOMPLETE at c0: the unit or slot title names a framework of several "
                      "items", provider.calls[1]["prompt"])
        self.assertIn("(the title promises 3 items; the html shows 0; list every item numbered in SOURCE_FACTS in "
                      "one bullets block)", provider.calls[1]["prompt"])
        # QC run 8de1c76b (Q3): a framework-only repair is a targeted one (low thinking).
        self.assertEqual(provider.calls[1]["thinking_level"], "low")
        self.assertIn('"unit_title":"Ba cấp độ nghiêm trọng"', provider.calls[0]["prompt"])
        self.assertIn("Đã tự sửa: IDM_W5_FRAMEWORK_INCOMPLETE.", quality.author_note)

    async def test_overview_the_repair_cannot_add_is_listed_from_the_facts(self) -> None:
        # QC run 8de1c76b (Q3): instead of keeping the slot for review, the server lists the three levels by the
        # names the facts give them; the unit stays reviewable so the author smooths the list into the text.
        body = severity_body()
        writer = levels_in_words(body)
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [slot_repair(writer, 0)], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "review_required", "provider")
        sections = data["unit"]["components"][0]["semantic_content"]["sections"]
        self.assertEqual(sections[1]["heading"], "Ba cấp độ gồm những gì?")
        self.assertEqual(sections[1]["blocks"][0]["items"],
                         ["Cấp 1: Ảnh hưởng thấp",
                          "Cấp 2: Thiệt hại tài chính dưới 50 triệu đồng hoặc khách hàng phàn nàn lần thứ hai",
                          "Cấp 3: Liên quan an toàn"])
        self.assertIn("Đã chèn danh sách 3 thành phần của khung (tên lấy từ tài liệu)", quality.author_note)
        self.assertNotIn("Cần xem (IDM_W5_FRAMEWORK_INCOMPLETE", quality.author_note)
        self.assertIn("IDM_W5_FRAMEWORK_LIST_INSERTED", quality.deterministic_codes)


class FrameworkOrientationDesignTests(unittest.IsolatedAsyncioTestCase):
    """W4: the first lesson serving an objective that names a framework opens with an orientation unit."""

    async def design(self, statement: str) -> dict[str, Any]:
        def change(raw: dict[str, Any]) -> None:
            raw["learning_objectives"][0]["statement"] = statement

        result, _, _ = await design_module(0, [module_response("mod_01")], context_change=change)
        return result

    async def test_missing_orientation_unit_is_noted_on_the_first_lesson(self) -> None:
        result = await self.design("Người học có thể áp dụng 5 chuyển dịch để phân loại khiếu nại theo nhóm")
        dumped = json.dumps(result["shard"], ensure_ascii=False)
        note = ('Lưu ý: mục tiêu nêu \\"5 chuyển dịch\\" nhưng chưa có unit định hướng liệt kê đủ các thành phần — '
                "nên thêm một unit tổng quan ở đầu mục này.")
        self.assertEqual(dumped.count(note), 1)
        self.assertEqual(result["content_origin"], "provider_validated")

    async def test_orientation_unit_titled_with_the_framework_needs_no_note(self) -> None:
        # The golden lsn_002 unit "Ba cấp độ nghiêm trọng" is the overview of "3 cấp độ".
        result = await self.design("Người học có thể phân loại khiếu nại theo nhóm và 3 cấp độ nghiêm trọng")
        self.assertNotIn("unit định hướng", json.dumps(result["shard"], ensure_ascii=False))

    def test_orientation_note_in_english(self) -> None:
        lesson = IdmLessonDesignV1.model_validate(module_response("mod_01")["lessons"][0])
        noted = with_orientation_note(lesson, ["5 shifts"], "en")
        self.assertTrue(noted.notes.endswith('Note: the objective names "5 shifts" but no orientation unit lists all '
                                             "of its items — add an overview unit at the start of this section."))


class ActionItemTests(unittest.TestCase):
    def test_action_items_reach_the_catalog_and_an_unlinked_one_warns(self) -> None:
        blocks = golden_content_blocks()
        mistakes = next(block for block in blocks if block.block_id == "cb_0011")
        text = {mistakes.fact_keys[-1]: "HÀNH ĐỘNG CEO: Chọn 01 quy trình hay xảy ra sự cố nhất và áp dụng 5Why.",
                blocks[0].fact_keys[0]: "Hành động của phòng năm 2009 là mở chi nhánh."}
        self.assertEqual(action_items(mistakes, text), [text[mistakes.fact_keys[-1]]])
        self.assertEqual(action_items(blocks[0], text), [])
        catalog = {entry["block_id"]: entry for entry in reduce_catalog(blocks, text)}
        self.assertEqual(catalog["cb_0011"]["action_items"], [text[mistakes.fact_keys[-1]]])
        self.assertNotIn("action_items", catalog["cb_0009"])
        self.assertNotIn("action_items", reduce_catalog(blocks)[10])
        found = validate_w1_reduce(golden_reduce_response(), blocks, CONTENT_CONTEXT, {"cb_0011", "cb_0009"})
        self.assertEqual([(issue.code, issue.severity) for issue in found],
                         [("IDM_W1_ACTION_ITEM_UNLINKED", "warning")])
        self.assertEqual(validate_w1_reduce(golden_reduce_response(), blocks, CONTENT_CONTEXT, {"cb_0009"}), [])


# --- N4: a block that defines a course term is never left out; removed blocks reach the author -------
DEFINITION = "CSKH 360 = Lắng nghe trọn vẹn + Phản hồi đúng hạn → Khách hàng trung thành."


def defining_facts() -> list[Any]:
    """The golden source with its history block (cb_0002, removed by W2) also stating the course formula."""

    facts = g.source_facts()
    return [fact.model_copy(update={"fact_text": DEFINITION}) if fact.fact_key == g.key(1, 3) else fact
            for fact in facts]


def defining_context(**changes: Any) -> dict[str, Any]:
    return {**g.project_context(), "course_title_hint": "CSKH 360: xử lý khiếu nại khách hàng", **changes}


class TermTests(unittest.TestCase):
    def test_course_terms_and_definitions(self) -> None:
        terms = course_terms(["QC check 3", "Executive Training Playbook - BiC Modun 1_ Change Mindset"],
                             ["Người học có thể đánh giá vị trí doanh nghiệp trên Ladder BiC 5.0",
                              "Người học có thể áp dụng mô hình Tam Doanh cho CEO"])
        self.assertEqual(terms, ["Tam Doanh", "Ladder", "BiC", "CEO", "QC"])
        for text in ("BiC = Lean System + Efficiency → Preventive & Proactive", "Công thức: BiC = Lean System",
                     "BiC (Business in Change) là chương trình nâng cao năng suất", "- Mô hình BiC 5.0: Proactive"):
            self.assertTrue(defines(text, "BiC"), text)
        for text in ("Năm 2008 Bộ KH&CN khởi động BiC", "Mục tiêu của BiC là tăng hiệu quả", "Lưu ý: BiC cần cam kết",
                     "BiC thế hệ thứ nhất là chương trình của Bộ"):
            self.assertFalse(defines(text, "BiC"), text)
        self.assertEqual(defined_terms(["Năm 2008 khởi động BiC.", "BiC = Lean + Efficiency"], ["CEO", "BiC"]),
                         ["BiC"])
        self.assertTrue(mentions("Đánh giá BiC 5.0 của doanh nghiệp", "BiC"))
        self.assertFalse(mentions("bicycle", "BiC"))

    def rows_and_blocks(self) -> tuple[Any, Any, Any]:
        design = golden_course_design()
        return design, list(design.blueprint), list(design.blocks)

    def test_golden_blueprint_is_unchanged(self) -> None:
        design, rows, blocks = self.rows_and_blocks()
        kept_rows, kept = keep_term_definitions(
            rows, blocks, context=design.project_context, objectives=design.learning_objectives,
            must_dos=design.must_dos, fact_text={fact.fact_key: fact.fact_text for fact in g.source_facts()})
        self.assertEqual((kept_rows, kept), (rows, []))

    def test_left_out_definition_becomes_must_know_or_reference(self) -> None:
        design, rows, blocks = self.rows_and_blocks()
        context = IdmProjectContextV1.model_validate(defining_context())
        text = {fact.fact_key: fact.fact_text for fact in defining_facts()}
        for proposed in ("remove", "nice_to_know"):
            demoted = [row.model_copy(update={"classification": proposed}) if row.block_id == "cb_0002" else row
                       for row in rows]
            kept_rows, kept = keep_term_definitions(demoted, blocks, context=context,
                                                    objectives=design.learning_objectives, must_dos=design.must_dos,
                                                    fact_text=text)
            row = next(item for item in kept_rows if item.block_id == "cb_0002")
            self.assertEqual((row.classification, row.placement, row.treatment, row.must_do_ids, row.lo_id),
                             ("must_know", "course", "condense", ["md_1"], "lo_1"))
            self.assertEqual(row.detail_level, "Giữ nguyên định nghĩa/công thức của CSKH; lược phần lịch sử và bối "
                                               "cảnh.")
            self.assertIn(f"Week 2 đề xuất {proposed}", row.rationale)
            self.assertEqual(kept, [KeptDefinition("cb_0002", ("CSKH",), proposed)])
        # No Must Do is served by a course block yet: the definition becomes Reference / Job Aid.
        outside = [row.model_copy(update={"classification": "remove", "placement": "excluded", "treatment": "remove",
                                          "must_do_ids": []}) for row in rows]
        kept_rows, _ = keep_term_definitions(outside, blocks, context=context, objectives=design.learning_objectives,
                                             must_dos=design.must_dos, fact_text=text)
        row = next(item for item in kept_rows if item.block_id == "cb_0002")
        self.assertEqual((row.classification, row.placement, row.treatment, row.must_do_ids),
                         ("reference", "reference_job_aid", "move_to_reference", []))


class TermDefinitionCourseTests(unittest.IsolatedAsyncioTestCase):
    async def test_definition_of_a_title_term_cannot_be_removed(self) -> None:
        # The QC regression: W2 removes the block that holds the course formula together with its history.
        course = json.loads(json.dumps(g.W4_COURSE))
        course["modules"][0]["lessons"][0]["block_ids"] = ["cb_0002", "cb_0003", "cb_0004"]
        provider = g.golden_provider({"IdmW4CourseResponseV1": [course]})
        request = golden_course_request(defining_facts(), defining_context())
        result, _ = await run_course_design(provider, request=request)
        design = IdmCourseDesignV1.model_validate(result["idm"])
        row = next(item for item in design.blueprint if item.block_id == "cb_0002")
        self.assertEqual((row.classification, row.must_do_ids), ("must_know", ["md_1"]))
        self.assertEqual({item.disposition for item in design.dispositions if item.block_id == "cb_0002"},
                         {"course"})
        self.assertEqual(design.stage_origins, {"w1_map": "provider", "w1_reduce": "provider", "w2": "provider",
                                                "w4": "provider"})
        self.assertIn("• Giữ lại vì định nghĩa thuật ngữ cốt lõi (Week 2 đề xuất loại) — 1 khối:\n"
                      "  - Phòng Chăm sóc Khách hàng đã phát triển ra sao?: CSKH (đề xuất ban đầu: Remove).",
                      design.notes.course)
        self.assertNotIn("(Remove) — ", design.notes.course)
        w2_prompt = next(call["prompt"] for call in provider.calls if call["schema"] == "IdmW2BlueprintResponseV1")
        self.assertIn("Never classify as\nnice_to_know or remove a block that defines", w2_prompt)
        w1_prompt = next(call["prompt"] for call in provider.calls if call["schema"] == "IdmW1SectionResponseV1")
        self.assertIn("A formula or definition of a term", w1_prompt)

    def test_course_notes_list_removed_blocks_with_their_reason(self) -> None:
        notes = golden_course_design().notes.course
        self.assertIn("• Nội dung đã loại khỏi khoá (Remove) — 1 khối, kèm lý do:\n"
                      "  - Phòng Chăm sóc Khách hàng đã phát triển ra sao?: Quyết định theo luồng Week 2.", notes)
        common: dict[str, Any] = {"ai_proposed": False, "total_minutes": 10, "lesson_count": 1, "module_count": 1,
                                  "reference_block_count": 0, "excluded_block_count": 1, "holds": [],
                                  "sme_questions": [], "fallback_stages": []}
        english = build_course_notes(locale="en", removed=[SkippedBlockNote("BiC history", "Outdated programme.")],
                                     kept_definitions=[KeptDefinitionNote("What is BiC?", ("BiC",), "nice_to_know")],
                                     **common)
        self.assertIn("• Kept because they define a core term (Week 2 proposed to leave them out) — 1 blocks:\n"
                      "  - What is BiC?: BiC (first proposed: Nice to know).", english)
        self.assertIn("• Content removed from the course (Remove) — 1 blocks, with the reason:\n"
                      "  - BiC history: Outdated programme.", english)
