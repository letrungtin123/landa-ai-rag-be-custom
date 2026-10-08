"""QC course 364564 (2026-10-08) regressions through the real ``/unit`` route and the IDM helpers.

Package A (deterministic MCQ and note fixes):

* N2: the correct option was the longest in 8 of 9 questions (a length cue, review note only).
* N3: an option copied the worked example of the worksheet shown just before the question.
* N8: codes found before a successful repair were printed as "còn cảnh báo" on a unit that passed.

Package B (prompts, judge and grounding):

* N7: the W6 judge rejected two teach-only units for practice criteria (it saw the lesson's practice).

The provider is the local ``FakeGenerate``; nothing reaches the network.
"""

from __future__ import annotations

import json
import unittest
from typing import Any

from app.idm.contracts import IdmJudgeFindingV1
from app.idm.mcq import labelled_letters, normalize_single_choice
from app.idm.qa import (
    JudgeOutcome,
    build_unit_author_note,
    criteria_summary,
    settle_applicability,
    unexplained_options,
)
from tests.idm_contract_bridge import writer_answer
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
        self.assertIn("c1 IDM_W5_ANSWER_LEAK at c1: an option of this question copies the example or text shown "
                      "before it: rewrite THAT option", provider.calls[1]["prompt"])
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
    async def test_far_longer_correct_option_is_a_review_note_without_repair(self) -> None:
        body = escalate_body()
        writer = escalate_writer(body)
        writer["components"]["c0"]["choices"][0]["text"] = "Hứa hoàn tiền ngay để giữ chân khách VIP"
        writer["components"]["c0"]["choices"][2]["text"] = "Chỉ thông báo trưởng nhóm vì đây là khách VIP"
        provider = FakeGenerate(**{WRITER: [writer], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "validated", "provider")
        self.assertEqual(provider.names, [WRITER, JUDGE])
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
