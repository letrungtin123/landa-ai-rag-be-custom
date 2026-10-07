"""W5 deterministic checks and the W6 judge (spec §6.8, §7.7.2(b), §7.7.5).

Every provider call goes to ``FakeIdmProvider``; nothing reaches the network.
"""

from __future__ import annotations

import copy
import json
import unittest
from typing import Any

from app.idm.contracts import IdmJudgeFindingV1, IdmUnitBriefV1, IdmUnitQualityV1
from app.idm.qa import (
    CRITERIA,
    JudgeOutcome,
    SlotFinding,
    blocking_count,
    build_unit_author_note,
    build_unit_quality,
    criteria_summary,
    deterministic_slot_findings,
    learner_view,
    repair_targets,
    run_judge,
    worst_counts,
)
from app.idm.runtime import IdmProviderError
from tests import idm_golden as g
from tests import idm_golden_unit as gu
from tests.idm_golden import FakeIdmProvider
from tests.idm_test_support import assert_node_trace, golden_design, golden_shard, make_runtime

JUDGE = "IdmJudgeResponseV1"
LONG_SOURCE = (
    "Bắt buộc escalate khi khiếu nại liên quan đến an toàn của khách hàng. Bắt buộc escalate khi khiếu nại có yếu "
    "tố pháp lý hoặc khách hàng đề cập đến kiện tụng. Bắt buộc escalate khi giá trị thiệt hại từ 50 triệu đồng trở "
    "lên. Với khách hàng VIP, nhân viên phải thông báo cho trưởng nhóm nhưng không bắt buộc escalate. Không được "
    "hứa bồi thường trước khi khiếu nại được phân loại mức độ."
)
GOOD_EXPLANATION = (
    "Tiêu chí: khách hàng phàn nàn lần thứ hai là dấu hiệu của cấp 2. A — sai vì lần phàn nàn thứ hai đã vượt cấp 1; "
    "B — đúng vì khớp dấu hiệu cấp 2; C — sai vì không có yếu tố an toàn, pháp lý hay truyền thông."
)


def brief() -> IdmUnitBriefV1:
    """lsn_002 unit 1: html (treatment keep) + problem practice (ai_drafted)."""

    design, _ = golden_design()
    body = gu.build_unit_request(design, golden_shard(0), chapter_index=0, lesson_position=1, unit_position=0,
                                 facts=g.source_facts())
    return IdmUnitBriefV1.model_validate(body["unit_contract"]["idm_unit_brief"])


def with_treatment(source: IdmUnitBriefV1, treatment: str) -> IdmUnitBriefV1:
    slot = source.components[0]
    treated = [item.model_copy(update={"treatment": treatment}) for item in slot.treatments]
    return source.model_copy(update={"components": [slot.model_copy(update={"treatments": treated}),
                                                    *source.components[1:]]})


def html(text: str) -> dict[str, Any]:
    return {"type": "html", "title": "Ba cấp độ", "component_plan_id": "cp2_" + "a" * 32,
            "source_fact_ids": ["d1-c4-f1"], "covered_source_fact_ids": ["d1-c4-f1"],
            "semantic_content": {"sections": [{"heading": "Khiếu nại thuộc cấp nào?", "learning_block_ids": [],
                                               "blocks": [{"kind": "paragraph", "text": text, "items": [],
                                                           "rows": []}]}]},
            "html": None}


def problem(explanation: str = GOOD_EXPLANATION, correct: tuple[bool, bool, bool] = (False, True, False),
            ) -> dict[str, Any]:
    texts = ("Cấp 1 vì chưa có thiệt hại tài chính", "Cấp 2 vì khách hàng phàn nàn lần thứ hai",
             "Cấp 3 vì cần escalate ngay cho quản lý")
    return {"type": "problem", "title": "Khiếu nại này thuộc cấp nào?", "component_plan_id": "cp2_" + "b" * 32,
            "source_fact_ids": [], "supporting_evidence_fact_ids": ["d1-c4-f2"], "problem_type": "multiple_choice",
            "question": "Khách hàng gọi lần thứ hai vì giao trễ. Khiếu nại này thuộc cấp độ nào?",
            "choices": [{"text": text, "correct": flag} for text, flag in zip(texts, correct, strict=True)],
            "explanation": explanation}


def unit(*components: dict[str, Any]) -> dict[str, Any]:
    return {"title": "Ba cấp độ nghiêm trọng", "source_fact_ids": ["d1-c4-f1"], "components": list(components)}


def finding(criterion: str, severity: str, index: int | None = None, witness: str = "trích dẫn") -> IdmJudgeFindingV1:
    return IdmJudgeFindingV1.model_validate({"criterion": criterion, "severity": severity,
                                             "component_index": index, "witness": witness})


def judge_payload(*overrides: tuple[str, str, int | None]) -> dict[str, Any]:
    severities = {criterion: ("pass", None) for criterion in CRITERIA}
    severities.update({criterion: (severity, index) for criterion, severity, index in overrides})
    verdict = ("reject" if any(s == "critical" for s, _ in severities.values())
               else "review_required" if any(s == "major" for s, _ in severities.values()) else "pass")
    return {"verdict": verdict, "findings": [
        {"criterion": criterion, "severity": severity, "component_index": index, "witness": "<b>trích</b>"}
        for criterion, (severity, index) in severities.items()]}


class DeterministicSlotFindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.brief = brief()
        self.owned = ["", ""]

    def findings(self, value: dict[str, Any], *, source: IdmUnitBriefV1 | None = None,
                 owned: list[str] | None = None) -> list[SlotFinding]:
        return deterministic_slot_findings(value, source or self.brief, owned or self.owned)

    def test_complete_practice_has_no_finding(self) -> None:
        self.assertEqual(self.findings(unit(html("Mỗi khiếu nại thuộc một trong ba cấp độ."), problem())), [])

    def test_practice_incomplete(self) -> None:
        one_option = "Tiêu chí: khách hàng phàn nàn lần thứ hai là dấu hiệu của cấp 2 nên cần thông báo trưởng nhóm."
        no_choices = {**problem(), "choices": None}
        for value in (problem(one_option), problem(correct=(True, True, False)), problem(correct=(False,) * 3),
                      no_choices):
            self.assertEqual(self.findings(unit(html("Ba cấp độ."), value)),
                             [SlotFinding("IDM_W5_PRACTICE_INCOMPLETE", 1)])
        by_text = ("Phương án Cấp 2 vì khách hàng phàn nàn lần thứ hai là đúng; phương án Cấp 3 vì cần escalate ngay "
                   "cho quản lý là sai vì không có yếu tố an toàn.")
        self.assertEqual(self.findings(unit(problem(by_text))), [])

    def test_feedback_not_teaching(self) -> None:
        self.assertEqual(self.findings(unit(problem("A — sai; B — đúng."))),
                         [SlotFinding("IDM_W5_FEEDBACK_NOT_TEACHING", 0)])
        self.assertEqual(self.findings(unit(problem("Đúng!"))),
                         [SlotFinding("IDM_W5_PRACTICE_INCOMPLETE", 0), SlotFinding("IDM_W5_FEEDBACK_NOT_TEACHING", 0)])

    def test_answer_leak_needs_the_answer_framed_as_answer_in_preceding_html(self) -> None:
        leak = html("Đáp án đúng: Cấp 2 vì khách hàng phàn nàn lần thứ hai.")
        self.assertEqual(self.findings(unit(leak, problem())), [SlotFinding("IDM_W5_ANSWER_LEAK", 1)])
        no_frame = html("Ví dụ: Cấp 2 vì khách hàng phàn nàn lần thứ hai.")
        self.assertEqual(self.findings(unit(no_frame, problem())), [])
        self.assertEqual(self.findings(unit(problem(), leak)), [])

    def test_verbatim_copy_only_when_treatment_is_not_keep(self) -> None:
        owned = [LONG_SOURCE, ""]
        copied = unit(html(LONG_SOURCE), problem())
        self.assertEqual({item.treatment for item in self.brief.components[0].treatments}, {"keep"})
        self.assertEqual(self.findings(copied, owned=owned), [])
        condensed = with_treatment(self.brief, "condense")
        self.assertEqual(self.findings(copied, source=condensed, owned=owned),
                         [SlotFinding("IDM_W5_VERBATIM_COPY", 0)])
        short = unit(html(LONG_SOURCE[:200]), problem())
        self.assertEqual(self.findings(short, source=condensed, owned=owned), [])
        rewritten = unit(html("Hãy escalate ngay khi có yếu tố an toàn, pháp lý hoặc thiệt hại lớn. " * 6), problem())
        self.assertEqual(self.findings(rewritten, source=condensed, owned=owned), [])

    def test_non_object_components_and_extra_slots_are_ignored(self) -> None:
        value = unit(html("Một"), problem(), html(LONG_SOURCE))
        value["components"].insert(0, "not-a-component")
        self.assertEqual(self.findings(value, source=with_treatment(self.brief, "rewrite"), owned=[LONG_SOURCE]),
                         [])


class LearnerViewTests(unittest.TestCase):
    def test_view_carries_learner_text_and_no_identifiers(self) -> None:
        view = learner_view(unit(html("Mỗi khiếu nại thuộc một cấp độ."), problem(), "skip"))
        self.assertEqual([entry["index"] for entry in view], [0, 1])
        self.assertEqual(set(view[0]), {"index", "type", "title", "text"})
        self.assertEqual(set(view[1]), {"index", "type", "title", "question", "choices", "explanation"})
        self.assertEqual(view[1]["choices"][1], {"text": "Cấp 2 vì khách hàng phàn nàn lần thứ hai", "correct": True})
        dumped = json.dumps(view, ensure_ascii=False)
        for marker in ("cp2_", "d1-c4", "component_plan_id", "source_fact_ids"):
            self.assertNotIn(marker, dumped)
        self.assertIn("Mỗi khiếu nại thuộc một cấp độ.", view[0]["text"])


class RunJudgeTests(unittest.IsolatedAsyncioTestCase):
    async def judge(self, answers: list[Any], *, mode: str = "observe", seconds: float = 500.0,
                    allowance: tuple[int, int] | None = (1_000_000, 100_000)) -> tuple[JudgeOutcome, Any, Any]:
        provider = FakeIdmProvider({JUDGE: answers})
        runtime = make_runtime(provider, seconds=seconds, allowance=allowance)
        outcome = await run_judge(runtime, mode=mode, plan_summary={"segment": "context_explain"},  # type: ignore[arg-type]
                                  facts=[("d1-c4-f2", "Cấp 1 | Ảnh hưởng thấp")], unit=unit(html("Ba cấp."), problem()))
        return outcome, provider, runtime

    async def test_statuses(self) -> None:
        outcome, provider, _ = await self.judge([], mode="off")
        self.assertEqual((outcome.status, provider.calls), ("not_run", []))
        outcome, provider, _ = await self.judge([], seconds=20)
        self.assertEqual((outcome.status, provider.calls), ("skipped_budget", []))
        outcome, provider, _ = await self.judge([], allowance=(1_000_000, 1_000))
        self.assertEqual((outcome.status, provider.calls), ("skipped_budget", []))
        outcome, _, runtime = await self.judge(["not json"])
        self.assertEqual(outcome.status, "failed")
        assert_node_trace(self, runtime.trace)
        outcome, _, _ = await self.judge([judge_payload()])
        self.assertEqual((outcome.status, len(outcome.findings)), ("pass", 9))
        outcome, _, _ = await self.judge([judge_payload(("Q4_feedback_teaches", "major", 1))])
        self.assertEqual(outcome.status, "review_required")
        outcome, _, _ = await self.judge([judge_payload(("Q5_grounded_criteria", "critical", 1))])
        self.assertEqual(outcome.status, "reject")

    async def test_provider_errors_never_discard_the_unit(self) -> None:
        for terminal in (False, True):
            outcome, provider, runtime = await self.judge([IdmProviderError("AI_PROVIDER_X", terminal=terminal)])
            self.assertEqual((outcome.status, outcome.findings), ("failed", []))
            self.assertEqual(len(provider.calls), 1)
            self.assertEqual(runtime.provider_failure_code, None if terminal else "AI_PROVIDER_X")

    async def test_findings_are_bounded_and_sanitised(self) -> None:
        payload = judge_payload(("Q1_support_sufficient", "minor", 3), ("Q4_feedback_teaches", "major", 1))
        outcome, provider, runtime = await self.judge([payload])
        self.assertEqual(len(outcome.findings), 8)  # component 3 does not exist in a two-component unit
        self.assertTrue(all(item.witness == "‹b›trích‹/b›" for item in outcome.findings))  # noqa: RUF001
        call = provider.calls[0]
        self.assertEqual((call["thinking_level"], call["max_output_tokens"]), ("low", 3_000))
        self.assertNotIn("cp2_", call["prompt"])
        self.assertEqual([(e["phase"], e["invocation_kind"]) for e in runtime.trace], [("idm_w6_judge", "evaluator")])


class AuthorNoteAndQualityTests(unittest.TestCase):
    def test_author_note_templates(self) -> None:
        review = JudgeOutcome("review_required", [finding("Q4_feedback_teaches", "major", 1),
                                                  finding("Q8_language", "minor", 0)])
        vi = build_unit_author_note(locale="vi", judge=review, deterministic_codes=["IDM_W5_X", "IDM_W5_X"],
                                    fallback_slots=[1], ai_drafted=True)
        self.assertEqual(vi, "QA tự động: 1 vấn đề cần xem — phản hồi chưa giải thích từng lựa chọn (Q4). "
                             "1 khối dùng bản dự phòng dựng từ tài liệu — cần biên tập. "
                             "Kiểm tra tự động còn cảnh báo: IDM_W5_X. "
                             "Tình huống minh hoạ do AI soạn, cần SME xác nhận tính thực tế.")
        en = build_unit_author_note(locale="en", judge=review, deterministic_codes=[], fallback_slots=[],
                                    ai_drafted=True)
        self.assertEqual(en, "Automatic QA: 1 issue(s) to review — feedback does not explain each option (Q4). "
                             "The illustrative scenario was drafted by AI; the SME should confirm it is realistic.")
        cases = {
            ("vi", "pass"): "QA tự động: không phát hiện vấn đề lớn.",
            ("en", "pass"): "Automatic QA: no major issue found.",
            ("vi", "skipped_budget"): "QA tự động: bỏ qua do hết thời gian.",
            ("en", "skipped_budget"): "Automatic QA: skipped (time budget).",
            ("vi", "failed"): "QA tự động: không chạy được, cần rà soát thủ công.",
            ("en", "failed"): "Automatic QA: unavailable, review manually.",
            ("vi", "not_run"): "",
        }
        for (locale, status), expected in cases.items():
            note = build_unit_author_note(locale=locale, judge=JudgeOutcome(status),  # type: ignore[arg-type]
                                          deterministic_codes=[], fallback_slots=[], ai_drafted=False)
            self.assertEqual(note, expected)
        en_fallback = build_unit_author_note(locale="en", judge=JudgeOutcome("not_run"),
                                             deterministic_codes=["B_CODE", "A_CODE"], fallback_slots=[0, 1],
                                             ai_drafted=False)
        self.assertEqual(en_fallback, "2 block(s) use the source-based fallback — edit them. "
                                      "Automatic checks still warn: A_CODE, B_CODE.")

    def test_author_note_is_bounded_and_has_no_angle_brackets(self) -> None:
        many = JudgeOutcome("reject", [finding(criterion, "critical", 0, "<script>") for criterion in CRITERIA])
        codes = [f"IDM_CODE_{index:02d}" for index in range(20)]
        for locale in ("vi", "en"):
            note = build_unit_author_note(locale=locale, judge=many, deterministic_codes=codes,
                                          fallback_slots=[0, 1, 2], ai_drafted=True)
            self.assertLessEqual(len(note), 1_500)
            self.assertNotIn("<", note)
            self.assertNotIn(">", note)
            self.assertIn("9", note.split(":")[1])
            self.assertEqual(note.count("(Q"), 4)
            self.assertIn("IDM_CODE_05", note)
            self.assertNotIn("IDM_CODE_06", note)

    def test_quality_counts_and_criteria(self) -> None:
        findings = [finding("Q4_feedback_teaches", "minor", 1), finding("Q4_feedback_teaches", "major", 1),
                    finding("Q4_feedback_teaches", "pass", 1), finding("Q5_grounded_criteria", "critical", 0),
                    finding("Q8_language", "minor"), finding("Q1_support_sufficient", "pass")]
        quality = build_unit_quality(mode="repair", judge=JudgeOutcome("reject", findings), repair_applied=True,
                                     deterministic_codes=["IDM_W5_B", "IDM_W5_A", "IDM_W5_B"], author_note="ghi chú")
        IdmUnitQualityV1.model_validate(quality.model_dump(mode="json"))
        self.assertEqual(quality.finding_counts.model_dump(), {"minor": 2, "major": 1, "critical": 1})
        self.assertEqual(quality.criteria, {"Q4_feedback_teaches": "major", "Q5_grounded_criteria": "critical",
                                            "Q8_language": "minor", "Q1_support_sufficient": "pass"})
        self.assertEqual(quality.deterministic_codes, ["IDM_W5_A", "IDM_W5_B"])
        self.assertEqual((quality.judge_mode, quality.judge_status, quality.repair_applied), ("repair", "reject", True))
        self.assertEqual(worst_counts([]).model_dump(), {"minor": 0, "major": 0, "critical": 0})
        self.assertEqual(criteria_summary([]), {})
        self.assertEqual(repair_targets(findings), [0, 1])
        self.assertEqual(blocking_count(findings), 2)
        capped = build_unit_quality(mode="off", judge=JudgeOutcome("not_run"), repair_applied=False,
                                    deterministic_codes=[f"IDM_C{index:02d}" for index in range(40)], author_note="")
        self.assertEqual(len(capped.deterministic_codes), 32)
        self.assertEqual(copy.deepcopy(capped.finding_counts.model_dump()), {"minor": 0, "major": 0, "critical": 0})


if __name__ == "__main__":
    unittest.main()
