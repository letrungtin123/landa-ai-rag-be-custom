"""Owner decisions of 2026-10-20 on the ``/unit`` route: repairs never leave a slot worse (P1), a second targeted
repair (P3), the judge's repair mode (P4) and FAQ wording findings settled without a repair call (P7).

``app.main.generate_content`` is a local fake replaying recorded-style answers; nothing reaches the network.
"""

from __future__ import annotations

import dataclasses
import json
import os
import unittest
from typing import Any
from unittest.mock import patch

from app.core.config import Settings
from app.idm.storyboard import run_idm_unit
from app.schemas.orchestration_v2 import RagLessonAuthorUnitV2Request
from app.services.orchestration_v2 import unit as unit_service
from tests.support.idm_test_support import make_runtime
from tests.support.log_support import logged_events
from tests.unit.idm.test_idm_faq import EXPLAINED, WHAT_IF, faq_writer
from tests.unit.idm.test_idm_fidelity_unit import (
    block,
    copied_c0,
    explain_body,
    explained_c0,
    html_texts,
    owned_of,
    writer_with,
)
from tests.unit.idm.test_idm_qc364564 import copying_worksheet
from tests.unit.idm.test_idm_storyboard import (
    JUDGE,
    ONE_OPTION,
    REPAIR,
    WRITER,
    FakeGenerate,
    StoryboardEndpointTestCase,
    escalate_body,
    escalate_writer,
    judge,
    served,
    severity_writer,
    slot_repair,
    worksheet_body,
)

# One prose sentence of 43 words: IDM_W5_SENTENCE_LENGTH (a wording finding).
LONG = ("Khi một khách hàng gọi đến để phàn nàn về một đơn hàng giao trễ hoặc một sản phẩm bị lỗi, bạn cần lắng nghe "
        "hết câu chuyện, ghi lại các dấu hiệu chính, đối chiếu với bảng cấp độ và chỉ sau đó mới quyết định ai sẽ xử "
        "lý khiếu nại này.")
SHORT = ["Hãy nghe hết câu chuyện của khách.", "Ghi lại các dấu hiệu chính.",
         "Đối chiếu với bảng cấp độ rồi mới quyết định ai xử lý."]
KEY = "Cấp 2 vì khách hàng phàn nàn lần thứ hai"


def long_c0(body: dict[str, Any]) -> dict[str, Any]:
    c0 = explained_c0(owned_of(body))
    c0["semantic_content"]["sections"][1]["blocks"].append(block("paragraph", text=LONG))
    return c0


def rewritten_c0(body: dict[str, Any], *extra: str) -> dict[str, Any]:
    """The long sentence split, plus ``extra`` paragraphs (what the repair adds)."""

    c0 = explained_c0(owned_of(body))
    c0["semantic_content"]["sections"][1]["blocks"].extend(block("paragraph", text=text) for text in [*SHORT, *extra])
    return c0


def two_slot_repair(body: dict[str, Any], c0: dict[str, Any]) -> dict[str, Any]:
    good = severity_writer(body)
    return {"components": {**slot_repair(writer_with(body, c0), 0)["components"],
                           **slot_repair(good, 1)["components"]}}


def guard_events(records: Any) -> list[dict[str, Any]]:
    return logged_events(records, "idm_unit_repair_guard")


class RepairGuardTests(StoryboardEndpointTestCase):
    """P1: after a repair the findings are compared slot by slot."""

    def writer(self, body: dict[str, Any]) -> dict[str, Any]:
        writer = writer_with(body, long_c0(body))
        writer["components"]["c1"]["explanation"] = ONE_OPTION
        return writer

    async def test_a_slot_the_repair_made_worse_keeps_the_writers_version(self) -> None:
        body = explain_body()
        # The repair fixes the question (c1) and splits the long sentence of c0, but names a company the facts never
        # name: c0 goes back to the writer's version, the fixed question stays.
        repair = two_slot_repair(body, rewritten_c0(body, "Công ty Vinamilk cũng xếp khiếu nại theo ba cấp như vậy."))
        provider = FakeGenerate(**{WRITER: [self.writer(body)], REPAIR: [repair], JUDGE: [judge()]})
        with self.assertLogs("app.idm", "INFO") as logs:
            status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "validated", "provider")
        self.assertEqual(provider.names, [WRITER, REPAIR, JUDGE])
        c0, c1 = data["unit"]["components"][:2]
        self.assertIn(LONG, html_texts(c0))
        self.assertNotIn("Vinamilk", json.dumps(c0, ensure_ascii=False))
        self.assertEqual(c1["explanation"], served(body, severity_writer(body), 1)["explanation"])
        self.assertTrue(quality.repair_applied)
        self.assertEqual([(event["reverted_slots"], event["whole_repair"], event["added_codes"])
                          for event in guard_events(logs.records)], [([0], False, ["IDM_W5_NEW_FACT"])])
        # The long sentence left is a polish note; the question was fixed.
        self.assertIn("Đã tự sửa: bài luyện tập thiếu phần.", quality.author_note)
        self.assertIn("Nên biên tập khối 1 (Lý thuyết): nhiều câu quá dài", quality.author_note)
        self.assertNotIn("số liệu hoặc tên riêng", quality.author_note)

    async def test_a_leak_the_new_html_puts_into_the_question_drops_the_whole_repair(self) -> None:
        body = explain_body()
        # The rewritten html now shows the key as its worked example: the question gives its answer away whatever its
        # own version (the writer's question copies the new example too), so nothing of the repair is kept.
        repair = two_slot_repair(body, rewritten_c0(body, f"Ví dụ minh hoạ: {KEY}."))
        provider = FakeGenerate(**{WRITER: [self.writer(body)], REPAIR: [repair], JUDGE: [judge()]})
        with self.assertLogs("app.idm", "INFO") as logs:
            status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "structured_fallback", "review_required", "provider")
        self.assertEqual(provider.names, [WRITER, REPAIR, JUDGE])
        c0, c1 = data["unit"]["components"][:2]
        self.assertIn(LONG, html_texts(c0))
        self.assertNotIn(KEY, json.dumps(c0, ensure_ascii=False))
        # The writer's question still misses its option reasons: the slot falls back as before the repair.
        self.assertTrue(c1["source_locked_fallback"])
        self.assertFalse(quality.repair_applied)
        self.assertEqual([(event["reverted_slots"], event["whole_repair"], event["added_codes"])
                          for event in guard_events(logs.records)], [([1], True, ["IDM_W5_ANSWER_LEAK"])])


class SecondRepairTests(StoryboardEndpointTestCase):
    """P3: a new fact the first repair left gets a second targeted repair that quotes it."""

    async def test_a_new_name_left_by_the_first_repair_is_quoted_and_removed_by_the_second(self) -> None:
        body = explain_body()
        named = rewritten_c0(body, "Công ty Vinamilk cũng xếp khiếu nại theo ba cấp như vậy.")
        writer = writer_with(body, named)
        fixed = rewritten_c0(body, "Nhiều công ty cũng xếp khiếu nại theo ba cấp như vậy.")
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [slot_repair(writer, 0), slot_repair(
            writer_with(body, fixed), 0)], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "validated", "provider")
        self.assertEqual(provider.names, [WRITER, REPAIR, REPAIR, JUDGE])
        first, second = provider.calls[1], provider.calls[2]
        self.assertIn('(new: "Vinamilk")', first["prompt"])
        self.assertNotIn("SECOND ATTEMPT", first["prompt"])
        self.assertEqual(second["thinking_level"], "low")
        self.assertIn('c0 IDM_W5_NEW_FACT: SECOND ATTEMPT: the listed texts still state the quoted numbers or names',
                      second["prompt"])
        self.assertIn('(new: "Vinamilk")', second["prompt"])
        self.assertNotIn("Vinamilk", json.dumps(data["unit"]["components"][0], ensure_ascii=False))
        self.assertIn("Đã tự sửa: số liệu hoặc tên riêng không có trong tài liệu.", quality.author_note)


    async def test_a_second_repair_that_fails_leaves_the_slot_for_review_and_says_why(self) -> None:
        body = worksheet_body()
        writer = copying_worksheet(body)
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [slot_repair(writer, 1), "{}"], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "review_required", "provider")
        self.assertEqual(provider.names, [WRITER, REPAIR, REPAIR, JUDGE])
        self.assertIn("Giữ bản AI để tác giả rà soát: khối 2 (Quiz): phương án lộ đáp án.", quality.author_note)
        self.assertIn("Sửa tự động không thành công: lần sửa của AI sai định dạng.", quality.author_note)


class UnitLevelRevertTests(unittest.IsolatedAsyncioTestCase):
    """The guard compares slots; a validator finding without a slot still drops a review-only repair."""

    @dataclasses.dataclass(frozen=True)
    class Finding:
        code: str
        path: str

    async def test_a_unit_level_finding_after_a_wording_repair_keeps_the_writers_draft(self) -> None:
        body = explain_body()
        writer = writer_with(body, copied_c0(owned_of(body)))
        repaired = slot_repair(writer_with(body, explained_c0(owned_of(body))), 0)
        request = RagLessonAuthorUnitV2Request.model_validate(body)
        deps = unit_service._idm_unit_deps(request)

        def validate(unit: dict[str, Any], expected: dict[str, Any]) -> Any:
            # The repaired slot explains its rows ("Cấp độ cho bạn biết điều gì?"): the unit fails as a whole then.
            if "Cấp độ cho bạn biết điều gì?" in json.dumps(unit, ensure_ascii=False):
                return self.Finding("UNIT_TITLE_INVALID", "title")
            return deps.validate_unit(unit, expected)

        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [repaired]})
        result = await run_idm_unit(
            contract=request.unit_contract, runtime=make_runtime(provider, allowance=None), fallback_only=False,
            deps=dataclasses.replace(deps, validate_unit=validate, judge_mode="off"))
        self.assertEqual(provider.names, [WRITER, REPAIR])
        self.assertEqual(result["content_origin"], "provider_validated")
        self.assertFalse(result["unit"]["idm_quality"]["repair_applied"])
        self.assertNotIn("Cấp độ cho bạn biết điều gì?", json.dumps(result["unit"], ensure_ascii=False))


class JudgeRepairTests(StoryboardEndpointTestCase):
    """P4: the judge repairs by default, under the same rules as the W5 repairs."""

    def test_repair_is_the_default_judge_mode_and_the_environment_overrides_it(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(Settings().idm_judge_mode, "repair")
        with patch.dict(os.environ, {"AI_RAG_IDM_JUDGE_MODE": "observe"}, clear=True):
            self.assertEqual(Settings().idm_judge_mode, "observe")

    async def test_a_judge_repair_that_fixes_a_slot_kept_for_review_clears_its_mark(self) -> None:
        body = worksheet_body()
        writer = copying_worksheet(body)
        fixed = slot_repair(writer, 1, choices=[
            {"text": "Cấp 1 vì chưa có thiệt hại tài chính", "correct": False},
            {"text": "Cấp 2, vì cùng một khách đã gọi phàn nàn lại", "correct": True},
            {"text": "Cấp 3 vì cần escalate ngay cho quản lý", "correct": False}])
        # Both W5 repairs leave the leak (kept for review); the judge finds it too and its repair settles it.
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [slot_repair(writer, 1)] * 2 + [fixed],
                                   JUDGE: [judge(("Q5_grounded_criteria", "major", 1)), judge()]})
        status, data, _ = await self.post(body, provider, mode="repair")
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "validated", "provider")
        self.assertEqual(provider.names, [WRITER, REPAIR, REPAIR, JUDGE, REPAIR, JUDGE])
        self.assertIn('{"code":"IDM_W6_Q5","path":"components[1]"}', provider.calls[4]["prompt"])
        self.assertEqual((quality.judge_status, quality.repair_applied, quality.judge_mode), ("pass", True, "repair"))
        self.assertIn("Đã tự sửa: phương án lộ đáp án.", quality.author_note)
        self.assertNotIn("Giữ bản AI", quality.author_note)

    async def test_a_judge_repair_keeps_only_the_slots_it_did_not_make_worse(self) -> None:
        body = escalate_body()
        writer = escalate_writer(body)
        better = writer["components"]["c0"]["explanation"] + " Hãy đối chiếu tiêu chí trước."
        items = writer["components"]["c1"]["items"]
        worse_faq = [{**items[0], "answer": items[0]["answer"] + " Khách VIP được giảm 20% phí."}, *items[1:]]
        repair = {"components": {**slot_repair(writer, 0, explanation=better)["components"],
                                 **slot_repair(writer, 1, items=worse_faq)["components"]}}
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [repair],
                                   JUDGE: [judge(("Q4_feedback_teaches", "major", 0), ("Q8_language", "major", 1)),
                                           judge()]})
        with self.assertLogs("app.idm", "INFO") as logs:
            status, data, _ = await self.post(body, provider, mode="repair")
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "validated", "provider")
        self.assertEqual(provider.names, [WRITER, JUDGE, REPAIR, JUDGE])
        c0, c1 = data["unit"]["components"][:2]
        self.assertTrue(c0["explanation"].endswith("Hãy đối chiếu tiêu chí trước."))
        self.assertEqual(c1["items"], items)
        self.assertEqual([(event["step"], event["reverted_slots"], event["whole_repair"])
                          for event in guard_events(logs.records)], [("w6", [1], False)])
        self.assertTrue(quality.repair_applied)


class StyleOnlyFaqTests(StoryboardEndpointTestCase):
    """P7: the repair call leaves out an FAQ slot whose findings are wording ones only."""

    async def test_the_question_is_repaired_without_its_recall_only_faq(self) -> None:
        body = escalate_body()
        writer = faq_writer(body, ("Khiếu nại cấp 3 là gì trong xử lý khiếu nại?", EXPLAINED),
                            ("Khiếu nại cấp 2 gồm những dấu hiệu nào?", WHAT_IF))
        good = escalate_writer(body)
        writer["components"]["c0"]["explanation"] = ONE_OPTION
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [slot_repair(good, 0)], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        self.assertEqual(provider.names, [WRITER, REPAIR, JUDGE])
        repair = provider.calls[1]
        self.assertEqual(list(repair["response_schema"].model_fields["components"].annotation.model_fields), ["c0"])
        self.assertNotIn("IDM_W5_FAQ_TYPE_MIX", repair["prompt"].split("REPAIR_REQUIREMENTS", 1)[1])
        self.assertEqual(data["unit"]["components"][1]["items"], writer["components"]["c1"]["items"])
        quality = self.assert_envelope(data, "provider_validated", "validated", "provider")
        self.assertIn("Nên biên tập khối 2 (Hỏi đáp): phần hỏi đáp chủ yếu hỏi lại kiến thức", quality.author_note)

    async def test_an_faq_with_a_new_fact_is_repaired_with_its_wording_findings(self) -> None:
        body = escalate_body()
        writer = faq_writer(body, ("Khiếu nại cấp 3 là gì trong xử lý khiếu nại?", EXPLAINED + " Hạn chót là 7 ngày."),
                            ("Khiếu nại cấp 2 gồm những dấu hiệu nào?", WHAT_IF))
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [slot_repair(escalate_writer(body), 1)],
                                   JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        rules = provider.calls[1]["prompt"].split("FAILING_RULES", 1)[1]
        self.assertIn("c1 IDM_W5_FAQ_NEW_FACT at c1.items[0]", rules)
        self.assertIn('(new: "7")', rules)
        self.assertIn("c1 IDM_W5_FAQ_TYPE_MIX at c1.items", rules)
        self.assert_envelope(data, "provider_validated", "validated", "provider")


if __name__ == "__main__":
    unittest.main()
