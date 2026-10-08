"""IDM ``generate_unit``: W5 storyboard writer + W6 judge (spec §6.8, §7.7, §8.3).

Unit tests drive ``/v1/lesson-author/orchestration-v2/unit`` through ``httpx.ASGITransport`` with
``app.main.generate_content`` replaced by a local fake; nothing reaches the network.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import unittest
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import HTTPException

from app import main
from app.api import deps as api_deps
from app.idm.contracts import IdmTreatmentRefV1, IdmUnitBriefV1, IdmUnitQualityV1, brief_hash_of
from app.idm.runtime import IdmStageError
from app.idm.storyboard import build_idm_expected, output_budget, parse_brief, run_idm_unit
from app.lesson_author_orchestration_v2_provider import UnitGenerationContractV2
from app.schemas.common import AiUsage
from app.schemas.orchestration_v2 import RagLessonAuthorUnitV2Request
from app.services.orchestration_v2 import unit as unit_service
from tests import idm_golden as g
from tests import idm_golden_unit as gu
from tests.idm_test_support import assert_node_trace, golden_design, golden_shard, make_runtime, rehash_contract

URL = "/v1/lesson-author/orchestration-v2/unit"
WRITER, REPAIR, JUDGE = "StagedInstancePayloadUnit", "StagedMultiRepair", "IdmJudgeResponseV1"
CRITERIA = ("Q1_support_sufficient", "Q2_not_copied", "Q3_practice_complete", "Q4_feedback_teaches",
            "Q5_grounded_criteria", "Q6_alignment", "Q7_cognitive_load", "Q8_language", "Q9_traceability")
ONE_OPTION = "Tiêu chí: khiếu nại liên quan an toàn bắt buộc phải escalate ngay cho quản lý; đây là cách xử lý đúng."


def body_for(chapter_index: int, lesson_position: int, unit_position: int) -> dict[str, Any]:
    design, _ = golden_design()
    return gu.build_unit_request(design, golden_shard(chapter_index), chapter_index=chapter_index,
                                 lesson_position=lesson_position, unit_position=unit_position, facts=g.source_facts())


def escalate_body() -> dict[str, Any]:
    """lsn_003 unit 2: problem (ai_drafted) + la_faq; no source-locked fallback can be built."""

    return body_for(1, 0, 1)


def severity_body() -> dict[str, Any]:
    """lsn_002 unit 1: html + problem (ai_drafted); the source-locked fallback is valid."""

    return body_for(0, 1, 0)


def escalate_writer(body: dict[str, Any]) -> dict[str, Any]:
    return gu.writer_response_escalate_practice(body["unit_contract"]["component_plan"])


def severity_writer(body: dict[str, Any], explanation: str | None = None) -> dict[str, Any]:
    owned = body["unit_contract"]["component_plan"][0]["source_fact_ids"]
    rows = [("Cấp 1", "Ảnh hưởng thấp, không thiệt hại tài chính. Nhân viên tự xử lý."),
            ("Cấp 2", "Thiệt hại tài chính dưới 50 triệu đồng hoặc khách hàng phàn nàn lần thứ hai. Thông báo "
                      "trưởng nhóm."),
            ("Cấp 3", "Liên quan an toàn, pháp lý hoặc truyền thông. Escalate ngay cho quản lý và bộ phận Pháp chế "
                      "theo quy trình CR-04.")]
    return {"components": {
        "c0": {"title": "Dấu hiệu của từng cấp độ", "selection_rationale": "Bảng so sánh ba cấp độ.",
               "covered_source_fact_ids": owned, "html": None, "semantic_content": {"sections": [{
                   "heading": "Khiếu nại thuộc cấp độ nào?", "learning_block_ids": [], "blocks": [
                       {"kind": "paragraph", "text": "Mỗi khiếu nại thuộc một trong ba cấp độ. Cấp độ quyết định "
                                                     "ai xử lý.", "items": [], "rows": []},
                       {"kind": "table", "text": None, "items": [],
                        "rows": [{"label": label, "value": value} for label, value in rows]}]}]}},
        "c1": {"title": "Khiếu nại này thuộc cấp nào?", "selection_rationale": "Tình huống phân loại cấp độ.",
               "covered_source_fact_ids": [], "problem_type": "multiple_choice",
               "question": "Một khách hàng gọi lần thứ hai để phàn nàn vì đơn hàng giao trễ, chưa có thiệt hại tài "
                           "chính. Khiếu nại này thuộc cấp độ nào?",
               "choices": [{"text": "Cấp 1 vì chưa có thiệt hại tài chính", "correct": False},
                           {"text": "Cấp 2 vì khách hàng phàn nàn lần thứ hai", "correct": True},
                           {"text": "Cấp 3 vì cần escalate ngay cho quản lý", "correct": False}],
               "explanation": explanation or (
                   "Tiêu chí: khách hàng phàn nàn lần thứ hai là dấu hiệu của cấp 2. A — sai vì lần phàn nàn thứ hai "
                   "đã vượt cấp 1; B — đúng vì khớp dấu hiệu cấp 2 trong bảng; C — sai vì không có yếu tố an toàn, "
                   "pháp lý hay truyền thông.")}}}


UNGROUNDED_ANSWER = ("Nên tặng phiếu giảm giá 20% cho khách VIP để giữ chân họ lâu dài hơn trong mùa mua sắm cao "
                     "điểm cuối năm.")


def escalate_with_faq(body: dict[str, Any], *answers: str) -> dict[str, Any]:
    writer = escalate_writer(body)
    writer["components"]["c1"]["items"] = [
        {"question": f"Điều gì dễ nhầm số {index} khi xử lý khiếu nại?", "answer": answer}
        for index, answer in enumerate(answers, start=1)]
    return writer


def worksheet_body() -> dict[str, Any]:
    """lsn_002 unit 1 with its html slot planned as the worksheet of the unit's practice (spec §10.1)."""

    body = severity_body()
    brief = body["unit_contract"]["idm_unit_brief"]
    brief["components"][0].update(role="practice", practice=brief["components"][1]["practice"])
    brief["brief_hash"] = brief_hash_of(brief)
    rehash_contract(body["unit_contract"])
    return body


def worksheet_writer(body: dict[str, Any]) -> dict[str, Any]:
    writer = severity_writer(body)
    rows = writer["components"]["c0"]["semantic_content"]["sections"][0]["blocks"][1]["rows"]

    def section(heading: str, block: dict[str, Any]) -> dict[str, Any]:
        return {"heading": heading, "learning_block_ids": [], "blocks": [{"text": None, "items": [], "rows": [],
                                                                         **block}]}

    writer["components"]["c0"]["semantic_content"] = {"sections": [
        section("Bạn cần làm gì với ba khiếu nại mẫu?", {"kind": "task", "text": (
            "Đọc mô tả ba khiếu nại mẫu, xác định cấp độ của từng khiếu nại và ghi người cần xử lý vào bảng "
            "của bạn. Dựa vào dấu hiệu của từng cấp độ trong bảng bên dưới.")}),
        section("Bảng cần điền cho từng khiếu nại", {"kind": "table", "rows": rows}),
        section("Ví dụ một dòng đã điền", {"kind": "paragraph", "text": (
            "Khách hàng gọi lần thứ hai vì đơn giao trễ, chưa có thiệt hại tài chính: ghi cấp 2 và thông báo "
            "trưởng nhóm.")}),
        section("Tự kiểm tra bảng của bạn", {"kind": "bullets", "items": [
            "Cấp 2 khi thiệt hại tài chính dưới 50 triệu đồng hoặc khách hàng phàn nàn lần thứ hai.",
            "Cấp 3 khi liên quan an toàn, pháp lý hoặc truyền thông: escalate ngay cho quản lý."]}),
    ]}
    return writer


def slot_repair(writer: dict[str, Any], index: int, **changes: Any) -> dict[str, Any]:
    payload = {k: v for k, v in writer["components"][f"c{index}"].items() if k != "covered_source_fact_ids"}
    return {"components": {f"c{index}": {**payload, **changes}}}


def judge(*overrides: tuple[str, str, int | None]) -> dict[str, Any]:
    severity = {criterion: ("pass", None) for criterion in CRITERIA}
    severity.update({criterion: (level, index) for criterion, level, index in overrides})
    levels = {level for level, _ in severity.values()}
    verdict = "reject" if "critical" in levels else "review_required" if "major" in levels else "pass"
    return {"verdict": verdict, "findings": [{"criterion": c, "severity": s, "component_index": i, "witness": "w"}
                                             for c, (s, i) in severity.items()]}


def unavailable() -> Exception:
    return HTTPException(status_code=503, detail={"code": "AI_PROVIDER_UNAVAILABLE", "message": "private"})


class FakeGenerate:
    """Replays answers per schema family; records schema names, options and prompts."""

    def __init__(self, **queues: list[Any]) -> None:
        self.queues = {name: list(items) for name, items in queues.items()}
        self.calls: list[dict[str, Any]] = []

    @property
    def names(self) -> list[str]:
        return [call["family"] for call in self.calls]

    async def __call__(self, api_key: str, model: str, prompt: str, **options: Any) -> tuple[str, Any]:
        name = options["response_schema"].__name__
        family = next(family for family in (WRITER, REPAIR, JUDGE) if name.startswith(family))
        self.calls.append({"family": family, "name": name, "prompt": prompt, **options})
        answer = self.queues[family].pop(0)
        if isinstance(answer, Exception):
            raise answer
        text = answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
        return text, AiUsage(inputTokens=100, outputTokens=50, totalTokens=150)


class StoryboardEndpointTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        main.app.dependency_overrides[api_deps.require_internal_token] = lambda: None

    def tearDown(self) -> None:
        main.app.dependency_overrides.pop(api_deps.require_internal_token, None)

    async def post(self, body: dict[str, Any], provider: FakeGenerate | None = None, *,
                   mode: str = "observe") -> tuple[int, dict[str, Any], FakeGenerate]:
        provider = provider or FakeGenerate()
        with patch("app.services.provider.generate_content", provider), \
                patch.object(main.settings, "idm_judge_mode", mode):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://t") as client:
                reply = await client.post(URL, json=body)
        return reply.status_code, reply.json(), provider

    def assert_envelope(self, data: dict[str, Any], origin: str, state: str, source: str) -> IdmUnitQualityV1:
        self.assertEqual((data["content_origin"], data["quality_state"], data["usage_source"]), (origin, state, source))
        assert_node_trace(self, data["attempt_trace"])
        return IdmUnitQualityV1.model_validate(data["unit"]["idm_quality"])


class BriefContractTests(unittest.TestCase):
    def contract(self, change: Any = None, *, rehash_brief: bool = True) -> UnitGenerationContractV2:
        raw = escalate_body()["unit_contract"]
        if change is not None:
            change(raw["idm_unit_brief"])
            if rehash_brief:
                raw["idm_unit_brief"]["brief_hash"] = brief_hash_of(raw["idm_unit_brief"])
        return UnitGenerationContractV2.model_validate(rehash_contract(raw))

    def test_matching_brief_is_parsed(self) -> None:
        brief = parse_brief(self.contract())
        self.assertEqual([slot.type for slot in brief.components], ["problem", "la_faq"])
        self.assertEqual(brief.components[0].practice.scenario_origin, "ai_drafted")  # type: ignore[union-attr]

    def test_mismatches_are_rejected(self) -> None:
        def slot(index: int, **values: Any) -> Any:
            return lambda brief: brief["components"][index].update(values)

        cases = {
            "hash": (lambda brief: brief.update(brief_hash="0" * 64), False),
            "schema": (lambda brief: brief.update(pipeline_version="idm-2"), True),
            "slot_count": (lambda brief: brief["components"].pop(), True),
            "plan_id": (slot(0, component_plan_id="cp2_" + "f" * 32), True),
            "type": (slot(1, type="html"), True),
            "owned": (slot(0, owned_fact_keys=[]), True),
            "supporting": (slot(1, supporting_fact_keys=[g.key(6, 2)]), True),
        }
        for name, (change, rehash) in cases.items():
            with self.subTest(case=name), self.assertRaises(IdmStageError) as caught:
                parse_brief(self.contract(change, rehash_brief=rehash))
            self.assertEqual(caught.exception.code, "IDM_W5_BRIEF_CONTRACT_MISMATCH")


class OutputBudgetTests(unittest.TestCase):
    def test_segment_budget_extra_blocks_and_clamp(self) -> None:
        contract = UnitGenerationContractV2.model_validate(severity_body()["unit_contract"])
        brief = IdmUnitBriefV1.model_validate(contract.idm_unit_brief)
        budget = output_budget(brief, contract)
        self.assertEqual((budget["max_words"], budget["max_visible_chars"]), (450, 3_150))
        self.assertEqual(budget["source_content_chars"], sum(len(f.fact_text) for f in contract.source_facts))
        self.assertGreater(budget["source_estimated_words"], 20)

        def variant(segment: str, blocks: int) -> int:
            treatments = [IdmTreatmentRefV1(block_id=f"cb_{n:04d}", treatment="condense", detail_level="d")
                          for n in range(1, blocks + 1)]
            slot = brief.components[0].model_copy(update={"treatments": treatments})
            return output_budget(brief.model_copy(update={"unit_segment": segment, "components": [slot]}),
                                 contract)["max_words"]

        expected = {("context_explain", 3): 690, ("example", 1): 350, ("practice_feedback", 2): 520,
                    ("summary_apply", 1): 250, ("job_aid", 1): 600, ("job_aid", 20): 1_600}
        for (segment, blocks), words in expected.items():
            self.assertEqual(variant(segment, blocks), words, (segment, blocks))
        self.assertEqual(variant("summary_apply", 0), 250)

    def test_expected_uses_the_idm_budget(self) -> None:
        body = severity_body()
        request = RagLessonAuthorUnitV2Request.model_validate(body)
        brief = parse_brief(request.unit_contract)
        expected = build_idm_expected(request.unit_contract, brief, unit_service._idm_unit_deps(request), "vi")
        self.assertEqual(expected["instructional_output_budget"]["max_words"], 450)
        self.assertEqual(expected["component_types"], ["html", "problem"])
        self.assertEqual(expected["diagram_relationships_by_plan_id"], {})


class UnitEndpointTests(StoryboardEndpointTestCase):
    async def test_happy_path_is_provider_validated(self) -> None:
        body = escalate_body()
        status, data, provider = await self.post(body, FakeGenerate(**{WRITER: [escalate_writer(body)],
                                                                       JUDGE: [judge()]}))
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "validated", "provider")
        self.assertTrue(data["usage_complete"])
        self.assertEqual(data["usage"]["totalTokens"], 300)
        self.assertEqual(provider.names, [WRITER, JUDGE])
        self.assertEqual((provider.calls[0]["thinking_level"], provider.calls[0]["max_output_tokens"]),
                         ("medium", 16_000))
        self.assertEqual((quality.judge_mode, quality.judge_status, quality.repair_applied),
                         ("observe", "pass", False))
        self.assertEqual(quality.author_note, "QA tự động: không phát hiện vấn đề lớn. Tình huống minh hoạ do AI "
                                              "soạn, cần SME xác nhận tính thực tế.")
        plans = body["unit_contract"]["component_plan"]
        self.assertEqual([c["component_plan_id"] for c in data["unit"]["components"]],
                         [plan["component_plan_id"] for plan in plans])
        self.assertEqual(data["unit"]["component_plan"], plans)
        self.assertEqual([(e["phase"], e["invocation_kind"], e["outcome"]) for e in data["attempt_trace"]],
                         [("idm_w5_writer", "writer", "succeeded"), ("idm_w6_judge", "evaluator", "succeeded")])

    async def test_invalid_writer_json_twice_uses_the_source_locked_unit(self) -> None:
        status, data, provider = await self.post(severity_body(), FakeGenerate(**{WRITER: ["not json", "{}"]}))
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "structured_fallback", "review_required", "reserved_upper_bound")
        self.assertFalse(data["usage_complete"])
        self.assertEqual(data["usage"], {"inputTokens": 0, "outputTokens": 0, "embeddingTokens": 0, "totalTokens": 0})
        self.assertEqual(provider.names, [WRITER, WRITER])
        self.assertIn("REPAIR_REQUIREMENTS", provider.calls[1]["prompt"])
        self.assertEqual(quality.judge_status, "not_run")
        self.assertIn("IDM_W5_INSTANCE_INVALID", quality.deterministic_codes)
        self.assertTrue(all(c["source_locked_fallback"] for c in data["unit"]["components"]))
        self.assertEqual([(e["invocation_kind"], e["outcome"], e["failure_code"]) for e in data["attempt_trace"]],
                         [("writer", "failed", "IDM_W5_INSTANCE_INVALID"),
                          ("repair", "failed", "IDM_W5_INSTANCE_INVALID"),
                          ("deterministic", "fallback", "IDM_W5_UNIT_FALLBACK")])

    async def test_no_buildable_fallback_is_422(self) -> None:
        status, data, _ = await self.post(escalate_body(), FakeGenerate(**{WRITER: ["{}", "{}"]}))
        self.assertEqual((status, data["detail"]["code"]), (422, "ORCHESTRATION_V2_UNIT_FALLBACK_INVALID"))

    async def test_incomplete_practice_gets_one_slot_repair(self) -> None:
        body = severity_body()
        writer = severity_writer(body, ONE_OPTION)
        good = severity_writer(body)
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [slot_repair(good, 1)], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "validated", "provider")
        self.assertEqual(provider.names, [WRITER, REPAIR, JUDGE])
        repair = provider.calls[1]
        self.assertEqual((repair["name"], repair["thinking_level"]), ("StagedMultiRepairIdmWire", "medium"))
        slots = repair["response_schema"].model_fields["components"].annotation
        self.assertEqual(list(slots.model_fields), ["c1"])
        self.assertIn('{"code":"IDM_W5_PRACTICE_INCOMPLETE","path":"components[1]"}', repair["prompt"])
        self.assertTrue(quality.repair_applied)
        self.assertEqual(quality.deterministic_codes, ["IDM_W5_PRACTICE_INCOMPLETE"])
        self.assertEqual(data["unit"]["components"][1]["explanation"], good["components"]["c1"]["explanation"])
        self.assertEqual([e["phase"] for e in data["attempt_trace"]],
                         ["idm_w5_writer", "idm_w5_repair", "idm_w6_judge"])

    async def test_failed_slot_repair_falls_back_for_that_slot(self) -> None:
        body = severity_body()
        writer = severity_writer(body, ONE_OPTION)
        failures = {"invalid": "{}", "still_bad": slot_repair(writer, 1), "provider": unavailable()}
        for name, answer in failures.items():
            with self.subTest(repair=name):
                provider = FakeGenerate(**{WRITER: [writer], REPAIR: [answer], JUDGE: [judge()]})
                status, data, _ = await self.post(body, provider)
                self.assertEqual(status, 200)
                source = "reserved_upper_bound" if name == "provider" else "provider"
                quality = self.assert_envelope(data, "structured_fallback", "review_required", source)
                components = data["unit"]["components"]
                self.assertNotIn("source_locked_fallback", components[0])
                self.assertTrue(components[1]["source_locked_fallback"])
                self.assertIn("1 khối dùng bản dự phòng", quality.author_note)
                # QC course 234653 (D16): the note says why the slot fell back.
                self.assertIn("Lý do: khối 2 (Quiz) — IDM_W5_PRACTICE_INCOMPLETE.", quality.author_note)
                repair_code = {"invalid": "IDM_W5_REPAIR_INVALID", "provider": "AI_PROVIDER_UNAVAILABLE"}.get(name)
                if repair_code:
                    self.assertIn(f"Sửa tự động không thành công: {repair_code}.", quality.author_note)
                self.assertEqual(data["attempt_trace"][-2 if name != "provider" else -1]["failure_code"],
                                 "IDM_W5_SLOT_FALLBACK")
                # A transient provider failure also blocks the judge call of the same task.
                expected = [WRITER, REPAIR] if name == "provider" else [WRITER, REPAIR, JUDGE]
                self.assertEqual(provider.names, expected)
                self.assertEqual(quality.judge_status, "failed" if name == "provider" else "pass")

    async def test_failed_slot_repair_without_slot_fallback_keeps_the_provider_slot_for_review(self) -> None:
        # Regression (run 2a5e9ff2): a prose-only practice slot has no source-locked rebuild. A slot that
        # only fails IDM pedagogy checks is kept as a reviewable draft instead of ending the unit in 422.
        body = escalate_body()
        writer = escalate_writer(body)
        writer["components"]["c0"]["explanation"] = ONE_OPTION
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: ["{}"], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "review_required", "provider")
        self.assertEqual(provider.names, [WRITER, REPAIR, JUDGE])
        self.assertEqual(data["unit"]["components"][0]["explanation"], ONE_OPTION)
        self.assertFalse(any(c.get("source_locked_fallback") for c in data["unit"]["components"]))
        self.assertEqual(quality.deterministic_codes, ["IDM_W5_PRACTICE_INCOMPLETE"])
        self.assertIn("Kiểm tra tự động còn cảnh báo: IDM_W5_PRACTICE_INCOMPLETE.", quality.author_note)
        self.assertNotIn("fallback", [e["outcome"] for e in data["attempt_trace"]])

    async def test_fallback_only_makes_no_provider_call(self) -> None:
        body = {**severity_body(), "fallback_only": True}
        status, data, provider = await self.post(body)
        self.assertEqual((status, provider.calls), (200, []))
        quality = self.assert_envelope(data, "structured_fallback", "review_required", "deterministic_fallback")
        self.assertTrue(data["usage_complete"])
        self.assertEqual(quality.judge_status, "not_run")
        self.assertEqual([e["event_code"] for e in data["attempt_trace"]], ["deterministic_fallback"])

    async def test_short_deadline_or_transient_writer_failure_falls_back(self) -> None:
        status, data, provider = await self.post({**severity_body(), "remaining_workflow_budget_ms": 1_000})
        self.assertEqual((status, provider.calls), (200, []))
        self.assert_envelope(data, "structured_fallback", "review_required", "reserved_upper_bound")
        status, data, provider = await self.post(severity_body(), FakeGenerate(**{WRITER: [unavailable()]}))
        self.assertEqual((status, provider.names), (200, [WRITER]))
        quality = self.assert_envelope(data, "structured_fallback", "review_required", "reserved_upper_bound")
        self.assertEqual(quality.author_note, "Cả bài dùng bản dự phòng dựng từ tài liệu — cần biên tập. "
                                              "Lý do: AI_PROVIDER_UNAVAILABLE.")
        self.assertEqual([(e["outcome"], e["failure_code"]) for e in data["attempt_trace"]],
                         [("failed", "AI_PROVIDER_UNAVAILABLE"), ("fallback", "IDM_W5_UNIT_FALLBACK")])

    async def test_ungrounded_faq_answer_gets_a_targeted_repair(self) -> None:
        # QC course 234653 (R6): an FAQ answer added a claim the source never makes.
        body = escalate_body()
        good = escalate_writer(body)
        items = good["components"]["c1"]["items"]
        bad = escalate_with_faq(body, items[0]["answer"], UNGROUNDED_ANSWER)
        provider = FakeGenerate(**{WRITER: [bad], REPAIR: [slot_repair(good, 1)], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "validated", "provider")
        self.assertEqual(provider.names, [WRITER, REPAIR, JUDGE])
        prompt = provider.calls[1]["prompt"]
        self.assertIn('{"code":"IDM_W5_FAQ_UNGROUNDED","path":"components[1]"}', prompt)
        self.assertIn("c1 IDM_W5_FAQ_UNGROUNDED at c1.items[1]: answer only from these facts", prompt)
        self.assertIn("Start from this slot's facts: [", prompt)
        self.assertNotIn(UNGROUNDED_ANSWER, prompt.split("REPAIR_REQUIREMENTS")[1])
        self.assertIn("la_faq answers restate only what SOURCE_FACTS or LESSON_CONTEXT_FACTS say", prompt)
        self.assertEqual(quality.deterministic_codes, ["IDM_W5_FAQ_UNGROUNDED"])
        self.assertEqual(data["unit"]["components"][1]["items"], items)

    async def test_still_ungrounded_faq_items_are_dropped_while_two_grounded_items_remain(self) -> None:
        body = escalate_body()
        grounded = [item["answer"] for item in escalate_writer(body)["components"]["c1"]["items"]]
        bad = escalate_with_faq(body, grounded[0], UNGROUNDED_ANSWER, grounded[1])
        provider = FakeGenerate(**{WRITER: [bad], REPAIR: [slot_repair(bad, 1)], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "validated", "provider")
        self.assertEqual([item["answer"] for item in data["unit"]["components"][1]["items"]], grounded)
        self.assertEqual(quality.deterministic_codes, ["IDM_W5_FAQ_ITEMS_DROPPED", "IDM_W5_FAQ_UNGROUNDED"])
        self.assertIn("Đã bỏ 1 câu hỏi đáp có nội dung ngoài tài liệu.", quality.author_note)
        self.assertNotIn("fallback", [e["outcome"] for e in data["attempt_trace"]])

    async def test_faq_without_two_grounded_items_follows_the_slot_fallback_rules(self) -> None:
        # The IDM FAQ fallback restates the unit's explicit source conditions (run c2e5ac41): the slot is replaced.
        body = escalate_body()
        grounded = escalate_writer(body)["components"]["c1"]["items"][0]["answer"]
        bad = escalate_with_faq(body, grounded, UNGROUNDED_ANSWER)
        provider = FakeGenerate(**{WRITER: [bad], REPAIR: [slot_repair(bad, 1)], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        self.assert_envelope(data, "structured_fallback", "review_required", "provider")
        faq = data["unit"]["components"][1]
        self.assertIs(faq["source_locked_fallback"], True)
        facts = " ".join(fact["fact_text"] for fact in body["unit_contract"]["source_facts"])
        self.assertTrue(all(item["answer"] in facts for item in faq["items"]))

    async def test_faq_without_two_grounded_items_and_no_fallback_is_kept_for_review(self) -> None:
        # No source-locked FAQ can be built for this unit: the slot is kept for author review, with the reason.
        body = escalate_body()
        grounded = escalate_writer(body)["components"]["c1"]["items"][0]["answer"]
        bad = escalate_with_faq(body, grounded, UNGROUNDED_ANSWER)
        provider = FakeGenerate(**{WRITER: [bad], REPAIR: [slot_repair(bad, 1)], JUDGE: [judge()]})
        with patch("app.services.orchestration_v2.unit.idm_source_faq", lambda *_args, **_kwargs: None):
            status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "review_required", "provider")
        self.assertEqual(len(data["unit"]["components"][1]["items"]), 2)
        self.assertIn("Giữ bản AI để tác giả rà soát: khối 2 (Hỏi đáp) — IDM_W5_FAQ_UNGROUNDED.",
                      quality.author_note)
        self.assertEqual(quality.deterministic_codes, ["IDM_W5_FAQ_UNGROUNDED"])

    async def test_worksheet_slot_for_a_doing_must_do(self) -> None:
        # QC course 234653 (R5): the html practice slot is written as a worksheet the problem then checks.
        body = worksheet_body()
        provider = FakeGenerate(**{WRITER: [worksheet_writer(body)], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200, data)
        self.assert_envelope(data, "provider_validated", "validated", "provider")
        self.assertEqual(provider.names, [WRITER, JUDGE])
        prompt = provider.calls[0]["prompt"]
        self.assertIn('is a WORKSHEET for its practice', prompt)
        self.assertIn('"role":"practice","title"', prompt)
        kinds = [block["kind"] for section in data["unit"]["components"][0]["semantic_content"]["sections"]
                 for block in section["blocks"]]
        self.assertEqual(kinds, ["task", "table", "paragraph", "bullets"])

    async def test_incomplete_worksheet_is_kept_for_review_not_replaced(self) -> None:
        body = worksheet_body()
        plain = severity_writer(body)
        provider = FakeGenerate(**{WRITER: [plain], REPAIR: [slot_repair(plain, 0)], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "review_required", "provider")
        self.assertIn("c0 IDM_W5_WORKSHEET_INCOMPLETE at c0: a worksheet slot needs a task block",
                      provider.calls[1]["prompt"])
        self.assertNotIn("source_locked_fallback", data["unit"]["components"][0])
        self.assertIn("Giữ bản AI để tác giả rà soát: khối 1 (Lý thuyết) — IDM_W5_WORKSHEET_INCOMPLETE.",
                      quality.author_note)

    async def test_brief_mismatch_is_422_without_provider_call(self) -> None:
        body = escalate_body()
        brief = body["unit_contract"]["idm_unit_brief"]
        brief["components"][0]["owned_fact_keys"] = []
        brief["brief_hash"] = brief_hash_of(brief)
        rehash_contract(body["unit_contract"])
        status, data, provider = await self.post(body)
        self.assertEqual((status, data["detail"]["code"], provider.calls), (422, "IDM_W5_BRIEF_CONTRACT_MISMATCH", []))


class JudgeEndpointTests(StoryboardEndpointTestCase):
    async def test_observe_major_finding_needs_review_without_repair(self) -> None:
        body = escalate_body()
        provider = FakeGenerate(**{WRITER: [escalate_writer(body)],
                                   JUDGE: [judge(("Q4_feedback_teaches", "major", 0))]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "review_required", "provider")
        self.assertEqual(provider.names, [WRITER, JUDGE])
        self.assertEqual((quality.judge_status, quality.finding_counts.major), ("review_required", 1))
        self.assertIn("1 vấn đề cần xem — phản hồi chưa giải thích từng lựa chọn (Q4)", quality.author_note)

    async def test_repair_mode_accepts_only_a_real_improvement(self) -> None:
        body = escalate_body()
        writer = escalate_writer(body)
        better = slot_repair(writer, 0, explanation=writer["components"]["c0"]["explanation"] + " Hãy đối chiếu "
                                                                                                "tiêu chí trước.")
        major = judge(("Q4_feedback_teaches", "major", 0))
        provider = FakeGenerate(**{WRITER: [writer], JUDGE: [major, judge()], REPAIR: [better]})
        status, data, _ = await self.post(body, provider, mode="repair")
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "validated", "provider")
        self.assertEqual(provider.names, [WRITER, JUDGE, REPAIR, JUDGE])
        self.assertIn('{"code":"IDM_W6_Q4","path":"components[0]"}', provider.calls[2]["prompt"])
        self.assertEqual((quality.judge_status, quality.repair_applied, quality.judge_mode), ("pass", True, "repair"))
        self.assertTrue(data["unit"]["components"][0]["explanation"].endswith("Hãy đối chiếu tiêu chí trước."))

        provider = FakeGenerate(**{WRITER: [writer], JUDGE: [major, major], REPAIR: [better]})
        status, data, _ = await self.post(body, provider, mode="repair")
        quality = self.assert_envelope(data, "provider_validated", "review_required", "provider")
        self.assertEqual(provider.names, [WRITER, JUDGE, REPAIR, JUDGE])
        self.assertEqual((quality.judge_status, quality.repair_applied), ("review_required", False))
        self.assertEqual(data["unit"]["components"][0]["explanation"], writer["components"]["c0"]["explanation"])

    async def test_repair_mode_keeps_the_draft_when_the_repair_fails(self) -> None:
        body = escalate_body()
        writer = escalate_writer(body)
        major = judge(("Q4_feedback_teaches", "major", 0), ("Q8_language", "minor", None))
        worse = slot_repair(writer, 0, explanation=ONE_OPTION)
        for name, answer in {"provider": unavailable(), "invalid": "{}", "fails_checks": worse}.items():
            with self.subTest(repair=name):
                provider = FakeGenerate(**{WRITER: [writer], JUDGE: [major], REPAIR: [answer]})
                status, data, _ = await self.post(body, provider, mode="repair")
                self.assertEqual(status, 200)
                # Node admits the validated lane only for provider-accounted content: a failed repair call
                # (reserved upper-bound accounting) makes the kept draft a reviewable structured draft.
                envelope = (("structured_fallback", "reserved_upper_bound") if name == "provider"
                            else ("provider_validated", "provider"))
                quality = self.assert_envelope(data, envelope[0], "review_required", envelope[1])
                self.assertEqual(provider.names, [WRITER, JUDGE, REPAIR])
                self.assertEqual((quality.judge_status, quality.repair_applied), ("review_required", False))
                self.assertEqual(data["unit"]["components"][0]["explanation"],
                                 writer["components"]["c0"]["explanation"])

    async def test_ai_drafted_scenario_needs_review_when_judge_does_not_pass_q5(self) -> None:
        body = escalate_body()
        status, data, provider = await self.post(body, FakeGenerate(**{WRITER: [escalate_writer(body)]}), mode="off")
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "review_required", "provider")
        self.assertEqual((provider.names, quality.judge_status, quality.judge_mode), ([WRITER], "not_run", "off"))
        self.assertEqual(quality.author_note,
                         "Tình huống minh hoạ do AI soạn, cần SME xác nhận tính thực tế.")
        for answer, status_name in (("not json", "failed"), (judge(("Q5_grounded_criteria", "minor", 0)), "pass")):
            provider = FakeGenerate(**{WRITER: [escalate_writer(body)], JUDGE: [answer]})
            _, data, _ = await self.post(body, provider)
            quality = self.assert_envelope(data, "provider_validated", "review_required", "provider")
            self.assertEqual(quality.judge_status, status_name)

    # Regression (fixed, spec §7.7.5): an ai_drafted scenario is ``validated`` only when the judge's Q5 verdict is pass.
    # A judge answer that omits Q5 leaves Q5 unconfirmed, yet storyboard._q5 returns "pass" for an empty
    # Q5 finding list (``all([])``), so the unit is marked validated without any grounding check.
    async def test_ai_drafted_scenario_without_a_q5_verdict_needs_review(self) -> None:
        body = escalate_body()
        no_q5 = judge()
        no_q5["findings"] = [item for item in no_q5["findings"] if item["criterion"] != "Q5_grounded_criteria"]
        provider = FakeGenerate(**{WRITER: [escalate_writer(body)], JUDGE: [no_q5]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        self.assertEqual(data["quality_state"], "review_required")

    async def test_source_scenario_with_judge_off_is_validated(self) -> None:
        body = severity_body()
        brief = body["unit_contract"]["idm_unit_brief"]
        brief["components"][1]["practice"]["scenario_origin"] = "source"
        brief["brief_hash"] = brief_hash_of(brief)
        rehash_contract(body["unit_contract"])
        provider = FakeGenerate(**{WRITER: [severity_writer(body)]})
        status, data, _ = await self.post(body, provider, mode="off")
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "validated", "provider")
        self.assertEqual((quality.judge_status, quality.author_note), ("not_run", ""))


class InjectedDepsTests(unittest.IsolatedAsyncioTestCase):
    """Branches that the real staged validators cannot reach deterministically."""

    @dataclasses.dataclass(frozen=True)
    class Finding:
        code: str
        path: str

    async def run_unit(self, validate: Any, *, review: bool = False,
                       unbuildable_slots: tuple[int, ...] = ()) -> dict[str, Any]:
        body = severity_body()
        request = RagLessonAuthorUnitV2Request.model_validate(body)
        deps = dataclasses.replace(unit_service._idm_unit_deps(request), validate_unit=validate, judge_mode="off",
                                   evidence_review_required=review)
        if unbuildable_slots:
            slots = [None if index in unbuildable_slots else component
                     for index, component in enumerate(deps.source_locked_components or [])]
            deps = dataclasses.replace(deps, source_locked_components=slots, source_locked_unit=None)
        generate = AsyncMock(side_effect=[(json.dumps(severity_writer(body), ensure_ascii=False),
                                           AiUsage(inputTokens=1, outputTokens=1, totalTokens=2))] * 2)
        return await run_idm_unit(contract=request.unit_contract, runtime=make_runtime(generate, allowance=None),
                                  deps=deps, fallback_only=False)

    async def test_unit_level_finding_without_slot_falls_back_for_the_whole_unit(self) -> None:
        def validate(unit: dict[str, Any], _expected: dict[str, Any]) -> Any:
            return None if unit.get("source_locked_fallback") else self.Finding("UNIT_TITLE_INVALID", "title")

        result = await self.run_unit(validate)
        self.assertEqual(result["content_origin"], "structured_fallback")
        self.assertTrue(result["unit"]["source_locked_fallback"])
        self.assertEqual(result["unit"]["idm_quality"]["deterministic_codes"], ["UNIT_TITLE_INVALID"])

    async def test_slot_that_stays_invalid_after_fallback_falls_back_for_the_whole_unit(self) -> None:
        def validate(unit: dict[str, Any], _expected: dict[str, Any]) -> Any:
            return None if unit.get("source_locked_fallback") else self.Finding("X_INVALID", "components[1].question")

        result = await self.run_unit(validate)
        self.assertTrue(result["unit"]["source_locked_fallback"])
        self.assertEqual([e["failure_code"] for e in result["attempt_trace"] if e["outcome"] == "fallback"],
                         ["IDM_W5_SLOT_FALLBACK", "IDM_W5_UNIT_FALLBACK"])

    async def test_evidence_review_marks_a_clean_unit_reviewable(self) -> None:
        result = await self.run_unit(lambda _unit, _expected: None, review=True)
        self.assertEqual((result["content_origin"], result["quality_state"]), ("provider_validated", "review_required"))

    async def test_slot_fallback_survives_an_unbuildable_sibling_slot(self) -> None:
        # Regression (run 2a5e9ff2): one unbuildable slot (e.g. la_faq without explicit conditions) used to
        # disable the source-locked fallback of every other slot of the unit.
        def validate(unit: dict[str, Any], _expected: dict[str, Any]) -> Any:
            first = unit["components"][0]
            return None if first.get("source_locked_fallback") else self.Finding("HTML_X", "components[0].html")

        result = await self.run_unit(validate, unbuildable_slots=(1,))
        self.assertEqual((result["content_origin"], result["quality_state"]),
                         ("structured_fallback", "review_required"))
        components = result["unit"]["components"]
        self.assertTrue(components[0]["source_locked_fallback"])
        self.assertNotIn("source_locked_fallback", components[1])
        self.assertNotIn("source_locked_fallback", result["unit"])
        self.assertEqual([e["failure_code"] for e in result["attempt_trace"] if e["outcome"] == "fallback"],
                         ["IDM_W5_SLOT_FALLBACK"])

    async def test_shared_validator_finding_on_an_unbuildable_slot_keeps_the_legacy_code(self) -> None:
        def validate(unit: dict[str, Any], _expected: dict[str, Any]) -> Any:
            return self.Finding("X_INVALID", "components[1].question")

        with self.assertRaises(IdmStageError) as caught:
            await self.run_unit(validate, unbuildable_slots=(1,))
        self.assertEqual(caught.exception.code, "ORCHESTRATION_V2_UNIT_FALLBACK_INVALID")

    async def test_invalid_fallback_raises_the_legacy_code(self) -> None:
        with self.assertRaises(IdmStageError) as caught:
            await self.run_unit(lambda _unit, _expected: self.Finding("X_INVALID", "title"))
        self.assertEqual(caught.exception.code, "ORCHESTRATION_V2_UNIT_FALLBACK_INVALID")


class LegacyUnitPathTests(StoryboardEndpointTestCase):
    async def test_request_without_brief_uses_the_legacy_writer(self) -> None:
        body = severity_body()
        contract = copy.deepcopy(body["unit_contract"])
        contract.pop("idm_unit_brief")
        body["unit_contract"] = rehash_contract(contract)
        legacy_writer = AsyncMock(side_effect=ValueError("legacy writer reached"))
        idm_unit = AsyncMock()
        with patch("app.services.lesson_author.staged.writer.generate_staged_lesson_author_proposal", legacy_writer), \
                patch("app.services.orchestration_v2.unit.run_idm_unit", idm_unit):
            status, data, provider = await self.post(body)
        self.assertEqual(status, 200)
        legacy_writer.assert_awaited_once()
        idm_unit.assert_not_awaited()
        self.assertEqual(provider.calls, [])
        self.assertNotIn("idm_quality", data["unit"])
        self.assertEqual(data["content_origin"], "structured_fallback")


if __name__ == "__main__":
    unittest.main()
