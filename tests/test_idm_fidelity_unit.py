"""QLT-5 C on the ``/unit`` route: an explained unit (content mode explain) end to end.

The golden severity unit (html + single-choice practice) with a brief of an onboarding course in explain mode: the
writer prompt and budget follow the mode, the W5 checks find copied or unexplained teaching and invented facts, a
targeted low-thinking repair runs, and a slot that still fails is kept for review, never replaced by the
source-locked verbatim copy. ``app.main.generate_content`` is a local fake; nothing reaches the network.
"""

from __future__ import annotations

import json
from typing import Any

from app.idm.contracts import brief_hash_of
from tests.idm_test_support import rehash_contract
from tests.test_idm_storyboard import (
    JUDGE,
    REPAIR,
    WRITER,
    FakeGenerate,
    StoryboardEndpointTestCase,
    judge,
    severity_body,
    severity_writer,
    slot_repair,
    unavailable,
)

TABLE_ROWS = [
    {"label": "Cấp 1", "value": "Ảnh hưởng thấp, không thiệt hại tài chính. Nhân viên tự xử lý."},
    {"label": "Cấp 2", "value": "Thiệt hại tài chính dưới 50 triệu đồng hoặc khách hàng phàn nàn lần thứ hai. Thông "
                                "báo trưởng nhóm."},
    {"label": "Cấp 3", "value": "Liên quan an toàn, pháp lý hoặc truyền thông. Escalate ngay cho quản lý và bộ phận "
                                "Pháp chế theo quy trình CR-04."},
]
# The three levels in short (the slot still lists every level; no cell repeats a paragraph below).
SHORT_ROWS = [{"label": "Cấp 1", "value": "Nhân viên tự xử lý."}, {"label": "Cấp 2", "value": "Thông báo trưởng nhóm."},
              {"label": "Cấp 3", "value": "Escalate cho quản lý và bộ phận Pháp chế."}]
COPIED = ["Ảnh hưởng thấp, không thiệt hại tài chính, nhân viên tự xử lý.",
          "Liên quan an toàn, pháp lý hoặc truyền thông: escalate ngay cho quản lý và bộ phận Pháp chế."]


def block(kind: str, **value: Any) -> dict[str, Any]:
    return {"kind": kind, "text": None, "items": [], "rows": [], **value}


def explain_body() -> dict[str, Any]:
    """The severity unit in an onboarding course explained to its learners (Node copies the design's mode)."""

    body = severity_body()
    brief = body["unit_contract"]["idm_unit_brief"]
    brief.update(purpose_profile="onboarding", fidelity_mode="explain",
                 purpose_summary="Giúp nhân viên mới biết cách công ty tiếp nhận và xử lý khiếu nại.")
    brief["brief_hash"] = brief_hash_of(brief)
    rehash_contract(body["unit_contract"])
    return body


def explained_c0(owned: list[str]) -> dict[str, Any]:
    return {"title": "Dấu hiệu của từng cấp độ", "selection_rationale": "Giải thích ba cấp độ.",
            "covered_source_fact_ids": owned, "html": None, "semantic_content": {"sections": [
                {"heading": "Cấp độ cho bạn biết điều gì?", "learning_block_ids": [], "blocks": [
                    block("paragraph", text="Mỗi khiếu nại thuộc một trong ba cấp độ. Cấp độ cho bạn biết ai xử lý và "
                                            "cần báo cho ai."),
                    block("paragraph", text="Vì sao quan trọng? Xếp đúng cấp giúp khách hàng được đúng người hỗ trợ, "
                                            "và bạn không phải tự quyết việc vượt quyền."),
                    block("table", rows=TABLE_ROWS)]},
                {"heading": "Áp dụng thế nào trong ca làm việc?", "learning_block_ids": [], "blocks": [
                    block("paragraph", text="Ví dụ: một khách hàng chỉ phản ánh việc chờ tổng đài hơi lâu và không "
                                            "có thiệt hại; bạn tự xử lý ngay trong cuộc gọi."),
                    block("paragraph", text="Dễ nhầm: khách nói gay gắt không làm khiếu nại lên cấp cao hơn; hãy xem "
                                            "dấu hiệu trong bảng.")]}]}}


def copied_c0(owned: list[str]) -> dict[str, Any]:
    return {"title": "Dấu hiệu của từng cấp độ", "selection_rationale": "Bảng ba cấp độ.",
            "covered_source_fact_ids": owned, "html": None, "semantic_content": {"sections": [
                {"heading": "Các cấp độ khiếu nại", "learning_block_ids": [], "blocks": [
                    block("paragraph", text="Mỗi khiếu nại thuộc một trong ba cấp độ. Cấp độ quyết định ai xử lý."),
                    block("table", rows=SHORT_ROWS),
                    *(block("paragraph", text=text) for text in COPIED)]}]}}


def writer_with(body: dict[str, Any], c0: dict[str, Any]) -> dict[str, Any]:
    writer = severity_writer(body)
    writer["components"]["c0"] = c0
    return writer


def owned_of(body: dict[str, Any]) -> list[str]:
    return list(body["unit_contract"]["component_plan"][0]["source_fact_ids"])


def html_texts(component: dict[str, Any]) -> list[str]:
    return [item["text"] for section in component["semantic_content"]["sections"] for item in section["blocks"]
            if item.get("text")]


class ExplainedUnitTests(StoryboardEndpointTestCase):
    async def test_explained_unit_follows_the_mode_and_lists_its_drafted_example(self) -> None:
        body = explain_body()
        writer = writer_with(body, explained_c0(owned_of(body)))
        provider = FakeGenerate(**{WRITER: [writer], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "validated", "provider")
        self.assertEqual(provider.names, [WRITER, JUDGE])
        prompt = provider.calls[0]["prompt"]
        self.assertIn("CONTENT MODE explain: teach, do not transcribe", prompt)
        self.assertIn('"examples":"ai_drafted"', prompt)
        # 450 words of a context_explain unit, raised by 35% (Node computes the same from the brief).
        self.assertIn("at most 607 words and at most 4249 visible characters", prompt)
        self.assertIn('"fidelity_mode":"explain"', provider.calls[1]["prompt"])
        # The drafted example was labelled by the server and is listed for the author.
        texts = html_texts(data["unit"]["components"][0])
        self.assertIn("Ví dụ minh hoạ: một khách hàng chỉ phản ánh việc chờ tổng đài hơi lâu và không có thiệt hại; "
                      "bạn tự xử lý ngay trong cuộc gọi.", texts)
        self.assertIn("Ví dụ minh hoạ do AI soạn — cần kiểm tra: 1 ví dụ (khối 1 (Lý thuyết)).", quality.author_note)
        self.assertEqual(quality.deterministic_codes, [])

    async def test_copied_unexplained_slot_gets_one_targeted_repair_then_review_never_the_source_copy(self) -> None:
        body = explain_body()
        writer = writer_with(body, copied_c0(owned_of(body)))
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [slot_repair(writer, 0)], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "review_required", "provider")
        self.assertEqual(provider.names, [WRITER, REPAIR, JUDGE])
        repair = provider.calls[1]
        self.assertEqual(repair["thinking_level"], "low")
        self.assertIn("c0 IDM_W5_SENTENCE_COPY at c0.semantic_content.sections[0].blocks[2], "
                      "c0.semantic_content.sections[0].blocks[3]", repair["prompt"])
        self.assertIn("(2 of 4 judged sentences repeat the facts word for word; at most 20% may)", repair["prompt"])
        self.assertIn("c0 IDM_W5_EXPLANATION_MISSING at c0", repair["prompt"])
        component = data["unit"]["components"][0]
        self.assertFalse(component.get("source_locked_fallback"))
        self.assertEqual(html_texts(component)[-2:], COPIED)
        self.assertEqual(set(quality.deterministic_codes), {"IDM_W5_EXPLANATION_MISSING", "IDM_W5_SENTENCE_COPY"})
        self.assertIn("Giữ bản AI để tác giả rà soát: khối 1 (Lý thuyết)", quality.author_note)
        self.assertIn("Cần xem khối 1 (Lý thuyết): nhiều câu gần như chép nguyên văn tài liệu", quality.author_note)
        self.assertNotIn("bản dự phòng", quality.author_note)
        self.assertNotIn("fallback", [event["outcome"] for event in data["attempt_trace"]])

    async def test_a_repair_that_explains_settles_the_slot(self) -> None:
        body = explain_body()
        writer = writer_with(body, copied_c0(owned_of(body)))
        fixed = explained_c0(owned_of(body))
        repair = slot_repair(writer, 0, **{key: value for key, value in fixed.items()
                                           if key != "covered_source_fact_ids"})
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [repair], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "validated", "provider")
        self.assertTrue(quality.repair_applied)
        self.assertIn("Đã tự sửa: IDM_W5_EXPLANATION_MISSING, IDM_W5_SENTENCE_COPY.", quality.author_note)

    async def test_invented_names_and_numbers_are_repaired_or_kept_for_review(self) -> None:
        body = explain_body()
        invented = explained_c0(owned_of(body))
        invented["semantic_content"]["sections"][1]["blocks"][0]["text"] = (
            "Ví dụ minh hoạ: chị Lan ở chi nhánh Hà Nội nhận 3 cuộc gọi phàn nàn trong 15 phút.")
        writer = writer_with(body, invented)
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [unavailable()], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        # The failed repair leaves the usage incomplete: Node files the unit under the reviewable lane; the slot is
        # still the provider draft.
        quality = self.assert_envelope(data, "structured_fallback", "review_required", "reserved_upper_bound")
        repair_prompt = provider.calls[1]["prompt"]
        self.assertIn("c0 IDM_W5_NEW_FACT at c0.semantic_content.sections[1].blocks[0]", repair_prompt)
        # Locations only: the rejected text never reaches the repair prompt's rule lines.
        self.assertNotIn("Hà Nội", repair_prompt.split("FAILING_RULES", 1)[1])
        self.assertFalse(data["unit"]["components"][0].get("source_locked_fallback"))
        self.assertIn("Cần xem khối 1 (Lý thuyết): có số liệu hoặc tên riêng không có trong tài liệu",
                      quality.author_note)

    async def test_a_repair_that_breaks_a_node_rule_is_dropped_for_the_reviewed_draft(self) -> None:
        # The repair explains the slot but writes a number right after "Hiện trạng", which Node reads as a page
        # locator. Without the revert the slot would fall back to the source-locked copy.
        body = explain_body()
        writer = writer_with(body, copied_c0(owned_of(body)))
        fixed = explained_c0(owned_of(body))
        fixed["semantic_content"]["sections"][1]["blocks"].append(
            block("paragraph", text="Hiện trạng 3 khiếu nại đang chờ xử lý trong ca."))
        repair = slot_repair(writer, 0, semantic_content=fixed["semantic_content"])
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [repair], JUDGE: [judge()]})
        with self.assertLogs("app.idm", "INFO") as logs:
            status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "review_required", "provider")
        component = data["unit"]["components"][0]
        self.assertFalse(component.get("source_locked_fallback"))
        self.assertNotIn("Hiện trạng", json.dumps(component, ensure_ascii=False))
        self.assertEqual(html_texts(component)[-2:], COPIED)
        self.assertFalse(quality.repair_applied)
        # Node's rule was seen on the repaired slot; the unit kept the writer's draft for review.
        self.assertTrue(any('"event": "idm_unit_quality"' in line and "HTML_SOURCE_LOCATOR@components[0]" in line
                            for line in logs.output))

    async def test_the_same_slot_in_keep_mode_has_no_copy_or_explanation_check(self) -> None:
        body = severity_body()
        self.assertEqual(body["unit_contract"]["idm_unit_brief"]["fidelity_mode"], "keep")
        writer = writer_with(body, copied_c0(owned_of(body)))
        provider = FakeGenerate(**{WRITER: [writer], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "validated", "provider")
        self.assertEqual(provider.names, [WRITER, JUDGE])
        self.assertIn("CONTENT MODE keep", provider.calls[0]["prompt"])
        self.assertIn("at most 450 words", provider.calls[0]["prompt"])
        self.assertEqual(quality.deterministic_codes, [])
