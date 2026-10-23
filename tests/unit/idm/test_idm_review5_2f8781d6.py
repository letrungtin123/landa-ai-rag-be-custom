"""Fixes from the independent review 5 of the run 2f8781d6 changes (7f67137..345663c).

Every case is synthetic text with the shape of the reviewer's reproduction scripts; every provider is a local fake.
"""

from __future__ import annotations

import json
from typing import Any

from tests.support.log_support import logged_events
from tests.unit.idm.test_idm_storyboard import (
    JUDGE,
    REPAIR,
    WRITER,
    FakeGenerate,
    StoryboardEndpointTestCase,
    escalate_body,
    escalate_writer,
    judge,
    severity_body,
    severity_writer,
    unavailable,
)

# H1: a quiz whose key the html repair below happens to restate ("hẹn gọi lại cho khách").
KEY = "Ghi nhận phản hồi và hẹn gọi lại cho khách trong ngày"
QUIZ = {
    "question": "Giả sử một khách hàng gọi lần thứ hai để phàn nàn vì đơn hàng giao trễ, chưa có thiệt hại tài "
                "chính. Bạn nên làm gì trước tiên?",
    "choices": [{"text": "Chuyển hồ sơ sang bộ phận Pháp chế ngay lập tức", "correct": False},
                {"text": KEY, "correct": True},
                {"text": "Đề nghị hoàn tiền toàn bộ giá trị của đơn hàng", "correct": False}],
    "explanation": "Nguyên tắc: khiếu nại cấp 2 cần ghi nhận và thông báo trưởng nhóm.\nA - Sai vì không có yếu tố "
                   "pháp lý.\nB - Đúng vì khách phàn nàn lần thứ hai là cấp 2.\nC - Sai vì chưa có thiệt hại."}
REPEATED = "Mỗi khiếu nại thuộc một trong ba cấp độ. Cấp độ quyết định ai xử lý."
FIXED = "Với khiếu nại cấp 2, nhân viên nên hẹn gọi lại cho khách và báo trưởng nhóm."


def writer_with(body: dict[str, Any], paragraphs: list[str]) -> dict[str, Any]:
    writer = severity_writer(body)
    blocks = writer["components"]["c0"]["semantic_content"]["sections"][0]["blocks"]
    blocks[:] = [{"kind": "paragraph", "text": text, "items": [], "rows": []} for text in paragraphs] + [blocks[1]]
    writer["components"]["c1"].update(QUIZ)
    return writer


class SoftFindingRepairTests(StoryboardEndpointTestCase):
    """H1 and M6: a soft finding never makes a repair worse, never vetoes it, and a failed call that only repairs soft
    findings leaves the unit as it was."""

    async def test_an_html_repair_is_kept_when_the_unchanged_quiz_turns_soft(self) -> None:
        body = severity_body()
        writer = writer_with(body, [REPEATED, REPEATED])  # the same paragraph twice: HTML_DUPLICATE_BLOCK
        fixed = writer_with(body, [REPEATED, FIXED])
        repair = {"components": {"c0": {key: value for key, value in fixed["components"]["c0"].items()
                                        if key != "covered_source_fact_ids"}}}
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [repair], JUDGE: [judge(), judge()]})
        with self.assertLogs("app.idm", level="INFO") as logs:
            status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200, data.get("detail"))
        # Old code (review 5): the guard dropped the whole repair and the html slot fell back.
        self.assertEqual((data["content_origin"], data["quality_state"]), ("provider_validated", "validated"))
        self.assertEqual(logged_events(logs.records, "idm_unit_repair_guard"), [])
        quality = logged_events(logs.records, "idm_unit_quality")[-1]
        self.assertEqual(quality["fallback_slots"], [])
        html = json.dumps(data["unit"]["components"][0], ensure_ascii=False)
        self.assertIn("hẹn gọi lại cho khách và báo trưởng nhóm", html)
        self.assertNotIn("source_locked_fallback", data["unit"]["components"][0])
        self.assertIn("Đã tự sửa: phần lý thuyết lặp lại cùng một đoạn", data["unit"]["idm_quality"]["author_note"])

    async def test_a_failed_soft_repair_leaves_the_unit_as_it_was(self) -> None:
        body = escalate_body()
        writer = escalate_writer(body)
        writer["components"]["c0"]["question"] = (
            "Một khách hàng VIP gọi đến vì sản phẩm gây chập điện, khiếu nại liên quan đến an toàn. Bạn nên làm gì "
            "trước tiên?")
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [unavailable(), unavailable()], JUDGE: [judge()]})
        with self.assertLogs("app.idm", level="INFO") as logs:
            status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200, data.get("detail"))
        # Old code: structured_fallback / review_required / reserved_upper_bound, and the judge never ran.
        self.assertEqual((data["content_origin"], data["quality_state"], data["usage_source"]),
                         ("provider_validated", "validated", "provider"))
        self.assertEqual(provider.names.count(JUDGE), 1)
        self.assertNotIn("source_locked_fallback", data["unit"]["components"][0])
        quality = logged_events(logs.records, "idm_unit_quality")[-1]
        self.assertEqual(quality["failure_codes"], [])
        note = data["unit"]["idm_quality"]["author_note"]
        self.assertIn("Nên biên tập khối 1 (Quiz): đáp án đúng", note)
        self.assertNotIn("Sửa tự động không thành công", note)
        self.assertNotIn("Cần tác giả xem lại", note)
        self.assertNotIn("không chạy được", note)

    async def test_a_failed_hard_repair_still_marks_the_unit(self) -> None:
        # Unchanged for any other finding: the failure is the unit's (here a leak, repaired at targeted level).
        body = escalate_body()
        writer = escalate_writer(body)
        writer["components"]["c0"]["choices"][0]["text"] = "Có"
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [unavailable(), unavailable()], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200, data.get("detail"))
        self.assertEqual(data["usage_source"], "reserved_upper_bound")
