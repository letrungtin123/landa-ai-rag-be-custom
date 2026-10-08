"""QC course 364564 (2026-10-08), package A regressions through the real ``/unit`` route (N2, N3, N8).

* N2: the correct option was the longest in 8 of 9 questions (a length cue, review note only).
* N3: an option copied the worked example of the worksheet shown just before the question.
* N8: codes found before a successful repair were printed as "còn cảnh báo" on a unit that passed.

The provider is the local ``FakeGenerate``; nothing reaches the network.
"""

from __future__ import annotations

from typing import Any

from tests.test_idm_storyboard import (
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
