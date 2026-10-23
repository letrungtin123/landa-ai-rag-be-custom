"""Fixes from the QC report of the real AI ID run 2f8781d6 (AI 7f67137, 2026-10-23, 7,1/10), wave B (B6-B9).

Every case below is synthetic text with the shape of a defect of that run (no course text of the run); every provider
is a local fake.
"""

from __future__ import annotations

import unittest
from typing import Any

from app.idm.answer_near import ANSWER_ITEM_LENGTH_CODE, ANSWER_NEAR_EXAMPLE_CODE, item_length_cue, near_example
from app.idm.qa import STYLE_CODES
from app.idm.qa_checks import advisory_slot_findings
from app.idm.storyboard_context import _REVIEW_FIRST_CODES, _REVIEW_ONLY_CODES, _TARGETED_REPAIR_CODES
from app.idm.storyboard_repair import SECOND_REPAIR_CODES
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
    slot_repair,
)


def html(*sections: tuple[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {"type": "html", "semantic_content": {"version": 2, "sections": [
        {"heading": heading, "blocks": blocks} for heading, blocks in sections]}}


def paragraph(text: str) -> dict[str, Any]:
    return {"kind": "paragraph", "text": text, "items": [], "rows": []}


def table(*rows: tuple[str, str]) -> dict[str, Any]:
    return {"kind": "table", "text": None, "items": [], "rows": [{"label": label, "value": value}
                                                                  for label, value in rows]}


def quiz(question: str, options: list[str], key: int = 0) -> dict[str, Any]:
    return {"type": "problem", "problem_type": "multiple_choice", "question": question,
            "choices": [{"text": text, "correct": index == key} for index, text in enumerate(options)],
            "explanation": "Nguyên tắc: …"}


# A three-layer table as in the run's service blueprint unit (c2.l4.u2): each layer names its own example.
LAYERS = html(("Ba tầng của sơ đồ dịch vụ gồm những gì?", [table(
    ("Tuyến trước", "Nhân viên gặp khách hàng, ví dụ chuyên viên tư vấn gọi điện chào mừng khách mới."),
    ("Hậu trường", "Các phòng ban chuẩn bị dữ liệu nội bộ, ví dụ phòng kinh doanh và phòng kỹ thuật phối hợp."),
    ("Quy trình hỗ trợ", "Hệ thống và quy định nền tảng, ví dụ phần mềm quản lý khách hàng tự động cấp tài khoản."))]))
# The row of the key's layer, and a profile row as in c1.l2.u1.
FEELINGS = html(("Tầng cảm xúc nói gì?", [table(
    ("Nỗi đau cảm xúc", "Căng thẳng, hoang mang, bực bội vì mất kiểm soát hệ thống."),
    ("Nỗi đau chức năng", "Bị chậm, tốn sức, tốn tiền vì quy trình rườm rà."))]))


class NearExampleTests(unittest.TestCase):
    """B6: a key the words right above give away, short of a leak (calibrated on run 2f8781d6)."""

    def test_a_key_that_copies_the_row_of_its_answer(self) -> None:
        question = quiz("Giả sử bạn thiết lập sơ đồ dịch vụ cho buổi đón khách mới. Hoạt động nào thuộc Hậu trường?",
                        ["Phòng kinh doanh và phòng kỹ thuật phối hợp chuẩn bị dữ liệu",
                         "Phần mềm tự động cấp tài khoản theo quy trình một chạm",
                         "Chuyên viên gửi thư tư vấn cho khách mới",
                         "Khách quét mã xác nhận tham gia buổi gặp mặt"])
        self.assertEqual(near_example(question, [LAYERS]), "option")

    def test_a_listed_item_that_restates_the_row_of_the_key(self) -> None:
        question = quiz("Một giám đốc mua hàng chia sẻ bốn phát biểu:\n- Phát biểu 1: Phần mềm hiện tại xử lý quá "
                        "chậm.\n- Phát biểu 2: Tôi luôn căng thẳng và bực bội vì mất kiểm soát hệ thống.\n- Phát biểu "
                        "3: Doanh nghiệp có nguy cơ tụt hậu.\n- Phát biểu 4: Dự án làm tăng chi phí vận hành.\nPhát "
                        "biểu nào là nỗi đau cảm xúc?", ["Phát biểu 2", "Phát biểu 1", "Phát biểu 3", "Phát biểu 4"])
        self.assertEqual(near_example(question, [FEELINGS]), "item")

    def test_a_case_that_names_the_need_in_the_key_s_words(self) -> None:
        question = quiz("Khách hàng cần kiểm soát tiến độ tự động cho nhân viên làm việc phân tán.\nĐề xuất nào phù "
                        "hợp nhất?", ["Phần mềm học trên di động và báo cáo tiến độ tự động",
                                      "Kho tài liệu chuyên sâu kèm tổng đài hỗ trợ",
                                      "Hội thảo đào tạo trực tiếp tại văn phòng",
                                      "Trợ lý giải đáp thắc mắc và kho lưu trữ"])
        self.assertEqual(near_example(question, []), "stem")

    def test_a_case_that_copies_the_example_holding_the_key(self) -> None:
        example = html(("Ví dụ", [paragraph(
            "Ban điều hành của một công ty giải pháp đào tạo nhận thấy khách hàng ký hợp đồng vì bản demo thực "
            "chiến.")]))
        question = quiz("Ban điều hành của một công ty giải pháp phần mềm đang đánh giá bốn đề xuất.\nĐề xuất nào tác "
                        "động lớn nhất đến quyết định mua?", ["Rút ngắn thời gian gửi bản demo cho khách",
                                                              "In hồ sơ năng lực cao cấp hơn",
                                                              "Tặng quà tri ân kèm hợp đồng",
                                                              "Thiết kế lại phiếu khảo sát cuối dự án"])
        self.assertEqual(near_example(question, [example]), "case")

    def test_good_questions_stay_quiet(self) -> None:
        # Every option applies a criterion in its own words (c1.l1.u2 of the run).
        question = quiz("Bạn chào bán giải pháp cho một doanh nghiệp sản xuất.\nAi giữ vai trò người quyết định?",
                        ["Tổng giám đốc ký duyệt ngân sách dự án", "Thư ký sắp xếp lịch hẹn ban đầu",
                         "Trưởng ca nhập liệu hằng ngày", "Chuyên gia độc lập góp ý chuyên môn"])
        self.assertIsNone(near_example(question, [LAYERS]))
        # A distractor that copies the html (c1.l2.u2 of the run) is a good distractor, never a give-away.
        decoy = quiz("Nhận định nào xếp đúng tầng?", ["Muốn giảm thời gian xử lý thủ công là việc chức năng",
                                                      "Căng thẳng, hoang mang, bực bội vì mất kiểm soát hệ thống là "
                                                      "nỗi đau chức năng", "Được công nhận là nỗi đau cảm xúc",
                                                      "Tự hào là việc xã hội"])
        self.assertIsNone(near_example(decoy, [FEELINGS]))
        # A key that opens with a taught label applies its rule by design.
        labelled = quiz("Khách phàn nàn lần thứ hai. Cấp nào?", ["Hậu trường vì các phòng ban chuẩn bị dữ liệu nội bộ",
                                                                  "Tuyến trước vì gặp khách",
                                                                  "Quy trình hỗ trợ vì có phần mềm"])
        self.assertIsNone(near_example(labelled, [LAYERS]))

    def test_the_key_s_item_clearly_the_longest(self) -> None:
        items = ["Chia nhóm phủ kín thị trường nhưng có khách thuộc hai nhóm.",
                 "Bỏ trùng lặp nhưng bỏ sót nhóm mới nổi.",
                 "Các nhóm độc lập, phủ toàn bộ thị trường và chọn nhóm theo khung 2-S để quyết định.",
                 "Chỉ phân tích nhân khẩu học để chọn hay bỏ nhóm."]
        stem = "Bạn xem bốn đề xuất:\n" + "\n".join(f"- Đề xuất {n}: {t}" for n, t in enumerate(items, 1))
        clear = quiz(stem + "\nĐề xuất nào đúng?", [f"Đề xuất {n}" for n in (3, 1, 2, 4)])
        self.assertEqual(item_length_cue(clear), (len(items[2]), len(items[0])))
        self.assertEqual(advisory_slot_findings({"components": [clear]})[0].code, ANSWER_ITEM_LENGTH_CODE)
        balanced = quiz(stem + "\nĐề xuất nào đúng?", [f"Đề xuất {n}" for n in (1, 2, 3, 4)])
        self.assertIsNone(item_length_cue(balanced))

    def test_both_are_soft(self) -> None:
        # Repaired once at low thinking, then an edit note; never a fallback, never a second repair.
        for code in (ANSWER_NEAR_EXAMPLE_CODE, ANSWER_ITEM_LENGTH_CODE):
            self.assertIn(code, _TARGETED_REPAIR_CODES)
            self.assertIn(code, STYLE_CODES)
            self.assertNotIn(code, SECOND_REPAIR_CODES)
        self.assertIn(ANSWER_NEAR_EXAMPLE_CODE, _REVIEW_FIRST_CODES)
        self.assertIn(ANSWER_NEAR_EXAMPLE_CODE, _REVIEW_ONLY_CODES)


class NearExampleEndpointTests(StoryboardEndpointTestCase):
    """B6 end to end: one targeted repair; a key the repair leaves near the case stays the AI's, with an edit note."""

    async def test_repaired_once_then_an_edit_note(self) -> None:
        body = escalate_body()
        writer = escalate_writer(body)
        writer["components"]["c0"]["question"] = (
            "Một khách hàng VIP gọi đến vì sản phẩm gây chập điện, khiếu nại liên quan đến an toàn. Bạn nên làm gì "
            "trước tiên?")
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [slot_repair(writer, 0)], JUDGE: [judge(), judge()]})
        with self.assertLogs("app.idm", level="INFO") as logs:
            status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200, data.get("detail"))
        self.assertEqual(provider.names.count(REPAIR), 1)
        self.assertEqual((data["content_origin"], data["quality_state"]), ("provider_validated", "validated"))
        self.assertNotIn("source_locked_fallback", data["unit"]["components"][0])
        quality = logged_events(logs.records, "idm_unit_quality")[-1]
        self.assertIn(ANSWER_NEAR_EXAMPLE_CODE, quality["remaining_codes"])
        note = data["unit"]["idm_quality"]["author_note"]
        self.assertIn("Nên biên tập khối 1 (Quiz): đáp án đúng (hoặc ý nó chỉ tới) lặp lại câu chữ", note)
        self.assertNotIn("Cần xem", note)
        repair_prompt = next(call["prompt"] for call in provider.calls if call["family"] == REPAIR)
        self.assertIn("IDM_W5_ANSWER_NEAR_EXAMPLE", repair_prompt)
        self.assertIn("change the case details and the wording", repair_prompt)


if __name__ == "__main__":
    unittest.main()
