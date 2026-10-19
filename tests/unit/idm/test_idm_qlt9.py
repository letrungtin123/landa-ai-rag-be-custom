"""QLT-9: the fixes found by the QC of run ffaf7318 (8.57/10) and the owner's rule for answer keys.

1. Owner 2026-10-16: the correct options of a course's questions are shuffled, never all B or C and never the A B C D
   cycle the run had: blocks of four seeded permutations, no letter twice in a row; W4 gives every single-choice
   question its letter, knowledge checks included (run c0030787), and the letter reaches the unit in its brief.
2. A key that rewrites the worked example's rows in other words keeps their words (c3.l2.u2); a key that applies one
   table row's criterion is no joined list, whatever words every row opens with (c3.l1.u1).
3. An over-long worksheet keeps its worked example as long as it can (3 of 5 lost it).
4. "Cần xem" marks what needs a decision; wording findings and source pages with images are notices; notes speak in
   plain words, never codes; an FAQ title may name the unit's topic in the unit's words.
5. A framework the course purpose names is presented whole; each unit opens its drafted example its own way; the case
   never names the cause the key fixes; an FAQ answer never classifies against the unit's question.

The provider is the local fake; nothing reaches the network.
"""

from __future__ import annotations

import collections
import itertools
import unittest
from typing import Any

from app.idm.answer_checks import restates_joined_example, restates_list, unit_teaching
from app.idm.contracts import IdmUnitBriefV1
from app.idm.faq_checks import faq_writer_rules
from app.idm.mcq import assign_key_letters, key_letters, normalize_single_choice
from app.idm.module_design import attach_purpose_framework
from app.idm.prompts import example_opening, unit_writer_prompt
from app.idm.qa import (
    REVIEW_REASON_EVIDENCE,
    REVIEW_REASON_STYLE,
    JudgeOutcome,
    SlotFinding,
    build_unit_author_note,
    faq_title_mismatch,
)
from app.idm.worksheet import compact_worksheet
from tests.support.idm_test_support import golden_shard
from tests.unit.idm.test_idm_module_design import scope_of
from tests.unit.idm.test_idm_storyboard import severity_body, severity_writer


def _html(*sections: tuple[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {"type": "html", "semantic_content": {"version": 1, "sections": [
        {"heading": heading, "blocks": [{"text": None, "items": [], "rows": [], **block} for block in blocks]}
        for heading, blocks in sections]}}


def _question(key: str, *distractors: str) -> dict[str, Any]:
    return {"type": "problem", "question": "Phương án nào thể hiện đúng tư duy kết nối mọi thứ?",
            "choices": [{"text": key, "correct": True}, *({"text": text, "correct": False} for text in distractors)]}


class KeyLetterTests(unittest.TestCase):
    def test_keys_are_shuffled_balanced_and_never_twice_in_a_row(self) -> None:
        worst = 0
        for seed in range(500):
            letters = key_letters(f"course-{seed}", 0, 16)
            self.assertEqual(letters, key_letters(f"course-{seed}", 0, 16))
            for block in range(4):
                self.assertEqual(sorted(letters[4 * block:4 * block + 4]), [0, 1, 2, 3])
            self.assertFalse(any(first == second for first, second in itertools.pairwise(letters)))
            worst = max(worst, *collections.Counter(key_letters(f"course-{seed}", 0, 10)).values())
        self.assertLessEqual(worst, 3)
        # Not the old cycle, and not one order for every course.
        orders = {tuple(key_letters(f"course-{seed}", 0, 8)) for seed in range(50)}
        self.assertGreater(len(orders), 20)
        self.assertNotIn((0, 1, 2, 3, 0, 1, 2, 3), orders)
        self.assertEqual(key_letters("abc", 3, 5), key_letters("abc", 0, 8)[3:])

    def test_w4_gives_every_question_of_the_shard_its_letter(self) -> None:
        # Run c0030787: 6 of 17 questions were knowledge checks without a practice id, so they kept the old cycle.
        lessons = golden_shard(0).lessons
        # A knowledge check without a practice, as in run c0030787 (the golden questions all have one).
        check = lessons[0].units[0].components[0].model_copy(update={"type": "problem", "practice_id": None})
        lessons = [lessons[0].model_copy(update={"units": [
            lessons[0].units[0].model_copy(update={"components": [check, *lessons[0].units[0].components[1:]]}),
            *lessons[0].units[1:]]}), *lessons[1:]]
        keyed, count = assign_key_letters(lessons, "snapshot-1", 2)
        components = [component for lesson in keyed for unit in lesson.units for component in unit.components]
        problems = [component for component in components if component.type == "problem"]
        self.assertEqual(count, len(problems))
        self.assertIsNone(problems[0].practice_id)
        self.assertEqual([component.key_letter for component in problems], key_letters("snapshot-1", 2, count))
        self.assertTrue(all(component.key_letter is None for component in components if component.type != "problem"))
        # The letter travels with the component: in its JSON form and in the design hash, absent where there is none.
        dumped = keyed[0].model_dump(mode="json")
        slots = [component for unit in dumped["units"] for component in unit["components"]]
        self.assertTrue(all(("key_letter" in slot) == (slot["type"] == "problem") for slot in slots))

    def test_a_brief_with_the_letter_serves_the_key_there(self) -> None:
        body = severity_body()
        brief = body["unit_contract"]["idm_unit_brief"]
        problem = next(index for index, slot in enumerate(brief["components"]) if slot["type"] == "problem")
        brief["components"][problem]["key_letter"] = 3
        parsed = IdmUnitBriefV1.model_validate(brief)
        self.assertEqual(parsed.components[problem].key_letter, 3)
        self.assertEqual(parsed.components[0].model_dump(mode="json").get("key_letter", "absent"), "absent")
        writer = severity_writer(body)
        served = normalize_single_choice(writer["components"][f"c{problem}"],
                                         brief["components"][problem]["component_plan_id"], 3)[0]
        self.assertTrue(served["choices"][3 % len(served["choices"])]["correct"])


class JoinedExampleTests(unittest.TestCase):
    # c3.l2.u2 of run ffaf7318, as written: the key rewrites three rows of the worksheet example in other words.
    EXAMPLE = _html(("Ví dụ tham khảo thực hiện biểu mẫu", [
        {"kind": "paragraph",
         "text": "Ví dụ minh hoạ: Một CEO doanh nghiệp thời trang quy mô vừa xác định các hạng mục như sau:"},
        {"kind": "table", "rows": [
            {"label": "Năng lực lõi giữ lại nội bộ",
             "value": "Năng lực sáng tạo mẫu mã độc quyền và định vị thương hiệu."},
            {"label": "Liên minh 1: Nền tảng công nghệ",
             "value": "Thuê nền tảng Cloud và AI có sẵn để tự động quản lý tồn kho đa kênh."},
            {"label": "Liên minh 2: Nguồn lực chuyên gia",
             "value": "Thuê chuyên gia cố vấn chiến lược chuỗi cung ứng theo hợp đồng tư vấn 90 ngày."},
            {"label": "Liên minh 3: Kênh phân phối thị trường",
             "value": "Hợp tác với mạng lưới đại lý bán lẻ đối tác để phủ thị trường nhanh chóng."}]}]))
    KEY = ("Doanh nghiệp giữ vững năng lực lõi nghiên cứu sản xuất, đồng thời liên minh với nền tảng Cloud có sẵn để "
           "quản lý bán hàng và hợp tác với mạng lưới phân phối đối tác.")
    DISTRACTORS = (
        "Doanh nghiệp thuê trọn gói đơn vị bên ngoài làm thay toàn bộ khâu sáng tạo mẫu mã và nghiên cứu lõi của sản "
        "phẩm, dồn toàn lực tự xây hệ sinh thái bán lẻ.",
        "Doanh nghiệp thế chấp tài sản để tự đầu tư toàn bộ chuỗi cửa hàng bán lẻ độc quyền, nhằm tránh việc phải phụ "
        "thuộc vào các đơn vị phân phối trung gian.",
        "Doanh nghiệp tuyển thêm 10 lập trình viên giỏi vào biên chế để tự viết phần mềm quản trị riêng, đảm bảo kiểm "
        "soát tuyệt đối mã nguồn và quy trình dữ liệu nội bộ.")

    def test_rows_rewritten_in_other_words_still_give_the_answer_away(self) -> None:
        self.assertTrue(restates_joined_example(_question(self.KEY, *self.DISTRACTORS), unit_teaching([self.EXAMPLE],
                                                                                                       [])))

    def test_applying_one_row_of_a_framed_table_is_no_joined_list(self) -> None:
        # c3.l1.u1 of run ffaf7318: every row opens with "Yêu cầu:" and "Nguy cơ cần chặn:".
        filter_table = _html(("Ba trụ cột", [{"kind": "table", "rows": [
            {"label": "Hiện đại hóa", "value": "Yêu cầu: Triển khai chuẩn quản trị quốc tế, quy trình số và tự động "
                                               "hóa. Nguy cơ cần chặn: Sự lạc hậu."},
            {"label": "Việt Nam hóa", "value": "Yêu cầu: Điều chỉnh chuẩn mực quốc tế cho tương thích với văn hóa ứng "
                                               "xử và nguồn lực. Nguy cơ cần chặn: Áp đặt rập khuôn."},
            {"label": "Lành mạnh hóa", "value": "Yêu cầu: Tăng trưởng có kỷ luật, tài chính minh bạch. Nguy cơ cần "
                                                "chặn: Kinh doanh chụp giật."}]}]))
        key = ("Yêu cầu điều chỉnh quy trình và cơ chế đánh giá để tương thích với văn hóa và nguồn lực của đội ngũ "
               "trước khi triển khai.")
        question = _question(key, "Phê duyệt áp dụng nguyên bản phần mềm để theo chuẩn quốc tế ngay lập tức.",
                             "Hủy bỏ toàn bộ kế hoạch và duy trì phương thức đánh giá theo kinh nghiệm.",
                             "Chỉ phê duyệt khi nhà cung cấp giảm chi phí bản quyền và nới lỏng yêu cầu minh bạch.")
        self.assertFalse(restates_list(question, unit_teaching([filter_table], [])))


class WorksheetTests(unittest.TestCase):
    def worksheet(self, guidance: str) -> dict[str, Any]:
        return _html(
            ("Nhiệm vụ của bạn", [{"kind": "task", "text": "Lập bản đồ 03 liên minh nguồn lực trong 90 ngày."}]),
            ("Mẫu cần điền", [{"kind": "table", "rows": [
                {"label": f"Liên minh {index}", "value": guidance} for index in range(1, 4)]}]),
            ("Ví dụ tham khảo", [{"kind": "table", "rows": [
                {"label": f"Liên minh {index}", "value": "Thuê nền tảng có sẵn trong 90 ngày."}
                for index in range(1, 4)]}]),
            ("Tự kiểm tra", [{"kind": "bullets", "items": ["Có đủ 03 liên minh.", "Có thời hạn 90 ngày."]}]))

    def test_the_worked_example_stays_when_shorter_guidance_is_enough(self) -> None:
        component = self.worksheet("Ai là đối tác? Vì sao chọn họ thay vì tự làm? Kết quả đo bằng gì?")
        compacted = compact_worksheet(component, lambda value: "Vì sao" not in str(value))
        assert compacted is not None
        value, keeps_example = compacted
        self.assertTrue(keeps_example)
        headings = [section["heading"] for section in value["semantic_content"]["sections"]]
        self.assertIn("Ví dụ tham khảo", headings)

    def test_the_example_is_cut_to_two_entries_then_left_out_last(self) -> None:
        component = self.worksheet("Ai là đối tác?")
        two = compact_worksheet(component, lambda value: str(value).count("Thuê nền tảng") <= 2)
        assert two is not None
        self.assertTrue(two[1])
        self.assertEqual(str(two[0]).count("Thuê nền tảng"), 2)
        dropped = compact_worksheet(component, lambda value: "Thuê nền tảng" not in str(value))
        assert dropped is not None
        self.assertFalse(dropped[1])

    def test_the_note_says_whether_the_example_stayed(self) -> None:
        kept = build_unit_author_note(locale="vi", judge=JudgeOutcome("pass"), deterministic_codes=[],
                                      fallback_slots=[], ai_drafted=False, worksheet_compacted=True,
                                      worksheet_example_kept=True)
        self.assertIn("đã rút gọn gợi ý trong bảng (ví dụ mẫu vẫn giữ", kept)
        self.assertNotIn("bỏ phần ví dụ mẫu", kept)


class NoteTests(unittest.TestCase):
    def test_notices_are_not_a_review_mark_and_notes_have_no_codes(self) -> None:
        note = build_unit_author_note(
            locale="vi", judge=JudgeOutcome("pass"), deterministic_codes=["IDM_W5_SENTENCE_LENGTH"],
            fallback_slots=[], ai_drafted=False, review_slots=[0], slot_types=["html"],
            slot_reasons={0: ["IDM_W5_SENTENCE_LENGTH"]}, remaining=[SlotFinding("IDM_W5_SENTENCE_LENGTH", 0)],
            notices=[REVIEW_REASON_EVIDENCE, REVIEW_REASON_STYLE])
        self.assertTrue(note.startswith("Lưu ý: tài liệu nguồn của phần này có hình ảnh hoặc bảng"))
        self.assertIn("QA tự động: không phát hiện vấn đề lớn.", note)
        self.assertIn("Giữ bản AI để tác giả rà soát: khối 1 (Lý thuyết): câu quá dài.", note)
        self.assertIn("Nên biên tập khối 1 (Lý thuyết): nhiều câu quá dài", note)
        self.assertNotIn("IDM_", note)
        self.assertNotIn("Cần tác giả xem lại", note)

    def test_a_marked_unit_never_says_no_major_issue(self) -> None:
        note = build_unit_author_note(locale="vi", judge=JudgeOutcome("pass"), deterministic_codes=[],
                                      fallback_slots=[], ai_drafted=True, review_reasons=["ai_scenario"])
        self.assertNotIn("không phát hiện vấn đề lớn", note)
        self.assertTrue(note.startswith("Cần tác giả xem lại vì:"))

    def test_an_faq_title_may_name_the_units_topic_in_the_units_words(self) -> None:
        # c2.l2.u1 of run ffaf7318: the questions say "đối thủ trong nước", the unit teaches "đối sánh nội địa".
        faq = {"type": "la_faq", "title": "Giải đáp về đối sánh nội địa và cách học hỏi chuẩn Best-in-Class",
               "items": [{"question": "Vì sao bám đuổi giá của đối thủ trong nước là sai lầm?",
                          "answer": "Vì doanh nghiệp chỉ cạnh tranh bằng giá, không tạo được khác biệt."},
                         {"question": "Làm thế nào kết hợp chuẩn mực quốc tế với giải pháp bản địa?",
                          "answer": "Học cách làm của người dẫn đầu rồi điều chỉnh cho nguồn lực của mình."}]}
        self.assertTrue(faq_title_mismatch(faq))
        self.assertFalse(faq_title_mismatch(faq, "Đối sánh nội địa khiến doanh nghiệp học theo chuẩn Best-in-Class "
                                                 "của thế giới thay vì đối thủ bên cạnh."))


class DesignTests(unittest.TestCase):
    def test_a_framework_the_purpose_names_is_listed_in_the_modules_first_teaching_html(self) -> None:
        lessons = golden_shard(0).lessons
        attached, done = attach_purpose_framework(lessons, scope_of(0), ["Học viên phân biệt 3 cấp độ khiếu nại."])
        self.assertTrue(done)
        briefs = [item.brief for lesson in attached for unit in lesson.units for component in unit.components
                  for item in component.support_items if item.brief.startswith("Liệt kê đủ 3 cấp độ")]
        self.assertEqual(len(briefs), 1)
        self.assertIn("Cấp 1", briefs[0])
        # Already presented: nothing more; a framework the module does not number: nothing.
        self.assertFalse(attach_purpose_framework(attached, scope_of(0), ["3 cấp độ khiếu nại"])[1])
        self.assertFalse(attach_purpose_framework(lessons, scope_of(0), ["7 thói quen hiệu quả"])[1])

    def test_each_unit_opens_its_drafted_example_its_own_way(self) -> None:
        paths = [f"chapter_{chapter}.lesson_{lesson}.unit_1" for chapter in range(1, 5) for lesson in range(1, 5)]
        openings = [example_opening(path, "vi") for path in paths]
        self.assertEqual(openings, [example_opening(path, "vi") for path in paths])
        self.assertGreaterEqual(len(set(openings)), 4)
        self.assertLessEqual(openings.count("Một CEO …"), len(openings) // 3)
        brief = IdmUnitBriefV1.model_validate(severity_body()["unit_contract"]["idm_unit_brief"])
        prompt = unit_writer_prompt("vi", course_title="C", audience=brief.target_audience, lesson_title="L",
                                    lesson_objective="O", practice_sentences=[], previous_title=None, next_title=None,
                                    unit_brief={}, facts=[], context_facts=[], example_opening="Hãy hình dung …")
        flat = " ".join(prompt.split())
        self.assertIn('in this unit the first drafted example or case opens with "Hãy hình dung …"', flat)
        self.assertIn("the case never names the cause, rule or criterion the key is about", flat)
        self.assertIn("never states a classification rule the facts do not", faq_writer_rules("explain"))


if __name__ == "__main__":
    unittest.main()
