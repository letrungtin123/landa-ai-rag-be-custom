"""QLT-8: the fixes found by the QC of run 0a11aa9e (8.7/10), tested with the run's own examples.

1. A question cut before it asks anything is found (IDM_W5_QUESTION_INCOMPLETE), repaired once, and a repair that cuts
   a whole question gives the slot its earlier version back (c1.l2.u4 reached the course without its question).
2. Keys that string several shown items together give the answer away: the five list heads of c1.l1.u1, the three
   worked-example rows of c2.l5.u1.
3. An FAQ floor: the W4 designer placed an FAQ in 3 of 23 units, so the server adds one to the units with the most to
   clarify until a quarter of a module's learning units have one.
4. The author note says why a unit needs review, and lists the abbreviations nobody explains; the ones the facts or the
   server's short list explain get their full form at first use. SME questions say "dòng N", not "Row N".
5. Writer and judge rules: the question ends with its question sentence, the worked example meets its own self-check,
   no classroom instructions, FAQ answers of about 60 words.

The provider is the local fake; nothing reaches the network.
"""

from __future__ import annotations

import math
import unittest
from typing import Any

from app.idm.acronyms import AcronymGuide, abbreviations, explained, learner_texts, with_expansion
from app.idm.answer_checks import restates_joined_example, restates_list, unit_teaching
from app.idm.artifacts import plain_row_refs
from app.idm.contracts import IdmUnitBriefV1
from app.idm.faq_checks import faq_writer_rules
from app.idm.faq_plan import FAQ_ADDED_CODE
from app.idm.mcq import QUESTION_INCOMPLETE_CODE, question_complete
from app.idm.policy import IDM_FAQ_MIN_UNIT_SHARE
from app.idm.prompts import judge_prompt, unit_writer_prompt
from app.idm.qa import (
    REVIEW_REASON_EVIDENCE,
    REVIEW_REASON_SCENARIO,
    JudgeOutcome,
    build_unit_author_note,
    deterministic_slot_findings,
)
from tests.support.idm_test_support import golden_shard
from tests.unit.idm.test_idm_fidelity_unit import html_texts
from tests.unit.idm.test_idm_module_design import design_module, module_response
from tests.unit.idm.test_idm_storyboard import (
    JUDGE,
    REPAIR,
    WRITER,
    FakeGenerate,
    StoryboardEndpointTestCase,
    judge,
    severity_body,
    severity_writer,
    slot_repair,
)

# c1.l2.u4 of run 0a11aa9e, as the course received it.
CUT_STEM = ("Một CEO điều hành công ty sản xuất đang duy trì đội xe vận tải riêng và kho bãi trung chuyển dù công suất "
            "khai thác thực tế chỉ đạt 40%. Sau khi đối chiếu hiện trạng trên với các tiêu chí của 5 trục chuyển dịch")


def _html(*blocks: dict[str, Any]) -> dict[str, Any]:
    return {"type": "html", "semantic_content": {"sections": [{"heading": "Nội dung", "blocks": [
        {"text": None, "items": [], "rows": [], **block} for block in blocks]}]}}


def _question(question: str, key: str, *distractors: str) -> dict[str, Any]:
    return {"type": "problem", "question": question,
            "choices": [{"text": key, "correct": True}, *({"text": text, "correct": False} for text in distractors)]}


# c1.l1.u1 of run 0a11aa9e: the key strings the five heads of the list together.
ATTRIBUTES = _html({"kind": "bullets", "items": [
    "Chất lượng đỉnh cao: sản phẩm đáp ứng hoàn hảo yêu cầu kỹ thuật và độ bền.",
    "Uy tín tuyệt đối: cam kết nhất quán về tiến độ và độ tin cậy khi bàn giao.",
    "Giá trị gia tăng lớn: tối ưu biên lợi nhuận thông qua hàm lượng chất xám cao.",
    "Đổi mới liên tục: không ngừng cải tiến tính năng nhằm giải quyết nhu cầu thị trường.",
    "Chuẩn mực toàn cầu: tuân thủ nghiêm ngặt các hệ thống tiêu chuẩn quốc tế."]})
ATTRIBUTES_KEY = ("Xây dựng sản phẩm đồng nghĩa với chất lượng đỉnh cao, uy tín tuyệt đối, giá trị gia tăng lớn, "
                  "đổi mới liên tục và chuẩn mực toàn cầu.")
ATTRIBUTES_DISTRACTORS = (
    "Ưu tiên tối ưu hóa chi phí ngắn hạn bằng việc hạ thấp các tiêu chuẩn kỹ thuật không bắt buộc của đơn hàng.",
    "Chấp nhận cung cấp sản phẩm chất lượng loại hai trên thị trường quốc tế để duy trì mức giá cạnh tranh tối đa.",
    "Tập trung khai thác triệt để lợi thế nhân công giá rẻ và nhận các đơn hàng gia công thô ráp để duy trì sản lượng.")
# c2.l5.u1 of run 0a11aa9e: the key joins three filled rows of the worksheet's worked example.
ALLIANCE = _html(
    {"kind": "paragraph", "text": "Ví dụ minh hoạ: Một CEO áp dụng tư duy hệ sinh thái để tái cấu trúc nguồn lực:"},
    {"kind": "table", "rows": [
        {"label": "Năng lực lõi nội bộ", "value": "Nghiên cứu công thức sản phẩm đặc thù và trải nghiệm khách hàng."},
        {"label": "Nguồn lực 1: Chuyên gia cố vấn",
         "value": "Ký hợp đồng cố vấn chiến lược với chuyên gia đầu ngành trong 90 ngày."},
        {"label": "Nguồn lực 2: Nền tảng AI/Cloud",
         "value": "Thuê nền tảng AI/Cloud có sẵn để tự động hóa xử lý dữ liệu đơn hàng."},
        {"label": "Nguồn lực 3: Đối tác phân phối",
         "value": "Liên kết với các hệ thống phân phối lớn để phủ thị trường nhanh chóng."}]})
ALLIANCE_KEY = ("Giữ lại công thức lõi của sản phẩm, đồng thời ký hợp đồng cố vấn chuyên gia bên ngoài, thuê nền tảng "
                "AI/Cloud sẵn có và bắt tay với đối tác phân phối.")
ALLIANCE_DISTRACTORS = (
    "Tuyển dụng toàn bộ chuyên gia giỏi vào biên chế dài hạn, đặt mua máy chủ riêng và tự viết phần mềm quản trị.",
    "Chuyển giao toàn bộ năng lực lõi và linh hồn sản phẩm cho đơn vị gia công, chỉ giữ lại bộ phận kế toán nội bộ.",
    "Dừng toàn bộ các dự án liên kết bên ngoài để dành ngân sách mua sắm trang thiết bị phục vụ tự chủ nghiên cứu.")


class QuestionCompletenessTests(unittest.TestCase):
    def test_a_case_without_its_question_is_incomplete(self) -> None:
        self.assertFalse(question_complete(CUT_STEM))
        self.assertFalse(question_complete(""))
        self.assertFalse(question_complete("Khách hàng gọi lần thứ hai để phàn nàn vì đơn hàng giao trễ."))
        for whole in (CUT_STEM + ", rào cản lớn nhất nằm ở đâu?", "Khiếu nại này thuộc cấp độ nào? ",
                      'Chọn phương án đúng nhất:', "Đọc tình huống. Hãy chọn cách xử lý đúng.",
                      "Which option meets the criterion?", "Một CEO hỏi: “Nên bắt đầu từ đâu?”"):
            with self.subTest(whole=whole):
                self.assertTrue(question_complete(whole))

    def test_the_cut_question_is_a_slot_finding(self) -> None:
        unit = {"components": [ATTRIBUTES, _question(CUT_STEM, *ALLIANCE_DISTRACTORS[:1], *ALLIANCE_DISTRACTORS[1:])]}
        body = severity_body()
        brief = IdmUnitBriefV1.model_validate(body["unit_contract"]["idm_unit_brief"])
        codes = [(item.code, item.component_index) for item in deterministic_slot_findings(unit, brief, ["", ""])]
        self.assertIn((QUESTION_INCOMPLETE_CODE, 1), codes)


class JoinedLeakTests(unittest.TestCase):
    def test_a_key_stringing_the_list_heads_together_is_a_leak(self) -> None:
        teaching = unit_teaching([ATTRIBUTES], [])
        self.assertTrue(restates_list(_question("Nhận định nào đúng?", ATTRIBUTES_KEY, *ATTRIBUTES_DISTRACTORS),
                                      teaching))
        # The same list, one criterion named by an application key: not a list restated.
        applied = _question("Doanh nghiệp này còn thiếu thuộc tính nào?", "Uy tín tuyệt đối, vì tiến độ giao hàng "
                            "thường trễ hai tuần.", "Chất lượng đỉnh cao, vì sản phẩm có lỗi kỹ thuật.",
                            "Đổi mới liên tục, vì sản phẩm chưa có tính năng mới.")
        self.assertFalse(restates_list(applied, teaching))

    def test_a_key_opening_with_a_taught_label_applies_it(self) -> None:
        ladder = _html({"kind": "table", "rows": [
            {"label": "Bậc 01", "value": "Gia công thuần túy, phụ thuộc thiết kế của khách hàng."},
            {"label": "Bậc 02", "value": "Có chứng chỉ quốc tế, tỷ lệ lỗi dưới ngưỡng cho phép."},
            {"label": "Bậc 03", "value": "Tối ưu chi phí ẩn, làm chủ chuỗi vật tư nội địa."}]})
        key = "Bậc 02, vì có chứng chỉ quốc tế và tỷ lệ lỗi dưới ngưỡng cho phép nhưng chưa tối ưu chi phí ẩn."
        question = _question("Doanh nghiệp này ở bậc nào?", key, "Bậc 01, vì còn gia công cho khách hàng.",
                             "Bậc 03, vì đã làm chủ chuỗi vật tư nội địa.")
        self.assertFalse(restates_list(question, unit_teaching([ladder], [])))

    def test_a_key_joining_the_worked_example_rows_is_a_leak(self) -> None:
        teaching = unit_teaching([ALLIANCE], [])
        question = _question("Đâu là kế hoạch 90 ngày thể hiện đúng tư duy Think Ecosystem?", ALLIANCE_KEY,
                             *ALLIANCE_DISTRACTORS)
        self.assertTrue(restates_joined_example(question, teaching))
        # Without the worked example the same key is no leak.
        self.assertFalse(restates_joined_example(question, unit_teaching([ATTRIBUTES], [])))


class AbbreviationTests(unittest.TestCase):
    FACTS = ("Row 3: DO | SPD (Standard Product Dev) | Chuẩn hóa quy trình nghiên cứu và phát triển",
             "Tình Huống Điển Hình: Triển Khai BiC Tại FPT Software HCM",
             "SHIFT 1: THINK BIG — Từ Tư Duy Tồn Tại Sang Khát Vọng Lớn",
             "Đóng gói thành quy trình chuẩn (SOP) và nhân rộng.")

    def test_the_guide_reads_full_forms_names_and_headings_from_the_facts(self) -> None:
        guide = AcronymGuide.build(self.FACTS, "vi")
        self.assertEqual(guide.expansion("SPD"), "Standard Product Dev")
        self.assertEqual(guide.expansion("ROI"), "tỷ suất hoàn vốn đầu tư")
        self.assertIsNone(guide.expansion("BQS"))
        for token in ("FPT", "THINK", "BIG", "SHIFT", "ACTION", "PLAN"):
            with self.subTest(token=token):
                self.assertFalse(guide.worth_noting(token))
        self.assertTrue(guide.worth_noting("BQS"))
        self.assertEqual(abbreviations(["Áp dụng BQS/BMQ, ERA5.0 và KPI; mã CR-04 giữ nguyên."]), ["BQS", "BMQ", "KPI"])

    def test_the_full_form_goes_after_the_first_use_once(self) -> None:
        html = _html({"kind": "paragraph", "text": "Đo ROI của dự án."}, {"kind": "bullets", "items": ["ROI cao."]})
        expanded = with_expansion(html, "ROI", "tỷ suất hoàn vốn đầu tư")
        assert expanded is not None
        self.assertEqual(learner_texts(expanded), ["Đo ROI (tỷ suất hoàn vốn đầu tư) của dự án.", "ROI cao."])
        self.assertIsNone(with_expansion(html, "SPD", "Standard Product Dev"))
        self.assertTrue(explained("SOP", None, ["Quy trình thao tác chuẩn (SOP) giúp nhân rộng."]))
        self.assertTrue(explained("ROI", "tỷ suất hoàn vốn đầu tư", ["Tỷ suất hoàn vốn đầu tư là thước đo."]))
        self.assertFalse(explained("ROI", "tỷ suất hoàn vốn đầu tư", ["Đo ROI của dự án."]))


class AuthorTextTests(unittest.TestCase):
    def test_sme_questions_name_table_rows_in_plain_words(self) -> None:
        self.assertEqual(plain_row_refs("Nội dung đầy đủ của Row 1 (Năng lực) và Row 3 (KPI) là gì?"),
                         "Nội dung đầy đủ của dòng 1 (Năng lực) và dòng 3 (KPI) là gì?")
        self.assertEqual(plain_row_refs("Which value belongs in Row 2?"), "Which value belongs in row 2?")
        self.assertEqual(plain_row_refs("Arrow 3 stays"), "Arrow 3 stays")

    def test_the_note_says_why_first_and_lists_unexplained_abbreviations(self) -> None:
        note = build_unit_author_note(locale="vi", judge=JudgeOutcome("pass"), deterministic_codes=[],
                                      fallback_slots=[], ai_drafted=True,
                                      review_reasons=[REVIEW_REASON_EVIDENCE, REVIEW_REASON_SCENARIO],
                                      unexplained_acronyms=["BQS", "G8D"])
        self.assertTrue(note.startswith("Cần tác giả xem lại vì: tài liệu nguồn của phần này có hình ảnh hoặc bảng "
                                        "hệ thống chưa đọc chắc chắn"))
        self.assertIn("câu hỏi dùng tình huống do AI soạn, SME cần xác nhận tình huống và đáp án.", note)
        self.assertNotIn("Tình huống minh hoạ do AI soạn", note)
        self.assertTrue(note.endswith("SME bổ sung nghĩa ở lần xuất hiện đầu: BQS, G8D."))
        english = build_unit_author_note(locale="en", judge=JudgeOutcome("pass"), deterministic_codes=[],
                                         fallback_slots=[], ai_drafted=False, review_reasons=[REVIEW_REASON_EVIDENCE])
        self.assertTrue(english.startswith("Needs the author's review because: the source of this section has images"))


class RuleTextTests(unittest.TestCase):
    def test_writer_judge_and_faq_rules(self) -> None:
        body = severity_body()
        brief = IdmUnitBriefV1.model_validate(body["unit_contract"]["idm_unit_brief"])
        prompt = unit_writer_prompt("vi", course_title="C", audience=brief.target_audience, lesson_title="L",
                                    lesson_objective="O", practice_sentences=[], previous_title=None, next_title=None,
                                    unit_brief={}, facts=[], context_facts=[])
        for rule in ('ends with the question sentence itself ("…?")',
                     "The worked example meets every self-check item", "self-paced e-learning",
                     "Single-choice: 4 options", "never invent a full form"):
            with self.subTest(rule=rule):
                self.assertIn(" ".join(rule.split()), " ".join(prompt.split()))
        judged = " ".join(judge_prompt("vi", plan_summary={}, facts=[], unit_content=[]).split())
        self.assertIn("a question whose text stops before it asks anything is critical", judged)
        self.assertIn("a worked example that breaks one of its own worksheet's self-check criteria", judged)
        self.assertIn("at most about 60 words", faq_writer_rules("explain"))


class FaqFloorTests(unittest.IsolatedAsyncioTestCase):
    def test_the_golden_module_gets_faqs_up_to_a_quarter_of_its_learning_units(self) -> None:
        added_anywhere = 0
        for chapter in range(3):
            with self.subTest(chapter=chapter):
                before = golden_shard(chapter)
                after = golden_shard(chapter, faq_floor=True)
                learning = [unit for lesson in after.lessons for unit in lesson.units if unit.segment != "job_aid"]
                with_faq = [unit for unit in learning if any(c.type == "la_faq" for c in unit.components)]
                had = sum(any(c.type == "la_faq" for c in unit.components) for lesson in before.lessons
                          for unit in lesson.units if unit.segment != "job_aid")
                target = math.ceil(IDM_FAQ_MIN_UNIT_SHARE * len(learning))
                # Never past the target, never fewer than before.
                self.assertLessEqual(len(with_faq), max(target, had))
                self.assertGreaterEqual(len(with_faq), had)
                added_anywhere += len(with_faq) - had
                for lesson in after.lessons:
                    added = [component for unit in lesson.units for component in unit.components
                             if component.type == "la_faq" and component.title.startswith("Hỏi đáp: ")]
                    self.assertLessEqual(len(added), 1)
                    for component in added:
                        self.assertEqual((component.role, component.practice_id), ("clarify", None))
                        self.assertGreaterEqual(len(component.faq_items), 2)
                        self.assertIn("Đã thêm phần hỏi đáp cho", lesson.notes)
        self.assertGreater(added_anywhere, 0)

    async def test_the_floor_is_counted_and_never_takes_an_obligation_slot(self) -> None:
        result, _provider, runtime = await design_module(0, [module_response("mod_01")])
        design = result["shard"]["idm_design"]
        added = [unit for lesson in design["lessons"] for unit in lesson["units"]
                 if any(c["type"] == "la_faq" and c["title"].startswith("Hỏi đáp: ") for c in unit["components"])]
        self.assertEqual(runtime.adjustments[FAQ_ADDED_CODE], len(added))
        self.assertTrue(added)
        for unit in added:
            self.assertLessEqual(len(unit["components"]), 4)
            self.assertEqual(unit["components"][-1]["type"], "la_faq")


class UnitFlowTests(StoryboardEndpointTestCase):
    async def test_a_cut_question_gets_one_targeted_repair(self) -> None:
        body = severity_body()
        writer = severity_writer(body)
        whole = writer["components"]["c1"]["question"]
        cut = {**writer, "components": {**writer["components"], "c1": {**writer["components"]["c1"],
                                                                        "question": whole.split(".")[0]}}}
        provider = FakeGenerate(**{WRITER: [cut], REPAIR: [slot_repair(writer, 1)], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "validated", "provider")
        self.assertEqual(provider.names, [WRITER, REPAIR, JUDGE])
        self.assertEqual(provider.calls[1]["thinking_level"], "low")
        self.assertIn("c1 IDM_W5_QUESTION_INCOMPLETE at c1: the question stops before it asks anything",
                      provider.calls[1]["prompt"])
        self.assertEqual(data["unit"]["components"][1]["question"], whole)
        self.assertIn("Đã tự sửa: câu hỏi bị cụt.", quality.author_note)

    async def test_a_repair_that_cuts_the_question_keeps_the_earlier_question(self) -> None:
        body = severity_body()
        writer = severity_writer(body)
        whole = writer["components"]["c1"]["question"]
        # The key copies the table row word for word: a leak, one targeted repair.
        leaking = [{"text": "Thiệt hại tài chính dưới 50 triệu đồng hoặc khách hàng phàn nàn lần thứ hai",
                    "correct": True},
                   {"text": "Cấp 1 vì chưa có thiệt hại tài chính", "correct": False},
                   {"text": "Cấp 3 vì cần escalate ngay cho quản lý", "correct": False}]
        leak = {**writer, "components": {**writer["components"], "c1": {**writer["components"]["c1"],
                                                                         "choices": leaking}}}
        cut_repair = slot_repair(writer, 1, question="Một khách hàng gọi lần thứ hai để phàn nàn vì đơn hàng giao trễ")
        # Owner P3: the leak left after the first repair gets a second targeted one, which cuts the question again.
        provider = FakeGenerate(**{WRITER: [leak], REPAIR: [cut_repair, cut_repair], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "review_required", "provider")
        self.assertEqual(provider.names, [WRITER, REPAIR, REPAIR, JUDGE])
        served = data["unit"]["components"][1]
        self.assertEqual(served["question"], whole)
        self.assertIn("Thiệt hại tài chính dưới 50 triệu đồng hoặc khách hàng phàn nàn lần thứ hai",
                      [choice["text"] for choice in served["choices"]])
        self.assertIn("Giữ bản AI để tác giả rà soát: khối 2 (Quiz): phương án lộ đáp án", quality.author_note)
        self.assertTrue(quality.author_note.startswith("Cần tác giả xem lại vì: có khối AI viết chưa đạt hết kiểm tra"))

    async def test_abbreviations_get_their_full_form_or_a_note(self) -> None:
        body = severity_body()
        writer = severity_writer(body)
        paragraph = writer["components"]["c0"]["semantic_content"]["sections"][0]["blocks"][0]
        paragraph["text"] = "Mỗi khiếu nại thuộc một trong ba cấp độ. Cấp độ quyết định ai xử lý theo KPI và BQS."
        provider = FakeGenerate(**{WRITER: [writer], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "validated", "provider")
        self.assertIn("Cấp độ quyết định ai xử lý theo KPI (chỉ số đo lường hiệu quả) và BQS.",
                      " ".join(html_texts(data["unit"]["components"][0])))
        self.assertTrue(quality.author_note.endswith(
            "Chữ viết tắt chưa có nghĩa đầy đủ trong tài liệu — SME bổ sung nghĩa ở lần xuất hiện đầu: BQS."))


if __name__ == "__main__":
    unittest.main()
