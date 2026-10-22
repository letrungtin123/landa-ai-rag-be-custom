"""QLT-7: five quality fixes found by the QC of run fae66524 (8.5/10), tested with the run's own examples.

1. Drafted examples take the learner's role from the audience (11 of 17 were about "một trưởng nhóm" in a course for
   CEOs); a drafted example of another level gets one targeted repair, then an author note.
2. Explained slots copy at most one sentence in five, keep their sentences short and explain with a labelled example,
   a clarifying phrase or a why sentence (generic words no longer count).
3. Single-choice questions: the length cue against the average distractor, a key that restates the unit's worked
   example, a case that literally meets another label's table row.
4. The FAQ gate needs more than a practice question, keeps at most one weak-signal FAQ per lesson, never answers an
   open SME question and never invents a reason for a threshold.
5. Titles in Vietnamese sentence case, acronyms, codes and the source's proper names kept.

The provider is the local fake; nothing reaches the network.
"""

from __future__ import annotations

import json
import unittest
from typing import Any

from app.idm.answer_checks import (
    contested_option,
    example_passages,
    fact_table_rows,
    restates_example,
    unit_teaching,
)
from app.idm.contracts import IdmFaqMaterialV1, IdmLessonDesignV1, IdmUnitBriefV1, IdmW4CourseResponseV1, brief_hash_of
from app.idm.example_roles import audience_roles, contradicting_role, example_body, protagonist_level, role_rule
from app.idm.faq_checks import (
    FAQ_OPEN_QUESTION_CODE,
    FaqChecker,
    asks_open_question,
    faq_prompt_material,
    faq_writer_rules,
    invents_reason,
    open_questions,
    reasoned,
    threshold_numbers,
    unsupported_reasoning,
)
from app.idm.faq_plan import FaqCandidate, gate_and_plan_faq, gate_signal
from app.idm.fidelity import writer_fidelity_rules
from app.idm.fidelity_checks import (
    EXAMPLE_ROLE_CODE,
    EXPLANATION_MISSING_CODE,
    SENTENCE_COPY_CODE,
    SENTENCE_LENGTH_CODE,
    FidelityChecker,
    explains,
    length_report,
)
from app.idm.framework import is_framework_brief
from app.idm.mcq import answer_length_cue, option_length_stats
from app.idm.qa import ANSWER_CONTESTED_CODE, ANSWER_LEAK_CODE, deterministic_slot_findings
from app.idm.text import content_words, evidence_index
from app.idm.titles import TitleCaser, capitalised, sentence_case_lessons, sentence_case_plan, vietnamese_word
from tests.support.idm_test_support import rehash_contract
from tests.unit.idm.test_idm_faq import EXPLAINED, WHAT_IF, faq_writer
from tests.unit.idm.test_idm_fidelity import COPIED_VALUES, EXACT_RULE, FACTS, NOVEL, NOVEL_2, brief, html
from tests.unit.idm.test_idm_fidelity_unit import block, explain_body, explained_c0, html_texts, owned_of, writer_with
from tests.unit.idm.test_idm_module_design import component, design_module, edited, scope_of
from tests.unit.idm.test_idm_qlt3 import SHIFT_BRIEF, SHIFT_DETAIL, SHIFT_HEADS
from tests.unit.idm.test_idm_storyboard import (
    JUDGE,
    REPAIR,
    WRITER,
    FakeGenerate,
    StoryboardEndpointTestCase,
    escalate_body,
    judge,
    severity_body,
    slot_repair,
)

EXECUTIVES = "CEO, Founder và đội ngũ quản lý cấp cao"
# Drafted examples of the run (html of c2.l1.u2, c2.l3.u1 worksheet, c1.l2.u1, c2.l1.u1, c2.l6.u1).
TEAM_LEAD_EXAMPLES = (
    "Ví dụ minh hoạ: Một trưởng nhóm sản xuất trước đây chỉ đặt mục tiêu gia công đủ đơn hàng để duy trì xưởng.",
    "Ví dụ minh hoạ: Sau đây là cách một trưởng nhóm thiết kế lại điểm chạm bàn giao kết quả công việc:",
    "Ví dụ minh hoạ: Khi phát sinh lỗi giao hàng trễ, một trưởng nhóm làm việc theo mô hình cũ sẽ tìm người chịu lỗi.",
)
LEADER_EXAMPLE = ("Ví dụ minh hoạ: Một lãnh đạo doanh nghiệp bán lẻ thay đổi nhận thức từ việc cạnh tranh giá sang xây "
                  "dựng trải nghiệm khách hàng vượt trội và liên kết hệ sinh thái.")
LATE_ROLE_EXAMPLE = ("Ví dụ minh hoạ: Ở trục ứng dụng AI, thay vì chỉ khuyến khích một nhân viên mới dùng công cụ, "
                     "lãnh đạo đặt chỉ tiêu năng suất cho từng phòng.")


def with_audience(audience: str, *, mode: str = "explain") -> IdmUnitBriefV1:
    return brief(mode).model_copy(update={"target_audience": audience})


def role_checker(audience: str, mode: str = "explain") -> FidelityChecker:
    unit_brief = with_audience(audience, mode=mode)
    return FidelityChecker.build(unit_brief, FACTS, evidence_index(FACTS.values()), locale="vi",
                                 plan_texts=[unit_brief.course_title, unit_brief.target_audience])


class ExampleRoleTests(unittest.TestCase):
    def test_roles_come_from_the_audience_then_the_profile(self) -> None:
        roles = audience_roles(EXECUTIVES, "soft_skills", "vi")
        self.assertEqual(roles.levels, frozenset({"executive"}))
        self.assertEqual(roles.phrases, ("một CEO", "một nhà sáng lập", "một lãnh đạo cấp cao"))
        self.assertEqual(audience_roles("Nhân viên mới gia nhập công ty", None, "en").phrases, ("a new employee",))
        # No role word: the onboarding and sales profiles imply one; any other profile keeps the generic rule.
        self.assertEqual(audience_roles("Người học", "onboarding", "vi").phrases, ("một nhân viên mới",))
        self.assertEqual(audience_roles(None, "general", "vi").phrases, ())
        self.assertEqual(role_rule(audience_roles(None, "general", "vi")),
                         "the learner's own role as TARGET_AUDIENCE describes it")
        # Diacritics decide: "được công nhận" (recognised) names no worker.
        self.assertEqual(audience_roles("Doanh nghiệp được công nhận chất lượng", None, "vi").levels, frozenset())

    def test_the_writer_rule_names_the_audience_roles_never_a_fixed_team_lead(self) -> None:
        rules = writer_fidelity_rules("explain", "vi", audience_roles(EXECUTIVES, "soft_skills", "vi"))
        self.assertIn("The main person of a drafted example has the learner's own role as TARGET_AUDIENCE describes "
                      'it, for example "một CEO", "một nhà sáng lập", "một lãnh đạo cấp cao"', rules)
        self.assertIn("no team lead or junior staff in a course for executives", rules)
        self.assertNotIn("một nhân viên mới, một khách hàng, trưởng nhóm", rules)

    def test_drafted_examples_of_another_level_are_found_conservatively(self) -> None:
        executives = audience_roles(EXECUTIVES, None, "vi")
        for text in TEAM_LEAD_EXAMPLES:
            with self.subTest(text=text):
                body = example_body(text)
                assert body is not None
                self.assertEqual(protagonist_level(body), "team_lead")
                self.assertTrue(contradicting_role(executives, body))
        leader = example_body(LEADER_EXAMPLE)
        late = example_body(LATE_ROLE_EXAMPLE)
        assert leader is not None and late is not None
        self.assertEqual(protagonist_level(leader), "executive")
        self.assertFalse(contradicting_role(executives, leader))
        # A role named after the opening words is someone the main person deals with.
        self.assertIsNone(protagonist_level(late))
        # Only the extremes: a manager course may show a staff member; a staff course never a CEO.
        manager = example_body("Ví dụ minh hoạ: Một nhân viên mới hỏi trưởng phòng về quy trình.")
        assert manager is not None
        self.assertFalse(contradicting_role(audience_roles("Trưởng phòng kinh doanh", None, "vi"), manager))
        ceo = example_body("Ví dụ minh hoạ: Một CEO duyệt ngay yêu cầu đổi trả của khách.")
        assert ceo is not None
        self.assertTrue(contradicting_role(audience_roles("Nhân viên mới", "onboarding", "vi"), ceo))
        # Only labelled (drafted) examples are read; a source example keeps its own people.
        self.assertIsNone(example_body("Ví dụ: Một trưởng nhóm sản xuất trước đây chỉ đặt mục tiêu gia công."))

    def test_any_html_slot_with_such_an_example_is_a_finding_with_its_location(self) -> None:
        component = html(NOVEL, TEAM_LEAD_EXAMPLES[0])
        for mode in ("explain", "structured", "keep"):
            with self.subTest(mode=mode):
                checker = role_checker(EXECUTIVES, mode)
                self.assertIn((EXAMPLE_ROLE_CODE, 0), checker.findings([component]))
                self.assertEqual(checker.role_locations(component), ["semantic_content.sections[0].blocks[1]"])
        self.assertNotIn((EXAMPLE_ROLE_CODE, 0), role_checker("Nhân viên mới").findings([component]))
        self.assertNotIn((EXAMPLE_ROLE_CODE, 0), role_checker(EXECUTIVES).findings([html(NOVEL, LEADER_EXAMPLE)]))


class ExplainedSlotTests(unittest.TestCase):
    def test_one_copied_sentence_in_four_is_too_many_when_explained(self) -> None:
        component = html(COPIED_VALUES, NOVEL, NOVEL_2, "Nghĩa là bạn sẽ thấy các giá trị này trong từng việc nhỏ hằng "
                         "ngày ở công ty.")
        explained_checker = role_checker("Nhân viên mới")
        report = explained_checker.report(0, component)
        self.assertEqual((report.judged, len(report.copied)), (4, 1))
        self.assertEqual(explained_checker.copy_limit(0), 0.2)
        self.assertIn((SENTENCE_COPY_CODE, 0), explained_checker.findings([component]))
        self.assertNotIn((SENTENCE_COPY_CODE, 0), role_checker("Nhân viên mới", "structured").findings([component]))
        # Rules, definitions and numbers stay exact and are not judged.
        self.assertEqual(explained_checker.report(0, html(EXACT_RULE, NOVEL, NOVEL_2)).judged, 2)

    def test_prose_sentences_are_measured_without_tables_lists_callouts_or_rules(self) -> None:
        # A sentence as long as the longest of run fae66524 (c4.l3.u1, 57 words) and a short one.
        long_sentence = ("Ví dụ minh hoạ: Một lãnh đạo dịch vụ khách hàng lựa chọn quy trình tiếp nhận phản ánh hay "
                         "phát sinh khiếu nại để phân tích 5Why nhằm loại bỏ nguyên nhân gốc rễ, đồng thời đưa công cụ "
                         "số hóa vào xử lý hồ sơ để giảm thiểu thời gian chờ đợi của khách hàng và tăng tốc độ phản "
                         "hồi trong toàn bộ hành trình dịch vụ.")
        short = "Khách hàng cần được phản hồi sớm."
        component = {"type": "html", "semantic_content": {"sections": [{"heading": "Vì sao cần làm ngay?", "blocks": [
            block("paragraph", text=f"{long_sentence} {short}"),
            block("bullets", items=[long_sentence]),
            block("table", rows=[{"label": "Bước 1", "value": long_sentence}]),
            block("warning", text=long_sentence),
            block("paragraph", text=EXACT_RULE + " " + EXACT_RULE.replace("14", "15"))]}]}}
        report = length_report(component)
        self.assertEqual(report.sentences, 2)
        self.assertGreater(report.longest, 40)
        self.assertEqual(report.long_locations, ("semantic_content.sections[0].blocks[0]",))
        self.assertTrue(report.too_long)
        # Average over 22 words with no sentence over 40 is too long as well; a short slot is fine.
        medium = " ".join(["Bạn kiểm tra thông tin khách hàng trên hệ thống, đối chiếu đơn hàng và ghi lại kết quả vào "
                           "phiếu theo dõi trước khi gọi lại cho khách hàng."] * 3)
        self.assertTrue(length_report(html(medium)).too_long)
        self.assertFalse(length_report(html(short, NOVEL)).too_long)
        self.assertFalse(length_report(html(medium[:150].rsplit(" ", 1)[0] + ".")).too_long)  # one sentence of < 40

    def test_an_explained_slot_with_long_sentences_is_a_finding(self) -> None:
        component = html("Nghĩa là " + " ".join(["mỗi ngày bạn làm việc cùng đồng nghiệp ở nhiều phòng ban khác nhau"]
                                                * 5) + ".", NOVEL)
        self.assertIn((SENTENCE_LENGTH_CODE, 0), role_checker("Nhân viên mới").findings([component]))
        self.assertNotIn((SENTENCE_LENGTH_CODE, 0),
                         role_checker("Nhân viên mới", "structured").findings([component]))

    def test_generic_words_alone_no_longer_explain(self) -> None:
        for text in ("Điều này giúp bạn làm việc tốt hơn để phục vụ khách hàng vì khách luôn đúng.",
                     "Bạn nên nắm rõ các giá trị so với năm trước."):
            with self.subTest(text=text):
                self.assertFalse(explains(text))
        for text in ("Nghĩa là bạn luôn hỏi lại khi chưa rõ.", "Ví dụ minh hoạ: Một nhân viên mới hỏi người phụ trách.",
                     "Vì sao giá trị cốt lõi quan trọng với bạn?", "Việc này quan trọng vì khách hàng nhìn vào bạn.",
                     "Vì vậy, hãy hỏi trước khi làm.", "This means you ask first.", "Why does it matter?",
                     "It matters because customers notice."):
            with self.subTest(text=text):
                self.assertTrue(explains(text))
        plain = html("Bạn sẽ làm việc cùng đồng nghiệp ở mọi phòng ban để giúp khách hàng.", NOVEL, NOVEL_2,
                     heading="Công ty mong đợi gì ở bạn?")
        self.assertIn((EXPLANATION_MISSING_CODE, 0), role_checker("Nhân viên mới").findings([plain]))


class LengthCueTests(unittest.TestCase):
    @staticmethod
    def question(key: str, *others: str) -> dict[str, Any]:
        return {"type": "problem", "choices": [{"text": key, "correct": True},
                                               *({"text": text, "correct": False} for text in others)]}

    def test_the_key_is_measured_against_the_average_and_the_longest_distractor(self) -> None:
        # c1.l1.u2 of the run: 30 characters against 22, 24 and 25 (1.27 x the average, 1.2 x the longest).
        ladder = self.question("Bậc 01 (Vietnam Manufacturing)", "Bậc 03 (Vietnam Value)", "Bậc 02 (Vietnam Quality)",
                               "Bậc 04 (Vietnam Know-how)")
        self.assertEqual(option_length_stats(ladder), (30, 71 / 3, 25))
        self.assertTrue(answer_length_cue(ladder))
        # c2.l1.u2: 177 characters against at most 147 (1.204 x the longest distractor, 1.23 x the average).
        self.assertTrue(answer_length_cue(self.question("k" * 177, "a" * 147, "b" * 145, "c" * 139)))
        # c1.l2.u2: 139 against 125 at most (1.11, 1.19 x the average) is no cue; neither are two options.
        self.assertFalse(answer_length_cue(self.question("k" * 139, "a" * 125, "b" * 115, "c" * 109)))
        self.assertIsNone(option_length_stats(self.question("k" * 90, "a" * 10)))


# c2.l3.u1 of the run: the worksheet's example row and the question that follows it.
WORKSHEET_EXAMPLE = {"semantic_content": {"sections": [
    {"heading": "Biểu mẫu tái thiết kế điểm chạm", "blocks": [block("table", rows=[
        {"label": "Tên điểm chạm trọng yếu", "value": "Điểm chạm nào trên hành trình cần tái thiết kế?"},
        {"label": "Phương án tạo cảm xúc mới",
         "value": "Hành động cụ thể nào mang lại trải nghiệm bất ngờ vượt trội?"}])]},
    {"heading": "Ví dụ minh họa mẫu hoàn thành", "blocks": [
        block("paragraph",
              text="Ví dụ minh hoạ: Sau đây là cách một lãnh đạo thiết kế lại điểm chạm bàn giao kết quả:"),
        block("table", rows=[
            {"label": "Tên điểm chạm trọng yếu", "value": "Bàn giao kết quả dự án cho khách hàng"},
            {"label": "Phương án tạo cảm xúc mới",
             "value": "Cung cấp video hướng dẫn cá nhân hóa và gửi quà tri ân bất ngờ"}])]}]}, "type": "html"}
WORKSHEET_QUESTION = {"type": "problem", "question": (
    "Một doanh nghiệp đang rà soát lại quy trình phục vụ khách hàng. Ban điều hành đề xuất bốn phương án cải tiến "
    "điểm chạm. Phương án nào thể hiện đúng tinh thần Chuyển dịch Think Customer hướng tới cảm xúc Wow & Delight và "
    "gắn kết trọn đời?"), "choices": [
    {"text": "Tăng cường kiểm soát biên bản kiểm thử để tỷ lệ giao hàng đạt 95% hợp đồng và hạn chế phát sinh lỗi kỹ "
             "thuật.", "correct": False},
    {"text": "Chủ động gửi hướng dẫn cá nhân hóa và dịch vụ tri ân bất ngờ để khách hàng tin tưởng tuyệt đối và vui vẻ "
             "trả giá cao.", "correct": True},
    {"text": "Tập trung rà soát danh mục bàn giao đúng cam kết hợp đồng để nghiệm thu nhanh và giảm thiểu phàn nàn kỹ "
             "thuật.", "correct": False},
    {"text": "Thiết lập hệ thống kiểm tra nội bộ nghiêm ngặt nhằm đảm bảo sản phẩm xuất xưởng không bị khách hàng phạt "
             "hợp đồng.", "correct": False}],
    "explanation": "B đúng vì vượt qua kiểm soát lỗi. A, C, D sai vì chỉ kiểm soát lỗi kỹ thuật theo tư duy cũ."}
# c1.l1.u2 of the run: the ladder rows of the html and the contested case.
LADDER_ROWS = [
    ("Bậc 01 - Vietnam Manufacturing", "Gia công thuần túy, phụ thuộc khách hàng cung cấp nguyên phụ liệu và thiết kế, "
                                       "biên lợi nhuận mỏng manh từ 3% đến 5%."),
    ("Bậc 02 - Vietnam Quality", "Doanh nghiệp đáp ứng quy chuẩn kỹ thuật độc lập, đạt chứng chỉ quốc tế như ISO, CE "
                                 "hoặc FDA, giữ tỷ lệ sai lỗi trong ngưỡng cho phép."),
    ("Bậc 03 - Vietnam Value", "Cắt giảm triệt để chi phí lãng phí, tự chủ nguồn cung vật tư nội địa, từ đó nâng cao "
                               "tỷ suất lợi nhuận và giá trị giữ lại."),
    ("Bậc 04 - Vietnam Know-how", "Nắm giữ công nghệ cốt lõi, sở hữu bí quyết sản xuất hoặc quy trình thiết kế riêng "
                                  "biệt khiến đối thủ khó sao chép."),
]
LADDER_FACTS = [
    "Row 1: CẤP ĐỘ | ĐỊNH VỊ | ĐẶC TRƯNG VẬN HÀNH",
    "Row 2: Bậc 01 | Vietnam Manufacturing | Gia công thuần túy, phụ thuộc khách hàng cung cấp nguyên phụ liệu và "
    "thiết kế. Biên lợi nhuận mỏng manh (3–5%).",  # noqa: RUF001 - the source's en dash
    "Row 3: Bậc 02 | Vietnam Quality | Đạt chuẩn kỹ thuật, có các chứng nhận quốc tế (ISO, CE, FDA). Sản phẩm ổn định, "
    "tỷ lệ lỗi dưới ngưỡng cho phép.",
    "Row 4: Bậc 03 | Vietnam Value | Tối ưu hóa chi phí ẩn, làm chủ chuỗi cung ứng vật tư trong nước.",
]
LADDER_CASE = ("Một doanh nghiệp dệt may SME đạt chứng nhận chất lượng ISO 9001 và chứng chỉ CE. Toàn bộ dây chuyền "
               "chỉ thực hiện may ráp theo đúng thiết kế và phụ liệu do đối tác nước ngoài gửi sang; doanh nghiệp "
               "không tự chủ nguồn vải, không tối ưu được chi phí ẩn và biên lợi nhuận ròng hàng năm dao động ở mức "
               "3,5% - 4%. Căn cứ Nấc thang Giá trị Made-in-World, doanh nghiệp này thực chất đang ở bậc năng lực nào?")


def ladder_question(case: str, key: int = 0) -> dict[str, Any]:
    labels = ["Bậc 01 (Vietnam Manufacturing)", "Bậc 03 (Vietnam Value)", "Bậc 02 (Vietnam Quality)",
              "Bậc 04 (Vietnam Know-how)"]
    return {"type": "problem", "question": case, "explanation": "A đúng; B, C, D sai theo bảng nấc thang.",
            "choices": [{"text": label, "correct": index == key} for index, label in enumerate(labels)]}


def ladder_html(rows: list[tuple[str, str]] = LADDER_ROWS) -> dict[str, Any]:
    return {"type": "html", "semantic_content": {"sections": [{"heading": "Sáu bậc gồm những gì?", "blocks": [
        block("table", rows=[{"label": label, "value": value} for label, value in rows])]}]}}


class QuestionTests(unittest.TestCase):
    def test_a_key_that_restates_the_worked_example_leaks_the_answer(self) -> None:
        passages = example_passages([WORKSHEET_EXAMPLE])
        self.assertIn("Cung cấp video hướng dẫn cá nhân hóa và gửi quà tri ân bất ngờ", passages)
        # The example label goes; template rows outside the example section are no example.
        self.assertIn("Sau đây là cách một lãnh đạo thiết kế lại điểm chạm bàn giao kết quả:", passages)
        self.assertNotIn("Điểm chạm nào trên hành trình cần tái thiết kế?", passages)
        teaching = unit_teaching([WORKSHEET_EXAMPLE], [])
        self.assertTrue(restates_example(WORKSHEET_QUESTION, teaching))
        unit = {"components": [WORKSHEET_EXAMPLE, WORKSHEET_QUESTION]}
        found = deterministic_slot_findings(unit, brief(None), ["", ""])
        self.assertIn(ANSWER_LEAK_CODE, [item.code for item in found])
        # The key applied to its own case in new words is no leak; a key naming a taught label never is.
        fresh = json.loads(json.dumps(WORKSHEET_QUESTION))
        fresh["choices"][1]["text"] = ("Gọi điện hỏi thăm sau bàn giao và tặng buổi huấn luyện riêng để khách tự tin "
                                       "giới thiệu thêm đối tác.")
        self.assertFalse(restates_example(fresh, teaching))
        labelled = {"choices": [{"text": "Cấp 2: thông báo trưởng nhóm vì khách phàn nàn lần hai", "correct": True},
                                {"text": "Cấp 1: tự xử lý ngay trong cuộc gọi", "correct": False},
                                {"text": "Cấp 3: escalate cho quản lý và pháp chế", "correct": False}]}
        filled = "Khiếu nại mẫu thứ nhất: Cấp 2: thông báo trưởng nhóm vì khách phàn nàn lần hai."
        example = {"type": "html", "semantic_content": {"sections": [{"heading": "Ví dụ một dòng đã điền", "blocks": [
            block("paragraph", text=filled)]}]}}
        self.assertFalse(restates_example(labelled, unit_teaching([example], [])))

    def test_a_case_that_literally_meets_a_higher_rung_contests_the_key(self) -> None:
        teaching = unit_teaching([ladder_html()], [])
        self.assertEqual(contested_option(ladder_question(LADDER_CASE), teaching), 2)
        unit = {"components": [ladder_html(), ladder_question(LADDER_CASE)]}
        self.assertIn(ANSWER_CONTESTED_CODE,
                      [item.code for item in deterministic_slot_findings(unit, brief(None), ["", ""])])
        # Negated in the case ("chưa đạt", "không có"), the criterion is not met.
        negated = LADDER_CASE.replace("đạt chứng nhận chất lượng ISO 9001 và chứng chỉ CE",
                                      "chưa đạt chứng nhận ISO nào, không có chứng chỉ CE")
        self.assertIsNone(contested_option(ladder_question(negated), teaching))
        # A higher rung meets the lower rungs' criteria by design: a Bậc 03 case with ISO and CE is not contested.
        higher = ("Một doanh nghiệp đã đạt chứng nhận ISO 9001 và chứng chỉ CE, cắt giảm triệt để chi phí lãng phí và "
                  "tự chủ nguồn cung vật tư nội địa. Doanh nghiệp đang ở bậc nào?")
        self.assertIsNone(contested_option(ladder_question(higher, key=1), teaching))

    def test_the_table_row_facts_count_when_the_html_shows_no_table(self) -> None:
        rows = fact_table_rows(LADDER_FACTS)
        self.assertEqual(rows[0], ("Bậc 01 - Vietnam Manufacturing", "Gia công thuần túy, phụ thuộc khách hàng cung "
                                   "cấp nguyên phụ liệu và thiết kế. Biên lợi nhuận mỏng manh (3–5%)."))  # noqa: RUF001
        self.assertEqual(len(rows), 3)  # the header row (every cell in capitals) is left out
        plain = {"type": "html", "semantic_content": {"sections": [{"heading": "Nấc thang giá trị", "blocks": [
            block("paragraph", text="Mỗi bậc là một mức năng lực vận hành mà doanh nghiệp phải tích lũy.")]}]}}
        self.assertIsNone(contested_option(ladder_question(LADDER_CASE), unit_teaching([plain], [])))
        self.assertEqual(contested_option(ladder_question(LADDER_CASE), unit_teaching([plain], [], rows)), 2)


# c4.l3.u1 of the run: the 30% threshold and the FAQ that invented its reason; c1.l3.u1: an open SME question.
QUICK_WIN_FACT = ("01 Ứng dụng AI hoặc số hóa cụ thể được đưa vào vận hành thực tế giúp nâng cao tối thiểu 30% năng "
                  "suất phòng ban.")
INVENTED_QUESTION = ("Vì sao cam kết ứng dụng AI lại đặt ra ngưỡng cải thiện tối thiểu 30% năng suất phòng ban mà "
                     "không phải mức thấp hơn?")
INVENTED_ANSWER = ("Ngưỡng tăng tối thiểu 30% năng suất đảm bảo việc ứng dụng AI tạo ra bước nhảy vọt rõ rệt trong "
                   "hiệu quả vận hành thực tế thay vì chỉ thử nghiệm hình thức.")
OPEN_SME = ("Thứ tự triển khai 9 công cụ này có bắt buộc tuần tự theo PDCA không, hay doanh nghiệp có thể chọn lọc "
            "công cụ tùy theo thực trạng?")
OPEN_FAQ = "Doanh nghiệp có cần triển khai đồng loạt toàn bộ 9 công cụ di sản theo đúng thứ tự PDCA không?"


class FaqTests(unittest.TestCase):
    def test_thresholds_deadlines_and_amounts_but_not_counts_or_levels(self) -> None:
        self.assertEqual(threshold_numbers(INVENTED_QUESTION), {"30"})
        self.assertEqual(threshold_numbers("Hoàn tất trong 90 ngày, thiệt hại dưới 50 triệu đồng"), {"90", "50"})
        self.assertEqual(threshold_numbers("Tối thiểu 3 người duyệt"), {"3"})
        self.assertEqual(threshold_numbers("6 bậc của nấc thang, 8 cam kết và Bậc 02"), set())

    def test_a_reason_the_facts_never_give_for_a_threshold_is_invented(self) -> None:
        self.assertTrue(invents_reason(INVENTED_QUESTION, INVENTED_ANSWER, [QUICK_WIN_FACT]))
        # The same answer to a question that asks what the threshold requires still gives its own reason.
        self.assertTrue(invents_reason("Cam kết ứng dụng AI cần đạt mức nào?", INVENTED_ANSWER, [QUICK_WIN_FACT]))
        # A reason the facts state, or no reason at all, is fine.
        stated = QUICK_WIN_FACT.replace("phòng ban.", "phòng ban, vì dưới mức này lợi ích không bù được chi phí.")
        self.assertTrue(reasoned({"30"}, [stated]))
        self.assertFalse(invents_reason(INVENTED_QUESTION, INVENTED_ANSWER, [stated]))
        self.assertFalse(invents_reason("Điều gì xảy ra nếu chỉ thử nghiệm AI cho có?", "Phòng ban sẽ không đạt mức "
                                        "tối thiểu 30% năng suất của cam kết.", [QUICK_WIN_FACT]))

    def test_an_faq_question_that_asks_an_open_sme_question_again(self) -> None:
        open_words = [frozenset(content_words(OPEN_SME))]
        self.assertTrue(asks_open_question(OPEN_FAQ, open_words))
        self.assertFalse(asks_open_question("Sai lầm phổ biến nào khiến doanh nghiệp thất bại khi áp dụng kiểm tra "
                                            "chất lượng ở giai đoạn Check?", open_words))

    def test_sme_questions_of_an_older_brief_reach_neither_the_checker_nor_the_writer(self) -> None:
        # Owner decision 2026-10-22: no question for an SME. A brief built from an older design may still carry some;
        # they are ignored (no open question, no OPEN_QUESTIONS rule, no question in the FAQ material).
        raw = escalate_body()["unit_contract"]["idm_unit_brief"]
        raw["faq_material"]["blocks"][0]["sme_questions"] = ["Có nên hứa hoàn tiền cho khách VIP trước khi xác minh "
                                                             "đơn hàng không?"]
        raw["faq_material"]["open_questions"] = ["Tiêu chí nào để đánh giá đạt bản cam kết?"]
        unit_brief = IdmUnitBriefV1.model_validate(raw)
        self.assertEqual(open_questions(unit_brief), ())
        checker = FaqChecker.build(unit_brief, {}, plan_texts=[])
        faq = {"type": "la_faq", "items": [
            {"question": "Có nên hứa hoàn tiền khi khách vừa gọi đến không?", "answer": EXPLAINED},
            {"question": "Vì sao cần xác minh?", "answer": WHAT_IF}]}
        self.assertEqual(checker.open_items(faq), [])
        self.assertNotIn(FAQ_OPEN_QUESTION_CODE, checker.findings(1, faq))
        rules = faq_writer_rules("explain")
        self.assertNotIn("OPEN_QUESTIONS", rules)
        self.assertNotIn("SME", rules)
        self.assertIn("Never give a reason, cause or purpose for a number, percentage, amount, deadline or threshold",
                      rules)
        material = faq_prompt_material(unit_brief, has_faq=True)["faq_material"]
        self.assertNotIn("open_questions", material)
        self.assertNotIn("sme_questions", material["blocks"][0])
        self.assertEqual(faq_prompt_material(unit_brief, has_faq=False), {})

    def test_open_questions_are_optional_in_the_material(self) -> None:
        material = {"blocks": [], "practices": []}
        self.assertEqual(IdmFaqMaterialV1.model_validate(material).model_dump(mode="json"), material)
        opened = {**material, "open_questions": ["Quy định nào đang áp dụng?"]}
        self.assertEqual(IdmFaqMaterialV1.model_validate(opened).model_dump(mode="json"), opened)
        # A brief without them (every brief built before QLT-7) keeps no such key: its hash stays as Node sealed it.
        raw = escalate_body()["unit_contract"]["idm_unit_brief"]
        self.assertNotIn("open_questions", raw["faq_material"])
        self.assertEqual(brief_hash_of(raw), raw["brief_hash"])
        self.assertNotIn("open_questions", IdmUnitBriefV1.model_validate(raw).model_dump(mode="json")["faq_material"])

    def test_a_reasoning_sentence_no_fact_holds_is_marked_inferred(self) -> None:
        self.assertTrue(unsupported_reasoning(INVENTED_ANSWER, [QUICK_WIN_FACT]))
        self.assertFalse(unsupported_reasoning("Ứng dụng AI giúp nâng cao năng suất phòng ban ít nhất 30%.",
                                               [QUICK_WIN_FACT]))
        self.assertFalse(unsupported_reasoning("Phòng ban cần đo năng suất trước và sau.", [QUICK_WIN_FACT]))

    def test_a_practice_question_alone_never_opens_the_gate_and_weak_signals_open_one_per_lesson(self) -> None:
        material = FaqCandidate("common_mistake", ("a",), "distractor", "material")
        weak = FaqCandidate("common_mistake", ("b",), "ai_drafted", "weak")
        strong = FaqCandidate("why", ("c",), "source")
        self.assertIsNone(gate_signal([material]))
        self.assertEqual(gate_signal([material, weak]), "weak")
        self.assertEqual(gate_signal([weak, strong]), "strong")
        scope = scope_of(2)
        for block_id in ("cb_0006", "cb_0008"):
            raw = scope.blocks[block_id].model_dump(mode="json")
            raw["gaps"] = [{"type": "missing_common_mistake", "note": "Nguồn không nêu lỗi hay gặp."}]
            scope.blocks[block_id] = type(scope.blocks[block_id]).model_validate(raw)

        def two_faqs(payload: dict[str, Any]) -> None:
            units = payload["lessons"][0]["units"]
            units[0]["components"].append(component("la_faq", "clarify", ["cb_0007"]))
            units[1]["components"].append(component("la_faq", "clarify", ["cb_0008"]))

        lesson = IdmLessonDesignV1.model_validate(edited("mod_03", two_faqs)["lessons"][0])
        result = gate_and_plan_faq([lesson], scope)
        self.assertEqual((result.planned, result.dropped), (1, 1))
        self.assertEqual([[item.type for item in unit.components] for unit in result.lessons[0].units],
                         [["html", "la_faq"], ["html", "la_sortable"]])
        self.assertIn('Đã bỏ phần hỏi đáp của "Hoàn tất và sắp xếp trình tự": tài liệu chưa nêu quyết định',
                      result.lessons[0].notes)


QC_TITLES = (
    ("Bối Cảnh Mới Và Tuyên Bố Sứ Mệnh Made-in-World", "Bối cảnh mới và tuyên bố sứ mệnh Made-in-World"),
    ("Chuyển Dịch Vận Hành Sang Chuẩn BiC 5.0 Và Bài Học FPT Software",
     "Chuyển dịch vận hành sang chuẩn BiC 5.0 và bài học FPT Software"),
    ("Chuyển Dịch Think Big: Xác Định Tham Vọng Ngành Và Nâng Cấp Năng Lực Lõi",
     "Chuyển dịch Think Big: Xác định tham vọng ngành và nâng cấp năng lực lõi"),
    ("Thiết Lập Lộ Trình Chuyển Đổi Thực Thi 90-Day Challenge",
     "Thiết lập lộ trình chuyển đổi thực thi 90-Day Challenge"),
    ("Checklist 8 Cam Kết Quick-Win Trong 90 Ngày", "Checklist 8 cam kết Quick-Win trong 90 ngày"),
    ("Thực Hành Hoàn Thiện CEO Change Mindset Canvas 9 Ô", "Thực hành hoàn thiện CEO Change Mindset Canvas 9 ô"),
    ("Lộ Trình 3 Giai Đoạn: See Different - Think Different - Act Different",
     "Lộ trình 3 giai đoạn: See Different - Think Different - Act Different"),
    ("Mô Thức Siêu Kết Nối 6 Trụ Cột Nguồn Lực ERA5.0", "Mô thức siêu kết nối 6 trụ cột nguồn lực ERA5.0"),
    ("Thẩm Định Đề Xuất Đổi Mới Theo Bộ Lọc Tam Hóa", "Thẩm định đề xuất đổi mới theo bộ lọc Tam Hóa"),
    ("Kiểm Tra Nhận Thức Về 4 Lực Đẩy Bối Cảnh Mới", "Kiểm tra nhận thức về 4 lực đẩy bối cảnh mới"),
    # Sentence case already, an English title, names and the existing html label behaviour stay as they are.
    ("Chuyển dịch Think Big: Thiết lập tham vọng và năng lực lõi",
     "Chuyển dịch Think Big: Thiết lập tham vọng và năng lực lõi"),
    ("Think Big Và Think Different", "Think Big Và Think Different"),
    ("Nguyễn Văn An Và Đội Ngũ", "Nguyễn Văn An và đội ngũ"),
    ("Hội Nhập Tại Thành Phố Hồ Chí Minh", "Hội nhập tại Thành Phố Hồ Chí Minh"),
    ("Bản đồ 5 chuyển dịch tư duy BiC 5.0", "Bản đồ 5 chuyển dịch tư duy BiC 5.0"),
)
# Facts of the run that write the proper names inside a sentence.
NAME_FACTS = ("Mô hình liên minh nguồn lực & Bộ lọc Tam Hóa", "Triết lý Tam Doanh gồm những thành tố nào?",
              "Mọi quyết định du nhập công nghệ đều phải đi qua bộ lọc Tam Hóa để tránh hai cực đoan nguy hiểm.",
              "Biết cách liên kết mạng lưới nguồn lực bên ngoài (Chuyên gia, AI, Vốn, Thị trường).")


def w4_plan(course_title: str, module_title: str, lesson_title: str) -> IdmW4CourseResponseV1:
    return IdmW4CourseResponseV1.model_validate({
        "course_title": course_title, "course_summary": "Khoá học giúp lãnh đạo tái định hình tư duy quản trị.",
        "assessment_strategy": "Bài luyện tập sau mỗi mục.", "prerequisites": [], "pattern_sections": [],
        "modules": [{"module_key": "mod_01", "title": module_title,
                     "performance_goal": "Định vị doanh nghiệp đúng bậc.", "lo_ids": ["lo_1"], "lessons": [{
                         "lesson_key": "lsn_001", "kind": "learning", "title": lesson_title,
                         "primary_must_do_id": "md_1", "secondary_must_do_ids": [], "block_ids": ["cb_0001"],
                         "est_screens": 3, "est_minutes": 6, "ordering_rationale": "Theo khung."}]}]})


class TitleTests(unittest.TestCase):
    def test_titles_of_the_run_in_sentence_case(self) -> None:
        caser = TitleCaser.from_texts(NAME_FACTS)
        self.assertEqual(caser.protected, frozenset({("tam", "hóa"), ("tam", "doanh")}))
        for title, expected in QC_TITLES:
            with self.subTest(title=title):
                self.assertEqual(caser.title(title), expected)
        self.assertEqual(caser.title("Nấc Thang Giá Trị Made-in-World Và Triết Lý Tam Doanh"),
                         "Nấc thang giá trị Made-in-World và triết lý Tam Doanh")
        # Without the facts, a proper name is just Vietnamese words.
        self.assertEqual(TitleCaser().title("Giải Đáp Thắc Mắc Về Bộ Lọc Tam Hóa"),
                         "Giải đáp thắc mắc về bộ lọc tam hóa")

    def test_vietnamese_words_and_capitals(self) -> None:
        for word in ("Hành", "Doanh", "Quan", "Tam", "Ô"):
            self.assertTrue(vietnamese_word(word) and capitalised(word), word)
        for word in ("See", "Lean", "Big", "Think", "BiC", "CEO", "ERA5", "Made-in-World"):
            self.assertFalse(vietnamese_word(word) and capitalised(word), word)

    def test_the_course_plan_keeps_the_author_course_name(self) -> None:
        caser = TitleCaser()
        plan, changed = sentence_case_plan(w4_plan("Best-in-Class (BiC) Cho Lãnh Đạo", "Định Vị Tổ Chức Và Chuẩn Mực "
                                                   "Quản Trị", "Định Vị Doanh Nghiệp Trên Nấc Thang Giá Trị"),
                                           caser, locale="vi", author_title="best-in-class (BiC) cho lãnh đạo")
        self.assertEqual(changed, 2)
        self.assertEqual(plan.course_title, "Best-in-Class (BiC) Cho Lãnh Đạo")
        self.assertEqual(plan.modules[0].title, "Định vị tổ chức và chuẩn mực quản trị")
        self.assertEqual(plan.modules[0].lessons[0].title, "Định vị doanh nghiệp trên nấc thang giá trị")
        renamed, _ = sentence_case_plan(w4_plan("Khoá Học Chuyển Đổi Tư Duy", "Chương Một Của Khoá",
                                                "Bài Học Đầu Tiên"), caser, locale="vi",
                                        author_title="Best-in-Class (BiC)")
        self.assertEqual(renamed.course_title, "Khoá học chuyển đổi tư duy")
        english = w4_plan("Customer Service Basics", "Handling Complaints", "First Steps")
        self.assertEqual(sentence_case_plan(english, caser, locale="en", author_title=None), (english, 0))

    def test_a_module_lesson_takes_its_w4_title_when_only_the_case_differs(self) -> None:
        lesson = IdmLessonDesignV1.model_validate(edited("mod_03", lambda _p: None)["lessons"][0])
        titled = lesson.model_copy(update={"title": "Tám Bước Đầu Của Quy Trình Tiếp Nhận", "units": [
            unit.model_copy(update={"title": "Tám Bước Đầu Của Quy Trình", "components": [
                item.model_copy(update={"title": "Thành Phần Giải Thích Quy Trình"}) for item in unit.components]})
            for unit in lesson.units]})
        result, changed = sentence_case_lessons([titled], TitleCaser(), locale="vi",
                                                plan_titles={lesson.lesson_key: "Tám bước đầu của quy trình tiếp nhận"})
        self.assertEqual(result[0].title, "Tám bước đầu của quy trình tiếp nhận")
        self.assertEqual({unit.title for unit in result[0].units}, {"Tám bước đầu của quy trình"})
        self.assertEqual({item.title for unit in result[0].units for item in unit.components},
                         {"Thành phần giải thích quy trình"})
        self.assertEqual(changed, 1 + 2 * 2 + 1)


class ModuleTitleTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_design_and_its_projection_name_every_title_alike(self) -> None:
        def title_case(payload: dict[str, Any]) -> None:
            lesson = payload["lessons"][0]
            lesson["title"] = "Tiêu Chí Bắt Buộc Và Quyết Định Chuyển Cấp"
            lesson["units"][0]["title"] = "Tiêu Chí Bắt Buộc Chuyển Cấp Khiếu Nại"

        result, _provider, runtime = await design_module(1, [edited("mod_02", title_case)])
        design = result["shard"]["idm_design"]["lessons"][0]
        projected = result["shard"]["lessons"][0]
        self.assertEqual(design["title"], "Tiêu chí bắt buộc và quyết định chuyển cấp")
        self.assertEqual(design["units"][0]["title"], "Tiêu chí bắt buộc chuyển cấp khiếu nại")
        self.assertEqual((projected["title"], projected["units"][0]["title"]),
                         (design["title"], design["units"][0]["title"]))
        self.assertGreaterEqual(runtime.adjustments["IDM_W4_TITLES_SENTENCE_CASE"], 2)


def executive_body() -> dict[str, Any]:
    """The explained severity unit in a course whose author entered executives as the audience."""

    body = explain_body()
    unit_brief = body["unit_contract"]["idm_unit_brief"]
    unit_brief.update(target_audience=EXECUTIVES, purpose_profile="soft_skills")
    unit_brief["brief_hash"] = brief_hash_of(unit_brief)
    rehash_contract(body["unit_contract"])
    return body


def example_c0(owned: list[str], example: str) -> dict[str, Any]:
    c0 = explained_c0(owned)
    c0["semantic_content"]["sections"][1]["blocks"][0]["text"] = example
    return c0


TEAM_LEAD_DRAFT = ("Ví dụ minh hoạ: Một trưởng nhóm tổng đài nghe khách phản ánh việc chờ lâu, chưa có thiệt hại, và "
                   "tự xử lý ngay.")
CEO_DRAFT = "Ví dụ minh hoạ: Một CEO nghe khách phản ánh việc chờ lâu, chưa có thiệt hại, và giao tổng đài xử lý ngay."


class UnitTests(StoryboardEndpointTestCase):
    async def test_a_team_lead_example_for_executives_gets_a_targeted_repair(self) -> None:
        body = executive_body()
        owned = owned_of(body)
        writer = writer_with(body, example_c0(owned, TEAM_LEAD_DRAFT))
        fixed = writer_with(body, example_c0(owned, CEO_DRAFT))
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [slot_repair(fixed, 0)], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "validated", "provider")
        self.assertEqual(provider.names, [WRITER, REPAIR, JUDGE])
        self.assertIn('for example "một CEO", "một nhà sáng lập", "một lãnh đạo cấp cao"', provider.calls[0]["prompt"])
        self.assertEqual(provider.calls[1]["thinking_level"], "low")
        self.assertIn("c0 IDM_W5_EXAMPLE_ROLE at c0.semantic_content.sections[1].blocks[0]: the listed drafted example "
                      "is about a person of another level than TARGET_AUDIENCE", provider.calls[1]["prompt"])
        self.assertIn("(roles of TARGET_AUDIENCE: một CEO, một nhà sáng lập, một lãnh đạo cấp cao)",
                      provider.calls[1]["prompt"])
        self.assertIn(CEO_DRAFT, html_texts(data["unit"]["components"][0]))
        self.assertIn("Đã tự sửa: ví dụ chưa đúng vai trò người học.", quality.author_note)
        # Left as it was by the repair: kept for the author with a plain note, never the source copy.
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [slot_repair(writer, 0)], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        quality = self.assert_envelope(data, "provider_validated", "validated", "provider")
        self.assertNotIn("source_locked_fallback", data["unit"]["components"][0])
        self.assertIn("Nên biên tập khối 1 (Lý thuyết): ví dụ minh hoạ kể về một người không thuộc đối tượng học",
                      quality.author_note)

    async def test_long_sentences_get_a_targeted_repair_with_the_measured_numbers(self) -> None:
        body = explain_body()
        owned = owned_of(body)
        long_c0 = explained_c0(owned)
        long_c0["semantic_content"]["sections"][0]["blocks"][0]["text"] = (
            "Mỗi khiếu nại mà bạn tiếp nhận qua tổng đài trong ca làm việc đều thuộc một trong ba cấp độ khác nhau, và "
            "chính cấp độ đó cho bạn biết ai sẽ là người xử lý, cần báo cho ai và cần xử lý nhanh đến mức nào để khách "
            "hàng hài lòng.")
        writer = writer_with(body, long_c0)
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [slot_repair(writer_with(body, explained_c0(owned)), 0)],
                                   JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "validated", "provider")
        self.assertEqual(provider.calls[1]["thinking_level"], "low")
        self.assertIn("c0 IDM_W5_SENTENCE_LENGTH at c0.semantic_content.sections[0].blocks[0]: the listed texts hold "
                      "sentences that are too long", provider.calls[1]["prompt"])
        self.assertIn("(6 prose sentences of 17.8 words on average, the longest 54; at most 22 on average and 40 for "
                      "any sentence)", provider.calls[1]["prompt"])
        self.assertIn("Đã tự sửa: câu quá dài.", quality.author_note)

    async def test_an_older_briefs_sme_question_is_ignored(self) -> None:
        # Owner decision 2026-10-22: the question an older design asked the SME neither reaches the writer nor drops an
        # FAQ item that is about the same point.
        body = escalate_body()
        unit_brief = body["unit_contract"]["idm_unit_brief"]
        unit_brief["faq_material"]["blocks"][0]["sme_questions"] = [
            "Có nên hứa hoàn tiền cho khách VIP trước khi xác minh đơn hàng không?"]
        unit_brief["brief_hash"] = brief_hash_of(unit_brief)
        rehash_contract(body["unit_contract"])
        writer = faq_writer(body, ("Có nên hứa hoàn tiền khi khách vừa gọi đến không?", EXPLAINED),
                            ("Điều gì xảy ra nếu xử lý khiếu nại mà bỏ qua yếu tố truyền thông?", WHAT_IF),
                            ("Vì sao khiếu nại cấp 3 hay bị ghi nhầm thành cấp 2?",
                             WHAT_IF.replace("Khi bỏ", "Nếu bỏ")))
        provider = FakeGenerate(**{WRITER: [writer], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "validated", "provider")
        self.assertEqual(provider.names, [WRITER, JUDGE])
        self.assertNotIn("Có nên hứa hoàn tiền cho khách VIP trước khi xác minh", provider.calls[0]["prompt"])
        self.assertNotIn("open_questions", provider.calls[0]["prompt"])
        self.assertEqual(len(data["unit"]["components"][1]["items"]), 3)
        self.assertNotIn("chờ SME", quality.author_note)


class FrameworkBriefTests(unittest.TestCase):
    def test_server_briefs_are_known_plan_text(self) -> None:
        self.assertTrue(is_framework_brief("Liệt kê đủ 5 chuyển dịch: SHIFT 1: THINK BIG"))
        self.assertTrue(is_framework_brief("Ý chính của từng chuyển dịch: (1) Từ tư duy tồn tại sang khát vọng lớn"))
        self.assertTrue(is_framework_brief("Main idea of each shift: (1) From surviving to ambition"))
        self.assertFalse(is_framework_brief("[AI soạn — tác giả kiểm tra lại] Ví dụ minh hoạ ngắn"))



class OrientationUnitTests(StoryboardEndpointTestCase):
    async def test_an_explained_orientation_unit_names_the_items_without_a_new_fact_finding(self) -> None:
        # c2.l1.u1 of the run: the five names came from the server's support brief, not from the unit's facts, and
        # the explain-mode check called "SHIFT 4", "SHIFT 5" new facts. The briefs are plan text now.
        body = severity_body()
        body["unit_contract"]["unit_title"] = "Bản đồ 5 chuyển dịch"
        unit_brief = body["unit_contract"]["idm_unit_brief"]
        unit_brief.update(fidelity_mode="explain", purpose_profile="soft_skills",
                          purpose_summary="Giúp lãnh đạo tái định hình tư duy quản trị.")
        unit_brief["components"][0]["support_items"] = [
            {"kind": "explain_concept", "brief": SHIFT_BRIEF, "block_id": "cb_0005"},
            {"kind": "explain_concept", "brief": SHIFT_DETAIL, "block_id": "cb_0005"}]
        unit_brief["brief_hash"] = brief_hash_of(unit_brief)
        rehash_contract(body["unit_contract"])
        c0 = explained_c0(owned_of(body))
        c0["semantic_content"]["sections"][0]["blocks"].append(
            block("bullets", items=[head.split(" — ")[0] for head in SHIFT_HEADS]))
        provider = FakeGenerate(**{WRITER: [writer_with(body, c0)], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "validated", "provider")
        self.assertEqual(provider.names, [WRITER, JUDGE])
        prompt = provider.calls[0]["prompt"]
        self.assertIn(json.dumps(SHIFT_DETAIL, ensure_ascii=False)[1:-1], prompt)
        self.assertIn('one short plain line on what it changes or is about (the support item that starts with "Ý chính '
                      'của từng" / "Main', prompt)
        self.assertNotIn("IDM_W5_NEW_FACT", quality.deterministic_codes)
