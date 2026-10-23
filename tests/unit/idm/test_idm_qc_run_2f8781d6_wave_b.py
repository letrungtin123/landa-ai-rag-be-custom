"""Fixes from the QC report of the real AI ID run 2f8781d6 (AI 7f67137, 2026-10-23, 7,1/10), wave B (B6-B9).

Every case below is synthetic text with the shape of a defect of that run (no course text of the run); every provider
is a local fake.
"""

from __future__ import annotations

import unittest
from typing import Any

from app.idm import prompts_unit
from app.idm.answer_near import ANSWER_ITEM_LENGTH_CODE, ANSWER_NEAR_EXAMPLE_CODE, item_length_cue, near_example
from app.idm.prompts_course import w1_section_prompt
from app.idm.qa import STYLE_CODES, JudgeOutcome, build_unit_author_note
from app.idm.qa_checks import advisory_slot_findings
from app.idm.slot_words import classroom_words, exercise_details
from app.idm.source_names import NAME_NOT_IN_SOURCE_CODE, NameGuide
from app.idm.storyboard import IdmUnitWriter, parse_brief
from app.idm.storyboard_context import _REVIEW_FIRST_CODES, _REVIEW_ONLY_CODES, _TARGETED_REPAIR_CODES
from app.idm.storyboard_repair import SECOND_REPAIR_CODES
from app.schemas.orchestration_v2 import RagLessonAuthorUnitV2Request
from app.services.orchestration_v2 import unit as unit_service
from tests.support.idm_test_support import make_runtime, rehash_contract
from tests.support.log_support import logged_events
from tests.unit.idm.test_idm_qc_run_2f8781d6 import writer_prompt_of
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


class LearnerRoleTests(unittest.TestCase):
    """B7: quiz and worksheet scenarios put the learner in the audience's role; other actors are only the customer or a
    stakeholder; every server-chosen opening puts the learner in the case."""

    def test_the_writer_is_told_and_the_openings_follow(self) -> None:
        prompt = writer_prompt_of(escalate_body())
        self.assertIn(prompts_unit._ROLE_RULE, prompt)
        self.assertIn("in the role TARGET_AUDIENCE names, in the learner's own\n  business setting", prompt)
        self.assertIn("appear only as the\n  customer or a stakeholder", prompt)
        self.assertIn("the learner is the one who acts or decides", prompt)
        self.assertNotIn('"Ban điều\n  hành của …"', prompt)
        openings = {prompts_unit.example_opening(f"chapter_{c}.lesson_{l}.unit_{u}", locale)
                    for c in range(1, 4) for l in range(1, 5) for u in range(1, 4) for locale in ("vi", "en")}  # noqa: E741
        for opening in openings:
            self.assertFalse(any(word in opening for word in ("Ban điều hành", "CEO", "quý vừa qua", "leadership",
                                                              "Last quarter")), opening)
            # Review 5 (L4), review 6 (L8): every opening is a supposition with the learner in it, in several forms.
            self.assertTrue(opening.startswith(("Giả sử", "Hãy hình dung", "Nếu", "Trong vai trò của mình, nếu",
                                                "Hãy đặt mình", "Suppose", "Imagine", "If", "In your role, if",
                                                "Picture yourself")), opening)
            self.assertTrue(" bạn" in opening or " you" in opening.replace("your", "you"), opening)
        self.assertGreaterEqual(len(openings), 8)


EXERCISE_FACTS = [
    "BÀI TẬP 10 phút",
    "Thực hành Bộ lọc: các nhóm được phát 5 yêu cầu của khách hàng và chỉ chọn ra đúng 2 sáng kiến khả thi nhất.",
    "Câu hỏi thảo luận 5 phút: đội của bạn đang phân nhóm khách hàng thế nào?",
    "Bài tập về nhà theo nhóm: hoàn thiện bản đồ hành trình cho dự án của bạn.",
    "Nhân viên phản hồi khiếu nại của khách hàng trong 30 phút.",
]


class ClassroomExerciseTests(unittest.TestCase):
    """B8: a classroom exercise of the source becomes a self-paced practice; what still carries its time limit, its
    count or its group or homework wording outside a worksheet is an edit note, never a failure."""

    def test_only_the_exercises_details_count(self) -> None:
        details = exercise_details(EXERCISE_FACTS)
        self.assertEqual(details, ["10 phút", "đúng 2", "5 phút", "về nhà", "theo nhóm"])
        self.assertNotIn("30 phút", details)  # a rule of the work, not of an exercise
        teaching = html(("Bộ lọc dự án dùng thế nào?", [paragraph(
            "Đội ngũ cần 10 phút phản biện và chọn đúng 2 sáng kiến. Nhân viên phản hồi khách trong 30 phút.")]))
        worksheet = html(("Nhiệm vụ", [{"kind": "task", "text": "Trong 10 phút, chọn đúng 2 sáng kiến.",
                                         "items": [], "rows": []}]))
        self.assertEqual(classroom_words([teaching, worksheet], {1}, details), {0: ["10 phút", "đúng 2"]})
        self.assertEqual(classroom_words([teaching], (), []), {})

    def test_the_note_names_them(self) -> None:
        note = build_unit_author_note(locale="vi", judge=JudgeOutcome("pass"), deterministic_codes=[],
                                      fallback_slots=[], ai_drafted=False, slot_types=["html", "la_faq"],
                                      classroom={0: ["10 phút", "đúng 2"], 1: ["theo nhóm"]})
        self.assertEqual(note, "Nên biên tập khối 1 (Lý thuyết), khối 2 (Hỏi đáp): còn chi tiết của bài tập trên lớp "
                               "trong tài liệu («10 phút», «đúng 2», «theo nhóm») — nên viết thành bài tự luyện trên "
                               "công ty của người học, bỏ thời lượng, số lượng phải chọn và cách làm theo nhóm hoặc ở "
                               "nhà.")

    def test_the_writer_reads_the_unit_s_exercises(self) -> None:
        body = escalate_body()
        facts = body["unit_contract"]["source_facts"]
        facts[0]["fact_text"] = EXERCISE_FACTS[1]
        rehash_contract(body["unit_contract"])
        request = RagLessonAuthorUnitV2Request.model_validate(body)
        writer = IdmUnitWriter(request.unit_contract, parse_brief(request.unit_contract),
                               unit_service._idm_unit_deps(request), make_runtime(None))
        components = escalate_writer(body)["components"]
        faq = {**components["c1"], "type": "la_faq", "items": [
            {"question": "Vì sao chỉ chọn hai sáng kiến?", "answer": "Vì bài tập yêu cầu chọn đúng 2 sáng kiến."},
            *components["c1"]["items"][1:]]}
        notes = writer.slot_notes([{**components["c0"], "type": "problem"}, faq], [], whole_fallback=False)
        self.assertEqual(notes["classroom"], {1: ["đúng 2"]})

    def test_the_prompts_say_so(self) -> None:
        self.assertIn("is practice_material and its summary says it is a\n     classroom exercise",
                      w1_section_prompt("vi", project_context={}, section_id="s1", section_title_path="T",
                                        facts=[("k1", "Một ý.", None)]))
        self.assertIn(prompts_unit._SELF_PACED_RULE, writer_prompt_of(escalate_body()))


class ClassroomEndpointTests(StoryboardEndpointTestCase):
    """B8 end to end: the note line reaches the unit; nothing is repaired or replaced for it."""

    async def test_the_note_line_reaches_the_unit(self) -> None:
        body = escalate_body()
        body["unit_contract"]["source_facts"][0]["fact_text"] = EXERCISE_FACTS[1]
        rehash_contract(body["unit_contract"])
        writer = escalate_writer(body)
        writer["components"]["c0"]["explanation"] += " Nhóm chỉ chọn đúng 2 sáng kiến."
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [slot_repair(writer, 0)] * 2, JUDGE: [judge(), judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200, data.get("detail"))
        self.assertIn("còn chi tiết của bài tập trên lớp trong tài liệu («đúng 2»)",
                      data["unit"]["idm_quality"]["author_note"])
        self.assertNotIn("source_locked_fallback", data["unit"]["components"][0])


# Synthetic source lines with the shapes of the run's tool names (the PDF glues words: "Bản đồThấu cảm").
NAME_SOURCE = [
    "Bản đồThấu cảm (Empathy Map) giúp hiểu điều khách hàng nghĩ và cảm nhận.",
    "Hồ sơ khách hàng B2B gồm công việc, nỗi đau và lợi ích của từng vai trò.",
    "Bản đồ hành trình khách hàng (Customer Journey Map) gồm 5 bước.",
    "Bộlọc Dự án giúp chọn sáng kiến khả thi.",
    "Ma trận ưu tiên xếp sáng kiến theo hai trục.",
    "Service Blueprint chia dịch vụ thành ba tầng.",
    "G-STIC: Goal, Strategy, Tactics, Implementation, Control.",
]


class SourceNameTests(unittest.TestCase):
    """B9: a tool named in a title or a heading by words the source does not use is caught; the source's own names,
    glued or not, plain words and time frames are not."""

    def test_the_run_s_titles(self) -> None:
        names = NameGuide.build(NAME_SOURCE)
        self.assertEqual(names.missing("Ma trận thấu cảm: công việc, nỗi đau và lợi ích"), ["Ma trận thấu cảm"])
        self.assertEqual(names.missing("Ma trận thấu cảm (công việc, nỗi đau)"), ["Ma trận thấu cảm"])
        self.assertEqual(names.missing("Value Proposition Canvas cho khách hàng"), ["Value Proposition Canvas"])
        # Words the source does not write after a tool noun: a new name.
        self.assertEqual(names.missing("Ma trận năng lực ba tầng"), ["Ma trận năng lực"])
        self.assertEqual(names.missing("Bản đồ ưu tiên"), ["Bản đồ ưu tiên"])  # the source's "ma trận ưu tiên"
        # Words the source writes after a tool noun no other tool takes: a plain description.
        for title in ("Sơ đồ dịch vụ ba tầng", "Bản đồ hành trình 4 tầng", "Lập bản đồ hành trình khách hàng",
                      "Bộ lọc Dự án chọn sáng kiến", "Ma trận ưu tiên hai trục",
                      "Sơ đồ Service Blueprint ba tầng", "Sơ đồ dịch vụ (Service Blueprint) ba tầng",
                      "Khung quản trị G-STIC cho dự án", "G-STIC là gì?", "Bản đồ thấu cảm khách hàng",
                      "Customer Journey Map", "Khung thời gian 90 ngày", "Mô hình đóng gói giải pháp",
                      "Khách hàng B2B cần gì?", ""):
            self.assertEqual(names.missing(title), [], title)

    def test_a_slot_s_title_and_headings(self) -> None:
        names = NameGuide.build(NAME_SOURCE)
        component = {**html(("Ma trận thấu cảm gồm những tầng nào?", [paragraph("Ba tầng.")]),
                            ("Bản đồ hành trình dùng khi nào?", [paragraph("Khi cần.")]),
                            ("Khi nào dùng ma trận thấu cảm?", [paragraph("Khi cần.")])),
                     "title": "Value Proposition Canvas"}
        self.assertEqual(names.slot_names(component), ["Value Proposition Canvas", "Ma trận thấu cảm"])
        self.assertEqual(names.slot_names(quiz("Ma trận thấu cảm có mấy tầng?", ["Ba tầng", "Hai tầng"])), [])

    def test_it_is_soft(self) -> None:
        # Repaired once at low thinking, then an edit note; never a fallback, never a second repair.
        self.assertIn(NAME_NOT_IN_SOURCE_CODE, _TARGETED_REPAIR_CODES)
        self.assertIn(NAME_NOT_IN_SOURCE_CODE, STYLE_CODES)
        self.assertIn(NAME_NOT_IN_SOURCE_CODE, _REVIEW_FIRST_CODES)
        self.assertIn(NAME_NOT_IN_SOURCE_CODE, _REVIEW_ONLY_CODES)
        self.assertNotIn(NAME_NOT_IN_SOURCE_CODE, SECOND_REPAIR_CODES)
        self.assertIn(NAME_NOT_IN_SOURCE_CODE, prompts_unit.UNIT_RULE_TEXT)

    def test_the_title_line(self) -> None:
        for locale, line in (("vi", "Tên bài hoặc tên bài học dùng «Ma trận thấu cảm», tên này không có trong tài "
                                    "liệu — nên đổi theo tên tài liệu dùng."),
                             ("en", "The unit or lesson title uses «Ma trận thấu cảm», a name the source does not use "
                                    "— rename it as the source does.")):
            note = build_unit_author_note(locale=locale, judge=JudgeOutcome("pass"), deterministic_codes=[],
                                          fallback_slots=[], ai_drafted=False, slot_types=["html", "problem"],
                                          title_names=["Ma trận thấu cảm", "Ma trận thấu cảm"])
            self.assertEqual(note, line)
        whole = build_unit_author_note(locale="vi", judge=JudgeOutcome("pass"), deterministic_codes=[],
                                       fallback_slots=[0, 1], ai_drafted=False, slot_types=["html", "problem"],
                                       whole_fallback=True, title_names=["Ma trận thấu cảm"])
        self.assertNotIn("Ma trận thấu cảm", whole)

    def test_the_writer_reads_the_unit_s_titles(self) -> None:
        body = severity_body()
        body["unit_contract"]["unit_title"] = "Ma trận rủi ro của khiếu nại"
        rehash_contract(body["unit_contract"])
        request = RagLessonAuthorUnitV2Request.model_validate(body)
        writer = IdmUnitWriter(request.unit_contract, parse_brief(request.unit_contract),
                               unit_service._idm_unit_deps(request), make_runtime(None))
        components = severity_writer(body)["components"]
        renamed = {**components["c0"], "type": "html", "title": "Ma trận khẩn cấp khiếu nại"}
        # The facts' "cấp độ" after a tool noun is a description, never a finding.
        self.assertEqual(writer.names.missing("Ma trận cấp độ khiếu nại"), [])
        notes = writer.slot_notes([renamed, {**components["c1"], "type": "problem"}], [], whole_fallback=False)
        self.assertEqual(notes["title_names"], ["Ma trận rủi ro"])
        self.assertEqual(notes["name_words"], {0: ["Ma trận khẩn cấp"]})
        self.assertEqual(writer.slot_notes([renamed, components["c1"]], [0], whole_fallback=False)["name_words"], {})
        self.assertEqual(writer.slot_hint(NAME_NOT_IN_SOURCE_CODE, renamed), ' (new: "Ma trận khẩn cấp")')


class SourceNameEndpointTests(StoryboardEndpointTestCase):
    """B9 end to end: one targeted repair with the rename hint; a name the repair keeps stays the AI's, with an edit
    note naming it; a renamed slot is settled."""

    @staticmethod
    def renamed_writer(body: dict[str, Any]) -> dict[str, Any]:
        writer = severity_writer(body)
        writer["components"]["c0"]["title"] = "Ma trận khẩn cấp khiếu nại"
        return writer

    async def test_repaired_once_then_an_edit_note(self) -> None:
        body = severity_body()
        writer = self.renamed_writer(body)
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [slot_repair(writer, 0)], JUDGE: [judge(), judge()]})
        with self.assertLogs("app.idm", level="INFO") as logs:
            status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200, data.get("detail"))
        self.assertEqual(provider.names.count(REPAIR), 1)
        self.assertEqual((data["content_origin"], data["quality_state"]), ("provider_validated", "validated"))
        self.assertNotIn("source_locked_fallback", data["unit"]["components"][0])
        quality = logged_events(logs.records, "idm_unit_quality")[-1]
        self.assertIn(NAME_NOT_IN_SOURCE_CODE, quality["remaining_codes"])
        note = data["unit"]["idm_quality"]["author_note"]
        self.assertIn("Nên biên tập khối 1 (Lý thuyết): tên («Ma trận khẩn cấp») không có trong tài liệu — nên dùng "
                      "đúng tên tài liệu gọi công cụ, mô hình hoặc bảng đó", note)
        self.assertNotIn("IDM_", note)
        repair_prompt = next(call["prompt"] for call in provider.calls if call["family"] == REPAIR)
        self.assertIn("IDM_W5_NAME_NOT_IN_SOURCE", repair_prompt)
        self.assertIn('(new: "Ma trận khẩn cấp")', repair_prompt)
        self.assertIn("give it the name the facts use", repair_prompt)

    async def test_a_renamed_slot_is_settled(self) -> None:
        body = severity_body()
        writer = self.renamed_writer(body)
        fixed = slot_repair(writer, 0, title="Dấu hiệu của từng cấp độ")
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [fixed], JUDGE: [judge(), judge()]})
        with self.assertLogs("app.idm", level="INFO") as logs:
            status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200, data.get("detail"))
        self.assertEqual(data["unit"]["components"][0]["title"], "Dấu hiệu của từng cấp độ")
        quality = logged_events(logs.records, "idm_unit_quality")[-1]
        self.assertNotIn(NAME_NOT_IN_SOURCE_CODE, quality["remaining_codes"])
        note = data["unit"]["idm_quality"]["author_note"]
        self.assertIn("Đã tự sửa: tên công cụ không có trong tài liệu.", note)
        self.assertNotIn("Nên biên tập khối 1", note)


if __name__ == "__main__":
    unittest.main()
