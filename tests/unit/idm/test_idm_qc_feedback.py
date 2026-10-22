"""QC feedback of 2026-10-21 (owner-approved change list A1-A7): the unit-level rules of the AI service.

* A1/A2 (contract K1): the stem and the explanation of a question are laid out in lines without changing a word; what
  the layout cannot fix (a principle that restates three definitions, empty praise instead of a reason, a verdict that
  contradicts the key, a stem of four sentences) is a code the writer repairs once, then a plain note.
* K1: every line feed of the question and the explanation reaches the unit Node receives, and its problem XML.
* A3: a callout that only repeats its slot's paragraphs, lists or tables is removed (its text stays there).
* A4: the description of a suggested infographic or video says what to draw (layout, text, one element per point);
  a thin one is completed from the content points and noted.
* Owner decisions 2026-10-22: no question for an SME; a contradictory, unclear or out-of-date source point marks its
  unit "Cần xem" with a sentence quoting the facts; the fallback question explains itself line by line.
* A5 (no Hold): a worksheet whose practice criteria the AI suggested from what the lesson teaches says so in the unit
  note; since run 5496155c its self-check reads "Tự kiểm tra" for the learner, never "(tiêu chí gợi ý)".

Every provider is a local fake.
"""

from __future__ import annotations

import unittest
from collections import Counter
from typing import Any

from app.contracts.orchestration_v2.blueprint import ArchitectureMediaBriefV2
from app.idm.artifacts import clean_text
from app.idm.contracts import IdmIssueV1, brief_hash_of
from app.idm.mcq import normalize_single_choice
from app.idm.media_detail import MEDIA_BRIEF_COMPLETED_CODE, complete_media_brief, media_brief_gaps
from app.idm.node_acceptance_problems import _problem_xml, workspace_problem_code
from app.idm.qa_checks import repeated_callouts, without_blocks
from app.idm.quiz_text import (
    EXPLANATION_FILLER_CODE,
    EXPLANATION_LAYOUT_CODE,
    EXPLANATION_LENGTH_CODE,
    EXPLANATION_RESTATES_CODE,
    EXPLANATION_VERDICT_CODE,
    QUESTION_LAYOUT_CODE,
    QUESTION_LENGTH_CODE,
    explanation_report,
    filler_phrases,
    lay_out_quiz,
    layout_explanation,
    layout_question,
    question_report,
    quiz_findings,
    quiz_hint,
)
from app.idm.source_flags import SourceFact, flag_sentence, flag_sentences
from app.idm.text import evidence_index
from app.idm.worksheet import plain_self_check
from tests.idm_golden import key
from tests.support.idm_test_support import rehash_contract
from tests.unit.idm.test_idm_module_design import design_module, edited
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
    worksheet_body,
    worksheet_writer,
)

SUGGESTED = "Tiêu chí gợi ý: cấp độ theo dấu hiệu bài học nêu cho từng cấp"
# The stem the QC department showed (2026-10-21), its list written inline.
WALL_STEM = ("Doanh nghiệp A đang phân bổ ngân sách cho ba phân khúc khách hàng như sau: - Phân khúc X: doanh thu 60%, "
             "chi phí phục vụ cao. - Phân khúc Y: doanh thu 30%, tăng trưởng nhanh. - Phân khúc Z: doanh thu 10%, chi "
             "phí thấp. Theo quy tắc tập trung nguồn lực, bạn nên đưa ra quyết định nào?")
# The rambling explanation of run ab8d67e1: three definitions before "A - …", all in one paragraph.
RAMBLING = ("Nguyên lý cốt lõi khẳng định: Công cụ quản trị không có lỗi. Công nghệ số không có lỗi. Sự thất bại "
            "của các dự án chuyển đổi bắt nguồn từ hệ điều hành tư duy cũ kỹ. A - Đúng vì khi móng tư duy không đổi, "
            "mọi công cụ trở nên rời rạc. B - Sai vì công cụ và công nghệ không có lỗi. C - Sai vì ngân sách không "
            "giải quyết được gốc rễ.")
LINES = ("Tiêu chí: khách hàng phàn nàn lần thứ hai là dấu hiệu của cấp 2.\n"
         "A - Sai vì cấp 1 chỉ dành cho lần đầu, chưa có thiệt hại tài chính.\n"
         "B - Đúng vì phàn nàn lần thứ hai là dấu hiệu bảng nêu cho cấp 2.\n"
         "C - Sai vì không có yếu tố an toàn, pháp lý hay truyền thông.")


def choices(correct: int = 1, count: int = 3) -> list[dict[str, Any]]:
    texts = ["Cấp 1 vì chưa có thiệt hại tài chính", "Cấp 2 vì khách hàng phàn nàn lần thứ hai",
             "Cấp 3 vì cần escalate ngay cho quản lý", "Không cần thông báo cho ai"]
    return [{"text": text, "correct": index == correct} for index, text in enumerate(texts[:count])]


def problem(explanation: str = LINES, question: str = "Khách gọi lần thứ hai. Khiếu nại thuộc cấp nào?",
            **changes: Any) -> dict[str, Any]:
    return {"type": "problem", "problem_type": "multiple_choice", "question": question, "choices": choices(),
            "explanation": explanation, **changes}


def words(text: str) -> list[str]:
    """The words of ``text`` without list markers and option separators (the layout must keep every word)."""

    return [word for word in text.split() if word not in {"-", "—", "–"}]  # noqa: RUF001 - dashes


class QuizLayoutTests(unittest.TestCase):
    def test_an_inline_list_and_the_question_get_their_own_lines(self) -> None:
        laid = layout_question(WALL_STEM)
        self.assertEqual(laid.split("\n"), [
            "Doanh nghiệp A đang phân bổ ngân sách cho ba phân khúc khách hàng như sau:",
            "- Phân khúc X: doanh thu 60%, chi phí phục vụ cao.",
            "- Phân khúc Y: doanh thu 30%, tăng trưởng nhanh.",
            "- Phân khúc Z: doanh thu 10%, chi phí thấp.",
            "Theo quy tắc tập trung nguồn lực, bạn nên đưa ra quyết định nào?"])
        self.assertEqual(words(laid), words(WALL_STEM))
        self.assertEqual(question_report({"question": laid}), {})
        self.assertEqual(layout_question(laid), laid)
        # A dash inside a sentence is no list; a one-sentence question stays one line.
        prose = "Tỷ lệ chi phí - lợi ích của dự án là bao nhiêu khi doanh thu 2019 - 2020 tăng?"
        self.assertEqual(layout_question(prose), prose)
        self.assertEqual(layout_question("Bối cảnh ngắn.\n• Ý một\n* Ý hai\nCâu hỏi nào đúng?"),
                         "Bối cảnh ngắn.\n- Ý một\n- Ý hai\nCâu hỏi nào đúng?")

    def test_a_paragraph_explanation_gets_one_line_per_option(self) -> None:
        paragraph = ("Tiêu chí: khách hàng phàn nàn lần thứ hai là dấu hiệu của cấp 2. A — sai vì lần phàn nàn thứ hai "
                     "đã vượt cấp 1; B — đúng vì khớp dấu hiệu cấp 2 trong bảng; C — sai vì không có yếu tố an toàn.")
        laid = layout_explanation(paragraph, 3)
        self.assertEqual(laid.split("\n"), [
            "Tiêu chí: khách hàng phàn nàn lần thứ hai là dấu hiệu của cấp 2.",
            "A - sai vì lần phàn nàn thứ hai đã vượt cấp 1.", "B - đúng vì khớp dấu hiệu cấp 2 trong bảng.",
            "C - sai vì không có yếu tố an toàn."])
        self.assertEqual([word.strip(";.") for word in words(laid)], [word.strip(";.") for word in words(paragraph)])
        self.assertEqual(layout_explanation(laid, 3), laid)
        # Options named out of order, one sentence each, are put in letter order.
        shuffled = "Tiêu chí X. B - Sai vì y. A - Đúng vì x. C - Sai vì z."
        self.assertEqual(layout_explanation(shuffled, 3),
                         "Tiêu chí X.\nA - Đúng vì x.\nB - Sai vì y.\nC - Sai vì z.")
        # An option missing: nothing is guessed.
        self.assertEqual(layout_explanation("Tiêu chí X. A - Đúng vì x. C - Sai vì z.", 3),
                         "Tiêu chí X. A - Đúng vì x. C - Sai vì z.")

    def test_the_layout_follows_the_shuffled_letters(self) -> None:
        component = problem(choices=choices(correct=0))
        component["explanation"] = LINES.replace("A - Sai vì", "A - Đúng vì").replace("B - Đúng vì", "B - Sai vì")
        shuffled, codes = normalize_single_choice(component, "cp2_" + "a" * 32, 2)
        laid, laid_codes = lay_out_quiz(shuffled)
        self.assertEqual(codes["w5_problem_explanation_relabelled"], 1)
        lines = laid["explanation"].split("\n")
        self.assertEqual([line[:4] for line in lines[1:]], ["A - ", "B - ", "C - "])
        key = next(index for index, choice in enumerate(laid["choices"]) if choice["correct"])
        self.assertEqual(key, 2)
        self.assertTrue(lines[key + 1].startswith("C - Đúng vì cấp 1 chỉ dành"))
        self.assertEqual(quiz_findings(laid), [])
        self.assertEqual(laid_codes["w5_problem_question_laid_out"], 1)

    def test_a_word_that_ends_one_line_and_starts_the_next_is_no_repeat(self) -> None:
        fixes: Counter[str] = Counter()
        self.assertEqual(clean_text("Nhóm khách hàng\nhàng đầu", frozenset(), fixes), "Nhóm khách hàng\nhàng đầu")
        self.assertEqual(clean_text("khách khách hàng", frozenset(), fixes), "khách hàng")


class QuizCheckTests(unittest.TestCase):
    def test_the_explanation_of_lines_passes(self) -> None:
        self.assertEqual(quiz_findings(problem()), [])

    def test_the_qc_examples_are_found(self) -> None:
        rambling = explanation_report(problem(layout_explanation(RAMBLING, 3), choices=choices(correct=0)))
        self.assertEqual(list(rambling), [EXPLANATION_LENGTH_CODE])
        self.assertIn("3 sentence(s)", rambling[EXPLANATION_LENGTH_CODE][0])
        hollow = LINES.replace("B - Đúng vì phàn nàn lần thứ hai là dấu hiệu bảng nêu cho cấp 2.",
                               "B - Đúng vì phương án này dũng cảm Say No để tạo điểm khác biệt tuyệt đối.")
        report = explanation_report(problem(hollow))
        self.assertEqual(report[EXPLANATION_FILLER_CODE], ["dũng cảm", "say no", "khác biệt tuyệt đối"])
        self.assertEqual(quiz_hint(EXPLANATION_FILLER_CODE, problem(hollow)),
                         ' (remove: "dũng cảm", "say no", "khác biệt tuyệt đối")')

    def test_verdict_reason_and_restating(self) -> None:
        wrong_key = LINES.replace("A - Sai vì", "A - Đúng vì")
        self.assertEqual(explanation_report(problem(wrong_key))[EXPLANATION_VERDICT_CODE],
                         ["the verdict of A contradicts the answer key"])
        no_reason = LINES.replace("C - Sai vì không có", "C - Không có")
        self.assertEqual(explanation_report(problem(no_reason))[EXPLANATION_LAYOUT_CODE],
                         ["no verdict and reason on the lines of C"])
        restating = LINES.replace("B - Đúng vì phàn nàn lần thứ hai là dấu hiệu bảng nêu cho cấp 2.",
                                  "B - Đúng vì cấp 2 vì khách hàng phàn nàn lần thứ hai.")
        self.assertIn(EXPLANATION_RESTATES_CODE, explanation_report(problem(restating)))
        english = ("Criterion: a second complaint is a sign of level 2.\nA - Incorrect because level 1 is a first call."
                   "\nB - Correct because the table lists a second complaint for level 2.\nC - Wrong because nothing "
                   "touches safety.")
        self.assertEqual(explanation_report(problem(english)), {})
        no_principle = "A - Sai vì x.\nB - Đúng vì y.\nC - Sai vì z."
        self.assertIn(EXPLANATION_LAYOUT_CODE, explanation_report(problem(no_principle)))

    def test_a_phrase_the_facts_use_is_no_filler(self) -> None:
        rule = "B - Đúng vì quy định tuyệt đối không hứa bồi thường trước khi phân loại."
        self.assertEqual(filler_phrases(rule), ["tuyệt đối"])
        self.assertEqual(filler_phrases(rule, evidence_index(["Tuyệt đối không hứa bồi thường trước."])), [])

    def test_the_stem_layout_and_length(self) -> None:
        four = "Câu một. Câu hai. Câu ba.\nCâu hỏi nào?"
        self.assertEqual(question_report({"question": four})[QUESTION_LAYOUT_CODE],
                         ["3 context sentences (at most 2)"])
        long = " ".join(["từ"] * 130) + " là gì?"
        self.assertIn("132 words of context and question", question_report({"question": long})[QUESTION_LENGTH_CODE][0])
        self.assertEqual(quiz_findings(problem(question=WALL_STEM)), [])
        self.assertEqual(quiz_findings({**problem(), "problem_type": "numerical"}), [])


class LineFeedEndToEndTests(StoryboardEndpointTestCase):
    async def test_line_feeds_reach_the_unit_and_its_problem_xml(self) -> None:
        body = severity_body()
        writer = severity_writer(body, explanation=LINES)
        writer["components"]["c1"]["question"] = ("Tại một doanh nghiệp, khách hàng gọi lần thứ hai như sau:\n"
                                                  "- Đơn hàng giao trễ\n- Chưa có thiệt hại tài chính\n"
                                                  "Khiếu nại này thuộc cấp độ nào?")
        provider = FakeGenerate(**{WRITER: [writer], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        self.assertEqual((data["content_origin"], data["quality_state"]), ("provider_validated", "validated"))
        problem_slot = data["unit"]["components"][1]
        self.assertEqual(problem_slot["question"].split("\n"), [
            "Tại một doanh nghiệp, khách hàng gọi lần thứ hai như sau:", "- Đơn hàng giao trễ",
            "- Chưa có thiệt hại tài chính", "Khiếu nại này thuộc cấp độ nào?"])
        self.assertEqual(len(problem_slot["explanation"].split("\n")), 4)
        xml = _problem_xml(problem_slot)
        assert xml is not None
        self.assertIn("<label>Tại một doanh nghiệp, khách hàng gọi lần thứ hai như sau:\n- Đơn hàng giao trễ\n", xml)
        self.assertIn("<p>Tiêu chí: khách hàng phàn nàn lần thứ hai là dấu hiệu của cấp 2.\n", xml)
        self.assertIsNone(workspace_problem_code(xml))
        self.assertIn("QUIZ TEXT of a single-choice slot", provider.calls[0]["prompt"])

    async def test_a_hollow_explanation_gets_one_targeted_repair(self) -> None:
        body = severity_body()
        hollow = LINES.replace("B - Đúng vì phàn nàn lần thứ hai là dấu hiệu bảng nêu cho cấp 2.",
                               "B - Đúng vì đây là lựa chọn dũng cảm, tạo điểm khác biệt tuyệt đối.")
        writer = severity_writer(body, explanation=hollow)
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [slot_repair(writer, 1, explanation=LINES)],
                                   JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        self.assertEqual(provider.names, [WRITER, REPAIR, JUDGE])
        self.assertEqual(provider.calls[1]["thinking_level"], "low")
        self.assertIn('c1 IDM_W5_EXPLANATION_FILLER at c1.explanation: replace the empty praise or slogan quoted below '
                      'with the concrete reason the unit teaches (the rule, criterion or step the option meets or '
                      'breaks) (remove: "dũng cảm", "khác biệt tuyệt đối")', provider.calls[1]["prompt"])
        note = data["unit"]["idm_quality"]["author_note"]
        self.assertIn("Đã tự sửa: lời giải dùng chữ sáo rỗng.", note)
        self.assertEqual(data["quality_state"], "validated")

    async def test_a_verdict_the_repair_cannot_fix_is_a_plain_review_note(self) -> None:
        body = severity_body()
        wrong = LINES.replace("A - Sai vì", "A - Đúng vì")
        writer = severity_writer(body, explanation=wrong)
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [slot_repair(writer, 1)], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        self.assertEqual(data["quality_state"], "review_required")
        note = data["unit"]["idm_quality"]["author_note"]
        self.assertIn("lời giải ghi một phương án là đúng hoặc sai ngược với đáp án — cần sửa cho khớp đáp án", note)
        self.assertNotIn("IDM_W5", note)
        self.assertIn("IDM_W5_EXPLANATION_VERDICT", data["unit"]["idm_quality"]["deterministic_codes"])


def suggested_worksheet_body() -> dict[str, Any]:
    """The worksheet unit whose practice has suggested criteria (its feedback criterion carries the label)."""

    body = worksheet_body()
    brief = body["unit_contract"]["idm_unit_brief"]
    for slot in brief["components"]:
        if slot.get("practice"):
            slot["practice"]["feedback_focus"]["criterion"] = SUGGESTED
    brief["brief_hash"] = brief_hash_of(brief)
    rehash_contract(body["unit_contract"])
    return body


class SuggestedCriteriaTests(StoryboardEndpointTestCase):
    async def test_suggested_criteria_are_said_in_the_note_only(self) -> None:
        body = suggested_worksheet_body()
        provider = FakeGenerate(**{WRITER: [worksheet_writer(body)], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        sections = data["unit"]["components"][0]["semantic_content"]["sections"]
        self.assertEqual(sections[-1]["heading"], "Tự kiểm tra bảng của bạn")
        self.assertIn('the heading stays "Tự kiểm tra" / "Self-check"', provider.calls[0]["prompt"])
        note = data["unit"]["idm_quality"]["author_note"]
        self.assertIn("phần Tự kiểm tra dùng tiêu chí AI gợi ý từ nội dung đã dạy — tác giả nên xem lại.", note)

    async def test_source_criteria_keep_their_heading_and_note(self) -> None:
        body = worksheet_body()
        provider = FakeGenerate(**{WRITER: [worksheet_writer(body)], JUDGE: [judge()]})
        _status, data, _ = await self.post(body, provider)
        sections = data["unit"]["components"][0]["semantic_content"]["sections"]
        self.assertEqual(sections[-1]["heading"], "Tự kiểm tra bảng của bạn")
        self.assertNotIn("tiêu chí gợi ý", data["unit"]["idm_quality"]["author_note"])

    def test_a_plain_self_check_is_left_alone(self) -> None:
        semantic = {"sections": [{"heading": "Nhiệm vụ", "blocks": [{"kind": "task", "text": "Làm."}]},
                                 {"heading": "Tự kiểm tra", "blocks": [{"kind": "bullets", "items": ["a", "b"]}]}]}
        self.assertIsNone(plain_self_check(semantic, "vi"))
        self.assertIsNone(plain_self_check({"sections": [{"heading": "a", "blocks": []}]}, "vi"))


def section(*blocks: dict[str, Any], heading: str = "Khiếu nại thuộc cấp độ nào?") -> dict[str, Any]:
    return {"heading": heading, "blocks": list(blocks)}


def block(kind: str, text: str | None = None, **fields: Any) -> dict[str, Any]:
    return {"kind": kind, "text": text, "items": fields.get("items", []), "rows": fields.get("rows", [])}


PARAGRAPH = "Nhân viên tuyệt đối không hứa bồi thường cho khách trước khi phân loại mức độ khiếu nại."
REPEAT = "Không hứa bồi thường cho khách trước khi phân loại mức độ khiếu nại."
LEVEL_3 = "Liên quan an toàn, pháp lý hoặc truyền thông. Escalate ngay cho quản lý và bộ phận Pháp chế."
NEW_POINT = "Khiếu nại cấp 3 phải escalate ngay cho quản lý và bộ phận Pháp chế."


class CalloutRepeatTests(StoryboardEndpointTestCase):
    """A3: a quotation or callout box adds a new point; one that repeats its slot's text goes, its text stays."""

    def test_a_callout_that_restates_the_paragraph_is_found(self) -> None:
        signs = block("table", rows=[{"label": "Cấp 1", "value": "Ảnh hưởng thấp, không thiệt hại tài chính"},
                                     {"label": "Cấp 3", "value": "Liên quan an toàn, pháp lý hoặc truyền thông"}])
        semantic = {"sections": [section(block("paragraph", PARAGRAPH), block("warning", REPEAT), signs,
                                         block("warning", NEW_POINT))]}
        self.assertEqual(repeated_callouts(semantic), [(0, 1)])
        # A list item, a table row or a callout before it in the section count too; a callout of three words is not
        # judged (review M3: only the blocks before it in its own section are compared).
        listed = {"sections": [section(block("bullets", items=[PARAGRAPH]), block("warning", REPEAT),
                                       block("warning", "Không hứa."))]}
        self.assertEqual(repeated_callouts(listed), [(0, 1)])
        table = {"sections": [section(block("table", rows=[{"label": "Cấp 3", "value": LEVEL_3}]),
                                      block("warning", "Cấp 3: " + LEVEL_3))]}
        self.assertEqual(repeated_callouts(table), [(0, 1)])
        twice = {"sections": [section(block("warning", NEW_POINT), block("warning", NEW_POINT))]}
        self.assertEqual(repeated_callouts(twice), [(0, 1)])
        self.assertEqual(repeated_callouts({"sections": [section(block("warning", NEW_POINT))]}), [])

    def test_dropping_a_callout_keeps_every_other_block(self) -> None:
        semantic = {"version": 1, "sections": [section(block("paragraph", PARAGRAPH)),
                                               section(block("warning", REPEAT), heading="Lưu ý")]}
        trimmed = without_blocks(semantic, [(1, 0)])
        self.assertEqual(trimmed, {"version": 1, "sections": [section(block("paragraph", PARAGRAPH))]})
        self.assertIsNone(without_blocks({"sections": [section(block("warning", REPEAT))]}, [(0, 0)]))

    async def test_the_repeating_callout_goes_without_a_repair_call(self) -> None:
        body = severity_body()
        writer = severity_writer(body, explanation=LINES)
        blocks = writer["components"]["c0"]["semantic_content"]["sections"][0]["blocks"]
        blocks.append(block("warning", "Cấp 3: liên quan an toàn, pháp lý hoặc truyền thông. Escalate ngay cho quản lý "
                                       "và bộ phận Pháp chế theo quy trình CR-04."))
        provider = FakeGenerate(**{WRITER: [writer], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        self.assertEqual(provider.names, [WRITER, JUDGE])
        kinds = [item["kind"] for item in data["unit"]["components"][0]["semantic_content"]["sections"][0]["blocks"]]
        self.assertEqual(kinds, ["paragraph", "table"])
        note = data["unit"]["idm_quality"]["author_note"]
        self.assertIn("Đã bỏ 1 khung trích dẫn/lưu ý chỉ nhắc lại đoạn văn có sẵn trong bài.", note)
        self.assertIn("never the paragraph or list above it again", provider.calls[0]["prompt"])


POINTS = ["Cấp 1: ảnh hưởng thấp, nhân viên tự xử lý", "Cấp 2: thiệt hại dưới 50 triệu đồng, báo trưởng nhóm",
          "Cấp 3: an toàn, pháp lý, truyền thông, escalate ngay"]
# The description QC found too thin to draw (a setting, no layout, no text, no point).
THIN = "Một bảng kiểm treo tại phòng họp."
RICH = ("Bố cục ba cột ngang, đọc từ trái sang phải, mỗi cột một cấp độ với nền màu tăng dần. Khung 1 ở bên trái ghi "
        "nhãn “Cấp 1” và dòng chữ “ảnh hưởng thấp, nhân viên tự xử lý”. Khung 2 ở giữa ghi “Cấp 2” với con số “dưới "
        "50 triệu đồng” và biểu tượng trưởng nhóm. Khung 3 bên phải ghi “Cấp 3” với ba biểu tượng an toàn, pháp lý, "
        "truyền thông và chữ “escalate ngay”. Mũi tên từ khung 1 sang khung 3 cho thấy mức độ tăng dần.")


def media(description: str, points: list[str] | None = None, kind: str = "static_infographic") -> Any:
    return ArchitectureMediaBriefV2.model_validate({
        "type": kind, "title": "Ba cấp độ nghiêm trọng", "content_points": points or POINTS,
        "context_description": description, "rationale": "Giúp so sánh nhanh ba cấp độ."})


class MediaBriefTests(unittest.IsolatedAsyncioTestCase):
    """A4: the description of a suggested infographic or video is a brief a designer can draw from."""

    def test_the_gaps_of_a_thin_description(self) -> None:
        # Owner decision 2026-10-22: judged by meaning (length, layout, something drawn or written); a "bảng kiểm" is
        # drawn.
        self.assertEqual(media_brief_gaps(media(THIN)), ["7 words (at least 40)", "no layout"])
        self.assertEqual(media_brief_gaps(media("Không khí vui vẻ.")),
                         ["4 words (at least 40)", "no layout", "nothing drawn or written on the image"])
        self.assertEqual(media_brief_gaps(media(RICH)), [])
        # A point the description names by its words (not its number) counts as described.
        by_words = RICH.replace("Khung 2 ở giữa ghi “Cấp 2” với con số “dưới 50 triệu đồng”",
                                "Ở giữa: thiệt hại dưới 50 triệu")
        self.assertEqual(media_brief_gaps(media(by_words)), [])

    def test_a_thin_description_gets_a_layout_from_the_points(self) -> None:
        completed = complete_media_brief(media(THIN), "vi")
        assert completed is not None
        text = completed.context_description
        self.assertTrue(text.startswith(THIN + " Bố cục cho người thiết kế: 3 khung đọc từ trái sang phải"))
        for number, point in enumerate(POINTS, start=1):
            self.assertIn(f"Khung {number}: “{point}”", text)
        self.assertIn("Chỉ nối các khung bằng mũi tên", text)
        self.assertEqual(media_brief_gaps(completed), [])
        self.assertIs(complete_media_brief(media(RICH), "vi").context_description, RICH)  # type: ignore[union-attr]
        video = complete_media_brief(media(THIN, kind="video"), "en")
        assert video is not None
        self.assertIn("Scene 2: “Cấp 2: thiệt hại dưới 50 triệu đồng, báo trưởng nhóm”", video.context_description)
        self.assertIsNone(complete_media_brief(None, "vi"))

    def test_the_completion_fits_the_contract(self) -> None:
        long_points = [" ".join(["Bước", str(number), *(["xử lý khiếu nại đúng quy trình"] * 20)])[:600]
                       for number in range(1, 7)]
        completed = complete_media_brief(media("Một cảnh văn phòng. " * 5, long_points), "vi")
        assert completed is not None
        self.assertLessEqual(len(completed.context_description), 1000)
        # Owner decision 2026-10-22: a long point is named by its number, never quoted cut.
        self.assertIn("Khung 6: ý 6 của nội dung", completed.context_description)
        self.assertNotIn("…", completed.context_description)
        ArchitectureMediaBriefV2.model_validate(completed.model_dump(mode="json"))

    async def test_the_module_stage_completes_a_thin_brief_without_a_call(self) -> None:
        def thin(payload: dict[str, Any]) -> None:
            payload["lessons"][0]["units"][0]["media_brief"] = media(THIN).model_dump(mode="json")

        result, provider, runtime = await design_module(0, [edited("mod_01", thin)])
        self.assertEqual(len(provider.calls), 1)
        lesson = result["shard"]["idm_design"]["lessons"][0]
        self.assertIn("Khung 3: “Cấp 3: an toàn, pháp lý, truyền thông, escalate ngay”",
                      lesson["units"][0]["media_brief"]["context_description"])
        self.assertIn("Gợi ý hình minh hoạ/video: phần mô tả AI viết còn sơ sài nên hệ thống đã bổ sung bố cục",
                      lesson["notes"])
        self.assertEqual(runtime.adjustments[MEDIA_BRIEF_COMPLETED_CODE], 1)
        self.assertIn("Its context_description is the brief a designer draws from", provider.calls[0]["prompt"])

    async def test_a_repair_the_answer_gets_anyway_lists_the_thin_brief(self) -> None:
        def thin_and_foreign(payload: dict[str, Any]) -> None:
            payload["lessons"][0]["units"][0]["media_brief"] = media(THIN).model_dump(mode="json")
            payload["lessons"][1]["practice_tasks"][0]["criteria_fact_keys"].append(key(2, 2))

        def rich(payload: dict[str, Any]) -> None:
            payload["lessons"][0]["units"][0]["media_brief"] = media(RICH).model_dump(mode="json")

        result, provider, _runtime = await design_module(0, [edited("mod_01", thin_and_foreign),
                                                             edited("mod_01", rich)])
        self.assertEqual(len(provider.calls), 2)
        repair = provider.calls[1]["prompt"].split("REPAIR_REQUIREMENTS", 1)[1]
        self.assertIn('{"code":"IDM_W4_MEDIA_BRIEF_THIN","path":"lessons[0].units[0].media_brief.context_description"}',
                      repair)
        lesson = result["shard"]["idm_design"]["lessons"][0]
        self.assertEqual(lesson["units"][0]["media_brief"]["context_description"], RICH)


class QuizLayoutBoundTests(unittest.TestCase):
    def test_the_layout_never_pushes_a_text_over_the_node_limit(self) -> None:
        # The last option segment gains a closing full stop: at the 1500 characters Node keeps, the layout would make
        # 1501, so the writer's paragraph stays as it is (the length check still asks the repair to shorten it).
        reason = "vì" + " tiêu chí" * 52
        options = f" A - Sai {reason}; B - Đúng {reason}; C - Sai {reason}"
        paragraph = ("Tiêu chí" + " x" * 100)[:1499 - len(options)] + "." + options
        self.assertEqual((len(paragraph), len(layout_explanation(paragraph, 3))), (1500, 1501))
        laid, codes = lay_out_quiz(problem(paragraph))
        self.assertEqual(laid["explanation"], paragraph)
        self.assertNotIn("w5_problem_explanation_laid_out", codes)
        shorter = paragraph[:1400]
        self.assertEqual(lay_out_quiz(problem(shorter))[0]["explanation"], layout_explanation(shorter, 3))


class FallbackFeedbackTests(StoryboardEndpointTestCase):
    """Owner decision 2026-10-22: the source-locked question that replaces a failed AI question explains itself in the
    same shape: the principle line, then one line per option, its letter matching the option shown."""

    async def test_a_fallback_question_has_one_line_per_option(self) -> None:
        body = severity_body()
        writer = severity_writer(body, explanation=LINES)
        for choice in writer["components"]["c1"]["choices"]:
            choice["correct"] = True
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [slot_repair(writer, 1)], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        slot = data["unit"]["components"][1]
        lines = slot["explanation"].splitlines()
        self.assertEqual(lines[0], "Cấp 1 tương ứng với Ảnh hưởng thấp, không thiệt hại tài chính.")
        self.assertEqual(len(lines), len(slot["choices"]) + 1)
        for letter, line, choice in zip("ABCD", lines[1:], slot["choices"], strict=False):
            self.assertTrue(line.startswith(f"{letter} - {'Đúng' if choice['correct'] else 'Sai'} vì đây là dấu hiệu"))
        self.assertEqual(explanation_report(slot), {})


def flagged_escalate_body(issues: list[dict[str, Any]], gaps: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """The escalate unit (its FAQ material describes block cb_0011) with W1 issues and gaps on that block."""

    body = escalate_body()
    brief = body["unit_contract"]["idm_unit_brief"]
    brief["faq_material"]["blocks"][0].update(issues=issues, gaps=gaps or [])
    brief["brief_hash"] = brief_hash_of(brief)
    rehash_contract(body["unit_contract"])
    return body


CONFLICT = {"type": "conflict", "note": "Ví dụ và lưu ý nêu hai cách xử lý khác nhau.",
            "fact_keys": ["d1-c8-f2", "d1-c8-f3"]}


class SourceFlagTests(StoryboardEndpointTestCase):
    """Owner decision 2026-10-22: a contradictory, unclear or out-of-date source point is taught and flagged for the
    author ("Cần xem"), quoted from the facts with its page; never a question for an SME."""

    def test_one_plain_sentence_per_point_from_the_facts(self) -> None:
        facts = {"k1": SourceFact("Nhân viên cần phản hồi khách hàng trong vòng 24 giờ.", 4),
                 "k2": SourceFact("Khiếu nại cấp 2 được phản hồi trong vòng 48 giờ.", 7)}
        conflict = IdmIssueV1(type="conflict", note="24 giờ hay 48 giờ? (ghi chú của AI)", fact_keys=["k1", "k2"])
        self.assertEqual(flag_sentence("Thời hạn phản hồi", conflict, facts, "vi"),
                         "Tài liệu ghi khác nhau ở phần “Thời hạn phản hồi”: “Nhân viên cần phản hồi khách hàng trong "
                         "vòng 24 giờ.” (trang 4) và “Khiếu nại cấp 2 được phản hồi trong vòng 48 giờ.” (trang 7). Hãy "
                         "kiểm tra trước khi áp dụng.")
        self.assertEqual(flag_sentence("Response time", conflict, facts, "en"),
                         "The source differs in “Response time”: “Nhân viên cần phản hồi khách hàng trong vòng 24 "
                         "giờ.” (page 4) and “Khiếu nại cấp 2 được phản hồi trong vòng 48 giờ.” (page 7). Check this "
                         "before applying it.")
        # Without facts it names the block; the AI's own note is never quoted. Run 5496155c: an unclear point is an
        # edit note, never a check before applying.
        unclear = IdmIssueV1(type="unclear", note="Không rõ ai duyệt.", fact_keys=[])
        self.assertEqual(flag_sentence("Ai duyệt hoàn tiền?", unclear, facts, "vi"),
                         "Tài liệu ghi chưa rõ ở phần “Ai duyệt hoàn tiền?” — nên đọc lại khi biên tập.")
        outdated = IdmIssueV1(type="outdated", note="CRM 2021.", fact_keys=["k1"])
        self.assertTrue(str(flag_sentence("CRM", outdated, facts, "en")).startswith("The source may be out of date"))
        for kind in ("duplicate", "too_general", "too_detailed"):
            self.assertIsNone(flag_sentence("X", IdmIssueV1(type=kind, note="n", fact_keys=["k1"]), facts, "vi"))
        self.assertEqual(flag_sentences([("A", [conflict]), ("A", [conflict])], facts, "vi"),
                         [flag_sentence("A", conflict, facts, "vi")])

    async def test_a_contradictory_block_marks_its_unit_for_review(self) -> None:
        body = flagged_escalate_body([CONFLICT])
        status, data, provider = await self.post(body, FakeGenerate(**{WRITER: [escalate_writer(body)],
                                                                       JUDGE: [judge()]}))
        self.assertEqual(status, 200)
        self.assertEqual((data["content_origin"], data["quality_state"]), ("provider_validated", "review_required"))
        self.assertEqual(provider.names, [WRITER, JUDGE])
        note = data["unit"]["idm_quality"]["author_note"]
        self.assertTrue(note.startswith("Cần tác giả xem lại vì: tài liệu nguồn có chỗ ghi khác nhau hoặc có thể đã "
                                        "cũ (chi tiết bên dưới). Tài liệu ghi khác nhau ở phần"))
        self.assertIn("Tài liệu ghi khác nhau ở phần “Những lỗi nào nhân viên thường mắc khi xử lý khiếu nại?”: "
                      "“Ví dụ: Nhân viên hứa hoàn tiền ngay", note)
        self.assertIn("(trang 2) và “Lưu ý: Nhiều nhân viên ghi nhận khiếu nại cấp 3 thành cấp 2", note)
        self.assertIn("Hãy kiểm tra trước khi áp dụng.", note)
        self.assertNotIn("Câu hỏi cho SME", note)
        self.assertNotIn(CONFLICT["note"], note)

    async def test_missing_criteria_alone_is_no_flag(self) -> None:
        body = flagged_escalate_body([], gaps=[{"type": "missing_criteria", "note": "Không có tiêu chí chấm."}])
        status, data, _ = await self.post(body, FakeGenerate(**{WRITER: [escalate_writer(body)], JUDGE: [judge()]}))
        self.assertEqual(status, 200)
        self.assertEqual(data["quality_state"], "validated")
        self.assertNotIn("Hãy kiểm tra trước khi áp dụng", data["unit"]["idm_quality"]["author_note"])

    async def test_a_unit_without_an_faq_reads_only_the_issues_of_its_material(self) -> None:
        # When Node sends the blocks of a unit without an FAQ slot, Python flags their points and never gives the
        # material to the writer.
        body = severity_body()
        brief = body["unit_contract"]["idm_unit_brief"]
        brief["faq_material"] = {"blocks": [{
            "block_id": "cb_0005", "name": "Đánh giá mức độ nghiêm trọng của khiếu nại như thế nào?",
            "intent": "decide", "content_kind": "decision", "support_role": None,
            "issues": [{"type": "outdated", "note": "Ngưỡng 50 triệu đồng có thể đã đổi.", "fact_keys": ["d1-c4-f4"]}],
            "gaps": [], "sme_questions": []}], "practices": []}
        brief["brief_hash"] = brief_hash_of(brief)
        rehash_contract(body["unit_contract"])
        provider = FakeGenerate(**{WRITER: [severity_writer(body, explanation=LINES)], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        self.assertEqual(data["quality_state"], "review_required")
        self.assertIn("Tài liệu có thể đã cũ ở phần “Đánh giá mức độ nghiêm trọng của khiếu nại như thế nào?”: “Cấp 2 "
                      "| Thiệt hại tài chính dưới 50 triệu", data["unit"]["idm_quality"]["author_note"])
        self.assertNotIn("FAQ_MATERIAL", provider.calls[0]["prompt"])
        self.assertNotIn("faq_material", provider.calls[0]["prompt"])
