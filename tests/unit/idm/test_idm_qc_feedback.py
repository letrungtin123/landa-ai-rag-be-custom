"""QC feedback of 2026-10-21 (owner-approved change list A1-A7): the unit-level rules of the AI service.

* A1/A2 (contract K1): the stem and the explanation of a question are laid out in lines without changing a word; what
  the layout cannot fix (a principle that restates three definitions, empty praise instead of a reason, a verdict that
  contradicts the key, a stem of four sentences) is a code the writer repairs once, then a plain note.
* K1: every line feed of the question and the explanation reaches the unit Node receives, and its problem XML.
* A3: a callout that only repeats its slot's paragraphs, lists or tables is removed (its text stays there).
* A4: the description of a suggested infographic or video says what to draw (layout, text, one element per point);
  a thin one is completed from the content points and noted.
* A5 (no Hold): a worksheet whose practice criteria the AI suggested from what the lesson teaches heads its self-check
  as suggested criteria, and the unit note says so.

Every provider is a local fake.
"""

from __future__ import annotations

import unittest
from collections import Counter
from typing import Any

from app.contracts.orchestration_v2.blueprint import ArchitectureMediaBriefV2
from app.idm.artifacts import clean_text
from app.idm.contracts import brief_hash_of
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
from app.idm.text import evidence_index
from app.idm.worksheet import label_suggested_self_check
from tests.idm_golden import key
from tests.support.idm_test_support import rehash_contract
from tests.unit.idm.test_idm_module_design import design_module, edited
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
        self.assertIn("132 words", question_report({"question": long})[QUESTION_LENGTH_CODE][0])
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
    async def test_the_self_check_of_suggested_criteria_says_so(self) -> None:
        body = suggested_worksheet_body()
        provider = FakeGenerate(**{WRITER: [worksheet_writer(body)], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        sections = data["unit"]["components"][0]["semantic_content"]["sections"]
        self.assertEqual(sections[-1]["heading"], "Tự kiểm tra bảng của bạn (tiêu chí gợi ý)")
        self.assertIn('"Tự kiểm tra (tiêu chí gợi ý)" / "Self-check (suggested criteria)"', provider.calls[0]["prompt"])
        note = data["unit"]["idm_quality"]["author_note"]
        self.assertIn("AI gợi ý tiêu chí từ nội dung đã dạy (ghi là tiêu chí gợi ý)", note)

    async def test_source_criteria_keep_their_heading_and_note(self) -> None:
        body = worksheet_body()
        provider = FakeGenerate(**{WRITER: [worksheet_writer(body)], JUDGE: [judge()]})
        _status, data, _ = await self.post(body, provider)
        sections = data["unit"]["components"][0]["semantic_content"]["sections"]
        self.assertEqual(sections[-1]["heading"], "Tự kiểm tra bảng của bạn")
        self.assertNotIn("tiêu chí gợi ý", data["unit"]["idm_quality"]["author_note"])

    def test_the_label_is_added_once_and_fits_the_heading_bound(self) -> None:
        semantic = {"sections": [{"heading": "Nhiệm vụ", "blocks": [{"kind": "task", "text": "Làm."}]},
                                 {"heading": "Tự kiểm tra", "blocks": [{"kind": "bullets", "items": ["a", "b"]}]}]}
        labelled = label_suggested_self_check(semantic, "vi", 120)
        assert labelled is not None
        self.assertEqual(labelled["sections"][1]["heading"], "Tự kiểm tra (tiêu chí gợi ý)")
        self.assertEqual(labelled["sections"][0], semantic["sections"][0])
        self.assertIsNone(label_suggested_self_check(labelled, "vi", 120))
        long = {"sections": [{"heading": "x" * 110, "blocks": [{"kind": "bullets", "items": ["a"]}]}]}
        english = label_suggested_self_check(long, "en", 120)
        self.assertEqual(english["sections"][0]["heading"] if english else None, "Self-check (suggested criteria)")
        self.assertIsNone(label_suggested_self_check({"sections": [{"heading": "a", "blocks": []}]}, "vi", 120))


def section(*blocks: dict[str, Any], heading: str = "Khiếu nại thuộc cấp độ nào?") -> dict[str, Any]:
    return {"heading": heading, "blocks": list(blocks)}


def block(kind: str, text: str | None = None, **fields: Any) -> dict[str, Any]:
    return {"kind": kind, "text": text, "items": fields.get("items", []), "rows": fields.get("rows", [])}


PARAGRAPH = "Nhân viên tuyệt đối không hứa bồi thường cho khách trước khi phân loại mức độ khiếu nại."
REPEAT = "Không hứa bồi thường trước khi phân loại mức độ."
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
        # A list item, a table row or a callout before it count too; a callout of three words is not judged.
        listed = {"sections": [section(block("bullets", items=[PARAGRAPH])), section(block("warning", REPEAT),
                                                                                      block("warning", "Không hứa."))]}
        self.assertEqual(repeated_callouts(listed), [(1, 0)])
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
        self.assertEqual(media_brief_gaps(media(THIN)), [
            "7 words (at least 40)", "no layout", "no text on the image", "content points 1, 2, 3 not described"])
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
        completed = complete_media_brief(media("Một cảnh văn phòng. " * 40, long_points), "vi")
        assert completed is not None
        self.assertLessEqual(len(completed.context_description), 1000)
        self.assertIn("Khung 6: “Bước 6", completed.context_description)
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
