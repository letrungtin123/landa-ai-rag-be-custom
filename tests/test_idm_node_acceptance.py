"""Node revision-0 acceptance mirror (``app.idm.node_acceptance``) and its use by the W5 writer (run c2e5ac41).

Run c2e5ac41 lost two worksheet units: Python validated them, Node rejected them at ``unit_acceptance``
(``ORCHESTRATION_V2_UNIT_BASELINE_INVALID``) and the ``fallback_only`` retry returned 422. Verdict parity with
the real Node code is checked by the backend's ``lesson-author-idm-acceptance.cross-language.test.ts``.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import unittest
from typing import Any
from unittest.mock import AsyncMock

from app import main
from app.idm.node_acceptance import (
    AcceptanceContext,
    html_contract_code,
    html_quality_code,
    idm_html_visible_text,
    instructional_fold,
    js_len,
    js_slice,
    node_acceptance_findings,
    plain_text_valid,
    render_semantic_html,
    single_choice_problem_code,
    visible_html_text,
    workspace_problem_code,
)
from app.idm.runtime import IdmStageError
from app.idm.source_locked import idm_source_faq
from app.idm.storyboard import acceptance_context, parse_brief, run_idm_unit
from app.lesson_author_orchestration_v2_provider import UnitGenerationContractV2
from app.schemas.orchestration_v2 import RagLessonAuthorUnitV2Request
from tests.idm_test_support import make_runtime
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
    unavailable,
    worksheet_body,
    worksheet_writer,
)

ASTRAL = chr(0x1D538)
FULLWIDTH_QUESTION = chr(0xFF1F)
BELL = chr(0x07)
GUIDANCE = "Ghi cấp độ và người cần xử lý vào ô này."
WORKSHEET_TABLES = (
    "<h2>Mẫu phiếu cần điền</h2><table><tbody>"
    f"<tr><th>Khiếu nại mẫu thứ nhất</th><td>{GUIDANCE}</td></tr>"
    f"<tr><th>Khiếu nại mẫu thứ hai</th><td>{GUIDANCE}</td></tr></tbody></table>"
    "<h2>Ví dụ đã điền</h2><table><tbody><tr><th>Khiếu nại mẫu thứ nhất</th>"
    "<td>Cấp 2: thông báo trưởng nhóm.</td></tr></tbody></table>"
)


def block(kind: str, **value: Any) -> dict[str, Any]:
    return {"kind": kind, "text": None, "items": [], "rows": [], **value}


def repeated_cells_worksheet(body: dict[str, Any]) -> dict[str, Any]:
    """The run c2e5ac41 worksheet shape: blank-cell guidance repeated, template labels reused by the example."""

    writer = worksheet_writer(body)
    sections = writer["components"]["c0"]["semantic_content"]["sections"]
    sections[1]["blocks"] = [block("table", rows=[{"label": f"Khiếu nại mẫu {label}", "value": GUIDANCE}
                                                  for label in ("thứ nhất", "thứ hai", "thứ ba")])]
    example = {"label": "Khiếu nại mẫu thứ nhất", "value": "Cấp 2: thông báo trưởng nhóm vì khách phàn nàn lần hai."}
    sections[2]["blocks"] = [block("table", rows=[example])]
    return writer


class JavaScriptSemanticsTests(unittest.TestCase):
    def test_lengths_slices_and_folding_follow_node(self) -> None:
        self.assertEqual((len(ASTRAL), js_len(ASTRAL)), (1, 2))
        self.assertEqual(js_len(js_slice(ASTRAL * 3, 3)), 3)  # a slice may end inside a surrogate pair, as in JS
        self.assertEqual(visible_html_text("<p>a &amp;\n b</p><p>c</p>"), "a b c")
        self.assertEqual(instructional_fold("Hiện trạng: 3 lần!"), "hien trang 3 lan")
        self.assertEqual(instructional_fold("Đoạn"), "đoan")  # Node keeps "đ": it has no combining mark

    def test_plain_text_rules_of_interactive_components(self) -> None:
        self.assertTrue(plain_text_valid("Phân loại rồi escalate."))
        self.assertFalse(plain_text_valid("phân loại -> escalate"))
        self.assertFalse(plain_text_valid(f"Câu hỏi{BELL}"))
        self.assertFalse(plain_text_valid("   "))


class HtmlRuleTests(unittest.TestCase):
    def test_quality_codes_match_the_node_validator(self) -> None:
        cases = {
            "<p>Kiểm tra điều kiện an toàn.</p><p>Kiểm tra điều kiện an toàn.</p>": "HTML_DUPLICATE_BLOCK",
            "<p>Xem chi tiết tại trang 12.</p>": "HTML_SOURCE_LOCATOR",
            "<table><tbody><tr><th>Hiện trạng</th><td>3 lần giao trễ.</td></tr></tbody></table>": "HTML_SOURCE_LOCATOR",
            "<p>Hiện trạng: có 3 lần giao trễ trong tháng.</p>": None,
            "<p>Theo tài liệu nguồn, cần phân loại.</p>": "HTML_SOURCE_ATTRIBUTION",
            "<p>Nội dung từ playbook.pdf.</p>": "HTML_SOURCE_FILENAME",
            "<p>Ghi chú p5-f2 cho người học.</p>": "HTML_INTERNAL_IDENTIFIER",
            "<p>vvvvvvvvvvvv</p>": "HTML_OCR_NOISE",
            "<ul><li>www.l-a.com.vn</li></ul>": "HTML_BOILERPLATE",
            "<p>Kiểm tra trang thiết bị trước khi làm việc.</p>": None,
        }
        for html, code in cases.items():
            with self.subTest(html=html):
                self.assertEqual(html_quality_code(html), code)
        self.assertEqual(html_quality_code("<p>Ghi chú d1-c0-f9 ở đây.</p>", ["d1-c0-f9"]), "HTML_INTERNAL_IDENTIFIER")

    def test_a_worksheet_may_repeat_table_cells_but_not_paragraphs(self) -> None:
        self.assertEqual(html_quality_code(WORKSHEET_TABLES), "HTML_DUPLICATE_BLOCK")
        self.assertIsNone(html_quality_code(WORKSHEET_TABLES, worksheet=True))
        twice = "<p>Đối chiếu từng dòng với tiêu chí.</p>" * 2
        self.assertEqual(html_quality_code(WORKSHEET_TABLES + twice, worksheet=True), "HTML_DUPLICATE_BLOCK")

    def test_contract_and_renderer(self) -> None:
        self.assertEqual(html_contract_code("<h2>Tiêu đề</h3>"), "HTML_INVALID_NESTING")
        self.assertEqual(html_contract_code("chữ<p>a</p>"), "HTML_TEXT_OUTSIDE_ROOT")
        self.assertEqual(html_contract_code(""), "HTML_EMPTY")
        rendered = render_semantic_html({"version": 2, "sections": [{"heading": "A & B", "blocks": [
            block("table", rows=[{"label": "x<y", "value": "'q'"}]), block("warning", text="Cẩn thận")]}]})
        self.assertEqual(rendered, "<h2>A &amp; B</h2><table><tbody><tr><th>x&lt;y</th><td>&#39;q&#39;</td></tr>"
                                   "</tbody></table><blockquote>Cẩn thận</blockquote>")
        self.assertEqual(render_semantic_html({"heading": "Tiêu đề", "paragraphs": [" Một "], "bullets": ["Hai"]}),
                         "<h2>Tiêu đề</h2><p>Một</p><ul><li>Hai</li></ul>")
        self.assertIsNone(render_semantic_html({}))

    def test_budget_text_is_python_visible_text(self) -> None:
        semantic = {"version": 2, "sections": [{"heading": "Không tính", "blocks": [
            block("paragraph", text="  một  hai "), block("table", rows=[{"label": " ba ", "value": "bốn"}])]}]}
        self.assertEqual(idm_html_visible_text({"semantic_content": semantic}), "một  hai ba bốn")
        self.assertEqual(idm_html_visible_text({"html": "<p>a &amp;\n\n b</p>"}), "a &amp; b")
        self.assertIsNone(idm_html_visible_text({"semantic_content": {"version": 2, "sections": []}}))


class ProblemRuleTests(unittest.TestCase):
    XML = ('<problem>\n  <multiplechoiceresponse>\n    <label>Khiếu nại này thuộc cấp nào trong quy trình?</label>\n'
           '    <choicegroup type="MultipleChoice">\n'
           '      <choice correct="false">Cấp 1 vì chưa có thiệt hại</choice>\n'
           '      <choice correct="true">Cấp 2 vì khách phàn nàn lần hai</choice>\n'
           '      <choice correct="false">Cấp 3 vì cần escalate ngay</choice>\n    </choicegroup>\n'
           '  </multiplechoiceresponse>\n  <solution><div class="detailed-solution"><p>Tiêu chí: lần phàn nàn '
           'thứ hai là cấp 2.</p></div></solution>\n</problem>')

    def test_single_choice_and_workspace_codes(self) -> None:
        self.assertIsNone(single_choice_problem_code(self.XML))
        self.assertIsNone(workspace_problem_code(self.XML))
        self.assertEqual(single_choice_problem_code(self.XML.replace("Cấp 1 vì chưa có thiệt hại", "Có")),
                         "PROBLEM_CHOICES_NOT_DISTINCT")
        self.assertEqual(single_choice_problem_code(self.XML.replace('correct="false"', 'correct="true"', 1)),
                         "PROBLEM_CORRECT_COUNT")
        short = self.XML.replace("Tiêu chí: lần phàn nàn thứ hai là cấp 2.", "Đúng.")
        self.assertEqual(single_choice_problem_code(short), "PROBLEM_EXPLANATION_MISSING")
        self.assertEqual(single_choice_problem_code(""), "PROBLEM_XML_EMPTY")
        self.assertEqual(workspace_problem_code(self.XML.replace("Cấp 3 vì cần escalate ngay", f"Cấp 3{BELL}")),
                         "PROBLEM_TEXT_INVALID")


def unit_with(components: list[dict[str, Any]],
              plans: list[dict[str, Any]]) -> tuple[dict[str, Any], AcceptanceContext]:
    unit = {"title": "Đơn vị", "source_fact_ids": ["f1"], "components": components}
    context = AcceptanceContext(unit_title="Đơn vị", unit_source_fact_ids=("f1",), plans=tuple(plans),
                                exact_identifiers=("f1",), budget={"max_words": 400, "max_visible_chars": 2_800})
    return unit, context


def plan(index: int, kind: str, owned: list[str]) -> dict[str, Any]:
    return {"component_plan_id": f"cp2_{index:032d}", "type": kind, "source_fact_ids": owned,
            "supporting_evidence_fact_ids": [] if owned else ["f1"], "learning_objective_refs": ["lo_1"],
            "required_artifacts": []}


def component(index: int, kind: str, owned: list[str], **payload: Any) -> dict[str, Any]:
    return {"type": kind, "title": "Khối", "component_plan_id": f"cp2_{index:032d}", "source_fact_ids": owned,
            "covered_source_fact_ids": owned, "supporting_evidence_fact_ids": [] if owned else ["f1"], **payload}


class UnitFindingTests(unittest.TestCase):
    def html(self, *texts: str) -> dict[str, Any]:
        return component(0, "html", ["f1"], html=None, semantic_content={"version": 2, "sections": [
            {"heading": "Phần", "blocks": [block("paragraph", text=text) for text in texts]}]})

    def faq(self, *pairs: tuple[str, str]) -> dict[str, Any]:
        return component(1, "la_faq", [], items=[{"question": q, "answer": a} for q, a in pairs])

    def test_findings_follow_node_order_and_name_items(self) -> None:
        long = "Giải thích chi tiết cho người học về quy trình phân loại khiếu nại khách hàng. " * 3
        faq = self.faq(("Câu hỏi đầu tiên về phân loại?", "Trả lời -> bước tiếp theo."),
                       (f"Câu hỏi thứ hai về phân loại{FULLWIDTH_QUESTION}", "Câu trả lời đầy đủ cho câu hỏi thứ hai."),
                       ("Câu hỏi thứ hai về phân loại?", "Một câu trả lời đầy đủ khác cho cùng câu hỏi."))
        unit, context = unit_with([self.html(long, long), faq], [plan(0, "html", ["f1"]), plan(1, "la_faq", [])])
        findings = node_acceptance_findings(unit, context, provider_validated=True)
        # Node stops at the first failing stage (coverage before the workspace payloads).
        self.assertEqual([item.as_log() for item in findings], [
            "coverage:HTML_DUPLICATE_BLOCK@components[0]",
            "workspace_component:WORKSPACE_COMPONENT_PAYLOAD_INVALID@components[1]#FAQ_TEXT_FORBIDDEN_CHARACTER"])
        self.assertEqual(findings[1].items, (0,))
        unit["components"][1]["items"].pop(0)
        findings = node_acceptance_findings(unit, context, provider_validated=True)
        self.assertEqual((findings[-1].code, findings[-1].detail, findings[-1].items),
                         ("WORKSPACE_COMPONENT_REFERENCE_INVALID", "FAQ_DUPLICATE_QUESTION", (1,)))

    def test_budget_applies_to_provider_content_only(self) -> None:
        text = " ".join(f"mục{index}" for index in range(250))
        unit, context = unit_with([self.html(text, text.replace("mục", "phần"))], [plan(0, "html", ["f1"])])
        self.assertEqual([item.as_log() for item in node_acceptance_findings(unit, context, provider_validated=True)],
                         ["idm_budget:IDM_HTML_DENSITY_EXCEEDED@components[0]"])
        self.assertEqual(node_acceptance_findings(unit, context, provider_validated=False), [])

    def test_normalization_and_identity(self) -> None:
        thin, context = unit_with([self.html("Quá ngắn.")], [plan(0, "html", ["f1"])])
        self.assertEqual(node_acceptance_findings(thin, context, provider_validated=False)[0].as_log(),
                         "normalization:HTML_TOO_THIN@components[0]")
        moved = copy.deepcopy(thin)
        moved["components"][0]["covered_source_fact_ids"] = []
        self.assertEqual(node_acceptance_findings(moved, context, provider_validated=False)[0].code, "HTML_TOO_THIN")
        renamed = node_acceptance_findings({**thin, "title": "Khác"}, context, provider_validated=False)
        self.assertEqual(renamed[0].as_log(), "response:RESPONSE_UNIT_IDENTITY@unit")
        sortable = component(0, "la_sortable", ["f1"], question_text="Sắp xếp", items=[{"text": "A < B"}, {"text": "b"},
                                                                                          {"text": "c"}])
        unit, context = unit_with([sortable], [plan(0, "la_sortable", ["f1"])])
        self.assertEqual(node_acceptance_findings(unit, context, provider_validated=False)[0].detail,
                         "SORTABLE_TEXT_FORBIDDEN_CHARACTER")


class SourceFaqTests(unittest.TestCase):
    def test_labelled_definitions_make_a_grounded_faq(self) -> None:
        facts = ["Quy luật chi phí: Đầu tư 1 đồng cho phòng ngừa lỗi từ đầu tiết kiệm 10 đồng chi phí kiểm tra.",
                 "Nguyên nhân sâu xa: Doanh nghiệp tiếc tiền đầu tư vào chuẩn hóa quy trình và đào tạo con người.",
                 "SHIFT 4: THINK SYSTEM Thoát Bẫy Chữa Cháy Sang Vận Hành Xuất Sắc",
                 "Ký hiệu: Chi phí kiểm tra luôn lớn hơn chi phí phòng ngừa, tức là 10 > 1 trong mọi trường hợp."]
        items = idm_source_faq("Thực hành 5Why", facts, locale="vi") or []
        self.assertEqual([item["question"] for item in items], ["Người học cần hiểu gì về Quy luật chi phí?",
                                                                 "Người học cần hiểu gì về Nguyên nhân sâu xa?"])
        self.assertTrue(all(item["answer"] in " ".join(facts) for item in items))
        self.assertIsNone(idm_source_faq("Thực hành", facts[:1], locale="vi"))
        prose = ["Một câu văn xuôi không có nhãn nào ở đầu câu cả."]
        self.assertIsNone(idm_source_faq("Thực hành", prose, locale="vi"))


class WriterIntegrationTests(StoryboardEndpointTestCase):
    async def test_worksheet_repeating_template_cells_is_accepted_without_repair(self) -> None:
        body = worksheet_body()
        provider = FakeGenerate(**{WRITER: [repeated_cells_worksheet(body)], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200, data)
        self.assertEqual(provider.names, [WRITER, JUDGE])
        self.assert_envelope(data, "provider_validated", "validated", "provider")

    async def test_node_rule_gets_a_named_repair_and_is_logged_with_the_node_code(self) -> None:
        body = worksheet_body()
        writer = worksheet_writer(body)
        broken = copy.deepcopy(writer)
        broken["components"]["c0"]["semantic_content"]["sections"][1]["blocks"][0]["rows"].insert(
            0, {"label": "Hiện trạng", "value": "3 khiếu nại đang chờ phân loại."})
        provider = FakeGenerate(**{WRITER: [broken], REPAIR: [slot_repair(writer, 0)], JUDGE: [judge()]})
        with self.assertLogs("app.idm", level="INFO") as logs:
            status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200, data)
        self.assert_envelope(data, "provider_validated", "validated", "provider")
        self.assertIn("c0 HTML_SOURCE_LOCATOR at c0: never put a number right after", provider.calls[1]["prompt"])
        validation = next(json.loads(line.split("lesson_author_idm ", 1)[1]) for line in logs.output
                          if '"idm_unit_validation"' in line)
        self.assertEqual(validation["node_acceptance"], ["coverage:HTML_SOURCE_LOCATOR@components[0]"])
        self.assertEqual(validation["finding"], "HTML_SOURCE_LOCATOR@components[0]")

    async def test_faq_item_node_rejects_is_dropped_when_two_items_remain(self) -> None:
        body = escalate_body()
        writer = escalate_writer(body)
        items = writer["components"]["c1"]["items"]
        items.append({"question": "Thứ tự xử lý khiếu nại an toàn là gì?", "answer": items[0]["answer"] + " (A -> B)"})
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [unavailable()], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200, data)
        self.assertEqual(len(data["unit"]["components"][1]["items"]), 2)
        # The failed repair call leaves reserved accounting: Node only admits that as a reviewable structured draft.
        quality = self.assert_envelope(data, "structured_fallback", "review_required", "reserved_upper_bound")
        self.assertIn("Đã bỏ 1 câu hỏi đáp hệ thống không lưu được", quality.author_note)
        self.assertNotIn("nội dung ngoài tài liệu", quality.author_note)


class FallbackOnlyTests(unittest.IsolatedAsyncioTestCase):
    async def run_fallback_only(self, body: dict[str, Any], slots: list[dict[str, Any] | None]) -> dict[str, Any]:
        request = RagLessonAuthorUnitV2Request.model_validate(body)
        deps = main._idm_unit_deps(request)
        deps = dataclasses.replace(deps, source_locked_components=slots, source_locked_unit=(
            main.build_orchestration_v2_source_locked_unit(request.unit_contract, "vi", components=slots)))
        return await run_idm_unit(contract=request.unit_contract, runtime=make_runtime(AsyncMock(), allowance=None),
                                  deps=deps, fallback_only=True)

    async def test_fallback_only_uses_the_idm_faq_rebuild(self) -> None:
        # Run c2e5ac41 (5Why unit): only the la_faq slot had no rebuild, so fallback_only returned 422.
        body = escalate_body()
        request = RagLessonAuthorUnitV2Request.model_validate(body)
        slots = list(main._idm_unit_deps(request).source_locked_components or [])
        self.assertEqual([slot is None for slot in slots], [True, False])
        practice = {**slots[1], **severity_writer(severity_body())["components"]["c1"]}  # type: ignore[dict-item]
        owner = body["unit_contract"]["component_plan"][0]
        practice.update({key: owner[key] for key in ("component_plan_id", "source_fact_ids",
                                                     "supporting_evidence_fact_ids", "learning_objective_refs")},
                        type="problem", covered_source_fact_ids=list(owner["source_fact_ids"]))
        practice.pop("items", None)
        result = await self.run_fallback_only(body, [practice, slots[1]])
        self.assertEqual((result["content_origin"], result["usage_source"]), ("structured_fallback",
                                                                               "deterministic_fallback"))
        contract = UnitGenerationContractV2.model_validate(body["unit_contract"])
        self.assertEqual(node_acceptance_findings(result["unit"], acceptance_context(contract, parse_brief(contract)),
                                                  provider_validated=False), [])

    async def test_a_whole_fallback_node_would_reject_is_422_with_the_reason_logged(self) -> None:
        body = severity_body()
        request = RagLessonAuthorUnitV2Request.model_validate(body)
        slots = list(main._idm_unit_deps(request).source_locked_components or [])
        # The shared validator reads "trạng" with its accent; Node folds it to "trang" + a number (a page locator).
        located = {**slots[0], "html": slots[0]["html"].replace("</td>", " Hiện trạng 3 lần</td>", 1)}  # type: ignore[index]
        self.assertNotEqual(located["html"], slots[0]["html"])  # type: ignore[index]
        with self.assertLogs("app.idm", level="INFO") as logs, self.assertRaises(IdmStageError) as caught:
            await self.run_fallback_only(body, [located, slots[1]])
        self.assertEqual(caught.exception.code, "ORCHESTRATION_V2_UNIT_FALLBACK_INVALID")
        reason = "coverage:HTML_SOURCE_LOCATOR@components[0]"
        self.assertTrue(any('"idm_unit_node_acceptance"' in line and reason in line for line in logs.output))
