"""QLT-5 D: an FAQ that digs deeper (plan 2026-10-13 §2.3, §4.3).

Measured on 100 FAQ items of 6 runs: 58% recall or yes/no fact checks, 5% what-if, about half of each answer's word
pairs straight from the source, 23% of the answers a near copy of one source sentence, an FAQ in 39 of 71 units. W4 now
plans an FAQ only where its unit has something to clarify, with a type, anchor facts and an origin per item; W5 writes
explanations grounded in those facts and its checks catch recall-only FAQs, copied answers and new numbers or names.

The provider is the local fake; nothing reaches the network.
"""

from __future__ import annotations

import dataclasses
import json
import unittest
from typing import Any

from app.idm.contracts import (
    IdmComponentDesignV1,
    IdmContentBlockV1,
    IdmFaqItemPlanV1,
    IdmJudgeFindingV1,
    IdmLessonDesignV1,
    IdmUnitBriefV1,
    IdmW3W4ModuleResponseV1,
    brief_hash_of,
)
from app.idm.faq_checks import (
    FAQ_NEW_FACT_CODE,
    FAQ_RESTATES_SOURCE_CODE,
    FAQ_TYPE_MIX_CODE,
    FaqChecker,
    _Fact,
    copies_one_fact,
    extra_recall_items,
    faq_writer_rules,
    lcs_length,
    question_type,
    type_mix_ok,
)
from app.idm.faq_plan import (
    FAQ_DROPPED_CODE,
    FAQ_PLANNED_CODE,
    FaqCandidate,
    faq_evidence,
    gate_and_plan_faq,
    plan_faq_items,
)
from app.idm.prompts import w1_section_prompt
from app.idm.qa import FAQ_VALUE_CRITERION, criteria_summary, settle_applicability
from tests.idm_golden import key, keys
from tests.idm_test_support import rehash_contract
from tests.test_idm_module_design import component, design_module, edited, scope_of
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
)

# lsn_003 unit 2 of the golden course: a decision practice (problem) and an FAQ over the common-mistake block cb_0011.
MISTAKE_KEYS = keys(8, 1, 3)
CRITERIA_KEYS = keys(6, 2, 6)
EXPLAINED = ("Vì lúc khách vừa gọi đến, bạn chưa biết đơn hàng có thuộc diện bồi thường hay không. Hứa trước rồi mới "
             "kiểm tra có thể buộc công ty chi trả sai. Hãy xác minh và phân loại trước, sau đó mới nói về hướng xử "
             "lý.")
COPIED = ("Nhân viên hứa hoàn tiền ngay khi khách hàng vừa gọi đến, trước khi xác minh đơn hàng; công ty phải chi trả "
          "cho một đơn hàng không thuộc diện bồi thường.")
# Reasoning the facts do not word: an inferred answer the SME checks (never a finding).
INFERRED = ("Một lời hứa vội vàng làm khách tin rằng tiền chắc chắn sẽ được trả. Khi phải rút lại lời hứa, niềm tin "
            "của khách giảm mạnh hơn cả lúc chưa hứa.")
WHAT_IF = ("Khi bỏ qua yếu tố truyền thông, một khiếu nại cấp 3 dễ bị ghi thành cấp 2 và không được escalate kịp. "
           "Vì vậy hãy kiểm tra cả an toàn, pháp lý và truyền thông trước khi chốt cấp độ.")


def faq_slot(*items: tuple[str, str], title: str = "Những nhầm lẫn thường gặp khi xử lý khiếu nại") -> dict[str, Any]:
    return {"type": "la_faq", "title": title, "items": [{"question": q, "answer": a} for q, a in items]}


def escalate_brief(mode: str | None = None) -> IdmUnitBriefV1:
    raw = escalate_body()["unit_contract"]["idm_unit_brief"]
    if mode is not None:
        raw["fidelity_mode"] = mode
    return IdmUnitBriefV1.model_validate(raw)


def checker(mode: str | None = None) -> FaqChecker:
    body = escalate_body()
    contract = body["unit_contract"]
    texts = {fact["fact_key"]: fact["fact_text"] for fact in contract["source_facts"]}
    texts.update({fact["fact_key"]: fact["fact_text"] for fact in contract["idm_unit_brief"]["lesson_context_facts"]})
    return FaqChecker.build(escalate_brief(mode), texts, plan_texts=["Quyết định tự xử lý hay escalate"])


class QuestionTypeTests(unittest.TestCase):
    def test_vietnamese_english_and_unaccented_questions(self) -> None:
        cases = {
            "Vì sao không được hứa bồi thường trước khi phân loại?": "why",
            "Vi sao phai escalate khieu nai an toan?": "why",
            "Nhằm mục đích gì mà phải phân loại trước?": "why",
            "Điều gì xảy ra nếu nhân viên hứa hoàn tiền ngay?": "what_if",
            "Neu khach VIP phan nan thi sao?": "what_if",
            "Khi khách nhắc đến kiện tụng thì nên làm gì?": "what_if",
            "Lỗi nào hay gặp nhất khi chốt cấp độ?": "common_mistake",
            "Vì sao khiếu nại cấp 3 hay bị ghi nhầm thành cấp 2?": "common_mistake",
            "Góp ý và khiếu nại khác nhau thế nào?": "comparison",
            "Làm thế nào để đánh giá mức độ nghiêm trọng?": "how_to_apply",
            "Khiếu nại được chia thành những nhóm nào?": "recall",
            "Khiếu nại cấp 2 phải phản hồi trong bao lâu?": "recall",
            "Thế nào là một khiếu nại?": "recall",
            "Những trường hợp nào phải escalate?": "recall",
            "Khi nào bắt buộc escalate?": "recall",
            # "nêu" (to state) is not "nếu" (if); "nhằm" (to aim) is not "nhầm" (a mistake).
            "Tài liệu nêu những gì về khiếu nại?": "recall",
            "Why must safety complaints be escalated?": "why",
            "What happens if I promise a refund before checking?": "what_if",
            "When a customer mentions a lawsuit, what should I do?": "what_if",
            "What is the most common mistake when grading a complaint?": "common_mistake",
            "How many complaint groups are there?": "recall",
            "How do I grade a complaint?": "how_to_apply",
            "What is a complaint?": "recall",
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                self.assertEqual(question_type(question), expected)

    def test_type_mix_allows_one_recall_and_needs_a_clarifying_question(self) -> None:
        self.assertTrue(type_mix_ok(["recall", "why"]))
        self.assertTrue(type_mix_ok(["what_if", "how_to_apply"]))
        self.assertFalse(type_mix_ok(["recall", "recall", "why"]))
        self.assertFalse(type_mix_ok(["how_to_apply", "comparison"]))
        self.assertEqual(extra_recall_items(["recall", "why", "recall", "recall"]), [2, 3])


class SourceCopyTests(unittest.TestCase):
    def test_longest_common_subsequence(self) -> None:
        self.assertEqual(lcs_length(["a", "b", "c", "d", "e"], ["a", "x", "c", "y", "e"]), 3)
        self.assertEqual(lcs_length([], ["a"]), 0)
        self.assertEqual(lcs_length(["a", "b"], ["z", "a", "q", "b", "r"]), 2)

    def test_an_answer_one_fact_holds_is_a_copy(self) -> None:
        facts = [_Fact.of(text) for text in (
            "Ví dụ: Nhân viên hứa hoàn tiền ngay khi khách hàng vừa gọi đến, trước khi xác minh đơn hàng; công ty phải "
            "chi trả cho một đơn hàng không thuộc diện bồi thường.",
            "Lưu ý: Nhiều nhân viên ghi nhận khiếu nại cấp 3 thành cấp 2 vì không kiểm tra yếu tố truyền thông.")]
        self.assertTrue(copies_one_fact(COPIED, facts))
        # The same words in order with a few inserted: the 5-grams break, the subsequence stays (>= 0.6).
        interleaved = ("Thực tế nhân viên hay hứa hoàn tiền ngay khi khách hàng vừa gọi, trước lúc xác minh đơn hàng, "
                       "nên công ty phải chi trả cho đơn hàng không thuộc diện bồi thường.")
        self.assertTrue(copies_one_fact(interleaved, facts))
        self.assertFalse(copies_one_fact(EXPLAINED, facts))
        self.assertFalse(copies_one_fact(WHAT_IF, facts))
        # Too short to tell.
        self.assertFalse(copies_one_fact("Nhân viên hứa hoàn tiền ngay.", facts))
        # Half of one fact and half of another is no single-fact copy.
        mixed = ("Nhân viên hứa hoàn tiền ngay khi khách hàng vừa gọi đến; nhiều nhân viên ghi nhận khiếu nại cấp 3 "
                 "thành cấp 2.")
        self.assertFalse(copies_one_fact(mixed, facts))


class FaqCheckerTests(unittest.TestCase):
    def test_the_golden_plan_reaches_the_brief(self) -> None:
        slot = escalate_brief().components[1]
        self.assertEqual([(item.question_type, item.origin) for item in slot.faq_items],
                         [("common_mistake", "source"), ("what_if", "source"), ("common_mistake", "distractor")])
        self.assertEqual(slot.faq_items[2].anchor_fact_keys, keys(6, 2, 4))

    def test_basis_is_the_item_anchors_and_the_slot_facts(self) -> None:
        check = checker()
        self.assertEqual(check.basis_keys(1, 1), [key(8, 3), *[k for k in MISTAKE_KEYS if k != key(8, 3)]])
        self.assertEqual(set(check.basis_keys(1, 2)), {*keys(6, 2, 4), *MISTAKE_KEYS})
        # An item past the plan may use every planned anchor.
        self.assertEqual(set(check.basis_keys(1, 5)), {*keys(8, 2, 3), *keys(6, 2, 4), key(8, 1)})
        self.assertIn("items[2] common_mistake on [d1-c6-f2], [d1-c6-f3], [d1-c6-f4].", check.plan_hint(1))
        self.assertEqual(check.plan_hint(0), "")

    def test_a_number_its_own_facts_do_not_state_is_a_new_fact(self) -> None:
        check = checker()
        fifty = "Thiệt hại từ 50 triệu đồng trở lên là trường hợp bắt buộc escalate, nên đừng tự xử lý một mình."
        component = faq_slot(("Vì sao không được hứa hoàn tiền ngay?", EXPLAINED),
                             ("Điều gì xảy ra nếu bỏ qua yếu tố truyền thông?", fifty),
                             ("Điều gì dễ nhầm khi thiệt hại lớn?", fifty))
        # Item 1 is planned on the communication fact only; item 2 on the escalation criteria that state 50 triệu.
        self.assertEqual(check.with_new_facts(1, component), [1])
        self.assertEqual(check.findings(1, component), [FAQ_NEW_FACT_CODE])
        named = faq_slot(("Vì sao không được hứa hoàn tiền ngay?", EXPLAINED + " Chị Lan ở Hà Nội đã gặp lỗi này."),
                         ("Điều gì xảy ra nếu bỏ qua yếu tố truyền thông?", WHAT_IF))
        self.assertEqual(check.with_new_facts(1, named), [0])

    def test_copies_are_measured_outside_the_keep_mode(self) -> None:
        component = faq_slot(("Điều gì dễ nhầm khi khách đòi hoàn tiền?", COPIED),
                             ("Điều gì xảy ra nếu bỏ qua yếu tố truyền thông?", WHAT_IF))
        # The golden course is compliance (keep): an answer may quote its rule.
        self.assertEqual(checker().restated(component), [])
        self.assertEqual(checker("explain").restated(component), [0])
        self.assertEqual(checker("structured").findings(1, component), [FAQ_RESTATES_SOURCE_CODE])

    def test_type_mix_and_inferred_items(self) -> None:
        check = checker()
        recall = faq_slot(("Lỗi thường gặp gồm những gì?", EXPLAINED), ("Khiếu nại cấp 3 là gì?", WHAT_IF))
        self.assertEqual(question_type("Lỗi thường gặp gồm những gì?"), "common_mistake")
        self.assertEqual(check.findings(1, recall), [])
        only_recall = faq_slot(("Khiếu nại cấp 3 là gì?", EXPLAINED), ("Cấp 2 gồm những dấu hiệu nào?", WHAT_IF))
        self.assertEqual(check.findings(1, only_recall), [FAQ_TYPE_MIX_CODE])
        component = faq_slot(("Vì sao không được hứa hoàn tiền ngay?", INFERRED),
                             ("Có thể ghi nhận cấp 3 thành cấp 2 không?",
                              "Nhiều nhân viên ghi nhận khiếu nại cấp 3 thành cấp 2 vì không kiểm tra yếu tố truyền "
                              "thông."))
        # The first answer explains beyond the words of its facts (for the SME); the second restates them.
        self.assertEqual(check.inferred(1, component), [0])
        self.assertEqual(check.inferred_counts([{}, component]), {1: 1})
        self.assertEqual(check.inferred_counts([{}, component], exempt=[1]), {})
        self.assertEqual(check.metrics([{}, component]), {"c1": {"planned": ["common_mistake", "what_if",
                                                                            "common_mistake"],
                                                                "written": ["why", "recall"], "inferred": 1}})

    def test_writer_rules_never_ask_to_restate(self) -> None:
        for mode in ("keep", "structured", "explain"):
            rules = faq_writer_rules(mode)  # type: ignore[arg-type]
            self.assertIn("At most ONE plain recall", rules)
            self.assertNotIn("restate only", rules)
            self.assertEqual("CONTENT MODE keep" in rules, mode == "keep")


class FaqPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scope = scope_of(1)
        self.lesson = IdmLessonDesignV1.model_validate(edited("mod_02", lambda _p: None)["lessons"][0])

    def test_evidence_of_the_golden_faq_unit(self) -> None:
        evidence = faq_evidence(self.lesson.units[1], self.lesson, self.scope)
        self.assertEqual(evidence[0], FaqCandidate("common_mistake", tuple(keys(6, 2, 4)), "distractor"))
        self.assertIn(FaqCandidate("common_mistake", (key(8, 2), key(8, 3)), "source"), evidence)
        self.assertIn(FaqCandidate("what_if", (key(8, 3),), "source"), evidence)
        # The decision block of unit 1 (cb_0009) gives a "why" on its rule facts.
        teach = faq_evidence(self.lesson.units[0], self.lesson, self.scope)
        self.assertEqual(teach[-1], FaqCandidate("why", tuple(keys(6, 2, 4)), "source"))

    def test_plan_takes_one_clarifying_item_per_type_then_fills_to_two(self) -> None:
        mistake = FaqCandidate("common_mistake", ("a",), "ai_drafted")
        source = FaqCandidate("common_mistake", ("b",), "source")
        what_if = FaqCandidate("what_if", ("c",), "source")
        why = FaqCandidate("why", ("d",), "source")
        filler = FaqCandidate("how_to_apply", ("e",), "source")
        plan = plan_faq_items([mistake, what_if, source, why], [filler])
        self.assertEqual([(item.question_type, item.anchor_fact_keys) for item in plan],
                         [("common_mistake", ["b"]), ("what_if", ["c"]), ("why", ["d"])])
        self.assertEqual([item.question_type for item in plan_faq_items([why], [filler, filler])],
                         ["why", "how_to_apply"])
        self.assertEqual(plan_faq_items([why, why], [filler]), [why.plan(), filler.plan()])

    def test_a_unit_with_nothing_to_clarify_loses_its_faq(self) -> None:
        scope = scope_of(2)
        lesson = IdmLessonDesignV1.model_validate(edited("mod_03", lambda p: p["lessons"][0]["units"][0][
            "components"].append(component("la_faq", "clarify", ["cb_0007"])))["lessons"][0])
        result = gate_and_plan_faq([lesson, *[IdmLessonDesignV1.model_validate(item) for item in
                                              edited("mod_03", lambda _p: None)["lessons"][1:]]], scope)
        self.assertEqual((result.dropped, result.planned), (1, 0))
        unit = result.lessons[0].units[0]
        self.assertEqual([(c.type, c.block_ids, c.component_index) for c in unit.components],
                         [("html", ["cb_0006", "cb_0007"], 1)])
        self.assertIn('Đã bỏ phần hỏi đáp của "Tám bước đầu của quy trình"', result.lessons[0].notes)

    def test_blocks_only_the_faq_used_join_the_teaching_html_and_the_note_follows_the_locale(self) -> None:
        scope = dataclasses.replace(scope_of(2), locale="en")

        def split(payload: dict[str, Any]) -> None:
            unit = payload["lessons"][0]["units"][0]
            unit["components"][0]["block_ids"] = ["cb_0006"]
            unit["components"].append(component("la_faq", "clarify", ["cb_0007"]))

        lesson = IdmLessonDesignV1.model_validate(edited("mod_03", split)["lessons"][0])
        result = gate_and_plan_faq([lesson], scope)
        self.assertEqual([c.block_ids for c in result.lessons[0].units[0].components], [["cb_0006", "cb_0007"]])
        self.assertIn('The FAQ of "Tám bước đầu của quy trình" was left out', result.lessons[0].notes)

    def test_an_faq_only_unit_becomes_its_teaching_html(self) -> None:
        scope = scope_of(2)

        def faq_only(payload: dict[str, Any]) -> None:
            payload["lessons"][0]["units"][0]["components"] = [component("la_faq", "clarify", ["cb_0006", "cb_0007"])]

        lesson = IdmLessonDesignV1.model_validate(edited("mod_03", faq_only)["lessons"][0])
        result = gate_and_plan_faq([lesson], scope)
        only = result.lessons[0].units[0].components
        self.assertEqual([(c.type, c.role, c.title) for c in only], [("html", "explain", "Tám bước đầu của quy trình")])
        no_html = gate_and_plan_faq([lesson], dataclasses.replace(scope, allowed_types=frozenset({"la_faq"})))
        self.assertEqual(no_html.dropped, 0)
        self.assertEqual(no_html.lessons[0].units[0].components[0].type, "la_faq")
        self.assertEqual([item.question_type for item in no_html.lessons[0].units[0].components[0].faq_items],
                         ["how_to_apply", "how_to_apply"])

    def test_issues_and_gaps_mark_a_likely_confusion(self) -> None:
        scope = scope_of(2)
        raw = scope.blocks["cb_0006"].model_dump(mode="json")
        raw["gaps"] = [{"type": "missing_common_mistake", "note": "Nguồn không nêu lỗi hay gặp."}]
        scope.blocks["cb_0006"] = IdmContentBlockV1.model_validate(raw)
        lesson = IdmLessonDesignV1.model_validate(edited("mod_03", lambda p: p["lessons"][0]["units"][0][
            "components"].append(component("la_faq", "clarify", ["cb_0007"])))["lessons"][0])
        evidence = faq_evidence(lesson.units[0], lesson, scope)
        self.assertEqual([(item.question_type, item.origin) for item in evidence], [("common_mistake", "ai_drafted")])
        result = gate_and_plan_faq([lesson], scope)
        self.assertEqual(result.planned, 1)
        self.assertEqual([item.question_type for item in result.lessons[0].units[0].components[1].faq_items],
                         ["common_mistake", "how_to_apply"])


class ContractTests(unittest.TestCase):
    def test_the_plan_is_server_owned_and_absent_when_empty(self) -> None:
        self.assertNotIn("faq_items", json.dumps(IdmW3W4ModuleResponseV1.model_json_schema()))
        raw = component("la_faq", "clarify", ["cb_0011"])
        self.assertNotIn("faq_items", IdmComponentDesignV1.model_validate(raw).model_dump(mode="json"))
        planned = {**raw, "faq_items": [{"question_type": "why", "anchor_fact_keys": [key(8, 2)], "origin": "source"}]}
        self.assertEqual(IdmComponentDesignV1.model_validate(planned).model_dump(mode="json")["faq_items"],
                         planned["faq_items"])

    def test_briefs_with_and_without_the_faq_fields_read(self) -> None:
        raw = escalate_body()["unit_contract"]["idm_unit_brief"]
        self.assertEqual(raw["faq_material"]["blocks"][0]["support_role"], "common_mistake")
        self.assertEqual(raw["faq_material"]["practices"][0]["feedback_focus"]["criterion"],
                         "Tiêu chí của quy trình")
        self.assertNotIn("faq_items", raw["components"][0])
        legacy = {k: v for k, v in raw.items() if k != "faq_material"}
        legacy["components"] = [{k: v for k, v in slot.items() if k != "faq_items"} for slot in raw["components"]]
        legacy["brief_hash"] = brief_hash_of(legacy)
        brief = IdmUnitBriefV1.model_validate(legacy)
        self.assertIsNone(brief.faq_material)
        self.assertEqual(brief.components[1].faq_items, [])
        self.assertEqual(IdmFaqItemPlanV1.model_validate(raw["components"][1]["faq_items"][0]).origin, "source")


class JudgeTests(unittest.TestCase):
    def test_faq_value_is_stored_under_not_copied_and_needs_an_faq(self) -> None:
        finding = IdmJudgeFindingV1(criterion=FAQ_VALUE_CRITERION, severity="major", component_index=1, witness="w")
        self.assertEqual(criteria_summary([finding]), {"Q2_not_copied": "major"})
        self.assertEqual(settle_applicability([finding], practice_slot=True, faq_slot=False)[0].severity,
                         "not_applicable")
        self.assertEqual(settle_applicability([finding], practice_slot=True)[0].severity, "major")


class W1PromptTests(unittest.TestCase):
    def test_w1_asks_for_mistakes_and_records_a_gap_otherwise(self) -> None:
        prompt = w1_section_prompt("vi", project_context={}, section_id="sec_001", section_title_path="A",
                                   facts=[("f1", "Một dòng", None)])
        self.assertIn("Look explicitly for common mistakes, misconceptions and exceptions", prompt)
        self.assertIn("record the gap missing_common_mistake", prompt)


def explain_escalate_body() -> dict[str, Any]:
    body = escalate_body()
    brief = body["unit_contract"]["idm_unit_brief"]
    brief.update(fidelity_mode="explain", purpose_profile="onboarding",
                 purpose_summary="Giúp nhân viên mới biết cách công ty tiếp nhận và xử lý khiếu nại.")
    brief["brief_hash"] = brief_hash_of(brief)
    rehash_contract(body["unit_contract"])
    return body


def faq_writer(body: dict[str, Any], *items: tuple[str, str]) -> dict[str, Any]:
    writer = escalate_writer(body)
    writer["components"]["c1"]["items"] = [{"question": q, "answer": a} for q, a in items]
    return writer


class UnitFaqTests(StoryboardEndpointTestCase):
    async def test_writer_gets_the_plan_the_material_and_the_rules(self) -> None:
        body = escalate_body()
        provider = FakeGenerate(**{WRITER: [escalate_writer(body)], JUDGE: [judge()]})
        with self.assertLogs("app.idm", "INFO") as logs:
            status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        self.assert_envelope(data, "provider_validated", "validated", "provider")
        prompt = provider.calls[0]["prompt"]
        self.assertIn('"faq_items":[{"question_type":"common_mistake","anchor_fact_keys":["d1-c8-f2","d1-c8-f3"],'
                      '"origin":"source"}', prompt)
        self.assertIn('"faq_material":{"blocks":[{"block_id":"cb_0011"', prompt)
        self.assertIn("An la_faq slot digs deeper than the html", prompt)
        self.assertIn("CONTENT MODE keep: an answer may quote its rule word for word", prompt)
        judge_prompt = provider.calls[1]["prompt"]
        self.assertIn('"has_faq_slot":true', judge_prompt)
        self.assertIn("Q13_faq_value", judge_prompt)
        quality_line = next(line for line in logs.output if '"event": "idm_unit_quality"' in line)
        self.assertIn('"faq": {"c1": {"inferred": 0, "planned": ["common_mistake", "what_if", "common_mistake"], '
                      '"written": ["recall", "common_mistake"]}}', quality_line)
        # A unit without an FAQ gets no FAQ rules.
        body = severity_body()
        provider = FakeGenerate(**{WRITER: [severity_writer(body)], JUDGE: [judge()]})
        await self.post(body, provider)
        self.assertNotIn("An la_faq slot digs deeper", provider.calls[0]["prompt"])
        self.assertIn('"has_faq_slot":false', provider.calls[1]["prompt"])

    async def test_recall_only_faq_gets_a_targeted_repair(self) -> None:
        body = escalate_body()
        recall = faq_writer(body, ("Khiếu nại cấp 3 là gì trong xử lý khiếu nại?", EXPLAINED),
                            ("Khiếu nại cấp 2 gồm những dấu hiệu nào?", WHAT_IF))
        fixed = faq_writer(body, ("Vì sao không nên hứa hoàn tiền khi khách khiếu nại vừa gọi đến?", EXPLAINED),
                           ("Điều gì xảy ra nếu xử lý khiếu nại mà bỏ qua yếu tố truyền thông?", WHAT_IF))
        provider = FakeGenerate(**{WRITER: [recall], REPAIR: [slot_repair(fixed, 1)], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "validated", "provider")
        self.assertEqual(provider.calls[1]["thinking_level"], "low")
        self.assertIn("c1 IDM_W5_FAQ_TYPE_MIX at c1.items: the questions mostly ask for facts again", provider.calls[1][
            "prompt"])
        self.assertIn("(2 recall question(s), 0 why, what-if or common-mistake question(s)) Plan: items[0]",
                      provider.calls[1]["prompt"])
        self.assertEqual(quality.deterministic_codes, [FAQ_TYPE_MIX_CODE])
        self.assertIn("Đã tự sửa: IDM_W5_FAQ_TYPE_MIX.", quality.author_note)

    async def test_extra_recall_items_are_dropped_when_a_clarifying_one_stays(self) -> None:
        body = escalate_body()
        items = (("Khiếu nại cấp 3 là gì trong xử lý khiếu nại?", EXPLAINED),
                 ("Khiếu nại cấp 2 gồm những dấu hiệu nào?", WHAT_IF),
                 ("Vì sao không nên hứa hoàn tiền khi khách khiếu nại vừa gọi đến?", EXPLAINED + " Hãy bình tĩnh."))
        writer = faq_writer(body, *items)
        provider = FakeGenerate(**{WRITER: [writer], REPAIR: [slot_repair(writer, 1)], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "validated", "provider")
        self.assertEqual([item["question"] for item in data["unit"]["components"][1]["items"]],
                         [items[0][0], items[2][0]])
        self.assertIn("Đã bỏ 1 câu hỏi đáp chỉ hỏi lại kiến thức đã học.", quality.author_note)

    async def test_copied_answer_is_repaired_and_never_replaced_by_the_source(self) -> None:
        body = explain_escalate_body()
        copied = faq_writer(body, ("Điều gì dễ nhầm khi khách khiếu nại đòi hoàn tiền ngay?", COPIED),
                            ("Điều gì xảy ra nếu xử lý khiếu nại mà bỏ qua yếu tố truyền thông?", WHAT_IF))
        fixed = faq_writer(body, ("Điều gì dễ nhầm khi khách khiếu nại đòi hoàn tiền ngay?", INFERRED),
                           ("Điều gì xảy ra nếu xử lý khiếu nại mà bỏ qua yếu tố truyền thông?", WHAT_IF))
        provider = FakeGenerate(**{WRITER: [copied], REPAIR: [slot_repair(fixed, 1)], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "validated", "provider")
        self.assertIn("c1 IDM_W5_FAQ_RESTATES_SOURCE at c1.items[0]: the listed answers repeat a source sentence",
                      provider.calls[1]["prompt"])
        self.assertEqual(data["unit"]["components"][1]["items"][0]["answer"], INFERRED)
        self.assertIn("Câu trả lời hỏi đáp có phần AI suy luận từ tài liệu — SME cần kiểm tra không có quy định mới",
                      quality.author_note)
        # Still copied after the repair, with too few items to drop it: kept for review, never the source FAQ.
        provider = FakeGenerate(**{WRITER: [copied], REPAIR: [slot_repair(copied, 1)], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        quality = self.assert_envelope(data, "provider_validated", "review_required", "provider")
        self.assertNotIn("source_locked_fallback", data["unit"]["components"][1])
        self.assertIn("Cần xem khối 2 (Hỏi đáp): câu trả lời hỏi đáp gần như chép nguyên câu tài liệu",
                      quality.author_note)

    async def test_the_judge_faq_criterion_is_stored_under_not_copied(self) -> None:
        body = escalate_body()
        provider = FakeGenerate(**{WRITER: [escalate_writer(body)],
                                   JUDGE: [judge(("Q13_faq_value", "major", 1))]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        quality = self.assert_envelope(data, "provider_validated", "review_required", "provider")
        self.assertEqual(quality.criteria["Q2_not_copied"], "major")
        self.assertIn("phần hỏi đáp chỉ nhắc lại nội dung, chưa giải thích vì sao hay lỗi hay gặp (Q13)",
                      quality.author_note)


class ModuleDesignFaqTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_kept_faq_carries_its_plan_and_a_dropped_one_is_counted(self) -> None:
        result, provider, runtime = await design_module(1, [edited("mod_02", lambda _p: None)])
        design = result["shard"]["idm_design"]
        faq = design["lessons"][0]["units"][1]["components"][1]
        self.assertEqual([item["question_type"] for item in faq["faq_items"]],
                         ["common_mistake", "what_if", "common_mistake"])
        self.assertNotIn("faq_items", design["lessons"][0]["units"][1]["components"][0])
        self.assertEqual(runtime.adjustments[FAQ_PLANNED_CODE], 1)

        def add_faq(payload: dict[str, Any]) -> None:
            payload["lessons"][0]["units"][0]["components"].append(component("la_faq", "clarify", ["cb_0007"]))

        result, provider, runtime = await design_module(2, [edited("mod_03", add_faq)])
        self.assertEqual((len(provider.calls), result["shard"]["idm_design"]["stage_origin"]), (1, "provider"))
        self.assertEqual(runtime.adjustments[FAQ_DROPPED_CODE], 1)
        self.assertEqual([c["type"] for c in result["shard"]["lessons"][0]["units"][0]["component_plan"]], ["html"])
