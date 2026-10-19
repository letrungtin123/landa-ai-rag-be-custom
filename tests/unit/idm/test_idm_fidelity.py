"""QLT-5 C content modes: resolution, W2 treatment policy, module supports, writer rules, budget and the W5 checks.

Pure functions only; the unit endpoint behaviour is in ``test_idm_fidelity_unit.py``. Nothing reaches the network.
"""

from __future__ import annotations

import types
import unittest
from typing import Any

from app.idm.blueprint import run_w2, validate_w2
from app.idm.contracts import (
    IdmBlueprintRowV1,
    IdmContentBlockV1,
    IdmProjectContextV1,
    IdmPurposeV1,
    IdmUnitBriefV1,
    IdmW1ReduceResponseV1,
    IdmW2BlueprintResponseV1,
)
from app.idm.fidelity import (
    TEACHING_SUPPORT_KINDS,
    budget_words,
    description_asks_to_keep,
    enforce_treatments,
    fidelity_note_line,
    module_fidelity_rules,
    plan_teaching_supports,
    purpose_mode,
    resolve_fidelity,
    slot_marks,
    w2_treatment_rules,
    writer_fidelity_rules,
)
from app.idm.fidelity_checks import (
    EXPLANATION_MISSING_CODE,
    NEW_FACT_CODE,
    SENTENCE_COPY_CODE,
    EvidenceGrams,
    FidelityChecker,
    carries_exact_claim,
    copy_report,
    explained,
    words_of,
)
from app.idm.html_rules import DENSITY_CODE, HtmlRuleViolation
from app.idm.module_design import run_idm_module_design
from app.idm.prompts import html_slot_rules, html_violation_line, module_prompt, unit_writer_prompt, w2_blueprint_prompt
from app.idm.purpose import resolve_purpose
from app.idm.qa import EXPLAINED_CRITERION, build_unit_author_note, criteria_summary, deterministic_slot_findings
from app.idm.qa import JudgeOutcome as Judge
from app.idm.storyboard import output_budget
from app.idm.text import evidence_index
from tests import idm_golden as g
from tests.support.idm_test_support import golden_design, golden_shard, make_runtime, module_inputs, module_response
from tests.unit.idm.test_idm_blueprint import golden_state, runtime

FACTS = {
    "f1": "Công ty Cổ phần Nesso được thành lập năm 2012 tại Thành phố Hồ Chí Minh.",
    "f2": "Sứ mệnh của công ty là mang đến giải pháp đào tạo giúp doanh nghiệp phát triển đội ngũ.",
    "f3": "Giá trị cốt lõi gồm tận tâm, sáng tạo và hợp tác trong mọi dự án với khách hàng.",
    "f4": "Nhân viên mới phải hoàn thành khoá hội nhập trong 14 ngày đầu tiên làm việc.",
    "f5": "Mỗi phòng ban có một người phụ trách hỗ trợ nhân viên mới khi cần giải đáp thắc mắc về quy trình nội bộ.",
}
COPIED_VALUES = "Giá trị cốt lõi gồm tận tâm, sáng tạo và hợp tác trong mọi dự án với khách hàng."
COPIED_SUPPORT = ("Mỗi phòng ban có một người phụ trách hỗ trợ nhân viên mới khi cần giải đáp thắc mắc về quy trình "
                  "nội bộ.")
NOVEL = "Bạn sẽ gặp các giá trị này hằng ngày khi làm việc cùng đồng nghiệp ở mọi phòng ban."
NOVEL_2 = "Khi chưa rõ việc gì, bạn cứ hỏi người phụ trách của phòng mình trước tiên."
EXACT_RULE = "Nhân viên mới phải hoàn thành khoá hội nhập trong 14 ngày đầu tiên làm việc."
EXACT_DEFINITION = "Sứ mệnh của công ty là mang đến giải pháp đào tạo giúp doanh nghiệp phát triển đội ngũ."
PLAN_ID = "cp2_" + "a" * 32


def brief(mode: str | None, *, treatments: tuple[str, ...] = ("condense",), segment: str = "context_explain",
          role: str = "explain") -> IdmUnitBriefV1:
    """One teaching html slot (facts f1-f5) and one practice question."""

    return IdmUnitBriefV1.model_validate({
        "pipeline_version": "idm-1", "course_title": "Hội nhập Nesso", "target_audience": "Nhân viên mới",
        "purpose_profile": "onboarding", "purpose_summary": "Giúp nhân viên mới làm quen với công ty.",
        "fidelity_mode": mode, "module_title": "Công ty là ai", "lesson_title": "Giá trị cốt lõi",
        "lesson_objective": "Giới thiệu giá trị cốt lõi của công ty", "lesson_practice_sentences": [],
        "previous_lesson_title": None, "next_lesson_title": None, "unit_segment": segment,
        "unit_purpose": "Giới thiệu công ty với nhân viên mới.",
        "components": [
            {"component_plan_id": PLAN_ID, "type": "html", "role": role, "title": "Công ty và giá trị cốt lõi",
             "support_items": [], "practice": None,
             "treatments": [{"block_id": f"cb_000{index}", "treatment": treatment, "detail_level": "Giữ ý chính."}
                            for index, treatment in enumerate(treatments, start=1)],
             "owned_fact_keys": list(FACTS), "supporting_fact_keys": []},
            {"component_plan_id": "cp2_" + "b" * 32, "type": "problem", "role": "practice", "title": "Kiểm tra nhanh",
             "support_items": [], "practice": None, "treatments": [], "owned_fact_keys": [],
             "supporting_fact_keys": list(FACTS)},
        ],
        "lesson_context_facts": [], "job_aid_signpost": None, "brief_hash": "0" * 64,
    })


def checker(mode: str | None, **options: Any) -> FidelityChecker:
    unit_brief = brief(mode, **options)
    return FidelityChecker.build(unit_brief, FACTS, evidence_index(FACTS.values()), locale="vi",
                                 plan_texts=[unit_brief.course_title, unit_brief.lesson_title])


def html(*texts: str, heading: str = "Công ty mong đợi gì ở bạn?") -> dict[str, Any]:
    return {"type": "html", "semantic_content": {"sections": [{"heading": heading, "blocks": [
        {"kind": "paragraph", "text": text, "items": [], "rows": []} for text in texts]}]}}


class ResolutionTests(unittest.TestCase):
    def test_description_asks_to_keep_only_for_clear_wording_requests(self) -> None:
        for text in ("Giữ nguyên câu chữ của quy định.", "Trình bày nguyên văn nội quy công ty",
                     "Khoá học cần giữ đúng câu chữ của SOP", "Keep the wording verbatim.",
                     "Không thay đổi câu chữ của chính sách."):
            with self.subTest(text=text):
                self.assertTrue(description_asks_to_keep(text))
        for text in (None, "", "Giúp nhân viên mới giữ nguyên giá trị cốt lõi của công ty.",
                     "Không cần giữ nguyên văn, hãy diễn giải dễ hiểu.", "No need to be verbatim, explain it."):
            with self.subTest(text=text):
                self.assertFalse(description_asks_to_keep(text))

    def test_the_profile_decides_and_a_wording_request_only_tightens_to_keep(self) -> None:
        self.assertEqual(resolve_fidelity("explain", "Khoá học hội nhập cho nhân viên mới."), "explain")
        self.assertEqual(resolve_fidelity("explain", "Hội nhập — giữ nguyên câu chữ của nội quy."), "keep")
        self.assertEqual(resolve_fidelity("structured", None), "structured")
        self.assertEqual(resolve_fidelity("keep", "Hãy diễn giải thật dễ hiểu."), "keep")

    def test_resolve_purpose_stores_the_mode_and_older_purposes_read_as_structured(self) -> None:
        base = {"locale": "vi", "source_documents": [{"document_id": "d1", "name": "tai-lieu.pdf", "type": "pdf"}],
                "target_audience": None, "learning_objectives": [], "duration_target_minutes": None}

        def mode_of(title: str, description: str | None) -> str | None:
            context = IdmProjectContextV1.model_validate({**base, "course_title_hint": title,
                                                          "course_description": description})
            return resolve_purpose(context, None)[0].fidelity_mode

        self.assertEqual(mode_of("Onboarding nhân viên mới", None), "explain")
        self.assertEqual(mode_of("Onboarding nhân viên mới", "Giữ nguyên câu chữ của nội quy công ty."), "keep")
        self.assertEqual(mode_of("Quy định an toàn lao động", None), "keep")
        self.assertEqual(mode_of("Kỹ năng bán hàng qua điện thoại", None), "structured")
        reduce = IdmW1ReduceResponseV1.model_validate({**g.W1_REDUCE, "purpose_profile": "product",
                                                       "purpose_summary": "Giúp nhân viên tư vấn đúng sản phẩm."})
        context = IdmProjectContextV1.model_validate({**base, "course_title_hint": "Khoá mới",
                                                      "course_description": None})
        self.assertEqual(resolve_purpose(context, reduce)[0].fidelity_mode, "explain")
        stored = IdmPurposeV1.model_validate({"profile": "onboarding", "summary": "Giúp nhân viên mới hội nhập.",
                                              "confidence": "high", "origin": "author_intent", "evidence": ["title"],
                                              "signals": [], "template_gaps": []})
        self.assertIsNone(stored.fidelity_mode)
        self.assertEqual(purpose_mode(stored), "structured")
        self.assertEqual(purpose_mode(None), "structured")


def block(block_id: str, kind: str) -> IdmContentBlockV1:
    return IdmContentBlockV1.model_validate({
        "block_id": block_id, "section_id": "sec_001", "name": f"Khối {block_id}", "summary": "Tóm tắt.",
        "intent": "know", "support_role": None, "content_kind": kind, "fact_keys": [block_id], "issues": [],
        "gaps": [], "sme_questions": [], "origin": "provider"})


def row(block_id: str, classification: str, treatment: str) -> IdmBlueprintRowV1:
    placement = {"must_do": "course", "must_know": "course", "reference": "reference_job_aid"}.get(classification,
                                                                                                  "excluded")
    return IdmBlueprintRowV1.model_validate({
        "block_id": block_id, "lo_id": "lo_1", "must_do_ids": ["md_1"] if placement == "course" else [],
        "classification": classification, "placement": placement, "treatment": treatment, "detail_level": "Giữ.",
        "hold": False, "rationale": "Lý do."})


class TreatmentPolicyTests(unittest.TestCase):
    BLOCKS = (block("cb_0001", "concept"), block("cb_0002", "policy"), block("cb_0003", "mindset"),
              block("cb_0004", "procedure"), block("cb_0005", "concept"), block("cb_0006", "reference"))
    ROWS = (row("cb_0001", "must_know", "condense"), row("cb_0002", "must_know", "rewrite"),
            row("cb_0003", "must_know", "explain"), row("cb_0004", "must_do", "explain"),
            row("cb_0005", "must_know", "keep"), row("cb_0006", "reference", "convert_to_job_aid"))

    def treatments(self, mode: str) -> tuple[list[str], int]:
        rows, changed = enforce_treatments(self.ROWS, self.BLOCKS, mode)  # type: ignore[arg-type]
        return [item.treatment for item in rows], changed

    def test_explain_mode_explains_must_know_concepts_and_keeps_rules_and_exact_wording(self) -> None:
        self.assertEqual(self.treatments("explain"),
                         (["explain", "rewrite", "explain", "explain", "keep", "convert_to_job_aid"], 1))

    def test_structured_mode_never_explains(self) -> None:
        self.assertEqual(self.treatments("structured"),
                         (["condense", "rewrite", "rewrite", "rewrite", "keep", "convert_to_job_aid"], 2))

    def test_keep_mode_never_rewrites_or_explains_a_rule(self) -> None:
        self.assertEqual(self.treatments("keep"),
                         (["condense", "keep", "condense", "keep", "keep", "convert_to_job_aid"], 3))

    def test_explain_is_a_valid_course_treatment_and_never_costs_a_repair(self) -> None:
        state = golden_state()
        rows = [{**item, "treatment": "explain"} if item["classification"] == "must_know" else item
                for item in g.W2_BLUEPRINT["rows"]]
        response = IdmW2BlueprintResponseV1.model_validate({**g.W2_BLUEPRINT, "rows": rows})
        self.assertEqual(validate_w2(response, state.blocks, state.objectives, state.must_dos), [])

    def test_w2_prompt_defines_every_treatment_and_the_mode(self) -> None:
        for mode in ("explain", "structured", "keep"):
            with self.subTest(mode=mode):
                prompt = w2_blueprint_prompt("vi", project_context={}, objectives=[], must_dos=[], catalog=[],
                                             treatment_rules=w2_treatment_rules(mode))  # type: ignore[arg-type]
                for name in ("keep =", "condense =", "rewrite =", "explain =", "combine =", "separate ="):
                    self.assertIn(name, prompt)
                self.assertIn(f"CONTENT MODE {mode}", prompt)
                self.assertNotIn("MINIMUM level of detail", prompt)
        default = w2_blueprint_prompt("vi", project_context={}, objectives=[], must_dos=[], catalog=[])
        self.assertIn("CONTENT MODE structured", default)


async def run_w2_in(mode: str) -> Any:
    """The golden W2 answer through ``run_w2`` with a purpose in ``mode``; (result, W2 prompt)."""

    state = golden_state()
    purpose = IdmPurposeV1.model_validate({"profile": "onboarding", "summary": "Giúp nhân viên mới hội nhập.",
                                           "confidence": "high", "origin": "author_intent", "evidence": ["title"],
                                           "signals": [], "template_gaps": [], "fidelity_mode": mode})
    provider = g.FakeIdmProvider({"IdmW2BlueprintResponseV1": [g.W2_BLUEPRINT]})
    result = await run_w2(runtime(provider), context=state.context, blocks=state.blocks, links=state.links,
                          objectives=state.objectives, must_dos=state.must_dos, index=state.index,
                          tail_reserve_tokens=0, purpose=purpose)
    return result, provider.calls[0]["prompt"]


class RunW2ModeTests(unittest.IsolatedAsyncioTestCase):
    async def test_explain_mode_rows_and_prompt(self) -> None:
        result, prompt = await run_w2_in("explain")
        self.assertIn("CONTENT MODE explain", prompt)
        treatments = {item.block_id: item.treatment for item in result.rows}
        # cb_0003 "Thế nào là một khiếu nại?" (concept, condensed by W2) is explained; the decision table keeps its
        # wording and the procedures keep theirs.
        self.assertEqual((treatments["cb_0003"], treatments["cb_0005"], treatments["cb_0006"]),
                         ("explain", "keep", "condense"))
        self.assertGreaterEqual(result.codes["IDM_W2_TREATMENT_POLICY"], 1)


class ModuleSupportTests(unittest.IsolatedAsyncioTestCase):
    def test_module_rules_only_in_explain_mode(self) -> None:
        self.assertEqual(module_fidelity_rules("structured", "vi"), "")
        self.assertEqual(module_fidelity_rules("keep", "vi"), "")
        rules = module_fidelity_rules("explain", "vi")
        self.assertIn("support item of kind example, worked_example, non_example", rules)
        self.assertIn("[AI soạn — cần SME xác nhận]", rules)
        prompt = module_prompt("vi", module_plan={}, lesson_plans=[], audience="Nhân viên mới", objectives=[],
                               must_dos=[], course_blocks=[], facts=[], allowed_components=["html"],
                               fidelity_rules=rules)
        self.assertIn("CONTENT MODE explain: every teaching html component", prompt)

    def test_every_teaching_html_plans_an_example_in_explain_mode(self) -> None:
        lessons = golden_shard(0).lessons
        names = {"cb_0004": "Khiếu nại được chia thành những nhóm nào?"}
        unchanged, none = plan_teaching_supports(lessons, names, "structured", "vi")
        self.assertEqual((unchanged, none), (list(lessons), 0))
        planned, added = plan_teaching_supports(lessons, names, "explain", "vi")
        self.assertGreater(added, 0)
        for lesson in planned:
            for unit in lesson.units:
                for component in unit.components:
                    teaching = component.type == "html" and component.role not in {"practice", "job_aid"}
                    has = any(item.kind in TEACHING_SUPPORT_KINDS for item in component.support_items)
                    self.assertTrue(has or not teaching, component.title)
        drafted = [item for lesson in planned for unit in lesson.units for component in unit.components
                   for item in component.support_items if item.brief.startswith("[AI soạn")]
        self.assertTrue(drafted)
        self.assertTrue(all(item.kind == "example" and "Ví dụ minh hoạ" in item.brief for item in drafted))
        # Practice components keep their plan.
        self.assertEqual([component.support_items for lesson in planned for unit in lesson.units
                          for component in unit.components if component.role == "practice"],
                         [component.support_items for lesson in lessons for unit in lesson.units
                          for component in unit.components if component.role == "practice"])

    async def test_module_stage_follows_the_design_mode(self) -> None:
        context, skeleton_plan, facts = module_inputs(0)
        assert context.purpose is not None
        explained_context = context.model_copy(update={"purpose": context.purpose.model_copy(
            update={"fidelity_mode": "explain"})})
        _design, skeleton = golden_design()
        response = module_response(context.module.module_key)
        provider = g.FakeIdmProvider({"IdmW3W4ModuleResponseV1": [response]})
        result = await run_idm_module_design(context=explained_context, skeleton=skeleton, plan=skeleton_plan,
                                             facts=facts, runtime=make_runtime(provider))
        self.assertIn("CONTENT MODE explain", provider.calls[0]["prompt"])
        lessons = result["shard"]["idm_design"]["lessons"]
        teaching = [component for lesson in lessons for unit in lesson["units"] for component in unit["components"]
                    if component["type"] == "html" and component["role"] not in {"practice", "job_aid"}]
        self.assertTrue(teaching)
        self.assertTrue(all(any(item["kind"] in TEACHING_SUPPORT_KINDS for item in component["support_items"])
                            for component in teaching))


WRITER_ARGS: dict[str, Any] = {"course_title": "Khoá", "audience": "Nhân viên", "lesson_title": "Mục",
                               "lesson_objective": "Làm", "practice_sentences": [], "previous_title": None,
                               "next_title": None, "unit_brief": {}, "facts": [], "context_facts": []}


class WriterRuleTests(unittest.TestCase):

    def test_writer_rules_per_mode(self) -> None:
        explain = unit_writer_prompt("vi", **WRITER_ARGS, fidelity_rules=writer_fidelity_rules("explain", "vi"))
        self.assertIn("CONTENT MODE explain: teach, do not transcribe", explain)
        self.assertIn('start it with "Ví dụ minh hoạ:"', explain)
        self.assertIn("why it matters to this learner", explain)
        self.assertIn("a common confusion or mistake, only when the facts support one", explain)
        # QLT-7: short sentences, one idea each; a generic word alone explains nothing; one in five may be copied.
        self.assertIn("one idea per sentence, on average at most 20 words, never more than 35", explain)
        self.assertIn('"nghĩa là …"', explain)
        self.assertIn('a generic word ("để", "giúp", "vì", "nên") alone explains nothing', explain)
        self.assertIn("at most one in five of a slot's sentences may repeat the facts", explain)
        # QLT-7: no fixed roles; the generic role rule without an audience.
        self.assertNotIn("trưởng nhóm)", explain)
        self.assertIn("the learner's own role as TARGET_AUDIENCE describes it", explain)
        english = writer_fidelity_rules("explain", "en")
        self.assertIn('"Illustrative example:"', english)
        keep = unit_writer_prompt("vi", **WRITER_ARGS, fidelity_rules=writer_fidelity_rules("keep", "vi"))
        self.assertIn("CONTENT MODE keep", keep)
        self.assertIn("Never reword, summarise or explain a rule", keep)
        self.assertNotIn("Ví dụ minh hoạ", keep)
        structured = unit_writer_prompt("vi", **WRITER_ARGS)
        self.assertIn("CONTENT MODE structured", structured)
        self.assertNotIn("Do not paste the source, except definitions", structured)
        for prompt in (explain, keep, structured):
            self.assertIn("explain -> plain language, why it matters to the", prompt)

    def test_teaching_slots_of_an_explained_unit_may_hold_drafted_examples(self) -> None:
        self.assertEqual(slot_marks(brief("explain"), 0), {"examples": "ai_drafted"})
        self.assertEqual(slot_marks(brief("explain"), 1), {})
        self.assertEqual(slot_marks(brief("structured"), 0), {})
        self.assertEqual(slot_marks(brief("explain", segment="job_aid"), 0), {})
        self.assertEqual(slot_marks(brief("explain", role="practice"), 0), {})

    def test_budget_sentence_and_density_repair_keep_the_explanation(self) -> None:
        self.assertIn("with the explanation and the example of each main idea.",
                      html_slot_rules(max_words=607, max_chars=4249, min_chars=None, explain=True))
        self.assertIn("Stay well inside the budget: teach only what the practice needs.",
                      html_slot_rules(max_words=450, max_chars=3150, min_chars=None))
        violation = HtmlRuleViolation(DENSITY_CODE, "semantic_content", (("words", 700, 607),))
        explained_line = html_violation_line("c0", violation, explain=True)
        self.assertIn("cut repetition first", explained_line)
        self.assertNotIn("examples the brief does not ask for", explained_line)
        self.assertIn("examples the brief does not ask for", html_violation_line("c0", violation))


class BudgetTests(unittest.TestCase):
    def test_explain_budget_is_135_percent_floored_except_job_aids(self) -> None:
        contract: Any = types.SimpleNamespace(source_facts=[])
        self.assertEqual(output_budget(brief("explain"), contract)["max_words"], 607)
        self.assertEqual(output_budget(brief("explain"), contract)["max_visible_chars"], 607 * 7)
        for mode in ("structured", "keep", None):
            self.assertEqual(output_budget(brief(mode), contract)["max_words"], 450)
        self.assertEqual(output_budget(brief("explain", segment="job_aid"), contract)["max_words"], 600)
        self.assertEqual(budget_words(250, "explain", "summary_apply"), 337)
        self.assertEqual(budget_words(1_600, "explain", "example"), 2_160)


class SentenceCopyTests(unittest.TestCase):
    def test_near_verbatim_by_five_grams_or_by_one_long_run(self) -> None:
        grams = EvidenceGrams(FACTS.values())
        self.assertTrue(grams.near_verbatim(words_of(COPIED_VALUES)))
        # 7 of 10 words in one run of the facts, only 3 of 6 five-grams: near-verbatim by the run rule.
        run_only = words_of("Theo chúng tôi, giá trị cốt lõi gồm tận tâm.")
        self.assertEqual(len(run_only), 10)
        self.assertTrue(grams.near_verbatim(run_only))
        self.assertFalse(grams.near_verbatim(words_of(NOVEL)))
        # Under six words a sentence is a label, never judged.
        self.assertFalse(grams.near_verbatim(words_of("Giá trị cốt lõi tận tâm")))

    def test_definitions_rules_and_numbers_are_not_judged(self) -> None:
        for sentence in (EXACT_RULE, EXACT_DEFINITION, "Sứ mệnh: mang đến giải pháp đào tạo cho doanh nghiệp.",
                         "Tối đa ba ngày làm việc để nộp hồ sơ.", "Cấp 2 là khiếu nại gây thiệt hại nhỏ."):
            with self.subTest(sentence=sentence):
                self.assertTrue(carries_exact_claim(sentence))
        for sentence in (COPIED_VALUES, "Đây là bước quan trọng nhất.", "Điều này nghĩa là bạn cần hỏi trước.",
                         "Ví dụ minh hoạ: một nhân viên mới hỏi trưởng phòng."):
            with self.subTest(sentence=sentence):
                self.assertFalse(carries_exact_claim(sentence))
        report = copy_report(html(EXACT_RULE, EXACT_DEFINITION, "Năm 2012 công ty được thành lập tại Thành phố Hồ Chí "
                                                                 "Minh."), EvidenceGrams(FACTS.values()))
        self.assertEqual((report.judged, report.copied), (0, ()))

    def test_thresholds_per_mode(self) -> None:
        mostly_copied = html(COPIED_VALUES, COPIED_SUPPORT, NOVEL, EXACT_RULE)  # 2 of 3 judged sentences
        half_copied = html(COPIED_VALUES, COPIED_SUPPORT, NOVEL, NOVEL_2)  # 2 of 4
        for mode, component, expected in (("explain", mostly_copied, True), ("explain", half_copied, True),
                                          ("structured", mostly_copied, True), ("structured", half_copied, False),
                                          ("keep", mostly_copied, False)):
            with self.subTest(mode=mode, copied=expected):
                codes = [code for code, _index in checker(mode).findings([component, {}])]
                self.assertEqual(SENTENCE_COPY_CODE in codes, expected)

    def test_keep_rows_are_exempt_only_in_structured_mode(self) -> None:
        component = html(COPIED_VALUES, COPIED_SUPPORT, NOVEL)
        self.assertNotIn(SENTENCE_COPY_CODE, [code for code, _ in checker("structured", treatments=("keep",))
                                              .findings([component, {}])])
        self.assertIn(SENTENCE_COPY_CODE, [code for code, _ in checker("explain", treatments=("keep",))
                                           .findings([component, {}])])

    def test_too_few_judged_sentences_and_non_teaching_slots_are_not_checked(self) -> None:
        self.assertNotIn(SENTENCE_COPY_CODE, [code for code, _ in checker("explain").findings(
            [html(COPIED_VALUES, COPIED_SUPPORT), {}])])
        practice = checker("explain", role="practice")
        self.assertEqual(practice.findings([html(COPIED_VALUES, COPIED_SUPPORT, NOVEL), {}]), [])


class ExplanationTests(unittest.TestCase):
    def test_a_device_counts_only_in_a_sentence_the_writer_did_not_copy(self) -> None:
        grams = EvidenceGrams(FACTS.values())
        # The definition copied from f2 contains "giúp" but repeats the facts: no explanation.
        self.assertFalse(explained(html(EXACT_DEFINITION, COPIED_VALUES), grams))
        self.assertTrue(explained(html(COPIED_VALUES, "Điều này nghĩa là bạn hỏi người phụ trách trước."), grams))
        self.assertTrue(explained(html(COPIED_VALUES, heading="Vì sao giá trị cốt lõi quan trọng?"), grams))
        self.assertTrue(explained(html("Ví dụ minh hoạ: một nhân viên mới chưa rõ quy trình nghỉ phép."), grams))
        self.assertFalse(explained(html("Ví như cánh chim bay xa."), grams))

    def test_explain_mode_reports_a_slot_without_explanation(self) -> None:
        bare = html(COPIED_VALUES, EXACT_RULE, heading="Giá trị cốt lõi")
        self.assertIn(EXPLANATION_MISSING_CODE, [code for code, _ in checker("explain").findings([bare, {}])])
        self.assertNotIn(EXPLANATION_MISSING_CODE, [code for code, _ in checker("structured").findings([bare, {}])])


class NewFactTests(unittest.TestCase):
    def test_new_numbers_and_names_are_found_with_their_location(self) -> None:
        check = checker("explain")
        invented = html("Ví dụ minh hoạ: Chị Lan ở Hà Nội nhận 3 cuộc gọi của khách trong buổi sáng.",
                        "Điều này giúp bạn biết nên hỏi ai trước.")
        self.assertEqual(check.new_fact_locations(invented), ["semantic_content.sections[0].blocks[0]"])
        # "Nội" alone is a known word ("nội bộ"); the name "Hà Nội" is not in the facts.
        self.assertEqual(check.new_names("Ví dụ minh hoạ: Chị Lan ở Hà Nội nhận cuộc gọi."), ["Lan", "Hà Nội"])
        self.assertEqual(check.new_names("Bạn làm việc tại Thành phố Hồ Chí Minh cùng Nesso."), [])
        self.assertIn(NEW_FACT_CODE, [code for code, _ in check.findings([invented, {}])])

    def test_facts_plan_titles_ordinals_and_sentence_starts_are_known(self) -> None:
        check = checker("explain")
        known = html("Công ty thành lập năm 2012 tại Thành phố Hồ Chí Minh, vì vậy bạn sẽ làm việc ở đó.",
                     "Bước 1: Đọc sổ tay nhân viên mới trong 14 ngày đầu.", "Khoá Hội nhập Nesso giúp bạn bắt đầu.")
        self.assertEqual(check.new_fact_locations(known), [])
        self.assertNotIn(NEW_FACT_CODE, [code for code, _ in checker("structured").findings(
            [html("Chị Lan nhận 3 cuộc gọi, vì vậy cô ấy báo trưởng nhóm."), {}])])


class ExampleLabelTests(unittest.TestCase):
    def test_a_drafted_example_gets_the_label_and_a_source_example_keeps_its_own(self) -> None:
        check = checker("explain")
        component = html("Ví dụ: một nhân viên mới hỏi trưởng phòng về quy trình nghỉ phép trong tuần đầu.",
                         f"Ví dụ: {COPIED_VALUES}", "Ví dụ minh hoạ: một bạn mới chưa biết hỏi ai.")
        labelled, count = check.label_examples(0, component)
        texts = [item["text"] for item in labelled["semantic_content"]["sections"][0]["blocks"]]
        self.assertEqual(count, 1)
        self.assertTrue(texts[0].startswith("Ví dụ minh hoạ: một nhân viên mới"))
        self.assertEqual(texts[1:], [f"Ví dụ: {COPIED_VALUES}", "Ví dụ minh hoạ: một bạn mới chưa biết hỏi ai."])
        self.assertEqual(check.drafted_examples([labelled, {}]), {0: 2})
        self.assertEqual(checker("structured").label_examples(0, component), (component, 0))
        english = FidelityChecker.build(brief("explain"), FACTS, evidence_index(FACTS.values()), locale="en",
                                        plan_texts=[])
        relabelled, _count = english.label_examples(0, html("Example: a new hire asks a colleague for help."))
        self.assertEqual(relabelled["semantic_content"]["sections"][0]["blocks"][0]["text"],
                         "Illustrative example: a new hire asks a colleague for help.")

    def test_author_note_lists_every_drafted_example_in_plain_words(self) -> None:
        note = build_unit_author_note(locale="vi", judge=Judge("pass"), deterministic_codes=[], fallback_slots=[],
                                      ai_drafted=False, slot_types=["html", "problem"], drafted_examples={0: 2},
                                      remaining=[])
        self.assertIn("Ví dụ minh hoạ do AI soạn — cần kiểm tra: 2 ví dụ (khối 1 (Lý thuyết)).", note)
        english = build_unit_author_note(locale="en", judge=Judge("pass"), deterministic_codes=[], fallback_slots=[],
                                         ai_drafted=False, slot_types=["html"], drafted_examples={0: 1},
                                         remaining=[])
        self.assertIn("Illustrative examples drafted by AI — please check: 1 example(s) (block 1 (Theory)).", english)

    def test_review_lines_for_content_mode_findings_are_plain(self) -> None:
        from app.idm.qa import SlotFinding  # noqa: PLC0415 - one test

        note = build_unit_author_note(locale="vi", judge=Judge("pass"), deterministic_codes=[SENTENCE_COPY_CODE],
                                      fallback_slots=[], ai_drafted=False, slot_types=["html"], review_slots=[],
                                      remaining=[SlotFinding(SENTENCE_COPY_CODE, 0), SlotFinding(NEW_FACT_CODE, 0)])
        self.assertIn("Nên biên tập khối 1 (Lý thuyết): nhiều câu gần như chép nguyên văn tài liệu", note)
        self.assertIn("Cần xem khối 1 (Lý thuyết): có số liệu hoặc tên riêng không có trong tài liệu", note)
        self.assertNotIn("IDM_W5_SENTENCE_COPY", note)
        self.assertNotIn("Kiểm tra tự động còn cảnh báo", note)


class QaIntegrationTests(unittest.TestCase):
    def test_the_slot_level_verbatim_check_only_runs_in_structured_mode(self) -> None:
        long_copy = html(" ".join([*FACTS.values(), *FACTS.values()]))
        unit = {"components": [long_copy, {"type": "problem"}]}
        explain_brief = brief("explain", treatments=("condense",))
        owned = [" ".join(FACTS.values()), ""]
        explained_codes = {item.code for item in deterministic_slot_findings(
            unit, explain_brief, owned, fidelity=checker("explain"))}
        self.assertNotIn("IDM_W5_VERBATIM_COPY", explained_codes)
        structured_codes = {item.code for item in deterministic_slot_findings(
            unit, brief("structured"), owned, fidelity=checker("structured"))}
        self.assertIn("IDM_W5_VERBATIM_COPY", structured_codes)
        self.assertIn("IDM_W5_VERBATIM_COPY", {item.code for item in deterministic_slot_findings(
            unit, brief("structured"), owned)})

    def test_q12_is_stored_under_q2_and_not_applicable_is_never_stored(self) -> None:
        from app.idm.contracts import IdmJudgeFindingV1  # noqa: PLC0415 - one test

        findings = [IdmJudgeFindingV1(criterion="Q2_not_copied", severity="pass", witness="w"),
                    IdmJudgeFindingV1(criterion=EXPLAINED_CRITERION, severity="major", component_index=0, witness="w")]
        self.assertEqual(criteria_summary(findings), {"Q2_not_copied": "major"})
        note = build_unit_author_note(locale="vi", judge=Judge("review_required", findings), deterministic_codes=[],
                                      fallback_slots=[], ai_drafted=False, remaining=[])
        self.assertIn("phần lý thuyết chưa được giảng giải cho người học", note)

    def test_course_note_says_how_the_content_is_written(self) -> None:
        self.assertIn('ví dụ do AI soạn được ghi "Ví dụ minh hoạ" và cần kiểm tra', fidelity_note_line("explain", "vi"))
        self.assertTrue(fidelity_note_line("keep", "vi", requested=True).endswith(
            "(theo yêu cầu trong mô tả khoá học)"))
        self.assertIn("Content style: organised for learning", fidelity_note_line("structured", "en"))


if __name__ == "__main__":
    unittest.main()
