"""IDM source-locked step diagram (run e869f43a, unit chapter_2.lesson_1.unit_1).

The per-slot fallback of an ``la_diagram`` slot drew a heading-prefixed slogan ("Title: no action
-> meaningless") as two nodes next to the real step chain, because the shared relationship
extraction treats every arrow as a relationship, and the same pair was required from the provider
diagram (``DIAGRAM_SOURCE_RELATION_MISSING``). IDM units now inject a step-only builder; the legacy
builder is unchanged. All fixtures are synthetic; no provider is called.
"""

from __future__ import annotations

import dataclasses
import unittest
from itertools import pairwise
from typing import Any

from app import main
from app.idm.contracts import brief_hash_of
from app.idm.diagram import idm_diagram_relationships, idm_source_step_diagram
from app.idm.storyboard import build_idm_expected, parse_brief, run_idm_unit
from app.instructional_quality import clean_source_facts, source_relationship_pairs
from app.schemas.orchestration_v2 import RagLessonAuthorUnitV2Request
from tests.idm_test_support import make_runtime, rehash_contract
from tests.test_idm_storyboard import REPAIR, WRITER, FakeGenerate, severity_body, severity_writer

CHAIN = ["PLAN", "DO", "CHECK", "ACT", "STANDARD"]
CHAIN_PAIRS = [list(pair) for pair in pairwise(CHAIN)]
# Same shape as the run e869f43a evidence: a chapter label, a heading-prefixed one-arrow slogan, a
# sentence, a section heading and the real arrow chain (spacing as extracted from the PDF).
STEP_FACTS = [
    "CHƯƠNG 05",
    "Vòng Lặp Cải Tiến: Không Đo Lường → Không Cải Tiến",
    "Một sáng kiến chỉ tạo giá trị khi được đo lường và chuẩn hoá thành cách làm việc hằng ngày của cả đội.",
    "DÒNG CHẢY CẢI TIẾN LIÊN TỤC",
    "PLAN →DO →CHECK →ACT →STANDARD",
]
NO_STEP_FACTS = [
    "CHƯƠNG 05",
    "Vòng Lặp Cải Tiến: Không Đo Lường → Không Cải Tiến",
    "Không hành động = không kết quả.",
    "DÒNG CHẢY CẢI TIẾN LIÊN TỤC",
    "Mỗi nhóm tự chọn cách cải tiến phù hợp với công việc của mình trong quý này.",
]


def labels(diagram: dict[str, Any]) -> list[str]:
    return [node["label"] for node in diagram["nodes"]]


def pairs(diagram: dict[str, Any]) -> list[tuple[str, str]]:
    names = labels(diagram)
    return [(names[edge["source"]], names[edge["target"]]) for edge in diagram["edges"]]


def diagram_body(facts: list[str]) -> dict[str, Any]:
    """The golden html + practice unit with its second slot turned into an la_diagram slot."""

    body = severity_body()
    contract = body["unit_contract"]
    for fact, text in zip(contract["source_facts"], facts, strict=True):
        fact["fact_text"] = text
    contract["component_plan"][1].update(type="la_diagram", purpose="relationship", title="Dòng chảy cải tiến")
    brief = contract["idm_unit_brief"]
    brief["components"][1].update(type="la_diagram", role="show", title="Dòng chảy cải tiến", practice=None)
    brief["brief_hash"] = brief_hash_of(brief)
    rehash_contract(contract)
    return body


def diagram_writer(body: dict[str, Any], names: list[str]) -> dict[str, Any]:
    writer = severity_writer(body)
    writer["components"]["c1"] = {
        "title": "Dòng chảy cải tiến", "selection_rationale": "Sơ đồ các bước cải tiến.",
        "covered_source_fact_ids": [], "name": "Dòng chảy cải tiến",
        "nodes": [{"label": name, "shape": "rounded", "tooltip": None} for name in names],
        "edges": [{"source": index, "target": index + 1, "label": None} for index in range(len(names) - 1)],
    }
    return writer


class StepDiagramBuilderTests(unittest.TestCase):
    def test_slogan_heading_and_sentences_never_become_nodes(self) -> None:
        facts = clean_source_facts(STEP_FACTS, preserve_table_numeric=True)
        legacy = main._orchestration_v2_source_relationship_diagram("Dòng chảy", facts, "vi")
        self.assertIn("Vòng Lặp Cải Tiến: Không Đo Lường", labels(legacy))  # the published garbage node
        self.assertIn(("Vòng Lặp Cải Tiến: Không Đo Lường", "Không Cải Tiến", "→"), source_relationship_pairs(facts))
        diagram = idm_source_step_diagram("Dòng chảy", facts, "vi")
        assert diagram is not None
        self.assertEqual(labels(diagram), CHAIN)
        self.assertEqual([list(pair) for pair in pairs(diagram)], CHAIN_PAIRS)
        self.assertTrue(all("label" not in edge for edge in diagram["edges"]))
        self.assertEqual(diagram["name"], "Dòng chảy")
        self.assertEqual(idm_diagram_relationships(STEP_FACTS), CHAIN_PAIRS)

    def test_fewer_than_three_step_nodes_means_no_fallback(self) -> None:
        cases = {
            "slogan_and_headings": NO_STEP_FACTS,
            "single_arrow": ["Đầu vào → Đầu ra"],
            "equation_chain": ["Không hành động = vô nghĩa → mất cơ hội → thua lỗ"],
            "sentence_inside_chain": ["Tiếp nhận → Sau đó nhân viên gọi lại cho khách. Ghi nhận → Đóng hồ sơ"],
            "quote_inside_chain": ["Bắt đầu → “Không có hành động” → Kết thúc"],
            "repeated_node": ["Đo lường → Cải tiến → Đo lường"],
            "bidirectional": ["Bộ phận A ↔ Bộ phận B ↔ Bộ phận C"],
            "two_numbered_steps": ["1. Tiếp nhận yêu cầu", "2. Phân loại mức độ"],
            "numbered_paragraph_step": ["1. Tiếp nhận yêu cầu", "2. Phân loại mức độ",
                                        "3. Gọi lại cho khách. Sau đó ghi nhận kết quả vào hệ thống."],
        }
        for name, facts in cases.items():
            with self.subTest(case=name):
                self.assertIsNone(idm_source_step_diagram("T", facts, "vi"))
                self.assertEqual(idm_diagram_relationships(facts), [])

    def test_chain_title_is_dropped_and_numbered_steps_form_a_sequence(self) -> None:
        titled = idm_source_step_diagram("T", ["Quy trình: Tiếp nhận → Phân loại → Xử lý → Đóng hồ sơ."], "en")
        assert titled is not None
        self.assertEqual(labels(titled), ["Tiếp nhận", "Phân loại", "Xử lý", "Đóng hồ sơ"])
        steps = ["Quy trình xử lý", "1. Tiếp nhận yêu cầu của khách", "2. Phân loại mức độ",
                 "3. Chuyển cho người phụ trách"]
        sequence = idm_source_step_diagram("T", steps, "vi")
        assert sequence is not None
        self.assertEqual(labels(sequence), ["Tiếp nhận yêu cầu của khách", "Phân loại mức độ",
                                            "Chuyển cho người phụ trách"])
        self.assertEqual({edge["label"] for edge in sequence["edges"]}, {"Tiếp theo"})
        self.assertEqual(idm_diagram_relationships(steps), [])  # drawn, but never required from the provider
        many = [f"Pha {index} mở → Pha {index} chạy → Pha {index} đóng" for index in range(6)]
        capped = idm_source_step_diagram("T", many, "vi")
        assert capped is not None
        self.assertEqual(len(capped["nodes"]), 10)
        self.assertTrue(all(edge["source"] < 10 and edge["target"] < 10 for edge in capped["edges"]))

    def test_idm_units_inject_the_step_builder_and_legacy_defaults_are_unchanged(self) -> None:
        request = RagLessonAuthorUnitV2Request.model_validate(diagram_body(STEP_FACTS))
        legacy = main.build_orchestration_v2_source_locked_components(request.unit_contract, "vi")[1]
        idm = main._idm_unit_deps(request).source_locked_components
        assert legacy is not None and idm is not None and idm[1] is not None
        plan = request.unit_contract.component_plan[1]
        facts = clean_source_facts([fact.fact_text for fact in request.unit_contract.source_facts
                                    if fact.fact_key in plan.supporting_evidence_fact_ids], preserve_table_numeric=True)
        expected_legacy = main._orchestration_v2_source_relationship_diagram(legacy["title"], facts, "vi")
        self.assertEqual({key: legacy[key] for key in ("name", "nodes", "edges")}, expected_legacy)
        self.assertEqual(labels(idm[1]), CHAIN)
        self.assertEqual({key: value for key, value in idm[1].items() if key not in {"name", "nodes", "edges"}},
                         {key: value for key, value in legacy.items() if key not in {"name", "nodes", "edges"}})
        none_request = RagLessonAuthorUnitV2Request.model_validate(diagram_body(NO_STEP_FACTS))
        slots = main._idm_unit_deps(none_request).source_locked_components
        assert slots is not None
        self.assertIsNone(slots[1])
        self.assertIsNotNone(main.build_orchestration_v2_source_locked_components(none_request.unit_contract, "vi")[1])


class DiagramSlotFlowTests(unittest.IsolatedAsyncioTestCase):
    async def run_unit(self, facts: list[str], *answers: tuple[str, Any]) -> tuple[dict[str, Any], FakeGenerate]:
        request = RagLessonAuthorUnitV2Request.model_validate(diagram_body(facts))
        deps = dataclasses.replace(main._idm_unit_deps(request), judge_mode="off")
        queues: dict[str, list[Any]] = {}
        for family, answer in answers:
            queues.setdefault(family, []).append(answer)
        provider = FakeGenerate(**queues)
        result = await run_idm_unit(contract=request.unit_contract, runtime=make_runtime(provider, allowance=None),
                                    deps=deps, fallback_only=False)
        return result, provider

    def test_required_relationships_are_the_step_chain_only(self) -> None:
        request = RagLessonAuthorUnitV2Request.model_validate(diagram_body(STEP_FACTS))
        contract = request.unit_contract
        expected = build_idm_expected(contract, parse_brief(contract), main._idm_unit_deps(request), "vi")
        plan_id = contract.component_plan[1].component_plan_id
        self.assertEqual(expected["diagram_relationships_by_plan_id"],
                         {plan_id: CHAIN_PAIRS})

    async def test_provider_step_chain_is_accepted_without_the_slogan_pair(self) -> None:
        body = diagram_body(STEP_FACTS)
        result, provider = await self.run_unit(STEP_FACTS, (WRITER, diagram_writer(body, CHAIN)))
        self.assertEqual(provider.names, [WRITER])
        self.assertEqual(result["content_origin"], "provider_validated")
        self.assertEqual(result["unit"]["idm_quality"]["deterministic_codes"], [])
        self.assertEqual(labels(result["unit"]["components"][1]), CHAIN)

    async def test_missing_chain_falls_back_to_the_clean_step_diagram(self) -> None:
        body = diagram_body(STEP_FACTS)
        result, provider = await self.run_unit(STEP_FACTS, (WRITER, diagram_writer(body, ["Bắt đầu", "Kết thúc"])),
                                               (REPAIR, "{}"))
        self.assertEqual(provider.names, [WRITER, REPAIR])
        self.assertIn("c1 DIAGRAM_SOURCE_RELATION_MISSING at c1.edges: keep every step chain the owned facts write "
                      "with arrows", provider.calls[1]["prompt"])
        self.assertEqual((result["content_origin"], result["quality_state"]),
                         ("structured_fallback", "review_required"))
        diagram = result["unit"]["components"][1]
        self.assertTrue(diagram["source_locked_fallback"])
        self.assertEqual(labels(diagram), CHAIN)
        self.assertFalse(any(":" in label for label in labels(diagram)))
        self.assertNotIn("source_locked_fallback", result["unit"]["components"][0])

    async def test_without_step_structure_a_valid_provider_diagram_is_kept(self) -> None:
        body = diagram_body(NO_STEP_FACTS)
        result, provider = await self.run_unit(NO_STEP_FACTS,
                                               (WRITER, diagram_writer(body, ["Đo lường", "Cải tiến", "Chuẩn hoá"])))
        self.assertEqual(provider.names, [WRITER])
        self.assertEqual(result["content_origin"], "provider_validated")
        self.assertEqual(labels(result["unit"]["components"][1]), ["Đo lường", "Cải tiến", "Chuẩn hoá"])


if __name__ == "__main__":
    unittest.main()
