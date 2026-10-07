"""W1 SME Content Map and W1-reduce + W0 (spec §7.2, §7.3, Appendix C)."""

from __future__ import annotations

import time
import unittest
from collections.abc import Callable
from typing import Any

from app.idm.content_map import (
    FactIndex,
    SectionMapResult,
    apply_client_context,
    apply_w1_reduce,
    fallback_w1_reduce,
    fallback_w1_section,
    map_section,
    map_sections,
    normalize_w1_section,
    page_furniture_keys,
    rekey_blocks,
    run_w1_reduce,
    small_block_warnings,
    validate_w1_reduce,
    validate_w1_section,
)
from app.idm.contracts import (
    IdmContentBlockV1,
    IdmProjectContextV1,
    IdmW1BlockDraftV1,
    IdmW1ReduceResponseV1,
    IdmW1SectionResponseV1,
)
from app.idm.policy import IDM_LO_MIN_COUNT
from app.idm.runtime import IdmProviderError, IdmRuntime
from app.idm.signals import IdmSection, compute_fact_signals, plan_idm_sections
from app.idm.validation import IdmIssue, code_counts, errors, warnings
from app.lesson_author_orchestration_v2_provider import SourceSnapshotFactV2
from tests.idm_golden import (
    DOCUMENT_ID,
    W1_REDUCE,
    W1_SECTION,
    FakeIdmProvider,
    key,
    keys,
    mutate,
    project_context,
    source_facts,
)

FACTS = source_facts()
INDEX = FactIndex.build(FACTS, compute_fact_signals(FACTS))
SECTION = plan_idm_sections(FACTS, compute_fact_signals(FACTS), {DOCUMENT_ID: "quy-trinh-khieu-nai.pdf"}).sections[0]
FURNITURE = page_furniture_keys(FACTS)
CONTEXT = IdmProjectContextV1.model_validate(project_context())
CLIENT_LOS = ["Phân loại khiếu nại theo nhóm và mức độ nghiêm trọng", "Quyết định tự xử lý hay escalate khiếu nại",
              "Thực hiện quy trình tiếp nhận khiếu nại đúng trình tự"]


def edited(payload: dict[str, Any], apply: Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
    def change(copy: dict[str, Any]) -> dict[str, Any]:
        apply(copy)
        return copy

    return mutate(payload, change)


def section_response(apply: Callable[[dict[str, Any]], Any] | None = None) -> IdmW1SectionResponseV1:
    return IdmW1SectionResponseV1.model_validate(edited(W1_SECTION, apply) if apply else W1_SECTION)


def reduce_response(apply: Callable[[dict[str, Any]], Any] | None = None) -> IdmW1ReduceResponseV1:
    return IdmW1ReduceResponseV1.model_validate(edited(W1_REDUCE, apply) if apply else W1_REDUCE)


def golden_blocks() -> list[IdmContentBlockV1]:
    drafts, noise = normalize_w1_section(section_response(), SECTION.fact_keys, INDEX)
    return rekey_blocks([SectionMapResult(SECTION, drafts, noise, "provider")], INDEX)[0]


def context(**changes: Any) -> IdmProjectContextV1:
    return IdmProjectContextV1.model_validate({**project_context(), **changes})


def runtime(provider: Any, seconds: float = 600.0) -> IdmRuntime:
    return IdmRuntime(generate=provider, api_key="test-key", model="fake-model", locale="vi",
                      deadline=time.monotonic() + seconds)


def codes(issues: list[IdmIssue]) -> set[str]:
    return {issue.code for issue in issues}


def content_block(block_id: str, section_id: str, fact_keys: list[str]) -> IdmContentBlockV1:
    return IdmContentBlockV1(block_id=block_id, section_id=section_id, name=f"Khối {block_id}", summary="Tóm tắt.",
                             intent="know", support_role=None, content_kind="concept", fact_keys=fact_keys,
                             issues=[], gaps=[], sme_questions=[], origin="provider")


class SectionValidatorTests(unittest.TestCase):
    def test_golden_section_has_no_errors(self) -> None:
        found = validate_w1_section(section_response(), SECTION.fact_keys)
        self.assertEqual(errors(found), [])
        self.assertEqual(warnings(found), [IdmIssue("IDM_W1_BLOCK_COMPOUND_NAME", "blocks[0].name", "warning")])

    def test_every_error_code(self) -> None:
        cases: list[tuple[Callable[[dict[str, Any]], Any], str, str]] = [
            (lambda p: p["noise"].pop(0), "IDM_W1_FACT_UNASSIGNED", "blocks"),
            (lambda p: p["blocks"][1]["fact_keys"].append(key(0, 2)), "IDM_W1_FACT_DUPLICATED", "blocks"),
            (lambda p: p["blocks"][1]["fact_keys"].append("d9-c0-f1"), "IDM_W1_FACT_UNKNOWN", "blocks[1].fact_keys"),
            (lambda p: p["noise"].append({"fact_key": "d9-c0-f1", "reason": "unreadable"}), "IDM_W1_FACT_UNKNOWN",
             "noise[4].fact_key"),
            (lambda p: p["blocks"][1].update(local_id="b1"), "IDM_W1_LOCAL_ID_DUPLICATED", "blocks[1].local_id"),
            (lambda p: p["blocks"][2].update(name="Giới thiệu"), "IDM_W1_BLOCK_NAME_GENERIC", "blocks[2].name"),
            (lambda p: p["blocks"][2].update(name="Phần 3"), "IDM_W1_BLOCK_NAME_GENERIC", "blocks[2].name"),
        ]
        for apply, code, path in cases:
            with self.subTest(code=code, path=path):
                found = errors(validate_w1_section(section_response(apply), SECTION.fact_keys))
                self.assertIn(IdmIssue(code, path), found)

    def test_warnings_never_block(self) -> None:
        cases: list[tuple[Callable[[dict[str, Any]], Any], str]] = [
            (lambda p: p["blocks"][3].update(name="Tiếp nhận và phân loại khiếu nại"), "IDM_W1_BLOCK_COMPOUND_NAME"),
            (lambda p: p["blocks"][3].update(name="Receive and classify complaints"), "IDM_W1_BLOCK_COMPOUND_NAME"),
            (lambda p: p["blocks"][3].update(sme_questions=["Phần nào quan trọng nhất trong tài liệu này vậy?"]),
             "IDM_W1_SME_QUESTION_TOO_BROAD"),
            (lambda p: p["blocks"][3].update(sme_questions=["Đúng không?"]), "IDM_W1_SME_QUESTION_TOO_BROAD"),
        ]
        for apply, code in cases:
            with self.subTest(code=code):
                found = validate_w1_section(section_response(apply), SECTION.fact_keys)
                self.assertEqual(errors(found), [])
                self.assertIn(code, codes(warnings(found)))

    def test_small_block_warning(self) -> None:
        def draft(fact_keys: list[str], role: str | None = None) -> IdmW1BlockDraftV1:
            return IdmW1BlockDraftV1(local_id="b1", name="Nhóm sản phẩm", summary="s", intent="know",
                                     support_role=role, content_kind="concept", fact_keys=fact_keys)

        self.assertEqual(small_block_warnings([draft([key(3, 2)])], INDEX), ["IDM_W1_BLOCK_TOO_SMALL"])
        self.assertEqual(small_block_warnings([draft([key(3, 2)], "example")], INDEX), [])
        self.assertEqual(small_block_warnings([draft([key(3, 2), key(3, 3)])], INDEX), [])
        self.assertEqual(small_block_warnings([draft([key(1, 4)])], INDEX), [])

    def test_issue_helpers(self) -> None:
        found = [IdmIssue("B", "x"), IdmIssue("A", "y", "warning"), IdmIssue("B", "z")]
        self.assertEqual(code_counts(found), {"A": 1, "B": 2})
        self.assertEqual(found[0].as_repair_item(), {"code": "B", "path": "x"})


class MapSectionTests(unittest.IsolatedAsyncioTestCase):
    async def map(self, provider: FakeIdmProvider, rt: IdmRuntime | None = None) -> SectionMapResult:
        return await map_section(rt or runtime(provider), SECTION, INDEX, project_context(), FURNITURE,
                                 tail_reserve_tokens=0)

    async def test_repair_after_a_validation_error(self) -> None:
        invalid = edited(W1_SECTION, lambda p: p["noise"].pop(0))
        provider = FakeIdmProvider({"IdmW1SectionResponseV1": [invalid, W1_SECTION]})
        rt = runtime(provider)
        result = await self.map(provider, rt)
        self.assertEqual(result.origin, "provider")
        self.assertEqual(len(result.blocks), 14)
        first, second = (call["prompt"] for call in provider.calls)
        self.assertNotIn("REPAIR_REQUIREMENTS", first)
        self.assertTrue(second.startswith(first))
        suffix = second[len(first):]
        self.assertIn("REPAIR_REQUIREMENTS", suffix)
        self.assertIn("IDM_W1_FACT_UNASSIGNED", suffix)
        for item in FACTS:
            self.assertNotIn(item.fact_text, suffix)
            self.assertNotIn(item.fact_key, suffix)
        self.assertEqual([event["invocation_kind"] for event in rt.trace], ["writer", "repair"])
        self.assertEqual(result.issue_codes["IDM_W1_FACT_UNASSIGNED"], 1)
        self.assertIn("IDM_W1_BLOCK_COMPOUND_NAME", result.warning_codes)

    async def test_repair_after_a_schema_error_lists_only_codes(self) -> None:
        provider = FakeIdmProvider({"IdmW1SectionResponseV1": ["not json", W1_SECTION]})
        result = await self.map(provider)
        self.assertEqual(result.origin, "provider")
        suffix = provider.calls[1]["prompt"][len(provider.calls[0]["prompt"]):]
        self.assertIn("REPAIR_REQUIREMENTS", suffix)
        self.assertIn("json_invalid", suffix)
        self.assertNotIn("not json", suffix)

    async def test_double_failure_falls_back_to_one_block_per_heading(self) -> None:
        invalid = edited(W1_SECTION, lambda p: p["noise"].pop(0))
        provider = FakeIdmProvider({"IdmW1SectionResponseV1": [invalid, invalid]})
        rt = runtime(provider)
        result = await self.map(provider, rt)
        self.assertEqual(result.origin, "deterministic_fallback")
        self.assertEqual(result.issue_codes["IDM_W1_FACT_UNASSIGNED"], 2)
        self.assertEqual([(item.fact_key, item.reason) for item in result.noise],
                         [(fact_key, "page_furniture") for fact_key in keys(12, 1, 3)])
        self.assertEqual(len(result.blocks), 12)
        self.assertTrue(all("H" in INDEX.flags[block.fact_keys[0]] for block in result.blocks))
        assigned = [fact_key for block in result.blocks for fact_key in block.fact_keys]
        self.assertEqual(sorted([*assigned, *(item.fact_key for item in result.noise)]), sorted(SECTION.fact_keys))
        by_first = {block.fact_keys[0]: block for block in result.blocks}
        self.assertEqual(by_first[key(0, 1)].name, "Quy trình xử lý khiếu nại khách hàng")
        self.assertEqual(by_first[key(5, 1)].content_kind, "procedure")
        self.assertEqual(by_first[key(10, 1)].content_kind, "reference")
        self.assertEqual(by_first[key(2, 1)].content_kind, "concept")
        self.assertEqual({block.intent for block in result.blocks}, {"know"})
        self.assertEqual(rt.trace[-1]["failure_code"], "IDM_W1_SECTION_FALLBACK")

    async def test_transient_provider_error_falls_back_and_stops_later_calls(self) -> None:
        unavailable = IdmProviderError("AI_PROVIDER_UNAVAILABLE", terminal=False)
        provider = FakeIdmProvider({"IdmW1SectionResponseV1": [unavailable, W1_SECTION]})
        rt = runtime(provider)
        first = await self.map(provider, rt)
        second = await self.map(provider, rt)
        self.assertEqual((first.origin, second.origin), ("deterministic_fallback", "deterministic_fallback"))
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(rt.provider_failure_code, "AI_PROVIDER_UNAVAILABLE")
        self.assertEqual(first.issue_codes, {"AI_PROVIDER_UNAVAILABLE": 1})
        self.assertFalse(rt.usage.complete)

    async def test_terminal_provider_error_propagates(self) -> None:
        provider = FakeIdmProvider({"IdmW1SectionResponseV1": [IdmProviderError("AI_KEY_INVALID", terminal=True)]})
        with self.assertRaises(IdmProviderError):
            await self.map(provider)

    async def test_budget_error_falls_back_without_a_call(self) -> None:
        provider = FakeIdmProvider({})
        result = await self.map(provider, runtime(provider, seconds=0.0))
        self.assertEqual(result.origin, "deterministic_fallback")
        self.assertEqual(result.issue_codes, {"IDM_DEADLINE_EXCEEDED": 1})
        self.assertEqual(provider.calls, [])

    async def test_map_sections_keeps_section_order(self) -> None:
        provider = FakeIdmProvider({"IdmW1SectionResponseV1": [W1_SECTION, W1_SECTION]})
        twin = IdmSection("sec_002", DOCUMENT_ID, "x", SECTION.fact_keys, SECTION.content_chars)
        results = await map_sections(runtime(provider), [SECTION, twin], INDEX, project_context(), FURNITURE,
                                     parallelism=1, tail_reserve_tokens=0)
        self.assertEqual([result.section.section_id for result in results], ["sec_001", "sec_002"])


class DeterministicW1Tests(unittest.TestCase):
    def test_fallback_without_headings_names_the_block_from_its_text(self) -> None:
        facts = [SourceSnapshotFactV2(document_id="d", fact_key=f"k{n}", scope_key="s", fact_text=text)
                 for n, text in enumerate(["ab", "Nội dung thứ hai của mục.", "Trang 2"])]
        index = FactIndex.build(facts, {})
        section = IdmSection("sec_001", "d", "Tài liệu A", ("k0", "k1", "k2"), 40)
        blocks, noise = fallback_w1_section(section, index, page_furniture_keys(facts))
        self.assertEqual([item.fact_key for item in noise], ["k2"])
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].name, "ab Tài liệu A")
        self.assertEqual(blocks[0].fact_keys, ["k0", "k1"])
        only_furniture = IdmSection("sec_002", "d", "Tài liệu A", ("k2",), 7)
        self.assertEqual(fallback_w1_section(only_furniture, index, {"k2"})[0], [])

    def test_page_furniture_keys(self) -> None:
        self.assertEqual(page_furniture_keys(FACTS), set(keys(12, 1, 3)))
        texts = ["Trang 5", "- 12 -", "Page 3 of 10", "12", "Bản quyền 2026 trang 1", "Bản quyền 2026 trang 2",
                 "Tài liệu nội bộ", "Tài liệu nội bộ", "Tài liệu nội bộ", "Nhân viên gọi lại sau 2 giờ."]
        facts = [SourceSnapshotFactV2(document_id="d", fact_key=f"k{n}", scope_key="s", fact_text=text)
                 for n, text in enumerate(texts)]
        self.assertEqual(page_furniture_keys(facts), {"k0", "k1", "k2", "k3"})

    def test_rekey_blocks_follows_document_order(self) -> None:
        late = IdmSection("sec_002", DOCUMENT_ID, "Quy trình", tuple(keys(5, 1, 13)), 1)
        early = IdmSection("sec_001", DOCUMENT_ID, "Định nghĩa", tuple(keys(2, 1, 3)), 1)

        def draft(local_id: str, fact_keys: list[str]) -> IdmW1BlockDraftV1:
            return IdmW1BlockDraftV1(local_id=local_id, name=f"Khối {local_id}", summary="s", intent="do",
                                     support_role=None, content_kind="procedure", fact_keys=fact_keys)

        results = [SectionMapResult(late, [draft("b2", keys(5, 7, 13)), draft("b1", keys(5, 1, 6))], [],
                                    "deterministic_fallback"),
                   SectionMapResult(early, [draft("b1", keys(2, 1, 3))], [], "provider")]
        blocks, noise = rekey_blocks(results, INDEX)
        self.assertEqual([block.block_id for block in blocks], ["cb_0001", "cb_0002", "cb_0003"])
        self.assertEqual([block.fact_keys[0] for block in blocks], [key(2, 1), key(5, 1), key(5, 7)])
        self.assertEqual([block.section_id for block in blocks], ["sec_001", "sec_002", "sec_002"])
        self.assertEqual([block.origin for block in blocks], ["provider", "deterministic_fallback",
                                                              "deterministic_fallback"])
        self.assertEqual(noise, [])

    def test_normalize_orders_facts_and_sanitises_text(self) -> None:
        response = section_response(lambda p: p["blocks"][3].update(
            name="Khiếu nại\n được   chia <thế nào>?", fact_keys=list(reversed(keys(3, 1, 6)))))
        blocks, _noise = normalize_w1_section(response, SECTION.fact_keys, INDEX)
        block = next(item for item in blocks if item.local_id == "b4")
        self.assertEqual(block.fact_keys, keys(3, 1, 6))
        self.assertEqual(block.name, "Khiếu nại được chia ‹thế nào›?")  # noqa: RUF001


class ReduceValidatorTests(unittest.TestCase):
    def test_golden_reduce_is_valid(self) -> None:
        self.assertEqual(validate_w1_reduce(reduce_response(), golden_blocks(), CONTEXT), [])

    def test_every_reduce_code(self) -> None:
        merge = {"keep_block_id": "cb_0006", "merged_block_ids": ["cb_0007"], "reason": "Cùng giai đoạn"}
        cases: list[tuple[Callable[[dict[str, Any]], Any], str]] = [
            (lambda p: p["learning_objectives"].pop(), "IDM_W0_LO_COUNT"),
            (lambda p: p["learning_objectives"][2].update(lo_id="lo_4"), "IDM_W0_LO_ID_SEQUENCE"),
            (lambda p: p["learning_objectives"][0].update(statement="Người học có thể hiểu các nhóm khiếu nại"),
             "IDM_W0_LO_UNMEASURABLE"),
            (lambda p: p["must_dos"][0].update(lo_id="lo_7"), "IDM_W0_MUST_DO_ORPHAN"),
            (lambda p: p["must_dos"][1].update(must_do_id="md_1"), "IDM_W0_MUST_DO_ID_DUPLICATED"),
            (lambda p: p.update(must_dos=[md for md in p["must_dos"] if md["lo_id"] != "lo_3"]),
             "IDM_W0_LO_WITHOUT_MUST_DO"),
            (lambda p: p["must_dos"][0].update(statement="Kiến thức về phân loại khiếu nại"),
             "IDM_W0_MUST_DO_IS_TOPIC"),
            (lambda p: p["must_dos"][0].update(statement="Tổng quan quy trình"), "IDM_W0_MUST_DO_IS_TOPIC"),
            (lambda p: p["merges"].append({**merge, "merged_block_ids": ["cb_0099"]}), "IDM_W1_MERGE_INVALID"),
            (lambda p: p["merges"].append({**merge, "merged_block_ids": ["cb_0006"]}), "IDM_W1_MERGE_INVALID"),
            (lambda p: p["merges"].append({**merge, "merged_block_ids": ["cb_0007", "cb_0007"]}),
             "IDM_W1_MERGE_INVALID"),
            (lambda p: p["merges"].extend([merge, {**merge, "keep_block_id": "cb_0008"}]), "IDM_W1_MERGE_INVALID"),
            (lambda p: p["lo_links"].append({"block_id": "cb_0099", "lo_id": "lo_1", "relation": "direct"}),
             "IDM_W1_LINK_INVALID"),
            (lambda p: p["lo_links"].append({"block_id": "cb_0001", "lo_id": "lo_8", "relation": "direct"}),
             "IDM_W1_LINK_INVALID"),
            (lambda p: p["conflicts"].append({"block_ids": ["cb_0099"], "note": "x", "sme_question": "Hỏi gì?"}),
             "IDM_W1_LINK_INVALID"),
        ]
        blocks = golden_blocks()
        for apply, code in cases:
            with self.subTest(code=code):
                found = validate_w1_reduce(reduce_response(apply), blocks, CONTEXT)
                self.assertIn(code, codes(found))
                self.assertTrue(all(issue.severity == "error" for issue in found))

    def test_client_objectives_change_count_and_measurability_rules(self) -> None:
        blocks = golden_blocks()
        unmeasurable = reduce_response(
            lambda p: p["learning_objectives"][0].update(statement="Hiểu các nhóm khiếu nại"))
        self.assertEqual(validate_w1_reduce(unmeasurable, blocks, context(learning_objectives=CLIENT_LOS)), [])
        mismatch = validate_w1_reduce(reduce_response(), blocks, context(learning_objectives=CLIENT_LOS[:2]))
        self.assertEqual(codes(mismatch), {"IDM_W0_LO_COUNT"})


class ApplyReduceTests(unittest.TestCase):
    def test_merge_unions_facts_in_document_order(self) -> None:
        merge = {"keep_block_id": "cb_0008", "merged_block_ids": ["cb_0006"], "reason": "Cùng giai đoạn tiếp nhận"}
        response = reduce_response(lambda p: p["merges"].append(merge))
        self.assertEqual(validate_w1_reduce(response, golden_blocks(), CONTEXT), [])
        blocks, links = apply_w1_reduce(response, golden_blocks(), INDEX, "vi")
        self.assertEqual(len(blocks), 13)
        self.assertNotIn("cb_0006", {block.block_id for block in blocks})
        kept = next(block for block in blocks if block.block_id == "cb_0008")
        self.assertEqual(kept.fact_keys, [*keys(5, 1, 5), *keys(5, 10, 13)])
        self.assertEqual(kept.issues[-1].type, "duplicate")
        self.assertIn("Tiếp nhận thông tin khiếu nại ban đầu", kept.issues[-1].note)
        self.assertIn("Cùng giai đoạn tiếp nhận", kept.issues[-1].note)
        order = [block.block_id for block in blocks]
        self.assertLess(order.index("cb_0008"), order.index("cb_0007"))
        self.assertNotIn("cb_0006", {link.block_id for link in links})
        self.assertIn(("cb_0008", "lo_3", "direct"), {(link.block_id, link.lo_id, link.relation) for link in links})

    def test_merge_note_follows_locale(self) -> None:
        merge = {"keep_block_id": "cb_0006", "merged_block_ids": ["cb_0007"], "reason": "Same stage"}
        blocks, _links = apply_w1_reduce(reduce_response(lambda p: p["merges"].append(merge)), golden_blocks(),
                                         INDEX, "en")
        kept = next(block for block in blocks if block.block_id == "cb_0006")
        self.assertTrue(kept.issues[-1].note.startswith("Merged duplicate content from:"))

    def test_conflicts_add_an_issue_and_an_sme_question(self) -> None:
        conflict = {"block_ids": ["cb_0007", "cb_0010"], "note": "Hai mục nêu thời hạn khác nhau.",
                    "sme_question": "Thời hạn nào đúng cho khiếu nại cấp 2?"}
        blocks, _links = apply_w1_reduce(reduce_response(lambda p: p["conflicts"].append(conflict)), golden_blocks(),
                                         INDEX, "vi")
        by_id = {block.block_id: block for block in blocks}
        for block_id in ("cb_0007", "cb_0010"):
            self.assertEqual((by_id[block_id].issues[-1].type, by_id[block_id].issues[-1].note),
                             ("conflict", conflict["note"]))
            self.assertEqual(by_id[block_id].sme_questions[-1], conflict["sme_question"])
        self.assertEqual(len(by_id["cb_0010"].sme_questions), 2)

    def test_conflict_on_a_merged_block_lands_once_on_the_kept_block(self) -> None:
        def apply(payload: dict[str, Any]) -> None:
            payload["merges"].append({"keep_block_id": "cb_0008", "merged_block_ids": ["cb_0006"], "reason": "x"})
            payload["conflicts"].append({"block_ids": ["cb_0006", "cb_0008"], "note": "Khác nhau.",
                                         "sme_question": "Bước nào đúng?"})

        blocks, _links = apply_w1_reduce(reduce_response(apply), golden_blocks(), INDEX, "vi")
        kept = next(block for block in blocks if block.block_id == "cb_0008")
        self.assertEqual([issue.type for issue in kept.issues], ["duplicate", "conflict"])


class FallbackReduceTests(unittest.TestCase):
    def sections_and_blocks(self, count: int) -> tuple[list[IdmSection], list[IdmContentBlockV1]]:
        sections = [IdmSection(f"sec_{n:03d}", DOCUMENT_ID, f"{n}. MỤC SỐ {n}", (f"k{n}",), 10)
                    for n in range(1, count + 1)]
        blocks = [content_block(f"cb_{n:04d}", f"sec_{n:03d}", [f"k{n}"]) for n in range(1, count + 1)]
        return sections, blocks

    def test_one_objective_per_section_group_with_one_must_do(self) -> None:
        sections, blocks = self.sections_and_blocks(3)
        response = fallback_w1_reduce(blocks, sections, CONTEXT)
        self.assertEqual([item.statement for item in response.learning_objectives],
                         [f"Người học có thể áp dụng Mục số {n} trong công việc" for n in (1, 2, 3)])
        self.assertEqual({(item.bloom, item.origin) for item in response.learning_objectives},
                         {("apply", "ai_proposed")})
        self.assertEqual([(md.must_do_id, md.lo_id, md.statement) for md in response.must_dos],
                         [(f"md_{n}", f"lo_{n}", f"Thực hiện đúng Mục số {n}") for n in (1, 2, 3)])
        self.assertEqual([(link.block_id, link.lo_id, link.relation) for link in response.lo_links],
                         [(f"cb_{n:04d}", f"lo_{n}", "direct") for n in (1, 2, 3)])
        self.assertEqual(response.target_audience.origin, "ai_proposed")
        self.assertEqual(response.target_audience.description,
                         "Nhân sự cần áp dụng nội dung Xử lý khiếu nại khách hàng trong công việc")
        self.assertEqual(validate_w1_reduce(response, blocks, CONTEXT), [])

    def test_objectives_are_capped_and_every_block_stays_linked(self) -> None:
        sections, blocks = self.sections_and_blocks(11)
        response = fallback_w1_reduce(blocks, sections, CONTEXT)
        self.assertEqual(len(response.learning_objectives), 8)
        self.assertEqual(sorted(link.block_id for link in response.lo_links), [block.block_id for block in blocks])

    def test_client_audience_and_english_locale(self) -> None:
        sections, blocks = self.sections_and_blocks(3)
        english = context(locale="en", course_title_hint=None, target_audience="Front-line call centre agents")
        response = fallback_w1_reduce(blocks, sections, english)
        self.assertEqual((response.target_audience.description, response.target_audience.origin),
                         ("Front-line call centre agents", "client"))
        self.assertEqual(response.learning_objectives[0].statement, "The learner can apply Mục số 1 at work")
        self.assertEqual(response.must_dos[0].statement, "Correctly carry out Mục số 1")
        no_audience = fallback_w1_reduce(blocks, sections, context(locale="en", course_title_hint=None))
        self.assertIn("quy-trinh-khieu-nai.pdf", no_audience.target_audience.description)

    # Regression (fixed): client objectives are authoritative (spec §6.2, §7.1.4, §7.3) but the W1-reduce fallback
    # discards them and proposes its own AI objectives.
    def test_fallback_keeps_client_objectives_verbatim(self) -> None:
        response = fallback_w1_reduce(golden_blocks(), [SECTION], context(learning_objectives=CLIENT_LOS))
        self.assertEqual([item.statement for item in response.learning_objectives], CLIENT_LOS)
        self.assertEqual({item.origin for item in response.learning_objectives}, {"client"})

    # Regression (fixed): spec §7.3 fallback is one objective per top-level heading; the code makes one per section
    # group, so the single golden section yields one objective named after its page footer.
    def test_fallback_proposes_one_objective_per_top_level_heading(self) -> None:
        drafts, noise = fallback_w1_section(SECTION, INDEX, FURNITURE)
        blocks, _noise = rekey_blocks([SectionMapResult(SECTION, drafts, noise, "deterministic_fallback")], INDEX)
        response = fallback_w1_reduce(blocks, [SECTION], CONTEXT, INDEX)
        self.assertGreaterEqual(len(response.learning_objectives), IDM_LO_MIN_COUNT)
        self.assertFalse(any("Trang" in item.statement for item in response.learning_objectives))


class RunReduceTests(unittest.IsolatedAsyncioTestCase):
    async def reduce(self, provider: FakeIdmProvider, ctx: IdmProjectContextV1 = CONTEXT,
                     seconds: float = 600.0) -> tuple[IdmW1ReduceResponseV1, str, dict[str, int]]:
        return await run_w1_reduce(runtime(provider, seconds), golden_blocks(), [SECTION], ctx,
                                   tail_reserve_tokens=0)

    async def test_client_objectives_and_audience_are_copied_verbatim(self) -> None:
        provider = FakeIdmProvider({"IdmW1ReduceResponseV1": [W1_REDUCE]})
        ctx = context(learning_objectives=CLIENT_LOS, target_audience="Nhân viên tổng đài mới vào nghề")
        response, origin, _codes = await self.reduce(provider, ctx)
        self.assertEqual(origin, "provider")
        self.assertEqual([item.statement for item in response.learning_objectives], CLIENT_LOS)
        self.assertEqual({item.origin for item in response.learning_objectives}, {"client"})
        self.assertEqual((response.target_audience.description, response.target_audience.origin),
                         ("Nhân viên tổng đài mới vào nghề", "client"))

    async def test_ai_objectives_are_marked_ai_proposed(self) -> None:
        payload = edited(W1_REDUCE, lambda p: p["target_audience"].update(origin="client"))
        response = apply_client_context(IdmW1ReduceResponseV1.model_validate(payload), CONTEXT)
        self.assertEqual(response.target_audience.origin, "ai_proposed")
        self.assertEqual({item.origin for item in response.learning_objectives}, {"ai_proposed"})

    async def test_repair_then_accept_without_sending_fact_text(self) -> None:
        invalid = edited(W1_REDUCE, lambda p: p["learning_objectives"][0].update(statement="Người học có thể hiểu X."))
        provider = FakeIdmProvider({"IdmW1ReduceResponseV1": [invalid, W1_REDUCE]})
        _response, origin, found = await self.reduce(provider)
        self.assertEqual(origin, "provider")
        self.assertEqual(found, {"IDM_W0_LO_UNMEASURABLE": 1})
        self.assertIn("IDM_W0_LO_UNMEASURABLE", provider.calls[1]["prompt"])
        self.assertNotIn(FACTS[20].fact_text, provider.calls[0]["prompt"])

    async def test_failures_fall_back(self) -> None:
        unavailable = IdmProviderError("AI_PROVIDER_TIMEOUT", terminal=False)
        for responses, seconds, expected in (
            (["{", "{"], 600.0, {"IDM_RESPONSE_SCHEMA_INVALID": 2}),
            ([unavailable], 600.0, {"AI_PROVIDER_TIMEOUT": 1}),
            ([], 0.0, {"IDM_DEADLINE_EXCEEDED": 1}),
        ):
            with self.subTest(expected=expected):
                provider = FakeIdmProvider({"IdmW1ReduceResponseV1": list(responses)})
                _response, origin, found = await self.reduce(provider, seconds=seconds)
                self.assertEqual(origin, "deterministic_fallback")
                self.assertEqual(found, expected)

    async def test_terminal_error_propagates(self) -> None:
        provider = FakeIdmProvider({"IdmW1ReduceResponseV1": [IdmProviderError("AI_KEY_INVALID", terminal=True)]})
        with self.assertRaises(IdmProviderError):
            await self.reduce(provider)


if __name__ == "__main__":
    unittest.main()
