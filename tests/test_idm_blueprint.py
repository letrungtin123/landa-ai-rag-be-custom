"""W2 Content Blueprint (spec §7.4): consistency matrix, combine/separate, dispositions, holds, fallback."""

from __future__ import annotations

import itertools
import re
import time
import unittest
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

from app.idm.blueprint import (
    apply_w2,
    blocked_must_dos,
    build_dispositions,
    fallback_w2,
    hold_items,
    run_w2,
    validate_w2,
)
from app.idm.content_map import FactIndex, SectionMapResult, apply_w1_reduce, normalize_w1_section, rekey_blocks
from app.idm.contracts import (
    IdmBlockLoLinkV1,
    IdmBlueprintRowV1,
    IdmContentBlockV1,
    IdmLearningObjectiveV1,
    IdmMustDoV1,
    IdmNoiseDraftV1,
    IdmProjectContextV1,
    IdmW1ReduceResponseV1,
    IdmW1SectionResponseV1,
    IdmW2BlueprintResponseV1,
)
from app.idm.runtime import IdmProviderError, IdmRuntime
from app.idm.signals import compute_fact_signals, plan_idm_sections
from app.idm.validation import IdmIssue, errors
from tests.idm_golden import (
    W1_REDUCE,
    W1_SECTION,
    W2_BLUEPRINT,
    FakeIdmProvider,
    key,
    keys,
    mutate,
    project_context,
    source_facts,
)

# Spec §7.4 consistency matrix: classification -> (required placement, allowed treatments).
SPEC_MATRIX: dict[str, tuple[str, set[str]]] = {
    "must_do": ("course", {"keep", "condense", "rewrite", "combine", "separate"}),
    "must_know": ("course", {"keep", "condense", "rewrite", "combine", "separate"}),
    "reference": ("reference_job_aid", {"move_to_reference", "convert_to_job_aid"}),
    "nice_to_know": ("excluded", {"remove"}),
    "remove": ("excluded", {"remove"}),
}
TREATMENTS = ("keep", "condense", "rewrite", "combine", "separate", "move_to_reference", "convert_to_job_aid",
              "remove")
PLACEMENTS = ("course", "reference_job_aid", "excluded")


@dataclass(frozen=True)
class GoldenState:
    """Golden blocks after W1-map and W1-reduce, ready for W2."""

    index: FactIndex
    blocks: list[IdmContentBlockV1]
    links: list[IdmBlockLoLinkV1]
    noise: list[IdmNoiseDraftV1]
    objectives: list[IdmLearningObjectiveV1]
    must_dos: list[IdmMustDoV1]
    context: IdmProjectContextV1


def golden_state() -> GoldenState:
    facts = source_facts()
    signals = compute_fact_signals(facts)
    index = FactIndex.build(facts, signals)
    section = plan_idm_sections(facts, signals, {}).sections[0]
    drafts, noise = normalize_w1_section(IdmW1SectionResponseV1.model_validate(W1_SECTION), section.fact_keys, index)
    blocks, noise = rekey_blocks([SectionMapResult(section, drafts, noise, "provider")], index)
    reduce = IdmW1ReduceResponseV1.model_validate(W1_REDUCE)
    blocks, links = apply_w1_reduce(reduce, blocks, index, "vi")
    return GoldenState(index, blocks, links, noise, list(reduce.learning_objectives), list(reduce.must_dos),
                       IdmProjectContextV1.model_validate(project_context()))


def edited(payload: dict[str, Any], apply: Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
    def change(copy: dict[str, Any]) -> dict[str, Any]:
        apply(copy)
        return copy

    return mutate(payload, change)


def blueprint(apply: Callable[[dict[str, Any]], Any] | None = None) -> IdmW2BlueprintResponseV1:
    return IdmW2BlueprintResponseV1.model_validate(edited(W2_BLUEPRINT, apply) if apply else W2_BLUEPRINT)


def row_of(payload: dict[str, Any], block_id: str) -> dict[str, Any]:
    return next(row for row in payload["rows"] if row["block_id"] == block_id)


def applied_golden(state: GoldenState) -> tuple[list[IdmContentBlockV1], list[IdmBlockLoLinkV1],
                                                list[IdmBlueprintRowV1]]:
    return apply_w2(blueprint().rows, state.blocks, state.links, state.index)


def runtime(provider: Any, seconds: float = 600.0) -> IdmRuntime:
    return IdmRuntime(generate=provider, api_key="test-key", model="fake-model", locale="vi",
                      deadline=time.monotonic() + seconds)


class ValidateW2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = golden_state()

    def found(self, apply: Callable[[dict[str, Any]], Any] | None = None) -> list[IdmIssue]:
        return validate_w2(blueprint(apply), self.state.blocks, self.state.objectives, self.state.must_dos)

    def codes(self, apply: Callable[[dict[str, Any]], Any] | None = None) -> set[str]:
        return {issue.code for issue in self.found(apply)}

    def test_golden_blueprint_is_valid(self) -> None:
        self.assertEqual(self.found(), [])

    def test_consistency_matrix(self) -> None:
        for classification, placement, treatment in itertools.product(SPEC_MATRIX, PLACEMENTS, TREATMENTS):
            required, allowed = SPEC_MATRIX[classification]
            expected = placement != required or treatment not in allowed

            def apply(payload: dict[str, Any], values: tuple[str, str, str] = (classification, placement,
                                                                               treatment)) -> None:
                row_of(payload, "cb_0004").update(classification=values[0], placement=values[1],
                                                  treatment=values[2])

            with self.subTest(classification=classification, placement=placement, treatment=treatment):
                self.assertEqual("IDM_W2_INCONSISTENT_ROW" in self.codes(apply), expected)

    def test_inconsistent_references(self) -> None:
        cases: list[tuple[Callable[[dict[str, Any]], Any], str]] = [
            (lambda p: row_of(p, "cb_0004").update(lo_id="lo_9"), "rows[3]"),
            (lambda p: row_of(p, "cb_0004").update(must_do_ids=["md_9"]), "rows[3]"),
            (lambda p: row_of(p, "cb_0006").update(must_do_ids=[]), "rows[5]"),
        ]
        for apply, path in cases:
            with self.subTest(path=path):
                self.assertIn(IdmIssue("IDM_W2_INCONSISTENT_ROW", path), self.found(apply))

    def test_must_know_needs_a_must_do(self) -> None:
        found = self.found(lambda p: row_of(p, "cb_0003").update(must_do_ids=[]))
        self.assertIn(IdmIssue("IDM_W2_MUST_KNOW_WITHOUT_MUST_DO", "rows[2].must_do_ids"), found)

    def test_hold_rules(self) -> None:
        cases: list[Callable[[dict[str, Any]], Any]] = [
            lambda p: row_of(p, "cb_0004").update(hold=True),
            lambda p: row_of(p, "cb_0004").update(hold=True, hold_reason="Thiếu tiêu chí"),
            lambda p: row_of(p, "cb_0004").update(hold=True, sme_question="Tiêu chí nào?"),
            lambda p: row_of(p, "cb_0013").update(hold=True, hold_reason="Bảng cũ", sme_question="Còn đúng không?"),
        ]
        for apply in cases:
            with self.subTest():
                self.assertIn("IDM_W2_HOLD_INVALID", self.codes(apply))

    def test_combine_rules(self) -> None:
        def chained(payload: dict[str, Any]) -> None:
            row_of(payload, "cb_0007").update(treatment="combine", combine_into="cb_0006")
            row_of(payload, "cb_0006").update(treatment="combine", combine_into="cb_0008")

        def into_separated(payload: dict[str, Any]) -> None:
            row_of(payload, "cb_0007").update(treatment="combine", combine_into="cb_0006")
            row_of(payload, "cb_0006").update(treatment="separate", separate_into=[
                {"name": "Chào và lắng nghe", "intent": "do", "fact_keys": keys(5, 1, 3)},
                {"name": "Ghi nhận và xác minh", "intent": "do", "fact_keys": keys(5, 4, 5)}])

        cases: list[Callable[[dict[str, Any]], Any]] = [
            lambda p: row_of(p, "cb_0007").update(treatment="combine"),
            lambda p: row_of(p, "cb_0007").update(combine_into="cb_0006"),
            lambda p: row_of(p, "cb_0007").update(treatment="combine", combine_into="cb_0099"),
            lambda p: row_of(p, "cb_0007").update(treatment="combine", combine_into="cb_0005"),
            lambda p: row_of(p, "cb_0007").update(treatment="combine", combine_into="cb_0007"),
            chained,
            into_separated,
        ]
        for position, apply in enumerate(cases):
            with self.subTest(case=position):
                self.assertIn("IDM_W2_COMBINE_INVALID", self.codes(apply))
        valid = self.codes(lambda p: row_of(p, "cb_0007").update(treatment="combine", combine_into="cb_0006"))
        self.assertEqual(valid, set())

    def test_separate_must_partition_the_block(self) -> None:
        def separate(parts: list[list[str]], treatment: str = "separate") -> Callable[[dict[str, Any]], Any]:
            return lambda p: row_of(p, "cb_0005").update(treatment=treatment, separate_into=[
                {"name": f"Phần {n}", "intent": "know", "fact_keys": part} for n, part in enumerate(parts, 1)])

        self.assertEqual(self.codes(separate([keys(4, 1, 3), keys(4, 4, 5)])), set())
        bad = [separate([]), separate([keys(4, 1, 3), keys(4, 4, 5)], "keep"), separate([keys(4, 1, 5)]),
               separate([keys(4, 1, 3), [key(4, 4)]]), separate([keys(4, 1, 3), keys(4, 3, 5)]),
               separate([keys(4, 1, 3), [*keys(4, 4, 5), key(6, 2)]])]
        for position, apply in enumerate(bad):
            with self.subTest(case=position):
                self.assertIn("IDM_W2_SEPARATE_PARTITION", self.codes(apply))

    def test_exactly_one_row_per_block(self) -> None:
        self.assertIn(IdmIssue("IDM_W2_ROW_MISSING", "rows[cb_0014]"), self.found(lambda p: p["rows"].pop()))
        self.assertIn("IDM_W2_ROW_DUPLICATED", self.codes(lambda p: p["rows"].append(dict(p["rows"][0]))))
        unknown = self.found(lambda p: p["rows"].append({**p["rows"][0], "block_id": "cb_0099"}))
        self.assertIn(IdmIssue("IDM_W2_ROW_UNKNOWN", "rows[14].block_id"), unknown)

    def test_fallback_rows_always_validate(self) -> None:
        rows = fallback_w2(self.state.blocks, self.state.links, self.state.must_dos, "vi")
        response = IdmW2BlueprintResponseV1(rows=rows)
        self.assertEqual(validate_w2(response, self.state.blocks, self.state.objectives, self.state.must_dos), [])


class ApplyW2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = golden_state()

    def apply(self, change: Callable[[dict[str, Any]], Any]) -> tuple[list[IdmContentBlockV1],
                                                                       list[IdmBlockLoLinkV1],
                                                                       list[IdmBlueprintRowV1]]:
        response = blueprint(change)
        self.assertEqual(errors(validate_w2(response, self.state.blocks, self.state.objectives,
                                            self.state.must_dos)), [])
        return apply_w2(response.rows, self.state.blocks, self.state.links, self.state.index)

    def test_combine_moves_facts_into_the_target_in_document_order(self) -> None:
        blocks, links, rows = self.apply(lambda p: row_of(p, "cb_0006").update(treatment="combine",
                                                                              combine_into="cb_0008"))
        ids = [block.block_id for block in blocks]
        self.assertNotIn("cb_0006", ids)
        self.assertLess(ids.index("cb_0008"), ids.index("cb_0007"))
        target = blocks[ids.index("cb_0008")]
        self.assertEqual(target.fact_keys, [*keys(5, 1, 5), *keys(5, 10, 13)])
        self.assertEqual([row.block_id for row in rows], ids)
        self.assertNotIn("cb_0006", {link.block_id for link in links})

    def test_combine_carries_issues_gaps_and_questions(self) -> None:
        blocks, _links, _rows = self.apply(lambda p: row_of(p, "cb_0012").update(treatment="combine",
                                                                                combine_into="cb_0010"))
        target = next(block for block in blocks if block.block_id == "cb_0010")
        self.assertEqual([issue.type for issue in target.issues], ["duplicate", "conflict", "outdated"])
        self.assertEqual([gap.type for gap in target.gaps], ["missing_step"])
        self.assertEqual(len(target.sme_questions), 2)
        self.assertEqual(target.fact_keys, [*keys(7, 1, 4), *keys(9, 1, 2)])

    def test_separate_creates_new_blocks_continuing_the_numbering(self) -> None:
        parts = [{"name": "Thời hạn\n chung", "intent": "know", "fact_keys": list(reversed(keys(7, 1, 3)))},
                 {"name": "Ngoại lệ cấp 2", "intent": "decide", "fact_keys": [key(7, 4)]}]
        blocks, links, rows = self.apply(lambda p: row_of(p, "cb_0010").update(treatment="separate",
                                                                              separate_into=parts))
        by_id = {block.block_id: block for block in blocks}
        self.assertNotIn("cb_0010", by_id)
        self.assertEqual((by_id["cb_0015"].fact_keys, by_id["cb_0016"].fact_keys), (keys(7, 1, 3), [key(7, 4)]))
        self.assertEqual((by_id["cb_0015"].name, by_id["cb_0016"].intent), ("Thời hạn chung", "decide"))
        self.assertEqual(len(by_id["cb_0015"].issues), 2)
        self.assertEqual((by_id["cb_0016"].issues, by_id["cb_0016"].sme_questions), ([], []))
        ids = [block.block_id for block in blocks]
        self.assertEqual(ids[ids.index("cb_0009") + 1: ids.index("cb_0011")], ["cb_0015", "cb_0016"])
        new_rows = {row.block_id: row for row in rows if row.block_id in {"cb_0015", "cb_0016"}}
        for row in new_rows.values():
            self.assertEqual((row.classification, row.treatment, row.separate_into, row.hold, row.must_do_ids),
                             ("must_know", "condense", [], True, ["md_5"]))
        copied = {(link.block_id, link.lo_id, link.relation) for link in links if link.block_id in new_rows}
        self.assertEqual(copied, {("cb_0015", "lo_3", "direct"), ("cb_0016", "lo_3", "direct")})
        self.assertNotIn("cb_0010", {link.block_id for link in links})

    # Regression (fixed): spec §7.4 says new blocks continue after the last number; when the highest block was
    # combined away its id (cb_0014) is reused for a different block.
    def test_separate_never_reuses_an_id_removed_by_combine(self) -> None:
        def change(payload: dict[str, Any]) -> None:
            row_of(payload, "cb_0013").update(classification="must_know", placement="course", treatment="keep",
                                              must_do_ids=["md_1"])
            row_of(payload, "cb_0014").update(classification="must_know", placement="course", treatment="combine",
                                              must_do_ids=["md_3"], combine_into="cb_0013")
            row_of(payload, "cb_0005").update(treatment="separate", separate_into=[
                {"name": "Dấu hiệu cấp độ", "intent": "know", "fact_keys": keys(4, 1, 3)},
                {"name": "Cách xử lý cấp cao", "intent": "decide", "fact_keys": keys(4, 4, 5)}])

        blocks, _links, _rows = self.apply(change)
        self.assertEqual(sorted(block.block_id for block in blocks)[-2:], ["cb_0015", "cb_0016"])


class DispositionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = golden_state()
        self.blocks, self.links, self.rows = applied_golden(self.state)

    def test_exactly_one_disposition_per_fact_in_document_order(self) -> None:
        dispositions = build_dispositions(self.rows, self.blocks, self.state.noise, self.state.index)
        self.assertEqual([item.fact_key for item in dispositions], [fact.fact_key for fact in source_facts()])
        counts = Counter(item.disposition for item in dispositions)
        self.assertEqual(counts, {"course": 36, "reference_job_aid": 10, "hold": 6, "noise": 4, "remove": 4,
                                  "nice_to_know": 1})
        by_key = {item.fact_key: item for item in dispositions}
        self.assertEqual((by_key[key(7, 4)].disposition, by_key[key(7, 4)].block_id), ("hold", "cb_0010"))
        self.assertEqual((by_key[key(1, 2)].disposition, by_key[key(0, 2)].disposition), ("remove", "nice_to_know"))
        self.assertEqual(by_key[key(10, 3)].disposition, "reference_job_aid")
        self.assertEqual((by_key[key(0, 1)].block_id, by_key[key(0, 1)].reason), (None, "toc_or_title_only"))

    def test_block_ownership_wins_over_noise(self) -> None:
        noise = [*self.state.noise, IdmNoiseDraftV1(fact_key=key(1, 2), reason="unreadable")]
        dispositions = build_dispositions(self.rows, self.blocks, noise, self.state.index)
        self.assertEqual(len(dispositions), 61)
        self.assertEqual(next(item for item in dispositions if item.fact_key == key(1, 2)).disposition, "remove")

    def test_blocked_must_dos_are_server_computed(self) -> None:
        self.assertEqual(blocked_must_dos(self.rows, self.state.must_dos), ["md_5"])
        released = [row.model_copy(update={"hold": False}) if row.block_id == "cb_0010" else row for row in self.rows]
        self.assertEqual(blocked_must_dos(released, self.state.must_dos), [])
        excluded = [row.model_copy(update={"placement": "excluded"}) if row.block_id in {"cb_0009", "cb_0011",
                                                                                          "cb_0005"} else row
                    for row in released]
        # cb_0005 also served md_2, so both Must Dos lose their last non-held course block.
        self.assertEqual(blocked_must_dos(excluded, self.state.must_dos), ["md_2", "md_3"])

    def test_hold_items_list_every_held_block(self) -> None:
        items = hold_items(self.rows, self.blocks, ["md_5"])
        self.assertEqual([item.block_id for item in items], ["cb_0010", "cb_0012"])
        self.assertEqual(items[0].name, "Phải phản hồi khách hàng trong bao lâu?")
        self.assertEqual(items[0].sme_question, "Khiếu nại cấp 2 phải phản hồi trong 24 giờ hay 48 giờ?")
        self.assertEqual((items[0].blocked_must_do_ids, items[1].blocked_must_do_ids), (["md_5"], []))
        bare = [row.model_copy(update={"hold_reason": None, "sme_question": None}) for row in self.rows]
        self.assertEqual({(item.reason, item.sme_question) for item in hold_items(bare, self.blocks, [])},
                         {("-", "-")})


class FallbackW2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = golden_state()

    def classify(self, rows: list[IdmBlueprintRowV1]) -> dict[str, str]:
        return {row.block_id: row.classification for row in rows}

    def test_direct_links_become_must_know_the_rest_nice_to_know(self) -> None:
        rows = fallback_w2(self.state.blocks, self.state.links, self.state.must_dos, "vi")
        direct = {"cb_0004", "cb_0005", "cb_0006", "cb_0007", "cb_0008", "cb_0009", "cb_0010"}
        for row in rows:
            with self.subTest(block=row.block_id):
                expected = "must_know" if row.block_id in direct else "nice_to_know"
                self.assertEqual(row.classification, expected)
                self.assertEqual(row.placement, SPEC_MATRIX[expected][0])
                self.assertEqual(row.treatment, "condense" if expected == "must_know" else "remove")
                self.assertFalse(row.hold)
        by_id = {row.block_id: row for row in rows}
        self.assertEqual((by_id["cb_0004"].lo_id, by_id["cb_0004"].must_do_ids), ("lo_1", ["md_1", "md_2"]))
        self.assertEqual((by_id["cb_0001"].lo_id, by_id["cb_0001"].must_do_ids), (None, []))
        self.assertTrue(by_id["cb_0001"].rationale.startswith("Phương án dự phòng"))

    def test_reference_kind_and_job_aids_become_reference(self) -> None:
        links = [IdmBlockLoLinkV1(block_id=block_id, lo_id="lo_1", relation="direct")
                 for block_id in ("cb_0003", "cb_0013", "cb_0014")]
        checklist = [block.model_copy(update={"support_role": "checklist"}) if block.block_id == "cb_0003" else block
                     for block in self.state.blocks]
        rows = {row.block_id: row for row in fallback_w2(checklist, links, self.state.must_dos, "en")}
        for block_id in ("cb_0003", "cb_0013", "cb_0014"):
            self.assertEqual((rows[block_id].classification, rows[block_id].placement, rows[block_id].treatment,
                              rows[block_id].must_do_ids),
                             ("reference", "reference_job_aid", "convert_to_job_aid", []))
        self.assertTrue(rows["cb_0003"].rationale.startswith("Automatic fallback"))

    def test_without_usable_links_every_block_is_kept(self) -> None:
        for links in ([], [IdmBlockLoLinkV1(block_id="cb_0004", lo_id="lo_1", relation="supporting")]):
            with self.subTest(links=len(links)):
                rows = fallback_w2(self.state.blocks, links, self.state.must_dos, "vi")
                self.assertEqual({row.classification for row in rows}, {"must_know", "reference"})
                self.assertEqual({row.lo_id for row in rows}, {"lo_1"})

    def test_direct_link_to_an_objective_without_must_dos_does_not_count(self) -> None:
        must_dos = [md for md in self.state.must_dos if md.lo_id != "lo_2"]
        rows = self.classify(fallback_w2(self.state.blocks, self.state.links, must_dos, "vi"))
        self.assertEqual(rows["cb_0009"], "nice_to_know")
        self.assertEqual(rows["cb_0004"], "must_know")


class RunW2Tests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.state = golden_state()

    async def run_stage(self, provider: FakeIdmProvider, seconds: float = 600.0) -> Any:
        state = self.state
        return await run_w2(runtime(provider, seconds), context=state.context, blocks=state.blocks,
                            links=state.links, objectives=state.objectives, must_dos=state.must_dos,
                            index=state.index, tail_reserve_tokens=0)

    async def test_provider_rows_are_normalised(self) -> None:
        payload = edited(W2_BLUEPRINT, lambda p: row_of(p, "cb_0004").update(detail_level="Giữ  bảng\n nhóm",
                                                                             must_do_ids=["md_1", "md_1"]))
        result = await self.run_stage(FakeIdmProvider({"IdmW2BlueprintResponseV1": [payload]}))
        self.assertEqual((result.origin, result.codes), ("provider", {}))
        row = next(row for row in result.rows if row.block_id == "cb_0004")
        self.assertEqual((row.detail_level, row.must_do_ids), ("Giữ bảng nhóm", ["md_1"]))

    async def test_repair_then_fallback(self) -> None:
        missing = edited(W2_BLUEPRINT, lambda p: p["rows"].pop())
        provider = FakeIdmProvider({"IdmW2BlueprintResponseV1": [missing, W2_BLUEPRINT]})
        result = await self.run_stage(provider)
        self.assertEqual(result.origin, "provider")
        self.assertIn("IDM_W2_ROW_MISSING", provider.calls[1]["prompt"].split("REPAIR_REQUIREMENTS", 1)[1])
        failing = await self.run_stage(FakeIdmProvider({"IdmW2BlueprintResponseV1": ["{", missing]}))
        self.assertEqual(failing.origin, "deterministic_fallback")
        self.assertEqual(failing.codes, {"IDM_RESPONSE_SCHEMA_INVALID": 1, "IDM_W2_ROW_MISSING": 1})

    async def test_transient_budget_and_terminal_failures(self) -> None:
        transient = IdmProviderError("SERVICE_BUSY", terminal=False)
        self.assertEqual((await self.run_stage(FakeIdmProvider({"IdmW2BlueprintResponseV1": [transient]}))).origin,
                         "deterministic_fallback")
        self.assertEqual((await self.run_stage(FakeIdmProvider({}), seconds=0.0)).codes,
                         {"IDM_DEADLINE_EXCEEDED": 1})
        terminal = IdmProviderError("AI_KEY_INVALID", terminal=True)
        with self.assertRaises(IdmProviderError):
            await self.run_stage(FakeIdmProvider({"IdmW2BlueprintResponseV1": [terminal]}))

    async def test_large_catalog_is_split_by_objective_with_partial_fallback(self) -> None:
        def rows_for(prompt: str) -> dict[str, Any]:
            catalog = prompt.split("\n<BLOCK_CATALOG>\n", 1)[1]
            ids = set(re.findall(r'"block_id":"(cb_\d{4})"', catalog))
            return {"rows": [row for row in W2_BLUEPRINT["rows"] if row["block_id"] in ids]}

        provider = FakeIdmProvider({"IdmW2BlueprintResponseV1": [rows_for, "{", "{", rows_for, rows_for]})
        with patch("app.idm.blueprint.IDM_W2_SINGLE_CALL_MAX_BLOCKS", 5):
            result = await self.run_stage(provider)
        self.assertEqual(result.origin, "partial_fallback")
        self.assertEqual(len(provider.calls), 5)
        self.assertEqual(len(result.rows), 14)
        batches = [set(re.findall(r'"block_id":"(cb_\d{4})"', call["prompt"])) for call in provider.calls]
        self.assertEqual(batches[0], {"cb_0003", "cb_0004", "cb_0005"})
        self.assertEqual(batches[-1], {"cb_0001", "cb_0002", "cb_0013", "cb_0014"})
        by_id = {row.block_id: row for row in result.rows}
        self.assertTrue(by_id["cb_0009"].rationale.startswith("Phương án dự phòng"))


if __name__ == "__main__":
    unittest.main()
