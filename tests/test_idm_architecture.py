"""W4 course architecture, assembly, block scopes, skeleton projection and notes (spec §5.3, §7.5, §8.1, §8.4)."""

from __future__ import annotations

import hashlib
import re
import time
import unittest
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from app.idm.architecture import (
    AssemblyInput,
    assemble_idm_course_design,
    bind_idm_course_skeleton,
    block_scope_key,
    fallback_w4,
    pending_objective_ids,
    project_course_skeleton,
    run_w4,
    validate_w4,
)
from app.idm.blueprint import blocked_must_dos, build_dispositions, hold_items
from app.idm.contracts import (
    IdmAudienceV1,
    IdmBlockLoLinkV1,
    IdmBlueprintRowV1,
    IdmContentBlockV1,
    IdmCourseDesignV1,
    IdmMustDoV1,
    IdmW4CourseResponseV1,
    design_hash_of,
)
from app.idm.notes import (
    HoldNote,
    SkippedBlockNote,
    build_course_notes,
    build_lesson_notes,
    build_module_notes,
    scrub_internal_ids,
)
from app.idm.runtime import IdmProviderError, IdmRuntime, IdmStageError
from app.idm.validation import errors
from app.lesson_author_orchestration_v2 import CourseSkeletonV2
from tests.idm_golden import SNAPSHOT_HASH, W4_COURSE, FakeIdmProvider, key, mutate, source_facts
from tests.test_idm_blueprint import GoldenState, applied_golden, edited, golden_state

AI_SUFFIX_VI = " (Đề xuất bởi AI — cần xác nhận)"
AI_SUFFIX_EN = " (Proposed by AI — please confirm)"
INTERNAL_ID_RE = re.compile(r"\b(cb_\d{4}|sec_\d{3}|lo_\d{1,2}|md_\d{1,2}|lsn_\d{3}|mod_\d{2}|idmcb_|scope3_)")


@dataclass(frozen=True)
class W4Inputs:
    state: GoldenState
    blocks: list[IdmContentBlockV1]
    links: list[IdmBlockLoLinkV1]
    rows: list[IdmBlueprintRowV1]
    course: list[IdmContentBlockV1]
    reference: list[IdmContentBlockV1]
    blocked: list[str]


def w4_inputs() -> W4Inputs:
    state = golden_state()
    blocks, links, rows = applied_golden(state)
    row_by_id = {row.block_id: row for row in rows}
    course = [block for block in blocks if row_by_id[block.block_id].placement == "course"
              and not row_by_id[block.block_id].hold]
    reference = [block for block in blocks if row_by_id[block.block_id].placement == "reference_job_aid"]
    return W4Inputs(state, blocks, links, rows, course, reference, blocked_must_dos(rows, state.must_dos))


def course_plan(apply: Callable[[dict[str, Any]], Any] | None = None) -> IdmW4CourseResponseV1:
    return IdmW4CourseResponseV1.model_validate(edited(W4_COURSE, apply) if apply else W4_COURSE)


def lesson_of(payload: dict[str, Any], lesson_key: str) -> dict[str, Any]:
    return next(lesson for module in payload["modules"] for lesson in module["lessons"]
                if lesson["lesson_key"] == lesson_key)


def assembly_input(inputs: W4Inputs, plan: IdmW4CourseResponseV1 | None = None, **changes: Any) -> AssemblyInput:
    state, facts = inputs.state, source_facts()
    data = AssemblyInput(
        source_snapshot_hash=SNAPSHOT_HASH, context=state.context, snapshot_fact_keys=[f.fact_key for f in facts],
        fact_text={f.fact_key: f.fact_text for f in facts}, fact_source_ref={f.fact_key: f.source_ref for f in facts},
        audience=IdmAudienceV1(description="Nhân viên chăm sóc khách hàng tuyến đầu", origin="ai_proposed"),
        objectives=state.objectives, must_dos=state.must_dos, blocks=inputs.blocks, links=inputs.links,
        rows=inputs.rows, blocked=inputs.blocked, holds=hold_items(inputs.rows, inputs.blocks, inputs.blocked),
        plan=plan or course_plan(),
        dispositions=build_dispositions(inputs.rows, inputs.blocks, state.noise, state.index),
        stage_origins={"w4": "provider", "w1_map": "provider", "w2": "provider", "w1_reduce": "provider"},
    )
    return replace(data, **changes)


def runtime(provider: Any, seconds: float = 600.0) -> IdmRuntime:
    return IdmRuntime(generate=provider, api_key="test-key", model="fake-model", locale="vi",
                      deadline=time.monotonic() + seconds)


class ValidateW4Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = w4_inputs()

    def found(self, plan: IdmW4CourseResponseV1, duration: int | None = None) -> list[Any]:
        inputs = self.inputs
        return validate_w4(plan, course_blocks=inputs.course, reference_blocks=inputs.reference,
                           must_dos=inputs.state.must_dos, blocked=inputs.blocked,
                           objectives=inputs.state.objectives, duration_target_minutes=duration)

    def codes(self, apply: Callable[[dict[str, Any]], Any]) -> set[str]:
        return {issue.code for issue in self.found(course_plan(apply))}

    def test_golden_plan_is_valid(self) -> None:
        self.assertEqual(self.found(course_plan()), [])

    def test_every_error_code(self) -> None:
        def move_reference(payload: dict[str, Any]) -> None:
            lesson_of(payload, "lsn_005")["block_ids"] = ["cb_0014"]
            lesson_of(payload, "lsn_004")["block_ids"].append("cb_0013")

        def job_aid_with_course_block(payload: dict[str, Any]) -> None:
            lesson_of(payload, "lsn_001")["block_ids"] = ["cb_0003"]
            lesson_of(payload, "lsn_005")["block_ids"].append("cb_0004")

        cases: list[tuple[Callable[[dict[str, Any]], Any], str]] = [
            (lambda p: lesson_of(p, "lsn_002").update(primary_must_do_id=None), "IDM_W4_MUST_DO_UNPLACED"),
            (lambda p: lesson_of(p, "lsn_002").update(primary_must_do_id="md_9"), "IDM_W4_MUST_DO_UNPLACED"),
            (lambda p: lesson_of(p, "lsn_004").update(secondary_must_do_ids=["md_5"]), "IDM_W4_MUST_DO_UNPLACED"),
            (lambda p: lesson_of(p, "lsn_002").update(secondary_must_do_ids=["md_1"]), "IDM_W4_MUST_DO_DUPLICATED"),
            (lambda p: lesson_of(p, "lsn_001").update(block_ids=["cb_0003"]), "IDM_W4_BLOCK_UNPLACED"),
            (lambda p: lesson_of(p, "lsn_002")["block_ids"].append("cb_0003"), "IDM_W4_BLOCK_DUPLICATED"),
            (lambda p: lesson_of(p, "lsn_002")["block_ids"].append("cb_0099"), "IDM_W4_BLOCK_UNKNOWN"),
            (lambda p: lesson_of(p, "lsn_004")["block_ids"].append("cb_0010"), "IDM_W4_BLOCK_UNKNOWN"),
            (lambda p: lesson_of(p, "lsn_004")["block_ids"].append("cb_0002"), "IDM_W4_BLOCK_UNKNOWN"),
            (move_reference, "IDM_W4_REFERENCE_IN_LEARNING_LESSON"),
            (job_aid_with_course_block, "IDM_W4_JOB_AID_LESSON_INVALID"),
            (lambda p: lesson_of(p, "lsn_005").update(primary_must_do_id="md_4"), "IDM_W4_JOB_AID_LESSON_INVALID"),
            (lambda p: lesson_of(p, "lsn_005").update(secondary_must_do_ids=["md_4"]),
             "IDM_W4_JOB_AID_LESSON_INVALID"),
            (lambda p: p["modules"][1].update(title="Tổng quan"), "IDM_W4_TITLE_GENERIC"),
            (lambda p: lesson_of(p, "lsn_003").update(title="Phần 1"), "IDM_W4_TITLE_GENERIC"),
            (lambda p: p["modules"][0].update(module_key="mod_02"), "IDM_W4_KEY_SEQUENCE"),
            (lambda p: lesson_of(p, "lsn_002").update(lesson_key="lsn_007"), "IDM_W4_KEY_SEQUENCE"),
            (lambda p: p["modules"][0].update(lo_ids=["lo_9"]), "IDM_W4_LO_UNKNOWN"),
        ]
        for position, (apply, code) in enumerate(cases):
            with self.subTest(case=position, code=code):
                found = self.codes(apply)
                self.assertIn(code, found)

    def test_course_size_limit(self) -> None:
        lesson = W4_COURSE["modules"][0]["lessons"][0]
        modules = [{**W4_COURSE["modules"][0], "module_key": f"mod_{m + 1:02d}", "lessons": [
            {**lesson, "lesson_key": f"lsn_{m * 25 + n + 1:03d}"} for n in range(25)]} for m in range(5)]
        plan = course_plan(lambda p: p.update(modules=modules))
        self.assertIn("IDM_W4_LIMIT", {issue.code for issue in self.found(plan)})

    def test_duration_over_target_is_only_a_warning(self) -> None:
        # Golden lessons sum to 35 minutes; the warning fires above 1.2 x target.
        for target, expected in ((29, True), (30, False), (None, False)):
            with self.subTest(target=target):
                found = self.found(course_plan(), target)
                self.assertEqual(errors(found), [])
                self.assertEqual(any(issue.code == "IDM_W4_DURATION_OVER_TARGET" and issue.severity == "warning"
                                     for issue in found), expected)


class FallbackW4Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = w4_inputs()

    def fallback(self, *, must_dos: list[IdmMustDoV1] | None = None, course: list[IdmContentBlockV1] | None = None,
                 reference: list[IdmContentBlockV1] | None = None, context: Any = None,
                 rows: list[IdmBlueprintRowV1] | None = None) -> IdmW4CourseResponseV1:
        inputs = self.inputs
        return fallback_w4(context=context or inputs.state.context, objectives=inputs.state.objectives,
                           must_dos=must_dos or inputs.state.must_dos, rows=rows or inputs.rows,
                           course_blocks=inputs.course if course is None else course,
                           reference_blocks=inputs.reference if reference is None else reference,
                           blocked=inputs.blocked)

    def valid(self, plan: IdmW4CourseResponseV1, must_dos: list[IdmMustDoV1] | None = None,
              course: list[IdmContentBlockV1] | None = None,
              reference: list[IdmContentBlockV1] | None = None) -> list[Any]:
        inputs = self.inputs
        return validate_w4(plan, course_blocks=inputs.course if course is None else course,
                           reference_blocks=inputs.reference if reference is None else reference,
                           must_dos=must_dos or inputs.state.must_dos, blocked=inputs.blocked,
                           objectives=inputs.state.objectives, duration_target_minutes=None)

    def test_one_module_per_objective_and_one_lesson_per_must_do(self) -> None:
        plan = self.fallback()
        self.assertEqual(self.valid(plan), [])
        self.assertEqual([module.lo_ids for module in plan.modules], [["lo_1"], ["lo_2"], ["lo_3"]])
        layout = [(lesson.lesson_key, lesson.kind, lesson.primary_must_do_id, lesson.block_ids)
                  for module in plan.modules for lesson in module.lessons]
        self.assertEqual(layout, [
            ("lsn_001", "learning", "md_1", ["cb_0003", "cb_0004"]), ("lsn_002", "learning", "md_2", ["cb_0005"]),
            ("lsn_003", "learning", "md_3", ["cb_0009", "cb_0011"]),
            ("lsn_004", "learning", "md_4", ["cb_0006", "cb_0007", "cb_0008"]),
            ("lsn_005", "job_aid", None, ["cb_0013", "cb_0014"]),
        ])
        lessons = [lesson for module in plan.modules for lesson in module.lessons]
        self.assertEqual({lesson.est_minutes for lesson in lessons}, {5})
        self.assertEqual(lessons[0].title, "Phân loại khiếu nại theo nhóm")
        self.assertEqual(lessons[-1].title, "Tài liệu tra cứu nhanh")
        self.assertEqual(plan.course_title, "Xử lý khiếu nại khách hàng")

    def test_must_do_without_blocks_shares_a_lesson_as_secondary(self) -> None:
        extra = IdmMustDoV1(must_do_id="md_6", lo_id="lo_1", statement="Ghi nhận mã khiếu nại", kind="do",
                            bloom="apply")
        must_dos = [*self.inputs.state.must_dos, extra]
        plan = self.fallback(must_dos=must_dos)
        self.assertEqual(plan.modules[0].lessons[0].secondary_must_do_ids, ["md_6"])
        self.assertEqual(self.valid(plan, must_dos=must_dos), [])

    def test_only_reference_blocks_make_a_job_aid_module(self) -> None:
        plan = self.fallback(course=[], context=self.inputs.state.context.model_copy(update={"locale": "en"}))
        self.assertEqual(len(plan.modules), 1)
        self.assertEqual(plan.modules[0].lessons[0].kind, "job_aid")
        self.assertEqual(plan.modules[0].title, "Quick reference")
        self.assertTrue(plan.course_summary.startswith("This course helps learners"))

    def test_nothing_to_teach_is_a_stage_error(self) -> None:
        with self.assertRaises(IdmStageError) as caught:
            self.fallback(course=[], reference=[])
        self.assertEqual(caught.exception.code, "IDM_COURSE_HAS_NO_TEACHABLE_CONTENT")

    def many_blocks(self, count: int) -> list[IdmContentBlockV1]:
        template = self.inputs.course[0]
        return [template.model_copy(update={"block_id": f"cb_{n:04d}", "fact_keys": [f"k{n}"]})
                for n in range(100, 100 + count)]

    # Regression (fixed): the deterministic fallback crashes with a raw ValidationError when one Must Do owns more
    # than 40 course blocks (lesson block_ids max 40); spec §7.5 fallback must always yield a plan.
    def test_fallback_handles_more_than_forty_blocks_per_must_do(self) -> None:
        course = self.many_blocks(45)
        rows = [IdmBlueprintRowV1(block_id=block.block_id, lo_id="lo_1", must_do_ids=["md_1"],
                                  classification="must_know", placement="course", treatment="keep",
                                  detail_level="x", hold=False, rationale="x") for block in course]
        plan = self.fallback(course=course, reference=[], rows=rows)
        self.assertEqual([issue.code for issue in self.valid(plan, course=course, reference=[])
                          if issue.code.startswith("IDM_W4_BLOCK")], [])

    # Regression (fixed): the fallback job aid keeps only the first 40 reference blocks; the rest are unplaced
    # (IDM_W4_BLOCK_UNPLACED) and assembly then fails with IDM_ACCOUNTING_INVALID.
    def test_fallback_places_every_reference_block(self) -> None:
        reference = self.many_blocks(45)
        plan = self.fallback(reference=reference)
        self.assertEqual(self.valid(plan, reference=reference), [])

    def test_fallback_titles_are_topics_not_objective_sentences(self) -> None:
        # QC course 234653: every chapter was "Người học có thể áp dụng {heading} trong công việc" and every
        # section "Thực hiện đúng {heading}" because fallback_w4 copied the fallback LO / Must Do sentences.
        inputs = self.inputs
        objectives = [objective.model_copy(update={
            "statement": f"Người học có thể áp dụng Chủ đề {n} trong công việc"})
            for n, objective in enumerate(inputs.state.objectives, start=1)]
        must_dos = [must_do.model_copy(update={"statement": f"Thực hiện đúng Chủ đề {must_do.must_do_id}"})
                    for must_do in inputs.state.must_dos]
        plan = fallback_w4(context=inputs.state.context, objectives=objectives, must_dos=must_dos, rows=inputs.rows,
                           course_blocks=inputs.course, reference_blocks=inputs.reference, blocked=inputs.blocked)
        self.assertEqual([module.title for module in plan.modules], ["Chủ đề 1", "Chủ đề 2", "Chủ đề 3"])
        self.assertEqual([module.performance_goal for module in plan.modules],
                         [objective.statement for objective in objectives])
        titles = [lesson.title for module in plan.modules for lesson in module.lessons]
        self.assertFalse(any(title.startswith(("Thực hiện đúng", "Người học")) for title in titles))
        self.assertEqual(titles[0], "Chủ đề md_1")
        # D14: the fallback cannot promise a practice in every section.
        self.assertIn("chưa có bài luyện tập được đánh dấu", plan.assessment_strategy)


class RunW4Tests(unittest.IsolatedAsyncioTestCase):
    async def run_stage(self, provider: FakeIdmProvider, *, course: bool = True, seconds: float = 600.0) -> Any:
        inputs = w4_inputs()
        rows = inputs.rows if course else [row.model_copy(update={"placement": "excluded"})
                                           if row.placement == "course" else row for row in inputs.rows]
        return await run_w4(runtime(provider, seconds), context=inputs.state.context,
                            audience=IdmAudienceV1(description="Nhân viên tổng đài", origin="ai_proposed"),
                            objectives=inputs.state.objectives, must_dos=inputs.state.must_dos,
                            blocks=inputs.blocks, rows=rows, blocked=inputs.blocked)

    async def test_provider_plan_is_normalised(self) -> None:
        payload = edited(W4_COURSE, lambda p: p.update(course_title="Xử lý   khiếu nại\n đúng quy trình"))
        result = await self.run_stage(FakeIdmProvider({"IdmW4CourseResponseV1": [payload]}))
        self.assertEqual((result.origin, result.plan.course_title), ("provider", "Xử lý khiếu nại đúng quy trình"))

    async def test_repair_fallback_and_failures(self) -> None:
        unplaced = edited(W4_COURSE, lambda p: lesson_of(p, "lsn_001").update(block_ids=["cb_0003"]))
        provider = FakeIdmProvider({"IdmW4CourseResponseV1": [unplaced, W4_COURSE]})
        self.assertEqual((await self.run_stage(provider)).origin, "provider")
        self.assertIn("IDM_W4_BLOCK_UNPLACED", provider.calls[1]["prompt"].split("REPAIR_REQUIREMENTS", 1)[1])
        failing = await self.run_stage(FakeIdmProvider({"IdmW4CourseResponseV1": ["{", unplaced]}))
        self.assertEqual(failing.origin, "deterministic_fallback")
        busy = IdmProviderError("SERVICE_BUSY", terminal=False)
        self.assertEqual((await self.run_stage(FakeIdmProvider({"IdmW4CourseResponseV1": [busy]}))).codes,
                         {"SERVICE_BUSY": 1})
        self.assertEqual((await self.run_stage(FakeIdmProvider({}), seconds=0.0)).codes,
                         {"IDM_DEADLINE_EXCEEDED": 1})
        with self.assertRaises(IdmProviderError):
            await self.run_stage(FakeIdmProvider({"IdmW4CourseResponseV1": [
                IdmProviderError("AI_KEY_INVALID", terminal=True)]}))

    async def test_no_course_blocks_skips_the_call(self) -> None:
        provider = FakeIdmProvider({})
        result = await self.run_stage(provider, course=False)
        self.assertEqual((result.origin, provider.calls), ("deterministic_fallback", []))


class BlockScopeKeyTests(unittest.TestCase):
    def test_scope_key_matches_the_spec_formula(self) -> None:
        fact_keys = [key(4, 3), key(4, 1), key(4, 2)]
        value = block_scope_key(SNAPSHOT_HASH, "cb_0005", fact_keys)
        self.assertRegex(value, r"^idmcb_[0-9a-f]{32}$")
        material = SNAPSHOT_HASH + "\x1e" + "cb_0005" + "\x1e" + ",".join(sorted(fact_keys))
        self.assertEqual(value, "idmcb_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:32])
        self.assertEqual(value, block_scope_key(SNAPSHOT_HASH, "cb_0005", sorted(fact_keys)))
        self.assertNotEqual(value, block_scope_key(SNAPSHOT_HASH, "cb_0006", fact_keys))
        self.assertNotEqual(value, block_scope_key("0" * 64, "cb_0005", fact_keys))
        self.assertNotEqual(value, block_scope_key(SNAPSHOT_HASH, "cb_0005", fact_keys[:2]))


class AssemblyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = w4_inputs()

    def test_assembly_builds_scopes_in_lesson_order_and_hashes_the_design(self) -> None:
        design = assemble_idm_course_design(assembly_input(self.inputs))
        lesson_blocks = [block_id for module in design.modules for lesson in module.lessons
                         for block_id in lesson.block_ids]
        self.assertEqual([scope.block_id for scope in design.block_scopes], lesson_blocks)
        texts = {fact.fact_key: fact.fact_text for fact in source_facts()}
        blocks = {block.block_id: block for block in design.blocks}
        for scope in design.block_scopes:
            block = blocks[scope.block_id]
            self.assertEqual(scope.scope_key, block_scope_key(SNAPSHOT_HASH, block.block_id, block.fact_keys))
            self.assertEqual((scope.title, scope.fact_keys, scope.fact_count),
                             (block.name, block.fact_keys, len(block.fact_keys)))
            self.assertEqual(scope.content_chars, sum(len(texts[item]) for item in block.fact_keys))
        self.assertEqual(design.block_scopes[0].source_ref, "src-2")
        self.assertEqual(list(design.stage_origins), ["w1_map", "w1_reduce", "w2", "w4"])
        self.assertEqual(design.design_hash, design_hash_of(design.model_dump(mode="json")))
        self.assertEqual(design.design_hash, assemble_idm_course_design(assembly_input(self.inputs)).design_hash)

    def test_accounting_errors_are_internal_stage_errors(self) -> None:
        dispositions = assembly_input(self.inputs).dispositions
        foreign = dispositions[0].model_copy(update={"fact_key": "d9-c0-f1"})
        without_lesson = course_plan(lambda p: lesson_of(p, "lsn_001").update(block_ids=["cb_0003"]))
        cases = {
            "missing": {"dispositions": dispositions[1:]},
            "duplicated": {"dispositions": [*dispositions, dispositions[0]]},
            "foreign": {"dispositions": [*dispositions[1:], foreign]},
            "course fact outside every lesson": {"plan": without_lesson},
        }
        for name, changes in cases.items():
            with self.subTest(case=name), self.assertRaises(IdmStageError) as caught:
                assemble_idm_course_design(assembly_input(self.inputs, **changes))
            self.assertEqual(caught.exception.code, "IDM_ACCOUNTING_INVALID")


class SkeletonProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = w4_inputs()

    def project(self, **changes: Any) -> tuple[IdmCourseDesignV1, CourseSkeletonV2]:
        design = assemble_idm_course_design(assembly_input(self.inputs, **changes))
        return design, project_course_skeleton(design, course_plan())

    def test_one_chapter_per_module_owning_its_block_scopes(self) -> None:
        design, skeleton = self.project()
        CourseSkeletonV2.model_validate(skeleton.model_dump())
        self.assertEqual([chapter.chapter_key for chapter in skeleton.chapters],
                         ["chapter-1", "chapter-2", "chapter-3"])
        self.assertEqual([chapter.order for chapter in skeleton.chapters], [0, 1, 2])
        statements = {item.lo_id: item.statement for item in design.learning_objectives}
        scope_of = {scope.block_id: scope.scope_key for scope in design.block_scopes}
        for chapter, module in zip(skeleton.chapters, design.modules, strict=True):
            self.assertEqual((chapter.title, chapter.objective), (module.title, module.performance_goal))
            self.assertEqual(chapter.learning_outcomes, [statements[lo_id] for lo_id in module.lo_ids])
            self.assertEqual(chapter.source_scope_ids, [scope_of[block_id] for lesson in module.lessons
                                                        for block_id in lesson.block_ids])
        assigned = [scope for chapter in skeleton.chapters for scope in chapter.source_scope_ids]
        self.assertEqual(sorted(assigned), sorted(scope_of.values()))
        self.assertEqual(skeleton.learning_outcomes, [item.statement for item in design.learning_objectives])
        self.assertEqual(skeleton.title, W4_COURSE["course_title"])

    def test_fully_held_objective_is_not_a_learner_outcome(self) -> None:
        # QC course 234653 (R3): LO3's only Must Do was on Hold, yet LO3 was shown as a course outcome.
        design, skeleton = self.project(blocked=["md_3", "md_5"])
        statements = {item.lo_id: item.statement for item in design.learning_objectives}
        self.assertEqual(pending_objective_ids(design.learning_objectives, design.must_dos,
                                               design.blocked_must_do_ids), ["lo_2"])
        self.assertEqual(skeleton.learning_outcomes, [statements["lo_1"], statements["lo_3"]])
        self.assertNotIn(statements["lo_2"], [item for chapter in skeleton.chapters
                                              for item in chapter.learning_outcomes])
        self.assertEqual(skeleton.chapters[1].learning_outcomes, [statements["lo_1"]])
        self.assertEqual([item.lo_id for item in design.learning_objectives], ["lo_1", "lo_2", "lo_3"])
        self.assertIn("Mục tiêu học tập chờ SME", design.notes.course)
        self.assertIn(statements["lo_2"], design.notes.course)
        self.assertIn("(chờ SME — chưa có bài dạy)", design.notes.modules["mod_02"])
        # A partly held objective (lo_3: md_5 held, md_4 taught) stays a learner outcome.
        _design, golden = self.project()
        self.assertEqual(golden.learning_outcomes, list(statements.values()))
        # Every objective held: the projection keeps them all (the skeleton needs at least one outcome).
        _design, everything = self.project(blocked=["md_1", "md_2", "md_3", "md_4", "md_5"])
        self.assertEqual(everything.learning_outcomes, list(statements.values()))

    def test_target_audience_suffix_only_when_ai_proposed(self) -> None:
        _design, skeleton = self.project()
        self.assertEqual(skeleton.target_audience, "Nhân viên chăm sóc khách hàng tuyến đầu" + AI_SUFFIX_VI)
        self.assertEqual(skeleton.assumptions, ["IDM pipeline idm-1", "LO đề xuất bởi AI — cần xác nhận"])
        client = IdmAudienceV1(description="Nhân viên tổng đài mới", origin="client")
        objectives = [item.model_copy(update={"origin": "client"}) for item in self.inputs.state.objectives]
        _design, skeleton = self.project(audience=client, objectives=objectives)
        self.assertEqual(skeleton.target_audience, "Nhân viên tổng đài mới")
        self.assertEqual(skeleton.assumptions, ["IDM pipeline idm-1"])

    def test_long_ai_audience_stays_within_the_limit(self) -> None:
        english = self.inputs.state.context.model_copy(update={"locale": "en"})
        audience = IdmAudienceV1(description="x" * 2_000, origin="ai_proposed")
        _design, skeleton = self.project(audience=audience, context=english)
        self.assertLessEqual(len(skeleton.target_audience), 2_000)
        self.assertTrue(skeleton.target_audience.endswith(AI_SUFFIX_EN))
        self.assertIn("Objectives proposed by AI — please confirm", skeleton.assumptions)

    def test_binding_rejects_a_missing_or_extra_scope(self) -> None:
        design, skeleton = self.project()
        first = skeleton.chapters[0]
        short = skeleton.model_copy(update={"chapters": [first.model_copy(update={
            "source_scope_ids": first.source_scope_ids[1:]}), *skeleton.chapters[1:]]})
        fewer = skeleton.model_copy(update={"chapters": skeleton.chapters[:2]})
        for broken in (short, fewer):
            with self.assertRaises(IdmStageError):
                bind_idm_course_skeleton(broken, design)


class NotesTests(unittest.TestCase):
    def assert_clean(self, text: str, limit: int, known: list[str]) -> None:
        self.assertLessEqual(len(text), limit)
        self.assertNotIn("<", text)
        self.assertNotIn(">", text)
        self.assertIsNone(INTERNAL_ID_RE.search(text))
        self.assertFalse(any(item in text for item in known))

    def course_notes(self, locale: str, hold_count: int = 1, **changes: Any) -> str:
        holds = [(f"Thời hạn <phản hồi> cb_0010 số {n}", "Mâu thuẫn 24h > 48h [d1-c7-f2]", "md_5: 24 hay 48 giờ?")
                 for n in range(hold_count)]
        arguments: dict[str, Any] = {
            "locale": locale, "ai_proposed": True, "total_minutes": 35, "lesson_count": 5, "module_count": 3,
            "reference_block_count": 2, "excluded_block_count": 2, "holds": holds,
            "sme_questions": ["Câu hỏi lo_1 về scope3_00_a?", "Câu hỏi lo_1 về scope3_00_a?", ""],
            "fallback_stages": ["w2"], "known_ids": ["d1-c7-f2"], **changes,
        }
        return build_course_notes(**arguments)

    def test_course_notes_vi(self) -> None:
        text = self.course_notes("vi")
        self.assert_clean(text, 7_000, ["d1-c7-f2"])
        for expected in ("[Thiết kế theo quy trình ID — idm-1]", "do AI đề xuất", "35 phút (5 mục, 3 chương)",
                         "Cần chuyên gia bổ sung (Hold) — chờ SME xác nhận (1)", "Câu hỏi cho SME:", "24 hay 48 giờ?",
                         "Câu hỏi khác cho SME:", "phương án dự phòng",
                         "Cấu trúc được thiết kế theo Must Do, không theo mục lục tài liệu."):
            self.assertIn(expected, text)
        self.assertEqual(text.count("Câu hỏi lo"), 0)
        self.assertEqual(text.count("Câu hỏi về"), 1)

    def test_course_notes_list_blocked_must_dos_pending_objectives_and_nice_to_know(self) -> None:
        # QC course 234653 (R3, R4): the author sees what Hold blocks and what was left out.
        hold = HoldNote("Thời hạn phản hồi khiếu nại", "Mâu thuẫn 24 giờ và 48 giờ.", "Cấp 2 phản hồi trong bao lâu?",
                        ("Phản hồi khách hàng đúng thời hạn",))
        expectations = {
            "vi": ("Must Do chưa dạy được: Phản hồi khách hàng đúng thời hạn.",
                   "• Mục tiêu học tập chờ SME (chưa hiển thị là kết quả đầu ra của khoá học):",
                   "• Nội dung tham khảo đã lược (Nice to know) — 2 khối, tác giả có thể bổ sung thủ công:"),
            "en": ("Must Do not taught yet: Phản hồi khách hàng đúng thời hạn.",
                   "• Learning objectives awaiting the SME (not shown as course outcomes):",
                   "• Reference content left out (Nice to know) — 2 blocks the author may add back:"),
        }
        for locale, labels in expectations.items():
            text = self.course_notes(locale, holds=[hold], pending_objectives=["Người học có thể phản hồi đúng hạn."],
                                     nice_to_know=[SkippedBlockNote("Lịch sử phòng CSKH", "Thành lập năm 2009. " * 30),
                                                   ("Bối cảnh cạnh tranh", "Thị trường thay đổi")])
            self.assert_clean(text, 7_000, ["d1-c7-f2"])
            for label in labels:
                self.assertIn(label, text)
            self.assertIn("Mâu thuẫn 24 giờ và 48 giờ.", text)
            self.assertNotIn("48 giờ..", text)
            self.assertIn("  - Người học có thể phản hồi đúng hạn.", text)
            self.assertIn("  - Bối cảnh cạnh tranh: Thị trường thay đổi", text)
            summary = next(line for line in text.splitlines() if "Lịch sử phòng CSKH" in line)
            self.assertLessEqual(len(summary), len("  - Lịch sử phòng CSKH: ") + 160)
        blank = self.course_notes("vi", holds=[HoldNote("Khối chưa rõ", "-", "-")])
        self.assertIn("  1. Khối chưa rõ.", blank)
        self.assertNotIn("Câu hỏi cho SME: -", blank)

    # Regression (fixed): scrub_internal_ids collapses every run of spaces, so the two-space list indentation of the
    # spec §8.4 template ("  1. {hold.name} — …") is lost before sanitize_author_text can keep it.
    def test_course_notes_keep_the_template_indentation(self) -> None:
        lines = self.course_notes("vi").splitlines()
        self.assertTrue(any(line.startswith("  1. ") for line in lines))
        self.assertTrue(any(line.startswith("  - ") for line in lines))

    def test_course_notes_en_and_caps(self) -> None:
        text = self.course_notes("en", hold_count=20, ai_proposed=False, fallback_stages=[],
                                 sme_questions=[f"Question number {n} for the SME?" for n in range(14)])
        self.assert_clean(text, 7_000, ["d1-c7-f2"])
        self.assertIn("[Designed with the ID workflow — idm-1]", text)
        self.assertIn("Needs SME input (Hold) — awaiting SME confirmation (20)", text)
        self.assertIn("and 5 more", text)
        self.assertNotIn("proposed by AI", text)
        # QC run 8de1c76b: no count cap on the SME questions, only the 7,000-character limit.
        self.assertEqual(text.count("Question number"), 14)

    def test_course_notes_fit_the_limit_and_count_what_does_not_fit(self) -> None:
        holds = [("Khối " + "a" * 170, "b" * 300, "c" * 400)] * 15
        text = self.course_notes("vi", holds=holds, nice_to_know=[("Khối tham khảo", "Tóm tắt")] * 5)
        self.assertLessEqual(len(text), 7_000)
        self.assertFalse(text.endswith("…"))
        self.assertTrue(text.endswith("Cấu trúc được thiết kế theo Must Do, không theo mục lục tài liệu."))
        shown = sum(line.startswith(f"  {index}. Khối") for line in text.splitlines() for index in range(1, 16))
        self.assertGreater(shown, 0)
        self.assertIn(f"  và {15 - shown} mục khác", text)

    def test_module_and_lesson_notes(self) -> None:
        for locale in ("vi", "en"):
            module = build_module_notes(locale=locale, performance_goal="Quyết định <đúng> khi nào escalate",
                                        objectives=["Người học có thể quyết định escalate"], lesson_count=2,
                                        total_minutes=16, known_ids=[])
            self.assert_clean(module, 3_000, [])
            lesson = build_lesson_notes(locale=locale, bloom="apply", est_screens=4, est_minutes=6,
                                        ordering_rationale="Nền tảng > nâng cao, lsn_002 trước", job_aid=False)
            self.assert_clean(lesson, 2_000, [])
            job_aid = build_lesson_notes(locale=locale, bloom=None, est_screens=2, est_minutes=3,
                                         ordering_rationale="Cuối khoá", job_aid=True)
            self.assertIn("Job Aid", job_aid)
        self.assertIn("Bloom: Vận dụng", build_lesson_notes(locale="vi", bloom="apply", est_screens=4,
                                                            est_minutes=6, ordering_rationale="x", job_aid=False))
        self.assertIn("Bloom: Evaluate", build_lesson_notes(locale="en", bloom="evaluate", est_screens=4,
                                                            est_minutes=6, ordering_rationale="x", job_aid=False))
        self.assertTrue(build_module_notes(locale="en", performance_goal="Goal", objectives=[], lesson_count=1,
                                           total_minutes=5).startswith("Performance goal: Goal"))

    def test_scrub_internal_ids(self) -> None:
        text = scrub_internal_ids("Xem [cb_0001], idmcb_" + "a" * 32 + " và d1-c1-f1 , rồi lo_2 .", ["d1-c1-f1"])
        self.assertEqual(text, "Xem, và, rồi.")

    def test_assembled_notes_list_holds_and_follow_the_plan(self) -> None:
        inputs = w4_inputs()
        design = assemble_idm_course_design(assembly_input(inputs))
        known = [fact.fact_key for fact in source_facts()]
        self.assert_clean(design.notes.course, 7_000, known)
        statements = {item.must_do_id: item.statement for item in design.must_dos}
        for item in design.hold_items:
            self.assertIn(item.name, design.notes.course)
            self.assertIn(item.sme_question, design.notes.course)
            for must_do_id in item.blocked_must_do_ids:
                self.assertIn(statements[must_do_id], design.notes.course)
        names = {block.block_id: block.name for block in design.blocks}
        for row in design.blueprint:
            if row.classification == "nice_to_know":
                self.assertIn(names[row.block_id], design.notes.course)
        for text in design.notes.modules.values():
            self.assert_clean(text, 3_000, known)
        for text in design.notes.lessons.values():
            self.assert_clean(text, 2_000, known)
        self.assertIn("Job Aid", design.notes.lessons["lsn_005"])
        self.assertIn("Bloom: Vận dụng", design.notes.lessons["lsn_001"])
        self.assertNotIn("dự phòng", design.notes.course)
        fallback = assemble_idm_course_design(assembly_input(inputs, stage_origins={"w2": "deterministic_fallback"}))
        self.assertIn("dự phòng", fallback.notes.course)


class MutateHelperTests(unittest.TestCase):
    def test_mutate_never_touches_the_fixture(self) -> None:
        before = W4_COURSE["course_title"]
        changed = mutate(W4_COURSE, lambda p: {**p, "course_title": "Khác"})
        self.assertEqual((changed["course_title"], W4_COURSE["course_title"]), ("Khác", before))


if __name__ == "__main__":
    unittest.main()
