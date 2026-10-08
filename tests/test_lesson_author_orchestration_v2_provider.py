import json
import unittest
from copy import deepcopy

from pydantic import ValidationError

from app.lesson_author_orchestration_v2 import (
    OrchestrationContractError,
    assemble_blueprint_v2,
    canonical_hash,
)
from app.lesson_author_orchestration_v2_provider import (
    CHAPTER_SHARD_PROVIDER_SCHEMA_METADATA,
    COURSE_SKELETON_PROVIDER_SCHEMA_METADATA,
    ChapterShardDraftV2,
    ChapterShardProviderWireV2,
    CourseSkeletonDraftV2,
    CourseSkeletonProviderWireV2,
    SourceOutlineAuthorityV2,
    SourceOutlineChapterV2,
    SourceScopeCatalogEntryV2,
    SourceSnapshotFactV2,
    UnitGenerationContractV2,
    bind_chapter_shard_v2,
    bind_course_skeleton_v2,
    build_source_snapshot_payload_v2,
    chapter_shard_prompt_v2,
    fallback_chapter_shard_draft_v2,
    fallback_course_skeleton_draft_v2,
    parse_chapter_shard_draft_v2,
    plan_chapter_shards_v2,
    salvage_chapter_shard_draft_v2,
    skeleton_prompt_v2,
    unit_contract_manifest_v2,
    unit_contract_v5_architecture_v2,
)
from tests.staged_schema_probe import capture_sdk_body

SOURCE_HASH = "a" * 64


def lesson(scopes: list[str], title: str = "Bài học") -> dict:
    return {
        "title": title, "objective": "Hiểu nội dung", "learning_objectives": ["Áp dụng"],
        "learning_activities": ["Đọc và thực hành"], "assessment": "Bài kiểm tra",
        "units": [{"title": "Nội dung", "purpose": "Giải thích nội dung nguồn",
                   "learning_objective_refs": ["lo_1"], "source_scope_ids": scopes,
                   "component_plan": [{"type": "html", "title": "Giải thích", "rationale": "Nội dung chính",
                                       "author_review": {"purpose": "Giải thích", "example_scenario": None,
                                                         "visual_asset": None, "user_behavior_navigation": None},
                                       "source_scope_ids": scopes}], "media_brief": None}],
    }


def scope(scope_key: str, fact_count: int, content_chars: int) -> SourceScopeCatalogEntryV2:
    return SourceScopeCatalogEntryV2(
        scope_key=scope_key,
        title=f"Scope {scope_key}",
        fact_count=fact_count,
        content_chars=content_chars,
    )


def skeleton_draft() -> CourseSkeletonDraftV2:
    return CourseSkeletonDraftV2.model_validate({
        "title": "Khóa học",
        "summary": "Tóm tắt",
        "target_audience": "Quản lý",
        "prerequisites": [],
        "learning_outcomes": ["Áp dụng nội dung"],
        "assessment_strategy": "Đánh giá theo chương",
        "assumptions": [],
        "chapters": [{
            "chapter_key": "chapter-1",
            "order": 0,
            "title": "Chương 1",
            "objective": "Mục tiêu 1",
            "learning_outcomes": ["Hoàn thành mục tiêu 1"],
            "source_scope_ids": ["scope-1", "scope-2", "scope-3"],
        }],
    })


class LessonAuthorOrchestrationV2ProviderTests(unittest.TestCase):
    def test_ready_evidence_compiler_adds_only_grounded_treatments(self) -> None:
        catalog = [scope("scope-1", 3, 300)]
        skeleton = bind_course_skeleton_v2(
            skeleton_draft().model_copy(update={
                "chapters": [skeleton_draft().chapters[0].model_copy(update={
                    "source_scope_ids": ["scope-1"],
                })],
            }), source_snapshot_hash=SOURCE_HASH, locale="vi", scope_catalog=catalog,
        )
        plan = plan_chapter_shards_v2(skeleton, catalog)[0]
        facts = [SourceSnapshotFactV2(
            document_id="document-1",
            fact_key=f"fact-{index}",
            scope_key="scope-1",
            fact_text=text,
            locator={
                "instructional_density_policy_version": "unit-content-v3-density-1",
                "source_evidence_status": "ready",
                "source_evidence_revision": "b" * 64,
            },
        ) for index, text in enumerate((
            "Bước 1: Nhận diện mối nguy tại khu vực làm việc trước khi bắt đầu.",
            "Bước 2: Đánh giá khả năng xảy ra và mức hậu quả theo tiêu chí.",
            "Bước 3: Chọn biện pháp kiểm soát và xác nhận kết quả thực hiện.",
        ), start=1)]
        diagnostics: list[dict] = []
        bound = bind_chapter_shard_v2(
            ChapterShardDraftV2(lessons=[lesson(["scope-1"])]),
            skeleton=skeleton,
            plan=plan,
            source_facts=facts,
            compiler_diagnostics=diagnostics,
        )
        self.assertEqual(
            [component.type for component in bound.lessons[0].units[0].component_plan],
            ["html", "la_sortable"],
        )
        self.assertEqual(diagnostics[0]["added_types"], ["la_sortable"])

        legacy = [fact.model_copy(update={"locator": {
            "instructional_density_policy_version": "unit-content-v3-density-1",
            "source_evidence_status": "legacy_review_required",
        }}) for fact in facts]
        legacy_bound = bind_chapter_shard_v2(
            ChapterShardDraftV2(lessons=[lesson(["scope-1"])]),
            skeleton=skeleton,
            plan=plan,
            source_facts=legacy,
        )
        self.assertEqual(
            [component.type for component in legacy_bound.lessons[0].units[0].component_plan],
            ["html"],
        )

    def test_ready_evidence_compiler_is_opportunity_driven_not_quota_driven(self) -> None:
        catalog = [scope("scope-1", 3, 300)]
        skeleton = bind_course_skeleton_v2(
            skeleton_draft().model_copy(update={
                "chapters": [skeleton_draft().chapters[0].model_copy(update={
                    "source_scope_ids": ["scope-1"],
                })],
            }), source_snapshot_hash=SOURCE_HASH, locale="vi", scope_catalog=catalog,
        )
        plan = plan_chapter_shards_v2(skeleton, catalog)[0]

        def compile_types(texts: tuple[str, ...], *, table: bool = False) -> tuple[list[str], dict]:
            facts = [SourceSnapshotFactV2(
                document_id="document-1",
                fact_key=f"fact-{index}",
                scope_key="scope-1",
                fact_text=text,
                locator={
                    "instructional_density_policy_version": "unit-content-v3-density-1",
                    "source_evidence_status": "ready",
                    "source_evidence_revision": "c" * 64,
                    "content_kinds": ["table", "text"] if table else ["text"],
                    "table_count": 1 if table else 0,
                },
            ) for index, text in enumerate(texts, start=1)]
            diagnostics: list[dict] = []
            bound = bind_chapter_shard_v2(
                ChapterShardDraftV2(lessons=[lesson(["scope-1"])]),
                skeleton=skeleton,
                plan=plan,
                source_facts=facts,
                compiler_diagnostics=diagnostics,
            )
            return (
                [component.type for component in bound.lessons[0].units[0].component_plan],
                diagnostics[0],
            )

        prose_types, prose_diagnostics = compile_types((
            "Văn hóa an toàn hình thành từ cam kết nhất quán của lãnh đạo và người lao động.",
            "Hoạt động trao đổi giúp các bên hiểu trách nhiệm trong công việc hằng ngày.",
            "Việc theo dõi kết quả hỗ trợ cải tiến cách tổ chức công việc theo thời gian.",
        ))
        self.assertEqual(prose_types, ["html"])
        self.assertEqual(prose_diagnostics["opportunity_types"], [])
        self.assertEqual(prose_diagnostics["added_types"], [])

        diagram_types, diagram_diagnostics = compile_types((
            "Row 1: Cấp quản trị | Vai trò",
            "Row 2: Lãnh đạo | Phê duyệt định hướng và nguồn lực",
            "Row 3: Quản lý trực tiếp | Điều phối và xác nhận thực hiện",
        ), table=True)
        self.assertEqual(diagram_types, ["html", "la_diagram"])
        self.assertEqual(diagram_diagnostics["added_types"], ["la_diagram"])

        incomplete_faq_types, incomplete_faq_diagnostics = compile_types((
            "Lưu ý: Dừng lại.",
            "nếu thiếu dữ liệu, người thực hiện phải hỏi người phụ trách trước khi tiếp tục công việc.",
            "Nội dung nền vẫn được giữ làm phần giải thích cho người học.",
        ))
        self.assertEqual(incomplete_faq_types, ["html"])
        self.assertNotIn("la_faq", incomplete_faq_diagnostics["opportunity_types"])

        complete_faq_types, complete_faq_diagnostics = compile_types((
            "Nếu phát hiện điều kiện không an toàn, người lao động phải dừng công việc và báo người phụ trách.",
            "Khi biện pháp kiểm soát chưa có hiệu lực, công việc chỉ được tiếp tục sau khi đánh giá lại.",
            "Nội dung nền vẫn được giữ làm phần giải thích cho người học.",
        ))
        self.assertEqual(complete_faq_types, ["html", "la_faq"])
        self.assertEqual(complete_faq_diagnostics["added_types"], ["la_faq"])

    def test_partial_salvage_keeps_proven_provider_unit_and_falls_back_only_missing_scope(self) -> None:
        catalog = [scope("scope-1", 1, 80), scope("scope-2", 1, 80)]
        skeleton = bind_course_skeleton_v2(
            skeleton_draft().model_copy(update={
                "chapters": [skeleton_draft().chapters[0].model_copy(update={
                    "source_scope_ids": ["scope-1", "scope-2"],
                })],
            }), source_snapshot_hash=SOURCE_HASH, locale="vi", scope_catalog=catalog,
        )
        plan = plan_chapter_shards_v2(skeleton, catalog)[0]
        raw_lesson = lesson(["scope-1"], "Bài provider đạt")
        invalid = deepcopy(raw_lesson["units"][0])
        invalid["source_scope_ids"] = ["scope-2"]
        invalid["component_plan"][0]["source_scope_ids"] = ["scope-2"]
        invalid.pop("purpose")
        raw_lesson["units"].append(invalid)
        facts = [SourceSnapshotFactV2(
            document_id="document-1", fact_key=f"fact-{index}", scope_key=scope_id,
            fact_text=f"Nội dung nguồn đủ dài và có ý nghĩa cho {scope_id}.", locator={},
        ) for index, scope_id in enumerate(("scope-1", "scope-2"), start=1)]

        result = salvage_chapter_shard_draft_v2(
            json.dumps({"lessons": [raw_lesson]}, ensure_ascii=False),
            skeleton=skeleton,
            plan=plan,
            source_facts=facts,
        )
        self.assertIsNotNone(result)
        shard, diagnostics = result  # type: ignore[misc]
        self.assertEqual(diagnostics["accepted_provider_unit_count"], 1)
        self.assertEqual(diagnostics["fallback_scope_count"], 1)
        self.assertEqual(shard.lessons[0].title, "Bài provider đạt")
        self.assertEqual(
            [scope_id for item in shard.lessons for unit in item.units for scope_id in unit.source_scope_ids],
            ["scope-1", "scope-2"],
        )

    def test_unit_architecture_allows_interaction_led_plan_and_enforces_display_order(self) -> None:
        payload = lesson(["scope-1"])
        unit = payload["units"][0]
        base = unit["component_plan"][0]
        unit["component_plan"] = [
            {**deepcopy(base), "type": "la_diagram", "title": "Quan hệ"},
            {**deepcopy(base), "type": "problem", "title": "Kiểm tra"},
            {**deepcopy(base), "type": "la_faq", "title": "Câu hỏi thường gặp"},
        ]
        parsed = ChapterShardDraftV2.model_validate({"lessons": [payload]})
        self.assertEqual(
            [component.type for component in parsed.lessons[0].units[0].component_plan],
            ["la_diagram", "problem", "la_faq"],
        )

        html_late = deepcopy(payload)
        html_late["units"][0]["component_plan"] = [
            {**deepcopy(base), "type": "la_diagram"},
            {**deepcopy(base), "type": "html"},
        ]
        with self.assertRaises(ValidationError):
            ChapterShardDraftV2.model_validate({"lessons": [html_late]})

        faq_early = deepcopy(payload)
        faq_early["units"][0]["component_plan"] = [
            {**deepcopy(base), "type": "la_faq"},
            {**deepcopy(base), "type": "problem"},
        ]
        with self.assertRaises(ValidationError):
            ChapterShardDraftV2.model_validate({"lessons": [faq_early]})

    def test_chapter_wire_normalizes_nullable_review_fields_and_component_aliases(self) -> None:
        payload = lesson(["scope-1"])
        unit = payload["units"][0]
        del unit["media_brief"]
        component = deepcopy(unit["component_plan"][0])
        component["type"] = "diagram"
        component["title"] = "Sơ đồ"
        component.pop("author_review")
        unit["component_plan"].append(component)
        parsed = parse_chapter_shard_draft_v2(json.dumps({"lessons": [payload]}))
        self.assertIsNone(parsed.lessons[0].units[0].media_brief)
        self.assertEqual(parsed.lessons[0].units[0].component_plan[1].type, "la_diagram")
        self.assertIsNone(parsed.lessons[0].units[0].component_plan[1].author_review.visual_asset)

    def test_deterministic_chapter_fallback_preserves_every_scope_once(self) -> None:
        catalog = [scope(f"scope-{index}", 1, 10) for index in range(1, 10)]
        skeleton = bind_course_skeleton_v2(
            skeleton_draft().model_copy(update={
                "chapters": [skeleton_draft().chapters[0].model_copy(update={
                    "source_scope_ids": [item.scope_key for item in catalog],
                })],
            }),
            source_snapshot_hash=SOURCE_HASH, locale="vi", scope_catalog=catalog,
        )
        plan = plan_chapter_shards_v2(skeleton, catalog, max_source_chars=1000)[0]
        facts = [SourceSnapshotFactV2(
            document_id="document-1", fact_key=f"fact-{index}", scope_key=item.scope_key,
            fact_text=f"Nội dung phần {index}",
        ) for index, item in enumerate(catalog, start=1)]
        fallback = fallback_chapter_shard_draft_v2(skeleton, plan, facts)
        allocated = [scope_id for lesson_item in fallback.lessons for unit in lesson_item.units
                     for scope_id in unit.source_scope_ids]
        self.assertEqual(allocated, plan.source_scope_ids)
        self.assertEqual(len(fallback.lessons), 2)
        self.assertNotEqual(fallback.lessons[0].objective, skeleton.chapters[0].objective)
        self.assertIn(skeleton.chapters[0].title, fallback.lessons[0].objective)
        self.assertIn(fallback.lessons[0].learning_objectives[0], fallback.lessons[0].units[0].purpose)

    def test_deterministic_chapter_fallback_bounds_long_lesson_objective(self) -> None:
        catalog = [SourceScopeCatalogEntryV2(
            scope_key=f"scope-{index}", title=(f"Phần {index} " + "Nội dung dài " * 20),
            fact_count=1, content_chars=100,
        ) for index in range(1, 7)]
        skeleton = bind_course_skeleton_v2(
            skeleton_draft().model_copy(update={
                "chapters": [skeleton_draft().chapters[0].model_copy(update={
                    "source_scope_ids": [item.scope_key for item in catalog],
                })],
            }), source_snapshot_hash=SOURCE_HASH, locale="vi", scope_catalog=catalog,
        )
        plan = plan_chapter_shards_v2(skeleton, catalog, max_source_chars=10_000)[0]
        facts = [SourceSnapshotFactV2(
            document_id="document-1", fact_key=f"fact-{index}", scope_key=item.scope_key,
            fact_text=f"Nội dung nguồn {index}",
        ) for index, item in enumerate(catalog, start=1)]
        fallback = fallback_chapter_shard_draft_v2(skeleton, plan, facts)
        self.assertTrue(fallback.lessons)
        self.assertTrue(all(len(item.objective) <= 500 for item in fallback.lessons))

    def test_provider_wire_schemas_serialize_through_pinned_google_sdk(self) -> None:
        for server_model, wire_model, metadata in (
            (CourseSkeletonDraftV2, CourseSkeletonProviderWireV2,
             COURSE_SKELETON_PROVIDER_SCHEMA_METADATA),
            (ChapterShardDraftV2, ChapterShardProviderWireV2,
             CHAPTER_SHARD_PROVIDER_SCHEMA_METADATA),
        ):
            with self.subTest(model=server_model.__name__):
                original = deepcopy(server_model.model_json_schema())
                body = capture_sdk_body(wire_model)
                sdk_schema = body["generationConfig"]["responseSchema"]
                self.assertEqual(set(sdk_schema["properties"]), set(original["properties"]))
                self.assertEqual(sdk_schema["required"], original["required"])
                if server_model is ChapterShardDraftV2:
                    media = sdk_schema["properties"]["lessons"]["items"]["properties"]["units"]["items"]["properties"]["media_brief"]
                    self.assertTrue(media["nullable"])
                    self.assertEqual(set(media["properties"]), {
                        "type", "title", "content_points", "context_description", "rationale",
                    })
                    self.assertEqual(set(media["required"]), set(media["properties"]))
                self.assertEqual(server_model.model_json_schema(), original)
                self.assertTrue(metadata["server_validation_unchanged"])
                self.assertGreater(
                    metadata["projected_unsupported_keyword_counts"].get("additionalProperties", 0),
                    0,
                )
                self.assertGreater(
                    sum(metadata["projected_unsupported_keyword_counts"].get(key, 0) for key in (
                        "minItems", "maxItems", "minLength", "maxLength", "minimum", "maximum",
                    )),
                    0,
                )

    def test_provider_projection_preserves_contract_shape_around_sdk_compatibility_changes(self) -> None:
        def visit(value):
            if isinstance(value, dict):
                yield value
                for child in value.values():
                    yield from visit(child)
            elif isinstance(value, list):
                for child in value:
                    yield from visit(child)

        for server_model, wire_model in (
            (CourseSkeletonDraftV2, CourseSkeletonProviderWireV2),
            (ChapterShardDraftV2, ChapterShardProviderWireV2),
        ):
            with self.subTest(model=server_model.__name__):
                server_schema = server_model.model_json_schema()
                wire_schema = wire_model.model_json_schema()
                self.assertEqual(set(wire_schema["properties"]), set(server_schema["properties"]))
                self.assertEqual(wire_schema["required"], server_schema["required"])
                self.assertEqual(wire_schema.get("title"), server_schema.get("title"))
                self.assertTrue(any("additionalProperties" in node for node in visit(server_schema)))
                self.assertTrue(all("additionalProperties" not in node for node in visit(wire_schema)))
                self.assertTrue(any(any(key in node for key in (
                    "minItems", "maxItems", "minLength", "maxLength", "minimum", "maximum",
                )) for node in visit(server_schema)))
                self.assertTrue(all(all(key not in node for key in (
                    "minItems", "maxItems", "minLength", "maxLength", "minimum", "maximum",
                )) for node in visit(wire_schema)))
                self.assertTrue(all(not (
                    isinstance(node.get("anyOf"), list)
                    and any(isinstance(item, dict) and "$ref" in item for item in node["anyOf"])
                    and any(isinstance(item, dict) and item.get("type") == "null" for item in node["anyOf"])
                ) for node in visit(wire_schema)))

    def test_provider_projection_keeps_strict_local_extra_field_rejection(self) -> None:
        skeleton = skeleton_draft().model_dump()
        skeleton["unexpected"] = "rejected"
        chapter = {"lessons": [lesson(["scope-1"])]}
        chapter["unexpected"] = "rejected"
        for model, payload in (
            (CourseSkeletonDraftV2, skeleton),
            (CourseSkeletonProviderWireV2, skeleton),
            (ChapterShardDraftV2, chapter),
            (ChapterShardProviderWireV2, chapter),
        ):
            with self.subTest(model=model.__name__), self.assertRaises(ValidationError):
                model.model_validate(payload)

    def test_provider_projection_keeps_strict_local_value_bounds(self) -> None:
        skeleton = skeleton_draft().model_dump()
        skeleton["title"] = ""
        for model in (CourseSkeletonDraftV2, CourseSkeletonProviderWireV2):
            with self.subTest(model=model.__name__), self.assertRaises(ValidationError):
                model.model_validate(skeleton)

        chapter = {"lessons": [lesson(["scope-1"])]}
        chapter["lessons"][0]["learning_objectives"] = []
        for model in (ChapterShardDraftV2, ChapterShardProviderWireV2):
            with self.subTest(model=model.__name__), self.assertRaises(ValidationError):
                model.model_validate(chapter)

    def test_unit_contract_adapts_exact_facts_and_stable_component_identity(self) -> None:
        base = {
            "contract_version": 2, "source_snapshot_hash": SOURCE_HASH, "assembly_hash": "b" * 64,
            "chapter_key": "chapter-1", "unit_path": "chapter_1.lesson_1.unit_1",
            "chapter_title": "Chương 1", "lesson_title": "Bài 1",
            "lesson_learning_objectives": ["Áp dụng"], "unit_title": "Nội dung",
            "unit_purpose": "Giải thích", "unit_learning_objective_refs": ["lo_1"],
            "unit_source_scope_ids": ["scope-1"], "unit_source_fact_ids": ["fact-1"],
            "component_plan": [{"component_plan_id": "cp2_" + "c" * 32, "type": "html",
                "title": "Giải thích", "rationale": "Nội dung chính", "purpose": "explain",
                "source_fact_ids": ["fact-1"], "supporting_evidence_fact_ids": [],
                "learning_objective_refs": ["lo_1"], "source_scope_ids": ["scope-1"],
                "content_requirements": [], "learning_block_ids": [], "required_artifacts": []}],
            "source_facts": [{"document_id": "document-1", "fact_key": "fact-1",
                "scope_key": "scope-1", "fact_text": "Alpha", "source_ref": None,
                "source_page": 1, "source_chunk": 0, "locator": {}}],
        }
        contract = UnitGenerationContractV2.model_validate({**base, "contract_hash": canonical_hash(base)})
        manifest = unit_contract_manifest_v2(contract)
        architecture = unit_contract_v5_architecture_v2(contract)
        self.assertEqual(manifest["facts"][0]["fact_id"], "fact-1")
        plan = architecture["lessons"][0]["units"][0]["component_plan"][0]
        self.assertEqual(plan["component_plan_id"], "cp2_" + "c" * 32)
        self.assertEqual(plan["source_fact_ids"], ["fact-1"])

        v3_base = deepcopy(base)
        v3_base["source_facts"][0]["locator"] = {
            "instructional_density_policy_version": "unit-content-v3-density-1",
        }
        v3_base["component_plan"].append({
            **deepcopy(v3_base["component_plan"][0]),
            "component_plan_id": "cp2_" + "e" * 32,
            "type": "problem",
            "title": "Kiểm tra",
            "purpose": "assess",
            "source_fact_ids": [],
            "supporting_evidence_fact_ids": ["fact-1"],
        })
        v3_contract = UnitGenerationContractV2.model_validate({
            **v3_base,
            "contract_hash": canonical_hash(v3_base),
        })
        self.assertEqual(v3_contract.component_plan[0].source_fact_ids, ["fact-1"])
        self.assertEqual(v3_contract.component_plan[1].source_fact_ids, [])
        self.assertEqual(v3_contract.component_plan[1].supporting_evidence_fact_ids, ["fact-1"])
        v3_unit = unit_contract_v5_architecture_v2(v3_contract)["lessons"][0]["units"][0]
        self.assertEqual(v3_unit["supporting_evidence_fact_ids"], ["fact-1"])
        self.assertEqual(v3_unit["instructional_density_policy_version"], "unit-content-v3-density-1")
        self.assertEqual(v3_unit["max_generated_visible_chars"], 4_000)
        self.assertEqual(v3_unit["max_generated_words"], 600)
        invalid_v3 = deepcopy(v3_base)
        invalid_v3["component_plan"][1]["source_fact_ids"] = ["fact-1"]
        invalid_v3["component_plan"][1]["supporting_evidence_fact_ids"] = []
        with self.assertRaisesRegex(ValueError, "ORCHESTRATION_V2_UNIT_PLAN_INVALID"):
            UnitGenerationContractV2.model_validate({
                **invalid_v3,
                "contract_hash": canonical_hash(invalid_v3),
            })

        with self.assertRaisesRegex(ValueError, "ORCHESTRATION_V2_UNIT_CONTRACT_HASH_INVALID"):
            UnitGenerationContractV2.model_validate({**base, "contract_hash": "d" * 64})
        wrong_type = {**base, "component_plan": [{**base["component_plan"][0], "type": "unknown"}]}
        with self.assertRaises(ValueError):
            UnitGenerationContractV2.model_validate({**wrong_type, "contract_hash": canonical_hash(wrong_type)})

    def test_actual_stage_two_contract_receives_structured_evidence_without_new_ownership(self) -> None:
        from app.services.lesson_author.staged.source_locked import build_staged_instructional_contract

        source_revision = "e" * 64
        asset_revision = "f" * 64
        base = {
            "contract_version": 2, "source_snapshot_hash": SOURCE_HASH, "assembly_hash": "b" * 64,
            "chapter_key": "chapter-1", "unit_path": "chapter_1.lesson_1.unit_1",
            "chapter_title": "Chương 1", "lesson_title": "Bài 1",
            "lesson_learning_objectives": ["Áp dụng"], "unit_title": "Bảng kiểm soát",
            "unit_purpose": "Giải thích quan hệ trong bảng", "unit_learning_objective_refs": ["lo_1"],
            "unit_source_scope_ids": ["scope-1"],
            "unit_source_fact_ids": ["fact-1", "fact-2"],
            "component_plan": [{"component_plan_id": "cp2_" + "c" * 32, "type": "html",
                "title": "Giải thích", "rationale": "Nội dung chính", "purpose": "explain",
                "source_fact_ids": ["fact-1", "fact-2"], "supporting_evidence_fact_ids": [],
                "learning_objective_refs": ["lo_1"], "source_scope_ids": ["scope-1"],
                "content_requirements": [], "learning_block_ids": [], "required_artifacts": []}],
            "source_facts": [
                {"document_id": "document-1", "fact_key": "fact-1", "scope_key": "scope-1",
                 "fact_text": "Row 1: Mối nguy | Biện pháp", "source_ref": "source-1",
                 "source_page": 1, "source_chunk": 0, "locator": {
                     "source_revision": source_revision,
                     "visual_prompt_text": "Biện pháp nào có ưu tiên cao hơn?",
                     "visual_regions": [{"region_kind": "embedded_image",
                         "asset_revision": asset_revision,
                         "locator": {"page": 1, "bbox_normalized": [0.1, 0.2, 0.8, 0.9]},
                         "observation": {"status": "unreviewed", "facts": []},
                         "inference": {"status": "not_performed", "claims": []}}],
                 }},
                {"document_id": "document-1", "fact_key": "fact-2", "scope_key": "scope-1",
                 "fact_text": "Row 2: Hóa chất | Thay thế", "source_ref": "source-1",
                 "source_page": 1, "source_chunk": 0,
                 "locator": {"source_revision": source_revision}},
            ],
        }
        contract = UnitGenerationContractV2.model_validate({
            **base,
            "contract_hash": canonical_hash(base),
        })
        manifest = unit_contract_manifest_v2(contract, locale="vi")
        expected = {
            "source_fact_ids": ["fact-1", "fact-2"],
            "supporting_evidence_fact_ids": [],
            "learning_objective_refs": ["lo_1"],
            "strict_v5_evidence": True,
        }

        writer_contract = build_staged_instructional_contract(expected, manifest)
        bundle = writer_contract["source_evidence_bundle"]
        self.assertIsNotNone(bundle)
        self.assertEqual(bundle["status"], "review_required")
        table = next(item for item in bundle["elements"] if item["kind"] == "table")
        self.assertEqual(table["payload"]["rows"], [
            ["Mối nguy", "Biện pháp"],
            ["Hóa chất", "Thay thế"],
        ])
        visual = next(item for item in bundle["elements"] if item["kind"] == "visual")
        self.assertEqual(visual["payload"]["prompt_text"], "Biện pháp nào có ưu tiên cao hơn?")
        self.assertEqual(manifest["represented_fact_count"], 2)
        self.assertEqual(
            [item["fact_id"] for item in manifest["facts"]],
            ["fact-1", "fact-2"],
        )

    def test_builds_exact_source_snapshot(self) -> None:
        manifest = {
            "parser_version": "test-v1",
            "facts": [
                {"fact_id": "fact-1", "document_id": "document-1", "text": "Alpha", "source_page": 1},
                {"fact_id": "fact-2", "document_id": "document-1", "text": "Beta", "source_page": 2},
            ],
        }
        source_map = {"source_evidence_scopes": [
            {"id": "scope-1", "source_ref": "Trang 1", "source_fact_ids": ["fact-1"]},
            {"id": "scope-2", "source_ref": "Trang 2", "source_fact_ids": ["fact-2"]},
        ]}
        payload = build_source_snapshot_payload_v2(source_map, manifest)
        self.assertEqual([fact.scope_key for fact in payload.facts], ["scope-1", "scope-2"])
        self.assertEqual([item.content_chars for item in payload.scopes], [5, 4])
        self.assertEqual(payload.facts[0].locator, {"parser_version": "test-v1"})

    def test_rejects_duplicate_or_missing_fact_ownership(self) -> None:
        manifest = {"facts": [
            {"fact_id": "fact-1", "document_id": "document-1", "text": "Alpha"},
            {"fact_id": "fact-2", "document_id": "document-1", "text": "Beta"},
        ]}
        with self.assertRaisesRegex(OrchestrationContractError, "SOURCE_SNAPSHOT_FACT_OWNERSHIP_INVALID"):
            build_source_snapshot_payload_v2({"source_evidence_scopes": [
                {"id": "scope-1", "source_fact_ids": ["fact-1"]},
                {"id": "scope-2", "source_fact_ids": ["fact-1"]},
            ]}, manifest)

    def test_binds_skeleton_only_with_exact_scope_allocation(self) -> None:
        catalog = [scope("scope-1", 1, 10), scope("scope-2", 1, 10), scope("scope-3", 1, 10)]
        bound = bind_course_skeleton_v2(
            skeleton_draft(), source_snapshot_hash=SOURCE_HASH, locale="vi", scope_catalog=catalog,
        )
        self.assertEqual(bound.source_snapshot_hash, SOURCE_HASH)
        bad = skeleton_draft().model_copy(deep=True)
        bad.chapters[0].source_scope_ids = ["scope-1", "scope-2"]
        with self.assertRaisesRegex(OrchestrationContractError, "ARCHITECTURE_SCOPE_ALLOCATION_INCOMPLETE"):
            bind_course_skeleton_v2(bad, source_snapshot_hash=SOURCE_HASH, locale="vi", scope_catalog=catalog)

    def test_skeleton_binding_replaces_provider_identity_with_server_owned_sequence(self) -> None:
        catalog = [scope("scope-1", 1, 10), scope("scope-2", 1, 10), scope("scope-3", 1, 10)]
        draft = skeleton_draft().model_copy(deep=True)
        second = draft.chapters[0].model_copy(deep=True)
        draft.chapters[0].source_scope_ids = ["scope-1"]
        second.source_scope_ids = ["scope-2", "scope-3"]
        draft.chapters.append(second)
        draft.chapters[0].chapter_key = "provider-first"
        draft.chapters[0].order = 9
        draft.chapters[1].chapter_key = "provider-second"
        draft.chapters[1].order = 4
        bound = bind_course_skeleton_v2(
            draft, source_snapshot_hash=SOURCE_HASH, locale="vi", scope_catalog=catalog,
        )
        self.assertEqual([chapter.chapter_key for chapter in bound.chapters], ["chapter-1", "chapter-2"])
        self.assertEqual([chapter.order for chapter in bound.chapters], [0, 1])

    def test_skeleton_fallback_bounds_chapters_and_preserves_every_scope_once(self) -> None:
        catalog = [scope(f"scope-{index + 1}", 1, 10) for index in range(513)]
        authority_payload = {
            "mode": "model_designed", "source": "none", "complete": True,
            "confidence": 0.0, "reason_codes": [], "chapters": [],
        }
        authority = SourceOutlineAuthorityV2(
            **authority_payload, structure_hash=canonical_hash(authority_payload),
        )
        draft = fallback_course_skeleton_draft_v2("vi", catalog, authority)
        self.assertLessEqual(len(draft.chapters), 512)
        self.assertEqual(
            [scope_id for chapter in draft.chapters for scope_id in chapter.source_scope_ids],
            [item.scope_key for item in catalog],
        )
        bound = bind_course_skeleton_v2(
            draft, source_snapshot_hash=SOURCE_HASH, locale="vi",
            scope_catalog=catalog, source_authority=authority,
        )
        self.assertEqual(len(bound.chapters), len(draft.chapters))

    def test_locked_toc_authority_preserves_all_source_chapters_and_scope_ownership(self) -> None:
        titles = [
            "WHY CHANGE", "THE BIC LEGACY", "THE BEST-IN-CLASS HOUSE",
            "THE VIETNAM DREAM", "BIC 5 MINDSET SHIFTS", "ERA5.0", "ACTION ROADMAP",
        ]
        catalog = [SourceScopeCatalogEntryV2(
            scope_key=f"scope-{index + 1}", title=title, source_ref=f"src-{index + 1:03d}",
            fact_count=1, content_chars=100,
        ) for index, title in enumerate(titles)]
        draft = skeleton_draft().model_copy(deep=True)
        template = draft.chapters[0]
        draft.chapters = [template.model_copy(update={
            "chapter_key": f"provider-{index}", "order": 99 - index,
            "title": f"Provider merged title {index}", "source_scope_ids": ["scope-1"],
        }) for index in range(len(titles))]
        authority_payload = {
            "mode": "locked", "source": "toc", "complete": True, "confidence": 0.95,
            "reason_codes": ["source_toc_complete"],
            "chapters": [SourceOutlineChapterV2(
                order=index, document_id="document-1", source_ref=f"src-{index + 1:03d}", title=title,
            ) for index, title in enumerate(titles)],
        }
        authority = SourceOutlineAuthorityV2(
            **authority_payload, structure_hash=canonical_hash({
                **authority_payload,
                "chapters": [chapter.model_dump(mode="json") for chapter in authority_payload["chapters"]],
            }),
        )
        bound = bind_course_skeleton_v2(
            draft, source_snapshot_hash=SOURCE_HASH, locale="vi",
            scope_catalog=catalog, source_authority=authority,
        )
        self.assertEqual([chapter.title for chapter in bound.chapters], titles)
        self.assertEqual([chapter.order for chapter in bound.chapters], list(range(7)))
        self.assertEqual([chapter.source_scope_ids for chapter in bound.chapters],
                         [[f"scope-{index + 1}"] for index in range(7)])
        shortened = draft.model_copy(update={"chapters": draft.chapters[:5]})
        with self.assertRaisesRegex(OrchestrationContractError, "ARCHITECTURE_SOURCE_CHAPTER_COUNT_MISMATCH"):
            bind_course_skeleton_v2(
                shortened, source_snapshot_hash=SOURCE_HASH, locale="vi",
                scope_catalog=catalog, source_authority=authority,
            )

    def test_plans_bounded_shards_and_assembles_them_deterministically(self) -> None:
        catalog = [scope("scope-1", 1, 60), scope("scope-2", 1, 60), scope("scope-3", 1, 40)]
        skeleton = bind_course_skeleton_v2(
            skeleton_draft(), source_snapshot_hash=SOURCE_HASH, locale="vi", scope_catalog=catalog,
        )
        plans = plan_chapter_shards_v2(skeleton, catalog, max_source_chars=100)
        self.assertEqual([plan.source_scope_ids for plan in plans], [["scope-1"], ["scope-2", "scope-3"]])
        shards = [bind_chapter_shard_v2(
            ChapterShardDraftV2(lessons=[lesson(plan.source_scope_ids, f"Bài {plan.shard_index + 1}")]),
            skeleton=skeleton,
            plan=plan,
        ) for plan in plans]
        assembled = assemble_blueprint_v2(
            skeleton,
            list(reversed(shards)),
            {"scope-1": ["fact-1"], "scope-2": ["fact-2"], "scope-3": ["fact-3"]},
        )
        self.assertEqual(
            [lesson["title"] for lesson in assembled.blueprint["chapters"][0]["lessons"]],
            ["Bài 1", "Bài 2"],
        )

    def test_rejects_incomplete_or_out_of_unit_scope_allocation(self) -> None:
        catalog = [scope("scope-1", 1, 10), scope("scope-2", 1, 10), scope("scope-3", 1, 10)]
        skeleton = bind_course_skeleton_v2(
            skeleton_draft(), source_snapshot_hash=SOURCE_HASH, locale="vi", scope_catalog=catalog,
        )
        plan = plan_chapter_shards_v2(skeleton, catalog, max_source_chars=30)[0]
        with self.assertRaisesRegex(OrchestrationContractError, "ARCHITECTURE_SHARD_UNIT_ALLOCATION_INCOMPLETE"):
            bind_chapter_shard_v2(ChapterShardDraftV2(lessons=[lesson(["scope-2"])]), skeleton=skeleton, plan=plan)
        invalid = lesson(plan.source_scope_ids)
        invalid["units"][0]["component_plan"][0]["source_scope_ids"] = ["scope-outside"]
        with self.assertRaisesRegex(ValueError, "component scope exceeds"):
            ChapterShardDraftV2(lessons=[invalid])

    def test_v3_density_rejects_oversized_unit_then_accepts_deterministic_split(self) -> None:
        catalog = [scope("scope3_one", 25, 2_500), scope("scope3_two", 25, 2_500)]
        draft = skeleton_draft().model_copy(update={
            "chapters": [skeleton_draft().chapters[0].model_copy(update={
                "source_scope_ids": [item.scope_key for item in catalog],
            })],
        })
        skeleton = bind_course_skeleton_v2(
            draft,
            source_snapshot_hash=SOURCE_HASH,
            locale="vi",
            scope_catalog=catalog,
        )
        plan = plan_chapter_shards_v2(skeleton, catalog)[0]
        facts = [
            SourceSnapshotFactV2(
                document_id="document-1",
                fact_key=f"{scope_id}-fact-{index}",
                scope_key=scope_id,
                fact_text=f"Canonical source statement {scope_id} {index}.",
                locator={"instructional_density_policy_version": "unit-content-v3-density-1"},
            )
            for scope_id in ("scope3_one", "scope3_two")
            for index in range(25)
        ]
        oversized = ChapterShardDraftV2(lessons=[lesson(plan.source_scope_ids)])
        with self.assertRaisesRegex(OrchestrationContractError, "ARCHITECTURE_UNIT_DENSITY_EXCEEDED"):
            bind_chapter_shard_v2(
                oversized,
                skeleton=skeleton,
                plan=plan,
                source_facts=facts,
            )

        fallback = fallback_chapter_shard_draft_v2(skeleton, plan, facts)
        bound = bind_chapter_shard_v2(
            fallback,
            skeleton=skeleton,
            plan=plan,
            source_facts=facts,
        )
        self.assertEqual(sum(len(item.units) for item in bound.lessons), 2)

        legacy_facts = [fact.model_copy(update={"locator": {}}) for fact in facts]
        legacy = bind_chapter_shard_v2(
            oversized,
            skeleton=skeleton,
            plan=plan,
            source_facts=legacy_facts,
        )
        self.assertEqual(len(legacy.lessons[0].units), 1)

    def test_v3_admits_optional_interactions_only_with_matching_source_evidence(self) -> None:
        def bound_component_types(component_type: str, texts: list[str]) -> list[str]:
            catalog = [scope("scope-1", len(texts), sum(len(text) for text in texts))]
            draft = skeleton_draft().model_copy(update={
                "chapters": [skeleton_draft().chapters[0].model_copy(update={
                    "source_scope_ids": ["scope-1"],
                })],
            })
            skeleton = bind_course_skeleton_v2(
                draft,
                source_snapshot_hash=SOURCE_HASH,
                locale="vi",
                scope_catalog=catalog,
            )
            plan = plan_chapter_shards_v2(skeleton, catalog)[0]
            lesson_payload = lesson(["scope-1"])
            optional = deepcopy(lesson_payload["units"][0]["component_plan"][0])
            optional.update(type=component_type, title="Tương tác", rationale="Thực hành theo nguồn")
            lesson_payload["units"][0]["component_plan"].append(optional)
            facts = [SourceSnapshotFactV2(
                document_id="document-1",
                fact_key=f"fact-{index}",
                scope_key="scope-1",
                fact_text=text,
                locator={"instructional_density_policy_version": "unit-content-v3-density-1"},
            ) for index, text in enumerate(texts, start=1)]
            bound = bind_chapter_shard_v2(
                ChapterShardDraftV2(lessons=[lesson_payload]),
                skeleton=skeleton,
                plan=plan,
                source_facts=facts,
            )
            return [component.type for component in bound.lessons[0].units[0].component_plan]

        prose = [
            "Người học nhận diện mối nguy trước khi bắt đầu công việc tại khu vực được giao.",
            "Biện pháp kiểm soát phải phù hợp với mối nguy và điều kiện làm việc thực tế.",
            "Kết quả kiểm tra phải được xác nhận theo yêu cầu của quy trình đã phê duyệt.",
        ]
        steps = [
            "Bước 1: Nhận diện mối nguy tại khu vực làm việc trước khi bắt đầu.",
            "Bước 2: Đánh giá khả năng xảy ra và mức hậu quả theo tiêu chí.",
            "Bước 3: Chọn biện pháp kiểm soát và xác nhận kết quả thực hiện.",
        ]
        definitions = [
            "Mối nguy cơ khí: Nguồn chuyển động có thể gây va đập, cuốn hoặc kẹp người lao động.",
            "Mối nguy điện: Nguồn điện không được kiểm soát có thể gây điện giật hoặc hồ quang.",
            "Mối nguy hóa chất: Phơi nhiễm cần được kiểm soát theo đặc tính của hóa chất.",
        ]

        self.assertEqual(bound_component_types("la_sortable", prose), ["html"])
        self.assertEqual(bound_component_types("la_sortable", steps), ["html", "la_sortable"])
        self.assertEqual(bound_component_types("la_crossword", steps), ["html"])
        self.assertEqual(bound_component_types("la_crossword", definitions), ["html", "la_crossword"])

    def test_missing_source_authored_quiz_becomes_traceable_obligation_not_fake_problem(self) -> None:
        catalog = [scope("scope-1", 1, 80)]
        draft = skeleton_draft().model_copy(update={
            "chapters": [skeleton_draft().chapters[0].model_copy(update={
                "source_scope_ids": ["scope-1"],
            })],
        })
        skeleton = bind_course_skeleton_v2(draft, source_snapshot_hash=SOURCE_HASH,
                                           locale="vi", scope_catalog=catalog)
        plan = plan_chapter_shards_v2(skeleton, catalog)[0]
        lesson_payload = lesson(["scope-1"])
        problem = deepcopy(lesson_payload["units"][0]["component_plan"][0])
        problem.update(type="problem", title="Kiểm tra", rationale="Kiểm tra mục tiêu")
        lesson_payload["units"][0]["component_plan"].append(problem)
        facts = [SourceSnapshotFactV2(
            document_id="document-1", fact_key="fact-1", scope_key="scope-1",
            fact_text="Người thực hiện phải kiểm tra điều kiện an toàn trước khi bắt đầu công việc.",
            locator={"instructional_density_policy_version": "unit-content-v3-density-1"},
        )]

        bound = bind_chapter_shard_v2(ChapterShardDraftV2(lessons=[lesson_payload]),
                                      skeleton=skeleton, plan=plan, source_facts=facts)

        self.assertEqual([item.type for item in bound.lessons[0].units[0].component_plan], ["html"])
        self.assertEqual(len(bound.assessment_obligations), 1)
        obligation = bound.assessment_obligations[0]
        self.assertEqual(obligation.required_assessment_kind, "single_choice")
        self.assertEqual(obligation.learning_objective_refs, ["lo_1"])
        self.assertEqual(obligation.relevant_evidence_fact_ids, ["fact-1"])
        self.assertEqual(obligation.status, "open")

    def test_assessment_obligation_is_bounded_to_persistable_slots(self) -> None:
        catalog = [scope("scope-1", 1, 80)]
        draft = skeleton_draft().model_copy(update={
            "chapters": [skeleton_draft().chapters[0].model_copy(update={
                "source_scope_ids": ["scope-1"],
            })],
        })
        skeleton = bind_course_skeleton_v2(
            draft, source_snapshot_hash=SOURCE_HASH, locale="vi", scope_catalog=catalog,
        )
        plan = plan_chapter_shards_v2(skeleton, catalog)[0]
        facts = [SourceSnapshotFactV2(
            document_id="document-1", fact_key="fact-1", scope_key="scope-1",
            fact_text="Người thực hiện phải kiểm tra điều kiện an toàn trước khi bắt đầu công việc.",
            locator={"instructional_density_policy_version": "unit-content-v3-density-1"},
        )]

        def bind_with_problem_at(slot: int) -> tuple[object, list[dict]]:
            payload = lesson(["scope-1"])
            base = payload["units"][0]["component_plan"][0]
            component_types = ["html", "la_diagram", "la_sortable", "problem"][:slot]
            payload["units"][0]["component_plan"] = [
                {**deepcopy(base), "type": component_type, "title": component_type}
                for component_type in component_types[:-1]
            ] + [{**deepcopy(base), "type": "problem", "title": "Kiểm tra"}]
            diagnostics: list[dict] = []
            bound = bind_chapter_shard_v2(
                ChapterShardDraftV2(lessons=[payload]), skeleton=skeleton, plan=plan,
                source_facts=facts, compiler_diagnostics=diagnostics,
            )
            return bound, diagnostics

        slot_three, _ = bind_with_problem_at(3)
        self.assertEqual(slot_three.assessment_obligations[0].component_index, 3)
        slot_four, diagnostics = bind_with_problem_at(4)
        self.assertEqual(slot_four.assessment_obligations, [])
        self.assertEqual(diagnostics[0]["obligation_omitted_component_indices"], [4])

    def test_rejects_indivisible_scope_above_shard_capacity(self) -> None:
        catalog = [scope("scope-1", 1, 101), scope("scope-2", 1, 1), scope("scope-3", 1, 1)]
        skeleton = bind_course_skeleton_v2(
            skeleton_draft(), source_snapshot_hash=SOURCE_HASH, locale="vi", scope_catalog=catalog,
        )
        with self.assertRaisesRegex(OrchestrationContractError, "ARCHITECTURE_SCOPE_EXCEEDS_SHARD_CAPACITY"):
            plan_chapter_shards_v2(skeleton, catalog, max_source_chars=100)

    def test_prompts_are_scope_bound_and_exclude_fact_ids_from_skeleton(self) -> None:
        catalog = [scope("scope-1", 1, 10), scope("scope-2", 1, 10), scope("scope-3", 1, 10)]
        skeleton = bind_course_skeleton_v2(
            skeleton_draft(), source_snapshot_hash=SOURCE_HASH, locale="vi", scope_catalog=catalog,
        )
        prompt = skeleton_prompt_v2("vi", catalog)
        self.assertIn("scope-1", prompt)
        self.assertNotIn("fact-1", prompt)
        plan = plan_chapter_shards_v2(skeleton, catalog, max_source_chars=30)[0]
        facts = [SourceSnapshotFactV2(
            document_id="document-1", fact_key=f"fact-{index}", scope_key=scope_id, fact_text="Nội dung",
        ) for index, scope_id in enumerate(plan.source_scope_ids, start=1)]
        shard_prompt = chapter_shard_prompt_v2("vi", skeleton, plan, facts)
        self.assertIn("SHARD_CONTEXT=", shard_prompt)
        self.assertIn("UNIT_DENSITY_BUDGET=", shard_prompt)
        self.assertIn("explicit pedagogical chain", shard_prompt)
        self.assertIn("align with its local learning_objective_refs", shard_prompt)
        facts[0] = facts[0].model_copy(update={"scope_key": "scope-khác"})
        with self.assertRaisesRegex(OrchestrationContractError, "ARCHITECTURE_SHARD_CONTEXT_MISMATCH"):
            chapter_shard_prompt_v2("vi", skeleton, plan, facts)


if __name__ == "__main__":
    unittest.main()
