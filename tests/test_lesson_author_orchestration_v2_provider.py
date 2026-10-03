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
        with self.assertRaisesRegex(ValueError, "ORCHESTRATION_V2_UNIT_CONTRACT_HASH_INVALID"):
            UnitGenerationContractV2.model_validate({**base, "contract_hash": "d" * 64})
        wrong_type = {**base, "component_plan": [{**base["component_plan"][0], "type": "unknown"}]}
        with self.assertRaises(ValueError):
            UnitGenerationContractV2.model_validate({**wrong_type, "contract_hash": canonical_hash(wrong_type)})

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
        self.assertIn("explicit pedagogical chain", shard_prompt)
        self.assertIn("align with its local learning_objective_refs", shard_prompt)
        facts[0] = facts[0].model_copy(update={"scope_key": "scope-khác"})
        with self.assertRaisesRegex(OrchestrationContractError, "ARCHITECTURE_SHARD_CONTEXT_MISMATCH"):
            chapter_shard_prompt_v2("vi", skeleton, plan, facts)


if __name__ == "__main__":
    unittest.main()
