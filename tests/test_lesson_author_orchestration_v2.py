import unittest

from app.lesson_author_orchestration_v2 import (
    ChapterBlueprintShardV2,
    CourseSkeletonV2,
    OrchestrationContractError,
    assemble_blueprint_v2,
)


def skeleton() -> CourseSkeletonV2:
    return CourseSkeletonV2.model_validate({
        "contract_version": 2,
        "source_snapshot_hash": "1" * 64,
        "locale": "vi",
        "title": "Khóa học",
        "summary": "Tóm tắt",
        "target_audience": "Quản lý",
        "prerequisites": [],
        "learning_outcomes": ["Áp dụng nội dung"],
        "assessment_strategy": "Đánh giá theo chương",
        "assumptions": [],
        "chapters": [
            {"chapter_key": "chapter-1", "order": 0, "title": "Chương 1", "objective": "Mục tiêu 1",
             "learning_outcomes": ["Kết quả 1"],
             "source_scope_ids": ["scope-1"]},
            {"chapter_key": "chapter-2", "order": 1, "title": "Chương 2", "objective": "Mục tiêu 2",
             "learning_outcomes": ["Kết quả 2"],
             "source_scope_ids": ["scope-2"]},
        ],
    })


def shard(key: str, order: int, title: str, scopes: list[str]) -> ChapterBlueprintShardV2:
    return ChapterBlueprintShardV2.model_validate({
        "contract_version": 2,
        "source_snapshot_hash": "1" * 64,
        "chapter_key": key,
        "order": order,
        "shard_index": 0,
        "shard_count": 1,
        "source_scope_ids": scopes,
        "title": title,
        "objective": f"Mục tiêu {order + 1}",
        "lessons": [{
            "title": "Bài học", "objective": "Hiểu nội dung", "learning_objectives": ["Áp dụng"],
            "learning_activities": ["Đọc và thực hành"], "assessment": "Bài kiểm tra",
            "units": [{"title": "Nội dung", "purpose": "Giải thích nội dung nguồn",
                       "learning_objective_refs": ["lo_1"], "source_scope_ids": scopes,
                       "component_plan": [{"type": "html", "title": "Giải thích", "rationale": "Nội dung chính",
                                           "author_review": {"purpose": "Giải thích nội dung", "example_scenario": None,
                                                             "visual_asset": None, "user_behavior_navigation": None},
                                           "source_scope_ids": scopes}], "media_brief": None}],
        }],
    })


class LessonAuthorOrchestrationV2Tests(unittest.TestCase):
    def test_assembles_out_of_order_shards_deterministically(self) -> None:
        result = assemble_blueprint_v2(skeleton(), [
            shard("chapter-2", 1, "Chương 2", ["scope-2"]),
            shard("chapter-1", 0, "Chương 1", ["scope-1"]),
        ], {"scope-1": ["fact-1", "fact-2"], "scope-2": ["fact-3"]})
        self.assertEqual([chapter["title"] for chapter in result.blueprint["chapters"]], ["Chương 1", "Chương 2"])
        self.assertEqual(result.admitted_fact_count, 3)
        self.assertEqual(result.covered_fact_count, 3)
        self.assertEqual(len(result.assembly_hash), 64)

    def test_rejects_missing_or_cross_chapter_coverage(self) -> None:
        with self.assertRaisesRegex(OrchestrationContractError, "ARCHITECTURE_SOURCE_SCOPE_FACTS_INVALID"):
            assemble_blueprint_v2(skeleton(), [
                shard("chapter-1", 0, "Chương 1", ["scope-1"]),
                shard("chapter-2", 1, "Chương 2", ["scope-2"]),
            ], {"scope-1": ["fact-1", "fact-2"], "scope-2": []})
        with self.assertRaisesRegex(OrchestrationContractError, "ARCHITECTURE_SHARD_COVERAGE_INCOMPLETE"):
            assemble_blueprint_v2(skeleton(), [
                shard("chapter-1", 0, "Chương 1", ["scope-2"]),
                shard("chapter-2", 1, "Chương 2", ["scope-1"]),
            ], {"scope-1": ["fact-1", "fact-2"], "scope-2": ["fact-3"]})

    def test_rejects_duplicate_scope_ownership_in_skeleton(self) -> None:
        value = skeleton().model_dump()
        value["chapters"][1]["source_scope_ids"] = ["scope-1"]
        with self.assertRaisesRegex(ValueError, "exactly one owning chapter"):
            CourseSkeletonV2.model_validate(value)

    def test_media_brief_is_explicit_typed_and_bullet_ready(self) -> None:
        value = shard("chapter-1", 0, "Chương 1", ["scope-1"]).model_dump()
        value["lessons"][0]["units"][0]["media_brief"] = {
            "type": "video",
            "title": "Tình huống lãnh đạo",
            "content_points": ["Bối cảnh", "Quyết định"],
            "context_description": "Cuộc họp điều hành",
            "rationale": "Giúp người học áp dụng mô hình",
        }
        parsed = ChapterBlueprintShardV2.model_validate(value)
        self.assertEqual(parsed.lessons[0].units[0].media_brief.content_points, ["Bối cảnh", "Quyết định"])
        del value["lessons"][0]["units"][0]["media_brief"]
        with self.assertRaises(ValueError):
            ChapterBlueprintShardV2.model_validate(value)


if __name__ == "__main__":
    unittest.main()
