from __future__ import annotations

from copy import deepcopy
import unittest

from app.source_evidence_bundle import build_source_evidence_bundle


SNAPSHOT = "b" * 64
SOURCE_REVISION = "a" * 64
ASSET_REVISION = "c" * 64


def fact(key: str, text: str, *, locator: dict | None = None) -> dict:
    return {
        "document_id": "document-1",
        "fact_key": key,
        "scope_key": "scope-1",
        "fact_text": text,
        "source_ref": "source-1",
        "source_page": 2,
        "source_chunk": 0,
        "locator": {"source_revision": SOURCE_REVISION, **(locator or {})},
    }


class SourceEvidenceBundleTests(unittest.TestCase):
    def test_legacy_source_revision_remains_visible_but_requires_review(self) -> None:
        bundle = build_source_evidence_bundle(
            source_snapshot_hash=SNAPSHOT,
            source_facts=[fact(
                "f1",
                "Nội dung từ chỉ mục cũ vẫn phải hiển thị cho người duyệt.",
                locator={"source_evidence_status": "legacy_review_required"},
            )],
            locale="vi",
        )

        self.assertEqual(bundle.status, "review_required")
        self.assertFalse(bundle.blocking)
        self.assertEqual(bundle.draft_visibility, "preserved")
        self.assertIn("STRUCTURED_EVIDENCE_REVISION_MISSING", bundle.review_requirements)

    def test_bundle_is_deterministic_and_preserves_table_positions(self) -> None:
        facts = [
            fact("f1", "Row 1: Hazard | Control | Owner"),
            fact(
                "f2",
                "Row 2: Chemical |  | HSE lead",
                locator={
                    "table_notes": ["Review after material substitution."],
                    "table_conditions": ["Use only for approved chemicals."],
                },
            ),
        ]

        first = build_source_evidence_bundle(
            source_snapshot_hash=SNAPSHOT,
            source_facts=facts,
            locale="vi",
        )
        second = build_source_evidence_bundle(
            source_snapshot_hash=SNAPSHOT,
            source_facts=deepcopy(facts),
            locale="vi",
        )

        self.assertEqual(first.bundle_hash, second.bundle_hash)
        self.assertEqual(first.materialized_source_revision, second.materialized_source_revision)
        table = next(element for element in first.elements if element.kind == "table")
        self.assertEqual(table.payload["headers"], ["Hazard", "Control", "Owner"])
        self.assertEqual(table.payload["rows"][1], ["Chemical", "", "HSE lead"])
        self.assertEqual(table.payload["notes"], ["Review after material substitution."])
        self.assertEqual(table.payload["conditions"], ["Use only for approved chemicals."])

    def test_process_and_hierarchy_relations_are_explicit(self) -> None:
        facts = [
            fact("f1", "Bước 1: Xác định mối nguy trong khu vực làm việc.", locator={"heading_path": ["HIRA", "Nhận diện"]}),
            fact("f2", "Bước 2: Đánh giá mức độ rủi ro theo tiêu chí.", locator={"heading_path": ["HIRA", "Đánh giá"]}),
            fact("f3", "Bước 3: Chọn biện pháp kiểm soát phù hợp.", locator={"heading_path": ["HIRA", "Kiểm soát"]}),
        ]

        bundle = build_source_evidence_bundle(
            source_snapshot_hash=SNAPSHOT,
            source_facts=facts,
            locale="vi",
        )

        process = next(element for element in bundle.elements if element.kind == "process")
        self.assertEqual(process.payload["order_constraints"], [
            {"before": 0, "after": 1},
            {"before": 1, "after": 2},
        ])
        hierarchy = next(element for element in bundle.elements if element.kind == "hierarchy")
        nodes = hierarchy.payload["nodes"]
        root = next(node for node in nodes if node["parent_id"] is None)
        self.assertEqual(root["label"], "HIRA")
        self.assertEqual(root["priority"], 0)
        self.assertTrue(all(node["parent_id"] == root["id"] for node in nodes if node is not root))

    def test_unreviewed_visual_keeps_draft_visible_and_requires_review(self) -> None:
        facts = [fact("f1", "Quan sát sơ đồ trước khi trả lời câu hỏi.", locator={
            "visual_prompt_text": "Biện pháp nào có mức ưu tiên cao nhất?",
            "visual_regions": [{
                "region_kind": "embedded_image",
                "asset_revision": ASSET_REVISION,
                "locator": {"page": 2, "bbox_normalized": [0.1, 0.2, 0.8, 0.9]},
                "observation": {"status": "unreviewed", "facts": []},
                "inference": {"status": "not_performed", "claims": []},
            }],
        })]

        bundle = build_source_evidence_bundle(
            source_snapshot_hash=SNAPSHOT,
            source_facts=facts,
            locale="vi",
        )

        visual = next(element for element in bundle.elements if element.kind == "visual")
        self.assertEqual(bundle.status, "review_required")
        self.assertFalse(bundle.blocking)
        self.assertEqual(bundle.draft_visibility, "preserved")
        self.assertEqual(visual.representation_status, "candidate_unverified")
        self.assertEqual(visual.payload["prompt_text"], "Biện pháp nào có mức ưu tiên cao nhất?")
        self.assertEqual(visual.payload["observed_facts"], [])

    def test_reviewed_observation_materializes_new_revision_without_changing_ownership(self) -> None:
        raw_facts = [fact("f1", "Đọc sơ đồ kiểm soát rủi ro.", locator={
            "visual_regions": [{
                "region_kind": "embedded_image",
                "asset_revision": ASSET_REVISION,
                "locator": {"page": 2, "bbox_normalized": [0.1, 0.2, 0.8, 0.9]},
                "observation": {"status": "unreviewed", "facts": []},
                "inference": {"status": "not_performed", "claims": []},
            }],
        })]
        owned_before = [item["fact_key"] for item in raw_facts]
        unreviewed = build_source_evidence_bundle(
            source_snapshot_hash=SNAPSHOT,
            source_facts=raw_facts,
            locale="vi",
        )
        reviewed_facts = deepcopy(raw_facts)
        reviewed_facts[0]["locator"]["visual_regions"][0]["observation"] = {
            "status": "observed",
            "facts": ["Sơ đồ sắp xếp năm tầng kiểm soát theo thứ tự ưu tiên."],
        }
        reviewed = build_source_evidence_bundle(
            source_snapshot_hash=SNAPSHOT,
            source_facts=reviewed_facts,
            locale="vi",
        )

        self.assertNotEqual(unreviewed.materialized_source_revision, reviewed.materialized_source_revision)
        self.assertEqual(unreviewed.status, "review_required")
        self.assertEqual(reviewed.status, "ready")
        self.assertEqual([item["fact_key"] for item in raw_facts], owned_before)
        self.assertEqual([item["fact_key"] for item in reviewed_facts], owned_before)
        visual = next(element for element in reviewed.elements if element.kind == "visual")
        self.assertEqual(visual.payload["observed_facts"], [
            "Sơ đồ sắp xếp năm tầng kiểm soát theo thứ tự ưu tiên."
        ])


if __name__ == "__main__":
    unittest.main()
