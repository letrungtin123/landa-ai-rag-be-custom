from __future__ import annotations

import unittest

from app.main import enrich_lesson_author_blueprint_media_review


def blueprint(units: list[dict]) -> dict:
    return {
        "architecture_contract_version": 5,
        "chapters": [{
            "title": "Synthetic chapter",
            "lessons": [{"title": "Synthetic lesson", "units": units}],
        }],
    }


class LessonAuthorMediaReviewTests(unittest.TestCase):
    def test_media_review_uses_explicit_vi_decisions_and_source_backed_brief(self) -> None:
        candidate = blueprint([
            {
                "title": "Quy trình tổng hợp",
                "source_fact_ids": ["fact-1"],
                "learning_blocks": [{"intent": "procedure"}],
            },
            {
                "title": "Khái niệm chữ",
                "source_fact_ids": ["fact-2"],
                "learning_blocks": [{"intent": "concept_explanation"}],
            },
            {
                "title": "Cảnh báo chưa có text",
                "source_fact_ids": ["fact-missing"],
                "learning_blocks": [{"intent": "warning"}],
            },
        ])
        enriched, metrics = enrich_lesson_author_blueprint_media_review(candidate, {
            "facts": [
                {"fact_id": "fact-1", "text": "Bước một chuẩn bị trước khi thao tác."},
                {"fact_id": "fact-2", "text": "Khái niệm chỉ cần mô tả văn bản."},
            ],
        }, "vi")

        units = enriched["chapters"][0]["lessons"][0]["units"]
        decisions = enriched["media_review"]["decisions"]
        self.assertEqual([item["status"] for item in decisions], ["PROPOSED", "NOT_NEEDED", "SOURCE_GAP"])
        self.assertEqual(units[0]["media_plan"]["type"], "video")
        self.assertIn("Bước một", units[0]["media_plan"]["content_outline"])
        self.assertNotIn("media_plan", units[1])
        self.assertNotIn("media_plan", units[2])
        self.assertEqual(metrics["media_proposed_count"], 1)
        self.assertEqual(metrics["media_source_gap_count"], 1)

    def test_media_review_preserves_english_brief_and_capacity_failure(self) -> None:
        units = [
            {
                "title": f"Procedure {index}",
                "source_fact_ids": [f"fact-{index}"],
                "learning_blocks": [{"intent": "procedure"}],
            }
            for index in range(1, 14)
        ]
        enriched, _metrics = enrich_lesson_author_blueprint_media_review(
            blueprint(units),
            {"facts": [{"fact_id": f"fact-{index}", "text": f"Approved procedure evidence {index}."} for index in range(1, 14)]},
            "en",
        )
        result_units = enriched["chapters"][0]["lessons"][0]["units"]
        decisions = enriched["media_review"]["decisions"]
        self.assertEqual(sum("media_plan" in unit for unit in result_units), 12)
        self.assertEqual(decisions[-1]["status"], "FAILED")
        self.assertEqual(decisions[-1]["reason_code"], "MEDIA_RECOMMENDATION_CAPACITY_EXCEEDED")
        self.assertTrue(result_units[0]["media_plan"]["title"].startswith("Proposed walkthrough:"))


if __name__ == "__main__":
    unittest.main()
