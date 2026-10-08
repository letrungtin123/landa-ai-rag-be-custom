from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.source_readiness import (
    MANUAL_REGION_REQUEST_VERSION,
    SOURCE_READINESS_VERSION,
    VISUAL_OBSERVATION_VERSION,
    VISUAL_REGION_VERSION,
    _recover_sparse_relation_regions,
    _sequence_observation,
    analyze_pdf_readiness,
    build_manual_region_request,
    build_visual_observation_record,
)


def _synthetic_pdf(path: Path) -> None:
    import pymupdf

    document = pymupdf.open()

    page = document.new_page(width=600, height=800)
    page.insert_text((60, 80), "Plain source page", fontsize=20)
    page.insert_text((60, 130), "This page contains one ordinary reading stream.", fontsize=12)

    page = document.new_page(width=600, height=800)
    page.insert_text((60, 80), "Two-column scenarios", fontsize=20)
    page.insert_textbox(pymupdf.Rect(50, 200, 260, 300), "Left item one\nLeft item two", fontsize=12)
    page.insert_textbox(pymupdf.Rect(340, 200, 550, 300), "Right item one\nRight item two", fontsize=12)

    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 240, 160), False)
    pixmap.clear_with(160)
    image = pixmap.tobytes("png")
    page = document.new_page(width=600, height=800)
    page.insert_text((60, 80), "Image situation", fontsize=20)
    page.insert_text((60, 130), "See image below", fontsize=12)
    page.insert_image(pymupdf.Rect(100, 180, 500, 500), stream=image)

    page = document.new_page(width=600, height=800)
    for offset in (0, 10, 20, 30):
        page.draw_line((50 + offset, 150), (550 - offset, 650), color=(1, 0, 0))
        page.draw_line((550 - offset, 150), (50 + offset, 650), color=(1, 0, 0))

    page = document.new_page(width=600, height=800)
    page.insert_text((60, 80), "Structured controls", fontsize=20)
    left, top, right, bottom = 80, 180, 520, 420
    for x in (left, 220, right):
        page.draw_line((x, top), (x, bottom), color=(0, 0, 0))
    for y in (top, 240, 300, 360, bottom):
        page.draw_line((left, y), (right, y), color=(0, 0, 0))
    values = (("Level", "Control"), ("1", "Eliminate"), ("2", "Substitute"), ("3", "Isolate"))
    for row, (first, second) in enumerate(values):
        y = 215 + row * 60
        page.insert_text((95, y), first, fontsize=11)
        page.insert_text((235, y), second, fontsize=11)

    page = document.new_page(width=600, height=800)
    page.insert_text((60, 80), "Vector composition", fontsize=20)
    for row in range(4):
        for column in range(5):
            x0 = 70 + column * 95
            y0 = 180 + row * 90
            page.draw_rect(
                pymupdf.Rect(x0, y0, x0 + 70, y0 + 55),
                color=(0.2, 0.4, 0.8),
                width=1,
            )
            page.insert_text((x0 + 8, y0 + 28), f"Card {row}-{column}", fontsize=8)

    document.save(path)
    document.close()


class SourceReadinessTests(unittest.TestCase):
    def test_runtime_readiness_routes_without_importing_golden_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cp3a-held-out-synthetic.pdf"
            _synthetic_pdf(path)
            result = analyze_pdf_readiness(path)

        self.assertEqual(result["readiness_version"], SOURCE_READINESS_VERSION)
        self.assertTrue(result["prototype_only"])
        self.assertFalse(result["network_used"])
        self.assertFalse(result["ocr_used"])
        pages = {page["page"]: page for page in result["pages"]}
        self.assertEqual(pages[1]["route"], "fast_parse")
        self.assertEqual(pages[2]["route"], "enrichment_required")
        self.assertIn("MULTI_COLUMN_READING_ORDER", pages[2]["flags"])
        self.assertEqual(pages[3]["route"], "enrichment_required")
        self.assertEqual(len(pages[3]["evidence_regions"]), 1)
        region = pages[3]["evidence_regions"][0]
        self.assertEqual(len(region["asset_revision"]), 64)
        self.assertEqual(region["observation"], {"status": "unreviewed", "facts": []})
        self.assertEqual(region["inference"], {"status": "not_performed", "claims": []})
        self.assertEqual(
            pages[3]["integrity_issues"][0]["code"],
            "REFERENCED_VISUAL_FOLLOWUP_PAGE_BROKEN",
        )
        self.assertEqual(pages[4]["route"], "review_required")
        self.assertIn("BROKEN_OR_EMPTY_PAGE", pages[4]["flags"])
        self.assertEqual(pages[5]["route"], "enrichment_required")
        self.assertGreaterEqual(
            pages[5]["runtime_observables"]["relation_table_candidate_count"],
            1,
        )
        self.assertTrue(all(
            "benchmark_text" not in candidate
            for page in result["pages"]
            for candidate in page["runtime_observables"]["relation_table_candidates"]
        ))
        self.assertTrue(pages[6]["runtime_observables"]["vector_visual_candidate"])
        self.assertEqual(pages[6]["runtime_observables"]["visual_region_count"], 1)
        vector_region = pages[6]["visual_regions"][0]
        self.assertEqual(vector_region["contract_version"], VISUAL_REGION_VERSION)
        self.assertEqual(vector_region["region_kind"], "vector_composition")
        self.assertEqual(len(vector_region["asset_revision"]), 64)
        self.assertLess(vector_region["locator"]["bbox_normalized"][2], 1.0)
        self.assertNotIn("expected_route", str(result))
        self.assertNotIn("critical_elements", str(result))

    def test_repeated_decorative_images_do_not_become_content_evidence(self) -> None:
        import pymupdf

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "repeated-decoration.pdf"
            pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 300, 200), False)
            pixmap.clear_with(220)
            image = pixmap.tobytes("png")
            document = pymupdf.open()
            for index in range(5):
                page = document.new_page(width=600, height=800)
                page.insert_text((60, 80), f"Page {index + 1}", fontsize=20)
                page.insert_text((60, 130), "Substantive text remains available.", fontsize=12)
                page.insert_image(pymupdf.Rect(300, 180, 560, 500), stream=image)
            document.save(path)
            document.close()

            result = analyze_pdf_readiness(path)

        self.assertTrue(all(not page["evidence_regions"] for page in result["pages"]))

    def test_manual_region_request_is_revision_and_locator_bound(self) -> None:
        request = build_manual_region_request(
            source_revision="source-revision-7",
            page=14,
            bbox_normalized=(0.2, 0.3, 0.8, 0.7),
            reason="  Verify the two severity tables.  ",
            requested_by_role="reviewer",
        )

        self.assertEqual(request["contract_version"], MANUAL_REGION_REQUEST_VERSION)
        self.assertEqual(request["source_revision"], "source-revision-7")
        self.assertEqual(request["locator"]["page"], 14)
        self.assertEqual(request["reason"], "Verify the two severity tables.")
        with self.assertRaises(ValueError):
            build_manual_region_request(
                source_revision="source-revision-7",
                page=14,
                bbox_normalized=(0.8, 0.3, 0.2, 0.7),
                reason="Invalid box",
                requested_by_role="reviewer",
            )
        with self.assertRaises(ValueError):
            build_manual_region_request(
                source_revision="source-revision-7",
                page=14,
                bbox_normalized=(0.2, 0.3, 0.8, 0.7),
                reason="Invalid role",
                requested_by_role="learner",
            )

    def test_sparse_full_page_table_recovers_only_bounded_stable_columns(self) -> None:
        class Row:
            def __init__(self, cells: list[tuple[float, float, float, float] | None]) -> None:
                self.cells = cells

        class Table:
            rows = [
                Row([(0, 0, 600, 800), None, None]),
                Row([None, (300, 200, 390, 240), (390, 200, 560, 240)]),
                Row([None, (300, 240, 390, 280), (390, 240, 560, 280)]),
                Row([None, (300, 280, 390, 320), (390, 280, 560, 320)]),
                Row([None, (300, 320, 390, 360), (390, 320, 560, 360)]),
            ]

        class Page:
            number = 0

            @staticmethod
            def get_text(*_args: object, **_kwargs: object) -> str:
                return "Group Value A Alpha B Beta C Gamma"

        recovered = _recover_sparse_relation_regions(
            Page(),
            Table(),
            [
                ["whole page", None, None],
                [None, "Group", "Value"],
                [None, "A", "Alpha"],
                [None, "B", "Beta"],
                [None, "C", "Gamma"],
            ],
            width=600,
            height=800,
            table_index=0,
            include_candidate_text=True,
        )

        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0]["row_count"], 4)
        self.assertEqual(recovered[0]["column_count"], 2)
        self.assertEqual(
            recovered[0]["extraction_method"],
            "full_page_sparse_region_recovery",
        )
        self.assertIn("Alpha", recovered[0]["benchmark_text"])

    def test_visual_observation_keeps_observation_separate_from_inference(self) -> None:
        record = build_visual_observation_record(
            source_revision="source-revision-9",
            asset_revision="a" * 64,
            page=22,
            bbox_normalized=(0.5, 0.2, 0.9, 0.7),
            observed_facts=["A five by five colored matrix is visible."],
            inference_claims=["The matrix may encode a risk level."],
            observer_role="reviewer",
        )

        self.assertEqual(record["contract_version"], VISUAL_OBSERVATION_VERSION)
        self.assertEqual(record["observation"]["status"], "observed")
        self.assertEqual(record["inference"]["status"], "requires_verification")
        with self.assertRaises(ValueError):
            build_visual_observation_record(
                source_revision="source-revision-9",
                asset_revision="not-a-digest",
                page=22,
                bbox_normalized=(0.5, 0.2, 0.9, 0.7),
                observed_facts=["Visible matrix."],
                observer_role="reviewer",
            )

    def test_sequence_observation_counts_arrow_timeline_and_checklist_structures(self) -> None:
        observation = _sequence_observation(
            "MINDSET → AMBITION → STRATEGY → ACTION\n"
            "30 Ngày: SEE DIFFERENT 30 Ngày: THINK DIFFERENT 30 Ngày: ACT DIFFERENT\n"
            "✔ First commitment\n✔ Second commitment\n✔ Third commitment"
        )

        self.assertEqual(observation["arrow_chain_count"], 1)
        self.assertEqual(observation["timed_stage_label_count"], 3)
        self.assertEqual(observation["bullet_step_count"], 3)
        self.assertEqual(observation["ordered_structure_candidate_count"], 3)
        self.assertTrue(observation["ordered_sequence_candidate"])


if __name__ == "__main__":
    unittest.main()
