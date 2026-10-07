from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import unittest

from app.main import build_chunks, extract_pdf
from app.instructional_quality import source_relationship_pairs


FIXTURE = Path(__file__).parent / "fixtures" / "ai_id_cp7_bic_quality_oracle.json"
DEFAULT_SOURCE = Path(
    r"C:\Users\PC\Downloads\Executive Training Playbook - BiC Modun 1_ Change Mindset.pdf"
)


def _normalized(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


class BicQualityOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_oracle_freezes_audited_baseline_and_release_authority(self) -> None:
        fixture = self.fixture
        self.assertEqual(
            fixture["contract_version"],
            "ai-id-cp7-bic-quality-oracle-v1",
        )
        baseline = fixture["audited_course_baseline"]
        self.assertEqual(baseline["component_counts"]["html"], 25)
        self.assertEqual(baseline["component_counts"]["la_diagram"], 2)
        self.assertEqual(baseline["open_single_choice_obligations"], 8)
        self.assertEqual(baseline["quality_state"], "needs_action")

        gates = fixture["release_gates"]
        self.assertFalse(gates["components"]["interaction_quota_required"])
        self.assertEqual(gates["components"]["problem_type"], "single_choice")
        self.assertFalse(gates["learner_content"]["source_citations_allowed"])
        self.assertFalse(
            gates["quality_state"]["validated_with_open_obligations_allowed"]
        )
        self.assertTrue(
            gates["quality_state"]["review_required_draft_must_remain_visible"]
        )

    def test_oracle_keeps_known_relations_explicit_and_non_ambiguous(self) -> None:
        cases = {
            item["case_id"]: item for item in self.fixture["reviewed_relations"]
        }
        self.assertEqual(
            cases["bic_made_in_world_ladder"]["required_pairs"],
            [
                ["Bậc 04", "Vietnam Know-how"],
                ["Bậc 05", "Vietnam Innovation"],
                ["Bậc 06", "Vietnam Brand"],
                ["ĐỈNH CAO", "MADE-IN-WORLD"],
            ],
        )
        self.assertEqual(
            [pair[0] for pair in cases["bic_pdca_full_cycle"]["required_pairs"]],
            ["PLAN", "DO", "CHECK", "ACTION", "ADVANCED"],
        )

    @unittest.skipUnless(DEFAULT_SOURCE.exists(), "real BiC source is not present")
    def test_current_extractor_preserves_reviewed_real_document_relations(self) -> None:
        expected_source = self.fixture["source"]
        self.assertEqual(
            hashlib.sha256(DEFAULT_SOURCE.read_bytes()).hexdigest(),
            expected_source["sha256"],
        )
        sections = extract_pdf(DEFAULT_SOURCE)
        self.assertEqual(len({section.page for section in sections}), 12)
        by_page = {int(section.page): section for section in sections if section.page}

        for relation in self.fixture["reviewed_relations"]:
            page = by_page[relation["page"]]
            normalized = _normalized(page.text)
            compiled_pairs = [
                (_normalized(left), _normalized(right))
                for left, right, _relation in source_relationship_pairs(page.text.splitlines())
            ]
            for term in relation["required_terms"]:
                self.assertIn(_normalized(term), normalized)
            for left, right in relation.get("required_pairs", []):
                self.assertRegex(
                    normalized,
                    re.escape(_normalized(left)) + r"[^\n]{0,160}" + re.escape(_normalized(right)),
                )
                required_left, required_right = _normalized(left), _normalized(right)
                self.assertTrue(
                    any(
                        actual_left == required_left
                        and (
                            actual_right == required_right
                            or actual_right.startswith(required_right + " ")
                        )
                        for actual_left, actual_right in compiled_pairs
                    ),
                    f"relationship compiler lost {relation['case_id']}: {left} -> {right}",
                )

        chunks = build_chunks(sections)
        self.assertTrue(chunks)
        self.assertTrue(all(chunk["metadata"].get("source_evidence_revision") for chunk in chunks))
        self.assertGreaterEqual(
            sum(int(section.metadata.get("table_count") or 0) for section in sections),
            8,
        )


if __name__ == "__main__":
    unittest.main()
