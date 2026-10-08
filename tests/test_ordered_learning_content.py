"""Ordered payload acceptance, ownership and observations; offline only."""
from copy import deepcopy
import unittest

from app.services.lesson_author.staged.validation import staged_instructional_finding, staged_component_repair_targets
from app.services.lesson_author.proposal_validation import semantic_learning_visible_text
from app.ordered_learning_content import ordered_content_fields
from app.lesson_content_observation import observe_lesson_content
from app.media_brief import build_media_brief


def payload():
    return {"version": 2, "sections": [
        {"heading": "First framework", "learning_block_ids": ["teach_a"], "blocks": [
            {"kind": "paragraph", "text": "Explain the approved framework and its conditions. " * 8},
            {"kind": "table", "rows": [{"label": "Level one", "value": "Recognize the stated condition."}]}]},
        {"heading": "Second framework", "learning_block_ids": ["teach_b"], "blocks": [
            {"kind": "bullets", "items": ["Compare the explicitly stated alternatives."]},
            {"kind": "task", "text": "Fill the source canvas; justify each entry using its stated categories."}]}]}


class OrderedContentTests(unittest.TestCase):
    def test_ordered_fields_preserve_context_and_input(self):
        value = payload()
        before = deepcopy(value)
        self.assertIsNone(semantic_learning_visible_text(value)[1])
        fields = ordered_content_fields(value)
        self.assertEqual(fields[2][0], "sections[0].blocks[1].rows[0]")
        self.assertEqual(fields[3], ("sections[1].heading", "Second framework"))
        self.assertEqual(value, before)

    def test_mixed_unknown_empty_and_aggregate_overflow_rejected(self):
        cases = []
        for field, value in [("paragraphs", ["silently discarded"]), ("version", 3), ("sections", [])]:
            p = payload(); p[field] = value; cases.append(p)
        for block in [{"kind": "unknown", "text": "bad"}, {"kind": "paragraph", "text": "ok", "items": ["bad"]},
                      {"kind": "table", "rows": [{"label": "empty value"}]}, {"kind": "paragraph", "text": "x" * 2001}]:
            p = payload(); p["sections"][0]["blocks"] = [block]; cases.append(p)
        p = payload(); p["sections"][0]["blocks"] = [{"kind": "bullets", "items": ["item"] * 11}]
        p["sections"][1]["blocks"] = [{"kind": "bullets", "items": ["item"] * 10}]; cases.append(p)
        for p in cases:
            self.assertIsNotNone(semantic_learning_visible_text(p)[1])

    def test_missing_group_repair_is_scoped_and_unknown_group_rejected(self):
        component = {"type": "html", "semantic_content": payload(), "source_fact_ids": [], "covered_source_fact_ids": []}
        plan = {"type": "html", "source_fact_ids": [], "learning_block_ids": ["teach_a", "teach_b", "teach_c"]}
        finding = staged_instructional_finding(component, 0, plan)
        self.assertEqual(finding.code, "HTML_TEACHING_GROUP_MISSING")
        self.assertTrue(finding.repairable)
        self.assertEqual(staged_component_repair_targets({"components": [component]}, {"component_plan": [plan]}), [0])
        plan["learning_block_ids"] = ["teach_a", "teach_b"]
        self.assertIsNone(staged_instructional_finding(component, 0, plan))
        plan["learning_block_ids"] = ["teach_a"]
        self.assertEqual(staged_instructional_finding(component, 0, plan).code, "HTML_TEACHING_GROUP_OUT_OF_SCOPE")

    def test_heading_or_metadata_is_not_teaching_witness(self):
        fact = "The third framework defines the approved sustainable operating condition."
        p = payload(); p["sections"][0]["heading"] = fact
        unit = {"components": [{"type": "html", "semantic_content": p, "covered_source_fact_ids": ["f1"]}]}
        expected = {"source_fact_ids": ["f1"], "component_plan": [{"type": "html", "source_fact_ids": ["f1"]}]}
        report = observe_lesson_content(unit, expected, {"facts": [{"fact_id": "f1", "text": fact}]})
        self.assertEqual(report["semantic_coverage"], "not_measured")
        self.assertNotIn("LEXICAL_WITNESS", report["fact_state_counts"])
        p["sections"][1]["blocks"].append({"kind": "paragraph", "text": fact})
        report = observe_lesson_content(unit, expected, {"facts": [{"fact_id": "f1", "text": fact}]})
        self.assertEqual(report["fact_state_counts"]["LEXICAL_WITNESS"], 1)
        self.assertEqual(report["semantic_coverage"], "not_measured")

    def test_legacy_remains_accepted(self):
        self.assertEqual(semantic_learning_visible_text({"paragraphs": ["Legacy text."]}), ("Legacy text.", None))

    def test_structural_quality_telemetry_contains_no_teaching_text(self):
        unit = {"components": [{"type": "html", "semantic_content": payload()},
                               {"type": "la_diagram", "nodes": [{"label": "A"}, {"label": "B"}, {"label": "C"}],
                                "edges": [{"source": 0, "target": 1}]}]}
        expected = {"component_plan": [{"type": "html", "learning_block_ids": ["teach_a", "teach_b", "teach_c"]}]}
        report = observe_lesson_content(unit, expected, {})
        self.assertEqual(report["ordered_section_count"], 2)
        self.assertEqual(report["practice_task_count"], 1)
        self.assertIn("HTML_TEACHING_GROUP_NOT_PRESENT", report["finding_counts"])
        self.assertIn("DIAGRAM_ISOLATED_NODES_REVIEW", report["finding_counts"])
        self.assertNotIn("Explain the approved", str(report))
        self.assertFalse(report["blocking"])

    def test_media_briefs_are_bounded_source_exact_and_locale_honest(self):
        texts = [f"Approved comparison point number {i} is stated in this source." for i in range(20)]
        for locale in ("vi", "en"):
            result = build_media_brief("Approved framework", "static_infographic", texts, locale)
            self.assertEqual(result, build_media_brief("Approved framework", "static_infographic", texts, locale))
            self.assertEqual(len(result["content_points"]), 6)
            self.assertEqual(result["content_points"][0], texts[0])
            self.assertEqual(result["content_points"][-1], texts[-1])
            self.assertTrue(all(p in texts for p in result["content_points"]))
            self.assertEqual(result["evidence_language"], "original")
            self.assertEqual(result["brief_version"], 2)
            self.assertIn("Approved framework", result["context_description"])
        self.assertIsNone(build_media_brief("Gap", "video", ["Page 7", "TITLE", "x" * 501], "en"))
