from __future__ import annotations

import json
import unittest
from app.lesson_author_blueprint import (
    LessonAuthorBlueprintValidationError,
    parse_and_validate_lesson_author_blueprint,
)
from app.source_map import build_course_architect_context, build_source_map, format_source_map_for_course_architect


def source_nodes() -> list[dict[str, object]]:
    return [
        {
            "document_id": "doc-1", "document_name": "Safety guide", "source_ref": "src-001",
            "title": "Foundations", "level": 1, "order": 1, "logical_page": 1,
        },
        {
            "document_id": "doc-1", "document_name": "Safety guide", "source_ref": "src-002",
            "parent_source_ref": "src-001", "title": "Applying controls", "level": 2, "order": 2,
            "logical_page": 2,
        },
    ]


def source_manifest() -> dict[str, object]:
    return {
        "truncated": False,
        "facts": [
            {"fact_id": "p1-f1", "document_id": "doc-1", "source_ref": "src-001", "source_page": 1, "source_chunk": 0},
            {"fact_id": "p2-f1", "document_id": "doc-1", "source_ref": "src-002", "source_page": 2, "source_chunk": 1},
        ],
    }


def scaled_source_nodes(section_count: int = 6) -> list[dict[str, object]]:
    return [
        {
            "document_id": "doc-scale", "document_name": "Scale guide", "source_ref": f"src-{index:03d}",
            "title": f"Section {index}", "level": 1, "order": index, "logical_page": index,
        }
        for index in range(1, section_count + 1)
    ]


def scaled_manifest(fact_count: int, section_count: int = 6) -> dict[str, object]:
    return {
        "truncated": False,
        "fact_scope_complete": True,
        "total_fact_count": fact_count,
        "represented_fact_count": fact_count,
        "facts": [
            {
                "fact_id": f"fact-{index:04d}",
                "document_id": "doc-scale",
                "source_ref": f"src-{((index - 1) % section_count) + 1:03d}",
                "source_page": ((index - 1) % section_count) + 1,
                "source_chunk": (index - 1) % 36,
                "text": f"Source fact {index} with stable provenance.",
            }
            for index in range(1, fact_count + 1)
        ],
    }


def source_map_blueprint(source_map: dict[str, object]) -> dict[str, object]:
    concepts = source_map["concepts"]  # type: ignore[index]
    foundation = concepts[0]["id"]  # type: ignore[index]
    application = concepts[1]["id"]  # type: ignore[index]
    return {
        "architecture_contract_version": 3,
        "content_contract_version": 1,
        "title": "Safety guide",
        "summary": "A source-grounded architecture.",
        "target_audience": "New operators",
        "prerequisites": [],
        "learning_outcomes": ["Identify hazards.", "Explain controls.", "Apply controls."],
        "course_outcomes": ["Identify hazards.", "Explain controls.", "Apply controls."],
        "assessment_strategy": "Assess application of controls.",
        "assumptions": [],
        "chapters": [{
            "title": "Safe work",
            "objective": "Apply safety controls.",
            "learning_objectives": ["lo_1"],
            "concept_ids": [foundation, application],
            "source_refs": ["src-001", "src-002"],
            "lessons": [{
                "title": "Foundations and controls",
                "objective": "Identify hazards and apply controls.",
                "learning_objectives": ["lo_1"],
                "learning_activities": ["Review evidence and practice."],
                "assessment": "Apply the control process.",
                "primary_concept_ids": [foundation, application],
                "supporting_concept_ids": [],
                "prerequisite_concept_ids": [],
                "assessment_required": True,
                "assessment_objective_refs": ["lo_1"],
                "source_refs": ["src-001", "src-002"],
                "units": [{
                    "title": "Control process",
                    "purpose": "Explain and assess the source-supported control process.",
                    "concept_ids": [foundation, application],
                    "learning_objective_refs": ["lo_1"],
                    "source_refs": ["src-001", "src-002"],
                    "source_fact_ids": ["p1-f1", "p2-f1"],
                    "learning_blocks": [{
                        "id": "lb_1", "intent": "concept_explanation", "importance": "core",
                        "source_fact_ids": ["p1-f1", "p2-f1"], "learning_objective_refs": ["lo_1"],
                        "content": {"relationship_evidence": True},
                    }],
                }],
            }],
        }],
    }


class SourceMapTests(unittest.TestCase):
    def test_all_sections_and_facts_keep_provenance(self) -> None:
        source_map = build_source_map(source_nodes(), source_manifest(), locale="en")

        self.assertEqual([section["source_ref"] for section in source_map["sections"]], ["src-001", "src-002"])
        self.assertEqual([fact["id"] for fact in source_map["facts"]], ["p1-f1", "p2-f1"])
        self.assertEqual(source_map["facts"][1]["source_ref"], "src-002")
        self.assertEqual(source_map["facts"][1]["page"], 2)
        self.assertEqual(source_map["concepts"][1]["prerequisite_concept_ids"], [source_map["concepts"][0]["id"]])
        self.assertTrue(source_map["coverage"]["fact_scope_complete"])

    def test_genuinely_incomplete_manifest_remains_explicitly_incomplete(self) -> None:
        manifest = source_manifest()
        manifest["total_fact_count"] = 3
        manifest["represented_fact_count"] = 2
        manifest["fact_scope_complete"] = False
        manifest["truncated"] = True
        manifest["incomplete_reason"] = "SOURCE_FACT_EXTRACTION_CAPACITY_EXCEEDED"
        source_map = build_source_map(source_nodes(), manifest)
        self.assertFalse(source_map["coverage"]["fact_scope_complete"])
        self.assertIn("SOURCE_MAP_FACT_SCOPE_INCOMPLETE", source_map["warnings"])

        _context, truncated = format_source_map_for_course_architect(source_map, max_chars=1)
        self.assertTrue(truncated)

    def test_uat_scale_371_facts_is_complete_with_bounded_hierarchical_context(self) -> None:
        source_map = build_source_map(scaled_source_nodes(), scaled_manifest(371), locale="en")
        context = build_course_architect_context(source_map, scaled_manifest(371), max_chars=6_000)

        self.assertEqual(source_map["coverage"]["total_fact_count"], 371)
        self.assertEqual(source_map["coverage"]["represented_fact_count"], 371)
        self.assertTrue(source_map["coverage"]["fact_scope_complete"])
        self.assertEqual(len(source_map["facts"]), 371)
        self.assertEqual(len(source_map["fact_partitions"]), 6)
        self.assertTrue(context["context_complete"])
        self.assertLessEqual(context["diagnostics"]["architect_context_size"], 6_000)
        self.assertEqual(context["diagnostics"]["source_total_facts"], 371)
        self.assertLess(context["diagnostics"]["architect_detail_fact_count"], 371)
        self.assertIn('"sections"', context["context"])
        self.assertIn('"concepts"', context["context"])

    def test_fact_counts_around_legacy_cap_and_1000_plus_are_lossless_and_deterministic(self) -> None:
        for fact_count in (239, 240, 241, 371, 1_001):
            manifest = scaled_manifest(fact_count)
            first = build_source_map(scaled_source_nodes(), manifest, locale="en")
            second = build_source_map(scaled_source_nodes(), manifest, locale="en")
            context = build_course_architect_context(first, manifest, max_chars=7_000)

            self.assertEqual(first["coverage"]["total_fact_count"], fact_count)
            self.assertEqual(first["coverage"]["represented_fact_count"], fact_count)
            self.assertTrue(first["coverage"]["fact_scope_complete"])
            self.assertEqual(len(first["facts"]), fact_count)
            self.assertEqual(first["facts"], second["facts"])
            if fact_count == 1_001:
                # V5 never removes a scope descriptor merely to satisfy an
                # artificial small Architect budget.  The production budget
                # is larger; at this test budget the complete inventory plus
                # one representative per scope must fail closed.
                self.assertFalse(context["context_complete"])
                self.assertEqual(context["error_code"], "EVIDENCE_SCOPE_CONTEXT_INCOMPLETE")
            else:
                self.assertTrue(context["context_complete"])
                self.assertLessEqual(context["diagnostics"]["architect_context_size"], 7_000)

    def test_architecture_contract_requires_semantic_fact_ownership(self) -> None:
        source_map = build_source_map(source_nodes(), source_manifest())
        parsed = parse_and_validate_lesson_author_blueprint(json.dumps(source_map_blueprint(source_map)))
        unit = parsed["chapters"][0]["lessons"][0]["units"][0]
        self.assertEqual(parsed["architecture_contract_version"], 3)
        self.assertEqual(unit["component_plan"], [])
        self.assertEqual(unit["learning_blocks"][0]["intent"], "concept_explanation")

        invalid = source_map_blueprint(source_map)
        invalid["chapters"][0]["lessons"][0]["units"][0]["learning_blocks"][0]["source_fact_ids"] = []  # type: ignore[index]
        with self.assertRaises(LessonAuthorBlueprintValidationError):
            parse_and_validate_lesson_author_blueprint(json.dumps(invalid))


if __name__ == "__main__":
    unittest.main()
