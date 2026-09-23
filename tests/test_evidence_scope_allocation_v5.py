from __future__ import annotations

import copy
import unittest

from app.main import (
    allocate_blueprint_source_fact_ids,
    allocate_source_map_architecture_facts,
    validate_course_architecture_evidence_scope,
    validate_course_architecture_workflow,
)
from app.source_map import build_course_architect_context, build_source_map


def _nodes(section_count: int) -> list[dict[str, object]]:
    return [{
        "document_id": "doc-v5",
        "document_name": "Evidence scope guide",
        "source_ref": f"src-{index:03d}",
        "title": f"Section {index}",
        "level": 1,
        "order": index,
        "logical_page": index,
    } for index in range(1, section_count + 1)]


def _manifest(fact_count: int, section_count: int) -> dict[str, object]:
    return {
        "truncated": False,
        "fact_scope_complete": True,
        "total_fact_count": fact_count,
        "represented_fact_count": fact_count,
        "facts": [{
            "fact_id": f"fact-{index:04d}",
            "document_id": "doc-v5",
            "source_ref": f"src-{((index - 1) % section_count) + 1:03d}",
            "source_page": ((index - 1) % section_count) + 1,
            "source_chunk": (index - 1) % 48,
            "text": f"Evidence fact {index}: source-backed procedure or definition.",
        } for index in range(1, fact_count + 1)],
    }


def _v5_blueprint(source_map: dict[str, object], *, broad_concept: bool = False) -> dict[str, object]:
    section_by_id = {str(section["id"]): section for section in source_map["sections"]}  # type: ignore[index]
    concept_by_ref = {
        str(section_by_id[str(concept["source_section_ids"][0])]["source_ref"]): str(concept["id"])
        for concept in source_map["concepts"]  # type: ignore[index]
    }
    scopes_by_ref: dict[str, list[dict[str, object]]] = {}
    for scope in source_map["source_evidence_scopes"]:  # type: ignore[index]
        scopes_by_ref.setdefault(str(scope["source_ref"]), []).append(scope)
    chapters: list[dict[str, object]] = []
    for index, (ref, scopes) in enumerate(sorted(scopes_by_ref.items()), start=1):
        concept_id = concept_by_ref[ref]
        primary_ids = [str(scope["id"]) for scope in scopes]
        unit = {
            "title": f"Unit {index}", "purpose": "Teach the bounded evidence scope.",
            "concept_ids": [concept_id], "primary_concept_ids": [concept_id],
            "learning_objective_refs": ["lo_1"], "source_refs": [ref],
            "learning_blocks": [{
                "id": f"lb_{index}", "intent": "concept_explanation", "importance": "core",
                "concept_ids": [concept_id], "primary_concept_ids": [concept_id],
                "primary_evidence_scope_ids": primary_ids,
                "supporting_evidence_scope_ids": [], "source_refs": [ref],
                "learning_objective_refs": ["lo_1"], "content": {},
            }],
        }
        lessons = [{
            "title": f"Lesson {index}", "objective": f"Identify and explain section {index}.",
            "learning_objectives": [f"Identify the source-grounded scope in section {index}."],
            "learning_activities": ["Review source evidence."], "assessment": "Use a grounded check.",
            "primary_concept_ids": [concept_id], "supporting_concept_ids": [],
            "prerequisite_concept_ids": [], "assessment_required": False,
            "assessment_objective_refs": [], "source_refs": [ref], "units": [unit],
        }]
        if broad_concept and index == 1 and len(primary_ids) >= 2:
            # The same broad concept is allowed across two lessons; ownership
            # remains exact at evidence-scope level rather than concept level.
            first, rest = primary_ids[:1], primary_ids[1:]
            lessons[0]["units"][0]["learning_blocks"][0]["primary_evidence_scope_ids"] = first  # type: ignore[index]
            lessons.append({
                "title": "Lesson 1 reinforcement", "objective": "Apply the same broad concept in a second coherent lesson.",
                "learning_objectives": ["Apply the source-grounded concept."], "learning_activities": ["Practice."],
                "assessment": "Check application.", "primary_concept_ids": [concept_id], "supporting_concept_ids": [],
                "prerequisite_concept_ids": [], "assessment_required": False, "assessment_objective_refs": [], "source_refs": [ref],
                "units": [{
                    "title": "Continuation", "purpose": "Continue the broad source concept.",
                    "concept_ids": [concept_id], "primary_concept_ids": [concept_id], "learning_objective_refs": ["lo_1"], "source_refs": [ref],
                    "learning_blocks": [{
                        "id": "lb_broad_2", "intent": "worked_example", "importance": "core", "concept_ids": [concept_id],
                        "primary_concept_ids": [concept_id], "primary_evidence_scope_ids": rest,
                        "supporting_evidence_scope_ids": [first[0]], "source_refs": [ref], "learning_objective_refs": ["lo_1"], "content": {},
                    }],
                }],
            })
        chapters.append({
            "title": f"Chapter {index}", "objective": f"Explain section {index}.",
            "learning_objectives": [f"Explain section {index}."], "concept_ids": [concept_id], "source_refs": [ref], "lessons": lessons,
        })
    return {
        "architecture_contract_version": 5, "content_contract_version": 1,
        "title": "Evidence scope guide", "summary": "Server-owned evidence scope blueprint.",
        "target_audience": "Operators", "prerequisites": [],
        "learning_outcomes": ["Identify source-backed concepts.", "Explain source-backed concepts.", "Apply source-backed concepts."],
        "course_outcomes": ["Identify source-backed concepts.", "Explain source-backed concepts.", "Apply source-backed concepts."],
        "assessment_strategy": "Grounded checks.", "assumptions": [], "chapters": chapters,
    }


class EvidenceScopeAllocationV5Tests(unittest.TestCase):
    def _finalize(self, fact_count: int, section_count: int, *, broad_concept: bool = False) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        manifest = _manifest(fact_count, section_count)
        source_map = build_source_map(_nodes(section_count), manifest, locale="en")
        candidate = _v5_blueprint(source_map, broad_concept=broad_concept)
        allocated = allocate_source_map_architecture_facts(candidate, source_map, manifest)
        finalized = allocate_blueprint_source_fact_ids(allocated, manifest, _nodes(section_count))
        return finalized, source_map, manifest

    def test_371_facts_allocate_once_and_supporting_reference_does_not_duplicate_facts(self) -> None:
        finalized, source_map, manifest = self._finalize(371, 6, broad_concept=True)
        allocation = finalized["source_fact_allocation"]  # type: ignore[index]
        scopes = finalized["source_evidence_scope_allocation"]  # type: ignore[index]
        self.assertEqual(allocation["allocated_count"], 371)
        self.assertTrue(allocation["complete"])
        self.assertTrue(scopes["complete"])
        self.assertEqual(len(allocation["unallocated"]), 0)
        self.assertFalse(validate_course_architecture_workflow(finalized, source_map, manifest, {f"src-{index:03d}" for index in range(1, 7)}).errors)
        blocks = finalized["chapters"][0]["lessons"]  # type: ignore[index]
        self.assertEqual(len(blocks), 2)
        support_block = blocks[1]["units"][0]["learning_blocks"][0]
        self.assertEqual(support_block["source_fact_ids"], [
            item["fact_id"] for item in allocation["allocations"] if item["learning_block_id"] == "lb_broad_2"
        ])

    def test_233_facts_create_multiple_bounded_scopes_with_representative_context(self) -> None:
        manifest = _manifest(233, 1)
        source_map = build_source_map(_nodes(1), manifest, locale="en")
        scopes = source_map["source_evidence_scopes"]  # type: ignore[index]
        self.assertGreater(len(scopes), 1)
        self.assertTrue(source_map["coverage"]["evidence_scope_complete"])  # type: ignore[index]
        self.assertTrue(all(scope["evidence_char_count"] > 0 for scope in scopes))
        context = build_course_architect_context(source_map, manifest)
        self.assertTrue(context["context_complete"])
        self.assertIn('"e"', context["context"])
        self.assertIn('"c"', context["context"])

    def test_unknown_duplicate_missing_and_cross_section_scopes_fail_closed(self) -> None:
        manifest = _manifest(20, 2)
        source_map = build_source_map(_nodes(2), manifest, locale="en")
        candidate = _v5_blueprint(source_map)
        block_one = candidate["chapters"][0]["lessons"][0]["units"][0]["learning_blocks"][0]  # type: ignore[index]
        block_two = candidate["chapters"][1]["lessons"][0]["units"][0]["learning_blocks"][0]  # type: ignore[index]
        duplicate_scope = block_one["primary_evidence_scope_ids"][0]
        block_two["primary_evidence_scope_ids"].append(duplicate_scope)
        block_one["supporting_evidence_scope_ids"] = ["unknown_scope"]
        block_one["concept_ids"] = block_two["concept_ids"]
        finding_codes = {issue["code"] for issue in validate_course_architecture_evidence_scope(candidate, source_map).issues}
        self.assertIn("UNKNOWN_EVIDENCE_SCOPE", finding_codes)
        self.assertIn("DUPLICATE_PRIMARY_EVIDENCE_SCOPE_OWNER", finding_codes)
        self.assertIn("EVIDENCE_SCOPE_CONCEPT_MISMATCH", finding_codes)
        allocated = allocate_source_map_architecture_facts(candidate, source_map, manifest)
        self.assertFalse(allocated["source_fact_allocation"]["complete"])  # type: ignore[index]

    def test_missing_primary_scope_owner_fails_without_positional_fallback(self) -> None:
        manifest = _manifest(20, 2)
        source_map = build_source_map(_nodes(2), manifest, locale="en")
        candidate = _v5_blueprint(source_map)
        candidate["chapters"][0]["lessons"][0]["units"][0]["learning_blocks"][0]["primary_evidence_scope_ids"] = []  # type: ignore[index]
        candidate["chapters"][0]["lessons"][0]["units"][0]["learning_blocks"][0]["supporting_evidence_scope_ids"] = []  # type: ignore[index]
        findings = validate_course_architecture_evidence_scope(candidate, source_map).issues
        self.assertIn("MISSING_PRIMARY_EVIDENCE_SCOPE_OWNER", {item["code"] for item in findings})
        allocated = allocate_source_map_architecture_facts(candidate, source_map, manifest)
        self.assertEqual(allocated["source_fact_allocation"]["allocated_count"], 0)  # type: ignore[index]

    def test_cross_document_scope_claim_fails_without_reassigning_facts(self) -> None:
        nodes = _nodes(2)
        nodes[1] = {**nodes[1], "document_id": "doc-v5-other", "document_name": "Other guide"}
        manifest = _manifest(20, 2)
        for fact in manifest["facts"]:  # type: ignore[index]
            if fact["source_ref"] == "src-002":
                fact["document_id"] = "doc-v5-other"
        source_map = build_source_map(nodes, manifest, locale="en")
        candidate = _v5_blueprint(source_map)
        first_block = candidate["chapters"][0]["lessons"][0]["units"][0]["learning_blocks"][0]  # type: ignore[index]
        other_scope = next(
            scope["id"] for scope in source_map["source_evidence_scopes"]  # type: ignore[index]
            if scope["document_id"] == "doc-v5-other"
        )
        first_block["supporting_evidence_scope_ids"] = [other_scope]
        findings = validate_course_architecture_evidence_scope(candidate, source_map).issues
        self.assertIn("CROSS_DOCUMENT_EVIDENCE_SCOPE_CLAIM", {item["code"] for item in findings})
        allocated = allocate_source_map_architecture_facts(candidate, source_map, manifest)
        self.assertFalse(allocated["source_fact_allocation"]["complete"])  # type: ignore[index]

    def test_1001_fact_allocation_is_deterministic_and_bounded(self) -> None:
        first, source_map, manifest = self._finalize(1_001, 12)
        second = allocate_blueprint_source_fact_ids(
            allocate_source_map_architecture_facts(_v5_blueprint(source_map), source_map, manifest), manifest, _nodes(12),
        )
        self.assertEqual(first["source_fact_allocation"], second["source_fact_allocation"])  # type: ignore[index]
        self.assertEqual(first["source_fact_allocation"]["allocated_count"], 1_001)  # type: ignore[index]
        self.assertTrue(first["source_fact_allocation"]["complete"])  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()
