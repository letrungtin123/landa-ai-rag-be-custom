from __future__ import annotations

import copy
import json
import unittest
from unittest.mock import AsyncMock, patch

from app.lesson_author_blueprint import LessonAuthorBlueprintValidationError
from app.main import lesson_author_blueprint
from app.schemas.common import AiUsage
from app.services.ingestion.extract import ExtractedSection
from app.services.lesson_author.blueprint import generate_validated_lesson_author_blueprint
from app.services.retrieval.structure import build_source_structure_context
from app.source_chapter_policy import bind_source_chapters, resolve_source_chapter_policy
from app.source_map import build_course_architect_context, build_source_map
from app.source_structure import analyze_source_structure, compact_structure
from tests.test_lesson_author_blueprint import blueprint_request, valid_blueprint


def document(text: str, identifier: str = "doc-1"):
    return {"document_id": identifier, "document_name": "PRIVATE_SOURCE_NAME",
            "structure": analyze_source_structure([ExtractedSection(text=text, page=1)])}


def flat_heading_document():
    """Sanitized shape of the Executive UAT:124 roots,50 numbered hints.

    Not a copy of customer content or a claim of live provider acceptance.
    """
    doc = document("1. Wash hands\n2. Dry hands")
    doc["structure"]["nodes"] = [
        {"source_ref": f"src-{i:03d}", "title": f"Evidence heading {i} " + "topic " * 8,
         "level": 1, "order": i, "parent_source_ref": None,
         "number_label": str(i) if i <= 50 else None, "page": (i - 1) // 12 + 1}
        for i in range(1, 125)
    ]
    return doc


class SourceChapterPolicyTests(unittest.TestCase):
    def test_page_less_chapter_table_is_bound_to_body_not_learning_outcomes(self):
        for label, marker in (("Chương", "MỤC LỤC & ĐỊNH VỊ CHƯƠNG TRÌNH"),
                              ("Chapter", "Course structure (Table of Contents)")):
            toc = marker + "\n" + "\n".join(f"{i}. Outcome {i}" for i in range(1, 7))
            toc += "\nCHAPTER / TITLE / OUTCOME\n" + "\n".join(f"{label} {i:02d}\nTopic {i}\nExpected result {i}" for i in range(1, 8))
            sections = [ExtractedSection(text=toc, page=2)]
            for i in range(1, 8):
                sections.append(ExtractedSection(text=f"{label.upper()} {i:02d}\nTopic {i}\nBody.", page=i + 2))
            structure = analyze_source_structure(sections)
            policy = resolve_source_chapter_policy([{"document_id": "doc-1", "structure": structure}])
            self.assertEqual(policy["mode"], "SOURCE_LOCKED_TOC")
            self.assertEqual(len(policy["chapters"]), 7)
            self.assertEqual([n["logical_page"] for n in structure["nodes"]], list(range(3, 10)))
            self.assertEqual(structure["chapter_authority"]["basis"], "CHAPTER_TABLE_BODY_LABEL_MATCH")
            # Multiple same-page chunks and continuation pages must not add chapters.
            split = [ExtractedSection(text=marker, page=2), ExtractedSection(text=toc[len(marker):], page=2), *sections[1:]]
            split.append(ExtractedSection(text=f"{label} 07\nTopic 7 (continued)", page=10))
            self.assertEqual(analyze_source_structure(split)["nodes"], structure["nodes"])

    def test_page_less_table_missing_reordered_or_duplicate_body_is_not_model_freedom(self):
        toc = ExtractedSection(text="MỤC LỤC & TỔNG QUAN\nChương 01\nTopic A\nChương 02\nTopic B", page=1)
        for labels in ([1], [2, 1], [1, 2, 1], [1, 2, 3], []):
            structure = analyze_source_structure([toc, *[
                ExtractedSection(text=f"Chương {n:02d}\nTopic {n}", page=i + 2) for i, n in enumerate(labels)]])
            policy = resolve_source_chapter_policy([{"document_id": "doc-1", "structure": structure}])
            self.assertEqual(policy["mode"], "NEEDS_STRUCTURE_REVIEW", labels)
            self.assertFalse(policy["complete"])
            self.assertIn("SOURCE_CHAPTER_TABLE_BODY_MISMATCH", structure["warnings"])

    def test_six_toc_roots_keep_children_and_exact_bindings(self):
        doc = document("CONTENTS\n" + "\n".join(
            f"{i}. Topic {i}........ {i * 3}\n{i}.1. Details {i}........ {i * 3 + 1}" for i in range(1, 7)))
        policy = resolve_source_chapter_policy([doc])
        self.assertEqual(policy["mode"], "SOURCE_LOCKED_TOC")
        self.assertEqual(len(policy["chapters"]), 6)
        candidate = {"chapters": [{"title": "Provider display title", "source_refs": [c["source_ref"]], "lessons": []} for c in policy["chapters"]]}
        result = bind_source_chapters(candidate, policy)
        self.assertEqual([c["title"] for c in result["chapters"]], [f"Topic {i}" for i in range(1, 7)])
        self.assertEqual(candidate["chapters"][0]["title"], "Provider display title")
        for changed in [list(reversed(candidate["chapters"])), candidate["chapters"][:-1], candidate["chapters"] * 2]:
            with self.assertRaises(LessonAuthorBlueprintValidationError):
                bind_source_chapters({"chapters": changed}, policy)
        wrong = copy.deepcopy(candidate)
        wrong["chapters"][0]["source_refs"] = ["src-unknown"]
        with self.assertRaises(LessonAuthorBlueprintValidationError):
            bind_source_chapters(wrong, policy)
        wrong["chapters"][0]["source_refs"] = wrong["chapters"][1]["source_refs"]
        with self.assertRaises(LessonAuthorBlueprintValidationError):
            bind_source_chapters(wrong, policy)

    def test_numbered_tree_and_explicit_chapters_but_not_body_lists(self):
        for text in ["1. Foundation\n1.1. Definition\nBody.\n2. Application\n2.1. Procedure\nBody.",
                     "Chapter 1: Foundation\nBody.\nChapter 2: Application\nBody."]:
            doc = document(text)
            self.assertEqual(resolve_source_chapter_policy([doc])["mode"], "SOURCE_LOCKED_HEADINGS")
            self.assertEqual(compact_structure(doc["structure"])["chapter_authority"], doc["structure"]["chapter_authority"])
        for text in ["1. Wash hands\n2. Dry hands", "BODY SLIDE TITLE\nSome prose.", "Unstructured prose with no heading."]:
            policy = resolve_source_chapter_policy([document(text)])
            self.assertNotIn(policy["mode"], {"SOURCE_LOCKED_HEADINGS", "SOURCE_LOCKED_TOC"})
        self.assertEqual(resolve_source_chapter_policy([document("Unstructured prose.")])["mode"], "MODEL_DESIGNED")

    def test_current_parser_non_authority_is_hint_but_legacy_numbering_needs_review(self):
        doc = {"document_id": "doc-1", "structure": analyze_source_structure([
            ExtractedSection(text="Chapter 1: Repeated header\nBody.", page=1),
            ExtractedSection(text="Chapter 1: Repeated header\nBody.", page=2),
        ])}
        self.assertEqual(resolve_source_chapter_policy([doc])["mode"], "MODEL_DESIGNED")
        doc = document("Chapter 1: Foundation\nChapter 2: Application")
        del doc["structure"]["chapter_authority"]
        self.assertIn("SOURCE_HEADING_AUTHORITY_UNVERIFIED", resolve_source_chapter_policy([doc])["reason_codes"])

    def test_mixed_documents_do_not_disable_authority_or_conflate_refs(self):
        docs = [document("CONTENTS\n1. Foundation......3\n2. Practice......5"), document("Unstructured prose.", "doc-2")]
        policy = resolve_source_chapter_policy(docs)
        self.assertEqual(policy["mode"], "NEEDS_STRUCTURE_REVIEW")
        self.assertIn("MULTI_DOCUMENT_CHAPTER_GROUPING_UNRESOLVED", policy["reason_codes"])
        self.assertTrue(all(c["document_id"] == "doc-1" for c in policy["chapters"]))

    def test_display_overflow_preserves_inventory_and_multi_document_guard(self):
        docs = [document("CONTENTS\n1. Foundation......3\n2. Practice......5", f"doc-{i}") for i in range(3)]
        with patch("app.services.retrieval.structure.MAX_SOURCE_OUTLINE_CHARS", 1):
            context = build_source_structure_context(docs, locale="en")
        self.assertEqual(context["structure_node_count"], 6)
        self.assertEqual({n["document_id"] for n in context["source_structure_nodes"]}, {"doc-0", "doc-1", "doc-2"})
        self.assertFalse(context["source_chapter_policy"]["complete"])
        self.assertTrue(context["source_outline_display_truncated"])
        self.assertEqual(context["source_chapter_policy"]["reason_codes"], ["MULTI_DOCUMENT_CHAPTER_GROUPING_UNRESOLVED"])

    def test_node_chapter_and_per_document_outline_caps_never_truncate_authority(self):
        doc = document("CONTENTS\n" + "\n".join(f"{i}. Topic {i}......{i}" for i in range(1, 14)))
        self.assertIn("SOURCE_CHAPTER_CAPACITY_EXCEEDED", resolve_source_chapter_policy([doc])["reason_codes"])
        with patch("app.source_structure.MAX_NODES", 2):
            capped = document("CONTENTS\n1. Foundation......3\n2. Practice......5\n3. Extension......8")
        self.assertIn("SOURCE_STRUCTURE_INCOMPLETE", resolve_source_chapter_policy([capped])["reason_codes"])
        long = document("CONTENTS\n" + "\n".join(f"{i}. {'Topic ' * 30}{i}......{i}" for i in range(1, 40)))
        context = build_source_structure_context([long], locale="en")
        self.assertTrue(context["source_outline_display_truncated"])
        self.assertIn("SOURCE_CHAPTER_CAPACITY_EXCEEDED", context["source_chapter_policy"]["reason_codes"])
        self.assertEqual(context["structure_node_count"], len(long["structure"]["nodes"]))

    def test_executive_shape_preserves_124_hints_without_locking_124_chapters(self):
        doc = flat_heading_document()
        original = copy.deepcopy(doc)
        for locale in ("en", "vi"):
            context = build_source_structure_context([doc], locale=locale)
            self.assertTrue(context["source_outline_display_truncated"])
            self.assertIn("SOURCE_OUTLINE_TRUNCATED", context["source_structure_warnings"])
            self.assertEqual(context["structure_node_count"], 124)
            self.assertEqual(len(context["source_structure_nodes"]), 124)
            self.assertEqual(len(context["known_source_refs"]), 124)
            self.assertEqual(context["source_chapter_policy"], {
                "version": 1, "mode": "MODEL_DESIGNED", "complete": True,
                "reason_codes": [], "chapters": [],
            })
            self.assertEqual(doc, original)

    def test_hse_six_verified_chapters_stay_locked_when_display_overflows(self):
        doc = document("CONTENTS\n" + "\n".join(
            f"{i}. Topic {i}......{i * 3}\n{i}.1. Detail {i}......{i * 3 + 1}" for i in range(1, 7)))
        expected = resolve_source_chapter_policy([doc])
        with patch("app.services.retrieval.structure.MAX_SOURCE_OUTLINE_CHARS", 1):
            context = build_source_structure_context([doc], locale="vi")
        self.assertTrue(context["source_outline_display_truncated"])
        self.assertEqual(context["source_chapter_policy"], expected)
        self.assertEqual(len(expected["chapters"]), 6)
        self.assertEqual(len(context["source_structure_nodes"]), 12)

    def test_numbered_hints_require_explicit_current_complete_no_authority_proof(self):
        for change in ({"complete": False}, {"complete": "true"}, {"basis": "UNKNOWN"},
                       {"heading_refs": None}, {"version": 2}, {"version": True}):
            with self.subTest(change=change):
                doc = document("1. Wash hands\n2. Dry hands")
                doc["structure"]["chapter_authority"].update(change)
                self.assertFalse(resolve_source_chapter_policy([doc])["complete"])
        doc = document("1. Wash hands\n2. Dry hands")
        doc["structure"]["parser_version"] = "source-structure-v1"
        self.assertFalse(resolve_source_chapter_policy([doc])["complete"])

    def test_columnar_context_is_lossless_bounded_and_does_not_change_source_map(self):
        from app.source_map import _compact_global_hierarchy

        context = build_source_structure_context([flat_heading_document()], locale="en")
        manifest = {"truncated": False, "facts": [
            {"fact_id": "f1", "document_id": "doc-1", "source_ref": "src-001",
             "source_page": 1, "source_chunk": 0, "text": "A source-supported instruction."}]}
        source_map = build_source_map(context["source_structure_nodes"], manifest, locale="en")
        original_map, original_manifest = copy.deepcopy(source_map), copy.deepcopy(manifest)
        original_hierarchy = _compact_global_hierarchy(source_map)
        encoded = build_course_architect_context(source_map, manifest)
        self.assertTrue(encoded["context_complete"])
        self.assertEqual(encoded["diagnostics"]["architect_hierarchy_encoding"], "columnar-v1")
        self.assertLessEqual(len(encoded["context"]), 48_000)
        decoded = json.loads(encoded["context"])
        for key in ("sections", "concepts"):
            table = decoded[key]
            restored = [dict(zip(table["columns"], row)) for row in table["rows"]]
            self.assertEqual(restored, original_hierarchy[key])
        self.assertEqual(decoded["documents"], original_hierarchy["documents"])
        self.assertEqual(decoded["coverage"], original_hierarchy["coverage"])
        self.assertEqual(len(decoded["source_evidence_scopes"]), len(source_map["source_evidence_scopes"]))
        self.assertTrue(all(scope["e"] for scope in decoded["source_evidence_scopes"]))
        self.assertEqual(source_map, original_map)
        self.assertEqual(manifest, original_manifest)
        self.assertEqual(build_course_architect_context(source_map, manifest), encoded)
        self.assertFalse(build_course_architect_context(source_map, manifest, max_chars=100)["context_complete"])

    def test_small_and_legacy_hierarchies_keep_object_encoding(self):
        source_map = build_source_map([], {"facts": []}, locale="en")
        compact = build_course_architect_context(source_map, {"facts": []})
        self.assertEqual(compact["diagnostics"]["architect_hierarchy_encoding"], "objects")
        source_map["version"] = "source-map-v1"
        legacy = build_course_architect_context(source_map, {"facts": []}, max_chars=1)
        self.assertFalse(legacy["context_complete"])
        self.assertEqual(legacy["diagnostics"]["architect_hierarchy_encoding"], "objects")

    def test_display_overflow_does_not_bypass_real_inventory_failures(self):
        for warning in ("SOURCE_STRUCTURE_CAPACITY_REACHED", "SOURCE_STRUCTURE_REBUILD_INCOMPLETE"):
            doc = flat_heading_document()
            doc["structure"]["warnings"].append(warning)
            context = build_source_structure_context([doc], locale="en")
            self.assertTrue(context["source_outline_display_truncated"])
            self.assertIn("SOURCE_STRUCTURE_INCOMPLETE", context["source_chapter_policy"]["reason_codes"])
        for kind in ("duplicate_ref", "invalid_parent", "mismatched_authority"):
            doc = flat_heading_document()
            if kind == "duplicate_ref":
                doc["structure"]["nodes"][-1]["source_ref"] = "src-001"
            elif kind == "invalid_parent":
                doc["structure"]["nodes"][-1]["parent_source_ref"] = "absent"
            else:
                doc["structure"]["chapter_authority"]["heading_refs"] = ["absent"]
            self.assertFalse(build_source_structure_context([doc], locale="en")["source_chapter_policy"]["complete"])

    def test_display_truncation_keeps_global_evidence_and_context_budget_strict(self):
        # The complete server inventory, not the display outline, feeds Source Map.
        doc = document("1. Wash hands\n2. Dry hands")
        with patch("app.services.retrieval.structure.MAX_SOURCE_OUTLINE_CHARS", 1):
            context = build_source_structure_context([doc], locale="en")
        manifest = {"truncated": False, "fact_scope_complete": True, "total_fact_count": 2,
                    "represented_fact_count": 2, "facts": [
                        {"fact_id": f"f{i}", "document_id": "doc-1", "source_ref": f"src-00{i}",
                         "source_page": 1, "source_chunk": 0, "text": f"Instruction {i}."}
                        for i in (1, 2)]}
        source_map = build_source_map(context["source_structure_nodes"], manifest, locale="en")
        self.assertTrue(source_map["coverage"]["fact_scope_complete"])
        self.assertTrue(build_course_architect_context(source_map, manifest)["context_complete"])
        too_small = build_course_architect_context(source_map, manifest, max_chars=1)
        self.assertFalse(too_small["context_complete"])
        self.assertEqual(too_small["error_code"], "EVIDENCE_SCOPE_CONTEXT_INCOMPLETE")

    def test_primary_evidence_cannot_cross_chapter_but_supporting_may_reference(self):
        policy = resolve_source_chapter_policy([document("CONTENTS\n1. Foundation......3\n2. Practice......5")])
        source_map = {"sections": [{"id": f"s{i}", "source_ref": f"src-00{i}", "document_id": "doc-1", "parent_id": None} for i in [1, 2]],
                      "source_evidence_scopes": [{"id": f"e{i}", "section_id": f"s{i}"} for i in [1, 2]]}
        candidate = {"chapters": [{"title": "Display", "source_refs": [f"src-00{i}"], "lessons": [{"units": [{"learning_blocks": [
            {"primary_evidence_scope_ids": [f"e{i}"], "supporting_evidence_scope_ids": []}]}]}]} for i in [1, 2]]}
        bind_source_chapters(candidate, policy, source_map)
        # Read-only reinforcement is not primary ownership; existing semantic
        # validators still judge whether this support is instructionally valid.
        candidate["chapters"][1]["lessons"][0]["units"][0]["learning_blocks"][0]["supporting_evidence_scope_ids"] = ["e1"]
        bind_source_chapters(candidate, policy, source_map)
        candidate["chapters"][0]["lessons"][0]["units"][0]["learning_blocks"][0]["primary_evidence_scope_ids"] = ["e2"]
        with self.assertRaises(LessonAuthorBlueprintValidationError) as rejected:
            bind_source_chapters(candidate, policy, source_map)
        self.assertEqual(rejected.exception.constraint, "SOURCE_CHAPTER_PRIMARY_SCOPE")

    def test_incomplete_parent_or_unverified_toc_never_becomes_model_designed(self):
        doc = document("CONTENTS\n1. Foundation......3\n1.1. Detail......4\n2. Practice......5")
        doc["structure"]["nodes"][1]["parent_source_ref"] = "missing"
        self.assertIn("SOURCE_STRUCTURE_PARENT_INVALID", resolve_source_chapter_policy([doc])["reason_codes"])
        doc["structure"]["nodes"][1]["parent_source_ref"] = doc["structure"]["nodes"][1]["source_ref"]
        self.assertIn("SOURCE_STRUCTURE_PARENT_INVALID", resolve_source_chapter_policy([doc])["reason_codes"])
        doc = document("CONTENTS\n1. Foundation......3\n2. Practice......5")
        doc["structure"]["confidence"] = 0.3
        self.assertIn("SOURCE_TOC_AUTHORITY_UNVERIFIED", resolve_source_chapter_policy([doc])["reason_codes"])


class SourceChapterPolicyIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_display_truncated_hints_reach_architect_with_full_inventory(self):
        from fastapi import HTTPException

        from app.workflows.contracts import WorkflowFailure

        context = build_source_structure_context([flat_heading_document()], locale="en")
        context["source_coverage_manifest"] = {
            "truncated": False, "fact_scope_complete": True, "total_fact_count": 2,
            "represented_fact_count": 2, "facts": [
                {"fact_id": f"f{i}", "document_id": "doc-1", "source_ref": f"src-{i:03d}",
                 "source_page": 1, "source_chunk": 0, "text": f"Instruction {i}."}
                for i in (1, 124)],
        }
        # Stop at the Architect boundary with a mocked error: no paid call,
        # and no claim that a generated Blueprint passed downstream validators.
        with patch("app.services.retrieval.search.retrieve_chunks", new=AsyncMock(return_value=([], AiUsage(), context))), \
             patch("app.services.lesson_author.blueprint.generate_validated_lesson_author_blueprint", new=AsyncMock(
                 side_effect=WorkflowFailure("PROVIDER_ERROR", "Mocked Architect boundary"))) as architect, \
             patch("app.services.provider.generate_content", new=AsyncMock()) as provider, \
             patch("app.main.logger.info") as log:
            with self.assertRaises(HTTPException) as failure:
                await lesson_author_blueprint(blueprint_request(), pool=None)
        self.assertEqual(failure.exception.detail["code"], "PROVIDER_ERROR")
        self.assertEqual(architect.await_count, 1)
        self.assertEqual(provider.await_count, 0)
        supplied = architect.await_args.kwargs
        self.assertEqual(supplied["source_chapter_policy"]["mode"], "MODEL_DESIGNED")
        self.assertEqual(len(supplied["source_structure_nodes"]), 124)
        self.assertEqual(supplied["v5_immutable_source_context"].source_map_copy()["coverage"]["represented_fact_count"], 2)
        logs = str(log.call_args_list)
        self.assertIn('"source_outline_display_truncated": true', logs)
        self.assertIn('"source_structure_node_count": 124', logs)
        self.assertNotIn("PRIVATE_SOURCE_NAME", logs)
        self.assertNotIn("Evidence heading", logs)

    async def test_actual_source_truncation_still_stops_before_architect(self):
        from fastapi import HTTPException

        context = build_source_structure_context([flat_heading_document()], locale="en")
        context["course_blueprint_source_scope_truncated"] = True
        with patch("app.services.retrieval.search.retrieve_chunks", new=AsyncMock(return_value=([], AiUsage(), context))), \
             patch("app.services.lesson_author.blueprint.generate_validated_lesson_author_blueprint", new=AsyncMock()) as architect:
            with self.assertRaises(HTTPException) as failure:
                await lesson_author_blueprint(blueprint_request(), pool=None)
        self.assertEqual(failure.exception.detail["code"], "SOURCE_SCOPE_INCOMPLETE")
        self.assertEqual(architect.await_count, 0)

    async def test_model_title_does_not_require_a_second_provider_call_when_binding_is_exact(self):
        import json
        policy = resolve_source_chapter_policy([document("CONTENTS\nChapter 1: Foundation\n3\nDetails\n4")])
        raw = valid_blueprint()
        raw["chapters"][0]["source_refs"] = ["src-001"]
        raw["chapters"][0]["title"] = "Provider display title"
        with patch("app.services.provider.generate_content", new=AsyncMock(return_value=(json.dumps(raw), AiUsage()))) as provider:
            result, _ = await generate_validated_lesson_author_blueprint(blueprint_request(), "synthetic prompt", source_chapter_policy=policy)
        self.assertEqual(provider.await_count, 1)
        self.assertEqual(result["chapters"][0]["title"], "Foundation")

    async def test_missing_legacy_proof_exits_before_architect_and_logs_only_metadata(self):
        from fastapi import HTTPException
        doc = document("1. Wash hands\n2. Dry hands")
        del doc["structure"]["chapter_authority"]
        context = build_source_structure_context([doc], locale="en")
        with patch("app.services.retrieval.search.retrieve_chunks", new=AsyncMock(return_value=([], AiUsage(), context))), \
             patch("app.services.provider.generate_content", new=AsyncMock()) as provider, \
             patch("app.main.logger.info") as log:
            with self.assertRaises(HTTPException) as failure:
                await lesson_author_blueprint(blueprint_request(), pool=None)
        self.assertEqual(failure.exception.status_code, 422)
        self.assertEqual(failure.exception.detail["code"], "SOURCE_STRUCTURE_REVIEW_REQUIRED")
        self.assertEqual(provider.await_count, 0)
        self.assertNotIn("PRIVATE_SOURCE_NAME", str(log.call_args_list))
        self.assertNotIn("Wash hands", str(log.call_args_list))
