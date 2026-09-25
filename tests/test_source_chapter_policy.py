from __future__ import annotations

import copy
import unittest
from unittest.mock import AsyncMock, patch

from app.main import (AiUsage, ExtractedSection, build_source_structure_context,
                      generate_validated_lesson_author_blueprint, lesson_author_blueprint)
from app.source_structure import analyze_source_structure, compact_structure
from app.source_chapter_policy import bind_source_chapters, resolve_source_chapter_policy
from app.lesson_author_blueprint import LessonAuthorBlueprintValidationError
from tests.test_lesson_author_blueprint import blueprint_request, valid_blueprint


def document(text: str, identifier: str = "doc-1"):
    return {"document_id": identifier, "document_name": "PRIVATE_SOURCE_NAME",
            "structure": analyze_source_structure([ExtractedSection(text=text, page=1)])}


class SourceChapterPolicyTests(unittest.TestCase):
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

    def test_repeated_headers_and_legacy_numbering_need_review(self):
        doc = {"document_id": "doc-1", "structure": analyze_source_structure([
            ExtractedSection(text="Chapter 1: Repeated header\nBody.", page=1),
            ExtractedSection(text="Chapter 1: Repeated header\nBody.", page=2),
        ])}
        self.assertEqual(resolve_source_chapter_policy([doc])["mode"], "NEEDS_STRUCTURE_REVIEW")
        doc = document("Chapter 1: Foundation\nChapter 2: Application")
        del doc["structure"]["chapter_authority"]
        self.assertIn("SOURCE_HEADING_AUTHORITY_UNVERIFIED", resolve_source_chapter_policy([doc])["reason_codes"])

    def test_mixed_documents_do_not_disable_authority_or_conflate_refs(self):
        docs = [document("CONTENTS\n1. Foundation......3\n2. Practice......5"), document("Unstructured prose.", "doc-2")]
        policy = resolve_source_chapter_policy(docs)
        self.assertEqual(policy["mode"], "NEEDS_STRUCTURE_REVIEW")
        self.assertIn("MULTI_DOCUMENT_CHAPTER_GROUPING_UNRESOLVED", policy["reason_codes"])
        self.assertTrue(all(c["document_id"] == "doc-1" for c in policy["chapters"]))

    def test_outline_overflow_keeps_all_document_inventory_and_fails_closed(self):
        docs = [document("CONTENTS\n1. Foundation......3\n2. Practice......5", f"doc-{i}") for i in range(3)]
        with patch("app.main.MAX_SOURCE_OUTLINE_CHARS", 1):
            context = build_source_structure_context(docs, locale="en")
        self.assertEqual(context["structure_node_count"], 6)
        self.assertEqual({n["document_id"] for n in context["source_structure_nodes"]}, {"doc-0", "doc-1", "doc-2"})
        self.assertFalse(context["source_chapter_policy"]["complete"])
        self.assertIn("SOURCE_OUTLINE_INCOMPLETE", context["source_chapter_policy"]["reason_codes"])

    def test_node_chapter_and_per_document_outline_caps_never_truncate_authority(self):
        doc = document("CONTENTS\n" + "\n".join(f"{i}. Topic {i}......{i}" for i in range(1, 14)))
        self.assertIn("SOURCE_CHAPTER_CAPACITY_EXCEEDED", resolve_source_chapter_policy([doc])["reason_codes"])
        with patch("app.source_structure.MAX_NODES", 2):
            capped = document("CONTENTS\n1. Foundation......3\n2. Practice......5\n3. Extension......8")
        self.assertIn("SOURCE_STRUCTURE_INCOMPLETE", resolve_source_chapter_policy([capped])["reason_codes"])
        long = document("CONTENTS\n" + "\n".join(f"{i}. {'Topic ' * 30}{i}......{i}" for i in range(1, 40)))
        self.assertIn("SOURCE_OUTLINE_INCOMPLETE", build_source_structure_context([long], locale="en")["source_chapter_policy"]["reason_codes"])

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
    async def test_model_title_does_not_require_a_second_provider_call_when_binding_is_exact(self):
        import json
        policy = resolve_source_chapter_policy([document("CONTENTS\nChapter 1: Foundation\n3\nDetails\n4")])
        raw = valid_blueprint()
        raw["chapters"][0]["source_refs"] = ["src-001"]
        raw["chapters"][0]["title"] = "Provider display title"
        with patch("app.main.generate_content", new=AsyncMock(return_value=(json.dumps(raw), AiUsage()))) as provider:
            result, _ = await generate_validated_lesson_author_blueprint(blueprint_request(), "synthetic prompt", source_chapter_policy=policy)
        self.assertEqual(provider.await_count, 1)
        self.assertEqual(result["chapters"][0]["title"], "Foundation")

    async def test_ambiguous_structure_exits_before_architect_and_logs_only_metadata(self):
        from fastapi import HTTPException
        context = build_source_structure_context([document("1. Wash hands\n2. Dry hands")], locale="en")
        with patch("app.main.retrieve_chunks", new=AsyncMock(return_value=([], AiUsage(), context))), \
             patch("app.main.generate_content", new=AsyncMock()) as provider, \
             patch("app.main.logger.info") as log:
            with self.assertRaises(HTTPException) as failure:
                await lesson_author_blueprint(blueprint_request(), pool=None)
        self.assertEqual(failure.exception.status_code, 422)
        self.assertEqual(failure.exception.detail["code"], "SOURCE_STRUCTURE_REVIEW_REQUIRED")
        self.assertEqual(provider.await_count, 0)
        self.assertNotIn("PRIVATE_SOURCE_NAME", str(log.call_args_list))
        self.assertNotIn("Wash hands", str(log.call_args_list))
