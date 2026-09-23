from __future__ import annotations

import copy
import unittest

from app.workflows.contracts import WorkflowGenerationResult, WorkflowValidationResult
from app.workflows.course_architecture import (
    CourseArchitectureWorkflowCallbacks,
    classify_course_repair_targets,
    run_course_architecture_workflow,
)
from app.workflows.lesson_generation import (
    LessonGenerationWorkflowCallbacks,
    classify_lesson_repair_targets,
    run_lesson_generation_workflow,
)
from app.workflows.contracts import WorkflowFailure
from app.main import (
    apply_course_architecture_repair_patches,
    apply_lesson_generation_repair_patches,
    build_course_architecture_repair_prompt,
    parse_course_architecture_repair_payload,
)


def issue(code: str, *, path: str = "chapter_1.lesson_1", severity: str = "error", related_paths: list[str] | None = None) -> dict[str, object]:
    value: dict[str, object] = {"code": code, "severity": severity, "message": code, "path": path}
    if related_paths:
        value["related_paths"] = related_paths
    return value


class CourseArchitectureWorkflowTests(unittest.IsolatedAsyncioTestCase):
    def callbacks(self, *, validation_sequence: list[list[dict[str, object]]], repairs: list[dict[str, object]] | None = None, source_issues: list[dict[str, object]] | None = None, repair_failure: bool = False):
        validation_calls = 0
        repair_calls: list[list[dict[str, object]]] = []

        async def architect(source_map):
            self.assertEqual(source_map["version"], "source-map-v1")
            return WorkflowGenerationResult({"architecture_contract_version": 3, "chapters": [{"lessons": [{"units": [{}]}]}]})

        def validate(blueprint, source_map):
            nonlocal validation_calls
            current = validation_sequence[min(validation_calls, len(validation_sequence) - 1)]
            validation_calls += 1
            return WorkflowValidationResult(current, {"source_coverage": 1.0, "concept_coverage": 1.0})

        async def repair(blueprint, targets, source_map):
            repair_calls.append(targets)
            if repair_failure:
                raise WorkflowFailure("ARCHITECTURE_REPAIR_INVALID", "invalid patch")
            replacement = (repairs or [{"architecture_contract_version": 3, "chapters": [{"lessons": [{"units": [{}]}]}]}])[min(len(repair_calls) - 1, len(repairs or [{}]) - 1)]
            return WorkflowGenerationResult(replacement)

        callbacks = CourseArchitectureWorkflowCallbacks(
            validate_source_scope=lambda: source_issues or [],
            build_source_map=lambda: {"version": "source-map-v1", "coverage": {"fact_scope_complete": True}},
            architect=architect,
            validate_blueprint=validate,
            repair_blueprint=repair,
        )
        return callbacks, repair_calls, lambda: validation_calls

    async def test_valid_blueprint_bypasses_repair(self):
        callbacks, repairs, _ = self.callbacks(validation_sequence=[[]])
        blueprint, diagnostics = await run_course_architecture_workflow(callbacks, request_context={}, max_repair_attempts=2)
        self.assertEqual(blueprint["architecture_contract_version"], 3)
        self.assertEqual(repairs, [])
        self.assertEqual(diagnostics["status"], "ready")
        self.assertIn("BLUEPRINT_READY", [event["code"] for event in diagnostics["progress"]])

    async def test_repairable_lesson_failure_repairs_only_affected_lesson_and_revalidates(self):
        callbacks, repairs, validation_calls = self.callbacks(validation_sequence=[
            [issue("LESSON_TOO_THIN", path="chapter_2.lesson_3")],
            [],
        ])
        _blueprint, diagnostics = await run_course_architecture_workflow(callbacks, request_context={}, max_repair_attempts=2)
        self.assertEqual(len(repairs), 1)
        self.assertEqual(repairs[0][0]["scope"], "lesson")
        self.assertEqual(repairs[0][0]["path"], "chapter_2.lesson_3")
        self.assertEqual(validation_calls(), 2)
        self.assertEqual(diagnostics["repair_count"], 1)

    async def test_non_repairable_source_scope_fails_without_architect_or_repair(self):
        callbacks, repairs, _ = self.callbacks(
            validation_sequence=[[]],
            source_issues=[issue("SOURCE_SCOPE_INCOMPLETE", path="course")],
        )
        with self.assertRaisesRegex(Exception, "Course architecture workflow") as raised:
            await run_course_architecture_workflow(callbacks, request_context={}, max_repair_attempts=2)
        self.assertEqual(getattr(raised.exception, "code"), "SOURCE_SCOPE_INCOMPLETE")
        self.assertEqual(repairs, [])

    async def test_non_repairable_architect_semantic_scope_preserves_safe_findings_without_scoped_repair(self):
        repair_calls = 0
        semantic_issue = issue(
            "AMBIGUOUS_INSTRUCTIONAL_SCOPE",
            path="chapter_1.lesson_1.unit_1.block_1",
            related_paths=[
                "chapter_1.lesson_1.unit_1.block_1",
                "chapter_1.lesson_2.unit_1.block_1",
            ],
        )
        semantic_issue.update({
            "concept_id": "concept_1",
            "section_id": "src_section_1",
            "expected_primary_owner_count": 1,
            "actual_primary_owner_count": 2,
            "ownership_level": "learning_block",
            "scope_classification": "concept_primary_ownership",
            "coverage_state": "ambiguous",
            "safe_reason": "MULTIPLE_PRIMARY_DESTINATIONS",
        })

        async def architect(_source_map):
            raise WorkflowFailure(
                "ARCHITECTURE_SCOPE_INCOMPLETE",
                "safe semantic scope failure",
                issues=[semantic_issue],
                internal_code="ARCH_SEMANTIC_SCOPE_ATTEMPTS_EXHAUSTED",
                failure_stage="course_architect_semantic_scope_validation",
            )

        async def repair(_blueprint, _targets, _source_map):
            nonlocal repair_calls
            repair_calls += 1
            raise AssertionError("semantic prerequisites must not enter scoped repair")

        callbacks = CourseArchitectureWorkflowCallbacks(
            validate_source_scope=lambda: [],
            build_source_map=lambda: {"version": "source-map-v1"},
            architect=architect,
            validate_blueprint=lambda _blueprint, _source_map: WorkflowValidationResult(),
            repair_blueprint=repair,
        )

        with self.assertRaises(WorkflowFailure) as raised:
            await run_course_architecture_workflow(
                callbacks,
                request_context={},
                max_repair_attempts=2,
            )

        self.assertEqual(repair_calls, 0)
        self.assertEqual(raised.exception.failure_stage, "course_architect_semantic_scope_validation")
        self.assertEqual(raised.exception.issues[0]["concept_id"], "concept_1")
        self.assertEqual(raised.exception.issues[0]["related_paths"], semantic_issue["related_paths"])

    async def test_unchanged_repair_fails_closed_without_a_second_provider_pass(self):
        callbacks, repairs, _ = self.callbacks(validation_sequence=[[issue("LESSON_TOO_THIN")]])
        with self.assertRaisesRegex(Exception, "Course architecture workflow") as raised:
            await run_course_architecture_workflow(callbacks, request_context={}, max_repair_attempts=2)
        self.assertEqual(getattr(raised.exception, "code"), "ARCHITECTURE_REPAIR_NO_PROGRESS")
        self.assertEqual(getattr(raised.exception, "internal_code"), "ARCH_REPAIR_NO_PROGRESS")
        self.assertEqual(getattr(raised.exception, "failure_stage"), "blueprint_revalidation")
        self.assertEqual(len(repairs), 1)

    async def test_repair_provider_failure_fails_without_a_second_validation_loop(self):
        callbacks, repairs, validation_calls = self.callbacks(
            validation_sequence=[[issue("LESSON_TOO_THIN")]],
            repair_failure=True,
        )
        with self.assertRaisesRegex(Exception, "Course architecture workflow") as raised:
            await run_course_architecture_workflow(callbacks, request_context={}, max_repair_attempts=2)
        self.assertEqual(getattr(raised.exception, "code"), "ARCHITECTURE_REPAIR_INVALID")
        self.assertEqual(len(repairs), 1)
        self.assertEqual(validation_calls(), 1)

    async def test_emits_one_node_owned_correlation_id_for_every_graph_stage(self):
        events: list[dict[str, object]] = []

        async def architect(_source_map):
            return WorkflowGenerationResult({"architecture_contract_version": 3, "chapters": [{"lessons": [{"units": [{}]}]}]})

        async def repair(blueprint, _targets, _source_map):
            return WorkflowGenerationResult(blueprint)

        validation_calls = 0
        def validate(_blueprint, _source_map):
            nonlocal validation_calls
            validation_calls += 1
            return WorkflowValidationResult([issue("LESSON_TOO_THIN")] if validation_calls == 1 else [])

        correlation_id = "f7e48b51-01d1-4fd7-a578-5d7e749f70e9"
        callbacks = CourseArchitectureWorkflowCallbacks(
            validate_source_scope=lambda: [],
            build_source_map=lambda: {"version": "source-map-v1"},
            architect=architect,
            validate_blueprint=validate,
            repair_blueprint=repair,
            emit_diagnostic=events.append,
        )
        await run_course_architecture_workflow(
            callbacks,
            request_context={"correlation_id": correlation_id},
            max_repair_attempts=2,
        )
        self.assertGreater(len(events), 4)
        self.assertTrue(all(event.get("correlation_id") == correlation_id for event in events))
        self.assertIn("blueprint_revalidation", {event.get("stage") for event in events})

    def test_duplicate_objectives_target_only_the_two_lessons(self):
        targets = classify_course_repair_targets([
            issue("DUPLICATE_OBJECTIVE", path="chapter_1.lesson_2", related_paths=["chapter_3.lesson_1"]),
        ])
        self.assertEqual({target["path"] for target in targets}, {"chapter_1.lesson_2", "chapter_3.lesson_1"})
        self.assertTrue(all(target["scope"] == "lesson" for target in targets))

    async def _assert_v5_same_layer_no_progress(
        self,
        *,
        code: str,
        path: str,
        repair_layer: str,
        **metadata: object,
    ) -> None:
        """A same-layer/same-target failure gets exactly one provider repair."""

        provider_repairs = 0
        finding = issue(code, path=path)
        finding.update({"repairable": True, "repair_layer": repair_layer, **metadata})

        async def architect(_source_map):
            return WorkflowGenerationResult({"architecture_contract_version": 5, "chapters": []})

        def validate(_blueprint, _source_map):
            # The provider returns the unchanged candidate, so the same
            # deterministic target/finding must fail closed on revalidation.
            return WorkflowValidationResult([copy.deepcopy(finding)])

        async def repair(blueprint, _targets, _source_map):
            nonlocal provider_repairs
            provider_repairs += 1
            return WorkflowGenerationResult(copy.deepcopy(blueprint))

        callbacks = CourseArchitectureWorkflowCallbacks(
            validate_source_scope=lambda: [],
            build_source_map=lambda: {"version": "source-map-v1", "coverage": {"fact_scope_complete": True}},
            architect=architect,
            validate_blueprint=validate,
            repair_blueprint=repair,
        )
        with self.assertRaises(WorkflowFailure) as raised:
            await run_course_architecture_workflow(callbacks, request_context={}, max_repair_attempts=2)
        self.assertEqual(raised.exception.code, "ARCHITECTURE_REPAIR_NO_PROGRESS")
        self.assertEqual(raised.exception.internal_code, "ARCH_REPAIR_NO_PROGRESS")
        self.assertEqual(provider_repairs, 1)

    async def test_v5_schema_no_progress_fails_closed_after_one_local_repair(self):
        await self._assert_v5_same_layer_no_progress(
            code="BLUEPRINT_INVALID_SCHEMA",
            path="chapter_1.lesson_1",
            repair_layer="SCHEMA",
            constraint="ARRAY_LENGTH",
            schema_path="chapters[0].lessons[0].units",
        )

    async def test_v5_evidence_semantic_no_progress_fails_closed_after_one_local_repair(self):
        await self._assert_v5_same_layer_no_progress(
            code="EVIDENCE_SCOPE_CONCEPT_MISMATCH",
            path="chapter_1.lesson_1.unit_1",
            repair_layer="EVIDENCE_SEMANTIC",
        )

    async def test_v5_instructional_coherence_no_progress_fails_closed_after_one_local_repair(self):
        await self._assert_v5_same_layer_no_progress(
            code="ACTION_OBJECTIVE_INSTRUCTION_MISMATCH",
            path="chapter_1.lesson_1.unit_1",
            repair_layer="PRE_ALLOCATION_COHERENCE",
        )

    async def test_v5_hard_global_cap_prevents_a_fourth_provider_repair(self):
        provider_repairs = 0
        events: list[dict[str, object]] = []
        validation_index = 0
        sequence = [
            ("BLUEPRINT_INVALID_SCHEMA", "chapter_1.lesson_1", "SCHEMA", {
                "constraint": "ARRAY_LENGTH", "schema_path": "chapters[0].lessons[0].units",
            }),
            ("EVIDENCE_SCOPE_CONCEPT_MISMATCH", "chapter_2.lesson_1.unit_1", "EVIDENCE_SEMANTIC", {}),
            ("ACTION_OBJECTIVE_INSTRUCTION_MISMATCH", "chapter_3.lesson_1.unit_1", "PRE_ALLOCATION_COHERENCE", {}),
            # A new SCHEMA target would otherwise ask for a fourth provider
            # call. V5's three-call cap and its per-layer bound both block it.
            ("BLUEPRINT_INVALID_SCHEMA", "chapter_4.lesson_1", "SCHEMA", {
                "constraint": "ARRAY_LENGTH", "schema_path": "chapters[3].lessons[0].units",
            }),
        ]

        async def architect(_source_map):
            return WorkflowGenerationResult({"architecture_contract_version": 5, "chapters": []})

        def validate(_blueprint, _source_map):
            nonlocal validation_index
            code, path, layer, metadata = sequence[min(validation_index, len(sequence) - 1)]
            validation_index += 1
            finding = issue(code, path=path)
            finding.update({"repairable": True, "repair_layer": layer, **metadata})
            return WorkflowValidationResult([finding])

        async def repair(blueprint, _targets, _source_map):
            nonlocal provider_repairs
            provider_repairs += 1
            return WorkflowGenerationResult(copy.deepcopy(blueprint))

        callbacks = CourseArchitectureWorkflowCallbacks(
            validate_source_scope=lambda: [],
            build_source_map=lambda: {"version": "source-map-v1", "coverage": {"fact_scope_complete": True}},
            architect=architect,
            validate_blueprint=validate,
            repair_blueprint=repair,
            emit_diagnostic=events.append,
        )
        with self.assertRaises(WorkflowFailure) as raised:
            await run_course_architecture_workflow(callbacks, request_context={}, max_repair_attempts=2)
        self.assertEqual(raised.exception.code, "ARCHITECTURE_REPAIR_EXHAUSTED")
        self.assertEqual(provider_repairs, 3)
        final_event = [event for event in events if event.get("stage") == "response_finalize"][-1]
        self.assertEqual(final_event["total_repair_provider_calls"], 3)


class LessonGenerationWorkflowTests(unittest.IsolatedAsyncioTestCase):
    def callbacks(self, *, content_sequence: list[list[dict[str, object]]], evidence_issues: list[dict[str, object]] | None = None, pedagogy_sequence: list[list[dict[str, object]]] | None = None, duplicate_sequence: list[list[dict[str, object]]] | None = None, diagnostic_events: list[dict[str, object]] | None = None):
        content_calls = 0
        pedagogy_calls = 0
        duplicate_calls = 0
        repairs: list[list[dict[str, object]]] = []

        async def retrieve():
            return {"source_document_count": 1, "retrieved_count": 4, "allowed_source_scope": True}

        def validate_content(proposal):
            nonlocal content_calls
            current = content_sequence[min(content_calls, len(content_sequence) - 1)]
            content_calls += 1
            return WorkflowValidationResult(current, {"source_coverage": 1.0, "component_counts": {"html": 1}})

        def validate_pedagogy(proposal):
            nonlocal pedagogy_calls
            sequence = pedagogy_sequence or [[]]
            current = sequence[min(pedagogy_calls, len(sequence) - 1)]
            pedagogy_calls += 1
            return WorkflowValidationResult(current, {"objective_coverage": 1.0, "instructional_depth": 1.0})

        def validate_duplicates(proposal):
            nonlocal duplicate_calls
            sequence = duplicate_sequence or [[]]
            current = sequence[min(duplicate_calls, len(sequence) - 1)]
            duplicate_calls += 1
            return WorkflowValidationResult(current, {"duplicate_count": len(current)})

        async def generate():
            return WorkflowGenerationResult({"chapters": [{"lessons": [{"units": [{"components": [{"type": "html"}]}]}]}]})

        async def repair(proposal, targets):
            repairs.append(targets)
            return WorkflowGenerationResult(proposal)

        callbacks = LessonGenerationWorkflowCallbacks(
            validate_contract=lambda: [],
            retrieve_evidence=retrieve,
            validate_evidence=lambda evidence: WorkflowValidationResult(evidence_issues or [], {"source_coverage": 1.0}),
            generate_proposal=generate,
            validate_content=validate_content,
            validate_pedagogy=validate_pedagogy,
            validate_duplicates=validate_duplicates,
            repair_content=repair,
            emit_diagnostic=diagnostic_events.append if diagnostic_events is not None else None,
        )
        return callbacks, repairs, lambda: content_calls

    async def test_valid_lesson_bypasses_repair(self):
        callbacks, repairs, _ = self.callbacks(content_sequence=[[]])
        _proposal, diagnostics = await run_lesson_generation_workflow(callbacks, request_context={}, max_repair_attempts=2)
        self.assertEqual(repairs, [])
        self.assertEqual(diagnostics["status"], "ready")
        self.assertIn("PROPOSAL_READY", [event["code"] for event in diagnostics["progress"]])

    async def test_invalid_component_repairs_only_component(self):
        callbacks, repairs, calls = self.callbacks(content_sequence=[
            [issue("COMPONENT_INVALID_PAYLOAD", path="chapter_1.lesson_1.unit_1.component_2")],
            [],
        ])
        await run_lesson_generation_workflow(callbacks, request_context={}, max_repair_attempts=2)
        self.assertEqual(repairs[0][0]["scope"], "component")
        self.assertEqual(repairs[0][0]["path"], "chapter_1.lesson_1.unit_1.component_2")
        self.assertEqual(calls(), 2)

    async def test_quality_failure_repairs_only_the_affected_component_then_revalidates(self):
        callbacks, repairs, calls = self.callbacks(
            content_sequence=[[], []],
            pedagogy_sequence=[
                [issue("ASSESSMENT_NOT_ALIGNED", path="chapter_1.lesson_1.unit_1.component_2")],
                [],
            ],
        )
        _proposal, diagnostics = await run_lesson_generation_workflow(callbacks, request_context={}, max_repair_attempts=2)
        self.assertEqual(repairs[0][0]["scope"], "component")
        self.assertEqual(repairs[0][0]["path"], "chapter_1.lesson_1.unit_1.component_2")
        self.assertEqual(calls(), 2)
        self.assertIn("VALIDATING_PEDAGOGY", [event["code"] for event in diagnostics["progress"]])
        self.assertIn("CHECKING_DUPLICATION", [event["code"] for event in diagnostics["progress"]])

    async def test_source_evidence_insufficient_never_generates_or_repairs(self):
        callbacks, repairs, _ = self.callbacks(
            content_sequence=[[]],
            evidence_issues=[issue("SOURCE_EVIDENCE_INSUFFICIENT", path="lesson")],
        )
        with self.assertRaisesRegex(Exception, "Lesson generation workflow") as raised:
            await run_lesson_generation_workflow(callbacks, request_context={}, max_repair_attempts=2)
        self.assertEqual(getattr(raised.exception, "code"), "SOURCE_EVIDENCE_INSUFFICIENT")
        self.assertEqual(repairs, [])

    async def test_repair_diagnostics_keep_one_correlation_and_expose_no_content(self):
        events: list[dict[str, object]] = []
        callbacks, _repairs, _ = self.callbacks(
            content_sequence=[[], []],
            pedagogy_sequence=[[issue("ASSESSMENT_NOT_ALIGNED", path="chapter_1.lesson_1.unit_1.component_1")], []],
            diagnostic_events=events,
        )
        correlation_id = "c13951f0-24ed-46e0-996a-777505be8ee8"
        await run_lesson_generation_workflow(callbacks, request_context={"correlation_id": correlation_id}, max_repair_attempts=2)
        self.assertTrue(events)
        self.assertTrue(all(event.get("correlation_id") == correlation_id for event in events))
        repair_targets = [event for event in events if event.get("stage") == "lesson_repair_target_generation"]
        self.assertEqual(len(repair_targets), 1)
        self.assertEqual(repair_targets[0]["repair_targets"][0]["path"], "chapter_1.lesson_1.unit_1.component_1")

    def test_source_scope_and_assets_are_non_repairable(self):
        targets = classify_lesson_repair_targets([
            issue("SOURCE_SCOPE_INCOMPLETE"),
            issue("ASSET_MISSING", path="chapter_1.lesson_1.unit_1.component_1"),
        ])
        self.assertEqual(targets, [])


class ScopedPatchSafetyTests(unittest.TestCase):
    def test_course_repair_cannot_introduce_a_new_source_fact(self):
        blueprint = {
            "chapters": [{"lessons": [{"units": [{"source_fact_ids": ["fact-1"], "learning_blocks": []}]}]}],
        }
        targets = [{
            "scope": "unit", "path": "chapter_1.lesson_1.unit_1", "codes": ["LESSON_TOO_THIN"],
            "allowed_fields": ["source_fact_ids", "learning_blocks"],
        }]
        with self.assertRaises(WorkflowFailure) as raised:
            apply_course_architecture_repair_patches(blueprint, targets, {
                "patches": [{"path": "chapter_1.lesson_1.unit_1", "replacement": {"source_fact_ids": ["fabricated-fact"]}}],
            })
        self.assertEqual(raised.exception.code, "ARCHITECTURE_REPAIR_INVALID")
        self.assertEqual(raised.exception.internal_code, "ARCH_REPAIR_FACT_OWNERSHIP_VIOLATION")

    def test_course_repair_json_failure_is_typed_at_the_parser(self):
        with self.assertRaises(WorkflowFailure) as raised:
            parse_course_architecture_repair_payload('{"patches": [')
        self.assertEqual(raised.exception.code, "ARCHITECTURE_REPAIR_INVALID")
        self.assertEqual(raised.exception.internal_code, "ARCH_REPAIR_JSON_INVALID")
        self.assertEqual(raised.exception.failure_stage, "architecture_repair_json_parser")

    def test_course_repair_target_whitelist_failure_is_typed_at_rejection(self):
        blueprint = {"chapters": [{"lessons": [{"units": [{"learning_blocks": []}]}]}]}
        targets = [{
            "scope": "unit", "path": "chapter_1.lesson_1.unit_1", "codes": ["LESSON_TOO_THIN"],
            "allowed_fields": ["learning_blocks"],
        }]
        with self.assertRaises(WorkflowFailure) as raised:
            apply_course_architecture_repair_patches(blueprint, targets, {
                "patches": [{"path": "chapter_1.lesson_1", "replacement": {"title": "x"}}],
            })
        self.assertEqual(raised.exception.internal_code, "ARCH_REPAIR_TARGET_OUT_OF_SCOPE")
        self.assertEqual(raised.exception.failure_stage, "architecture_repair_target_whitelist")

    def test_course_repair_patch_application_failure_is_typed_at_rejection(self):
        targets = [{
            "scope": "lesson", "path": "chapter_1.lesson_1", "codes": ["LESSON_TOO_THIN"],
            "allowed_fields": ["title"],
        }]
        with self.assertRaises(WorkflowFailure) as raised:
            apply_course_architecture_repair_patches({"chapters": []}, targets, {
                "patches": [{"path": "chapter_1.lesson_1", "replacement": {"title": "x"}}],
            })
        self.assertEqual(raised.exception.internal_code, "ARCH_REPAIR_PATCH_APPLY_FAILED")
        self.assertEqual(raised.exception.failure_stage, "architecture_repair_patch_apply")

    def test_lesson_repair_cannot_introduce_a_new_source_reference(self):
        proposal = {
            "chapters": [{"lessons": [{"units": [{"source_refs": ["src-001"], "components": [{"type": "html"}]}]}]}],
        }
        targets = [{
            "scope": "component", "path": "chapter_1.lesson_1.unit_1.component_1", "codes": ["COMPONENT_INVALID_PAYLOAD"],
            "allowed_fields": ["source_refs", "content"],
        }]
        with self.assertRaises(WorkflowFailure) as raised:
            apply_lesson_generation_repair_patches(proposal, targets, {
                "patches": [{"path": "chapter_1.lesson_1.unit_1.component_1", "replacement": {"source_refs": ["other-document"]}}],
            })
        self.assertEqual(raised.exception.code, "LESSON_REPAIR_INVALID")
        self.assertEqual(raised.exception.internal_code, "LESSON_REPAIR_SOURCE_SCOPE_EXPANSION")
        self.assertEqual(raised.exception.failure_stage, "lesson_repair_source_scope_guard")

    def test_lesson_repair_target_and_field_rejections_are_typed_at_the_guard(self):
        proposal = {"chapters": [{"lessons": [{"units": [{"components": [{"type": "html"}]}]}]}]}
        targets = [{
            "scope": "component", "path": "chapter_1.lesson_1.unit_1.component_1", "codes": ["COMPONENT_INVALID_PAYLOAD"],
            "allowed_fields": ["content"],
        }]
        with self.assertRaises(WorkflowFailure) as target_error:
            apply_lesson_generation_repair_patches(proposal, targets, {
                "patches": [{"path": "chapter_1.lesson_1", "replacement": {"content": "x"}}],
            })
        self.assertEqual(target_error.exception.internal_code, "LESSON_REPAIR_TARGET_OUT_OF_SCOPE")
        self.assertEqual(target_error.exception.failure_stage, "lesson_repair_target_whitelist")

        with self.assertRaises(WorkflowFailure) as field_error:
            apply_lesson_generation_repair_patches(proposal, targets, {
                "patches": [{"path": "chapter_1.lesson_1.unit_1.component_1", "replacement": {"type": "problem"}}],
            })
        self.assertEqual(field_error.exception.internal_code, "LESSON_REPAIR_FIELD_OUT_OF_SCOPE")
        self.assertEqual(field_error.exception.failure_stage, "lesson_repair_field_whitelist")

    def test_oversized_repair_snapshot_fails_closed_instead_of_truncating(self):
        blueprint = {"course_title": "x" * 25_000, "chapters": []}
        targets = [{
            "scope": "course", "path": "course", "codes": ["SOURCE_COVERAGE_INCOMPLETE"],
            "allowed_fields": ["course_title", "chapters"],
        }]
        with self.assertRaises(WorkflowFailure) as raised:
            build_course_architecture_repair_prompt(
                blueprint=blueprint,
                targets=targets,
                source_map_context="{}",
                locale="en",
            )
        self.assertEqual(raised.exception.code, "ARCHITECTURE_REPAIR_SCOPE_TOO_LARGE")
