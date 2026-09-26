"""Wire projection regression; mock HTTP only and preserve local acceptance."""
from copy import deepcopy
import json
import unittest

from pydantic import BaseModel, Field, ValidationError

from app import main
from app.lesson_author_provider_schema import staged_provider_response_model
from tests.staged_schema_probe import BOUNDS, capture_sdk_body, isolated_bound_variant, schema_cases, schema_metadata
from tests.test_checkpoint_component_quality_repair import instance_wire
from tests.test_staged_instance_output import checkpoint_instance_fixture


class StagedProviderProjectionTests(unittest.TestCase):
    def test_mixed_sdk_wire_matches_the_live_accepted_schema_fingerprint(self):
        projected, _ = staged_provider_response_model(schema_cases()["mixed"])
        # Approved synthetic call10, 2026-09-26: HTTP200 / STOP. This fingerprint
        # is an observed fixture, not a guarantee of future provider availability.
        self.assertEqual(schema_metadata(capture_sdk_body(projected))["schema_sha256"],
                         "72c97f8ae697eb01e956581a50a2964b8c9692c238d71511974724228bd389ac")

    def test_actual_sdk_body_matches_successful_live_variant_for_every_case(self):
        for name, model in schema_cases().items():
            with self.subTest(case=name):
                original = deepcopy(model.model_json_schema())
                projected, metadata = staged_provider_response_model(model)
                self.assertEqual(capture_sdk_body(projected), isolated_bound_variant(capture_sdk_body(model), BOUNDS))
                self.assertEqual(model.model_json_schema(), original)
                self.assertTrue(metadata["server_validation_unchanged"])
                self.assertEqual(metadata["schema_projection_version"], "staged-wire-shape-1")

    def test_local_validation_retains_fact_cardinality_length_and_numeric_limits(self):
        request, unit, _, _ = checkpoint_instance_fixture()
        plans = request.blueprint_architecture.lessons[0].units[0].component_plan
        original = main.build_staged_instance_response_model([p.model_dump() for p in plans])
        projected, _ = staged_provider_response_model(original)
        valid = instance_wire(unit)
        # Fixture omits nullable wire fields. Populate only for typed-model test;
        # production does not synthesize provider content.
        for slot_name, field in original.model_fields["components"].annotation.model_fields.items():
            for key, definition in field.annotation.model_fields.items():
                if key not in valid["components"][slot_name] and definition.is_required():
                    valid["components"][slot_name][key] = [] if getattr(definition.annotation, "__origin__", None) is list else None
        for slot in valid["components"].values():
            if slot.get("nodes"):
                for node in slot["nodes"]:
                    node.setdefault("shape", "rounded")
        original.model_validate(valid)
        self.assertEqual(original.model_validate(valid).model_dump(), projected.model_validate(valid).model_dump())
        changes = [
            ("c1", "covered_source_fact_ids", ["unowned"]),
            ("c3", "items", []),
            ("c2", "nodes", []),
            ("c2", "edges", [{"source": -1, "target": 1}]),
            ("c3", "items", [{"question": "x" * 501, "answer": "a"}] * 2),
            ("c0", "semantic_content", {"paragraphs": ["x" * 2001]}),
        ]
        for slot, key, value in changes:
            broken = deepcopy(valid)
            broken["components"][slot][key] = value
            for model in (original, projected):
                with self.subTest(slot=slot, field=key, model=model.__name__), self.assertRaises(ValidationError):
                    model.model_validate(broken)

    def test_projection_does_not_strip_properties_named_like_constraints(self):
        class Example(BaseModel):
            minimum: int = Field(ge=0)
            maxItems: str = Field(max_length=5)
        projected, _ = staged_provider_response_model(Example)
        self.assertEqual(set(projected.model_json_schema()["properties"]), {"minimum", "maxItems"})
        self.assertEqual(projected.model_json_schema()["required"], ["minimum", "maxItems"])
        with self.assertRaises(ValidationError):
            projected.model_validate({"minimum": -1, "maxItems": "ok"})

    def test_existing_payload_repair_uses_same_projection_without_losing_address(self):
        for kind in ("html", "problem", "la_faq", "la_crossword", "la_diagram", "la_sortable"):
            original = main.build_staged_lesson_content_response_model([kind], payload_only=True)
            projected, _ = staged_provider_response_model(original)
            body = capture_sdk_body(projected)
            component = body["generationConfig"]["responseSchema"]["properties"]["components"]["items"]
            self.assertIn("component_index", component["required"])
            self.assertNotIn("source_fact_ids", component["properties"])
            self.assertEqual(body, isolated_bound_variant(capture_sdk_body(original), BOUNDS))


if __name__ == "__main__":
    unittest.main()
