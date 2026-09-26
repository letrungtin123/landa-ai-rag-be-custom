"""Actual SDK HTTP serialization, not a live Gemini compatibility claim."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from google import genai
from google.genai import models, types
from pydantic import ValidationError

from tests.staged_schema_probe import (
    BOUNDS, capture_sdk_body, isolated_bound_variant, schema_cases, schema_metadata, visit_schema,
)


class StagedProviderWireTests(unittest.TestCase):
    def test_all_selected_component_schemas_serialize_without_network(self):
        with patch("socket.create_connection", side_effect=AssertionError("NETWORK_FORBIDDEN")):
            for name, schema in schema_cases().items():
                with self.subTest(case=name):
                    body = capture_sdk_body(schema)
                    metadata = schema_metadata(body)
                    self.assertEqual(metadata["arrays_missing_items"], 0)
                    self.assertEqual(metadata["missing_required_properties"], 0)
                    self.assertEqual(metadata["provider_acceptance"], "NOT_TESTED")
                    self.assertEqual(metadata["max_output_tokens"], 30000)
                    self.assertEqual(body, capture_sdk_body(schema))

    def test_sdk_root_guard_does_not_validate_nested_bounds(self):
        client = genai.Client(api_key="offline")
        with self.assertRaisesRegex(ValueError, "max_items"):
            models._Schema_to_mldev(client._api_client, types.Schema(type="ARRAY", items=types.Schema(type="STRING"), max_items=0))
        model = schema_cases()["la_faq"]
        wire = capture_sdk_body(model)["generationConfig"]["responseSchema"]
        self.assertEqual(wire["properties"]["components"]["properties"]["c0"]["properties"]["covered_source_fact_ids"]["max_items"], 0)

    def test_diagnostic_variant_changes_only_selected_schema_bounds(self):
        body = capture_sdk_body(schema_cases()["mixed"])
        before = deepcopy(body)
        variant = isolated_bound_variant(body, {"min_length", "max_length"})
        self.assertEqual(body, before)
        for (path, original), (other_path, changed) in zip(
            visit_schema(body["generationConfig"]["responseSchema"]),
            visit_schema(variant["generationConfig"]["responseSchema"]),
        ):
            self.assertEqual(path, other_path)
            self.assertNotIn("min_length", changed)
            self.assertNotIn("max_length", changed)
            for key in ("type", "required", "enum", "nullable", "min_items", "max_items", "minimum"):
                self.assertEqual(original.get(key), changed.get(key))
        self.assertEqual(body["contents"], variant["contents"])
        self.assertEqual(body["generationConfig"]["maxOutputTokens"], variant["generationConfig"]["maxOutputTokens"])
        with self.assertRaisesRegex(ValueError, "UNKNOWN_DIAGNOSTIC_BOUND"):
            isolated_bound_variant(body, {"required"})

    def test_server_constraints_remain_even_when_testing_wire_variants(self):
        schema = schema_cases()["la_faq"]
        capture_sdk_body(schema)
        slot = schema.model_fields["components"].annotation.model_fields["c0"].annotation
        minimal = {"covered_source_fact_ids": [], "title": None, "selection_rationale": None,
                   "items": [{"question": "Why?", "answer": "Because."}, {"question": "When?", "answer": "Now."}]}
        slot.model_validate(minimal)
        for bad in ({**minimal, "covered_source_fact_ids": ["unowned-fact"]}, {**minimal, "items": []}):
            with self.assertRaises(ValidationError):
                slot.model_validate(bad)
        body = capture_sdk_body(schema)
        report = json.dumps(schema_metadata(isolated_bound_variant(body, BOUNDS)))
        self.assertNotIn("contents", report)
        self.assertNotIn("offline-placeholder", report)


if __name__ == "__main__":
    unittest.main()
