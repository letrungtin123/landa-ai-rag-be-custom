"""Refactor step 2.3: one provider schema projector and one home for parsing provider JSON answers."""

from __future__ import annotations

import unittest
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.content.provider_json import parse_first_provider_json_value, parse_provider_json_object
from app.contracts.orchestration_v2.provider import orchestration_v2_provider_response_model
from app.contracts.schema_projection import ORCHESTRATION_V2_WIRE, STAGED_WIRE, project_provider_schema
from app.core.errors import ServiceError
from app.idm.runtime import idm_provider_response_model
from app.lesson_author_provider_schema import staged_provider_response_model


class Child(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=2, max_length=8, pattern=r"^[A-Z]+$")


class Parent(BaseModel):
    """Server model."""

    model_config = ConfigDict(extra="forbid")

    items: list[int] = Field(min_length=1, max_length=4)
    child: Child | None = None


def detail(error: ServiceError) -> Any:
    return error.detail


class SchemaProjectionTests(unittest.TestCase):
    def test_staged_profile_removes_value_bounds_only(self) -> None:
        projected, removed = project_provider_schema(Parent.model_json_schema(), STAGED_WIRE)
        self.assertEqual(removed, {"maxItems": 1, "maxLength": 1, "minItems": 1, "minLength": 1})
        self.assertFalse(projected["additionalProperties"])
        self.assertIn("anyOf", projected["properties"]["child"])

    def test_v2_profile_also_removes_additional_properties_and_inlines_nullable_refs(self) -> None:
        projected, removed = project_provider_schema(Parent.model_json_schema(), ORCHESTRATION_V2_WIRE)
        # The inlined copy is walked too, so the child's bounds count once in $defs and once inline.
        self.assertEqual(removed, {"additionalProperties": 2, "maxItems": 1, "maxLength": 2, "minItems": 1,
                                   "minLength": 2, "nullableRefAnyOf": 1})
        child = projected["properties"]["child"]
        self.assertTrue(child["nullable"])
        self.assertEqual(child["properties"]["code"]["pattern"], r"^[A-Z]+$")
        # The inlined copy keeps the keywords of the referenced definition (unchanged behaviour).
        self.assertFalse(child["additionalProperties"])

    def test_wire_models_share_the_projection_and_keep_their_names(self) -> None:
        staged, staged_meta = staged_provider_response_model(Parent)
        v2, v2_meta = orchestration_v2_provider_response_model(Parent)
        idm = idm_provider_response_model(Parent)
        self.assertEqual((staged.__name__, v2.__name__, idm.__name__),
                         ("StagedProviderWire", "ParentProviderWire", "ParentIdmWire"))
        self.assertEqual(staged_meta["projected_declared_bound_counts"],
                         project_provider_schema(Parent.model_json_schema(), STAGED_WIRE)[1])
        self.assertEqual(v2_meta["projected_unsupported_keyword_counts"],
                         project_provider_schema(Parent.model_json_schema(), ORCHESTRATION_V2_WIRE)[1])
        for wire in (staged, v2, idm):
            with self.subTest(wire=wire.__name__):
                self.assertEqual(wire.model_json_schema()["title"], "Parent")
                self.assertTrue(issubclass(wire, Parent))
        self.assertNotIn("pattern", str(idm.model_json_schema()))
        self.assertIn("pattern", str(v2.model_json_schema()))


class ProviderJsonTests(unittest.TestCase):
    def test_object_rule_accepts_the_whole_answer_or_the_widest_object_span(self) -> None:
        self.assertEqual(parse_provider_json_object('{"a": 1}', "proposal"), {"a": 1})
        self.assertEqual(parse_provider_json_object('Here: {"a": {"b": 2}} done', "proposal"), {"a": {"b": 2}})

    def test_object_rule_errors_keep_their_messages(self) -> None:
        cases = {"no json": "AI không trả JSON proposal hợp lệ.", "{bad} {": "AI không trả JSON proposal hợp lệ.",
                 "[1, 2]": "AI không trả JSON proposal dạng object hợp lệ."}
        for text, message in cases.items():
            with self.subTest(text=text), self.assertRaises(ServiceError) as caught:
                parse_provider_json_object(text, "proposal")
            self.assertEqual(caught.exception.status_code, 502)
            self.assertEqual(detail(caught.exception), message)

    def test_value_rule_returns_the_first_complete_value(self) -> None:
        self.assertEqual(parse_first_provider_json_value('x [1] {"a": 1}', "batch"), [1])
        self.assertEqual(parse_first_provider_json_value('{bad {"a": 1} tail', "batch"), {"a": 1})
        with self.assertRaises(ServiceError) as caught:
            parse_first_provider_json_value("none", "batch")
        self.assertEqual(detail(caught.exception), "AI không trả JSON batch hợp lệ.")


if __name__ == "__main__":
    unittest.main()
