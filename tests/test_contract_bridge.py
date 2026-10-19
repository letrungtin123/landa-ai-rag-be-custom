"""The public contract surface for Node (refactor step 2.6): bridge names, route table and the JSON Schema file."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from fastapi.routing import APIRoute
from pydantic import BaseModel

from app import contract_bridge, main
from app.contract_bridge import CONTRACT_SCHEMA_PATH, ROUTE_CONTRACTS, contract_schema_text, write_contract_schema
from app.schemas.chat import RagRequestBase
from app.services.response_contracts import OBSERVED_ROUTES

REGENERATE = "python -m app.contract_bridge"


def post_routes() -> dict[str, APIRoute]:
    return {route.path: route for route in main.app.routes if isinstance(route, APIRoute) and "POST" in route.methods}


def body_model(route: APIRoute) -> Any:
    (param,) = route.dependant.body_params
    return param.field_info.annotation


class ContractBridgeTests(unittest.TestCase):
    def test_committed_schema_matches_the_models(self) -> None:
        committed = CONTRACT_SCHEMA_PATH.read_text(encoding="utf-8")
        self.assertEqual(committed, contract_schema_text(), f"contract schema is stale; run `{REGENERATE}`")

    def test_writer_produces_the_committed_document(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            written = write_contract_schema(Path(folder) / "nested" / "schema.json")
            # Text comparison: a Windows checkout may turn the committed file's LF into CRLF.
            self.assertEqual(written.read_text(encoding="utf-8"), CONTRACT_SCHEMA_PATH.read_text(encoding="utf-8"))

    def test_route_table_matches_the_application(self) -> None:
        routes = post_routes()
        self.assertEqual(set(ROUTE_CONTRACTS), set(routes))
        for path, (request_model, response_type) in ROUTE_CONTRACTS.items():
            with self.subTest(path=path):
                self.assertIs(body_model(routes[path]), request_model)
                # An untyped route, and an observed generation route, answer through ``dict[str, Any]``.
                runtime = dict[str, Any] if response_type is None or path in OBSERVED_ROUTES else response_type
                self.assertEqual(routes[path].response_model, runtime)

    def test_every_route_has_its_own_top_level_request_model(self) -> None:
        # No route body inherits another route's body model; shared fields live in non-route bases.
        models = [request_model for request_model, _ in ROUTE_CONTRACTS.values()]
        self.assertEqual(len(models), len(set(models)))
        for model in models:
            others = [other for other in models if other is not model]
            with self.subTest(model=model.__name__):
                self.assertFalse([other.__name__ for other in others if issubclass(model, other)])

    def test_rag_routes_share_only_the_envelope_base(self) -> None:
        rag_models = [model for model, _ in ROUTE_CONTRACTS.values() if issubclass(model, RagRequestBase)]
        self.assertEqual(len(rag_models), 8)
        for model in rag_models:
            with self.subTest(model=model.__name__):
                self.assertEqual(list(model.model_fields)[:len(RagRequestBase.model_fields)],
                                 list(RagRequestBase.model_fields))

    def test_schema_document_shape(self) -> None:
        document = json.loads(contract_schema_text())
        self.assertEqual(list(document), ["$schema", "$comment", "routes", "$defs"])
        for path, (request_model, response_type) in ROUTE_CONTRACTS.items():
            with self.subTest(path=path):
                entry = document["routes"][path]
                self.assertEqual(entry["request"], {"$ref": f"#/$defs/{request_model.__name__}"})
                self.assertEqual(entry["response"] is None, response_type is None)
        skeleton = document["$defs"]["CourseSkeletonV2Response"]
        # Absent add-ons are optional and never null on the wire.
        self.assertNotIn("idm", skeleton["required"])
        self.assertEqual(skeleton["properties"]["idm"]["$ref"], "#/$defs/IdmCourseDesignV1")
        unit = document["$defs"]["UnitV2Response"]
        self.assertEqual(unit["properties"]["semantic_review"]["type"], "object")

    def test_app_main_reexports_the_bridge_names(self) -> None:
        names = [name for name in main.__all__ if name not in {"app", "create_app"}]
        self.assertEqual(len(names), 8)
        for name in names:
            with self.subTest(name=name):
                self.assertIn(name, contract_bridge.__all__)
                self.assertIs(getattr(main, name), getattr(contract_bridge, name))

    def test_response_types_are_models_or_annotated_unions(self) -> None:
        for path, (_, response_type) in ROUTE_CONTRACTS.items():
            if isinstance(response_type, type):
                with self.subTest(path=path):
                    self.assertTrue(issubclass(response_type, BaseModel))


if __name__ == "__main__":
    unittest.main()
