"""Stage-2 wire projection only; the inherited server model stays authoritative.

The combined component schema's value bounds cause a provider rejection in the
synthetic UAT reproduction. Keep types/required/enum/nullable/object structure on
the wire, and enforce value bounds locally. Do not use this for Course Architect.
"""
from collections import Counter
from copy import deepcopy
from typing import Any

from pydantic import BaseModel


STAGED_SCHEMA_PROJECTION_VERSION = "staged-wire-shape-1"
VALUE_BOUNDS = frozenset({"minItems", "maxItems", "minLength", "maxLength", "minimum", "maximum"})


def _project_schema(schema: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    result = deepcopy(schema)
    removed: Counter[str] = Counter()

    def visit(node: dict[str, Any]) -> None:
        for key in VALUE_BOUNDS:
            if key in node:
                removed[key] += 1
                del node[key]
        # Walk schema nodes, never property names or examples/default values.
        for key in ("properties", "$defs"):
            for child in node.get(key, {}).values():
                if isinstance(child, dict):
                    visit(child)
        if isinstance(node.get("items"), dict):
            visit(node["items"])
        for key in ("anyOf", "allOf", "oneOf"):
            for child in node.get(key, []):
                if isinstance(child, dict):
                    visit(child)

    visit(result)
    return result, dict(sorted(removed.items()))


def staged_provider_response_model(server_model: type[BaseModel]) -> tuple[type[BaseModel], dict[str, Any]]:
    """Override schema export, NOT field definitions or Pydantic validation.

    Returning a Pydantic subclass also retains google-genai1.0's guarded parse
    branch. Invalid content still proceeds to the existing bounded local repair
    or fail-closed path; no validation result is converted to success here.
    """
    _, removed = _project_schema(server_model.model_json_schema())

    class StagedProviderWire(server_model):
        @classmethod
        def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return _project_schema(super().model_json_schema(*args, **kwargs))[0]

    return StagedProviderWire, {
        "schema_projection_version": STAGED_SCHEMA_PROJECTION_VERSION,
        "server_validation_unchanged": True,
        "projected_declared_bound_counts": removed,
    }
