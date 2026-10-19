"""Provider wire projection of Pydantic JSON Schemas: the one implementation behind every structured-output call.

Google structured output accepts only a documented JSON Schema subset, and value bounds plus
``additionalProperties`` emitted by the strict server models have triggered provider-side ``INVALID_ARGUMENT``
responses. The projection removes those keywords from the provider wire only; the authoritative Pydantic models are
unchanged and still enforce every bound after generation.

Each wire keeps its own profile (the exact keywords and schema nodes it has always projected) and its own
projection version string, because both are part of the provider request and of its telemetry.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

VALUE_BOUNDS = frozenset({"minItems", "maxItems", "minLength", "maxLength", "minimum", "maximum"})
_DEFS_PREFIX = "#/$defs/"
_NULLABLE_PAIR = 2  # anyOf([value, null])


@dataclass(frozen=True)
class WireProfile:
    """What one provider wire removes from a schema and which schema nodes it walks."""

    removed_keywords: frozenset[str]
    remove_additional_properties: bool
    # Inline ``anyOf([$ref, null])`` as the referenced object plus ``nullable: true``.
    inline_nullable_refs: bool
    mapping_keys: tuple[str, ...]
    list_keys: tuple[str, ...]


# Legacy staged lesson writer (stage 2): value bounds only.
STAGED_WIRE = WireProfile(
    removed_keywords=VALUE_BOUNDS, remove_additional_properties=False, inline_nullable_refs=False,
    mapping_keys=("properties", "$defs"), list_keys=("anyOf", "allOf", "oneOf"),
)
# Orchestration v2 and IDM: value bounds, ``additionalProperties`` and nullable references.
ORCHESTRATION_V2_WIRE = WireProfile(
    removed_keywords=VALUE_BOUNDS, remove_additional_properties=True, inline_nullable_refs=True,
    mapping_keys=("properties", "$defs", "definitions"), list_keys=("anyOf", "allOf", "oneOf", "prefixItems"),
)


def _inline_nullable_ref(node: dict[str, Any], definitions: dict[str, Any], removed: Counter[str]) -> None:
    # Keep nullable nested objects explicit on the provider wire. Pydantic emits ``anyOf([$ref, null])`` for this
    # shape; inlining only that exact reference avoids SDK/version-specific conversion differences while preserving
    # the authoritative local model and nullable semantics.
    any_of = node.get("anyOf")
    if not isinstance(any_of, list) or len(any_of) != _NULLABLE_PAIR:
        return
    null_members = [item for item in any_of if isinstance(item, dict) and item.get("type") == "null"]
    value_members = [item for item in any_of if item not in null_members]
    if len(null_members) != 1 or len(value_members) != 1 or not isinstance(value_members[0], dict):
        return
    ref = value_members[0].get("$ref")
    target = definitions.get(ref[len(_DEFS_PREFIX):]) if isinstance(ref, str) and ref.startswith(_DEFS_PREFIX) else None
    if isinstance(target, dict):
        siblings = {key: value for key, value in node.items() if key != "anyOf"}
        node.clear()
        node.update(deepcopy(target))
        node.update(siblings)
        node["nullable"] = True
        removed["nullableRefAnyOf"] += 1


def project_provider_schema(schema: dict[str, Any], profile: WireProfile) -> tuple[dict[str, Any], dict[str, int]]:
    """Return the projected copy of ``schema`` and the sorted count of every removed keyword."""

    result = deepcopy(schema)
    definitions = result.get("$defs", {})
    removed: Counter[str] = Counter()

    def visit(node: dict[str, Any]) -> None:
        if profile.remove_additional_properties and "additionalProperties" in node:
            removed["additionalProperties"] += 1
            del node["additionalProperties"]
        for keyword in profile.removed_keywords:
            if keyword in node:
                removed[keyword] += 1
                del node[keyword]
        if profile.inline_nullable_refs:
            _inline_nullable_ref(node, definitions, removed)
        # Walk schema values, never property names or arbitrary examples/default values.
        for key in profile.mapping_keys:
            for child in node.get(key, {}).values():
                if isinstance(child, dict):
                    visit(child)
        if isinstance(node.get("items"), dict):
            visit(node["items"])
        for key in profile.list_keys:
            for child in node.get(key, []):
                if isinstance(child, dict):
                    visit(child)

    visit(result)
    return result, dict(sorted(removed.items()))


def provider_wire_model(
    server_model: type[BaseModel],
    profile: WireProfile,
    *,
    name: str | None = None,
    finish: Callable[[dict[str, Any]], None] | None = None,
) -> type[BaseModel]:
    """A subclass of ``server_model`` whose JSON Schema is the provider projection; validation is unchanged.

    Returning a Pydantic subclass also keeps google-genai's guarded parse branch. ``name`` renames the class (and
    its qualified name); ``finish`` edits the projected schema in place after the projection.
    """

    class ProviderWire(server_model):  # type: ignore[misc,valid-type]  # dynamic base class
        @classmethod
        def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
            projected = project_provider_schema(super().model_json_schema(*args, **kwargs), profile)[0]
            if finish is not None:
                finish(projected)
            # A dynamic subclass changes Pydantic's root title. Keep the server model title so the projection is
            # the only wire delta.
            if "title" in projected:
                projected["title"] = server_model.__name__
            return projected

    if name is not None:
        ProviderWire.__name__ = name
        ProviderWire.__qualname__ = name
    return ProviderWire
