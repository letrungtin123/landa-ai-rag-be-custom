"""Synthetic SDK-wire isolation. Offline by default; never reads keys or the DB.

The captured HTTP body is deliberately kept in memory. The printable report is
metadata only and does not equate SDK serialization with provider acceptance.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from unittest.mock import patch

from google import genai
from google.genai import errors, types
from pydantic import create_model
import requests

from app import main


KINDS = ("html", "problem", "la_faq", "la_crossword", "la_diagram", "la_sortable")
BOUNDS = {"min_items", "max_items", "min_length", "max_length", "minimum", "maximum"}


def synthetic_plans(kinds):
    return [{"component_plan_id": "cp2_" + f"{i + 1:032x}", "type": kind,
             "source_fact_ids": ["synthetic-fact"] if kind == "html" else [],
             "supporting_evidence_fact_ids": [] if kind == "html" else ["synthetic-fact"]}
            for i, kind in enumerate(kinds)]


def schema_cases():
    cases = {"control": create_model("ProbeControl", result=(str, ...))}
    for kind in KINDS:
        cases[kind] = main.build_staged_instance_response_model(synthetic_plans([kind]))
    cases["mixed"] = main.build_staged_instance_response_model(
        synthetic_plans(["html", "problem", "la_diagram", "la_faq"]))
    return cases


def capture_sdk_body(schema, model="offline-model", max_output_tokens=30000):
    """Run actual SDK serialization, but replace its entire HTTP transport."""
    error_response = requests.Response()
    error_response.status_code = 400
    error_response._content = b'{"error":{"code":400,"message":"offline capture"}}'
    with patch("requests.Session.request", return_value=error_response) as transport:
        try:
            genai.Client(api_key="offline-placeholder", http_options=types.HttpOptions(timeout=180000)).models.generate_content(
                model=model, contents="Synthetic contract probe. Return the smallest schema-valid JSON. No private data.",
                config={"response_schema": schema, "response_mime_type": "application/json",
                        "max_output_tokens": max_output_tokens, "temperature": main.settings.generation_temperature,
                        "thinking_config": types.ThinkingConfig(include_thoughts=False)})
        except errors.ClientError:
            pass
        if transport.call_count != 1:
            raise AssertionError("Expected exactly one mocked SDK dispatch")
        return json.loads(transport.call_args.kwargs["data"])


def visit_schema(node, path="responseSchema"):
    yield path, node
    for name, child in node.get("properties", {}).items():
        yield from visit_schema(child, path + "." + name)
    if isinstance(node.get("items"), dict):
        yield from visit_schema(node["items"], path + ".items")


def schema_metadata(body):
    schema = body["generationConfig"]["responseSchema"]
    serialized = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    counts = Counter()
    missing_required = missing_items = nodes = depth = 0
    for path, node in visit_schema(schema):
        nodes += 1
        depth = max(depth, path.count("."))
        counts.update(key for key in node if key in BOUNDS)
        missing_required += len(set(node.get("required", [])) - set(node.get("properties", {})))
        missing_items += int(node.get("type") == "ARRAY" and "items" not in node)
    return {"schema_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
            "schema_bytes": len(serialized.encode()), "schema_nodes": nodes, "depth": depth,
            "bound_counts": dict(sorted(counts.items())), "missing_required_properties": missing_required,
            "arrays_missing_items": missing_items, "provider_acceptance": "NOT_TESTED",
            "max_output_tokens": body["generationConfig"]["maxOutputTokens"]}


def isolated_bound_variant(body, removed_bounds):
    """Diagnostic-only variant; never imported by application code."""
    if not set(removed_bounds) <= BOUNDS:
        raise ValueError("UNKNOWN_DIAGNOSTIC_BOUND")
    variant = deepcopy(body)
    for _, node in visit_schema(variant["generationConfig"]["responseSchema"]):
        for key in removed_bounds:
            node.pop(key, None)
    return variant


if __name__ == "__main__":
    for name, schema in schema_cases().items():
        print(json.dumps({"case": name, "live_provider_calls": 0, **schema_metadata(capture_sdk_body(schema))}))
