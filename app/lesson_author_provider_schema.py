"""Stage-2 wire projection only; the inherited server model stays authoritative.

The combined component schema's value bounds cause a provider rejection in the
synthetic UAT reproduction. Keep types/required/enum/nullable/object structure on
the wire, and enforce value bounds locally. Do not use this for Course Architect.
"""
from typing import Any

from pydantic import BaseModel

from app.contracts.schema_projection import STAGED_WIRE, project_provider_schema, provider_wire_model

STAGED_SCHEMA_PROJECTION_VERSION = "staged-wire-shape-1"


def staged_provider_response_model(server_model: type[BaseModel]) -> tuple[type[BaseModel], dict[str, Any]]:
    """Override schema export, NOT field definitions or Pydantic validation.

    Returning a Pydantic subclass also retains google-genai1.0's guarded parse
    branch. Invalid content still proceeds to the existing bounded local repair
    or fail-closed path; no validation result is converted to success here.
    """
    _, removed = project_provider_schema(server_model.model_json_schema(), STAGED_WIRE)
    wire = provider_wire_model(server_model, STAGED_WIRE, name="StagedProviderWire")
    return wire, {
        "schema_projection_version": STAGED_SCHEMA_PROJECTION_VERSION,
        "server_validation_unchanged": True,
        "projected_declared_bound_counts": removed,
    }
