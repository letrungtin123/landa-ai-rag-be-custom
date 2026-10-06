"""Deterministic CP2A replay across the production Python boundaries.

This module is test-only. It never opens a database connection, performs HTTP,
or calls a model provider. Full candidate/source values stay in the frozen test
fixture; the returned lineage contains only safe identifiers, boundary names,
contract metadata, and accepted synthetic payloads needed by the Node replay.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any

from pydantic import ValidationError

from app.lesson_author_orchestration_v2 import canonical_hash
from app.lesson_author_orchestration_v2_provider import (
    ORCHESTRATION_V2_PROVIDER_SCHEMA_PROJECTION_VERSION,
    UnitGenerationContractV2,
    unit_contract_manifest_v2,
    unit_contract_v5_architecture_v2,
)
from app.main import (
    bind_staged_instance_payload,
    build_staged_instance_response_model,
    staged_response_schema_diagnostics,
    validate_staged_unit_content,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "ai_id_cp2a_boundary_replay_fixture.json"
DENSITY_VERSION = "unit-content-v3-density-1"


def load_fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _contract(case: dict[str, Any]) -> UnitGenerationContractV2:
    fact_ids = [fact["fact_key"] for fact in case["source_facts"]]
    facts = [{
        "document_id": "cp2a-reviewer-fixture",
        "fact_key": fact["fact_key"],
        "scope_key": "cp2a-scope-1",
        "fact_text": fact["fact_text"],
        "source_ref": "cp2a-reviewer-fixture",
        "source_page": 1,
        "source_chunk": index,
        "locator": {
            **deepcopy(fact.get("locator") or {}),
            "instructional_density_policy_version": DENSITY_VERSION,
        },
    } for index, fact in enumerate(case["source_facts"])]
    plans = [{
        **deepcopy(plan),
        "learning_objective_refs": ["lo_1"],
        "source_scope_ids": ["cp2a-scope-1"],
        "content_requirements": [],
        "learning_block_ids": [],
    } for plan in case["plans"]]
    base = {
        "contract_version": 2,
        "source_snapshot_hash": "a" * 64,
        "assembly_hash": "b" * 64,
        "chapter_key": "chapter-1",
        "unit_path": "chapter_1.lesson_1.unit_1",
        "chapter_title": "CP2A reviewer fixture",
        "lesson_title": "Boundary replay",
        "lesson_learning_objectives": ["Áp dụng đúng nội dung đã học."],
        "unit_title": f"CP2A {case['id']}",
        "unit_purpose": "Replay candidate qua từng boundary production đã khóa.",
        "unit_learning_objective_refs": ["lo_1"],
        "unit_source_scope_ids": ["cp2a-scope-1"],
        "unit_source_fact_ids": fact_ids,
        "component_plan": plans,
        "source_facts": facts,
    }
    return UnitGenerationContractV2.model_validate({
        **base,
        "contract_hash": canonical_hash(base),
    })


def _result(case: dict[str, Any], **values: Any) -> dict[str, Any]:
    return {
        "case_id": case["id"],
        "category": case["category"],
        "fixture_version": load_fixture()["fixture_version"],
        "provider_projection_version": ORCHESTRATION_V2_PROVIDER_SCHEMA_PROJECTION_VERSION,
        **values,
    }


def replay_case(case: dict[str, Any]) -> dict[str, Any]:
    contract = _contract(case)
    architecture = unit_contract_v5_architecture_v2(contract)
    manifest = unit_contract_manifest_v2(contract)

    if case.get("requires_visual_asset") is True:
        # The actual V2 writer input currently carries text facts but neither a
        # versioned asset reference nor image-region observations. Do not let a
        # synthetic paragraph masquerade as a replay of visual evidence.
        serialized_input = json.dumps({
            "manifest": manifest,
            "architecture": architecture,
        }, ensure_ascii=False)
        locator_has_asset = any(
            fact.locator.get("asset_revision")
            for fact in contract.source_facts
        )
        writer_has_asset = "asset_revision" in serialized_input or "image_asset" in serialized_input
        if not locator_has_asset or not writer_has_asset:
            return _result(
                case,
                status="inconclusive",
                first_failing_boundary="writer_input_assembly",
                classification="input_missing",
                code="IMAGE_ASSET_CONTEXT_UNAVAILABLE",
                contract=contract.model_dump(mode="json"),
                unit=None,
                schema_diagnostics=None,
            )

    expected = deepcopy(architecture["lessons"][0]["units"][0])
    expected["unit_title"] = expected["title"]
    expected["component_types"] = [plan.type for plan in contract.component_plan]
    model = build_staged_instance_response_model(expected["component_plan"])
    schema_diagnostics = staged_response_schema_diagnostics(model)
    try:
        parsed = model.model_validate(case["writer_payload"]).model_dump(mode="json")
    except ValidationError:
        return _result(
            case,
            status="rejected",
            first_failing_boundary="provider_wire_schema",
            classification="candidate_invalid",
            code="PROVIDER_WIRE_SCHEMA_REJECTED",
            contract=contract.model_dump(mode="json"),
            unit=None,
            schema_diagnostics=schema_diagnostics,
        )

    try:
        unit = bind_staged_instance_payload(parsed, expected)
    except Exception:
        return _result(
            case,
            status="rejected",
            first_failing_boundary="server_owned_binding",
            classification="adapter_rejection",
            code="SERVER_OWNED_BINDING_REJECTED",
            contract=contract.model_dump(mode="json"),
            unit=None,
            schema_diagnostics=schema_diagnostics,
        )

    finding = validate_staged_unit_content(unit, expected, strict_payload=True)
    if finding is not None:
        return _result(
            case,
            status="rejected",
            first_failing_boundary="python_instructional_validator",
            classification="validator_rejection",
            code=finding.code,
            path=finding.path,
            contract=contract.model_dump(mode="json"),
            unit=None,
            schema_diagnostics=schema_diagnostics,
        )

    return _result(
        case,
        status="accepted",
        first_failing_boundary=None,
        classification="accepted",
        code=None,
        contract=contract.model_dump(mode="json"),
        unit=unit,
        schema_diagnostics=schema_diagnostics,
    )


def replay_fixture() -> dict[str, Any]:
    fixture = load_fixture()
    return {
        "fixture_version": fixture["fixture_version"],
        "contract_versions": fixture["contract_versions"],
        "results": [replay_case(case) for case in fixture["cases"]],
    }


def main() -> None:
    raw = sys.stdin.read() if not sys.stdin.isatty() else ""
    if raw.strip():
        json.loads(raw)
    print(json.dumps(replay_fixture(), ensure_ascii=False))


if __name__ == "__main__":
    main()
