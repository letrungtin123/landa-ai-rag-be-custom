"""Synthetic subprocess boundary for Node tests. Forbids DB and real provider IO."""
import asyncio
import json
import sys
from copy import deepcopy
from unittest.mock import AsyncMock, patch

from app.lesson_author_checkpoint import ChapterCheckpointUnit
from app.schemas.common import AiUsage
from app.schemas.lesson_author import RagLessonAuthorCheckpointRequest
from tests.test_chapter_checkpoint import checkpoint_result, fixture, provider_result
from tests.test_checkpoint_component_quality_repair import instance_wire
from tests.test_staged_instance_output import checkpoint_instance_fixture


def run(payload):
    if payload["action"] in ("instance_repair", "coverage_repair", "duplicate_claim_repair", "null_claim_repair"):
        request, unit, _, manifest = checkpoint_instance_fixture()
        broken = instance_wire(unit)
        broken["components"]["c2"].pop("edges")
        provider = AsyncMock(side_effect=[(json.dumps(broken), AiUsage()),
            (json.dumps({"components": [{"component_index": 2, "edges": unit["components"][2]["edges"]}]}), AiUsage())])
        if payload["action"] == "coverage_repair":
            from tests.test_checkpoint_coverage_repair import coverage_fixture
            request, unit, broken_unit, _, manifest, delta = coverage_fixture()
            provider = AsyncMock(side_effect=[(json.dumps(instance_wire(broken_unit)), AiUsage()),
                                              (json.dumps(delta), AiUsage())])
        if payload["action"] in ("duplicate_claim_repair", "null_claim_repair"):
            from tests.test_coverage_claim_recovery import repaired_payload
            from tests.test_staged_ordered_writer import uat_fixture
            request, unit, _, manifest = uat_fixture()
            # The dense Python-only fixture replaces the unit scope; restore
            # the same approved treatment reasons as checkpoint_instance_fixture
            # before exercising Node's independent pedagogical gate.
            architecture = request.blueprint_architecture.model_dump()
            plans = architecture["lessons"][0]["units"][0]["component_plan"]
            plans[2]["reason_code"] = "RELATIONSHIP_VISUALIZATION"
            plans[3]["reason_code"] = "FAQ_ANTICIPATED_QUESTIONS"
            request = request.model_copy(update={"blueprint_architecture": type(request.blueprint_architecture).model_validate(architecture)})
            broken = instance_wire(unit)
            broken["components"]["c0"]["covered_source_fact_ids"] = (
                unit["source_fact_ids"] * 2 if payload["action"] == "duplicate_claim_repair" else None)
            provider = AsyncMock(side_effect=[(json.dumps(broken), AiUsage()),
                                              (json.dumps(repaired_payload(unit)), AiUsage())])
        def forbidden(*_args, **_kwargs):
            raise AssertionError("REAL_PROVIDER_OR_DATABASE_ACCESS_FORBIDDEN")
        events = []
        with patch("asyncpg.create_pool", forbidden), patch("app.services.provider.generate_content", provider):
            generated = asyncio.run(checkpoint_result(request, manifest, events))
            completed = request.model_copy(update={"checkpoint_action": "validate_chapter", "checkpoint_unit_index": None,
                "checkpoint_units": [ChapterCheckpointUnit(unit_index=0, unit=deepcopy(generated["unit"]))]})
            ready = asyncio.run(checkpoint_result(completed, manifest, events))
        return {"request": request.model_dump(), "result": ready, "unit_result": generated,
                "provider_calls": provider.await_count, "events": events}
    request, units, manifest = fixture()
    if payload["action"] == "fixture":
        return {"request": request.model_dump(), "manifest": manifest}
    data = {**request.model_dump(), **payload["request"]}
    data["checkpoint_units"] = data.get("checkpoint_units", [])
    if data["checkpoint_action"] == "validate_chapter":
        data["checkpoint_unit_index"] = None
    request = RagLessonAuthorCheckpointRequest.model_validate(data)
    index = request.checkpoint_unit_index
    provider = provider_result(units[index]) if index is not None else None
    def forbidden(*_args, **_kwargs):
        raise AssertionError("REAL_PROVIDER_OR_DATABASE_ACCESS_FORBIDDEN")
    with patch("asyncpg.create_pool", forbidden), patch("app.services.provider.generate_content", provider or forbidden):
        result = asyncio.run(checkpoint_result(request, manifest))
    return {"result": result, "provider_calls": provider.await_count if provider else 0}


if __name__ == "__main__":
    print(json.dumps(run(json.load(sys.stdin))))
