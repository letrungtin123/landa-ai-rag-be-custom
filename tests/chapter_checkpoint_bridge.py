"""Synthetic subprocess boundary for Node tests. Forbids DB and real provider IO."""
import asyncio
import json
import sys
from unittest.mock import AsyncMock, patch
from copy import deepcopy

from app import main
from tests.test_chapter_checkpoint import fixture, checkpoint_result, provider_result
from tests.test_staged_instance_output import checkpoint_instance_fixture
from tests.test_checkpoint_component_quality_repair import instance_wire


def run(payload):
    if payload["action"] in ("instance_repair", "coverage_repair"):
        request, unit, _, manifest = checkpoint_instance_fixture()
        broken = instance_wire(unit)
        broken["components"]["c2"].pop("edges")
        provider = AsyncMock(side_effect=[(json.dumps(broken), main.AiUsage()),
            (json.dumps({"components": [{"component_index": 2, "edges": unit["components"][2]["edges"]}]}), main.AiUsage())])
        if payload["action"] == "coverage_repair":
            from tests.test_checkpoint_coverage_repair import coverage_fixture
            request, unit, broken_unit, _, manifest, delta = coverage_fixture()
            provider = AsyncMock(side_effect=[(json.dumps(instance_wire(broken_unit)), main.AiUsage()),
                                              (json.dumps(delta), main.AiUsage())])
        def forbidden(*_args, **_kwargs):
            raise AssertionError("REAL_PROVIDER_OR_DATABASE_ACCESS_FORBIDDEN")
        events = []
        with patch("asyncpg.create_pool", forbidden), patch("app.main.generate_content", provider):
            generated = asyncio.run(checkpoint_result(request, manifest, events))
            completed = request.model_copy(update={"checkpoint_action": "validate_chapter", "checkpoint_unit_index": None,
                "checkpoint_units": [main.ChapterCheckpointUnit(unit_index=0, unit=deepcopy(generated["unit"]))]})
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
    request = main.RagLessonAuthorCheckpointRequest.model_validate(data)
    index = request.checkpoint_unit_index
    provider = provider_result(units[index]) if index is not None else None
    def forbidden(*_args, **_kwargs):
        raise AssertionError("REAL_PROVIDER_OR_DATABASE_ACCESS_FORBIDDEN")
    with patch("asyncpg.create_pool", forbidden), patch("app.main.generate_content", provider or forbidden):
        result = asyncio.run(checkpoint_result(request, manifest))
    return {"result": result, "provider_calls": provider.await_count if provider else 0}


if __name__ == "__main__":
    print(json.dumps(run(json.load(sys.stdin))))
