"""Offline QC replay of the AI ID pipeline (no network, no provider, no database).

Run from the repository root, with the root on ``sys.path`` and the test environment:

    AI_RAG_ENV=test PYTHONPATH=. .venv-dev/bin/python tools/qc_replay.py <out.json>

Two trees replay the same when their ``overall`` digests are equal (a refactor must keep it).

Replays, with recorded / hand-written provider answers (tests/idm_golden.py and tests/golden/ fixtures):
  1. the golden course design (W1 map/reduce, W2, W4) and its skeleton;
  2. the module design (W3/W4) of every chapter;
  3. every unit of every module through POST /v1/lesson-author/orchestration-v2/unit:
     a. fallback_only (deterministic source-locked content),
     b. writer unavailable (deterministic fallback after a provider failure),
     c. the two units with recorded writer answers (writer + judge),
     d. malformed writer answers that make the shared binder raise (TypeError/KeyError probes);
  4. every recorded route snapshot case (tests/snapshots/routes), its actual replayed response.

Writes every generated payload (volatile timings removed) and one sha256 per item.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx

VOLATILE = {"duration_ms", "elapsed_ms", "timestamp_utc", "node_durations_ms"}


def normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: "<volatile>" if key in VOLATILE else normalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize(item) for item in value]
    return value


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


async def post_unit(app: Any, body: dict[str, Any], provider: Any) -> dict[str, Any]:
    from app.core.config import settings

    with patch("app.services.provider.generate_content", provider), patch.object(settings, "idm_judge_mode", "observe"):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://replay") as client:
            reply = await client.post("/v1/lesson-author/orchestration-v2/unit", json=body)
    return {"status": reply.status_code, "body": normalize(reply.json())}


def main() -> None:
    out = Path(sys.argv[1])
    from app import main as app_main
    from app.api import deps as api_deps
    from app.idm.module_design import run_idm_module_design
    from tests import idm_golden as g
    from tests.contract.test_route_snapshots import replay, snapshot_files
    from tests.golden import idm_golden_module as gm
    from tests.golden import idm_golden_unit as gu
    from tests.support.idm_test_support import _design_json, golden_design, golden_shard, make_runtime
    from tests.unit.idm.test_idm_storyboard import (
        JUDGE,
        WRITER,
        FakeGenerate,
        escalate_body,
        escalate_writer,
        judge,
        severity_body,
        severity_writer,
        unavailable,
    )

    app_main.app.dependency_overrides[api_deps.require_internal_token] = lambda: None
    items: dict[str, Any] = {}

    design_json, skeleton_json = _design_json()
    items["course_design"] = {"idm": json.loads(design_json), "skeleton": json.loads(skeleton_json)}
    design, skeleton = golden_design()

    for chapter_index, module in enumerate(design.modules):
        context, plan, facts = gm.module_context(design, skeleton, chapter_index, g.source_facts())
        provider = g.FakeIdmProvider({"IdmW3W4ModuleResponseV1": [copy.deepcopy(gm.MODULE_RESPONSES[module.module_key])]})
        result = asyncio.run(run_idm_module_design(context=context, skeleton=skeleton, plan=plan, facts=facts,
                                                   runtime=make_runtime(provider)))
        items[f"module/{module.module_key}"] = normalize(result)
        shard = golden_shard(chapter_index)
        for lesson_position, lesson in enumerate(shard.lessons):
            for unit_position, _unit in enumerate(lesson.units):
                body = gu.build_unit_request(design, shard, chapter_index=chapter_index, lesson_position=lesson_position,
                                             unit_position=unit_position, facts=g.source_facts())
                name = f"unit/{lesson.lesson_key}/u{unit_position + 1}"
                fallback_only = {**copy.deepcopy(body), "fallback_only": True}
                items[f"{name}/fallback_only"] = asyncio.run(post_unit(app_main.app, fallback_only, FakeGenerate()))
                items[f"{name}/writer_unavailable"] = asyncio.run(post_unit(
                    app_main.app, copy.deepcopy(body), FakeGenerate(**{WRITER: [unavailable()] * 4})))

    for name, body, writer in (("escalate", escalate_body(), escalate_writer), ("severity", severity_body(),
                                                                                   severity_writer)):
        items[f"recorded/{name}"] = asyncio.run(post_unit(
            app_main.app, body, FakeGenerate(**{WRITER: [writer(body)], JUDGE: [judge()]})))

    # Malformed answers the schema-constrained provider should never send, but which reach the shared binder.
    body = severity_body()
    good = severity_writer(body)
    probes: dict[str, Any] = {
        "json_list": [],
        "components_list": {"components": [good["components"]["c0"], good["components"]["c1"]]},
        "component_null": {"components": {"c0": None, "c1": good["components"]["c1"]}},
        "component_string": {"components": {"c0": "text", "c1": good["components"]["c1"]}},
        "choices_null": {"components": {**good["components"], "c1": {**good["components"]["c1"], "choices": None}}},
        "semantic_null": {"components": {**good["components"],
                                         "c0": {**good["components"]["c0"], "semantic_content": None}}},
        "missing_slot": {"components": {"c1": good["components"]["c1"]}},
    }
    for name, answer in probes.items():
        provider = FakeGenerate(**{WRITER: [copy.deepcopy(answer)] * 4, JUDGE: [judge()] * 2})
        items[f"probe/{name}"] = asyncio.run(post_unit(app_main.app, copy.deepcopy(body), provider))
        items[f"probe/{name}"]["provider_calls"] = provider.names

    # Route snapshots replay with the real internal-auth dependency (their own token).
    app_main.app.dependency_overrides.pop(api_deps.require_internal_token, None)
    for path in snapshot_files():
        case = json.loads(path.read_text(encoding="utf-8"))
        items[f"snapshot/{path.stem}"] = replay(case)
    digests = {name: digest(value) for name, value in items.items()}
    out.write_text(json.dumps({"digests": digests, "overall": digest(digests), "items": items}, ensure_ascii=False,
                              indent=1, sort_keys=True, default=str), encoding="utf-8")
    print(f"items={len(items)} overall={digest(digests)}")


if __name__ == "__main__":
    main()
