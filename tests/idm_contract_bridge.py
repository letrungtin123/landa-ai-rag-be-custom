"""Node ↔ Python IDM contract bridge (spec §15.3) — test-only, never imported by the app.

Reads ``{"stage": ..., "request": {...}, "options": {...}}`` (or ``{"batch": [...]}``
of such messages) as JSON on stdin and writes ``{"status": <http status>,
"response": <json body>, "calls": [...]}`` (or ``{"results": [...]}``) as one JSON
line on stdout. Each stage posts the Node-built request to the real
FastAPI route through ``httpx.ASGITransport`` with ``app.main.generate_content``
replaced by an offline fake, so the request is validated by the production
request models and the real IDM code runs in-process. No network is used.

Stages:
* ``course_skeleton`` — golden W1/W1-reduce/W2/W4 answers (``tests/idm_golden.py``).
* ``chapter_shard``  — golden W3/W4 module answer for the requested module
  (``tests/idm_golden_module.py``); ``options.hold_lessons`` turns the practice
  of the listed lessons into a held practice without a practice component.
* ``unit``           — deterministic W5 writer answers generated from the
  request's contract and brief (or ``options.writer`` / ``options.repair`` answers
  given by the caller); the W6 judge always passes.
* ``golden_unit``    — the golden IDM unit requests of ``tests/test_idm_storyboard.py`` with a valid
  writer answer each (``worksheet``: html practice worksheet + problem; ``escalate``: problem +
  la_faq; ``severity``: html explanation + problem).
* ``acceptance``     — the Python mirror of Node's unit acceptance
  (``app.idm.node_acceptance``) for each ``request.units[i]`` against ``request.contract``:
  the first finding as ``{"check", "code", "path"}`` or ``null`` (Node parity test).
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from typing import Any
from unittest.mock import patch

import httpx
from fastapi import HTTPException

from app import main
from app.api import deps as api_deps
from app.idm.node_acceptance import node_acceptance_findings
from app.idm.storyboard import acceptance_context, parse_brief
from app.lesson_author_orchestration_v2_provider import UnitGenerationContractV2
from app.schemas.common import AiUsage
from tests import idm_golden as g
from tests import idm_golden_module as gm

ROUTES = {
    "course_skeleton": "/v1/lesson-author/orchestration-v2/course-skeleton",
    "chapter_shard": "/v1/lesson-author/orchestration-v2/chapter-shard",
    "unit": "/v1/lesson-author/orchestration-v2/unit",
}
CRITERIA = ("Q1_support_sufficient", "Q2_not_copied", "Q3_practice_complete", "Q4_feedback_teaches",
            "Q5_grounded_criteria", "Q6_alignment", "Q7_cognitive_load", "Q8_language", "Q9_traceability")
_STEP = re.compile(r"^(?:bước|step)\s*\d+\s*[:.)-]\s*", re.IGNORECASE)
_FILLER = ("Nắm chắc điểm này giúp nhân viên xử lý khiếu nại nhất quán và giải thích rõ ràng cho khách hàng "
           "trong từng tình huống.")


def _usage(prompt: str, text: str) -> Any:
    return AiUsage(inputTokens=len(prompt) // 4, outputTokens=len(text) // 4,
                        totalTokens=len(prompt) // 4 + len(text) // 4)


class GoldenGenerate:
    """Wrap a golden ``FakeIdmProvider`` as ``generate_content``."""

    def __init__(self, provider: g.FakeIdmProvider) -> None:
        self.provider = provider
        self.calls: list[str] = []

    async def __call__(self, api_key: str, model: str, prompt: str, **options: Any) -> tuple[str, Any]:
        self.calls.append(options["response_schema"].__name__)
        text, _usage_object = await self.provider(api_key, model, prompt, **options)
        return text, _usage(prompt, text)


# --- W5 writer answers ----------------------------------------------------------------------------
def _html(plan: dict[str, Any], slot: dict[str, Any], texts: dict[str, str]) -> dict[str, Any]:
    keys = plan["source_fact_ids"] or plan["supporting_evidence_fact_ids"]
    lines = [texts[key].strip().rstrip(".") for key in keys if key in texts]
    blocks = [{"kind": "paragraph", "text": f"{line}. {_FILLER}", "items": [], "rows": []} for line in lines]
    sections = [{"heading": slot["title"] if index == 0 else f"{slot['title']} ({index + 1})",
                 "learning_block_ids": [], "blocks": blocks[start:start + 12]}
                for index, start in enumerate(range(0, len(blocks), 12))]
    return {"title": slot["title"], "selection_rationale": "Giải thích phần cần thiết cho Must Do của mục.",
            "covered_source_fact_ids": list(plan["source_fact_ids"]), "html": None,
            "semantic_content": {"sections": sections}}


def _problem(plan: dict[str, Any], slot: dict[str, Any], texts: dict[str, str]) -> dict[str, Any]:
    practice = slot.get("practice") or {}
    criterion = next((texts[key] for key in practice.get("criteria_fact_keys", []) if key in texts), None) \
        or next((texts[key] for key in [*plan["source_fact_ids"], *plan["supporting_evidence_fact_ids"]]
                 if key in texts), slot["title"])
    criterion = criterion.strip().rstrip(".")
    situation = practice.get("context_input") or "Một tình huống khiếu nại cụ thể"
    return {
        "title": slot["title"], "selection_rationale": "Luyện tập áp dụng tiêu chí của Must Do.",
        "covered_source_fact_ids": list(plan["source_fact_ids"]), "problem_type": "multiple_choice",
        "question": f"{situation}: nhân viên cần quyết định cách xử lý tiếp theo. Cách làm nào đúng với quy trình?",
        "choices": [
            {"text": "Xử lý theo cảm tính mà không đối chiếu tiêu chí của quy trình", "correct": False},
            {"text": f"Làm đúng theo tiêu chí: {criterion[:300]}", "correct": True},
            {"text": "Chuyển ngay cho bộ phận khác mà không ghi nhận thông tin", "correct": False},
        ],
        "answer": None, "tolerance": None,
        "explanation": (f"Tiêu chí: {criterion[:300]}. A — sai vì bỏ qua tiêu chí của quy trình; "
                        "B — đúng vì làm đúng tiêu chí đã học; C — sai vì chuyển hồ sơ khi chưa ghi nhận thông tin."),
    }


def _faq(plan: dict[str, Any], slot: dict[str, Any], _texts: dict[str, str], brief: dict[str, Any]) -> dict[str, Any]:
    objective = brief["lesson_objective"].rstrip(".")
    return {"title": slot["title"], "selection_rationale": "Làm rõ những điểm dễ nhầm.",
            "covered_source_fact_ids": list(plan["source_fact_ids"]),
            "items": [
                {"question": "Lỗi nào hay gặp nhất khi thực hiện phần này?",
                 "answer": ("Bỏ qua bước đối chiếu tiêu chí. Hãy luôn kiểm tra tiêu chí trước khi "
                            f"{objective.lower()}.")},
                {"question": "Khi chưa chắc chắn thì nên làm gì?",
                 "answer": "Hỏi trưởng nhóm và ghi nhận đầy đủ thông tin trước khi trả lời khách hàng."},
            ]}


def _sortable(plan: dict[str, Any], slot: dict[str, Any], texts: dict[str, str]) -> dict[str, Any]:
    keys = [*plan["source_fact_ids"], *plan["supporting_evidence_fact_ids"]]
    steps = [_STEP.sub("", texts[key]).strip().rstrip(".") for key in keys
             if key in texts and _STEP.match(texts[key].strip())]
    return {"title": slot["title"], "selection_rationale": "Luyện tập trình tự các bước.",
            "covered_source_fact_ids": list(plan["source_fact_ids"]),
            "question_text": "Sắp xếp các bước theo đúng trình tự thực hiện.",
            "items": [{"text": step[:500]} for step in steps[:10]]}


def writer_answer(request: dict[str, Any]) -> dict[str, Any]:
    contract = request["unit_contract"]
    brief = contract["idm_unit_brief"]
    texts = {fact["fact_key"]: fact["fact_text"] for fact in contract["source_facts"]}
    texts.update({fact["fact_key"]: fact["fact_text"] for fact in brief["lesson_context_facts"]})
    slots = {}
    for index, (plan, slot) in enumerate(zip(contract["component_plan"], brief["components"], strict=True)):
        kind = plan["type"]
        if kind == "html":
            slots[f"c{index}"] = _html(plan, slot, texts)
        elif kind == "problem":
            slots[f"c{index}"] = _problem(plan, slot, texts)
        elif kind == "la_faq":
            slots[f"c{index}"] = _faq(plan, slot, texts, brief)
        elif kind == "la_sortable":
            slots[f"c{index}"] = _sortable(plan, slot, texts)
        else:
            raise ValueError(f"bridge has no writer answer for {kind}")
    return {"components": slots}


class UnitGenerate:
    """Writer answers from the contract (or the caller's), a passing judge, and the caller's repair answers."""

    def __init__(self, request: dict[str, Any], options: dict[str, Any] | None = None) -> None:
        self.request = request
        self.writer: list[Any] = list((options or {}).get("writer") or [])
        self.repair: list[Any] = list((options or {}).get("repair") or [])
        self.calls: list[str] = []

    async def __call__(self, api_key: str, model: str, prompt: str, **options: Any) -> tuple[str, Any]:
        name = options["response_schema"].__name__
        self.calls.append(name)
        if name.startswith("StagedInstancePayloadUnit"):
            text = json.dumps(self.writer.pop(0) if self.writer else writer_answer(self.request), ensure_ascii=False)
        elif name.startswith("StagedMultiRepair"):
            if not self.repair:
                raise HTTPException(status_code=503, detail={"code": "AI_PROVIDER_UNAVAILABLE",
                                                                  "message": "bridge has no repair answer"})
            text = json.dumps(self.repair.pop(0), ensure_ascii=False)
        elif name.startswith("IdmJudgeResponseV1"):
            text = json.dumps({"verdict": "pass", "findings": [
                {"criterion": criterion, "severity": "pass", "component_index": None, "witness": "ok"}
                for criterion in CRITERIA]})
        else:
            raise RuntimeError(f"bridge does not answer {name}")
        return text, _usage(prompt, text)


def _held_module(module_key: str, hold_lessons: list[str]) -> dict[str, Any]:
    response: dict[str, Any] = json.loads(json.dumps(gm.MODULE_RESPONSES[module_key], ensure_ascii=False))
    for lesson in response["lessons"]:
        if lesson["lesson_key"] not in hold_lessons:
            continue
        for practice in lesson["practice_tasks"]:
            practice["hold"] = True
            practice["hold_question"] = "SME xác nhận tiêu chí đúng/sai của tình huống này trước khi dựng câu hỏi."
        for unit in lesson["units"]:
            unit["components"] = [component for component in unit["components"] if component["role"] != "practice"]
            for position, component in enumerate(unit["components"], start=1):
                component["component_index"] = position
    return response


def golden_units() -> dict[str, Any]:
    """Golden unit requests with a valid writer answer each (``severity``: html explain + problem)."""

    from tests import test_idm_storyboard as sb  # noqa: PLC0415 - test fixtures, loaded only for this stage

    units = {"worksheet": (sb.worksheet_body, sb.worksheet_writer), "escalate": (sb.escalate_body, sb.escalate_writer),
             "severity": (sb.severity_body, sb.severity_writer)}
    result: dict[str, Any] = {}
    for name, (body_of, writer_of) in units.items():
        body = body_of()
        result[name] = {"request": body, "writer": writer_of(body)}
    return result


def acceptance(request: dict[str, Any]) -> list[dict[str, Any] | None]:
    contract = UnitGenerationContractV2.model_validate(request["contract"])
    context = acceptance_context(contract, parse_brief(contract))
    results: list[dict[str, Any] | None] = []
    for item in request["units"]:
        findings = node_acceptance_findings(item["unit"], context, provider_validated=item["provider_validated"])
        results.append(None if not findings else {"check": findings[0].check, "code": findings[0].code,
                                                  "path": findings[0].path})
    return results


async def run(stage: str, request: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
    if stage == "golden_unit":
        return {"status": 200, "response": golden_units(), "calls": []}
    if stage == "acceptance":
        return {"status": 200, "response": {"results": acceptance(request)}, "calls": []}
    if stage == "course_skeleton":
        fake: Any = GoldenGenerate(g.golden_provider())
    elif stage == "chapter_shard":
        module_key = request["idm_module_context"]["module"]["module_key"]
        answer = _held_module(module_key, list(options.get("hold_lessons", [])))
        fake = GoldenGenerate(g.FakeIdmProvider({"IdmW3W4ModuleResponseV1": [answer]}))
    elif stage == "unit":
        fake = UnitGenerate(request, options)
    else:
        raise ValueError(f"unknown stage {stage}")
    main.app.dependency_overrides[api_deps.require_internal_token] = lambda: None
    try:
        transport = httpx.ASGITransport(app=main.app)
        with patch("app.services.provider.generate_content", fake):
            async with httpx.AsyncClient(transport=transport, base_url="http://bridge") as client:
                reply = await client.post(ROUTES[stage], json=request)
    finally:
        main.app.dependency_overrides.pop(api_deps.require_internal_token, None)
    return {"status": reply.status_code, "response": reply.json(), "calls": fake.calls}


async def run_batch(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [await run(item["stage"], item["request"], item.get("options") or {}) for item in messages]


def main_cli() -> None:
    message = json.loads(sys.stdin.read())
    if "batch" in message:
        result: dict[str, Any] = {"results": asyncio.run(run_batch(message["batch"]))}
    else:
        result = asyncio.run(run(message["stage"], message["request"], message.get("options") or {}))
    sys.stdout.write("\n" + json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main_cli()
