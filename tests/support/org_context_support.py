"""A synthetic organization profile for the organization context tests (ORG_CONTEXT spec §2; not customer data).

Its texts are chosen so the tests can tell which view of the profile reached a prompt: every field carries words that
appear nowhere in the golden source (``tests/idm_golden.py``).
"""

from __future__ import annotations

import copy
import json
from typing import Any

from app.hashing import js_canonical_hash
from app.infra.gemini.usage import AiUsage

PROFILE: dict[str, Any] = {
    "organization_summary": ("Công ty Hải Đăng Logistics vận hành kho lạnh và giao hàng chặng cuối tại miền Trung.\n"
                             "Trung tâm điều phối làm việc theo ba ca."),
    "products_services": ["Gói Giao Nhanh Hải Đăng: giao trong ngày nội thành", "Kho lạnh Sơn Trà"],
    "learner_groups": [{"name": "Điều phối viên ca đêm", "description": "Phân tuyến xe và trả lời tài xế."},
                       {"name": "Nhân viên kho lạnh", "description": ""}],
    "glossary": [{"term": "Phiếu HD-07", "meaning": "Biên bản bàn giao hàng lạnh."},
                 {"term": "Tuyến xanh", "meaning": "Tuyến giao ưu tiên của khách hợp đồng năm."}],
    "key_processes": ["Kiểm nhiệt độ thùng hàng trước khi xuất kho."],
    "culture_tone": "Thân thiện, xưng hô anh/chị, nói ngắn gọn với tài xế.",
    "design_conventions": ["Mỗi bài mở đầu bằng một tình huống ở trung tâm điều phối."],
}
# What the distillation drafts from an answer of ``PROFILE``: no learner groups (owner 2026-10-22, contract K3).
DISTILLED: dict[str, Any] = {**PROFILE, "learner_groups": []}


DRAFT_SCHEMA = "IdmOrgProfileDraftV1"
# The reduce prompt's data block (the preamble of every distillation prompt names the tag).
PARTIALS = "\n<ORG_PARTIAL_PROFILES>\n"


def profile(**changes: Any) -> dict[str, Any]:
    return {**copy.deepcopy(PROFILE), **changes}


def org_context(value: dict[str, Any] | None = None, *, version: int = 3) -> dict[str, Any]:
    """The wire object Node freezes at launch: the profile and its ``canonicalJson`` hash."""

    body = copy.deepcopy(PROFILE) if value is None else value
    return {"profile_version": version, "profile_hash": js_canonical_hash(body), "profile": body}


class FakeContextPool:
    """Learned chunks of one tenant's knowledge base (``documents``: document id -> chunk texts in order), answered
    by the SQL ``app.repositories.org_context`` sends; another tenant or knowledge base gets no rows."""

    def __init__(self, documents: dict[str, list[str]], *, tenant: str, kb: str) -> None:
        self.documents, self.tenant, self.kb = documents, tenant, kb
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def owns(self, tenant: Any, kb: Any) -> bool:
        return (str(tenant), str(kb)) == (self.tenant, self.kb)

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        text = " ".join(sql.split())
        self.calls.append((text, args))
        if "FROM rag_document_indexes" in text:
            tenant, kb, ids = args
            return [{"index_id": f"idx-{document}", "document_id": document} for document in sorted(ids)
                    if self.owns(tenant, kb) and document in self.documents]
        tenant, kb, document, index = args[:4]
        texts = self.documents.get(document, []) if self.owns(tenant, kb) and index == f"idx-{document}" else []
        if "char_length" in text:
            after, limit = args[4], args[5]
            return [{"chunk_no": number, "chars": len(value)} for number, value in enumerate(texts)
                    if number > after][:limit]
        if "c.chunk_no=ANY" in text:
            return [{"chunk_no": number, "content": texts[number]} for number in sorted(args[4])
                    if number < len(texts)]
        raise AssertionError(f"unexpected SQL {text}")

    async def fetchrow(self, sql: str, *args: Any) -> Any:
        raise AssertionError("not used")

    async def fetchval(self, sql: str, *args: Any) -> Any:
        raise AssertionError("not used")

    async def execute(self, sql: str, *args: Any) -> str:
        raise AssertionError("the distillation never writes")

    async def executemany(self, command: str, args: Any) -> None:
        raise AssertionError("the distillation never writes")


class FakeProfileProvider:
    """``generate_content`` replacement for the distillation: map and reduce answers (or exceptions) in call order
    per kind; the last answer of a kind repeats."""

    def __init__(self, *, maps: list[Any] | None = None, reduces: list[Any] | None = None) -> None:
        self.maps, self.reduces = list(maps or [profile()]), list(reduces or [])
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, api_key: str, model: str, prompt: str, **options: Any) -> tuple[str, AiUsage]:
        schema = options["response_schema"].model_json_schema().get("title")
        if schema != DRAFT_SCHEMA:
            raise AssertionError(f"unexpected schema {schema}")
        kind = "reduce" if PARTIALS in prompt else "map"
        self.calls.append({"kind": kind, "prompt": prompt, "model": model, **options})
        queue = self.reduces if kind == "reduce" else self.maps
        answer = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(answer, Exception):
            raise answer
        text = answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
        return text, AiUsage(inputTokens=1_000, outputTokens=200, totalTokens=1_300)
