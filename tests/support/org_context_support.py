"""A synthetic organization profile for the organization context tests (ORG_CONTEXT spec §2; not customer data).

Its texts are chosen so the tests can tell which view of the profile reached a prompt: every field carries words that
appear nowhere in the golden source (``tests/idm_golden.py``).
"""

from __future__ import annotations

import copy
from typing import Any

from app.hashing import js_canonical_hash

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


def profile(**changes: Any) -> dict[str, Any]:
    return {**copy.deepcopy(PROFILE), **changes}


def org_context(value: dict[str, Any] | None = None, *, version: int = 3) -> dict[str, Any]:
    """The wire object Node freezes at launch: the profile and its ``canonicalJson`` hash."""

    body = copy.deepcopy(PROFILE) if value is None else value
    return {"profile_version": version, "profile_hash": js_canonical_hash(body), "profile": body}
