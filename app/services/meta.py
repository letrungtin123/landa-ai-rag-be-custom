"""Service identity and contract versions reported by ``/v1/meta``."""

from __future__ import annotations

import re
from typing import Any

from app.idm.policy import IDM_CONTRACT_VERSION, IDM_PIPELINE_VERSION, IDM_PROMPT_POLICY_VERSION
from app.lesson_author_orchestration_v2 import ORCHESTRATION_CONTRACT_VERSION
from app.lesson_author_orchestration_v2_provider import ORCHESTRATION_V2_PROVIDER_SCHEMA_PROJECTION_VERSION

API_VERSION = "0.1.0"


SERVICE_NAME = "landa-ai-rag"


# 2: /v1/kb/documents/index accepts source_download_url (backend-signed storage URL).
RAG_INDEX_REQUEST_VERSION = 2


BUILD_SHA_PATTERN = re.compile(r"^[0-9A-Za-z._-]{1,64}$")


def service_contract_versions() -> dict[str, Any]:
    return {
        "orchestration_v2_contract_version": ORCHESTRATION_CONTRACT_VERSION,
        "orchestration_v2_provider_schema_projection": ORCHESTRATION_V2_PROVIDER_SCHEMA_PROJECTION_VERSION,
        "idm_pipeline_version": IDM_PIPELINE_VERSION,
        "idm_contract_version": IDM_CONTRACT_VERSION,
        "idm_prompt_policy_version": IDM_PROMPT_POLICY_VERSION,
        "rag_index_request_version": RAG_INDEX_REQUEST_VERSION,
    }
