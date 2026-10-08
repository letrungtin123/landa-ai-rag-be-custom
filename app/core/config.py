from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

APP_ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)

Environment = Literal["development", "test", "production"]
AuthMode = Literal["token", "hmac", "token_or_hmac"]


def _read_environment(*, emit_legacy_warning: bool = True) -> Environment:
    explicit = os.getenv("AI_RAG_ENV", "").strip().lower()
    legacy = os.getenv("NODE_ENV", "").strip().lower()
    value = explicit or legacy or "development"
    if not explicit and legacy and emit_legacy_warning:
        logger.warning(
            "ai_rag_legacy_environment_variable variable=NODE_ENV replacement=AI_RAG_ENV"
        )
    if value not in {"development", "test", "production"}:
        raise RuntimeError("AI_RAG_ENV must be development, test, or production.")
    return value  # type: ignore[return-value]


def load_runtime_dotenv(
    path: Path,
    *,
    override: bool = False,
    allow_nul_sanitization: bool = False,
) -> bool:
    """Load a local dotenv file without ever logging values."""
    if allow_nul_sanitization:
        raise RuntimeError("NUL sanitization is no longer supported for AI RAG config.")
    return load_dotenv(path, override=override)


def load_runtime_environment(
    *,
    app_root: Path = APP_ROOT,
    repo_root: Path | None = None,
) -> Environment:
    """Load only this service's local dotenv; production uses process env only."""
    del repo_root  # Compatibility argument; no external repository is read.
    initial = _read_environment()
    if initial in {"production", "test"}:
        return initial

    load_runtime_dotenv(app_root / ".env", override=False)
    loaded = _read_environment(emit_legacy_warning=False)
    if loaded == "production":
        raise RuntimeError("AI_RAG_ENV=production must be supplied by the process environment.")
    return loaded


_environment = load_runtime_environment()


class Settings(BaseSettings):
    """Validated process configuration for the standalone AI RAG service."""

    model_config = SettingsConfigDict(
        env_file=None,
        extra="ignore",
        case_sensitive=False,
        validate_default=True,
        populate_by_name=True,
    )

    environment: Environment = Field(default=_environment, validation_alias="AI_RAG_ENV")
    host: str = Field(default="127.0.0.1", validation_alias="AI_RAG_HOST", min_length=1, max_length=255)
    port: int = Field(default=8010, validation_alias="AI_RAG_PORT", ge=1, le=65_535)
    workers: int = Field(default=1, validation_alias="AI_RAG_WORKERS", ge=1, le=64)
    proxy_headers: bool = Field(default=True, validation_alias="AI_RAG_PROXY_HEADERS")
    forwarded_allow_ips: str = Field(default="127.0.0.1", validation_alias="AI_RAG_FORWARDED_ALLOW_IPS")
    shutdown_grace_seconds: int = Field(
        default=60,
        validation_alias="AI_RAG_SHUTDOWN_GRACE_SECONDS",
        ge=1,
        le=600,
    )
    keep_alive_timeout_seconds: int = Field(
        default=5,
        validation_alias="AI_RAG_KEEP_ALIVE_TIMEOUT_SECONDS",
        ge=1,
        le=600,
    )
    readiness_db_timeout_ms: int = Field(
        default=1_000,
        validation_alias="AI_RAG_READINESS_DB_TIMEOUT_MS",
        ge=100,
        le=10_000,
    )

    # Concurrency: bounded per workload so one heavy request cannot starve others.
    max_concurrent_provider_calls: int = Field(
        default=8,
        validation_alias="AI_RAG_MAX_CONCURRENT_PROVIDER_CALLS",
        ge=1,
        le=256,
    )
    max_concurrent_index_jobs: int = Field(
        default=2,
        validation_alias="AI_RAG_MAX_CONCURRENT_INDEX_JOBS",
        ge=1,
        le=64,
    )
    # Python CPU work does not parallelise across threads; a small pool keeps
    # the event loop responsive without oversubscribing the GIL.
    cpu_workers: int = Field(
        default_factory=lambda: max(1, min(4, (os.cpu_count() or 2) - 1)),
        validation_alias="AI_RAG_CPU_WORKERS",
        ge=1,
        le=64,
    )
    limiter_acquire_timeout_ms: int = Field(
        default=30_000,
        validation_alias="AI_RAG_LIMITER_ACQUIRE_TIMEOUT_MS",
        ge=100,
        le=600_000,
    )
    extraction_executor: Literal["thread", "process"] = Field(
        default="thread",
        validation_alias="AI_RAG_EXTRACTION_EXECUTOR",
    )

    # Provider retry policy (5xx always; 429 only with a short server retry hint).
    provider_max_attempts: int = Field(
        default=2,
        validation_alias="AI_RAG_PROVIDER_MAX_ATTEMPTS",
        ge=1,
        le=5,
    )
    provider_retry_base_ms: int = Field(
        default=1_500,
        validation_alias="AI_RAG_PROVIDER_RETRY_BASE_MS",
        ge=0,
        le=30_000,
    )
    provider_retry_max_ms: int = Field(
        default=8_000,
        validation_alias="AI_RAG_PROVIDER_RETRY_MAX_MS",
        ge=0,
        le=60_000,
    )
    # Cumulative wait on per-minute provider rate limits inside one IDM provider call (429 with
    # RetryInfo / RPM / TPM). Legacy calls keep the single short retry above.
    provider_rate_limit_max_wait_ms: int = Field(
        default=60_000,
        validation_alias="AI_RAG_PROVIDER_RATE_LIMIT_MAX_WAIT_MS",
        ge=0,
        le=300_000,
    )

    # Overall per-request deadlines; keep them below the backend HTTP timeout.
    chat_deadline_ms: int = Field(
        default=180_000,
        validation_alias="AI_RAG_CHAT_DEADLINE_MS",
        ge=1_000,
        le=3_600_000,
    )
    index_deadline_ms: int = Field(
        default=585_000,
        validation_alias="AI_RAG_INDEX_DEADLINE_MS",
        ge=1_000,
        le=3_600_000,
    )
    lesson_author_deadline_ms: int = Field(
        default=585_000,
        validation_alias="AI_RAG_LESSON_AUTHOR_DEADLINE_MS",
        ge=1_000,
        le=3_600_000,
    )

    # IDM pipeline (spec §11.5). The judge observes by default and never blocks.
    idm_judge_mode: Literal["off", "observe", "repair"] = Field(
        default="observe",
        validation_alias="AI_RAG_IDM_JUDGE_MODE",
    )
    idm_w1_parallelism: int = Field(default=4, validation_alias="AI_RAG_IDM_W1_PARALLELISM", ge=1, le=8)
    idm_max_sections: int = Field(default=16, validation_alias="AI_RAG_IDM_MAX_SECTIONS", ge=4, le=16)
    idm_provider_call_timeout_ms: int = Field(
        default=180_000,
        validation_alias="AI_RAG_IDM_PROVIDER_CALL_TIMEOUT_MS",
        ge=30_000,
        le=300_000,
    )

    database_url: str = Field(default="", validation_alias="DATABASE_URL")
    supabase_url: str = Field(default="", validation_alias="SUPABASE_URL")
    supabase_service_key: str = Field(default="", validation_alias="SUPABASE_SERVICE_KEY")
    supabase_storage_bucket: str = Field(
        default="landa-storage",
        validation_alias="SUPABASE_STORAGE_BUCKET",
        min_length=1,
        max_length=255,
    )

    auth_mode: AuthMode = Field(default="token_or_hmac", validation_alias="AI_RAG_AUTH_MODE")
    service_token: str = Field(default="", validation_alias="AI_RAG_SERVICE_TOKEN")
    service_hmac_secrets: str = Field(default="", validation_alias="AI_RAG_SERVICE_HMAC_SECRETS")
    auth_clock_skew_seconds: int = Field(
        default=300,
        validation_alias="AI_RAG_AUTH_CLOCK_SKEW_SECONDS",
        ge=30,
        le=900,
    )
    auth_replay_ttl_seconds: int = Field(
        default=600,
        validation_alias="AI_RAG_AUTH_REPLAY_TTL_SECONDS",
        ge=60,
        le=3_600,
    )

    max_request_bytes: int = Field(
        default=16 * 1024 * 1024,
        validation_alias="AI_RAG_MAX_REQUEST_BYTES",
        ge=1_024,
        le=64 * 1024 * 1024,
    )
    idm_max_request_bytes: int = Field(
        default=24 * 1024 * 1024,
        validation_alias="AI_RAG_IDM_MAX_REQUEST_BYTES",
        ge=1_024,
        le=64 * 1024 * 1024,
    )
    max_document_bytes: int = Field(
        default=50 * 1024 * 1024,
        validation_alias="AI_RAG_MAX_DOCUMENT_BYTES",
        ge=1_024,
        le=512 * 1024 * 1024,
    )
    max_document_pages: int = Field(
        default=1_000,
        validation_alias="AI_RAG_MAX_DOCUMENT_PAGES",
        ge=1,
        le=10_000,
    )
    max_ooxml_uncompressed_bytes: int = Field(
        default=200 * 1024 * 1024,
        validation_alias="AI_RAG_MAX_OOXML_UNCOMPRESSED_BYTES",
        ge=1_024,
        le=2 * 1024 * 1024 * 1024,
    )
    max_ooxml_entries: int = Field(
        default=10_000,
        validation_alias="AI_RAG_MAX_OOXML_ENTRIES",
        ge=1,
        le=100_000,
    )
    max_ooxml_compression_ratio: float = Field(
        default=100.0,
        validation_alias="AI_RAG_MAX_OOXML_COMPRESSION_RATIO",
        ge=1.0,
        le=1_000.0,
    )
    max_xlsx_cells: int = Field(
        default=200_000,
        validation_alias="AI_RAG_MAX_XLSX_CELLS",
        ge=1,
        le=5_000_000,
    )
    libreoffice_timeout_seconds: int = Field(
        default=120,
        validation_alias="AI_RAG_LIBREOFFICE_TIMEOUT_SECONDS",
        ge=5,
        le=600,
    )

    chunk_max_chars: int = Field(default=3_200, validation_alias="AI_RAG_CHUNK_MAX_CHARS")
    chunk_overlap_chars: int = Field(default=350, validation_alias="AI_RAG_CHUNK_OVERLAP_CHARS")
    embedding_batch_size: int = Field(default=32, validation_alias="AI_RAG_EMBEDDING_BATCH_SIZE")
    provider_request_timeout_ms: int = Field(default=60_000, validation_alias="AI_RAG_PROVIDER_REQUEST_TIMEOUT_MS")
    blueprint_provider_request_timeout_ms: int = Field(
        default=300_000,
        validation_alias="AI_RAG_BLUEPRINT_PROVIDER_REQUEST_TIMEOUT_MS",
    )
    staged_lesson_content_provider_timeout_ms: int = Field(
        default=180_000,
        validation_alias="AI_RAG_STAGED_LESSON_CONTENT_PROVIDER_TIMEOUT_MS",
        gt=60_000,
        le=300_000,
    )
    staged_lesson_workflow_timeout_ms: int = Field(
        default=480_000,
        validation_alias="AI_RAG_STAGED_LESSON_WORKFLOW_TIMEOUT_MS",
        gt=0,
        le=480_000,
    )
    course_workflow_max_repair_attempts: int = Field(
        default=2,
        validation_alias="AI_RAG_COURSE_WORKFLOW_MAX_REPAIR_ATTEMPTS",
    )
    lesson_workflow_max_repair_attempts: int = Field(
        default=2,
        validation_alias="AI_RAG_LESSON_WORKFLOW_MAX_REPAIR_ATTEMPTS",
    )
    semantic_review_mode: Literal["off", "observe", "repair"] = Field(
        default="off",
        validation_alias="AI_RAG_SEMANTIC_REVIEW_MODE",
    )
    semantic_review_model: str = Field(default="", validation_alias="AI_RAG_SEMANTIC_REVIEW_MODEL", max_length=255)
    semantic_review_timeout_ms: int = Field(
        default=60_000,
        validation_alias="AI_RAG_SEMANTIC_REVIEW_TIMEOUT_MS",
        ge=1_000,
        le=120_000,
    )
    semantic_review_max_output_tokens: int = Field(
        default=4_096,
        validation_alias="AI_RAG_SEMANTIC_REVIEW_MAX_OUTPUT_TOKENS",
        ge=512,
        le=8_192,
    )
    semantic_review_provider_attempt_cap: int = Field(
        default=8,
        validation_alias="AI_RAG_SEMANTIC_REVIEW_PROVIDER_ATTEMPT_CAP",
        ge=2,
        le=8,
    )
    database_command_timeout_seconds: int = Field(
        default=600,
        validation_alias="AI_RAG_DATABASE_COMMAND_TIMEOUT_SECONDS",
    )
    top_k: int = Field(default=8, validation_alias="AI_RAG_TOP_K")
    max_context_chars: int = Field(default=18_000, validation_alias="AI_RAG_MAX_CONTEXT_CHARS")
    lesson_author_top_k: int = Field(default=24, validation_alias="AI_RAG_LESSON_AUTHOR_TOP_K")
    lesson_author_max_context_chars: int = Field(
        default=32_000,
        validation_alias="AI_RAG_LESSON_AUTHOR_MAX_CONTEXT_CHARS",
    )
    lesson_author_max_chunks_per_document: int = Field(
        default=12,
        validation_alias="AI_RAG_LESSON_AUTHOR_MAX_CHUNKS_PER_DOCUMENT",
    )
    lesson_author_scope_max_chunks: int = Field(
        default=48,
        validation_alias="AI_RAG_LESSON_AUTHOR_SCOPE_MAX_CHUNKS",
    )
    source_coverage_canonical_max_chars: int = Field(
        default=1_000_000,
        validation_alias="AI_RAG_SOURCE_COVERAGE_CANONICAL_MAX_CHARS",
    )
    source_map_architect_context_max_chars: int = Field(
        default=48_000,
        validation_alias="AI_RAG_SOURCE_MAP_ARCHITECT_CONTEXT_MAX_CHARS",
    )
    retrieval_candidate_multiplier: int = Field(
        default=4,
        validation_alias="AI_RAG_RETRIEVAL_CANDIDATE_MULTIPLIER",
    )
    retrieval_min_score: float = Field(default=0.25, validation_alias="AI_RAG_RETRIEVAL_MIN_SCORE")
    retrieval_keyword_min_score: float = Field(
        default=0.50,
        validation_alias="AI_RAG_RETRIEVAL_KEYWORD_MIN_SCORE",
    )
    retrieval_max_chunks_per_document: int = Field(
        default=4,
        validation_alias="AI_RAG_RETRIEVAL_MAX_CHUNKS_PER_DOCUMENT",
    )
    max_user_message_chars: int = Field(default=20_000, validation_alias="AI_RAG_MAX_USER_MESSAGE_CHARS")
    generation_temperature: float = Field(default=0.2, validation_alias="AI_RAG_GENERATION_TEMPERATURE")
    gemini_38_thinking_level: Literal["low", "medium", "high"] = Field(
        default="medium",
        validation_alias="AI_RAG_GEMINI_38_THINKING_LEVEL",
    )

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        return self.environment == "development"


def missing_required_setting_names(value: Settings) -> list[str]:
    required = {
        "DATABASE_URL": value.database_url,
        "SUPABASE_URL": value.supabase_url,
        "SUPABASE_SERVICE_KEY": value.supabase_service_key,
    }
    if (
        value.auth_mode in {"token", "token_or_hmac"}
        and not value.service_token
        and (value.auth_mode == "token" or not value.service_hmac_secrets)
    ):
        required["AI_RAG_SERVICE_TOKEN"] = value.service_token
    if (
        value.auth_mode in {"hmac", "token_or_hmac"}
        and not value.service_hmac_secrets
        and (value.auth_mode == "hmac" or not value.service_token)
    ):
        required["AI_RAG_SERVICE_HMAC_SECRETS"] = value.service_hmac_secrets
    return sorted(name for name, configured in required.items() if not configured)


def validate_startup_settings(value: Settings) -> None:
    missing = missing_required_setting_names(value)
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
    if value.host == "0.0.0.0" and value.auth_mode == "token":  # noqa: S104 - comparison, not a bind.
        logger.warning(
            "ai_rag_public_bind_without_hmac host=0.0.0.0 auth_mode=token"
        )


settings = Settings()
