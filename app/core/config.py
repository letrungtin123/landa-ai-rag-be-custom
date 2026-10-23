from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.logging import log_event

APP_ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)

Environment = Literal["development", "test", "production"]
AuthMode = Literal["token", "hmac", "token_or_hmac"]


def _read_environment(*, emit_legacy_warning: bool = True) -> Environment:
    explicit = os.getenv("AI_RAG_ENV", "").strip().lower()
    legacy = os.getenv("NODE_ENV", "").strip().lower()
    value = explicit or legacy or "development"
    if not explicit and legacy and emit_legacy_warning:
        log_event(logger, logging.WARNING, "ai_rag_legacy_environment_variable", variable="NODE_ENV",
                  replacement="AI_RAG_ENV")
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


# Loaded at import on purpose, before ``Settings()`` below: uvicorn worker processes import
# ``app.main:app`` directly (no entry hook runs first), several modules read settings into
# constants at import (provider retry policy, limiter sizes) and tests and scripts import app
# modules without an entry point. Production (and AI_RAG_ENV=test) never reads a dotenv file.
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
    # Idle keep-alive of uvicorn. It must outlive every client's idle timeout (nginx upstream 60 s,
    # the Node agent 30 s) so a reused socket is never closed by the server mid-request (ECONNRESET).
    keep_alive_timeout_seconds: int = Field(
        default=75,
        validation_alias="AI_RAG_KEEP_ALIVE_TIMEOUT_SECONDS",
        ge=1,
        le=600,
    )
    # uvicorn answers a plain 503 ("Service Unavailable", no app code runs) to a new request while this many
    # connections are open, idle keep-alive ones included, or this many requests run (security audit 2026-10-24,
    # L4): a ceiling on what a flood of connections can hold. Far above the real load: the provider, index and CPU
    # limiters (8 / 2 / up to 4 by default) bound the actual work and answer SERVICE_BUSY, and a long IDM request
    # holds one connection.
    limit_concurrency: int = Field(
        default=128,
        validation_alias="AI_RAG_LIMIT_CONCURRENCY",
        ge=16,
        le=100_000,
    )
    # Reported by GET /v1/meta; the container image sets it from the git sha.
    build_sha: str = Field(default="unknown", validation_alias="AI_RAG_BUILD_SHA", max_length=64)
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

    # IDM pipeline (spec §11.5). Owner P4 (2026-10-20): the judge repairs the slots it finds a major problem in, once,
    # when the repair and a second judge still fit the unit's time; "observe" only records its findings.
    idm_judge_mode: Literal["off", "observe", "repair"] = Field(
        default="repair",
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

    # Secrets are SecretStr: repr(settings) and validation errors never show them; read them with
    # get_secret_value() only where they are used.
    database_url: SecretStr = Field(default=SecretStr(""), validation_alias="DATABASE_URL")
    # Postgres pool (SEP-1). The DSN may also carry sslmode/sslrootcert; these settings win.
    db_pool_min: int = Field(default=1, validation_alias="AI_RAG_DB_POOL_MIN", ge=0, le=64)
    db_pool_max: int = Field(default=8, validation_alias="AI_RAG_DB_POOL_MAX", ge=1, le=128)
    # Unset: the DSN sslmode, else "require" in production ("prefer" for a loopback host, where TLS
    # protects nothing) and "prefer" elsewhere.
    db_ssl_mode: Literal["disable", "prefer", "require", "verify-ca", "verify-full"] | None = Field(
        default=None,
        validation_alias="AI_RAG_DB_SSL_MODE",
    )
    db_ssl_root_cert: str = Field(default="", validation_alias="AI_RAG_DB_SSL_ROOT_CERT", max_length=4_096)
    # 0 disables prepared-statement caching (required behind PgBouncer/Supavisor transaction mode).
    db_statement_cache_size: int = Field(
        default=100,
        validation_alias="AI_RAG_DB_STATEMENT_CACHE_SIZE",
        ge=0,
        le=10_000,
    )
    db_connect_timeout_seconds: int = Field(
        default=10,
        validation_alias="AI_RAG_DB_CONNECT_TIMEOUT_SECONDS",
        ge=1,
        le=120,
    )
    # Server-side TCP keepalives (tcp_keepalives_* startup settings). 0 sends none, e.g. when a
    # pooler rejects unknown startup parameters.
    db_tcp_keepalives_idle_seconds: int = Field(
        default=60,
        validation_alias="AI_RAG_DB_TCP_KEEPALIVES_IDLE_SECONDS",
        ge=0,
        le=7_200,
    )
    db_connect_retry_max_seconds: int = Field(
        default=30,
        validation_alias="AI_RAG_DB_CONNECT_RETRY_MAX_SECONDS",
        ge=1,
        le=300,
    )
    supabase_url: str = Field(default="", validation_alias="SUPABASE_URL")
    # Legacy storage access. Without it the backend must send a signed URL with every index request.
    supabase_service_key: SecretStr = Field(default=SecretStr(""), validation_alias="SUPABASE_SERVICE_KEY")
    # Comma-separated origins (scheme://host[:port]) a signed source URL may point to. Empty: the
    # origin of SUPABASE_URL. Plain http is accepted only for a loopback host.
    storage_allowed_origins: str = Field(default="", validation_alias="AI_RAG_STORAGE_ALLOWED_ORIGINS",
                                         max_length=4_096)
    # PEM bundle that signs the storage server certificate (internal CA). Empty: public CAs (certifi).
    storage_ca_file: str = Field(default="", validation_alias="AI_RAG_STORAGE_CA_FILE", max_length=4_096)
    storage_download_timeout_seconds: int = Field(
        default=120,
        validation_alias="AI_RAG_STORAGE_DOWNLOAD_TIMEOUT_SECONDS",
        ge=5,
        le=600,
    )
    # Read multi-column PDF pages column by column (QC D9). false restores top-to-bottom order.
    pdf_column_aware: bool = Field(default=True, validation_alias="AI_RAG_PDF_COLUMN_AWARE")
    supabase_storage_bucket: str = Field(
        default="landa-storage",
        validation_alias="SUPABASE_STORAGE_BUCKET",
        min_length=1,
        max_length=255,
    )

    # hmac by default (security audit 2026-10-24, L2): the static-token modes (token, token_or_hmac, kept for a key
    # rotation) apply only when set explicitly.
    auth_mode: AuthMode = Field(default="hmac", validation_alias="AI_RAG_AUTH_MODE")
    service_token: SecretStr = Field(default=SecretStr(""), validation_alias="AI_RAG_SERVICE_TOKEN")
    service_hmac_secrets: SecretStr = Field(default=SecretStr(""), validation_alias="AI_RAG_SERVICE_HMAC_SECRETS")
    auth_clock_skew_seconds: int = Field(
        default=300,
        validation_alias="AI_RAG_AUTH_CLOCK_SKEW_SECONDS",
        ge=30,
        le=900,
    )
    # Kept at least twice the clock skew (validate_startup_settings): a signed request is accepted for 2 x the skew.
    auth_replay_ttl_seconds: int = Field(
        default=600,
        validation_alias="AI_RAG_AUTH_REPLAY_TTL_SECONDS",
        ge=60,
        le=3_600,
    )
    # HMAC v1 signatures (no key id and no request id in the signed message, security audit 2026-10-24 M1) are
    # accepted only while this is true: the transition to v2. Each accepted one is logged and counted
    # (ai_rag_auth_legacy_accepted_total{reason="legacy_signature"}); set false once that stays at 0.
    hmac_accept_v1_signature: bool = Field(default=True, validation_alias="AI_RAG_HMAC_ACCEPT_V1_SIGNATURE")

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
    # Memory of one DOCX/PPTX/XLSX extraction (measured 2026-10-24 on the production interpreter): about 15 MB per
    # MB of XML (python-docx and python-pptx build every XML part as a whole tree; openpyxl 6 to 13) and about 2 MB
    # per MB of media. 64 MB in all with at most 12 MB of XML keeps one extraction near 300 MB above the service's
    # ~150 MB, so two concurrent ones stay under PM2's 768 MB restart. Pictures do not compress, so a document within
    # AI_RAG_MAX_DOCUMENT_BYTES keeps its media below the total.
    max_ooxml_uncompressed_bytes: int = Field(
        default=64 * 1024 * 1024,
        validation_alias="AI_RAG_MAX_OOXML_UNCOMPRESSED_BYTES",
        ge=1_024,
        le=2 * 1024 * 1024 * 1024,
    )
    # The part of that total whose content starts like XML (app.core.document_limits.validate_ooxml_archive).
    max_ooxml_xml_bytes: int = Field(
        default=12 * 1024 * 1024,
        validation_alias="AI_RAG_MAX_OOXML_XML_BYTES",
        ge=1_024,
        le=2 * 1024 * 1024 * 1024,
    )
    # .xlsx worksheets are streamed (openpyxl read_only): 199,200 cells with 12.6 MB of XML took +114 MB, 15 MB took
    # +159 MB, less than a 300-page DOCX within max_ooxml_xml_bytes. AI_RAG_MAX_XLSX_CELLS and the extracted-character
    # cap bound a workbook; this budget replaces max_ooxml_xml_bytes for .xlsx only.
    max_xlsx_xml_bytes: int = Field(
        default=32 * 1024 * 1024,
        validation_alias="AI_RAG_MAX_XLSX_XML_BYTES",
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
    # Text one document may yield (every page, slide and sheet): bounds the chunks, embeddings and rows of one index
    # job. A 1,000-page book is about 3 million characters.
    max_extracted_chars: int = Field(
        default=10_000_000,
        validation_alias="AI_RAG_MAX_EXTRACTED_CHARS",
        ge=1_000,
        le=500_000_000,
    )
    libreoffice_timeout_seconds: int = Field(
        default=120,
        validation_alias="AI_RAG_LIBREOFFICE_TIMEOUT_SECONDS",
        ge=5,
        le=600,
    )

    # The bounds of the tuning values below are wide on purpose: they reject a broken value (0, a
    # negative, a typo with extra digits) at startup without narrowing any value in use.
    chunk_max_chars: int = Field(default=3_200, validation_alias="AI_RAG_CHUNK_MAX_CHARS", ge=100, le=100_000)
    chunk_overlap_chars: int = Field(default=350, validation_alias="AI_RAG_CHUNK_OVERLAP_CHARS", ge=0, le=50_000)
    # The provider call clamps the batch to 1..100 (provider.embedding_batch_size).
    embedding_batch_size: int = Field(default=32, validation_alias="AI_RAG_EMBEDDING_BATCH_SIZE", ge=1, le=1_000)
    provider_request_timeout_ms: int = Field(default=60_000, validation_alias="AI_RAG_PROVIDER_REQUEST_TIMEOUT_MS",
                                             ge=1_000, le=600_000)
    blueprint_provider_request_timeout_ms: int = Field(
        default=300_000,
        validation_alias="AI_RAG_BLUEPRINT_PROVIDER_REQUEST_TIMEOUT_MS",
        ge=1_000,
        le=600_000,
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
    # The legacy workflows clamp both to 0..2.
    course_workflow_max_repair_attempts: int = Field(
        default=2,
        validation_alias="AI_RAG_COURSE_WORKFLOW_MAX_REPAIR_ATTEMPTS",
        ge=0,
        le=10,
    )
    lesson_workflow_max_repair_attempts: int = Field(
        default=2,
        validation_alias="AI_RAG_LESSON_WORKFLOW_MAX_REPAIR_ATTEMPTS",
        ge=0,
        le=10,
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
        ge=1,
        le=3_600,
    )
    top_k: int = Field(default=8, validation_alias="AI_RAG_TOP_K", ge=1, le=200)
    max_context_chars: int = Field(default=18_000, validation_alias="AI_RAG_MAX_CONTEXT_CHARS", ge=1_000,
                                   le=1_000_000)
    lesson_author_top_k: int = Field(default=24, validation_alias="AI_RAG_LESSON_AUTHOR_TOP_K", ge=1, le=500)
    lesson_author_max_context_chars: int = Field(
        default=32_000,
        validation_alias="AI_RAG_LESSON_AUTHOR_MAX_CONTEXT_CHARS",
        ge=1_000,
        le=2_000_000,
    )
    lesson_author_max_chunks_per_document: int = Field(
        default=12,
        validation_alias="AI_RAG_LESSON_AUTHOR_MAX_CHUNKS_PER_DOCUMENT",
        ge=1,
        le=1_000,
    )
    lesson_author_scope_max_chunks: int = Field(
        default=48,
        validation_alias="AI_RAG_LESSON_AUTHOR_SCOPE_MAX_CHUNKS",
        ge=1,
        le=5_000,
    )
    source_coverage_canonical_max_chars: int = Field(
        default=1_000_000,
        validation_alias="AI_RAG_SOURCE_COVERAGE_CANONICAL_MAX_CHARS",
        ge=10_000,
        le=50_000_000,
    )
    source_map_architect_context_max_chars: int = Field(
        default=48_000,
        validation_alias="AI_RAG_SOURCE_MAP_ARCHITECT_CONTEXT_MAX_CHARS",
        ge=1_000,
        le=2_000_000,
    )
    # Per-query HNSW settings of the vector search (audit 2026-10-24, P3). The index covers every tenant and the
    # tenant/KB/active-index filters apply after it, so a scan that stops at ef_search candidates (pgvector default
    # 40) can return fewer rows than asked: measured on the local database, 14 of 32. hnsw.ef_search is raised to
    # the query's row limit (at least this value, at most pgvector's 1,000) and, with pgvector 0.8 or later, the
    # scan goes on in exact distance order (strict_order) until the limit is met. 0 sends no per-query setting.
    vector_ef_search: int = Field(default=40, validation_alias="AI_RAG_VECTOR_EF_SEARCH", ge=0, le=1_000)
    vector_iterative_scan: Literal["off", "strict_order"] = Field(
        default="strict_order",
        validation_alias="AI_RAG_VECTOR_ITERATIVE_SCAN",
    )
    retrieval_candidate_multiplier: int = Field(
        default=4,
        validation_alias="AI_RAG_RETRIEVAL_CANDIDATE_MULTIPLIER",
        ge=1,
        le=50,
    )
    # Similarity thresholds: cosine similarity is -1..1, pg_trgm keyword similarity 0..1.
    retrieval_min_score: float = Field(default=0.25, validation_alias="AI_RAG_RETRIEVAL_MIN_SCORE", ge=-1.0, le=1.0)
    retrieval_keyword_min_score: float = Field(
        default=0.50,
        validation_alias="AI_RAG_RETRIEVAL_KEYWORD_MIN_SCORE",
        ge=0.0,
        le=1.0,
    )
    retrieval_max_chunks_per_document: int = Field(
        default=4,
        validation_alias="AI_RAG_RETRIEVAL_MAX_CHUNKS_PER_DOCUMENT",
        ge=1,
        le=200,
    )
    max_user_message_chars: int = Field(default=20_000, validation_alias="AI_RAG_MAX_USER_MESSAGE_CHARS", ge=1,
                                        le=1_000_000)
    # Gemini accepts 0.0-2.0.
    generation_temperature: float = Field(default=0.2, validation_alias="AI_RAG_GENERATION_TEMPERATURE", ge=0.0,
                                          le=2.0)
    gemini_38_thinking_level: Literal["low", "medium", "high"] = Field(
        default="medium",
        validation_alias="AI_RAG_GEMINI_38_THINKING_LEVEL",
    )

    @model_validator(mode="after")
    def validate_pool_bounds(self) -> Settings:
        if self.db_pool_min > self.db_pool_max:
            raise ValueError("AI_RAG_DB_POOL_MIN must not exceed AI_RAG_DB_POOL_MAX.")
        return self

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        return self.environment == "development"


def missing_required_setting_names(value: Settings) -> list[str]:
    # SUPABASE_SERVICE_KEY is optional since SEP-1: documents arrive through backend-signed URLs.
    # SUPABASE_URL is still needed to derive the storage allowlist unless it is configured.
    service_token = value.service_token.get_secret_value()
    hmac_secrets = value.service_hmac_secrets.get_secret_value()
    required = {
        "DATABASE_URL": bool(value.database_url.get_secret_value()),
        "SUPABASE_URL": bool(value.supabase_url or value.storage_allowed_origins),
    }
    if (
        value.auth_mode in {"token", "token_or_hmac"}
        and not service_token
        and (value.auth_mode == "token" or not hmac_secrets)
    ):
        required["AI_RAG_SERVICE_TOKEN"] = False
    if (
        value.auth_mode in {"hmac", "token_or_hmac"}
        and not hmac_secrets
        and (value.auth_mode == "hmac" or not service_token)
    ):
        required["AI_RAG_SERVICE_HMAC_SECRETS"] = False
    return sorted(name for name, configured in required.items() if not configured)


def validate_startup_settings(value: Settings) -> None:
    missing = missing_required_setting_names(value)
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
    if value.auth_replay_ttl_seconds < 2 * value.auth_clock_skew_seconds:
        # A signed request is accepted while its timestamp is within the skew on either side of the clock, so its
        # request id must stay in the replay cache for twice the skew or it can be sent again.
        raise RuntimeError("AI_RAG_AUTH_REPLAY_TTL_SECONDS must be at least twice AI_RAG_AUTH_CLOCK_SKEW_SECONDS.")
    if value.host == "0.0.0.0" and value.auth_mode == "token":  # noqa: S104 - comparison, not a bind.
        log_event(logger, logging.WARNING, "ai_rag_public_bind_without_hmac", host=value.host,
                  auth_mode=value.auth_mode)


settings = Settings()
