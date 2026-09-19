from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field


APP_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = APP_ROOT.parent

load_dotenv(APP_ROOT / ".env")
load_dotenv(REPO_ROOT / "landa-backend" / ".env.production", override=False)


class Settings(BaseModel):
    environment: str = Field(default_factory=lambda: os.getenv("NODE_ENV", "development").strip().lower() or "development")
    port: int = Field(default_factory=lambda: int(os.getenv("PORT", "8010")))
    database_url: str = Field(default_factory=lambda: os.getenv("DATABASE_URL", "").strip())
    supabase_url: str = Field(default_factory=lambda: os.getenv("SUPABASE_URL", "").strip())
    supabase_service_key: str = Field(default_factory=lambda: os.getenv("SUPABASE_SERVICE_KEY", "").strip())
    supabase_storage_bucket: str = Field(default_factory=lambda: os.getenv("SUPABASE_STORAGE_BUCKET", "landa-storage").strip())
    service_token: str = Field(default_factory=lambda: os.getenv("AI_RAG_SERVICE_TOKEN", "").strip())
    chunk_max_chars: int = Field(default_factory=lambda: int(os.getenv("AI_RAG_CHUNK_MAX_CHARS", "3200")))
    chunk_overlap_chars: int = Field(default_factory=lambda: int(os.getenv("AI_RAG_CHUNK_OVERLAP_CHARS", "350")))
    embedding_batch_size: int = Field(default_factory=lambda: int(os.getenv("AI_RAG_EMBEDDING_BATCH_SIZE", "32")))
    provider_request_timeout_ms: int = Field(default_factory=lambda: int(os.getenv("AI_RAG_PROVIDER_REQUEST_TIMEOUT_MS", "60000")))
    blueprint_provider_request_timeout_ms: int = Field(default_factory=lambda: int(os.getenv("AI_RAG_BLUEPRINT_PROVIDER_REQUEST_TIMEOUT_MS", "300000")))
    database_command_timeout_seconds: int = Field(default_factory=lambda: int(os.getenv("AI_RAG_DATABASE_COMMAND_TIMEOUT_SECONDS", "600")))
    top_k: int = Field(default_factory=lambda: int(os.getenv("AI_RAG_TOP_K", "8")))
    max_context_chars: int = Field(default_factory=lambda: int(os.getenv("AI_RAG_MAX_CONTEXT_CHARS", "18000")))
    # Authoring needs broader evidence than conversational Q&A. Keep these
    # separate so improving lesson completeness does not increase every chat
    # request's latency and token cost.
    lesson_author_top_k: int = Field(default_factory=lambda: int(os.getenv("AI_RAG_LESSON_AUTHOR_TOP_K", "24")))
    lesson_author_max_context_chars: int = Field(default_factory=lambda: int(os.getenv("AI_RAG_LESSON_AUTHOR_MAX_CONTEXT_CHARS", "32000")))
    lesson_author_max_chunks_per_document: int = Field(default_factory=lambda: int(os.getenv("AI_RAG_LESSON_AUTHOR_MAX_CHUNKS_PER_DOCUMENT", "12")))
    lesson_author_scope_max_chunks: int = Field(default_factory=lambda: int(os.getenv("AI_RAG_LESSON_AUTHOR_SCOPE_MAX_CHUNKS", "48")))
    retrieval_candidate_multiplier: int = Field(default_factory=lambda: int(os.getenv("AI_RAG_RETRIEVAL_CANDIDATE_MULTIPLIER", "4")))
    retrieval_min_score: float = Field(default_factory=lambda: float(os.getenv("AI_RAG_RETRIEVAL_MIN_SCORE", "0.25")))
    retrieval_keyword_min_score: float = Field(default_factory=lambda: float(os.getenv("AI_RAG_RETRIEVAL_KEYWORD_MIN_SCORE", "0.50")))
    retrieval_max_chunks_per_document: int = Field(default_factory=lambda: int(os.getenv("AI_RAG_RETRIEVAL_MAX_CHUNKS_PER_DOCUMENT", "4")))
    max_user_message_chars: int = Field(default_factory=lambda: int(os.getenv("AI_RAG_MAX_USER_MESSAGE_CHARS", "20000")))
    generation_temperature: float = Field(default_factory=lambda: float(os.getenv("AI_RAG_GENERATION_TEMPERATURE", "0.2")))

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

settings = Settings()
