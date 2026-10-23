# landa-ai-rag

Internal FastAPI service of the LANDA platform: document indexing for the knowledge base, RAG chat, and AI ID
course generation (the IDM instructional-design pipeline, Gemini provider). Only `landa-backend` calls it, with
HMAC-signed requests; browsers never reach it. The backend owns orchestration (durable V2 workers), persistence
and acceptance; this service turns one request into one validated result.

## Run locally

Python 3.12 only. Dependencies are hash-locked (`requirements.lock`, `requirements-dev.lock`).

    python -m venv .venv-dev
    .\.venv-dev\Scripts\python.exe -m pip install --require-hashes -r requirements.lock
    .\.venv-dev\Scripts\python.exe -m pip install --require-hashes -r requirements-dev.lock
    Copy-Item .env.example .env        # development only; fill DATABASE_URL, SUPABASE_URL, auth
    .\.venv-dev\Scripts\python.exe -m app

On Linux use `.venv-dev/bin/python` and `cp`. `python -m app` takes host, port, workers, keep-alive and graceful
shutdown from the settings below. A `.env` file is read only when `AI_RAG_ENV` is unset or `development`;
production and `AI_RAG_ENV=test` read the process environment only. `/docs` is served in development only.

## Quality gate

`scripts\check.ps1` (Windows) and `scripts/check.sh` (Linux, CI) run the same steps with `.venv-dev`:

1. ruff, every rule, on the strict code: all of `app` except the legacy pipeline (`app/services/lesson_author`,
   `app/workflows`, the legacy modules directly in `app`), plus every test directory except `tests/unit/legacy`
   and `tools/` (`app/content` and `app/services/staged_unit` keep their inline thresholds: `PLR2004` is off there);
2. ruff pyflakes rules on all of `app/services`, `app/content` and `app/contracts`;
3. mypy `--strict` on the strict code; legacy modules keep a "no new errors" baseline (`pyproject.toml`);
4. pytest with branch coverage, failing under 85%, then the coverage of each `app` package
   (`scripts/coverage_by_package.py`: a report, except `app/idm`, which fails under 90%);
5. pip-audit of `requirements.lock`.

Tests never reach the network or a database: providers are fakes at the `google.genai.Client` or
`app.services.provider.generate_content` boundary, the database is a recording fake. Recorded expectations are
updated deliberately, never by hand:

- route snapshots (`tests/snapshots/routes`): `LANDA_UPDATE_ROUTE_SNAPSHOTS=1`;
- IDM prompt manifest (`tests/fixtures/prompt_templates_manifest.json`): `LANDA_UPDATE_PROMPT_MANIFEST=1`;
- the settings table of this file: `LANDA_UPDATE_README=1 python -m pytest tests/contract/test_readme_settings.py`.

Test layout: `tests/unit/<package>/` holds the tests of one `app` package (`tests/unit/legacy/` those of the legacy
V5 pipeline, deleted with it); `tests/contract/` the HTTP wire format, route snapshots, error codes and repository
rules; `tests/golden/` the IDM golden builders and the recorded prompt hashes; `tests/support/` shared helpers;
`tools/` scripts that are not tests (QC replay, schema probe, source benchmark). The files left directly in `tests/`
are named by the backend's cross-language tests (`python -m tests.<module>`, `tests/idm_golden.py`): they move only
together with the backend.

`tests/contract/test_architecture_layers.py` holds the structure rules: layer imports, no `print`, internal auth on
every route except `/healthz` and `/readyz`, thin route handlers, structured log calls, and STD-5 sizes (strict code:
files at most 800 lines, functions at most 120, with a fixed allowlist that only shrinks).

## Architecture

| Package | Role |
|---|---|
| `app/main.py` | App factory only: middleware, error handlers, route registration. `app.main:app` is the ASGI entrypoint. |
| `app/api` | `deps.py` (internal auth, DB pool) and `routes/` (health, kb, chat, lesson_author_legacy, orchestration_v2): thin handlers. |
| `app/schemas` | Request and response models per route group (`extra="forbid"`, typed responses). |
| `app/core` | Settings, security (HMAC), concurrency limits, deadlines, errors, JSON logging, metrics, handled-exception reporting, accent folding (`text.py`). |
| `app/infra` | Postgres pool and schema check, signed-URL storage downloads, Gemini client pool, provider 429 classification, PDF layout. |
| `app/repositories` | All SQL (asyncpg); every tenant query filters by `tenant_id`. |
| `app/services` | `provider.py` (Gemini gateway: bounded retries, usage, telemetry), `ingestion`, `retrieval`, `chat`, `orchestration_v2` (V2 routes and IDM wiring), `staged_unit` (staged unit validation, instance binder and repair merges, shared by IDM W5 and the legacy writer), `runtime.py` (process resources), `lesson_author` (legacy V5 pipeline, deleted in PRD-3). |
| `app/idm` | The IDM pipeline: pure code (no web framework, DB or settings); the service injects an `IdmRuntime` and the shared helpers. |
| `app/content` | Shared content core used by IDM, the V2 contract, the staged validators and ingestion: source structure and readiness, evidence bundles, instructional quality and learner-content purity rules, ordered content. |
| `app/contracts` | `orchestration_v2`: the V2 contract models (`blueprint.py`) and provider drafts, binders and unit contract (`provider.py`). |
| `app/*.py` (others) | Legacy V2/V5 domain modules. The ones moved to `app/content` and `app/contracts` in refactor wave 2 keep their old path as an alias of the moved module (tests and the backend scripts import it); new code imports the new path. |

IDM stages (pipeline `idm-1`, used when the request carries an `idm` / `idm_module_context` /
`unit_contract.idm_unit_brief` field; otherwise the legacy V2 path runs unchanged):

- **course skeleton** (`course_design.py`): W1 content map per source section (in parallel), W1 reduce
  (objectives and audience), W2 blueprint (classification and treatment of every block), W4 course architecture
  (modules and lessons), then deterministic assembly; every source fact gets exactly one disposition;
- **chapter shard** (`module_design.py`): W3/W4 design of one module (practice tasks, units, components);
- **unit** (`storyboard.py`): W5 writer and targeted repairs per unit, W6 judge (`AI_RAG_IDM_JUDGE_MODE`);
- a document the author keeps as written (`design_mode` preserve) is laid out without provider calls
  (`preserve_*.py`).

Each provider call runs inside the task deadline and the token allowance Node grants; an invalid answer gets a
bounded repair, then deterministic source-locked content marked for review. Prompts live in `app/idm/prompts.py`
(and the rule builders listed in `app/idm/prompt_manifest.py`).

## Contract with the backend

- Routes: `GET /healthz` (liveness), `GET /readyz` (started, not draining, DB reachable, schema check passed),
  and with internal auth `GET /v1/meta`, `GET /metrics`, `POST /v1/kb/documents/index`,
  `POST /v1/kb/documents/delete`, `POST /v1/kb/delete`, `POST /v1/chat`, the legacy lesson-author routes
  (`/v1/lesson-author/proposal`, `/blueprint`, `/chapter-checkpoint`) and the orchestration V2 routes
  (`/v1/lesson-author/orchestration-v2/source-snapshot`, `/course-skeleton`, `/chapter-shard`, `/unit`).
- Auth: `AI_RAG_AUTH_MODE` `hmac` (the default); `token` or `token_or_hmac` (a move off the static token) only
  when set explicitly. HMAC secrets have at least 32 characters. HMAC headers
  `X-Landa-Key-Id`, `X-Landa-Timestamp` (Unix seconds, skew `AI_RAG_AUTH_CLOCK_SKEW_SECONDS`),
  `X-Landa-Request-Id` (a new UUID per send; replays are refused for `AI_RAG_AUTH_REPLAY_TTL_SECONDS`, which
  must be at least twice the skew), `X-Landa-Signature-Version: 2` and `X-Landa-Signature` = lowercase hex
  HMAC-SHA256 of `v2\n{key_id}\n{request_id}\n{timestamp}\n{METHOD}\n{path}\n{sha256_hex(body)}` (ids and
  timestamp exactly as sent). A request without the version header is v1, signed over
  `{timestamp}\n{METHOD}\n{path}\n{sha256_hex(body)}`; it is accepted only while
  `AI_RAG_HMAC_ACCEPT_V1_SIGNATURE` is true and counted in `ai_rag_auth_legacy_accepted_total`. Any other version
  is refused. `AI_RAG_SERVICE_HMAC_SECRETS` is `kid:secret[,kid:secret]`. A request whose headers already fail
  (missing, unknown key id, malformed timestamp, signature, request id or version) is refused before its body is
  read, with the same 401. uvicorn answers a plain 503 above `AI_RAG_LIMIT_CONCURRENCY` open connections.
- Errors: `{"detail": {"code": "STABLE_CODE", "message": "safe text", "correlation_id": "..."}}`. Transient
  failures (`AI_PROVIDER_UNAVAILABLE`, `AI_PROVIDER_RATE_LIMITED`, `AI_PROVIDER_TIMEOUT`, `SERVICE_BUSY`,
  `DATABASE_NOT_READY`) are 503/504 so the backend's durable workers retry; `AI_PROVIDER_QUOTA_EXHAUSTED` stops
  the run.
- `GET /v1/meta` reports the build sha, `contracts` (orchestration V2 contract 2, `idm-1`, IDM contract 3,
  `idm-prompt-1`, index request 2; the backend compares these keys at worker startup), the schema-check result
  and `prompt_templates_hash`, a hash of the IDM prompt template sources that changes with every prompt edit.
- The backend's cross-language tests run this checkout (`tests/idm_contract_bridge.py`,
  `python -c "from app.main import ..."`): keep the names re-exported by `app/main.py`.
- Database: the service never runs DDL. It reads 9 columns of `kb_documents` and owns `rag_document_indexes`,
  `rag_chunks` and `rag_document_structure_nodes` (the tenant data-quota trigger tables it writes through are
  listed in `app/infra/schema_check.py`); schema changes come from `supabase/manual_sql`, and the read-only
  schema check reports a mismatch on `/readyz`.

## Observability

- Logs: one JSON line per event on stdout (`app.core.logging`): `event`, `message`, `request_id`,
  `correlation_id`, `route`, `status`, `duration_ms` and the event's own fields. Code logs with
  `log_event(logger, level, "event_name", **fields)`; IDM and orchestration events keep the
  `lesson_author_idm` / `lesson_author_orchestration_v2` prefix in the message. Never log prompts, source text,
  provider output, keys or DSNs; the redaction filter is a safety net. Errors carry a sanitised traceback
  (`exception_frames`: module, function and line only).
- Broad exception handlers that turn an error into a fallback or a safe code log `broad_exception_handled`
  (stage, outcome, exception type, frames) and count `ai_rag_handled_exceptions_total`.
- Metrics (`GET /metrics`, Prometheus): HTTP requests and durations by route template, provider calls, durations
  and tokens by operation and pipeline `stage` (IDM stage, `course_skeleton`, `chapter_shard`, `chat`,
  embeddings), fallbacks, limiter use and rejections, document-limit rejections, deadline hits.

## Release and runtime

- Production on the LANDA server runs under PM2 from a separate release worktree with its own runtime
  interpreter and an env file outside the repositories (see the AI ID deploy runbooks in `AI-ID PLAN/`); this
  development tree and `.venv-dev` are never the production runtime. A release moves that worktree to a reviewed
  commit and restarts the PM2 process, only on the owner's request.
- The container image (`Dockerfile`, non-root, read-only root filesystem) is built, smoke-tested, scanned and
  given an SBOM by CI; `main` and `v*` tags push it to ECR for server B (`deploy/server-b/`).
- Deploy the backend first when a change says the backend must accept a new field.

## Settings

Generated from `app/core/config.py` (`LANDA_UPDATE_README=1 python -m pytest tests/contract/test_readme_settings.py`).
Secrets are `SecretStr` and never logged; `.env.example` holds placeholders.

<!-- settings-table:start -->
| Variable | Type | Default | Bounds |
|---|---|---|---|
| `AI_RAG_ENV` | `development` \| `test` \| `production` | `development` (or `NODE_ENV` when `AI_RAG_ENV` is unset) |  |
| `AI_RAG_HOST` | text | `127.0.0.1` | length >= 1, length <= 255 |
| `AI_RAG_PORT` | int | `8010` | >= 1, <= 65535 |
| `AI_RAG_WORKERS` | int | `1` | >= 1, <= 64 |
| `AI_RAG_PROXY_HEADERS` | bool | `true` |  |
| `AI_RAG_FORWARDED_ALLOW_IPS` | text | `127.0.0.1` |  |
| `AI_RAG_SHUTDOWN_GRACE_SECONDS` | int | `60` | >= 1, <= 600 |
| `AI_RAG_KEEP_ALIVE_TIMEOUT_SECONDS` | int | `75` | >= 1, <= 600 |
| `AI_RAG_LIMIT_CONCURRENCY` | int | `128` | >= 16, <= 100000 |
| `AI_RAG_BUILD_SHA` | text | `unknown` | length <= 64 |
| `AI_RAG_READINESS_DB_TIMEOUT_MS` | int | `1000` | >= 100, <= 10000 |
| `AI_RAG_MAX_CONCURRENT_PROVIDER_CALLS` | int | `8` | >= 1, <= 256 |
| `AI_RAG_MAX_CONCURRENT_INDEX_JOBS` | int | `2` | >= 1, <= 64 |
| `AI_RAG_CPU_WORKERS` | int | CPU count - 1, between 1 and 4 | >= 1, <= 64 |
| `AI_RAG_LIMITER_ACQUIRE_TIMEOUT_MS` | int | `30000` | >= 100, <= 600000 |
| `AI_RAG_EXTRACTION_EXECUTOR` | `thread` \| `process` | `thread` |  |
| `AI_RAG_PROVIDER_MAX_ATTEMPTS` | int | `2` | >= 1, <= 5 |
| `AI_RAG_PROVIDER_RETRY_BASE_MS` | int | `1500` | >= 0, <= 30000 |
| `AI_RAG_PROVIDER_RETRY_MAX_MS` | int | `8000` | >= 0, <= 60000 |
| `AI_RAG_PROVIDER_RATE_LIMIT_MAX_WAIT_MS` | int | `60000` | >= 0, <= 300000 |
| `AI_RAG_CHAT_DEADLINE_MS` | int | `180000` | >= 1000, <= 3600000 |
| `AI_RAG_INDEX_DEADLINE_MS` | int | `585000` | >= 1000, <= 3600000 |
| `AI_RAG_LESSON_AUTHOR_DEADLINE_MS` | int | `585000` | >= 1000, <= 3600000 |
| `AI_RAG_IDM_JUDGE_MODE` | `off` \| `observe` \| `repair` | `repair` |  |
| `AI_RAG_IDM_W1_PARALLELISM` | int | `4` | >= 1, <= 8 |
| `AI_RAG_IDM_MAX_SECTIONS` | int | `16` | >= 4, <= 16 |
| `AI_RAG_IDM_PROVIDER_CALL_TIMEOUT_MS` | int | `180000` | >= 30000, <= 300000 |
| `DATABASE_URL` | secret | empty |  |
| `AI_RAG_DB_POOL_MIN` | int | `1` | >= 0, <= 64 |
| `AI_RAG_DB_POOL_MAX` | int | `8` | >= 1, <= 128 |
| `AI_RAG_DB_SSL_MODE` | `disable` \| `prefer` \| `require` \| `verify-ca` \| `verify-full` \| unset | unset |  |
| `AI_RAG_DB_SSL_ROOT_CERT` | text | empty | length <= 4096 |
| `AI_RAG_DB_STATEMENT_CACHE_SIZE` | int | `100` | >= 0, <= 10000 |
| `AI_RAG_DB_CONNECT_TIMEOUT_SECONDS` | int | `10` | >= 1, <= 120 |
| `AI_RAG_DB_TCP_KEEPALIVES_IDLE_SECONDS` | int | `60` | >= 0, <= 7200 |
| `AI_RAG_DB_CONNECT_RETRY_MAX_SECONDS` | int | `30` | >= 1, <= 300 |
| `SUPABASE_URL` | text | empty |  |
| `SUPABASE_SERVICE_KEY` | secret | empty |  |
| `AI_RAG_STORAGE_ALLOWED_ORIGINS` | text | empty | length <= 4096 |
| `AI_RAG_STORAGE_CA_FILE` | text | empty | length <= 4096 |
| `AI_RAG_STORAGE_DOWNLOAD_TIMEOUT_SECONDS` | int | `120` | >= 5, <= 600 |
| `AI_RAG_PDF_COLUMN_AWARE` | bool | `true` |  |
| `SUPABASE_STORAGE_BUCKET` | text | `landa-storage` | length >= 1, length <= 255 |
| `AI_RAG_AUTH_MODE` | `token` \| `hmac` \| `token_or_hmac` | `hmac` |  |
| `AI_RAG_SERVICE_TOKEN` | secret | empty |  |
| `AI_RAG_SERVICE_HMAC_SECRETS` | secret | empty |  |
| `AI_RAG_AUTH_CLOCK_SKEW_SECONDS` | int | `300` | >= 30, <= 900 |
| `AI_RAG_AUTH_REPLAY_TTL_SECONDS` | int | `600` | >= 60, <= 3600 |
| `AI_RAG_HMAC_ACCEPT_V1_SIGNATURE` | bool | `true` |  |
| `AI_RAG_MAX_REQUEST_BYTES` | int | `16777216` | >= 1024, <= 67108864 |
| `AI_RAG_IDM_MAX_REQUEST_BYTES` | int | `25165824` | >= 1024, <= 67108864 |
| `AI_RAG_MAX_DOCUMENT_BYTES` | int | `52428800` | >= 1024, <= 536870912 |
| `AI_RAG_MAX_DOCUMENT_PAGES` | int | `1000` | >= 1, <= 10000 |
| `AI_RAG_MAX_OOXML_UNCOMPRESSED_BYTES` | int | `67108864` | >= 1024, <= 2147483648 |
| `AI_RAG_MAX_OOXML_XML_BYTES` | int | `12582912` | >= 1024, <= 2147483648 |
| `AI_RAG_MAX_XLSX_XML_BYTES` | int | `33554432` | >= 1024, <= 2147483648 |
| `AI_RAG_MAX_OOXML_ENTRIES` | int | `10000` | >= 1, <= 100000 |
| `AI_RAG_MAX_OOXML_COMPRESSION_RATIO` | number | `100.0` | >= 1.0, <= 1000.0 |
| `AI_RAG_MAX_XLSX_CELLS` | int | `200000` | >= 1, <= 5000000 |
| `AI_RAG_MAX_EXTRACTED_CHARS` | int | `10000000` | >= 1000, <= 500000000 |
| `AI_RAG_LIBREOFFICE_TIMEOUT_SECONDS` | int | `120` | >= 5, <= 600 |
| `AI_RAG_CHUNK_MAX_CHARS` | int | `3200` | >= 100, <= 100000 |
| `AI_RAG_CHUNK_OVERLAP_CHARS` | int | `350` | >= 0, <= 50000 |
| `AI_RAG_EMBEDDING_BATCH_SIZE` | int | `32` | >= 1, <= 1000 |
| `AI_RAG_PROVIDER_REQUEST_TIMEOUT_MS` | int | `60000` | >= 1000, <= 600000 |
| `AI_RAG_BLUEPRINT_PROVIDER_REQUEST_TIMEOUT_MS` | int | `300000` | >= 1000, <= 600000 |
| `AI_RAG_STAGED_LESSON_CONTENT_PROVIDER_TIMEOUT_MS` | int | `180000` | > 60000, <= 300000 |
| `AI_RAG_STAGED_LESSON_WORKFLOW_TIMEOUT_MS` | int | `480000` | > 0, <= 480000 |
| `AI_RAG_COURSE_WORKFLOW_MAX_REPAIR_ATTEMPTS` | int | `2` | >= 0, <= 10 |
| `AI_RAG_LESSON_WORKFLOW_MAX_REPAIR_ATTEMPTS` | int | `2` | >= 0, <= 10 |
| `AI_RAG_SEMANTIC_REVIEW_MODE` | `off` \| `observe` \| `repair` | `off` |  |
| `AI_RAG_SEMANTIC_REVIEW_MODEL` | text | empty | length <= 255 |
| `AI_RAG_SEMANTIC_REVIEW_TIMEOUT_MS` | int | `60000` | >= 1000, <= 120000 |
| `AI_RAG_SEMANTIC_REVIEW_MAX_OUTPUT_TOKENS` | int | `4096` | >= 512, <= 8192 |
| `AI_RAG_SEMANTIC_REVIEW_PROVIDER_ATTEMPT_CAP` | int | `8` | >= 2, <= 8 |
| `AI_RAG_DATABASE_COMMAND_TIMEOUT_SECONDS` | int | `600` | >= 1, <= 3600 |
| `AI_RAG_TOP_K` | int | `8` | >= 1, <= 200 |
| `AI_RAG_MAX_CONTEXT_CHARS` | int | `18000` | >= 1000, <= 1000000 |
| `AI_RAG_LESSON_AUTHOR_TOP_K` | int | `24` | >= 1, <= 500 |
| `AI_RAG_LESSON_AUTHOR_MAX_CONTEXT_CHARS` | int | `32000` | >= 1000, <= 2000000 |
| `AI_RAG_LESSON_AUTHOR_MAX_CHUNKS_PER_DOCUMENT` | int | `12` | >= 1, <= 1000 |
| `AI_RAG_LESSON_AUTHOR_SCOPE_MAX_CHUNKS` | int | `48` | >= 1, <= 5000 |
| `AI_RAG_SOURCE_COVERAGE_CANONICAL_MAX_CHARS` | int | `1000000` | >= 10000, <= 50000000 |
| `AI_RAG_SOURCE_MAP_ARCHITECT_CONTEXT_MAX_CHARS` | int | `48000` | >= 1000, <= 2000000 |
| `AI_RAG_RETRIEVAL_CANDIDATE_MULTIPLIER` | int | `4` | >= 1, <= 50 |
| `AI_RAG_RETRIEVAL_MIN_SCORE` | number | `0.25` | >= -1.0, <= 1.0 |
| `AI_RAG_RETRIEVAL_KEYWORD_MIN_SCORE` | number | `0.5` | >= 0.0, <= 1.0 |
| `AI_RAG_RETRIEVAL_MAX_CHUNKS_PER_DOCUMENT` | int | `4` | >= 1, <= 200 |
| `AI_RAG_MAX_USER_MESSAGE_CHARS` | int | `20000` | >= 1, <= 1000000 |
| `AI_RAG_GENERATION_TEMPERATURE` | number | `0.2` | >= 0.0, <= 2.0 |
| `AI_RAG_GEMINI_38_THINKING_LEVEL` | `low` \| `medium` \| `high` | `medium` |  |
<!-- settings-table:end -->
