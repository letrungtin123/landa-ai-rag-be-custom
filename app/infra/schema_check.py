"""Read-only startup check of the database objects the AI service's SQL needs (SEP-1 #2).

Mirrors the ownership contract of plan §1.1 and the grants of
``supabase/manual_sql/20261008_1600_ai_rag_least_privilege_role.sql``: tables and columns the
SQL in ``app/main.py`` touches, the ``vector`` / ``pg_trgm`` extensions it relies on, and that
``current_user`` holds the privileges (and, under row level security, a policy) for each use.
The check runs inside a READ ONLY transaction and only reads catalogs. A failure lists object
*names* only; it never includes data.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

Privilege = Literal["SELECT", "INSERT", "UPDATE", "DELETE"]
RLS_COMMAND_CODES: dict[str, frozenset[Privilege]] = {
    "*": frozenset({"SELECT", "INSERT", "UPDATE", "DELETE"}),
    "r": frozenset({"SELECT"}),
    "a": frozenset({"INSERT"}),
    "w": frozenset({"UPDATE"}),
    "d": frozenset({"DELETE"}),
}
MAX_REPORTED_MISSING = 50


@dataclass(frozen=True, slots=True)
class TableRequirement:
    """``grants``: (privilege, columns); ``None`` columns = table-level privilege."""

    table: str
    columns: tuple[str, ...]
    grants: tuple[tuple[Privilege, tuple[str, ...] | None], ...]
    required: bool = True

    def commands(self) -> frozenset[Privilege]:
        return frozenset(privilege for privilege, _ in self.grants)


_INDEX_COLUMNS = (
    "id", "tenant_id", "kb_id", "document_id", "version", "status", "is_active", "engine",
    "embedding_model", "embedding_dimensions", "content_sha256", "chunk_count", "error_reason",
    "started_at", "completed_at", "updated_at",
)
_CHUNK_COLUMNS = (
    "tenant_id", "kb_id", "document_id", "index_id", "chunk_no", "content", "content_hash",
    "token_count", "source_page", "source_section", "metadata", "embedding",
)
_STRUCTURE_COLUMNS = (
    "tenant_id", "kb_id", "document_id", "index_id", "source_ref", "parent_source_ref", "level",
    "node_type", "title", "number_label", "page_start", "logical_page", "sort_order", "confidence",
    "parser_version", "metadata",
)
_KB_DOCUMENT_COLUMNS = ("id", "tenant_id", "kb_id", "type", "name", "status", "source_info", "file_path", "content")
_WRITE_ALL: tuple[tuple[Privilege, None], ...] = (("SELECT", None), ("INSERT", None), ("UPDATE", None),
                                                  ("DELETE", None))

# What the AI SQL uses. The tenant data-quota trigger chain (optional tables) fires on writes to
# the rag tables and runs as the writer, so the role needs the same column grants as the SQL file.
AI_SCHEMA_REQUIREMENTS: tuple[TableRequirement, ...] = (
    TableRequirement("kb_documents", _KB_DOCUMENT_COLUMNS, (("SELECT", _KB_DOCUMENT_COLUMNS),)),
    TableRequirement("rag_document_indexes", _INDEX_COLUMNS, _WRITE_ALL),
    TableRequirement("rag_chunks", _CHUNK_COLUMNS, (("SELECT", None), ("INSERT", None))),
    TableRequirement("rag_document_structure_nodes", _STRUCTURE_COLUMNS, _WRITE_ALL, required=False),
    TableRequirement("tenants", (), (("SELECT", ("id", "data_limit_bytes")),), required=False),
    TableRequirement(
        "tenant_data_quota_table_registry", (),
        (("SELECT", ("relation_name", "tenant_column", "allows_system_rows", "is_active")),), required=False,
    ),
    TableRequirement(
        "tenant_data_quota_usage", (),
        (("SELECT", ("tenant_id", "database_used_bytes", "storage_used_bytes", "storage_reserved_bytes", "state",
                     "revision")),
         ("INSERT", ("tenant_id",)),
         ("UPDATE", ("database_used_bytes", "revision", "updated_at"))),
        required=False,
    ),
    TableRequirement(
        "tenant_data_quota_reconciliation_queue", (),
        (("SELECT", ("tenant_id", "due_at", "updated_at")), ("INSERT", ("tenant_id", "due_at", "updated_at")),
         ("UPDATE", ("due_at", "updated_at")), ("DELETE", None)),
        required=False,
    ),
    TableRequirement(
        "tenant_data_quota_reconciliation_runs", (),
        (("SELECT", ("tenant_id", "status", "lease_expires_at", "next_attempt_at")),), required=False,
    ),
    TableRequirement(
        "tenant_storage_quota_reservations", (),
        (("SELECT", ("tenant_id", "status", "expires_at")),), required=False,
    ),
)
REQUIRED_EXTENSIONS = ("vector", "pg_trgm")
# Resolved through the role's search_path, exactly as the unqualified SQL resolves them.
REQUIRED_FUNCTIONS = ("similarity(text,text)",)
REQUIRED_TYPES = ("vector",)
REQUIRED_OPERATORS = ("<=>(vector,vector)",)


@dataclass(frozen=True, slots=True)
class SchemaCheckResult:
    status: Literal["ok", "failed", "error"]
    missing: tuple[str, ...] = ()
    checked_at: float = field(default_factory=time.time)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def code(self) -> str | None:
        if self.status == "ok":
            return None
        return "SCHEMA_CHECK_FAILED" if self.status == "failed" else "SCHEMA_CHECK_UNAVAILABLE"

    def summary(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "code": self.code,
            "missing_count": len(self.missing),
            "missing": list(self.missing[:MAX_REPORTED_MISSING]),
            "checked_at": round(self.checked_at, 3),
        }


def _table_names(requirements: Iterable[TableRequirement]) -> list[str]:
    return [requirement.table for requirement in requirements]


async def _fetch(conn: Any, query: str, *args: Any) -> list[Any]:
    return list(await conn.fetch(query, *args))


async def collect_missing(conn: Any, requirements: tuple[TableRequirement, ...] = AI_SCHEMA_REQUIREMENTS) -> list[str]:
    """Return the names of missing objects/privileges; empty when everything is in place."""
    missing: list[str] = []
    rows = await _fetch(
        conn,
        "SELECT t.name, to_regclass(t.name)::oid AS oid FROM unnest($1::text[]) AS t(name)",
        _table_names(requirements),
    )
    oids: dict[str, int] = {str(row["name"]): int(row["oid"]) for row in rows if row["oid"] is not None}
    present = [requirement for requirement in requirements if requirement.table in oids]
    missing.extend(f"table:{requirement.table}" for requirement in requirements
                   if requirement.required and requirement.table not in oids)
    if not present:
        return missing

    columns: dict[int, set[str]] = {}
    for row in await _fetch(
        conn,
        "SELECT attrelid::oid AS oid, attname::text AS name FROM pg_attribute "
        "WHERE attrelid = ANY($1::oid[]) AND attnum > 0 AND NOT attisdropped",
        [oids[requirement.table] for requirement in present],
    ):
        columns.setdefault(int(row["oid"]), set()).add(str(row["name"]))

    checks: list[tuple[int, str, str | None, str]] = []
    for requirement in present:
        oid = oids[requirement.table]
        existing = columns.get(oid, set())
        for column in requirement.columns:
            if column not in existing:
                missing.append(f"column:{requirement.table}.{column}")
        for privilege, grant_columns in requirement.grants:
            if grant_columns is None:
                checks.append((oid, privilege, None, f"privilege:{privilege}:{requirement.table}"))
                continue
            for column in grant_columns:
                # Optional tables: like the grant script, only columns that exist are checked.
                if column in existing:
                    checks.append((oid, privilege, column, f"privilege:{privilege}:{requirement.table}.{column}"))
    if checks:
        granted = await _fetch(
            conn,
            "SELECT k.ord, CASE WHEN k.col IS NULL THEN has_table_privilege(k.rel, k.priv) "
            "ELSE has_column_privilege(k.rel, k.col, k.priv) END AS ok "
            "FROM unnest($1::oid[], $2::text[], $3::text[]) WITH ORDINALITY AS k(rel, priv, col, ord)",
            [check[0] for check in checks], [check[1] for check in checks], [check[2] for check in checks],
        )
        allowed = {int(row["ord"]) for row in granted if row["ok"]}
        missing.extend(check[3] for index, check in enumerate(checks, start=1) if index not in allowed)

    missing.extend(await _missing_rls_access(conn, present, oids))
    missing.extend(await _missing_extensions(conn))
    missing.extend(await _missing_schema_usage(conn, [oids[requirement.table] for requirement in present]))
    return missing


async def _missing_rls_access(
    conn: Any,
    present: list[TableRequirement],
    oids: dict[str, int],
) -> list[str]:
    """A table with RLS enabled and no applicable permissive policy silently returns 0 rows."""
    bypass_rows = await _fetch(
        conn, "SELECT rolsuper OR rolbypassrls AS bypass FROM pg_roles WHERE rolname = current_user",
    )
    if bypass_rows and bypass_rows[0]["bypass"]:
        return []
    table_rows = await _fetch(
        conn,
        "SELECT c.oid::oid AS oid, c.relrowsecurity AS rls, c.relforcerowsecurity AS forced, "
        "pg_has_role(c.relowner, 'USAGE') AS owner FROM pg_class c WHERE c.oid = ANY($1::oid[])",
        [oids[requirement.table] for requirement in present],
    )
    enforced = {int(row["oid"]) for row in table_rows if row["rls"] and (row["forced"] or not row["owner"])}
    if not enforced:
        return []
    covered: dict[int, set[Privilege]] = {}
    for row in await _fetch(
        conn,
        "SELECT pol.polrelid::oid AS oid, pol.polcmd::text AS cmd FROM pg_policy pol "
        "WHERE pol.polrelid = ANY($1::oid[]) AND pol.polpermissive "
        "AND (0::oid = ANY(pol.polroles) OR EXISTS (SELECT 1 FROM unnest(pol.polroles) AS r(role_oid) "
        "WHERE r.role_oid <> 0 AND pg_has_role(current_user, r.role_oid, 'MEMBER')))",
        sorted(enforced),
    ):
        covered.setdefault(int(row["oid"]), set()).update(RLS_COMMAND_CODES.get(str(row["cmd"]), frozenset()))
    missing: list[str] = []
    for requirement in present:
        oid = oids[requirement.table]
        if oid not in enforced:
            continue
        for command in sorted(requirement.commands() - covered.get(oid, set())):
            missing.append(f"rls_policy:{command}:{requirement.table}")
    return missing


async def _missing_extensions(conn: Any) -> list[str]:
    missing: list[str] = []
    installed = {str(row["extname"]) for row in await _fetch(
        conn, "SELECT extname::text AS extname FROM pg_extension WHERE extname = ANY($1::text[])",
        list(REQUIRED_EXTENSIONS),
    )}
    missing.extend(f"extension:{name}" for name in REQUIRED_EXTENSIONS if name not in installed)
    rows = await _fetch(
        conn,
        "SELECT 'function:' || f.sig AS name, to_regprocedure(f.sig) IS NOT NULL "
        "AND has_function_privilege(to_regprocedure(f.sig), 'EXECUTE') AS ok FROM unnest($1::text[]) AS f(sig) "
        "UNION ALL SELECT 'type:' || t.sig, to_regtype(t.sig) IS NOT NULL "
        "AND has_type_privilege(to_regtype(t.sig), 'USAGE') FROM unnest($2::text[]) AS t(sig) "
        "UNION ALL SELECT 'operator:' || o.sig, to_regoperator(o.sig) IS NOT NULL FROM unnest($3::text[]) AS o(sig)",
        list(REQUIRED_FUNCTIONS), list(REQUIRED_TYPES), list(REQUIRED_OPERATORS),
    )
    missing.extend(str(row["name"]) for row in rows if not row["ok"])
    return missing


async def _missing_schema_usage(conn: Any, table_oids: list[int]) -> list[str]:
    rows = await _fetch(
        conn,
        "SELECT DISTINCT n.nspname::text AS name, has_schema_privilege(n.oid, 'USAGE') AS ok "
        "FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE c.oid = ANY($1::oid[])",
        table_oids,
    )
    return [f"schema_usage:{row['name']}" for row in rows if not row["ok"]]


async def run_schema_check(pool: Any, *, timeout_seconds: float) -> SchemaCheckResult:
    """Run the check on one pooled connection in a READ ONLY transaction; never raises."""
    try:
        async with asyncio.timeout(timeout_seconds), pool.acquire() as conn, conn.transaction(readonly=True):
            missing = await collect_missing(conn)
    except Exception as error:
        logger.warning(
            "schema_check_unavailable",
            extra={"event": "schema_check_unavailable", "error_type": type(error).__name__},
        )
        return SchemaCheckResult(status="error")
    if missing:
        logger.error(
            "schema_check_failed",
            extra={"event": "schema_check_failed", "missing_count": len(missing),
                   "missing": missing[:MAX_REPORTED_MISSING]},
        )
        return SchemaCheckResult(status="failed", missing=tuple(missing))
    logger.info("schema_check_passed", extra={"event": "schema_check_passed"})
    return SchemaCheckResult(status="ok")


class SchemaGuard:
    """Caches the last result; a failed result is re-checked at most every ``retry_seconds``."""

    def __init__(self, *, retry_seconds: float = 30.0, clock: Any = time.monotonic) -> None:
        self.retry_seconds = retry_seconds
        self._clock = clock
        self.result: SchemaCheckResult | None = None
        self._checked_monotonic = 0.0

    def reset(self) -> None:
        self.result = None
        self._checked_monotonic = 0.0

    def record(self, result: SchemaCheckResult) -> SchemaCheckResult:
        self.result = result
        self._checked_monotonic = float(self._clock())
        return result

    async def refresh(self, pool: Any, *, timeout_seconds: float) -> SchemaCheckResult:
        return self.record(await run_schema_check(pool, timeout_seconds=timeout_seconds))

    async def current(self, pool: Any, *, timeout_seconds: float) -> SchemaCheckResult:
        result = self.result
        if result is not None and (result.ok or float(self._clock()) - self._checked_monotonic < self.retry_seconds):
            return result
        return await self.refresh(pool, timeout_seconds=timeout_seconds)

    def summary(self) -> dict[str, Any]:
        if self.result is None:
            return {"status": "pending", "code": None, "missing_count": 0, "missing": [], "checked_at": None}
        return self.result.summary()
