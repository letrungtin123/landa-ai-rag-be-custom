"""PRD-2: every SQL statement of the service lives in ``app/repositories``.

Keeping the SQL in one package lets these tests check it against the database contract the
least-privilege role is granted (``app.infra.schema_check.AI_SCHEMA_REQUIREMENTS``, which mirrors
``supabase/manual_sql/20261008_1600_ai_rag_least_privilege_role.sql``).
"""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

from app.infra.schema_check import AI_SCHEMA_REQUIREMENTS

APP_ROOT = Path(__file__).resolve().parents[1] / "app"
REPOSITORIES = APP_ROOT / "repositories"
EXECUTOR_METHODS = frozenset({"fetch", "fetchrow", "fetchval", "execute", "executemany"})
# Read-only catalog introspection of the startup schema check (pg_catalog / information_schema).
CATALOG_SQL_MODULES = frozenset({APP_ROOT / "infra" / "schema_check.py"})
SQL_STATEMENT = re.compile(r"\b(?:SELECT\b.+\bFROM|INSERT\s+INTO|DELETE\s+FROM|UPDATE\s+\w+\s+SET)\b", re.DOTALL)


def sql_calls(path: Path) -> list[tuple[int, str]]:
    """``<executor>.<method>("<SQL>", ...)`` calls with a literal statement."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        (node.lineno, node.args[0].value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in EXECUTOR_METHODS
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ]


def table_privileges(sql: str) -> set[tuple[str, str]]:
    """(privilege, table) pairs a statement needs, from its verbs and FROM/JOIN targets."""
    text = " ".join(sql.split())
    needed: set[tuple[str, str]] = set()
    for table in re.findall(r"\bINSERT INTO (\w+)", text):
        needed.add(("INSERT", table))
        if "ON CONFLICT" in text and "DO UPDATE" in text:
            needed.add(("UPDATE", table))
    for table in re.findall(r"\bUPDATE (\w+) SET\b", text):
        needed.add(("UPDATE", table))
    for table in re.findall(r"\bDELETE FROM (\w+)", text):
        needed.add(("DELETE", table))
    # Relations only: lateral subqueries and set-returning functions (``unnest(...)``) are not tables.
    select_tables = re.findall(r"\b(?:FROM|JOIN) (?!LATERAL\b)(\w+)\b(?!\s*\()", text)
    deleted = set(re.findall(r"\bDELETE FROM (\w+)", text))
    for table in select_tables:
        if table not in deleted:
            needed.add(("SELECT", table))
            if "FOR UPDATE" in text:
                needed.add(("UPDATE", table))
    return needed


class RepositorySqlTests(unittest.TestCase):
    def test_sql_is_executed_only_from_repositories(self) -> None:
        offenders = [
            f"{path.relative_to(APP_ROOT.parent).as_posix()}:{line}"
            for path in sorted(APP_ROOT.rglob("*.py"))
            if REPOSITORIES not in path.parents and path not in CATALOG_SQL_MODULES
            for line, _ in sql_calls(path)
        ]
        self.assertEqual(offenders, [])

    def test_no_sql_statement_text_outside_repositories(self) -> None:
        offenders = []
        for path in sorted(APP_ROOT.rglob("*.py")):
            if REPOSITORIES in path.parents or path in CATALOG_SQL_MODULES:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            offenders.extend(
                f"{path.relative_to(APP_ROOT.parent).as_posix()}:{node.lineno}"
                for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and SQL_STATEMENT.search(node.value)
            )
        self.assertEqual(offenders, [])

    def test_repository_statements_stay_within_the_granted_tables_and_privileges(self) -> None:
        granted = {(privilege, requirement.table)
                   for requirement in AI_SCHEMA_REQUIREMENTS for privilege in requirement.commands()}
        statements = [(path, line, sql) for path in sorted(REPOSITORIES.glob("*.py")) for line, sql in sql_calls(path)]
        # Guard against the scan silently finding nothing.
        self.assertGreaterEqual(len(statements), 20)
        missing = sorted(
            f"{path.name}:{line} {privilege} {table}"
            for path, line, sql in statements
            for privilege, table in table_privileges(sql)
            if (privilege, table) not in granted
        )
        self.assertEqual(missing, [])

    def test_privilege_parser_reads_verbs_joins_locks_and_upserts(self) -> None:
        self.assertEqual(
            table_privileges("SELECT a FROM t1 x JOIN t2 y ON y.id = x.id WHERE x.id = $1 FOR UPDATE"),
            {("SELECT", "t1"), ("SELECT", "t2"), ("UPDATE", "t1"), ("UPDATE", "t2")},
        )
        self.assertEqual(table_privileges("INSERT INTO t (a) VALUES ($1) ON CONFLICT (a) DO UPDATE SET a = $1"),
                         {("INSERT", "t"), ("UPDATE", "t")})
        self.assertEqual(table_privileges("DELETE FROM t WHERE a = $1"), {("DELETE", "t")})
        self.assertEqual(
            table_privileges("SELECT 1 FROM t LEFT JOIN LATERAL (SELECT 1 FROM unnest($1::text[]) AS p(v)) m ON true"),
            {("SELECT", "t")},
        )
        self.assertEqual(table_privileges("UPDATE t SET a = $1 WHERE b = $2"), {("UPDATE", "t")})


if __name__ == "__main__":
    unittest.main()
