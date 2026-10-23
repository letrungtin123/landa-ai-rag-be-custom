"""Batched chunk inserts (audit 2026-10-24, P1): multi-row INSERTs with the vector as rounded real values.

The stored float4 values must be those of the old text literal (pgvector reads its decimals with strtof, the
nearest float4); ``nearest_float4`` computes that exactly with fractions, so these checks need no database.
"""

from __future__ import annotations

import asyncio
import random
import re
import struct
import unittest
from fractions import Fraction
from typing import Any

from app.infra.schema_check import AI_SCHEMA_REQUIREMENTS
from app.repositories import indexing
from app.repositories.pgvector import vector_literal, vector_values
from tests.unit.repositories.test_repository_sql import table_privileges


def float4(value: float) -> float:
    """What asyncpg sends for a real: the C cast of the double (round to nearest, ties to even)."""
    return float(struct.unpack("<f", struct.pack("<f", value))[0])


def float4_neighbours(value: float) -> list[float]:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return [float(struct.unpack("<f", struct.pack("<I", bits + step))[0]) for step in (-1, 0, 1)
            if 0 <= bits + step < 0x7F800000]


def nearest_float4(decimal: str) -> float:
    """The float4 nearest the exact decimal (ties to an even mantissa), as strtof reads it."""
    exact = Fraction(decimal)
    candidates = float4_neighbours(abs(float4(float(exact))))
    best = min(candidates, key=lambda item: (abs(Fraction(item) - abs(exact)),
                                             struct.unpack("<I", struct.pack("<f", item))[0] & 1))
    return -best if exact < 0 else best


class VectorValuesTests(unittest.TestCase):
    def test_each_value_is_the_float4_pgvector_reads_from_the_literal(self) -> None:
        generator = random.Random(20261024)  # noqa: S311 - seeded test data, not security
        values = [generator.gauss(0, 0.04) for _ in range(60_000)]
        values += [generator.uniform(-1, 1) for _ in range(20_000)]
        values += [generator.uniform(-1, 1) * 10.0 ** -generator.randint(1, 9) for _ in range(20_000)]
        # 8-decimal numbers as close as they get to a float4 rounding midpoint.
        for _ in range(20_000):
            low = float4(generator.uniform(-2, 2))
            high = float4_neighbours(abs(low))[-1] * (1 if low >= 0 else -1)
            values.append(float((Fraction(low) + Fraction(high)) / 2))
        values += [0.0, -0.0, 1.0, -1.0, 0.5, 1e-8, -1e-8, 4.9999999e-9, 5e-9, 0.123456785, 65_535.999999995]
        literal = vector_literal(values)[1:-1].split(",")
        rounded = vector_values(values)
        self.assertEqual(len(rounded), len(literal))
        mismatches = [(decimal, value) for decimal, value in zip(literal, rounded, strict=True)
                      if nearest_float4(decimal) != float4(value)]
        self.assertEqual(mismatches, [])
        self.assertEqual(float4(vector_values([-0.0])[0]), -0.0)
        self.assertEqual(str(vector_values([-1e-12])[0]), "-0.0")  # "-0.00000000" in the literal

    def test_values_come_from_any_iterable(self) -> None:
        self.assertEqual(vector_values(iter([0.123456789, 2])), [0.12345679, 2.0])


class RecordingConnection:
    def __init__(self) -> None:
        self.statements: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, query: str, *args: Any) -> str:
        self.statements.append((query, args))
        return "INSERT 0 0"


def chunk_row(number: int) -> tuple[Any, ...]:
    return ("tenant", "kb", "document", "index", number, f"content {number}", f"hash {number}", 7, None,
            "section", "{}", [0.25, -0.5])


class InsertChunksTests(unittest.TestCase):
    def test_rows_go_in_order_in_batches_of_200(self) -> None:
        connection = RecordingConnection()
        rows = [chunk_row(number) for number in range(401)]
        asyncio.run(indexing.insert_chunks(connection, rows))
        self.assertEqual([len(args) // 12 for _, args in connection.statements], [200, 200, 1])
        sent = [args[start:start + 12] for _, args in connection.statements for start in range(0, len(args), 12)]
        self.assertEqual(sent, rows)
        for query, args in connection.statements:
            self.assertEqual(query, indexing.chunk_insert_sql(len(args) // 12))
            self.assertEqual(re.findall(r"\$(\d+)", query), [str(number) for number in range(1, len(args) + 1)])
            self.assertEqual(query.count("::real[]::vector"), len(args) // 12)

    def test_no_rows_no_statement(self) -> None:
        connection = RecordingConnection()
        asyncio.run(indexing.insert_chunks(connection, []))
        self.assertEqual(connection.statements, [])

    def test_one_row_has_the_columns_and_casts_of_the_former_statement(self) -> None:
        self.assertEqual(
            indexing.chunk_insert_sql(1),
            "INSERT INTO rag_chunks (tenant_id, kb_id, document_id, index_id, chunk_no, content, content_hash, "
            "token_count, source_page, source_section, metadata, embedding) VALUES ($1::uuid, $2::uuid, $3::uuid, "
            "$4::uuid, $5::int, $6, $7, $8::int, $9::int, $10, $11::jsonb, $12::real[]::vector)",
        )

    def test_the_statement_stays_within_the_granted_privileges_and_the_parameter_limit(self) -> None:
        granted = {(privilege, requirement.table)
                   for requirement in AI_SCHEMA_REQUIREMENTS for privilege in requirement.commands()}
        self.assertEqual(table_privileges(indexing.chunk_insert_sql(2)), {("INSERT", "rag_chunks")})
        self.assertLessEqual(table_privileges(indexing.chunk_insert_sql(2)), granted)
        self.assertLessEqual(indexing.CHUNK_INSERT_BATCH_ROWS * 12, 32_767)


if __name__ == "__main__":
    unittest.main()
