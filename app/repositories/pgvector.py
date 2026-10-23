"""pgvector parameter encoding."""

from __future__ import annotations

from collections.abc import Iterable


def vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{float(value):.8f}" for value in values) + "]"


def vector_values(values: Iterable[float]) -> list[float]:
    """The values ``vector_literal`` stores, for a ``real[]`` parameter cast to vector (audit 2026-10-24, P1).

    Each value is rounded to the literal's 8 decimals first. pgvector reads a literal's decimal as the nearest
    float4 (strtof); asyncpg sends the nearest float4 of the rounded double. The two are the same float4 for every
    |value| < 65,536: a float4 rounding midpoint differs from an 8-decimal number by far more than a double's
    rounding error, so no midpoint lies between the decimal and its double. Stored vectors stay bit for bit those
    of the literal (the index fingerprint does not change).
    """
    return [float(f"{float(value):.8f}") for value in values]
