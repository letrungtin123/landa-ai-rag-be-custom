"""pgvector parameter encoding."""

from __future__ import annotations


def vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{float(value):.8f}" for value in values) + "]"
