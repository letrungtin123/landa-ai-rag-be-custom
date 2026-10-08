"""Immutable V5 source context and the scoped repair target bound."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from app.workflows.contracts import RepairTarget, WorkflowFailure

MAX_WORKFLOW_REPAIR_TARGET_CHARS = 24000
MAX_V5_SCOPED_REPAIR_TARGETS = 8


def _v5_source_context_payload(
    source_map: dict[str, Any],
    source_coverage_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return provenance-only state used to fingerprint immutable V5 inputs.

    The payload intentionally excludes fact/source text. It contains just the
    server-owned identifiers and exact scope membership needed to prove that a
    request has not silently changed its canonical source basis between an
    Architect candidate, a repair patch, allocation and final validation.
    """

    manifest_fact_ids = sorted({
        str(fact.get("fact_id") or "").strip()
        for fact in (source_coverage_manifest or {}).get("facts", [])
        if isinstance(fact, dict) and str(fact.get("fact_id") or "").strip()
    })
    scopes: list[dict[str, Any]] = []
    for scope in source_map.get("source_evidence_scopes", []):
        if not isinstance(scope, dict):
            continue
        scope_id = str(scope.get("id") or "").strip()
        if not scope_id:
            continue
        scopes.append({
            "id": scope_id,
            "document_id": str(scope.get("document_id") or "").strip(),
            "section_id": str(scope.get("section_id") or "").strip(),
            "source_ref": str(scope.get("source_ref") or "").strip(),
            "concept_ids": sorted({
                str(value).strip()
                for value in scope.get("concept_ids", [])
                if isinstance(value, str) and value.strip()
            }),
            "source_fact_ids": sorted({
                str(value).strip()
                for value in scope.get("source_fact_ids", [])
                if isinstance(value, str) and value.strip()
            }),
        })
    documents = sorted({
        str(document.get("id") or "").strip()
        for document in source_map.get("documents", [])
        if isinstance(document, dict) and str(document.get("id") or "").strip()
    })
    return {
        "source_map_version": str(source_map.get("version") or ""),
        "document_ids": documents,
        "canonical_fact_ids": manifest_fact_ids,
        "evidence_scopes": sorted(scopes, key=lambda value: value["id"]),
        "sections": sorted({
            str(section.get("id") or "").strip()
            for section in source_map.get("sections", [])
            if isinstance(section, dict) and str(section.get("id") or "").strip()
        }),
        "concepts": sorted({
            str(concept.get("id") or "").strip()
            for concept in source_map.get("concepts", [])
            if isinstance(concept, dict) and str(concept.get("id") or "").strip()
        }),
    }


def _v5_source_context_fingerprint(
    source_map: dict[str, Any],
    source_coverage_manifest: dict[str, Any] | None,
) -> str:
    serialized = json.dumps(
        _v5_source_context_payload(source_map, source_coverage_manifest),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


@dataclass(frozen=True)
class V5ImmutableSourceContext:
    """One request-scoped, server-owned V5 source authority.

    No candidate Blueprint is ever used to rebuild this object. Callers get a
    deep copy for validation/allocation so provider repair cannot mutate the
    authoritative map, manifest or exact evidence-scope membership.
    """

    source_map: dict[str, Any]
    source_coverage_manifest: dict[str, Any]
    fingerprint: str
    canonical_fact_count: int
    evidence_scope_count: int
    source_document_ids: tuple[str, ...]

    def source_map_copy(self) -> dict[str, Any]:
        return deepcopy(self.source_map)

    def manifest_copy(self) -> dict[str, Any]:
        return deepcopy(self.source_coverage_manifest)


def create_v5_immutable_source_context(
    source_map: dict[str, Any],
    source_coverage_manifest: dict[str, Any] | None,
) -> V5ImmutableSourceContext:
    """Freeze a complete V5 Source Map/manifest before any provider output."""

    map_snapshot = deepcopy(source_map)
    manifest_snapshot = deepcopy(source_coverage_manifest or {})
    payload = _v5_source_context_payload(map_snapshot, manifest_snapshot)
    fact_ids = payload["canonical_fact_ids"]
    scope_fact_ids = [
        fact_id
        for scope in payload["evidence_scopes"]
        for fact_id in scope["source_fact_ids"]
    ]
    if (
        (fact_ids and not payload["evidence_scopes"])
        or len(scope_fact_ids) != len(fact_ids)
        or len(set(scope_fact_ids)) != len(scope_fact_ids)
        or set(scope_fact_ids) != set(fact_ids)
    ):
        raise WorkflowFailure(
            "V5_SOURCE_CONTEXT_INVARIANT_FAILED",
            "The immutable V5 Source Map does not contain exactly one evidence scope membership for every canonical fact.",
            internal_code="V5_SOURCE_CONTEXT_SCOPE_MEMBERSHIP_INVALID",
            failure_stage="source_map_build",
            diagnostics={
                "canonical_fact_count": len(fact_ids),
                "evidence_scope_fact_membership_count": len(scope_fact_ids),
                "evidence_scope_count": len(payload["evidence_scopes"]),
            },
        )
    return V5ImmutableSourceContext(
        source_map=map_snapshot,
        source_coverage_manifest=manifest_snapshot,
        fingerprint=_v5_source_context_fingerprint(map_snapshot, manifest_snapshot),
        canonical_fact_count=len(fact_ids),
        evidence_scope_count=len(payload["evidence_scopes"]),
        source_document_ids=tuple(payload["document_ids"]),
    )


def assert_v5_immutable_source_context(
    context: V5ImmutableSourceContext | None,
    *,
    stage: str,
) -> V5ImmutableSourceContext:
    """Fail closed if V5 source ownership context is absent or was mutated."""

    if context is None:
        raise WorkflowFailure(
            "V5_SOURCE_CONTEXT_INVARIANT_FAILED",
            "V5 Course Architecture requires an immutable server-owned source context.",
            internal_code="V5_SOURCE_CONTEXT_MISSING",
            failure_stage=stage,
        )
    fingerprint = _v5_source_context_fingerprint(
        context.source_map,
        context.source_coverage_manifest,
    )
    if fingerprint != context.fingerprint:
        raise WorkflowFailure(
            "V5_SOURCE_CONTEXT_INVARIANT_FAILED",
            "Immutable V5 source context changed during course architecture processing.",
            internal_code="V5_SOURCE_CONTEXT_MUTATED",
            failure_stage=stage,
            diagnostics={
                "canonical_fact_count": context.canonical_fact_count,
                "evidence_scope_count": context.evidence_scope_count,
            },
        )
    return context


def assert_v5_scoped_repair_target_bound(
    targets: list[RepairTarget],
    context: V5ImmutableSourceContext,
) -> None:
    """Reject a degraded V5 candidate before it can create a broad prompt."""

    if len(targets) > MAX_V5_SCOPED_REPAIR_TARGETS:
        raise WorkflowFailure(
            "ARCHITECTURE_REPAIR_SCOPE_TOO_LARGE",
            "The V5 candidate has too many repair targets for a bounded local patch request.",
            internal_code="V5_REPAIR_TARGET_SET_TOO_LARGE",
            failure_stage="architecture_repair_target_snapshot",
            diagnostics={
                "repair_target_count": len(targets),
                "max_repair_target_count": MAX_V5_SCOPED_REPAIR_TARGETS,
                "canonical_fact_count": context.canonical_fact_count,
                "evidence_scope_count": context.evidence_scope_count,
            },
        )
