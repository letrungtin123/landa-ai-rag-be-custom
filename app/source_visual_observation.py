from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable

from app.source_readiness import build_visual_observation_record

SHADOW_VISUAL_PIPELINE_VERSION = "source-visual-shadow-v1"
SHADOW_VISUAL_POLICY_VERSION = "human-review-shadow-v1"
SHADOW_VISUAL_TASK_VERSION = "source-visual-task-v1"


def _canonical_hash(payload: dict[str, Any]) -> str:
    value = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _digest(value: Any, *, field: str) -> str:
    normalized = str(value or "").strip().casefold()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise ValueError(f"{field} must be a SHA-256 hex digest")
    return normalized


def _bounded_text(value: Any, *, field: str, maximum: int) -> str:
    normalized = re.sub(r"\s+", " ", str(value or "")).strip()
    if not normalized or len(normalized) > maximum:
        raise ValueError(f"{field} must contain 1..{maximum} characters")
    return normalized


def _locator(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("locator must be an object")
    page = int(value.get("page") or 0)
    bbox = value.get("bbox_normalized")
    if page < 1 or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("locator must contain a positive page and four-value bbox")
    values = [float(item) for item in bbox]
    x0, y0, x1, y1 = values
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("locator bbox must be ordered inside 0..1")
    return {
        "page": page,
        "bbox_normalized": [round(item, 6) for item in values],
    }


def build_visual_cache_key(
    *,
    tenant_scope: str,
    source_revision: str,
    asset_revision: str,
    locator: dict[str, Any],
    locale: str,
    observer_config_version: str,
) -> dict[str, str]:
    """Build a tenant-scoped cache identity without exposing the tenant value."""

    tenant = _bounded_text(tenant_scope, field="tenant_scope", maximum=200)
    source = _digest(source_revision, field="source_revision")
    asset = _digest(asset_revision, field="asset_revision")
    normalized_locator = _locator(locator)
    normalized_locale = _bounded_text(locale, field="locale", maximum=20).casefold()
    config = _bounded_text(
        observer_config_version,
        field="observer_config_version",
        maximum=100,
    )
    tenant_scope_hash = hashlib.sha256(tenant.encode("utf-8")).hexdigest()
    cache_key = _canonical_hash({
        "pipeline_version": SHADOW_VISUAL_PIPELINE_VERSION,
        "policy_version": SHADOW_VISUAL_POLICY_VERSION,
        "tenant_scope_hash": tenant_scope_hash,
        "source_revision": source,
        "asset_revision": asset,
        "locator": normalized_locator,
        "locale": normalized_locale,
        "observer_config_version": config,
    })
    return {
        "cache_key": cache_key,
        "tenant_scope_hash": tenant_scope_hash,
    }


def _build_task(
    *,
    tenant_scope: str,
    source_revision: str,
    region: dict[str, Any],
    locale: str,
    observer_config_version: str,
) -> dict[str, Any]:
    asset_revision = _digest(region.get("asset_revision"), field="asset_revision")
    locator = _locator(region.get("locator"))
    region_kind = _bounded_text(region.get("region_kind"), field="region_kind", maximum=80)
    cache = build_visual_cache_key(
        tenant_scope=tenant_scope,
        source_revision=source_revision,
        asset_revision=asset_revision,
        locator=locator,
        locale=locale,
        observer_config_version=observer_config_version,
    )
    task_id = _canonical_hash({
        "contract_version": SHADOW_VISUAL_TASK_VERSION,
        "cache_key": cache["cache_key"],
    })
    return {
        "contract_version": SHADOW_VISUAL_TASK_VERSION,
        "pipeline_version": SHADOW_VISUAL_PIPELINE_VERSION,
        "policy_version": SHADOW_VISUAL_POLICY_VERSION,
        "task_id": task_id,
        "mode": "shadow",
        "blocking": False,
        "draft_visibility": "preserved",
        "status": "review_required",
        "reason": "HUMAN_VISUAL_OBSERVATION_REQUIRED",
        "tenant_scope_hash": cache["tenant_scope_hash"],
        "source_revision": _digest(source_revision, field="source_revision"),
        "asset_revision": asset_revision,
        "region_kind": region_kind,
        "locator": locator,
        "locale": _bounded_text(locale, field="locale", maximum=20).casefold(),
        "observer_config_version": _bounded_text(
            observer_config_version,
            field="observer_config_version",
            maximum=100,
        ),
        "cache_key": cache["cache_key"],
        "provider": {
            "policy": "disabled",
            "called": False,
            "attempt_count": 0,
        },
        "failure_behavior": "preserve_draft_and_require_review",
    }


def build_shadow_visual_observation_plan(
    readiness: dict[str, Any],
    *,
    tenant_scope: str,
    locale: str = "vi",
    observer_config_version: str = "human-review-v1",
    max_regions: int = 64,
) -> dict[str, Any]:
    """Plan bounded human-review tasks without calling a model or blocking drafts.

    This coordinator is intentionally persistence-free in CP3A.1. Invalid
    regions become diagnostics and remain review-required instead of raising a
    workflow-wide failure.
    """

    if not 1 <= int(max_regions) <= 256:
        raise ValueError("max_regions must be within 1..256")
    source = readiness.get("source") if isinstance(readiness, dict) else None
    pages = readiness.get("pages") if isinstance(readiness, dict) else None
    source_revision = source.get("sha256") if isinstance(source, dict) else None
    if not isinstance(pages, list):
        pages = []
    try:
        source_revision = _digest(source_revision, field="source_revision")
        _bounded_text(tenant_scope, field="tenant_scope", maximum=200)
        _bounded_text(locale, field="locale", maximum=20)
        _bounded_text(
            observer_config_version,
            field="observer_config_version",
            maximum=100,
        )
    except (TypeError, ValueError) as error:
        return {
            "pipeline_version": SHADOW_VISUAL_PIPELINE_VERSION,
            "policy_version": SHADOW_VISUAL_POLICY_VERSION,
            "mode": "shadow",
            "blocking": False,
            "draft_visibility": "preserved",
            "status": "degraded_review_required",
            "provider_call_count": 0,
            "tasks": [],
            "diagnostics": [{
                "code": "VISUAL_PLAN_INPUT_INVALID",
                "error_type": type(error).__name__,
            }],
        }

    tasks: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    total_regions = 0
    for page in pages:
        if not isinstance(page, dict):
            continue
        regions = page.get("visual_regions")
        if not isinstance(regions, list):
            regions = []
        for region_index, region in enumerate(regions):
            total_regions += 1
            if len(tasks) >= max_regions:
                continue
            try:
                task = _build_task(
                    tenant_scope=tenant_scope,
                    source_revision=source_revision,
                    region=region,
                    locale=locale,
                    observer_config_version=observer_config_version,
                )
            except (TypeError, ValueError) as error:
                diagnostics.append({
                    "code": "VISUAL_REGION_INVALID",
                    "page": int(page.get("page") or 0),
                    "region_index": region_index,
                    "error_type": type(error).__name__,
                })
                continue
            tasks.append(task)
    if total_regions > max_regions:
        diagnostics.append({
            "code": "VISUAL_REGION_LIMIT_REACHED",
            "total_region_count": total_regions,
            "scheduled_region_count": len(tasks),
            "max_regions": max_regions,
        })
    status = "not_required"
    if tasks or diagnostics:
        status = "review_required"
    return {
        "pipeline_version": SHADOW_VISUAL_PIPELINE_VERSION,
        "policy_version": SHADOW_VISUAL_POLICY_VERSION,
        "mode": "shadow",
        "blocking": False,
        "draft_visibility": "preserved",
        "status": status,
        "provider_call_count": 0,
        "source_revision": source_revision,
        "scheduled_region_count": len(tasks),
        "total_region_count": total_regions,
        "tasks": tasks,
        "diagnostics": diagnostics,
    }


def record_human_visual_observation(
    task: dict[str, Any],
    *,
    observed_facts: Iterable[str],
    observer_role: str,
    inference_claims: Iterable[str] = (),
) -> dict[str, Any]:
    """Bind a bounded human observation to one immutable shadow task."""

    if task.get("contract_version") != SHADOW_VISUAL_TASK_VERSION:
        raise ValueError("unsupported visual task contract")
    if task.get("provider", {}).get("called") is not False:
        raise ValueError("human shadow task cannot report a provider call")
    locator = _locator(task.get("locator"))
    record = build_visual_observation_record(
        source_revision=_digest(task.get("source_revision"), field="source_revision"),
        asset_revision=_digest(task.get("asset_revision"), field="asset_revision"),
        page=locator["page"],
        bbox_normalized=locator["bbox_normalized"],
        observed_facts=observed_facts,
        observer_role=observer_role,
        inference_claims=inference_claims,
    )
    return {
        "pipeline_version": SHADOW_VISUAL_PIPELINE_VERSION,
        "policy_version": SHADOW_VISUAL_POLICY_VERSION,
        "mode": "shadow",
        "blocking": False,
        "draft_visibility": "preserved",
        "status": "observed",
        "task_id": task["task_id"],
        "cache_key": task["cache_key"],
        "tenant_scope_hash": task["tenant_scope_hash"],
        "provider_call_count": 0,
        "record": record,
    }
