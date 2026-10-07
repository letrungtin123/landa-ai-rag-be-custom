from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.instructional_quality import ordered_source_steps


SOURCE_EVIDENCE_BUNDLE_VERSION = "source-evidence-bundle-v1"
SOURCE_EVIDENCE_ASSEMBLER_VERSION = "source-evidence-assembler-v1"
LEGACY_SOURCE_EVIDENCE_REVIEW_CODE = "STRUCTURED_EVIDENCE_REVISION_MISSING"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TABLE_ROW_RE = re.compile(r"^Row\s+(\d+)\s*:\s*(.+)$", re.IGNORECASE)
STEP_RE = re.compile(
    r"^(?:(?:step|bước|buoc)\s*)?(\d{1,3})\s*[:.)-]\s+(.+)$",
    re.IGNORECASE,
)


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _text(value: Any, *, maximum: int) -> str:
    normalized = re.sub(r"\s+", " ", str(value or "")).strip()
    if not normalized or len(normalized) > maximum:
        raise ValueError(f"text must contain 1..{maximum} characters")
    return normalized


def _text_list(
    values: Iterable[Any],
    *,
    maximum_items: int,
    maximum_chars: int,
) -> list[str]:
    result: list[str] = []
    for raw in values:
        value = _text(raw, maximum=maximum_chars)
        if value not in result:
            result.append(value)
        if len(result) > maximum_items:
            raise ValueError("text array exceeds its bounded contract")
    return result


class EvidenceLocatorV1(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    page: int | None = Field(default=None, ge=1)
    bbox_normalized: tuple[float, float, float, float] | None = None
    source_ref: str | None = Field(default=None, max_length=255)
    source_chunk: int | None = Field(default=None, ge=0)

    @field_validator("bbox_normalized")
    @classmethod
    def validate_bbox(
        cls,
        value: tuple[float, float, float, float] | None,
    ) -> tuple[float, float, float, float] | None:
        if value is None:
            return None
        x0, y0, x1, y1 = value
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError("bbox must be ordered inside 0..1")
        return tuple(round(item, 6) for item in value)


class SourceEvidenceElementV1(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    evidence_id: str = Field(pattern=r"^sev1_[a-f0-9]{32}$")
    kind: Literal["table", "process", "hierarchy", "visual"]
    representation_status: Literal["parsed", "reviewed", "candidate_unverified"]
    source_fact_ids: tuple[str, ...] = Field(max_length=32_768)
    locator: EvidenceLocatorV1
    payload: dict[str, Any]
    element_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_element(self) -> "SourceEvidenceElementV1":
        if len(set(self.source_fact_ids)) != len(self.source_fact_ids) or any(
            not value or len(value) > 255 for value in self.source_fact_ids
        ):
            raise ValueError("source fact references must be unique and bounded")
        payload = _validate_payload(self.kind, self.payload)
        base = self.model_dump(exclude={"element_hash"}, mode="json")
        base["payload"] = payload
        if canonical_hash(base) != self.element_hash:
            raise ValueError("evidence element hash is invalid")
        return self


class SourceEvidenceBundleV1(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    contract_version: Literal["source-evidence-bundle-v1"]
    assembler_version: Literal["source-evidence-assembler-v1"]
    source_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_source_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    materialized_source_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    locale: Literal["vi", "en"]
    status: Literal["ready", "review_required", "not_required"]
    blocking: Literal[False]
    draft_visibility: Literal["preserved"]
    elements: tuple[SourceEvidenceElementV1, ...] = Field(max_length=256)
    review_requirements: tuple[str, ...] = Field(max_length=256)
    bundle_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_bundle(self) -> "SourceEvidenceBundleV1":
        ids = [element.evidence_id for element in self.elements]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence identifiers must be unique")
        if len(set(self.review_requirements)) != len(self.review_requirements):
            raise ValueError("review requirements must be unique")
        expected_status = (
            "review_required"
            if self.review_requirements
            else "ready"
            if self.elements
            else "not_required"
        )
        if self.status != expected_status:
            raise ValueError("evidence bundle status is inconsistent")
        materialized = canonical_hash({
            "assembler_version": self.assembler_version,
            "base_source_revision": self.base_source_revision,
            "elements": [element.model_dump(mode="json") for element in self.elements],
            "review_requirements": list(self.review_requirements),
        })
        if materialized != self.materialized_source_revision:
            raise ValueError("materialized source revision is invalid")
        base = self.model_dump(exclude={"bundle_hash"}, mode="json")
        if canonical_hash(base) != self.bundle_hash:
            raise ValueError("evidence bundle hash is invalid")
        return self


def _validate_payload(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("evidence payload must be an object")
    if kind == "table":
        if set(payload) != {"headers", "rows", "notes", "conditions"}:
            raise ValueError("table payload shape is invalid")

        def cells(value: Any) -> list[str]:
            if not isinstance(value, list) or not 1 <= len(value) <= 24:
                raise ValueError("table cell array is invalid")
            result: list[str] = []
            for raw_cell in value:
                cell = re.sub(r"\s+", " ", str(raw_cell or "")).strip()
                if len(cell) > 1000:
                    raise ValueError("table cell exceeds its bounded contract")
                result.append(cell)
            if not any(result):
                raise ValueError("table row cannot be empty")
            return result

        raw_headers = payload.get("headers")
        headers = cells(raw_headers) if isinstance(raw_headers, list) and raw_headers else []
        raw_rows = payload.get("rows")
        if not isinstance(raw_rows, list) or not raw_rows or len(raw_rows) > 64:
            raise ValueError("table rows are invalid")
        rows = [cells(raw_row) for raw_row in raw_rows]
        width = max(len(row) for row in rows)
        if any(len(row) != width for row in rows):
            raise ValueError("table rows must preserve a stable column width")
        if headers and len(headers) != max(len(row) for row in rows):
            raise ValueError("table headers do not match row width")
        return {
            "headers": headers,
            "rows": rows,
            "notes": _text_list(
                payload.get("notes") or (), maximum_items=12, maximum_chars=500,
            ),
            "conditions": _text_list(
                payload.get("conditions") or (), maximum_items=12, maximum_chars=500,
            ),
        }
    if kind == "process":
        if set(payload) != {"steps", "order_constraints", "exceptions"}:
            raise ValueError("process payload shape is invalid")
        steps = _text_list(
            payload.get("steps") or (), maximum_items=20, maximum_chars=500,
        )
        if len(steps) < 2:
            raise ValueError("process requires at least two ordered steps")
        constraints = payload.get("order_constraints")
        if not isinstance(constraints, list) or len(constraints) > 40:
            raise ValueError("process order constraints are invalid")
        normalized_constraints: list[dict[str, int]] = []
        for item in constraints:
            if not isinstance(item, dict) or set(item) != {"before", "after"}:
                raise ValueError("process order constraint shape is invalid")
            before, after = item.get("before"), item.get("after")
            if type(before) is not int or type(after) is not int or not (0 <= before < after < len(steps)):
                raise ValueError("process order constraint is invalid")
            normalized_constraints.append({"before": before, "after": after})
        return {
            "steps": steps,
            "order_constraints": normalized_constraints,
            "exceptions": _text_list(
                payload.get("exceptions") or (), maximum_items=12, maximum_chars=500,
            ),
        }
    if kind == "hierarchy":
        if set(payload) != {"nodes", "root_ids"}:
            raise ValueError("hierarchy payload shape is invalid")
        raw_nodes = payload.get("nodes")
        if not isinstance(raw_nodes, list) or not raw_nodes or len(raw_nodes) > 64:
            raise ValueError("hierarchy nodes are invalid")
        nodes: list[dict[str, Any]] = []
        ids: set[str] = set()
        for item in raw_nodes:
            if not isinstance(item, dict) or set(item) != {"id", "label", "parent_id", "priority"}:
                raise ValueError("hierarchy node shape is invalid")
            node_id = _text(item.get("id"), maximum=96)
            if node_id in ids:
                raise ValueError("hierarchy node identity is duplicated")
            ids.add(node_id)
            parent = item.get("parent_id")
            if parent is not None:
                parent = _text(parent, maximum=96)
            priority = item.get("priority")
            if type(priority) is not int or priority < 0 or priority > 1024:
                raise ValueError("hierarchy priority is invalid")
            nodes.append({
                "id": node_id,
                "label": _text(item.get("label"), maximum=500),
                "parent_id": parent,
                "priority": priority,
            })
        if any(node["parent_id"] is not None and node["parent_id"] not in ids for node in nodes):
            raise ValueError("hierarchy parent is missing")
        root_ids = _text_list(
            payload.get("root_ids") or (), maximum_items=64, maximum_chars=96,
        )
        if not root_ids or any(value not in ids for value in root_ids):
            raise ValueError("hierarchy roots are invalid")
        return {"nodes": nodes, "root_ids": root_ids}
    if kind == "visual":
        if set(payload) != {
            "asset_revision", "region_kind", "prompt_text", "observed_facts", "inference_claims",
        }:
            raise ValueError("visual payload shape is invalid")
        asset_revision = str(payload.get("asset_revision") or "").strip().casefold()
        if not SHA256_RE.fullmatch(asset_revision):
            raise ValueError("visual asset revision is invalid")
        return {
            "asset_revision": asset_revision,
            "region_kind": _text(payload.get("region_kind"), maximum=80),
            "prompt_text": (
                _text(payload.get("prompt_text"), maximum=1200)
                if str(payload.get("prompt_text") or "").strip()
                else None
            ),
            "observed_facts": _text_list(
                payload.get("observed_facts") or (), maximum_items=20, maximum_chars=500,
            ),
            "inference_claims": _text_list(
                payload.get("inference_claims") or (), maximum_items=20, maximum_chars=500,
            ),
        }
    raise ValueError("unsupported evidence kind")


def _fact_value(fact: Any, field: str, default: Any = None) -> Any:
    if isinstance(fact, dict):
        return fact.get(field, default)
    return getattr(fact, field, default)


def _locator_for_fact(fact: Any) -> dict[str, Any]:
    locator = _fact_value(fact, "locator", {})
    return locator if isinstance(locator, dict) else {}


def _evidence_locator(
    facts: list[Any],
    *,
    explicit: dict[str, Any] | None = None,
) -> EvidenceLocatorV1:
    explicit = explicit if isinstance(explicit, dict) else {}
    first = facts[0] if facts else {}
    raw_bbox = explicit.get("bbox_normalized")
    bbox: tuple[float, float, float, float] | None = None
    if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
        bbox = tuple(float(item) for item in raw_bbox)
    page = explicit.get("page") or _fact_value(first, "source_page")
    source_chunk = _fact_value(first, "source_chunk")
    source_ref = _fact_value(first, "source_ref")
    return EvidenceLocatorV1(
        page=int(page) if type(page) is int and page > 0 else None,
        bbox_normalized=bbox,
        source_ref=str(source_ref).strip()[:255] if source_ref else None,
        source_chunk=int(source_chunk) if type(source_chunk) is int and source_chunk >= 0 else None,
    )


def _element(
    *,
    kind: Literal["table", "process", "hierarchy", "visual"],
    representation_status: Literal["parsed", "reviewed", "candidate_unverified"],
    fact_ids: Iterable[str],
    locator: EvidenceLocatorV1,
    payload: dict[str, Any],
) -> SourceEvidenceElementV1:
    normalized_payload = _validate_payload(kind, payload)
    normalized_fact_ids = tuple(dict.fromkeys(
        str(value).strip() for value in fact_ids if str(value).strip()
    ))
    identity = {
        "kind": kind,
        "representation_status": representation_status,
        "source_fact_ids": normalized_fact_ids,
        "locator": locator.model_dump(mode="json"),
        "payload": normalized_payload,
    }
    evidence_id = "sev1_" + canonical_hash(identity)[:32]
    base = {"evidence_id": evidence_id, **identity}
    return SourceEvidenceElementV1(
        evidence_id=evidence_id,
        kind=kind,
        representation_status=representation_status,
        source_fact_ids=normalized_fact_ids,
        locator=locator,
        payload=normalized_payload,
        element_hash=canonical_hash(base),
    )


def _table_elements(facts: list[Any]) -> list[SourceEvidenceElementV1]:
    groups: list[list[tuple[Any, int, list[str]]]] = []
    current: list[tuple[Any, int, list[str]]] = []
    previous_number = 0
    for fact in facts:
        text = re.sub(r"\s+", " ", str(_fact_value(fact, "fact_text", ""))).strip()
        match = TABLE_ROW_RE.match(text)
        if not match:
            if current:
                groups.append(current)
                current = []
                previous_number = 0
            continue
        number = int(match.group(1))
        # Empty cells are positional evidence. Do not collapse them: doing so
        # silently shifts every following value into the wrong source column.
        cells = [
            cell.strip().replace("¦", "|")
            for cell in re.split(r"\s*\|\s*", match.group(2))
        ][:24]
        if len(cells) < 2 or not any(cells):
            continue
        if current and number <= previous_number:
            groups.append(current)
            current = []
        current.append((fact, number, cells))
        previous_number = number
    if current:
        groups.append(current)

    elements: list[SourceEvidenceElementV1] = []
    for group in groups:
        rows = [cells for _fact, _number, cells in group][:64]
        width = max(len(row) for row in rows)
        normalized_rows = [row + [""] * (width - len(row)) for row in rows]
        # Retain row 1 in rows as well as headers. This is deliberate: the
        # writer can reproduce the exact relation and audits can compare every
        # original cell without a hidden header transformation.
        headers = normalized_rows[0] if len(normalized_rows) >= 2 else []
        group_facts = [item[0] for item in group]
        locator = _evidence_locator(group_facts)
        notes: list[str] = []
        conditions: list[str] = []
        for fact in group_facts:
            fact_locator = _locator_for_fact(fact)
            for key, target in (("table_notes", notes), ("table_conditions", conditions)):
                raw_values = fact_locator.get(key)
                if isinstance(raw_values, list):
                    for value in raw_values:
                        normalized = re.sub(r"\s+", " ", str(value)).strip()
                        if normalized and normalized not in target:
                            target.append(normalized[:500])
        elements.append(_element(
            kind="table",
            representation_status="parsed",
            fact_ids=[_fact_value(fact, "fact_key", "") for fact in group_facts],
            locator=locator,
            payload={
                "headers": headers,
                "rows": normalized_rows,
                "notes": notes[:12],
                "conditions": conditions[:12],
            },
        ))
    return elements


def _process_elements(facts: list[Any], *, locale: str) -> list[SourceEvidenceElementV1]:
    texts = [str(_fact_value(fact, "fact_text", "")) for fact in facts]
    steps = ordered_source_steps(texts, locale=locale)[:20]
    if len(steps) < 2:
        return []
    step_fact_ids = [
        str(_fact_value(fact, "fact_key", ""))
        for fact in facts
        if STEP_RE.match(re.sub(r"\s+", " ", str(_fact_value(fact, "fact_text", ""))).strip())
    ]
    if not step_fact_ids:
        return []
    return [_element(
        kind="process",
        representation_status="parsed",
        fact_ids=step_fact_ids,
        locator=_evidence_locator(facts),
        payload={
            "steps": steps,
            "order_constraints": [
                {"before": index, "after": index + 1}
                for index in range(len(steps) - 1)
            ],
            "exceptions": [],
        },
    )]


def _hierarchy_elements(facts: list[Any]) -> list[SourceEvidenceElementV1]:
    path_facts: dict[tuple[str, ...], list[str]] = {}
    for fact in facts:
        locator = _locator_for_fact(fact)
        raw_path = locator.get("heading_path") or locator.get("scope_title")
        if isinstance(raw_path, list):
            segments = [re.sub(r"\s+", " ", str(item)).strip() for item in raw_path]
        else:
            segments = [
                part.strip()
                for part in re.split(r"\s*[›>]\s*", str(raw_path or ""))
            ]
        segments = [part[:500] for part in segments if part]
        if not segments:
            continue
        path_facts.setdefault(tuple(segments[:8]), []).append(str(_fact_value(fact, "fact_key", "")))
    if not path_facts:
        return []
    nodes_by_path: dict[tuple[str, ...], dict[str, Any]] = {}
    for path in sorted(path_facts, key=lambda value: (len(value), value)):
        for index in range(len(path)):
            current = path[: index + 1]
            if current in nodes_by_path:
                continue
            if len(nodes_by_path) >= 64:
                break
            node_id = "hen1_" + canonical_hash(list(current))[:24]
            parent = nodes_by_path.get(current[:-1])
            if index > 0 and parent is None:
                break
            nodes_by_path[current] = {
                "id": node_id,
                "label": current[-1],
                "parent_id": parent["id"] if parent else None,
                "priority": index,
            }
    roots = [node["id"] for path, node in nodes_by_path.items() if len(path) == 1]
    fact_ids = [fact_id for values in path_facts.values() for fact_id in values]
    return [_element(
        kind="hierarchy",
        representation_status="parsed",
        fact_ids=fact_ids,
        locator=_evidence_locator(facts),
        payload={"nodes": list(nodes_by_path.values()), "root_ids": roots},
    )]


def _visual_elements(facts: list[Any]) -> tuple[list[SourceEvidenceElementV1], list[str]]:
    by_region: dict[str, dict[str, Any]] = {}
    for fact in facts:
        locator = _locator_for_fact(fact)
        regions = locator.get("visual_regions")
        if not isinstance(regions, list):
            continue
        for raw in regions[:8]:
            if not isinstance(raw, dict):
                continue
            asset_revision = str(raw.get("asset_revision") or "").strip().casefold()
            raw_locator = raw.get("locator") if isinstance(raw.get("locator"), dict) else {}
            raw_bbox = raw_locator.get("bbox_normalized")
            if not SHA256_RE.fullmatch(asset_revision) or not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
                continue
            key = canonical_hash({"asset_revision": asset_revision, "locator": raw_locator})
            entry = by_region.setdefault(key, {
                "facts": [], "asset_revision": asset_revision,
                "region_kind": str(raw.get("region_kind") or "image")[:80],
                "locator": raw_locator,
                "prompt_text": str(
                    raw.get("prompt_text") or locator.get("visual_prompt_text") or ""
                ).strip()[:1200] or None,
                "observed_facts": [], "inference_claims": [],
            })
            fact_id = str(_fact_value(fact, "fact_key", ""))
            if fact_id and fact_id not in entry["facts"]:
                entry["facts"].append(fact_id)
            observation = raw.get("observation") if isinstance(raw.get("observation"), dict) else {}
            inference = raw.get("inference") if isinstance(raw.get("inference"), dict) else {}
            observation_status = str(observation.get("status") or "").strip().casefold()
            observed_values = (
                observation.get("facts", [])
                if observation_status in {"observed", "reviewed"}
                and isinstance(observation.get("facts"), list)
                else []
            )
            for value in observed_values:
                normalized = re.sub(r"\s+", " ", str(value)).strip()
                if normalized and normalized not in entry["observed_facts"]:
                    entry["observed_facts"].append(normalized[:500])
            for value in inference.get("claims", []) if isinstance(inference.get("claims"), list) else []:
                normalized = re.sub(r"\s+", " ", str(value)).strip()
                if normalized and normalized not in entry["inference_claims"]:
                    entry["inference_claims"].append(normalized[:500])

    elements: list[SourceEvidenceElementV1] = []
    requirements: list[str] = []
    for entry in by_region.values():
        observed = entry["observed_facts"][:20]
        status: Literal["reviewed", "candidate_unverified"] = (
            "reviewed" if observed else "candidate_unverified"
        )
        element = _element(
            kind="visual",
            representation_status=status,
            fact_ids=entry["facts"],
            locator=_evidence_locator(facts, explicit=entry["locator"]),
            payload={
                "asset_revision": entry["asset_revision"],
                "region_kind": entry["region_kind"],
                "prompt_text": entry["prompt_text"],
                "observed_facts": observed,
                "inference_claims": entry["inference_claims"][:20],
            },
        )
        elements.append(element)
        if status == "candidate_unverified":
            requirements.append(f"VISUAL_OBSERVATION_REQUIRED:{element.evidence_id}")
        if entry["inference_claims"]:
            requirements.append(f"VISUAL_INFERENCE_REQUIRES_VERIFICATION:{element.evidence_id}")
    return elements, requirements


def _base_source_revision(source_snapshot_hash: str, facts: list[Any]) -> str:
    revisions = sorted({
        str(_locator_for_fact(fact).get("source_revision") or "").strip().casefold()
        for fact in facts
        if SHA256_RE.fullmatch(str(_locator_for_fact(fact).get("source_revision") or "").strip().casefold())
    })
    if len(revisions) == 1:
        return revisions[0]
    return canonical_hash({
        "source_snapshot_hash": source_snapshot_hash,
        "source_revisions": revisions,
        "fact_hashes": [
            canonical_hash({
                "fact_key": str(_fact_value(fact, "fact_key", "")),
                "fact_text": str(_fact_value(fact, "fact_text", "")),
                "locator": _locator_for_fact(fact),
            })
            for fact in facts
        ],
    })


def build_source_evidence_bundle(
    *,
    source_snapshot_hash: str,
    source_facts: Iterable[Any],
    locale: str,
) -> SourceEvidenceBundleV1:
    """Materialize one immutable writer bundle from a frozen fact contract.

    The function does not mutate the source facts and never changes canonical
    ownership. Structured evidence only references existing fact IDs. Missing
    visual observations keep the draft visible and explicitly review-required.
    """

    snapshot = str(source_snapshot_hash or "").strip().casefold()
    if not SHA256_RE.fullmatch(snapshot):
        raise ValueError("source_snapshot_hash must be a SHA-256 digest")
    normalized_locale = str(locale or "").strip().casefold()
    if normalized_locale not in {"vi", "en"}:
        raise ValueError("locale must be vi or en")
    facts = list(source_facts)
    fact_ids = [str(_fact_value(fact, "fact_key", "")).strip() for fact in facts]
    if not facts or any(not value or len(value) > 255 for value in fact_ids) or len(fact_ids) != len(set(fact_ids)):
        raise ValueError("source facts must have unique bounded identities")

    visual_elements, visual_requirements = _visual_elements(facts)
    elements = [
        *_table_elements(facts),
        *_process_elements(facts, locale=normalized_locale),
        *_hierarchy_elements(facts),
        *visual_elements,
    ]
    # Evidence identity is content-addressed, so a duplicate locator/content
    # can be removed without changing semantic ownership.
    unique_elements = {element.evidence_id: element for element in elements}
    ordered_elements = tuple(
        unique_elements[key]
        for key in sorted(unique_elements)
    )
    raw_review_requirements = list(dict.fromkeys(visual_requirements))
    if any(
        str(_locator_for_fact(fact).get("source_evidence_status") or "").strip().casefold()
        == "legacy_review_required"
        for fact in facts
    ):
        raw_review_requirements.append(LEGACY_SOURCE_EVIDENCE_REVIEW_CODE)
    if len(elements) > 256:
        raw_review_requirements.append("EVIDENCE_ELEMENT_LIMIT_REACHED")
    ordered_elements = ordered_elements[:256]
    if len(raw_review_requirements) > 256:
        raw_review_requirements = [
            *raw_review_requirements[:255],
            "EVIDENCE_REVIEW_REQUIREMENT_LIMIT_REACHED",
        ]
    review_requirements = tuple(dict.fromkeys(raw_review_requirements))
    base_revision = _base_source_revision(snapshot, facts)
    materialized_revision = canonical_hash({
        "assembler_version": SOURCE_EVIDENCE_ASSEMBLER_VERSION,
        "base_source_revision": base_revision,
        "elements": [element.model_dump(mode="json") for element in ordered_elements],
        "review_requirements": list(review_requirements),
    })
    status: Literal["ready", "review_required", "not_required"] = (
        "review_required"
        if review_requirements
        else "ready"
        if ordered_elements
        else "not_required"
    )
    base = {
        "contract_version": SOURCE_EVIDENCE_BUNDLE_VERSION,
        "assembler_version": SOURCE_EVIDENCE_ASSEMBLER_VERSION,
        "source_snapshot_hash": snapshot,
        "base_source_revision": base_revision,
        "materialized_source_revision": materialized_revision,
        "locale": normalized_locale,
        "status": status,
        "blocking": False,
        "draft_visibility": "preserved",
        "elements": tuple(element.model_dump(mode="json") for element in ordered_elements),
        "review_requirements": review_requirements,
    }
    model_base = {**base, "elements": ordered_elements}
    return SourceEvidenceBundleV1(**model_base, bundle_hash=canonical_hash(base))


def build_degraded_source_evidence_bundle(
    *,
    source_snapshot_hash: str,
    source_facts: Iterable[Any],
    locale: str,
    reason_code: str = "EVIDENCE_BUNDLE_ASSEMBLY_FAILED",
) -> SourceEvidenceBundleV1:
    """Return a content-addressed, review-required bundle after safe degradation.

    Optional evidence enrichment must never erase an otherwise valid draft. The
    degraded bundle contains no semantic claims, owns no facts and makes the
    missing representation explicit to downstream review policy.
    """

    snapshot = str(source_snapshot_hash or "").strip().casefold()
    if not SHA256_RE.fullmatch(snapshot):
        raise ValueError("source_snapshot_hash must be a SHA-256 digest")
    normalized_locale = str(locale or "").strip().casefold()
    if normalized_locale not in {"vi", "en"}:
        raise ValueError("locale must be vi or en")
    normalized_reason = str(reason_code or "").strip().upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,99}", normalized_reason):
        raise ValueError("reason_code must be a bounded machine code")
    facts = list(source_facts)
    base_revision = _base_source_revision(snapshot, facts)
    review_requirements = (normalized_reason,)
    materialized_revision = canonical_hash({
        "assembler_version": SOURCE_EVIDENCE_ASSEMBLER_VERSION,
        "base_source_revision": base_revision,
        "elements": [],
        "review_requirements": list(review_requirements),
    })
    base = {
        "contract_version": SOURCE_EVIDENCE_BUNDLE_VERSION,
        "assembler_version": SOURCE_EVIDENCE_ASSEMBLER_VERSION,
        "source_snapshot_hash": snapshot,
        "base_source_revision": base_revision,
        "materialized_source_revision": materialized_revision,
        "locale": normalized_locale,
        "status": "review_required",
        "blocking": False,
        "draft_visibility": "preserved",
        "elements": (),
        "review_requirements": review_requirements,
    }
    return SourceEvidenceBundleV1(**base, bundle_hash=canonical_hash(base))
