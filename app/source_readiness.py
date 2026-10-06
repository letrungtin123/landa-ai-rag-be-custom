from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable


SOURCE_READINESS_VERSION = "source-readiness-prototype-v3"
MANUAL_REGION_REQUEST_VERSION = "source-region-request-v1"
VISUAL_OBSERVATION_VERSION = "source-visual-observation-v1"
VISUAL_REGION_VERSION = "source-visual-region-v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bbox(values: Any) -> tuple[float, float, float, float] | None:
    if all(hasattr(values, field) for field in ("x0", "y0", "x1", "y1")):
        values = (values.x0, values.y0, values.x1, values.y1)
    elif not isinstance(values, (list, tuple)) or len(values) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(value) for value in values)
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _normalized_bbox(
    values: tuple[float, float, float, float],
    *,
    width: float,
    height: float,
) -> list[float]:
    return [
        round(max(0.0, min(1.0, values[0] / width)), 6),
        round(max(0.0, min(1.0, values[1] / height)), 6),
        round(max(0.0, min(1.0, values[2] / width)), 6),
        round(max(0.0, min(1.0, values[3] / height)), 6),
    ]


def _area_ratio(
    values: tuple[float, float, float, float],
    *,
    width: float,
    height: float,
) -> float:
    return max(0.0, values[2] - values[0]) * max(0.0, values[3] - values[1]) / (width * height)


def _image_revision(block: dict[str, Any]) -> str | None:
    image = block.get("image")
    if not isinstance(image, (bytes, bytearray)) or not image:
        return None
    return hashlib.sha256(bytes(image)).hexdigest()


def _rendered_region_revision(
    page: Any,
    values: tuple[float, float, float, float],
) -> str | None:
    """Hash rendered region pixels so vector evidence invalidates with its source."""

    try:
        import pymupdf

        pixmap = page.get_pixmap(clip=pymupdf.Rect(values), alpha=False)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return None
    digest = hashlib.sha256()
    digest.update(f"{pixmap.width}x{pixmap.height}x{pixmap.n}".encode("ascii"))
    digest.update(pixmap.samples)
    return digest.hexdigest()


def _page_blocks(page: Any) -> list[dict[str, Any]]:
    page_dict = page.get_text("dict", sort=True) or {}
    return [block for block in page_dict.get("blocks", []) if isinstance(block, dict)]


def _vector_visual_region(
    page: Any,
    drawings: Iterable[dict[str, Any]],
    *,
    width: float,
    height: float,
) -> dict[str, Any] | None:
    """Create one bounded, revisioned region for an unverified vector composition.

    Full-page backgrounds, footer rules, and broad decorative header bands are
    excluded. The region is a locator only; no semantic meaning is inferred.
    """

    boxes: list[tuple[float, float, float, float]] = []
    for drawing in drawings:
        bbox = _bbox(drawing.get("rect")) if isinstance(drawing, dict) else None
        if bbox is None:
            continue
        box_width = bbox[2] - bbox[0]
        box_height = bbox[3] - bbox[1]
        width_ratio = box_width / width
        height_ratio = box_height / height
        area_ratio = _area_ratio(bbox, width=width, height=height)
        if area_ratio < 0.0005 or area_ratio > 0.25:
            continue
        if width_ratio < 0.025 or height_ratio < 0.025:
            continue
        if bbox[1] >= height * 0.88:
            continue
        if width_ratio > 0.75 and height_ratio < 0.25:
            continue
        boxes.append(bbox)
    region = _union_bbox(boxes)
    if region is None:
        return None
    padding = max(4.0, min(width, height) * 0.01)
    region = (
        max(0.0, region[0] - padding),
        max(0.0, region[1] - padding),
        min(width, region[2] + padding),
        min(height, region[3] + padding),
    )
    if _area_ratio(region, width=width, height=height) > 0.85:
        return None
    revision = _rendered_region_revision(page, region)
    if revision is None:
        return None
    return {
        "contract_version": VISUAL_REGION_VERSION,
        "region_kind": "vector_composition",
        "asset_revision": revision,
        "locator": {
            "page": page.number + 1,
            "bbox_normalized": _normalized_bbox(region, width=width, height=height),
        },
        "representation_status": "candidate_unverified",
        "observation": {"status": "unreviewed", "facts": []},
        "inference": {"status": "not_performed", "claims": []},
    }


def _block_text(block: dict[str, Any]) -> str:
    if block.get("type") != 0:
        return ""
    lines: list[str] = []
    for line in block.get("lines", []):
        if not isinstance(line, dict):
            continue
        text = "".join(
            str(span.get("text") or "")
            for span in line.get("spans", [])
            if isinstance(span, dict)
        ).strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def _document_image_occurrences(document: Any) -> Counter[str]:
    occurrences: Counter[str] = Counter()
    for page in document:
        for block in _page_blocks(page):
            if block.get("type") != 1:
                continue
            revision = _image_revision(block)
            if revision:
                occurrences[revision] += 1
    return occurrences


def _has_vertical_overlap(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> bool:
    intersection = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    shorter = min(first[3] - first[1], second[3] - second[1])
    return shorter > 0 and intersection / shorter >= 0.2


def _detect_multi_column(
    text_blocks: Iterable[tuple[tuple[float, float, float, float], str]],
    *,
    width: float,
    height: float,
) -> bool:
    body = [
        (bbox, text)
        for bbox, text in text_blocks
        if bbox[1] >= height * 0.2
        and bbox[3] <= height * 0.9
        and len(text.strip()) >= 8
        and bbox[2] - bbox[0] <= width * 0.48
    ]
    left = [entry for entry in body if (entry[0][0] + entry[0][2]) / 2 <= width * 0.46]
    right = [entry for entry in body if (entry[0][0] + entry[0][2]) / 2 >= width * 0.54]
    if not left or not right:
        return False
    overlaps = sum(
        1
        for left_entry in left
        for right_entry in right
        if _has_vertical_overlap(left_entry[0], right_entry[0])
    )
    return bool(overlaps >= 1 or (len(left) >= 2 and len(right) >= 2))


def _text_corruption_signals(text: str) -> list[str]:
    signals: list[str] = []
    if "\ufffd" in text:
        signals.append("REPLACEMENT_CHARACTER_PRESENT")
    lines = [re.sub(r"\s+", " ", line).strip().casefold() for line in text.splitlines()]
    lines = [line for line in lines if len(line) >= 8]
    if lines:
        duplicate_ratio = 1.0 - len(set(lines)) / len(lines)
        if duplicate_ratio >= 0.25:
            signals.append("DUPLICATE_LINE_RATIO_HIGH")
    tokens = re.findall(r"\w+", text, flags=re.UNICODE)
    if len(tokens) >= 20 and sum(len(token) == 1 for token in tokens) / len(tokens) >= 0.35:
        signals.append("FRAGMENTED_TOKEN_RATIO_HIGH")
    return signals


def _meaningful_cell_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text or re.fullmatch(r"[vV|¦._\-–—]+", text):
        return ""
    return text


def _union_bbox(values: Iterable[tuple[float, float, float, float]]) -> tuple[float, float, float, float] | None:
    boxes = list(values)
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _region_text(page: Any, bbox: tuple[float, float, float, float]) -> str:
    try:
        import pymupdf

        value = page.get_text("text", clip=pymupdf.Rect(bbox), sort=True) or ""
    except (ImportError, AttributeError, TypeError, ValueError):
        value = ""
    return re.sub(r"\s+", " ", value).strip()[:20_000]


def _relation_candidate(
    page: Any,
    *,
    bbox: tuple[float, float, float, float],
    rows: list[list[str]],
    width: float,
    height: float,
    table_index: int,
    extraction_method: str,
    include_candidate_text: bool,
) -> dict[str, Any]:
    text = _region_text(page, bbox)
    payload = {
        "table_index": table_index,
        "locator": {
            "page": page.number + 1,
            "bbox_normalized": _normalized_bbox(bbox, width=width, height=height),
        },
        "row_count": len(rows),
        "column_count": max((len(row) for row in rows), default=0),
        "non_empty_cell_count": sum(bool(_meaningful_cell_text(cell)) for row in rows for cell in row),
        "extraction_method": extraction_method,
        "relation_revision": hashlib.sha256(
            json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "representation_status": "candidate_unverified",
    }
    if include_candidate_text:
        payload["benchmark_text"] = text
    return payload


def _recover_sparse_relation_regions(
    page: Any,
    table: Any,
    rows: list[list[Any]],
    *,
    width: float,
    height: float,
    table_index: int,
    include_candidate_text: bool,
) -> list[dict[str, Any]]:
    """Recover bounded relation regions hidden inside a false full-page table.

    PowerPoint-derived PDFs can expose one decorative page-sized table whose
    inner rows still retain useful cell geometry. This function accepts only
    repeated, spatially bounded column structure; it never invents cells from
    prose and keeps every result unverified until a later evidence stage.
    """

    table_rows = list(getattr(table, "rows", []) or [])
    observations: list[list[dict[str, Any]]] = []
    for row_index, row in enumerate(rows):
        row_cells = list(getattr(table_rows[row_index], "cells", []) or []) \
            if row_index < len(table_rows) else []
        entries: list[dict[str, Any]] = []
        for column_index, value in enumerate(row):
            text = _meaningful_cell_text(value)
            cell_bbox = _bbox(row_cells[column_index]) if column_index < len(row_cells) else None
            if not text or cell_bbox is None:
                continue
            if _area_ratio(cell_bbox, width=width, height=height) >= 0.7:
                continue
            entries.append({"column": column_index, "text": text, "bbox": cell_bbox})
        observations.append(entries)

    runs: list[list[list[dict[str, Any]]]] = []
    current: list[list[dict[str, Any]]] = []
    for entries in observations:
        if len(entries) >= 2:
            current.append(entries)
        else:
            if len(current) >= 4:
                runs.append(current)
            current = []
    if len(current) >= 4:
        runs.append(current)

    recovered: list[dict[str, Any]] = []
    for run in runs:
        column_counts: Counter[int] = Counter(
            entry["column"] for entries in run for entry in entries
        )
        minimum_presence = max(3, math.ceil(len(run) * 0.7))
        stable_columns = {
            column for column, count in column_counts.items() if count >= minimum_presence
        }
        if len(stable_columns) < 2:
            continue
        stable_rows = [
            [entry for entry in entries if entry["column"] in stable_columns]
            for entries in run
        ]
        stable_rows = [entries for entries in stable_rows if len(entries) >= 2]
        if len(stable_rows) < 4:
            continue
        region_bbox = _union_bbox(
            entry["bbox"] for entries in stable_rows for entry in entries
        )
        if region_bbox is None or _area_ratio(region_bbox, width=width, height=height) >= 0.5:
            continue
        structured_rows = [
            [entry["text"] for entry in sorted(entries, key=lambda item: item["column"])]
            for entries in stable_rows
        ]
        recovered.append(_relation_candidate(
            page,
            bbox=region_bbox,
            rows=structured_rows,
            width=width,
            height=height,
            table_index=table_index,
            extraction_method="full_page_sparse_region_recovery",
            include_candidate_text=include_candidate_text,
        ))
    return recovered


def _candidate_bbox(candidate: dict[str, Any], *, width: float, height: float) -> tuple[float, float, float, float]:
    values = candidate["locator"]["bbox_normalized"]
    return values[0] * width, values[1] * height, values[2] * width, values[3] * height


def _bbox_intersection_over_smaller(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    smaller = min(first_area, second_area)
    return intersection / smaller if smaller else 0.0


def _table_observations(
    page: Any,
    *,
    width: float,
    height: float,
    include_candidate_text: bool,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    suspect_full_page = 0
    detector_error: str | None = None
    try:
        tables = list(getattr(page.find_tables(), "tables", []) or [])
    except (AttributeError, TypeError, ValueError) as error:
        tables = []
        detector_error = type(error).__name__
    full_page_tables: list[tuple[int, Any, list[list[Any]]]] = []
    for index, table in enumerate(tables):
        bbox = _bbox(getattr(table, "bbox", None))
        if bbox is None:
            continue
        rows = table.extract() or []
        row_count = len(rows)
        column_count = max((len(row) for row in rows), default=0)
        non_empty_cells = sum(
            1
            for row in rows
            for cell in row
            if str(cell or "").strip()
        )
        cell_count = max(1, row_count * column_count)
        area_ratio = _area_ratio(bbox, width=width, height=height)
        if area_ratio >= 0.8:
            suspect_full_page += 1
            full_page_tables.append((index, table, rows))
        relation_candidate = bool(
            area_ratio < 0.7
            and row_count >= 2
            and column_count >= 2
            and non_empty_cells >= 4
            and non_empty_cells / cell_count >= 0.25
        )
        if relation_candidate:
            candidates.append(_relation_candidate(
                page,
                bbox=bbox,
                rows=[[str(cell or "") for cell in row] for row in rows],
                width=width,
                height=height,
                table_index=index,
                extraction_method="pymupdf_bounded_table",
                include_candidate_text=include_candidate_text,
            ))
    recovered: list[dict[str, Any]] = []
    for index, table, rows in full_page_tables:
        recovered.extend(_recover_sparse_relation_regions(
            page,
            table,
            rows,
            width=width,
            height=height,
            table_index=index,
            include_candidate_text=include_candidate_text,
        ))
    for candidate in recovered:
        bbox = _candidate_bbox(candidate, width=width, height=height)
        if any(
            _bbox_intersection_over_smaller(
                bbox,
                _candidate_bbox(existing, width=width, height=height),
            ) >= 0.8
            for existing in candidates
        ):
            continue
        candidates.append(candidate)
    return {
        "detected_table_count": len(tables),
        "relation_table_candidate_count": len(candidates),
        "relation_table_candidates": candidates,
        "suspect_full_page_table_count": suspect_full_page,
        "detector_error": detector_error,
    }


def _sequence_observation(text: str) -> dict[str, Any]:
    numbered = len(re.findall(r"(?m)^\s*\d+[.)]\s+", text))
    bullets = len(re.findall(r"(?m)^\s*[•●▪✓✔☑]\s*", text))
    acronym_steps = len(re.findall(r"(?:^|[;\n])\s*[A-Z]\s*[–—-]\s*\w", text))
    arrow_chains = sum(
        1
        for line in text.splitlines()
        if line.count("→") >= 2 or line.count("->") >= 2
    )
    timed_stage_labels = len(re.findall(
        r"\b\d{1,3}\s*(?:ngày|day|days|phút|minute|minutes)\s*:",
        text,
        flags=re.IGNORECASE,
    ))
    candidate_count = sum((
        numbered >= 3,
        bullets >= 3,
        acronym_steps >= 3,
        arrow_chains >= 1,
        timed_stage_labels >= 2,
    ))
    return {
        "numbered_step_count": numbered,
        "bullet_step_count": bullets,
        "acronym_step_count": acronym_steps,
        "arrow_chain_count": arrow_chains,
        "timed_stage_label_count": timed_stage_labels,
        "ordered_structure_candidate_count": candidate_count,
        "ordered_sequence_candidate": candidate_count > 0,
    }


def _visual_reference_detected(text: str) -> bool:
    return bool(re.search(
        r"\b(?:ảnh|hình|image|figure)\s+(?:sau|bên\s+dưới|dưới\s+đây|below|following)\b",
        text,
        flags=re.IGNORECASE,
    ))


def _page_readiness(
    page: Any,
    *,
    image_occurrences: Counter[str],
    page_count: int,
    include_candidate_text: bool,
) -> dict[str, Any]:
    width = float(page.rect.width)
    height = float(page.rect.height)
    blocks = _page_blocks(page)
    text_blocks: list[tuple[tuple[float, float, float, float], str]] = []
    for block in blocks:
        bbox = _bbox(block.get("bbox"))
        text = _block_text(block)
        if bbox is not None and text:
            text_blocks.append((bbox, text))
    text = "\n".join(value for _, value in text_blocks).strip()
    repeated_threshold = max(3, math.ceil(page_count * 0.2))
    image_regions: list[dict[str, Any]] = []
    image_count = 0
    for block in blocks:
        if block.get("type") != 1:
            continue
        bbox = _bbox(block.get("bbox"))
        revision = _image_revision(block)
        if bbox is None or revision is None:
            continue
        image_count += 1
        area_ratio = _area_ratio(bbox, width=width, height=height)
        repeated = image_occurrences[revision] >= repeated_threshold
        pixel_width = int(block.get("width") or 0)
        pixel_height = int(block.get("height") or 0)
        if repeated or area_ratio < 0.08 or pixel_width < 100 or pixel_height < 100:
            continue
        image_regions.append({
            "contract_version": VISUAL_REGION_VERSION,
            "region_kind": "embedded_image",
            "asset_revision": revision,
            "locator": {
                "page": page.number + 1,
                "bbox_normalized": _normalized_bbox(bbox, width=width, height=height),
            },
            "pixel_width": pixel_width,
            "pixel_height": pixel_height,
            "page_area_ratio": round(area_ratio, 6),
            "observation": {"status": "unreviewed", "facts": []},
            "inference": {"status": "not_performed", "claims": []},
        })

    table_observations = _table_observations(
        page,
        width=width,
        height=height,
        include_candidate_text=include_candidate_text,
    )
    drawings = list(page.get_drawings())
    drawing_count = len(drawings)
    multi_column = _detect_multi_column(text_blocks, width=width, height=height)
    corruption_signals = _text_corruption_signals(text)
    sequence_observation = _sequence_observation(text)
    visual_reference_detected = _visual_reference_detected(text)
    vector_visual_candidate = bool(
        drawing_count >= 16
        and table_observations["relation_table_candidate_count"] == 0
        and not image_regions
    )
    vector_region = _vector_visual_region(
        page,
        drawings,
        width=width,
        height=height,
    ) if vector_visual_candidate else None
    visual_regions = [*image_regions]
    if vector_region is not None:
        visual_regions.append(vector_region)

    flags: list[str] = []
    if image_regions:
        flags.append("CONTENT_IMAGE_REQUIRES_OBSERVATION")
    if multi_column:
        flags.append("MULTI_COLUMN_READING_ORDER")
    if table_observations["relation_table_candidate_count"]:
        flags.append("RELATION_TABLE_REQUIRES_VERIFICATION")
    if table_observations["suspect_full_page_table_count"]:
        flags.append("SUSPECT_FULL_PAGE_TABLE_DETECTION")
    if vector_visual_candidate:
        flags.append("POSSIBLE_RELATIONAL_VECTOR_VISUAL")
    flags.extend(corruption_signals)

    visible_text_chars = len(re.sub(r"\s+", "", text))
    broken_or_empty = bool(visible_text_chars < 20 and not image_regions and drawing_count >= 4)
    if broken_or_empty:
        flags.append("BROKEN_OR_EMPTY_PAGE")
        route = "review_required"
    elif (
        image_regions
        or multi_column
        or table_observations["relation_table_candidate_count"]
        or vector_visual_candidate
        or corruption_signals
    ):
        route = "enrichment_required"
    else:
        route = "fast_parse"

    return {
        "page": page.number + 1,
        "route": route,
        "flags": sorted(set(flags)),
        "runtime_observables": {
            "text_char_count": len(text),
            "text_block_count": len(text_blocks),
            "image_block_count": image_count,
            "content_image_region_count": len(image_regions),
            "visual_region_count": len(visual_regions),
            "drawing_count": drawing_count,
            "multi_column_detected": multi_column,
            "vector_visual_candidate": vector_visual_candidate,
            "visual_reference_detected": visual_reference_detected,
            **sequence_observation,
            **table_observations,
        },
        "evidence_regions": image_regions,
        "visual_regions": visual_regions,
        "integrity_issues": [],
        "unknown_signals": [
            "critical_element_ground_truth",
            "semantic_visual_observations",
            "source_author_intent",
        ],
    }


def _link_source_integrity(pages: list[dict[str, Any]]) -> None:
    for index in range(len(pages) - 1):
        current = pages[index]
        following = pages[index + 1]
        if not current["runtime_observables"].get("visual_reference_detected"):
            continue
        if "BROKEN_OR_EMPTY_PAGE" not in following.get("flags", []):
            continue
        issue = {
            "code": "REFERENCED_VISUAL_FOLLOWUP_PAGE_BROKEN",
            "source_page": current["page"],
            "related_page": following["page"],
            "status": "review_required",
            "semantic_claim": "not_performed",
        }
        current["integrity_issues"].append(issue)
        current["flags"] = sorted(set([
            *current["flags"],
            "REFERENCED_VISUAL_FOLLOWUP_PAGE_BROKEN",
        ]))
        following["integrity_issues"].append(issue)
        following["flags"] = sorted(set([
            *following["flags"],
            "REFERENCED_BY_VISUAL_PROMPT",
        ]))


def analyze_pdf_readiness(
    path: Path | str,
    *,
    include_candidate_text: bool = False,
) -> dict[str, Any]:
    """Inspect a PDF without OCR, provider calls, model downloads, or mutation.

    This is a CP3A prototype. It emits runtime observables and routing evidence;
    it never imports benchmark labels and does not claim semantic completeness.
    """
    import pymupdf

    source_path = Path(path)
    with pymupdf.open(str(source_path)) as document:
        image_occurrences = _document_image_occurrences(document)
        pages = [
            _page_readiness(
                page,
                image_occurrences=image_occurrences,
                page_count=document.page_count,
                include_candidate_text=include_candidate_text,
            )
            for page in document
        ]
        page_count = document.page_count
    _link_source_integrity(pages)
    route_counts = Counter(page["route"] for page in pages)
    return {
        "readiness_version": SOURCE_READINESS_VERSION,
        "prototype_only": True,
        "source": {
            "file_name": source_path.name,
            "sha256": _file_sha256(source_path),
            "byte_count": source_path.stat().st_size,
            "page_count": page_count,
        },
        "route_counts": dict(sorted(route_counts.items())),
        "pages": pages,
        "network_used": False,
        "ocr_used": False,
        "provider_used": False,
    }


def collect_pdf_visual_regions(document: Any) -> dict[int, list[dict[str, Any]]]:
    """Return bounded visual locator metadata for the ingestion pipeline.

    The result contains no inferred meaning and no image bytes. It reuses the
    exact CP3A detector so indexing and the offline readiness benchmark cannot
    silently disagree about candidate visual regions.
    """

    image_occurrences = _document_image_occurrences(document)
    result: dict[int, list[dict[str, Any]]] = {}
    for page in document:
        readiness = _page_readiness(
            page,
            image_occurrences=image_occurrences,
            page_count=document.page_count,
            include_candidate_text=False,
        )
        regions = readiness.get("visual_regions")
        if isinstance(regions, list) and regions:
            result[page.number + 1] = [
                dict(region)
                for region in regions[:8]
                if isinstance(region, dict)
            ]
    return result


def build_visual_observation_record(
    *,
    source_revision: str,
    asset_revision: str,
    page: int,
    bbox_normalized: Iterable[float],
    observed_facts: Iterable[str],
    observer_role: str,
    inference_claims: Iterable[str] = (),
) -> dict[str, Any]:
    """Build a bounded human observation record without treating inference as fact."""

    request = build_manual_region_request(
        source_revision=source_revision,
        page=page,
        bbox_normalized=bbox_normalized,
        reason="Record bounded visual evidence for source review.",
        requested_by_role=observer_role,
    )
    revision = asset_revision.strip().casefold()
    if not re.fullmatch(r"[0-9a-f]{64}", revision):
        raise ValueError("asset_revision must be a SHA-256 hex digest")

    def bounded(values: Iterable[str], *, field: str, maximum: int) -> list[str]:
        result: list[str] = []
        for value in values:
            normalized = re.sub(r"\s+", " ", str(value)).strip()
            if not normalized or len(normalized) > maximum:
                raise ValueError(f"{field} values must contain 1..{maximum} characters")
            result.append(normalized)
        if len(result) > 20:
            raise ValueError(f"{field} accepts at most 20 values")
        return result

    facts = bounded(observed_facts, field="observed_facts", maximum=500)
    if not facts:
        raise ValueError("observed_facts requires at least one fact")
    claims = bounded(inference_claims, field="inference_claims", maximum=500)
    return {
        "contract_version": VISUAL_OBSERVATION_VERSION,
        "source_revision": request["source_revision"],
        "asset_revision": revision,
        "locator": request["locator"],
        "observer_role": observer_role,
        "observation": {"status": "observed", "facts": facts},
        "inference": {
            "status": "requires_verification" if claims else "not_performed",
            "claims": claims,
        },
    }


def build_manual_region_request(
    *,
    source_revision: str,
    page: int,
    bbox_normalized: Iterable[float],
    reason: str,
    requested_by_role: str,
) -> dict[str, Any]:
    """Create the immutable input contract for author/reviewer re-reading.

    Persistence, authorization, and enrichment execution intentionally remain
    outside CP3A. The contract prevents a free-form request from losing its
    source revision or exact region when CP3B wires the durable workflow.
    """
    revision = source_revision.strip()
    if not revision:
        raise ValueError("source_revision is required")
    if page < 1:
        raise ValueError("page must be positive")
    values = [float(value) for value in bbox_normalized]
    if len(values) != 4:
        raise ValueError("bbox_normalized must have four values")
    x0, y0, x1, y1 = values
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("bbox_normalized must be ordered inside 0..1")
    normalized_reason = re.sub(r"\s+", " ", reason).strip()
    if not normalized_reason or len(normalized_reason) > 500:
        raise ValueError("reason must contain 1..500 characters")
    if requested_by_role not in {"author", "reviewer"}:
        raise ValueError("requested_by_role must be author or reviewer")
    return {
        "contract_version": MANUAL_REGION_REQUEST_VERSION,
        "source_revision": revision,
        "locator": {
            "page": page,
            "bbox_normalized": [round(value, 6) for value in values],
        },
        "reason": normalized_reason,
        "requested_by_role": requested_by_role,
        "status": "requested",
    }
