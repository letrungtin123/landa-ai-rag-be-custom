from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.services.ingestion.extract import extract_pdf
from app.source_readiness import analyze_pdf_readiness
from app.source_visual_observation import (
    build_shadow_visual_observation_plan,
    record_human_visual_observation,
)


VISUAL_ELEMENT_KINDS = {
    "content_image_asset",
    "image_situation_asset",
    "matrix_image_asset",
    "relational_vector_visual",
}


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _tokens(value: str) -> list[str]:
    return re.findall(r"[\w]+", value.casefold(), flags=re.UNICODE)


def _anchor_matches(anchor: str, candidate_text: str) -> bool:
    normalized_anchor = _normalize(anchor)
    normalized_candidate = _normalize(candidate_text)
    if normalized_anchor in normalized_candidate:
        return True
    anchor_tokens = _tokens(anchor)
    candidate_tokens = _tokens(candidate_text)
    if not anchor_tokens or len(anchor_tokens) > len(candidate_tokens):
        return False
    width = len(anchor_tokens)
    return any(
        candidate_tokens[index:index + width] == anchor_tokens
        for index in range(len(candidate_tokens) - width + 1)
    )


def _real_held_out_document_count(oracle: dict[str, Any]) -> int:
    return sum(
        1
        for document in oracle.get("documents", [])
        if isinstance(document, dict) and document.get("split") == "held_out_document"
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object in {path}")
    return value


def _run_candidate(
    *,
    python: Path,
    source: Path,
    pages: list[int],
) -> dict[str, Any]:
    bridge = Path(__file__).with_name("cp3a_layout_candidate.py")
    completed = subprocess.run(
        [
            str(python),
            str(bridge),
            "--source",
            str(source),
            "--pages",
            ",".join(str(page) for page in pages),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
        env={
            **__import__("os").environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "NO_PROXY": "*",
            "PYTHONIOENCODING": "utf-8",
        },
    )
    if completed.returncode != 0:
        return {
            "available": False,
            "return_code": completed.returncode,
            "failure": completed.stderr[-2_000:] or completed.stdout[-2_000:],
        }
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        return {
            "available": False,
            "return_code": completed.returncode,
            "failure": f"candidate output is not JSON: {error}",
        }
    return {"available": True, **result}


def _baseline_summary(source: Path) -> dict[str, Any]:
    started = time.perf_counter()
    sections = extract_pdf(source)
    elapsed = time.perf_counter() - started
    return {
        "parser": "production-pymupdf-structured-source-v1",
        "elapsed_seconds": round(elapsed, 4),
        "extracted_page_count": len({section.page for section in sections if section.page}),
        "pages": {
            int(section.page): {
                "text": section.text,
                "reported_table_count": int(section.metadata.get("table_count") or 0),
            }
            for section in sections
            if section.page is not None
        },
    }


def _routing_metrics(labels: list[dict[str, Any]], readiness_by_page: dict[int, dict[str, Any]]) -> dict[str, Any]:
    expected_non_fast = [label for label in labels if label["expected_route"] != "fast_parse"]
    expected_fast = [label for label in labels if label["expected_route"] == "fast_parse"]
    detected_non_fast = sum(
        readiness_by_page[label["page"]]["route"] != "fast_parse"
        for label in expected_non_fast
    )
    false_positives = sum(
        readiness_by_page[label["page"]]["route"] != "fast_parse"
        for label in expected_fast
    )
    exact = sum(
        readiness_by_page[label["page"]]["route"] == label["expected_route"]
        for label in labels
    )
    return {
        "sample_size": len(labels),
        "expected_non_fast_count": len(expected_non_fast),
        "expected_fast_count": len(expected_fast),
        "non_fast_recall": _ratio(detected_non_fast, len(expected_non_fast)),
        "false_negative_count": len(expected_non_fast) - detected_non_fast,
        "false_positive_rate": _ratio(false_positives, len(expected_fast)),
        "false_positive_count": false_positives,
        "exact_route_accuracy": _ratio(exact, len(labels)),
    }


def _candidate_table_metrics(
    labels: list[dict[str, Any]],
    candidate_by_page: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    expected_count = 0
    represented_count = 0
    required_anchor_count = 0
    matched_anchor_count = 0
    page_results: list[dict[str, Any]] = []
    for label in labels:
        elements = [element for element in label.get("critical_elements", []) if element.get("kind") == "table"]
        if not elements:
            continue
        page = int(label["page"])
        page_candidate = candidate_by_page.get(page, {})
        detected_tables = page_candidate.get("tables", []) if isinstance(page_candidate, dict) else []
        detected_count = len(detected_tables)
        page_expected = sum(int(element.get("count") or 0) for element in elements)
        expected_count += page_expected
        represented_count += min(page_expected, detected_count)
        table_text = "\n".join(str(table.get("text") or "") for table in detected_tables)
        anchors = [
            str(anchor)
            for element in elements
            for anchor in element.get("required_strings", [])
        ]
        matched = [anchor for anchor in anchors if _anchor_matches(anchor, table_text)]
        required_anchor_count += len(anchors)
        matched_anchor_count += len(matched)
        page_results.append({
            "page": page,
            "expected_table_count": page_expected,
            "detected_table_count": detected_count,
            "required_anchor_count": len(anchors),
            "matched_anchor_count": len(matched),
            "missing_anchors": [anchor for anchor in anchors if anchor not in matched],
        })
    return {
        "expected_table_count": expected_count,
        "represented_table_count": represented_count,
        "table_count_recall": _ratio(represented_count, expected_count),
        "required_anchor_count": required_anchor_count,
        "matched_anchor_count": matched_anchor_count,
        "table_anchor_recall": _ratio(matched_anchor_count, required_anchor_count),
        "pages": page_results,
    }


def _element_metrics(
    labels: list[dict[str, Any]],
    readiness_by_page: dict[int, dict[str, Any]],
    baseline_pages: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    expected: Counter[str] = Counter()
    represented: Counter[str] = Counter()
    for label in labels:
        page = int(label["page"])
        readiness = readiness_by_page[page]
        observables = readiness["runtime_observables"]
        baseline_text = str(baseline_pages.get(page, {}).get("text") or "")
        for element in label.get("critical_elements", []):
            kind = str(element.get("kind") or "")
            count = int(element.get("count") or 0)
            expected[kind] += count
            if kind == "table":
                actual = int(observables.get("relation_table_candidate_count") or 0)
            elif kind in {"content_image_asset", "image_situation_asset", "matrix_image_asset"}:
                actual = int(observables.get("content_image_region_count") or 0)
            elif kind == "multi_column_reading_order":
                actual = int(bool(observables.get("multi_column_detected")))
            elif kind == "relational_vector_visual":
                actual = int(bool(observables.get("vector_visual_candidate")))
            elif kind == "broken_or_empty_page":
                actual = int(readiness.get("route") == "review_required")
            elif kind == "ordered_process":
                actual = int(
                    observables.get("ordered_structure_candidate_count")
                    or bool(observables.get("ordered_sequence_candidate"))
                )
            elif kind == "scenario_group":
                actual = len(set(re.findall(r"Tình huống\s+(\d+)", baseline_text, flags=re.IGNORECASE)))
            elif kind == "source_integrity_issue":
                actual = len(readiness.get("integrity_issues", []))
            else:
                actual = 0
            represented[kind] += min(count, actual)
    return {
        kind: {
            "expected_count": expected[kind],
            "represented_count": represented[kind],
            "representation_recall": _ratio(represented[kind], expected[kind]),
        }
        for kind in sorted(expected)
    }


def _visual_observation_metrics(
    labels: list[dict[str, Any]],
    readiness: dict[str, Any],
) -> dict[str, Any]:
    expected_pages = sorted({
        int(label["page"])
        for label in labels
        if any(
            element.get("kind") in VISUAL_ELEMENT_KINDS
            for element in label.get("critical_elements", [])
        )
    })
    plan = build_shadow_visual_observation_plan(
        readiness,
        tenant_scope="cp3a-benchmark-tenant",
        locale="vi",
        observer_config_version="human-review-v1",
    )
    tasks_by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for task in plan.get("tasks", []):
        tasks_by_page[int(task["locator"]["page"])].append(task)
    task_covered_pages = [page for page in expected_pages if tasks_by_page.get(page)]
    reviewed_pages: list[int] = []
    observation_errors: list[dict[str, Any]] = []
    observed_fact_count = 0
    for label in labels:
        page = int(label["page"])
        if page not in expected_pages:
            continue
        facts = label.get("reviewed_visual_facts")
        if not isinstance(facts, list) or not facts or not tasks_by_page.get(page):
            continue
        try:
            receipt = record_human_visual_observation(
                tasks_by_page[page][0],
                observed_facts=[str(fact) for fact in facts],
                observer_role="reviewer",
            )
        except (TypeError, ValueError) as error:
            observation_errors.append({
                "page": page,
                "error_type": type(error).__name__,
            })
            continue
        reviewed_pages.append(page)
        observed_fact_count += len(receipt["record"]["observation"]["facts"])
    return {
        "policy_version": plan.get("policy_version"),
        "mode": plan.get("mode"),
        "expected_visual_page_count": len(expected_pages),
        "task_covered_page_count": len(task_covered_pages),
        "task_page_recall": _ratio(len(task_covered_pages), len(expected_pages)),
        "reviewed_visual_page_count": len(set(reviewed_pages)),
        "observation_page_recall": _ratio(len(set(reviewed_pages)), len(expected_pages)),
        "observed_fact_count": observed_fact_count,
        "provider_call_count": int(plan.get("provider_call_count") or 0),
        "non_blocking": plan.get("blocking") is False,
        "draft_visibility_preserved": plan.get("draft_visibility") == "preserved",
        "scheduled_region_count": int(plan.get("scheduled_region_count") or 0),
        "missing_task_pages": [page for page in expected_pages if page not in task_covered_pages],
        "missing_observation_pages": [page for page in expected_pages if page not in reviewed_pages],
        "observation_errors": observation_errors,
        "plan_diagnostics": plan.get("diagnostics", []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="CP3A source readiness/parser benchmark")
    parser.add_argument("--source", required=True)
    parser.add_argument(
        "--oracle",
        default=str(Path(__file__).with_name("fixtures") / "ai_id_cp3a_source_oracle.json"),
    )
    parser.add_argument("--candidate-python")
    parser.add_argument("--output")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    oracle = _load_json(Path(args.oracle).resolve())
    document = next((
        item for item in oracle.get("documents", [])
        if item.get("file_name") == source.name
    ), None)
    if document is None:
        raise ValueError(f"No human-reviewed oracle document matches {source.name}")
    readiness = analyze_pdf_readiness(source, include_candidate_text=True)
    if readiness["source"]["sha256"] != document["sha256"]:
        raise ValueError("Source SHA-256 does not match the human-reviewed oracle")
    if readiness["source"]["page_count"] != document["page_count"]:
        raise ValueError("Source page count does not match the human-reviewed oracle")

    labels = list(document.get("pages", []))
    labeled_pages = sorted({int(label["page"]) for label in labels})
    readiness_by_page = {int(page["page"]): page for page in readiness["pages"]}
    baseline = _baseline_summary(source)
    baseline_pages = baseline.pop("pages")

    candidate: dict[str, Any] = {"available": False, "failure": "candidate python not provided"}
    if args.candidate_python:
        candidate = _run_candidate(
            python=Path(args.candidate_python).resolve(),
            source=source,
            pages=labeled_pages,
        )
    candidate_by_page = {
        int(page["page"]): page
        for page in candidate.get("pages", [])
        if isinstance(page, dict) and page.get("page")
    } if candidate.get("available") else {}

    split_metrics: dict[str, Any] = {}
    for split in sorted({str(label.get("split") or "unknown") for label in labels}):
        split_labels = [label for label in labels if str(label.get("split") or "unknown") == split]
        split_metrics[split] = _routing_metrics(split_labels, readiness_by_page)

    routing = _routing_metrics(labels, readiness_by_page)
    elements = _element_metrics(labels, readiness_by_page, baseline_pages)
    visual_observations = _visual_observation_metrics(labels, readiness)
    relation_prototype_by_page = {
        page_number: {
            "tables": [
                {"text": candidate.get("benchmark_text", "")}
                for candidate in page["runtime_observables"].get("relation_table_candidates", [])
            ],
        }
        for page_number, page in readiness_by_page.items()
    }
    relation_prototype_metrics = _candidate_table_metrics(
        labels,
        relation_prototype_by_page,
    )
    table_metrics = _candidate_table_metrics(labels, candidate_by_page)
    blockers: list[str] = []
    held_out_requirement = oracle.get("held_out_document_requirement", {})
    held_out_document_count = _real_held_out_document_count(oracle)
    if held_out_document_count < int(held_out_requirement.get("required_real_document_count") or 0):
        blockers.append("REAL_HELD_OUT_DOCUMENT_MISSING")
    if routing["false_negative_count"]:
        blockers.append("READINESS_FALSE_NEGATIVE_PRESENT")
    if not candidate.get("available"):
        blockers.append("ISOLATED_LAYOUT_CANDIDATE_UNAVAILABLE")
    if relation_prototype_metrics["table_count_recall"] != 1.0:
        blockers.append("TABLE_RELATION_COUNT_INCOMPLETE")
    if relation_prototype_metrics["table_anchor_recall"] != 1.0:
        blockers.append("TABLE_RELATION_ANCHORS_INCOMPLETE")
    if (
        visual_observations["expected_visual_page_count"]
        and visual_observations["task_page_recall"] != 1.0
    ):
        blockers.append("VISUAL_REGION_TASK_COVERAGE_INCOMPLETE")
    if (
        visual_observations["expected_visual_page_count"]
        and visual_observations["observation_page_recall"] != 1.0
    ):
        blockers.append("SEMANTIC_VISUAL_OBSERVATIONS_NOT_IMPLEMENTED")
    if visual_observations["provider_call_count"]:
        blockers.append("UNAUTHORIZED_VISUAL_PROVIDER_CALL")
    if not (
        visual_observations["non_blocking"]
        and visual_observations["draft_visibility_preserved"]
    ):
        blockers.append("VISUAL_SHADOW_DRAFT_VISIBILITY_UNSAFE")
    if (
        "source_integrity_issue" in elements
        and elements["source_integrity_issue"].get("representation_recall") != 1.0
    ):
        blockers.append("SOURCE_INTEGRITY_LINKAGE_NOT_IMPLEMENTED")
    if any(
        metric.get("representation_recall") != 1.0
        for metric in elements.values()
    ):
        blockers.append("CRITICAL_ELEMENT_REPRESENTATION_INCOMPLETE")

    safe_candidate = {key: value for key, value in candidate.items() if key != "pages"}
    result = {
        "benchmark_version": "ai-id-cp3a-source-benchmark-4",
        "fixture_version": oracle.get("fixture_version"),
        "document": {
            "id": document.get("id"),
            "split": document.get("split"),
            "notes": document.get("notes"),
        },
        "source": readiness["source"],
        "mode": "offline_no_ocr_no_provider",
        "corpus_context": {
            "document_count": len(oracle.get("documents", [])),
            "real_held_out_document_count": held_out_document_count,
            "required_real_held_out_document_count": int(
                held_out_requirement.get("required_real_document_count") or 0
            ),
        },
        "readiness": {
            "version": readiness["readiness_version"],
            "route_counts": readiness["route_counts"],
            "routing_metrics": routing,
            "routing_metrics_by_split": split_metrics,
            "labeled_pages": [
                {
                    "page": int(label["page"]),
                    "split": label["split"],
                    "expected_route": label["expected_route"],
                    "observed_route": readiness_by_page[int(label["page"])]["route"],
                    "flags": readiness_by_page[int(label["page"])]["flags"],
                }
                for label in labels
            ],
        },
        "element_representation": elements,
        "visual_observation_shadow": visual_observations,
        "production_baseline": {
            **baseline,
            "labeled_pages_missing_from_extraction": [
                page for page in labeled_pages if page not in baseline_pages
            ],
        },
        "source_relation_prototype": {
            "version": readiness["readiness_version"],
            "table_metrics": relation_prototype_metrics,
        },
        "layout_candidate": {
            **safe_candidate,
            "table_metrics": table_metrics,
        },
        "integration_decision": "GO" if not blockers else "NO_GO",
        "integration_scope": "cp3b_prototype_eligibility_only",
        "integration_blockers": sorted(set(blockers)),
        "limitations": [
            "This result applies to the labeled document and corpus; it is not a universal customer-document specification.",
            "Human-reviewed visual observations are benchmarked; no model/provider visual inference was performed.",
            "No parser output was assembled into a writer payload in CP3A.",
        ],
    }
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).resolve().write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
