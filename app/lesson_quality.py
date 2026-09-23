from __future__ import annotations

"""Deterministic, bounded quality checks for generated lesson proposals.

These checks are generation-quality gates inside the request-scoped lesson
workflow. They never authorize a component, tenant, asset, or database write;
Node remains the authoritative proposal/Apply security boundary.
"""

from dataclasses import dataclass
from html import unescape
import re
import unicodedata
from typing import Any

from app.workflows.contracts import WorkflowIssue, WorkflowValidationResult


EXPLANATORY_TYPES = {"html", "la_faq"}
PRACTICE_OR_CHECK_TYPES = {"problem", "la_sortable", "la_crossword"}
ACTION_OBJECTIVE = re.compile(
    r"\b(?:apply|analyse|analyze|evaluate|perform|demonstrate|use|"
    r"áp\s+dụng|phan\s+tich|phân\s+tích|đánh\s+giá|thực\s+hiện|vận\s+dụng)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LessonQualityReport:
    status: str
    findings: list[WorkflowIssue]
    metrics: dict[str, Any]


def _record(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text_list(value: Any, max_items: int = 160, max_length: int = 2_000) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        text = item.strip()[:max_length]
        if text and text not in seen:
            seen.add(text)
            result.append(text)
        if len(result) >= max_items:
            break
    return result


def _parse_embedded(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            import json
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return {}
        return _record(parsed)
    return _record(value)


def _plain_text(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]*>", " ", unescape(str(value or "")))).strip()


def _normalized_text(value: Any) -> str:
    folded = unicodedata.normalize("NFD", _plain_text(value))
    folded = "".join(char for char in folded if unicodedata.category(char) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", folded.casefold())).strip()


def _tokens(value: Any) -> list[str]:
    return [token for token in _normalized_text(value).split(" ") if len(token) >= 2][:600]


def _jaccard(left: list[str], right: list[str]) -> float:
    left_set, right_set = set(left), set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _component_type(component: dict[str, Any]) -> str:
    return str(component.get("type") or component.get("block_type") or "").strip().casefold()


def _component_fact_ids(component: dict[str, Any]) -> list[str]:
    metadata = _record(component.get("metadata"))
    return _text_list(
        component.get("source_fact_ids")
        or metadata.get("source_fact_ids")
        or _record(component.get("data")).get("source_fact_ids"),
        max_length=96,
    )


def _component_supporting_evidence_fact_ids(component: dict[str, Any]) -> list[str]:
    metadata = _record(component.get("metadata"))
    return _text_list(
        component.get("supporting_evidence_fact_ids")
        or metadata.get("supporting_evidence_fact_ids")
        or _record(component.get("data")).get("supporting_evidence_fact_ids"),
        max_length=96,
    )


def _component_evidence_fact_ids(component: dict[str, Any]) -> list[str]:
    """Return declared canonical and read-only evidence without merging ownership."""
    return list(dict.fromkeys([
        *_component_fact_ids(component),
        *_component_supporting_evidence_fact_ids(component),
    ]))


def _nested_component_data(component: dict[str, Any], key: str) -> dict[str, Any]:
    metadata = _record(component.get("metadata"))
    data = _record(component.get("data"))
    return _parse_embedded(metadata.get(key) or data.get(key) or component.get(key))


def _problem_question(component: dict[str, Any]) -> str:
    direct = component.get("question") or component.get("prompt") or component.get("label")
    if isinstance(direct, str) and direct.strip():
        return _plain_text(direct)
    xml = str(component.get("data") or component.get("xml") or "")
    match = re.search(r"<label>([\s\S]*?)</label>", xml, flags=re.IGNORECASE)
    return _plain_text(match.group(1) if match else "")


def _problem_choices(component: dict[str, Any]) -> list[str]:
    direct = component.get("choices") or component.get("options")
    if isinstance(direct, list):
        values: list[str] = []
        for choice in direct:
            row = _record(choice)
            values.append(_plain_text(row.get("text") or choice))
        return [value for value in values if value]
    xml = str(component.get("data") or component.get("xml") or "")
    return [
        _plain_text(match.group(1))
        for match in re.finditer(r"<(?:choice|option)\b[^>]*>([\s\S]*?)</(?:choice|option)>", xml, flags=re.IGNORECASE)
        if _plain_text(match.group(1))
    ]


def _semantic_learning_text(value: Any) -> str:
    """Extract only text Node's deterministic semantic renderer can display."""
    content = _record(value)
    values: list[str] = []
    heading = content.get("heading")
    if isinstance(heading, str) and heading.strip():
        values.append(heading.strip())
    for field, alias in [
        ("paragraphs", None),
        ("bullet_points", "bullets"),
        ("ordered_steps", "steps"),
        ("warnings", "warning"),
    ]:
        raw_items = content.get(field, content.get(alias) if alias else None)
        if isinstance(raw_items, list):
            values.extend(item.strip() for item in raw_items if isinstance(item, str) and item.strip())
    rows = content.get("comparison_rows", content.get("table_rows"))
    if isinstance(rows, list):
        for row_value in rows:
            row = _record(row_value)
            values.extend(
                text.strip()
                for text in [row.get("label"), row.get("value")]
                if isinstance(text, str) and text.strip()
            )
    return " ".join(values)


def _component_text(component: dict[str, Any]) -> str:
    component_type = _component_type(component)
    if component_type == "html":
        semantic_text = _semantic_learning_text(component.get("semantic_content"))
        return semantic_text or _plain_text(component.get("html") or component.get("data") or component.get("content"))
    if component_type == "problem":
        return _problem_question(component)
    if component_type == "la_faq":
        items = _nested_component_data(component, "faq_data").get("items")
        if not isinstance(items, list):
            items = component.get("items")
        return " ".join(
            f"{row.get('question', '')} {row.get('answer', '')}"
            for item in items or []
            for row in [_record(item)]
        )
    if component_type == "la_crossword":
        words = _nested_component_data(component, "crossword_data").get("words")
        if not isinstance(words, list):
            words = component.get("words")
        return " ".join(
            f"{row.get('answer', '')} {row.get('clue', '')}"
            for item in words or []
            for row in [_record(item)]
        )
    if component_type == "la_sortable":
        sortable = _nested_component_data(component, "sortable_data")
        items = sortable.get("items") if isinstance(sortable.get("items"), list) else component.get("items")
        return " ".join(str(_record(item).get("text") or item or "") for item in items or [])
    return ""


def _html_artifact_item_count(component: dict[str, Any], artifact_type: str) -> int:
    """Count only presentation structures that Node can preserve deterministically.

    This deliberately does not infer whether arbitrary prose implies a
    warning, requirement, or comparison.  The quality gate can fail a missing
    requested structure, but never claims that a structural match proves the
    source statement itself is entailed.
    """
    semantic = _record(component.get("semantic_content"))
    html_value = str(component.get("html") or component.get("data") or component.get("content") or "")
    if artifact_type == "ordered_list":
        semantic_items = semantic.get("ordered_steps", semantic.get("steps"))
        if isinstance(semantic_items, list):
            return len([item for item in semantic_items if isinstance(item, str) and item.strip()])
        return len(re.findall(r"<li\b[^>]*>", re.sub(r"(?is)^.*?<ol\b[^>]*>", "", html_value))) if re.search(r"<ol\b", html_value, re.IGNORECASE) else 0
    if artifact_type == "checklist":
        semantic_items = semantic.get("bullet_points", semantic.get("bullets"))
        if isinstance(semantic_items, list):
            return len([item for item in semantic_items if isinstance(item, str) and item.strip()])
        return len(re.findall(r"<li\b[^>]*>", re.sub(r"(?is)^.*?<ul\b[^>]*>", "", html_value))) if re.search(r"<ul\b", html_value, re.IGNORECASE) else 0
    if artifact_type in {"table", "comparison"}:
        rows = semantic.get("comparison_rows", semantic.get("table_rows"))
        if isinstance(rows, list):
            return len([row for row in rows if isinstance(row, dict)])
        return len(re.findall(r"<tr\b[^>]*>", html_value, re.IGNORECASE)) - 1 if re.search(r"<table\b", html_value, re.IGNORECASE) else 0
    if artifact_type in {"warning", "requirement", "exception"}:
        warnings = semantic.get("warnings", semantic.get("warning"))
        if isinstance(warnings, list):
            return len([item for item in warnings if isinstance(item, str) and item.strip()])
        return len(re.findall(r"<blockquote\b[^>]*>", html_value, re.IGNORECASE))
    return 0


def _issue(
    code: str,
    message: str,
    *,
    path: str,
    severity: str = "error",
    related_paths: list[str] | None = None,
    objective_ids: list[str] | None = None,
    learning_block_ids: list[str] | None = None,
    source_fact_ids: list[str] | None = None,
) -> WorkflowIssue:
    issue: WorkflowIssue = {
        "code": code,
        "severity": severity,  # type: ignore[typeddict-item]
        "message": message,
        "path": path,
        "repairable": True,
    }
    if related_paths:
        issue["related_paths"] = related_paths
    if objective_ids:
        issue["objective_ids"] = objective_ids
    if learning_block_ids:
        issue["learning_block_ids"] = learning_block_ids
    if source_fact_ids:
        issue["source_fact_ids"] = source_fact_ids
    return issue


def _proposal_units(proposal: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for chapter_index, chapter_value in enumerate(proposal.get("chapters") or [], start=1):
        chapter = _record(chapter_value)
        for lesson_index, lesson_value in enumerate(chapter.get("lessons") or [], start=1):
            lesson = _record(lesson_value)
            for unit_index, unit_value in enumerate(lesson.get("units") or [], start=1):
                unit = _record(unit_value)
                result.append({
                    "chapter_index": chapter_index,
                    "lesson_index": lesson_index,
                    "unit_index": unit_index,
                    "path": f"chapter_{chapter_index}.lesson_{lesson_index}.unit_{unit_index}",
                    "unit": unit,
                })
    return result


def _proposal_components(proposal: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for location in _proposal_units(proposal):
        for component_index, component_value in enumerate(location["unit"].get("components") or location["unit"].get("blocks") or [], start=1):
            component = _record(component_value)
            result.append({
                **location,
                "component_index": component_index,
                "component_path": f"{location['path']}.component_{component_index}",
                "component": component,
                "type": _component_type(component),
                "fact_ids": _component_fact_ids(component),
                "supporting_evidence_fact_ids": _component_supporting_evidence_fact_ids(component),
                "evidence_fact_ids": _component_evidence_fact_ids(component),
                "text": _component_text(component),
            })
    return result


def detect_generated_content_duplicates(proposal: dict[str, Any]) -> list[WorkflowIssue]:
    """Find exact/high-similarity prose or quiz duplication in bounded scopes."""
    components = _proposal_components(proposal)
    findings: list[WorkflowIssue] = []
    comparisons = 0
    for left_index, left in enumerate(components):
        for right in components[left_index + 1:]:
            if comparisons >= 96:
                return findings
            if left["chapter_index"] != right["chapter_index"]:
                continue
            comparisons += 1
            if left["type"] == right["type"] == "problem":
                left_question = _normalized_text(_problem_question(left["component"]))
                right_question = _normalized_text(_problem_question(right["component"]))
                if left_question and left_question == right_question:
                    findings.append(_issue(
                        "DUPLICATE_QUIZ_QUESTION",
                        "A sibling unit repeats the same knowledge-check question.",
                        path=right["component_path"],
                        related_paths=[left["component_path"]],
                        source_fact_ids=right["fact_ids"],
                    ))
                continue
            if left["type"] not in EXPLANATORY_TYPES or right["type"] not in EXPLANATORY_TYPES:
                continue
            # Explanation followed by a practice/check is intentionally not in
            # this comparison set, so teach -> practice -> check remains valid.
            left_tokens, right_tokens = _tokens(left["text"]), _tokens(right["text"])
            if len(left_tokens) < 10 or len(right_tokens) < 10:
                continue
            fact_overlap = bool(set(left["fact_ids"]) & set(right["fact_ids"]))
            similarity = _jaccard(left_tokens, right_tokens)
            if (fact_overlap and similarity >= 0.86) or _normalized_text(left["text"]) == _normalized_text(right["text"]):
                findings.append(_issue(
                    "DUPLICATE_EXPLANATION",
                    "A sibling unit repeats substantially the same explanatory content.",
                    path=right["component_path"],
                    related_paths=[left["component_path"]],
                    source_fact_ids=right["fact_ids"],
                ))
    return findings


def _payload_quality_findings(components: list[dict[str, Any]]) -> list[WorkflowIssue]:
    findings: list[WorkflowIssue] = []
    for item in components:
        component, component_type = item["component"], item["type"]
        if component_type == "problem":
            question = _problem_question(component)
            choices = [_normalized_text(choice) for choice in _problem_choices(component) if _normalized_text(choice)]
            if not question:
                findings.append(_issue("ASSESSMENT_QUESTION_EMPTY", "Knowledge check has no learner-facing question.", path=item["component_path"]))
            if len(set(choices)) != len(choices):
                findings.append(_issue("ASSESSMENT_DISTRACTORS_DUPLICATED", "Knowledge check repeats an answer option.", path=item["component_path"]))
        if component_type == "la_faq":
            items = _nested_component_data(component, "faq_data").get("items")
            if not isinstance(items, list):
                items = component.get("items")
            questions = [_normalized_text(_record(item).get("question")) for item in items or []]
            questions = [question for question in questions if question]
            if len(set(questions)) != len(questions):
                findings.append(_issue("FAQ_QUESTION_DUPLICATED", "FAQ repeats a learner question.", path=item["component_path"]))
        if component_type == "la_crossword":
            words = _nested_component_data(component, "crossword_data").get("words")
            if not isinstance(words, list):
                words = component.get("words")
            terms = [_normalized_text(_record(word).get("answer")) for word in words or []]
            terms = [term for term in terms if term]
            if len(set(terms)) != len(terms):
                findings.append(_issue("CROSSWORD_TERM_DUPLICATED", "Crossword repeats a terminology answer.", path=item["component_path"]))
    return findings


def validate_lesson_pedagogical_quality(
    proposal: dict[str, Any],
    blueprint_architecture: dict[str, Any] | None = None,
    *,
    include_duplicates: bool = True,
) -> LessonQualityReport:
    """Evaluate final generated content against the approved lesson contract."""
    findings = _payload_quality_findings(_proposal_components(proposal))
    components = _proposal_components(proposal)
    objective_total = objective_covered = 0
    source_total = source_covered = 0
    assessment_total = assessment_covered = 0
    depth_total = depth_covered = 0
    purpose_total = purpose_covered = 0
    artifact_total = artifact_covered = 0

    expected_chapter = _record(blueprint_architecture)
    expected_lessons = expected_chapter.get("lessons") if isinstance(expected_chapter.get("lessons"), list) else []
    for lesson_index, lesson_value in enumerate(expected_lessons, start=1):
        expected_lesson = _record(lesson_value)
        lesson_path = f"chapter_1.lesson_{lesson_index}"
        lesson_components = [item for item in components if item["chapter_index"] == 1 and item["lesson_index"] == lesson_index]
        objectives = _text_list(expected_lesson.get("learning_objectives"), max_items=12, max_length=300)
        units = expected_lesson.get("units") if isinstance(expected_lesson.get("units"), list) else []
        for objective_index, _objective in enumerate(objectives, start=1):
            objective_total += 1
            objective_ref = f"lo_{objective_index}"
            mapped_units = [
                index for index, unit_value in enumerate(units, start=1)
                if objective_ref in _text_list(_record(unit_value).get("learning_objective_refs"), max_items=12, max_length=16)
            ]
            teaching_block_facts: set[str] = set()
            teaching_block_ids: set[str] = set()
            for unit_index in mapped_units:
                for block_value in _record(units[unit_index - 1]).get("learning_blocks") or []:
                    block = _record(block_value)
                    if objective_ref not in _text_list(block.get("learning_objective_refs"), max_items=12, max_length=16):
                        continue
                    if str(block.get("intent") or "").strip() == "knowledge_check":
                        continue
                    block_id = str(block.get("id") or "").strip()
                    if block_id:
                        teaching_block_ids.add(block_id)
                    teaching_block_facts.update(_text_list(block.get("source_fact_ids"), max_length=96))
            expected_html_plan_blocks = {
                block_id
                for unit_index in mapped_units
                for plan_value in (_record(units[unit_index - 1]).get("component_plan") or [])
                for plan in [_record(plan_value)]
                if str(plan.get("type") or "").strip() == "html"
                for block_id in _text_list(plan.get("learning_block_ids"), max_items=12, max_length=80)
            }
            taught = any(
                item["unit_index"] in mapped_units
                and item["type"] == "html"
                # A plan may omit block IDs on an old Blueprint. For a new V5
                # plan, an explicit mapping must point at a teaching block.
                and (not expected_html_plan_blocks or bool(expected_html_plan_blocks & teaching_block_ids))
                and len(_tokens(item["text"])) >= 18
                and (not teaching_block_facts or bool(set(item["evidence_fact_ids"]) & teaching_block_facts))
                for item in lesson_components
            )
            if taught:
                objective_covered += 1
            else:
                block_ids = [
                    str(_record(block).get("id") or "")
                    for unit_index in mapped_units
                    for block in (_record(units[unit_index - 1]).get("learning_blocks") or [])
                    if objective_ref in _text_list(_record(block).get("learning_objective_refs"), max_items=12, max_length=16)
                ]
                findings.append(_issue(
                    "OBJECTIVE_NOT_TAUGHT",
                    "An approved learning objective has no substantive explanatory treatment.",
                    path=f"{lesson_path}.unit_{mapped_units[0]}" if len(mapped_units) == 1 else lesson_path,
                    objective_ids=[objective_ref],
                    learning_block_ids=[item for item in block_ids if item],
                ))

        for unit_index, unit_value in enumerate(units, start=1):
            expected_unit = _record(unit_value)
            unit_path = f"{lesson_path}.unit_{unit_index}"
            actual_components = [item for item in lesson_components if item["unit_index"] == unit_index]
            expected_facts = _text_list(expected_unit.get("source_fact_ids"), max_length=96)
            declared_facts = {fact_id for item in actual_components for fact_id in item["fact_ids"]}
            source_total += len(expected_facts)
            for fact_id in expected_facts:
                if fact_id in declared_facts:
                    source_covered += 1
                else:
                    findings.append(_issue(
                        "SOURCE_FACT_NOT_TAUGHT",
                        "A source fact assigned to the approved unit is absent from generated component coverage.",
                        path=unit_path,
                        source_fact_ids=[fact_id],
                    ))

            plans = expected_unit.get("component_plan") if isinstance(expected_unit.get("component_plan"), list) else []
            actual_types = [item["type"] for item in actual_components]
            for plan_value in plans:
                plan = _record(plan_value)
                component_type = str(plan.get("type") or "")
                if not component_type:
                    continue
                purpose_total += 1
                if component_type in actual_types:
                    purpose_covered += 1
                else:
                    findings.append(_issue(
                        "COMPONENT_PURPOSE_INVALID",
                        f"The generated unit omitted the approved {component_type} learning treatment.",
                        path=unit_path,
                        learning_block_ids=_text_list(plan.get("learning_block_ids"), max_items=12, max_length=80),
                    ))
                required_reason = {
                    "la_sortable": "ORDERING_PRACTICE",
                    "la_crossword": "TERMINOLOGY_REINFORCEMENT",
                    "la_diagram": "RELATIONSHIP_VISUALIZATION",
                    "la_faq": "FAQ_ANTICIPATED_QUESTIONS",
                }.get(component_type)
                if required_reason and plan.get("reason_code") not in {None, "", required_reason}:
                    findings.append(_issue(
                        "COMPONENT_PURPOSE_INVALID",
                        f"{component_type} does not have its required pedagogical purpose.",
                        path=unit_path,
                        learning_block_ids=_text_list(plan.get("learning_block_ids"), max_items=12, max_length=80),
                    ))

                for artifact_value in plan.get("required_artifacts") or []:
                    artifact = _record(artifact_value)
                    artifact_type = str(artifact.get("type") or "").strip().casefold()
                    if artifact_type not in {
                        "ordered_list", "checklist", "table", "warning",
                        "requirement", "exception", "comparison",
                    }:
                        continue
                    artifact_total += 1
                    minimum_items = artifact.get("minimum_items")
                    required_count = minimum_items if isinstance(minimum_items, int) and minimum_items > 0 else 1
                    rendered_count = max(
                        (
                            _html_artifact_item_count(item["component"], artifact_type)
                            for item in actual_components
                            if item["type"] == component_type
                        ),
                        default=0,
                    )
                    if rendered_count >= required_count:
                        artifact_covered += 1
                    else:
                        findings.append(_issue(
                            "REQUIRED_ARTIFACT_NOT_PRESERVED",
                            f"The approved {artifact_type} treatment is absent or incomplete in generated content.",
                            path=unit_path,
                            learning_block_ids=_text_list(plan.get("learning_block_ids"), max_items=12, max_length=80),
                        ))

            complexity = len(expected_facts) + len(_text_list(expected_unit.get("concept_ids"), max_items=12, max_length=96)) + len(_text_list(expected_unit.get("learning_objective_refs"), max_items=12, max_length=16))
            if complexity >= 3:
                depth_total += 1
                explanatory_words = sum(len(_tokens(item["text"])) for item in actual_components if item["type"] == "html")
                block_count = len(expected_unit.get("learning_blocks") or [])
                if explanatory_words >= 45 or (block_count <= 1 and explanatory_words >= 28):
                    depth_covered += 1
                else:
                    findings.append(_issue(
                        "INSUFFICIENT_INSTRUCTIONAL_DEPTH",
                        "A complex unit has too little explanatory treatment for its approved concepts/facts/objectives.",
                        path=unit_path,
                        learning_block_ids=[str(_record(block).get("id") or "") for block in expected_unit.get("learning_blocks") or [] if str(_record(block).get("id") or "")],
                        source_fact_ids=expected_facts,
                    ))

        if expected_lesson.get("assessment_required") is True:
            assessment_total += 1
            explained_facts = {
                fact_id for item in lesson_components if item["type"] == "html" for fact_id in item["evidence_fact_ids"]
            }
            assessment_objectives = set(_text_list(expected_lesson.get("assessment_objective_refs"), max_items=12, max_length=80))
            taught_objectives = {
                objective_ref
                for objective_ref in assessment_objectives
                if any(
                    item["type"] == "html"
                    and len(_tokens(item["text"])) >= 18
                    and item["unit_index"] == unit_index
                    for unit_index, unit_value in enumerate(units, start=1)
                    if objective_ref in _text_list(_record(unit_value).get("learning_objective_refs"), max_items=12, max_length=16)
                    for item in lesson_components
                    if item["type"] == "html"
                    and len(_tokens(item["text"])) >= 18
                )
            }
            aligned = bool(assessment_objectives & taught_objectives) and any(
                item["type"] == "problem"
                and bool(set(item["evidence_fact_ids"]) & explained_facts)
                for item in lesson_components
            )
            if aligned:
                assessment_covered += 1
            else:
                findings.append(_issue(
                    "ASSESSMENT_NOT_ALIGNED",
                    "An assessment-required lesson needs a source-linked problem after explanatory teaching.",
                    path=lesson_path,
                    objective_ids=_text_list(expected_lesson.get("assessment_objective_refs"), max_items=12, max_length=80),
                ))
        if any(ACTION_OBJECTIVE.search(objective) for objective in objectives) and lesson_components and all(item["type"] == "html" for item in lesson_components):
            findings.append(_issue(
                "COGNITIVE_TREATMENT_WEAK",
                "An apply/analyze objective currently has explanation only; review whether a supported learner action is needed.",
                path=lesson_path,
                severity="warning",
            ))

    duplicates = detect_generated_content_duplicates(proposal) if include_duplicates else []
    findings.extend(duplicates)
    errors = [issue for issue in findings if issue.get("severity") == "error"]
    warnings = [issue for issue in findings if issue.get("severity") == "warning"]

    def ratio(passed: int, total: int) -> float | None:
        return round(passed / total, 4) if total else None

    return LessonQualityReport(
        status="FAIL" if errors else "PASS_WITH_WARNINGS" if warnings else "PASS",
        findings=findings,
        metrics={
            "objective_coverage": ratio(objective_covered, objective_total),
            "objective_treatment_coverage": ratio(objective_covered, objective_total),
            "source_fact_coverage": ratio(source_covered, source_total),
            "declared_source_coverage": ratio(source_covered, source_total),
            "artifact_coverage": ratio(artifact_covered, artifact_total),
            "assessment_alignment": ratio(assessment_covered, assessment_total),
            "instructional_depth": ratio(depth_covered, depth_total),
            "component_purpose": ratio(purpose_covered, purpose_total),
            "duplicate_count": len(duplicates),
            "pedagogical_warning_count": len(warnings),
            "factual_entailment_not_automatically_verified": True,
            "component_counts": {
                component_type: sum(1 for item in components if item["type"] == component_type)
                for component_type in sorted({item["type"] for item in components if item["type"]})
            },
        },
    )


def pedagogical_validation_result(
    proposal: dict[str, Any],
    blueprint_architecture: dict[str, Any] | None = None,
) -> WorkflowValidationResult:
    # The LangGraph has a separate duplication node so a duplicate finding is
    # emitted once, classified once, and repaired at one component/unit scope.
    report = validate_lesson_pedagogical_quality(proposal, blueprint_architecture, include_duplicates=False)
    return WorkflowValidationResult(report.findings, report.metrics)


def duplicate_validation_result(proposal: dict[str, Any]) -> WorkflowValidationResult:
    findings = detect_generated_content_duplicates(proposal)
    return WorkflowValidationResult(findings, {"duplicate_count": len(findings)})
