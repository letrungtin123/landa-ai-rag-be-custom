"""Bounded, non-blocking observations of generated content, never acceptance.

Lexical witnesses are NOT factual entailment. Paraphrases/translations without a
witness require review rather than being declared missing or false. No model,
retrieval, DB write, ownership mutation or repair dispatch occurs here.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from html import unescape
from typing import Any

from app.lesson_prompt_policy import component_instructional_brief, instructional_action_intents
from app.ordered_learning_content import ordered_content_fields

VERSION = "lesson-content-observation-3"
MAX_FACTS = 1200
MAX_MANIFEST_FACTS = 20000
MAX_COMPONENTS = 32
MAX_SEGMENTS = 512
MAX_TEXT_CHARS = 4000
MAX_CANDIDATES = 48
MAX_FINDINGS = 120
MAX_WITNESSES = 120
_NEGATION = re.compile(r"\b(?:not|never|no|without|khong|cam|chua)\b")
_NUMBERS = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?(?!\w)")
_INTERNAL_ID = re.compile(r"\bp\d+-f\d+\b|\bcp2_[a-f0-9]+\b")
_MULTIPLICATION = re.compile(r"(?<![\w.,-])(\d{1,6}(?:\.\d{1,6})?)\s*[x×*]\s*(\d{1,6}(?:\.\d{1,6})?)\s*=\s*(\d{1,12}(?:\.\d{1,12})?)(?![\w,]|\.\d)")
_INSTRUCTIONAL_TYPES = {"html", "la_diagram", "la_sortable", "la_crossword"}


def _record(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _ids(value: Any) -> list[str]:
    return list(dict.fromkeys(v for v in _list(value) if isinstance(v, str) and v))


def _normalize(text: str) -> str:
    text = unescape(re.sub(r"<[^>]*>", " ", text)).casefold().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    return " ".join(re.findall(r"\w+", "".join(c for c in text if unicodedata.category(c) != "Mn")))


@dataclass(frozen=True)
class TextSpan:
    component_index: int
    path: str
    raw: str
    normalized: str
    tokens: frozenset[str]


def _learner_fields(component: dict) -> list[tuple[str, str]]:
    """Only explicit display fields. Never stringify metadata/ownership arrays."""
    result = []
    for field in ("title", "question", "question_text", "explanation", "answer", "html", "content", "name"):
        if isinstance(component.get(field), str):
            result.append((field, component[field]))
    semantic = _record(component.get("semantic_content"))
    result.extend((f"semantic_content.{path}", text) for path, text in ordered_content_fields(semantic))
    if isinstance(semantic.get("heading"), str):
        result.append(("semantic_content.heading", semantic["heading"]))
    for field in ("paragraphs", "bullet_points", "ordered_steps", "warnings"):
        for i, item in enumerate(_list(semantic.get(field))):
            if isinstance(item, str):
                result.append((f"semantic_content.{field}[{i}]", item))
    for i, item in enumerate(_list(semantic.get("comparison_rows"))):
        row = _record(item)
        result.append((f"semantic_content.comparison_rows[{i}]",
                       " ".join(row[k] for k in ("label", "value") if isinstance(row.get(k), str))))
    for field in ("ordered_items", "steps"):
        for i, item in enumerate(_list(component.get(field))):
            if isinstance(item, str):
                result.append((f"{field}[{i}]", item))
    for field, keys in (("items", ("question", "answer", "text")), ("choices", ("text",)),
                        ("words", ("answer", "clue", "hint")),
                        ("nodes", ("label", "tooltip")), ("edges", ("label",))):
        for i, item in enumerate(_list(component.get(field))):
            if isinstance(item, str):
                result.append((f"{field}[{i}]", item))
            elif isinstance(item, dict):
                result.extend((f"{field}[{i}].{k}", item[k]) for k in keys if isinstance(item.get(k), str))
    return result


def _teaching_fields(component: dict) -> list[tuple[str, str]]:
    """Return only fields authorized to carry canonical instructional facts.

    Questions, distractors, FAQ answers and titles remain learner-facing, but
    cannot substitute for teaching.  A selected relationship diagram, ordered
    practice or terminology exercise can teach the exact fact assigned to its
    plan, so their semantic payload is observable alongside HTML.
    """
    fields = _learner_fields(component)
    component_type = str(component.get("type") or "").strip()
    if component_type == "html":
        return [(path, text) for path, text in fields
                if path in {"html", "content"}
                or (path.startswith("semantic_content.") and not path.endswith(".heading"))]
    if component_type == "la_diagram":
        return [(path, text) for path, text in fields
                if path.startswith("nodes[") or path.startswith("edges[")]
    if component_type == "la_sortable":
        return [(path, text) for path, text in fields
                if path.startswith("items[") or path.startswith("ordered_items[") or path.startswith("steps[")]
    if component_type == "la_crossword":
        return [(path, text) for path, text in fields if path.startswith("words[")]
    return []


def observe_lesson_content(unit: dict, expected: dict, manifest: dict | None) -> dict:
    """Return safe metadata only. All counts describe THIS UNIT, not a course."""
    findings: list[dict] = []
    finding_counts: Counter = Counter()

    def finding(code: str, path: str, **safe: Any) -> None:
        finding_counts[code] += 1
        if len(findings) < MAX_FINDINGS:
            findings.append({"code": code, "severity": "review", "path": path, **safe})

    plans = _list(expected.get("component_plan"))
    components = _list(unit.get("components"))
    spans: list[TextSpan] = []
    incomplete_teaching_scan = len(components) > MAX_COMPONENTS
    content_scan_limited = incomplete_teaching_scan
    owned_by_component: dict[int, set[str]] = {}
    ordered_section_count = 0
    practice_task_count = 0
    for i, raw in enumerate(components[:MAX_COMPONENTS]):
        component = _record(raw)
        plan = _record(plans[i]) if i < len(plans) else {}
        learner_fields = _learner_fields(component)
        arithmetic_paths = {path for path, _ in _teaching_fields(component)} if component.get("type") == "html" else set()
        if len(learner_fields) > MAX_SEGMENTS or any(len(text) > MAX_TEXT_CHARS for _, text in learner_fields):
            content_scan_limited = True
        for path, text in learner_fields[:MAX_SEGMENTS]:
            if _INTERNAL_ID.search(text[:MAX_TEXT_CHARS]):
                finding("LEARNER_INTERNAL_ID_VISIBLE", f"components[{i}].{path}")
            # Conservative arithmetic signal only; says nothing about whether
            # the formula or inputs are supported by the source. Never eval().
            # Only teaching fields: incorrect quiz distractors are intentional.
            if path in arithmetic_paths and any(Decimal(a) * Decimal(b) != Decimal(result)
                   for a, b, result in _MULTIPLICATION.findall(text[:MAX_TEXT_CHARS])):
                finding("ARITHMETIC_EXAMPLE_RESULT_MISMATCH", f"components[{i}].{path}")
        if component.get("type") == "la_diagram":
            nodes = _list(component.get("nodes"))[:64]
            connected = {endpoint for edge in _list(component.get("edges"))[:128]
                         for endpoint in (_record(edge).get("source"), _record(edge).get("target"))
                         if isinstance(endpoint, int) and not isinstance(endpoint, bool)}
            isolated = [n for n in range(len(nodes)) if n not in connected]
            if isolated:
                finding("DIAGRAM_ISOLATED_NODES_REVIEW", f"components[{i}].nodes", isolated_node_count=len(isolated))
            # Connectivity is structural only, never evidence that the edges are true.
        owned_by_component[i] = set(_ids(plan.get("source_fact_ids")))
        component_type = str(component.get("type") or "").strip()
        if component_type == "html" and plan.get("type") == "html":
            semantic = _record(component.get("semantic_content"))
            if semantic.get("version") == 2:
                sections = _list(semantic.get("sections"))[:12]
                ordered_section_count += len(sections)
                practice_task_count += sum(_record(b).get("kind") == "task" for s in sections for b in _list(_record(s).get("blocks"))[:12])
                declared = {ref for s in sections for ref in _ids(_record(s).get("learning_block_ids"))}
                missing = set(_ids(plan.get("learning_block_ids"))) - declared
                if missing:
                    finding("HTML_TEACHING_GROUP_NOT_PRESENT", f"components[{i}].semantic_content.sections", missing_group_count=len(missing))
        if component_type not in _INSTRUCTIONAL_TYPES or plan.get("type") != component_type:
            continue
        fields = _teaching_fields(component)
        for path, text in fields:
            if len(spans) >= MAX_SEGMENTS:
                incomplete_teaching_scan = True
                break
            if len(text) > MAX_TEXT_CHARS:
                incomplete_teaching_scan = True
                continue  # No truncated witness may masquerade as complete.
            normalized = _normalize(text)
            spans.append(TextSpan(i, f"components[{i}].{path}", text, normalized, frozenset(normalized.split())))
    content_scan_limited |= incomplete_teaching_scan

    # Inverted lexical index; bounded candidate comparisons, no whole-course N².
    index: dict[str, set[int]] = defaultdict(set)
    for i, span in enumerate(spans):
        for token in span.tokens:
            index[token].add(i)
    fact_rows = _list(_record(manifest).get("facts"))
    fact_index: dict[str, list[dict]] = defaultdict(list)
    for raw in fact_rows[:MAX_MANIFEST_FACTS]:
        row = _record(raw)
        if isinstance(row.get("fact_id"), str):
            fact_index[row["fact_id"]].append(row)
    owned = _ids(expected.get("source_fact_ids"))
    counts: Counter = Counter()
    witnesses = []
    comparisons = 0
    for fact_id in owned[:MAX_FACTS]:
        fact_key = sha256(fact_id.encode()).hexdigest()[:16]
        candidates = fact_index.get(fact_id, [])
        state = "NOT_ASSESSED"
        witness = None
        if len(candidates) != 1:
            code = ("FACT_OBSERVATION_BUDGET" if len(fact_rows) > MAX_MANIFEST_FACTS else "FACT_TEXT_UNAVAILABLE") if not candidates else "FACT_MANIFEST_AMBIGUOUS"
        else:
            fact_text = candidates[0].get("text")
            if not isinstance(fact_text, str) or not fact_text.strip():
                code = "FACT_TEXT_UNAVAILABLE"
            elif len(fact_text) > MAX_TEXT_CHARS:
                code = "FACT_OBSERVATION_BUDGET"
            else:
                norm = _normalize(fact_text)
                tokens = frozenset(norm.split())
                if len(tokens) < 4:
                    code = "FACT_FRAGMENT_REQUIRES_REVIEW"
                else:
                    votes: Counter = Counter()
                    for token in tokens:
                        for span_id in index.get(token, ()):
                            if fact_id in owned_by_component.get(spans[span_id].component_index, set()):
                                votes[span_id] += 1
                    ranked = sorted(votes, key=lambda n: (-votes[n], n))[:MAX_CANDIDATES]
                    comparisons += len(ranked)
                    ranked_spans = [spans[n] for n in ranked]
                    literal = next((s for s in ranked_spans if f" {norm} " in f" {s.normalized} "
                                    and bool(_NEGATION.search(norm)) == bool(_NEGATION.search(s.normalized))
                                    and set(_NUMBERS.findall(fact_text)).issubset(_NUMBERS.findall(s.raw))), None)
                    if literal:
                        state, code, witness = "LEXICAL_WITNESS", None, literal
                    elif incomplete_teaching_scan or len(votes) > MAX_CANDIDATES:
                        code = "FACT_OBSERVATION_BUDGET"
                    else:
                        state = "REVIEW_REQUIRED"
                        witness = ranked_spans[0] if ranked_spans and votes[ranked[0]] / len(tokens) >= 0.6 else None
                        code = "FACT_TEACHING_NOT_LOCATED"
                        if witness:
                            code = "FACT_PARAPHRASE_OR_DETAIL_REVIEW"
                            if bool(_NEGATION.search(norm)) != bool(_NEGATION.search(witness.normalized)):
                                code = "FACT_NEGATION_DIFFERENCE_REVIEW"
                            elif not set(_NUMBERS.findall(fact_text)).issubset(_NUMBERS.findall(witness.raw)):
                                code = "FACT_NUMBER_DIFFERENCE_REVIEW"
        counts[state] += 1
        path = witness.path if witness else "unit"
        if code:
            finding(code, path, fact_key=fact_key)
        if len(witnesses) < MAX_WITNESSES:
            witnesses.append({"fact_key": fact_key, "state": state, "candidate_path": witness.path if witness else None,
                              "segment_sha256": sha256(witness.raw.encode()).hexdigest()[:16] if witness else None})
    overflow = max(0, len(owned) - MAX_FACTS)
    counts["NOT_ASSESSED"] += overflow
    if overflow or len(fact_rows) > MAX_MANIFEST_FACTS or content_scan_limited:
        finding("OBSERVATION_SCOPE_LIMITED", "unit", unassessed_fact_overflow=overflow)

    brief = component_instructional_brief(expected)
    objectives = _list(expected.get("learning_objectives"))
    unit_refs = set(_ids(expected.get("learning_objective_refs")))
    requirements = []
    for n, objective in enumerate(objectives[:12], 1):
        ref = f"lo_{n}"
        if ref in unit_refs and isinstance(objective, str):
            requirements.extend((ref, action) for action in instructional_action_intents(objective))
    if "calculation" in brief["unit_action_hints"] and not any(a == "calculation" for _, a in requirements):
        requirements.append((None, "calculation"))
    check_components = [c for c in brief["components"][:MAX_COMPONENTS] if c["type"] == "problem"]
    for ref, action in requirements:
        checks = [c for c in check_components if ref is None or ref in c["local_objective_refs"]]
        check_texts = [str(_record(components[c["component_index"]]).get("question") or "")[:MAX_TEXT_CHARS]
                       for c in checks if c["component_index"] < len(components)]
        if not check_texts:
            finding("ACTION_ASSESSMENT_NOT_PLANNED_REVIEW", "unit", objective_ref=ref, action=action)
        elif action == "calculation" and not any(
                (re.search(r"\b(?:calculate|compute|tinh|bao nhieu|what.*score)\b", _normalize(q)))
                and len(_NUMBERS.findall(q)) >= 2 for q in check_texts):
            finding("CALCULATION_TASK_NOT_OBSERVED", "unit", objective_ref=ref)
        elif action in {"analysis", "selection", "procedure"} and not any(
                re.search(r"\b(?:scenario|suppose|given|because|justify|tinh huong|gia su|vi sao|giai thich|neu)\b", _normalize(q))
                for q in check_texts):
            finding("APPLICATION_TASK_REQUIRES_REVIEW", "unit", objective_ref=ref, action=action)
        if action == "calculation" and not any(
                re.search(r"\b(?:example|suppose|hypothetical|vi du|gia dinh|gia su)\b", s.normalized)
                and re.search(r"\d+\s*[x×*]\s*\d+\s*=\s*\d+", s.raw) for s in spans):
            finding("WORKED_CALCULATION_NOT_OBSERVED", "unit", objective_ref=ref)
    correct_positions = Counter()
    for i, raw in enumerate(components[:MAX_COMPONENTS]):
        c = _record(raw)
        if c.get("type") == "problem":
            positions = [n for n, option in enumerate(_list(c.get("choices"))) if _record(option).get("correct") is True]
            if len(positions) == 1:
                correct_positions[str(positions[0])] += 1
            if positions and len(positions) == 1 and len(_list(c.get("choices"))) > 1:
                choices = [_normalize(str(_record(o).get("text") or "")) for o in c["choices"]]
                if len(set(choices)) < len(choices):
                    finding("ASSESSMENT_DUPLICATE_CHOICES_REVIEW", f"components[{i}]")
    return {"observation_version": VERSION, "mode": "shadow", "blocking": False,
            "semantic_fidelity": "not_measured", "semantic_coverage": "not_measured",
            "owned_fact_count": len(owned), "fact_state_counts": dict(counts),
            "candidate_comparisons": comparisons, "teaching_span_count": len(spans),
            "ordered_section_count": ordered_section_count, "practice_task_count": practice_task_count,
            "fact_witnesses": witnesses, "omitted_witness_count": max(0, len(owned) - len(witnesses)),
            "findings": findings, "finding_counts": dict(finding_counts),
            "omitted_finding_count": sum(finding_counts.values()) - len(findings),
            "correct_choice_position_counts": dict(correct_positions),
            "assessment_scope": "current_unit_only", "content_scan_limited": content_scan_limited}
