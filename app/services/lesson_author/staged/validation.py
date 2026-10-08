"""Staged unit validation: instructional findings, repair targets and payload merges."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from pydantic import ValidationError

from app.component_capabilities import validate_instance_plan
from app.instructional_density import INSTRUCTIONAL_DENSITY_POLICY_VERSION
from app.instructional_quality import (
    MIN_FAQ_ANSWER_CHARS,
    MIN_FAQ_QUESTION_CHARS,
    contains_generic_review_language,
    faq_answer_is_complete,
)
from app.learner_content_purity import component_learner_text, learner_content_purity_finding
from app.ordered_learning_content import semantic_shape_diagnostics
from app.services.lesson_author.errors import LessonAuthorProposalValidationError
from app.services.lesson_author.proposal_validation import (
    MIN_STAGED_LESSON_AUTHOR_HTML_TEXT_CHARS,
    semantic_learning_visible_text,
    validate_lesson_author_proposal_shape,
)
from app.services.lesson_author.staged.plan import normalize_staged_component_type
from app.services.lesson_author.staged.provider_schemas import (
    STAGED_COMPONENT_PAYLOAD_FIELDS,
    StagedDiagramEdge,
    StagedDiagramNode,
    StagedSortableItem,
    staged_component_payload_code,
)


class StagedUnitFinding(str):
    """Legacy string feedback plus metadata assigned at the rejection boundary.

    Only diagnostic() is loggable; the legacy message may contain private IDs.
    """

    code: str
    path: str
    repairable: bool

    def __new__(cls, message: str, code: str, path: str = "unit", repairable: bool = False):
        value = super().__new__(cls, message)
        value.code, value.path, value.repairable = code, path, repairable
        return value

    def diagnostic(self) -> dict[str, Any]:
        limits = {
            "HTML_INSUFFICIENT_DEPTH": {"minimum_visible_chars": MIN_STAGED_LESSON_AUTHOR_HTML_TEXT_CHARS},
            "SORTABLE_STEP_INCOMPLETE": {"minimum_items": 3, "minimum_step_chars": 14},
            "FAQ_CLARIFICATION_INCOMPLETE": {
                "minimum_question_chars": MIN_FAQ_QUESTION_CHARS,
                "minimum_answer_chars": MIN_FAQ_ANSWER_CHARS,
            },
        }
        return {"code": self.code, "path": self.path, "repairable": self.repairable, **limits.get(self.code, {})}


def staged_html_contains_faq(component: dict[str, Any]) -> bool:
    """Detect explicit FAQ presentation inside an HTML teaching component.

    This intentionally targets structural FAQ signals, not an isolated
    pedagogical question, so reflective prompts remain valid HTML content.
    """
    import unicodedata

    def normalized(value: Any) -> str:
        text = unicodedata.normalize("NFD", str(value or ""))
        return re.sub(r"\s+", " ", "".join(char for char in text if unicodedata.category(char) != "Mn").casefold()).strip()

    semantic = component.get("semantic_content")
    headings: list[str] = []
    visible = ""
    if isinstance(semantic, dict):
        sections = semantic.get("sections")
        if isinstance(sections, list):
            headings = [normalized(section.get("heading")) for section in sections if isinstance(section, dict)]
        text, reason = semantic_learning_visible_text(semantic)
        if not reason:
            visible = normalized(text)
    else:
        raw = str(component.get("html") or component.get("data") or component.get("content") or "")
        if re.search(r"<\s*(?:details|summary)\b", raw, re.IGNORECASE):
            return True
        visible = normalized(re.sub(r"<[^>]+>", " ", raw))

    markers = ("faq", "frequently asked questions", "frequently asked question",
               "cau hoi thuong gap", "hoi dap thuong gap")
    if any(any(marker in heading for marker in markers) for heading in headings):
        return True
    if any(marker in visible for marker in markers):
        return True
    questions = len(re.findall(r"(?:^|\s)(?:q|question|cau hoi)\s*\d{0,2}\s*[:.-]", visible))
    answers = len(re.findall(r"(?:^|\s)(?:a|answer|tra loi)\s*\d{0,2}\s*[:.-]", visible))
    return questions >= 2 and answers >= 2


def _staged_semantic_artifact_count(semantic: Any, artifact_type: str) -> int:
    if not isinstance(semantic, dict):
        return 0
    sections = semantic.get("sections")
    if not isinstance(sections, list):
        return 0
    blocks = [
        block
        for section in sections
        if isinstance(section, dict)
        for block in section.get("blocks", [])
        if isinstance(block, dict)
    ]
    if artifact_type == "ordered_list":
        return sum(len(block.get("items", [])) for block in blocks if block.get("kind") == "steps")
    if artifact_type == "checklist":
        return sum(len(block.get("items", [])) for block in blocks if block.get("kind") == "bullets")
    if artifact_type in {"table", "comparison"}:
        return sum(len(block.get("rows", [])) for block in blocks if block.get("kind") == "table")
    if artifact_type in {"warning", "requirement", "exception"}:
        return sum(block.get("kind") == "warning" for block in blocks)
    return 0


def staged_instructional_finding(
    component: dict[str, Any],
    index: int,
    plan: dict[str, Any] | None = None,
    output_budget: dict[str, Any] | None = None,
    purity_context: dict[str, Any] | None = None,
    required_relationships: list[list[str]] | None = None,
) -> StagedUnitFinding | None:
    """Same content-quality requirements used by acceptance and scoped repair."""
    kind = normalize_staged_component_type(component.get("type"))
    path = f"components[{index}]"
    def fail(code: str, field: str, message: str) -> StagedUnitFinding:
        return StagedUnitFinding(message, code, f"{path}.{field}", True)
    purity_code = learner_content_purity_finding(component_learner_text(component), purity_context)
    if purity_code:
        return fail(
            purity_code,
            "content",
            "Learner-facing content contains source provenance or an internal identifier; provenance belongs only in server metadata.",
        )
    if kind == "html":
        source_locked = component.get("source_locked_fallback") is True
        if not source_locked and staged_html_contains_faq(component):
            return fail("HTML_FAQ_BOUNDARY_VIOLATION", "semantic_content",
                        "HTML teaching content must not contain an FAQ section or repeated question-and-answer pairs.")
        semantic = component.get("semantic_content")
        html = str(component.get("html") or component.get("data") or component.get("content") or "")
        if semantic is not None:
            text, reason = semantic_learning_visible_text(semantic)
            if reason:
                return fail("HTML_SEMANTIC_INVALID", "semantic_content", reason)
            if not source_locked and semantic.get("version") == 2 and plan is not None:
                allowed = set(plan.get("learning_block_ids") or [])
                referenced = {ref for section in semantic["sections"] for ref in section.get("learning_block_ids", [])}
                if referenced - allowed:
                    return fail("HTML_TEACHING_GROUP_OUT_OF_SCOPE", "semantic_content.sections", "Section references a teaching group outside this component plan.")
                if allowed - referenced:
                    return fail("HTML_TEACHING_GROUP_MISSING", "semantic_content.sections", "Every approved teaching group needs its own visible explanation; a coverage ID declaration is not enough.")
        else:
            text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()
        source_chars = output_budget.get("source_content_chars") if isinstance(output_budget, dict) else None
        source_relative_minimum = (
            max(20, min(MIN_STAGED_LESSON_AUTHOR_HTML_TEXT_CHARS, int(source_chars) // 2))
            if type(source_chars) is int and source_chars > 0 else MIN_STAGED_LESSON_AUTHOR_HTML_TEXT_CHARS
        )
        if source_locked:
            # A deterministic fallback must preserve the complete locked source,
            # including legitimately terse labels, without padding or invented
            # claims. Require all available short-source text (up to the normal
            # 20-character floor). Legacy contracts do not carry source length,
            # but their source-locked builders already reject empty evidence, so
            # a one-character floor preserves availability without fabricating
            # prose merely to satisfy a length target.
            minimum_html_chars = (
                max(1, min(20, int(source_chars)))
                if type(source_chars) is int and source_chars > 0
                else 1
            )
        else:
            minimum_html_chars = source_relative_minimum
        if len(text) < minimum_html_chars:
            return fail("HTML_INSUFFICIENT_DEPTH", "semantic_content",
                        f"Generated HTML explanation is too thin: minimum {minimum_html_chars} visible characters required.")
        if output_budget is not None:
            maximum_chars = output_budget.get("max_visible_chars")
            maximum_words = output_budget.get("max_words")
            if (output_budget.get("policy_version") != INSTRUCTIONAL_DENSITY_POLICY_VERSION
                    or type(maximum_chars) is not int or not 1 <= maximum_chars <= 18_000
                    or type(maximum_words) is not int or not 1 <= maximum_words <= 2_400):
                return StagedUnitFinding(
                    "The server instructional output budget is invalid.",
                    "INSTRUCTIONAL_OUTPUT_BUDGET_INVALID",
                    f"{path}.semantic_content",
                    False,
                )
            word_count = len(re.findall(r"\w+", text, flags=re.UNICODE))
            if len(text) > maximum_chars or word_count > maximum_words:
                return fail(
                    "HTML_INSTRUCTIONAL_DENSITY_EXCEEDED",
                    "semantic_content",
                    "Generated HTML exceeds the bounded teaching treatment for this unit; the source must remain split across coherent units.",
                )
        if contains_generic_review_language(text):
            return fail("HTML_GENERIC_REVIEW_COPY", "semantic_content",
                        "Learner content contains internal source-review instructions instead of instruction.")
        if re.search(r"(?:https?://|www\.|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,})", text, re.IGNORECASE):
            return fail("HTML_NON_INSTRUCTIONAL_CONTACT_COPY", "semantic_content",
                        "Learner content contains source contact or website boilerplate.")
        if re.search(r"(.)\1{5,}", text, re.IGNORECASE):
            return fail("HTML_OCR_NOISE", "semantic_content", "Learner content contains repeated OCR noise.")
        normalized_segments = [
            re.sub(r"[^\w\d]+", " ", value.casefold()).strip()
            for value in re.findall(r"<(?:p|li|th|td|blockquote)>\s*([^<]+?)\s*</", html if semantic is None else "", re.IGNORECASE)
        ]
        normalized_segments = [value for value in normalized_segments if len(value) >= 12]
        if len(normalized_segments) != len(set(normalized_segments)):
            return fail("HTML_DUPLICATE_BLOCK", "semantic_content",
                        "Learner content repeats the same paragraph, list item, or table cell.")
        if plan is not None:
            for artifact in plan.get("required_artifacts", []):
                if not isinstance(artifact, dict):
                    continue
                artifact_type = str(artifact.get("type") or "").strip().casefold()
                if artifact_type not in {
                    "ordered_list", "checklist", "table", "warning",
                    "requirement", "exception", "comparison",
                }:
                    continue
                minimum = artifact.get("minimum_items")
                required_count = minimum if type(minimum) is int and minimum > 0 else 1
                if artifact_type in {"table", "comparison"}:
                    required_count = max(2, required_count)
                if isinstance(semantic, dict):
                    preserved_count = _staged_semantic_artifact_count(semantic, artifact_type)
                elif artifact_type in {"table", "comparison"}:
                    preserved_count = len(re.findall(r"<tr>", html, re.IGNORECASE))
                elif artifact_type == "ordered_list":
                    preserved_count = len(re.findall(r"<li>", "".join(re.findall(r"<ol>(.*?)</ol>", html, re.IGNORECASE | re.DOTALL)), re.IGNORECASE))
                elif artifact_type == "checklist":
                    preserved_count = len(re.findall(r"<li>", "".join(re.findall(r"<ul>(.*?)</ul>", html, re.IGNORECASE | re.DOTALL)), re.IGNORECASE))
                else:
                    preserved_count = len(re.findall(r"<blockquote>", html, re.IGNORECASE))
                if preserved_count < required_count:
                    return fail(
                        "REQUIRED_ARTIFACT_NOT_PRESERVED",
                        "semantic_content",
                        f"The approved {artifact_type} source structure is absent or incomplete in generated teaching content.",
                    )
    elif kind == "problem":
        if component.get("problem_type") != "multiple_choice":
            return fail("PROBLEM_SINGLE_CHOICE_REQUIRED", "problem_type",
                        "AI Instructional Design problems must use one-answer multiple choice only.")
        question = re.sub(r"\s+", " ", str(component.get("question") or "")).strip()
        choices = component.get("choices") if isinstance(component.get("choices"), list) else []
        if len(question) < 20 or not 3 <= len(choices) <= 6:
            return fail("PROBLEM_CONTENT_INCOMPLETE", "choices",
                        "Multiple-choice problems require a complete question and three to six choices.")
        choice_texts: list[str] = []
        correct_count = 0
        for choice_index, value in enumerate(choices):
            if not isinstance(value, dict):
                return fail("PROBLEM_CHOICE_INVALID", f"choices[{choice_index}]", "Problem choices must be objects.")
            choice_text = re.sub(r"\s+", " ", str(value.get("text") or "")).strip()
            if len(choice_text) < 8:
                return fail("PROBLEM_CHOICE_INVALID", f"choices[{choice_index}].text", "Problem choice is incomplete.")
            choice_texts.append(choice_text.casefold())
            correct_count += value.get("correct") is True
        if len(choice_texts) != len(set(choice_texts)) or correct_count != 1:
            return fail("PROBLEM_CORRECT_ANSWER_INVALID", "choices",
                        "Multiple-choice problems require distinct choices and exactly one correct answer.")
        explanation = re.sub(r"\s+", " ", str(component.get("explanation") or "")).strip()
        if len(explanation) < 20:
            return fail("PROBLEM_EXPLANATION_INCOMPLETE", "explanation",
                        "The correct answer needs a source-grounded explanation.")
    elif kind == "la_sortable":
        items = component.get("items") if isinstance(component.get("items"), list) else []
        # Current typed payload uses {text}; legacy source fallback uses strings.
        texts = [re.sub(r"\s+", " ", str((x.get("text") if isinstance(x, dict) else x) or "")).strip() for x in items]
        if len(texts) < 3 or any(len(x) < 14 for x in texts):
            return fail("SORTABLE_STEP_INCOMPLETE", "items", "Sortable items must be at least three complete, meaningful ordered steps.")
        if len({x.casefold() for x in texts}) != len(texts):
            return fail("SORTABLE_DUPLICATE_STEP", "items", "Sortable items must not repeat a source fragment.")
        if any(x[:1].islower() for x in texts):
            return fail("SORTABLE_STEP_FRAGMENT", "items", "Sortable items contain a sentence fragment rather than a complete step.")
        if contains_generic_review_language(" ".join(texts) + " " + str(component.get("question_text") or "")):
            return fail("SORTABLE_GENERIC_SOURCE_ORDER", "items",
                        "Sortable must represent a real procedure, not the display order of source text.")
    elif kind == "la_diagram" and required_relationships:
        nodes = component.get("nodes") if isinstance(component.get("nodes"), list) else []
        edges = component.get("edges") if isinstance(component.get("edges"), list) else []

        def relation_key(value: Any) -> str:
            return re.sub(r"[^\w\d]+", " ", str(value or "").casefold()).strip()

        labels = [
            relation_key(node.get("label")) if isinstance(node, dict) else ""
            for node in nodes
        ]
        actual_pairs = {
            (labels[edge["source"]], labels[edge["target"]])
            for edge in edges
            if isinstance(edge, dict)
            and type(edge.get("source")) is int
            and type(edge.get("target")) is int
            and 0 <= edge["source"] < len(labels)
            and 0 <= edge["target"] < len(labels)
            and labels[edge["source"]]
            and labels[edge["target"]]
        }
        for relationship in required_relationships:
            if not isinstance(relationship, list) or len(relationship) < 2:
                continue
            required_pair = (relation_key(relationship[0]), relation_key(relationship[1]))
            if required_pair not in actual_pairs:
                return fail(
                    "DIAGRAM_SOURCE_RELATION_MISSING",
                    "edges",
                    "Diagram does not preserve every explicit relationship assigned by the locked source contract.",
                )
    elif kind == "la_faq":
        items = component.get("items") if isinstance(component.get("items"), list) else []
        if len(items) < 2:
            return fail("FAQ_ITEM_COUNT_INVALID", "items", "FAQ requires at least two complete source-grounded question-and-answer items.")
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                return fail("FAQ_ITEM_INVALID", f"items[{i}]", "FAQ items must be question-and-answer objects.")
            question = re.sub(r"\s+", " ", str(item.get("question") or "")).strip()
            answer = re.sub(r"\s+", " ", str(item.get("answer") or "")).strip()
            if len(question) < MIN_FAQ_QUESTION_CHARS or not faq_answer_is_complete(answer):
                return fail("FAQ_CLARIFICATION_INCOMPLETE", f"items[{i}]", "FAQ contains a partial source line rather than a complete answer.")
            if contains_generic_review_language(question + " " + answer):
                return fail("FAQ_GENERIC_REVIEW_COPY", f"items[{i}]",
                            "FAQ must answer a learner question instead of describing source review.")
    return None


def staged_coverage_claim_diagnostic(component: dict[str, Any]) -> dict[str, Any] | None:
    """Inspect a claim without repairing it. Return only enums and counts.

    Ownership is checked independently against the frozen server plan. Sets
    below count references only; they are never written back to the payload.
    """
    claim = component.get("covered_source_fact_ids")
    owned = component.get("source_fact_ids", [])
    owned = {x for x in owned if isinstance(x, str)} if isinstance(owned, list) else set()
    values = claim if isinstance(claim, list) else []
    strings = [x for x in values if isinstance(x, str) and x.strip()]
    # A scalar string is not an array, but must not hide an unknown reference.
    references = set(strings if isinstance(claim, list) else [claim] if isinstance(claim, str) and claim.strip() else [])
    reasons = []
    if "covered_source_fact_ids" not in component:
        reasons.append("FIELD_MISSING")
    elif not isinstance(claim, list):
        reasons.append("ARRAY_REQUIRED")
    non_string = sum(not isinstance(x, str) for x in values)
    empty = sum(isinstance(x, str) and not x.strip() for x in values)
    duplicates = len(strings) - len(set(strings))
    unexpected = len(references - owned)
    if non_string:
        reasons.append("NON_STRING_ITEM")
    if empty:
        reasons.append("EMPTY_STRING_ITEM")
    if duplicates:
        reasons.append("DUPLICATE_REFERENCE")
    shape_invalid = bool(reasons)
    if unexpected:
        reasons.append("OUT_OF_SCOPE_REFERENCE")
    missing = len(owned - references)
    if missing:
        reasons.append("MISSING_OWNED_REFERENCE")
    if not reasons:
        return None
    return {"code": "INVALID_COVERAGE_ID_ARRAY" if shape_invalid else "COMPONENT_COVERAGE_OUT_OF_SCOPE" if unexpected else "COMPONENT_COVERAGE_INCOMPLETE",
            "reason_codes": reasons, "item_count": len(values), "non_string_count": non_string,
            "empty_string_count": empty, "duplicate_count": duplicates, "unexpected_count": unexpected,
            "owned_count": len(owned), "covered_owned_count": len(owned & references), "missing_count": missing}


def staged_coverage_claim_repairable(component: dict[str, Any]) -> bool:
    owned = component.get("source_fact_ids", [])
    finding = staged_coverage_claim_diagnostic(component)
    return bool(staged_fact_membership_equal(owned, owned) and owned and finding and not finding["unexpected_count"])


def staged_coverage_claim_diagnostics(unit: Any) -> list[dict[str, Any]]:
    components = unit.get("components") if isinstance(unit, dict) else None
    if not isinstance(components, list):
        return []
    return [{"component_index": i, "path": f"components[{i}].covered_source_fact_ids", **finding}
            for i, c in enumerate(components) if isinstance(c, dict) and (finding := staged_coverage_claim_diagnostic(c))][:16]


def staged_component_repair_guard(unit: Any, expected: dict[str, Any], *, allow_partial_coverage: bool = False) -> dict[str, Any] | None:
    """First exact authority rejection, safe to log; no provider values."""
    def fail(code: str, path: str, **counts: int) -> dict[str, Any]:
        return {"code": code, "path": path, "repairable": False, **counts}
    if not isinstance(unit, dict):
        return fail("UNIT_NOT_OBJECT", "unit")
    for key in ("source_fact_ids", "supporting_evidence_fact_ids"):
        if not staged_fact_membership_equal(unit.get(key, []), expected.get(key, [])):
            return fail("UNIT_EVIDENCE_MEMBERSHIP_INVALID", f"unit.{key}")
    components = unit.get("components")
    plans = expected.get("component_plan", [])
    if not isinstance(components, list) or len(components) != len(plans):
        return fail("COMPONENT_COUNT_MISMATCH", "unit.components", expected_count=len(plans), actual_count=len(components) if isinstance(components, list) else 0)
    for index, (c, p) in enumerate(zip(components, plans)):
        path = f"components[{index}]"
        if not isinstance(c, dict) or c.get("type") != p.get("type"):
            return fail("COMPONENT_TYPE_PLAN_MISMATCH", path)
        if p.get("component_plan_id") and c.get("component_plan_id") != p["component_plan_id"]:
            return fail("COMPONENT_PLAN_INSTANCE_MISMATCH", f"{path}.component_plan_id")
        for key in ("source_fact_ids", "supporting_evidence_fact_ids"):
            if not staged_fact_membership_equal(c.get(key, []), p.get(key, [])):
                return fail("COMPONENT_EVIDENCE_MEMBERSHIP_INVALID", f"{path}.{key}")
        # A malformed content claim is not canonical ownership. Only the
        # existing bounded repair may replace it AND its component content.
        # Continue checking every other component's immutable authority.
        if allow_partial_coverage and staged_coverage_claim_repairable(c):
            continue
        claim_finding = staged_coverage_claim_diagnostic(c)
        if c.get("source_fact_ids") and claim_finding and claim_finding["unexpected_count"]:
            return fail("COMPONENT_COVERAGE_OUT_OF_SCOPE", f"{path}.covered_source_fact_ids")
        covered = c.get("covered_source_fact_ids", [])
        if not staged_fact_membership_equal(covered, covered):
            return fail("INVALID_COVERAGE_ID_ARRAY", f"{path}.covered_source_fact_ids")
        if not set(c.get("source_fact_ids", [])).issubset(covered) and not (allow_partial_coverage and covered):
            return fail("COMPONENT_COVERAGE_INCOMPLETE", f"{path}.covered_source_fact_ids", missing_count=len(set(c.get("source_fact_ids", [])) - set(covered)))
        if not set(covered).issubset(expected.get("source_fact_ids", [])):
            return fail("COMPONENT_COVERAGE_OUT_OF_SCOPE", f"{path}.covered_source_fact_ids")
        if p.get("component_plan_id") and not p.get("source_fact_ids") and covered:
            return fail("SUPPORTING_COMPONENT_CLAIMS_OWNERSHIP", f"{path}.covered_source_fact_ids")
    return None


def staged_component_repair_targets(unit: Any, expected: dict[str, Any]) -> list[int]:
    """Authorize content repair only when unit/instance ownership is intact."""
    if staged_component_repair_guard(unit, expected, allow_partial_coverage=True):
        return []
    components = unit["components"]
    targets: list[int] = []
    for index, component in enumerate(components):
        instructional = staged_instructional_finding(
            component,
            index,
            expected.get("component_plan", [])[index],
            expected.get("instructional_output_budget"),
            expected.get("learner_content_purity"),
        )
        if (staged_component_payload_code(component)
                or (instructional is not None and instructional.repairable)
                or staged_coverage_claim_repairable(component)):
            targets.append(index)
    return targets


def staged_coverage_repair_diagnostics(unit: Any, targets: list[int]) -> list[dict[str, Any]]:
    """Counts only; called after the exact ownership guard has authorized targets."""
    findings = []
    for index in targets:
        component = unit["components"][index]
        claim = staged_coverage_claim_diagnostic(component)
        if claim and staged_coverage_claim_repairable(component):
            fields = ("code", "owned_count", "covered_owned_count", "missing_count")
            findings.append({"component_index": index, "component_type": component["type"],
                             **{key: claim[key] for key in fields}})
    return findings


def staged_instructional_diagnostics(unit: Any) -> list[dict[str, Any]]:
    components = unit.get("components") if isinstance(unit, dict) else None
    if not isinstance(components, list):
        return []
    findings = []
    for i, component in enumerate(components):
        if isinstance(component, dict) and (finding := staged_instructional_finding(component, i)):
            findings.append(finding.diagnostic())
            if len(findings) == 16:
                break
    return findings


def merge_staged_component_payload_delta(baseline: dict[str, Any], delta: Any, targets: list[int],
                                         *, coverage_targets: list[int] | None = None) -> dict[str, Any]:
    """Accept only addressed payload edits; derive the full envelope on server.

    Validate every edit before returning a new unit. The old full-envelope
    merger remains strict for legacy callers; this is a narrower wire contract.
    """
    if not isinstance(delta, dict) or set(delta) != {"components"}:
        raise LessonAuthorProposalValidationError("COMPONENT_REPAIR_ENVELOPE_FORBIDDEN")
    changes = delta["components"]
    originals = baseline.get("components", [])
    if (not targets or len(set(targets)) != len(targets)
            or any(type(i) is not int or not 0 <= i < len(originals) for i in targets)
            or not isinstance(changes, list) or len(changes) != len(targets)):
        raise LessonAuthorProposalValidationError("COMPONENT_REPAIR_TARGET_COUNT_INVALID")
    indexed: dict[int, dict[str, Any]] = {}
    coverage_targets = coverage_targets or []
    if (len(set(coverage_targets)) != len(coverage_targets)
            or any(type(i) is not int or i not in targets for i in coverage_targets)):
        raise LessonAuthorProposalValidationError("COMPONENT_REPAIR_COVERAGE_TARGET_INVALID")
    coverage_updates: dict[int, list[str]] = {}
    wire_fields = set().union(*STAGED_COMPONENT_PAYLOAD_FIELDS.values()) | {"title", "selection_rationale"}
    protected = {"type", "component_plan_id", "source_fact_ids", "covered_source_fact_ids", "supporting_evidence_fact_ids"}
    for change in changes:
        if not isinstance(change, dict):
            raise LessonAuthorProposalValidationError("COMPONENT_REPAIR_NOT_OBJECT")
        index = change.get("component_index")
        if type(index) is not int or index not in targets or index in indexed:
            raise LessonAuthorProposalValidationError("COMPONENT_REPAIR_TARGET_INVALID")
        if index in coverage_targets:
            original = originals[index]
            owned = original.get("source_fact_ids", [])
            if not staged_coverage_claim_repairable(original):
                raise LessonAuthorProposalValidationError("COMPONENT_REPAIR_COVERAGE_TARGET_INVALID")
            claim = change.get("covered_source_fact_ids")
            if not staged_fact_membership_equal(claim, owned):
                raise LessonAuthorProposalValidationError("COMPONENT_REPAIR_COVERAGE_CLAIM_INVALID")
            # Filling IDs alone cannot repair missing instruction. Require a
            # substantive selected-type payload edit, then run all validators.
            payload_fields = STAGED_COMPONENT_PAYLOAD_FIELDS[original["type"]]
            if not any(k in change and change[k] not in (None, [], "", {}) and change[k] != original.get(k)
                       for k in payload_fields):
                raise LessonAuthorProposalValidationError("COMPONENT_REPAIR_COVERAGE_WITHOUT_CONTENT")
            coverage_updates[index] = list(claim)
            change = {k: v for k, v in change.items() if k != "covered_source_fact_ids"}
        if protected.intersection(change):
            raise LessonAuthorProposalValidationError("COMPONENT_REPAIR_PROTECTED_FIELD_EMITTED")
        if set(change) - wire_fields - {"component_index"}:
            raise LessonAuthorProposalValidationError("COMPONENT_REPAIR_FIELD_NOT_ALLOWED")
        indexed[index] = {k: v for k, v in change.items() if k != "component_index"}
    # Only the server reconstitutes immutable title/ownership/instance identity.
    envelope = {k: deepcopy(baseline.get(k, [])) for k in ("source_fact_ids", "supporting_evidence_fact_ids")}
    envelope["title"] = baseline.get("title")
    envelope["components"] = [indexed[index] for index in targets]
    result = merge_staged_component_repair(baseline, envelope, targets)
    for index, claim in coverage_updates.items():
        result["components"][index]["covered_source_fact_ids"] = claim
    return result


def merge_staged_component_repair(baseline: dict[str, Any], replacement: Any, targets: list[int]) -> dict[str, Any]:
    """Atomic exact-scope merge. Good components and provenance never change."""
    if not isinstance(replacement, dict):
        raise LessonAuthorProposalValidationError("COMPONENT_REPAIR_NOT_OBJECT")
    if any(replacement.get(k, []) != baseline.get(k, []) for k in ("source_fact_ids", "supporting_evidence_fact_ids")):
        raise LessonAuthorProposalValidationError("COMPONENT_REPAIR_SOURCE_SCOPE_CHANGED")
    if replacement.get("title") != baseline.get("title"):
        raise LessonAuthorProposalValidationError("COMPONENT_REPAIR_UNIT_CHANGED")
    changes = replacement.get("components")
    if not isinstance(changes, list) or len(changes) != len(targets) or len(set(targets)) != len(targets):
        raise LessonAuthorProposalValidationError("COMPONENT_REPAIR_TARGET_COUNT_INVALID")
    result = deepcopy(baseline)
    for index, change in zip(targets, changes):
        if not isinstance(change, dict) or not 0 <= index < len(result["components"]):
            raise LessonAuthorProposalValidationError("COMPONENT_REPAIR_TARGET_INVALID")
        original = baseline["components"][index]
        protected = ("type", "component_plan_id", "source_fact_ids", "covered_source_fact_ids", "supporting_evidence_fact_ids")
        if any(change.get(k, original.get(k)) != original.get(k) for k in protected):
            raise LessonAuthorProposalValidationError("COMPONENT_REPAIR_AUTHORITY_CHANGED")
        code = staged_component_payload_code({**original, **change})
        if code:
            raise LessonAuthorProposalValidationError(code)
        editable = STAGED_COMPONENT_PAYLOAD_FIELDS[original["type"]] | {"title", "selection_rationale"}
        wire_fields = set().union(*STAGED_COMPONENT_PAYLOAD_FIELDS.values())
        for key, value in change.items():
            if key not in editable and key not in protected and value != original.get(key):
                # The SDK envelope can require inapplicable null/empty fields.
                # It cannot confer authority for metadata/provenance/fallback flags.
                if key not in wire_fields or value not in (None, [], ""):
                    raise LessonAuthorProposalValidationError("COMPONENT_REPAIR_FIELD_NOT_ALLOWED")
        result["components"][index] = {**original, **{k: v for k, v in change.items() if k in editable}}
    return result


def merge_checkpoint_component_fallback(
    baseline: dict[str, Any],
    fallback: Any,
    targets: list[int],
    expected: dict[str, Any],
) -> dict[str, Any]:
    """Replace only failed component instances with server-built source content.

    Provider-authored siblings remain byte-for-byte unchanged. The fallback is
    allowed to correct an invalid coverage claim, but only when every immutable
    component identity and evidence field matches the approved server plan.
    """

    baseline_components = baseline.get("components") if isinstance(baseline, dict) else None
    fallback_components = fallback.get("components") if isinstance(fallback, dict) else None
    plans = expected.get("component_plan") if isinstance(expected, dict) else None
    if (not isinstance(baseline_components, list) or not isinstance(fallback_components, list)
            or not isinstance(plans, list) or len(baseline_components) != len(plans)
            or len(fallback_components) != len(plans)):
        raise LessonAuthorProposalValidationError("COMPONENT_FAIL_SOFT_ENVELOPE_INVALID")
    normalized_targets = list(dict.fromkeys(targets))
    if (not normalized_targets
            or any(type(index) is not int or not 0 <= index < len(plans) for index in normalized_targets)):
        raise LessonAuthorProposalValidationError("COMPONENT_FAIL_SOFT_TARGET_INVALID")
    if (fallback.get("title") != expected.get("unit_title")
            or not staged_fact_membership_equal(fallback.get("source_fact_ids"), expected.get("source_fact_ids"))
            or not staged_fact_membership_equal(
                fallback.get("supporting_evidence_fact_ids", []),
                expected.get("supporting_evidence_fact_ids", []),
            )):
        raise LessonAuthorProposalValidationError("COMPONENT_FAIL_SOFT_AUTHORITY_INVALID")
    result = deepcopy(baseline)
    for index in normalized_targets:
        replacement = fallback_components[index]
        plan = plans[index]
        if (not isinstance(replacement, dict) or not isinstance(plan, dict)
                or replacement.get("type") != plan.get("type")
                or replacement.get("component_plan_id") != plan.get("component_plan_id")
                or not staged_fact_membership_equal(replacement.get("source_fact_ids", []), plan.get("source_fact_ids", []))
                or not staged_fact_membership_equal(
                    replacement.get("supporting_evidence_fact_ids", []),
                    plan.get("supporting_evidence_fact_ids", []),
                )
                or not staged_fact_membership_equal(
                    replacement.get("covered_source_fact_ids", []),
                    plan.get("source_fact_ids", []),
                )):
            raise LessonAuthorProposalValidationError("COMPONENT_FAIL_SOFT_AUTHORITY_INVALID")
        result["components"][index] = deepcopy(replacement)
    return result


def staged_fact_membership_equal(actual: Any, approved: Any) -> bool:
    """Exact IDs/ownership, independent of serialization order; no ID repair."""
    def valid(value: Any) -> bool:
        return (isinstance(value, list) and all(isinstance(x, str) and bool(x.strip()) for x in value)
                and len(value) == len(set(value)))
    return valid(actual) and valid(approved) and set(actual) == set(approved)


def staged_diagram_shape_diagnostics(component: dict[str, Any]) -> list[dict[str, Any]]:
    """Bounded, validator-owned paths/codes only; exclude values and messages."""
    findings: list[dict[str, Any]] = []
    for field_name, model, minimum, maximum in (("nodes", StagedDiagramNode, 2, 20), ("edges", StagedDiagramEdge, 1, 40)):
        values = component.get(field_name)
        if not isinstance(values, list):
            findings.append({"path": field_name, "reason": "ARRAY_REQUIRED"})
        elif not minimum <= len(values) <= maximum:
            findings.append({"path": field_name, "reason": "CARDINALITY", "actual_count": len(values), "minimum": minimum, "maximum": maximum})
        else:
            for index, value in enumerate(values):
                try:
                    model.model_validate(value)
                except ValidationError as error:
                    for detail in error.errors(include_url=False, include_context=False, include_input=False):
                        loc = detail["loc"]
                        leaf = loc[0] if loc and loc[0] in model.model_fields else None
                        reasons = {"missing": "FIELD_REQUIRED", "literal_error": "INVALID_ENUM", "string_too_long": "TEXT_TOO_LONG",
                                   "string_too_short": "TEXT_TOO_SHORT", "greater_than_equal": "NEGATIVE_INDEX"}
                        findings.append({"path": f"{field_name}[{index}]" + (f".{leaf}" if leaf else ""),
                                         "reason": reasons.get(detail["type"], "FIELD_TYPE_INVALID")})
                        if len(findings) >= 8:
                            return findings
    return findings[:8]


def staged_payload_diagnostics(unit: Any) -> list[dict[str, Any]]:
    if not isinstance(unit, dict) or not isinstance(unit.get("components"), list):
        return []
    result = []
    for index, component in enumerate(unit["components"]):
        if not isinstance(component, dict):
            continue
        code = staged_component_payload_code(component)
        if code:
            choices = component.get("choices")
            result.append({
                "component_index": index,
                "component_type": normalize_staged_component_type(component.get("type")) or "unknown",
                "code": code,
                "choice_count": len(choices) if isinstance(choices, list) else 0,
                "correct_choice_count": sum(isinstance(c, dict) and c.get("correct") is True for c in choices) if isinstance(choices, list) else 0,
                # Validator-owned text describes shape/limits only, never values.
                **({"semantic_shape_reason": semantic_learning_visible_text(component.get("semantic_content"))[1]}
                   if code == "HTML_SEMANTIC_INVALID" else {}),
                **({"semantic_shape": semantic_shape_diagnostics(component.get("semantic_content"))}
                   if code == "HTML_SEMANTIC_INVALID" else {}),
                **({"shape_findings": staged_diagram_shape_diagnostics(component)}
                   if component.get("type") == "la_diagram" and code == "COMPONENT_PAYLOAD_SCHEMA_INVALID" else {}),
                **({"shape_findings": staged_sortable_shape_diagnostics(component)}
                   if component.get("type") == "la_sortable" and code == "COMPONENT_PAYLOAD_SCHEMA_INVALID" else {}),
            })
    return result


def staged_sortable_shape_diagnostics(component: dict[str, Any]) -> list[dict[str, Any]]:
    """Safe field/cardinality diagnostics; never include Pydantic input values."""
    items = component.get("items")
    if not isinstance(items, list):
        return [{"path": "items", "reason": "ARRAY_REQUIRED"}]
    if not 3 <= len(items) <= 10:
        return [{"path": "items", "reason": "CARDINALITY", "actual_count": len(items), "minimum": 3, "maximum": 10}]
    findings = []
    for i, item in enumerate(items):
        try:
            StagedSortableItem.model_validate(item)
        except ValidationError as error:
            for detail in error.errors(include_url=False, include_context=False, include_input=False):
                field = ".text" if detail["loc"] and detail["loc"][0] == "text" else ""
                reason = {"missing": "FIELD_REQUIRED", "string_too_long": "TEXT_TOO_LONG", "string_too_short": "TEXT_TOO_SHORT"}.get(detail["type"], "FIELD_TYPE_INVALID")
                findings.append({"path": f"items[{i}]{field}", "reason": reason})
                if len(findings) == 8:
                    return findings
    return findings


def staged_evidence_scope_diagnostics(unit: Any, expected: dict[str, Any]) -> list[dict[str, Any]]:
    """Bounded mismatch metadata, including order-only differences; no IDs/text."""
    if not isinstance(unit, dict):
        return [{"path": "unit", "reason": "MISSING_OR_INVALID_UNIT"}]
    pairs = [("unit", unit, expected)]
    components = unit.get("components", [])
    if isinstance(components, list):
        pairs += [(f"components[{i}]", c, p) for i, (c, p) in enumerate(zip(components, expected.get("component_plan", [])))
                  if isinstance(c, dict) and isinstance(p, dict)]
    findings = []
    for path, actual, approved in pairs:
        for field_name in ("source_fact_ids", "supporting_evidence_fact_ids"):
            value, required = actual.get(field_name, []), approved.get(field_name, [])
            if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
                findings.append({"path": f"{path}.{field_name}", "reason": "INVALID_ID_ARRAY"})
            elif value != required:
                got, want = set(value), set(required)
                findings.append({"path": f"{path}.{field_name}", "reason": "ORDER_ONLY" if got == want and len(value) == len(required) else "MEMBERSHIP_MISMATCH",
                                 "expected_count": len(required), "actual_count": len(value),
                                 "missing_count": len(want - got), "unexpected_count": len(got - want),
                                 "duplicate_count": len(value) - len(got)})
    return findings[:16]


def validate_staged_unit_content(
    unit: dict[str, Any],
    expected: dict[str, Any] | None = None,
    *, strict_payload: bool = False,
) -> StagedUnitFinding | None:
    """Validate one generated unit before it can poison the full proposal."""
    if isinstance(unit, dict):
        evidence_nodes = [("unit", unit)]
        components = unit.get("components", [])
        if isinstance(components, list):
            evidence_nodes += [(f"components[{i}]", c) for i, c in enumerate(components) if isinstance(c, dict)]
        for path, node in evidence_nodes:
            for field in ("source_fact_ids", "supporting_evidence_fact_ids", "covered_source_fact_ids"):
                ids = node.get(field, [])
                if not staged_fact_membership_equal(ids, ids):
                    repairable = bool(field == "covered_source_fact_ids" and path != "unit" and expected
                                      and staged_component_repair_guard(unit, expected, allow_partial_coverage=True) is None
                                      and staged_coverage_claim_repairable(node))
                    return StagedUnitFinding(f"{path}.{field}: INVALID_FACT_ID_ARRAY", "INVALID_FACT_ID_ARRAY", f"{path}.{field}", repairable)
                if field == "covered_source_fact_ids" and path != "unit" and expected and node.get("source_fact_ids"):
                    finding = staged_coverage_claim_diagnostic(node)
                    if finding and finding["unexpected_count"]:
                        return StagedUnitFinding("Coverage references exceed the component's exact owned scope.",
                                                "COMPONENT_COVERAGE_OUT_OF_SCOPE", f"{path}.{field}")
    if strict_payload:
        for finding in staged_payload_diagnostics(unit):
            return StagedUnitFinding(f"components[{finding['component_index']}]: {finding['code']}", finding["code"], f"components[{finding['component_index']}]", True)
    candidate = {
        "chapters": [{
            "title": "staged",
            "lessons": [{"title": "staged", "units": [unit]}],
        }],
    }
    try:
        validate_lesson_author_proposal_shape(candidate)
    except LessonAuthorProposalValidationError as error:
        return StagedUnitFinding(re.sub(r"\s+", " ", str(error)).strip()[:240], error.code, error.path, error.repairable)

    if expected:
        expected_types = [
            normalize_staged_component_type(value)
            for value in expected.get("component_types", [])
        ]
        actual_components = unit.get("components") if isinstance(unit.get("components"), list) else []
        actual_types = [
            normalize_staged_component_type(component.get("type"))
            for component in actual_components
            if isinstance(component, dict)
        ]
        instance_plans = expected.get("component_plan", [])
        instance_contract = any(p.get("component_plan_id") for p in instance_plans)
        if instance_contract:
            try:
                validate_instance_plan(instance_plans)
            except ValueError as error:
                return StagedUnitFinding(str(error), "APPROVED_COMPONENT_PLAN_INVALID", "unit.component_plan")
            if [p.get("component_plan_id") for p in instance_plans] != [c.get("component_plan_id") for c in actual_components]:
                return StagedUnitFinding("COMPONENT_PLAN_INSTANCE_MISMATCH", "COMPONENT_PLAN_INSTANCE_MISMATCH", "unit.components")
        if [value for value in actual_types if value] != [value for value in expected_types if value]:
            return StagedUnitFinding(
                "Component formats do not match the approved source-based plan: "
                f"expected {expected_types}, received {actual_types}.", "COMPONENT_TYPE_PLAN_MISMATCH", "unit.components"
            )

        expected_fact_ids = {
            str(fact_id).strip()
            for fact_id in expected.get("source_fact_ids", [])
            if str(fact_id).strip()
        }
        expected_supporting_evidence_fact_ids = {
            str(fact_id).strip()
            for fact_id in expected.get("supporting_evidence_fact_ids", [])
            if str(fact_id).strip()
        }
        unit_fact_ids = {
            str(fact_id).strip()
            for fact_id in unit.get("source_fact_ids", [])
            if str(fact_id).strip()
        }
        if unit_fact_ids != expected_fact_ids:
            missing = expected_fact_ids - unit_fact_ids
            unexpected = unit_fact_ids - expected_fact_ids
            details = []
            if missing:
                details.append(f"missing {sorted(missing)[:4]}")
            if unexpected:
                details.append(f"outside {sorted(unexpected)[:4]}")
            return StagedUnitFinding("Unit source facts do not exactly match the approved Blueprint assignment: " + "; ".join(details), "UNIT_FACT_OWNERSHIP_MISMATCH", "unit.source_fact_ids")
        unit_supporting_evidence_fact_ids = {
            str(fact_id).strip()
            for fact_id in unit.get("supporting_evidence_fact_ids", [])
            if str(fact_id).strip()
        }
        if unit_supporting_evidence_fact_ids != expected_supporting_evidence_fact_ids:
            return StagedUnitFinding("Unit supporting evidence does not exactly match the approved V5 evidence contract.", "UNIT_SUPPORTING_EVIDENCE_MISMATCH", "unit.supporting_evidence_fact_ids")
        assigned_fact_ids: set[str] = set()
        html_fact_ids: set[str] = set()
        plan_fact_ids_by_type = {
            normalize_staged_component_type(plan.get("type")): {
                str(fact_id).strip()
                for fact_id in plan.get("source_fact_ids", [])
                if str(fact_id).strip()
            }
            for plan in expected.get("component_plan", [])
            if isinstance(plan, dict) and normalize_staged_component_type(plan.get("type"))
        }
        plan_supporting_fact_ids_by_type = {
            normalize_staged_component_type(plan.get("type")): {
                str(fact_id).strip()
                for fact_id in plan.get("supporting_evidence_fact_ids", [])
                if str(fact_id).strip()
            }
            for plan in expected.get("component_plan", [])
            if isinstance(plan, dict) and normalize_staged_component_type(plan.get("type"))
        }
        enforce_component_ownership = bool(plan_fact_ids_by_type)
        for component_index, component in enumerate(actual_components):
            if not isinstance(component, dict):
                continue
            component_type = normalize_staged_component_type(component.get("type"))
            instructional_failure = staged_instructional_finding(
                component,
                component_index,
                instance_plans[component_index] if instance_contract else None,
                expected.get("instructional_output_budget"),
                expected.get("learner_content_purity"),
                expected.get("diagram_relationships_by_plan_id", {}).get(
                    str(component.get("component_plan_id") or ""),
                    [],
                ),
            )
            if instructional_failure:
                return instructional_failure
            component_fact_ids = {
                str(fact_id).strip()
                for fact_id in component.get("source_fact_ids", [])
                if str(fact_id).strip()
            }
            expected_instance = instance_plans[component_index] if instance_contract else None
            if expected_fact_ids and not component_fact_ids and not (expected_instance and expected_instance.get("supporting_evidence_fact_ids")):
                return StagedUnitFinding("Every component must declare the source_fact_ids supporting it.", "COMPONENT_FACTS_MISSING", f"components[{component_index}].source_fact_ids")
            invalid = component_fact_ids - expected_fact_ids
            if invalid:
                return StagedUnitFinding(f"Component declared source facts outside its unit: {sorted(invalid)[:4]}.", "COMPONENT_FACT_OUT_OF_SCOPE", f"components[{component_index}].source_fact_ids")
            expected_component_fact_ids = plan_fact_ids_by_type.get(component_type, set())
            if expected_instance is not None:
                expected_component_fact_ids = set(expected_instance.get("source_fact_ids", []))
                if component_fact_ids != expected_component_fact_ids:
                    return StagedUnitFinding("Component instance changed canonical ownership.", "COMPONENT_FACT_OWNERSHIP_MISMATCH", f"components[{component_index}].source_fact_ids")
            if expected_component_fact_ids and component_fact_ids != expected_component_fact_ids:
                return StagedUnitFinding("Component source facts do not match its approved Blueprint ownership contract.", "COMPONENT_FACT_OWNERSHIP_MISMATCH", f"components[{component_index}].source_fact_ids")
            expected_component_supporting_fact_ids = plan_supporting_fact_ids_by_type.get(component_type, set())
            if expected_instance is not None:
                expected_component_supporting_fact_ids = set(expected_instance.get("supporting_evidence_fact_ids", []))
            component_supporting_fact_ids = {
                str(fact_id).strip()
                for fact_id in component.get("supporting_evidence_fact_ids", [])
                if str(fact_id).strip()
            }
            if component_supporting_fact_ids != expected_component_supporting_fact_ids:
                return StagedUnitFinding("Component supporting evidence does not match its approved V5 evidence contract.", "COMPONENT_SUPPORTING_EVIDENCE_MISMATCH", f"components[{component_index}].supporting_evidence_fact_ids")
            if not expected_fact_ids and not component_supporting_fact_ids:
                return StagedUnitFinding("Supporting-only component requires resolved read-only evidence.", "COMPONENT_SUPPORTING_EVIDENCE_MISSING", f"components[{component_index}].supporting_evidence_fact_ids")
            covered_fact_ids = {
                str(fact_id).strip()
                for fact_id in component.get("covered_source_fact_ids", [])
                if str(fact_id).strip()
            }
            if expected_instance is not None and not expected_component_fact_ids and covered_fact_ids:
                return StagedUnitFinding("Supporting assessment instance must not claim canonical coverage.", "SUPPORTING_COMPONENT_CLAIMS_OWNERSHIP", f"components[{component_index}].covered_source_fact_ids")
            if enforce_component_ownership and expected_fact_ids and not covered_fact_ids and not (expected_instance and not expected_component_fact_ids):
                return StagedUnitFinding("Every component must declare covered_source_fact_ids.", "COMPONENT_COVERAGE_MISSING", f"components[{component_index}].covered_source_fact_ids",
                                        repairable=staged_component_repair_guard(unit, expected, allow_partial_coverage=True) is None)
            if enforce_component_ownership and not component_fact_ids.issubset(covered_fact_ids):
                return StagedUnitFinding("Component covered_source_fact_ids must include every source_fact_id it owns.", "COMPONENT_COVERAGE_INCOMPLETE", f"components[{component_index}].covered_source_fact_ids", repairable=True)
            if enforce_component_ownership and covered_fact_ids - expected_fact_ids:
                return StagedUnitFinding("Component covered_source_fact_ids contain facts outside its unit.", "COMPONENT_COVERAGE_OUT_OF_SCOPE", f"components[{component_index}].covered_source_fact_ids")
            assigned_fact_ids.update(component_fact_ids)
            if normalize_staged_component_type(component.get("type")) == "html":
                html_fact_ids.update(component_fact_ids)
        missing = expected_fact_ids - assigned_fact_ids
        if missing:
            return StagedUnitFinding(f"Components do not collectively cover source facts: {sorted(missing)[:6]}.", "UNIT_COVERAGE_INCOMPLETE")
        if not instance_contract:
            missing_from_html = expected_fact_ids - html_fact_ids
            if missing_from_html:
                return StagedUnitFinding(f"HTML explanation does not cover assigned source facts: {sorted(missing_from_html)[:6]}.", "HTML_FACT_COVERAGE_INCOMPLETE")
    return None


def lesson_author_proposal_quality_metrics(proposal: dict[str, Any]) -> dict[str, Any]:
    """Emit compact diagnostics without logging source text or learner content."""
    component_counts: dict[str, int] = {}
    html_lengths: list[int] = []
    source_locked_components = 0
    unit_count = 0
    for chapter in proposal.get("chapters", []):
        if not isinstance(chapter, dict):
            continue
        for lesson in chapter.get("lessons", []):
            if not isinstance(lesson, dict):
                continue
            for unit in lesson.get("units", []):
                if not isinstance(unit, dict):
                    continue
                unit_count += 1
                for component in unit.get("components", []):
                    if not isinstance(component, dict):
                        continue
                    component_type = normalize_staged_component_type(component.get("type")) or "unknown"
                    component_counts[component_type] = component_counts.get(component_type, 0) + 1
                    if component.get("source_locked_fallback") is True:
                        source_locked_components += 1
                    if component_type == "html":
                        semantic_content = component.get("semantic_content")
                        if semantic_content is not None:
                            visible_text, _semantic_failure = semantic_learning_visible_text(semantic_content)
                        else:
                            visible_text = re.sub(
                                r"\s+",
                                " ",
                                re.sub(r"<[^>]+>", " ", str(component.get("html") or component.get("data") or "")),
                            ).strip()
                        html_lengths.append(len(visible_text))
    return {
        "units": unit_count,
        "component_counts": component_counts,
        "min_html_text_chars": min(html_lengths) if html_lengths else None,
        "source_locked_components": source_locked_components,
    }
