"""Source-locked staged skeletons and deterministic source-locked units."""

from __future__ import annotations

import html
import re
from copy import deepcopy
from typing import Any

from app.instructional_quality import (
    build_source_grounded_single_choice,
    clean_source_facts,
    render_source_locked_html,
    source_clarification_signals,
)
from app.lesson_prompt_policy import component_instructional_brief
from app.schemas.lesson_author import RagLessonAuthorRequest
from app.services.lesson_author.errors import LessonAuthorProposalValidationError
from app.services.lesson_author.proposal_validation import _normalize_structural_title
from app.services.lesson_author.staged.plan import (
    MIN_SOURCE_LOCKED_HTML_TEXT_CHARS,
    STAGED_LESSON_AUTHOR_RECOVERY_UNITS,
    STAGED_LESSON_AUTHOR_UNIT_CONTEXT_CHARS,
    _source_locked_ordered_items,
    normalize_staged_component_type,
)
from app.services.lesson_author.staged.skeleton import _build_blueprint_locked_staged_skeleton, lesson_author_title_key
from app.services.retrieval.source_coverage import SOURCE_REF_RE, _source_chunk_number, _source_page_number
from app.services.text import clean_text


def _requested_staged_chapter_title(request: RagLessonAuthorRequest) -> str:
    """Derive a semantic title without trusting source filenames or ranges."""
    message = re.sub(r"\s+", " ", request.user_message).strip()
    match = re.search(
        r"\b(?:chương|chuong|chapter)\s*\d+\s*(?:[:.\-)]+\s*|\s+)(.+)$",
        message,
        flags=re.IGNORECASE,
    )
    candidate = match.group(1).strip() if match else message
    candidate = re.sub(
        r"^(?:hãy\s+)?(?:soạn|soan|viết|viet|tạo|tao|draft|write|create)"
        r"(?:\s+(?:chi tiết|chi tiet|nội dung|noi dung|detailed|content))*\s+",
        "",
        candidate,
        flags=re.IGNORECASE,
    ).strip(" :-")
    fallback = "Nội dung khóa học" if request.locale != "en" else "Course content"
    return _normalize_structural_title(candidate, fallback)


def _source_locked_unit_title(facts: list[dict[str, Any]], index: int, locale: str) -> str:
    fallback = f"Nội dung trọng tâm {index}" if locale != "en" else f"Core content {index}"
    raw = re.sub(r"\s+", " ", str(facts[0].get("text") or "")).strip() if facts else ""
    raw = re.sub(r"^(?:[-•]\s*|\d+\s*[.)-]\s*)", "", raw).strip()
    if not raw:
        return fallback
    prefix = raw.split(":", 1)[0].strip()
    if 8 <= len(prefix) <= 100:
        raw = prefix
    else:
        raw = re.split(r"(?<=[.!?])\s+", raw, maxsplit=1)[0].strip()
    return _normalize_structural_title(raw[:120], fallback)


def _blueprint_lesson_fact_groups(
    request: RagLessonAuthorRequest,
    facts: list[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    match = re.search(
        r"^Blueprint lessons:\s*(.+)$",
        request.target_scope_instruction or "",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if not match:
        return []
    entries = re.split(r"\s+\|\s+(?=\d+\.\s+)", match.group(1).strip())
    lesson_specs: list[tuple[str, set[str]]] = []
    for entry in entries:
        title_match = re.match(r"\d+\.\s+(.+?)\s+\(", entry.strip())
        refs = {value.casefold() for value in SOURCE_REF_RE.findall(entry)}
        if title_match and refs:
            lesson_specs.append((title_match.group(1).strip(), refs))
    if not lesson_specs:
        return []

    groups: list[list[dict[str, Any]]] = [[] for _ in lesson_specs]
    matched_count = 0
    for fact in facts:
        source_ref = str(fact.get("source_ref") or "").strip().casefold()
        target_index = next(
            (index for index, (_title, refs) in enumerate(lesson_specs) if source_ref in refs),
            None,
        )
        if target_index is None:
            # Chapter-level heading facts belong to the first approved lesson;
            # they remain covered without creating a synthetic extra lesson.
            target_index = 0
        else:
            matched_count += 1
        groups[target_index].append(fact)
    if matched_count == 0:
        return []
    return [
        (title, group)
        for (title, _refs), group in zip(lesson_specs, groups)
        if group
    ]


def _source_fact_group_key(fact: dict[str, Any]) -> tuple[str, Any, Any]:
    page = _source_page_number(fact.get("source_page"))
    if page is not None:
        return ("page", page, None)
    source_ref = str(fact.get("source_ref") or "").strip().casefold()
    if source_ref:
        return ("ref", str(fact.get("document_id") or ""), source_ref)
    return (
        "chunk",
        str(fact.get("document_id") or ""),
        _source_chunk_number(fact.get("source_chunk")),
    )


def build_source_locked_staged_skeleton(
    request: RagLessonAuthorRequest,
    manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a bounded, lossless skeleton when provider planning is malformed.

    Facts remain in source-page order, every fact is assigned exactly once,
    and content is still generated in the normal per-unit stage. This fallback
    changes only grouping, never source text.
    """
    if request.blueprint_architecture is not None:
        return _build_blueprint_locked_staged_skeleton(request, manifest)

    facts = [
        fact
        for fact in (manifest or {}).get("facts", [])
        if isinstance(fact, dict) and str(fact.get("fact_id") or "").strip()
    ]
    if not facts:
        raise LessonAuthorProposalValidationError(
            "Không có source fact để phục hồi cấu trúc nội dung chương.",
        )

    blueprint_groups = _blueprint_lesson_fact_groups(request, facts)
    grouped_titles: list[str | None] = []
    if blueprint_groups:
        grouped_facts = [group for _title, group in blueprint_groups]
        grouped_titles = [title for title, _group in blueprint_groups]
    else:
        locator_groups: list[list[dict[str, Any]]] = []
        current_locator: Any = object()
        for fact in facts:
            locator = _source_fact_group_key(fact)
            if not locator_groups or locator != current_locator:
                locator_groups.append([])
                current_locator = locator
            locator_groups[-1].append(fact)

        group_count = min(STAGED_LESSON_AUTHOR_RECOVERY_UNITS, len(locator_groups))
        base_size, remainder = divmod(len(locator_groups), group_count)
        grouped_facts = []
        cursor = 0
        for group_index in range(group_count):
            locator_count = base_size + (1 if group_index < remainder else 0)
            grouped_facts.append([
                fact
                for locator_group in locator_groups[cursor : cursor + locator_count]
                for fact in locator_group
            ])
            cursor += locator_count
        grouped_titles = [None] * len(grouped_facts)

    seen_titles: dict[str, int] = {}
    units: list[dict[str, Any]] = []
    for index, (unit_facts, approved_title) in enumerate(zip(grouped_facts, grouped_titles), start=1):
        title = (
            _normalize_structural_title(approved_title, "Nội dung trọng tâm")
            if approved_title
            else _source_locked_unit_title(unit_facts, index, request.locale)
        )
        title_key = lesson_author_title_key(title)
        seen_titles[title_key] = seen_titles.get(title_key, 0) + 1
        if seen_titles[title_key] > 1:
            suffix = f"phần {seen_titles[title_key]}" if request.locale != "en" else f"part {seen_titles[title_key]}"
            title = f"{title} - {suffix}"
        units.append({
            "title": title,
            "source_fact_ids": [str(fact["fact_id"]).strip() for fact in unit_facts],
            "component_plan": [],
        })

    chapter_title = _requested_staged_chapter_title(request)
    lesson_title = (
        "Thực hành và nội dung trọng tâm"
        if request.locale != "en" and "thực hành" in chapter_title.casefold()
        else "Nội dung trọng tâm"
        if request.locale != "en"
        else "Practice and core content"
        if "practice" in chapter_title.casefold()
        else "Core content"
    )
    return {
        "chapters": [{
            "title": chapter_title,
            "lessons": [{
                "title": lesson_title,
                "units": units,
            }],
        }],
    }


def staged_unit_source_material(
    expected: dict[str, Any],
    source_rows: list[dict[str, Any]],
    manifest: dict[str, Any] | None,
) -> tuple[str, str]:
    """Return owned facts plus separately named V5 read-only evidence."""
    expected_fact_ids = {
        str(fact_id).strip()
        for fact_id in expected.get("source_fact_ids", [])
        if isinstance(fact_id, str) and fact_id.strip()
    }
    supporting_evidence_fact_ids = {
        str(fact_id).strip()
        for fact_id in expected.get("supporting_evidence_fact_ids", [])
        if isinstance(fact_id, str) and fact_id.strip()
    }
    facts = [
        fact
        for fact in (manifest or {}).get("facts", [])
        if isinstance(fact, dict) and str(fact.get("fact_id") or "").strip() in expected_fact_ids
    ]
    supporting_facts = [
        fact
        for fact in (manifest or {}).get("supporting_evidence_facts", [])
        if isinstance(fact, dict) and str(fact.get("fact_id") or "").strip() in supporting_evidence_fact_ids
    ]
    if expected.get("strict_v5_evidence") is True:
        resolved_owned = {str(fact.get("fact_id") or "").strip() for fact in facts}
        resolved_supporting = {str(fact.get("fact_id") or "").strip() for fact in supporting_facts}
        missing_owned = sorted(expected_fact_ids - resolved_owned)
        missing_supporting = sorted(supporting_evidence_fact_ids - resolved_supporting)
        if missing_owned or missing_supporting:
            raise LessonAuthorProposalValidationError(
                "V5 unit evidence contract is unresolved. "
                f"Missing owned: {', '.join(missing_owned[:6]) or 'none'}; "
                f"missing supporting: {', '.join(missing_supporting[:6]) or 'none'}.",
            )
        if not facts and not supporting_facts:
            raise LessonAuthorProposalValidationError("V5 unit has no resolved evidence for staged generation.")
    coverage_lines: list[str] = []
    for fact in facts:
        fact_text = re.sub(r"\s+", " ", str(fact.get("text") or "")).strip()
        coverage_lines.append(f"- [{fact['fact_id']}] {fact_text}")
    for fact in supporting_facts:
        fact_text = re.sub(r"\s+", " ", str(fact.get("text") or "")).strip()
        coverage_lines.append(f"- [supporting:{fact['fact_id']}] {fact_text}")
    coverage = "\n".join(coverage_lines) or "Không có checklist fact riêng cho Mục này."
    all_evidence_facts = [*facts, *supporting_facts]
    if any(str(fact.get("source_ref") or "").strip() for fact in all_evidence_facts):
        # Inferred DOCX headings can share one physical chunk. Passing the
        # whole chunk would leak neighboring blueprint chapters into this unit.
        source_context = "\n".join(
            re.sub(r"\s+", " ", str(fact.get("text") or "")).strip()
            for fact in all_evidence_facts
            if str(fact.get("text") or "").strip()
        )
        return coverage, source_context
    pages: set[int] = set()
    for fact in all_evidence_facts:
        page = _source_page_number(fact.get("source_page"))
        if page is not None:
            pages.add(page)
    chunks = {
        (str(fact.get("document_id") or ""), _source_chunk_number(fact.get("source_chunk")))
        for fact in all_evidence_facts
        if _source_chunk_number(fact.get("source_chunk")) is not None
    }
    scoped_rows = []
    for row in source_rows:
        page = _source_page_number(row.get("source_page"))
        chunk_key = (
            str(row.get("document_id") or ""),
            _source_chunk_number(row.get("chunk_no")),
        )
        if (pages and page in pages) or (chunks and chunk_key in chunks):
            scoped_rows.append(row)
    if not scoped_rows and expected.get("strict_v5_evidence") is True:
        fact_context = "\n".join(
            re.sub(r"\s+", " ", str(fact.get("text") or "")).strip()
            for fact in all_evidence_facts
            if str(fact.get("text") or "").strip()
        )
        if fact_context:
            return coverage, fact_context
        raise LessonAuthorProposalValidationError("V5 unit evidence did not resolve to approved source text.")
    if not scoped_rows:
        scoped_rows = source_rows[:4]
    source_context_parts: list[str] = []
    source_context_chars = 0
    for row in scoped_rows:
        content = clean_text(str(row.get("content") or ""))
        if not content or source_context_chars + len(content) > STAGED_LESSON_AUTHOR_UNIT_CONTEXT_CHARS:
            continue
        source_context_parts.append(content)
        source_context_chars += len(content)
    source_context = "\n\n".join(source_context_parts)
    return coverage, source_context


def staged_unit_source_evidence_bundle(
    expected: dict[str, Any],
    manifest: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return the immutable structured evidence bundle only inside unit scope."""

    bundle = (manifest or {}).get("source_evidence_bundle")
    if not isinstance(bundle, dict):
        return None
    allowed_fact_ids = {
        str(value).strip()
        for field in ("source_fact_ids", "supporting_evidence_fact_ids")
        for value in expected.get(field, [])
        if isinstance(value, str) and value.strip()
    }
    elements = bundle.get("elements")
    if not isinstance(elements, list):
        return None
    for element in elements:
        if not isinstance(element, dict):
            return None
        element_fact_ids = {
            str(value).strip()
            for value in element.get("source_fact_ids", [])
            if isinstance(value, str) and value.strip()
        }
        if not element_fact_ids.issubset(allowed_fact_ids):
            if expected.get("strict_v5_evidence") is True:
                raise LessonAuthorProposalValidationError(
                    "Structured evidence bundle exceeds the immutable unit fact scope."
                )
            return None
    return deepcopy(bundle)


def build_staged_instructional_contract(
    expected: dict[str, Any],
    source_coverage_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the exact approved contract serialized into the Stage-2 prompt."""

    return {
        "component_instructional_brief": component_instructional_brief(expected),
        "lesson_learning_objectives": expected.get("learning_objectives", []),
        "assessment_required": expected.get("assessment_required") is True,
        "assessment_objective_refs": expected.get("assessment_objective_refs", []),
        "unit_purpose": expected.get("unit_purpose", ""),
        "concept_ids": expected.get("concept_ids", []),
        "learning_objective_refs": expected.get("learning_objective_refs", []),
        "learning_blocks": [
            {
                "id": str(block.get("id") or "")[:80],
                "intent": str(block.get("intent") or "")[:80],
                "learning_objective_refs": (
                    block.get("learning_objective_refs")
                    if isinstance(block.get("learning_objective_refs"), list)
                    else []
                ),
                "source_fact_ids": (
                    block.get("source_fact_ids")
                    if isinstance(block.get("source_fact_ids"), list)
                    else []
                ),
            }
            for block in expected.get("learning_blocks", [])
            if isinstance(block, dict)
        ],
        "supporting_evidence_fact_ids": expected.get("supporting_evidence_fact_ids", []),
        "instructional_output_budget": expected.get("instructional_output_budget"),
        "source_evidence_bundle": staged_unit_source_evidence_bundle(
            expected,
            source_coverage_manifest,
        ),
    }


def _source_locked_sequence_items(facts: list[dict[str, Any]]) -> list[str]:
    """Recover source-order items when a PDF extractor removed list markers.

    This is used only after an approved Blueprint has already selected an
    ordering component. The Blueprint is the evidence that these fact lines
    form a sequence; the fallback still copies each learner item from source
    text and never invents a step.
    """
    explicit = _source_locked_ordered_items(facts)
    if len(explicit) >= 3:
        return explicit

    candidates: list[str] = []
    seen: set[str] = set()
    for line in _source_locked_html_facts([
        str(fact.get("text") or "")
        for fact in facts
        if str(fact.get("text") or "").strip()
    ]):
        normalized = re.sub(r"^(?:\d+\s*[.)-]\s+|[-•●▪◦]\s+)", "", line)
        normalized = re.sub(r"\s+", " ", normalized).strip(" -:")
        # Discard only visual labels/fragments. Complete source statements,
        # including marker-less PDF list rows, remain in source order.
        if len(normalized) < 14:
            continue
        key = normalized.casefold()
        if key and key not in seen:
            seen.add(key)
            candidates.append(normalized[:500])
    return candidates[:12]


def prepare_source_locked_expected(
    expected: dict[str, Any],
    manifest: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    """Degrade a failed legacy batch to only deterministically recoverable slots.

    This legacy staged lane has no durable assessment-obligation ledger.  It is
    safer to return the valid source-backed teaching content and record the
    omitted format in diagnostics than to fabricate an interaction or fail the
    complete batch.  Orchestration V2 removes unsupported assessments earlier
    and persists them as CP2B obligations instead of entering this path.
    """
    requested_types = [
        normalized
        for value in expected.get("component_types", [])
        for normalized in [normalize_staged_component_type(value)]
        if normalized
    ]
    fact_ids = {
        str(fact_id).strip()
        for fact_id in expected.get("source_fact_ids", [])
        if str(fact_id).strip()
    }
    facts = [
        fact
        for fact in (manifest or {}).get("facts", [])
        if isinstance(fact, dict) and str(fact.get("fact_id") or "").strip() in fact_ids
    ]
    fact_texts = clean_source_facts(
        [fact.get("text") for fact in facts],
        preserve_table_numeric=True,
    )
    locale = str(expected.get("locale") or "vi")
    title = str(expected.get("unit_title") or "Nội dung bài học").strip()
    recoverable = {"html"}
    if build_source_grounded_single_choice(title, fact_texts, locale=locale) is not None:
        recoverable.add("problem")
    ordered_items = _source_locked_sequence_items(facts)
    if len(ordered_items) >= 2:
        recoverable.add("la_diagram")
    if len(ordered_items) >= 3:
        recoverable.add("la_sortable")
    if len(source_clarification_signals(fact_texts)) >= 2:
        recoverable.add("la_faq")

    selected_types = [value for value in requested_types if value in recoverable]
    if "html" not in selected_types:
        selected_types.insert(0, "html")
    dropped_types = [value for value in requested_types if value not in recoverable]
    selected_type_set = set(selected_types)
    component_plan = [
        dict(plan)
        for plan in expected.get("component_plan", [])
        if isinstance(plan, dict)
        and normalize_staged_component_type(plan.get("type")) in selected_type_set
    ]
    return {
        **expected,
        "component_types": selected_types,
        "component_plan": component_plan,
    }, dropped_types


def _source_locked_html_facts(fact_texts: list[str]) -> list[str]:
    """Split raw source facts into displayable source lines without rewriting them."""
    lines: list[str] = []
    for fact_text in fact_texts:
        text = re.sub(r"\s+", " ", fact_text).strip()
        if not text:
            continue
        numbered_starts = list(re.finditer(r"(?<!\w)\d+[.)]\s+", text))
        if not numbered_starts:
            lines.append(text)
            continue
        prefix = text[:numbered_starts[0].start()].strip()
        if prefix:
            lines.append(prefix)
        for index, match in enumerate(numbered_starts):
            end = numbered_starts[index + 1].start() if index + 1 < len(numbered_starts) else len(text)
            item = text[match.start():end].strip()
            if item:
                lines.append(item)
    return lines


def _source_locked_html_inline(text: str) -> str:
    """Preserve a definition label as emphasis while keeping source wording intact."""
    match = re.match(r"^([^:]{2,100}):\s+(.+)$", text)
    if not match:
        return html.escape(text)
    label, value = match.groups()
    return f"<strong>{html.escape(label)}:</strong> {html.escape(value)}"


def _source_locked_html_body(title: str, fact_texts: list[str]) -> str:
    """Render raw facts as semantic lesson HTML rather than a flat paragraph dump."""
    lines = _source_locked_html_facts(fact_texts)
    parts = [f"<h3>{html.escape(title)}</h3>"]
    index = 0
    while index < len(lines):
        line = lines[index]
        ordered_match = re.match(r"^\d+[.)]\s+(.+)$", line)
        bullet_match = re.match(r"^[\u2022*-]\s+(.+)$", line)
        if ordered_match or bullet_match:
            tag = "ol" if ordered_match else "ul"
            items: list[str] = []
            while index < len(lines):
                match = (
                    re.match(r"^\d+[.)]\s+(.+)$", lines[index])
                    if tag == "ol"
                    else re.match(r"^[\u2022*-]\s+(.+)$", lines[index])
                )
                if not match:
                    break
                items.append(f"<li>{_source_locked_html_inline(match.group(1).strip())}</li>")
                index += 1
            parts.append(f"<{tag}>{''.join(items)}</{tag}>")
            continue

        is_heading = (
            len(line) <= 140
            and not re.search(r"[.!?;:]$", line)
            and not re.match(r"^[\u2022*-]\s+", line)
        )
        if is_heading:
            heading_lines = [line]
            index += 1
            while index < len(lines):
                candidate = lines[index]
                if (
                    len(candidate) > 80
                    or re.search(r"[.!?;:]$", candidate)
                    or re.match(r"^(?:\d+[.)]|[\u2022*-])\s+", candidate)
                ):
                    break
                heading_lines.append(candidate)
                index += 1
            parts.append(f"<h3>{html.escape(heading_lines[0])}</h3>")
            if len(heading_lines) > 1:
                parts.append(
                    "<ul>"
                    + "".join(f"<li>{_source_locked_html_inline(item)}</li>" for item in heading_lines[1:])
                    + "</ul>"
                )
            else:
                body_lines: list[str] = []
                while index < len(lines):
                    candidate = lines[index]
                    if (
                        len(candidate) > 180
                        or not re.search(r"[.!?;:]$", candidate)
                        or re.match(r"^(?:\d+[.)]|[\u2022*-])\s+", candidate)
                    ):
                        break
                    body_lines.append(candidate)
                    index += 1
                if len(body_lines) >= 2:
                    parts.append(
                        "<ul>"
                        + "".join(f"<li>{_source_locked_html_inline(item)}</li>" for item in body_lines)
                        + "</ul>"
                    )
                elif body_lines:
                    parts.append(f"<p>{_source_locked_html_inline(body_lines[0])}</p>")
            continue

        parts.append(f"<p>{_source_locked_html_inline(line)}</p>")
        index += 1
    return "".join(parts)


def _source_locked_faq_question(answer: str, title: str, index: int, locale: str) -> str:
    label = re.split(r"[:.!?]", answer, maxsplit=1)[0].strip()
    if len(label) >= 6 and len(label) <= 100:
        return (
            f"What should learners understand about {label}?"
            if locale == "en"
            else f"Người học cần hiểu gì về {label}?"
        )
    if locale == "en":
        return f"What should learners remember about '{title}'?" if index == 0 else f"Which condition needs attention in '{title}'?"
    return f"Người học cần ghi nhớ gì về '{title}'?" if index == 0 else f"Điều kiện nào cần lưu ý trong '{title}'?"


def _source_locked_faq_answers(fact_texts: list[str]) -> list[str]:
    """Use only explicit source conditions/exceptions for a fallback FAQ."""
    answers = source_clarification_signals(fact_texts)
    return [answer[:600] for answer in answers[:2]] if len(answers) >= 2 else []


def build_source_locked_unit(
    expected: dict[str, Any],
    manifest: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Build a safe unit from assigned raw facts after model retry.

    This never invents assessment answers or unsupported formats. HTML,
    source-answer checks, ordered items, and sequential diagrams can be
    reconstructed losslessly from raw facts.
    """
    # The legacy type-keyed fallback cannot reconstruct distinct assessment
    # obligations. Never fabricate an instance identity to make it acceptable.
    if any(plan.get("component_plan_id") for plan in expected.get("component_plan", []) if isinstance(plan, dict)):
        return None
    expected_types = [
        normalize_staged_component_type(value)
        for value in expected.get("component_types", [])
    ]
    supported_types = {"html", "problem", "la_faq", "la_diagram", "la_sortable"}
    if not expected_types or any(value not in supported_types for value in expected_types):
        return None
    fact_ids = [
        str(fact_id).strip()
        for fact_id in expected.get("source_fact_ids", [])
        if str(fact_id).strip()
    ]
    fact_id_set = set(fact_ids)
    facts = [
        fact
        for fact in (manifest or {}).get("facts", [])
        if isinstance(fact, dict) and str(fact.get("fact_id") or "").strip() in fact_id_set
    ]
    fact_texts = clean_source_facts(
        [fact.get("text") for fact in facts],
        preserve_table_numeric=True,
    )
    if not fact_texts:
        return None
    title = str(expected.get("unit_title") or "Nội dung bài học").strip()
    locale = str(expected.get("locale") or "vi")
    html_content = render_source_locked_html(title, fact_texts, locale=locale)
    visible_text = re.sub(r"\s+", " ", " ".join(fact_texts)).strip()
    if len(visible_text) < MIN_SOURCE_LOCKED_HTML_TEXT_CHARS:
        return None
    ordered_items = _source_locked_sequence_items(facts)
    if "la_sortable" in expected_types and len(ordered_items) < 3:
        return None
    if "la_diagram" in expected_types and len(ordered_items) < 2:
        return None

    components_by_type: dict[str, dict[str, Any]] = {
        "html": {
            "type": "html",
            "title": title,
            "html": html_content,
            "source_fact_ids": fact_ids,
            "source_locked_fallback": True,
            "selection_rationale": "Nội dung HTML được dựng trực tiếp từ các fact nguồn đã khóa sau khi bản sinh tự động không đạt ngưỡng kiểm tra.",
        },
    }
    if "problem" in expected_types:
        problem = build_source_grounded_single_choice(title, fact_texts, locale=locale)
        if problem is None:
            return None
        components_by_type["problem"] = {
            "type": "problem",
            "title": title,
            **problem,
            "source_fact_ids": fact_ids,
            "source_locked_fallback": True,
            "selection_rationale": (
                "The correct choice and explanation are locked to the assigned source evidence."
                if locale == "en"
                else "Đáp án đúng và phần giải thích được khóa theo dữ kiện nguồn đã gán."
            ),
        }
    if "la_faq" in expected_types:
        answer_candidates = _source_locked_faq_answers(fact_texts)
        if len(answer_candidates) < 2:
            return None
        components_by_type["la_faq"] = {
            "type": "la_faq",
            "title": "Frequently asked questions" if locale == "en" else "Câu hỏi thường gặp",
            "items": [
                {
                    "question": _source_locked_faq_question(answer, title, index, locale),
                    "answer": answer,
                }
                for index, answer in enumerate(answer_candidates)
            ],
            "source_fact_ids": fact_ids,
            "source_locked_fallback": True,
            "selection_rationale": (
                "The FAQ answers are reconstructed from the assigned source facts without adding unsupported claims."
                if locale == "en"
                else "Câu trả lời FAQ được dựng từ các fact nguồn đã gán, không thêm khẳng định ngoài tài liệu."
            ),
        }
    if "la_sortable" in expected_types:
        components_by_type["la_sortable"] = {
            "type": "la_sortable",
            "title": title,
            "question_text": "Sắp xếp các bước thực hiện theo đúng trình tự.",
            "items": ordered_items,
            "source_fact_ids": fact_ids,
            "source_locked_fallback": True,
            "selection_rationale": "Các bước được trích nguyên văn từ các fact nguồn theo thứ tự xuất hiện.",
        }
    if "la_diagram" in expected_types:
        components_by_type["la_diagram"] = {
            "type": "la_diagram",
            "title": title,
            "name": title,
            "nodes": [
                {
                    "label": item,
                    "shape": "ellipse" if index == 0 else "rounded",
                    "tooltip": item,
                }
                for index, item in enumerate(ordered_items[:8])
            ],
            "edges": [
                {"source": index, "target": index + 1, "label": "Tiếp theo"}
                for index in range(min(len(ordered_items[:8]) - 1, 7))
            ],
            "source_fact_ids": fact_ids,
            "source_locked_fallback": True,
            "selection_rationale": "Sơ đồ nối tuần tự các bước đã xuất hiện trong raw KB, không thêm quan hệ ngoài nguồn.",
        }

    planned_fact_ids_by_type = {
        str(plan.get("type") or "").strip(): [
            str(fact_id).strip()
            for fact_id in plan.get("source_fact_ids", [])
            if str(fact_id).strip() in fact_id_set
        ]
        for plan in expected.get("component_plan", [])
        if isinstance(plan, dict)
    }
    for component_type, component in components_by_type.items():
        component_fact_ids = planned_fact_ids_by_type.get(component_type) or fact_ids
        component["source_fact_ids"] = component_fact_ids
        component["covered_source_fact_ids"] = component_fact_ids

    return {
        "title": title,
        "source_fact_ids": fact_ids,
        "component_plan": expected.get("component_plan", []),
        "components": [components_by_type[component_type] for component_type in expected_types],
        "source_locked_fallback": True,
    }


build_source_locked_html_unit = build_source_locked_unit
