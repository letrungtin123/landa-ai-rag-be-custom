"""W1 SME Content Map and W1-reduce + W0 (spec §7.2, §7.3)."""

from __future__ import annotations

import asyncio
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal

from app.idm.contracts import (
    IdmAudienceV1,
    IdmBlockLoLinkV1,
    IdmContentBlockV1,
    IdmIssueV1,
    IdmLearningObjectiveV1,
    IdmMustDoV1,
    IdmNoiseDraftV1,
    IdmProjectContextV1,
    IdmW1BlockDraftV1,
    IdmW1ReduceResponseV1,
    IdmW1SectionResponseV1,
)
from app.idm.policy import (
    BROAD_SME_QUESTION_PATTERN,
    IDM_BLOCK_TOO_SMALL_CHARS,
    IDM_FALLBACK_NAME_CHARS,
    IDM_FALLBACK_SUMMARY_CHARS,
    IDM_LO_MAX_COUNT,
    IDM_LO_MIN_COUNT,
    IDM_PAGE_FURNITURE_MIN_REPEATS,
    IDM_PROCEDURE_SHARE,
    IDM_SME_QUESTION_MIN_CHARS,
    IDM_TABLE_SHARE,
    IDM_W1_MAP_MAX_OUTPUT_TOKENS,
    IDM_W1_REDUCE_MAX_OUTPUT_TOKENS,
    MUST_DO_TOPIC_PREFIXES,
    THINKING_W1_MAP,
    THINKING_W1_REDUCE,
)
from app.idm.prompts import repair_suffix, w1_reduce_prompt, w1_section_prompt
from app.idm.runtime import (
    IdmBudgetError,
    IdmProviderError,
    IdmResponseInvalidError,
    IdmRuntime,
    idm_call,
    record_deterministic_fallback,
)
from app.idm.signals import IdmSection
from app.idm.text import idm_fold, is_generic_title, is_unmeasurable_objective, single_line
from app.idm.validation import IdmIssue, errors
from app.lesson_author_orchestration_v2_provider import SourceSnapshotFactV2

_BROAD_SME_RE: Final = re.compile(BROAD_SME_QUESTION_PATTERN)
_PAGE_NUMBER_RE: Final = re.compile(
    r"^(?:trang|page|tr\.?)\s*\d+(?:\s*(?:/|of|tren)\s*\d+)?$|^-?\s*\d+\s*-?$",
)
_HEADING_NUMBER_RE: Final = re.compile(r"^\s*(?:[0-9]+(?:\.[0-9]+)*\.?|[ivxlc]+\.)\s+", re.IGNORECASE)
_MAX_BLOCK_ISSUES: Final = 16
_MAX_BLOCK_SME: Final = 8
_MAX_REDUCE_LOS: Final = IDM_LO_MAX_COUNT
_MIN_NAME_CHARS: Final = 3
# Headers, footers and page numbers are short; long repeated lines are content.
_FURNITURE_MAX_CHARS: Final = 100
_TOP_NUMBERED_RE: Final = re.compile(r"^\d{1,2}[.)]?\s+\S")

StageOriginLiteral = Literal["provider", "deterministic_fallback"]


@dataclass(frozen=True)
class FactIndex:
    """Document order, text and signal flags of every snapshot fact."""

    order: dict[str, int]
    text: dict[str, str]
    flags: dict[str, list[str]]
    document: dict[str, str]

    @classmethod
    def build(cls, facts: Sequence[SourceSnapshotFactV2], signals: dict[str, list[str]]) -> FactIndex:
        return cls(
            order={fact.fact_key: position for position, fact in enumerate(facts)},
            text={fact.fact_key: fact.fact_text for fact in facts},
            flags={fact.fact_key: list(signals.get(fact.fact_key, [])) for fact in facts},
            document={fact.fact_key: fact.document_id for fact in facts},
        )

    def sort(self, keys: Sequence[str]) -> list[str]:
        return sorted(dict.fromkeys(keys), key=lambda key: self.order[key])


@dataclass
class SectionMapResult:
    section: IdmSection
    blocks: list[IdmW1BlockDraftV1]
    noise: list[IdmNoiseDraftV1]
    origin: StageOriginLiteral
    issue_codes: dict[str, int] = field(default_factory=dict)
    warning_codes: list[str] = field(default_factory=list)


def page_furniture_keys(facts: Sequence[SourceSnapshotFactV2]) -> set[str]:
    """Facts that are page furniture: page numbers or a footer repeated on >= 3 pages."""

    def shape(text: str) -> str:
        return re.sub(r"\d+", "#", idm_fold(text))

    counts = Counter(shape(fact.fact_text) for fact in facts)
    result: set[str] = set()
    for fact in facts:
        if len(fact.fact_text) > _FURNITURE_MAX_CHARS:
            continue
        folded = idm_fold(fact.fact_text)
        if _PAGE_NUMBER_RE.match(folded) or (
                "#" in shape(fact.fact_text) and counts[shape(fact.fact_text)] >= IDM_PAGE_FURNITURE_MIN_REPEATS):
            result.add(fact.fact_key)
    return result


# --- W1-map ----------------------------------------------------------------------------------
def validate_w1_section(response: IdmW1SectionResponseV1, section_keys: Sequence[str]) -> list[IdmIssue]:
    issues: list[IdmIssue] = []
    allowed = set(section_keys)
    seen: Counter[str] = Counter()
    local_ids = Counter(block.local_id for block in response.blocks)
    for index, block in enumerate(response.blocks):
        path = f"blocks[{index}]"
        if local_ids[block.local_id] > 1:
            issues.append(IdmIssue("IDM_W1_LOCAL_ID_DUPLICATED", f"{path}.local_id"))
        if is_generic_title(block.name):
            issues.append(IdmIssue("IDM_W1_BLOCK_NAME_GENERIC", f"{path}.name"))
        for key in block.fact_keys:
            seen[key] += 1
            if key not in allowed:
                issues.append(IdmIssue("IDM_W1_FACT_UNKNOWN", f"{path}.fact_keys"))
        if re.search(r"\s(và|and)\s", block.name, flags=re.IGNORECASE):
            issues.append(IdmIssue("IDM_W1_BLOCK_COMPOUND_NAME", f"{path}.name", "warning"))
        for question_index, question in enumerate(block.sme_questions):
            if len(question) < IDM_SME_QUESTION_MIN_CHARS or _BROAD_SME_RE.match(idm_fold(question)):
                issues.append(IdmIssue("IDM_W1_SME_QUESTION_TOO_BROAD",
                                       f"{path}.sme_questions[{question_index}]", "warning"))
    for index, item in enumerate(response.noise):
        seen[item.fact_key] += 1
        if item.fact_key not in allowed:
            issues.append(IdmIssue("IDM_W1_FACT_UNKNOWN", f"noise[{index}].fact_key"))
    issues.extend(IdmIssue("IDM_W1_FACT_DUPLICATED", "blocks") for key, count in seen.items() if count > 1
                  and key in allowed)
    if any(key not in seen for key in section_keys):
        issues.append(IdmIssue("IDM_W1_FACT_UNASSIGNED", "blocks"))
    return _dedupe(issues)


def _dedupe(issues: list[IdmIssue]) -> list[IdmIssue]:
    return list(dict.fromkeys(issues))


def small_block_warnings(blocks: Sequence[IdmW1BlockDraftV1], index: FactIndex) -> list[str]:
    return ["IDM_W1_BLOCK_TOO_SMALL" for block in blocks
            if len(block.fact_keys) == 1 and block.support_role is None
            and len(index.text.get(block.fact_keys[0], "")) < IDM_BLOCK_TOO_SMALL_CHARS]


def normalize_w1_section(
    response: IdmW1SectionResponseV1, section_keys: Sequence[str], index: FactIndex,
) -> tuple[list[IdmW1BlockDraftV1], list[IdmNoiseDraftV1]]:
    """Order facts by the document and sanitise author text; never changes ownership."""

    allowed = set(section_keys)
    blocks = []
    for block in response.blocks:
        blocks.append(block.model_copy(update={
            "name": single_line(block.name, 180),
            "summary": single_line(block.summary, 300),
            "fact_keys": index.sort(block.fact_keys),
            "issues": [issue.model_copy(update={
                "note": single_line(issue.note, 500),
                "fact_keys": [key for key in issue.fact_keys if key in allowed],
            }) for issue in block.issues],
            "gaps": [gap.model_copy(update={"note": single_line(gap.note, 500)}) for gap in block.gaps],
            "sme_questions": [single_line(question, 400) for question in block.sme_questions],
        }))
    blocks.sort(key=lambda block: index.order[block.fact_keys[0]])
    return blocks, list(response.noise)


def _clean_heading(text: str) -> str:
    stripped = _HEADING_NUMBER_RE.sub("", " ".join(text.split())).strip(" .:-")
    letters = [char for char in stripped if char.isalpha()]
    if letters and all(char.isupper() for char in letters):
        stripped = stripped[:1].upper() + stripped[1:].lower()
    return stripped or " ".join(text.split())


def fallback_w1_section(
    section: IdmSection, index: FactIndex, furniture: set[str],
) -> tuple[list[IdmW1BlockDraftV1], list[IdmNoiseDraftV1]]:
    """One block per heading run; page furniture becomes noise (spec §7.2)."""

    noise = [IdmNoiseDraftV1(fact_key=key, reason="page_furniture") for key in section.fact_keys if key in furniture]
    runs: list[tuple[str | None, list[str]]] = []
    for key in section.fact_keys:
        if key in furniture:
            continue
        if "H" in index.flags.get(key, []) or not runs:
            heading = index.text[key] if "H" in index.flags.get(key, []) else None
            runs.append((heading, [key]))
        else:
            runs[-1][1].append(key)
    blocks: list[IdmW1BlockDraftV1] = []
    for position, (heading, keys) in enumerate(runs, start=1):
        texts = [" ".join(index.text[key].split()) for key in keys]
        raw_name = _clean_heading(heading) if heading else (texts[0][:IDM_FALLBACK_NAME_CHARS] or section.title_path)
        if len(raw_name) < _MIN_NAME_CHARS:
            raw_name = f"{raw_name} {section.title_path}".strip()
        name = single_line(raw_name, 180).ljust(_MIN_NAME_CHARS, ".")
        flags = [index.flags.get(key, []) for key in keys]
        steps = sum("S" in item for item in flags) / len(keys)
        tables = sum("T" in item for item in flags) / len(keys)
        kind: Literal["procedure", "reference", "concept"] = (
            "procedure" if steps >= IDM_PROCEDURE_SHARE else "reference" if tables >= IDM_TABLE_SHARE else "concept")
        blocks.append(IdmW1BlockDraftV1(
            local_id=f"b{position}",
            name=name,
            summary=single_line(" ".join(texts), IDM_FALLBACK_SUMMARY_CHARS) or name,
            intent="know", support_role=None, content_kind=kind, fact_keys=keys,
            issues=[], gaps=[], sme_questions=[],
        ))
    if not blocks:
        # Only furniture in this section: nothing to teach, everything is accounted as noise.
        return [], noise
    return blocks, noise


async def map_section(
    runtime: IdmRuntime,
    section: IdmSection,
    index: FactIndex,
    project_context: dict[str, Any],
    furniture: set[str],
    *,
    tail_reserve_tokens: int,
) -> SectionMapResult:
    facts = [(key, index.text[key], index.flags.get(key, [])) for key in section.fact_keys]
    prompt = w1_section_prompt(runtime.locale, project_context=project_context, section_id=section.section_id,
                               section_title_path=section.title_path, facts=facts)
    issue_codes: Counter[str] = Counter()
    repair = ""
    for attempt in (1, 2):
        try:
            response = await idm_call(
                runtime, stage="idm_w1_map", prompt=prompt + repair, response_model=IdmW1SectionResponseV1,
                max_output_tokens=IDM_W1_MAP_MAX_OUTPUT_TOKENS, thinking_level=THINKING_W1_MAP,
                invocation_kind="writer" if attempt == 1 else "repair", reserve_after_tokens=tail_reserve_tokens,
            )
        except (IdmBudgetError, IdmResponseInvalidError) as error:
            issue_codes[error.code] += 1
            if isinstance(error, IdmBudgetError):
                break
            repair = repair_suffix([{"code": item["type"], "path": ".".join(item["loc"])} for item in error.errors])
            continue
        except IdmProviderError as error:
            if error.terminal:
                raise
            issue_codes[error.code] += 1
            break
        found = validate_w1_section(response, section.fact_keys)
        blocking = errors(found)
        issue_codes.update(issue.code for issue in found)
        if not blocking:
            blocks, noise = normalize_w1_section(response, section.fact_keys, index)
            warnings = [issue.code for issue in found if issue.severity == "warning"]
            warnings.extend(small_block_warnings(blocks, index))
            return SectionMapResult(section, blocks, noise, "provider", dict(issue_codes), warnings)
        repair = repair_suffix([issue.as_repair_item() for issue in blocking])
    record_deterministic_fallback(runtime, stage="idm_w1_map", code="IDM_W1_SECTION_FALLBACK")
    blocks, noise = fallback_w1_section(section, index, furniture)
    return SectionMapResult(section, blocks, noise, "deterministic_fallback", dict(issue_codes), [])


async def map_sections(
    runtime: IdmRuntime,
    sections: Sequence[IdmSection],
    index: FactIndex,
    project_context: dict[str, Any],
    furniture: set[str],
    *,
    parallelism: int,
    tail_reserve_tokens: int,
) -> list[SectionMapResult]:
    gate = asyncio.Semaphore(max(1, parallelism))

    async def run(section: IdmSection) -> SectionMapResult:
        async with gate:
            return await map_section(runtime, section, index, project_context, furniture,
                                     tail_reserve_tokens=tail_reserve_tokens)

    tasks = [asyncio.create_task(run(section)) for section in sections]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def rekey_blocks(results: Sequence[SectionMapResult], index: FactIndex) -> tuple[
        list[IdmContentBlockV1], list[IdmNoiseDraftV1]]:
    """Give blocks document-order ids ``cb_0001``… (by their first fact)."""

    drafts = [(result, block) for result in results for block in result.blocks]
    drafts.sort(key=lambda item: index.order[item[1].fact_keys[0]])
    blocks = [IdmContentBlockV1(
        block_id=f"cb_{position:04d}", section_id=result.section.section_id, name=block.name,
        summary=block.summary, intent=block.intent, support_role=block.support_role,
        content_kind=block.content_kind, fact_keys=list(block.fact_keys), issues=list(block.issues),
        gaps=list(block.gaps), sme_questions=list(block.sme_questions),
        origin="provider" if result.origin == "provider" else "deterministic_fallback",
    ) for position, (result, block) in enumerate(drafts, start=1)]
    noise = [item for result in results for item in result.noise]
    return blocks, noise


# --- W1-reduce + W0 --------------------------------------------------------------------------
def reduce_catalog(blocks: Sequence[IdmContentBlockV1]) -> list[dict[str, Any]]:
    return [{
        "block_id": block.block_id, "section_id": block.section_id, "name": block.name, "intent": block.intent,
        "support_role": block.support_role, "content_kind": block.content_kind, "fact_count": len(block.fact_keys),
        "summary": block.summary,
        "issues": [{"type": issue.type, "note": issue.note} for issue in block.issues],
        "gaps": [{"type": gap.type, "note": gap.note} for gap in block.gaps],
    } for block in blocks]


def validate_w1_reduce(
    response: IdmW1ReduceResponseV1, blocks: Sequence[IdmContentBlockV1], context: IdmProjectContextV1,
) -> list[IdmIssue]:
    issues: list[IdmIssue] = []
    block_ids = {block.block_id for block in blocks}
    lo_ids = [objective.lo_id for objective in response.learning_objectives]
    client = context.learning_objectives
    if client:
        if len(response.learning_objectives) != len(client):
            issues.append(IdmIssue("IDM_W0_LO_COUNT", "learning_objectives"))
    elif not IDM_LO_MIN_COUNT <= len(lo_ids) <= IDM_LO_MAX_COUNT:
        issues.append(IdmIssue("IDM_W0_LO_COUNT", "learning_objectives"))
    if lo_ids != [f"lo_{position}" for position in range(1, len(lo_ids) + 1)]:
        issues.append(IdmIssue("IDM_W0_LO_ID_SEQUENCE", "learning_objectives"))
    if not client:
        issues.extend(IdmIssue("IDM_W0_LO_UNMEASURABLE", f"learning_objectives[{position}].statement")
                      for position, objective in enumerate(response.learning_objectives)
                      if is_unmeasurable_objective(objective.statement))
    known_los = set(lo_ids)
    must_do_ids = Counter(item.must_do_id for item in response.must_dos)
    for position, must_do in enumerate(response.must_dos):
        path = f"must_dos[{position}]"
        if must_do.lo_id not in known_los:
            issues.append(IdmIssue("IDM_W0_MUST_DO_ORPHAN", f"{path}.lo_id"))
        if must_do_ids[must_do.must_do_id] > 1:
            issues.append(IdmIssue("IDM_W0_MUST_DO_ID_DUPLICATED", f"{path}.must_do_id"))
        if idm_fold(must_do.statement).startswith(MUST_DO_TOPIC_PREFIXES):
            issues.append(IdmIssue("IDM_W0_MUST_DO_IS_TOPIC", f"{path}.statement"))
    covered = {must_do.lo_id for must_do in response.must_dos}
    issues.extend(IdmIssue("IDM_W0_LO_WITHOUT_MUST_DO", f"learning_objectives[{position}]")
                  for position, lo_id in enumerate(lo_ids) if lo_id not in covered)
    merged_once: Counter[str] = Counter()
    for position, merge in enumerate(response.merges):
        members = [merge.keep_block_id, *merge.merged_block_ids]
        merged_once.update(members)
        if (any(member not in block_ids for member in members)
                or merge.keep_block_id in merge.merged_block_ids
                or len(set(merge.merged_block_ids)) != len(merge.merged_block_ids)):
            issues.append(IdmIssue("IDM_W1_MERGE_INVALID", f"merges[{position}]"))
    if any(count > 1 for count in merged_once.values()):
        issues.append(IdmIssue("IDM_W1_MERGE_INVALID", "merges"))
    for position, conflict in enumerate(response.conflicts):
        if any(block_id not in block_ids for block_id in conflict.block_ids):
            issues.append(IdmIssue("IDM_W1_LINK_INVALID", f"conflicts[{position}]"))
    for position, link in enumerate(response.lo_links):
        if link.block_id not in block_ids or link.lo_id not in known_los:
            issues.append(IdmIssue("IDM_W1_LINK_INVALID", f"lo_links[{position}]"))
    return _dedupe(issues)


def _top_heading_groups(
    blocks: Sequence[IdmContentBlockV1], sections: Sequence[IdmSection], index: FactIndex | None,
) -> list[tuple[str, list[IdmContentBlockV1]]]:
    """Group blocks under the top-level heading that precedes them (spec §7.3 fallback).

    Top-level headings are single-number headings ("1. …") when the source has
    them, otherwise every heading. Without a fact index, sections stand in.
    """

    if index is not None:
        headings = [key for key in sorted(index.order, key=index.order.__getitem__) if "H" in index.flags.get(key, [])]
        numbered = [key for key in headings if _TOP_NUMBERED_RE.match(index.text[key].strip())]
        tops = numbered or headings
        if tops:
            groups: dict[str, list[IdmContentBlockV1]] = {}
            for block in blocks:
                first = index.order[block.fact_keys[0]]
                owner = next((key for key in reversed(tops) if index.order[key] <= first), tops[0])
                groups.setdefault(owner, []).append(block)
            return [(index.text[key], groups[key]) for key in tops if key in groups]
    by_section: dict[str, list[IdmContentBlockV1]] = {}
    for block in blocks:
        by_section.setdefault(block.section_id, []).append(block)
    return [(section.title_path, by_section[section.section_id]) for section in sections
            if section.section_id in by_section]


def _merge_groups(
    groups: list[tuple[str, list[IdmContentBlockV1]]], count: int,
) -> list[tuple[str, list[IdmContentBlockV1]]]:
    """Merge consecutive groups down to ``count`` (each keeps its first heading)."""

    if len(groups) <= count:
        return groups
    merged: list[tuple[str, list[IdmContentBlockV1]]] = []
    for position, (title, items) in enumerate(groups):
        target = position * count // len(groups)
        if target == len(merged):
            merged.append((title, list(items)))
        else:
            merged[target][1].extend(items)
    return merged


def fallback_w1_reduce(
    blocks: Sequence[IdmContentBlockV1],
    sections: Sequence[IdmSection],
    context: IdmProjectContextV1,
    index: FactIndex | None = None,
) -> IdmW1ReduceResponseV1:
    """One objective per top-level heading; client objectives stay authoritative (spec §7.3)."""

    vi = context.locale == "vi"
    client = list(context.learning_objectives)
    groups = _merge_groups(_top_heading_groups(blocks, sections, index), len(client) or _MAX_REDUCE_LOS)
    objectives: list[IdmLearningObjectiveV1] = []
    must_dos: list[IdmMustDoV1] = []
    if client:
        for position, statement in enumerate(client, start=1):
            objectives.append(IdmLearningObjectiveV1(lo_id=f"lo_{position}", statement=statement.ljust(10, "."),
                                                     bloom="apply", origin="client"))
            must_dos.append(IdmMustDoV1(must_do_id=f"md_{position}", lo_id=f"lo_{position}",
                                        statement=single_line(statement, 280).ljust(5, "."), kind="do",
                                        bloom="apply"))
    else:
        for position, (title, _items) in enumerate(groups, start=1):
            heading = _clean_heading(title)[:200]
            objectives.append(IdmLearningObjectiveV1(
                lo_id=f"lo_{position}", bloom="apply", origin="ai_proposed",
                statement=single_line((f"Người học có thể áp dụng {heading} trong công việc" if vi
                                       else f"The learner can apply {heading} at work"), 500)))
            must_dos.append(IdmMustDoV1(
                must_do_id=f"md_{position}", lo_id=f"lo_{position}", kind="do", bloom="apply",
                statement=single_line((f"Thực hiện đúng {heading}" if vi else f"Correctly carry out {heading}"), 280)))
    links = [IdmBlockLoLinkV1(block_id=block.block_id, relation="direct",
                              lo_id=objectives[position * len(objectives) // max(1, len(groups))].lo_id)
             for position, (_title, items) in enumerate(groups) for block in items]
    if context.target_audience:
        audience = IdmAudienceV1(description=context.target_audience, origin="client")
    else:
        topic = context.course_title_hint or context.source_documents[0].name
        audience = IdmAudienceV1(
            description=single_line((f"Nhân sự cần áp dụng nội dung {topic} trong công việc" if vi
                                     else f"Staff who need to apply {topic} in their work"), 2000),
            origin="ai_proposed")
    return IdmW1ReduceResponseV1(merges=[], conflicts=[], target_audience=audience,
                                 learning_objectives=objectives, must_dos=must_dos, lo_links=links)


def apply_client_context(response: IdmW1ReduceResponseV1, context: IdmProjectContextV1) -> IdmW1ReduceResponseV1:
    """Client-provided audience/objectives are authoritative and copied verbatim."""

    update: dict[str, Any] = {}
    if context.target_audience:
        update["target_audience"] = IdmAudienceV1(description=context.target_audience, origin="client")
    else:
        update["target_audience"] = response.target_audience.model_copy(update={
            "origin": "ai_proposed", "description": single_line(response.target_audience.description, 2000)})
    if context.learning_objectives and len(context.learning_objectives) == len(response.learning_objectives):
        update["learning_objectives"] = [objective.model_copy(update={"statement": statement, "origin": "client"})
                                         for objective, statement in zip(response.learning_objectives,
                                                                         context.learning_objectives, strict=True)]
    else:
        update["learning_objectives"] = [objective.model_copy(update={
            "origin": "ai_proposed", "statement": single_line(objective.statement, 500)})
            for objective in response.learning_objectives]
    update["must_dos"] = [must_do.model_copy(update={"statement": single_line(must_do.statement, 280)})
                          for must_do in response.must_dos]
    return response.model_copy(update=update)


def apply_w1_reduce(
    response: IdmW1ReduceResponseV1, blocks: Sequence[IdmContentBlockV1], index: FactIndex, locale: str,
) -> tuple[list[IdmContentBlockV1], list[IdmBlockLoLinkV1]]:
    """Apply merges (union of facts, document order) and attach conflicts as issues."""

    by_id = {block.block_id: block for block in blocks}
    merged_into: dict[str, str] = {}
    for merge in response.merges:
        keep = by_id[merge.keep_block_id]
        names = [by_id[block_id].name for block_id in merge.merged_block_ids]
        facts = index.sort([*keep.fact_keys, *(key for block_id in merge.merged_block_ids
                                              for key in by_id[block_id].fact_keys)])
        note = single_line((f"Đã gộp nội dung trùng từ: {'; '.join(names)}. {merge.reason}" if locale == "vi"
                            else f"Merged duplicate content from: {'; '.join(names)}. {merge.reason}"), 500)
        by_id[keep.block_id] = keep.model_copy(update={
            "fact_keys": facts,
            "issues": [*keep.issues, IdmIssueV1(type="duplicate", note=note, fact_keys=[])][:_MAX_BLOCK_ISSUES],
        })
        for block_id in merge.merged_block_ids:
            merged_into[block_id] = keep.block_id
            del by_id[block_id]
    for conflict in response.conflicts:
        for block_id in dict.fromkeys(merged_into.get(item, item) for item in conflict.block_ids):
            block = by_id[block_id]
            by_id[block_id] = block.model_copy(update={
                "issues": [*block.issues, IdmIssueV1(type="conflict", note=single_line(conflict.note, 500),
                                                     fact_keys=[])][:_MAX_BLOCK_ISSUES],
                "sme_questions": list(dict.fromkeys(
                    [*block.sme_questions, single_line(conflict.sme_question, 400)]))[:_MAX_BLOCK_SME],
            })
    links: dict[tuple[str, str], IdmBlockLoLinkV1] = {}
    rank = {"direct": 0, "supporting": 1, "context": 2, "unknown": 3, "unrelated": 4}
    for link in response.lo_links:
        block_id = merged_into.get(link.block_id, link.block_id)
        key = (block_id, link.lo_id)
        candidate = link.model_copy(update={"block_id": block_id})
        if key not in links or rank[candidate.relation] < rank[links[key].relation]:
            links[key] = candidate
    ordered = sorted(by_id.values(), key=lambda block: index.order[block.fact_keys[0]])
    return ordered, sorted(links.values(), key=lambda link: (link.block_id, link.lo_id))


async def run_w1_reduce(
    runtime: IdmRuntime,
    blocks: Sequence[IdmContentBlockV1],
    sections: Sequence[IdmSection],
    context: IdmProjectContextV1,
    *,
    tail_reserve_tokens: int,
    index: FactIndex | None = None,
) -> tuple[IdmW1ReduceResponseV1, StageOriginLiteral, dict[str, int]]:
    prompt = w1_reduce_prompt(runtime.locale, project_context=context.model_dump(mode="json"),
                              catalog=reduce_catalog(blocks))
    codes: Counter[str] = Counter()
    repair = ""
    for attempt in (1, 2):
        try:
            response = await idm_call(
                runtime, stage="idm_w1_reduce", prompt=prompt + repair, response_model=IdmW1ReduceResponseV1,
                max_output_tokens=IDM_W1_REDUCE_MAX_OUTPUT_TOKENS, thinking_level=THINKING_W1_REDUCE,
                invocation_kind="writer" if attempt == 1 else "repair", reserve_after_tokens=tail_reserve_tokens,
            )
        except (IdmBudgetError, IdmResponseInvalidError) as error:
            codes[error.code] += 1
            if isinstance(error, IdmBudgetError):
                break
            repair = repair_suffix([{"code": item["type"], "path": ".".join(item["loc"])} for item in error.errors])
            continue
        except IdmProviderError as error:
            if error.terminal:
                raise
            codes[error.code] += 1
            break
        found = validate_w1_reduce(response, blocks, context)
        codes.update(issue.code for issue in found)
        if not errors(found):
            return apply_client_context(response, context), "provider", dict(codes)
        repair = repair_suffix([issue.as_repair_item() for issue in errors(found)])
    record_deterministic_fallback(runtime, stage="idm_w1_reduce", code="IDM_W1_REDUCE_FALLBACK")
    return fallback_w1_reduce(blocks, sections, context, index), "deterministic_fallback", dict(codes)
