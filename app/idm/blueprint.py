"""W2 Content Blueprint (spec §7.4): per-block decisions, combine/separate, dispositions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

from app.idm.content_map import FactIndex
from app.idm.contracts import (
    Classification,
    IdmBlockLoLinkV1,
    IdmBlueprintRowV1,
    IdmContentBlockV1,
    IdmDispositionV1,
    IdmHoldItemV1,
    IdmLearningObjectiveV1,
    IdmMustDoV1,
    IdmNoiseDraftV1,
    IdmProjectContextV1,
    IdmW2BlueprintResponseV1,
    Placement,
    StageOrigin,
    Treatment,
)
from app.idm.policy import IDM_W2_MAX_OUTPUT_TOKENS, IDM_W2_SINGLE_CALL_MAX_BLOCKS, THINKING_W2
from app.idm.prompts import COMPACT_W2, answer_repair, repair_suffix, w2_blueprint_prompt
from app.idm.runtime import (
    IdmBudgetError,
    IdmProviderError,
    IdmResponseInvalidError,
    IdmRuntime,
    ThinkingLevel,
    idm_call,
    record_deterministic_fallback,
    repair_thinking,
)
from app.idm.text import single_line
from app.idm.validation import IdmIssue, errors

PLACEMENT_BY_CLASSIFICATION: Final[dict[str, Placement]] = {
    "must_do": "course", "must_know": "course", "reference": "reference_job_aid",
    "nice_to_know": "excluded", "remove": "excluded",
}
TREATMENTS_BY_CLASSIFICATION: Final = {
    "must_do": {"keep", "condense", "rewrite", "combine", "separate"},
    "must_know": {"keep", "condense", "rewrite", "combine", "separate"},
    "reference": {"move_to_reference", "convert_to_job_aid"},
    "nice_to_know": {"remove"},
    "remove": {"remove"},
}
COURSE_CLASSIFICATIONS: Final = frozenset({"must_do", "must_know"})
FALLBACK_TREATMENT: Final[dict[str, Treatment]] = {
    "must_do": "condense", "must_know": "condense", "reference": "convert_to_job_aid",
    "nice_to_know": "remove", "remove": "remove",
}
_MIN_SEPARATE_PARTS: Final = 2
_MAX_BLOCK_ISSUES: Final = 16
_MAX_BLOCK_SME: Final = 8
_MAX_ROW_MUST_DOS: Final = 8
_RELATION_RANK: Final = {"direct": 0, "supporting": 1, "context": 2, "unknown": 3, "unrelated": 4}


@dataclass
class BlueprintResult:
    blocks: list[IdmContentBlockV1]
    links: list[IdmBlockLoLinkV1]
    rows: list[IdmBlueprintRowV1]
    origin: StageOrigin
    codes: dict[str, int]


def blueprint_catalog(blocks: Sequence[IdmContentBlockV1], links: Sequence[IdmBlockLoLinkV1]) -> list[dict[str, Any]]:
    by_block: dict[str, list[dict[str, str]]] = {}
    for link in links:
        by_block.setdefault(link.block_id, []).append({"lo_id": link.lo_id, "relation": link.relation})
    return [{
        "block_id": block.block_id, "name": block.name, "intent": block.intent, "support_role": block.support_role,
        "content_kind": block.content_kind, "summary": block.summary, "fact_keys": list(block.fact_keys),
        "lo_links": by_block.get(block.block_id, []),
        "issues": [{"type": issue.type, "note": issue.note} for issue in block.issues],
        "gaps": [{"type": gap.type, "note": gap.note} for gap in block.gaps],
    } for block in blocks]


def validate_w2(
    response: IdmW2BlueprintResponseV1,
    blocks: Sequence[IdmContentBlockV1],
    objectives: Sequence[IdmLearningObjectiveV1],
    must_dos: Sequence[IdmMustDoV1],
) -> list[IdmIssue]:
    issues: list[IdmIssue] = []
    by_block = {block.block_id: block for block in blocks}
    lo_ids = {objective.lo_id for objective in objectives}
    md_ids = {must_do.must_do_id for must_do in must_dos}
    counts = Counter(row.block_id for row in response.rows)
    rows = {row.block_id: row for row in response.rows}
    for block_id in by_block:
        if counts[block_id] == 0:
            issues.append(IdmIssue("IDM_W2_ROW_MISSING", f"rows[{block_id}]"))
    for position, row in enumerate(response.rows):
        path = f"rows[{position}]"
        if row.block_id not in by_block:
            issues.append(IdmIssue("IDM_W2_ROW_UNKNOWN", f"{path}.block_id"))
            continue
        if counts[row.block_id] > 1:
            issues.append(IdmIssue("IDM_W2_ROW_DUPLICATED", f"{path}.block_id"))
        if (PLACEMENT_BY_CLASSIFICATION[row.classification] != row.placement
                or row.treatment not in TREATMENTS_BY_CLASSIFICATION[row.classification]
                or (row.lo_id is not None and row.lo_id not in lo_ids)
                or any(md not in md_ids for md in row.must_do_ids)
                or (row.classification == "must_do" and not row.must_do_ids)):
            issues.append(IdmIssue("IDM_W2_INCONSISTENT_ROW", path))
        if row.classification == "must_know" and not row.must_do_ids:
            issues.append(IdmIssue("IDM_W2_MUST_KNOW_WITHOUT_MUST_DO", f"{path}.must_do_ids"))
        if row.hold and (not row.hold_reason or not row.sme_question
                         or row.classification not in COURSE_CLASSIFICATIONS):
            issues.append(IdmIssue("IDM_W2_HOLD_INVALID", path))
        if (row.combine_into is not None) != (row.treatment == "combine"):
            issues.append(IdmIssue("IDM_W2_COMBINE_INVALID", path))
        elif row.combine_into is not None:
            target = rows.get(row.combine_into)
            if (target is None or row.combine_into == row.block_id or target.classification != row.classification
                    or target.combine_into is not None or target.treatment == "separate"
                    or target.hold != row.hold):
                issues.append(IdmIssue("IDM_W2_COMBINE_INVALID", f"{path}.combine_into"))
        if bool(row.separate_into) != (row.treatment == "separate"):
            issues.append(IdmIssue("IDM_W2_SEPARATE_PARTITION", path))
        elif row.separate_into:
            parts = [key for part in row.separate_into for key in part.fact_keys]
            if (len(row.separate_into) < _MIN_SEPARATE_PARTS or len(parts) != len(set(parts))
                    or set(parts) != set(by_block[row.block_id].fact_keys)):
                issues.append(IdmIssue("IDM_W2_SEPARATE_PARTITION", f"{path}.separate_into"))
    return list(dict.fromkeys(issues))


def _normalize_row(row: IdmBlueprintRowV1) -> IdmBlueprintRowV1:
    return row.model_copy(update={
        "detail_level": single_line(row.detail_level, 300),
        "rationale": single_line(row.rationale, 300),
        "hold_reason": single_line(row.hold_reason, 300) if row.hold_reason else None,
        "sme_question": single_line(row.sme_question, 400) if row.sme_question else None,
        "must_do_ids": list(dict.fromkeys(row.must_do_ids)),
    })


def fallback_w2(
    blocks: Sequence[IdmContentBlockV1],
    links: Sequence[IdmBlockLoLinkV1],
    must_dos: Sequence[IdmMustDoV1],
    locale: str,
) -> list[IdmBlueprintRowV1]:
    """Directly linked blocks become Must Know; the rest Nice to Know (spec §7.4)."""

    vi = locale == "vi"
    must_dos_by_lo: dict[str, list[str]] = {}
    for must_do in must_dos:
        must_dos_by_lo.setdefault(must_do.lo_id, []).append(must_do.must_do_id)
    direct: dict[str, str] = {}
    for link in sorted(links, key=lambda item: _RELATION_RANK[item.relation]):
        if link.relation == "direct" and must_dos_by_lo.get(link.lo_id):
            direct.setdefault(link.block_id, link.lo_id)
    if not direct and must_dos:
        # The links carry no usable signal: keep every block rather than an empty course.
        first = must_dos[0]
        direct = {block.block_id: first.lo_id for block in blocks}
    rows = []
    for block in blocks:
        lo_id = direct.get(block.block_id)
        if lo_id is None:
            classification: Classification = "nice_to_know"
        elif block.content_kind == "reference" or block.support_role in {"job_aid", "checklist"}:
            classification = "reference"
        else:
            classification = "must_know"
        rows.append(IdmBlueprintRowV1(
            block_id=block.block_id, lo_id=lo_id,
            must_do_ids=must_dos_by_lo.get(lo_id, [])[:8] if lo_id and classification == "must_know" else [],
            classification=classification, placement=PLACEMENT_BY_CLASSIFICATION[classification],
            treatment=FALLBACK_TREATMENT[classification],
            detail_level=("Giữ nội dung cốt lõi cần để thực hiện Must Do." if vi
                          else "Keep the core content needed to perform the Must Do."),
            hold=False, hold_reason=None, sme_question=None, combine_into=None, separate_into=[],
            rationale=("Phương án dự phòng tự động theo liên kết mục tiêu học tập." if vi
                       else "Automatic fallback based on the objective links."),
        ))
    return rows


def apply_w2(
    rows: Sequence[IdmBlueprintRowV1],
    blocks: Sequence[IdmContentBlockV1],
    links: Sequence[IdmBlockLoLinkV1],
    index: FactIndex,
) -> tuple[list[IdmContentBlockV1], list[IdmBlockLoLinkV1], list[IdmBlueprintRowV1]]:
    """Apply combine then separate; new blocks continue the ``cb_NNNN`` numbering."""

    by_id = {block.block_id: block for block in blocks}
    row_by_id = {row.block_id: row for row in rows}
    link_list = list(links)
    next_number = max((int(block.block_id[3:]) for block in blocks), default=0) + 1
    for row in rows:
        if row.combine_into is None:
            continue
        source, target = by_id.pop(row.block_id), by_id[row.combine_into]
        by_id[target.block_id] = target.model_copy(update={
            "fact_keys": index.sort([*target.fact_keys, *source.fact_keys]),
            "issues": [*target.issues, *source.issues][:_MAX_BLOCK_ISSUES],
            "gaps": [*target.gaps, *source.gaps][:8],
            "sme_questions": list(dict.fromkeys([*target.sme_questions, *source.sme_questions]))[:_MAX_BLOCK_SME],
        })
        target_row = row_by_id[target.block_id]
        row_by_id[target.block_id] = target_row.model_copy(update={
            "must_do_ids": list(dict.fromkeys([*target_row.must_do_ids, *row.must_do_ids]))[:_MAX_ROW_MUST_DOS],
        })
        del row_by_id[row.block_id]
        link_list = [link.model_copy(update={"block_id": target.block_id}) if link.block_id == source.block_id
                     else link for link in link_list]
    for row in rows:
        if not row.separate_into or row.block_id not in by_id:
            continue
        original = by_id.pop(row.block_id)
        del row_by_id[row.block_id]
        original_links = [link for link in link_list if link.block_id == original.block_id]
        link_list = [link for link in link_list if link.block_id != original.block_id]
        for part_index, part in enumerate(row.separate_into):
            block_id = f"cb_{next_number:04d}"
            next_number += 1
            by_id[block_id] = original.model_copy(update={
                "block_id": block_id, "name": single_line(part.name, 180), "intent": part.intent,
                "fact_keys": index.sort(part.fact_keys),
                "issues": original.issues if part_index == 0 else [],
                "gaps": original.gaps if part_index == 0 else [],
                "sme_questions": original.sme_questions if part_index == 0 else [],
            })
            row_by_id[block_id] = row.model_copy(update={
                "block_id": block_id, "treatment": "condense", "separate_into": [],
            })
            link_list.extend(link.model_copy(update={"block_id": block_id}) for link in original_links)
    ordered_blocks = sorted(by_id.values(), key=lambda block: index.order[block.fact_keys[0]])
    deduped: dict[tuple[str, str], IdmBlockLoLinkV1] = {}
    for link in link_list:
        key = (link.block_id, link.lo_id)
        if key not in deduped or _RELATION_RANK[link.relation] < _RELATION_RANK[deduped[key].relation]:
            deduped[key] = link
    ordered_rows = [row_by_id[block.block_id] for block in ordered_blocks]
    return ordered_blocks, sorted(deduped.values(), key=lambda link: (link.block_id, link.lo_id)), ordered_rows


def blocked_must_dos(rows: Sequence[IdmBlueprintRowV1], must_dos: Sequence[IdmMustDoV1]) -> list[str]:
    """Must Dos with no course block left that is not on hold (server-computed)."""

    served = {md for row in rows if row.placement == "course" and not row.hold for md in row.must_do_ids}
    return [must_do.must_do_id for must_do in must_dos if must_do.must_do_id not in served]


def hold_items(
    rows: Sequence[IdmBlueprintRowV1], blocks: Sequence[IdmContentBlockV1], blocked: Sequence[str],
) -> list[IdmHoldItemV1]:
    names = {block.block_id: block.name for block in blocks}
    blocked_set = set(blocked)
    return [IdmHoldItemV1(
        block_id=row.block_id, name=names[row.block_id], reason=row.hold_reason or "-",
        sme_question=row.sme_question or "-",
        blocked_must_do_ids=[md for md in row.must_do_ids if md in blocked_set],
    ) for row in rows if row.hold]


def build_dispositions(
    rows: Sequence[IdmBlueprintRowV1],
    blocks: Sequence[IdmContentBlockV1],
    noise: Sequence[IdmNoiseDraftV1],
    index: FactIndex,
) -> list[IdmDispositionV1]:
    """Exactly one disposition per fact, in document order."""

    row_by_id = {row.block_id: row for row in rows}
    result: dict[str, IdmDispositionV1] = {}
    for block in blocks:
        row = row_by_id[block.block_id]
        if row.hold:
            disposition = "hold"
        elif row.placement == "course":
            disposition = "course"
        elif row.placement == "reference_job_aid":
            disposition = "reference_job_aid"
        else:
            disposition = row.classification
        for key in block.fact_keys:
            result[key] = IdmDispositionV1(fact_key=key, disposition=disposition,  # type: ignore[arg-type]
                                           block_id=block.block_id, reason=None)
    for item in noise:
        result.setdefault(item.fact_key, IdmDispositionV1(fact_key=item.fact_key, disposition="noise",
                                                          block_id=None, reason=item.reason))
    return [result[key] for key in sorted(result, key=lambda key: index.order[key])]


def _batches(blocks: Sequence[IdmContentBlockV1], links: Sequence[IdmBlockLoLinkV1],
             objectives: Sequence[IdmLearningObjectiveV1]) -> list[list[IdmContentBlockV1]]:
    if len(blocks) <= IDM_W2_SINGLE_CALL_MAX_BLOCKS:
        return [list(blocks)]
    first_lo: dict[str, str] = {}
    for link in sorted(links, key=lambda item: _RELATION_RANK[item.relation]):
        if link.relation in {"direct", "supporting"}:
            first_lo.setdefault(link.block_id, link.lo_id)
    groups: list[list[IdmContentBlockV1]] = [
        [block for block in blocks if first_lo.get(block.block_id) == objective.lo_id] for objective in objectives
    ]
    groups.append([block for block in blocks if block.block_id not in first_lo])
    batches: list[list[IdmContentBlockV1]] = []
    for group in groups:
        for start in range(0, len(group), IDM_W2_SINGLE_CALL_MAX_BLOCKS):
            if group[start:start + IDM_W2_SINGLE_CALL_MAX_BLOCKS]:
                batches.append(group[start:start + IDM_W2_SINGLE_CALL_MAX_BLOCKS])
    return batches


async def run_w2(
    runtime: IdmRuntime,
    *,
    context: IdmProjectContextV1,
    blocks: Sequence[IdmContentBlockV1],
    links: Sequence[IdmBlockLoLinkV1],
    objectives: Sequence[IdmLearningObjectiveV1],
    must_dos: Sequence[IdmMustDoV1],
    index: FactIndex,
    tail_reserve_tokens: int,
) -> BlueprintResult:
    codes: Counter[str] = Counter()
    rows: list[IdmBlueprintRowV1] = []
    fallback_batches = 0
    batches = _batches(blocks, links, objectives)
    for batch in batches:
        batch_ids = {block.block_id for block in batch}
        prompt = w2_blueprint_prompt(
            runtime.locale, project_context=context.model_dump(mode="json"),
            objectives=[item.model_dump(mode="json") for item in objectives],
            must_dos=[item.model_dump(mode="json") for item in must_dos],
            catalog=blueprint_catalog(batch, [link for link in links if link.block_id in batch_ids]),
        )
        accepted: list[IdmBlueprintRowV1] | None = None
        repair = ""
        thinking: ThinkingLevel = THINKING_W2
        for attempt in (1, 2):
            try:
                response = await idm_call(
                    runtime, stage="idm_w2", prompt=prompt + repair, response_model=IdmW2BlueprintResponseV1,
                    max_output_tokens=IDM_W2_MAX_OUTPUT_TOKENS, thinking_level=thinking,
                    invocation_kind="writer" if attempt == 1 else "repair",
                    reserve_after_tokens=tail_reserve_tokens,
                )
            except (IdmBudgetError, IdmResponseInvalidError) as error:
                codes[error.code] += 1
                if isinstance(error, IdmBudgetError):
                    break
                repair = answer_repair(error.code, error.errors, COMPACT_W2)
                thinking = repair_thinking(error.code, thinking)
                continue
            except IdmProviderError as error:
                if error.terminal:
                    raise
                codes[error.code] += 1
                break
            found = validate_w2(response, batch, objectives, must_dos)
            codes.update(issue.code for issue in found)
            if not errors(found):
                accepted = [_normalize_row(row) for row in response.rows]
                break
            repair = repair_suffix([issue.as_repair_item() for issue in errors(found)])
        if accepted is None:
            fallback_batches += 1
            record_deterministic_fallback(runtime, stage="idm_w2", code="IDM_W2_FALLBACK")
            accepted = fallback_w2(batch, [link for link in links if link.block_id in batch_ids], must_dos,
                                   runtime.locale)
        rows.extend(accepted)
    applied_blocks, applied_links, applied_rows = apply_w2(rows, blocks, links, index)
    origin: StageOrigin = ("provider" if fallback_batches == 0
                           else "deterministic_fallback" if fallback_batches == len(batches) else "partial_fallback")
    return BlueprintResult(applied_blocks, applied_links, applied_rows, origin, dict(codes))
