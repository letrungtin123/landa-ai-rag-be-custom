"""Organization profile distillation (ORG_CONTEXT spec §4): the context documents in, one draft profile out.

The service layer (``app.services.org_context.distill``) reads the learned chunks of the tenant's context documents
and hands their sizes, then their texts, to this module, which never touches the database:

1. ``sample_chunks``: every chunk when the source fits the read limit; otherwise an even sample over the whole source,
   in document and source order, that fits it (the answer then says ``truncated``).
2. ``batch_source``: the texts packed in order into batches (``ORG_DISTILL_BATCH_MAX_CHARS``).
3. ``distill_profile``: one map call per batch (bounded concurrency, the IDM runtime: deadline, JSON schema mode, usage
   ledger), then reduce calls that merge the partial profiles; a single batch is answered by its map call alone.
4. ``fit_profile``: every answer is brought into the profile v1 bounds by cleaning (as Node's normalisation does) and by
   dropping trailing list items, never by writing anything new.

Every rule text is server-owned (``app.idm.prompts_org_distill``); the documents are data, and the prompts ask for
what they state only, in the requested locale.
"""

from __future__ import annotations

import asyncio
import unicodedata
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Final

from app.idm.contracts import IdmOrgProfileDraftV1
from app.idm.org_profile import (
    ORG_CONVENTION_MAX,
    ORG_CONVENTIONS_MAX_ITEMS,
    ORG_CULTURE_TONE_MAX,
    ORG_GLOSSARY_MAX_ITEMS,
    ORG_MEANING_MAX,
    ORG_PROCESS_MAX,
    ORG_PROCESSES_MAX_ITEMS,
    ORG_PRODUCT_MAX,
    ORG_PRODUCTS_MAX_ITEMS,
    ORG_PROFILE_MAX_CHARS,
    ORG_SUMMARY_MAX,
    ORG_TERM_MAX,
    ORG_TEXT_MAX_LINES,
    IdmOrgGlossaryTermV1,
    IdmOrgLearnerGroupV1,
    IdmOrgProfileV1,
    org_profile_char_count,
)
from app.idm.policy import (
    ORG_DISTILL_ATTEMPTS,
    ORG_DISTILL_MAX_OUTPUT_TOKENS,
    ORG_DISTILL_REDUCE_FANIN,
    THINKING_ORG_DISTILL,
)
from app.idm.prompts_common import answer_repair
from app.idm.prompts_org_distill import COMPACT_ORG_PROFILE, distill_map_prompt, distill_reduce_prompt
from app.idm.runtime import (
    IdmResponseInvalidError,
    IdmRuntime,
    InvocationKind,
    ThinkingLevel,
    idm_call,
    repair_thinking,
)
from app.idm.text import trim_at_boundary

_STAGE: Final = "org_context_distill"
_LIST_FIELDS: Final = ("products_services", "learner_groups", "glossary", "key_processes", "design_conventions")


@dataclass(frozen=True)
class OrgSourceChunk:
    """One learned chunk of a context document: where it is and how many characters its text has."""

    document_id: str
    chunk_no: int
    chars: int


@dataclass(frozen=True)
class OrgSourceBatch:
    """The texts of one map call: (document number, texts in source order) per document, in document order."""

    documents: tuple[tuple[int, tuple[str, ...]], ...]
    chars: int


def _even(chunks: Sequence[OrgSourceChunk], keep: int) -> list[OrgSourceChunk]:
    """The middle chunk of each of ``keep`` equal runs of ``chunks`` (strictly increasing positions)."""

    count = len(chunks)
    return [chunks[(2 * index + 1) * count // (2 * keep)] for index in range(keep)]


def sample_chunks(chunks: Sequence[OrgSourceChunk], max_chars: int) -> tuple[list[OrgSourceChunk], bool]:
    """``(chunks to read, truncated)``: every chunk when their texts fit ``max_chars``; otherwise an even sample over
    the whole source that fits, as many chunks as a search over the sample size finds (at least one, which
    ``batch_source`` cuts to the limit)."""

    total = sum(chunk.chars for chunk in chunks)
    if total <= max_chars:
        return list(chunks), False
    # The average chunk size gives the first estimate; a sample of larger chunks needs fewer of them.
    low, high = 1, max(1, min(len(chunks), len(chunks) * max_chars // total))
    while low < high:
        middle = (low + high + 1) // 2
        if sum(chunk.chars for chunk in _even(chunks, middle)) <= max_chars:
            low = middle
        else:
            high = middle - 1
    return _even(chunks, low), True


def batch_source(documents: Sequence[Sequence[str]], *, max_chars: int, batch_chars: int) -> list[OrgSourceBatch]:
    """The texts of every document (in document and source order) packed into batches of at most ``batch_chars``
    characters, the whole at most ``max_chars``: a text past the limit is cut there, and nothing after it is read."""

    batches: list[OrgSourceBatch] = []
    current: dict[int, list[str]] = {}
    size, room = 0, max_chars

    def close() -> None:
        nonlocal current, size
        if current:
            batches.append(OrgSourceBatch(tuple((number, tuple(texts)) for number, texts in current.items()), size))
        current, size = {}, 0

    for number, texts in enumerate(documents, start=1):
        for text in texts:
            if room <= 0:
                break
            piece = text[:room]
            room -= len(piece)
            if current and size + len(piece) > batch_chars:
                close()
            current.setdefault(number, []).append(piece)
            size += len(piece)
    close()
    return batches


# --- Bounds (spec §2): clean like Node's normalisation, then drop trailing items ------------------------------------
def _line(value: str) -> str:
    """One line: control characters become spaces, runs of spaces collapse, the ends are trimmed."""

    return " ".join("".join(" " if unicodedata.category(char) == "Cc" else char for char in value).split())


def _text(value: str, cap: int, *, multiline: bool) -> str:
    lines = [line for line in (_line(part) for part in unicodedata.normalize("NFC", value).split("\n")) if line]
    if multiline and len(lines) > ORG_TEXT_MAX_LINES:
        lines = [*lines[:ORG_TEXT_MAX_LINES - 1], " ".join(lines[ORG_TEXT_MAX_LINES - 1:])]
    text = ("\n" if multiline else " ").join(lines)
    return text if len(text) <= cap else trim_at_boundary(text, cap)


def _items(values: Sequence[str], cap: int, limit: int) -> list[str]:
    """Clean items, without empty ones or repeats (first kept), at most ``limit``."""

    kept: dict[str, str] = {}
    for value in values:
        text = _text(value, cap, multiline=False)
        if text and text.casefold() not in kept:
            kept[text.casefold()] = text
    return list(kept.values())[:limit]


def _pairs(values: Sequence[tuple[str, str]], caps: tuple[int, int], limit: int, *,
           second_required: bool) -> list[tuple[str, str]]:
    """Clean (name, text) pairs keyed by the name: no empty name (or text when required), no repeated name."""

    kept: dict[str, tuple[str, str]] = {}
    for first, second in values:
        name, text = _text(first, caps[0], multiline=False), _text(second, caps[1], multiline=False)
        if name and (text or not second_required) and name.casefold() not in kept:
            kept[name.casefold()] = (name, text)
    return list(kept.values())[:limit]


def _profile(fields: dict[str, Any]) -> IdmOrgProfileV1:
    """An unvalidated profile of ``fields`` (strings, string lists and pairs), for counting."""

    return IdmOrgProfileV1.model_construct(
        organization_summary=fields["organization_summary"], culture_tone=fields["culture_tone"],
        products_services=fields["products_services"], key_processes=fields["key_processes"],
        design_conventions=fields["design_conventions"],
        learner_groups=[IdmOrgLearnerGroupV1.model_construct(name=name, description=text)
                        for name, text in fields["learner_groups"]],
        glossary=[IdmOrgGlossaryTermV1.model_construct(term=name, meaning=text) for name, text in fields["glossary"]])


def _list_chars(items: Sequence[str | tuple[str, str]]) -> int:
    return sum(len(item) if isinstance(item, str) else len(item[0]) + len(item[1]) for item in items)


def fit_profile(draft: IdmOrgProfileDraftV1) -> IdmOrgProfileV1:
    """The profile v1 of a provider draft: cleaned strings (NFC, no control characters, single spaces, no empty or
    repeated items, line breaks only in the two text fields), the per-field bounds, and the 6000-character total
    reached by dropping the last item of the list that holds the most text, repeatedly. Nothing is ever added.

    ``learner_groups`` is always [] (owner 2026-10-22, contract K3): the learners of a course come from its own
    audience, so the draft lists no groups even when the answer does."""

    fields: dict[str, Any] = {
        "organization_summary": _text(draft.organization_summary, ORG_SUMMARY_MAX, multiline=True),
        "culture_tone": _text(draft.culture_tone, ORG_CULTURE_TONE_MAX, multiline=True),
        "products_services": _items(draft.products_services, ORG_PRODUCT_MAX, ORG_PRODUCTS_MAX_ITEMS),
        "key_processes": _items(draft.key_processes, ORG_PROCESS_MAX, ORG_PROCESSES_MAX_ITEMS),
        "design_conventions": _items(draft.design_conventions, ORG_CONVENTION_MAX, ORG_CONVENTIONS_MAX_ITEMS),
        "learner_groups": [],
        "glossary": _pairs([(entry.term, entry.meaning) for entry in draft.glossary], (ORG_TERM_MAX, ORG_MEANING_MAX),
                           ORG_GLOSSARY_MAX_ITEMS, second_required=True),
    }
    while org_profile_char_count(_profile(fields)) > ORG_PROFILE_MAX_CHARS:
        # The two text fields hold at most 1,400 characters, so a list always has an item left to drop here.
        longest = max((name for name in _LIST_FIELDS if fields[name]), key=lambda name: _list_chars(fields[name]))
        fields[longest].pop()
    return IdmOrgProfileV1.model_validate({
        **fields,
        "learner_groups": [{"name": name, "description": text} for name, text in fields["learner_groups"]],
        "glossary": [{"term": name, "meaning": text} for name, text in fields["glossary"]],
    })


# --- Provider calls ---------------------------------------------------------------------------------------------------
async def _ask(runtime: IdmRuntime, prompt: str) -> IdmOrgProfileV1:
    """One profile answer: a writer call and, after an invalid answer, one repair (the last invalid answer raises)."""

    repair = ""
    thinking: ThinkingLevel = THINKING_ORG_DISTILL
    attempt = 1
    while True:
        kind: InvocationKind = "writer" if attempt == 1 else "repair"
        try:
            draft = await idm_call(runtime, stage=_STAGE, prompt=prompt + repair, response_model=IdmOrgProfileDraftV1,
                                   max_output_tokens=ORG_DISTILL_MAX_OUTPUT_TOKENS, thinking_level=thinking,
                                   invocation_kind=kind)
        except IdmResponseInvalidError as error:
            if attempt >= ORG_DISTILL_ATTEMPTS:
                raise
            repair = answer_repair(error.code, error.errors, COMPACT_ORG_PROFILE)
            thinking = repair_thinking(error.code, thinking)
            attempt += 1
            continue
        return fit_profile(draft)


async def _gather[T, R](items: Sequence[T], work: Callable[[T], Awaitable[R]], parallelism: int) -> list[R]:
    """``work`` on every item, at most ``parallelism`` at a time; the first failure cancels the rest."""

    gate = asyncio.Semaphore(max(1, parallelism))

    async def run(item: T) -> R:
        async with gate:
            return await work(item)

    tasks = [asyncio.create_task(run(item)) for item in items]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def distill_profile(runtime: IdmRuntime, batches: Sequence[OrgSourceBatch], *, parallelism: int,
                          excerpts: bool) -> IdmOrgProfileV1:
    """The profile of the whole source: map every batch, then reduce the partial profiles, at most
    ``ORG_DISTILL_REDUCE_FANIN`` per call, until one is left. Budget, provider and invalid-answer errors propagate."""

    if not batches:
        raise ValueError("distill_profile needs at least one batch")

    def map_batch(position: int) -> Awaitable[IdmOrgProfileV1]:
        return _ask(runtime, distill_map_prompt(runtime.locale, documents=batches[position].documents,
                                                part=position + 1, parts=len(batches), excerpts=excerpts))

    profiles = await _gather(range(len(batches)), map_batch, parallelism)
    while len(profiles) > 1:
        groups = [profiles[start:start + ORG_DISTILL_REDUCE_FANIN]
                  for start in range(0, len(profiles), ORG_DISTILL_REDUCE_FANIN)]
        profiles = await _gather(groups, lambda group: _reduce(runtime, group), parallelism)
    return profiles[0]


async def _reduce(runtime: IdmRuntime, group: Sequence[IdmOrgProfileV1]) -> IdmOrgProfileV1:
    if len(group) == 1:
        return group[0]
    return await _ask(runtime, distill_reduce_prompt(runtime.locale,
                                                     partials=[item.model_dump(mode="json") for item in group]))
