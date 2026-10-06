from __future__ import annotations

from dataclasses import dataclass, fields
from math import ceil
import re


INSTRUCTIONAL_DENSITY_POLICY_VERSION = "unit-content-v3-density-1"
SOURCE_SCOPE_CHUNKS_PER_GROUP = 4


@dataclass(frozen=True, slots=True)
class InstructionalDensityBudget:
    """Pedagogical limits for one independently teachable unit.

    These limits are intentionally much smaller than provider transport caps.
    A request can be valid JSON and still be too dense to become one useful
    lesson. The architecture stage uses this contract to decide when a source
    scope must be split before any component content is authored.
    """

    max_canonical_facts: int = 48
    max_source_chars: int = 12_000
    max_estimated_words: int = 1_600
    max_learning_objectives: int = 3
    max_independent_topics: int = 2

    def __post_init__(self) -> None:
        for field in fields(self):
            if getattr(self, field.name) < 1:
                raise ValueError(f"{field.name} must be positive")


DEFAULT_INSTRUCTIONAL_DENSITY_BUDGET = InstructionalDensityBudget()


@dataclass(frozen=True, slots=True)
class InstructionalDensityProfile:
    canonical_fact_count: int
    source_chars: int
    estimated_words: int
    learning_objective_count: int
    independent_topic_count: int

    def __post_init__(self) -> None:
        for field in fields(self):
            if getattr(self, field.name) < 0:
                raise ValueError(f"{field.name} cannot be negative")


@dataclass(frozen=True, slots=True)
class InstructionalDensityDecision:
    policy_version: str
    split_required: bool
    recommended_unit_count: int
    reason_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class InstructionalOutputBudget:
    """Bound learner-facing explanation size without mirroring transport caps."""

    max_visible_chars: int
    max_words: int

    def __post_init__(self) -> None:
        if self.max_visible_chars < 1 or self.max_words < 1:
            raise ValueError("instructional output limits must be positive")


_DIMENSIONS = (
    ("canonical_fact_count", "max_canonical_facts", "UNIT_FACT_DENSITY_EXCEEDED"),
    ("source_chars", "max_source_chars", "UNIT_SOURCE_TEXT_DENSITY_EXCEEDED"),
    ("estimated_words", "max_estimated_words", "UNIT_READING_TIME_EXCEEDED"),
    ("learning_objective_count", "max_learning_objectives", "UNIT_OBJECTIVE_DENSITY_EXCEEDED"),
    ("independent_topic_count", "max_independent_topics", "UNIT_TOPIC_DENSITY_EXCEEDED"),
)


def evaluate_instructional_density(
    profile: InstructionalDensityProfile,
    budget: InstructionalDensityBudget = DEFAULT_INSTRUCTIONAL_DENSITY_BUDGET,
) -> InstructionalDensityDecision:
    """Return a deterministic minimum split recommendation.

    The largest over-budget dimension wins. This prevents a very large source
    scope from being accepted merely because another dimension is small, while
    keeping the result explainable and stable for logs, tests and retries.
    """

    reason_codes: list[str] = []
    recommended_unit_count = 1
    for profile_field, budget_field, reason_code in _DIMENSIONS:
        observed = getattr(profile, profile_field)
        limit = getattr(budget, budget_field)
        if observed > limit:
            reason_codes.append(reason_code)
            recommended_unit_count = max(recommended_unit_count, ceil(observed / limit))

    return InstructionalDensityDecision(
        policy_version=INSTRUCTIONAL_DENSITY_POLICY_VERSION,
        split_required=bool(reason_codes),
        recommended_unit_count=recommended_unit_count,
        reason_codes=tuple(reason_codes),
    )


def profile_instructional_texts(
    texts: list[str],
    *,
    learning_objective_count: int,
    independent_topic_count: int,
) -> InstructionalDensityProfile:
    return InstructionalDensityProfile(
        canonical_fact_count=len(texts),
        source_chars=sum(len(text) for text in texts),
        estimated_words=sum(len(re.findall(r"\w+", text, flags=re.UNICODE)) for text in texts),
        learning_objective_count=learning_objective_count,
        independent_topic_count=independent_topic_count,
    )


def instructional_output_budget(
    profile: InstructionalDensityProfile,
) -> InstructionalOutputBudget:
    """Return a bounded expansion allowance for one source-safe unit.

    Sparse source facts need enough room for a useful explanation, while a
    dense source unit must not expand into a monolithic learner-facing page.
    Character and word ceilings are both enforced because Vietnamese and
    English have materially different average token/word lengths.
    """

    return InstructionalOutputBudget(
        max_visible_chars=min(18_000, max(4_000, ceil(profile.source_chars * 1.75))),
        max_words=min(2_400, max(600, ceil(profile.estimated_words * 1.75))),
    )


def partition_chunk_facts_for_density(
    fact_texts: list[str],
    *,
    chunks_per_scope: int = SOURCE_SCOPE_CHUNKS_PER_GROUP,
    budget: InstructionalDensityBudget = DEFAULT_INSTRUCTIONAL_DENSITY_BUDGET,
) -> list[list[str]]:
    """Partition one chunk into deterministic lanes safe to merge by group.

    Source snapshot pagination can begin between chunks, so scope identity
    cannot depend on mutable cross-page accumulators. Each chunk receives one
    quarter of the unit budget; equal lane indexes from four adjacent chunks
    may then share a scope without exceeding the complete unit budget.
    """

    if chunks_per_scope < 1:
        raise ValueError("chunks_per_scope must be positive")
    limits = {
        "facts": max(1, budget.max_canonical_facts // chunks_per_scope),
        "chars": max(1, budget.max_source_chars // chunks_per_scope),
        "words": max(1, budget.max_estimated_words // chunks_per_scope),
    }
    groups: list[list[str]] = []
    current: list[str] = []
    current_chars = 0
    current_words = 0
    for text in fact_texts:
        chars = len(text)
        words = len(re.findall(r"\w+", text, flags=re.UNICODE))
        if chars > limits["chars"] or words > limits["words"]:
            raise ValueError("one canonical fact exceeds the density lane budget")
        exceeds = (
            len(current) + 1 > limits["facts"]
            or current_chars + chars > limits["chars"]
            or current_words + words > limits["words"]
        )
        if exceeds and current:
            groups.append(current)
            current = []
            current_chars = 0
            current_words = 0
        current.append(text)
        current_chars += chars
        current_words += words
    if current:
        groups.append(current)
    return groups
