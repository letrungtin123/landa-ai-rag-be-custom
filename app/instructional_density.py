"""Old import path of ``app.content.instructional_density``, moved there in refactor wave 3b.

Strict code imports the new path; the legacy lesson-author pipeline and its tests keep importing these
names from here.
"""

from app.content.instructional_density import (
    DEFAULT_INSTRUCTIONAL_DENSITY_BUDGET,
    INSTRUCTIONAL_DENSITY_POLICY_VERSION,
    SOURCE_SCOPE_CHUNKS_PER_GROUP,
    InstructionalDensityBudget,
    InstructionalDensityDecision,
    InstructionalDensityProfile,
    InstructionalOutputBudget,
    evaluate_instructional_density,
    instructional_output_budget,
    partition_chunk_facts_for_density,
    profile_instructional_texts,
)

__all__ = [
    "DEFAULT_INSTRUCTIONAL_DENSITY_BUDGET",
    "INSTRUCTIONAL_DENSITY_POLICY_VERSION",
    "SOURCE_SCOPE_CHUNKS_PER_GROUP",
    "InstructionalDensityBudget",
    "InstructionalDensityDecision",
    "InstructionalDensityProfile",
    "InstructionalOutputBudget",
    "evaluate_instructional_density",
    "instructional_output_budget",
    "partition_chunk_facts_for_density",
    "profile_instructional_texts",
]
