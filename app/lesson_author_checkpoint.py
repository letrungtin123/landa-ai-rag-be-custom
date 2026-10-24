"""Old import path of ``app.contracts.lesson_author_checkpoint``, moved there in refactor wave 3b.

Strict code imports the new path; the legacy lesson-author pipeline and its tests keep importing these
names from here.
"""

from app.contracts.lesson_author_checkpoint import (
    CHECKPOINT_VERSION,
    MAX_CHECKPOINT_UNIT_BYTES,
    MAX_CHECKPOINT_UNITS,
    ChapterCheckpointUnit,
    assemble_checkpoint_chapter,
    checkpoint_expected_units,
    select_checkpoint_unit,
)

__all__ = [
    "CHECKPOINT_VERSION",
    "MAX_CHECKPOINT_UNITS",
    "MAX_CHECKPOINT_UNIT_BYTES",
    "ChapterCheckpointUnit",
    "assemble_checkpoint_chapter",
    "checkpoint_expected_units",
    "select_checkpoint_unit",
]
