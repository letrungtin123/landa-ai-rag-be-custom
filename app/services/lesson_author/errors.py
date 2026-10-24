"""Old import path of ``app.services.staged_unit.errors``, moved there in refactor wave 3b.

Strict code imports the new path; the legacy lesson-author pipeline and its tests keep importing these
names from here.
"""

from app.services.staged_unit.errors import (
    LessonAuthorProposalValidationError,
)

__all__ = [
    "LessonAuthorProposalValidationError",
]
