"""Lesson-author domain errors."""

from __future__ import annotations


class LessonAuthorProposalValidationError(ValueError):
    """Raised when a detailed lesson proposal cannot be applied safely."""
    def __init__(self, message: str, *, code: str = "UNIT_SHAPE_INVALID", path: str = "unit", repairable: bool = False):
        super().__init__(message)
        self.code, self.path, self.repairable = code, path, repairable
