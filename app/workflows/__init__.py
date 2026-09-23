"""Request-scoped LangGraph workflows for AI authoring.

The workflows in this package orchestrate generation and validation only. They
never open a database transaction, persist a proposal, or apply a course
change. Those responsibilities remain with the Node control plane.
"""

from .contracts import WorkflowFailure

__all__ = ["WorkflowFailure"]
