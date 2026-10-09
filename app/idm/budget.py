"""Thinking levels of the W5 unit steps from the time a unit has left (QC run ab8d67e1, R3).

Gemini 3.8 takes a thinking level, not a token budget, and thinking time is the bulk of a call. The unit request
ends 120 s after Node sends it; the writer, an optional repair and the judge must all fit. A step runs at medium
thinking only when its p90 time at medium still leaves room for what must follow it (a low repair when the unit
has a question or a worksheet, then the judge); otherwise at low thinking, or not at all when even a low-thinking
call could not finish. Every decision is logged by the caller with the remaining seconds (numbers only).
"""

from __future__ import annotations

from typing import Final

from app.idm.policy import (
    IDM_CALL_HEADROOM_SECONDS,
    IDM_JUDGE_MIN_REMAINING_SECONDS,
    IDM_TARGETED_REPAIR_MIN_SECONDS,
    IDM_W5_REPAIR_LOW_SECONDS,
    IDM_W5_REPAIR_MEDIUM_SECONDS,
    IDM_W5_WRITER_MEDIUM_SECONDS,
    THINKING_TARGETED_REPAIR,
)
from app.idm.runtime import ThinkingLevel

_MEDIUM: Final = "medium"
_LOW: Final = "low"


def writer_thinking(remaining_seconds: float, *, repair_room: bool, preferred: ThinkingLevel) -> ThinkingLevel:
    """``preferred`` (medium) when its p90 time, a low repair (``repair_room``: the unit holds a question or a
    worksheet, the slots that need one most often) and the judge fit in ``remaining_seconds``; else low."""

    if preferred != _MEDIUM:
        return preferred
    needed = (IDM_W5_WRITER_MEDIUM_SECONDS + (IDM_W5_REPAIR_LOW_SECONDS if repair_room else 0.0)
              + IDM_JUDGE_MIN_REMAINING_SECONDS + IDM_CALL_HEADROOM_SECONDS)
    return _MEDIUM if remaining_seconds >= needed else _LOW


def repair_thinking_for(remaining_seconds: float, *, targeted: bool, preferred: ThinkingLevel) -> ThinkingLevel | None:
    """The thinking level of a W5 repair, or ``None`` when it would not finish in time.

    A targeted repair (findings a deterministic step settles anyway: a leak, a length cue, an FAQ item, a callout,
    a missing framework list) is a scoped rewrite at low thinking, started with at least
    ``IDM_TARGETED_REPAIR_MIN_SECONDS`` left. Any other repair runs at ``preferred`` (medium) only when its p90 time
    and the judge still fit, at low when a low repair fits, and is skipped below that: the slot is then settled by
    the deterministic steps (review note or source fallback) instead of a call cut by the deadline.
    """

    if targeted:
        return THINKING_TARGETED_REPAIR if remaining_seconds >= IDM_TARGETED_REPAIR_MIN_SECONDS else None
    if preferred == _MEDIUM and remaining_seconds >= (IDM_W5_REPAIR_MEDIUM_SECONDS + IDM_JUDGE_MIN_REMAINING_SECONDS
                                                      + IDM_CALL_HEADROOM_SECONDS):
        return _MEDIUM
    if remaining_seconds >= IDM_W5_REPAIR_LOW_SECONDS + IDM_CALL_HEADROOM_SECONDS:
        return _LOW
    return None


def judge_repair_fits(remaining_seconds: float) -> bool:
    """A repair after the judge, and the second judge that checks it, still fit."""

    return remaining_seconds >= (IDM_W5_REPAIR_LOW_SECONDS + IDM_JUDGE_MIN_REMAINING_SECONDS
                                 + IDM_CALL_HEADROOM_SECONDS)


__all__ = ["judge_repair_fits", "repair_thinking_for", "writer_thinking"]
