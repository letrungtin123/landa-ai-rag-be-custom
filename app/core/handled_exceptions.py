"""What broad exception handlers turn into a fallback or a safe error code.

Several handlers catch every ``Exception`` (or ``TypeError``/``KeyError`` among recoverable answer
errors) and continue with deterministic content, ``AI_PROVIDER_UNAVAILABLE`` or an
``IDM_W5_*_INVALID`` answer code. That keeps a run alive, but it also makes a programming error look
like a provider or model failure. Narrowing them changes behaviour, so the service first records
what they catch: one WARNING ``broad_exception_handled`` line with the stage, the outcome, the
exception's bounded type label and its sanitised traceback (never its message: it can carry
prompts, provider output or personal data), and the ``ai_rag_handled_exceptions_total`` counter.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.core import metrics
from app.core.logging import exception_fields, exception_type_label, log_event

HANDLED_EXCEPTION_EVENT = "broad_exception_handled"
# Outcomes are the stable codes the handler answers with (AI_PROVIDER_UNAVAILABLE, ...).
OUTCOME_LABEL_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,95}")


def _outcome_label(outcome: str) -> str:
    return outcome if OUTCOME_LABEL_PATTERN.fullmatch(outcome) else metrics.INVALID_STAGE


def count_handled_exception(error: BaseException, *, stage: str, outcome: str) -> None:
    """Count a handled exception whose handler already logs it with its traceback."""

    metrics.HANDLED_EXCEPTIONS.labels(stage=metrics.stage_label(stage), exception_type=exception_type_label(error),
                                      outcome=_outcome_label(outcome)).inc()


def report_handled_exception(
    logger: logging.Logger,
    error: BaseException,
    *,
    stage: str,
    outcome: str,
    family: str | None = None,
    **fields: Any,
) -> None:
    """Count ``error`` and log its type, stage, outcome and sanitised traceback at WARNING.

    ``fields`` are identifiers only (correlation id, unit path, attempt): never content.
    """

    count_handled_exception(error, stage=stage, outcome=outcome)
    log_event(logger, logging.WARNING, HANDLED_EXCEPTION_EVENT, family=family, stage=metrics.stage_label(stage),
              outcome=_outcome_label(outcome), **exception_fields(error), **fields)


__all__ = ["HANDLED_EXCEPTION_EVENT", "count_handled_exception", "report_handled_exception"]
