"""Classify a provider HTTP 429 / RESOURCE_EXHAUSTED without exposing its message.

Gemini returns 429 for two different situations:

* the key cannot pay any more ("Your prepayment credits are depleted", "Your project has
  exceeded its monthly spending cap", billing disabled, a per-day quota): no retry inside
  the run can succeed, the run must stop and tell the user to top up or change the key;
* a per-minute rate limit (RPM/TPM, ``google.rpc.RetryInfo.retryDelay``, "Please retry in
  41s"): the same request succeeds after a short wait.

Before this split every 429 without a short retry hint became ``AI_PROVIDER_QUOTA_EXHAUSTED``
(production 2026-10-08: 41 rate-limited calls in seconds after a key switch were treated as an
exhausted key and silently replaced by deterministic content).

The provider message is read in memory only. Callers log the category and the hint seconds,
never the message: it can echo project identifiers and quota metric names.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Final, Literal

ProviderLimitKind = Literal["quota_exhausted", "rate_limited"]

QUOTA_EXHAUSTED_CODE: Final = "AI_PROVIDER_QUOTA_EXHAUSTED"
RATE_LIMITED_CODE: Final = "AI_PROVIDER_RATE_LIMITED"

# Billing state of the key. Deliberately not "billing details": Gemini's generic quota message
# ("please check your plan and billing details") also accompanies per-minute limits.
_BILLING_RE: Final = re.compile(
    r"prepayment|prepaid|credits? (?:are|is|have been|has been) (?:depleted|exhausted)|credit balance|"
    r"out of credits?|spending (?:cap|limit)|spend cap|billing account|billing (?:is|has been) "
    r"(?:disabled|not enabled)|enable billing|insufficient (?:funds|balance|credits?)|payment required",
)
# Daily quotas ("GenerateRequestsPerDayPerProjectPerModel") do not recover within a run.
_DAILY_RE: Final = re.compile(r"per ?day|requests? per day|\brpd\b|daily (?:limit|quota)")
# Per-minute / transient throttling: RPM/TPM quota metrics, RetryInfo, "Please retry in 41s",
# Vertex dynamic shared quota ("Resource has been exhausted (e.g. check quota)", "try again later").
_RATE_RE: Final = re.compile(
    r"per ?minute|\brpm\b|\btpm\b|rate.?limit|too many requests|retry ?in \d|retrydelay|retryinfo|"
    r"try again later|resource has been exhausted",
)
_MESSAGE_RETRY_RE: Final = re.compile(r"retry in (\d+(?:\.\d+)?)\s*(ms|s)\b")
_MAX_SCANNED_CHARS: Final = 16_000
_MS_PER_SECOND: Final = 1000.0


@dataclass(frozen=True)
class ProviderLimit:
    kind: ProviderLimitKind
    retry_after_seconds: float | None

    @property
    def code(self) -> str:
        return QUOTA_EXHAUSTED_CODE if self.kind == "quota_exhausted" else RATE_LIMITED_CODE


def _provider_text(error: BaseException) -> str:
    parts = [str(error)]
    details = getattr(error, "details", None)
    if details is not None:
        try:
            parts.append(json.dumps(details, default=str))
        except (TypeError, ValueError):
            parts.append(str(details))
    return " ".join(parts)[:_MAX_SCANNED_CHARS].casefold()


def _message_retry_seconds(text: str) -> float | None:
    match = _MESSAGE_RETRY_RE.search(text)
    if match is None:
        return None
    value = float(match.group(1))
    return value / _MS_PER_SECOND if match.group(2) == "ms" else value


def classify_provider_limit(error: BaseException, retry_hint_seconds: float | None) -> ProviderLimit:
    """Return the limit category of a 429 and the server's retry hint, if any.

    Billing and daily-quota markers win over a retry hint (an exhausted key may still carry
    one). A retry hint or a throttling marker means a rate limit. A bare RESOURCE_EXHAUSTED
    without either keeps the pre-classification meaning (exhausted quota): the legacy flows
    and their characterization tests depend on it, and Gemini rate limits carry a RetryInfo
    hint or a per-minute quota metric.
    """

    text = _provider_text(error)
    if _BILLING_RE.search(text) or _DAILY_RE.search(text):
        return ProviderLimit("quota_exhausted", None)
    hint = retry_hint_seconds if retry_hint_seconds is not None else _message_retry_seconds(text)
    if hint is not None or _RATE_RE.search(text):
        return ProviderLimit("rate_limited", hint)
    return ProviderLimit("quota_exhausted", None)
