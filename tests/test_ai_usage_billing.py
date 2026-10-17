"""AI credits P0: provider usage is recorded the way Google bills it."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from app.idm.runtime import IdmUsageLedger
from app.schemas.common import AiUsage
from app.services.provider import combine_usage, normalize_usage, usage_from_google_response


def _response(**meta: int) -> SimpleNamespace:
    return SimpleNamespace(usage_metadata=SimpleNamespace(**meta))


def test_thinking_is_billed_output_and_tool_prompt_is_input() -> None:
    usage = usage_from_google_response(_response(
        prompt_token_count=1_000, tool_use_prompt_token_count=300, candidates_token_count=200,
        thoughts_token_count=150, total_token_count=1_650), "", "")
    assert usage.model_dump() == {"inputTokens": 1_300, "outputTokens": 350, "embeddingTokens": 0, "totalTokens": 1_650}


def test_cached_input_is_kept_apart_and_bounded() -> None:
    usage = usage_from_google_response(_response(
        prompt_token_count=2_000, cached_content_token_count=1_500, candidates_token_count=100,
        total_token_count=2_100), "", "")
    assert usage.cachedInputTokens == 1_500
    assert usage.model_dump()["cachedInputTokens"] == 1_500
    assert normalize_usage(input_tokens=10, cached_input_tokens=99).cachedInputTokens == 10


def test_total_is_never_below_the_billed_sum() -> None:
    usage = usage_from_google_response(_response(
        prompt_token_count=100, candidates_token_count=10, thoughts_token_count=40, total_token_count=110), "", "")
    assert usage.totalTokens == 150


def test_missing_or_mocked_counters_fall_back_to_estimates() -> None:
    usage = usage_from_google_response(MagicMock(), "abcd" * 10, "xy")
    assert usage.model_dump() == {"inputTokens": 10, "outputTokens": 1, "embeddingTokens": 0, "totalTokens": 11}
    no_meta = usage_from_google_response(SimpleNamespace(), "abcd", "")
    assert no_meta.inputTokens == 1
    assert no_meta.outputTokens == 0


def test_wire_shape_is_unchanged_without_a_cache_hit() -> None:
    assert AiUsage(inputTokens=5, outputTokens=1, totalTokens=6).model_dump() == {
        "inputTokens": 5, "outputTokens": 1, "embeddingTokens": 0, "totalTokens": 6}
    assert "cachedInputTokens" not in AiUsage().model_dump_json()
    assert '"cachedInputTokens":3' in AiUsage(inputTokens=5, cachedInputTokens=3).model_dump_json()


def test_combined_usage_adds_the_cached_part() -> None:
    first = normalize_usage(input_tokens=100, output_tokens=5, cached_input_tokens=40)
    second = normalize_usage(input_tokens=50, output_tokens=5, cached_input_tokens=20)
    assert combine_usage(first, second).cachedInputTokens == 60


def test_idm_ledger_reports_the_cached_part_only_when_there_is_one() -> None:
    ledger = IdmUsageLedger(input_tokens=500, output_tokens=80, total_tokens=580)
    assert ledger.as_usage() == {"inputTokens": 500, "outputTokens": 80, "embeddingTokens": 0, "totalTokens": 580}
    ledger.cached_input_tokens = 200
    assert ledger.as_usage()["cachedInputTokens"] == 200
    ledger.cached_input_tokens = 900
    assert ledger.as_usage()["cachedInputTokens"] == 500
