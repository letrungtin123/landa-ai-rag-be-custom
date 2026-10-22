"""Deterministic repair decisions of the W5 unit writer (``app.idm.storyboard``), owner decisions of 2026-10-20.

The QC analysis of that day found about a third of the units ``review_required``, and repairs that left a slot with a
review-level finding the writer's answer did not have (a new number or name, a question that now gave its answer away):

* P1: after every repair the findings are compared slot by slot (``slot_findings``); a slot the repair made worse
  gets its pre-repair version back, and the whole repair is dropped when that still leaves a slot worse (a question
  that copies html the repair rewrote) (``guard_repair``).
* P3: a slot that still gives its answer away, states a new fact or has a contested key after the first repair gets one
  more targeted repair when the time left allows (``second_repair_issues``); a slot kept for review whose findings a
  later step settles loses its review mark (``refresh_review_marks``).
* P6: a repair call that failed transiently (provider timeout or unavailable) is sent once more when enough time is
  left (``retry_level``).
* P7: an FAQ slot whose findings are all wording ones is settled without a repair call (``without_style_faq``): such
  repairs flipped its question types.

Codes, slot indexes and counts only reach the logs, never learner text.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from app.idm.budget import repair_thinking_for
from app.idm.faq_checks import FAQ_HYPE_CODE
from app.idm.fidelity_checks import NEW_FACT_CODE
from app.idm.html_rules import DENSITY_CODE
from app.idm.policy import FAQ_COMPONENT_TYPE, IDM_REPAIR_RETRY_MIN_SECONDS
from app.idm.qa import (
    ANSWER_CONTESTED_CODE,
    ANSWER_LEAK_CODE,
    STYLE_CODES,
    advisory_slot_findings,
    deterministic_slot_findings,
)
from app.idm.runtime import IdmProviderError, ThinkingLevel, log_stage
from app.idm.storyboard_context import _Draft, _slot_of
from app.idm.storyboard_steps import IdmUnitWriterSteps

# The transient provider failures a repair is sent again after; SERVICE_BUSY is this service's own limiter.
RETRIED_REPAIR_CODES: Final = frozenset({"AI_PROVIDER_TIMEOUT", "AI_PROVIDER_UNAVAILABLE"})
# The findings a second targeted repair is spent on (owner P3).
SECOND_REPAIR_CODES: Final = frozenset({ANSWER_LEAK_CODE, NEW_FACT_CODE, ANSWER_CONTESTED_CODE})
# The key of a shared-validator finding whose path names no slot.
UNIT_LEVEL: Final = -1
_NO_CODES: Final[frozenset[str]] = frozenset()
# Slot index -> the codes of the rules it fails.
SlotCodes = dict[int, frozenset[str]]


def worse(before: frozenset[str], after: frozenset[str]) -> bool:
    """``after`` adds a review-level code (any code but a wording one, ``STYLE_CODES``) to ``before``, or adds a code
    while fixing none."""

    added = after - before
    return bool(added) and (before <= after or any(code not in STYLE_CODES for code in added))


def _by_slot(found: Sequence[tuple[str, int | None]], fallback_slots: Sequence[int]) -> SlotCodes:
    slots: dict[int, set[str]] = {}
    for code, index in found:
        slots.setdefault(UNIT_LEVEL if index is None else index, set()).add(code)
    return {index: frozenset(codes) for index, codes in slots.items() if index not in fallback_slots}


def _changed_slots(before: _Draft, after: _Draft) -> set[int]:
    old, new = before.unit.get("components", []), after.unit.get("components", [])
    return {index for index, component in enumerate(new) if index >= len(old) or old[index] != component}


class IdmUnitRepairSteps(IdmUnitWriterSteps):
    """The repair decisions of one unit that need no provider call."""

    def slot_findings(self, draft: _Draft, *, advisory: bool = True) -> tuple[SlotCodes, SlotCodes, bool]:
        """``(rules, checks, complete)`` of the provider slots of ``draft``: the shared-validator finding (the first one
        only) or else Node's acceptance rules, per slot; the IDM checks and, with ``advisory``, the length cue, per
        slot; and whether ``rules`` holds every slot's rules (the shared validator passed, so Node read the whole unit).
        Fallback slots are left out; a finding without a slot is under ``UNIT_LEVEL``."""

        expected = self.expected_fallback if draft.fallback_slots else self.expected
        finding = self.deps.validate_unit(draft.unit, expected)
        # Node's rules are recorded for the unit log (``acceptance_seen``) like every other draft the writer checks.
        rules = ([(finding.code, _slot_of(finding))] if finding is not None else
                 [(item.code, item.component_index) for item in self.node_acceptance(draft)])
        checks = deterministic_slot_findings(draft.unit, self.brief, self.owned_text, self.evidence, self.framework,
                                             fidelity=self.fidelity, faq_check=self.faq, fact_rows=self.fact_rows)
        notes = advisory_slot_findings(draft.unit) if advisory else []
        return (_by_slot(rules, draft.fallback_slots),
                _by_slot([(item.code, item.component_index) for item in [*checks, *notes]], draft.fallback_slots),
                finding is None)

    def slot_codes(self, draft: _Draft, *, advisory: bool = True) -> SlotCodes:
        """Every rule each provider slot of ``draft`` fails (``slot_findings`` merged)."""

        rules, checks, _complete = self.slot_findings(draft, advisory=advisory)
        return {index: rules.get(index, _NO_CODES) | checks.get(index, _NO_CODES) for index in {*rules, *checks}}

    def guard_repair(self, before: _Draft, after: _Draft, step: str) -> _Draft:
        """P1: ``after`` (a repair of ``before``) with every slot it made worse (``worse``) back in its pre-repair
        version; ``before`` when that still leaves a slot worse (a question that now copies html the repair rewrote).

        Every IDM check is compared on every slot. A shared-validator or Node rule counts only on a slot the repair
        changed, and only when ``before`` passed the shared validator: it reports one finding, so a rule another slot
        already broke before the repair is no new one."""

        old_rules, old_checks, complete = self.slot_findings(before)

        def worse_slots(draft: _Draft) -> dict[int, frozenset[str]]:
            """Slot -> the codes ``draft`` adds to ``before`` on each slot it makes worse."""

            rules, checks, _complete = self.slot_findings(draft)
            changed = _changed_slots(before, draft) if complete else set()

            def codes(slot_rules: SlotCodes, slot_checks: SlotCodes, index: int) -> frozenset[str]:
                return slot_checks.get(index, _NO_CODES) | (slot_rules.get(index, _NO_CODES) if index in changed
                                                            else _NO_CODES)

            found = {}
            for index in sorted({*rules, *checks, *old_rules, *old_checks} - {UNIT_LEVEL}):
                old, new = codes(old_rules, old_checks, index), codes(rules, checks, index)
                if worse(old, new):
                    found[index] = new - old
            return found

        made_worse = worse_slots(after)
        # Review of dfb6db2 (item 6): a worksheet repair that adds its input data and so goes over the budget is
        # compacted (as the settle step would) before it is judged worse; the compaction's note stays only when kept.
        over = [index for index, added in made_worse.items() if DENSITY_CODE in added and self.is_worksheet(index)]
        if over:
            noted = (self.worksheet_compacted, self.worksheet_example_kept, len(self.codes))
            compacted = after
            for index in over:
                compacted = self.compact_worksheet(compacted, index) or compacted
            if compacted is not after and not set(over) & set(worse_slots(compacted)):
                after, made_worse = compacted, worse_slots(compacted)
            else:
                self.worksheet_compacted, self.worksheet_example_kept = noted[0], noted[1]
                del self.codes[noted[2]:]
        if not made_worse:
            return after
        slots = sorted(made_worse)
        candidate = self.with_slots_of(after, before, slots)
        # Nothing of the repair left (every slot it changed was made worse) is the whole repair dropped too.
        whole = candidate.unit.get("components") == before.unit.get("components") or bool(worse_slots(candidate))
        self.runtime.adjustments[f"{step}_repair_reverted" if whole else f"{step}_repair_slot_reverted"] += (
            1 if whole else len(slots))
        log_stage("idm_unit_repair_guard", {
            "correlation_id": self.runtime.correlation_id, "unit_path": self.contract.unit_path, "step": step,
            "reverted_slots": slots, "whole_repair": whole,
            "added_codes": sorted({code for codes in made_worse.values() for code in codes}),
        })
        return before if whole else candidate

    @staticmethod
    def with_slots_of(target: _Draft, source: _Draft, slots: Sequence[int]) -> _Draft:
        """``target`` with the components of ``slots`` taken from ``source``."""

        components = list(target.unit.get("components", []))
        previous = source.unit.get("components", [])
        for index in slots:
            if 0 <= index < len(components) and index < len(previous):
                components[index] = previous[index]
        return _Draft({**target.unit, "components": components}, list(target.fallback_slots), target.repair_applied,
                      list(target.review_slots), dict(target.reasons))

    def refresh_review_marks(self, draft: _Draft) -> _Draft:
        """P3: a slot kept for review whose findings a later step settled loses its mark; one that still fails keeps it
        with the codes it still has (QC: a slot kept for review stayed marked whatever happened after)."""

        if not draft.review_slots:
            return draft
        current = self.slot_codes(draft, advisory=False)
        review: list[int] = []
        reasons = dict(draft.reasons)
        for index in draft.review_slots:
            codes = current.get(index, _NO_CODES)
            if not codes:
                reasons.pop(index, None)
                self.runtime.adjustments["w5_review_mark_cleared"] += 1
                continue
            review.append(index)
            kept = list(draft.reasons.get(index, ()))
            reasons[index] = [*(code for code in kept if code in codes), *sorted(codes.difference(kept))]
        if review == draft.review_slots and reasons == draft.reasons:
            return draft
        return _Draft(draft.unit, list(draft.fallback_slots), draft.repair_applied, review, reasons)

    def second_repair_issues(self, draft: _Draft) -> list[tuple[str, int]]:
        """P3: the leaks, new facts and contested keys still found after the first repair, in slots without a shared
        or Node finding (those are settled by their fallback)."""

        finding, idm = self.problems(draft)
        blocked = _slot_of(finding)
        return [(code, index) for code, index in idm if code in SECOND_REPAIR_CODES and index != blocked]

    def without_style_faq(self, issues: Sequence[tuple[str, int]]) -> list[tuple[str, int]]:
        """P7: ``issues`` without those of the FAQ slots whose every finding is a wording one; they are settled by
        dropping items or as a polish note."""

        faq = {index for _code, index in issues
               if 0 <= index < len(self.plans) and self.plans[index]["type"] == FAQ_COMPONENT_TYPE}
        # Run 5496155c (owner decision 2026-10-22): hype words are repaired once, then noted.
        style_only = faq - {index for code, index in issues if code not in STYLE_CODES or code == FAQ_HYPE_CODE}
        if style_only:
            self.runtime.adjustments["w5_faq_style_repair_skipped"] += len(style_only)
        return [(code, index) for code, index in issues if index not in style_only]

    def retry_level(self, error: IdmProviderError, *, targeted: bool, preferred: ThinkingLevel,
                    failed_before: str | None) -> ThinkingLevel | None:
        """P6: the thinking level to send a repair again at after ``error``, or ``None``: only a transient provider
        failure of this very call (``failed_before``: the task's provider failure before it), with enough time left
        for the retry."""

        remaining = self.runtime.remaining_seconds()
        if (error.terminal or error.code not in RETRIED_REPAIR_CODES or failed_before is not None
                or remaining < IDM_REPAIR_RETRY_MIN_SECONDS):
            return None
        level = repair_thinking_for(remaining, targeted=targeted, preferred=preferred)
        if level is None:
            return None
        # The failure stopped every later call of the task (``IdmRuntime.provider_failure_code``); one retry may run.
        self.runtime.provider_failure_code = None
        self.runtime.adjustments["w5_repair_retried"] += 1
        self.log_budget("repair_retry", level)
        return level


__all__ = ["RETRIED_REPAIR_CODES", "SECOND_REPAIR_CODES", "UNIT_LEVEL", "IdmUnitRepairSteps", "worse"]
