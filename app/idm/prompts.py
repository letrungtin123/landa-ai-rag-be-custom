"""Prompt builders of policy ``idm-prompt-1`` (spec Appendix A), one module per stage.

Builders are pure string functions. Every document-derived value is wrapped in a
tagged block whose closing tag is neutralised (SEC-9); the preamble tells the
model that tagged text is data. Repair prompts list only codes, paths, server-owned rule
text and server-measured numbers, never source text or the rejected answer.

The builders live in ``prompts_common`` (preamble, fact lines, repair suffixes), ``prompts_course`` (W1, W2, W4),
``prompts_module`` (W3/W4 module), ``prompts_unit`` (W5 writer and repair rule lines) and ``prompts_judge`` (W6).
This module re-exports them under their historical import path.
"""

from __future__ import annotations

from app.idm.prompts_common import (
    COMPACT_MODULE,
    COMPACT_UNIT,
    COMPACT_W1_MAP,
    COMPACT_W1_REDUCE,
    COMPACT_W2,
    COMPACT_W4,
    PROMPT_POLICY_VERSION,
    TRUNCATED_RESPONSE_CODE,
    UNTRUSTED_TAGS,
    Locale,
    answer_repair,
    fact_lines,
    learner_fact_lines,
    locale_name,
    preamble,
    repair_suffix,
    truncation_suffix,
)
from app.idm.prompts_course import w1_reduce_prompt, w1_section_prompt, w2_blueprint_prompt, w4_course_prompt
from app.idm.prompts_judge import judge_prompt
from app.idm.prompts_module import module_prompt
from app.idm.prompts_unit import (
    DENSITY_TARGET_SHARE,
    UNIT_RULE_TEXT,
    example_opening,
    html_slot_rules,
    html_violation_line,
    new_fact_hint,
    rule_line,
    second_repair_line,
    unit_repair_suffix,
    unit_writer_prompt,
)

__all__ = [
    "COMPACT_MODULE",
    "COMPACT_UNIT",
    "COMPACT_W1_MAP",
    "COMPACT_W1_REDUCE",
    "COMPACT_W2",
    "COMPACT_W4",
    "DENSITY_TARGET_SHARE",
    "PROMPT_POLICY_VERSION",
    "TRUNCATED_RESPONSE_CODE",
    "UNIT_RULE_TEXT",
    "UNTRUSTED_TAGS",
    "Locale",
    "answer_repair",
    "example_opening",
    "fact_lines",
    "html_slot_rules",
    "html_violation_line",
    "judge_prompt",
    "learner_fact_lines",
    "locale_name",
    "module_prompt",
    "new_fact_hint",
    "preamble",
    "repair_suffix",
    "rule_line",
    "second_repair_line",
    "truncation_suffix",
    "unit_repair_suffix",
    "unit_writer_prompt",
    "w1_reduce_prompt",
    "w1_section_prompt",
    "w2_blueprint_prompt",
    "w4_course_prompt",
]
