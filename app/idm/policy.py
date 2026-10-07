"""IDM pipeline identifiers, budgets and methodology word lists.

Every number that shapes IDM behaviour lives here with the reason it exists, so
stage modules never carry unexplained literals (STD-6).
"""

from __future__ import annotations

from typing import Final

IDM_PIPELINE_VERSION: Final = "idm-1"
LEGACY_PIPELINE_VERSION: Final = "v2-legacy"
IDM_CONTRACT_VERSION: Final = 1
IDM_PROMPT_POLICY_VERSION: Final = "idm-prompt-1"
IDM_SCOPE_KEY_PREFIX: Final = "idmcb_"
IDM_SCOPE_KEY_HEX_CHARS: Final = 32

# --- W1 sectioning (spec §5.5, §7.1) -------------------------------------------------
# A section is the unit of one W1-map call. The target keeps prompts well inside the
# model's attention window; the hard maximum is never exceeded.
IDM_W1_SECTION_TARGET_CHARS: Final = 18_000
IDM_W1_SECTION_MAX_CHARS: Final = 24_000
IDM_W1_SECTION_MAX_FACTS: Final = 320
# A heading only starts a new section once the current one carries real content.
IDM_W1_SECTION_MIN_CHARS_BEFORE_HEADING_CUT: Final = 4_000
IDM_W1_MAX_SECTIONS: Final = 16
IDM_SINGLE_TASK_MAX_SOURCE_CHARS: Final = 384_000
IDM_W1_PARALLELISM: Final = 4

# --- Output budgets per provider call (tokens) ----------------------------------------
IDM_W1_MAP_MAX_OUTPUT_TOKENS: Final = 6_000
IDM_W1_REDUCE_MAX_OUTPUT_TOKENS: Final = 8_000
IDM_W2_MAX_OUTPUT_TOKENS: Final = 12_000
IDM_W4_COURSE_MAX_OUTPUT_TOKENS: Final = 8_000
IDM_MODULE_MAX_OUTPUT_TOKENS: Final = 24_000
IDM_UNIT_WRITER_MAX_OUTPUT_TOKENS: Final = 16_000
IDM_JUDGE_MAX_OUTPUT_TOKENS: Final = 3_000

# --- Course-shape limits -----------------------------------------------------------------
IDM_MAX_BLOCKS: Final = 400
IDM_MAX_LESSONS: Final = 120
IDM_MAX_MODULES: Final = 24
IDM_MAX_LESSONS_PER_MODULE: Final = 30
# Upper bound of blocks in one lesson (IdmLessonPlanV1.block_ids).
IDM_MAX_LESSON_BLOCKS: Final = 40
IDM_SHARD_MAX_SOURCE_CHARS: Final = 90_000
# W2 is split per learning objective above this many blocks to bound one prompt.
IDM_W2_SINGLE_CALL_MAX_BLOCKS: Final = 200

# --- Runtime guards ------------------------------------------------------------------------
# A provider call is not started when less than this remains before the deadline:
# the call could not finish and its tokens would be wasted.
IDM_MIN_CALL_REMAINING_SECONDS: Final = 8.0
# A provider call must end this long before the task deadline (response + Node settlement).
IDM_CALL_HEADROOM_SECONDS: Final = 5.0
IDM_PROVIDER_CALL_TIMEOUT_MS: Final = 180_000
# Rough characters-per-token for budget pre-checks (Vietnamese text tokenises at ~3-4).
IDM_CHARS_PER_TOKEN_ESTIMATE: Final = 3.5
IDM_MAX_ATTEMPT_TRACE_EVENTS: Final = 64
# The judge is skipped when less than this remains (spec §7.7.5).
IDM_JUDGE_MIN_REMAINING_SECONDS: Final = 25.0

# --- Thinking levels per stage (spec §5.5) -------------------------------------------------
THINKING_W1_MAP: Final = "low"
THINKING_W1_REDUCE: Final = "high"
THINKING_W2: Final = "high"
THINKING_W4_COURSE: Final = "high"
THINKING_MODULE: Final = "high"
THINKING_W5: Final = "medium"
THINKING_W6: Final = "low"

# --- Validation thresholds -----------------------------------------------------------------
IDM_LO_MIN_COUNT: Final = 3
IDM_LO_MAX_COUNT: Final = 8
IDM_BLOCK_TOO_SMALL_CHARS: Final = 60
IDM_SME_QUESTION_MIN_CHARS: Final = 25
IDM_DUPLICATE_JACCARD_THRESHOLD: Final = 0.85
IDM_DUPLICATE_NGRAM: Final = 5
IDM_HEADING_MAX_CHARS: Final = 120
IDM_HEADING_ALL_CAPS_RATIO: Final = 0.6
IDM_STEP_RUN_MIN_LINES: Final = 3
IDM_DURATION_OVER_TARGET_RATIO: Final = 1.2
IDM_FALLBACK_SECTION_SHARE_FOR_STAGE_FALLBACK: Final = 0.3
IDM_FALLBACK_STAGES_FOR_REVIEW: Final = 2
IDM_THEORY_RUN_MAX_COMPONENTS: Final = 5
IDM_FALLBACK_SUMMARY_CHARS: Final = 300
IDM_FALLBACK_NAME_CHARS: Final = 80
IDM_FALLBACK_LESSON_MINUTES: Final = 5
IDM_FALLBACK_LESSON_SCREENS: Final = 4
IDM_PROCEDURE_SHARE: Final = 0.5
IDM_TABLE_SHARE: Final = 0.5
IDM_PAGE_FURNITURE_MIN_REPEATS: Final = 3

# --- Notes limits (spec §8.4) ----------------------------------------------------------------
IDM_NOTES_COURSE_MAX_CHARS: Final = 7_000
IDM_NOTES_MODULE_MAX_CHARS: Final = 3_000
IDM_NOTES_LESSON_MAX_CHARS: Final = 2_000
IDM_NOTES_MAX_HOLD_ITEMS: Final = 15
IDM_NOTES_MAX_SME_QUESTIONS: Final = 10
IDM_UNIT_AUTHOR_NOTE_MAX_CHARS: Final = 1_500

# --- Methodology word lists (folded: lower case, no diacritics, d for đ) ----------------
GENERIC_TITLES: Final = frozenset({
    "gioi thieu", "tong quan", "thong tin chung", "noi dung", "cac van de khac", "khac",
    "introduction", "overview", "general information", "content", "other", "miscellaneous",
})
GENERIC_TITLE_PATTERN: Final = r"^(phan|part|muc|section)\s*\d+$"
UNMEASURABLE_VERBS: Final = frozenset({
    "hieu", "biet", "nam duoc", "nam ro", "lam quen", "hieu ro",
    "understand", "know", "learn", "be aware", "appreciate",
})
MUST_DO_TOPIC_PREFIXES: Final = ("kien thuc ve", "tong quan", "gioi thieu", "overview of", "introduction to")
CAPABILITY_LEAD_INS: Final = ("co the", "can", "will be able to", "be able to")
BROAD_SME_QUESTION_PATTERN: Final = r"^(phan nao|noi dung nao) (quan trong|can thiet)"
AI_DRAFTED_MARKER_VI: Final = "[AI soạn — cần SME xác nhận]"
AI_DRAFTED_MARKER_EN: Final = "[AI-drafted — SME to confirm]"

ALLOWED_COMPONENT_TYPES_IDM: Final = ("html", "problem", "la_faq", "la_sortable", "la_crossword", "la_diagram")
MAX_COMPONENTS_PER_UNIT: Final = 4
# Assessment obligations exist only for slots 1..3 (SQL CHECK, spec §4.3 / IDM-0.5).
MAX_OBLIGATION_COMPONENT_INDEX: Final = 3

# Output budgets for the W5 writer (spec §7.7.3).
SEGMENT_WORD_BUDGET: Final = {
    "context_explain": 450,
    "example": 350,
    "practice_feedback": 400,
    "summary_apply": 250,
    "job_aid": 600,
}
EXTRA_WORDS_PER_MUST_KNOW_BLOCK: Final = 120
MIN_GENERATED_WORDS: Final = 250
MAX_GENERATED_WORDS: Final = 1_600
VISIBLE_CHARS_PER_WORD: Final = 7
