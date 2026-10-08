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
# Gemini counts thinking tokens against max_output_tokens. QC course 234653 (2026-10-08):
# every W1-reduce/W2/W4 call of two runs spent exactly its cap (8k/12k/8k with thinking
# "high") and returned cut JSON, so the course always fell back. A complete W1-reduce
# answer for 32 blocks x 6 objectives is ~5k JSON tokens, a W2 answer ~5k, a W4 answer
# ~3k; the caps leave room for thinking plus a margin, and module calls (24k, also cut)
# double. The task allowance Node grants (65,536 x 2 output tokens per course_skeleton or
# chapter task) still bounds the sum: later stages only reserve their minimum admissible
# share (half the cap, see ``idm_tail_reserve``), and the runtime shrinks a call to what
# is left. The unit writer keeps 16k: no unit answer reached it.
IDM_W1_MAP_MAX_OUTPUT_TOKENS: Final = 12_000
IDM_W1_REDUCE_MAX_OUTPUT_TOKENS: Final = 24_000
IDM_W2_MAX_OUTPUT_TOKENS: Final = 32_000
IDM_W4_COURSE_MAX_OUTPUT_TOKENS: Final = 16_000
IDM_MODULE_MAX_OUTPUT_TOKENS: Final = 48_000
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

# --- Thinking levels per stage (spec §5.5, revised after QC course 234653) ---------------
# W1-reduce, W2 and W4 classify and order a compact block catalog; "high" thinking used the
# whole output cap before any JSON was written. "medium" keeps the reasoning (one objective
# per action, Must Do vs topic, keep/remove per block) and leaves the cap for the answer.
# The module stage designs practice and layout for several lessons in one answer (10-20k
# JSON tokens); with "high" it reached its cap too, and a 48k answer at "high" would also
# risk the provider call timeout, so it uses "medium" as well.
THINKING_W1_MAP: Final = "low"
THINKING_W1_REDUCE: Final = "medium"
THINKING_W2: Final = "medium"
THINKING_W4_COURSE: Final = "medium"
THINKING_MODULE: Final = "medium"
THINKING_W5: Final = "medium"
THINKING_W6: Final = "low"
# The repair of an answer cut at max_output_tokens thinks less and asks for a shorter answer;
# re-sending the same prompt at the same level would be cut again.
THINKING_AFTER_TRUNCATION: Final = "low"

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
# Nice to Know blocks stay out of the lessons (methodology W2); the author gets their names and a
# one-line summary to add one back by hand (QC course 234653, R4). The full list is in the UI panel.
IDM_NOTES_MAX_NICE_TO_KNOW: Final = 12
IDM_NOTES_SUMMARY_CHARS: Final = 160
IDM_UNIT_AUTHOR_NOTE_MAX_CHARS: Final = 1_500

# --- FAQ evidence guard (QC course 234653, R6) ------------------------------------------------
# An la_faq answer must restate what the facts given to the writer say. Calibrated on the 18 FAQ
# answers of run e869f43a against the unit's facts: answers that only rephrase the source have
# >= 0.18 of their content word-pairs in the facts; answers that add advice, reasons or examples
# the source never states ("dừng ở tầng thứ 3 hoặc 7", "đội ngũ liên chức năng không cần tuyển
# thêm người") have <= 0.12. Function words never count (short connective phrasing is free).
IDM_FAQ_MIN_PAIR_SUPPORT: Final = 0.15
# A short answer can rephrase every word without keeping a single word pair.
IDM_FAQ_MIN_WORD_SUPPORT: Final = 0.6
# One invented sentence inside an otherwise grounded answer: a sentence with at least this many
# content words needs some support of its own.
IDM_FAQ_SENTENCE_MIN_WORDS: Final = 8
IDM_FAQ_SENTENCE_MIN_PAIR_SUPPORT: Final = 0.08
IDM_FAQ_SENTENCE_MIN_WORD_SUPPORT: Final = 0.45
# The shared payload validator needs two FAQ items; ungrounded items are only dropped above it.
IDM_FAQ_MIN_ITEMS: Final = 2
# Share of a worksheet's self-check items that must restate the facts (same test as an FAQ answer).
IDM_WORKSHEET_MIN_GROUNDED_SHARE: Final = 0.5
IDM_WORKSHEET_MIN_LIST_ITEMS: Final = 2

# --- Single-choice hygiene (QC course 364564, N2/N3) ------------------------------------------
# The correct option was the longest in 8 of 9 questions. Longer than every other option by more
# than this factor (characters), it is a length cue learners can guess from: a review note, never
# a rejection (a precise correct answer is often a little longer).
IDM_MCQ_LENGTH_CUE_RATIO: Final = 1.25
# Answer leak: an option whose word 4-grams (accent-folded) mostly appear in the html shown before
# the question in the same unit copies it ("ví dụ đạt chuẩn" -> correct option, "ví dụ không đạt
# chuẩn" -> distractor; QC c4.l2.u1 was a near-verbatim copy, ~1.0). A rule fragment shared with a
# scenario option ("liên quan đến an toàn") scores ~0.4, the rule restated in other words ~0.57, and
# an example that inserts a clause into the answer ("Cấp 2: thông báo trưởng nhóm vì khách hàng phàn
# nàn lần thứ hai", the backend acceptance fixture the Node test accepts without repair) 0.71: only a
# near-verbatim copy (>= 3/4 of the 4-grams) is a leak. Options under the minimum number of 4-grams
# (about 7 words) are too short to tell a copy from a term.
IDM_ANSWER_LEAK_NGRAM: Final = 4
IDM_ANSWER_LEAK_MIN_NGRAMS: Final = 4
IDM_ANSWER_LEAK_MIN_SHARE: Final = 0.75

# --- FAQ value and callout grounding (QC course 364564, N9/N11) --------------------------------
# 3 of 5 FAQs only repeated the table taught just above them. An FAQ answer restates the html before
# it in the unit when most of its word 4-grams (accent-folded, the answer-leak measure) appear in it:
# a table row restated as a sentence keeps ~0.6-0.8 of its 4-grams, an answer that adds a condition,
# an exception or a "what if" to a phrase of the row stays near 0.3. Answers under the minimum number
# of 4-grams (about 7 words) are too short to tell.
IDM_FAQ_RESTATE_NGRAM: Final = 4
IDM_FAQ_RESTATE_MIN_NGRAMS: Final = 4
IDM_FAQ_RESTATE_MIN_SHARE: Final = 0.6
# 2 FAQ titles ("Rào cản …", "… trong họp giao ban") did not match their questions. A title with at least
# this many key words (content words that are not FAQ boilerplate) must share at least half of them
# with its questions and answers.
IDM_FAQ_TITLE_MIN_KEY_WORDS: Final = 2
IDM_FAQ_TITLE_MIN_SHARED_SHARE: Final = 0.5
# Folded FAQ boilerplate a title may carry without the items repeating it.
FAQ_TITLE_GENERIC_WORDS: Final = frozenset({
    "cau", "hoi", "thuong", "gap", "dap", "giai", "faq", "luu", "nham", "lan", "thac", "mac", "hieu", "sai",
    "lam", "tuong", "diem", "chu", "quan", "trong", "van", "de",
    "frequently", "asked", "questions", "question", "common", "mistakes", "mistake", "misconceptions",
    "misconception", "notes", "note", "clarifications", "answers", "doubts", "faqs",
})

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
# No V2 component records a free-text or form answer (component registry: problem is single choice
# only, la_scenario_chat is manual-only). The practice of a Must Do of kind "do" is therefore a
# worksheet (spec §10.1 "Tạo sản phẩm đầu ra"): an html slot with role practice (task, template,
# worked example, self-check list from the criteria facts) checked by a single-choice problem on a
# sample answer; ordered procedures keep la_sortable.
WORKSHEET_COMPONENT_TYPE: Final = "html"
GRADED_PRACTICE_TYPES: Final = frozenset({"problem", "la_sortable", "la_crossword"})
DOING_PRACTICE_TYPES: Final = frozenset({WORKSHEET_COMPONENT_TYPE, "la_sortable"})
# A "do" Must Do that classifies, identifies or chooses applies a rule to a case: spec §10.1 keeps a
# single-choice problem for it ("Kiểm tra hiểu Must Know (nhận diện, phân loại)", "chọn bước tiếp,
# quyết định"). Folded leading verbs.
CASE_DECISION_VERBS: Final = (
    "phan loai", "xac dinh", "nhan dien", "nhan biet", "lua chon", "chon", "quyet dinh", "danh gia",
    "classify", "identify", "recognise", "recognize", "select", "choose", "decide", "evaluate", "assess",
)
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
