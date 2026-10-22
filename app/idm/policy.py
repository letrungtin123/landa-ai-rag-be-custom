"""IDM pipeline identifiers, budgets and methodology word lists.

Every number that shapes IDM behaviour lives here with the reason it exists, so
stage modules never carry unexplained literals (STD-6).
"""

from __future__ import annotations

from typing import Final

IDM_PIPELINE_VERSION: Final = "idm-1"
LEGACY_PIPELINE_VERSION: Final = "v2-legacy"
# 2 (QLT-5): project_context.course_description, IdmCourseDesignV1.purpose, the unit brief's purpose fields and the
# module context's purpose. Node reads stored contract-1 designs unchanged. QLT-5 C adds, still in contract 2 (not
# released yet), the optional purpose.fidelity_mode and unit brief fidelity_mode and the "explain" treatment.
# 3 (QLT-6): the launch brief in project_context (design_mode, learner_audience, course_purpose), the "author"
# audience origin, the "purpose" purpose evidence and the unit brief's design_mode. Node reads contracts 1-2 unchanged.
IDM_CONTRACT_VERSION: Final = 3
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

# The W4 module stage makes two content attempts (writer + one repair) and, after an answer that only failed
# the strict schema (a bound, a type), at most one schema repair that does not use a content attempt (QC run
# 8de1c76b, Q1b). Worst case 3 calls instead of 2; each is admitted only inside the task deadline.
IDM_MODULE_CONTENT_ATTEMPTS: Final = 2
IDM_MODULE_SCHEMA_REPAIRS: Final = 1

# --- Course-shape limits -----------------------------------------------------------------
IDM_MAX_BLOCKS: Final = 400
IDM_MAX_LESSONS: Final = 120
IDM_MAX_MODULES: Final = 24
IDM_MAX_LESSONS_PER_MODULE: Final = 30
# Upper bound of blocks in one lesson (IdmLessonPlanV1.block_ids).
IDM_MAX_LESSON_BLOCKS: Final = 40
# Bounds of IdmPracticeTaskV1.criteria_fact_keys and IdmComponentDesignV1/IdmUnitDesignV1.block_ids; the Node
# contract (``lesson-author-idm.contract.ts``) reads the same bounds, so they cannot be raised on one side. QC run
# 8de1c76b (Q1): one practice of the 33-fact Canvas block listed all 33 facts and the whole W4 answer was rejected;
# an over-long list is now cut to the facts that matter most (``module_autofix.trim_module_answer``).
IDM_PRACTICE_MAX_CRITERIA_FACTS: Final = 24
IDM_COMPONENT_MAX_BLOCKS: Final = 12
IDM_UNIT_MAX_BLOCKS: Final = 24
# A criteria fact this short (words) is a label or heading ("MINDSET CẦN BỎ"): it ranks after the statements of
# the same block when a list must be cut.
IDM_CRITERIA_LABEL_MAX_WORDS: Final = 4
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
# --- Unit time budget (QC run ab8d67e1, R3) ------------------------------------------------------------------------
# Node bounds a unit request to 120 s (IDM_UNIT_SOFT_DEADLINE, minus 5 s of response headroom here). In the run the
# medium-thinking writer took 10-78.5 s (p90 ~70 s, up to ~9k thinking tokens on question units), medium repairs
# 26-73 s, low-thinking repairs ~7 s and judges 6-12 s: two leak repairs started at medium with ~36 s left and were
# cut (AI_PROVIDER_TIMEOUT), and 4 units went unjudged. Each step now gets the thinking level its p90 time leaves
# room for, the judge included; the numbers are upper estimates with margin, not targets.
IDM_W5_WRITER_MEDIUM_SECONDS: Final = 80.0
IDM_W5_REPAIR_MEDIUM_SECONDS: Final = 75.0
IDM_W5_REPAIR_LOW_SECONDS: Final = 20.0
# A W5 repair whose findings a deterministic step settles anyway (a framework the html does not list) is a targeted
# repair at low thinking, started only with this much time left: QC run 8de1c76b (Q3) spent 57 s on the writer of
# "Bản đồ 5 chuyển dịch" and its medium-thinking repair was cut at 52.6 s by the 120 s unit deadline. Below the bound
# the deterministic step runs at once, so the extra latency of a targeted repair is at most one low-thinking call.
IDM_TARGETED_REPAIR_MIN_SECONDS: Final = 30.0
# Owner P6 (2026-10-20): the QC analysis of that day counted 6 fallbacks after one repair call that failed transiently
# (provider timeout or unavailable). Such a repair is sent once more when at least this much time is left.
IDM_REPAIR_RETRY_MIN_SECONDS: Final = 30.0

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
THINKING_TARGETED_REPAIR: Final = "low"

# --- Validation thresholds -----------------------------------------------------------------
IDM_LO_MIN_COUNT: Final = 3
IDM_LO_MAX_COUNT: Final = 8
IDM_BLOCK_TOO_SMALL_CHARS: Final = 60
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
# Owner decision 2026-10-22: no question for an SME anywhere; ``payload.idm.blocks[].sme_questions`` and
# ``payload.idm.hold_items`` stay in contract 3 and are always empty.
# One-line summary of a removed block in the course note. The Nice to Know list (QC course 234653, R4) is gone since the
# QC feedback of 2026-10-21 (A6): such blocks are stored and listed as Remove.
IDM_NOTES_SUMMARY_CHARS: Final = 160
# Review m6: the points to check in the source (owner decision 2026-10-22) take at most this much of the course note
# and of the unit note, so the sections and lines after them keep their room.
IDM_NOTES_SOURCE_CHECKS_MAX_CHARS: Final = 1_500
IDM_UNIT_SOURCE_FLAGS_MAX_CHARS: Final = 400
# Run 5496155c: a source page is an image page the system could not read when its unobserved images cover at least
# this share of it (``app.idm.visual_pages``); a picture beside slide text covered 12-37 %, the picture pages 41-99 %.
IDM_IMAGE_PAGE_MIN_SHARE: Final = 0.4
# Removed blocks with the W2 reason, and definition blocks kept against the W2 proposal (QC course 364564, N4).
IDM_NOTES_MAX_REMOVED: Final = 12
IDM_UNIT_AUTHOR_NOTE_MAX_CHARS: Final = 1_500

# --- Evidence guard of callouts and worksheet checklists (QC course 234653, R6) ------------------
# A callout or a self-check item must restate what the facts given to the writer say (until QLT-5 D an la_faq answer
# too; an FAQ answer now explains and is grounded in its own basis facts, and the measure below only marks it as
# inferred for the SME). Calibrated on the 18 FAQ answers of run e869f43a against the unit's facts: answers that only
# rephrase the source have
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
# The shared payload validator needs two FAQ items; items are only dropped above it.
IDM_FAQ_MIN_ITEMS: Final = 2
# Share of a worksheet's self-check items that must restate the facts (same test as an FAQ answer).
IDM_WORKSHEET_MIN_GROUNDED_SHARE: Final = 0.5
IDM_WORKSHEET_MIN_LIST_ITEMS: Final = 2

# --- Single-choice hygiene (QC course 364564, N2/N3) ------------------------------------------
# The correct option was the longest in 8 of 9 questions (and in 10 of 13 in QC run fae66524, QLT-7, where only the
# two keys over 1.25x the longest other option were flagged). A key longer than this factor times the AVERAGE length
# of the distractors (characters), or longer than the longest distractor by the second factor (with 3+ options), is a
# length cue learners can guess from: one targeted repair that balances the lengths, then a review note, never a
# rejection (a precise correct answer is often a little longer). Measured on the 13 questions of fae66524 (key / mean
# distractor, key / longest distractor): 1.38/1.31, 1.32/1.27, 1.27/1.20 and 1.23/1.204 are cues (4, the old rule
# found 2); the next one, 1.20/1.11, is not.
IDM_MCQ_LENGTH_CUE_RATIO: Final = 1.25
IDM_MCQ_LENGTH_CUE_LONGEST_RATIO: Final = 1.2
# The length the repair aims for: the key at most this factor times the average distractor (inside the cue bound).
IDM_MCQ_LENGTH_TARGET_RATIO: Final = 1.1

# --- Quiz text (QC feedback 2026-10-21, A1/A2; contract K1; ``app.idm.quiz_text``) ----------------------------------
# The QC department found the explanations rambling and hollow and the stems a wall of text. Measured on the 25
# questions of runs 8de1c76b and ab8d67e1: the principle before "A - …" took 36-61 words where there was one (8 of 16
# had none), an option's reason 16-83 words (median 28), a whole explanation 391-887 characters; stems ran 11-92 words
# in 1-4 sentences. The shape asked for: the principle in one line (one or two short sentences), then one line per
# option "A - Đúng vì …" / "A - Sai vì …" of one sentence; Vietnamese words count as syllables.
IDM_MCQ_EXPLANATION_PRINCIPLE_MAX_WORDS: Final = 35
IDM_MCQ_EXPLANATION_PRINCIPLE_MAX_SENTENCES: Final = 2
IDM_MCQ_EXPLANATION_OPTION_MAX_WORDS: Final = 35
IDM_MCQ_EXPLANATION_OPTION_MAX_SENTENCES: Final = 1
# Well inside the 1,500 characters the schema and Node accept (4 options x 35 words + the principle ~ 850).
IDM_MCQ_EXPLANATION_MAX_CHARS: Final = 1_000
# An option line restates its option when it holds at least this share of the option's word 3-grams (accent-folded);
# options of fewer words are too short to tell. A reason that names the criterion the option also names ("Sai vì cấp 1
# chỉ dành cho lần đầu, chưa có thiệt hại tài chính" for "Cấp 1 vì chưa có thiệt hại tài chính") holds about 0.57; the
# QC example copied its option almost whole.
IDM_MCQ_EXPLANATION_RESTATE_NGRAM: Final = 3
IDM_MCQ_EXPLANATION_RESTATE_SHARE: Final = 0.7
IDM_MCQ_EXPLANATION_RESTATE_MIN_WORDS: Final = 8
# The stem: at most two context sentences (list lines and the question line not counted), each list item on its own
# line starting "- ", the question alone on the last line. Owner decision 2026-10-22 (run dfe78482: three stems of
# 132-189 words, a short context, 4-5 scenario items and the question, were marked "Cần xem"): the readability cap
# counts the parts apart: the context and the question together at most this many words, at most this many listed
# items of at most this many words each. Over it is an edit note, never a review mark.
IDM_MCQ_STEM_CONTEXT_MAX_SENTENCES: Final = 2
IDM_MCQ_STEM_PROSE_MAX_WORDS: Final = 80
IDM_MCQ_STEM_MAX_ITEMS: Final = 6
IDM_MCQ_STEM_ITEM_MAX_WORDS: Final = 40
# Node's generation normaliser cuts a question at 1,000 and an explanation at 1,500 characters (a line feed counts one):
# the question's hard limit (``quiz_text.QUESTION_LIMIT_CODE``, blocking); the explanation's cap stays well below.
# Those Node limits: the layout never makes a text longer than them (it keeps the writer's text instead).
NODE_QUESTION_MAX_CHARS: Final = 1_000
NODE_EXPLANATION_MAX_CHARS: Final = 1_500
# Empty praise and slogans that give no reason ("B - Đúng vì phương án này dũng cảm Say No … tạo điểm khác biệt tuyệt
# đối", QC 2026-10-21). Matched as whole words, Vietnamese with its diacritics (NFC, lower case); a phrase the unit's
# facts use themselves (a rule that says "tuyệt đối không …") is allowed.
# Only true filler (review m10): a word a lesson may use for what it means ("xuất sắc" as a rating, a "superior" as
# the learner's boss, an "outstanding" balance, an "excellent" grade, an "absolute" value) is not on the lists.
QUIZ_FILLER_VI: Final = (
    "dũng cảm", "tuyệt đối", "vượt trội", "say no", "đột phá", "bứt phá", "tối thượng", "hoàn hảo",
    "đỉnh cao", "vô cùng", "cực kỳ", "cực kì", "thần kỳ", "siêu việt", "không thể phủ nhận", "hiển nhiên",
    "tất nhiên", "rõ ràng là", "khác biệt tuyệt đối", "chiến lược khôn ngoan", "nước đi thông minh",
)
QUIZ_FILLER_EN: Final = (
    "brave", "bravely", "boldly", "absolutely", "breakthrough", "game-changer", "game changer", "perfectly",
    "amazing", "incredible", "obviously", "of course", "undeniably", "say no", "smart move", "wise choice",
)
# Run 5496155c (owner decision 2026-10-22): hype words are banned in quiz explanations and FAQ answers even when the
# source uses them (one course repeated "vượt trội" 9 times, "tuyệt đối" 8, "dũng cảm" 6, "triệt để" 6, "sinh tử" 4,
# "đập tan" 4). Only inside a quoted name of a source concept ("chiến lược “Say No”": at most
# ``IDM_HYPE_NAME_MAX_WORDS`` words the facts write in that order) they stay.
HYPE_VI: Final = ("sinh tử", "dũng cảm", "đập tan", "triệt để", "khổng lồ", "tuyệt đối", "vượt trội", "say no")
HYPE_EN: Final = ("life-or-death", "brave", "bravely", "absolutely")
IDM_HYPE_NAME_MAX_WORDS: Final = 6
# QC feedback 2026-10-21 (A3): a callout (a "warning" block, shown as a quotation) adds a point; one that only repeats
# the paragraphs before it in its section goes (``app.idm.qa_checks.repeated_callouts``). Review M3 (2026-10-22): a
# callout is a repeat only when it states no number and no negation those paragraphs do not ("cấp 3 … 24 giờ" under
# "cấp 2 … 48 giờ", "Không áp dụng …" under "áp dụng …" are new points) and at least this share of its accent-folded
# word trigrams appear in them (trigrams never span two pieces). Callouts of fewer words are not judged. "Không hứa bồi
# thường cho khách trước khi phân loại mức độ khiếu nại." under "Nhân viên tuyệt đối không hứa bồi thường cho khách
# trước khi phân loại mức độ khiếu nại." holds all its trigrams; "Không hứa bồi thường trước khi phân loại mức độ." 6
# of 8, so it stays.
# QC feedback 2026-10-21 (A4): the context_description of a suggested infographic or video is a designer brief: the
# layout, the text and numbers on the image (or screen) and one element per content point, in at least this many
# words (``app.idm.media_detail``; the contract allows 1000 characters). A thinner one is completed by the server from
# the content points' own words, each quoted with at most this many words.
IDM_MEDIA_CONTEXT_MIN_WORDS: Final = 40
# Owner decision 2026-10-22 (run dfe78482): a description of this many words or more is never completed by the server,
# nor one that numbers its frames or scenes; a point is quoted whole when it has at most the label words, else named.
IDM_MEDIA_CONTEXT_DETAILED_WORDS: Final = 80
IDM_MEDIA_POINT_LABEL_MAX_WORDS: Final = 24
IDM_MEDIA_CONTEXT_MAX_CHARS: Final = 1000
IDM_CALLOUT_REPEAT_NGRAM: Final = 3
IDM_CALLOUT_REPEAT_SHARE: Final = 0.8
IDM_CALLOUT_REPEAT_MIN_WORDS: Final = 4
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
# A reworded leak (QC run 8de1c76b, Q4) keeps few 4-grams (0.28, 0.51, 0.12) but the key's distinctive wording
# still comes from the html before the question: its content word pairs ("cam hung nhat thoi", "dong goi phang")
# that no distractor and not the question use. Measured on the 9 questions of the run (pairs of two content
# words): the 3 leaks found 4/4, 19/27 and 11/22 of them in the html (0.50-1.00), the distractors at most 0.67,
# 0.00 and 0.25 (margin >= 0.25); a fourth hit (the worksheet check c3.l1.u1, 12/20 vs 0.20) copies the
# template's confirmation sentence and is a leak too. The non-leaks scored <= 0.47 or a negative margin; the
# taught rule restated in the key of the golden fixture ("Cấp 2 vì khách hàng phàn nàn lần thứ hai") has only 3
# such pairs after the question's own wording, under the minimum, so applying a rule to a case is not a leak.
IDM_ANSWER_LEAK_PAIR_MIN_HITS: Final = 4
IDM_ANSWER_LEAK_PAIR_MIN_SHARE: Final = 0.5
IDM_ANSWER_LEAK_PAIR_MIN_MARGIN: Final = 0.2
# A key that paraphrases ONE passage shown before it (a quotation sentence, a list item, a table row) keeps that
# passage's words in order even when few of its word pairs are distinctive (QC run ab8d67e1, R2: c4.l1.u1's key
# "Doanh nghiệp tất yếu sẽ bị thị trường đào thải và áp đặt sự thay đổi theo cách khắc nghiệt nhất" restates the
# blockquote "… doanh nghiệp sẽ bị thị trường áp đặt thay đổi theo cách tàn nhẫn nhất": 13 of its 21 words in order,
# 0.62, the distractors at most 0.24). Measured on the 16 questions of that run and the 9 of run 8de1c76b: the other
# keys reached at most 0.52 with a margin of at most 0.15 (options that each restate a table row score alike).
# Keys that name a taught label ("Cấp 2 vì …", "Bậc 01 - …") restate their rule by design and are not measured.
IDM_ANSWER_QUOTE_MIN_KEY_WORDS: Final = 8
IDM_ANSWER_QUOTE_MIN_SHARE: Final = 0.5
IDM_ANSWER_QUOTE_MIN_MARGIN: Final = 0.25
# A short key (at most this many words) that the html shows word for word while no distractor appears there
# (at most half of its words in order) is the only familiar option: a cue, not a test of the criterion.
IDM_ANSWER_SHORT_KEY_MAX_WORDS: Final = 4
IDM_ANSWER_SHORT_DISTRACTOR_MAX_SHARE: Final = 0.5
# The question's case copies an example passage that holds the key (and no distractor): >= this share of the
# passage's words in order. Highest share in the two runs: 0.16.
IDM_ANSWER_STEM_COPY_MIN_SHARE: Final = 0.6
IDM_ANSWER_STEM_PASSAGE_MIN_WORDS: Final = 6
# Contested key (QC run ab8d67e1, R2: c1.l2.u2's key "Bậc 01" while the case "đạt ISO 9001 và CE, tỷ lệ lỗi dưới
# ngưỡng" is the unit's own row for Bậc 02 and its FAQ says "ISO, CE, FDA chỉ giúp đạt Bậc 02"): when the options
# name labels the unit teaches (table rows, "label: …" list items), the case must meet the key's criteria (row value,
# FAQ sentences naming only that label) better than any distractor's. In the run the case shared 8 content words
# with the key's criteria and 15 with Bậc 02's; the case of a well-formed ladder question of run 8de1c76b (An
# Phát, Bậc 01) shares 9 with its key's row and at most 4 with another.
IDM_ANSWER_CONTESTED_MIN_WORDS: Final = 4
# QC run fae66524 (QLT-7): c1.l1.u2's case "đạt chứng nhận ISO 9001 và chứng chỉ CE" was keyed "Bậc 01" while the
# table row of Bậc 02 says "đạt chứng chỉ quốc tế như ISO, CE hoặc FDA"; the case was written to share more words
# with Bậc 01 (9) than with Bậc 02 (8), so the word count above did not fire. A case also contests its key when it
# literally meets a criterion only another label's row states: at least this many distinctive codes of that row
# (ISO, CE: all-capital tokens no other row has) in the case, not negated ("chưa có ISO"), or one code and this many
# distinctive content-word pairs, or the pairs alone. On a numbered ladder ("Bậc 01" ... "Bậc 06") only a HIGHER rung
# contests: a higher rung's case meets the lower rungs' criteria by design.
IDM_ANSWER_CONTESTED_MIN_CODES: Final = 2
IDM_ANSWER_CONTESTED_MIN_PAIRS: Final = 3
# Answer leak through the unit's own worked example (QC run fae66524, c2.l3.u1): the worksheet's "Ví dụ minh hoạ"
# filled the field with "Cung cấp video hướng dẫn cá nhân hóa và gửi quà tri ân bất ngờ" and the key read "Chủ động
# gửi hướng dẫn cá nhân hóa và dịch vụ tri ân bất ngờ …": 10 of the example's 15 words in order (0.67), every
# distractor at most 0.13. A key that holds at least this share of one example passage (in order) while no distractor
# holds more than the second share gives the answer away. Passages under ``IDM_ANSWER_STEM_PASSAGE_MIN_WORDS`` words
# are too short to tell.
IDM_ANSWER_EXAMPLE_MIN_SHARE: Final = 0.6
IDM_ANSWER_EXAMPLE_MAX_DISTRACTOR_SHARE: Final = 0.35
# QLT-8 (QC run 0a11aa9e): two keys gave themselves away by joining SEVERAL items the unit had just shown, which the
# one-passage measures above never see. c1.l1.u1's key "… chất lượng đỉnh cao, uy tín tuyệt đối, giá trị gia tăng lớn,
# đổi mới liên tục và chuẩn mực toàn cầu" strings the heads of the five-item list together: 0.72 of its content-word
# pairs come from that one list block, the best distractor 0.21. c2.l5.u1's key "… ký hợp đồng cố vấn chuyên gia bên
# ngoài, thuê nền tảng AI/Cloud sẵn có và bắt tay với đối tác phân phối" joins three rows of the worksheet's worked
# example: 0.38 of its pairs come from the example's passages, the best distractor 0.16. Measured on the 100 questions
# of 12 runs (2026-10-09 .. 2026-10-15): with the list or table read whole, a key must hold at least the first share,
# beat every distractor by the margin and touch at least two of the block's items (a key that restates ONE item is
# the one-passage measure's case; a criterion named by a key of an application question is one item); with the worked
# example read whole, the second share and margin over at least two of its passages. Keys that open with a taught
# label apply its rule by design and are not measured. In the corpus only the two keys above and one more pass the
# bounds: fae66524 c4.l1.u2, a worksheet check whose key copies three lines of the template (0.60 against 0.19).
IDM_ANSWER_LIST_MIN_SHARE: Final = 0.6
IDM_ANSWER_LIST_MIN_MARGIN: Final = 0.35
IDM_ANSWER_EXAMPLE_BAG_MIN_SHARE: Final = 0.35
IDM_ANSWER_EXAMPLE_BAG_MIN_MARGIN: Final = 0.2
# Items of the block (or example passages) the key must draw on, and the content-word pairs a key needs to be measured.
IDM_ANSWER_JOINED_MIN_ITEMS: Final = 2
IDM_ANSWER_JOINED_MIN_KEY_PAIRS: Final = 4
# QLT-9 (QC run ffaf7318): c3.l2.u2's key rewrote three rows of the worksheet example in other words ("liên minh với nền
# tảng Cloud có sẵn …" for "Thuê nền tảng Cloud và AI có sẵn …"), so its word PAIRS barely matched (0.32 against 0.18).
# Read as single content words the key holds 0.64 of its words from the example, the best distractor 0.43. On the 110
# questions of 13 runs a key holding at least this share of its words from the worked example (and the margin over
# every distractor, at least two passages touched, keys of at least the word count) is a leak in all 6 cases: the two
# keys above, 0a11aa9e c1.l2.u2, fae66524 c2.l1.u2 and c2.l4.u1, 6bdb51e3 c5.l1.u2. Read against the lists the same
# measure also flags application keys that name one criterion, so lists stay on word pairs.
IDM_ANSWER_EXAMPLE_WORDS_MIN_SHARE: Final = 0.5
IDM_ANSWER_EXAMPLE_WORDS_MIN_MARGIN: Final = 0.15
IDM_ANSWER_EXAMPLE_WORDS_MIN_KEY_WORDS: Final = 8
# Run 5496155c (QC 2026-10-22): the give-away sat in the stem's listed items, which the key-only measures above never
# read. c1.l2.u2's "Nhân sự 1: Trợ lý Tổng giám đốc tiếp nhận lịch hẹn và lọc hồ sơ dự thầu" holds 0.75 of its content
# words from the worked example right above it (whose answer, "Người gác cổng", the key names); c1.l3.u2's "Phản hồi 2:
# Mong muốn được vinh danh đúng tầm và kiến tạo di sản …" 0.55 of its content-word pairs from the table row whose label
# the key names; c3.l1.u2's key "Phương án 1" points to an item holding 0.62 of the example's pairs against 0.5 for the
# best other item. On the run's 14 questions (36 listed items) an item copies a passage when it holds at least the pair
# share (of at least this many pairs) or the word share (of at least this many words); the best other copy held 0.38 of
# the pairs and 0.58 of the words (c1.l3.u4, which the QC called a good question, 0.38 and 0.46). A key that is only an
# item's name is measured by its item's pairs, with the margin over every other item.
IDM_ANSWER_STEM_ITEM_MIN_PAIRS: Final = 3
IDM_ANSWER_STEM_ITEM_COPY_SHARE: Final = 0.5
IDM_ANSWER_STEM_ITEM_MIN_WORDS: Final = 4
IDM_ANSWER_STEM_ITEM_WORD_SHARE: Final = 0.7
IDM_ANSWER_STEM_ITEM_MARGIN: Final = 0.1

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

# --- FAQ that digs deeper (QLT-5 D, ``app.idm.faq_plan``, ``app.idm.faq_checks``) ----------------------------------
# Measured on 100 FAQ items of 6 runs (plan 2026-10-13 §2.3): 58% recalled a list or checked a yes/no fact, 5% asked
# "what if", about half of each answer's word pairs came straight from the source and 23% of the answers had a best
# single-fact LCS >= 0.6; an FAQ sat in 39 of 71 units. W4 now plans an FAQ only for a unit with something to clarify,
# with at most this many items, each built on at most this many anchor facts.
IDM_FAQ_PLAN_MAX_ITEMS: Final = 3
IDM_FAQ_ANCHOR_MAX_FACTS: Final = 3
# Plain recall questions ("là gì", "gồm những gì", "what is") allowed per FAQ (the QC target is <= 25% recall); an FAQ
# needs at least one question of a clarifying type.
IDM_FAQ_MAX_RECALL_ITEMS: Final = 1
FAQ_COMPONENT_TYPE: Final = "la_faq"
FAQ_DEEP_TYPES: Final = frozenset({"why", "what_if", "common_mistake"})
# An answer restates the source when ONE fact holds most of it: the longest common word subsequence (accent-folded)
# with one fact covers at least this share of the answer's words, or at least this share of the answer's word 5-grams
# are in one fact. Answers under the minimum number of words are too short to tell. The keep content mode (compliance)
# may restate its rules and is not measured.
IDM_FAQ_SOURCE_COPY_MIN_WORDS: Final = 8
IDM_FAQ_SOURCE_COPY_LCS_SHARE: Final = 0.6
IDM_FAQ_SOURCE_COPY_NGRAM: Final = 5
IDM_FAQ_SOURCE_COPY_NGRAM_SHARE: Final = 0.6
# QLT-7 (QC run fae66524: an FAQ in 11 of 19 units, the QC target 25-40%): a question of the unit's practice is no
# reason for an FAQ on its own, and an FAQ whose only reason is a W1 issue or gap (a confusion the AI infers) is kept
# in at most this many units per lesson.
IDM_FAQ_WEAK_SIGNAL_UNITS_PER_LESSON: Final = 1
# QLT-8 (QC run 0a11aa9e: an FAQ in 2 of 23 units, the QC target 25-40%): the W4 designer now places very few FAQs, so
# after the gate the server adds one to the learning units with the most to clarify until at least this share of a
# module's learning units has one: a strong signal first, then a W1 issue or gap, then the misconceptions behind the
# wrong options of the unit's own question; a unit whose lesson Must Do is a decision first; at most one added FAQ per
# lesson, lessons without an FAQ first. On the run's design this gives 6 of 23 units (26%).
IDM_FAQ_MIN_UNIT_SHARE: Final = 0.25
IDM_FAQ_FLOOR_PER_LESSON: Final = 1
# An FAQ question that asks an open SME question again (c1.l3.u1: "Doanh nghiệp có cần triển khai đồng loạt toàn bộ
# 9 công cụ di sản theo đúng thứ tự PDCA không?" against the open "Thứ tự triển khai 9 công cụ này có bắt buộc tuần
# tự theo PDCA không …?": 8 of its 15 content words): at least this many shared content words, and at least this
# share of the FAQ question's own. The other 24 FAQ questions of the run shared at most 3 words (0.22) with any open
# SME question of their unit.
IDM_FAQ_OPEN_QUESTION_MIN_SHARED_WORDS: Final = 4
IDM_FAQ_OPEN_QUESTION_MIN_SHARE: Final = 0.5
# An FAQ answer sentence that gives a reason ("vì", "nhằm", "giúp", "để" …) is marked inferred for the SME when no
# basis fact holds most of its content words: on the 25 answers of the run the whole-answer measure alone marked 3,
# with this one 14 (the QC read most "vì sao" answers as inference). The invented-reason check (a reason for a
# threshold its facts do not explain) flags only c4.l3.u1's "… tối thiểu 30% …" item of the run.
IDM_FAQ_REASON_SENTENCE_MIN_SUPPORT: Final = 0.5

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
# The mark of an AI-drafted scenario in the storyboard notes (owner decision 2026-10-22: addressed to the author).
AI_DRAFTED_MARKER_VI: Final = "[AI soạn — tác giả kiểm tra lại]"
AI_DRAFTED_MARKER_EN: Final = "[AI-drafted — author to check]"
# No Hold (owner decision 2026-10-22, QC feedback A5, contract K2): the AI never holds a block or a practice for an SME.
# A practice whose source states no grading criteria gets criteria drawn from what the lesson teaches, labelled so the
# author reads them as suggested: the module stage starts the practice's feedback_focus.criterion with the label
# (``text.is_suggested_criteria``) and the unit note says so; since run 5496155c the learner reads "Tự kiểm tra" only.
SUGGESTED_CRITERIA_LABEL_VI: Final = "Tiêu chí gợi ý"
SUGGESTED_CRITERIA_LABEL_EN: Final = "Suggested criteria"

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
# A Must Do of kind "decide" whose leading verb produces a work product is practised like a "do" Must Do (QC run
# 8de1c76b, Q5: W1 marked "Ký cam kết bản Action Commitment …" as a decision, so it got no worksheet). Vietnamese
# verbs keep their diacritics ("lập" to draw up, not "lặp" to repeat); English ones are folded.
OUTPUT_VERBS_VI: Final = ("ký", "lập", "điền", "soạn", "viết", "xây dựng", "thiết kế", "tái thiết kế", "vẽ")
OUTPUT_VERBS_EN: Final = ("sign", "draft", "write", "fill in", "fill out", "build", "design", "redesign", "map",
                          "draw up", "prepare")
MAX_COMPONENTS_PER_UNIT: Final = 4
# Bound of IdmComponentDesignV1.support_items (Node reads the same bound).
MAX_SUPPORT_ITEMS: Final = 6
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
# QLT-5 C: an explained unit (content mode "explain") adds, per main idea, why it matters and a short example; the
# measured output was ~1.3x the source words with no room for either. Its word budget (after the clamp above) is
# raised by this percentage, in integer arithmetic so Node (``IDM_EXPLAIN_BUDGET_PERCENT``,
# ``lesson-author-idm-unit.logic.ts``) computes exactly the same number. Job-aid units (lookup content) keep theirs.
EXPLAIN_BUDGET_PERCENT: Final = 135

# --- Content mode checks of teaching html slots (QLT-5 C, ``app.idm.fidelity_checks``) --------------------------
# Measured on 6 runs (plan 2026-10-13 §2.2): 14-28% of html sentences were near-verbatim copies of the source, with
# no explanation device in 83 of 89 units; the slot-level 8-gram check (>= 300 characters, share > 0.7, keep rows
# exempt) never fired. A sentence is judged when it has at least this many words; shorter ones are labels.
IDM_SENTENCE_MIN_WORDS: Final = 6
# Near-verbatim: at least this share of the sentence's word 5-grams (accent-folded) appear in the slot's facts, or
# one run of the facts covers at least this share of the sentence's words.
IDM_SENTENCE_COPY_NGRAM: Final = 5
IDM_SENTENCE_COPY_NGRAM_SHARE: Final = 0.6
IDM_SENTENCE_COPY_RUN_SHARE: Final = 0.7
# A slot with fewer judged sentences (definitions, rules and numbers excluded) is too short to tell.
IDM_SENTENCE_COPY_MIN_SENTENCES: Final = 3
# Highest share of near-verbatim judged sentences per teaching slot. QC run fae66524 (QLT-7) still had 14% of html
# sentences near-verbatim at 0.35 (the QC target is <= 10% on average): an explained slot now quotes at most one in
# five (4 of the run's 9 teaching slots were over it: 0.27-0.33); an organised slot stays close to today's tolerance.
# The keep mode has no copy check.
IDM_SENTENCE_COPY_MAX_SHARE_EXPLAIN: Final = 0.2
IDM_SENTENCE_COPY_MAX_SHARE_STRUCTURED: Final = 0.6
# Sentence length of an explained teaching slot (QLT-7): the run averaged 24 words per html sentence (the source 13,
# the QC target <= 20) and 30 per prose sentence, with sentences of up to 57 words. Prose sentences (paragraph and
# task text; not table rows, list items, callouts or sentences that state a rule) average at most this many words
# when there are at least ``IDM_SENTENCE_LENGTH_MIN_SENTENCES`` of them, and none is longer than the maximum.
IDM_SENTENCE_LENGTH_MAX_AVERAGE_WORDS: Final = 22
IDM_SENTENCE_LENGTH_MAX_WORDS: Final = 40
IDM_SENTENCE_LENGTH_MIN_SENTENCES: Final = 2
# Explanation devices (NFC, lower case, with diacritics: "vì" must not match "ví"). A teaching slot of an explained
# unit uses at least one of them in a sentence it did not copy from the facts. QLT-7: generic words alone ("để",
# "giúp", "vì", "nên", "so với") made the check pass on every slot; only a labelled example, a clarifying phrase or a
# why sentence (``fidelity_checks``) counts now.
EXPLANATION_DEVICES_VI: Final = (
    "ví dụ minh hoạ", "ví dụ minh họa", "chẳng hạn", "nghĩa là", "tức là", "nói cách khác", "điều này có nghĩa",
    "có nghĩa là", "hiểu đơn giản", "vì vậy", "do đó", "nhờ đó", "nhờ vậy", "bởi vì", "lý do", "dễ nhầm",
    "hãy hình dung",
)
EXPLANATION_DEVICES_EN: Final = (
    "illustrative example", "for example", "for instance", "this means", "in other words", "that is why",
    "as a result", "because of this", "the reason", "a common mistake", "a common confusion", "imagine",
)
# The label that marks an example the AI drafted (owner decision 2026-10-14); the author note lists every one.
EXAMPLE_LABEL_VI: Final = "Ví dụ minh hoạ:"
EXAMPLE_LABEL_EN: Final = "Illustrative example:"

# --- Keep the document as it is (QLT-6 preserve_source, ``app.idm.preserve_*``) --------------------------------------
# One provider call segments the document: it returns line numbers only (no text), so a small answer at low thinking
# is enough (~120 lessons x ~80 characters of JSON); one repair, then the deterministic segmentation.
IDM_PRESERVE_SEGMENT_MAX_OUTPUT_TOKENS: Final = 16_000
THINKING_PRESERVE_SEGMENT: Final = "low"
# Lines the model sees: heading candidates in full, body lines shortened (it never copies text, it only numbers it).
IDM_PRESERVE_PROMPT_BODY_CHARS: Final = 140
# A heading line the course can carry verbatim as a chapter, lesson or unit title (IdmLessonPlanV1.title bounds).
IDM_PRESERVE_TITLE_MIN_CHARS: Final = 3
IDM_PRESERVE_TITLE_MAX_CHARS: Final = 180
IDM_PRESERVE_TITLE_MAX_WORDS: Final = 30
# Page furniture removed before segmentation: page numbers, and a short line repeated identically on at least this
# many different pages (a running header or footer; its first occurrence stays, it may be the real heading).
IDM_PRESERVE_FURNITURE_MIN_PAGES: Final = 3
IDM_PRESERVE_FURNITURE_MAX_CHARS: Final = 100
# Unit size (reading units of a kept document): the job-aid reading budget per unit, raised up to the instructional
# density hard limits (``app.instructional_density``: 48 facts, 12,000 source characters, 1,600 words) when a lesson
# would otherwise need more than 12 units. Units split at line boundaries only; text is never cut.
IDM_PRESERVE_UNIT_TARGET_WORDS: Final = SEGMENT_WORD_BUDGET["job_aid"]
IDM_PRESERVE_UNIT_MAX_WORDS: Final = MAX_GENERATED_WORDS
IDM_PRESERVE_UNIT_MAX_FACTS: Final = 48
IDM_PRESERVE_UNIT_MAX_CHARS: Final = 12_000
# Rendered html stays under Node's 20,000-character unit html limit (``MAX_UNIT_HTML_CHARS``) with headroom.
IDM_PRESERVE_UNIT_MAX_HTML_CHARS: Final = 18_000
IDM_PRESERVE_MAX_UNITS_PER_LESSON: Final = 12
# Course bounds of the design contract (modules, lessons per module, lessons, blocks).
IDM_PRESERVE_MAX_CHAPTERS: Final = IDM_MAX_MODULES
IDM_PRESERVE_MAX_LESSONS_PER_CHAPTER: Final = IDM_MAX_LESSONS_PER_MODULE
IDM_PRESERVE_MAX_LESSONS: Final = IDM_MAX_LESSONS
IDM_PRESERVE_MAX_UNITS: Final = IDM_MAX_BLOCKS
# Size-based segmentation (no heading found): a lesson closes at the first page change after this many words.
IDM_PRESERVE_FALLBACK_LESSON_WORDS: Final = 1_500
# Reading speed for the author-facing lesson minutes estimate.
IDM_PRESERVE_WORDS_PER_MINUTE: Final = 200
# Table limits of the course html sanitizer (``sanitizeCourseHtmlData``): a longer run starts a new table, a wider
# row is shown as a line of text (its cells in order).
IDM_PRESERVE_TABLE_MAX_ROWS: Final = 100
IDM_PRESERVE_TABLE_MAX_COLUMNS: Final = 50

# --- Organization profile distillation (ORG_CONTEXT spec §4, ``app.idm.org_distill``) --------------------------------
# One map call reads one batch of the context documents' learned chunks; with more than one batch, reduce calls merge
# the partial profiles, at most this many per call (a partial profile is at most 6,000 characters of text, so a reduce
# prompt stays near 50,000 characters). A batch keeps the map prompt well inside the model's attention window while a
# 1.5 million character source (Node's default bound) needs about ten map calls.
ORG_DISTILL_BATCH_MAX_CHARS: Final = 150_000
ORG_DISTILL_REDUCE_FANIN: Final = 8
# A profile answer is at most ~6,000 characters of JSON text (~3,000 tokens of Vietnamese); the cap leaves room for
# low thinking. One repair after an invalid answer, as in the W1 stages.
ORG_DISTILL_MAX_OUTPUT_TOKENS: Final = 16_000
ORG_DISTILL_ATTEMPTS: Final = 2
THINKING_ORG_DISTILL: Final = "low"
