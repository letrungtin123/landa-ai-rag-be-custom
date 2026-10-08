"""IDM text helpers, deterministic fact signals and W1 sectioning (spec §7.1, §15.1)."""

from __future__ import annotations

import unicodedata
import unittest
from typing import Any

from app.idm.signals import (
    IdmCapacityError,
    compute_fact_signals,
    idm_has_ordered_steps,
    idm_relationship_pairs,
    idm_term_definitions,
    plan_idm_sections,
)
from app.idm.text import (
    char_ngrams,
    claim_support,
    evidence_index,
    fallback_must_do_statement,
    fallback_objective_statement,
    first_main_verb,
    grounding_verdict,
    idm_fold,
    is_generic_title,
    is_unmeasurable_objective,
    jaccard,
    must_do_title,
    ngram_overlap,
    objective_title,
    produces_output,
    sanitize_author_text,
    single_line,
    trim_at_boundary,
)
from app.lesson_author_orchestration_v2_provider import SourceSnapshotFactV2
from tests.idm_golden import DOCUMENT_ID, key, source_facts

MAX_CHARS = 24_000
MAX_FACTS = 320


def fact(fact_key: str, text: str, *, document: str = "doc-a", source_ref: str | None = None,
         **locator: Any) -> SourceSnapshotFactV2:
    return SourceSnapshotFactV2(document_id=document, fact_key=fact_key, scope_key="scope3_00", fact_text=text,
                                source_ref=source_ref, locator=locator)


def flags_of(*texts: str) -> list[list[str]]:
    facts = [fact(f"f{position}", text) for position, text in enumerate(texts)]
    signals = compute_fact_signals(facts)
    return [signals[item.fact_key] for item in facts]


def body(prefix: str, count: int, size: int) -> list[SourceSnapshotFactV2]:
    """``count`` prose facts of exactly ``size`` characters (no heading, table or step shape)."""

    return [fact(f"{prefix}{position}", ("nội dung " * size)[: size - 1] + ".") for position in range(count)]


def sizes(plan: Any) -> list[int]:
    return [len(section.fact_keys) for section in plan.sections]


class TextHelperTests(unittest.TestCase):
    def test_idm_fold_maps_d_strips_diacritics_and_normalises(self) -> None:
        self.assertEqual(idm_fold("ĐƯỜNG đi"), "duong di")
        self.assertEqual(idm_fold("Đ"), "d")
        self.assertEqual(idm_fold("  Quy   TRÌNH\txử lý \n"), "quy trinh xu ly")
        decomposed = unicodedata.normalize("NFD", "Khiếu nại đúng hạn")
        self.assertNotEqual(decomposed, "Khiếu nại đúng hạn")
        self.assertEqual(idm_fold(decomposed), idm_fold("Khiếu nại đúng hạn"))
        self.assertEqual(idm_fold(decomposed), "khieu nai dung han")

    def test_sanitize_author_text_replaces_angle_brackets_and_keeps_indentation(self) -> None:
        text = sanitize_author_text("Mức <b>cao</b>:\n  - mục   một\n          - mục hai\n\nkết", 200)
        self.assertNotIn("<", text)
        self.assertNotIn(">", text)
        self.assertEqual(text, "Mức ‹b›cao‹/b›:\n  - mục một\n    - mục hai\n\nkết")  # noqa: RUF001

    def test_sanitize_author_text_cuts_with_an_ellipsis(self) -> None:
        self.assertEqual(sanitize_author_text("abcdefghij", 5), "abcd…")
        self.assertEqual(sanitize_author_text("abcde", 5), "abcde")
        self.assertEqual(len(sanitize_author_text("x " * 500, 120)), 120)

    def test_single_line_collapses_every_break(self) -> None:
        self.assertEqual(single_line("dòng một\n  dòng   hai\n", 100), "dòng một dòng hai")
        self.assertEqual(single_line("<x>", 10), "‹x›")  # noqa: RUF001
        self.assertEqual(single_line("abcdefghij", 5), "abcd…")
        self.assertEqual(single_line("abcde", 5), "abcde")

    def test_trim_at_boundary_prefers_a_sentence_then_a_word(self) -> None:
        # QC course 364564 (N1): over-long provider text is shortened at a boundary, never mid-word.
        self.assertEqual(trim_at_boundary("Một câu.\n Hai   câu.", 40), "Một câu. Hai câu.")
        self.assertEqual(trim_at_boundary("Bước một xong. Bước hai đang làm dở dang", 26), "Bước một xong.")
        self.assertEqual(trim_at_boundary("alpha beta gamma delta", 15), "alpha beta…")
        self.assertEqual(trim_at_boundary("alpha, beta; gamma delta", 13), "alpha, beta…")
        # A sentence end too early in the bound, or no space at all: the word, then a hard cut.
        self.assertEqual(trim_at_boundary("A. bbbbbbbb cccccccc dddd", 22), "A. bbbbbbbb cccccccc…")
        self.assertEqual(trim_at_boundary("x" * 30, 10), "x" * 9 + "…")
        self.assertEqual(trim_at_boundary("<b> " * 10, 12), "‹b› ‹b› ‹b›…")  # noqa: RUF001 - editor-safe glyphs
        for limit in (5, 50, 200):
            self.assertLessEqual(len(trim_at_boundary("Người học ghi rõ mục tiêu. " * 40, limit)), limit)

    def test_is_generic_title(self) -> None:
        for title in ("Giới thiệu", "TỔNG QUAN", "Thông tin chung:", "Nội dung", "Phần 2", "Part 3", "mục 1",
                      "Introduction", "Other", "Section 12"):
            with self.subTest(title=title):
                self.assertTrue(is_generic_title(title))
        for title in ("Khi nào phải escalate khiếu nại?", "Phần mềm CRM", "Giới thiệu sản phẩm mới"):
            with self.subTest(title=title):
                self.assertFalse(is_generic_title(title))

    def test_is_unmeasurable_objective(self) -> None:
        unmeasurable = [
            "Người học có thể hiểu quy trình xử lý khiếu nại",
            "Học viên có thể nắm được các bước tiếp nhận",
            "Biết cách ghi phiếu KN-01",
            "Người học có thể hiểu rõ tiêu chí escalate",
            "Learners will be able to understand the escalation policy",
            "The learner can learn about the CRM",
            "Staff can be aware of the VIP exception",
        ]
        measurable = [
            "Người học có thể hiệu chỉnh thiết bị đo theo hướng dẫn",
            "Người học có thể phân loại khiếu nại theo mức độ nghiêm trọng",
            "Classify complaints by severity using the level table",
            "Learners will be able to decide when to escalate",
        ]
        for statement in unmeasurable:
            with self.subTest(statement=statement):
                self.assertTrue(is_unmeasurable_objective(statement))
        for statement in measurable:
            with self.subTest(statement=statement):
                self.assertFalse(is_unmeasurable_objective(statement))

    def test_first_main_verb_drops_subject_and_lead_in(self) -> None:
        self.assertEqual(first_main_verb("Người học có thể  Phân loại khiếu nại"), "phân loại khiếu nại")
        self.assertEqual(first_main_verb("Learners will be able to Decide"), "decide")

    def test_ngram_overlap(self) -> None:
        sentence = "nhân viên phải phản hồi khách hàng trong vòng hai mươi bốn giờ"
        self.assertEqual(ngram_overlap(sentence, "trước. " + sentence + " sau."), 1.0)
        self.assertEqual(ngram_overlap(sentence, "một câu hoàn toàn khác về chủ đề khác hẳn nhau luôn"), 0.0)
        self.assertEqual(ngram_overlap("quá ngắn", "quá ngắn"), 0.0)
        folded = "Nhân viên PHẢI phản hồi khách hàng trong vòng hai mươi bốn giờ"
        self.assertEqual(ngram_overlap(folded, idm_fold(sentence)), 1.0)
        self.assertAlmostEqual(ngram_overlap("a b c d", "a b x y", size=2), 1 / 3)

    def test_ngram_and_jaccard_edges(self) -> None:
        self.assertEqual(char_ngrams("ab", 5), {"ab"})
        self.assertEqual(char_ngrams("  ", 5), set())
        self.assertEqual(jaccard(set(), {"a"}), 0.0)
        self.assertEqual(jaccard({"a", "b"}, {"b", "c"}), 1 / 3)


class FactSignalTests(unittest.TestCase):
    def test_heading_flag(self) -> None:
        headings = ["1. Tổng quan quy trình", "1.2 Phạm vi áp dụng", "CHÍNH SÁCH ĐỔI TRẢ", "Chương 2 Quy định chung",
                    "Mục iv phạm vi"]
        for text, flags in zip(headings, flags_of(*headings), strict=True):
            with self.subTest(text=text):
                self.assertIn("H", flags)
        not_headings = ["1. Tổng quan quy trình.", "Nhân viên ghi nhận thông tin vào phiếu",
                        "CHÍNH SÁCH " * 12, "Quy định;"]
        for text, flags in zip(not_headings, flags_of(*not_headings), strict=True):
            with self.subTest(text=text):
                self.assertNotIn("H", flags)

    def test_heading_flag_from_locator_titles(self) -> None:
        facts = [fact("a", "Quy định về bảo hành", scope_title="quy dinh ve bao hanh"),
                 fact("b", "Điều kiện đổi trả", heading_path=["Chính sách", "Điều kiện đổi trả"]),
                 fact("c", "Bảng phí dịch vụ", source_ref="Bảng phí dịch vụ"),
                 fact("d", "Bảng phí dịch vụ khác", source_ref="Bảng phí dịch vụ")]
        signals = compute_fact_signals(facts)
        self.assertEqual([("H" in signals[item.fact_key]) for item in facts], [True, True, True, False])

    def test_table_row_flag(self) -> None:
        self.assertEqual(flags_of("Row 2: Cấp 1 | Thấp | Tự xử lý", "Hàng 2: Cấp 1 | Thấp"), [["T"], []])

    def test_step_flag_needs_three_consecutive_lines(self) -> None:
        three = ["Bước 1: Chào khách hàng.", "Bước 2: Lắng nghe khách hàng.", "Bước 3: Ghi nhận phiếu."]
        self.assertEqual(flags_of(*three), [["S"], ["S"], ["S"]])
        self.assertEqual(flags_of("Step 1 greet the caller.", "2) Listen fully.", "3. Record it."), [["S"]] * 3)
        self.assertEqual(flags_of(*three[:2], "Khách hàng đã được chào."), [[], [], []])
        interrupted = flags_of(three[0], "Khách hàng gọi điện.", three[1], three[2])
        self.assertTrue(all("S" not in flags for flags in interrupted))

    def test_step_runs_never_cross_documents(self) -> None:
        facts = [fact("a1", "Bước 1: Chào."), fact("a2", "Bước 2: Hỏi."),
                 fact("b1", "Bước 3: Ghi.", document="doc-b")]
        self.assertEqual(set(map(tuple, compute_fact_signals(facts).values())), {()})

    def test_rule_flag(self) -> None:
        rules = ["Nhân viên không được hứa bồi thường.", "Agents must not promise refunds.",
                 "Luôn luôn xác minh danh tính.", "Chỉ khi khách hàng đồng ý mới đóng hồ sơ.", "Unless approved."]
        self.assertTrue(all("R" in flags for flags in flags_of(*rules)))
        self.assertEqual(flags_of("Khách hàng gọi điện tới tổng đài."), [[]])

    def test_example_flag(self) -> None:
        examples = ["Ví dụ: khách hàng gọi lần thứ hai.", "For example, a VIP customer calls.",
                    "Tình huống: khách hàng đòi hoàn tiền.", "Chẳng hạn như đơn hàng giao trễ."]
        self.assertTrue(all("E" in flags for flags in flags_of(*examples)))

    def test_emphasis_flag(self) -> None:
        emphasis = ["Lưu ý: kiểm tra yếu tố truyền thông.", "Please READ THIS NOW before calling.",
                    "Đây là điểm quan trọng nhất."]
        self.assertTrue(all("M" in flags for flags in flags_of(*emphasis)))
        self.assertEqual(flags_of("Gọi cho KN-01 ngay."), [[]])

    def test_duplicate_flag_marks_later_near_duplicates_only(self) -> None:
        first = "Nhân viên cần phản hồi khách hàng trong vòng 24 giờ kể từ khi tiếp nhận khiếu nại qua tổng đài."
        near = first[:-1] + " nhé."
        exact = first.upper()
        flags = flags_of(first, "Một câu khác hẳn về bảng mã khiếu nại.", near, exact)
        self.assertEqual([("D" in item) for item in flags], [False, False, True, True])

    def test_blank_text_has_no_flags(self) -> None:
        self.assertEqual(flags_of(" ", " "), [[], []])

    def test_visual_flag(self) -> None:
        facts = [fact("a", "Màn hình tạo mới khiếu nại.", visual_regions=[{"page": 1}]),
                 fact("b", "Xem sơ đồ quy trình bên dưới."), fact("c", "Đóng hồ sơ.", visual_regions=[])]
        signals = compute_fact_signals(facts)
        self.assertEqual([("V" in signals[item.fact_key]) for item in facts], [True, True, False])

    def test_golden_signals_are_sorted_and_cover_every_fact(self) -> None:
        facts = source_facts()
        signals = compute_fact_signals(facts)
        self.assertEqual(list(signals), [item.fact_key for item in facts])
        self.assertTrue(all(flags == sorted(set(flags)) for flags in signals.values()))
        self.assertEqual(signals[key(0, 1)], ["H", "M"])
        self.assertEqual(signals[key(4, 2)], ["T"])
        self.assertIn("S", signals[key(5, 2)])
        self.assertIn("R", signals[key(6, 6)])
        self.assertIn("E", signals[key(8, 2)])
        self.assertIn("M", signals[key(8, 3)])


class SectionPlanTests(unittest.TestCase):
    def assert_within_limits(self, plan: Any, facts: list[SourceSnapshotFactV2]) -> None:
        texts = {item.fact_key: item.fact_text for item in facts}
        flat = [fact_key for section in plan.sections for fact_key in section.fact_keys]
        self.assertEqual(flat, [item.fact_key for item in facts])
        for position, section in enumerate(plan.sections, start=1):
            self.assertEqual(section.section_id, f"sec_{position:03d}")
            self.assertLessEqual(section.content_chars, MAX_CHARS)
            self.assertLessEqual(len(section.fact_keys), MAX_FACTS)
            self.assertEqual(section.content_chars, sum(len(texts[item]) for item in section.fact_keys))

    def test_golden_source_is_one_section(self) -> None:
        facts = source_facts()
        plan = plan_idm_sections(facts, compute_fact_signals(facts), {DOCUMENT_ID: "quy-trinh-khieu-nai.pdf"})
        self.assertEqual(len(plan.sections), 1)
        self.assertEqual(plan.sections[0].document_id, DOCUMENT_ID)
        self.assertEqual(plan.warnings, [])
        self.assert_within_limits(plan, facts)

    def test_empty_source_has_no_sections(self) -> None:
        plan = plan_idm_sections([], {}, {})
        self.assertEqual((plan.sections, plan.warnings), ([], []))

    def test_never_mixes_documents(self) -> None:
        facts = [fact("a0", "1. Tiếp nhận"), fact("a1", "Nội dung a."),
                 fact("b0", "Nội dung b.", document="doc-b"), fact("c0", "Nội dung c.", document="doc-c")]
        plan = plan_idm_sections(facts, compute_fact_signals(facts), {"doc-b": "Phụ lục B"})
        self.assertEqual([section.document_id for section in plan.sections], ["doc-a", "doc-b", "doc-c"])
        self.assertEqual([section.title_path for section in plan.sections], ["1. Tiếp nhận", "Phụ lục B", "doc-c"])
        self.assert_within_limits(plan, facts)

    def test_heading_cuts_only_after_four_thousand_chars(self) -> None:
        # "PHẦN MỘT" (8 chars) + body: 3,908 chars stays below the threshold, 4,008 reaches it.
        for body_count, expected in ((39, [41]), (40, [41, 1])):
            facts = [fact("h1", "PHẦN MỘT"), *body("p", body_count, 100), fact("h2", "PHẦN HAI")]
            plan = plan_idm_sections(facts, {"h1": ["H"], "h2": ["H"]}, {})
            with self.subTest(body_chars=body_count * 100):
                self.assertEqual(sizes(plan), expected)
                self.assert_within_limits(plan, facts)
                if len(expected) == 2:
                    self.assertEqual(plan.sections[1].fact_keys, ("h2",))
                    self.assertEqual(plan.sections[1].title_path, "PHẦN HAI")

    def test_cut_when_the_target_size_would_be_exceeded(self) -> None:
        facts = body("p", 19, 1_000)
        plan = plan_idm_sections(facts, {}, {})
        self.assertEqual(sizes(plan), [18, 1])
        self.assert_within_limits(plan, facts)

    def test_never_exceeds_the_fact_limit(self) -> None:
        facts = body("p", 330, 10)
        plan = plan_idm_sections(facts, {}, {})
        self.assertEqual(sizes(plan), [320, 10])
        self.assert_within_limits(plan, facts)

    def test_table_and_step_runs_are_not_cut_at_the_target(self) -> None:
        for flag in ("T", "S"):
            facts = [*body("p", 17, 1_000), *body("r", 3, 1_000), *body("q", 1, 1_000)]
            signals = {f"r{position}": [flag] for position in range(3)}
            plan = plan_idm_sections(facts, signals, {})
            with self.subTest(flag=flag):
                self.assertEqual(sizes(plan), [20, 1])
                self.assertEqual(plan.warnings, [])
                self.assert_within_limits(plan, facts)

    def test_run_longer_than_the_maximum_is_cut_with_a_warning(self) -> None:
        facts = body("r", 30, 1_000)
        plan = plan_idm_sections(facts, {item.fact_key: ["T"] for item in facts}, {})
        self.assertEqual(sizes(plan), [24, 6])
        self.assertEqual(plan.warnings, ["IDM_W1_SECTION_SPLIT_INSIDE_RUN"])
        self.assert_within_limits(plan, facts)

    def test_too_many_sections_is_a_capacity_error(self) -> None:
        facts = [fact(f"x{position}", "Nội dung.", document=f"doc-{position}") for position in range(17)]
        with self.assertRaises(IdmCapacityError) as caught:
            plan_idm_sections(facts, {}, {})
        self.assertEqual(caught.exception.code, "IDM_SOURCE_EXCEEDS_SINGLE_TASK_CAPACITY")
        self.assertEqual(len(plan_idm_sections(facts[:3], {}, {}, max_sections=3).sections), 3)
        with self.assertRaises(IdmCapacityError):
            plan_idm_sections(facts[:3], {}, {}, max_sections=2)

    # Spec §7.1: never cut inside a T/S run unless the run itself exceeds the maximum.
    def test_heading_flagged_step_does_not_cut_inside_a_step_run(self) -> None:
        facts = [*body("p", 41, 100), fact("s1", "Bước 1: Mở phiếu khiếu nại trên hệ thống."),
                 fact("s2", "Bước 2: KIỂM TRA ĐƠN HÀNG"), fact("s3", "Bước 3: Ghi nhận kết quả vào phiếu.")]
        signals = compute_fact_signals(facts)
        # The upper-case step is structure, not a heading.
        self.assertEqual(signals["s2"], ["M", "S"])
        plan = plan_idm_sections(facts, signals, {})
        owners = {fact_key: section.section_id for section in plan.sections for fact_key in section.fact_keys}
        self.assertEqual(len({owners["s1"], owners["s2"], owners["s3"]}), 1)

    # Spec §7.1: a section is titled by the heading it opens with (never a later footer).
    def test_section_title_is_the_heading_that_opens_it(self) -> None:
        facts = [fact("h1", "1. Tiếp nhận"), fact("a", "Nội dung a."), fact("h2", "2. Phân loại"),
                 fact("b", "Nội dung b.")]
        plan = plan_idm_sections(facts, compute_fact_signals(facts), {})
        self.assertEqual(plan.sections[0].title_path, "1. Tiếp nhận")
        golden = source_facts()
        golden_plan = plan_idm_sections(golden, compute_fact_signals(golden), {})
        self.assertNotIn("Trang", golden_plan.sections[0].title_path)


class TermAndStepHelperTests(unittest.TestCase):
    def test_term_definitions_accept_the_explicit_forms(self) -> None:
        pairs = dict(idm_term_definitions([
            "SLA: thời hạn cam kết phản hồi cho khách hàng theo hợp đồng",
            "Khiếu nại là phản ánh của khách hàng về sản phẩm không đạt cam kết",
            "Escalation means passing the complaint to a manager for decision",
            "CRM (viết tắt của Customer Relationship Management) là hệ thống quản lý khách hàng",
        ]))
        self.assertEqual(set(pairs), {"SLA", "KHIEUNAI", "ESCALATION", "CRM"})
        self.assertEqual(pairs["KHIEUNAI"], "phản ánh của khách hàng về sản phẩm không đạt cam kết")
        self.assertIn("Customer Relationship Management", pairs["CRM"])

    def test_term_definitions_reject_rows_short_duplicate_and_long_terms(self) -> None:
        pairs = idm_term_definitions([
            "Row 1: Mã là bảng mã khiếu nại theo nhóm sản phẩm", "Bước 1: Mở là thao tác đầu tiên của quy trình",
            "CRM là phần mềm.", "SLA: thời hạn cam kết phản hồi cho khách hàng theo hợp đồng",
            "SLA là thời hạn cam kết phản hồi cho khách hàng theo hợp đồng",
            "Một hai ba bốn năm sáu bảy tám chín là định nghĩa rất dài của một thuật ngữ", "", "   ",
        ])
        self.assertEqual([answer for answer, _definition in pairs], ["SLA"])

    def test_term_definitions_are_capped_at_ten(self) -> None:
        texts = [f"Mục {chr(65 + index)}X là định nghĩa đầy đủ số {index} của thuật ngữ này" for index in range(12)]
        self.assertEqual(len(idm_term_definitions(texts)), 10)

    def test_ordered_steps_need_three_steps(self) -> None:
        steps = ["Bước 1: Chào khách hàng thật lịch sự", "Bước 2: Lắng nghe toàn bộ nội dung",
                 "Bước 3: Ghi nhận thông tin vào phiếu"]
        self.assertTrue(idm_has_ordered_steps(steps, locale="vi"))
        self.assertFalse(idm_has_ordered_steps(steps[:2], locale="vi"))
        self.assertTrue(idm_has_ordered_steps(["Step 1: Greet the caller politely", "Step 2: Listen to the full "
                                               "complaint", "Step 3: Record it in the form"], locale="en"))

    def test_relationship_pairs_need_explicit_relations(self) -> None:
        self.assertEqual(idm_relationship_pairs(["Tiếp nhận → Phân loại → Xử lý"]),
                         [("Tiếp nhận", "Phân loại", "→"), ("Phân loại", "Xử lý", "→")])
        self.assertEqual(idm_relationship_pairs(["Nhân viên tiếp nhận khiếu nại."]), [])


class GroundingHelperTests(unittest.TestCase):
    """Deterministic evidence support behind the FAQ guard and the worksheet self-check (QC 234653, R5/R6)."""

    def test_function_words_and_numbers(self) -> None:
        index = evidence_index(["Bắt buộc escalate khi giá trị thiệt hại từ 50 triệu đồng trở lên."],
                               number_texts=["Chọn 01 quy trình, tối thiểu 30%"])
        self.assertEqual(index.numbers, frozenset({"50", "1", "30"}))
        self.assertIn(("bat", "buoc"), index.pairs)
        words, pairs, count = claim_support("Vì vậy, bắt buộc escalate khi thiệt hại từ 50 triệu đồng.", index)
        self.assertEqual((words, count), (1.0, 7))
        self.assertGreater(pairs, 0.7)
        self.assertEqual(claim_support("và là của", index), (1.0, 1.0, 0))
        self.assertEqual(grounding_verdict("Bắt buộc escalate khi thiệt hại từ 50 triệu đồng.", index), "grounded")
        self.assertEqual(grounding_verdict("Bắt buộc escalate khi thiệt hại từ 60 triệu đồng.", index), "numbers")
        self.assertEqual(grounding_verdict("Chọn 1 quy trình và tăng 30,0% năng suất.", index), "numbers")
        self.assertEqual(grounding_verdict("Tăng tối thiểu 30% khi chọn 001 quy trình.", index), "support")

    def test_produces_output(self) -> None:
        for statement in ("Điền hoàn chỉnh 9 ô nội dung trên bảng tự đánh giá CEO Change Mindset Canvas",
                          "Lập bản đồ 3 nguồn lực bên ngoài cần liên minh",
                          "Người học có thể soạn bản cam kết hành động", "Draft a 90-day plan"):
            self.assertTrue(produces_output(statement, "do"), statement)
        for statement, kind in (("Phân loại khiếu nại theo nhóm", "do"), ("Xác định vị trí hiện tại", "do"),
                                ("Choose the next step", "do"), ("Điền hoàn chỉnh 9 ô", "decide")):
            self.assertFalse(produces_output(statement, kind), statement)


class TitleHelperTests(unittest.TestCase):
    def test_objective_title_drops_learner_lead_in_and_fallback_template(self) -> None:
        cases = {
            "Người học có thể phân tích bốn lực đẩy của bối cảnh mới": "Phân tích bốn lực đẩy của bối cảnh mới",
            "Người học có thể áp dụng Thế Giới Mới: Biến Động trong công việc": "Thế Giới Mới: Biến Động",
            "The learner can apply Complaint triage at work": "Complaint triage",
            "Learners will be able to classify complaints.": "Classify complaints",
            "Phân loại khiếu nại theo nhóm": "Phân loại khiếu nại theo nhóm",
        }
        for statement, expected in cases.items():
            with self.subTest(statement=statement):
                self.assertEqual(objective_title(statement), expected)

    def test_must_do_title_only_rewrites_the_fallback_template(self) -> None:
        self.assertEqual(must_do_title("Thực hiện đúng Chu Trình PDCA"), "Chu Trình PDCA")
        self.assertEqual(must_do_title("Correctly carry out the intake steps"), "The intake steps")
        self.assertEqual(must_do_title("Quyết định tự xử lý hay escalate"), "Quyết định tự xử lý hay escalate")
        # A client objective copied as the fallback Must Do loses its learner lead-in in the lesson title.
        self.assertEqual(must_do_title("Người học có thể phân loại khiếu nại theo mức độ"),
                         "Phân loại khiếu nại theo mức độ")

    def test_fallback_statements_are_measurable_and_round_trip_to_their_topic(self) -> None:
        for locale in ("vi", "en"):
            with self.subTest(locale=locale):
                objective = fallback_objective_statement("Chu trình PDCA", locale)
                must_do = fallback_must_do_statement("Chu trình PDCA", locale)
                self.assertFalse(is_unmeasurable_objective(objective))
                self.assertNotIn("trong công việc", objective)
                self.assertEqual((objective_title(objective), must_do_title(must_do)),
                                 ("Chu trình PDCA", "Chu trình PDCA"))
        self.assertEqual(fallback_objective_statement("X", "vi"),
                         "Người học có thể áp dụng các điểm chính của X vào một tình huống công việc cụ thể")


if __name__ == "__main__":
    unittest.main()
