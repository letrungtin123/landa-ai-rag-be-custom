"""IDM html slot rules: validator parity, the deterministic normaliser and the W5 repair loop.

Regression for run e869f43a (2026-10-08): every W5 html repair was rejected as
``IDM_W5_REPAIR_INVALID`` with ``"errors": []``. The repair answer's ``semantic_content`` had no
server-owned ``version`` stamp, so the shared reader rejected even a valid slot; the repair prompt
named only ``HTML_SEMANTIC_INVALID`` / ``HTML_INSTRUCTIONAL_DENSITY_EXCEEDED`` and a slot path; and
the writer prompt never stated the ordered-content rules (the provider wire schema drops every
bound). All fixtures below are synthetic; no provider is called.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import re
import unittest
from collections import Counter
from typing import Any

from app import main
from app.idm.html_rules import (
    DEFAULT_MIN_VISIBLE_CHARS,
    GROUP_LIMITS,
    MAX_HEADING_CHARS,
    MAX_ROW_LABEL_CHARS,
    MAX_ROW_VALUE_CHARS,
    MAX_ROWS,
    html_rule_violations,
    minimum_visible_chars,
    normalize_html_semantic,
    structural_violations,
    visible_text,
)
from app.idm.prompts import unit_writer_prompt
from app.idm.storyboard import IdmUnitWriter, parse_brief, run_idm_unit, safe_error_details
from app.instructional_density import INSTRUCTIONAL_DENSITY_POLICY_VERSION
from app.schemas.orchestration_v2 import RagLessonAuthorUnitV2Request
from app.services.lesson_author import proposal_validation
from app.services.lesson_author.errors import LessonAuthorProposalValidationError
from app.services.lesson_author.proposal_validation import (
    MIN_STAGED_LESSON_AUTHOR_HTML_TEXT_CHARS,
    SEMANTIC_LEARNING_HTML_LIMITS,
)
from app.services.lesson_author.staged import provider_schemas as staged_schemas
from app.services.lesson_author.staged import validation as staged_validation
from app.workflows.contracts import WorkflowFailure
from tests.idm_test_support import make_runtime
from tests.test_idm_storyboard import (
    REPAIR,
    WRITER,
    FakeGenerate,
    escalate_body,
    severity_body,
    severity_writer,
    slot_repair,
)

ROWS = [("Cấp 1", "Ảnh hưởng thấp, nhân viên tự xử lý."), ("Cấp 2", "Phàn nàn lần thứ hai, báo trưởng nhóm."),
        ("Cấp 3", "Liên quan an toàn hoặc pháp lý, escalate ngay.")]


def para(text: str, kind: str = "paragraph") -> dict[str, Any]:
    return {"kind": kind, "text": text, "items": [], "rows": []}


def listing(*items: Any, kind: str = "bullets", text: str | None = None) -> dict[str, Any]:
    return {"kind": kind, "text": text, "items": list(items), "rows": []}


def table(*rows: Any, text: str | None = None) -> dict[str, Any]:
    return {"kind": "table", "text": text, "items": [], "rows": list(rows)}


def row(label: str, value: str) -> dict[str, str]:
    return {"label": label, "value": value}


def section(heading: Any, *blocks: Any) -> dict[str, Any]:
    return {"heading": heading, "learning_block_ids": [], "blocks": list(blocks)}


def semantic(*sections: Any, **extra: Any) -> dict[str, Any]:
    return {"version": 2, "sections": list(sections), **extra}


def shared_reason(value: Any) -> str | None:
    return proposal_validation.semantic_learning_visible_text(copy.deepcopy(value))[1]


def payload_code(value: Any) -> str | None:
    return staged_schemas.staged_component_payload_code({"type": "html", "semantic_content": copy.deepcopy(value)})


def codes(value: Any, **kwargs: Any) -> list[str]:
    return [item.code for item in html_rule_violations(value, **kwargs)]


def words(value: Any) -> Counter[str]:
    """Learner words of a semantic payload (markup and markers ignored)."""

    def strings(node: Any) -> list[str]:
        if isinstance(node, str):
            return [node]
        if isinstance(node, list):
            return [text for item in node for text in strings(item)]
        if isinstance(node, dict):
            return [text for key, item in node.items() if key not in {"kind", "version"} for text in strings(item)]
        return []

    joined = re.sub(r"<[^>]*>", " ", " ".join(strings(value)))
    return Counter(re.findall(r"\w+", joined.casefold()))


VALID = semantic(section("Khiếu nại thuộc cấp nào?", para("Mỗi khiếu nại thuộc một trong ba cấp độ."),
                         table(*(row(label, value) for label, value in ROWS))))


def violating_fixtures() -> dict[str, dict[str, Any]]:
    """One synthetic payload per ordered-content rule the shared reader enforces."""

    sections_13 = [section(f"Mục {n}", para(f"Nội dung mục {n}.")) for n in range(13)]
    return {
        "bullets_with_intro_text": semantic(section("H", listing("a1", "b1", text="Gồm các bước sau:"))),
        "paragraph_with_items": semantic(section("H", {"kind": "paragraph", "text": "", "items": ["x1"], "rows": []})),
        "table_with_text": semantic(section("H", table(row("a", "b"), text="Bảng dưới đây:"))),
        "legacy_with_sections": semantic(section("H", para("Một câu.")), paragraphs=["Một câu khác."]),
        "empty_unknown_legacy_key": semantic(section("H", para("Một câu.")), bullets=[]),
        "unknown_top_key": semantic(section("H", para("Một câu.")), extra="x"),
        "no_sections": semantic(),
        "thirteen_sections": semantic(*sections_13),
        "section_unknown_key": semantic({"heading": "H", "title": "T", "blocks": [para("Một câu.")]}),
        "empty_heading": semantic(section("  ", para("Một câu."))),
        "long_heading": semantic(section("x" * (MAX_HEADING_CHARS + 1), para("Một câu."))),
        "no_blocks": semantic(section("H")),
        "thirteen_blocks": semantic(section("H", *(para(f"Câu {n}.") for n in range(13)))),
        "unknown_kind": semantic(section("H", {"kind": "heading", "text": "Tiêu đề phụ", "items": [], "rows": []})),
        "unknown_block_key": semantic(section("H", {**para("Một câu."), "label": "x"})),
        "empty_text": semantic(section("H", para("   "))),
        "empty_items": semantic(section("H", listing())),
        "blank_item": semantic(section("H", listing("a1", " "))),
        "non_string_item": semantic(section("H", listing("a1", 7))),
        "thirteen_paragraphs": semantic(*(section(f"M{n}", para(f"Câu {n}."), listing(f"ý {n}"), para(f"Kết {n}."))
                                          for n in range(7))),
        "twenty_one_bullets": semantic(section("H", listing(*(f"ý {n}" for n in range(21))))),
        "nine_warnings": semantic(section("H", *(para(f"Cảnh báo {n}.", "warning") for n in range(9)))),
        "thirty_one_rows": semantic(section("H", table(*(row(f"r{n}", "v") for n in range(MAX_ROWS + 1))))),
        "long_paragraph": semantic(section("H", para("a" * (GROUP_LIMITS["paragraphs"][1] + 1)))),
        "long_bullet": semantic(section("H", listing("b" * (GROUP_LIMITS["bullet_points"][1] + 1)))),
        "incomplete_row": semantic(section("H", table(row("Ghi chú", "")))),
        "long_row_label": semantic(section("H", table(row("l" * (MAX_ROW_LABEL_CHARS + 1), "v")))),
        "long_row_value": semantic(section("H", table(row("l", "v" * (MAX_ROW_VALUE_CHARS + 1))))),
        "duplicate_refs": semantic({"heading": "H", "learning_block_ids": ["a", "a"], "blocks": [para("Một câu.")]}),
    }


class ValidatorParityTests(unittest.TestCase):
    def test_limits_mirror_the_shared_validator(self) -> None:
        limits = SEMANTIC_LEARNING_HTML_LIMITS
        self.assertEqual(limits["heading"], MAX_HEADING_CHARS)
        for group, bounds in GROUP_LIMITS.items():
            self.assertEqual(limits[group], bounds)
        self.assertEqual(limits["comparison_rows"], (MAX_ROWS, MAX_ROW_LABEL_CHARS, MAX_ROW_VALUE_CHARS))
        self.assertEqual(DEFAULT_MIN_VISIBLE_CHARS, MIN_STAGED_LESSON_AUTHOR_HTML_TEXT_CHARS)
        self.assertEqual([minimum_visible_chars(value) for value in (None, 0, 30, 500, 10_000)],
                         [320, 320, 20, 250, 320])

    def test_every_structural_rule_matches_the_shared_reader(self) -> None:
        self.assertIsNone(shared_reason(VALID))
        self.assertEqual(structural_violations(VALID), [])
        for name, value in violating_fixtures().items():
            with self.subTest(fixture=name):
                self.assertIsNotNone(shared_reason(value))
                self.assertEqual(payload_code(value), "HTML_SEMANTIC_INVALID")
                found = structural_violations(value)
                self.assertTrue(found)
                for item in found:
                    self.assertRegex(item.location, r"^semantic_content(\.[a-z_]+(\[\d+\])?)*$")

    def test_presentation_markup_matches_the_payload_check(self) -> None:
        for text in ('<div class="x">Một câu.</div>', '<span style="color:red">Một câu.</span>', "<script>x</script>"):
            value = semantic(section("H", para(text)))
            with self.subTest(text=text):
                self.assertEqual(payload_code(value), "HTML_PRESENTATION_FORBIDDEN")
                self.assertIn("HTML_PRESENTATION_MARKUP", codes(value))

    def test_density_and_depth_are_measured_like_the_shared_finding(self) -> None:
        budget = {"policy_version": INSTRUCTIONAL_DENSITY_POLICY_VERSION, "max_visible_chars": 400, "max_words": 30,
                  "source_content_chars": 40}
        for count, expected in ((30, None), (31, "HTML_INSTRUCTIONAL_DENSITY_EXCEEDED")):
            value = semantic(section("Tiêu đề không được tính", para(" ".join(["từ"] * count))))
            component = {"type": "html", "semantic_content": value}
            finding = staged_validation.staged_instructional_finding(component, 0, None, budget, None)
            with self.subTest(words=count):
                self.assertEqual(None if finding is None else finding.code,
                                 None if expected is None else expected)
                found = html_rule_violations(value, budget=budget, min_chars=minimum_visible_chars(40))
                self.assertEqual([item.code for item in found], [] if expected is None else [expected])
        found = html_rule_violations(value, budget=budget)[0]
        self.assertEqual(found.metrics, (("words", 31, 30), ("characters", len(visible_text(value)), 400)))
        self.assertEqual(found.as_log(0), f"HTML_INSTRUCTIONAL_DENSITY_EXCEEDED@components[0].semantic_content;"
                                          f"words=31/30;characters={len(visible_text(value))}/400")
        short = semantic(section("H", para("Ngắn.")))
        self.assertEqual(codes(short, min_chars=20), ["HTML_INSUFFICIENT_DEPTH"])
        finding = staged_validation.staged_instructional_finding({"type": "html", "semantic_content": short}, 0, None,
                                                                 budget, None)
        self.assertEqual(finding.code if finding else None, "HTML_INSUFFICIENT_DEPTH")


class NormalizerTests(unittest.TestCase):
    def assert_fixed(self, value: dict[str, Any], *fix_names: str, same_words: bool = True,
                     invalid_before: bool = True) -> dict[str, Any]:
        if invalid_before:
            self.assertIsNotNone(shared_reason(value) or payload_code(value))
        fixed, fixes = normalize_html_semantic(value)
        self.assertIsNone(shared_reason(fixed))
        self.assertIsNone(payload_code(fixed))
        for name in fix_names:
            self.assertIn(name, fixes)
        if same_words:
            # Nothing invented, nothing lost (list markers such as "1." are rendered by the list itself).
            self.assertEqual(Counter({word: count for word, count in words(fixed).items() if not word.isdigit()}),
                             Counter({word: count for word, count in words(value).items() if not word.isdigit()}))
        return fixed

    def test_valid_content_is_returned_unchanged(self) -> None:
        fixed, fixes = normalize_html_semantic(VALID)
        self.assertIs(fixed, VALID)
        self.assertEqual(fixes, Counter())

    def test_introduction_text_moves_out_of_list_and_table_blocks(self) -> None:
        value = semantic(section("Các bước xử lý", listing("Tiếp nhận yêu cầu", "Phân loại mức độ",
                                                            text="Thực hiện theo thứ tự sau:", kind="steps"),
                                 table(row("Cấp 1", "Tự xử lý"), text="Bảng phân cấp:")))
        fixed = self.assert_fixed(value, "block_fields_split")
        blocks = fixed["sections"][0]["blocks"]
        self.assertEqual([block["kind"] for block in blocks], ["paragraph", "steps", "paragraph", "table"])
        self.assertEqual(blocks[0]["text"], "Thực hiện theo thứ tự sau:")
        self.assertEqual(blocks[1]["items"], ["Tiếp nhận yêu cầu", "Phân loại mức độ"])

    def test_list_items_outside_a_list_become_a_list_block(self) -> None:
        cases = {  # (block, resulting kind, rejected by the shared reader before)
            "markdown": (para("- Kiểm tra hoá đơn\n- Đối chiếu đơn hàng"), "bullets", False),
            "numbered": (para("1. Tiếp nhận\n2. Phân loại\n3. Xử lý"), "steps", False),
            "html_list": (para("<ul><li>Kiểm tra hoá đơn</li><li>Đối chiếu đơn hàng</li></ul>"), "bullets", False),
            "items_on_paragraph": ({"kind": "paragraph", "text": None, "items": ["Một", "Hai"], "rows": []}, "bullets",
                                   True),
            "steps_without_items": (listing(kind="steps", text="1. Gọi lại\n2. Ghi nhận"), "steps", True),
        }
        for name, (block, kind, invalid) in cases.items():
            with self.subTest(case=name):
                fixed = self.assert_fixed(semantic(section("Việc cần làm", block)), invalid_before=invalid)
                self.assertEqual([item["kind"] for item in fixed["sections"][0]["blocks"]], [kind])
                self.assertFalse(any("<" in item or item.startswith(("-", "1.")) for item in
                                     fixed["sections"][0]["blocks"][0]["items"]))

    def test_sub_headings_become_sections_and_heading_markup_is_removed(self) -> None:
        value = semantic(section("<h2>Phân loại khiếu nại</h2>", para("Có ba cấp độ."),
                                 para("<h3>Khi nào escalate?</h3>"), para("Khi có yếu tố an toàn."),
                                 {"kind": "subheading", "text": "Ai xử lý?", "items": [], "rows": []},
                                 para("Quản lý trực tiếp.")))
        self.assertIsNotNone(shared_reason(value))  # the sub-heading block kind is unknown to the reader
        fixed, fixes = normalize_html_semantic(value)
        self.assertIsNone(shared_reason(fixed))
        self.assertEqual([item["heading"] for item in fixed["sections"]],
                         ["Phân loại khiếu nại", "Khi nào escalate?", "Ai xử lý?"])
        self.assertEqual(fixes["subheading_to_section"], 2)
        self.assertEqual(words(fixed), words(value))
        lone = semantic(section("H", para("### Chỉ có tiêu đề")))
        fixed, _ = normalize_html_semantic(lone)
        self.assertEqual(fixed["sections"], [section("H", para("Chỉ có tiêu đề"))])

    def test_stray_wrappers_and_empty_elements_are_removed(self) -> None:
        value = semantic(
            section("<strong>Lưu ý</strong>",
                    para('<div class="note"><p>Không hứa <b>bồi thường</b> trước.</p></div>'),
                    listing("<span style='x'>Xác minh</span>", "", "  "), para(" "),
                    {**para("Một câu."), "label": None}),
            section(" ", listing()),
            section("", para("Phần tiếp theo không có tiêu đề.")),
        )
        fixed = self.assert_fixed(value, "markup_unwrapped", "empty_item_dropped", "empty_block_dropped",
                                  "empty_section_dropped", "untitled_section_merged", "empty_field_dropped")
        self.assertEqual(len(fixed["sections"]), 1)
        blocks = fixed["sections"][0]["blocks"]
        self.assertEqual(blocks[0]["text"], "Không hứa bồi thường trước.")
        self.assertEqual(blocks[1]["items"], ["Xác minh"])
        self.assertEqual(blocks[-1]["text"], "Phần tiếp theo không có tiêu đề.")

    def test_too_many_paragraphs_are_joined_without_changing_text(self) -> None:
        value = semantic(*(section(f"Phần {n}", para(f"Câu thứ nhất của phần {n}."),
                                   para(f"Câu thứ hai của phần {n}.")) for n in range(7)))
        self.assertIn("paragraphs exceeds", shared_reason(value) or "")
        fixed = self.assert_fixed(value, "paragraphs_merged")
        self.assertEqual(visible_text(fixed), visible_text(value))  # same text, same density measure
        warnings = semantic(section("H", *(para(f"Cảnh báo {n}.", "warning") for n in range(9))))
        self.assert_fixed(warnings, "warnings_merged")

    def test_legacy_fields_are_dropped_only_when_duplicated_or_converted(self) -> None:
        duplicated = semantic(section("Phân loại", para("Có ba cấp độ.")), heading="Phân loại",
                              paragraphs=["Có ba cấp độ."])
        self.assert_fixed(duplicated, "legacy_duplicate_dropped", same_words=False)
        extra = semantic(section("Phân loại", para("Có ba cấp độ.")), paragraphs=["Một câu chưa có ở trên."])
        fixed, fixes = normalize_html_semantic(extra)
        self.assertEqual(fixes, Counter())
        self.assertIs(fixed, extra)  # never dropped: it is content the sections do not hold
        legacy = {"version": 2, "heading": "Phân loại", "paragraphs": ["Có ba cấp độ."],
                  "bullet_points": ["Cấp 1", "Cấp 2"], "sections": []}
        fixed = self.assert_fixed(legacy, "legacy_converted")
        self.assertEqual(fixed["sections"][0]["heading"], "Phân loại")

    def test_normaliser_is_total_idempotent_and_never_adds_words(self) -> None:
        malformed: list[Any] = [None, "text", [], {}, {"sections": "x"}, {"version": 2, "sections": [None, 3]},
                                semantic(section("H", None, 5, {"kind": 3}, {"kind": "paragraph", "text": 4})),
                                semantic({"heading": "H", "blocks": "x"}), {"version": 2, "heading": "Chỉ tiêu đề"}]
        for index, value in enumerate([VALID, *violating_fixtures().values(), *malformed]):
            with self.subTest(index=index):
                fixed, _ = normalize_html_semantic(value)
                again, fixes = normalize_html_semantic(fixed)
                self.assertEqual(fixes, Counter())
                self.assertIs(again, fixed)
                if isinstance(value, dict):
                    self.assertFalse(set(words(fixed)) - set(words(value)))

    def test_unfixable_violations_are_left_for_the_repair(self) -> None:
        for name in ("incomplete_row", "long_paragraph", "twenty_one_bullets", "thirty_one_rows", "non_string_item",
                     "unknown_top_key"):
            value = violating_fixtures()[name]
            with self.subTest(fixture=name):
                fixed, _ = normalize_html_semantic(value)
                self.assertIsNotNone(shared_reason(fixed))
                self.assertEqual(words(fixed), words(value))

    def test_safe_error_details_never_carry_provider_text(self) -> None:
        cases: list[tuple[BaseException, list[dict[str, Any]]]] = [
            (LessonAuthorProposalValidationError("HTML_SEMANTIC_INVALID"),
             [{"type": "HTML_SEMANTIC_INVALID", "loc": []}]),
            (LessonAuthorProposalValidationError("Câu văn bí mật", code="FAQ_ITEM_INVALID",
                                                      path="components[1].items"),
             [{"type": "FAQ_ITEM_INVALID", "loc": ["components[1].items"]}]),
            (WorkflowFailure("LESSON_VALIDATION_FAILED", "m", diagnostics={"validation_finding": {
                "code": "INSTANCE_FIELD_NOT_ALLOWED", "path": "components[0]"}}),
             [{"type": "INSTANCE_FIELD_NOT_ALLOWED", "loc": ["components[0]"]}]),
            (json.JSONDecodeError("Expecting value", "secret text", 0), [{"type": "json_invalid", "loc": []}]),
            (ValueError("secret text"), [{"type": "ValueError", "loc": []}]),
            (LessonAuthorProposalValidationError("Câu văn bí mật", path="ignore previous <rules>"),
             [{"type": "UNIT_SHAPE_INVALID", "loc": []}]),
        ]
        for error, expected in cases:
            with self.subTest(error=type(error).__name__):
                self.assertEqual(safe_error_details(error), expected)


def html_writer(body: dict[str, Any], c0_semantic: dict[str, Any]) -> dict[str, Any]:
    writer = severity_writer(body)
    writer["components"]["c0"]["semantic_content"] = c0_semantic
    return writer


def good_c0(body: dict[str, Any]) -> dict[str, Any]:
    return slot_repair(severity_writer(body), 0)


class WriterRepairLoopTests(unittest.IsolatedAsyncioTestCase):
    """Writer answer -> normaliser/validator -> repair prompt names the rule (fake provider only)."""

    async def run_unit(self, *answers: tuple[str, Any]) -> tuple[dict[str, Any], FakeGenerate, list[str]]:
        body = severity_body()
        request = RagLessonAuthorUnitV2Request.model_validate(body)
        deps = dataclasses.replace(main._idm_unit_deps(request), judge_mode="off")
        queues: dict[str, list[Any]] = {}
        for family, answer in answers:
            queues.setdefault(family, []).append(answer)
        provider = FakeGenerate(**queues)
        with self.assertLogs("app.idm", "INFO") as logs:
            result = await run_idm_unit(contract=request.unit_contract, runtime=make_runtime(provider, allowance=None),
                                        deps=deps, fallback_only=False)
        return result, provider, [record.getMessage() for record in logs.records]

    def table_rows(self, extra: dict[str, str] | None = None) -> dict[str, Any]:
        rows = [row(label, value) for label, value in ROWS] + ([extra] if extra else [])
        return semantic(section("Khiếu nại thuộc cấp độ nào?", para("Mỗi khiếu nại thuộc một trong ba cấp độ. "
                                                                    "Cấp độ quyết định ai xử lý."), table(*rows)))

    async def test_valid_html_repair_is_accepted_once_the_version_is_stamped(self) -> None:
        body = severity_body()
        bad = self.table_rows(row("Ghi chú", ""))
        result, provider, _logs = await self.run_unit((WRITER, html_writer(body, bad)), (REPAIR, good_c0(body)))
        self.assertEqual(provider.names, [WRITER, REPAIR])
        self.assertEqual(result["content_origin"], "provider_validated")
        unit = result["unit"]
        self.assertTrue(unit["idm_quality"]["repair_applied"])
        self.assertNotIn("source_locked_fallback", unit["components"][0])
        self.assertEqual(len(unit["components"][0]["semantic_content"]["sections"][0]["blocks"][1]["rows"]), 3)
        prompt = provider.calls[1]["prompt"]
        self.assertIn('{"code":"HTML_SEMANTIC_INVALID","path":"components[0]"}', prompt)
        self.assertIn("FAILING_RULES", prompt)
        self.assertIn("c0 HTML_ROW_INVALID at c0.semantic_content.sections[0].blocks[1].rows[3]: every table row is "
                      '{"label", "value"} with both non-empty', prompt)
        self.assertNotIn("Ghi chú", prompt.split("REPAIR_REQUIREMENTS", 1)[1])  # rules, never the rejected text

    def test_the_shared_merge_rejects_an_unstamped_html_repair(self) -> None:
        # Root cause evidence: the repair wire schema hides ``version`` and the shared reader needs it.
        body = severity_body()
        request = RagLessonAuthorUnitV2Request.model_validate(body)
        deps = main._idm_unit_deps(request)
        writer = IdmUnitWriter(request.unit_contract, parse_brief(request.unit_contract), deps, make_runtime(None))
        unit = writer.bind(json.dumps(severity_writer(body), ensure_ascii=False))
        text = json.dumps(good_c0(body), ensure_ascii=False)
        raw = deps.decode_repair(text, unit, [0], [], {})
        with self.assertRaises(LessonAuthorProposalValidationError) as caught:
            deps.merge_repair(unit, raw, [0], [])
        self.assertEqual(str(caught.exception), "HTML_SEMANTIC_INVALID")
        merged = deps.merge_repair(unit, writer.decode_repair(text, unit, [0], []), [0], [])
        self.assertIsNone(deps.validate_unit(merged, writer.expected))

    async def test_normaliser_fixes_the_writer_answer_without_a_repair_call(self) -> None:
        body = severity_body()
        messy = semantic(section("<h3>Khiếu nại thuộc cấp độ nào?</h3>",
                                 para("<p>Mỗi khiếu nại thuộc một trong ba cấp độ. Cấp độ quyết định ai xử lý.</p>"),
                                 table(*(row(label, value) for label, value in ROWS), text="Bảng phân cấp:")))
        self.assertEqual(payload_code(messy), "HTML_SEMANTIC_INVALID")
        result, provider, logs = await self.run_unit((WRITER, html_writer(body, messy)))
        self.assertEqual(provider.names, [WRITER])
        self.assertEqual(result["content_origin"], "provider_validated")
        sections = result["unit"]["components"][0]["semantic_content"]["sections"]
        self.assertEqual(sections[0]["heading"], "Khiếu nại thuộc cấp độ nào?")
        self.assertEqual([block["kind"] for block in sections[0]["blocks"]], ["paragraph", "paragraph", "table"])
        self.assertFalse(any("idm_unit_validation" in line for line in logs))

    async def test_density_only_failure_asks_to_condense_with_measured_value_and_limit(self) -> None:
        body = severity_body()
        sentences = [f"Ý thứ {n} giải thích tiêu chí phân loại khiếu nại theo mức độ ảnh hưởng tới khách hàng."
                     for n in range(1, 46)]
        long = semantic(section("Khiếu nại thuộc cấp độ nào?", *(para(" ".join(sentences[i:i + 9]))
                                                                for i in range(0, 45, 9)),
                                table(*(row(label, value) for label, value in ROWS))))
        text = visible_text(long)
        measured_words, measured_chars = len(re.findall(r"\w+", text)), len(text)
        self.assertGreater(measured_words, 450)
        result, provider, logs = await self.run_unit((WRITER, html_writer(body, long)), (REPAIR, good_c0(body)))
        self.assertEqual(provider.names, [WRITER, REPAIR])
        self.assertEqual(result["content_origin"], "provider_validated")
        prompt = provider.calls[1]["prompt"]
        self.assertIn(f"c0 HTML_INSTRUCTIONAL_DENSITY_EXCEEDED at c0.semantic_content: measured {measured_words} "
                      f"words, {measured_chars} characters; limit 450 words, 3150 characters. Condense this slot to "
                      "at most 405 words and 2835 characters", prompt)
        self.assertIn("Everything else in this slot is valid: keep its sections and their order, only shorten it.",
                      prompt)
        validation = next(line for line in logs if "idm_unit_validation" in line)
        self.assertIn(f"words={measured_words}/450", validation)
        self.assertNotIn("Ý thứ", "".join(logs))

    async def test_whole_slot_paragraph_limit_is_named_with_its_count(self) -> None:
        body = severity_body()
        crowded = semantic(*(section(f"Phần {n}", para(f"Mở đầu phần {n} về phân loại khiếu nại."),
                                     listing(f"Dấu hiệu {n}"), para(f"Kết luận phần {n} về người xử lý."))
                             for n in range(1, 8)))
        result, provider, _logs = await self.run_unit((WRITER, html_writer(body, crowded)), (REPAIR, good_c0(body)))
        self.assertEqual(result["content_origin"], "provider_validated")
        self.assertIn("c0 HTML_TOO_MANY_PARAGRAPHS at c0.semantic_content.sections: too many paragraph+task blocks in "
                      "the whole slot (all sections together)", provider.calls[1]["prompt"])
        self.assertIn("(measured 14 paragraph+task blocks; limit 12 paragraph+task blocks)",
                      provider.calls[1]["prompt"])

    async def test_rejected_repair_logs_safe_codes_and_paths(self) -> None:
        body = severity_body()
        bad = self.table_rows(row("Zebra riêng tư", ""))
        still_bad = slot_repair(html_writer(body, bad), 0)
        result, provider, logs = await self.run_unit((WRITER, html_writer(body, bad)), (REPAIR, still_bad))
        self.assertEqual(provider.names, [WRITER, REPAIR])
        self.assertEqual(result["content_origin"], "structured_fallback")
        self.assertTrue(result["unit"]["components"][0]["source_locked_fallback"])
        validation = json.loads(next(line for line in logs if "idm_unit_validation" in line).split(" ", 1)[1])
        self.assertEqual(validation["answer"], "writer")
        self.assertEqual(validation["finding"], "HTML_SEMANTIC_INVALID@components[0]")
        self.assertIn("HTML_ROW_INVALID@components[0].semantic_content.sections[0].blocks[1].rows[3]",
                      validation["html_rules"])
        rejected = json.loads(next(line for line in logs if "idm_call_rejected" in line).split(" ", 1)[1])
        self.assertEqual((rejected["code"], rejected["invocation_kind"]), ("IDM_W5_REPAIR_INVALID", "repair"))
        self.assertEqual(rejected["errors"], ["HTML_SEMANTIC_INVALID@",
                                              "HTML_ROW_INVALID@components[0].semantic_content.sections[0].blocks[1]"
                                              ".rows[3]"])
        self.assertNotIn("Zebra", "".join(logs))

    async def test_rejected_writer_answer_logs_the_binding_code_and_the_retry_names_it(self) -> None:
        body = severity_body()
        extra_field = severity_writer(body)
        extra_field["components"]["c0"]["notes"] = "Zebra"
        result, provider, logs = await self.run_unit((WRITER, extra_field), (WRITER, severity_writer(body)))
        self.assertEqual(result["content_origin"], "provider_validated")
        rejected = json.loads(next(line for line in logs if "idm_call_rejected" in line).split(" ", 1)[1])
        self.assertEqual(rejected["errors"], ["INSTANCE_FIELD_NOT_ALLOWED@components[0]"])
        self.assertIn('{"code":"INSTANCE_FIELD_NOT_ALLOWED","path":"components[0]"}', provider.calls[1]["prompt"])
        self.assertNotIn("Zebra", "".join(logs))

    def test_writer_prompt_states_the_html_rules_up_front_only_for_html_units(self) -> None:
        request = RagLessonAuthorUnitV2Request.model_validate(severity_body())
        writer = IdmUnitWriter(request.unit_contract, parse_brief(request.unit_contract),
                               main._idm_unit_deps(request), make_runtime(None))
        prompt = writer.writer_prompt()
        self.assertLess(prompt.index("HTML SLOT FORMAT"), prompt.index("Writing rules"))
        for expected in ("no HTML tags", "one heading level only", "fills only \"items\"",
                         "at most 12 paragraph+task blocks", "at most 20 bullet items", "30 table rows",
                         "at most 450 words and at most 3150 visible characters"):
            self.assertIn(expected, prompt)
        request = RagLessonAuthorUnitV2Request.model_validate(escalate_body())
        writer = IdmUnitWriter(request.unit_contract, parse_brief(request.unit_contract),
                               main._idm_unit_deps(request), make_runtime(None))
        self.assertNotIn("HTML SLOT FORMAT", writer.writer_prompt())
        bare = unit_writer_prompt("en", course_title="C", audience="A", lesson_title="L", lesson_objective="O",
                                  practice_sentences=[], previous_title=None, next_title=None,
                                  unit_brief={"segment": "example"}, facts=[], context_facts=[])
        self.assertIn("schema.\n\nWriting rules", bare)


if __name__ == "__main__":
    unittest.main()
