"""Server-side single-choice hygiene (``app.idm.mcq``, QC course 364564, N2).

The QC run had the correct answer at position B in 8 of 9 questions, the longest option in 8 of 9,
"A./B." prefixes on 3 of 9 and explanations written as "A - ...; B - ...". Nothing here calls a provider.
"""

from __future__ import annotations

import re
import unittest
from collections import Counter
from typing import Any

from app.idm.mcq import (
    LABELS_STRIPPED_CODE,
    OPTION_LETTERS,
    RELABELLED_CODE,
    SHUFFLED_CODE,
    answer_length_cue,
    normalize_single_choice,
    option_order,
    relabel_explanation,
)
from app.idm.node_acceptance import node_acceptance_findings
from app.idm.storyboard import acceptance_context, parse_brief
from app.lesson_author_orchestration_v2_provider import UnitGenerationContractV2
from tests.test_idm_storyboard import (
    JUDGE,
    WRITER,
    FakeGenerate,
    StoryboardEndpointTestCase,
    judge,
    severity_body,
    severity_writer,
)

REASONS = ("hứa bồi thường trước khi phân loại là sai quy định",
           "yếu tố an toàn là tiêu chí bắt buộc escalate",
           "quy định VIP chỉ áp dụng khi không có yếu tố bắt buộc",
           "trưởng nhóm không thay được bộ phận Pháp chế")
TEXTS = ("Hứa hoàn tiền ngay để giữ chân vị khách VIP này",
         "Escalate ngay cho quản lý vì liên quan an toàn",
         "Chỉ thông báo trưởng nhóm vì đây là khách VIP",
         "Chuyển hồ sơ cho trưởng nhóm tự quyết định sau")


def question(explanation: str | None = None, *, texts: tuple[str, ...] = TEXTS, correct: int = 1,
             kind: str = "multiple_choice") -> dict[str, Any]:
    """A provider question in the QC pattern: correct at B, explanation "A — ...; B — ...; ..."."""

    if explanation is None:
        explanation = "Tiêu chí: an toàn bắt buộc escalate. " + "; ".join(
            f"{OPTION_LETTERS[index]} — {'đúng' if index == correct else 'sai'} vì {REASONS[index]}"
            for index in range(len(texts))) + "."
    return {"type": "problem", "problem_type": kind, "question": "Khách báo sản phẩm gây chập điện. Bạn làm gì?",
            "choices": [{"text": text, "correct": index == correct} for index, text in enumerate(texts)],
            "explanation": explanation}


def seed_with(count: int, predicate: Any) -> str:
    return next(seed for seed in (f"cp2_{index:032x}" for index in range(500)) if predicate(option_order(seed, count)))


def segment_of(explanation: str, letter: str) -> str:
    match = re.search(rf"{letter} — ([^;.]*)", explanation)
    return match.group(1) if match else ""


class OptionOrderTests(unittest.TestCase):
    def test_order_is_a_reproducible_permutation_that_depends_on_the_seed(self) -> None:
        self.assertEqual(option_order("cp2_x", 4), option_order("cp2_x", 4))
        self.assertEqual(sorted(option_order("cp2_x", 4)), [0, 1, 2, 3])
        self.assertEqual(len({tuple(option_order(f"cp2_{index}", 4)) for index in range(200)}), 24)
        self.assertEqual(option_order("cp2_x", 1), [0])

    def test_the_correct_answer_no_longer_sits_at_b(self) -> None:
        # QC course 364564: 8 of 9 correct answers at B. Over many plans each position gets its share.
        positions = Counter(option_order(f"cp2_{index:032x}", 4).index(1) for index in range(400))
        self.assertEqual(set(positions), {0, 1, 2, 3})
        self.assertTrue(all(70 <= count <= 130 for count in positions.values()), positions)


class NormalizeSingleChoiceTests(unittest.TestCase):
    def test_shuffle_moves_each_explanation_segment_with_its_option(self) -> None:
        source = question()
        seed = seed_with(4, lambda order: order.index(1) != 1 and order != [0, 1, 2, 3])
        result, codes = normalize_single_choice(source, seed)
        self.assertEqual(dict(codes), {SHUFFLED_CODE: 1, RELABELLED_CODE: 1})
        reason_of = {text: REASONS[index] for index, text in enumerate(TEXTS)}
        for position, choice in enumerate(result["choices"]):
            letter = OPTION_LETTERS[position]
            self.assertEqual(segment_of(result["explanation"], letter),
                             f"{'đúng' if choice['correct'] else 'sai'} vì {reason_of[choice['text']]}")
        # Segments are re-ordered A, B, C, D and the closing full stop stays at the end.
        self.assertEqual(re.findall(r"([A-D]) —", result["explanation"]), ["A", "B", "C", "D"])
        self.assertTrue(result["explanation"].startswith("Tiêu chí: an toàn bắt buộc escalate. A — "))
        self.assertTrue(result["explanation"].endswith("."))
        self.assertNotIn(".;", result["explanation"])
        self.assertEqual(sorted(choice["text"] for choice in result["choices"]), sorted(TEXTS))
        self.assertEqual(sum(choice["correct"] for choice in result["choices"]), 1)
        self.assertEqual(source, question())  # the provider payload is not mutated

    def test_sentence_segments_and_a_closing_remark(self) -> None:
        seed = seed_with(3, lambda order: order == [2, 0, 1])
        sentences = "A — sai vì X1. B — đúng vì Y1. C — sai vì Z1."
        result, _ = normalize_single_choice(question(sentences, texts=TEXTS[:3]), seed)
        self.assertEqual(result["explanation"], "A — sai vì Z1. B — sai vì X1. C — đúng vì Y1.")
        # A closing remark after the last option would travel with it: relabel in place only.
        closing = "A — sai vì X1; B — đúng vì Y1; C — sai vì Z1. Hãy đối chiếu tiêu chí trước."
        result, _ = normalize_single_choice(question(closing, texts=TEXTS[:3]), seed)
        self.assertEqual(result["explanation"],
                         "B — sai vì X1; C — đúng vì Y1; A — sai vì Z1. Hãy đối chiếu tiêu chí trước.")

    def test_labels_in_prose_follow_their_option(self) -> None:
        mapping = {"A": "C", "B": "A", "C": "B"}
        text = ("Đáp án đúng là B. Phương án A và C đều sai; (B) đúng vì an toàn. A sai vì hứa trước, C không đúng. "
                "Answer B is right, options A, C are wrong.")
        self.assertEqual(relabel_explanation(text, mapping),
                         "Đáp án đúng là A. Phương án C và B đều sai; (A) đúng vì an toàn. C sai vì hứa trước, B không "
                         "đúng. Answer A is right, options C, B are wrong.")
        # Names and articles are not option labels.
        untouched = "Hàng loại A cần kiểm kê trước, nhóm B sau. A manager checks Plan C first."
        self.assertEqual(relabel_explanation(untouched, mapping), untouched)

    def test_option_label_prefixes_are_removed(self) -> None:
        texts = ("A. Hứa hoàn tiền ngay để giữ chân khách", "B) Escalate ngay cho quản lý vì an toàn",
                 "(C) Chỉ thông báo trưởng nhóm trực ca", "D - Chuyển hồ sơ cho trưởng nhóm sau")
        result, codes = normalize_single_choice(question(texts=texts), "cp2_seed")
        self.assertEqual(codes[LABELS_STRIPPED_CODE], 4)
        self.assertEqual(sorted(choice["text"] for choice in result["choices"]),
                         sorted(("Hứa hoàn tiền ngay để giữ chân khách", "Escalate ngay cho quản lý vì an toàn",
                                 "Chỉ thông báo trưởng nhóm trực ca", "Chuyển hồ sơ cho trưởng nhóm sau")))
        # A lone letter that does not label the options in order stays ("A.I.", one prefix only, "B." first).
        for kept in (("A.I. giúp phân loại khiếu nại", *TEXTS[1:]), ("A. Hứa hoàn tiền ngay", *TEXTS[1:]),
                     ("B. Hứa hoàn tiền ngay cho khách", "A. Escalate ngay cho quản lý", *TEXTS[2:])):
            result, codes = normalize_single_choice(question(texts=kept), "cp2_seed")
            self.assertEqual(codes[LABELS_STRIPPED_CODE], 0)
            self.assertEqual(sorted(choice["text"] for choice in result["choices"]), sorted(kept))

    def test_other_problem_shapes_are_left_alone(self) -> None:
        for value in (question(kind="multiple_select"), {**question(), "choices": None},
                      {**question(), "choices": ["a", "b"]}, {**question(), "choices": [{"text": "x"}]},
                      {**question(), "choices": [{"text": str(index), "correct": False} for index in range(7)]}):
            result, codes = normalize_single_choice(value, "cp2_seed")
            self.assertIs(result, value)
            self.assertEqual(codes, Counter())
        identity = seed_with(3, lambda order: order == [0, 1, 2])
        value = question(texts=TEXTS[:3])
        result, codes = normalize_single_choice(value, identity)
        self.assertEqual((result, codes), (value, Counter()))


class AnswerLengthCueTests(unittest.TestCase):
    def test_correct_option_far_longer_than_every_other(self) -> None:
        # The QC escalate question: 60 characters against 45 (1.33).
        long_answer = ("Hứa hoàn tiền ngay để giữ chân khách VIP",
                       "Escalate ngay cho quản lý vì khiếu nại liên quan đến an toàn",
                       "Chỉ thông báo trưởng nhóm vì đây là khách VIP")
        self.assertTrue(answer_length_cue(question(texts=long_answer)))
        self.assertFalse(answer_length_cue(question(texts=TEXTS)))
        self.assertFalse(answer_length_cue(question(texts=long_answer, correct=0)))  # a short correct option
        self.assertFalse(answer_length_cue(question(texts=long_answer[:2])))  # two options: no cue to read
        two_correct = question(texts=long_answer)
        two_correct["choices"][0]["correct"] = True
        self.assertFalse(answer_length_cue(two_correct))


class NodeAcceptanceOfShuffledUnitsTests(StoryboardEndpointTestCase):
    async def test_shuffled_unit_passes_node_acceptance(self) -> None:
        body = severity_body()
        writer = severity_writer(body)
        status, data, _ = await self.post(body, FakeGenerate(**{WRITER: [writer], JUDGE: [judge()]}))
        self.assertEqual(status, 200)
        self.assertEqual((data["content_origin"], data["quality_state"]), ("provider_validated", "validated"))
        problem = data["unit"]["components"][1]
        provider_choices = writer["components"]["c1"]["choices"]
        self.assertNotEqual(problem["choices"], provider_choices)
        self.assertEqual([choice["text"] for choice in problem["choices"]],
                         [provider_choices[old]["text"] for old in option_order(problem["component_plan_id"], 3)])
        correct = next(index for index, choice in enumerate(problem["choices"]) if choice["correct"])
        self.assertIn(f"{OPTION_LETTERS[correct]} — đúng vì khớp dấu hiệu cấp 2", problem["explanation"])
        contract = UnitGenerationContractV2.model_validate(body["unit_contract"])
        context = acceptance_context(contract, parse_brief(contract))
        unit = {key: value for key, value in data["unit"].items() if key != "idm_quality"}
        self.assertEqual(node_acceptance_findings(unit, context, provider_validated=True), [])


if __name__ == "__main__":
    unittest.main()
