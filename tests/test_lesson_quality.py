from __future__ import annotations

import unittest

from app.lesson_quality import (
    detect_generated_content_duplicates,
    pedagogical_validation_result,
    validate_lesson_pedagogical_quality,
)


LONG_EXPLANATION = (
    "Người học cần xác định đúng thiết bị bảo hộ trước khi bắt đầu công việc. "
    "Thiết bị phù hợp làm giảm tiếp xúc với rủi ro được tài liệu mô tả. "
    "Mỗi bước cần được kiểm tra theo điều kiện đã nêu và không được thay bằng lựa chọn không được phê duyệt. "
    "Cảnh báo phải được đọc trước khi người học thực hành hoặc trả lời câu hỏi kiểm tra."
)


def html(facts: list[str], text: str = LONG_EXPLANATION) -> dict:
    return {"type": "html", "html": f"<p>{text}</p>", "source_fact_ids": facts, "covered_source_fact_ids": facts}


def problem(facts: list[str], question: str = "Thiết bị nào cần kiểm tra trước khi làm việc?") -> dict:
    return {
        "type": "problem",
        "question": question,
        "choices": [{"text": "Thiết bị bảo hộ", "correct": True}, {"text": "Vật dụng không liên quan", "correct": False}],
        "source_fact_ids": facts,
        "covered_source_fact_ids": facts,
    }


def unit(components: list[dict], facts: list[str] | None = None) -> dict:
    return {"title": "Thiết bị bảo hộ", "source_fact_ids": facts or ["fact-1", "fact-2"], "components": components}


def proposal(units: list[dict]) -> dict:
    return {"summary": "Pending review", "chapters": [{"title": "An toàn", "lessons": [{"title": "Chuẩn bị", "units": units}]}]}


def blueprint(*, plan: list[dict] | None = None, assessment: bool = True, concepts: list[str] | None = None) -> dict:
    return {
        "chapter_title": "An toàn",
        "lessons": [{
            "title": "Chuẩn bị",
            "learning_objectives": ["Xác định thiết bị bảo hộ phù hợp."],
            "assessment_required": assessment,
            "assessment_objective_refs": ["lo_1"] if assessment else [],
            "units": [{
                "title": "Thiết bị bảo hộ",
                "purpose": "Giải thích và kiểm tra kiến thức an toàn.",
                "concept_ids": concepts or ["concept-ppe"],
                "learning_objective_refs": ["lo_1"],
                "source_fact_ids": ["fact-1", "fact-2"],
                "learning_blocks": [
                    {"id": "lb-explain", "intent": "concept_explanation", "learning_objective_refs": ["lo_1"], "source_fact_ids": ["fact-1", "fact-2"]},
                    {"id": "lb-check", "intent": "knowledge_check", "learning_objective_refs": ["lo_1"], "source_fact_ids": ["fact-1"]},
                ],
                "component_plan": plan or [
                    {"type": "html", "reason_code": "EXPLANATION_DEFAULT", "learning_block_ids": ["lb-explain"]},
                    {"type": "problem", "reason_code": "ASSESS_OBJECTIVE", "learning_block_ids": ["lb-check"]},
                ],
            }],
        }],
    }


class LessonQualityFixturesTests(unittest.TestCase):
    def test_terminology_fixture_supports_crossword_when_contract_requires_it(self):
        crossword = {
            "type": "la_crossword", "source_fact_ids": ["fact-1"],
            "words": [{"answer": "PPE", "clue": "Thiết bị bảo hộ"}, {"answer": "GLOVE", "clue": "Bảo vệ tay"}, {"answer": "MASK", "clue": "Bảo vệ hô hấp"}],
        }
        contract = blueprint(plan=[
            {"type": "html", "reason_code": "EXPLANATION_DEFAULT"},
            {"type": "la_crossword", "reason_code": "TERMINOLOGY_REINFORCEMENT"},
        ], assessment=False)
        report = validate_lesson_pedagogical_quality(proposal([unit([html(["fact-1", "fact-2"]), crossword])]), contract)
        self.assertEqual(report.status, "PASS")

    def test_procedural_fixture_does_not_require_sortable(self):
        contract = blueprint(plan=[{"type": "html", "reason_code": "PROCEDURE_EXPLANATION"}], assessment=False)
        report = validate_lesson_pedagogical_quality(proposal([unit([html(["fact-1", "fact-2"])])]), contract)
        self.assertNotIn("COMPONENT_PURPOSE_INVALID", [item["code"] for item in report.findings])

    def test_conceptual_fixture_covers_objective_and_source(self):
        report = validate_lesson_pedagogical_quality(proposal([unit([html(["fact-1", "fact-2"]), problem(["fact-1"])])]), blueprint())
        self.assertEqual(report.status, "PASS")
        self.assertEqual(report.metrics["objective_coverage"], 1.0)
        self.assertEqual(report.metrics["source_fact_coverage"], 1.0)

    def test_relationship_fixture_requires_approved_diagram_purpose(self):
        diagram = {"type": "la_diagram", "source_fact_ids": ["fact-1", "fact-2"], "nodes": [{"label": "A"}, {"label": "B"}]}
        contract = blueprint(plan=[
            {"type": "html", "reason_code": "EXPLANATION_DEFAULT"},
            {"type": "la_diagram", "reason_code": "RELATIONSHIP_VISUALIZATION"},
        ], assessment=False)
        report = validate_lesson_pedagogical_quality(proposal([unit([html(["fact-1", "fact-2"]), diagram])]), contract)
        self.assertEqual(report.status, "PASS")

    def test_knowledge_check_fixture_requires_teach_then_check_alignment(self):
        report = validate_lesson_pedagogical_quality(proposal([unit([html(["fact-1", "fact-2"]), problem(["fact-1"])])]), blueprint())
        self.assertEqual(report.metrics["assessment_alignment"], 1.0)

    def test_intentional_teach_practice_check_reinforcement_is_not_duplicate(self):
        candidate = proposal([unit([
            html(["fact-1", "fact-2"]),
            {"type": "la_sortable", "source_fact_ids": ["fact-1"], "items": [{"text": "Kiểm tra"}, {"text": "Mang"}, {"text": "Thực hiện"}]},
            problem(["fact-1"]),
        ])])
        self.assertEqual(detect_generated_content_duplicates(candidate), [])

    def test_duplicate_explanation_fixture_is_detected(self):
        candidate = proposal([unit([html(["fact-1", "fact-2"])]), unit([html(["fact-1", "fact-2"])])])
        codes = [issue["code"] for issue in detect_generated_content_duplicates(candidate)]
        self.assertIn("DUPLICATE_EXPLANATION", codes)

    def test_duplicate_quiz_fixture_is_detected(self):
        candidate = proposal([unit([html(["fact-1"]), problem(["fact-1"])]), unit([html(["fact-1"]), problem(["fact-1"])])])
        codes = [issue["code"] for issue in detect_generated_content_duplicates(candidate)]
        self.assertIn("DUPLICATE_QUIZ_QUESTION", codes)

    def test_thin_lesson_fixture_is_rejected(self):
        candidate = proposal([unit([html(["fact-1", "fact-2"], "Thiết bị rất quan trọng."), problem(["fact-1"])])])
        report = validate_lesson_pedagogical_quality(candidate, blueprint(concepts=["concept-ppe", "concept-risk"]))
        self.assertIn("INSUFFICIENT_INSTRUCTIONAL_DEPTH", [issue["code"] for issue in report.findings])

    def test_unsupported_source_gap_fixture_is_rejected(self):
        candidate = proposal([unit([html(["fact-1"]), problem(["fact-3"])])])
        result = pedagogical_validation_result(candidate, blueprint())
        self.assertIn("SOURCE_FACT_NOT_TAUGHT", [issue["code"] for issue in result.issues])


if __name__ == "__main__":
    unittest.main()
