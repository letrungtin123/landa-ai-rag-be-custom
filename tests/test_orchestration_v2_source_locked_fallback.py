import unittest

from app.lesson_author_orchestration_v2 import canonical_hash
from app.lesson_author_orchestration_v2_provider import (
    UnitGenerationContractV2,
    unit_contract_v5_architecture_v2,
)
from app.main import (
    build_orchestration_v2_source_locked_components,
    build_orchestration_v2_source_locked_unit,
    merge_checkpoint_component_fallback,
    validate_staged_unit_content,
)


class OrchestrationV2SourceLockedFallbackTests(unittest.TestCase):
    def test_html_fallback_strips_joined_source_locators_and_stays_valid(self) -> None:
        fact_ids = ["fact-1", "fact-2"]
        source_facts = [{
            "document_id": "document-1",
            "fact_key": fact_id,
            "scope_key": "scope-1",
            "fact_text": text,
            "source_ref": f"Trang {index + 27}",
            "source_page": index + 27,
            "source_chunk": index,
            "locator": {},
        } for index, (fact_id, text) in enumerate(zip(fact_ids, [
            "Trang 27: Nhận diện mối nguy trước khi bắt đầu công việc.",
            "Đánh giá mức rủi ro trước khi chọn biện pháp (slide 28, chunk 4).",
        ]))]
        plans = [{
            "component_plan_id": "cp2_" + "d" * 32,
            "type": "html",
            "title": "Slide 27 - Quy trình đánh giá",
            "rationale": "Giải thích dữ kiện được giao",
            "purpose": "explain",
            "source_fact_ids": fact_ids,
            "supporting_evidence_fact_ids": [],
            "learning_objective_refs": ["lo_1"],
            "source_scope_ids": ["scope-1"],
            "content_requirements": [],
            "learning_block_ids": [],
            "required_artifacts": [],
        }]
        payload = {
            "contract_version": 2,
            "source_snapshot_hash": "a" * 64,
            "assembly_hash": "b" * 64,
            "chapter_key": "chapter-7",
            "unit_path": "chapter_7.lesson_1.unit_3",
            "chapter_title": "Chương 7",
            "lesson_title": "Mục 1",
            "lesson_learning_objectives": ["Đánh giá rủi ro"],
            "unit_title": "Trang 27: Quy trình đánh giá",
            "unit_purpose": "Slide 27 - Giải thích quy trình",
            "unit_learning_objective_refs": ["lo_1"],
            "unit_source_scope_ids": ["scope-1"],
            "unit_source_fact_ids": fact_ids,
            "component_plan": plans,
            "source_facts": source_facts,
        }
        contract = UnitGenerationContractV2.model_validate({
            **payload,
            "contract_hash": canonical_hash(payload),
        })

        unit = build_orchestration_v2_source_locked_unit(contract, "vi")

        self.assertIsNotNone(unit)
        assert unit is not None
        visible = f"{unit['title']} {unit['components'][0]['title']} {unit['components'][0]['html']}"
        self.assertNotRegex(visible.casefold(), r"\b(?:trang|slide|chunk)\s*\d+")
        self.assertIn("Nhận diện mối nguy", visible)
        self.assertIn("Đánh giá mức rủi ro", visible)
        expected = unit_contract_v5_architecture_v2(contract)["lessons"][0]["units"][0]
        expected["component_types"] = ["html"]
        self.assertIsNone(validate_staged_unit_content(unit, expected, strict_payload=True))

    def test_explicit_source_sequence_produces_instructional_sortable_steps(self) -> None:
        fact_ids = ["fact-1", "fact-2", "fact-3"]
        source_facts = [
            {
                "document_id": "document-1",
                "fact_key": fact_id,
                "scope_key": "scope-1",
                "fact_text": text,
                "source_ref": "Trang 1",
                "source_page": 1,
                "source_chunk": index,
                "locator": {},
            }
            for index, (fact_id, text) in enumerate(zip(fact_ids, [
                "Bước 1: Kiểm tra điều kiện an toàn trước khi bắt đầu.",
                "Bước 2: Mang thiết bị bảo hộ phù hợp với mối nguy.",
                "Bước 3: Thực hiện công việc theo trình tự đã phê duyệt.",
            ]))
        ]
        plans = [
            {
                "component_plan_id": "cp2_" + "a" * 32,
                "type": "html",
                "title": "Nội dung nguồn",
                "rationale": "Giải thích toàn bộ dữ kiện được giao",
                "purpose": "explain",
                "source_fact_ids": fact_ids,
                "supporting_evidence_fact_ids": [],
                "learning_objective_refs": ["lo_1"],
                "source_scope_ids": ["scope-1"],
                "content_requirements": [],
                "learning_block_ids": [],
                "required_artifacts": [],
            },
            {
                "component_plan_id": "cp2_" + "b" * 32,
                "type": "la_sortable",
                "title": "Trình tự rà soát",
                "rationale": "Giữ đúng thứ tự hiển thị của dữ kiện nguồn",
                "purpose": "sequence",
                "source_fact_ids": fact_ids,
                "supporting_evidence_fact_ids": [],
                "learning_objective_refs": ["lo_1"],
                "source_scope_ids": ["scope-1"],
                "content_requirements": [],
                "learning_block_ids": [],
                "required_artifacts": [],
            },
        ]
        payload = {
            "contract_version": 2,
            "source_snapshot_hash": "a" * 64,
            "assembly_hash": "b" * 64,
            "chapter_key": "chapter-1",
            "unit_path": "chapter_1.lesson_1.unit_1",
            "chapter_title": "Chương 1",
            "lesson_title": "Mục 1",
            "lesson_learning_objectives": ["Rà soát nội dung nguồn"],
            "unit_title": "Bài học 1",
            "unit_purpose": "Rà soát các ý theo đúng thứ tự nguồn",
            "unit_learning_objective_refs": ["lo_1"],
            "unit_source_scope_ids": ["scope-1"],
            "unit_source_fact_ids": fact_ids,
            "component_plan": plans,
            "source_facts": source_facts,
        }
        contract = UnitGenerationContractV2.model_validate({
            **payload,
            "contract_hash": canonical_hash(payload),
        })

        unit = build_orchestration_v2_source_locked_unit(contract, "vi")
        self.assertIsNotNone(unit)
        assert unit is not None
        sortable = unit["components"][1]
        texts = [item["text"] for item in sortable["items"]]
        self.assertEqual(len(texts), 3)
        self.assertTrue(all(len(text) >= 14 and not text[:1].islower() for text in texts))
        self.assertEqual(len({text.casefold() for text in texts}), len(texts))
        self.assertTrue(all(source in text for source, text in zip(
            ["Kiểm tra điều kiện", "Mang thiết bị", "Thực hiện công việc"], texts,
        )))

        expected = unit_contract_v5_architecture_v2(contract)["lessons"][0]["units"][0]
        expected["component_types"] = [plan["type"] for plan in plans]
        self.assertIsNone(validate_staged_unit_content(unit, expected, strict_payload=True))

    def test_density_v3_interaction_uses_supporting_evidence_without_claiming_teaching_ownership(self) -> None:
        source_values = [
            "Câu hỏi: Việc nào phải thực hiện trước khi bắt đầu công việc?",
            "A. Kiểm tra điều kiện an toàn tại nơi làm việc",
            "B. Bỏ qua mối nguy đã nhận diện",
            "C. Chờ đến khi xảy ra sự cố mới kiểm tra",
            "Đáp án: A",
            "Giải thích: Tài liệu yêu cầu kiểm tra điều kiện an toàn trước khi công việc bắt đầu.",
        ]
        fact_ids = [f"fact-{index}" for index in range(1, len(source_values) + 1)]
        source_facts = [{
            "document_id": "document-1",
            "fact_key": fact_id,
            "scope_key": "scope-1",
            "fact_text": text,
            "source_ref": "Trang 1",
            "source_page": 1,
            "source_chunk": index,
            "locator": {"instructional_density_policy_version": "unit-content-v3-density-1"},
        } for index, (fact_id, text) in enumerate(zip(fact_ids, source_values))]
        common = {
            "learning_objective_refs": ["lo_1"],
            "source_scope_ids": ["scope-1"],
            "content_requirements": [],
            "learning_block_ids": [],
            "required_artifacts": [],
        }
        plans = [{
            **common,
            "component_plan_id": "cp2_" + "a" * 32,
            "type": "html",
            "title": "Nội dung nguồn",
            "rationale": "Giải thích dữ kiện nguồn",
            "purpose": "explain",
            "source_fact_ids": fact_ids,
            "supporting_evidence_fact_ids": [],
        }, {
            **common,
            "component_plan_id": "cp2_" + "b" * 32,
            "type": "problem",
            "title": "Kiểm tra hiểu",
            "rationale": "Kiểm tra sau phần giải thích",
            "purpose": "assess",
            "source_fact_ids": [],
            "supporting_evidence_fact_ids": fact_ids,
        }]
        payload = {
            "contract_version": 2,
            "source_snapshot_hash": "a" * 64,
            "assembly_hash": "b" * 64,
            "chapter_key": "chapter-1",
            "unit_path": "chapter_1.lesson_1.unit_1",
            "chapter_title": "Chương 1",
            "lesson_title": "Mục 1",
            "lesson_learning_objectives": ["Thực hiện an toàn"],
            "unit_title": "Bài học 1",
            "unit_purpose": "Giải thích và kiểm tra quy trình an toàn",
            "unit_learning_objective_refs": ["lo_1"],
            "unit_source_scope_ids": ["scope-1"],
            "unit_source_fact_ids": fact_ids,
            "component_plan": plans,
            "source_facts": source_facts,
        }
        contract = UnitGenerationContractV2.model_validate({
            **payload,
            "contract_hash": canonical_hash(payload),
        })
        unit = build_orchestration_v2_source_locked_unit(contract, "vi")
        self.assertIsNotNone(unit)
        assert unit is not None
        self.assertEqual(unit["components"][0]["covered_source_fact_ids"], fact_ids)
        self.assertEqual(unit["components"][1]["source_fact_ids"], [])
        self.assertEqual(unit["components"][1]["covered_source_fact_ids"], [])
        self.assertEqual(unit["components"][1]["supporting_evidence_fact_ids"], fact_ids)
        self.assertEqual(unit["components"][1]["problem_type"], "multiple_choice")
        self.assertEqual(sum(choice["correct"] for choice in unit["components"][1]["choices"]), 1)
        self.assertEqual(unit["supporting_evidence_fact_ids"], fact_ids)
        self.assertNotIn("Cách rà soát bản nháp", unit["components"][0]["html"])
        expected = unit_contract_v5_architecture_v2(contract)["lessons"][0]["units"][0]
        expected["unit_title"] = expected["title"]
        expected["component_types"] = [plan["type"] for plan in plans]
        self.assertIsNone(validate_staged_unit_content(unit, expected, strict_payload=True))

        provider_html = dict(unit["components"][0])
        provider_html["html"] = (
            "<h2>Nội dung do mô hình tạo</h2>"
            "<p>Trước khi bắt đầu công việc, người thực hiện cần kiểm tra đầy đủ điều kiện an toàn "
            "và nhận diện các mối nguy có thể phát sinh trong phạm vi được giao.</p>"
            "<p>Thiết bị bảo hộ phải phù hợp với từng mối nguy; sau đó công việc được tiến hành "
            "theo đúng trình tự đã phê duyệt và các điểm kiểm soát liên quan.</p>"
            "<p>Trong quá trình thực hiện, mọi thay đổi bất thường phải được dừng lại để đánh giá, "
            "báo cáo cho người phụ trách và chỉ tiếp tục khi biện pháp kiểm soát đã được xác nhận.</p>"
            "<p>Khi hoàn tất, nhóm thực hiện kiểm tra hiện trường, thu hồi dụng cụ và ghi nhận kết quả "
            "để bảo đảm các yêu cầu an toàn đã được tuân thủ nhất quán.</p>"
        )
        provider_html.pop("source_locked_fallback", None)
        provider_html["selection_rationale"] = "Provider-authored explanation"
        invalid_problem = dict(unit["components"][1])
        invalid_problem["covered_source_fact_ids"] = ["fact-1"]
        baseline = {**unit, "components": [provider_html, invalid_problem]}
        mixed = merge_checkpoint_component_fallback(baseline, unit, [1], expected)
        self.assertEqual(mixed["components"][0], provider_html)
        self.assertTrue(mixed["components"][1]["source_locked_fallback"])
        self.assertEqual(mixed["components"][1]["covered_source_fact_ids"], [])
        self.assertIsNone(validate_staged_unit_content(mixed, expected, strict_payload=True))

    def test_per_slot_components_isolate_an_unbuildable_slot(self) -> None:
        """Regression (run 2a5e9ff2): a prose-only problem has no deterministic rebuild; the html slot keeps one."""
        fact_ids = ["fact-1", "fact-2"]
        source_facts = [{
            "document_id": "document-1", "fact_key": fact_id, "scope_key": "scope-1", "fact_text": text,
            "source_ref": None, "source_page": None, "source_chunk": None, "locator": {},
        } for fact_id, text in zip(fact_ids, [
            "Nhân viên lắng nghe khách hàng và ghi nhận đầy đủ nội dung phản ánh trong ca làm việc.",
            "Quản lý cửa hàng xem xét phản ánh và phản hồi cho khách hàng trong thời gian sớm nhất.",
        ], strict=True)]
        base = {"rationale": "r", "learning_objective_refs": ["lo_1"], "source_scope_ids": ["scope-1"],
                "content_requirements": [], "learning_block_ids": [], "required_artifacts": []}
        plans = [
            {**base, "component_plan_id": "cp2_" + "a" * 32, "type": "html", "title": "Tiếp nhận phản ánh",
             "purpose": "explain", "source_fact_ids": fact_ids, "supporting_evidence_fact_ids": []},
            {**base, "component_plan_id": "cp2_" + "b" * 32, "type": "problem", "title": "Kiểm tra nhanh",
             "purpose": "assess", "source_fact_ids": [], "supporting_evidence_fact_ids": fact_ids},
        ]
        payload = {
            "contract_version": 2, "unit_content_policy_version": "unit-content-v4-alignment-1",
            "source_snapshot_hash": "a" * 64, "assembly_hash": "b" * 64,
            "chapter_key": "chapter-1", "unit_path": "chapter_1.lesson_1.unit_2", "chapter_title": "Chương 1",
            "lesson_title": "Bài 1", "lesson_learning_objectives": ["Tiếp nhận phản ánh"],
            "unit_title": "Tiếp nhận phản ánh", "unit_purpose": "Giải thích cách tiếp nhận phản ánh",
            "unit_learning_objective_refs": ["lo_1"], "unit_source_scope_ids": ["scope-1"],
            "unit_source_fact_ids": fact_ids, "component_plan": plans, "source_facts": source_facts,
        }
        contract = UnitGenerationContractV2.model_validate({**payload, "contract_hash": canonical_hash(payload)})

        components = build_orchestration_v2_source_locked_components(contract, "vi")

        self.assertEqual([component is None for component in components], [False, True])
        assert components[0] is not None
        self.assertEqual((components[0]["type"], components[0]["source_locked_fallback"]), ("html", True))
        # The whole-unit fallback stays all-or-nothing (legacy behaviour unchanged).
        self.assertIsNone(build_orchestration_v2_source_locked_unit(contract, "vi"))
        html_only = contract.model_copy(update={"component_plan": contract.component_plan[:1]})
        self.assertEqual(build_orchestration_v2_source_locked_unit(html_only, "vi"),
                         build_orchestration_v2_source_locked_unit(
                             html_only, "vi", components=build_orchestration_v2_source_locked_components(html_only, "vi")))


if __name__ == "__main__":
    unittest.main()
