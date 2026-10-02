import unittest

from app.lesson_author_orchestration_v2 import canonical_hash
from app.lesson_author_orchestration_v2_provider import (
    UnitGenerationContractV2,
    unit_contract_v5_architecture_v2,
)
from app.main import (
    build_orchestration_v2_source_locked_unit,
    validate_staged_unit_content,
)


class OrchestrationV2SourceLockedFallbackTests(unittest.TestCase):
    def test_sparse_source_fragments_produce_validator_safe_sortable_steps(self) -> None:
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
            for index, (fact_id, text) in enumerate(zip(fact_ids, ["mở đầu", "thực hành", "tổng kết"]))
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
        self.assertTrue(all(source in text for source, text in zip(["mở đầu", "thực hành", "tổng kết"], texts)))

        expected = unit_contract_v5_architecture_v2(contract)["lessons"][0]["units"][0]
        expected["component_types"] = [plan["type"] for plan in plans]
        self.assertIsNone(validate_staged_unit_content(unit, expected, strict_payload=True))


if __name__ == "__main__":
    unittest.main()
