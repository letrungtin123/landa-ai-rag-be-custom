"""``UnitGenerationContractV2.idm_unit_brief`` and the contract hash (spec §8.3, §11.2)."""

from __future__ import annotations

import copy
import unittest
from typing import Any

from pydantic import ValidationError

from app.idm.contracts import brief_hash_of
from app.lesson_author_orchestration_v2 import canonical_hash
from app.lesson_author_orchestration_v2_provider import UnitGenerationContractV2
from app.main import RagLessonAuthorUnitV2Request
from tests import idm_golden as g
from tests import idm_golden_unit as gu
from tests.idm_test_support import golden_design, golden_shard

HASH_INVALID = "ORCHESTRATION_V2_UNIT_CONTRACT_HASH_INVALID"


def idm_body() -> dict[str, Any]:
    design, _ = golden_design()
    return gu.build_unit_request(design, golden_shard(1), chapter_index=1, lesson_position=0, unit_position=1,
                                 facts=g.source_facts())


def hashed(contract: dict[str, Any]) -> dict[str, Any]:
    return {**contract, "contract_hash": canonical_hash({k: v for k, v in contract.items() if k != "contract_hash"})}


def legacy_contract() -> dict[str, Any]:
    """The same unit as Node built it before IDM: no ``idm_unit_brief`` key at all."""

    contract = {k: v for k, v in idm_body()["unit_contract"].items() if k != "idm_unit_brief"}
    return hashed(contract)


class UnitContractHashTests(unittest.TestCase):
    def test_legacy_contract_hash_is_unchanged_by_the_new_field(self) -> None:
        legacy = legacy_contract()
        model = UnitGenerationContractV2.model_validate(legacy)
        self.assertIsNone(model.idm_unit_brief)
        # The hash input of a legacy contract is exactly the pre-IDM wire (the None field is excluded).
        self.assertEqual(model.model_dump(mode="json", exclude={"contract_hash", "idm_unit_brief"}),
                         {k: v for k, v in legacy.items() if k != "contract_hash"})
        explicit = UnitGenerationContractV2.model_validate({**legacy, "idm_unit_brief": None})
        self.assertEqual(explicit.contract_hash, legacy["contract_hash"])
        with_none_in_hash = canonical_hash({**{k: v for k, v in legacy.items() if k != "contract_hash"},
                                            "idm_unit_brief": None})
        with self.assertRaisesRegex(ValidationError, HASH_INVALID):
            UnitGenerationContractV2.model_validate({**legacy, "contract_hash": with_none_in_hash})

    def test_legacy_contract_without_policy_marker_still_validates(self) -> None:
        legacy = {k: v for k, v in legacy_contract().items() if k != "unit_content_policy_version"}
        plans = legacy["component_plan"]
        # Pre-alignment contracts: the first component owns every fact.
        plans[0] = {**plans[0], "source_fact_ids": list(legacy["unit_source_fact_ids"]),
                    "supporting_evidence_fact_ids": []}
        plans[1] = {**plans[1], "source_fact_ids": list(legacy["unit_source_fact_ids"]),
                    "supporting_evidence_fact_ids": []}
        legacy = hashed(legacy)
        for extra in ({}, {"idm_unit_brief": None}, {"unit_content_policy_version": None}):
            model = UnitGenerationContractV2.model_validate({**legacy, **extra})
            self.assertEqual(model.contract_hash, legacy["contract_hash"])

    def test_brief_is_part_of_the_hash(self) -> None:
        contract = idm_body()["unit_contract"]
        model = UnitGenerationContractV2.model_validate(contract)
        self.assertEqual(model.idm_unit_brief, contract["idm_unit_brief"])
        legacy_hash = legacy_contract()["contract_hash"]
        self.assertNotEqual(contract["contract_hash"], legacy_hash)
        with self.assertRaisesRegex(ValidationError, HASH_INVALID):
            UnitGenerationContractV2.model_validate({**contract, "contract_hash": legacy_hash})
        changed = copy.deepcopy(contract)
        changed["idm_unit_brief"]["unit_purpose"] = "Người học quyết định đúng khi nào escalate."
        changed["idm_unit_brief"]["brief_hash"] = brief_hash_of(changed["idm_unit_brief"])
        with self.assertRaisesRegex(ValidationError, HASH_INVALID):
            UnitGenerationContractV2.model_validate(changed)
        rehashed = hashed(changed)
        self.assertNotEqual(rehashed["contract_hash"], contract["contract_hash"])
        UnitGenerationContractV2.model_validate(rehashed)

    def test_wrong_hash_is_rejected(self) -> None:
        for contract in (legacy_contract(), idm_body()["unit_contract"]):
            with self.assertRaisesRegex(ValidationError, HASH_INVALID):
                UnitGenerationContractV2.model_validate({**contract, "contract_hash": "d" * 64})

    def test_unit_request_accepts_both_shapes(self) -> None:
        body = idm_body()
        self.assertIsNotNone(RagLessonAuthorUnitV2Request.model_validate(body).unit_contract.idm_unit_brief)
        legacy = {**body, "unit_contract": legacy_contract()}
        self.assertIsNone(RagLessonAuthorUnitV2Request.model_validate(legacy).unit_contract.idm_unit_brief)


if __name__ == "__main__":
    unittest.main()
