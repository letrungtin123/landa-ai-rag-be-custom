"""Test-only bridge from the Node unit contract to the real Stage-2 adapter."""

from __future__ import annotations

import json
import sys

from app.lesson_author_orchestration_v2_provider import (
    UnitGenerationContractV2,
    unit_contract_manifest_v2,
    unit_contract_v5_architecture_v2,
)
from app.services.lesson_author.staged.source_locked import build_staged_instructional_contract


def main() -> None:
    payload = json.loads(sys.stdin.read())
    contract = UnitGenerationContractV2.model_validate(payload["unit_contract"])
    locale = payload.get("locale", "vi")
    manifest = unit_contract_manifest_v2(contract, locale=locale)
    architecture = unit_contract_v5_architecture_v2(contract)
    expected = architecture["lessons"][0]["units"][0]
    expected.update({
        "source_fact_ids": list(contract.unit_source_fact_ids),
        "supporting_evidence_fact_ids": list(dict.fromkeys(
            fact_id
            for plan in contract.component_plan
            for fact_id in plan.supporting_evidence_fact_ids
        )),
        "learning_objectives": list(contract.lesson_learning_objectives),
        "learning_objective_refs": list(contract.unit_learning_objective_refs),
        "strict_v5_evidence": True,
    })
    writer_contract = build_staged_instructional_contract(expected, manifest)
    print(json.dumps({
        "manifest_fact_ids": [fact["fact_id"] for fact in manifest["facts"]],
        "represented_fact_count": manifest["represented_fact_count"],
        "writer_source_evidence_bundle": writer_contract["source_evidence_bundle"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
