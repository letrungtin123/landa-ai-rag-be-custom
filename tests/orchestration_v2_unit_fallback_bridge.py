"""Offline Python boundary used by Node's V2 fallback compatibility test."""

import json
import sys

from app.main import build_orchestration_v2_source_locked_unit
from app.lesson_author_orchestration_v2_provider import UnitGenerationContractV2


def main() -> None:
    payload = json.load(sys.stdin)
    contract = UnitGenerationContractV2.model_validate(payload["contract"])
    unit = build_orchestration_v2_source_locked_unit(contract, payload.get("locale", "vi"))
    if unit is None:
        raise RuntimeError("ORCHESTRATION_V2_UNIT_FALLBACK_UNAVAILABLE")
    print(json.dumps(unit, ensure_ascii=False))


if __name__ == "__main__":
    main()
