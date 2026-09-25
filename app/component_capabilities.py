"""Node-owned, opt-in component instance contract. Never provider authority."""
from typing import Any, Literal
import re

from pydantic import BaseModel, ConfigDict, StrictBool


class ComponentCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal[2]
    max_components_per_unit: Literal[4]
    max_assessments_per_unit: Literal[3]
    assessment_enabled: StrictBool


def component_capabilities(value: Any) -> ComponentCapabilities | None:
    return ComponentCapabilities.model_validate(value) if value is not None else None


def validate_instance_plan(plans: list[dict[str, Any]]) -> None:
    """Instance-bearing plans are bounded and lossless, never type-deduped."""
    if not 1 <= len(plans) <= 4:
        raise ValueError("COMPONENT_PLAN_CAPACITY_EXCEEDED")
    seen: set[str] = set()
    for plan in plans:
        identifier = plan.get("component_plan_id")
        if not isinstance(identifier, str) or not re.fullmatch(r"cp2_[a-f0-9]{32}", identifier) or identifier in seen:
            raise ValueError("COMPONENT_PLAN_INSTANCE_INVALID")
        seen.add(identifier)
