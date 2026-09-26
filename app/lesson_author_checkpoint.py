"""Internal checkpoint transport contracts, not persistence or a provider policy.

Node owns selection, immutable snapshots and storage. A unit response is never a
review proposal. Completion must assemble the exact inventory and validate again.
"""
from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

CHECKPOINT_VERSION = 1
MAX_CHECKPOINT_UNITS = 512
MAX_CHECKPOINT_UNIT_BYTES = 2 * 1024 * 1024


class ChapterCheckpointUnit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    unit_index: int = Field(ge=0, lt=MAX_CHECKPOINT_UNITS, strict=True)
    unit: dict[str, Any]

    @field_validator("unit")
    @classmethod
    def validate_payload(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value.get("components"), list) or not value["components"]:
            raise ValueError("CHAPTER_CHECKPOINT_PAYLOAD_INVALID")
        try:
            size = len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"))
        except (TypeError, ValueError) as error:
            raise ValueError("CHAPTER_CHECKPOINT_PAYLOAD_INVALID") from error
        if size > MAX_CHECKPOINT_UNIT_BYTES:
            raise ValueError("CHAPTER_CHECKPOINT_PAYLOAD_TOO_LARGE")
        return value


def checkpoint_expected_units(batches: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    units = [unit for batch in batches for unit in batch]
    if not units or len(units) > MAX_CHECKPOINT_UNITS:
        raise ValueError("CHAPTER_CHECKPOINT_INVENTORY_INVALID")
    paths = [unit.get("unit_path") for unit in units]
    if any(not isinstance(path, str) or not path for path in paths) or len(set(paths)) != len(paths):
        raise ValueError("CHAPTER_CHECKPOINT_INVENTORY_INVALID")
    return units


def select_checkpoint_unit(batches: list[list[dict[str, Any]]], unit_index: int) -> dict[str, Any]:
    units = checkpoint_expected_units(batches)
    if type(unit_index) is not int or not 0 <= unit_index < len(units):
        raise ValueError("CHAPTER_CHECKPOINT_UNIT_OUT_OF_SCOPE")
    return units[unit_index]


def assemble_checkpoint_chapter(
    skeleton: dict[str, Any],
    expected_units: list[dict[str, Any]],
    completed: list[ChapterCheckpointUnit],
) -> dict[str, Any]:
    """Exact index/path assembly; no title matching, merging, inference or dropping."""
    if len(completed) != len(expected_units) or len({u.unit_index for u in completed}) != len(completed):
        raise ValueError("CHAPTER_CHECKPOINT_INCOMPLETE")
    by_index = {u.unit_index: u.unit for u in completed}
    if set(by_index) != set(range(len(expected_units))):
        raise ValueError("CHAPTER_CHECKPOINT_UNIT_OUT_OF_SCOPE")
    result = deepcopy(skeleton)
    chapters = result.get("chapters")
    if not isinstance(chapters, list) or len(chapters) != 1:
        raise ValueError("CHAPTER_CHECKPOINT_INVENTORY_INVALID")
    ordinal = 0
    for lesson_index, lesson in enumerate(chapters[0].get("lessons", []), start=1):
        for unit_index in range(len(lesson.get("units", []))):
            path = f"chapter_1.lesson_{lesson_index}.unit_{unit_index + 1}"
            if ordinal >= len(expected_units) or expected_units[ordinal]["unit_path"] != path:
                raise ValueError("CHAPTER_CHECKPOINT_INVENTORY_INVALID")
            lesson["units"][unit_index] = deepcopy(by_index[ordinal])
            ordinal += 1
    if ordinal != len(expected_units):
        raise ValueError("CHAPTER_CHECKPOINT_INVENTORY_INVALID")
    return result
