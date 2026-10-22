"""Guard: a source-locked slot that commit d4ff677 could build is still buildable.

A slot its facts cannot rebuild answers the ``fallback_only`` retry with 422, and the run ends ``needs_action``
(``UNIT_FALLBACK_INVALID``). A content fix may change what a fallback says, never turn a buildable slot into an
unbuildable one (lane Q2: the one-item-per-fact FAQ rule first made a lone "Lưu ý: X" slot unbuildable).

``tests/fixtures/source_locked_buildable_d4ff677.json`` records, for every input ``buildable_map`` enumerates, whether
the code of d4ff677 built the slot: every fallback-matrix case and every golden unit (each slot as planned, and its
last slot retyped to each source-locked type) in both locales, and the IDM FAQ and worksheet builders over the facts
of the QC fixtures (run ab8d67e1 blocks, the cp2a boundary cases, the CEO canvas). It was written by this module on a
checkout of d4ff677 with this file and the fallback matrix copied in::

    AI_RAG_ENV=test PYTHONPATH=. python -m tests.unit.idm.test_idm_fallback_buildable <out.json>

Re-recorded on 2026-10-22 the same way (d4ff677 code, today's ``tests`` folder) after the golden fixture changed for the
QC feedback A5 (no Hold: md_5 has its own practice lesson lsn_005, the job aid is lsn_006): only the inputs of those two
lessons differ from the first record.
"""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.idm.source_locked import idm_source_faq
from app.idm.worksheet import worksheet_fallback_html
from tests.support.idm_test_support import golden_design, golden_shard
from tests.unit.idm import test_idm_fallback_matrix as matrix
from tests.unit.idm.test_idm_qlt3 import CANVAS_FACTS
from tests.unit.idm.test_idm_storyboard import body_for

FIXTURES = Path(__file__).parents[2] / "fixtures"
BASELINE = FIXTURES / "source_locked_buildable_d4ff677.json"
SOURCE_LOCKED_TYPES = ("problem", "la_faq", "la_sortable", "la_crossword", "la_diagram")
LONE_CAUTION_KEY = ("slot|matrix/la_faq/lsn_003 u2, a lone labelled caution and no other sentence (kept repeated, "
                    "for review)|vi|1")


def _record_slots(result: dict[str, bool], name: str, body: dict[str, Any]) -> None:
    for locale in matrix.LOCALES:
        try:
            slots = matrix.source_locked_slots({**copy.deepcopy(body), "locale": locale, "fallback_only": True})
        except ValidationError:
            return  # a retype the contract refuses (an FAQ that is not the last slot) is no input
        for index, slot in enumerate(slots):
            result[f"slot|{name}|{locale}|{index}"] = slot is not None


def golden_units() -> list[tuple[str, dict[str, Any]]]:
    design, _ = golden_design()
    units = []
    for chapter_index, _module in enumerate(design.modules):
        for lesson_position, lesson in enumerate(golden_shard(chapter_index).lessons):
            for unit_position, _unit in enumerate(lesson.units):
                units.append((f"{lesson.lesson_key}/u{unit_position + 1}",
                              body_for(chapter_index, lesson_position, unit_position)))
    return units


def fixture_fact_groups() -> list[tuple[str, str, list[str]]]:
    """(name, title, facts) of the QC fixtures."""

    blocks = json.loads((FIXTURES / "qlt4_run_ab8d67e1_blocks.json").read_text("utf-8"))
    groups = [(f"qlt4/{chapter}/{block_id}", str(block["name"]), [str(text) for text in block["facts"].values()])
              for chapter, chapter_blocks in blocks["chapters"].items() for block_id, block in chapter_blocks.items()]
    boundary = json.loads((FIXTURES / "ai_id_cp2a_boundary_replay_fixture.json").read_text("utf-8"))
    groups.extend((f"cp2a/{case['id']}", str(case["id"]), [str(fact["fact_text"]) for fact in case["source_facts"]])
                  for case in boundary["cases"])
    groups.append(("qlt3/canvas", "Canvas", list(CANVAS_FACTS)))
    return groups


def buildable_map() -> dict[str, bool]:
    """Whether each enumerated slot or builder input gives source-locked content."""

    result: dict[str, bool] = {}
    for case in (*matrix.VALID_CASES, *matrix.UNBUILDABLE_CASES, *matrix.LAST_RESORT_CASES):
        _record_slots(result, f"matrix/{case.slot_type}/{case.name}", case.build())
    for name, body in golden_units():
        _record_slots(result, f"golden/{name}", body)
        last = len(body["unit_contract"]["component_plan"]) - 1
        for slot_type in SOURCE_LOCKED_TYPES:
            _record_slots(result, f"golden/{name}/as-{slot_type}", matrix.retype(copy.deepcopy(body), last, slot_type))
    for name, title, facts in fixture_fact_groups():
        for locale in matrix.LOCALES:
            result[f"faq|{name}|{locale}"] = idm_source_faq(title, facts, locale=locale) is not None
            result[f"worksheet|{name}|{locale}"] = worksheet_fallback_html(title, facts, None, locale) is not None
    return result


class BuildableGuardTests(unittest.TestCase):
    def test_no_slot_buildable_on_d4ff677_is_unbuildable_now(self) -> None:
        baseline: dict[str, bool] = json.loads(BASELINE.read_text("utf-8"))["buildable"]
        current = buildable_map()
        # Every recorded input is still enumerated, and the guard covers the lone caution and real QC facts.
        self.assertEqual(sorted(set(baseline) - set(current)), [])
        self.assertTrue(baseline[LONE_CAUTION_KEY])
        self.assertTrue(any(key.startswith("faq|qlt4/") and built for key, built in baseline.items()))
        self.assertEqual(sorted(key for key, built in baseline.items() if built and not current[key]), [])


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.write_text(json.dumps({"source": "buildable_map() of tests/unit/idm/test_idm_fallback_buildable.py",
                               "buildable": buildable_map()}, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                   encoding="utf-8")
