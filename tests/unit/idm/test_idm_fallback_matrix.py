"""Fallback test matrix of the IDM unit pipeline: every slot type under ``fallback_only=True``.

The durable ``fallback_only`` retry is the last chance of an IDM unit. It makes no provider call and returns the
source-locked fallback, so a fallback that Python or Node rejects loses the unit (run c2e5ac41). For every slot
type (``html``, ``problem``, ``la_faq``, ``la_sortable``, ``la_crossword``, ``la_diagram`` and the worksheet, an
``html`` slot planned as a practice) the matrix posts a unit through the real
``/v1/lesson-author/orchestration-v2/unit`` endpoint, in both locales, and requires one of two answers:

* a valid unit: the shared staged validator (``strict_payload=True`` with the writer's fallback ``expected``) and the
  mirror of Node's revision-0 acceptance find nothing, and the envelope is the provider-free deterministic one;
* ``422 ORCHESTRATION_V2_UNIT_FALLBACK_INVALID`` when the source facts genuinely cannot rebuild a slot (its
  source-locked builder returns ``None``), never an invalid unit.

Every type has at least one valid case. A case the golden design lacks is derived from a golden unit: a slot is
retyped (plan type and purpose, brief type and role) and, where the golden facts hold no such structure, the unit's
fact texts are rewritten; the brief and contract hashes are then recomputed. The provider is a local fake with empty
queues, so nothing reaches the network and any provider call fails the case.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

from app.idm.contracts import brief_hash_of
from app.idm.node_acceptance import node_acceptance_findings
from app.idm.storyboard import acceptance_context, build_idm_expected, parse_brief
from app.schemas.orchestration_v2 import RagLessonAuthorUnitV2Request
from app.services.orchestration_v2 import unit as unit_service
from app.services.staged_unit.validation import validate_staged_unit_content
from tests.golden import idm_golden_unit as gu
from tests.support.idm_test_support import rehash_contract
from tests.unit.idm.test_idm_storyboard import (
    StoryboardEndpointTestCase,
    body_for,
    escalate_body,
    severity_body,
    worksheet_body,
)

FALLBACK_INVALID = "ORCHESTRATION_V2_UNIT_FALLBACK_INVALID"
LOCALES = ("vi", "en")
SLOT_TYPES = frozenset({"html", "problem", "la_faq", "la_sortable", "la_crossword", "la_diagram", "worksheet"})
# The brief role a retyped slot gets (what W4 plans for that type).
ROLE_BY_TYPE = {"html": "explain", "problem": "practice", "la_faq": "clarify", "la_sortable": "practice",
                "la_crossword": "practice", "la_diagram": "show"}

# lsn_003 unit 1 (golden chunk 6, escalate rules) rewritten as three "Term: definition" facts: the golden source has
# no such definitions ("Khiếu nại là ..." has no colon), so it cannot build a crossword.
DEFINITION_FACTS = {
    "d1-c6-f2": "Khiếu nại: phản ánh của khách hàng về sản phẩm hoặc dịch vụ không đáp ứng cam kết, kèm yêu cầu "
                "xử lý.",
    "d1-c6-f3": "Góp ý: ý kiến của khách hàng không kèm yêu cầu xử lý, được chuyển cho bộ phận Marketing.",
    "d1-c6-f4": "Escalate: chuyển khiếu nại lên quản lý khi có yếu tố an toàn, pháp lý hoặc truyền thông.",
}
# The same unit with one arrow chain "A -> B -> C -> D" among its prose rules.
ARROW_FACTS = {"d1-c6-f2": "Tiếp nhận khiếu nại -> Phân loại mức độ -> Quyết định escalate -> Phản hồi khách hàng"}
# The same unit as a form to fill in: two field labels, then one guiding question per field.
FORM_FACTS = {
    "d1-c6-f1": "6. PHIẾU QUYẾT ĐỊNH ESCALATE",
    "d1-c6-f2": "Hãy trả lời ngắn gọn vào từng ô của phiếu dưới đây.",
    "d1-c6-f3": "YẾU TỐ AN TOÀN",
    "d1-c6-f4": "YẾU TỐ PHÁP LÝ",
    "d1-c6-f5": "Khiếu nại có liên quan đến an toàn của khách hàng không?",
    "d1-c6-f6": "Khách hàng có đề cập đến kiện tụng không?",
}


# --- request builders ------------------------------------------------------------------------------
def rehashed(body: dict[str, Any]) -> dict[str, Any]:
    """Recompute the brief hash, then the contract hash, as Node does after any change."""

    brief = body["unit_contract"]["idm_unit_brief"]
    brief["brief_hash"] = brief_hash_of(brief)
    rehash_contract(body["unit_contract"])
    return body


def retype(body: dict[str, Any], index: int, slot_type: str) -> dict[str, Any]:
    """Slot ``index`` planned as ``slot_type``: plan type and purpose, brief type and role (a practice is kept only
    on a practice role, so a retyped teaching slot never turns into a worksheet)."""

    contract = body["unit_contract"]
    contract["component_plan"][index].update(type=slot_type, purpose=gu.PURPOSE_BY_TYPE[slot_type])
    slot = contract["idm_unit_brief"]["components"][index]
    slot.update(type=slot_type, role=ROLE_BY_TYPE[slot_type])
    if slot["role"] != "practice":
        slot["practice"] = None
    return rehashed(body)


def with_fact_texts(body: dict[str, Any], texts: dict[str, str]) -> dict[str, Any]:
    facts = body["unit_contract"]["source_facts"]
    assert set(texts) <= {fact["fact_key"] for fact in facts}
    for fact in facts:
        fact["fact_text"] = texts.get(fact["fact_key"], fact["fact_text"])
    return rehashed(body)


def escalate_rules() -> dict[str, Any]:
    """lsn_003 unit 1: one html slot on the escalate rules (prose: no steps, table, arrow or definition)."""

    return body_for(1, 0, 0)


def procedure_steps() -> dict[str, Any]:
    """lsn_004 unit 1: one html slot on "Bước 1" ... "Bước 8" of the intake procedure."""

    return body_for(2, 0, 0)


def job_aid_table() -> dict[str, Any]:
    """lsn_005 unit 1: one html slot on the complaint code table and the contacts."""

    return body_for(2, 1, 0)


def form_worksheet() -> dict[str, Any]:
    """lsn_003 unit 1 as a worksheet whose facts hold a form: the structured worksheet fallback."""

    body = with_fact_texts(escalate_rules(), FORM_FACTS)
    practice = escalate_body()["unit_contract"]["idm_unit_brief"]["components"][0]["practice"]
    body["unit_contract"]["idm_unit_brief"]["components"][0].update(role="practice", practice=practice)
    return rehashed(body)


@dataclasses.dataclass(frozen=True)
class Case:
    slot_type: str
    name: str
    build: Callable[[], dict[str, Any]]
    # The slot whose type is under test.
    index: int = 0
    # Text the fallback html of that slot must hold (proves which builder produced it).
    html_marker: str | None = None


VALID_CASES = (
    Case("html", "golden lsn_003 u1, escalate rules", escalate_rules),
    Case("html", "golden lsn_004 u1, procedure steps", procedure_steps),
    Case("html", "golden lsn_005 u1, code table and contacts", job_aid_table),
    Case("problem", "golden lsn_002 u1, severity table question", severity_body, 1),
    Case("problem", "lsn_004 u1 retyped, first step of the procedure", lambda: retype(procedure_steps(), 0, "problem")),
    Case("problem", "lsn_005 u1 retyped, code table question", lambda: retype(job_aid_table(), 0, "problem")),
    Case("la_faq", "lsn_003 u2 with its problem slot retyped html, labelled cautions",
         lambda: retype(escalate_body(), 0, "html"), 1),
    Case("la_sortable", "golden lsn_004 u2, html + sortable on steps 9-12", lambda: body_for(2, 0, 1), 1),
    Case("la_sortable", "lsn_004 u1 retyped, steps 1-8", lambda: retype(procedure_steps(), 0, "la_sortable")),
    Case("la_crossword", "lsn_003 u1 retyped, Term: definition facts",
         lambda: retype(with_fact_texts(escalate_rules(), DEFINITION_FACTS), 0, "la_crossword")),
    Case("la_diagram", "lsn_004 u1 retyped, numbered steps", lambda: retype(procedure_steps(), 0, "la_diagram")),
    Case("la_diagram", "lsn_004 u2 sortable retyped, supporting steps", lambda: retype(body_for(2, 0, 1), 1,
                                                                                      "la_diagram"), 1),
    Case("la_diagram", "lsn_003 u1 retyped, arrow chain A -> B -> C -> D",
         lambda: retype(with_fact_texts(escalate_rules(), ARROW_FACTS), 0, "la_diagram")),
    Case("worksheet", "golden worksheet_body, reference table (plain source html)", worksheet_body),
    Case("worksheet", "lsn_003 u1 as a worksheet on form facts (structured form)", form_worksheet,
         html_marker="<td>Yếu tố pháp lý</td>"),
)

UNBUILDABLE_CASES = (
    Case("problem", "golden lsn_001 u1, prose definitions and categories", lambda: body_for(0, 0, 0), 1),
    Case("problem", "golden lsn_003 u2, prose common mistakes", escalate_body),
    Case("la_faq", "lsn_003 u1 retyped, a single explicit condition", lambda: retype(escalate_rules(), 0, "la_faq")),
    Case("la_sortable", "lsn_003 u1 retyped, rules without an ordered sequence",
         lambda: retype(escalate_rules(), 0, "la_sortable")),
    Case("la_sortable", "lsn_005 u1 retyped, table rows are not steps", lambda: retype(job_aid_table(), 0,
                                                                                       "la_sortable")),
    Case("la_crossword", "lsn_003 u1 retyped, rules without definitions",
         lambda: retype(escalate_rules(), 0, "la_crossword")),
    Case("la_crossword", "lsn_001 u1 problem retyped, 'X là ...' definitions without a colon",
         lambda: retype(body_for(0, 0, 0), 1, "la_crossword"), 1),
    Case("la_diagram", "lsn_003 u1 retyped, rules without steps or arrows",
         lambda: retype(escalate_rules(), 0, "la_diagram")),
    # The IDM diagram builder draws step chains only (app.idm.diagram): a reference table is not a sequence.
    Case("la_diagram", "lsn_005 u1 retyped, reference table", lambda: retype(job_aid_table(), 0, "la_diagram")),
)


def fallback_request(case: Case, locale: str) -> dict[str, Any]:
    # Each builder returns a fresh request, so cases never share state.
    return {**case.build(), "locale": locale, "fallback_only": True}


# --- validators ------------------------------------------------------------------------------------
def source_locked_slots(body: dict[str, Any]) -> list[dict[str, Any] | None]:
    """The per-slot source-locked rebuilds exactly as the endpoint builds them (``None``: not buildable)."""

    deps = unit_service._idm_unit_deps(RagLessonAuthorUnitV2Request.model_validate(body))
    return list(deps.source_locked_components or [])


def unit_findings(body: dict[str, Any], unit: dict[str, Any]) -> tuple[tuple[str, str] | None, list[str]]:
    """(shared staged validator finding, Node acceptance findings) of a returned unit.

    ``expected`` mirrors ``IdmUnitWriterSteps.__init__``: source-locked content restates whole facts, so the fallback
    is validated without the IDM teaching budget, and Node measures its budget on provider content only. The unit is
    checked as returned (with ``idm_quality``), which is what Node receives.
    """

    request = RagLessonAuthorUnitV2Request.model_validate(body)
    contract = request.unit_contract
    brief = parse_brief(contract)
    deps = unit_service._idm_unit_deps(request)
    expected = {**build_idm_expected(contract, brief, deps, request.locale), "instructional_output_budget": None}
    finding = validate_staged_unit_content(unit, expected, strict_payload=True)
    node = node_acceptance_findings(unit, acceptance_context(contract, brief), provider_validated=False)
    return (None if finding is None else (finding.code, finding.path)), [item.as_log() for item in node]


# --- tests -----------------------------------------------------------------------------------------
class FallbackMatrixTests(StoryboardEndpointTestCase):
    def test_matrix_covers_every_slot_type(self) -> None:
        self.assertEqual({case.slot_type for case in VALID_CASES}, SLOT_TYPES)
        self.assertEqual({case.slot_type for case in UNBUILDABLE_CASES}, SLOT_TYPES - {"html", "worksheet"})

    async def test_buildable_slots_give_a_valid_provider_free_unit(self) -> None:
        for case in VALID_CASES:
            for locale in LOCALES:
                with self.subTest(type=case.slot_type, case=case.name, locale=locale):
                    await self.check_valid(fallback_request(case, locale), case)

    async def test_unbuildable_slots_answer_422_never_an_invalid_unit(self) -> None:
        for case in UNBUILDABLE_CASES:
            for locale in LOCALES:
                with self.subTest(type=case.slot_type, case=case.name, locale=locale):
                    await self.check_unbuildable(fallback_request(case, locale), case)

    async def check_valid(self, body: dict[str, Any], case: Case) -> None:
        slots = source_locked_slots(body)
        self.assertNotIn(None, slots)
        status, data, provider = await self.post(body)
        self.assertEqual((status, provider.calls), (200, []), data.get("detail"))
        self.assert_fallback_envelope(body, data)
        unit = data["unit"]
        self.assertEqual(unit_findings(body, unit), (None, []))
        if case.html_marker is not None:
            self.assertIn(case.html_marker, unit["components"][case.index]["html"])

    async def check_unbuildable(self, body: dict[str, Any], case: Case) -> None:
        # The 422 must come from a slot the facts cannot rebuild, not from a rebuilt unit the validators reject.
        self.assertIsNone(source_locked_slots(body)[case.index])
        status, data, provider = await self.post(body)
        self.assertEqual((status, data.get("detail", {}).get("code"), provider.calls), (422, FALLBACK_INVALID, []))

    def assert_fallback_envelope(self, body: dict[str, Any], data: dict[str, Any]) -> None:
        quality = self.assert_envelope(data, "structured_fallback", "review_required", "deterministic_fallback")
        # The durable replay is provider-free, so its (zero) usage is complete.
        self.assertIs(data["usage_complete"], True)
        self.assertEqual(data["usage"]["totalTokens"], 0)
        self.assertEqual([event["event_code"] for event in data["attempt_trace"]], ["deterministic_fallback"])
        self.assertEqual(quality.judge_status, "not_run")
        unit = data["unit"]
        components = unit["components"]
        self.assertTrue(unit.get("source_locked_fallback") is True
                        or all(component.get("source_locked_fallback") is True for component in components))
        plans = body["unit_contract"]["component_plan"]
        self.assertEqual([(item["type"], item["component_plan_id"]) for item in components],
                         [(plan["type"], plan["component_plan_id"]) for plan in plans])
        self.assertEqual(unit["component_plan"], plans)
