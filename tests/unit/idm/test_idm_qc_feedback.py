"""QC feedback of 2026-10-21 (owner-approved change list A1-A7): the unit-level rules of the AI service.

A5 (no Hold): a worksheet whose practice criteria the AI suggested from what the lesson teaches heads its self-check
as suggested criteria, and the unit note says so. Every provider is a local fake.
"""

from __future__ import annotations

from typing import Any

from app.idm.contracts import brief_hash_of
from app.idm.worksheet import label_suggested_self_check
from tests.support.idm_test_support import rehash_contract
from tests.unit.idm.test_idm_storyboard import (
    JUDGE,
    WRITER,
    FakeGenerate,
    StoryboardEndpointTestCase,
    judge,
    worksheet_body,
    worksheet_writer,
)

SUGGESTED = "Tiêu chí gợi ý: cấp độ theo dấu hiệu bài học nêu cho từng cấp"


def suggested_worksheet_body() -> dict[str, Any]:
    """The worksheet unit whose practice has suggested criteria (its feedback criterion carries the label)."""

    body = worksheet_body()
    brief = body["unit_contract"]["idm_unit_brief"]
    for slot in brief["components"]:
        if slot.get("practice"):
            slot["practice"]["feedback_focus"]["criterion"] = SUGGESTED
    brief["brief_hash"] = brief_hash_of(brief)
    rehash_contract(body["unit_contract"])
    return body


class SuggestedCriteriaTests(StoryboardEndpointTestCase):
    async def test_the_self_check_of_suggested_criteria_says_so(self) -> None:
        body = suggested_worksheet_body()
        provider = FakeGenerate(**{WRITER: [worksheet_writer(body)], JUDGE: [judge()]})
        status, data, _ = await self.post(body, provider)
        self.assertEqual(status, 200)
        sections = data["unit"]["components"][0]["semantic_content"]["sections"]
        self.assertEqual(sections[-1]["heading"], "Tự kiểm tra bảng của bạn (tiêu chí gợi ý)")
        self.assertIn('"Tự kiểm tra (tiêu chí gợi ý)" / "Self-check (suggested criteria)"', provider.calls[0]["prompt"])
        note = data["unit"]["idm_quality"]["author_note"]
        self.assertIn("AI gợi ý tiêu chí từ nội dung đã dạy (ghi là tiêu chí gợi ý)", note)

    async def test_source_criteria_keep_their_heading_and_note(self) -> None:
        body = worksheet_body()
        provider = FakeGenerate(**{WRITER: [worksheet_writer(body)], JUDGE: [judge()]})
        _status, data, _ = await self.post(body, provider)
        sections = data["unit"]["components"][0]["semantic_content"]["sections"]
        self.assertEqual(sections[-1]["heading"], "Tự kiểm tra bảng của bạn")
        self.assertNotIn("tiêu chí gợi ý", data["unit"]["idm_quality"]["author_note"])

    def test_the_label_is_added_once_and_fits_the_heading_bound(self) -> None:
        semantic = {"sections": [{"heading": "Nhiệm vụ", "blocks": [{"kind": "task", "text": "Làm."}]},
                                 {"heading": "Tự kiểm tra", "blocks": [{"kind": "bullets", "items": ["a", "b"]}]}]}
        labelled = label_suggested_self_check(semantic, "vi", 120)
        assert labelled is not None
        self.assertEqual(labelled["sections"][1]["heading"], "Tự kiểm tra (tiêu chí gợi ý)")
        self.assertEqual(labelled["sections"][0], semantic["sections"][0])
        self.assertIsNone(label_suggested_self_check(labelled, "vi", 120))
        long = {"sections": [{"heading": "x" * 110, "blocks": [{"kind": "bullets", "items": ["a"]}]}]}
        english = label_suggested_self_check(long, "en", 120)
        self.assertEqual(english["sections"][0]["heading"] if english else None, "Self-check (suggested criteria)")
        self.assertIsNone(label_suggested_self_check({"sections": [{"heading": "a", "blocks": []}]}, "vi", 120))
