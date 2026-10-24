"""Synthetic finished-course document for the "keep the document as it is" mode (QLT-6 ``preserve_source``).

Four pages of a small feedback course: a document title, a running header on every page, page numbers, two
modules with two lessons each, a bullet list, a table, a numbered sub-heading and lines a learner-content purity
rule would flag (a source line with a page reference, a file name). No customer content, no provider output.
"""

from __future__ import annotations

import hashlib
from typing import Any

from app.contracts.orchestration_v2.provider import SourceSnapshotFactV2
from tests.idm_golden import FakeIdmProvider

DOCUMENT_ID = "00000000-0000-4000-8000-0000000000e1"
SNAPSHOT_HASH = hashlib.sha256(b"idm-preserve-golden-feedback").hexdigest()
EVIDENCE_REVISION = hashlib.sha256(b"idm-preserve-golden-evidence").hexdigest()
EN_DASH = chr(0x2013)
HEADER = "Tài liệu nội bộ — Phòng Đào tạo"

# (page, lines); one source chunk per page.
PAGES: list[tuple[int, list[str]]] = [
    (1, [
        "KHOÁ HỌC: KỸ NĂNG PHẢN HỒI TRONG CÔNG VIỆC",
        HEADER,
        "MODUN 1: VÌ SAO PHẢN HỒI QUAN TRỌNG",
        "Bài 1: Phản hồi là gì",
        "Phản hồi là thông tin về kết quả công việc giúp người nhận biết điều gì đang làm tốt và điều gì cần điều "
        "chỉnh.",
        "Một phản hồi tốt có ba đặc điểm:",
        "• Cụ thể: nêu rõ hành vi và kết quả.",
        "• Kịp thời: đưa ra ngay sau sự việc.",
        "• Hướng tới cải thiện: gợi ý bước tiếp theo.",
        "1",
    ]),
    (2, [
        HEADER,
        "Bài 2: Lợi ích của phản hồi",
        "Theo khảo sát nội bộ năm 2025, nhóm nhận phản hồi hằng tuần hoàn thành mục tiêu cao hơn 18%.",
        "Row 1: Nhóm | Tần suất phản hồi | Tỷ lệ hoàn thành mục tiêu",
        "Row 2: A | Hằng tuần | 92%",
        "Row 3: B | Hằng tháng | 74%",
        "Nguồn: Báo cáo nhân sự 2025, trang 12.",
        "Trang 2",
    ]),
    (3, [
        HEADER,
        "MODUN 2: CÁCH ĐƯA RA PHẢN HỒI",
        "Bài 3: Mô hình SBI",
        "SBI gồm ba bước:",
        f"Bước 1: Situation {EN_DASH} mô tả tình huống cụ thể.",
        f"Bước 2: Behavior {EN_DASH} mô tả hành vi quan sát được.",
        f"Bước 3: Impact {EN_DASH} nêu tác động của hành vi.",
        "3.1 Ví dụ áp dụng",
        "Trong cuộc họp sáng thứ Hai, bạn trình bày số liệu rất rõ, nhờ đó cả nhóm quyết định nhanh hơn.",
        "- 3 -",
    ]),
    (4, [
        HEADER,
        "Bài 4: Những lỗi thường gặp",
        'Nhận xét chung chung như "làm tốt lắm" không giúp người nhận biết cần giữ gì & cần đổi gì.',
        "Phản hồi trước mặt nhiều người dễ khiến người nhận phòng thủ.",
        "Xem thêm tài liệu: huong-dan-phan-hoi.pdf",
        "4",
    ]),
]
# Removed before segmentation: the four page numbers and the header on pages 2-4 (its first occurrence stays).
PAGE_NUMBER_KEYS = ("d1-c1-f10", "d1-c2-f8", "d1-c3-f10", "d1-c4-f6")
REPEATED_HEADER_KEYS = ("d1-c2-f1", "d1-c3-f1", "d1-c4-f1")

# The model's answer: line numbers of the 27 content lines only (no text).
SEGMENTATION: dict[str, Any] = {"chapters": [
    {"title_ordinal": 3, "first_ordinal": 1, "last_ordinal": 15, "lessons": [
        {"title_ordinal": 4, "first_ordinal": 1, "last_ordinal": 9, "unit_title_ordinals": []},
        {"title_ordinal": 10, "first_ordinal": 10, "last_ordinal": 15, "unit_title_ordinals": []}]},
    {"title_ordinal": 16, "first_ordinal": 16, "last_ordinal": 27, "lessons": [
        {"title_ordinal": 17, "first_ordinal": 16, "last_ordinal": 23, "unit_title_ordinals": [22]},
        {"title_ordinal": 24, "first_ordinal": 24, "last_ordinal": 27, "unit_title_ordinals": []}]},
]}


def key(page: int, number: int) -> str:
    return f"d1-c{page}-f{number}"


def source_facts() -> list[SourceSnapshotFactV2]:
    return [SourceSnapshotFactV2(
        document_id=DOCUMENT_ID, fact_key=key(page, number), scope_key=f"scope3_p{page:02d}", fact_text=line,
        source_ref=f"src-{page}", source_page=page, source_chunk=page - 1,
        locator={"source_evidence_status": "ready", "source_evidence_revision": EVIDENCE_REVISION,
                 "scope_title": lines[0]},
    ) for page, lines in PAGES for number, line in enumerate(lines, start=1)]


def content_texts() -> list[str]:
    """The 27 content lines in order (what the model numbers)."""

    removed = {*PAGE_NUMBER_KEYS, *REPEATED_HEADER_KEYS}
    return [fact.fact_text for fact in source_facts() if fact.fact_key not in removed]


def project_context(locale: str = "vi") -> dict[str, Any]:
    return {
        "locale": locale, "course_title_hint": "Kỹ năng phản hồi", "course_description": None,
        "design_mode": "preserve_source", "learner_audience": None, "course_purpose": None,
        "source_documents": [{"document_id": DOCUMENT_ID, "name": "ky-nang-phan-hoi.pdf", "type": "pdf"}],
        "target_audience": None, "learning_objectives": [], "duration_target_minutes": None,
    }


def preserve_provider(*answers: Any) -> FakeIdmProvider:
    """A fake provider answering the segmentation call (the golden answer by default) and nothing else."""

    return FakeIdmProvider({"IdmPreserveSegmentationV1": list(answers) or [SEGMENTATION]})
