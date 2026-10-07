"""Golden fixture "Quy trình xử lý khiếu nại khách hàng" (spec Appendix B).

Synthetic document, not customer data. Every non-empty line is one fact. The
stage responses below are hand-written to the expectations of Appendix B.3 and
are replayed by ``FakeIdmProvider`` so no test reaches the network.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

from app.lesson_author_orchestration_v2_provider import SourceSnapshotFactV2

DOCUMENT_ID = "00000000-0000-4000-8000-0000000000d1"
SNAPSHOT_HASH = hashlib.sha256(b"idm-golden-complaint-handling").hexdigest()
EVIDENCE_REVISION = hashlib.sha256(b"idm-golden-evidence").hexdigest()

# (chunk, lines). Chunk 0 is the title block, 1..11 the numbered sections, 12 the footer.
SOURCE_CHUNKS: list[tuple[int, list[str]]] = [
    (0, [
        "QUY TRÌNH XỬ LÝ KHIẾU NẠI KHÁCH HÀNG",
        "Công ty TNHH Dịch vụ An Phú — Phiên bản 3.2 — Áp dụng từ 01/03/2026",
    ]),
    (1, [
        "1. GIỚI THIỆU",
        "Phòng Chăm sóc Khách hàng được thành lập năm 2009 với 5 nhân sự đầu tiên tại chi nhánh Quận 1.",
        "Năm 2015, phòng mở rộng lên 40 nhân sự và chuyển sang mô hình tổng đài tập trung.",
        "Trong bối cảnh môi trường kinh doanh ngày càng cạnh tranh và nhu cầu khách hàng liên tục thay đổi, việc xử lý "
        "khiếu nại hiệu quả đóng vai trò rất quan trọng đối với sự phát triển bền vững của doanh nghiệp.",
    ]),
    (2, [
        "2. ĐỊNH NGHĨA",
        "Khiếu nại là phản ánh của khách hàng về việc sản phẩm, dịch vụ hoặc hành vi của nhân viên không đáp ứng "
        "cam kết của công ty, kèm yêu cầu được xử lý.",
        "Góp ý không kèm yêu cầu xử lý không được coi là khiếu nại và được chuyển cho bộ phận Marketing.",
    ]),
    (3, [
        "3. PHÂN LOẠI KHIẾU NẠI",
        "Khiếu nại về sản phẩm.",
        "Khiếu nại về dịch vụ lắp đặt.",
        "Khiếu nại về nhân viên.",
        "Khiếu nại về thanh toán.",
        "Khiếu nại về giao hàng.",
    ]),
    (4, [
        "4. MỨC ĐỘ NGHIÊM TRỌNG",
        "Row 1: Cấp độ | Dấu hiệu | Cách xử lý",
        "Row 2: Cấp 1 | Ảnh hưởng thấp, không thiệt hại tài chính | Nhân viên tự xử lý",
        "Row 3: Cấp 2 | Thiệt hại tài chính dưới 50 triệu đồng hoặc khách hàng phàn nàn lần thứ hai | "
        "Thông báo trưởng nhóm",
        "Row 4: Cấp 3 | Liên quan an toàn, pháp lý hoặc truyền thông | Escalate ngay cho quản lý và bộ phận Pháp chế "
        "theo quy trình CR-04",
    ]),
    (5, [
        "5. QUY TRÌNH TIẾP NHẬN",
        "Bước 1: Chào khách hàng và xác nhận danh tính.",
        "Bước 2: Lắng nghe toàn bộ nội dung khiếu nại, không ngắt lời.",
        "Bước 3: Ghi nhận thông tin vào phiếu KN-01.",
        "Bước 4: Xác minh đơn hàng trên hệ thống CRM.",
        "Bước 5: Phân loại khiếu nại theo nhóm.",
        "Bước 6: Đánh giá mức độ nghiêm trọng theo bảng cấp độ.",
        "Bước 7: Quyết định tự xử lý hay escalate theo tiêu chí escalate.",
        "Bước 8: Thông báo cho khách hàng hướng xử lý và thời hạn phản hồi.",
        "Bước 9: Thực hiện xử lý hoặc chuyển hồ sơ.",
        "Bước 10: Phản hồi kết quả cho khách hàng.",
        "Bước 11: Xác nhận khách hàng đồng ý với kết quả.",
        "Bước 12: Đóng hồ sơ và lưu trữ trên CRM.",
    ]),
    (6, [
        "6. TIÊU CHÍ ESCALATE",
        "Bắt buộc escalate khi khiếu nại liên quan đến an toàn của khách hàng.",
        "Bắt buộc escalate khi khiếu nại có yếu tố pháp lý hoặc khách hàng đề cập đến kiện tụng.",
        "Bắt buộc escalate khi giá trị thiệt hại từ 50 triệu đồng trở lên.",
        "Với khách hàng VIP, nhân viên phải thông báo cho trưởng nhóm nhưng không bắt buộc escalate.",
        "Không được hứa bồi thường trước khi khiếu nại được phân loại mức độ.",
    ]),
    (7, [
        "7. THỜI HẠN PHẢN HỒI",
        "Nhân viên cần phản hồi khách hàng trong vòng 24 giờ.",
        "Việc phản hồi phải được thực hiện không quá 24 giờ kể từ khi tiếp nhận.",
        "Theo slide đào tạo năm 2024, khiếu nại cấp 2 được phản hồi trong vòng 48 giờ.",
    ]),
    (8, [
        "8. LỖI THƯỜNG GẶP",
        "Ví dụ: Nhân viên hứa hoàn tiền ngay khi khách hàng vừa gọi đến, trước khi xác minh đơn hàng; công ty phải chi "
        "trả cho một đơn hàng không thuộc diện bồi thường.",
        "Lưu ý: Nhiều nhân viên ghi nhận khiếu nại cấp 3 thành cấp 2 vì không kiểm tra yếu tố truyền thông.",
    ]),
    (9, [
        "9. GHI NHẬN TRÊN HỆ THỐNG CRM",
        "Mở mục Khiếu nại, chọn Tạo mới, chọn mã khiếu nại và đính kèm phiếu KN-01 (xem ảnh chụp màn hình CRM "
        "phiên bản 2021).",
    ]),
    (10, [
        "10. BẢNG MÃ KHIẾU NẠI",
        "Row 1: Mã | Nhóm",
        "Row 2: KN-SP | Sản phẩm",
        "Row 3: KN-DV | Dịch vụ lắp đặt",
        "Row 4: KN-NV | Nhân viên",
        "Row 5: KN-TT | Thanh toán",
        "Row 6: KN-GH | Giao hàng",
    ]),
    (11, [
        "11. ĐẦU MỐI LIÊN HỆ",
        "Bộ phận Pháp chế: máy lẻ 204.",
        "Trưởng nhóm Chăm sóc Khách hàng ca ngày: máy lẻ 110.",
    ]),
    (12, [
        "Tài liệu nội bộ — Trang 1/3",
        "Tài liệu nội bộ — Trang 2/3",
        "Tài liệu nội bộ — Trang 3/3",
    ]),
]


def key(chunk: int, number: int) -> str:
    return f"d1-c{chunk}-f{number}"


def keys(chunk: int, first: int, last: int) -> list[str]:
    return [key(chunk, number) for number in range(first, last + 1)]


def source_facts() -> list[SourceSnapshotFactV2]:
    facts = []
    for chunk, lines in SOURCE_CHUNKS:
        heading = lines[0]
        for number, line in enumerate(lines, start=1):
            facts.append(SourceSnapshotFactV2(
                document_id=DOCUMENT_ID, fact_key=key(chunk, number),
                scope_key=f"scope3_{chunk:02d}", fact_text=line, source_ref=f"src-{chunk}",
                source_page=1 + chunk // 5, source_chunk=chunk,
                locator={"source_evidence_status": "ready", "source_evidence_revision": EVIDENCE_REVISION,
                         "scope_title": heading},
            ))
    return facts


def project_context() -> dict[str, Any]:
    return {
        "locale": "vi", "course_title_hint": "Xử lý khiếu nại khách hàng",
        "source_documents": [{"document_id": DOCUMENT_ID, "name": "quy-trinh-khieu-nai.pdf", "type": "pdf"}],
        "target_audience": None, "learning_objectives": [], "duration_target_minutes": None,
    }


def _block(local_id: str, name: str, intent: str, kind: str, fact_keys: list[str], *,
           support_role: str | None = None, issues: list[dict[str, Any]] | None = None,
           gaps: list[dict[str, Any]] | None = None, sme: list[str] | None = None) -> dict[str, Any]:
    return {"local_id": local_id, "name": name, "summary": f"Nội dung về: {name}", "intent": intent,
            "support_role": support_role, "content_kind": kind, "fact_keys": fact_keys,
            "issues": issues or [], "gaps": gaps or [], "sme_questions": sme or []}


W1_SECTION: dict[str, Any] = {
    "blocks": [
        _block("b1", "Phiên bản và ngày áp dụng của quy trình", "know", "reference", [key(0, 2)]),
        _block("b2", "Phòng Chăm sóc Khách hàng đã phát triển ra sao?", "know", "concept", keys(1, 1, 4)),
        _block("b3", "Thế nào là một khiếu nại?", "know", "concept", keys(2, 1, 3)),
        _block("b4", "Khiếu nại được chia thành những nhóm nào?", "know", "concept", keys(3, 1, 6)),
        _block("b5", "Đánh giá mức độ nghiêm trọng của khiếu nại như thế nào?", "decide", "decision",
               keys(4, 1, 5)),
        _block("b6", "Tiếp nhận thông tin khiếu nại ban đầu", "do", "procedure", keys(5, 1, 5)),
        _block("b7", "Xác định hướng xử lý sau khi phân loại", "do", "procedure", keys(5, 6, 9)),
        _block("b8", "Hoàn tất hồ sơ khiếu nại với khách hàng", "do", "procedure", keys(5, 10, 13)),
        _block("b9", "Khi nào bắt buộc phải escalate khiếu nại?", "decide", "decision", keys(6, 1, 6)),
        _block("b10", "Phải phản hồi khách hàng trong bao lâu?", "know", "policy", keys(7, 1, 4), issues=[
            {"type": "duplicate", "note": "Hai dòng cùng nêu thời hạn 24 giờ.", "fact_keys": [key(7, 2), key(7, 3)]},
            {"type": "conflict", "note": "24 giờ mâu thuẫn với 48 giờ cho khiếu nại cấp 2.",
             "fact_keys": [key(7, 2), key(7, 4)]},
        ], sme=["Khiếu nại cấp 2 phải phản hồi trong 24 giờ hay 48 giờ?"]),
        _block("b11", "Những lỗi nào nhân viên thường mắc khi xử lý khiếu nại?", "know", "case", keys(8, 1, 3),
               support_role="common_mistake"),
        _block("b12", "Ghi nhận khiếu nại trên CRM như thế nào?", "do", "system", keys(9, 1, 2), issues=[
            {"type": "outdated", "note": "Hướng dẫn dựa trên ảnh chụp CRM phiên bản 2021.", "fact_keys": [key(9, 2)]},
        ], gaps=[{"type": "missing_step", "note": "Thiếu các bước trên giao diện CRM hiện tại."}],
            sme=["Giao diện CRM hiện tại còn đúng các bước Tạo mới, chọn mã và đính kèm phiếu KN-01 không?"]),
        _block("b13", "Mã khiếu nại ứng với từng nhóm", "know", "reference", keys(10, 1, 7), support_role="job_aid"),
        _block("b14", "Cần liên hệ ai khi phải escalate?", "know", "reference", keys(11, 1, 3),
               support_role="job_aid"),
    ],
    "noise": [
        {"fact_key": key(0, 1), "reason": "toc_or_title_only"},
        *({"fact_key": item, "reason": "page_furniture"} for item in keys(12, 1, 3)),
    ],
}

W1_REDUCE: dict[str, Any] = {
    "merges": [],
    "conflicts": [],
    "target_audience": {
        "description": "Nhân viên Chăm sóc Khách hàng tuyến đầu tiếp nhận khiếu nại qua tổng đài, đã nắm sản phẩm "
                       "cơ bản và cần xử lý đúng quy trình, đúng thời hạn.",
        "origin": "ai_proposed",
    },
    "learning_objectives": [
        {"lo_id": "lo_1", "statement": "Người học có thể phân loại khiếu nại theo nhóm và mức độ nghiêm trọng dựa "
                                       "trên bảng cấp độ", "bloom": "apply", "origin": "ai_proposed"},
        {"lo_id": "lo_2", "statement": "Người học có thể quyết định tự xử lý hay escalate khiếu nại theo tiêu chí "
                                       "bắt buộc", "bloom": "evaluate", "origin": "ai_proposed"},
        {"lo_id": "lo_3", "statement": "Người học có thể thực hiện quy trình tiếp nhận và xử lý khiếu nại đúng "
                                       "trình tự", "bloom": "apply", "origin": "ai_proposed"},
    ],
    "must_dos": [
        {"must_do_id": "md_1", "lo_id": "lo_1", "statement": "Phân loại khiếu nại theo nhóm", "kind": "do",
         "bloom": "apply"},
        {"must_do_id": "md_2", "lo_id": "lo_1", "statement": "Đánh giá mức độ nghiêm trọng của khiếu nại",
         "kind": "decide", "bloom": "analyze"},
        {"must_do_id": "md_3", "lo_id": "lo_2", "statement": "Quyết định tự xử lý hay escalate", "kind": "decide",
         "bloom": "evaluate"},
        {"must_do_id": "md_4", "lo_id": "lo_3", "statement": "Thực hiện các bước tiếp nhận khiếu nại đúng trình tự",
         "kind": "do", "bloom": "apply"},
        {"must_do_id": "md_5", "lo_id": "lo_3", "statement": "Phản hồi khách hàng đúng thời hạn", "kind": "do",
         "bloom": "apply"},
    ],
    "lo_links": [
        {"block_id": "cb_0001", "lo_id": "lo_3", "relation": "unrelated"},
        {"block_id": "cb_0002", "lo_id": "lo_3", "relation": "unrelated"},
        {"block_id": "cb_0003", "lo_id": "lo_1", "relation": "supporting"},
        {"block_id": "cb_0004", "lo_id": "lo_1", "relation": "direct"},
        {"block_id": "cb_0005", "lo_id": "lo_1", "relation": "direct"},
        {"block_id": "cb_0005", "lo_id": "lo_2", "relation": "supporting"},
        {"block_id": "cb_0006", "lo_id": "lo_3", "relation": "direct"},
        {"block_id": "cb_0007", "lo_id": "lo_3", "relation": "direct"},
        {"block_id": "cb_0008", "lo_id": "lo_3", "relation": "direct"},
        {"block_id": "cb_0009", "lo_id": "lo_2", "relation": "direct"},
        {"block_id": "cb_0010", "lo_id": "lo_3", "relation": "direct"},
        {"block_id": "cb_0011", "lo_id": "lo_2", "relation": "supporting"},
        {"block_id": "cb_0012", "lo_id": "lo_3", "relation": "supporting"},
        {"block_id": "cb_0013", "lo_id": "lo_1", "relation": "context"},
        {"block_id": "cb_0014", "lo_id": "lo_2", "relation": "context"},
    ],
}


def _row(block_id: str, classification: str, treatment: str, *, lo_id: str | None = None,
         must_do_ids: list[str] | None = None, hold_reason: str | None = None,
         sme_question: str | None = None) -> dict[str, Any]:
    placement = {"must_do": "course", "must_know": "course", "reference": "reference_job_aid"}.get(
        classification, "excluded")
    return {"block_id": block_id, "lo_id": lo_id, "must_do_ids": must_do_ids or [],
            "classification": classification, "placement": placement, "treatment": treatment,
            "detail_level": "Giữ phần cần để thực hiện đúng Must Do.", "hold": hold_reason is not None,
            "hold_reason": hold_reason, "sme_question": sme_question, "combine_into": None,
            "separate_into": [], "rationale": "Quyết định theo luồng Week 2."}


W2_BLUEPRINT: dict[str, Any] = {
    "rows": [
        _row("cb_0001", "nice_to_know", "remove"),
        _row("cb_0002", "remove", "remove"),
        _row("cb_0003", "must_know", "condense", lo_id="lo_1", must_do_ids=["md_1"]),
        _row("cb_0004", "must_know", "keep", lo_id="lo_1", must_do_ids=["md_1"]),
        _row("cb_0005", "must_know", "keep", lo_id="lo_1", must_do_ids=["md_2", "md_3"]),
        _row("cb_0006", "must_do", "condense", lo_id="lo_3", must_do_ids=["md_4"]),
        _row("cb_0007", "must_do", "condense", lo_id="lo_3", must_do_ids=["md_4"]),
        _row("cb_0008", "must_do", "condense", lo_id="lo_3", must_do_ids=["md_4"]),
        _row("cb_0009", "must_do", "keep", lo_id="lo_2", must_do_ids=["md_3"]),
        _row("cb_0010", "must_know", "rewrite", lo_id="lo_3", must_do_ids=["md_5"],
             hold_reason="Thời hạn 24 giờ mâu thuẫn với 48 giờ cho khiếu nại cấp 2",
             sme_question="Khiếu nại cấp 2 phải phản hồi trong 24 giờ hay 48 giờ?"),
        _row("cb_0011", "must_know", "condense", lo_id="lo_2", must_do_ids=["md_3"]),
        _row("cb_0012", "must_know", "rewrite", lo_id="lo_3", must_do_ids=["md_4"],
             hold_reason="Hướng dẫn CRM dựa trên ảnh chụp phiên bản 2021, có thể đã lỗi thời",
             sme_question="Giao diện CRM hiện tại còn đúng các bước Tạo mới, chọn mã và đính kèm phiếu không?"),
        _row("cb_0013", "reference", "convert_to_job_aid", lo_id="lo_1"),
        _row("cb_0014", "reference", "convert_to_job_aid", lo_id="lo_2"),
    ],
    "blocked_must_do_ids": ["md_5"],
}


def _lesson(lesson_key: str, kind: str, title: str, primary: str | None, block_ids: list[str],  # noqa: PLR0917
            screens: int, minutes: int) -> dict[str, Any]:
    return {"lesson_key": lesson_key, "kind": kind, "title": title, "primary_must_do_id": primary,
            "secondary_must_do_ids": [], "block_ids": block_ids, "est_screens": screens, "est_minutes": minutes,
            "ordering_rationale": "Nền tảng trước, theo thứ tự công việc thực tế."}


W4_COURSE: dict[str, Any] = {
    "course_title": "Xử lý khiếu nại khách hàng đúng quy trình",
    "course_summary": "Khoá học giúp nhân viên Chăm sóc Khách hàng phân loại, đánh giá và quyết định hướng xử lý "
                      "khiếu nại đúng tiêu chí, tránh hứa hẹn sai và escalate đúng lúc.",
    "assessment_strategy": "Kiểm tra nhanh sau phần Must Know, bài tình huống quyết định escalate cuối mỗi mục.",
    "prerequisites": ["Biết sử dụng tổng đài và hệ thống CRM ở mức cơ bản"],
    "modules": [
        {"module_key": "mod_01", "title": "Phân loại và đánh giá mức độ khiếu nại",
         "performance_goal": "Phân loại đúng nhóm và mức độ của mọi khiếu nại tiếp nhận", "lo_ids": ["lo_1"],
         "lessons": [
             _lesson("lsn_001", "learning", "Phân loại khiếu nại theo nhóm", "md_1", ["cb_0003", "cb_0004"], 4, 6),
             _lesson("lsn_002", "learning", "Đánh giá mức độ nghiêm trọng", "md_2", ["cb_0005"], 4, 6),
         ]},
        {"module_key": "mod_02", "title": "Quyết định escalate khiếu nại",
         "performance_goal": "Quyết định đúng khi nào tự xử lý và khi nào phải escalate", "lo_ids": ["lo_2"],
         "lessons": [
             _lesson("lsn_003", "learning", "Quyết định tự xử lý hay escalate", "md_3", ["cb_0009", "cb_0011"], 6, 10),
         ]},
        {"module_key": "mod_03", "title": "Tiếp nhận khiếu nại theo quy trình",
         "performance_goal": "Thực hiện đúng trình tự tiếp nhận và xử lý khiếu nại", "lo_ids": ["lo_3"],
         "lessons": [
             _lesson("lsn_004", "learning", "Thực hiện quy trình tiếp nhận khiếu nại", "md_4",
                     ["cb_0006", "cb_0007", "cb_0008"], 6, 10),
             _lesson("lsn_005", "job_aid", "Bảng tra mã khiếu nại và đầu mối liên hệ", None,
                     ["cb_0013", "cb_0014"], 2, 3),
         ]},
    ],
}


class FakeUsage:
    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.inputTokens = input_tokens
        self.outputTokens = output_tokens
        self.totalTokens = input_tokens + output_tokens


class FakeIdmProvider:
    """Replays JSON responses per stage in call order and records prompts.

    ``responses`` maps a schema title (the server model name) to a list of
    payloads or callables ``(prompt) -> payload``; an ``Exception`` instance is
    raised instead of answering.
    """

    def __init__(self, responses: dict[str, list[Any]]) -> None:
        self.responses = {name: list(items) for name, items in responses.items()}
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, api_key: str, model: str, prompt: str, **kwargs: Any) -> tuple[str, FakeUsage]:
        schema = kwargs["response_schema"]
        name = schema.model_json_schema().get("title", schema.__name__)
        self.calls.append({"schema": name, "prompt": prompt, **{k: v for k, v in kwargs.items()
                                                                  if k != "on_provider_telemetry"}})
        queue = self.responses.get(name)
        if not queue:
            raise AssertionError(f"unexpected provider call for {name}")
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        payload = item(prompt) if callable(item) else item
        text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        return text, FakeUsage(len(prompt) // 4, len(text) // 4)


def golden_provider(
    overrides: dict[str, list[Any]] | None = None,
) -> FakeIdmProvider:
    responses: dict[str, list[Any]] = {
        "IdmW1SectionResponseV1": [W1_SECTION],
        "IdmW1ReduceResponseV1": [W1_REDUCE],
        "IdmW2BlueprintResponseV1": [W2_BLUEPRINT],
        "IdmW4CourseResponseV1": [W4_COURSE],
    }
    responses.update(overrides or {})
    return FakeIdmProvider(responses)


Mutator = Callable[[dict[str, Any]], dict[str, Any]]


def mutate(payload: dict[str, Any], change: Mutator) -> dict[str, Any]:
    return change(json.loads(json.dumps(payload, ensure_ascii=False)))
