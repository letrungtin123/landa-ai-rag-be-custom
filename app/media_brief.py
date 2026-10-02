"""Source-exact, bounded media briefs. Never invent or auto-translate evidence."""
import re
from typing import Any


def build_media_brief(unit_title: str, media_type: str, evidence_texts: list[str], locale: str) -> dict[str, Any] | None:
    # Whole canonical excerpts, never substring-truncated. These are not claimed
    # to be complete sentences or translated scripts; upstream facts may be
    # fragments. Skip obvious footer/title-only cards, never invent relations.
    candidates = []
    for raw in evidence_texts:
        text = re.sub(r"\s+", " ", raw).strip()
        if not 24 <= len(text) <= 500 or len(text.split()) < 3:
            continue
        if re.match(r"^(?:trang|page|chương|chapter)\s+\d+\b", text, re.I) or text.isupper():
            continue
        if text not in candidates:
            candidates.append(text)
    if not candidates:
        return None
    # Select a bounded subset in source order spanning the unit, not ownership
    # allocation. No facts are assigned/reassigned by this presentation helper.
    count = min(6, len(candidates))
    indices = [round(i * (len(candidates) - 1) / (count - 1)) for i in range(count)] if count > 1 else [0]
    points = [candidates[i] for i in indices]
    english = locale == "en"
    video = media_type == "video"
    title = (("Proposed walkthrough: " if video else "Proposed visual summary: ") if english else ("Video đề xuất: " if video else "Infographic đề xuất: ")) + unit_title
    context = (
        f'Presentation suggestion for "{unit_title}": use one scene per point, display the source wording, and highlight the action or relationship explicitly stated. Do not invent a case, outcome or sequence.' if video else
        f'Presentation suggestion for "{unit_title}": use one labelled panel per point. Preserve source categories and comparison labels; connect panels only where the source establishes a relationship.'
    ) if english else (
        f'Gợi ý dàn dựng cho "{unit_title}": mỗi ý là một cảnh, hiển thị nội dung nguồn và làm nổi bật hành động hoặc quan hệ được nêu rõ. Không tự dựng case, kết quả hay thứ tự ngoài nguồn.' if video else
        f'Gợi ý bố cục cho "{unit_title}": mỗi ý là một panel có nhãn riêng. Giữ đúng nhóm kiến thức và nhãn so sánh; chỉ nối các panel khi nguồn xác lập quan hệ.'
    )
    context += (f' Source anchor for this brief (original language): “{points[0]}”'
                if english else f' Nội dung nguồn làm điểm tựa cho brief này (ngôn ngữ gốc): “{points[0]}”')
    return {
        "type": media_type, "title": title[:180], "brief_version": 2,
        "content_points": points, "context_description": context,
        "evidence_language": "original", "content_basis": "SOURCE_EXCERPTS",
        "content_outline": ("Source excerpt (original language): " if english else "Trích nguồn (ngôn ngữ gốc): ") + points[0],
        "rationale": "A bounded source-backed visual brief, not a generated asset." if english else "Brief trực quan có cơ sở nguồn; chưa phải video hoặc hình ảnh đã tạo.",
    }
