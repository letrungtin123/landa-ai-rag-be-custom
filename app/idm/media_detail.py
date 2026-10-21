"""The designer brief of a suggested infographic or video (QC feedback 2026-10-21, A4).

QC: the "Mô tả bối cảnh" (``media_brief.context_description``) of a suggestion must let a designer draw it: the
layout, the elements, the text and numbers shown on the image, the arrows and their direction, each tied to a content
point, with a minimum length. The W3/W4 prompt asks for that brief (``prompts_module._MEDIA_BRIEF_RULE``);
``media_brief_gaps`` measures what a description lacks (a lesson warning, listed in a repair the module answer gets
anyway); ``complete_media_brief`` appends a layout built from the content points' own words when the description
stays thin.
The completion never adds a fact, a number or a relation: arrows are drawn only where the points are steps or a cause
and its effect, as the points themselves say.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final

from app.contracts.orchestration_v2.blueprint import ArchitectureMediaBriefV2
from app.idm.policy import IDM_MEDIA_CONTEXT_MAX_CHARS, IDM_MEDIA_CONTEXT_MIN_WORDS, IDM_MEDIA_POINT_LABEL_MAX_WORDS
from app.idm.text import content_pairs, content_words, idm_fold, single_line

MEDIA_BRIEF_THIN_CODE: Final = "IDM_W4_MEDIA_BRIEF_THIN"
MEDIA_BRIEF_COMPLETED_CODE: Final = "IDM_W4_MEDIA_BRIEF_COMPLETED"

# Accent-folded words that describe where things go (the layout) and what is written on the image.
_LAYOUT_RE: Final = re.compile(
    r"\b(?:bo cuc|khung|cot|hang ngang|hang doc|panel|so do|ben trai|ben phai|o giua|phia tren|phia duoi|trung tam|"
    r"chia|luoi|canh|man hinh|layout|column|row|grid|left|right|centre|center|top|bottom|scene|screen|split|box)\b")
_TEXT_RE: Final = re.compile(
    r"\b(?:chu|nhan|tieu de|ghi|so lieu|con so|dong chu|label|labels|title|text|caption|figure|number|heading|"
    r"reads)\b")
# "Khung 2", "ý 2", "Panel 2", "Scene 2" …: an element named by the number of its content point.
_POSITION_WORDS: Final = r"(?:khung|o|panel|canh|y|buoc|muc|phan|point|scene|step|box|item|part)"
_MIN_SHARED_WORDS: Final = 2
_MIN_KEPT_CHARS: Final = 80


def _named(point: str, folded: str, words: set[str], pairs: set[tuple[str, str]], number: int) -> bool:
    """The description names the content point: by its number, a content phrase of it, or two of its words."""

    if re.search(rf"\b{_POSITION_WORDS}\s*{number}\b", folded):
        return True
    own = content_words(point)
    return bool(content_pairs(point) & pairs) or (
        bool(own) and len(set(own) & words) >= min(_MIN_SHARED_WORDS, len(set(own))))


def media_brief_gaps(brief: ArchitectureMediaBriefV2) -> list[str]:
    """What the description of ``brief`` lacks for a designer, as short server-built reasons (empty when enough)."""

    text = brief.context_description
    folded = idm_fold(text)
    words, pairs = set(content_words(text)), content_pairs(text)
    gaps: list[str] = []
    if len(text.split()) < IDM_MEDIA_CONTEXT_MIN_WORDS:
        gaps.append(f"{len(text.split())} words (at least {IDM_MEDIA_CONTEXT_MIN_WORDS})")
    if not _LAYOUT_RE.search(folded):
        gaps.append("no layout")
    if not _TEXT_RE.search(folded) and "“" not in text and '"' not in text:
        gaps.append("no text on the image")
    missing = [str(number) for number, point in enumerate(brief.content_points, start=1)
               if not _named(point, folded, words, pairs, number)]
    if missing:
        gaps.append("content points " + ", ".join(missing) + " not described")
    return gaps


def _layout(points: Sequence[str], video: bool, vi: bool, label_words: int) -> str:
    labels = [" ".join(point.split()[:label_words]) + (" …" if len(point.split()) > label_words else "")
              for point in points]
    if video:
        head = (f"Dàn cảnh cho người dựng: {len(points)} cảnh theo thứ tự, mỗi cảnh một ý; chữ trên màn hình giữ đúng "
                "câu chữ và số liệu của ý." if vi
                else f"Scenes for the editor: {len(points)} scenes in order, one point each; the text on screen keeps "
                     "the point's own words and numbers.")
        items = [f"{'Cảnh' if vi else 'Scene'} {number}: “{label}”" for number, label in enumerate(labels, start=1)]
        return f"{head} {'; '.join(items)}."
    head = (f"Bố cục cho người thiết kế: {len(points)} khung đọc từ trái sang phải, trên xuống dưới, mỗi khung một ý; "
            "chữ trên hình giữ đúng câu chữ và số liệu của ý." if vi
            else f"Layout for the designer: {len(points)} panels read left to right, top to bottom, one point each; "
                 "the text on the image keeps the point's own words and numbers.")
    items = [f"{'Khung' if vi else 'Panel'} {number}: “{label}”" for number, label in enumerate(labels, start=1)]
    arrows = ("Chỉ nối các khung bằng mũi tên, từ ý trước sang ý sau, khi các ý là các bước nối tiếp hoặc nguyên nhân "
              "và kết quả." if vi
              else "Connect panels with arrows, from one point to the next, only where the points are steps or a cause "
                   "and its effect.")
    return f"{head} {'; '.join(items)}. {arrows}"


def complete_media_brief(brief: ArchitectureMediaBriefV2 | None, locale: str) -> ArchitectureMediaBriefV2 | None:
    """``brief`` with a layout from its content points appended when its description is thin for a designer (the
    description the writer gave stays first; both fit the contract's 1000 characters)."""

    if brief is None or not media_brief_gaps(brief):
        return brief
    vi, video = locale != "en", brief.type == "video"
    original = " ".join(brief.context_description.split())
    for label_words in range(IDM_MEDIA_POINT_LABEL_MAX_WORDS, 2, -2):
        layout = _layout(brief.content_points, video, vi, label_words)
        if len(original) + 1 + len(layout) <= IDM_MEDIA_CONTEXT_MAX_CHARS:
            break
    room = IDM_MEDIA_CONTEXT_MAX_CHARS - len(layout) - 1
    # The writer's words stay first while they leave the layout room enough to say something.
    text = f"{single_line(original, room)} {layout}" if room >= _MIN_KEPT_CHARS else layout
    return brief.model_copy(update={"context_description": single_line(text, IDM_MEDIA_CONTEXT_MAX_CHARS)})


__all__ = ["MEDIA_BRIEF_COMPLETED_CODE", "MEDIA_BRIEF_THIN_CODE", "complete_media_brief", "media_brief_gaps"]
