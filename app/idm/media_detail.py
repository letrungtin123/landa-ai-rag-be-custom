"""The designer brief of a suggested infographic or video (QC feedback 2026-10-21, A4).

QC: the "Mô tả bối cảnh" (``media_brief.context_description``) of a suggestion must let a designer draw it: the
layout, the elements, the text and numbers shown on the image, the arrows and their direction, each tied to a content
point, with a minimum length. The W3/W4 prompt asks for that brief (``prompts_module._MEDIA_BRIEF_RULE``);
``media_brief_gaps`` measures what a description lacks (a lesson warning, listed in a repair the module answer gets
anyway); ``complete_media_brief`` appends a layout built from the content points' own words when the description
is really thin (owner decision 2026-10-22: never a numbering that contradicts the writer's, never a cut quote).
The completion never adds a fact, a number or a relation: arrows are drawn only where the points are steps or a cause
and its effect, as the points themselves say.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final

from app.contracts.orchestration_v2.blueprint import ArchitectureMediaBriefV2
from app.idm.policy import (
    IDM_MEDIA_CONTEXT_DETAILED_WORDS,
    IDM_MEDIA_CONTEXT_MAX_CHARS,
    IDM_MEDIA_CONTEXT_MIN_WORDS,
    IDM_MEDIA_POINT_LABEL_MAX_WORDS,
)
from app.idm.text import idm_fold

MEDIA_BRIEF_THIN_CODE: Final = "IDM_W4_MEDIA_BRIEF_THIN"
MEDIA_BRIEF_COMPLETED_CODE: Final = "IDM_W4_MEDIA_BRIEF_COMPLETED"

# Accent-folded words that describe where things go (the layout), what is drawn (the elements) and what is written on
# the image.
_LAYOUT_RE: Final = re.compile(
    r"\b(?:bo cuc|khung|cot|hang ngang|hang doc|panel|so do|ben trai|ben phai|o giua|phia tren|phia duoi|trung tam|"
    r"chia|luoi|canh|man hinh|nua tren|nua duoi|layout|column|row|grid|left|right|centre|center|top|bottom|scene|"
    r"screen|split|box|half)\b")
_ELEMENT_RE: Final = re.compile(
    r"\b(?:bieu tuong|icon|hinh|khoi|hop|mui ten|duong|bang|the|nut|dai mau|vong tron|truc|do thi|bieu do|"
    r"arrow|shape|chart|graph|table|card|band|circle|axis)\b")
_TEXT_RE: Final = re.compile(
    r"\b(?:chu|nhan|tieu de|ghi|so lieu|con so|dong chu|van ban|tu khoa|cau hoi|label|labels|title|text|caption|"
    r"figure|number|heading|reads|keyword|question)\b")
# Frames, scenes, rows … the writer already numbered ("Khung 1", "Cảnh 2", "Hàng 3", "Tầng 4"): a completion numbering
# of its own would contradict them.
_NUMBERED_RE: Final = re.compile(
    r"\b(?:khung|o|panel|canh|hang|cot|tang|buoc|phan|man hinh|scene|row|column|step|frame|box|layer)\s*\d+\b")


def media_brief_gaps(brief: ArchitectureMediaBriefV2) -> list[str]:
    """Why the description of ``brief`` is too thin for a designer, as short server-built reasons; empty when it is
    enough, and then it is never completed (owner decision 2026-10-22, run dfe78482: a detailed 100-word brief got a
    second, conflicting "5 khung" numbering). A description of ``IDM_MEDIA_CONTEXT_DETAILED_WORDS`` words or more, or
    one that numbers its frames or scenes, is enough; a shorter one is thin when it has fewer than
    ``IDM_MEDIA_CONTEXT_MIN_WORDS`` words, says nothing of a layout, or names nothing to draw or write on the image."""

    text = brief.context_description
    folded = idm_fold(text)
    words = len(text.split())
    if words >= IDM_MEDIA_CONTEXT_DETAILED_WORDS or _NUMBERED_RE.search(folded):
        return []
    gaps: list[str] = []
    if words < IDM_MEDIA_CONTEXT_MIN_WORDS:
        gaps.append(f"{words} words (at least {IDM_MEDIA_CONTEXT_MIN_WORDS})")
    if not _LAYOUT_RE.search(folded):
        gaps.append("no layout")
    if not (_ELEMENT_RE.search(folded) or _TEXT_RE.search(folded) or "“" in text or '"' in text):
        gaps.append("nothing drawn or written on the image")
    return gaps


def _label(point: str, number: int, vi: bool, quote: bool) -> str:
    """A content point quoted whole when it is short, else named by its number: never a quote cut mid-word."""

    words = point.split()
    if quote and len(words) <= IDM_MEDIA_POINT_LABEL_MAX_WORDS:
        return f"“{' '.join(words)}”"
    return f"ý {number} của nội dung" if vi else f"content point {number}"


def _layout(points: Sequence[str], video: bool, vi: bool, quote: bool) -> str:
    labels = [_label(point, number, vi, quote) for number, point in enumerate(points, start=1)]
    if video:
        head = (f"Dàn cảnh cho người dựng: {len(points)} cảnh theo thứ tự, mỗi cảnh một ý; chữ trên màn hình giữ đúng "
                "câu chữ và số liệu của ý." if vi
                else f"Scenes for the editor: {len(points)} scenes in order, one point each; the text on screen keeps "
                     "the point's own words and numbers.")
        items = [f"{'Cảnh' if vi else 'Scene'} {number}: {label}" for number, label in enumerate(labels, start=1)]
        return f"{head} {'; '.join(items)}."
    head = (f"Bố cục cho người thiết kế: {len(points)} khung đọc từ trái sang phải, trên xuống dưới, mỗi khung một ý; "
            "chữ trên hình giữ đúng câu chữ và số liệu của ý." if vi
            else f"Layout for the designer: {len(points)} panels read left to right, top to bottom, one point each; "
                 "the text on the image keeps the point's own words and numbers.")
    items = [f"{'Khung' if vi else 'Panel'} {number}: {label}" for number, label in enumerate(labels, start=1)]
    arrows = ("Chỉ nối các khung bằng mũi tên, từ ý trước sang ý sau, khi các ý là các bước nối tiếp hoặc nguyên nhân "
              "và kết quả." if vi
              else "Connect panels with arrows, from one point to the next, only where the points are steps or a cause "
                   "and its effect.")
    return f"{head} {'; '.join(items)}. {arrows}"


def complete_media_brief(brief: ArchitectureMediaBriefV2 | None, locale: str) -> ArchitectureMediaBriefV2 | None:
    """``brief`` with a layout from its content points appended when its description is thin for a designer
    (``media_brief_gaps``). The writer's words stay whole and first; the short points are quoted whole, the others named
    by their number; when the quotes do not fit the contract's 1000 characters no point is quoted, and when even that
    does not fit the brief stays as written."""

    if brief is None or not media_brief_gaps(brief):
        return brief
    vi, video = locale != "en", brief.type == "video"
    original = " ".join(brief.context_description.split())
    for quote in (True, False):
        layout = _layout(brief.content_points, video, vi, quote)
        if len(original) + 1 + len(layout) <= IDM_MEDIA_CONTEXT_MAX_CHARS:
            return brief.model_copy(update={"context_description": f"{original} {layout}"})
    return brief


__all__ = ["MEDIA_BRIEF_COMPLETED_CODE", "MEDIA_BRIEF_THIN_CODE", "complete_media_brief", "media_brief_gaps"]
