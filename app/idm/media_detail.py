"""The designer brief of a suggested infographic or video (QC feedback 2026-10-21, A4).

QC: the "Mô tả bối cảnh" (``media_brief.context_description``) of a suggestion must let a designer draw it: the
layout, the elements, the text and numbers shown on the image, the arrows and their direction, each tied to a content
point, with a minimum length. The W3/W4 prompt asks for that brief (``prompts_module._MEDIA_BRIEF_RULE``);
``media_brief_gaps`` measures what a description lacks (a lesson warning, listed in a repair the module answer gets
anyway); ``complete_media_brief`` appends a layout built from the content points' own words when the description
is really thin (owner decision 2026-10-22: never a numbering that contradicts the writer's, never a cut quote).
Review 2 (2026-10-22): the words are matched with their accents ("tăng 20%" is no "tầng", "phòng họp" no "hộp", "the"
no "thẻ"), a layout is read from explicit phrases ("Khung 1", "3 khung", "hai cột", "cột thứ nhất", "bên trái",
"chia 3 phần", "Video gồm 3 cảnh"), and a description that describes any layout is never completed.
The completion never adds a fact, a number or a relation: arrows are drawn only where the points are steps or a cause
and its effect, as the points themselves say.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from typing import Final

from app.contracts.orchestration_v2.blueprint import ArchitectureMediaBriefV2
from app.idm.policy import (
    IDM_MEDIA_CONTEXT_DETAILED_WORDS,
    IDM_MEDIA_CONTEXT_MAX_CHARS,
    IDM_MEDIA_CONTEXT_MIN_WORDS,
    IDM_MEDIA_POINT_LABEL_MAX_WORDS,
)
from app.idm.quiz_text import utf16_length

MEDIA_BRIEF_THIN_CODE: Final = "IDM_W4_MEDIA_BRIEF_THIN"
MEDIA_BRIEF_COMPLETED_CODE: Final = "IDM_W4_MEDIA_BRIEF_COMPLETED"

# Layout nouns: a frame, a column, a scene … A number after one at the start of a phrase is the writer's numbering
# ("Khung 1", "Cột 2", "Cảnh 3", "Hàng 1", "Tầng 2", "Panel 1", "Scene 2"); a count before one is a layout ("3 khung",
# "hai cột", "Video gồm 3 cảnh"). "hàng", "tầng", "ô", "phần", "bước", "cảnh" … are also ordinary words ("kho hàng 2
# tầng", "bối cảnh"): numbered, they count only at the start of a phrase.
_CLEAR_NOUNS: Final = r"khung|cột|panel|slide|scene|frame|column"
_PLAIN_NOUNS: Final = r"cảnh|hàng|tầng|ô|phần|bước|màn hình|row|box|layer|step|part|section"
_NUMBERED_RE: Final = re.compile(
    rf"(?<!\w)(?:(?P<clear>{_CLEAR_NOUNS})|(?P<plain>{_PLAIN_NOUNS}))[ \t]+(?:số[ \t]+)?\d{{1,2}}(?![\d%]|[.,]\d)"
    rf"(?![ \t]*%)(?![ \t]+(?:{_CLEAR_NOUNS}|{_PLAIN_NOUNS}|người|ngày|tháng|năm|giờ|phút|lần|triệu|nghìn|tỷ|đồng)"
    r"(?!\w))", re.IGNORECASE)
# What may come before a numbered plain noun: the start, a mark, or a word that places it ("ở hàng 1", "từ cảnh 2").
_PHRASE_MARKS: Final = ".:;,()–—-•*\"“”'"  # noqa: RUF001 - dashes and quotes
_PLACING_WORDS: Final = frozenset({"ở", "tại", "trong", "trên", "dưới", "và", "với", "từ", "sang", "tới", "đến",
                                   "giữa", "mỗi", "là", "in", "on", "at", "and", "to", "from", "of", "the", "each"})
_COUNT: Final = r"\d{1,2}|hai|ba|bốn|năm|sáu|bảy|tám|two|three|four|five|six|seven|eight"
_LAYOUT_RES: Final = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    # A count of layout parts: "3 khung", "hai cột", "ba cảnh", "4 ô", "two columns", "Video gồm 3 cảnh".
    rf"(?<!\w)(?:{_COUNT})[ \t]+(?:khung|cột|cảnh|hàng|ô|ngăn|vùng|mảng|màn hình|panels?|columns?|rows?|scenes?|"
    r"frames?|slides?|boxes|sections?)(?!\w)",
    # Dividing the image: "chia 3 phần", "chia làm hai", "chia đôi", "split into three".
    rf"(?<!\w)(?:chia(?:[ \t]+(?:làm|thành))?[ \t]+(?:{_COUNT}|đôi)|split[ \t]+(?:into|in)[ \t]+(?:{_COUNT}|half|two))"
    r"(?!\w)",
    # An ordinal part: "cột thứ nhất", "khung thứ 2", "cảnh đầu", "cảnh cuối", "first column", "last scene".
    r"(?<!\w)(?:(?:khung|cột|cảnh|hàng|ô|phần|panel)[ \t]+(?:thứ[ \t]+\w+|đầu(?:[ \t]+tiên)?|cuối(?:[ \t]+cùng)?|"
    r"giữa|kế tiếp|tiếp theo|mở đầu|kết)|(?:first|second|third|last|middle|next)[ \t]+(?:column|panel|scene|frame|"
    r"row|slide))(?!\w)",
    # Places on the image: "bên trái", "ở giữa", "bên phải", "phía trên", "nửa dưới", "trên cùng", "từ trái sang phải".
    r"(?<!\w)(?:bên[ \t]+(?:trái|phải|trên|dưới)|ở[ \t]+giữa|chính[ \t]+giữa|phía[ \t]+(?:trên|dưới|trái|phải)|"
    r"nửa[ \t]+(?:trên|dưới|trái|phải)|trên[ \t]+cùng|dưới[ \t]+cùng|góc[ \t]+(?:trên|dưới|trái|phải)|"
    r"trái[ \t]+sang[ \t]+phải|trên[ \t]+xuống[ \t]+dưới|on[ \t]+the[ \t]+(?:left|right)|at[ \t]+the[ \t]+"
    r"(?:top|bottom)|in[ \t]+the[ \t]+(?:centre|center|middle)|left[ \t]+to[ \t]+right|top[ \t]+to[ \t]+bottom|"
    r"side[ \t]+by[ \t]+side)(?!\w)",
    # Layout words: "bố cục", "lưới", "sơ đồ", "hàng ngang", "mỗi khung", "chuyển cảnh", "layout", "grid".
    r"(?<!\w)(?:bố[ \t]+cục|lưới|sơ[ \t]+đồ|hàng[ \t]+(?:ngang|dọc)|cạnh[ \t]+nhau|(?:mỗi|các)[ \t]+(?:khung|cột|"
    r"cảnh|ô)|chuyển[ \t]+cảnh|layout|grid|split[ \t]+screen)(?!\w)",
))
# What is drawn or written on the image, as whole words with their accents (review 2: "the", "hop", "bang", "hinh",
# "truc" … match ordinary words once the accents are gone).
_DRAWN_RE: Final = re.compile(
    r"(?<!\w)(?:biểu[ \t]+tượng|icon|hình[ \t]+(?:minh[ \t]+(?:hoạ|họa)|vẽ|ảnh|tròn|người)|ảnh[ \t]+(?:chụp|minh"
    r"[ \t]+(?:hoạ|họa))|mũi[ \t]+tên|đồ[ \t]+thị|biểu[ \t]+đồ|bảng|thẻ|hộp|nút[ \t]+bấm|dải[ \t]+màu|vòng[ \t]+tròn|"
    r"trục[ \t]+(?:tung|hoành|ngang|dọc|thời[ \t]+gian)|nhãn|chữ|tiêu[ \t]+đề|số[ \t]+liệu|con[ \t]+số|văn[ \t]+bản|"
    r"từ[ \t]+(?:khoá|khóa)|câu[ \t]+hỏi|ghi[ \t]+chú|icons?|arrows?|charts?|graphs?|tables?|labels?|captions?|"
    r"headings?|titles?|text|numbers?|figures?|shapes?|cards?|circles?|axis|axes|photos?|illustrations?|diagrams?|"
    r"keywords?)(?!\w)", re.IGNORECASE)


def _numbered(text: str) -> bool:
    """Whether ``text`` numbers its frames, columns or scenes ("Khung 1", "ở hàng 2"); "kho hàng 2 tầng" does not."""

    for match in _NUMBERED_RE.finditer(text):
        if match.group("clear"):
            return True
        before = text[:match.start()].rstrip(" \t")
        words = before.split()
        if not before or before[-1] in _PHRASE_MARKS or before.endswith("\n") or (
                words and words[-1].casefold() in _PLACING_WORDS):
            return True
    return False


def describes_layout(text: str) -> bool:
    """Whether the writer's description already says where things go: a numbering of its frames or scenes, a count of
    them, a division of the image, an ordinal part or a place on the image (review 2: then it is never completed)."""

    normal = unicodedata.normalize("NFC", text)
    return _numbered(normal) or any(pattern.search(normal) for pattern in _LAYOUT_RES)


def media_brief_gaps(brief: ArchitectureMediaBriefV2) -> list[str]:
    """Why the description of ``brief`` is too thin for a designer, as short server-built reasons; empty when it is
    enough, and then it is never completed (owner decision 2026-10-22, run dfe78482: a detailed 100-word brief got a
    second, conflicting "5 khung" numbering). A description of ``IDM_MEDIA_CONTEXT_DETAILED_WORDS`` words or more, or
    one that describes any layout (``describes_layout``), is enough; a shorter one without a layout is thin, and the
    reasons say too when it has fewer than ``IDM_MEDIA_CONTEXT_MIN_WORDS`` words or names nothing drawn or written on
    the image."""

    text = unicodedata.normalize("NFC", brief.context_description)
    words = len(text.split())
    if words >= IDM_MEDIA_CONTEXT_DETAILED_WORDS or describes_layout(text):
        return []
    gaps: list[str] = []
    if words < IDM_MEDIA_CONTEXT_MIN_WORDS:
        gaps.append(f"{words} words (at least {IDM_MEDIA_CONTEXT_MIN_WORDS})")
    gaps.append("no layout")
    if not (_DRAWN_RE.search(text) or "“" in text or '"' in text):
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
    by their number; when the quotes do not fit the contract's 1000 characters (as Node counts them) no point is quoted,
    and when even that does not fit the brief stays as written: the very same object (``media_brief_completed``)."""

    if brief is None or not media_brief_gaps(brief):
        return brief
    vi, video = locale != "en", brief.type == "video"
    original = " ".join(brief.context_description.split())
    for quote in (True, False):
        layout = _layout(brief.content_points, video, vi, quote)
        if utf16_length(f"{original} {layout}") <= IDM_MEDIA_CONTEXT_MAX_CHARS:
            return brief.model_copy(update={"context_description": f"{original} {layout}"})
    return brief


def media_brief_completed(brief: ArchitectureMediaBriefV2 | None, locale: str) -> bool:
    """Whether ``complete_media_brief`` changes ``brief`` (review 2: only then the lesson gets the thin warning, its
    "the system added a layout" note and the completion count)."""

    return complete_media_brief(brief, locale) is not brief


__all__ = ["MEDIA_BRIEF_COMPLETED_CODE", "MEDIA_BRIEF_THIN_CODE", "complete_media_brief", "describes_layout",
           "media_brief_completed", "media_brief_gaps"]
