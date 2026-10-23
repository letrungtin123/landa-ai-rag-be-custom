"""The source pages the system could not read: image pages, never text pages with a decorative picture (run 5496155c).

The evidence bundle asks for a review of every embedded image or vector drawing nobody has observed. In the slide deck
of run 5496155c that was 34 of 48 pages: a photo beside the text of a slide (12-37 % of the page) marked 14 of 21 units
"tài liệu nguồn … có hình ảnh hoặc bảng hệ thống chưa đọc chắc chắn", and only about 5 of them rested on a page that
is a picture. The unit note now names a page only when its unobserved images cover at least
``IDM_IMAGE_PAGE_MIN_SHARE`` of it (the run's picture pages: 41-99 %); a document ingested before the visual regions
were recorded (``legacy_review_required``) keeps the notice, since nothing tells its pages apart.

Run 2f8781d6 (A4): the warning fired for two section cover pages (a title over a picture: pages 12, 45) and missed a
page whose picture holds its content beside one sentence (page 29: two pictures of 37 % in all). With the unit's fact
texts of each page (``page_texts``), a section cover never counts, and a page with at most one sentence besides its
titles counts from ``IDM_IMAGE_PAGE_SHORT_TEXT_MIN_SHARE``.

Review 5 (M5): a cover only when it is certain: no sentence, no numbered title, an opening title in capitals that names
a section ("MODULE 1: THẤU HIỆN TRẠNG", "BÀI TẬP VỀ NHÀ", "PHẦN 2 …") and at most one subtitle. A picture page whose
only text is chart labels ("OPEX", "CAPEX", "ROI") or a plain title ("BẢN ĐỒ HÀNH TRÌNH KHÁCH HÀNG") keeps the
warning, by the rule of run 5496155c when its words are only marks.

Review 6 (L5): a heading word is no signal ("CHƯƠNG TRÌNH KHÁCH HÀNG THÂN THIẾT" is a programme, "PHẦN MỀM CRM" a
screenshot). A cover is a page whose whole text is one heading line that opens a numbered section ("MODULE 2: …",
"PHẦN 3 …", "CHƯƠNG II …") or an exercise ("BÀI TẬP …"), and that the source structure starts a section with (the
``scope_title`` of a fact on that page). Anything else keeps the warning (run 2f8781d6's page 12, a title and a
subtitle under the scope "Confidential", warns again).

Reads only the fact locators and texts the backend already sends (``visual_regions`` of
``app.content.source_readiness``).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Any, Final

from app.idm.policy import IDM_IMAGE_PAGE_MIN_SHARE, IDM_IMAGE_PAGE_SHORT_TEXT_MIN_SHARE
from app.idm.text import idm_fold

_LEGACY_STATUS: Final = "legacy_review_required"
_OBSERVED: Final = frozenset({"observed", "reviewed"})
_BOX_VALUES: Final = 4  # x0, y0, x1, y1
# The bundle's own review requirements of an unread region; any other one means the pages cannot be told apart.
_VISUAL_REQUIREMENTS: Final = ("VISUAL_OBSERVATION_REQUIRED:", "VISUAL_INFERENCE_REQUIRES_VERIFICATION:")
# A title line of a page: a few words without a statement's full stop ("MODULE 1: THẤU HIỂU HIỆN TRẠNG", "Góc nhìn
# Outside-In: khách hàng đang trải qua điều gì?"); a line of fewer words is a page mark ("1", "OPEX").
_TITLE_MAX_WORDS: Final = 14
_MARK_MAX_WORDS: Final = 2
_STATEMENT_END_RE: Final = re.compile(r"[.!…]\W*$")
_ROW_MARK_RE: Final = re.compile(r"^\s*row\s+\d+\s*[:;]\s*\|?\s*", re.IGNORECASE)
# Review 6 (L5): the opening of a section cover, in capitals: a numbered section or an exercise.
_COVER_OPENER_RE: Final = re.compile(r"^(?:(?:module|phần|chương|part|chapter|section|session)\s*(?:\d+|[ivx]+)\b"
                                     r"|bài tập\b)", re.IGNORECASE)
_COVER_CAPITALS_SHARE: Final = 0.6
# A numbered section title ("1.5 THIẾT KẾ …", "2.4 MA TRẬN …"): a content page of the deck, never its cover.
_NUMBERED_TITLE_RE: Final = re.compile(r"^\d+(?:\.\d+)+\.?\s+\S")


def _share(region: Mapping[str, Any]) -> float | None:
    """The page share of a region: its ``page_area_ratio``, else the area of its normalised box (vector drawings);
    ``None`` without either."""

    ratio = region.get("page_area_ratio")
    if isinstance(ratio, (int, float)) and not isinstance(ratio, bool):
        return max(0.0, min(1.0, float(ratio)))
    locator = region.get("locator")
    box = locator.get("bbox_normalized") if isinstance(locator, Mapping) else None
    if (isinstance(box, (list, tuple)) and len(box) == _BOX_VALUES
            and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in box)):
        return max(0.0, min(1.0, (float(box[2]) - float(box[0])) * (float(box[3]) - float(box[1]))))
    return None


def _page(region: Mapping[str, Any]) -> int | None:
    place = region.get("locator")
    page = place.get("page") if isinstance(place, Mapping) else None
    return page if isinstance(page, int) and not isinstance(page, bool) else None


def _unread(region: Mapping[str, Any]) -> bool:
    observation = region.get("observation")
    status = str(observation.get("status") or "").casefold() if isinstance(observation, Mapping) else ""
    inference = region.get("inference")
    claims = inference.get("claims") if isinstance(inference, Mapping) else None
    return status not in _OBSERVED or bool(claims)


def _unread_regions(locators: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The unobserved regions of ``locators``, each once however many facts carry it."""

    regions: dict[str, Mapping[str, Any]] = {}
    for locator in locators:
        for region in locator.get("visual_regions") or ():
            if isinstance(region, Mapping) and _unread(region):
                key = json.dumps([region.get("asset_revision"), region.get("locator")], sort_keys=True, default=str)
                regions.setdefault(key, region)
    return list(regions.values())


def page_texts(facts: Iterable[tuple[Any, str]]) -> dict[int, list[str]]:
    """Page -> the fact texts of that page, from ``(source_page, fact_text)`` pairs (none for a fact without a page)."""

    pages: dict[int, list[str]] = {}
    for page, text in facts:
        if isinstance(page, int) and not isinstance(page, bool):
            pages.setdefault(page, []).append(text)
    return pages


def _cover_opener(line: str) -> bool:
    letters = [char for char in line if char.isalpha()]
    return (bool(_COVER_OPENER_RE.match(line)) and bool(letters)
            and sum(char.isupper() for char in letters) / len(letters) >= _COVER_CAPITALS_SHARE)


def _min_share(texts: Sequence[str] | None, starts: Collection[str] = ()) -> float | None:
    """The share of unread images from which a page with ``texts`` counts; ``None`` for a section cover (module
    docstring; ``starts``: the folded section titles the source structure starts on the page); without its texts, or
    with marks only, ``IDM_IMAGE_PAGE_MIN_SHARE``."""

    if texts is None:
        return IDM_IMAGE_PAGE_MIN_SHARE
    lines = [_ROW_MARK_RE.sub("", " ".join(text.split())).strip(" |") for text in texts]
    lines = [line for line in lines if len(line.split()) > _MARK_MAX_WORDS]
    if not lines:
        return IDM_IMAGE_PAGE_MIN_SHARE
    titles = [line for line in lines if len(line.split()) <= _TITLE_MAX_WORDS and not _STATEMENT_END_RE.search(line)]
    body = len(lines) - len(titles)
    if (len(lines) == 1 and titles and _cover_opener(titles[0]) and not _NUMBERED_TITLE_RE.match(titles[0])
            and idm_fold(titles[0]) in starts):
        return None
    return IDM_IMAGE_PAGE_SHORT_TEXT_MIN_SHARE if body <= 1 else IDM_IMAGE_PAGE_MIN_SHARE


def _section_starts(locators: Sequence[Mapping[str, Any]]) -> dict[int, set[str]]:
    """Page -> the folded section titles (``scope_title``) of the facts whose regions sit on it."""

    starts: dict[int, set[str]] = defaultdict(set)
    for locator in locators:
        title = locator.get("scope_title")
        regions = locator.get("visual_regions")
        for region in regions if isinstance(regions, list) and isinstance(title, str) else []:
            if isinstance(region, Mapping) and (page := _page(region)) is not None:
                starts[page].add(idm_fold(_ROW_MARK_RE.sub("", " ".join(str(title).split())).strip(" |")))
    return starts


def unread_image_pages(locators: Iterable[Mapping[str, Any]],
                       texts: Mapping[int, Sequence[str]] | None = None) -> list[int]:
    """The pages whose unobserved images cover at least ``IDM_IMAGE_PAGE_MIN_SHARE`` of them (each region once,
    however many facts carry it). With ``texts`` (page -> the unit's fact texts of it, ``page_texts``): never a cover
    page, and from ``IDM_IMAGE_PAGE_SHORT_TEXT_MIN_SHARE`` for a page of at most one sentence besides its titles."""

    listed = list(locators)
    shares: defaultdict[int, float] = defaultdict(float)
    for region in _unread_regions(listed):
        if (page := _page(region)) is not None:
            shares[page] += _share(region) or 0.0
    starts = _section_starts(listed)
    counted = []
    for page, share in shares.items():
        least = _min_share(texts.get(page) if texts is not None else None, starts.get(page, set()))
        if least is not None and share >= least:
            counted.append(page)
    return sorted(counted)


def evidence_needs_check(locators: Iterable[Mapping[str, Any]], requirements: Iterable[Any] = (),
                         texts: Mapping[int, Sequence[str]] | None = None) -> bool:
    """Whether the unit note asks the author to compare the source pages with the original document: a page that is
    mostly an unread image. The old warning stays whenever the pages cannot be told apart (review of dfb6db2): a
    legacy document, a bundle review requirement that is no visual one (``EVIDENCE_BUNDLE_ASSEMBLY_FAILED``, a limit
    reached), or an unread region without its page or its size. ``texts``: see ``unread_image_pages``."""

    listed = [locator for locator in locators if isinstance(locator, Mapping)]
    if any(str(locator.get("source_evidence_status") or "").casefold() == _LEGACY_STATUS for locator in listed):
        return True
    if any(not str(code).startswith(_VISUAL_REQUIREMENTS) for code in requirements):
        return True
    if any(_page(region) is None or _share(region) is None for region in _unread_regions(listed)):
        return True
    return bool(unread_image_pages(listed, texts))


__all__ = ["evidence_needs_check", "page_texts", "unread_image_pages"]
