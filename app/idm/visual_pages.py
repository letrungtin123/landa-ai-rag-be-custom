"""The source pages the system could not read: image pages, never text pages with a decorative picture (run 5496155c).

The evidence bundle asks for a review of every embedded image or vector drawing nobody has observed. In the slide deck
of run 5496155c that was 34 of 48 pages: a photo beside the text of a slide (12-37 % of the page) marked 14 of 21 units
"tài liệu nguồn … có hình ảnh hoặc bảng hệ thống chưa đọc chắc chắn", and only about 5 of them rested on a page that
is a picture. The unit note now names a page only when its unobserved images cover at least
``IDM_IMAGE_PAGE_MIN_SHARE`` of it (the run's picture pages: 41-99 %); a document ingested before the visual regions
were recorded (``legacy_review_required``) keeps the notice, since nothing tells its pages apart.

Reads only the fact locators the backend already sends (``visual_regions`` of ``app.content.source_readiness``).
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any, Final

from app.idm.policy import IDM_IMAGE_PAGE_MIN_SHARE

_LEGACY_STATUS: Final = "legacy_review_required"
_OBSERVED: Final = frozenset({"observed", "reviewed"})
_BOX_VALUES: Final = 4  # x0, y0, x1, y1


def _share(region: Mapping[str, Any]) -> float:
    """The page share of a region: its ``page_area_ratio``, else the area of its normalised box (vector drawings)."""

    ratio = region.get("page_area_ratio")
    if isinstance(ratio, (int, float)) and not isinstance(ratio, bool):
        return max(0.0, min(1.0, float(ratio)))
    locator = region.get("locator")
    box = locator.get("bbox_normalized") if isinstance(locator, Mapping) else None
    if (isinstance(box, (list, tuple)) and len(box) == _BOX_VALUES
            and all(isinstance(value, (int, float)) for value in box)):
        return max(0.0, min(1.0, (float(box[2]) - float(box[0])) * (float(box[3]) - float(box[1]))))
    return 0.0


def _unread(region: Mapping[str, Any]) -> bool:
    observation = region.get("observation")
    status = str(observation.get("status") or "").casefold() if isinstance(observation, Mapping) else ""
    inference = region.get("inference")
    claims = inference.get("claims") if isinstance(inference, Mapping) else None
    return status not in _OBSERVED or bool(claims)


def unread_image_pages(locators: Iterable[Mapping[str, Any]]) -> list[int]:
    """The pages whose unobserved images cover at least ``IDM_IMAGE_PAGE_MIN_SHARE`` of them (each region once,
    however many facts carry it)."""

    regions: dict[str, Mapping[str, Any]] = {}
    for locator in locators:
        for region in locator.get("visual_regions") or ():
            if isinstance(region, Mapping) and _unread(region):
                key = json.dumps([region.get("asset_revision"), region.get("locator")], sort_keys=True, default=str)
                regions.setdefault(key, region)
    shares: defaultdict[int, float] = defaultdict(float)
    for region in regions.values():
        place = region.get("locator")
        page = place.get("page") if isinstance(place, Mapping) else None
        if isinstance(page, int) and not isinstance(page, bool):
            shares[page] += _share(region)
    return sorted(page for page, share in shares.items() if share >= IDM_IMAGE_PAGE_MIN_SHARE)


def evidence_needs_check(locators: Iterable[Mapping[str, Any]]) -> bool:
    """Whether the unit note asks the author to compare the source pages with the original document: a page that is
    mostly an unread image, or a legacy document whose pages cannot be told apart."""

    listed = [locator for locator in locators if isinstance(locator, Mapping)]
    if any(str(locator.get("source_evidence_status") or "").casefold() == _LEGACY_STATUS for locator in listed):
        return True
    return bool(unread_image_pages(listed))


__all__ = ["evidence_needs_check", "unread_image_pages"]
