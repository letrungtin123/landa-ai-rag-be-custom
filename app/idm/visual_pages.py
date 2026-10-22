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
# The bundle's own review requirements of an unread region; any other one means the pages cannot be told apart.
_VISUAL_REQUIREMENTS: Final = ("VISUAL_OBSERVATION_REQUIRED:", "VISUAL_INFERENCE_REQUIRES_VERIFICATION:")


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


def unread_image_pages(locators: Iterable[Mapping[str, Any]]) -> list[int]:
    """The pages whose unobserved images cover at least ``IDM_IMAGE_PAGE_MIN_SHARE`` of them (each region once,
    however many facts carry it)."""

    shares: defaultdict[int, float] = defaultdict(float)
    for region in _unread_regions(locators):
        if (page := _page(region)) is not None:
            shares[page] += _share(region) or 0.0
    return sorted(page for page, share in shares.items() if share >= IDM_IMAGE_PAGE_MIN_SHARE)


def evidence_needs_check(locators: Iterable[Mapping[str, Any]], requirements: Iterable[Any] = ()) -> bool:
    """Whether the unit note asks the author to compare the source pages with the original document: a page that is
    mostly an unread image. The old warning stays whenever the pages cannot be told apart (review of dfb6db2): a
    legacy document, a bundle review requirement that is no visual one (``EVIDENCE_BUNDLE_ASSEMBLY_FAILED``, a limit
    reached), or an unread region without its page or its size."""

    listed = [locator for locator in locators if isinstance(locator, Mapping)]
    if any(str(locator.get("source_evidence_status") or "").casefold() == _LEGACY_STATUS for locator in listed):
        return True
    if any(not str(code).startswith(_VISUAL_REQUIREMENTS) for code in requirements):
        return True
    if any(_page(region) is None or _share(region) is None for region in _unread_regions(listed)):
        return True
    return bool(unread_image_pages(listed))


__all__ = ["evidence_needs_check", "unread_image_pages"]
