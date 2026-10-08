"""Column-aware reading order for PDF text lines (QC D9 / plan QLT-1 R2).

PyMuPDF returns text lines with bounding boxes. Sorting top-to-bottom then left-to-right
interleaves side-by-side columns, and PyMuPDF itself merges lines that share a baseline across
columns into one block: a right-column heading a few points higher than the left one is read
first ("After" before "Before") and canvas labels drift away from their questions.

``column_reading_order`` looks for a vertical gutter: a cut with lines entirely on both sides,
real whitespace between them, both sides wide enough to be columns, and content that stands
side by side. Lines (or tables) that cross the cut are full-width separators and stay in place;
between two separators the left column is read before the right one. Each side is searched
again for a further gutter (three-column pages). The function returns ``None`` whenever no such
gutter exists, so single-column pages keep the historical order exactly.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from itertools import pairwise

Box = tuple[float, float, float, float]  # x0, y0, x1, y1

# Whitespace between the two sides of a cut, in points.
MIN_GUTTER_POINTS = 6.0
# Each side of a cut must span at least this share of the page's text width (labels, page
# numbers or a right-aligned date are not columns).
MIN_COLUMN_WIDTH_FRACTION = 0.2
# Candidate cuts validated per level (best scores first).
MAX_CANDIDATES = 8
# Gutter search depth: 2 levels read up to three or four columns.
MAX_DEPTH = 2
MIN_SIDE_LINES = 1


def _sorted(indices: Sequence[int], boxes: Sequence[Box]) -> list[int]:
    return sorted(indices, key=lambda index: (boxes[index][1], boxes[index][0]))


def baseline_order(boxes: Sequence[Box]) -> list[int]:
    """Historical order: top to bottom, then left to right."""
    return _sorted(range(len(boxes)), boxes)


def _vertical_extent(indices: Sequence[int], boxes: Sequence[Box]) -> tuple[float, float]:
    return min(boxes[index][1] for index in indices), max(boxes[index][3] for index in indices)


def _side_by_side(left: Sequence[int], right: Sequence[int], boxes: Sequence[Box]) -> bool:
    if not left or not right:
        return False
    left_top, left_bottom = _vertical_extent(left, boxes)
    right_top, right_bottom = _vertical_extent(right, boxes)
    return max(left_top, right_top) < min(left_bottom, right_bottom)


def _horizontal_span(indices: Sequence[int], boxes: Sequence[Box]) -> float:
    return max(boxes[index][2] for index in indices) - min(boxes[index][0] for index in indices)


def _candidate_cuts(indices: Sequence[int], boxes: Sequence[Box]) -> list[float]:
    """Cut positions ranked by balance (lines on the weaker side) minus crossing lines."""
    starts = sorted(boxes[index][0] for index in indices)
    ends = sorted(boxes[index][2] for index in indices)
    edges = sorted({*starts, *ends})
    scored: list[tuple[int, float, float]] = []
    for lower, upper in pairwise(edges):
        cut = (lower + upper) / 2
        left_count = bisect_right(ends, cut)
        right_position = bisect_left(starts, cut)
        right_count = len(starts) - right_position
        if left_count < MIN_SIDE_LINES or right_count < MIN_SIDE_LINES:
            continue
        gap = starts[right_position] - ends[left_count - 1]
        if gap < MIN_GUTTER_POINTS:
            continue
        crossing = len(indices) - left_count - right_count
        scored.append((min(left_count, right_count) - crossing, gap, cut))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [cut for _, _, cut in scored[:MAX_CANDIDATES]]


def _split(
    indices: Sequence[int],
    boxes: Sequence[Box],
    width: float,
) -> tuple[list[int], list[int], list[int]] | None:
    for cut in _candidate_cuts(indices, boxes):
        left = [index for index in indices if boxes[index][2] <= cut]
        right = [index for index in indices if boxes[index][0] >= cut]
        if (
            _horizontal_span(left, boxes) >= MIN_COLUMN_WIDTH_FRACTION * width
            and _horizontal_span(right, boxes) >= MIN_COLUMN_WIDTH_FRACTION * width
            and _side_by_side(left, right, boxes)
        ):
            sides = set(left) | set(right)
            return left, right, [index for index in indices if index not in sides]
    return None


def _read_side(indices: list[int], boxes: Sequence[Box], width: float, depth: int) -> list[int]:
    nested = _order(indices, boxes, width, depth + 1) if depth + 1 < MAX_DEPTH and indices else None
    return nested if nested is not None else _sorted(indices, boxes)


def _order(indices: list[int], boxes: Sequence[Box], width: float, depth: int) -> list[int] | None:
    split = _split(indices, boxes, width)
    if split is None:
        return None
    left, right, separators = split
    remaining_left = _sorted(left, boxes)
    remaining_right = _sorted(right, boxes)
    ordered: list[int] = []

    def emit_band(band_left: list[int], band_right: list[int]) -> None:
        if _side_by_side(band_left, band_right, boxes):
            ordered.extend(_read_side(band_left, boxes, width, depth))
            ordered.extend(_read_side(band_right, boxes, width, depth))
        else:
            # Stacked, not parallel (e.g. a right-aligned line above a left-aligned paragraph).
            ordered.extend(_sorted([*band_left, *band_right], boxes))

    for separator in _sorted(separators, boxes):
        top = boxes[separator][1]
        band_left = [index for index in remaining_left if (boxes[index][1] + boxes[index][3]) / 2 < top]
        band_right = [index for index in remaining_right if (boxes[index][1] + boxes[index][3]) / 2 < top]
        emit_band(band_left, band_right)
        emitted = {*band_left, *band_right}
        remaining_left = [index for index in remaining_left if index not in emitted]
        remaining_right = [index for index in remaining_right if index not in emitted]
        ordered.append(separator)
    emit_band(remaining_left, remaining_right)
    return ordered


def column_reading_order(boxes: Sequence[Box]) -> list[int] | None:
    """Return a column-aware permutation of ``boxes`` or ``None`` when the baseline order applies."""
    if len(boxes) < 2:  # noqa: PLR2004 - two boxes are the minimum for two columns
        return None
    width = max(box[2] for box in boxes) - min(box[0] for box in boxes)
    if width <= 0:
        return None
    ordered = _order(list(range(len(boxes))), boxes, width, 0)
    if ordered is None or ordered == baseline_order(boxes):
        return None
    return ordered
