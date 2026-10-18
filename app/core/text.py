"""Accent folding for matching text without case and diacritics (STD-7, F-08).

Each variant is a separate, named function because they differ on purpose, mostly on "đ": NFD does
not decompose "đ" (U+0111), so a fold that only drops combining marks keeps it, and "đường" folds to
"đuong" there but to "duong" when "đ" is mapped to "d" first. Swapping one variant for another changes
which texts match: that is a behaviour change, never a cleanup.

Other folds keep their own code because their semantics differ from both variants here (for example
``app.idm.text.idm_fold``: NFC, ``lower()``, "đ" to "d", whitespace collapsed; or the Node mirrors in
``app.idm.node_acceptance``). ``tests/test_architecture_layers.py`` freezes where decomposing
normalisation may still appear outside this module, so a new private copy fails the gate.
"""

from __future__ import annotations

import unicodedata


def casefold_strip_marks(value: str) -> str:
    """Case-fold, decompose (NFD) and drop the combining marks (category Mn). "đ" and "Đ" give "đ"."""

    normalized = unicodedata.normalize("NFD", value.casefold())
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def casefold_strip_marks_and_d(value: str) -> str:
    """Case-fold, map "đ" to "d", decompose (NFD) and drop the combining marks. "đ" and "Đ" give "d"."""

    normalized = unicodedata.normalize("NFD", value.casefold().replace("đ", "d"))
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")
