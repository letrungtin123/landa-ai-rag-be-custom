"""Tag removal that keeps the result of ``re.sub(r"<[^>]+>", replacement, text)`` in linear time.

That regex is quadratic on text where "<" has no ">" after it: from every such "<", ``[^>]+`` runs to the end of
the text before the match fails, so a document of "<" characters takes minutes, and the C regex engine holds the
GIL while it runs, so every request of every tenant waits (security audit 2026-10-24, H1). A different pattern such
as ``<[^<>]*>`` is not a drop-in replacement: it reads "a << b > c" differently, and the Node mirrors in
``app.idm.node_acceptance_text`` must keep the Node result character for character. The same holds for
``<[^>]*>`` (which also removes "<>") and for the element pattern ``<style[^>]*>[\\s\\S]*?</style>`` (review of
wave 1, L-5): ``replace_tags(..., include_empty=True)`` and ``remove_elements`` give their results in linear time.
"""

from __future__ import annotations

import re
import string
from typing import Final

_TAG_RE: Final = re.compile(r"<[^>]+>")
_TAG_OR_EMPTY_RE: Final = re.compile(r"<[^>]*>")
# Only A-Z: the element patterns match case-insensitively under re.ASCII, so no other letter (e.g. U+017F LONG S or
# U+0130) stands for an ASCII one. Translating keeps every index, unlike str.lower().
_ASCII_LOWER: Final = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)


def replace_tags(text: str, replacement: str, *, include_empty: bool = False) -> str:
    """``re.sub(r"<[^>]+>", replacement, text)`` for every string, in time linear in ``len(text)``.

    ``include_empty``: ``re.sub(r"<[^>]*>", ...)`` instead, which also replaces "<>".

    A match always ends with a ">", so nothing after the last ">" can be part of one: the same regex runs on the
    text up to that ">" only. There every "<" has a ">" after it, so ``[^>]+`` (or ``[^>]*``) stops at the first
    one and either matches or (for "<>" without ``include_empty``) fails on its first character; no attempt scans
    the text twice. ``replacement`` is literal text (a backslash in it is not a group reference).
    """

    end = text.rfind(">") + 1
    if not end:
        return text
    pattern = _TAG_OR_EMPTY_RE if include_empty else _TAG_RE
    return pattern.sub(replacement.replace("\\", "\\\\"), text[:end]) + text[end:]


def remove_elements(text: str, name: str) -> str:
    """``re.sub(rf"<{name}[^>]*>[\\s\\S]*?</{name}>", "", text, flags=re.IGNORECASE | re.ASCII)`` in linear time.

    ``name`` is a lower-case ASCII element name (style, script). The regex is quadratic on unclosed elements: from
    every "<style" it scans to the end of the text for the closing tag. Its matches are found here directly: at
    each "<name", the opening tag ends at the first ">" after it and the element at the first "</name>" after that.
    When either is missing, no later "<name" can match either (its ">" and closing tag would have to come after
    these), so the rest of the text is kept as it is.
    """

    folded = text.translate(_ASCII_LOWER)
    opening, closing = f"<{name}", f"</{name}>"
    kept: list[str] = []
    position = 0
    while (start := folded.find(opening, position)) >= 0:
        tag_end = folded.find(">", start + len(opening))
        close = folded.find(closing, tag_end + 1) if tag_end >= 0 else -1
        if close < 0:
            break
        kept.append(text[position:start])
        position = close + len(closing)
    kept.append(text[position:])
    return "".join(kept)
