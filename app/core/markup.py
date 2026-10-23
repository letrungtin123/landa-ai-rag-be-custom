"""Tag removal that keeps the result of ``re.sub(r"<[^>]+>", replacement, text)`` in linear time.

That regex is quadratic on text where "<" has no ">" after it: from every such "<", ``[^>]+`` runs to the end of
the text before the match fails, so a document of "<" characters takes minutes, and the C regex engine holds the
GIL while it runs, so every request of every tenant waits (security audit 2026-10-24, H1). A different pattern such
as ``<[^<>]*>`` is not a drop-in replacement: it reads "a << b > c" differently, and the Node mirrors in
``app.idm.node_acceptance_text`` must keep the Node result character for character.
"""

from __future__ import annotations

import re
from typing import Final

_TAG_RE: Final = re.compile(r"<[^>]+>")


def replace_tags(text: str, replacement: str) -> str:
    """``re.sub(r"<[^>]+>", replacement, text)`` for every string, in time linear in ``len(text)``.

    A match always ends with a ">", so nothing after the last ">" can be part of one: the same regex runs on the
    text up to that ">" only. There every "<" has a ">" after it, so ``[^>]+`` stops at the first one and either
    matches or (for "<>") fails on its first character; no attempt scans the text twice. ``replacement`` is literal
    text (a backslash in it is not a group reference).
    """

    end = text.rfind(">") + 1
    if not end:
        return text
    return _TAG_RE.sub(replacement.replace("\\", "\\\\"), text[:end]) + text[end:]
