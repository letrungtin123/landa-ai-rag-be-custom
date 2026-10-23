"""Text normalization shared by ingestion, retrieval and lesson authoring."""

from __future__ import annotations

import html
import re

from app.core.markup import replace_tags


def clean_text(value: str) -> str:
    value = html.unescape(value)
    value = replace_tags(value, " ")
    value = value.replace("\x00", " ")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()
