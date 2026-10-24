"""Linear-time scans of staged unit HTML and learner text (the regexes they replace scanned to the end from every
starting point; audit 2026-10-24, wave 3)."""

from __future__ import annotations

import re

from app.core.markup import element_spans

# A website or an e-mail address in learner text: (?:https?://|www\.|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}) searched (re.I)
# by contains_contact_copy, the address found from its "@" (a word character, "." "+" or "-" before it is all the
# regex needs there): from every character of a long word the regex ran to the word's end looking for one (quadratic).
_WEB_LINK_RE = re.compile(r"https?://", re.IGNORECASE)
_WEB_HOST_RE = re.compile(r"www\.", re.IGNORECASE)
_EMAIL_RE = re.compile(r"(?<=[\w.+-])@[\w.-]+\.[A-Za-z]{2,}", re.IGNORECASE)


def contains_contact_copy(text: str) -> bool:
    """Whether the regex above finds a website or an e-mail address in ``text``, in linear time."""

    return bool(("://" in text and _WEB_LINK_RE.search(text)) or _WEB_HOST_RE.search(text)
                or ("@" in text and _EMAIL_RE.search(text)))


# (opening, closing) of the lists whose <li> items a required artifact counts (re.I as the former findall).
_LIST_TAGS: dict[str, tuple[re.Pattern[str], re.Pattern[str]]] = {
    tag: (re.compile(f"<{tag}>", re.IGNORECASE), re.compile(f"</{tag}>", re.IGNORECASE)) for tag in ("ol", "ul")
}


def preserved_list_items(html: str, tag: str) -> int:
    """The ``<li>`` tags of ``"".join(re.findall(rf"<{tag}>(.*?)</{tag}>", html, re.I | re.S))``, the list contents
    found in linear time (``app.core.markup.element_spans``; the findall scanned to the end from every unclosed list).
    """

    opening, closing = _LIST_TAGS[tag]
    contents = "".join(html[start:end] for _, start, end, _ in element_spans(html, opening, closing))
    return len(re.findall(r"<li>", contents, re.IGNORECASE))
