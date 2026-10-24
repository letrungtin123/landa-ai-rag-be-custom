"""Tag and element regexes in linear time (wave 3 of the hardening): ``app.core.markup.count_tags``, ``sub_tags`` and
``element_spans`` and the functions that use them, against the regexes they replace.

The references below are the code before the change, kept verbatim: the sanitizer tag pattern of the Node mirror,
``_count_tags`` / ``_count_list_items`` and ``workspace_html_code`` (``app.idm.node_acceptance_html``), the label,
choice and artifact patterns of ``app.lesson_quality`` and the list items of the staged unit check. Each was
quadratic on unclosed tags (20,000 "<style x": 23.6 s in the sanitizer).
"""

from __future__ import annotations

import random
import re
import time
import unittest
from collections.abc import Callable
from typing import Any

from app import lesson_quality
from app.core.markup import count_tags, element_spans, remove_elements, sub_tags
from app.idm import node_acceptance_html as html_rules
from app.idm.node_acceptance_text import _JS_WS, js_trim
from app.services.staged_unit.markup_scans import preserved_list_items

FUZZ_CASES = 50_000
MAX_SECONDS = 0.05
MEGABYTE = 1_000_000
LONG_S, DOTTED_I, DOTLESS_I, KELVIN = chr(0x17F), chr(0x130), chr(0x131), chr(0x212A)
NODE_TAGS = ("li", "ol", "ul", "table", "tr", "blockquote")
# Tags whole, cut, unclosed, with attributes and in every case; letters a case folding could take for an ASCII one;
# the JavaScript white space of the Node patterns; plain text.
PIECES = (
    "<li", "<li ", "<li>", "</li>", "<LI x>", "<li\tclass=a>", "<lix>", "<ol", "<ol>", "<ol type=1>", "</ol>", "</OL>",
    "<ul", "<ul>", "</ul>", "<table", "<table>", "<TABLE border=1>", "</table>", "<tr", "<tr>", "<TR x>", "</tr>",
    "</Tr>", "<td>", "<th x>", "<td", "</td>", "<blockquote>", "<blockquote cite=x>", "</blockquote>", "<choice",
    "<choice>", "<choice correct=\"true\">", "</choice>", "<option x>", "</option>", "<Option>", "<choices>",
    "<label>", "</label>", "<LABEL>", "<style x", "<style>", "</style>", "<script>", "</script>", "<p>", "</p>",
    "<h2>", "<b>", "<a href=x>", "<1>", "</", "<", ">", ">", "<>", "x", "y z", " ", "\n", "\t", "\u00a0", "\u2028",
    "\ufeff", "Đường", "ữ", LONG_S, DOTTED_I, DOTLESS_I, KELVIN, f"<{LONG_S}tyle>", f"<l{DOTTED_I}>",
    f"<cho{DOTLESS_I}ce>", f"</cho{DOTTED_I}ce>", f"<{KELVIN}>",
)


def fuzz_cases(seed: int, count: int = FUZZ_CASES) -> list[str]:
    generator = random.Random(seed)  # noqa: S311 - seeded test data, not security
    return ["".join(generator.choices(PIECES, k=generator.randint(0, 24))) for _ in range(count)]


# --- references: the code before the change -------------------------------------------------------------------
def old_count_tags(html: str, tag: str) -> int:
    return len(re.findall(f"<{tag}(?:[{_JS_WS}][^>]*)?>", html, flags=re.IGNORECASE | re.ASCII))


def old_count_list_items(html: str, tag: str) -> int:
    segments = re.findall(f"<{tag}(?:[{_JS_WS}][^>]*)?>([\\s\\S]*?)</{tag}>", html, flags=re.IGNORECASE | re.ASCII)
    return sum(old_count_tags(segment, "li") for segment in segments)


def old_sanitize(value: Any) -> str:
    text = value if isinstance(value, str) else ""
    for tag in ("script", "style", "iframe", "object", "embed"):
        text = remove_elements(text, tag)

    def keep(match: re.Match[str]) -> str:
        tag = match.group(1).lower()
        if tag not in html_rules._SAFE_HTML_TAGS:
            return ""
        return f"</{tag}>" if match.group(0).startswith("</") else f"<{tag}>"

    return js_trim(html_rules._SANITIZE_TAG_RE.sub(keep, text))


def old_workspace_html_code(html: str) -> str | None:
    if len(html.encode("utf-8", "surrogatepass")) > 250_000:
        return "HTML_TOO_LARGE"
    cells = 0
    for table in re.findall(r"<table\b[^>]*>[\s\S]*?</table>", html, re.IGNORECASE | re.ASCII):
        if len(re.findall(r"<tr\b[^>]*>", table, re.I | re.A)) > 100:
            return "HTML_TABLE_TOO_LARGE"
        for row in re.findall(r"<tr\b[^>]*>[\s\S]*?</tr>", table, re.IGNORECASE | re.ASCII):
            row_cells = len(re.findall(r"<(?:td|th)\b[^>]*>", row, re.IGNORECASE | re.ASCII))
            if row_cells > 50:
                return "HTML_TABLE_TOO_LARGE"
            cells += row_cells
    return "HTML_TABLE_TOO_LARGE" if cells > 10_000 else None


def old_label(xml: str) -> str:
    match = re.search(r"<label>([\s\S]*?)</label>", xml, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def old_choices(xml: str) -> list[str]:
    return [match.group(1) for match in
            re.finditer(r"<(?:choice|option)\b[^>]*>([\s\S]*?)</(?:choice|option)>", xml, flags=re.IGNORECASE)]


def old_list_count(html_value: str, tag: str) -> int:
    if not re.search(rf"<{tag}\b", html_value, re.IGNORECASE):
        return 0
    return len(re.findall(r"<li\b[^>]*>", re.sub(rf"(?is)^.*?<{tag}\b[^>]*>", "", html_value)))


def old_table_count(html_value: str) -> int:
    if not re.search(r"<table\b", html_value, re.IGNORECASE):
        return 0
    return len(re.findall(r"<tr\b[^>]*>", html_value, re.IGNORECASE)) - 1


def old_blockquote_count(html_value: str) -> int:
    return len(re.findall(r"<blockquote\b[^>]*>", html_value, re.IGNORECASE))


def old_preserved_list_items(html: str, tag: str) -> int:
    return len(re.findall(r"<li>", "".join(re.findall(rf"<{tag}>(.*?)</{tag}>", html, re.IGNORECASE | re.DOTALL)),
                          re.IGNORECASE))


def artifact_count(html_value: str, artifact_type: str) -> int:
    return lesson_quality._html_artifact_item_count({"html": html_value}, artifact_type)


# Every pair: (name, new, old) over one string.
PAIRS: tuple[tuple[str, Callable[[str], Any], Callable[[str], Any]], ...] = (
    ("sanitize_lesson_author_html", html_rules.sanitize_lesson_author_html, old_sanitize),
    *((f"_count_tags {tag}", lambda text, tag=tag: html_rules._count_tags(text, tag),
       lambda text, tag=tag: old_count_tags(text, tag)) for tag in NODE_TAGS),
    *((f"_count_list_items {tag}", lambda text, tag=tag: html_rules._count_list_items(text, tag),
       lambda text, tag=tag: old_count_list_items(text, tag)) for tag in ("ol", "ul")),
    ("workspace_html_code", html_rules.workspace_html_code, old_workspace_html_code),
    ("_problem_question", lambda text: lesson_quality._problem_question({"data": text}),
     lambda text: lesson_quality._plain_text(old_label(text))),
    ("_problem_choices", lambda text: lesson_quality._problem_choices({"data": text}),
     lambda text: [lesson_quality._plain_text(item) for item in old_choices(text) if lesson_quality._plain_text(item)]),
    ("ordered_list artifact", lambda text: artifact_count(text, "ordered_list"),
     lambda text: old_list_count(text, "ol")),
    ("checklist artifact", lambda text: artifact_count(text, "checklist"), lambda text: old_list_count(text, "ul")),
    ("table artifact", lambda text: artifact_count(text, "table"), old_table_count),
    ("warning artifact", lambda text: artifact_count(text, "warning"), old_blockquote_count),
    *((f"preserved_list_items {tag}", lambda text, tag=tag: preserved_list_items(text, tag),
       lambda text, tag=tag: old_preserved_list_items(text, tag)) for tag in ("ol", "ul")),
)

HAND_CASES = (
    "", "<li>", "<li", "<li x", "<li x>", "<ol><li>a</li></ol>", "<ol><li>a", "<ol>" * 3 + "</ol>", "<ol x<li>></ol>",
    "<OL><LI>a</li></OL><ol><li>b<li>c</ol>", "<ol <li>a</ol>", "<ol>\n<li x>a</li>\n</ol>", "<li\u00a0x>",
    "<table><tr><td>1</td><th>2</th></tr></table>", "<table><tr><td>1</tr>", "<table <tr>x</table>",
    "<table>" + "<tr><td>a</td></tr>" * 3 + "</table>" * 2, "<label>Q</label>", "<label>a<label>b</label>",
    "<LABEL>x</label>", "<choice>a</choice><option x>b</option><choice>c</option>", "<choices>a</choice>",
    "<choice>a</choice", "<style x<b>c</b>", "<p>a<style>b</style>c</p>", f"<cho{DOTLESS_I}ce>a</choice>",
    f"<{KELVIN}>", "<ul><li>a</li></ul><ol><li>b</li><li>c</li></ol>", "<blockquote x>q</blockquote>",
)


class TagPatternParityTests(unittest.TestCase):
    def test_hand_cases_give_the_regex_results(self) -> None:
        for name, new, old in PAIRS:
            for text in HAND_CASES:
                with self.subTest(name=name, text=text):
                    self.assertEqual(new(text), old(text))

    def test_random_markup_gives_the_regex_results(self) -> None:
        cases = fuzz_cases(20261024)
        for name, new, old in PAIRS:
            for text in cases:
                if new(text) != old(text):
                    self.fail(f"{name} differs from the regex on {text!r}: {new(text)!r} != {old(text)!r}")

    def test_the_helpers_give_the_regex_results(self) -> None:
        tag_re = re.compile(r"<(t[dh])\b[^>]*>", re.IGNORECASE | re.ASCII)
        opening, closing = re.compile(r"<tr\b[^>]*>", re.IGNORECASE), re.compile(r"</tr>", re.IGNORECASE)
        element_re = re.compile(r"<tr\b[^>]*>([\s\S]*?)</tr>", re.IGNORECASE)
        for text in fuzz_cases(7, 20_000) + list(HAND_CASES):
            self.assertEqual(count_tags(tag_re, text), len(tag_re.findall(text)), text)
            self.assertEqual(sub_tags(tag_re, lambda match: f"[{match.group(1)}]", text),
                             tag_re.sub(lambda match: f"[{match.group(1)}]", text), text)
            self.assertEqual(list(element_spans(text, opening, closing)),
                             [(match.start(), match.start(1), match.end(1), match.end())
                              for match in element_re.finditer(text)], text)


ADVERSARIAL: dict[str, str] = {
    "<style x": "<style x" * (MEGABYTE // 8),
    "<a ": "<a " * (MEGABYTE // 3),
    "<li x": "<li x" * (MEGABYTE // 5),
    "<li x then >": "<li x" * (MEGABYTE // 5) + ">",
    "<ol>": "<ol>" * (MEGABYTE // 4),
    "<ol x": "<ol x" * (MEGABYTE // 5),
    "<table>": "<table>" * (MEGABYTE // 7),
    "<table><tr>": "<table>" + "<tr>" * (MEGABYTE // 4) + "</table>",
    "<table><tr x": "<table>" + "<tr x" * (MEGABYTE // 5) + "</table>",
    "<label>": "<label>" * (MEGABYTE // 7),
    "<choice>": "<choice>" * (MEGABYTE // 8),
    "<choice x": "<choice x" * (MEGABYTE // 9),
    "<blockquote x": "<blockquote x" * (MEGABYTE // 13),
}


def fastest_seconds(function: Callable[[str], Any], text: str) -> float:
    timings = []
    for _ in range(3):
        started = time.perf_counter()
        function(text)
        timings.append(time.perf_counter() - started)
    return min(timings)


# Inputs of a megabyte of complete tags the sanitizer keeps or drops one by one (a Python call per tag, as before):
# linear for the regex too, so they are held to the linear bound only.
SANITIZER_MANY_TAGS = frozenset({"<ol>", "<table>", "<table><tr>", "<label>", "<choice>"})
LINEAR_SECONDS = 0.5


class TagPatternTimingTests(unittest.TestCase):
    def test_a_megabyte_of_unclosed_tags_takes_milliseconds(self) -> None:
        # Before: 20,000 "<style x" took 23.6 s in the sanitizer, _count_tags 8.9 s, _count_list_items 7.7 s,
        # workspace_html_code 21.7 s and the legacy choice pattern 27 s, growing with the square of the length.
        for name, function, _old in PAIRS:
            for input_name, text in ADVERSARIAL.items():
                many_tags = name == "sanitize_lesson_author_html" and input_name in SANITIZER_MANY_TAGS
                with self.subTest(function=name, input=input_name):
                    self.assertLess(fastest_seconds(function, text), LINEAR_SECONDS if many_tags else MAX_SECONDS)


if __name__ == "__main__":
    unittest.main()
