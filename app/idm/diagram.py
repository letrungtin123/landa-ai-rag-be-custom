"""Source-locked step diagram for IDM ``la_diagram`` slots (run e869f43a).

The shared builder (``app.main._orchestration_v2_source_relationship_diagram`` on top of
``app.instructional_quality.source_relationship_pairs``) treats every arrow as a
relationship. A slogan such as "Chain title: no action → meaningless" therefore became the
nodes "Chain title: no action" → "meaningless" next to the real step chain, and the same
pair was required from the provider diagram (``DIAGRAM_SOURCE_RELATION_MISSING``). The
legacy builder stays byte-identical; IDM units inject this one instead.

Only step-like source items become nodes:

* an arrow chain of at least three short labels ("A → B → C"); a chain title before a colon
  is dropped, a single arrow ("X → Y", an equation or a slogan) is not a sequence;
* otherwise a numbered procedure ("1. ...", "Bước 2: ...") of at least three steps.

Headings, sentences and equations are never nodes, and a chain with any such part is not
used at all (dropping a middle part would invent an edge). With fewer than three step nodes
there is no fallback (``None``), so the per-slot logic keeps a validator-clean provider slot
or marks the unit for review. :func:`idm_diagram_relationships` returns the required
provider edges from the same construction, so validation and fallback can never disagree.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Final

from app.instructional_quality import ORDERED_STEP_RE, TABLE_ROW_RE, clean_source_facts, normalize_visible_text

_ARROW_RE: Final = re.compile(r"\s*(?:→|->|=>|⇒|⟶|➔|➜|➝|⮕)\s*")
_BIDIRECTIONAL_RE: Final = re.compile(r"↔|<->|⇔|<=>")
_INLINE_STEP_RE: Final = re.compile(r"(?=(?<!\w)(?:step|bước|buoc)?\s*\d{1,3}\s*[:.)-]\s+)", re.IGNORECASE)
_EQUATION_RE: Final = re.compile(r"(?<![<>=!])=(?![>=])|≈|≠")
_SENTENCE_BREAK_RE: Final = re.compile(r"[.!?;](?:\s|$)")
_QUOTES: Final = "\"'“”‘’«»"  # noqa: RUF001 - typographic quotes are intended
_LABEL_TRIM: Final = " \t-–—:;,.|•*"  # noqa: RUF001 - en dash and bullet are intended
_MIN_CHAIN_NODES: Final = 3
_MIN_DIAGRAM_NODES: Final = 3
_MAX_NODES: Final = 10
_CHAIN_LABEL_MAX_WORDS: Final = 8
_CHAIN_LABEL_MAX_CHARS: Final = 80
_STEP_MAX_WORDS: Final = 24
_STEP_MAX_CHARS: Final = 160
_MIN_STEP_CHARS: Final = 2


@dataclass(frozen=True)
class _StepDiagram:
    labels: list[str]
    edges: list[tuple[int, int]]
    from_arrows: bool


def _label(value: str, *, max_words: int, max_chars: int) -> str | None:
    """A short step label, or ``None`` for a heading prefix, sentence, quote or equation."""

    text = normalize_visible_text(value).strip(_LABEL_TRIM)
    if (len(text) < _MIN_STEP_CHARS or len(text) > max_chars or len(text.split()) > max_words
            or ":" in text or _EQUATION_RE.search(text) or _SENTENCE_BREAK_RE.search(text)
            or text[:1] in _QUOTES or text[-1:] in _QUOTES or not any(char.isalpha() for char in text)):
        return None
    return text


def _arrow_chain(fact: str) -> list[str] | None:
    if TABLE_ROW_RE.match(fact) or _BIDIRECTIONAL_RE.search(fact):
        return None
    parts = _ARROW_RE.split(fact)
    if len(parts) < _MIN_CHAIN_NODES:
        return None
    if ":" in parts[0]:
        parts[0] = parts[0].rsplit(":", 1)[1]  # "Chain title: A → B → C" -> the chain starts at A
    labels = [_label(part, max_words=_CHAIN_LABEL_MAX_WORDS, max_chars=_CHAIN_LABEL_MAX_CHARS) for part in parts]
    if any(label is None for label in labels):
        return None
    return [label for label in labels if label is not None]


def _numbered_steps(facts: list[str]) -> list[str] | None:
    found: dict[int, str | None] = {}
    for fact in facts:
        if TABLE_ROW_RE.match(fact):
            continue
        for candidate in _INLINE_STEP_RE.split(fact):
            match = ORDERED_STEP_RE.match(candidate.strip())
            if match:
                found.setdefault(int(match.group(1)), _label(match.group(2), max_words=_STEP_MAX_WORDS,
                                                             max_chars=_STEP_MAX_CHARS))
    if len(found) < _MIN_DIAGRAM_NODES or any(label is None for label in found.values()):
        return None
    return [label for _order, label in sorted(found.items()) if label is not None]


def _key(label: str) -> str:
    return re.sub(r"[^\w\d]+", " ", label.casefold()).strip()


def _build(values: Iterable[Any], locale: str) -> _StepDiagram | None:
    del locale  # labels come from the source; edge labels are set by the caller
    facts = clean_source_facts(values, preserve_table_numeric=True)
    chains = [chain for fact in facts if (chain := _arrow_chain(fact)) is not None]
    from_arrows = bool(chains)
    if not chains:
        steps = _numbered_steps(facts)
        chains = [steps] if steps else []
    labels: list[str] = []
    index_of: dict[str, int] = {}
    edges: list[tuple[int, int]] = []
    for chain in chains:
        previous: int | None = None
        for label in chain:
            key = _key(label)
            if key not in index_of:
                if len(labels) >= _MAX_NODES:
                    break  # never an edge to a node that is not shown
                index_of[key] = len(labels)
                labels.append(label)
            current = index_of[key]
            if previous is not None and previous != current and (previous, current) not in edges:
                edges.append((previous, current))
            previous = current
    if len(labels) < _MIN_DIAGRAM_NODES or not edges:
        return None
    return _StepDiagram(labels, edges, from_arrows)


def idm_source_step_diagram(title: str, fact_texts: Iterable[Any], locale: str) -> dict[str, Any] | None:
    """The source-locked ``la_diagram`` payload of an IDM slot, or ``None`` (no step structure)."""

    diagram = _build(fact_texts, locale)
    if diagram is None:
        return None
    next_label = None if diagram.from_arrows else ("Next" if locale == "en" else "Tiếp theo")
    return {
        "name": title,
        "nodes": [{"label": label[:500], "shape": "rounded", "tooltip": label[:500]} for label in diagram.labels],
        "edges": [{"source": source, "target": target, **({"label": next_label} if next_label else {})}
                  for source, target in diagram.edges],
    }


def idm_diagram_relationships(fact_texts: Iterable[Any]) -> list[list[str]]:
    """Source relationships an IDM provider diagram must keep: the edges of an arrow step chain.

    Numbered procedures are drawn by the fallback but never required from the provider (the
    shared check compares exact labels, and the provider may word a step differently).
    """

    diagram = _build(fact_texts, "vi")
    if diagram is None or not diagram.from_arrows:
        return []
    return [[diagram.labels[source], diagram.labels[target]] for source, target in diagram.edges]
