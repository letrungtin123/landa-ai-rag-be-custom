"""Words of the provider slots a unit note names (run 2f8781d6): the hype words of a slot (A4) and the details of a
classroom exercise of the source its learner text still carries (B8).

QC of run 2f8781d6 (T1, L3): the source's classroom exercises ("BÀI TẬP 10 phút: … chỉ chọn ra đúng 2 sáng kiến",
"Câu hỏi thảo luận 5 phút", "Bài tập về nhà theo nhóm") were written as rules of real work ("đội ngũ cần 10 phút phản
biện", "gồm đúng 2 sáng kiến"). The writer is told to turn such an exercise into a self-paced practice; what still
carries its time limit, its count to pick or its group or homework wording outside a worksheet is an edit note, never a
failure (``classroom_words``). Only details an exercise of the unit's facts states count, so a rule of the work ("phản
hồi trong 30 phút") is never named.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Collection, Iterable, Sequence
from typing import Any, Final

from app.idm.acronyms import learner_texts
from app.idm.faq_checks import FaqChecker
from app.idm.policy import FAQ_COMPONENT_TYPE
from app.idm.quiz_text import EXPLANATION_FILLER_CODE, explanation_report, lay_out_quiz
from app.idm.source_locked import EXERCISE_RE
from app.idm.text import EvidenceIndex, idm_fold

# A classroom exercise of the slides, accent-folded: its opening (``EXERCISE_RE``) or one of these words anywhere.
_EXERCISE_WORDS_RE: Final = re.compile(r"(?<!\w)(?:bai tap|thao luan|ve nha|homework|exercise|discussion)(?!\w)")
# What belongs to the exercise, not to the work: its time, the count to pick, the group or homework wording.
_DETAIL_RE: Final = re.compile(r"(?<![\w.])(?:\d{1,3}\s*(?:phút|minutes?|mins?)|đúng\s+\d{1,2}|exactly\s+\d{1,2}"
                               r"|về nhà|homework|theo nhóm|in (?:groups|teams))(?!\w)", re.IGNORECASE)
_MAX_WORDS: Final = 4


def _normal(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).lower().split())


def exercise_details(fact_texts: Iterable[str]) -> list[str]:
    """The details of the classroom exercises among ``fact_texts`` ("10 phút", "đúng 2", "về nhà", "theo nhóm")."""

    found: list[str] = []
    for text in fact_texts:
        folded = idm_fold(text)
        if EXERCISE_RE.match(folded) or _EXERCISE_WORDS_RE.search(folded):
            found.extend(_normal(match.group(0)) for match in _DETAIL_RE.finditer(_normal(text)))
    return list(dict.fromkeys(found))


def classroom_words(components: Sequence[Any], exempt: Collection[int], details: Sequence[str]) -> dict[int, list[str]]:
    """Slot -> the exercise ``details`` its learner text carries (``exempt``: the fallback and worksheet slots)."""

    found: dict[int, list[str]] = {}
    for index, component in enumerate(components):
        if index in exempt or not isinstance(component, dict) or not details:
            continue
        text = _normal(" ".join(learner_texts(component)))
        words = [detail for detail in details if re.search(rf"(?<![\w.]){re.escape(detail)}(?!\w)", text)]
        if words:
            found[index] = words[:_MAX_WORDS]
    return found


def hype_words(components: Sequence[Any], exempt: Collection[int], evidence: EvidenceIndex,
               faq: FaqChecker) -> dict[int, list[str]]:
    """Run 2f8781d6 (A4): slot -> the empty-praise or hype words of its explanation or FAQ answers (words of the fixed
    lists, ``quiz_text.filler_phrases`` and ``FaqChecker.hype``), for the note line that names them."""

    found: dict[int, list[str]] = {}
    for index, component in enumerate(components):
        if index in exempt or not isinstance(component, dict):
            continue
        if component.get("type") == "problem":
            words = explanation_report(lay_out_quiz(dict(component))[0], evidence).get(EXPLANATION_FILLER_CODE, [])
        elif component.get("type") == FAQ_COMPONENT_TYPE:
            words = faq.hype(component)
        else:
            continue
        if words:
            found[index] = list(words)
    return found


__all__ = ["classroom_words", "exercise_details", "hype_words"]
