"""Words of the provider slots a unit note names (run 2f8781d6): the hype words of a slot (A4) and the details of a
classroom exercise of the source its learner text still carries (B8).

QC of run 2f8781d6 (T1, L3): the source's classroom exercises ("BÀI TẬP 10 phút: … chỉ chọn ra đúng 2 sáng kiến",
"Câu hỏi thảo luận 5 phút", "Bài tập về nhà theo nhóm") were written as rules of real work ("đội ngũ cần 10 phút phản
biện", "gồm đúng 2 sáng kiến"). The writer is told to turn such an exercise into a self-paced practice; what still
carries its time limit, its count to pick or its group or homework wording outside a worksheet is an edit note, never a
failure (``classroom_words``). Only details an exercise of the unit's facts states count, so a rule of the work ("phản
hồi trong 30 phút") is never named.

Review 5 (M8): a fact is an exercise only when it opens as one ("BÀI TẬP …", "Câu hỏi thảo luận …", "Thực hành …",
"Nhiệm vụ thực hành …"), never because it mentions "về nhà" or "thảo luận" ("… để được về nhà đúng giờ", "Trưởng nhóm
thảo luận với khách hàng trong 30 phút đầu …"); and its homework or group wording counts in a slot only after an
exercise word ("Bài tập về nhà theo nhóm …"), never in "gom khách hàng theo nhóm có đặc điểm chung".
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
from app.idm.text import EvidenceIndex, idm_fold

# A classroom exercise of the slides, accent-folded: how it opens.
_EXERCISE_START_RE: Final = re.compile(r"^\W*(?:nhiem vu\s+)?(?:bai tap|cau hoi thao luan|thao luan|thuc hanh|homework"
                                       r"|exercise|discussion|group work)(?!\w)")
# What belongs to the exercise, not to the work: its time, the count to pick, the group or homework wording.
_DETAIL_RE: Final = re.compile(r"(?<![\w.])(?:\d{1,3}\s*(?:phút|minutes?|mins?)|đúng\s+\d{1,2}|exactly\s+\d{1,2}"
                               r"|về nhà|homework|theo nhóm|in (?:groups|teams))(?!\w)", re.IGNORECASE)
_MAX_WORDS: Final = 4
# The homework or group wording of an exercise, and the exercise words it must follow in a slot (within a few words).
_WORDING: Final = frozenset({"về nhà", "homework", "theo nhóm", "in groups", "in teams"})
_EXERCISE_LEAD: Final = (r"(?<!\w)(?:bài tập|thảo luận|thực hành|nhiệm vụ|dự án|làm việc|làm|homework|exercise"
                         r"|discussion|assignment|work)(?:\W+\w+){0,4}?\W+")
_ALONE: Final = r"(?<![\w.])"


def _normal(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).lower().split())


def exercise_details(fact_texts: Iterable[str], kept: Iterable[str] = ()) -> list[str]:
    """The details of the classroom exercises among ``fact_texts`` ("10 phút", "đúng 2", "về nhà", "theo nhóm"), but
    the ones an approved practice sentence keeps (``kept``: the lesson's PRACTICE_TASKS, review 6 L8)."""

    found: list[str] = []
    for text in fact_texts:
        if _EXERCISE_START_RE.match(idm_fold(text)):
            found.extend(_normal(match.group(0)) for match in _DETAIL_RE.finditer(_normal(text)))
    approved = " ".join(_normal(sentence) for sentence in kept)
    return [detail for detail in dict.fromkeys(found) if not re.search(rf"(?<![\w.]){re.escape(detail)}(?!\w)",
                                                                       approved)]


def classroom_words(components: Sequence[Any], exempt: Collection[int], details: Sequence[str]) -> dict[int, list[str]]:
    """Slot -> the exercise ``details`` its learner text carries (``exempt``: the fallback and worksheet slots)."""

    found: dict[int, list[str]] = {}
    for index, component in enumerate(components):
        if index in exempt or not isinstance(component, dict) or not details:
            continue
        text = _normal(" ".join(learner_texts(component)))
        words = [detail for detail in details if re.search(
            (_EXERCISE_LEAD if detail in _WORDING else _ALONE) + re.escape(detail) + r"(?!\w)", text)]
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
