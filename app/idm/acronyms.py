"""Abbreviations in learner text (QLT-8, QC run 0a11aa9e: about 14 left unexplained, among them ROI, RACI, JD, BQS,
G8D and ATM).

After a unit is settled, the first use of an abbreviation in its provider html slots gets its full form in brackets
("SPD (Standard Product Dev)") when the facts the writer was given state one in that form, or when the server's
short list of common business abbreviations has it ("ROI" -> "tỷ suất hoàn vốn đầu tư"). An abbreviation the unit
already explains ("… (ROI)", "ROI (…)") or whose full form it already writes is left alone. Nothing is invented: an
abbreviation neither source knows (BQS, G8D, ATM in the run) is only listed in the author note for the SME.

Not abbreviations: words a heading writes in capitals ("THINK BIG", "SHIFT 2"), names the facts write before another
capitalised word ("FPT Software", "DEEP C Hải Phòng"), and the abbreviations every audience reads (CEO, AI, ISO …).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from app.idm.text import idm_fold

# Two to eight capitals or digits (and "&"), starting with a letter, not inside a word, a code ("CR-04") or a version
# ("ERA5.0").
ACRONYM_RE: Final = re.compile(r"(?<![\w&.\-])([A-ZĐ][A-ZĐ0-9&]{1,7})(?![\w&\-]|\.\d)")
# "SPD (Standard Product Dev)": the full form in brackets right after the abbreviation, two to ten words.
_EXPANDED_RE: Final = re.compile(r"(?<![\w&.\-])([A-ZĐ][A-ZĐ0-9&]{1,7})\s*\(([^()]{3,80})\)")
_MIN_EXPANSION_WORDS: Final = 2
_MAX_EXPANSION_WORDS: Final = 10
# A capitalised word after the abbreviation makes it part of a name ("FPT Software", "DEEP C"); a number makes it a
# label ("SHIFT 2", "MODUN 1").
_NAME_AFTER_RE: Final = r"\s+(?:[A-ZĐÀ-Ỹ][\w&.-]*|\d)"
# Two or more capitalised words in a row are a heading written in capitals ("THINK BIG", "SEE DIFFERENT").
_CAPS_RUN_RE: Final = re.compile(r"(?<![\w-])[A-ZĐ][A-ZĐ0-9&]+(?:\s+[A-ZĐ][A-ZĐ0-9&]+)+(?![\w-])")
_ROMAN_RE: Final = re.compile(r"^[IVXLC]+$")
# A word a slide writes in capitals ("ACTION", "CUSTOMER"): five letters or more with two vowels or more, or a short
# English word of a slide.
_WORD_MIN_LETTERS: Final = 5
_WORD_MIN_VOWELS: Final = 2
_SLIDE_WORDS: Final = frozenset({"PLAN", "DO", "CHECK", "ACT", "SEE", "BIG", "NEW", "OLD", "TOP", "KEY", "STEP", "NOTE",
                                  "TIP", "WOW", "GO", "YES", "NO", "ROW", "PART", "DAY", "WEEK", "THINK", "SHIFT",
                                  "MODUN", "LEVEL", "PHASE", "GOAL", "WHY", "HOW", "WHAT"})


def _slide_word(token: str) -> bool:
    letters = [char for char in token if char.isalpha()]
    vowels = sum(char in "AEIOUY" for char in letters)
    return token in _SLIDE_WORDS or (len(letters) == len(token) and len(letters) >= _WORD_MIN_LETTERS
                                     and vowels >= _WORD_MIN_VOWELS)

# Read by every audience, or names of certificates and currencies whose full form adds nothing.
KNOWN_ABBREVIATIONS: Final = frozenset({
    "CEO", "CFO", "COO", "CTO", "CIO", "CMO", "CPO", "CHRO", "AI", "IT", "HR", "ISO", "CE", "FDA", "USD", "VND", "EUR",
    "JPY", "OK", "PDF", "TV", "PC", "VN", "HCM", "TP", "ID", "URL", "EU", "US", "UK", "ASEAN", "GDP", "COVID", "VIP",
    "SMS", "OTP", "GPS", "QR", "USB", "PIN", "APP", "WEB", "EMAIL", "FAQ", "LED", "CV",
})
# Common business abbreviations with one usual meaning (the server's short list; a facts' own full form wins).
_COMMON_VI: Final[Mapping[str, str]] = {
    "ROI": "tỷ suất hoàn vốn đầu tư", "KPI": "chỉ số đo lường hiệu quả", "OKR": "mục tiêu và kết quả then chốt",
    "SOP": "quy trình thao tác chuẩn", "JD": "bản mô tả công việc",
    "RACI": "ma trận phân vai: người thực hiện, người chịu trách nhiệm, người được hỏi ý kiến, người được thông báo",
    "PDCA": "Lập kế hoạch - Thực hiện - Kiểm tra - Điều chỉnh", "SME": "doanh nghiệp nhỏ và vừa",
    "FDI": "đầu tư trực tiếp nước ngoài", "VUCA": "biến động, bất định, phức tạp, mơ hồ",
    "BANI": "mong manh, lo âu, phi tuyến, khó hiểu", "CNC": "điều khiển số bằng máy tính",
    "QA": "đảm bảo chất lượng", "QC": "kiểm soát chất lượng", "CRM": "quản lý quan hệ khách hàng",
    "ERP": "hoạch định nguồn lực doanh nghiệp", "B2B": "doanh nghiệp với doanh nghiệp",
    "B2C": "doanh nghiệp với người tiêu dùng", "ESG": "môi trường, xã hội và quản trị",
    "CSR": "trách nhiệm xã hội của doanh nghiệp", "SLA": "cam kết chất lượng dịch vụ",
    "R&D": "nghiên cứu và phát triển", "TNDN": "thu nhập doanh nghiệp", "GTGT": "giá trị gia tăng",
    "PCCC": "phòng cháy chữa cháy", "CNTT": "công nghệ thông tin", "BHXH": "bảo hiểm xã hội",
}
_COMMON_EN: Final[Mapping[str, str]] = {
    "ROI": "return on investment", "KPI": "key performance indicator", "OKR": "objectives and key results",
    "SOP": "standard operating procedure", "JD": "job description",
    "RACI": "responsible, accountable, consulted, informed", "PDCA": "Plan - Do - Check - Act",
    "SME": "small and medium-sized enterprise", "FDI": "foreign direct investment",
    "VUCA": "volatile, uncertain, complex, ambiguous", "BANI": "brittle, anxious, non-linear, incomprehensible",
    "CNC": "computer numerical control", "QA": "quality assurance", "QC": "quality control",
    "CRM": "customer relationship management", "ERP": "enterprise resource planning",
    "B2B": "business to business", "B2C": "business to consumer", "ESG": "environmental, social and governance",
    "CSR": "corporate social responsibility", "SLA": "service level agreement", "R&D": "research and development",
}
_TEXT_KINDS: Final = frozenset({"paragraph", "task", "warning"})
_LIST_KINDS: Final = frozenset({"bullets", "steps"})


@dataclass(frozen=True)
class AcronymGuide:
    """What the facts and the server know about the abbreviations of one unit."""

    expansions: Mapping[str, str]
    # Abbreviations the facts write as names or heading words, or with their full form before them ("… (SOP)").
    ignored: frozenset[str]

    @classmethod
    def build(cls, facts: Iterable[str], locale: str) -> AcronymGuide:
        texts = list(facts)
        found: dict[str, str] = {}
        ignored: set[str] = set()
        for text in texts:
            for match in _EXPANDED_RE.finditer(text):
                token, full = match.group(1), " ".join(match.group(2).split())
                words = full.split()
                if (_MIN_EXPANSION_WORDS <= len(words) <= _MAX_EXPANSION_WORDS
                        and idm_fold(words[0])[:1] == idm_fold(token)[:1] and not ACRONYM_RE.fullmatch(full)):
                    found.setdefault(token, full)
            for run in _CAPS_RUN_RE.findall(text):
                ignored.update(run.split())
            for match in ACRONYM_RE.finditer(text):
                token = match.group(1)
                if re.match(_NAME_AFTER_RE, text[match.end():]) and token not in found:
                    ignored.add(token)
                before = text[:match.start()].rstrip()
                if before.endswith("(") and text[match.end():].lstrip().startswith(")"):
                    ignored.add(token)
        common = _COMMON_VI if locale == "vi" else _COMMON_EN
        expansions = {**{token: full for token, full in common.items()}, **found}
        return cls(expansions, frozenset(ignored - set(expansions)))

    def expansion(self, token: str) -> str | None:
        return self.expansions.get(token)

    def worth_noting(self, token: str) -> bool:
        return (token not in KNOWN_ABBREVIATIONS and token not in self.ignored and not _ROMAN_RE.fullmatch(token)
                and not token.isdigit() and not _slide_word(token))


def _html_texts(component: Mapping[str, Any]) -> list[str]:
    semantic = component.get("semantic_content")
    texts: list[str] = []
    for section in semantic.get("sections") or [] if isinstance(semantic, dict) else []:
        for block in section.get("blocks") or [] if isinstance(section, dict) else []:
            if not isinstance(block, dict):
                continue
            if block.get("kind") in _TEXT_KINDS:
                texts.append(str(block.get("text") or ""))
            texts.extend(str(item) for item in block.get("items") or [])
            for row in block.get("rows") or []:
                if isinstance(row, dict):
                    texts.extend([str(row.get("label") or ""), str(row.get("value") or "")])
    return texts


def learner_texts(component: Mapping[str, Any]) -> list[str]:
    """The body texts of a slot, in reading order (no titles or headings)."""

    kind = component.get("type")
    if kind == "html":
        return _html_texts(component)
    if kind == "problem":
        choices = [str(choice.get("text") or "") for choice in component.get("choices") or []
                   if isinstance(choice, dict)]
        return [str(component.get("question") or ""), *choices, str(component.get("explanation") or "")]
    if kind == "la_faq":
        return [str(item.get(key) or "") for item in component.get("items") or [] if isinstance(item, dict)
                for key in ("question", "answer")]
    return []


def abbreviations(texts: Iterable[str]) -> list[str]:
    """The abbreviations of ``texts`` in order of first use."""

    return list(dict.fromkeys(match.group(1) for text in texts for match in ACRONYM_RE.finditer(text)))


def explained(token: str, full: str | None, texts: Sequence[str]) -> bool:
    """The unit already gives the full form: "(ROI)" after a phrase, "ROI (" with its meaning, or the full form."""

    joined = " ".join(texts)
    escaped = re.escape(token)
    if re.search(rf"\(\s*{escaped}\s*\)", joined) or re.search(rf"(?<![\w&]){escaped}\s*\(", joined):
        return True
    return full is not None and idm_fold(full) in idm_fold(joined)


def _insert(text: str, token: str, full: str) -> str | None:
    match = re.search(rf"(?<![\w&.\-]){re.escape(token)}(?![\w&\-]|\.\d)", text)
    if match is None:
        return None
    return f"{text[:match.end()]} ({full}){text[match.end():]}"


def with_expansion(component: dict[str, Any], token: str, full: str) -> dict[str, Any] | None:
    """``component`` (an html slot) with ``full`` in brackets after the first use of ``token`` in its body, or None
    when its body does not use it."""

    semantic = component.get("semantic_content")
    if not isinstance(semantic, dict):
        return None
    sections = [dict(section) for section in semantic.get("sections") or [] if isinstance(section, dict)]
    for section in sections:
        blocks = [dict(block) if isinstance(block, dict) else block for block in section.get("blocks") or []]
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if block.get("kind") in _TEXT_KINDS and (changed := _insert(str(block.get("text") or ""), token, full)):
                block["text"] = changed
            elif items := [str(item) for item in block.get("items") or []]:
                position = next((index for index, item in enumerate(items) if _insert(item, token, full)), None)
                if position is None:
                    continue
                items[position] = _insert(items[position], token, full) or items[position]
                block["items"] = items
            elif rows := [dict(row) for row in block.get("rows") or [] if isinstance(row, dict)]:
                spot = next(((index, key) for index, row in enumerate(rows) for key in ("label", "value")
                             if _insert(str(row.get(key) or ""), token, full)), None)
                if spot is None:
                    continue
                rows[spot[0]][spot[1]] = _insert(str(rows[spot[0]][spot[1]]), token, full)
                block["rows"] = rows
            else:
                continue
            section["blocks"] = blocks
            return {**component, "semantic_content": {**semantic, "sections": sections}}
    return None


__all__ = [
    "ACRONYM_RE", "KNOWN_ABBREVIATIONS", "AcronymGuide", "abbreviations", "explained", "learner_texts",
    "with_expansion",
]
