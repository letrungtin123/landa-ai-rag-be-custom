"""Who a drafted example is about (QLT-7, QC run fae66524).

The explain-mode writer rule named fixed roles for drafted examples ("một nhân viên mới, một khách hàng, trưởng
nhóm"). In a course the author entered for "CEO, Founder và đội ngũ quản lý cấp cao", 11 of the 17 drafted examples
were about "một trưởng nhóm" (the source never says it). The roles now come from the audience (the one the author
entered, else the design audience) and, when its words name no role, from the purpose profile:

* ``audience_roles``: the levels the audience names and server-owned role phrases for the writer rule ("CEO" ->
  "một CEO"); author text never reaches a rule line, only these phrases do;
* ``contradicting_role``: the main person of a drafted example ("Ví dụ minh hoạ: Một trưởng nhóm …") whose level is
  far from an audience that is clearly executive (or clearly front-line staff). Conservative on purpose: only the
  first role introduced near the start of the example counts ("một <role>" within its first words), and only those two
  extremes; a manager course may well show a staff member.

Pure functions; the W5 check that uses them is ``fidelity_checks.FidelityChecker`` (one targeted repair, then an author
note, never a fallback).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Final, Literal

RoleLevel = Literal["executive", "manager", "team_lead", "staff"]
EXECUTIVE: Final[RoleLevel] = "executive"
STAFF: Final[RoleLevel] = "staff"
# (lower-case words with diacritics, level, Vietnamese phrase, English phrase). Longest match first, so "quản lý cấp
# cao" is an executive and not a manager, "nhân viên mới" a new employee and not any employee.
_ROLE_TERMS: Final[tuple[tuple[str, RoleLevel, str, str], ...]] = (
    ("ceo", "executive", "một CEO", "a CEO"),
    ("coo", "executive", "một giám đốc vận hành", "a COO"),
    ("cfo", "executive", "một giám đốc tài chính", "a CFO"),
    ("cto", "executive", "một giám đốc công nghệ", "a CTO"),
    ("c level", "executive", "một lãnh đạo cấp cao", "a senior executive"),
    ("founder", "executive", "một nhà sáng lập", "a founder"),
    ("co founder", "executive", "một nhà đồng sáng lập", "a co-founder"),
    ("nhà sáng lập", "executive", "một nhà sáng lập", "a founder"),
    ("đồng sáng lập", "executive", "một nhà đồng sáng lập", "a co-founder"),
    ("chủ doanh nghiệp", "executive", "một chủ doanh nghiệp", "a business owner"),
    ("chủ tịch", "executive", "một chủ tịch", "a chair"),
    ("hội đồng quản trị", "executive", "một thành viên hội đồng quản trị", "a board member"),
    ("hđqt", "executive", "một thành viên hội đồng quản trị", "a board member"),
    ("tổng giám đốc", "executive", "một tổng giám đốc", "a managing director"),
    ("giám đốc điều hành", "executive", "một giám đốc điều hành", "a chief executive"),
    ("giám đốc", "executive", "một giám đốc", "a director"),
    ("ban giám đốc", "executive", "một thành viên ban giám đốc", "a member of the management board"),
    ("ban lãnh đạo", "executive", "một lãnh đạo cấp cao", "a senior leader"),
    ("lãnh đạo cấp cao", "executive", "một lãnh đạo cấp cao", "a senior leader"),
    ("quản lý cấp cao", "executive", "một lãnh đạo cấp cao", "a senior leader"),
    ("quản lí cấp cao", "executive", "một lãnh đạo cấp cao", "a senior leader"),
    ("lãnh đạo doanh nghiệp", "executive", "một lãnh đạo doanh nghiệp", "a business leader"),
    ("người đứng đầu", "executive", "một người đứng đầu doanh nghiệp", "a company head"),
    ("executive", "executive", "một lãnh đạo cấp cao", "an executive"),
    ("executives", "executive", "một lãnh đạo cấp cao", "an executive"),
    ("senior leader", "executive", "một lãnh đạo cấp cao", "a senior leader"),
    ("senior leaders", "executive", "một lãnh đạo cấp cao", "a senior leader"),
    ("senior management", "executive", "một lãnh đạo cấp cao", "a senior leader"),
    ("business owner", "executive", "một chủ doanh nghiệp", "a business owner"),
    ("managing director", "executive", "một tổng giám đốc", "a managing director"),
    ("quản lý cấp trung", "manager", "một quản lý cấp trung", "a middle manager"),
    ("quản lý", "manager", "một quản lý", "a manager"),
    ("quản lí", "manager", "một quản lý", "a manager"),
    ("trưởng phòng", "manager", "một trưởng phòng", "a department head"),
    ("phó phòng", "manager", "một phó phòng", "a deputy department head"),
    ("trưởng bộ phận", "manager", "một trưởng bộ phận", "a department head"),
    ("trưởng chi nhánh", "manager", "một trưởng chi nhánh", "a branch manager"),
    ("lãnh đạo", "manager", "một lãnh đạo", "a leader"),
    ("manager", "manager", "một quản lý", "a manager"),
    ("managers", "manager", "một quản lý", "a manager"),
    ("middle manager", "manager", "một quản lý cấp trung", "a middle manager"),
    ("department head", "manager", "một trưởng phòng", "a department head"),
    ("supervisor", "manager", "một giám sát viên", "a supervisor"),
    ("giám sát viên", "manager", "một giám sát viên", "a supervisor"),
    ("trưởng nhóm", "team_lead", "một trưởng nhóm", "a team lead"),
    ("nhóm trưởng", "team_lead", "một trưởng nhóm", "a team lead"),
    ("tổ trưởng", "team_lead", "một tổ trưởng", "a team lead"),
    ("trưởng ca", "team_lead", "một trưởng ca", "a shift lead"),
    ("team lead", "team_lead", "một trưởng nhóm", "a team lead"),
    ("team leader", "team_lead", "một trưởng nhóm", "a team lead"),
    ("shift lead", "team_lead", "một trưởng ca", "a shift lead"),
    ("nhân viên mới", "staff", "một nhân viên mới", "a new employee"),
    ("nhân viên kinh doanh", "staff", "một nhân viên kinh doanh", "a salesperson"),
    ("nhân viên bán hàng", "staff", "một nhân viên bán hàng", "a salesperson"),
    ("nhân viên", "staff", "một nhân viên", "an employee"),
    ("chuyên viên", "staff", "một chuyên viên", "a specialist"),
    ("công nhân", "staff", "một công nhân", "a worker"),
    ("thực tập sinh", "staff", "một thực tập sinh", "an intern"),
    ("người lao động", "staff", "một người lao động", "a worker"),
    ("tư vấn viên", "staff", "một tư vấn viên", "an advisor"),
    ("giao dịch viên", "staff", "một giao dịch viên", "a teller"),
    ("kỹ thuật viên", "staff", "một kỹ thuật viên", "a technician"),
    ("employee", "staff", "một nhân viên", "an employee"),
    ("employees", "staff", "một nhân viên", "an employee"),
    ("new hire", "staff", "một nhân viên mới", "a new hire"),
    ("new hires", "staff", "một nhân viên mới", "a new hire"),
    ("new employee", "staff", "một nhân viên mới", "a new employee"),
    ("new employees", "staff", "một nhân viên mới", "a new employee"),
    ("staff", "staff", "một nhân viên", "a staff member"),
    ("intern", "staff", "một thực tập sinh", "an intern"),
    ("worker", "staff", "một công nhân", "a worker"),
)
_TERMS_BY_LENGTH: Final = sorted(_ROLE_TERMS, key=lambda term: len(term[0].split()), reverse=True)
# The roles a purpose profile implies when the audience names none (the onboarding audience is new employees, the
# sales audience salespeople); other profiles keep the generic rule.
_PROFILE_ROLES: Final[dict[str, tuple[str, str]]] = {
    "onboarding": ("một nhân viên mới", "a new employee"),
    "sales": ("một nhân viên kinh doanh", "a salesperson"),
}
_WORD_RE: Final = re.compile(r"\w+")
# Possessive spaces (only spaces stand there): the same match without the quadratic split of one run of spaces
# between two \s* (wave 3 hardening).
_EXAMPLE_LABEL_RE: Final = re.compile(r"^\s*+[\"'(]*\s*+(?:ví dụ minh hoạ|ví dụ minh họa|illustrative example)\s*+:\s*",
                                      re.IGNORECASE)
_FIRST_SENTENCE_RE: Final = re.compile(r"[.!?;]\s")
# The main person is introduced near the start: "Một trưởng nhóm …", "Sau đây là cách một trưởng nhóm …", "Khi phát
# sinh lỗi giao hàng trễ, một trưởng nhóm …"; a role named later is usually someone the main person deals with.
_PROTAGONIST_MAX_WORDS_BEFORE: Final = 10
_ARTICLES: Final = frozenset({"một", "các", "những", "a", "an", "the"})
_MAX_PHRASES: Final = 3


def _words(text: str) -> list[str]:
    """Lower-case words with their diacritics: folded, "công nhận" (to recognise) would read as "công nhân" (worker)."""

    return _WORD_RE.findall(unicodedata.normalize("NFC", text).lower())


def _role_mentions(words: list[str]) -> list[tuple[int, tuple[str, RoleLevel, str, str]]]:
    """(position, term) of every role named in ``words``, longest terms first, never overlapping."""

    found: list[tuple[int, tuple[str, RoleLevel, str, str]]] = []
    taken: set[int] = set()
    for term in _TERMS_BY_LENGTH:
        size = len(term[0].split())
        target = term[0].split()
        for start in range(len(words) - size + 1):
            span = set(range(start, start + size))
            if words[start:start + size] == target and not span & taken:
                found.append((start, term))
                taken |= span
    return sorted(found, key=lambda item: item[0])


@dataclass(frozen=True)
class AudienceRoles:
    """The levels an audience names and the role phrases (server-owned, in the run locale) a drafted example may use."""

    levels: frozenset[RoleLevel]
    phrases: tuple[str, ...]


def audience_roles(audience: str | None, profile: str | None, locale: str) -> AudienceRoles:
    """The roles of ``audience`` (the author's words): known role words only; the profile's role when it names none."""

    vi = locale == "vi"
    mentions = _role_mentions(_words(audience or ""))
    phrases = list(dict.fromkeys(term[2] if vi else term[3] for _position, term in mentions))[:_MAX_PHRASES]
    if not phrases and profile in _PROFILE_ROLES:
        phrases = [_PROFILE_ROLES[profile][0 if vi else 1]]
    return AudienceRoles(frozenset(term[1] for _position, term in mentions), tuple(phrases))


def role_rule(roles: AudienceRoles) -> str:
    """The rule fragment naming the roles a drafted example may give its main person."""

    if not roles.phrases:
        return "the learner's own role as TARGET_AUDIENCE describes it"
    named = ", ".join(f'"{phrase}"' for phrase in roles.phrases)
    return f"the learner's own role as TARGET_AUDIENCE describes it, for example {named}"


def example_body(text: str) -> str | None:
    """The text after the illustrative-example label, or ``None`` when ``text`` is not a labelled example."""

    match = _EXAMPLE_LABEL_RE.match(unicodedata.normalize("NFC", text))
    return text[match.end():] if match else None


def protagonist_level(example: str) -> RoleLevel | None:
    """The level of the main person of an example: the first role introduced ("một <role>", or the opening words)
    within its first words; ``None`` when the example names none there."""

    first = _FIRST_SENTENCE_RE.split(" ".join(example.split()), maxsplit=1)[0]
    words = _words(first)
    for position, term in _role_mentions(words):
        if position > _PROTAGONIST_MAX_WORDS_BEFORE:
            return None
        if position == 0 or words[position - 1] in _ARTICLES:
            return term[1]
    return None


def contradicting_role(roles: AudienceRoles, example: str) -> bool:
    """The example's main person is a team lead or a staff member in a course whose audience is only executives, or
    an executive in a course whose audience is only front-line staff."""

    level = protagonist_level(example)
    if level is None or not roles.levels:
        return False
    if roles.levels == {EXECUTIVE}:
        return level in {"team_lead", STAFF}
    if roles.levels == {STAFF}:
        return level == EXECUTIVE
    return False


__all__ = [
    "AudienceRoles", "RoleLevel", "audience_roles", "contradicting_role", "example_body", "protagonist_level",
    "role_rule",
]
