"""The organization profile v1 and its frozen wire object (ORG_CONTEXT spec §2), shared with Node and the dashboard.

Authors keep one tenant-wide "Hồ sơ doanh nghiệp" (organization profile) that AI drafts from their context documents
(``app.idm.org_distill``) and they edit and confirm. A run started with it freezes the confirmed version as the
workspace's ``organization_context`` (``IdmOrganizationContextV1``): background for the design, never course knowledge
(``app.idm.org_context``).

Node normalises the profile once, when an author saves it (NFC, control characters, spaces, empty items), and hashes
that text with its ``canonicalJson``. This service never rewrites a received profile: the models below are strict and
closed but never strip a string, so the echo of a project context and the hash check see exactly what Node sent.
"""

from __future__ import annotations

from typing import Annotated, Final

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from app.hashing import js_canonical_hash

# Bounds of profile v1. Lengths are Unicode code points (``len`` here, ``[...s].length`` in JavaScript).
ORG_SUMMARY_MAX: Final = 800
ORG_CULTURE_TONE_MAX: Final = 600
ORG_PRODUCTS_MAX_ITEMS: Final = 15
ORG_PRODUCT_MAX: Final = 200
ORG_GROUPS_MAX_ITEMS: Final = 12
ORG_GROUP_NAME_MAX: Final = 100
ORG_GROUP_DESCRIPTION_MAX: Final = 300
ORG_GLOSSARY_MAX_ITEMS: Final = 40
ORG_TERM_MAX: Final = 80
ORG_MEANING_MAX: Final = 300
ORG_PROCESSES_MAX_ITEMS: Final = 15
ORG_PROCESS_MAX: Final = 300
ORG_CONVENTIONS_MAX_ITEMS: Final = 10
ORG_CONVENTION_MAX: Final = 200
# ``char_count``: the sum of the lengths of every string of the profile (``org_profile_char_count``).
ORG_PROFILE_MAX_CHARS: Final = 6000
# Node's normalisation keeps line breaks only in organization_summary and culture_tone, at most this many lines each;
# a distilled draft follows the same rule (``app.idm.org_distill.fit_profile``).
ORG_TEXT_MAX_LINES: Final = 12
# ``profile_version`` is a Postgres ``int``.
_MAX_VERSION: Final = 2_147_483_647

_Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_Product = Annotated[str, StringConstraints(min_length=1, max_length=ORG_PRODUCT_MAX)]
_Process = Annotated[str, StringConstraints(min_length=1, max_length=ORG_PROCESS_MAX)]
_Convention = Annotated[str, StringConstraints(min_length=1, max_length=ORG_CONVENTION_MAX)]


class IdmOrgModel(BaseModel):
    """Strict and closed like every IDM contract, without whitespace stripping (see the module docstring)."""

    model_config = ConfigDict(extra="forbid", strict=True)


class IdmOrgLearnerGroupV1(IdmOrgModel):
    name: str = Field(min_length=1, max_length=ORG_GROUP_NAME_MAX)
    description: str = Field(max_length=ORG_GROUP_DESCRIPTION_MAX)


class IdmOrgGlossaryTermV1(IdmOrgModel):
    term: str = Field(min_length=1, max_length=ORG_TERM_MAX)
    meaning: str = Field(min_length=1, max_length=ORG_MEANING_MAX)


class IdmOrgProfileV1(IdmOrgModel):
    """The organization profile v1: background for the course design, never course knowledge. All seven keys are
    always present; the total length of its strings is at most 6000 characters."""

    organization_summary: str = Field(max_length=ORG_SUMMARY_MAX)
    products_services: list[_Product] = Field(max_length=ORG_PRODUCTS_MAX_ITEMS)
    learner_groups: list[IdmOrgLearnerGroupV1] = Field(max_length=ORG_GROUPS_MAX_ITEMS)
    glossary: list[IdmOrgGlossaryTermV1] = Field(max_length=ORG_GLOSSARY_MAX_ITEMS)
    key_processes: list[_Process] = Field(max_length=ORG_PROCESSES_MAX_ITEMS)
    culture_tone: str = Field(max_length=ORG_CULTURE_TONE_MAX)
    design_conventions: list[_Convention] = Field(max_length=ORG_CONVENTIONS_MAX_ITEMS)

    @model_validator(mode="after")
    def _total_fits(self) -> IdmOrgProfileV1:
        if org_profile_char_count(self) > ORG_PROFILE_MAX_CHARS:
            raise ValueError("ORG_PROFILE_TOTAL_TOO_LONG")
        return self


def idm_profile_view(profile: IdmOrgProfileV1) -> IdmOrgProfileV1:
    """The profile as the IDM prompts and checks read it: without ``learner_groups`` (owner 2026-10-22, contract K3:
    the learners of a course come only from its own audience, "Đối tượng học", even when an older confirmed profile
    still lists groups). The field stays in profile v1 and in the frozen context, which the design echoes unchanged."""

    return profile.model_copy(update={"learner_groups": []}) if profile.learner_groups else profile


def org_profile_char_count(profile: IdmOrgProfileV1) -> int:
    """``char_count`` of spec §2: the lengths of every string (names, descriptions, terms, meanings, items) added up."""

    return (len(profile.organization_summary) + len(profile.culture_tone)
            + sum(len(item) for item in [*profile.products_services, *profile.key_processes,
                                         *profile.design_conventions])
            + sum(len(group.name) + len(group.description) for group in profile.learner_groups)
            + sum(len(entry.term) + len(entry.meaning) for entry in profile.glossary))


class IdmOrganizationContextV1(IdmOrgModel):
    """The confirmed profile version a run froze at launch: exactly the workspace's ``organization_context`` column.
    ``profile_hash`` is Node's ``canonicalJson`` hash of ``profile``; a profile that does not match it is a contract
    error."""

    profile_version: int = Field(ge=1, le=_MAX_VERSION)
    profile_hash: _Sha256
    profile: IdmOrgProfileV1

    @model_validator(mode="after")
    def _hash_matches(self) -> IdmOrganizationContextV1:
        if org_profile_hash(self.profile) != self.profile_hash:
            raise ValueError("ORG_CONTEXT_PROFILE_HASH_MISMATCH")
        return self


def org_profile_hash(profile: IdmOrgProfileV1) -> str:
    """sha256 hex of Node's ``canonicalJson(profile)`` (``app.hashing.js_canonical_hash``)."""

    return js_canonical_hash(profile.model_dump(mode="json"))
