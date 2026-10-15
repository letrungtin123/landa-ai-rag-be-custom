"""Strict data contracts of the IDM pipeline (spec §6, §11.2).

Every model forbids extra keys, validates strictly and strips string whitespace
before length checks, so a blank string is always rejected unless the field is
nullable. The TypeScript mirror lives in
the backend module ``lesson-author-idm.contract.ts``; both sides
hash the JSON form of these models with the same canonical encoding.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler, StringConstraints, model_serializer
from pydantic.json_schema import SkipJsonSchema

from app.lesson_author_orchestration_v2 import (
    ArchitectureComponentAuthorReviewV2,
    ArchitectureMediaBriefV2,
    canonical_hash,
)
from app.lesson_author_orchestration_v2_provider import SourceSnapshotFactV2

# --- Enums (spec §6.1) -------------------------------------------------------------------
Intent = Literal["know", "do", "decide"]
SupportRole = Literal["example", "common_mistake", "checklist", "job_aid", "practice_material"]
ContentKind = Literal["concept", "procedure", "skill", "decision", "policy", "mindset", "system", "case", "reference"]
IssueType = Literal["unclear", "duplicate", "conflict", "too_general", "too_detailed", "outdated"]
GapType = Literal[
    "missing_example", "missing_exception", "missing_criteria", "missing_step",
    "missing_sample_output", "missing_common_mistake", "missing_feedback_basis", "missing_unit_variation",
]
NoiseReason = Literal["page_furniture", "contact_or_promo", "toc_or_title_only", "duplicate_verbatim", "unreadable"]
LoRelation = Literal["direct", "supporting", "context", "unrelated", "unknown"]
Bloom = Literal["remember", "understand", "apply", "analyze", "evaluate", "create"]
Classification = Literal["must_do", "must_know", "reference", "nice_to_know", "remove"]
Placement = Literal["course", "reference_job_aid", "excluded"]
# "explain" (QLT-5 C): plain language + why it matters + an example, for a course explained to its learners.
Treatment = Literal[
    "keep", "condense", "rewrite", "explain", "combine", "separate", "move_to_reference", "convert_to_job_aid",
    "remove",
]
Disposition = Literal["course", "reference_job_aid", "nice_to_know", "remove", "hold", "noise"]
SupportKind = Literal[
    "explain_concept", "explain_principle", "example", "non_example", "worked_example", "demonstration",
]
ComponentRole = Literal["explain", "show", "practice", "clarify", "summary", "job_aid"]
UnitSegment = Literal["context_explain", "example", "practice_feedback", "summary_apply", "job_aid"]
ScenarioOrigin = Literal["source", "ai_drafted", "none"]
SignalFlag = Literal["H", "T", "S", "R", "E", "M", "D", "V"]
Origin = Literal["client", "ai_proposed"]
# Who set the course audience: the request ("client"), the author when starting AI ID ("author", QLT-6) or the AI.
AudienceOrigin = Literal["client", "ai_proposed", "author"]
StageOrigin = Literal["provider", "partial_fallback", "deterministic_fallback"]
IdmComponentType = Literal["html", "problem", "la_faq", "la_sortable", "la_crossword", "la_diagram"]
# Keys of the stored quality summary (``IdmUnitQualityV1.criteria``; Node mirror ``IDM_JUDGE_CRITERIA``).
JudgeCriterion = Literal[
    "Q1_support_sufficient", "Q2_not_copied", "Q3_practice_complete", "Q4_feedback_teaches",
    "Q5_grounded_criteria", "Q6_alignment", "Q7_cognitive_load", "Q8_language", "Q9_traceability",
]
JudgeSeverity = Literal["pass", "minor", "major", "critical"]
# What the judge answers (provider response only, never stored as such; QC course 364564, N7): one more
# criterion, summarised under Q9_traceability, and "not_applicable" for the practice criteria of a unit
# without a practice slot, which never counts as a finding.
JudgeAnswerCriterion = Literal[
    "Q1_support_sufficient", "Q2_not_copied", "Q3_practice_complete", "Q4_feedback_teaches",
    "Q5_grounded_criteria", "Q6_alignment", "Q7_cognitive_load", "Q8_language", "Q9_traceability",
    "Q10_title_matches", "Q11_purpose_alignment", "Q12_explained", "Q13_faq_value",
]
JudgeAnswerSeverity = Literal["pass", "minor", "major", "critical", "not_applicable"]
Locale = Literal["vi", "en"]
# Course purpose profiles (QLT-5 B, ``app.idm.purpose``): internal, inferred, never chosen by the author.
PurposeProfile = Literal["onboarding", "product", "compliance", "sales", "soft_skills", "general"]
PurposeConfidence = Literal["high", "medium", "low"]
PurposeOrigin = Literal["author_intent", "ai_inferred", "deterministic_fallback"]
# "purpose" (QLT-6): the course purpose the author entered when starting AI ID.
PurposeEvidence = Literal["title", "description", "source", "purpose"]
# How the author started AI ID (QLT-6): the full IDM design, or the document kept word for word. Absent (None) on runs
# started before QLT-6, which are designed as before.
DesignMode = Literal["instructional_design", "preserve_source"]
# How closely learner text follows the source wording (QLT-5 C, ``app.idm.fidelity``): derived from the purpose
# profile (and a clear wording request in the author's description), never chosen by authors.
FidelityMode = Literal["keep", "structured", "explain"]
# The question an FAQ item answers and where its point comes from (QLT-5 D, ``app.idm.faq_plan``): server-planned at
# W4 from the unit's evidence, never chosen by authors. "recall" is allowed at most once per FAQ.
FaqQuestionType = Literal["why", "what_if", "common_mistake", "how_to_apply", "comparison", "recall"]
FaqItemOrigin = Literal["source", "distractor", "ai_drafted"]

# --- Shared constrained strings ------------------------------------------------------------
FactKey = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)]
BlockId = Annotated[str, StringConstraints(pattern=r"^cb_[0-9]{4}$")]
SectionId = Annotated[str, StringConstraints(pattern=r"^sec_[0-9]{3}$")]
LoId = Annotated[str, StringConstraints(pattern=r"^lo_[1-9][0-9]?$")]
MustDoId = Annotated[str, StringConstraints(pattern=r"^md_[1-9][0-9]?$")]
LessonKey = Annotated[str, StringConstraints(pattern=r"^lsn_[0-9]{3}$")]
ModuleKey = Annotated[str, StringConstraints(pattern=r"^mod_[0-9]{2}$")]
ScopeKey = Annotated[str, StringConstraints(pattern=r"^idmcb_[0-9a-f]{32}$")]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
ComponentPlanId = Annotated[str, StringConstraints(pattern=r"^cp2_[a-f0-9]{32}$")]
SmeQuestion = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=400)]
PatternSectionKey = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z_]{1,39}$")]
PurposeSignal = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=40)]
Line300 = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=300)]
Line500 = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
Line280 = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=280)]


class IdmModel(BaseModel):
    """Base for every IDM contract: strict, closed and whitespace-normalised."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


# --- W0 / W1 (spec §6.2) ------------------------------------------------------------------
class IdmSourceDocumentRefV1(IdmModel):
    document_id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=255)
    type: str | None = Field(default=None, max_length=40)


class IdmProjectContextV1(IdmModel):
    """``course_title_hint`` and ``course_description`` are the author's intent (the CMS course name and
    description, sanitised by Node). They are data for the prompts, never instructions (QLT-5 A); a request
    of contract 1 has no description and reads as ``None``. Contract 3 (QLT-6) adds the launch brief
    (``app.idm.author_brief``): ``design_mode`` and, for a designed course, the ``learner_audience`` and
    ``course_purpose`` the author entered when starting AI ID; a request of contract 1 or 2 reads them as ``None``."""

    locale: Locale
    course_title_hint: str | None = Field(default=None, max_length=500)
    course_description: str | None = Field(default=None, max_length=2000)
    design_mode: DesignMode | None = None
    learner_audience: str | None = Field(default=None, max_length=300)
    course_purpose: str | None = Field(default=None, max_length=500)
    source_documents: list[IdmSourceDocumentRefV1] = Field(min_length=1, max_length=20)
    target_audience: str | None = Field(default=None, max_length=2000)
    learning_objectives: list[Line500] = Field(default_factory=list, max_length=8)
    duration_target_minutes: int | None = Field(default=None, ge=5, le=600)


class IdmFactSignalV1(IdmModel):
    fact_key: FactKey
    flags: list[SignalFlag] = Field(max_length=8)


class IdmIssueV1(IdmModel):
    type: IssueType
    note: str = Field(min_length=1, max_length=500)
    fact_keys: list[FactKey] = Field(default_factory=list, max_length=32)


class IdmGapV1(IdmModel):
    type: GapType
    note: str = Field(min_length=1, max_length=500)


class IdmW1BlockDraftV1(IdmModel):
    """One Content Block as proposed by the W1-map provider call."""

    local_id: str = Field(pattern=r"^b[1-9][0-9]{0,2}$")
    name: str = Field(min_length=3, max_length=180)
    summary: str = Field(min_length=1, max_length=300)
    intent: Intent
    support_role: SupportRole | None
    content_kind: ContentKind
    fact_keys: list[FactKey] = Field(min_length=1, max_length=400)
    issues: list[IdmIssueV1] = Field(default_factory=list, max_length=8)
    gaps: list[IdmGapV1] = Field(default_factory=list, max_length=8)
    sme_questions: list[SmeQuestion] = Field(default_factory=list, max_length=5)


class IdmNoiseDraftV1(IdmModel):
    fact_key: FactKey
    reason: NoiseReason


class IdmW1SectionResponseV1(IdmModel):
    blocks: list[IdmW1BlockDraftV1] = Field(min_length=1, max_length=40)
    noise: list[IdmNoiseDraftV1] = Field(default_factory=list, max_length=400)


class IdmContentBlockV1(IdmModel):
    """A Content Block after server re-keying (document-order ``cb_NNNN`` ids)."""

    block_id: BlockId
    section_id: SectionId
    name: str = Field(min_length=1, max_length=180)
    summary: str = Field(min_length=1, max_length=300)
    intent: Intent
    support_role: SupportRole | None
    content_kind: ContentKind
    fact_keys: list[FactKey] = Field(min_length=1)
    issues: list[IdmIssueV1] = Field(max_length=16)
    gaps: list[IdmGapV1] = Field(max_length=8)
    sme_questions: list[SmeQuestion] = Field(max_length=8)
    origin: Literal["provider", "deterministic_fallback"]


# --- W1-reduce + W0 (spec §6.3) ------------------------------------------------------------
class IdmLearningObjectiveV1(IdmModel):
    lo_id: LoId
    statement: str = Field(min_length=10, max_length=500)
    bloom: Bloom
    origin: Origin


class IdmMustDoV1(IdmModel):
    must_do_id: MustDoId
    lo_id: LoId
    statement: str = Field(min_length=5, max_length=280)
    kind: Literal["do", "decide"]
    bloom: Bloom


class IdmBlockLoLinkV1(IdmModel):
    block_id: BlockId
    lo_id: LoId
    relation: LoRelation


class IdmMergeV1(IdmModel):
    keep_block_id: BlockId
    merged_block_ids: list[BlockId] = Field(min_length=1, max_length=12)
    reason: str = Field(min_length=1, max_length=300)


class IdmConflictV1(IdmModel):
    block_ids: list[BlockId] = Field(min_length=1, max_length=8)
    note: str = Field(min_length=1, max_length=500)
    sme_question: str = Field(min_length=5, max_length=400)


class IdmAudienceV1(IdmModel):
    description: str = Field(min_length=10, max_length=2000)
    origin: AudienceOrigin


class IdmW1ReduceResponseV1(IdmModel):
    """The purpose fields are required on the provider wire but may be null: a missing or unusable answer falls
    back to the server classifier (``app.idm.purpose.resolve_purpose``) instead of rejecting the whole stage.
    Their bounds are generous for the same reason; the server trims the summary."""

    merges: list[IdmMergeV1] = Field(default_factory=list, max_length=100)
    conflicts: list[IdmConflictV1] = Field(default_factory=list, max_length=50)
    purpose_profile: PurposeProfile | None
    purpose_summary: str | None = Field(max_length=2000)
    purpose_evidence: list[PurposeEvidence] = Field(max_length=8)
    target_audience: IdmAudienceV1
    learning_objectives: list[IdmLearningObjectiveV1] = Field(min_length=1, max_length=8)
    must_dos: list[IdmMustDoV1] = Field(min_length=1, max_length=40)
    lo_links: list[IdmBlockLoLinkV1] = Field(max_length=2000)


# --- W2 (spec §6.4) --------------------------------------------------------------------------
class IdmSeparatePartV1(IdmModel):
    name: str = Field(min_length=3, max_length=180)
    intent: Intent
    fact_keys: list[FactKey] = Field(min_length=1, max_length=400)


class IdmBlueprintRowV1(IdmModel):
    block_id: BlockId
    lo_id: LoId | None
    must_do_ids: list[MustDoId] = Field(default_factory=list, max_length=8)
    classification: Classification
    placement: Placement
    treatment: Treatment
    detail_level: str = Field(min_length=1, max_length=300)
    hold: bool
    hold_reason: str | None = Field(default=None, max_length=300)
    sme_question: str | None = Field(default=None, max_length=400)
    combine_into: BlockId | None = None
    separate_into: list[IdmSeparatePartV1] = Field(default_factory=list, max_length=6)
    rationale: str = Field(min_length=1, max_length=300)


class IdmW2BlueprintResponseV1(IdmModel):
    rows: list[IdmBlueprintRowV1] = Field(min_length=1, max_length=400)
    blocked_must_do_ids: list[MustDoId] = Field(default_factory=list, max_length=40)


# --- W4 course (spec §6.5) -------------------------------------------------------------------
class IdmLessonPlanV1(IdmModel):
    lesson_key: LessonKey
    kind: Literal["learning", "job_aid"]
    title: str = Field(min_length=3, max_length=180)
    primary_must_do_id: MustDoId | None
    secondary_must_do_ids: list[MustDoId] = Field(default_factory=list, max_length=2)
    block_ids: list[BlockId] = Field(min_length=1, max_length=40)
    est_screens: int = Field(ge=2, le=30)
    est_minutes: int = Field(ge=2, le=60)
    ordering_rationale: str = Field(min_length=1, max_length=300)


class IdmModulePlanV1(IdmModel):
    module_key: ModuleKey
    title: str = Field(min_length=3, max_length=500)
    performance_goal: str = Field(min_length=10, max_length=2000)
    lo_ids: list[LoId] = Field(min_length=1, max_length=8)
    lessons: list[IdmLessonPlanV1] = Field(min_length=1, max_length=30)


class IdmPatternSectionMapV1(IdmModel):
    """Which lessons teach one part of the purpose's course pattern. Keys are checked by the server (an unknown
    key or lesson is ignored with a warning), so they are plain bounded strings here."""

    section_key: str = Field(min_length=1, max_length=60)
    lesson_keys: list[Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=16)]] = (
        Field(max_length=120))


class IdmW4CourseResponseV1(IdmModel):
    course_title: str = Field(min_length=3, max_length=500)
    course_summary: str = Field(min_length=20, max_length=2000)
    assessment_strategy: str = Field(min_length=10, max_length=2000)
    prerequisites: list[Line300] = Field(default_factory=list, max_length=10)
    modules: list[IdmModulePlanV1] = Field(min_length=1, max_length=24)
    pattern_sections: list[IdmPatternSectionMapV1] = Field(max_length=24)


# --- Course design (spec §6.6) ---------------------------------------------------------------
class IdmDispositionV1(IdmModel):
    fact_key: FactKey
    disposition: Disposition
    block_id: BlockId | None
    reason: str | None = Field(default=None, max_length=300)


class IdmBlockScopeV1(IdmModel):
    scope_key: ScopeKey
    block_id: BlockId
    title: str = Field(min_length=1, max_length=500)
    source_ref: str | None = Field(default=None, max_length=255)
    fact_count: int = Field(ge=1)
    content_chars: int = Field(ge=1)
    fact_keys: list[FactKey] = Field(min_length=1)


class IdmHoldItemV1(IdmModel):
    block_id: BlockId
    name: str = Field(min_length=1, max_length=180)
    reason: str = Field(min_length=1, max_length=300)
    sme_question: str = Field(min_length=1, max_length=400)
    blocked_must_do_ids: list[MustDoId] = Field(max_length=40)


class IdmAuthorNotesV1(IdmModel):
    course: str = Field(min_length=1, max_length=7000)
    modules: dict[str, Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=3000)]]
    lessons: dict[str, Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]]


class IdmTemplateGapV1(IdmModel):
    """A required part of the purpose's course pattern the source has no content for (author-facing note)."""

    section_key: PatternSectionKey
    note: str = Field(min_length=1, max_length=300)


class IdmPurposeV1(IdmModel):
    """The course purpose the pipeline designed for (QLT-5 B). ``evidence`` names the inputs that decided it,
    ``signals`` the server keywords found in the author's course name/description. ``fidelity_mode`` (QLT-5 C) is
    the content mode every later stage follows; a purpose stored without it reads as "structured" (today's rules)."""

    profile: PurposeProfile
    summary: str = Field(min_length=10, max_length=500)
    confidence: PurposeConfidence
    origin: PurposeOrigin
    evidence: list[PurposeEvidence] = Field(max_length=4)
    signals: list[PurposeSignal] = Field(max_length=8)
    template_gaps: list[IdmTemplateGapV1] = Field(max_length=12)
    fidelity_mode: FidelityMode | None = None


class IdmCourseDesignV1(IdmModel):
    """Contract 2 adds ``purpose`` and ``project_context.course_description``; contract 3 (QLT-6) the launch brief in
    ``project_context`` and the "author" audience origin. Node still reads stored designs of contracts 1 and 2
    (``lesson-author-idm.contract.ts``)."""

    pipeline_version: Literal["idm-1"]
    idm_contract_version: Literal[3]
    prompt_policy_version: Literal["idm-prompt-1"]
    source_snapshot_hash: Sha256
    project_context: IdmProjectContextV1
    purpose: IdmPurposeV1
    target_audience: IdmAudienceV1
    learning_objectives: list[IdmLearningObjectiveV1] = Field(min_length=1, max_length=8)
    must_dos: list[IdmMustDoV1] = Field(min_length=1, max_length=40)
    blocks: list[IdmContentBlockV1] = Field(min_length=1, max_length=2000)
    lo_links: list[IdmBlockLoLinkV1] = Field(max_length=16000)
    blueprint: list[IdmBlueprintRowV1] = Field(min_length=1, max_length=2000)
    blocked_must_do_ids: list[MustDoId] = Field(max_length=40)
    hold_items: list[IdmHoldItemV1] = Field(max_length=2000)
    modules: list[IdmModulePlanV1] = Field(min_length=1, max_length=24)
    block_scopes: list[IdmBlockScopeV1] = Field(min_length=1, max_length=2000)
    dispositions: list[IdmDispositionV1] = Field(min_length=1)
    notes: IdmAuthorNotesV1
    stage_origins: dict[str, StageOrigin]
    design_hash: Sha256


# --- W3/W4 module (spec §6.7) -----------------------------------------------------------------
class IdmFeedbackFocusV1(IdmModel):
    criterion: str = Field(min_length=5, max_length=400)
    rationale: str = Field(min_length=5, max_length=400)
    improvement: str = Field(min_length=5, max_length=400)


class IdmPracticeTaskV1(IdmModel):
    practice_id: str = Field(pattern=r"^pt_[1-9]$")
    sentence: str = Field(min_length=10, max_length=280)
    context_input: str = Field(min_length=3, max_length=300)
    learner_action: str = Field(min_length=3, max_length=200)
    result: str = Field(min_length=3, max_length=200)
    bloom: Bloom
    criteria_fact_keys: list[FactKey] = Field(default_factory=list, max_length=24)
    scenario_origin: ScenarioOrigin
    hold: bool
    hold_question: str | None = Field(default=None, max_length=400)
    feedback_focus: IdmFeedbackFocusV1


class IdmSupportItemV1(IdmModel):
    kind: SupportKind
    brief: str = Field(min_length=3, max_length=300)
    block_id: BlockId


class IdmFaqItemPlanV1(IdmModel):
    """One planned FAQ item (QLT-5 D): the question type, the facts its answer is built on and where its point comes
    from (a mistake or condition the source states, the misconception behind a practice question's wrong option, or a
    confusion inferred from the facts)."""

    question_type: FaqQuestionType
    anchor_fact_keys: list[FactKey] = Field(min_length=1, max_length=6)
    origin: FaqItemOrigin


def _without_empty_faq_items(data: Any) -> Any:
    """A slot without an FAQ plan serialises as before QLT-5 D (no ``faq_items`` key), so the hashes of designs and
    briefs that hold no FAQ plan do not change."""

    if isinstance(data, dict) and not data.get("faq_items"):
        data.pop("faq_items", None)
    return data


# ``faq_items`` (QLT-5 D) is the server's plan of an la_faq slot (``app.idm.faq_plan``): left out of the provider schema
# (the W4 writer never answers it) and out of the JSON form when empty. A comment, not a docstring: the docstring is
# the schema description the provider reads.
class IdmComponentDesignV1(IdmModel):
    component_index: int = Field(ge=1, le=4)
    type: IdmComponentType
    role: ComponentRole
    title: str = Field(min_length=3, max_length=180)
    rationale: str = Field(min_length=3, max_length=500)
    block_ids: list[BlockId] = Field(min_length=1, max_length=12)
    practice_id: str | None = Field(default=None, pattern=r"^pt_[1-9]$")
    support_items: list[IdmSupportItemV1] = Field(default_factory=list, max_length=6)
    author_review: ArchitectureComponentAuthorReviewV2
    faq_items: SkipJsonSchema[list[IdmFaqItemPlanV1]] = Field(default_factory=list, max_length=6)

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> Any:
        return _without_empty_faq_items(handler(self))


class IdmUnitDesignV1(IdmModel):
    unit_index: int = Field(ge=1, le=12)
    segment: UnitSegment
    title: str = Field(min_length=3, max_length=180)
    purpose: str = Field(min_length=5, max_length=500)
    block_ids: list[BlockId] = Field(min_length=1, max_length=24)
    components: list[IdmComponentDesignV1] = Field(min_length=1, max_length=4)
    media_brief: ArchitectureMediaBriefV2 | None


class IdmLessonDesignV1(IdmModel):
    lesson_key: LessonKey
    title: str = Field(min_length=3, max_length=180)
    objective: str = Field(min_length=5, max_length=500)
    learning_objectives: list[Line500] = Field(min_length=1, max_length=8)
    practice_tasks: list[IdmPracticeTaskV1] = Field(default_factory=list, max_length=3)
    assessment: str = Field(min_length=5, max_length=500)
    units: list[IdmUnitDesignV1] = Field(min_length=1, max_length=12)
    notes: str = Field(min_length=1, max_length=2000)


class IdmW3W4ModuleResponseV1(IdmModel):
    lessons: list[IdmLessonDesignV1] = Field(min_length=1, max_length=30)


class IdmShardDesignV1(IdmModel):
    pipeline_version: Literal["idm-1"]
    chapter_key: str = Field(min_length=1, max_length=160)
    shard_index: int = Field(ge=0, le=4095)
    lessons: list[IdmLessonDesignV1] = Field(min_length=1, max_length=30)
    lesson_index_offset: int = Field(ge=0, le=4095)
    stage_origin: StageOrigin
    design_hash: Sha256


# --- W5 brief and W6 judge (spec §6.8) -------------------------------------------------------
class IdmTreatmentRefV1(IdmModel):
    block_id: BlockId
    treatment: Treatment
    detail_level: str = Field(min_length=1, max_length=300)


class IdmContextFactV1(IdmModel):
    fact_key: FactKey
    fact_text: str = Field(min_length=1, max_length=32768)


class IdmBriefComponentV1(IdmModel):
    """``faq_items``: the W4 plan of an la_faq slot (QLT-5 D); absent for every other slot and in older briefs."""

    component_plan_id: ComponentPlanId
    type: IdmComponentType
    role: ComponentRole
    title: str = Field(min_length=1, max_length=180)
    support_items: list[IdmSupportItemV1] = Field(max_length=6)
    practice: IdmPracticeTaskV1 | None
    treatments: list[IdmTreatmentRefV1] = Field(max_length=24)
    owned_fact_keys: list[FactKey] = Field(max_length=32768)
    supporting_fact_keys: list[FactKey] = Field(max_length=32768)
    faq_items: list[IdmFaqItemPlanV1] = Field(default_factory=list, max_length=6)

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> Any:
        return _without_empty_faq_items(handler(self))


class IdmFaqBlockNoteV1(IdmModel):
    """What W1 found about one block of a unit with an FAQ (QLT-5 D): its role, the source issues and gaps and the
    open SME questions. Raw material for real confusions; never facts."""

    block_id: BlockId
    name: str = Field(min_length=1, max_length=180)
    intent: Intent
    content_kind: ContentKind
    support_role: SupportRole | None
    issues: list[IdmIssueV1] = Field(max_length=16)
    gaps: list[IdmGapV1] = Field(max_length=8)
    sme_questions: list[SmeQuestion] = Field(max_length=8)


class IdmFaqPracticeNoteV1(IdmModel):
    """A practice of the unit's lesson: what decides right and wrong, why the better choice is better and what to fix
    (the "why wrong" basis of the misconceptions behind its wrong options)."""

    practice_id: str = Field(pattern=r"^pt_[1-9]$")
    sentence: str = Field(min_length=10, max_length=280)
    feedback_focus: IdmFeedbackFocusV1


class IdmFaqMaterialV1(IdmModel):
    """``open_questions`` (QLT-7, contract 3): the SME questions of the lesson's Hold items and held practices, never to
    be answered by the FAQ; absent when there is none, so a brief without them reads and hashes as before."""

    blocks: list[IdmFaqBlockNoteV1] = Field(max_length=24)
    practices: list[IdmFaqPracticeNoteV1] = Field(max_length=3)
    open_questions: list[SmeQuestion] = Field(default_factory=list, max_length=8)

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> Any:
        data = handler(self)
        if isinstance(data, dict) and not data.get("open_questions"):
            data.pop("open_questions", None)
        return data


class IdmUnitBriefV1(IdmModel):
    """``purpose_profile``/``purpose_summary`` keep the writer and the judge on the course purpose; a brief built
    from a contract-1 design (no purpose) leaves them out or null. ``fidelity_mode`` is the design purpose's content
    mode (QLT-5 C); absent or null reads as "structured", today's behaviour. ``faq_material`` (QLT-5 D) is set only for
    a unit with an la_faq slot. ``design_mode`` (contract 3, QLT-6) is how the author started AI ID; absent or null in
    a brief built from an older design."""

    pipeline_version: Literal["idm-1"]
    course_title: str = Field(min_length=1, max_length=500)
    target_audience: str = Field(min_length=1, max_length=2000)
    purpose_profile: PurposeProfile | None = None
    purpose_summary: str | None = Field(default=None, max_length=500)
    fidelity_mode: FidelityMode | None = None
    design_mode: DesignMode | None = None
    module_title: str = Field(min_length=1, max_length=500)
    lesson_title: str = Field(min_length=1, max_length=180)
    lesson_objective: str = Field(min_length=1, max_length=500)
    lesson_practice_sentences: list[Line280] = Field(max_length=3)
    previous_lesson_title: str | None = Field(max_length=180)
    next_lesson_title: str | None = Field(max_length=180)
    unit_segment: UnitSegment
    unit_purpose: str = Field(min_length=1, max_length=500)
    components: list[IdmBriefComponentV1] = Field(min_length=1, max_length=4)
    lesson_context_facts: list[IdmContextFactV1] = Field(default_factory=list, max_length=80)
    job_aid_signpost: str | None = Field(default=None, max_length=300)
    faq_material: IdmFaqMaterialV1 | None = None
    brief_hash: Sha256


class IdmJudgeFindingV1(IdmModel):
    criterion: JudgeAnswerCriterion
    severity: JudgeAnswerSeverity
    component_index: int | None = Field(default=None, ge=0, le=3)
    witness: str = Field(max_length=300)


class IdmJudgeResponseV1(IdmModel):
    verdict: Literal["pass", "review_required", "reject"]
    findings: list[IdmJudgeFindingV1] = Field(max_length=36)


class IdmFindingCountsV1(IdmModel):
    minor: int = Field(ge=0, le=1000)
    major: int = Field(ge=0, le=1000)
    critical: int = Field(ge=0, le=1000)


class IdmUnitQualityV1(IdmModel):
    judge_mode: Literal["off", "observe", "repair"]
    judge_status: Literal["not_run", "pass", "review_required", "reject", "skipped_budget", "failed"]
    finding_counts: IdmFindingCountsV1
    criteria: dict[JudgeCriterion, JudgeSeverity]
    repair_applied: bool
    deterministic_codes: list[Annotated[str, StringConstraints(pattern=r"^[A-Z][A-Z0-9_]{0,95}$")]] = Field(
        max_length=32,
    )
    author_note: str = Field(max_length=1500)


# --- Requests (spec §11.2) -----------------------------------------------------------------
class IdmTokenAllowanceV1(IdmModel):
    input_tokens: int = Field(ge=1, le=2_000_000)
    output_tokens: int = Field(ge=1, le=131_072)


class IdmCourseSkeletonRequestV1(IdmModel):
    pipeline_version: Literal["idm-1"]
    project_context: IdmProjectContextV1
    source_facts: list[SourceSnapshotFactV2] = Field(min_length=1, max_length=20_000)
    token_allowance: IdmTokenAllowanceV1
    remaining_budget_ms: int = Field(ge=30_000, le=600_000)


class IdmModuleContextV1(IdmModel):
    """``purpose`` is the design's purpose; null (or absent) for a design of contract 1."""

    pipeline_version: Literal["idm-1"]
    project_context: IdmProjectContextV1
    purpose: IdmPurposeV1 | None = None
    target_audience: IdmAudienceV1
    module: IdmModulePlanV1
    learning_objectives: list[IdmLearningObjectiveV1] = Field(min_length=1, max_length=8)
    must_dos: list[IdmMustDoV1] = Field(min_length=1, max_length=40)
    blocks: list[IdmContentBlockV1] = Field(min_length=1, max_length=400)
    blueprint: list[IdmBlueprintRowV1] = Field(min_length=1, max_length=400)
    block_scopes: list[IdmBlockScopeV1] = Field(min_length=1, max_length=400)
    lesson_index_offset: int = Field(ge=0, le=4095)
    allowed_component_types: list[IdmComponentType] = Field(min_length=1, max_length=8)
    token_allowance: IdmTokenAllowanceV1
    remaining_budget_ms: int = Field(ge=30_000, le=600_000)
    design_hash: Sha256


def design_hash_of(payload: dict[str, Any]) -> str:
    """Hash a design payload exactly as Node does: canonical JSON without ``design_hash``."""

    return canonical_hash({key: value for key, value in payload.items() if key != "design_hash"})


def brief_hash_of(payload: dict[str, Any]) -> str:
    """Hash a unit brief exactly as Node does: canonical JSON without ``brief_hash``."""

    return canonical_hash({key: value for key, value in payload.items() if key != "brief_hash"})
