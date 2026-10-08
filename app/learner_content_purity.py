from __future__ import annotations

import html
import re
import unicodedata
from typing import Any, Iterable

LEARNER_CONTENT_PURITY_POLICY_VERSION = "learner-content-purity-2"

_FILE_NAME_RE = re.compile(
    r"(?<![\w.-])[^\n<>]{0,160}?\.(?:pdf|pptx?|docx?|xlsx?|csv|txt|rtf)(?![\w.-])",
    re.IGNORECASE,
)
_SOURCE_LOCATOR_RE = re.compile(
    r"\b(?:trang|page|slide|chunk|đoạn\s+nguồn|doan\s+nguon|mục\s+nguồn|muc\s+nguon)"
    r"\s*(?:số|so|number|no\.?|#)?\s*[:#-]?\s*\d{1,6}\b",
    re.IGNORECASE,
)
_SOURCE_ATTRIBUTION_RE = re.compile(
    r"(?:\b(?:theo|dựa\s+trên|dua\s+tren|trích\s+từ|trich\s+tu)\s+"
    r"(?:tài\s+liệu|tai\s+lieu|nguồn|nguon|source|document)\b)"
    r"|(?:\b(?:trong|inside)\s+(?:tài\s+liệu\s+nguồn|tai\s+lieu\s+nguon|source\s+document)\b)"
    r"|(?:\b(?:tài\s+liệu|tai\s+lieu|document|source)\s+"
    r"(?:nêu|neu|mô\s+tả|mo\s+ta|states?|describes?)\b)"
    r"|(?:\b(?:nguồn|nguon|source|tài\s+liệu\s+nguồn|tai\s+lieu\s+nguon)\s*[:：])",
    re.IGNORECASE,
)
_GENERIC_INTERNAL_ID_RE = re.compile(
    r"(?<![\w-])(?:p\d+[-_]f\d+|src[-_]\d+(?:[-_]f\d+)?|"
    r"(?:component|block|fact|scope)[-_](?:[a-z0-9]+[-_]?){1,8})(?![\w-])",
    re.IGNORECASE,
)
_STANDALONE_LOCATOR_RE = re.compile(
    r"^(?:(?:trang|page|slide|chunk)\s*(?:số|so|number|no\.?|#)?\s*[:#-]?\s*)?"
    r"\d{1,6}(?:\s*/\s*\d{1,6})?$",
    re.IGNORECASE,
)
_SOURCE_LABEL_ONLY_RE = re.compile(
    r"^(?:nguồn|nguon|source|tài\s+liệu|tai\s+lieu|document)\s*[:：-]\s*.{0,220}$",
    re.IGNORECASE,
)
_ATTRIBUTION_ONLY_RE = re.compile(
    r"^(?:nội\s+dung|noi\s+dung|ý|y|thông\s+tin|thong\s+tin)(?:\s+này|\s+nay)?\s+"
    r"(?:được\s+|duoc\s+)?(?:nêu|neu|mô\s+tả|mo\s+ta)\s+"
    r"(?:trong|inside)\s+(?:tài\s+liệu\s+nguồn|tai\s+lieu\s+nguon|source\s+document)"
    r"[\s.!?]*$",
    re.IGNORECASE,
)
_LEADING_ATTRIBUTION_RE = re.compile(
    r"^(?P<label>(?:giải\s+thích|giai\s+thich|explanation|rationale)\s*[:：]\s*)?"
    r"(?:(?:theo|dựa\s+trên|dua\s+tren|trích\s+từ|trich\s+tu|according\s+to|based\s+on)\s+"
    r"(?:the\s+)?(?:tài\s+liệu(?:\s+nguồn)?|tai\s+lieu(?:\s+nguon)?|nguồn|nguon|source|document|file)"
    r"\s*[,;:：-]\s*|"
    r"(?:tài\s+liệu(?:\s+nguồn)?|tai\s+lieu(?:\s+nguon)?|document|source)\s+"
    r"(?:nêu|neu|mô\s+tả|mo\s+ta|yêu\s+cầu|yeu\s+cau|cho\s+biết|cho\s+biet|"
    r"states?|describes?|requires?)\s+(?:rằng\s+|rang\s+|that\s+)?)"
    r"(?P<body>.+)$",
    re.IGNORECASE,
)
_WRAPPED_SOURCE_FURNITURE_RE = re.compile(
    r"[\[(]\s*(?:(?:nguồn|nguon|source|tài\s+liệu|tai\s+lieu|document|file)\s*[:：-]?\s*)?"
    r"(?:[^\])]{0,160}?\.(?:pdf|pptx?|docx?|xlsx?|csv|txt|rtf)|"
    r"(?:trang|page|slide|chunk|đoạn\s+nguồn|doan\s+nguon|mục\s+nguồn|muc\s+nguon)"
    r"\s*(?:số|so|number|no\.?|#)?\s*[:#-]?\s*\d{1,6}"
    r"(?:\s*[,;/|]\s*(?:(?:trang|page|slide|chunk)\s*)?\d{1,6})*)\s*[\])]",
    re.IGNORECASE,
)
_DANGLING_PROVENANCE_ONLY_RE = re.compile(
    r"^(?:(?:xem|see|tham\s+khảo|tham\s+khao|refer(?:\s+to)?)\s+)?"
    r"(?:(?:nội\s+dung|noi\s+dung|content)\s+)?(?:tại|tai|ở|o|at|on|in)?\s*$",
    re.IGNORECASE,
)


def _visible_text(value: Any) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", " ", str(value or "")))
    return re.sub(r"\s+", " ", text).strip()


def _fold(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value.casefold().replace("đ", "d"))
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def _bounded_exact_match(text: str, candidate: str) -> bool:
    candidate = candidate.strip()
    if len(candidate) < 3:
        return False
    return re.search(rf"(?<![\w]){re.escape(candidate)}(?![\w])", text, re.IGNORECASE) is not None


def sanitize_source_fact_for_learner(value: Any) -> str:
    """Strip source furniture while preserving the supported proposition.

    Source extractors may join a page/slide label and its proposition into one
    line.  The purity gate must reject that label in provider output, while the
    deterministic recovery path must be able to remove it without discarding
    the useful proposition that follows it.
    """

    text = _visible_text(value)
    if not text or _ATTRIBUTION_ONLY_RE.fullmatch(text):
        return ""
    match = _LEADING_ATTRIBUTION_RE.match(text)
    if match:
        body = match.group("body").strip()
        if not body:
            return ""
        body = body[:1].upper() + body[1:]
        text = f"{match.group('label') or ''}{body}".strip()

    # Remove only explicit source furniture.  Ordinary numbers and the word
    # "trang" in phrases such as "trang thiết bị" are intentionally outside
    # these patterns and remain untouched.
    text = _WRAPPED_SOURCE_FURNITURE_RE.sub(" ", text)
    text = _SOURCE_LOCATOR_RE.sub(" ", text)
    text = _SOURCE_ATTRIBUTION_RE.sub(" ", text)
    text = re.sub(r"[\[(]\s*[\])]", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"(?:\s*[,;:|]\s*){2,}", "; ", text)
    text = re.sub(r"\s+(?:[-–—]\s*)$", "", text)
    text = re.sub(r"^(?:\s*[-–—,:;|]\s*)+|(?:\s*[-–—,:;|]\s*)+$", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text or _DANGLING_PROVENANCE_ONLY_RE.fullmatch(text.strip(" .,:;!?-–—")):
        return ""
    return text


def build_learner_content_purity_context(
    expected: dict[str, Any],
    manifest: dict[str, Any] | None,
    source_rows: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Build an internal-only provenance deny-list for one immutable unit.

    The context is deliberately not part of the provider response or UI model.
    It contains identifiers and filenames only; source fact text remains in the
    ordinary evidence contract and is never copied into diagnostics.
    """

    allowed_ids = {
        str(value).strip()
        for field in ("source_fact_ids", "supporting_evidence_fact_ids")
        for value in expected.get(field, [])
        if isinstance(value, str) and value.strip()
    }
    identifiers = set(allowed_ids)
    filenames: set[str] = set()
    for collection in ("facts", "supporting_evidence_facts"):
        for fact in (manifest or {}).get(collection, []):
            if not isinstance(fact, dict) or str(fact.get("fact_id") or "").strip() not in allowed_ids:
                continue
            for key in ("fact_id", "source_ref"):
                value = str(fact.get(key) or "").strip()
                if value:
                    identifiers.add(value)
            for key in ("file_name", "filename", "document_name", "source_name"):
                value = str(fact.get(key) or "").strip()
                if value:
                    filenames.add(value)
    for row in source_rows:
        if not isinstance(row, dict):
            continue
        for key in ("file_name", "filename", "document_name", "source_name", "name"):
            value = str(row.get(key) or "").strip()
            if value and _FILE_NAME_RE.search(value):
                filenames.add(value)
    return {
        "policy_version": LEARNER_CONTENT_PURITY_POLICY_VERSION,
        "exact_identifiers": sorted(identifiers),
        "source_filenames": sorted(filenames),
    }


def learner_content_purity_finding(
    value: Any,
    context: dict[str, Any] | None = None,
) -> str | None:
    """Return a stable, non-sensitive rejection code for learner-facing text."""

    text = _visible_text(value)
    if not text:
        return None
    folded = _fold(text)
    if _SOURCE_ATTRIBUTION_RE.search(text) or re.search(
        r"\b(?:according\s+to|based\s+on|quoted\s+from)\s+(?:the\s+)?(?:source|document|file)\b",
        folded,
    ):
        return "LEARNER_CONTENT_SOURCE_ATTRIBUTION"
    if _SOURCE_LOCATOR_RE.search(text):
        return "LEARNER_CONTENT_SOURCE_LOCATOR"
    if _FILE_NAME_RE.search(text):
        return "LEARNER_CONTENT_SOURCE_FILENAME"
    if _GENERIC_INTERNAL_ID_RE.search(text):
        return "LEARNER_CONTENT_INTERNAL_IDENTIFIER"
    for candidate in (context or {}).get("source_filenames", []):
        if isinstance(candidate, str) and _bounded_exact_match(text, candidate):
            return "LEARNER_CONTENT_SOURCE_FILENAME"
    for candidate in (context or {}).get("exact_identifiers", []):
        if isinstance(candidate, str) and _bounded_exact_match(text, candidate):
            return "LEARNER_CONTENT_INTERNAL_IDENTIFIER"
    return None


def is_provenance_only_source_line(value: Any) -> bool:
    """Reject page furniture/citation rows without deleting substantive facts."""

    text = _visible_text(value).strip(" .,:;|-–—")
    if not text:
        return True
    if _STANDALONE_LOCATOR_RE.fullmatch(text):
        return True
    if _FILE_NAME_RE.fullmatch(text):
        return True
    if _SOURCE_LABEL_ONLY_RE.fullmatch(text) and (
        _FILE_NAME_RE.search(text) or _SOURCE_LOCATOR_RE.search(text) or len(text.split()) <= 12
    ):
        return True
    return False


_LEARNER_KEYS_TO_SKIP = {
    "type", "component_plan_id", "source_fact_ids", "covered_source_fact_ids",
    "supporting_evidence_fact_ids", "learning_objective_refs", "metadata",
    "source_scope_ids", "source_ref", "fact_id", "id", "version",
    "selection_rationale", "rationale", "reason_code",
}


def component_learner_text(component: dict[str, Any]) -> str:
    """Project only visible payload fields, excluding provenance/identity."""

    values: list[str] = []

    def visit(value: Any, key: str | None = None) -> None:
        if key in _LEARNER_KEYS_TO_SKIP:
            return
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                visit(child, key)
        elif isinstance(value, str):
            values.append(_visible_text(value))

    visit(component)
    return " ".join(value for value in values if value)
