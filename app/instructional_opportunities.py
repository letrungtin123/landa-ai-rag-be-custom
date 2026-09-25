"""Bounded, conservative treatment discovery from canonical evidence.

No CMS selection, model call, fact allocation or new source claim lives here.
Node's registry/tenant policy remains the component acceptance authority.
"""
from copy import deepcopy
import hashlib
import re
import unicodedata
from typing import Any

VERSION = "evidence-treatment-2"
MAX_SCOPE_FACTS = 128
MAX_SCOPE_CHARS = 64_000
# Reserve a clarification opportunity first; serialization still puts FAQ last.
# Ordering practice is more useful than a second rendering of the same flow.
TREATMENTS = ("faq", "practice", "relationship_visualization", "terminology_reinforcement")


def _fold(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text.lower().replace("đ", "d"))
                   if not unicodedata.combining(c))


def _is_instructional_text(text: str) -> bool:
    """Discovery eligibility only; never remove facts from the canonical manifest."""
    folded = _fold(text.strip())
    if re.search(r"https?://|www\.|[\w.+-]+@[\w.-]+", folded):
        return False
    if re.match(r"^(?:thank you|cam on|contact|lien he|hotline|email|copyright|all rights reserved)\b", folded):
        return False
    return len(re.findall(r"\w+", folded)) >= 6


def _structured_signals(texts: list[str]) -> dict[str, dict[str, Any]]:
    """Recover pedagogical structure within ONE page/section, never across scopes.

    Fact extraction may strip list numbers. An explicit N-step heading followed
    by N substantive items is still evidence of order; N arbitrary adjacent
    facts without that heading are not. Labels are not emitted as model-owned
    facts or included in diagnostics.
    """
    lines = [_fold(line.strip().lstrip("•*- ")) for raw in texts for line in raw.splitlines() if line.strip()]
    sequence_count = 0
    model_count = 0
    for index, line in enumerate(lines):
        counted = re.fullmatch(r"(?:quy trinh|procedure|process)\s+(\d{1,2})\s*(?:buoc|steps?)\s*:?[ .]*", line)
        if counted and 3 <= int(counted[1]) <= 10:
            count = int(counted[1])
            items = lines[index + 1:index + 1 + count]
            if len(items) == count and len(set(items)) == count and all(len(re.findall(r"\w+", x)) >= 4 for x in items):
                sequence_count = max(sequence_count, count)
        # Numbered lists need a procedure/step heading in the same local group.
        if re.search(r"\b(?:quy trinh|procedure|process|steps|trinh tu)\b", line):
            numbers = []
            for item in lines[index + 1:]:
                match = re.match(r"^(\d{1,2})[.)]\s+\S", item)
                if not match:
                    break
                numbers.append(int(match[1]))
            if (3 <= len(numbers) <= 10 and numbers == list(range(1, len(numbers) + 1))
                    and (not counted or int(counted[1]) == len(numbers))):
                sequence_count = max(sequence_count, len(numbers))
        # A source-spelled action mnemonic, e.g. four labelled actions whose
        # initials spell the source rule. A mere acronym glossary is not order.
        mnemonic = re.match(r"^(?:quy tac|rule|protocol)\s+([a-z]{3,10})\b", line)
        if mnemonic and re.search(r"\b(?:dung|su dung|thuc hien|using|use|perform)\b", line):
            actions = re.findall(r"(?:^|[;\n])\s*([a-z])\s*[–-]\s*[^;\n]+", "\n".join(lines[index + 1:index + 4]))
            if "".join(actions) == mnemonic[1]:
                sequence_count = max(sequence_count, len(actions))
    # Explicit named model plus its source-listed members. This licenses only
    # membership, NOT causal arrows or inferred temporal order.
    model_headers = [x for x in lines if re.match(r"^(?:mo hinh|model|cac nhom|types of|categories of)\b", x)]
    if model_headers:
        members = {x for x in lines if x not in model_headers and 3 <= len(x.split()) <= 22}
        if 3 <= len(members) <= 20:
            model_count = len(members)
    result: dict[str, dict[str, Any]] = {}
    if sequence_count:
        result["practice"] = {"requires_ordering_practice": True, "ordered_sequence": True, "sequence_item_count": sequence_count}
    if model_count:
        result["relationship_visualization"] = {"relationship_evidence": True, "relationship_count": model_count}
    return result


def _signals(texts: list[str]) -> dict[str, dict[str, Any]]:
    """Only explicit definitions, numbered steps, relationships, conditions.

    An arbitrary list is not an ordered procedure. A count/boolean from the
    Architect is not evidence. Descriptors contain no private source prose.
    """
    terms: set[str] = set()
    steps: list[int] = []
    clarifications: set[str] = set()
    relationship_count = 0
    for raw in texts:
        raw = raw.strip().lstrip("•*- ")
        text = _fold(raw.strip())
        definition = re.match(r"^([\w -]{2,60}?)\s+(?:la|nghia la|means|is defined as|refers to)\s+\S", text)
        acronym = re.match(r"^([A-Z]{2,12})\s*[:：]\s*\S.{8,}", raw.strip())
        # Source-defined label/description tables and acronym expansions are
        # usable vocabulary too, not only the literal phrase "X means Y".
        labelled = re.match(r"^([\w -]{2,45}):\s+(.+)$", text)
        expansion = re.match(r"^([A-Z]{2,12})\s*\(([^()]{8,100})\)", raw)
        label = definition.group(1) if definition else acronym.group(1).lower() if acronym else expansion.group(1).lower() if expansion else ""
        if not label and labelled and _is_instructional_text(labelled[2]) and not re.search(
            r"\d|\b(?:buoc|step|muc tieu|objective|vi du|example|chu y|note|cau hoi|question|tinh huong|scenario|quy trinh|procedure|lien he|contact|email|dia chi|address)\b", labelled[1]
        ):
            label = labelled[1]
        answer = re.sub(r"[^a-z]", "", label)
        if 2 <= len(answer) <= 24 and len(label.split()) <= 5:
            terms.add(answer)
        steps.extend(int(value) for value in re.findall(
            r"(?:^|[.;:\n])\s*(?:buoc|step|giai doan|stage)\s*(\d{1,2})\s*[:.)-]", text))
        if re.match(r"^(?:neu|khi|tru khi|chi khi|khong duoc|luu y|canh bao|if|when|unless|only if|must not|do not|warning|caution)\b", text):
            clarifications.add(text)
        # Arrows or explicit containment supply relationships, not inferred
        # edges between unrelated terms that merely share a source section.
        chain = re.split(r"\s*(?:->|→)\s*", text)
        if 2 <= len(chain) <= 12 and all(1 <= len(x.strip()) <= 100 for x in chain):
            relationship_count += len(chain) - 1
        hierarchy = re.match(r"^.{2,80}?\s+(?:bao gom|gom|consists of|includes)\s*:?\s*(.+)$", text)
        if hierarchy:
            children = re.split(r"[,;]", hierarchy.group(1))
            if 2 <= len(children) <= 10 and all(1 <= len(x.strip()) <= 80 for x in children):
                relationship_count += len(children)
    result: dict[str, dict[str, Any]] = {}
    if relationship_count:
        result["relationship_visualization"] = {"relationship_evidence": True, "relationship_count": min(relationship_count, 20)}
    if 3 <= len(steps) <= 20 and steps == list(range(1, len(steps) + 1)):
        result["practice"] = {"requires_ordering_practice": True, "ordered_sequence": True, "sequence_item_count": len(steps)}
    if len(terms) >= 3:
        result["terminology_reinforcement"] = {"definitions_supported": True, "terminology_count": min(len(terms), 20)}
    if len(clarifications) >= 2:
        result["faq"] = {"anticipated_questions": True, "question_count": min(len(clarifications), 8)}
    # Questions need not be pre-written in the source. Two distinct substantive
    # statements support bounded clarification/comparison questions at Stage 2.
    # This is an opportunity, not a claim that generated answers are verified.
    # Repeated lines whose only difference is an ordinal/number are one
    # statement, not two distinct clarification opportunities.
    statements = {re.sub(r"\d+", "#", _fold(t.strip())) for t in texts if _is_instructional_text(t)}
    if len(statements) >= 2 and sum(len(x.split()) for x in statements) >= 20:
        result.setdefault("faq", {"anticipated_questions": True, "question_count": min(3, len(statements))})
    return result


def compile_evidence_treatments(
    blueprint: dict[str, Any], source_map: dict[str, Any], manifest: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Add bounded optional supporting blocks to a validated V5 candidate.

    No primary owner, objective, unit boundary or existing block is rewritten.
    The pass is idempotent, reserves mandatory assessments, and never broadens
    a source scope. Whole scopes exceeding the scan bound are skipped visibly.
    """
    result = deepcopy(blueprint)
    profile = result.get("component_capabilities") or {}
    if result.get("architecture_contract_version") != 5 or profile.get("version") != 2:
        return result, []
    facts = {f["fact_id"]: f for f in manifest.get("facts", []) if isinstance(f, dict) and f.get("fact_id")}
    mapped = {f["id"]: f for f in source_map.get("facts", []) if isinstance(f, dict) and f.get("id")}
    scopes = {s["id"]: s for s in source_map.get("source_evidence_scopes", []) if isinstance(s, dict) and s.get("id")}
    inventory: dict[str, tuple[dict[str, dict[str, Any]], str]] = {}
    for sid, scope in scopes.items():
        ids = scope.get("source_fact_ids", [])
        valid = bool(ids) and len(ids) == len(set(ids)) and all(
            fid in facts and fid in mapped
            and mapped[fid].get("document_id") == scope.get("document_id")
            and mapped[fid].get("section_id") == scope.get("section_id")
            and facts[fid].get("document_id") == scope.get("document_id")
            and (facts[fid].get("source_ref") == mapped[fid].get("source_ref")
                 if facts[fid].get("source_ref") else
                 type(facts[fid].get("source_page")) is int
                 and facts[fid]["source_page"] == mapped[fid].get("page"))
            and isinstance(facts[fid].get("text"), str)
            for fid in ids)
        if not valid:
            inventory[sid] = ({}, "PROVENANCE_UNAVAILABLE")
            continue
        texts = [facts[fid]["text"] for fid in ids]
        if len(ids) > MAX_SCOPE_FACTS or sum(map(len, texts)) > MAX_SCOPE_CHARS:
            inventory[sid] = ({}, "EVIDENCE_SCAN_BOUND")
            continue
        signals = _signals(texts)
        signals.pop("practice", None)
        # Do not join an ordering prefix on one page with unrelated steps on
        # another. Missing page provenance uses the exact section/chunk key.
        groups: dict[tuple[Any, ...], list[str]] = {}
        for fid in ids:
            fact = facts[fid]
            key = (fact.get("source_page"), fact.get("source_ref"),
                   None if fact.get("source_page") else fact.get("source_chunk"))
            groups.setdefault(key, []).append(fact["text"])
        for group in groups.values():
            local = {**_signals(group), **_structured_signals(group)}
            for intent, descriptor in local.items():
                signals.setdefault(intent, descriptor)
        inventory[sid] = (signals, "EXPLICIT_EVIDENCE" if signals else "NO_EXPLICIT_TREATMENT_EVIDENCE")
    # Respect the existing downstream 72-component course budget as well as
    # the per-unit profile. Discovery must not manufacture a capacity failure.
    course_slots = max(0, 72 - sum(min(4, 1 + sum(
        b.get("intent") == "knowledge_check" or b.get("intent") in TREATMENTS
        for b in u.get("learning_blocks", [])))
        for c in result.get("chapters", []) for l in c.get("lessons", []) for u in l.get("units", [])))
    diagnostics: list[dict[str, Any]] = []
    for ci, chapter in enumerate(result.get("chapters", []), 1):
        for li, lesson in enumerate(chapter.get("lessons", []), 1):
            objectives = lesson.get("learning_objectives", [])
            for ui, unit in enumerate(lesson.get("units", []), 1):
                path = f"chapter_{ci}.lesson_{li}.unit_{ui}"
                blocks = unit.get("learning_blocks", [])
                present = {b.get("intent") for b in blocks}
                # Mandatory declarations keep priority; optional opportunities
                # must never create a new capability/capacity failure.
                reserved = 1 + sum(b.get("intent") == "knowledge_check" or b.get("intent") in TREATMENTS for b in blocks)
                slots = min(max(0, 4 - reserved), max(0, 12 - len(blocks)), course_slots)
                choices: dict[str, tuple[dict[str, Any], str, dict[str, Any]]] = {}
                reasons: set[str] = set()
                for anchor in blocks:
                    if anchor.get("intent") not in {"concept_explanation", "definition", "procedure", "comparison", "warning", "introduction"}:
                        continue
                    refs = anchor.get("learning_objective_refs", [])
                    if not refs or any(not re.fullmatch(r"lo_[1-9][0-9]*", r) or int(r[3:]) > len(objectives) for r in refs):
                        reasons.add("OBJECTIVE_LINK_UNAVAILABLE")
                        continue
                    for sid in anchor.get("primary_evidence_scope_ids", []):
                        scope = scopes.get(sid, {})
                        signals, reason = inventory.get(sid, ({}, "PROVENANCE_UNAVAILABLE"))
                        reasons.add(reason)
                        if not set(scope.get("concept_ids", [])).issubset(set(anchor.get("concept_ids", []))):
                            reasons.add("CONCEPT_SCOPE_MISMATCH")
                            continue
                        for intent, descriptor in signals.items():
                            if intent in present:
                                continue
                            # Reordering is useful for an action/sequence
                            # objective, not for unrelated factual recall.
                            objective_text = " ".join(_fold(str(objectives[int(r[3:]) - 1])) for r in refs)
                            if intent == "practice" and not re.search(r"\b(?:apply|perform|execute|order|sequence|implement|ap dung|thuc hien|sap xep|trinh tu|quy trinh|tien hanh)\b", objective_text):
                                reasons.add("NO_ORDERING_OBJECTIVE")
                                continue
                            choices.setdefault(intent, (anchor, sid, descriptor))
                added: list[str] = []
                for intent in TREATMENTS:
                    if intent not in choices or len(added) >= slots:
                        continue
                    anchor, sid, descriptor = choices[intent]
                    block_id = "treatment_" + hashlib.sha256(f"{path}:{sid}:{intent}".encode()).hexdigest()[:20]
                    if any(b.get("id") == block_id for b in blocks):
                        continue
                    blocks.append({
                        "id": block_id, "intent": intent, "importance": "supporting",
                        "concept_ids": list(anchor.get("concept_ids", [])), "primary_concept_ids": [],
                        "source_refs": list(anchor.get("source_refs", [])),
                        "primary_evidence_scope_ids": [], "supporting_evidence_scope_ids": [sid],
                        "learning_objective_refs": list(anchor["learning_objective_refs"]),
                        "content": deepcopy(descriptor),
                    })
                    added.append(intent)
                    course_slots -= 1
                # FAQ must follow instruction and checks in pedagogical order,
                # including after an idempotent second discovery pass.
                if added:
                    blocks[:] = [b for b in blocks if b.get("intent") != "faq"] + [b for b in blocks if b.get("intent") == "faq"]
                diagnostics.append({"unit_path": path, "candidate_intents": list(choices), "added_intents": added,
                                    "faq_status": "PRESENT" if "faq" in present or "faq" in added else "CAPACITY_BOUND" if "faq" in choices else "NO_GROUNDED_OPPORTUNITY",
                                    "omitted_intents": [intent for intent in choices if intent not in added],
                                    "decisions": [{
                                        "intent": intent,
                                        "status": "ADDED" if intent in added else "CAPACITY_BOUND",
                                        "evidence_scope_id": choice[1],
                                        "evidence_fact_count": len(scopes[choice[1]].get("source_fact_ids", [])),
                                        "basis": {"faq": "SOURCE_STATEMENT_CLARIFICATION", "practice": "SOURCE_SEQUENCE",
                                                  "relationship_visualization": "SOURCE_RELATIONSHIP_OPPORTUNITY",
                                                  "terminology_reinforcement": "SOURCE_DEFINITION"}[intent],
                                    } for intent, choice in choices.items()],
                                    "remaining_component_slots": max(0, slots - len(added)),
                                    "reason_codes": sorted(reasons | ({"CAPACITY_BOUND"} if len(choices) > slots else set()))})
    return result, diagnostics
