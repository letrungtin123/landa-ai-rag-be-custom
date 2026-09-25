from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import unittest

from app.instructional_opportunities import compile_evidence_treatments
from app.main import allocate_source_map_architecture_facts, allocate_blueprint_source_fact_ids, validate_course_architecture_evidence_scope, validate_v5_instructional_coherence
from app.source_map import build_source_map
from tests.test_evidence_scope_allocation_v5 import _manifest, _nodes, _v5_blueprint
from tests.test_component_instance_contract import PROFILE


def fixture(texts):
    manifest = _manifest(len(texts), 1)
    for fact, text in zip(manifest["facts"], texts):
        fact["text"] = text
    source_map = build_source_map(_nodes(1), manifest, locale="en")
    blueprint = _v5_blueprint(source_map)
    blueprint["component_capabilities"] = PROFILE.copy()
    blueprint["chapters"][0]["lessons"][0]["learning_objectives"] = ["Apply the documented procedure and explain its terms and conditions."]
    return blueprint, source_map, manifest


def blocks(blueprint):
    return blueprint["chapters"][0]["lessons"][0]["units"][0]["learning_blocks"]


class InstructionalOpportunityTests(unittest.TestCase):
    CASES = {
        "terminology_reinforcement": ["Hazard means a potential source of harm.", "Risk means likelihood combined with severity.", "Control means a measure reducing risk."],
        "practice": ["Step 1: Inspect readiness.", "Step 2: Perform the approved task.", "Step 3: Verify completion."],
        "relationship_visualization": ["Preparation -> Execution -> Verification"],
        "faq": ["If the guard is missing, stop the task.", "When the inspection fails, notify the supervisor."],
    }

    def test_each_treatment_is_derived_from_source_not_model_flags(self):
        for intent, texts in self.CASES.items():
            with self.subTest(intent=intent):
                blueprint, source_map, manifest = fixture(texts)
                before = deepcopy(blueprint)
                enriched, diagnostics = compile_evidence_treatments(blueprint, source_map, manifest)
                self.assertEqual(blueprint, before)
                self.assertEqual(blocks(enriched)[-1]["intent"], intent)
                self.assertEqual(blocks(enriched)[0], blocks(blueprint)[0])
                self.assertEqual(blocks(enriched)[-1]["primary_evidence_scope_ids"], [])
                self.assertEqual(blocks(enriched)[-1]["primary_concept_ids"], [])
                self.assertNotIn("source_fact_ids", blocks(enriched)[-1])
                self.assertFalse(validate_course_architecture_evidence_scope(enriched, source_map).errors)
                self.assertFalse(validate_v5_instructional_coherence(enriched).errors)
                self.assertNotIn(texts[0], json.dumps(diagnostics))
                again, _ = compile_evidence_treatments(enriched, source_map, manifest)
                self.assertEqual(again, enriched)

    def test_vietnamese_and_negative_evidence_and_objective_gates(self):
        cases = [
            (["Mối nguy là nguồn có thể gây hại.", "Rủi ro là kết hợp khả năng và hậu quả.", "Kiểm soát là biện pháp giảm rủi ro."], "terminology_reinforcement"),
            (["Bước 1: Kiểm tra.", "Bước 2: Thực hiện.", "Bước 3: Xác nhận."], "practice"),
            (["Nếu thiếu bảo hộ, dừng lại.", "Khi có sự cố, báo người phụ trách."], "faq"),
            (["Hệ thống bao gồm: bộ nhận, bộ xử lý, bộ xuất."], "relationship_visualization"),
        ]
        for texts, intent in cases:
            bp, sm, mf = fixture(texts)
            bp["chapters"][0]["lessons"][0]["learning_objectives"] = ["Thực hiện đúng quy trình và giải thích các điều kiện."]
            enriched, _ = compile_evidence_treatments(bp, sm, mf)
            self.assertEqual(blocks(enriched)[-1]["intent"], intent)
        for texts in (["1. Red", "2. Green", "3. Blue"], ["Some plain explanation."], ["If it fails, stop."] * 2,
                      ["Step 1: Inspect.", "Step 3: Complete.", "Step 4: Exit."]):
            bp, sm, mf = fixture(texts)
            blocks(bp)[0]["content"] = {"definitions_supported": True, "terminology_count": 99, "ordered_sequence": True}
            enriched, _ = compile_evidence_treatments(bp, sm, mf)
            self.assertEqual(enriched, bp)
        bp, sm, mf = fixture(self.CASES["practice"])
        bp["chapters"][0]["lessons"][0]["learning_objectives"] = ["Identify the name of a colour."]
        enriched, diagnostics = compile_evidence_treatments(bp, sm, mf)
        self.assertEqual(enriched, bp)
        self.assertIn("NO_ORDERING_OBJECTIVE", diagnostics[0]["reason_codes"])

    def test_scope_provenance_legacy_capacity_and_missing_evidence(self):
        original, sm, mf = fixture(self.CASES["terminology_reinforcement"])
        for version in (3, 4):
            bp = deepcopy(original)
            bp["architecture_contract_version"] = version
            self.assertEqual(compile_evidence_treatments(bp, sm, mf), (bp, []))
        for field, invalid in (("document_id", "other-doc"), ("source_ref", "other-section")):
            manifest = deepcopy(mf)
            manifest["facts"][0][field] = invalid
            enriched, d = compile_evidence_treatments(original, sm, manifest)
            self.assertEqual(enriched, original)
            self.assertIn("PROVENANCE_UNAVAILABLE", d[0]["reason_codes"])
        missing = {**mf, "facts": mf["facts"][:-1]}
        self.assertEqual(compile_evidence_treatments(original, sm, missing)[0], original)
        bp = deepcopy(original)
        for n in range(3):
            blocks(bp).append({**deepcopy(blocks(bp)[0]), "id": f"check_{n}", "intent": "knowledge_check"})
        enriched, d = compile_evidence_treatments(bp, sm, mf)
        self.assertEqual(enriched, bp)
        self.assertIn("CAPACITY_BOUND", d[0]["reason_codes"])
        manifest = deepcopy(mf)
        manifest["facts"][0]["text"] = "x" * 64001
        self.assertIn("EVIDENCE_SCAN_BOUND", compile_evidence_treatments(original, sm, manifest)[1][0]["reason_codes"])

    def test_paginated_manifest_uses_exact_document_page_not_invented_section(self):
        bp, sm, mf = fixture(self.CASES["terminology_reinforcement"])
        for fact in mf["facts"]:
            fact.pop("source_ref")
        self.assertEqual(blocks(compile_evidence_treatments(bp, sm, mf)[0])[-1]["intent"], "terminology_reinforcement")
        mf["facts"][0]["source_page"] = 999
        enriched, diag = compile_evidence_treatments(bp, sm, mf)
        self.assertEqual(enriched, bp)
        self.assertIn("PROVENANCE_UNAVAILABLE", diag[0]["reason_codes"])

    def test_course_capacity_and_missing_local_objective_do_not_add_blocks(self):
        bp, sm, mf = fixture(self.CASES["faq"])
        lesson = bp["chapters"][0]["lessons"][0]
        template = deepcopy(lesson["units"][0])
        template["learning_blocks"].extend([
            {**deepcopy(template["learning_blocks"][0]), "id": f"check_{n}", "intent": "knowledge_check"}
            for n in range(2)])
        lesson["units"] = [deepcopy(template) for _ in range(24)]
        self.assertEqual(compile_evidence_treatments(bp, sm, mf)[0], bp)
        bp, sm, mf = fixture(self.CASES["faq"])
        blocks(bp)[0]["learning_objective_refs"] = []
        self.assertEqual(compile_evidence_treatments(bp, sm, mf)[0], bp)

    def test_371_and_1001_facts_remain_owned_once_and_deterministic(self):
        for count in (371, 1001):
            mf = _manifest(count, 6)
            for fact in mf["facts"]:
                fact["text"] = "Preparation -> Execution -> Verification"
            sm = build_source_map(_nodes(6), mf, locale="en")
            bp = _v5_blueprint(sm)
            bp["component_capabilities"] = PROFILE.copy()
            enriched, _ = compile_evidence_treatments(bp, sm, mf)
            self.assertEqual(compile_evidence_treatments(bp, sm, mf)[0], enriched)
            self.assertFalse(validate_course_architecture_evidence_scope(enriched, sm).errors)
            allocated = allocate_source_map_architecture_facts(enriched, sm, mf)
            allocation = allocated["source_fact_allocation"]
            self.assertEqual(allocation["allocated_count"], count)
            self.assertEqual(allocation["unallocated"], [])
            self.assertEqual(len({x["fact_id"] for x in allocation["allocations"]}), count)
            for ch in allocated["chapters"]:
                for lesson in ch["lessons"]:
                    for unit in lesson["units"]:
                        for b in unit["learning_blocks"]:
                            if b["id"].startswith("treatment_"):
                                self.assertEqual(b["source_fact_ids"], [])

    def test_node_registry_maps_discovered_treatments_permissions_and_faq_last(self):
        cases = []
        blueprints = []
        for texts in [*self.CASES.values(), self.CASES["faq"] + self.CASES["terminology_reinforcement"]]:
            bp, sm, mf = fixture(texts)
            enriched, _ = compile_evidence_treatments(bp, sm, mf)
            allocated = allocate_source_map_architecture_facts(enriched, sm, mf)
            finalized = allocate_blueprint_source_fact_ids(deepcopy(allocated), mf, _nodes(1))
            blueprints.append({**finalized, "source_map": sm})
            unit = allocated["chapters"][0]["lessons"][0]["units"][0]
            cases.append({"blocks": unit["learning_blocks"], "unit_source_fact_ids": unit["source_fact_ids"],
                          "component_capabilities": PROFILE, "unit_path": "chapter_1.lesson_1.unit_1"})
        script = """
import fs from 'node:fs';
import pg from 'pg';
import {planSemanticLearningBlocks} from './src/modules/ai-chatbot/lesson-author-component-registry.logic.ts';
import {validateLessonAuthorBlueprintArchitecture} from './src/modules/ai-chatbot/lesson-author-blueprint-validator.logic.ts';
pg.Pool.prototype.query=()=>{throw new Error('TEST_DATABASE_ACCESS_FORBIDDEN')};
pg.Pool.prototype.connect=()=>{throw new Error('TEST_DATABASE_ACCESS_FORBIDDEN')};
globalThis.fetch=()=>{throw new Error('TEST_NETWORK_ACCESS_FORBIDDEN')};
const interval=globalThis.setInterval;
globalThis.setInterval=(...args)=>{const timer=interval(...args);timer.unref();return timer};
const {normalizeLessonAuthorBlueprint}=await import('./src/modules/ai-chatbot/chat.service.ts');
const {cases,blueprints}=JSON.parse(fs.readFileSync(0,'utf8'));
const persisted=blueprints.map(b=>{
 const parsed=normalizeLessonAuthorBlueprint(b,{requireContentArchitecture:true,requirePhaseOneContract:true});
 const validation=validateLessonAuthorBlueprintArchitecture(parsed,parsed.source_map);
 const read=normalizeLessonAuthorBlueprint(JSON.parse(JSON.stringify(parsed)));
 return {errors:validation.errors,types:read.chapters[0].lessons[0].units[0].component_plan.map(p=>p.type)};
});
console.log('RESULT:'+JSON.stringify({persisted,cases:cases.map(input=>({
  selected:planSemanticLearningBlocks(input).map(p=>p.type),
  restricted:planSemanticLearningBlocks({...input,allowed_component_types:new Set(['html','problem'])}).map(p=>p.type)
}))}));
"""
        result = subprocess.run([shutil.which("node") or "node", "--import", "tsx", "--input-type=module", "-e", script],
                                input=json.dumps({"cases": cases, "blueprints": blueprints}), text=True, encoding="utf-8", capture_output=True, timeout=30,
                                cwd=Path(__file__).resolve().parents[2] / "landa-backend")
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(next(line[7:] for line in result.stdout.splitlines() if line.startswith("RESULT:")))
        for item in output["persisted"]:
            self.assertEqual(item["errors"], [])
        actual = output["cases"]
        for item, kind in zip(actual, ("la_crossword", "la_sortable", "la_diagram", "la_faq")):
            self.assertIn(kind, item["selected"])
            self.assertEqual(item["restricted"], ["html"])
        self.assertEqual(actual[-1]["selected"], ["html", "la_crossword", "la_faq"])
        self.assertEqual(output["persisted"][-1]["types"], ["html", "la_crossword", "la_faq"])


if __name__ == "__main__":
    unittest.main()
