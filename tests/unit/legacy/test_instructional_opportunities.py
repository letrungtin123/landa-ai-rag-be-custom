import json
import os
import shutil
import subprocess
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

from app.instructional_opportunities import compile_evidence_treatments
from app.services.lesson_author.architecture_validation import validate_v5_instructional_coherence
from app.services.lesson_author.evidence_scope import (
    allocate_source_map_architecture_facts,
    validate_course_architecture_evidence_scope,
)
from app.services.lesson_author.granularity import allocate_blueprint_source_fact_ids
from app.source_map import build_source_map
from tests.test_component_instance_contract import PROFILE
from tests.unit.legacy.test_evidence_scope_allocation_v5 import _manifest, _nodes, _v5_blueprint

# Folders of the backend's src/modules/ai-chatbot/ai-id: "" is ai-id/ itself (the layout today), the others the
# planned move-only layout (landa-backend scripts/codemods/ai-id-moves.json; file names do not change).
AI_ID_FOLDERS = (
    "", "shared", "contracts", "content", "sessions", "sources", "orchestration", "orchestration/idm",
    "orchestration/steps", "orchestration/finalize", "orchestration/dispatch", "orchestration/worker",
    "orchestration/admission", "workspace/read", "workspace/stream", "workspace/edit", "workspace/apply",
    "workspace/launch", "v1", "legacy",
)


def ai_id_candidates(name: str) -> tuple[str, ...]:
    """Where an ai-id file can be, under src/modules/ai-chatbot: each ai-id folder, then the flat layout that came
    before the product sub-folders (refactor wave 2)."""
    return (*(f"ai-id/{folder}/{name}" if folder else f"ai-id/{name}" for folder in AI_ID_FOLDERS), name)


# Backend modules the Node half of the registry test imports, under src/modules/ai-chatbot.
BACKEND_MODULES = {
    "__REGISTRY__": ai_id_candidates("lesson-author-component-registry.logic.ts"),
    "__VALIDATOR__": ai_id_candidates("lesson-author-blueprint-validator.logic.ts"),
    "__CHAT_SERVICE__": ("chat/chat.service.ts", "chat.service.ts"),
}

# The Node half imports chat.service.ts, whose env.ts needs these settings when it loads. The child process gets
# its own environment: the placeholders of the backend's CI (every service on a closed port) plus what Node needs
# to start, never this process's settings (exporting them here would change the Python Settings tests).
NODE_TEST_ENV = {
    "NODE_ENV": "test",
    "DATABASE_URL": "postgresql://test:test@127.0.0.1:1/test",
    "DATABASE_POOLER_URL": "",
    "REDIS_URL": "redis://127.0.0.1:1",
    "RABBITMQ_URL": "amqp://test:test@127.0.0.1:1",
    "SUPABASE_URL": "http://127.0.0.1:1",
    "SUPABASE_SERVICE_KEY": "test-placeholder-not-a-key",
    "AI_RAG_SERVICE_URL": "http://127.0.0.1:1",
    "JWT_SECRET": "test-placeholder-not-a-secret",
    "JWT_ACCESS_EXPIRES_IN": "15m",
    "JWT_REFRESH_EXPIRES_IN": "7d",
    "BCRYPT_SALT_ROUNDS": "4",
    "CORS_ORIGIN": "http://localhost:5274",
    "PORT": "0",
}
NODE_INHERITED = {"PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP", "HOME", "USERPROFILE",
                  "APPDATA", "LOCALAPPDATA"}


def node_env() -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key.upper() in NODE_INHERITED}
    env.update(NODE_TEST_ENV)
    return env


def backend_module(backend: Path, candidates: tuple[str, ...]) -> str:
    for relative in candidates:
        if (backend / "src/modules/ai-chatbot" / relative).is_file():
            return f"./src/modules/ai-chatbot/{relative}"
    raise AssertionError(f"backend module not found under {backend}: {candidates}")


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
            self.assertIn(intent, [b["intent"] for b in blocks(enriched)])
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

    def test_source_defined_counted_process_survives_stripped_list_numbers(self):
        for header in ("Quy trình 4 bước", "Procedure 4 steps"):
            texts = [header, "Identify the documented hazard before starting work.",
                     "Evaluate the severity and the likelihood of the hazard.",
                     "Determine the risk level and the handling priority.",
                     "Propose appropriate controls for the identified risk."]
            bp, sm, mf = fixture(texts)
            enriched, diagnostics = compile_evidence_treatments(bp, sm, mf)
            self.assertIn("practice", [b["intent"] for b in blocks(enriched)])
            self.assertEqual(blocks(enriched)[-1]["intent"], "practice")
            self.assertEqual(diagnostics[0]["faq_status"], "NO_GROUNDED_OPPORTUNITY")
            self.assertEqual(compile_evidence_treatments(enriched, sm, mf)[0], enriched)

    def test_numbered_procedure_not_arbitrary_list_and_no_cross_page_assembly(self):
        texts = ["Approved procedure", "1. Inspect the initial working conditions.",
                 "2. Perform the approved operation carefully.", "3. Verify the final completed result."]
        bp, sm, mf = fixture(texts)
        self.assertIn("practice", [b["intent"] for b in blocks(compile_evidence_treatments(bp, sm, mf)[0])])
        for no_header in (texts[1:], ["Quy trình 4 bước", *texts[1:]]):
            bp, sm, mf = fixture(no_header)
            self.assertNotIn("practice", [b["intent"] for b in blocks(compile_evidence_treatments(bp, sm, mf)[0])])
        bp, sm, mf = fixture(self.CASES["practice"])
        mf["facts"][-1]["source_page"] = 2
        sm["facts"][-1]["page"] = 2
        self.assertNotIn("practice", [b["intent"] for b in blocks(compile_evidence_treatments(bp, sm, mf)[0])])

    def test_faq_synthesis_without_questions_but_not_empty_footer_or_repetition(self):
        statements = ["Workers inspect their protective equipment before using it at work.",
                      "Workers clean and store their protective equipment after using it."]
        bp, sm, mf = fixture(statements)
        enriched, _ = compile_evidence_treatments(bp, sm, mf)
        self.assertEqual(enriched, bp)
        for bad in ([statements[0]] * 3, [f"Evidence fact {i}: source-backed procedure or definition." for i in range(4)], ["THANK YOU for your participation in our training course.",
                    "Contact our team for more information at support@example.test."]):
            bp, sm, mf = fixture(bad)
            self.assertEqual(compile_evidence_treatments(bp, sm, mf)[0], bp)

    def test_model_membership_and_glossary_tables_without_causal_invention(self):
        # A literal source model heading, not an inferred title supplied by AI.
        texts = ["Model of implementation", "Leadership policy and commitment",
                 "Worker training and awareness", "Monitoring reporting and improvement"]
        bp, sm, mf = fixture(texts)
        self.assertIn("relationship_visualization", [b["intent"] for b in blocks(compile_evidence_treatments(bp, sm, mf)[0])])
        terms = ["Hazard: A potential source of harm at the workplace.",
                 "Risk: The combination of likelihood and severity of harm.",
                 "Control: An approved measure reducing an identified workplace risk."]
        bp, sm, mf = fixture(terms)
        enriched, _ = compile_evidence_treatments(bp, sm, mf)
        self.assertIn("terminology_reinforcement", [b["intent"] for b in blocks(enriched)])
        self.assertEqual(blocks(enriched)[-1]["intent"], "terminology_reinforcement")

    def test_faq_and_best_activity_fit_capacity_without_forcing_all_types(self):
        texts = self.CASES["terminology_reinforcement"] + self.CASES["faq"] + self.CASES["practice"]
        bp, sm, mf = fixture(texts)
        blocks(bp).append({**deepcopy(blocks(bp)[0]), "id": "check", "intent": "knowledge_check"})
        enriched, diag = compile_evidence_treatments(bp, sm, mf)
        intents = [b["intent"] for b in blocks(enriched)]
        self.assertEqual(intents, ["concept_explanation", "knowledge_check", "practice", "faq"])
        self.assertIn("terminology_reinforcement", diag[0]["omitted_intents"])
        self.assertIn("CAPACITY_BOUND", diag[0]["reason_codes"])

    def test_action_mnemonic_is_ordered_but_glossary_and_scenario_labels_are_not_terms(self):
        texts = ["Quy tắc PASS dùng bình chữa cháy:",
                 "P – Pull: Rút chốt an toàn; A – Aim: Nhắm vòi vào gốc lửa;",
                 "S – Squeeze: Bóp cò để xịt; S – Sweep: Quét vòi qua lại để dập tắt lửa"]
        bp, sm, mf = fixture(texts)
        self.assertIn("practice", [b["intent"] for b in blocks(compile_evidence_treatments(bp, sm, mf)[0])])
        bp, sm, mf = fixture([f"Scenario {i}: A worker performs an approved documented operation." for i in range(1, 4)])
        self.assertNotIn("terminology_reinforcement", [b["intent"] for b in blocks(compile_evidence_treatments(bp, sm, mf)[0])])

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
import {planSemanticLearningBlocks} from '__REGISTRY__';
import {validateLessonAuthorBlueprintArchitecture} from '__VALIDATOR__';
pg.Pool.prototype.query=()=>{throw new Error('TEST_DATABASE_ACCESS_FORBIDDEN')};
pg.Pool.prototype.connect=()=>{throw new Error('TEST_DATABASE_ACCESS_FORBIDDEN')};
globalThis.fetch=()=>{throw new Error('TEST_NETWORK_ACCESS_FORBIDDEN')};
const interval=globalThis.setInterval;
globalThis.setInterval=(...args)=>{const timer=interval(...args);timer.unref();return timer};
const {normalizeLessonAuthorBlueprint}=await import('__CHAT_SERVICE__');
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
        backend = Path(__file__).resolve().parents[4] / "landa-backend"
        for placeholder, relative in BACKEND_MODULES.items():
            script = script.replace(placeholder, backend_module(backend, relative))
        self.assertTrue((backend / ".env").is_file(),
                        f"{backend / '.env'} is missing: the backend's env.ts exits without it (copy .env.example)")
        result = subprocess.run([shutil.which("node") or "node", "--import", "tsx", "--input-type=module", "-e", script],
                                input=json.dumps({"cases": cases, "blueprints": blueprints}), text=True, encoding="utf-8", capture_output=True, timeout=30,
                                cwd=backend, env=node_env())
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


class BackendLookupTests(unittest.TestCase):
    def test_a_module_is_found_in_the_current_and_in_every_planned_ai_id_folder(self):
        name = "lesson-author-component-registry.logic.ts"
        for folder in AI_ID_FOLDERS:
            with self.subTest(folder=folder), tempfile.TemporaryDirectory() as tmp:
                relative = f"ai-id/{folder}/{name}" if folder else f"ai-id/{name}"
                target = Path(tmp) / "src/modules/ai-chatbot" / relative
                target.parent.mkdir(parents=True)
                target.write_text("", encoding="utf-8")
                found = backend_module(Path(tmp), ai_id_candidates(name))
                self.assertEqual(found, f"./src/modules/ai-chatbot/{relative}")

    def test_the_first_existing_place_wins_and_a_missing_module_fails(self):
        name = "lesson-author-blueprint-validator.logic.ts"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "src/modules/ai-chatbot"
            for relative in (f"ai-id/v1/{name}", f"ai-id/{name}"):
                (root / relative).parent.mkdir(parents=True, exist_ok=True)
                (root / relative).write_text("", encoding="utf-8")
            found = backend_module(Path(tmp), ai_id_candidates(name))
            self.assertEqual(found, f"./src/modules/ai-chatbot/ai-id/{name}")
            with self.assertRaises(AssertionError):
                backend_module(Path(tmp), ai_id_candidates("lesson-author-missing.logic.ts"))

    def test_the_node_process_gets_placeholders_and_none_of_this_process_settings(self):
        with mock.patch.dict(os.environ, {"SUPABASE_URL": "https://real.invalid", "LESSON_AUTHOR_FLAG": "1"}):
            env = node_env()
            self.assertEqual(os.environ["SUPABASE_URL"], "https://real.invalid")
        self.assertEqual(env["SUPABASE_URL"], "http://127.0.0.1:1")
        self.assertEqual(env["NODE_ENV"], "test")
        self.assertNotIn("LESSON_AUTHOR_FLAG", env)
        self.assertTrue(any(key.upper() == "PATH" for key in env))


if __name__ == "__main__":
    unittest.main()
