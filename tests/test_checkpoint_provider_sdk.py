"""Exercise the installed Google SDK, mocking only HTTP; no Gemini or database."""
import asyncio
import json
import unittest
from unittest.mock import patch

import httpx
import requests
from google.genai import errors

from app import main
from tests.test_staged_instance_output import checkpoint_instance_fixture
from tests.test_checkpoint_component_quality_repair import instance_wire
from tests.test_chapter_checkpoint import checkpoint_result


def response(status, body):
    result = requests.Response()
    result.status_code = status
    result._content = json.dumps(body).encode()
    result.headers["Content-Type"] = "application/json"
    return result


class CheckpointProviderSDKTests(unittest.TestCase):
    def send(self, request, manifest):
        async def generation(req, _pool):
            return await checkpoint_result(req, manifest)

        async def run():
            main.app.dependency_overrides[main.get_db] = lambda: None
            try:
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
                    return await client.post("/v1/lesson-author/chapter-checkpoint", json=request.model_dump(),
                                             headers={"X-Landa-AI-Service-Token": "offline-token"})
            finally:
                main.app.dependency_overrides.pop(main.get_db, None)

        with patch.object(main.settings, "service_token", "offline-token"), patch("app.main.lesson_author_proposal", generation):
            return asyncio.run(run())

    def test_real_sdk_code_not_status_code_is_classified(self):
        error = errors.ClientError(400, response(400, {"error": {
            "status": "INVALID_ARGUMENT", "message": "generation_config.response_schema max_items PRIVATE_SOURCE"}}))
        self.assertEqual(error.code, 400)
        self.assertFalse(hasattr(error, "status_code"))
        self.assertEqual(main.provider_http_error_status(error), 400)
        self.assertTrue(main.is_stage_two_provider_schema_error(error))
        diagnostics = main.safe_provider_error_diagnostics(error)
        self.assertEqual(diagnostics["provider_schema_constraint"], "MAX_ITEMS")
        self.assertNotIn("PRIVATE_SOURCE", json.dumps(diagnostics))

    def test_generic_provider_message_is_distinguished_from_redacted_details(self):
        for message, detail_list, message_class in (
            ("Request contains an invalid argument.", [], "GENERIC_INVALID_ARGUMENT"),
            ("PRIVATE_SOURCE", [{"message": "PRIVATE_DETAIL"}], "REDACTED_OTHER"),
        ):
            error = errors.ClientError(400, response(400, {"error": {"message": message, "details": detail_list}}))
            diagnostics = main.safe_provider_error_diagnostics(error)
            self.assertEqual(diagnostics["provider_message_class"], message_class)
            self.assertEqual(diagnostics["provider_error_detail_count"], len(detail_list))
            self.assertNotIn("PRIVATE", json.dumps(diagnostics))
        alternate = errors.ClientError(400, response(400, {"error": {
            "status": "INVALID_ARGUMENT", "message": "JSON schema maxItems must be greater than zero PRIVATE_SOURCE"}}))
        self.assertTrue(main.is_stage_two_provider_schema_error(alternate))
        diagnostics = main.safe_provider_error_diagnostics(alternate)
        self.assertIn("POSITIVE_BOUND", diagnostics["provider_error_markers"])
        self.assertNotIn("PRIVATE_SOURCE", json.dumps(diagnostics))

    def test_sdk_http_failures_reach_endpoint_as_provider_not_lesson_failure(self):
        request, _, _, manifest = checkpoint_instance_fixture()
        for status, message, expected in (
            (400, "generationConfig.responseSchema maxItems PRIVATE_SOURCE", "LESSON_PROVIDER_REQUEST_SCHEMA_INVALID"),
            (400, "invalid request PRIVATE_SOURCE", "LESSON_PROVIDER_REQUEST_FAILED"),
            (403, "permission denied PRIVATE_SOURCE", "LESSON_PROVIDER_REQUEST_FAILED"),
        ):
            with self.subTest(status=status, code=expected):
                body = {"error": {"code": status, "status": "INVALID_ARGUMENT", "message": message}}
                with patch("requests.Session.request", return_value=response(status, body)) as transport, \
                        self.assertLogs("app.main", "INFO") as logs:
                    result = self.send(request, manifest)
                self.assertEqual(transport.call_count, 1)  # no identical retry or content repair
                self.assertEqual(result.status_code, 502)
                detail = result.json()["detail"]
                self.assertEqual(detail["code"], "PROVIDER_ERROR")
                self.assertEqual(detail["internal_failure_code"], expected)
                self.assertEqual(detail["correlation_id"], request.correlation_id)
                joined = "\n".join(logs.output)
                final = next(line for line in logs.output if "lesson_author_checkpoint_diagnostic" in line)
                self.assertIn(f'"provider_http_status": {status}', final)
                self.assertIn('"provider_error_type": "ClientError"', final)
                self.assertIn('"usage_source": "unavailable"', final)
                self.assertIn(request.correlation_id, final)
                self.assertNotIn("PRIVATE_SOURCE", joined + result.text)
                self.assertNotIn("CHAPTER_CHECKPOINT_INTERNAL_ERROR", joined + result.text)
                self.assertNotIn("chapter_checkpoint_content_validation", joined)

    def test_transport_exception_keeps_unavailable_status_and_no_retry(self):
        request, _, _, manifest = checkpoint_instance_fixture()
        with patch("requests.Session.request", side_effect=requests.ConnectionError("PRIVATE_URL_AND_TOKEN")) as transport, \
                self.assertLogs("app.main", "INFO") as logs:
            result = self.send(request, manifest)
        self.assertEqual(transport.call_count, 1)
        self.assertEqual(result.status_code, 502)
        self.assertEqual(result.json()["detail"]["internal_failure_code"], "LESSON_PROVIDER_REQUEST_FAILED")
        final = next(line for line in logs.output if "lesson_author_checkpoint_diagnostic" in line)
        self.assertIn('"provider_http_status": null', final)
        self.assertIn('"provider_error_type": "ConnectionError"', final)
        self.assertNotIn("PRIVATE_URL_AND_TOKEN", "\n".join(logs.output) + result.text)

    def test_sdk_roundtrip_valid_payload_still_reaches_full_chapter_acceptance(self):
        request, valid, _, manifest = checkpoint_instance_fixture()
        request = request.model_copy(update={"max_output_tokens": 30000})
        body = {"candidates": [{"content": {"parts": [{"text": json.dumps(instance_wire(valid))}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 200, "totalTokenCount": 300}}
        with patch("requests.Session.request", return_value=response(200, body)) as transport, \
                self.assertLogs("app.main", "INFO") as logs:
            result = self.send(request, manifest)
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.json()["status"], "unit_ready")
            final_request = request.model_copy(update={"checkpoint_action": "validate_chapter", "checkpoint_unit_index": None,
                "checkpoint_units": [main.ChapterCheckpointUnit(unit_index=0, unit=result.json()["unit"])]})
            final = self.send(final_request, manifest)
        self.assertEqual(final.status_code, 200)
        self.assertEqual(final.json()["status"], "ready")
        self.assertEqual(transport.call_count, 1)
        payload = json.loads(transport.call_args.kwargs["data"])
        self.assertEqual(payload["generationConfig"]["maxOutputTokens"], 30000)
        self.assertEqual(transport.call_args.kwargs["timeout"], 180)
        slots = payload["generationConfig"]["responseSchema"]["properties"]["components"]
        self.assertEqual(set(slots["required"]), {"c0", "c1", "c2", "c3"})
        from tests.staged_schema_probe import BOUNDS, visit_schema
        self.assertTrue(all(not BOUNDS.intersection(node) for _, node in visit_schema(payload["generationConfig"]["responseSchema"])))
        joined = "\n".join(logs.output)
        self.assertIn('"schema_projection_version": "staged-wire-shape-1"', joined)
        self.assertIn('"server_validation_unchanged": true', joined)
        self.assertIn('"provider_finish_reason": "FinishReason.STOP"', joined)
        self.assertIn('"provider_total_tokens": 300', joined)
        self.assertIn('"usage_source": "provider"', joined)

    def test_quota_timeout_and_transient_retry_bounds_unchanged(self):
        for status, attempts, expected in ((429, 1, "AI_PROVIDER_QUOTA_EXHAUSTED"), (503, 2, "AI_PROVIDER_UNAVAILABLE")):
            events = []
            body = {"error": {"code": status, "message": "PRIVATE", "status": "RESOURCE_EXHAUSTED" if status == 429 else "UNAVAILABLE"}}
            with patch("requests.Session.request", return_value=response(status, body)) as transport, \
                    patch("app.main.PROVIDER_TRANSIENT_RETRY_DELAY_SECONDS", 0):
                with self.assertRaises(main.HTTPException) as failure:
                    asyncio.run(main.generate_content("offline-key", "offline-model", "PRIVATE", max_output_tokens=30000,
                                                      on_provider_telemetry=events.append))
            self.assertEqual(failure.exception.detail["code"], expected)
            self.assertEqual(transport.call_count, attempts)
            self.assertNotIn("PRIVATE", json.dumps(events))

    def test_wire_projection_does_not_allow_supporting_component_to_claim_owned_facts(self):
        request, valid, _, manifest = checkpoint_instance_fixture()
        wire = instance_wire(valid)
        wire["components"]["c1"]["covered_source_fact_ids"] = ["fact-0"]
        body = {"candidates": [{"content": {"parts": [{"text": json.dumps(wire)}]}, "finishReason": "STOP"}]}
        with patch("requests.Session.request", return_value=response(200, body)) as transport:
            result = self.send(request, manifest)
        self.assertNotEqual(result.status_code, 200)
        self.assertEqual(transport.call_count, 1)  # ownership violation never repaired
        self.assertNotIn("unit_ready", result.text)

    def test_wire_projection_still_rejects_bad_component_after_bounded_repair(self):
        request, valid, _, manifest = checkpoint_instance_fixture()
        wire = instance_wire(valid)
        wire["components"]["c2"]["nodes"] = []
        body = {"candidates": [{"content": {"parts": [{"text": json.dumps(wire)}]}, "finishReason": "STOP"}]}
        with patch("requests.Session.request", return_value=response(200, body)) as transport:
            result = self.send(request, manifest)
        self.assertNotEqual(result.status_code, 200)
        self.assertLessEqual(transport.call_count, 2)  # existing single repair, no new loop
        self.assertNotIn("unit_ready", result.text)


if __name__ == "__main__":
    unittest.main()
