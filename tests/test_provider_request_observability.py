import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi import HTTPException
from app.main import LESSON_AUTHOR_BLUEPRINT_RESPONSE_SCHEMA
from app.services.provider import generate_content, call_provider_with_timeout


class ProviderRequestObservabilityTests(unittest.TestCase):
    def test_sizes_only_and_configuration_unchanged(self):
        events = []
        response = SimpleNamespace(text='{}', parsed=None, candidates=[], usage_metadata=None, prompt_feedback=None)
        client = MagicMock()
        client.models.generate_content.return_value = response
        with patch('google.genai.Client', return_value=client):
            asyncio.run(generate_content('secret-key', 'test-model', 'PRIVATE_SOURCE_SENTINEL',
                max_output_tokens=65536, json_mode=True, response_schema=LESSON_AUTHOR_BLUEPRINT_RESPONSE_SCHEMA,
                request_timeout_ms=300000, on_provider_telemetry=events.append))
        started = next(e for e in events if e.get('event') == 'provider_request_started')
        self.assertEqual(started['prompt_chars'], len('PRIVATE_SOURCE_SENTINEL'))
        self.assertGreater(started['response_schema_chars'], 0)
        self.assertEqual(started['configured_max_output_tokens'], 65536)
        self.assertEqual(started['provider_timeout_ms'], 300000)
        self.assertEqual(started['usage_source'], 'unavailable')
        self.assertNotIn('PRIVATE_SOURCE_SENTINEL', json.dumps(events))
        self.assertNotIn('secret-key', json.dumps(events))
        self.assertEqual(client.models.generate_content.call_count, 1)

    def test_timeout_typed_at_boundary_no_identical_retry(self):
        events = []
        run = MagicMock(side_effect=TimeoutError())
        with self.assertRaises(HTTPException) as raised:
            asyncio.run(call_provider_with_timeout(run, 'test-model', request_timeout_ms=300000,
                on_provider_diagnostic=events.append))
        self.assertEqual(run.call_count, 1)
        self.assertEqual(raised.exception.detail['code'], 'AI_PROVIDER_TIMEOUT')
        self.assertEqual([event['event'] for event in events],
                         ['provider_http_attempt_started', 'provider_timeout'])
        timeout = events[-1]
        self.assertEqual(timeout['internal_failure_code'], 'AI_PROVIDER_TIMEOUT')
        self.assertIsNone(timeout['provider_http_status'])
        self.assertEqual(timeout['usage_source'], 'unavailable')
        self.assertEqual(timeout['provider_attempt'], 1)

    def test_success_records_exact_attempt_and_provider_reported_usage(self):
        events = []
        usage = SimpleNamespace(prompt_token_count=7, candidates_token_count=3, total_token_count=10)
        response = SimpleNamespace(text='{}', parsed=None, candidates=[], usage_metadata=usage, prompt_feedback=None)
        client = MagicMock()
        client.models.generate_content.return_value = response
        with patch('google.genai.Client', return_value=client):
            asyncio.run(generate_content('secret-key', 'test-model', 'PRIVATE_SOURCE_SENTINEL',
                max_output_tokens=100, request_timeout_ms=300000, on_provider_telemetry=events.append))
        received = next(event for event in events if event.get('event') == 'provider_response_received')
        self.assertEqual(received['provider_attempt'], 1)
        self.assertEqual(received['usage_source'], 'provider')
        self.assertEqual(received['provider_total_tokens'], 10)
        self.assertNotIn('PRIVATE_SOURCE_SENTINEL', json.dumps(events))

    def test_telemetry_sink_failure_does_not_change_generation(self):
        client = MagicMock()
        client.models.generate_content.return_value = SimpleNamespace(text='{}', parsed=None, candidates=[], usage_metadata=None, prompt_feedback=None)
        with patch('google.genai.Client', return_value=client):
            text, _ = asyncio.run(generate_content('test', 'test', 'private', max_output_tokens=100,
                on_provider_telemetry=MagicMock(side_effect=RuntimeError('sink unavailable'))))
        self.assertEqual(text, '{}')
