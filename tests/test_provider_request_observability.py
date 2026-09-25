import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi import HTTPException
from app.main import generate_content, call_provider_with_timeout, LESSON_AUTHOR_BLUEPRINT_RESPONSE_SCHEMA


class ProviderRequestObservabilityTests(unittest.TestCase):
    def test_sizes_only_and_configuration_unchanged(self):
        events = []
        response = SimpleNamespace(text='{}', parsed=None, candidates=[], usage_metadata=None, prompt_feedback=None)
        client = MagicMock()
        client.models.generate_content.return_value = response
        with patch('app.main.genai.Client', return_value=client):
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
        self.assertEqual(events[0]['internal_failure_code'], 'AI_PROVIDER_TIMEOUT')
        self.assertIsNone(events[0]['provider_http_status'])
        self.assertEqual(events[0]['usage_source'], 'unavailable')

    def test_telemetry_sink_failure_does_not_change_generation(self):
        client = MagicMock()
        client.models.generate_content.return_value = SimpleNamespace(text='{}', parsed=None, candidates=[], usage_metadata=None, prompt_feedback=None)
        with patch('app.main.genai.Client', return_value=client):
            text, _ = asyncio.run(generate_content('test', 'test', 'private', max_output_tokens=100,
                on_provider_telemetry=MagicMock(side_effect=RuntimeError('sink unavailable'))))
        self.assertEqual(text, '{}')
