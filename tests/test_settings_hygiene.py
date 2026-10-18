"""Settings hygiene (STD-3, F-10): secrets are SecretStr, every numeric setting is bounded.

Reads only ``.env.example`` (the committed template) and in-test values; never a live env file.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from annotated_types import Ge, Gt, Le, Lt
from pydantic import SecretStr, ValidationError

from app.core.config import Settings, missing_required_setting_names
from app.infra.db import DatabaseConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
SECRET_FIELDS = {"database_url", "service_token", "service_hmac_secrets", "supabase_service_key"}
SECRETS = {
    "DATABASE_URL": "postgresql://ai_rag:db-password-123@db.internal:5432/landa",
    "AI_RAG_SERVICE_TOKEN": "service-token-value-123",
    "AI_RAG_SERVICE_HMAC_SECRETS": "kid-1:hmac-secret-value-123",
    "SUPABASE_SERVICE_KEY": "supabase-service-key-123",
}


def numeric_fields() -> dict[str, Any]:
    return {name: field for name, field in Settings.model_fields.items()
            if field.annotation in {int, float}}


def env_example_values() -> dict[str, str]:
    values = {}
    for line in (REPO_ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


class SecretSettingsTests(unittest.TestCase):
    def test_secrets_are_secret_strings_and_never_shown(self) -> None:
        self.assertEqual({name for name, field in Settings.model_fields.items() if field.annotation is SecretStr},
                         SECRET_FIELDS)
        with patch.dict(os.environ, SECRETS, clear=True):
            configured = Settings()
        shown = repr(configured) + str(configured) + configured.model_dump_json()
        for value in SECRETS.values():
            self.assertNotIn(value, shown)
        self.assertEqual(configured.database_url.get_secret_value(), SECRETS["DATABASE_URL"])
        self.assertEqual(configured.service_token.get_secret_value(), SECRETS["AI_RAG_SERVICE_TOKEN"])

    def test_an_invalid_value_next_to_a_secret_does_not_echo_the_secret(self) -> None:
        with patch.dict(os.environ, {**SECRETS, "AI_RAG_TOP_K": "0"}, clear=True), \
                self.assertRaises(ValidationError) as caught:
            Settings()
        for value in SECRETS.values():
            self.assertNotIn(value, str(caught.exception))

    def test_required_secret_checks_read_the_secret_value(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            empty = Settings(AI_RAG_AUTH_MODE="hmac")
            configured = Settings(AI_RAG_AUTH_MODE="hmac", SUPABASE_URL="https://storage.invalid", **SECRETS)
        self.assertEqual(missing_required_setting_names(empty),
                         ["AI_RAG_SERVICE_HMAC_SECRETS", "DATABASE_URL", "SUPABASE_URL"])
        self.assertEqual(missing_required_setting_names(configured), [])

    def test_database_config_repr_hides_the_dsn(self) -> None:
        config = DatabaseConfig(dsn=SECRETS["DATABASE_URL"], production=True, pool_min=1, pool_max=8, ssl_mode=None,
                                ssl_root_cert="", statement_cache_size=100, connect_timeout_seconds=10.0,
                                command_timeout_seconds=600.0, tcp_keepalives_idle_seconds=60)
        self.assertNotIn("db-password-123", repr(config))


class NumericBoundsTests(unittest.TestCase):
    def test_every_numeric_setting_has_a_lower_and_an_upper_bound(self) -> None:
        unbounded = [name for name, field in numeric_fields().items()
                     if not any(isinstance(item, (Ge, Gt)) for item in field.metadata)
                     or not any(isinstance(item, (Le, Lt)) for item in field.metadata)]
        self.assertEqual(unbounded, [])

    def test_defaults_and_the_template_values_are_inside_the_bounds(self) -> None:
        template = env_example_values()
        aliases = {field.validation_alias: name for name, field in numeric_fields().items()}
        numeric = {alias: value for alias, value in template.items() if alias in aliases and value}
        self.assertGreaterEqual(len(numeric), 60)
        with patch.dict(os.environ, numeric, clear=True):
            configured = Settings()  # raises when a template value is out of bounds
        for alias, value in numeric.items():
            self.assertEqual(getattr(configured, aliases[alias]), type(getattr(configured, aliases[alias]))(value))

    def test_broken_values_fail_at_startup(self) -> None:
        for alias, value in {"AI_RAG_TOP_K": "0", "AI_RAG_CHUNK_MAX_CHARS": "-1", "AI_RAG_GENERATION_TEMPERATURE": "3",
                             "AI_RAG_PROVIDER_REQUEST_TIMEOUT_MS": "60000000", "AI_RAG_RETRIEVAL_MIN_SCORE": "1.5",
                             "AI_RAG_DATABASE_COMMAND_TIMEOUT_SECONDS": "0"}.items():
            with self.subTest(alias=alias), patch.dict(os.environ, {alias: value}, clear=True), \
                    self.assertRaises(ValidationError):
                Settings()


if __name__ == "__main__":
    unittest.main()
