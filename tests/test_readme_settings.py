"""README facts that must follow the code: the settings table and the coverage gate.

The settings table is generated from ``app.core.config.Settings``; rewrite it with
``LANDA_UPDATE_README=1 python -m pytest tests/test_readme_settings.py`` after a settings change.
"""

from __future__ import annotations

import os
import re
import types
import unittest
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin

from annotated_types import Ge, Gt, Le, Lt, MaxLen, MinLen
from pydantic import SecretStr
from pydantic.fields import FieldInfo

from app.core.config import Settings

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
START, END = "<!-- settings-table:start -->", "<!-- settings-table:end -->"
UPDATE_ENV = "LANDA_UPDATE_README"
# Defaults computed when the settings load; the table states the rule instead of this machine's value.
COMPUTED_DEFAULTS = {
    "environment": "`development` (or `NODE_ENV` when `AI_RAG_ENV` is unset)",
    "cpu_workers": "CPU count - 1, between 1 and 4",
}


def type_text(annotation: Any) -> str:
    if annotation is SecretStr:
        return "secret"
    if get_origin(annotation) is Literal:
        return " \\| ".join(f"`{value}`" for value in get_args(annotation))
    if isinstance(annotation, types.UnionType) or get_origin(annotation) is Union:
        parts = [type_text(item) for item in get_args(annotation) if item is not type(None)]
        return " \\| ".join(parts) + " \\| unset"
    return {bool: "bool", int: "int", float: "number", str: "text"}[annotation]


def default_text(name: str, field: FieldInfo) -> str:
    if name in COMPUTED_DEFAULTS:
        return COMPUTED_DEFAULTS[name]
    value = field.default
    if isinstance(value, SecretStr):
        value = value.get_secret_value()
    if value is None:
        return "unset"
    if value == "":
        return "empty"
    if isinstance(value, bool):
        return f"`{str(value).lower()}`"
    return f"`{value}`"


def bounds_text(field: FieldInfo) -> str:
    low = next((f">= {item.ge}" for item in field.metadata if isinstance(item, Ge)), None) \
        or next((f"> {item.gt}" for item in field.metadata if isinstance(item, Gt)), None)
    high = next((f"<= {item.le}" for item in field.metadata if isinstance(item, Le)), None) \
        or next((f"< {item.lt}" for item in field.metadata if isinstance(item, Lt)), None)
    length = [f"length >= {item.min_length}" for item in field.metadata if isinstance(item, MinLen)] + \
             [f"length <= {item.max_length}" for item in field.metadata if isinstance(item, MaxLen)]
    return ", ".join([part for part in (low, high) if part] + length)


def settings_table() -> str:
    rows = ["| Variable | Type | Default | Bounds |", "|---|---|---|---|"]
    for name, field in Settings.model_fields.items():
        rows.append(f"| `{field.validation_alias}` | {type_text(field.annotation)} | {default_text(name, field)} "
                    f"| {bounds_text(field)} |")
    return "\n".join(rows)


def readme_with_table(text: str) -> str:
    before, rest = text.split(START, 1)
    _old, after = rest.split(END, 1)
    return f"{before}{START}\n{settings_table()}\n{END}{after}"


class ReadmeTests(unittest.TestCase):
    maxDiff = None

    def test_settings_table_is_generated_from_the_settings_module(self) -> None:
        text = README.read_text(encoding="utf-8")
        expected = readme_with_table(text)
        if os.environ.get(UPDATE_ENV) == "1" and expected != text:
            README.write_text(expected, encoding="utf-8", newline="\n")
            text = expected
        self.assertEqual(text, expected, f"Settings changed: run {UPDATE_ENV}=1 python -m pytest {Path(__file__).name}")
        self.assertEqual(text.count("| `AI_RAG_") + text.count("| `DATABASE_URL`") + text.count("| `SUPABASE_"),
                         len(Settings.model_fields))

    def test_coverage_gate_in_the_readme_matches_both_gate_scripts(self) -> None:
        gates = {re.search(r"--cov-fail-under=(\d+)", (REPO_ROOT / "scripts" / name).read_text(encoding="utf-8"))
                 for name in ("check.ps1", "check.sh")}
        values = {match.group(1) for match in gates if match is not None}
        self.assertEqual(len(values), 1)
        self.assertIn(f"failing under {values.pop()}%", README.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
