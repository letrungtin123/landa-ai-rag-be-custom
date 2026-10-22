"""Content hash of the IDM prompt templates (``prompt_templates_hash``).

``IDM_PROMPT_POLICY_VERSION`` ("idm-prompt-1") is bound into every course design, the Node contract
and ``/v1/meta``, and it did not change across the prompt edits since IDM-1. This hash is computed
from the sources of the prompt templates, so any template change gives a new value without touching
that contract: ``/v1/meta`` and the IDM request logs carry it, and
``tests/fixtures/prompt_templates_manifest.json`` records it so a prompt change cannot land without
an explicit manifest update (``tests/golden/test_idm_prompt_manifest.py``).

An entry is a module that holds prompt text only, or ``module:qualname`` for a prompt builder inside a
larger module. Besides the source, an entry covers the plain constants (strings, numbers and
containers of them) in its module's namespace, including imported ones: every one for a module, the
ones it reads for a builder. Sources come from ``inspect`` (universal newlines), so LF and CRLF
checkouts give the same hash.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import inspect
import json
import types
from collections.abc import Callable
from typing import Any, Final

PROMPT_TEMPLATE_SOURCES: Final = (
    # Prompt-only modules: every line is template text or the code that lays it out.
    "app.prompt_safety",
    "app.idm.prompts_common",
    "app.idm.prompts_course",
    "app.idm.prompts_module",
    "app.idm.prompts_unit",
    "app.idm.prompts_judge",
    "app.idm.purpose",
    "app.idm.org_context",
    "app.idm.prompts_org_distill",
    # Rule text that the stages put into prompts.
    "app.idm.fidelity:w2_treatment_rules",
    "app.idm.fidelity:module_fidelity_rules",
    "app.idm.fidelity:writer_fidelity_rules",
    "app.idm.example_roles:role_rule",
    "app.idm.faq_checks:faq_writer_rules",
    "app.idm.faq_checks:faq_prompt_material",
    "app.idm.faq_checks:FaqChecker.plan_hint",
    "app.idm.faq_checks:FaqChecker.hype_hint",
    "app.idm.framework:framework_support_lines",
    "app.idm.quiz_text:quiz_hint",
    "app.idm.worksheet_input:input_hint",
    "app.idm.worksheet_input:reuse_hint",
    "app.idm.preserve_lines:heading_hint",
    "app.idm.preserve_segment:_prompt_line",
    "app.idm.preserve_segment:segment_prompt",
    # The W5 writer prompt and its repair rule lines.
    "app.idm.storyboard:IdmUnitWriter.writer_prompt",
    "app.idm.storyboard:IdmUnitWriter.rule_lines",
    "app.idm.storyboard:IdmUnitWriter.faq_rule_line",
    "app.idm.storyboard:IdmUnitWriter.callout_rule_line",
    "app.idm.storyboard:IdmUnitWriter.framework_rule_line",
    "app.idm.storyboard:IdmUnitWriter.contested_rule_line",
    "app.idm.storyboard:IdmUnitWriter.fidelity_rule_line",
    "app.idm.storyboard:IdmUnitWriter.length_cue_line",
    "app.idm.storyboard:IdmUnitWriter.fact_hint",
    "app.idm.storyboard:IdmUnitWriter.acceptance_rule_lines",
    "app.idm.storyboard_steps:IdmUnitWriterSteps.slot_hint",
)
UNAVAILABLE: Final = "unavailable"


def _canonical(value: Any) -> str | None:
    """A process-independent text of a plain constant (sets sorted), or ``None`` for anything else."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        items = [_canonical(item) for item in value]
        return None if None in items else "[" + ",".join(item for item in items if item is not None) + "]"
    if isinstance(value, (set, frozenset)):
        members = [_canonical(item) for item in value]
        return None if None in members else "{" + ",".join(sorted(item for item in members if item is not None)) + "}"
    if isinstance(value, dict):
        pairs = [(_canonical(key), _canonical(item)) for key, item in value.items()]
        if any(key is None or item is None for key, item in pairs):
            return None
        return "{" + ",".join(sorted(f"{key}:{item}" for key, item in pairs)) + "}"
    return None


def _code_names(code: types.CodeType) -> set[str]:
    names = set(code.co_names)
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            names |= _code_names(constant)
    return names


def _resolve(entry: str) -> tuple[types.ModuleType, Any]:
    module_name, _, qualname = entry.partition(":")
    module = importlib.import_module(module_name)
    target: Any = module
    for part in qualname.split(".") if qualname else ():
        target = getattr(target, part)
    return module, target


def entry_source(entry: str) -> str:
    """The text an entry's digest covers: its source, plus the plain module constants a builder reads."""

    module, target = _resolve(entry)
    source = inspect.getsource(target)
    if target is module:
        # Constants imported from other modules (html limits, purpose defaults) shape the text too.
        names = {name for name in module.__dict__ if not name.startswith("__")}
    else:
        function: Callable[..., Any] = inspect.unwrap(target)
        names = _code_names(function.__code__)
    constants = []
    for name in sorted(names):
        text = _canonical(module.__dict__[name]) if name in module.__dict__ else None
        if text is not None:
            constants.append(f"{name}={text}")
    return source + "\n# constants\n" + "\n".join(constants)


@functools.cache
def prompt_template_digests() -> dict[str, str]:
    """sha256 of each entry of ``PROMPT_TEMPLATE_SOURCES``."""

    return {entry: hashlib.sha256(entry_source(entry).encode("utf-8")).hexdigest()
            for entry in PROMPT_TEMPLATE_SOURCES}


@functools.cache
def prompt_templates_hash() -> str:
    """One hash over every prompt template (``unavailable`` when the sources are not installed)."""

    try:
        digests = prompt_template_digests()
    except (OSError, TypeError):  # sources missing from the installation (inspect cannot read them)
        return UNAVAILABLE
    return hashlib.sha256(json.dumps(digests, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


__all__ = ["PROMPT_TEMPLATE_SOURCES", "entry_source", "prompt_template_digests", "prompt_templates_hash"]
