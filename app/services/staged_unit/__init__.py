"""Staged unit core: provider response schemas, the instance binder, unit validation and repair merges.

IDM W5 (through ``app.services.orchestration_v2.unit``) and the legacy staged writer share these.
They moved verbatim out of the legacy ``app.services.lesson_author.staged`` package in refactor wave 2
(step 2.1) to be held to mypy strict; the old module paths are aliases of these modules.
"""
