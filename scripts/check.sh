#!/usr/bin/env sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
python="$repo_root/.venv-dev/bin/python"
if [ ! -x "$python" ]; then
  echo "Missing .venv-dev. Create it from requirements.lock and requirements-dev.lock." >&2
  exit 1
fi

# Code held to the full PRD standard (§19.3): ruff (all rules) + mypy strict.
strict_paths="app/__init__.py app/core app/infra app/idm app/prompt_safety.py app/__main__.py app/main.py app/api app/schemas app/repositories app/contracts app/services/__init__.py app/services/runtime.py app/services/meta.py app/services/deadlines.py app/services/text.py app/services/legacy_routes.py app/services/response_contracts.py app/services/provider.py app/services/chat app/services/retrieval app/services/ingestion app/services/orchestration_v2 app/services/staged_unit app/content app/hashing.py app/contract_bridge.py scripts/coverage_by_package.py"
# The legacy lesson-author pipeline (app/services/lesson_author, app/workflows and the legacy modules of app/)
# keeps the mypy "no new errors" baseline (pyproject overrides) and the pyflakes (F) step until PRD-3 deletes it.
# Test code held to the same lint profile: every test directory except the legacy pipeline's
# (tests/unit/legacy) and the cross-language files of tests/ that belong to it.
strict_tests="tests/idm_golden.py tests/idm_contract_bridge.py tests/support tests/golden tests/contract tests/unit/api tests/unit/content tests/unit/contracts tests/unit/core tests/unit/idm tests/unit/infra tests/unit/repositories tests/unit/services tools"

cd "$repo_root"
# shellcheck disable=SC2086 # word splitting of the path lists is intended
"$python" -m ruff check $strict_paths $strict_tests
"$python" -m ruff check app/services app/content app/contracts --select F
# shellcheck disable=SC2086
"$python" -m mypy $strict_paths
"$python" -m pytest -q --cov=app --cov-branch --cov-report=term --cov-fail-under=85
# Coverage of each app package: a report, except the packages with a target (hardening plan §19.6).
"$python" scripts/coverage_by_package.py --fail-under app/idm=90
"$python" -m pip_audit -r requirements.lock --disable-pip
