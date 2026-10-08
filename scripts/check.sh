#!/usr/bin/env sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
python="$repo_root/.venv-dev/bin/python"
if [ ! -x "$python" ]; then
  echo "Missing .venv-dev. Create it from requirements.lock and requirements-dev.lock." >&2
  exit 1
fi

# Code held to the full PRD standard (§19.3): ruff (all rules) + mypy strict.
strict_paths="app/core app/infra app/idm app/prompt_safety.py app/__main__.py app/main.py app/api app/schemas app/repositories app/services/runtime.py app/services/meta.py app/services/deadlines.py app/services/text.py app/hashing.py"
# Service code moved verbatim out of app/main.py in PRD-2: mypy strict, pyflakes (F) lint rules.
# The legacy lesson-author pipeline (app/services/lesson_author) keeps the mypy "no new errors"
# baseline (pyproject overrides) until PRD-3 deletes it.
mypy_strict_paths="app/services/provider.py app/services/chat app/services/retrieval app/services/ingestion app/services/orchestration_v2"
strict_tests="tests/test_prd0_security.py tests/test_document_limits.py tests/test_architecture_layers.py tests/test_prd1_runtime.py tests/test_prd1_endpoints.py tests/test_characterization_ingestion.py tests/test_characterization_retrieval_chat.py tests/idm_golden.py tests/idm_golden_module.py tests/idm_golden_unit.py tests/idm_test_support.py tests/idm_contract_bridge.py tests/test_route_snapshots.py tests/test_repository_sql.py tests/test_response_models.py"
# Every IDM test file is held to the strict lint profile.
for test_file in "$repo_root"/tests/test_idm_*.py; do
  [ "$(basename "$test_file")" = "test_idm_foundation.py" ] || strict_tests="$strict_tests tests/$(basename "$test_file")"
done

cd "$repo_root"
# shellcheck disable=SC2086 # word splitting of the path lists is intended
"$python" -m ruff check $strict_paths $strict_tests
"$python" -m ruff check app/services --select F
# shellcheck disable=SC2086
"$python" -m mypy $strict_paths $mypy_strict_paths
"$python" -m pytest -q --cov=app --cov-branch --cov-report=term --cov-fail-under=85
"$python" -m pip_audit -r requirements.lock --disable-pip
