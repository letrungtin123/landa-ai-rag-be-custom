#!/usr/bin/env sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
python="$repo_root/.venv-dev/bin/python"
if [ ! -x "$python" ]; then
  echo "Missing .venv-dev. Create it from requirements.lock and requirements-dev.lock." >&2
  exit 1
fi

# Code held to the full PRD standard (§19.3). Legacy app/main.py is held to
# pyflakes (F) rules until it is split in PRD-2.
strict_paths="app/core app/infra app/prompt_safety.py app/__main__.py"
strict_tests="tests/test_prd0_security.py tests/test_document_limits.py tests/test_architecture_layers.py tests/test_prd1_runtime.py tests/test_prd1_endpoints.py tests/test_characterization_ingestion.py tests/test_characterization_retrieval_chat.py"

cd "$repo_root"
# shellcheck disable=SC2086 # word splitting of the path lists is intended
"$python" -m ruff check $strict_paths $strict_tests
"$python" -m ruff check app/main.py --select F
# shellcheck disable=SC2086
"$python" -m mypy $strict_paths
"$python" -m pytest -q --cov=app --cov-branch --cov-report=term --cov-fail-under=82
"$python" -m pip_audit -r requirements.lock --disable-pip
