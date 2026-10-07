#!/usr/bin/env sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
python="$repo_root/.venv-dev/bin/python"
if [ ! -x "$python" ]; then
  echo "Missing .venv-dev. Create it from requirements.lock and requirements-dev.lock." >&2
  exit 1
fi

cd "$repo_root"
"$python" -m ruff check app/core tests/test_prd0_security.py tests/test_document_limits.py tests/test_architecture_layers.py
"$python" -m ruff check app/main.py --select F
"$python" -m mypy app/core
"$python" -m pytest -q --cov=app --cov-branch --cov-report=term
"$python" -m pip_audit -r requirements.lock --disable-pip
