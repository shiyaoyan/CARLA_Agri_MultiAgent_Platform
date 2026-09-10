#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CREWAI_PYTHON:-python3.10}"
VENV_DIR="${CREWAI_VENV:-$PROJECT_ROOT/.venv-crewai}"

cd "$PROJECT_ROOT"
"$PYTHON_BIN" - <<'PY'
import sys

if not ((3, 10) <= sys.version_info[:2] < (3, 14)):
    raise SystemExit(
        "CrewAI manager requires Python 3.10-3.13; found %s."
        % sys.version.split()[0]
    )
PY
"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_DIR/bin/python" -m pip install -e ".[manager]"
"$VENV_DIR/bin/python" -c \
  "import crewai; print('CrewAI manager ready:', crewai.__version__)"

printf '\nManager environment: %s\n' "$VENV_DIR"
printf 'Create .env from .env.example and insert your real API key locally.\n'
