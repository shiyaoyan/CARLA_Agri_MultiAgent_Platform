#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CARLA_PYTHON:-python3.8}"
VENV_DIR="${CARLA_VENV:-$PROJECT_ROOT/.venv-carla}"

cd "$PROJECT_ROOT"
"$PYTHON_BIN" - <<'PY'
import sys

if sys.version_info[:2] != (3, 8):
    raise SystemExit(
        "The CARLA worker requires Python 3.8; found %s."
        % sys.version.split()[0]
    )
PY
"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade "pip<25.1" setuptools wheel
"$VENV_DIR/bin/python" -m pip install -e .

printf '\nCARLA worker environment: %s\n' "$VENV_DIR"
printf 'CrewAI is intentionally not installed in this Python 3.8 environment.\n'
