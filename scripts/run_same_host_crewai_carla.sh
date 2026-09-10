#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANAGER_PY="${CREWAI_MANAGER_PYTHON:-$PROJECT_ROOT/.venv-crewai/bin/python}"
WORKER_PY="${CARLA_WORKER_PYTHON:-$PROJECT_ROOT/.venv-carla/bin/python}"
WORKER_PORT="${AGRI_CARLA_WORKER_PORT:-8765}"
CARLA_ROOT="${CARLA_ROOT:-$HOME/carla}"
CARLA_EGG="${CARLA_EGG:-$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.8-linux-x86_64.egg}"
WORKER_PYTHONPATH="$PROJECT_ROOT:$CARLA_ROOT/PythonAPI/carla:$CARLA_EGG"
MANAGER_PYTHONPATH="$PROJECT_ROOT"

if [[ "$#" -eq 0 ]]; then
  MANAGER_ARGS=(--request configs/yantai_tracking_experiment_request.json)
elif [[ "$1" == --* ]]; then
  MANAGER_ARGS=("$@")
else
  REQUEST_FILE="$1"
  shift
  MANAGER_ARGS=(--request "$REQUEST_FILE" "$@")
fi

if [[ ! -x "$MANAGER_PY" ]]; then
  printf 'CrewAI environment not found: %s\n' "$MANAGER_PY" >&2
  printf 'Run scripts/setup_crewai_manager.sh first.\n' >&2
  exit 1
fi

if [[ ! -x "$WORKER_PY" ]]; then
  printf 'CARLA Python 3.8 environment not found: %s\n' "$WORKER_PY" >&2
  printf 'Run scripts/setup_carla_worker_linux.sh first.\n' >&2
  exit 1
fi

if [[ ! -f "$CARLA_EGG" ]]; then
  printf 'CARLA egg not found: %s\n' "$CARLA_EGG" >&2
  exit 1
fi

cd "$PROJECT_ROOT"
if [[ -n "${REQUEST_FILE:-}" ]]; then
  if [[ ! -f "$REQUEST_FILE" ]]; then
    printf 'Request file not found: %s\n' "$REQUEST_FILE" >&2
    exit 1
  fi
fi

if [[ -z "${AGRI_CARLA_WORKER_TOKEN:-}" ]]; then
  AGRI_CARLA_WORKER_TOKEN="$("$MANAGER_PY" -c 'import secrets; print(secrets.token_urlsafe(32))')"
  export AGRI_CARLA_WORKER_TOKEN
fi

mkdir -p runtime
env PYTHONPATH="$WORKER_PYTHONPATH" "$WORKER_PY" scripts/run_carla_worker.py \
  --host 127.0.0.1 \
  --port "$WORKER_PORT" \
  --token-env AGRI_CARLA_WORKER_TOKEN \
  --tasks-dir runtime/carla_worker \
  >runtime/carla_worker.log 2>&1 &
WORKER_PID=$!
trap 'kill "$WORKER_PID" 2>/dev/null || true' EXIT INT TERM

WORKER_URL="http://127.0.0.1:$WORKER_PORT"
WORKER_READY=0
for _ in $(seq 1 30); do
  if ! kill -0 "$WORKER_PID" 2>/dev/null; then
    printf 'CARLA worker stopped during startup. See runtime/carla_worker.log.\n' >&2
    exit 1
  fi
  if env PYTHONPATH="$MANAGER_PYTHONPATH" "$MANAGER_PY" - "$WORKER_URL" <<'PY' >/dev/null 2>&1
import json
import os
import sys
import urllib.request

request = urllib.request.Request(sys.argv[1] + "/health")
request.add_header(
    "Authorization",
    "Bearer " + os.environ["AGRI_CARLA_WORKER_TOKEN"],
)
with urllib.request.urlopen(request, timeout=1.0) as response:
    payload = json.loads(response.read().decode("utf-8"))
if payload.get("status") != "ok":
    raise SystemExit(1)
PY
  then
    WORKER_READY=1
    break
  fi
  sleep 1
done

if [[ "$WORKER_READY" -ne 1 ]]; then
  printf 'CARLA worker health check timed out. See runtime/carla_worker.log.\n' >&2
  exit 1
fi

env PYTHONPATH="$MANAGER_PYTHONPATH" "$MANAGER_PY" -c \
  "import crewai; print('CrewAI manager:', crewai.__version__)"
env \
  PYTHONPATH="$MANAGER_PYTHONPATH" \
  AGRI_CARLA_WORKER_URL="$WORKER_URL" \
  AGRI_CARLA_WORKER_AGENTS="${AGRI_CARLA_WORKER_AGENTS:-EEA}" \
  "$MANAGER_PY" scripts/run_crewai.py \
  "${MANAGER_ARGS[@]}" \
  --worker-token-env AGRI_CARLA_WORKER_TOKEN
