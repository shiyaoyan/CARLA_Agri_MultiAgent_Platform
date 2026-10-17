# Installation

This guide installs and runs the research release of the CrewAI-managed CARLA platform for autonomous
agricultural machinery navigation, on a single Linux workstation with the project root as the working
directory.

## 1. Supported topology

The supported deployment is **two isolated Python environments on one Linux machine**, sharing one project
checkout:

| Environment | Directory | Python | Role |
|---|---|---|---|
| CrewAI manager | `.venv-crewai` | 3.10 - 3.13 | The single top-level `AgriculturalSimulationFlow`: PGA planning, step dispatch, state, retries, resume |
| CARLA worker | `.venv-carla` | 3.8 exactly | Executes validated CARLA domain steps (default: EEA) through `agri_carla/worker` |

One interpreter cannot satisfy both constraints: CARLA 0.9.15 ships its Python bindings as a CPython 3.8
extension (`carla-0.9.15-py3.8-linux-x86_64.egg`), while `crewai==1.15.12` requires Python 3.10 or newer.

The two processes exchange validated structured task payloads over HTTP on the loopback interface only
(`http://127.0.0.1:8765` by default) using a bearer token; any non-loopback worker URL is rejected on both
sides. There is no single-environment mode and no cross-machine worker; the CARLA server itself is a
separate native process.

## 2. Prerequisites

| Component | Required version | Notes |
|---|---|---|
| Host and CARLA server | Linux x86_64, CARLA 0.9.15 | Source build or packaged Linux release; `CARLA_ROOT` defaults to `$HOME/carla` |
| CARLA Python API | `carla-0.9.15-py3.8-linux-x86_64.egg` | In `<CARLA_ROOT>/PythonAPI/carla/dist/`; `<CARLA_ROOT>/PythonAPI/carla` supplies `BehaviorAgent` |
| Manager / worker Python | 3.10 - 3.13 / 3.8 | Manager and worker run on the same machine; 3.8 must match the egg's minor version |
| Vehicle blueprint | `vehicle.tractor.tractor` | Must already exist in the CARLA installation |
| LLM endpoint | OpenAI-compatible API | A real API key is required for Flow execution |

Dependencies come from `pyproject.toml`, mirrored in `requirements.txt`; on Python 3.13 the manager receives
the newer pins in the last column. The manager extra also adds `crewai==1.15.12` and `python-dotenv>=1.0,<2`;
the worker omits both, and `carla` comes from the egg through `PYTHONPATH`. The OpenAI-compatible LLM client
lives in `agri_carla/llm/openai_compatible.py`, so no separate API SDK is installed.

| Package | Manager (3.10 - 3.12) | Manager on 3.13 | Worker (3.8) |
|---|---|---|---|
| numpy | `>=1.23,<2.0` | `>=2.1,<3` | `>=1.23,<2.0` |
| pandas | `>=1.5,<2.1` | `>=2.2.3,<3` | `>=1.5,<2.1` |
| matplotlib | `>=3.6,<3.8` | `>=3.9,<4` | `>=3.6,<3.8` |
| Pillow | `>=9.2,<11` | `>=11,<12` | `>=9.2,<11` |
| Shapely | `>=1.8.5,<2.1` | `>=2.0.7,<3` | `>=1.8.5,<2.1` |
| networkx | `>=2.8.8,<3.2` | `>=3.4,<4` | `>=2.8.8,<3.2` |
| PyYAML | `>=6,<7` | `>=6,<7` | `>=6,<7` |

## 3. Install both environments

```bash
cd /path/to/CARLA_Agri_MultiAgent_Platform
export CARLA_ROOT="$HOME/carla"

CREWAI_PYTHON=python3.10 bash scripts/setup_crewai_manager.sh
CARLA_PYTHON=python3.8 bash scripts/setup_carla_worker_linux.sh
```

`scripts/setup_crewai_manager.sh` creates `.venv-crewai` and installs `-e ".[manager]"`;
`scripts/setup_carla_worker_linux.sh` creates `.venv-carla` with `pip<25.1` and installs `-e .` only. Both
fail fast on an unsupported interpreter, and both locations and interpreters can be overridden:

| Variable | Used by | Default |
|---|---|---|
| `CREWAI_PYTHON`, `CREWAI_VENV` | `scripts/setup_crewai_manager.sh` | `python3.10`, `<root>/.venv-crewai` |
| `CARLA_PYTHON`, `CARLA_VENV` | `scripts/setup_carla_worker_linux.sh` | `python3.8`, `<root>/.venv-carla` |
| `CARLA_ROOT` | launcher and scripts below | `$HOME/carla` |
| `CARLA_EGG` | launcher and scripts below | `$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.8-linux-x86_64.egg` |
| `CREWAI_MANAGER_PYTHON`, `CARLA_WORKER_PYTHON` | `scripts/run_same_host_crewai_carla.sh` | `.venv-crewai/bin/python`, `.venv-carla/bin/python` |
| `AGRI_CARLA_WORKER_PORT` | `scripts/run_same_host_crewai_carla.sh` | `8765` |

Check that the worker environment imports CARLA and the navigation agents:

```bash
EGG="$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.8-linux-x86_64.egg"
PYTHONPATH="$PWD:$CARLA_ROOT/PythonAPI/carla:$EGG" .venv-carla/bin/python -c \
  "import carla; from agents.navigation.behavior_agent import BehaviorAgent; print('CARLA worker OK')"
```

## 4. Configure `.env`

```bash
cp .env.example .env   # then edit .env and replace the placeholders
```

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | The real LLM API key; this is where the key goes. |
| `AGRI_CREW_MODEL` | CrewAI agent model as `provider/model`; defaults to `openai/gpt-5.6-sol`. |
| `CREWAI_DISABLE_TELEMETRY`, `OTEL_SDK_DISABLED` | Keep CrewAI telemetry and OpenTelemetry export disabled. |
| `AGRI_CARLA_WORKER_TOKEN` | Shared bearer token for the loopback worker interface; use a long random value. |
| `AGRI_CARLA_WORKER_URL` | Fixed same-host endpoint; must stay a loopback URL. |
| `AGRI_CARLA_WORKER_AGENTS` | Agents executed in the CARLA worker; default `EEA`. |

The manager loads `.env` from the project root through python-dotenv. For a non-default LLM endpoint set
`AGRI_CREW_BASE_URL`, or edit the request's `llm` block using `configs/llm/openai_compatible.example.json` as
the field reference; `configs/llm/gpt_5_6_sol.json` is used by the shipped requests.


## 5. Start CARLA and the worker

The launcher starts and stops its own worker; start CARLA once, and start a worker by hand only if it should
outlive a single command:

```bash
cd "$CARLA_ROOT" && ./CarlaUE4.sh -quality-level=Epic -carla-rpc-port=2000

cd /path/to/CARLA_Agri_MultiAgent_Platform
export CARLA_EGG="$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.8-linux-x86_64.egg"
export PYTHONPATH="$PWD:$CARLA_ROOT/PythonAPI/carla:$CARLA_EGG"
export AGRI_CARLA_WORKER_TOKEN="the same token as in .env"
.venv-carla/bin/python scripts/run_carla_worker.py --host 127.0.0.1 --port 8765
```

Confirm the worker is reachable, then that CARLA answers:

```bash
curl -s -H "Authorization: Bearer $AGRI_CARLA_WORKER_TOKEN" http://127.0.0.1:8765/health
# {"status": "ok", "protocol_version": "1.0"}
PYTHONPATH="$PWD:$CARLA_ROOT/PythonAPI/carla:$CARLA_EGG" .venv-carla/bin/python -c \
  "import carla; c=carla.Client('127.0.0.1',2000); c.set_timeout(20); \
   print(c.get_client_version(), c.get_server_version(), c.get_world().get_map().name)"
```

## 6. Run the manager

Planning only, without CARLA or the worker: PGA parses and validates the request and produces the execution
plan; no worker call is made, but the configured LLM is still used, so `OPENAI_API_KEY` is required. Only
direct domain scripts such as `scripts/process_trajectory.py` run fully offline.

```bash
.venv-crewai/bin/python scripts/run_crewai.py --request configs/yantai_tracking_experiment_request.json \
  --plan-only
```

Full same-host execution requires CARLA to be running. The launcher checks both environments and the egg,
starts the worker on `127.0.0.1:8765` with a random token if none is set, waits for its health check, runs
the CrewAI Flow with `AGRI_CARLA_WORKER_URL` pointed at that worker, hands the EEA step to the Python 3.8
environment, and stops the worker it started:

```bash
bash scripts/run_same_host_crewai_carla.sh configs/yantai_tracking_experiment_request.json
bash scripts/run_same_host_crewai_carla.sh configs/yantai_full_process_agent_request.json
bash scripts/run_same_host_crewai_carla.sh --paper-case section_3_2_tracking
```

Prompt-first cases are listed with `scripts/run_paper_case.py --list`. A natural language request uses the same
launcher with `--text` or `--text-file`, a request template, and an LLM configuration such as
`configs/llm/gpt_5_6_sol.json`.

The map required by the request must already be loaded in CARLA, and its name in `experiment.config.carla_map`
must match the imported map: `YantaiTrackingField` for tracking, `YantaiSatelliteRebuilt` for the full-process
case. Scene generation in a request does not update a running CARLA map. Tasks can be inspected or resumed
with `scripts/run_crewai.py --status TASK_ID` and `--resume TASK_ID`; a resumed task that still needs EEA
needs a running worker, started manually as in Section 5.

## 7. Import the CARLA scene

SGA generates OpenDRIVE/FBX/texture artifacts and a CARLA import package; importing them into a running
CARLA process is a separate manual step.

```bash
.venv-carla/bin/python scripts/generate_road_scene.py --request configs/yantai_tracking_scene.json \
  --output-dir outputs/yantai_tracking_scene/road_scene
```

The generator writes a run-time directory
`outputs/yantai_tracking_scene/road_scene/YantaiTrackingField_carla_import/` with the `.xodr`, `.fbx` and
`.tga` files plus the package manifest. Keep other JSON out of `Import/`: CARLA treats it as import manifests.

```bash
mkdir -p "$CARLA_ROOT/Import/YantaiAgri/YantaiTrackingField"
cp outputs/yantai_tracking_scene/road_scene/YantaiTrackingField_carla_import/* \
  "$CARLA_ROOT/Import/YantaiAgri/YantaiTrackingField/"
cd "$CARLA_ROOT" && make import ARGS="--package=YantaiAgri --no-carla-materials"
```

A packaged CARLA binary cannot hot-import raw `.fbx`/`.xodr`: build or obtain a Linux CARLA asset package,
place its `.tar.gz` in `<CARLA_ROOT>/Import/` and run `./ImportAssets.sh` there. Restart CARLA afterwards,
list the maps with `scripts/verify_carla_scene_import.py --list-maps`, then verify the road network:

```bash
export P="$PWD:$CARLA_ROOT/PythonAPI/carla:$CARLA_EGG"
PYTHONPATH="$P" .venv-carla/bin/python scripts/verify_carla_scene_import.py --map YantaiSatelliteRebuilt \
  --expected-xodr data/maps/yantai_satellite/YantaiSatelliteRebuilt.xodr --require-bidirectional \
  --min-junction-movements 6 --output outputs/carla_import_verification.json
```

`--current-world` inspects the level already open in UE Play mode instead of loading a map, and `--inspect-xodr`
performs a static XML-only check that needs no CARLA. `data/maps/yantai_satellite/` ships the road reference
for the paper's map; its FBX bundle must be imported separately.

### CARLA 0.9.15 compatibility notes

- The exporter splits the outermost 0.5 m of each open road end into a short boundary road element,
  controlled by `carla_0915_boundary_segment_m` in the scene request (default `0.5`, `0` disables it), so both
  driving lanes stay connected to the junction; ring roads and roads too short to segment get no such element.
- Vanilla 0.9.15 `OpenDriveActor::BuildRoutes` skips lane entries without a predecessor when it draws editor
  debug routes. Such entries are legal map boundaries, so the editor can still omit the red debug line at an
  open road end even though the XODR lanes and the FBX surface are complete; `editor_route_coverage.skipped_lanes`
  reports that behaviour rather than a map defect, and road marks or invented turnaround loops do not fix it.
- Accept route completeness from `scripts/verify_carla_scene_import.py` (`passed`, all declared driving lanes
  present, bidirectional transitions, junction movements). The release never creates a tractor blueprint:
  `vehicle.tractor.tractor` must already exist in CARLA.

## 8. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Cannot connect to CARLA worker at http://127.0.0.1:8765`, or the worker exits at startup | The worker is not running, listens on another port, or failed to start. Run it as in Section 5 with the same `AGRI_CARLA_WORKER_PORT` and read the traceback in `runtime/carla_worker.log`. |
| `CARLA worker HTTP 401: Unauthorized` | Token mismatch: `AGRI_CARLA_WORKER_TOKEN` must be identical in the worker process, the manager environment and `.env`. |
| `must use a same-host loopback URL` or `must bind to a loopback address` | A non-loopback address was configured. Use `http://127.0.0.1:8765` and `--host 127.0.0.1`; remote workers are not supported. |
| `CrewAI environment not found`, `CARLA Python 3.8 environment not found`, `CARLA egg not found` | A launcher pre-flight check failed. Re-run Section 3, or set `CREWAI_MANAGER_PYTHON`, `CARLA_WORKER_PYTHON`, `CARLA_EGG`. |
| `ModuleNotFoundError: No module named 'carla'`, or the egg fails to load | The egg path is missing from `PYTHONPATH`. Export `PYTHONPATH="$PWD:$CARLA_ROOT/PythonAPI/carla:$CARLA_EGG"` in the worker shell. |
| `The CARLA worker requires Python 3.8; found 3.x` | The egg is built for CPython 3.8. Install a 3.8 interpreter, pass `CARLA_PYTHON=python3.8`, and never install `.[manager]` into `.venv-carla`. |
| `CrewAI manager requires Python 3.10-3.13` | Use a 3.10 - 3.13 interpreter for `CREWAI_PYTHON`; CrewAI cannot run on 3.8. |
| `make import` fails, the map is missing, or raw `.fbx`/`.xodr` cannot be imported | Incomplete or misplaced bundle, or a packaged CARLA binary that accepts only prepared asset packages through `ImportAssets.sh`. Copy the whole `<MapName>_carla_import/` content into `<CARLA_ROOT>/Import/<Package>/<Map>/`, keep other JSON out of `Import/`, restart CARLA, or use a source build for raw imports. |
| Import succeeds but the request cannot find the map, or the tractor fails to spawn | The name in `experiment.config.carla_map` differs from the imported map name, or `vehicle.tractor.tractor` is absent from the installation. List maps with `scripts/verify_carla_scene_import.py --list-maps`. |
| `PermissionError` on `~/.cache/matplotlib` while plotting | The home directory is not writable, so the font cache cannot be built. Export `MPLCONFIGDIR=/tmp/mplconfig` (any writable directory) before running the manager or a standalone runner. |
| Plotting fails with no display | Use a non-interactive backend: export `MPLBACKEND=Agg`. The standalone runners set this default themselves. |
| CARLA fails or hangs at startup on a headless machine | The server needs a rendering context. Run it in a virtual framebuffer, for example `xvfb-run -a ./CarlaUE4.sh -quality-level=Low -carla-rpc-port=2000`, or use the off-screen/no-render option of your build's launcher (see `./CarlaUE4.sh --help`), then confirm the RPC port is listening. |
| OpenDRIVE import warnings, or red debug lines missing | Expected for open road ends under 0.9.15. Lane content and FBX geometry are complete; rely on the verification report rather than the editor drawing (Section 7). |

Agent roles, files and interfaces are described in `docs/agent_architecture.md`.
