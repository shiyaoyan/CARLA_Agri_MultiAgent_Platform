# CARLA Agri Multi-Agent Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[中文说明](README_CN.md)

This project is a CrewAI-managed agricultural autonomous-navigation platform
based on CARLA. It supports agricultural road-scene generation, tractor
configuration, real-trajectory tracking, controller optimization and
full-process agricultural navigation.

## Architecture

The project has one top-level workflow:
`AgriculturalSimulationFlow`, implemented as a persisted CrewAI `Flow`.

```text
Natural language, images, data, or JSON/YAML
                     |
            CrewAI persisted Flow
                     |
                 PGA Crew
      parsing, validation, execution plan
                     |
       SGA / MCA / CIA / EEA Crews
                              |
                         TPS and PES
```

- **PGA** parses multimodal input, validates a structured request and produces
  a dependency-aware execution plan.
- **SGA** generates CARLA-ready OpenDRIVE, FBX and texture artifacts, validates
  road topology and prepares vegetation placement.
- **MCA** selects an existing machinery blueprint and exports supported tractor
  parameters.
- **CIA** prepares pure-pursuit, PID and gear-state controls and optionally runs
  Bayesian controller-parameter optimization.
- **EEA** executes trajectory tracking or full-process navigation in CARLA.
- **TPS** and **PES** are EEA subagents for trajectory processing and
  evaluation. CIA remains a peer agent and is also used by EEA at run time.

Agent and task definitions are stored in
`agri_carla/crewai_runtime/config/agents.yaml` and `tasks.yaml`. Every planned
step creates a Crew with exactly one selected agent and one allow-listed domain
tool. CrewAI therefore participates in execution while deterministic project
code remains responsible for simulation and file generation.

## Single-computer dual-environment installation

The supported deployment uses two isolated Python environments on the same
Linux CARLA computer. CrewAI runs in Python 3.10-3.13, while the installed
CARLA 0.9.15 Python 3.8 egg remains in a dedicated Python 3.8 environment:

```bash
CREWAI_PYTHON=python3.10 bash scripts/setup_crewai_manager.sh
CARLA_PYTHON=python3.8 bash scripts/setup_carla_worker_linux.sh

cp .env.example .env
# Edit .env and insert the LLM API key locally.

bash scripts/run_same_host_crewai_carla.sh \
  configs/yantai_tracking_experiment_request.json
```

Both processes remain on the same CARLA computer and communicate only over
`127.0.0.1`. Non-loopback Worker addresses are rejected. The Python 3.8 Worker
contains no planning or top-level orchestration logic; the persisted CrewAI
Flow in the management environment remains the only workflow owner.
See [`docs/single_computer_deployment_CN.md`](docs/single_computer_deployment_CN.md)
for the complete setup and troubleshooting procedure. The mapping from official
CrewAI abstractions to project code is documented in
[`docs/crewai_compliance_CN.md`](docs/crewai_compliance_CN.md).

## Manager Planning and Local Tools

Without a CARLA worker, the manager can parse and plan tasks and run local
SGA/MCA utilities. This command validates and plans the request without
starting a vehicle experiment:

```bash
python scripts/run_crewai.py \
  --request configs/yantai_tracking_experiment_request.json \
  --plan-only
```

Run from natural language and input files:

```bash
python scripts/run_crewai.py \
  --text "Track the supplied real-tractor trajectory and evaluate tracking performance." \
  --data data/trajectory/yantai_quanlujing.csv \
  --request-template configs/existing_scene_tracking_request.yaml \
  --llm-config configs/llm/gpt_5_6_sol.json \
  --plan-only
```

Tracking and full-process navigation require a running CARLA server and the
same-host launcher from the installation section. `--plan-only` does not call
the CARLA Worker.

## Status and Resume

```bash
python scripts/run_crewai.py --status TASK_ID
python scripts/run_crewai.py --resume TASK_ID
```

Runtime records are stored in:

```text
runtime/crewai_runtime.sqlite3
runtime/runs/<task_id>/
```

The Flow records agent status, progress, errors, artifacts and successful-step
checkpoints. A resumed task skips completed steps.

## Main Commands

```bash
# Road scene
python scripts/generate_road_scene.py \
  --request configs/road_scene_request.yaml \
  --output-dir outputs/road_scene

# Trajectory standardization
python scripts/process_trajectory.py \
  --input data/trajectory/yantai_quanlujing.csv \
  --output outputs/trajectories/yantai_quanlujing_standard.csv

# Tractor template
python scripts/generate_machinery_template.py \
  --type tractor \
  --output outputs/templates/tractor_df1204.json

# Direct CARLA tracking runner (requires the CARLA Python environment and a running simulator)
python scripts/run_tracking_experiment.py \
  --trajectory data/trajectory/yantai_quanlujing.csv \
  --output-dir results/trajectory_tracking

# CrewAI single-trajectory Bayesian-optimization case
bash scripts/run_same_host_crewai_carla.sh \
  configs/yantai_tracking_optimization_request.json

# CrewAI full-process navigation case
bash scripts/run_same_host_crewai_carla.sh \
  configs/yantai_full_process_agent_request.json
```

The optional multi-trajectory optimization and independent-test case uses
`configs/yantai_tracking_dataset_optimization_request.json`. Its two dataset
directories contain documentation only in this public example; add CSV files
before running it. Independent-test metrics are not returned to the Bayesian
optimizer, and an empty optimization set is rejected explicitly.

## Experiment Implementation Map

CrewAI is the only top-level entry, so there is no separate script that bypasses
PGA to launch full-process navigation. The experiment code remains in the
following modules:

| Function | Entry | Implementation |
|---|---|---|
| CrewAI Flow | `scripts/run_crewai.py` | `agri_carla/crewai_runtime/flow.py` |
| Tracking experiment | EEA from the Flow | `agri_carla/simulation/tracking_experiment.py` |
| Bayesian optimization | CIA during tracking | `agri_carla/controllers/controller_optimization.py` |
| Full-process navigation | EEA from the Flow | `agri_carla/simulation/full_process_experiment.py` |
| Scene generation | SGA from the Flow | `agri_carla/road_generation/agent.py`, `roadrunner_mcp/realmap.py` |

## Raw-Image Scene Review Workflow

The first scene-planning request uses an original, unannotated satellite image
plus a natural-language description of the relevant roads, farmland and tree
belts. PGA asks the multimodal model to produce candidate geometry, performs a
second visual verification and deterministic geometry checks, and writes an
overlay preview. SGA is not allowed to execute until the candidate is confirmed.
Before running the example, replace the template's map scale and geographic
anchor with values for the supplied image; image dimensions are read from the
actual attachment.

Third-party satellite imagery is not bundled with the repository. Supply an
owned or redistributable image using `--image /path/to/satellite_image.png`.
The reusable prompt and request template are provided under `configs/prompts/`
and `configs/raw_scene_generation_template.example.json`.

```bash
# 1. First pass: raw image + spatial description -> JSON candidate + overlay preview
python scripts/run_crewai.py \
  --text-file configs/prompts/raw_scene_generation_prompt.example.txt \
  --image /path/to/original_satellite_image.png \
  --request-template configs/raw_scene_generation_template.example.json \
  --llm-config configs/llm/gpt_5_6_sol.json \
  --plan-only

# 2a. Candidate accepted: execute the saved request
python scripts/run_crewai.py \
  --request runtime/runs/<task_id>/structured_request.json \
  --confirm-scene

# 2b. Candidate rejected: rerun planning with clearer text and, only if useful,
# an optional marked correction image while retaining the original under --image
python scripts/run_crewai.py \
  --text "Keep the original image. Move the eastern road centerline slightly west and limit the northern farmland to the two rectangular plots." \
  --image /path/to/original_satellite_image.png \
  --correction-image /path/to/optional_scene_correction.png \
  --request-template configs/raw_scene_generation_template.example.json \
  --llm-config configs/llm/gpt_5_6_sol.json \
  --plan-only
```

Correction images are optional and belong only to the second review round. A
closed semi-transparent polygon corrects a farmland/grass region; an elongated
polygon or band corrects a tree belt. A single point is never interpreted as an
exact tree coordinate. SGA uses only the confirmed structured geometry and
then performs deterministic geometry and topology validation.

## Repository Layout

```text
agri_carla/crewai_runtime/  # Flow, Crew runner, tools and YAML definitions
agri_carla/execution/       # allow-listed domain dispatcher, no orchestration
agri_carla/worker/          # same-host CARLA worker protocol and server
agri_carla/agents/          # PGA, SGA, MCA and EEA implementations
agri_carla/controllers/     # CIA, controllers and Bayesian optimization
agri_carla/trajectory/      # TPS and trajectory utilities
agri_carla/evaluation/      # PES and metrics
agri_carla/simulation/      # CARLA experiment runners
roadrunner_mcp/             # SGA OpenDRIVE/FBX/TGA backend
configs/                    # example requests and schemas
data/experiment_results/    # published Section 3.1 tracking outputs
scripts/                    # CLI entry points
tests/                      # unit and architecture tests
```

The five published Section 3.1 runs, including per-tick records, metric
summaries and diagnostic figures, are documented in
[`data/experiment_results/section_3_1/`](data/experiment_results/section_3_1/README.md).

## Verification

```bash
python scripts/audit_release.py
python -m compileall -q agri_carla scripts tests
python -m unittest discover -s tests -v
```

The project is a research prototype. Generated scenes and simulated navigation
results do not replace final field validation.

## License

This project is released under the [MIT License](LICENSE).
