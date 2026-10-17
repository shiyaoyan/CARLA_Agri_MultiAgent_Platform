# CARLA Agri Multi-Agent Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A multi-agent simulation platform that adapts CARLA for the autonomous navigation of
agricultural machinery. A large language model parses natural-language navigation
specifications into structured task descriptions; a large vision model and satellite
imagery generate farmland and road scenes; recorded tractor trajectories drive the
machinery physical configuration and the control interface. Two application cases are
shipped as evidence: **in-field trajectory tracking** across five agricultural operation
types, and **multi-stage agricultural navigation** covering departure, road driving,
field entry, field operation and return.

This repository accompanies the paper *Agent-Generated CARLA-based Simulation Platform
for Autonomous Agricultural Machinery Navigation*.

## Features

- **Natural-language task specification.** Raw text, satellite images, trajectory data
  and JSON/YAML requests are parsed into a validated structured request and a
  dependency-aware execution plan.
- **Satellite-to-CARLA scene generation.** OpenDRIVE, FBX and texture artifacts with road
  topology validation and vegetation placement.
- **Trajectory-driven machinery configuration.** Tractor physics derived from technical
  specifications, historical trajectories and CARLA reference ranges, then refined by
  simulation.
- **Direction-aware control interface.** Pure-pursuit lateral control plus PID
  longitudinal control, with gear-state handling for forward/reverse switching and
  optional Bayesian parameter optimization.
- **Two closed-loop experiments.** In-field trajectory tracking with MAE/RMSE/MaxAE
  evaluation, and full-process navigation with stage-transition, distance and duration
  evaluation.
- **Self-contained runners.** Two single-file scripts reproduce either experiment
  without CrewAI, an LLM key or a project installation. Both need the CARLA Python API,
  numpy, pandas and matplotlib; the Section 3.3 runner additionally needs shapely and the
  CARLA official navigation agents.

## Architecture

One top-level workflow, `AgriculturalSimulationFlow`, runs as a persisted CrewAI `Flow`.

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

| Agent | Name | Responsibility |
|---|---|---|
| PGA | Plan Generation Agent | Parses multimodal input, validates the structured request, produces the execution plan |
| SGA | Scene Generation Agent | Generates CARLA-ready OpenDRIVE/FBX/textures, validates road topology, places vegetation |
| MCA | Machinery Configuration Agent | Selects the machinery blueprint and exports supported tractor parameters |
| CIA | Controller Interface Agent | Prepares pure-pursuit, PID and gear-state control; optional Bayesian tuning |
| EEA | Experiment Execution Agent | Executes trajectory tracking or full-process navigation in CARLA |
| TPS | Trajectory Preprocessing Sub-Agent | Converts raw recorded trajectories into the reference path format |
| PES | Performance Evaluation Subagent | Computes tracking, transition and task metrics from recorded states |

Agent and task definitions live in `agri_carla/crewai_runtime/config/agents.yaml` and
`tasks.yaml`. Every planned step creates a Crew with exactly one selected agent and one
allow-listed domain tool, so CrewAI participates in execution while deterministic project
code remains responsible for simulation and file generation. See
[docs/agent_architecture.md](docs/agent_architecture.md) for the full mapping.

## Repository layout

```text
agri_carla/
  crewai_runtime/   # Flow, Crew runner, tools and YAML definitions
  agents/           # PGA, SGA, MCA and EEA
  controllers/      # CIA, pure-pursuit, PID, gear state, Bayesian optimization
  trajectory/       # TPS and trajectory utilities
  evaluation/       # PES, metrics and path tracking
  simulation/       # CARLA experiment runners and configurations
  machinery/        # tractor physics and templates
  road_generation/  # road topology and CARLA import diagnostics
  execution/        # allow-listed domain dispatcher, no orchestration
  worker/           # same-host CARLA worker protocol and server
roadrunner_mcp/     # SGA OpenDRIVE/FBX/TGA backend
configs/            # paper-case manifest, prompts, request examples and schemas
data/
  trajectory/       # 15-trajectory dataset (TS-Tune, EXP-FTT) and its manifest
  experiment_results/  # published Section 3.2 and 3.3 outputs and the evidence manifest
  maps/             # rebuilt Yantai satellite OpenDRIVE map
docs/               # architecture, installation, reproduction and runner guides
scripts/            # CLI entry points
tests/              # unit, architecture and reconciliation tests
standalone_trajectory_tracking.py    # single-file Section 3.2 runner
standalone_yantai_full_process.py    # single-file Section 3.3 runner
```

## Requirements

- CARLA 0.9.15 server (Linux recommended) and its Python API matching the Python version
  of the execution environment
- Python 3.10-3.13 for the CrewAI management environment, Python 3.8 for the CARLA worker
- `numpy`, `pandas`, `matplotlib`, `pyyaml`, `crewai`, and an OpenAI-compatible LLM
  endpoint for the agent workflow
- An LLM API key for any prompt-first case; the offline verification and the standalone
  runners do not need one

## Installation

The supported deployment uses two isolated Python environments on one CARLA computer,
communicating only over `127.0.0.1`:

```bash
CREWAI_PYTHON=python3.10 bash scripts/setup_crewai_manager.sh
CARLA_PYTHON=python3.8  bash scripts/setup_carla_worker_linux.sh

cp .env.example .env    # insert the LLM API key locally; never commit .env
```

The complete procedure, the CARLA scene import step and a troubleshooting section are in
[docs/installation.md](docs/installation.md).

## Quick start

### 1. Verify the published results offline (no CARLA, no LLM)

```bash
python -m pip install -r requirements.txt
python scripts/audit_paper_results.py
```

This replays the Section 3.2 controller over the recorded vehicle states, recomputes every
metric table from the raw per-tick logs, reconciles the Section 3.3 run and verifies all
evidence hashes. It reports `"audit_passed": true` on the shipped data.

### 2. Run the prompt-first paper cases

```bash
python scripts/run_paper_case.py --list
python scripts/run_paper_case.py --case section_3_2_tracking
bash scripts/run_same_host_crewai_carla.sh --paper-case section_3_2_tracking --execute
```

| Case id | Paper function | Expected route |
|---|---|---|
| `scene_generation` | Satellite-image scene construction | PGA -> SGA |
| `machinery_configuration` | DF1204 parameter-template generation | PGA -> MCA |
| `section_3_2_tracking` | Real-tractor trajectory tracking | PGA -> MCA -> CIA -> EEA |
| `section_3_2_bayesian_optimization` | Controller tuning and independent validation | PGA -> CIA -> EEA |
| `section_3_3_full_process` | Multi-stage agricultural navigation | PGA -> MCA -> CIA -> EEA |

### 3. Re-execute an experiment directly in CARLA

```bash
# Section 3.2, with the frozen published configuration
python scripts/reproduce_tracking.py --all --prepare-only   # validate inputs, no CARLA
python scripts/reproduce_tracking.py --all

# Section 3.3, map and gate validation first, then the full mission
python scripts/run_full_process_experiment.py \
  --config configs/yantai_full_process_agent_request.json --plan-only
python scripts/run_full_process_experiment.py \
  --config configs/yantai_full_process_agent_request.json
```

### 4. Or use a standalone runner

```bash
python standalone_trajectory_tracking.py     # Section 3.2, edit TRAJECTORY_CSV first
python standalone_yantai_full_process.py     # Section 3.3
```

## Datasets

`data/trajectory/` ships 15 measured DF1204 tractor trajectories at 1 Hz, three per
operation type. See [data/trajectory/README.md](data/trajectory/README.md).

| Set | Path | Count | Role |
|---|---|---:|---|
| TS-Tune | `data/trajectory/TS-Tune/` | 10 | Machinery refinement and Bayesian controller tuning |
| EXP-FTT | `data/trajectory/EXP-FTT/` | 5 | Section 3.2 evaluation, one trajectory per Table 1 row |

`data/trajectory/yantai_quanlujing.csv` is the raw recorded example used by the
`section_3_2_tracking` paper case, before TPS standardisation.

TS-Stat, the external 960-trajectory set of Luo et al. (2026) used only to derive
parameter reference ranges, and EXP-MSN, the third-party satellite basemap, are documented
but not redistributed.

## Published results

### Section 3.2 - In-field trajectory tracking

| Trajectory type | Duration (s) | MAE (m) | RMSE (m) | Max AE (m) |
|---|---:|---:|---:|---:|
| Complex operation | 569 | 0.026 | 0.061 | 0.783 |
| Regular operation | 157 | 0.033 | 0.110 | 0.746 |
| Straight-line operation | 62 | 0.020 | 0.056 | 0.279 |
| Bulb turn | 60 | 0.033 | 0.043 | 0.097 |
| Fishtail turn | 84 | 0.083 | 0.139 | 0.788 |

### Section 3.3 - Multi-stage agricultural navigation

| Evaluation item | Result |
|---|---|
| Task completion | Completed |
| Stage transition | Completed |
| Total travel distance | 281.618 m |
| In-field operation path length | 130.299 m |
| Task duration | 458.250 s |

Full precision, the diagnostic valid-sample scope and the per-file column definitions are
in [data/experiment_results/README.md](data/experiment_results/README.md).

## Software usability

**What the platform does.** It converts a natural-language navigation specification into
a running CARLA experiment for agricultural machinery: scene construction from satellite
imagery, tractor physical configuration, direction-aware path tracking, and quantitative
evaluation of either a single in-field trajectory or a whole multi-stage operation.

**What it is validated for.** Low-speed tractor navigation (below roughly 8 km/h) on
farmland and farm roads, with large-radius headland turns and forward/reverse switching.
Only the tractor machinery type is implemented; another type is rejected rather than
silently approximated.

**What it does not do.** The physics model reproduces macroscopic tractor motion, not
soil-tyre interaction: slope-induced slip, operation resistance and drawbar load are not
simulated. Multimodal sensor-data generation is not included. Simulated results support
pre-deployment debugging and workflow verification, and do not replace final field
validation.

**Prerequisites.** A working CARLA 0.9.15 installation is required for any execution. The
prompt-first workflow additionally requires an LLM API key. The offline audit, the
geometry preview and the input-preparation modes run without either.

## Reproducibility

Every published number can be re-derived offline from the shipped raw logs, and the
experiments can be re-run in CARLA from a frozen configuration. The exact commands, the
file-by-file mapping from result file to table column, the deterministic and
machine-dependent parts of the pipeline, and the known limitations are documented in
[docs/reproducing_paper_results.md](docs/reproducing_paper_results.md).


## Tests

```bash
python scripts/audit_paper_results.py
python -m compileall -q agri_carla roadrunner_mcp scripts tests
python -m unittest discover -s tests -v
```

## License

Released under the [MIT License](LICENSE).
