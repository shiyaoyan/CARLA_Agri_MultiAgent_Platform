# Reproducing the paper results

This guide explains, step by step, how to reproduce the two application cases of the
paper and how to verify the results that are shipped with this repository. It also
states precisely which parts of the pipeline are deterministic and which parts depend on
the local CARLA installation or on a third-party language model.

## Contents

- [What is reproduced](#what-is-reproduced)
- [Reproduction levels](#reproduction-levels)
- [Level A - offline verification (no CARLA, no LLM)](#level-a---offline-verification-no-carla-no-llm)
- [Level B - prompt-first agent workflow](#level-b---prompt-first-agent-workflow)
- [Level C - direct CARLA re-execution](#level-c---direct-carla-re-execution)
- [Level D - standalone single-file runners](#level-d---standalone-single-file-runners)
- [Mapping results to the paper tables](#mapping-results-to-the-paper-tables)
- [Reproducibility boundaries](#reproducibility-boundaries)

## What is reproduced

| Paper item | Platform function | Shipped evidence |
|---|---|---|
| Table 1, Section 3.2 | In-field trajectory tracking for five operation types | `data/experiment_results/section_3_2_tracking/` |
| Table 2, Section 3.3 | Multi-stage agricultural navigation | `data/experiment_results/section_3_3/yantai_satellite/run_20261015T010039Z_f27a13ca/` |
| Section 3.1, data | TS-Stat / TS-Tune / EXP-FTT / EXP-MSN data sets | `data/trajectory/` and [its README](../data/trajectory/README.md) |

The platform itself is exercised through five prompt-first *paper cases* declared in
`configs/paper_cases.json`:

| Case id | Paper function | Expected agent route |
|---|---|---|
| `scene_generation` | Satellite-image agricultural scene construction | PGA -> SGA |
| `machinery_configuration` | DF1204 parameter-template generation | PGA -> MCA |
| `section_3_2_tracking` | Real-tractor trajectory tracking | PGA -> MCA -> CIA -> EEA (TPS/PES inside EEA) |
| `section_3_2_bayesian_optimization` | Multi-trajectory controller tuning and independent validation | PGA -> CIA -> EEA |
| `section_3_3_full_process` | Multi-stage agricultural navigation | PGA -> MCA -> CIA -> EEA (PES inside EEA) |

```bash
python scripts/run_paper_case.py --list
```

## Reproduction levels

| Level | Needs CARLA | Needs an LLM key | What it establishes |
|---|---|---|---|
| A - offline verification | no | no | The shipped metrics, phase summaries, tables and evidence hashes are exactly derivable from the shipped raw logs. |
| B - prompt-first agent workflow | for execution only | yes | Natural language reaches the structured request and the agent route through the real LLM and PGA. |
| C - direct CARLA re-execution | yes | no | The tracking and full-process runners can be re-driven end to end from the frozen configuration. |
| D - standalone runners | yes | no | The same experiment logic runs from a single file with no package installation. |

Level A is the strongest check that does not depend on the local machine. Levels B-D
re-create the experiment and are expected to agree with the shipped numbers within the
tolerances discussed in [Reproducibility boundaries](#reproducibility-boundaries).

## Level A - offline verification (no CARLA, no LLM)

Install only the numerical dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the evidence reconciliation. It replays the Section 3.2 controller over the recorded
vehicle states, recomputes every metric table from the raw per-tick logs, reconciles the
Section 3.3 run against the project code, and verifies every dataset and evidence hash in
`data/experiment_results/manifest.json`:

```bash
python scripts/audit_paper_results.py
```

On the shipped data this prints a report starting with `"audit_passed": true`.
To inspect the numbers without any controller replay:

```bash
# The 15-trajectory dataset manifest
column -s, -t data/trajectory/manifest.csv | less -S

# Recompute one run's metrics only
python - <<'PY'
from agri_carla.evaluation.metrics import summarize_tick_records
print(summarize_tick_records(
    "data/experiment_results/section_3_2_tracking/EXP01_complex_operation/tick_records.csv"))
PY
```

## Level B - prompt-first agent workflow

The published inputs are complete natural-language prompts stored under
`configs/prompts/paper_cases/`. The agent route is validated before anything is executed.

```bash
cp .env.example .env      # then insert your LLM API key locally
python scripts/run_paper_case.py --case section_3_2_tracking
```

Each run writes the prompt actually sent, the generated structured request, the
validation report and the execution plan into `runtime/runs/<task_id>/`. 



```bash
python scripts/run_paper_case.py \
  --case scene_generation \
  --image /path/to/your_satellite_image.png
```

Executing a case in CARLA requires the same-host worker described in
[docs/installation.md](installation.md):

```bash
bash scripts/run_same_host_crewai_carla.sh --paper-case section_3_2_tracking --execute
```

## Level C - direct CARLA re-execution

The direct runners bypass the agent layer and re-drive a fixed configuration. 

### Section 3.2

`scripts/reproduce_tracking.py` reads the frozen configuration from
`data/experiment_results/section_3_2_config.json` and refuses to start if the tractor
physics currently produced by the code differ from the archived snapshot.

```bash
# Standardise and inspect the inputs without CARLA
python scripts/reproduce_tracking.py --all --prepare-only

# Re-run all five EXP-FTT evaluation cases
python scripts/reproduce_tracking.py --all

# Re-run a single case, for example the bulb turn
python scripts/reproduce_tracking.py --case EXP04_bulb_turn
```

Outputs are written to a fresh timestamped directory under `results/paper_tracking/`; the
script refuses to write into `data/experiment_results/`.

The equivalent direct entry point, which accepts explicit settings instead of the frozen
snapshot, is:

```bash
python scripts/run_tracking_experiment.py \
  --trajectory data/trajectory/EXP-FTT/EXP01_complex_operation.csv \
  --carla-map YantaiTrackingField \
  --output-dir results/trajectory_tracking
```

### Section 3.2 controller tuning

```bash
python scripts/run_tracking_experiment.py \
  --parameter-optimization-set-dir data/trajectory/TS-Tune \
  --independent-validation-set-dir data/trajectory/EXP-FTT \
  --auto-optimize-controller --optimization-trials 24 \
  --carla-map YantaiTrackingField \
  --output-dir results/trajectory_tracking_optimized
```



### Section 3.3

Validate the runtime map and the road gates without spawning a vehicle:

```bash
python scripts/run_full_process_experiment.py \
  --config configs/yantai_full_process_agent_request.json --plan-only
```

Run the full mission:

```bash
python scripts/run_full_process_experiment.py \
  --config configs/yantai_full_process_agent_request.json \
  --output-dir results/full_process_navigation/yantai_satellite
```

Preview the mission geometry completely offline:

```bash
python scripts/preview_yantai_full_process.py --output-dir outputs/yantai_full_process_preview
```

### Section 3.1 scene and machinery utilities

```bash
python scripts/generate_road_scene.py \
  --request configs/road_scene_request.yaml --output-dir outputs/road_scene

python scripts/generate_machinery_template.py \
  --type tractor --output outputs/templates/tractor_df1204.json

python scripts/process_trajectory.py \
  --input data/trajectory/yantai_quanlujing.csv \
  --output outputs/trajectories/yantai_quanlujing_standard.csv
```

## Level D - standalone single-file runners

`standalone_trajectory_tracking.py` and `standalone_yantai_full_process.py` embed the
whole experiment — preprocessing, controller, tractor physics, recording, metrics and
plots — in one file each. Neither needs CrewAI, an LLM API key, an external request JSON
or a project installation.  See
[docs/standalone_runners.md](standalone_runners.md).

##  results to the paper tables

### Table 1 - lateral tracking errors



| Trajectory type | Input file | Duration (s) | MAE (m) | RMSE (m) | Max AE (m) |
|---|---|---:|---:|---:|---:|
| Complex operation | `EXP-FTT/EXP01_complex_operation.csv` | 569 | 0.026 | 0.061 | 0.783 |
| Regular operation | `EXP-FTT/EXP02_regular_operation.csv` | 157 | 0.033 | 0.110 | 0.746 |
| Straight-line operation | `EXP-FTT/EXP03_straight_line.csv` | 62 | 0.020 | 0.056 | 0.279 |
| Bulb turn | `EXP-FTT/EXP04_bulb_turn.csv` | 60 | 0.033 | 0.043 | 0.097 |
| Fishtail turn | `EXP-FTT/EXP05_fishtail_turn.csv` | 84 | 0.083 | 0.139 | 0.788 |



### Table 2 - multi-stage agricultural navigation

| Evaluation item | Source file | Value |
|---|---|---|
| Task completion | `full_process_task_evaluation.csv` | Completed |
| Stage transition | `full_process_task_evaluation.csv` | Completed |
| Total travel distance | `full_process_task_evaluation.csv` | 281.618 m |
| In-field operation path length | `full_process_task_evaluation.csv` | 130.299 m |
| Task duration | `full_process_task_evaluation.csv` | 458.250 s |



