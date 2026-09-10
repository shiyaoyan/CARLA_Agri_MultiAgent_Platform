# Linux CARLA Deployment and Experiment Guide

This guide targets the previously used environment:

- Linux;
- CARLA 0.9.15;
- Python 3.8;
- CARLA root: `~/carla`;
- tractor blueprint: `vehicle.tractor.tractor`;
- custom map: `YantaiTrackingField`.

## 1. Transfer the project

Copy the complete `CARLA_Agri_MultiAgent_Platform` directory to the CARLA
computer. The directory used for commands must directly contain:

```text
pyproject.toml
README.md
agri_carla/
roadrunner_mcp/  # SGA OpenDRIVE/FBX/TGA backend
scripts/
configs/
data/
```

Recommended location:

```text
~/carla/PythonAPI/CARLA_Agri_MultiAgent_Platform
```


Example transfer from another Linux/macOS computer:

```bash
tar \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='results' \
  --exclude='outputs' \
  -czf CARLA_Agri_MultiAgent_Platform.tar.gz \
  CARLA_Agri_MultiAgent_Platform

scp CARLA_Agri_MultiAgent_Platform.tar.gz \
  ysy@CARLA_COMPUTER_IP:~/carla/PythonAPI/
```

On the CARLA computer:

```bash
cd ~/carla/PythonAPI
tar -xzf CARLA_Agri_MultiAgent_Platform.tar.gz
```

Verify the project root:

```bash
export CARLA_ROOT="$HOME/carla"
export PROJECT_ROOT="$CARLA_ROOT/PythonAPI/CARLA_Agri_MultiAgent_Platform"
cd "$PROJECT_ROOT"
pwd
test -f pyproject.toml && echo "project root OK"
```

If `pyproject.toml` is missing, the project is probably nested one directory
deeper. Enter the inner directory before continuing.

## 2. Create the Python 3.8 environment

The CARLA egg is compiled for Python 3.8, so the project environment must use
the same Python minor version.

```bash
cd "$PROJECT_ROOT"
python3.8 --version
python3.8 -m venv .venv
source .venv/bin/activate
python --version
python -m pip install --upgrade "pip<25.1" "setuptools>=61" wheel
python -m pip install -e .
```

The current project metadata accepts Python 3.8:

```bash
grep requires-python pyproject.toml
```

Expected output:

```text
requires-python = ">=3.8"
```

Use `python -m pip` instead of a bare `pip` command to ensure that packages are
installed into `.venv`.

## 3. Configure the CARLA Python paths

The CARLA egg supplies the `carla` module, while
`PythonAPI/carla/agents/navigation` supplies `BehaviorAgent` for the
full-process case. Both locations are required.

```bash
export CARLA_EGG="$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.8-linux-x86_64.egg"
export PYTHONPATH="$PROJECT_ROOT:$CARLA_ROOT/PythonAPI/carla:$CARLA_EGG${PYTHONPATH:+:$PYTHONPATH}"
```

Repeat the exports after opening a new terminal. Verify all three imports:

```bash
python -c "import carla; print('CARLA module:', carla.__file__)"
python -c "import shapely; print('Shapely:', shapely.__version__)"
python -c "import networkx; print('NetworkX:', networkx.__version__)"
python -c "from agents.navigation.behavior_agent import BehaviorAgent; print('BehaviorAgent OK')"
python -c "import agri_carla; print('Project package OK')"
```


## 4. Verify the required input files

The final real trajectory is:

```text
data/trajectory/yantai_quanlujing.csv
```

Check the current experiment requests:

```bash
ls configs/yantai_tracking_experiment_request.json
ls configs/yantai_tracking_optimization_request.json
ls configs/yantai_full_process_agent_request.json
ls data/trajectory/yantai_quanlujing.csv
```

Files below `outputs/` and `results/` are generated during execution and do
not need to exist before the first run.

Run a CARLA-independent trajectory preprocessing smoke test:

```bash
python scripts/process_trajectory.py \
  --input data/trajectory/yantai_quanlujing.csv \
  --output outputs/preflight/yantai_quanlujing_standard.csv
```

Optional project regression test:

```bash
python -m unittest discover -s tests -v
```

## 5. Start and inspect CARLA

Open a separate terminal:

```bash
cd "$CARLA_ROOT"
./CarlaUE4.sh -quality-level=Epic -carla-rpc-port=2000
```

Keep this terminal running. In the project terminal, activate `.venv`, restore
`PYTHONPATH`, and test the connection:

```bash
python -c "import carla; c=carla.Client('localhost',2000); c.set_timeout(20); print('client=',c.get_client_version()); print('server=',c.get_server_version()); print('map=',c.get_world().get_map().name)"
```

List available maps:

```bash
python -c "import carla; c=carla.Client('localhost',2000); c.set_timeout(20); print('\n'.join(c.get_available_maps()))"
```

Check the tractor blueprint:

```bash
python -c "import carla; c=carla.Client('localhost',2000); c.set_timeout(20); w=c.get_world(); print(w.get_blueprint_library().find('vehicle.tractor.tractor').id)"
```

If the custom map and blueprint already exist, skip Section 6.

## 6. Generate or import the map only when necessary

Generate the road artifacts:

```bash
python scripts/generate_road_scene.py \
  --request configs/yantai_tracking_scene.json \
  --output-dir outputs/yantai_tracking_scene/road_scene
```

Confirm that
`outputs/yantai_tracking_scene/road_scene/YantaiTrackingField.topology_report.json`
reports `"valid": true`.

The import bundle is:

```text
outputs/yantai_tracking_scene/road_scene/YantaiTrackingField_carla_import/
```

It contains same-name `.fbx` and `.xodr` files plus the texture. SGA generates
these files but does not import them into a running CARLA process.

### CARLA source build

Copy the bundle into a package folder under `CARLA_ROOT/Import`, then run the
CARLA import command:

```bash
mkdir -p "$CARLA_ROOT/Import/YantaiAgri/YantaiTrackingField"
cp outputs/yantai_tracking_scene/road_scene/YantaiTrackingField_carla_import/* \
  "$CARLA_ROOT/Import/YantaiAgri/YantaiTrackingField/"
cd "$CARLA_ROOT"
make import ARGS="--package=YantaiAgri --no-carla-materials"
```

Restart CARLA after the import and verify that the map is listed.

### Packaged CARLA binary

Raw `.fbx/.xodr` files cannot be hot-imported directly into the packaged
simulator. Build or obtain a Linux CARLA asset package first, place its
`.tar.gz` in `CARLA_ROOT/Import`, then run:

```bash
cd "$CARLA_ROOT/Import"
./ImportAssets.sh
```

If `YantaiTrackingField` is already available, do not re-import it.

The project also does not convert a tractor mesh into a CARLA vehicle blueprint
at runtime. `vehicle.tractor.tractor` must already be imported and configured
in CARLA.

## 7. Optional vegetation placement

Generate a placement plan without changing CARLA:

```bash
python scripts/spawn_scene_vegetation.py \
  --request runtime/runs/<task_id>/structured_request.json \
  --dry-run
```

List available static vegetation blueprints before spawning:

```bash
python scripts/spawn_scene_vegetation.py \
  --request runtime/runs/<task_id>/structured_request.json \
  --list-static-blueprints outputs/yantai_tracking_scene/vegetation/static_blueprints.json
```

Place vegetation in the imported map:

```bash
python scripts/spawn_scene_vegetation.py \
  --request runtime/runs/<task_id>/structured_request.json \
  --spawn
```

Replace `<task_id>` with the PGA scene-planning task identifier. `--spawn` is
required when the reviewed request retains `dry_run: true`.
The grass debug-region mode is non-colliding. Tree collision behavior depends
on the selected CARLA static blueprint. Before spawning, confirm that the
vegetation request uses the same image dimensions, meters-per-pixel,
geographic anchor and world offset as the imported map; otherwise the
polygons will be placed at incorrect world coordinates.

## 8. Run Case 1: fixed-parameter trajectory tracking

First run a single fixed-parameter experiment:

```bash
python scripts/run_tracking_experiment.py \
  --carla-map YantaiTrackingField \
  --trajectory data/trajectory/yantai_quanlujing.csv \
  --processed-csv outputs/yantai_tracking_scene/trajectory/yantai_quanlujing_standard.csv \
  --vehicle-bp vehicle.tractor.tractor \
  --output-dir results/yantai_tracking_scene/tracking \
  --map-offset-x 50 \
  --map-offset-y -10
```

The real trajectory currently requires the default Y-axis flip. Do not add
`--no-flip-y` unless the input path is already expressed in CARLA world
coordinates.

The output is written to a timestamped subdirectory and includes:

```text
tick_records.csv
diagnostic_events.csv
phase_summary.csv
tracking_metrics_summary.csv
01_tracking_error_metrics.png
02_spatial_trajectory.png
README_diagnostics.txt
```

To execute the complete CrewAI Flow, including PGA planning, SGA output, MCA
template generation, CIA, EEA, TPS, and PES, use a Python 3.10+ manager
environment and the same-host Python 3.8 Worker described in
`docs/single_computer_deployment_CN.md`:

```bash
bash scripts/run_same_host_crewai_carla.sh \
  configs/yantai_tracking_experiment_request.json
```

The map must already be imported before this command. Scene generation inside
the request does not update the running CARLA map.

## 9. Run CIA Bayesian controller self-optimization

For the first smoke test, temporarily set `max_trials` to `2` in
`configs/yantai_tracking_optimization_request.json`. After confirming that two
trial folders and the summary are generated, restore `max_trials` to `24`.

Formal run:

```bash
bash scripts/run_same_host_crewai_carla.sh \
  configs/yantai_tracking_optimization_request.json
```

Important outputs under
`results/yantai_tracking_scene/tracking_optimized/controller_optimization/`
include:

```text
controller_optimization_trials.csv
controller_optimization_convergence.png
best_controller_config.json
controller_optimization_summary.json
trial_*/
confirmation_best/
```

Use the baseline and best-confirmation results from the same request when
reporting optimization improvement.

The formal request uses eight initial design trials, followed by
Gaussian-process Bayesian optimization with a Matérn-5/2 kernel and expected
improvement. A fixed random seed makes candidate generation reproducible.

## 10. Run Case 2: full-process navigation

Use the existing-map request:

```bash
bash scripts/run_same_host_crewai_carla.sh \
  configs/yantai_full_process_agent_request.json
```

This request intentionally uses:

```text
apply_agri_physics: false
```

Do not enable the optional agricultural physics template for the current
full-process case unless its engine and gear parameters are recalibrated for
the CARLA `BehaviorAgent`.

Expected outputs:

```text
results/full_process_navigation/yantai_agent_case/full_process_path.csv
results/full_process_navigation/yantai_agent_case/full_process_path.png
results/full_process_navigation/yantai_agent_case/full_process_task_evaluation.csv
results/full_process_navigation/yantai_agent_case/crewai_flow_report.json
```

The configured garage, road, field and return coordinates are specific to
`YantaiTrackingField`. They must be changed when another map is used.

## 11. Natural-language request and task recovery

PGA can use a configured OpenAI-compatible model to convert natural-language,
image and data input into a candidate JSON request. Local validation still
checks required values and input paths before execution. API credentials must
be provided through the environment and must not be committed to the project.

```bash
export OPENAI_API_KEY="YOUR_KEY"
bash scripts/run_same_host_crewai_carla.sh \
  --text-file configs/prompts/yantai_full_process_prompt.txt \
  --request-template configs/yantai_full_process_agent_request.json \
  --llm-config configs/llm/gpt_5_6_sol.json
```

Existing JSON/YAML requests skip the multimodal conversion step, but the
current PGA, SGA, MCA, CIA and EEA steps still run through CrewAI Crews.
Therefore the CrewAI model and API key remain required for Flow execution.
Only direct domain scripts such as `process_trajectory.py` run without an LLM.
Execution state is stored in `runtime/crewai_runtime.sqlite3`.

```bash
python scripts/run_crewai.py --status <TASK_ID>
python scripts/run_crewai.py --resume <TASK_ID>
```

Once PGA has produced and validated the structured request, the deterministic
domain tools receive fixed inputs, but CrewAI still invokes those tools and
therefore still requires its configured model connection.
