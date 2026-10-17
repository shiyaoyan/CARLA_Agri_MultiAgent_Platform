# Standalone Single-File Runners

The release ships two self-contained CARLA runners: `standalone_trajectory_tracking.py` reproduces the paper's Section 3.2 in-field trajectory tracking of measured DF1204 tractor paths, and `standalone_yantai_full_process.py` reproduces the Section 3.3 multi-stage navigation mission from the garage through two harvest passes with one headland turn and back.

## 1.  a standalone runner 

Each file carries the whole experiment inside it. The project modules are embedded as in-memory
packages (`_tracking_embedded.*` and `_yantai_embedded.*`) and registered by a small import hook at
the bottom of the file, so the same code the agent workflow executes is available without it:

- **3.2**: Trajectory Preprocessing Sub-Agent (TPS) column standardisation and forward/reverse
  labelling, nearest-path reference trajectory, pure-pursuit steering, speed PID, gear state
  machine, tractor physics, synchronous stepping, per-tick recording, metrics and plots.
- **3.3**: mission planning from the live map topology, bulb/omega headland geometry, mission path
  follower, CARLA `BehaviorAgent` road driving, front-wheel-steering enforcement, physics write-back
  and read-back verification, collision sensing, path tracking, task-level evaluation and plots.

You copy one file to the CARLA machine, edit the constants at the top and run it: no CrewAI, no LLM
API key, no `pip install` of this project, no external JSON request file and no input data other
than the CSV you choose. The embedded modules are generated from the project sources by
`scripts/build_tracking_standalone.py` and `scripts/build_yantai_standalone.py`; edit only the
top-of-file constants.

- **Shared requirements**: Python >= 3.8, the CARLA Python API matching the server, numpy, pandas
  and matplotlib (`requirements.txt`); 3.2 needs nothing further.
- **3.3 extras**: shapely (field boundary and footprint checks) and the CARLA official navigation
  agents from `PythonAPI/carla/agents` (`BehaviorAgent`, `GlobalRoutePlanner`), found automatically
  or via `--carla-root`.

## 2. Prerequisites and starting CARLA

1. Start the CARLA server on the machine that holds the map:

   ```bash
   ./CarlaUE4.sh -quality-level=Low -carla-rpc-port=2000   # CarlaUE4.exe on Windows
   ```

2. Run the script with an interpreter that can `import carla`, adding the matching CARLA Python API to the path if it is not installed as a wheel:

   ```bash
   export PYTHONPATH="$PYTHONPATH:/path/to/CARLA/PythonAPI/carla"
   ```

3. Open the right map. The 3.2 runner uses the currently open world while `carla_map` is
   empty, so open the scenario whose ground plane matches `map_offset_x` / `map_offset_y`
   / `spawn_z` (the archived runs used `+50.0 m`, `-10.0 m` and 2 m); a non-empty
   `carla_map` triggers `load_world`. The 3.3 runner requires a world already opened on
   `YantaiSatelliteRebuilt` and refuses any other map (`reload_world` is forced to
   `False`):

   ```bash
   python -c "import carla; c=carla.Client('localhost',2000); c.load_world('YantaiSatelliteRebuilt')"
   ```

   The matching OpenDRIVE reference is
   `data/maps/yantai_satellite/YantaiSatelliteRebuilt.xodr`; the FBX/XODR/texture bundle is
   imported into CARLA separately.

4. Both runners use synchronous mode with a fixed time step, then restore world settings and destroy their own actors on exit; existing actors are not cleared.

