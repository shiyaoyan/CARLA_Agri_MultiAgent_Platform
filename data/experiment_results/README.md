# Published experiment results

This directory holds the raw CARLA outputs behind the two application cases of the
paper, together with the small machine-readable manifest that the offline verification
scripts consume. Nothing here has been retouched: the metric summaries are the ones the
evaluation sub-agent wrote at run time, and the per-tick logs are complete.

```text
data/experiment_results/
├── manifest.json                  # trajectories, evidence hashes and run locations
├── section_3_2_config.json        # frozen Section 3.2 controller/physics configuration
├── section_3_2_tracking/          # Section 3.2, in-field trajectory tracking (Table 1)
│   ├── EXP01_complex_operation/
│   ├── EXP02_regular_operation/
│   ├── EXP03_straight_line/
│   ├── EXP04_bulb_turn/
│   └── EXP05_fishtail_turn/
└── section_3_3/
    └── yantai_satellite/
        └── run_20261015T010039Z_f27a13ca/   # Section 3.3, multi-stage navigation (Table 2)
```

## Section 3.2 - In-field trajectory tracking

Five CARLA runs, one per EXP-FTT evaluation trajectory, each with a different in-field
operation type. The controller parameters and the tractor physics are frozen in
[`section_3_2_config.json`](section_3_2_config.json).

| Run | Operation type | Recorded ticks | Reference duration (s) | MAE (m) | RMSE (m) | Max AE (m) | P95 (m) |
|---|---|---:|---:|---:|---:|---:|---:|
| `EXP01_complex_operation` | Complex operation | 11381 | 569 | 0.0264 | 0.0613 | 0.7826 | 0.0670 |
| `EXP02_regular_operation` | Regular operation | 3141 | 157 | 0.0334 | 0.1096 | 0.7460 | 0.1277 |
| `EXP03_straight_line` | Straight-line operation | 1540 | 62 | 0.0196 | 0.0557 | 0.2793 | 0.1475 |
| `EXP04_bulb_turn` | Bulb turn | 1215 | 60 | 0.0327 | 0.0425 | 0.0974 | 0.0827 |
| `EXP05_fishtail_turn` | Fishtail turn | 1681 | 84 | 0.0833 | 0.1389 | 0.7877 | 0.2815 |

The three error columns are the **all-sample** scope (`all_lateral_*` in
`tracking_metrics_summary.csv`): every recorded tick inside the reference time range is
included. `tracking_metrics_summary.csv` also carries a diagnostic `valid_lateral_*`
scope that drops start and end transients, low-speed samples and gear-shift transients.
The paper reports the all-sample scope; the valid scope is provided so the excluded
transients can be inspected rather than hidden.

| Run | Valid-sample ratio | Valid MAE (m) | Valid RMSE (m) | Valid Max AE (m) |
|---|---:|---:|---:|---:|
| `EXP01_complex_operation` | 0.732 | 0.0225 | 0.0545 | 0.7826 |
| `EXP02_regular_operation` | 0.789 | 0.0408 | 0.1233 | 0.7460 |
| `EXP03_straight_line` | 0.529 | 0.0331 | 0.0765 | 0.2793 |
| `EXP04_bulb_turn` | 0.612 | 0.0318 | 0.0403 | 0.0974 |
| `EXP05_fishtail_turn` | 0.509 | 0.0898 | 0.1594 | 0.7877 |

Every run directory contains the same eight files:

| File | Contents |
|---|---|
| `tick_records.csv` | one row per simulation tick: vehicle state, reference point, control command, lateral and heading errors, and the metric-validity flags |
| `diagnostic_events.csv` | controller and execution events, including the phase-shift and `FINISHED` markers |
| `phase_summary.csv` | per forward/reverse phase error statistics |
| `tracking_metrics_summary.csv` | overall metrics in both the all-sample and valid-sample scopes |
| `01_tracking_error_metrics.png` | lateral and heading error time series |
| `01_time_sync_diagnostics.png` | timing and synchronisation diagnostics |
| `02_spatial_trajectory.png` | reference and simulated trajectories |
| `README_diagnostics.txt` | metric definitions written by the evaluation sub-agent |

> **Known gap.** `EXP03_straight_line/diagnostic_events.csv` is empty: that run recorded
> no `FINISHED` event, so its terminal stop condition cannot be confirmed from the event
> log alone. All other four runs recorded a `FINISHED` event. The `EXP03` metrics are
> unaffected, and the offline audit reports this as a warning rather than an error.

## Section 3.3 - Multi-stage agricultural navigation

One archived closed-loop CARLA run on the rebuilt Yantai satellite map
(`YantaiSatelliteRebuilt`), evaluated in `full_process_task_evaluation.csv`:

| Evaluation item | Result |
|---|---|
| Task completion | Completed |
| Stage transition | Completed |
| Forward/reverse operation | Not applicable (forward-only task) |
| Collision event | 0 |
| Total travel distance | 281.618 m |
| In-field operation path length | 130.299 m |
| Task duration | 458.250 s |

Run characteristics from `run_status.json` and `vehicle_physics.json`:

| Property | Value |
|---|---|
| Map | `YantaiSatelliteRebuilt` |
| Stages | 9 of 9 completed, including two straight harvest passes and one bulb turn |
| Recorded ticks | 9165 at a fixed step of 0.05 s |
| In-field operation distance | 130.299 m, counting the `HARVEST_LINE` stages only (bulb turn and field access excluded) |
| Task duration | 458.250 s, excluding the 20 initialisation ticks |
| Sensor contacts | 0 fatal and 0 non-fatal |
| Garage return position error | 0.765 m against a 0.8 m tolerance |
| Garage return heading error | 7.003 deg against a 15 deg tolerance |
| Final speed / stop hold | 0.00248 m/s held for 1.0 s |
| Custom tractor physics | applied and verified by read-back |

The path figure uses the same `PathTracker` drawing function as the project code, and the
4461 samples in `full_process_path.csv` correspond exactly to the per-tick log under the
existing 0.05 m cumulative-displacement sampling rule. All 9165 ticks are retained; no
start-up, terminal or turning samples were dropped.

## Evidence manifest

[`manifest.json`](manifest.json) is the machine-readable index of the published
evidence. It records:

- `tracking_config` - the frozen Section 3.2 configuration;
- `navigation_run` - the Section 3.3 run directory;
- `trajectories` - all 15 dataset trajectories with their group, operation type and,
  for EXP-FTT, the matching Section 3.2 result directory;
- `sha256` - the SHA-256 of every evidence file listed above.

## Verifying the results offline

Both application cases can be re-derived from the shipped logs without a simulator and
without an LLM API call:

```bash
# Replay the Section 3.2 controller against the recorded ticks, recompute every
# metric and table from the raw tick logs, reconcile the Section 3.3 run, and
# check every dataset and evidence hash in the manifest.
python scripts/audit_paper_results.py

# Write the same report to a file instead of stdout
python scripts/audit_paper_results.py --report /tmp/paper_audit.json

# Reconcile only the Section 3.3 run
python scripts/audit_yantai_full_process_run.py \
  data/experiment_results/section_3_3/yantai_satellite/run_20261015T010039Z_f27a13ca
```

`audit_paper_results.py` re-drives the pure-pursuit, PID and gear-state controllers over
the recorded vehicle states and requires the recomputed commands, phase labels, metric
masks, phase summaries and overall metrics to match the archived files. On the shipped
data it reports `"audit_passed": true` with no errors; the only warning is the missing
`FINISHED` event for `EXP03_straight_line` described above.

`scripts/reproduce_tracking.py` goes one step further and re-runs the experiment in
CARLA using the frozen configuration, refusing to start if the current tractor physics
no longer matches the archived snapshot. See
[docs/reproducing_paper_results.md](../docs/reproducing_paper_results.md).
