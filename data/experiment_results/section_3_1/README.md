# Section 3.1 Trajectory-Tracking Results

This directory contains the five CARLA runs reported in Section 3.1 of the
paper. The files were copied from the original experiment outputs without
recalculating or filtering the recorded metrics.

| Directory | Paper trajectory type | Duration (s) | MAE (m) | RMSE (m) | Maximum error (m) |
|---|---|---:|---:|---:|---:|
| `complex_operation/` | Complex operation | 632 | 0.040 | 0.082 | 0.689 |
| `regular_operation/` | Regular operation | 157 | 0.033 | 0.102 | 0.722 |
| `straight_line_operation/` | Straight-line operation | 62 | 0.019 | 0.053 | 0.264 |
| `bulb_turn/` | Bulb turn | 49 | 0.032 | 0.039 | 0.102 |
| `fishtail_turn/` | Fishtail turn | 74 | 0.059 | 0.113 | 0.483 |

The paper-facing values use the `all_lateral_*` columns in
`tracking_metrics_summary.csv`. They include every sample within the reference
trajectory time range. The separate `valid_lateral_*` columns are retained for
diagnosis and are not used for the table above.

Each run contains:

- `tick_records.csv`: per-tick reference, vehicle state, control and error data;
- `diagnostic_events.csv`: controller and execution events;
- `phase_summary.csv`: forward/reverse phase summaries;
- `tracking_metrics_summary.csv`: complete and diagnostic-filtered metrics;
- `01_tracking_error_metrics.png`: tracking-error time series;
- `01_time_sync_diagnostics.png`: time-synchronization diagnostics;
- `02_spatial_trajectory.png`: reference and simulated trajectories;
- `README_diagnostics.txt`: definitions recorded by the evaluation module.
