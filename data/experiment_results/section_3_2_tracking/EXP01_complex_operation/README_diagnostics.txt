Trajectory tracking diagnostic notes
This report follows the nearest-path evaluation logic used by result_time.py.

1. pos_err_m is the longitudinal timing error projected onto the local trajectory tangent.
2. cmd_abs_kmh is the catch-up speed command generated from the timing error.
3. path_cte_m / cte_m is the signed lateral error to the nearest reference path segment.
4. time_sync_cte_m is retained as a diagnostic value for same-time reference-point deviation.
5. path_heading_error_deg is the difference between vehicle motion direction and nearest path heading.
6. metric_valid marks samples after excluding start, end, low-speed and shifting transients.

Overall nearest-path metrics:
- Total samples: 11381
- Post-endpoint samples: 0
- Lateral RMSE: 0.0613 m
- Lateral MAE: 0.0264 m
- Lateral P95: 0.0670 m
- Maximum lateral error: 0.7826 m
- Lateral standard deviation: 0.0612 m
- Heading RMSE: 35.3939 deg
- Heading MAE: 12.3621 deg
- Heading P95: 91.8832 deg
- Maximum heading error: 179.6781 deg
- Heading standard deviation: 35.3036 deg