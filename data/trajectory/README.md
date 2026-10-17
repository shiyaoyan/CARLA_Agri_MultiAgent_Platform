# Trajectory dataset

This directory holds the trajectory data used by the two application cases in the
paper. Every file is a plain CSV of measured DF1204 tractor GNSS records sampled at
1 Hz and is ready to be consumed by the Trajectory Preprocessing Sub-Agent (TPS).

```text
data/trajectory/
├── manifest.csv               # one row per trajectory: geometry, speed and provenance
├── EXP-FTT/                   # 5 evaluation trajectories (paper Section 3.2, Table 1)
│   ├── EXP01_complex_operation.csv
│   ├── EXP02_regular_operation.csv
│   ├── EXP03_straight_line.csv
│   ├── EXP04_bulb_turn.csv
│   └── EXP05_fishtail_turn.csv
├── TS-Tune/                   # 10 tuning trajectories (paper Section 2.4-2.5)
│   ├── TS01_straight_line.csv
│   ├── TS02_straight_line.csv
│   ├── TS03_regular_operation.csv
│   ├── TS04_regular_operation.csv
│   ├── TS05_complex_operation.csv
│   ├── TS06_complex_operation.csv
│   ├── TS07_bulb_turn.csv
│   ├── TS08_bulb_turn.csv
│   ├── TS09_fishtail_turn.csv
│   └── TS10_fishtail_turn.csv
└── yantai_quanlujing.csv      # raw recorded example, before TPS standardisation
```

## CSV format

Every trajectory file uses the same six columns:

```text
longitude,latitude,time,speed,direction,forward_reverse_label
121.512876295,37.434046581,2026-08-27 17:09:05,2.939572,82.241847,1
```

| Column | Unit | Meaning |
|---|---|---|
| `longitude` | degree (WGS84) | GNSS longitude |
| `latitude` | degree (WGS84) | GNSS latitude |
| `time` | `YYYY-MM-DD HH:MM:SS` | acquisition timestamp, strictly 1 Hz with no gaps |
| `speed` | **km/h** | ground speed along the trajectory |
| `direction` | degree | instantaneous motion heading, `atan2(dNorth, dEast)`, counter-clockwise from east |
| `forward_reverse_label` | 1 / 0 | 1 = forward motion, 0 = reverse motion |

> **Unit note.** `speed` is in **km/h** for all 15 files. The recorded speed range is
> 0.001-8.006 km/h (0-2.22 m/s), which matches low-speed in-field tractor operation.
> All 15 trajectories satisfy `path_length ~= sum(speed / 3.6)`, so the speed column and
> the coordinate column are self-consistent.

## Trajectory overview

| # | Trajectory | Set | Operation type | Points | Duration (s) | Path length (m) | Extent W x H (m) | Reverse points | Mean / max speed (km/h) |
|---|---|---|---|---:|---:|---:|---|---:|---|
| 1 | `TS01_straight_line` | TS-Tune | straight_line | 228 | 228 | 265.29 | 257.0 x 20.89 | 0 | 4.198 / 5.479 |
| 2 | `TS02_straight_line` | TS-Tune | straight_line | 187 | 187 | 264.65 | 258.85 x 16.04 | 0 | 5.105 / 7.334 |
| 3 | `TS03_regular_operation` | TS-Tune | regular_operation | 103 | 103 | 197.82 | 20.38 x 91.74 | 0 | 6.989 / 8.006 |
| 4 | `TS04_regular_operation` | TS-Tune | regular_operation | 189 | 189 | 177.88 | 19.29 x 85.03 | 0 | 3.41 / 4.28 |
| 5 | `TS05_complex_operation` | TS-Tune | complex_operation | 598 | 598 | 610.49 | 20.5 x 99.89 | 0 | 3.677 / 4.47 |
| 6 | `TS06_complex_operation` | TS-Tune | complex_operation | 610 | 610 | 604.94 | 19.74 x 93.93 | 0 | 3.573 / 4.537 |
| 7 | `TS07_bulb_turn` | TS-Tune | bulb_turn | 50 | 50 | 35.17 | 8.24 x 15.62 | 0 | 2.59 / 3.02 |
| 8 | `TS08_bulb_turn` | TS-Tune | bulb_turn | 48 | 48 | 38.86 | 7.81 x 17.55 | 0 | 2.98 / 3.475 |
| 9 | `TS09_fishtail_turn` | TS-Tune | fishtail_turn | 78 | 78 | 29.55 | 7.03 x 10.26 | 29 | 1.4 / 4.375 |
| 10 | `TS10_fishtail_turn` | TS-Tune | fishtail_turn | 61 | 61 | 31.16 | 7.82 x 10.97 | 21 | 1.885 / 4.108 |
| 11 | `EXP01_complex_operation` | EXP-FTT | complex_operation | 570 | 570 | 533.37 | 13.97 x 105.94 | 14 | 3.369 / 7.428 |
| 12 | `EXP02_regular_operation` | EXP-FTT | regular_operation | 158 | 158 | 188.43 | 13.69 x 90.11 | 0 | 4.294 / 7.072 |
| 13 | `EXP03_straight_line` | EXP-FTT | straight_line | 63 | 63 | 68.44 | 6.19 x 68.05 | 0 | 3.995 / 7.072 |
| 14 | `EXP04_bulb_turn` | EXP-FTT | bulb_turn | 61 | 61 | 47.35 | 6.57 x 21.66 | 0 | 2.848 / 5.379 |
| 15 | `EXP05_fishtail_turn` | EXP-FTT | fishtail_turn | 85 | 85 | 42.94 | 7.27 x 19.47 | 14 | 1.819 / 5.425 |

Operation-type distribution — every type contributes exactly three trajectories:

| Operation type | EXP-FTT | TS-Tune | Total |
|---|---:|---:|---:|
| straight_line | 1 | 2 | 3 |
| regular_operation | 1 | 2 | 3 |
| complex_operation | 1 | 2 | 3 |
| bulb_turn | 1 | 2 | 3 |
| fishtail_turn | 1 | 2 | 3 |
| **Total** | **5** | **10** | **15** |

## Role of each set

- **TS-Tune** (10 trajectories) is the controller tuning set of Sections 2.4 and 2.5.
  It is the input to machinery parameter refinement and to the Bayesian search over the
  seven controller parameters. All ten trajectories were recorded with the target
  DF1204 tractor.
- **EXP-FTT** (5 trajectories) is the evaluation set of Section 3.2. The five files map
  one-to-one onto the five rows of Table 1 and are tracked with the controller parameters
  frozen at the values selected on TS-Tune.
- **`yantai_quanlujing.csv`** is the raw recorded example kept as the input of the
  `section_3_2_tracking` paper case. `EXP-FTT/EXP01_complex_operation.csv` is its
  standardised form; TPS accepts either and produces the same reference path.

The two sets are disjoint: no exact `(longitude, latitude, time)` point is shared
between TS-Tune and EXP-FTT, so controller tuning cannot leak into the evaluation.

## Provenance and construction

Four data sets are used across the paper. Two of them are shipped here, and the other
two are documented but not redistributed.

- **TS-Stat** (*not bundled*) is the external data set of Luo et al. (2026): 960
  trajectories recorded between May and June 2023 in China with three Jiangsu World
  Agricultural Machinery tractor models (WD904, WD904G at 66.2 kW and WH1804 at 180 kW).
  It has seven columns per point, including torque percentage and engine speed, sampled
  every 5 s. It is used only to derive reference ranges for machinery parameters and is
  available from the original publication. Redistribution is not possible here.
- **TS-Tune** (10 trajectories, bundled) was recorded in Yantai, Shandong, China with the
  target Dongfeng DF1204 tractor. Ten trajectories are derived from two source
  recordings (`20251228_074155`, `20251230_060125`, `20260109_053725`, `20260827_085955`).
- **EXP-FTT** (5 trajectories, bundled) was recorded on 2026-05-12 with the same DF1204
  tractor.
- **EXP-MSN** (*not bundled*) consists only of satellite imagery covering the multi-stage
  navigation area. It is a third-party basemap and is therefore not redistributed; supply
  an owned image through the `scene_generation` paper case instead.

Construction notes, recorded so the numbers in `manifest.csv` can be audited:

1. **1 Hz extraction.** The `20260827_085955` source is 5 Hz. Trajectories derived from
   it keep the samples where `raw_index mod 5 == 4`, which reproduces the original
   segment CSVs exactly.
2. **Speed unit unification.** The two source batches disagreed: the segment exports were
   in m/s and the `yantai_*.csv` exports were already in km/h. All 15 trajectories were
   recomputed as `speed_i = planar_distance(p_i, p_i+1) / 1 s * 3.6` and
   `direction_i = atan2(dNorth, dEast)`. Longitude, latitude and timestamps were not
   altered.
3. **Bulb-turn repair.** The original bulb-turn cuts contained only the 180-220 degree
   arc with open ends. `TS07` and `TS08` were re-cut from the 5 Hz source with an extra
   10 s of neck at each end, giving closed bulb loops of 50 s and 48 s.
4. **Single-lap trimming.** `TS03` and `TS04` originally contained about two laps of the
   regular-operation circuit. Both are trimmed to exactly one lap, detected
   independently by accumulated heading turning through +-360 degrees and by returning
   to within 1 m of the start point.
5. **EXP-FTT internal overlap.** The five EXP-FTT trajectories are five labelled windows
   cut from **one continuous 570 s recording session** on 2026-05-12.
   `EXP01_complex_operation` is the full window; `EXP02`, `EXP03` and `EXP04` are
   entirely contained in it, and `EXP05` shares 78 of its 85 points. This is intentional:
   each EXP-FTT trajectory is scored independently for its own operation type in Table 1,
   and the property that matters for tuning integrity — zero overlap between TS-Tune and
   EXP-FTT — holds.

## Forward/reverse labels

`EXP01_complex_operation` and `EXP05_fishtail_turn` carry `forward_reverse_label = 0`
on 14 points each, and `TS09`/`TS10` carry 29 and 21 such points. The complex-operation
window therefore exercises forward/reverse switching as described in the paper.

The label cannot be re-derived from the five GNSS columns alone, because separating
forward from reverse motion requires the vehicle heading, which is not part of the
recorded schema. The labels come from the original acquisition pipeline, are preserved
verbatim here, and are not recomputed by TPS. If a claim depends on the exact
forward/reverse split, validate those points against the original controller-area
network or heading record.

## Using the data

```bash
# Standardise one trajectory with TPS and inspect the reference path
python scripts/process_trajectory.py \
  --input data/trajectory/EXP-FTT/EXP01_complex_operation.csv \
  --output outputs/trajectory/EXP01_complex_operation_standard.csv

# Re-run the published Section 3.2 evaluation cases with the frozen parameters
python scripts/reproduce_tracking.py --all --prepare-only   # no CARLA needed
python scripts/reproduce_tracking.py --all                  # needs a running CARLA

# Verify the shipped dataset and evidence without CARLA
python scripts/audit_paper_results.py
```

See [data/experiment_results/README.md](../experiment_results/README.md) for the
result-file formats and metric definitions, and
[docs/reproducing_paper_results.md](../../docs/reproducing_paper_results.md) for the full
reproduction procedure.
