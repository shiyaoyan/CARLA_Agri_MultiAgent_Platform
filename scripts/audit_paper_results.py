"""Recompute published summaries and replay tracking control from recorded states."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from agri_carla.controllers.interface_agent import ControllerInterfaceAgent
from agri_carla.evaluation.performance_evaluation_subagent import PerformanceEvaluationSubagent
from agri_carla.simulation.tracking_experiment import TrackingExperimentConfig
from agri_carla.trajectory.preprocessing_subagent import TrajectoryPreprocessingSubagent
from agri_carla.trajectory.reference_path import ReferenceTrajectory
from audit_yantai_full_process_run import audit as audit_navigation


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_table(expected, actual):
    if list(expected.columns) != list(actual.columns) or expected.shape != actual.shape:
        return False
    for column in expected:
        if pd.api.types.is_numeric_dtype(expected[column]):
            if not np.allclose(expected[column], actual[column], rtol=1e-9, atol=1e-9, equal_nan=True):
                return False
        elif not expected[column].fillna('').astype(str).equals(actual[column].fillna('').astype(str)):
            return False
    return True


def audit_tracking(input_csv, run_dir, config_data):
    input_csv, run_dir = Path(input_csv), Path(run_dir)
    config = TrackingExperimentConfig.from_dict(config_data)
    # Preserve the serialized clock at neutral-period boundaries during replay.
    records = pd.read_csv(run_dir / 'tick_records.csv', float_precision='round_trip').fillna({'diagnostic': '', 'metric_filter_reason': ''})
    errors, warnings = [], []
    with tempfile.TemporaryDirectory() as temporary:
        processed = Path(temporary) / 'standard.csv'
        TrajectoryPreprocessingSubagent().process(input_csv, processed)
        trajectory = ReferenceTrajectory(processed, flip_y=config.trajectory_flip_y)
        controller = ControllerInterfaceAgent(trajectory, config.map_offset_x, config.map_offset_y, config.controller)
        controller.reset()
        maximum_delta = {}
        string_mismatches = set()
        keys = ('ref_x', 'ref_y', 'target_abs_kmh', 'cmd_abs_kmh', 'pos_err_m',
                'geometric_err_m', 'throttle', 'brake', 'steer', 'path_cte_m',
                'path_distance_m', 'nearest_path_x', 'nearest_path_y', 'nearest_path_u',
                'target_idx', 'lookahead_idx', 'nearest_path_idx', 'time_sync_cte_m')
        for row in records.to_dict('records'):
            command = controller.compute({
                'x': row['x'], 'y': row['y'], 'yaw_deg': row['yaw_deg'],
                'actual_speed_ms': row['actual_abs_kmh'] / 3.6,
            }, row['time_s'])
            for key in keys:
                delta = abs(float(row[key]) - float(command[key]))
                maximum_delta[key] = max(maximum_delta.get(key, 0), delta)
            for key in ('is_forward', 'diagnostic'):
                if row[key] != command[key]:
                    string_mismatches.add(key)
            if row['phase_id'] != trajectory.segment_id[int(command['target_idx'])]:
                string_mismatches.add('phase_id')
            in_time = row['time_s'] <= trajectory.t[-1] + 1e-9
            reasons = []
            if not in_time:
                reasons.append('POST_ENDPOINT')
            if row['time_s'] < config.metric_skip_start_s:
                reasons.append('START_TRANSIENT')
            if row['time_s'] > trajectory.t[-1] - config.metric_skip_end_s:
                reasons.append('END_TRANSIENT')
            if row['actual_abs_kmh'] / 3.6 < config.metric_min_speed_ms:
                reasons.append('LOW_SPEED')
            if 'SHIFT_NEUTRAL' in command['diagnostic']:
                reasons.append('SHIFT_TRANSIENT')
            if (bool(row['in_reference_time']) != in_time or
                    bool(row['metric_valid']) != (not reasons) or
                    row['metric_filter_reason'] != '|'.join(reasons)):
                string_mismatches.add('metric_masks')
            for key in ('heading_error', 'path_heading_error', 'time_sync_heading_error'):
                delta = abs(row[key + '_deg'] - np.degrees(command[key + '_rad']))
                maximum_delta[key + '_deg'] = max(maximum_delta.get(key + '_deg', 0), delta)
        errors.extend('Replay mismatch: ' + key for key, value in maximum_delta.items() if value > 1e-6)
        errors.extend('Replay mismatch: ' + key for key in sorted(string_mismatches))
        if not np.allclose(records.time_s, np.arange(len(records)) * config.dt, atol=1e-8, rtol=0):
            errors.append('Nonuniform tick timestamps')
        if not np.isfinite(records[list(keys)].to_numpy(dtype=float)).all():
            errors.append('Nonfinite values in recorded fields')
        if not np.allclose(records.map_offset_x, config.map_offset_x) or not np.allclose(records.map_offset_y, config.map_offset_y):
            errors.append('Map offsets differ from the archived configuration')
        evaluator = PerformanceEvaluationSubagent(temporary)
        metrics = evaluator.build_overall_metrics(records)
        if not compare_table(metrics, pd.read_csv(run_dir / 'tracking_metrics_summary.csv')):
            errors.append('Overall metrics differ from the raw ticks')
        if not compare_table(evaluator.build_phase_summary(records), pd.read_csv(run_dir / 'phase_summary.csv')):
            errors.append('Phase metrics differ from the raw ticks')
        try:
            events = pd.read_csv(run_dir / 'diagnostic_events.csv')
        except pd.errors.EmptyDataError:
            events = pd.DataFrame(columns=['code'])
        finished = 'FINISHED' in events.code.values
        if not finished:
            warnings.append('No FINISHED event: output files alone do not establish the terminal stop condition.')
        if records.time_s.iloc[-1] < trajectory.t[-1] - 1e-8:
            errors.append('Recorded trace ends before the reference duration')
        metric = metrics.iloc[0].to_dict()
        return {
            'input': input_csv.name, 'run': run_dir.name, 'audit_passed': not errors,
            'errors': errors, 'warnings': warnings, 'input_sha256': sha256(input_csv),
            'input_points': trajectory.num_points, 'reference_duration_s': float(trajectory.t[-1]),
            'last_record_time_s': float(records.time_s.iloc[-1]),
            'finished_event': finished, 'maximum_replay_difference': maximum_delta,
            'metrics': metric,
        }


def audit_dataset(root, entries):
    point_sets, content_hashes, rows, errors = {}, {}, [], []
    for entry in entries:
        path = root / entry['input']
        df = pd.read_csv(path)
        times = pd.to_datetime(df.time)
        intervals = times.diff().dropna().dt.total_seconds()
        if not (intervals == 1).all():
            errors.append(entry['id'] + ': timestamps are not uniformly 1 second apart')
        if not set(df.forward_reverse_label).issubset({0, 1}):
            errors.append(entry['id'] + ': invalid motion labels')
        point_sets[entry['id']] = set(zip(df.longitude, df.latitude, df.time))
        content_hashes[entry['id']] = sha256(path)
        rows.append({'id': entry['id'], 'group': entry['group'], 'rows': len(df),
                     'duration_s': float((times.iloc[-1] - times.iloc[0]).total_seconds()),
                     'reverse_points': int((df.forward_reverse_label == 0).sum()),
                     'sha256': content_hashes[entry['id']]})
    overlaps = []
    for i, left in enumerate(entries):
        for right in entries[i + 1:]:
            common = len(point_sets[left['id']] & point_sets[right['id']])
            if common:
                overlaps.append({'left': left['id'], 'right': right['id'], 'shared_points': common})
                if left['group'] != right['group'] or left['group'] == 'TS-Tune':
                    errors.append('Overlapping tuning data: ' + left['id'] + ' / ' + right['id'])
            if content_hashes[left['id']] == content_hashes[right['id']]:
                errors.append('Duplicate input: ' + left['id'] + ' / ' + right['id'])
    return {'audit_passed': not errors, 'errors': errors, 'trajectories': rows,
            'exact_coordinate_timestamp_overlaps': overlaps,
            'scope': 'Exact coordinate/time overlap check; not proof of independent acquisition or vehicle identity.'}


def audit(root=ROOT):
    root = Path(root)
    manifest = json.loads((root / 'data/experiment_results/manifest.json').read_text())
    baseline = json.loads((root / manifest['tracking_config']).read_text())
    errors = []
    for relative, expected in manifest['sha256'].items():
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            errors.append('Evidence checksum differs: ' + relative)
    dataset = audit_dataset(root, manifest['trajectories'])
    tracking = [audit_tracking(root / item['input'], root / item['result'], baseline['config'])
                for item in manifest['trajectories'] if item.get('result')]
    navigation = audit_navigation(root / manifest['navigation_run'])
    passed = not errors and dataset['audit_passed'] and navigation['audit_passed'] and all(item['audit_passed'] for item in tracking)
    return {'audit_passed': passed, 'errors': errors, 'dataset': dataset,
            'tracking': tracking, 'navigation': navigation,
            'scope': 'Offline evidence reconciliation. No simulator or LLM run performed.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, help='Write a new report outside the published evidence directory.')
    args = parser.parse_args()
    report = audit()
    if args.report:
        evidence = (ROOT / 'data/experiment_results').resolve()
        destination = args.report.resolve()
        if destination == evidence or evidence in destination.parents:
            parser.error('Use an output path outside data/experiment_results.')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not report['audit_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
