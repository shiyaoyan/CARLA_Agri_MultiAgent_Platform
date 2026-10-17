"""Run the published fixed-parameter tracking case from the archived configuration."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agri_carla.agents.experiment_execution import ExperimentExecutionAgent
from agri_carla.machinery.physics import AgriPhysicsConfig
from agri_carla.simulation.tracking_experiment import TrackingExperimentConfig
from agri_carla.trajectory.preprocessing_subagent import TrajectoryPreprocessingSubagent
from agri_carla.trajectory.reference_path import ReferenceTrajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--case', help='Input identifier, e.g. EXP01_complex_operation or TS01_straight_line.')
    selection.add_argument('--all', action='store_true', help='Run all five EXP-FTT evaluation cases.')
    parser.add_argument('--prepare-only', action='store_true', help='Validate/standardize inputs without CARLA.')
    parser.add_argument('--output-dir', default='results/paper_tracking')
    parser.add_argument('--host', default=None)
    parser.add_argument('--port', type=int, default=None)
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'data/experiment_results/manifest.json').read_text())
    snapshot = json.loads((ROOT / manifest['tracking_config']).read_text())
    actual_physics = json.loads(json.dumps(AgriPhysicsConfig('tractor').to_dict()))
    if actual_physics != snapshot['tractor_physics']:
        parser.error('Current tractor physics differs from the archived case. Use the release physics before reproducing this run.')
    selected = [item for item in manifest['trajectories']
                if (args.all and item['group'] == 'EXP-FTT') or item['id'] == args.case]
    if not selected:
        parser.error('Unknown case. Available: ' + ', '.join(item['id'] for item in manifest['trajectories']))
    output = Path(args.output_dir).expanduser()
    if not output.is_absolute():
        output = ROOT / output
    evidence = (ROOT / 'data/experiment_results').resolve()
    if output.resolve() == evidence or evidence in output.resolve().parents:
        parser.error('Use a new output directory, not data/experiment_results.')
    output = output / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    os.environ.setdefault('MPLBACKEND', 'Agg')
    for item in selected:
        source = ROOT / item['input']
        processed = output / 'processed_trajectory' / (source.stem + '_standard.csv')
        config_data = copy.deepcopy(snapshot['config'])
        config_data['output_dir'] = str(output)
        if args.host is not None:
            config_data['host'] = args.host
        if args.port is not None:
            config_data['port'] = args.port
        config = TrackingExperimentConfig.from_dict(config_data)
        if args.prepare_only:
            TrajectoryPreprocessingSubagent().process(source, processed)
            ReferenceTrajectory(processed, flip_y=config.trajectory_flip_y).print_data_check()
        else:
            result = ExperimentExecutionAgent().run_trajectory_tracking(
                trajectory_csv=source, processed_csv=processed, config=config)
            print(json.dumps(result, indent=2, default=str))


if __name__ == '__main__':
    main()
