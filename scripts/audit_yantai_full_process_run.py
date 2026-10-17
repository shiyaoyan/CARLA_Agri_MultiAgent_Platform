"""Reconcile a saved CARLA run with project code without rewriting raw results."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import io
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agri_carla.controllers.full_process_control import MissionPathFollower
from agri_carla.evaluation.path_tracker import PathTracker
from agri_carla.evaluation.performance_evaluation_subagent import PerformanceEvaluationSubagent
from agri_carla.machinery.physics_audit import physics_report, SCALARS, WHEEL_FIELDS
from agri_carla.simulation.full_process_config import FullProcessExperimentConfig


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def normalized(value):
    return json.loads(json.dumps(value))


def audit(run_dir):
    run_dir = Path(run_dir)
    names = ("planned_mission.json", "vehicle_physics.json", "per_tick_states.csv",
             "stage_events.json", "collision_events.json", "startup_diagnostics.json",
             "run_status.json", "full_process_path.csv", "full_process_path.png",
             "full_process_task_evaluation.csv")
    hashes = {name: hashlib.sha256((run_dir/name).read_bytes()).hexdigest() for name in names}
    load = lambda name: json.loads((run_dir/name).read_text(encoding="utf-8"))
    plan, status, physics = [load(name) for name in (
        "planned_mission.json", "run_status.json", "vehicle_physics.json")]
    ticks = read_csv(run_dir/"per_tick_states.csv")
    events = load("stage_events.json")
    contacts = load("collision_events.json")
    errors = []

    def check(condition, description):
        if not condition:
            errors.append(description)

    def close(a, b, tolerance=1e-8):
        return math.isfinite(a) and math.isfinite(b) and abs(a-b) <= tolerance

    defaults = normalized(asdict(FullProcessExperimentConfig()))
    actual_config = plan["config"]
    for key, value in defaults.items():
        if key not in ("host", "port", "timeout", "output_dir"):
            check(actual_config.get(key) == value, "Case configuration differs: " + key)
    check(physics["requested"] == defaults["physics_overrides"], "Requested physics differs")
    check(status["map_name"] == defaults["carla_map"], "Map name differs")
    check(plan["map"].rsplit("/", 1)[-1] == defaults["carla_map"], "Planned map differs")
    map_sha = hashlib.sha256((ROOT/"data/maps/yantai_satellite/YantaiSatelliteRebuilt.xodr").read_bytes()).hexdigest()
    check(plan["xodr_sha256"] == map_sha, "Runtime OpenDRIVE differs from project map")
    check(len(ticks) == status["total_ticks"] and bool(ticks), "Tick count mismatch")
    check(status["status"] == "COMPLETED" and status["task_completed"], "Task is incomplete")
    check(status["stage_transition_completed"] and not status["failure_reason"], "Failed transitions")
    check(len(contacts) == status["collision_count"], "Collision count differs from raw events")
    check(status["fatal_collision_count"] + status["startup_contact_count"] == len(contacts),
          "Collision classification counts differ")
    check(status["collision_events"] == contacts, "Status contact records differ")
    check(physics["custom_physics_applied"] and physics["readback_matches"],
          "Recorded physics read-back failed")

    # Re-run the current physics comparator against the stored CARLA read-back.
    raw_physics = physics["actual"]
    fake = SimpleNamespace(
        **{key: raw_physics[key] for key in SCALARS},
        center_of_mass=SimpleNamespace(**dict(zip(("x", "y", "z"), raw_physics["center_of_mass"]))),
        use_gear_autobox=raw_physics["use_gear_autobox"],
        torque_curve=[SimpleNamespace(x=x, y=y) for x, y in raw_physics["torque_curve"]],
        forward_gears=[SimpleNamespace(ratio=r, down_ratio=d, up_ratio=u)
                       for r, d, u in raw_physics["forward_gears"]],
        wheels=[SimpleNamespace(**{key: wheel[key] for key in WHEEL_FIELDS}, radius=wheel["radius_cm"])
                for wheel in raw_physics["wheels"]],
    )
    replay_physics = physics_report(SimpleNamespace(get_physics_control=lambda: fake),
                                   defaults["physics_overrides"], True,
                                   expect_autobox=defaults["use_gear_autobox"])
    check(replay_physics["readback_matches"], "Recomputed physics audit failed")
    check(replay_physics == physics, "Physics report format/content differs")
    config = FullProcessExperimentConfig(**actual_config)
    config.wheelbase_m = status["control_geometry"]["wheelbase_m"]
    config.max_steer_angle_deg = status["control_geometry"]["front_max_steer_deg"]
    tracker = PathTracker()
    active = None
    observed, re_events, stage_stats = [], [], []
    follower = None
    max_obs_delta = max_control_delta = 0.0
    parked_ticks = 0
    for i, row in enumerate(ticks):
        index = int(row["task_index"])
        task = plan["tasks"][index]
        t = float(row["time_s"])
        check(close(t, (i+1)*config.fixed_delta_seconds), "Nonuniform tick time at " + str(i))
        check(row["stage"] == task["type"], "Stage identity mismatch at " + str(i))
        if index != active:
            check(index == len(observed), "Skipped/reordered stage at " + str(i))
            active = index
            follower = MissionPathFollower(task, config)
            observed.append(task["type"])
            re_events.append(dict(event="stage_started", index=index, stage=task["type"], time_s=t))
            stage_stats.append(dict(index=index, stage=task["type"], ticks=0, speed_sum=0.0,
                                    max_cross_track_m=0.0))
        position = tuple(float(row[key]) for key in ("x", "y", "z"))
        speed, yaw = float(row["speed_ms"]), float(row["yaw_deg"])
        obs = follower.observe(position, yaw)
        for key, value in obs.items():
            if key == "arrived":
                check((row[key] == "True") == value, "Arrival mismatch at " + str(i))
            else:
                delta = abs(float(row[key])-value)
                max_obs_delta = max(max_obs_delta, delta)
                check(close(float(row[key]), value), "Observation mismatch: " + key + " at " + str(i))
        # BehaviorAgent depends on CARLA. Replay only the deterministic off-road controller.
        if task["type"] not in ("ROAD_NAV", "ROAD_RETURN"):
            command = follower.control(SimpleNamespace(VehicleControl=SimpleNamespace),
                                       position, yaw, speed, obs)
            for key in ("throttle", "brake", "steer"):
                delta = abs(float(row[key])-getattr(command, key))
                max_control_delta = max(max_control_delta, delta)
                check(close(float(row[key]), getattr(command, key), 1e-6),
                      "Control mismatch: " + key + " at " + str(i))
        check(row["reverse"] == "False", "Unexpected reverse control")
        check(0 <= int(row["fatal_collision_count"]) <= int(row["collision_count"]) <= len(contacts),
              "Per-tick collision counts disagree with event records")
        tracker.update(SimpleNamespace(x=position[0], y=position[1]), task["type"])
        stats = stage_stats[-1]
        stats["ticks"] += 1
        stats["speed_sum"] += speed
        stats["max_cross_track_m"] = max(stats["max_cross_track_m"], obs["cross_track_m"])
        parked_ticks = parked_ticks+1 if index == len(plan["tasks"])-1 and obs["arrived"] and speed <= config.stop_speed_ms else 0
        last_in_stage = i == len(ticks)-1 or ticks[i+1]["task_index"] != row["task_index"]
        if last_in_stage:
            check(obs["arrived"], "Stage did not reach its gate: " + str(index))
            re_events.append(dict(event="stage_completed", index=index, stage=task["type"], time_s=t))
    check(observed == status["expected_stages"] == status["observed_stages"], "Stage sequence differs")
    check(events == re_events, "Stage events disagree with raw ticks")
    check(parked_ticks*config.fixed_delta_seconds >= config.parking_hold_s, "Parking hold not achieved")
    check(close(float(ticks[-1]["time_s"]), status["duration_s"]), "Duration differs")
    saved_path = read_csv(run_dir/"full_process_path.csv")
    check(len(saved_path) == len(tracker.history), "Path sample count differs")
    for expected, saved in zip(tracker.history, saved_path):
        check(expected["state"] == saved["state"] and
              all(close(expected[k], float(saved[k])) for k in ("x", "y")), "Path sample differs")
    expected_table = PerformanceEvaluationSubagent.task_level_summary(
        task_completed=True, stage_transition_completed=True, forward_reverse_completed=None,
        collision_count=len(contacts), total_distance_m=tracker.total_distance,
        operation_path_length_m=tracker.work_distance, task_duration_s=status["duration_s"])
    saved_table = read_csv(run_dir/"full_process_task_evaluation.csv")
    expected_rows = list(csv.DictReader(io.StringIO(expected_table.to_csv(index=False))))
    check(expected_rows == saved_table, "Evaluation table differs")
    for stats in stage_stats:
        stats["duration_s"] = stats["ticks"]*config.fixed_delta_seconds
        stats["mean_speed_ms"] = stats.pop("speed_sum")/stats["ticks"]
    last = ticks[-1]
    return dict(
        audit_passed=not errors, errors=sorted(set(errors)), run=run_dir.name,
        project_xodr_sha256=map_sha, runtime_xodr_matches_project=plan["xodr_sha256"] == map_sha,
        raw_sha256=hashes, total_ticks=len(ticks), path_samples=len(saved_path),
        total_distance_m=tracker.total_distance, harvest_line_distance_m=tracker.work_distance,
        duration_s=status["duration_s"], collision_count=len(contacts),
        final_endpoint_distance_m=float(last["endpoint_distance_m"]),
        final_heading_error_deg=float(last["heading_error_deg"]),
        final_speed_ms=float(last["speed_ms"]),
        parking_hold_s=parked_ticks*config.fixed_delta_seconds,
        observation_max_abs_difference=max_obs_delta,
        offroad_control_max_abs_difference=max_control_delta,
        stages=stage_stats,
        limitations=[
            "Offline reconciliation of author-supplied CARLA evidence, not a new CARLA run.",
            "No CrewAI/LLM execution evidence and no repeated-run success-rate estimate.",
            "BehaviorAgent was not replayed; its implementation and runtime dependencies need provenance.",
            "Distance uses unchanged PathTracker >0.05 m accumulation; harvest excludes the bulb turn.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = audit(args.run_dir)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.report:
        if args.report.resolve().parent == args.run_dir.resolve():
            raise ValueError("Store audit reports outside the immutable raw run directory")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text+"\n", encoding="utf-8")
    print(text)
    if not result["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
