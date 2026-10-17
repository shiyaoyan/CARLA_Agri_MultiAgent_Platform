"""Continuous low-speed path control and measured stage-arrival checks."""
import math
from types import SimpleNamespace

from agri_carla.controllers.pid import PIDController
from agri_carla.controllers.pure_pursuit import PurePursuitConfig, PurePursuitController
from agri_carla.simulation.full_process_geometry import angle_error, distance


class MissionPathFollower:
    def __init__(self, task, config):
        self.task, self.config = task, config
        self.points = task["points"]
        self.index = 0
        self.cumulative = [0.0]
        for a, b in zip(self.points, self.points[1:]):
            self.cumulative.append(self.cumulative[-1] + distance(a, b))
        # A tangent extension is used for steering only, never mission progress.
        yaw = math.radians(task["end_yaw"])
        end = self.points[-1]
        extended = self.points + [(end[0]+i*0.25*math.cos(yaw), end[1]+i*0.25*math.sin(yaw), end[2])
                                  for i in range(1, 21)]
        trajectory = SimpleNamespace(x=[p[0] for p in extended], y=[p[1] for p in extended],
                                     num_points=len(extended), is_forward=[True]*len(extended))
        self.pursuit = PurePursuitController(trajectory, config=PurePursuitConfig(
            wheelbase=config.wheelbase_m, max_steer_angle_deg=config.max_steer_angle_deg,
            fwd_lookahead_min=config.lookahead_m, fwd_lookahead_gain=0.2))
        self.pid = PIDController()
        self.last_throttle = 0.0

    def observe(self, position, yaw):
        # Local arc-length search prevents a narrow two-pass route or bulb
        # from jumping onto its nearby return branch.
        stop = self.index + 1
        while stop < len(self.points) and self.cumulative[stop]-self.cumulative[self.index] < 3.0:
            stop += 1
        self.index = min(range(self.index, stop), key=lambda i: distance(position, self.points[i]))
        remaining = self.cumulative[-1] - self.cumulative[self.index]
        endpoint_distance = distance(position, self.points[-1])
        yaw_error = abs(angle_error(yaw, self.task["end_yaw"]))
        return dict(progress_m=self.cumulative[self.index], remaining_m=remaining,
                    cross_track_m=distance(position, self.points[self.index]),
                    endpoint_distance_m=endpoint_distance, heading_error_deg=yaw_error,
                    arrived=(remaining <= self.config.gate_tolerance_m
                             and endpoint_distance <= self.config.gate_tolerance_m
                             and yaw_error <= self.config.heading_tolerance_deg))

    def control(self, carla, position, yaw, speed, observation):
        self.pursuit.index = self.index
        steer, _, _, _ = self.pursuit.compute_steer(position[0], position[1], yaw, speed, True)
        target = self.task["speed_kmh"] / 3.6
        if self.task["type"] == "GARAGE_RETURN":
            remaining = observation["remaining_m"]
            target = min(target, math.sqrt(1.2 * max(0.0, remaining-self.config.gate_tolerance_m*0.5)))
            if observation["arrived"]:
                target = 0.0
        throttle, brake, _ = self.pid.step(target, speed, self.config.fixed_delta_seconds)
        ramp = getattr(self.config, "throttle_ramp_per_frame", 0.15)
        throttle = min(throttle, self.last_throttle + ramp)
        self.last_throttle = throttle
        manual = not bool(getattr(self.config, "use_gear_autobox", True))
        return carla.VehicleControl(throttle=throttle, brake=brake, steer=steer,
                                    reverse=False, manual_gear_shift=manual, gear=1, hand_brake=False)


def parked(observation, speed, config):
    return observation["arrived"] and speed <= config.stop_speed_ms
