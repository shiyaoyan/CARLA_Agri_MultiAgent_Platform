"""Configuration for the satellite-based Yantai full-process experiment."""
from __future__ import annotations

from dataclasses import dataclass, field
import math

_SIDE_X = 65.0 / math.hypot(7.0, 65.0)
_SIDE_Y = -7.0 / math.hypot(7.0, 65.0)


def yantai_tractor_physics():
    """Section 3.3 snapshot from the author's 2026-10-15 CARLA run."""
    return dict(
        mass=5760.0, center_of_mass=-0.2, drag_coefficient=1.0,
        tire_friction=3.5, damping_rate=1.0, max_steer_angle=48.0,
        long_stiff_value=5000.0, lat_stiff_max_load=4.0, lat_stiff_value=20.0,
        max_brake_torque=1400.0, max_rpm=2400.0, moi=1.0,
        damping_rate_full_throttle=0.15, damping_rate_zero_throttle_clutch_engaged=2.0,
        torque_curve=[(0, 300), (700, 300), (1400, 500), (2100, 500), (2700, 500)],
        torque_multiplier=1.0, gear_switch_time=0.3, final_ratio=5.28,
        forward_gears=[(15.2, 0.2, 0.6)], gear_ratio_multiplier=1.0,
    )


@dataclass
class FullProcessExperimentConfig:
    host: str = "localhost"
    port: int = 2000
    timeout: float = 30.0
    carla_map: str = "YantaiSatelliteRebuilt"
    reload_world: bool = False
    vehicle_bp: str = "vehicle.tractor.tractor"
    vehicle_type: str = "tractor"
    output_dir: str = "results/full_process_navigation/yantai_satellite"
    fixed_delta_seconds: float = 0.05
    road_speed_kmh: float = 6.0
    field_line_speed_kmh: float = 4.0
    maneuver_speed_kmh: float = 3.0
    turn_speed_kmh: float = 2.5
    max_steps: int = 24000
    apply_front_wheel_steering: bool = True
    apply_agri_physics: bool = True
    physics_overrides: dict | None = field(default_factory=yantai_tractor_physics)
    harvest_width_m: float = 3.0
    harvest_passes: int = 2
    min_turn_radius_m: float = 5.5
    path_spacing_m: float = 0.25
    wheelbase_m: float = 2.5
    max_steer_angle_deg: float = 35.0
    lookahead_m: float = 1.5
    gate_tolerance_m: float = 0.8
    heading_tolerance_deg: float = 15.0
    stop_speed_ms: float = 0.12
    parking_hold_s: float = 1.0
    max_cross_track_m: float = 4.0
    stall_timeout_s: float = 30.0
    road_projection_limit_m: float = 2.0
    stop_on_collision: bool = True
    collision_arm_distance_m: float = 1.0
    spawn_height_offset_m: float = 0.0
    use_gear_autobox: bool = True
    throttle_ramp_per_frame: float = 0.15
    # CARLA metres, not pixels or north-up OpenDRIVE Y.
    garage_spawn: tuple = (-3.0, 24.0, 0.05, -90.0)
    garage_return: tuple = (-3.0, 24.0, 0.05)
    garage_return_yaw: float = 90.0
    road_entry: tuple = (5.0, 1.0, 0.05)
    field_entry_road: tuple = (14.0, 0.6, 0.05)
    field_exit_road: tuple = (11.0, -1.7, 0.05)
    road_return: tuple = (2.0, -1.3, 0.05)
    road_outbound_yaw: float = -2.5
    road_inbound_yaw: float = 177.5
    road_gate_road_id: int = 1
    # near-left, far-left, far-right, near-right; not compass north.
    field_top_left: tuple = (20-1.5*_SIDE_X, -17-1.5*_SIDE_Y, 0.05)
    field_top_right: tuple = (13-1.5*_SIDE_X, -82-1.5*_SIDE_Y, 0.05)
    field_bottom_right: tuple = (13+4.5*_SIDE_X, -82+4.5*_SIDE_Y, 0.05)
    field_bottom_left: tuple = (20+4.5*_SIDE_X, -17+4.5*_SIDE_Y, 0.05)
    field_maneuver_boundary: tuple = ((-5.0, -106.0), (20.0, -109.0),
                                      (32.0, -8.0), (7.0, -6.0))
    vehicle_clearance_m: float = 2.5
    vegetation_request: dict | None = None

    def __post_init__(self):
        if not self.carla_map or self.carla_map.rsplit("/", 1)[-1] == "YantaiTrackingField":
            raise ValueError("Unsupported full-process map; this case requires YantaiSatelliteRebuilt.")
        positive = ("fixed_delta_seconds", "road_speed_kmh", "field_line_speed_kmh",
                    "maneuver_speed_kmh", "turn_speed_kmh", "harvest_width_m",
                    "min_turn_radius_m", "path_spacing_m", "wheelbase_m",
                    "max_steer_angle_deg", "lookahead_m", "gate_tolerance_m",
                    "heading_tolerance_deg", "stop_speed_ms", "parking_hold_s",
                    "max_cross_track_m", "stall_timeout_s", "road_projection_limit_m",
                    "vehicle_clearance_m", "throttle_ramp_per_frame")
        for key in positive:
            if not math.isfinite(float(getattr(self, key))) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if not isinstance(self.max_steps, int) or self.max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        for key in ("collision_arm_distance_m", "spawn_height_offset_m"):
            if not math.isfinite(float(getattr(self, key))) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.harvest_passes != 2:
            raise ValueError("This paper case requires two harvest passes and one bulb turn.")
        if not 0 < self.max_steer_angle_deg < 90:
            raise ValueError("max_steer_angle_deg must be between 0 and 90")
        if self.min_turn_radius_m < self.wheelbase_m / math.tan(math.radians(self.max_steer_angle_deg)):
            raise ValueError("Turning radius is smaller than the configured bicycle-model limit")
        if self.harvest_width_m >= 2 * self.min_turn_radius_m:
            raise ValueError("Omega turns require row spacing below twice the turning radius")
        for key in ("garage_spawn", "garage_return", "road_entry", "field_entry_road",
                    "field_exit_road", "road_return", "field_top_left", "field_top_right",
                    "field_bottom_right", "field_bottom_left"):
            value = getattr(self, key)
            if len(value) != (4 if key == "garage_spawn" else 3) or not all(math.isfinite(float(x)) for x in value):
                raise ValueError(f"Invalid {key} coordinates")
        for key in ("garage_return_yaw", "road_outbound_yaw", "road_inbound_yaw"):
            if not math.isfinite(float(getattr(self, key))):
                raise ValueError(f"Invalid {key}")
        from agri_carla.simulation.full_process_geometry import harvest_rows
        harvest_rows(self.field_corners, self.harvest_width_m, self.harvest_passes)

    @property
    def field_corners(self):
        return [self.field_top_left, self.field_top_right, self.field_bottom_right, self.field_bottom_left]
