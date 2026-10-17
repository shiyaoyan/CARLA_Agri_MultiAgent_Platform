"""Read-back evidence for the full-process vehicle physics, not a calibration."""
import math


SCALARS = ("mass", "drag_coefficient", "max_rpm", "moi",
           "damping_rate_full_throttle", "damping_rate_zero_throttle_clutch_engaged",
           "gear_switch_time", "final_ratio")
WHEEL_FIELDS = ("tire_friction", "damping_rate", "max_steer_angle", "long_stiff_value",
                "lat_stiff_max_load", "lat_stiff_value", "max_brake_torque")


def physics_snapshot(vehicle):
    physics = vehicle.get_physics_control()
    return {
        **{name: float(getattr(physics, name)) for name in SCALARS},
        "center_of_mass": [float(getattr(physics.center_of_mass, a)) for a in ("x", "y", "z")],
        "use_gear_autobox": bool(physics.use_gear_autobox),
        "torque_curve": [[float(p.x), float(p.y)] for p in physics.torque_curve],
        "forward_gears": [[float(g.ratio), float(g.down_ratio), float(g.up_ratio)]
                          for g in physics.forward_gears],
        "wheels": [{**{name: float(getattr(wheel, name)) for name in WHEEL_FIELDS},
                    "radius_cm": float(wheel.radius)} for wheel in physics.wheels],
    }


def physics_report(vehicle, requested, front_only, expect_autobox=None):
    """Compare requested physics; None records but does not assert gearbox mode."""
    actual = physics_snapshot(vehicle)
    expected = {}
    if expect_autobox is not None:
        expected["use_gear_autobox"] = bool(expect_autobox)
    if requested:
        expected.update({name: requested[name] for name in SCALARS})
        expected.update({
            "center_of_mass": [0.0, 0.0, requested["center_of_mass"]],
            "torque_curve": [[rpm, torque * requested["torque_multiplier"]]
                             for rpm, torque in requested["torque_curve"]],
            "forward_gears": [[ratio * requested["gear_ratio_multiplier"], down, up]
                              for ratio, down, up in requested["forward_gears"]],
            "wheels": [{name: (0.0 if front_only and i >= 2 else
                                max(35.0, requested[name]) if front_only else requested[name])
                        if name == "max_steer_angle" else requested[name]
                        for name in WHEEL_FIELDS} for i in range(4)],
        })
    differences = []

    def compare(wanted, found, path):
        if isinstance(wanted, dict):
            for key, value in wanted.items():
                compare(value, found.get(key), f"{path}.{key}")
        elif isinstance(wanted, list):
            if not isinstance(found, list) or len(wanted) != len(found):
                differences.append(f"{path}: length mismatch")
                return
            for i, (a, b) in enumerate(zip(wanted, found)):
                compare(a, b, f"{path}[{i}]")
        elif found is None or not math.isclose(float(wanted), float(found), rel_tol=1e-4, abs_tol=1e-4):
            differences.append(f"{path}: requested {wanted!r}, read back {found!r}")

    compare(expected, actual, "physics")
    return dict(custom_physics_applied=bool(requested), requested=requested,
                effective_expected=expected, actual=actual, readback_matches=not differences,
                differences=differences, front_wheel_only=front_only,
                note="Read-back verification only; not a validation of real tractor dynamics.")
