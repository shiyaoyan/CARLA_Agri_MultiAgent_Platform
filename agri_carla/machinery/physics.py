"""面向低速农业作业的 CARLA 农机物理参数及应用过程。"""

from __future__ import annotations


class AgriPhysicsConfig:
    """CARLA 农机宏观运动参数模型。

    参数用于获得与拖拉机低速、较大转弯半径和换挡响应相符的宏观表现，并非
    完整轮胎—土壤高精度动力学模型。
    """

    def __init__(self, vehicle_type: str = "tractor"):
        """设置本文已验证的拖拉机质量、轮胎、发动机和传动系统默认值。"""
        if vehicle_type.lower() != "tractor":
            raise ValueError(
                "Only the validated tractor physics template is currently supported: "
                f"{vehicle_type}"
            )
        self.mass = 5760.0
        self.center_of_mass = -0.2
        self.drag_coefficient = 1.0

        self.tire_friction = 3.5
        self.damping_rate = 1.0
        self.max_steer_angle = 48.0
        self.long_stiff_value = 8000.0
        self.lat_stiff_max_load = 4.0
        self.lat_stiff_value = 20.0
        self.max_brake_torque = 1400.0

        self.max_rpm = 2400.0
        self.moi = 1.0
        self.damping_rate_full_throttle = 0.15
        self.damping_rate_zero_throttle_clutch_engaged = 2.0

        self.torque_curve = [
            (700, 200),
            (1400, 500),
            (1500, 500),
        ]
        self.torque_multiplier = 1.0

        self.gear_switch_time = 0.3
        self.final_ratio = 5.28
        self.forward_gears = [
            (15.2, 0.2, 0.6),
            (19.8, 0.2, 0.5),
            (29.35, 0.2, 0.4),
        ]
        self.gear_ratio_multiplier = 1.0

    def to_dict(self) -> dict[str, object]:
        """导出当前全部可配置物理字段。"""
        return {
            "mass": self.mass,
            "center_of_mass": self.center_of_mass,
            "drag_coefficient": self.drag_coefficient,
            "tire_friction": self.tire_friction,
            "damping_rate": self.damping_rate,
            "max_steer_angle": self.max_steer_angle,
            "long_stiff_value": self.long_stiff_value,
            "lat_stiff_max_load": self.lat_stiff_max_load,
            "lat_stiff_value": self.lat_stiff_value,
            "max_brake_torque": self.max_brake_torque,
            "max_rpm": self.max_rpm,
            "moi": self.moi,
            "damping_rate_full_throttle": self.damping_rate_full_throttle,
            "damping_rate_zero_throttle_clutch_engaged": self.damping_rate_zero_throttle_clutch_engaged,
            "torque_curve": self.torque_curve,
            "torque_multiplier": self.torque_multiplier,
            "gear_switch_time": self.gear_switch_time,
            "final_ratio": self.final_ratio,
            "forward_gears": self.forward_gears,
            "gear_ratio_multiplier": self.gear_ratio_multiplier,
        }


def apply_agri_physics(vehicle, config: AgriPhysicsConfig) -> None:
    """将农机参数写入已生成的 CARLA 车辆 actor。

    该函数会覆盖 actor 的质量、发动机、挡位和车轮参数；全流程导航若需使用
    CARLA 默认车辆响应，可由实验配置选择不调用本函数。
    """
    import carla

    physics_control = vehicle.get_physics_control()

    physics_control.mass = config.mass
    physics_control.drag_coefficient = config.drag_coefficient
    physics_control.center_of_mass = carla.Vector3D(0, 0, config.center_of_mass)

    physics_control.max_rpm = config.max_rpm
    physics_control.moi = config.moi
    physics_control.damping_rate_full_throttle = config.damping_rate_full_throttle
    physics_control.damping_rate_zero_throttle_clutch_engaged = config.damping_rate_zero_throttle_clutch_engaged

    physics_control.torque_curve = [
        carla.Vector2D(x=rpm, y=torque * config.torque_multiplier)
        for rpm, torque in config.torque_curve
    ]

    physics_control.final_ratio = config.final_ratio
    physics_control.gear_switch_time = config.gear_switch_time

    # 将简化的挡位元组转换为 CARLA GearPhysicsControl 对象列表。
    gear_list = []
    for ratio, down, up in config.forward_gears:
        gear = carla.GearPhysicsControl()
        gear.ratio = ratio * config.gear_ratio_multiplier
        gear.down_ratio = down
        gear.up_ratio = up
        gear_list.append(gear)
    physics_control.forward_gears = gear_list

    # CARLA 要求修改后重新设置完整车轮列表，不能只原地改变单个参数。
    new_wheels = []
    for wheel in physics_control.wheels:
        wheel.tire_friction = config.tire_friction
        wheel.damping_rate = config.damping_rate
        wheel.max_steer_angle = config.max_steer_angle
        wheel.long_stiff_value = config.long_stiff_value
        wheel.lat_stiff_max_load = config.lat_stiff_max_load
        wheel.lat_stiff_value = config.lat_stiff_value
        wheel.max_brake_torque = config.max_brake_torque
        new_wheels.append(wheel)

    physics_control.wheels = new_wheels
    vehicle.apply_physics_control(physics_control)
