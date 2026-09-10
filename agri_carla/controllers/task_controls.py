"""全流程导航各任务阶段使用的轻量车辆控制辅助函数。"""

from __future__ import annotations

import math


def speed_control(carla, vehicle, target_speed_kmh: float, throttle_limit: float = 0.5):
    """根据速度区间生成简单油门和制动指令。"""
    control = carla.VehicleControl()
    v = vehicle.get_velocity()
    cur_speed = 3.6 * math.sqrt(v.x ** 2 + v.y ** 2)
    if cur_speed < target_speed_kmh - 0.3:
        control.throttle = min(throttle_limit, 0.18 + 0.06 * (target_speed_kmh - cur_speed))
        control.brake = 0.0
    elif cur_speed > target_speed_kmh + 0.4:
        control.throttle = 0.0
        control.brake = min(0.5, 0.08 * (cur_speed - target_speed_kmh))
    else:
        control.throttle = 0.12
        control.brake = 0.0
    return control


def limit_road_speed(vehicle, control, target_speed_kmh: float = 13.5):
    """限制 BehaviorAgent 道路阶段输出，适配拖拉机低速行驶。"""
    v = vehicle.get_velocity()
    cur_speed = 3.6 * math.sqrt(v.x ** 2 + v.y ** 2)
    if cur_speed > target_speed_kmh + 0.5:
        control.throttle = 0.0
        control.brake = max(control.brake, 0.18)
    elif cur_speed < target_speed_kmh - 0.5:
        control.throttle = min(max(control.throttle, 0.55), 0.75)
        control.brake = 0.0
    else:
        control.throttle = min(max(control.throttle, 0.18), 0.35)
        control.brake = 0.0
    return control


def set_agent_target_speed(navigation_controller, speed_kmh: float) -> None:
    """兼容不同 CARLA 版本设置 BehaviorAgent 目标速度。"""
    try:
        navigation_controller.set_target_speed(speed_kmh)
        return
    except Exception:
        pass
    try:
        navigation_controller._local_planner.set_speed(speed_kmh)
        return
    except Exception:
        pass
    try:
        navigation_controller._behavior.max_speed = speed_kmh
        navigation_controller._behavior.speed_lim_dist = 0
    except Exception:
        pass


def should_advance_point_task(task: dict, distance: float, threshold: float, overshoot_margin: float = 1.5) -> bool:
    """判断点目标任务已到达或轻微越过，避免车辆在目标附近反复徘徊。"""
    if distance < threshold:
        return True
    best = task.get("best_distance")
    if best is None or distance < best:
        task["best_distance"] = distance
        task["stuck_ticks"] = 0
        return False
    task["stuck_ticks"] = task.get("stuck_ticks", 0) + 1
    if best < threshold + overshoot_margin and distance > best + 0.6:
        return True
    if task["stuck_ticks"] > 80 and best < threshold + overshoot_margin:
        return True
    return False


def enforce_front_wheel_steering(vehicle) -> None:
    """确保前轮可转向并关闭后轮转向，符合本文拖拉机蓝图设定。"""
    try:
        physics = vehicle.get_physics_control()
        if len(physics.wheels) >= 4:
            front_left = physics.wheels[0].max_steer_angle
            front_right = physics.wheels[1].max_steer_angle
            physics.wheels[0].max_steer_angle = max(front_left, 35.0)
            physics.wheels[1].max_steer_angle = max(front_right, 35.0)
            physics.wheels[2].max_steer_angle = 0.0
            physics.wheels[3].max_steer_angle = 0.0
            vehicle.apply_physics_control(physics)
            print("Front-wheel steering enforced for the agricultural vehicle.")
    except Exception as exc:
        print(f"Front-wheel steering setup failed, continuing with current physics: {exc}")


def get_offroad_control(carla, vehicle, target_location, mode: str = "NORMAL"):
    """为入田、地头转弯和停车区点目标生成非道路控制指令。"""
    trans = vehicle.get_transform()
    target_vec = target_location - trans.location
    target_yaw = math.atan2(target_vec.y, target_vec.x)
    veh_yaw = math.radians(trans.rotation.yaw)
    diff = target_yaw - veh_yaw
    while diff > math.pi:
        diff -= 2 * math.pi
    while diff < -math.pi:
        diff += 2 * math.pi

    control = carla.VehicleControl()
    steer_gain = 0.8 if mode in ["TURN", "FIELD_ENTRY"] else 1.2
    steer_limit = 0.65 if mode == "FIELD_ENTRY" else (0.75 if mode == "TURN" else 1.0)
    control.steer = max(-steer_limit, min(steer_limit, diff * steer_gain))

    v = vehicle.get_velocity()
    cur_speed = 3.6 * math.sqrt(v.x ** 2 + v.y ** 2)
    if mode == "TURN":
        target_speed, throttle_val = 5.0, 0.45
    elif mode == "GARAGE":
        target_speed, throttle_val = 4.0, 0.28
    elif mode == "FIELD_ENTRY":
        target_speed, throttle_val = 5.0, 0.50
    else:
        target_speed, throttle_val = 13.5, 0.48

    if cur_speed < target_speed:
        control.throttle = throttle_val
        control.brake = 0.0
        if mode in ["FIELD_ENTRY", "TURN"] and cur_speed < 0.8:
            control.throttle = max(control.throttle, 0.58)
    else:
        control.throttle = 0.0
        control.brake = 0.1

    control.manual_gear_shift = True
    control.gear = 1
    control.hand_brake = False
    return control, trans.location.distance(target_location)


def get_linear_control(carla, vehicle, start_loc, end_loc, target_speed: float = 7.5):
    """根据航向误差和横向偏差控制田间直线作业段。"""
    trans = vehicle.get_transform()
    veh_loc = trans.location
    path_vec = end_loc - start_loc
    path_len = math.sqrt(path_vec.x ** 2 + path_vec.y ** 2) + 0.0001
    path_angle = math.atan2(path_vec.y, path_vec.x)
    veh_yaw = math.radians(trans.rotation.yaw)
    heading_error = path_angle - veh_yaw
    while heading_error > math.pi:
        heading_error -= 2 * math.pi
    while heading_error < -math.pi:
        heading_error += 2 * math.pi

    veh_vec = veh_loc - start_loc
    cross_product = path_vec.x * veh_vec.y - path_vec.y * veh_vec.x
    cte = cross_product / path_len
    steer_angle = heading_error - math.atan2(1.0 * cte, 5.0)

    control = carla.VehicleControl()
    control.steer = max(-0.75, min(0.75, steer_angle))

    v = vehicle.get_velocity()
    cur_speed = 3.6 * math.sqrt(v.x ** 2 + v.y ** 2)
    if cur_speed < target_speed - 0.3:
        control.throttle = 0.55
        control.brake = 0.0
        if cur_speed < 0.8:
            control.throttle = 0.70
    elif cur_speed < target_speed + 0.4:
        control.throttle = 0.18
        control.brake = 0.0
    else:
        control.throttle = 0.0
        control.brake = 0.1
    control.manual_gear_shift = True
    control.gear = 1
    control.hand_brake = False
    return control, veh_loc.distance(end_loc), cte
