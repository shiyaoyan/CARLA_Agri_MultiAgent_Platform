"""全流程农业导航案例的任务队列生成工具。"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class FullProcessTaskConfig:
    """田间往复式作业与灯泡掉头的几何参数。"""
    harvest_width_m: float = 3.0
    bulb_overshoot_m: float = 5.0
    bulb_swing_m: float = 8.0
    bulb_side_m: float = 2.0


def location_distance_xy(a, b) -> float:
    """计算两个 CARLA 位置在水平面内的欧氏距离。"""
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def lerp_location(carla, a, b, ratio: float):
    """按比例在两个位置之间进行线性插值，并把比例限制在 [0, 1]。"""
    ratio = max(0.0, min(1.0, ratio))
    return carla.Location(
        x=a.x + (b.x - a.x) * ratio,
        y=a.y + (b.y - a.y) * ratio,
        z=a.z + (b.z - a.z) * ratio,
    )


def generate_boustrophedon_path(carla, field_corners, tool_width: float):
    """根据四个田块角点和作业幅宽生成往复式田间作业端点。

    相邻作业行的行进方向交替，返回值依次存放每条作业行的起点和终点。
    此处生成的是任务级几何路径，不涉及覆盖率优化或避障规划。
    """
    top_left, top_right, bottom_right, bottom_left = field_corners
    width_m = location_distance_xy(top_left, bottom_left)
    path_points = []
    num_passes = int(math.ceil(width_m / max(tool_width, 0.1)))
    for i in range(num_passes):
        offset = min(i * tool_width, width_m)
        ratio = offset / max(width_m, 0.001)
        left_point = lerp_location(carla, top_left, bottom_left, ratio)
        right_point = lerp_location(carla, top_right, bottom_right, ratio)
        if i % 2 == 0:
            p_start = left_point
            p_end = right_point
        else:
            p_start = right_point
            p_end = left_point
        path_points.append(p_start)
        path_points.append(p_end)
    return path_points


def make_bulb_turn_tasks(carla, prev_start, prev_end, curr_start, config: FullProcessTaskConfig, desc_prefix: str):
    """在相邻作业行之间构造灯泡掉头的四个顺序目标点。"""
    # ``dir_fwd`` 表示上一条作业行的延伸方向。
    vec_fwd = prev_end - prev_start
    len_fwd = math.sqrt(vec_fwd.x ** 2 + vec_fwd.y ** 2) + 0.001
    dir_fwd = carla.Location(vec_fwd.x / len_fwd, vec_fwd.y / len_fwd, 0)

    # ``dir_side`` 表示从上一行末端指向下一行起点的横向方向。
    vec_side = curr_start - prev_end
    len_side = math.sqrt(vec_side.x ** 2 + vec_side.y ** 2) + 0.001
    dir_side = carla.Location(vec_side.x / len_side, vec_side.y / len_side, 0)

    # p1 先向前越过地头，p2 向外摆动，p3 回到下一行延长线，最后进入下一行。
    p1 = carla.Location(
        x=prev_end.x + dir_fwd.x * config.bulb_overshoot_m,
        y=prev_end.y + dir_fwd.y * config.bulb_overshoot_m,
        z=prev_end.z,
    )
    p2 = carla.Location(
        x=prev_end.x + dir_fwd.x * config.bulb_swing_m + dir_side.x * config.bulb_side_m,
        y=prev_end.y + dir_fwd.y * config.bulb_swing_m + dir_side.y * config.bulb_side_m,
        z=prev_end.z,
    )
    p3 = carla.Location(
        x=curr_start.x + dir_fwd.x * config.bulb_overshoot_m,
        y=curr_start.y + dir_fwd.y * config.bulb_overshoot_m,
        z=curr_start.z,
    )
    return [
        {"type": "TURN_BULB", "target": p1, "desc": f"{desc_prefix}: Overshoot"},
        {"type": "TURN_BULB", "target": p2, "desc": f"{desc_prefix}: Swing Out"},
        {"type": "TURN_BULB", "target": p3, "desc": f"{desc_prefix}: Align"},
        {"type": "TURN_BULB", "target": curr_start, "desc": f"{desc_prefix}: Next Row Entry"},
    ]


class FullProcessTaskPlanner:
    """按实际作业顺序生成全流程农业导航任务队列。"""

    def __init__(self, carla, carla_map, config: FullProcessTaskConfig | None = None):
        """保存 CARLA 类型、当前地图及任务几何参数。"""
        self.carla = carla
        self.carla_map = carla_map
        self.config = config or FullProcessTaskConfig()

    def build(
        self,
        road_entry_loc,
        field_entry_point,
        field_exit_point,
        field_exit_road_point,
        garage_return_loc,
        field_corners,
    ) -> list[dict]:
        """生成从停车区出发、田间作业到返回停车区的有序任务。

        道路入口和入田点先投影到 CARLA 可驾驶道路，以供道路导航器使用；
        田块内部路径则直接使用给定局部坐标。每个字典的 ``type`` 决定实验
        运行器采用道路导航、点目标控制还是直线作业控制。
        """
        road_entry_wp = self.carla_map.get_waypoint(
            road_entry_loc,
            project_to_road=True,
            lane_type=self.carla.LaneType.Driving,
        )
        road_entry_fixed = road_entry_wp.transform.location
        field_entry_wp = self.carla_map.get_waypoint(field_entry_point, project_to_road=True)
        field_entry_road_loc = field_entry_wp.transform.location
        harvest_path = generate_boustrophedon_path(self.carla, field_corners, self.config.harvest_width_m)

        # 前四项依次对应出库、机耕道行驶、离开道路入田和到达首条作业行。
        task_queue: list[dict] = [
            {"type": "GARAGE_EXIT", "target": road_entry_fixed, "desc": "Phase 1: Garage Exit"},
            {"type": "ROAD_NAV", "target": field_entry_road_loc, "desc": "Phase 2: Road to Field"},
            {"type": "OFFROAD_APPROACH", "target": field_entry_point, "desc": "Phase 3: Enter Field Entry Point"},
            {"type": "OFFROAD_APPROACH", "target": harvest_path[0], "desc": "Phase 3: Move to First Operation Line"},
        ]

        for i in range(0, len(harvest_path), 2):
            p_start = harvest_path[i]
            p_end = harvest_path[i + 1]
            if i > 0:
                # 除第一条作业行外，每次换行前插入完整的灯泡掉头目标序列。
                prev_start = harvest_path[i - 2]
                prev_end = harvest_path[i - 1]
                task_queue.extend(
                    make_bulb_turn_tasks(
                        self.carla,
                        prev_start,
                        prev_end,
                        p_start,
                        self.config,
                        desc_prefix=f"Bulb Turn {i // 2}",
                    )
                )
            task_queue.append({
                "type": "HARVEST_LINE",
                "start": p_start,
                "target": p_end,
                "desc": f"Operation Line {i // 2 + 1}",
            })

        task_queue.extend([
            # 田间作业结束后依次离田、回到道路、沿路返回并驶入停车区。
            {"type": "FIELD_EXIT", "target": field_exit_point, "desc": "Phase 5: Exit Field"},
            {"type": "FIELD_EXIT", "target": field_exit_road_point, "desc": "Phase 5: Move to Exit Road"},
            {"type": "ROAD_RETURN", "target": road_entry_fixed, "desc": "Phase 6: Road Return"},
            {"type": "GARAGE_RETURN", "target": garage_return_loc, "desc": "Phase 7: Back to Garage"},
        ])
        return task_queue
