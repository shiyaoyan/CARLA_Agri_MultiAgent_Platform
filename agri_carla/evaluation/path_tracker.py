"""全流程导航的路径、阶段和里程记录器。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TypedDict

import pandas as pd


class PathHistoryPoint(TypedDict):
    """一条带导航阶段标签的二维路径采样。"""

    x: float
    y: float
    state: str


class PathTracker:
    """记录全流程车辆位置，并分别累计总里程和田间作业里程。"""

    def __init__(self):
        """初始化空路径及里程累计器。"""
        self.history: list[PathHistoryPoint] = []
        self.total_distance = 0.0
        self.work_distance = 0.0
        self.last_loc: tuple[float, float] | None = None

    def update(self, carla_loc, task_type: str | None) -> None:
        """接收当前车辆位置和阶段标签，更新路径与距离。

        位移小于或等于 0.05 m 的连续采样不写入历史，以降低车辆静止时的
        冗余记录；只有 ``HARVEST_LINE`` 阶段的位移计入作业路径长度。
        """
        curr_x, curr_y = float(carla_loc.x), float(carla_loc.y)
        state_tag = task_type or "IDLE"
        is_working = state_tag == "HARVEST_LINE"
        if self.last_loc:
            dist = math.sqrt((curr_x - self.last_loc[0]) ** 2 + (curr_y - self.last_loc[1]) ** 2)
            if dist > 0.05:
                self.total_distance += dist
                if is_working:
                    self.work_distance += dist
                self.history.append({"x": curr_x, "y": curr_y, "state": state_tag})
                self.last_loc = (curr_x, curr_y)
        else:
            self.last_loc = (curr_x, curr_y)
            self.history.append({"x": curr_x, "y": curr_y, "state": state_tag})

    @property
    def efficiency_percent(self) -> float:
        """返回作业路径长度占总行驶距离的百分比。"""
        if self.total_distance <= 0:
            return 0.0
        return 100.0 * self.work_distance / self.total_distance

    def to_dataframe(self) -> pd.DataFrame:
        """将路径历史转换为包含 x、y 和阶段标签的数据表。"""
        return pd.DataFrame(self.history)

    def save_csv(self, path: str | Path) -> Path:
        """以 UTF-8 BOM 编码保存路径数据，便于表格软件读取。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(path, index=False, encoding="utf-8-sig")
        return path

    def save_plot(self, filename: str | Path) -> Path | None:
        """按导航阶段着色绘制全流程定位路径。"""
        if not self.history:
            return None
        import matplotlib.pyplot as plt

        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        color_map = {
            "GARAGE_EXIT": "#d62728",
            "ROAD_NAV": "#17becf",
            "OFFROAD_APPROACH": "#ff7f0e",
            "HARVEST_LINE": "#2ca02c",
            "TURN_BULB": "#bcbd22",
            "FIELD_EXIT": "#9467bd",
            "ROAD_RETURN": "#1f77b4",
            "GARAGE_RETURN": "#8c564b",
            "IDLE": "#7f7f7f",
        }
        label_map = {
            "GARAGE_EXIT": "Garage Exit",
            "ROAD_NAV": "Road to Field",
            "OFFROAD_APPROACH": "Enter Field",
            "HARVEST_LINE": "Operation Line",
            "TURN_BULB": "Bulb Turn",
            "FIELD_EXIT": "Exit Field",
            "ROAD_RETURN": "Road Return",
            "GARAGE_RETURN": "Garage Return",
            "IDLE": "Idle",
        }

        plt.figure(figsize=(10, 10))
        plt.grid(True, linestyle="--", alpha=0.5)
        x_list: list[float] = []
        y_list: list[float] = []
        current_state = self.history[0]["state"]
        labels_added: set[str] = set()
        for point in self.history:
            state = point["state"]
            if state != current_state:
                # 阶段变化时结束上一段曲线，并保留连接点以避免视觉断裂。
                color = color_map.get(current_state, "#7f7f7f")
                label = None if current_state in labels_added else label_map.get(current_state, current_state)
                labels_added.add(current_state)
                plt.plot(x_list, y_list, color=color, linewidth=2, label=label)
                x_list = [x_list[-1], point["x"]] if x_list else [point["x"]]
                y_list = [y_list[-1], point["y"]] if y_list else [point["y"]]
                current_state = state
            else:
                x_list.append(point["x"])
                y_list.append(point["y"])

        color = color_map.get(current_state, "#7f7f7f")
        label = None if current_state in labels_added else label_map.get(current_state, current_state)
        plt.plot(x_list, y_list, color=color, linewidth=2, label=label)
        start = self.history[0]
        end = self.history[-1]
        plt.scatter(start["x"], start["y"], c="blue", marker="^", s=100, label="Start", zorder=5)
        plt.scatter(end["x"], end["y"], c="black", marker="x", s=100, label="End", zorder=5)
        plt.title(
            "Agricultural Path Analysis\n"
            f"Total: {self.total_distance:.1f} m | "
            f"Operation: {self.work_distance:.1f} m | "
            f"Efficiency: {self.efficiency_percent:.1f}%"
        )
        plt.xlabel("Map X (m)")
        plt.ylabel("Map Y (m)")
        plt.legend()
        plt.axis("equal")
        plt.savefig(filename, dpi=150)
        plt.close()
        return filename
