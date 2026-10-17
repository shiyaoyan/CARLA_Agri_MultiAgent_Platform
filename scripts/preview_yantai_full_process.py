"""Offline mission preview from the exported XODR; never a CARLA run."""
import argparse
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agri_carla.simulation.full_process_config import FullProcessExperimentConfig
from agri_carla.simulation.full_process_geometry import distance, heading
from agri_carla.simulation.task_planner import FullProcessTaskPlanner, serializable_tasks, xyz
from agri_carla.utils.io import write_json
from roadrunner_mcp.road_mesh import sample_lanes


class OfflineFarmRoad:
    """Sampled same-road adapter for this case, not a replacement CARLA router."""
    def __init__(self, xodr_path, road_id=1):
        self.root = ET.parse(xodr_path).getroot()
        road = next(r for r in self.root.findall("road") if int(r.get("id")) == road_id)
        self.waypoints = []
        self.lanes = {}
        for lane_id, samples in sample_lanes(road, 0.25).items():
            if lane_id > 0:
                samples.reverse()
            points = [(p.center[0], -p.center[1], 0.0) for p in samples]
            waypoints = []
            for i, (sample, p) in enumerate(zip(samples, points)):
                yaw = heading(points[i-1], p) if i == len(points)-1 else heading(p, points[i+1])
                waypoints.append(SimpleNamespace(road_id=road_id, section_id=0, lane_id=lane_id,
                    s=sample.s, is_junction=False, transform=SimpleNamespace(
                        location=SimpleNamespace(x=p[0], y=p[1], z=p[2]), rotation=SimpleNamespace(yaw=yaw))))
            self.lanes[lane_id] = waypoints
            self.waypoints.extend(waypoints)

    def generate_waypoints(self, spacing):
        return self.waypoints

    def trace_route(self, start, end):
        start = min(self.waypoints, key=lambda w: distance(xyz(w.transform.location), xyz(start)))
        end = min(self.waypoints, key=lambda w: distance(xyz(w.transform.location), xyz(end)))
        if start.lane_id != end.lane_id:
            return []
        lane = self.lanes[start.lane_id]
        a, b = lane.index(start), lane.index(end)
        return [(wp, None) for wp in lane[a:b+1]] if b > a else []


def build_preview(config, xodr):
    road = OfflineFarmRoad(xodr, config.road_gate_road_id)
    return FullProcessTaskPlanner(None, road, config, road).build(), road


def plot_plan(tasks, road, config, output, satellite=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    colors = {"GARAGE_EXIT": "#9255a2", "ROAD_NAV": "#196db5", "OFFROAD_APPROACH": "#e58225",
              "HARVEST_LINE": "#238249", "TURN_BULB": "#c2363a", "FIELD_EXIT": "#ca8d25",
              "ROAD_RETURN": "#2896a2", "GARAGE_RETURN": "#614780"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 9), layout="constrained")
    for ax in axes:
        if satellite:
            from PIL import Image
            image = Image.open(satellite)
            if image.size != (1532, 2040):
                raise ValueError("Satellite overlay requires the historical 1532x2040 source, not a resized crop")
            ax.imshow(image, extent=(-520*.09, (1532-520)*.09, (2040-1319.1)*.09, -1319.1*.09))
        else:
            ax.set_facecolor("#f0f4ed")
        for r in road.root.findall("road"):
            for samples in sample_lanes(r).values():
                poly = [p.inner for p in samples] + [p.outer for p in reversed(samples)]
                ax.add_patch(Polygon([(x,-y) for x,y in poly], color="#808888", alpha=.5, linewidth=0))
        ax.add_patch(Polygon(config.field_maneuver_boundary, fill=False, edgecolor="#297a4c", linestyle="--", linewidth=1))
        used = set()
        for t in tasks:
            x, y, _ = zip(*t["points"])
            label = t["type"].replace("_", " ").title() if t["type"] not in used else None
            ax.plot(x, y, color=colors[t["type"]], lw=2, label=label)
            used.add(t["type"])
        ax.scatter(*config.garage_spawn[:2], marker="s", color="#26353d", s=35, zorder=10)
        ax.set_aspect("equal")
        ax.set_xlabel("CARLA X (m)")
        ax.set_ylabel("CARLA Y (m), north up")
        ax.grid(alpha=.18)
    axes[0].set_xlim(-15, 37)
    axes[0].set_ylim(30, -111)
    axes[0].set_title("Full mission: planned reference")
    axes[1].set_xlim(-11, 31)
    axes[1].set_ylim(29, -24)
    axes[1].set_title("Garage / road / field transitions")
    axes[1].legend(loc="upper left", fontsize=8)
    fig.suptitle("YantaiSatelliteRebuilt | two harvest passes + one bulb turn", fontsize=13)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(ROOT / "configs/yantai_full_process_agent_request.json"))
    p.add_argument("--xodr", default=str(ROOT / "data/maps/yantai_satellite/YantaiSatelliteRebuilt.xodr"))
    p.add_argument("--output-dir", default="outputs/yantai_full_process_preview")
    p.add_argument("--satellite", default="")
    args = p.parse_args()
    request = json.loads(Path(args.config).read_text())
    config = FullProcessExperimentConfig(**request.get("experiment", {}).get("config", request))
    tasks, road = build_preview(config, args.xodr)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "planned_mission.json", dict(status="OFFLINE_PLANNED_ONLY", carla_simulation_executed=False,
        map=config.carla_map, coordinate_frame="CARLA metres, Y south", tasks=serializable_tasks(tasks)))
    plot_plan(tasks, road, config, out / "mission_preview.png")
    if args.satellite:
        plot_plan(tasks, road, config, out / "satellite_overlay.png", args.satellite)
    print(json.dumps({"status": "OFFLINE_PLANNED_ONLY", "tasks": [t["type"] for t in tasks],
                      "planned_length_m": sum(t["length_m"] for t in tasks), "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
