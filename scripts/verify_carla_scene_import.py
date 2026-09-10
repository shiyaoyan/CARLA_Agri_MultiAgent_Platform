"""连接 CARLA 并验证农业场景是否已成功导入和生成可导航路网。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agri_carla.utils.carla_import import import_carla


def _lane_key(waypoint) -> tuple[int, int, int]:
    """返回CARLA车道在路网图中的稳定标识。"""
    return int(waypoint.road_id), int(waypoint.section_id), int(waypoint.lane_id)


def _lane_successors(waypoint, spacing: float) -> set[tuple[int, int, int]]:
    """沿当前车道走到末端，返回跨道路后的后继车道。"""
    current_key = _lane_key(waypoint)
    lane_points = waypoint.next_until_lane_end(max(0.5, spacing))
    lane_end = lane_points[-1] if lane_points else waypoint
    for distance in (0.5, 1.0, 2.0, 5.0):
        successors = {
            _lane_key(candidate)
            for candidate in lane_end.next(distance)
            if _lane_key(candidate) != current_key
        }
        if successors:
            return successors
    return set()


def _runtime_topology_summary(waypoints: list, spacing: float) -> dict[str, object]:
    """汇总导入后双向车道和穿越路口的实际有向通行动作。"""
    representatives: dict[tuple[int, int, int], object] = {}
    junction_lane_keys: set[tuple[int, int, int]] = set()
    for waypoint in waypoints:
        key = _lane_key(waypoint)
        representatives.setdefault(key, waypoint)
        if waypoint.is_junction:
            junction_lane_keys.add(key)

    successors = {
        key: _lane_successors(waypoint, spacing)
        for key, waypoint in representatives.items()
    }
    normal_road_lanes: dict[int, set[int]] = {}
    for key in representatives:
        road_id, _, lane_id = key
        if key not in junction_lane_keys:
            normal_road_lanes.setdefault(road_id, set()).add(lane_id)

    bidirectional_road_ids = sorted(
        road_id
        for road_id, lane_ids in normal_road_lanes.items()
        if any(lane_id > 0 for lane_id in lane_ids)
        and any(lane_id < 0 for lane_id in lane_ids)
    )
    one_way_road_ids = sorted(set(normal_road_lanes) - set(bidirectional_road_ids))

    directed_movements: set[tuple[int, int]] = set()
    for start_key in representatives:
        if start_key in junction_lane_keys:
            continue
        start_road = start_key[0]
        frontier = [(candidate, candidate in junction_lane_keys) for candidate in successors.get(start_key, set())]
        visited: set[tuple[tuple[int, int, int], bool]] = set()
        while frontier:
            current, entered_junction = frontier.pop()
            state = (current, entered_junction)
            if state in visited:
                continue
            visited.add(state)
            if len(visited) > max(16, len(representatives) * 2):
                break
            if entered_junction and current not in junction_lane_keys:
                if current[0] != start_road:
                    directed_movements.add((start_road, current[0]))
                continue
            for candidate in successors.get(current, set()):
                frontier.append((candidate, entered_junction or candidate in junction_lane_keys))

    return {
        "normal_road_lane_ids": {
            str(road_id): sorted(lane_ids)
            for road_id, lane_ids in sorted(normal_road_lanes.items())
        },
        "bidirectional_road_ids": bidirectional_road_ids,
        "one_way_road_ids": one_way_road_ids,
        "junction_lane_count": len(junction_lane_keys),
        "junction_directed_movements": [
            {"incoming_road": incoming, "outgoing_road": outgoing}
            for incoming, outgoing in sorted(directed_movements)
        ],
        "junction_directed_movement_count": len(directed_movements),
    }


def main() -> None:
    """加载指定地图，检查地图名称、waypoint 和拓扑连接并输出报告。"""
    parser = argparse.ArgumentParser(
        description="Verify an imported CARLA agricultural scene and its road topology."
    )
    parser.add_argument("--map", default="YantaiAnnotatedFarmScene")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--waypoint-spacing", type=float, default=2.0)
    parser.add_argument("--min-waypoints", type=int, default=6)
    parser.add_argument("--min-topology-edges", type=int, default=3)
    parser.add_argument(
        "--require-bidirectional",
        action="store_true",
        help="Require every non-junction road to contain positive and negative driving lanes.",
    )
    parser.add_argument(
        "--min-junction-movements",
        type=int,
        default=0,
        help="Minimum number of directed movements that must cross junction lanes.",
    )
    parser.add_argument(
        "--list-maps",
        action="store_true",
        help="List maps visible to the connected simulator without loading one.",
    )
    parser.add_argument(
        "--output",
        default="outputs/carla_import_verification.json",
    )
    args = parser.parse_args()

    carla = import_carla()
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    available_maps = list(client.get_available_maps())
    if args.list_maps:
        print(json.dumps({"available_maps": available_maps}, indent=2))
        return

    # load_world 同时验证地图资源能否被 CARLA 正确解析；若名称不正确，
    # CARLA 会在此处给出明确错误，用户可先通过 --list-maps 查看完整路径。
    world = client.load_world(args.map)
    carla_map = world.get_map()
    topology = list(carla_map.get_topology())
    waypoints = list(carla_map.generate_waypoints(args.waypoint_spacing))
    runtime_topology = _runtime_topology_summary(waypoints, args.waypoint_spacing)
    loaded_name = str(carla_map.name)
    checks = {
        "map_name_matches": loaded_name.rstrip("/").split("/")[-1] == args.map.rstrip("/").split("/")[-1],
        "waypoints_generated": len(waypoints) >= args.min_waypoints,
        "topology_edges_generated": len(topology) >= args.min_topology_edges,
    }
    if args.require_bidirectional:
        checks["all_non_junction_roads_bidirectional"] = not runtime_topology["one_way_road_ids"]
    if args.min_junction_movements > 0:
        checks["junction_movements_generated"] = (
            runtime_topology["junction_directed_movement_count"]
            >= args.min_junction_movements
        )
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "requested_map": args.map,
        "loaded_map": loaded_name,
        "available_maps": available_maps,
        "waypoint_spacing_m": args.waypoint_spacing,
        "waypoint_count": len(waypoints),
        "topology_edge_count": len(topology),
        "runtime_topology": runtime_topology,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
