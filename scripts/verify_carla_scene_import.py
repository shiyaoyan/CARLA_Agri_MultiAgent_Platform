"""连接 CARLA 并验证农业场景是否已成功导入和生成可导航路网。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agri_carla.utils.carla_import import import_carla
from agri_carla.road_generation.carla_diagnostics import (
    check_junction_waypoints, check_direct_road_waypoints, check_lane_waypoints, draw_waypoints,
    editor_route_coverage, network_signature,
)


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


def _runtime_topology_summary(waypoints: list, spacing: float,
                              boundary_road_ids=frozenset()) -> dict[str, object]:
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
        if start_key in junction_lane_keys or start_key[0] in boundary_road_ids:
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
        "boundary_segment_road_ids": sorted(boundary_road_ids),
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
    parser.add_argument("--map", help="Map to load; default: YantaiAnnotatedFarmScene.")
    parser.add_argument("--current-world", action="store_true",
                        help="Inspect the running world without reloading it (useful in UE Play mode).")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--waypoint-spacing", type=float, default=2.0)
    parser.add_argument("--min-waypoints", type=int, default=6)
    parser.add_argument("--min-topology-edges", type=int, default=3)
    parser.add_argument("--expected-xodr", help="Fail if the loaded road/lane network differs from this file.")
    parser.add_argument("--offline-xodr", help="Parse with carla.Map without connecting to a simulator.")
    parser.add_argument("--inspect-xodr", help="Static topology/editor-coverage analysis; no CARLA installation needed.")
    parser.add_argument("--draw", action="store_true", help="Draw both lane signs and all junction branches.")
    parser.add_argument("--draw-seconds", type=float, default=120.0)
    parser.add_argument("--draw-height", type=float, default=0.25, help="Debug arrow offset above lane surface, in metres.")
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
    if args.waypoint_spacing <= 0:
        parser.error("--waypoint-spacing must be positive")
    if args.offline_xodr and (args.draw or args.list_maps):
        parser.error("--offline-xodr cannot be combined with --draw or --list-maps")
    if args.current_world and (args.map or args.offline_xodr or args.inspect_xodr or args.list_maps):
        parser.error("--current-world cannot be combined with --map, --offline-xodr, --inspect-xodr or --list-maps")
    if args.inspect_xodr:
        if args.offline_xodr or args.draw or args.list_maps:
            parser.error("--inspect-xodr cannot be combined with --offline-xodr, --draw or --list-maps")
        from agri_carla.road_generation.topology import RoadTopologyValidator
        validation = RoadTopologyValidator().validate_xodr(args.inspect_xodr)
        coverage = editor_route_coverage(Path(args.inspect_xodr).read_text(encoding="utf-8"))
        report = {
            "mode": "static_xml_only", "carla_runtime_tested": False,
            "topology_valid": validation.valid,
            "topology_issue_codes": [issue.code for issue in validation.issues],
            "editor_route_coverage": coverage,
            "note": "Editor route omissions are separate from lane existence and runtime navigability.",
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not validation.valid:
            raise SystemExit(1)
        return

    carla = import_carla()
    requested_map = args.map or "YantaiAnnotatedFarmScene"
    available_maps = []
    world = None
    if args.offline_xodr:
        carla_map = carla.Map(requested_map, Path(args.offline_xodr).read_text(encoding="utf-8"))
    else:
        client = carla.Client(args.host, args.port)
        client.set_timeout(args.timeout)
        available_maps = list(client.get_available_maps())
        if args.list_maps:
            print(json.dumps({"available_maps": available_maps}, indent=2))
            return
        world = client.get_world() if args.current_world else client.load_world(requested_map)
        carla_map = world.get_map()
    topology = list(carla_map.get_topology())
    waypoints = list(carla_map.generate_waypoints(args.waypoint_spacing))
    # Include short lanes which may be missed by regular spacing.
    seen = {(int(wp.road_id), int(wp.section_id), int(wp.lane_id), round(wp.s, 5)) for wp in waypoints}
    for pair in topology:
        for wp in pair:
            key = (int(wp.road_id), int(wp.section_id), int(wp.lane_id), round(wp.s, 5))
            if key not in seen:
                waypoints.append(wp)
                seen.add(key)
    loaded_xodr = carla_map.to_opendrive()
    editor_coverage = editor_route_coverage(loaded_xodr)
    boundary_road_ids = {entry["road_id"] for entry in editor_coverage.get("lanes", [])
                         if entry["is_boundary_segment"]}
    runtime_topology = _runtime_topology_summary(waypoints, args.waypoint_spacing, boundary_road_ids)
    loaded_name = str(carla_map.name)
    checks = {
        "waypoints_generated": len(waypoints) >= args.min_waypoints,
        "topology_edges_generated": len(topology) >= args.min_topology_edges,
    }
    if not args.current_world:
        checks["map_name_matches"] = loaded_name.rstrip("/").split("/")[-1] == requested_map.rstrip("/").split("/")[-1]
    expected_xodr = (Path(args.expected_xodr).read_text(encoding="utf-8")
                     if args.expected_xodr else loaded_xodr)
    lane_checks = check_lane_waypoints(carla_map, expected_xodr)
    checks["all_declared_driving_lanes_present"] = lane_checks["passed"]
    junction_checks = check_junction_waypoints(carla_map, loaded_xodr)
    direct_checks = check_direct_road_waypoints(carla_map, loaded_xodr)
    checks["all_direct_road_transitions_traversable"] = direct_checks["passed"]
    checks["all_junction_transitions_traversable"] = junction_checks["passed"]
    if args.expected_xodr:
        checks["loaded_network_matches_expected_xodr"] = (
            network_signature(loaded_xodr) == network_signature(expected_xodr)
        )
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
        "requested_map": None if args.current_world else requested_map,
        "loaded_map": loaded_name,
        "available_maps": available_maps,
        "waypoint_spacing_m": args.waypoint_spacing,
        "waypoint_count": len(waypoints),
        "topology_edge_count": len(topology),
        "runtime_topology": runtime_topology,
        "lane_waypoint_checks": lane_checks,
        "editor_route_coverage": editor_coverage,
        "direct_road_waypoint_checks": direct_checks,
        "junction_waypoint_checks": junction_checks,
        "mode": "carla_map_offline" if args.offline_xodr else "current_world" if args.current_world else "imported_world",
        "client_version": getattr(carla, "__version__", "unavailable"),
    }
    if world is not None:
        report["client_version"] = client.get_client_version()
        report["server_version"] = client.get_server_version()
    if args.draw:
        report["drawn_arrows"] = draw_waypoints(
            world, carla, waypoints, args.waypoint_spacing, args.draw_seconds, args.draw_height,
        )
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
