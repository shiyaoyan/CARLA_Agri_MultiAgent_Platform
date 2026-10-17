"""CARLA waypoint checks using explicit XODR lane IDs and every junction branch."""

from __future__ import annotations

import hashlib
import json
from xml.etree import ElementTree as ET


def network_signature(text: str) -> str:
    """Ignore formatting/header timestamps when checking which map was loaded."""
    root = ET.fromstring(text)

    def encode(node):
        attrs = []
        for key, value in sorted(node.attrib.items()):
            try:
                value = format(float(value), ".12g")
            except ValueError:
                pass
            attrs.append((key, value))
        return node.tag, attrs, [encode(child) for child in node]

    payload = [encode(node) for node in root if node.tag in {"road", "junction"}]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _key(waypoint):
    return int(waypoint.road_id), int(waypoint.lane_id)


def editor_route_coverage(xodr_text: str) -> dict:
    """Predict the 0.9.15 OpenDriveActor empty-predecessor omission.

    This is XML analysis, not a CARLA runtime or a geometry validation. Scope
    matches our exporter: RHT roads with one section and only driving lanes.
    Lane predecessors are obtained by inverting *directed* successor links,
    including the connector exits; a road-level link alone is not sufficient.
    """
    root = ET.fromstring(xodr_text)
    roads = {int(road.get("id")): road for road in root.findall("road")}
    junctions = {int(j.get("id")): j for j in root.findall("junction")}
    report = {
        "analysis": "static_carla_0.9.15_OpenDriveActor",
        "carla_runtime_tested": False,
        "source": "https://github.com/carla-simulator/carla/blob/0.9.15/Unreal/CarlaUE4/Plugins/Carla/Source/Carla/OpenDrive/OpenDriveActor.cpp#L150-L165",
    }
    lanes = {}
    for road_id, road in roads.items():
        sections = road.findall("./lanes/laneSection")
        if (road.get("rule", "RHT") != "RHT" or len(sections) != 1
                or float(sections[0].get("s", "0")) != 0):
            return dict(report, supported=False, reason="Requires RHT roads with one section at s=0.")
        for lane in sections[0].findall("./*/lane"):
            lane_id = int(lane.get("id"))
            if lane_id == 0:
                continue
            if lane.get("type") != "driving":
                return dict(report, supported=False, reason="Only driving lanes are supported by this predictor.")
            lanes[road_id, lane_id] = lane
    predecessors = {key: set() for key in lanes}
    successors = {key: set() for key in lanes}
    for (road_id, lane_id), lane in lanes.items():
        relation = "successor" if lane_id < 0 else "predecessor"
        link = roads[road_id].find(f"./link/{relation}")
        if link is None:
            continue
        target_id = int(link.get("elementId"))
        targets = []
        # Match CARLA 0.9.15 MapBuilder: road IDs take precedence over junctions.
        if target_id in roads:
            lane_link = lane.find(f"./link/{relation}")
            if lane_link is not None:
                targets.append((target_id, int(lane_link.get("id"))))
        elif target_id in junctions:
            for connection in junctions[target_id].findall("connection"):
                if int(connection.get("incomingRoad")) == road_id:
                    targets.extend((int(connection.get("connectingRoad")), int(ll.get("to")))
                                   for ll in connection.findall("laneLink")
                                   if int(ll.get("from")) == lane_id)
        else:
            return dict(report, supported=False, reason=f"Unresolved road/junction link: {target_id}.")
        for target in targets:
            if target not in predecessors:
                return dict(report, supported=False, reason=f"Unresolved target lane: {target}.")
            predecessors[target].add((road_id, lane_id))
            successors[road_id, lane_id].add(target)
    entries = []
    for (road_id, lane_id), previous in sorted(predecessors.items()):
        road = roads[road_id]
        boundary = road.find("userData[@code='carla_0915_boundary_segment']") is not None
        length = float(road.get("length"))
        entries.append({
            "road_id": road_id, "road_name": road.get("name", ""),
            "lane_id": lane_id, "is_junction": road.get("junction", "-1") != "-1",
            "entry_contact": "start" if lane_id < 0 else "end",
            "is_boundary_segment": boundary, "road_length_m": length,
            "predecessors": [list(key) for key in sorted(previous)],
            "skipped_by_stock_editor": not previous,
            # Default OpenDriveActor RoadAccuracy is 2 m; a shorter terminal
            # segment also has fewer than two positions and gets no route.
            "stock_editor_route_predicted": bool(previous) and (
                length >= 2.0 or bool(successors[road_id, lane_id])),
        })
    normal = [entry for entry in entries if not entry["is_junction"]]
    core = [entry for entry in normal if not entry["is_boundary_segment"]]
    return dict(report, supported=True, driving_lane_count=len(entries),
                normal_lane_count=len(normal),
                stock_editor_eligible_normal_lane_count=sum(not e["skipped_by_stock_editor"] for e in normal),
                core_road_lane_count=len(core),
                stock_editor_predicted_core_lane_count=sum(e["stock_editor_route_predicted"] for e in core),
                boundary_segment_count=sum(e["is_boundary_segment"] and e["lane_id"] < 0 for e in normal),
                prediction_road_accuracy_m=2.0,
                not_drawn_lanes=[e for e in entries if not e["stock_editor_route_predicted"]],
                skipped_lanes=[e for e in entries if e["skipped_by_stock_editor"]],
                lanes=entries)


def check_lane_waypoints(carla_map, xodr_text: str) -> dict:
    """Probe each declared driving lane near both ends and at its midpoint.

    Compare against XODR declarations, so an entirely missing road/direction
    cannot disappear from the denominator of the bidirectional check.
    """
    checks = []
    for road in ET.fromstring(xodr_text).findall("road"):
        road_id = int(road.get("id"))
        sections = road.findall("./lanes/laneSection")
        for index, section in enumerate(sections):
            start = float(section.get("s"))
            end = (float(sections[index + 1].get("s")) if index + 1 < len(sections)
                   else float(road.get("length")))
            epsilon = min(1e-4, (end - start) / 100)
            for lane in section.findall("./*/lane"):
                if lane.get("type") != "driving" or int(lane.get("id")) == 0:
                    continue
                lane_id = int(lane.get("id"))
                samples = []
                for s in (start + epsilon, (start + end) / 2, end - epsilon):
                    wp = carla_map.get_waypoint_xodr(road_id, lane_id, s)
                    samples.append({"s": s, "found": wp is not None and _key(wp) == (road_id, lane_id)})
                checks.append({"road_id": road_id, "lane_id": lane_id,
                               "section_index": index, "samples": samples,
                               "passed": all(sample["found"] for sample in samples)})
    return {"passed": bool(checks) and all(c["passed"] for c in checks),
            "declared_lane_count": len(checks),
            "failed_lane_count": sum(not c["passed"] for c in checks), "lanes": checks}


def check_direct_road_waypoints(carla_map, xodr_text: str) -> dict:
    """Check next() in both directions at ordinary splits, including boundaries."""
    roads = {r.get("id"): r for r in ET.fromstring(xodr_text).findall("road")}
    checks = []
    for source_id, source in roads.items():
        if source.get("junction", "-1") != "-1":
            continue
        for lane in source.findall("./lanes/laneSection/*/lane"):
            if lane.get("type") != "driving":
                continue
            lane_id = int(lane.get("id"))
            relation = "successor" if lane_id < 0 else "predecessor"
            link = source.find(f"./link/{relation}")
            if link is None or link.get("elementType") != "road":
                continue
            target = roads.get(link.get("elementId"))
            if target is None or target.get("junction", "-1") != "-1":
                continue
            target_id = int(target.get("id"))
            lane_link = lane.find(f"./link/{relation}")
            if lane_link is None:
                checks.append({"source": [int(source_id), lane_id], "passed": False,
                               "error": "missing direct lane link"})
                continue
            target_lane = int(lane_link.get("id"))
            source_length, target_length = float(source.get("length")), float(target.get("length"))
            epsilon = min(1e-4, source_length / 100, target_length / 100)
            s = source_length - epsilon if lane_id < 0 else epsilon
            target_s = epsilon if target_lane < 0 else target_length - epsilon
            waypoint = carla_map.get_waypoint_xodr(int(source_id), lane_id, s)
            target_wp = carla_map.get_waypoint_xodr(target_id, target_lane, target_s)
            actual = list(waypoint.next(min(0.02, target_length / 4))) if waypoint is not None else []
            gap = (waypoint.transform.location.distance(target_wp.transform.location)
                   if waypoint is not None and target_wp is not None else None)
            checks.append({"source": [int(source_id), lane_id], "target": [target_id, target_lane],
                           "actual_successors": [list(_key(wp)) for wp in actual],
                           "endpoint_gap_m": gap,
                           "passed": gap is not None and gap <= 0.15 and any(
                               _key(wp) == (target_id, target_lane) for wp in actual)})
    return {"passed": all(check["passed"] for check in checks), "transition_count": len(checks),
            "failed_transition_count": sum(not check["passed"] for check in checks), "transitions": checks}


def check_junction_waypoints(carla_map, xodr_text: str) -> dict:
    """Check each entry/exit with next(), not just an undirected XML graph."""
    root = ET.fromstring(xodr_text)
    roads = {road.get("id"): road for road in root.findall("road")}
    checks = []
    for junction in root.findall("junction"):
        for connection in junction.findall("connection"):
            connector_id = connection.get("connectingRoad")
            connector = roads[connector_id]
            incoming_id = connection.get("incomingRoad")
            entry_side = connection.get("contactPoint")
            relation = "successor" if entry_side == "start" else "predecessor"
            exit_link = connector.find(f"./link/{relation}")
            for lane_link in connection.findall("laneLink"):
                connector_lane = int(lane_link.get("to"))
                lane = next(lane for lane in connector.findall("./lanes/laneSection/*/lane")
                            if int(lane.get("id")) == connector_lane)
                outgoing_lane_link = lane.find(f"./link/{relation}")
                if exit_link is None or outgoing_lane_link is None:
                    checks.append({"junction": junction.get("id"), "connector": connector_id,
                                   "passed": False, "error": "missing connector exit link"})
                    continue
                transitions = [
                    (incoming_id, int(lane_link.get("from")), connector_id, connector_lane, "entry"),
                    (connector_id, connector_lane, exit_link.get("elementId"),
                     int(outgoing_lane_link.get("id")), "exit"),
                ]
                for source, lane_id, target, target_lane, phase in transitions:
                    source_length = float(roads[source].get("length"))
                    target_length = float(roads[target].get("length"))
                    # Stay close to the boundary, without jumping over a short connector.
                    epsilon = min(1e-4, source_length / 100, target_length / 100)
                    s = source_length - epsilon if lane_id < 0 else epsilon
                    step = min(0.02, target_length / 4)
                    waypoint = carla_map.get_waypoint_xodr(int(source), lane_id, s)
                    successors = list(waypoint.next(step)) if waypoint is not None else []
                    found = next((wp for wp in successors if _key(wp) == (int(target), target_lane)), None)
                    target_s = epsilon if target_lane < 0 else target_length - epsilon
                    target_wp = carla_map.get_waypoint_xodr(int(target), target_lane, target_s)
                    gap = None
                    if waypoint is not None and target_wp is not None:
                        gap = waypoint.transform.location.distance(target_wp.transform.location)
                    checks.append({
                        "junction": junction.get("id"), "connector": connector_id, "phase": phase,
                        "source": [int(source), lane_id], "target": [int(target), target_lane],
                        "actual_successors": [list(_key(wp)) for wp in successors],
                        "endpoint_gap_m": gap,
                        "passed": found is not None and gap is not None and gap <= 0.15,
                    })
    return {
        "passed": all(check["passed"] for check in checks),
        "transition_count": len(checks),
        "failed_transition_count": sum(not check["passed"] for check in checks),
        "transitions": checks,
    }


def draw_waypoints(world, carla, waypoints, spacing: float, life_time: float,
                   height: float = 0.25) -> int:
    """Draw both lane signs and ALL next() branches with a visible height offset."""
    def lifted(waypoint):
        loc = waypoint.transform.location
        return carla.Location(x=loc.x, y=loc.y, z=loc.z + height)

    count = 0
    for waypoint in waypoints:
        if waypoint.lane_type != carla.LaneType.Driving:
            continue
        color = (carla.Color(255, 190, 40) if waypoint.is_junction else
                 carla.Color(70, 190, 255) if waypoint.lane_id < 0 else
                 carla.Color(255, 90, 140))
        for successor in waypoint.next(spacing):
            world.debug.draw_arrow(lifted(waypoint), lifted(successor), thickness=0.025,
                                   arrow_size=0.08, color=color, life_time=life_time)
            count += 1
    return count
