"""Direction-aware mission planning for the satellite Yantai case."""
import math
from shapely.geometry import LineString, Polygon

from agri_carla.simulation.full_process_geometry import (
    angle_error, bulb_path, distance, harvest_rows, heading, path_length,
    sample_line, smooth_connection,
)


def xyz(location):
    return (float(location.x), float(location.y), float(location.z))


def lane_key(wp):
    return (wp.road_id, wp.section_id, wp.lane_id)


def select_gate(waypoints, position, yaw, config):
    candidates = [w for w in waypoints if not w.is_junction
                  and w.road_id == config.road_gate_road_id
                  and abs(angle_error(w.transform.rotation.yaw, yaw)) < 30
                  and distance(xyz(w.transform.location), position) <= config.road_projection_limit_m]
    if not candidates:
        raise ValueError(f"No compatible directed driving lane near {position}, yaw={yaw}")
    return min(candidates, key=lambda w: distance(xyz(w.transform.location), position))


def checked_road_route(planner, start, end, spacing):
    route = list(planner.trace_route(start.transform.location, end.transform.location))
    if not route:
        raise ValueError("No directed road route between the selected gates")
    if lane_key(route[0][0]) != lane_key(start) or lane_key(route[-1][0]) != lane_key(end):
        raise ValueError("Road planner selected a different entry/exit lane")
    if lane_key(start) != lane_key(end):
        raise ValueError("Satellite case expects both gates on the same farm-road lane")
    # GRP samples may extend past a short destination. Keep the directed
    # subroute and exact selected gates, without reversing at the endpoint.
    u = (math.cos(math.radians(start.transform.rotation.yaw)), math.sin(math.radians(start.transform.rotation.yaw)))
    a, b = xyz(start.transform.location), xyz(end.transform.location)
    along = lambda p: (p[0]-a[0])*u[0] + (p[1]-a[1])*u[1]
    if along(b) < 3.0:
        raise ValueError("Road gates are reversed or leave less than 3 m of road travel")
    if any(lane_key(wp) != lane_key(start) for wp, _ in route):
        raise ValueError("Unexpected road detour; inspect map links and gate directions")
    middle = [(wp, option) for wp, option in route if 1e-5 < along(xyz(wp.transform.location)) < along(b)-1e-5]
    route = [(start, route[0][1])] + middle + [(end, route[-1][1])]
    points = [xyz(w.transform.location) for w, _ in route]
    if any(distance(x, y) > max(2.0, spacing*3) for x, y in zip(points, points[1:])):
        raise ValueError("Road route has a discontinuity")
    if path_length(points) > 1.3 * distance(a, b):
        raise ValueError("Unexpected farm-road detour")
    return route


class FullProcessTaskPlanner:
    def __init__(self, carla, carla_map, config, route_planner):
        self.carla, self.carla_map = carla, carla_map
        self.config, self.route_planner = config, route_planner

    def build(self):
        c = self.config
        waypoints = list(self.carla_map.generate_waypoints(c.path_spacing_m))
        gates = {key: select_gate(waypoints, getattr(c, key), yaw, c) for key, yaw in (
            ("road_entry", c.road_outbound_yaw), ("field_entry_road", c.road_outbound_yaw),
            ("field_exit_road", c.road_inbound_yaw), ("road_return", c.road_inbound_yaw))}
        if lane_key(gates["road_entry"]) == lane_key(gates["road_return"]):
            raise ValueError("Outbound and return must use opposite driving lanes")
        rows = harvest_rows(c.field_corners, c.harvest_width_m, c.harvest_passes)
        tasks = []

        def add(mode, points, end_yaw, speed, **extra):
            if len(points) < 2 or path_length(points) <= 0:
                raise ValueError(f"Empty path for {mode}")
            tasks.append(dict(type=mode, points=points, end_yaw=end_yaw,
                              speed_kmh=speed, length_m=path_length(points), **extra))

        def connection(mode, start, end, start_yaw, end_yaw):
            points = smooth_connection(start, end, start_yaw, end_yaw,
                                       c.min_turn_radius_m, c.path_spacing_m)
            add(mode, points, end_yaw, c.maneuver_speed_kmh)

        def road(mode, a, b):
            route = checked_road_route(self.route_planner, gates[a], gates[b], c.path_spacing_m)
            add(mode, [xyz(w.transform.location) for w, _ in route],
                gates[b].transform.rotation.yaw, c.road_speed_kmh, route=route)

        p = lambda key: xyz(gates[key].transform.location)
        yaw = lambda key: gates[key].transform.rotation.yaw
        connection("GARAGE_EXIT", c.garage_spawn[:3], p("road_entry"), c.garage_spawn[3], yaw("road_entry"))
        road("ROAD_NAV", "road_entry", "field_entry_road")
        connection("OFFROAD_APPROACH", p("field_entry_road"), rows[0][0],
                   yaw("field_entry_road"), heading(*rows[0]))
        for i, (start, end) in enumerate(rows):
            if i:
                points = bulb_path(*rows[i-1], start, c.min_turn_radius_m, c.path_spacing_m)
                add("TURN_BULB", points, heading(start, end), c.turn_speed_kmh)
            add("HARVEST_LINE", sample_line(start, end, c.path_spacing_m), heading(start, end), c.field_line_speed_kmh)
        connection("FIELD_EXIT", rows[-1][1], p("field_exit_road"), heading(*rows[-1]), yaw("field_exit_road"))
        road("ROAD_RETURN", "field_exit_road", "road_return")
        connection("GARAGE_RETURN", p("road_return"), c.garage_return, yaw("road_return"), c.garage_return_yaw)

        boundary = Polygon(c.field_maneuver_boundary)
        if not boundary.is_valid or boundary.area <= 0:
            raise ValueError("Invalid field maneuver boundary")
        for task in tasks:
            if task["type"] in {"HARVEST_LINE", "TURN_BULB"}:
                footprint = LineString([p[:2] for p in task["points"]]).buffer(c.vehicle_clearance_m)
                if not boundary.covers(footprint):
                    raise ValueError("Harvest/turn footprint leaves the reserved maneuver area")
        for a, b in zip(tasks, tasks[1:]):
            if distance(a["points"][-1], b["points"][0]) > 1e-6:
                raise ValueError("Disconnected mission stages")
        return tasks


def serializable_tasks(tasks):
    return [{**{k: v for k, v in task.items() if k != "route"},
             **({"lane_route": [{"road_id": wp.road_id, "section_id": wp.section_id,
                                "lane_id": wp.lane_id, "s": wp.s} for wp, _ in task["route"]]}
                if "route" in task else {})} for task in tasks]
