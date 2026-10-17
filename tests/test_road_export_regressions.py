"""Regression tests for CARLA 0.9.15 junction dispatch and FBX/XODR alignment."""

import json
import re
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from xml.etree import ElementTree as ET

from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from agri_carla.road_generation.agent import AgriculturalRoadGenerator, RoadSceneRequest
from agri_carla.road_generation.carla_diagnostics import network_signature, draw_waypoints, check_junction_waypoints
from agri_carla.road_generation.topology import RoadTopologyValidator
from roadrunner_mcp.realmap import (
    GeoAnchor, RealMapTraceSpec, TraceNode, TraceRoad, generate_realmap_artifacts,
)
from roadrunner_mcp.road_mesh import polygon_mesh, road_surface_polygons, sample_lanes


def carla_0915_next_lanes(root, road_id, lane_id):
    """Independent oracle for MapBuilder.cpp GetLaneNext's ID dispatch.

    Mirrors only the boundary case for a single lane section; it is not a CARLA
    runtime test. The key behavior is checking road IDs before junction IDs.
    """
    roads = {int(road.get("id")): road for road in root.findall("road")}
    road = roads[road_id]
    relation = "successor" if lane_id < 0 else "predecessor"
    link = road.find(f"./link/{relation}")
    if link is None:
        return set()
    next_id = int(link.get("elementId"))
    if next_id in roads:
        lane = next(l for l in road.findall("./lanes/laneSection/*/lane")
                    if int(l.get("id")) == lane_id)
        next_lane = lane.find(f"./link/{relation}")
        return {(next_id, int(next_lane.get("id")))} if next_lane is not None else set()
    junction = next(j for j in root.findall("junction") if int(j.get("id")) == next_id)
    return {(int(c.get("connectingRoad")), int(l.get("to")))
            for c in junction.findall("connection") if int(c.get("incomingRoad")) == road_id
            for l in c.findall("laneLink") if int(l.get("from")) == lane_id}


def make_spec(arm_count=3):
    center = (60.0, 60.0)
    points = [(0.0, 60.0), (60.0, 0.0), (60.0, 120.0), (120.0, 60.0)][:arm_count]
    nodes = [TraceNode("j", center)]
    roads = []
    for i, point in enumerate(points):
        name = f"n{i}"
        nodes.append(TraceNode(name, point))
        # Exercise both start/end contacts and both positive/negative lanes.
        start, end, line = (name, "j", [point, center]) if i % 2 else ("j", name, [center, point])
        roads.append(TraceRoad(f"r{i}", start, end, line,
                               width_m=4.0 if i == 0 else 8.0,
                               surface="dirt" if i == 0 else "asphalt"))
    return RealMapTraceSpec("RegressionFarm", "", 140, 140, 1.0, False,
                            GeoAnchor(37, 121, 0, 0), roads, nodes)


class ExportRegressionTests(unittest.TestCase):
    def generate(self, spec):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        result = generate_realmap_artifacts(spec, Path(temp.name))
        return result, ET.parse(result.xodr_path).getroot()

    def test_all_six_movements_survive_carla_id_dispatch(self):
        result, root = self.generate(make_spec())
        for connection in root.findall("./junction/connection"):
            link = connection.find("laneLink")
            successors = carla_0915_next_lanes(root, int(connection.get("incomingRoad")), int(link.get("from")))
            self.assertIn((int(connection.get("connectingRoad")), int(link.get("to"))), successors)
        self.assertTrue(RoadTopologyValidator().validate_xodr(result.xodr_path).valid)

    def test_old_id_collision_is_rejected_and_reproduces_missing_waypoint_links(self):
        result, root = self.generate(make_spec())
        junction = root.find("junction")
        old_id = junction.get("id")
        junction.set("id", "1")
        for road in root.findall("road"):
            if road.get("junction") == old_id:
                road.set("junction", "1")
            for link in road.findall("./link/*"):
                if link.get("elementType") == "junction":
                    link.set("elementId", "1")
        for connection in junction.findall("connection"):
            lane = int(connection.find("laneLink").get("from"))
            self.assertEqual(carla_0915_next_lanes(root, int(connection.get("incomingRoad")), lane), set())
        ET.ElementTree(root).write(result.xodr_path)
        codes = {i.code for i in RoadTopologyValidator().validate_xodr(result.xodr_path).issues}
        self.assertIn("XODR_CARLA_ROAD_JUNCTION_ID_COLLISION", codes)

    def test_four_way_junction_has_twelve_traversable_movements(self):
        _, root = self.generate(make_spec(4))
        self.assertEqual(len(root.findall("./junction/connection")), 12)
        ids = {r.get("id") for r in root.findall("road")}
        self.assertFalse(ids & {j.get("id") for j in root.findall("junction")})
        for connection in root.findall("./junction/connection"):
            link = connection.find("laneLink")
            self.assertIn((int(connection.get("connectingRoad")), int(link.get("to"))),
                          carla_0915_next_lanes(root, int(connection.get("incomingRoad")), int(link.get("from"))))

    def test_connector_contact_direction_is_enforced(self):
        result, root = self.generate(make_spec())
        connection = root.find("./junction/connection")
        connection.set("contactPoint", "end" if connection.get("contactPoint") == "start" else "start")
        ET.ElementTree(root).write(result.xodr_path)
        codes = {i.code for i in RoadTopologyValidator().validate_xodr(result.xodr_path).issues}
        self.assertIn("XODR_LANE_LINK_TRAVEL_DIRECTION", codes)

    def test_taper_matches_both_widths_with_zero_endpoint_slope(self):
        _, root = self.generate(make_spec())
        for road in root.findall("road"):
            if road.get("junction") == "-1":
                continue
            length = float(road.get("length"))
            w = road.find("./lanes/laneSection/*/lane/width")
            a, b, c, d = (float(w.get(k)) for k in "abcd")
            self.assertAlmostEqual(b, 0, places=8)
            self.assertAlmostEqual(b + 2*c*length + 3*d*length**2, 0, places=7)
            self.assertAlmostEqual(a + b*length + c*length**2 + d*length**3,
                                   4.0 if a == 2.0 else 2.0 if abs(c) > 1e-9 else 4.0, places=7)

    def test_fbx_matches_lane_footprints_without_raised_overlapping_discs(self):
        result, root = self.generate(make_spec())
        surfaces = {"1": "dirt", "2": "asphalt", "3": "asphalt"}
        parts = road_surface_polygons(root, surfaces)
        self.assertLess(parts["dirt"].intersection(parts["asphalt"]).area, 1e-9)
        footprint = unary_union(list(parts.values()))
        text = result.fbx_path.read_text()
        blocks = re.findall(r'Geometry: \d+, "Geometry::([^"]+)", "Mesh" \{(.*?)\n  Model:', text, re.S)
        triangles = []
        for name, block in blocks:
            if "Roads_Road" not in name:
                continue
            vals = [float(x) for x in re.search(r"Vertices: \*\d+\s*\{\s*a: ([^}]+)", block).group(1).strip().split(",")]
            vertices = list(zip(vals[0::3], vals[1::3], vals[2::3]))
            self.assertTrue(all(z == 0 for x, y, z in vertices))
            indices = [int(x) for x in re.search(r"PolygonVertexIndex: \*\d+\s*\{\s*a: ([^}]+)", block).group(1).strip().split(",")]
            for i in range(0, len(indices), 3):
                a, b, c = (vertices[abs(v)-1 if v < 0 else v][:2] for v in indices[i:i+3])
                signed_area = ((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))/2
                # Ignore numerical sign noise in submicron boundary slivers.
                self.assertGreaterEqual(signed_area, -1e-8)
                triangles.append(Polygon([a,b,c]))
        actual = unary_union(triangles)
        self.assertLess(footprint.symmetric_difference(actual).area, 1e-3)
        self.assertLess(abs(sum(t.area for t in triangles) - actual.area), 1e-4)
        for road in root.findall("road"):
            for samples in sample_lanes(road).values():
                self.assertTrue(all(actual.buffer(1e-5).covers(Point(p.center)) for p in samples))

    def test_polygon_triangulation_handles_concavity_and_hole(self):
        shape = Polygon([(0,0), (12,0), (12,12), (8,12), (8,6), (0,6)],
                        holes=[[(2,2), (4,2), (4,4), (2,4)]])
        mesh = polygon_mesh(shape)
        triangles = [Polygon([mesh.vertices[i][:2] for i in face]) for face in mesh.faces]
        self.assertLess(unary_union(triangles).symmetric_difference(shape).area, 1e-8)

    def test_import_manifest_preserves_generated_materials(self):
        result, _ = self.generate(make_spec())
        manifest = json.loads(next(result.import_bundle_dir.glob("*_package.json")).read_text())
        entry = manifest["maps"][0]
        self.assertFalse(entry["use_carla_materials"])
        for key in ("source", "xodr"):
            self.assertTrue((result.import_bundle_dir / entry[key]).is_file())

    def test_signature_detects_stale_topology_but_ignores_header_date(self):
        result, root = self.generate(make_spec())
        before = result.xodr_path.read_text()
        root.find("header").set("date", "different")
        self.assertEqual(network_signature(before), network_signature(ET.tostring(root, encoding="unicode")))
        root.find("junction").set("id", "1")
        self.assertNotEqual(network_signature(before), network_signature(ET.tostring(root, encoding="unicode")))

    def test_yantai_configuration_regenerates(self):
        data = json.loads((Path(__file__).resolve().parents[1] / "configs/yantai_tracking_scene.json").read_text())
        data.pop("route_road_ids", None)
        with tempfile.TemporaryDirectory() as temp:
            result = AgriculturalRoadGenerator().generate(RoadSceneRequest(**data), temp)
            self.assertTrue(result.topology_valid)
            root = ET.parse(result.xodr_path).getroot()
            self.assertEqual(root.find("junction").get("id"), "10")
            self.assertEqual(result.connection_count, 6)

    def test_waypoint_drawing_keeps_both_signs_and_all_branches(self):
        arrows = []
        class Location:
            def __init__(self, x=0, y=0, z=0):
                self.x, self.y, self.z = x, y, z
        def waypoint(lane_id, successors):
            return SimpleNamespace(
                lane_id=lane_id, lane_type=1, is_junction=False,
                transform=SimpleNamespace(location=Location(y=lane_id)),
                next=lambda spacing: successors,
            )
        outgoing = waypoint(-1, [])
        forward = waypoint(-1, [outgoing, outgoing])
        reverse = waypoint(1, [outgoing])
        world = SimpleNamespace(debug=SimpleNamespace(draw_arrow=lambda *a, **kw: arrows.append((a, kw))))
        carla = SimpleNamespace(Location=Location, Color=lambda *rgb: rgb,
                                LaneType=SimpleNamespace(Driving=1))
        self.assertEqual(draw_waypoints(world, carla, [forward, reverse], 0.5, 10), 3)
        self.assertEqual(len({entry[1]["color"] for entry in arrows}), 2)
        self.assertTrue(all(entry[0][0].z == 0.25 for entry in arrows))

    def test_runtime_report_checks_every_entry_and_exit(self):
        _, root = self.generate(make_spec())
        class FakeMap:
            def get_waypoint_xodr(self, road_id, lane_id, s):
                def next_points(step):
                    return [self.get_waypoint_xodr(r, l, 0)
                            for r, l in carla_0915_next_lanes(root, road_id, lane_id)]
                return SimpleNamespace(
                    road_id=road_id, lane_id=lane_id, next=next_points,
                    transform=SimpleNamespace(location=SimpleNamespace(distance=lambda other: 0.0)),
                )
        report = check_junction_waypoints(FakeMap(), ET.tostring(root, encoding="unicode"))
        self.assertEqual(report["transition_count"], 12)
        self.assertTrue(report["passed"])
        original = FakeMap.get_waypoint_xodr
        def missing_lane(instance, road, lane, s):
            return None if road == 1 and lane == 1 else original(instance, road, lane, s)
        FakeMap.get_waypoint_xodr = missing_lane
        # Missing source/target waypoints must produce a failed report.
        report = check_junction_waypoints(FakeMap(), ET.tostring(root, encoding="unicode"))
        self.assertFalse(report["passed"])


if __name__ == "__main__":
    unittest.main()
