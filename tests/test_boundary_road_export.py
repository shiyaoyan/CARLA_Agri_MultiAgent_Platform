"""File-only CARLA compatibility must preserve geometry and directed travel."""

from dataclasses import replace
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from xml.etree import ElementTree as ET

from agri_carla.road_generation.carla_diagnostics import editor_route_coverage, check_direct_road_waypoints
from agri_carla.road_generation.topology import RoadTopologyValidator
from roadrunner_mcp.carla_boundary import split_open_boundaries
from roadrunner_mcp.realmap import TraceNode, TraceRoad, generate_realmap_artifacts
from roadrunner_mcp.road_mesh import road_surface_polygons
from test_road_export_regressions import make_spec, carla_0915_next_lanes


class BoundaryRoadExportTests(unittest.TestCase):
    def export(self, spec):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        result = generate_realmap_artifacts(spec, Path(temporary.name))
        return result, ET.parse(result.xodr_path).getroot()

    def isolated_spec(self):
        return replace(make_spec(), nodes=[TraceNode('a', (0, 0)), TraceNode('b', (30, 0))],
                       roads=[TraceRoad('straight', 'a', 'b', [(0, 0), (30, 0)], 4)])

    def validate_root(self, root):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'map.xodr'
            ET.ElementTree(root).write(path)
            return RoadTopologyValidator().validate_xodr(path)

    def test_t_junction_all_core_lanes_get_routes_with_only_short_terminal_omissions(self):
        result, root = self.export(make_spec())
        coverage = editor_route_coverage(ET.tostring(root, encoding='unicode'))
        self.assertEqual(result.road_count, 12)
        self.assertEqual(coverage['core_road_lane_count'], 6)
        self.assertEqual(coverage['stock_editor_predicted_core_lane_count'], 6)
        self.assertEqual(len(coverage['not_drawn_lanes']), 6)
        self.assertTrue(all(e['is_boundary_segment'] and e['road_length_m'] <= 0.5
                            for e in coverage['not_drawn_lanes']))
        self.assertTrue(RoadTopologyValidator().validate_xodr(result.xodr_path).valid)
        self.assertEqual(len(root.findall('./junction/connection')), 6)

    def test_each_new_split_is_traversable_in_both_directions(self):
        _, root = self.export(make_spec())
        count = 0
        for segment in root.findall("./road/userData[@code='carla_0915_boundary_segment']"):
            piece = next(road for road in root.findall('road') if segment in list(road))
            piece_id, core_id = int(piece.get('id')), int(segment.get('value'))
            # A terminal piece has one incoming and one outgoing driving lane.
            outward_lane = -1 if piece.find('./link/predecessor') is not None else 1
            inward_lane = -outward_lane
            self.assertEqual(carla_0915_next_lanes(root, piece_id, inward_lane), {(core_id, inward_lane)})
            self.assertIn((piece_id, outward_lane), carla_0915_next_lanes(root, core_id, outward_lane))
            self.assertEqual(carla_0915_next_lanes(root, piece_id, outward_lane), set())
            count += 1
        self.assertEqual(count, 3)

    def test_isolated_road_keeps_original_extent_area_length_and_open_ends(self):
        spec = self.isolated_spec()
        _, before = self.export(replace(spec, carla_0915_boundary_segment_m=0))
        result, after = self.export(spec)
        self.assertEqual(result.road_count, 3)
        self.assertEqual(len(after.findall('junction')), 0)
        self.assertAlmostEqual(sum(float(r.get('length')) for r in after.findall('road')), 30)
        old = road_surface_polygons(before, {'1': 'dirt'})['dirt']
        new = road_surface_polygons(after, {'1': 'dirt'})['dirt']
        self.assertLess(old.symmetric_difference(new).area, 1e-9)
        self.assertEqual(old.bounds, new.bounds)
        self.assertTrue(RoadTopologyValidator().validate_xodr(result.xodr_path).valid)

    def test_arc_split_uses_exact_positions_and_preserves_length(self):
        _, root = self.export(replace(self.isolated_spec(), carla_0915_boundary_segment_m=0))
        road = root.find('road')
        plan = road.find('planView')
        plan.clear()
        geometry = ET.SubElement(plan, 'geometry', {'s': '0', 'x': '0', 'y': '0', 'hdg': '0', 'length': '30'})
        ET.SubElement(geometry, 'arc', {'curvature': '0.02'})
        split_open_boundaries(root, 0.5)
        # Both cuts occur inside the same arc; no polyline approximation is used.
        for piece, expected_s in zip(root.findall('road'), (0.5, 0, 29.5)):
            g = piece.find('./planView/geometry')
            self.assertAlmostEqual(float(g.get('x')), 50 * math.sin(expected_s / 50), places=9)
            self.assertAlmostEqual(float(g.get('y')), 50 * (1 - math.cos(expected_s / 50)), places=9)
            self.assertAlmostEqual(float(g.get('hdg')), expected_s / 50, places=9)
        self.assertTrue(self.validate_root(root).valid)
        self.assertAlmostEqual(sum(float(r.get('length')) for r in root.findall('road')), 30)

    def test_four_way_junction_preserves_all_twelve_movements_and_disjoint_ids(self):
        result, root = self.export(make_spec(4))
        self.assertEqual(result.connection_count, 12)
        self.assertEqual(len(root.findall("./road/userData[@code='carla_0915_boundary_segment']")), 4)
        self.assertFalse({r.get('id') for r in root.findall('road')} & {j.get('id') for j in root.findall('junction')})
        report = editor_route_coverage(ET.tostring(root, encoding='unicode'))
        self.assertEqual(report['stock_editor_predicted_core_lane_count'], 8)

    def test_closed_loop_does_not_gain_boundary_roads(self):
        points = [(0, 0), (80, 0), (80, 80), (0, 80)]
        spec = replace(make_spec(), nodes=[TraceNode(str(i), p) for i, p in enumerate(points)],
                       roads=[TraceRoad(str(i), str(i), str((i+1)%4), [points[i], points[(i+1)%4]], 4) for i in range(4)])
        result, root = self.export(spec)
        self.assertFalse(root.findall("./road/userData[@code='carla_0915_boundary_segment']"))
        self.assertEqual(result.road_count, 12)
        self.assertTrue(RoadTopologyValidator().validate_xodr(result.xodr_path).valid)

    def test_boundary_road_split_is_idempotent(self):
        _, root = self.export(make_spec())
        before = ET.tostring(root)
        self.assertEqual(split_open_boundaries(root, 0.5), 0)
        self.assertEqual(ET.tostring(root), before)

    def test_missing_lane_link_and_shifted_piece_fail_validation(self):
        _, root = self.export(self.isolated_spec())
        core = root.find('road')
        link = core.find('./lanes/laneSection/right/lane/link')
        link.remove(link.find('predecessor'))
        codes = {issue.code for issue in self.validate_root(root).issues}
        self.assertIn('XODR_DIRECT_LANE_LINK_MISSING', codes)
        _, root = self.export(self.isolated_spec())
        root.findall('road')[1].find('./planView/geometry').set('y', '1')
        codes = {issue.code for issue in self.validate_root(root).issues}
        self.assertIn('XODR_DIRECT_LANE_CENTER_GAP', codes)

    def test_runtime_probe_covers_both_directions_and_detects_missing_transition(self):
        _, root = self.export(make_spec())
        class Map:
            def get_waypoint_xodr(self, road_id, lane_id, s):
                return SimpleNamespace(road_id=road_id, lane_id=lane_id,
                    next=lambda step: [self.get_waypoint_xodr(r, l, 0) for r,l in carla_0915_next_lanes(root, road_id, lane_id)],
                    transform=SimpleNamespace(location=SimpleNamespace(distance=lambda other: 0)))
        report = check_direct_road_waypoints(Map(), ET.tostring(root, encoding='unicode'))
        self.assertTrue(report['passed'])
        self.assertEqual(report['transition_count'], 6)
        class BrokenMap(Map):
            def get_waypoint_xodr(self, road_id, lane_id, s):
                wp = super().get_waypoint_xodr(road_id, lane_id, s)
                if road_id == 1:
                    wp.next = lambda step: []
                return wp
        self.assertFalse(check_direct_road_waypoints(BrokenMap(), ET.tostring(root, encoding='unicode'))['passed'])

    def test_invalid_boundary_length_rejected_and_zero_disables(self):
        for value in (-0.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                replace(make_spec(), carla_0915_boundary_segment_m=value).normalized()
        result, root = self.export(replace(make_spec(), carla_0915_boundary_segment_m=0))
        self.assertEqual(result.road_count, 9)
        self.assertFalse(root.findall("./road/userData[@code='carla_0915_boundary_segment']"))


if __name__ == '__main__':
    unittest.main()
