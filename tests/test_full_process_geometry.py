"""Focused geometry checks without CARLA or simulation package side effects."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import unittest


# The simulation package eagerly imports runners; keep pure tests independent.
MODULE_PATH = Path(__file__).resolve().parents[1] / "agri_carla/simulation/full_process_geometry.py"
SPEC = importlib.util.spec_from_file_location("full_process_geometry", MODULE_PATH)
geometry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(geometry)


def transform(point, yaw, reflection=1.0):
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    return (40.0 + c * point[0] - s * reflection * point[1],
            -25.0 + s * point[0] + c * reflection * point[1], point[2])


class GeometryAssertions(unittest.TestCase):
    def assertPointClose(self, actual, expected, places=9):
        for a, b in zip(actual, expected):
            self.assertAlmostEqual(a, b, places=places)

    def assertSampleGaps(self, points, spacing):
        for a, b in zip(points, points[1:]):
            gap = geometry.distance(a, b)
            self.assertGreater(gap, 0.0)
            self.assertLessEqual(gap, spacing + 1e-9)

    def assertEndpointYaws(self, points, start_yaw, end_yaw, tolerance):
        self.assertLessEqual(abs(geometry.angle_error(
            geometry.heading(points[0], points[1]), start_yaw)), tolerance)
        self.assertLessEqual(abs(geometry.angle_error(
            geometry.heading(points[-2], points[-1]), end_yaw)), tolerance)


class PrimitiveTests(GeometryAssertions):
    def test_module_runs_with_only_standard_library(self):
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-B", "-c",
             "import runpy, sys; g = runpy.run_path(sys.argv[1]); "
             "assert g['distance']((0,0,0), (3,4,9)) == 5; "
             "assert 'carla' not in sys.modules", str(MODULE_PATH)],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_xy_distance_and_path_length_ignore_finite_z(self):
        self.assertEqual(geometry.distance((0, 0, -50), (3, 4, 100)), 5.0)
        self.assertEqual(geometry.path_length(iter([(0, 0, 9), (3, 4, -3), (6, 0, 0)])), 10.0)
        self.assertEqual(geometry.path_length([]), 0.0)
        self.assertEqual(geometry.path_length([(1, 2, 3)]), 0.0)

    def test_carla_headings_and_signed_wrapping(self):
        for point, expected in [((1, 0, 0), 0), ((0, 1, 0), 90),
                                ((-1, 0, 0), 180), ((0, -1, 0), -90)]:
            self.assertEqual(geometry.heading((0, 0, 0), point), expected)
        for a, b, expected in [(179, -179, -2), (-179, 179, 2), (180, 0, -180),
                                (0, 180, -180), (725, 0, 5), (-725, 0, -5)]:
            self.assertEqual(geometry.angle_error(a, b), expected)
        self.assertTrue(math.isfinite(geometry.angle_error(1e308, -1e308)))
        with self.assertRaisesRegex(ValueError, "distinct XY"):
            geometry.heading((1, 2, 3), (1, 2, 4))

    def test_line_exact_endpoints_spacing_and_elevation(self):
        start, end = (2.0, -1.0, 0.5), (3.0, 0.0, 2.0)
        for spacing in (0.25, 0.07, 10.0):
            with self.subTest(spacing=spacing):
                points = geometry.sample_line(start, end, spacing)
                self.assertEqual(points[0], start)
                self.assertEqual(points[-1], end)
                self.assertSampleGaps(points, spacing)
                self.assertAlmostEqual(geometry.path_length(points), math.sqrt(2))
                for i, point in enumerate(points):
                    self.assertAlmostEqual(point[2], start[2] + (end[2] - start[2]) * i / (len(points) - 1))
        self.assertEqual(geometry.sample_line(start, start), [start])
        self.assertEqual(geometry.sample_line(start, (2, -1, 5)), [start, (2, -1, 5)])


class HarvestRowsTests(GeometryAssertions):
    corners = [(50, -80, 0.5), (150, -80, 0.5), (150, -74, 0.5), (50, -74, 0.5)]

    def test_two_lengthwise_passes_not_34_crosswise_passes(self):
        rows = geometry.harvest_rows(self.corners, 3.0)
        self.assertEqual(len(rows), 2)
        self.assertEqual(math.ceil(geometry.distance(self.corners[0], self.corners[1]) / 3), 34)
        self.assertEqual(rows, [((50, -78.5, 0.5), (150, -78.5, 0.5)),
                                ((150, -75.5, 0.5), (50, -75.5, 0.5))])
        self.assertEqual(sum(geometry.distance(*row) for row in rows), 200)
        self.assertEqual(geometry.angle_error(geometry.heading(*rows[0]), geometry.heading(*rows[1])), -180)

    def test_rotated_and_reflected_swath_centers(self):
        for yaw in (35, 90, 170):
            for reflection in (-1, 1):
                with self.subTest(yaw=yaw, reflection=reflection):
                    corners = [transform(p, yaw, reflection) for p in self.corners]
                    rows = geometry.harvest_rows(corners, 3)
                    for actual, row in zip(rows, geometry.harvest_rows(self.corners, 3)):
                        for point, expected in zip(actual, row):
                            self.assertPointClose(point, transform(expected, yaw, reflection))

    def test_parallelogram_uses_perpendicular_width_and_interpolates_z(self):
        corners = [(10, 5, 1), (10, 105, 3), (16, 113, 4), (16, 13, 2)]
        rows = geometry.harvest_rows(corners, 3)
        self.assertEqual(rows, [((11.5, 7, 1.25), (11.5, 107, 3.25)),
                                ((14.5, 111, 3.75), (14.5, 11, 1.75))])

    def test_expected_passes_checks_coverage_instead_of_truncating(self):
        corners = [(0, 0, 0), (100, 0, 0), (100, 9, 0), (0, 9, 0)]
        with self.assertRaisesRegex(ValueError, "3 passes.*expected_passes=2"):
            geometry.harvest_rows(corners, 3)
        self.assertEqual(len(geometry.harvest_rows(corners, 3, expected_passes=3)), 3)
        wrong_order = [self.corners[i] for i in (0, 3, 2, 1)]
        with self.assertRaisesRegex(ValueError, "34 passes"):
            geometry.harvest_rows(wrong_order, 3)
        for invalid in (0, -1, 2.0, 2.5, True, None, math.inf, math.nan):
            with self.subTest(expected_passes=invalid), self.assertRaises(ValueError):
                geometry.harvest_rows(self.corners, 3, invalid)

    def test_invalid_fields_and_outside_centers_are_rejected(self):
        invalid_fields = [self.corners[:3],
                          [(0, 0, 0), (0, 0, 0), (0, 6, 0), (0, 6, 0)],
                          [(0, 0, 0), (100, 0, 0), (100, 0, 0), (0, 0, 0)],
                          [(0, 0, 0), (100, 0, 0), (106, 0, 0), (6, 0, 0)],
                          [(0, 0, 0), (100, 0, 0), (100, 5, 0), (0, 6, 0)],
                          [self.corners[i] for i in (0, 2, 1, 3)]]
        for corners in invalid_fields:
            with self.subTest(corners=corners), self.assertRaises(ValueError):
                geometry.harvest_rows(corners, 3)
        with self.assertRaisesRegex(ValueError, "swath center"):
            geometry.harvest_rows([(0, 0, 0), (100, 0, 0), (100, 4, 0), (0, 4, 0)], 3)


class BulbPathTests(GeometryAssertions):
    def test_exact_three_arcs_forward_tangency_radius_and_length(self):
        radius, d, spacing = 5.5, 3.0, 0.25
        start, end, next_start = (-100, 0, 1), (0, 0, 2), (0, d, 5)
        points = geometry.bulb_path(start, end, next_start, radius, spacing)
        self.assertEqual(points[0], end)
        self.assertEqual(points[-1], next_start)
        self.assertSampleGaps(points, spacing)
        self.assertEndpointYaws(points, 0, 180, math.degrees(spacing / (2 * radius)) + 1e-9)
        a = math.acos((2 * radius + d) / (4 * radius))
        join1 = (radius * math.sin(a), radius * (math.cos(a) - 1), 0)
        join2 = (radius * math.sin(a), radius * (3 * math.cos(a) - 1), 0)
        indices = [0]
        for join in (join1, join2):
            index = min(range(len(points)), key=lambda i: geometry.distance(points[i], join))
            self.assertLess(geometry.distance(points[index], join), 1e-10)
            indices.append(index)
        indices.append(len(points) - 1)
        centers = [(0, -radius, 0),
                   (2 * radius * math.sin(a), 2 * radius * math.cos(a) - radius, 0),
                   (0, d + radius, 0)]
        for first, last, center, sign in zip(indices, indices[1:], centers, (-1, 1, -1)):
            arc = points[first:last + 1]
            for point in arc:
                self.assertAlmostEqual(geometry.distance(point, center), radius, places=10)
            for p, q in zip(arc, arc[1:]):
                radial = ((p[0] + q[0]) / 2 - center[0], (p[1] + q[1]) / 2 - center[1])
                tangent = (-sign * radial[1], sign * radial[0])
                chord = (q[0] - p[0], q[1] - p[1])
                cosine = sum(t * c for t, c in zip(tangent, chord)) / (math.hypot(*tangent) * math.hypot(*chord))
                self.assertAlmostEqual(cosine, 1.0, places=10)
        for index in indices[1:-1]:
            before = geometry.heading(points[index - 1], points[index])
            after = geometry.heading(points[index], points[index + 1])
            self.assertLess(abs(geometry.angle_error(before, after)), math.degrees(spacing / radius))
        exact_length = radius * (math.pi + 4 * a)
        self.assertLess(abs(geometry.path_length(points) - exact_length), exact_length * spacing ** 2 / (24 * radius ** 2))
        self.assertLess(min(p[1] for p in points), 0)
        self.assertGreater(max(p[1] for p in points), d)
        self.assertTrue(all(p[2] <= q[2] for p, q in zip(points, points[1:])))

    def test_rotations_reflections_and_varying_radius(self):
        for yaw, reflection, radius, d in [(37, 1, 5.5, 3), (-120, -1, 5.5, 3),
                                           (90, -1, 2, 1), (175, 1, 8, 15.9)]:
            with self.subTest(yaw=yaw, reflection=reflection, radius=radius):
                original = [(-80, 0, 1), (0, 0, 2), (0, d, 3)]
                start, end, next_start = [transform(p, yaw, reflection) for p in original]
                points = geometry.bulb_path(start, end, next_start, radius, spacing=0.1)
                reference = geometry.bulb_path(*original, radius, spacing=0.1)
                self.assertEqual(points[0], end)
                self.assertEqual(points[-1], next_start)
                self.assertEqual(len(points), len(reference))
                for point, expected in zip(points, reference):
                    self.assertPointClose(point, transform(expected, yaw, reflection))
                self.assertSampleGaps(points, 0.1)
                self.assertEndpointYaws(points, yaw, yaw + 180, math.degrees(0.1 / (2 * radius)) + 1e-8)

    def test_narrow_row_and_alignment_constraints(self):
        for d in (11, 12, -11):
            with self.subTest(d=d), self.assertRaisesRegex(ValueError, "narrow-row omega only"):
                geometry.bulb_path((-100, 0, 0), (0, 0, 0), (0, d, 0))
        for offset in (0.01, -0.01, 2):
            with self.subTest(offset=offset), self.assertRaisesRegex(ValueError, "aligned"):
                geometry.bulb_path((-100, 0, 0), (0, 0, 0), (offset, 3, 0))
        with self.assertRaisesRegex(ValueError, "previous row length"):
            geometry.bulb_path((0, 0, 0), (0, 0, 1), (0, 3, 0))
        with self.assertRaisesRegex(ValueError, "row spacing"):
            geometry.bulb_path((-100, 0, 0), (0, 0, 0), (0, 0, 0))

    def test_coarse_spacing_still_represents_an_omega(self):
        points = geometry.bulb_path((-10, 0, 0), (0, 0, 0), (0, 3, 0), spacing=100)
        self.assertGreater(len(points), 30)
        self.assertEndpointYaws(points, 0, 180, 3)
        self.assertGreater(geometry.path_length(points), 30)

    def test_spacing_one_float_below_two_radii_has_no_duplicate_endpoints(self):
        next_start = (0, 10.999999999999998, 0)
        points = geometry.bulb_path((-100, 0, 0), (0, 0, 0), next_start)
        self.assertEqual(points[0], (0, 0, 0))
        self.assertEqual(points[-1], next_start)
        self.assertSampleGaps(points, 0.25)
        self.assertEndpointYaws(points, 0, 180, 1e-5)


class SmoothConnectionTests(GeometryAssertions):
    def test_exact_endpoints_yaws_and_independently_recovered_curvature(self):
        start, end, radius = (4.0, -7.0, 1.0), (19.0, 8.0, 4.0), 5.5
        points = geometry.smooth_connection(start, end, 0, 90, radius)
        self.assertEqual(points[0], start)
        self.assertEqual(points[-1], end)
        self.assertSampleGaps(points, 0.05)
        self.assertEndpointYaws(points, 0, 90, 0.3)
        n = len(points) - 1
        delta = (end[0] - start[0], end[1] - start[1])
        # Recover both handles independently from two interior cubic samples.
        i, j = n // 3, 2 * n // 3
        t, s = i / n, j / n
        a, b = 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t ** 2
        c, d = 3 * (1 - s) ** 2 * s, 3 * (1 - s) * s ** 2
        p1, p2 = [], []
        for k in (0, 1):
            x = points[i][k] - start[k] - t ** 3 * delta[k]
            y = points[j][k] - start[k] - s ** 3 * delta[k]
            p1.append((d * x - b * y) / (a * d - b * c))
            p2.append((a * y - c * x) / (a * d - b * c))
        controls = [(0, 0), p1, p2, delta]
        edges = [(q[0] - p[0], q[1] - p[1]) for p, q in zip(controls, controls[1:])]
        for edge, yaw in ((edges[0], 0), (edges[-1], 90)):
            self.assertAlmostEqual(math.degrees(math.atan2(edge[1], edge[0])), yaw, places=9)
            fraction = math.hypot(*edge) / geometry.distance(start, end)
            self.assertLess(min(abs(fraction - f) for f in (.25, .4, .55, .7, .9, 1.1)), 1e-9)
        self.assertLessEqual(3 * max(math.hypot(*edge) for edge in edges) / n, 0.05 + 1e-12)
        for i, point in enumerate(points):
            t, q = i / n, 1 - i / n
            first = [3 * (q * q * edges[0][k] + 2 * q * t * edges[1][k] + t * t * edges[2][k]) for k in (0, 1)]
            second = [6 * (q * (edges[1][k] - edges[0][k]) + t * (edges[2][k] - edges[1][k])) for k in (0, 1)]
            curvature = abs(first[0] * second[1] - first[1] * second[0]) / math.hypot(*first) ** 3
            self.assertLessEqual(curvature, 1 / radius + 1e-10)
            self.assertAlmostEqual(point[2], q * start[2] + t * end[2], places=12)

    def test_straight_connection_and_requested_finer_spacing(self):
        start, end = (0, 0, 0), (2, 0, 1)
        points = geometry.smooth_connection(start, end, 360, -360, spacing=0.013)
        self.assertEqual(points[0], start)
        self.assertEqual(points[-1], end)
        self.assertAlmostEqual(geometry.path_length(points), 2.0)
        self.assertTrue(all(p[1] == 0 for p in points))
        self.assertSampleGaps(points, 0.013)
        self.assertEndpointYaws(points, 0, 0, 1e-10)

    def test_curves_with_both_turn_signs_and_translated_yaws(self):
        for yaw, reflection in ((125, 1), (-40, -1)):
            with self.subTest(yaw=yaw, reflection=reflection):
                start, end = transform((0, 0, 2), yaw, reflection), transform((15, 15, 4), yaw, reflection)
                points = geometry.smooth_connection(start, end, yaw, yaw + reflection * 90, spacing=0.03)
                self.assertEqual(points[0], start)
                self.assertEqual(points[-1], end)
                self.assertSampleGaps(points, 0.03)
                self.assertEndpointYaws(points, yaw, yaw + reflection * 90, 0.3)

    def test_sharp_connections_and_straight_cusps_are_rejected_without_snapping(self):
        for end, yaw0, yaw1 in [((1, 1, 0), 0, 90), ((0, 3, 0), 0, 180),
                                ((10, 0, 0), 180, 0), ((10, 0, 0), 0, 180),
                                ((10, 0, 0), 180, 180)]:
            with self.subTest(end=end, yaws=(yaw0, yaw1)), self.assertRaisesRegex(ValueError, "No smooth connection"):
                geometry.smooth_connection((0, 0, 0), end, yaw0, yaw1)
        with self.assertRaisesRegex(ValueError, "endpoint XY distance"):
            geometry.smooth_connection((0, 0, 0), (0, 0, 1), 0, 180)


class InputValidationTests(unittest.TestCase):
    def test_all_point_arguments_reject_nonfinite_xyz(self):
        origin, end, next_start = (0, 0, 0), (100, 0, 0), (100, 3, 0)
        for bad in (math.nan, math.inf, -math.inf):
            for axis in range(3):
                point = list(origin)
                point[axis] = bad
                calls = [lambda: geometry.distance(point, end), lambda: geometry.distance(end, point),
                         lambda: geometry.heading(point, end), lambda: geometry.heading(end, point),
                         lambda: geometry.path_length([point]),
                         lambda: geometry.path_length([origin, point, end]),
                         lambda: geometry.sample_line(point, end), lambda: geometry.sample_line(end, point)]
                for index in range(3):
                    args = [origin, end, next_start]
                    args[index] = point
                    calls.append(lambda args=args: geometry.bulb_path(*args))
                for index in range(2):
                    args = [origin, end]
                    args[index] = point
                    calls.append(lambda args=args: geometry.smooth_connection(*args, 0, 0))
                for index in range(4):
                    corners = list(HarvestRowsTests.corners)
                    corners[index] = point
                    calls.append(lambda corners=corners: geometry.harvest_rows(corners, 3))
                for index, call in enumerate(calls):
                    with self.subTest(bad=bad, axis=axis, call=index), self.assertRaisesRegex(ValueError, "finite"):
                        call()

    def test_nonfinite_angles_and_nonpositive_or_nonfinite_parameters(self):
        for bad in (math.nan, math.inf, -math.inf):
            for call in (lambda: geometry.angle_error(bad, 0), lambda: geometry.angle_error(0, bad),
                         lambda: geometry.smooth_connection((0, 0, 0), (10, 10, 0), bad, 0),
                         lambda: geometry.smooth_connection((0, 0, 0), (10, 10, 0), 0, bad)):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    call()
        for bad in (0, -1, math.nan, math.inf, -math.inf):
            calls = [lambda: geometry.harvest_rows(HarvestRowsTests.corners, bad),
                     lambda: geometry.sample_line((0, 0, 0), (0, 0, 0), bad),
                     lambda: geometry.bulb_path((-10, 0, 0), (0, 0, 0), (0, 3, 0), radius=bad),
                     lambda: geometry.bulb_path((-10, 0, 0), (0, 0, 0), (0, 3, 0), spacing=bad),
                     lambda: geometry.smooth_connection((0, 0, 0), (10, 0, 0), 0, 0, min_radius=bad),
                     lambda: geometry.smooth_connection((0, 0, 0), (10, 0, 0), 0, 0, spacing=bad)]
            for index, call in enumerate(calls):
                with self.subTest(bad=bad, call=index), self.assertRaises(ValueError):
                    call()

    def test_xyz_triples_are_required(self):
        for point in (None, 5, (1, 2), (1, 2, 3, 4), (1, 2, "bad")):
            with self.subTest(point=point), self.assertRaises(ValueError):
                geometry.distance(point, (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
