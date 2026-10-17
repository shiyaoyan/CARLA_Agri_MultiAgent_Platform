"""Pure XY planning geometry on finite ``(x, y, z)`` tuples.

Yaw is in CARLA degrees: zero along +X, +90 along +Y. Distances, spacing,
swath widths and curvature are planar; Z is interpolated, not terrain-checked.
No CARLA objects, map projection, collision checks or vehicle dynamics are used.
"""

from __future__ import annotations

import math
from numbers import Integral


__all__ = [
    "distance", "heading", "angle_error", "path_length", "sample_line",
    "harvest_rows", "bulb_path", "smooth_connection",
]


def _finite(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _positive(value, name):
    value = _finite(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _point(value, name):
    try:
        coordinates = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an xyz triple") from exc
    if len(coordinates) != 3:
        raise ValueError(f"{name} must be an xyz triple")
    return tuple(_finite(coordinate, name) for coordinate in coordinates)


def _distance(a, b):
    return _finite(math.hypot(b[0] - a[0], b[1] - a[1]), "XY distance")


def _lerp(a, b, t):
    return tuple((1.0 - t) * x + t * y for x, y in zip(a, b))


def distance(a, b):
    """Return XY distance, ignoring finite Z coordinates."""
    return _distance(_point(a, "a"), _point(b, "b"))


def heading(a, b):
    """Return the CARLA yaw of a nonzero XY segment in degrees."""
    a, b = _point(a, "a"), _point(b, "b")
    if _distance(a, b) == 0:
        raise ValueError("heading needs distinct XY endpoints")
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def angle_error(a, b):
    """Return signed ``a - b`` wrapped into [-180, 180) degrees."""
    a, b = _finite(a, "a"), _finite(b, "b")
    return ((a % 360.0) - (b % 360.0) + 180.0) % 360.0 - 180.0


def path_length(points):
    """Return the XY polyline length; empty and singleton paths have length 0."""
    points = [_point(point, "path point") for point in points]
    return _finite(math.fsum(_distance(a, b) for a, b in zip(points, points[1:])),
                   "path length")


def sample_line(start, end, spacing=0.25):
    """Include exact endpoints with XY gaps <= spacing and linearly varying Z."""
    start, end = _point(start, "start"), _point(end, "end")
    spacing = _positive(spacing, "spacing")
    if start == end:
        return [start]
    count = max(1, math.ceil(_distance(start, end) / spacing))
    return [start] + [_lerp(start, end, i / count) for i in range(1, count)] + [end]


def harvest_rows(corners, width, expected_passes=2):
    """Return alternating swath-center rows in a rectangle/parallelogram.

    Corners are [near_left, far_left, far_right, near_right]. ``width`` is
    perpendicular XY swath spacing, including for a sheared parallelogram.
    The covering pass count is ceil(field_width / width) and must equal the
    positive integer ``expected_passes``. Centers are at (i + .5) * width;
    a last center outside the field is rejected, never clamped to an edge.
    """
    corners = [_point(point, "corner") for point in corners]
    if len(corners) != 4:
        raise ValueError("corners must contain near_left, far_left, far_right, near_right")
    width = _positive(width, "width")
    if (isinstance(expected_passes, bool) or not isinstance(expected_passes, Integral)
            or expected_passes <= 0):
        raise ValueError("expected_passes must be a positive integer")
    near_left, far_left, far_right, near_right = corners
    length = _positive(_distance(near_left, far_left), "field length")
    side_length = _positive(_distance(near_left, near_right), "field width")
    along = tuple(far_left[k] - near_left[k] for k in range(3))
    side = tuple(near_right[k] - near_left[k] for k in range(3))
    # Opposite edges must agree, including the interpolated elevation plane.
    for k in range(3):
        opposite = far_right[k] - near_right[k]
        if not math.isclose(opposite, along[k], rel_tol=1e-9,
                            abs_tol=1e-9 * max(1.0, length, side_length)):
            raise ValueError("corners must form an ordered rectangle or parallelogram")
    ux, uy = along[0] / length, along[1] / length
    field_width = _positive(abs(ux * side[1] - uy * side[0]), "perpendicular field width")
    ratio = _finite(field_width / width, "pass count")
    nearest = round(ratio)
    passes = (nearest if nearest > 0 and math.isclose(ratio, nearest, abs_tol=1e-9, rel_tol=0.0)
              else math.ceil(ratio))
    if passes != expected_passes:
        raise ValueError(f"field width requires {passes} passes, expected_passes={expected_passes}; "
                         "check corner order and swath width")
    if (passes - 0.5) * width >= field_width:
        raise ValueError("last swath center lies on or outside the field; adjust width or corners")
    rows = []
    for i in range(passes):
        fraction = (i + 0.5) * width / field_width
        near = _lerp(near_left, near_right, fraction)
        far = _lerp(far_left, far_right, fraction)
        rows.append((near, far) if i % 2 == 0 else (far, near))
    return rows


def bulb_path(start, end, next_start, radius=5.5, spacing=0.25):
    """Sample a forward-only narrow-row omega from ``end`` to ``next_start``.

    ``start`` -> ``end`` defines the incoming unit tangent u. The next row
    must start abreast of end (XY longitudinal error <= 1e-9 * max(1, d))
    with perpendicular spacing 0 < d < 2R; its outgoing tangent is -u.
    In the orthonormal (u, v) frame, v points towards the next row and the
    RLR sweeps are -a, pi+2a, -a, with a = acos((2R+d)/(4R)). Reflection
    of this frame changes world turn signs without changing the radii.
    The continuous arcs have exact endpoint tangents; sampled chords
    approximate them. Z varies linearly with accumulated arc length.
    """
    start, end = _point(start, "start"), _point(end, "end")
    next_start = _point(next_start, "next_start")
    radius, spacing = _positive(radius, "radius"), _positive(spacing, "spacing")
    length = _positive(_distance(start, end), "previous row length")
    ux, uy = (end[0] - start[0]) / length, (end[1] - start[1]) / length
    dx, dy = next_start[0] - end[0], next_start[1] - end[1]
    lateral = ux * dy - uy * dx
    d = _positive(abs(lateral), "adjacent row spacing")
    if abs(ux * dx + uy * dy) > 1e-9 * max(1.0, d):
        raise ValueError("narrow-row omega requires aligned row ends perpendicular to the previous row")
    if d / radius >= 2.0:
        raise ValueError("narrow-row omega only: adjacent row spacing d must be less than 2 * radius")
    sign = math.copysign(1.0, lateral)
    vx, vy = -sign * uy, sign * ux
    a = math.acos(0.5 + d / radius / 4.0)
    if a == 0.0:
        # The acos argument can round to 1 for d just below 2R.
        a = 2.0 * math.asin(math.sqrt((2.0 - d / radius) / 8.0))
    sweeps = (-a, math.pi + 2.0 * a, -a)
    total = radius * (math.pi + 4.0 * a)
    points = [end]
    x, y, yaw, travelled = 0.0, 0.0, 0.0, 0.0
    for sweep in sweeps:
        arc_length = radius * abs(sweep)
        # Also retain the arc shape if the caller requests very coarse spacing.
        count = max(1, math.ceil(arc_length / spacing), math.ceil(abs(sweep) / 0.1))
        signed_radius = math.copysign(radius, sweep)
        for i in range(1, count + 1):
            theta = yaw + sweep * i / count
            px = x + signed_radius * (math.sin(theta) - math.sin(yaw))
            py = y + signed_radius * (math.cos(yaw) - math.cos(theta))
            t = (travelled + arc_length * i / count) / total
            points.append((end[0] + px * ux + py * vx,
                           end[1] + px * uy + py * vy,
                           (1.0 - t) * end[2] + t * next_start[2]))
        x, y, yaw = px, py, yaw + sweep
        travelled += arc_length
    # Only roundoff-sized closure errors are allowed; never bridge bad alignment.
    if _distance(points[-1], next_start) > 1e-8 * max(1.0, radius):
        raise ValueError("narrow-row omega did not close at the supplied endpoint")
    points[-1] = next_start
    return points


def _regular_tangent(edges):
    """Reject a zero of the quadratic Bezier tangent, including straight cusps."""
    coefficients = [
        (edges[0][k] - 2.0 * edges[1][k] + edges[2][k],
         2.0 * (edges[1][k] - edges[0][k]), edges[0][k])
        for k in (0, 1)
    ]
    scale = max(abs(value) for axis in coefficients for value in axis)
    coefficients = [tuple(value / scale for value in axis) for axis in coefficients]
    a, b, c = max(coefficients, key=lambda axis: max(map(abs, axis)))
    if abs(a) < 1e-14:
        roots = [-c / b] if abs(b) >= 1e-14 else []
    else:
        discriminant = b * b - 4.0 * a * c
        if discriminant < -1e-14:
            roots = []
        else:
            q = -0.5 * (b + math.copysign(math.sqrt(max(0.0, discriminant)), b))
            roots = [q / a, c / q] if q else [-b / (2.0 * a)]
    for t in roots:
        if 0.0 <= t <= 1.0:
            tangent = [(a * t + b) * t + c for a, b, c in coefficients]
            if math.hypot(*tangent) <= 1e-10:
                return False
    return True


def smooth_connection(start, end, start_yaw, end_yaw, min_radius=5.5, spacing=0.25):
    """Return the shortest sampled feasible cubic among 36 handle pairs.

    Handles are [.25, .4, .55, .7, .9, 1.1] times endpoint XY distance.
    Positive handles preserve the specified analytic endpoint tangents.
    Analytic curvature is checked at parameter steps whose arc length is
    bounded by min(spacing, .05) using a derivative-control-point speed bound.
    These dense samples, with exact endpoints and linear Z, are returned.
    This is a sampled curvature check, NOT a global curvature certificate or
    an exhaustive path search. Invalid endpoints are never moved or snapped.
    """
    start, end = _point(start, "start"), _point(end, "end")
    start_yaw, end_yaw = _finite(start_yaw, "start_yaw"), _finite(end_yaw, "end_yaw")
    min_radius = _positive(min_radius, "min_radius")
    spacing = _positive(spacing, "spacing")
    d = _positive(_distance(start, end), "endpoint XY distance")
    yaw0, yaw1 = math.radians(start_yaw % 360.0), math.radians(end_yaw % 360.0)
    u, v = (math.cos(yaw0), math.sin(yaw0)), (math.cos(yaw1), math.sin(yaw1))
    delta = (end[0] - start[0], end[1] - start[1])
    fractions = (0.25, 0.4, 0.55, 0.7, 0.9, 1.1)
    best, best_length = None, math.inf
    for a in fractions:
        for b in fractions:
            controls = ((0.0, 0.0), (a * d * u[0], a * d * u[1]),
                        (delta[0] - b * d * v[0], delta[1] - b * d * v[1]), delta)
            edges = [tuple(q[k] - p[k] for k in (0, 1))
                     for p, q in zip(controls, controls[1:])]
            if not _regular_tangent(edges):
                continue
            speed_bound = _positive(3.0 * max(math.hypot(*edge) for edge in edges),
                                    "Bezier speed bound")
            count = max(1, math.ceil(speed_bound / min(spacing, 0.05)))
            points, length, previous = [], 0.0, None
            for i in range(count + 1):
                t = i / count
                q = 1.0 - t
                tangent = [3.0 * (q * q * edges[0][k] + 2.0 * q * t * edges[1][k]
                                  + t * t * edges[2][k]) for k in (0, 1)]
                second = [6.0 * (q * (edges[1][k] - edges[0][k])
                                 + t * (edges[2][k] - edges[1][k])) for k in (0, 1)]
                speed = math.hypot(*tangent)
                if speed <= 1e-10 * d:
                    break
                curvature = abs(tangent[0] * second[1] - tangent[1] * second[0]) / speed ** 3
                if not math.isfinite(curvature) or curvature > 1.0 / min_radius:
                    break
                xy = tuple(3.0 * q * q * t * controls[1][k]
                           + 3.0 * q * t * t * controls[2][k]
                           + t ** 3 * controls[3][k] for k in (0, 1))
                if previous is not None:
                    length += math.hypot(xy[0] - previous[0], xy[1] - previous[1])
                previous = xy
                points.append((start[0] + xy[0], start[1] + xy[1],
                               q * start[2] + t * end[2]))
            else:
                if length < best_length:
                    best, best_length = points, length
    if best is None:
        raise ValueError("No smooth connection satisfies the sampled curvature limit for min_radius; "
                         "move the endpoints farther apart or adjust their yaws")
    best[0], best[-1] = start, end
    return best
