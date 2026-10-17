"""Boundary-conforming farmland surfaces, shared by maps and standalone props."""

from __future__ import annotations

import math

from shapely.geometry import Polygon
from shapely.ops import triangulate, unary_union

from .fbx import Mesh


def validated_polygon(points):
    if len(points) < 3 or not all(math.isfinite(float(v)) for p in points for v in p):
        raise ValueError("A farmland/vegetation boundary needs at least three finite points.")
    polygon = Polygon(points)
    if not polygon.is_valid or polygon.area <= 1e-8:
        raise ValueError("A farmland/vegetation boundary must be simple and have positive area.")
    return polygon


def farmland_mesh(polygons, z=0.0):
    """Clip triangulation at concave boundaries; never fill their bounding boxes."""
    mesh = Mesh(vertices=[], faces=[])
    if not polygons:
        return mesh
    shape = unary_union([validated_polygon(points) for points in polygons])
    pending = list(shape.geoms) if shape.geom_type == "MultiPolygon" else [shape]
    triangles = []
    for _ in range(8):
        if not pending:
            break
        next_pending = []
        for polygon in pending:
            for triangle in triangulate(polygon):
                if polygon.covers(triangle):
                    triangles.append(triangle)
                else:
                    clipped = triangle.intersection(polygon)
                    parts = list(clipped.geoms) if hasattr(clipped, "geoms") else [clipped]
                    next_pending.extend(p for p in parts if p.geom_type == "Polygon" and p.area > 1e-12)
        pending = next_pending
    area = sum(t.area for t in triangles)
    if abs(area - shape.area) > max(1e-7, shape.area * 1e-8):
        raise ValueError("Farmland triangulation does not cover the supplied boundary.")
    for triangle in triangles:
        points = list(triangle.exterior.coords)[:3]
        a, b, c = points
        if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) < 0:
            points.reverse()
        start = len(mesh.vertices)
        mesh.vertices.extend((x, y, float(z)) for x, y in points)
        mesh.faces.append((start, start + 1, start + 2))
    return mesh
