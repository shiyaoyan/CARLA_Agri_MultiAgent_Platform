"""Sample the generated OpenDRIVE lanes for both FBX and diagnostic previews.

This exporter supports the generator's flat, single-section line/arc roads.
It deliberately rejects other geometry instead of silently approximating it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from xml.etree import ElementTree as ET

from .fbx import Mesh


@dataclass(frozen=True)
class LaneSample:
    s: float
    inner: tuple[float, float]
    outer: tuple[float, float]
    center: tuple[float, float]


def _polynomial(records, s: float, offset_name: str) -> float:
    record = next((item for item in reversed(records)
                   if float(item.get(offset_name, "0")) <= s + 1e-8), None)
    if record is None:
        return 0.0
    ds = max(0.0, s - float(record.get(offset_name, "0")))
    return sum(float(record.get(key, "0")) * ds ** i
               for i, key in enumerate(("a", "b", "c", "d")))


def sample_lanes(road: ET.Element, spacing: float = 0.5) -> dict[int, list[LaneSample]]:
    """Evaluate lane boundaries with analytic normals, including tapered widths."""
    sections = road.findall("./lanes/laneSection")
    if len(sections) != 1 or float(sections[0].get("s", "0")) != 0:
        raise ValueError("The flat FBX exporter requires one lane section at s=0.")
    lanes = sorted((lane for lane in sections[0].findall("./*/lane")
                    if lane.get("type") == "driving"),
                   key=lambda lane: abs(int(lane.get("id"))))
    samples = {int(lane.get("id")): [] for lane in lanes}
    offsets = road.findall("./lanes/laneOffset")
    for geom in road.findall("./planView/geometry"):
        child = next(iter(geom))
        if child.tag not in {"line", "arc"}:
            raise ValueError(f"Unsupported FBX geometry: {child.tag}")
        length = float(geom.get("length"))
        k = float(child.get("curvature", "0"))
        x, y, h, s0 = (float(geom.get(key)) for key in ("x", "y", "hdg", "s"))
        # Limit chord length and angular step, especially at sharp junctions.
        steps = max(1, math.ceil(length / spacing), math.ceil(abs(k * length) / 0.03))
        for index in range(steps + 1):
            ds = length * index / steps
            s, heading = s0 + ds, h + k * ds
            if abs(k) > 1e-12:
                px = x + (math.sin(heading) - math.sin(h)) / k
                py = y + (math.cos(h) - math.cos(heading)) / k
            else:
                px, py = x + math.cos(h) * ds, y + math.sin(h) * ds
            offset = _polynomial(offsets, s, "s")
            side_widths = {-1: 0.0, 1: 0.0}
            for lane in lanes:
                lane_id = int(lane.get("id"))
                side = 1 if lane_id > 0 else -1
                width = _polynomial(lane.findall("width"), s, "sOffset")
                inner = offset + side * side_widths[side]
                outer = inner + side * width
                side_widths[side] += width
                if width <= 0 or min(1 - k * inner, 1 - k * outer) <= 1e-4:
                    raise ValueError(
                        f"Road {road.get('id')} lane {lane_id} at s={s:.3f}: "
                        "non-positive width or folded lane; enlarge the bend/junction."
                    )
                def at(t):
                    return (px - math.sin(heading) * t, py + math.cos(heading) * t)
                sample = LaneSample(s, at(inner), at(outer), at((inner + outer) / 2))
                if samples[lane_id] and abs(samples[lane_id][-1].s - s) < 1e-7:
                    samples[lane_id][-1] = sample
                else:
                    samples[lane_id].append(sample)
    return samples


def road_surface_polygons(root: ET.Element, surfaces: dict[str, str]):
    """Union actual lane footprints and partition material overlaps at z=0."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    by_surface = {}
    main = {road.get("id"): road for road in root.findall("road")
            if road.get("junction", "-1") == "-1"}
    for road in root.findall("road"):
        surface = surfaces.get(road.get("id"))
        if surface is None:
            # Use the wider adjoining road's material for the whole connector.
            linked = [main[link.get("elementId")] for link in road.findall("./link/*")
                      if link.get("elementId") in main]
            widest = max(linked, key=lambda r: sum(
                float(w.get("a", "0")) for w in r.findall("./lanes/laneSection/*/lane/width")
            ))
            surface = surfaces[widest.get("id")]
        parts = by_surface.setdefault(surface, [])
        for lane_id, samples in sample_lanes(road).items():
            for a, b in zip(samples, samples[1:]):
                cell = Polygon([a.inner, a.outer, b.outer, b.inner])
                if not cell.is_valid:
                    raise ValueError(f"Road {road.get('id')} lane {lane_id}: self-crossing mesh cell.")
                if cell.area > 1e-12:
                    parts.append(cell)
    # Deterministic priority retains the paved main road through mixed junctions.
    priority = {"asphalt": 0, "concrete": 1, "gravel": 2, "dirt": 3}
    occupied = Polygon()
    result = {}
    for surface in sorted(by_surface, key=lambda name: (priority.get(name, 4), name)):
        footprint = unary_union(by_surface[surface])
        result[surface] = footprint.difference(occupied)
        occupied = occupied.union(footprint)
    return result


def _polygons(shape):
    if shape.is_empty:
        return
    if shape.geom_type == "Polygon":
        yield shape
    elif hasattr(shape, "geoms"):
        for part in shape.geoms:
            yield from _polygons(part)


def polygon_mesh(shape) -> Mesh:
    """Triangulate clipped polygons, weld vertices and verify covered area.

    Shapely's unconstrained Delaunay triangles can cross concave boundaries.
    Clip first; never retain an outside triangle based only on its centroid.
    """
    from shapely.ops import transform, triangulate

    # Use the same precision as vertex welding before triangulation. Intersecting
    # lane ribbons can otherwise leave near-coincident vertices that break GEOS.
    original = shape
    shape = transform(lambda x, y, z=None: (round(x, 9), round(y, 9)), shape).buffer(0)
    if original.symmetric_difference(shape).area > max(1e-6, original.area * 1e-8):
        raise ValueError("Road mesh precision normalization changed the footprint.")

    mesh = Mesh(vertices=[], faces=[])
    indices = {}
    for polygon in _polygons(shape):
        triangles = []
        pending = [polygon]
        for _ in range(5):
            if not pending:
                break
            next_pending = []
            for piece in pending:
                for triangle in triangulate(piece):
                    if piece.covers(triangle):
                        triangles.append(triangle)
                    else:
                        for clipped in _polygons(triangle.intersection(piece)):
                            if clipped.area > 1e-12:
                                next_pending.append(clipped)
            pending = next_pending
        area = sum(triangle.area for triangle in triangles)
        if abs(area - polygon.area) > max(1e-7, polygon.area * 1e-8):
            raise ValueError("Road mesh triangulation left uncovered area.")
        for triangle in triangles:
            if triangle.area <= 1e-12:
                continue
            coords = list(triangle.exterior.coords)[:3]
            a, b, c = coords
            if (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]) < 0:
                coords.reverse()
            face = []
            for x, y in coords:
                key = (round(x, 9), round(y, 9))
                if key not in indices:
                    indices[key] = len(mesh.vertices)
                    mesh.vertices.append((key[0], key[1], 0.0))
                face.append(indices[key])
            if len(set(face)) == 3:
                mesh.faces.append(tuple(face))
    return mesh
