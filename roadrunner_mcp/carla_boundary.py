"""Preserve physical roads while limiting CARLA 0.9.15 editor omissions.

The stock editor skips an entire road-entry lane if it has no predecessor.
Split a short *existing* terminal portion into its own ordinary road so that
the remaining road has a predecessor. No geometry is extended and no loop,
U-turn, duplicate surface or synthetic off-image road is added. The actual
outer boundary remains open; its short entry may still be absent in the
editor's debug routes. Runtime driving lanes cover the whole original road.
"""

from __future__ import annotations

from copy import deepcopy
import math
from xml.etree import ElementTree as ET


BOUNDARY_TAG = "carla_0915_boundary_segment"


def _fmt(number: float) -> str:
    return f"{number:.12f}".rstrip("0").rstrip(".") or "0"


def _slice_road(original: ET.Element, start: float, end: float) -> ET.Element:
    """Exact line/arc subinterval for the exporter's constant-width flat roads."""
    result = deepcopy(original)
    result.set("length", _fmt(end - start))
    plan = result.find("planView")
    plan.clear()
    for geometry in original.findall("./planView/geometry"):
        s0, length = float(geometry.get("s")), float(geometry.get("length"))
        lo, hi = max(start, s0), min(end, s0 + length)
        if hi - lo <= 1e-10:
            continue
        shape = next(iter(geometry))
        if shape.tag not in {"line", "arc"}:
            raise ValueError("Boundary compatibility supports only line/arc geometry.")
        k = float(shape.get("curvature", "0"))
        x, y, heading = (float(geometry.get(key)) for key in ("x", "y", "hdg"))
        ds = lo - s0
        if abs(k) > 1e-12:
            x += (math.sin(heading + k * ds) - math.sin(heading)) / k
            y += (math.cos(heading) - math.cos(heading + k * ds)) / k
        else:
            x += math.cos(heading) * ds
            y += math.sin(heading) * ds
        segment = ET.SubElement(plan, "geometry", {
            "s": _fmt(lo - start), "x": _fmt(x), "y": _fmt(y),
            "hdg": _fmt(heading + k * ds), "length": _fmt(hi - lo),
        })
        segment.append(deepcopy(shape))
    if not len(plan):
        raise ValueError("Boundary split produced an empty road geometry.")
    return result


def _set_link(road: ET.Element, relation: str, target_id: str, contact: str) -> None:
    link = road.find("link")
    if link is None:
        link = ET.Element("link")
        road.insert(0, link)
    previous = link.find(relation)
    if previous is not None:
        link.remove(previous)
    entry = ET.Element(relation, {"elementType": "road", "elementId": target_id,
                                  "contactPoint": contact})
    link.insert(0 if relation == "predecessor" else len(link), entry)
    for lane in road.findall("./lanes/laneSection/*/lane"):
        if lane.get("type") != "driving":
            continue
        lane_link = lane.find("link")
        if lane_link is None:
            lane_link = ET.Element("link")
            lane.insert(0, lane_link)
        old = lane_link.find(relation)
        if old is not None:
            lane_link.remove(old)
        lane_link.insert(0 if relation == "predecessor" else len(lane_link),
                         ET.Element(relation, {"id": lane.get("id")}))


def split_open_boundaries(root: ET.Element, segment_m: float) -> int:
    """Split terminal pieces in place, retain core road IDs and all junctions.

    Zero disables the adapter. The default caller uses 0.5 m; on shorter
    roads each terminal piece is capped at one quarter of the road length.
    Unsupported profiles fail explicitly instead of silently moving them.
    """
    if not math.isfinite(segment_m) or segment_m < 0:
        raise ValueError("carla_0915_boundary_segment_m must be finite and non-negative.")
    if segment_m == 0:
        return 0
    next_id = max((int(item.get("id")) for item in root
                   if item.tag in {"road", "junction"}), default=0) + 1
    added = []
    for road in list(root.findall("road")):
        if road.get("junction", "-1") != "-1" or road.find(f"userData[@code='{BOUNDARY_TAG}']") is not None:
            continue
        open_start = road.find("./link/predecessor") is None
        open_end = road.find("./link/successor") is None
        if not (open_start or open_end):
            continue
        sections = road.findall("./lanes/laneSection")
        if len(sections) != 1 or float(sections[0].get("s")) != 0:
            raise ValueError("Boundary compatibility requires a single lane section at s=0.")
        for path, offset in (("./elevationProfile/elevation", "s"), ("./lanes/laneOffset", "s"),
                             ("./lanes/laneSection/*/lane/width", "sOffset")):
            for record in road.findall(path):
                if float(record.get(offset, "0")) != 0 or any(float(record.get(k, "0")) != 0 for k in "bcd"):
                    raise ValueError("Boundary compatibility requires constant profiles/widths.")
        length = float(road.get("length"))
        cut = min(segment_m, length / 4)
        if cut < 0.05:
            raise ValueError("Road too short for stable boundary splitting; use segment_m=0 or a longer trace.")
        start, end = (cut if open_start else 0.0), (length - cut if open_end else length)
        core = _slice_road(road, start, end)
        # Preserve the original Element identity and its position before junctions.
        road_id = road.get("id")
        for is_start, lo, hi in ((True, 0.0, start), (False, end, length)):
            if hi <= lo:
                continue
            piece = _slice_road(road, lo, hi)
            piece_id = str(next_id)
            next_id += 1
            piece.set("id", piece_id)
            piece.set("name", road.get("name", "") + ("__boundary_start" if is_start else "__boundary_end"))
            piece.find("link").clear()
            for lane_link in piece.findall("./lanes/laneSection/*/lane/link"):
                lane_link.clear()
            _set_link(piece, "successor" if is_start else "predecessor", road_id,
                      "start" if is_start else "end")
            _set_link(core, "predecessor" if is_start else "successor", piece_id,
                      "end" if is_start else "start")
            ET.SubElement(piece, "userData", {"code": BOUNDARY_TAG, "value": road_id})
            added.append(piece)
        road.attrib.clear()
        road.attrib.update(core.attrib)
        road[:] = list(core)
    insert_at = next((i for i, child in enumerate(root) if child.tag == "junction"), len(root))
    for index, piece in enumerate(added):
        root.insert(insert_at + index, piece)
    return len(added)
