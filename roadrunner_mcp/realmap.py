"""经确认的卫星图道路几何到 CARLA 场景文件的确定性转换实现。"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple
from xml.dom import minidom
from xml.etree import ElementTree as ET

from .fbx import Mesh, MeshLayer, _build_ascii_fbx_layers, _surface_style, _write_tga_texture


Point = Tuple[float, float]


def sanitize_map_name(value: str) -> str:
    """把地图、道路或节点名称规范为 CARLA 资产可用的标识符。"""
    cleaned = "".join(
        character if character.isalnum() or character == "_" else "_"
        for character in str(value).strip()
    )
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    if not cleaned:
        return "generated_agricultural_scene"
    if cleaned[0].isdigit():
        cleaned = "map_" + cleaned
    return cleaned


@dataclass(frozen=True)
class GeoAnchor:
    """用于建立局部平面坐标系的经纬度与像素控制点。"""
    latitude: float
    longitude: float
    pixel_x: float
    pixel_y: float


@dataclass(frozen=True)
class TraceNode:
    """道路端点或交叉口的标识和图像像素坐标。"""
    node_id: str
    pixel: Point


@dataclass(frozen=True)
class TraceRoad:
    """一条道路的起止节点、像素中心线、实际宽度和材质。"""
    road_id: str
    start_node: str
    end_node: str
    pixel_polyline: list[Point]
    width_m: float = 3.0
    surface: str = "dirt"


@dataclass(frozen=True)
class RealMapTraceSpec:
    """从卫星图道路标注生成真实布局场景所需的完整规格。"""
    map_name: str
    image_path: str
    image_width_px: int
    image_height_px: int
    meters_per_pixel: float
    north_up: bool
    geo_anchor: GeoAnchor
    roads: list[TraceRoad]
    nodes: list[TraceNode]
    speed_kph: float = 20.0
    scene_profile: str = "agricultural"
    description: str = ""
    farmland_polygons: list[list[Point]] | None = None
    junction_clearance_m: float | None = None
    carla_0915_boundary_segment_m: float = 0.5

    def normalized(self) -> "RealMapTraceSpec":
        """规范道路名称、宽度、材质、比例尺和目标速度。"""
        clearance = self.junction_clearance_m
        boundary_segment = float(self.carla_0915_boundary_segment_m)
        if not math.isfinite(boundary_segment) or boundary_segment < 0:
            raise ValueError("carla_0915_boundary_segment_m must be finite and non-negative.")
        if clearance is not None:
            clearance = float(clearance)
            if not math.isfinite(clearance) or clearance <= 0:
                raise ValueError("junction_clearance_m must be finite and positive.")
        surface_roads = [
            TraceRoad(
                road_id=sanitize_map_name(road.road_id),
                start_node=road.start_node,
                end_node=road.end_node,
                pixel_polyline=list(road.pixel_polyline),
                width_m=max(2.0, min(12.0, float(road.width_m))),
                surface=road.surface.lower().strip() if road.surface.lower().strip() in {"asphalt", "concrete", "gravel", "dirt"} else "dirt",
            )
            for road in self.roads
        ]
        return RealMapTraceSpec(
            map_name=sanitize_map_name(self.map_name),
            image_path=self.image_path,
            image_width_px=int(self.image_width_px),
            image_height_px=int(self.image_height_px),
            meters_per_pixel=max(0.01, float(self.meters_per_pixel)),
            north_up=bool(self.north_up),
            geo_anchor=self.geo_anchor,
            roads=surface_roads,
            nodes=list(self.nodes),
            speed_kph=max(5.0, min(60.0, float(self.speed_kph))),
            scene_profile=self.scene_profile,
            description=self.description,
            farmland_polygons=self.farmland_polygons,
            junction_clearance_m=clearance,
            carla_0915_boundary_segment_m=boundary_segment,
        )

    def to_dict(self) -> dict[str, object]:
        """转换为可保存和复现的结构化字典。"""
        return asdict(self)


@dataclass(frozen=True)
class RealMapGenerationResult:
    """真实布局场景生成后的文件路径、拓扑计数和世界边界。"""
    xodr_path: Path
    fbx_path: Path
    texture_path: Path
    texture_paths: tuple[Path, ...]
    spec_path: Path
    preview_svg_path: Path
    overlay_png_path: Path | None
    import_bundle_dir: Path
    road_count: int
    junction_count: int
    connection_count: int
    bounds: dict[str, float]


@dataclass(frozen=True)
class WorldRoad:
    """已由图像像素坐标转换到局部米制坐标的道路。"""
    index: int
    source: TraceRoad
    polyline: list[Point]

    @property
    def length(self) -> float:
        """返回道路折线的累计长度。"""
        return _polyline_length(self.polyline)


@dataclass(frozen=True)
class EndpointRef:
    """道路在某个交叉节点处的端点方向与连接边界。"""
    road: WorldRoad
    node_id: str
    at_start: bool
    away_vector: Point
    boundary: Point

    @property
    def contact_point(self) -> str:
        """返回该端点在 OpenDRIVE 道路参考线上的 start/end。"""
        return "start" if self.at_start else "end"

    @property
    def incoming_lane(self) -> int:
        """返回车辆沿此道路进入节点时使用的车道编号。"""
        return 1 if self.at_start else -1

    @property
    def outgoing_lane(self) -> int:
        """返回车辆由节点驶入此道路时使用的车道编号。"""
        return -1 if self.at_start else 1


@dataclass(frozen=True)
class JunctionConnection:
    """真实布局路口的一项道路连接和单车道映射。"""
    connection_id: int
    incoming_road_id: int
    connecting_road_id: int
    contact_point: str
    lane_from: int
    lane_to: int


@dataclass(frozen=True)
class PlanPrimitive:
    """OpenDRIVE参考线中的一段直线或恒曲率圆弧。"""

    kind: str
    start: Point
    heading: float
    length: float
    curvature: float = 0.0


def generate_realmap_artifacts(spec: RealMapTraceSpec, output_dir: Path) -> RealMapGenerationResult:
    """生成真实道路布局的 XODR、FBX、纹理、预览和导入包。"""
    spec = spec.normalized()
    output_dir.mkdir(parents=True, exist_ok=True)
    world_roads = _world_roads(spec)
    bounds = _bounds(spec, world_roads)
    xodr_text, road_count, junction_count, connection_count = _build_opendrive(spec, world_roads, bounds)

    xodr_path = output_dir / f"{spec.map_name}.xodr"
    fbx_path = output_dir / f"{spec.map_name}.fbx"
    spec_path = output_dir / f"{spec.map_name}.rrspec.json"
    preview_svg_path = output_dir / f"{spec.map_name}.preview.svg"
    overlay_candidate = output_dir / f"{spec.map_name}.overlay.png"
    overlay_png_path: Path | None = overlay_candidate
    import_bundle_dir = output_dir / f"{spec.map_name}_carla_import"

    xodr_path.write_text(xodr_text, encoding="utf-8")
    texture_paths = _write_realmap_fbx(spec, world_roads, bounds, fbx_path, xodr_text)
    primary_name = f"{spec.map_name}_{_primary_surface(spec)}_diffuse.tga"
    texture_path = next(path for path in texture_paths if path.name == primary_name)
    spec_path.write_text(json.dumps(spec.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    preview_svg_path.write_text(_build_svg_preview(spec, world_roads, bounds, xodr_text), encoding="utf-8")
    if not _write_overlay_png(spec, overlay_candidate):
        overlay_png_path = None

    # 导入包只放地图资源和 CARLA package manifest，不混入诊断 JSON。
    if import_bundle_dir.exists():
        shutil.rmtree(import_bundle_dir)
    import_bundle_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(xodr_path, import_bundle_dir / xodr_path.name)
    shutil.copyfile(fbx_path, import_bundle_dir / fbx_path.name)
    for generated_texture in texture_paths:
        shutil.copyfile(
            generated_texture,
            import_bundle_dir / generated_texture.name,
        )
    (import_bundle_dir / f"{spec.map_name}_package.json").write_text(
        json.dumps({"maps": [{
            "name": spec.map_name,
            "source": fbx_path.name,
            "xodr": xodr_path.name,
            "use_carla_materials": False,
        }], "props": []}, indent=2) + "\n",
        encoding="utf-8",
    )

    return RealMapGenerationResult(
        xodr_path=xodr_path,
        fbx_path=fbx_path,
        texture_path=texture_path,
        texture_paths=tuple(texture_paths),
        spec_path=spec_path,
        preview_svg_path=preview_svg_path,
        overlay_png_path=overlay_png_path,
        import_bundle_dir=import_bundle_dir,
        road_count=road_count,
        junction_count=junction_count,
        connection_count=connection_count,
        bounds=bounds,
    )


def _build_opendrive(
    spec: RealMapTraceSpec,
    world_roads: list[WorldRoad],
    bounds: dict[str, float],
) -> tuple[str, int, int, int]:
    """由米制道路折线建立可供 CARLA 导航的 OpenDRIVE 拓扑。

    节点度大于 1 时视为路口。每条相连道路保留双向车道，并为不同道路
    之间的每个允许连接建立一条路口连接道路和 laneLink。
    """
    root = ET.Element("OpenDRIVE")
    header = ET.SubElement(
        root,
        "header",
        {
            "revMajor": "1",
            "revMinor": "4",
            "name": spec.map_name,
            "version": "1.00",
            "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "north": _fmt(bounds["north"]),
            "south": _fmt(bounds["south"]),
            "east": _fmt(bounds["east"]),
            "west": _fmt(bounds["west"]),
            "vendor": "agri-drive-gen-realmap",
        },
    )
    geo = ET.SubElement(header, "geoReference")
    # 控制点经纬度作为局部横轴墨卡托投影原点，不把像素位置冒充真实 GNSS。
    geo.text = (
        f"+proj=tmerc +lat_0={spec.geo_anchor.latitude:.8f} "
        f"+lon_0={spec.geo_anchor.longitude:.8f} +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )

    node_degree = _node_degree(spec)
    # CARLA 0.9.15 MapBuilder::GetLaneNext uses ContainsRoad(link_id) to
    # distinguish roads from junctions. Reserve IDs beyond ALL main roads and
    # connectors; separate XML element types alone do not prevent a collision.
    road_id_ceiling = len(world_roads) + sum(
        degree * (degree - 1) for degree in node_degree.values() if degree > 1
    )
    junction_nodes = {
        node_id: road_id_ceiling + index + 1
        for index, node_id in enumerate(sorted(
            node for node, degree in node_degree.items() if degree > 1
        ))
    }
    junction_radii = _junction_radii(
        world_roads, set(junction_nodes), spec.junction_clearance_m,
    )
    endpoint_refs, trimmed_roads = _junction_endpoint_refs(
        world_roads,
        set(junction_nodes),
        junction_radii,
    )
    road_count = 0

    # 主路在路口边界处终止，路口内部仅由连接道路承担。这样可保证道路
    # reference line、junction connection 和 laneLink 在同一位置连续衔接。
    for road in trimmed_roads:
        predecessor: tuple[str, int] | None = (
            ("junction", junction_nodes[road.source.start_node])
            if road.source.start_node in junction_nodes
            else None
        )
        successor: tuple[str, int] | None = (
            ("junction", junction_nodes[road.source.end_node])
            if road.source.end_node in junction_nodes
            else None
        )
        _add_road(
            root,
            road.index,
            road.source.road_id,
            -1,
            road.polyline,
            max(road.source.width_m / 2.0, 0.75),
            spec.speed_kph,
            predecessor,
            successor,
        )
        road_count += 1

    junction_connections: dict[str, list[JunctionConnection]] = {node_id: [] for node_id in junction_nodes}
    connector_id = len(world_roads) + 1
    for node_id in sorted(junction_nodes):
        refs = [ref for ref in endpoint_refs if ref.node_id == node_id]
        center = _pixel_to_world(
            spec,
            next(node.pixel for node in spec.nodes if node.node_id == node_id),
        )
        local_connection_id = 0
        for incoming in refs:
            for outgoing in refs:
                if incoming.road.index == outgoing.road.index:
                    continue
                local_connection_id += 1
                incoming_lane = incoming.incoming_lane
                outgoing_lane = outgoing.outgoing_lane
                if incoming_lane < 0:
                    # 右侧车道与道路参考线方向一致。
                    polyline = [incoming.boundary, center, outgoing.boundary]
                    connector_lane = -1
                    connector_predecessor = (
                        "road",
                        incoming.road.index,
                        incoming.contact_point,
                    )
                    connector_successor = (
                        "road",
                        outgoing.road.index,
                        outgoing.contact_point,
                    )
                    predecessor_lane = incoming_lane
                    successor_lane = outgoing_lane
                    lane_side = "right"
                    contact_point = "start"
                    start_lane_width = incoming.road.source.width_m / 2.0
                    end_lane_width = outgoing.road.source.width_m / 2.0
                else:
                    # 左侧车道的行驶方向与道路参考线相反，连接道路需反向书写。
                    polyline = [outgoing.boundary, center, incoming.boundary]
                    connector_lane = 1
                    connector_predecessor = (
                        "road",
                        outgoing.road.index,
                        outgoing.contact_point,
                    )
                    connector_successor = (
                        "road",
                        incoming.road.index,
                        incoming.contact_point,
                    )
                    predecessor_lane = outgoing_lane
                    successor_lane = incoming_lane
                    lane_side = "left"
                    contact_point = "end"
                    start_lane_width = outgoing.road.source.width_m / 2.0
                    end_lane_width = incoming.road.source.width_m / 2.0
                _add_connector_road(
                    root,
                    connector_id,
                    f"J_{node_id}_{incoming.road.source.road_id}_to_{outgoing.road.source.road_id}",
                    junction_nodes[node_id],
                    polyline,
                    max(start_lane_width, 0.75),
                    max(end_lane_width, 0.75),
                    spec.speed_kph,
                    connector_predecessor,
                    connector_successor,
                    connector_lane,
                    predecessor_lane,
                    successor_lane,
                    lane_side,
                )
                junction_connections[node_id].append(
                    JunctionConnection(
                        connection_id=local_connection_id,
                        incoming_road_id=incoming.road.index,
                        connecting_road_id=connector_id,
                        contact_point=contact_point,
                        lane_from=incoming_lane,
                        lane_to=connector_lane,
                    )
                )
                connector_id += 1
                road_count += 1

    # 最后将连接道路登记到 OpenDRIVE junction/connection/laneLink 层级。
    for node_id, junction_id in junction_nodes.items():
        junction = ET.SubElement(root, "junction", {"id": str(junction_id), "name": f"J_{node_id}"})
        for connection in junction_connections[node_id]:
            node = ET.SubElement(
                junction,
                "connection",
                {
                    "id": str(connection.connection_id),
                    "incomingRoad": str(connection.incoming_road_id),
                    "connectingRoad": str(connection.connecting_road_id),
                    "contactPoint": connection.contact_point,
                },
            )
            ET.SubElement(node, "laneLink", {"from": str(connection.lane_from), "to": str(connection.lane_to)})

    from .carla_boundary import split_open_boundaries
    road_count += split_open_boundaries(root, spec.carla_0915_boundary_segment_m)
    return _prettify_xml(root), road_count, len(junction_nodes), sum(len(items) for items in junction_connections.values())


def _add_road(
    root: ET.Element,
    road_id: int,
    name: str,
    junction_id: int,
    polyline: list[Point],
    lane_width: float,
    speed_kph: float,
    predecessor: tuple[str, int] | None,
    successor: tuple[str, int] | None,
) -> None:
    """添加一条经过圆弧平滑的双向、无标线农业道路。"""
    primitives = _build_plan_primitives(polyline, corner_fraction=0.25)
    road = ET.SubElement(root, "road", {"name": name, "length": _fmt(_plan_length(primitives)), "id": str(road_id), "junction": str(junction_id)})
    _add_road_link(road, predecessor, successor)
    _add_road_type(road, speed_kph)
    _add_plan_view(road, primitives)
    _add_flat_profile(road)
    _add_two_way_lane_section(road, lane_width, road_mark_type="none")


def _add_connector_road(
    root: ET.Element,
    road_id: int,
    name: str,
    junction_id: int,
    polyline: list[Point],
    start_lane_width: float,
    end_lane_width: float,
    speed_kph: float,
    predecessor: tuple[str, int, str],
    successor: tuple[str, int, str],
    connector_lane: int,
    predecessor_lane: int,
    successor_lane: int,
    lane_side: str,
) -> None:
    """添加路口内部单向连接道路并关联前后车道。"""
    primitives = _build_plan_primitives(polyline, corner_fraction=0.75)
    road = ET.SubElement(root, "road", {"name": name, "length": _fmt(_plan_length(primitives)), "id": str(road_id), "junction": str(junction_id)})
    _add_road_link(road, predecessor, successor)
    _add_road_type(road, speed_kph)
    _add_plan_view(road, primitives)
    _add_flat_profile(road)
    lanes = ET.SubElement(road, "lanes")
    ET.SubElement(lanes, "laneOffset", {"s": "0.0", "a": "0.0", "b": "0.0", "c": "0.0", "d": "0.0"})
    lane_section = ET.SubElement(lanes, "laneSection", {"s": "0.0"})
    if lane_side == "left":
        left = ET.SubElement(lane_section, "left")
        _add_lane(
            left,
            connector_lane,
            "driving",
            start_lane_width,
            predecessor_lane,
            successor_lane,
            end_lane_width=end_lane_width,
            road_length=_plan_length(primitives),
        )
        _add_center(lane_section)
    else:
        _add_center(lane_section)
        right = ET.SubElement(lane_section, "right")
        _add_lane(
            right,
            connector_lane,
            "driving",
            start_lane_width,
            predecessor_lane,
            successor_lane,
            end_lane_width=end_lane_width,
            road_length=_plan_length(primitives),
        )


def _add_road_link(road: ET.Element, predecessor: tuple | None, successor: tuple | None) -> None:
    """写入道路与相邻道路或路口的前驱/后继关系。"""
    link = ET.SubElement(road, "link")
    if predecessor is not None:
        attrib = {"elementType": str(predecessor[0]), "elementId": str(predecessor[1])}
        if len(predecessor) > 2:
            attrib["contactPoint"] = str(predecessor[2])
        ET.SubElement(link, "predecessor", attrib)
    if successor is not None:
        attrib = {"elementType": str(successor[0]), "elementId": str(successor[1])}
        if len(successor) > 2:
            attrib["contactPoint"] = str(successor[2])
        ET.SubElement(link, "successor", attrib)


def _add_road_type(road: ET.Element, speed_kph: float) -> None:
    """将道路声明为 rural，并把 km/h 转换为 OpenDRIVE 所用 m/s。"""
    road_type = ET.SubElement(road, "type", {"s": "0.0", "type": "rural"})
    ET.SubElement(road_type, "speed", {"max": _fmt(speed_kph / 3.6), "unit": "m/s"})


def _add_plan_view(road: ET.Element, primitives: list[PlanPrimitive]) -> None:
    """把已平滑的直线和圆弧依次写入OpenDRIVE planView。"""
    plan_view = ET.SubElement(road, "planView")
    s = 0.0
    for primitive in primitives:
        geom = ET.SubElement(
            plan_view,
            "geometry", {
                "s": _fmt(s),
                "x": _fmt(primitive.start[0]),
                "y": _fmt(primitive.start[1]),
                "hdg": _fmt(primitive.heading),
                "length": _fmt(primitive.length),
            },
        )
        if primitive.kind == "arc":
            ET.SubElement(
                geom,
                "arc",
                {"curvature": _fmt(primitive.curvature)},
            )
        else:
            ET.SubElement(geom, "line")
        s += primitive.length


def _add_flat_profile(road: ET.Element) -> None:
    """为当前不含坡度的真实布局道路写入零高程剖面。"""
    elevation = ET.SubElement(road, "elevationProfile")
    ET.SubElement(elevation, "elevation", {"s": "0.0", "a": "0.0", "b": "0.0", "c": "0.0", "d": "0.0"})
    ET.SubElement(road, "lateralProfile")


def _add_two_way_lane_section(road: ET.Element, lane_width: float, road_mark_type: str) -> None:
    """在道路参考线两侧各创建一条行车道。"""
    lanes = ET.SubElement(road, "lanes")
    ET.SubElement(lanes, "laneOffset", {"s": "0.0", "a": "0.0", "b": "0.0", "c": "0.0", "d": "0.0"})
    lane_section = ET.SubElement(lanes, "laneSection", {"s": "0.0"})
    left = ET.SubElement(lane_section, "left")
    _add_lane(left, 1, "driving", lane_width)
    _add_center(lane_section, road_mark_type)
    right = ET.SubElement(lane_section, "right")
    _add_lane(right, -1, "driving", lane_width)


def _add_center(lane_section: ET.Element, road_mark_type: str = "none") -> None:
    """添加无可驾驶宽度的 0 号中心车道。"""
    center = ET.SubElement(lane_section, "center")
    lane = ET.SubElement(center, "lane", {"id": "0", "type": "none", "level": "false"})
    ET.SubElement(
        lane,
        "roadMark",
        {"sOffset": "0.0", "type": road_mark_type, "weight": "standard", "color": "standard", "width": "0.0", "laneChange": "none"},
    )


def _add_lane(
    parent: ET.Element,
    lane_id: int,
    lane_type: str,
    lane_width: float,
    predecessor_lane_id: int | None = None,
    successor_lane_id: int | None = None,
    end_lane_width: float | None = None,
    road_length: float | None = None,
) -> None:
    """添加一条行车道及可选的前驱和后继车道链接。"""
    lane = ET.SubElement(parent, "lane", {"id": str(lane_id), "type": lane_type, "level": "false"})
    link = ET.SubElement(lane, "link")
    if predecessor_lane_id is not None:
        ET.SubElement(link, "predecessor", {"id": str(predecessor_lane_id)})
    if successor_lane_id is not None:
        ET.SubElement(link, "successor", {"id": str(successor_lane_id)})
    width_c = width_d = 0.0
    if (
        end_lane_width is not None
        and road_length is not None
        and road_length > 1e-9
    ):
        # Smoothstep taper: zero width derivative at both connections.
        delta = end_lane_width - lane_width
        width_c = 3.0 * delta / road_length ** 2
        width_d = -2.0 * delta / road_length ** 3
    ET.SubElement(
        lane,
        "width",
        {
            "sOffset": "0.0",
            "a": _fmt(lane_width),
            "b": "0.0",
            "c": _fmt(width_c),
            "d": _fmt(width_d),
        },
    )
    ET.SubElement(
        lane,
        "roadMark",
        {"sOffset": "0.0", "type": "none", "weight": "standard", "color": "standard", "width": "0.0", "laneChange": "none"},
    )


def _write_realmap_fbx(
    spec: RealMapTraceSpec,
    roads: list[WorldRoad],
    bounds: dict[str, float],
    fbx_path: Path,
    xodr_text: str,
) -> list[Path]:
    """从最终 XODR 车道足迹构建道路，合并路口重叠面后写出 FBX。"""
    from .road_mesh import polygon_mesh, road_surface_polygons

    surface_order = list(dict.fromkeys(road.source.surface for road in roads))
    layer_surfaces = ["field", *surface_order]
    textures = {
        surface: fbx_path.with_name(f"{spec.map_name}_{surface}_diffuse.tga")
        for surface in layer_surfaces
    }
    for surface, path in textures.items():
        _write_tga_texture(path, _surface_style(surface))

    meshes = {
        surface: Mesh(vertices=[], faces=[])
        for surface in layer_surfaces
    }
    from .farmland import farmland_mesh

    field_polygons = spec.farmland_polygons
    if field_polygons is None:
        field_polygons = [[
            (bounds["west"], bounds["south"]),
            (bounds["east"], bounds["south"]),
            (bounds["east"], bounds["north"]),
            (bounds["west"], bounds["north"]),
        ]]
    meshes["field"] = farmland_mesh(field_polygons, z=-0.005)
    surfaces = {str(road.index): road.source.surface for road in roads}
    for surface, footprint in road_surface_polygons(
        ET.fromstring(xodr_text), surfaces,
    ).items():
        meshes[surface] = polygon_mesh(footprint)

    layers = [
        MeshLayer(
            name=(
                "Terrain_Ground_Layer0"
                if surface == "field"
                else f"Roads_Road_{surface}_Layer0"
            ),
            mesh=meshes[surface],
            style=_surface_style(surface),
            texture_filename=textures[surface].name,
        )
        for surface in layer_surfaces
        if meshes[surface].faces
    ]
    fbx_path.write_text(
        _build_ascii_fbx_layers(spec.map_name, layers),
        encoding="utf-8",
    )
    return [textures[surface] for surface in layer_surfaces]


def _add_polyline_ribbon(
    mesh: Mesh,
    polyline: list[Point],
    width: float,
) -> None:
    """沿平滑中心线生成共享边界顶点的连续道路带状网格。"""
    points = _remove_duplicate_points(polyline)
    if len(points) < 2:
        return
    half_width = width / 2.0
    sides: list[tuple[Point, Point]] = []
    for index, point in enumerate(points):
        if index == 0:
            tangent = _normalize(_subtract(points[1], point))
        elif index == len(points) - 1:
            tangent = _normalize(_subtract(point, points[index - 1]))
        else:
            incoming = _normalize(_subtract(point, points[index - 1]))
            outgoing = _normalize(_subtract(points[index + 1], point))
            tangent = _normalize((incoming[0] + outgoing[0], incoming[1] + outgoing[1]))
            if abs(tangent[0]) + abs(tangent[1]) <= 1e-9:
                tangent = outgoing
        normal = (-tangent[1], tangent[0])
        left = (point[0] + normal[0] * half_width, point[1] + normal[1] * half_width)
        right = (point[0] - normal[0] * half_width, point[1] - normal[1] * half_width)
        sides.append((left, right))

    base = len(mesh.vertices)
    for left, right in sides:
        mesh.vertices.extend(((left[0], left[1], 0.0), (right[0], right[1], 0.0)))
    for index in range(len(sides) - 1):
        left_index = base + index * 2
        right_index = left_index + 1
        next_left_index = left_index + 2
        next_right_index = left_index + 3
        # 顶点按从上方观察的逆时针方向排列，使道路面的几何法向稳定朝向 +Z。
        mesh.faces.append((left_index, next_right_index, next_left_index))
        mesh.faces.append((left_index, right_index, next_right_index))


def _add_disc(
    mesh: Mesh,
    center: Point,
    radius: float,
    z: float,
    segments: int = 32,
) -> None:
    """以圆形覆盖路口中心，避免方形补片在斜交道路外侧露出。"""
    base = len(mesh.vertices)
    mesh.vertices.append((center[0], center[1], z))
    for index in range(segments):
        angle = 2.0 * math.pi * index / segments
        mesh.vertices.append((
            center[0] + math.cos(angle) * radius,
            center[1] + math.sin(angle) * radius,
            z,
        ))
    for index in range(segments):
        current = base + 1 + index
        following = base + 1 + (index + 1) % segments
        mesh.faces.append((base, current, following))


def _add_quad(mesh: Mesh, points: list[Point], z: float) -> None:
    """把四个平面点写成指定高度的两个三角面。"""
    base = len(mesh.vertices)
    for x, y in points:
        mesh.vertices.append((x, y, z))
    mesh.faces.append((base, base + 1, base + 2))
    mesh.faces.append((base, base + 2, base + 3))


def _build_svg_preview(spec: RealMapTraceSpec, roads: list[WorldRoad], bounds: dict[str, float], xodr_text: str) -> str:
    """生成真实布局道路场景的矢量预览。"""
    width = 960
    height = 720
    margin = 40
    west, east, south, north = bounds["west"], bounds["east"], bounds["south"], bounds["north"]
    scale = min((width - 2 * margin) / max(1.0, east - west), (height - 2 * margin) / max(1.0, north - south))

    def sx(x: float) -> float:
        """将局部世界 x 坐标映射到预览图。"""
        return margin + (x - west) * scale

    def sy(y: float) -> float:
        """将局部世界 y 坐标映射到预览图并翻转纵轴。"""
        return height - margin - (y - south) * scale

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#eef2e6"/>',
    ]
    for polygon in spec.farmland_polygons or []:
        points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in polygon)
        lines.append(f'<polygon points="{points}" fill="#93b778" stroke="#527943" stroke-width="1.5"/>')
    preview_colors = {
        "asphalt": "#4b5563",
        "concrete": "#8b8f87",
        "gravel": "#8a735b",
        "dirt": "#7c4a2f",
    }
    from .road_mesh import _polygons, road_surface_polygons, sample_lanes

    root = ET.fromstring(xodr_text)
    surfaces = {str(road.index): road.source.surface for road in roads}
    for surface, footprint in road_surface_polygons(root, surfaces).items():
        for polygon in _polygons(footprint):
            rings = [polygon.exterior, *polygon.interiors]
            path = " ".join(
                "M " + " L ".join(f"{sx(x):.3f},{sy(y):.3f}" for x, y in ring.coords) + " Z"
                for ring in rings
            )
            lines.append(f'<path d="{path}" fill="{preview_colors[surface]}" fill-rule="evenodd"/>')
    for road in root.findall("road"):
        for lane_id, samples in sample_lanes(road).items():
            color = "#f59e0b" if road.get("junction") != "-1" else ("#38bdf8" if lane_id < 0 else "#fb7185")
            points = " ".join(f"{sx(p.center[0]):.3f},{sy(p.center[1]):.3f}" for p in samples)
            lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1" opacity=".9"/>')
    width_labels = "/".join(
        f"{width:g}" for width in sorted({road.source.width_m for road in roads})
    ) or "n/a"
    surface_labels = "/".join(
        sorted({road.source.surface for road in roads})
    ) or "n/a"
    lines.append(f'<text x="32" y="38" font-family="Arial" font-size="22" font-weight="700" fill="#1f2937">{spec.map_name}</text>')
    lines.append(f'<text x="32" y="66" font-family="Arial" font-size="14" fill="#1f2937">real satellite trace, {len(roads)} roads, widths {width_labels} m, {surface_labels}</text>')
    lines.append('<text x="32" y="88" font-family="Arial" font-size="13" fill="#1f2937">Lane centers: blue = -1, pink = +1, amber = junction (not painted road markings)</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _write_overlay_png(spec: RealMapTraceSpec, output_path: Path) -> bool:
    """若 Pillow 和原卫星图可用，则绘制道路中心线与节点叠加预览。"""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return False

    image_path = Path(spec.image_path)
    if not image_path.is_file():
        return False
    image = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for road in spec.roads:
        draw.line(road.pixel_polyline, fill=(0, 168, 255, 230), width=8, joint="curve")
        draw.line(road.pixel_polyline, fill=(255, 255, 255, 210), width=2)
    for node in spec.nodes:
        degree = sum(1 for road in spec.roads if road.start_node == node.node_id or road.end_node == node.node_id)
        color = (255, 214, 10, 255) if degree > 1 else (39, 174, 96, 240)
        x, y = node.pixel
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color, outline=(20, 20, 20, 255), width=1)
    Image.alpha_composite(image, overlay).save(output_path)
    return True


def _junction_endpoint_refs(
    roads: list[WorldRoad],
    junction_nodes: set[str],
    junction_radii: dict[str, float],
) -> tuple[list[EndpointRef], list[WorldRoad]]:
    """把主路裁切到路口边界，并返回每个连接端点的方向。"""
    refs: list[EndpointRef] = []
    trimmed_roads: list[WorldRoad] = []
    for road in roads:
        total = road.length
        start_is_junction = road.source.start_node in junction_nodes
        end_is_junction = road.source.end_node in junction_nodes
        start_radius = junction_radii.get(road.source.start_node, 0.0)
        end_radius = junction_radii.get(road.source.end_node, 0.0)
        if start_is_junction and end_is_junction:
            start_trim = min(start_radius, total * 0.35)
            end_trim = min(end_radius, total * 0.35)
        else:
            start_trim = min(start_radius, total * 0.45) if start_is_junction else 0.0
            end_trim = min(end_radius, total * 0.45) if end_is_junction else 0.0

        trimmed = _slice_polyline(road.polyline, start_trim, total - end_trim)
        trimmed_road = WorldRoad(road.index, road.source, trimmed)
        trimmed_roads.append(trimmed_road)
        if start_is_junction:
            tangent = _normalize(_subtract(trimmed[1], trimmed[0]))
            refs.append(EndpointRef(
                road,
                road.source.start_node,
                True,
                tangent,
                trimmed[0],
            ))
        if end_is_junction:
            tangent = _normalize(_subtract(trimmed[-1], trimmed[-2]))
            refs.append(EndpointRef(
                road,
                road.source.end_node,
                False,
                (-tangent[0], -tangent[1]),
                trimmed[-1],
            ))
    return refs, trimmed_roads


def _world_roads(spec: RealMapTraceSpec) -> list[WorldRoad]:
    """将所有道路中心线从像素坐标批量转换为局部米制坐标。"""
    return [
        WorldRoad(index=index + 1, source=road, polyline=[_pixel_to_world(spec, pixel) for pixel in road.pixel_polyline])
        for index, road in enumerate(spec.roads)
    ]


def _pixel_to_world(spec: RealMapTraceSpec, pixel: Point) -> Point:
    """以地理控制点为局部原点，按米/像素比例转换单个像素点。

    对北向上的图像，图像 y 向下增长而局部世界 y 向北增长，因此 y 轴取反。
    """
    x = (pixel[0] - spec.geo_anchor.pixel_x) * spec.meters_per_pixel
    y_sign = -1.0 if spec.north_up else 1.0
    y = (pixel[1] - spec.geo_anchor.pixel_y) * spec.meters_per_pixel * y_sign
    return (x, y)


def _node_degree(spec: RealMapTraceSpec) -> dict[str, int]:
    """统计每个节点连接的道路数量，用于识别交叉口。"""
    degree = {node.node_id: 0 for node in spec.nodes}
    for road in spec.roads:
        degree[road.start_node] = degree.get(road.start_node, 0) + 1
        degree[road.end_node] = degree.get(road.end_node, 0) + 1
    return degree


def _bounds(
    spec: RealMapTraceSpec,
    roads: list[WorldRoad],
) -> dict[str, float]:
    """按卫星图实际覆盖范围建立与道路同源的局部地图边界。"""
    points = [point for road in roads for point in road.polyline]
    points.extend(point for polygon in spec.farmland_polygons or [] for point in polygon)
    if spec.image_width_px > 0 and spec.image_height_px > 0:
        max_pixel_x = float(max(0, spec.image_width_px - 1))
        max_pixel_y = float(max(0, spec.image_height_px - 1))
        points.extend(
            _pixel_to_world(spec, pixel)
            for pixel in (
                (0.0, 0.0),
                (max_pixel_x, 0.0),
                (max_pixel_x, max_pixel_y),
                (0.0, max_pixel_y),
            )
        )
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    pad = max(max((road.source.width_m for road in roads), default=3.0), 5.0)
    return {"west": min(xs) - pad, "east": max(xs) + pad, "south": min(ys) - pad, "north": max(ys) + pad}


def _junction_radius(roads: list[WorldRoad]) -> float:
    """根据最宽道路确定主路裁切和路口连接的共同边界半径。"""
    maximum_width = max((road.source.width_m for road in roads), default=3.0)
    return max(maximum_width * 0.75, 4.0)


def _junction_radii(
    roads: list[WorldRoad],
    junction_nodes: set[str],
    minimum_clearance_m: float | None = None,
) -> dict[str, float]:
    """分别按各路口的相邻道路宽度计算裁切半径，避免远处宽路放大所有路口。"""
    return {
        node_id: max(minimum_clearance_m or 0.0, _junction_radius([
            road
            for road in roads
            if node_id in (road.source.start_node, road.source.end_node)
        ]))
        for node_id in junction_nodes
    }


def _slice_polyline(
    polyline: list[Point],
    start_s: float,
    end_s: float,
) -> list[Point]:
    """按累计距离截取折线，保留截取区间内的原始控制点。"""
    points = _remove_duplicate_points(polyline)
    total = _polyline_length(points)
    start_s = min(max(0.0, start_s), total)
    end_s = min(max(start_s, end_s), total)
    if end_s - start_s <= 1e-6:
        return [points[0], points[-1]]
    start_point, _ = _point_and_tangent_at_s(points, start_s)
    end_point, _ = _point_and_tangent_at_s(points, end_s)
    result = [start_point]
    cumulative = 0.0
    for first, second in zip(points, points[1:]):
        cumulative += _distance(first, second)
        if start_s + 1e-6 < cumulative < end_s - 1e-6:
            result.append(second)
    result.append(end_point)
    return _remove_duplicate_points(result)


def _point_and_tangent_at_s(
    polyline: list[Point],
    target_s: float,
) -> tuple[Point, Point]:
    """返回折线累计距离处的位置和沿参考线正方向的单位切向量。"""
    remaining = max(0.0, target_s)
    for start, end in zip(polyline, polyline[1:]):
        segment_length = _distance(start, end)
        if segment_length <= 1e-9:
            continue
        tangent = ((end[0] - start[0]) / segment_length, (end[1] - start[1]) / segment_length)
        if remaining <= segment_length:
            ratio = remaining / segment_length
            return (
                start[0] + (end[0] - start[0]) * ratio,
                start[1] + (end[1] - start[1]) * ratio,
            ), tangent
        remaining -= segment_length
    tangent = _normalize(_subtract(polyline[-1], polyline[-2]))
    return polyline[-1], tangent


def _build_plan_primitives(
    polyline: list[Point],
    corner_fraction: float,
) -> list[PlanPrimitive]:
    """以直线和相切圆弧平滑折线，同时保持首尾位置和方向。"""
    points = _remove_duplicate_points(polyline)
    if len(points) < 2:
        return []
    corners: dict[int, tuple[Point, Point, float, float, float]] = {}
    fraction = min(max(corner_fraction, 0.05), 0.9)
    for index in range(1, len(points) - 1):
        previous, corner, following = points[index - 1:index + 2]
        incoming_vector = _subtract(corner, previous)
        outgoing_vector = _subtract(following, corner)
        incoming_length = math.hypot(*incoming_vector)
        outgoing_length = math.hypot(*outgoing_vector)
        if incoming_length <= 1e-6 or outgoing_length <= 1e-6:
            continue
        incoming = _normalize(incoming_vector)
        outgoing = _normalize(outgoing_vector)
        dot = min(1.0, max(-1.0, incoming[0] * outgoing[0] + incoming[1] * outgoing[1]))
        turn_angle = math.acos(dot)
        cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
        if turn_angle < 1e-6 or turn_angle > math.radians(175.0) or abs(cross) < 1e-9:
            continue
        tangent_distance = min(incoming_length, outgoing_length) * fraction
        tangent_half = math.tan(turn_angle / 2.0)
        if tangent_half <= 1e-9:
            continue
        radius = tangent_distance / tangent_half
        if radius < 0.2:
            continue
        arc_start = (
            corner[0] - incoming[0] * tangent_distance,
            corner[1] - incoming[1] * tangent_distance,
        )
        arc_end = (
            corner[0] + outgoing[0] * tangent_distance,
            corner[1] + outgoing[1] * tangent_distance,
        )
        curvature = (1.0 if cross > 0.0 else -1.0) / radius
        corners[index] = (
            arc_start,
            arc_end,
            math.atan2(incoming[1], incoming[0]),
            radius * turn_angle,
            curvature,
        )

    primitives: list[PlanPrimitive] = []
    cursor = points[0]
    for index in range(1, len(points) - 1):
        if index not in corners:
            _append_line_primitive(primitives, cursor, points[index])
            cursor = points[index]
            continue
        arc_start, arc_end, heading, arc_length, curvature = corners[index]
        _append_line_primitive(primitives, cursor, arc_start)
        primitives.append(PlanPrimitive(
            kind="arc",
            start=arc_start,
            heading=heading,
            length=arc_length,
            curvature=curvature,
        ))
        cursor = arc_end
    _append_line_primitive(primitives, cursor, points[-1])
    return primitives


def _append_line_primitive(
    primitives: list[PlanPrimitive],
    start: Point,
    end: Point,
) -> None:
    """仅在长度有效时向参考线追加直线几何。"""
    length = _distance(start, end)
    if length <= 1e-6:
        return
    primitives.append(PlanPrimitive(
        kind="line",
        start=start,
        heading=math.atan2(end[1] - start[1], end[0] - start[0]),
        length=length,
    ))


def _sample_plan_primitives(
    primitives: list[PlanPrimitive],
    sample_spacing_m: float,
) -> list[Point]:
    """按固定最大间距采样OpenDRIVE几何，供FBX使用同一参考线。"""
    sampled: list[Point] = []
    spacing = max(0.2, sample_spacing_m)
    for primitive in primitives:
        steps = max(1, int(math.ceil(primitive.length / spacing)))
        if not sampled:
            sampled.append(primitive.start)
        for index in range(1, steps + 1):
            sampled.append(_primitive_point(
                primitive,
                primitive.length * index / steps,
            ))
    return _remove_duplicate_points(sampled)


def _primitive_point(primitive: PlanPrimitive, distance: float) -> Point:
    """计算直线或圆弧上距起点指定弧长的位置。"""
    distance = min(max(0.0, distance), primitive.length)
    if primitive.kind != "arc" or abs(primitive.curvature) <= 1e-12:
        return (
            primitive.start[0] + math.cos(primitive.heading) * distance,
            primitive.start[1] + math.sin(primitive.heading) * distance,
        )
    end_heading = primitive.heading + primitive.curvature * distance
    return (
        primitive.start[0]
        + (math.sin(end_heading) - math.sin(primitive.heading)) / primitive.curvature,
        primitive.start[1]
        + (-math.cos(end_heading) + math.cos(primitive.heading)) / primitive.curvature,
    )


def _plan_length(primitives: list[PlanPrimitive]) -> float:
    """累计OpenDRIVE参考线中全部几何段长度。"""
    return sum(primitive.length for primitive in primitives)


def _remove_duplicate_points(polyline: list[Point]) -> list[Point]:
    """删除连续重复或近似重复的二维点。"""
    cleaned: list[Point] = []
    for point in polyline:
        candidate = (float(point[0]), float(point[1]))
        if not cleaned or _distance(cleaned[-1], candidate) > 1e-8:
            cleaned.append(candidate)
    return cleaned


def _subtract(first: Point, second: Point) -> Point:
    """返回两个二维点之差。"""
    return (first[0] - second[0], first[1] - second[1])


def _primary_surface(spec: RealMapTraceSpec) -> str:
    """选择第一条道路的材质作为当前单材质 FBX 的主表面。"""
    return spec.roads[0].surface if spec.roads else "dirt"


def _polyline_length(polyline: list[Point]) -> float:
    """累计折线各相邻点之间的长度。"""
    return sum(_distance(a, b) for a, b in zip(polyline, polyline[1:]))


def _distance(a: Point, b: Point) -> float:
    """计算两个局部平面点的欧氏距离。"""
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _normalize(vector: Point) -> Point:
    """返回二维单位向量；零长度时使用 x 正方向作为退化值。"""
    length = math.hypot(vector[0], vector[1])
    if length <= 1e-9:
        return (1.0, 0.0)
    return (vector[0] / length, vector[1] / length)


def _fmt(value: float) -> str:
    """生成简洁且稳定的 XML 浮点数字符串。"""
    if abs(value) < 1e-9:
        value = 0.0
    return f"{value:.12f}".rstrip("0").rstrip(".")


def _prettify_xml(root: ET.Element) -> str:
    """将 OpenDRIVE XML 格式化为带缩进的 UTF-8 文本。"""
    rough = ET.tostring(root, encoding="utf-8")
    parsed = minidom.parseString(rough)
    return parsed.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")
