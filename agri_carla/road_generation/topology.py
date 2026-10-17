"""农业道路规格与 OpenDRIVE 文件的拓扑检查和有限自动修复。"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET


@dataclass(frozen=True)
class TopologyValidationConfig:
    """控制拓扑检查严格程度、自动修复范围和最大再生成次数。"""
    enabled: bool = True
    auto_repair: bool = True
    max_regeneration_attempts: int = 2
    endpoint_tolerance_m: float = 0.5
    max_endpoint_snap_m: float = 20.0
    merge_node_tolerance_m: float = 0.75
    bridge_max_gap_m: float = 8.0
    geometry_tolerance_m: float = 0.05
    heading_tolerance_rad: float = 0.05
    road_length_tolerance_m: float = 0.05
    lane_center_tolerance_m: float = 0.15
    intersection_tolerance_m: float = 0.10
    require_single_component: bool = True
    require_bidirectional_roads: bool = True
    require_complete_junction_movements: bool = True
    fail_on_unresolved: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TopologyValidationConfig":
        """从请求中读取已支持的拓扑参数，忽略解释性扩展字段。"""
        if not data:
            return cls()
        valid = {field.name for field in cls.__dataclass_fields__.values()}
        values = {key: value for key, value in data.items() if key in valid}
        return cls(**values)


@dataclass(frozen=True)
class TopologyIssue:
    """一项具有稳定代码、严重程度和详细信息的拓扑问题。"""
    severity: str
    code: str
    message: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转换为拓扑报告可写入的字典。"""
        return asdict(self)


@dataclass(frozen=True)
class TopologyCheckResult:
    """一次检查的有效性、问题列表和连通分量统计。"""
    valid: bool
    issues: list[TopologyIssue]
    component_count: int
    components: list[list[str]]

    def to_dict(self) -> dict[str, Any]:
        """序列化检查结果。"""
        return {
            "valid": self.valid,
            "component_count": self.component_count,
            "components": self.components,
            "issues": [issue.to_dict() for issue in self.issues],
        }


class RoadTopologyValidator:
    """检查并有限修复农业道路的几何端点和逻辑连通关系。"""

    def __init__(self, config: TopologyValidationConfig | None = None):
        """保存拓扑容差和自动修复策略。"""
        self.config = config or TopologyValidationConfig()

    def validate_spec(self, spec) -> TopologyCheckResult:
        """在生成 XODR 前检查节点、道路端点、重复点和路网连通性。"""
        issues: list[TopologyIssue] = []
        node_ids = [node.node_id for node in spec.nodes]
        duplicate_nodes = _duplicates(node_ids)
        if duplicate_nodes:
            issues.append(_error(
                "DUPLICATE_NODE_ID",
                "Node identifiers must be unique.",
                node_ids=duplicate_nodes,
            ))

        road_ids = [road.road_id for road in spec.roads]
        duplicate_roads = _duplicates(road_ids)
        if duplicate_roads:
            issues.append(_error(
                "DUPLICATE_ROAD_ID",
                "Road identifiers must be unique.",
                road_ids=duplicate_roads,
            ))

        if not spec.roads:
            issues.append(_error("NO_ROADS", "The road specification contains no roads."))

        node_map = {node.node_id: node for node in spec.nodes}
        referenced_nodes: set[str] = set()
        graph_edges: list[tuple[str, str]] = []

        for road in spec.roads:
            missing = [
                node_id
                for node_id in (road.start_node, road.end_node)
                if node_id not in node_map
            ]
            if missing:
                issues.append(_error(
                    "UNKNOWN_ROAD_NODE",
                    f"Road {road.road_id} refers to undefined nodes.",
                    road_id=road.road_id,
                    node_ids=missing,
                ))
                continue

            referenced_nodes.update((road.start_node, road.end_node))
            graph_edges.append((road.start_node, road.end_node))
            if len(road.pixel_polyline) < 2:
                issues.append(_error(
                    "INVALID_POLYLINE",
                    f"Road {road.road_id} requires at least two polyline points.",
                    road_id=road.road_id,
                ))
                continue

            start_gap = _pixel_distance(
                road.pixel_polyline[0],
                node_map[road.start_node].pixel,
                spec.meters_per_pixel,
            )
            end_gap = _pixel_distance(
                road.pixel_polyline[-1],
                node_map[road.end_node].pixel,
                spec.meters_per_pixel,
            )
            if start_gap > self.config.endpoint_tolerance_m:
                issues.append(_error(
                    "ENDPOINT_GAP",
                    f"Road {road.road_id} does not reach its declared start node.",
                    road_id=road.road_id,
                    endpoint="start",
                    node_id=road.start_node,
                    gap_m=start_gap,
                ))
            if end_gap > self.config.endpoint_tolerance_m:
                issues.append(_error(
                    "ENDPOINT_GAP",
                    f"Road {road.road_id} does not reach its declared end node.",
                    road_id=road.road_id,
                    endpoint="end",
                    node_id=road.end_node,
                    gap_m=end_gap,
                ))

            zero_segments = [
                index
                for index, (start, end) in enumerate(
                    zip(road.pixel_polyline, road.pixel_polyline[1:])
                )
                if _pixel_distance(start, end, spec.meters_per_pixel) <= 1e-6
            ]
            if zero_segments:
                issues.append(_error(
                    "ZERO_LENGTH_SEGMENT",
                    f"Road {road.road_id} contains duplicate consecutive points.",
                    road_id=road.road_id,
                    segment_indices=zero_segments,
                ))

        isolated = sorted(set(node_ids) - referenced_nodes)
        if isolated:
            issues.append(_warning(
                "ISOLATED_NODE",
                "Some nodes are not referenced by any road.",
                node_ids=isolated,
            ))

        issues.extend(_road_intersection_issues(spec, self.config))

        components = _graph_components(referenced_nodes, graph_edges)
        if self.config.require_single_component and len(components) > 1:
            issues.append(_error(
                "DISCONNECTED_COMPONENTS",
                "The agricultural road network contains disconnected components.",
                components=components,
            ))

        return TopologyCheckResult(
            valid=not any(issue.severity == "error" for issue in issues),
            issues=issues,
            component_count=len(components),
            components=components,
        )

    def validate_xodr(self, xodr_path: str | Path) -> TopologyCheckResult:
        """解析生成后的 XODR，检查道路、路口、connection 和 laneLink。"""
        issues: list[TopologyIssue] = []
        try:
            root = ET.parse(xodr_path).getroot()
        except (ET.ParseError, OSError) as exc:
            return TopologyCheckResult(
                valid=False,
                issues=[_error("XODR_PARSE_ERROR", "OpenDRIVE could not be parsed.", error=str(exc))],
                component_count=0,
                components=[],
            )

        road_nodes = root.findall("road")
        road_map = {road.get("id", ""): road for road in road_nodes}
        road_ids = [road.get("id", "") for road in road_nodes]
        road_id_set = set(road_ids)
        duplicate_ids = _duplicates(road_ids)
        if duplicate_ids:
            issues.append(_error(
                "XODR_DUPLICATE_ROAD_ID",
                "OpenDRIVE road identifiers must be unique.",
                road_ids=duplicate_ids,
            ))

        junction_nodes = root.findall("junction")
        junction_ids = {junction.get("id", "") for junction in junction_nodes}
        duplicate_junction_ids = _duplicates(j.get("id", "") for j in junction_nodes)
        if duplicate_junction_ids:
            issues.append(_error(
                "XODR_DUPLICATE_JUNCTION_ID",
                "OpenDRIVE junction identifiers must be unique.",
                junction_ids=duplicate_junction_ids,
            ))
        # CARLA 0.9.15 resolves a linked integer ID as a road first, even if
        # elementType says junction. This is a CARLA compatibility constraint.
        try:
            road_numbers = {int(value) for value in road_ids}
            junction_numbers = {int(value) for value in junction_ids}
            collisions = sorted(road_numbers & junction_numbers)
            if collisions:
                issues.append(_error(
                    "XODR_CARLA_ROAD_JUNCTION_ID_COLLISION",
                    "CARLA 0.9.15 treats a junction link as a road when their IDs overlap.",
                    colliding_ids=collisions,
                ))
        except ValueError:
            issues.append(_error(
                "XODR_CARLA_NON_NUMERIC_ID",
                "CARLA 0.9.15 requires numeric road and junction identifiers.",
            ))
        graph_edges: list[tuple[str, str]] = []
        lane_ids = {
            road.get("id", ""): {
                lane.get("id", "")
                for lane in road.findall("./lanes/laneSection/*/lane")
            }
            for road in road_nodes
        }
        road_geometry: dict[str, dict[str, Any]] = {}

        for road in road_nodes:
            road_id = road.get("id", "")
            try:
                length = float(road.get("length", "0"))
            except ValueError:
                length = 0.0
            if length <= 0.0 or not road.findall("./planView/geometry"):
                issues.append(_error(
                    "XODR_INVALID_ROAD_GEOMETRY",
                    f"OpenDRIVE road {road_id} has no valid geometry.",
                    road_id=road_id,
                ))
            geometry_result, geometry_issues = _xodr_geometry_summary(
                road,
                self.config,
            )
            road_geometry[road_id] = geometry_result
            issues.extend(geometry_issues)

            if (
                self.config.require_bidirectional_roads
                and road.get("junction", "-1") == "-1"
            ):
                driving_lane_ids = {
                    int(lane.get("id", "0"))
                    for lane in road.findall("./lanes/laneSection/*/lane")
                    if lane.get("type") == "driving"
                }
                if not any(lane_id > 0 for lane_id in driving_lane_ids):
                    issues.append(_error(
                        "XODR_MISSING_REVERSE_DRIVING_LANE",
                        "A bidirectional agricultural road requires a positive-id driving lane.",
                        road_id=road_id,
                        driving_lane_ids=sorted(driving_lane_ids),
                    ))
                if not any(lane_id < 0 for lane_id in driving_lane_ids):
                    issues.append(_error(
                        "XODR_MISSING_FORWARD_DRIVING_LANE",
                        "A bidirectional agricultural road requires a negative-id driving lane.",
                        road_id=road_id,
                        driving_lane_ids=sorted(driving_lane_ids),
                    ))

            for relation in ("predecessor", "successor"):
                link = road.find(f"./link/{relation}")
                if link is None:
                    continue
                element_type = link.get("elementType")
                element_id = link.get("elementId", "")
                if element_type == "road":
                    if element_id not in road_id_set:
                        issues.append(_error(
                            "XODR_BROKEN_ROAD_LINK",
                            f"Road {road_id} links to an undefined road.",
                            road_id=road_id,
                            linked_road_id=element_id,
                        ))
                    else:
                        graph_edges.append((road_id, element_id))
                elif element_type == "junction":
                    if element_id not in junction_ids:
                        issues.append(_error(
                            "XODR_BROKEN_JUNCTION_LINK",
                            f"Road {road_id} links to an undefined junction.",
                            road_id=road_id,
                            junction_id=element_id,
                        ))
                else:
                    issues.append(_error(
                        "XODR_UNKNOWN_LINK_TYPE",
                        f"Road {road_id} contains an unsupported link type.",
                        road_id=road_id,
                        element_type=element_type,
                    ))

        issues.extend(_validate_direct_road_links(road_map, road_geometry, self.config))

        for junction in junction_nodes:
            junction_id = junction.get("id", "")
            connections = junction.findall("connection")
            actual_movements: set[tuple[str, str]] = set()
            duplicate_connection_ids = _duplicates(
                connection.get("id", "") for connection in connections
            )
            if duplicate_connection_ids:
                issues.append(_error(
                    "XODR_DUPLICATE_CONNECTION_ID",
                    "Junction connection identifiers must be unique within a junction.",
                    junction_id=junction_id,
                    connection_ids=duplicate_connection_ids,
                ))
            if not connections:
                issues.append(_error(
                    "XODR_EMPTY_JUNCTION",
                    f"Junction {junction_id} has no connections.",
                    junction_id=junction_id,
                ))
            for connection in connections:
                incoming = connection.get("incomingRoad", "")
                connecting = connection.get("connectingRoad", "")
                if incoming not in road_id_set or connecting not in road_id_set:
                    issues.append(_error(
                        "XODR_BROKEN_CONNECTION",
                        f"Junction {junction_id} refers to an undefined road.",
                        junction_id=junction_id,
                        incoming_road=incoming,
                        connecting_road=connecting,
                    ))
                    continue
                if road_map[connecting].get("junction", "-1") != junction_id:
                    issues.append(_error(
                        "XODR_CONNECTING_ROAD_WRONG_JUNCTION",
                        "A connecting road must belong to the junction that references it.",
                        junction_id=junction_id,
                        connecting_road=connecting,
                        road_junction=road_map[connecting].get("junction", "-1"),
                    ))
                graph_edges.append((incoming, connecting))
                lane_links = connection.findall("laneLink")
                if not lane_links:
                    issues.append(_error(
                        "XODR_MISSING_LANE_LINK",
                        f"Junction connection in {junction_id} has no laneLink.",
                        junction_id=junction_id,
                        incoming_road=incoming,
                        connecting_road=connecting,
                    ))
                for lane_link in lane_links:
                    from_lane = lane_link.get("from", "")
                    to_lane = lane_link.get("to", "")
                    if from_lane not in lane_ids.get(incoming, set()):
                        issues.append(_error(
                            "XODR_INVALID_FROM_LANE",
                            "A junction laneLink refers to an undefined incoming lane.",
                            road_id=incoming,
                            lane_id=from_lane,
                        ))
                    if to_lane not in lane_ids.get(connecting, set()):
                        issues.append(_error(
                            "XODR_INVALID_TO_LANE",
                            "A junction laneLink refers to an undefined connecting lane.",
                            road_id=connecting,
                            lane_id=to_lane,
                        ))

                    incoming_side = _road_junction_contact_side(
                        road_map.get(incoming),
                        junction_id,
                    )
                    contact_point = connection.get("contactPoint", "")
                    if from_lane in lane_ids.get(incoming, set()) and to_lane in lane_ids.get(connecting, set()):
                        valid_incoming = (
                            (incoming_side == "start" and int(from_lane) > 0)
                            or (incoming_side == "end" and int(from_lane) < 0)
                        )
                        valid_connector = (
                            (contact_point == "start" and int(to_lane) < 0)
                            or (contact_point == "end" and int(to_lane) > 0)
                        )
                        if not valid_incoming or not valid_connector:
                            issues.append(_error(
                                "XODR_LANE_LINK_TRAVEL_DIRECTION",
                                "Junction contact points must match the driving-lane travel direction.",
                                incoming_road=incoming, from_lane=from_lane,
                                connecting_road=connecting, to_lane=to_lane,
                            ))
                        relation = "predecessor" if contact_point == "start" else "successor"
                        entry_road_link = road_map[connecting].find(f"./link/{relation}")
                        entry_lane = next((lane for lane in road_map[connecting].findall(
                            "./lanes/laneSection/*/lane") if lane.get("id") == to_lane), None)
                        entry_lane_link = entry_lane.find(f"./link/{relation}") if entry_lane is not None else None
                        if (entry_road_link is None or entry_lane_link is None
                                or entry_road_link.get("elementId") != incoming
                                or entry_road_link.get("contactPoint") != incoming_side
                                or entry_lane_link.get("id") != from_lane):
                            issues.append(_error(
                                "XODR_CONNECTOR_ENTRY_LINK_MISMATCH",
                                "Connector road/lane links disagree with the junction entry.",
                                incoming_road=incoming, connecting_road=connecting,
                            ))
                    if (
                        incoming_side in {"start", "end"}
                        and contact_point in {"start", "end"}
                        and from_lane in lane_ids.get(incoming, set())
                        and to_lane in lane_ids.get(connecting, set())
                        and _lane_type_at_contact(
                            road_map[incoming],
                            int(from_lane),
                            incoming_side,
                        ) == "driving"
                        and _lane_type_at_contact(
                            road_map[connecting],
                            int(to_lane),
                            contact_point,
                        ) == "driving"
                    ):
                        incoming_center = _lane_center_at_contact(
                            road_map[incoming],
                            int(from_lane),
                            incoming_side,
                            road_geometry[incoming],
                        )
                        connecting_center = _lane_center_at_contact(
                            road_map[connecting],
                            int(to_lane),
                            contact_point,
                            road_geometry[connecting],
                        )
                        if incoming_center is not None and connecting_center is not None:
                            center_gap = math.dist(incoming_center, connecting_center)
                            if center_gap > self.config.lane_center_tolerance_m:
                                issues.append(_error(
                                    "XODR_LANE_LINK_CENTER_GAP",
                                    "Linked driving-lane centerlines are not spatially continuous.",
                                    junction_id=junction_id,
                                    incoming_road=incoming,
                                    connecting_road=connecting,
                                    from_lane=from_lane,
                                    to_lane=to_lane,
                                    gap_m=center_gap,
                                ))
                        incoming_heading = _lane_travel_heading_at_contact(
                            road_geometry[incoming],
                            int(from_lane),
                            incoming_side,
                        )
                        connecting_heading = _lane_travel_heading_at_contact(
                            road_geometry[connecting],
                            int(to_lane),
                            contact_point,
                        )
                        if incoming_heading is not None and connecting_heading is not None:
                            heading_gap = abs(_wrap_angle(connecting_heading - incoming_heading))
                            if heading_gap > self.config.heading_tolerance_rad:
                                issues.append(_error(
                                    "XODR_LANE_LINK_HEADING_GAP",
                                    "Linked driving lanes do not share a continuous travel direction.",
                                    junction_id=junction_id,
                                    incoming_road=incoming,
                                    connecting_road=connecting,
                                    from_lane=from_lane,
                                    to_lane=to_lane,
                                    heading_gap_rad=heading_gap,
                                ))

                    movement = _junction_movement(
                        road_map,
                        connecting,
                        contact_point,
                        to_lane,
                    )
                    if movement is not None:
                        outgoing_road_id, outgoing_lane_id, connector_exit_side, outgoing_side = movement
                        if not ((outgoing_side == "start" and outgoing_lane_id < 0)
                                or (outgoing_side == "end" and outgoing_lane_id > 0)):
                            issues.append(_error(
                                "XODR_CONNECTOR_EXIT_TRAVEL_DIRECTION",
                                "A junction connector enters an outgoing lane against its travel direction.",
                                connecting_road=connecting, outgoing_road=outgoing_road_id,
                            ))
                        actual_movements.add((incoming, outgoing_road_id))
                        if (
                            outgoing_road_id in road_geometry
                            and to_lane in lane_ids.get(connecting, set())
                            and str(outgoing_lane_id) in lane_ids.get(outgoing_road_id, set())
                            and _lane_type_at_contact(
                                road_map[connecting],
                                int(to_lane),
                                connector_exit_side,
                            ) == "driving"
                            and _lane_type_at_contact(
                                road_map[outgoing_road_id],
                                outgoing_lane_id,
                                outgoing_side,
                            ) == "driving"
                        ):
                            connecting_exit_center = _lane_center_at_contact(
                                road_map[connecting],
                                int(to_lane),
                                connector_exit_side,
                                road_geometry[connecting],
                            )
                            outgoing_center = _lane_center_at_contact(
                                road_map[outgoing_road_id],
                                outgoing_lane_id,
                                outgoing_side,
                                road_geometry[outgoing_road_id],
                            )
                            if connecting_exit_center is not None and outgoing_center is not None:
                                exit_gap = math.dist(connecting_exit_center, outgoing_center)
                                if exit_gap > self.config.lane_center_tolerance_m:
                                    issues.append(_error(
                                        "XODR_LANE_LINK_EXIT_CENTER_GAP",
                                        "A junction connector does not reach the outgoing driving-lane centerline.",
                                        junction_id=junction_id,
                                        connecting_road=connecting,
                                        outgoing_road=outgoing_road_id,
                                        connector_lane=to_lane,
                                        outgoing_lane=outgoing_lane_id,
                                        gap_m=exit_gap,
                                    ))
                            connector_exit_heading = _lane_travel_heading_at_contact(
                                road_geometry[connecting],
                                int(to_lane),
                                connector_exit_side,
                            )
                            outgoing_heading = _lane_travel_heading_at_contact(
                                road_geometry[outgoing_road_id],
                                outgoing_lane_id,
                                outgoing_side,
                            )
                            if connector_exit_heading is not None and outgoing_heading is not None:
                                exit_heading_gap = abs(_wrap_angle(connector_exit_heading - outgoing_heading))
                                if exit_heading_gap > self.config.heading_tolerance_rad:
                                    issues.append(_error(
                                        "XODR_LANE_LINK_EXIT_HEADING_GAP",
                                        "A junction connector and its outgoing lane do not share a continuous travel direction.",
                                        junction_id=junction_id,
                                        connecting_road=connecting,
                                        outgoing_road=outgoing_road_id,
                                        heading_gap_rad=exit_heading_gap,
                                    ))

                    else:
                        issues.append(_error(
                            "XODR_CONNECTOR_MISSING_EXIT_LINK",
                            "A junction connector must link to an existing outgoing road and lane.",
                            connecting_road=connecting,
                        ))

                connecting_geometry = road_geometry.get(connecting, {})
                contact_point = connection.get("contactPoint", "")
                if contact_point not in {"start", "end"}:
                    issues.append(_error(
                        "XODR_INVALID_CONNECTION_CONTACT_POINT",
                        "A junction connection must use start or end as contactPoint.",
                        junction_id=junction_id,
                        connecting_road=connecting,
                        contact_point=contact_point,
                    ))
                elif connecting_geometry:
                    expected_side = connecting_geometry.get(contact_point)
                    incoming_endpoint = _road_junction_endpoint(
                        road_nodes,
                        incoming,
                        junction_id,
                        road_geometry,
                    )
                    if expected_side is not None and incoming_endpoint is not None:
                        gap = math.dist(expected_side, incoming_endpoint)
                        if gap > self.config.geometry_tolerance_m:
                            issues.append(_warning(
                                "XODR_JUNCTION_GEOMETRY_GAP",
                                "Junction reference lines do not coincide; lane-level links remain authoritative.",
                                junction_id=junction_id,
                                incoming_road=incoming,
                                connecting_road=connecting,
                                gap_m=gap,
                            ))

            if self.config.require_complete_junction_movements:
                incident_roads = sorted(
                    road_id
                    for road_id, road in road_map.items()
                    if (
                        road.get("junction", "-1") == "-1"
                        and _road_junction_contact_side(road, junction_id) is not None
                    )
                )
                expected_movements = {
                    (incoming_road, outgoing_road)
                    for incoming_road in incident_roads
                    for outgoing_road in incident_roads
                    if incoming_road != outgoing_road
                }
                missing_movements = sorted(expected_movements - actual_movements)
                if missing_movements:
                    issues.append(_error(
                        "XODR_MISSING_JUNCTION_MOVEMENT",
                        "The junction does not provide every directed movement between its incident roads.",
                        junction_id=junction_id,
                        incident_road_ids=incident_roads,
                        expected_movement_count=len(expected_movements),
                        actual_movement_count=len(expected_movements & actual_movements),
                        missing_movements=[
                            {"incoming_road": incoming, "outgoing_road": outgoing}
                            for incoming, outgoing in missing_movements
                        ],
                    ))

        # connecting road 的 predecessor/successor 还必须分别落在所声明主路
        # contactPoint 上；这一检查可发现XML关系正确但道路空间分离的问题。
        for road in road_nodes:
            if road.get("junction", "-1") == "-1":
                continue
            road_id = road.get("id", "")
            geometry = road_geometry.get(road_id, {})
            for relation, connector_side in (("predecessor", "start"), ("successor", "end")):
                link = road.find(f"./link/{relation}")
                if link is None or link.get("elementType") != "road":
                    continue
                linked_id = link.get("elementId", "")
                linked_side = link.get("contactPoint", "")
                linked_geometry = road_geometry.get(linked_id, {})
                if linked_side not in {"start", "end"}:
                    continue
                source_point = geometry.get(connector_side)
                target_point = linked_geometry.get(linked_side)
                if source_point is None or target_point is None:
                    continue
                gap = math.dist(source_point, target_point)
                if gap > self.config.geometry_tolerance_m:
                    issues.append(_warning(
                        "XODR_ROAD_LINK_GEOMETRY_GAP",
                        "Linked road reference lines do not coincide; lane offsets may account for the displacement.",
                        road_id=road_id,
                        relation=relation,
                        linked_road_id=linked_id,
                        gap_m=gap,
                    ))

        components = _graph_components(road_id_set, graph_edges)
        if self.config.require_single_component and len(components) > 1:
            issues.append(_error(
                "XODR_DISCONNECTED_ROADS",
                "OpenDRIVE contains roads that are not connected through road or junction links.",
                components=components,
            ))

        return TopologyCheckResult(
            valid=not any(issue.severity == "error" for issue in issues),
            issues=issues,
            component_count=len(components),
            components=components,
        )

    def repair_spec(self, spec) -> tuple[Any, list[dict[str, Any]]]:
        """删除连续重复点、吸附可修复端点并尝试连接近距离分量。

        修复仅在配置阈值内进行；距离过大的断路不会被武断连接，而会继续作为
        未解决错误写入报告。
        """
        from roadrunner_mcp.realmap import TraceRoad

        actions: list[dict[str, Any]] = []
        nodes = list(spec.nodes)
        node_map = {node.node_id: node for node in nodes}
        repaired_roads = []

        for road in spec.roads:
            points = _remove_consecutive_duplicates(road.pixel_polyline)
            if len(points) != len(road.pixel_polyline):
                actions.append({
                    "action": "remove_duplicate_polyline_points",
                    "road_id": road.road_id,
                    "removed_count": len(road.pixel_polyline) - len(points),
                })

            if len(points) >= 2 and road.start_node in node_map:
                gap = _pixel_distance(points[0], node_map[road.start_node].pixel, spec.meters_per_pixel)
                if (
                    gap > self.config.endpoint_tolerance_m
                    and gap <= self.config.max_endpoint_snap_m
                ):
                    points[0] = node_map[road.start_node].pixel
                    actions.append({
                        "action": "snap_road_endpoint",
                        "road_id": road.road_id,
                        "endpoint": "start",
                        "node_id": road.start_node,
                        "original_gap_m": gap,
                    })

            if len(points) >= 2 and road.end_node in node_map:
                gap = _pixel_distance(points[-1], node_map[road.end_node].pixel, spec.meters_per_pixel)
                if (
                    gap > self.config.endpoint_tolerance_m
                    and gap <= self.config.max_endpoint_snap_m
                ):
                    points[-1] = node_map[road.end_node].pixel
                    actions.append({
                        "action": "snap_road_endpoint",
                        "road_id": road.road_id,
                        "endpoint": "end",
                        "node_id": road.end_node,
                        "original_gap_m": gap,
                    })

            repaired_roads.append(replace(road, pixel_polyline=points))

        repaired_spec = replace(spec, nodes=nodes, roads=repaired_roads)
        repaired_spec, connection_actions = self._connect_components(
            repaired_spec,
            TraceRoad,
        )
        actions.extend(connection_actions)
        return repaired_spec, actions

    def _connect_components(self, spec, trace_road_cls):
        """通过合并近邻节点或增加短桥接道路逐步减少断开分量。"""
        actions: list[dict[str, Any]] = []
        current_spec = spec

        while True:
            node_ids = {node.node_id for node in current_spec.nodes}
            edges = [
                (road.start_node, road.end_node)
                for road in current_spec.roads
                if road.start_node in node_ids and road.end_node in node_ids
            ]
            components = _graph_components(
                {node_id for edge in edges for node_id in edge},
                edges,
            )
            if len(components) <= 1:
                break

            node_map = {node.node_id: node for node in current_spec.nodes}
            nearest = _nearest_component_nodes(
                components,
                node_map,
                current_spec.meters_per_pixel,
            )
            if nearest is None:
                break
            first_id, second_id, gap_m = nearest
            if gap_m > self.config.bridge_max_gap_m:
                break

            if gap_m <= self.config.merge_node_tolerance_m:
                current_spec = _merge_nodes(
                    current_spec,
                    keep_node_id=first_id,
                    remove_node_id=second_id,
                )
                actions.append({
                    "action": "merge_close_nodes",
                    "kept_node_id": first_id,
                    "removed_node_id": second_id,
                    "gap_m": gap_m,
                })
                continue

            road_id = _unique_bridge_id(current_spec.roads)
            first = node_map[first_id]
            second = node_map[second_id]
            width, surface = _bridge_style(current_spec.roads, first_id, second_id)
            bridge = trace_road_cls(
                road_id=road_id,
                start_node=first_id,
                end_node=second_id,
                pixel_polyline=[first.pixel, second.pixel],
                width_m=width,
                surface=surface,
            )
            current_spec = replace(current_spec, roads=[*current_spec.roads, bridge])
            actions.append({
                "action": "add_bridge_road",
                "road_id": road_id,
                "start_node": first_id,
                "end_node": second_id,
                "gap_m": gap_m,
                "width_m": width,
                "surface": surface,
            })

        return current_spec, actions


def _validate_direct_road_links(roads, geometries, config):
    """Check both lane directions across ordinary road splits, outside junctions."""
    issues = []
    for road_id, road in roads.items():
        if road.get("junction", "-1") != "-1":
            continue
        for relation, side in (("predecessor", "start"), ("successor", "end")):
            link = road.find(f"./link/{relation}")
            if link is None or link.get("elementType") != "road":
                continue
            target_id = link.get("elementId")
            target = roads.get(target_id)
            if target is None or target.get("junction", "-1") != "-1":
                continue
            target_side = link.get("contactPoint")
            if target_side not in {"start", "end"}:
                issues.append(_error("XODR_DIRECT_ROAD_CONTACT", "A direct road link needs a valid contactPoint.", road_id=road_id))
                continue
            opposite_relation = "predecessor" if target_side == "start" else "successor"
            back = target.find(f"./link/{opposite_relation}")
            if (back is None or back.get("elementType") != "road"
                    or back.get("elementId") != road_id or back.get("contactPoint") != side):
                issues.append(_error("XODR_DIRECT_ROAD_RECIPROCAL_LINK", "Direct road links must be reciprocal.", road_id=road_id, target_road=target_id))
            sections = road.findall("./lanes/laneSection")
            section = sections[0 if side == "start" else -1] if sections else None
            if section is None:
                continue
            target_sections = target.findall("./lanes/laneSection")
            target_section = target_sections[0 if target_side == "start" else -1] if target_sections else None
            for lane in section.findall("./*/lane"):
                if lane.get("type") != "driving":
                    continue
                lane_id = int(lane.get("id"))
                lane_link = lane.find(f"./link/{relation}")
                if lane_link is None:
                    issues.append(_error("XODR_DIRECT_LANE_LINK_MISSING", "Direct road splits need explicit lane links in both directions.", road_id=road_id, lane_id=lane_id))
                    continue
                target_lane_id = int(lane_link.get("id"))
                target_lane = next((item for item in target_section.findall("./*/lane")
                                    if int(item.get("id")) == target_lane_id), None) if target_section is not None else None
                if target_lane is None or target_lane.get("type") != "driving":
                    issues.append(_error("XODR_DIRECT_LANE_TARGET", "Direct lane link must reach an existing driving lane.", road_id=road_id, lane_id=lane_id))
                    continue
                back_lane = target_lane.find(f"./link/{opposite_relation}")
                if back_lane is None or int(back_lane.get("id")) != lane_id:
                    issues.append(_error("XODR_DIRECT_LANE_RECIPROCAL_LINK", "Direct lane links must be reciprocal.", road_id=road_id, lane_id=lane_id))
                source_exits = (lane_id < 0 and side == "end") or (lane_id > 0 and side == "start")
                target_exits = (target_lane_id < 0 and target_side == "end") or (target_lane_id > 0 and target_side == "start")
                if source_exits == target_exits:
                    issues.append(_error("XODR_DIRECT_LANE_DIRECTION", "A direct lane link must connect an exit and an entry.", road_id=road_id, lane_id=lane_id))
                a = _lane_center_at_contact(road, lane_id, side, geometries[road_id])
                b = _lane_center_at_contact(target, target_lane_id, target_side, geometries[target_id])
                if a is not None and b is not None and math.dist(a, b) > config.lane_center_tolerance_m:
                    issues.append(_error("XODR_DIRECT_LANE_CENTER_GAP", "Directly linked lane centers do not meet.", road_id=road_id, target_road=target_id, gap_m=math.dist(a, b)))
                ah = _lane_travel_heading_at_contact(geometries[road_id], lane_id, side)
                bh = _lane_travel_heading_at_contact(geometries[target_id], target_lane_id, target_side)
                if ah is not None and bh is not None and abs(_wrap_angle(ah - bh)) > config.heading_tolerance_rad:
                    issues.append(_error("XODR_DIRECT_LANE_HEADING_GAP", "Directly linked lane travel headings do not match.", road_id=road_id, target_road=target_id))
    return issues


def _xodr_geometry_summary(
    road: ET.Element,
    config: TopologyValidationConfig,
) -> tuple[dict[str, Any], list[TopologyIssue]]:
    """计算道路参考线首尾位置，并检查几何段的位置、方向和里程连续性。"""
    geometries = road.findall("./planView/geometry")
    if not geometries:
        return {}, []
    road_id = road.get("id", "")
    issues: list[TopologyIssue] = []
    first = geometries[0]
    start_point = (float(first.get("x", "0")), float(first.get("y", "0")))
    previous_end: tuple[float, float] | None = None
    previous_heading: float | None = None
    cumulative_s = 0.0
    end_point = start_point
    end_heading = float(first.get("hdg", "0"))

    for index, geometry in enumerate(geometries):
        geometry_s = float(geometry.get("s", "0"))
        x = float(geometry.get("x", "0"))
        y = float(geometry.get("y", "0"))
        heading = float(geometry.get("hdg", "0"))
        length = float(geometry.get("length", "0"))
        if abs(geometry_s - cumulative_s) > config.road_length_tolerance_m:
            issues.append(_error(
                "XODR_GEOMETRY_S_DISCONTINUITY",
                "OpenDRIVE geometry s values are not continuous.",
                road_id=road_id,
                geometry_index=index,
                expected_s=cumulative_s,
                actual_s=geometry_s,
            ))
        if previous_end is not None:
            gap = math.dist(previous_end, (x, y))
            if gap > config.geometry_tolerance_m:
                issues.append(_error(
                    "XODR_PLANVIEW_POSITION_GAP",
                    "Consecutive OpenDRIVE geometries do not share an endpoint.",
                    road_id=road_id,
                    geometry_index=index,
                    gap_m=gap,
                ))
        if previous_heading is not None:
            heading_gap = abs(_wrap_angle(heading - previous_heading))
            if heading_gap > config.heading_tolerance_rad:
                issues.append(_error(
                    "XODR_PLANVIEW_HEADING_GAP",
                    "Consecutive OpenDRIVE geometries are not tangent-continuous.",
                    road_id=road_id,
                    geometry_index=index,
                    heading_gap_rad=heading_gap,
                ))

        child = next(iter(geometry), None)
        if child is None or child.tag == "line":
            end_point = (
                x + math.cos(heading) * length,
                y + math.sin(heading) * length,
            )
            end_heading = heading
        elif child.tag == "arc":
            curvature = float(child.get("curvature", "0"))
            if abs(curvature) <= 1e-12:
                end_point = (
                    x + math.cos(heading) * length,
                    y + math.sin(heading) * length,
                )
                end_heading = heading
            else:
                end_heading = heading + curvature * length
                end_point = (
                    x + (math.sin(end_heading) - math.sin(heading)) / curvature,
                    y + (-math.cos(end_heading) + math.cos(heading)) / curvature,
                )
        elif child.tag == "spiral":
            curvature_start = float(child.get("curvStart", "0"))
            curvature_end = float(child.get("curvEnd", "0"))
            end_point, end_heading = _spiral_endpoint(
                (x, y),
                heading,
                length,
                curvature_start,
                curvature_end,
            )
        else:
            # 当前生成器只写入line和arc；对其他第三方几何保留警告，
            # 并利用下一段起点近似当前段终点以继续检查后续关系。
            issues.append(_warning(
                "XODR_GEOMETRY_TYPE_NOT_EVALUATED",
                "This geometry type was not included in the continuity calculation.",
                road_id=road_id,
                geometry_index=index,
                geometry_type=child.tag,
            ))
            if index + 1 < len(geometries):
                following = geometries[index + 1]
                end_point = (
                    float(following.get("x", "0")),
                    float(following.get("y", "0")),
                )
                end_heading = float(following.get("hdg", "0"))
            else:
                end_point = (
                    x + math.cos(heading) * length,
                    y + math.sin(heading) * length,
                )
                end_heading = heading
        previous_end = end_point
        previous_heading = end_heading
        cumulative_s = geometry_s + length

    declared_length = float(road.get("length", "0"))
    if abs(declared_length - cumulative_s) > config.road_length_tolerance_m:
        issues.append(_error(
            "XODR_ROAD_LENGTH_MISMATCH",
            "The declared road length differs from its planView length.",
            road_id=road_id,
            declared_length=declared_length,
            planview_length=cumulative_s,
        ))
    return {
        "start": start_point,
        "end": end_point,
        "start_heading": float(first.get("hdg", "0")),
        "end_heading": end_heading,
        "length": cumulative_s,
    }, issues


def _road_junction_endpoint(
    roads: list[ET.Element],
    road_id: str,
    junction_id: str,
    geometry: dict[str, dict[str, Any]],
) -> tuple[float, float] | None:
    """返回主路与指定junction相接的start或end世界坐标。"""
    road = next((item for item in roads if item.get("id") == road_id), None)
    side = _road_junction_contact_side(road, junction_id)
    return geometry.get(road_id, {}).get(side) if side is not None else None


def _road_junction_contact_side(
    road: ET.Element | None,
    junction_id: str,
) -> str | None:
    """返回一条主路通过start还是end与指定路口相接。"""
    if road is None:
        return None
    predecessor = road.find("./link/predecessor")
    if (
        predecessor is not None
        and predecessor.get("elementType") == "junction"
        and predecessor.get("elementId") == junction_id
    ):
        return "start"
    successor = road.find("./link/successor")
    if (
        successor is not None
        and successor.get("elementType") == "junction"
        and successor.get("elementId") == junction_id
    ):
        return "end"
    return None


def _junction_movement(
    road_map: dict[str, ET.Element],
    connecting_road_id: str,
    entry_contact_point: str,
    connector_lane_id: str,
) -> tuple[str, int, str, str] | None:
    """解析一条路口连接道路最终驶入的主路和车道。

    ``connection.contactPoint`` 表示车辆进入连接道路的一端，因此出口位于
    另一端。连接道路在出口端的 road link 和 lane link 共同给出驶出主路、
    主路接触端以及目标车道。
    """
    connector = road_map.get(connecting_road_id)
    if connector is None or entry_contact_point not in {"start", "end"}:
        return None
    exit_side = "end" if entry_contact_point == "start" else "start"
    relation = "successor" if exit_side == "end" else "predecessor"
    road_link = connector.find(f"./link/{relation}")
    if road_link is None or road_link.get("elementType") != "road":
        return None
    outgoing_road_id = road_link.get("elementId", "")
    outgoing_side = road_link.get("contactPoint", "")
    if outgoing_road_id not in road_map or outgoing_side not in {"start", "end"}:
        return None

    connector_lane = next(
        (
            lane
            for lane in connector.findall("./lanes/laneSection/*/lane")
            if lane.get("id") == connector_lane_id
        ),
        None,
    )
    if connector_lane is None:
        return None
    outgoing_lane_link = connector_lane.find(f"./link/{relation}")
    if outgoing_lane_link is None:
        return None
    try:
        outgoing_lane_id = int(outgoing_lane_link.get("id", ""))
    except ValueError:
        return None
    return outgoing_road_id, outgoing_lane_id, exit_side, outgoing_side


def _lane_center_at_contact(
    road: ET.Element,
    lane_id: int,
    side: str,
    geometry: dict[str, Any],
) -> tuple[float, float] | None:
    """计算OpenDRIVE车道中心在道路start/end处的世界坐标。"""
    if side not in {"start", "end"} or not geometry:
        return None
    road_length = float(road.get("length", "0"))
    s = 0.0 if side == "start" else road_length
    reference = geometry.get(side)
    heading = geometry.get(f"{side}_heading")
    if reference is None or heading is None:
        return None
    lateral_offset = _lane_center_offset_at_s(road, lane_id, s)
    if lateral_offset is None:
        return None
    return (
        reference[0] - math.sin(heading) * lateral_offset,
        reference[1] + math.cos(heading) * lateral_offset,
    )


def _lane_center_offset_at_s(
    road: ET.Element,
    lane_id: int,
    s: float,
) -> float | None:
    """计算车道中心相对参考线的有符号横向偏移。"""
    if lane_id == 0:
        return 0.0
    sections = sorted(
        road.findall("./lanes/laneSection"),
        key=lambda section: float(section.get("s", "0")),
    )
    section = next(
        (item for item in reversed(sections) if float(item.get("s", "0")) <= s + 1e-9),
        None,
    )
    if section is None:
        return None
    section_s = float(section.get("s", "0"))
    lane_elements = {
        int(lane.get("id", "0")): lane
        for lane in section.findall("./left/lane") + section.findall("./right/lane")
    }
    if lane_id not in lane_elements:
        return None

    same_side_ids = sorted(
        (
            candidate
            for candidate in lane_elements
            if candidate * lane_id > 0 and abs(candidate) <= abs(lane_id)
        ),
        key=abs,
    )
    widths: dict[int, float] = {}
    local_s = max(0.0, s - section_s)
    for candidate in same_side_ids:
        widths[candidate] = _lane_width_at_offset(
            lane_elements[candidate],
            local_s,
        )
    target_width = widths.get(lane_id)
    if target_width is None:
        return None
    inner_width = sum(
        width for candidate, width in widths.items() if abs(candidate) < abs(lane_id)
    )
    sign = 1.0 if lane_id > 0 else -1.0
    return _lane_offset_at_s(road, s) + sign * (inner_width + target_width / 2.0)


def _lane_type_at_contact(
    road: ET.Element,
    lane_id: int,
    side: str,
) -> str | None:
    """返回指定车道在道路端点所属laneSection中的类型。"""
    road_length = float(road.get("length", "0"))
    s = 0.0 if side == "start" else road_length
    sections = sorted(
        road.findall("./lanes/laneSection"),
        key=lambda section: float(section.get("s", "0")),
    )
    section = next(
        (item for item in reversed(sections) if float(item.get("s", "0")) <= s + 1e-9),
        None,
    )
    if section is None:
        return None
    lane = next(
        (
            item
            for item in section.findall("./left/lane") + section.findall("./right/lane")
            if int(item.get("id", "0")) == lane_id
        ),
        None,
    )
    return lane.get("type") if lane is not None else None


def _lane_width_at_offset(lane: ET.Element, section_offset: float) -> float:
    """求一条车道在laneSection局部里程处的宽度多项式值。"""
    records = sorted(
        lane.findall("width"),
        key=lambda width: float(width.get("sOffset", "0")),
    )
    record = next(
        (
            item
            for item in reversed(records)
            if float(item.get("sOffset", "0")) <= section_offset + 1e-9
        ),
        None,
    )
    if record is None:
        return 0.0
    ds = section_offset - float(record.get("sOffset", "0"))
    return max(0.0, sum(
        float(record.get(coefficient, "0")) * ds ** power
        for power, coefficient in enumerate(("a", "b", "c", "d"))
    ))


def _lane_offset_at_s(road: ET.Element, s: float) -> float:
    """求道路参考线在指定里程处的laneOffset多项式值。"""
    records = sorted(
        road.findall("./lanes/laneOffset"),
        key=lambda offset: float(offset.get("s", "0")),
    )
    record = next(
        (item for item in reversed(records) if float(item.get("s", "0")) <= s + 1e-9),
        None,
    )
    if record is None:
        return 0.0
    ds = s - float(record.get("s", "0"))
    return sum(
        float(record.get(coefficient, "0")) * ds ** power
        for power, coefficient in enumerate(("a", "b", "c", "d"))
    )


def _lane_travel_heading_at_contact(
    geometry: dict[str, Any],
    lane_id: int,
    side: str,
) -> float | None:
    """返回指定车道在道路端点处按允许行驶方向计算的航向。"""
    heading = geometry.get(f"{side}_heading")
    if heading is None:
        return None
    return _wrap_angle(heading + (math.pi if lane_id > 0 else 0.0))


def _wrap_angle(angle: float) -> float:
    """将角度差规整到[-pi, pi]。"""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _undirected_heading_gap(first: float, second: float) -> float:
    """返回两条中心线切向的最小夹角，不把相反参考方向误判为折断。"""
    directed = abs(_wrap_angle(first - second))
    return min(directed, abs(math.pi - directed))


def _spiral_endpoint(
    start: tuple[float, float],
    heading: float,
    length: float,
    curvature_start: float,
    curvature_end: float,
) -> tuple[tuple[float, float], float]:
    """数值积分OpenDRIVE clothoid，得到其终点和终点航向。"""
    if length <= 1e-12:
        return start, heading
    steps = max(16, int(math.ceil(length / 0.05)))
    ds = length / steps
    curvature_rate = (curvature_end - curvature_start) / length
    x, y = start
    for index in range(steps):
        s_mid = (index + 0.5) * ds
        theta_mid = (
            heading
            + curvature_start * s_mid
            + 0.5 * curvature_rate * s_mid * s_mid
        )
        x += math.cos(theta_mid) * ds
        y += math.sin(theta_mid) * ds
    end_heading = (
        heading
        + curvature_start * length
        + 0.5 * curvature_rate * length * length
    )
    return (x, y), end_heading


def _road_intersection_issues(
    spec,
    config: TopologyValidationConfig,
) -> list[TopologyIssue]:
    """发现未声明交叉、重叠道路和单条道路自相交。"""
    issues: list[TopologyIssue] = []
    tolerance_px = config.intersection_tolerance_m / max(
        float(spec.meters_per_pixel),
        1e-9,
    )
    node_map = {node.node_id: node for node in spec.nodes}

    for road in spec.roads:
        segments = list(zip(road.pixel_polyline, road.pixel_polyline[1:]))
        for first_index, first in enumerate(segments):
            for second_index in range(first_index + 2, len(segments)):
                # 相邻线段共享控制点；闭合线的首尾相接当前也不属于受支持道路。
                second = segments[second_index]
                intersection = _segment_intersection(first[0], first[1], second[0], second[1])
                if intersection is None:
                    continue
                point, kind, overlap_px = intersection
                if kind == "point" and _near_any(
                    point,
                    (road.pixel_polyline[first_index + 1],),
                    tolerance_px,
                ):
                    continue
                issues.append(_error(
                    "SELF_INTERSECTING_ROAD",
                    "A road centerline intersects itself and cannot define one unambiguous reference line.",
                    road_id=road.road_id,
                    first_segment=first_index,
                    second_segment=second_index,
                    intersection_pixel=list(point),
                    overlap_m=overlap_px * spec.meters_per_pixel,
                ))
                break
            else:
                continue
            break

    for first_index, first_road in enumerate(spec.roads):
        first_segments = list(zip(
            first_road.pixel_polyline,
            first_road.pixel_polyline[1:],
        ))
        for second_road in spec.roads[first_index + 1:]:
            second_segments = list(zip(
                second_road.pixel_polyline,
                second_road.pixel_polyline[1:],
            ))
            common_nodes = {
                first_road.start_node,
                first_road.end_node,
            } & {
                second_road.start_node,
                second_road.end_node,
            }
            declared_pixels = [
                node_map[node_id].pixel
                for node_id in common_nodes
                if node_id in node_map
            ]
            pair_issue: TopologyIssue | None = None
            for first_segment_index, first_segment in enumerate(first_segments):
                for second_segment_index, second_segment in enumerate(second_segments):
                    intersection = _segment_intersection(
                        first_segment[0],
                        first_segment[1],
                        second_segment[0],
                        second_segment[1],
                    )
                    if intersection is None:
                        continue
                    point, kind, overlap_px = intersection
                    if (
                        kind == "point"
                        and _near_any(point, declared_pixels, tolerance_px)
                    ):
                        continue
                    code = (
                        "OVERLAPPING_ROADS"
                        if kind == "overlap"
                        else "UNDECLARED_ROAD_INTERSECTION"
                    )
                    message = (
                        "Two road centerlines overlap without an explicit shared topology node."
                        if kind == "overlap"
                        else "Road centerlines intersect without an explicit shared junction node."
                    )
                    pair_issue = _error(
                        code,
                        message,
                        first_road_id=first_road.road_id,
                        second_road_id=second_road.road_id,
                        first_segment=first_segment_index,
                        second_segment=second_segment_index,
                        intersection_pixel=list(point),
                        overlap_m=overlap_px * spec.meters_per_pixel,
                    )
                    break
                if pair_issue is not None:
                    break
            if pair_issue is not None:
                issues.append(pair_issue)
    return issues


def _segment_intersection(
    first_start,
    first_end,
    second_start,
    second_end,
) -> tuple[tuple[float, float], str, float] | None:
    """返回两线段交点，或共线重叠区间的中点与长度。"""
    p = (float(first_start[0]), float(first_start[1]))
    q = (float(second_start[0]), float(second_start[1]))
    r = (float(first_end[0]) - p[0], float(first_end[1]) - p[1])
    s = (float(second_end[0]) - q[0], float(second_end[1]) - q[1])
    r_cross_s = _cross_2d(r, s)
    q_minus_p = (q[0] - p[0], q[1] - p[1])
    epsilon = 1e-9
    if abs(r_cross_s) <= epsilon:
        if abs(_cross_2d(q_minus_p, r)) > epsilon:
            return None
        r_length_sq = r[0] * r[0] + r[1] * r[1]
        if r_length_sq <= epsilon:
            return None
        t0 = (q_minus_p[0] * r[0] + q_minus_p[1] * r[1]) / r_length_sq
        t1 = t0 + (s[0] * r[0] + s[1] * r[1]) / r_length_sq
        overlap_start = max(0.0, min(t0, t1))
        overlap_end = min(1.0, max(t0, t1))
        if overlap_end < overlap_start - epsilon:
            return None
        midpoint_t = (overlap_start + overlap_end) / 2.0
        point = (p[0] + r[0] * midpoint_t, p[1] + r[1] * midpoint_t)
        overlap_length = max(0.0, overlap_end - overlap_start) * math.hypot(*r)
        return point, ("overlap" if overlap_length > epsilon else "point"), overlap_length

    t = _cross_2d(q_minus_p, s) / r_cross_s
    u = _cross_2d(q_minus_p, r) / r_cross_s
    if -epsilon <= t <= 1.0 + epsilon and -epsilon <= u <= 1.0 + epsilon:
        return (p[0] + t * r[0], p[1] + t * r[1]), "point", 0.0
    return None


def _cross_2d(first, second) -> float:
    """计算两个二维向量的叉积标量。"""
    return first[0] * second[1] - first[1] * second[0]


def _near_any(point, candidates, tolerance: float) -> bool:
    """判断点是否落在任一候选点的给定容差内。"""
    return any(math.dist(point, candidate) <= tolerance for candidate in candidates)


def _duplicates(values: Iterable[str]) -> list[str]:
    """返回序列中重复出现的标识符。"""
    seen: set[str] = set()
    duplicate: set[str] = set()
    for value in values:
        if value in seen:
            duplicate.add(value)
        seen.add(value)
    return sorted(duplicate)


def _graph_components(
    nodes: Iterable[str],
    edges: Iterable[tuple[str, str]],
) -> list[list[str]]:
    """用无向邻接关系计算道路节点或道路编号的连通分量。"""
    node_set = set(nodes)
    if not node_set:
        return []
    adjacency: dict[str, set[str]] = {node: set() for node in node_set}
    for first, second in edges:
        if first not in adjacency or second not in adjacency:
            continue
        adjacency[first].add(second)
        adjacency[second].add(first)

    components: list[list[str]] = []
    remaining = set(node_set)
    while remaining:
        start = min(remaining)
        stack = [start]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(adjacency[current] - component)
        remaining -= component
        components.append(sorted(component))
    return sorted(components, key=lambda values: values[0] if values else "")


def _nearest_component_nodes(
    components: list[list[str]],
    node_map: dict[str, Any],
    meters_per_pixel: float,
) -> tuple[str, str, float] | None:
    """寻找不同连通分量之间距离最近的一对节点。"""
    nearest: tuple[str, str, float] | None = None
    for first_index, first_component in enumerate(components):
        for second_component in components[first_index + 1:]:
            for first_id in first_component:
                for second_id in second_component:
                    if first_id not in node_map or second_id not in node_map:
                        continue
                    gap = _pixel_distance(
                        node_map[first_id].pixel,
                        node_map[second_id].pixel,
                        meters_per_pixel,
                    )
                    if nearest is None or gap < nearest[2]:
                        nearest = (first_id, second_id, gap)
    return nearest


def _merge_nodes(spec, keep_node_id: str, remove_node_id: str):
    """合并两个近邻节点，并同步更新所有相关道路端点。"""
    from roadrunner_mcp.realmap import TraceRoad

    node_map = {node.node_id: node for node in spec.nodes}
    keep_pixel = node_map[keep_node_id].pixel
    roads = []
    for road in spec.roads:
        start_node = keep_node_id if road.start_node == remove_node_id else road.start_node
        end_node = keep_node_id if road.end_node == remove_node_id else road.end_node
        points = list(road.pixel_polyline)
        if points and start_node == keep_node_id:
            points[0] = keep_pixel
        if points and end_node == keep_node_id:
            points[-1] = keep_pixel
        roads.append(TraceRoad(
            road_id=road.road_id,
            start_node=start_node,
            end_node=end_node,
            pixel_polyline=points,
            width_m=road.width_m,
            surface=road.surface,
        ))
    nodes = [node for node in spec.nodes if node.node_id != remove_node_id]
    return replace(spec, nodes=nodes, roads=roads)


def _bridge_style(roads, first_node: str, second_node: str) -> tuple[float, str]:
    """根据相邻道路推断自动桥接段的宽度和材质。"""
    adjacent = [
        road
        for road in roads
        if first_node in {road.start_node, road.end_node}
        or second_node in {road.start_node, road.end_node}
    ]
    if not adjacent:
        return 3.0, "dirt"
    width = sum(float(road.width_m) for road in adjacent) / len(adjacent)
    surfaces = {road.surface for road in adjacent}
    surface = next(iter(surfaces)) if len(surfaces) == 1 else "dirt"
    return width, surface


def _unique_bridge_id(roads) -> str:
    """生成不会与用户道路编号冲突的桥接道路编号。"""
    existing = {road.road_id for road in roads}
    index = 1
    while f"auto_bridge_{index}" in existing:
        index += 1
    return f"auto_bridge_{index}"


def _remove_consecutive_duplicates(points) -> list[tuple[float, float]]:
    """移除折线中相邻的重复像素点，保留原有顺序。"""
    output: list[tuple[float, float]] = []
    for point in points:
        normalized = (float(point[0]), float(point[1]))
        if not output or math.hypot(
            normalized[0] - output[-1][0],
            normalized[1] - output[-1][1],
        ) > 1e-9:
            output.append(normalized)
    return output


def _pixel_distance(first, second, meters_per_pixel: float) -> float:
    """将两像素点距离按影像比例换算为米。"""
    return math.hypot(
        float(second[0]) - float(first[0]),
        float(second[1]) - float(first[1]),
    ) * float(meters_per_pixel)


def _error(code: str, message: str, **details) -> TopologyIssue:
    """构造会使拓扑判定无效的错误项。"""
    return TopologyIssue("error", code, message, details)


def _warning(code: str, message: str, **details) -> TopologyIssue:
    """构造不阻止生成但需要记录的警告项。"""
    return TopologyIssue("warning", code, message, details)
