"""验证 SGA 道路拓扑检查、自动修复和重新生成机制。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree as ET

from agri_carla.road_generation.agent import (
    AgriculturalRoadGenerator,
    RoadSceneRequest,
)
from agri_carla.agents.scene_generation import SceneGenerationAgent
from agri_carla.road_generation.topology import (
    RoadTopologyValidator,
    TopologyValidationConfig,
)
from roadrunner_mcp.realmap import (
    GeoAnchor,
    RealMapTraceSpec,
    TraceNode,
    TraceRoad,
    generate_realmap_artifacts,
)


class RoadTopologyValidatorTests(unittest.TestCase):
    """使用人工构造的断裂道路检查端点吸附和桥接策略。"""

    def test_sga_generates_from_prevalidated_geometry_without_source_image(self):
        """复用已验证几何时，SGA 不应强制要求原始卫星图仍在本地。"""
        request = {
            "map_name": "OfflineStructuredRoad",
            "image_width_px": 100,
            "image_height_px": 100,
            "meters_per_pixel": 1.0,
            "geo_anchor": {
                "latitude": 37.0,
                "longitude": 121.0,
                "pixel_x": 0.0,
                "pixel_y": 0.0,
            },
            "nodes": [
                {"node_id": "start", "pixel": [10.0, 50.0]},
                {"node_id": "end", "pixel": [90.0, 50.0]},
            ],
            "roads": [{
                "road_id": "farm_road",
                "start_node": "start",
                "end_node": "end",
                "pixel_polyline": [[10.0, 50.0], [90.0, 50.0]],
                "width_m": 4.0,
                "surface": "dirt",
            }],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            result = SceneGenerationAgent().run(request, temp_dir)
            generated_files = [
                Path(result.xodr_path),
                Path(result.fbx_path),
                Path(result.texture_path),
            ]

            self.assertTrue(result.topology_valid)
            self.assertTrue(all(path.is_file() for path in generated_files))

    def test_generation_agent_repairs_and_regenerates_invalid_topology(self):
        """SGA 应修复无效输入、重新生成文件并给出有效报告。"""
        request = RoadSceneRequest(
            map_name="RegeneratedFarmRoad",
            image_path="",
            image_width_px=100,
            image_height_px=100,
            meters_per_pixel=1.0,
            north_up=True,
            geo_anchor={
                "latitude": 37.0,
                "longitude": 121.0,
                "pixel_x": 0.0,
                "pixel_y": 0.0,
            },
            nodes=[
                {"node_id": "n1", "pixel": [0.0, 0.0]},
                {"node_id": "n2", "pixel": [10.0, 0.0]},
                {"node_id": "n3", "pixel": [14.0, 0.0]},
                {"node_id": "n4", "pixel": [24.0, 0.0]},
            ],
            roads=[
                {
                    "road_id": "r1",
                    "start_node": "n1",
                    "end_node": "n2",
                    "pixel_polyline": [[1.0, 0.0], [9.0, 0.0]],
                },
                {
                    "road_id": "r2",
                    "start_node": "n3",
                    "end_node": "n4",
                    "pixel_polyline": [[14.0, 0.0], [24.0, 0.0]],
                },
            ],
            topology_validation={
                "endpoint_tolerance_m": 0.1,
                "max_endpoint_snap_m": 2.0,
                "bridge_max_gap_m": 5.0,
                "max_regeneration_attempts": 2,
            },
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            result = AgriculturalRoadGenerator().generate(request, temp_dir)
            report_exists = Path(result.topology_report_path).exists()

        self.assertTrue(result.topology_valid)
        self.assertEqual(result.regeneration_count, 1)
        self.assertGreaterEqual(result.repair_action_count, 2)
        self.assertTrue(report_exists)

    def test_repairs_endpoint_gaps_and_nearby_disconnected_components(self):
        """允许范围内应同时修正端点偏差并桥接相邻断开分量。"""
        spec = RealMapTraceSpec(
            map_name="BrokenFarmRoad",
            image_path="",
            image_width_px=100,
            image_height_px=100,
            meters_per_pixel=1.0,
            north_up=True,
            geo_anchor=GeoAnchor(37.0, 121.0, 0.0, 0.0),
            nodes=[
                TraceNode("n1", (0.0, 0.0)),
                TraceNode("n2", (10.0, 0.0)),
                TraceNode("n3", (14.0, 0.0)),
                TraceNode("n4", (24.0, 0.0)),
            ],
            roads=[
                TraceRoad("r1", "n1", "n2", [(1.0, 0.0), (9.0, 0.0)]),
                TraceRoad("r2", "n3", "n4", [(14.0, 0.0), (24.0, 0.0)]),
            ],
        )
        validator = RoadTopologyValidator(TopologyValidationConfig(
            endpoint_tolerance_m=0.1,
            max_endpoint_snap_m=2.0,
            bridge_max_gap_m=5.0,
        ))

        before = validator.validate_spec(spec)
        self.assertFalse(before.valid)
        repaired, actions = validator.repair_spec(spec)
        after = validator.validate_spec(repaired)

        self.assertTrue(after.valid)
        self.assertTrue(any(action["action"] == "snap_road_endpoint" for action in actions))
        self.assertTrue(any(action["action"] == "add_bridge_road" for action in actions))
        self.assertEqual(len(repaired.roads), 3)

        with tempfile.TemporaryDirectory() as temp_dir:
            result = generate_realmap_artifacts(repaired, Path(temp_dir))
            xodr_check = validator.validate_xodr(result.xodr_path)
        self.assertTrue(
            xodr_check.valid,
            [issue.to_dict() for issue in xodr_check.issues],
        )

    def test_does_not_bridge_components_beyond_the_configured_limit(self):
        """超过最大间隙的独立道路不得被自动桥接。"""
        spec = RealMapTraceSpec(
            map_name="SeparatedFarmRoads",
            image_path="",
            image_width_px=200,
            image_height_px=100,
            meters_per_pixel=1.0,
            north_up=True,
            geo_anchor=GeoAnchor(37.0, 121.0, 0.0, 0.0),
            nodes=[
                TraceNode("n1", (0.0, 0.0)),
                TraceNode("n2", (10.0, 0.0)),
                TraceNode("n3", (100.0, 0.0)),
                TraceNode("n4", (110.0, 0.0)),
            ],
            roads=[
                TraceRoad("r1", "n1", "n2", [(0.0, 0.0), (10.0, 0.0)]),
                TraceRoad("r2", "n3", "n4", [(100.0, 0.0), (110.0, 0.0)]),
            ],
        )
        validator = RoadTopologyValidator(TopologyValidationConfig(
            bridge_max_gap_m=5.0,
        ))
        repaired, actions = validator.repair_spec(spec)

        self.assertFalse(validator.validate_spec(repaired).valid)
        self.assertFalse(any(action["action"] == "add_bridge_road" for action in actions))

    def test_realmap_bundle_keeps_mixed_road_materials(self):
        """混合道路场景应分别输出农田、土路和沥青路材质。"""
        spec = RealMapTraceSpec(
            map_name="MixedSurfaceFarm",
            image_path="",
            image_width_px=100,
            image_height_px=100,
            meters_per_pixel=1.0,
            north_up=True,
            geo_anchor=GeoAnchor(37.0, 121.0, 0.0, 0.0),
            nodes=[
                TraceNode("west", (0.0, 50.0)),
                TraceNode("junction", (50.0, 50.0)),
                TraceNode("north", (50.0, 0.0)),
            ],
            roads=[
                TraceRoad(
                    "dirt_road",
                    "west",
                    "junction",
                    [(0.0, 50.0), (50.0, 50.0)],
                    width_m=5.0,
                    surface="dirt",
                ),
                TraceRoad(
                    "asphalt_road",
                    "junction",
                    "north",
                    [(50.0, 50.0), (50.0, 0.0)],
                    width_m=8.0,
                    surface="asphalt",
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            result = generate_realmap_artifacts(spec, Path(temp_dir))
            texture_names = {path.name for path in result.texture_paths}
            bundle_names = {
                path.name for path in result.import_bundle_dir.iterdir()
            }
            fbx_text = result.fbx_path.read_text(encoding="utf-8")

        expected_textures = {
            "MixedSurfaceFarm_field_diffuse.tga",
            "MixedSurfaceFarm_dirt_diffuse.tga",
            "MixedSurfaceFarm_asphalt_diffuse.tga",
        }
        self.assertEqual(texture_names, expected_textures)
        self.assertTrue(expected_textures.issubset(bundle_names))
        self.assertIn("M_Agricultural_Field", fbx_text)
        self.assertIn("M_Dirt_Farm_Road", fbx_text)
        self.assertIn("M_Asphalt_Road", fbx_text)

    def test_fbx_axis_metadata_matches_roadrunner_carla_convention(self):
        """FBX应使用RoadRunner导出文件相同的Z-up、-Y-front和米制声明。"""
        spec = RealMapTraceSpec(
            map_name="AxisAlignedFarm",
            image_path="",
            image_width_px=50,
            image_height_px=50,
            meters_per_pixel=1.0,
            north_up=True,
            geo_anchor=GeoAnchor(37.0, 121.0, 0.0, 0.0),
            nodes=[TraceNode("start", (0.0, 0.0)), TraceNode("end", (40.0, 0.0))],
            roads=[TraceRoad("road", "start", "end", [(0.0, 0.0), (40.0, 0.0)])],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            result = generate_realmap_artifacts(spec, Path(temp_dir))
            fbx_text = result.fbx_path.read_text(encoding="utf-8")

        self.assertIn('P: "UpAxis", "int", "Integer", "",2', fbx_text)
        self.assertIn('P: "FrontAxisSign", "int", "Integer", "",-1', fbx_text)
        self.assertIn('P: "OriginalUpAxis", "int", "Integer", "",-1', fbx_text)
        self.assertIn('P: "UnitScaleFactor", "double", "Number", "",100', fbx_text)
        self.assertIn('P: "OriginalUnitScaleFactor", "double", "Number", "",1', fbx_text)

    def test_fbx_road_surfaces_have_upward_normals_and_small_height_offsets(self):
        """道路面应朝向+Z，且道路、路口和农田之间不得形成可见碰撞台阶。"""
        spec = RealMapTraceSpec(
            map_name="SurfaceAlignedFarm",
            image_path="",
            image_width_px=80,
            image_height_px=80,
            meters_per_pixel=1.0,
            north_up=False,
            geo_anchor=GeoAnchor(37.0, 121.0, 0.0, 0.0),
            nodes=[
                TraceNode("west", (0.0, 40.0)),
                TraceNode("junction", (40.0, 40.0)),
                TraceNode("north", (40.0, 0.0)),
            ],
            roads=[
                TraceRoad("west_road", "west", "junction", [(0.0, 40.0), (40.0, 40.0)]),
                TraceRoad("north_road", "junction", "north", [(40.0, 40.0), (40.0, 0.0)]),
            ],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            result = generate_realmap_artifacts(spec, Path(temp_dir))
            fbx_text = result.fbx_path.read_text(encoding="utf-8")

        self.assertIn('MappingInformationType: "ByPolygonVertex"', fbx_text)
        self.assertIn('Type: "LayerElementNormal"', fbx_text)
        self.assertIn('a: 0,0,1,0,0,1', fbx_text)
        self.assertIn('-0.005', fbx_text)
        self.assertIn(',0.001', fbx_text)

    def test_unrepairable_topology_is_rejected_before_artifact_generation(self):
        """未声明交叉口无法安全修复时，应只写报告而不生成误导性的地图文件。"""
        request = RoadSceneRequest(
            map_name="RejectUndeclaredCrossing",
            image_path="",
            image_width_px=100,
            image_height_px=100,
            meters_per_pixel=1.0,
            north_up=False,
            geo_anchor={
                "latitude": 37.0,
                "longitude": 121.0,
                "pixel_x": 0.0,
                "pixel_y": 0.0,
            },
            nodes=[
                {"node_id": "west", "pixel": [0.0, 50.0]},
                {"node_id": "east", "pixel": [100.0, 50.0]},
                {"node_id": "north", "pixel": [50.0, 0.0]},
                {"node_id": "south", "pixel": [50.0, 100.0]},
            ],
            roads=[
                {
                    "road_id": "horizontal",
                    "start_node": "west",
                    "end_node": "east",
                    "pixel_polyline": [[0.0, 50.0], [100.0, 50.0]],
                },
                {
                    "road_id": "vertical",
                    "start_node": "north",
                    "end_node": "south",
                    "pixel_polyline": [[50.0, 0.0], [50.0, 100.0]],
                },
            ],
            topology_validation={
                "auto_repair": True,
                "max_regeneration_attempts": 2,
                "require_single_component": False,
                "fail_on_unresolved": True,
            },
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            with self.assertRaises(RuntimeError):
                AgriculturalRoadGenerator().generate(request, output_dir)
            report = output_dir / "RejectUndeclaredCrossing.topology_report.json"
            self.assertTrue(report.exists())
            self.assertFalse((output_dir / "RejectUndeclaredCrossing.xodr").exists())
            self.assertFalse((output_dir / "RejectUndeclaredCrossing.fbx").exists())

    def test_realmap_generator_rejects_unimplemented_slope_requests(self):
        """坡度尚未进入真实布局高程模型时，不得静默输出平地并声称已应用。"""
        request = RoadSceneRequest(
            map_name="UnsupportedSlopeFarm",
            image_path="",
            image_width_px=100,
            image_height_px=100,
            meters_per_pixel=1.0,
            north_up=False,
            flat_terrain=False,
            geo_anchor={
                "latitude": 37.0,
                "longitude": 121.0,
                "pixel_x": 0.0,
                "pixel_y": 0.0,
            },
            nodes=[
                {"node_id": "start", "pixel": [0.0, 0.0]},
                {"node_id": "end", "pixel": [50.0, 0.0]},
            ],
            roads=[{
                "road_id": "road",
                "start_node": "start",
                "end_node": "end",
                "pixel_polyline": [[0.0, 0.0], [50.0, 0.0]],
            }],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "supports flat terrain only"):
                AgriculturalRoadGenerator().generate(request, temp_dir)

    def test_junction_connectors_are_curved_and_spatially_continuous(self):
        """路口连接道路应以圆弧转弯，并在主路边界处保持零间隙。"""
        spec = RealMapTraceSpec(
            map_name="ContinuousJunctionFarm",
            image_path="",
            image_width_px=100,
            image_height_px=100,
            meters_per_pixel=1.0,
            north_up=False,
            geo_anchor=GeoAnchor(37.0, 121.0, 0.0, 0.0),
            nodes=[
                TraceNode("west", (0.0, 50.0)),
                TraceNode("junction", (50.0, 50.0)),
                TraceNode("north", (50.0, 0.0)),
            ],
            roads=[
                TraceRoad("west_road", "west", "junction", [(0.0, 50.0), (50.0, 50.0)]),
                TraceRoad("north_road", "junction", "north", [(50.0, 50.0), (50.0, 0.0)]),
            ],
        )
        validator = RoadTopologyValidator()
        with tempfile.TemporaryDirectory() as temp_dir:
            result = generate_realmap_artifacts(spec, Path(temp_dir))
            root = ET.parse(result.xodr_path).getroot()
            valid = validator.validate_xodr(result.xodr_path)
            connector = next(road for road in root.findall("road") if road.get("junction") != "-1")
            connector_geometry = connector.findall("./planView/geometry")
            first_geometry = connector_geometry[0]
            first_geometry.set("x", str(float(first_geometry.get("x", "0")) + 2.0))
            broken_path = Path(temp_dir) / "broken.xodr"
            ET.ElementTree(root).write(broken_path, encoding="utf-8", xml_declaration=True)
            broken = validator.validate_xodr(broken_path)

        self.assertTrue(valid.valid, [issue.to_dict() for issue in valid.issues])
        self.assertGreater(len(root.findall(".//arc")), 0)
        self.assertFalse(broken.valid)
        self.assertTrue(any(
            issue.code in {"XODR_JUNCTION_GEOMETRY_GAP", "XODR_ROAD_LINK_GEOMETRY_GAP"}
            for issue in broken.issues
        ))

    def test_curved_trace_uses_tangent_continuous_arcs(self):
        """多控制点弯曲道路不应继续由带方向突变的纯直线折段表示。"""
        spec = RealMapTraceSpec(
            map_name="CurvedFarmRoad",
            image_path="",
            image_width_px=100,
            image_height_px=100,
            meters_per_pixel=1.0,
            north_up=False,
            geo_anchor=GeoAnchor(37.0, 121.0, 0.0, 0.0),
            nodes=[TraceNode("start", (0.0, 0.0)), TraceNode("end", (60.0, 60.0))],
            roads=[TraceRoad(
                "curve",
                "start",
                "end",
                [(0.0, 0.0), (30.0, 0.0), (50.0, 20.0), (60.0, 60.0)],
            )],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            result = generate_realmap_artifacts(spec, Path(temp_dir))
            root = ET.parse(result.xodr_path).getroot()
            check = RoadTopologyValidator().validate_xodr(result.xodr_path)

        self.assertGreaterEqual(len(root.findall(".//arc")), 2)
        self.assertTrue(check.valid, [issue.to_dict() for issue in check.issues])

    def test_mixed_width_connectors_taper_without_lane_center_gap(self):
        """宽度不同的道路连接时应渐变车道宽度，并保持车道中心连续。"""
        spec = RealMapTraceSpec(
            map_name="VariableWidthJunction",
            image_path="",
            image_width_px=120,
            image_height_px=120,
            meters_per_pixel=1.0,
            north_up=False,
            geo_anchor=GeoAnchor(37.0, 121.0, 0.0, 0.0),
            nodes=[
                TraceNode("west", (0.0, 60.0)),
                TraceNode("junction", (60.0, 60.0)),
                TraceNode("north", (60.0, 0.0)),
            ],
            roads=[
                TraceRoad("narrow", "west", "junction", [(0.0, 60.0), (60.0, 60.0)], width_m=4.0),
                TraceRoad("wide", "junction", "north", [(60.0, 60.0), (60.0, 0.0)], width_m=8.0),
            ],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            result = generate_realmap_artifacts(spec, Path(temp_dir))
            root = ET.parse(result.xodr_path).getroot()
            check = RoadTopologyValidator().validate_xodr(result.xodr_path)

        connector_widths = [
            width
            for road in root.findall("road")
            if road.get("junction") != "-1"
            for width in road.findall("./lanes/laneSection/*/lane/width")
        ]
        self.assertTrue(any(abs(float(width.get("b", "0"))) > 1e-9 for width in connector_widths))
        self.assertTrue(check.valid, [issue.to_dict() for issue in check.issues])

    def test_three_way_junction_has_six_directed_movements_and_two_way_arms(self):
        """三岔口应连接全部六个行驶方向，且每条支路均可双向行驶。"""
        spec = RealMapTraceSpec(
            map_name="BidirectionalThreeWayJunction",
            image_path="",
            image_width_px=140,
            image_height_px=140,
            meters_per_pixel=1.0,
            north_up=False,
            geo_anchor=GeoAnchor(37.0, 121.0, 0.0, 0.0),
            nodes=[
                TraceNode("west", (0.0, 70.0)),
                TraceNode("junction", (70.0, 70.0)),
                TraceNode("north", (70.0, 0.0)),
                TraceNode("south", (70.0, 140.0)),
            ],
            roads=[
                TraceRoad("west_arm", "west", "junction", [(0.0, 70.0), (70.0, 70.0)]),
                TraceRoad("north_arm", "north", "junction", [(70.0, 0.0), (70.0, 70.0)]),
                TraceRoad("south_arm", "junction", "south", [(70.0, 70.0), (70.0, 140.0)]),
            ],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            result = generate_realmap_artifacts(spec, Path(temp_dir))
            root = ET.parse(result.xodr_path).getroot()
            check = RoadTopologyValidator().validate_xodr(result.xodr_path)

        normal_roads = {
            road.get("id", ""): road
            for road in root.findall("road")
            if road.get("junction", "-1") == "-1"
        }
        for road in normal_roads.values():
            driving_lanes = {
                int(lane.get("id", "0"))
                for lane in road.findall("./lanes/laneSection/*/lane")
                if lane.get("type") == "driving"
            }
            self.assertEqual(driving_lanes, {-1, 1})

        junction = root.find("junction")
        self.assertIsNotNone(junction)
        self.assertEqual(len(junction.findall("connection")), 6)
        self.assertTrue(check.valid, [issue.to_dict() for issue in check.issues])

    def test_validator_rejects_missing_junction_movement_and_one_way_arm(self):
        """连接项或反向车道缺失时，不得把断开的路口判为可导入。"""
        spec = RealMapTraceSpec(
            map_name="BrokenThreeWayJunction",
            image_path="",
            image_width_px=140,
            image_height_px=140,
            meters_per_pixel=1.0,
            north_up=False,
            geo_anchor=GeoAnchor(37.0, 121.0, 0.0, 0.0),
            nodes=[
                TraceNode("west", (0.0, 70.0)),
                TraceNode("junction", (70.0, 70.0)),
                TraceNode("north", (70.0, 0.0)),
                TraceNode("south", (70.0, 140.0)),
            ],
            roads=[
                TraceRoad("west_arm", "west", "junction", [(0.0, 70.0), (70.0, 70.0)]),
                TraceRoad("north_arm", "north", "junction", [(70.0, 0.0), (70.0, 70.0)]),
                TraceRoad("south_arm", "junction", "south", [(70.0, 70.0), (70.0, 140.0)]),
            ],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            result = generate_realmap_artifacts(spec, output_dir)
            root = ET.parse(result.xodr_path).getroot()
            junction = root.find("junction")
            junction.remove(junction.findall("connection")[0])

            first_main_road = next(
                road for road in root.findall("road")
                if road.get("junction", "-1") == "-1"
            )
            left = first_main_road.find("./lanes/laneSection/left")
            first_main_road.find("./lanes/laneSection").remove(left)

            broken_path = output_dir / "broken_three_way.xodr"
            ET.ElementTree(root).write(broken_path, encoding="utf-8", xml_declaration=True)
            check = RoadTopologyValidator().validate_xodr(broken_path)

        issue_codes = {issue.code for issue in check.issues}
        self.assertFalse(check.valid)
        self.assertIn("XODR_MISSING_JUNCTION_MOVEMENT", issue_codes)
        self.assertIn("XODR_MISSING_REVERSE_DRIVING_LANE", issue_codes)

    def test_undeclared_road_intersection_is_rejected(self):
        """中心线相交但没有共享节点时，不得生成看似连通的伪路口。"""
        spec = RealMapTraceSpec(
            map_name="UndeclaredCrossing",
            image_path="",
            image_width_px=100,
            image_height_px=100,
            meters_per_pixel=1.0,
            north_up=False,
            geo_anchor=GeoAnchor(37.0, 121.0, 0.0, 0.0),
            nodes=[
                TraceNode("west", (0.0, 50.0)),
                TraceNode("east", (100.0, 50.0)),
                TraceNode("north", (50.0, 0.0)),
                TraceNode("south", (50.0, 100.0)),
            ],
            roads=[
                TraceRoad("horizontal", "west", "east", [(0.0, 50.0), (100.0, 50.0)]),
                TraceRoad("vertical", "north", "south", [(50.0, 0.0), (50.0, 100.0)]),
            ],
        )

        check = RoadTopologyValidator(TopologyValidationConfig(
            require_single_component=False,
        )).validate_spec(spec)

        self.assertFalse(check.valid)
        self.assertTrue(any(
            issue.code == "UNDECLARED_ROAD_INTERSECTION"
            for issue in check.issues
        ))

    def test_each_junction_uses_only_its_incident_road_widths(self):
        """远处宽路不应放大窄路路口的裁切范围。"""
        spec = RealMapTraceSpec(
            map_name="LocalJunctionRadius",
            image_path="",
            image_width_px=220,
            image_height_px=120,
            meters_per_pixel=1.0,
            north_up=False,
            geo_anchor=GeoAnchor(37.0, 121.0, 0.0, 0.0),
            nodes=[
                TraceNode("west", (0.0, 60.0)),
                TraceNode("j1", (50.0, 60.0)),
                TraceNode("j2", (150.0, 60.0)),
                TraceNode("east", (210.0, 60.0)),
                TraceNode("north1", (50.0, 0.0)),
                TraceNode("north2", (150.0, 0.0)),
            ],
            roads=[
                TraceRoad("west_arm", "west", "j1", [(0.0, 60.0), (50.0, 60.0)], width_m=3.0),
                TraceRoad("between", "j1", "j2", [(50.0, 60.0), (150.0, 60.0)], width_m=3.0),
                TraceRoad("east_arm", "j2", "east", [(150.0, 60.0), (210.0, 60.0)], width_m=12.0),
                TraceRoad("north_arm_1", "j1", "north1", [(50.0, 60.0), (50.0, 0.0)], width_m=3.0),
                TraceRoad("north_arm_2", "j2", "north2", [(150.0, 60.0), (150.0, 0.0)], width_m=12.0),
            ],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            result = generate_realmap_artifacts(spec, Path(temp_dir))
            root = ET.parse(result.xodr_path).getroot()

        between = next(road for road in root.findall("road") if road.get("name") == "between")
        self.assertAlmostEqual(float(between.get("length")), 87.0, places=5)


if __name__ == "__main__":
    unittest.main()
