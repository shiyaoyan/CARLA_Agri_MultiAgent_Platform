"""Exercise boundary meshes, roadside sampling and CARLA integration without a server."""

from __future__ import annotations

import fnmatch
import json
import math
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from shapely.geometry import Point, Polygon

from agri_carla.agents.scene_generation import SceneGenerationAgent, vegetation_with_road_context
from agri_carla.environment.vegetation import VegetationPlacementConfig, VegetationPlacer, restore_experiment_vegetation
from agri_carla.execution.service import DomainExecutionService
from roadrunner_mcp.farmland import farmland_mesh


def road_request():
    return {
        "map_name": "EnrichmentTest", "image_width_px": 120, "image_height_px": 100,
        "meters_per_pixel": 1.0, "north_up": True,
        "geo_anchor": {"latitude": 37, "longitude": 121, "pixel_x": 0, "pixel_y": 0},
        "nodes": [{"node_id": "a", "pixel": [10, 50]}, {"node_id": "b", "pixel": [110, 50]}],
        "roads": [{"road_id": "main", "start_node": "a", "end_node": "b",
                   "pixel_polyline": [[10, 50], [110, 50]], "width_m": 6, "surface": "dirt"}],
    }


def tree_region(**kwargs):
    return {"name": "trees", "coordinate_mode": "world",
            "polygon": [[0, 0], [30, 0], [30, 30], [0, 30]],
            "density_per_100m2": 1, "min_spacing_m": 3, **kwargs}


def fake_carla(blueprint_ids=("static.prop.tree_oak",), spawn_success=True, synchronous=False):
    blueprints = []
    for name in blueprint_ids:
        bp = Mock(id=name)
        bp.has_attribute.return_value = False
        blueprints.append(bp)
    library = Mock()
    library.filter.side_effect = lambda pattern: [bp for bp in blueprints if fnmatch.fnmatch(bp.id, pattern)]
    world = Mock()
    world.get_blueprint_library.return_value = library
    world.get_map.return_value.name = "/Game/Maps/EnrichmentTest"
    world.get_settings.return_value.synchronous_mode = synchronous
    world.try_spawn_actor.return_value = SimpleNamespace(id=1) if spawn_success else None
    client = Mock()
    client.get_world.return_value = world
    client.load_world.return_value = world
    carla = SimpleNamespace(Client=Mock(return_value=client),
                            Location=lambda **kwargs: SimpleNamespace(**kwargs),
                            Rotation=lambda **kwargs: SimpleNamespace(**kwargs),
                            Transform=lambda *args: args)
    return carla, client, world


class SceneEnrichmentTests(unittest.TestCase):
    def test_farmland_mesh_follows_concave_boundary(self):
        points = [[0, 0], [12, 0], [12, 3], [3, 3], [3, 10], [0, 10]]
        mesh = farmland_mesh([points], z=-0.005)
        boundary = Polygon(points)
        faces = [Polygon([mesh.vertices[i][:2] for i in face]) for face in mesh.faces]
        self.assertAlmostEqual(sum(p.area for p in faces), boundary.area)
        self.assertTrue(all(boundary.covers(p) for p in faces))
        self.assertTrue(all(p[2] == -0.005 for p in mesh.vertices))

    def test_farmland_overlaps_are_not_double_filled(self):
        mesh = farmland_mesh([[[0, 0], [10, 0], [10, 10], [0, 10]],
                              [[5, 0], [15, 0], [15, 10], [5, 10]]])
        area = sum(Polygon([mesh.vertices[i][:2] for i in f]).area for f in mesh.faces)
        self.assertAlmostEqual(area, 150)

    def test_invalid_boundary_and_sampling_parameters_fail(self):
        with self.assertRaises(ValueError):
            farmland_mesh([[[0, 0], [10, 10], [0, 10], [10, 0]]])
        for update in ({"grid_spacing_m": 0}, {"density_per_100m2": -1}, {"min_spacing_m": math.nan}, {"scale": 0}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                VegetationPlacer()._build_plan(VegetationPlacementConfig(tree_regions=[tree_region(**update)]))

    def test_roadside_points_respect_both_sides_clearance_and_spacing(self):
        config = VegetationPlacementConfig.from_dict(vegetation_with_road_context({
            "roadside_trees": {"enabled": True, "band_width_m": 8, "grid_spacing_m": 3, "min_spacing_m": 4},
            "road_clearance_m": 2, "world_offset": [7, 9, 0],
        }, road_request()))
        placer = VegetationPlacer()
        plan = placer._build_plan(config)
        points = [Point(t["x"], t["y"]) for t in plan["trees"]]
        road = placer._road_area(config)
        self.assertGreater(len(points), 10)
        self.assertTrue(all(2 - 1e-8 <= p.distance(road) <= 10 + 1e-8 for p in points))
        self.assertTrue(any(p.y > -36 for p in points))
        self.assertTrue(any(p.y < -46 for p in points))
        self.assertTrue(all(a.distance(b) >= 4 - 1e-8 for i, a in enumerate(points) for b in points[i+1:]))
        self.assertEqual(plan, placer._build_plan(config))

    def test_loop_road_does_not_plant_in_field_interior(self):
        road = road_request()
        ring = [[20, 20], [100, 20], [100, 90], [20, 90], [20, 20]]
        road["roads"][0]["pixel_polyline"] = ring
        config = VegetationPlacementConfig.from_dict(vegetation_with_road_context({
            "roadside_trees": {"enabled": True, "band_width_m": 4},
        }, road))
        placer = VegetationPlacer()
        road_area = placer._road_area(config)
        points = placer._build_plan(config)["trees"]
        self.assertTrue(points)
        self.assertTrue(all(Point(p["x"], p["y"]).distance(road_area) <= 5 + 1e-8 for p in points))

    def test_density_and_cross_region_spacing(self):
        config = VegetationPlacementConfig(tree_regions=[tree_region(), tree_region(name="overlap")])
        plan = VegetationPlacer()._build_plan(config)
        self.assertEqual(len(plan["trees"]), 18)
        points = [Point(t["x"], t["y"]) for t in plan["trees"]]
        self.assertTrue(all(a.distance(b) >= 3 - 1e-8 for i, a in enumerate(points) for b in points[i+1:]))

    def test_small_density_does_not_force_one_tree(self):
        config = VegetationPlacementConfig(tree_regions=[tree_region(density_per_100m2=0.001)])
        self.assertEqual(VegetationPlacer()._build_plan(config)["trees"], [])

    def test_inconsistent_pixel_metadata_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "meters_per_pixel"):
            vegetation_with_road_context({"meters_per_pixel": 0.5}, road_request())

    def test_farmland_dry_run_exports_real_mesh_and_no_carla_connection(self):
        with tempfile.TemporaryDirectory() as folder, patch("agri_carla.environment.vegetation.import_carla") as connect:
            result = VegetationPlacer().run({"dry_run": True, "output_dir": folder,
                "grass_regions": [{"name": "field", "coordinate_mode": "world",
                                   "polygon": [[0, 0], [10, 0], [10, 8], [0, 8]]}]})
            connect.assert_not_called()
            bundle = Path(result["farmland_import_bundle"])
            manifests = list(bundle.glob("*.json"))
            self.assertEqual(len(manifests), 1)
            prop = json.loads(manifests[0].read_text())["props"][0]
            self.assertTrue((bundle / prop["source"]).is_file())
            self.assertIn("PolygonVertexIndex", (bundle / prop["source"]).read_text())
            plan = json.loads(Path(result["placement_plan"]).read_text())
            self.assertEqual(plan["grass_regions"][0]["visual_mode"], "mesh")
            self.assertEqual(plan["grass_regions"][0]["blueprint_filter"], "static.prop." + prop["name"])

    def test_sga_bakes_boundary_into_map_and_links_final_road_geometry(self):
        with tempfile.TemporaryDirectory() as folder:
            request = {"task_type": "configuration_only", "road_scene_output_dir": folder,
                       "road_scene_request": road_request(),
                       "vegetation_request": {"dry_run": True, "tree_regions": [],
                           "roadside_trees": {"enabled": True, "band_width_m": 6},
                           "grass_regions": [{"name": "field", "polygon": [[15, 60], [50, 60], [50, 90], [15, 90]]}]}}
            service = DomainExecutionService()
            result = service.invoke("SGA", "generate_scene", request)
            spec = json.loads(Path(result["spec_path"]).read_text())
            self.assertEqual(spec["farmland_polygons"][0], [[15, -60], [50, -60], [50, -90], [15, -90]])
            self.assertIn("Terrain_Ground_Layer0", Path(result["fbx_path"]).read_text())
            field_block = re.search(r'Geometry::[^"\n]*Terrain_Ground_Layer0.*?Vertices: \*\d+ \{\s*a: ([^}]+)',
                                    Path(result["fbx_path"]).read_text(), re.S)
            coordinates = [float(value) for value in field_block.group(1).strip().split(",")]
            boundary = Polygon(spec["farmland_polygons"][0])
            self.assertTrue(all(boundary.covers(Point(coordinates[i], coordinates[i+1]))
                                for i in range(0, len(coordinates), 3)))
            derived = json.loads(Path(result["vegetation_request_path"]).read_text())
            self.assertEqual(derived["grass_regions"], [])
            self.assertTrue(Path(derived["road_xodr_path"]).is_file())
            placed = service.invoke("SGA", "place_vegetation", request)
            self.assertGreater(placed["planned_tree_count"], 0)
            self.assertEqual(placed["status"], "planned")

    def test_missing_farmland_asset_prevents_even_available_trees_from_spawning(self):
        carla, _, world = fake_carla()
        with tempfile.TemporaryDirectory() as folder, patch("agri_carla.environment.vegetation.import_carla", return_value=carla):
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                VegetationPlacer().run({"output_dir": folder, "tree_regions": [tree_region()],
                    "grass_regions": [{"name": "field", "coordinate_mode": "world",
                                       "polygon": [[0, 0], [8, 0], [8, 10], [0, 10]]}]})
            world.try_spawn_actor.assert_not_called()

    def test_imported_farmland_is_spawned_once_at_origin(self):
        with tempfile.TemporaryDirectory() as folder:
            request = {"output_dir": folder, "dry_run": True,
                       "grass_regions": [{"name": "field", "coordinate_mode": "world",
                                          "polygon": [[15, 10], [25, 10], [25, 20], [15, 20]]}]}
            placer = VegetationPlacer()
            result = placer.run(request)
            plan = json.loads(Path(result["placement_plan"]).read_text())
            bp = plan["grass_regions"][0]["blueprint_filter"]
            carla, _, world = fake_carla([bp], synchronous=True)
            with patch("agri_carla.environment.vegetation.import_carla", return_value=carla):
                request["dry_run"] = False
                result = placer.run(request)
            self.assertEqual(result["spawned_grass_count"], 1)
            world.try_spawn_actor.assert_called_once()
            self.assertEqual(world.try_spawn_actor.call_args.args[1], ())
            world.tick.assert_called_once()

    def test_pga_accepts_roadside_only_request(self):
        from agri_carla.agents.plan_generation import PlanGenerationAgent
        request = {"image_width_px": 120, "image_height_px": 100, "meters_per_pixel": 1,
                   "tree_regions": [], "grass_regions": [], "roadside_trees": {"enabled": True}}
        self.assertEqual(PlanGenerationAgent._vegetation_geometry_errors(request), [])

    def test_eea_service_forwards_environment_to_both_experiments(self):
        service = DomainExecutionService()
        service._experiment_execution_agent = Mock()
        request = {"vegetation_request": {"tree_regions": [tree_region()]}, "experiment": {"config": {}}}
        service.invoke("EEA", "run_trajectory_tracking", request)
        tracking = service._experiment_execution_agent.run_trajectory_tracking.call_args.kwargs["config"]
        self.assertEqual(tracking.vegetation_request, request["vegetation_request"])
        service.invoke("EEA", "run_full_process_navigation", request)
        full = service._experiment_execution_agent.run_full_process_navigation.call_args.args[0]
        self.assertEqual(full.vegetation_request, request["vegetation_request"])

    def test_missing_assets_fail_before_spawning_and_write_report(self):
        carla, _, world = fake_carla(blueprint_ids=())
        with tempfile.TemporaryDirectory() as folder, patch("agri_carla.environment.vegetation.import_carla", return_value=carla):
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                VegetationPlacer().run({"output_dir": folder, "tree_regions": [tree_region()]})
            world.try_spawn_actor.assert_not_called()
            report = json.loads((Path(folder) / "vegetation_spawn_report.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["missing_blueprints"], ["static.prop.tree*"])

    def test_existing_map_is_not_reloaded_and_async_world_is_not_ticked(self):
        carla, client, world = fake_carla()
        with tempfile.TemporaryDirectory() as folder, patch("agri_carla.environment.vegetation.import_carla", return_value=carla):
            result = VegetationPlacer().run({"output_dir": folder, "carla_map": "EnrichmentTest", "tree_regions": [tree_region()]})
            self.assertEqual(result["status"], "completed")
            self.assertGreater(result["spawned_tree_count"], 0)
            client.load_world.assert_not_called()
            world.tick.assert_not_called()
            world.wait_for_tick.assert_called_once()

    def test_wrong_map_requires_explicit_loading(self):
        carla, client, world = fake_carla()
        with tempfile.TemporaryDirectory() as folder, patch("agri_carla.environment.vegetation.import_carla", return_value=carla):
            with self.assertRaisesRegex(RuntimeError, "differs"):
                VegetationPlacer().run({"output_dir": folder, "carla_map": "WrongMap"})
            client.load_world.assert_not_called()

    def test_failed_actor_creation_is_not_reported_as_success(self):
        carla, _, _ = fake_carla(spawn_success=False)
        with tempfile.TemporaryDirectory() as folder, patch("agri_carla.environment.vegetation.import_carla", return_value=carla):
            result = VegetationPlacer().run({"output_dir": folder, "strict_spawn": False, "tree_regions": [tree_region()]})
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["spawned_tree_count"], 0)
            self.assertGreater(result["skipped_tree_count"], 0)

    def test_restore_uses_experiment_server_without_reloading_it(self):
        config = SimpleNamespace(vegetation_request={"tree_regions": [tree_region()]},
            host="127.0.0.1", port=2345, timeout=30, carla_map="EnrichmentTest", output_dir="results/test")
        with patch.object(VegetationPlacer, "run", return_value={}) as run:
            restore_experiment_vegetation(config)
        request = run.call_args.args[0]
        self.assertEqual(request["port"], 2345)
        self.assertFalse(request["load_map"])
        self.assertTrue(request["strict_spawn"])

    def test_both_experiments_restore_props_after_loading_map(self):
        from agri_carla.simulation.tracking_experiment import TrackingExperimentRunner, TrackingExperimentConfig
        from agri_carla.simulation.full_process_experiment import FullProcessNavigationExperiment, FullProcessExperimentConfig

        for module, runner, config_type in (
            ("tracking_experiment", TrackingExperimentRunner, TrackingExperimentConfig),
            ("full_process_experiment", FullProcessNavigationExperiment, FullProcessExperimentConfig),
        ):
            carla, client, _ = fake_carla()
            config = config_type(carla_map="EnrichmentTest", vegetation_request={"tree_regions": [tree_region()]})
            if module == "full_process_experiment":
                config.reload_world = True
            events = []
            client.load_world.side_effect = lambda name: (events.append("load"), client.get_world())[1]
            with patch(f"agri_carla.simulation.{module}.import_carla", return_value=carla), patch(
                "agri_carla.environment.vegetation.restore_experiment_vegetation", side_effect=lambda c: events.append("restore")
            ):
                runner(config)
            self.assertEqual(events, ["load", "restore"])


if __name__ == "__main__":
    unittest.main()
