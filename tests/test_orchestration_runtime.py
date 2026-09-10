"""验证 PGA 解析修复、CrewAI 运行记录和执行端协议。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from PIL import Image

from agri_carla.agents.plan_generation import PlanGenerationAgent
from agri_carla.crewai_runtime.flow import AgriculturalSimulationFlow
from agri_carla.environment import VegetationPlacementConfig
from agri_carla.execution import DomainExecutionService
from agri_carla.orchestration import ApplicationRunLedger
from agri_carla.orchestration.errors import NeedsUserInputError
from agri_carla.worker.server import CarlaWorkerApplication, WorkerTaskRepository


class FakeJSONLLM:
    """按调用顺序返回预设 JSON 的模型替身。"""
    def __init__(self, responses):
        """保存响应队列和调用记录。"""
        self.responses = list(responses)
        self.calls = []

    def generate_json(self, **kwargs):
        """返回下一项预设响应。"""
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("No fake response remains.")
        return self.responses.pop(0)


class OrchestrationRuntimeTests(unittest.TestCase):
    """检查 PGA 与 CrewAI 支撑服务在典型流程中的行为。"""

    def test_pga_uses_input_files_and_repairs_invalid_model_output(self):
        """PGA 应利用输入文件补字段，并对首次无效模型结果进行一次修复。"""
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            trajectory = root / "trajectory.csv"
            trajectory.write_text(
                "longitude,latitude,time,speed\n"
                "121.0,37.0,0,1.0\n",
                encoding="utf-8",
            )
            fake_llm = FakeJSONLLM(
                [
                    {"experiment": {}},
                    {
                        "task_type": "trajectory_tracking",
                        "experiment": {
                            "config": {
                                "carla_map": "YantaiTrackingField",
                                "vehicle_bp": "vehicle.tractor.tractor",
                            }
                        },
                    },
                ]
            )
            planner = PlanGenerationAgent(
                llm_client=fake_llm,
                project_root=root,
            )
            result = planner.run(
                {
                    "natural_language_request": "Track the supplied trajectory.",
                    "data_files": ["trajectory.csv"],
                    "llm": {
                        "enabled": True,
                        "max_repair_attempts": 2,
                    },
                }
            )

            self.assertEqual(len(fake_llm.calls), 2)
            self.assertEqual(
                result.structured_request["experiment"]["trajectory_csv"],
                "trajectory.csv",
            )
            self.assertEqual(
                [step.agent for step in result.execution_plan.steps],
                ["CIA", "EEA"],
            )
            self.assertEqual(
                result.validation_report["llm_repair_attempts"],
                1,
            )

    def test_vegetation_config_ignores_multimodal_provenance_fields(self):
        """SGA 配置应容忍模型附带的图像路径和坐标原点说明。"""
        config = VegetationPlacementConfig.from_dict(
            {
                "carla_map": "TestMap",
                "image_width_px": 100,
                "image_height_px": 80,
                "meters_per_pixel": 0.2,
                "image_path": "scene.png",
                "coordinate_origin": "top-left",
            }
        )
        self.assertEqual(config.carla_map, "TestMap")
        self.assertEqual(config.image_width_px, 100)

    def test_pga_multimodal_geometry_is_independently_verified(self):
        """PGA 应复核模型生成的道路几何并仅把连通结果交给 SGA。"""
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            image_path = root / "scene.png"
            image_path.write_bytes(b"image-placeholder")
            metadata = {
                "task_type": "configuration_only",
                "road_scene_request": {
                    "map_name": "DirectGeometryScene",
                    "image_path": "scene.png",
                    "image_width_px": 100,
                    "image_height_px": 100,
                    "meters_per_pixel": 0.2,
                    "north_up": True,
                    "geo_anchor": {
                        "latitude": 37.0,
                        "longitude": 121.0,
                        "pixel_x": 50,
                        "pixel_y": 50,
                    },
                },
                "road_scene_output_dir": "outputs/direct_geometry",
            }
            initial_candidate = {
                "task_type": "configuration_only",
                "road_scene_request": {
                    "nodes": [
                        {"node_id": "west", "pixel": [10, 50]},
                        {"node_id": "junction", "pixel": [50, 50]},
                    ],
                    "roads": [
                        {
                            "road_id": "yellow_road",
                            "start_node": "west",
                            "end_node": "junction",
                            "width_m": 4.0,
                            "surface": "dirt",
                            "pixel_polyline": [[10, 50], [50, 50]],
                        }
                    ],
                },
            }
            verified_candidate = {
                "task_type": "configuration_only",
                "road_scene_request": {
                    "nodes": [
                        {"node_id": "west", "pixel": [10, 50]},
                        {"node_id": "junction", "pixel": [50, 50]},
                        {"node_id": "north", "pixel": [50, 10]},
                        {"node_id": "south", "pixel": [50, 90]},
                    ],
                    "roads": [
                        {
                            "road_id": "yellow_road",
                            "start_node": "west",
                            "end_node": "junction",
                            "width_m": 4.0,
                            "surface": "dirt",
                            "pixel_polyline": [[10, 50], [50, 50]],
                        },
                        {
                            "road_id": "white_road_north",
                            "start_node": "north",
                            "end_node": "junction",
                            "width_m": 7.0,
                            "surface": "concrete",
                            "pixel_polyline": [[50, 10], [50, 50]],
                        },
                        {
                            "road_id": "white_road_south",
                            "start_node": "junction",
                            "end_node": "south",
                            "width_m": 7.0,
                            "surface": "concrete",
                            "pixel_polyline": [[50, 50], [50, 90]],
                        },
                    ],
                },
            }
            fake_llm = FakeJSONLLM(
                [initial_candidate, verified_candidate]
            )
            planner = PlanGenerationAgent(
                llm_client=fake_llm,
                project_root=root,
            )
            result = planner.run(
                {
                    "natural_language_request": (
                        "Generate the two visible roads and their junction."
                    ),
                    "image_paths": ["scene.png"],
                    "request_template": metadata,
                    "template_policy": "locked",
                    "llm": {"enabled": True},
                    "validation": {"strict_paths": True},
                }
            )

            self.assertEqual(len(fake_llm.calls), 2)
            initial_prompts = (
                fake_llm.calls[0]["system_prompt"]
                + fake_llm.calls[0]["user_prompt"]
            )
            self.assertIn("road_scene_request", initial_prompts)
            self.assertIn("pixel_polyline", initial_prompts)
            self.assertEqual(
                len(result.structured_request["road_scene_request"]["roads"]),
                3,
            )
            self.assertEqual(
                result.structured_request["multimodal_geometry"]["recognition_source"],
                "multimodal_llm",
            )
            self.assertTrue(
                result.validation_report["llm_geometry_verified"]
            )
            self.assertEqual(
                [item["stage"] for item in result.validation_report["llm_trace"]],
                ["initial", "geometry_verification"],
            )
            self.assertEqual(
                [step.agent for step in result.execution_plan.steps],
                ["SGA"],
            )

    def test_pga_raw_image_scene_candidate_requires_review_and_has_preview(self):
        """原始图首次识别应生成候选叠加图，并等待用户确认。"""
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            image_path = root / "raw_scene.png"
            Image.new("RGB", (120, 90), (92, 126, 76)).save(image_path)
            template = {
                "task_type": "configuration_only",
                "road_scene_request": {
                    "map_name": "RawImageScene",
                    "image_path": "raw_scene.png",
                    "image_width_px": 120,
                    "image_height_px": 90,
                    "meters_per_pixel": 0.25,
                    "north_up": True,
                    "geo_anchor": {
                        "latitude": 37.0,
                        "longitude": 121.0,
                        "pixel_x": 60,
                        "pixel_y": 45,
                    },
                },
                "vegetation_request": {
                    "carla_map": "RawImageScene",
                    "image_width_px": 120,
                    "image_height_px": 90,
                    "meters_per_pixel": 0.25,
                    "dry_run": True,
                    "output_dir": "outputs/vegetation",
                },
            }
            candidate = {
                "task_type": "configuration_only",
                "road_scene_request": {
                    "nodes": [
                        {"node_id": "west", "pixel": [5, 70]},
                        {"node_id": "east", "pixel": [115, 65]},
                    ],
                    "roads": [
                        {
                            "road_id": "farm_road",
                            "start_node": "west",
                            "end_node": "east",
                            "width_m": 4.0,
                            "surface": "dirt",
                            "pixel_polyline": [[5, 70], [60, 68], [115, 65]],
                        }
                    ],
                },
                "vegetation_request": {
                    "grass_regions": [
                        {
                            "name": "north_field",
                            "coordinate_mode": "pixel",
                            "polygon": [[8, 8], [54, 8], [52, 55], [10, 58]],
                            "visual_mode": "debug_area",
                        }
                    ],
                    "tree_regions": [
                        {
                            "name": "roadside_tree_belt",
                            "coordinate_mode": "pixel",
                            "polygon": [[80, 5], [90, 5], [105, 60], [95, 60]],
                        }
                    ],
                },
            }
            fake_llm = FakeJSONLLM([candidate, candidate])
            result = PlanGenerationAgent(
                llm_client=fake_llm,
                project_root=root,
            ).run(
                {
                    "natural_language_request": (
                        "Use the visible east-west farm road, the upper-left field, "
                        "and the tree belt beside the sloping road."
                    ),
                    "image_paths": ["raw_scene.png"],
                    "request_template": template,
                    "template_policy": "locked",
                    "llm": {"enabled": True, "model": "test-multimodal"},
                    "validation": {"strict_paths": True},
                }
            )

            review = result.structured_request["scene_review"]
            self.assertEqual(review["status"], "candidate")
            self.assertTrue(review["requires_confirmation"])
            self.assertEqual(review["revision_round"], 1)
            self.assertTrue((root / review["preview_path"]).is_file())
            self.assertEqual(
                result.structured_request["multimodal_geometry"]["input_mode"],
                "raw_satellite_image_with_text_guidance",
            )
            prompt = fake_llm.calls[0]["system_prompt"].lower()
            self.assertIn("original, unannotated satellite image", prompt)
            self.assertIn("spatial description", prompt)

    def test_pga_distinguishes_raw_and_correction_images(self):
        """修订图只作为第二轮纠正依据，不能覆盖原始卫星图角色。"""
        planner = PlanGenerationAgent()
        summary = planner._attachment_summary(
            {
                "image_paths": ["raw.png"],
                "correction_image_paths": ["marked_revision.png"],
            }
        )
        self.assertIn("role=PRIMARY_RAW_SATELLITE_IMAGE path=raw.png", summary)
        self.assertIn(
            "role=OPTIONAL_SECOND_ROUND_CORRECTION_IMAGE path=marked_revision.png",
            summary,
        )

    def test_crewai_flow_blocks_unconfirmed_scene_candidate(self):
        """未经确认的多模态场景候选不得进入 SGA。"""
        state = SimpleNamespace(
            structured_request={
                "scene_review": {
                    "status": "candidate",
                    "requires_confirmation": True,
                    "preview_path": "outputs/review.png",
                }
            }
        )
        flow_like = SimpleNamespace(state=state)
        with self.assertRaises(NeedsUserInputError):
            AgriculturalSimulationFlow._require_scene_confirmation(flow_like)

        state.structured_request["scene_review"].update(
            {"status": "confirmed", "requires_confirmation": False}
        )
        AgriculturalSimulationFlow._require_scene_confirmation(flow_like)

    def test_pga_geometry_repair_remains_model_based(self):
        """视觉复核结果无效时仍应由模型直接修正道路几何。"""
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "scene.png").write_bytes(b"image-placeholder")
            metadata = {
                "task_type": "configuration_only",
                "road_scene_request": {
                    "map_name": "DirectRepairScene",
                    "image_path": "scene.png",
                    "image_width_px": 100,
                    "image_height_px": 100,
                    "meters_per_pixel": 0.2,
                    "north_up": True,
                    "geo_anchor": {
                        "latitude": 37.0,
                        "longitude": 121.0,
                        "pixel_x": 50,
                        "pixel_y": 50,
                    },
                    "geometry_validation": {
                        "endpoint_tolerance_px": 5.0,
                    },
                },
                "road_scene_output_dir": "outputs/direct_repair",
            }
            invalid = {
                "task_type": "configuration_only",
                "road_scene_request": {
                    "nodes": [
                        {"node_id": "west", "pixel": [10, 50]},
                        {"node_id": "east", "pixel": [90, 50]},
                    ],
                    "roads": [
                        {
                            "road_id": "road",
                            "start_node": "west",
                            "end_node": "east",
                            "width_m": 4.0,
                            "surface": "dirt",
                            "pixel_polyline": [[30, 50], [90, 50]],
                        }
                    ],
                },
            }
            repaired = {
                "task_type": "configuration_only",
                "road_scene_request": {
                    "nodes": [
                        {"node_id": "west", "pixel": [10, 50]},
                        {"node_id": "east", "pixel": [90, 50]},
                    ],
                    "roads": [
                        {
                            "road_id": "road",
                            "start_node": "west",
                            "end_node": "east",
                            "width_m": 4.0,
                            "surface": "dirt",
                            "pixel_polyline": [[10, 50], [90, 50]],
                        }
                    ],
                },
            }
            fake_llm = FakeJSONLLM([invalid, invalid, repaired])
            result = PlanGenerationAgent(
                llm_client=fake_llm,
                project_root=root,
            ).run(
                {
                    "natural_language_request": "Generate the visible road.",
                    "image_paths": ["scene.png"],
                    "request_template": metadata,
                    "template_policy": "locked",
                    "llm": {
                        "enabled": True,
                        "max_repair_attempts": 1,
                    },
                    "validation": {"strict_paths": True},
                }
            )

            self.assertEqual(len(fake_llm.calls), 3)
            repair_prompt = fake_llm.calls[2]["system_prompt"]
            self.assertIn("correct road_scene_request directly", repair_prompt)
            self.assertIn("Return no color thresholds", repair_prompt)
            self.assertNotIn("local code will create", repair_prompt)
            self.assertEqual(result.validation_report["llm_repair_attempts"], 1)

    def test_pga_validates_direct_vegetation_polygons(self):
        """PGA 应校验模型直接给出的植被多边形而不读取图像像素。"""
        planner = PlanGenerationAgent()
        request = {
            "task_type": "configuration_only",
            "vegetation_request": {
                "image_width_px": 100,
                "image_height_px": 80,
                "meters_per_pixel": 0.2,
                "grass_regions": [
                    {
                        "name": "grass_1",
                        "coordinate_mode": "pixel",
                        "polygon": [[10, 10], [60, 10], [60, 50], [10, 50]],
                    }
                ],
            },
        }
        errors, _ = planner.validate_request(
            request,
            strict_paths=False,
            require_execution_identifiers=False,
        )
        self.assertEqual(errors, [])

        request["vegetation_request"]["grass_regions"][0]["polygon"][2] = [120, 50]
        errors, _ = planner.validate_request(
            request,
            strict_paths=False,
            require_execution_identifiers=False,
        )
        self.assertTrue(any("outside the image" in item for item in errors))

        request["vegetation_request"]["grass_regions"][0]["polygon"] = [[20, 20]]
        errors, _ = planner.validate_request(
            request,
            strict_paths=False,
            require_execution_identifiers=False,
        )
        self.assertTrue(
            any("at least three points" in item for item in errors)
        )

    def test_pga_accepts_confirmed_road_geometry_without_source_image(self):
        """已保存的结构化道路任务应可在原图不在本机时重复执行。"""
        request = {
            "task_type": "configuration_only",
            "road_scene_request": {
                "map_name": "ConfirmedRoadScene",
                "image_width_px": 100,
                "image_height_px": 80,
                "meters_per_pixel": 0.2,
                "geo_anchor": {
                    "latitude": 37.0,
                    "longitude": 121.0,
                    "pixel_x": 0.0,
                    "pixel_y": 0.0,
                },
                "nodes": [
                    {"node_id": "west", "pixel": [10, 40]},
                    {"node_id": "east", "pixel": [90, 40]},
                ],
                "roads": [{
                    "road_id": "farm_road",
                    "start_node": "west",
                    "end_node": "east",
                    "width_m": 4.0,
                    "surface": "dirt",
                    "pixel_polyline": [[10, 40], [90, 40]],
                }],
            },
        }

        errors, _ = PlanGenerationAgent().validate_request(
            request,
            strict_paths=True,
            require_execution_identifiers=False,
        )
        self.assertEqual(errors, [])

    def test_pga_multimodal_vegetation_is_independently_verified(self):
        """PGA 应在 SGA 执行前用第二次视觉调用复核环境区域。"""
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "scene.png").write_bytes(b"image-placeholder")
            template = {
                "task_type": "scene_enrichment",
                "vegetation_request": {
                    "carla_map": "VegetationScene",
                    "image_width_px": 100,
                    "image_height_px": 80,
                    "meters_per_pixel": 0.2,
                    "dry_run": True,
                    "output_dir": "outputs/vegetation",
                },
            }
            initial = {
                "task_type": "scene_enrichment",
                "vegetation_request": {
                    "tree_regions": [],
                    "grass_regions": [
                        {
                            "name": "field",
                            "coordinate_mode": "pixel",
                            "polygon": [[10, 10], [60, 10], [60, 50], [10, 50]],
                            "visual_mode": "debug_area",
                        }
                    ],
                },
            }
            verified = {
                "task_type": "scene_enrichment",
                "vegetation_request": {
                    "tree_regions": [],
                    "grass_regions": [
                        {
                            "name": "field",
                            "coordinate_mode": "pixel",
                            "polygon": [[8, 8], [62, 8], [62, 52], [8, 52]],
                            "visual_mode": "debug_area",
                        }
                    ],
                },
            }
            fake_llm = FakeJSONLLM([initial, verified])
            result = PlanGenerationAgent(
                llm_client=fake_llm,
                project_root=root,
            ).run(
                {
                    "natural_language_request": (
                        "Use only the upper-left rectangular plot as farmland fill."
                    ),
                    "image_paths": ["scene.png"],
                    "request_template": template,
                    "template_policy": "locked",
                    "llm": {"enabled": True},
                    "validation": {"strict_paths": True},
                }
            )

            self.assertEqual(len(fake_llm.calls), 2)
            self.assertIn("farmland", fake_llm.calls[1]["user_prompt"].lower())
            parser_prompt = fake_llm.calls[0]["system_prompt"].lower()
            self.assertIn("semi-transparent polygon", parser_prompt)
            self.assertIn("original, unannotated satellite image", parser_prompt)
            self.assertIn("tree belts", parser_prompt)
            self.assertIn("individual-tree", parser_prompt)
            self.assertEqual(
                result.structured_request["vegetation_request"]["grass_regions"][0]["polygon"],
                [[8, 8], [62, 8], [62, 52], [8, 52]],
            )
            self.assertEqual(
                result.structured_request["multimodal_geometry"]["geometry_types"],
                ["vegetation_regions"],
            )
            self.assertEqual(
                [step.step_id for step in result.execution_plan.steps],
                ["sga_vegetation"],
            )

    def test_crewai_store_keeps_checkpoints_and_agent_records(self):
        """运行存储只记录状态与检查点，不负责决定下一步执行对象。"""
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            store = ApplicationRunLedger(root / "runtime.sqlite3", root / "runs")
            task_id = store.create_task({"task_type": "configuration_only"})
            run_id = store.start_agent_run(
                task_id,
                "mca_configuration",
                "MCA",
                "generate_machinery_template",
                1,
            )
            output = {"machinery_type": "tractor"}
            store.finish_agent_run(run_id, "SUCCEEDED", output=output)
            checkpoint = store.save_checkpoint(task_id, "mca_configuration", output)

            self.assertTrue(checkpoint.is_file())
            self.assertEqual(
                store.completed_step_outputs(task_id),
                {"mca_configuration": output},
            )

    def test_domain_service_forwards_tracking_dataset_directories(self):
        """EEA 执行服务应完整接收优化集、测试集和标准化输出目录。"""
        service = DomainExecutionService()
        service._experiment_execution_agent = Mock()
        service._experiment_execution_agent.run_trajectory_tracking.return_value = {
            "optimization_enabled": True
        }
        result = service.invoke(
            "EEA",
            "run_trajectory_tracking",
            {
                "task_type": "trajectory_tracking",
                "experiment": {
                    "parameter_optimization_set_dir": "data/optimization",
                    "independent_test_set_dir": "data/test",
                    "processed_dataset_dir": "outputs/processed",
                    "config": {"controller_optimization": {"enabled": True}},
                },
            },
        )
        self.assertTrue(result["optimization_enabled"])
        call = service._experiment_execution_agent.run_trajectory_tracking.call_args
        self.assertEqual(call.kwargs["parameter_optimization_set_dir"], "data/optimization")
        self.assertEqual(call.kwargs["independent_test_set_dir"], "data/test")
        self.assertEqual(call.kwargs["processed_dataset_dir"], "outputs/processed")

    def test_worker_application_executes_whitelisted_step(self):
        """CARLA 执行端应异步执行领域步骤并持久化可查询终态。"""
        with tempfile.TemporaryDirectory() as temporary_dir:
            repository = WorkerTaskRepository(Path(temporary_dir) / "tasks")
            application = CarlaWorkerApplication(repository)
            application.executor.submit = Mock(
                side_effect=lambda callback, *args: callback(*args)
            )
            response = application.submit(
                {
                    "worker_task_id": "task-1--cia--1",
                    "manager_task_id": "task-1",
                    "step_id": "cia_configuration",
                    "agent": "CIA",
                    "operation": "prepare_controller",
                    "request": {
                        "task_type": "trajectory_tracking",
                        "experiment": {"config": {}},
                    },
                }
            )
            self.assertEqual(response["worker_task_id"], "task-1--cia--1")
            record = repository.get("task-1--cia--1")
            self.assertEqual(record["status"], "SUCCEEDED")
            self.assertTrue(record["output"]["validated"])


if __name__ == "__main__":
    unittest.main()
