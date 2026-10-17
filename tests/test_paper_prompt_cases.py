"""Regression tests for prompt-first reproduction of every paper function."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from agri_carla.agents.plan_generation import PlanGenerationAgent
from agri_carla.paper_cases import PaperCaseCatalog


ROOT = Path(__file__).resolve().parents[1]


class FakeJSONLLM:
    def __init__(self, responses):
        self.responses = [copy.deepcopy(item) for item in responses]
        self.calls = []

    def generate_json(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("No fake LLM response remains.")
        return self.responses.pop(0)


class PaperPromptCaseTests(unittest.TestCase):
    def setUp(self):
        self.catalog = PaperCaseCatalog(project_root=ROOT)

    def test_catalog_covers_all_paper_functions(self):
        self.assertEqual(
            [
                "scene_generation",
                "machinery_configuration",
                "section_3_2_tracking",
                "section_3_2_bayesian_optimization",
                "section_3_3_full_process",
            ],
            [case.case_id for case in self.catalog.list_cases()],
        )
        for case in self.catalog.list_cases():
            self.assertGreater(len(self.catalog.prompt_text(case.case_id)), 80)

    def test_case_payloads_cannot_bypass_the_llm(self):
        for case in self.catalog.list_cases():
            images = ["owned_satellite_image.png"] * case.minimum_images
            payload = self.catalog.build_payload(
                case.case_id,
                image_paths=images,
            )
            self.assertIn("natural_language_request", payload)
            self.assertNotIn("task_type", payload)
            self.assertNotIn("structured_request", payload)
            self.assertNotIn("structured_request_file", payload)
            self.assertNotIn("request_template", payload)

    def test_non_visual_prompts_reach_pga_and_plan_expected_agents(self):
        candidates = {
            "machinery_configuration": {
                "task_type": "configuration_only",
                "machinery": {
                    "type": "tractor",
                    "model_file": "vehicle.tractor.tractor",
                    "output_file": (
                        "outputs/paper_cases/machinery_configuration/"
                        "df1204_tractor_physics.json"
                    ),
                },
            },
            "section_3_2_tracking": self._tracking_candidate(),
            "section_3_2_bayesian_optimization": self._json_request(
                "configs/yantai_tracking_optimization_request.json"
            ),
            "section_3_3_full_process": self._json_request(
                "configs/yantai_full_process_agent_request.json"
            ),
        }
        for case_id, candidate in candidates.items():
            with self.subTest(case=case_id):
                fake_llm = FakeJSONLLM([candidate])
                payload = self.catalog.build_payload(case_id)
                payload["validation"] = {"strict_paths": False}
                planning = PlanGenerationAgent(
                    llm_client=fake_llm,
                    project_root=ROOT,
                ).run(payload)
                result = planning.to_dict()
                validation = self.catalog.validate_result(case_id, result)
                self.assertEqual("llm", planning.source)
                self.assertTrue(validation["valid"])
                self.assertEqual(1, len(fake_llm.calls))
                self.assertIn(
                    self.catalog.prompt_text(case_id),
                    fake_llm.calls[0]["user_prompt"],
                )

    def test_scene_prompt_reaches_multimodal_pga_and_plans_both_sga_steps(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            image_path = root / "owned_satellite_image.png"
            Image.new("RGB", (100, 80), "white").save(image_path)
            candidate = {
                "task_type": "configuration_only",
                "road_scene_output_dir": (
                    "outputs/paper_cases/scene_generation/road_scene"
                ),
                "road_scene_request": {
                    "map_name": "PromptGeneratedAgriculturalScene",
                    "image_width_px": 100,
                    "image_height_px": 80,
                    "meters_per_pixel": 0.09,
                    "north_up": True,
                    "geo_anchor": {
                        "latitude": 37.43327171,
                        "longitude": 121.5127619,
                        "pixel_x": 50,
                        "pixel_y": 40,
                    },
                    "nodes": [
                        {"node_id": "west", "pixel": [10, 40]},
                        {"node_id": "east", "pixel": [90, 40]},
                    ],
                    "roads": [
                        {
                            "road_id": "farm_road",
                            "start_node": "west",
                            "end_node": "east",
                            "width_m": 5.0,
                            "surface": "dirt",
                            "pixel_polyline": [[10, 40], [90, 40]],
                        }
                    ],
                },
                "vegetation_request": {
                    "carla_map": "PromptGeneratedAgriculturalScene",
                    "image_width_px": 100,
                    "image_height_px": 80,
                    "meters_per_pixel": 0.09,
                    "north_up": True,
                    "output_dir": (
                        "outputs/paper_cases/scene_generation/vegetation"
                    ),
                    "tree_regions": [],
                    "grass_regions": [
                        {
                            "name": "field",
                            "kind": "grass",
                            "coordinate_mode": "pixel",
                            "polygon": [[10, 50], [90, 50], [90, 70], [10, 70]],
                            "visual_mode": "debug_area",
                        }
                    ],
                },
            }
            fake_llm = FakeJSONLLM([candidate, candidate])
            payload = self.catalog.build_payload(
                "scene_generation",
                image_paths=[str(image_path)],
            )
            payload["scene_review"] = {
                "preview_path": str(root / "candidate_preview.png")
            }
            planning = PlanGenerationAgent(
                llm_client=fake_llm,
                project_root=ROOT,
            ).run(payload)
            validation = self.catalog.validate_result(
                "scene_generation",
                planning.to_dict(),
            )
            self.assertTrue(validation["valid"])
            self.assertEqual(2, len(fake_llm.calls))
            self.assertEqual([str(image_path)], fake_llm.calls[0]["image_paths"])
            self.assertTrue((root / "candidate_preview.png").is_file())

    def test_execution_contract_rejects_model_invented_fields(self):
        request = self._tracking_candidate()
        request["experiment"]["config"]["simulation_step_s"] = 0.05
        request["experiment"]["config"]["controller"]["pid"][
            "fake_gain"
        ] = 1.0
        errors, _ = PlanGenerationAgent(project_root=ROOT).validate_request(
            request,
            strict_paths=False,
        )
        self.assertTrue(
            any("simulation_step_s" in error for error in errors),
            errors,
        )
        self.assertTrue(any("fake_gain" in error for error in errors), errors)

    @staticmethod
    def _json_request(relative_path: str) -> dict:
        data = json.loads((ROOT / relative_path).read_text(encoding="utf-8"))
        data.pop("natural_language_request", None)
        data.pop("flow_report", None)
        return data

    def _tracking_candidate(self) -> dict:
        data = self._json_request("configs/yantai_tracking_experiment_request.json")
        data.pop("road_scene_request", None)
        data.pop("road_scene_output_dir", None)
        data["experiment"]["config"]["controller"] = {
            "max_speed_kmh": 8.0,
            "k_pos": 0.8,
            "pid": {"kp": 0.95, "ki": 0.22, "kd": 0.03},
            "pure_pursuit": {
                "wheelbase": 2.5,
                "max_steer_angle_deg": 48.0,
                "fwd_lookahead_min": 1.0,
                "fwd_lookahead_gain": 0.2,
                "rev_lookahead_min": 0.5,
                "rev_lookahead_gain": 0.1,
            },
        }
        return data


if __name__ == "__main__":
    unittest.main()
