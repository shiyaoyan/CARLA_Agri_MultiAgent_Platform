"""验证 CrewAI 架构中的智能体层级与执行边界。"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from agri_carla.agents import (
    SceneGenerationAgent,
    ControllerInterfaceAgent,
    MachineryConfigurationAgent,
    ExperimentExecutionAgent,
    PlanGenerationAgent,
)
from agri_carla.agents.plan_generation import PlanGenerationAgent as Planner
from agri_carla.crewai_runtime.config import CrewAIManagerConfig, ExecutionTargetConfig
from agri_carla.crewai_runtime.flow import AgriculturalSimulationFlow
from agri_carla.crewai_runtime.tools import PlatformAgentTool
from agri_carla.evaluation.performance_evaluation_subagent import PerformanceEvaluationSubagent
from agri_carla.execution import DomainExecutionService
from agri_carla.trajectory.preprocessing_subagent import TrajectoryPreprocessingSubagent


class AgentArchitectureTests(unittest.TestCase):
    """检查同层智能体、EEA 子智能体和 CrewAI 执行计划。"""

    def test_peer_level_agents_have_independent_public_classes(self):
        expected_modules = {
            SceneGenerationAgent: "agri_carla.agents.scene_generation",
            MachineryConfigurationAgent: "agri_carla.agents.machinery_configuration",
            ExperimentExecutionAgent: "agri_carla.agents.experiment_execution",
            ControllerInterfaceAgent: "agri_carla.controllers.interface_agent",
            PlanGenerationAgent: "agri_carla.agents.plan_generation",
        }
        for agent_class, expected_module in expected_modules.items():
            with self.subTest(agent=agent_class.__name__):
                self.assertEqual(agent_class.__module__, expected_module)

    def test_tps_and_pes_remain_eea_subagents(self):
        eea = ExperimentExecutionAgent()
        self.assertIsInstance(
            eea.trajectory_preprocessing_subagent,
            TrajectoryPreprocessingSubagent,
        )
        self.assertIs(
            eea.performance_evaluation_subagent_class,
            PerformanceEvaluationSubagent,
        )

    def test_cia_is_a_peer_service_used_by_eea(self):
        eea = ExperimentExecutionAgent()
        self.assertIs(eea.controller_agent_class, ControllerInterfaceAgent)

    def test_execution_plan_places_cia_before_eea(self):
        plan = Planner().build_execution_plan(
            {
                "task_type": "trajectory_tracking",
                "experiment": {"trajectory_csv": "trajectory.csv", "config": {}},
            }
        )
        self.assertEqual(
            [(step.agent, step.step_id) for step in plan.steps],
            [("CIA", "cia_configuration"), ("EEA", "eea_tracking")],
        )
        self.assertEqual(plan.steps[1].depends_on, ["cia_configuration"])

    def test_domain_service_is_not_a_top_level_orchestrator(self):
        service = DomainExecutionService()
        self.assertIsNone(service._scene_generation_agent)
        self.assertIsNone(service._machinery_agent)
        self.assertIsNone(service._experiment_execution_agent)
        self.assertFalse(hasattr(service, "run"))
        self.assertFalse(hasattr(service, "resume"))

    def test_same_host_worker_defaults_to_eea_only(self):
        with patch.dict(
            "os.environ",
            {
                "AGRI_CARLA_WORKER_URL": "",
                "AGRI_CARLA_WORKER_AGENTS": "",
            },
        ):
            target = ExecutionTargetConfig.from_request({})
        self.assertEqual(target.worker_url, "http://127.0.0.1:8765")
        self.assertTrue(target.uses_worker("EEA"))
        self.assertFalse(target.uses_worker("SGA"))
        self.assertFalse(target.uses_worker("CIA"))

    def test_same_host_worker_rejects_non_loopback_url(self):
        with self.assertRaisesRegex(ValueError, "same-host loopback"):
            ExecutionTargetConfig.from_request(
                {"execution": {"worker_url": "http://192.168.1.50:8765"}}
            )

    def test_worker_agent_selection_uses_same_host_environment(self):
        target = ExecutionTargetConfig.from_request(
            {"execution": {"worker_agents": ["SGA", "EEA"]}}
        )
        self.assertEqual(target.worker_agents, {"SGA", "EEA"})
        self.assertTrue(target.uses_worker("SGA"))
        self.assertTrue(target.uses_worker("EEA"))

    def test_crewai_manager_has_the_project_default_model(self):
        settings = CrewAIManagerConfig.from_request({})
        self.assertEqual(settings.model, "openai/gpt-5.6-sol")

    def test_crewai_is_the_only_top_level_workflow(self):
        root = Path(__file__).resolve().parents[1]
        flow_source = (root / "agri_carla/crewai_runtime/flow.py").read_text(
            encoding="utf-8"
        )
        crew_source = (root / "agri_carla/crewai_runtime/crew.py").read_text(
            encoding="utf-8"
        )
        listener_source = (
            root / "agri_carla/crewai_runtime/listeners.py"
        ).read_text(encoding="utf-8")
        runtime_source = (
            root / "agri_carla/crewai_runtime/runtime.py"
        ).read_text(encoding="utf-8")
        self.assertIn("@persist", flow_source)
        self.assertIn("@start()", flow_source)
        self.assertIn("@listen(plan_task)", flow_source)
        self.assertIn("class AgriculturalSimulationFlow", flow_source)
        self.assertIn("@CrewBase", crew_source)
        self.assertIn("@agent", crew_source)
        self.assertIn("@task", crew_source)
        self.assertIn("@crew", crew_source)
        self.assertIn("Agent(", crew_source)
        self.assertIn("Task(", crew_source)
        self.assertIn("Crew(", crew_source)
        self.assertIn("BaseEventListener", listener_source)
        self.assertIn("FlowStartedEvent", listener_source)
        self.assertIn('"id": task_id', runtime_source)
        self.assertNotIn("restore_from_state_id=", runtime_source)

    def test_platform_tool_preserves_the_exact_domain_result(self):
        class RuntimeStub:
            def invoke(self, agent, **kwargs):
                return {
                    "agent": agent,
                    "operation": kwargs["operation"],
                    "nested": {"valid": True},
                }

        tool = PlatformAgentTool(
            "SGA",
            RuntimeStub(),
            task_id="task-1",
            step_id="sga-scene",
            operation="generate_scene",
            attempt=1,
            payload={"task_type": "configuration_only"},
        )
        tool._run()
        self.assertEqual(tool.call_count, 1)
        self.assertEqual(
            tool.last_result,
            {
                "agent": "SGA",
                "operation": "generate_scene",
                "nested": {"valid": True},
            },
        )
        self.assertIsNone(tool.last_error)

    def test_platform_tool_preserves_the_original_error_for_flow_retry(self):
        class RuntimeStub:
            def invoke(self, agent, **kwargs):
                raise RuntimeError(
                    "LLM connection failed: SSL: UNEXPECTED_EOF_WHILE_READING"
                )

        tool = PlatformAgentTool(
            "PGA",
            RuntimeStub(),
            task_id="task-2",
            step_id="pga",
            operation="parse_validate_and_plan",
            attempt=1,
            payload={},
        )
        with self.assertRaisesRegex(RuntimeError, "UNEXPECTED_EOF"):
            tool._run()
        self.assertEqual(tool.call_count, 1)
        self.assertIsNone(tool.last_result)
        self.assertIsInstance(tool.last_error, RuntimeError)
        self.assertTrue(AgriculturalSimulationFlow._is_retryable(tool.last_error))


if __name__ == "__main__":
    unittest.main()
