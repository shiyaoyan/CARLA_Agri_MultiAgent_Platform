"""把已校验任务步骤映射到项目现有农业功能实现。"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class DomainExecutionService:
    """执行一个白名单内的农业功能步骤，不承担顶层流程编排。"""

    def __init__(self):
        self._scene_generation_agent = None
        self._machinery_agent = None
        self._experiment_execution_agent = None

    def invoke(
        self,
        agent: str,
        operation: str,
        request: dict[str, Any],
        *,
        context=None,
        step_id: str = "",
    ) -> dict[str, Any]:
        """调用指定智能体能力，并返回可序列化结果。"""
        if agent == "SGA" and operation == "generate_scene":
            output_dir = request.get("road_scene_output_dir", "outputs/road_scene")
            result = self._sga().run(
                request["road_scene_request"],
                output_dir,
                context=context,
            )
            return dict(result.__dict__)

        if agent == "SGA" and operation == "place_vegetation":
            return self._sga().run_vegetation(
                request["vegetation_request"],
                context=context,
            )

        if agent == "MCA" and operation == "generate_machinery_template":
            machinery = request["machinery"]
            template = self._mca().run(
                machinery.get("type", "tractor"),
                model_file=machinery.get("model_file", ""),
                output_file=machinery.get("output_file"),
                context=context,
            )
            if context is not None and machinery.get("output_file"):
                output_path = Path(machinery["output_file"])
                if output_path.is_file():
                    context.artifact(
                        output_path,
                        agent="MCA",
                        step_id=step_id or "mca_configuration",
                        artifact_type="machinery_parameter_template",
                    )
            return template.to_dict()

        if agent == "CIA" and operation == "prepare_controller":
            from agri_carla.controllers.interface_agent import (
                ControllerInterfaceAgent,
            )

            return ControllerInterfaceAgent.prepare_experiment_configuration(
                str(request.get("task_type", "")),
                request.get("experiment"),
            )

        if agent == "EEA" and operation == "run_trajectory_tracking":
            from agri_carla.simulation.tracking_experiment import (
                TrackingExperimentConfig,
            )

            experiment = request.get("experiment", {})
            tracking_config = TrackingExperimentConfig.from_dict(
                experiment.get("config", {})
            )
            return self._eea().run_trajectory_tracking(
                trajectory_csv=experiment.get("trajectory_csv"),
                processed_csv=experiment.get("processed_csv"),
                config=tracking_config,
                parameter_optimization_set_dir=experiment.get(
                    "parameter_optimization_set_dir"
                ),
                independent_test_set_dir=experiment.get(
                    "independent_test_set_dir"
                ),
                processed_dataset_dir=experiment.get("processed_dataset_dir"),
                context=context,
            )

        if agent == "EEA" and operation == "run_full_process_navigation":
            from agri_carla.simulation.full_process_experiment import (
                FullProcessExperimentConfig,
            )

            experiment = request.get("experiment", {})
            full_process_config = FullProcessExperimentConfig(
                **experiment.get("config", {})
            )
            return self._eea().run_full_process_navigation(
                full_process_config,
                context=context,
            )

        raise ValueError("Unsupported execution step: %s/%s" % (agent, operation))

    def _sga(self):
        if self._scene_generation_agent is None:
            from agri_carla.agents.scene_generation import (
                SceneGenerationAgent,
            )

            self._scene_generation_agent = SceneGenerationAgent()
        return self._scene_generation_agent

    def _mca(self):
        if self._machinery_agent is None:
            from agri_carla.agents.machinery_configuration import (
                MachineryConfigurationAgent,
            )

            self._machinery_agent = MachineryConfigurationAgent()
        return self._machinery_agent

    def _eea(self):
        if self._experiment_execution_agent is None:
            from agri_carla.agents.experiment_execution import (
                ExperimentExecutionAgent,
            )

            self._experiment_execution_agent = ExperimentExecutionAgent()
        return self._experiment_execution_agent
