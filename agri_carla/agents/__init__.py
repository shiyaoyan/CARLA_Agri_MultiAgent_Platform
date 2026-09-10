"""农业 CARLA 平台对外公开的智能体接口。

PGA、SGA、MCA、CIA 和 EEA 由 CrewAI Flow 管理；TPS 与 PES 保持为 EEA
的下属智能体。这里采用延迟导入，避免只做场景或配置工作时加载 CARLA 依赖。
"""

__all__ = [
    "PlanGenerationAgent",
    "SceneGenerationAgent",
    "MachineryConfigurationAgent",
    "ExperimentExecutionAgent",
    "ControllerInterfaceAgent",
]


def __getattr__(name):
    """首次访问公开智能体名称时再导入对应实现。"""
    if name == "PlanGenerationAgent":
        from .plan_generation import PlanGenerationAgent

        return PlanGenerationAgent
    if name == "SceneGenerationAgent":
        from .scene_generation import SceneGenerationAgent

        return SceneGenerationAgent
    if name == "MachineryConfigurationAgent":
        from .machinery_configuration import MachineryConfigurationAgent

        return MachineryConfigurationAgent
    if name == "ExperimentExecutionAgent":
        from .experiment_execution import ExperimentExecutionAgent

        return ExperimentExecutionAgent
    if name == "ControllerInterfaceAgent":
        from agri_carla.controllers.interface_agent import ControllerInterfaceAgent

        return ControllerInterfaceAgent
    raise AttributeError(name)
