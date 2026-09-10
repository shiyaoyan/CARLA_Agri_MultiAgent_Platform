"""使用 CrewAI 官方声明式 API 定义平台功能 Crew。"""

from __future__ import annotations

from typing import Any

from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task

from .config import CrewAIManagerConfig
from .listeners import CrewAIExecutionMetadata, crewai_execution_scope
from .tools import CrewAIToolRuntime, PlatformAgentTool


class _PlatformCrewMixin:
    """为五个声明式 Crew 提供一致的 Agent、Task 与 Crew 构造参数。"""

    # CrewAI 的 @CrewBase 会在类构造阶段把 YAML 路径替换为解析后的配置字典。
    agents_config: Any
    tasks_config: Any
    agents: list[Agent]
    tasks: list[Task]

    def __init__(self, *, llm: LLM, tool: PlatformAgentTool, verbose: bool):
        self._llm = llm
        self._tool = tool
        self._verbose = verbose

    def _build_agent(self, key: str) -> Agent:
        config = dict(self.agents_config[key])
        config["verbose"] = self._verbose
        return Agent(config=config, tools=[self._tool], llm=self._llm)

    def _build_task(self, key: str, assigned_agent: Agent) -> Task:
        return Task(config=dict(self.tasks_config[key]), agent=assigned_agent)

    def _build_crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=self._verbose,
            cache=False,
            memory=False,
            share_crew=False,
        )

    def crew(self) -> Crew:
        """由各个 `@CrewBase` 子类声明式实现。"""
        raise NotImplementedError


@CrewBase
class PlanGenerationCrew(_PlatformCrewMixin):
    """执行一次 PGA 解析、校验与计划生成任务。"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def pga(self) -> Agent:
        return self._build_agent("pga")

    @task
    def pga_task(self) -> Task:
        return self._build_task("pga", self.pga())

    @crew
    def crew(self) -> Crew:
        return self._build_crew()


@CrewBase
class SceneGenerationCrew(_PlatformCrewMixin):
    """执行一次 SGA 场景生成或环境填充任务。"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def sga(self) -> Agent:
        return self._build_agent("sga")

    @task
    def sga_task(self) -> Task:
        return self._build_task("sga", self.sga())

    @crew
    def crew(self) -> Crew:
        return self._build_crew()


@CrewBase
class MachineryConfigurationCrew(_PlatformCrewMixin):
    """执行一次 MCA 农机配置任务。"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def mca(self) -> Agent:
        return self._build_agent("mca")

    @task
    def mca_task(self) -> Task:
        return self._build_task("mca", self.mca())

    @crew
    def crew(self) -> Crew:
        return self._build_crew()


@CrewBase
class ControllerInterfaceCrew(_PlatformCrewMixin):
    """执行一次 CIA 控制配置任务。"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def cia(self) -> Agent:
        return self._build_agent("cia")

    @task
    def cia_task(self) -> Task:
        return self._build_task("cia", self.cia())

    @crew
    def crew(self) -> Crew:
        return self._build_crew()


@CrewBase
class ExperimentExecutionCrew(_PlatformCrewMixin):
    """执行一次 EEA 仿真实验任务。"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def eea(self) -> Agent:
        return self._build_agent("eea")

    @task
    def eea_task(self) -> Task:
        return self._build_task("eea", self.eea())

    @crew
    def crew(self) -> Crew:
        return self._build_crew()


class AgriculturalAgentCrewRunner:
    """按 Flow 计划选择一个官方 CrewBase Crew 并执行。"""

    _CREWS = {
        "PGA": PlanGenerationCrew,
        "SGA": SceneGenerationCrew,
        "MCA": MachineryConfigurationCrew,
        "CIA": ControllerInterfaceCrew,
        "EEA": ExperimentExecutionCrew,
    }

    def __init__(
        self,
        *,
        settings: CrewAIManagerConfig,
        tool_runtime: CrewAIToolRuntime,
    ):
        self.settings = settings
        self.tool_runtime = tool_runtime
        llm_options: dict[str, Any] = {
            "model": settings.model,
            "api_key": settings.api_key(),
            "timeout": 120.0,
            "max_retries": 2,
        }
        if settings.base_url:
            llm_options["base_url"] = settings.base_url
        self.llm = LLM(**llm_options)

    def run(
        self,
        agent_id: str,
        *,
        task_id: str,
        step_id: str,
        operation: str,
        attempt: int,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """通过声明式 Agent、Task、Crew 和自定义 Tool 执行计划步骤。"""
        try:
            crew_class = self._CREWS[agent_id]
        except KeyError as exc:
            raise ValueError("CrewAI agent is not registered: %s" % agent_id) from exc

        tool = PlatformAgentTool(
            agent_id,
            self.tool_runtime,
            task_id=task_id,
            step_id=step_id,
            operation=operation,
            attempt=attempt,
            payload=payload,
        )
        configured_crew = crew_class(
            llm=self.llm,
            tool=tool,
            verbose=self.settings.verbose,
        ).crew()
        inputs = {
            "task_id": task_id,
            "step_id": step_id,
            "operation": operation,
            "attempt": attempt,
        }
        metadata = CrewAIExecutionMetadata(
            task_id=task_id,
            step_id=step_id,
            agent=agent_id,
            operation=operation,
            attempt=attempt,
            ledger=self.tool_runtime.store,
        )
        with crewai_execution_scope(metadata):
            output = configured_crew.kickoff(inputs=inputs)
        if tool.last_result is not None:
            return dict(tool.last_result)
        if tool.last_error is not None:
            raise tool.last_error
        raw = getattr(output, "raw", output)
        raise RuntimeError(
            "CrewAI %s task completed without calling its bound platform tool. "
            "Crew output was: %s" % (agent_id, str(raw)[:500])
        )
