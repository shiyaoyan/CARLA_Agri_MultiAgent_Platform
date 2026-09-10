"""CrewAI Agent 可调用的白名单平台工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, PrivateAttr

from agri_carla.agents.plan_generation import PlanGenerationAgent
from agri_carla.execution import DomainExecutionService
from agri_carla.orchestration import ExecutionContext
from agri_carla.worker import CarlaWorkerClient, CarlaWorkerConfig

from .config import ExecutionTargetConfig


class PlatformToolInput(BaseModel):
    """无模型可修改字段的工具输入结构。"""

    # 任务上下文在 Tool 创建时由 Flow 绑定。空 schema 可以防止模型修改任务
    # 编号、白名单操作或结构化请求，同时仍遵循 CrewAI BaseTool 的 Pydantic
    # 输入约定。
    pass


class CrewAIToolRuntime:
    """为 CrewAI 工具提供 PGA、运行记录和同机隔离执行能力。"""

    def __init__(
        self,
        *,
        project_root: str | Path,
        store,
        execution_target: ExecutionTargetConfig,
    ):
        self.project_root = Path(project_root).resolve()
        self.store = store
        self.execution_target = execution_target
        self.domain_service = DomainExecutionService()

    def invoke(
        self,
        agent: str,
        *,
        task_id: str,
        step_id: str,
        operation: str,
        attempt: int,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if agent == "PGA":
            planning = PlanGenerationAgent(project_root=self.project_root).run(payload)
            return planning.to_dict()

        context = ExecutionContext(task_id, self.store)
        if self.execution_target.uses_worker(agent):
            response = self._worker_client().execute(
                worker_task_id=self._worker_task_id(task_id, step_id, attempt),
                manager_task_id=task_id,
                step_id=step_id,
                agent=agent,
                operation=operation,
                request=payload,
            )
            self._replay_worker_events(context, response.get("events", []))
            return dict(response.get("output", {}))

        return self.domain_service.invoke(
            agent,
            operation,
            payload,
            context=context,
            step_id=step_id,
        )

    def _worker_client(self) -> CarlaWorkerClient:
        target = self.execution_target
        return CarlaWorkerClient(
            CarlaWorkerConfig(
                base_url=target.worker_url,
                token_env=target.token_env,
                poll_interval_s=target.poll_interval_s,
                timeout_s=target.timeout_s,
            )
        )

    @staticmethod
    def _worker_task_id(task_id: str, step_id: str, attempt: int) -> str:
        safe_task = "".join(char if char.isalnum() or char in "_.-" else "-" for char in task_id)
        safe_step = "".join(char if char.isalnum() or char in "_.-" else "-" for char in step_id)
        return "%s--%s--%d" % (safe_task, safe_step, attempt)

    @staticmethod
    def _replay_worker_events(context: ExecutionContext, events: list[dict[str, Any]]) -> None:
        run_ids: dict[str, str] = {}
        for event in events:
            event_type = event.get("type")
            if event_type == "run_started":
                run_ids[str(event["run_id"])] = context.start(
                    str(event["agent"]),
                    str(event["step_id"]),
                    str(event["operation"]),
                    attempt=int(event.get("attempt", 1)),
                    input_payload=dict(event.get("input", {})),
                )
            elif event_type == "run_succeeded":
                local_run_id = run_ids.get(str(event.get("run_id", "")))
                if local_run_id:
                    context.succeed(local_run_id, event.get("output"))
            elif event_type == "run_failed":
                local_run_id = run_ids.get(str(event.get("run_id", "")))
                if local_run_id:
                    context.fail(
                        local_run_id,
                        RuntimeError(str(event.get("error_message", "Worker failure."))),
                    )
            elif event_type == "progress":
                context.progress(
                    str(event.get("agent", "EEA")),
                    str(event.get("step_id", "")),
                    str(event.get("message", "Worker progress update.")),
                    **dict(event.get("payload", {})),
                )
            elif event_type == "artifact":
                context.artifact(
                    str(event.get("path", "")),
                    agent=str(event.get("agent", "EEA")),
                    step_id=str(event.get("step_id", "")),
                    artifact_type=str(event.get("artifact_type", "file")),
                    metadata=dict(event.get("metadata", {})),
                )


class PlatformAgentTool(BaseTool):
    """将一个 CrewAI 角色限制为单一项目能力入口。"""

    name: str = "Agricultural platform operation"
    description: str = "Execute one validated agricultural CARLA platform operation."
    args_schema: Type[BaseModel] = PlatformToolInput
    _agent_id: str = PrivateAttr()
    _runtime: CrewAIToolRuntime = PrivateAttr()
    _invocation: dict[str, Any] = PrivateAttr()
    _last_result: dict[str, Any] | None = PrivateAttr(default=None)
    _last_error: BaseException | None = PrivateAttr(default=None)
    _call_count: int = PrivateAttr(default=0)

    def __init__(
        self,
        agent_id: str,
        runtime: CrewAIToolRuntime,
        *,
        task_id: str,
        step_id: str,
        operation: str,
        attempt: int,
        payload: dict[str, Any],
    ):
        super().__init__(
            name="%s platform tool" % agent_id,
            description=(
                "Execute the validated %s operation. Call exactly once and return "
                "the result without modification." % agent_id
            ),
            result_as_answer=True,
        )
        self._agent_id = agent_id
        self._runtime = runtime
        self._invocation = {
            "task_id": task_id,
            "step_id": step_id,
            "operation": operation,
            "attempt": attempt,
            "payload": payload,
        }

    @property
    def last_result(self) -> dict[str, Any] | None:
        """返回本次 Crew 执行中平台工具产生的原始结构化结果。"""
        return self._last_result

    @property
    def last_error(self) -> BaseException | None:
        """返回平台工具的原始异常，避免 CrewAI 将其改写成普通文本。"""
        return self._last_error

    @property
    def call_count(self) -> int:
        """返回 CrewAI Agent 实际调用平台工具的次数。"""
        return self._call_count

    def _run(self) -> str:
        invocation = self._invocation
        self._call_count += 1
        self._last_result = None
        self._last_error = None
        try:
            result = self._runtime.invoke(
                self._agent_id,
                task_id=str(invocation["task_id"]),
                step_id=str(invocation["step_id"]),
                operation=str(invocation["operation"]),
                attempt=int(invocation["attempt"]),
                payload=dict(invocation["payload"]),
            )
        except BaseException as exc:
            self._last_error = exc
            raise
        self._last_result = dict(result)
        return json.dumps(result, ensure_ascii=False, default=str)
