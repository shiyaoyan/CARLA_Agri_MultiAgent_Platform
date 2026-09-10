"""CrewAI Flow 的创建、运行、查询和恢复入口。"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from agri_carla.orchestration import ApplicationRunLedger, RunStatus

from .config import CrewAIManagerConfig, ExecutionTargetConfig
from .crew import AgriculturalAgentCrewRunner
from .flow import AgriculturalSimulationFlow
from .listeners import CrewAIExecutionMetadata, crewai_execution_scope
from .tools import CrewAIToolRuntime


class CrewAIRuntime:
    """创建唯一顶层 Flow，并向 CLI 提供稳定的运行管理接口。"""

    def __init__(
        self,
        *,
        store: ApplicationRunLedger | None = None,
        project_root: str | Path | None = None,
        retry_delay_s: float = 1.0,
    ):
        self.store = store or ApplicationRunLedger()
        self.project_root = Path(project_root or Path.cwd()).resolve()
        self.retry_delay_s = retry_delay_s
        self.last_task_id: str | None = None

    def run(self, request: dict[str, Any], *, plan_only: bool = False) -> dict[str, Any]:
        flow = self._build_flow(request)
        # 新 Flow 已由 CrewAI 自动分配唯一 state id。直接复用该编号可避免把
        # 首次执行误判为“按既有 id 恢复”的查询。
        task_id = str(flow.state.id or uuid.uuid4())
        if not flow.state.id:
            flow.state.id = task_id
        self.store.create_task(request, task_id=task_id)
        self.last_task_id = task_id
        metadata = self._flow_metadata(task_id, "run")
        try:
            with crewai_execution_scope(metadata):
                return flow.kickoff(
                    inputs={
                        "task_id": task_id,
                        "original_request": request,
                        "plan_only": plan_only,
                    }
                )
        finally:
            self._record_flow_state(task_id, flow)

    def resume(self, task_id: str) -> dict[str, Any]:
        task = self.store.get_task(task_id)
        if task["status"] == RunStatus.SUCCEEDED.value and task["final_output"]:
            return task["final_output"]
        request = dict(task["original_input"])
        flow = self._build_flow(request)
        self.last_task_id = task_id
        self.store.update_task(task_id, RunStatus.READY)
        metadata = self._flow_metadata(task_id, "resume")
        try:
            # inputs["id"] 表示在原 Flow lineage 上恢复；
            # restore_from_state_id 会创建分叉，不适用于原任务续跑。
            with crewai_execution_scope(metadata):
                return flow.kickoff(
                    inputs={
                        "id": task_id,
                        "task_id": task_id,
                        "original_request": request,
                        "planning_source": "resume",
                    }
                )
        finally:
            self._record_flow_state(task_id, flow, restored_from=task_id)

    def status(self, task_id: str) -> dict[str, Any]:
        task = self.store.get_task(task_id)
        latest_progress: dict[str, Any] = {}
        for event in task["events"]:
            if event["event_type"] == "AGENT_PROGRESS" and event["step_id"]:
                latest_progress[event["step_id"]] = event["payload"]
        return {
            "task_id": task_id,
            "status": task["status"],
            "current_step": task["current_step"],
            "error_type": task["error_type"],
            "error_message": task["error_message"],
            "agents": [
                {
                    "agent": run["agent"],
                    "step_id": run["step_id"],
                    "operation": run["operation"],
                    "attempt": run["attempt"],
                    "status": run["status"],
                    "error_message": run["error_message"],
                    "progress": latest_progress.get(run["step_id"], {}),
                }
                for run in task["agent_runs"]
            ],
            "recent_events": task["events"][-10:],
        }

    def _build_flow(self, request: dict[str, Any]) -> AgriculturalSimulationFlow:
        manager_config = CrewAIManagerConfig.from_request(request)
        execution_target = ExecutionTargetConfig.from_request(request)
        tool_runtime = CrewAIToolRuntime(
            project_root=self.project_root,
            store=self.store,
            execution_target=execution_target,
        )
        runner = AgriculturalAgentCrewRunner(
            settings=manager_config,
            tool_runtime=tool_runtime,
        )
        return AgriculturalSimulationFlow(
            store=self.store,
            crew_runner=runner,
            retry_delay_s=self.retry_delay_s,
        )

    def _record_flow_state(
        self,
        task_id: str,
        flow: AgriculturalSimulationFlow,
        *,
        restored_from: str = "",
    ) -> None:
        self.store.add_event(
            task_id,
            "CREWAI_FLOW_STATE",
            {
                "flow_state_id": str(flow.state.id),
                "restored_from": restored_from or None,
            },
        )

    def _flow_metadata(self, task_id: str, operation: str) -> CrewAIExecutionMetadata:
        return CrewAIExecutionMetadata(
            task_id=task_id,
            step_id="flow",
            agent="CrewAIFlow",
            operation=operation,
            attempt=1,
            ledger=self.store,
        )
