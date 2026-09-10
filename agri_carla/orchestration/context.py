"""功能智能体向 CrewAI Flow 报告状态和产物的统一通道。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .run_ledger import ApplicationRunLedger
from .models import RunStatus


class ExecutionContext:
    """由 CrewAI Flow 创建并传入功能智能体的状态报告上下文。

    智能体不直接操作任务数据库，而是通过该对象报告开始、进度、成功、失败和
    输出文件，保持不同智能体的记录格式一致。
    """

    def __init__(self, task_id: str, memory: ApplicationRunLedger):
        """绑定当前任务编号和持久化存储。"""
        self.task_id = task_id
        self.memory = memory

    def start(
        self,
        agent: str,
        step_id: str,
        operation: str,
        *,
        attempt: int = 1,
        input_payload: dict[str, Any] | None = None,
    ) -> str:
        """登记一次智能体调用并返回本次运行记录编号。"""
        return self.memory.start_agent_run(
            self.task_id,
            step_id,
            agent,
            operation,
            attempt,
            input_payload,
        )

    def succeed(self, run_id: str, output: Any = None) -> None:
        """将智能体调用标记为成功并保存其结构化输出。"""
        self.memory.finish_agent_run(
            run_id,
            RunStatus.SUCCEEDED,
            output=output,
        )

    def fail(
        self,
        run_id: str,
        error: BaseException,
        *,
        status: RunStatus = RunStatus.FAILED,
    ) -> None:
        """记录异常及最终状态，供编排器决定重试或终止。"""
        self.memory.finish_agent_run(run_id, status, error=error)

    def progress(
        self,
        agent: str,
        step_id: str,
        message: str,
        **payload: Any,
    ) -> None:
        """写入不改变任务状态的中间进度事件。"""
        event_payload = {"message": message}
        event_payload.update(payload)
        self.memory.add_event(
            self.task_id,
            "AGENT_PROGRESS",
            event_payload,
            agent=agent,
            step_id=step_id,
        )

    def artifact(
        self,
        path: str | Path,
        *,
        agent: str,
        step_id: str,
        artifact_type: str = "file",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """登记智能体生成的场景、轨迹、图表或评价结果文件。"""
        self.memory.add_artifact(
            self.task_id,
            path,
            step_id=step_id,
            agent=agent,
            artifact_type=artifact_type,
            metadata=metadata,
        )
