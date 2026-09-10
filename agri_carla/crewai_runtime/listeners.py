"""CrewAI 官方事件总线到项目运行账本的轻量监听器。"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator

from crewai.events import (
    AgentExecutionCompletedEvent,
    AgentExecutionErrorEvent,
    AgentExecutionStartedEvent,
    BaseEventListener,
    CrewKickoffCompletedEvent,
    CrewKickoffFailedEvent,
    CrewKickoffStartedEvent,
    FlowFailedEvent,
    FlowFinishedEvent,
    FlowStartedEvent,
    MethodExecutionFailedEvent,
    MethodExecutionFinishedEvent,
    MethodExecutionStartedEvent,
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskStartedEvent,
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
)


@dataclass(frozen=True)
class CrewAIExecutionMetadata:
    """把 CrewAI 生命周期事件关联到一个平台任务步骤。"""

    task_id: str
    step_id: str
    agent: str
    operation: str
    attempt: int
    ledger: Any = None


_CURRENT_EXECUTION: ContextVar[CrewAIExecutionMetadata | None] = ContextVar(
    "agri_carla_crewai_execution",
    default=None,
)


@contextmanager
def crewai_execution_scope(
    metadata: CrewAIExecutionMetadata,
) -> Iterator[CrewAIExecutionMetadata]:
    """在一次 Crew kickoff 期间提供事件关联上下文。"""
    token = _CURRENT_EXECUTION.set(metadata)
    try:
        yield metadata
    finally:
        _CURRENT_EXECUTION.reset(token)


class AgriculturalCrewAIEventListener(BaseEventListener):
    """使用 CrewAI BaseEventListener 记录 Crew、Agent、Task 和 Tool 事件。"""

    def setup_listeners(self, crewai_event_bus) -> None:
        registrations = (
            (CrewKickoffStartedEvent, "CREW_KICKOFF_STARTED"),
            (CrewKickoffCompletedEvent, "CREW_KICKOFF_COMPLETED"),
            (CrewKickoffFailedEvent, "CREW_KICKOFF_FAILED"),
            (AgentExecutionStartedEvent, "AGENT_EXECUTION_STARTED"),
            (AgentExecutionCompletedEvent, "AGENT_EXECUTION_COMPLETED"),
            (AgentExecutionErrorEvent, "AGENT_EXECUTION_ERROR"),
            (TaskStartedEvent, "TASK_EXECUTION_STARTED"),
            (TaskCompletedEvent, "TASK_EXECUTION_COMPLETED"),
            (TaskFailedEvent, "TASK_EXECUTION_FAILED"),
            (ToolUsageStartedEvent, "TOOL_USAGE_STARTED"),
            (ToolUsageFinishedEvent, "TOOL_USAGE_FINISHED"),
            (ToolUsageErrorEvent, "TOOL_USAGE_ERROR"),
            (FlowStartedEvent, "FLOW_STARTED"),
            (FlowFinishedEvent, "FLOW_FINISHED"),
            (FlowFailedEvent, "FLOW_FAILED"),
            (MethodExecutionStartedEvent, "FLOW_METHOD_STARTED"),
            (MethodExecutionFinishedEvent, "FLOW_METHOD_FINISHED"),
            (MethodExecutionFailedEvent, "FLOW_METHOD_FAILED"),
        )
        for event_class, event_name in registrations:
            self._register(crewai_event_bus, event_class, event_name)

    @staticmethod
    def _register(crewai_event_bus, event_class, event_name: str) -> None:
        @crewai_event_bus.on(event_class)
        def on_event(source, event) -> None:
            metadata = _CURRENT_EXECUTION.get()
            if metadata is None or metadata.ledger is None:
                return
            payload = {
                "operation": metadata.operation,
                "attempt": metadata.attempt,
                "crewai_event_type": str(getattr(event, "type", event_name)),
                "source_type": type(source).__name__,
            }
            error = getattr(event, "error", None)
            if error:
                payload["error"] = str(error)
            try:
                metadata.ledger.add_event(
                    metadata.task_id,
                    event_name,
                    payload,
                    agent=metadata.agent,
                    step_id=metadata.step_id,
                )
            except Exception:
                # 监听器不得反向影响 CrewAI 的业务执行。
                return


# 按 CrewAI 官方要求在模块导入时实例化，确保处理器注册并保持存活。
agricultural_crewai_event_listener = AgriculturalCrewAIEventListener()


__all__ = [
    "AgriculturalCrewAIEventListener",
    "CrewAIExecutionMetadata",
    "agricultural_crewai_event_listener",
    "crewai_execution_scope",
]
