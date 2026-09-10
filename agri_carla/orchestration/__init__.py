"""农业多智能体平台的运行状态与计划公共接口。

外部模块从这里导入任务状态、执行计划、异常、运行上下文和持久化存储，避免
依赖编排子模块的内部文件布局。
"""

from .context import ExecutionContext
from .errors import (
    DependencyUnavailableError,
    InputValidationError,
    NeedsUserInputError,
    OrchestrationError,
    TransientExecutionError,
)
from .run_ledger import ApplicationRunLedger
from .models import ExecutionPlan, RunStatus, TaskStep
from .registry import AgentDescriptor, AgentRegistry, build_default_registry

__all__ = [
    "AgentDescriptor",
    "AgentRegistry",
    "DependencyUnavailableError",
    "ExecutionContext",
    "ExecutionPlan",
    "InputValidationError",
    "NeedsUserInputError",
    "OrchestrationError",
    "ApplicationRunLedger",
    "RunStatus",
    "TaskStep",
    "TransientExecutionError",
    "build_default_registry",
]
