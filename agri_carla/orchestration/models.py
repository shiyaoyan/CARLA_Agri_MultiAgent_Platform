"""CrewAI Flow 使用的任务状态、执行步骤和规划结果数据模型。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class RunStatus(str, Enum):
    """任务或智能体运行记录的统一状态集合。"""

    # 这些值会持久化到 SQLite 数据库，因此使用稳定字符串而不是自动编号。
    RECEIVED = "RECEIVED"
    PLANNING = "PLANNING"
    READY = "READY"
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    RETRYING = "RETRYING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    NEEDS_INPUT = "NEEDS_INPUT"
    SKIPPED = "SKIPPED"
    CANCELLED = "CANCELLED"


@dataclass
class TaskStep:
    """执行计划中的一个可重试智能体操作。

    ``depends_on`` 记录前置步骤编号；CrewAI Flow 只有在依赖步骤成功后才会调用
    当前智能体。``max_attempts`` 控制临时故障的最多尝试次数。
    """
    step_id: str
    agent: str
    operation: str
    depends_on: list[str] = field(default_factory=list)
    max_attempts: int = 1
    optional: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为可写入 JSON 和数据库的普通字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskStep":
        """从持久化字典恢复执行步骤，并规范布尔值和重试次数。"""
        return cls(
            step_id=str(data["step_id"]),
            agent=str(data["agent"]),
            operation=str(data["operation"]),
            depends_on=list(data.get("depends_on", [])),
            max_attempts=max(1, int(data.get("max_attempts", 1))),
            optional=bool(data.get("optional", False)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class ExecutionPlan:
    """PGA 针对一次用户任务生成的有序智能体调用计划。"""
    task_type: str
    steps: list[TaskStep] = field(default_factory=list)
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        """序列化完整计划，供检查点保存和实验复现。"""
        return {
            "schema_version": self.schema_version,
            "task_type": self.task_type,
            "steps": [step.to_dict() for step in self.steps],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionPlan":
        """从任务数据库中恢复执行计划。"""
        return cls(
            task_type=str(data.get("task_type", "")),
            steps=[TaskStep.from_dict(item) for item in data.get("steps", [])],
            schema_version=str(data.get("schema_version", "1.0")),
        )

    def validate_dependencies(self) -> None:
        """检查步骤编号唯一性及依赖引用完整性。

        这里只验证静态依赖关系；实际执行时，CrewAI Flow 还会确认依赖步骤已经
        成功并存在检查点输出。
        """
        known = {step.step_id for step in self.steps}
        if len(known) != len(self.steps):
            raise ValueError("Execution-plan step identifiers must be unique.")
        for step in self.steps:
            unknown = set(step.depends_on) - known
            if unknown:
                raise ValueError(
                    "Step %s depends on unknown steps: %s"
                    % (step.step_id, ", ".join(sorted(unknown)))
                )


@dataclass
class PlanningResult:
    """PGA 返回给 CrewAI Flow 的完整规划结果。"""
    structured_request: dict[str, Any]
    execution_plan: ExecutionPlan
    validation_report: dict[str, Any]
    source: str

    def to_dict(self) -> dict[str, Any]:
        """将结构化请求、执行计划和校验报告合并为可记录结果。"""
        return {
            "structured_request": self.structured_request,
            "execution_plan": self.execution_plan.to_dict(),
            "validation_report": self.validation_report,
            "source": self.source,
        }
