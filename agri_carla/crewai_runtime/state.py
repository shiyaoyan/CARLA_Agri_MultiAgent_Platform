"""CrewAI Flow 的结构化状态模型。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AgriculturalFlowState(BaseModel):
    """保存一次农业仿真任务从输入到结果的完整状态。"""

    id: str = ""
    task_id: str = ""
    original_request: dict[str, Any] = Field(default_factory=dict)
    structured_request: dict[str, Any] = Field(default_factory=dict)
    execution_plan: dict[str, Any] = Field(default_factory=dict)
    validation_report: dict[str, Any] = Field(default_factory=dict)
    planning_source: str = ""
    completed_steps: dict[str, Any] = Field(default_factory=dict)
    results: dict[str, Any] = Field(default_factory=dict)
    status: str = "RECEIVED"
    current_step: str = ""
    error_type: str = ""
    error_message: str = ""
    plan_only: bool = False
    final_output: dict[str, Any] = Field(default_factory=dict)
