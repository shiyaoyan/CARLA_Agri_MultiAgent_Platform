"""智能体能力白名单及平台默认层级关系。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class AgentDescriptor:
    """描述一个允许被 PGA 编排的智能体及其调度层级。"""
    agent_id: str
    name: str
    role: str
    scheduling: str = "top_level"

    def to_dict(self) -> dict[str, str]:
        """转换为提示词和校验报告可使用的字典。"""
        return asdict(self)


class AgentRegistry:
    """PGA 与 CrewAI Flow 共用的智能体白名单。

    注册表限制大模型只能规划项目确实实现的智能体，不能通过自然语言构造任意
    Python 调用。描述信息用于规划，handler 则为可选的实际执行函数。
    """

    def __init__(self):
        """初始化空的智能体描述和处理函数映射。"""
        self._descriptors: dict[str, AgentDescriptor] = {}
        self._handlers: dict[str, Callable[..., Any] | None] = {}

    def register(
        self,
        descriptor: AgentDescriptor,
        handler: Callable[..., Any] | None = None,
    ) -> None:
        """注册新智能体；重复编号会直接报错，防止路由含义被覆盖。"""
        if descriptor.agent_id in self._descriptors:
            raise ValueError("Agent already registered: %s" % descriptor.agent_id)
        self._descriptors[descriptor.agent_id] = descriptor
        self._handlers[descriptor.agent_id] = handler

    def descriptor(self, agent_id: str) -> AgentDescriptor:
        """获取智能体描述，并对未注册编号给出明确错误。"""
        try:
            return self._descriptors[agent_id]
        except KeyError as exc:
            raise KeyError("Agent is not registered: %s" % agent_id) from exc

    def handler(self, agent_id: str):
        """返回智能体的可调用处理函数。"""
        self.descriptor(agent_id)
        return self._handlers[agent_id]

    def set_handler(self, agent_id: str, handler: Callable[..., Any]) -> None:
        """为已登记智能体绑定实际处理函数。"""
        self.descriptor(agent_id)
        self._handlers[agent_id] = handler

    def contains(self, agent_id: str) -> bool:
        """判断智能体编号是否位于允许列表。"""
        return agent_id in self._descriptors

    def describe(self) -> list[dict[str, str]]:
        """按编号稳定输出所有能力，便于加入 PGA 提示词。"""
        return [
            self._descriptors[key].to_dict()
            for key in sorted(self._descriptors)
        ]


def build_default_registry() -> AgentRegistry:
    """建立与论文总体架构一致的默认智能体注册表。

    PGA、SGA、MCA、CIA 和 EEA 属于同一调度层；TPS 与 PES 仅作为 EEA
    的下属智能体运行。CIA 标记为并列服务，表示 EEA 会调用其控制能力，但
    CIA 并不从属于 EEA。
    """
    registry = AgentRegistry()
    registry.register(
        AgentDescriptor("PGA", "Plan Generation Agent", "Input interpretation and task planning")
    )
    registry.register(
        AgentDescriptor("SGA", "Scene Generation Agent", "Agricultural scene generation")
    )
    registry.register(
        AgentDescriptor("MCA", "Machinery Configuration Agent", "Machinery template configuration")
    )
    registry.register(
        AgentDescriptor(
            "CIA",
            "Controller Interface Agent",
            "Control service and optional parameter self-optimization",
            scheduling="peer_service",
        )
    )
    registry.register(
        AgentDescriptor("EEA", "Experiment Execution Agent", "CARLA experiment execution")
    )
    registry.register(
        AgentDescriptor(
            "TPS",
            "Trajectory Preprocessing Sub-Agent",
            "Trajectory standardization and direction recognition",
            scheduling="eea_subagent",
        )
    )
    registry.register(
        AgentDescriptor(
            "PES",
            "Performance Evaluation Subagent",
            "Experiment recording and evaluation",
            scheduling="eea_subagent",
        )
    )
    return registry
