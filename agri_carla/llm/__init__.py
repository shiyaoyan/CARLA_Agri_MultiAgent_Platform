"""仅供任务规划智能体使用、与具体服务商尽量无关的大模型客户端。

大模型在本项目中只负责把自然语言和图像语义整理为 JSON 候选请求，不直接执行
CARLA 命令，也不替代本地的结构、几何与拓扑校验和智能体调度。
"""

from .base import JSONLLMClient, LLMConfig, LLMResponseError
from .openai_compatible import OpenAICompatibleJSONClient, build_llm_client

__all__ = [
    "JSONLLMClient",
    "LLMConfig",
    "LLMResponseError",
    "OpenAICompatibleJSONClient",
    "build_llm_client",
]
