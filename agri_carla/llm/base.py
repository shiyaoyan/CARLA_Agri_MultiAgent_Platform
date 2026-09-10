"""大模型配置、响应异常和 JSON 客户端协议定义。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class LLMResponseError(RuntimeError):
    """模型请求失败，或响应不能解析为单个 JSON 对象。"""


@dataclass
class LLMConfig:
    """PGA 调用 OpenAI 兼容接口所需的可移植配置。

    ``api_key_env`` 保存环境变量名称而不保存密钥本身；令牌参数和推理强度字段
    允许兼容不同模型接口。``max_repair_attempts`` 限制请求自动修复次数。
    """
    enabled: bool = False
    provider: str = "openai_compatible"
    model: str = ""
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    timeout_s: float = 120.0
    temperature: float = 0.0
    send_temperature: bool = True
    max_tokens: int = 6000
    max_tokens_parameter: str = "max_tokens"
    reasoning_effort: str = ""
    image_detail: str = "auto"
    include_images: bool = True
    max_repair_attempts: int = 2
    extra_headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "LLMConfig":
        """从用户配置中读取已知字段，忽略与当前客户端无关的扩展项。"""
        values = dict(data or {})
        known = cls.__dataclass_fields__
        return cls(**{key: value for key, value in values.items() if key in known})


class JSONLLMClient(Protocol):
    """PGA 所依赖的最小 JSON 生成接口，便于替换具体模型服务。"""
    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        """根据系统约束、用户输入和可选图像返回 JSON 字典。"""
        ...
