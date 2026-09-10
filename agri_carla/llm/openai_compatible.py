"""基于 Python 标准库实现的 OpenAI Chat Completions 兼容客户端。"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .base import LLMConfig, LLMResponseError


def _endpoint(base_url: str) -> str:
    """将服务根地址规范为 chat/completions 请求端点。"""
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return normalized + "/chat/completions"


def _image_data_url(path: str | Path) -> str:
    """把本地图像编码为多模态接口可接收的 Base64 data URL。"""
    image_path = Path(path)
    media_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return "data:%s;base64,%s" % (media_type, encoded)


def _extract_json_object(text: str) -> dict[str, Any]:
    """从纯 JSON 或 Markdown 代码块形式的模型输出中提取字典。

    响应存在附加说明时，会尝试截取最外层花括号；无法可靠解析时立即报错，
    不使用字符串替换猜测字段，以免产生不可追踪的任务配置。
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise LLMResponseError("The model response does not contain a JSON object.")
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMResponseError(
                "The model returned malformed JSON: %s" % exc
            ) from exc
    if not isinstance(parsed, dict):
        raise LLMResponseError("The model response must be one JSON object.")
    return parsed


class OpenAICompatibleJSONClient:
    """精简的 OpenAI 兼容 Chat Completions 客户端。

    实现只使用 Python 标准库，使 CARLA 的 Python 3.8 环境无需安装额外 SDK。
    """

    def __init__(self, config: LLMConfig):
        """保存接口地址、模型、超时和密钥环境变量等调用配置。"""
        self.config = config

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        """发送文本和可选图像，并返回经过严格解析的 JSON 对象。"""
        if not self.config.model:
            raise LLMResponseError("llm.model must be configured.")

        # 文本和图像按多模态消息格式组合；图像只有在配置允许且文件存在时发送。
        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": user_prompt}
        ]
        if self.config.include_images:
            for image_path in image_paths or []:
                path = Path(image_path)
                if not path.is_file():
                    raise LLMResponseError("Image does not exist: %s" % path)
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _image_data_url(path),
                            "detail": self.config.image_detail,
                        },
                    }
                )

        # response_format 明确要求 JSON，解析层仍会再次检查实际返回内容。
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "response_format": {"type": "json_object"},
        }
        if self.config.send_temperature:
            payload["temperature"] = self.config.temperature
        token_parameter = self.config.max_tokens_parameter
        if token_parameter not in {"max_tokens", "max_completion_tokens"}:
            raise LLMResponseError(
                "llm.max_tokens_parameter must be max_tokens or "
                "max_completion_tokens."
            )
        payload[token_parameter] = self.config.max_tokens
        if self.config.reasoning_effort:
            payload["reasoning_effort"] = self.config.reasoning_effort
        headers = {"Content-Type": "application/json"}
        # 密钥在请求发送前才从环境变量读取，不写入配置、日志或结构化任务文件。
        api_key = os.environ.get(self.config.api_key_env, "")
        if "api.openai.com" in self.config.base_url and not api_key:
            raise LLMResponseError(
                "Environment variable %s is not set."
                % self.config.api_key_env
            )
        if api_key:
            headers["Authorization"] = "Bearer " + api_key
        headers.update(self.config.extra_headers)

        request = urllib.request.Request(
            _endpoint(self.config.base_url),
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.config.timeout_s,
            ) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMResponseError(
                "LLM HTTP error %s: %s" % (exc.code, detail[:1000])
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise LLMResponseError("LLM connection failed: %s" % exc) from exc

        try:
            content = response_data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError(
                "Unexpected LLM response structure: %s"
                % json.dumps(response_data, ensure_ascii=False)[:1000]
            ) from exc

        if isinstance(content, list):
            content = "".join(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict)
            )
        return _extract_json_object(str(content))


def build_llm_client(config: LLMConfig):
    """根据配置创建客户端；未知服务协议在执行前明确拒绝。"""
    if config.provider != "openai_compatible":
        raise ValueError("Unsupported LLM provider: %s" % config.provider)
    return OpenAICompatibleJSONClient(config)
