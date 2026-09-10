"""验证 OpenAI 兼容大模型客户端的请求参数和图像载荷。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agri_carla.llm import LLMConfig, OpenAICompatibleJSONClient


class _Response:
    """模拟 urllib 返回的最小 Chat Completions 响应。"""
    def __enter__(self):
        """支持响应对象作为上下文管理器使用。"""
        return self

    def __exit__(self, exc_type, exc, traceback):
        """不屏蔽上下文中的异常。"""
        return False

    def read(self):
        """返回包含 JSON 文本内容的字节响应。"""
        return json.dumps(
            {"choices": [{"message": {"content": '{"task_type":"test"}'}}]}
        ).encode("utf-8")


class LLMClientTests(unittest.TestCase):
    """检查模型特定参数不会在兼容 API 请求中丢失。"""

    def test_gpt_5_6_payload_uses_configured_parameters_and_image_detail(self):
        """请求应包含所配模型、推理强度、token 参数和图像清晰度。"""
        with tempfile.TemporaryDirectory() as temporary_dir:
            image_path = Path(temporary_dir) / "satellite.png"
            image_path.write_bytes(b"not-a-real-image")
            config = LLMConfig(
                enabled=True,
                model="gpt-5.6-sol",
                send_temperature=False,
                max_tokens=8000,
                max_tokens_parameter="max_completion_tokens",
                reasoning_effort="low",
                image_detail="high",
            )
            client = OpenAICompatibleJSONClient(config)

            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), patch(
                "urllib.request.urlopen", return_value=_Response()
            ) as urlopen:
                result = client.generate_json(
                    system_prompt="Return JSON.",
                    user_prompt="Create a task.",
                    image_paths=[str(image_path)],
                )

            request = urlopen.call_args.args[0]
            payload = json.loads(request.data.decode("utf-8"))
            self.assertEqual(result["task_type"], "test")
            self.assertEqual(payload["model"], "gpt-5.6-sol")
            self.assertEqual(payload["max_completion_tokens"], 8000)
            self.assertEqual(payload["reasoning_effort"], "low")
            self.assertNotIn("temperature", payload)
            self.assertNotIn("max_tokens", payload)
            self.assertEqual(
                payload["messages"][1]["content"][1]["image_url"]["detail"],
                "high",
            )


if __name__ == "__main__":
    unittest.main()
