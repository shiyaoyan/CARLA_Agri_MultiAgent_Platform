"""CrewAI 管理环境和同机 CARLA 执行环境配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from dotenv import load_dotenv


# 遵循 CrewAI 官方项目约定，从项目根目录的 .env 读取模型与执行端配置。
load_dotenv()


@dataclass
class CrewAIManagerConfig:
    """CrewAI Agent 使用的模型与运行选项。"""

    model: str
    base_url: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    verbose: bool = False

    @classmethod
    def from_request(cls, request: dict[str, Any]) -> "CrewAIManagerConfig":
        crew = dict(request.get("crewai", {}))
        llm = dict(request.get("llm", {}))
        model = str(
            crew.get("model")
            or os.environ.get("AGRI_CREW_MODEL", "")
            or llm.get("model", "")
            or "openai/gpt-5.6-sol"
        ).strip()
        if "/" not in model:
            model = "openai/" + model
        return cls(
            model=model,
            base_url=str(
                crew.get("base_url")
                or os.environ.get("AGRI_CREW_BASE_URL", "")
                or llm.get("base_url", "")
            ).strip(),
            api_key_env=str(
                crew.get("api_key_env")
                or os.environ.get("AGRI_CREW_API_KEY_ENV", "")
                or llm.get("api_key_env", "OPENAI_API_KEY")
            ),
            verbose=bool(crew.get("verbose", False)),
        )

    def api_key(self) -> str:
        value = os.environ.get(self.api_key_env, "")
        if not value:
            raise ValueError(
                "CrewAI API key environment variable %s is not set."
                % self.api_key_env
            )
        return value


@dataclass
class ExecutionTargetConfig:
    """定义经本机回环接口交给 CARLA Python 3.8 环境的智能体。"""

    worker_url: str = "http://127.0.0.1:8765"
    worker_agents: set[str] = field(default_factory=lambda: {"EEA"})
    token_env: str = "AGRI_CARLA_WORKER_TOKEN"
    poll_interval_s: float = 1.0
    timeout_s: float = 7200.0

    @classmethod
    def from_request(cls, request: dict[str, Any]) -> "ExecutionTargetConfig":
        execution = dict(request.get("execution", {}))
        worker_url = str(
            execution.get("worker_url")
            or os.environ.get("AGRI_CARLA_WORKER_URL", "")
            or "http://127.0.0.1:8765"
        ).strip()
        parsed = urlsplit(worker_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.port is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "The CARLA worker must use a same-host loopback URL, for example "
                "http://127.0.0.1:8765."
            )

        configured_agents = execution.get("worker_agents")
        if configured_agents is None:
            environment_agents = os.environ.get("AGRI_CARLA_WORKER_AGENTS", "")
            configured_agents = (
                [item.strip() for item in environment_agents.split(",") if item.strip()]
                if environment_agents
                else ["EEA"]
            )
        worker_agents = {str(item).upper() for item in configured_agents}
        unsupported = worker_agents - {"SGA", "MCA", "CIA", "EEA"}
        if unsupported:
            raise ValueError(
                "Unsupported same-host worker agents: %s"
                % ", ".join(sorted(unsupported))
            )
        return cls(
            worker_url=worker_url,
            worker_agents=worker_agents,
            token_env=str(
                execution.get("worker_token_env", "AGRI_CARLA_WORKER_TOKEN")
            ),
            poll_interval_s=float(execution.get("poll_interval_s", 1.0)),
            timeout_s=float(execution.get("timeout_s", 7200.0)),
        )

    def uses_worker(self, agent: str) -> bool:
        """判断该步骤是否必须进入同机 CARLA 隔离环境。"""
        return agent.upper() in self.worker_agents
