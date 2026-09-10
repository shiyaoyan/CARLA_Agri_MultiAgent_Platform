"""CrewAI 管理环境使用的同机 CARLA Worker 客户端。"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from agri_carla.orchestration.errors import (
    DependencyUnavailableError,
    TransientExecutionError,
)


@dataclass
class CarlaWorkerConfig:
    """本机执行环境地址、认证、轮询周期和总等待时间。"""

    base_url: str
    token_env: str = "AGRI_CARLA_WORKER_TOKEN"
    poll_interval_s: float = 1.0
    timeout_s: float = 7200.0

    @property
    def token(self) -> str:
        return os.environ.get(self.token_env, "")


class CarlaWorkerClient:
    """通过版本化 JSON/HTTP 协议提交并轮询农业仿真步骤。"""

    def __init__(self, config: CarlaWorkerConfig):
        self.config = config
        self.base_url = config.base_url.rstrip("/")

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def execute(
        self,
        *,
        worker_task_id: str,
        manager_task_id: str,
        step_id: str,
        agent: str,
        operation: str,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        """提交一步任务，并等待执行端返回成功或失败终态。"""
        accepted = self._request(
            "POST",
            "/v1/tasks",
            {
                "worker_task_id": worker_task_id,
                "manager_task_id": manager_task_id,
                "step_id": step_id,
                "agent": agent,
                "operation": operation,
                "request": request,
            },
        )
        task_id = str(accepted["worker_task_id"])
        deadline = time.monotonic() + max(1.0, self.config.timeout_s)
        while True:
            task = self._request("GET", "/v1/tasks/" + task_id)
            status = str(task.get("status", ""))
            if status == "SUCCEEDED":
                return task
            if status in {"FAILED", "BLOCKED"}:
                error_type = str(task.get("error_type", "RuntimeError"))
                message = str(task.get("error_message", "CARLA worker execution failed."))
                if status == "BLOCKED":
                    raise DependencyUnavailableError(
                        "%s: %s" % (error_type, message)
                    )
                raise RuntimeError("%s: %s" % (error_type, message))
            if time.monotonic() >= deadline:
                raise TransientExecutionError(
                    "Timed out while waiting for CARLA worker task %s." % task_id
                )
            time.sleep(max(0.05, self.config.poll_interval_s))

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.config.token:
            headers["Authorization"] = "Bearer " + self.config.token
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=30.0) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise DependencyUnavailableError(
                "CARLA worker HTTP %d: %s" % (exc.code, detail)
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise TransientExecutionError(
                "Cannot connect to CARLA worker at %s: %s"
                % (self.base_url, exc)
            ) from exc
