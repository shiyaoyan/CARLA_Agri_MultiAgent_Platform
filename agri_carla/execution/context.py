"""同机 CARLA Worker 使用的内存状态上下文。"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any


class BufferedExecutionContext:
    """收集子智能体状态，使执行端可以一次性返回给 CrewAI 管理端。"""

    def __init__(self, task_id: str):
        self.task_id = task_id
        self.events: list[dict[str, Any]] = []
        self._runs: dict[str, dict[str, Any]] = {}

    def start(
        self,
        agent: str,
        step_id: str,
        operation: str,
        *,
        attempt: int = 1,
        input_payload: dict[str, Any] | None = None,
    ) -> str:
        run_id = uuid.uuid4().hex
        run = {
            "run_id": run_id,
            "agent": agent,
            "step_id": step_id,
            "operation": operation,
            "attempt": attempt,
            "status": "RUNNING",
            "input": input_payload or {},
        }
        self._runs[run_id] = run
        self.events.append({"type": "run_started", **run})
        return run_id

    def succeed(self, run_id: str, output: Any = None) -> None:
        run = self._runs[run_id]
        run["status"] = "SUCCEEDED"
        self.events.append({"type": "run_succeeded", **run, "output": output})

    def fail(self, run_id: str, error: BaseException, *, status=None) -> None:
        run = self._runs[run_id]
        run["status"] = getattr(status, "value", status) or "FAILED"
        self.events.append(
            {
                "type": "run_failed",
                **run,
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )

    def progress(
        self,
        agent: str,
        step_id: str,
        message: str,
        **payload: Any,
    ) -> None:
        self.events.append(
            {
                "type": "progress",
                "agent": agent,
                "step_id": step_id,
                "message": message,
                "payload": payload,
            }
        )

    def artifact(
        self,
        path: str | Path,
        *,
        agent: str,
        step_id: str,
        artifact_type: str = "file",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.events.append(
            {
                "type": "artifact",
                "agent": agent,
                "step_id": step_id,
                "path": str(path),
                "artifact_type": artifact_type,
                "metadata": metadata or {},
            }
        )
