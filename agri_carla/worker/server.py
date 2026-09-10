"""兼容 Python 3.8 的 CARLA 异步执行端。"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from agri_carla.execution import BufferedExecutionContext, DomainExecutionService
from agri_carla.orchestration.errors import DependencyUnavailableError
from agri_carla.utils.io import ensure_dir, write_json


_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


class WorkerTaskRepository:
    """以独立 JSON 文件保存执行端任务状态，支持重启后查询。"""

    def __init__(self, root: str | Path):
        self.root = ensure_dir(root)
        self._lock = threading.Lock()

    def path(self, task_id: str) -> Path:
        if not _SAFE_ID.fullmatch(task_id):
            raise ValueError("Invalid worker_task_id.")
        return self.root / (task_id + ".json")

    def get(self, task_id: str) -> dict[str, Any]:
        path = self.path(task_id)
        if not path.is_file():
            raise KeyError(task_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def save(self, task_id: str, payload: dict[str, Any]) -> None:
        with self._lock:
            write_json(self.path(task_id), payload)


class CarlaWorkerApplication:
    """验证请求、后台执行领域步骤并维护任务终态。"""

    def __init__(self, repository: WorkerTaskRepository, max_workers: int = 1):
        self.repository = repository
        self.executor = ThreadPoolExecutor(max_workers=max(1, max_workers))

    def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        required = (
            "worker_task_id",
            "manager_task_id",
            "step_id",
            "agent",
            "operation",
            "request",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError("Missing worker request fields: %s" % ", ".join(missing))
        task_id = str(payload["worker_task_id"])
        try:
            existing = self.repository.get(task_id)
            return {
                "worker_task_id": task_id,
                "status": existing["status"],
                "reused": True,
            }
        except KeyError:
            pass
        record = {
            "worker_task_id": task_id,
            "manager_task_id": str(payload["manager_task_id"]),
            "step_id": str(payload["step_id"]),
            "agent": str(payload["agent"]),
            "operation": str(payload["operation"]),
            "status": "QUEUED",
            "output": {},
            "events": [],
            "error_type": None,
            "error_message": None,
        }
        self.repository.save(task_id, record)
        self.executor.submit(self._run, task_id, payload)
        return {"worker_task_id": task_id, "status": "QUEUED", "reused": False}

    def _run(self, task_id: str, payload: dict[str, Any]) -> None:
        record = self.repository.get(task_id)
        record["status"] = "RUNNING"
        self.repository.save(task_id, record)
        context = BufferedExecutionContext(str(payload["manager_task_id"]))
        try:
            output = DomainExecutionService().invoke(
                str(payload["agent"]),
                str(payload["operation"]),
                dict(payload["request"]),
                context=context,
                step_id=str(payload["step_id"]),
            )
            record["status"] = "SUCCEEDED"
            record["output"] = output
        except Exception as exc:
            dependency_errors = (
                DependencyUnavailableError,
                FileNotFoundError,
                ConnectionError,
                TimeoutError,
                ImportError,
                ModuleNotFoundError,
            )
            record["status"] = "BLOCKED" if isinstance(exc, dependency_errors) else "FAILED"
            record["error_type"] = type(exc).__name__
            record["error_message"] = str(exc)
        record["events"] = context.events
        self.repository.save(task_id, record)


def build_handler(application: CarlaWorkerApplication, token: str):
    """创建绑定当前应用和认证令牌的 HTTP 请求处理器。"""

    class Handler(BaseHTTPRequestHandler):
        server_version = "AgriCARLAWorker/1.0"

        def do_GET(self):
            if not self._authorized():
                return
            if self.path == "/health":
                self._json(200, {"status": "ok", "protocol_version": "1.0"})
                return
            prefix = "/v1/tasks/"
            if self.path.startswith(prefix):
                task_id = unquote(self.path[len(prefix):])
                try:
                    self._json(200, application.repository.get(task_id))
                except KeyError:
                    self._json(404, {"error": "Unknown worker task."})
                except ValueError as exc:
                    self._json(400, {"error": str(exc)})
                return
            self._json(404, {"error": "Unknown endpoint."})

        def do_POST(self):
            if not self._authorized():
                return
            if self.path != "/v1/tasks":
                self._json(404, {"error": "Unknown endpoint."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                self._json(202, application.submit(payload))
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._json(400, {"error": str(exc)})

        def _authorized(self) -> bool:
            if not token:
                return True
            if self.headers.get("Authorization", "") == "Bearer " + token:
                return True
            self._json(401, {"error": "Unauthorized."})
            return False

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            print("[CARLA worker] " + format % args)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CARLA execution worker.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--project-root", default="")
    parser.add_argument(
        "--tasks-dir",
        "--task-dir",
        dest="tasks_dir",
        default="runtime/worker_tasks",
    )
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--token-env", default="AGRI_CARLA_WORKER_TOKEN")
    args = parser.parse_args()

    if args.project_root:
        os.chdir(str(Path(args.project_root).resolve()))
    token = os.environ.get(args.token_env, "")
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit(
            "This project supports only a same-host dual-environment deployment; "
            "the CARLA worker must bind to a loopback address."
        )
    application = CarlaWorkerApplication(
        WorkerTaskRepository(args.tasks_dir),
        max_workers=args.max_workers,
    )
    server = ThreadingHTTPServer(
        (args.host, args.port),
        build_handler(application, token),
    )
    print("CARLA worker listening on http://%s:%d" % (args.host, args.port))
    server.serve_forever()


if __name__ == "__main__":
    main()
