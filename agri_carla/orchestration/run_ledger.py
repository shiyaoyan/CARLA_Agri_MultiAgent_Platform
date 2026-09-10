"""农业仿真实验的 SQLite 运行账本与文件检查点存储。

该模块同时保存任务状态、每次智能体调用、过程事件和输出产物。数据库用于可靠
查询，任务目录中的 JSON 文件便于人工审查和论文实验复现。
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from agri_carla.utils.io import ensure_dir, write_json

from .models import RunStatus


def _utc_now() -> str:
    """返回带时区的 UTC 时间字符串，统一不同计算机上的运行记录。"""
    return datetime.now(timezone.utc).isoformat()


def _json_default(value):
    """将路径、枚举、数据类和平台模型转换为 JSON 可序列化形式。"""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return repr(value)


def _json_dumps(value: Any) -> str:
    """序列化数据库字段，并保留中文内容。"""
    return json.dumps(value, ensure_ascii=False, default=_json_default)


def _json_loads(value: str | None, default):
    """从可空数据库字段恢复对象；空字段返回调用方指定的默认值。"""
    if not value:
        return default
    return json.loads(value)


class ApplicationRunLedger:
    """由 CrewAI Flow 写入的项目审计记录、领域检查点与产物清单。

    本类不规划、排序、路由或调度任务。顶层执行完全由 CrewAI Flow 决定；
    该账本只保存论文平台需要查询的实验状态、领域步骤输出和文件摘要。
    """

    def __init__(
        self,
        database_path: str | Path | None = None,
        runs_dir: str | Path | None = None,
    ):
        """确定数据库和任务目录位置，并初始化所需数据表。"""
        runtime_root = Path(
            os.environ.get("AGRI_CARLA_RUNTIME_DIR", "runtime")
        )
        self.database_path = Path(database_path or runtime_root / "crewai_runtime.sqlite3")
        self.runs_dir = Path(runs_dir or runtime_root / "runs")
        ensure_dir(self.database_path.parent)
        ensure_dir(self.runs_dir)
        self._initialize()

    @contextmanager
    def _connect(self):
        """建立短连接，并在事务完成或异常回滚后显式关闭。"""
        connection = sqlite3.connect(str(self.database_path))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        """创建任务、智能体运行、事件和产物四类数据表。"""
        # 使用 IF NOT EXISTS 可在重复启动平台时复用原数据库，不会清空历史实验。
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    original_input_json TEXT NOT NULL,
                    structured_request_json TEXT,
                    execution_plan_json TEXT,
                    final_output_json TEXT,
                    current_step TEXT,
                    error_type TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS agent_runs (
                    run_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    step_id TEXT NOT NULL,
                    agent TEXT NOT NULL,
                    operation TEXT,
                    attempt INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    input_json TEXT,
                    output_json TEXT,
                    error_type TEXT,
                    error_message TEXT,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    FOREIGN KEY(task_id) REFERENCES tasks(task_id)
                );

                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    agent TEXT,
                    step_id TEXT,
                    event_type TEXT NOT NULL,
                    payload_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(task_id) REFERENCES tasks(task_id)
                );

                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    step_id TEXT,
                    agent TEXT,
                    artifact_type TEXT,
                    path TEXT NOT NULL,
                    sha256 TEXT,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(task_id) REFERENCES tasks(task_id)
                );
                """
            )

    def create_task(
        self,
        original_input: dict[str, Any],
        task_id: str | None = None,
    ) -> str:
        """登记新任务，保存原始输入并返回唯一任务编号。"""
        task_id = task_id or datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
        now = _utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO tasks (
                    task_id, status, original_input_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    RunStatus.RECEIVED.value,
                    _json_dumps(original_input),
                    now,
                    now,
                ),
            )
        task_dir = self.task_dir(task_id)
        write_json(task_dir / "input" / "original_input.json", original_input)
        self.add_event(task_id, "TASK_CREATED", {"status": RunStatus.RECEIVED.value})
        return task_id

    def task_dir(self, task_id: str) -> Path:
        """返回某项任务的独立文件目录，不存在时自动创建。"""
        return ensure_dir(self.runs_dir / task_id)

    def update_task(
        self,
        task_id: str,
        status: RunStatus | str,
        *,
        current_step: str | None = None,
        error: BaseException | None = None,
        final_output: dict[str, Any] | None = None,
    ) -> None:
        """更新任务状态、当前步骤、异常或最终输出，并同步写入状态事件。"""
        status_value = status.value if isinstance(status, RunStatus) else str(status)
        error_type = type(error).__name__ if error else None
        error_message = str(error) if error else None
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE tasks SET
                    status = ?,
                    current_step = ?,
                    error_type = ?,
                    error_message = ?,
                    final_output_json = COALESCE(?, final_output_json),
                    updated_at = ?
                WHERE task_id = ?
                """,
                (
                    status_value,
                    current_step,
                    error_type,
                    error_message,
                    _json_dumps(final_output) if final_output is not None else None,
                    _utc_now(),
                    task_id,
                ),
            )
        self.add_event(
            task_id,
            "TASK_STATUS",
            {
                "status": status_value,
                "current_step": current_step,
                "error_type": error_type,
                "error_message": error_message,
            },
        )

    def save_planning(
        self,
        task_id: str,
        structured_request: dict[str, Any],
        execution_plan: dict[str, Any],
        validation_report: dict[str, Any],
    ) -> None:
        """持久化 PGA 产生的结构化请求、执行计划和校验报告。"""
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE tasks SET
                    structured_request_json = ?,
                    execution_plan_json = ?,
                    updated_at = ?
                WHERE task_id = ?
                """,
                (
                    _json_dumps(structured_request),
                    _json_dumps(execution_plan),
                    _utc_now(),
                    task_id,
                ),
            )
        task_dir = self.task_dir(task_id)
        write_json(task_dir / "structured_request.json", structured_request)
        write_json(task_dir / "execution_plan.json", execution_plan)
        write_json(task_dir / "validation_report.json", validation_report)

    def start_agent_run(
        self,
        task_id: str,
        step_id: str,
        agent: str,
        operation: str,
        attempt: int,
        input_payload: dict[str, Any] | None = None,
    ) -> str:
        """为一次智能体调用创建 RUNNING 记录并返回 ``run_id``。"""
        run_id = uuid.uuid4().hex
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO agent_runs (
                    run_id, task_id, step_id, agent, operation, attempt,
                    status, input_json, started_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    task_id,
                    step_id,
                    agent,
                    operation,
                    attempt,
                    RunStatus.RUNNING.value,
                    _json_dumps(input_payload or {}),
                    _utc_now(),
                ),
            )
        self.add_event(
            task_id,
            "AGENT_STARTED",
            {"agent": agent, "operation": operation, "attempt": attempt},
            agent=agent,
            step_id=step_id,
        )
        return run_id

    def finish_agent_run(
        self,
        run_id: str,
        status: RunStatus | str,
        *,
        output: Any = None,
        error: BaseException | None = None,
    ) -> None:
        """结束一次智能体调用，保存输出或异常并生成完成事件。"""
        status_value = status.value if isinstance(status, RunStatus) else str(status)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT task_id, agent, step_id FROM agent_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                raise KeyError("Unknown agent run: %s" % run_id)
            connection.execute(
                """
                UPDATE agent_runs SET
                    status = ?,
                    output_json = ?,
                    error_type = ?,
                    error_message = ?,
                    ended_at = ?
                WHERE run_id = ?
                """,
                (
                    status_value,
                    _json_dumps(output) if output is not None else None,
                    type(error).__name__ if error else None,
                    str(error) if error else None,
                    _utc_now(),
                    run_id,
                ),
            )
        self.add_event(
            row["task_id"],
            "AGENT_FINISHED",
            {
                "status": status_value,
                "error_type": type(error).__name__ if error else None,
                "error_message": str(error) if error else None,
            },
            agent=row["agent"],
            step_id=row["step_id"],
        )

    def add_event(
        self,
        task_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        agent: str | None = None,
        step_id: str | None = None,
    ) -> None:
        """追加不可变过程事件，用于状态监控和事后审计。"""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO events (
                    task_id, agent, step_id, event_type, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    agent,
                    step_id,
                    event_type,
                    _json_dumps(payload or {}),
                    _utc_now(),
                ),
            )

    def save_checkpoint(self, task_id: str, step_id: str, output: Any) -> Path:
        """将成功步骤的输出写入检查点，供任务恢复时跳过已完成工作。"""
        safe_step_id = step_id.replace("/", "__")
        return write_json(
            self.task_dir(task_id) / "checkpoints" / (safe_step_id + ".json"),
            {"step_id": step_id, "output": output},
        )

    def add_artifact(
        self,
        task_id: str,
        path: str | Path,
        *,
        step_id: str | None = None,
        agent: str | None = None,
        artifact_type: str = "file",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """登记输出文件及其 SHA-256 摘要，支持结果完整性核验。"""
        # 仅普通文件计算摘要；目录或尚未落盘的路径仍可登记，但摘要留空。
        artifact_path = Path(path)
        sha256 = None
        if artifact_path.is_file():
            digest = hashlib.sha256()
            with artifact_path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            sha256 = digest.hexdigest()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO artifacts (
                    artifact_id, task_id, step_id, agent, artifact_type,
                    path, sha256, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex,
                    task_id,
                    step_id,
                    agent,
                    artifact_type,
                    str(artifact_path),
                    sha256,
                    _json_dumps(metadata or {}),
                    _utc_now(),
                ),
            )

    def get_task(self, task_id: str) -> dict[str, Any]:
        """汇总任务主体、智能体运行和事件，返回完整可读视图。"""
        with self._connect() as connection:
            task = connection.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            if task is None:
                raise KeyError("Unknown task: %s" % task_id)
            runs = connection.execute(
                """
                SELECT * FROM agent_runs
                WHERE task_id = ?
                ORDER BY started_at, rowid
                """,
                (task_id,),
            ).fetchall()
            events = connection.execute(
                """
                SELECT * FROM events
                WHERE task_id = ?
                ORDER BY event_id
                """,
                (task_id,),
            ).fetchall()
        data = dict(task)
        json_fields: tuple[tuple[str, dict[str, Any]], ...] = (
            ("original_input_json", {}),
            ("structured_request_json", {}),
            ("execution_plan_json", {}),
            ("final_output_json", {}),
        )
        for key, default in json_fields:
            data[key[:-5]] = _json_loads(data.pop(key), default)
        data["agent_runs"] = [self._decode_run(dict(row)) for row in runs]
        data["events"] = [self._decode_event(dict(row)) for row in events]
        return data

    def completed_step_outputs(self, task_id: str) -> dict[str, Any]:
        """读取已成功顶层步骤的输出，作为断点恢复依据。

        带 ``/`` 的步骤编号表示 EEA 内部子智能体运行，不属于顶层执行计划，
        因此不会被当作可跳过的主步骤。
        """
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT step_id, output_json FROM agent_runs
                WHERE task_id = ? AND status = ? AND step_id NOT LIKE '%/%'
                ORDER BY ended_at
                """,
                (task_id, RunStatus.SUCCEEDED.value),
            ).fetchall()
        return {
            row["step_id"]: _json_loads(row["output_json"], {})
            for row in rows
        }

    @staticmethod
    def _decode_run(data: dict[str, Any]) -> dict[str, Any]:
        """解码智能体运行记录中的输入和输出 JSON 字段。"""
        data["input"] = _json_loads(data.pop("input_json"), {})
        data["output"] = _json_loads(data.pop("output_json"), {})
        return data

    @staticmethod
    def _decode_event(data: dict[str, Any]) -> dict[str, Any]:
        """解码事件载荷 JSON 字段。"""
        data["payload"] = _json_loads(data.pop("payload_json"), {})
        return data
