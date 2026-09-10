"""农业 CARLA 平台唯一的 CrewAI 顶层 Flow。"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from crewai.flow.flow import Flow, listen, start
from crewai.flow.persistence import SQLiteFlowPersistence, persist

from agri_carla.orchestration import (
    ApplicationRunLedger,
    DependencyUnavailableError,
    ExecutionContext,
    ExecutionPlan,
    NeedsUserInputError,
    RunStatus,
)
from agri_carla.utils.io import write_json

from .crew import AgriculturalAgentCrewRunner
from .state import AgriculturalFlowState


_FLOW_DATABASE = (
    Path(os.environ.get("AGRI_CARLA_RUNTIME_DIR", "runtime"))
    / "crewai_flow_states.sqlite3"
)
_FLOW_DATABASE.parent.mkdir(parents=True, exist_ok=True)


@persist(SQLiteFlowPersistence(str(_FLOW_DATABASE)))
class AgriculturalSimulationFlow(Flow[AgriculturalFlowState]):
    """规划、执行、监控并恢复一次农业自主导航仿真任务。"""

    def __init__(
        self,
        *,
        store: ApplicationRunLedger,
        crew_runner: AgriculturalAgentCrewRunner,
        retry_delay_s: float = 1.0,
    ):
        super().__init__()
        self._store = store
        self._crew_runner = crew_runner
        self._retry_delay_s = max(0.0, float(retry_delay_s))

    @start()
    def plan_task(self) -> dict[str, Any]:
        """由 PGA Crew 解析输入、校验请求并生成依赖计划。"""
        if not self.state.task_id:
            self.state.task_id = str(self.state.id)
        task_id = self.state.task_id
        if self.state.structured_request and self.state.execution_plan:
            self._store.add_event(
                task_id,
                "FLOW_PLANNING_RESTORED",
                {"source": self.state.planning_source or "resume"},
                agent="PGA",
                step_id="pga",
            )
            return {
                "structured_request": self.state.structured_request,
                "execution_plan": self.state.execution_plan,
            }

        context = ExecutionContext(task_id, self._store)
        self.state.status = RunStatus.PLANNING.value
        self.state.current_step = "pga"
        self._store.update_task(task_id, RunStatus.PLANNING, current_step="pga")
        max_attempts = 2
        for attempt in range(1, max_attempts + 1):
            run_id = context.start(
                "PGA",
                "pga",
                "parse_validate_and_plan",
                attempt=attempt,
                input_payload=self._safe_input_summary(self.state.original_request),
            )
            try:
                planning = self._crew_runner.run(
                    "PGA",
                    task_id=task_id,
                    step_id="pga",
                    operation="parse_validate_and_plan",
                    attempt=attempt,
                    payload=self.state.original_request,
                )
                if not isinstance(planning.get("structured_request"), dict):
                    raise RuntimeError(
                        "PGA did not return a structured_request object."
                    )
                if not isinstance(planning.get("execution_plan"), dict):
                    raise RuntimeError(
                        "PGA did not return an execution_plan object."
                    )
                context.succeed(run_id, planning)
                self.state.structured_request = dict(planning["structured_request"])
                self.state.execution_plan = dict(planning["execution_plan"])
                self.state.validation_report = dict(planning.get("validation_report", {}))
                self.state.planning_source = str(planning.get("source", "crewai_pga"))
                self.state.status = RunStatus.READY.value
                self.state.current_step = ""
                self._store.save_planning(
                    task_id,
                    self.state.structured_request,
                    self.state.execution_plan,
                    self.state.validation_report,
                )
                self._store.update_task(task_id, RunStatus.READY)
                return planning
            except NeedsUserInputError as exc:
                self._planning_failure(context, run_id, exc, RunStatus.NEEDS_INPUT)
                raise
            except Exception as exc:
                retryable = self._is_retryable(exc) and attempt < max_attempts
                if retryable:
                    context.fail(run_id, exc, status=RunStatus.RETRYING)
                    self.state.status = RunStatus.RETRYING.value
                    self._store.update_task(
                        task_id,
                        RunStatus.RETRYING,
                        current_step="pga",
                        error=exc,
                    )
                    context.progress(
                        "PGA",
                        "pga",
                        "CrewAI Flow will retry transient PGA planning failure.",
                        attempt=attempt,
                        max_attempts=max_attempts,
                        error=str(exc),
                    )
                    time.sleep(self._retry_delay_s)
                    continue
                status = RunStatus.BLOCKED if self._is_dependency_failure(exc) else RunStatus.FAILED
                self._planning_failure(context, run_id, exc, status)
                raise
        raise RuntimeError("PGA planning exhausted all retry attempts.")

    @listen(plan_task)
    def execute_plan(self, _planning: dict[str, Any]) -> dict[str, Any]:
        """按 PGA 依赖计划启动相应 CrewAI Agent，并保存每步检查点。"""
        if self.state.plan_only:
            return {"plan_only": True}
        task_id = self.state.task_id
        try:
            self._require_scene_confirmation()
        except NeedsUserInputError as exc:
            self.state.status = RunStatus.NEEDS_INPUT.value
            self.state.current_step = "scene_review"
            self.state.error_type = type(exc).__name__
            self.state.error_message = str(exc)
            self._store.update_task(
                task_id,
                RunStatus.NEEDS_INPUT,
                current_step="scene_review",
                error=exc,
            )
            self._write_failure_report(exc, RunStatus.NEEDS_INPUT)
            raise
        plan = ExecutionPlan.from_dict(self.state.execution_plan)
        plan.validate_dependencies()
        completed = self._store.completed_step_outputs(task_id)
        self.state.completed_steps = dict(completed)
        for step in plan.steps:
            if step.step_id in completed:
                self.state.results[self._result_key(step.operation)] = completed[step.step_id]
                self._store.add_event(
                    task_id,
                    "STEP_SKIPPED_FROM_CHECKPOINT",
                    {"reason": "already_succeeded"},
                    agent=step.agent,
                    step_id=step.step_id,
                )

        self.state.status = RunStatus.RUNNING.value
        self._store.update_task(task_id, RunStatus.RUNNING)
        try:
            for step in plan.steps:
                if step.step_id in completed:
                    continue
                missing = [item for item in step.depends_on if item not in completed]
                if missing:
                    raise RuntimeError(
                        "Step %s cannot run; dependencies are incomplete: %s"
                        % (step.step_id, ", ".join(missing))
                    )
                self.state.current_step = step.step_id
                self._store.update_task(
                    task_id,
                    RunStatus.RUNNING,
                    current_step=step.step_id,
                )
                output = self._run_step_with_retry(step)
                completed[step.step_id] = output
                self.state.completed_steps[step.step_id] = output
                self.state.results[self._result_key(step.operation)] = output
                self._store.save_checkpoint(task_id, step.step_id, output)
            return dict(self.state.results)
        except Exception as exc:
            if isinstance(exc, NeedsUserInputError):
                status = RunStatus.NEEDS_INPUT
            else:
                status = (
                    RunStatus.BLOCKED
                    if self._is_dependency_failure(exc)
                    else RunStatus.FAILED
                )
            self.state.status = status.value
            self.state.error_type = type(exc).__name__
            self.state.error_message = str(exc)
            self._store.update_task(
                task_id,
                status,
                current_step=self.state.current_step,
                error=exc,
            )
            self._write_failure_report(exc, status)
            raise

    @listen(execute_plan)
    def finalize_task(self, _results: dict[str, Any]) -> dict[str, Any]:
        """形成稳定的最终输出并登记 Flow 级结果文件。"""
        status = RunStatus.READY if self.state.plan_only else RunStatus.SUCCEEDED
        output = {
            "task_id": self.state.task_id,
            "flow_state_id": str(self.state.id),
            "task_directory": str(self._store.task_dir(self.state.task_id)),
            "task_type": self.state.structured_request.get("task_type"),
            "status": status.value,
            "planning_source": self.state.planning_source,
            "structured_request": self.state.structured_request if self.state.plan_only else None,
            "execution_plan": self.state.execution_plan,
            "validation_report": self.state.validation_report,
            "results": dict(self.state.results),
        }
        if output["structured_request"] is None:
            output.pop("structured_request")
        self.state.status = status.value
        self.state.current_step = ""
        self.state.final_output = output
        self._store.update_task(
            self.state.task_id,
            status,
            final_output=output,
        )
        self._write_flow_report(output)
        return output

    def _require_scene_confirmation(self) -> None:
        """阻止未经用户复核的多模态场景候选进入 SGA。"""
        review = self.state.structured_request.get("scene_review")
        if not isinstance(review, dict):
            return
        if not bool(review.get("requires_confirmation", False)):
            return
        if str(review.get("status", "")).lower() == "confirmed":
            return
        preview_path = str(review.get("preview_path", "")).strip()
        message = (
            "Scene candidate requires user confirmation before SGA execution. "
            "Review the generated overlay"
        )
        if preview_path:
            message += " at %s" % preview_path
        message += (
            ". If correct, rerun with --request <task_directory>/structured_request.json "
            "--confirm-scene. Otherwise rerun planning with a more detailed description "
            "and optionally --correction-image."
        )
        raise NeedsUserInputError([message])

    def _run_step_with_retry(self, step) -> dict[str, Any]:
        context = ExecutionContext(self.state.task_id, self._store)
        last_error: BaseException | None = None
        for attempt in range(1, step.max_attempts + 1):
            run_id = context.start(
                step.agent,
                step.step_id,
                step.operation,
                attempt=attempt,
                input_payload={"task_type": self.state.structured_request.get("task_type")},
            )
            try:
                output = self._crew_runner.run(
                    step.agent,
                    task_id=self.state.task_id,
                    step_id=step.step_id,
                    operation=step.operation,
                    attempt=attempt,
                    payload=self.state.structured_request,
                )
                context.succeed(run_id, output)
                self._register_output_artifacts(context, step, output)
                return output
            except Exception as exc:
                last_error = exc
                retryable = self._is_retryable(exc) and attempt < step.max_attempts
                context.fail(
                    run_id,
                    exc,
                    status=RunStatus.RETRYING if retryable else RunStatus.FAILED,
                )
                if not retryable:
                    raise
                self.state.status = RunStatus.RETRYING.value
                self._store.update_task(
                    self.state.task_id,
                    RunStatus.RETRYING,
                    current_step=step.step_id,
                    error=exc,
                )
                context.progress(
                    step.agent,
                    step.step_id,
                    "CrewAI Flow will retry this transient failure.",
                    attempt=attempt,
                    max_attempts=step.max_attempts,
                    error=str(exc),
                )
                if self._retry_delay_s:
                    time.sleep(self._retry_delay_s)
                self.state.status = RunStatus.RUNNING.value
        raise last_error or RuntimeError("Unknown CrewAI execution failure.")

    def _planning_failure(self, context, run_id, exc, status) -> None:
        context.fail(run_id, exc, status=status)
        self.state.status = status.value
        self.state.error_type = type(exc).__name__
        self.state.error_message = str(exc)
        self._store.update_task(
            self.state.task_id,
            status,
            current_step="pga",
            error=exc,
        )

    def _write_flow_report(self, output: dict[str, Any]) -> None:
        report_path = self.state.structured_request.get("flow_report")
        if not report_path:
            return
        write_json(report_path, output)
        context = ExecutionContext(self.state.task_id, self._store)
        try:
            context.artifact(
                report_path,
                agent="CrewAIFlow",
                step_id="final",
                artifact_type="flow_report",
            )
        except Exception as exc:
            context.progress(
                "CrewAIFlow",
                "final",
                "The Flow report was written, but artifact registration failed.",
                artifact_path=str(report_path),
                artifact_error=str(exc),
            )

    def _write_failure_report(self, exc: BaseException, status: RunStatus) -> None:
        output = {
            "task_id": self.state.task_id,
            "task_type": self.state.structured_request.get("task_type"),
            "status": status.value,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "results": dict(self.state.results),
        }
        self._write_flow_report(output)

    @staticmethod
    def _result_key(operation: str) -> str:
        return {
            "generate_scene": "road_scene",
            "place_vegetation": "vegetation",
            "generate_machinery_template": "machinery",
            "prepare_controller": "controller",
            "run_trajectory_tracking": "trajectory_tracking",
            "run_full_process_navigation": "full_process_navigation",
        }.get(operation, operation)

    @staticmethod
    def _safe_input_summary(request: dict[str, Any]) -> dict[str, Any]:
        return {
            "task_type": request.get("task_type"),
            "has_natural_language_request": bool(request.get("natural_language_request")),
            "image_count": len(request.get("image_paths", [])),
            "data_file_count": len(request.get("data_files", [])),
            "keys": sorted(request.keys()),
        }

    @staticmethod
    def _is_retryable(exc: BaseException) -> bool:
        if isinstance(exc, (ConnectionError, TimeoutError, DependencyUnavailableError)):
            return True
        text = str(exc).lower()
        return any(
            marker in text
            for marker in (
                "connection refused",
                "failed to connect",
                "connection failed",
                "connection timed out",
                "time-out of",
                "temporarily unavailable",
                "cannot connect to carla worker",
                "llm connection failed",
                "unexpected_eof",
                "unexpected eof",
                "urlopen error",
                "ssl:",
            )
        )

    @classmethod
    def _is_dependency_failure(cls, exc: BaseException) -> bool:
        return isinstance(
            exc,
            (ConnectionError, TimeoutError, DependencyUnavailableError, FileNotFoundError),
        ) or cls._is_retryable(exc)

    @staticmethod
    def _register_output_artifacts(context, step, output) -> None:
        if not isinstance(output, dict):
            return
        for key, value in output.items():
            if not isinstance(value, str) or not value:
                continue
            path = Path(value)
            if not path.is_file():
                continue
            try:
                context.artifact(
                    path,
                    agent=step.agent,
                    step_id=step.step_id,
                    artifact_type=key,
                )
            except Exception as exc:
                context.progress(
                    step.agent,
                    step.step_id,
                    "The output exists, but artifact registration failed.",
                    artifact_path=str(path),
                    artifact_error=str(exc),
                )
