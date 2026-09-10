"""仿真实验执行智能体（EEA）的统一入口。

EEA 负责组织实验，不在本文件中重新实现轨迹处理、控制器或评价算法。
轨迹处理智能体（TPS）和实验评价智能体（PES）是 EEA 的从属智能体；
控制器接口智能体（CIA）与 EEA 同层，由 EEA 在实验运行时调用。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agri_carla.controllers.interface_agent import ControllerInterfaceAgent
from agri_carla.evaluation.performance_evaluation_subagent import PerformanceEvaluationSubagent
from agri_carla.simulation.full_process_experiment import FullProcessExperimentConfig, FullProcessNavigationExperiment
from agri_carla.simulation.tracking_experiment import TrackingExperimentConfig, TrackingExperimentRunner
from agri_carla.trajectory.preprocessing_subagent import TrajectoryPreprocessingSubagent


class ExperimentExecutionAgent:
    """Coordinates experiments through internal TPS/PES subagents and peer CIA.

    TPS and PES serve only the simulation-experiment workflow. CIA remains a
    peer-level agent and is called by EEA when controller services are needed.
    """

    def __init__(self, controller_agent_class=ControllerInterfaceAgent):
        """注入 CIA 类型，并创建 EEA 所属的 TPS 与 PES 服务。"""
        self.controller_agent_class = controller_agent_class
        self.trajectory_preprocessing_subagent = TrajectoryPreprocessingSubagent()
        self.performance_evaluation_subagent_class = PerformanceEvaluationSubagent

    def preprocess_trajectory(
        self,
        trajectory_csv: str | Path,
        processed_csv: str | Path,
        context=None,
    ) -> Path:
        """调用 TPS 将一条原始轨迹转换为平台统一格式。

        当传入运行上下文时，同时记录 TPS 的开始、成功或失败状态，并把
        标准化 CSV 注册为实验产物。产物登记失败不会删除已经生成的轨迹。
        """
        step_id = "eea_tracking/tps/%s" % Path(trajectory_csv).stem
        run_id = None
        if context is not None:
            run_id = context.start(
                "TPS",
                step_id,
                "standardize_and_recognize_direction",
                input_payload={
                    "trajectory_csv": str(trajectory_csv),
                    "processed_csv": str(processed_csv),
                },
            )
        try:
            output = self.trajectory_preprocessing_subagent.process(
                trajectory_csv,
                processed_csv,
            )
            if context is not None and run_id is not None:
                context.succeed(run_id, {"processed_csv": str(output)})
                try:
                    context.artifact(
                        output,
                        agent="TPS",
                        step_id=step_id,
                        artifact_type="processed_trajectory",
                    )
                except Exception as artifact_error:
                    context.progress(
                        "TPS",
                        step_id,
                        "The processed trajectory was produced, but artifact "
                        "registration failed.",
                        artifact_error=str(artifact_error),
                    )
            return output
        except Exception as exc:
            if context is not None and run_id is not None:
                context.fail(run_id, exc)
            raise

    def run_trajectory_tracking(
        self,
        trajectory_csv: str | Path | None = None,
        processed_csv: str | Path | None = None,
        config: TrackingExperimentConfig | None = None,
        parameter_optimization_set_dir: str | Path | None = None,
        independent_test_set_dir: str | Path | None = None,
        processed_dataset_dir: str | Path | None = None,
        context=None,
    ) -> dict[str, Any]:
        """执行固定参数跟踪实验或带贝叶斯优化的多轮跟踪实验。

        ``controller_optimization.enabled`` 是两种运行方式的唯一开关。
        开启时先准备参数优化集和独立测试集，再把每组候选参数交给 CIA；
        关闭时仅对指定轨迹运行一次固定参数实验。
        """
        experiment_config = config or TrackingExperimentConfig()
        optimization_config = self.controller_agent_class.self_optimization_config(
            experiment_config.controller_optimization
        )
        if optimization_config.enabled:
            # 兼容单轨迹调用：没有显式优化集目录时，将当前轨迹作为优化数据。
            fallback_optimization_csv = trajectory_csv
            if (
                parameter_optimization_set_dir is None
                and trajectory_csv is not None
                and processed_csv is not None
            ):
                if context is None:
                    fallback_optimization_csv = self.preprocess_trajectory(
                        trajectory_csv,
                        processed_csv,
                    )
                else:
                    fallback_optimization_csv = self.preprocess_trajectory(
                        trajectory_csv,
                        processed_csv,
                        context=context,
                    )
            optimization_csvs = self._prepare_dataset(
                directory=parameter_optimization_set_dir,
                fallback_csv=fallback_optimization_csv,
                processed_dataset_dir=processed_dataset_dir,
                subset_name="parameter_optimization_set",
                context=context,
            )
            test_csvs = self._prepare_dataset(
                directory=independent_test_set_dir,
                fallback_csv=None,
                processed_dataset_dir=processed_dataset_dir,
                subset_name="independent_test_set",
                required=False,
                context=context,
            )
            return self._run_tracking_with_status(
                context=context,
                operation="bayesian_controller_optimization",
                callback=lambda: self.controller_agent_class.run_self_optimization(
                    config=optimization_config,
                    parameter_optimization_csvs=optimization_csvs,
                    independent_test_csvs=test_csvs,
                    experiment_config=experiment_config,
                    run_trial=lambda trial_config, trial_csv: TrackingExperimentRunner(
                        trial_config,
                        controller_agent_class=self.controller_agent_class,
                        performance_evaluation_subagent_class=(
                            self.performance_evaluation_subagent_class
                        ),
                    ).run(trial_csv),
                    progress_callback=(
                        None
                        if context is None
                        else lambda update: self._report_controller_progress(
                            context,
                            update,
                        )
                    ),
                ),
            )

        if trajectory_csv is None:
            raise ValueError(
                "trajectory_csv is required when controller optimization is disabled."
            )
        csv_to_run = trajectory_csv
        if processed_csv is not None:
            # 固定参数模式允许同时完成原始轨迹标准化和跟踪验证。
            if context is None:
                self.preprocess_trajectory(trajectory_csv, processed_csv)
            else:
                self.preprocess_trajectory(
                    trajectory_csv,
                    processed_csv,
                    context=context,
                )
            csv_to_run = processed_csv
        return self._run_tracking_with_status(
            context=context,
            operation="fixed_parameter_tracking_control",
            callback=lambda: TrackingExperimentRunner(
                experiment_config,
                controller_agent_class=self.controller_agent_class,
                performance_evaluation_subagent_class=(
                    self.performance_evaluation_subagent_class
                ),
            ).run(csv_to_run),
        )

    def _prepare_dataset(
        self,
        *,
        directory: str | Path | None,
        fallback_csv: str | Path | None,
        processed_dataset_dir: str | Path | None,
        subset_name: str,
        required: bool = True,
        context=None,
    ) -> list[Path]:
        """收集某一数据子集，并按需调用 TPS 生成标准化副本。

        参数优化集必须非空；独立测试集可缺省。若指定
        ``processed_dataset_dir``，原始 CSV 不会被覆盖，而是在对应子目录中
        生成同名标准化文件。
        """
        if directory:
            dataset_dir = Path(directory)
            if not dataset_dir.is_dir():
                raise FileNotFoundError(
                    f"{subset_name} directory does not exist: {dataset_dir}"
                )
            source_csvs = sorted(
                path for path in dataset_dir.glob("*.csv") if path.is_file()
            )
            if not source_csvs:
                raise ValueError(
                    f"{subset_name} contains no CSV files: {dataset_dir}"
                )
        elif fallback_csv is not None:
            source_csvs = [Path(fallback_csv)]
        elif required:
            raise ValueError(
                "Provide parameter_optimization_set_dir or trajectory_csv when "
                "controller optimization is enabled."
            )
        else:
            return []

        if processed_dataset_dir is None:
            return source_csvs

        output_dir = Path(processed_dataset_dir) / subset_name
        processed_csvs: list[Path] = []
        for source_csv in source_csvs:
            output_csv = output_dir / source_csv.name
            processed_csvs.append(
                self.preprocess_trajectory(source_csv, output_csv)
                if context is None
                else self.preprocess_trajectory(
                    source_csv,
                    output_csv,
                    context=context,
                )
            )
        return processed_csvs

    def run_full_process_navigation(
        self,
        config: FullProcessExperimentConfig | None = None,
        context=None,
    ) -> dict[str, str]:
        """运行全流程农业导航，并以 PES 身份登记评价过程状态。"""
        pes_run = None
        if context is not None:
            pes_run = context.start(
                "PES",
                "eea_full_process/pes",
                "record_and_evaluate_full_process",
            )
        try:
            result = FullProcessNavigationExperiment(
                config,
                performance_evaluation_subagent_class=(
                    self.performance_evaluation_subagent_class
                ),
            ).run()
            if context is not None and pes_run is not None:
                context.succeed(pes_run, result)
            return result
        except Exception as exc:
            if context is not None and pes_run is not None:
                context.fail(pes_run, exc)
            raise

    def _run_tracking_with_status(self, *, context, operation, callback):
        """在执行回调前后同步 CIA 与 PES 的运行状态。"""
        cia_run = None
        pes_run = None
        if context is not None:
            cia_run = context.start(
                "CIA",
                "eea_tracking/cia",
                operation,
            )
            pes_run = context.start(
                "PES",
                "eea_tracking/pes",
                "record_and_evaluate_tracking",
            )
        try:
            result = callback()
            if context is not None:
                if cia_run is not None:
                    context.succeed(cia_run, result)
                if pes_run is not None:
                    context.succeed(pes_run, result)
            return result
        except Exception as exc:
            if context is not None:
                if cia_run is not None:
                    context.fail(cia_run, exc)
                if pes_run is not None:
                    context.fail(pes_run, exc)
            raise

    @staticmethod
    def _report_controller_progress(context, update):
        """把 CIA 多轮参数优化的中间进度写入 CrewAI 运行记录。"""
        payload = dict(update)
        message = str(payload.pop("message", "Controller progress update."))
        context.progress(
            "CIA",
            "eea_tracking/cia",
            message,
            **payload,
        )
