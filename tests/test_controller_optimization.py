"""验证 CIA 贝叶斯参数优化、目标函数和数据集隔离行为。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from agri_carla.controllers.controller_optimization import (
    ControllerSelfOptimizationConfig,
    GaussianProcessBayesianOptimizer,
    TrackingObjectiveConfig,
    evaluate_tracking_result,
)
from agri_carla.controllers.interface_agent import ControllerInterfaceAgent
from agri_carla.evaluation.performance_evaluation_subagent import PerformanceEvaluationSubagent
from agri_carla.agents.experiment_execution import ExperimentExecutionAgent
from agri_carla.simulation.tracking_experiment import TrackingExperimentConfig


class ControllerOptimizationTests(unittest.TestCase):
    """覆盖优化器数学行为及 CIA/EEA 集成接口。"""

    def test_bayesian_optimizer_never_replaces_best_with_worse_candidate(self):
        """较差的新候选不应覆盖历史最优参数和目标值。"""
        optimizer = GaussianProcessBayesianOptimizer(
            initial_parameters={"pid.kp": 1.0},
            search_space={"pid.kp": (0.0, 2.0)},
            initial_random_trials=2,
            candidate_pool_size=256,
            random_seed=7,
            kernel_length_scale=0.35,
            gp_noise=1e-6,
            expected_improvement_xi=0.01,
        )
        baseline, _ = optimizer.ask()
        self.assertEqual(baseline["pid.kp"], 1.0)
        optimizer.tell(10.0)

        improved, _ = optimizer.ask()
        self.assertGreaterEqual(improved["pid.kp"], 0.0)
        self.assertLessEqual(improved["pid.kp"], 2.0)
        optimizer.tell(5.0)
        best_after_improvement = dict(optimizer.best_parameters)

        bayesian_candidate, metadata = optimizer.ask()
        self.assertEqual(metadata["candidate_type"], "bayesian_acquisition")
        self.assertGreaterEqual(bayesian_candidate["pid.kp"], 0.0)
        self.assertLessEqual(bayesian_candidate["pid.kp"], 2.0)
        optimizer.tell(8.0)
        self.assertEqual(optimizer.best_score, 5.0)
        self.assertEqual(optimizer.best_parameters, best_after_improvement)

    def test_bayesian_optimizer_is_reproducible_for_a_fixed_seed(self):
        """相同随机种子与观测序列应产生完全相同的候选参数。"""
        kwargs = {
            "initial_parameters": {"pid.kp": 1.0, "pid.ki": 0.2},
            "search_space": {
                "pid.kp": (0.2, 2.0),
                "pid.ki": (0.0, 0.8),
            },
            "initial_random_trials": 3,
            "candidate_pool_size": 256,
            "random_seed": 42,
            "kernel_length_scale": 0.35,
            "gp_noise": 1e-6,
            "expected_improvement_xi": 0.01,
        }
        first = GaussianProcessBayesianOptimizer(**kwargs)
        second = GaussianProcessBayesianOptimizer(**kwargs)
        for score in (1.0, 0.7, 0.8, 0.5):
            first_candidate, _ = first.ask()
            second_candidate, _ = second.ask()
            self.assertEqual(first_candidate, second_candidate)
            first.tell(score)
            second.tell(score)

    def test_bayesian_optimizer_improves_a_smooth_two_parameter_objective(self):
        """优化器应能在平滑二维目标上明显优于默认参数。"""
        optimizer = GaussianProcessBayesianOptimizer(
            initial_parameters={"x": 0.9, "y": 0.1},
            search_space={"x": (0.0, 1.0), "y": (0.0, 1.0)},
            initial_random_trials=6,
            candidate_pool_size=1024,
            random_seed=42,
            kernel_length_scale=0.35,
            gp_noise=1e-6,
            expected_improvement_xi=0.01,
        )
        baseline_score = None
        for _ in range(20):
            candidate, _ = optimizer.ask()
            score = (
                (candidate["x"] - 0.23) ** 2
                + (candidate["y"] - 0.67) ** 2
            )
            if baseline_score is None:
                baseline_score = score
            optimizer.tell(score)

        self.assertIsNotNone(baseline_score)
        self.assertLess(optimizer.best_score, float(baseline_score) * 0.05)

    def test_tracking_objective_uses_lateral_speed_and_heading_metrics(self):
        """综合目标应成功读取横向、速度和航向三类误差。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = _write_fake_result(Path(temp_dir), quality=0.4)
            evaluation = evaluate_tracking_result(
                result,
                TrackingObjectiveConfig(),
            )
        self.assertEqual(evaluation["status"], "completed")
        self.assertGreater(evaluation["objective"], 0.0)
        self.assertAlmostEqual(evaluation["lateral_rmse_m"], 0.4)
        self.assertAlmostEqual(evaluation["speed_rmse_kmh"], 0.4)

    def test_cia_runs_trials_and_writes_best_controller_configuration(self):
        """CIA 应完成多轨迹多轮优化、独立测试并保存全部结果文件。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_config = TrackingExperimentConfig(
                output_dir=temp_dir,
                controller_optimization={"enabled": True},
            )
            optimization = ControllerSelfOptimizationConfig(
                enabled=True,
                max_trials=6,
                confirm_best=True,
                initial_random_trials=3,
                candidate_pool_size=512,
                random_seed=42,
                search_space={
                    "pure_pursuit.fwd_lookahead_min": (0.6, 3.0),
                },
            )

            trial_calls = []

            def fake_trial(trial_config, trajectory_csv):
                trial_calls.append(str(trajectory_csv))
                lookahead = trial_config.controller.pure_pursuit.fwd_lookahead_min
                dataset_offset = 0.01 if "optimization_b" in str(trajectory_csv) else 0.0
                quality = abs(lookahead - 1.36) + 0.05 + dataset_offset
                return _write_fake_result(
                    Path(trial_config.output_dir),
                    quality=quality,
                )

            progress_events = []
            result = ControllerInterfaceAgent.run_self_optimization(
                config=optimization,
                parameter_optimization_csvs=[
                    "optimization_a.csv",
                    "optimization_b.csv",
                ],
                independent_test_csvs=["independent_test.csv"],
                experiment_config=experiment_config,
                run_trial=fake_trial,
                progress_callback=progress_events.append,
            )

            self.assertTrue(Path(result["trial_history"]).exists())
            self.assertTrue(Path(result["best_controller_config"]).exists())
            self.assertTrue(Path(result["optimization_summary"]).exists())
            self.assertLessEqual(
                result["best_objective"],
                result["baseline_objective"],
            )
            history = pd.read_csv(result["trial_history"])
            self.assertEqual(len(history), 6)
            self.assertIn("bayesian_acquisition", history["candidate_type"].tolist())
            self.assertEqual(len(trial_calls), 16)
            self.assertEqual(result["independent_test"]["trajectory_count"], 1)
            self.assertTrue(
                Path(result["independent_test"]["comparison_table"]).exists()
            )
            completed_trials = [
                event
                for event in progress_events
                if event["phase"] == "trial_completed"
            ]
            self.assertEqual(len(completed_trials), 6)
            self.assertEqual(completed_trials[-1]["trial"], 6)

    def test_eea_discovers_dataset_csvs_without_including_other_files(self):
        """EEA 只按文件名排序收集 CSV，不把说明文件加入实验集。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "parameter_optimization_set"
            dataset_dir.mkdir()
            (dataset_dir / "b.csv").touch()
            (dataset_dir / "a.csv").touch()
            (dataset_dir / "README.md").touch()

            csvs = ExperimentExecutionAgent()._prepare_dataset(
                directory=dataset_dir,
                fallback_csv=None,
                processed_dataset_dir=None,
                subset_name="parameter_optimization_set",
            )

        self.assertEqual([path.name for path in csvs], ["a.csv", "b.csv"])

    def test_eea_rejects_an_empty_required_optimization_set(self):
        """启用优化时，空参数优化集应被明确拒绝。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "contains no CSV files"):
                ExperimentExecutionAgent()._prepare_dataset(
                    directory=temp_dir,
                    fallback_csv=None,
                    processed_dataset_dir=None,
                    subset_name="parameter_optimization_set",
                )

    def test_disabled_switch_runs_a_single_tracking_experiment(self):
        """优化开关关闭后只应调用一次固定参数跟踪运行器。"""
        expected = {"tracking_metrics_summary": "single_run.csv"}
        with patch(
            "agri_carla.agents.experiment_execution.TrackingExperimentRunner"
        ) as runner_class:
            runner_class.return_value.run.return_value = expected
            result = ExperimentExecutionAgent().run_trajectory_tracking(
                trajectory_csv="synthetic.csv",
                config=TrackingExperimentConfig(
                    controller_optimization={"enabled": False},
                ),
            )

        self.assertEqual(result, expected)
        runner_class.assert_called_once()
        _, runner_kwargs = runner_class.call_args
        self.assertIs(
            runner_kwargs["controller_agent_class"],
            ControllerInterfaceAgent,
        )
        self.assertIs(
            runner_kwargs["performance_evaluation_subagent_class"],
            PerformanceEvaluationSubagent,
        )
        runner_class.return_value.run.assert_called_once_with("synthetic.csv")

    def test_enabled_switch_uses_cia_self_optimization_entrypoint(self):
        """优化开关开启后应进入现有 CIA 的自优化方法。"""
        expected = {"optimization_enabled": True}
        with patch.object(
            ControllerInterfaceAgent,
            "run_self_optimization",
            return_value=expected,
        ) as optimize:
            result = ExperimentExecutionAgent().run_trajectory_tracking(
                trajectory_csv="synthetic.csv",
                config=TrackingExperimentConfig(
                    controller_optimization={
                        "enabled": True,
                        "max_trials": 2,
                    },
                ),
            )

        self.assertEqual(result, expected)
        optimize.assert_called_once()

    def test_single_trajectory_optimization_uses_processed_csv(self):
        """单轨迹调用应先标准化，并使用处理后的 CSV 参与优化。"""
        expected = {"optimization_enabled": True}
        eea = ExperimentExecutionAgent()
        with patch.object(
            eea,
            "preprocess_trajectory",
            return_value=Path("processed.csv"),
        ) as preprocess, patch.object(
            ControllerInterfaceAgent,
            "run_self_optimization",
            return_value=expected,
        ) as optimize:
            result = eea.run_trajectory_tracking(
                trajectory_csv="raw.csv",
                processed_csv="processed.csv",
                config=TrackingExperimentConfig(
                    controller_optimization={"enabled": True, "max_trials": 2},
                ),
            )

        self.assertEqual(result, expected)
        preprocess.assert_called_once_with("raw.csv", "processed.csv")
        self.assertEqual(
            optimize.call_args.kwargs["parameter_optimization_csvs"],
            [Path("processed.csv")],
        )


def _write_fake_result(output_dir: Path, quality: float) -> dict[str, str]:
    """生成误差大小可控的最小跟踪结果，替代耗时的真实 CARLA 实验。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "tracking_metrics_summary.csv"
    tick_path = output_dir / "tick_records.csv"
    pd.DataFrame([{
        "valid_ratio": 1.0,
        "valid_lateral_mean_abs_m": quality * 0.8,
        "valid_lateral_rmse_m": quality,
        "valid_lateral_p95_abs_m": quality * 1.2,
        "valid_lateral_max_abs_m": quality * 1.5,
        "valid_heading_rmse_deg": quality * 10.0,
    }]).to_csv(metrics_path, index=False)
    pd.DataFrame([
        {
            "actual_abs_kmh": 5.0 + quality,
            "target_abs_kmh": 5.0,
            "in_reference_time": True,
            "metric_valid": True,
        },
        {
            "actual_abs_kmh": 5.0 - quality,
            "target_abs_kmh": 5.0,
            "in_reference_time": True,
            "metric_valid": True,
        },
    ]).to_csv(tick_path, index=False)
    return {
        "tracking_metrics_summary": str(metrics_path),
        "tick_records": str(tick_path),
    }


if __name__ == "__main__":
    unittest.main()
