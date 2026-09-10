"""CIA 内部的控制器贝叶斯优化、综合目标计算和独立测试流程。"""

from __future__ import annotations

import math
from datetime import datetime
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from agri_carla.controllers.gear_state_machine import GearStateConfig
from agri_carla.controllers.interface_agent import ControllerInterfaceConfig
from agri_carla.controllers.pid import PIDConfig
from agri_carla.controllers.pure_pursuit import PurePursuitConfig
from agri_carla.utils.io import ensure_dir, write_json


# 七个待优化变量均映射到有限区间，其他控制器参数在各轮实验中保持不变。
DEFAULT_SEARCH_SPACE = {
    "pure_pursuit.fwd_lookahead_min": (0.6, 3.0),
    "pure_pursuit.fwd_lookahead_gain": (0.0, 0.8),
    "pure_pursuit.rev_lookahead_min": (0.3, 2.0),
    "pure_pursuit.rev_lookahead_gain": (0.0, 0.5),
    "pid.kp": (0.2, 2.0),
    "pid.ki": (0.0, 0.8),
    "pid.kd": (0.0, 0.2),
}


@dataclass(frozen=True)
class TrackingObjectiveConfig:
    """综合目标中各误差指标的权重、尺度和无效实验惩罚。"""
    metric_scope: str = "valid"
    # Metric weights
    lateral_mae_weight: float = 0.25
    lateral_rmse_weight: float = 0.35
    lateral_p95_weight: float = 0.20
    speed_rmse_weight: float = 0.15
    heading_rmse_weight: float = 0.05

    # Scale factors
    lateral_scale_m: float = 1.0
    speed_scale_kmh: float = 5.0
    heading_scale_deg: float = 30.0
    minimum_valid_ratio: float = 0.5

    # Penalties
    maximum_lateral_error_m: float = 5.0
    invalid_trial_penalty: float = 1000.0
    low_valid_ratio_penalty: float = 10.0
    excessive_error_penalty_per_m: float = 20.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TrackingObjectiveConfig":
        """从配置读取已支持的评价字段。"""
        if not data:
            return cls()
        valid = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in data.items() if key in valid})


@dataclass(frozen=True)
class ControllerSelfOptimizationConfig:
    """贝叶斯优化轮次、初始设计、代理模型和搜索空间配置。"""
    enabled: bool = False
    max_trials: int = 24
    confirm_best: bool = True
    method: str = "bayesian_optimization"
    initial_random_trials: int = 8
    candidate_pool_size: int = 4096
    random_seed: int = 42
    kernel_length_scale: float = 0.35
    gp_noise: float = 1e-6
    expected_improvement_xi: float = 0.01
    search_space: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_SEARCH_SPACE)
    )
    objective: TrackingObjectiveConfig = field(default_factory=TrackingObjectiveConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ControllerSelfOptimizationConfig":
        """解析并校验 CIA 参数自优化配置。"""
        if not data:
            return cls()
        values = dict(data)
        values["objective"] = TrackingObjectiveConfig.from_dict(values.get("objective"))
        if "search_space" in values:
            values["search_space"] = _parse_search_space(values["search_space"])
        valid = set(cls.__dataclass_fields__)
        config = cls(**{key: value for key, value in values.items() if key in valid})
        if config.method != "bayesian_optimization":
            raise ValueError(
                "Unsupported controller optimization method: "
                f"{config.method}. Use bayesian_optimization."
            )
        if config.max_trials < 1:
            raise ValueError("controller_optimization.max_trials must be at least 1.")
        if config.initial_random_trials < 1:
            raise ValueError(
                "controller_optimization.initial_random_trials must be at least 1."
            )
        if config.candidate_pool_size < 128:
            raise ValueError(
                "controller_optimization.candidate_pool_size must be at least 128."
            )
        if config.kernel_length_scale <= 0.0:
            raise ValueError(
                "controller_optimization.kernel_length_scale must be positive."
            )
        if config.gp_noise <= 0.0:
            raise ValueError("controller_optimization.gp_noise must be positive.")
        if config.expected_improvement_xi < 0.0:
            raise ValueError(
                "controller_optimization.expected_improvement_xi cannot be negative."
            )
        return config


class GaussianProcessBayesianOptimizer:
    """面向高成本 CARLA 实验的有界高斯过程贝叶斯优化器。"""

    def __init__(
        self,
        initial_parameters: dict[str, float],
        search_space: dict[str, tuple[float, float]],
        initial_random_trials: int,
        candidate_pool_size: int,
        random_seed: int,
        kernel_length_scale: float,
        gp_noise: float,
        expected_improvement_xi: float,
    ):
        """建立归一化参数空间、拉丁超立方初始设计和随机数生成器。"""
        self.search_space = search_space
        self.names = list(search_space)
        if not self.names:
            raise ValueError("Bayesian optimization requires at least one parameter.")
        self.lower = np.array(
            [search_space[name][0] for name in self.names],
            dtype=float,
        )
        self.upper = np.array(
            [search_space[name][1] for name in self.names],
            dtype=float,
        )
        if np.any(self.upper <= self.lower):
            raise ValueError("Every Bayesian optimization bound must satisfy lower < upper.")

        self.best_parameters = {
            name: _clip(initial_parameters[name], *search_space[name])
            for name in self.names
        }
        self.best_score = math.inf
        self.initial_random_trials = max(1, int(initial_random_trials))
        self.candidate_pool_size = int(candidate_pool_size)
        self.kernel_length_scale = float(kernel_length_scale)
        self.gp_noise = float(gp_noise)
        self.expected_improvement_xi = float(expected_improvement_xi)
        self.rng = np.random.default_rng(int(random_seed))
        self.initial_design = _latin_hypercube(
            max(0, self.initial_random_trials - 1),
            len(self.names),
            self.rng,
        )
        self.observed_x: list[np.ndarray] = []
        self.observed_y: list[float] = []
        self.pending: dict[str, Any] | None = None

    def ask(self) -> tuple[dict[str, float], dict[str, Any]]:
        """给出下一组待运行参数及其候选来源信息。

        第一次返回默认参数，随后完成拉丁超立方初始设计；积累足够观测后，
        使用高斯过程与期望改进函数选择新候选。
        """
        if self.pending is not None:
            raise RuntimeError("tell() must be called before requesting another candidate.")

        observation_count = len(self.observed_y)
        if observation_count == 0:
            normalized = self._encode(self.best_parameters)
            metadata = {
                "candidate_type": "baseline",
                "acquisition": "baseline",
                "acquisition_value": float("nan"),
                "posterior_mean": float("nan"),
                "posterior_std": float("nan"),
            }
        elif observation_count <= len(self.initial_design):
            normalized = self.initial_design[observation_count - 1].copy()
            metadata = {
                "candidate_type": "initial_design",
                "acquisition": "latin_hypercube",
                "acquisition_value": float("nan"),
                "posterior_mean": float("nan"),
                "posterior_std": float("nan"),
            }
        else:
            normalized, metadata = self._next_bayesian_candidate()

        candidate = self._decode(normalized)
        self.pending = {
            "candidate": candidate,
            "normalized": normalized,
            "metadata": metadata,
        }
        return candidate, metadata

    def tell(self, score: float) -> None:
        """把刚完成实验的综合目标反馈给优化器并更新历史最优值。"""
        if self.pending is None:
            raise RuntimeError("ask() must be called before tell().")

        candidate = self.pending["candidate"]
        normalized = np.asarray(self.pending["normalized"], dtype=float)
        observed_score = float(score)
        if not math.isfinite(observed_score):
            observed_score = 1e12
        self.observed_x.append(normalized)
        self.observed_y.append(observed_score)

        improved = observed_score < self.best_score
        if improved:
            self.best_score = observed_score
            self.best_parameters = dict(candidate)
        self.pending = None

    def _next_bayesian_candidate(self) -> tuple[np.ndarray, dict[str, Any]]:
        """拟合高斯过程并选择期望改进值最大的未测试候选。"""
        observed_x = np.asarray(self.observed_x, dtype=float)
        observed_y = np.asarray(self.observed_y, dtype=float)
        candidates = self._candidate_pool(observed_x)
        mean, std = _gaussian_process_predict(
            observed_x,
            observed_y,
            candidates,
            length_scale=self.kernel_length_scale,
            noise=self.gp_noise,
        )

        y_mean = float(np.mean(observed_y))
        y_std = float(np.std(observed_y))
        if y_std < 1e-12:
            y_std = 1.0
        best_standardized = (float(np.min(observed_y)) - y_mean) / y_std
        expected_improvement = _expected_improvement(
            mean,
            std,
            best_standardized,
            self.expected_improvement_xi,
        )
        best_index = int(np.argmax(expected_improvement))
        return candidates[best_index], {
            "candidate_type": "bayesian_acquisition",
            "acquisition": "expected_improvement",
            "acquisition_value": float(expected_improvement[best_index]),
            "posterior_mean": float(mean[best_index]),
            "posterior_std": float(std[best_index]),
        }

    def _candidate_pool(self, observed_x: np.ndarray) -> np.ndarray:
        """组合全局随机样本和当前最优附近样本，并排除已测试点。"""
        dimension = len(self.names)
        random_count = max(128, int(self.candidate_pool_size * 0.75))
        candidates = self.rng.random((random_count, dimension))

        if math.isfinite(self.best_score):
            best = self._encode(self.best_parameters)
            local_count = self.candidate_pool_size - random_count
            local_scale = max(
                0.03,
                0.25 / math.sqrt(max(1, len(self.observed_y))),
            )
            local = np.clip(
                best
                + self.rng.normal(
                    0.0,
                    local_scale,
                    size=(local_count, dimension),
                ),
                0.0,
                1.0,
            )
            candidates = np.vstack([candidates, local])

        if observed_x.size:
            distance = np.sqrt(
                np.sum(
                    (candidates[:, None, :] - observed_x[None, :, :]) ** 2,
                    axis=2,
                )
            )
            keep = np.min(distance, axis=1) > 1e-8
            candidates = candidates[keep]
        if len(candidates) == 0:
            return self.rng.random((1, dimension))
        return candidates

    def _encode(self, parameters: dict[str, float]) -> np.ndarray:
        """把物理参数按上下界映射到零至一的统一搜索空间。"""
        values = np.array([parameters[name] for name in self.names], dtype=float)
        return np.clip((values - self.lower) / (self.upper - self.lower), 0.0, 1.0)

    def _decode(self, normalized: np.ndarray) -> dict[str, float]:
        """将归一化向量还原为具有实际单位的控制器参数。"""
        values = self.lower + np.asarray(normalized, dtype=float) * (
            self.upper - self.lower
        )
        return {
            name: float(values[index])
            for index, name in enumerate(self.names)
        }


def _latin_hypercube(
    sample_count: int,
    dimension: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """生成覆盖各维分层区间的拉丁超立方初始样本。"""
    if sample_count <= 0:
        return np.empty((0, dimension), dtype=float)
    design = np.empty((sample_count, dimension), dtype=float)
    for index in range(dimension):
        intervals = rng.permutation(sample_count)
        design[:, index] = (
            intervals + rng.random(sample_count)
        ) / sample_count
    return design


def _gaussian_process_predict(
    observed_x: np.ndarray,
    observed_y: np.ndarray,
    candidates: np.ndarray,
    *,
    length_scale: float,
    noise: float,
) -> tuple[np.ndarray, np.ndarray]:
    """根据已有参数—目标值样本预测候选点的后验均值和标准差。"""
    y_mean = float(np.mean(observed_y))
    y_std = float(np.std(observed_y))
    if y_std < 1e-12:
        y_std = 1.0
    standardized_y = (observed_y - y_mean) / y_std

    covariance = _matern52_kernel(
        observed_x,
        observed_x,
        length_scale,
    )
    cross_covariance = _matern52_kernel(
        observed_x,
        candidates,
        length_scale,
    )
    candidate_variance = np.ones(len(candidates), dtype=float)

    # 逐步增加对角抖动项，降低协方差矩阵接近奇异时的数值失败风险。
    jitter = max(float(noise), 1e-12)
    cholesky = None
    for _ in range(8):
        try:
            cholesky = np.linalg.cholesky(
                covariance + np.eye(len(observed_x)) * jitter
            )
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0
    if cholesky is None:
        return (
            np.zeros(len(candidates), dtype=float),
            np.ones(len(candidates), dtype=float),
        )

    alpha = np.linalg.solve(
        cholesky.T,
        np.linalg.solve(cholesky, standardized_y),
    )
    posterior_mean = cross_covariance.T @ alpha
    projected = np.linalg.solve(cholesky, cross_covariance)
    posterior_variance = np.maximum(
        candidate_variance - np.sum(projected ** 2, axis=0),
        1e-12,
    )
    return posterior_mean, np.sqrt(posterior_variance)


def _matern52_kernel(
    first: np.ndarray,
    second: np.ndarray,
    length_scale: float,
) -> np.ndarray:
    """计算归一化参数空间中的 Matérn 5/2 协方差矩阵。"""
    scaled = (
        first[:, None, :] - second[None, :, :]
    ) / max(float(length_scale), 1e-12)
    distance = np.sqrt(np.sum(scaled ** 2, axis=2))
    sqrt_five_distance = math.sqrt(5.0) * distance
    return (
        1.0
        + sqrt_five_distance
        + (5.0 / 3.0) * distance ** 2
    ) * np.exp(-sqrt_five_distance)


def _expected_improvement(
    posterior_mean: np.ndarray,
    posterior_std: np.ndarray,
    best_standardized: float,
    exploration_xi: float,
) -> np.ndarray:
    """计算最小化问题的期望改进采集函数。

    后验均值反映预测目标值，后验标准差反映未知程度；期望改进在利用当前较优
    区域和探索高不确定区域之间取得平衡。
    """
    improvement = (
        float(best_standardized)
        - posterior_mean
        - float(exploration_xi)
    )
    safe_std = np.maximum(posterior_std, 1e-12)
    z_value = improvement / safe_std
    expected = (
        improvement * _normal_cdf(z_value)
        + safe_std * _normal_pdf(z_value)
    )
    return np.where(posterior_std > 1e-12, np.maximum(expected, 0.0), 0.0)


def _normal_pdf(value: np.ndarray) -> np.ndarray:
    """标准正态分布概率密度函数。"""
    return np.exp(-0.5 * value ** 2) / math.sqrt(2.0 * math.pi)


def _normal_cdf(value: np.ndarray) -> np.ndarray:
    """标准正态分布累积分布函数的数值近似。"""
    absolute = np.abs(value)
    t_value = 1.0 / (1.0 + 0.2316419 * absolute)
    polynomial = (
        0.319381530 * t_value
        - 0.356563782 * t_value ** 2
        + 1.781477937 * t_value ** 3
        - 1.821255978 * t_value ** 4
        + 1.330274429 * t_value ** 5
    )
    upper_tail = _normal_pdf(absolute) * polynomial
    return np.where(value >= 0.0, 1.0 - upper_tail, upper_tail)


class ControllerSelfOptimizationSession:
    """CIA 内部负责反复运行实验、评价并更新控制参数的优化会话。"""

    def __init__(self, config: ControllerSelfOptimizationConfig | None = None):
        """保存优化算法和综合评价目标配置。"""
        self.config = config or ControllerSelfOptimizationConfig()

    def run(
        self,
        parameter_optimization_csvs: Sequence[str | Path],
        independent_test_csvs: Sequence[str | Path],
        experiment_config,
        run_trial: Callable[[Any, str | Path], dict[str, Any]],
        progress_callback=None,
    ) -> dict[str, Any]:
        """在优化集选择参数，按需确认最优值，再运行独立测试集。

        地图、拖拉机、轨迹和实验初始条件由调用方固定，每轮只替换搜索空间中的
        控制器参数。独立测试结果在最优参数锁定后产生，不反馈给优化器。
        """
        optimization_csvs = [Path(path) for path in parameter_optimization_csvs]
        test_csvs = [Path(path) for path in independent_test_csvs]
        if not optimization_csvs:
            raise ValueError(
                "At least one parameter-optimization trajectory CSV is required."
            )
        output_dir = ensure_dir(
            Path(experiment_config.output_dir)
            / "controller_optimization"
            / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        initial_controller = _materialize_controller(
            experiment_config.controller,
            experiment_config.dt,
        )
        initial_parameters = _controller_parameters(initial_controller)
        search_space = {
            name: bounds
            for name, bounds in self.config.search_space.items()
            if name in initial_parameters
        }
        if not search_space:
            raise ValueError("Controller optimization search_space contains no supported parameters.")

        search = GaussianProcessBayesianOptimizer(
            initial_parameters=initial_parameters,
            search_space=search_space,
            initial_random_trials=min(
                self.config.initial_random_trials,
                self.config.max_trials,
            ),
            candidate_pool_size=self.config.candidate_pool_size,
            random_seed=self.config.random_seed,
            kernel_length_scale=self.config.kernel_length_scale,
            gp_noise=self.config.gp_noise,
            expected_improvement_xi=self.config.expected_improvement_xi,
        )
        trial_rows: list[dict[str, Any]] = []
        trial_results: list[dict[str, Any]] = []

        # ask 提出参数、CARLA 返回跟踪结果、tell 更新代理模型，构成一次完整迭代。
        for trial_index in range(self.config.max_trials):
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "trial_started",
                        "trial": trial_index + 1,
                        "max_trials": self.config.max_trials,
                        "message": "Starting Bayesian controller trial.",
                    }
                )
            parameters, candidate_meta = search.ask()
            controller = _controller_from_parameters(
                initial_controller,
                parameters,
            )
            trial_config = replace(
                experiment_config,
                controller=controller,
                controller_optimization=None,
                output_dir=str(output_dir / f"trial_{trial_index:03d}"),
                debug_life_time=min(float(experiment_config.debug_life_time), 1.0),
            )
            result, evaluation, _ = self._run_dataset(
                trajectory_csvs=optimization_csvs,
                experiment_config=trial_config,
                controller=controller,
                run_trial=run_trial,
                output_dir=Path(trial_config.output_dir),
            )
            status = evaluation["status"]
            error_message = str(evaluation.get("error", ""))

            score = float(evaluation["objective"])
            search.tell(score)
            row = {
                "trial": trial_index,
                "status": status,
                "objective": score,
                "best_objective_so_far": search.best_score,
                "candidate_type": candidate_meta.get("candidate_type", ""),
                "acquisition": candidate_meta.get("acquisition", ""),
                "acquisition_value": candidate_meta.get(
                    "acquisition_value",
                    float("nan"),
                ),
                "posterior_mean": candidate_meta.get(
                    "posterior_mean",
                    float("nan"),
                ),
                "posterior_std": candidate_meta.get(
                    "posterior_std",
                    float("nan"),
                ),
                "error": error_message,
                **{
                    key: value
                    for key, value in evaluation.items()
                    if key not in {"status", "objective", "error"}
                },
                **{f"parameter.{key}": value for key, value in parameters.items()},
            }
            trial_rows.append(row)
            trial_results.append(result)
            _write_trial_history(output_dir, trial_rows)
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "trial_completed",
                        "trial": trial_index + 1,
                        "max_trials": self.config.max_trials,
                        "objective": score,
                        "best_objective": search.best_score,
                        "status": status,
                        "message": "Bayesian controller trial completed.",
                    }
                )

        successful = [
            row
            for row in trial_rows
            if row["status"] == "completed" and math.isfinite(float(row["objective"]))
        ]
        if not successful:
            summary_path = write_json(
                output_dir / "controller_optimization_summary.json",
                {
                    "status": "failed",
                    "algorithm": self.config.method,
                    "trial_count": len(trial_rows),
                    "message": "No controller candidate produced valid tracking metrics.",
                },
            )
            raise RuntimeError(
                "Controller optimization failed because no trial produced valid metrics. "
                f"See {summary_path}."
            )

        # “最优参数”指所有成功候选中综合目标值最小的一组，而非某个单项误差最小。
        best_row = min(successful, key=lambda row: float(row["objective"]))
        best_trial_index = int(best_row["trial"])
        best_parameters = {
            name: float(best_row[f"parameter.{name}"])
            for name in search_space
        }
        best_controller = _controller_from_parameters(
            initial_controller,
            best_parameters,
        )
        best_config_path = write_json(
            output_dir / "best_controller_config.json",
            _controller_to_dict(best_controller),
        )
        convergence_plot = _plot_convergence(output_dir, trial_rows)

        confirmation_result: dict[str, Any] = {}
        confirmation_evaluation: dict[str, Any] = {}
        if self.config.confirm_best:
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "confirmation_started",
                        "best_trial": best_trial_index + 1,
                        "message": "Confirming the selected controller parameters.",
                    }
                )
            confirmation_config = replace(
                experiment_config,
                controller=best_controller,
                controller_optimization=None,
                output_dir=str(output_dir / "confirmation_best"),
            )
            (
                confirmation_result,
                confirmation_evaluation,
                _,
            ) = self._run_dataset(
                trajectory_csvs=optimization_csvs,
                experiment_config=confirmation_config,
                controller=best_controller,
                run_trial=run_trial,
                output_dir=Path(confirmation_config.output_dir),
            )

        independent_test = {}
        if test_csvs:
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "independent_test_started",
                        "trajectory_count": len(test_csvs),
                        "message": "Starting independent controller evaluation.",
                    }
                )
            independent_test = self._run_independent_test(
                trajectory_csvs=test_csvs,
                experiment_config=experiment_config,
                baseline_controller=initial_controller,
                optimized_controller=best_controller,
                run_trial=run_trial,
                output_dir=output_dir / "independent_test",
            )

        baseline_objective = float(trial_rows[0]["objective"])
        best_objective = float(best_row["objective"])
        improvement_percent = (
            100.0 * (baseline_objective - best_objective) / baseline_objective
            if baseline_objective > 0.0 and math.isfinite(baseline_objective)
            else 0.0
        )
        summary = {
            "status": "completed",
            "algorithm": self.config.method,
            "surrogate_model": "gaussian_process_matern52",
            "acquisition_function": "expected_improvement",
            "optimizer_configuration": asdict(self.config),
            "parameter_optimization_trajectories": [
                str(path) for path in optimization_csvs
            ],
            "independent_test_trajectories": [str(path) for path in test_csvs],
            "trial_count": len(trial_rows),
            "best_trial": best_trial_index,
            "baseline_objective": baseline_objective,
            "best_objective": best_objective,
            "improvement_percent": improvement_percent,
            "initial_parameters": {
                name: initial_parameters[name]
                for name in search_space
            },
            "best_parameters": best_parameters,
            "best_trial_metrics": {
                key: value
                for key, value in best_row.items()
                if key.startswith(("lateral_", "speed_", "heading_", "valid_"))
            },
            "confirmation_evaluation": confirmation_evaluation,
            "best_trial_result": trial_results[best_trial_index],
            "confirmation_result": confirmation_result,
            "independent_test": independent_test,
        }
        summary_path = write_json(
            output_dir / "controller_optimization_summary.json",
            summary,
        )
        return {
            "optimization_enabled": True,
            "algorithm": self.config.method,
            "trial_history": str(output_dir / "controller_optimization_trials.csv"),
            "convergence_plot": str(convergence_plot) if convergence_plot else "",
            "best_controller_config": str(best_config_path),
            "optimization_summary": str(summary_path),
            "best_trial": best_trial_index,
            "baseline_objective": baseline_objective,
            "best_objective": best_objective,
            "improvement_percent": improvement_percent,
            "best_parameters": best_parameters,
            "best_trial_result": trial_results[best_trial_index],
            "confirmation_result": confirmation_result,
            "independent_test": independent_test,
        }

    def _run_dataset(
        self,
        *,
        trajectory_csvs: Sequence[Path],
        experiment_config,
        controller: ControllerInterfaceConfig,
        run_trial: Callable[[Any, str | Path], dict[str, Any]],
        output_dir: Path,
    ) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
        """用同一控制器运行一组轨迹并汇总综合评价。"""
        ensure_dir(output_dir)
        results: dict[str, Any] = {}
        rows: list[dict[str, Any]] = []
        evaluations: list[dict[str, Any]] = []

        for index, trajectory_csv in enumerate(trajectory_csvs):
            trajectory_id = f"{index:03d}_{_safe_name(trajectory_csv.stem)}"
            trajectory_config = replace(
                experiment_config,
                controller=controller,
                controller_optimization=None,
                output_dir=str(output_dir / trajectory_id),
            )
            try:
                result = run_trial(trajectory_config, trajectory_csv)
                evaluation = evaluate_tracking_result(
                    result,
                    self.config.objective,
                )
            except Exception as exc:
                result = {}
                evaluation = _failed_evaluation(
                    self.config.objective.invalid_trial_penalty,
                    str(exc),
                )

            results[trajectory_id] = result
            evaluations.append(evaluation)
            rows.append(
                {
                    "trajectory_id": trajectory_id,
                    "trajectory_csv": str(trajectory_csv),
                    **evaluation,
                }
            )

        pd.DataFrame(rows).to_csv(
            output_dir / "dataset_evaluation.csv",
            index=False,
            encoding="utf-8-sig",
        )
        return results, _aggregate_evaluations(evaluations), rows

    def _run_independent_test(
        self,
        *,
        trajectory_csvs: Sequence[Path],
        experiment_config,
        baseline_controller: ControllerInterfaceConfig,
        optimized_controller: ControllerInterfaceConfig,
        run_trial: Callable[[Any, str | Path], dict[str, Any]],
        output_dir: Path,
    ) -> dict[str, Any]:
        """在未参与调参的轨迹上分别运行默认和优化控制器并生成对比表。"""
        baseline_results, baseline_evaluation, baseline_rows = self._run_dataset(
            trajectory_csvs=trajectory_csvs,
            experiment_config=experiment_config,
            controller=baseline_controller,
            run_trial=run_trial,
            output_dir=output_dir / "baseline",
        )
        optimized_results, optimized_evaluation, optimized_rows = self._run_dataset(
            trajectory_csvs=trajectory_csvs,
            experiment_config=experiment_config,
            controller=optimized_controller,
            run_trial=run_trial,
            output_dir=output_dir / "optimized",
        )

        metrics_path, comparison_path = _write_independent_test_tables(
            output_dir,
            baseline_rows,
            optimized_rows,
        )
        objective_improvement = _improvement_percent(
            float(baseline_evaluation["objective"]),
            float(optimized_evaluation["objective"]),
        )
        summary = {
            "status": (
                "completed"
                if baseline_evaluation["status"] == "completed"
                and optimized_evaluation["status"] == "completed"
                else "failed"
            ),
            "selection_isolation": (
                "Independent-test results were produced only after the best "
                "controller parameters were locked and were not supplied to "
                "the Bayesian optimizer."
            ),
            "trajectory_count": len(trajectory_csvs),
            "baseline_evaluation": baseline_evaluation,
            "optimized_evaluation": optimized_evaluation,
            "objective_improvement_percent": objective_improvement,
            "metrics_table": str(metrics_path),
            "comparison_table": str(comparison_path),
            "baseline_results": baseline_results,
            "optimized_results": optimized_results,
        }
        summary_path = write_json(
            output_dir / "independent_test_summary.json",
            summary,
        )
        return {
            "status": summary["status"],
            "trajectory_count": len(trajectory_csvs),
            "objective_improvement_percent": objective_improvement,
            "metrics_table": str(metrics_path),
            "comparison_table": str(comparison_path),
            "summary": str(summary_path),
        }


def evaluate_tracking_result(
    result: dict[str, Any],
    config: TrackingObjectiveConfig,
) -> dict[str, Any]:
    """从 PES 输出中计算单次轨迹实验的综合目标值。

    各指标先按量纲尺度归一化再加权；有效数据比例不足和最大横向误差超限会
    追加惩罚，因此低目标值表示整体跟踪表现更好且评价数据有效。
    """
    metrics_path = result.get("tracking_metrics_summary")
    tick_path = result.get("tick_records")
    if not metrics_path or not Path(metrics_path).exists():
        return _failed_evaluation(
            config.invalid_trial_penalty,
            "tracking_metrics_summary.csv is missing.",
        )

    metrics = pd.read_csv(metrics_path)
    if metrics.empty:
        return _failed_evaluation(
            config.invalid_trial_penalty,
            "tracking_metrics_summary.csv contains no rows.",
        )
    row = metrics.iloc[0]
    prefix = config.metric_scope if config.metric_scope in {"valid", "all"} else "valid"
    required = {
        "lateral_mae_m": f"{prefix}_lateral_mean_abs_m",
        "lateral_rmse_m": f"{prefix}_lateral_rmse_m",
        "lateral_p95_m": f"{prefix}_lateral_p95_abs_m",
        "lateral_max_m": f"{prefix}_lateral_max_abs_m",
        "heading_rmse_deg": f"{prefix}_heading_rmse_deg",
    }
    values: dict[str, float] = {}
    for output_name, column in required.items():
        if column not in row or not math.isfinite(float(row[column])):
            return _failed_evaluation(
                config.invalid_trial_penalty,
                f"Required metric {column} is missing or invalid.",
            )
        values[output_name] = float(row[column])

    speed_rmse = _speed_rmse_from_ticks(tick_path, prefix)
    if not math.isfinite(speed_rmse):
        return _failed_evaluation(
            config.invalid_trial_penalty,
            "A valid speed RMSE could not be computed.",
        )
    valid_ratio = float(row.get("valid_ratio", 0.0))
    values["speed_rmse_kmh"] = speed_rmse
    values["valid_ratio"] = valid_ratio

    score = (
        config.lateral_mae_weight
        * values["lateral_mae_m"]
        / max(config.lateral_scale_m, 1e-9)
        + config.lateral_rmse_weight
        * values["lateral_rmse_m"]
        / max(config.lateral_scale_m, 1e-9)
        + config.lateral_p95_weight
        * values["lateral_p95_m"]
        / max(config.lateral_scale_m, 1e-9)
        + config.speed_rmse_weight
        * values["speed_rmse_kmh"]
        / max(config.speed_scale_kmh, 1e-9)
        + config.heading_rmse_weight
        * values["heading_rmse_deg"]
        / max(config.heading_scale_deg, 1e-9)
    )
    if valid_ratio < config.minimum_valid_ratio:
        score += (
            config.minimum_valid_ratio - valid_ratio
        ) * config.low_valid_ratio_penalty
    if values["lateral_max_m"] > config.maximum_lateral_error_m:
        score += (
            values["lateral_max_m"] - config.maximum_lateral_error_m
        ) * config.excessive_error_penalty_per_m

    return {"status": "completed", "objective": float(score), **values}


def _speed_rmse_from_ticks(tick_path: str | Path | None, scope: str) -> float:
    """从逐周期记录计算目标速度与实际速度的均方根误差。"""
    if not tick_path or not Path(tick_path).exists():
        return float("nan")
    ticks = pd.read_csv(tick_path)
    if ticks.empty:
        return float("nan")
    if "in_reference_time" in ticks:
        ticks = ticks[ticks["in_reference_time"].astype(bool)]
    if scope == "valid" and "metric_valid" in ticks:
        ticks = ticks[ticks["metric_valid"].astype(bool)]
    if ticks.empty:
        return float("nan")
    error = (
        ticks["actual_abs_kmh"].astype(float).values
        - ticks["target_abs_kmh"].astype(float).values
    )
    return float(math.sqrt(np.mean(error ** 2)))


def _materialize_controller(
    controller: ControllerInterfaceConfig | None,
    dt: float,
) -> ControllerInterfaceConfig:
    """补齐 PID、纯追踪和换挡配置，形成可被优化的完整控制器。"""
    base = controller or ControllerInterfaceConfig(dt=dt)
    return replace(
        base,
        dt=dt,
        pid=base.pid or PIDConfig(),
        pure_pursuit=base.pure_pursuit or PurePursuitConfig(),
        gear=base.gear or GearStateConfig(),
    )


def _controller_parameters(config: ControllerInterfaceConfig) -> dict[str, float]:
    """提取当前实现允许搜索的七个参数。"""
    pure_pursuit = config.pure_pursuit or PurePursuitConfig()
    pid = config.pid or PIDConfig()
    return {
        "pure_pursuit.fwd_lookahead_min": float(pure_pursuit.fwd_lookahead_min),
        "pure_pursuit.fwd_lookahead_gain": float(pure_pursuit.fwd_lookahead_gain),
        "pure_pursuit.rev_lookahead_min": float(pure_pursuit.rev_lookahead_min),
        "pure_pursuit.rev_lookahead_gain": float(pure_pursuit.rev_lookahead_gain),
        "pid.kp": float(pid.kp),
        "pid.ki": float(pid.ki),
        "pid.kd": float(pid.kd),
    }


def _controller_from_parameters(
    base: ControllerInterfaceConfig,
    parameters: dict[str, float],
) -> ControllerInterfaceConfig:
    """在保留其他字段的前提下，将候选值写入控制器配置副本。"""
    base_pure_pursuit = base.pure_pursuit or PurePursuitConfig()
    base_pid = base.pid or PIDConfig()
    pure_pursuit = replace(
        base_pure_pursuit,
        fwd_lookahead_min=parameters.get(
            "pure_pursuit.fwd_lookahead_min",
            base_pure_pursuit.fwd_lookahead_min,
        ),
        fwd_lookahead_gain=parameters.get(
            "pure_pursuit.fwd_lookahead_gain",
            base_pure_pursuit.fwd_lookahead_gain,
        ),
        rev_lookahead_min=parameters.get(
            "pure_pursuit.rev_lookahead_min",
            base_pure_pursuit.rev_lookahead_min,
        ),
        rev_lookahead_gain=parameters.get(
            "pure_pursuit.rev_lookahead_gain",
            base_pure_pursuit.rev_lookahead_gain,
        ),
    )
    pid = replace(
        base_pid,
        kp=parameters.get("pid.kp", base_pid.kp),
        ki=parameters.get("pid.ki", base_pid.ki),
        kd=parameters.get("pid.kd", base_pid.kd),
    )
    return replace(base, pure_pursuit=pure_pursuit, pid=pid)


def _controller_to_dict(config: ControllerInterfaceConfig) -> dict[str, Any]:
    """把嵌套控制器数据类转换为可保存字典。"""
    return asdict(config)


def _parse_search_space(data: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """兼容列表或 min/max 字典形式的参数上下界。"""
    output: dict[str, tuple[float, float]] = {}
    for name, bounds in data.items():
        if isinstance(bounds, dict):
            lower_value = bounds.get("min")
            upper_value = bounds.get("max")
        else:
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError(f"Controller search bounds for {name} must contain two values.")
            lower_value, upper_value = bounds
        if not isinstance(lower_value, (int, float, str)) or not isinstance(
            upper_value, (int, float, str)
        ):
            raise ValueError(f"Controller search bounds for {name} must be numeric: {bounds}")
        lower = float(lower_value)
        upper = float(upper_value)
        if not lower < upper:
            raise ValueError(f"Invalid controller search bounds for {name}: {bounds}")
        output[name] = (lower, upper)
    return output


def _aggregate_evaluations(
    evaluations: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """汇总多条优化轨迹评价：普通指标取均值，最大误差取最大值。"""
    if not evaluations:
        return _failed_evaluation(1000.0, "No trajectory evaluation was produced.")

    metric_names = (
        "objective",
        "lateral_mae_m",
        "lateral_rmse_m",
        "lateral_p95_m",
        "lateral_max_m",
        "speed_rmse_kmh",
        "heading_rmse_deg",
        "valid_ratio",
    )
    aggregate: dict[str, Any] = {
        "status": (
            "completed"
            if all(item.get("status") == "completed" for item in evaluations)
            else "failed"
        ),
        "trajectory_count": len(evaluations),
    }
    errors = [
        str(item.get("error", ""))
        for item in evaluations
        if item.get("error")
    ]
    aggregate["error"] = " | ".join(errors)
    for name in metric_names:
        values = [
            float(item[name])
            for item in evaluations
            if name in item and math.isfinite(float(item[name]))
        ]
        if name == "lateral_max_m":
            aggregate[name] = float(max(values)) if values else float("nan")
        else:
            aggregate[name] = float(np.mean(values)) if values else float("nan")
    return aggregate


def _write_independent_test_tables(
    output_dir: Path,
    baseline_rows: Sequence[dict[str, Any]],
    optimized_rows: Sequence[dict[str, Any]],
) -> tuple[Path, Path]:
    """写出独立测试长表以及默认/优化参数逐轨迹对比表。"""
    ensure_dir(output_dir)
    metrics_path = output_dir / "independent_test_metrics.csv"
    comparison_path = output_dir / "independent_test_comparison.csv"

    long_rows = [
        {"parameter_setting": "baseline", **row} for row in baseline_rows
    ] + [
        {"parameter_setting": "optimized", **row} for row in optimized_rows
    ]
    pd.DataFrame(long_rows).to_csv(
        metrics_path,
        index=False,
        encoding="utf-8-sig",
    )

    optimized_by_csv = {
        str(row["trajectory_csv"]): row for row in optimized_rows
    }
    error_metrics = (
        "objective",
        "lateral_mae_m",
        "lateral_rmse_m",
        "lateral_p95_m",
        "lateral_max_m",
        "speed_rmse_kmh",
        "heading_rmse_deg",
    )
    comparison_rows: list[dict[str, Any]] = []
    for baseline in baseline_rows:
        trajectory_csv = str(baseline["trajectory_csv"])
        optimized = optimized_by_csv.get(trajectory_csv, {})
        row: dict[str, Any] = {
            "trajectory_id": baseline["trajectory_id"],
            "trajectory_csv": trajectory_csv,
            "baseline_status": baseline.get("status", ""),
            "optimized_status": optimized.get("status", "missing"),
        }
        for metric in error_metrics:
            baseline_value = float(baseline.get(metric, float("nan")))
            optimized_value = float(optimized.get(metric, float("nan")))
            row[f"baseline_{metric}"] = baseline_value
            row[f"optimized_{metric}"] = optimized_value
            row[f"{metric}_improvement_percent"] = _improvement_percent(
                baseline_value,
                optimized_value,
            )
        row["baseline_valid_ratio"] = baseline.get("valid_ratio", float("nan"))
        row["optimized_valid_ratio"] = optimized.get("valid_ratio", float("nan"))
        comparison_rows.append(row)

    pd.DataFrame(comparison_rows).to_csv(
        comparison_path,
        index=False,
        encoding="utf-8-sig",
    )
    return metrics_path, comparison_path


def _improvement_percent(baseline: float, optimized: float) -> float:
    """计算误差型指标相对默认参数的下降百分比。"""
    if not math.isfinite(baseline) or not math.isfinite(optimized) or baseline <= 0.0:
        return float("nan")
    return 100.0 * (baseline - optimized) / baseline


def _safe_name(value: str) -> str:
    """把轨迹文件名转换为安全的输出目录名称。"""
    safe = "".join(character if character.isalnum() else "_" for character in value)
    return safe.strip("_") or "trajectory"


def _failed_evaluation(penalty: float, error: str) -> dict[str, Any]:
    """为失败实验返回固定惩罚和缺失指标占位值。"""
    return {
        "status": "failed",
        "objective": float(penalty),
        "error": error,
        "lateral_mae_m": float("nan"),
        "lateral_rmse_m": float("nan"),
        "lateral_p95_m": float("nan"),
        "lateral_max_m": float("nan"),
        "speed_rmse_kmh": float("nan"),
        "heading_rmse_deg": float("nan"),
        "valid_ratio": 0.0,
    }


def _write_trial_history(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    """每轮后重写完整历史表，确保中断时仍保留已完成结果。"""
    path = output_dir / "controller_optimization_trials.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _plot_convergence(
    output_dir: Path,
    rows: list[dict[str, Any]],
) -> Path | None:
    """绘制各轮综合目标和历史最优目标的收敛曲线。"""
    if not rows:
        return None
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None

    trials = [int(row["trial"]) for row in rows]
    objectives = [float(row["objective"]) for row in rows]
    best_so_far = [float(row["best_objective_so_far"]) for row in rows]
    path = output_dir / "controller_optimization_convergence.png"
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(trials, objectives, "o-", color="#7c8da6", label="Trial objective")
    ax.plot(trials, best_so_far, "-", color="#c43c39", linewidth=2.0, label="Best so far")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Composite tracking objective")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _clip(value: float, lower: float, upper: float) -> float:
    """将初始参数限制在用户给定搜索边界内。"""
    return min(max(float(value), float(lower)), float(upper))
