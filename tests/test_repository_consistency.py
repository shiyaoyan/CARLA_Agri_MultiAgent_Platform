"""发布文档、示例配置和实际代码入口的一致性测试。"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
import unittest
from pathlib import Path

import yaml

from agri_carla import __version__
from agri_carla.agents.plan_generation import PlanGenerationAgent
from agri_carla.machinery.templates import MachineryTemplateBuilder
from agri_carla.orchestration.registry import build_default_registry


ROOT = Path(__file__).resolve().parents[1]


class RepositoryConsistencyTests(unittest.TestCase):
    """防止发布说明再次引用已删除代码或不可直接运行的示例。"""

    def test_package_and_project_versions_match(self):
        """包版本与 pyproject 发布版本应保持一致。"""
        project_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r'^version\s*=\s*"([^"]+)"', project_text, re.MULTILINE)
        self.assertIsNotNone(match)
        self.assertEqual(__version__, match.group(1))

    def test_agent_names_match_the_paper_architecture(self):
        """代码、配置与文档只使用论文确定的七个智能体名称。"""
        descriptors = {
            item["agent_id"]: item for item in build_default_registry().describe()
        }
        self.assertEqual(
            {
                "PGA": "Plan Generation Agent",
                "SGA": "Scene Generation Agent",
                "MCA": "Machinery Configuration Agent",
                "CIA": "Controller Interface Agent",
                "EEA": "Experiment Execution Agent",
                "TPS": "Trajectory Preprocessing Sub-Agent",
                "PES": "Performance Evaluation Subagent",
            },
            {key: value["name"] for key, value in descriptors.items()},
        )
        self.assertEqual("eea_subagent", descriptors["TPS"]["scheduling"])
        self.assertEqual("eea_subagent", descriptors["PES"]["scheduling"])

        scanned_files = list((ROOT / "agri_carla").rglob("*.py"))
        scanned_files.extend((ROOT / "scripts").glob("*.py"))
        scanned_files.extend((ROOT / "tests").glob("*.py"))
        scanned_files.extend((ROOT / "docs").glob("*.md"))
        scanned_files.extend((ROOT / "configs").rglob("*.yaml"))
        scanned_files.extend([ROOT / "README.md", ROOT / "README_CN.md"])
        scanned_files = [path for path in scanned_files if path.exists()]
        old_tokens = re.compile(r"\b(?:TPA|ASGA|SEEA|TrPA|REA)\b")
        old_identifiers = re.compile(
            r"TaskPlanningAgent|AgriculturalRoadSceneGenerationAgent|"
            r"SimulationExperimentExecutionAgent|TrajectoryProcessingAgent|"
            r"RecordingEvaluationAgent|task_planning|road_scene_generation|"
            r"simulation_execution"
        )
        violations = []
        for path in scanned_files:
            if path.resolve() == Path(__file__).resolve():
                continue
            text = path.read_text(encoding="utf-8")
            if old_tokens.search(text) or old_identifiers.search(text):
                violations.append(str(path.relative_to(ROOT)))
        self.assertEqual([], violations)

    def test_published_task_schemas_are_identical(self):
        """包内 Schema 与公开配置目录中的副本不得漂移。"""
        package_schema = json.loads(
            (ROOT / "agri_carla/llm/task_request.schema.json").read_text(
                encoding="utf-8"
            )
        )
        public_schema = json.loads(
            (ROOT / "configs/schemas/task_request.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(package_schema, public_schema)

    def test_documented_source_and_example_paths_exist(self):
        """README 与部署文档中的活动脚本、配置和数据路径必须存在。"""
        documents = [ROOT / "README.md"]
        documents.extend((ROOT / "docs").glob("*.md"))
        documents.extend((ROOT / "data").glob("*/README.md"))
        pattern = re.compile(
            r"(?:scripts|configs|data|docs|agri_carla|roadrunner_mcp)/"
            r"[A-Za-z0-9_./-]+"
        )
        missing: list[str] = []
        for document in documents:
            for relative_path in pattern.findall(
                document.read_text(encoding="utf-8")
            ):
                relative_path = relative_path.rstrip(".,;:)`")
                if "<" in relative_path:
                    continue
                if not (ROOT / relative_path).exists():
                    missing.append(f"{document.name}: {relative_path}")
        self.assertEqual([], sorted(set(missing)))

    def test_paper_cases_have_real_implementation_modules(self):
        """两个论文案例及参数优化不能只剩 README 命令。"""
        expected = (
            "scripts/run_crewai.py",
            "agri_carla/crewai_runtime/flow.py",
            "agri_carla/simulation/tracking_experiment.py",
            "agri_carla/controllers/controller_optimization.py",
            "agri_carla/simulation/full_process_experiment.py",
        )
        for relative_path in expected:
            self.assertTrue((ROOT / relative_path).is_file(), relative_path)

    def test_formal_requests_plan_the_expected_agents(self):
        """正式示例应能由 PGA 解析为与论文架构一致的步骤序列。"""
        cases = {
            "configs/yantai_tracking_experiment_request.json": [
                "SGA",
                "MCA",
                "CIA",
                "EEA",
            ],
            "configs/yantai_tracking_optimization_request.json": ["CIA", "EEA"],
            "configs/yantai_full_process_agent_request.json": [
                "MCA",
                "CIA",
                "EEA",
            ],
        }
        planner = PlanGenerationAgent(project_root=ROOT)
        for relative_path, expected_agents in cases.items():
            request = json.loads((ROOT / relative_path).read_text(encoding="utf-8"))
            result = planner.run(request)
            actual_agents = [step.agent for step in result.execution_plan.steps]
            self.assertEqual(expected_agents, actual_agents, relative_path)

    def test_bayesian_example_uses_two_documented_dataset_directories(self):
        """正式优化案例应清楚预留参数优化集和独立验证集。"""
        request = json.loads(
            (ROOT / "configs/yantai_tracking_optimization_request.json").read_text(
                encoding="utf-8"
            )
        )
        experiment = request["experiment"]
        optimization_dir = ROOT / experiment["parameter_optimization_set_dir"]
        validation_dir = ROOT / experiment["independent_validation_set_dir"]
        self.assertTrue(optimization_dir.is_dir())
        self.assertTrue(validation_dir.is_dir())
        self.assertNotIn("trajectory_csv", experiment)

        # The two sets must be the shipped tuning and evaluation sets, and must not
        # share any trajectory file.
        optimization_files = {path.name for path in optimization_dir.glob("*.csv")}
        validation_files = {path.name for path in validation_dir.glob("*.csv")}
        self.assertTrue(optimization_files)
        self.assertTrue(validation_files)
        self.assertEqual(set(), optimization_files & validation_files)

        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("section_3_2_bayesian_optimization", readme)
        self.assertIn(experiment["parameter_optimization_set_dir"], readme)
        self.assertIn(experiment["independent_validation_set_dir"], readme)

    def test_env_example_uses_only_same_host_worker(self):
        """公开模板必须把执行环境固定在本机回环接口。"""
        values = {}
        for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
            if line and not line.lstrip().startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip()
        self.assertEqual(
            "http://127.0.0.1:8765",
            values.get("AGRI_CARLA_WORKER_URL"),
        )
        self.assertEqual("EEA", values.get("AGRI_CARLA_WORKER_AGENTS"))

    def test_env_example_contains_no_real_api_key(self):
        """公开环境模板只能包含占位符，不能泄露真实 API key。"""
        text = (ROOT / ".env.example").read_text(encoding="utf-8")
        self.assertNotIn("sk-proj-", text)
        self.assertIn("OPENAI_API_KEY=replace_with_your_openai_api_key", text)

    def test_repository_ignores_private_and_generated_artifacts(self):
        """版本控制规则应覆盖密钥、环境、缓存、数据库和运行输出。"""
        ignore_text = (ROOT / ".gitignore").read_text(encoding="utf-8")
        for pattern in (
            ".env", ".venv*/", "__pycache__/", "*.pyc", "outputs/",
            "results/", "runtime/", "*.sqlite3", "*.log",
        ):
            self.assertIn(pattern, ignore_text)

        secret_pattern = re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}")
        text_suffixes = {
            ".cfg", ".ini", ".json", ".md", ".py", ".sh", ".toml",
            ".txt", ".yaml", ".yml",
        }
        leaked = []
        for path in ROOT.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in text_suffixes:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if secret_pattern.search(text):
                leaked.append(str(path.relative_to(ROOT)))
        self.assertEqual([], leaked)

    def test_mit_license_is_declared_consistently(self):
        """发布目录、包元数据和说明文件应声明同一个 MIT 许可证。"""
        license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertTrue(license_text.startswith("MIT License\n"))
        self.assertIn("Copyright (c) 2026 shiyaoyan", license_text)

        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('license = {file = "LICENSE"}', pyproject)
        self.assertIn("License :: OSI Approved :: MIT License", pyproject)

        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("MIT", readme)
        self.assertIn("LICENSE", readme)

    def test_section_3_2_results_are_complete_and_use_all_sample_metrics(self):
        """论文 3.2 节五组发布结果必须完整且固定采用 all 指标口径。"""
        results_root = ROOT / "data" / "experiment_results" / "section_3_2_tracking"
        expected = {
            "EXP01_complex_operation": (0.026389139687361982, 0.061317635335325076, 0.7825675515411455),
            "EXP02_regular_operation": (0.033433756722604194, 0.10955721857820439, 0.7459752167182401),
            "EXP03_straight_line": (0.019592023373115247, 0.055736316410794455, 0.27927216187506393),
            "EXP04_bulb_turn": (0.03265211533054119, 0.042470538652310014, 0.0974258106984592),
            "EXP05_fishtail_turn": (0.0833333038529306, 0.1388920560157072, 0.7876795423966949),
        }
        required_files = {
            "tick_records.csv",
            "diagnostic_events.csv",
            "phase_summary.csv",
            "tracking_metrics_summary.csv",
            "01_tracking_error_metrics.png",
            "01_time_sync_diagnostics.png",
            "02_spatial_trajectory.png",
            "README_diagnostics.txt",
        }

        for directory, metric_values in expected.items():
            run_dir = results_root / directory
            self.assertTrue(run_dir.is_dir(), directory)
            self.assertTrue(required_files.issubset({path.name for path in run_dir.iterdir()}))
            with (run_dir / "tracking_metrics_summary.csv").open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                row = next(csv.DictReader(handle))
            actual = (
                float(row["all_lateral_mean_abs_m"]),
                float(row["all_lateral_rmse_m"]),
                float(row["all_lateral_max_abs_m"]),
            )
            for observed, published in zip(actual, metric_values):
                self.assertAlmostEqual(published, observed, places=6)

    def test_evidence_manifest_matches_the_published_results(self):
        """证据清单必须覆盖全部结果与数据，且哈希与文件一致。"""
        manifest = json.loads(
            (ROOT / "data/experiment_results/manifest.json").read_text(encoding="utf-8")
        )
        self.assertTrue((ROOT / manifest["tracking_config"]).is_file())
        self.assertTrue((ROOT / manifest["navigation_run"]).is_dir())

        groups = {entry["group"] for entry in manifest["trajectories"]}
        self.assertEqual({"TS-Tune", "EXP-FTT"}, groups)
        self.assertEqual(10, sum(1 for item in manifest["trajectories"] if item["group"] == "TS-Tune"))
        self.assertEqual(5, sum(1 for item in manifest["trajectories"] if item["group"] == "EXP-FTT"))

        for entry in manifest["trajectories"]:
            self.assertTrue((ROOT / entry["input"]).is_file(), entry["input"])
            if entry["group"] == "EXP-FTT":
                self.assertTrue((ROOT / entry["result"]).is_dir(), entry["result"])

        for relative_path, expected_hash in manifest["sha256"].items():
            path = ROOT / relative_path
            self.assertTrue(path.is_file(), relative_path)
            self.assertEqual(
                expected_hash,
                hashlib.sha256(path.read_bytes()).hexdigest(),
                relative_path,
            )

    def test_single_computer_deployment_entrypoints_exist(self):
        """发布项目只保留同机双环境入口。"""
        expected = (
            "scripts/setup_crewai_manager.sh",
            "scripts/setup_carla_worker_linux.sh",
            "scripts/run_same_host_crewai_carla.sh",
            "docs/installation.md",
        )
        for relative_path in expected:
            self.assertTrue((ROOT / relative_path).is_file(), relative_path)
        removed = (
            "scripts/setup_unified_linux.sh",
            "scripts/run_unified_linux.sh",
            "docs/crewai_manager_carla_worker_CN.md",
            "README_CN.md",
        )
        for relative_path in removed:
            self.assertFalse((ROOT / relative_path).exists(), relative_path)

        launcher = (ROOT / "scripts/run_same_host_crewai_carla.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("--host 127.0.0.1", launcher)
        self.assertNotIn("--host 0.0.0.0", launcher)

    def test_published_machinery_capability_is_tractor_only(self):
        """公开接口不得宣称尚未实现和验证的收割机配置能力。"""
        builder = MachineryTemplateBuilder()
        self.assertEqual("tractor", builder.generate().machinery_type)
        with self.assertRaises(ValueError):
            builder.generate("harvester")

    def test_all_configs_parse_and_python38_sources_parse(self):
        """公开配置可读取，核心与脚本保持 CARLA Python 3.8 语法兼容。"""
        config_paths = list((ROOT / "configs").rglob("*.json"))
        config_paths.extend((ROOT / "configs").rglob("*.yaml"))
        for path in config_paths:
            text = path.read_text(encoding="utf-8")
            parsed = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
            self.assertIsNotNone(parsed, str(path))

        source_paths = list((ROOT / "agri_carla").rglob("*.py"))
        source_paths.extend((ROOT / "roadrunner_mcp").rglob("*.py"))
        source_paths.extend((ROOT / "scripts").glob("*.py"))
        for path in source_paths:
            ast.parse(
                path.read_text(encoding="utf-8"),
                filename=str(path),
                feature_version=(3, 8),
            )


if __name__ == "__main__":
    unittest.main()
