"""Prompt-first catalogue and validation for the paper reproduction cases."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from agri_carla.utils.io import read_yaml


DEFAULT_MANIFEST = "configs/paper_cases.json"


@dataclass(frozen=True)
class PaperCase:
    """One reproducible natural-language input and its expected agent route."""

    case_id: str
    title: str
    paper_section: str
    prompt_file: str
    data_files: tuple[str, ...]
    minimum_images: int
    expected_task_type: str
    expected_steps: tuple[tuple[str, str], ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PaperCase":
        expected_steps = tuple(
            (str(item[0]), str(item[1]))
            for item in data.get("expected_steps", [])
        )
        return cls(
            case_id=str(data["id"]),
            title=str(data["title"]),
            paper_section=str(data["paper_section"]),
            prompt_file=str(data["prompt_file"]),
            data_files=tuple(str(item) for item in data.get("data_files", [])),
            minimum_images=max(0, int(data.get("minimum_images", 0))),
            expected_task_type=str(data["expected_task_type"]),
            expected_steps=expected_steps,
        )


class PaperCaseCatalog:
    """Load prompt cases and build requests that cannot bypass the LLM path."""

    def __init__(
        self,
        *,
        project_root: str | Path = ".",
        manifest_path: str | Path = DEFAULT_MANIFEST,
    ):
        self.project_root = Path(project_root).resolve()
        manifest = self._resolve(manifest_path)
        data = json.loads(manifest.read_text(encoding="utf-8"))
        self.schema_version = str(data.get("schema_version", ""))
        self.default_llm_config = str(data.get("default_llm_config", ""))
        cases = [PaperCase.from_dict(item) for item in data.get("cases", [])]
        self._cases = {case.case_id: case for case in cases}
        if not self._cases:
            raise ValueError("The paper-case manifest contains no cases.")
        if len(self._cases) != len(cases):
            raise ValueError("Paper-case identifiers must be unique.")
        self._validate_manifest()

    def list_cases(self) -> list[PaperCase]:
        return list(self._cases.values())

    def get(self, case_id: str) -> PaperCase:
        try:
            return self._cases[case_id]
        except KeyError as exc:
            available = ", ".join(self._cases)
            raise KeyError(
                "Unknown paper case %r. Available cases: %s"
                % (case_id, available)
            ) from exc

    def prompt_text(self, case_id: str) -> str:
        case = self.get(case_id)
        return self._resolve(case.prompt_file).read_text(encoding="utf-8").strip()

    def build_payload(
        self,
        case_id: str,
        *,
        image_paths: Iterable[str] = (),
        data_files: Iterable[str] | None = None,
        llm_config_path: str | Path | None = None,
        execution: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a natural-language payload with no prebuilt structured request."""
        case = self.get(case_id)
        images = [str(item) for item in image_paths]
        if len(images) < case.minimum_images:
            raise ValueError(
                "Paper case %s requires at least %d image attachment(s)."
                % (case.case_id, case.minimum_images)
            )
        selected_data = (
            list(case.data_files)
            if data_files is None
            else [str(item) for item in data_files]
        )
        config_path = llm_config_path or self.default_llm_config
        llm = read_yaml(self._resolve(config_path)) if config_path else {}
        if not isinstance(llm, dict):
            raise ValueError("The LLM configuration must be a JSON/YAML object.")
        llm = dict(llm)
        llm["enabled"] = True
        payload: dict[str, Any] = {
            "natural_language_request": self.prompt_text(case_id),
            "image_paths": images,
            "data_files": selected_data,
            "llm": llm,
            "validation": {"strict_paths": True},
            "paper_case": {
                "id": case.case_id,
                "paper_section": case.paper_section,
                "prompt_file": case.prompt_file,
                "manifest_schema_version": self.schema_version,
            },
        }
        if execution:
            payload["execution"] = dict(execution)
        self.assert_prompt_first_payload(payload)
        return payload

    @staticmethod
    def assert_prompt_first_payload(payload: dict[str, Any]) -> None:
        """Prevent a paper case from silently falling back to final JSON input."""
        forbidden = {
            "task_type",
            "structured_request",
            "structured_request_file",
            "request_template",
        }
        present = sorted(forbidden & set(payload))
        if present:
            raise ValueError(
                "Prompt-first paper cases cannot contain prebuilt request fields: %s."
                % ", ".join(present)
            )
        if not str(payload.get("natural_language_request", "")).strip():
            raise ValueError("A paper case requires a non-empty natural-language prompt.")

    def validate_result(
        self,
        case_id: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Verify that PGA used the LLM and produced the expected allow-listed plan."""
        case = self.get(case_id)
        source = str(result.get("planning_source", result.get("source", "")))
        if source != "llm":
            raise RuntimeError(
                "Paper case %s did not use the LLM planning path; source=%r."
                % (case.case_id, source)
            )
        task_type = str(
            result.get("task_type")
            or (result.get("structured_request") or {}).get("task_type", "")
        )
        if task_type != case.expected_task_type:
            raise RuntimeError(
                "Paper case %s produced task_type %r; expected %r."
                % (case.case_id, task_type, case.expected_task_type)
            )
        if isinstance(result.get("structured_request"), dict):
            self.validate_case_request(case_id, result["structured_request"])
        plan = result.get("execution_plan") or {}
        actual_steps = tuple(
            (str(item.get("agent", "")), str(item.get("operation", "")))
            for item in plan.get("steps", [])
        )
        if actual_steps != case.expected_steps:
            raise RuntimeError(
                "Paper case %s produced agent route %r; expected %r."
                % (case.case_id, actual_steps, case.expected_steps)
            )
        report = result.get("validation_report") or {}
        if report.get("valid") is not True:
            raise RuntimeError(
                "Paper case %s did not produce a valid structured request."
                % case.case_id
            )
        return {
            "case_id": case.case_id,
            "paper_section": case.paper_section,
            "prompt_file": case.prompt_file,
            "planning_source": source,
            "task_type": task_type,
            "agent_route": [
                {"agent": agent, "operation": operation}
                for agent, operation in actual_steps
            ],
            "valid": True,
        }

    @staticmethod
    def validate_case_request(case_id, request):
        """Check this experiment before downstream execution."""
        if case_id != "section_3_3_full_process":
            return
        from agri_carla.simulation.full_process_config import FullProcessExperimentConfig
        config = request.get("experiment", {}).get("config", {})
        if request.get("task_type") != "full_process_navigation":
            raise ValueError("Section 3.3 requires full_process_navigation")
        if config.get("carla_map") != "YantaiSatelliteRebuilt":
            raise ValueError("Section 3.3 requires the confirmed YantaiSatelliteRebuilt map")
        if config.get("harvest_passes") != 2:
            raise ValueError("Section 3.3 requires two harvest passes")
        if config.get("apply_agri_physics", True) is not True:
            raise ValueError("Section 3.3 requires the configured tractor physics")
        if config.get("reload_world") or config.get("vegetation_request"):
            raise ValueError("Section 3.3 must preserve the already loaded world")
        if request.get("road_scene_request") or request.get("vegetation_request"):
            raise ValueError("Section 3.3 must use the existing map without regenerating scenery")
        FullProcessExperimentConfig(**config)

    def _validate_manifest(self) -> None:
        if not self.default_llm_config:
            raise ValueError("The paper-case manifest requires default_llm_config.")
        if not self._resolve(self.default_llm_config).is_file():
            raise FileNotFoundError(self.default_llm_config)
        for case in self._cases.values():
            prompt_path = self._resolve(case.prompt_file)
            if not prompt_path.is_file():
                raise FileNotFoundError(case.prompt_file)
            if not prompt_path.read_text(encoding="utf-8").strip():
                raise ValueError("Paper-case prompt is empty: %s" % case.prompt_file)
            if not case.expected_steps:
                raise ValueError(
                    "Paper case %s has no expected agent route." % case.case_id
                )

    def _resolve(self, path: str | Path) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return (self.project_root / candidate).resolve()
