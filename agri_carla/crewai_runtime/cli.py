"""CrewAI Flow 管理端命令行入口。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from agri_carla.orchestration import ApplicationRunLedger
from agri_carla.utils.io import read_yaml, write_json

from .runtime import CrewAIRuntime


def _read_text(value: str, text_file: str) -> str:
    if text_file:
        return Path(text_file).read_text(encoding="utf-8")
    return value


def _apply_runtime_overrides(request: dict, args) -> dict:
    request = dict(request)
    if getattr(args, "confirm_scene", False):
        review = dict(request.get("scene_review", {}))
        review.update(
            {
                "status": "confirmed",
                "requires_confirmation": False,
                "confirmed_by_user": True,
            }
        )
        request["scene_review"] = review
    llm = dict(request.get("llm", {}))
    if args.model:
        llm["model"] = args.model
    if args.base_url:
        llm["base_url"] = args.base_url
    if args.api_key_env:
        llm["api_key_env"] = args.api_key_env
    if llm:
        llm["enabled"] = True
        request["llm"] = llm

    crewai = dict(request.get("crewai", {}))
    if args.crew_model:
        crewai["model"] = args.crew_model
    if args.crew_verbose:
        crewai["verbose"] = True
    if crewai:
        request["crewai"] = crewai

    execution = dict(request.get("execution", {}))
    if args.worker_agent:
        execution["worker_agents"] = args.worker_agent
    if args.worker_token_env:
        execution["worker_token_env"] = args.worker_token_env
    if execution:
        request["execution"] = execution
    return request


def _build_request(args) -> dict:
    if args.request:
        return _apply_runtime_overrides(read_yaml(args.request), args)

    natural_request = _read_text(args.text, args.text_file).strip()
    if not natural_request:
        raise ValueError("Provide --request, --text, or --text-file.")
    llm = read_yaml(args.llm_config) if args.llm_config else {}
    request = {
        "natural_language_request": natural_request,
        "image_paths": args.image,
        "correction_image_paths": args.correction_image,
        "data_files": args.data,
        "llm": llm,
        "validation": {"strict_paths": not args.no_strict_paths},
    }
    if args.request_template:
        request["request_template"] = args.request_template
        request["template_policy"] = args.template_policy
    return _apply_runtime_overrides(request, args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run, inspect, or resume a CrewAI-managed agricultural CARLA task."
    )
    parser.add_argument("--request", default="", help="Existing JSON/YAML request.")
    parser.add_argument("--text", default="", help="Natural-language request.")
    parser.add_argument("--text-file", default="", help="UTF-8 request text file.")
    parser.add_argument("--image", action="append", default=[])
    parser.add_argument(
        "--correction-image",
        action="append",
        default=[],
        help=(
            "Optional second-round marked image used only to correct a rejected "
            "scene candidate. The original satellite image remains under --image."
        ),
    )
    parser.add_argument("--data", action="append", default=[])
    parser.add_argument("--request-template", default="")
    parser.add_argument(
        "--template-policy",
        choices=("locked", "defaults"),
        default="locked",
    )
    parser.add_argument("--llm-config", default="")
    parser.add_argument("--model", default="", help="PGA multimodal model.")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key-env", default="")
    parser.add_argument("--crew-model", default="", help="CrewAI agent model.")
    parser.add_argument("--crew-verbose", action="store_true")
    parser.add_argument(
        "--worker-agent",
        action="append",
        default=[],
        choices=("SGA", "MCA", "CIA", "EEA"),
        help="Agent executed in the same-host CARLA Python 3.8 worker.",
    )
    parser.add_argument("--worker-token-env", default="")
    parser.add_argument("--no-strict-paths", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--confirm-scene",
        action="store_true",
        help=(
            "Confirm a previously reviewed structured scene request and allow SGA "
            "execution. Use with --request pointing to structured_request.json."
        ),
    )
    parser.add_argument("--resume", default="", metavar="TASK_ID")
    parser.add_argument("--status", default="", metavar="TASK_ID")
    parser.add_argument("--database", default="")
    parser.add_argument("--runs-dir", default="")
    parser.add_argument("--project-root", default="")
    parser.add_argument("--output", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.project_root:
        os.chdir(str(Path(args.project_root).resolve()))
    store = ApplicationRunLedger(
        database_path=args.database or None,
        runs_dir=args.runs_dir or None,
    )
    runtime = CrewAIRuntime(
        store=store,
        project_root=args.project_root or Path.cwd(),
    )
    try:
        if args.status:
            result = runtime.status(args.status)
        elif args.resume:
            result = runtime.resume(args.resume)
        else:
            result = runtime.run(_build_request(args), plan_only=args.plan_only)
    except Exception as exc:
        status = "FAILED"
        if runtime.last_task_id:
            try:
                status = runtime.status(runtime.last_task_id)["status"]
            except Exception:
                pass
        failure = {
            "task_id": runtime.last_task_id,
            "status": status,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        print(json.dumps(failure, indent=2, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
    if args.output:
        write_json(args.output, result)
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
