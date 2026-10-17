#!/usr/bin/env python3
"""Run a paper case from its natural-language prompt through CrewAI and PGA."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from agri_carla.crewai_runtime import CrewAIRuntime
from agri_carla.orchestration import ApplicationRunLedger
from agri_carla.paper_cases import DEFAULT_MANIFEST, PaperCaseCatalog
from agri_carla.utils.io import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce a paper case from a natural-language prompt. Planning is "
            "the default; pass --execute only on the configured CARLA computer."
        )
    )
    parser.add_argument("--case", default="", help="Case id from the manifest.")
    parser.add_argument("--list", action="store_true", help="List available cases.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--project-root", default="")
    parser.add_argument("--image", action="append", default=[])
    parser.add_argument(
        "--data",
        action="append",
        default=None,
        help="Override the case's default data attachments.",
    )
    parser.add_argument("--llm-config", default="")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker-agent", action="append", default=[])
    parser.add_argument("--worker-token-env", default="")
    parser.add_argument("--database", default="")
    parser.add_argument("--runs-dir", default="")
    parser.add_argument("--output", default="")
    return parser


def _print_cases(catalog: PaperCaseCatalog) -> None:
    for case in catalog.list_cases():
        route = " -> ".join(agent for agent, _ in case.expected_steps)
        print(
            "%s\t%s\t%s\t%s"
            % (case.case_id, case.paper_section, route, case.prompt_file)
        )


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(args.project_root or Path.cwd()).resolve()
    os.chdir(project_root)
    catalog = PaperCaseCatalog(
        project_root=project_root,
        manifest_path=args.manifest,
    )
    if args.list:
        _print_cases(catalog)
        return
    if not args.case:
        raise SystemExit("Provide --case CASE_ID or use --list.")

    execution: dict[str, object] = {}
    if args.worker_agent:
        execution["worker_agents"] = args.worker_agent
    if args.worker_token_env:
        execution["worker_token_env"] = args.worker_token_env
    payload = catalog.build_payload(
        args.case,
        image_paths=args.image,
        data_files=args.data,
        llm_config_path=args.llm_config or None,
        execution=execution,
    )
    store = ApplicationRunLedger(
        database_path=args.database or None,
        runs_dir=args.runs_dir or None,
    )
    runtime = CrewAIRuntime(store=store, project_root=project_root)
    try:
        result = runtime.run(payload, plan_only=not args.execute)
        result["paper_case_validation"] = catalog.validate_result(
            args.case,
            result,
        )
    except Exception as exc:
        status = "FAILED"
        if runtime.last_task_id:
            try:
                status = str(runtime.status(runtime.last_task_id)["status"])
            except Exception:
                pass
        failure = {
            "paper_case": args.case,
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
