"""在上传 GitHub 前检查发布目录中是否混入私有或生成文件。"""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_ROOTS = (
    ".env",
    ".idea",
    ".venv",
    ".venv-crewai",
    ".venv-carla",
    "outputs",
    "results",
    "runtime",
    "build",
    "dist",
)
FORBIDDEN_NAMES = {
    "__MACOSX",
    "__pycache__",
    ".DS_Store",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".venv-carla",
    ".venv-crewai",
}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo", ".sqlite", ".sqlite3", ".db", ".log"}
TEXT_SUFFIXES = {
    ".cfg", ".csv", ".ini", ".json", ".md", ".py", ".sh", ".toml", ".txt",
    ".yaml", ".yml",
}
SECRET_PATTERN = re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}")


def audit_release() -> list[str]:
    """返回所有阻止发布的问题；空列表表示可以上传。"""
    problems: list[str] = []
    for relative_path in FORBIDDEN_ROOTS:
        if (ROOT / relative_path).exists():
            problems.append("forbidden path: %s" % relative_path)

    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if any(part in FORBIDDEN_NAMES for part in relative.parts):
            problems.append("generated cache: %s" % relative)
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append("generated file: %s" % relative)
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if SECRET_PATTERN.search(text):
            problems.append("possible API key: %s" % relative)
    return sorted(set(problems))


def main() -> None:
    problems = audit_release()
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        raise SystemExit(1)
    print("Release audit passed: no private or generated artifacts found.")


if __name__ == "__main__":
    main()
