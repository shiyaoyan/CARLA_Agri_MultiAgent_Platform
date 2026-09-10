"""项目配置、轨迹数据和结果文件的统一读写工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在并返回标准化后的 ``Path`` 对象。"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_fallback(path: str | Path):
    """读取可能采用 UTF-8 或 GBK 编码的农业轨迹 CSV。

    实车采集软件导出的 CSV 编码并不统一，因此先尝试兼容 Excel 的
    ``utf-8-sig``，仅在解码失败时回退到常见中文编码 GBK。
    """
    import pandas as pd

    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk")


def write_json(path: str | Path, data: dict[str, Any]) -> Path:
    """以可读格式写入 JSON，并保留中文字符。"""
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def read_yaml(path: str | Path) -> dict[str, Any]:
    """读取 JSON 或 YAML 配置，并统一返回字典。

    JSON 不依赖额外软件包；只有读取 YAML 时才延迟导入 PyYAML，使仅使用
    JSON 配置的 CARLA 计算机不必安装不需要的依赖。
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)

    try:
        import yaml  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyYAML is required to read .yaml files. Install project requirements "
            "or provide the same request as a .json file."
        ) from exc

    data = yaml.safe_load(text)
    return data or {}


def write_yaml(path: str | Path, data: dict[str, Any]) -> Path:
    """根据扩展名写入 JSON 或 YAML 格式的结构化数据。"""
    path = Path(path)
    ensure_dir(path.parent)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    import yaml  # type: ignore[import-untyped]

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return path
