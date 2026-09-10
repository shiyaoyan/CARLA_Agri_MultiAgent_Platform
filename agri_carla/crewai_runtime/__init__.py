"""CrewAI 管理端公共接口，采用延迟导入避免影响 CARLA Python 3.8。"""

import os
from pathlib import Path

from dotenv import load_dotenv


# 必须在首次导入 CrewAI 之前准备运行环境。CrewAI 1.15.12 的 Flow 会自动
# 创建官方 Memory；将其写入项目 runtime，避免依赖操作系统用户数据目录。
load_dotenv()
_runtime_root = Path(os.environ.get("AGRI_CARLA_RUNTIME_DIR", "runtime")).resolve()
_crewai_storage = _runtime_root / "crewai_storage"
_crewai_storage.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("CREWAI_STORAGE_DIR", str(_crewai_storage))

# 平台已有本地事件账本，不需要把运行轨迹上传到 CrewAI 遥测服务。
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

__all__ = ["AgriculturalSimulationFlow", "CrewAIRuntime"]


def __getattr__(name):
    if name == "AgriculturalSimulationFlow":
        from .flow import AgriculturalSimulationFlow

        return AgriculturalSimulationFlow
    if name == "CrewAIRuntime":
        from .runtime import CrewAIRuntime

        return CrewAIRuntime
    raise AttributeError(name)
