"""从源码目录启动 CrewAI Flow 管理端。"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agri_carla.crewai_runtime.cli import main


if __name__ == "__main__":
    main()
