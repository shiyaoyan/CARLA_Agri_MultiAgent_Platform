"""从源码目录启动 Python 3.8 CARLA 执行端。"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agri_carla.worker.server import main


if __name__ == "__main__":
    main()
