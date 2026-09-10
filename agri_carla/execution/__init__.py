"""CrewAI 工具和 CARLA 执行端共用的领域执行接口。"""

from .context import BufferedExecutionContext
from .service import DomainExecutionService

__all__ = ["BufferedExecutionContext", "DomainExecutionService"]
