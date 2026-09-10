"""供 CrewAI Flow 判断后续动作的语义化异常类型。

异常类别不仅用于显示错误，还决定任务应标记为失败、等待用户补充信息、因依赖
缺失而阻塞，还是允许自动重试。
"""


class OrchestrationError(RuntimeError):
    """所有编排层可识别异常的基类。"""


class InputValidationError(OrchestrationError):
    """结构化任务请求未通过字段或文件校验。"""

    def __init__(self, errors):
        # 保留独立错误列表供 PGA 修复或前端逐项展示，同时构造普通异常消息。
        self.errors = list(errors)
        super().__init__("; ".join(str(item) for item in self.errors))


class NeedsUserInputError(InputValidationError):
    """缺少地图、轨迹或参数等用户信息，当前任务不能继续执行。"""


class DependencyUnavailableError(OrchestrationError):
    """CARLA、外部工具或输入文件等运行依赖当前不可用。"""


class TransientExecutionError(OrchestrationError):
    """连接超时等可能通过再次执行恢复的临时错误。"""
