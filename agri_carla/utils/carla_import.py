"""CARLA Python API 的可选导入工具。

项目中的场景生成、轨迹预处理和结果分析可以在没有安装 CARLA 的计算机上运行。
只有真正执行仿真实验时才调用本模块，从而避免包导入阶段因缺少 CARLA egg 而失败。
"""


def import_carla():
    """导入 CARLA API，并将底层导入错误转换为更明确的运行提示。"""
    try:
        import carla  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "CARLA Python API is not available. Install CARLA or add the CARLA "
            "egg path before running simulation experiments."
        ) from exc
    return carla
