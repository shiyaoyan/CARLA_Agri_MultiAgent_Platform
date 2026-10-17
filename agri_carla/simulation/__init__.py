"""CARLA 仿真实验运行模块。

本包提供田内轨迹跟踪和全流程农业导航两个可直接调用的实验运行器。
"""

__all__ = [
    "TrackingExperimentRunner",
    "FullProcessNavigationExperiment",
]


def __getattr__(name):
    # Geometry is also used by controllers and offline checks; importing it
    # must not eagerly import the runners back into a partially loaded control.
    if name == "TrackingExperimentRunner":
        from .tracking_experiment import TrackingExperimentRunner
        return TrackingExperimentRunner
    if name == "FullProcessNavigationExperiment":
        from .full_process_experiment import FullProcessNavigationExperiment
        return FullProcessNavigationExperiment
    raise AttributeError(name)
