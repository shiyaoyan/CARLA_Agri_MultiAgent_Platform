"""农机配置智能体（MCA）的顶层接口。"""

from __future__ import annotations

from pathlib import Path

from agri_carla.machinery.templates import MachineryTemplate, MachineryTemplateBuilder


class MachineryConfigurationAgent:
    """根据农机类型生成 CARLA 可参考的物理参数模板。"""

    def __init__(self, template_builder: MachineryTemplateBuilder | None = None):
        """注入模板构建器，默认使用项目内置参数。"""
        self.template_builder = template_builder or MachineryTemplateBuilder()

    def run(
        self,
        machinery_type: str,
        model_file: str = "",
        output_file: str | Path | None = None,
        context=None,
    ) -> MachineryTemplate:
        """生成参数模板，并通过编排上下文报告农机类型和模型来源。"""
        if context is not None:
            context.progress(
                "MCA",
                "mca_configuration",
                "Generating machinery parameter template.",
                machinery_type=machinery_type,
                model_file=model_file,
            )
        return self.template_builder.generate(
            machinery_type,
            model_file=model_file,
            output_file=output_file,
        )
