"""根据项目默认物理模型生成可保存、可修改的农机参数模板。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agri_carla.machinery.physics import AgriPhysicsConfig
from agri_carla.utils.io import write_yaml


@dataclass
class MachineryTemplate:
    """农机类型、三维模型来源和物理参数的结构化集合。"""
    machinery_type: str
    model_file: str
    physics: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        """转换为 JSON/YAML 可保存的字典。"""
        return {
            "machinery_type": self.machinery_type,
            "model_file": self.model_file,
            "physics": self.physics,
        }


class MachineryTemplateBuilder:
    """依据默认物理配置生成农机参数模板。"""

    def generate(
        self,
        machinery_type: str = "tractor",
        model_file: str = "",
        output_file: str | Path | None = None,
    ) -> MachineryTemplate:
        """生成本文已验证的拖拉机模板，并按需写入配置文件。"""
        machinery_type = machinery_type.lower()
        if machinery_type != "tractor":
            raise ValueError(f"Unsupported machinery type: {machinery_type}")

        config = AgriPhysicsConfig(machinery_type)
        template = MachineryTemplate(
            machinery_type=machinery_type,
            model_file=model_file,
            physics=config.to_dict(),
        )
        if output_file:
            write_yaml(output_file, template.to_dict())
        return template
