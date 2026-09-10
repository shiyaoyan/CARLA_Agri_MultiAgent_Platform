"""道路可视化网格、ASCII FBX 和程序化表面纹理生成。"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

@dataclass
class Mesh:
    """由三维顶点和三角面索引组成的最小道路网格。"""
    vertices: list[tuple[float, float, float]]
    faces: list[tuple[int, int, int]]


@dataclass(frozen=True)
class SurfaceStyle:
    """某类道路材质的颜色、反光和纹理重复参数。"""
    material_name: str
    diffuse: tuple[float, float, float]
    specular: tuple[float, float, float]
    base_rgb: tuple[int, int, int]
    accent_rgb: tuple[int, int, int]
    repeat_m: float


@dataclass(frozen=True)
class MeshLayer:
    """FBX 中具有独立网格、材质和纹理的一个可视化层。"""

    name: str
    mesh: Mesh
    style: SurfaceStyle
    texture_filename: str


def _build_ascii_fbx_layers(map_name: str, layers: list[MeshLayer]) -> str:
    """将多个独立材质网格写入同一个 FBX 7.4 ASCII 场景。"""
    if not layers:
        raise ValueError("At least one FBX mesh layer is required.")
    created = datetime.now(timezone.utc)
    object_blocks: list[str] = []
    connection_blocks: list[str] = []
    for index, layer in enumerate(layers):
        base_id = 100000 + index * 10
        geometry_id = base_id
        model_id = base_id + 1
        material_id = base_id + 2
        texture_id = base_id + 3
        video_id = base_id + 4
        mesh = layer.mesh
        style = layer.style
        texture_filename = layer.texture_filename
        vertex_values = ",".join(
            _fmt(coord) for vertex in mesh.vertices for coord in vertex
        )
        polygon_values = ",".join(
            str(vertex_index if offset < 2 else -vertex_index - 1)
            for face in mesh.faces
            for offset, vertex_index in enumerate(face)
        )
        uv_values, uv_indices = _build_uv_data(mesh, style.repeat_m)
        normal_values = ",".join("0,0,1" for _ in range(len(mesh.faces) * 3))
        layer_name = _escape_name(layer.name)
        diffuse = ",".join(_fmt(value) for value in style.diffuse)
        specular = ",".join(_fmt(value) for value in style.specular)
        object_blocks.append(
            f'''  Geometry: {geometry_id}, "Geometry::{_escape_name(map_name)}_{layer_name}", "Mesh" {{
    Vertices: *{len(mesh.vertices) * 3} {{
      a: {vertex_values}
    }}
    PolygonVertexIndex: *{len(mesh.faces) * 3} {{
      a: {polygon_values}
    }}
    LayerElementNormal: 0 {{
      Version: 101
      Name: ""
      MappingInformationType: "ByPolygonVertex"
      ReferenceInformationType: "Direct"
      Normals: *{len(mesh.faces) * 9} {{
        a: {normal_values}
      }}
    }}
    LayerElementUV: 0 {{
      Version: 101
      Name: "UVMap"
      MappingInformationType: "ByPolygonVertex"
      ReferenceInformationType: "IndexToDirect"
      UV: *{len(uv_values)} {{
        a: {",".join(_fmt(value) for value in uv_values)}
      }}
      UVIndex: *{len(uv_indices)} {{
        a: {",".join(str(value) for value in uv_indices)}
      }}
    }}
    LayerElementMaterial: 0 {{
      Version: 101
      Name: ""
      MappingInformationType: "AllSame"
      ReferenceInformationType: "IndexToDirect"
      Materials: *1 {{
        a: 0
      }}
    }}
    Layer: 0 {{
      Version: 100
      LayerElement:  {{
        Type: "LayerElementNormal"
        TypedIndex: 0
      }}
      LayerElement:  {{
        Type: "LayerElementUV"
        TypedIndex: 0
      }}
      LayerElement:  {{
        Type: "LayerElementMaterial"
        TypedIndex: 0
      }}
    }}
  }}
  Model: {model_id}, "Model::{_escape_name(map_name)}_{layer_name}", "Mesh" {{
    Version: 232
    Properties70:  {{
      P: "Lcl Translation", "Lcl Translation", "", "A",0,0,0
      P: "Lcl Rotation", "Lcl Rotation", "", "A",0,0,0
      P: "Lcl Scaling", "Lcl Scaling", "", "A",1,1,1
    }}
    Shading: T
    Culling: "CullingOff"
  }}
  Material: {material_id}, "Material::{style.material_name}", "" {{
    Version: 102
    ShadingModel: "phong"
    MultiLayer: 0
    Properties70:  {{
      P: "DiffuseColor", "Color", "", "A",{diffuse}
      P: "SpecularColor", "Color", "", "A",{specular}
      P: "EmissiveColor", "Color", "", "A",0,0,0
      P: "AmbientColor", "Color", "", "A",0.02,0.02,0.02
    }}
  }}
  Video: {video_id}, "Video::{texture_filename}", "Clip" {{
    Type: "Clip"
    Properties70:  {{
      P: "Path", "KString", "XRefUrl", "", "{texture_filename}"
    }}
    FileName: "{texture_filename}"
    RelativeFilename: "{texture_filename}"
  }}
  Texture: {texture_id}, "Texture::{texture_filename}", "" {{
    Type: "TextureVideoClip"
    Version: 202
    TextureName: "Texture::{texture_filename}"
    Media: "Video::{texture_filename}"
    FileName: "{texture_filename}"
    RelativeFilename: "{texture_filename}"
    ModelUVTranslation: 0,0
    ModelUVScaling: 1,1
    Texture_Alpha_Source: "None"
    Cropping: 0,0,0,0
  }}'''
        )
        connection_blocks.extend(
            [
                f'  C: "OO",{model_id},0',
                f'  C: "OO",{geometry_id},{model_id}',
                f'  C: "OO",{material_id},{model_id}',
                f'  C: "OP",{texture_id},{material_id},"DiffuseColor"',
                f'  C: "OO",{video_id},{texture_id}',
            ]
        )

    layer_count = len(layers)
    return f"""; FBX 7.4.0 project file
FBXHeaderExtension:  {{
  FBXHeaderVersion: 1003
  FBXVersion: 7400
  CreationTimeStamp:  {{
    Version: 1000
    Year: {created.year}
    Month: {created.month}
    Day: {created.day}
    Hour: {created.hour}
    Minute: {created.minute}
    Second: {created.second}
    Millisecond: 0
  }}
  Creator: "carla-agri-sga"
}}
GlobalSettings:  {{
  Version: 1000
  Properties70:  {{
    P: "UpAxis", "int", "Integer", "",2
    P: "UpAxisSign", "int", "Integer", "",1
    P: "FrontAxis", "int", "Integer", "",1
    P: "FrontAxisSign", "int", "Integer", "",-1
    P: "CoordAxis", "int", "Integer", "",0
    P: "CoordAxisSign", "int", "Integer", "",1
    P: "OriginalUpAxis", "int", "Integer", "",-1
    P: "OriginalUpAxisSign", "int", "Integer", "",1
    P: "UnitScaleFactor", "double", "Number", "",100
    P: "OriginalUnitScaleFactor", "double", "Number", "",1
  }}
}}
Documents:  {{
  Count: 1
  Document: 123456789, "Doc", "Scene" {{
    Properties70:  {{
      P: "SourceObject", "object", "", ""
      P: "ActiveAnimStackName", "KString", "", "", ""
    }}
    RootNode: 0
  }}
}}
References:  {{
}}
Definitions:  {{
  Version: 100
  Count: 5
  ObjectType: "Geometry" {{
    Count: {layer_count}
  }}
  ObjectType: "Model" {{
    Count: {layer_count}
  }}
  ObjectType: "Material" {{
    Count: {layer_count}
  }}
  ObjectType: "Texture" {{
    Count: {layer_count}
  }}
  ObjectType: "Video" {{
    Count: {layer_count}
  }}
}}
Objects:  {{
{chr(10).join(object_blocks)}
}}
Connections:  {{
{chr(10).join(connection_blocks)}
}}
"""


def _build_uv_data(mesh: Mesh, repeat_m: float) -> tuple[list[float], list[int]]:
    """按世界坐标和纹理重复尺度为每个三角面顶点生成 UV。"""
    uv_values: list[float] = []
    uv_indices: list[int] = []
    for face in mesh.faces:
        for vertex_index in face:
            x, y, _ = mesh.vertices[vertex_index]
            uv_values.extend([x / repeat_m, y / repeat_m])
            uv_indices.append(len(uv_indices))
    return uv_values, uv_indices


def _write_tga_texture(path: Path, style: SurfaceStyle) -> None:
    """写出固定随机种子的 128×128、24 位未压缩 TGA 纹理。"""
    width = 128
    height = 128
    header = bytearray(18)
    header[2] = 2
    header[12] = width & 0xFF
    header[13] = (width >> 8) & 0xFF
    header[14] = height & 0xFF
    header[15] = (height >> 8) & 0xFF
    header[16] = 24
    header[17] = 0x20
    rng = random.Random(42)
    pixels = bytearray()
    for y in range(height):
        for x in range(width):
            r, g, b = _texture_pixel(x, y, style, rng)
            pixels.extend([b, g, r])
    path.write_bytes(bytes(header) + bytes(pixels))


def _texture_pixel(x: int, y: int, style: SurfaceStyle, rng: random.Random) -> tuple[int, int, int]:
    """混合底色、纹理色、颗粒和条带，生成一个道路纹理像素。"""
    base = style.base_rgb
    accent = style.accent_rgb
    grain = rng.randint(-16, 16)
    speckle = ((x * 37 + y * 17 + x * y * 3) % 31) == 0
    band = int(((x * 0.7 + y * 0.23) % 29) < 2)
    mix = 0.22 if speckle else 0.08 if band else 0.0
    return (
        _clamp_rgb(int(base[0] * (1 - mix) + accent[0] * mix + grain)),
        _clamp_rgb(int(base[1] * (1 - mix) + accent[1] * mix + grain)),
        _clamp_rgb(int(base[2] * (1 - mix) + accent[2] * mix + grain)),
    )


def _surface_style(surface: str) -> SurfaceStyle:
    """将 dirt、gravel、concrete 或 asphalt 映射为材质预设。"""
    if surface == "field":
        return SurfaceStyle("M_Agricultural_Field", (0.25, 0.31, 0.18), (0.01, 0.01, 0.01), (74, 88, 50), (112, 102, 63), 14.0)
    if surface == "dirt":
        return SurfaceStyle("M_Dirt_Farm_Road", (0.42, 0.28, 0.16), (0.04, 0.03, 0.02), (111, 78, 45), (164, 130, 85), 9.0)
    if surface == "gravel":
        return SurfaceStyle("M_Gravel_Farm_Road", (0.42, 0.38, 0.31), (0.06, 0.055, 0.05), (108, 100, 84), (171, 165, 143), 7.0)
    if surface == "concrete":
        return SurfaceStyle("M_Concrete_Road", (0.48, 0.48, 0.45), (0.08, 0.08, 0.075), (132, 132, 126), (190, 188, 178), 8.0)
    return SurfaceStyle("M_Asphalt_Road", (0.075, 0.075, 0.075), (0.02, 0.02, 0.02), (42, 42, 42), (92, 92, 88), 6.0)


def _fmt(value: float) -> str:
    """将浮点数压缩为适合 FBX 文本的稳定表示。"""
    if abs(value) < 1e-9:
        value = 0.0
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _escape_name(value: str) -> str:
    """替换不适合 FBX 对象名称的字符。"""
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in value)


def _clamp_rgb(value: int) -> int:
    """把颜色通道限制在 8 位 RGB 范围。"""
    return max(0, min(255, value))
