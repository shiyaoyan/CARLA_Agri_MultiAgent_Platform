"""依据用户给定多边形在 CARLA 场景中规划或放置树木和草地区域。"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Tuple

from agri_carla.utils.carla_import import import_carla
from agri_carla.utils.io import ensure_dir, write_json


Point = Tuple[float, float]


@dataclass
class VegetationRegion:
    """单个树木或草地区域的几何、密度和显示参数。"""
    name: str
    kind: str
    polygon: list[Point]
    coordinate_mode: str = "pixel"
    blueprint_filter: str = ""
    density_per_100m2: float = 0.0
    grid_spacing_m: float = 4.0
    min_spacing_m: float = 4.0
    z: float = 0.05
    yaw_deg: float | None = None
    scale: float = 1.0
    visual_mode: str = "blueprint"


@dataclass
class VegetationPlacementConfig:
    """植被坐标转换、CARLA 连接和输出目录配置。"""
    carla_map: str = ""
    host: str = "localhost"
    port: int = 2000
    timeout: float = 30.0
    image_width_px: int = 0
    image_height_px: int = 0
    meters_per_pixel: float = 1.0
    north_up: bool = True
    geo_anchor: dict[str, float] = field(default_factory=dict)
    world_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    seed: int = 7
    dry_run: bool = False
    output_dir: str = "outputs/scene_enrichment"
    tree_regions: list[dict[str, Any]] = field(default_factory=list)
    grass_regions: list[dict[str, Any]] = field(default_factory=list)
    road_scene_request: dict[str, Any] = field(default_factory=dict)
    road_xodr_path: str = ""
    roadside_trees: dict[str, Any] = field(default_factory=dict)
    road_clearance_m: float = 1.0
    load_map: bool = False
    strict_spawn: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VegetationPlacementConfig":
        """从结构化请求创建配置，并忽略仅用于视觉溯源的附加字段。"""
        valid_keys = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in valid_keys})


class VegetationPlacer:
    """规划路旁树木、导出农田网格，并放置已导入 CARLA 的资源。"""

    def run(self, config: VegetationPlacementConfig | dict[str, Any]) -> dict[str, Any]:
        """先生成可审查放置计划，再按 ``dry_run`` 决定是否连接 CARLA。"""
        if isinstance(config, dict):
            config = VegetationPlacementConfig.from_dict(config)

        output_dir = ensure_dir(config.output_dir)
        plan = self._build_plan(config)
        mesh_bundle = self._export_farmland(output_dir, plan)
        plan_path = write_json(output_dir / "vegetation_placement_plan.json", plan)
        preview_path = self._write_preview_svg(output_dir / "vegetation_placement_preview.svg", plan)

        if config.dry_run:
            return {
                "placement_plan": str(plan_path),
                "preview_svg": str(preview_path),
                "spawned_tree_count": 0,
                "spawned_grass_count": 0,
                "dry_run": True,
                "status": "planned",
                "planned_tree_count": len(plan["trees"]),
                "farmland_import_bundle": mesh_bundle,
                "warnings": plan["warnings"],
            }

        spawn_result = self._spawn_in_carla(config, plan)
        result = {
            "placement_plan": str(plan_path),
            "preview_svg": str(preview_path),
            **spawn_result,
            "dry_run": False,
            "farmland_import_bundle": mesh_bundle,
        }
        report_path = write_json(output_dir / "vegetation_spawn_report.json", result)
        if config.strict_spawn and result["status"] != "completed":
            raise RuntimeError(f"Scene enrichment is incomplete; inspect {report_path}.")
        return result

    def _build_plan(self, config: VegetationPlacementConfig) -> dict[str, Any]:
        """将所有多边形转换为确定的树木采样点和草地填充点。"""
        # 固定随机种子保证相同请求得到相同树木位置和朝向，便于重复实验。
        rng = random.Random(config.seed)
        tree_items: list[dict[str, Any]] = []
        grass_items: list[dict[str, Any]] = []
        region_summaries: list[dict[str, Any]] = []
        warnings: list[str] = []
        road_area = self._road_area(config)
        if not math.isfinite(config.road_clearance_m) or config.road_clearance_m < 0:
            raise ValueError("road_clearance_m must be finite and non-negative.")
        excluded = road_area.buffer(config.road_clearance_m)
        tree_regions = list(config.tree_regions)
        roadside = config.roadside_trees
        if roadside.get("enabled", False):
            if road_area.is_empty:
                raise ValueError("Automatic roadside trees require finalized road geometry.")
            width = float(roadside.get("band_width_m", 8.0))
            if not math.isfinite(width) or width <= 0:
                raise ValueError("roadside_trees.band_width_m must be positive.")
            belt = road_area.buffer(config.road_clearance_m + width).difference(excluded)
            parts = list(belt.geoms) if hasattr(belt, "geoms") else [belt]
            for index, polygon in enumerate(parts):
                tree_regions.append({
                    **roadside, "name": f"roadside_{index}",
                    "generated_roadside": True,
                    "polygon": [(x - config.world_offset[0], y - config.world_offset[1])
                                for x, y in polygon.exterior.coords],
                    "coordinate_mode": "world",
                })

        from shapely.geometry import Point as GeometryPoint
        from roadrunner_mcp.farmland import validated_polygon

        for raw in tree_regions:
            region = _region_from_dict(raw, kind="tree")
            world_polygon = self._polygon_to_world(config, region)
            polygon = validated_polygon(world_polygon)
            available = polygon.difference(excluded)
            if raw.get("generated_roadside"):
                available = available.intersection(belt)
            occupied = [(item["x"], item["y"], item["min_spacing_m"]) for item in tree_items]
            points = _sample_polygon(
                world_polygon,
                spacing=max(0.5, region.grid_spacing_m),
                density_per_100m2=region.density_per_100m2,
                min_spacing_m=max(0.0, region.min_spacing_m),
                rng=rng,
                allowed=lambda p: available.covers(GeometryPoint(p)),
                available_area=available.area,
                occupied=occupied,
            )
            for x, y in points:
                tree_items.append(
                    {
                        "region": region.name,
                        "kind": "tree",
                        "x": x,
                        "y": y,
                        "z": region.z + config.world_offset[2],
                        "yaw_deg": region.yaw_deg if region.yaw_deg is not None else rng.uniform(0.0, 360.0),
                        "blueprint_filter": region.blueprint_filter or "static.prop.tree*",
                        "collision_expected": True,
                        "min_spacing_m": region.min_spacing_m,
                        "scale": region.scale,
                    }
                )
            summary = _region_summary(region, world_polygon, len(points))
            summary["available_area_m2"] = available.area
            if region.density_per_100m2 > 0:
                requested = int(round(available.area * region.density_per_100m2 / 100.0))
                summary["requested_count"] = requested
                if len(points) < requested:
                    warnings.append(f"{region.name}: placed {len(points)}/{requested}; spacing/road clearance constrains density.")
            region_summaries.append(summary)

        for raw in config.grass_regions:
            region = _region_from_dict(raw, kind="grass")
            world_polygon = self._polygon_to_world(config, region)
            validated_polygon(world_polygon)
            fill_points = _grid_points_in_polygon(world_polygon, max(1.0, region.grid_spacing_m))
            if not fill_points:
                point = validated_polygon(world_polygon).representative_point()
                fill_points = [(point.x, point.y)]
            grass_items.append(
                {
                    "region": region.name,
                    "kind": "grass",
                    "polygon": [[x, y] for x, y in world_polygon],
                    "fill_points": [[x, y] for x, y in fill_points],
                    "z": region.z + config.world_offset[2],
                    "blueprint_filter": region.blueprint_filter or "static.prop.grass*",
                    "visual_mode": region.visual_mode,
                    "grid_spacing_m": region.grid_spacing_m,
                    "collision_expected": False if region.visual_mode == "debug_area" else None,
                    "scale": region.scale,
                }
            )
            region_summaries.append(_region_summary(region, world_polygon, 1))
            if region.visual_mode == "debug_area":
                warnings.append(f"{region.name}: debug_area is a diagnostic overlay, not a farmland surface.")

        return {
            "metadata": {
                "carla_map": config.carla_map,
                "meters_per_pixel": config.meters_per_pixel,
                "north_up": config.north_up,
                "world_offset": list(config.world_offset),
                "seed": config.seed,
            },
            "regions": region_summaries,
            "trees": tree_items,
            "grass_regions": grass_items,
            "warnings": warnings,
        }

    def _road_area(self, config):
        from shapely.geometry import LineString, Polygon
        from shapely.ops import unary_union

        if config.road_xodr_path:
            from xml.etree import ElementTree as ET
            from shapely.affinity import translate
            from roadrunner_mcp.road_mesh import road_surface_polygons

            root = ET.parse(config.road_xodr_path).getroot()
            surfaces = {road.get("id"): "dirt" for road in root.findall("road")}
            area = unary_union(list(road_surface_polygons(root, surfaces).values()))
            return translate(area, xoff=config.world_offset[0], yoff=config.world_offset[1])
        roads = config.road_scene_request.get("roads", [])
        parts = []
        for road in roads:
            region = VegetationRegion("road", "road", road["pixel_polyline"])
            points = self._polygon_to_world(config, region)
            width = float(road.get("width_m", 3.0))
            if len(points) < 2 or not math.isfinite(width) or width <= 0:
                raise ValueError("Roads need at least two points and positive width_m.")
            parts.append(LineString(points).buffer(width / 2.0))
        return unary_union(parts) if parts else Polygon()

    def _export_farmland(self, output_dir, plan):
        """Export real textured surfaces; CARLA must ingest these before spawning."""
        from roadrunner_mcp.farmland import farmland_mesh
        from roadrunner_mcp.fbx import MeshLayer, _build_ascii_fbx_layers, _surface_style, _write_tga_texture

        mesh_regions = [item for item in plan["grass_regions"] if item["visual_mode"] == "mesh"]
        if not mesh_regions:
            return ""
        bundle_hash = hashlib.sha256(json.dumps(mesh_regions, sort_keys=True).encode()).hexdigest()[:12]
        bundle = ensure_dir(output_dir / f"farmland_import_{bundle_hash}")
        props = []
        for item in mesh_regions:
            digest = hashlib.sha256(json.dumps([item["polygon"], item["z"]], sort_keys=True).encode()).hexdigest()[:12]
            name = f"farmland_{digest}"
            folder = ensure_dir(bundle / "Props" / name)
            texture_name = f"{name}_diffuse.tga"
            style = _surface_style("field")
            _write_tga_texture(folder / texture_name, style)
            mesh = farmland_mesh([item["polygon"]], z=item["z"])
            (folder / f"{name}.fbx").write_text(_build_ascii_fbx_layers(name, [
                MeshLayer("Terrain_Ground_Layer0", mesh, style, texture_name)
            ]), encoding="utf-8")
            props.append({"name": name, "source": f"Props/{name}/{name}.fbx", "size": "huge", "tag": "Ground"})
            item["blueprint_filter"] = f"static.prop.{name}"
            # A generated ground asset is not a debug overlay; collision depends on its import settings.
            item["collision_expected"] = None
        package_hash = hashlib.sha256(json.dumps(props, sort_keys=True).encode()).hexdigest()[:12]
        write_json(bundle / f"Farmland_{package_hash}.json", {"maps": [], "props": props})
        return str(bundle)

    def _polygon_to_world(self, config: VegetationPlacementConfig, region: VegetationRegion) -> list[Point]:
        """将像素多边形或世界坐标多边形统一转换为 CARLA XY 坐标。"""
        if len(config.world_offset) != 3 or not all(math.isfinite(float(v)) for v in config.world_offset):
            raise ValueError("world_offset must contain three finite numbers.")
        if region.coordinate_mode == "world":
            return [(float(x) + config.world_offset[0], float(y) + config.world_offset[1]) for x, y in region.polygon]
        if region.coordinate_mode != "pixel":
            raise ValueError(f"Unsupported coordinate_mode: {region.coordinate_mode}")
        if not math.isfinite(config.meters_per_pixel) or config.meters_per_pixel <= 0:
            raise ValueError("meters_per_pixel must be finite and positive.")
        if config.image_width_px <= 0 or config.image_height_px <= 0:
            raise ValueError("Pixel regions require positive image dimensions.")

        anchor_x = float(config.geo_anchor.get("pixel_x", config.image_width_px / 2.0))
        anchor_y = float(config.geo_anchor.get("pixel_y", config.image_height_px / 2.0))
        y_sign = -1.0 if config.north_up else 1.0
        points = []
        for px, py in region.polygon:
            x = (float(px) - anchor_x) * config.meters_per_pixel + config.world_offset[0]
            y = (float(py) - anchor_y) * config.meters_per_pixel * y_sign + config.world_offset[1]
            points.append((x, y))
        return points

    def _spawn_in_carla(self, config: VegetationPlacementConfig, plan: dict[str, Any]) -> dict[str, Any]:
        """连接 CARLA 并按照计划创建树木或绘制草地区域。"""
        # 网格和静态蓝图的碰撞由导入设置决定；只有调试绘图没有碰撞实体。
        carla = import_carla()
        client = carla.Client(config.host, config.port)
        client.set_timeout(config.timeout)
        world = client.get_world()
        if config.carla_map:
            actual_map = world.get_map().name
            matches = (actual_map == config.carla_map if "/" in config.carla_map
                       else actual_map.rsplit("/", 1)[-1] == config.carla_map)
            if not matches:
                if not config.load_map:
                    raise RuntimeError(f"Current map {actual_map!r} differs from {config.carla_map!r}; load the imported map first or set load_map=true.")
                world = client.load_world(config.carla_map)
        blueprint_library = world.get_blueprint_library()

        missing = sorted({item["blueprint_filter"] for item in plan["trees"] + plan["grass_regions"]
                          if item.get("visual_mode", "blueprint") != "debug_area"
                          and _first_blueprint(blueprint_library, item["blueprint_filter"]) is None})
        if missing:
            return {"status": "failed", "spawned_tree_count": 0, "spawned_grass_count": 0,
                    "missing_blueprints": missing,
                    "error": "Required assets are unavailable. Import the generated farmland package and register tree props; no actors were spawned."}
        for item in plan["trees"] + plan["grass_regions"]:
            if item.get("visual_mode", "blueprint") != "debug_area":
                _configure_scale(_first_blueprint(blueprint_library, item["blueprint_filter"]), item["scale"])

        spawned_trees = []
        skipped_trees = []
        for item in plan["trees"]:
            blueprint = _first_blueprint(blueprint_library, item["blueprint_filter"])
            if blueprint is None:
                skipped_trees.append({**item, "reason": "blueprint_not_found"})
                continue
            transform = carla.Transform(
                carla.Location(x=float(item["x"]), y=float(item["y"]), z=float(item["z"])),
                carla.Rotation(yaw=float(item["yaw_deg"])),
            )
            _configure_scale(blueprint, item["scale"])
            actor = world.try_spawn_actor(blueprint, transform)
            if actor is None:
                skipped_trees.append({**item, "reason": "spawn_failed"})
                continue
            spawned_trees.append({"id": actor.id, **item})

        grass_visual_count = 0
        spawned_grass = []
        skipped_grass = []
        for item in plan["grass_regions"]:
            if item["visual_mode"] == "mesh":
                blueprint = _first_blueprint(blueprint_library, item["blueprint_filter"])
                actor = world.try_spawn_actor(blueprint, carla.Transform())
                if actor is None:
                    skipped_grass.append({"region": item["region"], "reason": "spawn_failed"})
                else:
                    spawned_grass.append({"id": actor.id, "region": item["region"], "kind": "farmland_mesh"})
                continue
            if item.get("visual_mode", "debug_area") == "blueprint":
                blueprint = _first_blueprint(blueprint_library, item["blueprint_filter"])
                if blueprint is None:
                    skipped_grass.append({**item, "reason": "blueprint_not_found"})
                    continue
                _configure_scale(blueprint, item["scale"])
                for x, y in _polygon_vertices(item.get("fill_points", [])):
                    transform = carla.Transform(carla.Location(x=x, y=y, z=float(item["z"])))
                    actor = world.try_spawn_actor(blueprint, transform)
                    if actor:
                        spawned_grass.append({"id": actor.id, "region": item["region"], "x": x, "y": y})
                    else:
                        skipped_grass.append({"region": item["region"], "x": x, "y": y, "reason": "spawn_failed"})
                continue

            self._draw_grass_region(world, carla, item)
            grass_visual_count += 1

        if world.get_settings().synchronous_mode:
            world.tick()
        else:
            world.wait_for_tick(config.timeout)
        return {
            "status": "partial" if skipped_trees or skipped_grass else "completed",
            "warnings": plan["warnings"],
            "spawned_tree_count": len(spawned_trees),
            "skipped_tree_count": len(skipped_trees),
            "grass_region_count": len(plan["grass_regions"]),
            "debug_grass_region_count": grass_visual_count,
            "spawned_grass_count": len(spawned_grass),
            "spawned_trees": spawned_trees,
            "skipped_trees": skipped_trees,
            "spawned_grass": spawned_grass,
            "skipped_grass": skipped_grass,
        }

    def _draw_grass_region(self, world, carla, item: dict[str, Any]) -> None:
        """用 CARLA 调试线框和填充点显示无碰撞草地区域。"""
        points = [carla.Location(x=x, y=y, z=float(item["z"])) for x, y in _polygon_vertices(item["polygon"])]
        color = carla.Color(42, 160, 75, 160)
        for start, end in zip(points, points[1:] + points[:1]):
            world.debug.draw_line(start, end, thickness=0.15, color=color, life_time=0.0)
        for point in points:
            world.debug.draw_point(point, size=0.35, color=color, life_time=0.0)
        for x, y in _polygon_vertices(item.get("fill_points", [])):
            world.debug.draw_point(carla.Location(x=x, y=y, z=float(item["z"])), size=0.18, color=color, life_time=0.0)

    def _write_preview_svg(self, path: Path, plan: dict[str, Any]) -> Path:
        """在运行 CARLA 前输出俯视 SVG，供用户核对区域与采样密度。"""
        tree_points = [(item["x"], item["y"]) for item in plan["trees"]]
        grass_polygons = [
            [(float(x), float(y)) for x, y in item["polygon"]]
            for item in plan["grass_regions"]
        ]
        all_points = tree_points + [point for polygon in grass_polygons for point in polygon]
        if not all_points:
            path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"960\" height=\"640\"></svg>\n", encoding="utf-8")
            return path

        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        west, east = min(xs) - 10.0, max(xs) + 10.0
        south, north = min(ys) - 10.0, max(ys) + 10.0
        width, height, margin = 960, 640, 36
        scale = min((width - 2 * margin) / max(1.0, east - west), (height - 2 * margin) / max(1.0, north - south))

        def sx(x: float) -> float:
            return margin + (x - west) * scale

        def sy(y: float) -> float:
            return height - margin - (y - south) * scale

        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#f7faf5"/>',
        ]
        for polygon in grass_polygons:
            points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in polygon)
            lines.append(f'<polygon points="{points}" fill="#86c884" opacity="0.35" stroke="#2f855a" stroke-width="2"/>')
        for item in plan["grass_regions"]:
            for x, y in _polygon_vertices(item.get("fill_points", [])):
                lines.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="1.3" fill="#38a169" opacity="0.45"/>')
        for x, y in tree_points:
            lines.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="4" fill="#276749"/>')
        lines.append('<text x="28" y="32" font-family="Arial" font-size="20" font-weight="700" fill="#1f2937">Vegetation placement preview</text>')
        lines.append("</svg>")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


def _region_from_dict(data: dict[str, Any], kind: str) -> VegetationRegion:
    """补充区域名称和类型默认值，并构造强类型区域。"""
    region = VegetationRegion(
        name=data.get("name", kind),
        kind=data.get("kind", kind),
        polygon=[(float(x), float(y)) for x, y in data.get("polygon", [])],
        coordinate_mode=data.get("coordinate_mode", "pixel"),
        blueprint_filter=data.get("blueprint_filter", ""),
        density_per_100m2=float(data.get("density_per_100m2", 0.0)),
        grid_spacing_m=float(data.get("grid_spacing_m", 4.0)),
        min_spacing_m=float(data.get("min_spacing_m", 4.0)),
        z=float(data.get("z", 0.05)),
        yaw_deg=data.get("yaw_deg"),
        scale=float(data.get("scale", 1.0)),
        visual_mode=data.get("visual_mode", "mesh" if kind == "grass" else "blueprint"),
    )
    for key in ("grid_spacing_m", "min_spacing_m", "density_per_100m2", "z", "scale"):
        if not math.isfinite(getattr(region, key)):
            raise ValueError(f"{region.name}.{key} must be finite.")
    if region.grid_spacing_m <= 0 or region.min_spacing_m < 0 or region.density_per_100m2 < 0 or region.scale <= 0:
        raise ValueError(f"Invalid spacing, density or scale in {region.name}.")
    if region.visual_mode not in {"mesh", "blueprint", "debug_area"}:
        raise ValueError(f"Unsupported visual_mode: {region.visual_mode}")
    if region.visual_mode == "mesh" and region.scale != 1:
        raise ValueError("Farmland mesh scale must remain 1 to preserve its boundary.")
    if region.yaw_deg is not None and not math.isfinite(float(region.yaw_deg)):
        raise ValueError("yaw_deg must be finite.")
    return region


def _region_summary(region: VegetationRegion, world_polygon: list[Point], placement_count: int) -> dict[str, Any]:
    """汇总区域面积、边界和计划放置数量。"""
    return {
        "name": region.name,
        "kind": region.kind,
        "coordinate_mode": region.coordinate_mode,
        "area_m2": _polygon_area(world_polygon),
        "placement_count": placement_count,
        "world_polygon": [[x, y] for x, y in world_polygon],
    }


def _sample_polygon(
    polygon: list[Point],
    spacing: float,
    density_per_100m2: float,
    min_spacing_m: float,
    rng: random.Random,
    allowed=None,
    available_area=None,
    occupied=(),
) -> list[Point]:
    """在多边形内按网格、密度和最小间距选择树木位置。"""
    if len(polygon) < 3:
        return []
    west, east, south, north = _bounds(polygon)
    area = _polygon_area(polygon) if available_area is None else available_area
    target_count = int(round(area * density_per_100m2 / 100.0)) if density_per_100m2 > 0 else None
    if target_count == 0:
        return []
    if density_per_100m2 > 0:
        spacing = min(spacing, math.sqrt(100.0 / density_per_100m2))
    _check_grid_size(west, east, south, north, spacing)
    candidates: list[Point] = []
    y = south
    while y <= north:
        x = west
        while x <= east:
            jitter_x = rng.uniform(-0.35 * spacing, 0.35 * spacing)
            jitter_y = rng.uniform(-0.35 * spacing, 0.35 * spacing)
            point = (x + jitter_x, y + jitter_y)
            if _point_in_polygon(point, polygon) and (allowed is None or allowed(point)):
                candidates.append(point)
            x += spacing
        y += spacing

    rng.shuffle(candidates)
    selected: list[Point] = []
    min_dist2 = min_spacing_m * min_spacing_m
    for point in candidates:
        if any((point[0] - x) ** 2 + (point[1] - y) ** 2 < max(min_spacing_m, spacing_) ** 2
               for x, y, spacing_ in occupied):
            continue
        if all((point[0] - old[0]) ** 2 + (point[1] - old[1]) ** 2 >= min_dist2 for old in selected):
            selected.append(point)
            if target_count is not None and len(selected) >= target_count:
                break
    return selected


def _grid_points_in_polygon(polygon: list[Point], spacing: float) -> list[Point]:
    """生成覆盖多边形内部而非仅边界的规则网格点。"""
    if len(polygon) < 3:
        return []
    west, east, south, north = _bounds(polygon)
    _check_grid_size(west, east, south, north, spacing)
    points: list[Point] = []
    y = south
    while y <= north:
        x = west
        while x <= east:
            point = (x, y)
            if _point_in_polygon(point, polygon):
                points.append(point)
            x += spacing
        y += spacing
    return points


def _point_in_polygon(point: Point, polygon: list[Point]) -> bool:
    """使用射线法判断点是否位于多边形内部。"""
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, current in enumerate(polygon):
        xi, yi = current
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def _check_grid_size(west, east, south, north, spacing):
    count = (math.ceil((east - west) / spacing) + 1) * (math.ceil((north - south) / spacing) + 1)
    if count > 250000:
        raise ValueError("Vegetation sampling exceeds 250000 candidates; reduce density or increase spacing.")


def _polygon_area(polygon: list[Point]) -> float:
    """使用鞋带公式计算二维多边形面积。"""
    if len(polygon) < 3:
        return 0.0
    total = 0.0
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]):
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def _bounds(points: list[Point]) -> tuple[float, float, float, float]:
    """返回点集的轴对齐包围盒。"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), max(xs), min(ys), max(ys)


def _first_blueprint(blueprint_library, blueprint_filter: str):
    """按过滤表达式返回第一个可用 CARLA 静态物体蓝图。"""
    matches = sorted(blueprint_library.filter(blueprint_filter), key=lambda bp: bp.id)
    return matches[0] if matches else None


def _configure_scale(blueprint, scale):
    if blueprint.has_attribute("scale"):
        blueprint.set_attribute("scale", str(scale))
    elif scale != 1:
        raise ValueError(f"{blueprint.id} does not support the requested scale {scale}.")


def restore_experiment_vegetation(experiment_config):
    """A map reload destroys runtime props; restore them before vehicle spawning."""
    request = dict(experiment_config.vegetation_request)
    if request.get("dry_run", False):
        return None
    request.update(host=experiment_config.host, port=experiment_config.port,
                   timeout=experiment_config.timeout,
                   carla_map=experiment_config.carla_map or request.get("carla_map", ""),
                   load_map=False, dry_run=False, strict_spawn=True,
                   output_dir=str(Path(experiment_config.output_dir) / "scene_enrichment"))
    return VegetationPlacer().run(request)


def _polygon_vertices(polygon: list[list[float]]) -> list[Point]:
    """把 JSON 列表坐标转换为浮点元组。"""
    return [(float(x), float(y)) for x, y in polygon]


def dump_available_static_blueprints(host: str, port: int, timeout: float, output_path: str | Path) -> Path:
    """导出当前 CARLA 地图可用静态蓝图，帮助用户选择树木和草地资源。"""
    carla = import_carla()
    client = carla.Client(host, port)
    client.set_timeout(timeout)
    world = client.get_world()
    blueprints = sorted(bp.id for bp in world.get_blueprint_library().filter("static.*"))
    return write_json(output_path, {"static_blueprints": blueprints})
