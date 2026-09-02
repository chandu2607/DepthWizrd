"""Adapter for validated Indian Phase 89/91 scene artifacts.

This module is deliberately artifact-driven: it loads persisted scene meshes and
component assignments, validates their identity, and emits the geometry schema
consumed by the existing Three.js viewer. It does not run inference.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rasterio
from depthwizard.data.raster_loader import RasterInput
from depthwizard.viz.interactive_viewer import triangulate_polygon_earcut
from rasterio.coords import BoundingBox

ROOT = Path(__file__).resolve().parents[2]
PHASE89 = ROOT / "runs" / "phase89_height_mapping_fix"
SCENE_ROOTS = {
    "uttarakhand": ROOT / "runs" / "phase90_indian_3d_scene" / "UTTARAKHAND_SCENE",
    "himachal": ROOT / "runs" / "phase91_valid_himachal_scene" / "HIMACHAL_SCENE",
}


def _read_obj(path: Path) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    vertices: list[list[float]] = []
    faces: list[tuple[int, int, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif parts[0] == "f":
            indices = [int(part.split("/")[0]) - 1 for part in parts[1:4]]
            if len(indices) == 3:
                faces.append(tuple(indices))
    return np.asarray(vertices, dtype=np.float32), faces


def _rgb_for_scene(region: str, crop: dict[str, Any]) -> np.ndarray:
    source_dir = ROOT / "runs" / "phase68_india_benchmark_ready" / "ORIGINAL_DATA" / region
    row_offset = crop.get("row_off", crop.get("row_offset"))
    column_offset = crop.get("col_off", crop.get("column_offset"))
    window = rasterio.windows.Window(column_offset, row_offset, crop["width"], crop["height"])
    paths = [source_dir / f"{region}_B04.tif", source_dir / f"{region}_B03.tif", source_dir / f"{region}_B02.tif"]
    bands = [rasterio.open(path).read(1, window=window) for path in paths]
    rgb = np.stack(bands, axis=-1).astype(np.float32)
    return np.clip(rgb / 10000.0 * 255.0, 0, 255).astype(np.uint8)


def _texture_base64(rgb: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise ValueError("Could not encode scene RGB texture")
    return base64.b64encode(encoded).decode("ascii")


def _canny_base64(rgb: np.ndarray) -> str:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    rgba = np.zeros((*edges.shape, 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 1] = 190
    rgba[..., 2] = 20
    rgba[..., 3] = np.where(edges > 0, 210, 0)
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    if not ok:
        raise ValueError("Could not encode Canny overlay")
    return base64.b64encode(encoded).decode("ascii")


def _colors(count: int, color: tuple[float, float, float]) -> list[float]:
    return list(np.tile(np.asarray(color, dtype=np.float32), count).tolist())


def _uvs(vertices: np.ndarray, x_max: float, z_max: float) -> list[float]:
    if not len(vertices):
        return []
    u = np.clip(vertices[:, 0] / max(x_max, 1.0), 0, 1)
    v = np.clip(1.0 - vertices[:, 1] / max(z_max, 1.0), 0, 1)
    return np.stack([u, v], axis=1).astype(np.float32).ravel().tolist()


def _load_assignment_records(region: str) -> dict[int, dict[str, Any]]:
    if region == "uttarakhand":
        payload = json.loads((PHASE89 / "HEIGHT_ASSIGNMENTS.json").read_text(encoding="utf-8"))
        records = payload[region]["records"]
    else:
        scene_dir = SCENE_ROOTS[region]
        metadata = json.loads((scene_dir / "scene_metadata.json").read_text(encoding="utf-8"))
        unavailable = json.loads((scene_dir / "height_unavailable_footprints.json").read_text(encoding="utf-8"))
        building_vertices, _ = _read_obj(scene_dir / "building_meshes_finite_height.obj")
        vertex_offset = 0
        finite_records = []
        for item in metadata.get("mesh_records", []):
            vertex_count = int(item["vertices"])
            vertices = building_vertices[vertex_offset:vertex_offset + vertex_count]
            vertex_offset += vertex_count
            half = vertex_count // 2
            height = float(np.median(vertices[half:, 2] - vertices[:half, 2]))
            finite_records.append({
                "component_id": item["component_id"],
                "area_m2": 0.0,
                "predicted_height_m": height,
                "selected_by_model": True,
            })
        records = [
            *finite_records,
        ] + unavailable["components"]
    return {int(record["component_id"]): record for record in records}


def _viewer_vertex(vertex: np.ndarray, z_min: float) -> list[float]:
    # Artifact OBJ uses (east, north, elevation); viewer uses (x, vertical, z).
    return [float(vertex[0]), float(vertex[2] - z_min), float(vertex[1])]


def _build_scene_geometry(region: str, scene_dir: Path, metadata: dict[str, Any], assignment_records: dict[int, dict[str, Any]], rgb: np.ndarray) -> dict[str, Any]:
    terrain_vertices_obj, terrain_faces = _read_obj(scene_dir / "terrain_mesh.obj")
    terrain_z = terrain_vertices_obj[:, 2]
    z_min = float(np.min(terrain_z)) if len(terrain_z) else 0.0
    terrain_positions = [_viewer_vertex(vertex, z_min) for vertex in terrain_vertices_obj]
    terrain_indices = [index for face in terrain_faces for index in face]
    terrain = {
        "positions": np.asarray(terrain_positions, dtype=np.float32).ravel().tolist(),
        "uvs": _uvs(terrain_vertices_obj[:, [0, 1]], max(float(terrain_vertices_obj[:, 0].max()) if len(terrain_vertices_obj) else 1.0, 1.0), max(float(terrain_vertices_obj[:, 1].max()) if len(terrain_vertices_obj) else 1.0, 1.0)),
        "indices": terrain_indices,
        "elev_colors": _colors(len(terrain_vertices_obj), (0.34, 0.58, 0.32)),
        "slope_colors": _colors(len(terrain_vertices_obj), (0.25, 0.55, 0.38)),
        "n_verts": len(terrain_vertices_obj),
        "n_faces": len(terrain_faces),
    }

    building_obj, _ = _read_obj(scene_dir / "building_meshes_finite_height.obj")
    mesh_records = metadata.get("mesh_records", [])
    roofs_positions: list[list[float]] = []
    roofs_uvs: list[list[float]] = []
    roofs_indices: list[int] = []
    walls_positions: list[list[float]] = []
    walls_indices: list[int] = []
    buildings: list[dict[str, Any]] = []
    offset = 0
    x_max = max(float(terrain_vertices_obj[:, 0].max()) if len(terrain_vertices_obj) else 1.0, 1.0)
    y_max = max(float(terrain_vertices_obj[:, 1].max()) if len(terrain_vertices_obj) else 1.0, 1.0)
    for mesh_record in mesh_records:
        component_id = int(mesh_record["component_id"])
        vertex_count = int(mesh_record["vertices"])
        vertices = building_obj[offset:offset + vertex_count]
        offset += vertex_count
        if len(vertices) != vertex_count or vertex_count < 6 or vertex_count % 2:
            raise ValueError(f"Invalid persisted building mesh for component {component_id}")
        n = vertex_count // 2
        base_vertices = vertices[:n]
        top_vertices = vertices[n:]
        roof_start = len(roofs_positions)
        roofs_positions.extend([_viewer_vertex(vertex, z_min) for vertex in top_vertices])
        roofs_uvs.extend([[float(vertex[0] / x_max), float(1.0 - vertex[1] / y_max)] for vertex in top_vertices])
        for first, second, third in triangulate_polygon_earcut(top_vertices[:, [0, 1]]):
            roofs_indices.extend([roof_start + first, roof_start + second, roof_start + third])
        wall_start = len(walls_positions)
        for base, top in zip(base_vertices, top_vertices):
            walls_positions.extend([_viewer_vertex(base, z_min), _viewer_vertex(top, z_min)])
        for index in range(n):
            next_index = (index + 1) % n
            a = wall_start + 2 * index
            b = wall_start + 2 * next_index
            c = wall_start + 2 * index + 1
            d = wall_start + 2 * next_index + 1
            walls_indices.extend([a, c, d, a, d, b])
        assignment = assignment_records.get(component_id, {})
        height = assignment.get("predicted_height_m", mesh_record.get("height_m"))
        if height is None:
            raise ValueError(f"Finite mesh component {component_id} has no persisted height")
        height = float(height)
        base_elevation = float(np.median(base_vertices[:, 2]))
        building = {
            "id": component_id,
            "component_id": component_id,
            "orig_id": str(component_id),
            "area_m2": float(assignment.get("area_m2", assignment.get("area_m2_at_10m_gsd", 0.0))),
            "z_ground": base_elevation - z_min,
            "z_roof": base_elevation + height - z_min,
            "height_m": height,
            "height_available": True,
            "selected_by_model": True,
            "cx": float(np.mean(top_vertices[:, 0])),
            "cy": float(base_elevation + height - z_min),
            "cz": float(np.mean(top_vertices[:, 1])),
        }
        buildings.append(building)
    for component_id, assignment in assignment_records.items():
        if component_id not in {building["component_id"] for building in buildings}:
            buildings.append({
                "id": component_id,
                "component_id": component_id,
                "orig_id": str(component_id),
                "area_m2": float(assignment.get("area_m2", assignment.get("area_m2_at_10m_gsd", 0.0))),
                "z_ground": None,
                "z_roof": None,
                "height_m": None,
                "height_available": False,
                "selected_by_model": bool(assignment.get("selected_by_model", False)),
                "missing_height_reason": assignment.get("missing_height_reason"),
                "cx": None,
                "cy": None,
                "cz": None,
            })
    return {
        "terrain": terrain,
        "roofs": {"positions": np.asarray(roofs_positions, dtype=np.float32).ravel().tolist(), "uvs": np.asarray(roofs_uvs, dtype=np.float32).ravel().tolist(), "indices": roofs_indices, "elev_colors": _colors(len(roofs_positions), (0.70, 0.42, 0.28)), "height_colors": _colors(len(roofs_positions), (0.92, 0.28, 0.20)), "n_verts": len(roofs_positions), "n_faces": len(roofs_indices) // 3},
        "walls": {"positions": np.asarray(walls_positions, dtype=np.float32).ravel().tolist(), "indices": walls_indices, "elev_colors": _colors(len(walls_positions), (0.30, 0.35, 0.40)), "height_colors": _colors(len(walls_positions), (0.80, 0.24, 0.16)), "n_verts": len(walls_positions), "n_faces": len(walls_indices) // 3},
        "buildings": buildings,
        "rejected": [],
        "texture_base64": _texture_base64(rgb),
        "canny_overlay_base64": _canny_base64(rgb),
        "bounds": {"w_m": float(metadata["scene_bounds"]["x_m"][1]), "h_m": float(metadata["scene_bounds"]["y_m"][1]), "z_min": z_min, "z_max": float(metadata["scene_bounds"]["z_m"][1]), "max_dim": float(max(metadata["scene_bounds"]["x_m"][1], metadata["scene_bounds"]["y_m"][1], metadata["scene_bounds"]["z_m"][1] - z_min, 1.0))},
        "metadata": {"region": region, "terrain_source": "REAL_GEOREFERENCED_DEM", "building_source": "SINGLE-VIEW MODEL PREDICTIONS", "height_mapping": "PHASE 89 COMPONENT-ID CORRECTED", "height_accuracy": "UNVALIDATED", "crs": metadata["crs"], "resolution_m": metadata["resolution_m"], "bounds": metadata["scene_bounds"], "height_available_count": int(metadata["finite_height_building_count"]), "height_unavailable_count": int(metadata["height_unavailable_count"])},
    }


def load_phase89_scene(region: str, *, canny_refinement: bool = False, point_cloud_enabled: bool = False) -> dict[str, Any]:
    """Load one validated Indian scene without running or changing inference."""
    if region not in SCENE_ROOTS:
        raise ValueError(f"Unsupported Indian region: {region}")
    scene_dir = SCENE_ROOTS[region]
    metadata = json.loads((scene_dir / "scene_metadata.json").read_text(encoding="utf-8"))
    if region == "uttarakhand":
        phase89 = json.loads((PHASE89 / "RESULTS.json").read_text(encoding="utf-8"))
        crop = phase89["regions"][region]["crop"]
    else:
        search = json.loads((ROOT / "runs" / "phase91_valid_himachal_scene" / "WINDOW_SEARCH.json").read_text(encoding="utf-8"))
        crop = search["selected_window"]
    rgb = _rgb_for_scene(region, crop)
    assignments = _load_assignment_records(region)
    scene = _build_scene_geometry(region, scene_dir, metadata, assignments, rgb)
    dem_path = ROOT / "runs" / "phase72_common_grid_forensics" / "common_grid" / region / "aligned_DEM.tif"
    with rasterio.open(dem_path) as dem_source:
        scene["metadata"]["terrain_raster"] = {
            "path": str(dem_path),
            "crs": str(dem_source.crs),
            "resolution_m": [float(abs(dem_source.transform.a)), float(abs(dem_source.transform.e))],
            "bounds": [float(value) for value in dem_source.bounds],
            "elevation_units": "meters",
            "nodata": None if dem_source.nodata is None or not np.isfinite(dem_source.nodata) else float(dem_source.nodata),
        }
    if scene["metadata"]["terrain_raster"]["crs"] != scene["metadata"]["crs"]:
        raise ValueError("Phase 72 DEM CRS does not match persisted scene CRS")
    component_ids = [building["component_id"] for building in scene["buildings"]]
    if len(component_ids) != len(set(component_ids)):
        raise ValueError("Duplicate component_id in adapter scene")
    finite = [building for building in scene["buildings"] if building["height_available"]]
    if any(building["height_m"] is None for building in finite):
        raise ValueError("Finite building is missing height")
    scene["canny"] = {"enabled": bool(canny_refinement), "purpose": "optional structural-boundary inspection only", "low_threshold": 50, "high_threshold": 150, "aperture_size": 3, "edge_fraction": float(np.mean(cv2.Canny(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 50, 150, apertureSize=3) > 0)) if canny_refinement else None}
    scene["point_cloud"] = {"enabled": bool(point_cloud_enabled), "purpose": "optional XYZ representation only", "positions": None}
    if point_cloud_enabled:
        terrain_vertices, _ = _read_obj(scene_dir / "terrain_mesh.obj")
        points = [_viewer_vertex(vertex, scene["bounds"]["z_min"]) for vertex in terrain_vertices]
        building_vertices, _ = _read_obj(scene_dir / "building_meshes_finite_height.obj")
        points.extend([_viewer_vertex(vertex, scene["bounds"]["z_min"]) for vertex in building_vertices])
        scene["point_cloud"]["positions"] = points
        scene["point_cloud"]["unavailable_component_ids"] = [
            building["component_id"] for building in scene["buildings"] if not building["height_available"]
        ]
    return scene


def load_phase89_raster_input(region: str) -> RasterInput:
    """Load the validated Indian RGB crop as the application's RasterInput."""
    if region not in SCENE_ROOTS:
        raise ValueError(f"Unsupported Indian region: {region}")
    if region == "uttarakhand":
        phase89 = json.loads((PHASE89 / "RESULTS.json").read_text(encoding="utf-8"))
        crop = phase89["regions"][region]["crop"]
    else:
        search = json.loads((ROOT / "runs" / "phase91_valid_himachal_scene" / "WINDOW_SEARCH.json").read_text(encoding="utf-8"))
        crop = search["selected_window"]
    row_offset = crop.get("row_off", crop.get("row_offset"))
    column_offset = crop.get("col_off", crop.get("column_offset"))
    window = rasterio.windows.Window(column_offset, row_offset, crop["width"], crop["height"])
    rgb = _rgb_for_scene(region, crop)
    with rasterio.open(ROOT / "runs" / "phase72_common_grid_forensics" / "common_grid" / region / "aligned_DEM.tif") as source:
        transform = source.window_transform(window)
        bounds = rasterio.windows.bounds(window, source.transform)
        crs = str(source.crs)
        gsd = (abs(float(source.transform.a)), abs(float(source.transform.e)))
    return RasterInput(rgb, f"phase89_{region}.tif", True, crs=crs, transform=transform, bounds=BoundingBox(*bounds), gsd=gsd)
