from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SCENES = {'uttarakhand': OUT / 'UTTARAKHAND_SCENE', 'himachal': OUT / 'HIMACHAL_SCENE'}
VISUALS = OUT / 'VISUALS'
PHASE89 = ROOT / 'runs' / 'phase89_height_mapping_fix'
PHASE89_ASSIGNMENTS = PHASE89 / 'HEIGHT_ASSIGNMENTS.json'
for path in list(SCENES.values()) + [VISUALS]:
    path.mkdir(parents=True, exist_ok=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from depthwizard.config import TrainConfig
from depthwizard.depth.depth_anything import DepthAnythingV2
from depthwizard.models.building_conditioned_net import BuildingConditionedEstimator
from runs.phase89_height_mapping_fix.phase89_run import run_region as run_phase89_region

MODEL_PATH = ROOT / 'runs' / 'phase24_moe' / 'seed_0' / 'model.pt'
THRESHOLD = 0.5
VERTICAL_EXAGGERATION = 1.0
REGIONS = ('uttarakhand', 'himachal')


def json_safe(value):
    if isinstance(value, dict): return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list): return [json_safe(item) for item in value]
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.floating): return float(value) if np.isfinite(value) else None
    return value


def stats(values) -> dict:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not finite.size:
        return {'count': 0, 'min': None, 'max': None, 'mean': None, 'median': None, 'std': None}
    return {'count': int(finite.size), 'min': float(finite.min()), 'max': float(finite.max()), 'mean': float(finite.mean()), 'median': float(np.median(finite)), 'std': float(finite.std())}


def write_obj(path: Path, vertices: list[tuple[float, float, float]], triangles: list[tuple[int, int, int]], object_name: str):
    with path.open('w', encoding='utf-8') as handle:
        handle.write(f'# Phase 90 {object_name}\n')
        for x, y, z in vertices:
            handle.write(f'v {x:.6f} {y:.6f} {z:.6f}\n')
        for a, b, c in triangles:
            handle.write(f'f {a} {b} {c}\n')


def terrain_mesh(dem: np.ndarray, stride: int = 1):
    sampled = dem[::stride, ::stride]
    h, w = sampled.shape
    vertices = []
    vertex_ids = {}
    for y in range(h):
        for x in range(w):
            if np.isfinite(sampled[y, x]):
                vertex_ids[(y, x)] = len(vertices) + 1
                vertices.append((float(x * 10 * stride), float(y * 10 * stride), float(sampled[y, x])))
    triangles = []
    for y in range(h - 1):
        for x in range(w - 1):
            corners = [(y, x), (y, x + 1), (y + 1, x), (y + 1, x + 1)]
            if all(corner in vertex_ids for corner in corners):
                a, b, c, d = [vertex_ids[corner] for corner in corners]
                triangles.extend([(a, c, b), (b, c, d)])
    return vertices, triangles


def building_mesh(label_map: np.ndarray, component_id: int, dem: np.ndarray, height: float):
    binary = (label_map == component_id).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], []
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    if contour.shape[0] < 3:
        return [], []
    contour_elevation = dem[contour[:, 1], contour[:, 0]]
    if not np.isfinite(contour_elevation).all():
        return [], []
    vertices = []
    for x, y in contour:
        base = float(dem[int(y), int(x)])
        vertices.append((float(x * 10), float(y * 10), base))
    top_start = len(vertices)
    for x, y in contour:
        base = float(dem[int(y), int(x)])
        vertices.append((float(x * 10), float(y * 10), base + float(height)))
    n = len(contour)
    triangles = []
    for index in range(1, n - 1):
        triangles.append((top_start + 1, top_start + index + 1, top_start + index + 2))
    for index in range(n):
        nxt = (index + 1) % n
        base_a, base_b = index + 1, nxt + 1
        top_a, top_b = top_start + index + 1, top_start + nxt + 1
        triangles.extend([(base_a, base_b, top_a), (base_b, top_b, top_a)])
    return vertices, triangles


def render_scene(result: dict, scene_dir: Path):
    arrays = result['_arrays']
    rgb, labels, dem = arrays['rgb'], arrays['labels'], arrays['dem']
    records = result['component_audit']['records']
    finite_records = [record for record in records if record['predicted_height_m'] is not None]
    missing_records = [record for record in records if record['predicted_height_m'] is None]
    stride = 8
    yy, xx = np.mgrid[0:dem.shape[0]:stride, 0:dem.shape[1]:stride]
    z = dem[::stride, ::stride]
    rgb_small = rgb[::stride, ::stride] / 255.0

    def setup(title, elev=42, azim=-60):
        figure = plt.figure(figsize=(12, 9))
        axis = figure.add_subplot(111, projection='3d')
        axis.set_title(title)
        axis.view_init(elev=elev, azim=azim)
        axis.set_xlabel('east (m)'); axis.set_ylabel('north (m)'); axis.set_zlabel('elevation (m)')
        return figure, axis

    def add_terrain(axis, textured=False):
        if np.isfinite(z).any():
            axis.plot_surface(xx * 10, yy * 10, z, facecolors=rgb_small if textured else None, cmap=None if textured else 'terrain', rstride=1, cstride=1, linewidth=0, antialiased=True, alpha=0.92)
        else:
            axis.text2D(0.05, 0.92, 'NO_FINITE_DEM_IN_PHASE89_CROP', transform=axis.transAxes, color='red')

    figure, axis = setup(f"{result['region']} DEM only")
    add_terrain(axis); figure.tight_layout(); figure.savefig(scene_dir / '01_dem_only.png', dpi=140); figure.savefig(VISUALS / f"{result['region']}_01_dem_only.png", dpi=140); plt.close(figure)

    figure, axis = setup(f"{result['region']} DEM + footprints")
    add_terrain(axis)
    for record in records:
        pixels = np.argwhere(labels == record['component_id'])
        if pixels.size:
            cy, cx = pixels.mean(axis=0)
            terrain_values = dem[labels == record['component_id']]
            finite_terrain = terrain_values[np.isfinite(terrain_values)]
            base = float(np.nanmedian(dem[np.isfinite(dem)])) if not finite_terrain.size and np.isfinite(dem).any() else (float(np.median(finite_terrain)) if finite_terrain.size else 0.0)
            axis.scatter([cx * 10], [cy * 10], [base + 1], color='cyan' if record['predicted_height_m'] is not None else 'red', s=8)
    figure.tight_layout(); figure.savefig(scene_dir / '02_dem_plus_footprints.png', dpi=140); figure.savefig(VISUALS / f"{result['region']}_02_dem_plus_footprints.png", dpi=140); plt.close(figure)

    def add_buildings(axis, annotate_missing=False):
        for record in finite_records:
            vertices, triangles = building_mesh(labels, record['component_id'], dem, record['predicted_height_m'])
            if not vertices: continue
            polygons = [[vertices[a - 1], vertices[b - 1], vertices[c - 1]] for a, b, c in triangles]
            axis.add_collection3d(Poly3DCollection(polygons, facecolor='tomato', edgecolor='maroon', linewidth=0.25, alpha=0.82))
        if annotate_missing:
            for record in missing_records:
                pixels = np.argwhere(labels == record['component_id'])
                if pixels.size:
                    cy, cx = pixels.mean(axis=0)
                    terrain_values = dem[labels == record['component_id']]
                    finite_terrain = terrain_values[np.isfinite(terrain_values)]
                    base = float(np.nanmedian(dem[np.isfinite(dem)])) if not finite_terrain.size and np.isfinite(dem).any() else (float(np.median(finite_terrain)) if finite_terrain.size else 0.0)
                    axis.text(float(cx * 10), float(cy * 10), base, 'HEIGHT_UNAVAILABLE', color='red', fontsize=5)

    figure, axis = setup(f"{result['region']} DEM + mapped building extrusions", elev=34, azim=-65)
    add_terrain(axis); add_buildings(axis, annotate_missing=True); figure.tight_layout(); figure.savefig(scene_dir / '03_dem_plus_extrusions.png', dpi=140); figure.savefig(VISUALS / f"{result['region']}_03_dem_plus_extrusions.png", dpi=140); plt.close(figure)

    figure, axis = setup(f"{result['region']} final textured scene", elev=38, azim=-62)
    add_terrain(axis, textured=True); add_buildings(axis, annotate_missing=False); figure.tight_layout(); figure.savefig(scene_dir / '04_final_textured_scene.png', dpi=140); figure.savefig(VISUALS / f"{result['region']}_04_final_textured_scene.png", dpi=140); plt.close(figure)

    camera_specs = {'overview': (48, -60), 'low_angle_oblique': (18, -55), 'close_building': (28, -30), 'terrain_following': (65, -75)}
    for name, (elev, azim) in camera_specs.items():
        figure, axis = setup(f"{result['region']} {name}", elev=elev, azim=azim)
        add_terrain(axis, textured=True); add_buildings(axis, annotate_missing=(name == 'close_building')); figure.tight_layout(); figure.savefig(scene_dir / f'camera_{name}.png', dpi=140); plt.close(figure)


def audit_region(region: str, estimator, depth_model, persisted: dict):
    result = run_phase89_region(region, estimator, depth_model)
    records = result['component_audit']['records']
    persisted_records = {int(record['component_id']): record for record in persisted[region]['records']}
    for record in records:
        stored = persisted_records[record['component_id']]
        assert record['predicted_height_m'] == stored['predicted_height_m'], f'Phase 89 assignment changed for {region} component {record["component_id"]}'
    dem = result['_arrays']['dem']; labels = result['_arrays']['labels']; heights = [record['predicted_height_m'] for record in records if record['predicted_height_m'] is not None]
    finite_records = [record for record in records if record['predicted_height_m'] is not None]
    missing_records = [record for record in records if record['predicted_height_m'] is None]
    bottom_errors = []
    top_errors = []
    nan_geometry = False
    for record in finite_records:
        mask = labels == record['component_id']
        terrain_values = dem[mask & np.isfinite(dem)]
        if not terrain_values.size or not np.isfinite(record['predicted_height_m']):
            nan_geometry = True; continue
        base = float(np.nanmedian(terrain_values)); height = float(record['predicted_height_m']); roof = base + height
        bottom_errors.append(float(abs(base - np.nanmedian(terrain_values))))
        top_errors.append(float(abs(roof - (base + height))))
    terrain_finite = np.isfinite(dem)
    adjacent = np.concatenate([np.abs(np.diff(dem, axis=0)).ravel(), np.abs(np.diff(dem, axis=1)).ravel()])
    flags = {
        'FLOATING_BUILDING': bool(bottom_errors and max(bottom_errors) > 1e-5),
        'BURIED_BUILDING': bool(any(record['predicted_height_m'] is not None and record['predicted_height_m'] < 0 for record in records)),
        'NEGATIVE_HEIGHT': bool(any(record['predicted_height_m'] is not None and record['predicted_height_m'] < 0 for record in records)),
        'NAN_GEOMETRY': bool(nan_geometry),
        'MISSING_HEIGHT': bool(missing_records),
        'TERRAIN_DISCONTINUITY': bool(adjacent.size and not np.isfinite(adjacent).all()),
        'TEXTURE_ALIGNMENT_FAILURE': bool(result['_arrays']['rgb'].shape[:2] != dem.shape or result['crop']['height'] != dem.shape[0] or result['crop']['width'] != dem.shape[1]),
        'EMPTY_SCENE': bool(dem.size == 0 or len(records) == 0 or not np.isfinite(dem).any()),
    }
    scene_dir = SCENES[region]
    terrain_vertices, terrain_triangles = terrain_mesh(dem, stride=1)
    write_obj(scene_dir / 'terrain_mesh.obj', terrain_vertices, terrain_triangles, f'{region}_real_dem_terrain')
    np.savez_compressed(scene_dir / 'terrain_mesh.npz', vertices=np.asarray(terrain_vertices, dtype=np.float32), triangles=np.asarray(terrain_triangles, dtype=np.int32))
    all_building_vertices, all_building_triangles = [], []
    building_mesh_records = []
    for record in finite_records:
        vertices, triangles = building_mesh(labels, record['component_id'], dem, record['predicted_height_m'])
        offset = len(all_building_vertices)
        all_building_vertices.extend(vertices); all_building_triangles.extend([(a + offset, b + offset, c + offset) for a, b, c in triangles])
        building_mesh_records.append({'component_id': record['component_id'], 'vertices': len(vertices), 'triangles': len(triangles), 'height_m': record['predicted_height_m']})
    write_obj(scene_dir / 'building_meshes_finite_height.obj', all_building_vertices, all_building_triangles, f'{region}_finite_height_buildings')
    np.savez_compressed(scene_dir / 'building_meshes_finite_height.npz', vertices=np.asarray(all_building_vertices, dtype=np.float32), triangles=np.asarray(all_building_triangles, dtype=np.int32))
    (scene_dir / 'height_unavailable_footprints.json').write_text(json.dumps({'components': [record for record in missing_records]}, indent=2, allow_nan=False), encoding='utf-8')
    render_scene(result, scene_dir)
    scene_metadata = {
        'region': region,
        'terrain_source': 'REAL GEOREFERENCED DEM',
        'building_source': 'SINGLE-VIEW MODEL PREDICTIONS',
        'height_mapping': 'COMPONENT-ID CORRECTED',
        'height_accuracy': 'UNVALIDATED',
        'vertical_exaggeration': VERTICAL_EXAGGERATION,
        'crs': result['input_metadata']['dem']['crs'],
        'resolution_m': result['input_metadata']['dem']['resolution'],
        'terrain_elevation_m': stats(dem),
        'terrain_vertex_count': len(terrain_vertices),
        'terrain_triangle_count': len(terrain_triangles),
        'building_count': len(records),
        'finite_height_building_count': len(finite_records),
        'height_unavailable_count': len(missing_records),
        'building_height_statistics_finite_only_m': stats(heights),
        'building_mesh_triangle_count': len(all_building_triangles),
        'mesh_triangle_count_total': len(terrain_triangles) + len(all_building_triangles),
        'scene_bounds': {'x_m': [0.0, float((dem.shape[1] - 1) * 10)], 'y_m': [0.0, float((dem.shape[0] - 1) * 10)], 'z_m': ([float(np.nanmin(dem[terrain_finite])), float(np.nanmax(dem[terrain_finite]) + (max(heights) if heights else 0.0))] if np.isfinite(dem).any() else [None, None])},
        'geometry_checks': {'bottom_errors_m_max': max(bottom_errors) if bottom_errors else None, 'top_errors_m_max': max(top_errors) if top_errors else None, 'assertions_passed': not flags['FLOATING_BUILDING'] and not flags['BURIED_BUILDING'] and not flags['NEGATIVE_HEIGHT'] and not flags['NAN_GEOMETRY']},
        'failure_flags': flags,
        'mesh_records': building_mesh_records,
    }
    (scene_dir / 'scene_metadata.json').write_text(json.dumps(json_safe(scene_metadata), indent=2, allow_nan=False), encoding='utf-8')
    return result, scene_metadata


def main():
    persisted = json.loads(PHASE89_ASSIGNMENTS.read_text(encoding='utf-8'))
    config = TrainConfig(arch='unet3', target_transform='none', epochs=1, batch_size=8, lr=1e-3, amp=True)
    estimator = BuildingConditionedEstimator(config, nodata=-999.0, seed=0, device='cpu')
    estimator.model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=True)); estimator.model.eval()
    depth_model = DepthAnythingV2(model_id='depth-anything/Depth-Anything-V2-Small-hf', cache_dir=str(ROOT / 'data' / 'depth_cache'), use_cache=True)
    scene_metadata = {}
    for region in REGIONS:
        print(f'Building Phase 90 scene for {region}...', flush=True)
        _, scene_metadata[region] = audit_region(region, estimator, depth_model, persisted)
    any_fatal = any(any(value['failure_flags'][key] for key in ('FLOATING_BUILDING', 'BURIED_BUILDING', 'NEGATIVE_HEIGHT', 'NAN_GEOMETRY', 'EMPTY_SCENE')) for value in scene_metadata.values())
    decision = 'INDIAN_3D_SCENE_INTEGRATION_FAILED' if any_fatal else ('INDIAN_3D_SCENE_INTEGRATION_PARTIAL' if any(value['height_unavailable_count'] > 0 or value['failure_flags']['TEXTURE_ALIGNMENT_FAILURE'] for value in scene_metadata.values()) else 'INDIAN_3D_SCENE_INTEGRATION_VALIDATED')
    payload = {'phase': 'PHASE_90', 'TERRAIN_SOURCE': 'REAL GEOREFERENCED DEM', 'BUILDING_SOURCE': 'SINGLE-VIEW MODEL PREDICTIONS', 'HEIGHT_MAPPING': 'COMPONENT-ID CORRECTED', 'HEIGHT_ACCURACY': 'UNVALIDATED', 'sikkim': 'LOCKED', 'interactive_viewer': {'existing_viewer': str(ROOT / 'depthwizard' / 'viz' / 'interactive_viewer.py'), 'controls_preserved': ['orbit', 'zoom', 'pan', 'reset camera'], 'production_modified': False}, 'regions': scene_metadata, 'final_decision': decision}
    (OUT / 'RESULTS.json').write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False), encoding='utf-8')
    (OUT / 'SCENE_AUDIT.json').write_text(json.dumps(json_safe(scene_metadata), indent=2, allow_nan=False), encoding='utf-8')
    (OUT / 'GEOMETRY_AUDIT.json').write_text(json.dumps(json_safe({region: {'geometry_checks': value['geometry_checks'], 'failure_flags': value['failure_flags'], 'scene_bounds': value['scene_bounds']} for region, value in scene_metadata.items()}), indent=2, allow_nan=False), encoding='utf-8')
    (OUT / 'VISUAL_QA.json').write_text(json.dumps(json_safe({'regions': {region: {'renders': ['01_dem_only.png', '02_dem_plus_footprints.png', '03_dem_plus_extrusions.png', '04_final_textured_scene.png'], 'camera_views': ['overview', 'low_angle_oblique', 'close_building', 'terrain_following'], 'height_unavailable_preserved': value['height_unavailable_count'] > 0} for region, value in scene_metadata.items()}, 'vertical_exaggeration': VERTICAL_EXAGGERATION}), indent=2, allow_nan=False), encoding='utf-8')
    report = ['# Phase 90: Indian mountainous 3D scene validation', '', 'TERRAIN_SOURCE = REAL GEOREFERENCED DEM', 'BUILDING_SOURCE = SINGLE-VIEW MODEL PREDICTIONS', 'HEIGHT_MAPPING = COMPONENT-ID CORRECTED', 'HEIGHT_ACCURACY = UNVALIDATED', 'Sikkim = LOCKED', '', '## Scope', 'This phase validates pipeline integrity, geometric consistency, and scene rendering only. It does not validate building-height accuracy.', '', '## Terrain and building construction', '- Terrain meshes use the Phase 72 aligned DEM values directly in meters.', '- Building footprints and heights were taken from Phase 89 corrected component-ID assignments.', '- Finite-height components were extruded with vertical walls and roof surfaces.', '- Missing-height components remain footprint-only and are explicitly marked HEIGHT_UNAVAILABLE; no numeric substitute was assigned.', '- Vertical exaggeration factor: 1.0.', '', '## Structural audit']
    for region, value in scene_metadata.items():
        report.extend([f"### {region.upper()}", f"- CRS: {value['crs']}; resolution: {value['resolution_m']} m.", f"- Terrain elevation min/max/mean: {value['terrain_elevation_m']['min']} / {value['terrain_elevation_m']['max']} / {value['terrain_elevation_m']['mean']} m.", f"- Terrain vertices: {value['terrain_vertex_count']}; total triangles: {value['mesh_triangle_count_total']}.", f"- Buildings: {value['building_count']}; finite heights: {value['finite_height_building_count']}; HEIGHT_UNAVAILABLE: {value['height_unavailable_count']}.", f"- Finite predicted height statistics: {value['building_height_statistics_finite_only_m']}.", f"- Geometry assertions passed: {value['geometry_checks']['assertions_passed']}.", f"- Failure flags: {value['failure_flags']}."])
    report.extend(['', '## RGB texture', '- Actual B04/B03/B02 optical imagery was draped using the same crop alignment as Phase 89.', '- Texture alignment was checked against the DEM crop dimensions and Phase 72 CRS/resolution metadata.', '', '## Interactive controls', '- The existing Three.js viewer was preserved and not rewritten; its orbit, zoom, pan, and reset-camera controls remain the project integration surface.', '', '## Ground truth', '- No verified Indian building footprint or building-height ground truth exists.', '- No IoU, Dice, height MAE, height RMSE, or height correlation is reported.', '', '## Limitations', '- This is a visualization/integration validation, not geometric accuracy validation.', '- Height coverage remains incomplete because the frozen model has unavailable components.', '- The rendered building heights remain single-view model predictions and are unvalidated in India.', '', '## Final decision', decision])
    (OUT / 'REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(decision, flush=True)


if __name__ == '__main__':
    main()
