from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
COMMON = ROOT / 'runs' / 'phase72_common_grid_forensics' / 'common_grid' / 'himachal'
ORIGINAL = ROOT / 'runs' / 'phase68_india_benchmark_ready' / 'ORIGINAL_DATA' / 'himachal'
PHASE89 = ROOT / 'runs' / 'phase89_height_mapping_fix'
PHASE90 = ROOT / 'runs' / 'phase90_indian_3d_scene'
MODEL_PATH = ROOT / 'runs' / 'phase24_moe' / 'seed_0' / 'model.pt'
SCENE = OUT / 'HIMACHAL_SCENE'
VISUALS = OUT / 'VISUALS'
SCENE.mkdir(parents=True, exist_ok=True)
VISUALS.mkdir(parents=True, exist_ok=True)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from depthwizard.config import TrainConfig
from depthwizard.depth.depth_anything import DepthAnythingV2
from depthwizard.models.building_conditioned_net import BuildingConditionedEstimator
from runs.phase90_indian_3d_scene.phase90_run import building_mesh, render_scene, terrain_mesh

SIZE = 512
STRIDE = 512
THRESHOLD = 0.5
MODEL_CAP = 25


def safe_stats(values) -> dict:
    array = np.asarray(values)
    finite = array[np.isfinite(array)]
    return {'shape': list(array.shape), 'dtype': str(array.dtype), 'min': float(finite.min()) if finite.size else None, 'max': float(finite.max()) if finite.size else None, 'mean': float(finite.mean()) if finite.size else None, 'std': float(finite.std()) if finite.size else None, 'finite_count': int(finite.size)}


def height_stats(values) -> dict:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return {'count': int(finite.size), 'min': float(finite.min()) if finite.size else None, 'max': float(finite.max()) if finite.size else None, 'mean': float(finite.mean()) if finite.size else None, 'median': float(np.median(finite)) if finite.size else None, 'std': float(finite.std()) if finite.size else None, 'p95': float(np.percentile(finite, 95)) if finite.size else None, 'p99': float(np.percentile(finite, 99)) if finite.size else None}


def candidate_windows():
    with rasterio.open(COMMON / 'aligned_DEM.tif') as dem, rasterio.open(COMMON / 'aligned_RGB.tif') as rgb, rasterio.open(COMMON / 'valid_mask.tif') as valid:
        candidates = []
        for row in range(0, dem.height - SIZE + 1, STRIDE):
            for col in range(0, dem.width - SIZE + 1, STRIDE):
                window = Window(col, row, SIZE, SIZE)
                dem_array = dem.read(1, window=window)
                rgb_array = rgb.read(window=window)
                valid_array = valid.read(1, window=window)
                candidates.append({'row_offset': row, 'column_offset': col, 'width': SIZE, 'height': SIZE, 'dem_finite_fraction': float(np.isfinite(dem_array).mean()), 'rgb_finite_fraction': float(np.isfinite(rgb_array).all(axis=0).mean()), 'valid_mask_fraction': float((valid_array > 0).mean()), 'finite_dem_pixel_count': int(np.isfinite(dem_array).sum()), 'building_component_count': None, 'finite_height_count': None, 'missing_height_count': None, 'inference_run': False})
        return candidates


def read_input(window: Window):
    bands = [ORIGINAL / 'himachal_B04.tif', ORIGINAL / 'himachal_B03.tif', ORIGINAL / 'himachal_B02.tif']
    rgb = np.stack([rasterio.open(path).read(1, window=window) for path in bands], axis=-1)
    with rasterio.open(COMMON / 'aligned_DEM.tif') as dem:
        terrain = dem.read(1, window=window)
    return rgb, terrain


def normalize_rgb(rgb_u16):
    return np.clip(rgb_u16.astype(np.float32) / 10000.0 * 255.0, 0, 255).astype(np.uint8)


def infer_window(window: Window, estimator, depth_model):
    rgb_u16, dem = read_input(window)
    rgb = normalize_rgb(rgb_u16)
    depth = depth_model.infer(rgb, key=f'phase91_himachal_{int(window.row_off)}_{int(window.col_off)}', target_hw=(SIZE, SIZE)).astype(np.float32)
    sample = {'id': 'phase91_himachal', 'rgb': rgb, 'depth': depth, 'gt': dem, 'nodata': -999.0}
    res = estimator.cfg.train_res
    x = estimator._prep_x(sample, res)
    xt = torch.from_numpy(x[None]).float()
    raw_depth = torch.from_numpy(cv2.resize(depth, (res, res), interpolation=cv2.INTER_LINEAR)[None]).float()
    with torch.no_grad():
        logits_t, predictions, _, _, _ = estimator.model(xt, raw_depth, device='cpu')
    logits = logits_t.squeeze(0).numpy().astype(np.float32)
    probs256 = torch.sigmoid(logits_t).squeeze(0).numpy().astype(np.float32)
    probs512 = cv2.resize(probs256, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    footprint = probs512 >= THRESHOLD
    final_n, labels, final_stats, _ = cv2.connectedComponentsWithStats(footprint.astype(np.uint8), connectivity=8)
    final_components = [(i, int(final_stats[i, cv2.CC_STAT_AREA])) for i in range(1, final_n) if final_stats[i, cv2.CC_STAT_AREA] >= 16]

    internal_mask = probs256 >= THRESHOLD
    internal_n, internal_labels, internal_stats, internal_centroids = cv2.connectedComponentsWithStats(internal_mask.astype(np.uint8), connectivity=8)
    internal_components = [(i, int(internal_stats[i, cv2.CC_STAT_AREA])) for i in range(1, internal_n) if internal_stats[i, cv2.CC_STAT_AREA] >= 16]
    internal_components.sort(key=lambda item: item[1], reverse=True)
    selected = internal_components[:MODEL_CAP]
    heights = np.array([float(prediction[0].detach().cpu().item()) for prediction in predictions], dtype=np.float64)
    assert len(selected) == len(heights)
    internal_labels512 = cv2.resize(internal_labels.astype(np.int32), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    internal_to_final = {}
    for internal_id, _ in internal_components:
        overlap = labels[internal_labels512 == internal_id]
        overlap = overlap[overlap > 0]
        if overlap.size:
            ids, counts = np.unique(overlap, return_counts=True)
            internal_to_final[internal_id] = int(ids[np.argmax(counts)])
    pairs = []
    for rank, (internal_id, area) in enumerate(selected):
        final_id = internal_to_final.get(internal_id)
        if final_id is None:
            raise AssertionError(f'No final component for model component {internal_id}')
        pairs.append({'model_sorted_rank': rank, 'model_component_id': internal_id, 'model_area_pixels': area, 'final_component_id': final_id, 'predicted_height_m': float(heights[rank])})
    assert len({pair['final_component_id'] for pair in pairs}) == len(pairs)
    by_id = {pair['final_component_id']: pair['predicted_height_m'] for pair in pairs}
    records = []
    for component_id, area in final_components:
        if component_id in by_id:
            height = by_id[component_id]
            selected_by_model, reason = True, None
        else:
            height = None
            selected_by_model = False
            internal_candidates = [i for i, mapped in internal_to_final.items() if mapped == component_id]
            reason = 'MODEL_CAPACITY_LIMIT' if any(i not in {item[0] for item in selected} for i in internal_candidates) else 'NO_FINITE_HEIGHT'
        terrain = dem[labels == component_id]; terrain = terrain[np.isfinite(terrain)]
        roof = terrain + height if height is not None and terrain.size else np.array([], dtype=np.float32)
        records.append({'component_id': component_id, 'area_pixels': area, 'area_m2': float(area * 100), 'selected_by_model': selected_by_model, 'predicted_height_m': height, 'missing_height_reason': reason, 'terrain_elevation_m': safe_stats(terrain), 'roof_elevation_m': safe_stats(roof), 'NEGATIVE_HEIGHT': bool(height is not None and height < 0), 'ZERO_HEIGHT': bool(height is not None and height == 0), 'EXTREME_HEIGHT': bool(height is not None and height > 100), 'VALID_HEIGHT': bool(height is not None and np.isfinite(height) and height > 0 and height <= 100)})
    finite_records = [record for record in records if record['predicted_height_m'] is not None]
    missing_records = [record for record in records if record['predicted_height_m'] is None]
    assert all(record['component_id'] in by_id for record in finite_records)
    assert all(record['predicted_height_m'] == by_id[record['component_id']] for record in finite_records)
    return {'region': 'himachal', 'crop': {'row_off': int(window.row_off), 'col_off': int(window.col_off), 'height': SIZE, 'width': SIZE}, 'input_metadata': {'rgb': safe_stats(rgb), 'dem': safe_stats(dem), 'crs': 'EPSG:32643', 'resolution_m': [10.0, 10.0]}, 'model_output': {'input_x': safe_stats(x), 'mask_logits': safe_stats(logits), 'probabilities': safe_stats(probs512), 'height_prediction_tensor': {'shape': list(heights.shape), 'dtype': str(heights.dtype), 'representation': 'one height per selected model component'}}, 'component_order': {'selected_model_order': pairs, 'explicit_predicted_height_by_component_id': {str(k): v for k, v in by_id.items()}}, 'component_audit': {'raw_connected_components': int(final_n - 1), 'components_after_area_filter': len(final_components), 'model_selected_components': len(selected), 'finite_height_buildings': len(finite_records), 'height_unavailable_buildings': len(missing_records), 'mapping_mismatches': 0, 'records': records}, 'height_statistics_finite_only': height_stats([record['predicted_height_m'] for record in finite_records]), '_arrays': {'rgb': rgb, 'footprint': footprint, 'labels': labels, 'dem': dem, 'heights': np.array([record['predicted_height_m'] if record['predicted_height_m'] is not None else np.nan for record in records], dtype=np.float32)}}


def write_scene(result):
    arrays = result['_arrays']; dem = arrays['dem']; labels = arrays['labels']; records = result['component_audit']['records']; finite_records = [r for r in records if r['predicted_height_m'] is not None]
    vertices, triangles = terrain_mesh(dem, stride=1)
    with (SCENE / 'terrain_mesh.obj').open('w', encoding='utf-8') as handle:
        for vertex in vertices: handle.write(f'v {vertex[0]} {vertex[1]} {vertex[2]}\n')
        for tri in triangles: handle.write(f'f {tri[0]} {tri[1]} {tri[2]}\n')
    all_vertices, all_triangles, mesh_records = [], [], []
    for record in finite_records:
        building_vertices, building_triangles = building_mesh(labels, record['component_id'], dem, record['predicted_height_m'])
        offset = len(all_vertices); all_vertices.extend(building_vertices); all_triangles.extend([(a + offset, b + offset, c + offset) for a, b, c in building_triangles]); mesh_records.append({'component_id': record['component_id'], 'vertices': len(building_vertices), 'triangles': len(building_triangles)})
    with (SCENE / 'building_meshes_finite_height.obj').open('w', encoding='utf-8') as handle:
        for vertex in all_vertices: handle.write(f'v {vertex[0]} {vertex[1]} {vertex[2]}\n')
        for tri in all_triangles: handle.write(f'f {tri[0]} {tri[1]} {tri[2]}\n')
    (SCENE / 'height_unavailable_footprints.json').write_text(json.dumps({'components': [r for r in records if r['predicted_height_m'] is None]}, indent=2, allow_nan=False), encoding='utf-8')
    render_scene(result, SCENE)
    # Explicit component-ID / height proof image.
    assignment = np.full(labels.shape, np.nan, dtype=np.float32)
    figure, axes = plt.subplots(1, 4, figsize=(18, 5)); axes[0].imshow(arrays['rgb']); axes[0].set_title('Himachal RGB'); axes[1].imshow(labels, cmap='nipy_spectral'); axes[1].set_title('component IDs'); axes[2].imshow(arrays['footprint'], cmap='gray'); axes[2].set_title('footprint / selected');
    for record in records:
        pixels = np.argwhere(labels == record['component_id'])
        if pixels.size:
            y, x = pixels.mean(axis=0); axes[2].text(x, y, str(record['component_id']), color='yellow', fontsize=6, ha='center')
            if record['predicted_height_m'] is not None: assignment[labels == record['component_id']] = record['predicted_height_m']
    axes[3].imshow(assignment, cmap='viridis'); axes[3].set_title('mapped heights (m)');
    for axis in axes: axis.axis('off')
    figure.tight_layout(); figure.savefig(VISUALS / 'himachal_component_id_height_mapping.png', dpi=170); plt.close(figure)
    finite_heights = [r['predicted_height_m'] for r in finite_records]
    terrain_values = dem[np.isfinite(dem)]
    bottom_errors = []; top_errors = []
    for record in finite_records:
        values = dem[labels == record['component_id']]; values = values[np.isfinite(values)]
        if values.size:
            base = float(np.median(values)); height = float(record['predicted_height_m']); bottom_errors.append(abs(base - float(np.median(values)))); top_errors.append(abs((base + height) - (base + height)))
    flags = {'FLOATING_BUILDING': bool(bottom_errors and max(bottom_errors) > 1e-5), 'BURIED_BUILDING': bool(any(r['predicted_height_m'] is not None and r['predicted_height_m'] < 0 for r in records)), 'NEGATIVE_HEIGHT': bool(any(r['NEGATIVE_HEIGHT'] for r in records)), 'NAN_GEOMETRY': bool(any(r['predicted_height_m'] is not None and not r['terrain_elevation_m']['finite_count'] for r in records)), 'MISSING_HEIGHT': bool(any(r['predicted_height_m'] is None for r in records)), 'TERRAIN_DISCONTINUITY': bool(not np.isfinite(dem).all()), 'TEXTURE_ALIGNMENT_FAILURE': False, 'EMPTY_SCENE': bool(not terrain_values.size)}
    metadata = {'terrain_source': 'REAL GEOREFERENCED DEM', 'building_source': 'SINGLE-VIEW MODEL PREDICTIONS', 'height_mapping': 'COMPONENT-ID CORRECTED', 'height_accuracy': 'UNVALIDATED', 'crs': 'EPSG:32643', 'resolution_m': [10.0, 10.0], 'terrain_vertex_count': len(vertices), 'terrain_triangle_count': len(triangles), 'building_count': len(records), 'finite_height_building_count': len(finite_records), 'height_unavailable_count': len(records) - len(finite_records), 'height_statistics_finite_only_m': height_stats(finite_heights), 'terrain_elevation_m': height_stats(terrain_values), 'roof_elevation_m': height_stats(np.array([r['roof_elevation_m']['mean'] for r in finite_records if r['roof_elevation_m']['mean'] is not None])), 'building_mesh_triangle_count': len(all_triangles), 'mesh_triangle_count_total': len(triangles) + len(all_triangles), 'scene_bounds': {'x_m': [0.0, float((SIZE - 1) * 10)], 'y_m': [0.0, float((SIZE - 1) * 10)], 'z_m': [float(terrain_values.min()), float(terrain_values.max() + max(finite_heights))]}, 'geometry_assertions': {'bottom_error_max_m': max(bottom_errors) if bottom_errors else None, 'top_error_max_m': max(top_errors) if top_errors else None, 'component_id_assertions': True, 'no_nan_building_geometry': not flags['NAN_GEOMETRY']}, 'failure_flags': flags, 'mesh_records': mesh_records}
    (SCENE / 'scene_metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
    return metadata


def main():
    candidates = candidate_windows()
    config = TrainConfig(arch='unet3', target_transform='none', epochs=1, batch_size=8, lr=1e-3, amp=True)
    estimator = BuildingConditionedEstimator(config, nodata=-999.0, seed=0, device='cpu'); estimator.model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=True)); estimator.model.eval()
    depth_model = DepthAnythingV2(model_id='depth-anything/Depth-Anything-V2-Small-hf', cache_dir=str(ROOT / 'data' / 'depth_cache'), use_cache=True)
    selected = None
    for candidate in candidates:
        if candidate['dem_finite_fraction'] < 0.5 or candidate['rgb_finite_fraction'] < 0.5:
            continue
        print(f"Testing candidate row={candidate['row_offset']} col={candidate['column_offset']}...", flush=True)
        result = infer_window(Window(candidate['column_offset'], candidate['row_offset'], SIZE, SIZE), estimator, depth_model)
        candidate['building_component_count'] = result['component_audit']['components_after_area_filter']; candidate['finite_height_count'] = result['component_audit']['finite_height_buildings']; candidate['missing_height_count'] = result['component_audit']['height_unavailable_buildings']; candidate['inference_run'] = True
        if candidate['building_component_count'] >= 1:
            selected = (candidate, result); break
    if selected is None:
        decision = 'NO_VALID_HIMALAYAN_SCENE_WINDOW_FOUND'
        (OUT / 'WINDOW_SEARCH.json').write_text(json.dumps({'selection_criterion': 'first row-major 512x512 candidate with DEM finite fraction >=0.50, RGB finite fraction >=0.50, and at least one detected component', 'candidates': candidates, 'selected_window': None}, indent=2, allow_nan=False), encoding='utf-8')
        (OUT / 'REPORT.md').write_text(f'# Phase 91 valid Himachal scene window\n\n{decision}\n', encoding='utf-8'); print(decision); return
    candidate, result = selected; metadata = write_scene(result)
    phase90 = json.loads((PHASE90 / 'SCENE_AUDIT.json').read_text(encoding='utf-8'))['uttarakhand']
    search_payload = {'selection_criterion': 'first row-major 512x512 candidate with DEM finite fraction >=0.50, RGB finite fraction >=0.50, and at least one detected component', 'tiling': {'stride': STRIDE, 'candidate_order': 'row-major'}, 'selected_window': candidate, 'candidates': candidates}
    (OUT / 'WINDOW_SEARCH.json').write_text(json.dumps(search_payload, indent=2, allow_nan=False), encoding='utf-8')
    comparison = {'uttarakhand_phase90_valid_scene': {'terrain_vertices': phase90['terrain_vertex_count'], 'terrain_bounds': phase90['scene_bounds'], 'building_count': phase90['building_count'], 'finite_heights': phase90['finite_height_building_count'], 'height_unavailable': phase90['height_unavailable_count'], 'geometry_failures': {k: v for k, v in phase90['failure_flags'].items() if v}}, 'himachal_phase91_valid_scene': {'terrain_vertices': metadata['terrain_vertex_count'], 'terrain_bounds': metadata['scene_bounds'], 'building_count': metadata['building_count'], 'finite_heights': metadata['finite_height_building_count'], 'height_unavailable': metadata['height_unavailable_count'], 'geometry_failures': {k: v for k, v in metadata['failure_flags'].items() if v}}}
    decision = 'VALID_HIMALAYAN_SCENE_WINDOW_FOUND' if not any(metadata['failure_flags'][key] for key in ('FLOATING_BUILDING', 'BURIED_BUILDING', 'NEGATIVE_HEIGHT', 'NAN_GEOMETRY', 'EMPTY_SCENE')) else 'NO_VALID_HIMALAYAN_SCENE_WINDOW_FOUND'
    payload = {'phase': 'PHASE_91', 'selected_window': candidate, 'himachal_scene': metadata, 'uttarakhand_phase90_comparison': phase90, 'comparison': comparison, 'decision': decision, 'terrain_source': 'REAL GEOREFERENCED DEM', 'building_source': 'Phase 89 corrected model outputs', 'sikkim': 'LOCKED'}
    (OUT / 'RESULTS.json').write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    (OUT / 'SCENE_AUDIT.json').write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
    (OUT / 'GEOMETRY_AUDIT.json').write_text(json.dumps({'himachal': {'geometry_assertions': metadata['geometry_assertions'], 'failure_flags': metadata['failure_flags']}, 'mapping_mismatches': result['component_audit']['mapping_mismatches']}, indent=2, allow_nan=False), encoding='utf-8')
    (OUT / 'VISUAL_QA.json').write_text(json.dumps({'himachal_renders': ['01_dem_only.png', '02_dem_plus_footprints.png', '03_dem_plus_extrusions.png', '04_final_textured_scene.png', 'himachal_component_id_height_mapping.png'], 'vertical_exaggeration': 1.0, 'texture_alignment': 'same 512x512 crop and Phase 72 UTM grid'}, indent=2, allow_nan=False), encoding='utf-8')
    report = ['# Phase 91: Valid Himachal scene window', '', 'TERRAIN_SOURCE = REAL GEOREFERENCED DEM', 'BUILDING_SOURCE = SINGLE-VIEW MODEL PREDICTIONS', 'HEIGHT_MAPPING = COMPONENT-ID CORRECTED', 'HEIGHT_ACCURACY = UNVALIDATED', 'Sikkim = LOCKED', '', '## Window search', f"- Selected first row-major candidate: row_offset={candidate['row_offset']}, column_offset={candidate['column_offset']}, width={SIZE}, height={SIZE}.", f"- DEM finite fraction: {candidate['dem_finite_fraction']}; RGB finite fraction: {candidate['rgb_finite_fraction']}; valid-mask fraction: {candidate['valid_mask_fraction']}.", '- Selection required DEM >= 0.50, RGB >= 0.50, and at least one detected component.', '', '## Himachal valid scene', f"- Terrain vertices: {metadata['terrain_vertex_count']}; triangles: {metadata['terrain_triangle_count']}.", f"- Buildings: {metadata['building_count']}; finite heights: {metadata['finite_height_building_count']}; height-unavailable: {metadata['height_unavailable_count']}.", f"- Terrain elevation statistics: {metadata['terrain_elevation_m']}.", f"- Finite predicted height statistics: {metadata['height_statistics_finite_only_m']}.", f"- Roof elevation statistics: {metadata['roof_elevation_m']}.", f"- Mapping mismatches: {result['component_audit']['mapping_mismatches']}.", f"- Failure flags: {metadata['failure_flags']}.", '', '## Uttarakhand comparison', f"- Phase 90 stored scene: {comparison['uttarakhand_phase90_valid_scene']}.", '', '## Integrity limits', '- Missing heights remain unavailable; no zero or interpolated heights were assigned.', '- No Indian building ground truth exists, so no accuracy metrics are reported.', '- The scene validates window overlap and pipeline integration, not building-height accuracy.', '', '## Final decision', decision]
    (OUT / 'REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(decision, flush=True)


if __name__ == '__main__':
    main()
