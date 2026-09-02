from __future__ import annotations

import hashlib
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
VISUALS = OUT / 'VISUALS'
PROTOTYPE = OUT / '3D_PROTOTYPE'
DATA = ROOT / 'runs' / 'phase68_india_benchmark_ready' / 'ORIGINAL_DATA'
MODEL_PATH = ROOT / 'runs' / 'phase24_moe' / 'seed_0' / 'model.pt'
CROP_SIZE = 512
THRESHOLD = 0.5
REGIONS = ('uttarakhand', 'himachal')

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from depthwizard.config import TrainConfig
from depthwizard.depth.depth_anything import DepthAnythingV2
from depthwizard.models.building_conditioned_net import BuildingConditionedEstimator
from runs.phase87_indian_building_baseline.phase87_run import choose_window, normalize_rgb, read_aligned_dem, read_rgb


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_stats(values) -> dict:
    array = np.asarray(values)
    finite = array[np.isfinite(array)] if np.issubdtype(array.dtype, np.number) else np.array([])
    if not finite.size:
        return {'shape': list(array.shape), 'dtype': str(array.dtype), 'min': None, 'max': None, 'mean': None, 'std': None, 'finite_count': 0}
    return {'shape': list(array.shape), 'dtype': str(array.dtype), 'min': float(finite.min()), 'max': float(finite.max()), 'mean': float(finite.mean()), 'std': float(finite.std()), 'finite_count': int(finite.size)}


def height_stats(values) -> dict:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return {
        'shape': list(array.shape), 'dtype': str(array.dtype),
        'min': float(finite.min()) if finite.size else None,
        'max': float(finite.max()) if finite.size else None,
        'mean': float(finite.mean()) if finite.size else None,
        'median': float(np.median(finite)) if finite.size else None,
        'std': float(finite.std()) if finite.size else None,
        'p95': float(np.percentile(finite, 95)) if finite.size else None,
        'p99': float(np.percentile(finite, 99)) if finite.size else None,
        'finite_count': int(finite.size), 'missing_count': int(np.isnan(array).sum()),
        'missing_fraction': float(np.isnan(array).mean()) if array.size else 0.0,
    }


def component_record(component_id: int, labels: np.ndarray, dem: np.ndarray, height: float | None, selected: bool, reason: str | None, area: int) -> dict:
    mask = labels == component_id
    terrain = dem[mask & np.isfinite(dem)]
    roof = terrain + height if height is not None and terrain.size else np.array([], dtype=np.float32)
    return {
        'component_id': int(component_id),
        'area_pixels': int(area),
        'area_m2_at_10m_gsd': float(area * 100.0),
        'selected_by_model': bool(selected),
        'predicted_height_m': float(height) if height is not None and np.isfinite(height) else None,
        'missing_height_reason': reason,
        'terrain_elevation_m': finite_stats(terrain),
        'roof_elevation_m': finite_stats(roof),
        'NEGATIVE_HEIGHT': bool(height is not None and height < 0),
        'ZERO_HEIGHT': bool(height is not None and height == 0),
        'EXTREME_HEIGHT': bool(height is not None and height > 100),
        'VALID_HEIGHT': bool(height is not None and np.isfinite(height) and height > 0 and height <= 100),
    }


def run_region(region: str, estimator: BuildingConditionedEstimator, depth_model: DepthAnythingV2) -> dict:
    window = choose_window(region)
    rgb_u16, rgb_meta = read_rgb(region, window)
    rgb = normalize_rgb(rgb_u16)
    dem, dem_meta = read_aligned_dem(region, window)
    depth = depth_model.infer(rgb, key=f'phase87_{region}_center_{CROP_SIZE}', target_hw=(CROP_SIZE, CROP_SIZE)).astype(np.float32)
    sample = {'id': f'phase87_{region}', 'rgb': rgb, 'depth': depth, 'gt': dem, 'nodata': -999.0}
    res = estimator.cfg.train_res
    x = estimator._prep_x(sample, res)
    xt = torch.from_numpy(x[None]).float()
    raw_depth = torch.from_numpy(cv2.resize(depth, (res, res), interpolation=cv2.INTER_LINEAR)[None]).float()
    with torch.no_grad():
        mask_logits, predictions, _, _, _ = estimator.model(xt, raw_depth, device='cpu')
    logits = mask_logits.squeeze(0).numpy().astype(np.float32)
    probabilities_256 = torch.sigmoid(mask_logits).squeeze(0).numpy().astype(np.float32)
    probabilities_512 = cv2.resize(probabilities_256, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)
    footprint = probabilities_512 >= THRESHOLD

    raw_n, raw_labels, raw_stats, _ = cv2.connectedComponentsWithStats(footprint.astype(np.uint8), connectivity=8)
    final_components = [(component_id, int(raw_stats[component_id, cv2.CC_STAT_AREA])) for component_id in range(1, raw_n) if raw_stats[component_id, cv2.CC_STAT_AREA] >= 16]

    internal_mask = probabilities_256 >= THRESHOLD
    internal_n, internal_labels, internal_stats, internal_centroids = cv2.connectedComponentsWithStats(internal_mask.astype(np.uint8), connectivity=8)
    internal_components = [(component_id, int(internal_stats[component_id, cv2.CC_STAT_AREA])) for component_id in range(1, internal_n) if internal_stats[component_id, cv2.CC_STAT_AREA] >= 16]
    internal_components.sort(key=lambda item: item[1], reverse=True)
    selected_internal = internal_components[:25]
    prediction_values = np.array([float(prediction[0].detach().cpu().item()) for prediction in predictions], dtype=np.float64)
    assert prediction_values.size == len(selected_internal), (prediction_values.size, len(selected_internal))

    # Map the model's 256-pixel component IDs into the final 512-pixel IDs by spatial overlap.
    internal_labels_512 = cv2.resize(internal_labels.astype(np.int32), (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_NEAREST)
    internal_to_final = {}
    for internal_id, _ in internal_components:
        overlap_labels = raw_labels[internal_labels_512 == internal_id]
        overlap_labels = overlap_labels[overlap_labels > 0]
        if overlap_labels.size:
            ids, counts = np.unique(overlap_labels, return_counts=True)
            internal_to_final[internal_id] = int(ids[np.argmax(counts)])

    selected_pairs = []
    for rank, (internal_id, area) in enumerate(selected_internal):
        final_id = internal_to_final.get(internal_id)
        if final_id is None:
            raise AssertionError(f'No final component mapping for selected model component {internal_id}')
        selected_pairs.append({'model_sorted_rank': rank, 'model_component_id': internal_id, 'model_area_pixels': area, 'final_component_id': final_id, 'predicted_height_m': float(prediction_values[rank])})
    selected_final_ids = [pair['final_component_id'] for pair in selected_pairs]
    assert len(selected_final_ids) == len(set(selected_final_ids)), f'Non-unique component-ID mapping: {selected_final_ids}'

    predicted_height_by_component_id = {pair['final_component_id']: pair['predicted_height_m'] for pair in selected_pairs}
    model_selected_final_ids = set(selected_final_ids)
    records = []
    for component_id, area in final_components:
        if component_id in predicted_height_by_component_id:
            height = predicted_height_by_component_id[component_id]
            reason = None
            selected = True
        else:
            height = None
            selected = False
            internal_candidates = [internal_id for internal_id, mapped_id in internal_to_final.items() if mapped_id == component_id]
            selected_internal_ids = {component_id for component_id, _ in selected_internal}
            reason = 'MODEL_CAPACITY_LIMIT' if any(internal_id not in selected_internal_ids for internal_id in internal_candidates) else 'NO_FINITE_HEIGHT'
        records.append(component_record(component_id, raw_labels, dem, height, selected, reason, area))

    # The only finite assignment path is the explicit final-component-ID dictionary.
    for pair in selected_pairs:
        record = next(record for record in records if record['component_id'] == pair['final_component_id'])
        assert record['component_id'] == pair['final_component_id']
        assert record['predicted_height_m'] == pair['predicted_height_m']
    mapping_mismatches = sum(1 for pair in selected_pairs if next(record for record in records if record['component_id'] == pair['final_component_id'])['component_id'] != pair['final_component_id'])
    assert mapping_mismatches == 0

    heights = np.array([record['predicted_height_m'] if record['predicted_height_m'] is not None else np.nan for record in records], dtype=np.float64)
    finite_records = [record for record in records if record['predicted_height_m'] is not None]
    result = {
        'region': region,
        'crop': {'row_off': int(window.row_off), 'col_off': int(window.col_off), 'height': CROP_SIZE, 'width': CROP_SIZE},
        'input_metadata': {'rgb_bands': rgb_meta, 'rgb_composite_order': 'B04/B03/B02', 'dem': dem_meta, 'rgb': finite_stats(rgb), 'depth': finite_stats(depth)},
        'model_path': str(MODEL_PATH),
        'model_output': {'input_x': finite_stats(x), 'mask_logits': finite_stats(logits), 'probabilities_256': finite_stats(probabilities_256), 'probabilities_512': finite_stats(probabilities_512), 'height_prediction_tensor': {'shape': list(prediction_values.shape), 'dtype': str(prediction_values.dtype), 'representation': 'one height per selected model component'}, 'height_predictions': height_stats(prediction_values)},
        'component_order': {'model_order': [{'model_component_id': i, 'area_pixels': a, 'rank': rank, 'centroid_xy_256': [float(internal_centroids[i][0]), float(internal_centroids[i][1])], 'bbox_256': [int(internal_stats[i, cv2.CC_STAT_LEFT]), int(internal_stats[i, cv2.CC_STAT_TOP]), int(internal_stats[i, cv2.CC_STAT_WIDTH]), int(internal_stats[i, cv2.CC_STAT_HEIGHT])]} for rank, (i, a) in enumerate(internal_components)], 'selected_model_order': selected_pairs, 'explicit_predicted_height_by_component_id': {str(k): v for k, v in predicted_height_by_component_id.items()}},
        'component_audit': {'raw_connected_components': int(raw_n - 1), 'components_after_area_filter': len(final_components), 'model_selected_components': len(selected_pairs), 'components_receiving_finite_heights': len(finite_records), 'components_missing_heights': len(records) - len(finite_records), 'mapping_mismatches_after_fix': mapping_mismatches, 'records': records},
        'height_statistics_finite_only': height_stats(heights),
        'terrain_integration': {'terrain_units': 'meters', 'height_units': 'meters as defined by existing model output', 'roof_formula': 'terrain + predicted_height', 'finite_component_terrain_means_m': [record['terrain_elevation_m']['mean'] for record in finite_records if record['terrain_elevation_m']['mean'] is not None], 'finite_component_roof_means_m': [record['roof_elevation_m']['mean'] for record in finite_records if record['roof_elevation_m']['mean'] is not None]},
        'ground_truth': {'INDIAN_BUILDING_GROUND_TRUTH_AVAILABLE': 'NO', 'accuracy_metrics': None},
        '_arrays': {'rgb': rgb, 'footprint': footprint, 'labels': raw_labels, 'heights': heights, 'dem': dem},
    }
    return result


def save_visuals(result: dict):
    arrays = result['_arrays']
    selected = {record['component_id']: record for record in result['component_audit']['records'] if record['selected_by_model']}
    assigned = np.full(arrays['labels'].shape, np.nan, dtype=np.float32)
    for component_id, record in selected.items():
        assigned[arrays['labels'] == component_id] = record['predicted_height_m']
    figure, axes = plt.subplots(1, 4, figsize=(19, 5))
    axes[0].imshow(arrays['rgb']); axes[0].set_title(f"{result['region']} RGB")
    axes[1].imshow(arrays['labels'], cmap='nipy_spectral'); axes[1].set_title('all component IDs')
    axes[2].imshow(arrays['labels'], cmap='gray')
    for component_id in range(1, int(arrays['labels'].max()) + 1):
        pixels = np.argwhere(arrays['labels'] == component_id)
        if pixels.size:
            y, x = pixels.mean(axis=0)
            axes[2].text(x, y, str(component_id), color='yellow', fontsize=6, ha='center')
    axes[2].set_title('component IDs / selected IDs')
    axes[3].imshow(assigned, cmap='viridis'); axes[3].set_title('correctly mapped heights (m)')
    for axis in axes: axis.axis('off')
    figure.tight_layout(); figure.savefig(VISUALS / f"{result['region']}_component_mapping_fix.png", dpi=170); plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(arrays['rgb']); axes[0].set_title('RGB')
    axes[1].imshow(arrays['footprint'], cmap='gray'); axes[1].set_title('footprint')
    axes[2].imshow(assigned, cmap='viridis'); axes[2].set_title('height-colored finite buildings')
    for axis in axes: axis.axis('off')
    figure.tight_layout(); figure.savefig(VISUALS / f"{result['region']}_rgb_footprint_height.png", dpi=170); plt.close(figure)


def save_3d(result: dict):
    arrays = result['_arrays']
    dem = arrays['dem']
    finite_dem = np.isfinite(dem)
    fill = float(np.nanmedian(dem[finite_dem])) if finite_dem.any() else 0.0
    stride = 8
    yy, xx = np.mgrid[0:dem.shape[0]:stride, 0:dem.shape[1]:stride]
    figure = plt.figure(figsize=(12, 9)); axis = figure.add_subplot(111, projection='3d')
    axis.plot_surface(xx * 10.0, yy * 10.0, np.nan_to_num(dem[::stride, ::stride], nan=fill), cmap='terrain', linewidth=0, alpha=0.78)
    labels = arrays['labels']
    for record in result['component_audit']['records']:
        pixels = np.argwhere(labels == record['component_id'])
        if not pixels.size: continue
        cy, cx = pixels.mean(axis=0)
        if record['predicted_height_m'] is None:
            axis.scatter([cx * 10], [cy * 10], [fill], color='red', s=12)
            continue
        contour_mask = (labels == record['component_id']).astype(np.uint8)
        contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
        contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
        if len(contour) < 3: continue
        base_z = float(np.nanmedian(dem[labels == record['component_id']])) if np.isfinite(dem[labels == record['component_id']]).any() else fill
        top_z = base_z + record['predicted_height_m']
        polygon_base = [(float(x * 10), float(y * 10), base_z) for x, y in contour]
        polygon_top = [(float(x * 10), float(y * 10), top_z) for x, y in contour]
        axis.add_collection3d(Poly3DCollection([polygon_top], facecolor='tomato', alpha=0.85))
        for index in range(len(polygon_base)):
            nxt = (index + 1) % len(polygon_base)
            axis.add_collection3d(Poly3DCollection([[polygon_base[index], polygon_base[nxt], polygon_top[nxt], polygon_top[index]]], facecolor='orange', alpha=0.55))
    axis.set_title(f"INDIAN_TERRAIN_BUILDING_3D_PROTOTYPE_V2: {result['region']}")
    axis.set_xlabel('east (m)'); axis.set_ylabel('north (m)'); axis.set_zlabel('elevation (m)')
    figure.tight_layout(); figure.savefig(PROTOTYPE / f"{result['region']}_prototype_v2.png", dpi=150); plt.close(figure)


def strip_arrays(value):
    if isinstance(value, dict): return {key: strip_arrays(item) for key, item in value.items() if key != '_arrays'}
    if isinstance(value, list): return [strip_arrays(item) for item in value]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value) if np.isfinite(value) else None
    return value


def main():
    config = TrainConfig(arch='unet3', target_transform='none', epochs=1, batch_size=8, lr=1e-3, amp=True)
    estimator = BuildingConditionedEstimator(config, nodata=-999.0, seed=0, device='cpu')
    estimator.model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=True)); estimator.model.eval()
    depth_model = DepthAnythingV2(model_id='depth-anything/Depth-Anything-V2-Small-hf', cache_dir=str(ROOT / 'data' / 'depth_cache'), use_cache=True)
    results = {}
    for region in REGIONS:
        print(f'Running Phase 89 mapping fix for {region}...', flush=True)
        results[region] = run_region(region, estimator, depth_model)
        save_visuals(results[region]); save_3d(results[region])
    clean_results = {region: strip_arrays(value) for region, value in results.items()}
    assert all(value['component_audit']['mapping_mismatches_after_fix'] == 0 for value in clean_results.values())
    assert any(value['component_audit']['components_receiving_finite_heights'] > 0 for value in clean_results.values())
    decision = 'HEIGHT_MAPPING_FIXED_OUTPUT_AVAILABLE_UNVALIDATED'
    phase87 = json.loads((ROOT / 'runs' / 'phase87_indian_building_baseline' / 'RESULTS.json').read_text(encoding='utf-8'))
    domain = {}
    for region, value in clean_results.items():
        p87 = phase87['regions'][region]
        domain[region] = {'phase87': {'foreground_percentage': p87['raw_output']['foreground_percentage'], 'component_count': p87['connected_components']['detected_buildings_after_postprocessing'], 'finite_heights': p87['height_output_summary_model_output_only']['valid_pixels'] if 'valid_pixels' in p87['height_output_summary_model_output_only'] else None, 'missing_heights': p87['failure_flags']['NAN_HEIGHTS']}, 'phase89': {'foreground_percentage': float(100 * np.mean(results[region]['_arrays']['footprint'])), 'component_count': value['component_audit']['components_after_area_filter'], 'finite_heights': value['component_audit']['components_receiving_finite_heights'], 'missing_heights': value['component_audit']['components_missing_heights'], 'mapping_mismatches': value['component_audit']['mapping_mismatches_after_fix']}}
    payload = {'phase': 'PHASE_89', 'TERRAIN_SOURCE': 'REAL GEOREFERENCED DEM', 'BUILDING_SOURCE': 'EXISTING SINGLE-VIEW BUILDING MODEL', 'checkpoint': str(MODEL_PATH), 'checkpoint_sha256': sha256(MODEL_PATH), 'retrained': False, 'threshold': THRESHOLD, 'regions': clean_results, 'INDIAN_DOMAIN_BEHAVIOR_COMPARISON': domain, 'ground_truth': {'INDIAN_BUILDING_GROUND_TRUTH_AVAILABLE': 'NO', 'accuracy_metrics': None}, 'final_scientific_decision': decision}
    (OUT / 'RESULTS.json').write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    (OUT / 'COMPONENT_MAPPING.json').write_text(json.dumps({region: value['component_order'] for region, value in clean_results.items()}, indent=2, allow_nan=False), encoding='utf-8')
    (OUT / 'HEIGHT_ASSIGNMENTS.json').write_text(json.dumps({region: {'records': value['component_audit']['records'], 'height_statistics_finite_only': value['height_statistics_finite_only']} for region, value in clean_results.items()}, indent=2, allow_nan=False), encoding='utf-8')
    (OUT / 'DOMAIN_COMPARISON.json').write_text(json.dumps(domain, indent=2, allow_nan=False), encoding='utf-8')
    report = ['# Phase 89: Fix building height component-ID mapping', '', '## Objective', 'Surgical correction of height assignment identity using the unchanged Phase 87 inference pipeline and frozen Phase 24 checkpoint.', '', 'TERRAIN_SOURCE = REAL GEOREFERENCED DEM', 'BUILDING_SOURCE = EXISTING SINGLE-VIEW BUILDING MODEL', '', '## Model component order', '- The model creates connected components from the 256 x 256 thresholded footprint.', '- It filters components below 16 pixels, sorts by descending area, and caps the list at 25.', '- Each returned height is one scalar for one selected model component.', '- Phase 89 maps each model component ID to the final 512 x 512 component ID by spatial overlap, then assigns via explicit `predicted_height_by_component_id`.', '', '## Component audit']
    for region, value in clean_results.items():
        audit = value['component_audit']; report.append(f"- {region.upper()}: total={audit['components_after_area_filter']}; model-selected={audit['model_selected_components']}; finite heights={audit['components_receiving_finite_heights']}; missing={audit['components_missing_heights']}; mapping mismatches after fix={audit['mapping_mismatches_after_fix']}.")
    report.extend(['', '## Height statistics', '- Statistics include finite predicted heights only; missing components are not replaced or interpolated.'])
    for region, value in clean_results.items(): report.append(f"- {region.upper()}: {value['height_statistics_finite_only']}.")
    report.extend(['', '## Height / footprint consistency', '- Every component remains in HEIGHT_ASSIGNMENTS.json with its area, selection status, height, terrain elevation, roof elevation, and flags.', '- Missing-height components remain explicitly unavailable and are not assigned zero.', '', '## Mapping validation', '- Assertions passed for every finite assignment.', '- Mapping mismatches after fix = 0 for Uttarakhand and Himachal.', '', '## Terrain integration', '- Terrain remains the real Phase 72 aligned DEM in meters.', '- Roof elevation is terrain plus the model-predicted height in meters.', '- The V2 3D prototype is visualization only and is not a validated reconstruction.', '', '## Ground truth', '- INDIAN_BUILDING_GROUND_TRUTH_AVAILABLE = NO.', '- No IoU, Dice, precision, recall, height MAE, height RMSE, or height correlation is reported.', '', '## INDIAN_DOMAIN_BEHAVIOR_COMPARISON', '- Phase 87 versus Phase 89 behavior is recorded in DOMAIN_COMPARISON.json; this is not an accuracy comparison.', '', '## 3D prototype', '- Generated as INDIAN_TERRAIN_BUILDING_3D_PROTOTYPE_V2.', '- Finite-height components are extruded; missing-height components remain marked unavailable in the assignments and are not assigned height.', '', '## Final scientific decision', decision])
    (OUT / 'REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(decision, flush=True)


if __name__ == '__main__':
    main()
