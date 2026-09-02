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

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
VISUALS = OUT / 'VISUALS'
DATA = ROOT / 'runs' / 'phase68_india_benchmark_ready' / 'ORIGINAL_DATA'
MODEL_PATH = ROOT / 'runs' / 'phase24_moe' / 'seed_0' / 'model.pt'
PHASE87_SCRIPT = ROOT / 'runs' / 'phase87_indian_building_baseline' / 'phase87_run.py'
CROP_SIZE = 512
THRESHOLD = 0.5
REGIONS = ('uttarakhand', 'himachal')

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from depthwizard.config import TrainConfig
from depthwizard.depth.depth_anything import DepthAnythingV2
from depthwizard.models.building_conditioned_net import BuildingConditionedEstimator
from runs.phase87_indian_building_baseline.phase87_run import (
    choose_window,
    normalize_rgb,
    read_aligned_dem,
    read_rgb,
)


def stats(value) -> dict:
    array = np.asarray(value)
    finite = array[np.isfinite(array)] if np.issubdtype(array.dtype, np.number) else np.array([], dtype=np.float32)
    if not finite.size:
        return {'shape': list(array.shape), 'dtype': str(array.dtype), 'min': None, 'max': None, 'mean': None, 'std': None, 'finite_count': 0}
    return {'shape': list(array.shape), 'dtype': str(array.dtype), 'min': float(finite.min()), 'max': float(finite.max()), 'mean': float(finite.mean()), 'std': float(finite.std()), 'finite_count': int(finite.size)}


def height_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {
            'shape': list(values.shape),
            'dtype': str(values.dtype),
            'min': None,
            'max': None,
            'mean': None,
            'median': None,
            'std': None,
            'p95': None,
            'p99': None,
            'finite_count': 0,
            'nan_count': int(np.isnan(values).sum()),
            'nan_fraction': 0.0,
            'positive_count': 0,
            'zero_count': 0,
            'negative_count': 0,
        }
    return {
        'shape': list(values.shape),
        'dtype': str(values.dtype),
        'min': float(finite.min()) if finite.size else None,
        'max': float(finite.max()) if finite.size else None,
        'mean': float(finite.mean()) if finite.size else None,
        'median': float(np.median(finite)) if finite.size else None,
        'std': float(finite.std()) if finite.size else None,
        'p95': float(np.percentile(finite, 95)) if finite.size else None,
        'p99': float(np.percentile(finite, 99)) if finite.size else None,
        'finite_count': int(finite.size),
        'nan_count': int(np.isnan(values).sum()),
        'nan_fraction': float(np.isnan(values).mean()),
        'positive_count': int((finite > 0).sum()),
        'zero_count': int((finite == 0).sum()),
        'negative_count': int((finite < 0).sum()),
    }


def component_stats(mask: np.ndarray, labels: np.ndarray, component_id: int, dem: np.ndarray, height: float | None, status: str) -> dict:
    pixels = labels == component_id
    area = int(pixels.sum())
    terrain = dem[pixels & np.isfinite(dem)]
    roof = terrain + height if height is not None and terrain.size else np.array([], dtype=np.float32)
    return {
        'component_id': component_id,
        'area_pixels': area,
        'area_m2_at_10m_gsd': float(area * 100.0),
        'height_m': height,
        'terrain_m': height_stats(terrain),
        'roof_elevation_m': height_stats(roof),
        'status': status,
        'NEGATIVE_HEIGHT': bool(height is not None and height < 0),
        'ZERO_HEIGHT': bool(height is not None and height == 0),
        'EXTREME_HEIGHT': bool(height is not None and height > 100),
        'NAN_HEIGHT': bool(height is None),
        'MISSING_HEIGHT': bool(height is None),
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
    depth_r = cv2.resize(depth, (res, res), interpolation=cv2.INTER_LINEAR)
    raw_depth = torch.from_numpy(depth_r[None]).float()
    with torch.no_grad():
        raw_model_outputs = estimator.model(xt, raw_depth, device='cpu')
    mask_logits, predictions, _, _, _ = raw_model_outputs
    logits = mask_logits.squeeze(0).numpy().astype(np.float32)
    probs_256 = torch.sigmoid(mask_logits).squeeze(0).numpy().astype(np.float32)
    probs = cv2.resize(probs_256, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)
    footprint = probs >= THRESHOLD
    n_labels, labels, cc_stats, _ = cv2.connectedComponentsWithStats(footprint.astype(np.uint8), connectivity=8)

    label_components = []
    for component_id in range(1, n_labels):
        area = int(cc_stats[component_id, cv2.CC_STAT_AREA])
        if area >= 16:
            label_components.append((component_id, area))
    sorted_components = sorted(label_components, key=lambda item: item[1], reverse=True)
    model_components = sorted_components[:25]
    height_values = np.array([float(item[0].detach().cpu().item()) for item in predictions], dtype=np.float64)

    # This reproduces Phase 87's record construction: label-order components
    # receive prediction list entries by position, despite model-side area sorting.
    phase87_records = []
    for position, (component_id, _) in enumerate(label_components):
        height = float(height_values[position]) if position < height_values.size else None
        status = 'FINITE_ASSIGNED' if height is not None else 'NAN_MISSING_FROM_CAPACITY'
        phase87_records.append(component_stats(footprint, labels, component_id, dem, height, status))

    expected_by_component = {component_id: height_values[position] for position, (component_id, _) in enumerate(model_components) if position < height_values.size}
    mapping_checks = []
    for position, (component_id, area) in enumerate(label_components):
        expected = expected_by_component.get(component_id)
        assigned = phase87_records[position]['height_m']
        mapping_checks.append({
            'component_id': component_id,
            'label_order_position': position,
            'area_pixels': area,
            'model_sorted_rank': next((rank for rank, (candidate, _) in enumerate(model_components) if candidate == component_id), None),
            'phase87_assigned_height_m': assigned,
            'model_component_height_m': float(expected) if expected is not None else None,
            'finite_assignment_matches_model_component': bool(expected is not None and assigned is not None and np.isclose(assigned, expected)),
            'missing_due_to_capacity': bool(expected is None),
        })

    finite_records = [record for record in phase87_records if record['height_m'] is not None]
    heights = np.array([record['height_m'] if record['height_m'] is not None else np.nan for record in phase87_records], dtype=np.float64)
    mapped_finite_mismatch = sum(1 for item in mapping_checks if item['model_component_height_m'] is not None and not item['finite_assignment_matches_model_component'])
    first_nan_stage = 'final Phase 87 per-component building-height records' if np.isnan(heights).any() else 'not observed'

    result = {
        'region': region,
        'crop': {'row_off': int(window.row_off), 'col_off': int(window.col_off), 'height': CROP_SIZE, 'width': CROP_SIZE},
        'input_stage': {'rgb_u16': stats(rgb_u16), 'rgb_uint8_after_existing_preprocessing': stats(rgb), 'depth_raw': stats(depth), 'model_input_x': stats(x), 'raw_depth_tensor': stats(raw_depth.numpy())},
        'model_output_stage': {
            'mask_logits': stats(logits),
            'probabilities_256': stats(probs_256),
            'probabilities_512': stats(probs),
            'height_prediction_tensor': {'shape': list(height_values.shape), 'dtype': str(height_values.dtype), 'representation': 'one scalar per model-selected connected component, area-sorted, capped at 25; not one value per pixel'},
            'height_predictions': height_stats(height_values),
            'model_internal_component_order': [{'component_id': component_id, 'area_pixels': area, 'rank': rank} for rank, (component_id, area) in enumerate(model_components)],
        },
        'postprocessing_stage': {
            'threshold': THRESHOLD,
            'thresholded_footprint': stats(footprint.astype(np.uint8)),
            'connected_component_labels': stats(labels),
            'raw_connected_components_excluding_background': int(n_labels - 1),
            'components_after_area_filter_16_pixels': len(label_components),
            'height_capacity': 25,
            'components_receiving_height_predictions': int(min(len(label_components), 25)),
            'components_without_height_predictions': int(max(len(label_components) - 25, 0)),
            'nan_first_appears_at': first_nan_stage,
        },
        'component_capacity_audit': {
            'total_components': len(label_components),
            'height_capacity': 25,
            'predicted_heights': int(np.isfinite(heights).sum()),
            'missing_heights': int(np.isnan(heights).sum()),
        },
        'phase87_height_records': phase87_records,
        'height_value_audit': height_stats(heights),
        'height_footprint_alignment': {
            'model_sorts_by_area_descending': True,
            'phase87_records_iterate_connected_component_label_order': True,
            'finite_mapping_mismatches': mapped_finite_mismatch,
            'mapping_bug_observed': bool(mapped_finite_mismatch > 0),
            'checks': mapping_checks,
        },
        'terrain_integration': {
            'terrain_source': 'Phase 72 aligned DEM',
            'terrain_units': 'meters',
            'height_units': 'meters as defined by existing model output convention; no re-interpretation of relative depth',
            'roof_formula': 'terrain_base + predicted_height',
            'terrain_crop': height_stats(dem[np.isfinite(dem)]),
            'finite_height_component_roof_summary': height_stats(np.array([record['roof_elevation_m']['mean'] for record in finite_records if record['roof_elevation_m']['mean'] is not None], dtype=np.float64)),
        },
        'component_summary': {
            'all_records': phase87_records,
            'finite_records': finite_records,
            'area_stats_all_components': height_stats(np.array([item[1] for item in label_components], dtype=np.float64)),
            'finite_height_component_area_stats': height_stats(np.array([item['area_pixels'] for item in finite_records], dtype=np.float64)),
            'finite_height_stats': height_stats(heights),
            'finite_terrain_stats': height_stats(np.array([record['terrain_m']['mean'] for record in finite_records if record['terrain_m']['mean'] is not None], dtype=np.float64)),
            'finite_roof_stats': height_stats(np.array([record['roof_elevation_m']['mean'] for record in finite_records if record['roof_elevation_m']['mean'] is not None], dtype=np.float64)),
        },
        'ground_truth': {'INDIAN_BUILDING_GROUND_TRUTH_AVAILABLE': 'NO', 'accuracy_metrics': None},
        '_arrays': {'rgb': rgb, 'footprint': footprint, 'labels': labels, 'heights': heights, 'dem': dem},
    }
    return result


def save_visual(result: dict):
    arrays = result['_arrays']
    labels = arrays['labels']
    heights = arrays['heights']
    assignment = np.full(labels.shape, np.nan, dtype=np.float32)
    for component_id in range(1, labels.max() + 1):
        if component_id - 1 < heights.size and np.isfinite(heights[component_id - 1]):
            assignment[labels == component_id] = heights[component_id - 1]
    figure, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(arrays['rgb']); axes[0].set_title(f"{result['region']} RGB")
    axes[1].imshow(arrays['footprint'], cmap='gray'); axes[1].set_title('footprint')
    axes[2].imshow(labels, cmap='nipy_spectral'); axes[2].set_title('component IDs')
    axes[3].imshow(assignment, cmap='viridis'); axes[3].set_title('Phase 87 height assignment (m)')
    for axis in axes: axis.axis('off')
    figure.tight_layout()
    figure.savefig(VISUALS / f"{result['region']}_height_alignment_audit.png", dpi=170)
    plt.close(figure)


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items() if key != '_arrays'}
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value) if np.isfinite(value) else None
    return value


def main():
    config = TrainConfig(arch='unet3', target_transform='none', epochs=1, batch_size=8, lr=1e-3, amp=True)
    estimator = BuildingConditionedEstimator(config, nodata=-999.0, seed=0, device='cpu')
    estimator.model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=True))
    estimator.model.eval()
    depth_model = DepthAnythingV2(model_id='depth-anything/Depth-Anything-V2-Small-hf', cache_dir=str(ROOT / 'data' / 'depth_cache'), use_cache=True)
    results = {}
    for region in REGIONS:
        print(f'Auditing {region}...', flush=True)
        results[region] = run_region(region, estimator, depth_model)
        save_visual(results[region])

    clean_results = {region: clean(value) for region, value in results.items()}
    parameter_count = int(sum(parameter.numel() for parameter in estimator.model.parameters()))
    trainable_count = int(sum(parameter.numel() for parameter in estimator.model.parameters() if parameter.requires_grad))
    mapping_bug = any(value['height_footprint_alignment']['mapping_bug_observed'] for value in clean_results.values())
    capacity_missing = any(value['component_capacity_audit']['missing_heights'] > 0 for value in clean_results.values())
    final_label = 'HEIGHT_OUTPUT_MAPPING_BUG' if mapping_bug else ('HEIGHT_CAPACITY_LIMITATION' if capacity_missing else 'HEIGHT_VALUES_ARE_AVAILABLE_BUT_UNVALIDATED')

    payload = {
        'phase': 'PHASE_88',
        'TERRAIN_SOURCE': 'REAL GEOREFERENCED DEM',
        'BUILDING_SOURCE': 'EXISTING SINGLE-VIEW BUILDING MODEL',
        'checkpoint': str(MODEL_PATH),
        'checkpoint_sha256': __import__('hashlib').sha256(MODEL_PATH.read_bytes()).hexdigest(),
        'phase87_pipeline_source': str(PHASE87_SCRIPT),
        'model_architecture': {
            'model_class': 'BuildingConditionedEstimator / BuildingConditionedHeightNet / SmallFusionUNet',
            'parameter_count': parameter_count,
            'trainable_parameter_count': trainable_count,
            'input_channels': 4,
            'output_channels': 'C_feat=16 feature channels + 1 footprint logit',
            'heads_and_activations': ['backbone dense feature channels and footprint logit', 'sigmoid converts footprint logit to probability', 'gate MLP uses ReLU and softmax over 3 experts', 'three expert MLPs use ReLU then scalar log-residual', 'height uses exp(clamped residual)', 'high-rise base uses softplus(alpha), softplus(beta)'],
            'semantic_building_segmentation': True,
            'instance_separation': 'connected components only; no learned instance head',
            'building_height_regression': True,
            'object_level_height_prediction': True,
        },
        'regions': clean_results,
        'INDIAN_DOMAIN_BEHAVIOR_COMPARISON': {
            region: {
                'foreground_percentage': value['postprocessing_stage']['thresholded_footprint']['finite_count'] and float(100.0 * np.mean(results[region]['_arrays']['footprint'])) or 0.0,
                'component_count': value['component_capacity_audit']['total_components'],
                'finite_height_count': value['component_capacity_audit']['predicted_heights'],
                'nan_height_count': value['component_capacity_audit']['missing_heights'],
                'height_distribution': value['height_value_audit'],
            } for region, value in clean_results.items()
        },
        'no_ground_truth_claim': True,
        'final_diagnosis': final_label,
    }
    (OUT / 'RESULTS.json').write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    (OUT / 'HEIGHT_PIPELINE.json').write_text(json.dumps({'architecture': payload['model_architecture'], 'checkpoint': payload['checkpoint'], 'regions': {region: value['model_output_stage'] for region, value in clean_results.items()}}, indent=2, allow_nan=False), encoding='utf-8')
    (OUT / 'COMPONENT_AUDIT.json').write_text(json.dumps({region: value['component_capacity_audit'] | {'alignment': value['height_footprint_alignment'], 'records': value['phase87_height_records']} for region, value in clean_results.items()}, indent=2, allow_nan=False), encoding='utf-8')
    (OUT / 'HEIGHT_DISTRIBUTION.json').write_text(json.dumps({region: value['height_value_audit'] | {'finite_component_records': value['component_summary']['finite_records']} for region, value in clean_results.items()}, indent=2, allow_nan=False), encoding='utf-8')
    (OUT / 'DOMAIN_COMPARISON.json').write_text(json.dumps(payload['INDIAN_DOMAIN_BEHAVIOR_COMPARISON'], indent=2, allow_nan=False), encoding='utf-8')

    lines = [
        '# Phase 88: Forensic audit of Indian building height pipeline', '',
        '## Objective',
        'Forensic-only tracing of the exact Phase 87 model, tensor path, connected-component path, height capacity, and final record mapping.', '',
        'TERRAIN_SOURCE = REAL GEOREFERENCED DEM',
        'BUILDING_SOURCE = EXISTING SINGLE-VIEW BUILDING MODEL', '',
        '## 1. Architecture',
        f"- Model class: {payload['model_architecture']['model_class']}.",
        f"- Parameters: {parameter_count}; trainable: {trainable_count}.",
        '- Input: 4 channels, RGB plus relative-depth channel.',
        '- Output: 16 feature channels plus one footprint logit.',
        '- The model has semantic footprint segmentation and object-level height regression, but instance separation is connected-components post-processing, not a learned instance head.', '',
        '## 2. Height output path',
        '- Input -> `_prep_x` -> model forward -> footprint logit/probability -> threshold -> connected components -> model-selected object predictions -> Phase 87 per-component records.',
        f'- NAN FIRST APPEARS AT = {clean_results[REGIONS[0]]["postprocessing_stage"]["nan_first_appears_at"]}.',
        '- The model prediction list itself contains finite object heights; missing values enter when Phase 87 assigns that shorter list to the longer label-order component record list.', '',
        '## 3. Component capacity audit',
    ]
    for region, value in clean_results.items():
        audit = value['component_capacity_audit']
        lines.append(f"- {region.upper()}: total components={audit['total_components']}; height-capacity={audit['height_capacity']}; predicted heights={audit['predicted_heights']}; missing heights={audit['missing_heights']}.")
    lines.extend(['', '## 4. Height representation', '- The actual height prediction tensor is one scalar per selected connected component, represented as a 1D tensor/list of length up to 25.', '- It is not one value per pixel and not a dense height map.', '', '## 5. Height value audit'])
    for region, value in clean_results.items():
        lines.append(f"- {region.upper()}: {value['height_value_audit']}.")
    lines.extend(['', '## 6. Building-height / footprint consistency', '- Per-component flags, areas, terrain elevations, and roof elevations are preserved in COMPONENT_AUDIT.json.', '- Roof elevation is computed as terrain base in meters plus predicted height in meters; no relative-depth reinterpretation is performed.', '', '## 7. Height-footprint alignment'])
    for region, value in clean_results.items():
        align = value['height_footprint_alignment']
        lines.append(f"- {region.upper()}: finite mapping mismatches={align['finite_mapping_mismatches']}; mapping_bug_observed={align['mapping_bug_observed']}.")
    lines.extend(['- The model sorts components by descending area before producing predictions.', '- Phase 87 iterates components in connected-component label order before assigning predictions by position.', '- Therefore positional correspondence is invalid whenever those orders differ.', '', '## 8. Terrain integration', '- Terrain is the Phase 72 aligned DEM in meters.', '- Height outputs are existing model outputs; the audit does not claim independent Indian height accuracy.', '- DSM_CANDIDATE/roof elevations are not validated against Indian building truth.', '', '## 9. Ground truth', '- INDIAN_BUILDING_GROUND_TRUTH_AVAILABLE = NO.', '- IoU, Dice, precision, recall, height MAE, height RMSE, and height correlation are not calculated.', '', '## 10. INDIAN_DOMAIN_BEHAVIOR_COMPARISON', '- Uttarakhand and Himachal distributions are recorded in DOMAIN_COMPARISON.json. These are behavior comparisons, not accuracy or generalization claims.', '', '## 11. Final diagnosis', final_label])
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(final_label, flush=True)


if __name__ == '__main__':
    main()
