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
import rasterio
import torch
from rasterio.warp import reproject
from rasterio.enums import Resampling

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = ROOT / 'runs' / 'phase68_india_benchmark_ready' / 'ORIGINAL_DATA'
COMMON = ROOT / 'runs' / 'phase72_common_grid_forensics' / 'common_grid'
VISUALS = OUT / 'VISUALS'
PROTOTYPE = OUT / '3D_PROTOTYPE'
for directory in (VISUALS, PROTOTYPE):
    directory.mkdir(parents=True, exist_ok=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from depthwizard.config import TrainConfig
from depthwizard.depth.depth_anything import DepthAnythingV2
from depthwizard.models.building_conditioned_net import BuildingConditionedEstimator

MODEL_PATH = ROOT / 'runs' / 'phase24_moe' / 'seed_0' / 'model.pt'
MODEL_NAME = 'BuildingConditionedEstimator -> BuildingConditionedHeightNet -> SmallFusionUNet'
CROP_SIZE = 512
THRESHOLD = 0.5
DEPTH_CACHE = ROOT / 'data' / 'depth_cache'
REGIONS = {
    'uttarakhand': {'split': 'development', 'state': 'Uttarakhand', 'utm': 'EPSG:32644'},
    'himachal': {'split': 'validation', 'state': 'Himachal Pradesh', 'utm': 'EPSG:32643'},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def finite_stats(array: np.ndarray) -> dict:
    values = np.asarray(array)
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {
            'shape': list(values.shape),
            'dtype': str(values.dtype),
            'min': None,
            'max': None,
            'mean': None,
            'std': None,
            'finite_fraction': 0.0,
            'valid_pixels': 0,
        }
    return {
        'shape': list(values.shape),
        'dtype': str(values.dtype),
        'min': float(finite.min()) if finite.size else None,
        'max': float(finite.max()) if finite.size else None,
        'mean': float(finite.mean()) if finite.size else None,
        'std': float(finite.std()) if finite.size else None,
        'finite_fraction': float(np.isfinite(values).mean()),
        'valid_pixels': int(np.isfinite(values).sum()),
    }


def raster_metadata(path: Path) -> dict:
    with rasterio.open(path) as source:
        array = source.read()
        nodata = source.nodata
        if nodata is not None and not np.isfinite(nodata):
            nodata = None
        return {
            'path': str(path),
            'width': int(source.width),
            'height': int(source.height),
            'bands': int(source.count),
            'dtype': list(source.dtypes),
            'crs': str(source.crs),
            'resolution': [float(source.res[0]), float(source.res[1])],
            'bounds': [float(value) for value in source.bounds],
            'nodata': nodata,
            'stats': finite_stats(array),
        }


def read_rgb(region: str, window: rasterio.windows.Window):
    region_dir = DATA / region
    paths = [region_dir / f'{region}_B04.tif', region_dir / f'{region}_B03.tif', region_dir / f'{region}_B02.tif']
    arrays = []
    metadata = []
    for path in paths:
        with rasterio.open(path) as source:
            arrays.append(source.read(1, window=window))
            metadata.append(raster_metadata(path))
    rgb_u16 = np.stack(arrays, axis=-1)
    return rgb_u16, metadata


def read_aligned_dem(region: str, window: rasterio.windows.Window):
    aligned_path = COMMON / region / 'aligned_DEM.tif'
    if aligned_path.exists():
        with rasterio.open(aligned_path) as source:
            return source.read(1, window=window), raster_metadata(aligned_path)

    region_dir = DATA / region
    optical_path = region_dir / f'{region}_B04.tif'
    dem_path = region_dir / f'{region}_dem.tif'
    with rasterio.open(optical_path) as optical, rasterio.open(dem_path) as dem_source:
        destination = np.full((optical.height, optical.width), np.nan, dtype=np.float32)
        reproject(
            source=dem_source.read(1),
            destination=destination,
            src_transform=dem_source.transform,
            src_crs=dem_source.crs,
            src_nodata=dem_source.nodata,
            dst_transform=optical.transform,
            dst_crs=optical.crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
        return destination[window.row_off:window.row_off + window.height, window.col_off:window.col_off + window.width], raster_metadata(dem_path)


def choose_window(region: str) -> rasterio.windows.Window:
    path = DATA / region / f'{region}_B04.tif'
    with rasterio.open(path) as source:
        row = (source.height - CROP_SIZE) // 2
        col = (source.width - CROP_SIZE) // 2
        return rasterio.windows.Window(col, row, CROP_SIZE, CROP_SIZE)


def normalize_rgb(rgb_u16: np.ndarray) -> np.ndarray:
    # Sentinel-2 L2A reflectance is stored as scaled uint16; map 0..10000 to 0..255.
    return np.clip(rgb_u16.astype(np.float32) / 10000.0 * 255.0, 0.0, 255.0).astype(np.uint8)


def load_depth_model():
    return DepthAnythingV2(
        model_id='depth-anything/Depth-Anything-V2-Small-hf',
        cache_dir=str(DEPTH_CACHE),
        use_cache=True,
    )


def infer_region(region: str, estimator: BuildingConditionedEstimator, depth_model: DepthAnythingV2) -> dict:
    window = choose_window(region)
    rgb_u16, rgb_metadata = read_rgb(region, window)
    rgb = normalize_rgb(rgb_u16)
    dem, dem_metadata = read_aligned_dem(region, window)
    depth = depth_model.infer(rgb, key=f'phase87_{region}_center_{CROP_SIZE}', target_hw=(CROP_SIZE, CROP_SIZE)).astype(np.float32)

    sample = {'id': f'phase87_{region}', 'rgb': rgb, 'depth': depth, 'gt': dem, 'nodata': -999.0}
    res = estimator.cfg.train_res
    x = estimator._prep_x(sample, res)
    xt = torch.from_numpy(x[None]).float().to(estimator.device)
    depth_r = cv2.resize(depth, (res, res), interpolation=cv2.INTER_LINEAR)
    raw_depth = torch.from_numpy(depth_r[None]).float().to(estimator.device)
    with torch.no_grad():
        mask_logits, predictions, _, _, _ = estimator.model(xt, raw_depth, device=estimator.device)
    logits = mask_logits.squeeze(0).cpu().numpy().astype(np.float32)
    probabilities_256 = 1.0 / (1.0 + np.exp(-logits))
    probabilities = cv2.resize(probabilities_256, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)
    footprint = probabilities >= THRESHOLD

    number, labels, stats, _ = cv2.connectedComponentsWithStats(footprint.astype(np.uint8), connectivity=8)
    components = []
    for index in range(1, number):
        area_px = int(stats[index, cv2.CC_STAT_AREA])
        if area_px < 16:
            continue
        mask = labels == index
        predicted_height = float(predictions[len(components)][0].detach().cpu().item()) if len(components) < len(predictions) else None
        terrain_values = dem[mask & np.isfinite(dem)]
        if predicted_height is not None:
            roof_values = terrain_values + predicted_height if terrain_values.size else np.array([], dtype=np.float32)
        else:
            roof_values = np.array([], dtype=np.float32)
        components.append({
            'component_id': len(components) + 1,
            'area_pixels': area_px,
            'area_m2_at_10m_gsd': float(area_px * 100.0),
            'bbox_pixels': [int(stats[index, cv2.CC_STAT_LEFT]), int(stats[index, cv2.CC_STAT_TOP]), int(stats[index, cv2.CC_STAT_WIDTH]), int(stats[index, cv2.CC_STAT_HEIGHT])],
            'estimated_height_m': predicted_height,
            'terrain_at_footprint_m': finite_stats(terrain_values),
            'estimated_roof_elevation_m': finite_stats(roof_values),
            'negative_height': bool(predicted_height is not None and predicted_height < 0.0),
            'nan_height': bool(predicted_height is None or not np.isfinite(predicted_height)),
            'extreme_height_over_100m': bool(predicted_height is not None and predicted_height > 100.0),
        })

    valid_dem = dem[np.isfinite(dem)]
    heights = np.array([item['estimated_height_m'] for item in components if item['estimated_height_m'] is not None], dtype=np.float32)
    foreground_pixels = int(footprint.sum())
    largest = max((item['area_pixels'] for item in components), default=0)
    flags = {
        'ALL_IMAGE_FOREGROUND': bool(foreground_pixels == footprint.size),
        'NO_BUILDINGS_DETECTED': bool(len(components) == 0),
        'ONE_MASSIVE_COMPONENT': bool(len(components) == 1 and largest / max(foreground_pixels, 1) > 0.90),
        'HEIGHT_EXPLOSION': bool(heights.size and np.nanmax(heights) > 100.0),
        'NEGATIVE_HEIGHTS': bool(heights.size and np.nanmin(heights) < 0.0),
        'NAN_HEIGHTS': bool(any(item['nan_height'] for item in components)),
        'NEAR_ZERO_HEIGHTS': bool(heights.size and float(np.nanmedian(heights)) < 1.0),
    }
    flags['VALID_STRUCTURAL_OUTPUT'] = bool(len(components) > 0 and not any(flags[key] for key in flags if key != 'VALID_STRUCTURAL_OUTPUT'))

    result = {
        'region': region,
        'state': REGIONS[region]['state'],
        'split': REGIONS[region]['split'],
        'crop': {'row_off': int(window.row_off), 'col_off': int(window.col_off), 'height': int(window.height), 'width': int(window.width)},
        'optical_input': {'composite_order': 'RGB = Sentinel-2 B04/B03/B02', 'radiometry': 'uint16 L2A scaled reflectance mapped 0..10000 to uint8 0..255 for existing model input', 'bands': rgb_metadata, 'composite_stats': finite_stats(rgb)},
        'terrain_base': {'source': 'Phase 72 aligned DEM', 'usage': 'terrain base only; not passed to building model', 'metadata': dem_metadata, 'crop_stats': finite_stats(valid_dem)},
        'depth_prior': {'model_id': 'depth-anything/Depth-Anything-V2-Small-hf', 'output_stats': finite_stats(depth)},
        'building_model': {'name': MODEL_NAME, 'checkpoint': str(MODEL_PATH), 'checkpoint_sha256': sha256(MODEL_PATH), 'input_channels': 4, 'input_description': 'RGB + raw relative depth normalization', 'output_channels': '16 feature channels + 1 footprint logit', 'threshold': THRESHOLD, 'train_res': int(res), 'height_output': 'object-level predicted above-ground height from existing model head; model output, not Indian ground truth'},
        'raw_output': {'logit_stats': finite_stats(logits), 'probability_stats': finite_stats(probabilities), 'foreground_pixels': foreground_pixels, 'foreground_percentage': float(100.0 * foreground_pixels / footprint.size)},
        'connected_components': {'all_components_in_thresholded_mask': int(number - 1), 'components_after_area_filter_16_pixels': len(components), 'largest_component_pixels': largest, 'largest_component_ratio_of_foreground': float(largest / max(foreground_pixels, 1)), 'detected_buildings_after_postprocessing': len(components)},
        'components': components,
        'height_output_summary_model_output_only': finite_stats(heights),
        'failure_flags': flags,
        'ground_truth': {'INDIAN_BUILDING_GROUND_TRUTH_AVAILABLE': 'NO', 'footprint_metrics': None, 'height_metrics': None, 'reason': 'Phase 72 contains optical imagery and terrain DEM only; no verified Indian footprint or building-height reference was found in the repository.'},
        '_arrays': {'rgb': rgb, 'probabilities': probabilities, 'footprint': footprint, 'heights': heights, 'dem': dem},
    }
    return result


def save_visuals(result: dict):
    region = result['region']
    arrays = result['_arrays']
    rgb = arrays['rgb']
    footprint = arrays['footprint']
    dem = arrays['dem']
    height_map = np.zeros_like(dem, dtype=np.float32)
    components = result['components']
    number, labels, stats, _ = cv2.connectedComponentsWithStats(footprint.astype(np.uint8), connectivity=8)
    for item in components:
        label = item['component_id']
        value = item['estimated_height_m']
        if value is not None and label < number:
            height_map[labels == label] = value
    dsm_candidate = dem + height_map
    figure, axes = plt.subplots(1, 5, figsize=(22, 5))
    axes[0].imshow(rgb)
    axes[0].set_title(f'{region} optical RGB')
    axes[1].imshow(footprint, cmap='gray')
    axes[1].set_title('predicted footprint')
    axes[2].imshow(np.ma.masked_where(~footprint, height_map), cmap='viridis')
    axes[2].set_title('predicted height (m)')
    axes[3].imshow(dem, cmap='terrain')
    axes[3].set_title('real DEM terrain (m)')
    axes[4].imshow(dsm_candidate, cmap='terrain')
    axes[4].set_title('DSM_CANDIDATE (m)')
    for axis in axes:
        axis.axis('off')
    figure.tight_layout()
    figure.savefig(VISUALS / f'{region}_phase87_audit.png', dpi=160)
    plt.close(figure)


def strip_arrays(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != '_arrays'}


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f'Missing frozen building checkpoint: {MODEL_PATH}')
    config = TrainConfig(arch='unet3', target_transform='none', epochs=1, batch_size=8, lr=1e-3, amp=True)
    estimator = BuildingConditionedEstimator(config, nodata=-999.0, seed=0, device='cpu')
    estimator.model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=True))
    estimator.model.eval()
    depth_model = load_depth_model()

    results = {}
    for region in REGIONS:
        print(f'Running Phase 87 inference for {region}...', flush=True)
        results[region] = infer_region(region, estimator, depth_model)
        save_visuals(results[region])

    clean = {region: strip_arrays(value) for region, value in results.items()}
    domain_rows = []
    for region, value in clean.items():
        raw = value['raw_output']
        comps = value['connected_components']
        heights = value['height_output_summary_model_output_only']
        domain_rows.append({'region': region, 'split': value['split'], 'foreground_percentage': raw['foreground_percentage'], 'components_after_area_filter': comps['components_after_area_filter_16_pixels'], 'largest_component_ratio': comps['largest_component_ratio_of_foreground'], 'height_mean_m': heights['mean'], 'height_std_m': heights['std'], 'height_max_m': heights['max']})

    final_decision = 'INDIAN_BUILDING_OUTPUT_AVAILABLE_BUT_UNVALIDATED'
    if all(value['failure_flags']['NO_BUILDINGS_DETECTED'] or value['failure_flags']['ALL_IMAGE_FOREGROUND'] or value['failure_flags']['ONE_MASSIVE_COMPONENT'] for value in clean.values()):
        final_decision = 'INDIAN_BUILDING_MODEL_DOMAIN_FAILURE'
    elif all(value['failure_flags']['NO_BUILDINGS_DETECTED'] for value in clean.values()):
        final_decision = 'INDIAN_BUILDING_MODEL_DOMAIN_FAILURE'

    payload = {
        'phase': 'PHASE_87',
        'objective': 'Real Indian mountainous building-structure baseline using real DEM only as terrain base',
        'TERRAIN_SOURCE': 'REAL_GEOREFERENCED_DEM',
        'BUILDING_SOURCE': 'SINGLE_VIEW_OPTICAL_MODEL',
        'sikkim_evaluated': False,
        'sikkim_lock_reason': 'No verified Indian building footprint or height ground truth is present.',
        'model_reuse': {'model_name': MODEL_NAME, 'checkpoint': str(MODEL_PATH), 'checkpoint_sha256': sha256(MODEL_PATH), 'retrained': False},
        'regions': clean,
        'final_scientific_decision': final_decision,
        'limitations': ['No Indian building footprint ground truth.', 'No Indian building-height ground truth.', 'Predicted heights are model outputs only.', 'DSM_CANDIDATE is not a validated DSM.', 'One deterministic crop per region is a baseline, not regional generalization evidence.'],
    }
    (OUT / 'RESULTS.json').write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    (OUT / 'BUILDING_INFERENCE.json').write_text(json.dumps({'model': payload['model_reuse'], 'regions': clean}, indent=2, allow_nan=False), encoding='utf-8')
    (OUT / 'DOMAIN_COMPARISON.json').write_text(json.dumps({'comparison_scope': 'Indian regions only; no compatible non-Indian rerun performed', 'rows': domain_rows, 'non_indian_comparison': 'Existing repository evidence is not directly comparable because it uses different imagery, ground truth, resolution, and scene regime.'}, indent=2, allow_nan=False), encoding='utf-8')

    report = [
        '# Phase 87: Indian mountain building height baseline', '',
        '## 1. Objective',
        'Test whether the existing building-oriented optical model produces structurally usable building outputs on real Indian mountainous optical imagery while the real georeferenced DEM supplies the terrain base.', '',
        'TERRAIN_SOURCE = REAL_GEOREFERENCED_DEM',
        'BUILDING_SOURCE = SINGLE_VIEW_OPTICAL_MODEL', '',
        '## 2. Indian data used',
        '- Development: Uttarakhand.',
        '- Validation: Himachal Pradesh.',
        '- Sikkim: not evaluated; locked because verified Indian building ground truth is unavailable.',
        '- Optical: real Sentinel-2 B04/B03/B02 composites from the frozen Phase 68 source.',
        '- Crop policy: deterministic center 512 x 512 crop per region.', '',
        '## 3. Existing building model used',
        f'- Model: `{MODEL_NAME}`.',
        f'- Checkpoint: `{MODEL_PATH}`.',
        '- Reused without retraining or threshold tuning.',
        '- Input: 4 channels, RGB plus normalized relative depth.',
        '- Footprint threshold: inherited sigmoid probability threshold 0.5.',
        '- Height output: existing object-level model output; not ground truth.', '',
        '## 4. Terrain DEM information',
        '- Phase 72 aligned DEM used only as the metric terrain base.',
        '- DEM was not passed into the building detector.', '',
        '## 5. Building inference results',
    ]
    for region, value in clean.items():
        report.extend([
            f"### {value['state']}",
            f"- Foreground: {value['raw_output']['foreground_pixels']} pixels ({value['raw_output']['foreground_percentage']:.4f}%).",
            f"- Probability range: {value['raw_output']['probability_stats']['min']} to {value['raw_output']['probability_stats']['max']}.",
            f"- Components after area filter: {value['connected_components']['components_after_area_filter_16_pixels']}.",
            f"- Largest component ratio: {value['connected_components']['largest_component_ratio_of_foreground']:.4f}.",
            f"- Detected buildings after post-processing: {value['connected_components']['detected_buildings_after_postprocessing']}.",
        ])
    report.extend(['', '## 6. Height inference results', '- Heights below are MODEL OUTPUT, not accuracy against Indian building-height truth.'])
    for region, value in clean.items():
        summary = value['height_output_summary_model_output_only']
        report.extend([f"- {value['state']}: min={summary['min']}, max={summary['max']}, mean={summary['mean']}, std={summary['std']} m."])
    report.extend([
        '', '## 7. Ground-truth availability',
        '- INDIAN_BUILDING_GROUND_TRUTH_AVAILABLE = NO.',
        '- No IoU, Dice, precision, recall, height MAE, height RMSE, or height correlation is reported.',
        '', '## 8. Domain-shift observations',
        '- Indian-only output distributions are recorded in DOMAIN_COMPARISON.json.',
        '- Existing non-Indian results are not numerically comparable because the imagery, labels, resolution, and scene regime differ.',
        '', '## 9. Failure flags',
    ])
    for region, value in clean.items():
        active = [key for key, active in value['failure_flags'].items() if active]
        report.append(f"- {value['state']}: {', '.join(active) if active else 'none'}.")
    report.extend([
        '', '## 10. 3D prototype status',
        '- No 3D renderer was run in this forensic baseline.',
        '- The diagnostic views show the conceptual DSM_CANDIDATE only; it is not validated 3D reconstruction.',
        '', '## 11. Limitations',
        '- One crop per region cannot establish regional generalization.',
        '- No verified Indian building labels exist in the repository.',
        '- Optical imagery is 10 m resolution, so small buildings may be unresolved.',
        '- Predicted height distributions may reflect domain shift and must not be interpreted as measured heights.',
        '', '## 12. Final scientific decision',
        final_decision,
    ])
    (OUT / 'REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(final_decision, flush=True)


if __name__ == '__main__':
    main()
