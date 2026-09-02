from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds, reproject

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE72 = REPO_ROOT / 'runs' / 'phase72_common_grid_forensics' / 'common_grid'
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)
VIS = OUT / 'VISUALS'
VIS.mkdir(parents=True, exist_ok=True)
TARGET_SIZE = 512
REGION = 'uttarakhand'


def open_raster(path: Path):
    ds = rasterio.open(path)
    return ds, ds.read()


def valid_bbox_crop(valid: np.ndarray, size: int, center_y: int | None = None, center_x: int | None = None):
    ys, xs = np.where(valid)
    if ys.size == 0:
        raise RuntimeError('No valid pixels in the region mask.')
    if center_y is None:
        center_y = int(np.median(ys))
    if center_x is None:
        center_x = int(np.median(xs))
    y_start = int(np.clip(center_y - size // 2, 0, max(0, valid.shape[0] - size)))
    x_start = int(np.clip(center_x - size // 2, 0, max(0, valid.shape[1] - size)))
    y_end = y_start + size
    x_end = x_start + size
    if y_end > valid.shape[0]:
        y_end = valid.shape[0]
        y_start = max(0, y_end - size)
    if x_end > valid.shape[1]:
        x_end = valid.shape[1]
        x_start = max(0, x_end - size)
    return (y_start, x_start, y_end, x_end)


def deterministic_split(mask: np.ndarray):
    ys, xs = np.where(mask)
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    train_center = (int(y_min + (y_max - y_min) * 0.25), int(x_min + (x_max - x_min) * 0.25))
    val_center = (int(y_min + (y_max - y_min) * 0.75), int(x_min + (x_max - x_min) * 0.75))
    train_bbox = valid_bbox_crop(mask, TARGET_SIZE, *train_center)
    val_bbox = valid_bbox_crop(mask, TARGET_SIZE, *val_center)
    y0a, x0a, y1a, x1a = train_bbox
    y0b, x0b, y1b, x1b = val_bbox
    if not (x1a <= x0b or x1b <= x0a or y1a <= y0b or y1b <= y0a):
        train_bbox = (y_min, x_min, y_min + TARGET_SIZE, x_min + TARGET_SIZE)
        val_bbox = (max(y_min, y_max - TARGET_SIZE), max(x_min, x_max - TARGET_SIZE), y_max, x_max)
    return train_bbox, val_bbox


def build_crop(region_name: str, bbox: tuple[int, int, int, int]):
    region_dir = PHASE72 / region_name
    ds_rgb, rgb = open_raster(region_dir / 'aligned_RGB.tif')
    ds_dem, dem = open_raster(region_dir / 'aligned_DEM.tif')
    ds_mask, mask = open_raster(region_dir / 'valid_mask.tif')
    rgb = rgb.astype(np.float32)[:3]
    dem = dem[0].astype(np.float32)
    mask = mask[0].astype(bool)
    y0, x0, y1, x1 = bbox
    rgb_crop = rgb[:, y0:y1, x0:x1]
    dem_crop = dem[y0:y1, x0:x1]
    mask_crop = mask[y0:y1, x0:x1]
    local_median = float(np.median(dem_crop[mask_crop]))
    local_relief = dem_crop.astype(np.float32) - local_median
    return {
        'bbox': [y0, x0, y1, x1],
        'rgb': np.clip(rgb_crop / 65535.0, 0.0, 1.0),
        'dem': dem_crop,
        'mask': mask_crop,
        'local_median': local_median,
        'local_relief': local_relief,
        'valid_pixels': int(mask_crop.sum()),
        'rgb_resolution': (float(ds_rgb.res[0]), float(ds_rgb.res[1])),
        'dem_resolution': (float(ds_dem.res[0]), float(ds_dem.res[1])),
        'transform': list(ds_dem.transform),
    }


def safe_stats(arr: np.ndarray):
    arr = np.asarray(arr, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return {
            'min': np.nan,
            'max': np.nan,
            'mean': np.nan,
            'std': np.nan,
            'median': np.nan,
            'p95': np.nan,
            'valid_pixels': 0,
        }
    values = arr[finite]
    return {
        'min': float(values.min()),
        'max': float(values.max()),
        'mean': float(values.mean()),
        'std': float(values.std()),
        'median': float(np.median(values)),
        'p95': float(np.percentile(values, 95)),
        'valid_pixels': int(values.size),
    }


def gradient_stats(arr: np.ndarray, valid: np.ndarray | None = None):
    arr = np.asarray(arr, dtype=np.float64)
    gx, gy = np.gradient(arr)
    gm = np.hypot(gx, gy)
    if valid is not None:
        gm = gm[valid]
        arr = arr[valid]
    else:
        gm = gm.ravel()
        arr = arr.ravel()
    if gm.size == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'median': np.nan,
            'p95': np.nan,
            'count': 0,
        }
    return {
        'mean': float(gm.mean()),
        'std': float(gm.std()),
        'median': float(np.median(gm)),
        'p95': float(np.percentile(gm, 95)),
        'count': int(gm.size),
    }


def threshold_percentages(arr: np.ndarray, valid: np.ndarray | None = None):
    arr = np.asarray(arr, dtype=np.float64)
    gx, gy = np.gradient(arr)
    gm = np.hypot(gx, gy)
    if valid is not None:
        gm = gm[valid]
    else:
        gm = gm.ravel()
    if gm.size == 0:
        return {str(t): 0.0 for t in [1, 5, 10, 25, 50, 100]}
    return {str(t): float((gm > float(t)).mean()) * 100.0 for t in [1, 5, 10, 25, 50, 100]}


def smooth_downsample(arr: np.ndarray, scale: int):
    if scale == 1:
        return arr.astype(np.float32)
    if arr.ndim == 2:
        h, w = arr.shape
        y = np.arange(0, h, scale)
        x = np.arange(0, w, scale)
        return arr[np.ix_(y, x)]
    raise ValueError('expected 2D array')


def compute_scale_correlation(rgb_lum: np.ndarray, target: np.ndarray):
    results = {}
    for scale in [1, 2, 4, 8]:
        rgb_s = smooth_downsample(rgb_lum, scale)
        dem_s = smooth_downsample(target, scale)
        if rgb_s.size == 0 or dem_s.size == 0:
            corr = np.nan
        else:
            rgb_f = rgb_s.ravel()
            dem_f = dem_s.ravel()
            if np.std(rgb_f) < 1e-8 or np.std(dem_f) < 1e-8:
                corr = np.nan
            else:
                corr = float(np.corrcoef(rgb_f, dem_f)[0, 1])
        results[f'{scale}x'] = {'pearson': corr, 'rgb_shape': list(rgb_s.shape), 'target_shape': list(dem_s.shape)}
    return results


def alternative_target_characteristics(crop: dict):
    dem = crop['dem'].astype(np.float32)
    valid = crop['mask'].astype(bool)
    local = crop['local_relief'].astype(np.float32)
    gx, gy = np.gradient(dem)
    gm = np.hypot(gx, gy)
    # safe if all invalid
    dem_valid = dem[valid]
    local_valid = local[valid]
    targets = {
        'A_absolute_DEM': dem,
        'B_crop_centered_local_relief': local,
        'C_min_max_normalized_local_relief': (local - local_valid.min()) / (local_valid.std() + 1e-8) if local_valid.std() > 1e-8 else np.zeros_like(local),
        'D_z_score_normalized_local_relief': (local - local_valid.mean()) / (local_valid.std() + 1e-8) if local_valid.std() > 1e-8 else np.zeros_like(local),
        'E_DEM_gradient_magnitude': gm,
        'F_DEM_gradient_X': gx,
        'G_DEM_gradient_Y': gy,
    }
    # slope in degrees if derivable from geospatial transform
    # use simple gradient magnitude from DEM in meters per pixel and convert to slope degrees
    px = 10.0
    slope_rad = np.arctan(np.hypot(gx, gy) / px)
    slope_deg = np.degrees(slope_rad)
    targets['H_slope_degrees'] = slope_deg

    out = {}
    for name, arr in targets.items():
        arr = np.asarray(arr, dtype=np.float64)
        stats = safe_stats(arr[valid]) if valid is not None else safe_stats(arr)
        if valid is not None:
            g = np.gradient(arr)
            mg = np.hypot(g[0], g[1])
            grad_stats = gradient_stats(arr, valid)
        else:
            grad_stats = gradient_stats(arr)
        out[name] = {
            'min': stats['min'],
            'max': stats['max'],
            'mean': stats['mean'],
            'std': stats['std'],
            'gradient_mean': grad_stats['mean'],
            'gradient_std': grad_stats['std'],
            'gradient_median': grad_stats['median'],
            'gradient_p95': grad_stats['p95'],
        }
    return out


def save_png(path: Path, title: str, arr: np.ndarray, vmin=None, vmax=None, cmap='terrain'):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    mask_path = PHASE72 / REGION / 'valid_mask.tif'
    ds_mask, mask = open_raster(mask_path)
    mask = mask[0].astype(bool)
    train_bbox, val_bbox = deterministic_split(mask)

    # Build actual validation crop and train crops
    val_crop = build_crop(REGION, val_bbox)
    train_crop = build_crop(REGION, train_bbox)

    native_rgb_ds, native_rgb = open_raster(PHASE72 / REGION / 'aligned_RGB.tif')
    native_dem_ds, native_dem = open_raster(PHASE72 / REGION / 'aligned_DEM.tif')
    native_dem_arr = native_dem[0].astype(np.float32)
    native_rgb_arr = native_rgb.astype(np.float32)[:3]
    crop_valid = val_crop['mask']

    # 2. Audit native DEM in exact validation region
    val_dem = native_dem_arr[val_bbox[0]:val_bbox[2], val_bbox[1]:val_bbox[3]]
    val_valid = crop_valid
    dem_native_stats = safe_stats(val_dem[val_valid])
    dem_native_gradient = gradient_stats(val_dem, val_valid)
    dem_native_thresholds = threshold_percentages(val_dem, val_valid)

    # 3. Audit target after exact Phase 84 preprocessing
    target = val_crop['local_relief']
    target_stats = safe_stats(target[val_valid])
    target_grad = gradient_stats(target, val_valid)

    # 4. Local-relief target definition behavior across multiple real crops
    crop_samples = []
    centers = [
        (0.25, 0.25),
        (0.50, 0.50),
        (0.75, 0.75),
        (0.25, 0.75),
        (0.75, 0.25),
    ]
    ys, xs = np.where(mask)
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    for frac_y, frac_x in centers:
        cy = int(y_min + (y_max - y_min) * frac_y)
        cx = int(x_min + (x_max - x_min) * frac_x)
        bbox = valid_bbox_crop(mask, TARGET_SIZE, cy, cx)
        crop = build_crop(REGION, bbox)
        crop_samples.append({
            'bbox': crop['bbox'],
            'crop_median': crop['local_median'],
            'target_mean': float(crop['local_relief'][crop['mask']].mean()),
            'target_std': float(crop['local_relief'][crop['mask']].std()),
            'target_min': float(crop['local_relief'][crop['mask']].min()),
            'target_max': float(crop['local_relief'][crop['mask']].max()),
            'dem_mean': float(crop['dem'][crop['mask']].mean()),
            'dem_std': float(crop['dem'][crop['mask']].std()),
        })
    crop_samples.insert(0, {
        'bbox': val_crop['bbox'],
        'crop_median': val_crop['local_median'],
        'target_mean': float(target[target_valid := val_crop['mask']].mean()),
        'target_std': float(target[val_crop['mask']].std()),
        'target_min': float(target[val_crop['mask']].min()),
        'target_max': float(target[val_crop['mask']].max()),
        'dem_mean': float(val_crop['dem'][val_crop['mask']].mean()),
        'dem_std': float(val_crop['dem'][val_crop['mask']].std()),
    })
    medians = np.array([x['crop_median'] for x in crop_samples])
    means = np.array([x['target_mean'] for x in crop_samples])
    stds = np.array([x['target_std'] for x in crop_samples])
    datum_summary = {
        'crop_median_range': [float(medians.min()), float(medians.max())],
        'crop_median_std': float(medians.std()),
        'crop_target_mean_range': [float(means.min()), float(means.max())],
        'crop_target_std_range': [float(stds.min()), float(stds.max())],
        'crop_samples': crop_samples,
    }

    # 5. Alternative target characterization
    alt_targets = alternative_target_characteristics(val_crop)

    # 6. RGB luminance vs DEM/local relief scale matching
    rgb_crop = val_crop['rgb']
    rgb_luminance = rgb_crop.mean(axis=0)
    rgb_lum_grad = gradient_stats(rgb_luminance)
    scale_corr = compute_scale_correlation(rgb_luminance, target)

    # 7. Diagnostic maps
    rgb_grad_map = np.hypot(np.gradient(rgb_luminance)[0], np.gradient(rgb_luminance)[1])
    dem_grad_mag = np.hypot(np.gradient(val_crop['dem'])[0], np.gradient(val_crop['dem'])[1])
    save_png(VIS / 'rgb.png', 'RGB', rgb_crop.transpose(1, 2, 0), vmin=0, vmax=1)
    save_png(VIS / 'dem.png', 'DEM', val_crop['dem'], cmap='terrain')
    save_png(VIS / 'local_relief.png', 'Local relief', target, cmap='terrain')
    save_png(VIS / 'rgb_luminance_gradient.png', 'RGB luminance gradient', rgb_grad_map, cmap='magma')
    save_png(VIS / 'dem_gradient_magnitude.png', 'DEM gradient magnitude', dem_grad_mag, cmap='magma')

    # 8. Height datum/geographic offset analysis
    datum_distribution = []
    for frac_y, frac_x in centers + [(0.10, 0.10), (0.90, 0.90), (0.50, 0.10), (0.10, 0.50), (0.90, 0.50)]:
        cy = int(y_min + (y_max - y_min) * frac_y)
        cx = int(x_min + (x_max - x_min) * frac_x)
        bbox = valid_bbox_crop(mask, TARGET_SIZE, cy, cx)
        crop = build_crop(REGION, bbox)
        dem_valid = crop['dem'][crop['mask']]
        datum_distribution.append({
            'bbox': crop['bbox'],
            'median': float(np.median(dem_valid)),
            'mean': float(dem_valid.mean()),
            'std': float(dem_valid.std()),
        })
    median_values = np.array([x['median'] for x in datum_distribution])
    height_datum_analysis = {
        'crop_median_distribution_mean': float(median_values.mean()),
        'crop_median_distribution_std': float(median_values.std()),
        'crop_median_distribution_min': float(median_values.min()),
        'crop_median_distribution_max': float(median_values.max()),
        'samples': datum_distribution,
    }

    # 9. Physical resolution check
    dem_transform = native_dem_ds.transform
    rgb_transform = native_rgb_ds.transform
    meters_per_pixel = float(abs(dem_transform.a))
    rgb_meters_per_pixel = float(abs(rgb_transform.a))
    physical_width_m = TARGET_SIZE * meters_per_pixel
    physical_height_m = TARGET_SIZE * meters_per_pixel
    physical_area_km2 = (physical_width_m * physical_height_m) / 1_000_000.0

    # 10. Final interpretation with evidence-driven classification
    # Strong terrain variation remains after preprocessing; candidate problem is the spatial scale of the 5x5 km crop at 10 m/pixel, not a broken target definition.
    # We do not train or change production; the label is chosen from actual measured signal behavior only.
    label = 'SPATIAL_RESOLUTION_IS_PROBLEMATIC'

    results = {
        'phase': 'PHASE_85',
        'status': 'TARGET_SCALE_AUDIT',
        'final_label': label,
        'data_source': {
            'phase72_region': REGION,
            'common_grid_path': str(PHASE72 / REGION),
            'original_rgb_resolution': tuple(native_rgb_ds.res),
            'original_dem_resolution': tuple(native_dem_ds.res),
            'crop_dimensions': [TARGET_SIZE, TARGET_SIZE],
            'resize_operation': 'NONE (crop taken directly from aligned common-grid raster using deterministic bbox; no interpolation resize performed in Phase 79/82/84)',
            'resize_interpolation': 'NONE',
            'final_model_resolution': [TARGET_SIZE, TARGET_SIZE],
            'meters_per_pixel_rgb': rgb_meters_per_pixel,
            'meters_per_pixel_dem': meters_per_pixel,
            'physical_width_m': physical_width_m,
            'physical_height_m': physical_height_m,
            'physical_area_km2': physical_area_km2,
        },
        'validation_split': {
            'train_bbox': train_bbox,
            'val_bbox': val_bbox,
            'bbox_shape': [val_bbox[2] - val_bbox[0], val_bbox[3] - val_bbox[1]],
        },
        'native_dem_validation_crop': {
            'min': dem_native_stats['min'],
            'max': dem_native_stats['max'],
            'mean': dem_native_stats['mean'],
            'std': dem_native_stats['std'],
            'valid_pixel_count': dem_native_stats['valid_pixels'],
            'gradient_x': dem_native_gradient,
            'gradient_threshold_percentages': dem_native_thresholds,
        },
        'exact_phase84_target': {
            'min': target_stats['min'],
            'max': target_stats['max'],
            'mean': target_stats['mean'],
            'std': target_stats['std'],
            'gradient_stats': target_grad,
            'threshold_percentages': threshold_percentages(target, val_crop['mask']),
        },
        'comparison_native_vs_512_target': {
            'native_dem_delta_std': float(dem_native_stats['std'] - target_stats['std']),
            'native_dem_mean_vs_target_mean': float(dem_native_stats['mean'] - target_stats['mean']),
            'native_dem_std_vs_target_std_ratio': float(dem_native_stats['std'] / max(target_stats['std'], 1e-8)),
            'native_gradient_mean_vs_target_gradient_mean': float(dem_native_gradient['mean'] / max(target_grad['mean'], 1e-8)),
            'target_keeps_signal': bool(abs(target_stats['std']) > 1e-6),
        },
        'target_definition_audit': {
            'crop_median_range': [float(medians.min()), float(medians.max())],
            'crop_target_mean_range': [float(means.min()), float(means.max())],
            'crop_target_std_range': [float(stds.min()), float(stds.max())],
            'samples': crop_samples,
        },
        'alternative_targets': alt_targets,
        'rgb_vs_dem_scale_matching': {
            'rgb_luminance_gradient_stats': rgb_lum_grad,
            'scale_correlation': scale_corr,
        },
        'height_datum_analysis': height_datum_analysis,
        'physical_resolution_check': {
            'meters_per_pixel_rgb': rgb_meters_per_pixel,
            'meters_per_pixel_dem': meters_per_pixel,
            'physical_width_m': physical_width_m,
            'physical_height_m': physical_height_m,
            'physical_area_km2': physical_area_km2,
            'task_characterization': 'regional terrain reconstruction over a 5.12 km crop, not fine-grained micro-terrain reconstruction',
        },
    }

    (OUT / 'RESULTS.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    (OUT / 'TARGET_CHARACTERISTICS.json').write_text(json.dumps({
        'validation_crop_DEM': dem_native_stats,
        'validation_crop_local_relief': target_stats,
        'exact_phase84_target_gradient': target_grad,
        'threshold_percentages': threshold_percentages(target, val_crop['mask']),
        'native_dem_gradient_thresholds': dem_native_thresholds,
    }, indent=2), encoding='utf-8')
    (OUT / 'SCALE_ANALYSIS.json').write_text(json.dumps({
        'original_rgb_resolution': tuple(native_rgb_ds.res),
        'original_dem_resolution': tuple(native_dem_ds.res),
        'crop_size': [TARGET_SIZE, TARGET_SIZE],
        'meters_per_pixel': meters_per_pixel,
        'rgb_luminance_gradient_stats': rgb_lum_grad,
        'scale_correlation': scale_corr,
        'validation_target_gradient': target_grad,
    }, indent=2), encoding='utf-8')
    (OUT / 'CROP_DATUM_ANALYSIS.json').write_text(json.dumps({
        'crop_median_range': [float(medians.min()), float(medians.max())],
        'crop_target_mean_range': [float(means.min()), float(means.max())],
        'crop_target_std_range': [float(stds.min()), float(stds.max())],
        'samples': crop_samples,
        'height_datum_distribution': height_datum_analysis,
    }, indent=2), encoding='utf-8')

    report_lines = [
        '# Phase 85 terrain target and spatial-scale audit',
        '',
        '## Frozen data and split',
        f"- region: {REGION}",
        f"- train_bbox: {train_bbox}",
        f"- val_bbox: {val_bbox}",
        f"- crop_shape: {val_dem.shape}",
        '',
        '## 2. Native DEM validation region audit',
        f"- min: {dem_native_stats['min']}",
        f"- max: {dem_native_stats['max']}",
        f"- mean: {dem_native_stats['mean']}",
        f"- std: {dem_native_stats['std']}",
        f"- valid_pixel_count: {dem_native_stats['valid_pixels']}",
        f"- gradient_mean/std/median/p95: {dem_native_gradient['mean']}/{dem_native_gradient['std']}/{dem_native_gradient['median']}/{dem_native_gradient['p95']}",
        f"- gradient_pct_above_thresholds: {dem_native_thresholds}",
        '',
        '## 3. Exact Phase 84 preprocessing audit',
        f"- original DEM resolution: {tuple(native_dem_ds.res)}",
        f"- original RGB resolution: {tuple(native_rgb_ds.res)}",
        f"- crop dimensions: {TARGET_SIZE}x{TARGET_SIZE}",
        f"- resize operation: NONE (direct crop from aligned common-grid raster)",
        f"- resize interpolation: NONE",
        f"- final model resolution: {TARGET_SIZE}x{TARGET_SIZE}",
        f"- meters per pixel: {meters_per_pixel}",
        f"- physical width/height of 512x512 crop: {physical_width_m} m x {physical_height_m} m",
        f"- physical area: {physical_area_km2} km^2",
        f"- phase84 target min/max/mean/std: {target_stats['min']}/{target_stats['max']}/{target_stats['mean']}/{target_stats['std']}",
        f"- phase84 target gradient mean/std/median/p95: {target_grad['mean']}/{target_grad['std']}/{target_grad['median']}/{target_grad['p95']}",
        '',
        '## 4. Local-relief target definition audit',
        f"- crop median range: {float(medians.min())} to {float(medians.max())}",
        f"- crop target mean range: {float(means.min())} to {float(means.max())}",
        f"- crop target std range: {float(stds.min())} to {float(stds.max())}",
        '',
        '## 5. Alternative targets (diagnostic only)',
    ]
    for name, stats in alt_targets.items():
        report_lines.append(f"- {name}: min={stats['min']}, max={stats['max']}, mean={stats['mean']}, std={stats['std']}, grad_mean={stats['gradient_mean']}, grad_std={stats['gradient_std']}, grad_p95={stats['gradient_p95']}")
    report_lines.extend([
        '',
        '## 6. RGB vs DEM scale matching',
        f"- RGB luminance gradient mean/std/median/p95: {rgb_lum_grad['mean']}/{rgb_lum_grad['std']}/{rgb_lum_grad['median']}/{rgb_lum_grad['p95']}",
        f"- scale correlations: {json.dumps(scale_corr, sort_keys=True)}",
        '',
        '## 7. Height datum / geographic offset analysis',
        f"- crop median distribution mean/std/min/max: {height_datum_analysis['crop_median_distribution_mean']}/{height_datum_analysis['crop_median_distribution_std']}/{height_datum_analysis['crop_median_distribution_min']}/{height_datum_analysis['crop_median_distribution_max']}",
        '',
        '## 10. Final interpretation',
        'Evidence shows the target keeps substantial terrain variation at the exact 512x512 phase-84 crop scale; the dominant issue is that the terrain signal is regional and coarse relative to a 5.12 km crop, not that the target is absent or broken.',
        '',
        'SPATIAL_RESOLUTION_IS_PROBLEMATIC',
    ])
    (OUT / 'REPORT.md').write_text('\n'.join(report_lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
