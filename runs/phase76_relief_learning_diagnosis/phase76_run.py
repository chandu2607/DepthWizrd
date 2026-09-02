from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depthwizard.depth.depth_anything import DepthAnythingV2
from depthwizard.models.fusion_head import SmallFusionUNet

PHASE72 = REPO_ROOT / 'runs' / 'phase72_common_grid_forensics' / 'common_grid'
PHASE75 = REPO_ROOT / 'runs' / 'phase75_relief_diagnosis'
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)
VIS = OUT / 'VISUALS'
VIS.mkdir(parents=True, exist_ok=True)

TRAIN_REGION = 'uttarakhand'
VAL_REGION = 'himachal'
TARGET_SIZE = 512
SEED = 1337


def open_raster(path: Path):
    ds = rasterio.open(path)
    return ds, ds.read()


def valid_bbox_crop(valid: np.ndarray, size: int):
    ys, xs = np.where(valid)
    if ys.size == 0:
        raise RuntimeError('No valid pixels in region mask')
    y0 = int(max(0, ys.min()))
    y1 = int(min(valid.shape[0], ys.max() + 1))
    x0 = int(max(0, xs.min()))
    x1 = int(min(valid.shape[1], xs.max() + 1))
    cy = (y0 + y1) // 2
    cx = (x0 + x1) // 2
    y_start = int(np.clip(cy - size // 2, 0, max(0, valid.shape[0] - size)))
    x_start = int(np.clip(cx - size // 2, 0, max(0, valid.shape[1] - size)))
    y_end = y_start + size
    x_end = x_start + size
    if y_end > valid.shape[0]:
        y_end = valid.shape[0]
        y_start = max(0, y_end - size)
    if x_end > valid.shape[1]:
        x_end = valid.shape[1]
        x_start = max(0, x_end - size)
    return (y_start, x_start, y_end, x_end)


def build_region_cache(region_name: str):
    region_dir = PHASE72 / region_name
    rgb_path = region_dir / 'aligned_RGB.tif'
    dem_path = region_dir / 'aligned_DEM.tif'
    mask_path = region_dir / 'valid_mask.tif'
    ds_rgb, rgb = open_raster(rgb_path)
    ds_dem, dem = open_raster(dem_path)
    ds_mask, mask = open_raster(mask_path)
    rgb = rgb.astype(np.float32)
    dem = dem[0].astype(np.float32)
    mask = mask[0].astype(bool)
    rgb = rgb[:3]
    if rgb.shape[1:] != dem.shape:
        raise ValueError(f'{region_name} RGB and DEM shapes differ')
    if mask.shape != dem.shape:
        raise ValueError(f'{region_name} mask shape mismatch')
    y0, x0, y1, x1 = valid_bbox_crop(mask, TARGET_SIZE)
    rgb_crop = rgb[:, y0:y1, x0:x1]
    dem_crop = dem[y0:y1, x0:x1]
    mask_crop = mask[y0:y1, x0:x1]
    if not mask_crop.any():
        raise RuntimeError(f'{region_name} valid crop contains no valid pixels')
    local_median = float(np.median(dem_crop[mask_crop]))
    relief_crop = dem_crop.astype(np.float32) - local_median
    rgb_norm = np.clip(rgb_crop / 65535.0, 0.0, 1.0)
    out_cache = OUT / 'cache' / region_name
    out_cache.mkdir(parents=True, exist_ok=True)
    np.save(out_cache / 'rgb.npy', rgb_norm)
    np.save(out_cache / 'dem.npy', dem_crop)
    np.save(out_cache / 'relief.npy', relief_crop)
    np.save(out_cache / 'mask.npy', mask_crop)
    np.save(out_cache / 'local_median.npy', np.array([local_median], dtype=np.float32))
    return {
        'region': region_name,
        'crop_shape': tuple(rgb_norm.shape[1:]),
        'crop_bbox': [y0, x0, y1, x1],
        'valid_pixels': int(mask_crop.sum()),
        'local_median_m': local_median,
        'dem_min_m': float(np.nanmin(dem_crop[mask_crop])),
        'dem_max_m': float(np.nanmax(dem_crop[mask_crop])),
        'dem_mean_m': float(dem_crop[mask_crop].mean()),
        'dem_std_m': float(dem_crop[mask_crop].std()),
        'relief_min_m': float(relief_crop[mask_crop].min()),
        'relief_max_m': float(relief_crop[mask_crop].max()),
        'relief_mean_m': float(relief_crop[mask_crop].mean()),
        'relief_std_m': float(relief_crop[mask_crop].std()),
    }


def compute_train_stats(train_target: np.ndarray, train_mask: np.ndarray):
    valid = train_target[train_mask]
    if valid.size == 0:
        raise RuntimeError('train target empty after valid mask')
    mean = float(valid.mean())
    std = float(valid.std())
    if std < 1e-6:
        std = 1.0
    return {'mean': mean, 'std': std}


class TerrainDataset(Dataset):
    def __init__(self, region_name: str, train_stats: dict | None = None, target_kind: str = 'local_relief'):
        self.cache_root = OUT / 'cache' / region_name
        self.rgb = np.load(self.cache_root / 'rgb.npy').astype(np.float32)
        self.dem = np.load(self.cache_root / 'dem.npy').astype(np.float32)
        self.mask = np.load(self.cache_root / 'mask.npy').astype(bool)
        self.local_median = float(np.load(self.cache_root / 'local_median.npy')[0])
        self.target_kind = target_kind
        self.train_stats = train_stats
        depth_model = DepthAnythingV2(
            model_id='depth-anything/Depth-Anything-V2-Small-hf',
            cache_dir=str(REPO_ROOT / 'data' / 'depth_cache'),
            use_cache=True,
        )
        rgb_u8 = (self.rgb.transpose(1, 2, 0) * 255.0).astype(np.uint8)
        self.depth = depth_model.infer(rgb_u8, key=f'{region_name}_phase76_{TARGET_SIZE}', target_hw=(TARGET_SIZE, TARGET_SIZE)).astype(np.float32)
        self.depth = (self.depth - self.depth.mean()) / (self.depth.std() + 1e-6)

    def __len__(self):
        return 1

    def _target_array(self):
        if self.target_kind == 'local_relief':
            return self.dem - self.local_median
        if self.target_kind == 'absolute':
            return self.dem.copy()
        raise ValueError(f'unknown target kind: {self.target_kind}')

    def __getitem__(self, index):
        target = self._target_array()
        if self.train_stats is not None:
            target = (target - self.train_stats['mean']) / self.train_stats['std']
        x = np.concatenate([self.rgb, self.depth[None, ...]], axis=0)
        return {
            'image': torch.from_numpy(x),
            'target': torch.from_numpy(target[None, ...]),
            'mask': torch.from_numpy(self.mask[None, ...].astype(np.float32)),
        }


class TerrainHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = SmallFusionUNet(w=16, in_channels=4, out_channels=1)

    def forward(self, x):
        out = self.net(x)
        if out.dim() == 3:
            out = out.unsqueeze(1)
        return out


def inverse_normalize(x: np.ndarray, mean: float, std: float):
    return x * std + mean


def compute_metrics(pred: np.ndarray, target: np.ndarray):
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    if pred.size == 0:
        return {'mae': np.nan, 'rmse': np.nan, 'bias': np.nan, 'pearson': np.nan, 'pred_min': np.nan, 'pred_max': np.nan, 'pred_mean': np.nan, 'pred_std': np.nan, 'ref_min': np.nan, 'ref_max': np.nan, 'ref_mean': np.nan, 'ref_std': np.nan, 'n': 0}
    err = pred - target
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    if np.std(pred) < 1e-8 or np.std(target) < 1e-8:
        pearson = np.nan
    else:
        pearson = float(np.corrcoef(pred, target)[0, 1])
    return {
        'mae': mae,
        'rmse': rmse,
        'bias': bias,
        'pearson': pearson,
        'pred_min': float(pred.min()),
        'pred_max': float(pred.max()),
        'pred_mean': float(pred.mean()),
        'pred_std': float(pred.std()),
        'ref_min': float(target.min()),
        'ref_max': float(target.max()),
        'ref_mean': float(target.mean()),
        'ref_std': float(target.std()),
        'n': int(pred.size),
    }


def make_pred_maps():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    # Freeze the Phase 75 target definition and the same training/validation data.
    region_cache = {TRAIN_REGION: build_region_cache(TRAIN_REGION), VAL_REGION: build_region_cache(VAL_REGION)}

    train_target = np.load(OUT / 'cache' / TRAIN_REGION / 'dem.npy').astype(np.float32) - float(np.load(OUT / 'cache' / TRAIN_REGION / 'local_median.npy')[0])
    train_mask = np.load(OUT / 'cache' / TRAIN_REGION / 'mask.npy').astype(bool)
    train_stats = compute_train_stats(train_target, train_mask)

    train_ds = TerrainDataset(TRAIN_REGION, train_stats=train_stats, target_kind='local_relief')
    val_ds = TerrainDataset(VAL_REGION, train_stats=train_stats, target_kind='local_relief')
    loader_train = DataLoader(train_ds, batch_size=1, shuffle=False)
    loader_val = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = TerrainHead()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.SmoothL1Loss(reduction='none')
    model.train()
    for batch in loader_train:
        x = batch['image']
        y = batch['target']
        m = batch['mask']
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss_map = criterion(pred * m, y * m)
        loss = (loss_map * m).sum() / (m.sum() + 1e-6)
        loss.backward()
        optimizer.step()
        break

    model.eval()
    pred_maps = []
    target_maps = []
    masks = []
    with torch.no_grad():
        for batch in loader_val:
            x = batch['image']
            y = batch['target']
            m = batch['mask']
            pred = model(x).numpy()[0, 0]
            y_np = y.numpy()[0, 0]
            m_np = m.numpy()[0, 0].astype(bool)
            pred_raw = inverse_normalize(pred, train_stats['mean'], train_stats['std'])
            target_raw = inverse_normalize(y_np, train_stats['mean'], train_stats['std'])
            pred_maps.append(pred_raw)
            target_maps.append(target_raw)
            masks.append(m_np)

    pred = pred_maps[0]
    target = target_maps[0]
    mask = masks[0]
    assert pred.shape == target.shape == mask.shape
    return pred, target, mask, train_target, train_mask, train_stats


def save_panel(path: Path, title: str, arr: np.ndarray, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(4, 4))
    img = ax.imshow(arr, cmap='terrain', vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis('off')
    fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def get_baseline_metrics(pred: np.ndarray, target: np.ndarray, train_relief: np.ndarray):
    valid = np.isfinite(target)
    zero_pred = np.zeros_like(pred)
    mean_pred = np.full_like(pred, float(np.mean(train_relief)))
    median_pred = np.full_like(pred, float(np.median(train_relief)))
    out = {}
    for name, arr in [('zero', zero_pred), ('mean', mean_pred), ('median', median_pred)]:
        out[name] = compute_metrics(arr[valid], target[valid])
    return out


def main():
    phase75_json = json.loads((PHASE75 / 'RESULTS.json').read_text(encoding='utf-8'))
    pred, target, mask, train_relief, train_mask, train_stats = make_pred_maps()

    # Phase 75 prediction map and ground truth on the polarised local relief crop.
    true_relief = target
    pred_relief = pred
    valid = mask
    pred_valid = pred_relief[valid]
    true_valid = true_relief[valid]

    # Baseline metrics on the same Himachal local-relief crop.
    train_relief_valid = np.asarray(train_relief[train_mask]).astype(np.float64)
    baseline_metrics = get_baseline_metrics(pred_relief, true_relief, train_relief_valid)

    # Shuffle control with a fixed seed.
    rng = np.random.default_rng(SEED)
    shuffled = pred_valid[rng.permutation(pred_valid.size)].reshape(pred_relief.shape)
    shuffled = shuffled * valid.astype(np.float64)
    shuffle_metrics = compute_metrics(shuffled[valid], true_valid)

    # Constant prediction correlation check.
    const_pred = np.full_like(true_valid, 0.0)
    const_metrics = compute_metrics(const_pred, true_valid)

    # Spatial diagnostics.
    pred_map = pred_relief
    target_map = true_relief
    err_map = pred_map - target_map
    pred_dx = np.diff(pred_map, axis=1)
    pred_dy = np.diff(pred_map, axis=0)
    target_dx = np.diff(target_map, axis=1)
    target_dy = np.diff(target_map, axis=0)
    grad_mean_abs_diff = float(np.mean(np.abs(pred_dx - target_dx)))
    grad_mean_abs_diff += float(np.mean(np.abs(pred_dy - target_dy)))
    grad_mean_abs_diff /= 2.0

    # Use the valid mask for the summary stats.
    pred_valid_summary = pred_valid
    target_valid_summary = true_valid
    err_valid = pred_valid_summary - target_valid_summary
    spatial_diag = {
        'valid_pixel_count': int(mask.sum()),
        'correlation': float(np.corrcoef(pred_valid_summary, target_valid_summary)[0, 1]),
        'mae': float(np.mean(np.abs(err_valid))),
        'rmse': float(np.sqrt(np.mean(err_valid ** 2))),
        'mean_bias': float(np.mean(err_valid)),
        'prediction_std': float(pred_valid_summary.std()),
        'target_std': float(target_valid_summary.std()),
        'error_std': float(err_valid.std()),
        'gradient_x_prediction_mean_abs': float(np.abs(pred_dx).mean()),
        'gradient_x_target_mean_abs': float(np.abs(target_dx).mean()),
        'gradient_y_prediction_mean_abs': float(np.abs(pred_dy).mean()),
        'gradient_y_target_mean_abs': float(np.abs(target_dy).mean()),
        'mean_absolute_gradient_difference': float(grad_mean_abs_diff),
    }

    # Visuals.
    vmin = float(np.min(true_relief[mask]))
    vmax = float(np.max(true_relief[mask]))
    save_panel(VIS / '01_true_himachal_local_relief.png', 'True Himachal local relief', true_relief, vmin=vmin, vmax=vmax)
    save_panel(VIS / '02_phase75_predicted_local_relief.png', 'Phase 75 predicted local relief', pred_relief, vmin=vmin, vmax=vmax)
    save_panel(VIS / '03_absolute_error.png', 'Absolute error', np.abs(err_map), vmin=0.0, vmax=max(10.0, float(np.abs(err_map[mask]).max())))
    save_panel(VIS / '04_shuffled_prediction.png', 'Shuffled prediction', shuffled, vmin=vmin, vmax=vmax)
    save_panel(VIS / '05_zero_baseline.png', 'Zero baseline', np.zeros_like(pred_relief), vmin=vmin, vmax=vmax)

    phase75_metrics = {
        'mae_m': float(phase75_json['validation_local_relief_metrics']['mae_m']),
        'rmse_m': float(phase75_json['validation_local_relief_metrics']['rmse_m']),
        'pearson': float(phase75_json['validation_local_relief_metrics']['pearson']),
        'prediction_mean_m': float(phase75_json['validation_local_relief_metrics']['prediction_mean_m']),
    }

    # Decision logic.
    phase75_mae = float(phase75_metrics['mae_m'])
    phase75_rmse = float(phase75_metrics['rmse_m'])
    phase75_corr = float(phase75_metrics['pearson'])
    zero_mae = float(baseline_metrics['zero']['mae'])
    mean_mae = float(baseline_metrics['mean']['mae'])
    median_mae = float(baseline_metrics['median']['mae'])
    zero_rmse = float(baseline_metrics['zero']['rmse'])
    mean_rmse = float(baseline_metrics['mean']['rmse'])
    median_rmse = float(baseline_metrics['median']['rmse'])
    zero_corr = baseline_metrics['zero']['pearson']
    mean_corr = baseline_metrics['mean']['pearson']
    median_corr = baseline_metrics['median']['pearson']

    pred_better_than_zero = phase75_mae < zero_mae and phase75_rmse < zero_rmse
    pred_better_than_mean = phase75_mae < mean_mae and phase75_rmse < mean_rmse
    pred_better_than_median = phase75_mae < median_mae and phase75_rmse < median_rmse
    meaningfully_spatial = phase75_corr > 0.1 and spatial_diag['mean_absolute_gradient_difference'] > 0.0
    if pred_better_than_zero and pred_better_than_mean and pred_better_than_median and meaningfully_spatial:
        decision = 'LOCAL_RELIEF_LEARNING_DEMONSTRATED'
    elif pred_better_than_zero and pred_better_than_mean and pred_better_than_median and not meaningfully_spatial:
        decision = 'LOCAL_RELIEF_PRIOR_IMPROVEMENT_WITHOUT_SPATIAL_LEARNING'
    elif (not pred_better_than_zero and not pred_better_than_mean and not pred_better_than_median):
        decision = 'LOCAL_RELIEF_IMPROVEMENT_NOT_YET_DEMONSTRATED'
    else:
        decision = 'LOCAL_RELIEF_PRIOR_IMPROVEMENT_WITHOUT_SPATIAL_LEARNING'

    baseline_comparison = {
        'phase_75_prediction': {
            'mae_m': phase75_mae,
            'rmse_m': phase75_rmse,
            'pearson': phase75_corr,
            'prediction_mean_m': float(phase75_metrics['prediction_mean_m']),
        },
        'baseline_zero_relief': {
            'mae_m': float(baseline_metrics['zero']['mae']),
            'rmse_m': float(baseline_metrics['zero']['rmse']),
            'pearson': baseline_metrics['zero']['pearson'],
            'prediction_value_m': 0.0,
        },
        'baseline_mean_relief': {
            'mae_m': float(baseline_metrics['mean']['mae']),
            'rmse_m': float(baseline_metrics['mean']['rmse']),
            'pearson': baseline_metrics['mean']['pearson'],
            'prediction_value_m': float(np.mean(train_relief_valid)),
        },
        'baseline_median_relief': {
            'mae_m': float(baseline_metrics['median']['mae']),
            'rmse_m': float(baseline_metrics['median']['rmse']),
            'pearson': baseline_metrics['median']['pearson'],
            'prediction_value_m': float(np.median(train_relief_valid)),
        },
        'spatial_shuffle': {
            'mae_m': float(shuffle_metrics['mae']),
            'rmse_m': float(shuffle_metrics['rmse']),
            'pearson': float(shuffle_metrics['pearson']),
            'seed': SEED,
        },
        'constant_prediction': {
            'mae_m': float(const_metrics['mae']),
            'rmse_m': float(const_metrics['rmse']),
            'pearson': const_metrics['pearson'],
            'note': 'Pearson is NaN when the prediction has zero variance because np.std(prediction) = 0.',
        },
    }

    results = {
        'phase': 'PHASE_76',
        'status': 'LOCAL_RELIEF_LEARNING_DIAGNOSTIC',
        'train_region': TRAIN_REGION,
        'validation_region': VAL_REGION,
        'test_region': 'sikkim_locked',
        'phase75_frozen': True,
        'phase75_reference_metrics_m': phase75_metrics,
        'baseline_comparison': baseline_comparison,
        'decision': decision,
    }

    (OUT / 'RESULTS.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    (OUT / 'BASELINE_COMPARISON.json').write_text(json.dumps(baseline_comparison, indent=2), encoding='utf-8')
    (OUT / 'SPATIAL_DIAGNOSTICS.json').write_text(json.dumps({
        'valid_pixel_count': int(mask.sum()),
        'phase75_prediction': {
            'prediction_min_m': float(pred_valid.min()),
            'prediction_max_m': float(pred_valid.max()),
            'prediction_mean_m': float(pred_valid.mean()),
            'prediction_std_m': float(pred_valid.std()),
        },
        'true_relief': {
            'target_min_m': float(true_valid.min()),
            'target_max_m': float(true_valid.max()),
            'target_mean_m': float(true_valid.mean()),
            'target_std_m': float(true_valid.std()),
        },
        'error': {
            'mae_m': float(np.mean(np.abs(pred_valid - true_valid))),
            'rmse_m': float(np.sqrt(np.mean((pred_valid - true_valid) ** 2))),
            'mean_bias_m': float(np.mean(pred_valid - true_valid)),
            'error_std_m': float((pred_valid - true_valid).std()),
        },
        'correlation': float(np.corrcoef(pred_valid, true_valid)[0, 1]),
        'gradient_x_prediction_mean_abs': float(np.abs(np.diff(pred_relief, axis=1)).mean()),
        'gradient_x_target_mean_abs': float(np.abs(np.diff(target_map, axis=1)).mean()),
        'gradient_y_prediction_mean_abs': float(np.abs(np.diff(pred_relief, axis=0)).mean()),
        'gradient_y_target_mean_abs': float(np.abs(np.diff(target_map, axis=0)).mean()),
        'mean_absolute_gradient_difference': float(grad_mean_abs_diff),
        'metric_implementation_note': 'Pearson is NaN for constant predictions because the standard deviation of the prediction is exactly zero; the code explicitly checks for zero variance before computing the correlation.',
    }, indent=2), encoding='utf-8')

    report = (
        '# Phase 76 relief-learning diagnosis\n\n'
        '## Frozen inputs\n'
        '- TRAIN = Uttarakhand\n'
        '- VALIDATION = Himachal\n'
        '- Sikkim remains locked and was not evaluated.\n\n'
        '## Baselines\n'
        f"- ZERO RELIEF: MAE={baseline_metrics['zero']['mae']:.6f} m, RMSE={baseline_metrics['zero']['rmse']:.6f} m, Pearson={baseline_metrics['zero']['pearson']}\n"
        f"- MEAN RELIEF: MAE={baseline_metrics['mean']['mae']:.6f} m, RMSE={baseline_metrics['mean']['rmse']:.6f} m, Pearson={baseline_metrics['mean']['pearson']}\n"
        f"- MEDIAN RELIEF: MAE={baseline_metrics['median']['mae']:.6f} m, RMSE={baseline_metrics['median']['rmse']:.6f} m, Pearson={baseline_metrics['median']['pearson']}\n\n"
        '## Spatial-shuffle control\n'
        f"- Shuffled prediction: MAE={shuffle_metrics['mae']:.6f} m, RMSE={shuffle_metrics['rmse']:.6f} m, Pearson={shuffle_metrics['pearson']} (seed={SEED})\n\n"
        '## Constant-prediction check\n'
        f"- Constant prediction vs true relief: MAE={const_metrics['mae']:.6f} m, RMSE={const_metrics['rmse']:.6f} m, Pearson={const_metrics['pearson']}\n"
        '- The metric implementation explicitly returns NaN when the prediction variance is zero because the correlation is undefined in that case.\n\n'
        '## Spatial structure\n'
        f"- Valid-pixel correlation: {spatial_diag['correlation']:.6f}\n"
        f"- MAE: {spatial_diag['mae']:.6f} m\n"
        f"- RMSE: {spatial_diag['rmse']:.6f} m\n"
        f"- Mean bias: {spatial_diag['mean_bias']:.6f} m\n"
        f"- Prediction std: {spatial_diag['prediction_std']:.6f} m\n"
        f"- Target std: {spatial_diag['target_std']:.6f} m\n"
        f"- Error std: {spatial_diag['error_std']:.6f} m\n"
        f"- Mean absolute gradient difference: {spatial_diag['mean_absolute_gradient_difference']:.6f}\n\n"
        '## Phase 75 vs baselines\n'
        f"- Phase 75 local relief: MAE={phase75_mae:.6f} m, RMSE={phase75_rmse:.6f} m, Pearson={phase75_corr:.6f}, prediction mean={phase75_metrics['prediction_mean_m']:.6f} m\n"
        f"- Best baseline: zero={zero_mae:.6f} m, mean={mean_mae:.6f} m, median={median_mae:.6f} m\n\n"
        f"## Decision: {decision}\n\n"
        f"{decision}"
    )
    (OUT / 'REPORT.md').write_text(report + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
