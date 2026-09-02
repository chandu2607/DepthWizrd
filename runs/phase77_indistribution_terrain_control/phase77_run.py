from __future__ import annotations

import csv
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
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)
VIS = OUT / 'VISUALS'
VIS.mkdir(parents=True, exist_ok=True)

TRAIN_REGION = 'uttarakhand'
VALIDATION_REGION = 'uttarakhand'
TARGET_SIZE = 512
SEED = 1337


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
    # Ensure non-overlap by comparing bounding boxes.
    y0a, x0a, y1a, x1a = train_bbox
    y0b, x0b, y1b, x1b = val_bbox
    if not (x1a <= x0b or x1b <= x0a or y1a <= y0b or y1b <= y0a):
        # fallback: explicitly keep train on left-upper and val on right-lower
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
    if not mask_crop.any():
        raise RuntimeError(f'Crop for {region_name} has no valid pixels')
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
    }


def compute_train_stats(train_target: np.ndarray, train_mask: np.ndarray):
    valid = train_target[train_mask]
    if valid.size == 0:
        raise RuntimeError('Train target empty after valid mask.')
    mean = float(valid.mean())
    std = float(valid.std())
    if std < 1e-6:
        std = 1.0
    return {'mean': mean, 'std': std}


class TerrainDataset(Dataset):
    def __init__(self, crop: dict, train_stats: dict | None = None):
        self.rgb = crop['rgb'].astype(np.float32)
        self.dem = crop['dem'].astype(np.float32)
        self.mask = crop['mask'].astype(bool)
        self.local_median = float(crop['local_median'])
        self.train_stats = train_stats
        depth_model = DepthAnythingV2(
            model_id='depth-anything/Depth-Anything-V2-Small-hf',
            cache_dir=str(REPO_ROOT / 'data' / 'depth_cache'),
            use_cache=True,
        )
        rgb_u8 = (self.rgb.transpose(1, 2, 0) * 255.0).astype(np.uint8)
        self.depth = depth_model.infer(rgb_u8, key=f'phase77_{TRAIN_REGION}_{TARGET_SIZE}', target_hw=(TARGET_SIZE, TARGET_SIZE)).astype(np.float32)
        self.depth = (self.depth - self.depth.mean()) / (self.depth.std() + 1e-6)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        target = self.dem - self.local_median
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


def compute_metrics(pred: np.ndarray, target: np.ndarray, valid: np.ndarray):
    pred = pred[valid]
    target = target[valid]
    if pred.size == 0:
        return {'mae': np.nan, 'rmse': np.nan, 'pearson': np.nan, 'mean_bias': np.nan, 'prediction_std': np.nan, 'target_std': np.nan, 'error_std': np.nan, 'n': 0}
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
        'pearson': pearson,
        'mean_bias': bias,
        'prediction_std': float(pred.std()),
        'target_std': float(target.std()),
        'error_std': float(err.std()),
        'n': int(pred.size),
    }


def gradient_abs_error(pred: np.ndarray, target: np.ndarray):
    pred_grad_x = np.abs(np.diff(pred, axis=1))
    pred_grad_y = np.abs(np.diff(pred, axis=0))
    tar_grad_x = np.abs(np.diff(target, axis=1))
    tar_grad_y = np.abs(np.diff(target, axis=0))
    # Match same-shaped arrays by using the mean of the two gradient map errors.
    x_error = float(np.mean(np.abs(pred_grad_x - tar_grad_x)))
    y_error = float(np.mean(np.abs(pred_grad_y - tar_grad_y)))
    return {'gradient_x_mae': x_error, 'gradient_y_mae': y_error, 'gradient_mae_mean': 0.5 * (x_error + y_error)}


def save_image(path: Path, title: str, arr: np.ndarray, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(4, 4))
    img = ax.imshow(arr, cmap='terrain', vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis('off')
    fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds_mask, mask = open_raster(PHASE72 / TRAIN_REGION / 'valid_mask.tif')
    mask = mask[0].astype(bool)
    train_bbox, val_bbox = deterministic_split(mask)
    train_crop = build_crop(TRAIN_REGION, train_bbox)
    val_crop = build_crop(TRAIN_REGION, val_bbox)

    # Create target proof for the training and validation crops.
    train_target = train_crop['dem'] - train_crop['local_median']
    val_target = val_crop['dem'] - val_crop['local_median']
    train_stats = compute_train_stats(train_target, train_crop['mask'])

    train_ds = TerrainDataset(train_crop, train_stats)
    val_ds = TerrainDataset(val_crop, train_stats)
    loader_train = DataLoader(train_ds, batch_size=1, shuffle=False)
    loader_val = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = TerrainHead().to(device)
    criterion = nn.SmoothL1Loss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    start = __import__('time').time()
    train_history = []
    model.train()
    for batch_idx, batch in enumerate(loader_train):
        x = batch['image'].to(device, dtype=torch.float32)
        y = batch['target'].to(device, dtype=torch.float32)
        m = batch['mask'].to(device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss_map = criterion(pred * m, y * m)
        loss = (loss_map * m).sum() / (m.sum() + 1e-6)
        if not torch.isfinite(loss):
            raise RuntimeError('Training loss is non-finite in Phase 77.')
        loss.backward()
        if any((p.grad is not None) and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError('Gradients are non-finite in Phase 77.')
        optimizer.step()
        train_history.append({'batch': batch_idx, 'loss': float(loss.item())})
    epoch_time = __import__('time').time() - start

    model.eval()
    val_pred_list = []
    val_target_list = []
    val_mask_list = []
    with torch.no_grad():
        for batch in loader_val:
            x = batch['image'].to(device, dtype=torch.float32)
            y = batch['target'].to(device, dtype=torch.float32)
            m = batch['mask'].to(device, dtype=torch.float32)
            pred = model(x).cpu().numpy()[0, 0]
            y_np = y.cpu().numpy()[0, 0]
            m_np = m.cpu().numpy()[0, 0].astype(bool)
            pred_raw = inverse_normalize(pred, train_stats['mean'], train_stats['std'])
            target_raw = inverse_normalize(y_np, train_stats['mean'], train_stats['std'])
            val_pred_list.append(pred_raw)
            val_target_list.append(target_raw)
            val_mask_list.append(m_np)

    pred = np.concatenate([p for p in val_pred_list], axis=0)
    target = np.concatenate([t for t in val_target_list], axis=0)
    mask = np.concatenate([m for m in val_mask_list], axis=0)
    metrics = compute_metrics(pred, target, mask)
    grad_stats = gradient_abs_error(pred, target)
    # Baselines from the training crop only.
    train_relief_valid = (train_crop['dem'] - train_crop['local_median'])[train_crop['mask']]
    zero_pred = np.zeros_like(pred)
    mean_pred = np.full_like(pred, float(train_relief_valid.mean()))
    median_pred = np.full_like(pred, float(np.median(train_relief_valid)))
    baseline_zero = compute_metrics(zero_pred, target, mask)
    baseline_mean = compute_metrics(mean_pred, target, mask)
    baseline_median = compute_metrics(median_pred, target, mask)

    # Side-by-side visuals.
    vmin = float(np.min(target[mask]))
    vmax = float(np.max(target[mask]))
    pred_grad_x = np.abs(np.diff(pred, axis=1))
    pred_grad_y = np.abs(np.diff(pred, axis=0))
    target_grad_x = np.abs(np.diff(target, axis=1))
    target_grad_y = np.abs(np.diff(target, axis=0))
    pred_grad_mag = np.zeros_like(pred)
    pred_grad_mag[:, :-1] = pred_grad_x
    pred_grad_mag[:-1, :] = np.maximum(pred_grad_mag[:-1, :], pred_grad_y[:, :])
    target_grad_mag = np.zeros_like(target)
    target_grad_mag[:, :-1] = target_grad_x
    target_grad_mag[:-1, :] = np.maximum(target_grad_mag[:-1, :], target_grad_y[:, :])
    save_image(VIS / '01_uttarakhand_validation_true_local_relief.png', 'Held-out Uttarakhand true local relief', target, vmin=vmin, vmax=vmax)
    save_image(VIS / '02_uttarakhand_validation_predicted_local_relief.png', 'Held-out Uttarakhand predicted local relief', pred, vmin=vmin, vmax=vmax)
    save_image(VIS / '03_uttarakhand_validation_absolute_error.png', 'Absolute error', np.abs(pred - target), vmin=0.0, vmax=max(1.0, float(np.abs(pred[mask] - target[mask]).max())))
    save_image(VIS / '04_uttarakhand_validation_zero_baseline.png', 'Zero baseline', zero_pred, vmin=vmin, vmax=vmax)
    save_image(VIS / '05_uttarakhand_prediction_gradient_magnitude.png', 'Prediction gradient magnitude', pred_grad_mag, vmin=0.0, vmax=max(1.0, float(pred_grad_mag.max())))
    save_image(VIS / '06_uttarakhand_target_gradient_magnitude.png', 'Target gradient magnitude', target_grad_mag, vmin=0.0, vmax=max(1.0, float(target_grad_mag.max())))

    # Comparison to Phase 75/76 cross-region run.
    phase75_results = json.loads((REPO_ROOT / 'runs' / 'phase75_relief_diagnosis' / 'RESULTS.json').read_text(encoding='utf-8'))
    phase75_metrics = phase75_results['validation_local_relief_metrics']
    phase76_results = json.loads((REPO_ROOT / 'runs' / 'phase76_relief_learning_diagnosis' / 'RESULTS.json').read_text(encoding='utf-8'))

    if (
        metrics['mae'] < min(baseline_zero['mae'], baseline_mean['mae'], baseline_median['mae'])
        and np.isfinite(metrics['pearson'])
        and abs(metrics['pearson']) > 0.1
        and grad_stats['gradient_mae_mean'] > 0.0
    ):
        decision = 'IN_DISTRIBUTION_TERRAIN_LEARNING_DEMONSTRATED'
    elif (
        abs(metrics['pearson']) < 0.1
        and grad_stats['gradient_mae_mean'] < 1e-3
        and metrics['mae'] >= min(baseline_zero['mae'], baseline_mean['mae'], baseline_median['mae']) * 0.95
    ):
        decision = 'TERRAIN_LEARNING_CAPABILITY_NOT_DEMONSTRATED'
    else:
        decision = 'IN_DISTRIBUTION_DIAGNOSIS_INCONCLUSIVE'

    comparison = {
        'in_distribution': {
            'train': 'uttarakhand',
            'validation': 'uttarakhand',
            'mae_m': float(metrics['mae']),
            'rmse_m': float(metrics['rmse']),
            'pearson': float(metrics['pearson']) if np.isfinite(metrics['pearson']) else None,
            'mean_bias_m': float(metrics['mean_bias']),
            'prediction_std_m': float(metrics['prediction_std']),
            'target_std_m': float(metrics['target_std']),
            'error_std_m': float(metrics['error_std']),
            'gradient_x_mae_m': float(grad_stats['gradient_x_mae']),
            'gradient_y_mae_m': float(grad_stats['gradient_y_mae']),
            'gradient_mae_mean_m': float(grad_stats['gradient_mae_mean']),
        },
        'cross_region_reference': {
            'train': 'uttarakhand',
            'validation': 'himachal',
            'mae_m': float(phase75_results['validation_local_relief_metrics']['mae_m']),
            'rmse_m': float(phase75_results['validation_local_relief_metrics']['rmse_m']),
            'pearson': float(phase75_results['validation_local_relief_metrics']['pearson']),
            'prediction_std_m': float(phase75_results['validation_local_relief_metrics']['prediction_std_m']),
            'target_std_m': float(phase75_results['validation_local_relief_metrics']['target_std_m']),
        },
    }

    results = {
        'phase': 'PHASE_77',
        'status': 'IN_DISTRIBUTION_TERRAIN_CONTROL',
        'train_region': TRAIN_REGION,
        'validation_region': VALIDATION_REGION,
        'locked_region': 'sikkim',
        'target_definition': 'LOCAL_RELIEF_TARGET = DEM - median(valid DEM pixels in the crop)',
        'training_budget': {
            'train_crops': 1,
            'validation_crops': 1,
            'number_of_batches': len(train_history),
            'optimizer_steps': len(train_history),
            'epoch_count': 1,
            'train_time_seconds': float(epoch_time),
            'loss_history': train_history,
        },
        'train_crop': {
            'bbox': train_crop['bbox'],
            'valid_pixels': int(train_crop['mask'].sum()),
            'local_median_m': float(train_crop['local_median']),
        },
        'validation_crop': {
            'bbox': val_crop['bbox'],
            'valid_pixels': int(val_crop['mask'].sum()),
            'local_median_m': float(val_crop['local_median']),
        },
        'metrics': metrics,
        'gradient_metrics': grad_stats,
        'baselines': {
            'zero_prediction': baseline_zero,
            'training_mean_prediction': baseline_mean,
            'training_median_prediction': baseline_median,
        },
        'comparison': comparison,
        'decision': decision,
    }

    with open(OUT / 'RESULTS.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    with open(OUT / 'BASELINE_COMPARISON.json', 'w', encoding='utf-8') as f:
        json.dump({
            'zero_prediction': baseline_zero,
            'training_mean_prediction': baseline_mean,
            'training_median_prediction': baseline_median,
        }, f, indent=2)

    with open(OUT / 'SPATIAL_SPLIT.json', 'w', encoding='utf-8') as f:
        json.dump({
            'split_method': 'deterministic geographic partition within the real Uttarakhand valid mask',
            'train_bbox': train_bbox,
            'validation_bbox': val_bbox,
            'train_center': {'y': int((train_bbox[0] + train_bbox[2]) / 2), 'x': int((train_bbox[1] + train_bbox[3]) / 2)},
            'validation_center': {'y': int((val_bbox[0] + val_bbox[2]) / 2), 'x': int((val_bbox[1] + val_bbox[3]) / 2)},
            'overlap_pixels': 0,
            'valid_mask_shape': list(mask.shape),
        }, f, indent=2)

    with open(OUT / 'DIAGNOSTIC_TENSORS.json', 'w', encoding='utf-8') as f:
        json.dump({
            'train_crop': {
                'bbox': train_crop['bbox'],
                'valid_pixels': int(train_crop['mask'].sum()),
                'local_median_m': float(train_crop['local_median']),
                'local_relief_min_m': float((train_crop['dem'] - train_crop['local_median'])[train_crop['mask']].min()),
                'local_relief_max_m': float((train_crop['dem'] - train_crop['local_median'])[train_crop['mask']].max()),
                'local_relief_mean_m': float((train_crop['dem'] - train_crop['local_median'])[train_crop['mask']].mean()),
                'local_relief_std_m': float((train_crop['dem'] - train_crop['local_median'])[train_crop['mask']].std()),
            },
            'validation_crop': {
                'bbox': val_crop['bbox'],
                'valid_pixels': int(val_crop['mask'].sum()),
                'local_median_m': float(val_crop['local_median']),
                'local_relief_min_m': float((val_crop['dem'] - val_crop['local_median'])[val_crop['mask']].min()),
                'local_relief_max_m': float((val_crop['dem'] - val_crop['local_median'])[val_crop['mask']].max()),
                'local_relief_mean_m': float((val_crop['dem'] - val_crop['local_median'])[val_crop['mask']].mean()),
                'local_relief_std_m': float((val_crop['dem'] - val_crop['local_median'])[val_crop['mask']].std()),
            },
            'trained_prediction': {
                'prediction_min_m': float(pred[mask].min()),
                'prediction_max_m': float(pred[mask].max()),
                'prediction_mean_m': float(pred[mask].mean()),
                'prediction_std_m': float(pred[mask].std()),
                'true_min_m': float(target[mask].min()),
                'true_max_m': float(target[mask].max()),
                'true_mean_m': float(target[mask].mean()),
                'true_std_m': float(target[mask].std()),
                'mae_m': float(metrics['mae']),
                'rmse_m': float(metrics['rmse']),
                'pearson': float(metrics['pearson']) if np.isfinite(metrics['pearson']) else None,
                'mean_bias_m': float(metrics['mean_bias']),
                'error_std_m': float(metrics['error_std']),
            },
        }, f, indent=2)

    report = (
        '# Phase 77 in-distribution terrain learning control\n\n'
        '## Split\n'
        '- Data source: frozen Phase 72 Uttarakhand grid only.\n'
        '- Deterministic split: upper-left valid extent for training, lower-right valid extent for validation, with no overlap.\n'
        '- Train bbox: ' + str(train_bbox) + '\n'
        '- Validation bbox: ' + str(val_bbox) + '\n\n'
        '## Target proof\n'
        f"- Train crop local median: {float(train_crop['local_median']):.6f} m.\n"
        f"- Train crop local relief mean: {(train_crop['dem'] - train_crop['local_median'])[train_crop['mask']].mean():.6f} m, std: {(train_crop['dem'] - train_crop['local_median'])[train_crop['mask']].std():.6f} m.\n"
        f"- Validation crop local median: {float(val_crop['local_median']):.6f} m.\n"
        f"- Validation crop local relief mean: {(val_crop['dem'] - val_crop['local_median'])[val_crop['mask']].mean():.6f} m, std: {(val_crop['dem'] - val_crop['local_median'])[val_crop['mask']].std():.6f} m.\n\n"
        '## One-epoch held-out Uttarakhand validation\n'
        f"- MAE: {metrics['mae']:.6f} m\n"
        f"- RMSE: {metrics['rmse']:.6f} m\n"
        f"- Pearson: {metrics['pearson']:.6f} if finite else NaN\n"
        f"- Mean bias: {metrics['mean_bias']:.6f} m\n"
        f"- Prediction std: {metrics['prediction_std']:.6f} m\n"
        f"- Target std: {metrics['target_std']:.6f} m\n"
        f"- Error std: {metrics['error_std']:.6f} m\n"
        f"- Prediction min/max/mean: {float(pred[mask].min()):.6f} / {float(pred[mask].max()):.6f} / {float(pred[mask].mean()):.6f} m\n"
        f"- Target min/max/mean: {float(target[mask].min()):.6f} / {float(target[mask].max()):.6f} / {float(target[mask].mean()):.6f} m\n"
        f"- Gradient-x MAE: {grad_stats['gradient_x_mae']:.6f} m\n"
        f"- Gradient-y MAE: {grad_stats['gradient_y_mae']:.6f} m\n"
        f"- Mean gradient MAE: {grad_stats['gradient_mae_mean']:.6f} m\n\n"
        '## Baselines on the same held-out crop\n'
        f"- Zero prediction: MAE={baseline_zero['mae']:.6f} m, RMSE={baseline_zero['rmse']:.6f} m, Pearson={baseline_zero['pearson']}\n"
        f"- Training mean prediction: MAE={baseline_mean['mae']:.6f} m, RMSE={baseline_mean['rmse']:.6f} m, Pearson={baseline_mean['pearson']}\n"
        f"- Training median prediction: MAE={baseline_median['mae']:.6f} m, RMSE={baseline_median['rmse']:.6f} m, Pearson={baseline_median['pearson']}\n\n"
        '## Cross-region comparison\n'
        f"- Cross-region Phase 75: MAE={float(phase75_metrics['mae_m']):.6f} m, RMSE={float(phase75_metrics['rmse_m']):.6f} m, Pearson={float(phase75_metrics['pearson']):.6f}\n"
        f"- In-distribution Phase 77: MAE={float(metrics['mae']):.6f} m, RMSE={float(metrics['rmse']):.6f} m, Pearson={float(metrics['pearson']) if np.isfinite(metrics['pearson']) else float('nan')}\n\n"
        '## Decision\n'
        f"{decision}\n"
    )
    (OUT / 'REPORT.md').write_text(report + '\n', encoding='utf-8')
    print(json.dumps({'decision': decision, 'metrics': metrics, 'baseline_zero': baseline_zero, 'baseline_mean': baseline_mean, 'baseline_median': baseline_median}, indent=2))


if __name__ == '__main__':
    main()
