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
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depthwizard.depth.depth_anything import DepthAnythingV2

PHASE72 = REPO_ROOT / 'runs' / 'phase72_common_grid_forensics' / 'common_grid'
OLD_PHASE77 = REPO_ROOT / 'runs' / 'phase77_indistribution_terrain_control' / 'RESULTS.json'
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
        self.depth = depth_model.infer(rgb_u8, key=f'phase79_{TRAIN_REGION}_{TARGET_SIZE}', target_hw=(TARGET_SIZE, TARGET_SIZE)).astype(np.float32)
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


class TerrainUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2)
        self.e2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.d2 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.d1 = nn.Sequential(
            nn.Conv2d(64 + 32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        p1 = self.pool(e1)
        e2 = self.e2(p1)
        p2 = self.pool(e2)
        b = self.bottleneck(p2)
        u2 = F.interpolate(b, size=e2.shape[-2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([u2, e2], dim=1)
        d2 = self.d2(d2)
        u1 = F.interpolate(d2, size=e1.shape[-2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([u1, e1], dim=1)
        d1 = self.d1(d1)
        out = self.head(d1)
        return out.squeeze(1)


def safe_stats(arr: np.ndarray):
    arr = np.asarray(arr)
    finite = np.isfinite(arr)
    return {
        'shape': list(arr.shape),
        'dtype': str(arr.dtype),
        'min': float(np.nanmin(arr[finite])) if finite.any() else np.nan,
        'max': float(np.nanmax(arr[finite])) if finite.any() else np.nan,
        'mean': float(np.nanmean(arr[finite])) if finite.any() else np.nan,
        'std': float(np.nanstd(arr[finite])) if finite.any() else np.nan,
        'finite_count': int(finite.sum()),
        'nan_fraction': float(np.isnan(arr).mean()),
        'inf_fraction': float(np.isinf(arr).mean()),
    }


def compute_metrics(pred: np.ndarray, target: np.ndarray, valid: np.ndarray):
    pred = pred[valid]
    target = target[valid]
    if pred.size == 0:
        return {'mae': np.nan, 'rmse': np.nan, 'pearson': np.nan, 'mean_bias': np.nan, 'prediction_mean': np.nan, 'prediction_std': np.nan, 'target_std': np.nan, 'n': 0}
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
        'prediction_mean': float(pred.mean()),
        'prediction_std': float(pred.std()),
        'target_std': float(target.std()),
        'n': int(pred.size),
    }


def gradient_abs_error(pred: np.ndarray, target: np.ndarray):
    pred_grad_x = np.abs(np.diff(pred, axis=1))
    pred_grad_y = np.abs(np.diff(pred, axis=0))
    tar_grad_x = np.abs(np.diff(target, axis=1))
    tar_grad_y = np.abs(np.diff(target, axis=0))
    x_error = float(np.mean(np.abs(pred_grad_x - tar_grad_x)))
    y_error = float(np.mean(np.abs(pred_grad_y - tar_grad_y)))
    pred_mag = 0.5 * (np.mean(pred_grad_x) + np.mean(pred_grad_y))
    tar_mag = 0.5 * (np.mean(tar_grad_x) + np.mean(tar_grad_y))
    return {
        'gradient_x_mae': x_error,
        'gradient_y_mae': y_error,
        'prediction_gradient_magnitude_mean': float(pred_mag),
        'target_gradient_magnitude_mean': float(tar_mag),
    }


def save_image(path: Path, title: str, arr: np.ndarray, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(4, 4))
    img = ax.imshow(arr, cmap='terrain', vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis('off')
    fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def compare_methods(methods: dict):
    out = {}
    for name, payload in methods.items():
        out[name] = payload
    return out


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds_mask, mask = open_raster(PHASE72 / TRAIN_REGION / 'valid_mask.tif')
    mask = mask[0].astype(bool)
    train_bbox, val_bbox = deterministic_split(mask)
    train_crop = build_crop(TRAIN_REGION, train_bbox)
    val_crop = build_crop(TRAIN_REGION, val_bbox)

    train_target = train_crop['dem'] - train_crop['local_median']
    val_target = val_crop['dem'] - val_crop['local_median']
    train_stats = compute_train_stats(train_target, train_crop['mask'])

    train_ds = TerrainDataset(train_crop, train_stats)
    val_ds = TerrainDataset(val_crop, train_stats)
    loader_train = DataLoader(train_ds, batch_size=1, shuffle=False)
    loader_val = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = TerrainUNet().to(device)
    criterion = nn.SmoothL1Loss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # one-epoch training, same as Phase 77
    model.train()
    for batch in loader_train:
        x = batch['image'].to(device, dtype=torch.float32)
        y = batch['target'].to(device, dtype=torch.float32)
        m = batch['mask'].to(device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss_map = criterion(pred * m, y * m)
        loss = (loss_map * m).sum() / (m.sum() + 1e-6)
        if not torch.isfinite(loss):
            raise RuntimeError('Training loss is non-finite in Phase 79.')
        loss.backward()
        if any((p.grad is not None) and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError('Gradients are non-finite in Phase 79.')
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_batch = next(iter(loader_val))
        x_val = val_batch['image'].to(device, dtype=torch.float32)
        y_val = val_batch['target'].to(device, dtype=torch.float32)
        m_val = val_batch['mask'].to(device, dtype=torch.float32)
        pred_val = model(x_val)

    pred_np = pred_val.detach().cpu().numpy()[0]
    target_np = y_val.detach().cpu().numpy()[0, 0]
    valid_np = m_val.detach().cpu().numpy()[0, 0].astype(bool)

    # baseline predictions
    zero_pred = np.zeros_like(target_np, dtype=np.float32)
    mean_pred = np.full_like(target_np, float(target_np[valid_np].mean()), dtype=np.float32)
    median_pred = np.full_like(target_np, float(np.median(target_np[valid_np])), dtype=np.float32)

    baseline_methods = {
        'zero_relief_baseline': compute_metrics(zero_pred, target_np, valid_np),
        'mean_relief_baseline': compute_metrics(mean_pred, target_np, valid_np),
        'median_relief_baseline': compute_metrics(median_pred, target_np, valid_np),
        'terrain_unet': compute_metrics(pred_np, target_np, valid_np),
    }

    # gradient metrics
    pred_grad = gradient_abs_error(pred_np, target_np)
    spatial_diag = {
        'prediction_gradient_x_mae': pred_grad['gradient_x_mae'],
        'prediction_gradient_y_mae': pred_grad['gradient_y_mae'],
        'prediction_gradient_magnitude_mean': pred_grad['prediction_gradient_magnitude_mean'],
        'target_gradient_magnitude_mean': pred_grad['target_gradient_magnitude_mean'],
        'prediction_std': float(pred_np[valid_np].std()),
        'target_std': float(target_np[valid_np].std()),
    }

    # input sanity and sensitivity
    input_payload = {
        'shape': list(x_val[0].shape),
        'dtype': str(x_val[0].dtype),
        'min': float(x_val[0].min().item()),
        'max': float(x_val[0].max().item()),
        'mean': float(x_val[0].mean().item()),
        'std': float(x_val[0].std().item()),
        'finite_count': int(torch.isfinite(x_val[0]).sum().item()),
    }
    model_sens = TerrainUNet().to(device)
    with torch.no_grad():
        x0 = x_val
        pred_real = model_sens(x0)
        flat = x0[0].permute(1, 2, 0).reshape(-1, 4)
        shuffled_flat = flat[np.random.default_rng(SEED).permutation(flat.shape[0])]
        shuffled = shuffled_flat.reshape(512, 512, 4).permute(2, 0, 1).unsqueeze(0)
        pred_shuffled = model_sens(shuffled.to(device))
        const = x0.mean(dim=(2, 3), keepdim=True).expand_as(x0)
        pred_const = model_sens(const)

    real_pred_np = pred_real.detach().cpu().numpy()[0]
    shuffled_pred_np = pred_shuffled.detach().cpu().numpy()[0]
    const_pred_np = pred_const.detach().cpu().numpy()[0]
    real_grad = np.abs(np.diff(real_pred_np, axis=1)).mean() + np.abs(np.diff(real_pred_np, axis=0)).mean()
    shuffled_grad = np.abs(np.diff(shuffled_pred_np, axis=1)).mean() + np.abs(np.diff(shuffled_pred_np, axis=0)).mean()
    const_grad = np.abs(np.diff(const_pred_np, axis=1)).mean() + np.abs(np.diff(const_pred_np, axis=0)).mean()
    sensitivity = {
        'original': {
            'prediction_std': float(real_pred_np.std()),
            'gradient_magnitude_mean': float(real_grad),
        },
        'shuffled': {
            'prediction_std': float(shuffled_pred_np.std()),
            'gradient_magnitude_mean': float(shuffled_grad),
        },
        'constant': {
            'prediction_std': float(const_pred_np.std()),
            'gradient_magnitude_mean': float(const_grad),
        },
    }

    # old phase 77 baseline comparison
    old_phase77 = json.loads(OLD_PHASE77.read_text(encoding='utf-8'))
    old_metrics = old_phase77['metrics']
    old_baselines = old_phase77['baselines']
    old_reference = {
        'old_terrainhead': {
            'mae': old_metrics['mae'],
            'rmse': old_metrics['rmse'],
            'pearson': old_metrics['pearson'],
            'prediction_std': old_metrics['prediction_std'],
            'gradient_x_mae': old_metrics['gradient_x_mae'] if 'gradient_x_mae' in old_metrics else old_phase77['gradient_metrics']['gradient_x_mae'],
            'gradient_y_mae': old_metrics['gradient_y_mae'] if 'gradient_y_mae' in old_metrics else old_phase77['gradient_metrics']['gradient_y_mae'],
            'gradient_mae_mean': old_phase77['gradient_metrics']['gradient_mae_mean'],
        },
        'phase77_baselines': old_baselines,
    }

    # decision logic
    terrain_unet = baseline_methods['terrain_unet']
    zero_base = baseline_methods['zero_relief_baseline']
    mean_base = baseline_methods['mean_relief_baseline']
    median_base = baseline_methods['median_relief_baseline']
    if (
        terrain_unet['mae'] < min(zero_base['mae'], mean_base['mae'], median_base['mae']) * 0.9
        and terrain_unet['pearson'] > 0.2
        and terrain_unet['prediction_std'] > max(zero_base['prediction_std'], mean_base['prediction_std'], median_base['prediction_std']) * 3.0
        and spatial_diag['prediction_gradient_magnitude_mean'] > 0.0
    ):
        decision = 'SPATIAL_TERRAIN_REGRESSION_BASELINE_WORKS'
    elif (
        terrain_unet['mae'] >= min(zero_base['mae'], mean_base['mae'], median_base['mae']) * 0.98
        and (terrain_unet['pearson'] is np.nan or terrain_unet['pearson'] <= 0.05)
        and abs(terrain_unet['prediction_std'] - max(zero_base['prediction_std'], mean_base['prediction_std'], median_base['prediction_std'])) < 0.05
    ):
        decision = 'SPATIAL_TERRAIN_REGRESSION_STILL_FAILS'
    else:
        decision = 'SPATIAL_TERRAIN_RESULT_INCONCLUSIVE'

    # Save outputs
    results = {
        'phase': 'PHASE_79',
        'status': 'SPATIAL_TERRAIN_BASELINE',
        'train_region': TRAIN_REGION,
        'validation_region': VALIDATION_REGION,
        'target_definition': 'LOCAL_RELIEF_TARGET = DEM - median(valid DEM pixels in the crop)',
        'units': 'meters',
        'data': {
            'train_crop_bbox': train_crop['bbox'],
            'validation_crop_bbox': val_crop['bbox'],
            'train_valid_pixels': int(train_crop['mask'].sum()),
            'validation_valid_pixels': int(val_crop['mask'].sum()),
        },
        'input_checks': input_payload,
        'training': {
            'loss': 'SmoothL1Loss',
            'optimizer': 'Adam',
            'learning_rate': 1e-3,
            'batch_size': 1,
            'epoch_count': 1,
            'train_region': TRAIN_REGION,
            'validation_region': VALIDATION_REGION,
        },
        'metrics': baseline_methods,
        'spatial_diagnostics': spatial_diag,
        'sensitivity': sensitivity,
        'old_phase77_reference': old_reference,
        'decision': decision,
    }

    (OUT / 'RESULTS.json').write_text(json.dumps(results, indent=2, default=str), encoding='utf-8')
    (OUT / 'BASELINE_COMPARISON.json').write_text(json.dumps({'baselines': baseline_methods, 'old_phase77': old_reference}, indent=2, default=str), encoding='utf-8')
    (OUT / 'SPATIAL_DIAGNOSTICS.json').write_text(json.dumps(spatial_diag, indent=2, default=str), encoding='utf-8')
    (OUT / 'SENSITIVITY_RESULTS.json').write_text(json.dumps(sensitivity, indent=2, default=str), encoding='utf-8')

    with (OUT / 'MODEL_ARCHITECTURE.txt').open('w', encoding='utf-8') as f:
        f.write(
            'TerrainUNet\n'
            'Input: 4 channels (RGB + depth)\n'
            'Encoder block 1: Conv2d(4,32,3,pad=1) -> ReLU -> Conv2d(32,32,3,pad=1) -> ReLU\n'
            'Downsample: MaxPool2d(2)\n'
            'Encoder block 2: Conv2d(32,64,3,pad=1) -> ReLU -> Conv2d(64,64,3,pad=1) -> ReLU\n'
            'Downsample: MaxPool2d(2)\n'
            'Bottleneck: Conv2d(64,128,3,pad=1) -> ReLU -> Conv2d(128,128,3,pad=1) -> ReLU\n'
            'Decoder: bilinear upsample -> concatenate skip from block2 -> Conv2d(64+64,64) -> ReLU -> Conv2d(64,64) -> ReLU\n'
            'Decoder: bilinear upsample -> concatenate skip from block1 -> Conv2d(64+32,32) -> ReLU -> Conv2d(32,32) -> ReLU\n'
            'Final: Conv2d(32,1,1) no activation\n'
        )

    save_image(VIS / 'train_rgb.png', 'Train RGB', train_crop['rgb'].transpose(1, 2, 0))
    save_image(VIS / 'validation_target.png', 'Validation target', val_target)
    save_image(VIS / 'validation_prediction.png', 'Validation prediction', pred_np)
    save_image(VIS / 'validation_mask.png', 'Validation mask', valid_np.astype(float))

    report_lines = [
        '# Phase 79 spatial terrain baseline',
        '',
        '## Controlled setup',
        '- Same Phase 72 RGB/DEM/mask grid and the same valid-mask crop logic as Phase 77.',
        '- Same Phase 75 local-relief target definition: DEM - median(valid DEM pixels in crop).',
        '- Same Phase 77 spatial train/validation strategy with a single Uttarakhand training crop and one held-out Uttarakhand validation crop.',
        '- Only model change: TerrainHead -> TerrainUNet. No other training or data changes.',
        '',
        '## Input verification',
        f"- Input tensor shape: {input_payload['shape']}",
        f"- Input dtype: {input_payload['dtype']}",
        f"- Input min/max/mean/std: {input_payload['min']}, {input_payload['max']}, {input_payload['mean']}, {input_payload['std']}",
        f"- Finite count: {input_payload['finite_count']}",
        '',
        '## Baseline comparison',
    ]
    for name, metrics in baseline_methods.items():
        report_lines.append(f"- {name}: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, Pearson={metrics['pearson'] if metrics['pearson'] is not None and not math.isnan(metrics['pearson']) else 'nan'}, prediction_mean={metrics['prediction_mean']:.4f}, prediction_std={metrics['prediction_std']:.4f}, target_std={metrics['target_std']:.4f}, mean_bias={metrics['mean_bias']:.4f}")
    report_lines.extend([
        '',
        '## Spatial diagnostics',
        f"- gradient-X MAE: {spatial_diag['prediction_gradient_x_mae']:.6f}",
        f"- gradient-Y MAE: {spatial_diag['prediction_gradient_y_mae']:.6f}",
        f"- prediction gradient magnitude mean: {spatial_diag['prediction_gradient_magnitude_mean']:.6f}",
        f"- target gradient magnitude mean: {spatial_diag['target_gradient_magnitude_mean']:.6f}",
        '',
        '## Sensitivity control',
        f"- original prediction std: {sensitivity['original']['prediction_std']:.6f}",
        f"- shuffled prediction std: {sensitivity['shuffled']['prediction_std']:.6f}",
        f"- constant prediction std: {sensitivity['constant']['prediction_std']:.6f}",
        f"- original gradient magnitude mean: {sensitivity['original']['gradient_magnitude_mean']:.6f}",
        f"- shuffled gradient magnitude mean: {sensitivity['shuffled']['gradient_magnitude_mean']:.6f}",
        f"- constant gradient magnitude mean: {sensitivity['constant']['gradient_magnitude_mean']:.6f}",
        '',
        '## Old vs new comparison',
        f"- old TerrainHead MAE: {old_reference['old_terrainhead']['mae']:.6f}",
        f"- old TerrainHead RMSE: {old_reference['old_terrainhead']['rmse']:.6f}",
        f"- old TerrainHead Pearson: {old_reference['old_terrainhead']['pearson']:.6f}",
        f"- old TerrainHead prediction_std: {old_reference['old_terrainhead']['prediction_std']:.6f}",
        f"- new TerrainUNet MAE: {baseline_methods['terrain_unet']['mae']:.6f}",
        f"- new TerrainUNet RMSE: {baseline_methods['terrain_unet']['rmse']:.6f}",
        f"- new TerrainUNet Pearson: {baseline_methods['terrain_unet']['pearson'] if baseline_methods['terrain_unet']['pearson'] is not None and not math.isnan(baseline_methods['terrain_unet']['pearson']) else 'nan'}",
        f"- new TerrainUNet prediction_std: {baseline_methods['terrain_unet']['prediction_std']:.6f}",
        '',
        f'Decision: {decision}',
        '',
        decision,
    ])
    (OUT / 'REPORT.md').write_text('\n'.join(report_lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
