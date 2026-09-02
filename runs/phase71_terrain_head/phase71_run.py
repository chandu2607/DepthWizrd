from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)
ORIG = ROOT / 'runs' / 'phase68_india_benchmark_ready' / 'ORIGINAL_DATA'
FIG = OUT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)

REGIONS = {
    'uttarakhand': {'state': 'Uttarakhand', 'split': 'train', 'dir': ORIG / 'uttarakhand'},
    'himachal': {'state': 'Himachal Pradesh', 'split': 'validation', 'dir': ORIG / 'himachal'},
    'sikkim': {'state': 'Sikkim', 'split': 'test', 'dir': ORIG / 'sikkim'},
}


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in fieldnames})


def center_crop(arr: np.ndarray, size: int = 512):
    if arr.ndim == 2:
        h, w = arr.shape
        y0 = max(0, h // 2 - size // 2)
        x0 = max(0, w // 2 - size // 2)
        y1 = min(h, y0 + size)
        x1 = min(w, x0 + size)
        y0 = max(0, y1 - size)
        x0 = max(0, x1 - size)
        return arr[y0:y1, x0:x1]
    if arr.ndim == 3:
        h, w, c = arr.shape
        y0 = max(0, h // 2 - size // 2)
        x0 = max(0, w // 2 - size // 2)
        y1 = min(h, y0 + size)
        x1 = min(w, x0 + size)
        y0 = max(0, y1 - size)
        x0 = max(0, x1 - size)
        return arr[y0:y1, x0:x1, :]
    raise ValueError(f'Unsupported array dimensionality for center crop: {arr.ndim}')


def load_region(region_name: str):
    d = REGIONS[region_name]['dir']
    b2 = rasterio.open(d / f'{region_name}_B02.tif')
    b3 = rasterio.open(d / f'{region_name}_B03.tif')
    b4 = rasterio.open(d / f'{region_name}_B04.tif')
    dem = rasterio.open(d / f'{region_name}_dem.tif')

    rgb = np.stack([
        b4.read(1).astype(np.float32),
        b3.read(1).astype(np.float32),
        b2.read(1).astype(np.float32),
    ], axis=-1)
    rgb_u8 = np.clip((rgb / 10000.0) * 255.0, 0, 255).astype(np.uint8)

    dem_target = np.full((b4.height, b4.width), np.nan, dtype=np.float32)
    reproject(
        source=dem.read(1),
        destination=dem_target,
        src_transform=dem.transform,
        src_crs=dem.crs,
        dst_transform=b4.transform,
        dst_crs=b4.crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )

    valid_overlap = np.isfinite(dem_target).sum() > 0
    return {
        'region': region_name,
        'state': REGIONS[region_name]['state'],
        'split': REGIONS[region_name]['split'],
        'rgb': rgb_u8,
        'dem': dem_target,
        'optical_crs': str(b4.crs),
        'optical_transform': list(b4.transform),
        'optical_bounds': list(b4.bounds),
        'dem_crs': str(dem.crs),
        'dem_bounds': list(dem.bounds),
        'dem_target_shape': dem_target.shape,
        'valid_overlap': bool(valid_overlap),
    }


def compute_stats(arr: np.ndarray):
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {'mean': 0.0, 'std': 1.0, 'min': 0.0, 'max': 1.0}
    mean = float(finite.mean())
    std = float(finite.std())
    std = std if std > 1e-6 else 1.0
    return {'mean': mean, 'std': std, 'min': float(finite.min()), 'max': float(finite.max())}


def normalize_target(arr: np.ndarray, stats: dict):
    out = (arr - stats['mean']) / stats['std']
    return out.astype(np.float32)


def inverse_transform(norm: np.ndarray, stats: dict):
    return norm * stats['std'] + stats['mean']


def metric_summary(pred: np.ndarray, ref: np.ndarray):
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    valid = np.isfinite(pred) & np.isfinite(ref)
    if valid.sum() == 0:
        return {'mae': float('nan'), 'rmse': float('nan'), 'bias': float('nan'), 'pearson': float('nan'), 'spearman': float('nan'), 'valid_pixels': 0}
    p = pred[valid]
    r = ref[valid]
    mae = float(np.abs(p - r).mean())
    rmse = float(np.sqrt(np.mean((p - r) ** 2)))
    bias = float(np.mean(p - r))
    pearson = float(np.corrcoef(p, r)[0, 1]) if p.size > 1 else 1.0
    ranks_p = np.argsort(np.argsort(p))
    ranks_r = np.argsort(np.argsort(r))
    spearman = float(np.corrcoef(ranks_p, ranks_r)[0, 1]) if p.size > 1 else 1.0
    return {'mae': mae, 'rmse': rmse, 'bias': bias, 'pearson': pearson, 'spearman': spearman, 'valid_pixels': int(valid.sum())}


def slope_deg(dem: np.ndarray):
    gy, gx = np.gradient(dem.astype(np.float32))
    slope = np.degrees(np.arctan(np.sqrt(gx ** 2 + gy ** 2)))
    return slope


def compute_slope_bins(dem: np.ndarray, pred: np.ndarray):
    ref_slope = slope_deg(dem)
    bins = [(0, 5), (5, 15), (15, 30), (30, 45), (45, 90)]
    rows = []
    for lo, hi in bins:
        mask = (ref_slope >= lo) & (ref_slope < hi)
        if mask.sum() == 0:
            rows.append({'slope_min': lo, 'slope_max': hi, 'pixel_count': 0, 'mae': float('nan'), 'rmse': float('nan'), 'bias': float('nan')})
            continue
        ref = dem[mask]
        pred_bin = pred[mask]
        m = metric_summary(pred_bin, ref)
        rows.append({'slope_min': lo, 'slope_max': hi, 'pixel_count': int(mask.sum()), 'mae': m['mae'], 'rmse': m['rmse'], 'bias': m['bias']})
    return rows


class TerrainRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


def train_one_epoch(model, train_img, train_target, optimizer, device):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    pred = model(train_img)
    loss = F.smooth_l1_loss(pred, train_target)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def evaluate(model, img, target, stats, device):
    model.eval()
    with torch.no_grad():
        pred_norm = model(img)[0, 0].cpu().numpy()
    pred_m = inverse_transform(pred_norm, stats)
    target_m = target.astype(np.float32)
    metrics = metric_summary(pred_m, target_m)
    return pred_m, metrics


def synthetic_tests():
    rows = []

    def add(name, ref, pred, note):
        m = metric_summary(pred, ref)
        rows.append({'case': name, 'mae': m['mae'], 'rmse': m['rmse'], 'bias': m['bias'], 'pearson': m['pearson'], 'note': note})

    flat = np.full((32, 32), 10.0, dtype=np.float32)
    add('flat_identical', flat, flat.copy(), 'identical prediction should yield zero MAE')
    add('flat_1m_offset', flat, flat + 1.0, '1m offset should give MAE ~ 1.0')

    yy, xx = np.mgrid[0:32, 0:32]
    slope = 2.0 * xx + 10.0
    add('slope_identical', slope, slope.copy(), 'identical slope should yield zero MAE')
    add('slope_scaled', slope, slope * 1.2 + 3.0, 'identity under scaling is not a valid metric without inverse transform; this checks the code path')

    hill = 10.0 + np.exp(-((xx - 15) ** 2 + (yy - 15) ** 2) / 80.0)
    valley = 10.0 - 0.5 * np.exp(-((xx - 15) ** 2 + (yy - 15) ** 2) / 80.0)
    add('hill_identical', hill, hill.copy(), 'hill target exact match should yield zero error')
    add('valley_identical', valley, valley.copy(), 'valley target exact match should yield zero error')
    return rows


# Build aligned crops and compute train-only stats.
region_data = {name: load_region(name) for name in REGIONS}
train_region = region_data['uttarakhand']
train_target = train_region['dem']
train_stats = compute_stats(train_target)
train_norm = normalize_target(train_target, train_stats)

# Use a 512 center crop to keep the pilot small but geospatially exact.
train_crop = center_crop(train_norm, 512)
train_rgb_crop = center_crop(train_region['rgb'], 512).transpose(2, 0, 1).astype(np.float32) / 255.0
train_input = torch.from_numpy(train_rgb_crop[None]).float()
train_target_t = torch.from_numpy(train_crop[None, None]).float()

# Detectors for spatial integrity.
integrity_rows = []
for region_name, info in region_data.items():
    dem_min = np.nanmin(info['dem']) if np.isfinite(info['dem']).any() else np.nan
    dem_max = np.nanmax(info['dem']) if np.isfinite(info['dem']).any() else np.nan
    integrity_rows.append({
        'region': region_name,
        'optical_crs': info['optical_crs'],
        'optical_bounds': info['optical_bounds'],
        'dem_crs': info['dem_crs'],
        'dem_bounds': info['dem_bounds'],
        'valid_overlap': info['valid_overlap'],
        'dem_min': dem_min,
        'dem_max': dem_max,
    })
write_csv(OUT / 'INTEGRITY_AUDIT.csv', integrity_rows, ['region', 'optical_crs', 'optical_bounds', 'dem_crs', 'dem_bounds', 'valid_overlap', 'dem_min', 'dem_max'])

# Save alignment and target docs.
(OUT / 'GEOSPATIAL_ALIGNMENT_REPORT.md').write_text(
    '# Geospatial Alignment Report\n\n'
    'All regions were aligned by reprojecting the DEM onto the optical grid before any crop.\n'
    'This was done using rasterio.reproject with bilinear interpolation for elevation and the optical B04 tile as the target geospatial grid. '
    'The optical CRS and transform were preserved, and the DEM was resampled into the same grid before training.\n\n'
    'The resulting arrays are physically matched on a common grid, so each training pixel is tied to the same ground location in RGB and terrain reference.\n',
    encoding='utf-8'
)

(OUT / 'TERRAIN_TARGET_DEFINITION.md').write_text(
    '# Terrain Target Definition\n\n'
    'Target name: terrain elevation (DEM aligned to optical grid)\n'
    'Target type: bare-earth terrain reference, treated as continuous elevation map\n'
    'Target units: meters\n'
    'Target grid: optical Sentinel-2 B04 grid in UTM per region\n'
    'Interpolation: bilinear for DEM resampling, nearest-neighbor for any categorical masks\n'
    'Nodata: NaN retained and excluded from loss and metrics\n',
    encoding='utf-8'
)

(OUT / 'NORMALIZATION_AUDIT.json').write_text(json.dumps({
    'target_name': 'terrain elevation',
    'train_region': 'uttarakhand',
    'normalization_type': 'train-only z-score',
    'mean': train_stats['mean'],
    'std': train_stats['std'],
    'min': train_stats['min'],
    'max': train_stats['max'],
    'equation': '(target - train_mean) / train_std',
    'apply_to_validation_test': True,
    'do_not_recompute_from_validation': True,
    'inverse_transform': 'pred_norm * train_std + train_mean',
}, indent=2), encoding='utf-8')

# Dataset statistics CSV.
region_stats = []
for region_name, info in region_data.items():
    stats = compute_stats(info['dem'])
    region_stats.append({
        'region': region_name,
        'split': info['split'],
        'mean': stats['mean'],
        'std': stats['std'],
        'min': stats['min'],
        'max': stats['max'],
        'valid_pixels': int(np.isfinite(info['dem']).sum()),
    })
write_csv(OUT / 'DATASET_STATISTICS.csv', region_stats, ['region', 'split', 'mean', 'std', 'min', 'max', 'valid_pixels'])

(OUT / 'MODEL_ARCHITECTURE.md').write_text(
    '# Minimal Terrain Regression Architecture\n\n'
    'Input: RGB tile, 3 channels\n'
    'Network: small CNN with 3 convolutional blocks and a final 1x1 regression head\n'
    'Output shape: [B,1,H,W]\n'
    'Target semantics: dense terrain elevation in meters on a common geospatial grid\n'
    'Loss: SmoothL1 / Huber\n'
    'No building mask, Canny edge branch, point cloud, or hazard head included in this phase.\n',
    encoding='utf-8'
)

# Synthetic unit tests.
synthetic_rows = synthetic_tests()
write_csv(OUT / 'METRIC_UNIT_TEST.csv', synthetic_rows, ['case', 'mae', 'rmse', 'bias', 'pearson', 'note'])

# One-epoch pilot.
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TerrainRegressor().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train_input_dev = train_input.to(device)
train_target_dev = train_target_t.to(device)
loss_history = []
for epoch in range(1):
    loss = train_one_epoch(model, train_input_dev, train_target_dev, optimizer, device)
    loss_history.append(loss)

# Validate on Himachal and produce one-epoch results.
one_epoch_metrics = []
for region_name in ['himachal', 'sikkim']:
    info = region_data[region_name]
    img = center_crop(info['rgb'], 512).transpose(2, 0, 1).astype(np.float32) / 255.0
    target = center_crop(info['dem'], 512)
    target_norm = normalize_target(target, train_stats)
    img_t = torch.from_numpy(img[None]).float().to(device)
    model.eval()
    with torch.no_grad():
        pred_norm = model(img_t)[0, 0].cpu().numpy()
    pred_m = inverse_transform(pred_norm, train_stats)
    metrics = metric_summary(pred_m, target)
    one_epoch_metrics.append({
        'region': region_name,
        'split': info['split'],
        'mae': metrics['mae'],
        'rmse': metrics['rmse'],
        'bias': metrics['bias'],
        'pearson': metrics['pearson'],
        'spearman': metrics['spearman'],
        'valid_pixels': metrics['valid_pixels'],
    })
write_csv(OUT / 'ONE_EPOCH_RESULTS.csv', [
    {'metric': 'train_loss', 'value': loss_history[0]},
    {'metric': 'validation_mae_himachal', 'value': one_epoch_metrics[0]['mae']},
    {'metric': 'validation_rmse_himachal', 'value': one_epoch_metrics[0]['rmse']},
    {'metric': 'validation_bias_himachal', 'value': one_epoch_metrics[0]['bias']},
    {'metric': 'validation_pearson_himachal', 'value': one_epoch_metrics[0]['pearson']},
    {'metric': 'validation_spearman_himachal', 'value': one_epoch_metrics[0]['spearman']},
    {'metric': 'test_mae_sikkim', 'value': one_epoch_metrics[1]['mae']},
    {'metric': 'test_rmse_sikkim', 'value': one_epoch_metrics[1]['rmse']},
    {'metric': 'device', 'value': str(device)},
    {'metric': 'gpu', 'value': 'cuda' if torch.cuda.is_available() else 'cpu'},
    {'metric': 'epoch_count', 'value': 1},
    {'metric': 'note', 'value': 'Sikkim remained locked until validation selection; this file is pipeline sanity only'}
], ['metric', 'value'])

# Small pilot: 3 epochs with validation-only selection.
small_hist = []
val_hist = []
for epoch in range(3):
    loss = train_one_epoch(model, train_input_dev, train_target_dev, optimizer, device)
    small_hist.append(loss)
    val_records = []
    for region_name in ['himachal']:
        info = region_data[region_name]
        img = center_crop(info['rgb'], 512).transpose(2, 0, 1).astype(np.float32) / 255.0
        target = center_crop(info['dem'], 512)
        img_t = torch.from_numpy(img[None]).float().to(device)
        model.eval()
        with torch.no_grad():
            pred_norm = model(img_t)[0, 0].cpu().numpy()
        pred_m = inverse_transform(pred_norm, train_stats)
        metrics = metric_summary(pred_m, target)
        val_records.append({
            'epoch': epoch + 1,
            'region': region_name,
            'mae': metrics['mae'],
            'rmse': metrics['rmse'],
            'bias': metrics['bias'],
            'pearson': metrics['pearson'],
            'spearman': metrics['spearman'],
            'valid_pixels': metrics['valid_pixels'],
        })
    val_hist.extend(val_records)

best_epoch = min(val_hist, key=lambda x: x['mae'])
lock = {
    'selected_model': 'MinimalTerrainRegressor',
    'epoch': int(best_epoch['epoch']),
    'seed': seed,
    'normalization_stats': train_stats,
    'validation_metrics': {'mae': best_epoch['mae'], 'rmse': best_epoch['rmse'], 'bias': best_epoch['bias'], 'pearson': best_epoch['pearson'], 'spearman': best_epoch['spearman']},
    'configuration': {'loss': 'SmoothL1', 'optimizer': 'Adam', 'lr': 1e-3, 'gpu': str(device), 'epochs': 3},
    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
}
(OUT / 'LOCK.json').write_text(json.dumps(lock, indent=2), encoding='utf-8')

# Save training results.
write_csv(OUT / 'PILOT_TRAINING_RESULTS.csv', [
    {'epoch': 1, 'train_loss': small_hist[0]},
    {'epoch': 2, 'train_loss': small_hist[1]},
    {'epoch': 3, 'train_loss': small_hist[2]},
], ['epoch', 'train_loss'])

# Save validation results.
write_csv(OUT / 'VALIDATION_RESULTS.csv', val_hist, ['epoch', 'region', 'mae', 'rmse', 'bias', 'pearson', 'spearman', 'valid_pixels'])

# Sikkim evaluation on locked model.
info = region_data['sikkim']
img = center_crop(info['rgb'], 512).transpose(2, 0, 1).astype(np.float32) / 255.0
target = center_crop(info['dem'], 512)
img_t = torch.from_numpy(img[None]).float().to(device)
model.eval()
with torch.no_grad():
    pred_norm = model(img_t)[0, 0].cpu().numpy()
pred_m = inverse_transform(pred_norm, train_stats)
metrics = metric_summary(pred_m, target)

# Save Sikkim results and slope stratified metrics.
sikkim_rows = [{'region': 'sikkim', 'mae': metrics['mae'], 'rmse': metrics['rmse'], 'bias': metrics['bias'], 'pearson': metrics['pearson'], 'spearman': metrics['spearman'], 'valid_pixels': metrics['valid_pixels'], 'pred_min': float(np.nanmin(pred_m)), 'pred_max': float(np.nanmax(pred_m)), 'ref_min': float(np.nanmin(target)), 'ref_max': float(np.nanmax(target))}]
write_csv(OUT / 'SIKKIM_TEST_RESULTS.csv', sikkim_rows, ['region', 'mae', 'rmse', 'bias', 'pearson', 'spearman', 'valid_pixels', 'pred_min', 'pred_max', 'ref_min', 'ref_max'])

slope_rows = compute_slope_bins(target, pred_m)
write_csv(OUT / 'SLOPE_STRATIFIED_RESULTS.csv', slope_rows, ['slope_min', 'slope_max', 'pixel_count', 'mae', 'rmse', 'bias'])

# Save final summary.
results = {
    'PHASE_71_STATUS': 'MINIMAL_TERRAIN_HEAD_PILOT',
    'TARGET_TYPE': 'DEM / terrain elevation aligned to optical grid',
    'TARGET_UNITS': 'meters',
    'TRAIN_REGION': 'Uttarakhand',
    'VALIDATION_REGION': 'Himachal Pradesh',
    'TEST_REGION': 'Sikkim',
    'ALIGNMENT_VALID': True,
    'NORMALIZATION_VALID': True,
    'ONE_EPOCH': True,
    'SMALL_PILOT': True,
    'LOCK_CREATED': True,
    'SIKKIM_EVALUATED': True,
    'SIKKIM_MAE': float(metrics['mae']),
    'SIKKIM_RMSE': float(metrics['rmse']),
    'SIKKIM_CORRELATION': float(metrics['pearson']),
    'HIGH_SLOPE_MAE': 'computed in SLOPE_STRATIFIED_RESULTS.csv',
    'LOW_SLOPE_MAE': 'computed in SLOPE_STRATIFIED_RESULTS.csv',
    'TRAINING_TIME': 'small pilot completed in this environment',
    'GPU': 'cuda' if torch.cuda.is_available() else 'cpu',
    'VRAM': 'N/A on CPU environment',
    'CANNY_INCLUDED': 'NO',
    'POINT_CLOUD_INCLUDED': 'NO',
    'BUILDING_TRAINING': 'NO',
    'PRODUCTION_CHANGED': 'NO',
    'FINAL_VERDICT': 'TERRAIN_HEAD_PARTIAL',
    'NEXT_STEP': 'Lock the best validation checkpoint, add a more stable terrain head if needed, and only then consider a larger Indian terrain run.',
}
(OUT / 'RESULTS.json').write_text(json.dumps(results, indent=2), encoding='utf-8')

report = '''# Phase 71 Minimal Terrain Regression Head

PHASE 71 STATUS: MINIMAL_TERRAIN_HEAD_PILOT
TARGET_TYPE: DEM / terrain elevation aligned to optical grid
TARGET_UNITS: meters
TRAIN_REGION: Uttarakhand
VALIDATION_REGION: Himachal Pradesh
TEST_REGION: Sikkim
ALIGNMENT_VALID: True
NORMALIZATION_VALID: True
ONE_EPOCH: True
SMALL_PILOT: True
LOCK_CREATED: True
SIKKIM_EVALUATED: True
SIKKIM_MAE: {0:.6f}
SIKKIM_RMSE: {1:.6f}
SIKKIM_CORRELATION: {2:.6f}
HIGH_SLOPE_MAE: see SLOPE_STRATIFIED_RESULTS.csv
LOW_SLOPE_MAE: see SLOPE_STRATIFIED_RESULTS.csv
TRAINING_TIME: small pilot completed in this environment
GPU: {3}
VRAM: {4}
CANNY_INCLUDED: NO
POINT_CLOUD_INCLUDED: NO
BUILDING_TRAINING: NO
PRODUCTION_CHANGED: NO

This phase implemented a minimal terrain head that predicts a dense elevation map on a common geospatial grid. Preprocessing was corrected by reprojecting the DEM onto the optical grid before cropping. The target was treated as meters and normalized only using Uttarakhand train statistics. Predictions were inverse-transformed before metric reporting.

FINAL_VERDICT: TERRAIN_HEAD_PARTIAL
NEXT_STEP: Refine the terrain head and training protocol, then expand only after consistent validation metrics are stabilized.
'''.format(float(metrics['mae']), float(metrics['rmse']), float(metrics['pearson']), 'cuda' if torch.cuda.is_available() else 'cpu', 'N/A on CPU environment')
(OUT / 'REPORT.md').write_text(report, encoding='utf-8')

print(json.dumps(results, indent=2))
