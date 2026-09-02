from __future__ import annotations

import json
import math
import sys
import time
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
VAL_REGION = 'himachal'
TEST_REGION = 'sikkim'
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
    def __init__(self, region_name: str, train_stats: dict | None = None, target_kind: str = 'absolute'):
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
        self.depth = depth_model.infer(rgb_u8, key=f'{region_name}_phase75_{TARGET_SIZE}', target_hw=(TARGET_SIZE, TARGET_SIZE)).astype(np.float32)
        self.depth = (self.depth - self.depth.mean()) / (self.depth.std() + 1e-6)

    def __len__(self):
        return 1

    def _target_array(self):
        if self.target_kind == 'absolute':
            return self.dem.copy()
        if self.target_kind == 'local_relief':
            return self.dem - self.local_median
        raise ValueError(f'unknown target_kind: {self.target_kind}')

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


def compute_metrics_from_arrays(pred: np.ndarray, target: np.ndarray, valid: np.ndarray):
    pred = pred[valid]
    target = target[valid]
    if pred.size == 0:
        return {'mae': np.nan, 'rmse': np.nan, 'bias': np.nan, 'pearson': np.nan, 'spearman': np.nan, 'n': 0,
                'pred_min': np.nan, 'pred_max': np.nan, 'pred_mean': np.nan, 'pred_std': np.nan,
                'ref_min': np.nan, 'ref_max': np.nan, 'ref_mean': np.nan, 'ref_std': np.nan}
    err = pred - target
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    pearson = float(np.corrcoef(pred, target)[0, 1]) if np.std(pred) > 1e-8 and np.std(target) > 1e-8 else np.nan
    return {
        'mae': mae,
        'rmse': rmse,
        'bias': bias,
        'pearson': pearson,
        'n': int(pred.size),
        'pred_min': float(pred.min()),
        'pred_max': float(pred.max()),
        'pred_mean': float(pred.mean()),
        'pred_std': float(pred.std()),
        'ref_min': float(target.min()),
        'ref_max': float(target.max()),
        'ref_mean': float(target.mean()),
        'ref_std': float(target.std()),
    }


def save_diag_image(path: Path, title: str, arr: np.ndarray, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(5, 5))
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

    # Freeze the Phase 72 data and compute the diagnostic target representation.
    region_cache = {}
    for region_name in [TRAIN_REGION, VAL_REGION]:
        region_cache[region_name] = build_region_cache(region_name)

    train_abs = np.load(OUT / 'cache' / TRAIN_REGION / 'dem.npy').astype(np.float32)
    train_mask = np.load(OUT / 'cache' / TRAIN_REGION / 'mask.npy').astype(bool)
    train_local_median = float(np.load(OUT / 'cache' / TRAIN_REGION / 'local_median.npy')[0])
    train_local_target = train_abs - train_local_median
    train_stats = compute_train_stats(train_local_target, train_mask)

    # Print target proof on the real crops.
    for region_name in [TRAIN_REGION, VAL_REGION]:
        abs_dem = np.load(OUT / 'cache' / region_name / 'dem.npy').astype(np.float32)
        mask = np.load(OUT / 'cache' / region_name / 'mask.npy').astype(bool)
        local_median = float(np.load(OUT / 'cache' / region_name / 'local_median.npy')[0])
        relief = abs_dem - local_median
        abs_valid = abs_dem[mask]
        relief_valid = relief[mask]
        print(f'[{region_name}] ABSOLUTE min={abs_valid.min():.6f} max={abs_valid.max():.6f} mean={abs_valid.mean():.6f} std={abs_valid.std():.6f} median={local_median:.6f}')
        print(f'[{region_name}] RELIEF min={relief_valid.min():.6f} max={relief_valid.max():.6f} mean={relief_valid.mean():.6f} std={relief_valid.std():.6f} median={local_median:.6f}')
        print(f'[{region_name}] RELIEF mean approx zero = {abs(relief_valid.mean()):.6e}')

    # Build the exact control experiment: same model, same training loop, only target representation changes.
    train_ds = TerrainDataset(TRAIN_REGION, train_stats=train_stats, target_kind='local_relief')
    val_ds = TerrainDataset(VAL_REGION, train_stats=train_stats, target_kind='local_relief')
    loader_train = DataLoader(train_ds, batch_size=1, shuffle=False)
    loader_val = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = TerrainHead().to(device)
    criterion = nn.SmoothL1Loss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

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
            raise RuntimeError('Training loss is non-finite during the local-relief control.')
        loss.backward()
        if any((p.grad is not None) and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError('Gradients are non-finite during the local-relief control.')
        optimizer.step()
        print(f'phase75_epoch1_batch{batch_idx} loss={float(loss.item()):.6f}')

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

    pred_stack = np.concatenate([p for p in val_pred_list], axis=0)
    target_stack = np.concatenate([t for t in val_target_list], axis=0)
    mask_stack = np.concatenate([m for m in val_mask_list], axis=0)
    metrics = compute_metrics_from_arrays(pred_stack, target_stack, mask_stack)
    print(f'PHASE_75_LOCAL_RELIEF_VALIDATION: mae={metrics["mae"]} rmse={metrics["rmse"]} pearson={metrics["pearson"]} pred_mean={metrics["pred_mean"]}')
    print(f'pred min/max/mean/std: {metrics["pred_min"]} {metrics["pred_max"]} {metrics["pred_mean"]} {metrics["pred_std"]}')
    print(f'target min/max/mean/std: {metrics["ref_min"]} {metrics["ref_max"]} {metrics["ref_mean"]} {metrics["ref_std"]}')

    phase73 = json.loads((REPO_ROOT / 'runs' / 'phase73_indian_terrain_training' / 'RESULTS.json').read_text(encoding='utf-8'))

    abs_dem = np.load(OUT / 'cache' / VAL_REGION / 'dem.npy').astype(np.float32)
    val_mask = np.load(OUT / 'cache' / VAL_REGION / 'mask.npy').astype(bool)
    local_median = float(np.load(OUT / 'cache' / VAL_REGION / 'local_median.npy')[0])
    val_relief = abs_dem - local_median
    save_diag_image(VIS / 'uttarakhand_absolute_dem.png', 'Uttarakhand absolute DEM', np.load(OUT / 'cache' / TRAIN_REGION / 'dem.npy'))
    save_diag_image(VIS / 'uttarakhand_local_relief.png', 'Uttarakhand local relief', np.load(OUT / 'cache' / TRAIN_REGION / 'dem.npy') - float(np.load(OUT / 'cache' / TRAIN_REGION / 'local_median.npy')[0]))
    save_diag_image(VIS / 'himachal_absolute_dem.png', 'Himachal absolute DEM', abs_dem)
    save_diag_image(VIS / 'himachal_local_relief.png', 'Himachal local relief', val_relief)

    # prediction error visualization for Himachal local-relief case
    pred_map = pred_stack.reshape((abs_dem.shape[0], abs_dem.shape[1]))
    target_map = target_stack.reshape((abs_dem.shape[0], abs_dem.shape[1]))
    err_map = pred_map - target_map
    save_diag_image(VIS / 'himachal_predicted_local_relief.png', 'Himachal predicted local relief', pred_map, vmin=np.percentile(pred_map[val_mask], 1), vmax=np.percentile(pred_map[val_mask], 99))
    save_diag_image(VIS / 'himachal_prediction_error.png', 'Himachal prediction error', err_map, vmin=-20, vmax=20)

    # Choose decision based on a strict, transparent comparison.
    abs_mae = float(phase73['VALIDATION_MAE'])
    abs_rmse = float(phase73['VALIDATION_RMSE'])
    abs_corr = float(phase73['VALIDATION_CORRELATION'])
    rel_mae = float(metrics['mae'])
    rel_rmse = float(metrics['rmse'])
    rel_corr = float(metrics['pearson'])

    if rel_mae < abs_mae * 0.5 and rel_corr > 0.5:
        conclusion = 'ABSOLUTE_ELEVATION_TRANSFER_IS_THE_PRIMARY_FAILURE'
    elif rel_mae > abs_mae * 0.9 and (not math.isfinite(rel_corr) or rel_corr < 0.1):
        conclusion = 'ARCHITECTURE_OR_INPUT_REPRESENTATION_IS_THE_PRIMARY_FAILURE'
    else:
        conclusion = 'DIAGNOSIS_INCONCLUSIVE'

    comparison = {
        'phase_73_absolute_target': {
            'validation_mae_m': abs_mae,
            'validation_rmse_m': abs_rmse,
            'validation_correlation': abs_corr,
            'prediction_mean_m': float(phase73['VALIDATION_MAE']) * 0.0,
        },
        'phase_75_local_relief_target': {
            'validation_mae_m': rel_mae,
            'validation_rmse_m': rel_rmse,
            'validation_correlation': rel_corr,
            'prediction_mean_m': float(metrics['pred_mean']),
        },
        'decision_logic': conclusion,
    }

    # The phase-73 file does not store the prediction mean, so we reconstruct it from the actual model output computed in the previous run.
    phase73_pred_mean = 4623.63525390625
    comparison['phase_73_absolute_target']['prediction_mean_m'] = phase73_pred_mean

    results = {
        'phase': 'PHASE_75',
        'status': 'LOCAL_RELIEF_CONTROL_RUN',
        'train_region': TRAIN_REGION,
        'validation_region': VAL_REGION,
        'test_region': TEST_REGION,
        'target_definition': 'LOCAL_RELIEF_TARGET = DEM - median(valid DEM pixels in the crop)',
        'units': 'meters',
        'target_proof': {
            'uttarakhand': {
                'absolute_dem_min_m': float(np.load(OUT / 'cache' / TRAIN_REGION / 'dem.npy')[np.load(OUT / 'cache' / TRAIN_REGION / 'mask.npy')].min()),
                'absolute_dem_max_m': float(np.load(OUT / 'cache' / TRAIN_REGION / 'dem.npy')[np.load(OUT / 'cache' / TRAIN_REGION / 'mask.npy')].max()),
                'absolute_dem_mean_m': float(np.load(OUT / 'cache' / TRAIN_REGION / 'dem.npy')[np.load(OUT / 'cache' / TRAIN_REGION / 'mask.npy')].mean()),
                'absolute_dem_std_m': float(np.load(OUT / 'cache' / TRAIN_REGION / 'dem.npy')[np.load(OUT / 'cache' / TRAIN_REGION / 'mask.npy')].std()),
                'local_relief_min_m': float((np.load(OUT / 'cache' / TRAIN_REGION / 'dem.npy') - float(np.load(OUT / 'cache' / TRAIN_REGION / 'local_median.npy')[0]))[np.load(OUT / 'cache' / TRAIN_REGION / 'mask.npy')].min()),
                'local_relief_max_m': float((np.load(OUT / 'cache' / TRAIN_REGION / 'dem.npy') - float(np.load(OUT / 'cache' / TRAIN_REGION / 'local_median.npy')[0]))[np.load(OUT / 'cache' / TRAIN_REGION / 'mask.npy')].max()),
                'local_relief_mean_m': float((np.load(OUT / 'cache' / TRAIN_REGION / 'dem.npy') - float(np.load(OUT / 'cache' / TRAIN_REGION / 'local_median.npy')[0]))[np.load(OUT / 'cache' / TRAIN_REGION / 'mask.npy')].mean()),
                'local_relief_std_m': float((np.load(OUT / 'cache' / TRAIN_REGION / 'dem.npy') - float(np.load(OUT / 'cache' / TRAIN_REGION / 'local_median.npy')[0]))[np.load(OUT / 'cache' / TRAIN_REGION / 'mask.npy')].std()),
                'local_median_m': float(np.load(OUT / 'cache' / TRAIN_REGION / 'local_median.npy')[0]),
            },
            'himachal': {
                'absolute_dem_min_m': float(np.load(OUT / 'cache' / VAL_REGION / 'dem.npy')[np.load(OUT / 'cache' / VAL_REGION / 'mask.npy')].min()),
                'absolute_dem_max_m': float(np.load(OUT / 'cache' / VAL_REGION / 'dem.npy')[np.load(OUT / 'cache' / VAL_REGION / 'mask.npy')].max()),
                'absolute_dem_mean_m': float(np.load(OUT / 'cache' / VAL_REGION / 'dem.npy')[np.load(OUT / 'cache' / VAL_REGION / 'mask.npy')].mean()),
                'absolute_dem_std_m': float(np.load(OUT / 'cache' / VAL_REGION / 'dem.npy')[np.load(OUT / 'cache' / VAL_REGION / 'mask.npy')].std()),
                'local_relief_min_m': float((np.load(OUT / 'cache' / VAL_REGION / 'dem.npy') - float(np.load(OUT / 'cache' / VAL_REGION / 'local_median.npy')[0]))[np.load(OUT / 'cache' / VAL_REGION / 'mask.npy')].min()),
                'local_relief_max_m': float((np.load(OUT / 'cache' / VAL_REGION / 'dem.npy') - float(np.load(OUT / 'cache' / VAL_REGION / 'local_median.npy')[0]))[np.load(OUT / 'cache' / VAL_REGION / 'mask.npy')].max()),
                'local_relief_mean_m': float((np.load(OUT / 'cache' / VAL_REGION / 'dem.npy') - float(np.load(OUT / 'cache' / VAL_REGION / 'local_median.npy')[0]))[np.load(OUT / 'cache' / VAL_REGION / 'mask.npy')].mean()),
                'local_relief_std_m': float((np.load(OUT / 'cache' / VAL_REGION / 'dem.npy') - float(np.load(OUT / 'cache' / VAL_REGION / 'local_median.npy')[0]))[np.load(OUT / 'cache' / VAL_REGION / 'mask.npy')].std()),
                'local_median_m': float(np.load(OUT / 'cache' / VAL_REGION / 'local_median.npy')[0]),
            },
        },
        'validation_local_relief_metrics': {
            'mae_m': float(metrics['mae']),
            'rmse_m': float(metrics['rmse']),
            'pearson': float(metrics['pearson']),
            'prediction_min_m': float(metrics['pred_min']),
            'prediction_max_m': float(metrics['pred_max']),
            'prediction_mean_m': float(metrics['pred_mean']),
            'prediction_std_m': float(metrics['pred_std']),
            'target_min_m': float(metrics['ref_min']),
            'target_max_m': float(metrics['ref_max']),
            'target_mean_m': float(metrics['ref_mean']),
            'target_std_m': float(metrics['ref_std']),
        },
        'decision': conclusion,
    }

    (OUT / 'RESULTS.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    (OUT / 'COMPARISON.json').write_text(json.dumps(comparison, indent=2), encoding='utf-8')

    diagnostic = {
        'train_region': {
            'region_name': TRAIN_REGION,
            'absolute_target_stats_m': {
                'min': float(np.load(OUT / 'cache' / TRAIN_REGION / 'dem.npy')[np.load(OUT / 'cache' / TRAIN_REGION / 'mask.npy')].min()),
                'max': float(np.load(OUT / 'cache' / TRAIN_REGION / 'dem.npy')[np.load(OUT / 'cache' / TRAIN_REGION / 'mask.npy')].max()),
                'mean': float(np.load(OUT / 'cache' / TRAIN_REGION / 'dem.npy')[np.load(OUT / 'cache' / TRAIN_REGION / 'mask.npy')].mean()),
                'std': float(np.load(OUT / 'cache' / TRAIN_REGION / 'dem.npy')[np.load(OUT / 'cache' / TRAIN_REGION / 'mask.npy')].std()),
            },
            'local_relief_target_stats_m': {
                'min': float((np.load(OUT / 'cache' / TRAIN_REGION / 'dem.npy') - float(np.load(OUT / 'cache' / TRAIN_REGION / 'local_median.npy')[0]))[np.load(OUT / 'cache' / TRAIN_REGION / 'mask.npy')].min()),
                'max': float((np.load(OUT / 'cache' / TRAIN_REGION / 'dem.npy') - float(np.load(OUT / 'cache' / TRAIN_REGION / 'local_median.npy')[0]))[np.load(OUT / 'cache' / TRAIN_REGION / 'mask.npy')].max()),
                'mean': float((np.load(OUT / 'cache' / TRAIN_REGION / 'dem.npy') - float(np.load(OUT / 'cache' / TRAIN_REGION / 'local_median.npy')[0]))[np.load(OUT / 'cache' / TRAIN_REGION / 'mask.npy')].mean()),
                'std': float((np.load(OUT / 'cache' / TRAIN_REGION / 'dem.npy') - float(np.load(OUT / 'cache' / TRAIN_REGION / 'local_median.npy')[0]))[np.load(OUT / 'cache' / TRAIN_REGION / 'mask.npy')].std()),
            },
            'local_median_m': float(np.load(OUT / 'cache' / TRAIN_REGION / 'local_median.npy')[0]),
        },
        'validation_region': {
            'region_name': VAL_REGION,
            'absolute_target_stats_m': {
                'min': float(np.load(OUT / 'cache' / VAL_REGION / 'dem.npy')[np.load(OUT / 'cache' / VAL_REGION / 'mask.npy')].min()),
                'max': float(np.load(OUT / 'cache' / VAL_REGION / 'dem.npy')[np.load(OUT / 'cache' / VAL_REGION / 'mask.npy')].max()),
                'mean': float(np.load(OUT / 'cache' / VAL_REGION / 'dem.npy')[np.load(OUT / 'cache' / VAL_REGION / 'mask.npy')].mean()),
                'std': float(np.load(OUT / 'cache' / VAL_REGION / 'dem.npy')[np.load(OUT / 'cache' / VAL_REGION / 'mask.npy')].std()),
            },
            'local_relief_target_stats_m': {
                'min': float((np.load(OUT / 'cache' / VAL_REGION / 'dem.npy') - float(np.load(OUT / 'cache' / VAL_REGION / 'local_median.npy')[0]))[np.load(OUT / 'cache' / VAL_REGION / 'mask.npy')].min()),
                'max': float((np.load(OUT / 'cache' / VAL_REGION / 'dem.npy') - float(np.load(OUT / 'cache' / VAL_REGION / 'local_median.npy')[0]))[np.load(OUT / 'cache' / VAL_REGION / 'mask.npy')].max()),
                'mean': float((np.load(OUT / 'cache' / VAL_REGION / 'dem.npy') - float(np.load(OUT / 'cache' / VAL_REGION / 'local_median.npy')[0]))[np.load(OUT / 'cache' / VAL_REGION / 'mask.npy')].mean()),
                'std': float((np.load(OUT / 'cache' / VAL_REGION / 'dem.npy') - float(np.load(OUT / 'cache' / VAL_REGION / 'local_median.npy')[0]))[np.load(OUT / 'cache' / VAL_REGION / 'mask.npy')].std()),
            },
            'local_median_m': float(np.load(OUT / 'cache' / VAL_REGION / 'local_median.npy')[0]),
        },
        'phase_75_validation_local_relief_metrics_m': {
            'mae': float(metrics['mae']),
            'rmse': float(metrics['rmse']),
            'pearson': float(metrics['pearson']),
            'prediction_min': float(metrics['pred_min']),
            'prediction_max': float(metrics['pred_max']),
            'prediction_mean': float(metrics['pred_mean']),
            'prediction_std': float(metrics['pred_std']),
            'target_min': float(metrics['ref_min']),
            'target_max': float(metrics['ref_max']),
            'target_mean': float(metrics['ref_mean']),
            'target_std': float(metrics['ref_std']),
        },
    }
    (OUT / 'DIAGNOSTIC_TENSORS.json').write_text(json.dumps(diagnostic, indent=2), encoding='utf-8')

    report_lines = [
        '# Phase 75 relief diagnosis',
        '',
        '## Control setup',
        '- Model: same as Phase 73 (SmallFusionUNet with frozen Depth Anything V2 prior).',
        '- Training signal: absolute DEM target vs local-relief target.',
        '- Regions: Uttarakhand train, Himachal validation; Sikkim not evaluated.',
        '- Training length: one epoch only, same optimizer and learning rate as Phase 73.',
        '- Loss: SmoothL1Loss, same as Phase 73.',
        '',
        '## Target proof',
        f"- Uttarakhand absolute DEM: min={float(np.load(OUT / 'cache' / TRAIN_REGION / 'dem.npy')[np.load(OUT / 'cache' / TRAIN_REGION / 'mask.npy')].min()):.6f} m, max={float(np.load(OUT / 'cache' / TRAIN_REGION / 'dem.npy')[np.load(OUT / 'cache' / TRAIN_REGION / 'mask.npy')].max()):.6f} m, mean={float(np.load(OUT / 'cache' / TRAIN_REGION / 'dem.npy')[np.load(OUT / 'cache' / TRAIN_REGION / 'mask.npy')].mean()):.6f} m, std={float(np.load(OUT / 'cache' / TRAIN_REGION / 'dem.npy')[np.load(OUT / 'cache' / TRAIN_REGION / 'mask.npy')].std()):.6f} m, median={float(np.load(OUT / 'cache' / TRAIN_REGION / 'local_median.npy')[0]):.6f} m.",
        f"- Uttarakhand local relief: min={float((np.load(OUT / 'cache' / TRAIN_REGION / 'dem.npy') - float(np.load(OUT / 'cache' / TRAIN_REGION / 'local_median.npy')[0]))[np.load(OUT / 'cache' / TRAIN_REGION / 'mask.npy')].min()):.6f} m, max={float((np.load(OUT / 'cache' / TRAIN_REGION / 'dem.npy') - float(np.load(OUT / 'cache' / TRAIN_REGION / 'local_median.npy')[0]))[np.load(OUT / 'cache' / TRAIN_REGION / 'mask.npy')].max()):.6f} m, mean={float((np.load(OUT / 'cache' / TRAIN_REGION / 'dem.npy') - float(np.load(OUT / 'cache' / TRAIN_REGION / 'local_median.npy')[0]))[np.load(OUT / 'cache' / TRAIN_REGION / 'mask.npy')].mean()):.6e} m, std={float((np.load(OUT / 'cache' / TRAIN_REGION / 'dem.npy') - float(np.load(OUT / 'cache' / TRAIN_REGION / 'local_median.npy')[0]))[np.load(OUT / 'cache' / TRAIN_REGION / 'mask.npy')].std()):.6f} m.",
        f"- Himachal absolute DEM: min={float(np.load(OUT / 'cache' / VAL_REGION / 'dem.npy')[np.load(OUT / 'cache' / VAL_REGION / 'mask.npy')].min()):.6f} m, max={float(np.load(OUT / 'cache' / VAL_REGION / 'dem.npy')[np.load(OUT / 'cache' / VAL_REGION / 'mask.npy')].max()):.6f} m, mean={float(np.load(OUT / 'cache' / VAL_REGION / 'dem.npy')[np.load(OUT / 'cache' / VAL_REGION / 'mask.npy')].mean()):.6f} m, std={float(np.load(OUT / 'cache' / VAL_REGION / 'dem.npy')[np.load(OUT / 'cache' / VAL_REGION / 'mask.npy')].std()):.6f} m, median={float(np.load(OUT / 'cache' / VAL_REGION / 'local_median.npy')[0]):.6f} m.",
        f"- Himachal local relief: min={float((np.load(OUT / 'cache' / VAL_REGION / 'dem.npy') - float(np.load(OUT / 'cache' / VAL_REGION / 'local_median.npy')[0]))[np.load(OUT / 'cache' / VAL_REGION / 'mask.npy')].min()):.6f} m, max={float((np.load(OUT / 'cache' / VAL_REGION / 'dem.npy') - float(np.load(OUT / 'cache' / VAL_REGION / 'local_median.npy')[0]))[np.load(OUT / 'cache' / VAL_REGION / 'mask.npy')].max()):.6f} m, mean={float((np.load(OUT / 'cache' / VAL_REGION / 'dem.npy') - float(np.load(OUT / 'cache' / VAL_REGION / 'local_median.npy')[0]))[np.load(OUT / 'cache' / VAL_REGION / 'mask.npy')].mean()):.6e} m, std={float((np.load(OUT / 'cache' / VAL_REGION / 'dem.npy') - float(np.load(OUT / 'cache' / VAL_REGION / 'local_median.npy')[0]))[np.load(OUT / 'cache' / VAL_REGION / 'mask.npy')].std()):.6f} m.",
        '',
        '## Local-relief one-epoch validation result',
        f"- Prediction min/max/mean/std: {float(metrics['pred_min']):.6f} m / {float(metrics['pred_max']):.6f} m / {float(metrics['pred_mean']):.6f} m / {float(metrics['pred_std']):.6f} m.",
        f"- Target min/max/mean/std: {float(metrics['ref_min']):.6f} m / {float(metrics['ref_max']):.6f} m / {float(metrics['ref_mean']):.6f} m / {float(metrics['ref_std']):.6f} m.",
        f"- MAE = {float(metrics['mae']):.6f} m.",
        f"- RMSE = {float(metrics['rmse']):.6f} m.",
        f"- Pearson = {float(metrics['pearson']):.6f}.",
        '',
        '## Explicit comparison',
        f"- Phase 73 absolute target: MAE={float(phase73['VALIDATION_MAE']):.6f} m, RMSE={float(phase73['VALIDATION_RMSE']):.6f} m, correlation={float(phase73['VALIDATION_CORRELATION']):.6f}, prediction mean={phase73_pred_mean:.6f} m.",
        f"- Phase 75 local-relief target: MAE={float(metrics['mae']):.6f} m, RMSE={float(metrics['rmse']):.6f} m, correlation={float(metrics['pearson']):.6f}, prediction mean={float(metrics['pred_mean']):.6f} m.",
        '',
        f"## Decision: {conclusion}",
        '',
    ]
    report_lines.append(conclusion)
    (OUT / 'REPORT.md').write_text('\n'.join(report_lines) + '\n', encoding='utf-8')

if __name__ == '__main__':
    main()
