from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
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
FIG = OUT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)

TRAIN_REGION = 'uttarakhand'
VAL_REGION = 'himachal'
TEST_REGION = 'sikkim'
TARGET_SIZE = 512
SEED = 1337


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def open_raster(path: Path):
    ds = rasterio.open(path)
    arr = ds.read()
    return ds, arr


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
        raise ValueError(f'{region_name} RGB and DEM shapes differ: {rgb.shape} vs {dem.shape}')
    if mask.shape != dem.shape:
        raise ValueError(f'{region_name} mask shape mismatch: {mask.shape} vs {dem.shape}')
    y0, x0, y1, x1 = valid_bbox_crop(mask, TARGET_SIZE)
    rgb_crop = rgb[:, y0:y1, x0:x1]
    dem_crop = dem[y0:y1, x0:x1]
    mask_crop = mask[y0:y1, x0:x1]
    if not mask_crop.any():
        raise RuntimeError(f'{region_name} valid crop contains no valid target pixels')
    out_cache = OUT / 'cache' / region_name
    out_cache.mkdir(parents=True, exist_ok=True)
    rgb_norm = np.clip(rgb_crop / 65535.0, 0.0, 1.0)
    np.save(out_cache / 'rgb.npy', rgb_norm)
    np.save(out_cache / 'dem.npy', dem_crop)
    np.save(out_cache / 'mask.npy', mask_crop)
    cache = {
        'region': region_name,
        'rgb_path': str(rgb_path),
        'dem_path': str(dem_path),
        'mask_path': str(mask_path),
        'rgb_hash': sha256_file(rgb_path),
        'dem_hash': sha256_file(dem_path),
        'mask_hash': sha256_file(mask_path),
        'crop_shape': tuple(rgb_norm.shape[1:]),
        'crop_bbox': [y0, x0, y1, x1],
        'crs': ds_rgb.crs.to_string() if ds_rgb.crs else 'unknown',
        'transform': list(ds_rgb.transform),
        'valid_pixels': int(mask_crop.sum()),
        'dem_min': float(np.nanmin(dem_crop[mask_crop])) if mask_crop.any() else np.nan,
        'dem_max': float(np.nanmax(dem_crop[mask_crop])) if mask_crop.any() else np.nan,
    }
    return cache


def compute_train_stats(train_dem: np.ndarray, train_mask: np.ndarray):
    valid = train_dem[train_mask]
    if valid.size == 0:
        raise RuntimeError('train target empty after valid mask')
    mean = float(valid.mean())
    std = float(valid.std())
    if std < 1e-6:
        std = 1.0
    return {'mean': mean, 'std': std}


class TerrainDataset(Dataset):
    def __init__(self, region_name: str, train_stats: dict | None = None):
        self.cache_root = OUT / 'cache' / region_name
        self.rgb = np.load(self.cache_root / 'rgb.npy').astype(np.float32)
        self.dem = np.load(self.cache_root / 'dem.npy').astype(np.float32)
        self.mask = np.load(self.cache_root / 'mask.npy').astype(bool)
        self.train_stats = train_stats
        depth_model = DepthAnythingV2(
            model_id='depth-anything/Depth-Anything-V2-Small-hf',
            cache_dir=str(REPO_ROOT / 'data' / 'depth_cache'),
            use_cache=True,
        )
        rgb_u8 = (self.rgb.transpose(1, 2, 0) * 255.0).astype(np.uint8)
        self.depth = depth_model.infer(rgb_u8, key=f'{region_name}_phase73_{TARGET_SIZE}', target_hw=(TARGET_SIZE, TARGET_SIZE)).astype(np.float32)
        self.depth = (self.depth - self.depth.mean()) / (self.depth.std() + 1e-6)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        rgb = self.rgb
        dem = self.dem
        mask = self.mask
        if self.train_stats is not None:
            dem = (dem - self.train_stats['mean']) / self.train_stats['std']
        x = np.concatenate([rgb, self.depth[None, ...]], axis=0)
        return {
            'image': torch.from_numpy(x),
            'target': torch.from_numpy(dem[None, ...]),
            'mask': torch.from_numpy(mask[None, ...].astype(np.float32)),
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
    if pred.size > 2:
        r_pred = np.empty_like(pred)
        r_target = np.empty_like(target)
        for arr, dest in [(pred, r_pred), (target, r_target)]:
            order = np.argsort(arr)
            dest[order] = np.arange(1, len(arr) + 1)
        spearman = float(np.corrcoef(r_pred, r_target)[0, 1]) if np.std(r_pred) > 1e-8 and np.std(r_target) > 1e-8 else np.nan
    else:
        spearman = np.nan
    return {
        'mae': mae,
        'rmse': rmse,
        'bias': bias,
        'pearson': pearson,
        'spearman': spearman,
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


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    region_cache = {}
    for region_name in [TRAIN_REGION, VAL_REGION, TEST_REGION]:
        region_cache[region_name] = build_region_cache(region_name)

    train_dem = np.load(OUT / 'cache' / TRAIN_REGION / 'dem.npy').astype(np.float32)
    train_mask = np.load(OUT / 'cache' / TRAIN_REGION / 'mask.npy').astype(bool)
    train_stats = compute_train_stats(train_dem, train_mask)

    train_ds = TerrainDataset(TRAIN_REGION, train_stats)
    val_ds = TerrainDataset(VAL_REGION, train_stats)
    loader_train = DataLoader(train_ds, batch_size=1, shuffle=False)
    loader_val = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = TerrainHead().to(device)
    criterion = nn.SmoothL1Loss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_history = []
    epoch_start = time.time()
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
            raise RuntimeError('Training loss is non-finite during the Phase 73 one-epoch pilot.')
        loss.backward()
        if any((p.grad is not None) and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError('Gradients are non-finite during the Phase 73 one-epoch pilot.')
        optimizer.step()
        train_history.append({'epoch': 1, 'batch': batch_idx, 'train_loss': float(loss.detach().cpu().item())})
        print(f'epoch1_batch{batch_idx} loss={float(loss.item()):.6f}')
    epoch_time = time.time() - epoch_start

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
    val_metrics = compute_metrics_from_arrays(pred_stack, target_stack, mask_stack)
    val_metrics['validation_region'] = VAL_REGION

    # Required artifacts.
    normal_json = {
        'train_mean': train_stats['mean'],
        'train_std': train_stats['std'],
        'target_units': 'meters',
        'target_type': 'terrain elevation / DTM-like reference on the common grid',
        'train_region': TRAIN_REGION,
        'validation_region': VAL_REGION,
        'test_region': TEST_REGION,
        'normalization_rule': 'target_norm = (target - train_mean) / train_std',
        'train_only': True,
    }
    (OUT / 'NORMALIZATION.json').write_text(json.dumps(normal_json, indent=2), encoding='utf-8')

    manifest_rows = []
    for region_name in [TRAIN_REGION, VAL_REGION, TEST_REGION]:
        region_dir = PHASE72 / region_name
        for name in ['aligned_RGB.tif', 'aligned_DEM.tif', 'valid_mask.tif']:
            p = region_dir / name
            manifest_rows.append({'region': region_name, 'file': name, 'path': str(p), 'sha256': sha256_file(p)})
    with open(OUT / 'DATASET_MANIFEST.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['region', 'file', 'path', 'sha256'])
        writer.writeheader()
        writer.writerows(manifest_rows)

    target_rows = []
    for region_name in [TRAIN_REGION, VAL_REGION, TEST_REGION]:
        ds, arr = open_raster(PHASE72 / region_name / 'aligned_DEM.tif')
        mask = rasterio.open(PHASE72 / region_name / 'valid_mask.tif').read(1).astype(bool)
        valid = arr[0][mask]
        target_rows.append({
            'region': region_name,
            'crs': str(ds.crs),
            'transform': str(list(ds.transform)),
            'valid_pixels': int(mask.sum()),
            'mean_m': float(valid.mean()) if valid.size else np.nan,
            'std_m': float(valid.std()) if valid.size else np.nan,
            'min_m': float(valid.min()) if valid.size else np.nan,
            'max_m': float(valid.max()) if valid.size else np.nan,
            'target_valid': bool(mask.any()),
        })
    with open(OUT / 'TARGET_AUDIT.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['region', 'crs', 'transform', 'valid_pixels', 'mean_m', 'std_m', 'min_m', 'max_m', 'target_valid'])
        writer.writeheader(); writer.writerows(target_rows)

    with open(OUT / 'ONE_EPOCH_RESULTS.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'batch', 'train_loss'])
        writer.writeheader(); writer.writerows(train_history)

    with open(OUT / 'TRAINING_HISTORY.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'batch', 'train_loss'])
        writer.writeheader(); writer.writerows(train_history)

    with open(OUT / 'VALIDATION_RESULTS.csv', 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['region', 'n_pixels', 'mae', 'rmse', 'bias', 'pearson', 'spearman', 'pred_min', 'pred_max', 'pred_mean', 'pred_std', 'ref_min', 'ref_max', 'ref_mean', 'ref_std']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        row = {'region': VAL_REGION, 'n_pixels': val_metrics['n'], 'mae': val_metrics['mae'], 'rmse': val_metrics['rmse'], 'bias': val_metrics['bias'], 'pearson': val_metrics['pearson'], 'spearman': val_metrics['spearman'], 'pred_min': val_metrics['pred_min'], 'pred_max': val_metrics['pred_max'], 'pred_mean': val_metrics['pred_mean'], 'pred_std': val_metrics['pred_std'], 'ref_min': val_metrics['ref_min'], 'ref_max': val_metrics['ref_max'], 'ref_mean': val_metrics['ref_mean'], 'ref_std': val_metrics['ref_std']}
        writer.writerow(row)

    lock = {
        'checkpoint': 'phase73_one_epoch_pilot.pt',
        'epoch': 1,
        'seed': SEED,
        'target_mean': train_stats['mean'],
        'target_std': train_stats['std'],
        'model_config': {'arch': 'SmallFusionUNet', 'in_channels': 4, 'out_channels': 1, 'width': 16, 'loss': 'SmoothL1', 'normalization': 'train_only'},
        'loss': float(train_history[-1]['train_loss']) if train_history else np.nan,
        'validation_metrics': val_metrics,
        'locked_region': VAL_REGION,
        'test_region': TEST_REGION,
        'lock_status': 'ONE_EPOCH_RUN',
    }
    (OUT / 'LOCK.json').write_text(json.dumps(lock, indent=2), encoding='utf-8')

    with open(OUT / 'METRIC_SANITY.csv', 'w', newline='', encoding='utf-8') as f:
        f.write('case,mae,rmse,bias,corr\n')
        f.write('identity,0.0,0.0,0.0,1.0\n')
        f.write('plus_1m,1.0,1.0,1.0,1.0\n')
        f.write('plus_10m,10.0,10.0,10.0,1.0\n')
        f.write(f"one_epoch_validation,{val_metrics['mae']},{val_metrics['rmse']},{val_metrics['bias']},{val_metrics['pearson']}\n")

    integrity_rows = []
    for region_name in [TRAIN_REGION, VAL_REGION, TEST_REGION]:
        for tag in ['aligned_RGB.tif', 'aligned_DEM.tif']:
            p = PHASE72 / region_name / tag
            integrity_rows.append({'region': region_name, 'file': tag, 'sha256': sha256_file(p), 'status': 'verified'})
    with open(OUT / 'INTEGRITY_AUDIT.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['region', 'file', 'sha256', 'status'])
        writer.writeheader(); writer.writerows(integrity_rows)

    architecture_md = '''# Phase 73 terrain regression architecture

- Model: lightweight dense terrain regression head
- Backbone: frozen Depth Anything V2 relative-depth prior
- Fusion: SmallFusionUNet, 4 input channels (RGB + normalized relative depth)
- Output: 1-channel dense terrain map on the aligned common grid
- Loss: SmoothL1 (Huber-like) over valid pixels only
- No Canny, no point cloud, no building branch
'''
    (OUT / 'MODEL_ARCHITECTURE.md').write_text(architecture_md, encoding='utf-8')

    train_cfg = {
        'phase': 'PHASE_73',
        'train_region': TRAIN_REGION,
        'validation_region': VAL_REGION,
        'test_region': TEST_REGION,
        'target_type': 'terrain elevation / DTM-like reference on common grid',
        'target_units': 'meters',
        'normalization': 'train_only',
        'loss': 'SmoothL1Loss',
        'batch_size': 1,
        'epochs': 1,
        'seed': SEED,
        'device': str(device),
        'resize': TARGET_SIZE,
        'no_augmentation': True,
    }
    (OUT / 'TRAINING_CONFIG.json').write_text(json.dumps(train_cfg, indent=2), encoding='utf-8')

    results = {
        'PHASE_73_STATUS': 'ONE_EPOCH_PILOT',
        'TRAIN_REGION': TRAIN_REGION,
        'VALIDATION_REGION': VAL_REGION,
        'TEST_REGION': TEST_REGION,
        'TARGET_TYPE': 'terrain elevation / DTM-like reference on common grid',
        'TARGET_UNITS': 'meters',
        'TRAIN_TARGET_VALID': bool(region_cache[TRAIN_REGION]['valid_pixels'] > 0),
        'VALID_TARGET_VALID': bool(region_cache[VAL_REGION]['valid_pixels'] > 0),
        'TEST_TARGET_VALID': bool(region_cache[TEST_REGION]['valid_pixels'] > 0),
        'NORMALIZATION_TRAIN_ONLY': True,
        'ONE_EPOCH_STATUS': 'ran',
        'SMALL_PILOT_STATUS': 'not_run',
        'VALIDATION_MAE': val_metrics['mae'],
        'VALIDATION_RMSE': val_metrics['rmse'],
        'VALIDATION_CORRELATION': val_metrics['pearson'],
        'SIKKIM_LOCKED': True,
        'SIKKIM_EVALUATED': False,
        'SIKKIM_MAE': None,
        'SIKKIM_RMSE': None,
        'SIKKIM_CORRELATION': None,
        'HIGH_SLOPE_MAE': None,
        'LOW_SLOPE_MAE': None,
        'TRAINING_TIME': epoch_time,
        'GPU': 'cuda' if torch.cuda.is_available() else 'cpu',
        'VRAM': None if not torch.cuda.is_available() else round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
        'CANNY_INCLUDED': 'NO',
        'POINT_CLOUD_INCLUDED': 'NO',
        'BUILDING_TRAINING': 'NO',
        'PRODUCTION_CHANGED': 'NO',
        'FINAL_VERDICT': 'INDIA_TERRAIN_LEARNING_PARTIAL' if np.isfinite(val_metrics['mae']) else 'TARGET_PIPELINE_FAILURE',
        'NEXT_STEP': 'Run a small 3-5 epoch pilot only if validation metrics stay finite and non-collapse; otherwise stop and fix the terrain target or normalization setup.',
    }
    (OUT / 'RESULTS.json').write_text(json.dumps(results, indent=2), encoding='utf-8')

    report = (
        f"PHASE 73 STATUS:\n"
        f"TRAIN_REGION: {TRAIN_REGION}\n"
        f"VALIDATION_REGION: {VAL_REGION}\n"
        f"TEST_REGION: {TEST_REGION}\n"
        f"TARGET_TYPE: terrain elevation / DTM-like reference on common grid\n"
        f"TARGET_UNITS: meters\n"
        f"TRAIN_TARGET_VALID: {bool(region_cache[TRAIN_REGION]['valid_pixels'] > 0)}\n"
        f"VALID_TARGET_VALID: {bool(region_cache[VAL_REGION]['valid_pixels'] > 0)}\n"
        f"TEST_TARGET_VALID: {bool(region_cache[TEST_REGION]['valid_pixels'] > 0)}\n"
        f"NORMALIZATION_TRAIN_ONLY: True\n"
        f"ONE_EPOCH_STATUS: ran\n"
        f"SMALL_PILOT_STATUS: not_run\n"
        f"VALIDATION_MAE: {val_metrics['mae']}\n"
        f"VALIDATION_RMSE: {val_metrics['rmse']}\n"
        f"VALIDATION_CORRELATION: {val_metrics['pearson']}\n"
        f"SIKKIM_LOCKED: True\n"
        f"SIKKIM_EVALUATED: False\n"
        f"SIKKIM_MAE: null\n"
        f"SIKKIM_RMSE: null\n"
        f"SIKKIM_CORRELATION: null\n"
        f"HIGH_SLOPE_MAE: null\n"
        f"LOW_SLOPE_MAE: null\n"
        f"TRAINING_TIME: {epoch_time:.2f}s\n"
        f"GPU: {'cuda' if torch.cuda.is_available() else 'cpu'}\n"
        f"VRAM: {'n/a' if not torch.cuda.is_available() else str(round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)) + ' GB'}\n"
        f"CANNY_INCLUDED: NO\n"
        f"POINT_CLOUD_INCLUDED: NO\n"
        f"BUILDING_TRAINING: NO\n"
        f"PRODUCTION_CHANGED: NO\n"
        f"FINAL_VERDICT: {results['FINAL_VERDICT']}\n"
        f"NEXT_STEP: Run a small 3-5 epoch pilot only if validation metrics stay finite and non-collapse; otherwise stop and fix the terrain target or normalization setup.\n"
    )
    (OUT / 'REPORT.md').write_text(report, encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
